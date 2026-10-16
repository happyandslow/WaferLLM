#!/usr/bin/env python3

"""
Routines for querying the license server (using lmstat)

"""
import logging
import os
import re
import shutil
import subprocess
import time
import datetime
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import timedelta
from dateutil.parser import parse
from dateutil.tz import tzlocal

LMSTAT_DB = {}
LMSTAT_DB_HAS_ALL = False
LMSTAT_DB_LICENSE_FILE = ""
LMSTAT_DB_DATE = None

def clear_db():
    """Clear global database
    """
    global LMSTAT_DB
    global LMSTAT_DB_HAS_ALL
    global LMSTAT_DB_LICENSE_FILE
    global LMSTAT_DB_DATE
    LMSTAT_DB = {}
    LMSTAT_DB_HAS_ALL = False
    LMSTAT_DB_LICENSE_FILE = ""
    LMSTAT_DB_DATE = None

def run_command(cmd):
    """Run command

    Returns output of command.
    """
    try:
        out = subprocess.check_output(cmd, universal_newlines=True)
    except subprocess.CalledProcessError as ex:
        logging.error(' '.join(cmd) + ": " + ex.output)
    return out

def exec_lmstat(features, license_file=""):
    """Execute lmstat

    features is a list of features.  An empty list implies we want
    to run lmstat for all features.

    Returns a list of outputs from the lmstat commands (per feature)
    """
    global LMSTAT_DB_HAS_ALL
    global LMSTAT_DB_DATE
    cmds = []
    outputs = []
    lmstat_bin = "lmstat"
    if shutil.which(lmstat_bin) is None:
        lmstat_bin = "/tools/synopsys/scl/2018.06/linux64/bin/lmstat"

    if not features:
        cmd = [lmstat_bin, "-a"]
        if license_file and license_file != "":
            cmd += ["-c", license_file]
        cmds.append(cmd)
        LMSTAT_DB_HAS_ALL = True
    for feature in features:
        cmd = [lmstat_bin, "-f", feature]
        if license_file and license_file != "":
            cmd += ["-c", license_file]
        cmds.append(cmd)

    with ProcessPoolExecutor(max_workers=4) as executor:
        outputs = list(executor.map(run_command, cmds))

    LMSTAT_DB_DATE = datetime.datetime.now()
    return outputs

def parse_lmstat(features, outputs):
    """Parse lmstat

    features is a list of features.  An empty list implies we want
    to run lmstat for all features.

    outputs is a list of lmstat outputs

    Parses the outputs and populates the LMSTAT_DB global database.
    """
    if features:
        # clear out the count for the feature subset
        for feature in features:
            LMSTAT_DB[feature]["used"]   = 0
            LMSTAT_DB[feature]["avail"]  = 0
            LMSTAT_DB[feature]["issued"] = 0
    else:
        # clear out the count for all features
        for feature in LMSTAT_DB:
            LMSTAT_DB[feature]["used"]   = 0
            LMSTAT_DB[feature]["avail"]  = 0
            LMSTAT_DB[feature]["issued"] = 0

    for out in outputs:
        feature = ""
        for line in out.split('\n'):
            match = re.match(r'Users of ([a-zA-Z0-9_-]+):\s*\(Total of (\d+) licenses? issued;'
                             r'\s+Total of (\d+) licenses? in use', line)
            if match:
                feature = match.group(1)
                issued = int(match.group(2))
                used = int(match.group(3))
                avail = issued - used
                if feature not in LMSTAT_DB.keys():
                    LMSTAT_DB[feature] = { "used"   : 0,
                                           "avail"  : 0,
                                           "issued" : 0,
                                           "users"  : [] }
                LMSTAT_DB[feature]["used"]   += used
                LMSTAT_DB[feature]["avail"]  += avail
                LMSTAT_DB[feature]["issued"] += issued
                continue

            match = re.match(r'\s+(\S+)\s+(\S+)\s+(.*)'
                             r'\((.*)\)\s+\((\S+) (\d+)\),\s+start\s+(.*)', line)

            if match:
                LMSTAT_DB[feature]["users"].append({
                            "user" : match.group(1),
                            "host" : match.group(2),
                            "display" : match.group(3),
                            "version" : match.group(4),
                            "server" : match.group(5),
                            "handle" : match.group(6),
                            "start" : match.group(7)
                           })

def get_lmstat(features=None, license_file=""):
    """Return license usage info

    Returns a dictionary where the keys are the license features, and the values
    are dicts representing (used, avail, issued, users)
    """
    global LMSTAT_DB_LICENSE_FILE
    if features is None:
        features = []

    # convert single string into a one element list
    if isinstance(features, str):
        features = [features]

    if license_file != "" and license_file != LMSTAT_DB_LICENSE_FILE:
        # different license file than before, clear the DB
        clear_db()
        LMSTAT_DB_LICENSE_FILE = license_file
    if LMSTAT_DB_DATE:
        current_time = datetime.datetime.now()
        if (current_time - LMSTAT_DB_DATE).total_seconds() > 30:
            clear_db()

    # user requested one or more features, return it if we already have it
    if features and set(features).issubset(LMSTAT_DB.keys()):
        return {f: deepcopy(LMSTAT_DB[f]) for f in features if f in LMSTAT_DB}

    # user requested all features, return it if we already have it
    if not features and LMSTAT_DB_HAS_ALL:
        return deepcopy(LMSTAT_DB)

    for feature in features:
        if feature not in LMSTAT_DB:
            LMSTAT_DB[feature] = { "used"   : 0,
                                   "avail"  : 0,
                                   "issued" : 0,
                                   "users"  : [] }

    lmstat_out = exec_lmstat(features, license_file)
    parse_lmstat(features, lmstat_out)

    if features:
        return {f: deepcopy(LMSTAT_DB[f]) for f in features if f in LMSTAT_DB}
    else:
        return deepcopy(LMSTAT_DB)

def get_lmstat_users(feature):
    """Return license users

    Returns a list of dicts.
    """

    lmstat = get_lmstat(feature)

    if feature in lmstat:
        return deepcopy(lmstat[feature]["users"])
    return {}

def parse_lmgrd_log(server="lic0",
                    filename="lmgrd.log"):
    """Parse an lmgrd log

    Try to print graphs of usage
    """
    usage = {}
    prev_datetime = None
    curr_datetime = None

    with open(filename, 'r') as f:
        for line in f:
            # match
            # 16:10:10 (snpslmd) IN: "PrimeTime" vivian@pd1 [J3_1fac5e2f_31662_1]
            #  4:35:32 (lmgrd) TIMESTAMP 10/15/2018
            try:
                logtime, daemon, line = line.split(maxsplit=2)
            except ValueError:
                continue

            if not re.match(r'\(.*\)', daemon):
                continue

            if curr_datetime is None:
                match = re.match(r'.*Server\'s System Date and Time: (.*)', line)
                if match:
                    curr_datetime = parse(match.group(1))
                    print("Start on:", curr_datetime)
            else:
                match = re.match(r'TIMESTAMP\s+(\S*)', line)
                if match:
                    prev_datetime = curr_datetime
                    curr_datetime = parse(match.group(1) + " " + logtime)
                    curr_datetime = curr_datetime.replace(tzinfo=tzlocal())
                else:
                    print(logtime)
                    prev_datetime = curr_datetime
                    curr_datetime = parse(str(curr_datetime.date()) + " " + logtime)
                    curr_datetime = curr_datetime.replace(tzinfo=tzlocal())
                    if curr_datetime < prev_datetime:
                        curr_datetime = curr_datetime + timedelta(days=1)

            match = re.match(r'(IN|OUT):\s+'
                             r'"(\S+)"\s+(\S+)\s+(\S*)', line)
            if match:
                assert not curr_datetime is None
                inout = match.group(1)
                feature = match.group(2)
                # user_host = match.group(3)
                if server not in usage:
                    usage[server] = {}
                if daemon not in usage[server]:
                    usage[server][daemon] = {}
                if feature not in usage[server][daemon]:
                    usage[server][daemon][feature] = {}
                    usage[server][daemon][feature]["current"] = 0
                    usage[server][daemon][feature]["watermark"] = 0
                    usage[server][daemon][feature]["total_out"] = 0
                    usage[server][daemon][feature]["total_in"] = 0
                    # for graphing
                    usage[server][daemon][feature]["used"] = []
                    usage[server][daemon][feature]["times"] = []

                # record
                if curr_datetime > curr_datetime.now(tzlocal()) - timedelta(days=30):
                    usage[server][daemon][feature]["times"].append(curr_datetime)
                    usage[server][daemon][feature]["used"].append(
                        usage[server][daemon][feature]["current"])

                if inout == "OUT":
                    usage[server][daemon][feature]["current"] += 1
                    usage[server][daemon][feature]["total_out"] += 1
                if inout == "IN":
                    usage[server][daemon][feature]["current"] -= 1
                    usage[server][daemon][feature]["total_in"] += 1
                assert usage[server][daemon][feature]["current"] >= 0
                if (usage[server][daemon][feature]["current"] >
                    usage[server][daemon][feature]["watermark"]):
                    usage[server][daemon][feature]["watermark"] = \
                        usage[server][daemon][feature]["current"]

                # record
                if curr_datetime > curr_datetime.now(tzlocal()) - timedelta(days=30):
                    usage[server][daemon][feature]["times"].append(curr_datetime)
                    usage[server][daemon][feature]["used"].append(
                        usage[server][daemon][feature]["current"])
            elif " IN:" in line or " OUT:" in line:
                assert False, " Failed to parse:" + line

    # Now that we have collected usage, make some plots
    import matplotlib.pyplot as plt
    for daemon in sorted(usage[server].keys()):
        for feature in sorted(usage[server][daemon].keys()):
            print(feature)
            print(usage[server][daemon][feature]["watermark"])
            print(usage[server][daemon][feature]["total_out"])
            print(usage[server][daemon][feature]["total_in"])

            fig, _ = plt.subplots()
            plt.title(server + " " + feature)
            plt.plot(
                usage[server][daemon][feature]["times"],
                usage[server][daemon][feature]["used"])
            fig.autofmt_xdate()
            plt.savefig(".".join([server, daemon, feature, "png"]))

            plt.clf()
            plt.cla()
            plt.close()


def lmstat_to_output(lmstat, current_time_ns, data_format):
    """lmstat output to specific data format
    """
    if data_format == "json":
        json = []
        for feature in lmstat.keys():
            feature_info = lmstat[feature]
            json.append(
                {
                 "measurement": "lmstat",
                 "tags": {
                     "feature" : feature,
                 },
                 "fields": {
                     "used"   : feature_info["used"],
                     "avail"  : feature_info["avail"],
                     "issued" : feature_info["issued"],
                 },
                 "time" : current_time_ns
                }
            )
            for user_info in feature_info["users"]:
                json.append(
                    {
                     "measurement": "lmstat",
                     "tags": {
                         "feature" : feature,
                         "user"    : user_info["user"],
                         "from"    : user_info["host"],
                         # server field has spaces, but is also redundant
                         # in our monitoring scheme, because each server
                         # should only report its own licenses.
                         #"server"  : user_info["server"],
                         "handle"  : user_info["handle"],
                     },
                     "fields": {
                         "count"   : 1,
                     },
                     "time" : current_time_ns
                    }
                )
        return json
    if data_format == "line":
        lines = []
        for feature in lmstat.keys():
            feature_info = lmstat[feature]
            lines.append("lmstat,feature={} used={},avail={},issued={} {}"
                         .format(feature,
                                 feature_info["used"],
                                 feature_info["avail"],
                                 feature_info["issued"],
                                 current_time_ns))
            for user_info in feature_info["users"]:
                lines.append("lmstat,feature={},user={},from={},handle={} "
                             "count=1 {}"
                             .format(feature,
                                     user_info["user"],
                                     user_info["host"],
                                     user_info["handle"],
                                     current_time_ns))
        return lines
    assert False, "Unknown format: {}".format(data_format)
    return None

def push_influx(interval=5):
    """Pushes data continuously into an influx database

    This function will not return.  Database setup is hardcoded below.

    Args:
      interval: polling interval in seconds.  This interval is approximate since
      it just feeds into a sleep() call.
    """
    from influxdb import InfluxDBClient # pylint: disable=import-error
    client = InfluxDBClient(host='it2',
                            port=8086,
                            username='root',
                            password='root',
                            ssl=False, verify_ssl=False)
    client.create_database('lmstat')
    client.switch_database('lmstat')
    while True:
        lmstat = get_lmstat()
        current_time = int(time.time())
        current_time_ns = current_time*(10**9)
        json = lmstat_to_output(lmstat, current_time_ns, "json")
        client.write_points(json, time_precision='s')
        time.sleep(interval)

def time_to_required_free(feature, issued, curr_time):
    """ Return number of required free licenses based on the time.

    Have a set of required free licenses during working hours (9-5pm M-F)
    """
    # print("hour", curr_time.hour)
    # print("day", curr_time.weekday())
    required_free = 0
    # weekday 0 => Mon
    # weekday 1 => Tue
    # weekday 2 => Wed
    # weekday 3 => Thu
    # weekday 4 => Fri
    # weekday 5 => Sat
    # weekday 6 => Sun

    if feature != 'VCSRuntime_Net':
        return 0

    # On weekdays from 9am-9pm, require 1/8 of the total issues licenses
    # to be free.
    if (curr_time.weekday() <= 4 and
        curr_time.hour >= 9 and curr_time.hour < (12+9)):
        required_free = issued//8
    return required_free

def lic_limit(lminfo, feature, user, apply_limit=True, sge_limit=False, slurm_limit=False):
    """Return max limit of licenses for the feature and license

    This is currently meant for VCS, where the build user's license usage
    should be limited during the day based on a target number of available
    licenses.

    If apply_limit is set, then apply the license limit.  This is meant to be
    executed on sge master nodes by a user with admin priviledges on sge.
    """
    issued = lminfo[feature]["issued"]
    now = datetime.datetime.now()
    required_free = time_to_required_free(feature, issued, now)
    #print("user/avail/issued: %d/%d/%d" %
    #      (lminfo[feature]["used"],
    #       lminfo[feature]["avail"],
    #       lminfo[feature]["issued"]))

    # Testing
    # if True:
    #     lminfo[feature]["used"] = 25
    #     lminfo[feature]["avail"] = 7
    #     lminfo[feature]["issued"] = 32

    build_min = issued//8
    build_limit = issued - required_free
    build_used = sum(1 for lmuser in lminfo[feature]["users"]
                     if lmuser['user'] == user)
    if required_free > lminfo[feature]["avail"]:
        # not enough free licenses, reduce build usage by required_free/2
        build_limit = build_used - (required_free - lminfo[feature]["avail"])
        build_limit = max(build_limit, build_min)
    if apply_limit:
        logging.info("date: %s" % now)
        logging.info("required_free: %2d" % required_free)
        logging.info("current_free:  %2d" % lminfo[feature]["avail"])
        logging.info("build_used:    %2d" % build_used)
        logging.info("build_min:     %2d" % build_min)
        logging.info("build_limit:   %2d" % build_limit)

    # To check limit:
    #     /opt/sge/bin/lx-amd64/qconf -sc | grep vcs
    #     /opt/sge/bin/lx-amd64/qconf -se global | grep complex_values
    #     /opt/slurm/bin/sacctmgr show resource

    if apply_limit:
        if sge_limit:
            os.environ['SGE_ROOT'] = '/opt/sge'
            cmd = ['/opt/sge/bin/lx-amd64/qconf',
                   '-mattr', 'exechost', 'complex_values', 'vcs=%d' % issued, 'global']
            try:
                subprocess.check_output(" ".join(cmd),
                                        shell=True, universal_newlines=True, env=os.environ)
            except subprocess.CalledProcessError as ex:
                logging.error(' '.join(cmd) + ": " + ex.output)

            cmd = ['/opt/sge/bin/lx-amd64/qconf',
                   '-mattr', 'exechost', 'complex_values', 'vcsbld=%d' % build_limit, 'global']
            try:
                subprocess.check_output(" ".join(cmd),
                                        shell=True, universal_newlines=True, env=os.environ)
            except subprocess.CalledProcessError as ex:
                logging.error(' '.join(cmd) + ": " + ex.output)
        if slurm_limit:
            cmd = ['/opt/slurm/bin/sacctmgr',
                   '-i', 'modify', 'resource', 'name=vcs', 'set', 'count=%d' % issued]
            try:
                subprocess.check_output(" ".join(cmd),
                                        shell=True, universal_newlines=True, env=os.environ)
            except subprocess.CalledProcessError as ex:
                logging.error(' '.join(cmd) + ": " + ex.output)

            cmd = ['/opt/slurm/bin/sacctmgr',
                   '-i', 'modify', 'resource', 'name=vcsbld', 'set', 'count=%d' % build_limit]
            try:
                subprocess.check_output(" ".join(cmd),
                                        shell=True, universal_newlines=True, env=os.environ)
            except subprocess.CalledProcessError as ex:
                logging.error(' '.join(cmd) + ": " + ex.output)

    return build_limit


def main():
    """example code"""
    import argparse
    parser = argparse.ArgumentParser(description='lmstat.py - licence status',
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     epilog='''\
OVERVIEW
    Print out licence info

EXAMPLES

    > lmstat.py -c 27020@server0 --json
    > lmstat.py -c 27020@lic0 --line
    > lmstat.py -c 27020@lic0 --sge_limit # (only on an sge master)
    > lmstat.py -c 27020@lic0 --slurm_limit # (only on a slurm master)
''')
    parser.add_argument("-c",
                        action='store',
                        dest='license_file',
                        default=False,
                        help='license file')
    parser.add_argument("--json",
                        action='store_true',
                        dest='json',
                        default=False,
                        help='dump json')
    parser.add_argument("--line",
                        action='store_true',
                        dest='line',
                        default=False,
                        help='dump influx line format')
    parser.add_argument("--sge_limit",
                        action='store_true',
                        dest='sge_limit',
                        default=False,
                        help='apply limit license usage for sge')
    parser.add_argument("--slurm_limit",
                        action='store_true',
                        dest='slurm_limit',
                        default=False,
                        help='apply limit license usage for slurm')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.line:
        lmstat = get_lmstat(license_file=args.license_file)
        current_time = int(time.time())
        current_time_ns = current_time*(10**9)
        print("\n".join(lmstat_to_output(lmstat, current_time_ns, "line")))
        feature = "VCSRuntime_Net"
        if feature in lmstat:
            user = "build"
            print("lmstat,feature={},user={} limit={} {}".format(
                  feature, user,
                  lic_limit(lmstat, feature, user, apply_limit=False),
                  current_time_ns
                  ))
        return
    if args.json:
        assert not args.sge_limit, "Do not support --sge_limit with --json"
        assert not args.slurm_limit, "Do not support --slurm_limit with --json"
        import json
        lmstat = get_lmstat(license_file=args.license_file)
        current_time = int(time.time())
        current_time_ns = current_time*(10**9)
        print(json.dumps(lmstat_to_output(lmstat, current_time_ns, "json")))
        return
    if args.sge_limit or args.slurm_limit:
        feature = "VCSRuntime_Net"
        user = "build"
        lmstat = get_lmstat(license_file=args.license_file)
        lic_limit(lmstat, feature, user,
                  apply_limit=True, sge_limit=args.sge_limit, slurm_limit=args.slurm_limit)
        return

    from devinfra.common import cbtable
    all_features = get_lmstat().keys()
    lminfo = get_lmstat()
    print("All license info:")
    table = cbtable.CBTable(["Feature", "Used", "Issued"])
    for feature in sorted(lminfo.keys()):
        table.add_row((feature,
                       lminfo[feature]["used"],
                       lminfo[feature]["issued"]))
    print(table.render())

    lminfo = get_lmstat("VCSRuntime_Net")
    print("License info for VCS:")
    table = cbtable.CBTable(["Feature", "Used", "Issued"])
    lminfo = get_lmstat("VCSRuntime_Net")
    for feature in sorted(lminfo.keys()):
        table.add_row((feature,
                       lminfo[feature]["used"],
                       lminfo[feature]["issued"]))
    print(table.render())

    lminfo = get_lmstat_users("VCSRuntime_Net")
    print("Users of VCS:")
    table = cbtable.CBTable(["User", "Host", "Started"])
    for userinfo in lminfo:
        table.add_row((userinfo["user"], userinfo["host"], userinfo["start"]))
    print(table.render())

    for feature in all_features:
        lminfo = get_lmstat_users(feature)
        if lminfo:
            print("Users of %s:" % feature)
            table = cbtable.CBTable(["User", "Host", "Display", "Version", "Started"])
            for userinfo in lminfo:
                table.add_row((userinfo["user"],
                               userinfo["host"],
                               userinfo["display"],
                               userinfo["version"],
                               userinfo["start"]))
            print(table.render())


if __name__ == "__main__":
    main()
