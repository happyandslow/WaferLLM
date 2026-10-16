from typing import Dict, List

from sqlalchemy import create_engine, text

# Define SQLAlchemy engine connection string.
PROD_DB_HOST = 'cbda-postgres.cerebras.aws'
DB_NAME = 'cbdapg'
USERNAME = 'cerebras'
PASSWORD = 'cheetahDB100'
PG_CONNECTION = f'postgresql://{USERNAME}:{PASSWORD}@{PROD_DB_HOST}/{DB_NAME}'


def query_glitches(
    host_names: List[str], start_time: int, end_time: int
) -> Dict[str, List]:
    """
    Query the system telemetry DB for glitches.
    Note that since the glitch data is aggregated per hour, this can't be done
    at stall artifact collection time.
    The DB queries are adapted from the Grafana dashboard, e.g.
    http://cbda-grafana.cerebras.aws/d/a8e13673-463f-485f-b5c5-4ad13743f48c/system-glitch-monitor?orgId=1&var-host_name=xs10091&var-host_name_GLTCH=xs10091&var-threshold=0&from=1732356000000&to=1732366800000
    Click the three dots menu at the top right of the info pane and Explore/Inspect.
    host_name: system name
    start_time: start of time period in seconds since the epoch
    end_time: end of time period in seconds since the epoch
    Returns a Dict mapping system name to the glitch data (if any).
    """
    with create_engine(PG_CONNECTION).connect() as conn:
        # Dict mapping system name to glitch info
        results = {}
        for host_name in host_names:
            result = conn.execute(
                text(
                    f"""SELECT DISTINCT on (host_name, sensor_name)
      host_name,
      time,
      max,
      sensor_name
    FROM system_telemetry.telemetry_10s
    WHERE host_name IN ('{host_name}')
      AND sensor_name LIKE 'GLTCH_EXCUR_%_NEG_%'
      AND (
        TO_TIMESTAMP({end_time}) - TO_TIMESTAMP({start_time}) < '5 mins'::interval
        OR TO_TIMESTAMP({start_time}) >= now() - '5 mins'::interval
      )
      AND time >= TO_TIMESTAMP({start_time})
      AND time <= TO_TIMESTAMP({end_time})
      AND max > cast(0 as INTEGER)
    UNION ALL
    SELECT DISTINCT on (host_name, sensor_name)
      host_name,
      time,
      max,
      sensor_name
    FROM system_telemetry.telemetry_1m
    WHERE host_name IN ('{host_name}')
      AND sensor_name LIKE 'GLTCH_EXCUR_%_NEG_%'
      AND (
        TO_TIMESTAMP({end_time}) - TO_TIMESTAMP({start_time}) BETWEEN '5 min'::interval AND '1 hour'::interval
        OR TO_TIMESTAMP({start_time}) BETWEEN now() - '5 mins'::interval AND now() - '2 hours'::interval
      )
      AND time >= TO_TIMESTAMP({start_time})
      AND time <= TO_TIMESTAMP({end_time})
      AND max > cast(0 as INTEGER)
    UNION ALL
    SELECT DISTINCT on (host_name, sensor_name)
      host_name,
      time,
      max,
      sensor_name
    FROM system_telemetry.telemetry_1h
    WHERE host_name IN ('{host_name}')
      AND sensor_name LIKE 'GLTCH_EXCUR_%_NEG_%'
      AND (
        TO_TIMESTAMP({end_time}) - TO_TIMESTAMP({start_time}) > '2 hours'::interval
        OR TO_TIMESTAMP(({start_time})) >= now() - '2 hours'::interval
      )
      AND time >= TO_TIMESTAMP({start_time})
      AND time <= TO_TIMESTAMP({end_time})
      AND max > cast(0 as INTEGER)
    ORDER BY host_name, sensor_name, time"""
                )
            )
            ret = result.all()
            if ret:
                results[host_name] = ret
        return results


# Sample query with glitches
# tstart = time.time()
# ret = query_glitches([f"xs{sys}" for sys in range(10020, 10084)], 1732356000, 1732366800)
# tend = time.time()
# print(ret)
# print(f"query took {tend - tstart}")
