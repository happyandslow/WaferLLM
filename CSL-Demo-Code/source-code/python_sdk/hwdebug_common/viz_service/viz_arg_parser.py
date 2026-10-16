import argparse


def parse_arguments(suppress_path=False):
    parser = argparse.ArgumentParser(
        description="""
This is a visualization tool to interact with compile results, execution results, and debug tools.
The tool will scan the path specified (including subdirectories) for compile and execution results.
Then, it will present a web server URL
that the user can right click on to
analyze and display results.


Generally, the user should cd to the work_dir,
and type 'csviz .' to scan files in the current directory.
"""
    )

    parser.add_argument(
        "--host",
        help="address to bind to, default is INADDR_ANY or 0.0.0.0",
        default='0.0.0.0',
    )

    parser.add_argument(
        "--webdev", help="act as proxy for web pages",
    )

    parser.add_argument(
        "--idle_timeout",
        help="timeout to stop server when idle.  Defaults to 10 minutes",
        default=10 * 60,
    )

    parser.add_argument(
        "--proxy", help="act as proxy for API", action='store_true'
    )

    parser.add_argument(
        "--port",
        help="port to bind, default is to find a port in range 8000-8999",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--no_session",
        help="don't use session id for security",
        action='store_true',
    )

    parser.add_argument(
        "--enforce",
        help="don't enforce correct session id for security",
        action='store_true',
    )

    parser.add_argument(
        "--noretry",
        help="do not retry binding to port, just increment port and try again",
        action='store_true',
    )

    parser.add_argument(
        "--debug", help="print log messages", action='store_true'
    )

    parser.add_argument("--filename", help="log filename", type=str)

    parser.add_argument(
        "--arch", help="target architecture", type=str, default="FYN"
    )

    # debugviz does not require path
    if not suppress_path:
        parser.add_argument(
            "path",
            help="directory you would like to scan for execution results.  This is generally a work directory containing compile results and execution results.",
        )

    return parser
