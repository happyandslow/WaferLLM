#!/usr/bin/env python3
"""
SdkLauncher script for Decode WSE-3-KV module.

Dispatches a pre-compiled artifact to the appliance via SdkLauncher,
stages the host execution script (launch_sim.py) and config file,
then runs the host code on the appliance.

Prerequisites:
    python compile.py <args>   # produces compile_out/artifact_{P}_{group_num}.json

Usage:
    python run_sdk_launcher.py --config model_config/test.json
    python run_sdk_launcher.py --config model_config/test.json --simulator
    python run_sdk_launcher.py --config model_config/test.json --warmup 5 --repeat 50
    python run_sdk_launcher.py --config model_config/test.json --once
"""
import argparse
import json
import os
import sys

from cerebras.sdk.client import SdkLauncher


def main():
    parser = argparse.ArgumentParser(
        description="SdkLauncher dispatch for Decode WSE-3-KV"
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to JSON config file"
    )
    parser.add_argument(
        "--simulator", action="store_true",
        help="Run in appliance simulator mode"
    )
    parser.add_argument(
        "--warmup", default=5, type=int,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--repeat", default=50, type=int,
        help="Number of repeat iterations"
    )
    parser.add_argument(
        "--once", action="store_true",
        help="(ignored, kept for CLI compatibility)"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    with open(args.config, "r", encoding="utf8") as f:
        config = json.load(f)

    P = config["P"]
    group_num = config["group_num"]

    artifact_json_path = f"compile_out/artifact_{P}_{group_num}.json"
    if not os.path.exists(artifact_json_path):
        print(
            f"Error: artifact JSON not found: {artifact_json_path}\n"
            f"Run compile.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(artifact_json_path, "r", encoding="utf8") as f:
        artifact_id = json.load(f)["artifact_id"]

    config_basename = os.path.basename(args.config)

    run_cmd = (
        f"cs_python launch_sim.py --config {config_basename} "
        f"--cmaddr %CMADDR%"
    )

    print(f"=== Decode WSE-3-KV: SdkLauncher Dispatch ===")
    print(f"Config       : {args.config}")
    print(f"Artifact     : {artifact_id}")
    print(f"Simulator    : {args.simulator}")
    print(f"Run command  : {run_cmd}")
    print()

    with SdkLauncher(artifact_id, simulator=args.simulator,
                     disable_version_check=True) as launcher:
        launcher.stage("launch_sim.py")
        launcher.stage(args.config)

        print(f"Executing on appliance...")
        response = launcher.run(run_cmd)

    print("Appliance response:")
    print(response)

    return response


if __name__ == "__main__":
    main()
