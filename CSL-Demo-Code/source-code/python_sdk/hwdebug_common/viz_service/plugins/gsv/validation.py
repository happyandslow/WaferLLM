import os
from pathlib import Path
from typing import Optional

GLOBAL_STALL_VIEW = "global_stall_view.json"
COMPLETION_MARKER = ".retention-marker.out"
GLOBAL_STALL_SUMMARY_DB = "gsv_db.sqlite"
VIZ_DATAFLOW_VIEW = "viz_dataflow_view.json"


def find_gsv_dir(workdir: str) -> str:
    """
    Supports two types of directory structures:
    1) ApplianceTestEnv/RT only - workdir is coordinator dir
    2) Appliance HW run - workdir wsjob dir
    """
    workdir_path = Path(workdir)
    if os.path.exists(workdir_path / GLOBAL_STALL_VIEW):
        return workdir
    coordinator_dir = workdir_path / "coordinator-0"
    latest_session_dir = coordinator_dir / "sessions" / "latest"
    if os.path.exists(coordinator_dir) and os.path.exists(
        latest_session_dir / GLOBAL_STALL_VIEW
    ):
        return str(latest_session_dir)
    return None


def validate_gsv_workdir(workdir: str) -> Optional[dict]:
    """Check if the given workdir is a valid Global Stall View workdir.
    If the workdir is invalid, returns a validation error response.
    Otherwise, returns None.
    # For 2.5 Appliance HW e2e run where artifacts are automatically log-exported,
    # we need to check for a completion marker
    # log export without using debug volume: */wsjob-<x>/log-export-wsjob-<x>-<hash>/
    # log export using debug volume (only available starting from rel-2.5):  /<n1|pvc>/debug-artifact/*/wsjob-<x>/
    """
    gsv_path = os.path.join(workdir, GLOBAL_STALL_VIEW)
    if workdir is None or not os.path.exists(gsv_path):
        return {
            "message": (
                "Invalid workdir. "
                + f"Locate the directory that contains {GLOBAL_STALL_VIEW}"
            ),
            "repair": False,
        }
    # workdir is wsjob-<x>/coordinator/sessions/latest
    job_dir = Path(workdir).parent.parent.parent
    marker_path = job_dir / COMPLETION_MARKER
    if (
        "/n1/debug-artifact/" in workdir or "/pvc/debug-artifact/" in workdir
    ) and not os.path.exists(marker_path):
        return {
            "message": (
                "Check back later, log-export has not completed for the workdir. "
                + "Contact Cerebras SW if problem persists."
            ),
            "repair": False,
        }


def validate_summary_view(workdir: str) -> Optional[dict]:
    """Check if the given workdir supports GSV summary view.
    If summary view is not supported, returns a validation error response.
    Otherwise, returns None.
    """
    if workdir is None or not os.path.exists(
        os.path.join(workdir, GLOBAL_STALL_SUMMARY_DB)
    ):
        return {
            "message": "Summary view not available.",
            "repair": True,
        }


def validate_dataflow_view(workdir: str) -> Optional[dict]:
    """Check if the given workdir supports GSV dataflow view.
    If dataflow view is not supported, returns a validation error response.
    Otherwise, returns None.
    """
    if workdir is None or not os.path.exists(
        os.path.join(workdir, VIZ_DATAFLOW_VIEW)
    ):
        return {
            "message": "Dataflow view not available.",
            "repair": True,
        }
