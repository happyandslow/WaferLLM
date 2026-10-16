from cerebras.hwdebug_common.viz_service.plugins.sdk.utils import check_params


def check_trace_params(mgr, reqpath, gridx, gridy, cycle_start, cycle_end):
    """
    Throw ValueError is the params are not valid.
    """
    check_params(mgr, reqpath, gridx, gridy)

    if cycle_start > cycle_end and cycle_end != 0:
        print(f"Unexpected cycle_start:{cycle_start} or cycle_end:{cycle_end}")
        raise ValueError(
            f"Unexpected cycle_start:{cycle_start} or cycle_end:{cycle_end}"
        )
