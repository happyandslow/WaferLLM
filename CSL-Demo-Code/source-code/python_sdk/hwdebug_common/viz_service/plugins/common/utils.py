import os
from google.protobuf.json_format import ParseDict
from cerebras.hwdebug_common.viz_service.plugins.common.core_cache import (
    CoreCache,
)

SIMFAB_TRACES = 'simfab_traces'

def query_string_to_request(qs, req):
    """
    Parse query string and fill req with values.

    Params:
    qs: Processed map of query strings returned by urllib.parse.parse_qs
        Example: {'grid_x': ['1'], 'grid_y': ['1'], 'colors': ['[2,3]']}
    req: Request proto obj to be filled
    """
    # convert url args to a dict
    qsdict = {}
    for x, y in qs.items():
        query_value = y[0]
        if query_value.startswith('[') and query_value.endswith(']'):
            # repeated field, like 'colors': '[2,3]' (or '[2]', or '[]')
            value_bracket_removed = query_value[1:-1]
            if not value_bracket_removed:
                # case of '[]'
                qsdict[x] = []
            else:
                # case of '[2,3]' or '[2]'
                qsdict[x] = value_bracket_removed.split(',')
        else:
            # single field, like 'grid_x': '1'
            qsdict[x] = query_value

    ParseDict(qsdict, req)


def get_core_cache(mgr, reqpath: str) -> CoreCache:
    """
    get_core_cache creates core cache if it has not been initialized,
    otherwise uses the previous initialized one.
    Throw ValueError if reqpath, request path, does not exist.
    """
    if not os.path.exists(reqpath):
        print(f'File {reqpath} does not exist')
        raise ValueError(f'File {reqpath} does not exist')

    with mgr.lock:
        cache = mgr.core_directory.get(reqpath, None)
        if cache is None:
            cache = CoreCache(reqpath)
            mgr.core_directory[reqpath] = cache

    return cache
