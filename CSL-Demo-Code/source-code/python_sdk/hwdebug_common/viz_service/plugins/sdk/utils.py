import os

from cerebras.hwdebug_common.viz_service.plugins.sdk.compile_cache import CompileCache

def get_compile_cache(mgr, reqpath: str) -> CompileCache:
    """
    get_compile_cache creates compile cache if it has not been initialized,
    otherwise uses the previous initialized one.
    Throw ValueError if reqpath, request path, does not exist.
    """
    if not os.path.isdir(reqpath):
        print(f'Directory {reqpath} does not exist')
        raise ValueError(f'Directory {reqpath} does not exist')

    elf_dir = get_elf_dir(reqpath)
    if not os.path.isdir(elf_dir):
        print(f'Directory {elf_dir} does not exist')
        raise ValueError(f'Directory {elf_dir} does not exist')

    with mgr.lock:
        cache = mgr.compile_directory.get(elf_dir, None)
        if cache is None:
            # read compile artifacts
            cache = CompileCache(elf_dir)
            mgr.compile_directory[elf_dir] = cache

    return cache


def check_params(mgr, reqpath, gridx, gridy):
    """
    Throw ValueError is the params are not valid.
    """
    cache = get_compile_cache(mgr, reqpath)

    xsize, ysize = cache.getFabricSize()
    if not 0 <= gridx < xsize or not 0 <= gridy < ysize:
        print(f'Unexpected gridx:{gridx} or gridy:{gridy}')
        raise ValueError(f'Unexpected gridx:{gridx} or gridy:{gridy}')


def get_elf_dir(compile_dir):
    return os.path.join(compile_dir, "bin")
