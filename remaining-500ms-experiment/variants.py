from loader import PackedEngine
import ctypes

def engine_with_library(library,mode='balanced'):
    engine=PackedEngine(mode);old=engine.lib;engine.lib=ctypes.CDLL(library)
    for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16','mm_run_prepared_scatter_bf16'):
        src=getattr(old,name);dst=getattr(engine.lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
    engine.restore_forwards=None
    return engine
