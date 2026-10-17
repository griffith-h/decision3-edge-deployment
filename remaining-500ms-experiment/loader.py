import sys
sys.path[:0]=['/projection','/remainder','/base','/combined','/scan','/release']
import setup_helpers
from common import load,compare,stats,state
from adapter import install
from scattered_adapter import PackedEngine
SNAPSHOT='/models/models--vllm-sr--d3-edge/snapshots/20c745653400d191ecacd99c4eafc0c5900dc15e'
def open_model(profile=False,library=None):
    import ctypes
    model,old,native,close=load(SNAPSHOT,safe_output=True)
    old.restore_forwards();old.restore_forwards=None
    engine=PackedEngine('balanced')
    if profile or library:
        original=engine.lib;engine.lib=ctypes.CDLL(library or '/work/libprofile_scatter.so')
        for name in ('mm_error','mm_create_core','mm_run','mm_destroy','mm_prepare_bf16','mm_copy_prepared','mm_run_prepared_bf16','mm_run_bf16','mm_run_prepared_scatter_bf16'):
            src=getattr(original,name);dst=getattr(engine.lib,name);dst.argtypes=src.argtypes;dst.restype=src.restype
        if profile:engine.lib.mm_profile_read.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_double),ctypes.c_int]
    engine.groups=[];engine.restore_forwards=install(model,engine)
    def cleanup():
        if engine.restore_forwards:engine.restore_forwards();engine.restore_forwards=None
        engine.close();close()
    return model,engine,native,cleanup
