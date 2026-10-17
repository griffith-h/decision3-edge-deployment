import sys,os,ctypes
from pathlib import Path
sys.path[:0]=['/previous','/attn','/work']
from loader import open_model,compare,stats,state as oldstate
from conv_btc import install as install_conv
from prepared_scan import install as install_prepare
import chunk_blas2 as chunk_module
if os.environ.get('D3_CPU_TUNE')=='a72':
    old=chunk_module.lib;new=ctypes.CDLL(str(Path(__file__).resolve().parent/'a72/libchunk_blas2.so'))
    for name in ('d3_set_strsm','d3_set_sgemm','d3_chunk_scan'):
        getattr(new,name).argtypes=getattr(old,name).argtypes;getattr(new,name).restype=getattr(old,name).restype
    new.d3_set_strsm(ctypes.cast(chunk_module.blas.scipy_cblas_strsm64_,ctypes.c_void_p));new.d3_set_sgemm(ctypes.cast(chunk_module.blas.scipy_cblas_sgemm64_,ctypes.c_void_p));chunk_module.lib=new
from chunk_blas2 import install as install_chunk
from setup_helpers import cases

def load_baseline(profile=False,library=None):
    if os.environ.get('D3_CPU_TUNE')=='a72' and library and library.startswith('/work/libpipeline_'):library=library.replace('/work/','/work/a72/')
    model,engine,native,close=open_model(profile=profile,library=library or ('/previous/libprofile_conversion.so' if profile else '/previous/libconversion_bridge.so'))
    rconv,cconv=install_conv();rprep,cprep=install_prepare();rscan,cscan=install_chunk(32,64)
    def cleanup():rscan();rprep();rconv();close()
    return model,engine,{'conv':cconv,'prepare':cprep,'scan':cscan},cleanup

def state():
    import ctypes,os,torch
    from pathlib import Path
    from chunk_blas2 import blas
    result=oldstate()
    blas.scipy_openblas_get_num_threads64_.restype=ctypes.c_int64
    result['numpy_blas_threads']=blas.scipy_openblas_get_num_threads64_()
    result['mkldnn']=torch.backends.mkldnn.enabled
    result['pid']=os.getpid()
    for key,path in [('dmc_hz','/sys/class/devfreq/dmc/cur_freq'),('gpu_hz','/sys/class/devfreq/27800000.gpu/cur_freq')]:
        try:result[key]=Path(path).read_text().strip()
        except OSError:result[key]=None
    result['thermal']={p.parent.name:p.read_text().strip() for p in Path('/sys/class/thermal').glob('thermal_zone*/temp')}
    return result
