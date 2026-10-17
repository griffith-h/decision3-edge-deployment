import sys
sys.path[:0]=['/previous','/attn','/work']
from loader import open_model,compare,stats,state
from conv_btc import install as install_conv
from prepared_scan import install as install_prepare
from chunk_blas2 import install as install_chunk
from setup_helpers import cases

def load_baseline(profile=False):
    model,engine,native,close=open_model(profile=profile,library='/previous/libprofile_conversion.so' if profile else '/previous/libconversion_bridge.so')
    rconv,cconv=install_conv();rprep,cprep=install_prepare();rscan,cscan=install_chunk(32,64)
    def cleanup():rscan();rprep();rconv();close()
    return model,engine,{'conv':cconv,'prepare':cprep,'scan':cscan},cleanup
