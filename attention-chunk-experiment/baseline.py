import sys
sys.path[:0]=['/previous','/work']
from loader import open_model,compare,stats,state
from conv_btc import install as install_conv
from prepared_scan import install as install_prepare,prepare
from setup_helpers import cases

def load_baseline():
    model,engine,native,close=open_model(library='/previous/libconversion_bridge.so')
    rconv,cconv=install_conv();rprep,cprep=install_prepare()
    def cleanup():rprep();rconv();close()
    return model,engine,{'conv':cconv,'prepare':cprep},cleanup
