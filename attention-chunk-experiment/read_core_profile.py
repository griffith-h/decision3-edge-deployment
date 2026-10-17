import core_graph as graph,ctypes,numpy as np,json
from pathlib import Path
old=graph.lib;lib=ctypes.CDLL('/work/libcore_profile.so')
for name in ('graph_create','graph_error','graph_input_name','graph_run','graph_destroy'):
    getattr(lib,name).argtypes=getattr(old,name).argtypes;getattr(lib,name).restype=getattr(old,name).restype
graph.lib=lib;lib.graph_profile.argtypes=[ctypes.c_void_p];lib.graph_profile.restype=ctypes.c_char_p
records=[]
for i in range(1):
    with np.load(f'/work/core-mask-{i}-all.npz') as d:inputs={k:d[k] for k in ('q','k','v','mask')}
    g=graph.Graph('/work/attention-core.rknn')
    try:
        out=g.run(inputs);assert np.isfinite(out).all();profile=lib.graph_profile(g.handle).decode(errors='replace');Path('/work/logs/core-perf.txt').write_text(profile);records.append({'layer':i,'bytes':len(profile)})
    finally:g.close()
print(records,flush=True)
