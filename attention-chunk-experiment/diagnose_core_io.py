import core_graph,ctypes,numpy as np,json
from pathlib import Path
original=core_graph.lib;rows=[]
for variant in ('diagnostic','nchw'):
    lib=ctypes.CDLL('/work/libcore_'+variant+'.so')
    for name in ('graph_create','graph_error','graph_input_name','graph_run','graph_destroy'):
        getattr(lib,name).argtypes=getattr(original,name).argtypes;getattr(lib,name).restype=getattr(original,name).restype
    core_graph.lib=lib;lib.graph_attributes.argtypes=[ctypes.c_void_p];lib.graph_attributes.restype=ctypes.c_char_p
    g=core_graph.Graph('/work/attention-core.rknn')
    try:
        print(variant,lib.graph_attributes(g.handle).decode(),flush=True)
        for label in ('all','left','last_only'):
            with np.load(f'/work/core-mask-0-{label}.npz') as d:inputs={k:d[k] for k in ('q','k','v','mask')};ref=d['reference']
            out=g.run(inputs);row={'variant':variant,'mask':label,'rel_l2':float(np.linalg.norm(out-ref)/np.linalg.norm(ref)),'max_abs':float(np.max(np.abs(out-ref)))}
            rows.append(row);print(row,flush=True)
    finally:g.close()
Path('/work/logs/core-io-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')
