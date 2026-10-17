import argparse,json,time,traceback
from pathlib import Path
from rknn.api import RKNN
p=argparse.ArgumentParser();p.add_argument('name');a=p.parse_args();root=Path('/work');result={'name':a.name,'quantized':False,'target':'rk3576'}
r=RKNN(verbose=True,verbose_file=str(root/'logs'/('rknn-'+a.name+'.log')))
try:
    t=time.perf_counter();r.config(target_platform='rk3576',optimization_level=3)
    result['load_return']=r.load_onnx(model=str(root/(a.name+'.onnx')))
    if result['load_return']==0:
        result['build_return']=r.build(do_quantization=False)
        if result['build_return']==0:result['export_return']=r.export_rknn(str(root/(a.name+'.rknn')))
    result['seconds']=time.perf_counter()-t
except Exception:result['error']=traceback.format_exc();print(result['error'],flush=True)
finally:
    r.release();(root/'logs'/('compile-'+a.name+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
