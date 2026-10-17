"""One thread only inside compiled CPU kernels; Torch/native scan remain three."""
import setup_helpers,runpy,sys,json
from pathlib import Path
import torch
args=sys.argv[1:];script=sys.argv.pop(1);original=torch.compile
def one_thread(*pos,**kwargs):
    if kwargs.get('backend','inductor')=='inductor':
        kwargs['options']={**(kwargs.get('options') or {}),'cpp.threads':1}
    return original(*pos,**kwargs)
torch.compile=one_thread
try:runpy.run_path('/work/'+script,run_name='__main__')
finally:torch.compile=original
mode=args[args.index('--mode')+1];cpu=args[args.index('--compile')+1]
name=mode+'-'+cpu+'-static'
if 'paired.py' in args:name='paired-'+name
if 'scatter_runner.py' in args:name='scatter-'+name
p=Path('/work/logs/'+name+'.json');d=json.loads(p.read_text());d['cpp_kernel_threads_override']=1
target=p.with_name(p.stem+'-cpp1.json');p.rename(target);target.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
