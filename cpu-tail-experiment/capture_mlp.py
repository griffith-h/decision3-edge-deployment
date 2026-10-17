import runpy,json,torch
from pathlib import Path
from baseline import load_baseline
from lut_ops import Tables
from candidate import Candidate
from validate_candidate import Evaluation
# Focused failure/ownership regression checks share the imported runtime.
runpy.run_path('/work/test_pipeline_guards.py',run_name='__main__')
runpy.run_path('/work/test_gpu_guards.py',run_name='__main__')
prior=json.loads(Path('/work/logs/pipeline-screen.json').read_text())['variants'];valid={n:v for n,v in prior.items() if v['passed']};selected=min(valid,key=lambda n:valid[n]['timing']['median']);cfg=valid[selected]['config']
model,engine,bc,close=load_baseline(library='/work/libpipeline_bridge.so');candidate=Candidate(model,engine,Tables());candidate.use(cfg);ev=Evaluation(model);hooks=[];rows=[]
try:
 ev.invoke()
 for name,m in model.backbone.named_modules():
  if type(m).__name__!='Qwen3_5MLP':continue
  row={'name':name,'weights':{key:getattr(m,key).weight.detach().clone() for key in ('gate_proj','up_proj','down_proj')}};rows.append(row)
  def before(m,args,_r=row):_r['input']=args[0].detach().clone()
  def after(m,args,out,_r=row):_r['output']=out.detach().clone()
  hooks.extend([m.register_forward_pre_hook(before),m.register_forward_hook(after)])
 response=ev.invoke();assert ev.check(response)['passed'];assert len(rows)==12 and all(r['input'].shape==(1,115,1024) for r in rows)
 torch.save(rows,'/work/mlp-layers.pt');Path('/work/logs/mlp-capture.json').write_text(json.dumps({'layers':[{'name':r['name'],'shape':list(r['input'].shape)} for r in rows],'reference':ev.check(response),'config':cfg},indent=2)+'\n');print('MLP CAPTURE',len(rows),flush=True)
finally:
 for h in hooks:h.remove()
 candidate.reset();close()
