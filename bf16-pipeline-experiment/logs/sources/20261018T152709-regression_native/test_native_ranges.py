"""Public SDK synthetic exact matrices exercise native FP32 guards, not a model."""
import torch,ctypes,json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from split_mlp import SplitMlp
p=Path('/experiment');torch.set_num_threads(3)
raw=torch.arange(65536,dtype=torch.int32).to(torch.uint16).view(torch.bfloat16);table=SimpleNamespace(silu=torch.nn.functional.silu(raw));engine=SimpleNamespace(pool=ThreadPoolExecutor(2),core={},bucket={});bank=SplitMlp(engine,table);r={'cases':[],'completed':False};keep=[]
def members():
 values=[torch.nn.Linear(128,128,bias=False,dtype=torch.bfloat16) for _ in range(3)]
 for m in values:m.weight.copy_(torch.eye(128,dtype=torch.bfloat16));m.eval()
 engine.core.update({id(values[0]):1,id(values[1]):2});keep.append(values);return values
def expect(name,modules,x,code):
 try:bank.run(modules,x);raise AssertionError('Guard did not reject '+name)
 except ValueError as exc:assert str(exc).endswith(str(code)),(name,str(exc));r['cases'].append({'name':name,'code':code,'rejected':True})
try:
 with torch.inference_mode():
  standard=members();x=torch.ones(1,615,128,dtype=torch.bfloat16);out=bank.run(standard,x);assert torch.isfinite(out).all()
  expect('finite_product_overflow',standard,torch.full_like(x,256.),-2)
  expect('input_overflow',standard,torch.full_like(x,100000.),-2)
  expect('input_nan',standard,torch.full_like(x,float('nan')),-2)
  expect('input_inf',standard,torch.full_like(x,float('inf')),-2)
  assert torch.equal(bank.run(standard,x),out);r['recover_after_range']=True
  near=members();near[0].weight.zero_();near[0].weight[:,0]=512.;near[0].weight[:,1]=-16.;xx=torch.zeros_like(x);xx[:,:,0]=128.;xx[:,:,1]=2.5
  expect('finite_raw_65496_original_saturation_guard',near,xx,-3)
  assert torch.isfinite(bank.run(near,x)).all()
  down=members();down[1].weight.mul_(1/128);down[2].weight.mul_(512.)
  expect('finite_down_output_65536',down,torch.full_like(x,128.),-3)
  assert torch.isfinite(bank.run(down,x)).all();r['completed']=True
finally:
 engine.pool.shutdown(wait=True);bank.close();bank.close();r['handles_after_close']=len(bank.handles);(p/'logs/native-ranges.json').write_text(json.dumps(r,indent=2))
print('NATIVE RANGES',r,flush=True)
