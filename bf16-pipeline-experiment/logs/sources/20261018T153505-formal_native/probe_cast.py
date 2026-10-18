"""Public Toolkit BF16 cast acceptance probe; not an accuracy claim."""
import json,traceback
from pathlib import Path
import numpy as np,onnx
from onnx import helper as h,TensorProto as T,numpy_helper as nh
from rknn.api import RKNN
p=Path('/experiment');records=[]
for mode in ('bf16','half','round'):
 nodes=[]
 if mode=='round':
  # These values stay small enough for a half multiplier. This probe only
  # checks support, not equivalence to BF16 at arbitrary exponents.
  init=[nh.from_array(np.array(128,dtype='float32'),'scale')]
  nodes=[h.make_node('Mul',['x','scale'],['scaled']),h.make_node('Round',['scaled'],['rounded']),h.make_node('Div',['rounded','scale'],['y'])]
 else:
  init=[];nodes=[h.make_node('Cast',['x'],['intermediate'],to=T.BFLOAT16 if mode=='bf16' else T.FLOAT16),h.make_node('Cast',['intermediate'],['y'],to=T.FLOAT)]
 graph=h.make_graph(nodes,'boundary',[h.make_tensor_value_info('x',T.FLOAT,[1,128,1024])],[h.make_tensor_value_info('y',T.FLOAT,[1,128,1024])],initializer=init)
 model=h.make_model(graph,opset_imports=[h.make_opsetid('',18)]);model.ir_version=9;onnx.save(model,str(p/f'probe-{mode}.onnx'))
 r=RKNN(verbose=True,verbose_file=str(p/f'logs/cast-{mode}.log'));rec={'mode':mode}
 try:
  rec['config']=r.config(target_platform='rk3576',optimization_level=3)
  rec['load']=r.load_onnx(model=str(p/f'probe-{mode}.onnx'))
  if rec['load']==0:
   rec['build']=r.build(do_quantization=False)
   if rec['build']==0:rec['export']=r.export_rknn(str(p/f'probe-{mode}.rknn'))
 except BaseException:rec['error']=traceback.format_exc()
 finally:r.release()
 records.append(rec);(p/'logs/cast-build.json').write_text(json.dumps(records,indent=2));print('CAST',rec,flush=True)
