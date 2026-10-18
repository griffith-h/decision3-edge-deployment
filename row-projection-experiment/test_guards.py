from test_rows import Engine,fixture
from row_projection import RowProjection
from pathlib import Path
import torch,json,copy
# Fresh fixture proves grad guard, rather than relying on an already invalid weight.
e=Engine();model,parent=fixture(e);c=RowProjection(model,e);c.mode='chunked'
try:
 with torch.enable_grad():
  x=torch.ones(1,513,32,dtype=torch.bfloat16,requires_grad=True);out=parent(x);assert out.requires_grad and not e.calls;out.sum().backward();assert x.grad is not None
finally:c.close()
# Constructor failure must restore an already wrapped parent.
e=Engine();model,parent=fixture(e);saved=parent.forward;e.groups.append({'parent':'missing','members':[]})
try:RowProjection(model,e);raise AssertionError('Expected constructor failure')
except KeyError:pass
assert parent.forward==saved
(P:=Path('/experiment/logs/extra-guards.json')).write_text(json.dumps({'passed':True,'grad_guard_preserves_autograd':True,'partial_constructor_restores_parent':True},indent=2)+'\n');print('EXTRA GUARDS PASS')
