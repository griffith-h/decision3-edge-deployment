"""Read only owned Torch zip metadata/int64 selected positions, no Torch runtime."""
import collections,hashlib,io,json,pickle,struct,zipfile
from pathlib import Path
class Reader(pickle.Unpickler):
 def find_class(self,module,name):
  if module=='torch' and name in ('BFloat16Storage','LongStorage'):return name
  if module=='torch._utils' and name=='_rebuild_tensor_v2':
   return lambda storage,offset,size,stride,*_:dict(storage=storage,offset=offset,size=size,stride=stride)
  if module=='collections' and name=='OrderedDict':return collections.OrderedDict
  raise ValueError((module,name))
 def persistent_load(self,pid):
  assert pid[0]=='storage';return {'dtype':pid[1],'key':pid[2],'count':pid[4]}
root=Path('/opt/decision3-official-opt-20261019-r089')
run='20261020T051005-selected-dot-diagnosis-full-batch-sparse-request-scope-plain'
meta=json.loads((root/'results'/run/'selected-dot-diagnosis.json').read_text());rows=[]
for cap in meta['captures']:
 file=root/'captures'/run/cap['capture_file'];assert hashlib.sha256(file.read_bytes()).hexdigest()==cap['capture_file_sha256']
 with zipfile.ZipFile(file) as z:
  prefix=next(n[:-8] for n in z.namelist() if n.endswith('data.pkl'))
  data=Reader(io.BytesIO(z.read(prefix+'data.pkl'))).load();pos=data['positions']
  assert pos['storage']['dtype']=='LongStorage' and pos['offset']==0 and tuple(pos['stride'])==(2,1)
  raw=z.read(prefix+'data/'+pos['storage']['key'])
  assert hashlib.sha256(raw).hexdigest()==cap['positions_sha256']
  pairs=list(struct.iter_unpack('qq',raw));assert len(pairs)==cap['selected']
  counts=collections.Counter(r for r,c in pairs);assert all(0<=r<cap['shape'][0] for r in counts)
  widths={w:{'groups':sum((n+w-1)//w for n in counts.values()),'computed_lanes':sum((n+w-1)//w*w for n in counts.values())} for w in (4,8,16)}
  for w in widths:
   widths[w]['extra_lanes_fraction']=widths[w]['computed_lanes']/cap['selected']-1
   widths[w]['K_hot_instruction_ratio_to_old']=widths[w]['groups']*{4:31,8:57,16:113}[w]/(widths[4]['groups']*31)
  rows.append({'shape':cap['shape'],'selected':cap['selected'],'widths':widths})
print(json.dumps({'rows':rows,'scope':'Only aggregation of exact stored selected-row counts and frozen hot-loop instruction counts; not timings or a scheduling model.'},indent=2))
