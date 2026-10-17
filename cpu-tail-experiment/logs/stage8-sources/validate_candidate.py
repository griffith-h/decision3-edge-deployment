"""Shared fixed numerical references and full-case validation, no shape coercion."""
import json,torch,hashlib
from pathlib import Path
from baseline import compare,cases
class Evaluation:
 def __init__(self,model):
  self.model=model;self.all=cases();self.single=next(c for c in self.all if c['id']=='single')
  self.refs=json.loads(Path('/attn/logs/final-single-scan.json').read_text())['cases']
  self.fp32=json.loads(Path('/remainder/logs/model-results.json').read_text())['baseline_cases']
  p=json.loads(Path('/work/logs/lut-screen.json').read_text());self.extra=p['extra_cases'];self.extra_refs=p['extra_refs']
 def invoke(self,c=None):
  c=c or self.single
  return self.model.system_one(**{k:c[k] for k in ('state','questions','images','videos') if k in c})
 def check(self,r,c=None):
  c=c or self.single;a=compare(r,self.refs[c['id']]['response']);b=compare(r,self.fp32[c['id']]);return {'vs_442':a,'vs_fp32':b,'passed':a['passed'] and b['passed']}
 def validate(self):
  row={'cases':{},'extra':{}}
  for c in self.all:
   out=self.invoke(c);row['cases'][c['id']]={'response':out,**self.check(out,c)}
  for c in self.extra:
   out=self.invoke(c);row['extra'][c['id']]={'response':out,**compare(out,self.extra_refs[c['id']])}
  row['passed']=all(v['passed'] for k in ('cases','extra') for v in row[k].values());return row

def sources():
 files=[p for p in Path('/work').iterdir() if p.suffix in ('.py','.c','.cpp','.so')]
 files+=list(Path('/work/a72').glob('*.so'))
 return {str(p.relative_to('/work')):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
