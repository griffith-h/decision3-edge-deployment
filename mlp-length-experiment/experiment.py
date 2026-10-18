"""One model owner, frozen CPU kernels, persistent graph bank, synchronized toggles."""
import sys,json,time,math,statistics,hashlib
from pathlib import Path
sys.path[:0]=['/experiment','/work']
from baseline import load_baseline,state,stats
from candidate import Candidate
from lut_ops import Tables
from row_mlp import RowBank
P=Path('/experiment')
def numeric(x,path='',confidence=False):
 if isinstance(x,dict):return {a:b for k,v in x.items() if k!='legend' and (confidence or k!='confidence') for a,b in numeric(v,path+'/'+k,confidence).items()}
 if type(x) in (int,float):
  assert math.isfinite(x),(path,x)
  return {path:x}
 return {}
def compare(out,ref):
 x,y=numeric(out['answers']),numeric(ref['answers']);assert x.keys()==y.keys();a,b=numeric(out['answers'],confidence=True),numeric(ref['answers'],confidence=True)
 same=all(v.get('choice')==ref['answers'][k].get('choice') for k,v in out['answers'].items());d=max(abs(v-y[k]) for k,v in x.items());c=max((abs(v-b[k]) for k,v in a.items() if k.endswith('/confidence')),default=0)
 return {'max_delta':d,'confidence_max_delta':c,'all_numeric_max_delta':max(abs(v-b[k]) for k,v in a.items()),'choices_unchanged':same,'passed':d<=.01 and same}
class Owner:
 def __init__(self):
  self.model,self.engine,self.bc,self.cleanup=load_baseline(library='/work/libpipeline_bridge.so');self.candidate=Candidate(self.model,self.engine,Tables());cfg=json.loads((P/'config.json').read_text());self.candidate.use(cfg)
  self.bank=RowBank(self.model)
 def invoke(self,c):return self.model.system_one(**{k:v for k,v in c.items() if k in ('state','questions','images','videos')})
 def use(self,mode):self.bank.mode=mode;self.bank.reset_counts()
 def close(self):self.bank.close();self.candidate.reset();self.cleanup()
