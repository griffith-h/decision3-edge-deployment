from experiment import *
from bucket_mlp import BucketBank
import copy
from types import MethodType
app=Owner();app.bank.close();app.bank=BucketBank(app.model)
cases=json.loads((P/'logs/bucket-screen.json').read_text())['cases'];ids=['novel-math','words-64','words-128','chinese-0','words-256']
r={}
try:
 for key in ids:
  c=cases[key]['request'];r[key]={}
  for mode in ['fixed','bucket']:
   app.use(mode);app.invoke(c);records=[];restores=[];sample=[]
   for i,m in enumerate(x for x in app.model.backbone.modules() if type(x).__name__=='Qwen3_5MLP'):
    old=m.forward
    def forward(layer,x,_old=old,_i=i):
     t=time.perf_counter();out=_old(x);sample.append({'layer':_i,'seconds':time.perf_counter()-t});return out
    m.forward=MethodType(forward,m);restores.append((m,old))
   try:
    for _ in range(5):
     sample.clear();t=time.perf_counter();app.invoke(c);records.append({'seconds':time.perf_counter()-t,'mlp_seconds':sum(x['seconds'] for x in sample),'layers':copy.deepcopy(sample)})
   finally:
    for m,old in restores:m.forward=old
   r[key][mode]={'records':records,'total':stats([x['seconds'] for x in records]),'mlp':stats([x['mlp_seconds'] for x in records])};print('PROFILE',key,mode,r[key][mode]['total']['median'],r[key][mode]['mlp']['median'],flush=True)
 (P/'logs/bucket-profile.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
