from experiment import *
from planned_mlp import PlannedBank
import copy
from types import MethodType
app=Owner();app.bank.close();app.bank=PlannedBank(app.model);(P/'logs/planned-profile-calibration.json').write_text(json.dumps(app.bank.calibration,indent=2)+'\n')
cases=json.loads((P/'logs/planned-screen.json').read_text())['cases'];ids=['novel-math','words-64','words-128','chinese-0','words-256']
r={}
try:
 for key in ids:
  c=cases[key]['request'];r[key]={}
  for mode in ['fixed','planned']:
   app.use(mode);app.invoke(c);records=[];restores=[];sample=[]
   for i,m in enumerate(x for x in app.model.backbone.modules() if type(x).__name__=='Qwen3_5MLP'):
    old=m.forward
    def forward(layer,x,_old=old,_i=i):
     t=time.perf_counter();out=_old(x);sample.append({'layer':_i,'seconds':time.perf_counter()-t});return out
    m.forward=MethodType(forward,m);restores.append((m,old))
   try:
    for _ in range(5):
     sample.clear();app.bank.reset_counts();t=time.perf_counter();app.invoke(c);records.append({'seconds':time.perf_counter()-t,'mlp_seconds':sum(x['seconds'] for x in sample),'layers':copy.deepcopy(sample),'counts':copy.deepcopy(app.bank.counts)})
   finally:
    for m,old in restores:m.forward=old
   r[key][mode]={'records':records,'total':stats([x['seconds'] for x in records]),'mlp':stats([x['mlp_seconds'] for x in records])};print('PROFILE',key,mode,r[key][mode]['total']['median'],r[key][mode]['mlp']['median'],flush=True)
 (P/'logs/planned-profile.json').write_text(json.dumps(r,indent=2)+'\n')
finally:app.close()
