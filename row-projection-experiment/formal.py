from fused_owner import *
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--candidate',choices=['chunked','fused','fused_guarded'],default='fused_guarded');selected=parser.parse_args().candidate
from pathlib import Path
from metrics import detailed
app=Owner();screen=json.loads((P/'logs/screen.json').read_text());r={'candidate':selected,'settings':state(),'inventory':app.projections.inventory,'cases':{},'extra':[],'scope':'single 20 pairs; wide 5 interleaved MLP-only controls and 20 chunked samples, fresh current reference per input'};path=P/'logs/formal.json'
def save():path.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
def invoke(c,mode,ref):
 app.use(mode);before=app.engine.cpu_fallbacks;kernel_before={'scan':dict(app.bc['scan']),'attention':dict(app.candidate.counts['attention'])};t=time.perf_counter();out=app.invoke(c);elapsed=time.perf_counter()-t
 rec={'seconds':elapsed,**detailed(out,ref),**app.counts(),'engine_cpu_fallbacks':app.engine.cpu_fallbacks-before};kernel_after={'scan':app.bc['scan'],'attention':app.candidate.counts['attention']};rec['cpu_kernel_delta']={k:{n:v-kernel_before[k][n] for n,v in values.items()} for k,values in kernel_after.items()};return out,rec
try:
 for key in ['single','words-410','words-512','multi-wide']:
  c=screen['cases'][key]['request'];row={'request':c,'variants':{m:{'records':[]} for m in ['mlp_only',selected]}};r['cases'][key]=row
  print('FORMAL BEGIN',key,flush=True);app.use('current');t=time.perf_counter();ref=app.invoke(c);row['reference']=ref;row['first_current_seconds']=time.perf_counter()-t;row['input_tokens']=ref['usage']['input_tokens'];save()
  for mode in ['mlp_only',selected]:invoke(c,mode,ref)
  row['handles_after_warmup']=len(app.engine.handles);row['create_seconds_after_warmup']=app.engine.create_seconds
  ncontrol=20 if key=='single' else 5;nper=1 if key in ('single','multi-wide') else 4
  for block in range(ncontrol):
   modes=['mlp_only']+[selected]*nper if block%2==0 else [selected]*nper+['mlp_only']
   for mode in modes:
    out,rec=invoke(c,mode,ref);row['variants'][mode]['records'].append(rec);save()
    assert not rec['projections']['errors'],(key,mode,rec)
    if mode==selected:assert rec['passed'],(key,mode,rec)
    if mode==selected and key!='single':assert rec['engine_cpu_fallbacks']==3*rec.get('mlp_policy_fallbacks',0) and rec['projections']['optimized_projections']>0 and rec['projections']['native_max_rows']<=512 and rec['projections']['range_fallback']==rec['projections']['runtime_fallback']==rec['projections']['guard_fallback']==0
  for v in row['variants'].values():v['timing']=stats([x['seconds'] for x in v['records']])
  row['handles_after_timing']=len(app.engine.handles);assert row['handles_after_timing']==row['handles_after_warmup']
  row['reduction_vs_mlp']=1-row['variants'][selected]['timing']['median']/row['variants']['mlp_only']['timing']['median'];save();print('FORMAL',key,[(m,v['timing']['median']) for m,v in row['variants'].items()],row['reduction_vs_mlp'],flush=True)
 # Repeated natural 115-token requests cover bypass paths and unchanged scoring.
 for c in json.loads(Path('/mlp/http-cases.json').read_text())[7:]:
  app.use('current');ref=app.invoke(c)
  for _ in range(2):
   out,rec=invoke(c,selected,ref);r['extra'].append({'id':c['id'],'input_tokens':out['usage']['input_tokens'],**rec});assert rec['passed'] and not rec['projections']['errors']
  save()
 r['completed']=True;r['passed']=True;r['final_settings']=state();save()
finally:app.close()
