from fp32_owner import *
from pathlib import Path
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());app=None;r={'cases':{},'completed':False,'scope':'Known public FP32 output restored for B1/T615 only, disjoint context banks; all615 BF16 MLP boundaries. Original-current gate authoritative.'}
try:
 app=Owner();r['execution']=app.execution
 for key,ref in refs['cases'].items():
  if ref['output']['usage']['input_tokens']!=615:continue
  row={'request':ref['request'],'current_reference':ref['output'],'records':[]};r['cases'][key]=row
  for mode in ('float_readout','mlponly_readout','projection_readout'):
   app.use(mode);app.invoke(ref['request']);app.use(mode);t=time.perf_counter();out=app.invoke(ref['request']);elapsed=time.perf_counter()-t
   rec={'mode':mode,'seconds':elapsed,'output':out,'vs_current':detailed(out,ref['output']),**app.counts()};row['records'].append(rec)
   print('DOMAIN SCREEN',key,mode,elapsed,rec['vs_current'],rec['precise_errors'],flush=True);(p/'logs/screen-domains.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True
finally:
 if app:app.close()
 (p/'logs/screen-domains.json').write_text(json.dumps(r,indent=2)+'\n')
