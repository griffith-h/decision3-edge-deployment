from precision_readout_owner import *
from pathlib import Path
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());app=None;r={'cases':{},'completed':False,'scope':'Restores original BF16 MLP intermediate boundaries for every615 row MLP, not response adjustment.'}
try:
 app=Owner();r['execution']=app.execution
 for key,ref in refs['cases'].items():
  if ref['family'] not in ('intended_matched615','existing') or ref['output']['usage']['input_tokens']!=615:continue
  row={'request':ref['request'],'current_reference':ref['output'],'records':[]};r['cases'][key]=row
  for mode in ('candidate','readout','precise','precise_readout','precisewide','precisewide_readout'):
   app.use(mode);app.invoke(ref['request']);app.use(mode);t=time.perf_counter();out=app.invoke(ref['request']);elapsed=time.perf_counter()-t;rec={'mode':mode,'seconds':elapsed,'output':out,'vs_current':detailed(out,ref['output']),**app.counts()};row['records'].append(rec);print('PRECISION615',key,mode,elapsed,rec['vs_current'],flush=True);(p/'logs/screen-precision615.json').write_text(json.dumps(r,indent=2)+'\n')
 r['completed']=True
finally:
 if app:app.close()
 (p/'logs/screen-precision615.json').write_text(json.dumps(r,indent=2)+'\n')
