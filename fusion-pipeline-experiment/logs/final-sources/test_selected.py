"""Integration fault boundaries for the actual frozen selected Owner."""
from selected_owner import *
from pathlib import Path
import long_kernels as lk
p=Path('/experiment');refs=json.loads((p/'logs/references.json').read_text());case=refs['cases']['words-512']['request'];app=None;r={'passed':False}
try:
 app=Owner();app.use('candidate');out=app.invoke(case);r['normal_counts']=app.counts();r['vs_current']=detailed(out,refs['cases']['words-512']['output']);assert r['vs_current']['passed']
 readout=getattr(app,'last_readout',None)
 if readout and app.execution.get('readout',False):
  saved_mode=app.model.attention_mode;app.model.attention_mode='unsupported-for-private-readout'
  try:assert not readout.good([[0]*615],[3])
  finally:app.model.attention_mode=saved_mode
  r['changed_attention_mode_bypasses']=True
  app.use('candidate');readout.enabled=False;full=app.invoke(case)
  old=readout.encode_last;readout.encode_last=lambda *a:(_ for _ in ()).throw(RuntimeError('injected selected readout failure'))
  try:
   app.use('candidate');fallback=app.invoke(case);assert readout.disabled and detailed(fallback,full)['all_numeric_max_delta']==0
  finally:readout.encode_last=old
  app.use('candidate');assert detailed(app.invoke(case),full)['all_numeric_max_delta']==0;r['whole_readout_fallback_persistent_disable']=True
 # The existing qualified wrapper must discard a failed native attention and
 # persist on original attention; compare to that exact original-attention path.
 app.use('candidate');app.long.enabled.discard('attention');expected=app.invoke(case)
 app.use('candidate');saved=lk.attention
 lk.attention=lambda *a:(_ for _ in ()).throw(RuntimeError('injected selected attention failure'))
 try:
  actual=app.invoke(case);assert detailed(actual,expected)['all_numeric_max_delta']==0 and 'attention' in app.long.disabled and app.long.counts['attention']['native_fallback']==1
 finally:lk.attention=saved
 app.use('candidate');assert detailed(app.invoke(case),expected)['all_numeric_max_delta']==0;r['whole_attention_fallback_persistent_disable']=True
 app.close();app.close();assert not app.engine.handles and not app.native_attention.handles and app.model.logits==app.last_readout.original;r['closed_zero_handles']=True;r['passed']=True
finally:
 if app:app.close()
 (p/'logs/selected-unit.json').write_text(json.dumps(r,indent=2)+'\n')
print('SELECTED UNIT PASS',r,flush=True)
