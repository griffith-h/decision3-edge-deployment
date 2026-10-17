import json
from pathlib import Path
root=Path('/work/logs');read=lambda n:json.loads((root/('environment-'+n+'.json')).read_text())
a,b=read('default-before'),read('default-after');assert a['passed'] and b['passed']
base=[a['timing']['median'],b['timing']['median']];stable=max(base)/min(base)<1.05
options={n:read(n) for n in ('spin0','spin1000','spin10000')};qualified={n:r for n,r in options.items() if r['passed']};chosen=min(qualified,key=lambda n:qualified[n]['timing']['median']) if qualified else None
if not stable or chosen is None or qualified[chosen]['timing']['median']>=.95*min(base):chosen='default'
result={'spin':None if chosen=='default' else {'spin0':'0','spin1000':'1000','spin10000':'10000'}[chosen],'selected_environment':chosen,'default_medians':base,'controls_stable_5pct':stable,'policy_medians':{n:r['timing']['median'] for n,r in options.items()},'rule':'Keep default unless candidate clears .01 numerical checks and is >=5pct faster than both default controls, with default controls agreeing within5pct.'}
Path('/work/runtime-selection.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
