import json
from pathlib import Path

def cpu_choice():
 for name in ('thread-sweep.json','pipeline-screen.json','prepare-screen.json'):
  path=Path('/work/logs')/name
  if not path.exists():continue
  data=json.loads(path.read_text())
  if not data.get('completed'):continue
  valid={n:r for n,r in data['variants'].items() if r['passed']}
  if valid:
   selected=min(valid,key=lambda n:valid[n]['timing']['median'])
   if name=='thread-sweep.json' and 't3' in valid and valid[selected]['timing']['median']>=.99*valid['t3']['timing']['median']:selected='t3'
   return name,selected,dict(valid[selected]['config'])
 raise RuntimeError('No completed qualified CPU candidate')
