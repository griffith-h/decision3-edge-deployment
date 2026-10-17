# -*- coding: utf-8 -*-
"""Final deployment audit; reads target state without exposing environment secrets."""
import datetime,hashlib,json,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
code="""import json,subprocess,hashlib
from pathlib import Path
h=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def c(name):
 d=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0];host=d['HostConfig'].copy();host.pop('RestartPolicy',None)
 return {'name':name,'id':d['Id'],'started_at':d['State']['StartedAt'],'running':d['State']['Running'],'status':d['State']['Status'],'health':d['State'].get('Health',{}).get('Status'),'oom_killed':d['State']['OOMKilled'],'exit_code':d['State']['ExitCode'],'image':d['Image'],'restart':d['HostConfig']['RestartPolicy'],'config_sha256':h(d['Config']),'host_config_without_restart_sha256':h(host),'mounts':[{'source':m['Source'],'destination':m['Destination'],'rw':m['RW']} for m in d['Mounts']],'ports':d['HostConfig']['PortBindings']}
p=Path('/opt/decision3-edge-optimized-20261017-v1');release=Path('/opt/decision3-edge-graph-20261018-v1')
subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=release,check=True)
print(json.dumps({'new':c('decision3-edge-graph'),'old':c('decision3-edge-optimized'),'old_source_hashes':{str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in p.rglob('*') if f.is_file() and '__pycache__' not in str(f)},'remaining_staging':subprocess.check_output(['docker','ps','-a','--filter','name=decision3-edge-graph-staging','--format','{{.Names}}'],text=True).splitlines(),'package_manifest_sha256':hashlib.sha256((release/'manifest.json').read_bytes()).hexdigest(),'package_manifest':json.loads((release/'manifest.json').read_text())}))
"""
s=subprocess.run(['ssh','root@192.168.1.164','python3 -'],input=code,text=True,capture_output=True,check=True);state=json.loads(s.stdout)
old=json.loads((P/'previous-state.json').read_text());now=state['old']
for k in ('id','image','config_sha256','host_config_without_restart_sha256'):assert now[k]==old[k],k
assert state['old_source_hashes']==old['source_hashes'] and not now['running'] and now['restart']['Name']=='no'
new=state['new'];assert new['running'] and new['health']=='healthy' and new['restart']['Name']=='unless-stopped' and not new['oom_killed']
assert not state['remaining_staging']
prod=json.loads((P/'production-http.json').read_text());stage=json.loads((P/'staging-http.json').read_text());rollback=json.loads((P/'rollback-rehearsal.json').read_text())
assert prod['passed'] and stage['passed'] and rollback['passed'] and prod['timing']['count']==100 and len(prod['cases'])==19
maxdelta=max(x['vs_frozen_including_confidence']['max_delta'] for x in prod['cases'].values());assert maxdelta<=1e-6
health=prod['final_health'];assert health['deployment']['release']=='2026-10-18-graph-half-v1'
result={'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':True,'release':'2026-10-18-graph-half-v1','release_directory':'/opt/decision3-edge-graph-20261018-v1','endpoint':'http://192.168.1.164:8900/v1/systemone','new_container':new,'previous_container':now,'previous_config_and_files_preserved_except_restart':True,'actual_rollback_and_return_passed':True,'staging_removed':True,'package_manifest_sha256':state['package_manifest_sha256'],'staging_timing':stage['timing'],'health':health,'http':{'timing':prod['timing'],'max_delta_vs_frozen':maxdelta,'cases':{k:{'seconds':v['seconds'],'route':v['route'],'vs_frozen_including_confidence':v['vs_frozen_including_confidence'],'vs_442_excluding_confidence':v['vs_442_excluding_confidence'],'vs_442_including_confidence':v['vs_442_including_confidence']} for k,v in prod['cases'].items()}},'limitations':['graph fast path B1T115 only; other shapes CPU fallback','318.51ms historical warm system_one method; current HTTP measured separately','historical .01 gate excludes confidence; confidence max difference .01002684 separately disclosed','no new labelled-task accuracy, video, concurrent-throughput or cold-start SLA evaluation']}
(P/'package-manifest.json').write_text(json.dumps(state['package_manifest'],indent=2)+'\n');(P/'deployment-result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('DEPLOYMENT AUDIT PASS',prod['timing'],maxdelta)
