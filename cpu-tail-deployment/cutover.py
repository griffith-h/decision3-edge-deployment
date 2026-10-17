"""Guarded switch on prepared RK host; restore old container on failed activation."""
import json,subprocess,time,urllib.request
from pathlib import Path
OLD='decision3-edge-optimized';NEW='decision3-edge-graph';STAGE='decision3-edge-graph-staging'
RELEASE='2026-10-18-graph-half-v1'
def docker(*args):return subprocess.check_output(['docker',*args],text=True).strip()
def inspect(name):return json.loads(docker('inspect',name))[0]
def health(port):
    with urllib.request.urlopen(f'http://192.168.1.164:{port}/health',timeout=5) as r:return json.load(r)
def wait(port,release):
    for _ in range(180):
        try:
            d=health(port)
            if d['status']=='ready' and d['deployment']['release']==release:return d
        except (OSError,ValueError,KeyError):pass
        time.sleep(2)
    raise RuntimeError('Service did not become ready: '+release)
def activate(root):
    old=inspect(OLD)
    if not old['State']['Running']:raise RuntimeError('Previous service is not running')
    # Do all prerequisite checks before stopping either healthy endpoint.
    if health(8901)['deployment']['release']!=RELEASE:raise RuntimeError('Staging not verified')
    if subprocess.run(['docker','inspect',NEW],capture_output=True).returncode==0:raise RuntimeError('New container already exists; inspect before retrying')
    subprocess.run(['sha256sum','--quiet','-c','SHA256SUMS'],cwd=root,check=True)
    result={'old_id':old['Id'],'release':RELEASE,'events':[]}
    old_changed=False
    try:
        docker('stop','-t','90',STAGE);docker('rm',STAGE);result['events'].append('staging_stopped_cleanly')
        docker('update','--restart=no',OLD);old_changed=True
        docker('stop','-t','90',OLD);result['events'].append('old_stopped')
        subprocess.run(['bash',str(root/'run.sh'),NEW,'8900','unless-stopped'],check=True)
        result['health']=wait(8900,RELEASE);result['new_id']=inspect(NEW)['Id'];result['passed']=True
    except BaseException:
        if old_changed:
            subprocess.run(['bash',str(root/'rollback.sh')],check=True)
        raise
    finally:
        (root/'cutover-result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
if __name__=='__main__':print(json.dumps(activate(Path(__file__).resolve().parent)))
