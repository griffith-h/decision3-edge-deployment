"""Run one private diagnostic, bound inference time, restore retained service on exit."""
import argparse,datetime,json,os,selectors,shlex,signal,subprocess,time,re,fcntl,math
from contextlib import contextmanager
from pathlib import Path
HOST='root@192.168.1.164';REMOTE='/opt/decision3-official-opt-20261019-r001';OLD='decision3-edge-graph';NAME='decision3-official-opt'
def ssh(command,**kwargs):
 kwargs.setdefault('timeout',110)
 return subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,command],check=True,**kwargs)
def interrupt_safely(signum,frame):
 signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
 raise KeyboardInterrupt('Interrupted; cleaning owned worker and restoring retained service')
def request_budget(event,maximum):
 # A diagnostic can narrow a phase's cap (e.g. candidate120, CPU reference600),
 # never widen the caller-authorized limit or silently accept invalid values.
 value=event.get('budget_seconds',maximum)
 if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
  raise ValueError('Invalid per-phase diagnostic budget')
 return min(value,maximum)
@contextmanager
def exclusive_controller():
 path=Path('output/decision-index-audit/iterations/.controller.lock')
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('a') as lock:
  try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise RuntimeError('Another diagnostic controller is running or restoring the retained service; refusing concurrent launch') from None
  try:yield
  finally:fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
def main():
 # Hold ownership through worker removal, evidence copy AND service health.
 with exclusive_controller():run()
def run():
 p=argparse.ArgumentParser();p.add_argument('--iteration',default='r001');p.add_argument('--case',required=True);p.add_argument('--repeat',type=int,default=1);p.add_argument('--profile',action='store_true');p.add_argument('--variant',choices=['baseline','batch','mlp','batch-mlp','batch-mlp-projection','batch-mlp-projection-fp32','batch-mlp-projection-fp32-attention','batch-mlp-projection-fp32-scan','batch-mlp-projection-fp32-scan-npu-attention','batch-mlp-projection-fp32-scan-npu-attention-readout','medium-npu-attention','full-batch','full-batch-precise-medium','full-batch-sparse','full-batch-sparse-cblas','full-batch-sparse-native','full-batch-sparse-fused','full-batch-sparse-flash','full-batch-dense-medium','full-batch-sparse-vector','full-batch-sparse-diagonal','full-batch-sparse-compact','full-batch-sparse-band2','full-batch-sparse-sequential','full-batch-sparse-band25','full-batch-sparse-native-scan','full-batch-sparse-visual-scope','full-batch-sparse-request-scope','full-batch-sparse-no-output','reference','cpu-full'],default='baseline');p.add_argument('--budget',type=float,default=120);a=p.parse_args()
 if not re.fullmatch(r'r[0-9]{3}',a.iteration):raise ValueError('Invalid iteration identity')
 if not 1<=a.repeat<=5 or not 0<a.budget<=600:raise ValueError('Invalid development repeat or budget')
 remote=REMOTE if a.iteration=='r001' else REMOTE.rsplit('-',1)[0]+'-'+a.iteration
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+a.case+('-'+a.variant)+('-profile' if a.profile else '-plain')
 out=Path('output/decision-index-audit/iterations')/a.iteration/stamp;out.mkdir()
 existing=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,'docker inspect '+NAME],capture_output=True,timeout=15)
 if existing.returncode!=1:raise RuntimeError('Worker already exists or availability could not be verified; refusing concurrent launch')
 info=json.loads(ssh('docker inspect '+OLD,capture_output=True,text=True).stdout)[0];original={'id':info['Id'],'was_running':info['State']['Running']};status={'iteration':a.iteration,'run':stamp,'case':a.case,'profile':a.profile,'variant':a.variant,'budget':a.budget,'repeat':a.repeat,'retained':original};process=None
 args=['python3',remote+'/host_run.py','--run',stamp,'--case',a.case]
 if a.profile:args+=['--profile']
 args+=['--variant',a.variant,'--repeat',str(a.repeat)]
 signal.signal(signal.SIGTERM,interrupt_safely);signal.signal(signal.SIGINT,interrupt_safely)
 try:
  ssh('docker stop -t 90 '+OLD,stdout=subprocess.DEVNULL)
  with (out/'stderr.log').open('wb') as errors,(out/'stdout.log').open('wb') as log:
   process=subprocess.Popen(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,shlex.join(args)],stdout=subprocess.PIPE,stderr=errors,bufsize=0)
   selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ);buffer=b'';deadline=time.monotonic()+600
   while True:
    if time.monotonic()>deadline:
     status['budget_exceeded']=True;print('DEVELOPMENT_BUDGET_EXCEEDED',stamp,flush=True);break
    if not selector.select(1):
     if process.poll() is not None:break
     continue
    chunk=os.read(process.stdout.fileno(),65536)
    if not chunk:break
    log.write(chunk);log.flush();buffer+=chunk
    while b'\n' in buffer:
     line,buffer=buffer.split(b'\n',1);marker=line.find(b'@@D3OPT@@')
     if marker<0:continue
     event=json.loads(line[marker+9:]);status.setdefault('events',[]).append(event);print(json.dumps(event),flush=True)
     if event['event']=='request_begin':deadline=time.monotonic()+request_budget(event,a.budget)
     if event['event']=='complete':deadline=time.monotonic()+60
   selector.close()
 finally:
  errors=[]
  try:
   inspected=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8',HOST,'docker inspect '+NAME],capture_output=True,text=True,timeout=15)
   if inspected.returncode==0:
    container=json.loads(inspected.stdout)[0]
    if container['Config'].get('Labels',{}).get('decision3.opt.run')!=stamp:raise RuntimeError('Refusing to remove unowned experiment container')
    state=container.get('State',{})
    status['worker_final_state']={k:state.get(k) for k in ('Status','Running','ExitCode','OOMKilled','Error','StartedAt','FinishedAt')}
    ssh('docker rm -f '+NAME,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   if process:
    try:process.wait(timeout=15)
    except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=10)
    status['exit_code']=process.returncode
   copied=subprocess.run(['scp','-q','-r',HOST+':'+remote+'/results/'+stamp+'/.',str(out)],capture_output=True,timeout=60)
   status['copied']=copied.returncode==0
  except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
  finally:
   try:
    if original['was_running']:
     current=json.loads(ssh('docker inspect '+OLD,capture_output=True,text=True).stdout)[0]
     assert current['Id']==original['id'],'Retained service identity changed'
     ssh('docker start '+OLD,stdout=subprocess.DEVNULL)
     for _ in range(90):
      health=ssh("docker inspect --format '{{.State.Health.Status}}' "+OLD,capture_output=True,text=True).stdout.strip()
      if health=='healthy':break
      time.sleep(2)
     status['retained_healthy']=health=='healthy'
   except Exception as exc:errors.append(type(exc).__name__+': '+str(exc))
   finally:
    status['request_completed']=any(e['event']=='complete' for e in status.get('events',[]))
    status['cleanup_errors']=errors
    (out/'execution.json').write_text(json.dumps(status,indent=2)+'\n');print('DONE',stamp,'completed',status['request_completed'],'service_healthy',status.get('retained_healthy'),flush=True)
  if errors or original['was_running'] and not status.get('retained_healthy'):raise RuntimeError('Cleanup or retained-service restoration failed; see execution.json')
 if not status.get('budget_exceeded') and (status.get('exit_code')!=0 or not status['request_completed']):raise RuntimeError('Probe failed before completion; see stderr.log and execution.json')
if __name__=='__main__':main()
