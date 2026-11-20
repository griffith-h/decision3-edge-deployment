"""RK host launcher, historical source mounts remain read-only."""
import argparse
import os
from pathlib import Path
import re
import shutil
import sys
sys.path.insert(0,'/opt/decision3-official-evaluation-final-20261019')
from stages import RELEASE,mounts


def build_command(root,run_id,case,profile=False,variant='baseline',repeat=1,retain_exit_state=False):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in ('.','..'):
        raise ValueError('Invalid run identity')
    result=root/'results'/run_id
    result.mkdir(parents=True,exist_ok=False)
    source_snapshot=root/'sources'/run_id
    source_snapshot.mkdir(parents=True,exist_ok=False)
    for file in root.iterdir():
        if file.is_file() and file.suffix in ('.py','.c','.cpp','.inc','.h','.so','.sh','.json','.md'):
            shutil.copy2(file,source_snapshot/file.name)
    cmd=['docker','run']+([] if retain_exit_state else ['--rm'])+['-i','--init','--name','decision3-official-opt',
         '--label','decision3.opt.run='+run_id,'--cpuset-cpus','4-7',
         '--memory','8g','--memory-swap','8g','--device','/dev/dri',
         '--ulimit','core=0','--ulimit','nofile=4096:4096',
         '-e','PYTHONDONTWRITEBYTECODE=1','-e','OMP_NUM_THREADS=3',
         '-e','OMP_THREAD_LIMIT=4','-e','OPENBLAS_NUM_THREADS=3',
         '-e','TOKENIZERS_PARALLELISM=false']
    if case=='request-cost-ledger':
        cmd+=['-e','LD_PRELOAD=/lab/librequest_sdk_trace.so']
    for alias,source in mounts('native-bf16').items():
        cmd+=['--mount',f'type=bind,source={source},target=/{alias},readonly']
    bindings=[('evaluation','/opt/decision3-official-evaluation-final-20261019',True),
              ('lab',source_snapshot,True),('inputs',root/'inputs',True),('results',result,False)]
    if case=='selected-dot-diagnosis':
        capture=root/'captures'/run_id;capture.mkdir(parents=True,mode=0o700,exist_ok=False)
        bindings.extend([('pmu',root/'pmu',True),('captures',capture,False)])
    for alias,source,readonly in bindings:
        cmd+=['--mount',f'type=bind,source={source},target=/{alias}'+(',readonly' if readonly else '')]
    cmd+=['--workdir','/experiment','--entrypoint','python',
          (RELEASE/'image-id.txt').read_text().strip(),'-u','/lab/probe.py','--case',case]
    if profile:cmd+=['--profile']
    cmd+=['--variant',variant,'--repeat',str(repeat)]
    return cmd


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--repeat',type=int,default=1);p.add_argument('--case',required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--variant',choices=['baseline','batch','mlp','batch-mlp','batch-mlp-projection','batch-mlp-projection-fp32','batch-mlp-projection-fp32-attention','batch-mlp-projection-fp32-scan','batch-mlp-projection-fp32-scan-npu-attention','batch-mlp-projection-fp32-scan-npu-attention-readout','medium-npu-attention','full-batch','full-batch-precise-medium','full-batch-sparse','full-batch-sparse-cblas','full-batch-sparse-native','full-batch-sparse-fused','full-batch-sparse-flash','full-batch-dense-medium','full-batch-sparse-vector','full-batch-sparse-diagonal','full-batch-sparse-compact','full-batch-sparse-band2','full-batch-sparse-sequential','full-batch-sparse-band25','full-batch-sparse-native-scan','full-batch-sparse-visual-scope','full-batch-sparse-request-scope','reference','cpu-full'],default='baseline');a=p.parse_args()
    cmd=build_command(Path(__file__).resolve().parent,a.run,a.case,a.profile,a.variant,a.repeat,retain_exit_state=True)
    os.execvp(cmd[0],cmd)


if __name__=='__main__':main()
