"""Runs on the RK host; mounts historical stages read-only for the serial worker."""
import argparse
import os
from pathlib import Path

from stages import RELEASE, mounts, stage_spec


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', required=True)
    a = p.parse_args()
    spec = stage_spec(a.stage)
    evaluation = Path(__file__).resolve().parent
    image = (RELEASE / 'image-id.txt').read_text().strip()
    cmd = ['docker', 'run', '--rm', '-i', '--init', '--name', 'decision3-official-eval',
           '--ulimit', 'core=0', '--ulimit', 'nofile=4096:4096', '--cpuset-cpus', '4-7',
           '--memory', '8g', '--memory-swap', '8g', '--device', '/dev/dri',
           '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'TOKENIZERS_PARALLELISM=false',
           '-e', f'OMP_NUM_THREADS={spec["threads"]}',
           '-e', f'OMP_THREAD_LIMIT={spec["thread_limit"]}',
           '-e', f'OPENBLAS_NUM_THREADS={spec["threads"]}']
    for target, source in mounts(a.stage).items():
        if not Path(source).is_dir():
            raise FileNotFoundError(source)
        cmd += ['--mount', f'type=bind,source={source},target=/{target},readonly']
    cmd += ['--mount', f'type=bind,source={evaluation},target=/evaluation,readonly',
            '--workdir', '/experiment' if spec['experiment'] else '/work',
            '--entrypoint', 'python', image, '-u', '/evaluation/board.py', '--stage', a.stage]
    os.execvp(cmd[0], cmd)


if __name__ == '__main__':
    main()
