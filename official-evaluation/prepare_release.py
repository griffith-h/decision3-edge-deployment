"""Freeze the evaluated final Owner after the complete sample run; no service changes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

from stages import HISTORY, PIN, RELEASE, mounts
from evaluation_verdict import require_final_complete


def digest(file):
    h = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_sources(identity, bindings, evaluation_root):
    for virtual, expected in identity['files'].items():
        parts = Path(virtual).parts
        alias = parts[1]
        if alias == 'evaluation':
            # The frozen benchmark worker directory is never overwritten during testing.
            source = Path(evaluation_root) / Path(*parts[2:])
        else:
            source = Path(bindings[alias]) / Path(*parts[2:])
        if digest(source) != expected:
            raise RuntimeError('Evaluated source changed: ' + virtual)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--verdict', required=True)
    p.add_argument('--identity', required=True)
    p.add_argument('--destination', default='/opt/decision3-edge-native-bf16-20261019-v1')
    p.add_argument('--evaluation-root', default='/opt/decision3-official-evaluation-final-20261019')
    a = p.parse_args()
    verdict = json.loads(Path(a.verdict).read_text())
    identity = json.loads(Path(a.identity).read_text())
    require_final_complete(verdict)
    if verdict['final_stage'] != 'native-bf16' or identity['stage']['id'] != 'native-bf16':
        raise RuntimeError('Wrong final stage')
    if PIN not in identity['model']:
        raise RuntimeError('Wrong model revision')
    target = Path(a.destination).resolve()
    if target.exists():
        raise FileExistsError('Refusing to overwrite a release: ' + str(target))
    original = mounts('native-bf16')
    verify_sources(identity, original, a.evaluation_root)
    target.mkdir()
    bindings = dict(original)
    selected_suffixes = {'.py', '.so', '.rknn', '.json', '.c', '.cpp', '.inc', '.sh'}
    try:
        for alias in ('mlp', 'row', 'long', 'tail', 'fusion', 'native'):
            destination = target / 'deps' / alias
            destination.mkdir(parents=True)
            for source in Path(HISTORY[alias]).iterdir():
                if source.is_file() and source.suffix in selected_suffixes:
                    shutil.copy2(source, destination / source.name)
            bindings[alias] = str(destination)
        # The final CoreOwner freezes its historical per-shape plans from this file.
        tail_logs = target / 'deps/tail/logs'
        tail_logs.mkdir()
        shutil.copy2(Path(HISTORY['tail']) / 'logs/formal.json', tail_logs / 'formal.json')
        bindings['experiment'] = bindings['native']
        tools = target / 'service'
        tools.mkdir()
        for name in ('board.py', 'stages.py', 'serve_final.py', 'evaluation_verdict.py'):
            source = Path(__file__).resolve().parent / name
            evaluated = identity['files'].get('/evaluation/' + name)
            if evaluated != digest(source):
                raise RuntimeError('Service factory differs from the evaluated source: ' + name)
            shutil.copy2(source, tools / name)
        shutil.copy2(a.verdict, tools / 'evaluation-result.json')
        shutil.copy2(a.identity, target / 'evaluated-identity.json')
        bindings['evaluation'] = str(tools)
        (target / 'bindings.json').write_text(json.dumps(bindings, indent=2) + '\n')
        image = (RELEASE / 'image-id.txt').read_text().strip()
        (target / 'image-id.txt').write_text(image + '\n')
        files = {str(f.relative_to(target)): digest(f) for f in target.rglob('*') if f.is_file()}
        manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    'image': image, 'model_revision': PIN, 'sample_rows_sha256': verdict['sample_rows_sha256'],
                    'files': files, 'bindings': bindings,
                    'note': 'Existing graph-half frozen work/dependencies and external model snapshot remain read-only mounts.'}
        (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        # Check that source assets stayed identical during the copy.
        verify_sources(identity, original, a.evaluation_root)
        for relative, expected in files.items():
            if digest(target / relative) != expected:
                raise RuntimeError('Release copy hash mismatch: ' + relative)
        print(json.dumps({'release': str(target), 'files': len(files), 'image': image,
                          'sample_rows_sha256': verdict['sample_rows_sha256']}), flush=True)
    except BaseException:
        # Preserve partial files for diagnosis, never advertise them as a finished release.
        (target / 'INCOMPLETE').touch()
        raise


if __name__ == '__main__':
    main()
