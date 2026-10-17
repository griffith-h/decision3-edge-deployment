"""Fetch unmodified SDK files from the pinned official repository; verify hashes."""
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

destination, manifest_path = map(Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text())
base = f"https://raw.githubusercontent.com/airockchip/rknn-toolkit2/{manifest['commit']}/"
paths = {
    'librknnrt.so': 'rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so',
    'rknn_api.h': 'rknpu2/runtime/Linux/librknn_api/include/rknn_api.h',
    'rknn_matmul_api.h': 'rknpu2/runtime/Linux/librknn_api/include/rknn_matmul_api.h',
    'LICENSE': 'LICENSE',
}
destination.mkdir(parents=True, exist_ok=True)
for name, path in paths.items():
    for attempt in range(3):
        try:
            with urllib.request.urlopen(base + path, timeout=120) as response:
                data = response.read()
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(2)
    if name in manifest['files']:
        assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256'], name
    (destination / name).write_bytes(data)
    print(f'Verified/downloaded official SDK: {name}', flush=True)
