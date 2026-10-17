"""Network-enabled setup step. Inference containers stay offline."""
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = '20c745653400d191ecacd99c4eafc0c5900dc15e'
path = Path(snapshot_download('vllm-sr/d3-edge', revision=REVISION, local_dir='/models/d3-edge'))
import sys
sys.path.insert(0, str(path))
from d3_runtime import verify_package
verify_package(path, mode='full')
print(f'Pinned d3-edge downloaded and fully verified: {REVISION}')
