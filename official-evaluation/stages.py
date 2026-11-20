"""Historical adopted endpoints; every entry uses original checked-in math."""
from pathlib import Path

PIN = '20c745653400d191ecacd99c4eafc0c5900dc15e'
MODEL = '/models/models--vllm-sr--d3-edge/snapshots/' + PIN
RELEASE = Path('/opt/decision3-edge-graph-20261018-v1')
HISTORY = {
    'references': '/opt/decision3-edge-matmul',
    'combined': '/opt/decision3-edge-bridge-parallel-20261017',
    'scan': '/opt/decision3-edge-scan-20261017',
    'remainder': '/opt/decision3-edge-remainder-20261017',
    'projection': '/opt/decision3-edge-projection-compile-20261017',
    'previous': '/opt/decision3-edge-remaining-500ms-20261017',
    'attn': '/opt/decision3-edge-attention-chunk-20261017',
    'mlp': '/opt/decision3-edge-mlp-length-20261018',
    'row': '/opt/decision3-edge-row-projection-20261018',
    'long': '/opt/decision3-edge-long-sequence-20261018',
    'tail': '/opt/decision3-edge-long-tail-20261018',
    'fusion': '/opt/decision3-edge-fusion-pipeline-20261018',
    'native': '/opt/decision3-edge-bf16-pipeline-20261018',
}
# id, label, work alias, experiment alias, Torch/OMP threads, OMP upper limit
STAGES = [
    ('official-bf16', '原始 BF16 CPU 参照', 'references', None, 4, 4),
    ('cpu-fp32', 'CPU Linear / Conv FP32 计算适配', 'references', None, 4, 4),
    ('rknn-initial', '初版 CPU + NPU MatMul', 'references', None, 4, 4),
    ('cpu-three-thread', 'CPU 算子与三线程部署版', 'references', None, 3, 3),
    ('bridge-parallel', '桥接融合与并行', 'combined', None, 3, 3),
    ('scan-native', '自有 CPU 扫描', 'scan', None, 3, 3),
    ('safe-half', '安全 FP16 输出', 'remainder', None, 3, 3),
    ('projection-merge', '投影合并', 'projection', None, 3, 3),
    ('layout-scatter', '布局与数据搬运融合', 'previous', None, 3, 3),
    ('scan-blas2', 'BLAS2 扫描', 'attn', None, 3, 3),
    ('graph-half', 'graph-half v1，原线上版本', 'release', None, 3, 3),
    ('mlp-length', '多长度 MLP 图计划', 'release', 'mlp', 3, 3),
    ('row-projection', '长输入投影与融合', 'release', 'row', 3, 3),
    ('long-sequence', '长序列注意力与扫描', 'release', 'long', 3, 3),
    ('long-tail', '长输入检查与 MLP 调度', 'release', 'tail', 3, 3),
    ('fusion-precision', '动态注意力、精度与 readout', 'release', 'fusion', 3, 4),
    ('native-bf16', 'native BF16 边界 MLP，最新候选', 'release', 'native', 3, 4),
]


def stage_spec(stage):
    row = next(r for r in STAGES if r[0] == stage)
    return dict(zip(('id', 'label', 'work', 'experiment', 'threads', 'thread_limit'), row))


def mounts(stage):
    spec = stage_spec(stage)
    paths = {key: str(RELEASE / 'deps' / key) for key in
             ('references', 'combined', 'scan', 'remainder', 'projection', 'previous', 'attn')}
    paths.update({key: HISTORY[key] for key in ('mlp', 'row', 'long', 'tail', 'fusion', 'native')})
    paths.update(release=str(RELEASE / 'deps/release'),
                 base=str(RELEASE / 'deps/release/base'), service=str(RELEASE / 'service'))
    # Early stages use the original workspace where relative shared libraries live.
    paths['work'] = str(RELEASE / 'work') if spec['work'] == 'release' else HISTORY[spec['work']]
    if spec['experiment']:
        paths['experiment'] = HISTORY[spec['experiment']]
    paths['models'] = '/root/.local/state/vllm-sr/engine/vllm-sr--d3-edge/models/model-runtime'
    return paths
