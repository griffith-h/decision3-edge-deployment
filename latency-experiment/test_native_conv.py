"""Reuse the focused convolution checks and explicitly prove native dispatch."""
import json
from pathlib import Path
import runpy
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
import cpu_attention_optimized as cpu
from native_norm import install_native_conv

original = qwen.causal_conv1d_fn
vectorized = cpu.causal_conv1d_vectorized
stats = {}
unit_path = Path('/work/logs/conv-unit.json')
previous_unit_log = unit_path.read_text() if unit_path.exists() else None
restore = install_native_conv(stats)
cpu.causal_conv1d_vectorized = qwen.causal_conv1d_fn
try:
    with torch.inference_mode():
        runpy.run_path('/work/test_causal_conv.py')
    assert stats['native_calls'] > 20
    exact = []
    # Stock Transformers 5.17 ignores the extra block kwargs in this function.
    # Verify the native path also accepts them and is bitwise equal to our
    # vectorized baseline, including non-contiguous inputs.
    with torch.inference_mode():
        for width in (1, 3, 17, 115, 129):
            for bias_on in (False, True):
                x = torch.randn(2, 32, width*2).bfloat16()[:, :, ::2]
                w = (torch.randn(32, 4)*.03).bfloat16()
                b = torch.randn(32).bfloat16() if bias_on else None
                expected = vectorized(x, w, b, 'silu')
                actual = qwen.causal_conv1d_fn(x, w, b, 'silu', position_embeddings=('unused',))
                assert torch.equal(actual, expected)
                exact.append(dict(width=width, bias=bias_on, bitwise_equal=True))
    result = json.loads(Path('/work/logs/conv-unit.json').read_text())
    result['native_calls'] = stats['native_calls']
    result['bitwise_equivalence_cases'] = exact
    Path('/work/logs/native-conv-unit.json').write_text(json.dumps(result, indent=2)+'\n')
finally:
    if previous_unit_log is not None:
        unit_path.write_text(previous_unit_log)
    cpu.causal_conv1d_vectorized = vectorized
    restore()
assert qwen.causal_conv1d_fn is original
print('PASS: 40 native/fallback conv cases + 10 bitwise vectorized-equivalence cases; dispatch/kwargs/restoration verified', flush=True)
