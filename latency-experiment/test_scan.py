"""Test optimized scan against stock and preserve causal/cache boundaries."""
import json
from pathlib import Path
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from cpu_attention_optimized import chunk_delta_prefill, install_scan

torch.set_num_threads(4)
torch.manual_seed(29)
records = []
for dtype in (torch.float32, torch.bfloat16):
    for batch, width in ((1, 1), (2, 63), (1, 64), (2, 65), (1, 115), (2, 129), (1, 257)):
        query = torch.randn(batch, width, 4, 16).to(dtype)
        key = torch.randn_like(query)
        value = torch.randn(batch, width, 4, 12).to(dtype)
        decay = -torch.rand(batch, width, 4)
        beta = torch.rand(batch, width, 4).to(dtype)
        args = (query, key, value, decay, beta)
        expected, _ = qwen.torch_chunk_gated_delta_rule(*args, use_qk_l2norm_in_kernel=True)
        actual, state = chunk_delta_prefill(*args, use_qk_l2norm_in_kernel=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert state is None
        if width > 1:
            cut = width//2
            changed = value.clone()
            changed[:, cut:] = 100
            future, _ = chunk_delta_prefill(query, key, changed, decay, beta, use_qk_l2norm_in_kernel=True)
            assert torch.equal(actual[:, :cut], future[:, :cut])
        records.append({'dtype': str(dtype), 'batch': batch, 'length': width, 'exact_equal': True})
original = qwen.torch_chunk_gated_delta_rule
restore = install_scan()
try:
    initial = torch.randn(batch, 4, 16, 12)
    expected, e_state = original(*args, initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True)
    actual, a_state = qwen.torch_chunk_gated_delta_rule(*args, initial_state=initial, output_final_state=True, use_qk_l2norm_in_kernel=True)
    assert torch.equal(actual, expected) and torch.equal(a_state, e_state)
finally:
    restore()
assert qwen.torch_chunk_gated_delta_rule is original
Path('/work/logs/scan-unit.json').write_text(json.dumps({'passed': True, 'checks': records,
    'initial_state_and_final_state_fallback_exact': True, 'restore_passed': True}, indent=2)+'\n')
print('PASS: 14 scan cases exact, chunk edges, future-token invariance, cached/state fallback, restoration', flush=True)
