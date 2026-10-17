"""Alternate chunk partitions preserve stock equations and prefix causality."""
import json
from pathlib import Path
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from cpu_attention_optimized import chunk_delta_prefill, install_scan

torch.set_num_threads(4)
torch.manual_seed(48)
records = []
for chunk in (128, 256):
    for width in (1, 63, 64, 65, 115, 127, 128, 129, 255, 256, 257):
        for dtype in (torch.float32, torch.bfloat16):
            query = torch.randn(1, width, 2, 16).to(dtype)
            key = torch.randn_like(query)
            value = torch.randn(1, width, 2, 12).to(dtype)
            decay = -torch.rand(1, width, 2)
            beta = torch.rand(1, width, 2).to(dtype)
            args = (query, key, value, decay, beta)
            expected, _ = qwen.torch_chunk_gated_delta_rule(*args, chunk_size=chunk, use_qk_l2norm_in_kernel=True)
            actual, _ = chunk_delta_prefill(*args, chunk_size=chunk, use_qk_l2norm_in_kernel=True)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            restore = install_scan(chunk)
            try:
                installed, _ = qwen.torch_chunk_gated_delta_rule(*args, use_qk_l2norm_in_kernel=True)
                assert torch.equal(installed, actual)
            finally:
                restore()
            if width > 1:
                cut = width//2
                changed = value.clone()
                changed[:, cut:] = 100
                future, _ = chunk_delta_prefill(query, key, changed, decay, beta, chunk_size=chunk,
                                               use_qk_l2norm_in_kernel=True)
                assert torch.equal(actual[:, :cut], future[:, :cut])
            records.append(dict(chunk=chunk, width=width, dtype=str(dtype), exact=True))
Path('/work/logs/chunk-sizes-unit.json').write_text(json.dumps({'passed': True, 'cases': records}, indent=2)+'\n')
print('PASS: 44 chunk partition cases exact vs stock same partition; future-token invariance, install/restore', flush=True)
