"""Focused causal/padding/dtype checks against the installed stock formula."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from cpu_attention_optimized import causal_conv1d_vectorized

torch.set_num_threads(4)
torch.manual_seed(17)
records = []
for dtype in (torch.bfloat16, torch.float32):
    for batch, channels, width in ((1, 8, 1), (2, 16, 3), (2, 32, 17), (1, 6144, 115), (3, 64, 129)):
        x = torch.randn(batch, channels, width).to(dtype)
        weight = (torch.randn(channels, 4) * 0.03).to(dtype)
        for with_bias in (False, True):
            bias = torch.randn(channels).to(dtype) if with_bias else None
            for activation in (None, 'silu'):
                expected = F.conv1d(x, weight[:, None], bias, padding=3, groups=channels)[:, :, :width]
                if activation:
                    expected = F.silu(expected)
                actual = causal_conv1d_vectorized(x, weight, bias, activation)
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                torch.testing.assert_close(actual.float(), expected.float(), rtol=0.008 if dtype == torch.bfloat16 else 1e-5,
                                           atol=0.008 if dtype == torch.bfloat16 else 1e-6)
                if width > 1:
                    cut = width // 2
                    perturbed = x.clone()
                    perturbed[:, :, cut:] = 100
                    future = causal_conv1d_vectorized(perturbed, weight, bias, activation)
                    assert torch.equal(actual[:, :, :cut], future[:, :, :cut])
                records.append({'dtype': str(dtype), 'shape': list(x.shape), 'bias': with_bias,
                                'activation': activation, 'max_absolute_delta': (actual.float()-expected.float()).abs().max().item()})
Path('/work/logs/conv-unit.json').write_text(json.dumps({'passed': True, 'checks': records}, indent=2)+'\n')
print('PASS: 40 convolution cases, short-sequence padding, BF16/FP32, bias/SiLU, future-token invariance', flush=True)
