"""Check BF16 boundaries, noncontiguous inputs, fallback and restoration."""
import json
from pathlib import Path
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from native_norm import install_norm

torch.set_num_threads(4)
torch.manual_seed(71)
records = []
for cls in (qwen.Qwen3_5RMSNorm, qwen.Qwen3_5RMSNormGated):
    for dim in (17, 128, 1024):
        for noncontiguous in (False, True):
            m = cls(dim).bfloat16()
            m.weight.data.uniform_(-.5, 1.5)
            x = torch.randn(2, dim, 19).transpose(1, 2).bfloat16()
            if not noncontiguous:
                x = x.contiguous()
            gate = torch.randn_like(x)
            args = (x, gate) if cls is qwen.Qwen3_5RMSNormGated else (x,)
            original = m.forward
            with torch.inference_mode():
                expected = original(*args)
                stats = {}
                restore = install_norm(m, stats)
                try:
                    actual = m(*args)
                    assert stats['native_calls'] == 1
                    torch.testing.assert_close(actual, expected, atol=.03125, rtol=.01)
                    fp_args = tuple(t.float() for t in args)
                    assert torch.equal(m(*fp_args), original(*fp_args))
                    assert stats['native_calls'] == 1
                finally:
                    restore()
            assert m.forward == original
            # Gradient-enabled calls use the original implementation.
            restore = install_norm(m)
            try:
                y = x.detach().requires_grad_()
                out = m(y, gate) if len(args) == 2 else m(y)
                out.float().sum().backward()
                assert y.grad is not None
            finally:
                restore()
            records.append(dict(cls=cls.__name__, dim=dim, noncontiguous=noncontiguous,
                                max_delta=(actual.float()-expected.float()).abs().max().item()))
Path('/work/logs/native-norm-unit.json').write_text(json.dumps({'passed': True, 'cases': records}, indent=2)+'\n')
print('PASS: 12 native norm cases, BF16 boundaries, FP32/gradient fallback, restoration', flush=True)
