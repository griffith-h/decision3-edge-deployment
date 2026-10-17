"""Text-only precision path, noncausal/padding masks, vision fallthrough."""
import json
from pathlib import Path
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from cpu_attention_optimized import install_sdpa

torch.manual_seed(37)
interface = qwen.ALL_ATTENTION_FUNCTIONS
original = interface["sdpa"]
config = Qwen3_5TextConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=2,
                          head_dim=16, num_hidden_layers=1, layer_types=["full_attention"])
text_module = qwen.Qwen3_5Attention(config, layer_idx=0)
other_module = torch.nn.Module()
other_module.is_causal = False
records = []
restore = install_sdpa()
try:
    for dtype in (torch.float32, torch.bfloat16):
        q = torch.randn(2, 2, 17, 16).to(dtype)
        k, v = torch.randn_like(q), torch.randn_like(q)
        valid = torch.ones(2, 1, 1, 17, dtype=torch.bool)
        valid[1, ..., :3] = False
        for float_mask in (False, True):
            mask = torch.zeros(valid.shape, dtype=dtype).masked_fill(~valid, float('-inf')) if float_mask else valid
            expected, _ = original(text_module, q.float(), k.float(), v.float(), mask.float() if float_mask else mask, is_causal=False)
            actual, _ = interface["sdpa"](text_module, q, k, v, attention_mask=mask, is_causal=False)
            assert torch.equal(actual, expected.to(dtype))
            changed = v.clone()
            changed[1, :, :3] += 100
            padding, _ = interface["sdpa"](text_module, q, k, changed, mask, is_causal=False)
            assert torch.equal(actual[1], padding[1])
            changed = v.clone()
            changed[0, :, -1] += 10
            future, _ = interface["sdpa"](text_module, q, k, changed, mask, is_causal=False)
            assert not torch.equal(actual[0, 0], future[0, 0])
            vision_expected, _ = original(other_module, q, k, v, mask, is_causal=False)
            vision_actual, _ = interface["sdpa"](other_module, q, k, v, mask, is_causal=False)
            assert torch.equal(vision_actual, vision_expected)
            records.append({"dtype": str(dtype), "float_mask": float_mask, "passed": True})
finally:
    restore()
assert interface["sdpa"] is original
Path("/work/logs/sdpa-unit.json").write_text(json.dumps({"passed": True, "checks": records,
    "future_keys_visible": True, "left_padding_hidden": True, "non_text_attention_exact_fallthrough": True,
    "restore_passed": True}, indent=2)+"\n")
print("PASS: text-only FP32/BF16 attention, masks, future keys, padding, exact non-text fallthrough", flush=True)
