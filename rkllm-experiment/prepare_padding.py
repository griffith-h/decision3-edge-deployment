#!/usr/bin/env python3
"""Pad the irregular d3 layout to LLF repeats using verified identity layers."""
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

root = Path("/work")
source = root / "source-model"
dest = root / "prepared-model"
dest.mkdir(exist_ok=True)
config = json.loads((source / "config.json").read_text())
text = dict(config["text_config"])
old_types = text["layer_types"]
mapping = {}
dummy_indices = []
new_types = []
for old, kind in enumerate(old_types):
    while ("full_attention" if len(new_types) % 3 == 2 else "linear_attention") != kind:
        if len(new_types) % 3 == 2:
            raise RuntimeError("This padding scheme only inserts linear identity layers")
        dummy_indices.append(len(new_types))
        new_types.append("linear_attention")
    mapping[old] = len(new_types)
    new_types.append(kind)
text.update(num_hidden_layers=len(new_types), layer_types=new_types, full_attention_interval=3)
new_config = Qwen3_5TextConfig(**text)

# Check a real identity layer, including the recurrent attention implementation.
identity = Qwen3_5DecoderLayer(new_config, dummy_indices[0]).float().eval()
with torch.no_grad():
    for name, parameter in identity.named_parameters():
        parameter.fill_(1 if name.endswith("norm.weight") else 0)
    hidden = torch.randn(1, 7, text["hidden_size"])
    positions = torch.arange(7).view(1, -1)
    cos = torch.ones(1, 7, text["head_dim"])
    sin = torch.zeros_like(cos)
    actual = identity(hidden, position_embeddings=(cos, sin), attention_mask=None,
                      position_ids=positions, past_key_values=None, use_cache=False)
    if isinstance(actual, tuple):
        actual = actual[0]
    assert torch.equal(hidden, actual), (hidden - actual).abs().max().item()
    template = {name: tensor.to(torch.bfloat16) for name, tensor in identity.state_dict().items()}

weights = load_file(source / "model-00001-of-00001.safetensors")
new_weights = {}
for name, tensor in weights.items():
    if name.startswith("language_model.layers."):
        components = name.split(".")
        components[2] = str(mapping[int(components[2])])
        name = ".".join(components)
    new_weights[name] = tensor
for index in dummy_indices:
    for name, tensor in template.items():
        new_weights[f"language_model.layers.{index}.{name}"] = tensor.clone()
filename = "model-00001-of-00001.safetensors"
save_file(new_weights, dest / filename, metadata={"format": "pt"})
config["text_config"] = text
(dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")
for item in source.iterdir():
    if item.name in {"config.json", "MODEL_MANIFEST.json", "model.safetensors.index.json", filename}:
        continue
    if item.is_file() and not item.name.startswith("model-"):
        shutil.copyfile(item, dest / item.name)
index = {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in new_weights.values())},
         "weight_map": {name: filename for name in new_weights}}
(dest / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
report = {"original_layers": len(old_types), "padded_layers": len(new_types),
          "full_attention_interval": 3, "mapping": mapping, "identity_layers": dummy_indices,
          "identity_forward_exact": True, "original_weights_preserved": True,
          "attention_mode": "original noncausal requirement still needs RKLLM support"}
(dest / "padding-manifest.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
