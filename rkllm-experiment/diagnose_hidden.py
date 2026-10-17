#!/usr/bin/env python3
"""Compare original/padded CPU semantics and NPU hidden-state interpretation."""
import json
import gc
from pathlib import Path
import torch
from rkllm.api import RKLLM
from safetensors.torch import load_file

torch.set_num_threads(4)
root = Path("/work")
records = json.loads((root / "inputs/requests.json").read_text())
case = next(r for r in records if r["id"] == "single-kind")
ids = torch.tensor([[int(x) for x in (root / case["token_file"]).read_text().split()]])
readout = load_file(root / "source-model/readout.safetensors")["weight"].float()
npu = torch.from_numpy(__import__("numpy").fromfile(root / case["hidden_file"], dtype="float32"))

def summarize(vector):
    logits = readout[:3] @ vector.float()
    return {"norm": vector.float().norm().item(), "min": vector.min().item(),
            "max": vector.max().item(), "logits": logits.tolist(),
            "probabilities": torch.softmax(logits, dim=-1).tolist()}

outputs = {"npu": summarize(npu), "cpu_service": case["baseline"]}
vectors = {}
for label in ("source-model", "prepared-model"):
    llm = RKLLM()
    assert llm.load_huggingface(model=str(root / label), device="cpu", dtype="float32") == 0
    core = llm.base.model.model
    core.eval().requires_grad_(False)
    with torch.inference_mode():
        causal = core(input_ids=ids, use_cache=False).last_hidden_state[:, -1][0]
        vectors[label + "/causal"] = causal.clone()
        outputs[label + "/causal"] = summarize(causal)
        present = torch.ones_like(ids)
        embeds = core.embed_tokens(ids)
        ramp = torch.arange(ids.shape[1]).view(1, 1, -1).expand(4, 1, -1)
        rope = core.rotary_emb(embeds, ramp[1:])
        keys = present.bool()[:, None, None, :]
        hidden = embeds
        for layer, kind in zip(core.layers, core.config.layer_types):
            hidden = layer(hidden, position_embeddings=rope,
                           attention_mask=keys if kind == "full_attention" else None,
                           position_ids=ramp[0], past_key_values=None, use_cache=False)
            if isinstance(hidden, tuple):
                hidden = hidden[0]
        noncausal = core.norm(hidden)[:, -1][0]
        vectors[label + "/noncausal"] = noncausal.clone()
        outputs[label + "/noncausal"] = summarize(noncausal)
        print(label, json.dumps({k:v for k,v in outputs.items() if k.startswith(label)},indent=2),flush=True)
    del llm, core, hidden, embeds
    gc.collect()
for mode in ("causal", "noncausal"):
    original, padded = vectors["source-model/" + mode], vectors["prepared-model/" + mode]
    outputs["padding_" + mode + "_max_hidden_difference"] = (original - padded).abs().max().item()
    outputs["npu_cosine_" + mode] = torch.nn.functional.cosine_similarity(npu, padded, dim=0).item()
(root / "logs/hidden-diagnosis.json").write_text(json.dumps(outputs,indent=2)+'\n')
print(json.dumps(outputs,indent=2))
