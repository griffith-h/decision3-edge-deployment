#!/usr/bin/env python3
"""Apply the original decision readout to NPU hidden vectors and compare outputs."""
import json
import re
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

root = Path("/work")
weights = load_file(root / "source-model/readout.safetensors")["weight"]
records = json.loads((root / "inputs/requests.json").read_text())
timings = {}
for line in (root / "logs/npu-run.log").read_text(errors="replace").splitlines():
    if line.startswith("RESULT "):
        data = json.loads(line[7:])
        timings[Path(data["input"]).stem] = data
results = []
for record in records:
    p = root / record["hidden_file"]
    if not p.exists():
        results.append({"id": record["id"], "missing_hidden_output": True})
        continue
    hidden = np.fromfile(p, dtype=np.float32)
    assert hidden.shape == (1024,) and np.isfinite(hidden).all(), (p, hidden.shape)
    logits = (weights @ hidden)[:len(record["option_keys"])]
    exp = np.exp((logits - logits.max()).astype(np.float64))
    probs = exp / exp.sum()
    question = record["question"]
    baseline = record["baseline"]
    kind = question["type"]
    result = {"id": record["id"], "kind": kind,
              "timing": timings.get(record["id"]), "logits": logits.tolist()}
    if kind == "noul":
        value = float(probs[1])
        result.update(npu_noul=value, baseline_noul=baseline["noul"],
                      max_probability_delta=abs(value - baseline["noul"]))
    else:
        keys = list(baseline["probabilities"])
        values = dict(zip(keys, map(float, probs)))
        result.update(npu_probabilities=values, baseline_probabilities=baseline["probabilities"],
                      max_probability_delta=max(abs(values[k] - baseline["probabilities"][k]) for k in keys))
        if kind == "choice":
            choice = keys[int(probs.argmax())]
            result.update(npu_choice=choice, baseline_choice=baseline["choice"],
                          choice_matches=choice == baseline["choice"])
        else:
            result.update(npu_score=float(np.arange(len(probs)) @ probs), baseline_score=baseline["score"])
    results.append(result)
missing = sum(x.get("missing_hidden_output", False) for x in results)
deltas = [x["max_probability_delta"] for x in results if "max_probability_delta" in x]
choices = [x["choice_matches"] for x in results if "choice_matches" in x]
loads = [int(x) for x in re.findall(r"Core\d+:\s+(\d+)%", (root / "logs/npu-load.log").read_text())]
report = {"results": results, "missing": missing,
          "max_probability_delta": max(deltas) if deltas else None,
          "choices_matched": sum(choices), "choices_total": len(choices),
          "max_observed_npu_load_percent": max(loads) if loads else None,
          "cpu_equivalence_tolerance": 0.01,
          "parity_passed": not missing and all(choices) and all(x <= 0.01 for x in deltas)}
(root / "logs/npu-comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
