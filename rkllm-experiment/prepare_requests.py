#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from transformers import AutoTokenizer

root = Path("/work")
sys.path.insert(0, str(root / "source-model"))
from d3_format import render, options
tokenizer = AutoTokenizer.from_pretrained(root / "source-model")
codes = json.loads((root / "source-model/decision_config.json").read_text())["codes"]
cases = json.loads((root / "request-cases.json").read_text())
inputs = root / "inputs"
inputs.mkdir(exist_ok=True)
records = []
for case in cases:
    for name, question in case["questions"].items():
        rendered = dict(question)
        if question["type"] == "score":
            rendered = {"type": "choice", "instructions": question["instructions"],
                        "criteria": {text: None for text in question["criteria"]}}
        prompt = render(tokenizer, case["state"], rendered, codes)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tokens) >= 512:
            raise RuntimeError(f"Input exceeds experimental context: {len(tokens)}")
        stem = f"{case['id']}-{name}"
        (inputs / f"{stem}.tokens").write_text(" ".join(map(str, tokens)) + "\n")
        keys, descriptions = options(rendered)
        records.append({"id": stem, "token_count": len(tokens), "question": question,
                        "option_keys": keys, "state": case["state"], "prompt": prompt,
                        "baseline": case.get("baseline", {}).get(name),
                        "token_file": f"inputs/{stem}.tokens", "hidden_file": f"output/{stem}.f32"})
(root / "inputs/requests.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
print(json.dumps([{k: r[k] for k in ["id", "token_count"]} for r in records], indent=2))
