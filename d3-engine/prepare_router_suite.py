"""Prepare a fixed Chinese panel from vLLM Semantic Router's public fresh evaluation files."""

import argparse
import hashlib
import json
import math
import runpy
import urllib.request
from pathlib import Path


REVISION = "fa08b2a642df30955ad2ad2206d74050c7f12b5c"
BASE = f"https://huggingface.co/datasets/vllm-sr/router-signal-suite/resolve/{REVISION}/"
SPEC = "builder/tasks.py"
FILES = {
    SPEC: "e357c30dc32fef548a1e61699500abc31274e17f04f037f0412af2e7dff78a01",
    "text/acebench/tool_need/fresh-acebench-zh.jsonl": "9108827155bac735053d1e595035047786ed24693c38a231e2f99425cb05deb9",
    "text/crosswoz/feedback/fresh-crosswoz.jsonl": "8665ec50d630080aec208b9792448feb0c98b00c5d8b94e231490f0cd0802811",
}


def checked_file(cache, path):
    file = cache / path
    if not file.is_file():
        with urllib.request.urlopen(BASE + path, timeout=60) as source:
            data = source.read()
        if hashlib.sha256(data).hexdigest() != FILES[path]:
            raise ValueError(f"download hash mismatch for {path}")
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
    if hashlib.sha256(file.read_bytes()).hexdigest() != FILES[path]:
        raise ValueError(f"cached hash mismatch for {path}")
    return file


def select_rows(rows):
    # Nearest-rank P50/P90 by the suite's token count, within each source and gold class.
    for label in sorted({r["label"] for r in rows}):
        group = sorted((r for r in rows if r["label"] == label), key=lambda r: (r["tokens"], r["id"]))
        for percentile in (50, 90):
            yield percentile, group[math.ceil(percentile * len(group) / 100) - 1]


def make_request(task, spec, row, source, percentile):
    state = {"request": row["text"]}
    if "tools" in row:
        state["tools"] = row["tools"]
    if spec["kind"] == "noul":
        criteria = {key: spec[key] for key in ("false", "true")}
        gold = row["label"] == spec["positive"]
    else:
        criteria = {label: spec["options"][label] for label in spec["labels"]}
        gold = row["label"]
    questions = {task: {"type": spec["kind"], "instructions": spec["question"], "criteria": criteria}}
    wire = json.dumps({"state": state, "questions": questions}, ensure_ascii=False, separators=(",", ":")).encode()
    return {"id": row["id"], "state": state, "questions": questions, "expected": {task: gold},
            "_evaluation": {"source": source, "revision": REVISION, "source_label": row["label"],
                            "source_tokens": row["tokens"], "within_label_percentile": percentile,
                            "payload_sha256": hashlib.sha256(wire).hexdigest()}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("cache", type=Path)
    cli.add_argument("destination", type=Path)
    args = cli.parse_args()
    # The pinned module defines the authors' label order, question text and criteria.
    specs = runpy.run_path(str(checked_file(args.cache, SPEC)))["TASKS"]
    args.destination.mkdir(parents=True, exist_ok=True)
    for path in FILES:
        if path == SPEC:
            continue
        task = Path(path).parent.name
        rows = [json.loads(line) for line in checked_file(args.cache, path).read_text().splitlines()]
        for percentile, row in select_rows(rows):
            case = make_request(task, specs[task], row, path, percentile)
            name = args.destination / (case["id"] + ".json")
            name.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
            print(name, case["_evaluation"]["payload_sha256"])


if __name__ == "__main__":
    main()
