#!/usr/bin/env python3
"""Measure differences from the router's published CPU reference without hiding them."""
import json
from pathlib import Path
from smoke_test import request


def numbers(value, path=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(numbers(child, f"{path}/{key}"))
        return result
    if isinstance(value, (int, float)):
        return {path: value}
    return {}


folder = Path(__file__).parent
reference = json.loads((folder / "golden-reference.json").read_text())
expected = reference["reference"]["answers"]["cpu"]
records = []
for kind in ("text", "image"):
    body = reference[kind]
    if kind == "image":
        body["state"] = "Describe the attached image."
    body["model"] = "vllm-sr/d3-edge"
    response, seconds = request("http://192.168.1.164:8899", "/v1/systemone", body)
    actual = response["answers"]
    expected_subset = {key: expected[key] for key in actual}
    a, b = numbers(actual), numbers(expected_subset)
    assert set(a) == set(b), (a, b)
    deltas = {key: abs(a[key] - b[key]) for key in a}
    record = {"kind": kind, "seconds": seconds, "response": response,
              "expected_cpu": expected_subset,
              "max_absolute_difference": max(deltas.values()), "differences": deltas}
    records.append(record)
    (folder / "golden-results.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
print("CPU golden tolerance 0.001:",
      "PASS" if all(x["max_absolute_difference"] <= 0.001 for x in records) else "DIFFERS")
