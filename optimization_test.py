#!/usr/bin/env python3
"""Compare optimized service with recorded baseline requests and probabilities."""
import argparse
import json
from pathlib import Path
from smoke_test import request

folder = Path(__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://192.168.1.164:8899")
url = parser.parse_args().url
questions = {
    "kind": {"type": "choice", "instructions": "这个请求属于哪类任务？",
             "criteria": {"code": "编写、解释或修复程序", "math": "数学计算或证明",
                          "chat": "日常聊天和生活话题"}},
    "programming": {"type": "noul", "instructions": "这个请求是否要求编写程序？"},
    "complexity": {"type": "score", "instructions": "这个请求的复杂程度是多少？",
                   "criteria": ["简单", "中等", "复杂"]},
}


def numeric(value, path=""):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            if key != "confidence":
                result.update(numeric(child, path + "/" + key))
        return result
    if isinstance(value, (int, float)):
        return {path: value}
    return {}


cases = []
for item in json.loads((folder / "smoke-results.json").read_text())["requests"]:
    cases.append(("chinese", {"state": item["state"], "questions": questions}, item))
reference = json.loads((folder / "golden-reference.json").read_text())
for item in json.loads((folder / "golden-results.json").read_text()):
    body = reference[item["kind"]]
    if item["kind"] == "image":
        body["state"] = "Describe the attached image."
    cases.append((item["kind"], body, item))

records = []
for kind, body, baseline in cases:
    body["model"] = "vllm-sr/d3-edge"
    response, seconds = request(url, "/v1/systemone", body)
    actual = response["answers"]
    old = baseline["response"]["answers"]
    a, b = numeric(actual), numeric(old)
    assert set(a) == set(b), (a, b)
    for key in old:
        if "choice" in old[key]:
            assert actual[key]["choice"] == old[key]["choice"], (old, actual)
    max_delta = max(abs(a[key] - b[key]) for key in a)
    speedup = baseline["seconds"] / seconds
    record = {"kind": kind, "baseline_seconds": baseline["seconds"],
              "seconds": seconds, "speedup": speedup,
              "max_probability_or_score_delta": max_delta, "response": response}
    records.append(record)
    (folder / "optimization-results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    assert max_delta <= 0.01, record
    assert speedup >= 2, record
print("PASS: all sample choices unchanged, numeric delta <= 0.01, speedup >= 2x.")
