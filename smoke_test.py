#!/usr/bin/env python3
"""Check the deployed native Decision API from a LAN client."""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=payload,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=300) as response:
            value = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode()}") from error
    return value, round(time.monotonic() - started, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://192.168.1.164:8899")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("smoke-results.json"))
    args = parser.parse_args()
    health, _ = request(args.url, "/health")
    assert health["status"] == "ready", health
    discovery, seconds = request(args.url, "/v1/models")
    print("Discovery:", json.dumps(discovery, ensure_ascii=False), flush=True)
    model = "vllm-sr/d3-edge"
    if model not in json.dumps(discovery):
        raise RuntimeError("d3-edge is absent from native model discovery")
    record = {"url": args.url, "health": health, "discovery": discovery,
              "discovery_seconds": seconds, "requests": []}
    questions = {
        "kind": {"type": "choice", "instructions": "这个请求属于哪类任务？",
                 "criteria": {"code": "编写、解释或修复程序", "math": "数学计算或证明",
                              "chat": "日常聊天和生活话题"}},
        "programming": {"type": "noul", "instructions": "这个请求是否要求编写程序？"},
        "complexity": {"type": "score", "instructions": "这个请求的复杂程度是多少？",
                       "criteria": ["简单", "中等", "复杂"]},
    }
    states = ["请写一个 Python 函数，把两个有序列表合并成一个有序列表。",
              "你好，今天心情怎么样？"]
    for state in states:
        value, seconds = request(args.url, "/v1/systemone", {
            "model": model, "state": state, "questions": questions,
        })
        answers = value["answers"]
        probabilities = answers["kind"]["probabilities"]
        assert set(probabilities) == {"code", "math", "chat"}, value
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in probabilities.values()), value
        assert abs(sum(probabilities.values()) - 1) < 0.001, value
        assert 0 <= answers["programming"]["noul"] <= 1, value
        assert answers["complexity"]["type"] == "score", value
        record["requests"].append({"state": state, "seconds": seconds, "response": value})
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(record["requests"][-1], ensure_ascii=False, indent=2), flush=True)
    assert record["requests"][0]["response"]["answers"]["kind"]["choice"] == "code"
    assert record["requests"][1]["response"]["answers"]["kind"]["choice"] == "chat"
    print("PASS: LAN discovery, choice/noul/score response structure and sample classification.")


if __name__ == "__main__":
    main()
