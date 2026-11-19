"""Measure the released D3 CPU runtime with one short input in a fresh container."""

import time

entry = time.perf_counter()

import argparse
import json
import math
import os
import platform
import resource
import sys
from pathlib import Path


def memory():
    status = Path("/proc/self/status").read_text().splitlines()
    rss = next(line for line in status if line.startswith("VmRSS:"))
    values = {
        "rss_bytes": int(rss.split()[1]) * 1024,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    for name in ("memory.current", "memory.peak", "memory.swap.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            values[name] = int(path.read_text())
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkpoint))

    import torch
    import transformers
    from d3_runtime import D3

    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    request = json.loads(args.request.read_text())
    inputs = {key: request[key] for key in ("state", "questions")}
    options = set(inputs["questions"]["kind"]["criteria"])
    config = json.loads((args.checkpoint / "decision_config.json").read_text())
    data = {
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cpus": sorted(os.sched_getaffinity(0)),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "batch_size": 1,
            "attention_mode": config["attention_mode"],
        },
        "input": inputs,
        "calls": [],
    }
    started = time.perf_counter()
    model = D3.from_pretrained(str(args.checkpoint), device="cpu", batch_size=1, verify="fast")
    loaded = time.perf_counter()
    data["setup_before_load_seconds"] = started - entry
    data["model_load_seconds"] = loaded - started
    data["memory_after_load"] = memory()
    data["runtime"]["device"] = str(model.device)
    print("loaded", data["model_load_seconds"], "seconds", data["memory_after_load"], flush=True)

    for index in range(4):
        start = time.perf_counter()
        response = model.system_one(**inputs)
        end = time.perf_counter()
        usage = memory()
        if index == 0:
            data["entry_to_first_answer_seconds"] = end - entry
        answer = response["answers"]["kind"]
        probabilities = answer["probabilities"]
        if set(probabilities) != options or answer["choice"] not in options:
            raise ValueError(f"Unexpected options in {answer}")
        if not all(0 <= p <= 1 for p in probabilities.values()) or not math.isclose(
            sum(probabilities.values()), 1.0, abs_tol=1e-5
        ):
            raise ValueError(f"Invalid probabilities in {answer}")
        data["calls"].append({
            "phase": "first" if index == 0 else "warm",
            "seconds": end - start,
            "memory": usage,
            "response": response,
        })
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        print("call", index + 1, end - start, "seconds", answer["choice"], usage, flush=True)


if __name__ == "__main__":
    main()
