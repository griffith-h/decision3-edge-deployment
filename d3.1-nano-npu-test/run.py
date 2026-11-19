"""Benchmark one short request and collect separate hardware activity evidence."""

import json
import math
import os
from contextlib import ExitStack
from pathlib import Path
import re
import threading
import time

import torch
import transformers
from cpu_probe import memory
from adapter import ShortTextNPU
from d3_runtime import D3


def sample_npu(stop, samples):
    while not stop.is_set():
        text = Path("/rknpu-load").read_text()
        loads = dict((core, int(load)) for core, load in re.findall(r"Core(\d+):\s*(\d+)%", text))
        samples.append({"monotonic": time.monotonic(), "core_percent": loads})
        stop.wait(0.1)


def main():
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    request = json.loads(Path("/bench/request.json").read_text())
    inputs = {k: request[k] for k in ("state", "questions")}
    dest = Path("/out/measurements.json")
    data = {
        "input": inputs, "calls": [], "torch": torch.__version__, "transformers": transformers.__version__,
        "cpus": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(), "batch_size": 1,
    }
    def save():
        dest.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    start = time.perf_counter()
    model = D3.from_pretrained("/checkpoint", device="cpu", batch_size=1, verify="fast")
    loaded = time.perf_counter()
    data["model_load_seconds"] = loaded - start
    data["attention_mode"] = model.attention_mode
    data["state_once_class"] = type(model.state_once).__name__
    owner = ShortTextNPU(model)
    stack = ExitStack()
    stack.callback(owner.close)
    cpu_counters = {}
    try:
        if os.environ.get("D3_STATE_ONCE") == "0":
            from chunk_blas2 import install as install_scan
            from conv_btc import install as install_conv
            from small_linear import install as install_small

            if model.attention_mode != "prefix_bidir" or model.state_once is not None:
                raise RuntimeError("Expected nano's one-pass prefix-bidirectional path")
            for label, install, args in (("conv", install_conv, ()), ("scan", install_scan, (32, 64)),
                                         ("small_linears", install_small, (model, False))):
                restore, cpu_counters[label] = install(*args)
                stack.callback(restore)
        data["adapter_setup_seconds"] = time.perf_counter() - loaded
        data["installed"] = {"mlps": owner.mlps, "attention_projections": owner.projections}
        print("Installed", data["installed"], "attention", data["attention_mode"], flush=True)
        for index in range(4):
            counts = owner.counts.copy()
            rows = owner.rows.copy()
            cpu_before = {name: counts.copy() for name, counts in cpu_counters.items()}
            start = time.perf_counter()
            response = model.system_one(**inputs)
            seconds = time.perf_counter() - start
            delta = dict(owner.counts - counts)
            memory_state = memory()
            probabilities = response["answers"]["kind"]["probabilities"]
            if (set(probabilities) != set(inputs["questions"]["kind"]["criteria"])
                    or not all(0 <= p <= 1 for p in probabilities.values())
                    or not math.isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)
                    or response["usage"] != {"input_tokens": 115, "output_tokens": 0}
                    or delta.get("successful_npu_matmuls", 0) == 0):
                raise RuntimeError(f"Unexpected result or no successful NPU execution: {response}, {delta}")
            cpu_delta = {name: {k: v - cpu_before[name].get(k, 0) for k, v in counts.items()}
                         for name, counts in cpu_counters.items()}
            data["calls"].append({"phase": "first" if index == 0 else "warm", "seconds": seconds,
                                 "response": response, "counters": delta, "rows": dict(owner.rows - rows),
                                 "cpu_kernels": cpu_delta,
                                 "resident_handles": len(owner.kernels.handles), "memory": memory_state})
            save()
            print("call", index + 1, seconds, probabilities, delta, flush=True)

        samples = []
        stop = threading.Event()
        poll = threading.Thread(target=sample_npu, args=(stop, samples))
        poll.start()
        counts = owner.counts.copy()
        try:
            proof_response = model.system_one(**inputs)
        finally:
            stop.set()
            poll.join()
        data["hardware_proof"] = {"samples": samples, "counters": dict(owner.counts - counts),
                                  "response": proof_response}
        if not any(any(s["core_percent"].values()) for s in samples):
            raise RuntimeError("No nonzero NPU load sampled")
    finally:
        stack.close()
        data["handles_after_close"] = len(owner.kernels.handles)
        save()


if __name__ == "__main__":
    main()
