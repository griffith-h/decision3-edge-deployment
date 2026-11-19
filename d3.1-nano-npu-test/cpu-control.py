"""Independent one-pass probability control with the same three edge CPU kernels."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import time

import torch
from chunk_blas2 import install as install_scan
from conv_btc import install as install_conv
from cpu_probe import memory
from d3_runtime import D3
from small_linear import install as install_small


def main():
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    if Path("/dev/dri").exists():
        raise RuntimeError("Unexpected accelerator device in CPU control")
    model = D3.from_pretrained("/checkpoint", device="cpu", batch_size=1, verify="fast")
    if model.attention_mode != "prefix_bidir" or model.state_once is not None:
        raise RuntimeError("Expected the one-pass prefix-bidirectional runtime")
    request = json.loads(Path("/bench/request.json").read_text())
    inputs = {k: request[k] for k in ("state", "questions")}
    with ExitStack() as stack:
        counts = {}
        for label, install, args in (("conv", install_conv, ()), ("scan", install_scan, (32, 64)),
                                     ("small_linears", install_small, (model, False))):
            restore, counts[label] = install(*args)
            stack.callback(restore)
        start = time.perf_counter()
        response = model.system_one(**inputs)
        elapsed = time.perf_counter() - start
        result = {"input": inputs, "response": response, "seconds": elapsed, "cpu_kernels": counts,
                  "cpus": sorted(os.sched_getaffinity(0)), "threads": torch.get_num_threads(),
                  "attention_mode": model.attention_mode, "memory": memory(), "dri_present": False}
        Path("/out/cpu-control.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print("CPU control", elapsed, response, counts, flush=True)


if __name__ == "__main__":
    main()
