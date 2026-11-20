"""Replay captured query rows concurrently on ARM and the two RKNN heads."""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from fast_prefix import cpu_segment, split_segment
from prefix_rknn import RknnSegments


@torch.inference_mode()
def replay(captures, output, threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    results = []
    with ThreadPoolExecutor(2) as npu_pool, ThreadPoolExecutor(1) as cpu_pool:
        chip = RknnSegments(npu_pool)
        try:
            for name in ("questions", "state"):
                data = torch.load(captures / (name + ".pt"), map_location="cpu", weights_only=True)
                q, k, v, scale = (data[key] for key in ("q", "k", "v", "scale"))
                reference = cpu_segment(q, k, v, scale)
                for fraction in (0, .25, .375, .5, .5, .375, .25, 0):
                    start, cpu_start = time.perf_counter(), time.process_time()
                    if fraction:
                        rows = min(q.shape[2] - 16, max(1, round(q.shape[2] * fraction)))
                        if rows < 1:
                            raise ValueError("capture needs at least 17 query positions")
                        got = split_segment(q, k, v, scale, chip, cpu_pool, rows)
                    else:
                        got = chip(q, k, v, scale)
                    seconds = time.perf_counter() - start
                    cpu_seconds = time.process_time() - cpu_start
                    if got is None:
                        raise ValueError(f"{name}: RKNN could not process the captured values")
                    row = {
                        "segment": name,
                        "arm_fraction": fraction,
                        "seconds": seconds,
                        "average_cores": cpu_seconds / seconds,
                        "max_output_delta": (got[0] - reference[0]).abs().max().item(),
                        "max_lse_delta": (got[1] - reference[1]).abs().max().item(),
                    }
                    results.append(row)
                    output.write_text(json.dumps(results, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
        finally:
            chip.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captures", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=int(os.environ.get("D3_THREADS", 8)))
    args = parser.parse_args()
    replay(args.captures, args.output, args.threads)
