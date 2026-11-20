"""Capture a real long prefix and replay the same complete tensors through ARM and RKNN."""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import torch


@contextmanager
def capture(branches, directory):
    directory.mkdir(parents=True, exist_ok=True)
    original = branches.segment
    seen = set()

    def record(q, kv, scale):
        k, v = kv
        if k.shape[2] >= 8192 and q.shape[2] >= 16:
            name = "state" if q.shape[0] == 1 and q.shape[2] == k.shape[2] else "questions"
            if name not in seen:
                path = directory / (name + ".pt")
                if not path.exists():
                    torch.save({"q": q.bfloat16().contiguous().clone(), "k": k.bfloat16().contiguous().clone(),
                                "v": v.bfloat16().contiguous().clone(), "scale": scale}, path)
                seen.add(name)
        return original(q, kv, scale)

    branches.segment = record
    try:
        yield
    finally:
        del branches.segment


@torch.inference_mode()
def main():
    from fast_prefix import cpu_segment
    from prefix_rknn import RknnSegments

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    torch.set_num_threads(int(os.environ.get("D3_THREADS", 8)))
    torch.set_num_interop_threads(1)
    reports = []
    with ThreadPoolExecutor(2) as pool:
        chip = RknnSegments(pool)
        try:
            for name in ("state", "questions"):
                path = args.directory / (name + ".pt")
                data = torch.load(path, map_location="cpu", weights_only=True)
                q, k, v, scale = (data[key] for key in ("q", "k", "v", "scale"))
                ref = None
                trials = []
                for backend in ("arm", "rknn", "rknn", "arm"):
                    start = time.perf_counter()
                    out, lse = (chip if backend == "rknn" else cpu_segment)(q, k, v, scale)
                    elapsed = time.perf_counter() - start
                    if ref is None:
                        ref = out, lse
                    trials.append({"backend": backend, "seconds": elapsed,
                                   "max_output_delta": (ref[0] - out).abs().max().item(),
                                   "max_lse_delta": (ref[1] - lse).abs().max().item()})
                    print(name, backend, round(elapsed, 3), flush=True)
                reports.append({"segment": name, "q_shape": list(q.shape), "kv_shape": list(k.shape),
                                "capture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "trials": trials})
                args.output.write_text(json.dumps(reports, indent=2) + "\n")
        finally:
            chip.close()


if __name__ == "__main__":
    main()
