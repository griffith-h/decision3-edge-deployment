"""Run complete text requests through the checkpoint API and record every answer."""

import argparse
import hashlib
import json
import os
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import torch
import transformers
from d3_runtime import D3


def peak_memory():
    mem = {"peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}
    mem["open_fds_after_call"] = len(list(Path("/proc/self/fd").iterdir()))
    for name in ("memory.peak", "memory.swap.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            mem[name] = int(path.read_text())
    return mem


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("backend", choices=["cpu", "rknn"])
    cli.add_argument("output", type=Path)
    cli.add_argument("cases", type=Path, nargs="+")
    cli.add_argument("--repeats", type=int, default=1)
    cli.add_argument("--question-rows", type=int, default=4)
    cli.add_argument("--permutation-average", action="store_true",
                     help="score original and reversed Choice options as in the released Decision Index run")
    cli.add_argument("--checkpoint-state-kernels", action="store_true",
                     help="diagnose probability differences with the checkpoint convolution and delta rule")
    cli.add_argument("--attention", choices=("checkpoint", "probe", "fp32"), default="checkpoint",
                     help="probe replays both CPU attentions and returns BF16; fp32 uses the alternative")
    args = cli.parse_args()
    if args.repeats < 1:
        cli.error("repeats must be positive")
    if args.backend == "cpu" and args.attention != "checkpoint":
        cli.error("the pure CPU control always uses the checkpoint attention")
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    checkpoint = Path(os.environ.get("D3_CHECKPOINT", "/checkpoint"))
    model = D3.from_pretrained(str(checkpoint), device="cpu", batch_size=1, verify="fast",
                               permutation_average=args.permutation_average)
    data = {"backend": args.backend, "torch": torch.__version__, "transformers": transformers.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "threads": torch.get_num_threads(),
            "attention": model.attention_mode, "state_once": type(getattr(model, "state_once", None)).__name__,
            "permutation_average": model.permutation_average, "model_load_s": time.perf_counter() - start, "requests": [],
            "checkpoint_manifest_sha256": hashlib.sha256((checkpoint / "MODEL_MANIFEST.json").read_bytes()).hexdigest()}
    with ExitStack() as stack:
        engine = None
        infer = model.system_one
        if args.checkpoint_state_kernels and (args.backend != "rknn" or model.attention_mode != "prefix_bidir"):
            cli.error("the checkpoint state-kernel option needs the prefix_bidir RKNN engine")
        if args.backend == "rknn":
            from engine import D3Engine
            precision = "fp32" if args.attention == "fp32" else "checkpoint"
            engine = stack.enter_context(D3Engine(model, question_rows=args.question_rows, attention=precision,
                                                  checkpoint_state_kernels=args.checkpoint_state_kernels))
            data["max_question_rows"] = args.question_rows
            data["sdpa_precision"] = precision
            if args.checkpoint_state_kernels:
                data["state_kernels"] = "checkpoint"
            infer = engine.system_one
        probe = None
        if args.attention == "probe":
            if not engine or model.attention_mode != "prefix_bidir":
                cli.error("the attention probe needs the RKNN prefix_bidir backend")
            from cpu_attention import profile
            probe = stack.enter_context(profile(engine))
            data["attention_trial"] = args.attention
        for path in args.cases:
            case = json.loads(path.read_text())
            inputs = {key: case[key] for key in ("state", "questions")}
            evaluation = case["_evaluation"]
            # Jev Index hashes the original API payload, including its fixed model identifier.
            payload = {"model": "jev-1.13.0", **inputs} if "catalog_id" in evaluation else inputs
            wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            digest = hashlib.sha256(wire).hexdigest()
            if digest != evaluation["payload_sha256"]:
                raise ValueError(f"changed benchmark payload {case['id']}")
            prep = model.prepare(**inputs)
            if prep.errors:
                raise ValueError(f"invalid benchmark input {prep.errors}")
            row = {"id": case["id"], "payload_sha256": digest,
                   "question_lengths": {k: len(seq) for k, seq in prep.sequences.items()},
                   "state_lengths": getattr(prep, "states", {}), "expected": case["expected"], "calls": []}
            if args.permutation_average:
                row["reversed_question_lengths"] = {k: len(seq) for k, seq in prep.reversed_sequences.items()}
            data["requests"].append(row)
            print("start", row["id"], row["question_lengths"], flush=True)
            for _ in range(args.repeats):
                before = engine.counts.copy() if engine else {}
                if probe is not None:
                    probe.clear()
                start = time.perf_counter()
                answer = infer(**inputs)
                elapsed = time.perf_counter() - start
                if (set(answer["answers"]) != set(inputs["questions"])
                        or any("error" in a for a in answer["answers"].values())):
                    raise ValueError(f"incomplete answers {answer}")
                for key, question in inputs["questions"].items():
                    if (question["type"] == "choice"
                            and set(answer["answers"][key]["probabilities"]) != set(question["criteria"])):
                        raise ValueError(f"incomplete option distribution for {key}")
                    if question["type"] == "score":
                        expected_levels = {str(i) for i in range(len(question["criteria"]))}
                        if set(map(str, answer["answers"][key]["probabilities"])) != expected_levels:
                            raise ValueError(f"incomplete rubric distribution for {key}")
                call = {"seconds": elapsed, "usage": answer["usage"], "answers": answer["answers"], **peak_memory()}
                if engine:
                    call["operations"] = dict(engine.counts - before)
                if probe is not None:
                    call["probe"] = dict(probe)
                row["calls"].append(call)
                tmp = args.output.with_suffix(".tmp")
                tmp.write_text(json.dumps(data, indent=2) + "\n")
                tmp.replace(args.output)
                print("completed", row["id"], elapsed, call.get("operations"), flush=True)


if __name__ == "__main__":
    main()
