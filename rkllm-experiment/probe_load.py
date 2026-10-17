#!/usr/bin/env python3
"""Probe official RKLLM loading; keep the original checkpoint read-only."""
import argparse
import json
import time
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="/work/source-model")
parser.add_argument("--report", default="/work/logs/load-original.json")
parser.add_argument("--no-weight", action="store_true")
args = parser.parse_args()
report = {"model": args.model, "load_weight": not args.no_weight}
started = time.monotonic()
try:
    from rkllm.api import RKLLM
    llm = RKLLM()
    print("Loading model through official RKLLM loader", flush=True)
    result = llm.load_huggingface(model=args.model, device="cpu", dtype="float32",
                                  load_weight=not args.no_weight)
    report["load_return_code"] = result
    report["base_attributes"] = [x for x in dir(llm.base) if not x.startswith("__")]
except Exception as error:
    report["exception"] = repr(error)
    traceback.print_exc()
finally:
    report["seconds"] = time.monotonic() - started
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
if report.get("load_return_code") != 0:
    raise SystemExit(1)
