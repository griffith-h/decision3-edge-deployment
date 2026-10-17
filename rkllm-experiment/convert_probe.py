#!/usr/bin/env python3
"""Isolated RKLLM conversion experiment. Export is not a proof of d3 parity."""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from rkllm.api import RKLLM

parser = argparse.ArgumentParser()
parser.add_argument("--quant", default="w8a8")
parser.add_argument("--max-context", type=int, default=512)
parser.add_argument("--model", default="/work/source-model")
parser.add_argument("--label", default="experimental")
parser.add_argument("--no-quant", action="store_true")
parser.add_argument("--optimization-level", type=int, default=1)
parser.add_argument("--noncausal", action="store_true")
args = parser.parse_args()
root = Path("/work")
out = root / "output"
out.mkdir(exist_ok=True)
report = {"quant": args.quant, "max_context": args.max_context,
          "target": "rk3576", "cores": 2, "attention_parity": "unverified",
          "model": args.model, "do_quantization": not args.no_quant}
started = time.monotonic()
try:
    if args.noncausal:
        # The pinned writer exposes this metadata field, although the high-level
        # conversion API has no attention-mode argument. Inject before serializing
        # metadata and verify on the board rather than assuming runtime support.
        import rkllm.base.converter as converter
        original_write_header = converter.LLMWriter.write_header_to_file
        def noncausal_write_header(writer, *positional, **keywords):
            writer.add_causal_attention(False)
            report["noncausal_metadata_written"] = True
            report["metadata_hook"] = "before header and key-value counts are serialized"
            print("Export metadata: causal_attention=False", flush=True)
            return original_write_header(writer, *positional, **keywords)
        converter.LLMWriter.write_header_to_file = noncausal_write_header
    llm = RKLLM()
    report["load_return_code"] = llm.load_huggingface(
        model=args.model, device="cpu", dtype="float32")
    if report["load_return_code"] != 0:
        raise RuntimeError("RKLLM load failed")
    model = llm.base.model
    report["loaded_class"] = type(model).__name__
    report["loaded_config"] = model.config.to_dict()
    report["weight_keys_preview"] = list(model.state_dict())[:12]
    original = json.loads((Path(args.model) / "config.json").read_text())["text_config"]
    core = model.model
    report["layer_types_preserved"] = core.config.layer_types == original["layer_types"]
    report["attention_layer_properties"] = [
        {"index": i, "class": type(layer).__name__,
         "is_causal": getattr(getattr(layer, "self_attn", None), "is_causal", None)}
        for i, layer in enumerate(core.layers)
    ]
    print(json.dumps({k: v for k, v in report.items() if k != "loaded_config"}, indent=2), flush=True)
    if not report["layer_types_preserved"]:
        raise RuntimeError("RKLLM loader changed the pruned layer layout")

    # Calibrate on the exact native d3 prompt format, without training weights.
    sys.path.insert(0, str(root / "source-model"))
    from d3_format import render
    codes = json.loads((root / "source-model/decision_config.json").read_text())["codes"]
    states = ["Write a Python function that merges two sorted lists.",
              "Solve the equation 2*x + 3 = 7.", "Hello, how are you today?",
              "这段 Python 程序出现了索引越界，请帮忙修复。",
              "求三角形三个内角的和。", "周末想在家做一道简单的菜。"]
    questions = [
        {"type": "choice", "instructions": "What kind of work is this?",
         "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
        {"type": "noul", "instructions": "Does this request need multi-step reasoning?"},
        {"type": "choice", "instructions": "How difficult is this request?",
         "criteria": {"Trivial": None, "Moderate": None, "Hard": None}},
    ]
    data = [{"input": render(llm.base.tokenizer, state, question, codes), "target": "A"}
            for state in states for question in questions]
    dataset = root / "calibration.json"
    dataset.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print("Calibration examples", len(data), flush=True)
    report["build_return_code"] = llm.build(
        do_quantization=not args.no_quant, optimization_level=args.optimization_level,
        quantized_dtype=args.quant, quantized_algorithm="normal",
        target_platform="rk3576", num_npu_core=2, dataset=str(dataset),
        max_context=args.max_context)
    if report["build_return_code"] != 0:
        raise RuntimeError("RKLLM build failed")
    dtype_label = "unquantized" if args.no_quant else args.quant
    artifact = out / f"d3-edge-{dtype_label}-rk3576-{args.label}.rkllm"
    report["export_return_code"] = llm.export_rkllm(str(artifact))
    report["artifact"] = str(artifact)
    report["artifact_bytes"] = artifact.stat().st_size if artifact.exists() else None
    if args.noncausal and not report.get("noncausal_metadata_written"):
        raise RuntimeError("Noncausal metadata hook was not called")
except Exception as error:
    report["exception"] = repr(error)
    traceback.print_exc()
finally:
    report["seconds"] = time.monotonic() - started
    (root / f"logs/convert-{args.label}-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("FINAL REPORT", json.dumps({k: v for k, v in report.items() if k != "loaded_config"}, indent=2), flush=True)
if report.get("export_return_code") != 0:
    raise SystemExit(1)
