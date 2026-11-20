"""Read one D3 text request per JSON line and write its answer using a resident RK3576 model."""

import argparse
import json
import os
import sys

def serve_lines(engine, inputs, outputs):
    for line in inputs:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or set(request) != {"state", "questions"}:
                raise ValueError("expected an object with state and questions")
            response = engine.system_one(**request)
        except (ValueError, TypeError) as error:
            response = {"error": "invalid_request", "message": str(error)}
        print(json.dumps(response, ensure_ascii=False, allow_nan=False), file=outputs, flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--question-rows", type=int, default=4)
    precision = cli.add_mutually_exclusive_group()
    precision.add_argument("--attention", choices=("auto", "checkpoint", "fp32"), default="auto")
    precision.add_argument("--checkpoint-math", action="store_true",
                           help="keep checkpoint attention and state math, with RKNN projection and MLP")
    cli.add_argument("--permutation-average", action="store_true")
    cli.add_argument("--aggressive", action="store_true")
    cli.add_argument("--prefix-backend", choices=("arm", "rknn"))
    args = cli.parse_args()
    # Keep native SDK diagnostics off the machine-readable output channel.
    with os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8") as responses:
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        import torch
        from d3_runtime import D3
        from engine import D3Engine

        torch.set_num_threads(int(os.environ.get("D3_THREADS", 3)))
        torch.set_num_interop_threads(1)
        model = D3.from_pretrained(os.environ["D3_CHECKPOINT"], device="cpu", batch_size=1,
                                   permutation_average=args.permutation_average, verify="fast")
        attention = "checkpoint" if args.checkpoint_math else args.attention
        with D3Engine(model, question_rows=args.question_rows, attention=attention,
                      checkpoint_state_kernels=args.checkpoint_math, aggressive=args.aggressive,
                      prefix_backend=args.prefix_backend) as engine:
            serve_lines(engine, sys.stdin, responses)


if __name__ == "__main__":
    main()
