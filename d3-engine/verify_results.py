"""Recompute the archived D3 probability comparisons using only the Python standard library."""

import json
import subprocess
import sys
from pathlib import Path


ENGINE = Path(__file__).resolve().parent
RESULTS = ENGINE / "results"
CPU_CONTROLS = ("--also-cpu", "router-cpu.json", "--also-cpu", "workflow-cpu.json")
CUSTOMER = "typesafe-customer-service-access-2fa-old-number-flight-t0-triage-a990c4f9e3b5"
INVOICE = "typesafe-invoice-processing-ap_00000-semantic-c99c14de6091"

# Exit 1 is the comparator's recorded numerical-gate failure for these runs.
COMPARISONS = (
    ("cpu-comparison.json", 1, ("cpu.json", "rknn-fp32.json")),
    ("gpu-comparison.json", 0, ("--published-jev", "published-jev.json", "jev-double.json")),
    ("router-comparison.json", 0, ("router-cpu.json", "router-npu.json")),
    ("workflow-cpu-comparison.json", 0, ("--request", CUSTOMER, "workflow-cpu.json", "workflow-npu.json")),
    ("checkpoint-math-comparison.json", 0, ("cpu.json", "checkpoint-math.json", *CPU_CONTROLS)),
    ("eight-core/suite8-cpu.json", 1, (
        "cpu.json", "eight-core/suite8.json", *CPU_CONTROLS, "--compare-cpu-schedules",
    )),
    ("eight-core/long8-vs-previous.json", 1, (
        "eight-core/reference.json", "eight-core/long8.json", "--engine-request", INVOICE,
    )),
)


def main():
    for report, expected_exit, args in COMPARISONS:
        run = subprocess.run(
            [sys.executable, "-B", str(ENGINE / "compare.py"), *args],
            cwd=RESULTS, stdout=subprocess.PIPE, text=True,
        )
        if run.returncode != expected_exit:
            raise SystemExit(f"{report}: comparator exited {run.returncode}, expected {expected_exit}")
        if json.loads(run.stdout) != json.loads((RESULTS / report).read_text()):
            raise SystemExit(f"{report}: recomputed values differ from the archive")
        print(f"{report}: matches archive (comparator exit {expected_exit})")


if __name__ == "__main__":
    main()
