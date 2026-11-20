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
    ("long-prefix/four-vs-eight.json", 0, (
        "long-prefix/four.json", "long-prefix/eight.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/hybrid-vs-eight.json", 0, (
        "long-prefix/eight.json", "long-prefix/hybrid.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/hybrid-vs-previous.json", 1, (
        "eight-core/reference.json", "long-prefix/hybrid.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/combined-vs-hybrid.json", 0, (
        "long-prefix/hybrid.json", "long-prefix/combined.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/four-suite-cpu.json", 1, (
        "cpu.json", "long-prefix/four-suite.json", *CPU_CONTROLS, "--compare-cpu-schedules",
    )),
    ("long-prefix/padded-vs-combined.json", 0, (
        "long-prefix/combined.json", "long-prefix/padded.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/padded-vs-eight.json", 0, (
        "long-prefix/eight.json", "long-prefix/padded.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/shapes-vs-padded.json", 1, (
        "long-prefix/padded.json", "long-prefix/shapes.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/shapes-vs-eight.json", 1, (
        "long-prefix/eight.json", "long-prefix/shapes.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/shapes-vs-previous.json", 0, (
        "eight-core/reference.json", "long-prefix/shapes.json", "--engine-request", INVOICE,
    )),
    ("long-prefix/shapes-suite-cpu.json", 1, (
        "cpu.json", "long-prefix/shapes-suite.json", *CPU_CONTROLS, "--compare-cpu-schedules",
    )),
    ("long-prefix/shapes-suite-cpu-eight-feedback.json", 0, (
        "long-prefix/cpu-eight.json", "long-prefix/shapes-suite.json", "--request", "feedback-fresh-crosswoz-000261",
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
