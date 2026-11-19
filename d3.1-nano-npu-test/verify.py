"""Recompute the timing and probability comparisons from the recorded board results."""

import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent


def options(response):
    assert response["usage"] == {"input_tokens": 115, "output_tokens": 0}
    answer = response["answers"]["kind"]
    p = answer["probabilities"]
    assert set(p) == {"code", "math", "chat"}
    assert all(0 <= value <= 1 for value in p.values())
    assert math.isclose(sum(p.values()), 1, abs_tol=1e-5)
    assert answer["choice"] == max(p, key=p.get)
    return p


def compare(data, baseline):
    cpu = baseline["models"]["nano"]
    cpu_calls = cpu["benchmark"]["calls"]
    original = options(cpu_calls[0]["response"])
    assert all(options(call["response"]) == original for call in cpu_calls)
    cpu_warm = statistics.median(call["seconds"] for call in cpu_calls if call["phase"] == "warm")
    audit = data["board_audit"]
    assert audit["model_sha256"] == baseline["nano_download"]["model_sha256"]
    assert audit["request_sha256"] == baseline["final_state"]["board_request_sha256"]
    assert audit["edge_state_before"] == audit["edge_state_after"] == "exited"
    assert audit["test_containers_removed"]
    for name, inspection in audit["containers"].items():
        assert inspection["image"] == cpu["container"]["image"]
        assert inspection["exit_code"] == 0 and not inspection["oom_killed"]
        assert inspection["cpuset"] == "4-7" and inspection["network"] == "none"
        assert inspection["memory_limit"] == inspection["memory_and_swap_limit"] == 8 * 2**30
        assert bool(inspection["devices"]) == (name != "control")

    control = data["control"]
    assert not control["dri_present"] and control["input"] == cpu["benchmark"]["input"]
    matched = options(control["response"])
    summary = {"original_cpu_warm_median_s": cpu_warm, "original_cpu_probabilities": original,
               "one_pass_cpu_control_probabilities": matched}
    for name, expected_runs in (("default", 300), ("single", 150)):
        record = data[name]
        assert record["input"] == control["input"] and record["attention_mode"] == "prefix_bidir"
        assert record["cpus"] == [4, 5, 6, 7] and record["torch_threads"] == 3 and record["batch_size"] == 1
        assert record["handles_after_close"] == 0
        calls = record["calls"]
        assert [call["phase"] for call in calls] == ["first", "warm", "warm", "warm"]
        proof = record["hardware_proof"]
        peak = {core: max(s["core_percent"][core] for s in proof["samples"]) for core in ("0", "1")}
        assert min(peak.values()) > 0
        p = options(proof["response"])
        for call in [*calls, proof]:
            assert options(call["response"]) == p
            assert call["counters"]["successful_npu_matmuls"] == expected_runs
            assert not (call["counters"].get("cpu_guard") or call["counters"].get("cpu_range"))
        if name == "single":
            assert all(call["cpu_kernels"] == control["cpu_kernels"] for call in calls)
        seconds = [c["seconds"] for c in calls[1:]]
        median = statistics.median(seconds)
        delta = {k: p[k] - original[k] for k in p}
        matched_cpu = original if name == "default" else matched
        summary[name] = {
            "first_s_including_lazy_weight_packing": calls[0]["seconds"], "warm_s": seconds,
            "warm_median_s": median, "speedup_vs_original_cpu": cpu_warm / median,
            "probabilities": p, "delta_vs_original_cpu": delta,
            "max_abs_delta_vs_original_cpu": max(map(abs, delta.values())),
            "delta_vs_matched_cpu": {k: p[k] - matched_cpu[k] for k in p},
            "npu_matmuls_per_request": expected_runs, "sampled_npu_peak_percent": peak,
            "peak_rss_bytes": max(c["memory"]["peak_rss_bytes"] for c in calls),
        }
    return summary


if __name__ == "__main__":
    evidence = json.loads((HERE / "results.json").read_text())
    baseline = json.loads((HERE.parent / "d3.1-nano-cpu-test/results.json").read_text())
    summary = compare(evidence, baseline)
    assert summary == evidence["summary"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
