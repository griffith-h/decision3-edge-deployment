"""Compare complete D3 text responses. Probability differences use the [0, 1] scale."""

import argparse
import json
import math
import statistics
from pathlib import Path


RUNTIME_KEYS = ("attention", "state_once", "threads", "torch", "transformers", "cpu_affinity", "checkpoint_manifest_sha256")
REQUEST_KEYS = ("payload_sha256", "question_lengths", "state_lengths", "reversed_question_lengths", "expected")


def combine_cpu_controls(first, others):
    combined = {**first, "requests": list(first["requests"])}
    for run in others:
        if (run["backend"] != "cpu" or first["backend"] != "cpu"
                or run.get("permutation_average", False) != first.get("permutation_average", False)
                or any(run[key] != first[key] for key in RUNTIME_KEYS)):
            raise ValueError("cannot combine CPU controls from different runtimes or checkpoints")
        combined["requests"].extend(run["requests"])
    return combined


def distribution(answer):
    if answer["type"] == "noul":
        yes = answer["noul"]
        if not 0 <= yes <= 1:
            raise ValueError(f"invalid noul probability {answer}")
        return {"false": 1 - yes, "true": yes}
    p = answer["probabilities"]
    if (not p or not 0 <= answer["confidence"] <= 1 or not all(0 <= x <= 1 for x in p.values())
            or not math.isclose(sum(p.values()), 1, abs_tol=1e-5)):
        raise ValueError(f"invalid probability distribution {answer}")
    if answer["type"] == "score":
        if (set(p) != {str(i) for i in range(len(p))}
                or not math.isclose(answer["score"], sum(int(i)*v for i, v in p.items()), abs_tol=1e-6)):
            raise ValueError(f"invalid expected score {answer}")
    elif answer["type"] != "choice" or p.get(answer["choice"]) != max(p.values()):
        raise ValueError(f"invalid choice {answer}")
    return p


def decision(answer):
    if answer["type"] == "score":
        return max(answer["probabilities"], key=answer["probabilities"].get)
    return answer["noul"] >= 0.5 if answer["type"] == "noul" else answer["choice"]


def compare_answers(reference, answers):
    if reference.keys() != answers.keys():
        raise ValueError("response fields differ")
    diff = {}
    confidence_deltas = []
    matches = 0
    for key, base in reference.items():
        answer = answers[key]
        p, ref_p = distribution(answer), distribution(base)
        if answer["type"] != base["type"] or p.keys() != ref_p.keys():
            raise ValueError(f"option set differs for {key}")
        diff[key] = {opt: p[opt] - ref_p[opt] for opt in ref_p}
        matches += decision(answer) == decision(base)
        if base["type"] != "noul":
            confidence_deltas.append(abs(answer["confidence"] - base["confidence"]))
    return matches, diff, max(confidence_deltas, default=None)


def option_table(reference, answers, changes):
    table = {}
    for key, delta in changes.items():
        base, candidate = distribution(reference[key]), distribution(answers[key])
        table[key] = {option: {"reference": base[option], "rk3576": candidate[option], "delta": value}
                      for option, value in delta.items()}
    return table


def verified_calls(row):
    calls = row["calls"]
    if not calls or set(calls[0]["answers"]) != set(row["expected"]):
        raise ValueError("missing calls or unexpected response fields")
    for call in calls:
        if (call["answers"] != calls[0]["answers"] or call["usage"] != calls[0]["usage"]
                or call["operations"].get("npu_matmuls", 0) < 1):
            raise ValueError("inconsistent repeated inference or no NPU execution")
    return calls


def compare_engine_request(reference, candidate, request_id):
    """Compare warm calls to the same complete request across two board engines."""
    if reference["backend"] != "rknn" or candidate["backend"] != "rknn":
        raise ValueError("both engine records must come from the RKNN board")
    for key in ("checkpoint_manifest_sha256", "torch", "transformers", "attention", "state_once", "permutation_average"):
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"different model or runtime setting {key}")
    rows, answers, timings = [], [], {}
    for name, run in (("reference", reference), ("candidate", candidate)):
        found = [(i, row) for i, row in enumerate(run["requests"]) if row["id"] == request_id]
        if len(found) != 1:
            raise ValueError(f"expected a single {request_id} in {name}")
        index, row = found[0]
        calls = verified_calls(row)
        warm = calls if index else calls[1:]
        if not warm:
            raise ValueError(f"no warm measurement for {request_id} in {name}")
        rows.append(row)
        answers.append(warm[0]["answers"])
        timings[name] = {"cpu_affinity": run["cpu_affinity"], "threads": run["threads"],
                         "sdpa_precision": run["sdpa_precision"], "question_rows": run["max_question_rows"],
                         "prefix_backend": run.get("aggressive_prefix_backend"),
                         "warm_seconds": [c["seconds"] for c in warm]}
    if (any(rows[0].get(k) != rows[1].get(k) for k in REQUEST_KEYS)
            or rows[0]["calls"][0]["usage"] != rows[1]["calls"][0]["usage"]):
        raise ValueError("different request, labels or token accounting")
    matches, diff, confidence = compare_answers(*answers)
    errors = [abs(x) for field in diff.values() for x in field.values()]
    score_error = max((abs(a["score"] - answers[0][k]["score"]) for k, a in answers[1].items()
                       if a["type"] == "score"), default=0.0)
    return {"id": request_id, "payload_sha256": rows[0]["payload_sha256"], "usage": rows[0]["calls"][0]["usage"],
            "fields": len(answers[0]), "matched_fields": matches, "options": len(errors),
            "options_within_one_pp": sum(e <= 0.01 for e in errors), "max_abs_probability_delta": max(errors),
            "max_abs_confidence_delta": confidence, "max_abs_score_delta": score_error, **timings,
            "probabilities_by_option": option_table(*answers, diff),
            "choice_probability_gate": matches == len(answers[0]) and max(errors) <= 0.01 and score_error <= 0.01}


def summarize(cpu, candidate, request_id=None, *, compare_cpu_schedules=False):
    for key in RUNTIME_KEYS:
        if compare_cpu_schedules and key in ("cpu_affinity", "threads"):
            continue
        if cpu[key] != candidate[key]:
            raise ValueError(f"different runtime setting {key}")
    if cpu.get("permutation_average", False) != candidate.get("permutation_average", False):
        raise ValueError("different Choice permutation modes")
    if (cpu["backend"], candidate["backend"]) != ("cpu", "rknn"):
        raise ValueError("expected an official CPU run and an RKNN run")
    refs = {r["id"]: r for r in cpu["requests"]}
    ids = [r["id"] for r in candidate["requests"]]
    if not ids or len(set(ids)) != len(ids) or len(refs) != len(cpu["requests"]):
        raise ValueError("need distinct request IDs in each run")
    if request_id is None:
        if set(ids) != refs.keys():
            raise ValueError("the CPU and RKNN runs must cover the same requests")
    elif request_id not in ids or request_id not in refs:
        raise ValueError(f"request is missing from one of the runs: {request_id}")
    reports = []
    for index, row in enumerate(candidate["requests"]):
        if request_id is not None and row["id"] != request_id:
            continue
        calls = verified_calls(row)
        # Only the very first request in this process incurs lazy weight packing.
        warm = calls if index else calls[1:]
        if not warm:
            raise ValueError("include at least one warm measurement per request")
        first = warm[0]
        answers = first["answers"]
        actual = {key: decision(a) for key, a in answers.items()}
        latency = statistics.median(call["seconds"] for call in warm)
        labelled = {k: v for k, v in row["expected"].items() if v is not None}
        item = {"id": row["id"], "fields": len(answers), "labelled_fields": len(labelled),
                "input_tokens": first["usage"]["input_tokens"],
                "first_s": calls[0]["seconds"], "warm_median_s": latency,
                "gold_correct": sum(actual[k] == v for k, v in labelled.items()),
                "process_peak_rss_gib": max(c["peak_rss_bytes"] for c in calls) / 2**30}
        if compare_cpu_schedules:
            item["cpu_schedule"] = {
                label: {key: run[key] for key in ("cpu_affinity", "threads")}
                for label, run in (("reference", cpu), ("candidate", candidate))}
        ref = refs[row["id"]]
        for key in REQUEST_KEYS:
            if ref.get(key) != row.get(key):
                raise ValueError(f"different request representation {key}")
        base = ref["calls"][0]
        if set(base["answers"]) != set(answers) or base["usage"] != first["usage"]:
            raise ValueError("CPU fields or usage differ")
        choices, diff, confidence_error = compare_answers(base["answers"], answers)
        max_error = max(abs(d) for opts in diff.values() for d in opts.values())
        score_error = max((abs(a["score"] - base["answers"][k]["score"]) for k, a in answers.items()
                           if a["type"] == "score"), default=0.0)
        item.update(cpu_seconds=base["seconds"], cpu_over_warm=base["seconds"]/latency,
                    cpu_peak_rss_gib=base["peak_rss_bytes"] / 2**30,
                    cpu_gold_correct=sum(decision(base["answers"][k]) == v for k, v in labelled.items()),
                    cpu_choices_matched=choices, max_abs_probability_delta=max_error,
                    max_abs_confidence_delta=confidence_error, max_abs_score_delta=score_error,
                    probabilities_by_option=option_table(base["answers"], answers, diff),
                    choice_probability_gate=(choices == len(answers) and max_error <= 0.01 and score_error <= 0.01))
        reports.append(item)
    return reports


def published_jev(archive, candidate):
    """The vLLM release contains Jev requests with both Choice option orders averaged."""
    by_id = {r["group_id"]: r for r in archive["records"]}
    rows = candidate["requests"]
    if (not candidate["permutation_average"] or candidate["backend"] != "rknn" or not rows
            or {r["id"] for r in rows} != by_id.keys() or len(rows) != len(by_id) or len(rows) != len(archive["records"])):
        raise ValueError("need identical, distinct requests and the official double-order mode")
    reports = []
    for row in rows:
        ref = by_id[row["id"]]
        if ref["status"] != "ok" or ref["payload_sha256"] != row["payload_sha256"]:
            raise ValueError(f"published payload differs for {row['id']}")
        call = verified_calls(row)[-1]
        if ref["response"]["usage"] != call["usage"] or row["expected"].keys() != row["reversed_question_lengths"].keys():
            raise ValueError("different token usage or incomplete option reversal")
        answers = call["answers"]
        matches, diff, confidence = compare_answers(ref["response"]["answers"], answers)
        max_error = max(abs(d) for opts in diff.values() for d in opts.values())
        reports.append({"id": row["id"], "fields": len(answers), "matched_gpu": matches,
                        "correct": sum(decision(answers[k]) == v for k, v in row["expected"].items()),
                        "gpu_correct": sum(decision(ref["response"]["answers"][k]) == v for k, v in row["expected"].items()),
                        "max_abs_probability_delta": max_error,
                        "max_abs_confidence_delta": confidence,
                        "choice_probability_gate": matches == len(answers) and max_error <= 0.01,
                        "probabilities_by_option": option_table(ref["response"]["answers"], answers, diff)})
    return reports


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("reference", type=Path)
    cli.add_argument("candidate", type=Path)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--published-jev", action="store_true", help="compare to the public MI300X double-order artifact")
    mode.add_argument("--request", help="compare one named request present in both board runs")
    mode.add_argument("--engine-request", help="compare a named request between two RKNN board runs")
    cli.add_argument("--also-cpu", action="append", type=Path, default=[], help="another disjoint CPU control file from the same runtime")
    cli.add_argument("--compare-cpu-schedules", action="store_true",
                     help="record both CPU sets and thread counts when evaluating a scheduling change")
    args = cli.parse_args()
    reference = json.loads(args.reference.read_text())
    candidate = json.loads(args.candidate.read_text())
    if (args.published_jev or args.engine_request) and (args.compare_cpu_schedules or args.also_cpu):
        cli.error("the CPU comparison options require an official CPU reference")
    if args.also_cpu:
        reference = combine_cpu_controls(reference, (json.loads(p.read_text()) for p in args.also_cpu))
    if args.published_jev:
        report = published_jev(reference, candidate)
    elif args.engine_request:
        report = [compare_engine_request(reference, candidate, args.engine_request)]
    else:
        report = summarize(reference, candidate, args.request, compare_cpu_schedules=args.compare_cpu_schedules)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if all(r["choice_probability_gate"] for r in report) else 1)
