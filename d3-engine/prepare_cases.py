"""Extract the fixed public requests without editing their state, question order or options."""

import argparse
import gzip
import hashlib
import json
import urllib.request
from pathlib import Path

CASES = {
    "BFCL-tool-selection:live_multiple:live_multiple_51-21-0":
        ("order", "c736a52e92a1c6110da340ee13dbb4c6f39e4d285c2a2d7eb6f7f9f951ccd170"),
    "home-appliance:test:juniper:09":
        ("home", "a22135bf96a509ece3767496159a8651cf98a3acc3f0bfafb2d9eaf296fc620a"),
    "ContractNLI:test:401":
        ("contract", "631bf3e354d0f3f9982f79e1863cdb2a1ec962f6b7a02741fc233df8b09a5e88"),
}
LABEL_SHA256 = {
    "order": "578935df4606f26772eb5418ce4da9808115ed0006fcd8bfc4b300761f25bcbe",
    "home": "b4563f0e400a8cef47b73605dc8eb16e9704c67f1922b8a3578862b3411117ca",
    "contract": "06f01d958505325bc03aabd19fec8ec4dfd1a25e0f802047403ba3aa59866b03",
}

GPU_RESULTS = ("https://huggingface.co/datasets/vllm-sr/d3.1-nano-decision-index/resolve/"
               "767d71ddb868c0c2be015ea0c5b9110cbbd86451/runs/d3.1-nano/results.jsonl?download=true")
GPU_SHA256 = "b3e2dc83309ade81722d5c833047163ca8bbff94ebe63f6acdff8076dd53b026"


def extract_gpu_reference(destination):
    digest = hashlib.sha256()
    size = rows = 0
    records = {}
    with urllib.request.urlopen(GPU_RESULTS, timeout=90) as stream:
        for line in stream:
            digest.update(line)
            size += len(line)
            rows += 1
            row = json.loads(line)
            key = row.get("group_id")
            if key not in CASES:
                continue
            if key in records or row["status"] != "ok" or row["payload_sha256"] != CASES[key][1]:
                raise ValueError(f"unexpected GPU reference {key}")
            records[key] = row
    if records.keys() != CASES.keys() or digest.hexdigest() != GPU_SHA256:
        raise ValueError("incomplete or different public GPU artifact")
    selection = {"source": GPU_RESULTS, "bytes": size, "rows": rows, "sha256": digest.hexdigest(),
                 "records": [records[key] for key in CASES]}
    destination.write_text(json.dumps(selection, indent=2) + "\n")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("corpus", type=Path)
    cli.add_argument("destination", type=Path)
    cli.add_argument("--gpu-reference", type=Path, help="stream the public results and save the three matching rows here")
    args = cli.parse_args()
    found = {}
    opener = gzip.open if args.corpus.suffix == ".gz" else open
    with opener(args.corpus, "rt", encoding="utf-8") as rows:
        for line in rows:
            row = json.loads(line)
            if row["id"] not in CASES:
                continue
            name, digest = CASES[row["id"]]
            payload = {"model": "jev-1.13.0", "state": row["state"], "questions": row["questions"]}
            wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
            if name in found or hashlib.sha256(wire).hexdigest() != digest or row["_evaluation"]["payload_sha256"] != digest:
                raise ValueError(f"duplicate or changed request {row['id']}")
            labels = json.dumps(row["expected"], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
            if hashlib.sha256(labels).hexdigest() != LABEL_SHA256[name]:
                raise ValueError(f"changed benchmark labels {row['id']}")
            found[name] = line
    if len(found) != len(CASES):
        raise ValueError(f"expected all {len(CASES)} requests, found {list(found)}")
    args.destination.mkdir(parents=True, exist_ok=True)
    fields = {}
    for name, _ in CASES.values():
        line = found[name]
        (args.destination / f"{name}.json").write_text(line)
        row = json.loads(line)
        fields[row["id"]] = {"payload_sha256": row["_evaluation"]["payload_sha256"],
                             "questions": row["questions"]}
    (args.destination / "decision-fields.json").write_text(json.dumps(fields, ensure_ascii=False, indent=2) + "\n")
    if args.gpu_reference:
        extract_gpu_reference(args.gpu_reference)


if __name__ == "__main__":
    main()
