"""Run the first TypeSafe cases through their published policies using the RK3576's saved model calls.

Run with the pinned WorkflowEvals virtual environment and checkout on PYTHONPATH.
Each missing model call is written as a bench.py input. Replay its results to follow the model's own branches.
"""

import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
from core.dataset import Dataset
from core.question_types import SystemOneResponse, question_record
from core.scoring import compare


REVISIONS = {"customer-service": "b1342f5a704587dbc465867c38c2694348ff86e4",
             "invoice-processing": "6beeb2d2acd65c086c835022f5f4d7434114cafc"}
SOURCE_HASHES = {
    "customer-service": {"dataset.json": "2179468e73465c384cf0c6a807d895f857f2d8ac13ad43342e06a5c87a49cb54",
                         "data/cases.parquet": "ad67fa6da6f2e68e2bbcf3dd65654640339a8f851df6e1789b35ab4400ac5aba",
                         "data/questions.parquet": "06fdb2f494d3254c1a11cbead5cb6743fe4f5d6827d3cec861e63d68d41a7c92"},
    "invoice-processing": {"dataset.json": "72e0cea51735ae3d24dba0dad53536aca72b5698262d57d2506db1b900730e37",
                           "data/cases.parquet": "6a62a51f4b6cda738e0be6a376284ec8c864d580259dd9a6aa30e77685404854",
                           "data/questions.parquet": "416db987cf173b5916c79f7f1d537d2a237e32301aeab4a6d3813bc61d1080bb"},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def question_key(node, state, name, question):
    return node, canonical(state), name, canonical({k: question.get(k) for k in ("type", "instructions", "criteria")})


class MissingCall(Exception):
    pass


class Replay:
    def __init__(self, workflow, case_id, policies, labels, results, destination):
        self.workflow, self.case_id, self.policy_ids = workflow, case_id, policies
        self.labels, self.results, self.destination = labels, results, destination
        self.config = SimpleNamespace(model="d3.1-nano", thinking="default")
        self.node = "semantic"
        self.calls = []

    def system_one(self, model, state, questions):
        questions = {key: question_record(question) for key, question in questions.items()}
        payload = {"state": state, "questions": questions}
        wire = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        digest = hashlib.sha256(wire).hexdigest()
        key = f"typesafe-{self.workflow}-{self.case_id.replace('/', '-')}-{self.node}-{digest[:12]}"
        self.calls.append(key)
        expected = {name: self.labels.get(question_key(self.node, state, name, q)) for name, q in questions.items()}
        data = {"id": key, **payload, "expected": expected,
                "_evaluation": {"workflow": self.workflow, "case_id": self.case_id, "node": self.node,
                                "dataset_revision": REVISIONS[self.workflow], "payload_sha256": digest}}
        file = self.destination / (key + ".json")
        file.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        row = self.results.get(key)
        if row is None:
            raise MissingCall(str(file))
        if row["payload_sha256"] != digest or not row["calls"]:
            raise ValueError(f"incomplete or changed model call {key}")
        call = row["calls"][-1]
        # The SDK uses integer rubric keys in Python and their string spelling on the JSON wire.
        response = {"model": model, "usage": call["usage"], "answers": call["answers"]}
        return SystemOneResponse.model_validate_json(json.dumps(response))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("datasets", type=Path, help="directory containing the two frozen dataset snapshots")
    cli.add_argument("destination", type=Path)
    cli.add_argument("results", type=Path, nargs="*", help="bench.py result files for one backend and precision")
    cli.add_argument("--report", type=Path)
    cli.add_argument("--workflow", choices=REVISIONS, help="run one workflow from the pinned snapshots")
    args = cli.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    rows = {}
    runtime = None
    for path in args.results:
        run = json.loads(path.read_text())
        settings = tuple(run.get(k) for k in ("backend", "sdpa_precision", "checkpoint_manifest_sha256", "permutation_average"))
        if runtime is not None and settings != runtime:
            raise ValueError("workflow replay cannot mix model settings")
        runtime = settings
        for row in run["requests"]:
            if row["id"] in rows:
                raise ValueError(f"duplicate model response {row['id']}")
            rows[row["id"]] = row
    report = {}
    for name in ([args.workflow] if args.workflow else REVISIONS):
        root = args.datasets / name
        for relative, digest in SOURCE_HASHES[name].items():
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"different TypeSafe snapshot: {name}/{relative}")
        dataset = Dataset("typesafeAI/evalsafe-" + name, root)
        module = import_module("evals." + name.replace("-", "_") + ".eval_adapter")
        adapter = module.Adapter(Path(module.__file__).parent, dataset)
        dataset.attach_labels(adapter.bundle)
        case_id = min(dataset.inputs)  # WorkflowEvals --limit 1.
        labels = {}
        for row in pq.read_table(root / "data/questions.parquet", filters=[("case_id", "=", case_id)]).to_pylist():
            consensus = row["consensus"]
            if consensus["status"] == "answered":
                qk = question_key(row["node_id"], json.loads(row["state_json"]),
                                  row["question_id"], json.loads(row["question_json"]))
                labels[qk] = json.loads(consensus["answer_json"])
        policies = [p.policy_id for p in adapter.bundle.policies]
        session = Replay(name, case_id, policies, labels, rows, args.destination)
        result = {"case_id": case_id, "dataset_revision": REVISIONS[name], "calls": session.calls}
        report[name] = result
        try:
            actual = adapter.execute(case_id, session)
        except MissingCall as missing:
            result["missing"] = Path(str(missing)).name
            print("requires board inference", missing)
            continue
        label_set = next(s for s in adapter.bundle.label_sets if s.label_set_id == "labels-consensus")
        gold = next(c for c in label_set.cases if c.case_id == case_id)
        reference = {d.policy_id: d for d in gold.decisions}
        if {d.policy_id for d in actual.decisions} != set(policies) or reference.keys() != set(policies):
            raise ValueError("missing policy decisions")
        result["policies"] = {}
        for d in actual.decisions:
            e = reference[d.policy_id]
            if d.status != "ok" or e.status != "ok":
                raise ValueError("unscored policy decision")
            result["policies"][d.policy_id] = {
                "actual": [a.model_dump(mode="json") for a in d.actions],
                "consensus": [a.model_dump(mode="json") for a in e.actions],
                "metrics": {m.metric_id: compare(m, d, e) for m in adapter.bundle.metrics}}
            if d.primary_action is not None or e.primary_action is not None:
                result["policies"][d.policy_id]["primary_actual"] = d.primary_action and d.primary_action.model_dump(mode="json")
                result["policies"][d.policy_id]["primary_consensus"] = e.primary_action and e.primary_action.model_dump(mode="json")
        print(name, case_id, {p: v["metrics"] for p, v in result["policies"].items()})
    if args.report:
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
