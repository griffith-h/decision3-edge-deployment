import unittest
import json
from pathlib import Path

from compare import combine_cpu_controls, compare_answers, published_jev


class CompareTests(unittest.TestCase):
    def test_combining_independent_cpu_controls_checks_model_and_preserves_inputs(self):
        directory = Path(__file__).with_name("results")
        cpu = json.loads((directory / "cpu.json").read_text())
        router = json.loads((directory / "router-cpu.json").read_text())
        self.assertEqual(len(combine_cpu_controls(cpu, [router])["requests"]), 11)
        self.assertEqual(len(cpu["requests"]), 3)
        with self.assertRaisesRegex(ValueError, "different runtimes or checkpoints"):
            combine_cpu_controls(cpu, [{**router, "checkpoint_manifest_sha256": "different"}])

    def test_noul_flip_and_choice_label_order(self):
        a = {"type": "choice", "probabilities": {"x": .75, "y": .25}, "choice": "x", "confidence": .2}
        b = {"type": "choice", "probabilities": {"y": .125, "x": .875}, "choice": "x", "confidence": .3}
        matches, changes, confidence = compare_answers(
            {"flag": {"type": "noul", "noul": .499}, "route": a},
            {"flag": {"type": "noul", "noul": .501}, "route": b})
        self.assertEqual(matches, 1)
        self.assertAlmostEqual(changes["flag"]["true"], .002)
        self.assertAlmostEqual(changes["flag"]["false"], -.002)
        self.assertEqual(changes["route"], {"x": .125, "y": -.125})
        self.assertAlmostEqual(confidence, .1)

    def test_public_reference_requires_two_orders(self):
        with self.assertRaisesRegex(ValueError, "double-order"):
            published_jev({"records": []}, {"permutation_average": False, "backend": "rknn", "requests": []})

    def test_public_gate_detects_probability_drift_with_same_decision(self):
        directory = Path(__file__).with_name("results")
        published = json.loads((directory / "published-jev.json").read_text())
        board = json.loads((directory / "jev-double.json").read_text())
        self.assertTrue(all(r["choice_probability_gate"] for r in published_jev(published, board)))
        response = published["records"][0]["response"]["answers"]
        answer = next(a for a in response.values() if len(a["probabilities"]) == 2 and max(a["probabilities"].values()) > .9)
        winner = answer["choice"]
        other = next(x for x in answer["probabilities"] if x != winner)
        answer["probabilities"][winner] -= .02
        answer["probabilities"][other] += .02
        result = next(r for r in published_jev(published, board) if r["id"] == published["records"][0]["group_id"])
        self.assertEqual(result["matched_gpu"], result["fields"])
        self.assertFalse(result["choice_probability_gate"])


if __name__ == "__main__":
    unittest.main()
