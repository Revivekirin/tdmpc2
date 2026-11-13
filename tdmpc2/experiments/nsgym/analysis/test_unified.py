"""Regression against saved scientific tables; no simulator or CUDA."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from .metrics import paired_improvements
from .engine import registry
from .sources import available, ingest

RESULTS = Path(__file__).resolve().parents[4] / "results/nsgym"


class UnifiedAnalysisTests(unittest.TestCase):
    def test_source_counts_and_reacher_gate(self):
        self.assertEqual(available(RESULTS, "halfcheetah")["paired_rows"], 720)
        self.assertEqual(available(RESULTS, "reacher")["matched_shifted_episodes"], 1440)
        h = ingest(RESULTS, "halfcheetah", {"frozen", "dynamics_only", "dynamics_q", "joint"})
        r = ingest(RESULTS, "reacher", {"frozen", "dynamics_only", "joint"})
        self.assertEqual(sum(x["metric"] == "episode_return" for x in h[0]), 2880)
        self.assertEqual(sum(x["metric"] == "episode_return" for x in r[0]), 8640)
        self.assertFalse(any(x["method"] == "dynamics_q" for x in r[0]))
        self.assertTrue(any(x["metric"] == "distance_final" for x in r[0]))
        mass2 = next(x for x in r[2] if x["shift_family"] == "M0" and
                     float(x["severity_multiplier"]) == 2 and x["metric"] == "delta_tip_m")
        self.assertAlmostEqual(float(mass2["applied_parameter_value"]), .07120943348136864)

    def test_paired_arithmetic_regression(self):
        h = ingest(RESULTS, "halfcheetah", {"frozen", "dynamics_only", "dynamics_q", "joint"})
        _, comparisons = paired_improvements(h[0], {"frozen", "dynamics_only", "dynamics_q", "joint"})
        select = lambda m: next(x for x in comparisons if x["shift_parameter"] == "bthigh_damping"
                                and float(x["severity_multiplier"]) == 2 and x["method"] == m)
        self.assertAlmostEqual(select("dynamics_q")["mean"]-select("dynamics_only")["mean"], 487.1018, places=3)
        self.assertAlmostEqual(select("joint")["mean"], -524.338, places=2)
        self.assertEqual(select("joint")["n_adaptation_streams"], 3)
        self.assertEqual(select("joint")["evaluation_seeds_per_stream"], 20)

    def test_reacher_nominal_alias_not_independent(self):
        r = ingest(RESULTS, "reacher", {"frozen", "dynamics_only", "joint"})
        _, comparisons = paired_improvements(r[0], {"frozen", "dynamics_only", "joint"})
        aliases = [x for x in comparisons if float(x["severity_multiplier"]) == 1]
        self.assertTrue(aliases)
        self.assertTrue(all(x["nominal_duplicate"] == "True" for x in aliases))

    def test_q_saved_metrics_not_redecoded(self):
        q = ingest(RESULTS, "halfcheetah", {"frozen", "dynamics_only", "dynamics_q", "joint"})[3]
        self.assertTrue(any(x["metric"] == "entropy" and float(x["value"]) > 0 for x in q))
        self.assertTrue(any(x["metric"] == "js" for x in q))
        self.assertFalse(any(x["metric"] == "calibration_error" for x in q))

    def test_reacher_dq_completion_gate_late_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"reacher_dynamics_q": {}}
            for stage in ("train", "prediction", "control", "aggregate"):
                p = root / stage / "completion.json"
                p.parent.mkdir()
                p.write_text(json.dumps({"status": "PASS"}))
                config["reacher_dynamics_q"][stage if stage == "train" else stage] = str(p.relative_to(root))
            paths = config["reacher_dynamics_q"]
            paths["training_completion"] = paths.pop("train")
            for stage in ("prediction", "control", "aggregate"):
                paths[stage + "_completion"] = paths.pop(stage)
            (root / "aggregate/four_way_per_seed.csv").write_text("family,severity\n")
            paths["aggregate_per_seed_csv"] = "aggregate/four_way_per_seed.csv"
            self.assertFalse(registry(root, config)["reacher"]["dynamics_q_gate"]["passed"])
            (root / "control/completion.json").write_text(json.dumps({"status": "PARTIAL"}))
            self.assertFalse(registry(root, config)["reacher"]["dynamics_q_gate"]["passed"])


if __name__ == "__main__":
    unittest.main()
