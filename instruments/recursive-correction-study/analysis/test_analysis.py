import copy
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile
import unittest

import analyze_stage_a as a
from synthetic_fixture import make_fixture


def rehash(plan):
    plan["plan_sha256"] = a.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    return plan


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.plan, self.records = make_fixture("known_positive", 8)

    def test_complete_grid_and_known_effects(self):
        vectors, models = a.cluster_contrasts(self.plan, self.records)
        means = [sum(v[j] for v in vectors) / len(vectors) for j in range(3)]
        self.assertEqual(means, [0.375, 0.25, 0.25])
        self.assertEqual(set(models), set(self.plan["model_block_ids"]))

    def test_balanced_null_has_zero_means(self):
        _, plan, records = make_fixture("balanced_null", 16)
        vectors, _ = a.cluster_contrasts(plan, records)
        self.assertEqual([sum(v[j] for v in vectors) / 16 for j in range(3)], [0.0, 0.0, 0.0])

    def test_unfavourable_effects_not_clipped(self):
        _, plan, records = make_fixture("known_negative", 16)
        vectors, _ = a.cluster_contrasts(plan, records)
        self.assertEqual([sum(v[j] for v in vectors) / 16 for j in range(3)], [-0.375, -0.25, -0.25])

    def test_one_deleted_endpoint_is_rejected(self):
        with self.assertRaisesRegex(a.AnalysisError, "incomplete endpoint grid"):
            a.analyse(self.plan, self.records[:-1])

    def test_explicit_missing_endpoint_is_failure(self):
        row = next(r for r in self.records if r["final_success"])
        row["endpoint_status"] = "missing"
        row["final_success"] = False
        result = a.analyse(self.plan, self.records)
        self.assertEqual(sum(x["endpoint_status_counts"]["missing"] for x in result["arm_summaries"]), 1)
        self.assertEqual(result["scheduled_endpoints"], len(self.records))

    def test_missing_cannot_count_as_success(self):
        row = next(r for r in self.records if r["final_success"])
        row["endpoint_status"] = "missing"
        with self.assertRaisesRegex(a.AnalysisError, "must score false"):
            a.validate_records(self.plan, self.records)

    def test_duplicate_endpoint_is_rejected(self):
        with self.assertRaisesRegex(a.AnalysisError, "duplicate scheduled"):
            a.validate_records(self.plan, self.records + [self.records[0]])

    def test_labels_fail_closed(self):
        for field, value in (("feedback", "masked"), ("feedback", None),
                             ("placement", "during"), ("depth", 3), ("depth", True),
                             ("model_block_id", "undeclared"), ("replicate_id", "new-seed"),
                             ("task_cluster_id", "new-task"), ("endpoint_status", "timeout")):
            with self.subTest(field=field, value=value):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                with self.assertRaises((a.AnalysisError, TypeError)):
                    a.validate_records(self.plan, records)

    def test_unscored_outcomes_fail_closed(self):
        for field in ("initial_success", "final_success"):
            for value in (None, 0, 1, 0.5, "true"):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(a.AnalysisError):
                    a.validate_records(self.plan, records)

    def test_initial_pairing_must_match(self):
        records = copy.deepcopy(self.records)
        records[0]["initial_success"] = not records[0]["initial_success"]
        with self.assertRaisesRegex(a.AnalysisError, "share the same initial"):
            a.validate_records(self.plan, records)
        records = copy.deepcopy(self.records)
        records[0]["initial_artefact_sha256"] = "f" * 64
        with self.assertRaisesRegex(a.AnalysisError, "share the same initial"):
            a.validate_records(self.plan, records)

    def test_hash_and_scorer_mismatches_rejected(self):
        records = copy.deepcopy(self.records)
        records[0]["scorer_sha256"] = "f" * 64
        with self.assertRaisesRegex(a.AnalysisError, "scorer digest"):
            a.validate_records(self.plan, records)
        records[0]["trace_sha256"] = None
        with self.assertRaises(a.AnalysisError):
            a.validate_records(self.plan, records)

    def test_mutated_plan_is_not_silently_accepted(self):
        self.plan["analysis"]["seed"] = 18
        with self.assertRaisesRegex(a.AnalysisError, "plan hash mismatch"):
            a.validate_plan(self.plan)

    def test_confirmation_label_is_rejected_even_with_new_hash(self):
        self.plan["dataset_kind"] = "confirmation"
        rehash(self.plan)
        with self.assertRaisesRegex(a.AnalysisError, "confirmation is unsupported"):
            a.validate_plan(self.plan)

    def test_three_contrast_family_cannot_be_selected_after_results(self):
        self.plan["analysis"]["primary_contrasts"] = [a.CONTRASTS[0]]
        rehash(self.plan)
        with self.assertRaisesRegex(a.AnalysisError, "three-contrast family"):
            a.validate_plan(self.plan)

    def test_unknown_fields_rejected(self):
        self.records[0]["registered"] = True
        with self.assertRaises(a.AnalysisError):
            a.validate_records(self.plan, self.records)

    def test_json_nonfinite_and_duplicate_keys_rejected(self):
        for content in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(a.AnalysisError):
                a.parse_json(content)

    def test_replication_does_not_inflate_cluster_count(self):
        plan = copy.deepcopy(self.plan)
        plan["replicate_ids"] = ["SYNTHETIC-REPLICATE-0", "SYNTHETIC-REPLICATE-1"]
        rehash(plan)
        second = copy.deepcopy(self.records)
        for row in second:
            row["replicate_id"] = "SYNTHETIC-REPLICATE-1"
        result = a.analyse(plan, self.records + second)
        self.assertEqual(result["task_clusters"], 8)
        self.assertEqual(result["replicates_per_task_model_arm"], 2)
        self.assertEqual([x["estimate"] for x in result["primary_contrasts"]], [0.375, 0.25, 0.25])

    def test_identical_model_blocks_do_not_shrink_standard_error(self):
        model_a, model_b = self.plan["model_block_ids"]
        outcomes = {a._key(r)[0:1] + a._key(r)[2:]: r["final_success"]
                    for r in self.records if r["model_block_id"] == model_a}
        for row in self.records:
            if row["model_block_id"] == model_b:
                row["final_success"] = outcomes[a._key(row)[0:1] + a._key(row)[2:]]
        report = a.analyse(self.plan, self.records)
        _, by_model = a.cluster_contrasts(self.plan, self.records)
        expected = statistics.stdev(v[0] for v in by_model[model_a]) / math.sqrt(8)
        self.assertAlmostEqual(report["primary_contrasts"][0]["cluster_standard_error"], expected)

    def test_synthetic_never_qualifies_as_registered_evidence(self):
        result = a.analyse(self.plan, self.records)
        self.assertFalse(result["registered"])
        self.assertFalse(result["confirmatory_eligible"])
        self.assertIsNone(result["proposition_verdict"])
        self.assertIn("synthetic_records_are_not_empirical_evidence", result["inferential_interpretation_blocks"])
        self.assertTrue(all(x["directional_or_practical_effect_verdict"] is None for x in result["primary_contrasts"]))

    def test_pilot_independence_declaration_does_not_authorise_confirmation(self):
        self.plan["dataset_kind"] = "pilot"
        self.plan["independence_basis"] = "declared_independent"
        rehash(self.plan)
        for row in self.records:
            row["dataset_kind"] = "pilot"
        result = a.analyse(self.plan, self.records)
        self.assertFalse(result["confirmatory_eligible"])
        self.assertIsNone(result["proposition_verdict"])

    def test_original_inputs_unchanged(self):
        before = a.canonical([self.plan, self.records])
        a.analyse(self.plan, self.records)
        self.assertEqual(a.canonical([self.plan, self.records]), before)

    def test_input_reordering_does_not_change_analysis(self):
        result = a.analyse(self.plan, self.records)
        random.Random(9).shuffle(self.records)
        reordered = a.analyse(self.plan, self.records)
        self.assertEqual(result, reordered)

    def test_outcome_change_alters_digest(self):
        original = a.analyse(self.plan, self.records)
        self.records[0]["final_success"] = not self.records[0]["final_success"]
        revised = a.analyse(self.plan, self.records)
        self.assertNotEqual(original["endpoints_canonical_sha256"], revised["endpoints_canonical_sha256"])

    def test_degenerate_population_uncertainty_is_not_reported_as_zero(self):
        for row in self.records:
            row["final_success"] = True
        result = a.analyse(self.plan, self.records)
        for estimate in result["primary_contrasts"]:
            self.assertEqual(estimate["estimate"], 0)
            self.assertTrue(estimate["bootstrap_degenerate"])
            self.assertIsNone(estimate["bootstrap_percentile_interval_bonferroni_approx"])
            self.assertIsNone(estimate["holm_adjusted_p_approx"])
            self.assertGreater(estimate["hoeffding_interval_bonferroni_conditional"][1], 0)


class StatisticsTests(unittest.TestCase):
    def test_holm_example_and_original_order(self):
        self.assertEqual(a.holm_adjust([0.04, 0.001, 0.01]), [0.04, 0.003, 0.02])

    def test_holm_cumulative_max_is_required(self):
        self.assertEqual(a.holm_adjust([0.01, 0.014, 0.02]), [0.03, 0.03, 0.03])

    def test_unavailable_p_stays_in_family(self):
        self.assertEqual(a.holm_adjust([0.01, None, 0.04]), [0.03, None, 0.08])

    def test_holm_never_below_raw_p(self):
        rng = random.Random(4)
        for _ in range(50):
            values = [rng.random() for _ in range(3)]
            adjusted = a.holm_adjust(values)
            self.assertTrue(all(p <= q <= 1 for p, q in zip(values, adjusted)))

    def test_quantile_linear_interpolation(self):
        self.assertEqual(a.quantile_sorted([0, 1, 2, 3, 4], 0.25), 1)
        self.assertEqual(a.quantile_sorted([0, 10], 0.25), 2.5)

    def test_hoeffding_width_respects_contrast_range_and_multiplicity(self):
        ci1 = a.hoeffding_interval(0.2, 128, -1, 1, 0.05)
        ci2 = a.hoeffding_interval(0.2, 128, -2, 2, 0.05)
        width = 2 * math.sqrt(math.log(120) / 256)
        self.assertAlmostEqual(ci1[1] - 0.2, width)
        self.assertAlmostEqual(ci2[1] - 0.2, 2 * width)
        single = a.hoeffding_interval(0.2, 128, -1, 1, 0.05, family_size=1)
        self.assertGreater(ci1[1], single[1])

    def test_bootstrap_resamples_whole_vectors(self):
        vectors = [(1.0, -1.0, 2.0), (-1.0, 1.0, -2.0), (0.0, 0.0, 0.0)]
        samples = a.bootstrap_cluster_means(vectors, 1000, 10)
        for x, y, z in zip(*samples):
            self.assertAlmostEqual(y, -x)
            self.assertAlmostEqual(z, 2 * x)

    def test_bonferroni_percentile_uses_both_tails(self):
        _, plan, records = make_fixture("known_positive", 32)
        report = a.analyse(plan, records)
        vectors, _ = a.cluster_contrasts(plan, records)
        samples = a.bootstrap_cluster_means(vectors, 1000, 20261010)
        tail = 0.05 / 6
        expected = [a.quantile_sorted(sorted(samples[0]), tail),
                    a.quantile_sorted(sorted(samples[0]), 1 - tail)]
        self.assertEqual(report["primary_contrasts"][0]["bootstrap_percentile_interval_bonferroni_approx"], expected)

    def test_balanced_null_p_values_not_false_certainty(self):
        _, plan, records = make_fixture("balanced_null", 32)
        report = a.analyse(plan, records)
        for result in report["primary_contrasts"]:
            self.assertIn(result["centred_bootstrap_two_sided_p_approx"], (None, 1.0))


class CLITests(unittest.TestCase):
    def test_cli_round_trip_and_exclusive_output(self):
        _, plan, records = make_fixture("known_positive", 8)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            plan_path, input_path, output_path = base / "plan.json", base / "records.jsonl", base / "report.json"
            plan_path.write_text(json.dumps(plan))
            input_path.write_text("".join(a.canonical(row) + "\n" for row in records))
            command = [sys.executable, str(Path(a.__file__)), "--plan", str(plan_path),
                       "--records", str(input_path), "--out", str(output_path)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output_path.read_text())
            self.assertFalse(report["confirmatory_eligible"])
            self.assertIn("raw_input_file_sha256", report)
            original = output_path.read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertEqual(output_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
