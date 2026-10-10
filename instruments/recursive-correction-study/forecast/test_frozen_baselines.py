import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

import frozen_baselines as fb


def outcomes_for(ledger, status="correct"):
    return [{**{k: row[k] for k in ("task_id", "task_sha256", "configuration_id", "horizon")},
             "final_status": status, "collected_at": fb.now_iso()} for row in ledger["forecasts"]]


def rehash(ledger):
    ledger["ledger_sha256"] = fb.digest({k: v for k, v in ledger.items() if k != "ledger_sha256"})


class FrozenBaselineTests(unittest.TestCase):
    def setUp(self):
        self.plan, self.records = fb.demo_inputs()

    def frozen(self):
        return fb.freeze(self.plan, self.records)

    def score(self, ledger, outcomes):
        return fb.evaluate(ledger, outcomes, ledger["ledger_sha256"], bootstrap=100)

    def test_mixture_counterexample(self):
        result = fb.mixture_counterexample()
        self.assertAlmostEqual(result["mixture_error_depth_1"], .5)
        self.assertAlmostEqual(result["mixture_error_depth_2"], .34)
        self.assertAlmostEqual(result["pooled_prediction_depth_2"], .25)

    def test_rates_have_real_state_exposures(self):
        rates = fb.fit(self.plan, self.records)["mock-policy"]
        self.assertEqual(rates["easier"]["correct_exposure"], 10)
        self.assertEqual(rates["easier"]["error_exposure"], 10)
        self.assertEqual(rates["easier"]["a"], 0)
        self.assertEqual(rates["easier"]["b"], .8)
        self.assertEqual(rates["harder"]["b"], .2)
        self.assertEqual(rates["*"]["b"], .5)

    def test_beta_one_one_is_explicit_plugin(self):
        self.plan["smoothing"] = {"success_pseudocount": 1, "failure_pseudocount": 1}
        r = fb.fit(self.plan, self.records)["mock-policy"]["easier"]
        self.assertAlmostEqual(r["a"], 1/12)
        self.assertAlmostEqual(r["b"], 9/12)
        self.assertEqual(r["estimator"], "prespecified smoothed plug-in")

    def test_no_prior_only_transition(self):
        self.plan["calibration_tasks"] = [t for t in self.plan["calibration_tasks"] if t["initial_error"]]
        ids = {t["task_id"] for t in self.plan["calibration_tasks"]}
        records = [r for r in self.records if r["task_id"] in ids]
        self.plan["smoothing"] = {"success_pseudocount": 1, "failure_pseudocount": 1}
        with self.assertRaisesRegex(fb.ForecastError, "No required state exposure"):
            fb.fit(self.plan, records)

    def test_plan_disallows_duplicate_content(self):
        self.plan["prediction_tasks"][1]["task_sha256"] = self.plan["prediction_tasks"][0]["task_sha256"]
        with self.assertRaisesRegex(fb.ForecastError, "Duplicate task"):
            self.frozen()

    def test_plan_disallows_shared_calibration_cluster(self):
        self.plan["prediction_tasks"][0]["cluster_id"] = self.plan["calibration_tasks"][0]["cluster_id"]
        with self.assertRaisesRegex(fb.ForecastError, "overlap"):
            self.frozen()

    def test_unknown_or_missing_stratum_fails(self):
        self.plan["prediction_tasks"][0]["stratum"] = "outcome_selected"
        with self.assertRaisesRegex(fb.ForecastError, "Undeclared stratum"):
            self.frozen()

    def test_moving_model_alias_fails(self):
        self.plan["configurations"][0]["model_revision"] = "latest"
        with self.assertRaisesRegex(fb.ForecastError, "moving model alias"):
            self.frozen()

    def test_weights_and_horizons_validate(self):
        self.plan["strata_weights"] = {"easier": .5, "harder": .6}
        with self.assertRaises(fb.ForecastError):
            self.frozen()
        self.plan["strata_weights"] = {"easier": .5, "harder": .5}
        self.plan["horizons"] = [True, 2]
        with self.assertRaises(fb.ForecastError):
            self.frozen()

    def test_duplicate_calibration_row_fails(self):
        with self.assertRaisesRegex(fb.ForecastError, "duplicate calibration"):
            fb.freeze(self.plan, self.records + [self.records[0]])

    def test_missing_calibration_row_needs_explicit_record(self):
        with self.assertRaisesRegex(fb.ForecastError, "Missing calibration records"):
            fb.freeze(self.plan, self.records[:-1])

    def test_missing_calibration_response_is_unsuccessful(self):
        self.records[0]["final_status"] = "missing"
        r = fb.fit(self.plan, self.records)["mock-policy"]["easier"]
        self.assertEqual(r["missing"], 1)
        self.assertAlmostEqual(r["a"], .1)

    def test_forecasts_recover_target_mixture(self):
        ledger = self.frozen()
        rows = [r for r in ledger["forecasts"] if r["horizon"] == 2]
        self.assertAlmostEqual(sum(r["predicted_error"]["stratified"] for r in rows)/len(rows), .34)
        self.assertTrue(all(r["predicted_error"]["pooled"] == .25 for r in rows))

    def test_freeze_does_not_share_mutable_input(self):
        ledger = self.frozen()
        self.plan["population"] = "changed"
        self.records[0]["final_status"] = "missing"
        fb.verify_ledger(ledger, ledger["ledger_sha256"])

    def test_external_digest_pin_and_forecast_derivation(self):
        ledger = self.frozen()
        with self.assertRaisesRegex(fb.ForecastError, "hash mismatch"):
            fb.verify_ledger(ledger, "f"*64)
        ledger["forecasts"][0]["predicted_error"]["stratified"] = .99
        rehash(ledger)
        with self.assertRaisesRegex(fb.ForecastError, "frozen calibration derivation"):
            fb.verify_ledger(ledger, ledger["ledger_sha256"])

    def test_cannot_promote_development_flag(self):
        ledger = self.frozen()
        ledger["confirmatory_eligible"] = True
        rehash(ledger)
        with self.assertRaisesRegex(fb.ForecastError, "scope has changed"):
            fb.verify_ledger(ledger, ledger["ledger_sha256"])

    def test_chronology_rejects_late_calibration(self):
        self.plan["calibration_completed_at"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        with self.assertRaisesRegex(fb.ForecastError, "cannot follow"):
            self.frozen()

    def test_chronology_rejects_prior_outcomes(self):
        ledger = self.frozen()
        outcomes = outcomes_for(ledger)
        outcomes[0]["collected_at"] = ledger["frozen_at"]
        with self.assertRaisesRegex(fb.ForecastError, "after freeze"):
            self.score(ledger, outcomes)

    def test_chronology_rejects_future_outcome(self):
        ledger = self.frozen()
        outcomes = outcomes_for(ledger)
        outcomes[0]["collected_at"] = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        with self.assertRaisesRegex(fb.ForecastError, "not in the future"):
            self.score(ledger, outcomes)

    def test_holdout_matrix_complete_and_unique(self):
        ledger = self.frozen()
        outcomes = outcomes_for(ledger)
        with self.assertRaisesRegex(fb.ForecastError, "Incomplete scheduled"):
            self.score(ledger, outcomes[:-1])
        with self.assertRaisesRegex(fb.ForecastError, "duplicate outcome"):
            self.score(ledger, outcomes + [outcomes[0]])

    def test_missing_and_invalid_stay_in_denominator(self):
        ledger = self.frozen()
        outcomes = outcomes_for(ledger)
        outcomes[0]["final_status"], outcomes[1]["final_status"] = "missing", "invalid"
        report = self.score(ledger, outcomes)
        self.assertEqual(report["scheduled_rows"], 100)
        self.assertEqual(report["scored_rows"], 100)
        self.assertEqual(report["status_counts"]["missing"], 1)
        self.assertEqual(report["status_counts"]["invalid"], 1)
        self.assertEqual(sum(r["observed_error"] for r in report["rows"]), 2)

    def test_brier_is_mean_of_squared_paired_task_errors(self):
        ledger = self.frozen()
        report = self.score(ledger, outcomes_for(ledger))
        # All successes: pooled p=.5,.25; stratified p=.2,.04,.8,.64.
        self.assertAlmostEqual(report["overall"]["mean_brier"]["pooled"], (.5**2+.25**2)/2)
        self.assertAlmostEqual(report["overall"]["mean_brier"]["stratified"], (.2**2+.04**2+.8**2+.64**2)/4)
        self.assertAlmostEqual(report["overall"]["pooled_minus_stratified"],
                               report["overall"]["mean_brier"]["pooled"] - report["overall"]["mean_brier"]["stratified"])

    def test_predeclared_weights_not_sample_counts(self):
        self.plan["strata_weights"] = {"easier": .8, "harder": .2}
        ledger = self.frozen()
        report = self.score(ledger, outcomes_for(ledger))
        expected = .8*(.2**2+.04**2)/2+.2*(.8**2+.64**2)/2
        self.assertAlmostEqual(report["overall"]["mean_brier"]["stratified"], expected)

    def test_clusters_are_not_individual_rows(self):
        for task in self.plan["prediction_tasks"]:
            task["cluster_id"] = "one-cluster-" + task["stratum"]
        ledger = self.frozen()
        report = self.score(ledger, outcomes_for(ledger))
        self.assertEqual(report["clusters_by_stratum"], {"easier": 1, "harder": 1})
        self.assertIsNone(report["paired_cluster_bootstrap"]["pooled_minus_stratified_interval"])

    def test_identical_paired_configuration_effect_is_zero(self):
        self.plan["configurations"].append({"configuration_id": "other", "model_revision": "synthetic-fixture-v1", "policy_sha256": fb.digest("other")})
        self.plan["intervention_pairs"] = [{"control": "mock-policy", "treatment": "other"}]
        self.records += [{**r, "configuration_id": "other"} for r in list(self.records)]
        ledger = self.frozen()
        report = self.score(ledger, outcomes_for(ledger))
        self.assertEqual(report["scored_rows"], 200)
        for effect in report["intervention_effect_forecasts"]:
            self.assertEqual(effect["observed_error_difference"], 0)
            self.assertEqual(effect["absolute_effect_forecast_error"], {"pooled": 0, "stratified": 0})

    def test_reordering_outcomes_keeps_paired_bootstrap(self):
        ledger = self.frozen()
        outcomes = outcomes_for(ledger)
        first, second = self.score(ledger, outcomes), self.score(ledger, list(reversed(outcomes)))
        self.assertEqual(first["overall"], second["overall"])
        self.assertEqual(first["paired_cluster_bootstrap"], second["paired_cluster_bootstrap"])

    def test_json_parser_rejects_duplicates_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.json"
            for source in ('{"x":1,"x":2}', '{"x":NaN}'):
                path.write_text(source)
                with self.assertRaises(fb.ForecastError):
                    fb.read_json(path)

    def test_exclusive_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"record.json"
            fb.write_new(path, {"x": 1})
            with self.assertRaises(FileExistsError):
                fb.write_new(path, {"x": 2})
            self.assertEqual(json.loads(path.read_text()), {"x": 1})

    def test_invalid_numeric_parameters(self):
        for value in (float("nan"), True, -1, 2, 10**1000):
            with self.assertRaises(fb.ForecastError):
                fb.transition_forecast(value, .5, 1, 2)


if __name__ == "__main__":
    unittest.main()
