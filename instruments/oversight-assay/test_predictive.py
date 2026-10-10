import json
import math
import tempfile
import unittest
from pathlib import Path

import assay
import predictive


class PredictiveTests(unittest.TestCase):
    def setUp(self):
        self.panel = assay.make_panel(107, 20)

    def records(self, mode):
        rows = []
        for item in self.panel["items"]:
            expected = assay.answer(item)
            if mode == "perfect":
                response = {"decision": "accept", "replacement": None} if item["proposed_answer"] == expected else {"decision": "reject", "replacement": expected}
            elif mode == "accept":
                response = {"decision": "accept", "replacement": None}
            elif mode == "wrong":
                response = {"decision": "reject", "replacement": expected + 1}
            else:
                response = {"invalid": True}
            rows.append({"item_id": item["id"], "panel_sha256": self.panel["panel_sha256"], "raw_response": json.dumps(response)})
        return rows

    def test_perfect_oracle_clears_errors(self):
        report = predictive.forecast_report(self.panel, self.records("perfect"), .8, 5)
        self.assertEqual(report["predicted_error"], [.8, 0, 0, 0, 0, 0])
        self.assertFalse(report["confirmatory_eligible"])
        self.assertIsNone(report["proposition_verdict"])

    def test_accept_all_preserves_error(self):
        counts = predictive.transition_counts(self.panel, self.records("accept"))
        self.assertEqual((counts["a_hat"], counts["b_hat"]), (0, 0))
        self.assertEqual(predictive.error_forecast(0, 0, .4, 4), [.4] * 5)

    def test_missing_cannot_look_safe(self):
        report = predictive.forecast_report(self.panel, [], 0, 3)
        self.assertEqual(report["predicted_error"], [0, 1, 1, 1])
        self.assertEqual(report["source"]["coverage"]["missing"], 40)
        self.assertEqual(report["source"]["correct_inputs_with_invalid_or_missing_output"], 20)
        self.assertEqual(report["source"]["correct_inputs_with_valid_but_incorrect_output"], 0)

    def test_malformed_is_not_success(self):
        counts = predictive.transition_counts(self.panel, self.records("invalid"))
        self.assertEqual((counts["a_hat"], counts["b_hat"]), (1, 0))

    def test_bad_repair_can_destroy_correct_work(self):
        counts = predictive.transition_counts(self.panel, self.records("wrong"))
        self.assertEqual((counts["a_hat"], counts["b_hat"]), (1, 0))

    def test_refinement_can_hurt_an_initially_good_system(self):
        series = predictive.error_forecast(.1, .3, .01, 20)
        self.assertGreater(series[-1], series[0])
        self.assertAlmostEqual(series[-1], .25, places=4)

    def test_refinement_can_help_an_initially_bad_system(self):
        series = predictive.error_forecast(.1, .3, .9, 20)
        self.assertLess(series[-1], series[0])

    def test_noncontracting_alternation_is_not_saturation(self):
        self.assertEqual(predictive.error_forecast(1, 1, 0, 4), [0, 1, 0, 1, 0])

    def test_forecast_hash_binds_parameters_and_responses(self):
        one = predictive.forecast_report(self.panel, self.records("accept"), .4, 3)
        two = predictive.forecast_report(self.panel, self.records("accept"), .5, 3)
        self.assertNotEqual(one["forecast_sha256"], two["forecast_sha256"])

    def test_existing_oracle_rejects_cross_panel_data(self):
        rows = self.records("accept")
        rows[0]["panel_sha256"] = "0" * 64
        with self.assertRaises(assay.AssayError):
            predictive.transition_counts(self.panel, rows)

    def test_floor_axis_is_not_identified(self):
        result = predictive.log_range([1.0, 1.0, 1.0])
        self.assertEqual(result["range_dex"], 0)
        self.assertTrue(result["constant_axis"])
        self.assertFalse(result["exponent_identified"])

    def test_varying_axis_alone_does_not_identify_exponent(self):
        result = predictive.log_range([1, 10, 100])
        self.assertEqual(result["range_dex"], 2)
        self.assertFalse(result["exponent_identified"])

    def test_zero_event_bound_satisfies_exact_probability(self):
        upper = predictive.zero_event_upper_bound(100, .05)
        self.assertAlmostEqual((1 - upper) ** 100, .05, places=12)
        self.assertGreater(upper, .029)
        self.assertLess(upper, .030)

    def test_invalid_numbers_fail(self):
        for bad in (True, -1, 2, math.nan, math.inf, "0.5", 10 ** 1000):
            with self.subTest(bad=bad), self.assertRaises(assay.AssayError):
                predictive.error_forecast(bad, .5, .5, 2)
        for bad in ([1, 0], [1, math.inf], [True, 1], [1], [1, 10 ** 1000]):
            with self.subTest(bad=bad), self.assertRaises(assay.AssayError):
                predictive.log_range(bad)
        for n in (0, True, 1.5):
            with self.subTest(n=n), self.assertRaises(assay.AssayError):
                predictive.zero_event_upper_bound(n)

    def test_existing_exclusive_write_preserves_prior_forecast(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "forecast.json"
            assay.write_new(path, "original")
            with self.assertRaises(FileExistsError):
                assay.write_new(path, "replacement")
            self.assertEqual(path.read_text(), "original")


if __name__ == "__main__":
    unittest.main()
