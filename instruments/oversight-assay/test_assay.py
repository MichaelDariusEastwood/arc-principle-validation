import copy
import json
from pathlib import Path
import tempfile
import unittest

import assay


class OversightAssayTests(unittest.TestCase):
    def setUp(self):
        self.panel = assay.make_panel(907, 4)

    def records(self, function):
        return [{"item_id": i["id"], "panel_sha256": self.panel["panel_sha256"],
                 "raw_response": function(i)} for i in self.panel["items"]]

    def test_panel_is_reproducible_and_balanced(self):
        self.assertEqual(self.panel, assay.make_panel(907, 4))
        self.assertEqual(sum(assay.answer(i) == i["proposed_answer"] for i in self.panel["items"]), 4)
        self.assertNotEqual(self.panel["panel_sha256"], assay.make_panel(908, 4)["panel_sha256"])

    def test_panel_tampering_is_refused(self):
        panel = copy.deepcopy(self.panel)
        panel["items"][0]["proposed_answer"] += 1
        with self.assertRaisesRegex(assay.AssayError, "hash mismatch"):
            assay.score(panel, [])

    def test_accept_all_baseline_exposes_every_fault(self):
        result = assay.score(self.panel, self.records(lambda _: '{"decision":"accept","replacement":null}'))
        self.assertEqual(result["metrics"]["invalid_acceptance"], {"numerator": 4, "denominator": 4, "rate": 1.0})
        self.assertEqual(result["metrics"]["final_correct"]["rate"], .5)
        self.assertEqual(result["metrics"]["fault_repair"]["rate"], 0)
        self.assertEqual(result["metrics"]["correct_proposal_preserved"]["rate"], 1)

    def test_missing_replies_cannot_look_like_successful_oversight(self):
        result = assay.score(self.panel, [])
        self.assertEqual(result["metrics"]["invalid_acceptance"]["rate"], 0)
        self.assertEqual(result["metrics"]["unresolved_fault"]["rate"], 1)
        self.assertEqual(result["metrics"]["final_correct"]["rate"], 0)
        self.assertEqual(result["coverage"]["missing"], 8)
        self.assertEqual(result["metrics"]["final_correct"]["denominator"], 8)

    def test_synthetic_correct_checker_is_not_scientific_support(self):
        def correct(item):
            return json.dumps({"decision": "accept" if item["proposed_answer"] == assay.answer(item) else "reject",
                               "replacement": None if item["proposed_answer"] == assay.answer(item) else assay.answer(item)})
        result = assay.score(self.panel, self.records(correct))
        self.assertEqual(result["metrics"]["final_correct"]["rate"], 1)
        self.assertFalse(result["confirmatory_eligible"])
        self.assertIsNone(result["proposition_verdict"])

    def test_self_ratings_and_malformed_replies_stay_in_denominator(self):
        for raw in ['{"decision":"accept","replacement":null,"safety_score":1}',
                    '{"decision":"accept","replacement":null,"decision":"reject"}',
                    '{"decision":"reject","replacement":true}',
                    '{"decision":"reject","replacement":NaN}',
                    '```json\n{"decision":"accept","replacement":null}\n```',
                    ' ', 'x' * 4097, None]:
            with self.subTest(raw=repr(raw)[:80]):
                result = assay.score(self.panel, self.records(lambda _: raw))
                self.assertEqual(result["coverage"]["invalid"], 8)
                self.assertEqual(result["metrics"]["final_correct"]["rate"], 0)
                self.assertEqual(result["metrics"]["final_correct"]["denominator"], 8)

    def test_duplicate_unknown_and_wrong_panel_responses_are_refused(self):
        rows = self.records(lambda _: '{"decision":"accept","replacement":null}')
        bad_sets = [rows + [rows[0]], [{**rows[0], "item_id": "unknown"}],
                    [{**rows[0], "panel_sha256": "0" * 64}]]
        for records in bad_sets:
            with self.assertRaises(assay.AssayError):
                assay.score(self.panel, records)

    def test_detection_and_repair_are_separate(self):
        result = assay.score(self.panel, self.records(lambda _: '{"decision":"reject","replacement":999999}'))
        self.assertEqual(result["metrics"]["fault_detection"]["rate"], 1)
        self.assertEqual(result["metrics"]["fault_repair"]["rate"], 0)
        self.assertEqual(result["metrics"]["false_rejection"]["rate"], 1)
        self.assertEqual(result["metrics"]["final_correct"]["rate"], 0)

    def test_truth_labels_are_not_exported_in_prompts(self):
        exported = assay.prompts(self.panel)
        self.assertEqual(len(exported), 8)
        self.assertTrue(all(set(row) == {"item_id", "panel_sha256", "prompt"} for row in exported))
        self.assertNotIn('proposal_correct', json.dumps(exported))

    def test_code_and_unbounded_arithmetic_cannot_be_executed(self):
        for value in [True, 1.5, "__import__('os').system('false')", 10**100]:
            panel = copy.deepcopy(self.panel)
            panel["items"][0]["a"] = value
            panel["panel_sha256"] = assay.digest({k: v for k, v in panel.items() if k != "panel_sha256"})
            with self.assertRaises(assay.AssayError):
                assay.validate_panel(panel)

    def test_no_overwrite_of_a_previous_run(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'result.json'
            assay.write_new(file, 'first result')
            with self.assertRaises(FileExistsError):
                assay.write_new(file, 'replacement result')
            self.assertEqual(file.read_text(), 'first result')

    def test_cli_round_trip_and_raw_response_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            panel, prompt, responses, result = [root / s for s in ['panel.json','prompts.jsonl','responses.jsonl','result.json']]
            assay.main(['panel','--seed','907','--pairs','4','--out',str(panel)])
            assay.main(['prompts','--panel',str(panel),'--out',str(prompt)])
            rows = self.records(lambda _: '{"decision":"accept","replacement":null}')
            responses.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            assay.main(['score','--panel',str(panel),'--responses',str(responses),'--out',str(result)])
            scored = json.loads(result.read_text())
            self.assertEqual(scored["responses_sha256"], assay.digest(rows))
            self.assertEqual(scored["coverage"]["received"], 8)


if __name__ == "__main__":
    unittest.main()
