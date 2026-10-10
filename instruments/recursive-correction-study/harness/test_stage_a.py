import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest

import batch_adapter
import stage_a as stage


class StageATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = stage.demo_study(1)
        cls.ledger = []
        while True:
            ready = stage.pending(cls.study, cls.ledger)
            if not ready:
                break
            cls.ledger = stage.import_responses(cls.study, cls.ledger,
                [stage.synthetic_response(cls.study, r) for r in ready])

    def test_full_grid_has_one_shared_initial_and_matched_call_counts(self):
        nodes = stage.graph(self.study)
        self.assertEqual(len(nodes), 228)
        initial = [n for n in nodes if n["kind"] == "initial"]
        self.assertEqual(len(initial), 4)
        for task in self.study["panel"]["items"]:
            for model in self.study["models"]:
                block = [n for n in nodes if n["task_cluster_id"] == task["id"] and n["model_block_id"] == model["block_id"]]
                self.assertEqual(len(block), 57)
                self.assertEqual(sum(n["kind"] == "final" for n in block), 8)
                for depth in (2, 4):
                    for feedback in stage.FEEDBACK:
                        for placement in stage.PLACEMENT:
                            policy = {"depth": depth, "feedback": feedback, "placement": placement}
                            arm = [n for n in block if n["policy"] == policy]
                            self.assertEqual(sum(n["kind"] == "critic" for n in arm), depth)
                            self.assertEqual(sum(n["kind"] in ("rewrite", "final") for n in arm), depth+1)

    def test_terminal_critics_see_same_draft_and_all_advice_is_applied(self):
        nodes = stage.graph(self.study)
        for final in (n for n in nodes if n["kind"] == "final" and n["policy"]["placement"] == "terminal_panel"):
            critics = [n for n in nodes if n["call_id"] in final["critics"]]
            self.assertEqual(len(critics), final["policy"]["depth"])
            self.assertEqual({c["artefact"] for c in critics}, {final["artefact"]})
            self.assertTrue(all(c["critics"] == [] for c in critics))

    def test_neutral_mask_is_identical_across_truth_and_format(self):
        item = self.study["panel"]["items"][0]
        variants = [stage.canonical({"answer": stage.assay.answer(item)}), '{"answer": 93241}', 'bad', '{"answer": true}']
        self.assertEqual(len({stage.canonical(stage.diagnostic(item, raw, "neutral_masked")) for raw in variants}), 1)
        self.assertEqual(stage.diagnostic(item, variants[0], "diagnostic")["check"], "passes")
        self.assertEqual(stage.diagnostic(item, "bad", "diagnostic")["format"], "invalid")

    def test_prompts_omit_truth_labels_panel_proposal_and_design_ids(self):
        req = stage.pending(self.study, [])[0]
        text = req["body"]["input"]
        for forbidden in ("proposed_answer", "panel_sha256", "initial_success", "model_block_id", "seed", "item-", "neutral_masked"):
            self.assertNotIn(forbidden, text)

    def test_dependency_order_and_duplicate_responses_are_rejected(self):
        with self.assertRaises(stage.StudyError):
            stage.validate_ledger(self.study, list(reversed(self.ledger)))
        with self.assertRaises(stage.StudyError):
            stage.validate_ledger(self.study, self.ledger + [self.ledger[0]])

    def test_response_cannot_be_reassigned_to_another_request(self):
        ready = stage.pending(self.study, [])
        row = stage.synthetic_response(self.study, ready[0])
        row["call_id"] = ready[1]["call_id"]
        with self.assertRaises(stage.StudyError):
            stage.import_responses(self.study, [], [row])

    def test_changed_runner_or_claim_status_invalidates_study(self):
        for field, value in (("confirmatory_eligible", True), ("registered", True), ("runner_sha256", "0"*64), ("suite_id", "real_alignment")):
            changed = copy.deepcopy(self.study)
            changed[field] = value
            with self.assertRaises(stage.StudyError):
                stage.validate_study(changed)

    def test_synthetic_record_cannot_be_promoted_to_model_observation(self):
        ready = stage.pending(self.study, [])[0]
        row = stage.synthetic_response(self.study, ready)
        row["source_kind"] = "model_response"
        with self.assertRaises(stage.StudyError):
            stage.import_responses(self.study, [], [row])

    def test_incomplete_collection_is_not_silently_scored(self):
        with self.assertRaises(stage.StudyError):
            stage.endpoint_records(self.study, [])
        rows = stage.endpoint_records(self.study, [], allow_aborted=True)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(r["endpoint_status"] == "missing" and not r["final_success"] for r in rows))

    def test_malformed_endpoint_stays_in_denominator(self):
        final_ids = {n["call_id"] for n in stage.graph(self.study) if n["kind"] == "final"}
        ledger = copy.deepcopy(self.ledger)
        target = next(row for row in ledger if row["call_id"] in final_ids)
        target["raw_output"] = '{"answer": true}'
        rows = stage.endpoint_records(self.study, ledger)
        self.assertEqual(len(rows), 32)
        self.assertEqual(sum(r["endpoint_status"] == "invalid" for r in rows), 1)

    def test_initial_failure_blocks_descendants_without_making_model_calls(self):
        ready = stage.pending(self.study, [])
        failures = []
        for req in ready:
            row = stage.synthetic_response(self.study, req)
            row.update(status="infrastructure_failure", completion_reason="provider_error", raw_output="", model_id=None)
            failures.append(row)
        ledger = stage.import_responses(self.study, [], failures)
        self.assertEqual(stage.pending(self.study, ledger), [])
        rows = stage.endpoint_records(self.study, ledger)
        self.assertEqual(len(rows), 32)
        self.assertTrue(all(r["endpoint_status"] == "infrastructure_failure" for r in rows))

    def test_shared_initial_bytes_survive_all_eight_policies(self):
        rows = stage.endpoint_records(self.study, self.ledger)
        grouped = {}
        for row in rows:
            key = (row["task_cluster_id"], row["model_block_id"], row["replicate_id"])
            grouped.setdefault(key, set()).add((row["initial_artefact_sha256"], row["initial_success"]))
        self.assertTrue(all(len(v) == 1 for v in grouped.values()))

    def test_long_raw_output_is_retained_and_replaced_by_fixed_invalid_candidate(self):
        ready = stage.pending(self.study, [])[0]
        row = stage.synthetic_response(self.study, ready)
        row["raw_output"] = "x" * (stage.MAX_OUTPUT_BYTES + 1)
        ledger = stage.import_responses(self.study, [], [row])
        self.assertEqual(ledger[0]["raw_output"], row["raw_output"])
        self.assertIn("EXCEEDED", stage.visible_output(ledger[0]))
        self.assertFalse(stage.outcome(self.study["panel"]["items"][0], row["raw_output"])[1])

    def test_model_budget_failure_continues_schedule_without_becoming_an_outage(self):
        ready = stage.pending(self.study, [])[0]
        row = stage.synthetic_response(self.study, ready)
        row.update(status="model_unsuccessful", completion_reason="max_output_tokens")
        ledger = stage.import_responses(self.study, [], [row])
        next_calls = stage.pending(self.study, ledger)
        self.assertTrue(any("DID NOT COMPLETE" in r["body"]["input"] for r in next_calls))

    def test_terminal_prompt_cap_covers_worst_case_json_escaping(self):
        nodes = stage.graph(self.study)
        node = next(n for n in nodes if n["kind"] == "final" and n["policy"]["placement"] == "terminal_panel" and n["policy"]["depth"] == 4)
        by_id = {r["call_id"]: copy.deepcopy(r) for r in self.ledger}
        for cid in node["dependencies"]:
            by_id[cid]["raw_output"] = "\x00" * stage.MAX_OUTPUT_BYTES
        req = stage.request(self.study, node, by_id)
        self.assertLessEqual(len(req["body"]["input"].encode()), stage.MAX_PROMPT_BYTES)

    def test_parser_rejects_duplicates_nan_bool_unbounded_and_prose(self):
        for raw in ('{"answer": 3, "answer": 4}', '{"answer": NaN}', '{"answer": true}', '{"answer": 1000001}', 'Answer: 3', '```json\n{"answer":3}\n```'):
            with self.assertRaises(stage.StudyError, msg=raw):
                stage.parse_answer(raw)
        self.assertEqual(stage.parse_answer('{"answer": -12}'), -12)

    def test_returned_model_and_usage_caps_are_checked(self):
        req = stage.pending(self.study, [])[0]
        row = stage.synthetic_response(self.study, req)
        row["model_id"] = "different"
        with self.assertRaises(stage.StudyError):
            stage.import_responses(self.study, [], [row])
        row = stage.synthetic_response(self.study, req)
        row["usage"]["output_tokens"] = 257
        with self.assertRaises(stage.StudyError):
            stage.import_responses(self.study, [], [row])

    def test_analysis_contract_works_and_does_not_infer_from_item_labels(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
        from analyze_stage_a import make_plan, analyse
        plan = make_plan([i["id"] for i in self.study["panel"]["items"]], ["A", "B"], ["r1"],
                         self.study["scorer_sha256"], self.study["study_sha256"],
                         study_id=self.study["study_id"], dataset_kind="synthetic", independence_basis="not_established", resamples=2000)
        report = analyse(plan, stage.endpoint_records(self.study, self.ledger))
        self.assertFalse(report["confirmatory_eligible"])
        self.assertIsNone(report["proposition_verdict"])


class BatchAdapterTests(unittest.TestCase):
    def setUp(self):
        models = [{"block_id": block, "model_id": "fixture-snapshot-2026-01-01", "backend": "openai_responses",
                   "revision": "fixture-only", "revision_evidence_sha256": stage.digest({"fixture": block}),
                   "generation_parameters": {}} for block in ("A", "B")]
        self.study = stage.make_study(stage.assay.make_panel(1, 1), models, dataset_kind="pilot")
        self.requests = stage.pending(self.study, [])

    def completed(self, req, text='{"answer": 1}'):
        return {"id": "fixture", "custom_id": req["call_id"], "error": None,
                "response": {"status_code": 200, "body": {"status": "completed", "error": None,
                "model": req["body"]["model"], "output": [{"type": "reasoning"},
                {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
                "usage": {"input_tokens": 55, "output_tokens": 9, "output_tokens_details": {"reasoning_tokens": 2}}}}}

    def test_exports_only_provider_body_and_opaque_id(self):
        groups = batch_adapter.export_batches(self.requests)
        self.assertEqual(len(groups), 1)
        rows = next(iter(groups.values()))
        self.assertEqual(set(rows[0]), {"custom_id", "method", "url", "body"})
        self.assertEqual(rows[0]["url"], "/v1/responses")
        self.assertNotIn("task_cluster_id", stage.canonical(rows[0]))

    def test_separate_models_are_never_mixed_in_a_batch(self):
        requests = copy.deepcopy(self.requests)
        requests[0]["body"]["model"] = "second-fixture-2026-02-02"
        requests[0]["request_sha256"] = stage.digest({k:v for k,v in requests[0].items() if k != "request_sha256"})
        groups = batch_adapter.export_batches(requests)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(len({r["body"]["model"] for r in rows}) == 1 for rows in groups.values()))

    def test_result_matching_uses_id_not_return_order(self):
        incoming = [self.completed(req) for req in reversed(self.requests)]
        rows = batch_adapter.normalise_results(self.requests, incoming)
        ledger = stage.import_responses(self.study, [], rows)
        self.assertEqual(len(ledger), len(incoming))
        self.assertTrue(all(row["usage"]["reasoning_tokens"] == 2 for row in ledger))

    def test_unknown_duplicate_and_changed_model_results_rejected(self):
        req = self.requests[0]
        row = self.completed(req)
        for rows in ([row, row], [{**row, "custom_id": "unknown"}]):
            with self.assertRaises(stage.StudyError):
                batch_adapter.normalise_results(self.requests, rows)
        row["response"]["body"]["model"] = "unexpected"
        with self.assertRaises(stage.StudyError):
            batch_adapter.normalise_results(self.requests, [row])

    def test_missing_results_remain_pending_and_explicit_errors_are_preserved(self):
        self.assertEqual(batch_adapter.normalise_results(self.requests, []), [])
        req = self.requests[0]
        error = {"custom_id": req["call_id"], "response": None, "error": {"code": "batch_expired"}}
        rows = batch_adapter.normalise_results(self.requests, [error])
        self.assertEqual(rows[0]["status"], "infrastructure_failure")
        self.assertIsNone(rows[0]["usage"]["output_tokens"])

    def test_synthetic_requests_and_aliases_cannot_be_exported(self):
        with self.assertRaises(stage.StudyError):
            batch_adapter.export_batches(stage.pending(stage.demo_study(1), []))
        req = copy.deepcopy(self.requests[0])
        req["body"]["model"] = "model-latest"
        req["request_sha256"] = stage.digest({k:v for k,v in req.items() if k != "request_sha256"})
        with self.assertRaises(stage.StudyError):
            batch_adapter.export_batches([req])

    def test_token_limit_and_refusal_are_unsuccessful_model_outputs_not_outages(self):
        req = self.requests[0]
        row = self.completed(req)
        row["response"]["body"]["status"] = "incomplete"
        row["response"]["body"]["incomplete_details"] = {"reason": "max_output_tokens"}
        normal = batch_adapter.normalise_results(self.requests, [row])[0]
        self.assertEqual((normal["status"], normal["completion_reason"]), ("model_unsuccessful", "max_output_tokens"))
        self.assertEqual(normal["usage"]["output_tokens"], 9)
        row = self.completed(req)
        row["response"]["body"]["output"] = [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "I cannot complete this request."}]}]
        normal = batch_adapter.normalise_results(self.requests, [row])[0]
        self.assertEqual((normal["status"], normal["completion_reason"]), ("model_unsuccessful", "refusal"))


if __name__ == "__main__":
    unittest.main()
