#!/usr/bin/env python3
"""Offline OpenAI Responses Batch file adapter. No authentication or network.

Only ready requests are exported, in separate files for each model. An actual
operator must verify model support, prices and caps before submitting files.
An exported file is not evidence of submission, completion or registration.
"""
import argparse
from pathlib import Path
import re

import stage_a as stage


def export_batches(requests):
    groups, seen = {}, set()
    for req in requests:
        fields = {"call_id", "model_block_id", "backend", "body", "request_sha256"}
        if not isinstance(req, dict) or set(req) != fields:
            raise stage.StudyError("unexpected request fields")
        expected = stage.digest({k: v for k, v in req.items() if k != "request_sha256"})
        if req["request_sha256"] != expected or req["call_id"] in seen:
            raise stage.StudyError("modified or duplicate request")
        if req["backend"] != "openai_responses":
            raise stage.StudyError("only an explicitly declared OpenAI Responses backend can be exported")
        body = req["body"]
        if not isinstance(body, dict) or not re.search(r"-\d{4}-\d{2}-\d{2}$", body.get("model", "")):
            raise stage.StudyError("a dated model snapshot is required; verify actual availability separately")
        if set(body) - {"model", "input", "max_output_tokens", "store", "temperature", "top_p", "reasoning"}:
            raise stage.StudyError("unsupported request body fields")
        if body.get("store") is not False or not isinstance(body.get("input"), str):
            raise stage.StudyError("expected a stateless text request")
        stage.bounded_int(body.get("max_output_tokens"), "max_output_tokens", 16, 4096)
        seen.add(req["call_id"])
        groups.setdefault(body["model"], []).append({"custom_id": req["call_id"], "method": "POST",
                                                   "url": "/v1/responses", "body": body})
    return groups


def normalise_results(requests, batch_rows):
    """Preserve actual usage, absence and errors; do not infer data from order."""
    export_batches(requests)  # Check the exact request identities and provider.
    by_id = {r["call_id"]: r for r in requests}
    seen, result = set(), []
    for envelope in batch_rows:
        if not isinstance(envelope, dict) or envelope.get("custom_id") not in by_id:
            raise stage.StudyError("unknown batch custom_id")
        call_id = envelope["custom_id"]
        if call_id in seen:
            raise stage.StudyError("duplicate batch result")
        seen.add(call_id)
        req = by_id[call_id]
        response = envelope.get("response")
        body = response.get("body") if isinstance(response, dict) else None
        ok = isinstance(body, dict) and response.get("status_code") == 200 and envelope.get("error") is None
        terminal_output = ok and body.get("status") in ("completed", "incomplete") and body.get("error") is None
        text, model_id, status, reason = "", None, "infrastructure_failure", "provider_error"
        if terminal_output:
            if body.get("model") != req["body"]["model"]:
                raise stage.StudyError("provider returned a different model snapshot")
            model_id = body["model"]
            parts, has_refusal = [], False
            output = body.get("output")
            if not isinstance(output, list):
                raise stage.StudyError("completed response has no output array")
            for item in output:
                if not isinstance(item, dict):
                    raise stage.StudyError("invalid output item")
                if item.get("type") == "message" and item.get("role") == "assistant":
                    content = item.get("content")
                    if not isinstance(content, list):
                        raise stage.StudyError("invalid assistant content")
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            if not isinstance(part.get("text"), str):
                                raise stage.StudyError("invalid output_text")
                            parts.append(part["text"])
                        elif isinstance(part, dict) and part.get("type") == "refusal":
                            refusal = part.get("refusal")
                            if not isinstance(refusal, str):
                                raise stage.StudyError("invalid refusal content")
                            parts.append(refusal)
                            has_refusal = True
            text = "\n".join(parts)
            if len(text.encode()) > stage.MAX_STORED_OUTPUT_BYTES:
                raise stage.StudyError("raw response exceeds custody limit; retain original and report incident")
            if has_refusal:
                status, reason = "model_unsuccessful", "refusal"
            elif body["status"] == "incomplete":
                details = body.get("incomplete_details")
                detail = details.get("reason") if isinstance(details, dict) else None
                status = "model_unsuccessful"
                reason = detail if detail in ("max_output_tokens", "content_filter") else "other_incomplete"
            else:
                status, reason = "completed", "completed"
        usage = body.get("usage") if isinstance(body, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        details = usage.get("output_tokens_details")
        details = details if isinstance(details, dict) else {}
        result.append({"schema": stage.RESPONSE_SCHEMA, "call_id": call_id,
                       "request_sha256": req["request_sha256"], "source_kind": "model_response",
                       "status": status, "completion_reason": reason,
                       "raw_output": text, "model_id": model_id,
                       "usage": {"input_tokens": usage.get("input_tokens"),
                                 "output_tokens": usage.get("output_tokens"),
                                 "reasoning_tokens": details.get("reasoning_tokens"),
                                 "latency_ms": None},
                       "provider_response_sha256": stage.digest(envelope)})
    # Unreturned IDs remain pending. Never manufacture zero observations or
    # retries while an external job may still be running.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--requests", required=True)
    exp.add_argument("--out", required=True)
    imp = sub.add_parser("normalise")
    imp.add_argument("--requests", required=True)
    imp.add_argument("--results", required=True)
    imp.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        requests = stage.read_jsonl(args.requests)
        if args.command == "export":
            groups = export_batches(requests)
            root = Path(args.out)
            root.mkdir(parents=True, exist_ok=False)
            manifest = {"status": "UNSUBMITTED_DEVELOPMENT_REQUESTS", "files": []}
            for index, (model, rows) in enumerate(sorted(groups.items())):
                name = f"model-{index+1}.jsonl"
                stage.write_new(root / name, rows, True)
                manifest["files"].append({"file": name, "model": model, "requests": len(rows),
                                          "requests_sha256": stage.digest(rows)})
            stage.write_new(root / "export-receipt.json", manifest)
            print(stage.canonical({"files": len(groups), "submitted": False}))
        else:
            rows = normalise_results(requests, stage.read_jsonl(args.results))
            stage.write_new(args.out, rows, True)
            print(stage.canonical({"normalised_rows": len(rows), "registered": False}))
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
