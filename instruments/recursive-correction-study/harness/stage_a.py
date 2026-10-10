#!/usr/bin/env python3
"""Provider-independent Stage A DEVELOPMENT harness. Python 3.9+, no network.

This runs the schedule on the existing exact-arithmetic development substrate.
It is not the proposed program-repair confirmation bank, a registration client,
an inference service, or an instrument that decides an ARC proposition.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re

import assay

SCHEMA = "arc-stage-a-development/0.1"
RESPONSE_SCHEMA = "arc-stage-a-response/0.1"
FEEDBACK = ("diagnostic", "neutral_masked")
PLACEMENT = ("during_revision", "terminal_panel")
DEPTHS = (2, 4)
MAX_OUTPUT_BYTES = 4096
MAX_STORED_OUTPUT_BYTES = 1_000_000
# Up to five accepted visible strings in a terminal prompt, each expanded by
# at most sixfold JSON escaping, plus fixed task/instruction structure.
MAX_PROMPT_BYTES = 6 * 5 * MAX_OUTPUT_BYTES + 4096
HASH = re.compile(r"^[0-9a-f]{64}$")
IDENT = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")
PROMPT_VERSION = "arithmetic-schedule-development-20261010-v1"


class StudyError(ValueError):
    pass


def canonical(obj):
    return assay.canonical(obj)


def digest(obj):
    return assay.digest(obj)


def source_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def scorer_hash():
    # The complete wrapper is bound because its answer parser also matters.
    return digest({"wrapper": source_hash(),
                   "arithmetic_oracle": hashlib.sha256(Path(assay.__file__).read_bytes()).hexdigest()})


def check_id(value, label):
    if not isinstance(value, str) or not IDENT.fullmatch(value):
        raise StudyError(f"invalid {label}")


def check_hash(value, label):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise StudyError(f"invalid {label}")


def bounded_int(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise StudyError(f"{label} must be an integer from {low} to {high}")


def parse_answer(raw):
    if not isinstance(raw, str) or len(raw.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise StudyError("missing or oversized artefact")
    try:
        obj = assay.parse_json(raw)
    except (ValueError, TypeError, RecursionError) as exc:
        raise StudyError("invalid JSON artefact") from exc
    if not isinstance(obj, dict) or set(obj) != {"answer"}:
        raise StudyError("artefact must contain exactly answer")
    try:
        return assay.integer(obj["answer"], "answer")
    except ValueError as exc:
        raise StudyError("invalid bounded answer") from exc


def outcome(item, raw):
    try:
        value = parse_answer(raw)
    except StudyError:
        return "invalid", False
    return "valid", value == assay.answer(item)


def diagnostic(item, raw, feedback):
    """Both arms execute the same checker. The mask is independent of its result."""
    status, success = outcome(item, raw)
    checked = {"format": status, "check": "passes" if success else "fails"}
    if feedback == "diagnostic":
        return checked
    if feedback == "neutral_masked":
        return {"format": "withheld", "check": "withheld"}
    raise StudyError("unknown feedback level")


def visible_output(row):
    """Preserve raw bytes in custody; use a fixed unsuccessful artefact downstream."""
    if row["status"] == "model_unsuccessful":
        return "[MODEL OUTPUT DID NOT COMPLETE SUCCESSFULLY]"
    if len(row["raw_output"].encode()) > MAX_OUTPUT_BYTES:
        return "[MODEL OUTPUT EXCEEDED THE DECLARED ARTEFACT BYTE LIMIT]"
    return row["raw_output"]


def dependency_completed(row):
    return row["status"] in ("completed", "model_unsuccessful")


def make_study(panel, models, *, dataset_kind="synthetic", study_id="stage-a-development",
               replicate_ids=("r1",), order_seed=20261010, max_output_tokens=256):
    assay.validate_panel(panel)
    if dataset_kind not in ("synthetic", "pilot"):
        raise StudyError("only synthetic or pilot studies are supported")
    check_id(study_id, "study_id")
    bounded_int(order_seed, "order_seed", 0, 2**32 - 1)
    bounded_int(max_output_tokens, "max_output_tokens", 16, 4096)
    if not isinstance(models, list) or len(models) != 2:
        raise StudyError("exactly two fixed model blocks are required")
    seen = set()
    for model in models:
        fields = {"block_id", "model_id", "backend", "revision", "revision_evidence_sha256", "generation_parameters"}
        if not isinstance(model, dict) or set(model) != fields:
            raise StudyError("model fields must match the documented schema")
        check_id(model["block_id"], "model block")
        if model["block_id"] in seen:
            raise StudyError("duplicate model block")
        seen.add(model["block_id"])
        if not isinstance(model["model_id"], str) or not 1 <= len(model["model_id"]) <= 200:
            raise StudyError("missing model ID")
        if not isinstance(model["revision"], str) or not model["revision"].strip():
            raise StudyError("a declared revision is required")
        check_hash(model["revision_evidence_sha256"], "revision evidence hash")
        if model["backend"] not in ("synthetic", "openai_responses", "external_replay"):
            raise StudyError("unsupported backend")
        if (model["backend"] == "synthetic") != (dataset_kind == "synthetic"):
            raise StudyError("synthetic and model-response studies cannot be mixed")
        params = model["generation_parameters"]
        if not isinstance(params, dict) or set(params) - {"temperature", "top_p", "reasoning"}:
            raise StudyError("unsupported generation parameter")
        if "temperature" in params and "top_p" in params:
            raise StudyError("specify at most one sampling control")
        for key, value in params.items():
            if key == "reasoning":
                if not isinstance(value, dict) or set(value) != {"effort"} or value["effort"] not in (
                    "none", "minimal", "low", "medium", "high", "xhigh"):
                    raise StudyError("unsupported reasoning declaration")
            else:
                try:
                    valid = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= (2 if key == "temperature" else 1)
                except OverflowError:
                    valid = False
                if not valid:
                    raise StudyError("invalid sampling control")
    if not isinstance(replicate_ids, (tuple, list)) or not replicate_ids or len(set(replicate_ids)) != len(replicate_ids):
        raise StudyError("replicate IDs must be nonempty and unique")
    for rep in replicate_ids:
        check_id(rep, "replicate ID")
    study = {"schema": SCHEMA, "purpose": "instrument-development-only", "study_id": study_id,
             "suite_id": "arithmetic_development", "dataset_kind": dataset_kind,
             "panel": panel, "models": models, "replicate_ids": list(replicate_ids),
             "order_seed": order_seed, "max_output_tokens": max_output_tokens,
             "prompt_version": PROMPT_VERSION, "runner_sha256": source_hash(),
             "scorer_sha256": scorer_hash(), "task_independence": "not_established",
             "planned_model_calls": len(panel["items"]) * len(models) * len(replicate_ids) * 57,
             "registered": False, "confirmatory_eligible": False, "proposition_verdict": None}
    study["study_sha256"] = digest(study)
    return study


def validate_study(study):
    if not isinstance(study, dict):
        raise StudyError("study must be an object")
    try:
        expected = make_study(study["panel"], study["models"], dataset_kind=study["dataset_kind"],
                              study_id=study["study_id"], replicate_ids=study["replicate_ids"],
                              order_seed=study["order_seed"], max_output_tokens=study["max_output_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyError("invalid study or changed instrument") from exc
    if canonical(expected) != canonical(study):
        raise StudyError("study hash, scope or instrument version mismatch")
    return study


def graph(study):
    """DAG nodes contain local design labels; only request bodies go to models."""
    validate_study(study)
    nodes = []
    for item, model, rep in itertools.product(study["panel"]["items"], study["models"], study["replicate_ids"]):
        base = {"task_cluster_id": item["id"], "model_block_id": model["block_id"], "replicate_id": rep}

        def add(label, kind, artefact=None, critics=(), policy=None):
            node = {**base, "label": label, "kind": kind, "artefact": artefact,
                    "critics": list(critics), "policy": policy}
            node["call_id"] = "c-" + digest({"study": study["study_sha256"], **node})[:48]
            node["dependencies"] = ([artefact] if artefact else []) + list(critics)
            nodes.append(node)
            return node["call_id"]

        initial = add("initial", "initial")
        for feedback, placement, depth in itertools.product(FEEDBACK, PLACEMENT, DEPTHS):
            policy = {"feedback": feedback, "placement": placement, "depth": depth}
            prefix = f"{feedback}/{placement}/{depth}"
            current = initial
            if placement == "during_revision":
                for step in range(depth):
                    critic = add(f"{prefix}/critic/{step}", "critic", current, policy=policy)
                    current = add(f"{prefix}/rewrite/{step}", "rewrite", current, [critic], policy)
                add(f"{prefix}/final", "final", current, policy=policy)
            else:
                for step in range(depth):
                    current = add(f"{prefix}/rewrite/{step}", "rewrite", current, policy=policy)
                critics = [add(f"{prefix}/critic/{step}", "critic", current, policy=policy) for step in range(depth)]
                add(f"{prefix}/final", "final", current, critics, policy)
    if len(nodes) != study["planned_model_calls"] or len({n["call_id"] for n in nodes}) != len(nodes):
        raise StudyError("schedule integrity failure")
    return nodes


def request(study, node, by_id):
    item = next(i for i in study["panel"]["items"] if i["id"] == node["task_cluster_id"])
    model = next(m for m in study["models"] if m["block_id"] == node["model_block_id"])
    context = {k: item[k] for k in ("a", "b", "operation")}
    task = "Compute the stated integer operation. "
    if node["kind"] == "critic":
        raw = visible_output(by_id[node["artefact"]])
        observation = diagnostic(item, raw, node["policy"]["feedback"])
        content = {"calculation": context, "candidate": raw, "diagnostic": observation}
        prompt = "Review this candidate calculation and give concise advice for a subsequent revision. The diagnostic may be withheld. Do not request other tools.\n" + canonical(content)
    else:
        content = {"calculation": context}
        if node["artefact"]:
            content["candidate"] = visible_output(by_id[node["artefact"]])
        if node["critics"]:
            # Fixed critic order from the graph; never selected using final success.
            content["review_notes"] = [visible_output(by_id[c]) for c in node["critics"]]
        prompt = task + 'Return exactly {"answer": integer} with no other text. You may preserve a correct candidate.\n' + canonical(content)
    if len(prompt.encode()) > MAX_PROMPT_BYTES:
        raise StudyError("prompt exceeds the frozen byte cap; no silent truncation")
    body = {"model": model["model_id"], "input": prompt,
            "max_output_tokens": study["max_output_tokens"], "store": False,
            **model["generation_parameters"]}
    result = {"call_id": node["call_id"], "model_block_id": model["block_id"],
              "backend": model["backend"], "body": body}
    result["request_sha256"] = digest(result)
    return result


def validate_response(study, req, row):
    required = {"schema", "call_id", "request_sha256", "source_kind", "status", "raw_output",
                "model_id", "usage", "completion_reason", "provider_response_sha256"}
    if not isinstance(row, dict) or set(row) != required or row["schema"] != RESPONSE_SCHEMA:
        raise StudyError("unexpected response envelope")
    if row["call_id"] != req["call_id"] or row["request_sha256"] != req["request_sha256"]:
        raise StudyError("response/request mismatch")
    source = "synthetic" if study["dataset_kind"] == "synthetic" else "model_response"
    if row["source_kind"] != source:
        raise StudyError("response origin does not match study kind")
    if row["status"] not in ("completed", "model_unsuccessful", "missing", "infrastructure_failure"):
        raise StudyError("unknown response status")
    reasons = {"completed": {"completed"}, "model_unsuccessful": {"max_output_tokens", "content_filter", "refusal", "other_incomplete"},
               "missing": {"missing"}, "infrastructure_failure": {"provider_error"}}
    if row["completion_reason"] not in reasons[row["status"]]:
        raise StudyError("completion reason does not match response status")
    if not isinstance(row["raw_output"], str) or len(row["raw_output"].encode()) > MAX_STORED_OUTPUT_BYTES:
        raise StudyError("response exceeds custody limit; retain original externally and report the incident")
    if not dependency_completed(row) and row["raw_output"] != "":
        raise StudyError("absence/failure envelope must have empty model-visible output")
    if dependency_completed(row) and row["model_id"] != req["body"]["model"]:
        raise StudyError("returned model differs from the declared snapshot")
    if not dependency_completed(row) and row["model_id"] not in (None, req["body"]["model"]):
        raise StudyError("unexpected failure model ID")
    check_hash(row["provider_response_sha256"], "provider response hash")
    usage = row["usage"]
    if not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens", "reasoning_tokens", "latency_ms"}:
        raise StudyError("unexpected usage fields")
    for key, value in usage.items():
        if value is None:
            continue
        bounded_int(value, key, 0, 10**12)
    if usage["output_tokens"] is not None and usage["output_tokens"] > study["max_output_tokens"]:
        raise StudyError("observed output usage exceeds the declared cap")
    if usage["reasoning_tokens"] is not None and usage["output_tokens"] is not None and usage["reasoning_tokens"] > usage["output_tokens"]:
        raise StudyError("reasoning usage cannot exceed total output usage")


def validate_ledger(study, ledger, nodes=None):
    nodes = graph(study) if nodes is None else nodes
    if not isinstance(ledger, list):
        raise StudyError("response ledger must be a list")
    by_id = {}
    node_map = {n["call_id"]: n for n in nodes}
    for row in ledger:
        if not isinstance(row, dict) or row.get("call_id") not in node_map or row["call_id"] in by_id:
            raise StudyError("unknown or duplicate response")
        node = node_map[row["call_id"]]
        if any(d not in by_id or not dependency_completed(by_id[d]) for d in node["dependencies"]):
            raise StudyError("response supplied before successful dependencies")
        req = request(study, node, by_id)
        validate_response(study, req, row)
        by_id[row["call_id"]] = row
    return by_id


def pending(study, ledger):
    nodes = graph(study)
    by_id = validate_ledger(study, ledger, nodes)
    available = [n for n in nodes if n["call_id"] not in by_id
                 and all(d in by_id and dependency_completed(by_id[d]) for d in n["dependencies"])]
    # Each call has its own ordering key. Adding another policy does not consume
    # the random stream of the existing policy. Provider sampling is separate.
    available.sort(key=lambda n: digest([study["order_seed"], n["call_id"]]))
    return [request(study, n, by_id) for n in available]


def import_responses(study, ledger, incoming):
    ready = {r["call_id"]: r for r in pending(study, ledger)}
    if not isinstance(incoming, list):
        raise StudyError("incoming responses must be a list")
    seen = set()
    for row in incoming:
        if not isinstance(row, dict) or row.get("call_id") not in ready or row["call_id"] in seen:
            raise StudyError("unexpected, nonready or duplicate incoming response")
        validate_response(study, ready[row["call_id"]], row)
        seen.add(row["call_id"])
    result = ledger + sorted(incoming, key=lambda row: row["call_id"])
    validate_ledger(study, result)
    return result


def endpoint_records(study, ledger, *, allow_aborted=False):
    nodes = graph(study)
    by_id = validate_ledger(study, ledger, nodes)
    if pending(study, ledger) and not allow_aborted:
        raise StudyError("unresolved scheduled calls; explicitly declare an aborted run")
    node_map = {n["call_id"]: n for n in nodes}
    initial = {(n["task_cluster_id"], n["model_block_id"], n["replicate_id"]): n
               for n in nodes if n["kind"] == "initial"}
    items = {i["id"]: i for i in study["panel"]["items"]}

    def ancestors(call_id):
        found, todo = set(), [call_id]
        while todo:
            current = todo.pop()
            if current not in found:
                found.add(current)
                todo.extend(node_map[current]["dependencies"])
        return sorted(found)

    records = []
    for node in (n for n in nodes if n["kind"] == "final"):
        key = (node["task_cluster_id"], node["model_block_id"], node["replicate_id"])
        first = by_id.get(initial[key]["call_id"])
        first_raw = first["raw_output"] if first else ""
        first_success = bool(first and first["status"] == "completed" and outcome(items[key[0]], visible_output(first))[1])
        row = by_id.get(node["call_id"])
        lineage = [{"call_id": cid, "response": by_id.get(cid)} for cid in ancestors(node["call_id"])]
        if row and row["status"] == "completed":
            status, success = outcome(items[key[0]], row["raw_output"])
        elif row and row["status"] == "model_unsuccessful":
            status, success = "invalid", False
        else:
            failed = any(v["response"] and v["response"]["status"] == "infrastructure_failure" for v in lineage)
            status, success = ("infrastructure_failure" if failed else "missing"), False
        records.append({"schema": "rd-stage-a-endpoint/0.1", "study_id": study["study_id"],
                        "dataset_kind": study["dataset_kind"], "task_cluster_id": key[0],
                        "model_block_id": key[1], "replicate_id": key[2], **node["policy"],
                        "initial_artefact_sha256": hashlib.sha256(first_raw.encode()).hexdigest(),
                        "endpoint_status": status, "initial_success": first_success,
                        "final_success": bool(success), "trace_sha256": digest(lineage),
                        "scorer_sha256": study["scorer_sha256"]})
    return records


def quality_report(study, ledger, endpoints):
    initial_rows = {(r["task_cluster_id"], r["model_block_id"], r["replicate_id"]): r["initial_success"] for r in endpoints}
    n_correct = sum(initial_rows.values())
    observed_outputs = [r["usage"]["output_tokens"] for r in ledger if r["usage"]["output_tokens"] is not None]
    return {"schema": SCHEMA, "dataset_kind": study["dataset_kind"], "registered": False,
            "confirmatory_eligible": False, "proposition_verdict": None,
            "planned_model_calls": study["planned_model_calls"], "recorded_calls": len(ledger),
            "ready_unresolved_calls": len(pending(study, ledger)), "endpoint_count": len(endpoints),
            "initial_correct": n_correct, "initial_not_successful": len(initial_rows)-n_correct,
            "both_initial_outcome_states_present": 0 < n_correct < len(initial_rows),
            "endpoint_status_counts": {s: sum(r["endpoint_status"] == s for r in endpoints) for s in ("valid", "invalid", "missing", "infrastructure_failure")},
            "reported_output_tokens": sum(observed_outputs), "calls_without_output_usage": len(ledger)-len(observed_outputs),
            "completion_reason_counts": {s: sum(r["completion_reason"] == s for r in ledger) for s in ("completed", "max_output_tokens", "content_filter", "refusal", "other_incomplete", "provider_error", "missing")},
            "overlength_outputs_retained": sum(len(r["raw_output"].encode()) > MAX_OUTPUT_BYTES for r in ledger),
            "sampling_unit_warning": "Arithmetic items do not establish independent program-template clusters.",
            "scope": "Schedule, parsing and data-contract development only. No ARC exponent, alignment or registration verdict."}


def synthetic_response(study, req):
    if study["dataset_kind"] != "synthetic":
        raise StudyError("synthetic backend cannot fill a pilot ledger")
    node = next(n for n in graph(study) if n["call_id"] == req["call_id"])
    item = next(i for i in study["panel"]["items"] if i["id"] == node["task_cluster_id"])
    if node["kind"] == "critic":
        raw = "Recalculate and preserve the answer if it is correct."
    else:
        # Deliberately artificial varied outputs test both outcome states.
        correct = int(digest([req["call_id"], "fixture"])[:8], 16) % 3 != 0
        raw = canonical({"answer": assay.answer(item) + (0 if correct else 1)})
    return {"schema": RESPONSE_SCHEMA, "call_id": req["call_id"],
            "request_sha256": req["request_sha256"], "source_kind": "synthetic",
            "status": "completed", "completion_reason": "completed", "raw_output": raw, "model_id": req["body"]["model"],
            "usage": {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "latency_ms": None},
            "provider_response_sha256": digest({"fixture": True, "raw": raw, "call_id": req["call_id"]})}


def demo_study(pairs=2):
    models = [{"block_id": block, "model_id": "synthetic-" + block, "backend": "synthetic",
               "revision": "fixture-v1", "revision_evidence_sha256": digest({"fixture": block}),
               "generation_parameters": {}} for block in ("A", "B")]
    return make_study(assay.make_panel(271828, pairs), models)


def read_json(path):
    return assay.parse_json(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [assay.parse_json(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_new(path, obj, jsonl=False):
    with Path(path).open("x", encoding="utf-8") as stream:
        if jsonl:
            for row in obj:
                stream.write(canonical(row) + "\n")
        else:
            stream.write(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--out", required=True)
    demo.add_argument("--pairs", type=int, default=2)
    new = sub.add_parser("new")
    new.add_argument("--panel", required=True)
    new.add_argument("--models", required=True)
    new.add_argument("--study-id", required=True)
    new.add_argument("--out", required=True)
    for name in ("pending", "import", "endpoints"):
        p = sub.add_parser(name)
        p.add_argument("--study", required=True)
        p.add_argument("--ledger")
        p.add_argument("--out", required=True)
        if name == "import":
            p.add_argument("--incoming", required=True)
        if name == "endpoints":
            p.add_argument("--allow-aborted", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            root = Path(args.out)
            root.mkdir(parents=True, exist_ok=False)
            study, ledger = demo_study(args.pairs), []
            while True:
                batch = pending(study, ledger)
                if not batch:
                    break
                ledger = import_responses(study, ledger, [synthetic_response(study, req) for req in batch])
            endpoints = endpoint_records(study, ledger)
            write_new(root / "study.json", study)
            write_new(root / "responses.jsonl", ledger, True)
            write_new(root / "endpoints.jsonl", endpoints, True)
            write_new(root / "quality.json", quality_report(study, ledger, endpoints))
            print(canonical({"status": "synthetic_development_only", "directory": str(root), "calls": len(ledger)}))
        elif args.command == "new":
            study = make_study(read_json(args.panel), read_json(args.models), dataset_kind="pilot", study_id=args.study_id)
            write_new(args.out, study)
            print("Created an instrument-development pilot record. This is not a preregistration.")
        else:
            study = read_json(args.study)
            ledger = read_jsonl(args.ledger) if args.ledger else []
            if args.command == "pending":
                output = pending(study, ledger)
            elif args.command == "import":
                output = import_responses(study, ledger, read_jsonl(args.incoming))
            else:
                output = endpoint_records(study, ledger, allow_aborted=args.allow_aborted)
            write_new(args.out, output, True)
            print(canonical({"rows_written": len(output), "confirmatory_eligible": False}))
    except (ValueError, TypeError, KeyError, OSError, RecursionError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
