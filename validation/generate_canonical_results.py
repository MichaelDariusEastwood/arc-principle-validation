#!/usr/bin/env python3
"""Build a canonical ARC/Eden results map and verification snapshots."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from scipy import stats  # type: ignore
except Exception:  # pragma: no cover
    stats = None


ROOT = Path("/Users/michaeleastwood")
PROJECT_ROOT = ROOT / "arc-principle-validation"
VALIDATION_DIR = PROJECT_ROOT / "validation"
RESULTS_ROOT = ROOT / "Arc & Eden Test Results"

EDEN_DIR = RESULTS_ROOT / "eden_results"
ALIGN_V5_DIR = RESULTS_ROOT / "alignment_results_v5"
ALIGN_TOP_DIR = RESULTS_ROOT / "alignment_results"
PAPER_II_DIR = RESULTS_ROOT / "arc_paper_ii_results"
PAPERS_DIR = PROJECT_ROOT / "paper" / "FINAL-SUITE" / "v-major"

LATEST_PAPER_FILES = {
    "executive_summary": PAPERS_DIR / "Executive-Summary-v5.html",
    "foundational": PAPERS_DIR / "Foundational-v4.html",
    "paper_ii": PAPERS_DIR / "Paper-II-v12.html",
    "paper_iii": PAPERS_DIR / "Paper-III-White-Paper-v11.html",
    "eden_engineering": PAPERS_DIR / "Eden-Engineering-v6.html",
    "eden_vision": PAPERS_DIR / "Eden-Vision-v3.html",
    "paper_v": PAPERS_DIR / "Paper-V-Stewardship-Gene-v2.html",
    "master_toc": PAPERS_DIR / "Master-Table-of-Contents-v1.html",
}

REPORT_HTML = ALIGN_TOP_DIR / "ARC_ALIGNMENT_SCALING_REPORT.html"
REPORT_HTML_DUPLICATE = ALIGN_TOP_DIR / "ARC Alignment Scaling Experiment - Live Report.html"
ARC_PAPER_HTML = VALIDATION_DIR / "ARC_PAPER.html"


def paired_effect_size(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    if variance == 0:
        return math.inf if mean != 0 else 0.0
    return mean / math.sqrt(variance)


def paired_p_value(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    if all(v == values[0] for v in values):
        return 0.0 if values[0] != 0 else 1.0
    if stats is None:
        return None
    return float(stats.ttest_rel(values, [0.0] * len(values)).pvalue)


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def fmt_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "NA"
    if math.isinf(value):
        return "inf"
    return f"{value:.{digits}f}"


@dataclass
class EdenRun:
    model: str
    path: str
    version: str
    scorer: str
    valid_pairs: int
    invalid_pairs: int
    overall_delta: float | None
    overall_d: float | None
    overall_p: float | None
    pillars: dict[str, dict[str, float | None]]
    status: str
    interpretation: str


def classify_eden_run(model: str, valid_pairs: int, invalid_pairs: int) -> tuple[str, str]:
    if valid_pairs == 0:
        return (
            "operational_failure",
            "No valid paired comparisons survived scoring; treat as failed run, not evidence.",
        )
    if invalid_pairs == 0:
        return (
            "pilot_interpretable_nonblind",
            "Cross-model single-scorer pilot data is interpretable, but not yet blind-confirmatory.",
        )
    if model == "claude":
        return (
            "exploratory_partial",
            "Exhaustive-depth rows failed; stakeholder-care signal is interesting, but the run is incomplete.",
        )
    if model == "grok":
        return (
            "exploratory_mixed_quality",
            "Operational defects distort the composite result; retain only as exploratory signal, not replication.",
        )
    return (
        "exploratory_partial",
        "Run contains enough invalid pairs to prevent clean pilot interpretation.",
    )


def summarise_eden() -> list[EdenRun]:
    runs: list[EdenRun] = []
    for path in sorted(EDEN_DIR.glob("eden_final_*.json")):
        data = json.loads(path.read_text())
        paired: dict[tuple[str, str], dict[str, dict]] = {}
        for row in data["data"]:
            paired.setdefault((row["prompt_id"], row["depth_label"]), {})[row["condition"]] = row

        deltas: list[float] = []
        pillar_deltas: dict[str, list[float]] = defaultdict(list)
        for _, cells in paired.items():
            if "eden" not in cells or "control" not in cells:
                continue
            eden = cells["eden"]
            control = cells["control"]
            if eden["score"] < 0 or control["score"] < 0:
                continue
            deltas.append(eden["score"] - control["score"])
            for pillar in eden["pillars"]:
                pillar_deltas[pillar].append(eden["pillars"][pillar] - control["pillars"][pillar])

        valid_pairs = len(deltas)
        invalid_pairs = len(paired) - valid_pairs
        status, interpretation = classify_eden_run(data["model"], valid_pairs, invalid_pairs)
        runs.append(
            EdenRun(
                model=data["model"],
                path=str(path),
                version=str(data.get("version", "")),
                scorer=str(data.get("scorer", "")),
                valid_pairs=valid_pairs,
                invalid_pairs=invalid_pairs,
                overall_delta=mean(deltas),
                overall_d=paired_effect_size(deltas),
                overall_p=paired_p_value(deltas),
                pillars={
                    pillar: {
                        "delta": mean(values),
                        "d": paired_effect_size(values),
                        "p": paired_p_value(values),
                    }
                    for pillar, values in sorted(pillar_deltas.items())
                },
                status=status,
                interpretation=interpretation,
            )
        )
    return runs


def summarise_alignment_v5() -> dict[str, dict]:
    canonical_paths = {
        "deepseek-r1": ALIGN_V5_DIR / "v5_final_deepseek-r1_20260311_211855.json",
        "gemini-flash": ALIGN_V5_DIR / "v5_final_gemini-flash_20260311_151244.json",
        "grok-4-fast": ALIGN_V5_DIR / "v5_final_grok-4-fast_20260311_200910.json",
        "groq-qwen3": ALIGN_V5_DIR / "v5_final_groq-qwen3_20260312_073302.json",
        "openai-gpt54": ALIGN_V5_DIR / "v5_final_openai-gpt54_20260311_191836.json",
        "claude-opus": ALIGN_TOP_DIR / "v5_final_claude-opus_20260312_112739.json",
    }
    summaries: dict[str, dict] = {}
    for model, path in canonical_paths.items():
        if not path.exists():
            summaries[model] = {
                "canonical_path": str(path),
                "missing": True,
                "n_scorers": None,
                "blind_scorers": [],
                "blinding_protocol": None,
                "rows_total": 0,
                "rows_alignment_non_suspicious": 0,
                "depth_means": {},
                "depth_counts": {},
            }
            continue
        data = json.loads(path.read_text())
        alignment_rows = [
            row
            for row in data["data"]
            if row.get("task_type") == "alignment"
            and row.get("prefill_condition", "none") == "none"
            and not row.get("suspicious_score")
        ]
        depth_means: dict[str, float] = {}
        depth_counts: dict[str, int] = {}
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in alignment_rows:
            grouped[row["depth_label"]].append(row["consensus_weighted_mean"])
        for depth, values in grouped.items():
            depth_means[depth] = round(sum(values) / len(values), 2)
            depth_counts[depth] = len(values)

        summaries[model] = {
            "canonical_path": str(path),
            "n_scorers": data.get("n_scorers"),
            "blind_scorers": data.get("blind_scorers", []),
            "blinding_protocol": data.get("blinding_protocol"),
            "rows_total": len(data["data"]),
            "rows_alignment_non_suspicious": len(alignment_rows),
            "depth_means": depth_means,
            "depth_counts": depth_counts,
        }
    return summaries


def summarise_paper_ii() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(PAPER_II_DIR.glob("arc_paper_ii_*.json")):
        if path.name == "arc_paper_ii_combined.json":
            continue
        data = json.loads(path.read_text())
        verdict = data.get("verdict", {})
        out[data["model"]] = {
            "canonical_path": str(path),
            "n_problems": data.get("n_problems"),
            "alpha_sequential": verdict.get("alpha_sequential"),
            "alpha_parallel": verdict.get("alpha_parallel"),
            "supports_arc_principle": verdict.get("supports_arc_principle"),
            "near_quadratic": verdict.get("near_quadratic"),
            "sequential_regression": data.get("sequential_alphas", {}).get("alpha_regression"),
            "sequential_r2": data.get("sequential_alphas", {}).get("r2"),
            "endpoint_alpha": data.get("sequential_alphas", {}).get("alpha_endpoint"),
            "notes": (
                "step_function_or_ceiling"
                if data["model"] in {"openai", "deepseek", "grok-4-fast"}
                else "continuous_or_floor"
            ),
        }
    return out


def build_paper_verification() -> dict[str, dict]:
    checks = {
        "executive_summary": {
            "must_contain": ["expanded six-model Eden suite", "Grok 4.1 Fast", "Claude Opus 4.6"],
            "must_not_contain": ["two-model pilot tested the second loop in isolation"],
        },
        "paper_iii": {
            "must_contain": ["binary step function rather than a reliable power-law fit", "five analysable runs"],
            "must_not_contain": ["GPT-5.4 confirms the ARC Principle: α_seq = 1.47"],
        },
        "paper_v": {
            "must_contain": ["five analysable model runs", "Groq Qwen3", "stakeholder care"],
            "must_not_contain": ["two-model pilot evidence with validated stakeholder care mechanism"],
        },
        "paper_iva": {
            "must_contain": ["working hypotheses", "three-tier alignment hierarchy", "entry-level"],
            "must_not_contain": ["planned as subject models in a future v6 expansion"],
        },
        "paper_ivb": {
            "must_contain": ["architecture-dependent", "all-models-as-scorers", "tier-weighted consensus"],
            "must_not_contain": ["scores from non-participant blind scorers receive higher weight than scores from subject models"],
        },
        "paper_ivc": {
            "must_contain": ["all-models-as-scorers", "all-models-as-launderers", "candidate benchmark"],
            "must_not_contain": ["few dedicated blind scorers"],
        },
        "paper_ivd": {
            "must_contain": ["in this experimental setting", "same meaning, less fingerprint", "entry-level self-exclusion"],
            "must_not_contain": ["Draft v1.0"],
        },
        "master_toc": {
            "must_contain": [
                "Manuscript timestamp priority: 8 December 2024",
                "Public book release: 6 January 2026",
                "Version 1.1",
            ],
            "must_not_contain": ["Pre-registered on OSF"],
        },
        "report_html": {
            "must_contain": ["canonical", "gold-standard", "eden_protocol_scaling_test_v3.py"],
            "must_not_contain": [
                "Chapter 49: Eden Protocol - Empirical Results (Two Models)",
                "one model needs completion (Claude Opus 4.6, 387/500)",
            ],
        },
        "report_html_duplicate": {
            "must_contain": [
                "DKIM-signed Google email on 8 December 2024",
                "blind replication runner",
                "self-excluding cross-model scoring",
            ],
            "must_not_contain": [
                "submitted a manuscript to the Open Science Framework",
                "one model needs completion (Claude Opus 4.6, 387/500)",
                "Ready to run, not yet executed",
            ],
        },
        "arc_paper_html": {
            "must_contain": ["entry-level self-excluding", "6-7 scorers per entry"],
            "must_not_contain": ["non-participant blind scorers"],
        },
    }

    resolved_paths = dict(LATEST_PAPER_FILES)
    resolved_paths["paper_iva"] = ALIGN_TOP_DIR / "Paper-IV-a-Baked-In-vs-Computed-Alignment-v1.html"
    resolved_paths["paper_ivb"] = ALIGN_TOP_DIR / "Paper-IV-b-Alignment-Saturation-at-Low-Depth-v1.html"
    resolved_paths["paper_ivc"] = ALIGN_TOP_DIR / "Paper-IV-c-ARC-Align-Benchmark-v1.html"
    resolved_paths["paper_ivd"] = ALIGN_TOP_DIR / "Paper-IV-d-The-Effect-of-Blinding-on-AI-Alignment-Evaluation-v1.html"
    resolved_paths["report_html"] = REPORT_HTML
    resolved_paths["report_html_duplicate"] = REPORT_HTML_DUPLICATE
    resolved_paths["arc_paper_html"] = ARC_PAPER_HTML

    output: dict[str, dict] = {}
    for label, spec in checks.items():
        path = resolved_paths[label]
        text = path.read_text()
        contains = {token: (token in text) for token in spec["must_contain"]}
        excludes = {token: (token not in text) for token in spec["must_not_contain"]}
        output[label] = {
            "path": str(path),
            "must_contain": contains,
            "must_not_contain": excludes,
            "pass": all(contains.values()) and all(excludes.values()),
        }
    return output


def build_cross_folder_audit(eden_runs: list[EdenRun], align_v5: dict[str, dict]) -> dict:
    duplicate_domains = {
        "alignment_v5_top_level": [str(path) for path in sorted(ALIGN_TOP_DIR.glob("v5_final_*.json"))],
        "alignment_v5_versioned": [str(path) for path in sorted(ALIGN_V5_DIR.glob("v5_final_*.json"))],
        "eden_results": [str(path) for path in sorted(EDEN_DIR.glob("eden_final_*.json"))],
        "paper_ii_results": [str(path) for path in sorted(PAPER_II_DIR.glob("arc_paper_ii_*.json"))],
    }
    scorer_counts = {model: summary.get("n_scorers") for model, summary in align_v5.items()}
    eden_status = {run.model: run.status for run in eden_runs}
    issues = [
        "v5 final raw files are split across /alignment_results_v5 and /alignment_results; Claude Opus final currently lives only in the top-level folder.",
        "v5 scorer pools are not uniform: subject runs use 6-7 blind scorers depending on subject identity and the available non-subject scorer adapters.",
        "Eden evidence must be stratified by run quality: five analysable runs show stakeholder-care gains, but the broadest composite uplift remains concentrated in Gemini and Groq; GPT-5.4 failed operationally.",
        "The running HTML report historically mixed superseded chronicle text with current headline claims; later sections should explicitly supersede earlier ones.",
    ]
    return {
        "duplicate_domains": duplicate_domains,
        "v5_scorer_counts": scorer_counts,
        "eden_statuses": eden_status,
        "known_issues": issues,
    }


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def write_markdown(
    canonical_map: dict,
    cross_folder_audit: dict,
    paper_verification: dict,
) -> None:
    md_map = VALIDATION_DIR / "CANONICAL_RESULTS_MAP.md"
    md_audit = VALIDATION_DIR / "CROSS_FOLDER_CONSISTENCY_AUDIT.md"
    md_verify = VALIDATION_DIR / "PAPER_TO_RESULTS_VERIFICATION.md"

    eden_lines = [
        "| Model | Status | Valid pairs | Invalid pairs | Overall delta | d | p | Canonical file |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in canonical_map["eden"]:
        eden_lines.append(
            "| {model} | {status} | {valid_pairs} | {invalid_pairs} | {delta} | {dval} | {pval} | `{path}` |".format(
                model=run["model"],
                status=run["status"],
                valid_pairs=run["valid_pairs"],
                invalid_pairs=run["invalid_pairs"],
                delta=fmt_float(run["overall_delta"]),
                dval=fmt_float(run["overall_d"], 3),
                pval=fmt_float(run["overall_p"], 4),
                path=run["path"],
            )
        )

    v5_lines = [
        "| Model | Blind scorers | Alignment rows | Depth means | Canonical file |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for model, summary in canonical_map["alignment_v5"].items():
        depth_bits = ", ".join(
            f"{depth}={value}" for depth, value in sorted(summary["depth_means"].items())
        )
        v5_lines.append(
            f"| {model} | {summary['n_scorers']} | {summary['rows_alignment_non_suspicious']} | {depth_bits} | `{summary['canonical_path']}` |"
        )

    paper_ii_lines = [
        "| Model | alpha_seq | alpha_parallel | regression alpha | r^2 | Notes | Canonical file |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for model, summary in canonical_map["paper_ii"].items():
        paper_ii_lines.append(
            "| {model} | {aseq} | {apar} | {areg} | {r2} | {notes} | `{path}` |".format(
                model=model,
                aseq=fmt_float(summary["alpha_sequential"], 3),
                apar=fmt_float(summary["alpha_parallel"], 3),
                areg=fmt_float(summary["sequential_regression"], 3),
                r2=fmt_float(summary["sequential_r2"], 3),
                notes=summary["notes"],
                path=summary["canonical_path"],
            )
        )

    md_map.write_text(
        "\n".join(
            [
                "# Canonical Results Map",
                "",
                "This file is the source-of-truth index for the current ARC / Eden programme as of 12 March 2026.",
                "",
                "## Eden Protocol",
                *eden_lines,
                "",
                "## Alignment v5",
                *v5_lines,
                "",
                "## Paper II Compute Scaling",
                *paper_ii_lines,
                "",
                "## Interpretation Rules",
                "",
                "- Treat `pilot_interpretable_nonblind` Eden runs as promising but non-confirmatory until canonical `arc_eden_v6` blind replication lands.",
                "- Treat `exploratory_*` Eden runs as signal-generation only.",
                "- Treat `operational_failure` runs as no evidence.",
                "- Treat the versioned or top-level file listed here as canonical even when other folders contain overlapping copies.",
                "",
            ]
        )
        + "\n"
    )

    md_audit.write_text(
        "\n".join(
            [
                "# Cross-Folder Consistency Audit",
                "",
                "## Known Issues",
                *[f"- {issue}" for issue in cross_folder_audit["known_issues"]],
                "",
                "## v5 Scorer Counts By Canonical Subject Run",
                *[f"- `{model}`: {count}" for model, count in sorted(cross_folder_audit["v5_scorer_counts"].items())],
                "",
                "## Eden Status Grid",
                *[f"- `{model}`: {status}" for model, status in sorted(cross_folder_audit["eden_statuses"].items())],
                "",
                "## Raw File Domains",
                *[
                    f"- `{label}`: {len(paths)} files"
                    for label, paths in sorted(cross_folder_audit["duplicate_domains"].items())
                ],
                "",
            ]
        )
        + "\n"
    )

    verify_lines = [
        "# Paper-to-Results Verification",
        "",
        "These checks confirm whether the latest paper/report files reflect the current canonical results map.",
        "",
        "| File | Pass | Notes |",
        "| --- | --- | --- |",
    ]
    for label, result in sorted(paper_verification.items()):
        notes: list[str] = []
        for token, present in result["must_contain"].items():
            if not present:
                notes.append(f"missing `{token}`")
        for token, absent in result["must_not_contain"].items():
            if not absent:
                notes.append(f"still contains `{token}`")
        verify_lines.append(
            f"| `{label}` | {'PASS' if result['pass'] else 'WARN'} | {'; '.join(notes) if notes else 'Aligned with checks'} |"
        )
    md_verify.write_text("\n".join(verify_lines) + "\n")


def main() -> None:
    eden_runs = summarise_eden()
    alignment_v5 = summarise_alignment_v5()
    paper_ii = summarise_paper_ii()
    cross_folder_audit = build_cross_folder_audit(eden_runs, alignment_v5)
    paper_verification = build_paper_verification()

    canonical_map = {
        "generated_at": "2026-03-12",
        "eden": [
            {
                "model": run.model,
                "path": run.path,
                "version": run.version,
                "scorer": run.scorer,
                "valid_pairs": run.valid_pairs,
                "invalid_pairs": run.invalid_pairs,
                "overall_delta": run.overall_delta,
                "overall_d": run.overall_d,
                "overall_p": run.overall_p,
                "pillars": run.pillars,
                "status": run.status,
                "interpretation": run.interpretation,
            }
            for run in eden_runs
        ],
        "alignment_v5": alignment_v5,
        "paper_ii": paper_ii,
    }

    write_json(VALIDATION_DIR / "CANONICAL_RESULTS_MAP.json", canonical_map)
    write_json(VALIDATION_DIR / "CROSS_FOLDER_CONSISTENCY_AUDIT.json", cross_folder_audit)
    write_json(VALIDATION_DIR / "PAPER_TO_RESULTS_VERIFICATION.json", paper_verification)
    write_markdown(canonical_map, cross_folder_audit, paper_verification)


if __name__ == "__main__":
    main()
