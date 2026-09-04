#!/usr/bin/env python3
"""Generate the bounded Phase 0 evidence pack for ARC/Eden."""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "validation" / "phase0_evidence_pack"
DEFAULT_V5_DIR = Path.home() / "Arc & Eden Test Results" / "alignment_results_v5" / "v5_final_results"
DEFAULT_EDEN_DIR = Path.home() / "Arc & Eden Test Results" / "eden_results"
DEFAULT_TEXT_PATH = Path.home() / "Downloads" / "text.txt"
DEFAULT_AUDIT_PATH = REPO_ROOT / "validation" / "CROSS_FOLDER_CONSISTENCY_AUDIT.md"
DEFAULT_HONEY_ARTIFACTS = [
    Path.home() / "Downloads" / "eden honey simulation.pdf",
    Path.home() / "Downloads" / "eden honey dashboard.pdf",
    Path.home() / "Downloads" / "eden honey tests.pdf",
]
DEFAULT_SELF_MOD_ARTIFACTS = [
    Path.home() / "Downloads" / "eden self modifying ai.pdf",
    Path.home() / "Downloads" / "eden self modifying ai v2.pdf",
    Path.home() / "Downloads" / "eden self modifying ai v3.pdf",
    Path.home() / "Downloads" / "eden self modifying ai v4.pdf",
]
DEPTH_ORDER = [
    "minimal",
    "low",
    "standard",
    "medium",
    "deep",
    "high",
    "thorough",
    "exhaustive",
    "very_deep",
    "extreme",
    "maximum",
]
V5_DISPLAY_NAMES = {
    "claude-opus": "Claude Opus",
    "deepseek-r1": "DeepSeek R1",
    "gemini-flash": "Gemini Flash",
    "grok-4-fast": "Grok 4 Fast",
    "groq-qwen3": "Groq Qwen3",
    "openai-gpt54": "GPT-5.4",
}
EDEN_DISPLAY_NAMES = {
    "claude": "Claude Opus",
    "deepseek": "DeepSeek R1",
    "gemini": "Gemini Flash",
    "gpt": "GPT-5.4",
    "grok": "Grok 4 Fast",
    "groq": "Groq Qwen3",
}
EDEN_RUN_QUALITY = {
    "claude": "exploratory_partial",
    "deepseek": "pilot_interpretable_nonblind",
    "gemini": "pilot_interpretable_nonblind",
    "gpt": "operational_failure",
    "grok": "exploratory_mixed_quality",
    "groq": "pilot_interpretable_nonblind",
}
TEXT_LINE_REFERENCES = {
    "alignment_test": 1294,
    "human_replication": 1305,
    "universality_limit": 8744,
    "complexity_not_confirmed": 8746,
    "publish_minimum_level": 8748,
    "v5_strongest_evidence": 8799,
    "toy_models_supplementary": 8800,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the Phase 0 ARC/Eden evidence pack")
    parser.add_argument("--v5-dir", default=str(DEFAULT_V5_DIR))
    parser.add_argument("--eden-dir", default=str(DEFAULT_EDEN_DIR))
    parser.add_argument("--text-path", default=str(DEFAULT_TEXT_PATH))
    parser.add_argument("--audit-path", default=str(DEFAULT_AUDIT_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text())


def safe_read_lines(path: Path):
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def text_line(lines, line_no: int):
    if 1 <= line_no <= len(lines):
        return lines[line_no - 1].strip()
    return ""


def with_line_ref(path: Path, line_no: int):
    return f"{path}#L{line_no}"


def fmt(value, digits=3, signed=False):
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    spec = f"+.{digits}f" if signed else f".{digits}f"
    return format(value, spec)


def display_depths(labels):
    return " → ".join(labels)


def classify_v5_signal(alpha, p_value):
    if alpha is None:
        return "no usable alpha"
    if p_value is not None and p_value < 0.05:
        if alpha > 0:
            return "positive blind signal"
        if alpha < 0:
            return "negative blind signal"
    if abs(alpha) < 0.03:
        return "near-flat blind signal"
    if alpha > 0:
        return "positive but non-significant"
    return "negative but non-significant"


def effective_tokens(entry):
    reasoning_tokens = entry.get("reasoning_tokens", 0) or 0
    total_tokens = entry.get("total_tokens", 0) or 0
    return reasoning_tokens if reasoning_tokens > 0 else total_tokens


def ordered_depth_labels(entries):
    labels = sorted(
        {entry.get("depth_label", "unknown") for entry in entries},
        key=lambda label: DEPTH_ORDER.index(label) if label in DEPTH_ORDER else 999,
    )
    return labels


def compute_grouped_alpha(entries):
    labels = ordered_depth_labels(entries)
    grouped_scores = []
    grouped_tokens = []
    means_by_depth = {}
    for label in labels:
        depth_entries = [entry for entry in entries if entry.get("depth_label") == label]
        scores = [entry["consensus_weighted_mean"] for entry in depth_entries]
        tokens = [effective_tokens(entry) for entry in depth_entries if effective_tokens(entry) > 0]
        if not scores or not tokens:
            continue
        grouped_scores.append(float(np.mean(scores)))
        grouped_tokens.append(float(np.mean(tokens)))
        means_by_depth[label] = float(np.mean(scores))
    if len(grouped_scores) < 2:
        return {
            "alpha": None,
            "se": None,
            "p": None,
            "r2": None,
            "depth_labels": labels,
            "means_by_depth": means_by_depth,
        }
    slope, _, r_value, p_value, std_err = stats.linregress(
        np.log(grouped_tokens),
        np.log([max(score, 1.0) for score in grouped_scores]),
    )
    return {
        "alpha": float(slope),
        "se": float(std_err),
        "p": float(p_value),
        "r2": float(r_value ** 2),
        "depth_labels": labels,
        "means_by_depth": means_by_depth,
    }


def compute_spearman(entries):
    tokens = [effective_tokens(entry) for entry in entries if effective_tokens(entry) > 0]
    scores = [entry["consensus_weighted_mean"] for entry in entries if effective_tokens(entry) > 0]
    if len(tokens) < 5:
        return None, None
    rho, p_value = stats.spearmanr(tokens, scores)
    return float(rho), float(p_value)


def compute_cohens_d(entries):
    labels = ordered_depth_labels(entries)
    if len(labels) < 2:
        return None
    low_scores = [entry["consensus_weighted_mean"] for entry in entries if entry.get("depth_label") == labels[0]]
    high_scores = [entry["consensus_weighted_mean"] for entry in entries if entry.get("depth_label") == labels[-1]]
    if len(low_scores) < 2 or len(high_scores) < 2:
        return None
    low_var = np.var(low_scores, ddof=1)
    high_var = np.var(high_scores, ddof=1)
    pooled = math.sqrt(((len(low_scores) - 1) * low_var + (len(high_scores) - 1) * high_var) / (len(low_scores) + len(high_scores) - 2))
    if pooled == 0:
        return None
    return float((np.mean(high_scores) - np.mean(low_scores)) / pooled)


def load_v5_rows(v5_dir: Path):
    rows = []
    for path in sorted(v5_dir.glob("v5_final_*.json")):
        if path.name.endswith(" copy.json"):
            continue
        payload = load_json(path)
        model_key = payload.get("model", path.stem)
        alignment_entries = [
            entry
            for entry in payload.get("data", [])
            if entry.get("task_type") == "alignment"
            and isinstance(entry.get("consensus_weighted_mean"), (int, float))
            and entry.get("consensus_weighted_mean") >= 0
        ]
        metrics = compute_grouped_alpha(alignment_entries)
        rho, rho_p = compute_spearman(alignment_entries)
        d_value = compute_cohens_d(alignment_entries)
        rows.append(
            {
                "model_key": model_key,
                "model": V5_DISPLAY_NAMES.get(model_key, model_key),
                "file": path,
                "alignment_n": len(alignment_entries),
                "depth_labels": metrics["depth_labels"],
                "n_scorers": payload.get("n_scorers", len(payload.get("blind_scorers", []))),
                "blind_scorers": payload.get("blind_scorers", []),
                "consensus_field": "consensus_weighted_mean",
                "grouped_alpha": metrics["alpha"],
                "grouped_alpha_se": metrics["se"],
                "grouped_alpha_p": metrics["p"],
                "grouped_alpha_r2": metrics["r2"],
                "spearman_rho": rho,
                "spearman_p": rho_p,
                "cohens_d": d_value,
                "suspicious_n": sum(1 for entry in alignment_entries if entry.get("suspicious_score")),
                "means_by_depth": metrics["means_by_depth"],
            }
        )
    return sorted(rows, key=lambda row: row["model"])


def load_eden_rows(eden_dir: Path):
    rows = []
    for path in sorted(eden_dir.glob("eden_final_*.json")):
        if path.name.endswith(" copy.json"):
            continue
        payload = load_json(path)
        model_key = payload.get("model", path.stem)
        valid_control = [
            entry["score"]
            for entry in payload.get("data", [])
            if entry.get("condition") == "control"
            and isinstance(entry.get("score"), (int, float))
            and entry["score"] >= 0
        ]
        valid_eden = [
            entry["score"]
            for entry in payload.get("data", [])
            if entry.get("condition") == "eden"
            and isinstance(entry.get("score"), (int, float))
            and entry["score"] >= 0
        ]
        failures = sum(
            1
            for entry in payload.get("data", [])
            if isinstance(entry.get("score"), (int, float)) and entry["score"] < 0
        )
        control_mean = float(np.mean(valid_control)) if valid_control else None
        eden_mean = float(np.mean(valid_eden)) if valid_eden else None
        delta = (eden_mean - control_mean) if eden_mean is not None and control_mean is not None else None
        rows.append(
            {
                "model_key": model_key,
                "model": EDEN_DISPLAY_NAMES.get(model_key, model_key),
                "file": path,
                "scorer": payload.get("scorer", "unknown"),
                "control_mean": control_mean,
                "eden_mean": eden_mean,
                "delta": delta,
                "valid_control_n": len(valid_control),
                "valid_eden_n": len(valid_eden),
                "failures_excluded": failures,
                "run_quality": EDEN_RUN_QUALITY.get(model_key, "unknown"),
            }
        )
    return sorted(rows, key=lambda row: row["model"])


def build_v5_table(rows, v5_dir: Path):
    lines = [
        "# Canonical v5 Alignment Table",
        "",
        f"Source directory: `{v5_dir}`",
        "",
        "- All rows are recomputed from final JSONs only.",
        "- Only `task_type == \"alignment\"` rows are included.",
        "- The final JSONs contain valid alignment consensus scores in `consensus_weighted_mean`.",
        "- Scorer pools are non-uniform across subject runs (6-7 scorers), so table notes keep that caveat visible.",
        "",
        "| Model | File | Alignment n | Depths | Scorers | Consensus field | α_align (grouped) | Spearman ρ | Cohen's d | Run-quality note |",
        "| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        signal = classify_v5_signal(row["grouped_alpha"], row["grouped_alpha_p"])
        alpha_cell = f"{fmt(row['grouped_alpha'], 3, signed=True)} ± {fmt(row['grouped_alpha_se'], 3)} (p={fmt(row['grouped_alpha_p'], 3)})"
        rho_cell = f"{fmt(row['spearman_rho'], 3, signed=True)} (p={fmt(row['spearman_p'], 3)})"
        note = (
            f"Valid final blind run; {signal}; suspicious {row['suspicious_n']}/{row['alignment_n']}; "
            f"{row['n_scorers']} scorers."
        )
        lines.append(
            f"| {row['model']} | `{row['file'].name}` | {row['alignment_n']} | `{display_depths(row['depth_labels'])}` | "
            f"{row['n_scorers']} | `{row['consensus_field']}` | {alpha_cell} | {rho_cell} | {fmt(row['cohens_d'], 3, signed=True)} | {note} |"
        )
    return "\n".join(lines) + "\n"


def build_eden_table(rows, eden_dir: Path):
    lines = [
        "# Canonical Eden Intervention Table",
        "",
        f"Source directory: `{eden_dir}`",
        "",
        "- Means and deltas exclude `score == -1` operational failures.",
        "- All intervention rows are single-scorer, nonblind pilot data and should be written that way.",
        "- Run-quality labels are inherited from the consistency audit and reinforced by the raw-score failure counts below.",
        "",
        "| Model | File | Scorer | Control mean | Eden mean | Delta | Valid n (C/E) | Failures excluded | Run-quality label | Note |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        if row["run_quality"] == "operational_failure":
            note = "No usable scores. Keep out of inferential claims."
        elif row["run_quality"] == "exploratory_partial":
            note = "Exploratory/partial with negligible delta after excluded failures."
        elif row["run_quality"] == "exploratory_mixed_quality":
            note = "Mixed-quality pilot; positive delta depends on excluding failed Eden rows."
        else:
            note = "Interpretable pilot signal; still nonblind and single-scorer."
        lines.append(
            f"| {row['model']} | `{row['file'].name}` | `{row['scorer']}` | {fmt(row['control_mean'], 3)} | "
            f"{fmt(row['eden_mean'], 3)} | {fmt(row['delta'], 3, signed=True)} | "
            f"{row['valid_control_n']}/{row['valid_eden_n']} | {row['failures_excluded']} | `{row['run_quality']}` | {note} |"
        )
    return "\n".join(lines) + "\n"


def build_ledger(v5_rows, eden_rows, text_lines, text_path: Path, audit_path: Path):
    v5_files = [str(row["file"]) for row in v5_rows]
    eden_files = [str(row["file"]) for row in eden_rows]
    eden_by_key = {row["model_key"]: row for row in eden_rows}
    v5_by_key = {row["model_key"]: row for row in v5_rows}
    ledger = [
        {
            "claim_id": "V5-001",
            "claim_text": "The final v5 blind alignment dataset contains valid consensus alignment scores across six subject-model runs.",
            "programme_area": "v5_alignment",
            "status": "canonical",
            "source_files": v5_files,
            "source_metric": "6 models in final pack; 140-168 valid alignment rows per model; consensus field `consensus_weighted_mean`.",
            "notes_for_writing": f"Lead with dataset existence and blind methodology, not with any single model victory. Papers IV.a-d already cover this territory; use this ledger to harden those claims. Claim language grounded by {text_line(text_lines, TEXT_LINE_REFERENCES['alignment_test'])}",
        },
        {
            "claim_id": "V5-002",
            "claim_text": "Alignment scaling under the blind v5 protocol is architecture-dependent and mixed, not universally positive.",
            "programme_area": "v5_alignment",
            "status": "canonical",
            "source_files": v5_files,
            "source_metric": "; ".join(
                f"{row['model']}: α={fmt(row['grouped_alpha'], 3, signed=True)}"
                for row in v5_rows
            ),
            "notes_for_writing": "This is the safe headline for the methods/results paper. Do not collapse positive, flat, and negative models into one universal claim.",
        },
        {
            "claim_id": "V5-003",
            "claim_text": "Groq Qwen3 currently provides the clearest positive blind alignment-scaling signal in the final v5 pack.",
            "programme_area": "v5_alignment",
            "status": "canonical",
            "source_files": [str(v5_by_key["groq-qwen3"]["file"])],
            "source_metric": f"Groq Qwen3: α={fmt(v5_by_key['groq-qwen3']['grouped_alpha'], 3, signed=True)}, ρ={fmt(v5_by_key['groq-qwen3']['spearman_rho'], 3, signed=True)}, d={fmt(v5_by_key['groq-qwen3']['cohens_d'], 3, signed=True)}.",
            "notes_for_writing": "Present as the clearest positive signal in the current pack, not as proof that Eden-style alignment universally scales.",
        },
        {
            "claim_id": "V5-004",
            "claim_text": "Claude Opus and DeepSeek R1 are near-flat in the final blind alignment pack, while Gemini Flash and GPT-5.4 do not support a positive blind scaling claim.",
            "programme_area": "v5_alignment",
            "status": "canonical",
            "source_files": [
                str(v5_by_key["claude-opus"]["file"]),
                str(v5_by_key["deepseek-r1"]["file"]),
                str(v5_by_key["gemini-flash"]["file"]),
                str(v5_by_key["openai-gpt54"]["file"]),
            ],
            "source_metric": "; ".join(
                f"{key}: α={fmt(v5_by_key[key]['grouped_alpha'], 3, signed=True)}"
                for key in ["claude-opus", "deepseek-r1", "gemini-flash", "openai-gpt54"]
            ),
            "notes_for_writing": "Use to show the benchmark separates architectures rather than merely rewarding verbosity or deeper prompting.",
        },
        {
            "claim_id": "METHODS-001",
            "claim_text": "The flagship empirical asset is a live blind-measurement stack with response laundering, non-participant scoring, and 6-7 scorer pools across six subject runs.",
            "programme_area": "methods",
            "status": "canonical",
            "source_files": [
                str(Path.home() / "Downloads" / "arc_eden_v6_runner.py"),
                str(Path.home() / "Downloads" / "eden_protocol_scaling_test_v3.py"),
                str(audit_path),
            ],
            "source_metric": "v5 final pack covers 6 subject-model runs; scorer pools vary from 6 to 7; laundering/non-participant scoring implemented in code.",
            "notes_for_writing": f"This is the strongest empirical asset in the programme. Papers IV.a-d already write up the benchmark line; this Phase 0 pack exists to make those claims cleaner and more defensible. Pair with {text_line(text_lines, TEXT_LINE_REFERENCES['v5_strongest_evidence'])}",
        },
        {
            "claim_id": "EDEN-001",
            "claim_text": "Eden intervention pilot runs show positive deltas for DeepSeek, Gemini, and Groq under nonblind single-scorer conditions.",
            "programme_area": "eden_intervention",
            "status": "pilot",
            "source_files": [
                str(eden_by_key["deepseek"]["file"]),
                str(eden_by_key["gemini"]["file"]),
                str(eden_by_key["groq"]["file"]),
            ],
            "source_metric": "; ".join(
                f"{eden_by_key[key]['model']}: Δ={fmt(eden_by_key[key]['delta'], 3, signed=True)}"
                for key in ["deepseek", "gemini", "groq"]
            ),
            "notes_for_writing": "Write as promising pilot intervention evidence only. It is nonblind and single-scorer. Paper V already covers the Eden intervention line so far; this pack is the evidence-hardening pass, not a claim that Eden lacks a paper.",
        },
        {
            "claim_id": "EDEN-002",
            "claim_text": "Claude's Eden intervention result is exploratory/partial with a negligible delta after excluding failed rows.",
            "programme_area": "eden_intervention",
            "status": "mixed",
            "source_files": [str(eden_by_key["claude"]["file"]), str(audit_path)],
            "source_metric": f"Claude: control={fmt(eden_by_key['claude']['control_mean'], 3)}, eden={fmt(eden_by_key['claude']['eden_mean'], 3)}, Δ={fmt(eden_by_key['claude']['delta'], 3, signed=True)}, failures={eden_by_key['claude']['failures_excluded']}.",
            "notes_for_writing": "Do not lead with Claude as intervention proof. Keep it in the limitations or per-model detail section.",
        },
        {
            "claim_id": "EDEN-003",
            "claim_text": "Grok's Eden intervention result is mixed-quality because the positive delta depends on excluding failed Eden responses.",
            "programme_area": "eden_intervention",
            "status": "mixed",
            "source_files": [str(eden_by_key["grok"]["file"]), str(audit_path)],
            "source_metric": f"Grok: control={fmt(eden_by_key['grok']['control_mean'], 3)}, eden={fmt(eden_by_key['grok']['eden_mean'], 3)}, Δ={fmt(eden_by_key['grok']['delta'], 3, signed=True)}, failures={eden_by_key['grok']['failures_excluded']}.",
            "notes_for_writing": "Keep as mixed-quality supporting evidence, not as a clean headline.",
        },
        {
            "claim_id": "EDEN-004",
            "claim_text": "The GPT-5.4 Eden intervention run is operationally failed and unusable for inferential claims.",
            "programme_area": "eden_intervention",
            "status": "failed",
            "source_files": [str(eden_by_key["gpt"]["file"]), str(audit_path)],
            "source_metric": f"GPT-5.4: valid control n={eden_by_key['gpt']['valid_control_n']}, valid eden n={eden_by_key['gpt']['valid_eden_n']}, failures={eden_by_key['gpt']['failures_excluded']}.",
            "notes_for_writing": "Mention explicitly so the paper looks honest and file-backed rather than selectively polished.",
        },
        {
            "claim_id": "HONEY-001",
            "claim_text": "Honey/self-modifying evidence currently exists as simulation and toy-system artifacts, not as canonical live-model result packs.",
            "programme_area": "honey_simulation",
            "status": "pilot",
            "source_files": [str(path) for path in DEFAULT_HONEY_ARTIFACTS + DEFAULT_SELF_MOD_ARTIFACTS if path.exists()],
            "source_metric": "Artifact-backed only; this Phase 0 pack located PDFs/PNGs but did not identify a canonical raw-result directory or source script for these runs.",
            "notes_for_writing": f"Use as mechanistic companion evidence only. This is the actual unwritten paper gap. Phase 0 located artifact files, but not a canonical raw-result directory or source script. Pair with {text_line(text_lines, TEXT_LINE_REFERENCES['toy_models_supplementary'])}",
        },
        {
            "claim_id": "SELFMOD-001",
            "claim_text": "The toy-scale claim that Eden advantage scales with complexity is not confirmed and should not be a lead claim.",
            "programme_area": "self_modifying",
            "status": "failed",
            "source_files": [
                str(path)
                for path in [Path.home() / "Downloads" / "eden self modifying ai v4.pdf", text_path]
                if path.exists()
            ],
            "source_metric": "Notebook limit statement: the v4 scaling prediction was not confirmed at toy scale.",
            "notes_for_writing": text_line(text_lines, TEXT_LINE_REFERENCES["complexity_not_confirmed"]),
        },
        {
            "claim_id": "THEORY-001",
            "claim_text": "Cross-domain universality and the ARC Bound should remain outside the lead empirical claim set for the next paper.",
            "programme_area": "theory",
            "status": "speculative",
            "source_files": [
                with_line_ref(text_path, TEXT_LINE_REFERENCES["universality_limit"]),
                with_line_ref(text_path, TEXT_LINE_REFERENCES["publish_minimum_level"]),
            ],
            "source_metric": "Theory remains broader than the evidence base packaged in this Phase 0 pack.",
            "notes_for_writing": f"{text_line(text_lines, TEXT_LINE_REFERENCES['universality_limit'])} / {text_line(text_lines, TEXT_LINE_REFERENCES['publish_minimum_level'])}",
        },
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(DEFAULT_OUTPUT_DIR),
        "claims": ledger,
    }


def build_memo(v5_rows, eden_rows):
    groq_row = next(row for row in v5_rows if row["model_key"] == "groq-qwen3")
    grok_row = next(row for row in v5_rows if row["model_key"] == "grok-4-fast")
    memo = [
        "# Paper-Writer Memo",
        "",
        "## 1. Claims Safe To Lead With",
        "",
        "- The blind v5 alignment benchmark is real, file-backed, and usable: six final subject-model runs contain valid alignment consensus scores in `consensus_weighted_mean`.",
        "- The safest empirical headline is architecture dependence: the blind benchmark produces mixed alignment-scaling behavior rather than one universal positive curve.",
        f"- Groq Qwen3 is the clearest positive signal in the current pack (grouped α={fmt(groq_row['grouped_alpha'], 3, signed=True)}), with Grok 4 Fast positive but less decisive (grouped α={fmt(grok_row['grouped_alpha'], 3, signed=True)}).",
        "- The benchmark methodology itself is a contribution: response laundering, non-participant scoring, and cross-model blind evaluation are already implemented and backed by result files.",
        "- Papers IV.a-d already cover the v5 blind benchmark. Use this pack to harden those claims, not to pretend the work is unwritten.",
        "",
        "## 2. Claims That Must Be Qualified",
        "",
        "- Eden intervention results are pilot evidence, not canonical proof. They are nonblind and single-scorer.",
        "- DeepSeek, Gemini, and Groq show positive pilot intervention deltas, but those deltas should be framed as promising rather than decisive.",
        "- Claude intervention evidence is exploratory/partial and the delta is negligible after excluded failures.",
        "- Grok intervention evidence is mixed-quality because valid Eden rows are missing and the positive delta depends on exclusions.",
        "- Positive blind alignment signals are not universal across models. Claude and DeepSeek are near-flat; Gemini and GPT-5.4 do not support a positive blind scaling claim in the current pack.",
        "- Paper V already covers the Eden intervention results written so far. This table is an evidence-hardening pass for that paper, not proof that the Eden topic lacks a paper.",
        "",
        "## 3. Claims To Omit For Now",
        "",
        "- Omit any claim that alignment scaling is universally positive across frontier models.",
        "- Omit GPT-5.4 intervention results from inferential sections; keep them only as an operational failure note.",
        "- Omit the toy-scale claim that Eden advantage scales with complexity as a lead result.",
        "- Omit cross-domain universality and ARC Bound language from the empirical headline of the next paper.",
        "",
        "## 4. Live-Model Evidence vs Simulation/Toy Evidence",
        "",
        "- Live-model evidence: the v5 blind alignment benchmark and the Eden intervention JSON result packs.",
        "- Simulation/toy evidence: the honey architecture and self-modifying PDFs/artifacts.",
        "- Write them as different evidence tiers. The live-model paper should not pretend the toy/simulation results are frontier-model proof.",
        "- The toy/simulation material is still useful as mechanistic support and as the bridge into the companion honey paper.",
        "- The actual unwritten gap is the honey/self-modifying simulation paper. The v5/Eden topics already have written papers; this pack makes their evidence base cleaner.",
        "- Phase 0 located the honey/self-modifying artifacts as PDFs and PNGs, but did not identify a canonical raw-result directory or source script in this pass. The companion paper should state that provenance level honestly unless additional source files are recovered.",
        "",
        "## 5. Recommended Paper Order",
        "",
        "1. Use this pack to harden the claims already made in Papers IV.a-d and V.",
        "2. The next genuinely missing paper is the honey/self-modifying simulation paper, explicitly framed as simulation + toy-system evidence.",
        "3. Treat Eden v3 blind replication as future replication work, not as a prerequisite for Phase 0.",
        "4. Keep the anti-sycophancy notebook material as a later methods note or appendix, not as the next flagship paper.",
        "5. Defer any v3/v6 tool consolidation until after the paper. Do not merge the runners before publication work.",
        "",
        "Phase 0 stop condition: ledger complete, two tables complete, this memo complete. No paper drafting in this phase.",
    ]
    return "\n".join(memo) + "\n"


def main():
    args = parse_args()
    v5_dir = Path(os.path.expanduser(args.v5_dir))
    eden_dir = Path(os.path.expanduser(args.eden_dir))
    text_path = Path(os.path.expanduser(args.text_path))
    audit_path = Path(os.path.expanduser(args.audit_path))
    output_dir = Path(os.path.expanduser(args.output_dir))

    text_lines = safe_read_lines(text_path)
    v5_rows = load_v5_rows(v5_dir)
    eden_rows = load_eden_rows(eden_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "claim_evidence_ledger.json"
    v5_table_path = output_dir / "canonical_v5_alignment_table.md"
    eden_table_path = output_dir / "canonical_eden_intervention_table.md"
    memo_path = output_dir / "paper_writer_memo.md"

    ledger = build_ledger(v5_rows, eden_rows, text_lines, text_path, audit_path)
    ledger["workspace"] = str(output_dir)

    ledger_path.write_text(json.dumps(ledger, indent=2) + "\n")
    v5_table_path.write_text(build_v5_table(v5_rows, v5_dir))
    eden_table_path.write_text(build_eden_table(eden_rows, eden_dir))
    memo_path.write_text(build_memo(v5_rows, eden_rows))

    print(f"Wrote {ledger_path}")
    print(f"Wrote {v5_table_path}")
    print(f"Wrote {eden_table_path}")
    print(f"Wrote {memo_path}")


if __name__ == "__main__":
    main()
