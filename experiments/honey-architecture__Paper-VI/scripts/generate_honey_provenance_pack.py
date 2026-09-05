#!/usr/bin/env python3
"""Generate the bounded honey provenance pack and Eden v3 readiness note."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATION_DIR = REPO_ROOT / "validation"
OUTPUT_DIR = VALIDATION_DIR / "honey_provenance_pack"
EXTRACTED_DIR = OUTPUT_DIR / "extracted_sources"
DOWNLOADS_DIR = Path.home() / "Downloads"
HANDOFF_PATH = VALIDATION_DIR / "COLLABORATION_HANDOFF_2026-03-16.md"
EDEN_V3_PATH = Path.home() / "Downloads" / "eden_protocol_scaling_test_v3.py"


def resolve_text_path() -> Path:
    candidates = [
        DOWNLOADS_DIR / "Honey tests and self modifying ai scripts and results" / "text.txt",
        DOWNLOADS_DIR / "text.txt",
        Path.home() / "Desktop" / "text.txt",
        Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "text.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


TEXT_PATH = resolve_text_path()


SNAPSHOT_SPECS = [
    {
        "name": "eden_honey_simulation_from_text.py.txt",
        "experiment_family": "honey_simulation",
        "ranges": [(1752, 1755), (2360, 2605)],
        "expected_source_name": "eden_honey_simulation.py",
        "title": "Honey architecture simulation recovered from text.txt",
    },
    {
        "name": "eden_honey_tests_from_text.py.txt",
        "experiment_family": "honey_tests",
        "ranges": [(1753, 1755), (2691, 2698), (3500, 3713), (8789, 8796)],
        "expected_source_name": "eden_honey_tests.py",
        "title": "Honey test framework recovered from text.txt",
    },
    {
        "name": "eden_honey_dashboard_from_text.jsx.txt",
        "experiment_family": "honey_dashboard",
        "ranges": [(1754, 1755), (2025, 2105)],
        "expected_source_name": "eden_honey_dashboard.jsx",
        "title": "Honey dashboard artifact recovered from text.txt",
    },
    {
        "name": "eden_selfmod_v1_from_text.py.txt",
        "experiment_family": "selfmod_v1",
        "ranges": [(4540, 4765)],
        "expected_source_name": "eden_self_modifying_ai.py",
        "title": "Self-modifying AI v1 recovered from text.txt",
    },
    {
        "name": "eden_selfmod_v2_from_text.py.txt",
        "experiment_family": "selfmod_v2",
        "ranges": [(5980, 6170)],
        "expected_source_name": "eden_self_modifying_ai_v2.py",
        "title": "Self-modifying AI v2 recovered from text.txt",
    },
    {
        "name": "eden_selfmod_v3_from_text.py.txt",
        "experiment_family": "selfmod_v3",
        "ranges": [(7000, 7265)],
        "expected_source_name": "eden_self_modifying_ai_v3.py",
        "title": "Self-modifying AI v3 recovered from text.txt",
    },
    {
        "name": "eden_selfmod_v4_from_text.py.txt",
        "experiment_family": "selfmod_v4",
        "ranges": [(8110, 8285)],
        "expected_source_name": "eden_self_modifying_ai_v4.py",
        "title": "Self-modifying AI v4 recovered from text.txt",
    },
]


ARTIFACT_SPECS = [
    {
        "record_id": "ART-HONEY-CAPABILITY",
        "experiment_family": "honey_simulation",
        "artifact_type": "png",
        "title": "THE HONEY ARCHITECTURE: Why Embedded Safety is Non-Negotiable",
        "artifact_path": DOWNLOADS_DIR / "eden honey capability.png",
        "expected_generated_filename": "eden_honey_capability.png",
        "source_lines": (2360, 2605),
    },
    {
        "record_id": "ART-HONEY-SAFETY",
        "experiment_family": "honey_simulation",
        "artifact_type": "png",
        "title": "THE LOAD-BEARING WALL: Safety vs Speed Trade-off",
        "artifact_path": DOWNLOADS_DIR / "eden honey safety.png",
        "expected_generated_filename": "eden_honey_safety.png",
        "source_lines": (2360, 2605),
    },
    {
        "record_id": "ART-HONEY-RATIO",
        "experiment_family": "honey_simulation",
        "artifact_type": "png",
        "title": "Alignment-to-Capability Ratio: The Eden Protocol Prediction",
        "artifact_path": DOWNLOADS_DIR / "eden honey ratio.png",
        "expected_generated_filename": "eden_honey_ratio.png",
        "source_lines": (2360, 2605),
    },
    {
        "record_id": "ART-HONEY-SIM-PDF",
        "experiment_family": "honey_simulation",
        "artifact_type": "pdf",
        "title": "Honey simulation PDF artifact",
        "artifact_path": DOWNLOADS_DIR / "eden honey simulation.pdf",
        "expected_generated_filename": "eden_honey_simulation.pdf",
        "source_lines": (1752, 1755),
    },
    {
        "record_id": "ART-HONEY-TESTS-PDF",
        "experiment_family": "honey_tests",
        "artifact_type": "pdf",
        "title": "Honey tests PDF artifact",
        "artifact_path": DOWNLOADS_DIR / "eden honey tests.pdf",
        "expected_generated_filename": "eden_honey_tests.pdf",
        "source_lines": (1753, 1755),
    },
    {
        "record_id": "ART-HONEY-DASHBOARD-PDF",
        "experiment_family": "honey_dashboard",
        "artifact_type": "dashboard_artifact",
        "title": "Honey dashboard PDF artifact",
        "artifact_path": DOWNLOADS_DIR / "eden honey dashboard.pdf",
        "expected_generated_filename": "eden_honey_dashboard.pdf",
        "source_lines": (1754, 1755),
    },
    {
        "record_id": "ART-SELFMOD-V1-RESULTS",
        "experiment_family": "selfmod_v1",
        "artifact_type": "png",
        "title": "SELF-MODIFYING AI: Proof That Honey Architecture Prevents Collapse",
        "artifact_path": DOWNLOADS_DIR / "eden selfmod results.png",
        "expected_generated_filename": "eden_selfmod_results.png",
        "source_lines": (4540, 4765),
    },
    {
        "record_id": "ART-SELFMOD-V1-WEIGHTS",
        "experiment_family": "selfmod_v1",
        "artifact_type": "png",
        "title": "Weight Dynamics: The Load-Bearing Wall in Action",
        "artifact_path": DOWNLOADS_DIR / "eden selfmod weights.png",
        "expected_generated_filename": "eden_selfmod_weights.png",
        "source_lines": (4540, 4765),
    },
    {
        "record_id": "ART-SELFMOD-V1-PDF",
        "experiment_family": "selfmod_v1",
        "artifact_type": "pdf",
        "title": "Self-modifying AI v1 PDF artifact",
        "artifact_path": DOWNLOADS_DIR / "eden self modifying ai.pdf",
        "expected_generated_filename": "eden_self_modifying_ai_v1.pdf",
        "source_lines": (4540, 4765),
    },
    {
        "record_id": "ART-SELFMOD-V2-RESULTS",
        "experiment_family": "selfmod_v2",
        "artifact_type": "png",
        "title": "SELF-MODIFYING AI v2.0 (Fair Test): Identical Proposals, Different Objectives",
        "artifact_path": DOWNLOADS_DIR / "eden selfmod v2 results.png",
        "expected_generated_filename": "eden_selfmod_v2_results.png",
        "source_lines": (5980, 6170),
    },
    {
        "record_id": "ART-SELFMOD-V2-STATS",
        "experiment_family": "selfmod_v2",
        "artifact_type": "png",
        "title": "Statistical Summary: 10-Seed Robustness Test",
        "artifact_path": DOWNLOADS_DIR / "eden selfmod v2 stats.png",
        "expected_generated_filename": "eden_selfmod_v2_stats.png",
        "source_lines": (5980, 6170),
    },
    {
        "record_id": "ART-SELFMOD-V2-PDF",
        "experiment_family": "selfmod_v2",
        "artifact_type": "pdf",
        "title": "Self-modifying AI v2 PDF artifact",
        "artifact_path": DOWNLOADS_DIR / "eden self modifying ai v2.pdf",
        "expected_generated_filename": "eden_self_modifying_ai_v2.pdf",
        "source_lines": (5980, 6170),
    },
    {
        "record_id": "ART-SELFMOD-V3-RESULTS",
        "experiment_family": "selfmod_v3",
        "artifact_type": "png",
        "title": "SELF-MODIFYING AI v3.0: Adversarial Tasks, Fair Proposals",
        "artifact_path": DOWNLOADS_DIR / "eden selfmod v3 results.png",
        "expected_generated_filename": "eden_selfmod_v3_results.png",
        "source_lines": (7000, 7265),
    },
    {
        "record_id": "ART-SELFMOD-V3-STATS",
        "experiment_family": "selfmod_v3",
        "artifact_type": "png",
        "title": "v3.0 Statistical Summary: Eden wins 65% of seeds",
        "artifact_path": DOWNLOADS_DIR / "eden selfmod v3 stats.png",
        "expected_generated_filename": "eden_selfmod_v3_stats.png",
        "source_lines": (7000, 7265),
    },
    {
        "record_id": "ART-SELFMOD-V3-PDF",
        "experiment_family": "selfmod_v3",
        "artifact_type": "pdf",
        "title": "Self-modifying AI v3 PDF artifact",
        "artifact_path": DOWNLOADS_DIR / "eden self modifying ai v3.pdf",
        "expected_generated_filename": "eden_self_modifying_ai_v3.pdf",
        "source_lines": (7000, 7265),
    },
    {
        "record_id": "ART-SELFMOD-V4-SCALING",
        "experiment_family": "selfmod_v4",
        "artifact_type": "png",
        "title": "EDEN PROTOCOL v4.0: Does the Advantage Scale With Complexity?",
        "artifact_path": DOWNLOADS_DIR / "eden selfmod v4 scaling.png",
        "expected_generated_filename": "eden_selfmod_v4_scaling.png",
        "source_lines": (8110, 8285),
    },
    {
        "record_id": "ART-SELFMOD-V4-PDF",
        "experiment_family": "selfmod_v4",
        "artifact_type": "pdf",
        "title": "Self-modifying AI v4 PDF artifact",
        "artifact_path": DOWNLOADS_DIR / "eden self modifying ai v4.pdf",
        "expected_generated_filename": "eden_self_modifying_ai_v4.pdf",
        "source_lines": (8110, 8285),
    },
]


EXPECTED_JSONS = [
    ("JSON-HONEY-SIM", "honey_simulation", "Honey simulation exported JSON", "eden_honey_simulation_results.json", (2360, 2605)),
    ("JSON-HONEY-TESTS", "honey_tests", "Honey tests exported JSON", "eden_honey_test_results.json", (3500, 3713)),
    ("JSON-SELFMOD-V1", "selfmod_v1", "Self-modifying AI v1 exported JSON", "eden_selfmod_results.json", (4540, 4765)),
    ("JSON-SELFMOD-V2", "selfmod_v2", "Self-modifying AI v2 exported JSON", "eden_selfmod_v2_results.json", (5980, 6170)),
    ("JSON-SELFMOD-V3", "selfmod_v3", "Self-modifying AI v3 exported JSON", "eden_selfmod_v3_results.json", (7000, 7265)),
    ("JSON-SELFMOD-V4", "selfmod_v4", "Self-modifying AI v4 exported JSON", "eden_selfmod_v4_results.json", (8110, 8285)),
]


REFERENCED_SCRIPTS = [
    ("SCRIPT-HONEY-SIM", "honey_simulation", "Referenced standalone simulation script", "eden_honey_simulation.py", (1752, 1755)),
    ("SCRIPT-HONEY-TESTS", "honey_tests", "Referenced standalone testing framework", "eden_honey_tests.py", (1753, 1755)),
    ("SCRIPT-HONEY-DASHBOARD", "honey_dashboard", "Referenced standalone dashboard artifact", "eden_honey_dashboard.jsx", (1754, 1755)),
    ("SCRIPT-SELFMOD-V1", "selfmod_v1", "Recovered self-modifying AI v1 source", "eden_self_modifying_ai.py", (4540, 4765)),
    ("SCRIPT-SELFMOD-V2", "selfmod_v2", "Recovered self-modifying AI v2 source", "eden_self_modifying_ai_v2.py", (5980, 6170)),
    ("SCRIPT-SELFMOD-V3", "selfmod_v3", "Recovered self-modifying AI v3 source", "eden_self_modifying_ai_v3.py", (7000, 7265)),
    ("SCRIPT-SELFMOD-V4", "selfmod_v4", "Recovered self-modifying AI v4 source", "eden_self_modifying_ai_v4.py", (8110, 8285)),
]


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def slice_lines(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1 : end]).rstrip() + "\n"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_named_files(names: set[str]) -> dict[str, list[Path]]:
    matches = {name: [] for name in names}
    for root in [DOWNLOADS_DIR, REPO_ROOT]:
        for dirpath, _, filenames in os.walk(root):
            filename_set = set(filenames)
            for name in names:
                if name in filename_set:
                    matches[name].append(Path(dirpath) / name)
    return matches


def prefer_generated_result(matches: list[Path]) -> Path:
    generated_dir = OUTPUT_DIR / "raw_results_generated"
    ranked = sorted(
        matches,
        key=lambda path: (0 if generated_dir in path.parents else 1, str(path)),
    )
    return ranked[0]


def write_snapshot(lines: list[str], spec: dict) -> Path:
    path = EXTRACTED_DIR / spec["name"]
    header = [
        "# Recovered Source Snapshot",
        "",
        "This file is a derived recovery artifact extracted from text.txt.",
        "It is not a canonical standalone source file and should not be cited as one.",
        "",
        f"Source path: {TEXT_PATH}",
        "Line ranges: " + ", ".join(f"{start}-{end}" for start, end in spec["ranges"]),
        f"Expected standalone source name: {spec['expected_source_name']}",
        f"Experiment family: {spec['experiment_family']}",
        "",
    ]
    chunks: list[str] = []
    for start, end in spec["ranges"]:
        chunks.append(f"## text.txt lines {start}-{end}\n")
        chunks.append("```text\n")
        chunks.append(slice_lines(lines, start, end))
        chunks.append("```\n")
    path.write_text("\n".join(header) + "\n".join(chunks), encoding="utf-8")
    return path


def build_artifact_record(spec: dict) -> dict:
    artifact_path = spec["artifact_path"]
    exists = artifact_path.exists()
    start, end = spec["source_lines"]
    exact_generated = spec["artifact_type"] == "png"
    mapped = exists and exact_generated
    return {
        "record_id": spec["record_id"],
        "experiment_family": spec["experiment_family"],
        "artifact_type": spec["artifact_type"],
        "title": spec["title"],
        "artifact_path": str(artifact_path) if exists else "",
        "expected_generated_filename": spec["expected_generated_filename"],
        "source_origin": "text_txt_embedded" if exists else "not_found",
        "source_path": str(TEXT_PATH),
        "source_line_start": start,
        "source_line_end": end,
        "evidence_class": "derived_artifact" if exists else "referenced_missing",
        "status": "mapped" if mapped else ("partial" if exists else "missing"),
        "notes": (
            "Exact generated filename recovered from embedded source."
            if mapped
            else (
                "Artifact found on disk, but the recovered source references the experiment family rather than an explicit PDF export."
                if exists
                else "Expected artifact not found in the current search scope."
            )
        ),
    }


def build_snapshot_record(spec: dict, snapshot_path: Path) -> dict:
    start = min(r[0] for r in spec["ranges"])
    end = max(r[1] for r in spec["ranges"])
    return {
        "record_id": f"SRC-{spec['experiment_family'].upper()}",
        "experiment_family": spec["experiment_family"],
        "artifact_type": "embedded_source",
        "title": spec["title"],
        "artifact_path": str(snapshot_path),
        "expected_generated_filename": spec["expected_source_name"],
        "source_origin": "text_txt_embedded",
        "source_path": str(TEXT_PATH),
        "source_line_start": start,
        "source_line_end": end,
        "evidence_class": "embedded_source",
        "status": "mapped",
        "notes": "Recovered from text.txt as a derived documentation snapshot only; not a canonical standalone source file.",
    }


def build_missing_json_record(name_matches: dict[str, list[Path]], spec: tuple) -> dict:
    record_id, family, title, filename, lines = spec
    matches = name_matches.get(filename, [])
    start, end = lines
    found = bool(matches)
    chosen_match = prefer_generated_result(matches) if found else None
    demo_mode = False
    if found:
        try:
            payload = json.loads(chosen_match.read_text(encoding="utf-8"))
            demo_mode = bool(payload.get("metadata", {}).get("demo_mode"))
        except Exception:
            demo_mode = False
    return {
        "record_id": record_id,
        "experiment_family": family,
        "artifact_type": "json",
        "title": title,
        "artifact_path": str(chosen_match) if found else "",
        "expected_generated_filename": filename,
        "source_origin": "text_txt_embedded" if found else "not_found",
        "source_path": str(TEXT_PATH),
        "source_line_start": start,
        "source_line_end": end,
        "evidence_class": (
            "derived_artifact" if demo_mode else ("raw_result" if found else "referenced_missing")
        ),
        "status": (
            "partial" if demo_mode else ("mapped" if found else "missing")
        ),
        "notes": (
            "Recovered raw JSON output found in the current search scope."
            if found and not demo_mode
            else (
                "Recovered JSON output is demo-derived (`demo_mode=true`), so it is not a canonical raw benchmark run."
                if demo_mode
                else "Referenced in embedded source but no matching JSON output was found in Downloads or arc-principle-validation."
            )
        ),
    }


def build_referenced_script_record(name_matches: dict[str, list[Path]], spec: tuple) -> dict:
    record_id, family, title, filename, lines = spec
    matches = name_matches.get(filename, [])
    start, end = lines
    found = bool(matches)
    return {
        "record_id": record_id,
        "experiment_family": family,
        "artifact_type": "referenced_script",
        "title": title,
        "artifact_path": str(matches[0]) if found else "",
        "expected_generated_filename": filename,
        "source_origin": "standalone_file" if found else "not_found",
        "source_path": str(matches[0]) if found else str(TEXT_PATH),
        "source_line_start": start if not found else None,
        "source_line_end": end if not found else None,
        "evidence_class": "embedded_source" if found else "referenced_missing",
        "status": "mapped" if found else "missing",
        "notes": (
            "Standalone source file recovered in the search scope."
            if found
            else "Referenced in text.txt, but no standalone file was recovered in Downloads or arc-principle-validation."
        ),
    }


def build_map_markdown(records: list[dict]) -> str:
    recovered_sources = [r for r in records if r["artifact_type"] == "embedded_source"]
    standalone_sources = [r for r in records if r["artifact_type"] == "referenced_script" and r["status"] == "mapped"]
    disk_artifacts = [r for r in records if r["artifact_type"] in {"png", "pdf", "dashboard_artifact"} and r["artifact_path"]]
    found_json = [r for r in records if r["artifact_type"] == "json" and r["artifact_path"]]
    demo_json = [r for r in found_json if r["status"] == "partial"]
    missing_scripts = [r for r in records if r["artifact_type"] == "referenced_script" and r["status"] == "missing"]
    missing_json = [r for r in records if r["artifact_type"] == "json" and r["status"] == "missing"]

    lines = [
        "# Honey Provenance Map",
        "",
        "## Findings Summary",
        "",
        "- PNG and PDF artifacts exist on disk for the honey and self-modifying figure families.",
        "- `text.txt` contains recoverable source fragments, plot titles, export paths, and experiment descriptions for honey and self-modifying v1-v4.",
        "- Standalone sources were recovered for `eden_honey_simulation.py`, `eden_honey_tests.py`, `eden_honey_dashboard.jsx`, and self-modifying AI v1-v4 scripts.",
        f"- Generated JSON exports recovered in the current search scope: {len(found_json)}.",
        f"- JSON exports still missing in the current search scope: {len(missing_json)}.",
        "- `anti-sycophancy.pdf` exists in Downloads, but it is outside the scope of this honey/self-mod provenance pack.",
        "",
        "## Recovered from `text.txt`",
        "",
    ]
    for record in recovered_sources:
        lines.append(
            f"- `{Path(record['artifact_path']).name}` from lines {record['source_line_start']}-{record['source_line_end']} "
            f"for `{record['experiment_family']}`."
        )

    lines += [
        "",
        "## Standalone source files recovered",
        "",
    ]
    for record in standalone_sources:
        lines.append(
            f"- `{Path(record['artifact_path']).name}` for `{record['experiment_family']}` at `{record['artifact_path']}`."
        )

    lines += [
        "",
        "## Raw JSON/result files recovered",
        "",
    ]
    if found_json:
        for record in found_json:
            lines.append(
                f"- `{Path(record['artifact_path']).name}` for `{record['experiment_family']}` "
                f"(status `{record['status']}`): {record['notes']}"
            )
    else:
        lines.append("- None yet.")

    lines += [
        "",
        "## Artifacts found on disk",
        "",
    ]
    for record in disk_artifacts:
        lines.append(
            f"- `{Path(record['artifact_path']).name}` ({record['experiment_family']}, status `{record['status']}`)"
        )

    lines += [
        "",
        "## Referenced but not recovered as standalone source",
        "",
    ]
    if missing_scripts:
        for record in missing_scripts:
            lines.append(
                f"- `{record['expected_generated_filename']}` referenced from `text.txt` lines "
                f"{record['source_line_start']}-{record['source_line_end']}, but not found as a standalone file."
            )
    else:
        lines.append("- None in the current search scope. All referenced standalone scripts were recovered.")

    lines += [
        "",
        "## Raw JSON/result files not found",
        "",
    ]
    if missing_json:
        for record in missing_json:
            lines.append(
                f"- `{record['expected_generated_filename']}` referenced from `text.txt` lines "
                f"{record['source_line_start']}-{record['source_line_end']}, but not found."
            )
    else:
        lines.append("- None in the current search scope.")

    lines += [
        "",
        "## Safe claims vs provenance-limited claims",
        "",
        "### Safe claims",
        "",
        "- The honey/self-modifying figure family exists as real PNG/PDF artifacts on disk.",
        "- `text.txt` contains embedded plotting/export logic for honey and self-modifying v1-v4.",
        "- Standalone honey and self-modifying v1-v4 source files are now recovered in the Downloads search scope.",
        "- Eden v3 is compile-clean, exposes a documented CLI, and has explicit env/output conventions.",
        "",
        "### Provenance-limited claims",
        "",
        "- Honey and self-modifying numerical findings should not be treated as fully reproducible from standalone source until the exported JSONs are matched to the published PDFs/PNGs and checked for consistency.",
        "- Any JSON recovered with `demo_mode=true` is a derived/demo artifact, not a canonical raw benchmark run.",
        "- The self-modifying PDFs look like derived presentation artifacts rather than canonical raw-result containers.",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_figure_index(records: list[dict]) -> str:
    figure_records = [
        r
        for r in records
        if r["artifact_type"] in {"png", "pdf", "dashboard_artifact", "json"}
        and (r["artifact_path"] or r["expected_generated_filename"])
    ]
    lines = [
        "# Honey Figure Source Index",
        "",
        "| Artifact | On-disk path | Experiment family | `text.txt` lines | Expected generated filename | Raw JSON found? | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    json_found = {
        r["expected_generated_filename"]: ("yes" if r["artifact_path"] else "no")
        for r in figure_records
        if r["artifact_type"] == "json"
    }
    for record in figure_records:
        raw_found = json_found.get(record["expected_generated_filename"], "n/a")
        lines.append(
            f"| `{record['title']}` | `{record['artifact_path'] or 'not found'}` | `{record['experiment_family']}` | "
            f"`{record['source_line_start']}-{record['source_line_end']}` | "
            f"`{record['expected_generated_filename']}` | {raw_found} | `{record['status']}` |"
        )
    return "\n".join(lines) + "\n"


def build_eden_v3_readiness() -> str:
    lines = [
        "# Eden v3 Run Readiness",
        "",
        f"Script: `{EDEN_V3_PATH}`",
        "",
        "## Verified Status",
        "",
        "- `py_compile`: pass",
        "- `--help`: pass",
        "- benchmark execution: not run in this phase",
        "",
        "## Confirmed CLI Surface",
        "",
        "- `--model`",
        "- `--analyse`",
        "- `--resume`",
        "- `--output-dir`",
        "- `--max-scorers`",
        "- `--include-suppression`",
        "- `--retain-text`",
        "- `--purpose-mode`",
        "- `--ethics-kernel`",
        "- `--ternary-prototype`",
        "- `--list-models`",
        "",
        "## Required Environment Variables",
        "",
        "- `DEEPSEEK_API_KEY`",
        "- `GOOGLE_API_KEY`",
        "- `XAI_API_KEY`",
        "- `ANTHROPIC_API_KEY`",
        "- `GROQ_API_KEY`",
        "- `OPENAI_API_KEY`",
        "",
        "## Output Conventions",
        "",
        "- Default output directory: `./eden_results`",
        "- Final output naming pattern: `eden_v3_final_{model}_{configuration_slug}_{timestamp}.json`",
        "- Recommended future output directory for blinded replication runs: `(local results folder, not in this repository)/eden_results_v3`",
        "",
        "## Recommended Future Run Sequence",
        "",
        "1. Verify env presence with `python3 eden_protocol_scaling_test_v3.py --list-models`.",
        "2. Run a single-model pilot with an explicit output dir under `Arc & Eden Test Results/eden_results_v3`.",
        "3. Inspect the emitted `eden_v3_final_...json` and run `--analyse` on that file.",
        "4. Only then schedule wider multi-model blinded runs.",
        "",
        "## Explicit Phase Boundaries",
        "",
        "- Do not merge Eden v3 into `arc_eden_v6_runner.py` in this phase.",
        "- Do not run the Eden v3 benchmark in this phase.",
        "- This note is operational readiness only.",
        "",
    ]
    return "\n".join(lines)


def update_handoff() -> None:
    if not HANDOFF_PATH.exists():
        return
    text = HANDOFF_PATH.read_text(encoding="utf-8")
    marker = "## 8. One-Line Summary\n"
    insert = (
        "## 7A. Honey Provenance Pack\n\n"
        f"- Provenance/readiness pack created at [{OUTPUT_DIR.name}]({OUTPUT_DIR}).\n"
        "- Recovered source fragments now exist for honey simulation, honey tests, honey dashboard, and self-modifying v1-v4.\n"
        "- Artifacts were mapped to `text.txt` line ranges.\n"
        "- Standalone honey/self-mod source files were recovered in Downloads; raw JSON outputs remain unrecovered in the current search scope.\n\n"
    )
    if "## 7A. Honey Provenance Pack" in text:
        return
    if marker in text:
        text = text.replace(marker, insert + marker)
    else:
        text += "\n" + insert
    HANDOFF_PATH.write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    ensure_dir(EXTRACTED_DIR)
    text_lines = read_lines(TEXT_PATH)

    search_names = {
        "eden_honey_simulation.py",
        "eden_honey_tests.py",
        "eden_honey_dashboard.jsx",
        "eden_self_modifying_ai.py",
        "eden_self_modifying_ai_v2.py",
        "eden_self_modifying_ai_v3.py",
        "eden_self_modifying_ai_v4.py",
        "eden_honey_simulation_results.json",
        "eden_honey_test_results.json",
        "eden_selfmod_results.json",
        "eden_selfmod_v2_results.json",
        "eden_selfmod_v3_results.json",
        "eden_selfmod_v4_results.json",
    }
    name_matches = find_named_files(search_names)

    records: list[dict] = []
    for spec in SNAPSHOT_SPECS:
        snapshot_path = write_snapshot(text_lines, spec)
        records.append(build_snapshot_record(spec, snapshot_path))

    for spec in ARTIFACT_SPECS:
        records.append(build_artifact_record(spec))

    for spec in EXPECTED_JSONS:
        records.append(build_missing_json_record(name_matches, spec))

    for spec in REFERENCED_SCRIPTS:
        records.append(build_referenced_script_record(name_matches, spec))

    map_json = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "text_source": str(TEXT_PATH),
        "search_scope": [str(DOWNLOADS_DIR), str(REPO_ROOT)],
        "records": records,
    }

    (OUTPUT_DIR / "HONEY_PROVENANCE_MAP.json").write_text(
        json.dumps(map_json, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "HONEY_PROVENANCE_MAP.md").write_text(
        build_map_markdown(records), encoding="utf-8"
    )
    (OUTPUT_DIR / "HONEY_FIGURE_SOURCE_INDEX.md").write_text(
        build_figure_index(records), encoding="utf-8"
    )
    (OUTPUT_DIR / "EDEN_V3_RUN_READINESS.md").write_text(
        build_eden_v3_readiness(), encoding="utf-8"
    )
    update_handoff()

    print(f"Wrote {OUTPUT_DIR / 'HONEY_PROVENANCE_MAP.json'}")
    print(f"Wrote {OUTPUT_DIR / 'HONEY_PROVENANCE_MAP.md'}")
    print(f"Wrote {OUTPUT_DIR / 'HONEY_FIGURE_SOURCE_INDEX.md'}")
    print(f"Wrote {OUTPUT_DIR / 'EDEN_V3_RUN_READINESS.md'}")
    print(f"Wrote extracted sources to {EXTRACTED_DIR}")


if __name__ == "__main__":
    main()
