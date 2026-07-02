#!/usr/bin/env python3
"""
Generate a structured miss-analysis report for the empirical curve-fit cohort.

This report is meant to separate likely framework failures from weak-domain
implementation problems such as undersampled data, mixed-regime physics, or
finite-size truncation.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "results_50_domain_validation.json"
JSON_OUT = REPO_ROOT / "results" / "empirical_miss_analysis.json"
MD_OUT = REPO_ROOT / "results" / "empirical_miss_analysis.md"


MISS_NOTES = {
    1: {
        "suspected_issue": "Small legacy dataset and wide body-mass span let flexible bounded curves overfit the upper tail.",
        "likely_driver": "Dataset choice / sample size",
        "recommended_action": (
            "Refit against a larger modern mammal compilation such as White et al. or a comparable multi-hundred-species dataset."
        ),
        "proposed_followup": (
            "Keep as a miss in the canonical suite; test whether the miss survives on an updated compendium before changing any headline claim."
        ),
    },
    3: {
        "suspected_issue": "Noisy small-island archipelago data with high scatter makes bounded alternatives competitive.",
        "likely_driver": "Noisy empirical regime",
        "recommended_action": (
            "Use a larger island system or preregister a replacement biodiversity dataset with broader area coverage and clearer sampling quality."
        ),
        "proposed_followup": (
            "Treat as an ambiguous ecological domain rather than a clean falsifier until replicated on a stronger species-area dataset."
        ),
    },
    6: {
        "suspected_issue": "Finite-corpus truncation and extreme-rank zeros collapse the fit quality and allow bounded models to win trivially.",
        "likely_driver": "Finite-size artifact / corpus truncation",
        "recommended_action": (
            "Refit on a much larger corpus and preregister the usable rank window to avoid tail-collapse artifacts."
        ),
        "proposed_followup": (
            "Re-run with a modern large corpus and an explicit mid-rank inclusion rule rather than retrofitting the current Brown Corpus result."
        ),
    },
    15: {
        "suspected_issue": "The recorded series drops to zero before a clean asymptotic hyperbolic tail is expressed, making exponential decay competitive.",
        "likely_driver": "Incomplete regime coverage",
        "recommended_action": (
            "Replace with a modern force-velocity dataset or a digitized raw series spanning the hyperbolic regime without zero-clipped tail compression."
        ),
        "proposed_followup": (
            "Keep as a miss now; the dataset likely under-resolves the exact bounded subfamily rather than disproving the bounded family itself."
        ),
    },
    20: {
        "suspected_issue": "Only four points are available, so onset behavior dominates and saturation cannot be distinguished cleanly from a power-law rise.",
        "likely_driver": "Undersampled dataset",
        "recommended_action": (
            "Obtain supplementary or follow-on time-crystal measurements with denser coverage across the control parameter."
        ),
        "proposed_followup": (
            "Classify as a weak-domain miss and do not overinterpret it until more points exist."
        ),
    },
    21: {
        "suspected_issue": "High-mass stellar scatter and mixed physical regimes make a single global power law unstable against bounded alternatives.",
        "likely_driver": "Mixed-regime physics",
        "recommended_action": (
            "Restrict to a preregistered homogeneous main-sequence subset or piecewise mass band rather than fitting one curve across mixed stellar regimes."
        ),
        "proposed_followup": (
            "Use as a case for regime splitting in the next extension, not as a retroactive patch to the canonical 25-domain cohort."
        ),
    },
}


def load_results() -> dict:
    return json.loads(RESULTS_PATH.read_text())


def build_payload(results_json: dict) -> dict:
    misses = []
    for result in results_json["results"]:
        if result.get("tier") != "empirical_curve_fit" or result.get("family_match") is not False:
            continue
        notes = MISS_NOTES[result["domain_id"]]
        misses.append(
            {
                "domain_id": result["domain_id"],
                "name": result["name"],
                "predicted_family": result["predicted_family"],
                "best_family": result["best_family"],
                "best_model": result["best_model"],
                "top_fits": result["top_fits"],
                "suspected_issue": notes["suspected_issue"],
                "likely_driver": notes["likely_driver"],
                "recommended_action": notes["recommended_action"],
                "proposed_followup": notes["proposed_followup"],
            }
        )

    return {
        "version": "2026-03-17",
        "purpose": "Structured analysis of the six empirical misses in the canonical 25-domain cohort",
        "miss_count": len(misses),
        "misses": misses,
    }


def build_markdown(payload: dict) -> str:
    lines = [
        "# Empirical Miss Analysis",
        "",
        "This report separates likely framework misses from domains where the current dataset is too weak, truncated, or regime-mixed to carry much evidential weight.",
        "",
        f"- Empirical misses in current canonical cohort: `{payload['miss_count']}`",
        "- Rule: keep every current miss as a miss in the canonical result; do not retroactively promote any domain without a preregistered follow-up dataset.",
        "",
    ]

    for miss in payload["misses"]:
        lines.extend(
            [
                f"## {miss['domain_id']}. {miss['name']}",
                "",
                f"- Predicted family: `{miss['predicted_family']}`",
                f"- Current best family: `{miss['best_family']}`",
                f"- Current best model: `{miss['best_model']}`",
                f"- Likely driver: {miss['likely_driver']}",
                f"- Suspected issue: {miss['suspected_issue']}",
                f"- Recommended action: {miss['recommended_action']}",
                f"- Follow-up posture: {miss['proposed_followup']}",
                "",
                "Top competing fits:",
                "",
            ]
        )
        for fit in miss["top_fits"][:3]:
            lines.append(
                f"- `{fit['name']}` ({fit['family']}): AICc `{fit['aicc']:.3f}`, R^2 `{fit['r2']:.4f}`"
            )
        lines.append("")

    lines.extend(
        [
            "## Recommended order of attack",
            "",
            "1. Time crystal: more points first.",
            "2. Kleiber and Zipf: stronger modern datasets first.",
            "3. Stellar mass-luminosity: preregistered regime split.",
            "4. Species-area and force-velocity: replacement-quality datasets before interpretive escalation.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    results_json = load_results()
    payload = build_payload(results_json)
    JSON_OUT.write_text(json.dumps(payload, indent=2))
    MD_OUT.write_text(build_markdown(payload))
    print(f"Wrote {JSON_OUT}")
    print(f"Wrote {MD_OUT}")


if __name__ == "__main__":
    main()
