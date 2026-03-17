#!/usr/bin/env python3
"""
Build a genuinely blinded operator-classification packet for the empirical
curve-fit cohort.

The goal is to let an external assessor classify the composition operator from
system mechanics alone, without seeing the manifest's operator_class,
predicted_family, or model-hinting labels such as "logistic" or "Hill".
"""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "canonical_50_domain_manifest.json"
PACKET_JSON = REPO_ROOT / "data" / "blinded_operator_classification_packet.json"
TEMPLATE_JSON = REPO_ROOT / "data" / "blinded_operator_classification_template.json"
INSTRUCTIONS_MD = REPO_ROOT / "results" / "blinded_operator_classification_instructions.md"


BLINDED_OVERRIDES = {
    1: {
        "blind_name": "Mammalian basal metabolism vs body size",
        "system_description": "Basal metabolic rate measured across mammalian species as a function of body mass.",
    },
    2: {
        "blind_name": "Metropolitan economic output vs population size",
        "system_description": "Gross metropolitan product measured against city population across major metropolitan areas.",
    },
    3: {
        "blind_name": "Island species count vs island area",
        "system_description": "Species richness measured across islands of different physical area in one archipelago.",
    },
    4: {
        "blind_name": "Technology unit cost vs cumulative production",
        "system_description": "Unit cost measured against cumulative installed production for a manufactured technology.",
    },
    5: {
        "blind_name": "Distinct vocabulary vs corpus size",
        "system_description": "Number of distinct observed word types measured against total words processed in a text corpus.",
    },
    6: {
        "blind_name": "Corpus word frequency vs rank",
        "system_description": "Observed token frequency measured against rank order in a natural-language corpus.",
    },
    7: {
        "blind_name": "Task time vs cumulative practice",
        "system_description": "Time per unit task measured against cumulative number of prior repetitions in a skilled manual task.",
    },
    8: {
        "blind_name": "Integrated-circuit transistor count vs calendar year",
        "system_description": "Transistor count on leading semiconductor chips measured across calendar years.",
    },
    9: {
        "blind_name": "Remaining radioactive fraction vs elapsed time",
        "system_description": "Fraction of undecayed radioactive material measured against elapsed time.",
    },
    10: {
        "blind_name": "Earthquake event frequency vs magnitude",
        "system_description": "Observed event counts measured across earthquake magnitude thresholds.",
    },
    11: {
        "blind_name": "Batch-culture microbial density vs elapsed time",
        "system_description": "Microbial population density measured over time in a finite batch-culture environment.",
    },
    12: {
        "blind_name": "Binding saturation vs ligand partial pressure",
        "system_description": "Saturation level of a finite-site binding system measured against the driving ligand partial pressure.",
    },
    13: {
        "blind_name": "Cumulative epidemic cases vs elapsed time",
        "system_description": "Cumulative cases in a finite outbreak measured over time under real-world intervention and depletion effects.",
    },
    14: {
        "blind_name": "Parallel speedup vs processor count",
        "system_description": "Observed program speedup measured against the number of available processor cores.",
    },
    15: {
        "blind_name": "Muscle shortening velocity vs applied load",
        "system_description": "Measured shortening velocity of a muscle preparation as a function of external load.",
    },
    16: {
        "blind_name": "Social-platform active users vs calendar year",
        "system_description": "Monthly active users of a consumer social platform measured across calendar years.",
    },
    17: {
        "blind_name": "Mean-squared displacement vs elapsed time",
        "system_description": "Mean-squared displacement of diffusing particles measured against elapsed time.",
    },
    18: {
        "blind_name": "River-network stream counts vs stream order",
        "system_description": "Observed counts of streams measured against discrete stream-order levels in a drainage network.",
    },
    19: {
        "blind_name": "Language-model loss vs parameter count",
        "system_description": "Observed language-model loss measured against total trainable parameter count across model scales.",
    },
    20: {
        "blind_name": "Driven quantum-system order parameter vs control setting",
        "system_description": "A normalized order parameter in a driven quantum many-body system measured against an external control parameter.",
    },
    21: {
        "blind_name": "Main-sequence stellar luminosity vs stellar mass",
        "system_description": "Luminosity of main-sequence stars measured against stellar mass.",
    },
    22: {
        "blind_name": "Resting heart rate vs body mass",
        "system_description": "Resting heart rate across mammalian species measured against body mass.",
    },
    23: {
        "blind_name": "Chip external pin count vs internal gate count",
        "system_description": "Observed external pin count of chip modules measured against internal logic-gate count.",
    },
    24: {
        "blind_name": "Population variance vs population mean",
        "system_description": "Observed variance of population density measured against the corresponding mean density across samples.",
    },
    25: {
        "blind_name": "Main stream length vs drainage basin area",
        "system_description": "Longest-stream length in river basins measured against basin drainage area.",
    },
}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def build_packet(manifest: dict) -> dict:
    empirical_domains = [d for d in manifest["domains"] if d["evidence_tier"] == "empirical_curve_fit"]
    packet_domains = []
    for domain in empirical_domains:
        override = BLINDED_OVERRIDES[domain["id"]]
        packet_domains.append(
            {
                "blind_id": f"EMP-{domain['id']:02d}",
                "source_domain_id": domain["id"],
                "blind_name": override["blind_name"],
                "system_description": override["system_description"],
                "source": domain["source"],
                "notes": domain.get("notes"),
                "instructions": (
                    "Classify the operator class as multiplicative, additive, or bounded "
                    "from the physical/system mechanism only. Do not infer from expected curve shape alone."
                ),
            }
        )

    return {
        "packet_name": "Blinded operator-classification packet for Paper VII empirical cohort",
        "version": "2026-03-17",
        "purpose": (
            "Independent operator classification of the 25 empirical curve-fit domains "
            "without access to operator_class, predicted_family, or model-name hints."
        ),
        "response_schema": {
            "operator_class_allowed_values": ["multiplicative", "additive", "bounded"],
            "confidence_allowed_values": ["low", "medium", "high"],
        },
        "domains": packet_domains,
    }


def build_template(packet: dict) -> dict:
    return {
        "packet_name": packet["packet_name"],
        "version": packet["version"],
        "responses": [
            {
                "blind_id": domain["blind_id"],
                "operator_class": "",
                "confidence": "",
                "justification": "",
            }
            for domain in packet["domains"]
        ],
    }


def build_instructions(packet: dict) -> str:
    lines = [
        "# Blinded Operator Classification Instructions",
        "",
        "This packet is the canonical external-review input for Paper VII's empirical cohort.",
        "",
        "## Assessor task",
        "",
        "For each domain, assign exactly one operator class:",
        "",
        "- `multiplicative`: recursive gains compose multiplicatively across scale.",
        "- `additive`: equal additive steps in the independent variable produce equal multiplicative changes in the response.",
        "- `bounded`: the system is constrained by a carrying capacity, finite-site saturation, or hard physical ceiling.",
        "",
        "## Blinding rules",
        "",
        "- Do not inspect the canonical manifest while classifying.",
        "- Do not use the Paper VII predicted families.",
        "- Use system mechanics, not curve-fit results.",
        "- Record uncertainty when the mechanism is genuinely ambiguous.",
        "",
        "## Packet summary",
        "",
        f"- Domains: {len(packet['domains'])}",
        f"- Packet JSON: `{PACKET_JSON.name}`",
        f"- Response template: `{TEMPLATE_JSON.name}`",
        "",
        "## Submission format",
        "",
        "Return one response object per blind ID with:",
        "",
        "- `operator_class`",
        "- `confidence`",
        "- `justification`",
        "",
        "The packet intentionally neutralizes labels such as `logistic`, `Hill`, or other model-name hints.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    manifest = load_manifest()
    packet = build_packet(manifest)
    template = build_template(packet)

    PACKET_JSON.write_text(json.dumps(packet, indent=2))
    TEMPLATE_JSON.write_text(json.dumps(template, indent=2))
    INSTRUCTIONS_MD.write_text(build_instructions(packet))

    print(f"Wrote {PACKET_JSON}")
    print(f"Wrote {TEMPLATE_JSON}")
    print(f"Wrote {INSTRUCTIONS_MD}")


if __name__ == "__main__":
    main()
