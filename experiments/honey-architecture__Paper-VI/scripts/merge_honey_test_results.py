#!/usr/bin/env python3
"""Merge honey API result JSONs from separate model batches."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


TEST_KEYS = [
    "test_1_alignment_scaling",
    "test_2_monitoring_removal",
    "test_3_coupling_degradation",
    "test_4_eden_intervention",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def merge_results(files: list[Path]) -> dict[str, Any]:
    merged: dict[str, Any] = {
        "metadata": {
            "experiment": "eden_honey_tests_merged",
            "author": "Codex merge utility",
            "timestamp": datetime.now(UTC).isoformat(),
            "demo_mode": False,
            "models_tested": [],
            "scorer": None,
            "merged_from": [str(path) for path in files],
        }
    }

    for test_key in TEST_KEYS:
        merged[test_key] = {}

    models_seen: list[str] = []
    scorers_seen: list[str] = []

    for path in files:
        data = load_json(path)
        metadata = data.get("metadata", {})
        if metadata.get("demo_mode"):
            raise ValueError(f"Refusing to merge demo-mode file: {path}")

        scorer = metadata.get("scorer")
        if scorer and scorer not in scorers_seen:
            scorers_seen.append(scorer)

        for model in metadata.get("models_tested", []):
            if model not in models_seen:
                models_seen.append(model)

        for test_key in TEST_KEYS:
            section = data.get(test_key, {})
            if not isinstance(section, dict):
                continue
            for model_key, payload in section.items():
                if model_key in merged[test_key]:
                    raise ValueError(
                        f"Duplicate model '{model_key}' in section '{test_key}'"
                    )
                merged[test_key][model_key] = payload

    merged["metadata"]["models_tested"] = models_seen
    merged["metadata"]["scorer"] = scorers_seen[0] if len(scorers_seen) == 1 else scorers_seen

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Input result JSON files to merge")
    parser.add_argument(
        "--output",
        required=True,
        help="Destination for the merged JSON",
    )
    args = parser.parse_args()

    input_paths = [Path(p).expanduser().resolve() for p in args.inputs]
    output_path = Path(args.output).expanduser().resolve()

    merged = merge_results(input_paths)
    output_path.write_text(json.dumps(merged, indent=2))
    print(f"saved {output_path}")
    print(f"models_tested {merged['metadata']['models_tested']}")
    print(f"scorer {merged['metadata']['scorer']}")


if __name__ == "__main__":
    main()
