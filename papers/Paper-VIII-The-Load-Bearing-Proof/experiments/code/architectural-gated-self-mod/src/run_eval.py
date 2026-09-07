from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from metrics.acceleration import compute_series_acceleration
from metrics.logging import write_json


def load_summaries(root: Path) -> List[dict]:
    return [json.loads(p.read_text()) for p in sorted(root.glob("*_*/summary.json"))] if False else []


def find_summaries(root: Path) -> List[Path]:
    return sorted(root.glob("*/summary.json"))


def mann_whitney(a: List[float], b: List[float]) -> dict:
    if len(a) < 2 or len(b) < 2:
        return {"u": None, "p": None}
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"u": float(u), "p": float(p)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate Eden self-mod harness results")
    parser.add_argument("--results-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()

    summary_paths = find_summaries(args.results_dir)
    if not summary_paths:
        raise SystemExit(f"No summary.json files found under {args.results_dir}")

    by_condition: Dict[str, List[dict]] = defaultdict(list)
    per_run = []
    for path in summary_paths:
        data = json.loads(path.read_text())
        accel = compute_series_acceleration(data["tracks"]["capability"], data["tracks"]["compute_steps"])
        row = {
            "condition": data["condition"],
            "seed": data["seed"],
            "final_capability": float(data["tracks"]["capability"][-1]),
            "final_safety": float(data["tracks"]["safety"][-1]),
            "final_combined": float(data["tracks"]["combined"][-1]),
            "accel": accel.normalized_mean_second_delta,
            "auc_capability": accel.auc,
        }
        per_run.append(row)
        by_condition[data["condition"]].append({"summary": data, "row": row})

    comparisons = {}
    if "eden" in by_condition and "babylon" in by_condition:
        comparisons["eden_vs_babylon_final_combined"] = mann_whitney(
            [x["row"]["final_combined"] for x in by_condition["eden"]],
            [x["row"]["final_combined"] for x in by_condition["babylon"]],
        )
    if "eden" in by_condition and "static" in by_condition:
        comparisons["eden_vs_static_final_combined"] = mann_whitney(
            [x["row"]["final_combined"] for x in by_condition["eden"]],
            [x["row"]["final_combined"] for x in by_condition["static"]],
        )
    if "eden" in by_condition and "babylon" in by_condition:
        comparisons["eden_vs_babylon_acceleration"] = mann_whitney(
            [x["row"]["accel"] for x in by_condition["eden"]],
            [x["row"]["accel"] for x in by_condition["babylon"]],
        )

    out_dir = args.results_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)

    # aggregate plots
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for cond, runs in by_condition.items():
        cap = np.vstack([r["summary"]["tracks"]["capability"] for r in runs])
        saf = np.vstack([r["summary"]["tracks"]["safety"] for r in runs])
        comb = np.vstack([r["summary"]["tracks"]["combined"] for r in runs])
        x = np.arange(cap.shape[1])
        for arr, ax, title in [(cap, axes[0], "Capability"), (saf, axes[1], "Safety"), (comb, axes[2], "Combined")]:
            mean = arr.mean(axis=0)
            se = arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0]) if arr.shape[0] > 1 else np.zeros_like(mean)
            ax.plot(x, mean, label=cond)
            ax.fill_between(x, mean - se, mean + se, alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("Iteration")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "tracks.png", dpi=160)
    plt.close(fig)

    report = {
        "per_run": per_run,
        "comparisons": comparisons,
    }
    write_json(out_dir / "report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
