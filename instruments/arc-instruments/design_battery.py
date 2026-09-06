#!/usr/bin/env python3
"""A battery of simulations over the plausible operating range of the P5 and P16 designs.

Each row is one configuration of a design run through its complete decision procedure many times, under
the truth and under the separated alternatives, so that the charters can see where the designs hold
their operating characteristics and where they stop. Writes a JSON table and a Markdown summary. Design
calculations under stipulated noise models; never evidence about any real system.

Run: python3 design_battery.py --out <dir> [--sims 150]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os

from arc_instruments import designs as ds

P5_GRID = [
    dict(label="baseline", sims=None),
    dict(label="fewer systems (4)", n_systems=4),
    dict(label="more systems (12)", n_systems=12),
    dict(label="noisier calibration (0.30)", noise=0.30),
    dict(label="noisier trajectories (0.08)", traj_noise=0.08),
    dict(label="capability measurement error 0.25", cap_noise=0.25),
    dict(label="capability measurement error 0.50", cap_noise=0.50),
    dict(label="longer window (128)", window=128.0),
    dict(label="shorter window (8)", window=8.0),
    dict(label="alternative shift 0.15", alt_shift=0.15),
    dict(label="alternative shift 0.20", alt_shift=0.20),
    dict(label="alternative shift 0.40", alt_shift=0.40),
    dict(label="tighter band 0.05", band=0.05),
    dict(label="smaller crossed grid (3x3x3)", states=3, retentions=3, reps=3),
    dict(label="weak nuisance rate 0.10", lam=0.10),
    dict(label="true coupling 0.3", beta=0.3),
    dict(label="true coupling 0.7 (alternative shift 0.20)", beta=0.7, alt_shift=0.20),
    dict(label="noisier trajectories (0.08) with 4 replicate trajectories", traj_noise=0.08, replicates_per_holdout=4),
    dict(label="window 128 with alternative shift 0.15", window=128.0, alt_shift=0.15),
]

P16_GRID = [
    dict(label="baseline"),
    dict(label="fewer calibration systems (6)", n_calibration=6),
    dict(label="more calibration systems (24)", n_calibration=24),
    dict(label="fewer held-out systems (4)", n_holdout=4),
    dict(label="noisier margins (0.08)", margin_noise=0.08),
    dict(label="quieter margins (0.02)", margin_noise=0.02),
    dict(label="more system variation (0.10)", system_sd=0.10),
    dict(label="shallower slope (0.06)", kappa=0.06),
    dict(label="steeper slope (0.24)", kappa=0.24),
    dict(label="tighter window (half a doubling)", timing_tolerance_log=0.35),
    dict(label="persistence 3", persistence=3),
    dict(label="dense checkpoints", checkpoints=(1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128)),
    dict(label="per-system forecast sealed after 3 early checkpoints", early_checkpoints=3),
    dict(label="per-system forecast, more system variation (0.10)", early_checkpoints=3, system_sd=0.10),
    dict(label="shallower slope (0.06) with horizon 512", kappa=0.06, horizon=512.0, checkpoints=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)),
]


def run(out_dir: str, sims: int) -> dict:
    rows_p5, rows_p16, rows_pat = [], [], []
    for cfg in P5_GRID:
        kw = {k: v for k, v in cfg.items() if k not in ("label", "sims")}
        r = ds.p5_design_power(sims=sims, **kw)
        rows_p5.append({"label": cfg["label"], **{k: v for k, v in r.items() if k != "design"}})
    for cfg in P16_GRID:
        kw = {k: v for k, v in cfg.items() if k != "label"}
        r = ds.p16_design_power(sims=sims, **kw)
        rows_p16.append({"label": cfg["label"], **{k: v for k, v in r.items() if k != "design"}})
    rows_pat.append({"test": "P5 trajectory pattern, mechanism true", **ds.p5_trajectory_pattern(sims=sims, truth="mechanism")})
    rows_pat.append({"test": "P5 trajectory pattern, pure power true", **ds.p5_trajectory_pattern(sims=sims, truth="pure_power")})
    rows_pat.append({"test": "P5 trajectory pattern, noisy (0.08), mechanism true", **ds.p5_trajectory_pattern(sims=sims, truth="mechanism", traj_noise=0.08)})
    rows_pat.append({"test": "P16 intervention pattern, factor 1.25", **ds.p16_intervention_pattern(sims=sims, factor=1.25)})
    rows_pat.append({"test": "P16 intervention pattern, factor 1.5", **ds.p16_intervention_pattern(sims=sims, factor=1.5)})
    result = {"generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sims_per_row": sims,
              "p5": rows_p5, "p16": rows_p16, "patterns": rows_pat,
              "note": "Design calculations under stipulated noise models; targets are ruling 10's (false affirmative at most 0.05, detection at least 0.80)."}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "DESIGN-SIMULATION-BATTERY.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)
    lines = ["# Design simulation battery for the P5 and P16 experiments (%s, %d simulations per row)" % (result["generated_utc"], sims), "",
             "Targets (ruling 10): false affirmative at most 0.05; detection at least 0.80. A row that misses a target is a boundary of the design, not a defect of the theory.", "",
             "## P5", "", "| Configuration | Support under truth | Refute under truth | False affirmative (shifted coupling) | Detection (shifted) | Non-identification declared | Mean fitted coupling | Shift resolvable by band |", "|---|---|---|---|---|---|---|---|"]
    for r in rows_p5:
        lines.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.3f | %.2f |" % (r["label"], r["support_rate_under_truth"], r["refutation_rate_under_truth"], r["false_affirmative_rate_under_shifted_coupling"], r["detection_rate_of_shifted_coupling"], r["non_identification_declared_under_nuisance_rate"], r["mean_calibration_beta_hat"], r["coupling_shift_resolvable_by_band"]))
    lines += ["", "## P16", "", "| Configuration | Support under truth | Refute under truth | Support under null | Inconclusive under null | Non-crossing false alarms |", "|---|---|---|---|---|---|"]
    for r in rows_p16:
        lines.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f |" % (r["label"], r["support_rate_under_truth"], r["refutation_rate_under_truth"], r["support_rate_under_null_no_boundary"], r["inconclusive_rate_under_null"], r["non_crossing_control_false_alarm_rate"]))
    lines += ["", "## Patterns across arms", ""]
    for r in rows_pat:
        lines.append("- %s: %s" % (r["test"], ", ".join("%s %.2f" % (k, v) for k, v in r.items() if k not in ("test", "truth", "factor") and isinstance(v, float))))
    with open(os.path.join(out_dir, "DESIGN-SIMULATION-BATTERY.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sims", type=int, default=150)
    a = ap.parse_args()
    r = run(a.out, a.sims)
    print(json.dumps({"p5_rows": len(r["p5"]), "p16_rows": len(r["p16"]), "patterns": len(r["patterns"]), "sims": a.sims}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
