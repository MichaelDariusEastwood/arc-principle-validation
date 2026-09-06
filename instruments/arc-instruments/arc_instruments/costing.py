"""Compute costing for the experimental programme: a cheap first run and an intensive run.

Every number is a design assumption made visible so that it can be replaced: the call structure comes
from the P5 and P16 designs and the P1 titration; the token sizes per call are stated defaults; the
prices are placeholders in dollars per million tokens (input, output) to be replaced by the vendors'
current price sheets on the day of the run. The invariant is the token count; the cost is the token
count times whatever the prices are. Human rating for the scorer channel is not compute and is listed
separately. Cached-input discounts are ignored, which makes the input side conservative.
"""
from __future__ import annotations

from typing import Dict

# Assumed token sizes per call (input, output).
TOKENS = {
    "generation": (6000, 3000),     # one revision round: base prompt, retained prior output, task context; the artefact out
    "evaluation": (1200, 400),      # one checkable task from the capability battery
    "burden_assay": (5000, 1000),   # masked reference procedure over an artefact at a checkpoint
    "load_test": (5000, 2000),      # one offered-load batch to the corrector at a checkpoint
    "judge": (2500, 300),           # one judged scoring of one answer under one label condition
}

# Placeholder prices in dollars per million tokens (input, output); replace with the vendors' sheets.
PRICES = {
    "open_weights_selfhosted": (0.30, 0.90),
    "budget_api": (0.50, 2.00),
    "frontier_mid": (3.00, 15.00),
    "frontier_top": (15.00, 75.00),
}

CHEAP = dict(
    n_systems=3, states=3, retentions=3, reps=3, cal_rounds=3, second_route_fraction=0.5, negative_control_fraction=0.3,
    holdout_systems=3, replicates=2, window=32, checkpoints=7, eval_tasks=30, breadth_arm=True, repetition=False,
    p16_cal=4, p16_hold=3, p16_controls=2, p16_intervention_pairs=2, horizon=32, p16_checkpoints=6, load_levels=3, burden_calls=2,
    stageA_bank=300, judge_families=3, label_conditions=3, sessions=2,
)

INTENSIVE = dict(
    n_systems=8, states=5, retentions=5, reps=5, cal_rounds=4, second_route_fraction=1.0, negative_control_fraction=0.5,
    holdout_systems=8, replicates=4, window=128, checkpoints=9, eval_tasks=100, breadth_arm=True, repetition=True,
    p16_cal=12, p16_hold=8, p16_controls=4, p16_intervention_pairs=8, horizon=128, p16_checkpoints=8, load_levels=5, burden_calls=2,
    stageA_bank=1000, judge_families=3, label_conditions=3, sessions=2,
)


def call_counts(p: Dict[str, object]) -> Dict[str, Dict[str, int]]:
    """Calls per stage and call type, from the designs."""
    runs = p["states"] * p["retentions"] * p["reps"]
    blocks = 1.0 + p["second_route_fraction"] + p["negative_control_fraction"]
    p5_cal_gen = int(p["n_systems"] * runs * p["cal_rounds"] * blocks)
    p5_cal_eval = int(p["n_systems"] * runs * 2 * p["eval_tasks"] * blocks)
    hold_traj = p["holdout_systems"] * p["replicates"] * (2 if p["repetition"] else 1)
    arms = 2 if p["breadth_arm"] else 1
    p5_hold_gen = int(hold_traj * p["window"] * arms)
    p5_hold_eval = int(hold_traj * p["checkpoints"] * p["eval_tasks"] * arms)
    p16_traj = p["p16_cal"] + (p["p16_hold"] + p["p16_controls"] + p["p16_intervention_pairs"]) * (2 if p["repetition"] else 1)
    p16_gen = int(p16_traj * p["horizon"])
    p16_eval = int(p16_traj * p["p16_checkpoints"] * p["eval_tasks"])
    p16_burden = int(p16_traj * p["p16_checkpoints"] * p["burden_calls"])
    p16_load = int(p16_traj * p["p16_checkpoints"] * p["load_levels"])
    judge = int(p["stageA_bank"] * p["judge_families"] * p["label_conditions"] * p["sessions"])
    return {
        "P5 calibration (two routes, negative controls)": {"generation": p5_cal_gen, "evaluation": p5_cal_eval},
        "P5 held-out and P1 titration (depth and breadth arms)": {"generation": p5_hold_gen, "evaluation": p5_hold_eval},
        "P16 calibration, held-out, controls, intervention": {"generation": p16_gen, "evaluation": p16_eval, "burden_assay": p16_burden, "load_test": p16_load},
        "Stage A scorer channel (label experiment)": {"judge": judge},
    }


def token_totals(p: Dict[str, object]) -> Dict[str, object]:
    stages = call_counts(p)
    out: Dict[str, object] = {"stages": {}, "total_in": 0, "total_out": 0, "total_calls": 0}
    for stage, calls in stages.items():
        tin = tout = n = 0
        for kind, c in calls.items():
            i, o = TOKENS[kind]
            tin += c * i; tout += c * o; n += c
        out["stages"][stage] = {"calls": n, "tokens_in": tin, "tokens_out": tout}
        out["total_in"] += tin; out["total_out"] += tout; out["total_calls"] += n
    return out


def cost(p: Dict[str, object], prices: Dict[str, tuple] = PRICES) -> Dict[str, float]:
    t = token_totals(p)
    return {tier: (t["total_in"] * pi + t["total_out"] * po) / 1e6 for tier, (pi, po) in prices.items()}


def mixed_portfolio_cost(p: Dict[str, object], shares: Dict[str, float] = None) -> float:
    """Three lineages with different price tiers, weighted by share of the token volume."""
    shares = shares or {"frontier_top": 1 / 3, "frontier_mid": 1 / 3, "budget_api": 1 / 3}
    c = cost(p)
    return sum(c[t] * s for t, s in shares.items())


def gpu_hours_for_share(p: Dict[str, object], share: float, tokens_per_gpu_hour: float = 1.0e6) -> float:
    """Self-hosted open-weights lineage: its share of the total tokens at an assumed throughput."""
    t = token_totals(p)
    return (t["total_in"] + t["total_out"]) * share / tokens_per_gpu_hour


def wall_clock_days(p: Dict[str, object], concurrency: int = 16, seconds: Dict[str, float] = None) -> float:
    """Wall clock from total calls at a concurrency, with assumed seconds per call by type; rate limits and
    retries make the real figure longer, and evaluation calls dominate the count."""
    seconds = seconds or {"generation": 20.0, "evaluation": 5.0, "burden_assay": 15.0, "load_test": 15.0, "judge": 5.0}
    total = 0.0
    for calls in call_counts(p).values():
        for kind, c in calls.items():
            total += c * seconds[kind]
    return total / concurrency / 86400.0


def report(p: Dict[str, object], label: str) -> str:
    t = token_totals(p); c = cost(p)
    lines = ["## %s" % label, "", "| Stage | Calls | Tokens in (M) | Tokens out (M) |", "|---|---|---|---|"]
    for stage, v in t["stages"].items():
        lines.append("| %s | %s | %.1f | %.1f |" % (stage, format(v["calls"], ","), v["tokens_in"] / 1e6, v["tokens_out"] / 1e6))
    lines.append("| Total | %s | %.1f | %.1f |" % (format(t["total_calls"], ","), t["total_in"] / 1e6, t["total_out"] / 1e6))
    lines += ["", "| Price tier (assumed, dollars per million in / out) | Cost |", "|---|---|"]
    for tier, (pi, po) in PRICES.items():
        lines.append("| %s (%.2f / %.2f) | $%s |" % (tier, pi, po, format(int(round(c[tier])), ",")))
    lines.append("| mixed portfolio (a third each: top, mid, budget) | $%s |" % format(int(round(mixed_portfolio_cost(p))), ","))
    lines.append("")
    lines.append("Self-hosted share of one third at one million tokens per GPU hour: about %d GPU hours. Wall clock: about %.1f days at concurrency 16, %.1f at 64 (rate limits and retries extend both)." % (int(round(gpu_hours_for_share(p, 1 / 3))), wall_clock_days(p, 16), wall_clock_days(p, 64)))
    return "\n".join(lines)

# Flagship profiles for a first decisive run under a small budget (5 September 2026): one proposition answered
# exceptionally well rather than the whole programme thinly. eval_tasks=0 is the ladder reading of capability
# (the artefact's pass count on a frozen hidden ladder, mechanical, no tokens); a positive eval_tasks is the
# battery reading (a separate evaluation call per task per checkpoint). Three systems from three lineages.
_P5_FLAGSHIP_BASE = dict(
    n_systems=3, states=3, retentions=3, reps=3, cal_rounds=4, second_route_fraction=1.0, negative_control_fraction=0.5,
    holdout_systems=3, replicates=4, window=128, checkpoints=9, eval_tasks=0, breadth_arm=True, repetition=True,
    p16_cal=0, p16_hold=0, p16_controls=0, p16_intervention_pairs=0, horizon=128, p16_checkpoints=8, load_levels=0, burden_calls=0,
    stageA_bank=0, judge_families=3, label_conditions=3, sessions=2,
)
P5_FLAGSHIP_LADDER = dict(_P5_FLAGSHIP_BASE)
P5_FLAGSHIP_NO_BREADTH = dict(_P5_FLAGSHIP_BASE, breadth_arm=False)
P5_FLAGSHIP_BATTERY_50 = dict(_P5_FLAGSHIP_BASE, eval_tasks=50)
P5_FLAGSHIP_BATTERY_625 = dict(_P5_FLAGSHIP_BASE, eval_tasks=625)   # two per cent capability precision by battery near one half
P16_FORECAST_SECONDARY = dict(
    n_systems=0, states=0, retentions=0, reps=0, cal_rounds=0, second_route_fraction=0.0, negative_control_fraction=0.0,
    holdout_systems=0, replicates=0, window=0, checkpoints=0, eval_tasks=0, breadth_arm=False, repetition=False,
    p16_cal=6, p16_hold=3, p16_controls=2, p16_intervention_pairs=2, horizon=128, p16_checkpoints=8, load_levels=3, burden_calls=2,
    stageA_bank=0, judge_families=3, label_conditions=3, sessions=2,
)

USD_PER_GBP = 1.3513   # the rate quoted on 5 September 2026; replace on the day


def lineage_mix_cost(p: Dict[str, object], tiers=("frontier_mid", "budget_api", "open_weights_selfhosted")) -> float:
    """Three lineages, one per named tier, each carrying a third of the token volume."""
    c = cost(p)
    return sum(c[t] for t in tiers) / len(tiers)


def gbp(usd: float, rate: float = USD_PER_GBP) -> float:
    return usd / rate


def flagship_menu() -> str:
    rows = [("Pilot, cheap profile (never scored)", CHEAP), ("P5 flagship, ladder-measured capability", P5_FLAGSHIP_LADDER),
            ("P5 flagship without the breadth arm (P1 deferred)", P5_FLAGSHIP_NO_BREADTH), ("P5 flagship, 50-task battery", P5_FLAGSHIP_BATTERY_50),
            ("P5 flagship, 625-task battery (two per cent precision)", P5_FLAGSHIP_BATTERY_625), ("P16 forecast-transport secondary", P16_FORECAST_SECONDARY)]
    lines = ["| Run | Calls | Tokens (M) | Budget API alone | Mid frontier alone | Three lineages (mid, budget, open weights) | Pounds |", "|---|---|---|---|---|---|---|"]
    for name, p in rows:
        t = token_totals(p); c = cost(p); m = lineage_mix_cost(p)
        lines.append("| %s | %s | %.1f | $%d | $%d | $%d | £%d |" % (name, format(t["total_calls"], ","), (t["total_in"] + t["total_out"]) / 1e6, round(c["budget_api"]), round(c["frontier_mid"]), round(m), round(gbp(m))))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------
# Runner-faithful round counts (5 September 2026). The profiles above model the bank as calibration rounds per
# system with the second route and the negative controls as separate cell sets, which is the design battery's
# older model and counts about seven times the rounds the runner actually issues. arc_runner.p5 runs ONE crossed
# bank per lineage, one round per cell, estimates both routes from the same cells, and adds control cells as a
# registered fraction; arc_runner.p16 runs every arm system for the horizon. Cost these as they run.
# ---------------------------------------------------------------------------------------------------------------

def runner_p5_rounds(states: int = 5, fractions: int = 5, reps: int = 16, control_fraction: float = 0.2, lineages: int = 3,
                     heldout_per_lineage: int = 1, replicates: int = 4, window: int = 128, cal_depth: int = 4) -> Dict[str, int]:
    """Generation rounds the P5 runner issues at a configuration. Ladder reads are not counted: on a checkable
    ladder a read is a run of the hidden suite and costs no generation."""
    per_cell_ctrl = max(1, int(round(reps * control_fraction)))
    bank = states * fractions * reps * lineages
    controls = states * fractions * per_cell_ctrl * lineages
    heldout_systems = lineages * heldout_per_lineage
    calibration = heldout_systems * cal_depth
    sealed_window = heldout_systems * replicates * (window - cal_depth)
    total = bank + controls + calibration + sealed_window
    return {"bank": bank, "controls": controls, "calibration": calibration, "sealed_window": sealed_window, "total": total}


def runner_p16_rounds(dose_arms: int = 5, systems_per_arm: int = 3, sham: int = 3, baseline: int = 3,
                      locating_systems: int = 6, horizon: int = 96) -> Dict[str, int]:
    """Generation rounds the P16 driven titration issues: every arm system and every boundary-locating system
    runs for the horizon. The balance objects are read from the checkable suite and cost no generation."""
    arms = dose_arms * systems_per_arm + sham + baseline
    return {"arm_systems": arms, "locating_systems": locating_systems, "rounds_per_system": horizon,
            "total": (arms + locating_systems) * horizon}


def rounds_cost_usd(rounds: int, tier: str, prices: Dict[str, tuple] = PRICES) -> float:
    gi, go = TOKENS["generation"]
    pi, po = prices[tier]
    return rounds * (gi * pi + go * po) / 1e6


def rounds_cost_mix_gbp(rounds: int, tiers=("frontier_mid", "budget_api", "open_weights_selfhosted")) -> float:
    """Three lineages, one per tier, each carrying a third of the rounds; in pounds at USD_PER_GBP."""
    return gbp(sum(rounds_cost_usd(rounds, t) for t in tiers) / len(tiers))
