#!/usr/bin/env python3
"""The calibration of P16's COMPLETE decision family, with every rate reported as an interval.

WHY THIS SCRIPT EXISTS. Finding A3: `P16Config.z_threshold` carried a comment reporting "an
approximately 0.052 flat-null alarm rate from 400 series at z = 4.0" as though it were demonstrated
control at or below 0.05. Three things are wrong with reading it that way.

  It is a point estimate.       Twenty-one alarms in 400 series is an observed 0.0525 whose exact 95
                                per cent interval is [0.033, 0.079]. The observation is consistent
                                with a true rate half again above 0.05, and a rate of zero in n runs
                                is not zero risk either: it has its own upper bound.
  It is one arm, not the family.  The decision family is every look in every arm, plus the run-level
                                rules that read those arms: the support branch, the refutation rule,
                                the mislocation branch, the specificity branch and the component
                                wrapper. A per-arm rate is not that family's rate.
  It is one world.              A flat null with independent noise says nothing about a flat null
                                whose noise is autocorrelated, which is the null the runner will
                                actually meet, and nothing about how often a true world is refuted.

So this script runs the family itself, world by world, and reports each rate with its exact binomial
interval from `arc_instruments.mc_interval`. It calibrates and does not assert. Nothing here is
evidence about any real system: every world is simulated.

  python3 p16_calibration.py --out <dir> [--seeds 200]

A run of the registered configuration takes about half a second, so 200 seeds across the worlds here
is several minutes. `calibrate` and `render_report` are importable so that the arithmetic can be exercised
on a handful of seeds without waiting for a full calibration.
"""
from __future__ import annotations

import argparse
import collections
import datetime as _dt
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional

from arc_instruments import mc_interval as MC
from arc_runner import observation as OBS, p16, p16_sequential as SEQ

# --------------------------------------------------------------------------------------------------
# The worlds
# --------------------------------------------------------------------------------------------------

# The bands and horizons a calibration has to supply, because the components refuse to invent them
# and a family with unsupplied components makes no decisions to calibrate. These are the CANDIDATE
# registered numbers this script measures the family under; they are not registered by the contract,
# and a calibration run under different ones must say so, which is why they travel in every row.
#
# THEY ARE READ FROM THE PACKAGE RATHER THAN RESTATED HERE. The demonstration path runs on the same
# five, labelled as candidates, so two copies of this dictionary would eventually differ and a
# calibration would then describe a family nothing else was ever exercised under.
CANDIDATE = dict(p16.CANDIDATE_QUANTITIES)


def config(**kw) -> p16.P16Config:
    base = dict(CANDIDATE)
    base.update(kw)
    return p16.P16Config(**base)


def world(name: str, line_slope: float, zero: float, level: float = 0.0, rho: float = 0.0,
          noise: float = 0.02, error_labels: Optional[List[str]] = None,
          correct_labels: Optional[List[str]] = None, truth: str = "",
          overrides: Optional[Dict[str, Any]] = None,
          reversal_round: Optional[int] = None, reversal_level: float = 0.0) -> Dict[str, Any]:
    """One simulated world, with the labels that would be errors in it named before it runs.

    `overrides` are configuration changes this row is measured under, so that a row may measure what
    one clause of the frozen rule costs while every other clause stays where it is.

    `reversal_round` and `reversal_level` make the world NON-STATIONARY in Delta: the dosed arms hold
    `level` until that round and `reversal_level` after it. Every other world here is constant in
    Delta within an arm, so the whole-window mean equals the terminal margin in all of them and a
    calibration built only from those worlds is blind to the difference between a margin that stayed
    above the band and one that only averaged above it. That difference is exactly what the
    across-window predicate exists to see, so a world containing a late reversal is the one that
    measures whether it sees it.
    """
    return {"name": name, "line_slope": line_slope, "zero": zero, "level": level, "rho": rho,
            "noise": noise, "truth": truth, "overrides": dict(overrides or {}),
            "reversal_round": reversal_round, "reversal_level": float(reversal_level),
            "error_labels": error_labels or [], "correct_labels": correct_labels or []}


SUPPORTED = "SUPPORTED"
REFUTED_NO_REVERSAL = "REFUTED (no reversal)"
REFUTED_MISLOCATED = "REFUTED (boundary mislocated)"
INCONCLUSIVE_SILENT = "INCONCLUSIVE (no reversal, and none demonstrated absent)"

# THE TWO NON-STATIONARY WORLDS, AND WHY THERE ARE TWO. Every other world here is constant in Delta
# within an arm, so its whole-window mean IS its terminal margin and no world could distinguish a
# margin that stayed above the band from one that only averaged above it. These two can. The first
# reverses far enough from the horizon for the registered resolution to see it, and the family must
# not refute in it. The second reverses INSIDE the last segment, where a two-segment reading cannot
# see it, and the family does refute in it: that is the cost of the resolution, measured rather than
# argued, and it is the reason the refutation record carries the segment length it was read at. A
# reader who wants that error smaller registers more segments and pays for it in the world where
# refutation is warranted, where thirds hold the margin in forty-seven arms of ninety against ninety
# of ninety in halves.
REVERSAL_SEEN = world(
    "a margin held positive and then REVERSED with a quarter of the window left: refutation is wrong "
    "here and a window mean cannot tell it from the world above",
    line_slope=0.0, zero=2.0, level=0.30, rho=0.0, reversal_round=70, reversal_level=-0.05,
    truth="every arm's margin is firmly positive for most of the window and negative for the last "
          "twenty-six rounds, so the reversal has happened and nothing may be refuted. The "
          "whole-window mean is still above the band, which is why this world is here: it is a world "
          "in which the mean reading and the across-window reading disagree",
    error_labels=[SUPPORTED, REFUTED_NO_REVERSAL, REFUTED_MISLOCATED])

REVERSAL_INSIDE_SEGMENT = world(
    "the same reversal, confined to the last segment the registered resolution reads: the limit of "
    "that resolution, measured",
    line_slope=0.0, zero=2.0, level=0.30, rho=0.0, reversal_round=80, reversal_level=-0.05,
    truth="the margin reverses with sixteen rounds of the window left, which is well inside one "
          "half of it, so a two-segment reading averages the reversal away and the family refutes a "
          "proposition it should not. This world is not a demonstration that the rule works: it is "
          "the price of reading the window in two pieces, and the segment length travels with every "
          "refutation so that a reader knows which reversals it excludes",
    error_labels=[SUPPORTED, REFUTED_NO_REVERSAL, REFUTED_MISLOCATED])

WORLDS = [
    world("flat null, independent noise: no boundary and a margin of exactly zero",
          line_slope=0.0, zero=2.0, level=0.0, rho=0.0,
          truth="no arm's balance depends on its dose and every arm's margin is zero, so no "
                "proposition-level claim is warranted in either direction",
          error_labels=[SUPPORTED, REFUTED_NO_REVERSAL, REFUTED_MISLOCATED]),
    world("flat null, autocorrelated noise (AR(1) at 0.8): the null the runner will actually meet",
          line_slope=0.0, zero=2.0, level=0.0, rho=0.8,
          truth="as above, with the round-to-round dependence an independence standard error "
                "under-reports",
          error_labels=[SUPPORTED, REFUTED_NO_REVERSAL, REFUTED_MISLOCATED]),
    world("true world: the sealed line, slope minus one half with its zero at two",
          line_slope=-0.5, zero=2.0, rho=0.0,
          truth="the proposition holds exactly as sealed",
          error_labels=[REFUTED_NO_REVERSAL, REFUTED_MISLOCATED], correct_labels=[SUPPORTED]),
    world("flat null at 0.8, read with a first look of twenty-four rounds instead of four",
          line_slope=0.0, zero=2.0, level=0.0, rho=0.8, overrides={"min_look_points": 24},
          truth="the same null, measuring what the short-window clause of the frozen rule costs: the "
                "earliest looks of a growing window are the ones a persistent series wins on, "
                "because at four points no variance estimator can see a long dependence",
          error_labels=[SUPPORTED, REFUTED_NO_REVERSAL, REFUTED_MISLOCATED]),
    world("no boundary and a margin held firmly positive everywhere: refutation IS warranted here",
          line_slope=0.0, zero=2.0, level=0.30, rho=0.0,
          truth="no arm reverses and every arm's margin is measurably positive past the horizon, "
                "which is the only evidence the registered rule accepts for a refutation",
          error_labels=[SUPPORTED], correct_labels=[REFUTED_NO_REVERSAL]),
    REVERSAL_SEEN,
    REVERSAL_INSIDE_SEGMENT,
]


def source_for(w: Dict[str, Any], cfg: p16.P16Config):
    """A world on the registered coordinate: Q, W, R and U, with the balance line built to order.

    log(Q/W) travels as Delta log R with Delta = level + line_slope (alpha - zero), so a world's line
    and its resting margin move independently, which is what separates a flat null (Delta zero
    everywhere) from a world whose margin is firmly positive everywhere (Delta at the level, no
    dependence on the dose). The noise is AR(1) in the log ratio at the world's rho, which is the
    temporal correlation the sequential rule's standard error exists to survive.
    """
    alpha_base = float(w["zero"]) - 0.4
    R_switch = float(cfg.switch_round + 1)
    rho, sd = float(w["rho"]), float(w["noise"])
    rev = w.get("reversal_round")
    R_rev = None if rev is None else float(int(rev) + 1)
    state: Dict[str, float] = {}

    def delta_of(alpha: float) -> float:
        return float(w["level"]) + float(w["line_slope"]) * (float(alpha) - float(w["zero"]))

    def src(arm, alpha_arm, r, rng):
        R = float(r + 1)
        control = arm in ("sham", "baseline")
        dosed = (r >= cfg.switch_round) and not control
        a = float(alpha_arm) if dosed else alpha_base
        log_R0 = math.log(min(R, R_switch))
        # THE DOSED SEGMENT, SPLIT AT THE REVERSAL WHERE THE WORLD HAS ONE. log(Q/W) travels as the
        # integral of Delta with respect to log R, so a world whose Delta changes part-way through
        # the window is built by integrating each piece over its own stretch of log R rather than by
        # editing the series afterwards: the reversal is then in the measured quantity in the same
        # way a real one would be.
        top = R if R_rev is None else min(R, R_rev)
        log_R1 = math.log(top / R_switch) if top > R_switch else 0.0
        log_R2 = 0.0 if (R_rev is None or R <= R_rev) else math.log(R / R_rev)
        log_U = alpha_base * log_R0 + a * (log_R1 + log_R2)
        after = float(w.get("reversal_level", 0.0)) if dosed else delta_of(a)
        log_ratio = delta_of(alpha_base) * log_R0 + delta_of(a) * log_R1 + after * log_R2
        key = "%s|%s" % (arm, alpha_arm)
        if r == 0:
            state[key] = 0.0
        e = rho * state.get(key, 0.0) + rng.normal(0, sd) if sd > 0 else 0.0
        state[key] = e
        U = math.exp(log_U)
        W = a * U / R
        Q = W * math.exp(log_ratio + e)
        return {"round": r, "Q": Q, "W": W, "R": R, "U": U,
                "extra": {OBS.DELIVERY_KEY: {"applied": bool(dosed),
                                             "lever": ("growth exponent %.4f" % a) if dosed else None}}}

    return OBS.declare(src, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=cfg.chi_hat, source="calibration-world", simulated=True,
        note="a calibration world with line slope %g, zero %g, resting margin %g, AR(1) noise at %g%s"
             % (w["line_slope"], w["zero"], w["level"], rho,
                "" if w.get("reversal_round") is None else
                (" and a reversal of the dosed arms to %g at round %d"
                 % (w.get("reversal_level", 0.0), int(w["reversal_round"]))))))


# --------------------------------------------------------------------------------------------------
# The calibration
# --------------------------------------------------------------------------------------------------

def calibrate_world(w: Dict[str, Any], seeds: int, cfg: Optional[p16.P16Config] = None) -> Dict[str, Any]:
    """One world, run `seeds` times, with every rate carrying its exact interval.

    Four rates are reported and they are never mixed. The per-arm alarm rate is the one the reference
    comment quoted. The family-wise rate is the rate at which the run-level family makes a claim that
    is wrong in this world, which is the quantity finding A3 asks for. The refutation-evidence rate
    says how often a refutation was reached on demonstrated positivity, and the non-alarm census says
    what the silences actually were, because a family whose refusals are all censoring has a
    different problem from one whose refusals are all low information.
    """
    cfg = cfg if cfg is not None else config()
    if w.get("overrides"):
        cfg = p16.P16Config(**(dict(cfg.__dict__) | dict(w["overrides"])))
    rule = SEQ.rule_from_config(cfg)
    patterns: collections.Counter = collections.Counter()
    wrapper: collections.Counter = collections.Counter()
    silences: collections.Counter = collections.Counter()
    family_errors = 0
    correct = 0
    refuted_on_evidence = 0
    arm_alarms = {"above": [0, 0], "below": [0, 0], "control": [0, 0]}   # [alarms, arms]
    for s in range(seeds):
        res = p16.run_p16(source_for(w, cfg), cfg, 4000 + s, "none", "mock")
        v = res["verdicts"]
        patterns[v["run_pattern"]] += 1
        wrapper[v["P16"]] += 1
        if v["run_pattern"] in w["error_labels"]:
            family_errors += 1
        if v["run_pattern"] in w["correct_labels"]:
            correct += 1
        if v["refutation"]["refutes"]:
            refuted_on_evidence += 1
        for cl in [c for rows in v["non_alarm"].values() for c in rows]:
            silences[cl["state"]] += 1
        for a in res["arms"]:
            name = str(a.get("arm", ""))
            if name in ("sham", "baseline"):
                key = "control"
            elif float(a.get("alpha", 0.0)) > cfg.alpha_crit_hat:
                key = "above"
            else:
                key = "below"
            arm_alarms[key][1] += 1
            arm_alarms[key][0] += int(a.get("declared_round") is not None)
    n_arms = len(cfg.dose_offsets) * cfg.systems_per_arm + 2 * cfg.systems_per_arm
    return {
        "world": w["name"], "truth": w["truth"], "seeds": seeds,
        "candidate_registered_numbers": {k: getattr(cfg, k) for k in CANDIDATE},
        "configuration_overrides": dict(w.get("overrides") or {}),
        "sequential_rule": rule.as_record(),
        "family_size": SEQ.family_size(rule, n_arms),
        "run_pattern": {k: v / float(seeds) for k, v in patterns.items()},
        "wrapper": {k: v / float(seeds) for k, v in wrapper.items()},
        "labels_counted_as_error": list(w["error_labels"]),
        "family_wise_error": SEQ.rate_with_interval(family_errors, seeds),
        "correct_claim_rate": SEQ.rate_with_interval(correct, seeds),
        "refutation_on_demonstrated_positivity": SEQ.rate_with_interval(refuted_on_evidence, seeds),
        "per_arm_alarm_rate": {k: SEQ.rate_with_interval(v[0], v[1]) for k, v in arm_alarms.items() if v[1]},
        "non_alarm_census": dict(silences),
    }


def calibrate(seeds: int = 200, worlds: Optional[List[Dict[str, Any]]] = None,
              cfg: Optional[p16.P16Config] = None) -> List[Dict[str, Any]]:
    return [calibrate_world(w, seeds, cfg) for w in (worlds if worlds is not None else WORLDS)]


def render_report(rows: List[Dict[str, Any]], seeds: int, generated_utc: str) -> str:
    """The markdown, with an interval beside every rate and no rate quoted without one."""
    out = ["# P16: the calibrated error of the complete decision family (%s, %d runs per world)"
           % (generated_utc, seeds), "",
           "Every rate below is an estimate from %d simulated runs and is reported with its exact "
           "binomial interval. A rate of zero in %d runs is not zero risk: its one-sided 95 per cent "
           "upper bound is %.4f. Nothing here is evidence about any real system."
           % (seeds, seeds, MC.exact_upper(0, seeds)), "",
           "The family is every look in every arm plus the run-level rules that read them. A per-arm "
           "rate is not the family's rate and the two are never reported here as one.", ""]
    if rows:
        fs = rows[0]["family_size"]
        out += ["| family | value |", "|---|---|",
                "| arms per run | %d |" % fs["arms"],
                "| looks per arm | %d |" % fs["looks_per_arm"],
                "| arm-look tests per run | %d |" % fs["arm_look_tests"],
                "| variance estimator | %s |" % rows[0]["sequential_rule"]["variance_estimator"],
                "| threshold z | %.2f |" % rows[0]["sequential_rule"]["threshold_z"],
                "| informative horizon (rounds after the switch) | %s |"
                % rows[0]["sequential_rule"]["informative_horizon_rounds_after_switch"],
                "| practical-absence band | %s |" % rows[0]["sequential_rule"]["practical_absence_band"],
                "| segments the window is read across | %s |"
                % rows[0]["sequential_rule"]["across_window_segments"],
                ""]
    out += ["## Family-wise error, world by world", "",
            "| World | wrong claims counted | family-wise error | exact 95 per cent interval | "
            "one-sided upper |", "|---|---|---|---|---|"]
    for r in rows:
        e = r["family_wise_error"]
        out.append("| %s | %s | %.3f | [%.3f, %.3f] | %.3f |"
                   % (r["world"], ", ".join(r["labels_counted_as_error"]) or "none", e["rate"],
                      e["exact_two_sided"][0], e["exact_two_sided"][1], e["exact_one_sided_upper"]))
    out += ["", "## The per-arm alarm rate, which is not the family's rate", "",
            "| World | above the boundary | below it | controls |", "|---|---|---|---|"]
    for r in rows:
        cells = []
        for k in ("above", "below", "control"):
            d = r["per_arm_alarm_rate"].get(k)
            cells.append("n/a" if d is None else "%.3f [%.3f, %.3f] (%d of %d)"
                         % (d["rate"], d["exact_two_sided"][0], d["exact_two_sided"][1],
                            d["successes"], d["outer_repetitions"]))
        out.append("| %s | %s | %s | %s |" % (r["world"], cells[0], cells[1], cells[2]))
    out += ["", "## What the silences were", "",
            "| World | census of the four readings of a non-alarm | refutation on demonstrated "
            "positivity |", "|---|---|---|"]
    for r in rows:
        c = r["refutation_on_demonstrated_positivity"]
        out.append("| %s | %s | %.3f [%.3f, %.3f] |"
                   % (r["world"],
                      ", ".join("%s %d" % (k.lower(), v) for k, v in sorted(r["non_alarm_census"].items()))
                      or "none",
                      c["rate"], c["exact_two_sided"][0], c["exact_two_sided"][1]))
    out += ["", "## The run-level labels in full", ""]
    for r in rows:
        out += ["### %s" % r["world"], "", "_%s_" % r["truth"], "",
                "| label | rate | wrapper label | rate |", "|---|---|---|---|"]
        pats = sorted(r["run_pattern"].items(), key=lambda kv: -kv[1])
        wraps = sorted(r["wrapper"].items(), key=lambda kv: -kv[1])
        for i in range(max(len(pats), len(wraps))):
            p = ("%s | %.3f" % pats[i]) if i < len(pats) else " | "
            w_ = ("%s | %.3f" % wraps[i]) if i < len(wraps) else " | "
            out.append("| %s | %s |" % (p, w_))
        out.append("")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=200)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = calibrate(a.seeds)
    generated = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = {"generated_utc": generated, "seeds_per_world": a.seeds, "worlds": rows,
               "note": "The calibrated error of P16's complete decision family under simulated "
                       "worlds, every rate with its exact binomial interval. Never evidence about "
                       "any real system."}
    json.dump(payload, open(os.path.join(a.out, "P16-CALIBRATION.json"), "w"), indent=1, default=float)
    open(os.path.join(a.out, "P16-CALIBRATION.md"), "w").write(render_report(rows, a.seeds, generated))
    print(json.dumps({"worlds": len(rows), "seeds_per_world": a.seeds,
                      "family_wise_error": {r["world"]: r["family_wise_error"]["exact_two_sided"]
                                            for r in rows}}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
