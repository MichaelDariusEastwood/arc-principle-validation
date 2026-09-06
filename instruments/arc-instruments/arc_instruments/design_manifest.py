"""One manifest per experiment, and everything countable is generated from it.

WHY THIS EXISTS. A design is stated in three places that drift: the registration's prose, the runner's
configuration, and the cost forecast. On 5 September 2026 a reader outside the programme found the
shape of that failure in a P16 description that said six arms while listing five dose arms, a
coefficient-only sham and an untouched baseline, which is seven. Both a six-arm and a seven-arm design
can be legitimate; what is never legitimate is a document whose arm table, sample count, simulation
configuration and cost forecast were maintained separately and no longer describe one experiment.

WHAT THIS MODULE GUARANTEES AND WHAT IT DOES NOT. It guarantees that the arm table, the systems, the
generation-round count and the cost are computed from one configuration object, so they cannot
disagree with each other. It does not guarantee that the configuration is the one the registration
intends: that is what `assert_matches` is for, and the test suite pins the numbers each protocol
states in prose so that changing one without the other fails a test rather than reaching a reader.

A manifest is hashed so it can enter a seal. The hash covers the configuration and the derived
counts, not the prices, because prices move with the vendors and a design does not change when they do.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from . import costing


def p16_arms(dose_offsets: Sequence[float], systems_per_arm: int, sham_systems: int,
             baseline_systems: int) -> List[Dict[str, Any]]:
    """The arm table. Controls are arms and are counted as arms, which is where six became seven."""
    arms = [{"arm": "dose%+.1f" % o, "kind": "dose", "offset": float(o), "systems": int(systems_per_arm)}
            for o in dose_offsets]
    arms.append({"arm": "sham", "kind": "control", "offset": 0.0, "systems": int(sham_systems)})
    arms.append({"arm": "baseline", "kind": "control", "offset": 0.0, "systems": int(baseline_systems)})
    return arms


def p16_manifest(dose_offsets: Sequence[float] = (-0.6, -0.3, 0.3, 0.6, 0.9), systems_per_arm: int = 3,
                 sham_systems: int = 3, baseline_systems: int = 3, locating_systems: int = 6,
                 horizon: int = 96, switch_round: int = 8, settling: int = 6, z_threshold: float = 4.0,
                 timing_tolerance: int = 4, location_tolerance: float = 0.15,
                 lineages: int = 3) -> Dict[str, Any]:
    arms = p16_arms(dose_offsets, systems_per_arm, sham_systems, baseline_systems)
    arm_systems = sum(a["systems"] for a in arms)
    rounds = costing.runner_p16_rounds(dose_arms=len(dose_offsets), systems_per_arm=systems_per_arm,
                                       sham=sham_systems, baseline=baseline_systems,
                                       locating_systems=locating_systems, horizon=horizon)
    m = {"experiment": "P16", "arms": arms, "arm_count": len(arms), "arm_systems": arm_systems,
         "locating_systems": locating_systems, "horizon": horizon, "switch_round": switch_round,
         "settling": settling, "z_threshold": z_threshold, "timing_tolerance": timing_tolerance,
         "location_tolerance": location_tolerance, "lineages": lineages, "rounds": rounds}
    return _finish(m, rounds["total"])


def p5_manifest(states: int = 5, fractions: int = 5, reps: int = 16, control_fraction: float = 0.2,
                lineages: int = 3, heldout_per_lineage: int = 1, replicates: int = 4,
                window: int = 128, cal_depth: int = 4, margin: float = 0.10) -> Dict[str, Any]:
    rounds = costing.runner_p5_rounds(states=states, fractions=fractions, reps=reps,
                                      control_fraction=control_fraction, lineages=lineages,
                                      heldout_per_lineage=heldout_per_lineage, replicates=replicates,
                                      window=window, cal_depth=cal_depth)
    m = {"experiment": "P5", "states": states, "fractions": fractions, "reps_per_cell": reps,
         "control_fraction": control_fraction, "control_cells_per_cell": max(1, int(round(reps * control_fraction))),
         "lineages": lineages, "heldout_systems": lineages * heldout_per_lineage, "replicates": replicates,
         "window": window, "calibration_depth": cal_depth, "margin": margin, "rounds": rounds}
    return _finish(m, rounds["total"])


def _finish(m: Dict[str, Any], total_rounds: int) -> Dict[str, Any]:
    m["cost"] = {"generation_rounds": total_rounds,
                 "usd_by_tier": {t: round(costing.rounds_cost_usd(total_rounds, t), 2) for t in costing.PRICES},
                 "gbp_three_lineage_mix": round(costing.rounds_cost_mix_gbp(total_rounds), 2),
                 "prices_are_placeholders": True}
    payload = {k: v for k, v in m.items() if k != "cost"}
    payload["generation_rounds"] = total_rounds
    m["sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return m


def assert_matches(manifest: Dict[str, Any], **stated) -> None:
    """Pin the numbers a protocol states in prose against the manifest that generates them. Raises
    with both values named, because a mismatch is a document that no longer describes one experiment."""
    flat = dict(manifest)
    flat.update({"generation_rounds": manifest["rounds"]["total"],
                 "gbp_three_lineage_mix": manifest["cost"]["gbp_three_lineage_mix"]})
    bad = []
    for k, want in stated.items():
        got = flat.get(k)
        if isinstance(want, float) and isinstance(got, (int, float)):
            ok = abs(got - want) <= max(0.5, abs(want) * 0.02)
        else:
            ok = got == want
        if not ok:
            bad.append("%s: manifest says %r, the document says %r" % (k, got, want))
    if bad:
        raise AssertionError("the design manifest and the document disagree: " + "; ".join(bad))


def render(manifest: Dict[str, Any]) -> str:
    """The arm and cost table a protocol quotes, generated rather than typed."""
    lines = ["%s design manifest  sha256 %s" % (manifest["experiment"], manifest["sha256"][:16]), ""]
    if manifest["experiment"] == "P16":
        lines += ["arm            kind      offset  systems", "-" * 42]
        for a in manifest["arms"]:
            lines.append("%-14s %-9s %+6.1f  %7d" % (a["arm"], a["kind"], a["offset"], a["systems"]))
        lines += ["-" * 42,
                  "%d arms, %d arm systems, %d locating systems, horizon %d"
                  % (manifest["arm_count"], manifest["arm_systems"], manifest["locating_systems"], manifest["horizon"])]
    else:
        lines.append("%d states x %d fractions x %d replicates x %d lineages, %d control cells per cell"
                     % (manifest["states"], manifest["fractions"], manifest["reps_per_cell"],
                        manifest["lineages"], manifest["control_cells_per_cell"]))
        lines.append("%d held-out systems x %d replicate continuations, window %d to %d"
                     % (manifest["heldout_systems"], manifest["replicates"], manifest["calibration_depth"],
                        manifest["window"]))
    r = manifest["rounds"]
    lines += ["", "generation rounds: " + ", ".join("%s %d" % (k, v) for k, v in sorted(r.items()) if k != "total"),
              "total %d rounds" % r["total"],
              "cost: three-lineage mix GBP %.0f (prices are placeholders)" % manifest["cost"]["gbp_three_lineage_mix"]]
    return "\n".join(lines)
