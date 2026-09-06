#!/usr/bin/env python3
"""The runner battery: the operating characteristics of the ACTUAL analysis code, world by world.

The design battery (design_battery.py) measured the designs under stipulated noise models. This measures
the runner, arc_runner.p5 and arc_runner.p16, which is the code that will score real data. Every row is
one world crossed with one configuration, run many times from different seeds, and the verdict rates are
tabulated. A world in which the framework is true must come back SUPPORTED at the registered rate; a
world in which it is false in a named way must come back with the right refusal. Design calculation under
simulated systems; never evidence about any real system.

Run: python3 runner_battery.py --out <dir> [--seeds 40]
"""
from __future__ import annotations
import argparse, collections, datetime as _dt, json, os, sys, time
import numpy as np
from arc_runner import adapters, ladder as L, p5, p16
from arc_instruments import mc_interval as MC

def p5_world(name, beta=0.5, theta=0.0, noise=0.03, n_items=20000, scale=400.0, reps=8, window=32,
             cal_reads=16, states=None, systems=3, heldout_beta=None, control_leak=0.0, saturation=None,
             retention_exponent=None, available_rate_exponent=0.0):
    # `retention_exponent` and `available_rate_exponent` are finding A6's two worlds: an elasticity in
    # the retention fraction that differs from the one in the capability state, and a nuisance rate
    # that scales with the available capability and therefore adds the same amount to both.
    return dict(name=name, beta=beta, theta=theta, noise=noise, n_items=n_items, scale=scale, reps=reps,
                window=window, cal_reads=cal_reads, states=states, systems=systems, heldout_beta=heldout_beta,
                control_leak=control_leak, saturation=saturation, retention_exponent=retention_exponent,
                available_rate_exponent=available_rate_exponent)

P5_WORLDS = [
    p5_world("true, registered configuration"),
    p5_world("true, cheap ladder (4,000 items)", n_items=4000),
    p5_world("true, four calibration reads instead of sixteen", cal_reads=4),
    p5_world("true, four replicates instead of eight", reps=4),
    p5_world("true, noisier system (0.08)", noise=0.08),
    p5_world("true, one held-out system", systems=1),
    p5_world("rate confound (theta 0.2)", theta=0.2, scale=2000.0),
    p5_world("rate confound (theta 0.1)", theta=0.1, scale=1000.0),
    p5_world("true at a different coupling (0.35): mechanism holds, must SUPPORT", beta=0.35),
    p5_world("REGIME CHANGE: bank at 0.50, held-out evolves at 0.35; must REFUTE", beta=0.50, heldout_beta=0.35),
    p5_world("REGIME CHANGE: bank at 0.50, held-out evolves at 0.65; must REFUTE", beta=0.50, heldout_beta=0.65, scale=2000.0),
    p5_world("REGIME CHANGE, small: held-out at 0.40 (the registered 0.10 shift)", beta=0.50, heldout_beta=0.40),
    p5_world("no coupling (beta 0): linear growth, mechanism holds", beta=0.0),
    p5_world("ladder without headroom (scale 100)", scale=100.0),
    # The registered window is 128. Reading a 0.5-coupled system to depth 128 from a start near 20 needs a ladder
    # with headroom past about 2,300 and enough items to resolve increments near 20, which is ten times the items
    # of the window-32 ladder at equal per-item precision. These rows measure whether the longer window separates
    # a coupling shift from the per-system rate that a 32-round window lets the calibration absorb.
    p5_world("true, REGISTERED WINDOW 128 (ladder scale 10,000, 200,000 items)", window=128, scale=10000.0, n_items=200000),
    p5_world("REGIME CHANGE 0.50 to 0.35 at window 128; must REFUTE", window=128, scale=10000.0, n_items=200000, heldout_beta=0.35),
    p5_world("REGIME CHANGE 0.50 to 0.40 at window 128 (the registered 0.10 shift)", window=128, scale=10000.0, n_items=200000, heldout_beta=0.40),
    p5_world("no coupling (beta 0) at window 128", window=128, scale=10000.0, n_items=200000, beta=0.0),
    p5_world("rate confound (theta 0.2) at window 128", window=128, scale=10000.0, n_items=200000, theta=0.2),
    # the two remaining registered hypotheses: the negative control (identification) and H3 (the premise)
    # the mock's leak acts on available capability, so an increment coupling of 0.30 in the control cells is a
    # leak of 0.30 / beta = 0.60 on that scale
    p5_world("NEGATIVE CONTROL LEAKS: control cells carry increment coupling 0.30; must be NOT IDENTIFIED", control_leak=0.60),
    p5_world("NEGATIVE CONTROL LEAKS, mild: control coupling 0.15", control_leak=0.30),
    # the window-128 ladder above has the per-item precision of a 20,000-item ladder at scale 1,000, which is
    # too coarse at the bank's low states for the predicted exponent's interval to clear the margin at the
    # registered window (the exponent's sensitivity to the coupling roughly doubles between windows 32 and 128).
    # This row gives the ladder four times the items, which is the precision the charter's own coupling
    # standard error of about 0.013 implies.
    p5_world("true, REGISTERED WINDOW 128, ladder with 800,000 items (bank-precise)", window=128, scale=10000.0, n_items=800000),
    p5_world("REGIME CHANGE 0.50 to 0.35 at window 128, bank-precise ladder; must REFUTE", window=128, scale=10000.0, n_items=800000, heldout_beta=0.35),
    p5_world("REGIME CHANGE 0.50 to 0.40 at window 128, bank-precise ladder (the 0.10 shift)", window=128, scale=10000.0, n_items=800000, heldout_beta=0.40),
    # CANDIDATE REGISTERED CONFIGURATION: the registered window, a ladder precise at the bank's low states and
    # with headroom at depth 128, and sixteen bank replicates per cell, which is the bank size at which the
    # coupling's standard error reaches the charter's own 0.013 and the predicted exponent's interval clears
    # the margin at that window
    p5_world("CANDIDATE: window 128, 800,000-item ladder, 16 bank replicates; true", window=128, scale=10000.0, n_items=800000, reps=16),
    p5_world("CANDIDATE, REGIME CHANGE 0.50 to 0.35; must REFUTE", window=128, scale=10000.0, n_items=800000, reps=16, heldout_beta=0.35),
    p5_world("CANDIDATE, REGIME CHANGE 0.50 to 0.40 (the 0.10 shift)", window=128, scale=10000.0, n_items=800000, reps=16, heldout_beta=0.40),
    p5_world("CANDIDATE, noisier system (0.08)", window=128, scale=10000.0, n_items=800000, reps=16, noise=0.08),
    p5_world("CANDIDATE, control leaks 0.30; must be NOT IDENTIFIED", window=128, scale=10000.0, n_items=800000, reps=16, control_leak=0.60),
    p5_world("CANDIDATE, saturation at 400 (H3 must refute the premise)", window=128, scale=10000.0, n_items=800000, reps=16, saturation=400.0),
    # H3 is a check on the bank's titrated range. A saturation that bites above that range and inside the window
    # (400 against states to 250) is caught by H3 in about a third of runs and by the curve comparison in another
    # third. These rows extend the bank's states to the range the sealed window will be read over.
    p5_world("CANDIDATE, bank states to 1,000 (30, 100, 250, 500, 1000); true", window=128, scale=10000.0, n_items=800000, reps=16, states=(30, 100, 250, 500, 1000)),
    p5_world("CANDIDATE, bank states to 1,000; saturation at 400 (H3 must refute)", window=128, scale=10000.0, n_items=800000, reps=16, states=(30, 100, 250, 500, 1000), saturation=400.0),
    p5_world("CANDIDATE, bank states to 1,000; REGIME CHANGE 0.50 to 0.35", window=128, scale=10000.0, n_items=800000, reps=16, states=(30, 100, 250, 500, 1000), heldout_beta=0.35),
    p5_world("SATURATION (H3): increment saturates at available 60; premise must be REFUTED", saturation=60.0),
    p5_world("SATURATION (H3), mild: saturates at 400", saturation=400.0),
    # FINDING A6's two worlds. Neither can be read from the identification column alone, which is why
    # the route agreement is tabulated beside it: the first must come back INCONSISTENT with the
    # capability elasticity still near the truth, and the second must come back CONSISTENT with the
    # capability elasticity displaced by the nuisance, which is agreement without identification.
    p5_world("CROSSED: retention exponent 0.9 against coupling 0.5; routes must be INCONSISTENT",
             retention_exponent=0.9),
    p5_world("CROSSED, mild: retention exponent 0.65 against coupling 0.5", retention_exponent=0.65),
    p5_world("SHARED RATE FACTOR: nuisance scaling with available capability at 0.2; routes must AGREE "
             "at 0.7 while the coupling is 0.5", available_rate_exponent=0.2, scale=2000.0),
]

def run_p5_world(w, seeds):
    counts = collections.Counter(); idents = collections.Counter(); premises = collections.Counter(); betas = []
    agreements = collections.Counter(); capability = []
    # THE PANEL'S OWN BOOKKEEPING (finding A7). The aggregate is now stated over the ASSIGNED panel,
    # so the rate of each panel result cannot be read without the rate at which systems left the
    # comparison: a world in which half the panel reaches the ladder ceiling can no longer reach
    # SUPPORTED, and that is a property of the world and the ladder rather than of the estimator.
    panel = collections.Counter(); panel_systems = 0
    for s in range(seeds):
        ad = adapters.MockCouplingAdapter(beta=w["beta"], theta=w["theta"], noise=w["noise"], control_leak=w["control_leak"],
                                          saturation=w["saturation"], retention_exponent=w.get("retention_exponent"),
                                          available_rate_exponent=w.get("available_rate_exponent", 0.0))
        lad = L.MockLadder(n_items=w["n_items"], scale=w["scale"])
        place = lambda st: {"kind": "mock", "capability": float(st), "rounds": 0}
        start = lambda n: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": n}
        cfg = p5.P5Config(reps=w["reps"], window_end=w["window"], checkpoints=tuple(d for d in (4, 8, 16, 32, 64, 128) if d <= w["window"]),
                          cal_reads=w["cal_reads"])
        if w["states"]: cfg.states = w["states"]
        had = adapters.MockCouplingAdapter(beta=w["heldout_beta"], theta=w["theta"], noise=w["noise"]) if w.get("heldout_beta") is not None else None
        res = p5.run_p5(ad, lad, cfg, 1000 + s, place, start, ["S%d" % i for i in range(1, w["systems"] + 1)], heldout_adapter=had)
        # `diagnostics`, which is the name a simulated recovery is exported under. It is the same
        # object the run built as `verdicts`, and reading it by the name that says what it is keeps
        # this battery from being quotable as a table of verdicts.
        d = res["diagnostics"]
        counts[d["PREDICTION"]] += 1; idents[d["IDENTIFICATION"]] += 1
        premises[d.get("PREMISE", "n/a")] += 1
        agreements[d.get("ROUTE_AGREEMENT") or "n/a"] += 1
        pan = d.get("panel") or {}
        for k in ("agrees", "disagrees", "unresolved", "not_evaluable"):
            panel[k] += int(pan.get(k, 0))
        panel_systems += int(pan.get("assigned", 0))
        b = res["routes"].get("beta_pooled");
        if b is not None and np.isfinite(b): betas.append(b)
        bc = res["routes"].get("beta_capability")
        if bc is not None and np.isfinite(bc): capability.append(bc)
    n = float(seeds)
    # every reported rate is an estimate from `seeds` OUTER repetitions; the INNER resamples are the
    # bootstrap draws the analysis uses inside each run (cfg.bootstrap) and never narrow these intervals
    inner = int(p5.P5Config().bootstrap)
    def with_ci(counter):
        return {k: MC.RateWithUncertainty(v, seeds, inner).as_dict() for k, v in counter.items()}
    return {"world": w["name"], "true_beta": w["beta"], "theta": w["theta"], "seeds": seeds, "inner_resamples_per_run": inner,
            "prediction": {k: v / n for k, v in counts.items()}, "identification": {k: v / n for k, v in idents.items()},
            "premise": {k: v / n for k, v in premises.items()},
            "route_agreement": {k: v / n for k, v in agreements.items()},
            "assigned_systems_total": panel_systems,
            "per_system_share": ({k: v / float(panel_systems) for k, v in panel.items()}
                                 if panel_systems else {}),
            "prediction_ci": with_ci(counts), "identification_ci": with_ci(idents), "premise_ci": with_ci(premises),
            "route_agreement_ci": with_ci(agreements),
            "mean_beta_pooled": float(np.mean(betas)) if betas else None, "sd_beta_pooled": float(np.std(betas)) if betas else None,
            "mean_beta_capability": float(np.mean(capability)) if capability else None,
            "sd_beta_capability": float(np.std(capability)) if capability else None}

def p16_world(name, true_alpha_crit=2.0, located=2.0, jump=0.15, noise=0.04, systems=3, horizon=96, boundary=True, generic=False, z=4.0):
    return dict(name=name, true_alpha_crit=true_alpha_crit, located=located, jump=jump, noise=noise, systems=systems,
                horizon=horizon, boundary=boundary, generic=generic, z=z)

P16_WORLDS = [
    p16_world("true, registered configuration"),
    p16_world("true, cheap tier (48 rounds, one system per arm)", horizon=48, systems=1),
    p16_world("true, noisier margins (0.08)", noise=0.08),
    p16_world("true, boundary mislocated by 0.3", true_alpha_crit=2.3, located=2.0),
    p16_world("no boundary", boundary=False),
    p16_world("generic deterioration in every arm", generic=True),
    p16_world("true, threshold 3.0 (uncalibrated)", z=3.0),
]

def p16_source(w, cfg):
    def src(arm, alpha_arm, r, rng):
        base = 0.5
        if r < cfg.switch_round: return base + rng.normal(0, w["noise"])
        if w["generic"]:
            trend = -0.01 * (r - cfg.switch_round)          # every arm falls, controls included
        elif not w["boundary"]:
            trend = 0.0
        else:
            trend = 0.0 if arm in ("sham", "baseline") else (w["true_alpha_crit"] - alpha_arm) * 0.01 * (r - cfg.switch_round)
        return base + w["jump"] + trend + rng.normal(0, w["noise"])
    return src

def run_p16_world(w, seeds):
    counts = collections.Counter()
    for s in range(seeds):
        cfg = p16.P16Config(systems_per_arm=w["systems"], horizon=w["horizon"], alpha_crit_hat=w["located"], z_threshold=w["z"], margin_noise=w["noise"])
        res = p16.run_p16(p16_source(w, cfg), cfg, 2000 + s, "none", "mock")
        # `run_pattern`, not `P16`: this battery measures the DETECTION RULE's operating
        # characteristics world by world, and the run's reading of its own arms is that rule's
        # output. Since finding A2 the `P16` key is the component wrapper's result, which on these
        # undeclared sources refuses at proposition level and would make every row identical. The
        # block is read under the exported name `diagnostics` for the reason the P5 rows are.
        counts[res["diagnostics"]["run_pattern"]] += 1
    return {"world": w["name"], "seeds": seeds, "inner_resamples_per_run": None,
            "verdicts": {k: v / float(seeds) for k, v in counts.items()},
            "verdicts_ci": {k: MC.RateWithUncertainty(v, seeds, None).as_dict() for k, v in counts.items()}}

def render_report(rows5, rows16, seeds, generated_utc):
    """The markdown report, factored out so its intervals and counts can be exercised on stub rows without
    running the battery, which matters when the analysis code under it is mid-repair."""
    L_ = ["# Runner battery: the analysis code's operating characteristics, world by world (%s, %d seeds per row)" % (generated_utc, seeds), "",
          "The runner is the code that will score real data. Each row runs it many times in one simulated world. Never evidence about any real system.", "",
          "## P5 (arc_runner.p5)", "",
          "The route-agreement columns and the identification columns answer different questions and "
          "are tabulated separately (finding A6). Route agreement is a numerical consistency check "
          "between two regression directions through one bank: it tests whether the elasticity in "
          "the retention fraction and the elasticity in the capability state are one number. "
          "Identification asks whether the capability elasticity is identified at all, and reaches "
          "IDENTIFIED only on an independent capability manipulation, which none of these simulated "
          "worlds supplies, so NOT ESTABLISHED is the correct reading of every row that carries it.", "",
          "| World | true beta | SUPPORTED | REFUTED | INCONCLUSIVE | NOT EVALUABLE | routes CONSISTENT | routes INCONSISTENT | routes UNRESOLVED | ident NOT IDENTIFIED | ident NOT ESTABLISHED | premise HOLDS | premise NOT REFUTED | premise REFUTED | mean beta pooled (sd) | mean capability elasticity (sd) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows5:
        p, i, q = r["prediction"], r["identification"], r["premise"]
        g = r.get("route_agreement", {})
        L_.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %s | %s |" % (
                  r["world"], r["true_beta"], p.get("SUPPORTED", 0), p.get("REFUTED", 0), p.get("INCONCLUSIVE", 0), p.get("NOT EVALUABLE", 0),
                  g.get("CONSISTENT", 0), g.get("INCONSISTENT", 0), g.get("UNRESOLVED", 0),
                  i.get("NOT IDENTIFIED", 0), i.get("NOT ESTABLISHED", 0),
                  q.get("HOLDS", 0), q.get("NOT REFUTED", 0), q.get("REFUTED (curvature)", 0),
                  ("%.3f (%.3f)" % (r["mean_beta_pooled"], r["sd_beta_pooled"])) if r["mean_beta_pooled"] is not None else "n/a",
                  ("%.3f (%.3f)" % (r["mean_beta_capability"], r["sd_beta_capability"])) if r.get("mean_beta_capability") is not None else "n/a"))
    L_ += ["", "### Uncertainty on the P5 rates (exact two-sided 95 per cent intervals; %d outer repetitions per row; %d inner resamples per run inside the analysis, which do not narrow these)" % (seeds, rows5[0]["inner_resamples_per_run"] if rows5 else 0), "",
           "| World | headline | exact interval | one-sided upper on the OTHER verdicts |", "|---|---|---|---|"]
    for r in rows5:
        pc = r["prediction_ci"]
        head = max(pc, key=lambda k: pc[k]["rate"]) if pc else "n/a"
        others = ", ".join("%s at most %.3f" % (k, pc[k]["exact_one_sided_upper"]) for k in sorted(pc) if k != head) or "none observed"
        if pc:
            L_.append("| %s | %s %.2f | [%.3f, %.3f] | %s |" % (r["world"], head, pc[head]["rate"], pc[head]["exact_two_sided"][0], pc[head]["exact_two_sided"][1], others))
        # verdicts never observed in this world still carry an upper bound: zero of n is not zero risk
        absent = [k for k in ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "NOT EVALUABLE") if k not in pc]
        if absent:
            L_.append("| %s (unobserved) | %s | 0 of %d | each at most %.4f |" % (r["world"], ", ".join(absent), seeds, MC.exact_upper(0, seeds)))
    # the sixth column arrived with finding A3: a run whose above-boundary arms simply fell silent no
    # longer refutes, because silence is not a measured absence. It refutes only where the silence
    # demonstrated a positive margin past the informative horizon, which is why the two are separate
    # columns and never one.
    L_ += ["", "## P16 (arc_runner.p16)", "", "| World | SUPPORTED | REFUTED (no reversal) | REFUTED (mislocated) | NOT SPECIFIC | INCONCLUSIVE | INCONCLUSIVE (silent, nothing demonstrated) |", "|---|---|---|---|---|---|---|"]
    for r in rows16:
        v = r["verdicts"]
        L_.append("| %s | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |" % (r["world"], v.get("SUPPORTED", 0), v.get("REFUTED (no reversal)", 0), v.get("REFUTED (boundary mislocated)", 0), v.get("NOT SPECIFIC (INCONCLUSIVE)", 0), v.get("INCONCLUSIVE", 0), v.get("INCONCLUSIVE (no reversal, and none demonstrated absent)", 0)))
    L_ += ["", "### Uncertainty on the P16 rates (exact two-sided 95 per cent intervals; %d outer repetitions per row; the detection rule uses no inner resampling)" % seeds, "",
           "| World | headline | exact interval | unobserved verdicts, each at most |", "|---|---|---|---|"]
    for r in rows16:
        vc = r["verdicts_ci"]
        head = max(vc, key=lambda k: vc[k]["rate"]) if vc else "n/a"
        if vc:
            L_.append("| %s | %s %.2f | [%.3f, %.3f] | %.4f |" % (r["world"], head, vc[head]["rate"], vc[head]["exact_two_sided"][0], vc[head]["exact_two_sided"][1], MC.exact_upper(0, seeds)))
    L_ += ["", "A rate of zero in %d runs is not zero risk: its one-sided 95 per cent upper bound is %.4f, and it says nothing about worlds this battery did not run." % (seeds, MC.exact_upper(0, seeds))]
    return "\n".join(L_) + "\n"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--seeds", type=int, default=40)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
    rows5 = [run_p5_world(w, a.seeds) for w in P5_WORLDS]
    rows16 = [run_p16_world(w, a.seeds) for w in P16_WORLDS]
    out = {"generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "seeds_per_row": a.seeds,
           "p5": rows5, "p16": rows16, "seconds": round(time.time() - t0, 1),
           "note": "Operating characteristics of the runner's own analysis code under simulated systems. Targets (ruling 10): false affirmative at most 0.05, detection at least 0.80. Never evidence about any real system."}
    json.dump(out, open(os.path.join(a.out, "RUNNER-BATTERY.json"), "w"), indent=1)
    open(os.path.join(a.out, "RUNNER-BATTERY.md"), "w").write(render_report(rows5, rows16, a.seeds, out["generated_utc"]))
    print(json.dumps({"p5_rows": len(rows5), "p16_rows": len(rows16), "seconds": out["seconds"]}))

if __name__ == "__main__":
    sys.exit(main())
