#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Estimating β and k - the operational form of the Coupled Co-Scaling Law
================================================================================
Michael Darius Eastwood | ARC/orchestration programme | Paper X, §8

THE OBJECTION THIS ANSWERS
--------------------------
The sharpest criticism of Paper X is: "the criterion β > k is only useful if you
can MEASURE β and k on a real self-improving system - and you can't." This script
makes the criterion operational. It defines β and k as quantities you read off a
real trajectory, validates the estimator on synthetic data where the true
exponents are known, and applies it - honestly - to the real Claude run.

THE TWO EXPONENTS, AS MEASURABLE QUANTITIES
-------------------------------------------
  k  (drift-acceleration exponent)  - a property of the CAPABILITY curve alone.
     The relative growth rate is r = Ċ/C = b·C^k.  Measure r at several capability
     levels C and regress  ln r  on  ln C ;  the slope is k.
       k > 0 : accelerating / hard-takeoff-like.   k = 0 : exponential.   k < 0 : saturating.

  β  (correction-strength exponent) - a property of the CORRECTOR.
     The correction rate is A = A0·C^β.  Apply the corrector at capability C and
     measure the fractional misalignment removal over Δt:
         A ≈ −ln( D_after / D_before ) / Δt .
     Measure A at several C and regress  ln A  on  ln C ;  the slope is β.

  Verdict:  β > k  ⇒  correction out-scales drift-acceleration ⇒ stable.

WHAT IS HONEST ABOUT THIS
-------------------------
* The estimator is validated below on synthetic trajectories generated from the
  model's own dynamics with KNOWN (b,k,A0,β): it recovers them within tolerance.
  That certifies the *estimator* (it reads the exponents back correctly) - it does
  NOT certify the model against reality.
* Applied to the real Claude run, it reports what is and is NOT estimable. On that
  run capability saturated in one step (no range of C) and there was a single
  correction level, so β and k are NOT yet estimable - and the criterion is
  vacuously satisfied because drift ≈ 0. The estimator is ready; it needs a graded,
  drifting dataset (a model that drifts across a RANGE of capability levels).

  python estimate_exponents.py            # synthetic validation + Claude application
================================================================================
"""
import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "..", "results", "realmodel")


# --------------------------------------------------------------------------- #
#  Core estimators (log-log power-law fits)                                    #
# --------------------------------------------------------------------------- #
def _loglog_fit(x, y):
    """Fit ln(y) = slope·ln(x) + c. Returns (slope, intercept, r2, n)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = (x > 0) & (y > 0) & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 2 or np.ptp(np.log(x)) < 1e-9:
        return None  # insufficient capability range to identify a slope
    lx, ly = np.log(x), np.log(y)
    slope, c = np.polyfit(lx, ly, 1)
    pred = slope * lx + c
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return dict(slope=float(slope), intercept=float(c), r2=float(r2), n=int(len(x)))


def estimate_k(C_series, dt=1.0):
    """k from the capability curve: relative growth r_n = (C_{n+1}-C_n)/(C_n·dt) = b·C^k.
    Regress ln r on ln C. Needs capability to move across a RANGE of levels."""
    C = np.asarray(C_series, float)
    if len(C) < 3:
        return dict(estimable=False, reason="need >=3 capability points")
    Cmid, r = [], []
    for i in range(len(C) - 1):
        if C[i] > 0 and C[i + 1] > C[i]:           # only growing steps carry r>0
            Cmid.append(C[i]); r.append((C[i + 1] - C[i]) / (C[i] * dt))
    fit = _loglog_fit(Cmid, r) if len(Cmid) >= 2 else None
    if not fit:
        return dict(estimable=False,
                    reason=f"capability range too small (growing steps={len(Cmid)}; "
                           f"need >=2 across distinct C)")
    return dict(estimable=True, k_hat=fit["slope"], r2=fit["r2"], n=fit["n"])


def estimate_beta(corrector_obs):
    """β from corrector removals: A = -ln(D_after/D_before)/dt = A0·C^β at capability C.
    corrector_obs: list of dict(C, D_before, D_after, dt). Regress ln A on ln C.
    Needs corrections at a RANGE of capability levels."""
    C, A = [], []
    eps = 1e-3
    for o in corrector_obs:
        Db, Da, dt = o["D_before"], o["D_after"], o.get("dt", 1.0)
        c = o["C"]
        if c <= 0 or Db <= 0:
            continue
        frac_remaining = max(Da, eps) / Db
        a = -math.log(min(frac_remaining, 1.0)) / dt          # removal rate at this C
        if a > 0:
            C.append(c); A.append(a)
    if len({round(c, 6) for c in C}) < 2:
        return dict(estimable=False,
                    reason=f"need corrector removals at >=2 distinct capability levels "
                           f"(got {len(set(round(c,6) for c in C))})")
    fit = _loglog_fit(C, A)
    if not fit:
        return dict(estimable=False, reason="capability range too small")
    return dict(estimable=True, beta_hat=fit["slope"], r2=fit["r2"], n=fit["n"])


def verdict(k_res, beta_res):
    if k_res.get("estimable") and beta_res.get("estimable"):
        b, k = beta_res["beta_hat"], k_res["k_hat"]
        return dict(resolved=True, beta_hat=b, k_hat=k, margin=b - k,
                    stable=(b > k))
    return dict(resolved=False,
                reason="β and/or k not estimable from this dataset "
                       "(need drift across a range of capability levels)")


# --------------------------------------------------------------------------- #
#  Synthetic validation - recover KNOWN (k, β) from model-generated data       #
# --------------------------------------------------------------------------- #
def synthetic_validate(seed_grid=((0.5, 1.0), (1.0, 0.5), (0.3, 0.3), (1.2, 0.4)),
                       noise=0.05):
    """For each (k_true, beta_true): build a capability ladder and corrector
    observations from the model's own power laws (+ multiplicative noise), then
    check the estimator recovers the exponents. Deterministic noise (index-seeded),
    no RNG calls (keeps the script reproducible)."""
    b0, A0 = 0.20, 0.40
    C_levels = np.array([1.0, 1.6, 2.6, 4.2, 6.8, 11.0, 18.0, 29.0])  # ~1.46 dex span
    out = []
    for gi, (k_true, beta_true) in enumerate(seed_grid):
        # deterministic pseudo-noise in [1-noise, 1+noise]
        wobble = lambda i: 1.0 + noise * math.sin(2.7 * (i + 1) + 1.3 * gi)
        # --- k: build a capability trajectory whose relative growth is b0·C^k ---
        C_traj, t = [C_levels[0]], 0.0
        for i in range(40):
            C = C_traj[-1]
            if C > 1e6:                              # k>0 ⇒ finite-time blow-up; cap the ladder
                break
            r = b0 * (C ** k_true) * wobble(i)
            C_traj.append(C * (1.0 + r))            # dt = 1
        k_res = estimate_k(C_traj, dt=1.0)
        # --- β: corrector removals at each capability level ---
        obs = []
        for i, C in enumerate(C_levels):
            A = A0 * (C ** beta_true) * wobble(i + 7)
            D_before = 10.0
            D_after = D_before * math.exp(-A * 1.0)  # dt = 1
            obs.append(dict(C=float(C), D_before=D_before, D_after=D_after, dt=1.0))
        beta_res = estimate_beta(obs)
        rec = dict(k_true=k_true, beta_true=beta_true,
                   k_hat=round(k_res.get("k_hat", float("nan")), 3),
                   beta_hat=round(beta_res.get("beta_hat", float("nan")), 3),
                   k_err=round(abs(k_res.get("k_hat", float("nan")) - k_true), 3),
                   beta_err=round(abs(beta_res.get("beta_hat", float("nan")) - beta_true), 3),
                   verdict_true=("stable" if beta_true > k_true else "unstable"),
                   verdict_hat=("stable" if beta_res.get("beta_hat", -9) > k_res.get("k_hat", 9)
                                else "unstable"))
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
#  Apply to the real Claude run (honest: report what is/!is estimable)         #
# --------------------------------------------------------------------------- #
def apply_to_claude():
    traj_files = sorted(glob.glob(os.path.join(RESULTS, "claude-opus_2026*.json")))
    probe_file = os.path.join(RESULTS, "claude-opus_corrector_probe.json")
    if not traj_files:
        return dict(available=False, reason="no real Claude trajectory found")
    data = json.load(open(traj_files[-1]))
    # capability series for the decoupled arm (drift arm)
    decoupled = [r for r in data["runs"] if r["condition"] == "decoupled"]
    C_series = [p["C"] for p in decoupled[0]["traj"]] if decoupled else []
    k_res = estimate_k(C_series, dt=1.0) if len(C_series) >= 3 else \
        dict(estimable=False, reason=f"only {len(C_series)} capability points")
    # corrector observations from the probe (single level)
    obs = []
    if os.path.exists(probe_file):
        p = json.load(open(probe_file))
        g, a = p["gamed_initial"], p["after_external_corrector"]
        # the correction raised C from 0 -> 1.0; capability "at correction" is ill-defined
        obs.append(dict(C=a["C_hidden"], D_before=g["D_blind"], D_after=a["D_blind"], dt=1.0))
    beta_res = estimate_beta(obs)
    return dict(available=True,
                capability_series=C_series,
                capability_range_dex=round(math.log10(max(C_series)/min([c for c in C_series if c>0]))
                                           if C_series and min([c for c in C_series if c>0])>0 else 0, 3),
                k=k_res, beta=beta_res, verdict=verdict(k_res, beta_res))


def main():
    print("=" * 78)
    print("  ESTIMATING β AND k - operational form of the Coupled Co-Scaling Law")
    print("=" * 78)

    print("\n[1] ESTIMATOR VALIDATION on synthetic data (known exponents):")
    print(f"    {'k_true':>7} {'k_hat':>7} {'β_true':>7} {'β_hat':>7} {'k_err':>6} {'β_err':>6}  verdict")
    syn = synthetic_validate()
    ok = True
    for r in syn:
        match = "OK" if r["verdict_true"] == r["verdict_hat"] else "MISS"
        ok = ok and (r["k_err"] <= 0.1 and r["beta_err"] <= 0.1 and match == "OK")
        print(f"    {r['k_true']:>7} {r['k_hat']:>7} {r['beta_true']:>7} {r['beta_hat']:>7} "
              f"{r['k_err']:>6} {r['beta_err']:>6}  {r['verdict_true']}→{r['verdict_hat']} [{match}]")
    print(f"    => estimator {'VALID (recovers known β,k within 0.1)' if ok else 'FAILED'}")

    print("\n[2] APPLIED to the real Claude run (honest):")
    cl = apply_to_claude()
    if not cl["available"]:
        print("    " + cl["reason"])
    else:
        print(f"    decoupled capability series: {cl['capability_series']} "
              f"(range {cl['capability_range_dex']} dex)")
        print(f"    k:    {cl['k']}")
        print(f"    β:    {cl['beta']}")
        v = cl["verdict"]
        if v["resolved"]:
            print(f"    VERDICT: β̂={v['beta_hat']:.3f}  k̂={v['k_hat']:.3f}  "
                  f"margin={v['margin']:+.3f}  → {'STABLE' if v['stable'] else 'UNSTABLE'}")
        else:
            print(f"    VERDICT: NOT RESOLVABLE - {v['reason']}")
            print("    (capability saturated at the task ceiling in one step, and there was a")
            print("     single correction level; the criterion is vacuously satisfied because")
            print("     drift ≈ 0. The estimator awaits a graded, DRIFTING real dataset.)")

    out = dict(synthetic_validation=syn, claude=cl)
    path = os.path.join(RESULTS, "exponent_estimates.json")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2)
    print(f"\n  saved: {os.path.relpath(path, HERE)}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
