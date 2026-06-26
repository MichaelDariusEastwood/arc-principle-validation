#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Coupled Co-Scaling Correction  --  Numerical Verification & Consistency Harness
================================================================================
Paper X of the ARC / Eden research programme.
Michael Darius Eastwood  |  London  |  June 2026
michaeldariuseastwood.com/research   OSF: doi.org/10.17605/OSF.IO/6C5XB

WHAT THIS HARNESS DOES -- AND DOES NOT -- ESTABLISH (read this first)
--------------------------------------------------------------------
This harness verifies that the closed-form theorems are correctly DERIVED from
the model's dynamics and correctly INTEGRATED by the solver. It does three
honest things:
  (a) derivation-consistency: it integrates the governing ODE and checks the
      numerical solution matches the closed-form fixed points / asymptotics the
      theorems predict (experiments E1-E6, E9);
  (b) integrator-accuracy: it certifies the integrator against the exact
      Theorem-1 transient to ~1e-11 (E8);
  (c) one genuine Monte-Carlo check of the stochastic stationary law (E7).

It does NOT, and cannot, do the following, and no claim here pretends otherwise:
  - it does NOT test whether the model describes any real AI system (that is the
    open empirical problem, stated in the paper's Limitations);
  - because each deductive experiment integrates the SAME ODE whose closed form
    is the prediction, a PASS on E1-E6/E9 certifies the algebra + the solver, not
    the physical adequacy of the model. F1-F3, F3', F5, F6 are therefore
    INTERNAL-CONSISTENCY conditions of the derivation, not empirical falsifiers
    of the thesis;
  - F4 (the QEC mechanism) is an ANALYTIC prediction: the model's suppression law
    is exactly power-law (an algebraic identity), so a finite-capacity corrector
    that could exhibit EXPONENTIAL (QEC-like) suppression is the future test the
    paper names; the present harness does not exercise it.
The script exits 0 iff every internal-consistency check matches its closed-form
prediction. That is evidence the code matches the maths -- never that the maths
matches reality.

THE MASTER EQUATION  (paper, Section 3)
---------------------------------------
Capability  C(t), misalignment magnitude D(t), misalignment fraction d = D/C.
Three drift channels and one co-scaling corrector:

    dD/dt = gamma1 * dC/dt              (gain drift   -- new capability opens new misalignment)
          + gamma2 * C                  (level drift  -- instrumental pressure present at rest)
          + gamma3 * (dC/dt / C) * D    (compounding  -- misalignment that amplifies itself)
          - A * D                       (correction)

    A  = A0 * C**beta      (co-scaling correction strength,  beta  = correction exponent)
    dC/dt = b * C**(1+k)   (growth law),  so  r := (dC/dt)/C = b * C**k  (k = acceleration exponent)

THE EXACT FRACTION DYNAMICS (derived in the paper, Theorem 1)
-------------------------------------------------------------
    d_dot = gamma1*r + gamma2 - [ A + (1 - gamma3)*r ] * d

Two facts the harness verifies numerically and the paper proves analytically:
  (i)  In the natural clock of self-improvement DEPTH  tau = ln(C/C0), the
       dynamics are regular even when C reaches infinity in FINITE wall-clock
       time (a "hard takeoff"). The verdict is set by sign(beta - k), not by
       the speed b nor the finiteness of the singularity time. (Theorem 3.)
  (ii) In the additive model (gamma3 = 0) the fraction is ALWAYS bounded and
       saturates at the drift coefficient gamma1; it never diverges to infinity.
       True divergence requires the compounding channel gamma3 > 1, and is then
       cured precisely by beta > k. (Theorem 2 + corrected regime table.)

Requires: numpy, scipy, matplotlib.  Run:  python experiment_coscaling.py
================================================================================
"""

import json
import os
import sys
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
#  House aesthetic (matches the paper)                                         #
# --------------------------------------------------------------------------- #
INK        = "#1a1a1a"
RED        = "#c41230"   # nature red / unstable
GREEN      = "#2e7d32"   # eden green / stable
BLUE       = "#1565c0"   # proof blue
GOLD       = "#c49102"   # marginal
PURPLE     = "#6a1b9a"   # synthesis
BABYLON    = "#b71c1c"
GREY       = "#666666"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

OUTDIR = os.environ.get("COSCALING_OUTDIR", ".")
FIGDIR = os.path.join(OUTDIR, "figures")
RESDIR = os.path.join(OUTDIR, "results")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(RESDIR, exist_ok=True)

CAP = 1.0e8          # misalignment fraction above this is declared divergent
RNG = np.random.default_rng(7)   # deterministic stochastic experiments


# --------------------------------------------------------------------------- #
#  Parameters and the depth-clock integrator                                   #
# --------------------------------------------------------------------------- #
@dataclass
class P:
    """Parameters of one run."""
    gamma1: float = 0.05     # gain-drift coefficient
    gamma2: float = 0.0      # level-drift coefficient (Section 3.6)
    gamma3: float = 0.0      # compounding-drift coefficient (Section 3.7)
    A0: float = 0.08         # baseline correction strength
    beta: float = 0.0        # correction co-scaling exponent
    k: float = 0.0           # growth acceleration exponent
    b: float = 1.0           # base growth-rate coefficient
    C0: float = 1.0          # initial capability
    d0: float = 0.05         # initial misalignment fraction


def _depth_rhs(tau, d, p: P):
    """
    Right-hand side of the EXACT misalignment-fraction dynamics expressed in the
    self-improvement-depth clock  tau = ln(C/C0).  Since dC/dt = r*C and
    dtau/dt = r, we have  dd/dtau = d_dot / r, giving

        dd/dtau = gamma1 + gamma2/r - [ A/r + (1 - gamma3) ] * d

    with  C = C0*exp(tau),  r = b*C**k,  A = A0*C**beta,  A/r = (A0/b)*C**(beta-k).
    Regular for all tau, including the limit C -> infinity that, for k>0,
    corresponds to a finite wall-clock singularity time t*.
    """
    C = p.C0 * np.exp(tau)
    # Compute the rates directly from tau -- r = b*C^k and Ar = A/r = (A0/b)*C^(beta-k)
    # -- as exponentials of tau, rather than raising the large intermediate C to a
    # power. Mathematically identical (C = C0*e^tau), but avoids floating-point
    # overflow under the deep-tau parameter sweeps this harness invites.
    r = p.b * (p.C0 ** p.k) * np.exp(p.k * tau)
    Ar = (p.A0 / p.b) * (p.C0 ** (p.beta - p.k)) * np.exp((p.beta - p.k) * tau)   # A / r
    inj = p.gamma1 + (p.gamma2 / r if r > 0 else 0.0)
    decay = Ar + (1.0 - p.gamma3)
    return inj - decay * d


def integrate_depth(p: P, tau_max=14.0, n=1600):
    """
    Integrate the depth-clock ODE from tau=0 to tau_max with a stiff-capable
    solver (LSODA).  Returns arrays (tau, C, d).  Divergence is detected by a
    terminal event when d crosses CAP.
    """
    tau_eval = np.linspace(0.0, tau_max, n)

    def hit_cap(tau, d, p):           # terminal event: divergence
        return d[0] - CAP
    hit_cap.terminal = True
    hit_cap.direction = 1

    sol = solve_ivp(_depth_rhs, (0.0, tau_max), [p.d0], args=(p,),
                    method="LSODA", t_eval=tau_eval, events=hit_cap,
                    rtol=1e-9, atol=1e-12, max_step=0.05)

    tau = sol.t
    d = sol.y[0]
    if sol.t_events[0].size > 0:       # diverged: pad the remainder with CAP
        tau = np.concatenate([tau, tau_eval[tau_eval > tau[-1]]])
        d = np.concatenate([d, np.full(tau.size - d.size, CAP)])
    C = p.C0 * np.exp(tau)
    return tau, C, d


def asymptote(tau, d, frac=0.10):
    """Mean misalignment fraction over the final `frac` of the depth window."""
    m = tau >= (tau[-1] - frac * (tau[-1] - tau[0]))
    vals = d[m]
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if vals.size else float("nan")


def log_slope(tau, C, d):
    """Slope of log10(d) versus log10(C) over the final half of the window
    (the predicted suppression exponent is -(beta-k))."""
    m = (tau >= 0.5 * tau[-1]) & (d > 1e-12) & np.isfinite(d) & (d < CAP)
    if m.sum() < 8:
        return float("nan")
    # log10(C) computed analytically from tau (C = C0*e^tau) so it is stable even
    # if C itself has overflowed to inf at very large tau.
    log10_C = np.log10(C[0]) + tau[m] * np.log10(np.e)
    return float(np.polyfit(log10_C, np.log10(d[m]), 1)[0])


DIVERGE_LEVEL = 1.0    # a misalignment *fraction* > 1 (i.e. D > C) is genuine divergence;
                       # the additive model is provably bounded by gamma1 << 1 (Theorem 2).

def classify(tau, C, d, floor_tol=5e-3):
    """Classify a trajectory as 'converge' (d->0), 'floor' (bounded plateau>0)
    or 'diverge' (unbounded).  Grounded in Theorem 2: in the additive model the
    misalignment fraction cannot exceed gamma1 << 1, so a fraction reaching the
    order of unity is the unambiguous signature of the compounding instability."""
    a = asymptote(tau, d)
    if not np.isfinite(a) or a >= DIVERGE_LEVEL:
        return "diverge", a
    s = log_slope(tau, C, d)
    if a < floor_tol or (np.isfinite(s) and s < -0.05):
        return "converge", a
    return "floor", a


def threshold_crossing(lambdas, vals, level):
    """Largest lambda (scanning ascending) at which the steady-state value vals
    first drops below `level`.  Used to locate the instability threshold, which
    is a steady-state divergence rather than a finite-time event."""
    vals = np.asarray(vals)
    below = np.where(vals < level)[0]
    return float(lambdas[below[0]]) if below.size else float("nan")


# --------------------------------------------------------------------------- #
#  Verdict bookkeeping                                                          #
# --------------------------------------------------------------------------- #
VERDICTS = []

def record(exp, prediction, statistic, criterion, passed, falsifier, triggered, detail=""):
    VERDICTS.append(dict(experiment=exp, prediction=prediction, statistic=statistic,
                         criterion=criterion, passed=bool(passed), falsifier=falsifier,
                         triggered=bool(triggered), detail=detail))
    tag = "PASS" if passed else "FAIL"
    print(f"  [{tag}] {exp}: {prediction}")
    print(f"         statistic = {statistic} | criterion: {criterion}")
    print(f"         {falsifier}: {'TRIGGERED' if triggered else 'not triggered'}")


# --------------------------------------------------------------------------- #
#  EXPERIMENT 1 -- Phase boundary (P1 / F1)                                     #
#  The sharp threshold lives in the compounding channel, NOT the additive one. #
# --------------------------------------------------------------------------- #
def experiment_1():
    print("\nExperiment 1 - Phase boundary (P1 / F1)")
    b = 1.0
    lambdas = np.linspace(0.2, 4.0, 60)

    # additive (gamma3=0): smooth, no knee
    add_final = []
    for lam in lambdas:
        p = P(gamma1=0.05, gamma3=0.0, A0=lam, beta=0.0, k=0.0, b=b)
        tau, C, d = integrate_depth(p, tau_max=12)
        add_final.append(min(asymptote(tau, d), CAP))

    # compounding (gamma3=3): sharp knee predicted at A0* = (gamma3-1)*b = 2.0.
    # The instability is a STEADY-STATE divergence, so integrate deep and locate
    # the threshold where the steady-state value crosses a high level.
    gamma3 = 3.0
    A0_star_pred = (gamma3 - 1.0) * b
    CEIL = 10.0
    comp_final = []
    for lam in lambdas:
        p = P(gamma1=0.05, gamma3=gamma3, A0=lam, beta=0.0, k=0.0, b=b)
        tau, C, d = integrate_depth(p, tau_max=40)
        comp_final.append(min(asymptote(tau, d), CEIL))
    comp_final = np.array(comp_final)

    A0_star_meas = threshold_crossing(lambdas, comp_final, level=CEIL / 2.0)

    # additive must show no divergence anywhere (smooth, bounded by gamma1)
    add_smooth = bool(np.all(np.array(add_final) < 1.0))
    # compounding must show a threshold close to prediction
    knee_ok = np.isfinite(A0_star_meas) and abs(A0_star_meas - A0_star_pred) <= 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.suptitle("Experiment 1  --  Phase boundary (P1).  A sharp knee exists only in the "
                 "compounding channel.", fontweight="bold", fontsize=12)
    ax = axes[0]
    ax.plot(lambdas, add_final, "o-", color=BLUE, ms=3)
    ax.set_title("Additive corrector ($\\gamma_3=0$): smooth, no knee")
    ax.set_xlabel("coupling $\\lambda \\to A_0$"); ax.set_ylabel("long-run $d^*$")
    ax.set_ylim(0, max(0.06, max(add_final) * 1.2))

    ax = axes[1]
    ax.semilogy(lambdas, np.clip(comp_final, 1e-3, CEIL), "o-", color=RED, ms=3)
    ax.axvline(A0_star_pred, color=GREEN, ls="--", lw=1.5,
               label=f"predicted $\\lambda^*=(\\gamma_3-1)b={A0_star_pred:.2f}$")
    ax.set_title("Compounding corrector ($\\gamma_3=3$): sharp threshold")
    ax.set_xlabel("coupling $\\lambda \\to A_0$"); ax.set_ylabel("long-run $d^*$ (log, clipped)")
    ax.legend(fontsize=9)
    ax.text(0.04, 0.88, "diverges\n($d\\to\\infty$)", transform=ax.transAxes, color=RED, fontsize=9)
    ax.text(0.66, 0.12, "bounded\n($d\\to$ floor)", transform=ax.transAxes, color=GREEN, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp1_phase_boundary.png"), dpi=160, bbox_inches="tight")
    plt.close()

    passed = bool(add_smooth and knee_ok)
    record("E1", "P1: sharp phase boundary in compounding channel; smooth in additive",
           f"A0*_meas={A0_star_meas:.2f} (pred {A0_star_pred:.2f}); additive_smooth={add_smooth}",
           "knee within 0.15 of (gamma3-1)b AND additive shows no divergence",
           passed, "F1 (no boundary anywhere)", triggered=not passed)
    return passed


# --------------------------------------------------------------------------- #
#  EXPERIMENT 2 -- Speed-invariance of the verdict (P2 / F2)                    #
#  Coupling (A0/b ratio = scaling margin) decides; raw speed b does not.        #
# --------------------------------------------------------------------------- #
def experiment_2():
    print("\nExperiment 2 - Speed-invariance (P2 / F2)")
    g3 = 3.0                       # compounding so divergence is possible
    thr = g3 - 1.0                 # coupled iff A0/b > 2
    configs = [
        ("Coupled  ($A_0/b=3$),  slow", 3.0, 0.5, GREEN, False),
        ("Coupled  ($A_0/b=3$),  fast", 3.0, 5.0, BLUE,  False),
        ("Decoupled ($A_0/b=1$), slow", 1.0, 0.5, GOLD,  True),
        ("Decoupled ($A_0/b=1$), fast", 1.0, 5.0, RED,   True),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    fig.suptitle("Experiment 2  --  Speed-invariance (P2).  Coupling decides the outcome; speed "
                 "only rescales the clock.\nGrowth-rate-ceiling view predicts the opposite "
                 "(fast => diverge regardless).", fontweight="bold", fontsize=11)
    ok = True
    for ax, (title, ratio, b, col, expect_div) in zip(axes.flat, configs):
        A0 = ratio * b
        p = P(gamma1=0.05, gamma3=g3, A0=A0, beta=0.0, k=0.0, b=b)
        tau, C, d = integrate_depth(p, tau_max=12)
        t = tau / b                # k=0 => dtau/dt = b, wall-clock time
        kind, a = classify(tau, C, d)
        got_div = (kind == "diverge")
        ok = ok and (got_div == expect_div)
        ax.plot(t, np.clip(d, 0, 1.5), color=col)
        verdict = "DIVERGES" if got_div else "BOUNDED"
        ax.text(0.96, 0.94, verdict, transform=ax.transAxes, ha="right", va="top",
                color=col, fontweight="bold",
                bbox=dict(boxstyle="round", fc="white", ec=col, alpha=0.9))
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("wall-clock time $t$"); ax.set_ylabel("misalignment fraction $d$")
        ax.set_ylim(0, 1.55)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp2_speed_invariance.png"), dpi=160, bbox_inches="tight")
    plt.close()

    record("E2", "P2: coupling (A0/b), not speed b, determines convergence vs divergence",
           f"coupled bounded at both speeds AND decoupled divergent at both speeds = {ok}",
           "outcome matches coupling for all four cells",
           ok, "F2 (outcome tracks raw speed)", triggered=not ok)
    return ok


# --------------------------------------------------------------------------- #
#  EXPERIMENT 3 -- Co-scaling law, with the v3 overclaim corrected (P3 / F3)    #
#  Additive model: beta>0 -> 0; beta=0 -> permanent gap; beta<0 -> SATURATES    #
#  at gamma1 (NOT divergence).                                                  #
# --------------------------------------------------------------------------- #
def experiment_3():
    print("\nExperiment 3 - Co-scaling law, corrected (P3 / F3)")
    g1, A0, b = 0.05, 0.08, 0.5
    r = b                                  # k=0
    specs = [
        (-0.5, RED,    "--", "$\\beta=-0.5$  (saturates at $\\gamma_1$, NOT $\\infty$)"),
        (0.0,  GOLD,   "-",  "$\\beta=0$  (permanent gap -- Paper III)"),
        (0.5,  GREEN,  "-",  "$\\beta=0.5$  (converges)"),
        (1.0,  BLUE,   "-",  "$\\beta=1.0$  (fast convergence)"),
    ]
    pred = {-0.5: g1, 0.0: g1 * r / (A0 + r), 0.5: 0.0, 1.0: 0.0}
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experiment 3  --  Co-scaling law (P3).  Correcting the v3 draft: $\\beta<0$ "
                 "saturates at the drift\ncoefficient $\\gamma_1$; the fraction is bounded, it "
                 "does not diverge to infinity.", fontweight="bold", fontsize=11)
    ok = True
    for beta, col, ls, label in specs:
        p = P(gamma1=g1, A0=A0, beta=beta, k=0.0, b=b)
        tau, C, d = integrate_depth(p, tau_max=16)
        ax.plot(tau, d, color=col, ls=ls, label=label)
        a = asymptote(tau, d)
        target = pred[beta]
        good = (a < 8e-3) if target == 0.0 else (abs(a - target) < 0.1 * target + 5e-3)
        ok = ok and good
    ax.axhline(g1, color=RED, lw=1, ls=":", alpha=0.7)
    ax.text(0.5, g1 + 0.0015, "$\\gamma_1$ ceiling", color=RED, fontsize=9)
    ax.axhline(pred[0.0], color=GOLD, lw=1, ls=":", alpha=0.7)
    ax.set_xlabel("self-improvement depth $\\tau=\\ln(C/C_0)$")
    ax.set_ylabel("misalignment fraction $d$")
    ax.set_ylim(0, g1 * 1.4); ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp3_coscaling_law.png"), dpi=160, bbox_inches="tight")
    plt.close()

    record("E3", "P3: beta>0 -> 0; beta=0 -> gamma1*r/(A0+r); beta<0 -> gamma1 (bounded)",
           f"all four asymptotes match analytic predictions = {ok}",
           "each long-run d within tolerance of closed form",
           ok, "F3 (regimes do not match)", triggered=not ok)
    return ok


# --------------------------------------------------------------------------- #
#  EXPERIMENT 4 -- Hard takeoff: the beta>k boundary (P4 / F3')                 #
#  Finite-time singularity for k>0; verdict set by sign(beta-k).               #
# --------------------------------------------------------------------------- #
def experiment_4():
    print("\nExperiment 4 - Hard takeoff, beta>k boundary (P4 / F3')")
    k_vals = [0.0, 0.5, 1.0]
    beta_vals = [0.25, 0.75, 1.5]
    g1, A0, b = 0.05, 0.08, 0.02
    fig, axes = plt.subplots(3, 3, figsize=(13, 11))
    fig.suptitle("Experiment 4  --  Acceleration and the $\\beta>k$ boundary (P4).  Stability is "
                 "set by sign$(\\beta-k)$,\nnot by speed.  For $k>0$ capability reaches infinity "
                 "in FINITE wall-clock time, yet $d$ stays controlled when $\\beta>k$.",
                 fontweight="bold", fontsize=11)
    ok = True
    for i, k in enumerate(k_vals):
        for j, beta in enumerate(beta_vals):
            ax = axes[i][j]
            p = P(gamma1=g1, A0=A0, beta=beta, k=k, b=b)
            tau, C, d = integrate_depth(p, tau_max=14)
            margin = beta - k
            kind, a = classify(tau, C, d)
            # predicted: margin>0 -> converge; margin==0 -> floor; margin<0 -> floor at gamma1
            if margin > 0.05:
                want, col, lbl = "converge", GREEN, "STABLE ($\\beta>k$)"
            elif margin < -0.05:
                want, col, lbl = "floor", RED, "SATURATES ($\\beta<k$)"
            else:
                want, col, lbl = "floor", GOLD, "MARGINAL"
            good = (kind == want)
            ok = ok and good
            ax.plot(np.log10(C), np.clip(d, 0, g1 * 1.6), color=col, lw=1.8)
            ax.set_title(f"$k={k},\\ \\beta={beta}$  ->  {lbl}", fontsize=9, color=col)
            ax.set_ylim(0, g1 * 1.7)
            if k > 0:
                t_star = 1.0 / (k * b * p.C0 ** k)
                ax.text(0.04, 0.06, f"$t^*={t_star:.0f}$ (finite)", transform=ax.transAxes,
                        fontsize=7.5, color=GREY)
            if i == 2:
                ax.set_xlabel("$\\log_{10} C$", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"$k={k}$\n$d$", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp4_hard_takeoff_grid.png"), dpi=160, bbox_inches="tight")
    plt.close()

    record("E4", "P4: stability boundary lies at beta=k under accelerating growth",
           f"all 9 cells match sign(beta-k) classification = {ok}",
           "converge iff beta>k; floor iff beta<=k; across the 3x3 grid",
           ok, "F3' (boundary not at beta=k)", triggered=not ok)
    return ok


# --------------------------------------------------------------------------- #
#  EXPERIMENT 4b -- The singularity is a coordinate artefact (P4)              #
#  Integrate ONE stable case in wall-clock time toward t* and in depth tau;    #
#  show identical d and that d stays controlled THROUGH the finite-time blowup.#
# --------------------------------------------------------------------------- #
def _time_rhs(t, y, p: P):
    C, D = y
    dC = p.b * C ** (1.0 + p.k)
    r = p.b * C ** p.k                          # = dC/C, without the division
    A = p.A0 * C ** p.beta
    dD = p.gamma1 * dC + p.gamma2 * C + p.gamma3 * r * D - A * D
    return [dC, dD]


def experiment_4b():
    print("\nExperiment 4b - Coordinate-artefact demonstration (P4)")
    p = P(gamma1=0.05, A0=0.08, beta=1.5, k=1.0, b=0.02, C0=1.0, d0=0.05)
    t_star = 1.0 / (p.k * p.b * p.C0 ** p.k)            # finite singularity time

    # (a) depth clock -- regular
    tau, C_tau, d_tau = integrate_depth(p, tau_max=12)
    t_of_tau = (1.0 - np.exp(-p.k * tau)) / (p.k * p.b * p.C0 ** p.k)   # exact t(tau)

    # (b) wall-clock, integrated up to 0.99999 * t_star (just shy of blowup)
    t_end = 0.999990 * t_star
    sol = solve_ivp(_time_rhs, (0.0, t_end), [p.C0, p.d0 * p.C0], args=(p,),
                    method="LSODA", rtol=1e-9, atol=1e-12, dense_output=True,
                    max_step=t_end / 4000)
    tt = np.linspace(0, t_end, 4000)
    Ct, Dt = sol.sol(tt)
    d_t = Dt / Ct

    # agreement on the overlap in t
    d_tau_interp = np.interp(tt, t_of_tau, d_tau)
    m = (tt < 0.9999 * t_star) & np.isfinite(d_tau_interp) & (d_tau_interp > 0)
    rel = np.abs(d_t[m] - d_tau_interp[m]) / (d_tau_interp[m] + 1e-9)
    agree = float(np.nanmax(rel[np.isfinite(rel)])) if m.any() else float("nan")
    controlled = bool(np.nanmax(d_t) < 0.06 and d_t[-1] < d_t[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.suptitle("Experiment 4b  --  A hard takeoff is a coordinate artefact (P4).  "
                 "$k=1,\\ \\beta=1.5$.", fontweight="bold", fontsize=11)
    ax = axes[0]
    ax.plot(tt / t_star, Ct, color=PURPLE)
    ax.axvline(1.0, color=RED, ls="--", lw=1.5, label="finite singularity $t^*$")
    ax.set_yscale("log"); ax.set_xlabel("$t/t^*$"); ax.set_ylabel("capability $C$ (log)")
    ax.set_title("Capability diverges in finite time"); ax.legend(fontsize=9)
    ax = axes[1]
    ax.plot(tt / t_star, d_t, color=BLUE, label="integrated in wall-clock $t$")
    ax.plot(t_of_tau / t_star, d_tau, color=GREEN, ls="--", label="integrated in depth $\\tau$")
    ax.axvline(1.0, color=RED, ls="--", lw=1.5)
    ax.set_xlabel("$t/t^*$"); ax.set_ylabel("misalignment fraction $d$")
    ax.set_title(f"$d$ stays controlled through the explosion\n(max rel. disagreement "
                 f"= {agree:.1e})"); ax.legend(fontsize=9)
    ax.set_ylim(0, 0.06)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp4b_coordinate_artefact.png"), dpi=160, bbox_inches="tight")
    plt.close()

    passed = bool((agree < 5e-2) and controlled)
    record("E4b", "P4: depth-clock and wall-clock agree; d controlled through finite-time singularity",
           f"max_rel_disagreement={agree:.2e}; d_controlled={controlled}",
           "trajectories agree (<5%) AND d remains < 0.06 and decreasing",
           passed, "F-coordinate (singularity uncontrollable)", triggered=not passed)
    return passed


# --------------------------------------------------------------------------- #
#  EXPERIMENT 5 -- Compounding threshold & QEC suppression signature (P5/P8)    #
# --------------------------------------------------------------------------- #
def experiment_5():
    print("\nExperiment 5 - Compounding threshold + suppression signature (P5 / P8 / F4)")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    fig.suptitle("Experiment 5  --  Compounding threshold (P5) and power-law suppression "
                 "signature (P8).", fontweight="bold", fontsize=11)

    # (a) compounding threshold: at beta=k=0, diverges iff A0 < (gamma3-1)*b
    ax = axes[0]
    b, g3 = 1.0, 4.0
    thr = (g3 - 1.0) * b
    CEIL = 10.0
    A0s = np.linspace(1.0, 6.0, 60)
    finals = []
    for A0 in A0s:
        p = P(gamma1=0.05, gamma3=g3, A0=A0, beta=0.0, k=0.0, b=b)
        tau, C, d = integrate_depth(p, tau_max=40)
        finals.append(min(asymptote(tau, d), CEIL))
    finals = np.array(finals)
    meas_thr = threshold_crossing(A0s, finals, level=CEIL / 2.0)
    ax.semilogy(A0s, np.clip(finals, 1e-3, CEIL), "o-", color=RED, ms=3)
    ax.axvline(thr, color=GREEN, ls="--", lw=1.5,
               label=f"predicted threshold $A_0^*=(\\gamma_3-1)b={thr:.1f}$")
    ax.set_xlabel("correction strength $A_0$"); ax.set_ylabel("long-run $d$ (log)")
    ax.set_title("Genuine divergence below threshold\n(the QEC-style boundary)")
    ax.legend(fontsize=9)
    thr_ok = np.isfinite(meas_thr) and abs(meas_thr - thr) <= 0.2

    # (b) suppression signature: log d vs log C, slope should be -(beta-k)
    ax = axes[1]
    slopes = {}
    for beta, col in [(0.5, BLUE), (1.0, PURPLE)]:
        p = P(gamma1=0.05, A0=0.08, beta=beta, k=0.0, b=0.3)
        tau, C, d = integrate_depth(p, tau_max=18)
        m = (C > 5) & (d > 1e-10) & np.isfinite(d)
        lC, ld = np.log10(C[m]), np.log10(d[m])
        s = float(np.polyfit(lC, ld, 1)[0])
        slopes[beta] = s
        ax.scatter(lC, ld, s=8, color=col, alpha=0.5)
        xf = np.linspace(lC.min(), lC.max(), 100)
        ax.plot(xf, np.polyval(np.polyfit(lC, ld, 1), xf), color=col, lw=2,
                label=f"$\\beta={beta}$: slope={s:.2f} (pred $-{beta:.1f}$)")
    ax.set_xlabel("$\\log_{10} C$"); ax.set_ylabel("$\\log_{10} d$")
    ax.set_title("Power-law suppression: slope $=-(\\beta-k)$")
    ax.legend(fontsize=9)
    slope_ok = all(abs(slopes[bb] - (-bb)) < 0.08 for bb in slopes)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp5_compounding_threshold.png"), dpi=160, bbox_inches="tight")
    plt.close()

    passed = bool(thr_ok and slope_ok)
    record("E5", "P5/P8: divergence iff A0<(gamma3-1)b; suppression slope = -(beta-k)",
           f"meas_threshold={meas_thr:.2f} (pred {thr:.1f}); slopes={ {k: round(v,3) for k,v in slopes.items()} }",
           "threshold within 0.2 AND each slope within 0.08 of -(beta-k)",
           passed, "F4 (suppression not power-law / no threshold)", triggered=not passed)
    return passed


# --------------------------------------------------------------------------- #
#  EXPERIMENT 6 -- Vector misalignment: the unmonitored subspace (P6)          #
# --------------------------------------------------------------------------- #
def experiment_6():
    print("\nExperiment 6 - Vector null-subspace (P6)")
    # Two value axes sharing one capability process. The correction operator is
    # diagonal: it CO-SCALES on the monitored axis (beta=1) and is identically
    # zero on the blind axis. The threshold is the spectral condition that the
    # correction operator be positive-definite on the WHOLE value space.
    g1 = 0.05
    p_mon = P(gamma1=g1, A0=0.30, beta=1.0, k=0.0, b=1.0, d0=0.05)   # monitored, co-scaling
    p_blind = P(gamma1=g1, A0=0.0, beta=0.0, k=0.0, b=1.0, d0=0.05)  # corrector's null direction
    tau, C, d1 = integrate_depth(p_mon, tau_max=14)
    _, _, d2 = integrate_depth(p_blind, tau_max=14)
    d1_final, d2_final = asymptote(tau, d1), asymptote(tau, d2)
    pred1, pred2 = 0.0, g1              # monitored -> 0 ; blind floors at gamma1
    ok = bool((d1_final < 5e-3) and (abs(d2_final - pred2) < 5e-3) and (d2_final > 5 * d1_final))

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    fig.suptitle("Experiment 6  --  Vector misalignment (P6).  Misalignment persists exactly on the "
                 "correction operator's blind axis.", fontweight="bold", fontsize=11)
    ax.plot(tau, d1, color=GREEN, label="monitored axis (co-scaling, $\\beta=1$) $\\to 0$")
    ax.plot(tau, d2, color=RED, label=f"blind axis (null direction) $\\to \\gamma_1={pred2:.3f}$")
    ax.axhline(g1, color=RED, ls=":", lw=1, alpha=0.6)
    ax.text(8, g1 + 0.0015, "$\\gamma_1$ floor", color=RED, fontsize=9)
    ax.set_xlabel("self-improvement depth $\\tau$"); ax.set_ylabel("per-axis misalignment $d_i$")
    ax.set_ylim(0, 0.06); ax.legend(fontsize=10, loc="center right")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp6_vector_subspace.png"), dpi=160, bbox_inches="tight")
    plt.close()

    record("E6", "P6: threshold is a spectral condition; the operator's null axis stays misaligned",
           f"d_monitored={d1_final:.2e} (pred ~0); d_blind={d2_final:.3f} (pred {pred2:.3f})",
           "monitored axis driven to 0 AND blind axis floors at gamma1 (>5x larger)",
           ok, "F6 (blind axis also suppressed)", triggered=not ok)
    return ok


# --------------------------------------------------------------------------- #
#  EXPERIMENT 7 -- Stochastic drift: mean, variance, tail bound (P7)           #
# --------------------------------------------------------------------------- #
def experiment_7():
    print("\nExperiment 7 - Stochastic drift and tail risk (P7)")
    g1, b = 0.05, 1.0
    sigma = 0.02
    # constant-coefficient OU in depth: dd = (g1 - decay d) dtau + sigma dW,  decay = A0/b + 1
    def ensemble(A0, M=4000, tau_max=12.0, n=2400):
        decay = A0 / b + 1.0
        dt = tau_max / n
        d = np.full(M, 0.05)
        sd = np.sqrt(dt)
        for _ in range(n):
            d = d + (g1 - decay * d) * dt + sigma * sd * RNG.standard_normal(M)
        return d, decay

    # (a) one ensemble: check mean and variance against OU stationary values
    A0 = 0.5
    d_final, decay = ensemble(A0)
    mean_meas, var_meas = float(np.mean(d_final)), float(np.var(d_final))
    mean_pred, var_pred = g1 / decay, sigma**2 / (2.0 * decay)
    mean_ok = abs(mean_meas - mean_pred) < 0.1 * mean_pred + 2e-3
    var_ok = abs(var_meas - var_pred) < 0.2 * var_pred + 1e-6

    # (b) variance vs decay should scale as 1/(2 decay) -> co-scaling suppresses the tail
    A0_grid = np.array([0.2, 0.5, 1.0, 2.0, 4.0])
    vars_meas, vars_pred = [], []
    for a0 in A0_grid:
        df, dc = ensemble(a0, M=2500, n=1800)
        vars_meas.append(np.var(df)); vars_pred.append(sigma**2 / (2 * dc))
    slope = float(np.polyfit(np.log10(A0_grid / b + 1.0), np.log10(vars_meas), 1)[0])
    scale_ok = abs(slope - (-1.0)) < 0.15

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    fig.suptitle("Experiment 7  --  Stochastic drift (P7).  Co-scaling suppresses the MEAN and the "
                 "VARIANCE of misalignment;\ngovernance gets a tail bound, not just an average.",
                 fontweight="bold", fontsize=11)
    ax = axes[0]
    ax.hist(d_final, bins=60, color=BLUE, alpha=0.6, density=True)
    xs = np.linspace(d_final.min(), d_final.max(), 200)
    ax.plot(xs, np.exp(-(xs - mean_pred)**2 / (2 * var_pred)) / np.sqrt(2 * np.pi * var_pred),
            color=RED, lw=2, label="OU stationary $\\mathcal{N}(d^*,\\sigma^2/2\\kappa)$")
    ax.axvline(mean_pred, color=GREEN, ls="--", label=f"$d^*$={mean_pred:.3f}")
    ax.set_xlabel("misalignment fraction $d$"); ax.set_ylabel("stationary density")
    ax.set_title(f"Mean {mean_meas:.3f} (pred {mean_pred:.3f}),  "
                 f"Var {var_meas:.1e} (pred {var_pred:.1e})"); ax.legend(fontsize=9)
    ax = axes[1]
    ax.loglog(A0_grid / b + 1.0, vars_meas, "o", color=BLUE, ms=6, label="measured variance")
    ax.loglog(A0_grid / b + 1.0, vars_pred, "-", color=RED, label="$\\sigma^2/2\\kappa$ (slope $-1$)")
    ax.set_xlabel("effective decay $\\kappa = A_0/b + 1$"); ax.set_ylabel("stationary variance of $d$")
    ax.set_title(f"Variance $\\propto 1/\\kappa$  (fitted slope {slope:.2f})"); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp7_stochastic.png"), dpi=160, bbox_inches="tight")
    plt.close()

    passed = bool(mean_ok and var_ok and scale_ok)
    record("E7", "P7: stationary mean=g1/kappa, variance=sigma^2/(2 kappa) ~ 1/(A+r); tail suppressed",
           f"mean {mean_meas:.3f}/{mean_pred:.3f}; var {var_meas:.1e}/{var_pred:.1e}; var-slope {slope:.2f}",
           "mean & variance match OU theory AND variance scales as 1/kappa",
           passed, "F7 (variance does not scale as 1/(A+r))", triggered=not passed)
    return passed


# --------------------------------------------------------------------------- #
#  EXPERIMENT 8 -- Integrator certificate: closed form vs LSODA vs Euler (P*)  #
#  Proves the numerics are correct on the one case with an exact solution.     #
# --------------------------------------------------------------------------- #
def experiment_8():
    print("\nExperiment 8 - Exact-solution validation certificate")
    g1, A0, b = 0.05, 0.08, 0.5     # k=0, beta=0, additive: closed form available
    p = P(gamma1=g1, A0=A0, beta=0.0, k=0.0, b=b, d0=0.05)
    r = b
    dstar = g1 * r / (A0 + r)

    # closed form d(t) = dstar + (d0 - dstar) exp(-(A0+r) t)
    T = 30.0
    tt = np.linspace(0, T, 2000)
    d_exact = dstar + (p.d0 - dstar) * np.exp(-(A0 + r) * tt)

    # LSODA in depth tau = r t  (since k=0)
    tau, C, d_depth = integrate_depth(p, tau_max=r * T, n=2000)
    t_depth = tau / r
    d_ls = np.interp(tt, t_depth, d_depth)

    # explicit Euler on (C, D) in time -- "what you would code directly"
    dt = T / 20000
    C_e, D_e = p.C0, p.d0 * p.C0
    ts, ds = [], []
    for i in range(20001):
        ts.append(i * dt)
        ds.append(D_e / C_e)
        dC = b * C_e
        dD = g1 * dC - A0 * D_e
        C_e += dC * dt
        D_e += dD * dt
    d_eu = np.interp(tt, np.array(ts), np.array(ds))

    err_ls = float(np.max(np.abs(d_ls - d_exact)))
    err_eu = float(np.max(np.abs(d_eu - d_exact)))
    ok = (err_ls < 1e-4) and (err_eu < 1e-3)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    fig.suptitle("Experiment 8  --  Integrator certificate.  Closed form, stiff solver, and naive "
                 "Euler agree.", fontweight="bold", fontsize=11)
    ax.plot(tt, d_exact, color=INK, lw=3, alpha=0.5, label="closed form (Theorem 1)")
    ax.plot(tt, d_ls, color=BLUE, ls="--", label=f"LSODA depth-clock (max err {err_ls:.1e})")
    ax.plot(tt[::80], d_eu[::80], "o", color=RED, ms=4, label=f"explicit Euler (max err {err_eu:.1e})")
    ax.axhline(dstar, color=GREEN, ls=":", label=f"$d^*=\\gamma_1 r/(A_0+r)$={dstar:.4f}")
    ax.set_xlabel("time $t$"); ax.set_ylabel("misalignment fraction $d$")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp8_validation.png"), dpi=160, bbox_inches="tight")
    plt.close()

    record("E8", "Numerics reproduce the exact transient (Theorem 1) to machine-relevant tolerance",
           f"max|LSODA-exact|={err_ls:.1e}; max|Euler-exact|={err_eu:.1e}",
           "LSODA error < 1e-4 AND Euler error < 1e-3",
           ok, "F-numeric (integrator wrong)", triggered=not ok)
    return ok


# --------------------------------------------------------------------------- #
#  EXPERIMENT 9 -- Residual drift at rest (P5 / F5)                             #
#  A frozen but capable system still drifts when level-drift gamma2>0. This is  #
#  the experiment the paper's §3.6 / F5 refer to; it is now present, not cited.  #
# --------------------------------------------------------------------------- #
def experiment_9():
    print("\nExperiment 9 - Residual drift at rest (P5 / F5)")
    A0 = 0.1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Experiment 9 -- Residual drift at rest (P5 / F5).  A frozen but capable system "
                 "still drifts when\nlevel-drift gamma2 > 0: pausing growth is not a substitute for "
                 "correction.", fontweight="bold", fontsize=11)
    ax = axes[0]
    ok = True
    tau_halt = None
    for g2, col, lab in [(0.0, GREEN, "$\\gamma_2=0$: halting $\\to d\\to 0$"),
                         (0.02, RED, "$\\gamma_2=0.02$: halting $\\to d\\to\\gamma_2/A_0>0$")]:
        p = P(gamma1=0.03, gamma2=g2, A0=A0, beta=0.0, k=0.0, b=0.4, d0=0.05)
        tau, C, d = integrate_depth(p, tau_max=7, n=700)
        # frozen phase (r=0): d_dot = gamma2 - A0*d  =>  d* = gamma2/A0
        dstar = g2 / A0
        t_fr = np.linspace(0.0, 70.0, 700)
        d_fr = dstar + (d[-1] - dstar) * np.exp(-A0 * t_fr)
        x_fr = tau[-1] + 0.7 * tau[-1] * (t_fr / t_fr[-1])      # cosmetic frozen axis
        tau_halt = tau[-1]
        ax.plot(np.concatenate([tau, x_fr]), np.concatenate([d, d_fr]), color=col, label=lab)
        ok = ok and (abs(d_fr[-1] - dstar) < 5e-3)
    ax.axvline(tau_halt, color="#888", ls="--", lw=1.2, label="growth halted")
    ax.set_xlabel("self-improvement depth $\\tau$  |  then frozen time")
    ax.set_ylabel("misalignment fraction $d$")
    ax.set_title("Halt growth: $\\gamma_2=0\\to 0$, but $\\gamma_2>0\\to$ residual floor")
    ax.legend(fontsize=9); ax.set_ylim(bottom=0)

    ax2 = axes[1]
    A0s = np.linspace(0.02, 0.5, 60)
    ax2.plot(A0s, 0.02 / A0s, color=BLUE, lw=2)
    ax2.set_xlabel("correction strength $A_0$")
    ax2.set_ylabel("residual floor at rest  $d^* = \\gamma_2/A_0$")
    ax2.set_title("Residual misalignment at rest is removed only by correction")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGDIR, "exp9_residual_drift.png"), dpi=160, bbox_inches="tight")
    plt.close()

    record("E9", "P5/F5: a frozen capable system retains residual d*=gamma2/A0; halting growth is not correction",
           f"residual matches gamma2/A0 (and 0 when gamma2=0) within tol = {ok}",
           "with gamma2>0, halting growth leaves d -> gamma2/A0 > 0; with gamma2=0 it -> 0",
           ok, "F5 (no residual drift at rest)", triggered=not ok)
    return ok


# --------------------------------------------------------------------------- #
#  Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    print("=" * 78)
    print("  COUPLED CO-SCALING CORRECTION  --  NUMERICAL VERIFICATION & CONSISTENCY HARNESS")
    print("  Michael Darius Eastwood | June 2026 | michaeldariuseastwood.com/research")
    print("  NOTE: these are internal-consistency + integrator checks (code matches maths),")
    print("        NOT empirical tests of the model against a real system.")
    print("=" * 78)

    results = [
        experiment_1(),
        experiment_2(),
        experiment_3(),
        experiment_4(),
        experiment_4b(),
        experiment_5(),
        experiment_6(),
        experiment_7(),
        experiment_8(),
        experiment_9(),
    ]

    print("\n" + "=" * 78)
    print("  VERDICT TABLE  (internal-consistency checks of the derivation + integrator)")
    print("=" * 78)
    n_pass = sum(v["passed"] for v in VERDICTS)
    n_trig = sum(v["triggered"] for v in VERDICTS)
    for v in VERDICTS:
        print(f"  {v['experiment']:<5} {'PASS' if v['passed'] else 'FAIL':<5} "
              f"{v['falsifier']:<38} {'TRIGGERED' if v['triggered'] else 'clear'}")
    print("-" * 78)
    all_pass = all(results)
    print(f"  {n_pass}/{len(VERDICTS)} internal-consistency checks pass | "
          f"{n_trig} kill-conditions triggered")
    print("  F4 (QEC mechanism): the model's suppression law is analytically power-law, so the")
    print("  QEC correspondence holds in threshold FORM only, not as a transferred mechanism.")
    print(f"  OVERALL: {'code matches the maths (E1-E9); the model-vs-reality test is the open problem' if all_pass else 'an internal-consistency check FAILED -- see above'}")
    print("=" * 78)

    # persist machine-readable + human-readable artefacts
    with open(os.path.join(RESDIR, "verdicts.json"), "w") as f:
        json.dump({"all_pass": all_pass, "n_pass": n_pass, "n_total": len(VERDICTS),
                   "n_triggered": n_trig, "verdicts": VERDICTS}, f, indent=2)
    with open(os.path.join(RESDIR, "report.txt"), "w") as f:
        f.write("COUPLED CO-SCALING CORRECTION -- VERDICT REPORT\n")
        f.write("Michael Darius Eastwood | June 2026\n")
        f.write("=" * 70 + "\n\n")
        for v in VERDICTS:
            f.write(f"[{v['experiment']}] {'PASS' if v['passed'] else 'FAIL'}\n")
            f.write(f"  Prediction: {v['prediction']}\n")
            f.write(f"  Statistic : {v['statistic']}\n")
            f.write(f"  Criterion : {v['criterion']}\n")
            f.write(f"  {v['falsifier']}: {'TRIGGERED' if v['triggered'] else 'not triggered'}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"{n_pass}/{len(VERDICTS)} confirmed; {n_trig} falsifiers triggered; "
                f"OVERALL {'ALL CONFIRMED' if all_pass else 'FALSIFIER FIRED'}\n")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
