"""P5's observation model: the paired readings, their covariance, and the increments a log cannot hold.

WHY THIS FILE EXISTS. The bank measured a state, ran one round, measured again, and then handed two
DERIVED numbers to a regression: an available capability equal to the retention fraction times the
before reading, and an increment equal to the after reading minus the before reading. The before
reading is inside both of them. Whatever error that one reading carried entered the predictor with a
plus sign and the response with a minus sign, so the two errors were not independent, they were
correlated by construction, and the estimator was an errors-in-variables correction parameterised by
a single ratio of two error variances. A ratio cannot express a covariance. That is the first half of
finding A5.

The second half is what the code did with the increments it did not like. Before taking logarithms it
removed every row whose increment was not positive, counted how many it had removed, and declared the
bank inconclusive only if the count passed a ceiling. A row is removed exactly when the read noise
happened to run downwards further than the round grew, so the removal keeps the upward errors and
discards the downward ones, and it does so hardest wherever the increment is smallest relative to the
read error. That is a position on the very axis whose slope is being measured, and which position it
is depends on the ladder: on the simulated ladder here the read error is largest in the middle of the
range, so the removal falls hardest on the upper half. Measured on a bank whose true increment does
not depend on the available capability at all, the surviving cells carried a fitted exponent of plus
0.12 where the truth was zero. A genuine regression, a genuine zero and a genuine decrement were all
deleted by the same rule, and the rule reported them as cells the ladder could not resolve. An
inconvenient outcome had become a generic measurement failure.

WHAT IS MEASURED, AND WHAT IS DERIVED. The measurements are the two readings. Everything else is a
function of them, and a function of a measurement carries the measurement's error in whatever way the
function's derivatives say it does. So the fit is on the readings.

THE MODEL, STATED IN FULL.

For bank cell i placed at capability state s(i) with retention fraction f_i, which the experimenter
sets and therefore knows exactly:

    the artefact placed at state s has one true capability U_s
    the available capability of the round is A_i = f_i * U_s, exactly, because f_i carries no error
    the round's true increment is G_i = a * A_i ** beta * exp(u_i), with u_i normal, mean zero,
        standard deviation sigma, being the round to round process variability
    the before reading is X0_i = U_s + e0_i, with e0_i of known variance v0_i
    the after reading is X1_i = U_s + G_i + e1_i, with e1_i of known variance v1_i
    the two read errors have a declared correlation rho, which is zero for every ladder that draws a
        fresh item form or redraws a stochastic outcome on each read, and is not zero for a ladder
        that reads the SAME drawn form before and after

v0_i and v1_i are not assumed: they are what `arc_runner.sampling` computes from the ladder's declared
sampling unit, and for a deterministic whole-pool read they are exactly zero.

THE LIKELIHOOD. Write D_i = X1_i - X0_i for the measured increment, which is G_i + e1_i - e0_i. The
pair (X0_i, D_i) is what was measured, and the joint density factorises into the before reading and
the increment given it:

    X0_i ~ Normal(U_s, v0_i)
    D_i | X0_i ~ Normal( m_i * exp(sigma ** 2 / 2) + (k_i - 1) * (X0_i - U_s),
                         m_i ** 2 * tau ** 2 + v1_i * (1 - rho ** 2) )

with m_i = a * A_i ** beta, k_i = rho * sqrt(v1_i / v0_i), and tau ** 2 = (exp(sigma ** 2) - 1) *
exp(sigma ** 2), the relative variance of a lognormal process error.

The conditional mean is the whole of finding A5's first half, written down. Conditioning on the before
reading makes its error e0_i = X0_i - U_s KNOWN, and the increment contains that error with a minus
sign, so the expected increment is the model's increment minus the before error (plus whatever part of
the after error the declared correlation makes predictable from it). That is the covariance, handled
exactly, and it is a term in a mean rather than a ratio in a correction. When rho is zero the
coefficient is exactly minus one, which says that a before reading a hundred items high produces an
increment a hundred items low, which is what subtracting one number from another does.

The conditional variance is the second consequence. Conditioning removes e0 entirely, and removes the
part of e1 that rho makes predictable from it, so what is left is the process variability, which
scales with the size of the increment, and the after read error. Setting rho to zero leaves the whole
of v1 in the variance and is therefore the WIDER reading, which is why zero is the fail closed default
here: a run that has not settled whether its two reads share their items gets the more cautious
weights rather than the more flattering ones.

WHY THIS ADMITS A ZERO AND A NEGATIVE INCREMENT. The likelihood is written for D_i, not for log D_i. A
measured increment of zero, or of minus forty items, is an ordinary draw from a normal distribution
whose mean happens to be small, and it is kept, weighted and fitted like every other row. Nothing is
removed, so nothing can be removed selectively. The registered growth law is still a power law and the
process error is still multiplicative: what has gone is the log transform of the RESPONSE, which was
never part of the registered model and was only ever a way of fitting it.

WHAT THE LATENT STATE COSTS AND WHY IT IS SHARED. U_s is one parameter per capability state, not one
per row, because `place_at_state` places the same artefact for every replicate at a state: the
simulated loader sets the same latent capability and the code domain's loader reads the same saved
checkpoint out of the store. So the replicates at a state are repeated readings of one artefact, they
estimate one U_s between them, and the estimate of a state is as precise as its replicate count makes
it. The assumption is checked rather than assumed: `state_homogeneity` compares the observed scatter
of the before readings within each state against the scatter their own read variances predict, and an
over-dispersed bank is reported as such. Where the before readings carry no error at all the latent
state is not a free parameter: the reading IS the state, and the fit says so by not estimating one.

WHAT THIS FILE DOES NOT CLAIM. The estimator is a maximum likelihood fit of the model above, so it is
only as good as the model. Three things are therefore reported beside every estimate and never folded
into it. Whether the rate is distinguishable from zero at all, because when it is not there is no
elasticity to report and a fitted exponent is a description of noise: this is the `zero growth` case,
and a runner that reports a slope there has invented one. Whether the fitted rate is negative, because
the registered process is a growth process and a systematic decrement is outside it: that is reported
as an inadequacy of the model rather than converted into an exponent. And the dispersion of the
standardised residuals, which is one when the variance model is right and is reported when it is not.

AND NOTHING IS DROPPED ONE LEVEL UP EITHER. Removing the row filter is only half of what the second
half of the finding asks for. The routes were built by fitting each registered level separately and
then averaging the levels that produced a usable exponent, and a level produces none exactly when its
own increment is not distinguishable from zero, which is the same selection on the same axis moved
from rows to registered levels. Measured on a bank at the registered replicate count whose round
genuinely could not use the smallest retained fraction, that average reported 0.5274 from four of the
five registered levels while the precision condition passed and nothing in the record said a level
was missing. The registration forbids it twice over: "Every run at a registered seed and retention
level is included. No run is excluded on its outcome", and "No retention level is added, dropped or
reweighted after any increment is seen". So a route is now ONE fit over every cell of the bank with
one common exponent and one rate per registered level, which is `fit_paired(..., group_by=...)`; the
level whose round produced nothing carries its own rate near zero and keeps its cells and its place
in the likelihood. That is also the structure the registration itself names for the panel, being a
common slope with a per-set intercept absorbing that set's own rate. `combine`, which is what the
routes used to be built with, now refuses to average a set of fits when any member of the set carries
no usable exponent, rather than quietly averaging the survivors.

THE REGISTERED ESTIMATOR IS COMPUTED IN EVERY RUN AND ITS DEMOTION IS DECLARED, NOT ASSUMED. The
registration names its primary: the attenuation-corrected slope of the log increment on the log
available capability, with logarithms on both sides. The paired fit above is a different estimator of
the same estimand, and it is what finding A5 asks for, because the registered one cannot be computed
without excluding rows on their outcome, which the same registration forbids. Those two registered
sentences cannot both be honoured on a bank with a nonpositive increment, and this module does not
settle which gives way. `registered_log_scale_estimate` computes the registered quantity, says
whether computing it required excluding cells and how many, quotes the sentences at issue, and marks
the substitution as an amendment that has NOT been ratified and that belongs to the author. The
unit's registration has ruled on a case of this exact shape once already, for the linearity statistic
it registers as secondary and calls the better one: a unit does not quietly substitute a better
statistic for the one the theory registration names.

THE LOG SCALE ESTIMATOR IS KEPT AS A DIAGNOSTIC AND LABELLED. `log_scale_corrected_slope` is the other
estimator the finding names: a moment corrected regression on the log scale using the known read
variances per row, with the derivative factors of the transform and the retention fraction handled
explicitly. It is the right closed form for its own model and it is reported beside the primary fit so
that the size of the repair can be seen. It is NOT the primary, for two reasons that are measured
rather than argued. It still cannot take the logarithm of a number that is not positive, so it still
has to drop rows, and it says how many it dropped. And the transform's second order term, being the
read variance over twice the squared increment, is larger where the increment is smaller, so it tilts
the fitted line by an amount that no first order covariance correction removes. Measured over two
hundred simulated banks of the runner's own shape, with a true exponent of one half: the paired fit
returned 0.5007, the covariance corrected log scale slope returned 0.5123, and the same slope with
the covariance term omitted returned 0.5113. The covariance term is real and is exact for its own
model, which tests/test_p5_observation.py checks as arithmetic rather than blessing by simulation; on
this bank it is worth a thousandth and the transform's own tilt is worth twelve, and this file does
not pretend otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

# The smallest relative process standard deviation the fit will report. A bank whose residuals are
# exactly reproduced by the read variances alone drives sigma to zero and the log likelihood to minus
# infinity, so the parameter is bounded away from it and the bound is reported by `at_floor`. This is
# a numerical floor and never a claim that the process varies.
MIN_PROCESS_SD = 1e-4
MAX_PROCESS_SD = 5.0

ADEQUATE = "ADEQUATE"
NO_MEASURABLE_GROWTH = "NO MEASURABLE GROWTH"
NEGATIVE_RATE = "NEGATIVE MEAN INCREMENT"
DID_NOT_CONVERGE = "DID NOT CONVERGE"
TOO_FEW_ROWS = "TOO FEW ROWS"
OVERDISPERSED = "OVERDISPERSED RESIDUALS"


@dataclass(frozen=True)
class ObservationModel:
    """What a run declares about how its two readings were taken, once, in one place.

    `read_correlation` is the correlation between the before and the after read errors. It is zero
    whenever the two reads draw independently, which is every ladder in this package: a fresh item
    form each read, or a fresh realisation of a stochastic scoring process, or an exact deterministic
    read with no error to correlate. It is not zero for a ladder that draws ONE item form and reads
    the artefact on it before and after, because then the difficulty of the drawn items is common to
    both readings. Zero is the fail closed default because it leaves the whole after read variance in
    the conditional variance and therefore gives the wider interval.

    `state_is_shared` records the design fact the latent state rests on: that every replicate at a
    capability state is a reading of the same placed artefact. It is true for both loaders this
    package ships and is checked against the data by `state_homogeneity`.
    """

    read_correlation: float = 0.0
    state_is_shared: bool = True
    min_process_sd: float = MIN_PROCESS_SD

    def as_record(self) -> Dict[str, Any]:
        return {"read_correlation": float(self.read_correlation),
                "state_is_shared": bool(self.state_is_shared),
                "min_process_sd": float(self.min_process_sd),
                "response_scale": "the measured increment on its own scale; no logarithm is taken of "
                                  "the response, so a zero and a negative increment are observations",
                "process_error": "multiplicative lognormal on the increment",
                "predictor": "available capability, being the exact retention fraction times the "
                             "latent capability state, with the before reading's error carried into "
                             "the increment's conditional mean rather than into an error ratio"}


def read_correlation_of(ladder: Any) -> float:
    """The ladder's declared read error correlation, or the fail closed zero.

    A ladder that has not settled the question gets the value that widens its intervals rather than
    the value that narrows them, which is the same discipline `arc_runner.sampling` applies to the
    sampling unit.
    """
    try:
        rho = float(getattr(ladder, "read_error_correlation", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return float(min(max(rho, -0.999), 0.999))


def model_for(ladder: Any) -> ObservationModel:
    """The observation model a ladder implies. Used by `run_bank` so the declaration travels with the
    rows rather than being supplied later by whoever happens to analyse them."""
    return ObservationModel(read_correlation=read_correlation_of(ladder))


# --------------------------------------------------------------------------------------------------
# The paired readings, packed once
# --------------------------------------------------------------------------------------------------


def _row_pair(r: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """The two readings and their variances, from a bank row.

    Rows written before this module existed carry only `available`, `increment` and a combined
    `read_sd`. They are reconstructed rather than refused, because a bundle saved by an earlier run
    must still re-score, which is finding A8's requirement and not something this repair may quietly
    withdraw: the before reading is the available capability over the retention fraction, and the
    after reading is that plus the increment.

    WHAT THE SPLIT OF THE COMBINED READ ERROR IS, STATED EXACTLY AND NOT FLATTERED. `read_sd` was
    written as the hypotenuse of the two read standard deviations, so it records their SUM OF SQUARES
    and nothing else. The two sides are NOT recoverable from it: a cell whose two readings sat at
    different pass rates carries different variances on the two sides and the same hypotenuse. The
    reconstruction therefore splits the total variance equally, which reproduces the sum exactly and
    is wrong about each side by however much the two differed, and being wrong about each side moves
    the conditional variance and the latent state's weights. That cost is counted rather than hidden:
    `pack` reports how many rows it reconstructed and `PairedFit.rows_reconstructed` carries the
    count into every record, so a re-scored legacy bundle is readable as one whose per-side variances
    rest on an equal split. A bank recorded since this change carries `var_before` and `var_after`
    and never comes down this path.
    """
    if "before" in r and "after" in r:
        b, a = float(r["before"]), float(r["after"])
        v0 = float(r.get("var_before", 0.0)); v1 = float(r.get("var_after", 0.0))
        return b, a, v0, v1
    f = float(r.get("fraction", 1.0)) or 1.0
    b = float(r["available"]) / f
    a = b + float(r["increment"])
    half = float(r.get("read_sd", 0.0)) ** 2 / 2.0
    return b, a, half, half


def pack(rows: Sequence[Dict[str, Any]], groups: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """The arrays the likelihood needs, and the reference scale that keeps it conditioned.

    `groups` labels each row with the SET whose own rate it shares. The default is one rate for every
    row, which is the pooled and the crossed fit. A ROUTE uses one rate per registered level instead,
    and that is the whole of the repair to finding A5's second refutation: a route used to be the
    MEAN of separate fits to its levels, and a level whose round produced nothing returned no usable
    exponent and was left out of that mean, so the route was reported from the levels that happened
    to grow. That is the finding's own mechanism moved from rows up to registered levels, and the
    registration forbids it in two places at once: "Every run at a registered seed and retention
    level is included. No run is excluded on its outcome", and "No retention level is added, dropped
    or reweighted after any increment is seen". One fit with one exponent and one rate per level
    keeps every level and every cell, and it is also the structure the registration names for the
    panel, being a common slope with a per-set intercept absorbing that set's own rate.
    """
    rows = list(rows)
    states = sorted({r["state"] for r in rows})
    index = {s: i for i, s in enumerate(states)}
    si = np.array([index[r["state"]] for r in rows], int)
    f = np.array([float(r["fraction"]) for r in rows], float)
    if groups is None:
        group_labels: list = [None]
        gi = np.zeros(len(rows), int)
    else:
        seen = list(dict.fromkeys(list(groups)))
        try:
            group_labels = sorted(seen)
        except TypeError:                       # labels of mixed type keep the order they arrived in
            group_labels = seen
        gpos = {g: i for i, g in enumerate(group_labels)}
        gi = np.array([gpos[g] for g in groups], int)
    # A row written before the two readings were recorded is reconstructed by `_row_pair` under an
    # equal split of the combined read variance, which is exact in the sum and wrong on each side.
    # The count travels with the fit so that a re-scored legacy bundle says so of itself.
    reconstructed = sum(1 for r in rows if not ("before" in r and "after" in r))
    pairs = [_row_pair(r) for r in rows]
    X0 = np.array([p[0] for p in pairs], float)
    X1 = np.array([p[1] for p in pairs], float)
    v0 = np.array([max(p[2], 0.0) for p in pairs], float)
    v1 = np.array([max(p[3], 0.0) for p in pairs], float)
    D = X1 - X0
    design = np.zeros((len(rows), len(states)))
    design[np.arange(len(rows)), si] = 1.0
    Ubar = np.array([float(np.mean(X0[si == i])) for i in range(len(states))], float)
    # The reference available capability keeps the scale parameter near the size of a typical
    # increment rather than at whatever the units happen to make it, so the two structural parameters
    # are not separated by six orders of magnitude before the optimiser has started.
    Aref = float(np.exp(np.mean(np.log(np.maximum(f * X0, 1e-12)))))
    # The retention fraction on its own log scale, centred, for the crossed model. It is centred on
    # its own geometric mean for the same reason the available capability is: the crossed parameter
    # and the scale parameter are otherwise separated by whatever the units of f happen to be, and a
    # crossed fit started from an uncentred offset spends its first hundred iterations moving the
    # scale rather than the exponent.
    fref = float(np.exp(np.mean(np.log(np.maximum(f, 1e-12)))))
    logf = np.log(np.maximum(f, 1e-300) / fref)
    return {"states": states, "si": si, "f": f, "X0": X0, "X1": X1, "D": D, "v0": v0, "v1": v1,
            "design": design, "Ubar": Ubar, "Aref": Aref, "fref": fref, "logf": logf, "n": len(rows),
            "gi": gi, "G": len(group_labels), "group_labels": group_labels,
            "reconstructed": int(reconstructed),
            "latent": bool(np.any(v0 > 0)), "n_fractions": len({float(v) for v in f}),
            "floor": (1e-9 * max(float(np.max(np.abs(D))) if len(rows) else 1.0, 1e-9)) ** 2 + 1e-300}


def _parts(x: np.ndarray, P: Dict[str, Any], rho: float) -> Dict[str, Any]:
    """The pieces of the likelihood at a parameter vector.

    THE PARAMETER VECTOR IS [beta, ONE RATE PER GROUP, log sigma, kappa] FOLLOWED BY THE LATENT
    STATES, and kappa is the crossed model's excess retention exponent (finding A6). The mean
    increment is

        m_i = c_g(i) * (A_i / Aref) ** beta * (f_i / fref) ** kappa

    with A_i = f_i * U_s. With one group, which is the default and is what the pooled fit and the
    crossed fit use, the vector is [beta, c, log sigma, kappa, ...] exactly as it was, so nothing
    about those fits has moved. A ROUTE carries one rate per registered level and one common
    exponent, which is how a level whose round produced nothing keeps its cells and its place in the
    fit instead of being left out of a mean over the levels that grew: see `pack`.

    Writing the mean that way rather than as f ** theta * U ** beta keeps `beta` the CAPABILITY
    elasticity in both models, so the single-exponent fit is exactly this fit with kappa held at
    zero, and the retention elasticity is theta = beta + kappa. The parameter is present in the
    vector even when it is held at zero, so that one likelihood, one gradient and one sandwich serve
    both models; `_minimise` kills its search direction when it is held.
    """
    G = int(P["G"])
    beta = float(x[0])
    cvec = np.asarray(x[1:1 + G], float)
    ls, kappa = float(x[1 + G]), float(x[2 + G])
    Uv = np.asarray(x[3 + G:], float) if P["latent"] else P["Ubar"]
    U = Uv[P["si"]]
    c = cvec[P["gi"]]
    A = P["f"] * U
    # The power is taken through the exponential with the exponent clipped, so that an optimiser
    # stepping into an absurd exponent on a bank whose capabilities span four decades returns a large
    # finite objective instead of an overflow and a not-a-number. The clip is far outside any
    # exponent the registration admits and is never active at a solution.
    logA = np.log(np.maximum(A, 1e-300) / P["Aref"])
    base = np.exp(np.clip(beta * logA + kappa * P["logf"], -600.0, 600.0))
    m = c * base
    # The bound is applied to the logarithm before the exponential, not after it: a line search that
    # steps to a log standard deviation of nine hundred overflows the exponential on its way to being
    # clipped, and an overflow inside an objective is a warning the reader has to learn to ignore.
    s = float(np.exp(min(max(ls, np.log(MIN_PROCESS_SD)), np.log(MAX_PROCESS_SD))))
    s2 = s * s
    es = np.exp(s2)
    tau2 = (es - 1.0) * es
    g = np.exp(s2 / 2.0)
    v0, v1 = P["v0"], P["v1"]
    pos = v0 > 0
    safe0 = np.where(pos, v0, 1.0)
    e0 = np.where(pos, P["X0"] - U, 0.0)
    k = np.where(pos, rho * np.sqrt(v1 / safe0), 0.0)
    W = np.maximum(m * m * tau2 + v1 * (1.0 - rho * rho), P["floor"])
    r = P["D"] - m * g - (k - 1.0) * e0
    return {"beta": beta, "c": c, "cvec": cvec, "kappa": kappa, "U": U, "A": A, "base": base, "m": m, "s2": s2,
            "es": es, "tau2": tau2, "g": g, "W": W, "r": r, "e0": e0, "k": k, "pos": pos,
            "safe0": safe0, "logA": logA}


def nll_rows(x: np.ndarray, P: Dict[str, Any], rho: float = 0.0) -> np.ndarray:
    """Minus the log likelihood, one entry per row, which the sandwich standard error needs."""
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or (P["latent"] and np.any(x[3 + int(P["G"]):] <= 0)):
        return np.full(P["n"], 1e12 / max(P["n"], 1))
    q = _parts(x, P, rho)
    out = 0.5 * (np.log(q["W"]) + q["r"] ** 2 / q["W"])
    out = out + np.where(q["pos"], 0.5 * (np.log(q["safe0"]) + q["e0"] ** 2 / q["safe0"]), 0.0)
    return np.where(np.isfinite(out), out, 1e12 / max(P["n"], 1))


def nll(x: np.ndarray, P: Dict[str, Any], rho: float = 0.0) -> float:
    return float(np.sum(nll_rows(x, P, rho)))


def grad(x: np.ndarray, P: Dict[str, Any], rho: float = 0.0) -> np.ndarray:
    """The analytic gradient. It is here because the numerical alternative made the fit forty times
    slower than the bootstrap it replaces, and because a gradient checked against finite differences
    is a check on the likelihood itself."""
    x = np.asarray(x, float)
    if not np.isfinite(x).all() or (P["latent"] and np.any(x[3 + int(P["G"]):] <= 0)):
        return np.zeros_like(x)
    q = _parts(x, P, rho)
    m, W, r, tau2, g = q["m"], q["W"], q["r"], q["tau2"], q["g"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dL_dm = m * tau2 / W - r * g / W - r * r * m * tau2 / (W * W)
    dL_dm = np.where(np.isfinite(dL_dm), dL_dm, 0.0)
    logA = q["logA"]
    G = int(P["G"])
    d_beta = float(np.sum(dL_dm * m * logA))
    # One derivative per rate, being the sum over the rows of that rate's own set. With one group
    # this is the scalar it was.
    d_c = np.bincount(P["gi"], weights=dL_dm * q["base"], minlength=G).astype(float)
    dtau2 = q["es"] * (2.0 * q["es"] - 1.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        per_ls = 0.5 * m * m * dtau2 / W - 0.5 * r * m * g / W - 0.5 * r * r * m * m * dtau2 / (W * W)
    d_ls = float(2.0 * q["s2"] * np.sum(np.where(np.isfinite(per_ls), per_ls, 0.0)))
    # The crossed parameter enters the mean exactly as the exponent does, through the log of the
    # retention fraction, which carries no error because the experimenter sets it.
    d_kappa = float(np.sum(dL_dm * m * P["logf"]))
    out = np.concatenate([[d_beta], d_c, [d_ls, d_kappa]])
    if not P["latent"]:
        return np.where(np.isfinite(out), out, 0.0)
    U = q["U"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dW_dU = 2.0 * m * tau2 * (m * q["beta"] / U)
        dr_dU = -g * m * q["beta"] / U + (q["k"] - 1.0)
        per = 0.5 * dW_dU / W + r * dr_dU / W - 0.5 * r * r * dW_dU / (W * W)
    per = np.where(np.isfinite(per), per, 0.0)
    per = per + np.where(q["pos"], -q["e0"] / q["safe0"], 0.0)
    full = np.concatenate([out, P["design"].T @ per])
    return np.where(np.isfinite(full), full, 0.0)


def _start(P: Dict[str, Any], beta0: float) -> np.ndarray:
    G = int(P["G"])
    base = np.exp(np.clip(beta0 * np.log(np.maximum(P["f"] * P["Ubar"][P["si"]], 1e-300) / P["Aref"]),
                          -600.0, 600.0))
    c0 = np.zeros(G)
    for g in range(G):
        sel = P["gi"] == g
        denom = float(np.sum(base[sel] * base[sel]))
        c0[g] = float(np.sum(P["D"][sel] * base[sel]) / denom) if denom > 1e-12 else 0.0
    scale = max(float(np.mean(np.abs(c0))), 1e-9)
    spread = float(np.std(P["D"] - c0[P["gi"]] * base)) / scale
    ls0 = np.log(min(max(spread, 1e-3), 1.0))
    # the crossed excess starts at zero, being the single-exponent model
    head = [beta0] + list(c0) + [ls0, 0.0]
    return np.array(head + list(P["Ubar"])) if P["latent"] else np.array(head)


def _diagonal_scale(x: np.ndarray, P: Dict[str, Any], rho: float) -> np.ndarray:
    """A preconditioner from the curvature at the start point.

    The parameters differ by four orders of magnitude in scale (an exponent near one half beside a
    latent capability of twelve thousand items) and by six in curvature, and a quasi-Newton method
    started from an identity Hessian on such a problem stops on its own step tolerance a long way from
    the optimum. It did: the unpreconditioned fit reported a process standard deviation of 0.21 and a
    log likelihood a hundred units worse than the optimum, and it reported it as convergence.
    """
    h = np.maximum(np.abs(x) * 1e-4, 1e-6)
    d = np.ones_like(x)
    for i in range(x.size):
        a = x.copy(); a[i] += h[i]
        b = x.copy(); b[i] -= h[i]
        curvature = (grad(a, P, rho)[i] - grad(b, P, rho)[i]) / (2.0 * h[i])
        if curvature > 1e-12 and np.isfinite(curvature):
            d[i] = 1.0 / np.sqrt(curvature)
    return d


def _minimise(x0: np.ndarray, P: Dict[str, Any], rho: float,
              held: Sequence[int] = ()) -> Tuple[np.ndarray, float, bool]:
    """Minimise, with the held coordinates pinned at their starting values.

    A held coordinate is pinned by giving its search direction a scale of zero rather than by
    building a shorter parameter vector, so that one likelihood and one gradient serve the
    single-exponent and the crossed model without either of them being written twice. The pinned
    direction contributes nothing to the objective and nothing to the projected gradient, which is
    what holding a parameter means.
    """
    d = _diagonal_scale(x0, P, rho)
    for i in held:
        d[i] = 0.0
    res = minimize(lambda z: nll(x0 + d * z, P, rho), np.zeros_like(x0),
                   jac=lambda z: grad(x0 + d * z, P, rho) * d, method="L-BFGS-B",
                   options={"maxiter": 4000, "maxfun": 8000, "ftol": 1e-15, "gtol": 1e-12})
    return x0 + d * res.x, float(res.fun), bool(res.success)


def _restricted_nll(P: Dict[str, Any], rho: float) -> float:
    """The best log likelihood attainable with no growth at all, in closed form.

    With the rate set to zero the increment's mean is entirely the carried before error, the process
    variability drops out of the model, and the only free parameters are the latent states, each of
    which minimises a quadratic. This is the null the growth is tested against, and it is closed form
    because a screen that costs another optimisation is a screen that gets skipped.
    """
    v0, v1, X0, D = P["v0"], P["v1"], P["X0"], P["D"]
    pos = v0 > 0
    W = np.maximum(v1 * (1.0 - rho * rho), P["floor"])
    k = np.where(pos, rho * np.sqrt(v1 / np.where(pos, v0, 1.0)), 0.0)
    total = 0.0
    for j in range(len(P["states"])):
        sel = P["si"] == j
        if not sel.any():
            continue
        if np.all(pos[sel]):
            a = (1.0 - k[sel]) ** 2 / W[sel] + 1.0 / v0[sel]
            b = (1.0 - k[sel]) * D[sel] / W[sel]
            U = float((np.sum(b) + np.sum(a * X0[sel])) / max(float(np.sum(a)), 1e-300))
        else:
            U = float(np.mean(X0[sel]))          # an exact read is the state; there is nothing to fit
        t = np.where(pos[sel], X0[sel] - U, 0.0)
        r = D[sel] + (1.0 - k[sel]) * t
        total += float(np.sum(0.5 * (np.log(W[sel]) + r * r / W[sel])))
        if pos[sel].any():
            m = pos[sel]
            total += float(np.sum(0.5 * (np.log(v0[sel][m]) + t[m] ** 2 / v0[sel][m])))
    return total


def expected_nonpositive_fraction(P: Dict[str, Any], q: Dict[str, Any], rho: float) -> float:
    """The model's own expectation of the number the registered precision condition H2 counts.

    H2 asks what fraction of the bank's cells returned an increment the ladder could not resolve, and
    it is answered by counting them. That count is kept as the gate, because it is what the
    registration names and this repair is not licensed to restate a registered threshold. What the
    count cannot do is separate a bank that is genuinely near the ladder's resolution from a bank
    whose noise happened to fall one way in twelve cells, so the model's expectation of the same
    number is reported beside it: the probability that a cell of this design returns a nonpositive
    increment, being the modelled increment over the standard deviation of the increment's read
    error, read off a normal distribution. An exact read resolves any increment that is not zero and
    says so by contributing nothing.

    AN OPEN DECISION, NAMED RATHER THAN TAKEN. Whether H2's ceiling should be read against the
    realised count or against this expectation is the author's to settle. The realised count is
    implemented, because it is the registered quantity and because it is the reading that does not
    move any existing operating point; the expectation sits beside it in every record so that the
    difference can be seen before anybody decides.
    """
    from math import erf, sqrt
    if not P["n"]:
        return float("nan")
    sd = np.sqrt(np.maximum(P["v0"] + P["v1"] - 2.0 * rho * np.sqrt(P["v0"] * P["v1"]), 0.0))
    m = np.asarray(q["m"], float)
    out = np.empty_like(m)
    for i in range(m.size):
        if sd[i] <= 0:
            out[i] = 1.0 if m[i] <= 0 else 0.0
        else:
            out[i] = 0.5 * (1.0 + erf(-m[i] / (sd[i] * sqrt(2.0))))
    return float(np.mean(out))


def state_homogeneity(P: Dict[str, Any]) -> Dict[str, Any]:
    """Are the replicates at a state readings of ONE artefact, as the design says they are?

    The latent state is one parameter per state because `place_at_state` places the same artefact for
    every replicate. If a run's loader does not, the before readings within a state will scatter by
    more than their own read variances allow, and the fit's precision will be overstated. The check is
    the ratio of the observed within-state variance to the mean read variance, which is one when the
    design holds. It is reported and never silently acted on: a run whose loader is not doing what the
    registration says it does is a fact about that run, not a parameter to be adjusted away.
    """
    ratios, weights = [], []
    for j in range(len(P["states"])):
        sel = (P["si"] == j) & (P["v0"] > 0)
        if int(sel.sum()) < 3:
            continue
        expected = float(np.mean(P["v0"][sel]))
        if expected <= 0:
            continue
        ratios.append(float(np.var(P["X0"][sel], ddof=1)) / expected)
        weights.append(int(sel.sum()))
    if not ratios:
        return {"dispersion": float("nan"), "states_checked": 0,
                "note": "the before readings are exact, so there is no within-state scatter to check"}
    disp = float(np.average(ratios, weights=weights))
    return {"dispersion": disp, "states_checked": len(ratios), "per_state": ratios,
            "note": "the within-state scatter of the before readings over the scatter their own read "
                    "variances predict; one means the replicates at a state are readings of one "
                    "artefact, which is what the placement loader is registered to do"}


# --------------------------------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------------------------------


@dataclass
class PairedFit:
    """One route's coupling, fitted to the readings rather than to two numbers derived from them."""

    beta: float
    beta_se: float
    rate: float
    rate_se: float
    process_sd: float
    process_sd_at_floor: bool
    n_rows: int
    n_states: int
    n_nonpositive: int
    expected_nonpositive_fraction: float
    latent_states: bool
    converged: bool
    nll: float
    deviance_against_no_growth: float
    growth_detected: bool
    dispersion: float
    homogeneity: Dict[str, Any]
    adequacy: str
    adequacy_reason: str
    model: ObservationModel
    # THE CROSSED MODEL (finding A6). `crossed` says whether the retention exponent was free. When it
    # was, `retention_excess` is kappa = theta - beta, being how much the elasticity of the increment
    # in the RETENTION fraction exceeds its elasticity in the CAPABILITY state, and it is estimated
    # once with its own standard error rather than differenced between two fits that share every
    # cell. When it was not, kappa is held at zero and these read as such.
    crossed: bool = False
    retention_excess: float = 0.0
    retention_excess_se: float = float("nan")
    # THE RATE SETS (finding A5, second refutation). A route is one fit with one exponent and one
    # rate per registered level, so a level whose round produced nothing keeps its cells and its
    # place in the fit. `group_labels` is every level the fit was given, in order; the fit was given
    # all of them or it was not run. `groups_without_measurable_growth` names the levels whose own
    # rate is not distinguishable from zero, which is the diagnostic the old code acted on and this
    # one only reports.
    grouped_by: Optional[str] = None
    group_labels: Sequence[Any] = ()
    group_rates: Sequence[float] = ()
    group_rate_ses: Sequence[float] = ()
    groups_without_measurable_growth: Sequence[Any] = ()
    # The fitted mean increment of these cells and a first-order standard error for it, holding the
    # exponent fixed. It is what the negative control is read on when its rate is not distinguishable
    # from zero: an elasticity of an increment that is not there does not exist, but the increment
    # itself is measured on the same capability scale as the bank's and can be compared with it.
    mean_increment: float = float("nan")
    mean_increment_se: float = float("nan")
    # How many of these rows were written before the two readings were recorded and were therefore
    # reconstructed under an equal split of the combined read variance. See `_row_pair`.
    rows_reconstructed: int = 0

    @property
    def retention_elasticity(self) -> float:
        """theta, the elasticity of the increment in the retention fraction. Equal to beta by
        construction in the single-exponent model, which is the assumption finding A6 names."""
        return float(self.beta + self.retention_excess)

    @property
    def usable(self) -> bool:
        """True when a fitted exponent may be read at all. A rate indistinguishable from zero has no
        elasticity to report, and a run that reports one has described its own noise."""
        return self.adequacy in (ADEQUATE, OVERDISPERSED) and np.isfinite(self.beta)

    def as_record(self) -> Dict[str, Any]:
        return {"beta": float(self.beta), "beta_se": float(self.beta_se), "rate": float(self.rate),
                "rate_se": float(self.rate_se), "process_sd": float(self.process_sd),
                "process_sd_at_floor": bool(self.process_sd_at_floor), "n_rows": int(self.n_rows),
                "n_states": int(self.n_states), "n_nonpositive": int(self.n_nonpositive),
                "expected_nonpositive_fraction": float(self.expected_nonpositive_fraction),
                "latent_states_estimated": bool(self.latent_states), "converged": bool(self.converged),
                "nll": float(self.nll),
                "deviance_against_no_growth": float(self.deviance_against_no_growth),
                "growth_detected": bool(self.growth_detected), "dispersion": float(self.dispersion),
                "state_homogeneity": self.homogeneity, "adequacy": self.adequacy,
                "adequacy_reason": self.adequacy_reason, "usable": bool(self.usable),
                "observation_model": self.model.as_record(),
                "crossed": bool(self.crossed),
                "capability_elasticity": float(self.beta),
                "retention_elasticity": float(self.retention_elasticity),
                "retention_excess": float(self.retention_excess),
                "retention_excess_se": float(self.retention_excess_se),
                "grouped_by": self.grouped_by, "n_groups": len(self.group_labels),
                "group_labels": list(self.group_labels),
                "group_rates": [float(v) for v in self.group_rates],
                "group_rate_ses": [float(v) for v in self.group_rate_ses],
                "groups_without_measurable_growth": list(self.groups_without_measurable_growth),
                "mean_increment": float(self.mean_increment),
                "mean_increment_se": float(self.mean_increment_se),
                "rows_reconstructed": int(self.rows_reconstructed),
                "levels_dropped": 0,
                "inclusion": "every cell and every rate set given to this fit is in it; no row and no "
                             "registered level is excluded on its outcome",
                "estimator": ("maximum likelihood on the paired readings: the before reading and the "
                              "increment given it, with the read covariance in the conditional mean, "
                              "the process variability in the conditional variance, and no row dropped"
                              + ("; the retention exponent is free, so the capability elasticity is "
                                 "the exponent in the state and the retention elasticity is estimated "
                                 "beside it" if self.crossed else
                                 "; one exponent on the available capability, which assumes the "
                                 "retention and capability elasticities are the same number"))}


def _nan_fit(P: Dict[str, Any], model: ObservationModel, reason: str, adequacy: str = TOO_FEW_ROWS
             ) -> PairedFit:
    n = int(P["n"]) if P else 0
    return PairedFit(beta=float("nan"), beta_se=float("nan"), rate=float("nan"),
                     rate_se=float("nan"), process_sd=float("nan"), process_sd_at_floor=False,
                     n_rows=n, n_states=len(P.get("states", ())) if P else 0,
                     n_nonpositive=int(np.sum(P["D"] <= 0)) if P else 0,
                     expected_nonpositive_fraction=float("nan"),
                     latent_states=bool(P.get("latent")) if P else False, converged=False,
                     nll=float("nan"), deviance_against_no_growth=float("nan"), growth_detected=False,
                     dispersion=float("nan"), homogeneity={}, adequacy=adequacy,
                     adequacy_reason=reason, model=model,
                     group_labels=list(P.get("group_labels", ())) if P else (),
                     rows_reconstructed=int(P.get("reconstructed", 0)) if P else 0)


def _chi2_two_df(level: float) -> float:
    """The critical value of a chi-square on two degrees of freedom, without a scipy import at call
    time: the distribution is exponential with mean two, so the quantile is closed form."""
    level = min(max(float(level), 1e-6), 1 - 1e-12)
    return float(-2.0 * np.log(1.0 - level))


def _normal_level(z: float) -> float:
    """The two sided coverage a z multiplier carries, so the deviance screen and the intervals are
    read at one registered level rather than at two conventions that happen to be nearby."""
    from math import erf, sqrt
    return float(erf(abs(float(z)) / sqrt(2.0)))


def _within_group_predictor_variation(P: Dict[str, Any]) -> float:
    """The variation in the log available capability that lies WITHIN the rate sets.

    The exponent is identified only by variation the rates cannot absorb. In a route fit each level
    carries its own rate, so variation BETWEEN levels tells the exponent nothing and the quantity
    that matters is the variation inside them. A subset that varies only between its rate sets is
    refused rather than fitted, because such a fit returns whatever the optimiser started from.
    """
    A = P["f"] * P["Ubar"][P["si"]]
    la = np.log(np.maximum(A, 1e-300))
    total = 0.0
    for g in range(int(P["G"])):
        sel = P["gi"] == g
        if int(sel.sum()) >= 2:
            total += float(np.var(la[sel])) * int(sel.sum())
    return float(total)


def fit_paired(rows: Sequence[Dict[str, Any]], model: Optional[ObservationModel] = None,
               z: float = 1.96, starts: Sequence[float] = (0.0, 0.5, 1.0),
               crossed: bool = False, group_by: Any = None) -> PairedFit:
    """Fit the observation model to a set of bank rows and report what it can and cannot support.

    `group_by` gives each registered level its own rate under one common exponent, and is how a route
    is fitted (finding A5, second refutation). Pass the row key whose values name the levels, being
    "fraction" for the route that responds to the capability state at fixed retention and "state" for
    the route that responds to retention at fixed state, or pass an explicit label per row. Every
    level given is in the fit. A route was previously the MEAN of separate fits to its levels, and a
    level whose round produced nothing returned no usable exponent and was therefore left out of that
    mean: the registration forbids exactly that, in "Every run at a registered seed and retention
    level is included. No run is excluded on its outcome" and again in "No retention level is added,
    dropped or reweighted after any increment is seen". One fit with one rate per level keeps them
    all, and it is also the structure the registration names, being a common slope with a per-set
    intercept absorbing that set's own rate.

    `crossed` frees the retention exponent (finding A6). The single-exponent model writes the mean
    increment as a power of the AVAILABLE capability, which is the retention fraction times the
    state, and that is not a neutral choice: it asserts that a tenth of the capability and a tenth of
    the retention do the same thing to the increment. In the general process the two exponents are
    separate numbers, the capability elasticity remains identified by the crossed design, and the
    excess is what the two regression directions of this bank differ by. Freeing it costs one
    parameter and requires the design to vary both factors, which the crossed bank does by
    construction and a subset of it may not.
    """
    model = model or ObservationModel()
    rows = list(rows)
    grouped_by, groups = None, None
    if group_by is not None:
        if isinstance(group_by, str):
            grouped_by, groups = group_by, [r[group_by] for r in rows]
        else:
            grouped_by, groups = "an explicit label per row", list(group_by)
    if len(rows) < 4:
        return _nan_fit(pack(rows, groups) if rows else {}, model,
                        "fewer than four cells: the observation model has three structural parameters "
                        "and one latent state per capability state")
    P = pack(rows, groups)
    G = int(P["G"])
    rho = float(model.read_correlation)
    if crossed and (P["n_fractions"] < 2 or len(P["states"]) < 2):
        # A crossed fit needs both factors to move. A subset that varies only one of them carries no
        # information about the difference between the two exponents at all, and a fit that reports
        # one from such a subset has reported its own starting value.
        return _nan_fit(P, model, "the crossed model needs the capability state and the retention "
                                  "fraction to vary; this set has %d state(s) and %d fraction(s)"
                                  % (len(P["states"]), P["n_fractions"]))
    # A SATURATED SUBSET IS REFUSED RATHER THAN FITTED. The model carries three structural parameters
    # and, where the before readings carry error, one latent state per capability state, so a subset
    # of six cells spanning three states has six parameters and no residual degrees of freedom at
    # all. Such a fit reproduces its own data exactly and returns whatever exponent the last decimal
    # of the noise asked for: on a twelve cell bank a route subset of six returned an exponent of
    # 1.55, which is outside the domain in which the growth solution is even defined. Two residual
    # degrees of freedom are required, because a variance estimated from one is not an estimate.
    n_params = 2 + G + int(bool(crossed)) + (len(P["states"]) if P["latent"] else 0)
    if P["n"] < n_params + 2:
        return _nan_fit(P, model, "%d cells cannot support %d parameters with two degrees of freedom "
                                  "left over; this subset is refused rather than fitted"
                                  % (P["n"], n_params))
    if len({(r["state"], r["fraction"]) for r in rows}) < 2 or _within_group_predictor_variation(P) <= 0:
        return _nan_fit(P, model, "the available capability does not vary within any rate set of "
                                  "these cells, so no elasticity is identified by them")
    if crossed and G > 1:
        # The crossed excess is an exponent on the retention fraction, and a rate set within which
        # the fraction does not move carries no information about it. A route fit is never crossed
        # for that reason, and asking for one is refused rather than answered from a starting value.
        within_f = sum(float(np.var(P["logf"][P["gi"] == g])) for g in range(G))
        if within_f <= 0:
            return _nan_fit(P, model, "the retention fraction does not vary within any rate set, so "
                                      "the crossed model's excess retention exponent is not "
                                      "identified by these cells")
    held = () if crossed else (2 + G,)
    best, best_f, converged = None, np.inf, False
    for b0 in starts:
        x, fv, ok = _minimise(_start(P, b0), P, rho, held)
        if fv < best_f:
            best, best_f, converged = x, fv, ok
    for _ in range(2):                              # one restart from the incumbent: the preconditioner
        x, fv, ok = _minimise(best, P, rho, held)   # is rebuilt at the better point
        if fv < best_f - 1e-9:
            best, best_f, converged = x, fv, ok
        else:
            converged = converged or ok
            break
    sigma = float(min(max(np.exp(best[1 + G]), MIN_PROCESS_SD), MAX_PROCESS_SD))
    at_floor = sigma <= MIN_PROCESS_SD * 1.5
    rate_idx = list(range(1, 1 + G))
    free = [0] + rate_idx + ([] if at_floor else [1 + G]) + ([2 + G] if crossed else []) \
        + list(range(3 + G, best.size))
    cov, idx = _sandwich_cov(best, P, rho, free)
    pos = {int(v): i for i, v in enumerate(idx)} if cov is not None else {}

    def _se(j: int) -> float:
        return (float(np.sqrt(max(float(np.diag(cov)[pos[j]]), 0.0)))
                if cov is not None and j in pos else float("nan"))

    se_beta = _se(0)
    cvec = np.asarray(best[1:1 + G], float)
    rate = float(np.mean(cvec))
    # THE RATE REPORTED FOR A GROUPED FIT IS THE MEAN OF ITS SETS' RATES, and its standard error is
    # the standard error of that mean taken from the whole rate block of the sandwich rather than
    # from the diagonal alone: the rates of two sets fitted together are not independent, because
    # they share one exponent, one process variability and the latent states, and a mean whose
    # variance ignored that would be reported more precisely than it is.
    if cov is not None and all(j in pos for j in rate_idx):
        block = cov[np.ix_([pos[j] for j in rate_idx], [pos[j] for j in rate_idx])]
        se_rate = float(np.sqrt(max(float(np.sum(block)), 0.0)) / G)
        group_ses = [float(np.sqrt(max(float(block[i, i]), 0.0))) for i in range(G)]
    else:
        block, se_rate = None, float("nan")
        group_ses = [float("nan")] * G
    kappa = float(best[2 + G]) if crossed else 0.0
    kappa_se = _se(2 + G) if crossed else float("nan")
    q = _parts(best, P, rho)
    # The fitted mean increment of these cells, on the capability scale the readings are on, with a
    # first-order standard error that holds the exponent fixed. It exists because a rate set whose
    # increment is not distinguishable from zero still HAS a measured increment with a bound on it,
    # and that bound is what the negative control is read on when no elasticity can be read at all.
    mean_increment = float(np.mean(q["m"])) if P["n"] else float("nan")
    if block is not None and P["n"]:
        w = np.array([float(np.sum(q["base"][P["gi"] == g])) / P["n"] for g in range(G)])
        mean_increment_se = float(np.sqrt(max(float(w @ block @ w), 0.0)))
    else:
        mean_increment_se = float("nan")
    dead_groups = [P["group_labels"][g] for g in range(G)
                   if not (np.isfinite(group_ses[g]) and abs(float(cvec[g])) > abs(z) * group_ses[g])]
    dispersion = float(np.mean(q["r"] ** 2 / q["W"])) if P["n"] else float("nan")
    unresolved = expected_nonpositive_fraction(P, q, rho)
    deviance = 2.0 * (_restricted_nll(P, rho) - best_f)
    level = _normal_level(z)
    # THE GROWTH SCREEN'S TWO DEGREES OF FREEDOM ARE AN APPROXIMATION, AND THEY ARE ONE IN THE
    # CROSSED FIT TOO, which is stated rather than adjusted. The null is a rate of zero, and under it
    # the exponents multiply a mean that is identically zero, so they are not identified under the
    # null and the naive count of freed parameters is not the reference distribution either way. The
    # screen that actually binds is the one below it, which asks whether the fitted rate itself is
    # distinguishable from zero at the same level, and that screen counts no degrees of freedom at
    # all. A run that wants an exact reference distribution for the deviance needs a parametric
    # bootstrap under the null, which is a different piece of work and is not pretended to here.
    detected = bool(np.isfinite(deviance) and deviance > _chi2_two_df(level))
    if np.isfinite(se_rate) and abs(rate) <= abs(z) * se_rate:
        detected = False
    homog = state_homogeneity(P)

    adequacy, reason = ADEQUATE, "the fitted rate is positive and distinguishable from zero"
    if not np.isfinite(best_f) or not np.isfinite(best[0]):
        adequacy, reason = DID_NOT_CONVERGE, "the likelihood did not reach a finite optimum"
    elif not detected:
        adequacy = NO_MEASURABLE_GROWTH
        reason = ("the increment is not distinguishable from zero on this bank (deviance %.2f against "
                  "a critical value of %.2f), so there is no elasticity of it to report; a fitted "
                  "exponent here would describe the read noise" % (deviance, _chi2_two_df(level)))
    elif rate < 0:
        adequacy = NEGATIVE_RATE
        reason = ("the fitted mean increment is negative, so the artefact lost capability on average; "
                  "the registered process is a growth process and does not admit that, and the "
                  "exponent below is the elasticity of a decrement and not of growth")
    elif np.isfinite(dispersion) and (dispersion > 2.0 or dispersion < 0.5):
        adequacy = OVERDISPERSED
        reason = ("the standardised residuals have dispersion %.2f where the variance model predicts "
                  "one, so the intervals rest on the sandwich rather than on the model" % dispersion)
    return PairedFit(beta=float(best[0]), beta_se=se_beta, rate=rate, rate_se=se_rate,
                     process_sd=sigma, process_sd_at_floor=at_floor, n_rows=int(P["n"]),
                     n_states=len(P["states"]), n_nonpositive=int(np.sum(P["D"] <= 0)),
                     expected_nonpositive_fraction=unresolved,
                     latent_states=bool(P["latent"]), converged=bool(converged), nll=float(best_f),
                     deviance_against_no_growth=float(deviance), growth_detected=detected,
                     dispersion=dispersion, homogeneity=homog, adequacy=adequacy,
                     adequacy_reason=reason, model=model, crossed=bool(crossed),
                     retention_excess=kappa, retention_excess_se=kappa_se,
                     grouped_by=grouped_by, group_labels=list(P["group_labels"]),
                     group_rates=[float(v) for v in cvec], group_rate_ses=group_ses,
                     groups_without_measurable_growth=dead_groups,
                     mean_increment=mean_increment, mean_increment_se=mean_increment_se,
                     rows_reconstructed=int(P["reconstructed"]))


def _sandwich_cov(x: np.ndarray, P: Dict[str, Any], rho: float, free: Sequence[int]
                  ) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """The standard errors, from the observed information sandwiched by the outer product of the row
    scores, over the free coordinates and returned whole.

    THE SANDWICH RATHER THAN THE PLAIN INVERSE INFORMATION, because the variance model is the part of
    this likelihood most likely to be wrong: the process error is taken as lognormal, the read
    variances are plug in estimates from the reads themselves, and one latent state is estimated for
    every capability state. Under the model the two agree and neither is wrong to use; where they
    disagree the sandwich is the one that does not assume the variance model, so it is the one
    reported. Measured over two hundred simulated banks of the runner's own shape, the sandwich
    covered a true exponent of one half in 184 of them against a nominal 190, which is the small
    optimism the latent states cost, and it is reported here rather than corrected by a factor
    nobody derived.

    THE WHOLE MATRIX RATHER THAN A PAIR OF DIAGONAL ENTRIES, because more than two are needed: the equivalence interval on the route gap
    is an interval on kappa, and a run that also wants an interval on the retention elasticity needs
    the covariance between kappa and beta as well. Returning the matrix rather than a pair of numbers
    keeps one derivative loop rather than two, which matters because the loop is the expensive part.
    A grouped fit needs the whole rate block as well, because the rate it reports is the mean over
    its sets and the sets are not independent of one another.
    """
    h = np.maximum(np.abs(x) * 1e-5, 1e-7)
    idx = np.array(list(free), int)
    k = idx.size
    if k == 0:
        return None, idx
    H = np.zeros((k, k))
    G = np.zeros((P["n"], k))
    for ii, i in enumerate(idx):
        a = x.copy(); a[i] += h[i]
        b = x.copy(); b[i] -= h[i]
        H[ii, :] = (grad(a, P, rho)[idx] - grad(b, P, rho)[idx]) / (2.0 * h[i])
        G[:, ii] = (nll_rows(a, P, rho) - nll_rows(b, P, rho)) / (2.0 * h[i])
    H = 0.5 * (H + H.T)
    try:
        Hi = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None, idx
    cov = Hi @ (G.T @ G) @ Hi
    if not np.all(np.isfinite(np.diag(cov))):
        return None, idx
    return cov, idx


def combine(fits: Sequence[PairedFit]) -> Dict[str, Any]:
    """The mean of a set of subset fits, and NOTHING when any subset of the set is missing from it.

    THIS IS NO LONGER HOW A ROUTE IS ESTIMATED, and the reason is finding A5's second refutation. The
    routes used to be built here: one fit per registered level, then the mean of the levels that
    produced a usable exponent. A level produces no usable exponent exactly when its own increment is
    not distinguishable from zero, which is the same selection the row filter made, moved up one
    level: hardest where the increment is smallest beside the read noise, and on the very axis whose
    slope is being measured. Measured on a bank at the registered replicate count whose round could
    not use the smallest retained fraction, the mean over the four levels that grew reported 0.5274
    while the registered precision condition passed and nothing said a level was missing. The routes
    are now one fit each with one rate per level (`fit_paired(..., group_by=...)`), so every level is
    in the estimate.

    What is left here is the arithmetic for a set of genuinely disjoint fits, and it now REFUSES
    rather than averages when any member carries no usable exponent. The registration is explicit
    twice over: "Every run at a registered seed and retention level is included. No run is excluded
    on its outcome", and "No retention level is added, dropped or reweighted after any increment is
    seen". A mean over the survivors is a reweighting to zero of the ones that did not survive, so
    the honest return is no number and the reason.
    """
    fits = list(fits)
    unusable = [f for f in fits if not (f.usable and np.isfinite(f.beta))]
    if not fits or unusable:
        return {"beta": float("nan"), "se": float("nan"), "n_subsets": len(fits),
                "n_usable": len(fits) - len(unusable),
                "unusable": [f.adequacy for f in unusable],
                "reason": ("no set is dropped from a mean: %d of %d carried no usable exponent (%s), "
                           "and a mean over the rest would reweight those to zero after their "
                           "increments had been seen"
                           % (len(unusable), len(fits),
                              ", ".join(sorted({f.adequacy for f in unusable})) or "none")
                           if fits else "no subsets were supplied")}
    beta = float(np.mean([f.beta for f in fits]))
    ses = [f.beta_se for f in fits if np.isfinite(f.beta_se)]
    se = float(np.sqrt(np.sum(np.square(ses))) / len(fits)) if len(ses) == len(fits) else float("nan")
    return {"beta": beta, "se": se, "n_subsets": len(fits), "n_usable": len(fits), "unusable": []}


# --------------------------------------------------------------------------------------------------
# The estimator the registration names, computed and reported beside the one in use
# --------------------------------------------------------------------------------------------------

# The registered sentences this module's primary estimator stands against, quoted so that a reader of
# a run record does not have to go and find them. They are from the P5 registration, STEP 6.
REGISTERED_ESTIMATOR_SENTENCES = (
    "The coupling is the slope of log increment on log available capability, estimated across the "
    "titration panel with a per-system intercept and a common slope, the intercept absorbing each "
    "system's own rate.",
    "The ratio of the two error variances is estimated from repeated observations on the same cells "
    "and registered before any fitting, and the attenuation-corrected estimate is the one that "
    "enters the sealed prediction and the verdict.",
    "Transformations: Logarithms on both sides of the dose-response.",
    "Data inclusion and exclusion: Every run at a registered seed and retention level is included. "
    "No run is excluded on its outcome.",
)


def registered_log_scale_estimate(rows: Sequence[Dict[str, Any]], rho: float = 0.0) -> Dict[str, Any]:
    """The registered estimator, computed, reported, and its demotion declared as an amendment.

    WHY THIS FUNCTION EXISTS. The registration names a primary: the attenuation-corrected slope of
    the log increment on the log available capability, with logarithms on both sides. This module
    fits something else, being the maximum likelihood of the paired readings, and finding A5 asks for
    that because the registered estimator cannot be computed at all without excluding rows: the
    logarithm of a nonpositive increment does not exist, and the same registration says every run is
    included and none is excluded on its outcome. The two registered sentences cannot both be
    honoured on a bank that produced a nonpositive increment.

    THIS FUNCTION DOES NOT SETTLE THAT, AND MUST NOT. It computes the registered quantity, says
    whether computing it required excluding rows and how many, and marks the substitution as an
    amendment that has not been ratified and that belongs to the author. The unit's own registration
    has already ruled on a case of exactly this shape, for the linearity statistic it registers as
    secondary: "It is the better statistic. It is NOT made primary here ... a unit does not quietly
    substitute a better statistic for the one the theory registration names. Promoting it is an
    amendment to P5 and is the author's to make." Removing the row filter needed no amendment,
    because the filter contradicted a registered exclusion rule and was therefore a defect. Changing
    which estimator is primary is not a defect repair and is recorded here as what it is.
    """
    rows = list(rows)
    out = log_scale_corrected_slope(rows, rho)
    n_excluded = len(rows) - int(out.get("n_used", 0))
    computable = n_excluded == 0
    return {"registered_quantity": "the attenuation-corrected slope of log increment on log "
                                   "available capability, with the uncorrected slope beside it",
            "slope": out.get("slope"), "slope_without_covariance": out.get("slope_without_covariance"),
            "n_rows": len(rows), "n_used": out.get("n_used"), "n_excluded": int(n_excluded),
            "computable_without_excluding_any_row": bool(computable),
            "registered_sentences": list(REGISTERED_ESTIMATOR_SENTENCES),
            "estimator_actually_used": "the maximum likelihood fit of the paired readings in "
                                       "arc_runner.p5_observation, which keeps every row because it "
                                       "takes no logarithm of the response",
            "why_not_the_registered_one": ("the registered transform cannot be applied to a "
                                           "nonpositive increment, so computing it here required "
                                           "excluding %d of %d cells on their outcome, which the "
                                           "registered exclusion rule forbids"
                                           % (n_excluded, len(rows)) if not computable else
                                           "every cell of this bank returned a positive increment, "
                                           "so the registered transform excluded nothing here and "
                                           "the two estimators are computed on the same cells"),
            "amendment_required": True,
            "amendment_status": "NOT RATIFIED. The registered primary has been superseded in the "
                                "implementation and no amendment to the registration has been "
                                "recorded. The verdict computed from the paired fit is reported "
                                "under that condition.",
            "decision_owner": "the author",
            "open_decision": "whether the paired fit becomes the registered primary, or the "
                             "registered log-scale estimator is restored with an exclusion rule the "
                             "registration would have to name, is the author's decision and is not "
                             "taken by this code"}


# --------------------------------------------------------------------------------------------------
# The log scale diagnostic: the covariance correction in closed form, and what it does not repair
# --------------------------------------------------------------------------------------------------


def log_scale_error_terms(rows: Sequence[Dict[str, Any]], rho: float = 0.0) -> Dict[str, np.ndarray]:
    """The error variances and the error COVARIANCE of the two log scale axes, per row.

    The transform's derivatives do the work, and the retention fraction is handled explicitly because
    it is exact. Writing A_i for the available capability and D_i for the measured increment:

        x_i = log A_i = log f_i + log X0_i, so the retention fraction is an offset carrying no error
              and dx / dX0 = 1 / X0, giving var(x) = v0 / X0 ** 2

        y_i = log D_i, and D_i = G_i + e1_i - e0_i, so dy / dD = 1 / D, giving
              var(y) = (v0 + v1 - 2 rho sqrt(v0 v1)) / D ** 2

        cov(x, y) = cov(e0 / X0, (e1 - e0) / D) = (rho sqrt(v0 v1) - v0) / (X0 D)

    The covariance is negative whenever the reads are less than perfectly correlated, and it is the
    term a scalar ratio of the two variances cannot carry. The previous code divided the before read
    standard deviation by the AVAILABLE capability rather than by the before reading, which overstated
    the predictor's relative error by one over the retention fraction, so at a fifth retention it was
    five times too large.
    """
    X0, D, f, v0, v1, keep = [], [], [], [], [], []
    for r in rows:
        b, a, s0, s1 = _row_pair(r)
        d = a - b
        keep.append(d > 0 and b > 0)
        X0.append(b); D.append(d); f.append(float(r.get("fraction", 1.0))); v0.append(s0); v1.append(s1)
    X0 = np.array(X0); D = np.array(D); f = np.array(f); v0 = np.array(v0); v1 = np.array(v1)
    keep = np.array(keep, bool)
    cross = rho * np.sqrt(np.maximum(v0 * v1, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        var_x = v0 / np.where(X0 > 0, X0 ** 2, np.nan)
        var_y = (v0 + v1 - 2.0 * cross) / np.where(D > 0, D ** 2, np.nan)
        cov_xy = (cross - v0) / np.where((X0 > 0) & (D > 0), X0 * D, np.nan)
        x = np.log(np.where(f > 0, f, np.nan)) + np.log(np.where(X0 > 0, X0, np.nan))
        y = np.log(np.where(D > 0, D, np.nan))
    return {"x": x, "y": y, "var_x": var_x, "var_y": var_y, "cov_xy": cov_xy, "keep": keep}


def log_scale_corrected_slope(rows: Sequence[Dict[str, Any]], rho: float = 0.0) -> Dict[str, Any]:
    """The moment corrected errors-in-variables slope on the log scale, WITH the covariance.

    With known error moments the correction is closed form: the sample covariance of the two axes
    loses the error covariance, and the sample variance of the predictor loses the predictor's error
    variance, so the slope is their corrected ratio. Setting the error moments to zero returns
    ordinary least squares, which is the right limit for an exact read.

    This is a diagnostic and is labelled one. It cannot take the logarithm of an increment that is not
    positive, so it reports how much of the bank it had to drop, and dropping is the selection the
    primary estimator exists to avoid. Its second order transform bias is not removed by any first
    order correction. Read it to see the size of the covariance term, not to decide anything.
    """
    t = log_scale_error_terms(rows, rho)
    keep = t["keep"] & np.isfinite(t["x"]) & np.isfinite(t["y"])
    n = int(keep.sum())
    dropped = 1.0 - n / max(len(list(rows)), 1)
    if n < 3:
        return {"slope": float("nan"), "slope_without_covariance": float("nan"),
                "dropped_fraction": dropped, "n_used": n,
                "note": "fewer than three cells survived the log transform"}
    x, y = t["x"][keep], t["y"][keep]
    sxx = float(np.var(x, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    mean_vx = float(np.nanmean(t["var_x"][keep]))
    mean_cxy = float(np.nanmean(t["cov_xy"][keep]))
    denom = sxx - mean_vx
    slope = float((sxy - mean_cxy) / denom) if abs(denom) > 1e-12 else float("nan")
    without = float(sxy / denom) if abs(denom) > 1e-12 else float("nan")
    return {"slope": slope, "slope_without_covariance": without, "dropped_fraction": dropped,
            "n_used": n, "mean_var_x": mean_vx, "mean_cov_xy": mean_cxy,
            "note": "a log scale diagnostic: it drops every nonpositive increment before the "
                    "transform and it does not remove the transform's second order bias, which is "
                    "why the reported coupling is fitted to the paired readings instead"}
