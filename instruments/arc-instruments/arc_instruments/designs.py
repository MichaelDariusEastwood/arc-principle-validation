"""Design sensitivity for the two crown-jewel experiments: P5 (a sealed coupling predicts the finite-window
growth exponent) and P16 (a pre-estimated boundary predicts a later balance-margin reversal).

Both functions simulate the complete decision procedure the registration scores, under the truth and
under the separated alternatives the charter must name, and report the operating characteristics the
rulings of 5 September 2026 require (false affirmative at most 0.05; detection at least 0.80). They are
design calculations under stipulated noise models, never evidence about any real system. The P16 model
deliberately does not presuppose the ceiling formula: it tests whether a margin trajectory calibrated on
disjoint systems transports to held-out systems, which is what P16 claims and no more.

Two facts these simulators established on 5 September 2026, recorded here because a charter will need
them: the registered finite-window statistic must carry the uncertainty of the stipulated noise model,
not the residual of a straight-line fit to a curved trajectory (which is model misfit, not noise); and a
band of 0.10 on the finite-window exponent at a = 0.1, U0 = 1 over depths 1 to 32 corresponds to a
coupling shift of about 0.26 (the exponent moves about 0.38 per unit coupling there), so the separated
alternative for the false-affirmative target must be at least that large, or the window longer, or the
band tighter; the charter chooses and says which.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from .coupling_identification import endpoint_slope, trajectory
from .verdicts import INCONCLUSIVE, REFUTED, SUPPORTED, p5_agreement


def _fit_beta(rng: np.random.Generator, states: int, retentions: int, reps: int, noise: float, beta: float,
              theta: float, lam: float, cap_noise: float) -> Dict[str, float]:
    """One calibration experiment: crossed capability states by retention fractions; returns the fitted
    capability slope and its standard error, with an unmodelled nuisance-rate exponent lam and capability
    measurement error cap_noise (log scale)."""
    lc_true, lf = np.meshgrid(np.log(np.geomspace(1.0, 16.0, states)), np.log(np.geomspace(0.125, 1.0, retentions)), indexing="ij")
    lc_true = np.repeat(lc_true.ravel(), reps)
    lf = np.repeat(lf.ravel(), reps)
    y = -2.0 + (beta + lam) * lc_true + theta * lf + rng.normal(0.0, noise, lc_true.size)
    lc_obs = lc_true + rng.normal(0.0, cap_noise, lc_true.size)
    X = np.column_stack([np.ones(lc_obs.size), lc_obs, lf])
    coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    df = lc_obs.size - 3
    sigma2 = (float(res[0]) if len(res) else float(np.sum((y - X @ coef) ** 2))) / df
    se = float(np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1]))
    return {"beta_hat": float(coef[1]), "se": se}


def exponent_sensitivity_to_coupling(beta: float, a: float = 0.1, U0: float = 1.0, window: float = 32.0, h: float = 1e-4) -> float:
    """d(finite-window endpoint exponent) / d(coupling) at the design's settings."""
    return (endpoint_slope(U0, a, beta + h, 1.0, window) - endpoint_slope(U0, a, beta - h, 1.0, window)) / (2 * h)


def p5_design_power(n_systems: int = 8, states: int = 5, retentions: int = 5, reps: int = 5, noise: float = 0.15,
                    cap_noise: float = 0.0, beta: float = 0.5, theta: float = 0.8, a: float = 0.1, U0: float = 1.0,
                    window: float = 32.0, traj_noise: float = 0.03, band: float = 0.10, alt_shift: float = 0.30,
                    lam: float = 0.3, route_disagreement: float = 0.10, replicates_per_holdout: int = 1,
                    sims: int = 300, seed: int = 20260905) -> Dict[str, object]:
    """Operating characteristics of the P5 decision.

    Per simulated experiment: a crossed calibration per system estimates the coupling; the sealed
    finite-window exponent is the endpoint exponent of the registered solution at the estimated coupling
    (delta-method uncertainty from the coupling's standard error); each held-out trajectory is generated
    from the true mechanism with log noise, and its observed endpoint exponent carries the uncertainty of
    that noise at the two endpoints (the registered statistic's own uncertainty, not a straight-line
    residual); the difference interval is scored by the registered rule (plus or minus `band`) per system,
    and the proposition by the majority of the frozen system weight. Reports: support and refutation
    rates under the truth; the false-affirmative rate when the true coupling is shifted by `alt_shift`;
    the rate at which a capability-dependent nuisance rate (exponent `lam`) is declared non-identified by
    the two-route check; the mean fitted coupling under the design's capability measurement error (the
    attenuation a band can hide); and the exponent's sensitivity to the coupling at the design."""
    rng = np.random.default_rng(seed)
    reps_h = max(1, int(replicates_per_holdout))
    se_obs = traj_noise * (2.0 ** 0.5) / np.log(window) / (reps_h ** 0.5)

    def observed_exponent(beta_true: float) -> float:
        vals = []
        for _ in range(reps_h):
            U = trajectory(U0, a, beta_true, np.array([1.0, window])) * np.exp(rng.normal(0.0, traj_noise, 2))
            vals.append(float(np.log(U[1] / U[0]) / np.log(window)))
        return float(np.mean(vals))

    def one_experiment(true_beta_holdout: float, cal_lam: float) -> str:
        verdicts = []
        for _ in range(n_systems):
            fit = _fit_beta(rng, states, retentions, reps, noise, beta, theta, cal_lam, cap_noise)
            pred = endpoint_slope(U0, a, fit["beta_hat"], 1.0, window)
            se_pred = abs(exponent_sensitivity_to_coupling(fit["beta_hat"], a, U0, window)) * fit["se"]
            diff = pred - observed_exponent(true_beta_holdout)
            se = (se_pred ** 2 + se_obs ** 2) ** 0.5
            verdicts.append(p5_agreement((diff - 1.96 * se, diff + 1.96 * se), band))
        n_sup = sum(v == SUPPORTED for v in verdicts); n_ref = sum(v == REFUTED for v in verdicts)
        if n_sup * 2 > n_systems:
            return SUPPORTED
        if n_ref * 2 > n_systems:
            return REFUTED
        return INCONCLUSIVE

    truth = [one_experiment(beta, 0.0) for _ in range(sims)]
    alt = [one_experiment(beta + alt_shift, 0.0) for _ in range(sims)]
    flagged = 0
    betas = []
    for _ in range(sims):
        r1 = _fit_beta(rng, states, retentions, reps, noise, beta, theta, lam, cap_noise)
        r2 = _fit_beta(rng, states, retentions, reps, noise, beta, theta, 0.0, cap_noise)
        betas.append(r2["beta_hat"])
        gap = abs(r1["beta_hat"] - r2["beta_hat"]); se = (r1["se"] ** 2 + r2["se"] ** 2) ** 0.5
        flagged += int(gap - 1.96 * se > route_disagreement)
    return {
        "support_rate_under_truth": truth.count(SUPPORTED) / sims,
        "refutation_rate_under_truth": truth.count(REFUTED) / sims,
        "false_affirmative_rate_under_shifted_coupling": alt.count(SUPPORTED) / sims,
        "detection_rate_of_shifted_coupling": alt.count(REFUTED) / sims,
        "non_identification_declared_under_nuisance_rate": flagged / sims,
        "mean_calibration_beta_hat": float(np.mean(betas)),
        "exponent_sensitivity_to_coupling": exponent_sensitivity_to_coupling(beta, a, U0, window),
        "coupling_shift_resolvable_by_band": band / abs(exponent_sensitivity_to_coupling(beta, a, U0, window)),
        "design": {"n_systems": n_systems, "states": states, "retentions": retentions, "reps": reps, "noise": noise,
                   "cap_noise": cap_noise, "traj_noise": traj_noise, "band": band, "alt_shift": alt_shift, "lam": lam, "window": window},
    }


def p5_trajectory_pattern(n_systems: int = 8, beta: float = 0.5, a: float = 0.1, U0: float = 1.0, traj_noise: float = 0.03,
                          checkpoints: Sequence[float] = (1, 2, 4, 8, 16, 24, 32), beta_se: float = 0.015,
                          truth: str = "mechanism", sims: int = 200, seed: int = 20260905) -> Dict[str, float]:
    """The pattern test behind P5's single statistic: the sealed full trajectory of the registered solution
    against a pure power law carrying the same sealed endpoint exponent (same endpoint, different shape).
    Under the mechanism the shifted curve wins the held-out trajectory loss at the interior checkpoints;
    when the truth is a pure power law with that endpoint exponent (`truth="pure_power"`), the mechanism
    must lose. Reports the fraction of held-out systems where the sealed mechanism beats the same-endpoint
    power law, averaged over experiments, under each truth; a design predicts the right thing only if the
    first is high and the second is low."""
    rng = np.random.default_rng(seed)
    R = np.asarray(checkpoints, float)
    wins = 0.0
    for _ in range(sims):
        beta_hat = beta + rng.normal(0.0, beta_se)
        sealed = trajectory(U0, a, beta_hat, R)
        e = endpoint_slope(U0, a, beta_hat, 1.0, float(R[-1]))
        same_endpoint_power = U0 * R ** e
        w = 0
        for _ in range(n_systems):
            if truth == "mechanism":
                obs = trajectory(U0, a, beta, R)
            else:
                obs = U0 * R ** endpoint_slope(U0, a, beta, 1.0, float(R[-1]))
            obs = obs * np.exp(rng.normal(0.0, traj_noise, R.size))
            loss_m = float(np.mean((np.log(obs) - np.log(sealed)) ** 2))
            loss_p = float(np.mean((np.log(obs) - np.log(same_endpoint_power)) ** 2))
            w += int(loss_m < loss_p)
        wins += w / n_systems
    return {"truth": truth, "mechanism_wins_fraction": wins / sims}


def p16_intervention_pattern(n_calibration: int = 12, n_holdout: int = 8, delta0: float = 0.30, kappa: float = 0.12,
                             factor: float = 1.25, system_sd: float = 0.05, margin_noise: float = 0.04,
                             checkpoints: Sequence[float] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
                             persistence: int = 2, timing_tolerance_log: float = 0.70, sims: int = 200,
                             seed: int = 20260905) -> Dict[str, float]:
    """The pattern across arms behind P16: a control arm and an intervention arm whose correction
    coefficient is multiplied by `factor` (a level shift, adding log factor to the margin). The sealed
    prediction is a pair: the control's event depth and the intervention's later event depth, with the
    trend slope unchanged. SUPPORTED where both arms' events fall inside their sealed windows and the
    intervention arm's fitted slope agrees with the control's within their joint uncertainty (a level
    shift, not a ceiling shift). Reports the support rate under the truth and under a mislabelled
    alternative in which the intervention also changes the slope, which the pattern must catch as a
    different mechanism. The intervention is sized to the slope: in this margin model a coefficient
    factor f moves the reversal depth by f^(1/kappa), so a doubling at kappa 0.12 moves it three
    hundredfold (past any horizon) while a factor of 1.25 moves it about sixfold; the charter sizes the
    intervention so that both arms' events sit inside the horizon."""
    rng = np.random.default_rng(seed)
    cps = np.asarray(checkpoints, float); lcps = np.log(cps)
    X = np.column_stack([np.ones(cps.size), -lcps])

    def detect(margin_path: np.ndarray) -> float:
        neg = margin_path < 0
        for i in range(cps.size - persistence + 1):
            if neg[i:i + persistence].all():
                return float(cps[i])
        return float("inf")

    def fit(m: np.ndarray) -> Sequence[float]:
        c, *_ = np.linalg.lstsq(X, m, rcond=None)
        return c

    def one(slope_change: float) -> str:
        est = [fit(delta0 + rng.normal(0.0, system_sd) - (kappa + rng.normal(0.0, system_sd / 2)) * lcps + rng.normal(0.0, margin_noise, cps.size)) for _ in range(n_calibration)]
        dbar = float(np.mean([e[0] for e in est])); kbar = float(np.mean([e[1] for e in est]))
        se_k = float(np.std([e[1] for e in est], ddof=1) / np.sqrt(n_calibration))
        if kbar <= 2 * se_k:
            return INCONCLUSIVE
        pred_c = detect(dbar - kbar * lcps); pred_i = detect(dbar + np.log(factor) - kbar * lcps)
        if not (np.isfinite(pred_c) and np.isfinite(pred_i)):
            return INCONCLUSIVE
        ok = 0
        resid_se = margin_noise / np.sqrt(float(np.sum((lcps - lcps.mean()) ** 2)))
        for _ in range(n_holdout):
            d_i = delta0 + rng.normal(0.0, system_sd); k_i = kappa + rng.normal(0.0, system_sd / 2)
            m_c = d_i - k_i * lcps + rng.normal(0.0, margin_noise, cps.size)
            m_i = d_i + np.log(factor) - (k_i + slope_change) * lcps + rng.normal(0.0, margin_noise, cps.size)
            ev_c, ev_i = detect(m_c), detect(m_i)
            kc, ki = fit(m_c)[1], fit(m_i)[1]
            # the sealed pattern is relative: the intervention arm's event sits log(factor)/slope later than
            # its paired control's, with the slope unchanged; both are predicted from the control arm and
            # the population, and the pair shares its cloned starting state
            if not (np.isfinite(ev_c) and np.isfinite(ev_i)) or kc <= 0:
                continue
            pred_ratio = np.log(factor) / kc
            se_ratio = np.log(factor) * resid_se / kc ** 2
            inside = abs((np.log(ev_i) - np.log(ev_c)) - pred_ratio) <= timing_tolerance_log + 1.96 * se_ratio
            slope_same = abs(kc - ki) <= 1.96 * resid_se * (2 ** 0.5)
            ok += int(inside and slope_same)
        return SUPPORTED if ok * 2 > n_holdout else (REFUTED if (n_holdout - ok) * 2 > n_holdout else INCONCLUSIVE)

    level = [one(0.0) for _ in range(sims)]
    slope = [one(kappa * 0.6) for _ in range(sims)]
    return {"support_rate_level_shift_truth": level.count(SUPPORTED) / sims,
            "support_rate_when_intervention_also_changes_slope": slope.count(SUPPORTED) / sims,
            "factor": factor}


def p16_design_power(n_calibration: int = 12, n_holdout: int = 8, delta0: float = 0.30, kappa: float = 0.12,
                     system_sd: float = 0.05, margin_noise: float = 0.04, checkpoints: Sequence[float] = (1, 2, 4, 8, 16, 32, 64, 128),
                     persistence: int = 2, timing_tolerance_log: float = 0.70, horizon: float = 128.0,
                     early_checkpoints: int = 0, sims: int = 300, seed: int = 20260905) -> Dict[str, object]:
    """Operating characteristics of the P16 margin-transport decision under a stipulated margin model.

    Each system's measured balance margin follows Delta(R) = delta0_i - kappa_i log R with system-level
    variation (sd `system_sd`) and per-checkpoint measurement noise. Calibration systems (source-disjoint)
    give the population estimate of (delta0, kappa) and its uncertainty; the sealed forecast for a held-out
    system is the event depth the registered detection rule (the margin below zero at `persistence`
    consecutive checkpoints) returns on the sealed mean path, with a log-depth window of plus or minus
    `timing_tolerance_log` (one doubling by default, the checkpoint spacing) widened by the calibration
    uncertainty; the observed event applies the same rule to the noisy measured margin. SUPPORTED where
    the event lies in the window for more than half the held-out systems; REFUTED where more than half
    fall outside it or never reverse by the horizon; else INCONCLUSIVE. The null sets the population
    kappa to zero (no boundary relation); non-crossing controls have kappa at or below zero and their
    persistent-reversal rate is the false-alarm rate."""
    rng = np.random.default_rng(seed)
    cps = np.asarray(checkpoints, float)
    lcps = np.log(cps)

    def detect(margin_path: np.ndarray) -> float:
        neg = margin_path < 0
        for i in range(cps.size - persistence + 1):
            if neg[i:i + persistence].all():
                return float(cps[i])
        return float("inf")

    def one(kappa_pop: float) -> str:
        d0 = delta0 + rng.normal(0.0, system_sd, n_calibration)
        kk = kappa_pop + rng.normal(0.0, system_sd / 2, n_calibration)
        est_d, est_k = [], []
        X = np.column_stack([np.ones(cps.size), -lcps])
        for i in range(n_calibration):
            m = d0[i] - kk[i] * lcps + rng.normal(0.0, margin_noise, cps.size)
            c, *_ = np.linalg.lstsq(X, m, rcond=None)
            est_d.append(c[0]); est_k.append(c[1])
        dbar, kbar = float(np.mean(est_d)), float(np.mean(est_k))
        se_k = float(np.std(est_k, ddof=1) / np.sqrt(n_calibration))
        se_d = float(np.std(est_d, ddof=1) / np.sqrt(n_calibration))
        if kbar <= 2 * se_k:
            return INCONCLUSIVE                                   # no resolved boundary relation: nothing to seal
        sealed_event = detect(dbar - kbar * lcps)
        if not np.isfinite(sealed_event) or sealed_event > horizon:
            return INCONCLUSIVE                                   # the sealed path never reverses inside the horizon
        cross_se = (se_d / kbar) ** 2 + (dbar * se_k / kbar ** 2) ** 2
        half = timing_tolerance_log + 1.96 * cross_se ** 0.5
        inside = outside = 0
        for _ in range(n_holdout):
            d_i = delta0 + rng.normal(0.0, system_sd); k_i = kappa_pop + rng.normal(0.0, system_sd / 2)
            m = d_i - k_i * lcps + rng.normal(0.0, margin_noise, cps.size)
            if early_checkpoints > 0:
                # per-system forecast sealed at an early depth: the system's own first checkpoints give its
                # level, the calibration population gives the slope; the forecast is sealed before the
                # later checkpoints are observed and only later events count
                d_hat = float(np.mean(m[:early_checkpoints] + kbar * lcps[:early_checkpoints]))
                pred_ev = detect(d_hat - kbar * lcps)
                if not np.isfinite(pred_ev) or pred_ev <= cps[early_checkpoints - 1]:
                    outside += 1
                    continue
                pred_log = np.log(pred_ev)
                half_i = timing_tolerance_log + 1.96 * ((margin_noise / early_checkpoints ** 0.5 / kbar) ** 2 + (d_hat * se_k / kbar ** 2) ** 2) ** 0.5
            else:
                pred_log = np.log(sealed_event); half_i = half
            ev = detect(m)
            if np.isfinite(ev) and abs(np.log(ev) - pred_log) <= half_i:
                inside += 1
            else:
                outside += 1
        if inside * 2 > n_holdout:
            return SUPPORTED
        if outside * 2 > n_holdout:
            return REFUTED
        return INCONCLUSIVE

    truth = [one(kappa) for _ in range(sims)]
    null = [one(0.0) for _ in range(sims)]
    false_alarms = 0
    for _ in range(sims):
        k_c = min(0.0, rng.normal(0.0, system_sd / 2))
        false_alarms += int(np.isfinite(detect(delta0 + rng.normal(0.0, system_sd) - k_c * lcps + rng.normal(0.0, margin_noise, cps.size))))
    return {
        "support_rate_under_truth": truth.count(SUPPORTED) / sims,
        "refutation_rate_under_truth": truth.count(REFUTED) / sims,
        "support_rate_under_null_no_boundary": null.count(SUPPORTED) / sims,
        "inconclusive_rate_under_null": null.count(INCONCLUSIVE) / sims,
        "non_crossing_control_false_alarm_rate": false_alarms / sims,
        "design": {"n_calibration": n_calibration, "n_holdout": n_holdout, "delta0": delta0, "kappa": kappa, "system_sd": system_sd,
                   "margin_noise": margin_noise, "checkpoints": list(map(float, cps)), "persistence": persistence,
                   "timing_tolerance_log": timing_tolerance_log, "horizon": horizon},
    }
