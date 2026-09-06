"""Finding A5: the correlated before-and-after error, and the positive-increment filter.

The five acceptance cases the finding names are here under their own names, and each is assessed on
COVERAGE and on WRONG VERDICTS rather than on point recovery alone: a constant true increment with
noisy shared baselines, true zero growth, negative growth, unequal before and after precision, and
correlated item errors. Beside them are the two structural claims the repair rests on, being that the
error on the two derived axes is correlated by construction and that dropping the nonpositive
increments selects on the sign of the read noise.

The worlds are built as bank rows directly rather than through an adapter, because what is under test
is the estimator and not the loop, and because a hundred simulated banks at forty milliseconds each is
a test and at four seconds each is not.
"""
import numpy as np
import pytest

from arc_runner import adapters, ladder as L, p5
from arc_runner import p5_observation as PO

STATES = (30.0, 60.0, 100.0, 160.0, 250.0)
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def bank_rows(rng, increment, reps=5, process_sd=0.05, n_items=20000, scale=400.0,
              reads=4, reads_after=None, read_rho=0.0, states=STATES, fractions=FRACTIONS):
    """A crossed bank of paired readings, with the shape `p5.run_bank` produces.

    `increment(available, state)` is the TRUE mean increment of the round in capability units. The two
    readings are means of `reads` item forms, so their errors are the errors a real ladder has, and
    `read_rho` correlates them the way a ladder that draws one item form and reads it twice would.
    """
    after_reads = reads_after or reads
    rows = []
    for s in states:
        for f in fractions:
            for r in range(reps):
                U = float(s)
                g = increment(f * U, U)
                if process_sd > 0 and g != 0:
                    g = g * float(np.exp(rng.normal(0.0, process_sd)))
                p0 = min(max(U / scale, 0.0), 1.0)
                p1 = min(max((U + g) / scale, 0.0), 1.0)
                sd0 = float(np.sqrt(n_items * p0 * (1 - p0) / reads))
                sd1 = float(np.sqrt(n_items * p1 * (1 - p1) / after_reads))
                z = rng.normal(size=2)
                if read_rho:
                    z[1] = read_rho * z[0] + np.sqrt(1.0 - read_rho ** 2) * z[1]
                before = n_items * p0 + sd0 * z[0]
                after = n_items * p1 + sd1 * z[1]
                rows.append({"state": s, "fraction": f, "rep": r, "control": False,
                             "before": before, "after": after,
                             "var_before": sd0 ** 2, "var_after": sd1 ** 2,
                             "available": f * before, "increment": after - before,
                             "read_sd": float(np.hypot(sd0, sd1))})
    return rows


def previous_estimator(rows):
    """The estimator this finding replaces: drop the nonpositive increments, then take the Deming
    slope of the log increment on the log available capability with a single error ratio."""
    kept = [r for r in rows if r["increment"] > 0]
    if len(kept) < 4:
        return float("nan"), 1.0 - len(kept) / len(rows)
    delta = p5._log_error_ratio(kept)
    slope = p5._deming(np.log([r["available"] for r in kept]),
                       np.log([r["increment"] for r in kept]), delta)
    return slope, 1.0 - len(kept) / len(rows)


# --------------------------------------------------------------------------------------------------
# The two structural claims
# --------------------------------------------------------------------------------------------------

def test_the_two_derived_axes_carry_the_same_reading_and_therefore_a_covariance():
    """The available capability and the increment are built from one before reading, so their errors
    are correlated by construction. The estimator now states that covariance per row; a single ratio
    of two error variances has no place to put it."""
    rows = bank_rows(np.random.default_rng(1), lambda a, u: 0.70 * a ** 0.5)
    terms = PO.log_scale_error_terms(rows, 0.0)
    X0 = np.array([r["before"] for r in rows])
    D = np.array([r["increment"] for r in rows])
    v0 = np.array([r["var_before"] for r in rows])
    # the exact first-order covariance of the two log axes, and it is negative everywhere
    assert np.allclose(terms["cov_xy"], -v0 / (X0 * D))
    assert np.all(terms["cov_xy"] < 0)
    # and it is not small beside the predictor's own error variance, which is what the ratio carried
    assert np.mean(np.abs(terms["cov_xy"])) > 5.0 * np.mean(terms["var_x"])


def test_reads_that_share_their_item_form_cancel_in_the_difference():
    """A declared read correlation is not decoration. Where the two reads of a cell share their item
    form completely and are equally precise, the common error cancels in the subtraction and the
    covariance between the two axes is zero; the previous parameterisation could not express either
    end of that."""
    rows = bank_rows(np.random.default_rng(2), lambda a, u: 0.70 * a ** 0.5)
    for r in rows:
        r["var_after"] = r["var_before"]
    assert np.all(PO.log_scale_error_terms(rows, 0.0)["cov_xy"] < 0)
    assert np.allclose(PO.log_scale_error_terms(rows, 1.0)["cov_xy"], 0.0)


def test_the_retention_fraction_is_exact_and_carries_no_error_onto_the_predictor():
    """The retention fraction is set by the experimenter, so log(f * X0) = log f + log X0 and the
    predictor's error is the before reading's error over the before READING. The previous code
    divided the read standard deviation by the AVAILABLE capability, which overstated the predictor's
    relative error by one over the retention fraction: five times over at a fifth retention."""
    rows = bank_rows(np.random.default_rng(3), lambda a, u: 0.70 * a ** 0.5)
    terms = PO.log_scale_error_terms(rows, 0.0)
    X0 = np.array([r["before"] for r in rows])
    v0 = np.array([r["var_before"] for r in rows])
    assert np.allclose(terms["var_x"], v0 / X0 ** 2)
    # the quantity the previous ratio was built from, on the same rows, and the factor it was out by
    previous = np.array([(r["read_sd"] / np.sqrt(2)) / r["available"] for r in rows]) ** 2
    fractions = np.array([r["fraction"] for r in rows])
    lowest = fractions == min(FRACTIONS)
    assert np.mean(previous[lowest] / terms["var_x"][lowest]) > 10.0


def test_the_corrected_moment_slope_is_the_stated_identity():
    """The closed form is checked as arithmetic, not blessed by a simulation: the corrected slope is
    the sample covariance less the mean error covariance, over the sample variance of the predictor
    less its mean error variance. Omitting the covariance term changes the answer, which is the whole
    of the first half of the finding."""
    rows = bank_rows(np.random.default_rng(4), lambda a, u: 0.70 * a ** 0.5)
    out = PO.log_scale_corrected_slope(rows, 0.0)
    t = PO.log_scale_error_terms(rows, 0.0)
    keep = t["keep"] & np.isfinite(t["x"]) & np.isfinite(t["y"])
    x, y = t["x"][keep], t["y"][keep]
    sxx = float(np.var(x, ddof=1))
    sxy = float(np.cov(x, y, ddof=1)[0, 1])
    expected = (sxy - float(np.nanmean(t["cov_xy"][keep]))) / (sxx - float(np.nanmean(t["var_x"][keep])))
    assert out["slope"] == pytest.approx(expected, rel=1e-12)
    assert out["slope"] != pytest.approx(out["slope_without_covariance"], rel=1e-9)


def test_dropping_the_nonpositive_increments_selects_on_the_sign_of_the_read_noise():
    """The filter is not neutral. On a bank whose true increment does not depend on the available
    capability at all, the cells it removes are the cells whose read noise ran downwards, and they
    are concentrated wherever the increment is smallest relative to that noise, so what survives has
    a slope that nothing in the world put there. On this ladder the read error is largest in the
    middle of the range, so the removal falls hardest on the upper half; on another ladder it would
    fall elsewhere. What matters is that it does not fall evenly, and that the surviving cells
    therefore carry a tilt.
    """
    banks = [bank_rows(np.random.default_rng(2000 + s), lambda a, u: 0.5) for s in range(12)]
    low_rates, high_rates, previous, ours = [], [], [], []
    for rows in banks:
        assert any(r["increment"] <= 0 for r in rows), "this world must produce nonpositive cells"
        median = float(np.median([r["available"] for r in rows]))
        low_rates.append(np.mean([r["increment"] <= 0 for r in rows if r["available"] < median]))
        high_rates.append(np.mean([r["increment"] <= 0 for r in rows if r["available"] >= median]))
        previous.append(previous_estimator(rows)[0])
        ours.append(PO.fit_paired(rows).beta)
    assert abs(float(np.mean(low_rates)) - float(np.mean(high_rates))) > 0.03, \
        "the filter removes %.3f of the low half and %.3f of the high half" \
        % (float(np.mean(low_rates)), float(np.mean(high_rates)))
    # and the tilt is in the surviving cells: the previous estimator reads a slope off a flat world
    assert float(np.nanmean(previous)) > 0.05, "the truncation tilt: %.4f" % float(np.nanmean(previous))
    assert abs(float(np.mean(ours))) < 0.02, "and the fit that keeps every cell has none: %.4f" \
                                             % float(np.mean(ours))
    # the fit sees every cell, including the ones the filter would have removed
    fit = PO.fit_paired(banks[0])
    assert fit.n_rows == len(banks[0])
    assert fit.n_nonpositive == len([r for r in banks[0] if r["increment"] <= 0]) > 0


# --------------------------------------------------------------------------------------------------
# Acceptance case 1: a constant true increment with noisy shared baselines
# --------------------------------------------------------------------------------------------------

def test_a_constant_true_increment_with_noisy_shared_baselines():
    """The true exponent is zero: the round adds the same amount whatever it was given. Assessed on
    coverage and on the wrong verdict, which here is reporting an exponent whose interval excludes
    zero when there is no dependence on the available capability at all."""
    covered = wrong = 0
    previous_wrong = 0
    trials = 30
    for seed in range(trials):
        rows = bank_rows(np.random.default_rng(400 + seed), lambda a, u: 0.8)
        fit = PO.fit_paired(rows)
        assert fit.usable
        covered += abs(fit.beta) <= 1.96 * fit.beta_se
        wrong += abs(fit.beta) > 1.96 * fit.beta_se
        prev, _ = previous_estimator(rows)
        previous_wrong += abs(prev) > 1.96 * fit.beta_se
    assert covered >= 26, "coverage %d of %d at a nominal 28" % (covered, trials)
    assert wrong <= 4
    # the estimator this replaces is further from zero than the repaired interval allows, more often
    assert previous_wrong > wrong


# --------------------------------------------------------------------------------------------------
# Acceptance case 2: true zero growth
# --------------------------------------------------------------------------------------------------

def test_true_zero_growth_is_reported_as_no_measurable_growth_and_never_as_a_coupling():
    """A bank in which the round does nothing at all. There is no elasticity of an increment that is
    not there, so the honest result is that the rate is not distinguishable from zero, and the wrong
    verdict is a coupling fitted to the half of the cells whose noise ran upwards."""
    reported = previous_reported = 0
    trials = 25
    previous_slopes = []
    for seed in range(trials):
        rows = bank_rows(np.random.default_rng(600 + seed), lambda a, u: 0.0)
        fit = PO.fit_paired(rows)
        reported += fit.usable
        if fit.adequacy == PO.NO_MEASURABLE_GROWTH:
            assert "not distinguishable from zero" in fit.adequacy_reason
        prev, dropped = previous_estimator(rows)
        previous_slopes.append(prev)
        previous_reported += np.isfinite(prev)
        assert dropped > 0.3, "about half the cells go the wrong way when the truth is zero growth"
    assert reported <= 3, "a bank with no growth reported a coupling %d times in %d" % (reported, trials)
    assert previous_reported == trials, "the estimator this replaces reported one every time"
    # and what it reported was not near zero either: it was the slope the truncation put there
    assert abs(float(np.nanmean(previous_slopes))) > 0.1


def test_a_zero_growth_bank_stops_the_runner_reporting_a_coupling_at_all():
    """The route level, not only the fit: `estimate_routes` returns the reason rather than a number.

    The assertions below are chosen to be ones the estimator this replaces cannot satisfy. It also
    declined this bank, but by the precision condition and after deleting the half of the cells whose
    noise ran downwards, so it never had a fit of the whole bank to report and never said that the
    increment was not distinguishable from zero. Here the fit saw every cell, the count of the ones
    that came back nonpositive is carried rather than acted on, and the adequacy is the observation
    model's own word.
    """
    rows = bank_rows(np.random.default_rng(11), lambda a, u: 0.0)
    routes = p5.estimate_routes({"rows": rows}, p5.P5Config())
    assert not np.isfinite(routes["beta_pooled"])
    assert routes["identification"] == "INCONCLUSIVE"
    assert PO.NO_MEASURABLE_GROWTH in routes["reason"] or "H2 fails" in routes["reason"]
    # the whole bank was fitted, nonpositive cells included, and the fit says what it found
    assert routes["pooled_fit"]["n_rows"] == len(rows)
    assert routes["pooled_fit"]["n_nonpositive"] == len([r for r in rows if r["increment"] <= 0]) > 0
    assert routes["pooled_fit"]["adequacy"] == PO.NO_MEASURABLE_GROWTH
    assert routes["pooled_fit"]["levels_dropped"] == 0
    # and the estimator the registration names is reported beside it, with what it had to throw away
    reg = routes["registered_estimator"]
    assert reg["n_excluded"] == routes["pooled_fit"]["n_nonpositive"]
    assert reg["computable_without_excluding_any_row"] is False


# --------------------------------------------------------------------------------------------------
# Acceptance case 3: negative growth
# --------------------------------------------------------------------------------------------------

def test_negative_growth_is_measured_and_named_rather_than_deleted():
    """Every cell of this bank loses capability. The previous estimator dropped all of them and
    reported the bank as cells the ladder could not resolve, which is an inconvenient outcome
    becoming a generic measurement failure. The model admits the decrement, measures its exponent,
    and reports that the registered growth process does not describe it."""
    rows = bank_rows(np.random.default_rng(7), lambda a, u: -0.5 * a ** 0.5)
    fit = PO.fit_paired(rows)
    assert fit.n_rows == len(rows) and fit.n_nonpositive == len(rows)
    assert fit.rate < 0
    assert fit.adequacy == PO.NEGATIVE_RATE
    assert "does not admit" in fit.adequacy_reason
    assert not fit.usable, "a decrement is not read as the registered coupling"
    assert fit.beta == pytest.approx(0.5, abs=0.1), \
        "the exponent of the decrement is measured, not discarded: %r" % fit.beta
    prev, dropped = previous_estimator(rows)
    assert dropped == 1.0 and not np.isfinite(prev)


def test_a_bank_of_regressions_is_not_reported_as_a_ladder_that_could_not_resolve_it():
    rows = bank_rows(np.random.default_rng(8), lambda a, u: -0.5 * a ** 0.5)
    routes = p5.estimate_routes({"rows": rows}, p5.P5Config())
    assert routes["pooled_fit"]["n_rows"] == len(rows), "no cell was removed before the fit"
    assert routes["pooled_fit"]["adequacy"] == PO.NEGATIVE_RATE
    assert routes["pooled_fit"]["rate"] < 0


# --------------------------------------------------------------------------------------------------
# Acceptance case 4: unequal before and after precision
# --------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("reads,reads_after", [(4, 64), (64, 4)])
def test_unequal_before_and_after_precision(reads, reads_after):
    """The two readings need not be equally precise, and which of them is the precise one changes the
    covariance as well as the variances. Assessed on coverage over repeated banks."""
    covered = 0
    trials = 20
    for seed in range(trials):
        rows = bank_rows(np.random.default_rng(800 + seed), lambda a, u: 0.70 * a ** 0.5,
                         reads=reads, reads_after=reads_after)
        fit = PO.fit_paired(rows)
        assert fit.usable
        covered += abs(fit.beta - 0.5) <= 1.96 * fit.beta_se
    assert covered >= 17, "coverage %d of %d at a nominal 19" % (covered, trials)


# --------------------------------------------------------------------------------------------------
# Acceptance case 5: correlated item errors
# --------------------------------------------------------------------------------------------------

def test_correlated_read_errors_are_carried_when_they_are_declared():
    """A ladder that draws one item form and reads the artefact on it before and after gives the two
    readings a common error. Declared, it is carried through both the conditional mean and the
    conditional variance; the fail-closed default of zero is the wider reading, so an undeclared run
    is not flattered."""
    covered = covered_undeclared = 0
    trials = 20
    widths, widths_undeclared = [], []
    for seed in range(trials):
        rows = bank_rows(np.random.default_rng(900 + seed), lambda a, u: 0.70 * a ** 0.5, read_rho=0.8)
        declared = PO.fit_paired(rows, PO.ObservationModel(read_correlation=0.8))
        undeclared = PO.fit_paired(rows, PO.ObservationModel(read_correlation=0.0))
        covered += abs(declared.beta - 0.5) <= 1.96 * declared.beta_se
        covered_undeclared += abs(undeclared.beta - 0.5) <= 1.96 * undeclared.beta_se
        widths.append(declared.beta_se)
        widths_undeclared.append(undeclared.beta_se)
    assert covered >= 17, "coverage %d of %d at a nominal 19" % (covered, trials)
    assert covered_undeclared >= 17, "the fail-closed default must not lose coverage"
    assert np.mean(widths) < np.mean(widths_undeclared), \
        "declaring the shared item form is what buys the narrower interval; omitting it must cost"


# --------------------------------------------------------------------------------------------------
# Coverage and wrong verdicts, which is what the finding asks to be assessed
# --------------------------------------------------------------------------------------------------

def test_coverage_and_bias_against_the_estimator_this_replaces():
    """The headline comparison, on the clean world where the previous estimator was thought to work.

    The paired fit is assessed on the coverage of its interval and on its bias; the previous
    estimator is assessed on the same banks. Recovery of a point estimate is not the test: an
    estimator that is close on average and wrong about how close it is has not measured anything.
    """
    trials = 40
    covered = 0
    ours, theirs = [], []
    for seed in range(trials):
        rows = bank_rows(np.random.default_rng(1200 + seed), lambda a, u: 0.70 * a ** 0.5)
        fit = PO.fit_paired(rows)
        assert fit.usable
        covered += abs(fit.beta - 0.5) <= 1.96 * fit.beta_se
        ours.append(fit.beta)
        theirs.append(previous_estimator(rows)[0])
    ours = np.asarray(ours); theirs = np.asarray(theirs)
    assert covered >= 34, "coverage %d of %d at a nominal 38" % (covered, trials)
    assert abs(ours.mean() - 0.5) < abs(np.nanmean(theirs) - 0.5), \
        "paired %.4f against previous %.4f" % (ours.mean(), np.nanmean(theirs))
    assert ours.std(ddof=1) < np.nanstd(theirs, ddof=1), \
        "the fit uses every cell, so it must also be the more precise of the two"


# --------------------------------------------------------------------------------------------------
# What the fit refuses, and what it reports rather than adjusting away
# --------------------------------------------------------------------------------------------------

def test_a_saturated_subset_is_refused_rather_than_fitted():
    """Six cells over three states carry six parameters and no residual degrees of freedom, and such
    a fit returns whatever exponent the last decimal of the noise asked for."""
    rows = bank_rows(np.random.default_rng(9), lambda a, u: 0.70 * a ** 0.5, reps=1,
                     states=(30.0, 60.0, 100.0), fractions=(0.5, 1.0))
    fit = PO.fit_paired(rows)
    assert fit.adequacy == PO.TOO_FEW_ROWS and not fit.usable
    assert "degrees of freedom" in fit.adequacy_reason


def test_a_loader_that_does_not_place_one_artefact_is_reported():
    """The latent state is one parameter per capability state because the placement loader places the
    same artefact for every replicate. A loader that does not shows as over-dispersion of the before
    readings within a state, and it is reported rather than absorbed."""
    rng = np.random.default_rng(10)
    good = bank_rows(rng, lambda a, u: 0.70 * a ** 0.5)
    assert PO.state_homogeneity(PO.pack(good))["dispersion"] == pytest.approx(1.0, abs=0.6)
    drifting = bank_rows(np.random.default_rng(10), lambda a, u: 0.70 * a ** 0.5)
    for i, r in enumerate(drifting):                       # each replicate placed a little differently
        shift = 60.0 * ((i % 5) - 2)
        r["before"] += shift
        r["after"] += shift
    assert PO.state_homogeneity(PO.pack(drifting))["dispersion"] > 3.0


def test_the_precision_condition_still_counts_the_registered_quantity():
    """H2 is unchanged: the gate counts the cells that came back nonpositive, because a repair to the
    estimator is not a licence to restate a registered threshold. What changed is that those cells
    are then fitted rather than deleted. The model's expectation of the same fraction is reported
    beside the count so that the difference can be seen before anybody decides which to use."""
    rows = bank_rows(np.random.default_rng(12), lambda a, u: 0.5)
    routes = p5.estimate_routes({"rows": rows}, p5.P5Config())
    observed = float(np.mean([r["increment"] <= 0 for r in rows]))
    assert routes["unresolved_fraction"] == pytest.approx(observed)
    assert routes["expected_nonpositive_fraction"] == pytest.approx(observed, abs=0.15)
    assert observed > 0.2 and "H2 fails" in routes["reason"]
    # The count is the same count the estimator this replaces made. What is not the same is what
    # happened to the cells afterwards, which is the part these assertions pin: they were fitted.
    assert routes["pooled_fit"]["n_rows"] == len(rows)
    assert routes["pooled_fit"]["n_nonpositive"] == int(round(observed * len(rows)))
    assert routes["pooled_fit"]["levels_dropped"] == 0


# --------------------------------------------------------------------------------------------------
# Wiring: the bank records the readings, and a bank saved before this change still re-scores
# --------------------------------------------------------------------------------------------------

def test_the_bank_records_the_two_readings_and_their_variances():
    ad = adapters.MockCouplingAdapter(beta=0.5)
    lad = L.MockLadder(n_items=4000, scale=400.0)
    cfg = p5.P5Config(states=(60.0, 120.0), fractions=(0.5, 1.0), reps=2, control_reads_multiplier=1)
    bank = p5.run_bank(ad, lad, cfg, np.random.default_rng(1),
                       lambda s: {"kind": "mock", "capability": float(s), "rounds": 0})
    for r in bank["rows"]:
        assert r["after"] - r["before"] == pytest.approx(r["increment"])
        assert r["fraction"] * r["before"] == pytest.approx(r["available"])
        assert np.hypot(np.sqrt(r["var_after"]), np.sqrt(r["var_before"])) == pytest.approx(r["read_sd"])
    assert bank["observation_model"]["read_correlation"] == 0.0
    assert "no logarithm is taken of the response" in bank["observation_model"]["response_scale"]


def test_a_row_saved_before_this_change_is_reconstructed_rather_than_refused():
    """Finding A8 requires a bundle written by an earlier run to re-score, so a row carrying only the
    available capability, the increment and the combined read standard deviation is reconstructed."""
    rows = bank_rows(np.random.default_rng(13), lambda a, u: 0.70 * a ** 0.5)
    legacy = [{k: r[k] for k in ("state", "fraction", "rep", "available", "increment", "read_sd",
                                 "control")} for r in rows]
    for old, new in zip(legacy, rows):
        b, a, v0, v1 = PO._row_pair(old)
        assert b == pytest.approx(new["before"])
        assert a == pytest.approx(new["after"])
        assert v0 + v1 == pytest.approx(new["var_before"] + new["var_after"])
    assert PO.fit_paired(legacy).beta == pytest.approx(0.5, abs=0.1)


def test_the_gradient_is_the_gradient_of_the_likelihood():
    """The analytic gradient is what makes the fit affordable, and a wrong one would move the answer
    without moving any test. It is checked against central differences at a point that is not the
    optimum, with a declared read correlation, so every term is exercised."""
    rows = bank_rows(np.random.default_rng(14), lambda a, u: 0.70 * a ** 0.5, reps=3)
    P = PO.pack(rows)
    x = PO._start(P, 0.4)
    x[2] = np.log(0.07)
    x[3:] = x[3:] * 1.01
    analytic = PO.grad(x, P, 0.3)
    numeric = np.zeros_like(x)
    for i in range(x.size):
        h = max(abs(x[i]) * 1e-6, 1e-8)
        a = x.copy(); a[i] += h
        b = x.copy(); b[i] -= h
        numeric[i] = (PO.nll(a, P, 0.3) - PO.nll(b, P, 0.3)) / (2 * h)
    assert np.max(np.abs(analytic - numeric) / np.maximum(np.abs(numeric), 1e-6)) < 1e-5


# --------------------------------------------------------------------------------------------------
# The selection one level up: registered levels, not rows
#
# Removing the row filter is half of finding A5. The other half is that a route was the MEAN of
# separate fits to its registered levels, taken over the levels that produced a usable exponent, and
# a level produces none exactly when its own increment is not distinguishable from zero. That is the
# same selection on the same axis, moved from rows to levels, and the registration forbids it twice:
# "Every run at a registered seed and retention level is included. No run is excluded on its
# outcome", and "No retention level is added, dropped or reweighted after any increment is seen".
# --------------------------------------------------------------------------------------------------

def _dead_retention_level(a, u):
    """A world in which the round genuinely cannot use the smallest retained fraction."""
    return 0.0 if abs(a / u - min(FRACTIONS)) < 1e-9 else 0.70 * a ** 0.5


def _dead_capability_state(a, u):
    """A world with a capability floor: at the lowest state the round produces nothing."""
    return 0.0 if u == min(STATES) else 0.70 * a ** 0.5


def test_a_registered_retention_level_that_produced_nothing_stays_in_the_state_route():
    """At the registered replicate count, on a bank whose round cannot use the smallest retention.

    The level is kept, named as one whose rate is not distinguishable from zero, and fitted with its
    own rate under the common exponent. What is asserted against is the estimate the selected mean
    would have produced on the identical cells, which is what this repair removes.
    """
    rows = bank_rows(np.random.default_rng(31), _dead_retention_level, reps=16)
    cfg = p5.P5Config(reps=16)
    routes = p5.estimate_routes({"rows": rows}, cfg)
    route = routes["state_route_fit"]
    assert route["n_levels"] == len(FRACTIONS) and route["levels_dropped"] == 0
    assert route["levels_missing"] == []
    assert route["n_rows"] == len(rows), "the route fit saw every cell of the bank"
    assert min(FRACTIONS) in route["levels_without_measurable_growth"], \
        "the dead level is named rather than removed"
    # the registered precision condition passes on this bank, so nothing else would have stopped it
    assert routes["unresolved_fraction"] <= cfg.max_unresolved_fraction
    # the per-level fits are still reported, and one of them carries no readable exponent, which is
    # exactly the level the mean would have left out
    per_level = [f["adequacy"] for f in routes["state_route_fits"]]
    assert per_level.count(PO.NO_MEASURABLE_GROWTH) == 1
    assert routes["per_level_fits_are_diagnostics"] is True
    # and the selection the route no longer makes: the mean over the levels that grew is computed
    # here on the same cells, so the size of what was removed is visible in the test itself
    fits = [PO.fit_paired([r for r in rows if r["fraction"] == f]) for f in cfg.fractions]
    selected = [f.beta for f in fits if f.usable]
    assert len(selected) == len(FRACTIONS) - 1
    assert abs(float(np.mean(selected)) - 0.5) > abs(routes["beta_state_route"] - 0.5)


def test_a_capability_state_that_produced_nothing_stays_in_the_retention_route():
    """The same repair on the other route: a capability floor removes no state from the fit."""
    rows = bank_rows(np.random.default_rng(41), _dead_capability_state, reps=16)
    cfg = p5.P5Config(reps=16)
    routes = p5.estimate_routes({"rows": rows}, cfg)
    route = routes["retention_route_fit"]
    assert route["n_levels"] == len(STATES) and route["levels_dropped"] == 0
    assert route["n_rows"] == len(rows)
    assert min(STATES) in route["levels_without_measurable_growth"]
    fits = [PO.fit_paired([r for r in rows if r["state"] == s]) for s in cfg.states]
    assert len([f for f in fits if f.usable]) == len(STATES) - 1, \
        "one registered state carries no readable exponent, and it is still in the route"


def test_combine_refuses_to_average_the_sets_that_happened_to_grow():
    """The arithmetic the routes used to be built with, now fail-closed.

    A mean over the subsets that produced an exponent reweights the ones that did not to zero, after
    their increments have been seen. It returns no number and the reason instead.
    """
    rows = bank_rows(np.random.default_rng(31), _dead_retention_level, reps=16)
    fits = [PO.fit_paired([r for r in rows if r["fraction"] == f]) for f in FRACTIONS]
    assert len([f for f in fits if f.usable]) == len(FRACTIONS) - 1
    out = PO.combine(fits)
    assert not np.isfinite(out["beta"]) and out["n_usable"] == len(FRACTIONS) - 1
    assert "reweight" in out["reason"]
    # and it still combines a set in which every member is usable
    ok = PO.combine([f for f in fits if f.usable])
    assert np.isfinite(ok["beta"]) and ok["unusable"] == []


# --------------------------------------------------------------------------------------------------
# The negative control: a control that produced nothing is the world the control is built for
# --------------------------------------------------------------------------------------------------

def control_rows(rng, increment, reps=4, reads=64, states=STATES, fractions=FRACTIONS, **kw):
    """Control cells in the shape `p5.run_bank` writes them: retained material the round cannot use."""
    rows = bank_rows(rng, increment, reps=reps, reads=reads, states=states, fractions=fractions, **kw)
    for r in rows:
        r["control"] = True
    return rows


def test_a_perfect_negative_control_is_read_as_no_material_coupling_and_not_as_unresolved():
    """The control the design is built for produces no increment at all.

    There is no elasticity of an increment that is not there, so no exponent can be read from these
    cells, and reading that as UNRESOLVED would make the registered IDENTIFIED unreachable for a
    PERFECT control while a control that still grew a little could pass. The judgement is made on the
    increment instead, which still exists when the rate is zero and is on the same capability scale
    as the bank's.
    """
    main = bank_rows(np.random.default_rng(50), lambda a, u: 0.70 * a ** 0.5)
    ctrl = control_rows(np.random.default_rng(51), lambda a, u: 0.0)
    model = PO.ObservationModel()
    cfg = p5.P5Config()
    bank_fit = PO.fit_paired(main, model, cfg.z_interval)
    out = p5._control_coupling(main + ctrl, cfg, model, bank_fit)
    assert out["fit"]["adequacy"] == PO.NO_MEASURABLE_GROWTH, \
        "a control with no increment carries no exponent, which is the situation under test"
    assert out["status"] == "NO MATERIAL COUPLING"
    assert out["basis"].startswith("the increment")
    assert out["control_mean_increment_upper"] < out["bank_mean_increment_lower"]
    assert out["state_sets_dropped"] == 0
    assert out["n_state_sets_in_the_fit"] == len(STATES)
    assert "open_decision" in out
    # and the switch that hands the other reading back to the author actually changes the answer
    strict = p5._control_coupling(main + ctrl, p5.P5Config(control_no_increment_passes=False),
                                  model, bank_fit)
    assert strict["status"] == "UNRESOLVED"


def test_a_control_that_cannot_bound_its_own_increment_stays_unresolved():
    """A control read at a precision at which it could not fail is not a control that passed.

    The registration's own defect log records reading the controls at such a precision once. Here the
    control produced no measurable growth, and so would have passed on the adequacy label alone, but
    the largest increment consistent with its cells is not below the smallest consistent with the
    bank, so it has established nothing and says so.
    """
    main = bank_rows(np.random.default_rng(60), lambda a, u: 0.05 * a ** 0.5)
    ctrl = control_rows(np.random.default_rng(62), lambda a, u: 0.0, reps=2, reads=1,
                        states=(30.0, 100.0, 250.0), fractions=(0.2, 0.6, 1.0))
    model = PO.ObservationModel()
    cfg = p5.P5Config()
    bank_fit = PO.fit_paired(main, model, cfg.z_interval)
    out = p5._control_coupling(main + ctrl, cfg, model, bank_fit)
    assert out["fit"]["adequacy"] == PO.NO_MEASURABLE_GROWTH
    assert out["status"] == "UNRESOLVED"
    assert out["control_mean_increment_upper"] > out["bank_mean_increment_lower"]
    assert "not bounded below the bank" in out["reason"]


def test_a_control_that_carries_the_bank_s_own_coupling_is_still_caught():
    """The repair must not turn the control into something that cannot fail."""
    main = bank_rows(np.random.default_rng(50), lambda a, u: 0.70 * a ** 0.5)
    ctrl = control_rows(np.random.default_rng(53), lambda a, u: 0.70 * a ** 0.5)
    model = PO.ObservationModel()
    cfg = p5.P5Config()
    out = p5._control_coupling(main + ctrl, cfg, model, PO.fit_paired(main, model, cfg.z_interval))
    assert out["status"] == "MATERIAL COUPLING"
    assert out["beta_control"] == pytest.approx(0.5, abs=0.1)
    assert out["basis"].startswith("the exponent")


# --------------------------------------------------------------------------------------------------
# The registered estimator, and the amendment its demotion needs
# --------------------------------------------------------------------------------------------------

def test_the_registered_estimator_is_reported_and_its_demotion_is_declared_unratified():
    """The registration names the log-scale attenuation-corrected slope as the estimate that enters
    the verdict. This module fits something else. That substitution is an amendment and it belongs to
    the author, so it is declared in every run record rather than assumed by the code."""
    rows = bank_rows(np.random.default_rng(70), lambda a, u: 0.70 * a ** 0.5)
    routes = p5.estimate_routes({"rows": rows}, p5.P5Config())
    reg = routes["registered_estimator"]
    assert reg["amendment_required"] is True
    assert reg["amendment_status"].startswith("NOT RATIFIED")
    assert reg["decision_owner"] == "the author"
    assert any("No run is excluded on its outcome" in s for s in reg["registered_sentences"])
    assert any("Logarithms on both sides" in s for s in reg["registered_sentences"])
    assert np.isfinite(reg["slope"]) and reg["n_rows"] == len(rows)
    # on a bank with a nonpositive increment the registered transform cannot be applied to every
    # cell, which is the reason the two registered sentences cannot both be honoured
    flat = bank_rows(np.random.default_rng(71), lambda a, u: 0.5)
    reg_flat = PO.registered_log_scale_estimate(flat)
    assert reg_flat["n_excluded"] == len([r for r in flat if r["increment"] <= 0]) > 0
    assert reg_flat["computable_without_excluding_any_row"] is False
    assert "forbids" in reg_flat["why_not_the_registered_one"]


def test_a_legacy_row_declares_that_its_two_read_variances_are_not_recoverable():
    """`read_sd` is the hypotenuse of two generally unequal read standard deviations, so only their
    sum of squares survives. The reconstruction splits the total equally, which is exact in the sum
    and wrong on each side, and the count of rows it did that to travels with the fit."""
    rows = bank_rows(np.random.default_rng(13), lambda a, u: 0.70 * a ** 0.5)
    legacy = [{k: r[k] for k in ("state", "fraction", "rep", "available", "increment", "read_sd",
                                 "control")} for r in rows]
    assert PO.pack(legacy)["reconstructed"] == len(legacy)
    assert PO.pack(rows)["reconstructed"] == 0
    fit = PO.fit_paired(legacy)
    assert fit.rows_reconstructed == len(legacy)
    assert fit.as_record()["rows_reconstructed"] == len(legacy)
    assert PO.fit_paired(rows).rows_reconstructed == 0
    # the equal split is exact in the total and wrong on each side, and the docstring says so
    unequal = [r for r in rows if abs(r["var_before"] - r["var_after"]) > 1e-6]
    assert unequal, "this ladder reads the two sides at different precisions, which is the point"
    b, a, v0, v1 = PO._row_pair({k: unequal[0][k] for k in
                                 ("state", "fraction", "rep", "available", "increment", "read_sd")})
    assert v0 + v1 == pytest.approx(unequal[0]["var_before"] + unequal[0]["var_after"])
    assert v0 == pytest.approx(v1) and v0 != pytest.approx(unequal[0]["var_before"])
    assert "NOT recoverable" in PO._row_pair.__doc__
