"""Finding A4's acceptance cases: repeating a deterministic score manufactures no precision.

Three questions run through these tests, and each is the finding's own.

Does repeating an unchanged deterministic score improve the claimed precision? It must not, at 1, at
16 and at 256 repeats, and the tests carry the arithmetic the reference runner would have printed so
that the size of the manufactured claim is on the record rather than merely asserted away.

Does a genuinely fresh sample propagate its actual uncertainty? It must, and the claimed figure is
checked against the spread of many independent reads rather than against itself, because a formula
that agrees with nothing outside its own module is a formula nobody has tested.

Does a response that contains an answer count as having solved the task? It must not. A page of
candidate answers passes a substring check and solves nothing, so the reference ladder's check is an
exact one and the substring check is refused on the deciding path wherever it is attached.
"""
import functools

import numpy as np
import pytest

from arc_runner import code_domain as CD, custody, ladder as L, mode as MODE, p5, sampling as S


# --------------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------------

def _lib(n_correct, n_total):
    lines = []
    for i in range(n_total):
        body = "return x + y" if i < n_correct else "return None"
        lines.append("def add_%d(x, y):\n    %s" % (i, body))
    return "\n".join(lines)


def _registered_pool(n=60):
    tasks = []
    for i in range(n):
        name = "add_%d" % i
        checks = ("assert %s(3, 4) == 7" % name, "assert %s(0, 0) == 0" % name,
                  "assert %s(-1, 1) == 0" % name)
        tasks.append(CD.Task(id=name, statement="Define %s(x, y) returning the sum." % name,
                             signature="def %s(x, y): ..." % name, shown_examples=(checks[0],),
                             checks=checks))
    return CD.TaskPool(tasks, name="registered-domain-pool")


def _suite(n=100, subset=None):
    return CD.SuiteLadder(_registered_pool(n), subset_size=subset,
                          batch_runner=CD.inprocess_batch_runner())


REPEATS = (1, 16, 256)


# --------------------------------------------------------------------------------------------------
# Acceptance case 1: repeating an unchanged deterministic score at 1, 16 and 256 reads
# --------------------------------------------------------------------------------------------------

def test_repeating_a_fixed_form_score_does_not_improve_the_claimed_precision():
    """The reference case: fixed verifiers, a fixed item set, one unchanged artefact.

    The score is exact conditional on that set, so every repeat returns the first read again. The
    claimed precision must be identical at 1, 16 and 256 repeats, and the reading must say in words
    that its repeats resampled nothing.
    """
    lad = L.reference_checkable_ladder(24)
    art = {"text": "\n".join("%s: %s" % (it["id"], it["answer"]) for it in lad.items[:10])}
    rng = np.random.default_rng(1)
    seen = []
    for r in REPEATS:
        mean, unc, _ = L.read_with_uncertainty(lad, art, rng, r)
        seen.append((mean, unc))
        assert unc.unit is S.AssayUnit.FIXED_FORM
        assert unc.independent_draws == 1, "a repeat of a deterministic score is not a second draw"
        assert unc.exact is True and unc.sd == 0.0
        assert "resamples nothing" in unc.note
    assert len({m for m, _ in seen}) == 1, "the reading itself must not move either"
    assert len({u.sd for _, u in seen}) == 1


def test_the_arithmetic_the_repeats_would_have_bought_is_named_and_refused():
    """What the reference runner claimed, stated as a number so the defect is on the record.

    `read_mean` divided the binomial standard deviation by the square root of the read count whatever
    the ladder was. On this ladder that is a claim of sixteenfold precision at 256 reads, bought from
    255 repetitions of one deterministic score. The test asserts the size of the claim and then that
    the reading does not make it.
    """
    lad = L.reference_checkable_ladder(24)
    art = {"text": "\n".join("%s: %s" % (it["id"], it["answer"]) for it in lad.items[:12])}
    rng = np.random.default_rng(2)
    one = lad.score(art, rng)
    naive = {r: one.binomial_sd / np.sqrt(r) for r in REPEATS}
    assert naive[256] == pytest.approx(naive[1] / 16.0)          # the manufactured claim, exactly
    assert naive[1] > 1.0                                        # and it was not a rounding artefact
    for r in REPEATS:
        _, unc, _ = L.read_with_uncertainty(lad, art, rng, r)
        assert unc.sd == 0.0 and unc.sd < naive[r]


def test_a_whole_pool_read_of_the_suite_has_no_sampling_error_and_repeats_buy_nothing():
    """The SuiteLadder case the finding names: the subset IS the pool.

    A read of every item in a frozen pool is the population and not a sample of it, so it has no
    sampling error against that pool, and reading it again draws the same items again.
    """
    lad = _suite(60)                                             # subset_size defaults to the pool
    art = CD.new_artefact(_lib(30, 60))
    rng = np.random.default_rng(3)
    for r in REPEATS:
        mean, unc, _ = L.read_with_uncertainty(lad, art, rng, r)
        assert mean == 30.0
        assert unc.sd == 0.0 and unc.exact is True and unc.independent_draws == 1
        assert unc.population_size == 60 and unc.subset_size == 60
    # and the figure the finding forbids: a binomial with the subset as its own denominator
    binomial = np.sqrt(60 * 0.5 * 0.5)
    assert binomial > 3.8
    assert L.read_with_uncertainty(lad, art, rng, 1)[1].sd < binomial


def test_score_all_carries_the_denominator_that_makes_it_the_population():
    """The whole-pool read used by the control cells and the balance objects reports its own zero."""
    lad = _suite(40)
    result, passed = lad.score_all(CD.new_artefact(_lib(17, 40)))
    assert result.passes == 17 and len(passed) == 17
    assert result.population_size == 40
    assert result.sampling_sd == 0.0
    assert result.binomial_sd > 0.0, "the raw binomial figure is kept, and is not the sampling error"


# --------------------------------------------------------------------------------------------------
# Acceptance case 2: fresh independently sampled forms propagate their actual uncertainty
# --------------------------------------------------------------------------------------------------

def test_a_subset_read_reports_the_finite_population_error_and_not_a_binomial():
    """A fresh item form drawn without replacement from a frozen pool.

    The draw is hypergeometric, so the claimed error is the binomial figure times the finite
    population correction. The test checks the claim against the closed form and against the spread
    of 400 independent reads, because a formula checked only against itself has not been checked.
    """
    N, k = 100, 40
    lad = _suite(N, subset=k)
    art = CD.new_artefact(_lib(50, N))                            # half the pool passes
    rng = np.random.default_rng(4)
    _, unc, _ = L.read_with_uncertainty(lad, art, rng, 1)
    assert unc.unit is S.AssayUnit.ITEM_FORM and unc.exact is False
    assert unc.population_size == N and unc.subset_size == k

    p = 0.5
    binomial = np.sqrt(k * p * (1 - p))
    closed = binomial * np.sqrt((N - k) / (N - 1.0))
    assert unc.sd == pytest.approx(closed, rel=0.12)
    assert unc.sd < binomial, "a draw without replacement is tighter than one with it"

    empirical = float(np.std([lad.score(art, rng).score for _ in range(400)], ddof=1))
    assert unc.sd == pytest.approx(empirical, rel=0.20), \
        "the claimed error must be the error the reads actually have"


def test_fresh_forms_earn_the_square_root_that_a_repeated_fixed_form_does_not():
    """The one place the square root of the repeat count is earned, and the evidence that it is.

    Each read here draws a different item form, so the repeats are independent draws and the mean of
    r of them really is tighter. The claimed reduction is checked against the spread of 300 repeated
    four-read means, not merely against the single-read figure divided by two.
    """
    N, k = 120, 30
    lad = _suite(N, subset=k)
    art = CD.new_artefact(_lib(60, N))
    rng = np.random.default_rng(5)
    one = L.read_with_uncertainty(lad, art, rng, 1)[1]
    four = L.read_with_uncertainty(lad, art, rng, 4)[1]
    assert four.independent_draws == 4 and one.independent_draws == 1
    assert four.sd == pytest.approx(one.sd / 2.0, rel=0.15)
    means = [float(np.mean([lad.score(art, rng).score for _ in range(4)])) for _ in range(300)]
    assert four.sd == pytest.approx(float(np.std(means, ddof=1)), rel=0.25)


def test_the_stochastic_ladder_keeps_the_square_root_it_earns():
    """The simulated ladder redraws its outcome on every read, so its repeats are fresh draws.

    This is the case that made the defect invisible: the same arithmetic was right here and wrong on
    a deterministic ladder, and nothing in the code distinguished them.
    """
    lad = L.MockLadder(n_items=500, scale=1000.0)
    art = {"kind": "mock", "capability": 400.0}
    rng = np.random.default_rng(6)
    one = L.read_with_uncertainty(lad, art, rng, 1)[1]
    sixteen = L.read_with_uncertainty(lad, art, rng, 16)[1]
    assert one.unit is S.AssayUnit.STOCHASTIC_EXECUTION and one.exact is False
    assert sixteen.independent_draws == 16
    assert sixteen.sd == pytest.approx(one.sd / 4.0, rel=0.1)


def test_the_finite_population_correction_is_monotone_and_ends_at_zero():
    """Reading more of the pool can only tighten the read, and reading all of it settles it.

    The comparison is on the proportion scale and not on the count scale, because the count scale
    changes with the subset size: a read of 100 items has a larger count error than a read of 10 and
    a smaller error about the pool, which is the quantity the read is an estimate of.
    """
    N = 200
    rel = [S.finite_population_sd(int(0.5 * k), k, N) / k for k in (10, 50, 100, 150, 199)]
    assert all(a > b for a, b in zip(rel, rel[1:])), rel
    assert S.finite_population_sd(100, 200, N) == 0.0
    # a boundary subset does not establish that the whole pool passes, so it is not reported as exact
    assert S.finite_population_sd(20, 20, N) > 0.0
    assert S.finite_population_sd(0, 20, N) > 0.0


# --------------------------------------------------------------------------------------------------
# Acceptance case 3: a response carrying many candidate answers has not solved the task
# --------------------------------------------------------------------------------------------------

def test_a_response_enumerating_candidates_passes_the_substring_check_and_solves_nothing():
    """The defect, demonstrated on the check that had it.

    A page listing every plausible value contains the answer to every item, so the substring check
    scores it as a perfect artefact. It solved nothing: it answered nothing.
    """
    lad = L.reference_checkable_ladder(20)
    enumerating = {"text": "\n".join(str(v) for v in range(0, 20000))}
    substring_passes = sum(1 for it in lad.items if L.contains_answer(enumerating["text"], it))
    assert substring_passes == len(lad.items), "the substring check is satisfied by enumeration"
    assert lad.score(enumerating, np.random.default_rng(7)).passes == 0, \
        "the exact check must count none of them as solved"


def test_the_exact_check_requires_one_labelled_answer_and_the_right_one():
    item = {"id": "arith-001", "answer": "42"}
    assert L.exact_answer("arith-001: 42", item) is True
    assert L.exact_answer("preamble\narith-001 = 42\ntrailing", item) is True
    assert L.exact_answer("arith-001: 41", item) is False
    assert L.exact_answer("the answer is 42", item) is False
    assert L.exact_answer("arith-002: 42", item) is False, "another item's line is not this one's"
    assert L.exact_answer("arith-001: 41\narith-001: 42", item) is False, \
        "a response that gave two answers did not give this one"


def test_the_reference_ladder_is_scored_by_the_exact_check_and_a_correct_response_still_passes():
    """A gate that refuses everything is an outage: the exact check must pass a solved item."""
    lad = L.reference_checkable_ladder(12)
    assert [it["verifier"] for it in lad.items] == ["exact_answer"] * 12
    solved = {"text": "\n".join("%s: %s" % (it["id"], it["answer"]) for it in lad.items)}
    assert lad.score(solved, np.random.default_rng(8)).passes == 12
    assert "Answer on one line" in lad.items[0]["prompt"], \
        "a scoring rule the system was never told measures guessing the rule"


# --------------------------------------------------------------------------------------------------
# The deciding path: the reference pool, the substring check and an undeclared unit are all refused
# --------------------------------------------------------------------------------------------------

def _inputs(ladder):
    return MODE.ConfirmatoryInputs(ladder=ladder)


def _ladder_refusals(ladder):
    return [r for r in MODE.missing_confirmatory_inputs(_inputs(ladder))
            if r.startswith("domain-ladder")]


def test_the_reference_arithmetic_pool_cannot_enter_confirmatory_mode():
    for lad in (L.reference_checkable_ladder(10),
                CD.SuiteLadder(CD.reference_pool(8), batch_runner=CD.inprocess_batch_runner())):
        assert any("reference smoke pool" in r for r in _ladder_refusals(lad))


def test_the_substring_verifier_is_refused_even_on_a_registered_pool():
    """Barring the pool does not bar the check: the check is refused on its own marker."""
    items = [{"id": "i-%d" % i, "verifier": "contains_answer", "prompt": "p", "answer": str(i)}
             for i in range(6)]
    lad = L.CheckableLadder(items, {"contains_answer": L.contains_answer})
    assert lad.smoke_only is False, "this ladder's pool is not flagged; only its check is"
    assert custody.smoke_verifiers(lad) == ["verifier:contains_answer"]
    refusals = _ladder_refusals(lad)
    assert any("substring smoke tests" in r for r in refusals), refusals


def test_a_ladder_that_declares_no_sampling_unit_is_refused_on_the_deciding_path():
    """Silence is safe in a demonstration and not good enough in a deciding run."""

    class UndeclaredLadder(L.Ladder):
        def __init__(self):
            super().__init__(10, {"kind": "undeclared"})

        def score(self, artefact, rng):
            return L.LadderResult(passes=5, n_items=10, at_ceiling=False)

    lad = UndeclaredLadder()
    assert S.unit_of(lad) is S.AssayUnit.FIXED_FORM, "silence fails closed to the safe reading"
    assert L.read_with_uncertainty(lad, {}, np.random.default_rng(9), 64)[1].sd == 0.0
    assert any("declares no sampling unit" in r for r in _ladder_refusals(lad))


def test_a_ladder_claiming_to_resample_the_artefact_is_refused_rather_than_given_a_model():
    """A ladder reads the artefact it is handed and cannot resample it. That unit lives above it."""

    class WrongUnitLadder(L.Ladder):
        assay_unit = S.AssayUnit.ARTEFACT_REPEAT
        assay_unit_declared = True

        def __init__(self):
            super().__init__(10, {"kind": "wrong-unit"})

        def score(self, artefact, rng):
            return L.LadderResult(passes=5, n_items=10, at_ceiling=False)

    lad = WrongUnitLadder()
    assert any("cannot resample it" in r for r in _ladder_refusals(lad))
    unc = L.read_with_uncertainty(lad, {}, np.random.default_rng(10), 8)[1]
    assert np.isnan(unc.sd), "no read-level error is defined for that unit and none is invented"


def test_a_registered_suite_ladder_passes_every_ladder_requirement():
    """The gate must let the right ladder through, or it is measuring nothing."""
    lad = _suite(60, subset=20)
    assert _ladder_refusals(lad) == []
    rec = S.read_model_record(lad)
    assert rec["assay_unit"] == "item-form-from-frozen-pool" and rec["declared"] is True
    assert rec["population_size"] == 60 and rec["repeats_are_independent"] is True


# --------------------------------------------------------------------------------------------------
# The sampling unit is a pre-run commitment, and the bank records which one it read under
# --------------------------------------------------------------------------------------------------

def test_the_sampling_unit_is_inside_the_ladder_hash_and_therefore_inside_the_seal():
    """Which population a run measures is sealed the way its item set is: a run scored under one
    sampling model having sealed another is a custody failure, not a silent change of estimand."""
    pool = _registered_pool(20)

    class RelabelledSuite(CD.SuiteLadder):
        assay_unit = S.AssayUnit.STOCHASTIC_EXECUTION

    a = CD.SuiteLadder(pool, subset_size=10, batch_runner=CD.inprocess_batch_runner())
    b = RelabelledSuite(pool, subset_size=10, batch_runner=CD.inprocess_batch_runner())
    assert a.spec["assay_unit"] == "item-form-from-frozen-pool"
    assert a.sha256 != b.sha256, "the same items read under a different sampling model is a different ladder"
    assert custody.ladder_identity(a)["spec"]["assay_unit_declared"] is True


def test_the_bank_records_the_read_model_beside_the_rows():
    """Pass counts with no statement of what they sample cannot be re-analysed by anybody."""
    lad = _suite(40, subset=20)
    from arc_runner import adapters

    class Still:
        name = "still"

        def revise(self, artefact, retained, task, rng):
            return dict(artefact)

    cfg = p5.P5Config(states=(1.0,), fractions=(1.0,), reps=1, control_reads_multiplier=1)
    bank = p5.run_bank(Still(), lad, cfg, np.random.default_rng(11),
                       lambda s: CD.new_artefact(_lib(20, 40)))
    assert bank["read_model"]["assay_unit"] == "item-form-from-frozen-pool"
    assert bank["read_model"]["target_population"].startswith("the artefact's pass fraction")


def test_an_exact_read_takes_ordinary_least_squares_rather_than_inventing_an_error_ratio():
    """P5's errors-in-variables step consumes the read error. When the read is exact there is no
    attenuation to correct, and the estimator is the limit of the correction rather than a ratio
    manufactured from a standard deviation the read does not have."""
    rows = [{"available": float(a), "increment": float(a) ** 0.5, "read_sd": 0.0}
            for a in (10, 20, 40, 80, 160)]
    delta = p5._log_error_ratio(rows)
    assert not np.isfinite(delta)
    x = np.log([r["available"] for r in rows]); y = np.log([r["increment"] for r in rows])
    assert p5._deming(x, y, delta) == pytest.approx(0.5, abs=1e-6)


def test_p5_runs_end_to_end_on_an_exact_whole_pool_ladder(tmp_path):
    """The whole pipeline on the reading the finding is about: a deterministic whole-pool read.

    Every read here has zero sampling error, so the errors-in-variables step has nothing to correct
    and every quantity derived from `read_sd` is zero. The run must still reach a coupling rather
    than a division by a fabricated standard deviation, and it must recover the coupling the system
    was built with, because a repair that makes the exact case unrunnable is not a repair.
    """
    n = 200

    class ScriptedCodeSystem:
        name = "scripted-code"

        def __init__(self, n_total, beta=0.5, a=1.6):
            self.n_total, self.beta, self.a = n_total, beta, a

        def revise(self, artefact, retained, task, rng):
            done = artefact.get("text", "").count("return x + y")
            available = max(float(retained.get("fraction", 1.0)) * max(done, 1), 1e-9)
            step = self.a * available ** self.beta
            new_done = min(self.n_total, done + max(1, int(round(step))))
            out = dict(artefact)
            out["text"] = _lib(new_done, self.n_total)
            out["rounds"] = int(artefact.get("rounds", 0)) + 1
            return out

    lad = _suite(n)                                               # whole pool, so every read is exact
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(4, n)))
    for target in (20, 40, 70):
        store.save(CD.state_name(target), CD.new_artefact(_lib(target, n)))
    cfg = p5.P5Config(states=(20, 40, 70), fractions=(0.4, 0.7, 1.0), reps=2, window_end=16,
                      checkpoints=(4, 8, 16), calibration_depths=(1, 2, 3, 4), cal_reads=1, reads=1,
                      heldout_reads=1, replicates=2, bootstrap=20, control_reads_multiplier=1)
    res = p5.run_p5(ScriptedCodeSystem(n), lad, cfg, 11, CD.place_at_state_factory(store),
                    CD.start_for_factory(store), ["S1"])
    assert all(row["read_sd"] == 0.0 for row in res["bank"]["rows"])
    assert res["routes"]["eiv_correction"].startswith("none:")
    assert res["routes"]["beta_pooled"] == pytest.approx(0.5, abs=0.05), res["routes"]


# --------------------------------------------------------------------------------------------------
# Acceptance case 1 again, one level down: the DECLARATION is not the guarantee, the readings are
#
# The first repair of A4 moved the guarantee from the arithmetic into the ladder's declared sampling
# unit, which left the finding's own headline arithmetic one wrong word away. A deterministic ladder
# declaring the stochastic unit was handed the square root of its read count exactly as the reference
# runner had handed it to everything: five identical scores of 20 reported at 3.16, 0.79 and 0.20 for
# 1, 16 and 256 reads, and no gate saw it. These cases run the mislabelled ladders themselves.
# --------------------------------------------------------------------------------------------------

class _MislabelledCheckable(L.CheckableLadder):
    """Deterministic in every respect, and says it redraws its outcome on every read."""

    assay_unit = S.AssayUnit.STOCHASTIC_EXECUTION
    assay_unit_declared = True


def _mislabelled_checkable(n=24):
    items = [{"id": "arith-%03d" % i, "verifier": "exact_answer", "prompt": "p", "answer": str(i)}
             for i in range(n)]
    return _MislabelledCheckable(items, {"exact_answer": L.exact_answer})


def test_a_deterministic_ladder_declaring_the_stochastic_unit_still_buys_no_precision():
    """The refuting case, run: the label says fresh draws and the readings say otherwise.

    Every read returns the same count over the same item form, so the readings themselves refute the
    declaration and the per-read error is reported undivided: identical at 1, 16 and 256 reads. The
    arithmetic the declaration would have bought is asserted first, so the size of the claim that is
    being refused is on the record.
    """
    lad = _mislabelled_checkable(24)
    art = {"text": "\n".join("arith-%03d: %d" % (i, i) for i in range(12))}
    rng = np.random.default_rng(21)
    one = lad.score(art, rng)
    assert one.passes == 12
    declared = {r: one.binomial_sd / np.sqrt(r) for r in REPEATS}
    assert declared[256] == pytest.approx(declared[1] / 16.0) and declared[1] > 1.0

    seen = []
    for r in REPEATS:
        _, unc, _ = L.read_with_uncertainty(lad, art, rng, r)
        seen.append(unc)
        assert unc.independent_draws == 1, "identical readings are one reading counted r times"
        assert unc.sd == pytest.approx(one.binomial_sd), "the per-read error, undivided"
        if r > 1:
            assert unc.resampling_witnessed is False
            assert "identical in outcome and in the form they drew" in unc.note
    # To within floating point: averaging 256 identical values and averaging 16 of them differ in the
    # last bit of the summation, which is not a claim about precision.
    assert max(u.sd for u in seen) == pytest.approx(min(u.sd for u in seen), rel=1e-12), \
        "1, 16 and 256 reads must claim the same precision"
    assert seen[-1].sd > declared[256] * 15, "and the sixteenfold claim is what was refused"


def test_a_suite_ladder_that_draws_the_same_form_every_read_buys_no_precision():
    """The second refuting case: a registered pool, the item-form unit, and a fixed draw.

    The subclass scores with its own generator, so every read draws the same items and returns the
    same count. It is an item-form ladder by declaration and a fixed form in fact, and the reading
    reports the per-read finite-population error undivided rather than dividing by four or by sixteen.
    """

    class OneFormSuite(CD.SuiteLadder):
        def score(self, artefact, rng):
            return super().score(artefact, np.random.default_rng(7))   # the same form, every read

    lad = OneFormSuite(_registered_pool(100), subset_size=40, batch_runner=CD.inprocess_batch_runner())
    art = CD.new_artefact(_lib(50, 100))
    rng = np.random.default_rng(22)
    sds = []
    for r in REPEATS:
        _, unc, _ = L.read_with_uncertainty(lad, art, rng, r)
        sds.append(unc.sd)
        assert unc.independent_draws == 1 and unc.unit is S.AssayUnit.ITEM_FORM
    assert max(sds) == pytest.approx(min(sds), rel=1e-12) and sds[0] > 0.0, sds


def test_two_item_forms_that_count_the_same_passes_are_still_two_independent_draws():
    """The check must not be a count comparison, or it would take back a square root that was earned.

    Every item of this pool fails, so every read counts zero whatever it drew. The forms differ, the
    reads are genuine draws, and the reading says so: the drawn form is part of what two readings are
    compared on, precisely so that a coincidence of counts does not look like a repeat.
    """
    lad = _suite(120, subset=30)
    art = CD.new_artefact(_lib(0, 120))                       # nothing passes, so every count is zero
    rng = np.random.default_rng(23)
    mean, unc, _ = L.read_with_uncertainty(lad, art, rng, 4)
    assert mean == 0.0
    assert unc.resampling_witnessed is True and unc.independent_draws == 4
    assert unc.sd > 0.0, "a boundary count does not establish that the whole pool fails"


# --------------------------------------------------------------------------------------------------
# The gate asks the ladder to SHOW what a repeat resamples, before the run spends anything
# --------------------------------------------------------------------------------------------------

def test_the_gate_refuses_a_ladder_that_cannot_show_what_a_repeat_would_draw():
    """A declaration of independent repeats is checkable for nothing, so a deciding run checks it.

    The mislabelled ladder above has no witness to give: it draws nothing, and `resampling_witness`
    returns the base class's None. The refusal names what is missing rather than merely refusing.
    """
    lad = _mislabelled_checkable(12)
    refusals = _ladder_refusals(lad)
    assert any("cannot show what a repeat would draw afresh" in r for r in refusals), refusals
    assert S.resampling_witness_of(lad)[0] == "absent"


def test_a_ladder_whose_witness_never_varies_is_refused_too():
    """A witness that answers the same thing every time has shown that repeats draw the same thing."""

    class ConstantWitnessSuite(CD.SuiteLadder):
        def resampling_witness(self, rng):
            return "always-the-same-form"

    lad = ConstantWitnessSuite(_registered_pool(60), subset_size=20,
                              batch_runner=CD.inprocess_batch_runner())
    assert S.resampling_witness_of(lad)[0] == "constant"
    assert any("cannot show what a repeat would draw afresh" in r for r in _ladder_refusals(lad))


def test_a_whole_pool_ladder_needs_no_witness_because_it_claims_no_reduction():
    """The exemption, and the reason for it: a whole-pool read IS the population and reports zero.

    Its witness is constant by construction, so a rule that asked every item-form ladder for a
    varying one would refuse the most exact reading the runner can take. A gate that refuses the
    right ladder is an outage.
    """
    whole = _suite(60)
    assert S.claims_a_reduction(whole) is False
    assert _ladder_refusals(whole) == []
    assert S.claims_a_reduction(_suite(60, subset=20)) is True


def test_the_honest_ladders_can_show_their_resampling():
    """Both ladders whose repeats really are draws exhibit two different draws on demand."""
    assert S.resampling_witness_of(_suite(80, subset=20))[0] == "varies"
    assert S.resampling_witness_of(L.MockLadder(n_items=500, scale=400.0))[0] == "varies"
    rec = S.read_model_record(_suite(80, subset=20))
    assert rec["resampling_witness"] == "varies"
    assert S.read_model_record(L.reference_checkable_ladder(8))["resampling_witness"] == "not-claimed"


# --------------------------------------------------------------------------------------------------
# The substring check is refused wherever it is attached AND however it is wrapped
# --------------------------------------------------------------------------------------------------

def _substring_items(n=6):
    return [{"id": "i-%d" % i, "verifier": "contains_answer", "prompt": "p", "answer": str(i)}
            for i in range(n)]


def test_a_partial_or_a_wraps_wrapper_around_the_substring_check_is_still_refused():
    """Binding an option to a verifier is the ordinary way to reuse one, and it defeated the marker.

    `functools.partial(ladder.contains_answer)` and a `functools.wraps` wrapper are both new callables
    carrying no mark of their own. The marker is now read through what they delegate to, so a
    registered pool scored by either is refused exactly as the bare function is.
    """
    items = _substring_items()

    @functools.wraps(L.contains_answer)
    def wrapped(text, item):
        return L.contains_answer(text, item)

    for verifier in (functools.partial(L.contains_answer), wrapped, L.contains_answer):
        lad = L.CheckableLadder(items, {"contains_answer": verifier})
        assert lad.smoke_only is False, "the pool is not flagged; only the check is"
        assert custody.smoke_verifiers(lad) == ["verifier:contains_answer"]
        assert any("substring smoke tests" in r for r in _ladder_refusals(lad))


def test_an_opaque_wrapper_carries_no_mark_and_is_refused_for_attesting_nothing():
    """The evasion the negative marker cannot close, closed by requiring a positive attestation.

    A one-line lambda around the substring check delegates to nothing the marker can be read through.
    A deciding run therefore asks each check to attest that it decides whether an item was SOLVED, and
    refuses what attests nothing: an unmarked wrapper around a smoke check is indistinguishable from
    an unmarked measurement, and the wrapper's author re-attesting is a deliberate act.
    """
    lad = L.CheckableLadder(_substring_items(),
                            {"contains_answer": lambda text, item: L.contains_answer(text, item)})
    assert custody.smoke_verifiers(lad) == [], "the mark is genuinely unreachable through this wrapper"
    assert custody.unattested_verifiers(lad) == ["verifier:contains_answer"]
    refusals = _ladder_refusals(lad)
    assert any("do not attest" in r for r in refusals), refusals


def test_an_attested_check_passes_and_the_reference_checks_are_attested():
    """The gate must pass the checks that do decide items, or it is an outage in the shape of a gate."""
    assert custody.unattested_verifiers(_suite(40, subset=10)) == []
    lad = L.CheckableLadder([{"id": "a", "verifier": "exact_answer", "prompt": "p", "answer": "1"}],
                            {"exact_answer": L.exact_answer})
    assert custody.unattested_verifiers(lad) == []
    assert not any("do not attest" in r for r in _ladder_refusals(lad))
    assert custody.unattested_verifiers(
        CD.SuiteLadder(_registered_pool(8), verifier=CD.inprocess_verifier())) == []


def test_a_ladder_cannot_widen_its_population_after_the_read_that_drew_from_it():
    """The denominator is the one the READING had, and never one a reconfigured ladder supplies.

    A ladder claiming a pool of ten thousand behind a read of the whole hundred would report a
    sampling error where the read has none, which is the finite-population correction running
    backwards. The reading is asked first, so the claim on the ladder cannot reach the arithmetic.
    """

    class OverstatedPool(CD.SuiteLadder):
        def population_size(self):
            return 10000

    lad = OverstatedPool(_registered_pool(100), batch_runner=CD.inprocess_batch_runner())
    _, unc, _ = L.read_with_uncertainty(lad, CD.new_artefact(_lib(50, 100)),
                                        np.random.default_rng(24), 3)
    assert unc.population_size == 100 and unc.sd == 0.0 and unc.exact is True


def test_every_recorded_read_says_which_form_it_drew():
    """A bundle row carrying a pass count and no form cannot be re-analysed: two rows counting the
    same passes are two draws or one draw counted twice, and only the form digest separates them."""
    lad = _suite(80, subset=20)
    custody.attach_read_log(lad)
    L.read_with_uncertainty(lad, CD.new_artefact(_lib(40, 80)), np.random.default_rng(25), 3, "cell")
    rows = lad.read_log
    assert len(rows) == 3 and all(r["form_sha256"] for r in rows)
    assert len({r["form_sha256"] for r in rows}) == 3, "three draws, three forms"
    assert all(r["assay_unit"] == "item-form-from-frozen-pool" and r["population_size"] == 80
               for r in rows)


def test_the_ladder_identity_records_what_each_check_declared_itself_to_be():
    """A bundle reader sees what the gate saw: the source digest says which implementation decided the
    items and cannot say whether its author held it out as a measurement or as a wiring smoke test."""
    smoke = L.CheckableLadder(_substring_items(),
                              {"contains_answer": functools.partial(L.contains_answer)})
    entry = custody.ladder_identity(smoke)["verifiers"][0]
    assert entry["declares_substring_smoke_test"] is True and entry["attests_exact_check"] is False
    good = custody.ladder_identity(_suite(40, subset=10))["verifiers"][0]
    assert good["attests_exact_check"] is True and good["declares_substring_smoke_test"] is False
