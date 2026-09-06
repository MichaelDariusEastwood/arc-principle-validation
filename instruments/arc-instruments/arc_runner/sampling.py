"""What a repeat of a ladder reading resamples, and therefore what a repeat may claim.

WHY THIS FILE EXISTS. `read_mean` divided a binomial standard deviation by the square root of the
number of repeated reads, whatever the ladder underneath it was. For a ladder that draws a fresh
stochastic realisation on every read that is the standard deviation of a mean and it is right. For a
ladder that applies fixed verifiers to a fixed item set it is precision manufactured out of nothing,
because the second read of a deterministic score is the first read again. Reading one unchanged
artefact 256 times reported a claimed precision sixteen times tighter than reading it once, and not
one of those 256 reads carried anything the first did not. That is finding A4.

THE SEPARATION THIS FILE MAKES, AND IT IS THE WHOLE POINT. Deterministic verification and measurement
uncertainty are two different things and the reference runner ran them together. Whether an artefact
passes item 42 of a frozen pool is settled by running the check: the answer is exact, it has no
standard error, and repeating it cannot make it more exact. What HAS a standard error is the answer
to a question about a POPULATION that the read is a sample of, and a question about a population has
no answer at all until the sampling unit is named. So a ladder declares its unit, and the declaration
decides the arithmetic.

THE ADMISSIBLE UNITS.

FIXED_FORM is a deterministic score over a fixed item set: fixed verifiers, fixed items, fixed
artefact. A repeat resamples nothing whatsoever. The reported sampling error is zero, which is not a
claim of certainty about anything broader: it is the statement that the reading is EXACT CONDITIONAL
on that item set and that artefact, and that this ladder supplies no uncertainty about any wider
population because it draws no sample from one. A run that wants a wider population states a wider
sampling unit; it does not get one by reading the same thing again.

ITEM_FORM is a fresh item form drawn without replacement from a larger frozen pool: the sampling unit
is the item form, the target is the pool, and the uncertainty is the finite-population sampling error
of the subset against the pool. The finite population correction is applied, so a subset that IS the
whole pool has zero sampling error and says so, rather than reporting a binomial standard deviation
with the pool as its own denominator. Reads here are genuinely independent draws, so repeats do
reduce the error by the square root of the count: this is the one place where that arithmetic is
earned, and it is earned because the items differ between reads.

STOCHASTIC_EXECUTION is a fresh independent realisation of a stochastic scoring process on the same
artefact. The simulated ladder is this: it redraws its binomial outcome on every read, so its repeats
carry information that a repeated fixed-form score does not. Repeats reduce the error by the square
root of the count.

ARTEFACT_REPEAT is the unit for repeated artefacts or repeated generation processes. It is named here
because it is a legitimate assay unit and a run may declare it, but a LADDER never supplies it: a
ladder reads an artefact and cannot resample the artefact it was handed. A ladder that declares it is
refused rather than quietly given one of the other models. In this runner that unit is supplied above
the ladder, by P5's replicate continuations from the one sealed checkpoint.

SILENCE FAILS CLOSED. A ladder that declares nothing is read as FIXED_FORM, so an undeclared ladder
reports no reduction from repeats and cannot manufacture precision by omission. A confirmatory run
goes further and refuses an undeclared unit outright, because which population a deciding run
measures is a thing its author states rather than a thing its default supplies.

AND A DECLARATION IS NEVER TAKEN ON TRUST. The first repair of finding A4 moved the guarantee from
the arithmetic to the declaration, which put the whole finding one wrong word away from returning: a
deterministic ladder that declared STOCHASTIC_EXECUTION was handed the square root again, and five
identical scores of 20 were reported at 3.16, 0.79 and 0.20 for 1, 16 and 256 reads. That is the
finding's own headline arithmetic, bought from 255 repetitions of one deterministic score, and no
gate saw it. So the reduction is now earned by the readings and not by the label. `read_uncertainty`
holds every reading it is about to average, and readings that are identical in outcome AND in the
form they drew are not independent draws whatever the ladder calls them: the per-read error is
reported undivided, so 1, 16 and 256 repeats report the same number. The undivided per-read error is
taken rather than the exact zero a fixed form would report, because a misdeclared ladder has not
established which of the two it is and the wider reading is the safe one.

The comparison is on the outcome AND on the drawn form together, which matters in both directions. A
genuine item form that draws different items and happens to count the same passes is still two draws,
because the form digest differs. And a ladder that redraws nothing shows neither, which is the case
this exists to catch. A ladder that reports no form is compared on its outcome alone, so a genuinely
stochastic ladder whose repeats coincide by chance loses the reduction for that reading: the cost is
an interval too wide, which is the direction this module errs in on purpose.

A ladder also SHOWS what its repeats resample before a deciding run starts. `resampling_witness`
returns a digest of what one read would draw afresh, without reading any artefact and without
spending anything, and a ladder that declares independent repeats and cannot produce two different
witnesses is refused by `unit_refusals`. A whole-pool item-form ladder is exempt because it claims no
reduction at all: its reads are the population and it reports zero error.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


class AssayUnit(str, Enum):
    """A string enum so that a manifest carries the word itself and a reader needs no code."""

    FIXED_FORM = "fixed-form-deterministic"
    ITEM_FORM = "item-form-from-frozen-pool"
    STOCHASTIC_EXECUTION = "stochastic-execution"
    ARTEFACT_REPEAT = "artefact-or-process-repeat"

    @property
    def resamples(self) -> str:
        """What one repeat of a reading draws afresh. The sentence a reader needs in order to know
        whether repeating the reading bought anything."""
        return _RESAMPLES[self]

    @property
    def target(self) -> str:
        """The population the reading estimates, being the thing the standard error is about."""
        return _TARGETS[self]

    @property
    def repeats_are_independent(self) -> bool:
        """True when a repeat is a fresh draw, which is the only condition under which the square
        root of the repeat count belongs in the arithmetic."""
        return self in (AssayUnit.ITEM_FORM, AssayUnit.STOCHASTIC_EXECUTION)

    @property
    def supplied_by_a_ladder(self) -> bool:
        """False for the unit whose repeats live above the ladder. See the module docstring."""
        return self is not AssayUnit.ARTEFACT_REPEAT


_RESAMPLES = {
    AssayUnit.FIXED_FORM:
        "nothing: the same verifiers over the same items on the same artefact return the same score",
    AssayUnit.ITEM_FORM:
        "a fresh item form drawn without replacement from the frozen pool",
    AssayUnit.STOCHASTIC_EXECUTION:
        "a fresh independent realisation of the scoring process on the same artefact",
    AssayUnit.ARTEFACT_REPEAT:
        "a fresh artefact or a fresh run of the generating process, which a ladder read cannot do",
}

_TARGETS = {
    AssayUnit.FIXED_FORM:
        "this artefact on this item set, exactly; this ladder samples no wider population and "
        "reports no uncertainty about one",
    AssayUnit.ITEM_FORM:
        "the artefact's pass fraction over the whole frozen pool, of which each read is a sample",
    AssayUnit.STOCHASTIC_EXECUTION:
        "the artefact's expected score under the scoring process, of which each read is one draw",
    AssayUnit.ARTEFACT_REPEAT:
        "the process that produced the artefact, over repeats of that process",
}


@dataclass(frozen=True)
class ReadUncertainty:
    """The sampling error of a reading, with the statement of what it is the sampling error OF.

    A bare number cannot be checked. `unit`, `target` and `independent_draws` travel with `sd` so that
    a later reader can see whether the repeats bought anything, and `note` says in words what the
    arithmetic did, because the defect this replaces was invisible in a bare number.
    """

    sd: float
    unit: AssayUnit
    target: str
    reads: int
    independent_draws: int
    exact: bool
    subset_size: int
    population_size: Optional[int]
    note: str
    # Whether the readings THEMSELVES showed the resampling the declared unit claims: True when they
    # differed, False when a unit claiming independent repeats produced identical readings, and None
    # when one reading was taken and there was nothing to compare. It travels with the number because
    # a claim of independence that the readings refute is the defect, and a reader cannot see it in
    # the standard deviation alone.
    resampling_witnessed: Optional[bool] = None

    def as_record(self) -> Dict[str, Any]:
        return {"sd": float(self.sd), "assay_unit": self.unit.value, "target_population": self.target,
                "reads": int(self.reads), "independent_draws": int(self.independent_draws),
                "exact": bool(self.exact), "subset_size": int(self.subset_size),
                "population_size": (int(self.population_size) if self.population_size is not None else None),
                "resamples": self.unit.resamples,
                "resampling_witnessed": (None if self.resampling_witnessed is None
                                         else bool(self.resampling_witnessed)),
                "note": self.note}


def unit_of(ladder: Any) -> AssayUnit:
    """The ladder's declared unit, or the fail-closed default. See the module docstring: silence is
    read as a deterministic fixed form, which is the reading that cannot manufacture precision."""
    declared = getattr(ladder, "assay_unit", None)
    if declared is None:
        return AssayUnit.FIXED_FORM
    try:
        return declared if isinstance(declared, AssayUnit) else AssayUnit(str(declared))
    except ValueError:
        return AssayUnit.FIXED_FORM


def population_size_of(ladder: Any, results: Sequence[Any] = ()) -> Optional[int]:
    """The size of the frozen pool the read samples, where there is one.

    The reading is asked first and the ladder second, because the reading is the thing that was
    actually drawn: a ladder reconfigured between a read and its analysis would otherwise supply a
    denominator the read never had.
    """
    for r in results:
        n = getattr(r, "population_size", None)
        if n is not None:
            return int(n)
    fn = getattr(ladder, "population_size", None)
    if callable(fn):
        try:
            n = fn()
        except Exception:
            n = None
        return int(n) if n is not None else None
    return int(fn) if isinstance(fn, int) else None


def finite_population_sd(passes: int, n_items: int, population_size: Optional[int]) -> float:
    """The sampling error of a pass count drawn without replacement from a frozen pool, on the count
    scale of the read itself.

    The draw is hypergeometric, so the variance is the binomial variance times the finite population
    correction (N - n) / (N - 1). Two consequences are the finding, and both are enforced here rather
    than left to a caller. A read of the WHOLE pool has n = N, the correction is zero, and the read
    has no sampling error at all: it is the population. And a read of a subset is not binomial with
    the subset as its own denominator, which is what a whole-pool read was previously reported as.

    A subset in which every item passed does not establish that every item in the pool passes, so the
    variance at either boundary is taken at the continuity-corrected proportion. Reporting zero
    sampling error from a boundary subset would be the same manufactured precision this module exists
    to prevent, arriving through a different door.
    """
    n = int(n_items)
    if n <= 0:
        return 0.0
    N = int(population_size) if population_size is not None else None
    if N is not None and N <= n:
        return 0.0                        # the read was the population
    p = float(passes) / n
    if passes <= 0 or passes >= n:
        p = (float(passes) + 0.5) / (n + 1.0)
    var = n * p * (1.0 - p)
    if N is not None and N > 1:
        var *= float(N - n) / float(N - 1)
    return float(np.sqrt(max(var, 0.0)))


def reading_identity(result: Any) -> Tuple[Any, ...]:
    """Everything about one reading that a second reading of the same artefact could differ in.

    The drawn form is part of the identity and not only the count. Two item forms of thirty items can
    both count eighteen passes and still be two independent draws, so comparing counts alone would
    strip the square root from a ladder that had earned it; and a ladder that draws the same form
    every time shows the same digest, which is the case the comparison exists to catch. A ladder that
    reports no form is compared on its outcome alone, which is the most that can be said about it.
    """
    return (int(getattr(result, "passes", 0)), int(getattr(result, "n_items", 0)),
            bool(getattr(result, "at_ceiling", False)), getattr(result, "form_sha256", None),
            # And which items produced the count, where the reading says. A reading that carries it
            # separates two draws that agreed on a total from one draw counted twice, which the total
            # alone cannot do. Absent on a ladder that reports no outcome digest, and then the
            # comparison rests on the fields above, which is the most that can be said about it.
            getattr(result, "outcome_sha256", None))


def readings_differ(results: Sequence[Any]) -> bool:
    """True when the readings handed in are not all the same reading.

    This is the whole check that finding A4's headline arithmetic is unreachable. A unit whose repeats
    are declared independent earns the square root of the read count only where this is True: 256
    identical readings are one reading counted 256 times, whatever the ladder declares itself to be.
    """
    return len({reading_identity(r) for r in results}) > 1


def resampling_witness_of(ladder: Any, draws: int = 4, seed: int = 20260906) -> Tuple[str, str]:
    """What this ladder says two reads would differ in, asked WITHOUT reading any artefact.

    Returns (status, detail). The point of asking before the run rather than after it is cost: a
    ladder is asked to exhibit its resampling for nothing, where probing it with real reads would
    spend, and a deciding run that cannot show what its repeats resample should stop before it spends
    anything at all.

    A witness is a digest of what one read would draw afresh: for a suite ladder the item form, for a
    simulated ladder the realisation. `absent` means the ladder offers none, `unreadable` means asking
    raised, `constant` means the ladder answered the same thing every time, and `varies` is the only
    answer consistent with a claim that repeats are independent draws.

    The generator is seeded here and is never the run's own, for two reasons. The gate's answer does
    not then depend on when it was asked, so a setup that refuses refuses reproducibly; and asking
    cannot advance the run's stream, so a ladder that was asked and a ladder that was not read the
    same items in the run that follows.
    """
    fn = getattr(ladder, "resampling_witness", None)
    if not callable(fn):
        return ("absent", "the ladder has no resampling_witness")
    rng = np.random.default_rng(seed)
    seen = []
    for _ in range(max(2, int(draws))):
        try:
            w = fn(rng)
        except Exception as exc:                       # a witness that raises has witnessed nothing
            return ("unreadable", "%s: %s" % (type(exc).__name__, exc))
        if w is None:
            return ("absent", "resampling_witness returned nothing")
        seen.append(str(w))
    return ("varies", seen[0]) if len(set(seen)) > 1 else ("constant", seen[0])


def claims_a_reduction(ladder: Any) -> bool:
    """True when this ladder's declared unit would divide its error by the square root of the read
    count. A whole-pool item form is excluded: it claims no reduction, because its reads ARE the
    population and it reports zero sampling error, so it has nothing to exhibit."""
    unit = unit_of(ladder)
    if not unit.repeats_are_independent:
        return False
    if unit is AssayUnit.ITEM_FORM:
        N = population_size_of(ladder)
        n_read = int(getattr(ladder, "n_items", 0) or 0)
        if N is not None and n_read >= N:
            return False
    return True


def unit_refusals(ladder: Any) -> List[str]:
    """Every way this ladder's declared sampling unit fails a deciding run, named at once.

    Three failures, and none is a matter of taste. An undeclared unit means the run has not said
    which population it measures, so the fail-closed default is doing the author's work for them. A
    ladder declaring the artefact-repeat unit has claimed to resample the artefact it was handed,
    which no ladder read does; the repeats it needs are supplied above it. And a ladder that declares
    its repeats are independent draws must be able to SHOW what a repeat would draw afresh, because
    the declaration alone is what a deterministic ladder mislabelling itself as stochastic also has,
    and a run that trusts the label hands that ladder the square root of its read count.

    The showing costs nothing and reads nothing: `resampling_witness` is asked what a read would draw,
    not made to take one. A ladder whose witness is absent, unreadable or constant is refused here,
    before the run spends. A ladder that answers honestly at the gate and then resamples nothing at
    run time is caught by `read_uncertainty`, which compares the readings it is about to average
    rather than the label they were taken under: the two checks are deliberately not the same check.
    """
    out: List[str] = []
    if not bool(getattr(ladder, "assay_unit_declared", False)):
        out.append("assay-unit: this ladder declares no sampling unit, so what a repeated read "
                   "resamples is decided by the fail-closed default rather than by the registration; "
                   "a deciding run states the unit and its target population (see arc_runner.sampling)")
    unit = unit_of(ladder)
    if not unit.supplied_by_a_ladder:
        out.append("assay-unit: this ladder declares %r, whose repeats are repeats of the artefact or "
                   "of the process that produced it. A ladder reads the artefact it is handed and "
                   "cannot resample it; that unit is supplied above the ladder" % unit.value)
    if claims_a_reduction(ladder):
        status, detail = resampling_witness_of(ladder)
        if status != "varies":
            out.append("assay-unit: this ladder declares %r, so its repeats are counted as "
                       "independent draws, but it cannot show what a repeat would draw afresh (%s: "
                       "%s). A ladder that claims the square root of its read count exhibits its "
                       "resampling in resampling_witness, which reads no artefact and spends nothing"
                       % (unit.value, status, detail))
    return out


def read_uncertainty(ladder: Any, results: Sequence[Any]) -> ReadUncertainty:
    """The sampling error of the MEAN of `results`, under the ladder's declared unit.

    The three live branches are the three claims a repeat can honestly make. A fixed form repeats
    itself, so the mean of one read and the mean of 256 carry the same information and the same
    error, and the error is zero conditional on the item set. An item form draws fresh items each
    time, so the repeats are independent and the square root of the count is earned, over a per-read
    error that carries the finite population correction. A stochastic execution redraws its outcome,
    so the same square root is earned over the binomial error the process actually has.

    THE SQUARE ROOT IS EARNED BY THE READINGS AND NOT BY THE LABEL. Both reducing branches ask
    `readings_differ` before dividing. A ladder that declares independent repeats and then returns the
    same reading every time has resampled nothing, whatever it calls itself, and it gets its per-read
    error undivided: identical at 1, at 16 and at 256 reads. Without that check the whole of finding
    A4 was one wrong declaration away from returning, because a deterministic ladder declaring
    STOCHASTIC_EXECUTION was handed the square root exactly as the reference runner had handed it to
    every ladder. The undivided per-read error is the conservative reading and is deliberately not the
    exact zero of a fixed form: a ladder whose declaration its readings refute has not established
    that it is a fixed form either, and the wider of the two readings is the one to report.
    """
    results = list(results)
    reads = max(1, len(results))
    unit = unit_of(ladder)
    subset = int(getattr(results[0], "n_items", 0)) if results else 0
    N = population_size_of(ladder, results)
    # None where there was one reading and therefore nothing to compare it with.
    witnessed = readings_differ(results) if reads > 1 else None
    # The reduction applies to the reads that were shown to be draws. One read divides by one either
    # way, so a single reading keeps its per-read error and makes no claim about repeats.
    draws = reads if (reads == 1 or witnessed) else 1
    refuted = ("; the %d readings were identical in outcome and in the form they drew, so they are "
               "one reading counted %d times and no reduction is taken from them" % (reads, reads))

    if unit is AssayUnit.STOCHASTIC_EXECUTION:
        # Unchanged arithmetic for the unchanged case: the per-read binomial standard deviation, and
        # the square root of the read count, because each read really is another draw. A read count
        # that the readings did not earn is not used, which is the only change from the first repair.
        per = float(np.mean([float(getattr(r, "binomial_sd", 0.0)) for r in results])) if results else 0.0
        sd = per / float(np.sqrt(draws))
        return ReadUncertainty(sd=sd, unit=unit, target=unit.target, reads=reads,
                               independent_draws=draws, exact=False, subset_size=subset,
                               population_size=N, resampling_witnessed=witnessed,
                               note="each read is an independent draw of the scoring process, so the "
                                    "mean of %d reads has the per-read error over the square root of %d"
                                    % (reads, draws) + ("" if draws == reads else refuted))

    if unit is AssayUnit.ITEM_FORM:
        whole_pool = (N is not None and subset >= N)
        per = float(np.mean([finite_population_sd(int(getattr(r, "passes", 0)),
                                                  int(getattr(r, "n_items", 0)), N) for r in results])) \
            if results else 0.0
        if whole_pool:
            return ReadUncertainty(sd=0.0, unit=unit, target=unit.target, reads=reads,
                                   independent_draws=1, exact=True, subset_size=subset,
                                   population_size=N, resampling_witnessed=witnessed,
                                   note="the read was the whole frozen pool, so it has no sampling "
                                        "error against that pool and repeating it draws the same "
                                        "items again; %d reads carry what one read carries" % reads)
        sd = per / float(np.sqrt(draws))
        return ReadUncertainty(sd=sd, unit=unit, target=unit.target, reads=reads,
                               independent_draws=draws, exact=False, subset_size=subset,
                               population_size=N, resampling_witnessed=witnessed,
                               note="each read drew a fresh item form of %d from a frozen pool of %s, "
                                    "so the per-read error carries the finite population correction "
                                    "and %d of the %d reads count as independent draws"
                                    % (subset, "an undeclared size" if N is None else str(N),
                                       draws, reads) + ("" if draws == reads else refuted))

    if unit is AssayUnit.ARTEFACT_REPEAT:
        # A refusal belongs at the gate and not here, because this function is also called on a
        # demonstration path that has no gate. What it must not do is invent one of the other models.
        return ReadUncertainty(sd=float("nan"), unit=unit, target=unit.target, reads=reads,
                               independent_draws=1, exact=False, subset_size=subset,
                               population_size=N, resampling_witnessed=witnessed,
                               note="this ladder declares a unit whose repeats are repeats of the "
                                    "artefact, which a ladder read does not perform; no read-level "
                                    "sampling error is defined and none is invented")

    return ReadUncertainty(sd=0.0, unit=unit, target=unit.target, reads=reads, independent_draws=1,
                           exact=True, subset_size=subset, population_size=N,
                           resampling_witnessed=witnessed,
                           note="the score is deterministic in the artefact and the item set, so it "
                                "is exact conditional on them and repeating it resamples nothing; "
                                "%d reads carry exactly what one read carries" % reads)


def read_model_record(ladder: Any) -> Dict[str, Any]:
    """What the run records once about how its readings are to be read, for the manifest and the
    evidence bundle. A bundle that carries pass counts without the sampling unit they were counted
    under cannot be re-analysed, because the counts do not say what they are a sample of."""
    unit = unit_of(ladder)
    # What the ladder could show about its own resampling, recorded beside the claim rather than only
    # checked at the gate: a bundle whose reader can see `constant` here can see why the run was
    # refused, or, on a path with no gate, what the run's reduction was resting on.
    status, detail = resampling_witness_of(ladder) if claims_a_reduction(ladder) \
        else ("not-claimed", "this unit takes no reduction from repeats, so it exhibits none")
    return {"assay_unit": unit.value, "declared": bool(getattr(ladder, "assay_unit_declared", False)),
            "resamples": unit.resamples, "target_population": unit.target,
            "repeats_are_independent": bool(unit.repeats_are_independent),
            "population_size": population_size_of(ladder),
            "resampling_witness": status, "resampling_witness_detail": detail,
            "refusals_on_the_deciding_path": unit_refusals(ladder)}
