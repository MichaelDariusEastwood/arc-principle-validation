"""The frozen hidden capability ladder, and the three conditions ruling 29 attaches to it.

Capability is the artefact's verified pass count on a frozen ladder of checkable items. The ladder is
hashed, the hash goes into the seal, and the loop never sees the items: the loop sees a task and never
the ladder. Headroom is checked, and a system that reaches the top rung before the final checkpoint is
reported NOT EVALUABLE for that reason rather than scored at a ceiling. The ladder's own sampling
precision is exposed so that the errors-in-variables correction can use it, and where a reading is
exact there is no precision to expose and no attenuation to correct.

Two ladders ship. MockLadder measures a mock artefact's latent capability with binomial noise, which is
exactly the measurement error a real ladder has. CheckableLadder runs verifier functions over a real
artefact's text, one per item, and counts passes. A verifier is any callable returning True or False;
the reference set is arithmetic and string tasks with exact answers, and a real experiment replaces it
with a domain ladder registered and hashed before any run.

WHAT A REPEATED READING BUYS, AND WHAT IT DOES NOT (finding A4). Every ladder declares the sampling
unit its repeats draw from, and the declaration decides the arithmetic in `read_mean`. A ladder that
applies fixed verifiers to a fixed item set is exact conditional on that set: repeating it resamples
nothing, and it reports zero sampling error rather than a binomial standard deviation divided by the
square root of a repeat count that bought no information. A ladder that draws a fresh item form from
a larger frozen pool reports the finite-population sampling error of that form against the pool, so a
whole-pool read reports zero and a subset read is not binomial with the subset as its own
denominator. A ladder that redraws a stochastic outcome keeps the square root, which it earns.
Silence fails closed to the deterministic reading. See arc_runner.sampling.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .custody import attest_exact_check, mark_substring_smoke_test
from .sampling import AssayUnit, ReadUncertainty, finite_population_sd, read_uncertainty, unit_of


def _length_delimited(parts: Sequence[str]) -> bytes:
    """The same length-delimited encoding `arc_runner.custody.code_identity` uses, and for the same
    reason: without the lengths the concatenation is ambiguous, so two different sequences of
    identifiers can produce identical bytes and a substitution can be concealed."""
    h = bytearray()
    for part in parts:
        pb = str(part).encode("utf-8")
        h += str(len(pb)).encode() + b":" + pb
    return bytes(h)


def outcome_digest(item_ids: Sequence[str], passed_ids: Any) -> str:
    """What each drawn item DID, as one value.

    `form_sha256` already says which items a read drew, which finding A4 put there. It says nothing
    about what any of them did, so a read log carrying it still cannot be checked against a response:
    two artefacts read on the same form give the same form digest and different answers. This digest
    carries the identifier beside its outcome, in the DRAW ORDER, so a permuted pass vector is a
    different value rather than the same one, and a count of eighteen passes from one set of items is
    distinguishable from a count of eighteen from another.
    """
    ids = [str(i) for i in item_ids]
    passed = {str(i) for i in (passed_ids or ())}
    return hashlib.sha256(b"arc-read-outcome/1" + _length_delimited(
        ["%s=%d" % (i, 1 if i in passed else 0) for i in ids])).hexdigest()


def artefact_digest(artefact: Optional[Dict[str, Any]]) -> Optional[str]:
    """A hash of the artefact that was read, so a count can be tied to the thing it was a count of.

    The artefact's own text is the system's output and is held in the checkpoint store; the digest is
    what travels in the public bundle, which is the same boundary the read digests keep. Anything that
    will not serialise is hashed by its repr rather than stopping the record: a missing digest is a
    read that cannot be traced, which is the defect being repaired.
    """
    if artefact is None:
        return None
    try:
        payload = json.dumps(artefact, sort_keys=True, default=repr).encode("utf-8")
    except (TypeError, ValueError):
        payload = repr(artefact).encode("utf-8")
    return hashlib.sha256(b"arc-artefact/1" + payload).hexdigest()


@dataclass
class LadderResult:
    passes: float
    n_items: int
    at_ceiling: bool
    # The frozen pool this read was drawn from, where the read was a draw from one. It travels with
    # the result because the finite population correction needs the denominator the read actually
    # had, and a ladder reconfigured between a read and its analysis would otherwise supply another.
    population_size: Optional[int] = None
    # WHAT EACH DRAWN ITEM DID (finding A8). The read log recorded a subset size and a pass count and
    # nothing tying either to a reading of anything: an analyst could re-count the saved numbers and
    # re-fit them, and could not check that any count came from a response. `form_sha256` below says
    # which items were drawn; this says what they did. Together they bind the count to a specific
    # draw against a specific artefact, so somebody holding the hidden suite recomputes both and
    # compares. It is a DIGEST and not the identifiers with their outcomes, because the bundle is
    # public and the suite is held out: a digest lets the holder verify and tells a reader who does
    # not hold it nothing about which items exist. A ladder that reads no item set leaves it None,
    # which is a statement and not an omission.
    outcome_sha256: Optional[str] = None
    # WHAT THIS READ DREW, where the read drew anything (finding A4). A digest of the item form, so
    # that two readings can be compared on the form as well as on the count. Two different item forms
    # counting the same passes are still two independent draws and must keep the square root they
    # earned; a ladder that draws one form and reads it again has drawn nothing new, and the count
    # alone cannot tell those apart. None where the read draws no form.
    form_sha256: Optional[str] = None

    @property
    def score(self) -> float:
        return float(self.passes)

    @property
    def binomial_sd(self) -> float:
        """The binomial standard deviation of this count, which is the sampling error ONLY when the
        read is a draw with replacement from an unbounded population. A read of a subset of a frozen
        pool is a draw without replacement and its error is smaller; a read of the whole pool has no
        sampling error at all. Use `sampling_sd`, which applies the finite population correction, or
        `arc_runner.sampling.read_uncertainty`, which applies the whole declared model."""
        p = max(min(self.passes / self.n_items, 1 - 1e-9), 1e-9)
        return float(np.sqrt(self.n_items * p * (1 - p)))

    @property
    def sampling_sd(self) -> float:
        """This read's sampling error against the pool it was drawn from: zero when it was the pool."""
        return finite_population_sd(self.passes, self.n_items, self.population_size)


class Ladder:
    # Two declarations every ladder carries, because a confirmatory run has to be able to ask. A
    # ladder that reads a simulated latent capability is not reading a system, and a ladder built on
    # a pool nobody wrote for the study is a smoke test. Finding A9: the reference runner said this
    # in a line of stderr text, which a manifest cannot record and a gate cannot read.
    simulated = False
    smoke_only = False
    # WHAT A REPEAT OF A READING RESAMPLES (finding A4). Silence fails closed to the deterministic
    # fixed form, which is the reading under which repeats buy nothing, so a ladder that has not
    # thought about its sampling unit cannot manufacture precision by omitting to declare one. A
    # ladder that HAS settled its unit sets both attributes, and a confirmatory run refuses a ladder
    # that has not: which population a deciding run measures is stated, never defaulted.
    assay_unit: AssayUnit = AssayUnit.FIXED_FORM
    assay_unit_declared = False
    # WHETHER THE BEFORE AND AFTER READS OF ONE CELL SHARE THEIR ERROR (finding A5). P5 subtracts one
    # reading from another, so how much of the two errors is common decides how much of it survives
    # the subtraction. Every ladder here draws independently between reads (a fresh item form, a fresh
    # stochastic realisation, or an exact deterministic score with no error at all), so zero is right
    # as well as being the fail-closed default: it leaves the whole after-read variance in the
    # increment and gives the wider interval. A ladder that draws ONE item form and reads the artefact
    # on it both before and after declares the correlation its shared item difficulty implies.
    read_error_correlation: float = 0.0

    def __init__(self, n_items: int, spec: Dict[str, Any]):
        if isinstance(n_items, bool) or not isinstance(n_items, (int, np.integer)) or int(n_items) <= 0:
            raise ValueError(f"n_items must be a positive integer, got {n_items!r}")
        self.n_items = int(n_items)
        # The sampling unit goes INTO the spec, and therefore into the ladder hash and the seal. Which
        # population the run measures and what its repeats resample is a pre-run commitment in exactly
        # the way the item set is: a run scored under one sampling model having sealed another is a
        # custody failure and not a silent change of estimand.
        spec = dict(spec)
        # `unit_of` and not `self.assay_unit.value`, so that a subclass declaring its unit as the
        # plain string the enum carries is recorded as that unit rather than raising here.
        spec["assay_unit"] = unit_of(self).value
        spec["assay_unit_declared"] = bool(self.assay_unit_declared)
        # Sealed for the same reason: the read error correlation is part of the observation model P5
        # estimates under, so a run scored under one and sealed under another has changed its
        # estimand rather than its arithmetic. See arc_runner.p5_observation.
        spec["read_error_correlation"] = float(getattr(self, "read_error_correlation", 0.0) or 0.0)
        self.spec = spec
        self.sha256 = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        # Off unless a run asks for it. Finding A8: the saved outputs kept the fitted summaries and
        # threw away the readings they were fitted from, so nobody could recount a pass rate. A run
        # that is saving evidence turns this on with arc_runner.custody.attach_read_log; a run that
        # is not pays nothing, and a ladder reused across runs never accumulates another run's reads.
        self.read_log: Optional[List[Dict[str, Any]]] = None

    def score(self, artefact: Dict[str, Any], rng: np.random.Generator) -> LadderResult:
        raise NotImplementedError

    def resampling_witness(self, rng: np.random.Generator) -> Optional[str]:
        """A digest of what ONE read would draw afresh, produced without reading any artefact.

        WHY A LADDER IS ASKED TO SHOW THIS (finding A4). A ladder that declares its repeats are
        independent draws is handed the square root of its read count, and the declaration alone is
        also what a deterministic ladder mislabelling itself as stochastic has. Asking it to exhibit
        two different draws costs nothing, spends nothing and reads nothing, and a ladder that cannot
        is refused before the run begins by `arc_runner.sampling.unit_refusals`.

        None is the honest answer for a fixed form, which draws nothing: it takes no reduction from
        repeats and is never asked. A ladder whose reads ARE draws overrides this and returns the
        identity of the draw, so that two calls differ exactly when two reads would.
        """
        return None

    def population_size(self) -> Optional[int]:
        """The frozen pool a read samples from, where the read is a sample. None where it is not:
        the simulated ladder draws from a process and not from a pool, and a fixed form IS its own
        item set. A ladder that returns a size is promising that a read of that many items is the
        whole population and therefore has no sampling error."""
        return None

    def record_read(self, result: LadderResult, context: Optional[str] = None,
                    artefact: Optional[Dict[str, Any]] = None) -> None:
        """One line of evidence per reading: the subset actually drawn and the passes counted in it.
        The subset size is taken from the result rather than from the ladder, because a whole-pool
        read and a subset read are both readings and their denominators differ.

        AND WHAT THE COUNT CAME FROM (finding A8). A subset size and a pass count can be re-counted
        and re-fitted and cannot be traced to any reading of anything, so the line also carries three
        digests: the item form that was drawn, the outcome each of those items produced, and the
        artefact that was read. A holder of the hidden suite and of the checkpoint store recomputes
        all three and finds out whether the count came from a response; a reader who holds neither
        learns nothing from them, which is what keeps the bundle public. `artefact` is optional
        because a caller that has none must still be able to record a read, and its absence is
        written as absence.
        """
        log = getattr(self, "read_log", None)
        if log is None:
            return
        log.append({"context": context, "subset_size": int(result.n_items), "passes": int(result.passes),
                    "at_ceiling": bool(result.at_ceiling),
                    # The denominator and the sampling unit travel with the count, because a pass
                    # count with no statement of what it is a sample of cannot be re-analysed.
                    "population_size": (int(result.population_size)
                                        if result.population_size is not None else None),
                    "assay_unit": unit_of(self).value,
                    "form_sha256": result.form_sha256,
                    "outcome_sha256": result.outcome_sha256,
                    "artefact_sha256": artefact_digest(artefact)})

    def headroom_ok(self, result: LadderResult) -> bool:
        return not result.at_ceiling


class MockLadder(Ladder):
    """Measures a latent capability with binomial noise; scale maps capability to pass probability."""

    simulated = True
    # Every read redraws the binomial outcome, so a repeat here IS a fresh independent draw and the
    # square root of the read count is earned. Finding A4 names this ladder as the reason the defect
    # was invisible: the mock's repeats really did carry new information, and the same arithmetic was
    # applied to a deterministic ladder whose repeats carried none.
    assay_unit = AssayUnit.STOCHASTIC_EXECUTION
    assay_unit_declared = True

    def __init__(self, n_items: int = 2000, scale: float = 400.0):
        # scale is the capability at which every item passes. The first end-to-end run used 100 and
        # every held-out system reached the top rung by depth 32, so the runner returned NOT EVALUABLE
        # for all of them under ruling 29's headroom rule, which is the rule doing its job. A ladder
        # is registered with headroom for the depth it will be read at, and this default has it to 128.
        super().__init__(n_items, {"kind": "mock", "n_items": n_items, "scale": scale})
        self.scale = float(scale)

    def score(self, artefact, rng):
        p = min(float(artefact["capability"]) / self.scale, 1.0)
        passes = int(rng.binomial(self.n_items, p))
        return LadderResult(passes=passes, n_items=self.n_items, at_ceiling=(passes >= self.n_items))

    def resampling_witness(self, rng):
        """What a repeat of a read here draws afresh: the binomial realisation, and nothing else. The
        witness draws one at the mid-point pass probability, so two calls differ exactly as two reads
        of one artefact do, and it touches no artefact and costs nothing."""
        return "mock-binomial:%d" % int(rng.binomial(self.n_items, 0.5))


def read_with_uncertainty(ladder: Ladder, artefact: Dict[str, Any], rng: np.random.Generator,
                          reads: int = 1, context: Optional[str] = None
                          ) -> Tuple[float, ReadUncertainty, bool]:
    """Repeated readings of one artefact, averaged, WITH the statement of what the repeats resampled.

    A single reading's noise can be larger than a one-round increment, which is the two per cent
    precision problem the P5 registration names, and repeated reads are how a run with a stochastic or a
    subsetting ladder reaches it. What repeated reads cannot do is make a deterministic score more
    exact than it already was, which is why the uncertainty comes from the ladder's declared sampling
    unit and never from the read count alone. See arc_runner.sampling.

    `context` names where in the run the reading was taken, and is recorded with it when the ladder
    is keeping a read log. A pass count with no idea which cell produced it cannot be re-counted."""
    rs = []
    for _ in range(max(1, int(reads))):
        r = ladder.score(artefact, rng)
        ladder.record_read(r, context, artefact)
        rs.append(r)
    mean = float(np.mean([r.score for r in rs]))
    return mean, read_uncertainty(ladder, rs), any(r.at_ceiling for r in rs)


def read_mean(ladder: Ladder, artefact: Dict[str, Any], rng: np.random.Generator, reads: int = 1,
              context: Optional[str] = None):
    """`read_with_uncertainty` with the uncertainty reduced to its standard deviation, for the callers
    that consume a number. The number is the sampling error of the mean under the ladder's declared
    unit, which for a deterministic fixed form and for a whole-pool read is zero."""
    mean, unc, ceiling = read_with_uncertainty(ladder, artefact, rng, reads, context)
    return mean, unc.sd, ceiling


class CheckableLadder(Ladder):
    """A real ladder: frozen items, each a verifier over the artefact's text.

    THIS LADDER IS DETERMINISTIC AND SAYS SO. It applies fixed verifiers to a fixed item set and never
    touches the generator it is handed, so its score is exact conditional on that set and reading it
    again returns the same number. Finding A4: the previous reading model divided a binomial standard
    deviation by the square root of the read count here, which reported a precision that no repeat had
    bought. The declared unit now makes the repeats report what they are, being repeats.
    """

    assay_unit = AssayUnit.FIXED_FORM
    assay_unit_declared = True

    def __init__(self, items: Sequence[Dict[str, Any]], verifiers: Dict[str, Callable[[str, Dict[str, Any]], bool]]):
        super().__init__(len(items), {"kind": "checkable", "items": [{k: v for k, v in it.items() if k != "answer"} for it in items],
                                       "answer_sha256": [hashlib.sha256(str(it.get("answer", "")).encode()).hexdigest() for it in items]})
        self.items = list(items)
        self.verifiers = verifiers

    def population_size(self) -> Optional[int]:
        """The item set is the population: it is read whole, every time, by construction."""
        return self.n_items

    def form_digest(self) -> str:
        """This ladder's item form, which is the same form on every read. It travels on the result so
        that a reading carries the identity of what it read and not only the count: a subclass of this
        ladder that declares its repeats are fresh draws is contradicted by two readings of the same
        form, and `arc_runner.sampling.readings_differ` is what sees the contradiction."""
        h = hashlib.sha256(b"arc-item-form/1")
        for it in self.items:
            b = str(it.get("id", "")).encode("utf-8")
            h.update(str(len(b)).encode()); h.update(b":"); h.update(b)
        return h.hexdigest()

    def score(self, artefact, rng):
        text = artefact.get("text", "")
        passed = [str(it.get("id", "")) for it in self.items
                  if self.verifiers[it["verifier"]](text, it)]
        passes = len(passed)
        return LadderResult(passes=passes, n_items=self.n_items, at_ceiling=(passes >= self.n_items),
                            population_size=self.n_items, form_sha256=self.form_digest(),
                            # Which items passed, as a digest, so the count can be checked against a
                            # response by somebody holding the item set. Finding A8.
                            outcome_sha256=outcome_digest(
                                [str(it.get("id", "")) for it in self.items], passed))


# --------------------------------------------------------------------------------------------------
# The reference verifiers, and the difference between checking an answer and finding one in the text
# --------------------------------------------------------------------------------------------------

ANSWER_LINE = re.compile(r"^\s*([A-Za-z0-9_.:-]+)\s*[:=]\s*(\S+)\s*$")


def exact_answer(text: str, item: Dict[str, Any]) -> bool:
    """The item is solved when the response carries exactly one answer line labelled with the item's
    own identifier, and the value on that line is exactly the answer.

    WHY THIS IS NOT A SUBSTRING TEST (finding A4). The verifier this replaces asked whether the
    answer appeared anywhere in the response, which a response enumerating candidates satisfies
    without solving anything: a page of every integer to a thousand passes every item of an addition
    ladder, and a hedged response listing three possible answers passes on the one that happens to be
    right. Both are scored as capability by a substring test and neither solved the intended task.

    Two answer lines for one item fail the item rather than passing on the better one, because a
    response that gave two answers did not give this one.
    """
    wanted = str(item.get("answer", ""))
    ident = str(item.get("id", ""))
    found = []
    for line in str(text).splitlines():
        m = ANSWER_LINE.match(line)
        if m and m.group(1) == ident:
            found.append(m.group(2))
    return len(found) == 1 and found[0] == wanted


def contains_answer(text: str, item: Dict[str, Any]) -> bool:
    """The substring check, KEPT AND FLAGGED rather than deleted, because a wiring smoke test is a
    legitimate use of it and removing it would only push somebody to rewrite it unmarked.

    It is not a measure of whether a task was solved: see `exact_answer` for what it misses. The
    marker below is read by arc_runner.mode, which refuses any ladder carrying a verifier so marked
    on the deciding path, whatever pool the verifier was attached to.
    """
    return str(item.get("answer", "")) in str(text)


mark_substring_smoke_test(contains_answer)     # arc_runner.mode refuses it, and refuses it wrapped
# And the check that IS a measurement says so, because the deciding path asks a check to attest that
# it decides whether an item was solved rather than asking it to confess that it does not. An
# unattested callable, which is what wrapping any check in a lambda produces, is refused.
attest_exact_check(exact_answer)


def reference_checkable_ladder(n: int = 60, seed: int = 20260905) -> CheckableLadder:
    """A small arithmetic ladder with exact answers, for smoke tests of the real path. A registered
    experiment replaces this with its domain ladder; this one exists so the real path has a test.

    It is scored by `exact_answer` and not by a substring test, so a response listing candidates
    passes nothing. The prompt says how an answer is to be given, because a scoring rule the system
    was never told is a measure of guessing the rule.
    """
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        # The upper bound rises a decade every twenty items, so the ladder has rungs. It starts at
        # two decades and not at one: `integers(10, 10)` is an empty range, and the previous bound of
        # 10 ** (1 + i // 20) made every one of the first twenty items raise rather than exist, which
        # is how this function came to be shipped having never once been called.
        hi = 10 ** (2 + i // 20)
        a, b = int(rng.integers(10, hi)), int(rng.integers(10, hi))
        ident = "arith-%03d" % i
        items.append({"id": ident, "verifier": "exact_answer",
                      "prompt": "%s: what is %d + %d? Answer on one line as `%s: <value>`."
                                % (ident, a, b, ident),
                      "answer": str(a + b)})
    lad = CheckableLadder(items, {"exact_answer": exact_answer})
    lad.smoke_only = True          # a confirmatory run refuses it; see arc_runner.mode
    return lad
