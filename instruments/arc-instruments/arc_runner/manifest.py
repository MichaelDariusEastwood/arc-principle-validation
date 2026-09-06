"""Run manifests, the seal, the execution mode, and what each mode may be read as.

Ruling 27: seal, then generate. The seal covers the predictions, the analysis code by hash and the
ladder by hash, and it is written before any held-out continuation exists. Ruling 28: a pilot is never
scored. A manifest flagged as a pilot makes `require_reportable` and `require_scoreable` raise, and
every verdict path in this package calls one of them first. The anchor (an external timestamp on the
seal's hash) is the operator's act and is recorded here by its identifier when it exists; this code
never performs it.

THE MODE IS PART OF THE MANIFEST, NOT PART OF THE COMMENTARY. Finding A9: a manifest that does not
say which kind of run produced it cannot stop a simulated recovery being read later as a measurement,
and the reference runner distinguished a rehearsal from a deciding run by a line of stderr text.
Every manifest now carries `execution_mode`, whether the run was `simulated`, whether it is
`scoreable_at_proposition_level`, and the sentence a reader sees at the top of any summary.

TWO GATES, BECAUSE THERE ARE TWO QUESTIONS. `require_reportable` asks whether a result may be printed
at all with its mode stated: a demonstration may, a pilot may not. `require_scoreable` asks the
question that decides a proposition: only a confirmatory run passes it. A demonstration keeps its
verdicts, labelled as the simulated recoveries they are, and is refused the moment anyone asks to
score them.

AND THE THIRD QUESTION, WHICH THIS FILE USED TO SKIP. Finding A8: `require_scoreable` checked a flag
and the existence of a seal, so it passed a run whose predictions had been edited after sealing, a
run whose configuration had moved, and a run whose commitment nothing outside this code had ever
attested. It now goes through `arc_runner.custody`: a deciding run needs an anchor receipt that is
not a mock, the receipt must still attest the seal record's recomputed digest, the specification hash
must still recompute to the sealed value from the manifest's own fields, and the predictions and the
configuration handed in at scoring time must be the ones that were sealed. Every failure is named at
once rather than the first one found.

AND THE PART OF THAT QUESTION NO HASH CAN ANSWER. The finding names a third requirement beside those
two: that the material deciding the predictions was not already inspected when they were chosen. A
hash proves the material has not moved since it was hashed and says nothing about who had read it
before, so the seal carries a named party's statement with the digest of the material it speaks
about, written before the anchor and therefore inside the anchored digest. A deciding run whose seal
carries only the runner's placeholder is refused at proposition level, as it is for a mock receipt.

AND THE QUESTION CUSTODY DOES NOT ASK. Custody settles whether a record is the record that was
sealed and whether somebody outside this code attested it. It says nothing about what was measured,
so a simulated system read through a checked ladder, carrying a genuine receipt and a named
attestation, satisfies every part of it. This file once answered the other question by refusing
confirmatory collection outright, on the ground that no real assay is released here; that refusal
also barred the case the four modes exist to serve, so `require_released_apparatus` keeps the part
of it that is true of everything this package can point at. Every system shipped here declares
itself a simulation, and a deciding run built out of one, or out of a system that declared nothing,
is refused at proposition level as well as at the gate before the first paid call.

AND THE MODE IS INSIDE THE COMMITMENT, NOT BESIDE IT. The specification hash once covered the ladder,
the code, the configuration and the ladder identity, so `execution_mode` and everything travelling
with it sat outside both the seal and the anchor. A demonstration against the simulated system could
therefore take a genuine anchor receipt, have two fields of its own bundle rewritten afterwards, and
pass every custody check as a deciding run. `custody.spec_hash_of` now covers the experiment, the
mode, the pilot flag, the scoreability flag, the mode label, the adapter, the confirmatory inputs and
the environment, all of which `new_manifest` writes before `seal_predictions` runs, so a rehearsal
relabelled a deciding run refuses instead of scoring.

The code identity in a manifest is portable: relative paths, length-delimited entries, the ladder's
verifier implementations bound beside the pool, and the dependency manifest naming every third-party
distribution the packages import with its version. The same bytes checked out in another directory
produce the same manifest identity, which is what lets somebody else recompute it; the dependency
manifest is what lets them tell whether they recomputed it with the same numerical libraries.

A LOCAL SEAL DETECTS CHANGED CONTENT AND NOTHING MORE. Its clock is an assertion by the party being
checked, so the seal note says what the seal is (a content commitment) rather than claiming a
precedence the clock cannot support, and the independent part is the anchor receipt.

THE STRICT SERIALISATION, AND WHY A MANIFEST NEEDS ONE. A record written through a JSON writer that
accepts NaN is not JSON, so the one artefact an independent analyst must be able to read back is the
one this package could write in a dialect nothing else parses. `canonical_bytes` refuses non-standard
numbers and `normalise` gives every unavailable value an explicit type instead, so a missing figure
travels as a named absence rather than as a token a reader may silently turn into a number.

TWO VOCABULARIES FOR THE MODE, ONE SOURCE. `execution_mode` carries the four words this package
decides on. `mode` carries the shorter word an evidence bundle and its replay read, and it is derived
from `execution_mode` rather than written beside it, so the two cannot be edited apart: the integrity
check recomputes the derived fields and refuses a manifest whose compatibility word, whose simulated
flag or whose eligibility flag disagrees with the mode that was sealed. That is what stops a record
promoting its own eligibility, which no run may do to itself.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from typing import Any, Dict, Optional

from arc_instruments import sealing

from . import custody
from .mode import ExecutionMode, resolve as resolve_mode


class NotScoreable(RuntimeError):
    """The run may exist and may be reported; it may not be read at proposition level."""


class PilotNotScoreable(NotScoreable):
    """Ruling 28, kept as its own type because callers have caught it by name since the first run."""


class InstrumentNotReleased(RuntimeError):
    """A collection this package will not perform, whatever the mode says.

    It is not a verdict and it is not a refusal of the mode. It is the statement that the apparatus a
    real collection would need is not released: a P5 bank has no released assay or negative-control
    implementation, and a P16 titration has no released service and burden measurements. A run that
    hands either of them a remote endpoint is refused at the library boundary rather than at the
    command line, because the boundary is the place a second caller cannot walk around.
    """


MANIFEST_SCHEMA = "arc-run/2"

# The compatibility words, and the four-mode enum each is derived from. The short word is what an
# evidence bundle and its replay read; the enum is what this package decides on. Deriving one from
# the other is what keeps them from being edited apart.
MODES = ("demo", "smoke", "pilot", "confirmatory")
_MODE_WORDS = {ExecutionMode.DEMONSTRATION: "demo", ExecutionMode.SMOKE: "smoke",
               ExecutionMode.PILOT: "pilot", ExecutionMode.CONFIRMATORY: "confirmatory"}


def mode_word(m: ExecutionMode) -> str:
    """The short word for a mode. Every mode has one; silence is never a word."""
    return _MODE_WORDS[m]


def check_mode(mode: str, pilot: bool = False) -> str:
    """The compatibility word, checked against the pilot flag it must agree with."""
    if mode not in MODES:
        raise ValueError("execution mode must be one of %s" % ", ".join(MODES))
    if bool(pilot) != (mode == "pilot"):
        raise ValueError("pilot flag and execution mode disagree")
    return mode


def normalise(value):
    """Strict JSON representation; unavailable numerical values remain explicitly typed."""
    if isinstance(value, dict):
        return {str(k): normalise(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalise(v) for v in value]
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return {"$arc_nonfinite": "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")}
    return value


def denormalise(value):
    """The inverse of `normalise`, so a kept payload can be checked against the seal it was taken
    under. The seal is over the payload as it was handed in; the manifest keeps the strict form so
    that the record is readable, and the check reverses the one before comparing with the other."""
    if isinstance(value, dict):
        if set(value) == {"$arc_nonfinite"}:
            return {"nan": float("nan"), "+inf": float("inf"), "-inf": -float("inf")}[value["$arc_nonfinite"]]
        return {k: denormalise(v) for k, v in value.items()}
    if isinstance(value, list):
        return [denormalise(v) for v in value]
    return value


def canonical_bytes(value) -> bytes:
    return json.dumps(normalise(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def sha256_of(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def code_sha256(paths, root: Optional[str] = None) -> str:
    """The portable identity of these source files. The old version fed absolute paths into the
    digest, so the same bytes in another directory were a different identity and no independent
    analyst could recompute it; `custody.code_identity` hashes relative paths with length-delimited
    entries instead. `root` defaults to the common parent of the paths given, which is what makes a
    relocated but unchanged tree hash to the same value."""
    return custody.code_identity(paths, root)


def package_code_paths(root: Optional[str] = None):
    return custody.package_code_paths(root)


def new_manifest(experiment: str, pilot: bool, ladder_sha256: str, config: Dict[str, Any],
                 adapter_name: str, mode: Any = None,
                 confirmatory_inputs: Optional[Dict[str, Any]] = None,
                 ladder: Any = None) -> Dict[str, Any]:
    """`mode` is any ExecutionMode or its word; None resolves fail closed to a demonstration, or to a
    pilot when the pilot flag is set. The two are never allowed to disagree.

    `ladder` is the ladder object where the caller holds one. It is recorded as an identity, being
    the spec plus a hash of every verifier implementation, because a pool hash says which items were
    asked and says nothing about what counted as passing them. A caller that has only the hash passes
    the string, and the identity records that the verifiers were never bound rather than pretending
    they were.

    AND THE TWO LADDER FIELDS CANNOT BE WRITTEN DISAGREEING. While every identity was derived from
    `ladder_sha256` itself the two agreed by construction. Now that a caller holding the object
    passes both, one manifest could name one pool in the field a reader reads first and another in
    the identity the verifier binding hangs off, with both inside the sealed specification hash, so
    the seal would preserve the contradiction instead of catching it. A run may hand over a hash, or
    an object, or both when they are the same pool; it may not hand over two different pools and ask
    this record to say which was read.
    """
    m = resolve_mode(mode, pilot)
    identity = custody.ladder_identity(ladder if ladder is not None else ladder_sha256)
    if ladder is not None and identity.get("ladder_sha256") and ladder_sha256 \
            and identity["ladder_sha256"] != ladder_sha256:
        raise custody.CustodyRefusal(
            "this run was told its ladder hashes to %s and was handed a ladder that hashes to %s; "
            "one manifest cannot record both, and nothing here may choose between them"
            % (str(ladder_sha256)[:16], str(identity["ladder_sha256"])[:16]), ("ladder-identity",))
    return {
        # The schema, so that a reader of a saved record knows which shape it is being asked to read
        # and a record of an older shape is refused rather than half-understood.
        "schema": MANIFEST_SCHEMA,
        "experiment": experiment,
        "pilot": bool(pilot or m is ExecutionMode.PILOT),
        "execution_mode": m.value,
        "simulated": m.simulated,
        "scoreable_at_proposition_level": m.scoreable,
        # THE SHORT WORD, DERIVED AND NEVER WRITTEN BESIDE. An evidence bundle and its replay read
        # `mode`; this package decides on `execution_mode`. Both come from one enum here, and
        # `require_integrity` recomputes them, so a record cannot carry a demonstration's mode and a
        # deciding run's word.
        "mode": mode_word(m),
        # AND A RECORD MAY NOT PROMOTE ITS OWN ELIGIBILITY. The flag is the mode's own scoreability
        # and is checked against it, so a manifest edited to claim empirical eligibility without
        # changing the mode that was sealed refuses instead of scoring.
        "empirical_confirmatory_eligible": bool(m.scoreable),
        "mode_label": m.label,
        "confirmatory_inputs": confirmatory_inputs,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ladder_sha256": ladder_sha256,
        "ladder_identity": identity,
        "code_sha256": custody.package_code_identity(),
        "code_identity_schema": custody.CODE_IDENTITY_SCHEMA,
        # THE DEPENDENCY MANIFEST, WHICH IS THE OTHER HALF OF THE IDENTITY. Finding A8's action asks
        # for relative-path, length-delimited entries AND a manifest of all required dependencies. The
        # package hash says which analysis script ran; it does not say which scipy did the fitting or
        # which numpy produced the resampling stream, and an analyst regenerating the results under
        # unstated versions of those has regenerated them under an unstated method. It is written
        # before the seal, so it is inside the sealed specification hash.
        "environment": custody.environment_manifest(),
        "adapter": adapter_name,
        "config": config,
        "seal": None,
        "anchor_identifier": None,
    }


def seal_predictions(manifest: Dict[str, Any], predictions: Dict[str, Any], sealed_by: str,
                     anchor: Optional[Any] = None,
                     attestation: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Seal the predictions into the manifest, attest what was unseen, then have the seal anchored.

    Must happen before any continuation is generated; callers assert that by construction. The seal's
    own timestamp is not the evidence, which is the correction finding A8 asks for: a local clock is
    an assertion by the party being checked. `anchor` is the operator's anchoring service, a callable
    handed the seal record's digest and returning a receipt; an absent service attaches a receipt
    labelled a mock, which `require_scoreable` then refuses at proposition level.

    `attestation` is the named party's statement that the material deciding these predictions was
    unseen when they were fixed, being the third thing the finding names and the one no hash can
    supply. It is written on the seal BEFORE the anchor, so that the anchored digest covers it: an
    attestation added or edited afterwards makes the recomputed digest disagree with the receipt. An
    absent one attaches the placeholder, which is labelled and is refused on the deciding path
    exactly as a mock receipt is.
    """
    # THE PAYLOAD IS KEPT, NOT ONLY ITS HASH. A seal holding a digest and nothing else can be
    # checked by nobody: the reveal has to come from somewhere, and where it came from the manifest
    # was the analysis's own live object. It is kept in the strict serialisation so that the record
    # stays readable JSON, and the check below reverses that before comparing with the seal, which
    # is over the payload as it was handed in.
    manifest["sealed_predictions"] = normalise(copy.deepcopy(predictions))
    spec_sha = custody.spec_hash_of(manifest)
    manifest["seal"] = sealing.seal(predictions, spec_sha, sealed_by,
                                    note="local content commitment; independent temporal precedence "
                                         "is not established by this clock")
    manifest["seal"]["predictions_sha256"] = sealing.sha256_of(predictions)
    custody.attach_attestation(manifest, attestation)      # inside the digest the receipt attests
    custody.anchor_seal(manifest, anchor)
    return manifest


def mode_of(manifest: Dict[str, Any]) -> ExecutionMode:
    """The manifest's mode. A manifest written before the mode existed reads as a demonstration,
    which is the fail-closed direction: it is never promoted to the deciding path by silence."""
    return resolve_mode(manifest.get("execution_mode"), False) if manifest.get("execution_mode") \
        else (ExecutionMode.PILOT if manifest.get("pilot") else ExecutionMode.DEMONSTRATION)


def require_integrity(manifest: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None,
                      *, current_code: bool = True) -> str:
    """Is this record still the record it was sealed as? Returns the compatibility mode word.

    It answers three questions the mode gates do not. Does the manifest still recompute to the shape
    it declares, so that the derived fields agree with the mode inside the seal. Does the kept
    payload still verify against the seal and against the specification hash, so that a prediction
    edited after sealing refuses. And is the analysis in hand the analysis that was frozen, both the
    predictions the caller is scoring and the code doing the scoring.

    It is not a proposition-level gate and never promotes anything: a demonstration passes it, which
    is the point, because a demonstration whose record has been altered is not a demonstration of
    anything. `current_code` is turned off when a saved bundle is being read on a tree that has
    moved on, and the caller is then told the check was skipped rather than that it passed.
    """
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unsupported run manifest schema")
    m = mode_of(manifest)
    word = check_mode(manifest.get("mode"), bool(manifest.get("pilot")))
    if word != mode_word(m) or bool(manifest.get("simulated")) != m.simulated \
            or bool(manifest.get("scoreable_at_proposition_level")) != m.scoreable:
        raise ValueError("the manifest's derived mode fields disagree with its execution mode")
    if bool(manifest.get("empirical_confirmatory_eligible")) != m.scoreable:
        raise ValueError("this record cannot promote its own empirical eligibility")
    if not manifest.get("seal") or "sealed_predictions" not in manifest:
        raise ValueError("missing prediction commitment")
    payload = denormalise(manifest["sealed_predictions"])
    if not sealing.verify(manifest["seal"], payload, custody.spec_hash_of(manifest))["verified"]:
        raise ValueError("manifest or prediction commitment changed")
    if sealing.sha256_of(payload) != manifest["seal"].get("predictions_sha256"):
        raise ValueError("prediction digest mismatch")
    if predictions is not None and sha256_of(predictions) != sha256_of(payload):
        raise ValueError("analysis payload differs from the sealed predictions")
    if current_code and custody.package_code_identity() != manifest.get("code_sha256"):
        raise ValueError("analysis code differs from the frozen implementation")
    return word


def require_diagnostic(manifest: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None) -> str:
    """May these readings be computed and printed as diagnostics? A pilot may not, and a record whose
    integrity has gone may not either. This is the gate every reporting path passes, deciding or not:
    the reference runner checked a flag and the existence of a seal, so it printed readings from a run
    whose predictions had been edited after sealing and said nothing."""
    if manifest.get("pilot"):
        raise PilotNotScoreable("pilot measurements may not be scored against a proposition")
    return require_integrity(manifest, predictions)


def label_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Keep diagnostic recovery distinct from the actual evidence status at every export.

    The readings are exported as `diagnostics` as well as under the name the analysis built them
    with, because a reader who sees a key called `verdicts` on a simulated recovery will quote it as
    one. Both names address the same object, so nothing is lost to a reader or a bundle written
    before the second name existed, and `evidence_status` says in words which of the two this run's
    mode entitles it to.
    """
    if "verdicts" in result:
        result["diagnostics"] = result["verdicts"]
    m = mode_of(result["manifest"])
    result["evidence_status"] = ("SIMULATION/DEVELOPMENT ONLY" if m is ExecutionMode.DEMONSTRATION
                                 else "PILOT, NOT SCOREABLE" if m is ExecutionMode.PILOT
                                 else "SMOKE TEST, NOT SCOREABLE" if m is ExecutionMode.SMOKE
                                 else "CONFIRMATORY, READ AT PROPOSITION LEVEL")
    # NOT TESTED until a deciding run says otherwise, and a deciding run is the only kind that may.
    result["empirical_verdict"] = "TESTED" if m.scoreable else "NOT TESTED"
    return result


def require_reportable(manifest: Dict[str, Any]) -> str:
    """May this run's verdicts be computed and printed at all, and under what sentence?

    A pilot may not, which is ruling 28 and is unchanged. A demonstration and a smoke test may, and
    the sentence returned is the label that must appear with them. The label is returned rather than
    left to the caller because a demonstration verdict printed without it is the defect A9 names.
    """
    m = mode_of(manifest)
    if manifest.get("pilot") or m is ExecutionMode.PILOT:
        raise PilotNotScoreable("this run is flagged as a pilot and may not be scored against any verdict rule; "
                                "its measurements feed only the sizing rule written before it ran")
    if not manifest.get("seal"):
        raise RuntimeError("no seal on this manifest; a verdict without a prior seal is not a registered result")
    return m.label


def require_released_apparatus(manifest: Dict[str, Any]) -> None:
    """Was the thing this run measured a released system, or was it one of the simulators?

    THE FLOOR THIS FILE ONCE CARRIED, AND WHY IT IS BACK IN A DIFFERENT SHAPE. An earlier version of
    this module refused confirmatory collection outright, in three places and in one sentence: the
    real assay, the identification, the complete procedure calibration and the independently verified
    commitment are not released, so no run of this package may issue an empirical confirmatory
    verdict. That was a release gate, and what replaced it is a custody gate, which is a different
    question with a different answer. Custody asks whether the record is still the record that was
    sealed and whether somebody outside this code attested it; it is a complete answer about the
    record, and a simulated system scored through a checked ladder, with a genuine receipt and a
    named attestation, satisfies every part of it. Restoring the old refusal verbatim is not
    available: it also refused the case the four modes exist to serve, a caller with a real system
    who is entitled to a deciding run.

    What is restored is the part of it that is true of everything this package can point at. Every
    system shipped here is a simulation and says so in its own metadata, so a deciding run assembled
    out of them is refused, and a deciding run whose system said nothing about itself is refused with
    them, because an undeclared system cannot be told apart from one. The declaration is written into
    the confirmatory inputs before the seal and is therefore inside the sealed specification hash, so
    a run cannot acquire an honest declaration afterwards, and this refusal reaches a bundle
    re-scored from disk long after the gate that first asked has gone.

    `mode._system_refusals` asks the same question before the first paid call. Both, because they
    catch different runs: the gate is the only place that can still refuse for nothing, and this is
    the only place a saved record is read.
    """
    if not mode_of(manifest).scoreable:
        return
    record = manifest.get("confirmatory_inputs") or {}
    declarations = list(record.get("observing_systems") or ())
    if not declarations:
        raise InstrumentNotReleased(
            "this run is labelled a deciding run and its record does not name the system it "
            "measured. No assay this package ships is a released one: every system here models a "
            "world, so a record that does not say which system produced its readings cannot be read "
            "at proposition level")
    bad = [d for d in declarations if not d.get("declared") or d.get("simulated")]
    if bad:
        raise InstrumentNotReleased(
            "this run is labelled a deciding run and %d of the %d systems it names cannot decide a "
            "proposition: %s. A simulated system may demonstrate the whole apparatus and may not "
            "measure one, and a system that declared nothing cannot be told apart from the "
            "simulators this package ships"
            % (len(bad), len(declarations),
               "; ".join("%s (%s)" % (d.get("system"), "declares itself a simulation"
                                      if d.get("simulated") else "declares nothing about itself")
                         for d in bad)))


def require_scoreable(manifest: Dict[str, Any], predictions: Optional[Dict[str, Any]] = None,
                      config: Any = None, verify_code: bool = False, ladder: Any = None,
                      verify_environment: bool = False,
                      heldout_sha256: Optional[str] = None) -> Dict[str, Any]:
    """May this run be read at proposition level? Only the deciding path may, and only if it can show
    its custody.

    The mode gate is unchanged. What follows it is finding A8: a deciding run must carry an anchor
    receipt that is not a mock, the receipt must still attest the seal record's recomputed digest,
    and the specification hash must still recompute from the manifest's own fields to the value that
    was sealed, so that a configuration edited after the seal refuses instead of scoring. Where the
    caller holds the live predictions and the live configuration, they are compared with the sealed
    ones as well: this is the check that catches a prediction altered between the seal and the
    verdict, which nothing in the reference runner could see.

    AND THE THIRD THING THE FINDING NAMES. A deciding run must also carry a named party's statement
    that the material deciding its predictions was unseen when they were chosen. Nothing computable
    settles that question, so the runner requires the sentence and refuses its own placeholder here
    exactly as it refuses its own anchor. `heldout_sha256` is the digest of that material where the
    scorer holds it, and a disagreement with the digest the attestation speaks about refuses; where
    the scorer does not hold it the comparison is reported as not made rather than as passed.

    `ladder` is the LIVE ladder where the caller holds one, and it is compared with the identity the
    manifest recorded rather than the recorded identity being compared with itself. Without it the
    verifier binding was documentary: a run could seal under one checking rule, swap the ladder's rule
    for a permissive one and score, with the recorded identity untouched and the specification hash
    intact. A re-scoring from a saved bundle has no ladder object and passes None, and the returned
    report says the check was not performed rather than saying it passed.

    Returns the custody report so that a caller may record what was checked rather than assert it.
    """
    m = mode_of(manifest)
    if manifest.get("pilot") or m is ExecutionMode.PILOT:
        raise PilotNotScoreable("this run is flagged as a pilot and may not be scored against any verdict rule; "
                                "its measurements feed only the sizing rule written before it ran")
    if not m.scoreable:
        raise NotScoreable(
            "this run was made in %s mode and may not be scored at proposition level. %s" % (m.value, m.label))
    if not manifest.get("seal"):
        raise RuntimeError("no seal on this manifest; a verdict without a prior seal is not a registered result")
    # CUSTODY FIRST, THEN THE APPARATUS, and the order is deliberate. `require_custody` names every
    # custody failure at once rather than the first one found, and a single-issue refusal placed in
    # front of it would turn that complete report back into a first-failure one. Nothing is spent on
    # either path here, the run having already happened, so the order decides only which sentence a
    # caller reads and not whether the run is refused. Before the first paid call the apparatus does
    # speak first, in `mode._system_refusals`, where refusing early is what costs nothing.
    report = custody.require_custody(manifest, predictions=predictions, config=config,
                                     external_anchor_required=True, verify_code=verify_code,
                                     ladder=ladder, verify_environment=verify_environment,
                                     heldout_sha256=heldout_sha256)
    require_released_apparatus(manifest)
    return report


def write(manifest: Dict[str, Any], path: str) -> str:
    """Write a sealed record, once. The mode is create-only: a sealed record replaced in place leaves
    no trace that it was replaced, and a second run naming the same path is refused rather than
    quietly overwriting the first run's commitment. The bytes are the strict serialisation, so what
    is on disk is the same canonical form the digests were taken over."""
    data = canonical_bytes(manifest) + b"\n"
    with open(path, "xb") as fh:
        fh.write(data)
    return hashlib.sha256(data).hexdigest()
