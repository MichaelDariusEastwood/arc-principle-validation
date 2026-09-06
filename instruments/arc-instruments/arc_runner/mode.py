"""The execution mode: which kind of run this is, said once, explicitly, and fail closed.

WHY THIS FILE EXISTS. Before it, the only thing separating a rehearsal from a deciding run was a
pair of flags and a line of stderr text. The real P5 path constructed the provider adapter, printed
"this is a smoke test", handed the bank the reference arithmetic pool and started spending. A reader
of the manifest could not tell a recovery of a known simulated coupling from a measurement of a real
system, because the manifest said neither. The defect that prevents is the one that costs most: a
run that was never an instrument being read, later, as though it were.

THE FOUR MODES, AND WHAT EACH ONE MAY CLAIM.

DEMONSTRATION runs against the simulated system. Its verdicts are recoveries of a test case whose
answer the test already knows, so they are evidence about the pipeline and never about any real
system. Its manifest says so and its printed summary says so.

SMOKE runs against a real provider with a pool that was not written for the study (the reference
arithmetic pool), for the bounded purpose of proving that the wiring works. It is a separate command
rather than a confirmatory run distinguished only by a warning, which is finding A9's whole point:
the difference between a rehearsal and a deciding run must be a choice the operator typed, not a
line of text they may not have read.

PILOT runs against a real provider to size the instrument. Ruling 28: it is never scored.

CONFIRMATORY is the deciding path, and the only mode that may be scored at proposition level. It
refuses to begin unless every input the registration names is present BEFORE the first provider
call: a domain ladder that is not the reference smoke pool, that is not scored by a substring smoke
check, and that declares which population its readings sample and what a repeat of one resamples; a
checkpoint store holding every state the bank will place; a resolved configuration carrying every
registered quantity its verdict will be read with; a spending CONTROLLER, which is the object that
reserves before each dispatch rather than a figure nothing reserves against; and an anchoring service
that will attest the seal. A missing input stops the run with a named refusal listing every
requirement that failed, not only the first, so that an operator fixes the setup once rather than
four times.

AND THE REGISTERED NUMBERS ARE INPUTS TOO. A configuration can be complete as an experiment and still
carry none of the five numbers its verdict rule reads, and a run in that state pays for every arm and
can only end NOT EVALUABLE: on the shipped defaults a titration reports location NOT SUPPLIED,
slope-magnitude NOT SUPPLIED, controls UNRESOLVED and P16 NOT EVALUABLE, whatever it is measuring.
All five are checked here, and the reason all five are checked is not that each one alone ends the
run at NOT EVALUABLE, because measured on the run only two of them do. The calibration uncertainty
and the slope band each leave a contract-required component NOT SUPPLIED on their own. The horizon,
the practical-absence band and the across-window resolution do something quieter and worse: the
verdict keeps the name it would have had, and the controls component and the refutation rule reach it
having read evidence the registration does not have. A number missing from the rule that reads it is
refused either way, and the refusals below say which of the two it is rather than saying the louder
one about all five. The gate never supplies one: a number this runner filled in would be this
runner's choice wearing the registration's name.

THE ANCHORING SERVICE IS AN INPUT AND NOT AN AFTERTHOUGHT (finding A8). A deciding run whose seal
nothing outside the runner attests cannot be scored, so discovering the absence at scoring time means
the whole run was spent for nothing. It is therefore checked here, with the ladder and the allowance,
before the first paid call.

WHAT A REFUSAL IS NOT. It is not a verdict, it is not a scientific result, and it is not a reason to
run in a weaker mode instead. A confirmatory run that cannot start has produced no evidence at all.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .budget import Allowance, BudgetController


class ExecutionMode(str, Enum):
    """The mode is a string enum so that a manifest carries the word itself and a reader of the JSON
    needs no code to interpret it."""

    DEMONSTRATION = "demonstration"
    SMOKE = "smoke"
    PILOT = "pilot"
    CONFIRMATORY = "confirmatory"

    @property
    def simulated(self) -> bool:
        """True when nothing in the run touched a real system."""
        return self is ExecutionMode.DEMONSTRATION

    @property
    def scoreable(self) -> bool:
        """True only for the deciding path. Everything else is refused at proposition level."""
        return self is ExecutionMode.CONFIRMATORY

    @property
    def spends(self) -> bool:
        """True when the mode reaches a paid provider, which is what makes the preflight urgent."""
        return self in (ExecutionMode.SMOKE, ExecutionMode.PILOT, ExecutionMode.CONFIRMATORY)

    @property
    def label(self) -> str:
        """The sentence that goes in the manifest and at the top of every printed summary."""
        return _LABELS[self]


_LABELS = {
    ExecutionMode.DEMONSTRATION:
        "DEMONSTRATION: this run scored a simulated system whose coupling the test already knew. "
        "Every verdict below is a simulated recovery of that known test case and is evidence about "
        "the pipeline, never a measurement of any real system.",
    ExecutionMode.SMOKE:
        "SMOKE TEST: this run used a real provider with the reference pool, which was not written "
        "for the study. It proves the wiring and nothing else. No proposition-level reading may be "
        "taken from it.",
    ExecutionMode.PILOT:
        "PILOT: this run measured a real system in order to size the instrument. Ruling 28: it is "
        "never scored against any verdict rule, and its measurements feed only the sizing rule "
        "written before it ran.",
    ExecutionMode.CONFIRMATORY:
        "CONFIRMATORY: the deciding run. Its inputs were checked before the first provider call and "
        "its verdicts may be read at proposition level.",
}


class ModeRefusal(RuntimeError):
    """A named refusal. `requirements` lists every input that failed, so that the message is a
    complete setup report rather than the first thing that happened to be missing."""

    def __init__(self, message: str, requirements: Sequence[str] = ()):
        super().__init__(message)
        self.requirements = tuple(requirements)


def resolve(mode: Any, pilot: bool = False) -> ExecutionMode:
    """Turn whatever an entry point was given into one mode, and refuse the ambiguous cases.

    Fail closed on silence: an unstated mode is DEMONSTRATION, never the deciding path. A caller who
    wants a confirmatory run says so. The defect this prevents is a deciding run that nobody chose,
    which is how the reference runner reached a paid provider with placeholder loaders.
    """
    if mode is None:
        return ExecutionMode.PILOT if pilot else ExecutionMode.DEMONSTRATION
    m = mode if isinstance(mode, ExecutionMode) else ExecutionMode(str(mode))
    if pilot and m is not ExecutionMode.PILOT:
        raise ModeRefusal(
            "the pilot flag and the execution mode %r disagree; a run has one mode and it is stated "
            "once" % m.value, ("execution-mode",))
    return m


@dataclass
class ConfirmatoryInputs:
    """Everything a deciding run must hold before it calls a provider once.

    The fields the runner can fill from its own arguments (the ladder, the two loaders, the states
    the bank will place and the configuration) are overwritten by the runner rather than trusted from
    the caller, so this object cannot describe a setup other than the one about to run.
    """

    ladder: Any = None
    place_at_state: Optional[Callable[[Any], Dict[str, Any]]] = None
    start_for: Optional[Callable[[str], Dict[str, Any]]] = None
    checkpoint_store: Any = None
    states: Sequence[Any] = ()
    seed_name: str = "seed"
    config: Any = None
    config_resolution: Optional[Dict[str, Any]] = None
    allowance: Optional[Any] = None
    # The operator's anchoring service: a callable handed the sealed record's digest, returning a
    # receipt. It is supplied and never manufactured here, because an anchor performed by the runner
    # is an assertion by the party being checked. See arc_runner.custody.
    anchor: Optional[Callable[[str], Dict[str, Any]]] = None
    # The named party's statement that the material deciding this run's predictions was unseen when
    # they were fixed, being the third thing finding A8 requires of a deciding run and the one no
    # hash can supply. It travels with the other pre-run inputs because it is a pre-run act: an
    # attestation composed after the material was read attests nothing. It is a record and never a
    # callable, since the runner asks nobody for it and writes none itself. See
    # arc_runner.custody.attestation, and `as_record` for what the manifest keeps of it.
    attestation: Optional[Dict[str, Any]] = None
    # The domain's independent capability manipulations, being the second placement channels the
    # identification judgement needs. They are gated here with everything else because a manipulation
    # is the only route to IDENTIFIED and nothing on the deciding path had ever inspected one: its
    # loader never met the placeholder refusal below, so a deciding run could reach that word from a
    # channel that manufactured its artefacts from a number. See `_manipulation_refusals`.
    manipulations: Sequence[Any] = ()
    # THE THING BEING MEASURED, WHICH THE GATE HAD NEVER BEEN SHOWN. Every other field here is part
    # of the apparatus around the system: the ladder that reads it, the store its artefacts come
    # from, the ceiling its calls spend against, the receipt that anchors its seal. None of them
    # asks what the system IS, so a deciding run could be assembled entirely out of checked parts
    # around a simulator and the record would say `simulated: false`, because that field is derived
    # from the mode word and not from the object. These are the objects that actually produce the
    # readings this run will score, written in by the runner from its own arguments for the same
    # reason the ladder and the loaders are: an inputs object that can name a system other than the
    # one about to run is a record of an intention. See `_system_refusals`.
    observing_systems: Sequence[Any] = ()
    notes: Dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> Dict[str, Any]:
        """What the manifest keeps: the identities, never the objects and never a key."""
        res = dict(self.config_resolution or {})
        allowance = self.allowance
        if isinstance(allowance, BudgetController):
            allowance_record = allowance.report()
            # A controller is the object that reserves before each dispatch, so the figure it reports
            # bounds what this run asks for. The controller's own report says what that does and does
            # not promise; `enforced` is here so that a reader of the manifest can tell the two cases
            # apart without knowing either type.
            allowance_record["enforced"] = True
        elif isinstance(allowance, Allowance):
            # AND WHAT A BARE ALLOWANCE IS, SAID EXACTLY (finding A9). The record used to carry the
            # controller's sentence, "this bounds what the runner asks for", against an object that
            # bounds nothing: an Allowance is an approved figure, and until a BudgetController holds
            # it nothing reserves against it call by call. A manifest that claims enforcement the run
            # did not have is the same defect as a mode the manifest did not state.
            allowance_record = {"allowance_gbp": allowance.total_gbp(), "limit_gbp": allowance.limit_gbp,
                                "reserve_gbp": allowance.reserve_gbp, "includes": dict(allowance.includes),
                                "enforced": False,
                                "guarantee": "an approved ceiling recorded with this run. A bare "
                                             "allowance is a figure and not a controller: nothing "
                                             "reserved against it call by call, and no local "
                                             "controller can bound what a vendor charges"}
            # THE BRANCH STAYS ALTHOUGH THE DECIDING GATE NOW REFUSES THE OBJECT. Describing and
            # requiring are two jobs: `_allowance_refusals` says what the deciding path must hold,
            # and this says what an object IS. A record that could not describe an unenforced ceiling
            # could not explain the refusal, and the rehearsal modes still hold a figure.
        else:
            allowance_record = None
        from .sampling import read_model_record
        return {"ladder_sha256": getattr(self.ladder, "sha256", None),
                "ladder_smoke_only": bool(getattr(self.ladder, "smoke_only", False)),
                "ladder_simulated": bool(getattr(self.ladder, "simulated", False)),
                # Which population the readings are of, and what a repeat of one resamples. A record
                # of a deciding run that omits this cannot be re-analysed: the pass counts do not say
                # what they are a sample of. Finding A4.
                "read_model": read_model_record(self.ladder) if self.ladder is not None else None,
                "checkpoint_store_root": getattr(self.checkpoint_store, "root", None),
                "states": [float(s) for s in self.states] if self.states else [],
                "config_resolution": res, "allowance": allowance_record,
                # The second channels as they were declared, never the callables. A record of a
                # deciding run that reached IDENTIFIED and does not say which channels it rested on
                # cannot be read at all.
                "capability_manipulations": [m.as_record(self.place_at_state)
                                             for m in (self.manipulations or ())
                                             if hasattr(m, "as_record")],
                # WHAT EACH OBSERVING SYSTEM SAYS IT IS, by name, and never the object. It is
                # written before the seal, so it is inside the sealed specification hash and a run
                # cannot acquire an honest declaration after the fact.
                "observing_systems": [system_declaration(s) for s in (self.observing_systems or ())],
                "anchor_service": _anchor_name(self.anchor),
                # WHO ATTESTED AND WHAT THEY SPOKE ABOUT, never the sentence's whole record: the
                # record itself travels on the seal, where the anchor covers it, and a manifest field
                # that could disagree with it would be a second copy nothing binds.
                "prior_inspection_attester": (self.attestation or {}).get("attester"),
                "prior_inspection_heldout_sha256": (self.attestation or {}).get("heldout_sha256")}


def system_declaration(system: Any) -> Dict[str, Any]:
    """What an observing system says it is: a name, whether it declared at all, and whether it
    declares itself a simulation.

    Two conventions are read, because the package already has two and inventing a third would leave
    one of them unchecked. A margin source carries an `ObservationSpec` attached by
    `arc_runner.observation.declare`, whose `simulated` field the assay gate already reads. A model
    adapter carries `metadata()`, and both shipped adapters already answer `simulated` in it: the
    mock says True and the endpoint adapter says False. A system that answers neither has said
    nothing about itself, and silence is reported as silence rather than resolved in either
    direction, because a default here would be this code answering a question about somebody else's
    apparatus.
    """
    name = getattr(system, "name", None) or type(system).__name__
    spec = getattr(system, "observation", None)
    if spec is not None and hasattr(spec, "simulated") and hasattr(spec, "source"):
        # DECLARED MEANS WHAT THE ASSAY GATE MEANS BY IT, and not something stricter. `require_assay`
        # refuses the undeclared sentinel by that name; a spec that names a registered quantity has
        # spoken, whether or not its author also filled in a source string. Two gates asking the same
        # object the same question and answering differently is the defect this package keeps
        # repairing, so this reads the sentinel and nothing else.
        source = str(getattr(spec, "source", "") or "")
        return {"system": source or str(name), "declared": source != "undeclared",
                "simulated": bool(spec.simulated), "declaration": "observation-spec"}
    fn = getattr(system, "metadata", None)
    if callable(fn):
        try:
            record = dict(fn())
        except Exception as exc:                  # a system's bookkeeping never decides a proposition
            return {"system": str(name), "declared": False, "simulated": None,
                    "declaration": "metadata() raised: %s" % exc}
        named = str(record.get("adapter") or name)
        if "simulated" in record:
            return {"system": named, "declared": True, "simulated": bool(record["simulated"]),
                    "declaration": "adapter-metadata"}
        return {"system": named, "declared": False, "simulated": None,
                "declaration": "adapter-metadata with no simulated field"}
    return {"system": str(name), "declared": False, "simulated": None, "declaration": "none"}


def _system_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    """THE APPARATUS QUESTION, WHICH IS NOT THE CUSTODY QUESTION.

    Custody asks whether a record is still the record that was sealed and whether somebody outside
    this code attested it. It is a complete answer about the record and says nothing at all about
    what was measured: a simulated system, scored through a checked ladder, with a real receipt and a
    named attestation, passes every custody check there is. The reference floor this package used to
    carry answered the other question in one line by refusing confirmatory collection outright, on
    the ground that no real assay is released here. That floor cannot simply be restored, because a
    caller with a real system and a real assay is entitled to a deciding run and the four modes exist
    so that they can have one. What can be restored, and is what the floor was actually protecting,
    is the part that is true of everything THIS package can point at: every system it ships is a
    simulation and says so, so a deciding run assembled out of them refuses.

    Silence is refused with the simulation, for the reason `resolve` fails closed on an unstated mode
    and `observation.UNDECLARED` refuses an unstated quantity: an undeclared system is
    indistinguishable from a simulator that did not mention it, and the one reading that cannot
    decide anything by accident is to require the sentence. A caller whose system is real writes the
    declaration, and the manifest then records who said so.
    """
    systems = tuple(s for s in (inputs.observing_systems or ()) if s is not None)
    if not systems:
        return ["observing-system: no system was supplied, so nothing in this setup says what was "
                "read. A deciding run names the thing it measured"]
    out: List[str] = []
    for declaration in (system_declaration(s) for s in systems):
        if not declaration["declared"]:
            out.append("observing-system: %s declares nothing about what it is (declaration: %s). A "
                       "system that has not said whether it models a world or reads one cannot be "
                       "told apart from the simulators this package ships, so silence is refused "
                       "here as it is for an unstated mode and an unstated observed quantity; "
                       "declare it with metadata() or arc_runner.observation.declare"
                       % (declaration["system"], declaration["declaration"]))
        elif declaration["simulated"]:
            out.append("observing-system: %s declares itself a simulation, so its readings are drawn "
                       "from a model of the world and are not a reading of one. It may demonstrate "
                       "the whole apparatus and it may not decide a proposition, for the reason the "
                       "mock anchor and a simulated ladder are refused" % declaration["system"])
    return out


def _anchor_name(anchor: Any) -> Optional[str]:
    """What the manifest records about the anchoring service: a name and never the object, and never
    anything the service might carry that a public manifest must not."""
    if anchor is None:
        return None
    return str(getattr(anchor, "anchor_service", None) or getattr(anchor, "__name__", None)
               or type(anchor).__name__)


def _anchor_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    a = inputs.anchor
    if a is None:
        return ["anchor-service: no anchoring service; a deciding run's seal must be attested by "
                "something outside this runner, and a run that discovers this at scoring time has "
                "already been paid for"]
    if not callable(a):
        return ["anchor-service: %r is not callable; the service is a callable handed the sealed "
                "record's digest that returns a receipt" % type(a).__name__]
    from .custody import mock_anchor
    if a is mock_anchor:
        return ["anchor-service: the mock anchor was supplied. It manufactures its own receipt and "
                "attests nothing to anybody outside this runner, so it is refused on the deciding path"]
    return []


def _ladder_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    from .custody import smoke_verifiers, unattested_verifiers
    from .sampling import unit_refusals

    lad = inputs.ladder
    if lad is None:
        return ["domain-ladder: no ladder was supplied"]
    out = []
    if getattr(lad, "simulated", False):
        out.append("domain-ladder: this ladder measures a simulated latent capability, so it reads a "
                   "mock and not a system")
    if getattr(lad, "smoke_only", False):
        out.append("domain-ladder: this is the reference smoke pool, which nobody wrote for the study; "
                   "a deciding run needs the registered domain pool (--pool-module)")
    # THE POOL AND THE CHECK ARE TWO SEPARATE REFUSALS (finding A4). A registered pool scored by a
    # substring check is not a measurement of whether its tasks were solved: a response enumerating
    # candidate answers passes it. Barring the reference pool leaves that check free to be attached
    # to a better pool, so the check is refused on its own marker.
    smoke = smoke_verifiers(lad)
    if smoke:
        out.append("domain-ladder: the verifier(s) %s declare themselves substring smoke tests. A "
                   "response that contains the answer among others has not solved the task, so this "
                   "check measures presence and not capability; a deciding run needs exact checks"
                   % ", ".join(smoke))
    # AND THE MARK ALONE IS NOT ENOUGH (finding A4, second reading). The substring refusal above reads
    # a mark on the callable, and `functools.partial(ladder.contains_answer)` or a one-line lambda
    # around it is a new callable carrying no mark: binding one option to a smoke check put it through
    # this gate. The chain is now followed through a partial, a wrapper and a bound method, and beyond
    # that the deciding path asks each check to attest that it decides whether an item was SOLVED. An
    # unattested callable is refused, which is what any opaque wrapper becomes.
    unattested = unattested_verifiers(lad)
    if unattested:
        out.append("domain-ladder: the check(s) %s do not attest that they decide whether an item was "
                   "solved. A deciding run scores items with checks whose author marked them with "
                   "arc_runner.custody.attest_exact_check; an unmarked wrapper around a smoke check "
                   "is indistinguishable from an unmarked measurement, so silence is refused"
                   % ", ".join(unattested))
    # WHICH POPULATION THIS LADDER MEASURES, AND WHAT ITS REPEATS RESAMPLE (finding A4). A deciding
    # run states its sampling unit; it does not inherit the fail-closed default that exists so that
    # an undeclared ladder cannot manufacture precision.
    out += ["domain-ladder: %s" % r for r in unit_refusals(lad)]
    return out


def _store_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    from .code_domain import state_name          # imported here so that mode.py stays free of the domain

    store = inputs.checkpoint_store
    if store is None:
        return ["checkpoint-store: no store was supplied, so the bank has nowhere to place a cell"]
    out = []
    if not store.has(inputs.seed_name):
        out.append("checkpoint-store: no %r artefact in %s" % (inputs.seed_name, getattr(store, "root", "?")))
    # A cell placed at a state that was never built is not a cell at that state, so every state the
    # bank will place is checked here rather than discovered as a FileNotFoundError halfway through a
    # paid bank. The reference runner discovered it that way.
    missing = [s for s in (inputs.states or ()) if not store.has(state_name(s))]
    if missing:
        out.append("checkpoint-store: no checkpoint for state(s) %s; build the state ladder before the "
                   "bank runs" % ", ".join(str(s) for s in missing))
    return out


# The name `start_for` is probed with. It has to be a name no checkpoint store would ever hold, and
# the probe says so rather than reading a real system's checkpoint if some store does hold it.
_START_PROBE_SYSTEM = "__gate_probe_system_no_store_holds__"


def declared_store(fn: Any) -> Any:
    """The checkpoint store a loader declares it reads from, or None. The declaration is a statement
    about which store, never evidence that the loader reads one: see `_loader_refusals`."""
    return getattr(fn, "checkpoint_store", None)


def loader_store(*fns: Any) -> Any:
    """The first store any of these loaders declares. A runner uses it to write the store the loaders
    actually read into the gate's copy when the caller named none, so that the object the gate checks
    describes the run that is about to happen."""
    for fn in fns:
        st = declared_store(fn)
        if st is not None:
            return st
    return None


def _same_store(a: Any, b: Any) -> bool:
    """Two references to the same artefacts. Identity first, then the directory: a second
    CheckpointStore opened on the same root reads the same files and is the same store for every
    purpose this gate has, and refusing that would refuse a legitimate setup."""
    if a is b:
        return True
    ra, rb = getattr(a, "root", None), getattr(b, "root", None)
    if ra is None or rb is None:
        return False
    try:
        return os.path.realpath(str(ra)) == os.path.realpath(str(rb))
    except (TypeError, ValueError):
        return False


def _artefact_refusals(what: str, got: Any, expected: Dict[str, Any]) -> List[str]:
    """One placement, compared with the bytes the store holds for it."""
    if not isinstance(got, dict):
        return ["checkpoint-loaders: %s returned %s and not an artefact" % (what, type(got).__name__)]
    if "text" not in got:
        return ["checkpoint-loaders: %s returned an object with no artefact text, so it manufactured "
                "a cell rather than loading one" % what]
    if got.get("text") != expected.get("text"):
        return ["checkpoint-loaders: %s did not return the artefact the store holds under that name; "
                "a cell placed from something other than the checkpoint is not a cell at that state"
                % what]
    return []


def _placement_refusals(inputs: ConfirmatoryInputs, store: Any) -> List[str]:
    """Ask the loaders what they return, and compare it with what the store holds.

    Every read here is a local file read, so the whole probe happens before the first paid call and
    costs nothing. A state the store does not hold is skipped, because `_store_refusals` has already
    refused it and the loader is not at fault for it.
    """
    from .code_domain import state_name

    out: List[str] = []
    place, start = inputs.place_at_state, inputs.start_for
    states = tuple(inputs.states or ())
    for st in states:
        name = state_name(st)
        if not store.has(name):
            continue
        try:
            got = place(st)
        except Exception as exc:                       # a loader that cannot place is not a loader
            out.append("checkpoint-loaders: place_at_state(%s) raised %s: %s"
                       % (st, type(exc).__name__, exc))
            continue
        out += _artefact_refusals("place_at_state(%s)" % st, got, store.load(name))
    if not states and declared_store(place) is None:
        # A run that places no states (P16) gives the probe above nothing to read, so the only thing
        # left to ask of the placement loader is which store it says it reads. That is weaker than
        # the probe and it is said plainly here rather than left to look like the same check.
        out.append("checkpoint-loaders: place_at_state declares no checkpoint store, and this run "
                   "places no states, so nothing can be read back from it; a loader that neither "
                   "declares a store nor can be shown to return one's artefacts is a placeholder")
    if store.has(_START_PROBE_SYSTEM):
        out.append("checkpoint-loaders: the store holds an artefact named %r, which this gate uses to "
                   "probe start_for; rename it" % _START_PROBE_SYSTEM)
    elif store.has(inputs.seed_name):                  # an absent seed is `_store_refusals`' refusal
        try:
            got = start(_START_PROBE_SYSTEM)
        except Exception as exc:
            out.append("checkpoint-loaders: start_for(%r) raised %s: %s"
                       % (_START_PROBE_SYSTEM, type(exc).__name__, exc))
        else:
            out += _artefact_refusals("start_for", got, store.load(inputs.seed_name))
    return out


def _manipulation_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    """Are the second placement channels documented, and do their loaders read the store?

    WHY THIS GATE EXISTS AT ALL (a defect found in review of finding A6's repair). The
    identification judgement reaches IDENTIFIED only on an independent capability manipulation, so a
    manipulation is now the single most load-bearing object on the deciding path, and this gate had
    never seen one. Its loader was not probed, so the placeholder that finding A9 refuses for the
    bank was admissible for the second channel; its documentation was checked only inside `p5`, after
    the gate had passed and before the money was spent, which is the right order for a refusal and
    the wrong place for a deciding requirement.

    The two checks are the two the deciding path already makes of the bank's own loader, applied to
    the second one: the documentation the identification rests on, re-derived from the manipulation's
    own description, and a read-back of every state this channel will place against the artefacts the
    store holds. Both are local, so both happen before the first paid call.
    """
    from .code_domain import state_name
    from .p5_identification import failures_in_record

    ms = list(inputs.manipulations or ())
    if not ms:
        return []
    out: List[str] = []
    store = inputs.checkpoint_store
    for m in ms:
        name = getattr(m, "name", None) or "unnamed"
        if not hasattr(m, "as_record"):
            out.append("capability-manipulation %r: this is not an "
                       "arc_runner.p5_identification.CapabilityManipulation, so its documentation "
                       "cannot be read" % name)
            continue
        for f in failures_in_record(m.as_record(inputs.place_at_state)):
            out.append("capability-manipulation %r: %s" % (name, f))
        place = getattr(m, "place_at_state", None)
        if place is None or store is None:
            continue
        for st in tuple(getattr(m, "states", None) or inputs.states or ()):
            label = state_name(st)
            if not store.has(label):
                out.append("capability-manipulation %r: no checkpoint for state %s; a second channel "
                           "places its own states and they are built before it runs, exactly as the "
                           "bank's are" % (name, st))
                continue
            try:
                got = place(st)
            except Exception as exc:                   # a loader that cannot place is not a loader
                out.append("capability-manipulation %r: place_at_state(%s) raised %s: %s"
                           % (name, st, type(exc).__name__, exc))
                continue
            out += ["capability-manipulation %r: %s" % (name, r)
                    for r in _artefact_refusals("place_at_state(%s)" % st, got, store.load(label))]
    return out


def _loader_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    """Do the loaders read the store this gate checked, and do they return what it holds?

    WHY A MARKER WAS NOT ENOUGH, AND WHAT REPLACED IT. The first version of this rule refused a
    loader carrying no `checkpoint_store` attribute. An attribute is a label an author attaches, so
    attaching it to the placeholder lambda finding A9 names, the one that manufactures an artefact
    from a number, made a setup pass the gate whose whole purpose was to refuse it: the acceptance
    case was enforced by an opt-in convention rather than by anything about the artefact returned.
    The rule now asks the loaders what they return. For every state the bank will place,
    `place_at_state` must return the artefact the store holds at that state; `start_for` must return
    the frozen seed for a system the store has never seen. A placeholder cannot satisfy either,
    whatever it is labelled.

    AND WHICH STORE THEY READ. The store this gate validates is `inputs.checkpoint_store`; the store
    the bank places from is whichever one the loaders close over. Nothing compared the two, so a
    complete store supplied beside loaders bound to an incomplete one passed the gate, reached the
    provider, and failed inside the paid bank with the missing checkpoint the gate exists to catch.
    A disagreement is now refused in its own words, before the probe, because a probe against the
    wrong store answers the wrong question.
    """
    out: List[str] = []
    store = inputs.checkpoint_store
    declared: Dict[str, Any] = {}
    for name, fn in (("place_at_state", inputs.place_at_state), ("start_for", inputs.start_for)):
        if fn is None or not callable(fn):
            out.append("checkpoint-loaders: %s is not callable" % name)
            continue
        st = declared_store(fn)
        declared[name] = st
        if st is not None and store is not None and not _same_store(st, store):
            out.append("checkpoint-loaders: %s reads the checkpoint store at %r while the store this "
                       "gate checked is %r; the states that were verified are not the states this "
                       "run will place" % (name, getattr(st, "root", "?"), getattr(store, "root", "?")))
    if len(declared) == 2 and all(v is not None for v in declared.values()) \
            and not _same_store(declared["place_at_state"], declared["start_for"]):
        out.append("checkpoint-loaders: the two loaders read different checkpoint stores (%r and %r), "
                   "so the bank and the held-out panel are not reading one checkpoint bank"
                   % (getattr(declared["place_at_state"], "root", "?"),
                      getattr(declared["start_for"], "root", "?")))
    if out or store is None:
        # A missing, uncallable or differently bound loader has already failed, and an absent store is
        # named by `_store_refusals`. Probing after either would only repeat what is said above.
        return out
    return _placement_refusals(inputs, store)


# THERE ARE TWO EXPERIMENTS, SO THERE ARE TWO RULE SETS (finding A9, third refutation). One P5-shaped
# rule set was applied to whatever configuration reached the gate, and P16's configuration carries no
# states, no retention fractions, no replicates and no margin, so `confirm p16` refused every setup
# with four requirements no flag could ever satisfy. The documented deciding path for P16 was
# unreachable: a gate that refuses everything is an outage in the shape of a safeguard, and it passes
# refusal tests perfectly. Each rule set below is checked only against the configuration it belongs
# to, and a configuration matching neither shape is refused rather than passed, because a
# configuration the gate cannot check is not a checked configuration.
_P5_CONFIG_RULES = (
    ("states", lambda c: len(getattr(c, "states", ()) or ()) > 0, "the bank has no states"),
    ("fractions", lambda c: len(getattr(c, "fractions", ()) or ()) > 0, "the bank has no retention fractions"),
    ("reps", lambda c: int(getattr(c, "reps", 0) or 0) >= 1, "the bank has no replicates"),
    ("margin", lambda c: float(getattr(c, "margin", 0.0) or 0.0) > 0, "the margin is not positive"),
    ("window", lambda c: (not getattr(c, "checkpoints", ())) or
                         max(getattr(c, "checkpoints")) == getattr(c, "window_end", None),
     "the final checkpoint and the window end disagree"),
)

# The P16 rules ask of the titration what the P5 rules ask of the bank: is there an experiment here at
# all, and can the sealed line's numbers be reached by the arms as configured? Each one names a
# configuration under which the run could not produce the evidence its verdict rule reads.
_P16_CONFIG_RULES = (
    ("dose_offsets", lambda c: len(getattr(c, "dose_offsets", ()) or ()) > 0,
     "the titration has no dose offsets, so no arm is placed either side of the boundary"),
    ("systems_per_arm", lambda c: int(getattr(c, "systems_per_arm", 0) or 0) >= 1,
     "no systems per arm, so no arm has a system in it"),
    ("horizon", lambda c: int(getattr(c, "horizon", 0) or 0) >= 1,
     "the horizon is not a number of rounds"),
    ("switch_round", lambda c: 0 <= int(getattr(c, "switch_round", -1)) < int(getattr(c, "horizon", 0) or 0),
     "the switch round does not fall inside the horizon, so no arm is ever dosed"),
    ("settling", lambda c: int(getattr(c, "settling", 0) or 0) >= 0 and
                           int(getattr(c, "switch_round", 0) or 0) + int(getattr(c, "settling", 0) or 0)
                           < int(getattr(c, "horizon", 0) or 0),
     "the settling period runs to or past the horizon, so no round after the dose is ever read"),
    ("timing_tolerance", lambda c: int(getattr(c, "timing_tolerance", 0) or 0) > 0,
     "the timing tolerance is not positive, so the timing component can never be met"),
    ("chi_hat", lambda c: 0.0 <= float(getattr(c, "chi_hat", -1.0)) < 1.0,
     "chi is outside [0, 1), so the sealed line's zero at 1/(1 - chi) is not a point an arm can reach"),
    ("z_threshold", lambda c: float(getattr(c, "z_threshold", 0.0) or 0.0) > 0,
     "the alarm threshold is not positive, so every look alarms"),
    ("alpha_crit_hat", lambda c: math.isfinite(float(getattr(c, "alpha_crit_hat", float("nan")))),
     "the located boundary is not a finite number"),
)


# THE FIVE REGISTERED QUANTITIES A P16 RESULT IS DECIDED BY, AND WHY AN ABSENT ONE STOPS THE RUN HERE
# RATHER THAN AT SCORING TIME. The rules above ask whether the titration can be run at all. This rule
# asks the second question, which costs exactly as much to get wrong: with every arm collected and
# paid for, is there anything a verdict rule can read? Five numbers decide the P16 result and none of
# them has a default, because choosing the width of a band that decides a proposition is the author's
# act and not this module's. Under the shipped configuration all five are unset, and a run in that
# state is not a weaker run, it is one that can only end NOT EVALUABLE: a titration on the shipped
# defaults reports location NOT SUPPLIED, slope-magnitude NOT SUPPLIED, controls UNRESOLVED and P16
# NOT EVALUABLE, whatever it is measuring. That is the reasoning which put the anchoring service in
# this gate. A
# deciding run that discovers at scoring time that it has no anchor has already been paid for, and so
# has one that discovers it has no band.
#
# AND THE FIVE DO NOT ALL FAIL THE SAME WAY, WHICH THE REFUSALS BELOW HAVE TO SAY RATHER THAN ROUND
# OFF. The sentence above is measured on the shipped configuration, where all five are unset. Said of
# any ONE of them it is false, and it was false here until it was measured: dropping the informative
# horizon alone, the practical-absence band alone or the across-window resolution alone leaves the
# verdict exactly where the complete configuration left it, because NOT EVALUABLE needs a
# contract-required component in NOT SUPPLIED and the only ones any of the five can put there are
# location and slope-magnitude, both of which are reached through the calibration uncertainty and the
# slope band (`p16_components.location_component` and `slope_component`). The other three are refused
# for the quieter failure: the verdict keeps its name and is reached on evidence the registration does
# not have. `alone_not_evaluable` below records which of the two a quantity is, so that an operator
# reading a refusal is told what their run would actually have done. A gate that overstates its
# consequence teaches the next reader to discount it, and this one is the last thing standing between
# an unregistered band and a paid run.
#
# AND A VALUE IS NOT YET A REGISTRATION. The demonstration is given the candidate numbers
# `p16_calibration.py` measures the decision family under, so a configuration reaching this gate may
# carry all five and have registered none of them. The names of those travel in the configuration's
# own `candidate_quantities`, inside the sealed specification hash, and this rule refuses them beside
# the absent ones.
#
# WHAT THIS RULE DOES NOT DO. It does not supply a number and it does not bound a registered
# magnitude. The refusal names the field and what reads it; the value comes from the registration.
# Where a supplied value is refused below it is refused on the criterion the rules above use, being a
# configuration under which the run could not produce the evidence its verdict rule reads, and never
# for being larger or smaller than anything here would have chosen.
_P16_REGISTERED_QUANTITIES = (
    # name, what it is, what a run without it does, and whether that alone ends the run at NOT
    # EVALUABLE. The last field is measured on the run and not reasoned about here: a titration was
    # run with each quantity dropped in turn, and only the first two moved the verdict.
    ("chi_hat_se", "the calibration uncertainty on the located boundary",
     "the sealed zero is 1 / (1 - chi) and the sealed slope is minus (1 - chi), so without it the "
     "location and slope-magnitude components read NOT SUPPLIED rather than compare anything, which "
     "ends the run at NOT EVALUABLE on its own: the only comparison left treats the boundary as "
     "exactly known, which is the point comparison those components exist to replace", True),
    ("slope_equivalence", "the equivalence band on the sealed slope",
     "without it the slope-magnitude component reads NOT SUPPLIED and the run ends at NOT EVALUABLE "
     "on its own, because there is no band the fitted slope can be shown to lie inside; the only "
     "test left is that the line falls, which a line an order of magnitude too shallow satisfies",
     True),
    ("informative_horizon", "the rounds after the switch beyond which silence is informative",
     "without it every non-alarm is censored, so no control's silence is demonstrated, the controls "
     "component cannot be satisfied and no arm's silence is admissible for refutation. The verdict "
     "keeps the name it would have had and is reached without that evidence, which is worse than a "
     "refusal to score and is why the run is stopped here instead", False),
    ("practical_absence_band", "the band about zero inside which a measured margin is practically nil",
     "without it there is no band a measured margin can be shown to lie inside, so the "
     "practical-absence reading is unreachable and a control whose margin IS nil reads LOW "
     "INFORMATION rather than demonstrating its steadiness; and no reading at all is admissible for "
     "refutation, since a margin merely above zero is not positivity beyond a registered band. A "
     "control whose margin sits clearly off zero still demonstrates a sign and can still satisfy the "
     "controls component, so this is not a refusal that everything goes unmeasured: it is that the "
     "one arm the band exists to read, the steady one, is the arm that stops being readable", False),
    ("across_window_segments", "the resolution the across-window predicate is read at",
     "without it the registered predicate is evaluated in no arm, so the refutation rule has no "
     "evidence to read and no control's silence is demonstrated. The verdict keeps its name and is "
     "reached on a window mean, which is the reading the registered predicate replaces", False),
)


def _p16_observable_rounds(cfg: Any) -> Tuple[Optional[int], Optional[int]]:
    """How far past the switch this titration observes, and how many rounds its post-settling window
    holds. Both are None when the configuration's own numbers are unreadable or leave no window at
    all, because `_P16_CONFIG_RULES` refuses that in its own words and a second complaint about it
    would say the same thing twice."""
    try:
        horizon = int(getattr(cfg, "horizon"))
        switch = int(getattr(cfg, "switch_round"))
        settling = int(getattr(cfg, "settling"))
    except (AttributeError, TypeError, ValueError):
        return None, None
    start = switch + settling
    if horizon <= start or switch < 0 or settling < 0:
        return None, None
    return horizon - 1 - switch, horizon - start


def _p16_registered_quantity_refusals(cfg: Any) -> List[str]:
    """Every registered quantity this P16 configuration lacks, and every one it carries that could
    never decide anything."""
    from .observation import MIN_POINTS

    out: List[str] = []
    missing = [n for n, _, _, _ in _P16_REGISTERED_QUANTITIES if getattr(cfg, n, None) is None]
    if missing:
        # Said once with every name in it, because these are answered by ONE act rather than five:
        # they are registration decisions and they go to whoever holds the registration, together.
        #
        # AND THE CONSEQUENCE IS THE MEASURED ONE FOR THIS CONFIGURATION, not the loudest of the
        # five. This line used to end "a deciding run missing any of them pays for every arm and can
        # only end NOT EVALUABLE", which is true when all five are unset and false when the missing
        # one is the horizon, the band or the resolution: those three leave the verdict where the
        # complete configuration left it and change what it was reached on. An operator who reads
        # that they would have got NOT EVALUABLE, runs anyway and gets a named verdict has been told
        # something untrue by the gate, and the next refusal they read is worth less for it.
        fatal = [n for n, _, _, alone in _P16_REGISTERED_QUANTITIES
                 if alone and getattr(cfg, n, None) is None]
        quiet = [n for n in missing if n not in fatal]
        if fatal:
            consequence = ("%s %s a contract-required component NOT SUPPLIED, so this run pays for "
                           "every arm and ends at NOT EVALUABLE"
                           % (" and ".join(fatal),
                              "alone leaves" if len(fatal) == 1 else "each leave"))
            if quiet:
                consequence += ("; %s %s what the verdict is reached on rather than its name"
                                % (", ".join(quiet),
                                   "changes" if len(quiet) == 1 else "change"))
        else:
            consequence = ("none of them alone ends the run at NOT EVALUABLE: each removes evidence "
                           "the controls component and the refutation rule read, so this run pays "
                           "for every arm and reaches a named verdict on evidence the registration "
                           "does not have, which no later analysis can repair")
        out.append("registered-quantity: a P16 result is decided by %d registered quantities and %d "
                   "of them %s unset (%s). Each is a registered choice this runner will not make, "
                   "and %s" % (len(_P16_REGISTERED_QUANTITIES), len(missing),
                               "is" if len(missing) == 1 else "are",
                               ", ".join(missing), consequence))
    for name, what, consequence, _alone in _P16_REGISTERED_QUANTITIES:
        if getattr(cfg, name, None) is None:
            out.append("registered-quantity: %s is unset. It is %s, and %s"
                       % (name, what, consequence))

    # AND A NUMBER LABELLED A CANDIDATE IS NOT A REGISTRATION. A configuration may carry a value for
    # all five and still have registered none of them: the demonstration runs on the candidate numbers
    # `p16_calibration.py` measures the decision family under, and those numbers travel labelled
    # inside the sealed configuration. A present value therefore satisfies the absence rule above and
    # says nothing about whose choice it was, which is the second question this gate has to ask. It is
    # asked here, before the money, for the reason the first one is: a deciding run that learns at
    # scoring time that its bands were candidates has already been paid for, and it would have decided
    # a proposition on a width nobody registered.
    labelled = [str(n) for n in (getattr(cfg, "candidate_quantities", ()) or ())]
    if labelled:
        out.append("registered-quantity: %d of this titration's quantities carry CANDIDATE values "
                   "rather than registered ones (%s). A candidate is a number this package measures "
                   "its own decision family under so that the family can be exercised at all; it is "
                   "not the author's registered choice, and a proposition decided from one would be "
                   "decided on a width nobody registered" % (len(labelled), ", ".join(labelled)))
    for name in labelled:
        out.append("registered-quantity: %s carries the candidate value %r. Register the value and "
                   "supply it as a registration, or run this titration as a demonstration, which is "
                   "the mode a candidate number exists for" % (name, getattr(cfg, name, None)))

    # AND A VALUE THAT IS PRESENT BUT CAN DECIDE NOTHING IS REFUSED BESIDE THE ABSENT ONES. Each case
    # here leaves the component that reads the number unable to reach any state, which is the same
    # outage as an unset field wearing a number. Nothing here refuses a magnitude for its size.
    observed, window = _p16_observable_rounds(cfg)

    def _number(name):
        raw = getattr(cfg, name, None)
        if raw is None:
            return None, None
        try:
            return raw, float(raw)
        except (TypeError, ValueError):
            return raw, float("nan")

    raw, se = _number("chi_hat_se")
    if raw is not None and (not math.isfinite(se) or se < 0):
        out.append("registered-quantity: a calibration uncertainty of %r is not a standard error. A "
                   "negative value is squared into the propagated uncertainty and read as though its "
                   "magnitude had been the registered one, and an unbounded value makes the location "
                   "and slope-magnitude intervals infinite, so neither component could ever be "
                   "satisfied. A registered ZERO is not refused here: it states that the boundary "
                   "was located exactly, which is a claim made in the open and sealed with the "
                   "specification, and setting a floor above it is the author's act" % (raw,))
    raw, band = _number("slope_equivalence")
    if raw is not None and (not math.isfinite(band) or band <= 0):
        out.append("registered-quantity: an equivalence band of %r on the sealed slope has no width, "
                   "so no interval can lie inside it and the slope-magnitude component can never be "
                   "satisfied" % (raw,))
    raw, absence = _number("practical_absence_band")
    if raw is not None and (not math.isfinite(absence) or absence < 0):
        out.append("registered-quantity: a practical-absence band of %r is not a band about zero. A "
                   "negative width admits nothing, and an unbounded one calls every measured margin "
                   "practically nil including one that plainly is not, so every silence would be "
                   "demonstrated by a band that excludes nothing. A registered ZERO is not refused: "
                   "it is the strict reading, in which only a margin measured at zero is nil"
                   % (raw,))
    raw = getattr(cfg, "informative_horizon", None)
    if raw is not None:
        try:
            horizon_after = int(raw)
        except (TypeError, ValueError):
            horizon_after = 0
        if horizon_after < 1:
            out.append("registered-quantity: an informative horizon of %r declares silence "
                       "informative before a single round after the switch has been observed, which "
                       "is the refutation from silence alone that a horizon exists to prevent"
                       % (raw,))
        elif observed is not None and horizon_after > observed:
            out.append("registered-quantity: the informative horizon is %d rounds after the switch "
                       "and this titration observes %d, so every arm's silence is censored in every "
                       "run and no silence can be read at all" % (horizon_after, observed))
    raw = getattr(cfg, "across_window_segments", None)
    if raw is not None:
        try:
            segments = int(raw)
        except (TypeError, ValueError):
            segments = 0
        if segments < 2:
            out.append("registered-quantity: an across-window resolution of %r reads the window in "
                       "one piece, which is its mean and not a reading across it: the registered "
                       "predicate is a lower bound that STAYS above the band, and one segment cannot "
                       "show that anything stayed anywhere. This is the refusal "
                       "arc_runner.observation.window_segments raises, asked before the money rather "
                       "than after it" % (raw,))
        elif window is not None and window < segments * MIN_POINTS:
            out.append("registered-quantity: the post-settling window holds %d rounds and a "
                       "resolution of %d needs %d at %d rounds a segment, so no arm's window would "
                       "be read at the registered resolution and the across-window predicate would "
                       "be unevaluated in every one" % (window, segments, segments * MIN_POINTS,
                                                        MIN_POINTS))
    return out


def _config_rules_for(cfg: Any) -> Tuple[Optional[str], Tuple]:
    """Which experiment's configuration this is, decided by the fields it carries rather than by its
    class name, so that a caller's own configuration object is checked on the same terms."""
    if hasattr(cfg, "dose_offsets") and hasattr(cfg, "switch_round"):
        return "P16", _P16_CONFIG_RULES
    if hasattr(cfg, "states") and hasattr(cfg, "fractions"):
        return "P5", _P5_CONFIG_RULES
    return None, ()


def _convention_refusals(cfg: Any) -> List[str]:
    """Every way this configuration WEAKENS the registered interval and boundary convention.

    WHY A DECIDING RUN IS GATED ON THIS (finding A7). The convention is deliberately configurable and
    lives in one place, `arc_runner.p5_prediction.REGISTERED`, so that the prediction interval, the
    per-system comparison and the coupling domain cannot drift apart. Configurable in one place is
    what stops the convention being three conventions; it is not a licence to soften the rule on the
    run that decides a proposition. Each departure below makes agreement EASIER to reach than the
    registered convention makes it: a closed boundary calls exact contact clearance, a smaller
    multiplier reads a narrower interval at a lower level, a wider coupling domain scores predictions
    the runner cannot represent, an excursion tolerance above the registered one scores them anyway,
    a resampling count below the registered one reports a half width biased low and widely scattered,
    and a positive prediction-observation correlation subtracts from the variance of the difference.
    Departures in the STRICT direction pass: a run may hold itself to more than the registered
    convention and say so. Every rule below is written against that criterion and not against mere
    well-formedness: a count of one draw forms no interval at all AND a count of fifty is a weakening,
    and the second is the one a gate written only against well-formedness lets through.

    A configuration that does not carry these fields at all passes untouched, because
    `convention_from_config` then reads the registered convention, which is the value being checked
    for. That is also what keeps this rule from refusing a P16 configuration, which has none of them.

    CONSERVATIVE READING, NAMED AS OPEN. The contract names the ninety per cent two one-sided tests
    interval as the registered choice and strict clearance as the convention, and does not say
    whether a recorded resolution may license a weaker one on the deciding path. The reading taken
    here is that it may not: a weaker convention is a registration change and belongs in the
    registered defaults, where it is visible to everyone, rather than in one run's configuration. If
    the author settles that a resolved configuration may depart, the refusals below become warnings
    carried into the manifest and this is the one place that changes.

    WHAT THIS RULE DELIBERATELY DOES NOT BOUND, AND WHY. The margin is not checked here beyond being
    a positive number, although a larger margin makes agreement easier than any field below does. The
    margin is the registered EFFECT SIZE and not part of the interval convention: it is the size of a
    difference the proposition says does not matter, the contract states it as a registered quantity,
    and bounding it here would be choosing a ceiling the contract does not state. It is sealed inside
    the specification hash, so it cannot move after the seal, and a run that registers a wide margin
    has registered a weak proposition in the open rather than softened a convention on the way past.
    Whether the deciding path should also carry a registered ceiling on the margin is the author's
    decision, named here rather than taken.
    """
    from . import p5_prediction as PRED           # imported here: mode is imported by p5, and this
                                                  # rule is only reached on the deciding path
    reg = PRED.REGISTERED
    out: List[str] = []

    def _stated(name):
        v = getattr(cfg, name, None)
        return None if v is None else v

    if _stated("strict_clearance") is not None and not bool(cfg.strict_clearance):
        out.append("interval-convention: strict clearance is off, so an interval landing exactly on "
                   "the margin would be read as agreement; the registered convention is strict")
    z = _stated("equivalence_z")
    if z is not None:
        try:
            z = float(z)
        except (TypeError, ValueError):
            z = float("nan")
        if not (z == z) or z <= 0:
            out.append("interval-convention: the equivalence multiplier is not a positive number, so "
                       "the prediction's interval is not an interval")
        elif z < reg.equivalence_z - 1e-12:
            out.append("interval-convention: the equivalence multiplier %.4f is below the registered "
                       "%.4f, which reads a narrower interval at a lower level than the registered "
                       "choice (%s)" % (z, reg.equivalence_z, reg.level))
    dom = _stated("coupling_domain")
    if dom is not None:
        try:
            lo, hi = float(dom[0]), float(dom[1])
        except (TypeError, ValueError, IndexError):
            lo, hi = float("nan"), float("nan")
        if not (lo == lo and hi == hi) or not (lo < hi):
            out.append("interval-convention: the permitted coupling domain is not an ordered pair of "
                       "finite numbers")
        elif lo < reg.coupling_domain[0] - 1e-12 or hi > reg.coupling_domain[1] + 1e-12:
            out.append("interval-convention: the permitted coupling domain [%g, %g] is wider than the "
                       "registered [%g, %g], so a prediction the runner cannot represent would be "
                       "scored" % (lo, hi, reg.coupling_domain[0], reg.coupling_domain[1]))
    tol = _stated("domain_excursion_tolerance")
    if tol is not None and float(tol) > reg.domain_excursion_tolerance + 1e-12:
        out.append("interval-convention: the coupling domain's excursion tolerance %g is above the "
                   "registered %g, so a coupling interval reaching the domain would be scored against "
                   "anyway" % (float(tol), reg.domain_excursion_tolerance))
    draws = _stated("prediction_draws")
    if draws is not None:
        try:
            n_draws = int(draws)
        except (TypeError, ValueError):
            n_draws = -1
        if n_draws < 2:
            out.append("interval-convention: %d resampling draw(s) cannot produce a prediction "
                       "interval, and a prediction with no interval is not compared" % n_draws)
        elif n_draws < int(reg.prediction_draws):
            # THE DRAW COUNT IS PART OF THE CONVENTION, NOT ONLY A WELL-FORMEDNESS QUESTION. This rule
            # used to refuse a draw count only below two, on the ground that fewer than two draws form
            # no interval at all. That left every count between two and the registered eight hundred
            # passing untouched, and those counts are a weakening of exactly the kind this gate
            # exists to refuse: the half width is the resampled standard deviation times the
            # multiplier, the standard deviation of a small sample is biased low and scattered wide,
            # and a run that happens to draw a narrow one clears a margin the registered count would
            # not have cleared. Measured on the propagation this rule protects, at a coupling of 0.50
            # with a standard error of 0.03 read to window 128, over 200 seeds: the registered 800
            # draws give a mean half width of 0.0910 with a minimum of 0.0851; two draws give a mean
            # of 0.0673, a minimum of 0.0001, and a width narrower than the same seed's registered
            # width in 74 per cent of seeds; fifty draws still reach a minimum of 0.0640 and are
            # narrower in 56 per cent. The
            # bias vanishes well before eight hundred, and the scatter does not, so the direction of
            # the departure is one-sided and the registered count is the floor rather than a target.
            # MORE draws than the registered count pass: they can only sharpen the estimate of a
            # width nothing else in the comparison depends on the size of.
            out.append("interval-convention: %d resampling draws is below the registered %d. The "
                       "prediction's half width is estimated from those draws, and a smaller count "
                       "reports a width that is biased low and widely scattered, so agreement is "
                       "easier to reach than the registered convention makes it; a larger count is "
                       "not refused" % (n_draws, int(reg.prediction_draws)))
    rho = _stated("prediction_observation_correlation")
    if rho is not None:
        rho = float(rho)
        if not (-1.0 <= rho <= 1.0):
            out.append("interval-convention: the prediction-observation correlation %g is not a "
                       "correlation" % rho)
        elif rho > 0.0:
            out.append("interval-convention: a positive prediction-observation correlation (%g) "
                       "subtracts from the variance of the difference; the registered value is zero "
                       "because the two sides share no reading, and a domain claiming they do must "
                       "register that claim rather than configure it into one run" % rho)
    return out


def _config_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    out = []
    res = inputs.config_resolution
    if not isinstance(res, dict) or not res.get("resolved_by") or not res.get("resolved_utc"):
        # CONSERVATIVE READING, NAMED AS OPEN. The contract says a confirmatory run needs "a resolved
        # configuration" without saying what makes one resolved. The reading taken here is the one
        # that cannot be satisfied by accident: a person and a time are recorded against the exact
        # configuration about to run, and the configuration is internally consistent. If the author
        # settles a stricter rule (for example a countersigned specification hash), it belongs here.
        out.append("resolved-configuration: no resolution record naming who resolved this "
                   "configuration and when")
    cfg = inputs.config
    if cfg is None:
        out.append("resolved-configuration: no configuration was supplied, so there is nothing for a "
                   "resolution record to be a record of")
        return out
    kind, rules = _config_rules_for(cfg)
    if kind is None:
        out.append("resolved-configuration: a %s is neither a P5 bank configuration nor a P16 "
                   "titration configuration, and this gate has no rules for it. A configuration the "
                   "gate cannot check is not a checked configuration" % type(cfg).__name__)
        return out
    for _, rule, why in rules:
        try:
            ok = bool(rule(cfg))
        except Exception:
            ok = False
        if not ok:
            out.append("resolved-configuration: %s" % why)
    if kind == "P16":
        # AND THE REGISTERED NUMBERS THE VERDICT WILL BE READ WITH. They are checked here, with the
        # rest of the configuration and before the first paid call, for the reason the anchoring
        # service is checked here: a deciding run that learns at scoring time that it has no band has
        # already been paid for. They are P16 rules because the quantities belong to the titration's
        # components and its sequential rule, which a P5 bank configuration does not have.
        out += _p16_registered_quantity_refusals(cfg)
    if kind == "P5":
        # AND THE INTERVAL CONVENTION THE COMPARISON WILL BE READ UNDER (finding A7). It is checked
        # here, with the rest of the configuration and before the first paid call, because a run
        # scored under a weaker convention than the registered one has changed what agreement means
        # after the money was spent. It is a P5 rule: the fields it reads belong to the prediction
        # interval and the per-system comparison, which P16 does not have.
        out += _convention_refusals(cfg)
    return out


def _allowance_refusals(inputs: ConfirmatoryInputs, require_controller: bool = False) -> List[str]:
    """The approved ceiling, and on the deciding path the object that holds the run to it.

    WHY A FIGURE IS NOT ENOUGH WHERE THE RUN DECIDES SOMETHING. An Allowance is an approved number; a
    BudgetController is the object that reserves the conservative maximum before each dispatch,
    settles it afterwards, and halts the run when the remainder cannot cover the next call. This gate
    took either one, recorded `enforced: False` against the bare figure and let the run start, so the
    manifest said truthfully that nothing reserved against the ceiling, which is an accurate record
    of a deciding run whose spending had no ceiling on it. The command line has built a controller
    since the metered adapter was written, so the caller this refusal is for is the one that reaches
    the gate through the API, where the approved figure was still only a number in a record.

    A REHEARSAL KEEPS THE OLDER RULE. A smoke test and a pilot are refused without a figure and are
    not asked for a controller here, because that requirement belongs to the path that decides a
    proposition; `require_spending_allowance` asks for the controller on the deciding path alone.
    """
    a = inputs.allowance
    if a is None:
        return ["spending-allowance: no allowance from arc_runner.budget; a deciding run does not start "
                "without a figure it may spend"]
    if isinstance(a, BudgetController):
        if a.halted_reason:
            return ["spending-allowance: the controller is already halted (%s)" % a.halted_reason]
        return [] if a.available_gbp > 0 else ["spending-allowance: nothing left to reserve"]
    if isinstance(a, Allowance):
        if require_controller:
            return ["spending-allowance: an arc_runner.budget.Allowance is an approved figure and not "
                    "a controller. Nothing reserves against a figure call by call, so on the deciding "
                    "path the ceiling would bound the manifest and not the spending. Hold it in an "
                    "arc_runner.budget.BudgetController, which is the object the metered adapter "
                    "reserves against before each dispatch and halts the run on"]
        return [] if a.total_gbp() > 0 else ["spending-allowance: the approved figure is zero"]
    return ["spending-allowance: %r is not an Allowance or a BudgetController from arc_runner.budget"
            % type(a).__name__]


def _attestation_refusals(inputs: ConfirmatoryInputs) -> List[str]:
    """A supplied attestation must be one the seal can actually carry, asked here and not at the seal.

    WHAT THIS DOES AND DOES NOT DECIDE. It does not ask whether a deciding run must hold an
    attestation at all; that requirement lives in the custody gate, where the registration puts it,
    and a run holding none reaches it there. It asks only that a record which WAS supplied is
    complete, because `custody.attach_attestation` refuses an incomplete one outright rather than
    filling the missing fields in, and the seal it would refuse at is taken after the bank and the
    calibration in P5. A malformed record discovered there has cost the run its bank; discovered
    here it has cost nothing, which is the reason the anchoring service is asked for at this gate
    and not at the seal either.
    """
    rec = inputs.attestation
    if rec is None:
        return []
    if not isinstance(rec, dict):
        return ["prior-inspection: %r is not an attestation record; it is the mapping "
                "arc_runner.custody.attestation returns, and never a callable: this runner asks "
                "nobody for one and writes none itself" % type(rec).__name__]
    if rec.get("mock"):
        return ["prior-inspection: the runner's own placeholder attestation was supplied. It attests "
                "nothing to anybody outside this code and is refused at proposition level, so a "
                "deciding run holding it would pay for its arms and then be unable to produce a "
                "verdict"]
    absent = [f for f in ("attester", "heldout_sha256", "statement", "attested_utc")
              if not rec.get(f)]
    if absent:
        return ["prior-inspection: the attestation is missing %s. Nothing fills those in: the "
                "sentence and the time are the attester's own words, and a digest written by this "
                "code would be the party being checked answering the question asked of it"
                % ", ".join(absent)]
    return []


def missing_confirmatory_inputs(inputs: ConfirmatoryInputs) -> List[str]:
    """Every named requirement this setup fails, in the order the registration lists them."""
    out: List[str] = []
    # THE SYSTEM SPEAKS FIRST, for the reason the assay speaks before the setup gate in P16: the
    # other requirements describe the apparatus around the thing being measured, and a complete
    # apparatus around a simulator is still a simulator.
    out += _system_refusals(inputs)
    out += _ladder_refusals(inputs)
    out += _store_refusals(inputs)
    out += _loader_refusals(inputs)
    out += _manipulation_refusals(inputs)
    out += _config_refusals(inputs)
    out += _allowance_refusals(inputs, require_controller=True)
    out += _anchor_refusals(inputs)
    out += _attestation_refusals(inputs)
    return out


def require_confirmatory_inputs(inputs: Optional[ConfirmatoryInputs],
                                deferred: Sequence[str] = ()) -> ConfirmatoryInputs:
    """The gate. Called before the first provider call, never after it.

    A confirmatory run with no inputs object at all is refused in exactly the same words as one with
    an empty object, so that forgetting to pass the inputs cannot look like passing them.

    `deferred` names requirement families this particular caller cannot answer yet AND that a later
    gate on the same run does answer. There is one, and it is the command line's pre-flight: that
    gate runs before the adapter is constructed, deliberately, so that an operator learns what is
    missing without a provider key in the environment, and the observing system does not exist to be
    asked at that moment. `run_p5` and `run_p16` ask it of the object they actually hold. The
    deferral is a named argument rather than a silence so that a reader can see which question was
    postponed and a caller cannot postpone one by accident.
    """
    inputs = inputs if inputs is not None else ConfirmatoryInputs()
    missing = [r for r in missing_confirmatory_inputs(inputs)
               if not any(r.startswith(prefix) for prefix in (deferred or ()))]
    if missing:
        raise ModeRefusal(
            "a confirmatory run refuses to start: %d required input(s) are missing or are placeholders, "
            "and nothing has been spent.\n  - %s" % (len(missing), "\n  - ".join(missing)),
            missing)
    return inputs


def require_spending_allowance(mode: ExecutionMode, allowance: Any) -> Any:
    """Every mode that reaches a paid provider needs an approved ceiling before it reaches one.

    WHY THIS IS NOT THE DECIDING PATH'S RULE ALONE (finding A9's action, read whole). The action names
    a cost ceiling among the things required "before any real provider call", and the first
    implementation required one only of the confirmatory run. A smoke test and a pilot both dispatch
    to a real provider, so on those two paths the approved figure was a command line option nothing
    ever asked for and nothing ever consulted. A rehearsal that empties the account has stopped the
    deciding run as surely as a refused gate would have, and it does it without a refusal to read.

    A demonstration touches nothing and is returned untouched. The refusals are the allowance
    refusals, in the same words the deciding gate uses, because it is the same requirement.

    AND THE DECIDING PATH IS ASKED FOR THE CONTROLLER HERE TOO, in those same words, so that the two
    places a paid run is stopped cannot disagree about what the deciding path holds. A rehearsal is
    unchanged: it is refused without a figure and is not asked to hold one in a controller.
    """
    if not mode.spends:
        return allowance
    refusals = _allowance_refusals(ConfirmatoryInputs(allowance=allowance),
                                   require_controller=mode is ExecutionMode.CONFIRMATORY)
    if refusals:
        raise ModeRefusal(
            "a %s run reaches a paid provider and refuses to start without an approved ceiling, and "
            "nothing has been spent.\n  - %s" % (mode.value, "\n  - ".join(refusals)), refusals)
    return allowance
