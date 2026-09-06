"""Command line: python -m arc_runner.cli <mode> p5|p16 [options]

THE MODE IS THE COMMAND, and there are four of them. Finding A9: the reference command line separated
a rehearsal from a deciding run with `--mock`, `--pilot` and a line of stderr text, so the real P5
path constructed the provider adapter, warned that the ladder was a smoke test, and spent anyway. A
mode that has to be typed cannot be missed, and a mode that is a separate command cannot be reached
by leaving a flag off.

  demonstrate p5|p16   the simulated system. Nothing is measured; the verdicts are recoveries of a
                       known test case and every line of output says so.
  smoke p5|p16         a real provider, the reference pool, a bounded purpose: proving the wiring.
                       Labelled throughout and refused at proposition level.
  pilot p5|p16         a real provider, for sizing. Ruling 28: never scored. Needs --pool-module,
                       because a run on the reference pool is a smoke test and has its own command.
  confirm p5|p16       the deciding run. Refuses to start unless the domain ladder is a registered
                       pool, the checkpoint store holds every state the bank will place AND is the
                       store the loaders read from, the loaders return the artefacts that store
                       holds, the configuration is resolved and carries every registered quantity
                       its verdict will be read with, and the approved figure is held in a controller
                       that reserves before each call rather than being a figure in a record. The
                       refusal happens before the adapter is constructed, so before a penny is spent.

EVERY PAID MODE NEEDS A CEILING, not only the deciding one. `smoke`, `pilot` and `confirm` all reach a
real provider, so all three need --allowance-gbp and --max-call-gbp, and all three dispatch through
the controller those two build: it reserves the per-call maximum before each request leaves and halts
the run when the remainder cannot cover the next one. The first version of this file required the
figure of `confirm` alone, so a rehearsal could empty the account that the deciding run was to be paid
for out of, and the option was inert on the two commands that spend without deciding anything.

--out is the evidence bundle and not a manifest file. It names a DIRECTORY that receives the manifest
before the first paid call, the seal and its anchor receipt at the moment of sealing, and at the end
the complete record: the bank rows, every ladder read with its subset size and pass count, every
held-out replicate series, every P16 arm's margin series, the verdicts and the provider metadata. A
P16 run reads the frozen pool once per arm per round, and those readings are in it too where the run
was given the ladder object that took them.
Finding A8: the old --out wrote the manifest alone, so everything an independent analyst needs to
regenerate the table existed only inside one process. The bundle carries no answer key, no hidden
check and no credential.

`confirm p16` ALSO NEEDS THE NUMBERS ITS VERDICT WILL BE READ WITH, and there is no default for any
of them. --chi-hat-se, --slope-equivalence, --informative-horizon, --practical-absence-band and
--across-window-segments carry the registered quantities the P16 components and the sequential rule
read; the deciding gate refuses a titration missing any of them, because such a run pays for every
arm and then reads its verdict under a rule missing the number that rule reads. Missing the
calibration uncertainty or the slope band that is NOT EVALUABLE outright, a contract-required
component having nothing to compare. Missing the horizon, the practical-absence band or the
across-window resolution it is quieter and worse: the verdict keeps the name it would have had, and
the controls component and the refutation rule reach it on evidence the registration does not have.
They are options rather than defaults for the reason the mode is
a command rather than a flag: a band that decides a proposition is registered by the author, and a
number this command line filled in would be this command line's choice wearing the registration's
name. A smoke test and a pilot ignore them unless they are typed, and typing one there only records it.

`demonstrate p16` IS THE ONE PATH THAT IS GIVEN THEM, AND IT SAYS SO IN EVERY PLACE THE RESULT IS
READ. With none of them set the component wrapper withholds every result, so the demonstration
reported NOT EVALUABLE and demonstrated nothing of the branch it exists for. It now runs on the
candidate numbers `p16_calibration.py` measures the decision family under, each named as a candidate
in the sealed configuration, in the verdict block and at the top of the summary beside the mode label;
a quantity typed on the command line replaces a candidate and is no longer named as one. The deciding
gate refuses a titration carrying the label, so a candidate cannot travel from a rehearsal into a run
that decides something, and `require_scoreable` refuses the demonstration exactly as it did before.

--anchor-module names an importable module exposing `anchor(sha256) -> receipt`, which is the
operator's commitment service. `confirm` requires it, because a deciding run whose seal nothing
outside this code attests cannot be scored, and discovering that after the run has been paid for is
the expensive way to learn it. This package performs no anchoring itself and never will.

--attestation-module names an importable module exposing `attestation() -> record`, being the named
party's statement that the material deciding this run's predictions was unseen when they were fixed.
It is the third thing a deciding run's custody needs and the one no hash can supply: a digest proves
the material has not moved since it was hashed and says nothing about who had read it before. Build
the record with `arc_runner.custody.attestation`, which asks for the attester, the digest of the
material they are speaking about and the sentence. Without it the seal carries the runner's own
labelled placeholder, which attests nothing and is refused at proposition level; this package writes
no attestation and fills in no digest, for the same reason it performs no anchoring.

Real modes need ARC_RUNNER_MODEL, ARC_RUNNER_BASE_URL and ARC_RUNNER_API_KEY in the environment; the
key is never printed. They also need --domain code: a frozen hashed task pool as the hidden suite, a
checkpoint store the bank places its states from, and for P16 the four balance objects read from the
same suite. Build the state ladder with build_state_ladder BEFORE the bank runs, because a cell placed
at a state that was never built is not a cell at that state. --pool-module names an importable module
exposing `pool()`, which is how a registered run supplies the items a person wrote for the study.
"""
from __future__ import annotations

# The future import must be the first statement in the module, so every other import sits below it.
# An earlier version put two of them above it and the module stopped importing at all.
import argparse
import json
import os
import sys
import time

from . import adapters, budget as B, ladder as L, manifest as M, mode as MODE, p5, p16


def _pool_and_parts(a, need_registered_pool: bool):
    """The code domain's three parts, and the refusals that belong before the adapter exists."""
    from . import code_domain as CD
    if a.checkpoints is None:
        raise SystemExit("this run needs --checkpoints naming a store holding the seed artefact")
    if a.pool_module:
        import importlib
        pool = importlib.import_module(a.pool_module).pool()
    elif need_registered_pool:
        raise SystemExit("this run needs --pool-module naming the registered domain pool. The reference "
                         "arithmetic pool is a smoke test and has its own command: `smoke`.")
    else:
        pool = CD.reference_pool()
        print("the reference arithmetic pool is in use: this is a smoke test of the wiring and not an "
              "instrument, and no proposition may be read from it", file=sys.stderr)
    lad = CD.SuiteLadder(pool, subset_size=a.subset_size)
    store = CD.CheckpointStore(a.checkpoints)
    if not store.has("seed"):
        raise SystemExit("the checkpoint store holds no seed artefact at %s" % store.path("seed"))
    return pool, lad, store


def _anchor(a):
    """The operator's anchoring service, imported by name. Nothing here manufactures a receipt: a
    commitment attested by the party being checked is not a commitment, so an absent module means a
    mock receipt off the deciding path and a refusal on it."""
    if not a.anchor_module:
        return None
    import importlib
    mod = importlib.import_module(a.anchor_module)
    fn = getattr(mod, "anchor", None)
    if not callable(fn):
        raise SystemExit("the module %r exposes no callable `anchor(sha256)`; an anchoring service is "
                         "a callable handed the sealed record's digest that returns a receipt"
                         % a.anchor_module)
    setattr(fn, "anchor_service", a.anchor_module)
    return fn


def _attestation(a):
    """The named party's statement, imported by name. Nothing here composes one: an attestation
    written by the party being checked is not an attestation, so an absent module means the labelled
    placeholder off the deciding path and a refusal at proposition level on it."""
    if not a.attestation_module:
        return None
    import importlib
    mod = importlib.import_module(a.attestation_module)
    fn = getattr(mod, "attestation", None)
    if not callable(fn):
        raise SystemExit("the module %r exposes no callable `attestation()`; it returns the record "
                         "arc_runner.custody.attestation builds, naming the attester, the digest of "
                         "the material they say was unseen, and the sentence" % a.attestation_module)
    rec = fn()
    if not isinstance(rec, dict):
        raise SystemExit("the module %r returned %r rather than an attestation record"
                         % (a.attestation_module, type(rec).__name__))
    # AND THE RECORD IS CHECKED HERE, WHERE NOTHING HAS BEEN SPENT. `custody.attach_attestation`
    # refuses an incomplete record rather than filling the missing fields in, and the seal it refuses
    # at is taken after P5's bank and calibration: a module that omits the sentence or the time would
    # otherwise cost a run its bank to find out. The words are the deciding gate's own, imported
    # rather than repeated, so the two places cannot drift apart.
    refusals = MODE._attestation_refusals(MODE.ConfirmatoryInputs(attestation=rec))
    if refusals:
        raise SystemExit("the module %r returned a record the seal cannot carry, and nothing has "
                         "been spent.\n  - %s" % (a.attestation_module, "\n  - ".join(refusals)))
    return rec


# The registered quantities `confirm p16` takes from the command line, named once so that the parser,
# the configuration and the resolution record cannot drift apart. They are the fields the deciding
# gate names in `arc_runner.mode._P16_REGISTERED_QUANTITIES`.
P16_REGISTERED_OPTIONS = ("chi_hat_se", "slope_equivalence", "informative_horizon",
                          "practical_absence_band", "across_window_segments")


def _controller(a):
    """The controller a paid mode spends under, built from the figure the operator stated.

    IT IS A CONTROLLER AND NOT A BARE ALLOWANCE (finding A9). An Allowance is an approved figure;
    only a BudgetController reserves against it call by call, and until the adapter dispatched through
    one the ceiling bounded the manifest and not the spending. The controller built here is the same
    object that goes into the confirmatory gate's record and that wraps the adapter below, so the
    figure the manifest reports is the figure the run was actually held to.
    """
    if a.allowance_gbp is None:
        return None
    return B.BudgetController(B.Allowance(limit_gbp=float(a.allowance_gbp),
                                          reserve_gbp=float(a.allowance_reserve_gbp or 0.0)))


def _paid_adapter(a, m, controller):
    """The provider adapter every paid mode reaches, and the two refusals that precede it.

    Both refusals happen before `OpenAICompatibleAdapter` is constructed, so before a key is even
    required and long before a call. The per-call maximum is asked for separately from the total
    because a total alone cannot bound a dispatch: the controller has to know what to hold before it
    lets a request leave. There is no default for it, for the reason there is no default mode.
    """
    try:
        MODE.require_spending_allowance(m, controller)
    except MODE.ModeRefusal as exc:
        raise SystemExit(str(exc))
    if a.max_call_gbp is None:
        raise SystemExit(
            "a %s run reaches a paid provider and needs --max-call-gbp, the conservative maximum one "
            "call may cost. Without a per-call figure the allowance is a number in a record and not a "
            "bound on what this run may spend." % m.value)
    return B.MeteredAdapter(adapters.OpenAICompatibleAdapter(), controller, float(a.max_call_gbp))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["demonstrate", "smoke", "pilot", "confirm"],
                    help="the kind of run; there is no default and there is no flag that changes it")
    ap.add_argument("experiment", choices=["p5", "p16"])
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--beta", type=float, default=0.5, help="demonstrate only: the true coupling")
    ap.add_argument("--theta", type=float, default=0.0, help="demonstrate only: rate confound exponent")
    ap.add_argument("--out", default=None, help="directory for the evidence bundle")
    ap.add_argument("--anchor-module", default=None,
                    help="importable module exposing anchor(sha256) returning a receipt; required by confirm")
    ap.add_argument("--attestation-module", default=None,
                    help="importable module exposing attestation() returning the record that names "
                         "the party, the digest of the material they say was unseen when the "
                         "predictions were fixed, and the sentence; a deciding run is refused at "
                         "proposition level without one")
    ap.add_argument("--checkpoints", default=None, help="the checkpoint store directory")
    ap.add_argument("--pool-module", default=None, help="importable module exposing pool()")
    ap.add_argument("--subset-size", type=int, default=None, help="items drawn per read")
    ap.add_argument("--states", default=None, help="comma-separated target readings")
    ap.add_argument("--window", type=int, default=None, help="the sealed window's final depth")
    ap.add_argument("--allowance-gbp", type=float, default=None,
                    help="the approved figure this run may spend; required by every paid mode")
    ap.add_argument("--allowance-reserve-gbp", type=float, default=0.0)
    ap.add_argument("--max-call-gbp", type=float, default=None,
                    help="the conservative maximum one provider call may cost, reserved before each "
                         "dispatch; required by every paid mode")
    ap.add_argument("--resolved-by", default=None,
                    help="who resolved this configuration; required by confirm and recorded in the manifest")
    # THE REGISTERED QUANTITIES A P16 VERDICT IS READ WITH. No default, for the reason the mode has
    # no default: each is an author's registered choice, and the deciding gate refuses a titration
    # that carries none of them rather than let a run pay for arms nothing can score.
    ap.add_argument("--chi-hat-se", type=float, default=None,
                    help="p16: the registered calibration uncertainty on the located boundary")
    ap.add_argument("--slope-equivalence", type=float, default=None,
                    help="p16: the registered equivalence band on the sealed slope")
    ap.add_argument("--informative-horizon", type=int, default=None,
                    help="p16: the registered rounds after the switch beyond which silence is informative")
    ap.add_argument("--practical-absence-band", type=float, default=None,
                    help="p16: the registered band about zero inside which a margin is practically nil")
    ap.add_argument("--across-window-segments", type=int, default=None,
                    help="p16: the registered resolution the across-window predicate is read at")
    a = ap.parse_args(argv)
    # A sealed record is never replaced in place, so an --out path that already exists is refused
    # HERE, before any compute, rather than at the moment of writing. The write is create-only either
    # way; the difference is whether the refusal arrives before the run is paid for or after it, and
    # a refusal after the run has spent its calls has cost the whole run to say the same thing.
    if getattr(a, "out", None) and os.path.exists(str(a.out)):
        sys.exit("--out %s already exists; choose a new path, a sealed record is never overwritten"
                 % a.out)

    m = MODE.ExecutionMode({"demonstrate": "demonstration", "smoke": "smoke",
                            "pilot": "pilot", "confirm": "confirmatory"}[a.mode])
    inputs = None
    # The ladder object this run actually scored with, kept so that the custody check below can
    # recompute its verifier binding rather than read the manifest's own record of it back. A P16
    # demonstration reads a simulated margin and has no ladder at all, which stays None and is
    # reported as a check that was not performed.
    live_ladder = None
    controller = _controller(a)          # the object the paid modes reserve against, built once

    if a.experiment == "p5":
        if m is MODE.ExecutionMode.DEMONSTRATION:
            ad = adapters.MockCouplingAdapter(beta=a.beta, theta=a.theta)
            # The mock world grows at the design's rate, so a 128-round window outruns any ladder that
            # also measures the bank precisely. The mock runs the window the end-to-end tests proved;
            # a registered run sizes its ladder to the trajectory it will read, which is ruling 29's
            # headroom condition and is checked by the runner rather than assumed.
            lad = L.MockLadder(n_items=20000, scale=400.0)
            live_ladder = lad
            place = lambda s: {"kind": "mock", "capability": float(s), "rounds": 0}
            start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
            cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
        else:
            from . import code_domain as CD
            cfg = p5.P5Config()
            if a.states:
                cfg.states = tuple(int(s) for s in a.states.split(","))
            if a.window:
                cfg.window_end = a.window
                cfg.checkpoints = tuple(d for d in (4, 8, 16, 32, 64, 128, 256) if d <= a.window)
            pool, lad, store = _pool_and_parts(a, need_registered_pool=m in (MODE.ExecutionMode.PILOT,
                                                                            MODE.ExecutionMode.CONFIRMATORY))
            live_ladder = lad
            place = CD.place_at_state_factory(store)
            start = CD.start_for_factory(store)
            if m is MODE.ExecutionMode.CONFIRMATORY:
                # THE GATE RUNS HERE, before the adapter is constructed and therefore before the first
                # paid call. run_p5 gates again on the objects it actually holds; this one exists so
                # that an operator learns what is missing without an API key in the environment.
                inputs = MODE.ConfirmatoryInputs(
                    ladder=lad, place_at_state=place, start_for=start, checkpoint_store=store,
                    states=tuple(cfg.states), config=cfg, allowance=controller, anchor=_anchor(a),
                    config_resolution={"resolved_by": a.resolved_by, "resolved_utc": _utc(),
                                       "states": list(cfg.states), "window_end": cfg.window_end})
                _gate(inputs)
            ad = _paid_adapter(a, m, controller)
        res = p5.run_p5(ad, lad, cfg, a.seed, place, start, ["S1", "S2", "S3"], mode=m,
                        confirmatory_inputs=inputs, bundle=a.out, anchor=_anchor(a),
                        attestation=_attestation(a))
    else:
        # The adapter the arms ran against, where there is one. A demonstration reads a simulated
        # margin and has no provider at all, and the bundle says so rather than naming one.
        arm_adapter = None
        # Typed quantities are applied in every mode. The demonstration below is the one path that
        # also supplies a value nobody typed, and it names every one it supplies as a candidate; on
        # the three paid paths an unsupplied quantity stays unset, which is what the deciding gate
        # refuses and what leaves a rehearsal exactly as it was.
        registered = {name: getattr(a, name) for name in P16_REGISTERED_OPTIONS
                      if getattr(a, name) is not None}
        if m is MODE.ExecutionMode.DEMONSTRATION:
            # THE DEMONSTRATION IS THE ONE PATH THAT IS GIVEN NUMBERS, AND IT LABELS THEM. On the
            # shipped defaults this command reported NOT EVALUABLE with four components NOT SUPPLIED,
            # so the branch it exists to show was never shown. The numbers are the candidates
            # p16_calibration.py measures the decision family under, they are named as candidates in
            # the sealed configuration, in the verdict block and in the summary below, and the
            # deciding gate refuses a titration that carries the label. Anything typed replaces a
            # candidate and stops being one.
            cfg = p16.demonstration_config(**registered)
            # AND IT READS THE SIMULATED SYSTEM ON THE REGISTERED COORDINATE. `mock_margin_source`
            # supplies no delivery record, no capability and no recursive coordinate, so delivery,
            # realised-exposure, location and slope-magnitude are NOT SUPPLIED in it whatever numbers
            # are set and no run on it can reach a result. `mock_balance_source` is the simulation the
            # module documents as the one that demonstrates the registered coordinate. It is a
            # simulation either way, it says so in its own declaration, and it decides nothing.
            src, sha, name = (p16.mock_balance_source(cfg, true_alpha_crit=cfg.alpha_crit_hat),
                              "none-mock", "mock")
        else:
            cfg = p16.P16Config()
            for name, value in registered.items():
                setattr(cfg, name, value)
            from . import code_domain as CD
            pool, lad, store = _pool_and_parts(a, need_registered_pool=m in (MODE.ExecutionMode.PILOT,
                                                                            MODE.ExecutionMode.CONFIRMATORY))
            live_ladder = lad
            if m is MODE.ExecutionMode.CONFIRMATORY:
                inputs = MODE.ConfirmatoryInputs(
                    ladder=lad, place_at_state=CD.place_at_state_factory(store),
                    start_for=CD.start_for_factory(store), checkpoint_store=store, states=(), config=cfg,
                    allowance=controller, anchor=_anchor(a),
                    config_resolution={"resolved_by": a.resolved_by, "resolved_utc": _utc(),
                                       "horizon": cfg.horizon, "switch_round": cfg.switch_round,
                                       # AND WHICH REGISTERED QUANTITIES THIS RUN WAS RESOLVED WITH.
                                       # A record of a deciding run that does not say which bands its
                                       # verdict was read under cannot be re-scored against them.
                                       "registered_quantities": dict(registered)})
                _gate(inputs)
            ad = _paid_adapter(a, m, controller)
            arm_adapter = ad
            dose = CD.DoseSchedule()
            src = CD.suite_margin_source(ad, lad, store, "Improve the solutions library.",
                                         dose.dose_for, cfg.switch_round)
            sha, name = lad.sha256, ad.name
        # THE LADDER AND THE ADAPTER GO IN, not only their hash and their name. A real P16 arm reads
        # the frozen pool once per round and the adapter keeps its own account of what it served, and
        # a bundle written without these two carries neither. The demonstration passes None for both,
        # which is the truth about it: it reads a simulated margin and has no provider.
        res = p16.run_p16(src, cfg, a.seed, ladder_sha256=sha, adapter_name=name, mode=m,
                          confirmatory_inputs=inputs, bundle=a.out, anchor=_anchor(a),
                          attestation=_attestation(a), ladder=live_ladder, adapter=arm_adapter)

    man = res["manifest"]
    print(man["mode_label"])                       # the label leads the summary, never trails it
    # And P16's provisional label leads it too (finding A2): a single run is not a completed finding,
    # and a reader who has to reach the `provisional` field to learn that will quote the verdict
    # without it.
    if (res.get("verdicts") or {}).get("provisional_label"):
        print(res["verdicts"]["provisional_label"])
    # AND THE CANDIDATE LABEL LEADS IT TOO, for the same reason and in the same place. A result read
    # with a band nobody registered is not a weaker result, it is a result about a rule this package
    # chose, and a reader who has to reach the configuration to learn that will quote it without.
    if (res.get("verdicts") or {}).get("candidate_label"):
        print(res["verdicts"]["candidate_label"])
    # AND P5'S IDENTIFICATION LEADS THE SUMMARY FOR THE SAME REASON (finding A6). The two regression
    # directions of one bank can agree closely and identify nothing, and a reader who sees only a
    # small route gap in the printed record will read it as an identification unless the run says in
    # words what is missing.
    ident = (res.get("verdicts") or {}).get("identification_judgement") or {}
    if ident.get("label"):
        print("IDENTIFICATION %s: %s" % (ident["label"], ident.get("reason", "")))
        if not ident.get("n_manipulations_supplied"):
            # AND THE COMMAND LINE SAYS WHAT IT CANNOT DO. A second capability manipulation is a
            # domain object carrying a placement loader, so there is no option here that could
            # supply one, and IDENTIFIED is therefore unreachable from this entry point in every
            # mode. A reader who sees NOT ESTABLISHED on every run is entitled to know whether that
            # is a finding about the bank or a fact about the way the run was started.
            print("  this command line supplies no capability manipulation, so IDENTIFIED is not "
                  "reachable from it in any mode: a second placement channel is a domain object "
                  "with its own loader and it is passed to arc_runner.p5.run_p5 by a caller that "
                  "holds one")
    # AND P5'S DENOMINATOR IS PRINTED WITH ITS RESULT (finding A7). A panel result read without the
    # denominator it was decided on is the defect itself: a panel of five of which three reached the
    # ladder ceiling used to leave a majority of two, and the printed word was the same word.
    panel = (res.get("verdicts") or {}).get("panel") or {}
    if panel.get("assigned"):
        print("PREDICTION %s on the assigned panel of %d: %d agree, %d disagree, %d unresolved, "
              "%d not evaluable" % (panel["result"], panel["assigned"], panel["agrees"],
                                    panel["disagrees"], panel["unresolved"], panel["not_evaluable"]))
        if panel.get("n_unassigned_systems_present"):
            # The population may not grow either: a system scored without being on the sealed panel
            # is reported here rather than being left for a reader to find by comparing two lists.
            print("  and %d system(s) scored but not on the sealed panel, reported and not counted: %s"
                  % (panel["n_unassigned_systems_present"],
                     ", ".join(panel["unassigned_systems_present"])))
        if not panel.get("denominator_frozen", True):
            # WHERE THE DENOMINATOR CAME FROM, when it was not a commitment. A reader of the line
            # above cannot tell a frozen panel from one read off the systems that produced a fit, and
            # the second is the population finding A7 forbids deciding on.
            print("  the denominator was not frozen before the run: %s"
                  % panel.get("denominator_source", "source not recorded"))
            if panel.get("denominator_note"):
                print("  %s" % panel["denominator_note"])
    # BOTH NAMES FOR THE READINGS ARE IN THE FILTER, and the filter is a membership test rather than
    # a lookup: a summary built from a fixed key name silently empties itself the day the export
    # renames one, which is exactly what a reader would not notice.
    summary = {k: v for k, v in res.items()
               if k in ("routes", "verdicts", "diagnostics", "evidence_status", "sealed")}
    summary["manifest"] = {k: man[k] for k in ("experiment", "execution_mode", "simulated", "pilot",
                                               "created_utc", "code_sha256", "ladder_sha256")}
    seal = man.get("seal") or {}
    rec = seal.get("anchor_receipt") or {}
    summary["sealed_at"] = seal.get("sealed_at_utc")
    # The receipt goes in the summary beside the sealing time, because the local time is the runner's
    # own word and the receipt is the part somebody else can check. A mock says so here too.
    summary["anchor"] = {"identifier": rec.get("anchor_identifier"), "service": rec.get("service"),
                         "mock": bool(rec.get("mock")), "attests_sha256": rec.get("attests_sha256")}
    try:
        # The summary carries the answer to the only question that matters about a run: may it be
        # read at proposition level? A refusal is printed in the summary rather than left to the
        # reader to infer from the mode, because inference is what finding A9 is about.
        #
        # The live predictions and the live configuration go in with it (finding A8). This is the
        # scoring moment, and at the scoring moment the question is not only which mode the run was
        # in: it is whether the predictions in hand are the predictions that were sealed and whether
        # the configuration is still the one the seal was made under.
        live_predictions = (res.get("heldout") or {}).get("sealed_predictions") or res.get("sealed")
        summary["custody"] = M.require_scoreable(man, predictions=live_predictions, config=cfg,
                                                 ladder=live_ladder)
        summary["proposition_level"] = "may be read at proposition level"
    except RuntimeError as exc:              # NotScoreable, PilotNotScoreable, CustodyRefusal, no seal
        summary["proposition_level"] = "REFUSED: %s" % exc
    print(json.dumps(summary, indent=1, default=float))
    if a.out:
        # A run given --out always writes its bundle, in both acts: the seal reached the directory
        # at the moment of sealing, and this is the complete record.
        print("evidence bundle written", res.get("bundle_path", a.out), file=sys.stderr)
        print("  reload it with arc_runner.custody.load_bundle and recompute the verdicts with "
              "arc_runner.custody.recompute_verdicts", file=sys.stderr)
    return 0


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _gate(inputs: MODE.ConfirmatoryInputs) -> None:
    """Turn the gate's refusal into the command line's own exit, with every requirement named.

    One requirement is postponed here and nowhere else. This pre-flight runs before the adapter is
    constructed, so that an operator learns what is missing without a provider key in the
    environment, which means the system that will do the observing does not yet exist to be asked
    what it is. `run_p5` and `run_p16` ask it of the object they hold, so the question is answered on
    this run and not skipped; naming it in the call is what keeps that a decision rather than a gap.
    """
    try:
        MODE.require_confirmatory_inputs(inputs, deferred=("observing-system:",))
    except MODE.ModeRefusal as exc:
        raise SystemExit(str(exc))


if __name__ == "__main__":
    sys.exit(main())
