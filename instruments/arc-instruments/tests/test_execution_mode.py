"""The demonstration-to-confirmation boundary: finding A9's acceptance cases.

Every test here is about one question: can a run that is not an instrument be read as though it were?
The answer must be no in four separate ways, and yes exactly once, for a setup that holds everything
the registration names. The last part matters as much as the first four: a gate that refuses every
setup is not a gate, it is an outage, and it passes refusal tests perfectly.

The adapter used by the refusal cases raises if anything asks it to revise an artefact. That is how
"fail before the first paid call" is tested rather than asserted: a refusal that arrives after one
call would fail the test with the words "a provider call was made".
"""
import json
import sys

import numpy as np
import pytest

from arc_instruments import verdicts as V
from arc_runner import adapters, code_domain as CD, cli, custody, ladder as L, manifest as M, mode as MODE, p5, p16
from arc_runner import budget as B, observation as OBS, p16_components as C
from arc_runner import p16_sequential as SEQ


# --------------------------------------------------------------------------------------------------
# Fixtures: a registered domain pool, a system that writes code, and an adapter that must not be called
# --------------------------------------------------------------------------------------------------

def _lib(n_correct, n_total):
    lines = []
    for i in range(n_total):
        body = "return x + y" if i < n_correct else "return None"
        lines.append("def add_%d(x, y):\n    %s" % (i, body))
    return "\n".join(lines)


def _registered_pool(n=60):
    """A pool that stands in for one a person wrote for the study: same shape as the reference pool,
    but not marked smoke-only, so the gate has something it is allowed to accept."""
    tasks = []
    for i in range(n):
        name = "add_%d" % i
        checks = ("assert %s(3, 4) == 7" % name, "assert %s(0, 0) == 0" % name, "assert %s(-1, 1) == 0" % name)
        tasks.append(CD.Task(id=name, statement="Define %s(x, y) returning the sum." % name,
                             signature="def %s(x, y): ..." % name, shown_examples=(checks[0],), checks=checks))
    return CD.TaskPool(tasks, name="registered-domain-pool")


class RefusingAdapter:
    """A provider that has not been paid, and says so if anyone tries."""
    name = "must-not-be-called"

    def revise(self, artefact, retained, task, rng):
        raise AssertionError("a provider call was made after a confirmatory run should have refused")


class ScriptedCodeSystem:
    """A system whose revisions are real code and whose growth follows the registered law, so that the
    one passing case exercises the whole path rather than a stub.

    It declares itself NOT a simulation, for the reason `_stub_anchor` returns the shape an external
    service returns: the deciding gate now asks every observing system what it is and refuses a
    simulation and a silence alike, so a stand-in that declared nothing would stop this case short of
    the path it exists to exercise. The declaration is what the gate reads and what the manifest
    records, and it is the caller's statement about the caller's own apparatus in exactly the way the
    anchor and the attestation are.
    """
    name = "scripted-code"

    def __init__(self, n_total, beta=0.5, a=1.6):
        self.n_total, self.beta, self.a = n_total, beta, a
        self.calls = 0

    def metadata(self):
        return {"adapter": self.name, "simulated": False, "calls": self.calls,
                "returned_models": ["scripted-code-v1"], "usage_totals": {}, "records": []}

    def revise(self, artefact, retained, task, rng):
        self.calls += 1
        done = artefact.get("text", "").count("return x + y")
        available = max(float(retained.get("fraction", 1.0)) * max(done, 1), 1e-9)
        new_done = min(self.n_total, done + max(1, int(round(self.a * available ** self.beta))))
        out = dict(artefact)
        out["text"] = _lib(new_done, self.n_total)
        out["rounds"] = int(artefact.get("rounds", 0)) + 1
        return out


def _store(tmp_path, n_total, states=(10, 20, 40)):
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(4, n_total)))
    for s in states:
        store.save(CD.state_name(s), CD.new_artefact(_lib(int(s), n_total)))
    return store


def _cfg(states=(10, 20, 40)):
    return p5.P5Config(states=states, fractions=(0.5, 1.0), reps=2, window_end=8, checkpoints=(4, 8),
                       calibration_depths=(1, 2, 3, 4), cal_reads=2, reads=1, heldout_reads=1,
                       replicates=2, bootstrap=20, control_reads_multiplier=2)


def _resolution():
    return {"resolved_by": "the operator", "resolved_utc": "2026-09-05T00:00:00Z"}


def _stub_anchor(digest):
    """A stand-in for the operator's anchoring service. It is NOT a mock receipt: it returns the shape
    an external service returns, so that the gate and `require_scoreable` are exercised on the path a
    deciding run actually takes. Finding A8 added the anchoring service to the confirmatory inputs,
    so every confirmatory case here supplies one."""
    return custody.receipt("stub-anchor:%s" % digest[:12], digest, service="test-stub")


_stub_anchor.anchor_service = "test-stub"


def _stub_attestation(heldout_sha256="9" * 64):
    """A stand-in for the sentence a named party signs before the seal, and NOT the runner's
    placeholder: finding A8's third requirement is that somebody says the material deciding the
    predictions was unseen when they were fixed, and nothing computable can say it. Every
    confirmatory case here supplies one for the same reason it supplies an anchoring service."""
    return custody.attestation("the custodian of the held-out material", heldout_sha256,
                               material="the frozen hidden suite this run reads")


# The deciding path requires a controller and not a bare figure, so the default here is one: a
# BudgetController is what reserves the per-call maximum before each dispatch and halts the run when
# the remainder cannot cover the next call. A fresh one per call, because a controller carries a
# ledger and two cases must not spend each other's. `None` still means no allowance at all, which is
# a separate refusal and has its own case below.
_DEFAULT_ALLOWANCE = "a fresh controller"


def _controller(limit_gbp=12.0):
    return B.BudgetController(B.Allowance(limit_gbp=limit_gbp))


def _inputs(store, allowance=_DEFAULT_ALLOWANCE, resolution=None, anchor=_stub_anchor,
            systems=None):
    allowance = _controller() if allowance is _DEFAULT_ALLOWANCE else allowance
    return MODE.ConfirmatoryInputs(checkpoint_store=store, allowance=allowance, anchor=anchor,
                                   attestation=_stub_attestation(),
                                   observing_systems=((ScriptedCodeSystem(40),) if systems is None
                                                      else systems),
                                   config_resolution=_resolution() if resolution is None else resolution)


def _mock():
    ad = adapters.MockCouplingAdapter(beta=0.5)
    lad = L.MockLadder(n_items=2000, scale=400.0)
    place = lambda s: {"kind": "mock", "capability": float(s), "rounds": 0}
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    cfg = p5.P5Config(reps=2, window_end=16, checkpoints=(4, 8, 16), bootstrap=20, control_reads_multiplier=2)
    return ad, lad, cfg, place, start


# --------------------------------------------------------------------------------------------------
# Acceptance case 1: a demonstration says what it is and cannot be scored
# --------------------------------------------------------------------------------------------------

def test_a_demonstration_labels_its_manifest_and_require_scoreable_refuses_it():
    ad, lad, cfg, place, start = _mock()
    res = p5.run_p5(ad, lad, cfg, 3, place, start, ["S1"])
    man = res["manifest"]
    assert man["execution_mode"] == "demonstration"
    assert man["simulated"] is True and man["scoreable_at_proposition_level"] is False
    assert "simulated recovery" in man["mode_label"]
    # the verdicts still exist, because recovering a known coupling is the proof the pipeline works,
    # and they carry the sentence that says what they are
    assert res["verdicts"]["execution_mode"] == "demonstration"
    assert "simulated recovery" in res["verdicts"]["interpretation"]
    assert res["verdicts"]["scoreable_at_proposition_level"] is False
    with pytest.raises(M.NotScoreable):
        M.require_scoreable(man)


def test_an_unstated_mode_resolves_to_a_demonstration_and_never_to_the_deciding_path():
    assert MODE.resolve(None) is MODE.ExecutionMode.DEMONSTRATION
    assert MODE.resolve(None, pilot=True) is MODE.ExecutionMode.PILOT
    assert MODE.resolve("confirmatory") is MODE.ExecutionMode.CONFIRMATORY
    with pytest.raises(MODE.ModeRefusal):
        MODE.resolve("confirmatory", pilot=True)          # one run, one mode, stated once


def test_a_pilot_carries_its_mode_and_is_refused_by_both_gates():
    ad, lad, cfg, place, start = _mock()
    res = p5.run_p5(ad, lad, cfg, 3, place, start, ["S1"], mode="pilot")
    man = res["manifest"]
    assert man["execution_mode"] == "pilot" and man["pilot"] is True
    assert "verdicts" not in res
    with pytest.raises(M.PilotNotScoreable):
        M.require_reportable(man)
    with pytest.raises(M.PilotNotScoreable):
        M.require_scoreable(man)


def test_a_smoke_run_is_labelled_and_refused_at_proposition_level():
    cfg = p16.P16Config(systems_per_arm=1, horizon=30)
    res = p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2.0), cfg, 5, "none", "mock", mode="smoke")
    assert res["manifest"]["execution_mode"] == "smoke"
    assert "wiring" in res["verdicts"]["interpretation"]
    with pytest.raises(M.NotScoreable):
        M.require_scoreable(res["manifest"])


# --------------------------------------------------------------------------------------------------
# Acceptance cases 2 to 4: a confirmatory run refuses a missing input before the first paid call
# --------------------------------------------------------------------------------------------------

def _requirement_names(exc):
    return [r.split(":", 1)[0] for r in exc.value.requirements]


def test_confirmatory_with_the_reference_pool_refuses(tmp_path):
    pool = CD.reference_pool(40)                     # the smoke pool, which nobody wrote for the study
    lad = CD.SuiteLadder(pool, subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, 40)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store))
    assert "domain-ladder" in _requirement_names(exc)
    assert lad.smoke_only is True and lad.spec["pool_smoke_only"] is True


def test_confirmatory_with_a_missing_state_refuses(tmp_path):
    pool = _registered_pool(40)
    lad = CD.SuiteLadder(pool, subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, 40, states=(10, 20))    # the bank will also place 40, which was never built
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(states=(10, 20, 40)), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store))
    assert "checkpoint-store" in _requirement_names(exc)
    assert "40" in " ".join(exc.value.requirements)


def test_confirmatory_without_an_allowance_refuses(tmp_path):
    pool = _registered_pool(40)
    lad = CD.SuiteLadder(pool, subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, 40)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store, allowance=None))
    assert "spending-allowance" in _requirement_names(exc)
    # an exhausted controller is refused for the same reason a missing allowance is
    ctrl = B.BudgetController(B.Allowance(limit_gbp=0.0))
    assert MODE.missing_confirmatory_inputs(MODE.ConfirmatoryInputs(allowance=ctrl))


def test_confirmatory_without_a_resolved_configuration_refuses(tmp_path):
    pool = _registered_pool(40)
    lad = CD.SuiteLadder(pool, subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, 40)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store, resolution={"resolved_by": "", "resolved_utc": ""}))
    assert "resolved-configuration" in _requirement_names(exc)


def test_confirmatory_with_placeholder_loaders_refuses(tmp_path):
    """The defect A9 names: a loader that manufactures an artefact from a number, so that no cell is
    at the state it claims. It is refused whatever else is in place."""
    pool = _registered_pool(40)
    lad = CD.SuiteLadder(pool, subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, 40)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1,
                  lambda s: CD.new_artefact(""), lambda name: CD.new_artefact(""), ["S1"],
                  mode="confirmatory", confirmatory_inputs=_inputs(store))
    assert "checkpoint-loaders" in _requirement_names(exc)


def test_a_confirmatory_run_with_no_inputs_at_all_names_every_requirement(tmp_path):
    pool = _registered_pool(40)
    lad = CD.SuiteLadder(pool, subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, 40)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory")
    names = set(_requirement_names(exc))
    # THE CHECKPOINT STORE IS NOT AMONG THEM, AND THAT IS THE REPAIR RATHER THAN A GAP. The loaders
    # here are the real factories, so they declare the store they read and the runner writes that
    # store into the gate's copy: this run does hold a complete checkpoint bank, and the gate checks
    # the bank the loaders read rather than a field the caller happened to fill in. A run whose
    # loaders declare no store either is still told the bank is missing, which is the second half of
    # this case.
    assert {"resolved-configuration", "spending-allowance", "anchor-service"} <= names
    assert "checkpoint-store" not in names
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1, lambda s: CD.new_artefact(""),
                  lambda name: CD.new_artefact(""), ["S1"], mode="confirmatory")
    assert "checkpoint-store" in _requirement_names(exc)


def test_p16_confirmatory_refuses_without_a_store_and_names_the_seed(tmp_path):
    cfg = p16.P16Config(systems_per_arm=1, horizon=20)

    def must_not_be_called(arm, alpha_arm, r, rng):
        raise AssertionError("a provider call was made after a confirmatory run should have refused")

    # The source is DECLARED because this case is about the mode gate and not about the assay. P16's
    # deciding path now asks what the source measures before it asks whether the apparatus is present
    # (see p16.run_p16 and tests/test_p16_observation.py), so an undeclared source here would be
    # refused on the assay and this case would stop naming the requirement it exists to name.
    OBS.declare(must_not_be_called, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=0.5, source="a-declared-source"))

    with pytest.raises(MODE.ModeRefusal) as exc:
        p16.run_p16(must_not_be_called, cfg, 5, "sha", "adapter", mode="confirmatory",
                    confirmatory_inputs=MODE.ConfirmatoryInputs(allowance=B.Allowance(limit_gbp=5.0),
                                                                config_resolution=_resolution()))
    assert "checkpoint-store" in _requirement_names(exc)


# --------------------------------------------------------------------------------------------------
# The case that keeps the gate honest: a complete setup runs and is scoreable
# --------------------------------------------------------------------------------------------------

def test_a_complete_confirmatory_setup_passes_the_gate_and_is_scoreable(tmp_path):
    # A hundred and twenty items with a forty-item read leaves the held-out panel headroom to the
    # final checkpoint, so this case exercises the bank, the seal, the continuation and the verdict
    # rather than stopping at NOT EVALUABLE. The verdict itself is not the subject: the gate is.
    n = 120
    pool = _registered_pool(n)
    lad = CD.SuiteLadder(pool, subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, n)
    inputs = _inputs(store)
    res = p5.run_p5(ScriptedCodeSystem(n), lad, _cfg(), 11, CD.place_at_state_factory(store),
                    CD.start_for_factory(store), ["S1"], mode="confirmatory", confirmatory_inputs=inputs)
    man = res["manifest"]
    assert man["execution_mode"] == "confirmatory"
    assert man["simulated"] is False and man["scoreable_at_proposition_level"] is True
    M.require_scoreable(man)                                   # the one mode that passes this gate
    assert res["verdicts"]["PREDICTION"] in ("SUPPORTED", "REFUTED", "INCONCLUSIVE", "NOT EVALUABLE",
                                             "UNRESOLVED")
    # the manifest records what the gate checked, so that a reader does not have to trust the run
    rec = man["confirmatory_inputs"]
    assert rec["ladder_sha256"] == lad.sha256 and rec["ladder_smoke_only"] is False
    assert rec["allowance"]["allowance_gbp"] == pytest.approx(12.0)
    assert rec["config_resolution"]["resolved_by"] == "the operator"
    assert rec["states"] == [10.0, 20.0, 40.0]


# --------------------------------------------------------------------------------------------------
# The apparatus question: custody is about the record, and this is about what was measured
# --------------------------------------------------------------------------------------------------


class UndeclaredSystem:
    """A system that says nothing about itself, which is what almost every object is."""
    name = "says-nothing"

    def revise(self, artefact, retained, task, rng):
        raise AssertionError("a provider call was made after a confirmatory run should have refused")


def test_a_deciding_run_refuses_a_simulated_system_and_one_that_declares_nothing(tmp_path):
    """THE FLOOR THIS PACKAGE ONCE CARRIED, IN THE SHAPE THAT LEAVES THE DECIDING PATH REACHABLE.

    An earlier manifest module refused confirmatory collection outright: the real assay is not
    released, so no run here may issue an empirical verdict. What replaced it is custody, which asks
    a different question and answers it completely: is this record the record that was sealed, and
    did somebody outside this code attest it. A simulated system read through a checked ladder, with
    a genuine receipt and a named attestation, satisfies every part of custody, and until this case
    existed the only thing that could reach a confirmatory verdict here was a simulation: the
    remote-endpoint refusal bars a real endpoint on the ground that no assay is released, and nothing
    barred a local object that models the world.

    Both refusals arrive before the first call, which is why the two systems below raise if anything
    asks them to revise an artefact.
    """
    n = 60
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, n)

    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(adapters.MockCouplingAdapter(beta=0.5), lad, _cfg(), 1,
                  CD.place_at_state_factory(store), CD.start_for_factory(store), ["S1"],
                  mode="confirmatory", confirmatory_inputs=_inputs(store, systems=()))
    assert "observing-system" in _requirement_names(exc)
    assert "declares itself a simulation" in " ".join(exc.value.requirements)

    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(UndeclaredSystem(), lad, _cfg(), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store, systems=()))
    assert "declares nothing about what it is" in " ".join(exc.value.requirements)

    # AND THE HELD-OUT PANEL IS ASKED TOO, because it decides the verdict as much as the bank does:
    # a real bank whose held-out systems are simulated is a comparison against a model.
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(ScriptedCodeSystem(n), lad, _cfg(), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  heldout_adapter=adapters.MockCouplingAdapter(beta=0.9),
                  confirmatory_inputs=_inputs(store, systems=()))
    assert "mock-coupling" in " ".join(exc.value.requirements)

    # and the titration asks the same question of its margin source
    cfg = p16.P16Config(systems_per_arm=1, horizon=24, dose_offsets=(-0.3, 0.3))
    simulated = OBS.declare(_declared_source("a-simulated-source"),
                            OBS.log_service_ratio_observation(supplies_q_and_w=True, chi_hat=0.5,
                                                              simulated=True,
                                                              source="a-simulated-source"))
    with pytest.raises(OBS.ObservationRefusal):
        p16.run_p16(simulated, cfg, 5, "sha", "an-adapter", mode="confirmatory",
                    confirmatory_inputs=_complete_p16_inputs(tmp_path / "p16", cfg))


def test_perfect_custody_over_a_simulated_system_still_refuses_at_proposition_level():
    """The case that shows custody and release are two questions rather than one.

    Nothing about this record is wrong. The seal verifies, the specification hash recomputes, the
    receipt is a real one from a service and it still attests the digest, and a named party has
    stated what was unseen. `custody_failures` is empty, which is the correct answer to the question
    custody asks. The run is still not evidence about any system, because the thing it measured
    models the world, and the record says so in the field the seal covers.
    """
    man = M.new_manifest("P5", False, "a-ladder-hash", {"seed": 1}, "mock", mode="confirmatory",
                         confirmatory_inputs=MODE.ConfirmatoryInputs(
                             observing_systems=(adapters.MockCouplingAdapter(),)).as_record())
    M.seal_predictions(man, {"S1": {"predicted_exponent": 0.5}}, "test",
                       anchor=_stub_anchor, attestation=_stub_attestation())
    assert custody.custody_failures(man, external_anchor_required=True) == []
    with pytest.raises(M.InstrumentNotReleased) as exc:
        M.require_scoreable(man)
    assert "declares itself a simulation" in str(exc.value)

    # a record that names no system at all is refused in its own words rather than passed
    bare = M.new_manifest("P5", False, "a-ladder-hash", {"seed": 2}, "mock", mode="confirmatory")
    M.seal_predictions(bare, {"S1": {"predicted_exponent": 0.5}}, "test",
                       anchor=_stub_anchor, attestation=_stub_attestation())
    with pytest.raises(M.InstrumentNotReleased) as exc:
        M.require_scoreable(bare)
    assert "does not name the system it measured" in str(exc.value)

    # AND THE DECLARATION CANNOT BE IMPROVED AFTER THE SEAL. It is written before `seal_predictions`
    # and is inside the specification hash, so a record edited to say the simulator was a real system
    # fails custody instead of passing the release check.
    promoted = json.loads(json.dumps(man))
    promoted["confirmatory_inputs"]["observing_systems"][0]["simulated"] = False
    failures = custody.custody_failures(promoted, external_anchor_required=True)
    assert failures and any("specification" in f or "changed" in f for f in failures)


def test_a_declared_system_is_recorded_by_name_in_the_manifest(tmp_path):
    """The other half, which keeps the refusal from being an outage: a system that declares itself
    passes, and what it said is in the record rather than in the caller's memory."""
    n = 120
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, n)
    res = p5.run_p5(ScriptedCodeSystem(n), lad, _cfg(), 11, CD.place_at_state_factory(store),
                    CD.start_for_factory(store), ["S1"], mode="confirmatory",
                    confirmatory_inputs=_inputs(store, systems=()))
    declared = res["manifest"]["confirmatory_inputs"]["observing_systems"]
    assert declared == [{"system": "scripted-code", "declared": True, "simulated": False,
                         "declaration": "adapter-metadata"}]
    M.require_scoreable(res["manifest"])


def test_an_inconsistent_configuration_is_not_a_resolved_one(tmp_path):
    store = _store(tmp_path, 40)
    cfg = _cfg()
    cfg.window_end = 16                                        # the final checkpoint is still 8
    bad = MODE.ConfirmatoryInputs(checkpoint_store=store, allowance=_controller(1.0),
                                  config_resolution=_resolution(), config=cfg, states=cfg.states,
                                  place_at_state=CD.place_at_state_factory(store),
                                  start_for=CD.start_for_factory(store), ladder=_ladder_for(store))
    assert any("window" in r for r in MODE.missing_confirmatory_inputs(bad))


def _ladder_for(store):
    return CD.SuiteLadder(_registered_pool(40), subset_size=20, batch_runner=CD.inprocess_batch_runner())


# --------------------------------------------------------------------------------------------------
# The command line: the mode is the command, and the refusal precedes the adapter
# --------------------------------------------------------------------------------------------------

def test_the_command_line_refuses_a_confirmatory_run_before_it_builds_an_adapter(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_RUNNER_API_KEY", raising=False)    # no key: the adapter would refuse to construct
    with pytest.raises(SystemExit) as exc:
        cli.main(["confirm", "p5"])
    assert "--checkpoints" in str(exc.value)
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(1, 4)))
    with pytest.raises(SystemExit) as exc:
        cli.main(["confirm", "p5", "--checkpoints", str(tmp_path)])
    assert "--pool-module" in str(exc.value) and "smoke" in str(exc.value)


def test_the_command_line_has_no_flag_that_turns_a_rehearsal_into_a_deciding_run():
    with pytest.raises(SystemExit):
        cli.main(["p5", "--mock"])                             # the old shape is gone, not aliased


# --------------------------------------------------------------------------------------------------
# The gate checks the bank the loaders read, and it asks the loaders what they return
#
# Three ways a deciding run reached a paid provider with placement it could not honour, each of which
# passed the first version of this gate and each of which is now refused before the first call.
# --------------------------------------------------------------------------------------------------

def test_the_gate_checks_the_store_the_loaders_read_and_not_a_second_one(tmp_path):
    """A complete store beside loaders bound to an incomplete one.

    Everything else about this setup is correct: a registered pool, a resolved configuration, an
    allowance and an anchoring service, and a `checkpoint_store` that holds every state the bank will
    place. The loaders read a DIFFERENT store, which holds state-10 and state-20 and not state-40, so
    the states that were verified are not the states this run would place. The first version of the
    gate validated the store it was handed and only asked the loaders to declare that they read some
    store, so this setup passed, reached the provider, and failed inside the paid bank with the
    missing checkpoint the gate exists to catch.
    """
    n = 40
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=20, batch_runner=CD.inprocess_batch_runner())
    complete = _store(tmp_path / "complete", n, states=(10, 20, 40))
    partial = _store(tmp_path / "partial", n, states=(10, 20))
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(states=(10, 20, 40)), 1,
                  CD.place_at_state_factory(partial), CD.start_for_factory(partial), ["S1"],
                  mode="confirmatory", confirmatory_inputs=_inputs(complete))
    assert "checkpoint-loaders" in _requirement_names(exc)
    assert "partial" in " ".join(exc.value.requirements)
    # and the same setup with the loaders bound to the store that was checked is not refused for this
    assert not [r for r in MODE.missing_confirmatory_inputs(
        MODE.ConfirmatoryInputs(ladder=lad, checkpoint_store=complete, states=(10, 20, 40),
                                config=_cfg(states=(10, 20, 40)), config_resolution=_resolution(),
                                allowance=_controller(), anchor=_stub_anchor,
                                observing_systems=(ScriptedCodeSystem(n),),
                                place_at_state=CD.place_at_state_factory(complete),
                                start_for=CD.start_for_factory(complete)))]


def test_the_two_loaders_may_not_read_two_different_banks(tmp_path):
    """The bank and the held-out panel read one checkpoint bank or they are not one experiment."""
    n = 40
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=20, batch_runner=CD.inprocess_batch_runner())
    one, two = _store(tmp_path / "one", n), _store(tmp_path / "two", n)
    refusals = MODE.missing_confirmatory_inputs(
        MODE.ConfirmatoryInputs(ladder=lad, checkpoint_store=one, states=(10, 20, 40), config=_cfg(),
                                config_resolution=_resolution(), allowance=_controller(1.0),
                                anchor=_stub_anchor, place_at_state=CD.place_at_state_factory(one),
                                start_for=CD.start_for_factory(two)))
    assert any("different checkpoint stores" in r for r in refusals)


def test_a_loader_labelled_with_a_store_it_does_not_read_is_still_a_placeholder(tmp_path):
    """The defect A9 names, wearing the label the first version of this gate asked for.

    `place` here is the reference runner's own placeholder, the lambda that manufactures an artefact
    from a number, with the store attribute the gate used to look for attached to it. Under the
    marker rule that setup passed with no requirements outstanding at all, so the acceptance case
    "real confirmation with placeholder loaders must fail before the first paid call" was enforced by
    an opt-in convention rather than by anything about the artefact returned. The gate now reads the
    artefact back and compares it with the bytes the store holds, which a placeholder cannot match
    whatever it is labelled.
    """
    n = 40
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, n)
    place = lambda s: {"kind": "mock", "capability": float(s), "rounds": 0}
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    place.checkpoint_store = store
    start.checkpoint_store = store
    refusals = MODE.missing_confirmatory_inputs(
        MODE.ConfirmatoryInputs(ladder=lad, checkpoint_store=store, states=(10, 20, 40), config=_cfg(),
                                config_resolution=_resolution(), allowance=_controller(1.0),
                                anchor=_stub_anchor, place_at_state=place, start_for=start))
    assert [r for r in refusals if r.startswith("checkpoint-loaders")]
    assert any("no artefact text" in r for r in refusals)
    # and it is refused on the run itself, before the adapter is asked for anything
    with pytest.raises(MODE.ModeRefusal):
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1, place, start, ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store))


def test_a_loader_that_returns_another_state_s_artefact_is_refused(tmp_path):
    """A loader that reads the store and returns the wrong cell from it. It carries the store, it
    raises nothing, and every state it is asked for exists: only the bytes say it is wrong."""
    n = 40
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, n)
    real = CD.place_at_state_factory(store)

    def always_the_lowest_state(target):
        return real(10)                       # every cell placed from one checkpoint, whatever was asked

    always_the_lowest_state.checkpoint_store = store
    refusals = MODE.missing_confirmatory_inputs(
        MODE.ConfirmatoryInputs(ladder=lad, checkpoint_store=store, states=(10, 20, 40), config=_cfg(),
                                config_resolution=_resolution(), allowance=_controller(1.0),
                                anchor=_stub_anchor, place_at_state=always_the_lowest_state,
                                start_for=CD.start_for_factory(store)))
    assert any("did not return the artefact the store holds" in r for r in refusals)
    assert not any("place_at_state(10)" in r for r in refusals)   # that one it did place correctly


# --------------------------------------------------------------------------------------------------
# P16's deciding path exists and is reachable: the gate is not an outage in the shape of a safeguard
# --------------------------------------------------------------------------------------------------

def _declared_assay(cfg, zero=2.0, line_slope=-0.5):
    """A margin source that declares what it measures and supplies Q and W separately.

    It stands in for a real assay in the way `ScriptedCodeSystem` stands in for a real provider and
    `_stub_anchor` stands in for the operator's commitment service: it returns the shape a real one
    returns so that the deciding path is exercised on the path a deciding run actually takes. It is
    not an instrument and no run should ever use it.
    """
    import math

    alpha_base = float(zero) - 0.4
    change_at = int(cfg.switch_round)
    r_change = float(change_at + 1)

    def src(arm, alpha_arm, r, rng):
        R = float(r + 1)
        control = arm in ("sham", "baseline")
        changed = (r >= change_at) and not control
        a = float(alpha_arm) if changed else alpha_base
        delta_base = line_slope * (alpha_base - zero)
        delta_arm = line_slope * ((float(alpha_arm) if changed else alpha_base) - zero)
        log_r0 = math.log(min(R, r_change))
        log_r1 = math.log(R / r_change) if R > r_change else 0.0
        U = math.exp(alpha_base * log_r0 + a * log_r1)
        W = a * U / R
        Q = W * math.exp(delta_base * log_r0 + delta_arm * log_r1)
        return {"round": r, "Q": Q, "W": W, "R": R, "U": U}

    return OBS.declare(src, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=cfg.chi_hat, source="a-declared-test-assay"))


def _registered_p16(**kw):
    """A titration carrying the five quantities its verdict is read with.

    The numbers stand in for registered ones exactly as `_stub_anchor` stands in for the operator's
    anchoring service: they are the shape the deciding path needs, so that the path is exercised
    rather than described. Nothing here blesses these widths. What the gate decides is whether the
    numbers are PRESENT and can decide anything, and the tests below hold it to both.
    """
    base = dict(systems_per_arm=1, horizon=24, dose_offsets=(-0.3, 0.3), chi_hat_se=0.02,
                slope_equivalence=0.15, informative_horizon=10, practical_absence_band=0.05,
                across_window_segments=2)
    base.update(kw)
    return p16.P16Config(**base)


def test_a_complete_p16_confirmatory_setup_passes_the_gate_and_is_scoreable(tmp_path):
    """The P16 half of the case that keeps the gate honest, and it had no test at all.

    One P5-shaped rule set was applied to whatever configuration reached the gate, so every
    `confirm p16` run was refused with four requirements a P16 configuration cannot carry and no flag
    could ever supply: the documented deciding path for P16 was unreachable. A refusal test passes
    perfectly against an outage, which is why this case asserts that a complete setup RUNS.

    The configuration now carries the five registered quantities as well, for the same reason it
    carries a store and an anchor: they are inputs the deciding path requires, and a complete setup
    is one that holds all of them.
    """
    cfg = _registered_p16()
    lad = CD.SuiteLadder(_registered_pool(20), subset_size=10, batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(2, 20)))
    res = p16.run_p16(_declared_assay(cfg), cfg, 5, lad.sha256, "a-declared-adapter",
                      mode="confirmatory", confirmatory_inputs=MODE.ConfirmatoryInputs(
                          ladder=lad, checkpoint_store=store, allowance=_controller(5.0),
                          anchor=_stub_anchor, attestation=_stub_attestation(),
                          config_resolution=_resolution()))
    man = res["manifest"]
    assert man["execution_mode"] == "confirmatory" and man["scoreable_at_proposition_level"] is True
    M.require_scoreable(man)
    assert res["arms"], "a confirmatory P16 run that reaches its arms has arms"
    assert man["confirmatory_inputs"]["ladder_sha256"] == lad.sha256


def test_a_titration_told_about_one_ladder_and_handed_another_refuses(tmp_path):
    """THE LADDER ARGUMENT IS A CLAIM, AND THREE THINGS NOW CHECK IT.

    P5 holds the ladder it scores with, so its gate, its readings and its seal cannot come apart. A
    P16 margin source is built by the caller over its own ladder and the run is only TOLD which one
    that is, and the argument that tells it went in below the gate: a deciding run was approved on
    the ladder its inputs named and sealed the one its argument named, including a simulated ladder
    the gate refuses on its own, with `all_bound` false in the record and no failure anywhere. A
    run's two hands may not name different pools; the hand that names the pool the run binds is the
    one the gate is shown; and where the named ladder records no reading at all while the arms are
    collected, the claim has failed in the one way anything here can observe.
    """
    cfg = _registered_p16()
    lad = CD.SuiteLadder(_registered_pool(20), subset_size=10, batch_runner=CD.inprocess_batch_runner())
    other = CD.SuiteLadder(_registered_pool(24), subset_size=10, batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path / "store"))
    store.save("seed", CD.new_artefact(_lib(2, 20)))

    def _inputs_for(ladder):
        return MODE.ConfirmatoryInputs(ladder=ladder, checkpoint_store=store,
                                       allowance=_controller(5.0), anchor=_stub_anchor,
                                       attestation=_stub_attestation(), config_resolution=_resolution())

    # TWO HANDS, TWO POOLS: refused, and neither is preferred
    with pytest.raises(MODE.ModeRefusal) as exc:
        p16.run_p16(_declared_assay(cfg), cfg, 5, lad.sha256, "a-declared-adapter", mode="confirmatory",
                    ladder=other, confirmatory_inputs=_inputs_for(lad))
    assert "handed two different ladders" in str(exc.value)

    # THE GATE IS SHOWN THE OBJECT THE RUN BINDS. A simulated ladder supplied as the argument used to
    # reach the seal through inputs that named nothing, because the gate read the inputs and the seal
    # read the argument.
    with pytest.raises(MODE.ModeRefusal) as exc:
        p16.run_p16(_declared_assay(cfg), cfg, 5, lad.sha256, "a-declared-adapter", mode="confirmatory",
                    ladder=L.MockLadder(n_items=2000, scale=400.0),
                    confirmatory_inputs=_inputs_for(None))
    assert any("measures a simulated latent capability" in r for r in exc.value.requirements)

    # AND A NAMED LADDER THAT NOTHING READ. `_declared_assay` reads no ladder at all, which is what a
    # margin source built over a different one looks like from here: the arms arrive, the log stays
    # empty, and the verifier binding sealed with the predictions is about another pool. The evidence
    # is written first and the refusal comes after it, so the run leaves what it paid for.
    out = tmp_path / "evidence"
    with pytest.raises(custody.CustodyRefusal) as exc:
        p16.run_p16(_declared_assay(cfg), cfg, 5, lad.sha256, "a-declared-adapter", mode="confirmatory",
                    ladder=lad, bundle=str(out), confirmatory_inputs=_inputs_for(lad))
    assert "does not read the ladder this run named" in str(exc.value)
    bundle = custody.load_bundle(str(out))
    assert bundle["arms"] and bundle["reads"] == []
    assert "is not the object its source reads" in bundle["reads_note"]

    # the same setup with no bundle attaches no log and asks nothing, exactly as P5 does: a run that
    # is not saving evidence does not pay for a list that grows with every read
    res = p16.run_p16(_declared_assay(cfg), cfg, 5, lad.sha256, "a-declared-adapter",
                      mode="confirmatory", ladder=lad, confirmatory_inputs=_inputs_for(lad))
    assert res["manifest"]["ladder_identity"]["all_bound"] is True


def test_an_attestation_the_seal_could_not_carry_is_refused_before_anything_is_spent(tmp_path):
    """WHERE A MALFORMED RECORD IS CAUGHT, AND WHY IT IS NOT AT THE SEAL.

    `custody.attach_attestation` refuses an incomplete record rather than filling the missing fields
    in, which is right and is late: P5 seals after its bank and its calibration window, so a module
    that omits the sentence or the time would cost a deciding run its bank to discover. This gate
    asks only that a record which WAS supplied is complete. It does not ask whether a deciding run
    must hold one at all, which is the custody gate's question and is left there.
    """
    def _asked(**kw):
        return [r for r in MODE.missing_confirmatory_inputs(MODE.ConfirmatoryInputs(**kw))
                if r.startswith("prior-inspection")]

    complete = _stub_attestation()
    assert _asked(attestation=complete) == []
    # a run supplying none is not asked here at all: whether a deciding run must hold one is the
    # custody gate's question, and this gate is deliberately silent on it
    assert _asked(attestation=None) == [] and _asked() == []

    for absent in ("attester", "heldout_sha256", "statement", "attested_utc"):
        short = {k: v for k, v in complete.items() if k != absent}
        refusals = _asked(attestation=short)
        assert refusals and absent in refusals[0]

    # the runner's own placeholder is not somebody's sentence, and a deciding run holding it would
    # pay for its arms and then be unable to produce a verdict at all
    placeholder = _asked(attestation=custody.mock_attestation())
    assert placeholder and "placeholder" in placeholder[0]
    callable_one = _asked(attestation=lambda: complete)
    assert callable_one and "never a callable" in callable_one[0]

    # and the command line refuses the module's record in the same words, before it starts
    mod = tmp_path / "half_an_attestation.py"
    mod.write_text("def attestation():\n    return {'attester': 'somebody', 'heldout_sha256': '9' * 64}\n",
                   encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(SystemExit) as exc:
            cli.main(["demonstrate", "p16", "--attestation-module", "half_an_attestation"])
        assert "the seal cannot carry" in str(exc.value) and "statement" in str(exc.value)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("half_an_attestation", None)


def test_a_p16_configuration_is_checked_against_p16_rules_and_never_p5_ones():
    """Both directions, because either one alone hides the other's defect."""
    def _resolved(cfg):
        return [r for r in MODE.missing_confirmatory_inputs(
            MODE.ConfirmatoryInputs(config=cfg, config_resolution=_resolution()))
            if r.startswith("resolved-configuration")]

    # the registered titration is not refused, and above all not for lacking a P5 bank's fields.
    # `_resolved` filters to the configuration rules, which is what this case is about; a titration
    # missing its registered quantities is refused under its own prefix and has its own case below.
    assert _resolved(p16.P16Config()) == []
    assert _resolved(p16.P16Config(systems_per_arm=1, horizon=24)) == []
    assert _resolved(_registered_p16()) == []
    # and the gate is not vacuous on this path: each of these makes the titration unrunnable
    assert any("no systems per arm" in r for r in _resolved(p16.P16Config(systems_per_arm=0)))
    assert any("dose offsets" in r for r in _resolved(p16.P16Config(dose_offsets=())))
    assert any("switch round" in r for r in _resolved(p16.P16Config(switch_round=96)))
    assert any("settling" in r for r in _resolved(p16.P16Config(horizon=10, switch_round=8, settling=6)))
    assert any("timing tolerance" in r for r in _resolved(p16.P16Config(timing_tolerance=0)))
    assert any("chi" in r for r in _resolved(p16.P16Config(chi_hat=1.0)))
    # a P5 bank configuration is still checked against the P5 rules
    assert any("no states" in r for r in _resolved(p5.P5Config(states=())))


def test_a_configuration_the_gate_has_no_rules_for_is_refused_and_never_passed(tmp_path):
    """The fail-closed direction of splitting the rule sets: an unrecognised configuration is not a
    configuration that passed, it is one nothing checked."""
    class SomethingElse:
        margin = 0.10

    refusals = MODE.missing_confirmatory_inputs(
        MODE.ConfirmatoryInputs(config=SomethingElse(), config_resolution=_resolution()))
    assert any("has no rules for it" in r for r in refusals)
    assert any("no configuration was supplied" in r for r in MODE.missing_confirmatory_inputs(
        MODE.ConfirmatoryInputs(config=None, config_resolution=_resolution())))


# --------------------------------------------------------------------------------------------------
# The ceiling belongs to every paid mode, and the deciding path is reachable from the command line
# --------------------------------------------------------------------------------------------------

def _seeded_store_dir(tmp_path):
    store = CD.CheckpointStore(str(tmp_path / "store"))
    store.save("seed", CD.new_artefact(_lib(1, 4)))
    return str(tmp_path / "store")


def test_a_smoke_run_refuses_without_an_approved_ceiling_before_it_builds_an_adapter(tmp_path,
                                                                                     monkeypatch):
    """A rehearsal spends real money, and it used to spend it under no figure at all.

    `--allowance-gbp` was required of `confirm` alone, so on the two commands that reach a provider
    without deciding anything it was an option nothing asked for and nothing consulted. A rehearsal
    that empties the account has stopped the deciding run as surely as a refused gate would have. The
    API key is removed so that reaching the adapter at all would fail differently and visibly.
    """
    monkeypatch.delenv("ARC_RUNNER_API_KEY", raising=False)
    d = _seeded_store_dir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["smoke", "p5", "--checkpoints", d])
    assert "approved ceiling" in str(exc.value)
    # and a total with no per-call figure is a number in a record, not a bound on a dispatch
    with pytest.raises(SystemExit) as exc:
        cli.main(["smoke", "p5", "--checkpoints", d, "--allowance-gbp", "5"])
    assert "--max-call-gbp" in str(exc.value)


def test_a_pilot_refuses_without_an_approved_ceiling_too(tmp_path, monkeypatch):
    monkeypatch.delenv("ARC_RUNNER_API_KEY", raising=False)
    d = _seeded_store_dir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        cli.main(["pilot", "p5", "--checkpoints", d, "--pool-module", "tests.pool_for_tests"])
    assert "approved ceiling" in str(exc.value)


def test_the_command_line_reaches_the_p16_deciding_path_and_stops_at_the_missing_key(tmp_path,
                                                                                     monkeypatch):
    """`confirm p16` is documented as a deciding run, and no setup could reach it.

    Every P16 configuration was refused by four P5-shaped requirements it cannot carry, so the last
    thing this command could ever say was that the bank had no states. With a complete setup the run
    now passes the gate and stops at the one thing this test refuses to supply, which is the provider
    key: the gate is no longer the obstacle, and nothing has been spent because no adapter exists.
    """
    monkeypatch.delenv("ARC_RUNNER_API_KEY", raising=False)
    d = _seeded_store_dir(tmp_path)
    registered = ["--chi-hat-se", "0.02", "--slope-equivalence", "0.15", "--informative-horizon", "40",
                  "--practical-absence-band", "0.05", "--across-window-segments", "2"]
    argv = ["confirm", "p16", "--checkpoints", d, "--pool-module", "tests.pool_for_tests",
            "--allowance-gbp", "25", "--max-call-gbp", "0.5", "--resolved-by", "the operator",
            "--anchor-module", "tests.anchor_for_tests"] + registered
    with pytest.raises(RuntimeError) as exc:
        cli.main(argv)
    msg = str(exc.value)
    assert "no API key" in msg
    assert "refuses to start" not in msg and "the bank has no states" not in msg
    # and the gate is still the thing that speaks when an input is missing
    with pytest.raises(SystemExit) as exc:
        cli.main([a for a in argv if a not in ("--resolved-by", "the operator")])
    assert "resolved-configuration" in str(exc.value)
    # AND THE REGISTERED NUMBERS ARE INPUTS LIKE THE REST OF THEM. Without them this command reached
    # the provider and would have paid for arms whose only possible reading is NOT EVALUABLE, so the
    # gate now speaks first and names every quantity that is missing. They are options with no
    # default rather than numbers this command line chooses, which is why the run above is reachable
    # at all: the fail-closed rule must not become an outage in the shape of a safeguard.
    with pytest.raises(SystemExit) as exc:
        cli.main([a for a in argv if a not in registered])
    refused = str(exc.value)
    assert "registered-quantity" in refused
    for name in cli.P16_REGISTERED_OPTIONS:
        assert name in refused, refused


def test_a_bare_allowance_records_that_nothing_reserved_against_it(tmp_path):
    """The manifest used to carry the controller's sentence against an object that bounds nothing."""
    store = _store(tmp_path, 40)
    bare = MODE.ConfirmatoryInputs(allowance=B.Allowance(limit_gbp=12.0),
                                   checkpoint_store=store).as_record()
    assert bare["allowance"]["enforced"] is False
    assert "not a controller" in bare["allowance"]["guarantee"]
    held = MODE.ConfirmatoryInputs(allowance=B.BudgetController(B.Allowance(limit_gbp=12.0)),
                                   checkpoint_store=store).as_record()
    assert held["allowance"]["enforced"] is True
    assert held["allowance"]["allowance_gbp"] == pytest.approx(12.0)


# --------------------------------------------------------------------------------------------------
# The numbers a P16 verdict is read with are inputs, and the ceiling is an object rather than a figure
#
# Two ways a deciding run could still start, spend, and produce nothing a rule could read. Both were
# reachable through a complete setup, which is what makes them worth a case each: nothing was
# missing that anybody had thought to look for.
# --------------------------------------------------------------------------------------------------

def _refusing_assay(cfg):
    """A declared source that refuses to be read, so that "before the first reading" is tested."""
    def src(arm, alpha_arm, r, rng):
        raise AssertionError("a margin reading was taken after a confirmatory run should have refused")

    return OBS.declare(src, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=cfg.chi_hat, source="a-declared-test-assay"))


def _declared_source(name="a-declared-source"):
    """A margin source carrying the declaration the deciding path reads, and nothing else. It is the
    P16 half of the same stand-in `ScriptedCodeSystem` is on the P5 side: the gate asks the observing
    system what it is, and a source that said nothing would refuse before the case's own subject."""
    def source(arm, alpha_arm, r, rng):
        raise AssertionError("a provider call was made after a confirmatory run should have refused")
    return OBS.declare(source, OBS.log_service_ratio_observation(
        supplies_q_and_w=True, chi_hat=0.5, source=name))


def _complete_p16_inputs(tmp_path, cfg, allowance=None):
    """Everything the deciding gate names, so that what a case removes is the only thing missing."""
    lad = CD.SuiteLadder(_registered_pool(20), subset_size=10, batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(2, 20)))
    return MODE.ConfirmatoryInputs(
        ladder=lad, place_at_state=CD.place_at_state_factory(store),
        start_for=CD.start_for_factory(store), checkpoint_store=store, states=(), config=cfg,
        allowance=_controller(25.0) if allowance is None else allowance, anchor=_stub_anchor,
        observing_systems=(_declared_source(),),
        config_resolution=_resolution())


def test_a_deciding_p16_run_refuses_its_unregistered_quantities_before_it_pays(tmp_path):
    """A complete setup, a shipped configuration, and a run that could only ever end NOT EVALUABLE.

    Everything the gate had ever asked for was here: a registered pool, a real checkpoint store, real
    loaders, a live controller, a service anchor and a resolution record. The gate passed it, and the
    run would have reached a paid provider, collected every arm and then reported NOT EVALUABLE,
    because the five numbers its components and its sequential rule read are unset in the shipped
    configuration and none of them has a default. A titration on those defaults reports location NOT
    SUPPLIED, slope-magnitude NOT SUPPLIED, controls UNRESOLVED and P16 NOT EVALUABLE, whatever it is
    measuring, which is what the demonstration reported until it was given candidates to run on.

    That is the case the anchoring service was added to this gate for, said again about a band: a
    deciding run which learns at scoring time that it cannot be scored has already been paid for.
    """
    inputs = _complete_p16_inputs(tmp_path, p16.P16Config())
    refusals = MODE.missing_confirmatory_inputs(inputs)
    assert refusals, "a shipped P16 configuration passed a gate that exists to refuse it"
    named = [r for r in refusals if r.startswith("registered-quantity")]
    assert len(named) == len(refusals), refusals            # nothing else about this setup is missing
    for quantity in ("chi_hat_se", "slope_equivalence", "informative_horizon",
                     "practical_absence_band", "across_window_segments"):
        assert any(quantity in r for r in named), (quantity, named)
    # and the run itself refuses, before the source is read and therefore before anything is paid for
    cfg = p16.P16Config(systems_per_arm=1, horizon=24, dose_offsets=(-0.3, 0.3))
    with pytest.raises(MODE.ModeRefusal) as exc:
        p16.run_p16(_refusing_assay(cfg), cfg, 5, "sha", "an-adapter", mode="confirmatory",
                    confirmatory_inputs=_complete_p16_inputs(tmp_path, cfg))
    assert "registered-quantity" in " ".join(exc.value.requirements)
    # THE OTHER HALF, WHICH IS WHAT KEEPS THIS FROM BEING AN OUTAGE. The same setup with the five
    # registered passes, and the same shipped configuration still runs as a demonstration: what is
    # refused is deciding a proposition on unregistered numbers, not the numbers being unset.
    assert MODE.missing_confirmatory_inputs(_complete_p16_inputs(tmp_path, _registered_p16())) == []
    demo = p16.run_p16(p16.mock_margin_source(p16.P16Config(), true_alpha_crit=2.0),
                       p16.P16Config(systems_per_arm=1, horizon=24), 5, "none-mock", "mock")
    assert demo["manifest"]["execution_mode"] == "demonstration"


def test_a_registered_quantity_that_can_never_decide_anything_is_refused_with_the_absent_ones(tmp_path):
    """A number is not a registration if the component that reads it can never reach a state.

    Each configuration here carries all five quantities, so the absence rule is satisfied and the
    value rule is the only thing that can speak. None of these is refused for being large or small:
    each is refused because under it the run could not produce the evidence its verdict rule reads,
    which is the criterion the rest of the P16 configuration rules are written against.
    """
    def _named(cfg):
        return [r for r in MODE.missing_confirmatory_inputs(_complete_p16_inputs(tmp_path, cfg))
                if r.startswith("registered-quantity")]

    # one segment is the window mean, which is the reading the across-window predicate replaces. The
    # module raises this at run time; the gate asks it before the money.
    assert any("one piece" in r for r in _named(_registered_p16(across_window_segments=1)))
    # a resolution the window is too short to be read at leaves the predicate unevaluated in every arm
    assert any("post-settling window holds" in r
               for r in _named(_registered_p16(across_window_segments=4)))
    # a horizon no arm reaches censors every silence in every run
    assert any("censored in every run" in r for r in _named(_registered_p16(informative_horizon=40)))
    # and a horizon of zero calls silence informative before a round has been observed
    assert any("before a single round" in r for r in _named(_registered_p16(informative_horizon=0)))
    # a band of no width contains no interval, so the component can never be satisfied
    assert any("has no width" in r for r in _named(_registered_p16(slope_equivalence=0.0)))
    # and a negative calibration standard error is not one
    assert any("not a standard error" in r for r in _named(_registered_p16(chi_hat_se=-0.01)))
    # THE STRICT DIRECTIONS PASS, which is the half that keeps this from refusing a registration. A
    # zero calibration uncertainty and a zero practical-absence band are both claims made in the open
    # and sealed with the specification, and neither is this gate's to overrule.
    assert _named(_registered_p16(chi_hat_se=0.0)) == []
    assert _named(_registered_p16(practical_absence_band=0.0)) == []
    assert _named(_registered_p16(informative_horizon=15, across_window_segments=2)) == []


def test_the_deciding_path_holds_its_ceiling_in_a_controller_and_not_in_a_figure(tmp_path):
    """An approved figure nothing reserves against bounds the manifest and not the spending.

    The gate took either object and recorded `enforced: False` against the bare one, which is an
    accurate record of a deciding run with no ceiling on what it spent. The command line has built a
    controller since the metered adapter was written, so the caller this refusal is for is the one
    that reaches the gate through the API.
    """
    n = 120
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path, n)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 11, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store, allowance=B.Allowance(limit_gbp=12.0)))
    assert "spending-allowance" in _requirement_names(exc)
    assert "not a controller" in " ".join(exc.value.requirements)
    # the same setup holding the same figure in a controller passes
    assert MODE.missing_confirmatory_inputs(MODE.ConfirmatoryInputs(
        checkpoint_store=store, allowance=_controller(), anchor=_stub_anchor, ladder=lad,
        place_at_state=CD.place_at_state_factory(store), start_for=CD.start_for_factory(store),
        observing_systems=(ScriptedCodeSystem(n),),
        states=tuple(_cfg().states), config=_cfg(), config_resolution=_resolution())) == []
    # AND THE REHEARSALS ARE UNCHANGED: a smoke test and a pilot are refused without a figure and are
    # not asked to hold one in a controller, because the requirement belongs to the path that decides
    # a proposition. A demonstration is not asked for either.
    bare = B.Allowance(limit_gbp=5.0)
    for m in (MODE.ExecutionMode.SMOKE, MODE.ExecutionMode.PILOT):
        assert MODE.require_spending_allowance(m, bare) is bare
        with pytest.raises(MODE.ModeRefusal):
            MODE.require_spending_allowance(m, None)
    assert MODE.require_spending_allowance(MODE.ExecutionMode.DEMONSTRATION, None) is None
    # and the deciding path says the same thing in both places it is asked
    with pytest.raises(MODE.ModeRefusal) as exc:
        MODE.require_spending_allowance(MODE.ExecutionMode.CONFIRMATORY, bare)
    assert "not a controller" in str(exc.value)
    # the record still describes the object it refuses, because saying what a thing IS and saying
    # what the deciding path REQUIRES are two jobs and the refusal needs the first one to explain it
    rec = MODE.ConfirmatoryInputs(allowance=bare, checkpoint_store=store).as_record()
    assert rec["allowance"]["enforced"] is False and "not a controller" in rec["allowance"]["guarantee"]


def test_the_command_line_offers_exactly_the_quantities_the_gate_names():
    """A drift test. The parser, the configuration and the gate name one set of quantities between
    them, and an option the gate does not read would be a number that looks registered and is not."""
    assert set(cli.P16_REGISTERED_OPTIONS) == {n for n, _, _, _ in MODE._P16_REGISTERED_QUANTITIES}
    for name in cli.P16_REGISTERED_OPTIONS:
        assert hasattr(p16.P16Config(), name)
        assert getattr(p16.P16Config(), name) is None, \
            "%s ships with a default, so the gate would be refusing a number somebody chose here" % name



# --------------------------------------------------------------------------------------------------
# The demonstration reaches a result, on numbers it names candidates, and the deciding path refuses
# the same configuration
#
# THE DEFECT THESE CASES PIN. All five registered quantities are unset on the shipped defaults, and
# the components that read them are NOT SUPPLIED until an author sets them, which is right and stays
# right on every paid path. `demonstrate p16` inherited that state, so the one command whose job is to
# show what the branch does reported NOT EVALUABLE and showed nothing: four components unsupplied, no
# result, and a reader none the wiser about the wrapper, the line, the timing or the controls. A
# demonstration is the mode a candidate number exists for, so the demonstration is given the
# candidates `p16_calibration.py` measures the decision family under and says so wherever the result
# is read. Every case below asserts both halves: that the demonstration now decides something, and
# that nothing it was given can travel to a run that decides a proposition.
# --------------------------------------------------------------------------------------------------

def _summary_from(out: str) -> dict:
    """The JSON block the command line prints, taken from the printed text and not from the call."""
    return json.loads(out[out.index("{"):])


def test_the_demonstration_reaches_a_result_and_is_still_refused_at_proposition_level(capsys):
    """The whole command, read the way an operator reads it.

    Two things have to be true at once, and the second is what keeps the first from being a loosening:
    the wrapper must reach one of its results rather than withhold every one, AND the run must still
    be refused the moment anybody asks to score it. A demonstration that could be scored would be the
    defect finding A9 exists for, wearing the repair's clothes.
    """
    assert cli.main(["demonstrate", "p16"]) == 0
    printed = capsys.readouterr().out
    summary = _summary_from(printed)
    v = summary["verdicts"]

    # a result, and not the wrapper's refusal to weigh anything
    assert v["P16"] != V.NOT_EVALUABLE, v["component_reason"]
    blocking = [n for n in v["components_not_supplied"] if n != C.REPETITION]
    assert blocking == [], "the demonstration still withholds a result: %s" % v["component_reason"]
    # every component the contract names was decided on its own evidence, the repetition apart: a
    # single run has no fresh-data repetition and the wrapper labels it provisional rather than
    # pretending otherwise
    assert set(v["component_states"]) == set(C.CONTRACT_COMPONENTS)
    assert v["component_states"][C.REPETITION] == C.NOT_SUPPLIED
    assert v["provisional"] is True

    # and it may still not be read at proposition level
    assert summary["proposition_level"].startswith("REFUSED")
    assert "demonstration mode" in summary["proposition_level"]
    with pytest.raises(M.NotScoreable):
        M.require_scoreable(summary["manifest"])


def test_the_demonstration_names_every_candidate_number_in_the_summary_and_the_verdict(capsys):
    """The label leads the summary beside the mode label, and travels inside the verdict block.

    A band nobody registered does not make a result weaker, it makes it a result about a rule this
    package chose, and a reader who has to open the configuration to learn that will quote the verdict
    without it. So it is printed where the mode label is printed and carried where `provisional` is
    carried.
    """
    assert cli.main(["demonstrate", "p16"]) == 0
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert lines[0].startswith("DEMONSTRATION"), lines[0]
    candidate_line = [ln for ln in lines[:4] if ln.startswith("CANDIDATE NUMBERS")]
    assert candidate_line, lines[:4]
    for name in cli.P16_REGISTERED_OPTIONS:
        assert name in candidate_line[0], candidate_line[0]

    v = _summary_from(printed)["verdicts"]
    assert v["candidate_label"] == candidate_line[0]
    assert v["candidate_quantities"] == p16.CANDIDATE_QUANTITIES


def test_the_demonstrations_manifest_names_every_candidate_number_as_a_candidate(tmp_path):
    """And the record says it, not only the summary, and says it inside the seal.

    The names are a field of the configuration rather than a note beside it, so they are inside
    `custody.spec_hash_of` with the rest of the configuration: a bundle cannot have the label stripped
    afterwards and then be read as though its bands had been registered. The analyst who re-scores the
    saved bundle, who was not present for the run and sees no summary, is the reader this matters
    most to, so the re-scoring is asserted here too.
    """
    cfg = p16.demonstration_config()
    res = p16.run_p16(p16.mock_balance_source(cfg, true_alpha_crit=cfg.alpha_crit_hat), cfg, 5,
                      "none-mock", "mock", bundle=str(tmp_path / "bundle"))
    recorded = res["manifest"]["config"]
    assert tuple(recorded["candidate_quantities"]) == tuple(p16.CANDIDATE_QUANTITIES)
    declared = recorded["candidate_quantities_declared"]
    assert declared["values"] == p16.CANDIDATE_QUANTITIES
    assert sorted(declared["names"]) == sorted(p16.CANDIDATE_QUANTITIES)
    for name, value in p16.CANDIDATE_QUANTITIES.items():
        assert "%s=%s" % (name, value) in declared["label"]
    # the label is inside the commitment: the sealed specification hash still recomputes from the
    # manifest's own fields, and the configuration is one of them
    assert custody.spec_hash_of(res["manifest"]) == res["manifest"]["seal"]["spec_sha256"]
    # and the analyst re-scoring the bundle alone reaches the same result under the same label
    rescored = custody.recompute_verdicts(custody.load_bundle(res["bundle_path"]))
    assert rescored["P16"] == res["verdicts"]["P16"]
    assert rescored["candidate_quantities"] == p16.CANDIDATE_QUANTITIES
    assert rescored["candidate_label"] == res["verdicts"]["candidate_label"]


def test_the_same_configuration_on_the_deciding_path_is_refused(tmp_path):
    """A value is not a registration, and a complete setup carrying candidates must not start.

    Every absence rule is satisfied here: all five numbers are present and each is one the value rules
    accept. What is missing is the only thing that matters, which is that somebody registered them,
    and the gate has to be able to see it. It is refused before the adapter exists, so nothing is
    spent, and the refusal names every candidate rather than the first one found.
    """
    cfg = p16.demonstration_config()
    refusals = MODE.missing_confirmatory_inputs(_complete_p16_inputs(tmp_path, cfg))
    assert refusals, "a titration carrying five candidate numbers passed the deciding gate"
    assert all(r.startswith("registered-quantity") for r in refusals), refusals
    for name in cli.P16_REGISTERED_OPTIONS:
        assert any(name in r and "candidate" in r for r in refusals), (name, refusals)
    # and the gate itself refuses rather than merely listing
    with pytest.raises(MODE.ModeRefusal) as exc:
        MODE.require_confirmatory_inputs(_complete_p16_inputs(tmp_path, cfg))
    assert "CANDIDATE" in str(exc.value)
    # the same five numbers registered by an author, with nothing labelled, pass exactly as before:
    # what is refused is the label and never the width
    registered = p16.P16Config(**dict(p16.CANDIDATE_QUANTITIES))
    assert registered.candidate_quantities == ()
    assert MODE.missing_confirmatory_inputs(_complete_p16_inputs(tmp_path, registered)) == []


def test_the_paid_rehearsals_do_not_inherit_the_demonstrations_candidates(tmp_path, monkeypatch):
    """A smoke test and a pilot reach a real provider, so neither may be handed a number to spend on.

    The configuration each of them builds is captured on its way into the runner, because that is the
    object the question is about: a rehearsal that quietly acquired the demonstration's bands would
    carry them into whatever a later reader did with its bundle.
    """
    seen = {}

    class _Stop(Exception):
        pass

    class _StubAdapter:
        name = "stub-adapter"

    def _capture(src, cfg, *args, **kw):
        seen["cfg"] = cfg
        raise _Stop()

    monkeypatch.setattr(cli, "_paid_adapter", lambda a, m, controller: _StubAdapter())
    monkeypatch.setattr(p16, "run_p16", _capture)
    d = _seeded_store_dir(tmp_path)
    for word, extra in (("smoke", []), ("pilot", ["--pool-module", "tests.pool_for_tests"])):
        seen.clear()
        with pytest.raises(_Stop):
            cli.main([word, "p16", "--checkpoints", d, "--allowance-gbp", "5",
                      "--max-call-gbp", "0.5"] + extra)
        cfg = seen["cfg"]
        assert cfg.candidate_quantities == (), word
        for name in cli.P16_REGISTERED_OPTIONS:
            assert getattr(cfg, name) is None, (word, name)
    # and a quantity typed on a rehearsal is still recorded, which is the behaviour the option has
    # always had: what a rehearsal may not have is a number nobody typed
    seen.clear()
    with pytest.raises(_Stop):
        cli.main(["smoke", "p16", "--checkpoints", d, "--allowance-gbp", "5",
                  "--max-call-gbp", "0.5", "--slope-equivalence", "0.2"])
    assert seen["cfg"].slope_equivalence == pytest.approx(0.2)
    assert seen["cfg"].chi_hat_se is None and seen["cfg"].candidate_quantities == ()


def test_a_quantity_typed_on_the_demonstration_replaces_a_candidate_and_is_not_named_as_one():
    """The label is a statement about where each value came from, not about which command was typed.

    A number the operator supplies is the operator's, so it stops being a candidate and drops out of
    the label; the ones nobody supplied are still named. A label that ignored this would tell a reader
    that a value they had chosen themselves was one this package had.
    """
    cfg = p16.demonstration_config(slope_equivalence=0.25)
    assert cfg.slope_equivalence == pytest.approx(0.25)
    assert "slope_equivalence" not in cfg.candidate_quantities
    assert set(cfg.candidate_quantities) == set(p16.CANDIDATE_QUANTITIES) - {"slope_equivalence"}
    record = p16.candidate_record(cfg)
    assert "slope_equivalence" not in record["label"]
    assert "chi_hat_se" in record["label"]
    # and a configuration whose numbers all came from an author says nothing at all
    empty = p16.candidate_record(p16.P16Config(**dict(p16.CANDIDATE_QUANTITIES)))
    assert empty == {"names": [], "values": {}, "label": ""}


def test_the_candidate_numbers_are_declared_in_one_place():
    """A drift test, of the same kind as the one above it. The calibration script measures the
    decision family under these numbers and the demonstration is read with them, so two copies of the
    dictionary would eventually differ and the calibration would then describe a family nothing else
    was ever exercised under."""
    import p16_calibration as CAL

    assert CAL.CANDIDATE == p16.CANDIDATE_QUANTITIES
    assert set(p16.CANDIDATE_QUANTITIES) == set(cli.P16_REGISTERED_OPTIONS)


# --------------------------------------------------------------------------------------------------
# What the refusal SAYS, against what the run does
#
# The gate's decisions were right and its words were not. Two sentences claimed a consequence nobody
# had measured: that a run missing any ONE of the five could only end NOT EVALUABLE, and that without
# a practical-absence band no control's silence is demonstrated. Both were generalised from the case
# where all five are unset, which is the only case that had been run. A gate whose refusal overstates
# what would have happened is worth less than one that says less: the operator who runs anyway, gets
# a named verdict and finds the warning was false has been taught to discount the next refusal too,
# and the next refusal is the one standing between an unregistered band and a paid run.
# --------------------------------------------------------------------------------------------------

def _p16_verdict_without(quantity, horizon=48):
    """The verdict a titration reaches with one registered quantity removed and the rest in place.

    The demonstration world, because it is the only one that runs without a provider, and the
    candidate numbers, because the question here is what the ABSENCE of one does rather than what any
    width decides. Small enough to run six times in a test and long enough for the informative
    horizon to be reached, which is what makes the three quiet quantities quiet.
    """
    cfg = p16.demonstration_config()
    cfg.horizon = horizon
    cfg.systems_per_arm = 1
    if quantity is not None:
        setattr(cfg, quantity, None)
    res = p16.run_p16(p16.mock_balance_source(cfg, true_alpha_crit=cfg.alpha_crit_hat), cfg,
                      seed=5, ladder_sha256="none-mock", adapter_name="mock")
    return res["verdicts"]["P16"]


def test_the_gate_states_the_consequence_the_run_actually_reaches(tmp_path):
    """Each refusal's claim about NOT EVALUABLE, measured on the run rather than reasoned about.

    NOT EVALUABLE needs a contract-required component in NOT SUPPLIED, and of the five quantities
    only the calibration uncertainty and the slope band can put one there: they are what
    `location_component` and `slope_component` read before they compare anything. The horizon, the
    practical-absence band and the across-window resolution reach the controls component and the
    refutation rule instead, and removing one of those leaves the verdict exactly where the complete
    configuration left it. So the gate may say NOT EVALUABLE about the first two and must not say it
    about the other three, and this test measures which is which rather than trusting the table.
    """
    intact = _p16_verdict_without(None)
    measured_fatal = set()
    for name, _what, _consequence, _alone in MODE._P16_REGISTERED_QUANTITIES:
        verdict = _p16_verdict_without(name)
        if verdict == V.NOT_EVALUABLE:
            measured_fatal.add(name)
        else:
            assert verdict == intact, (name, verdict, intact)   # quiet: the name does not change
    assert measured_fatal == {"chi_hat_se", "slope_equivalence"}, measured_fatal
    # and the table records what was measured, which is what every refusal below is built from
    flagged = {n for n, _, _, alone in MODE._P16_REGISTERED_QUANTITIES if alone}
    assert flagged == measured_fatal, (flagged, measured_fatal)

    def _named(cfg):
        return [r for r in MODE.missing_confirmatory_inputs(_complete_p16_inputs(tmp_path, cfg))
                if r.startswith("registered-quantity")]

    # A configuration missing ONLY quiet quantities must not be told it would have been NOT EVALUABLE.
    # The refusal is allowed to name the verdict in order to DENY it, so the denial is removed before
    # the text is searched: what must not survive is a claim that this run ends there.
    denial = "none of them alone ends the run at NOT EVALUABLE"
    for quiet in sorted(set(n for n, _, _, _ in MODE._P16_REGISTERED_QUANTITIES) - measured_fatal):
        refusals = _named(_registered_p16(**{quiet: None}))
        assert refusals, quiet
        joined = " ".join(refusals)
        assert denial in joined, (quiet, refusals)
        assert V.NOT_EVALUABLE not in joined.replace(denial, ""), (quiet, refusals)
        assert "evidence the registration does not have" in joined, (quiet, refusals)
    # and one missing a fatal quantity is told exactly that, because there it is true
    for fatal in sorted(measured_fatal):
        joined = " ".join(_named(_registered_p16(**{fatal: None})))
        assert "ends at %s" % V.NOT_EVALUABLE in joined, (fatal, joined)
        assert denial not in joined, (fatal, joined)
    # the shipped configuration, missing all five, is still told the loud thing, since with the two
    # fatal ones among them the run does end there: the sentence was never wrong about THIS case
    assert "ends at %s" % V.NOT_EVALUABLE in " ".join(_named(p16.P16Config()))


def test_an_unregistered_practical_absence_band_still_leaves_a_control_able_to_demonstrate():
    """The band's refusal claimed that without it no control's silence is demonstrated. It is false.

    Measured on `p16_sequential.classify`: a control whose terminal interval sits clearly off zero
    and stays there across the window reads DEMONSTRATED SIGN with `demonstrated` true whether or not
    a band is registered, and `controls_component` then reports SATISFIED on those silences. What the
    missing band actually costs is the other arm, the steady one: with no band there is nothing for a
    nil margin to lie inside, so it reads LOW INFORMATION and demonstrates nothing, and no reading at
    all is admissible for refutation, since a margin merely above zero is not positivity. That is a
    sound reason to refuse and it is not the reason the gate was giving.
    """
    def _rule(**over):
        kw = dict(switch_round=8, settling=6, start_round=14, horizon=96, min_points=6,
                  z_threshold=2.0, terminal_z=1.96, variance_estimator="independent",
                  informative_horizon=10, practical_absence_band=0.05, across_window_segments=2,
                  alarm_rate_null=0.05)
        kw.update(over)
        return SEQ.SequentialRule(**kw)

    def _terminal(delta, se, last_round=39):
        seg = lambda first, last: {"first_round": first, "last_round": last, "delta": delta,
                                   "se": se, "n_used": 13,
                                   "interval": [delta - 2 * se, delta + 2 * se]}
        return {"delta": delta, "se": se, "se_independent": se, "se_robust": None,
                "variance_estimator": "independent", "n_used": 26, "n_excluded": 0,
                "first_round": 14, "last_round": last_round,
                "rounds_after_switch": last_round - 8, "declared_round": None,
                "looks_taken": 20, "looks_available": 20,
                "across_window": {"n_segments": 2, "segments": [seg(14, 26), seg(27, last_round)]}}

    controls = [{"arm": "sham", "declared_round": None}, {"arm": "baseline", "declared_round": None}]
    no_band = _rule(practical_absence_band=None)

    # a control sitting clearly off zero: demonstrated either way, and the controls are satisfied
    off_zero = SEQ.classify(_terminal(0.30, 0.01), no_band)
    assert off_zero["state"] == SEQ.DEMONSTRATED_SIGN
    assert off_zero["demonstrated"] is True
    assert off_zero["refutation_admissible"] is False        # and nothing is admissible without a band
    assert C.controls_component(controls, alarm_rate_null=0.05,
                                readings=[off_zero, off_zero]).state == C.SATISFIED
    # the same arm with the band registered is the one reading that may refute
    assert SEQ.classify(_terminal(0.30, 0.01), _rule())["refutation_admissible"] is True

    # and the arm the band exists to read, the steady one, is the arm that stops being readable
    nil = SEQ.classify(_terminal(0.0, 0.01), no_band)
    assert nil["state"] == SEQ.LOW_INFORMATION and nil["demonstrated"] is False
    assert C.controls_component(controls, alarm_rate_null=0.05,
                                readings=[nil, nil]).state == C.UNRESOLVED
    assert SEQ.classify(_terminal(0.0, 0.01), _rule())["state"] == SEQ.PRACTICAL_ABSENCE

    # so the refusal says the true thing and no longer says the false one
    band = [c for c in MODE._P16_REGISTERED_QUANTITIES if c[0] == "practical_absence_band"][0]
    assert "no control's silence is demonstrated" not in band[2], band[2]
    assert "LOW INFORMATION" in band[2]
    # the two whose silences genuinely all go undemonstrated may still say it, because they were
    # measured saying it: with no horizon every non-alarm is censored, and with no resolution the
    # across-window predicate is evaluated in no arm, so nothing is demonstrated in either
    for name, drop in (("informative_horizon", dict(informative_horizon=None)),
                       ("across_window_segments", dict(across_window_segments=None))):
        reading = SEQ.classify(_terminal(0.30, 0.01), _rule(**drop))
        assert reading["demonstrated"] is False, name
        assert C.controls_component(controls, alarm_rate_null=0.05,
                                    readings=[reading, reading]).state == C.UNRESOLVED, name
        entry = [c for c in MODE._P16_REGISTERED_QUANTITIES if c[0] == name][0]
        assert "no control's silence is demonstrated" in entry[2], name
