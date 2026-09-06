"""Custody: finding A8's acceptance cases.

One question runs through all of them: can a result be read at proposition level without the chain
that makes it checkable by somebody who was not present? The answer must be no when the predictions
moved after the seal, no when the configuration moved, no when nothing outside the runner attested
the commitment, and no when the receipt is one the runner manufactured for itself. It must be yes,
exactly once, for a run that holds the whole chain, because a gate that refuses everything is an
outage and passes every refusal test perfectly.

The other three cases are about the evidence surviving the run: an identity that another directory
reproduces, an identity that a changed verifier does not, and a bundle an analyst can reload and
re-score without the process that wrote it.
"""
import hashlib
import json
import os
import shutil

import numpy as np
import pytest

from arc_runner import adapters, code_domain as CD, custody, ladder as L, manifest as M, mode as MODE, p5, p16
from arc_runner import budget as B


# --------------------------------------------------------------------------------------------------
# Fixtures: the same shapes the execution-mode tests use, plus an anchoring service that is not a mock
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
        checks = ("assert %s(3, 4) == 7" % name, "assert %s(0, 0) == 0" % name, "assert %s(-1, 1) == 0" % name)
        tasks.append(CD.Task(id=name, statement="Define %s(x, y) returning the sum." % name,
                             signature="def %s(x, y): ..." % name, shown_examples=(checks[0],), checks=checks))
    return CD.TaskPool(tasks, name="registered-domain-pool")


class ScriptedCodeSystem:
    """A system whose revisions are real code and whose growth follows the registered law."""
    name = "scripted-code"

    def __init__(self, n_total, beta=0.5, a=1.6):
        self.n_total, self.beta, self.a = n_total, beta, a
        self.calls = 0

    def metadata(self):
        return {"adapter": self.name, "simulated": False, "calls": self.calls,
                "returned_models": ["scripted-code-v1"], "usage_totals": {"total_tokens": 7 * self.calls},
                "records": []}

    def revise(self, artefact, retained, task, rng):
        self.calls += 1
        done = artefact.get("text", "").count("return x + y")
        available = max(float(retained.get("fraction", 1.0)) * max(done, 1), 1e-9)
        new_done = min(self.n_total, done + max(1, int(round(self.a * available ** self.beta))))
        out = dict(artefact)
        out["text"] = _lib(new_done, self.n_total)
        out["rounds"] = int(artefact.get("rounds", 0)) + 1
        return out


def _lib_of(correct, n_total):
    """The same library as `_lib`, with the passing items named rather than counted. `_lib` makes the
    first k items correct, so which items pass is always a prefix and a system built on it can never
    break one it had already fixed. The four balance objects count regressions, and a round that
    regresses nothing offers no burden at all: W is zero, the service ratio does not exist, and every
    reading is excluded. A titration needs a world where both terms are measured."""
    return "\n".join("def add_%d(x, y):\n    %s" % (i, "return x + y" if i in correct else "return None")
                     for i in range(n_total))


def _correct_ids(text, n_total):
    return {i for i in range(n_total)
            if ("def add_%d(x, y):\n    return x + y" % i) in text}


class RevisingCodeSystem:
    """A system whose rounds both repair and break real code, so that Q and W are both counted.

    It repairs from the front, in a number that rises with the retained fraction, and regresses one
    item that was passing when the round began. Nothing about its behaviour is a claim: the arms of a
    titration are what this test is about, and what matters is that the whole pool is genuinely read
    once per round and that the readings reach the bundle."""

    name = "revising-code"

    def __init__(self, n_total, gain=2.0):
        self.n_total, self.gain, self.calls = n_total, gain, 0

    def metadata(self):
        return {"adapter": self.name, "simulated": False, "calls": self.calls,
                "returned_models": ["revising-code-v1"],
                "usage_totals": {"total_tokens": 5 * self.calls}, "records": []}

    def revise(self, artefact, retained, task, rng):
        self.calls += 1
        before = _correct_ids(artefact.get("text", ""), self.n_total)
        correct = set(before)
        broken = [i for i in range(self.n_total) if i not in correct]
        available = max(float(retained.get("fraction", 1.0)) * max(len(correct), 1), 1e-9)
        for i in broken[:max(1, int(round(self.gain * available ** 0.5)))]:
            correct.add(i)
        if before:
            correct.discard(min(before))          # one regression a round, so the burden is measured
        out = dict(artefact)
        out["text"] = _lib_of(correct, self.n_total)
        out["rounds"] = int(artefact.get("rounds", 0)) + 1
        return out


def _store(tmp_path, n_total, states=(10, 20, 40)):
    store = CD.CheckpointStore(str(tmp_path))
    store.save("seed", CD.new_artefact(_lib(4, n_total)))
    for s in states:
        store.save(CD.state_name(s), CD.new_artefact(_lib(int(s), n_total)))
    return store


def _cfg(states=(10, 20, 40)):
    # Three replicates per cell and four reads per cell, where this fixture used two and one. The
    # observation model carries three structural parameters and one latent capability state per
    # state, so a route subset of three states by two replicates has six cells against six parameters
    # and no residual degrees of freedom at all, and `p5_observation.fit_paired` refuses such a
    # subset rather than returning the exponent the last decimal of the noise asked for. One read of
    # a forty item form also leaves an increment of one or two items under the read noise, so a third
    # of the cells came back nonpositive and the bank failed its own precision condition. Custody is
    # what these tests are about, so the bank is given enough cells and enough reads to be analysable
    # at all; nothing about the custody assertions changed.
    return p5.P5Config(states=states, fractions=(0.5, 1.0), reps=3, window_end=8, checkpoints=(4, 8),
                       calibration_depths=(1, 2, 3, 4), cal_reads=2, reads=4, heldout_reads=1,
                       replicates=2, bootstrap=20, control_reads_multiplier=2)


def _resolution():
    return {"resolved_by": "the operator", "resolved_utc": "2026-09-05T00:00:00Z"}


def anchor_service(digest):
    """A stand-in for the operator's commitment service: it returns the shape an external service
    returns, so the deciding path is exercised rather than simulated by a mock."""
    return custody.receipt("stub-anchor:%s" % digest[:12], digest, service="test-stub")


anchor_service.anchor_service = "test-stub"

# The material these tests stand in for. Nothing in the runner computes this digest and nothing in it
# may: whether a person read the held-out material before the predictions were fixed is not a fact
# this code can establish, so the digest arrives with the attester's sentence and never from here.
HELDOUT_MATERIAL = b"the held-out material of the custody tests: a frozen suite nobody read"


def attestation_record(heldout_sha256=None, attester="the custodian of the held-out material"):
    """A stand-in for the sentence a named party signs before the seal, and NOT the placeholder the
    runner attaches when nobody signs: the deciding path is exercised on the record a real run has to
    carry. Finding A8's third requirement, and the one no hash can supply."""
    return custody.attestation(attester,
                               heldout_sha256 or hashlib.sha256(HELDOUT_MATERIAL).hexdigest(),
                               material="the frozen hidden suite this run reads")


# The sentinel that tells `_inputs` to build the stub. `attestation=None` therefore means what it
# says: a deciding run that nobody attested, which is the case the refusal tests below need.
_STUB_ATTESTATION = "a stub attestation"


def _inputs(store, anchor=anchor_service, attestation=_STUB_ATTESTATION):
    # A controller and not a bare figure: the deciding path requires the object that reserves before
    # each dispatch, because an approved figure nothing reserves against bounds the record and not
    # the spending. A fresh one per call, since a controller carries a ledger.
    return MODE.ConfirmatoryInputs(checkpoint_store=store,
                                   allowance=B.BudgetController(B.Allowance(limit_gbp=12.0)),
                                   anchor=anchor,
                                   attestation=(attestation_record()
                                                if attestation is _STUB_ATTESTATION else attestation),
                                   config_resolution=_resolution())


def _confirmatory_run(tmp_path, bundle=None, n=120, attestation=_STUB_ATTESTATION):
    pool = _registered_pool(n)
    lad = CD.SuiteLadder(pool, subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path / "store", n)
    return p5.run_p5(ScriptedCodeSystem(n), lad, _cfg(), 11, CD.place_at_state_factory(store),
                     CD.start_for_factory(store), ["S1"], mode="confirmatory",
                     confirmatory_inputs=_inputs(store, attestation=attestation), bundle=bundle)


def _mock_run(bundle=None, mode=None, seed=3, anchor=None, attestation=None):
    """`anchor` is the honest case that makes the promotion test worth running: a rehearsal MAY take a
    real receipt from the operator's service, and the shipped command line gives it one in every mode.
    What must not follow is the rehearsal being relabelled a deciding run afterwards."""
    ad = adapters.MockCouplingAdapter(beta=0.5)
    lad = L.MockLadder(n_items=2000, scale=400.0)
    place = lambda s: {"kind": "mock", "capability": float(s), "rounds": 0}
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    cfg = p5.P5Config(reps=2, window_end=16, checkpoints=(4, 8, 16), bootstrap=20,
                      control_reads_multiplier=2)
    return p5.run_p5(ad, lad, cfg, seed, place, start, ["S1"], mode=mode, bundle=bundle,
                     anchor=anchor, attestation=attestation)


# --------------------------------------------------------------------------------------------------
# Acceptance case: an altered prediction after sealing refuses
# --------------------------------------------------------------------------------------------------

def test_a_prediction_altered_after_sealing_refuses(tmp_path):
    res = _confirmatory_run(tmp_path)
    man, heldout, cfg = res["manifest"], res["heldout"], _cfg()
    M.require_scoreable(man, predictions=heldout["sealed_predictions"], config=cfg)   # as run, it holds
    tampered = json.loads(json.dumps(heldout["sealed_predictions"]))
    tampered["S1"]["predicted_exponent"] = 0.99
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man, predictions=tampered, config=cfg)
    assert "not the same predictions" in str(exc.value)
    # and the verdict path refuses too, which is where a real caller would meet it
    res["heldout"]["sealed_predictions"]["S1"]["predicted_exponent"] = 0.99
    with pytest.raises(custody.CustodyRefusal):
        p5.verdicts(man, res["routes"], res["heldout"], cfg)


def test_a_configuration_altered_after_sealing_refuses(tmp_path):
    res = _confirmatory_run(tmp_path)
    man = res["manifest"]
    live = _cfg()
    live.margin = 0.5                                   # the margin moved after the seal
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man, config=live)
    assert "'margin' changed after sealing" in str(exc.value)
    # editing the manifest's own record instead is caught by the specification hash, which is
    # recomputed rather than trusted
    man2 = json.loads(json.dumps(res["manifest"]))
    man2["config"]["margin"] = 0.5
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man2)
    assert "changed after sealing" in str(exc.value)


# --------------------------------------------------------------------------------------------------
# Acceptance case: a missing receipt, and a mock one, refuse on the deciding path
# --------------------------------------------------------------------------------------------------

def test_a_confirmatory_run_without_a_receipt_refuses(tmp_path):
    res = _confirmatory_run(tmp_path)
    man = json.loads(json.dumps(res["manifest"]))
    man["seal"].pop("anchor_receipt")
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man)
    assert "no anchor receipt" in str(exc.value)


def test_a_mock_receipt_is_refused_on_the_deciding_path_and_accepted_off_it(tmp_path):
    # off the deciding path: a demonstration seals, receives a mock receipt, and says so in three places
    demo = _mock_run()
    rec = demo["manifest"]["seal"]["anchor_receipt"]
    assert rec["mock"] is True
    assert rec["anchor_identifier"].startswith("mock-anchor:")
    assert "MOCK ANCHOR" in rec["label"]
    assert demo["manifest"]["anchor_identifier"] == rec["anchor_identifier"]
    assert custody.custody_failures(demo["manifest"], external_anchor_required=False) == []
    # on it: the same receipt is refused, and the gate refuses the mock anchor as an input as well
    res = _confirmatory_run(tmp_path)
    man = json.loads(json.dumps(res["manifest"]))
    man["seal"]["anchor_receipt"] = custody.mock_anchor(custody.seal_digest(man["seal"]))
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man)
    assert "is a mock" in str(exc.value)
    bad = MODE.ConfirmatoryInputs(anchor=custody.mock_anchor)
    assert any(r.startswith("anchor-service") for r in MODE.missing_confirmatory_inputs(bad))


def test_a_receipt_for_another_document_is_not_a_receipt_for_this_one(tmp_path):
    res = _confirmatory_run(tmp_path)
    man = json.loads(json.dumps(res["manifest"]))
    man["seal"]["sealed_at_utc"] = "2001-01-01T00:00:00Z"     # the sealed record moved after anchoring
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man)
    assert "changed after it was anchored" in str(exc.value)
    with pytest.raises(custody.CustodyRefusal):
        custody.anchor_seal({"seal": dict(res["manifest"]["seal"])},
                            lambda digest: custody.receipt("elsewhere", "0" * 64, service="wrong"))


# --------------------------------------------------------------------------------------------------
# Acceptance case: the third question, which no digest can answer
# --------------------------------------------------------------------------------------------------

def test_a_deciding_run_needs_a_named_party_to_say_the_held_out_material_was_unseen(tmp_path):
    """THE THIRD THING `require_scoreable` DID NOT DO. The finding names three questions: are these
    the predictions that were sealed, is this the configuration they were sealed under, and had the
    material deciding them already been inspected when they were chosen. The first two are answered
    by the anchor receipt and the recomputed specification hash. Nothing anywhere asked the third,
    and no hash can: a digest proves the material has not moved since it was hashed and says nothing
    about who had read it before. A run whose author had read the held-out suite before choosing its
    predictions passed every custody check there was, and the record said nothing about it either
    way. What is required is a named party's sentence, made before the seal.
    """
    res = _confirmatory_run(tmp_path)
    man = res["manifest"]
    att = man["seal"]["heldout_attestation"]
    assert att["mock"] is False and att["attester"] == "the custodian of the held-out material"
    assert att["heldout_sha256"] == hashlib.sha256(HELDOUT_MATERIAL).hexdigest()
    assert att["attested_utc"] and "had not been inspected" in att["statement"]

    report = M.require_scoreable(man, predictions=res["heldout"]["sealed_predictions"], config=_cfg())
    assert report["prior_inspection_attested"] is True
    assert report["attester"] == "the custodian of the held-out material"
    # nobody handed the scorer the material, so the digest comparison is reported as not made rather
    # than as passed: those are different statements, as they are for the ladder
    assert report["heldout_material_checked"] is False

    # OFF THE DECIDING PATH the placeholder is honest and is accepted, and it says so three times.
    # It carries no digest, because the runner does not know what anybody read and may not answer a
    # question that was asked of somebody else.
    demo = _mock_run()
    placeholder = demo["manifest"]["seal"]["heldout_attestation"]
    assert placeholder["mock"] is True and placeholder["attester"] is None
    assert placeholder["heldout_sha256"] is None
    assert "PLACEHOLDER ATTESTATION" in placeholder["label"]
    assert "manufactured by" in placeholder["note"]
    assert custody.custody_failures(demo["manifest"], external_anchor_required=False) == []

    # ON IT the same placeholder is refused, and the deciding run cannot produce a verdict at all
    with pytest.raises(custody.CustodyRefusal) as exc:
        _confirmatory_run(tmp_path / "unattested", attestation=None)
    assert "prior-inspection attestation is the runner's own placeholder" in str(exc.value)


def test_an_attestation_for_another_artefact_or_none_at_all_refuses():
    """A sentence that names no material attests nothing, because anything at all can be said not to
    have been read; and a sentence about another artefact is not a sentence about this one. The
    record is written on the seal BEFORE the anchor, so it is inside the digest the receipt attests
    and cannot be added, removed or improved afterwards.

    The manifest is built here rather than run, because the subject is the record and every one of
    these cases is a question about the seal a run leaves behind. It names its observing system for
    the same reason it is given a real receipt and a named attestation: a deciding run holds all
    three, and a fixture missing one would stop this case on the requirement it is not about."""
    man = M.new_manifest("P5", False, "a-ladder-hash", {"seed": 1}, "adapter", mode="confirmatory",
                         confirmatory_inputs=MODE.ConfirmatoryInputs(
                             observing_systems=(ScriptedCodeSystem(4),)).as_record())
    M.seal_predictions(man, {"x": 1.0}, "test", anchor=anchor_service,
                       attestation=attestation_record())
    assert custody.custody_failures(man, external_anchor_required=True) == []

    # the material in hand is the material attested, and a scorer holding it may say so
    ours = hashlib.sha256(HELDOUT_MATERIAL).hexdigest()
    assert M.require_scoreable(man, heldout_sha256=ours)["heldout_material_checked"] is True
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man, heldout_sha256=hashlib.sha256(b"another suite entirely").hexdigest())
    assert "not an attestation for this one" in str(exc.value)

    # removing the sentence removes the attestation AND breaks the anchored digest
    stripped = json.loads(json.dumps(man))
    stripped["seal"].pop("heldout_attestation")
    failures = custody.custody_failures(stripped, external_anchor_required=True)
    assert any("no prior-inspection attestation" in f for f in failures)
    assert any("changed after it was anchored" in f for f in failures)

    # and so does replacing it with a more convenient one after the fact
    swapped = json.loads(json.dumps(man))
    swapped["seal"]["heldout_attestation"] = attestation_record(attester="somebody else entirely")
    assert any("changed after it was anchored" in f
               for f in custody.custody_failures(swapped, external_anchor_required=True))

    # the record refuses to be built out of half of itself, in either direction, and refuses to be
    # attached as something other than a record
    with pytest.raises(custody.CustodyRefusal):
        custody.attestation("", ours)
    with pytest.raises(custody.CustodyRefusal):
        custody.attestation("the custodian of the held-out material", "")
    with pytest.raises(custody.CustodyRefusal):
        custody.attach_attestation({"seal": {"sealed_by": "test"}}, "a sentence in a string")
    with pytest.raises(custody.CustodyRefusal):
        custody.attach_attestation({"seal": {"sealed_by": "test"}},
                                   {"attester": "the custodian of the held-out material"})
    with pytest.raises(custody.CustodyRefusal):
        custody.attach_attestation({"seal": None}, attestation_record())


def test_an_incomplete_attestation_is_refused_rather_than_completed_by_the_runner(tmp_path):
    """A REVIEW OF THE REPAIR ABOVE: the runner was writing the sentence it exists in order not to
    write.

    `attach_attestation` filled four fields in with `setdefault`, and two of them are the attester's
    own words. A record carrying an attester and a digest and nothing else therefore reached the seal
    with the shipped sentence written into it and a time this code had read off its own clock, and
    the deciding path then reported a named party's attestation that the named party had never made.
    The gate's own statement check could not catch it, because the line that composed the statement
    ran first. Nothing is filled in now: what is missing belongs to somebody outside this code, so an
    incomplete record is refused and the run stops before the seal rather than after it.
    """
    whole = attestation_record()
    assert whole["statement"] and whole["attested_utc"]      # the constructor's record is complete

    def _seal_with(rec):
        man = M.new_manifest("P5", False, "a-ladder-hash", {"seed": 1}, "adapter", mode="confirmatory")
        M.seal_predictions(man, {"x": 1.0}, "test", anchor=anchor_service, attestation=rec)
        return man

    for absent in ("statement", "attested_utc", "attester", "heldout_sha256"):
        short = {k: v for k, v in whole.items() if k != absent}
        with pytest.raises(custody.CustodyRefusal) as exc:
            _seal_with(short)
        assert absent in str(exc.value) and "party being checked" in str(exc.value)

    # the complete record reaches the seal unaltered, which is the whole of what this code may do
    # with it: the sentence and the time are the ones the attester made, not ones taken from a clock
    man = _seal_with(whole)
    att = man["seal"]["heldout_attestation"]
    assert att["statement"] == whole["statement"] and att["attested_utc"] == whole["attested_utc"]
    assert custody.custody_failures(man, external_anchor_required=True) == []

    # a seal assembled by hand around an undated sentence is refused on the deciding path, which is
    # now the only way an undated one can arrive
    undated = json.loads(json.dumps(man))
    undated["seal"]["heldout_attestation"].pop("attested_utc")
    assert any("carries no time" in f
               for f in custody.custody_failures(undated, external_anchor_required=True))

    # and an ABSENT record is untouched by all of it: the placeholder is still attached and labelled,
    # because a run that attests nothing must still carry the shape and say what it is
    demo = _mock_run()
    assert demo["manifest"]["seal"]["heldout_attestation"]["mock"] is True


def test_a_manifest_cannot_name_one_ladder_at_the_top_and_bind_another_below(tmp_path):
    """Two ladder fields, and nothing compared them with each other.

    `ladder_sha256` is the pool a reader of the manifest reads first and `ladder_identity` is what
    the verifier binding hangs off. While every identity was derived from that same string they
    agreed by construction; a caller holding the object passes both, and a run told about one pool
    and handed another wrote both, sealed the disagreement inside the specification hash and passed
    every custody check, because `ladder_differences` compares the live object with the identity that
    object itself produced and had nothing to say. A reader of such a record cannot tell which pool
    was read.
    """
    lad = CD.SuiteLadder(_registered_pool(20), subset_size=10, batch_runner=CD.inprocess_batch_runner())
    other = CD.SuiteLadder(_registered_pool(24), subset_size=10, batch_runner=CD.inprocess_batch_runner())
    assert lad.sha256 != other.sha256
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.new_manifest("P16", False, other.sha256, {"seed": 1}, "mock", ladder=lad)
    assert "one manifest cannot record both" in str(exc.value)

    # the run that used to write one is refused where the record is made, so no evidence is written.
    # The horizon clears the switch and the settling period, because a titration that does not is
    # refused on its shape before it reaches the record, and this case is about the record.
    cfg = p16.P16Config(systems_per_arm=1, horizon=24)
    with pytest.raises(custody.CustodyRefusal):
        p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2.0), cfg, 5, other.sha256, "mock",
                    bundle=str(tmp_path / "never-written"), ladder=lad)
    assert not os.path.exists(str(tmp_path / "never-written" / "bundle.json"))

    # an honest manifest holds, and the same manifest edited afterwards to name another pool is
    # caught at scoring time too: a seal preserves a contradiction it cannot see
    man = M.new_manifest("P16", False, lad.sha256, {"seed": 1}, "mock", ladder=lad)
    M.seal_predictions(man, {"x": 1.0}, "test", anchor=anchor_service, attestation=attestation_record())
    assert custody.custody_failures(man, external_anchor_required=True, ladder=lad) == []
    edited = json.loads(json.dumps(man))
    edited["ladder_sha256"] = other.sha256
    assert any("name different pools" in f for f in custody.custody_failures(edited, ladder=lad))


def test_a_bundle_says_whether_an_empty_read_list_is_no_log_or_no_reading(tmp_path):
    """An empty list of readings had three different meanings and one appearance.

    `reads` was written as `list(reads or [])`, so a run that attached no log, a run whose log
    recorded nothing and a run that genuinely read nothing all left the same empty list. The first is
    the ordinary state of a run holding only a ladder's hash; the second says the object the run
    named is not the object its source reads, which is the one failure the ladder argument makes
    possible and the only one anything here can observe. The provider block beside it already said in
    words when it had been given a name instead of an adapter.
    """
    n = 16
    cfg = p16.P16Config(systems_per_arm=1, horizon=16, switch_round=4, settling=2,
                        dose_offsets=(-0.3, 0.3))
    n_arms = (len(cfg.dose_offsets) + 2) * cfg.systems_per_arm

    def _real_source(where):
        lad = CD.SuiteLadder(_registered_pool(n), subset_size=n, batch_runner=CD.inprocess_batch_runner())
        store = CD.CheckpointStore(str(where))
        store.save("seed", CD.new_artefact(_lib_of({0, 1}, n)))
        adapter = RevisingCodeSystem(n)
        return lad, adapter, CD.suite_margin_source(adapter, lad, store, "Improve the solutions library.",
                                                    CD.DoseSchedule().dose_for, cfg.switch_round)

    # NO LOG: the run holds the hash and never the object, which is a complete and honest record
    lad, adapter, src = _real_source(tmp_path / "s1")
    p16.run_p16(src, cfg, 5, lad.sha256, adapter.name, bundle=str(tmp_path / "hash-only"))
    b = custody.load_bundle(str(tmp_path / "hash-only"))
    assert b["reads"] == [] and "no read log was attached" in b["reads_note"]
    assert adapter.calls, "the arms were collected; only the readings were never recorded"

    # A LOG THAT RECORDED EVERY READING
    lad, adapter, src = _real_source(tmp_path / "s2")
    p16.run_p16(src, cfg, 5, lad.sha256, adapter.name, bundle=str(tmp_path / "logged"),
                ladder=lad, adapter=adapter)
    b = custody.load_bundle(str(tmp_path / "logged"))
    assert len(b["reads"]) == n_arms * cfg.horizon
    assert "every reading this run took" in b["reads_note"]

    # A LOG THAT RECORDED NOTHING while the arms were collected: the named ladder is not the ladder
    # this source reads, and the identity sealed beside it is about a pool nothing here touched
    unread = CD.SuiteLadder(_registered_pool(n), subset_size=n, batch_runner=CD.inprocess_batch_runner())
    res = p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2.0), cfg, 5, unread.sha256, "mock",
                      bundle=str(tmp_path / "unread"), ladder=unread)
    b = custody.load_bundle(str(tmp_path / "unread"))
    assert res["arms"] and b["reads"] == []
    assert "is not the object its source reads" in b["reads_note"]


# --------------------------------------------------------------------------------------------------
# Acceptance case: relocation preserves identity, a changed verifier does not
# --------------------------------------------------------------------------------------------------

def test_relocating_unchanged_code_preserves_its_identity(tmp_path):
    """The defect: the old hash fed absolute paths into the digest, so the same bytes checked out in
    another directory were a different identity and nobody else could recompute it."""
    root = custody.package_root()
    other = tmp_path / "elsewhere" / "renamed-checkout"
    os.makedirs(other)
    for pkg in ("arc_runner", "arc_instruments"):
        shutil.copytree(os.path.join(root, pkg), os.path.join(str(other), pkg),
                        ignore=shutil.ignore_patterns("__pycache__"))
    assert custody.package_code_identity(str(other)) == custody.package_code_identity(root)
    # and a single changed byte anywhere in the tree is a different identity
    with open(os.path.join(str(other), "arc_runner", "ladder.py"), "a", encoding="utf-8") as fh:
        fh.write("\n# one added comment\n")
    assert custody.package_code_identity(str(other)) != custody.package_code_identity(root)


def test_the_length_delimiters_stop_a_rename_being_hidden_by_an_edit(tmp_path):
    """Without the lengths, a file named `a` holding `bc` and a file named `ab` holding `c` hash
    identically, so a rename could be concealed by a compensating edit."""
    d = tmp_path / "one"
    e = tmp_path / "two"
    os.makedirs(d); os.makedirs(e)
    (d / "a").write_text("bc", encoding="utf-8")
    (e / "ab").write_text("c", encoding="utf-8")
    assert custody.code_identity([str(d / "a")], str(d)) != custody.code_identity([str(e / "ab")], str(e))


def test_a_changed_verifier_changes_the_ladder_identity():
    """A pool hash says which items were asked. It says nothing about what counted as passing them,
    which is why the verifier implementation is bound beside it."""
    pool = _registered_pool(20)
    one = CD.SuiteLadder(pool, subset_size=10, batch_runner=CD.inprocess_batch_runner())

    def permissive_batch_runner(text, items):
        return {it.id for it in items}                 # a different rule for what counts as a pass

    two = CD.SuiteLadder(pool, subset_size=10, batch_runner=permissive_batch_runner)
    assert one.sha256 == two.sha256                    # the pool is identical, and so is the ladder spec
    a, b = custody.ladder_identity(one), custody.ladder_identity(two)
    assert a["verifier_sha256"] and b["verifier_sha256"]
    assert a["verifier_sha256"] != b["verifier_sha256"]
    assert a["all_bound"] and b["all_bound"]
    # the identity is in the manifest and inside the sealed specification hash, so a run scored with a
    # different verifier than it sealed under refuses
    man = M.new_manifest("P5", False, one.sha256, {"seed": 1}, "adapter", ladder=one)
    M.seal_predictions(man, {"x": 1.0}, "test")
    man["ladder_identity"] = b
    assert any("changed after sealing" in f for f in custody.custody_failures(man))


def test_a_simulated_ladder_records_that_no_verifier_was_bound():
    ident = custody.ladder_identity(L.MockLadder(n_items=100, scale=400.0))
    assert ident["verifier_sha256"] is None and ident["verifiers"] == []
    assert ident["simulated"] is True and ident["all_bound"] is False
    assert custody.ladder_identity("a-hash-only")["note"]


# --------------------------------------------------------------------------------------------------
# Acceptance case: the bundle is complete, is public, and can be re-scored on its own
# --------------------------------------------------------------------------------------------------

def test_a_saved_bundle_carries_the_whole_run_and_recomputes_its_own_verdict(tmp_path):
    res = _confirmatory_run(tmp_path, bundle=str(tmp_path / "evidence"))
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    assert bundle["schema"] == custody.BUNDLE_SCHEMA
    # the full bank rows, every read with its subset size and pass count, the replicate series
    assert len(bundle["bank"]["rows"]) == bundle["bank"]["cells"] + bundle["bank"]["control_cells"]
    assert bundle["reads"] and all({"subset_size", "passes"} <= set(r) for r in bundle["reads"])
    assert any(r["context"] and r["context"].startswith("bank ") for r in bundle["reads"])
    reps = bundle["heldout"]["fitted"]["S1"]["replicates"]
    assert reps and all(r["depths"] and r["scores"] for r in reps)
    # the manifest, the seal, the receipt and the provider metadata the adapter no longer discards
    assert bundle["manifest"]["execution_mode"] == "confirmatory"
    assert bundle["anchor_receipt"]["mock"] is False
    assert bundle["seal_sha256"] == bundle["anchor_receipt"]["attests_sha256"]
    assert bundle["provider"]["bank"]["calls"] > 0
    assert bundle["provider"]["bank"]["returned_models"] == ["scripted-code-v1"]
    # and the table comes back from the bundle alone
    again = custody.recompute_verdicts(bundle)
    assert again["PREDICTION"] == res["verdicts"]["PREDICTION"]
    assert again["IDENTIFICATION"] == res["verdicts"]["IDENTIFICATION"]
    assert again["per_system"].keys() == res["verdicts"]["per_system"].keys()
    # re-estimating the routes from the saved rows returns the run's own coupling, which is the check
    # that the saved rows are the rows the run analysed and not a summary of them
    routes = custody.reanalyse_bank(bundle)
    assert routes["beta_pooled"] == pytest.approx(res["routes"]["beta_pooled"], abs=1e-12)
    assert routes["beta_state_route"] == pytest.approx(res["routes"]["beta_state_route"], abs=1e-12)


def test_a_p16_bundle_keeps_every_margin_series_and_re_scores(tmp_path):
    cfg = p16.P16Config(systems_per_arm=2, horizon=60)
    res = p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2.0), cfg, 5, "none", "mock",
                      bundle=str(tmp_path / "evidence"))
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    assert len(bundle["arms"]) == len(res["arms"])
    assert all(len(a["margin"]) == cfg.horizon for a in bundle["arms"])
    assert "margin" not in res["arms"][0]              # the printed summary still carries the summary
    assert custody.recompute_verdicts(bundle)["P16"] == res["verdicts"]["P16"]


def test_a_p16_bundle_from_a_real_ladder_carries_every_read_and_the_provider_record(tmp_path):
    """THE OTHER HALF OF THE ACCEPTANCE CASE: an independent analyst regenerating every result from
    the saved evidence.

    THE DEFECT. It was covered for P5 alone. Both P16 bundle cases ran the simulated margin, which
    has no ladder and no adapter, so neither could see what a real titration was losing: `run_p16`
    turned no read log on and passed no readings and no provider record into the bundle. A real arm
    reads the WHOLE frozen pool once per round through `code_domain.suite_margin_source`, and every
    one of those readings was counted into the four balance objects and then discarded, while the
    model identifier and the usage figures the adapter had already recorded were thrown away on the
    only path where a P16 run has a provider. A saved P16 bundle could be re-scored from its own
    arms and could not be re-counted from anything.

    This is that run: a registered pool, a checkable ladder, a checkpoint store and an adapter, with
    the system simulated in this process and no provider called.
    """
    n = 16
    pool = _registered_pool(n)
    lad = CD.SuiteLadder(pool, subset_size=n, batch_runner=CD.inprocess_batch_runner())
    store = CD.CheckpointStore(str(tmp_path / "store"))
    store.save("seed", CD.new_artefact(_lib_of({0, 1}, n)))
    adapter = RevisingCodeSystem(n)
    cfg = p16.P16Config(systems_per_arm=1, horizon=16, switch_round=4, settling=2,
                        dose_offsets=(-0.3, 0.3))
    src = CD.suite_margin_source(adapter, lad, store, "Improve the solutions library.",
                                 CD.DoseSchedule().dose_for, cfg.switch_round)
    out = tmp_path / "evidence"
    res = p16.run_p16(src, cfg, 5, lad.sha256, adapter.name, bundle=str(out), ladder=lad,
                      adapter=adapter)
    bundle = custody.load_bundle(str(out))

    n_arms = (len(cfg.dose_offsets) + 2) * cfg.systems_per_arm
    assert len(bundle["arms"]) == n_arms
    # one whole-pool read per arm per round, all of them in the bundle and none of them a summary
    assert len(bundle["reads"]) == n_arms * cfg.horizon
    assert all(r["context"] == "whole-pool" for r in bundle["reads"])
    assert all({"subset_size", "passes", "form_sha256", "outcome_sha256", "artefact_sha256"} <= set(r)
               for r in bundle["reads"])
    assert all(r["population_size"] == n and r["subset_size"] == n for r in bundle["reads"])
    # each reading is of its own artefact, which is what makes the count traceable to a response
    assert len({r["artefact_sha256"] for r in bundle["reads"]}) == len(bundle["reads"])
    # and they reach disk as the arms are collected rather than living in memory until the last write
    progress = custody.load_progress(str(out))
    assert sum(len(line["reads"]) for line in progress) == len(bundle["reads"])
    # the provider's own account of what it served, which a P16 bundle used to discard. The count is
    # the adapter's own and not a recount here: at least one revision per arm per round, and exactly
    # what the thing that made them says it made.
    assert bundle["provider"]["arms"]["calls"] == adapter.calls >= n_arms * cfg.horizon
    assert bundle["provider"]["arms"]["returned_models"] == ["revising-code-v1"]
    # the ladder that took the readings is the identity the manifest sealed, verifiers bound, and it
    # is the pool the manifest names at the top as well: two fields, and they may not disagree
    assert res["manifest"]["ladder_identity"]["ladder_sha256"] == lad.sha256
    assert res["manifest"]["ladder_sha256"] == lad.sha256
    assert res["manifest"]["ladder_identity"]["all_bound"] is True
    # THE TABLE COMES BACK FROM THE BUNDLE ALONE, and this asserts more than the one word it used to.
    # Under the shipped unregistered bands a P16 verdict reads NOT EVALUABLE, so comparing that word
    # with itself compares two constants and would hold against a re-scoring that read nothing at
    # all. What is compared is the whole table, every component's own state, the line fitted from the
    # arms and the per-arm estimates it is fitted from; and dropping the arms from a copy changes
    # what comes back, which is what says the numbers were recomputed rather than read back.
    def _table(v):
        # canonical text and not field equality, because an unresolved figure in this table is NaN
        # and NaN is equal to nothing, including the same NaN recomputed from the same rows
        return json.dumps(v, sort_keys=True, default=str)

    again = custody.recompute_verdicts(bundle)
    assert again["P16"] == res["verdicts"]["P16"]
    assert again["component_states"] == res["verdicts"]["component_states"]
    assert _table(again) == _table(res["verdicts"])          # every field of it, not the word alone
    thinned = json.loads(json.dumps(bundle))
    thinned["arms"] = thinned["arms"][:1]
    assert _table(custody.recompute_verdicts(thinned)) != _table(again)
    # and the hidden suite never travels with the evidence
    text = open(os.path.join(str(out), "bundle.json"), encoding="utf-8").read()
    assert "assert add_0(3, 4) == 7" not in text


def test_the_bundle_never_carries_an_answer_key_a_hidden_check_or_a_credential(tmp_path):
    bundle = custody.EvidenceBundle(str(tmp_path / "evidence"))
    bundle.write_bundle({"schema": custody.BUNDLE_SCHEMA, "rows": [{"passes": 3, "answer": "42"}],
                         "item": {"checks": ["assert f(1) == 2"], "statement": "kept"},
                         "env": {"api_key": "sk-not-in-a-bundle", "base_url": "kept"}})
    text = open(os.path.join(str(tmp_path / "evidence"), "bundle.json"), encoding="utf-8").read()
    for forbidden in ("42", "assert f(1) == 2", "sk-not-in-a-bundle"):
        assert forbidden not in text
    saved = custody.load_bundle(str(tmp_path / "evidence"))
    assert saved["rows"][0]["passes"] == 3 and saved["item"]["statement"] == "kept"
    assert saved["env"]["base_url"] == "kept"
    assert "redacted_paths" in saved and len(saved["redacted_paths"]) == 3


def test_a_real_ladder_bundle_carries_no_check_from_the_hidden_suite(tmp_path):
    """The suite ladder's spec names the pool by hash and never by content, and the bundle's identity
    block is written through the same redaction as everything else."""
    res = _confirmatory_run(tmp_path, bundle=str(tmp_path / "evidence"), n=120)
    text = open(os.path.join(str(tmp_path / "evidence"), "bundle.json"), encoding="utf-8").read()
    assert "assert add_0(3, 4) == 7" not in text
    assert res["manifest"]["ladder_identity"]["ladder_sha256"] in text


# --------------------------------------------------------------------------------------------------
# Acceptance case: a crash after the seal leaves the seal and the receipt on disk
# --------------------------------------------------------------------------------------------------

class _DiesAfterTheSeal:
    """A system that answers the calibration window and then stops, which is where a real run stops:
    after the commitment and before the held-out continuation is complete."""
    name = "dies-after-the-seal"

    def __init__(self, n_total, die_after):
        self.inner = ScriptedCodeSystem(n_total)
        self.die_after, self.calls = die_after, 0

    def metadata(self):
        return self.inner.metadata()

    def revise(self, artefact, retained, task, rng):
        self.calls += 1
        if self.calls > self.die_after:
            raise RuntimeError("the provider went away mid-continuation")
        return self.inner.revise(artefact, retained, task, rng)


def test_a_crash_after_the_seal_leaves_the_seal_and_the_receipt_on_disk(tmp_path):
    n = 120
    pool = _registered_pool(n)
    lad = CD.SuiteLadder(pool, subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path / "store", n)
    cfg = _cfg()
    # enough calls for the bank and the calibration window, then nothing: the run dies with its
    # commitment made and its continuation incomplete
    n_bank = len(cfg.states) * len(cfg.fractions) * (cfg.reps + max(1, int(round(cfg.reps * cfg.control_fraction_of_bank))))
    adapter = _DiesAfterTheSeal(n, n_bank + len(cfg.calibration_depths))
    out = tmp_path / "evidence"
    with pytest.raises(RuntimeError):
        p5.run_p5(adapter, lad, cfg, 11, CD.place_at_state_factory(store), CD.start_for_factory(store),
                  ["S1"], mode="confirmatory", confirmatory_inputs=_inputs(store), bundle=str(out))
    seal = json.load(open(os.path.join(str(out), "seal.json"), encoding="utf-8"))
    manifest = json.load(open(os.path.join(str(out), "manifest.json"), encoding="utf-8"))
    assert seal["seal"]["predictions_sha256"] and seal["anchor_receipt"]["anchor_identifier"]
    assert seal["anchor_receipt"]["mock"] is False
    assert seal["seal_sha256"] == custody.seal_digest(manifest["seal"])
    assert not os.path.exists(os.path.join(str(out), "bundle.json"))     # the run never finished
    # the commitment on disk is checkable on its own: it is the same seal, and its receipt still holds
    assert custody.custody_failures(manifest, external_anchor_required=True) == []


def test_the_manifest_reaches_disk_before_the_first_paid_call(tmp_path):
    """A run that dies inside the bank has spent money and must still leave the setup it spent under."""
    n = 120
    pool = _registered_pool(n)
    lad = CD.SuiteLadder(pool, subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path / "store", n)
    out = tmp_path / "evidence"
    with pytest.raises(RuntimeError):
        p5.run_p5(_DiesAfterTheSeal(n, 0), lad, _cfg(), 11, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store), bundle=str(out))
    manifest = json.load(open(os.path.join(str(out), "manifest.json"), encoding="utf-8"))
    assert manifest["execution_mode"] == "confirmatory" and manifest["seal"] is None
    assert manifest["confirmatory_inputs"]["anchor_service"] == "test-stub"


# --------------------------------------------------------------------------------------------------
# The gate keeps the anchoring service honest before the money, not after it
# --------------------------------------------------------------------------------------------------

def test_a_confirmatory_run_without_an_anchoring_service_refuses_before_the_first_paid_call(tmp_path):
    class RefusingAdapter:
        name = "must-not-be-called"

        def revise(self, artefact, retained, task, rng):
            raise AssertionError("a provider call was made after a confirmatory run should have refused")

    n = 40
    lad = CD.SuiteLadder(_registered_pool(n), subset_size=20, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path / "store", n)
    with pytest.raises(MODE.ModeRefusal) as exc:
        p5.run_p5(RefusingAdapter(), lad, _cfg(), 1, CD.place_at_state_factory(store),
                  CD.start_for_factory(store), ["S1"], mode="confirmatory",
                  confirmatory_inputs=_inputs(store, anchor=None))
    assert any(r.startswith("anchor-service") for r in exc.value.requirements)


def test_the_command_line_takes_the_anchoring_service_by_module_name_and_never_invents_one(tmp_path,
                                                                                          monkeypatch):
    from arc_runner import cli
    monkeypatch.delenv("ARC_RUNNER_API_KEY", raising=False)
    store = CD.CheckpointStore(str(tmp_path / "store"))
    store.save("seed", CD.new_artefact(_lib(1, 4)))
    with pytest.raises(SystemExit) as exc:
        cli.main(["confirm", "p5", "--checkpoints", str(tmp_path / "store"),
                  "--pool-module", "tests.pool_for_tests", "--allowance-gbp", "5",
                  "--resolved-by", "the operator"])
    assert "anchor-service" in str(exc.value)


def test_a_demonstration_writes_a_bundle_whose_mode_travels_with_it(tmp_path):
    res = _mock_run(bundle=str(tmp_path / "evidence"))
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    assert bundle["execution_mode"] == "demonstration"
    assert bundle["scoreable_at_proposition_level"] is False
    assert "simulated recovery" in bundle["mode_label"]
    assert bundle["provider"]["bank"]["simulated"] is True
    assert custody.recompute_verdicts(bundle)["PREDICTION"] == res["verdicts"]["PREDICTION"]
    with pytest.raises(M.NotScoreable):
        M.require_scoreable(bundle["manifest"])          # the bundle does not promote its own mode


# --------------------------------------------------------------------------------------------------
# Acceptance case: the commitment commits to the run's own kind, so a rehearsal cannot be promoted
# --------------------------------------------------------------------------------------------------

def test_a_rehearsal_relabelled_a_deciding_run_refuses(tmp_path):
    """THE DEFECT, DEMONSTRATED BEFORE IT WAS REPAIRED. The sealed specification hash covered the
    ladder, the code, the configuration and the ladder identity, and the anchored digest covered the
    seal record, which carries no mode. `execution_mode`, `pilot`,
    `scoreable_at_proposition_level`, the adapter, the experiment and the whole confirmatory-inputs
    record therefore sat outside both. A demonstration against the simulated system, honestly
    anchored by a real service (which the shipped command line does in every mode), could have two
    fields rewritten in its own bundle and would then pass every custody check as a confirmatory run:
    no failures, a verified report with a non-mock identifier, and a proposition-level table. That is
    the promotion ruling 28 forbids.
    """
    res = _mock_run(bundle=str(tmp_path / "evidence"), anchor=anchor_service,
                    attestation=attestation_record())
    honest = res["manifest"]
    assert honest["execution_mode"] == "demonstration"
    assert honest["seal"]["anchor_receipt"]["mock"] is False       # the rehearsal is honestly anchored
    assert honest["seal"]["heldout_attestation"]["mock"] is False  # and honestly attested
    assert custody.custody_failures(honest, external_anchor_required=True) == []

    promoted = json.loads(json.dumps(honest))
    promoted["execution_mode"] = "confirmatory"
    promoted["scoreable_at_proposition_level"] = True
    promoted["simulated"] = False
    failures = custody.custody_failures(promoted, external_anchor_required=True)
    assert failures and any("changed after sealing" in f for f in failures)
    with pytest.raises(custody.CustodyRefusal):
        M.require_scoreable(promoted)

    # and through the saved bundle, which is the surface an outside reader actually receives
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    bundle["manifest"]["execution_mode"] = "confirmatory"
    bundle["manifest"]["scoreable_at_proposition_level"] = True
    with pytest.raises(custody.CustodyRefusal):
        custody.recompute_verdicts(bundle)


def test_a_pilot_relabelled_a_deciding_run_refuses_and_so_does_every_other_unsealed_field(tmp_path):
    """Ruling 28 says a pilot is never scored, and a pilot whose mode field can be edited is scored by
    anybody willing to edit it. The same commitment covers the fields that travel with the mode: the
    adapter, the experiment and the confirmatory-inputs record, each of which describes what the run
    was and none of which was inside the seal."""
    res = _mock_run(mode="pilot", anchor=anchor_service)
    man = res["manifest"]
    assert man["pilot"] is True
    with pytest.raises(M.PilotNotScoreable):
        M.require_scoreable(man)

    promoted = json.loads(json.dumps(man))
    promoted["pilot"] = False
    promoted["execution_mode"] = "confirmatory"
    promoted["scoreable_at_proposition_level"] = True
    with pytest.raises(custody.CustodyRefusal):
        M.require_scoreable(promoted)

    for field_name, value in (("adapter", "some-other-provider"),
                              ("experiment", "P16"),
                              ("confirmatory_inputs", {"anchor_service": "invented"}),
                              ("mode_label", "CONFIRMATORY: read this at proposition level")):
        edited = json.loads(json.dumps(man))
        edited[field_name] = value
        assert any("changed after sealing" in f for f in custody.custody_failures(edited)), field_name


def test_the_specification_hash_names_its_own_recipe(tmp_path):
    """The recipe changed, so it says which recipe it is. A reader who finds version 1 in an old seal
    is reading a commitment that did not bind the mode, which is a fact about that seal."""
    res = _mock_run(anchor=anchor_service)
    man = res["manifest"]
    assert custody.SPEC_HASH_SCHEMA.endswith("/2")
    assert man["seal"]["spec_sha256"] == custody.spec_hash_of(man)
    # the mode is genuinely an input to the hash and not merely recorded beside it
    other = dict(man, execution_mode="confirmatory")
    assert custody.spec_hash_of(other) != custody.spec_hash_of(man)


# --------------------------------------------------------------------------------------------------
# Acceptance case: an independent analyst needs the dependencies, not only the package bytes
# --------------------------------------------------------------------------------------------------

def test_the_manifest_carries_a_dependency_manifest_read_from_the_code(tmp_path):
    """Finding A8's action asks for relative-path, length-delimited entries AND a manifest of all
    required dependencies. The fits go through scipy's optimiser and the streams through numpy's
    generator, so a result regenerated under unstated versions of those was regenerated under an
    unstated method."""
    env = custody.environment_manifest()
    assert env["schema"] == custody.DEPENDENCY_SCHEMA
    assert env["python_version"] and env["environment_sha256"]
    # read from the imports rather than from a hand-written list, so it cannot drift from the code
    assert "numpy" in env["dependencies"] and "scipy" in env["dependencies"]
    assert env["dependencies"]["numpy"]["version"]
    for stdlib_name in ("json", "hashlib", "typing", "dataclasses"):
        assert stdlib_name not in env["dependencies"]
    assert "numpy" in custody.imported_top_level_names()

    res = _mock_run(bundle=str(tmp_path / "evidence"), anchor=anchor_service)
    assert res["manifest"]["environment"]["environment_sha256"] == env["environment_sha256"]
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    assert bundle["code_identity"]["environment"]["dependencies"]["numpy"]["version"]
    assert "environment_differences" in bundle["how_to_check"]


def test_a_changed_dependency_is_named_and_an_edited_dependency_record_refuses(tmp_path):
    """The comparison is REPORTED and not refused by default, which is the conservative reading named
    in `environment_differences`; what is refused is the recorded manifest being edited after the
    seal, because that is a change to the commitment rather than a change to the analyst's machine."""
    res = _mock_run(anchor=anchor_service)
    man = res["manifest"]
    live = custody.environment_manifest()
    assert custody.environment_differences(man["environment"], live) == []

    pretend = json.loads(json.dumps(live))
    pretend["dependencies"]["numpy"]["version"] = "0.0.0-not-this-one"
    pretend["python_version"] = "2.7.0"
    diffs = custody.environment_differences(man["environment"], pretend)
    assert any("numpy" in d for d in diffs) and any("python_version" in d for d in diffs)
    assert custody.environment_differences(None)          # no manifest at all is itself a difference

    # off by default, and asked for when an analyst wants it
    assert custody.custody_failures(man) == []
    assert custody.custody_failures(man, verify_environment=True) == []
    edited = json.loads(json.dumps(man))
    edited["environment"]["dependencies"]["numpy"]["version"] = "0.0.0-not-this-one"
    assert any("changed after sealing" in f for f in custody.custody_failures(edited))


# --------------------------------------------------------------------------------------------------
# Acceptance case: a changed verifier is refused by the RUN, not only by the identity function
# --------------------------------------------------------------------------------------------------

def test_a_verifier_swapped_after_the_seal_refuses_at_scoring(tmp_path):
    """THE DEFECT. The ladder identity was captured once when the manifest was made and thereafter
    only ever compared with itself through the specification hash, so it caught somebody hand-editing
    the stored record and caught nothing else. Sealing under one checking rule and then replacing the
    ladder's own rule with a permissive one left the recorded identity untouched, the specification
    hash intact and every custody check passing. The binding was documentary; it is now recomputed
    from the live object."""
    n = 120
    pool = _registered_pool(n)
    lad = CD.SuiteLadder(pool, subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path / "store", n)
    cfg = _cfg()
    res = p5.run_p5(ScriptedCodeSystem(n), lad, cfg, 11, CD.place_at_state_factory(store),
                    CD.start_for_factory(store), ["S1"], mode="confirmatory",
                    confirmatory_inputs=_inputs(store))
    man, heldout = res["manifest"], res["heldout"]
    # as run, with the ladder that did the scoring, it holds
    report = M.require_scoreable(man, predictions=heldout["sealed_predictions"], config=cfg, ladder=lad)
    assert report["ladder_checked"] is True

    lad.batch_runner = lambda text, items: {it.id for it in items}       # everything passes now
    assert lad.sha256 == man["ladder_sha256"]                # the pool is untouched, and so is the spec
    assert custody.custody_failures(man, external_anchor_required=True) == []   # nothing else can see it
    failures = custody.custody_failures(man, external_anchor_required=True, ladder=lad)
    assert any("verifier implementations" in f for f in failures)
    with pytest.raises(custody.CustodyRefusal) as exc:
        M.require_scoreable(man, predictions=heldout["sealed_predictions"], config=cfg, ladder=lad)
    assert "verifier implementations" in str(exc.value)
    # and the verdict path refuses, which is where the run itself meets it
    with pytest.raises(custody.CustodyRefusal):
        p5.verdicts(man, res["routes"], heldout, cfg, ladder=lad)


def test_a_re_scoring_with_no_ladder_says_the_check_was_not_performed(tmp_path):
    """A bundle re-scored by an analyst has no ladder object to recompute, and the report must say
    that rather than say the check passed: those are different statements."""
    res = _confirmatory_run(tmp_path, bundle=str(tmp_path / "evidence"))
    report = res["verdicts"]["custody"]
    assert report["verified"] is True and report["ladder_checked"] is True
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    again = custody.recompute_verdicts(bundle)
    assert again["custody"]["verified"] is True
    assert again["custody"]["ladder_checked"] is False
    assert again["custody"]["execution_mode_sealed"] == "confirmatory"


# --------------------------------------------------------------------------------------------------
# Acceptance case: a crash MIDWAY THROUGH COLLECTION, which the seal file does not cover
# --------------------------------------------------------------------------------------------------

class _DiesDuringTheContinuation:
    """A system that answers the bank, the calibration windows and part of the held-out panel, then
    stops. This is the case a stop at a boundary does not exercise: the commitment is made, some of
    the continuation is bought and paid for, and the run never reaches its final write."""
    name = "dies-during-the-continuation"

    def __init__(self, n_total, die_after):
        self.inner = ScriptedCodeSystem(n_total)
        self.die_after, self.calls = die_after, 0

    def metadata(self):
        return self.inner.metadata()

    def revise(self, artefact, retained, task, rng):
        self.calls += 1
        if self.calls > self.die_after:
            raise RuntimeError("the provider went away part-way through the held-out panel")
        return self.inner.revise(artefact, retained, task, rng)


def test_a_crash_midway_through_collection_leaves_what_had_been_collected(tmp_path):
    """THE DEFECT. `bundle.json` was written once, at the end, so a run that stopped anywhere before
    that left its commitment and none of the rows it had already paid for. The seal file covers the
    stop immediately after sealing and covers nothing that happens afterwards, so the second
    acceptance case was satisfied only by the record the first one already produced."""
    n = 120
    pool = _registered_pool(n)
    lad = CD.SuiteLadder(pool, subset_size=40, batch_runner=CD.inprocess_batch_runner())
    store = _store(tmp_path / "store", n)
    cfg = _cfg()
    n_bank = len(cfg.states) * len(cfg.fractions) * (
        cfg.reps + max(1, int(round(cfg.reps * cfg.control_fraction_of_bank))))
    # the bank, both calibration windows, and the FIRST system's replicate continuations, then
    # nothing. A continuation runs from the sealed checkpoint to the final one, so it is one call per
    # round of that window and not one per checkpoint.
    rounds_per_replicate = max(cfg.checkpoints) - cfg.calibration_depths[-1]
    die_after = (n_bank + 2 * len(cfg.calibration_depths)
                 + cfg.replicates * rounds_per_replicate)
    out = tmp_path / "evidence"
    with pytest.raises(RuntimeError):
        p5.run_p5(_DiesDuringTheContinuation(n, die_after), lad, cfg, 11,
                  CD.place_at_state_factory(store), CD.start_for_factory(store), ["S1", "S2"],
                  mode="confirmatory", confirmatory_inputs=_inputs(store), bundle=str(out))
    assert not os.path.exists(os.path.join(str(out), "bundle.json"))     # the run never finished
    progress = custody.load_progress(str(out))
    stages = [line["stage"] for line in progress]
    assert "bank" in stages and stages.count("calibration") == 2
    assert "heldout" in stages, "the systems already continued must be on disk"
    bank_line = next(l for l in progress if l["stage"] == "bank")
    assert bank_line["bank"]["rows"], "the bank rows were paid for and must survive the stop"
    heldout_line = next(l for l in progress if l["stage"] == "heldout")
    assert heldout_line["system"] == "S1"
    assert heldout_line["fitted"]["replicates"][0]["scores"]
    # and the readings taken so far travel with the milestones rather than dying in memory
    assert sum(len(l["reads"]) for l in progress) > 0
    assert all({"subset_size", "passes"} <= set(r) for l in progress for r in l["reads"])
    # the commitment is still there beside the partial collection, and still checks out
    manifest = json.load(open(os.path.join(str(out), "manifest.json"), encoding="utf-8"))
    assert custody.custody_failures(manifest, external_anchor_required=True) == []


def test_a_p16_run_records_each_arm_as_it_is_collected(tmp_path):
    cfg = p16.P16Config(systems_per_arm=1, horizon=40)
    out = tmp_path / "evidence"
    p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2.0), cfg, 5, "none", "mock",
                bundle=str(out))
    progress = custody.load_progress(str(out))
    assert [l["stage"] for l in progress] == ["arm"] * (len(cfg.dose_offsets) + 2)
    assert all(len(l["arm"]["margin"]) == cfg.horizon for l in progress)


# --------------------------------------------------------------------------------------------------
# The access boundary: not only the names somebody remembered to list
# --------------------------------------------------------------------------------------------------

def test_held_out_material_under_an_unlisted_key_and_a_credential_by_shape_are_both_removed(tmp_path):
    """THE DEFECT. Redaction matched whole key names and five substrings, so a held-out value filed
    under `hidden_answer` or `gold` travelled through unchanged and a token pasted into a note was
    never looked at. The boundary rested on naming discipline the docstring itself called crude."""
    bundle = custody.EvidenceBundle(str(tmp_path / "evidence"))
    bundle.write_bundle({"schema": custody.BUNDLE_SCHEMA,
                         "row": {"hidden_answer": "forty-two", "gold": "also-forty-two",
                                 "expected_solutions": ["one", "two"], "passes": 3},
                         "note": "the endpoint was called with Authorization: Bearer "
                                 "abcdefghijklmnopqrstuvwxyz012345",
                         # the fields the word rule must NOT eat, because they are the registered
                         # configuration and the run's own setup
                         "config": {"checkpoints": [4, 8], "checkpoint_store_root": "/tmp/store",
                                    "total_tokens": 91}})
    text = open(os.path.join(str(tmp_path / "evidence"), "bundle.json"), encoding="utf-8").read()
    for forbidden in ("forty-two", "also-forty-two", "abcdefghijklmnopqrstuvwxyz012345"):
        assert forbidden not in text
    saved = custody.load_bundle(str(tmp_path / "evidence"))
    assert saved["row"] == {"passes": 3}
    assert custody.CREDENTIAL_PLACEHOLDER in saved["note"]
    assert saved["config"] == {"checkpoints": [4, 8], "checkpoint_store_root": "/tmp/store",
                               "total_tokens": 91}
    assert len(saved["redacted_paths"]) == 4


def test_the_word_rule_never_eats_the_registered_configuration(tmp_path):
    """A boundary that removed `checkpoints` would break the specification hash it exists to protect,
    so the word rule deliberately does not treat `check` or `key` as words. They stay whole key names,
    where they cannot reach either field."""
    for safe in ("checkpoints", "checkpoint_store_root", "calibration_depths", "total_tokens",
                 "monkeypatch", "keyword_arguments"):
        assert not custody._redacted_key(safe), safe
    for removed in ("answer", "hidden_answer", "gold", "expected_solutions", "api_key",
                    "ANSWER_KEY", "checks", "hidden_checks", "user_password"):
        assert custody._redacted_key(removed), removed


# --------------------------------------------------------------------------------------------------
# The record binds a count to a reading, and a call to its own text
# --------------------------------------------------------------------------------------------------

def test_a_read_binds_its_count_to_the_form_the_outcome_and_the_artefact(tmp_path):
    """THE DEFECT. The read log carried a subset size, a pass count, a ceiling flag, a population size
    and a sampling unit. An analyst could re-count those numbers and re-fit them and could not check
    that any count came from a reading of anything. The three digests are what a holder of the hidden
    suite and the checkpoint store recomputes; they are digests and not identifiers because the
    bundle is public and the suite is not."""
    res = _confirmatory_run(tmp_path, bundle=str(tmp_path / "evidence"))
    bundle = custody.load_bundle(str(tmp_path / "evidence"))
    reads = bundle["reads"]
    assert reads and all(r["form_sha256"] and r["outcome_sha256"] and r["artefact_sha256"]
                         for r in reads)
    # the digests distinguish the readings rather than being a constant stamped on all of them
    assert len({r["outcome_sha256"] for r in reads}) > 1
    # and nothing recomputable from the bundle alone discloses the suite
    text = open(os.path.join(str(tmp_path / "evidence"), "bundle.json"), encoding="utf-8").read()
    assert "assert add_0(3, 4) == 7" not in text
    del res

    # the outcome digest is a function of WHICH items passed, not only of how many
    items = ["a", "b", "c"]
    two_of_three = L.outcome_digest(items, {"a", "b"})
    other_two = L.outcome_digest(items, {"b", "c"})
    assert two_of_three != other_two
    assert two_of_three == L.outcome_digest(items, {"b", "a"})     # a set is a set
    assert L.artefact_digest({"text": "one"}) != L.artefact_digest({"text": "two"})
    assert L.artefact_digest(None) is None


def test_the_provider_record_binds_each_call_to_its_own_text_without_carrying_it(monkeypatch):
    """Finding A8 names a raw-response record. The text itself cannot travel: a response here is an
    artefact that passes the hidden suite's checks, so a public bundle carrying it publishes solutions
    to the held-out material. The digest and the length are what bind a call to an artefact held in
    the checkpoint store, and the key appears in none of it."""
    monkeypatch.setenv("ARC_RUNNER_MODEL", "a-model")
    monkeypatch.setenv("ARC_RUNNER_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("ARC_RUNNER_API_KEY", "sk-must-never-appear-in-any-record")
    ad = adapters.OpenAICompatibleAdapter()
    ad._record({"id": "resp-1", "model": "a-model-actually-served", "usage": {"total_tokens": 12},
                "choices": [{"finish_reason": "stop"}]},
               prompt="Task: improve it.", response="the revised artefact")
    rec = ad.metadata()["records"][0]
    assert rec["response_chars"] == len("the revised artefact")
    assert rec["response_sha256"] and rec["prompt_sha256"]
    assert rec["model_returned"] == "a-model-actually-served"
    assert ad.metadata()["model_substituted"] is True
    text = json.dumps(ad.metadata())
    for absent in ("sk-must-never-appear-in-any-record", "the revised artefact", "Task: improve it."):
        assert absent not in text
