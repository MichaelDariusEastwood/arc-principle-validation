"""Regression cases for the seven defects found while reviewing public PR #38."""
import copy
import functools
import json

import numpy as np
import pytest

from arc_instruments import parity
from arc_runner import calibration_gate as CG, custody, code_domain as CD, mode, p16
from arc_runner import observation as OBS
from trusted_fixtures import with_fixture_calibration


def test_reusing_a_bundle_directory_preserves_every_byte(tmp_path):
    path = tmp_path / "run"
    bundle = custody.EvidenceBundle(path)
    bundle.write_manifest({"original": 1})
    bundle.record_progress("original", {"row": 1})
    before = {f.name: f.read_bytes() for f in path.iterdir()}
    with pytest.raises(FileExistsError):
        custody.EvidenceBundle(path)
    with pytest.raises(FileExistsError):
        bundle.write_manifest({"replacement": 2})
    assert before == {f.name: f.read_bytes() for f in path.iterdir()}


def test_parity_refuses_extra_files_and_empty_or_missing_roots(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    assert not parity.compare_trees(str(left), str(right))["parity"]
    left.mkdir(); right.mkdir()
    assert not parity.compare_trees(str(left), str(right))["parity"]
    for root in (left, right):
        (root / "a.py").write_text("a = 1\n")
    assert parity.main([str(left), str(right)]) == 0
    (right / "shadow.py").write_text("raise RuntimeError('unexpected module')\n")
    assert parity.main([str(left), str(right)]) == 1


@pytest.mark.parametrize("factory, expected", [(OBS.service_ratio_observation, 1.0),
                                             (OBS.log_service_ratio_observation, 0.0)])
def test_ratio_is_derived_even_when_cached_value_is_supplied(factory, expected):
    spec = factory(supplies_q_and_w=True)
    read = OBS.read(lambda *args: {"Q": 1.0, "W": 1.0, "value": 123.0, "R": 1.0},
                    "sham", 2.0, 0, None, spec)
    assert read.value == expected
    assert OBS.normalise_reading(OBS.Reading.from_dict(read.as_dict()), spec).value == expected


@pytest.mark.parametrize("source", [lambda *args: {"round": 8, "value": 1},
                                    lambda *args: OBS.Reading(8, value=1)])
def test_wrong_round_is_refused_instead_of_relabelled(source):
    with pytest.raises(OBS.ObservationRefusal, match="requested round"):
        OBS.read(source, "sham", 2.0, 0, None, OBS.balance_elasticity_observation())


def _bundle(tmp_path):
    cfg = p16.P16Config(systems_per_arm=1, horizon=24, dose_offsets=(-0.3, 0.3),
                        switch_round=4, settling=2)
    p16.run_p16(p16.mock_margin_source(cfg, true_alpha_crit=2), cfg, 1, "none", "mock",
                bundle=tmp_path / "run")
    return custody.load_bundle(str(tmp_path / "run"))


def test_replay_ignores_forged_derived_summaries(tmp_path):
    bundle = _bundle(tmp_path)
    before = custody.recompute_verdicts(bundle)
    for arm in bundle["arms"]:
        arm.update(arm_delta=123, declared_round=0, terminal={}, delivery={}, realised_exposure={},
                   change_point={}, event="FORGED", observation_status="FORGED")
    assert json.dumps(custody.recompute_verdicts(bundle), sort_keys=True) == json.dumps(before, sort_keys=True)


@pytest.mark.parametrize("damage", ["drop_round", "duplicate_round", "drop_arm", "duplicate_arm", "alpha"])
def test_replay_refuses_incomplete_or_misassigned_measurements(tmp_path, damage):
    bundle = _bundle(tmp_path)
    if damage == "drop_round":
        bundle["arms"][0]["readings"].pop()
    elif damage == "duplicate_round":
        bundle["arms"][0]["readings"][1]["round"] = 0
    elif damage == "drop_arm":
        bundle["arms"].pop()
    elif damage == "duplicate_arm":
        bundle["arms"].append(copy.deepcopy(bundle["arms"][0]))
    else:
        bundle["arms"][0]["alpha"] = 99
    with pytest.raises((custody.CustodyRefusal, OBS.ObservationRefusal)):
        custody.recompute_verdicts(bundle)


def test_changes_to_raw_measurements_change_recomputed_estimate(tmp_path):
    bundle = _bundle(tmp_path)
    cfg = p16.P16Config(**{k: v for k, v in bundle["config"].items() if k in p16.P16Config.__dataclass_fields__})
    saved = bundle["arms"][0]
    spec = OBS.ObservationSpec.from_record(bundle["sealed_predictions"]["observation"])
    readings = [OBS.Reading.from_dict(r) for r in saved["readings"]]
    original = p16.analyse_arm(readings, saved["arm"], saved["alpha"], cfg, spec)
    altered = [OBS.Reading(r.round_index, value=4 + r.round_index) for r in readings]
    rebuilt = p16.analyse_arm(altered, saved["arm"], saved["alpha"], cfg, spec)
    assert rebuilt["arm_delta"] != pytest.approx(original["arm_delta"])


@pytest.mark.parametrize("stamp", ["not-a-date", "2030-01-01T00:00:00", "2100-01-01T00:00:00Z"])
def test_bad_attestation_stops_before_collection(stamp):
    rec = custody.attestation("test-only", "a" * 64, attested_utc=stamp)
    assert mode._attestation_refusals(mode.ConfirmatoryInputs(attestation=rec))


def test_late_attestation_is_refused_against_the_seal(tmp_path):
    man = _bundle(tmp_path)["manifest"]
    man["seal"]["heldout_attestation"] = custody.attestation(
        "test-only", "a" * 64, attested_utc="2100-01-01T00:00:00Z")
    assert any("after the prediction seal" in f for f in custody.custody_failures(man, external_anchor_required=True))
    assert custody.parse_utc("2026-09-06T01:00:00+01:00") == custody.parse_utc("2026-09-06T00:00:00Z")


def test_replay_refuses_an_unsealed_configuration_change(tmp_path):
    bundle = _bundle(tmp_path)
    bundle["config"]["z_threshold"] = 1.0
    with pytest.raises(custody.CustodyRefusal, match="configuration differs"):
        custody.recompute_verdicts(bundle)


def test_development_grader_does_not_protect_its_result_channel():
    # Harmless controlled counterexample: no solutions, no network, no reads outside child argv.
    # Keep the reason for the confirmatory prohibition executable; do not call this a secure judge.
    source = ("import atexit, json, sys\n"
              "atexit.register(lambda: print(json.dumps([r['id'] for r in json.loads(sys.argv[1])])) )\n")
    task = CD.Task("missing_function", "Return seven", "def missing_function(): ...",
                   ("assert missing_function() == 7",),
                   ("assert missing_function() == 7", "assert callable(missing_function)"))
    grader = CD.subprocess_batch_runner()
    assert grader(source, [task]) == {task.id}
    assert grader.development_only is True
    assert not getattr(grader, "exact_check", False)


@pytest.mark.parametrize("factory", [CD.subprocess_batch_runner, CD.inprocess_batch_runner])
def test_shared_interpreter_graders_cannot_be_promoted_by_exact_check_marker(factory):
    grader = custody.attest_exact_check(factory())
    lad = CD.SuiteLadder(CD.TaskPool([CD.Task("add_0", "Add", "def add_0(x,y): ...",
                        ("assert add_0(1,2)==3",), ("assert add_0(1,2)==3", "assert add_0(4,5)==9"))],
                                    name="test-only"), batch_runner=functools.partial(grader), subset_size=1)
    assert "batch_runner" in custody.development_verifiers(lad)
    assert any("development-only" in f for f in mode._ladder_refusals(mode.ConfirmatoryInputs(ladder=lad)))


def test_calibration_requires_frozen_complete_procedure_with_error_uncertainty():
    cfg = p16.P16Config()
    assert CG.refusals(cfg)
    with_fixture_calibration(cfg)
    assert CG.refusals(cfg) == []  # invented fixture counts, never experimental evidence
    cfg.horizon += 1
    assert any("configuration" in f for f in CG.refusals(cfg))
    with_fixture_calibration(cfg)
    cfg.alarm_rate_null_calibration["code_sha256"] = "0" * 64
    assert any("implementation" in f for f in CG.refusals(cfg))
    with_fixture_calibration(cfg)
    cfg.alarm_rate_null_calibration["worlds"][0]["independent_runs"] = 10
    assert any("upper bound" in f for f in CG.refusals(cfg))  # zero errors is not a zero error rate
    with_fixture_calibration(cfg)
    cfg.alarm_rate_null_calibration["worlds"].pop()
    assert any("missing required worlds" in f for f in CG.refusals(cfg))
    with_fixture_calibration(cfg)
    cfg.alarm_rate_null_calibration["endpoint"] = "run_pattern"
    assert any("final P16 wrapper" in f for f in CG.refusals(cfg))
