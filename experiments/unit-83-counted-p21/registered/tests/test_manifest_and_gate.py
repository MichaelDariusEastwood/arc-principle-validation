"""Authored cases for the closed build (41-D-COUNTED, section 12, packages A and H): the manifest made by command lists
every file of the package with its bytes' hash, the repair chains of the registration gate, and the lane's order and
fixed recovery causes. Not run before both registrations read REGISTERED."""
import importlib.util
import json

from p21base import registration_gate as gate
from p21lane import lane

from support import STAGE


def _manifest_tool():
    spec = importlib.util.spec_from_file_location('make_manifest', STAGE / 'tools' / 'make_manifest.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_manifest_lists_every_file_of_the_package():
    tool = _manifest_tool()
    manifest = json.loads((STAGE / gate.MANIFEST).read_text('utf-8'))
    assert manifest['study_id'] == gate.STUDY_ID
    listed = set(tool.listed_files(STAGE))
    assert set(manifest['files']) == listed
    for rel in listed:
        assert manifest['files'][rel] == tool.sha256(STAGE / rel), rel
    for package in gate.PACKAGES:
        for path in (STAGE / package).rglob('*.py'):
            assert path.relative_to(STAGE).as_posix() in listed
    assert gate.CONFIG_REL in listed and 'p21lane/lane.py' in listed


def test_repair_chains():
    registered = {'p21lane/analysis.py': 'a' * 64, 'configs/p21.json': 'c' * 64}
    entries = [{'line': 1, 'file': 'p21lane/analysis.py', 'replaced_sha256': 'a' * 64, 'new_sha256': 'b' * 64},
               {'line': 2, 'file': 'p21lane/analysis.py', 'replaced_sha256': 'b' * 64, 'new_sha256': 'd' * 64}]
    assert gate.repair_chains(entries, registered, {'configs/p21.json'}) == {'p21lane/analysis.py': 'd' * 64}
    for bad in ([{'line': 1, 'file': 'configs/p21.json', 'replaced_sha256': 'c' * 64, 'new_sha256': 'e' * 64}],
                [{'line': 1, 'file': 'p21lane/analysis.py', 'replaced_sha256': 'f' * 64, 'new_sha256': 'e' * 64}],
                [{'line': 1, 'file': 'p21lane/other.py', 'replaced_sha256': 'a' * 64, 'new_sha256': 'e' * 64}]):
        try:
            gate.repair_chains(bad, registered, {'configs/p21.json'})
        except RuntimeError:
            pass
        else:
            raise AssertionError('a repair outside its chain was accepted')


def test_a_registration_record_reads_registered_only_when_public_and_approved():
    good = {'public': True, 'date_registered': '2026-11-03T05:00:00.000000'}
    assert gate.registration_problems(good) == []
    assert gate.registration_problems(dict(good, pending_registration_approval=True)) == ['pending registration approval']
    assert 'not public' in gate.registration_problems(dict(good, public=False))
    assert 'no date_registered' in gate.registration_problems(dict(good, date_registered=None))


def test_the_steps_run_in_the_registered_order():
    assert lane.STEPS == ('attest', 'init', 'admit', 'freeze', 'collect_stage_one', 'look', 'collect_stage_two', 'report')
    state = {'done': {'attest': {}, 'init': {}}}
    lane.require_order(state, 'admit')
    try:
        lane.require_order(state, 'freeze')
    except lane.LaneStop as stop:
        assert stop.step == 'freeze' and 'admit' in str(stop)
    else:
        raise AssertionError('a step ran out of order')


def test_the_recovery_causes_are_a_fixed_list():
    actions = {action for _, action, _ in lane.RECOVERY_CAUSES}
    assert actions == {'reissue', 'abandon_study'}
    assert len({cause for cause, _, _ in lane.RECOVERY_CAUSES}) == len(lane.RECOVERY_CAUSES)
