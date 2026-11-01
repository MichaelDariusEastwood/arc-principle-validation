"""One synthetic end-to-end case (41-D-COUNTED, section 12, package H; section 11.3, step 2): a stand-in client whose
replies follow a stated world, through the collector, the blind scoring, both looks and the report, and the
look-failure path of section 7.4. It builds a synthetic workspace that no registration names, at small sizes set in a
test copy of the configuration, and so calls the steps' inner functions, never their gated wrappers. Not run before
both registrations read REGISTERED."""
import hashlib
import json

from p21base.common import file_sha, save
from p21lane import analysis, collect, forms, generate, lane, schedule, seal, sentences

from support import WorldClient, load_config, temp_root

ENGINES = ['claude-haiku-5-5', 'gpt-6-luna']
TIER = 1
OSF_ID, WHEN = 'abcde', '2026-11-03T05:00:00.000000'
MARKS = (analysis.SUPPORTED, analysis.REFUTED, analysis.INCONCLUSIVE, analysis.NOT_EVALUABLE, analysis.CONTINUE)


def small_config():
    config = load_config()
    config['looks'] = {'sizes': [6, 12], 'levels': [0.995, 0.955], 'gates': [0.00206, 0.00264],
                       'sign_flip_alpha': [0.0025, 0.0225]}
    config['positive_control_pairs'] = 6
    config['secondary_pairs'] = 3
    config['bootstrap'] = {'resamples': 200, 'seed': 'test-bootstrap'}
    config['sign_flip'] = {'flips': 2000, 'seed': 'test-sign-flip'}
    return config


def synthetic_workspace(config):
    """Banks, schedules and sealed answers for two engines, written as init writes them, with a commitment whose
    derivation the analysis can check; no registry is read."""
    root = temp_root('p21-end-to-end-')
    seed = seal.derive(config['registered_seed'], OSF_ID, WHEN)
    required_by_text, answers = {}, {}
    for engine in ENGINES:
        bank = generate.make_bank(seed, TIER, engine, config['looks']['sizes'][-1])
        plan = schedule.make(engine, len(bank['pairs']), bank, seed)
        save(seal.bank_path(root, TIER, engine), generate.bank_to_json(bank))
        save(seal.schedule_path(root, TIER, engine), plan)
        for pair in bank['pairs']:
            for case in pair['cases']:
                line = forms.line(forms.required(case))
                answers[seal.case_key(case['case_id'], seed)] = line
                required_by_text[forms.case_text(case)] = line
    save(seal.answers_path(root, TIER), answers)
    save(root / 'public' / 'public_sets.json', {})
    save(root / 'public' / 'answers.json', {})
    files = {path.relative_to(root).as_posix(): file_sha(path)
             for folder in ('bank', 'schedule', 'answers') for path in sorted((root / folder).glob('*.json'))}
    save(root / seal.COMMITMENT, {
        'study_id': config['study_id'], 'files': files,
        'derivation': {'registered_seed': config['registered_seed'], 'osf_id': OSF_ID, 'date_registered': WHEN,
                       'counter': 0, 'bank_seed_sha256': hashlib.sha256(seed).hexdigest()},
        'public_sets_sha256': file_sha(root / 'public' / 'public_sets.json'),
        'public_answers_sha256': file_sha(root / 'public' / 'answers.json')})
    return root, required_by_text


def context(root, config, client):
    return collect.Context(root, config, client, tier=TIER, in_flight={e: 3 for e in ENGINES},
                           generating={e: True for e in ENGINES})


def test_both_looks_and_the_report():
    config = small_config()
    root, required = synthetic_workspace(config)
    client = WorldClient(required, ENGINES)
    ctx = context(root, config, client)
    status = collect.stage(ctx, 1, 6)
    assert all(row['completed_prefix'] == 6 for row in status['engines'].values())
    log = collect.pair_logs(root, 'gpt-6-luna', 6)[0]
    assert log['arms'] == ['C', 'M', 'T', 'W', 'G', 'I'] and len(log['blocks']) == 2
    assert all('text' not in record for rnd in log['blocks'][0]['rounds'] for record in rnd['records'].values())
    printed, state = [], lane.read_state(root)
    decision, sha = lane.run_first_look(root, config, state, TIER, ENGINES, show=printed.append)
    assert decision in ('stop', 'continue') and len(printed) == 1
    assert printed[0] == 'FIRST LOOK: %s; sealed record SHA-256 %s.' % (decision, sha)
    sealed = root / 'looks' / 'look1.json'
    assert hashlib.sha256(sealed.read_bytes()).hexdigest() == sha
    first = json.loads(sealed.read_text('utf-8'))
    assert first['mark'] in MARKS and first['n_pairs'] == 6
    assert set(first['confirmation']) == set(analysis.CONDITIONS)
    # Collection runs to the second look's size whatever the first look said, so that both looks are exercised.
    collect.stage(context(root, config, client), 7, 12)
    state['decision'] = 'continue'
    record = lane.run_second_look(root, config, state, TIER, ENGINES)
    assert record['look'] == 2 and record['n_pairs'] == 12 and record['mark'] in MARKS[:4]
    path = sentences.report(root, config)
    text = path.read_text('utf-8')
    assert 'Domain: counted, exact comparison on a frozen, sealed suite; no judged term.' in text
    assert 'Depth is the count of administered arrivals' in text


def test_the_look_failure_path():
    config = small_config()
    root, required = synthetic_workspace(config)
    client = WorldClient(required, ENGINES)
    collect.stage(context(root, config, client), 1, 6)
    state, printed = lane.read_state(root), []
    original = analysis.look

    def failing(*args, **kwargs):
        raise RuntimeError('a stand-in fault of the analysis')

    analysis.look = failing
    try:
        decision, sha = lane.run_first_look(root, config, state, TIER, ENGINES, show=printed.append)
    finally:
        analysis.look = original
    assert decision == 'continue' and sha is None and state['look1_failed']
    failure = json.loads((root / 'looks' / 'look1_failure.json').read_text('utf-8'))
    trace = (root / 'looks' / failure['trace_file']).read_bytes()
    assert failure['error_trace_sha256'] == hashlib.sha256(trace).hexdigest() and failure['look'] == 1
    assert failure['error_trace_sha256'] in printed[0] and not (root / 'looks' / 'look1.json').exists()
    collect.stage(context(root, config, client), 7, 12)
    record = lane.run_second_look(root, config, state, TIER, ENGINES)
    assert record['look'] == 2
    assert (root / 'looks' / 'look1_record.json').exists()          # computed for the record; it stops nothing


def test_a_replay_rebuilds_the_registers_exactly():
    config = small_config()
    root, required = synthetic_workspace(config)
    client = WorldClient(required, ENGINES)
    collect.stage(context(root, config, client), 1, 3)
    before = collect.pair_logs(root, 'claude-haiku-5-5', 3)
    calls = client.calls
    collect.stage(context(root, config, client), 1, 3)              # every pair complete: nothing is asked again
    assert client.calls == calls
    assert collect.pair_logs(root, 'claude-haiku-5-5', 3) == before
