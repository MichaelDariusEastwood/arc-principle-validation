"""How the bank is sealed and committed, with the answers off the registry (41-D-COUNTED, section 4.9).

Package D of the build list (DESIGN-FREEZE.md, section 12). At `init`, after both registrations read REGISTERED, the
bank seed is the SHA-256 of three things joined by a vertical bar: the registered seed; this unit's registration
identifier on OSF (the five characters of its address); and the registration record's `date_registered` value exactly
as the registry's public interface serves it. `init` reads both values from that interface, checks that the record is
public and not pending approval, refuses to proceed if either value differs from the author's attestation, and stores
the SHA-256 of the registry's response in the commitment. From that seed it makes the three tiers' banks and schedules
for 660 pairs an engine, checks every property of section 4.6, and writes `bank_commitment.json`: the derivation's
inputs and the SHA-256 of every bank, schedule and answer file, with the public sets' hashes (not the sets). A failure
of property 2 or 3 moves to the next bank of the registered sequence (the derivation with a counter appended), at most
four times, every attempt recorded; any other failure is an implementation defect. Nothing here is executed before both
registrations read REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of this file's text.
"""
import hashlib
import json
import os
from pathlib import Path

from p21base import registration_gate as gate
from p21base.common import GateError, file_sha, now, save

from . import forms, generate, schedule

MAX_SEQUENCE = 4
COMMITMENT = 'bank_commitment.json'


def derive(registered_seed, osf_id, date_registered, counter=0):
    """The bank seed: SHA-256 of the registered seed, the identifier and date_registered joined by vertical bars, the
    counter appended only when it is above zero."""
    for value in (registered_seed, osf_id, date_registered):
        if not isinstance(value, str) or not value:
            raise GateError('every input of the derivation is non-empty text')
    if not isinstance(counter, int) or isinstance(counter, bool) or counter < 0:
        raise GateError('the counter is a whole number')
    parts = [registered_seed, osf_id, date_registered] + ([str(counter)] if counter > 0 else [])
    return hashlib.sha256('|'.join(parts).encode('utf-8')).digest()


def public_seed_bytes(public_seed):
    return hashlib.sha256(public_seed.encode('utf-8')).digest()


def read_registry(osf_id):
    """The registration's public record, read without a key, and the SHA-256 of the registry's response."""
    record = gate.registration_record(osf_id)
    return record, record['response_sha256']


def case_key(case_id, seed_bytes):
    """The case key the scorer sees (section 6.4): SHA-256 of the case identifier and the seed. It carries no label."""
    return hashlib.sha256((case_id + '|' + bytes(seed_bytes).hex()).encode('utf-8')).hexdigest()


def entry_key(case_key_value, line):
    """The key of one scoring row: SHA-256 of the case key and the entry line."""
    return hashlib.sha256((case_key_value + '|' + line).encode('utf-8')).hexdigest()


def bank_path(root, tier, engine):
    return Path(root) / 'bank' / ('tier%d-%s.json' % (tier, engine))


def schedule_path(root, tier, engine):
    return Path(root) / 'schedule' / ('tier%d-%s.json' % (tier, engine))


def answers_path(root, tier):
    return Path(root) / 'answers' / ('tier%d.json' % tier)


def engines_of(config):
    return [config['engines'][key]['model'] for key in config['engine_order']]


def make_all(seed, config, oracle):
    """Every bank and schedule for one seed, each checked; raises PropertyFailure on the first failure."""
    n_pairs = config['looks']['sizes'][-1]
    banks, schedules = {}, {}
    for tier in generate.TIERS:
        for engine in engines_of(config):
            bank = generate.make_bank(seed, tier, engine, n_pairs)
            plan = schedule.make(engine, n_pairs, bank, seed)
            generate.check(bank, plan, oracle)
            banks[(tier, engine)] = bank
            schedules[(tier, engine)] = plan
    return banks, schedules


def init(root, config):
    """Step 3 of section 11.3. It cannot be repeated in the same workspace."""
    root = Path(root)
    if (root / COMMITMENT).exists() or (root / 'bank').exists():
        raise GateError('init cannot be repeated in the same workspace; after a logged repair it runs again in a fresh one')
    attestation = gate.read_attestation(root)
    record, response_sha = read_registry(attestation['registration_id'])
    problems = gate.registration_problems(record)
    if problems:
        raise GateError('the registration does not read REGISTERED: ' + ', '.join(problems))
    if record['identifier'] != attestation['registration_id'] or record['date_registered'] != attestation['date_registered']:
        raise GateError('the registry\'s identifier or date_registered differs from the author\'s attestation; init refuses to proceed')
    # Conditional on review D's adoption of its D1-01 and D1-08 (section 11.3): where the configuration names the v7
    # suite's registration, init reads it as a third record by the same test, and it must match the attestation.
    if config.get('suite_registration'):
        suite = gate.registration_record(config['suite_registration'])
        suite_problems = gate.registration_problems(suite)
        if suite_problems:
            raise GateError('the suite\'s registration does not read REGISTERED: ' + ', '.join(suite_problems))
        if suite['identifier'] != attestation.get('suite_registration_id') or suite['date_registered'] != attestation.get('suite_date_registered'):
            raise GateError('the suite\'s registration differs from the author\'s attestation; init refuses to proceed')
    theory = gate.registration_record(config['theory_registration'])
    theory_problems = gate.registration_problems(theory)
    if theory_problems:
        raise GateError('1.103 does not read REGISTERED: ' + ', '.join(theory_problems))
    oracle = generate._load_oracle()
    attempts, chosen = [], None
    for counter in range(MAX_SEQUENCE + 1):
        seed = derive(config['registered_seed'], record['identifier'], record['date_registered'], counter)
        try:
            banks, schedules = make_all(seed, config, oracle)
        except generate.PropertyFailure as failure:
            if failure.number not in (2, 3):
                raise GateError('implementation defect, ' + str(failure) + '; a logged repair, then init again in a fresh workspace')
            attempts.append({'counter': counter, 'property': failure.number, 'message': str(failure), 'utc': now()})
            continue
        chosen = (counter, seed, banks, schedules)
        break
    if chosen is None:
        save(root / 'init_failures.json', {'attempts': attempts, 'utc': now()}, exclusive=True)
        raise GateError('a fifth failure of a draw-dependent property: the generator\'s ranges need a registration update')
    counter, seed, banks, schedules = chosen
    sets = generate.public_sets(config['public_seed'])
    try:
        generate.check_public(sets, oracle)
    except generate.PropertyFailure as failure:
        raise GateError('implementation defect in the public sets, ' + str(failure))
    files = {}
    answers = {tier: {} for tier in generate.TIERS}
    for (tier, engine), bank in banks.items():
        save(bank_path(root, tier, engine), generate.bank_to_json(bank), exclusive=True)
        save(schedule_path(root, tier, engine), schedules[(tier, engine)], exclusive=True)
        for pair in bank['pairs']:
            for case in pair['cases']:
                answers[tier][case_key(case['case_id'], seed)] = forms.line(forms.required(case))
    for tier in generate.TIERS:
        save(answers_path(root, tier), answers[tier], exclusive=True)
    public_seed = public_seed_bytes(config['public_seed'])
    public_answers = {case_key(item['case']['case_id'], public_seed): forms.line(forms.required(item['case']))
                      for item in generate.public_items(sets)}
    save(root / 'public' / 'public_sets.json', generate.public_to_json(sets), exclusive=True)
    save(root / 'public' / 'answers.json', public_answers, exclusive=True)
    for folder in ('bank', 'schedule', 'answers'):
        for path in sorted((root / folder).glob('*.json')):
            files[path.relative_to(root).as_posix()] = file_sha(path)
    commitment = {
        'study_id': config['study_id'], 'generator_version': generate.GENERATOR_VERSION,
        'schedule_version': schedule.SCHEDULE_VERSION,
        'derivation': {'registered_seed': config['registered_seed'], 'osf_id': record['identifier'],
                       'date_registered': record['date_registered'], 'counter': counter,
                       'bank_seed_sha256': hashlib.sha256(seed).hexdigest(),
                       'registry_response_sha256': response_sha},
        'attempts': attempts, 'n_pairs': config['looks']['sizes'][-1], 'engines': engines_of(config),
        'tiers': list(generate.TIERS), 'files': files,
        'public_sets_sha256': file_sha(root / 'public' / 'public_sets.json'),
        'public_answers_sha256': file_sha(root / 'public' / 'answers.json'),
        'created_utc': now(),
    }
    save(root / COMMITMENT, commitment, exclusive=True)
    os.chmod(root / COMMITMENT, 0o444)
    return commitment


def bank_seed(root, config):
    """The bank seed, recomputed from the commitment's derivation and checked against its hash."""
    commitment = json.loads((Path(root) / COMMITMENT).read_text('utf-8'))
    d = commitment['derivation']
    seed = derive(config['registered_seed'], d['osf_id'], d['date_registered'], d['counter'])
    if hashlib.sha256(seed).hexdigest() != d['bank_seed_sha256']:
        raise GateError('the bank seed does not match its commitment')
    return seed


def verify(root):
    """Every committed file still holds its committed bytes."""
    commitment = json.loads((Path(root) / COMMITMENT).read_text('utf-8'))
    for rel, expected in commitment['files'].items():
        if file_sha(Path(root) / rel) != expected:
            raise GateError('a sealed file changed: ' + rel)
    for rel, key in (('public/public_sets.json', 'public_sets_sha256'), ('public/answers.json', 'public_answers_sha256')):
        if file_sha(Path(root) / rel) != commitment[key]:
            raise GateError('a sealed public file changed: ' + rel)
    return commitment


def load_bank(root, tier, engine):
    return generate.bank_from_json(json.loads(bank_path(root, tier, engine).read_text('utf-8')))


def load_schedule(root, tier, engine):
    return json.loads(schedule_path(root, tier, engine).read_text('utf-8'))


def load_public(root):
    return generate.public_from_json(json.loads((Path(root) / 'public' / 'public_sets.json').read_text('utf-8')))
