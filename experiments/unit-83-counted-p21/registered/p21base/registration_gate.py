"""Registration-first admission for 41-D-COUNTED, using standard-library checks only.

Carried from the compact Screen's version 0.3 (`registration_gate.py`, 239 lines, sha256
0cdf701e8170453891c5060dbb9a39cfd10695695b6cda3ed4b89f1afb1eeba4): the repair and attestation logic of its lines 63
to 217 is kept, and the rest is changed as the design freeze's package A says (DESIGN-FREEZE.md, sections 11.2, 11.3
and 12). Nothing here is imported or executed before both registrations read REGISTERED (SD-001, SD-012); before then
the only mechanical check is a parse of this file's text.

What changed. The attestation is written by a command that reads both registrations, 1.103 and this unit's, from the
registry's public interface (identifier, `date_registered`, public, not pending approval), shows the values and asks for
the author's confirmation, so that no value of the bank's derivation is typed (the operations attack, O-10 and O-16).
The constants name this study. The file walk refuses a Python file of the lane's packages that the manifest does not
list. The admission fields (the engines admitted, the generating arm admitted on each engine, the tier) live in the
admission receipts and never change the registered configuration.

Defect repairs. After registration and before the runtime freeze, an implementation defect may be repaired if the
repair changes no registered choice. Each repair is one line of repairs.jsonl in the workspace, with eight fields: seq,
utc, file, replaced_sha256, new_sha256, failing_case, unchanged_statement and author_approved. A registered file passes
when its bytes equal its registered hash, or the hash on which its chain of logged repairs ends. The manifest and the
registered configuration are never repairable. Once the runtime freeze exists, a repair dated after it refuses every
command, with one exception, the look-failure rule of section 7.4: a repair of the analysis made from the error trace
alone, one line of look_repairs.jsonl naming the failed look, the hash of its error trace and whether the public sets
show the defect, and only for the four analysis files.
"""
from __future__ import annotations
import hashlib
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

STUDY_ID = '41-D-COUNTED-v0.1'
WORKSPACE_ENV = 'P21LANE_WORKSPACE'
MANIFEST = 'registration_manifest.json'
ATTESTATION = 'registration_approval.json'
CONFIG_REL = 'configs/p21.json'
PACKAGES = ('p21base', 'p21lane')
REGISTRY = 'https://api.osf.io/v2/registrations/'
REGISTRY_USER_AGENT = 'p21-lane-registry-reader'
REPAIR_LOG = 'repairs.jsonl'
REPAIR_FIELDS = frozenset(('seq', 'utc', 'file', 'replaced_sha256', 'new_sha256',
                           'failing_case', 'unchanged_statement', 'author_approved'))
LOOK_REPAIR_LOG = 'look_repairs.jsonl'
LOOK_REPAIR_FIELDS = frozenset(('seq', 'utc', 'file', 'replaced_sha256', 'new_sha256', 'error_trace_sha256', 'look',
                                'public_sets_show_defect', 'unchanged_statement', 'author_approved'))
LOOK_REPAIRABLE = ('p21lane/score.py', 'p21lane/analysis.py', 'p21lane/stats.py', 'p21lane/sentences.py')
ATTESTATION_FIELDS = frozenset(('study_id', 'registration_id', 'registration_url', 'date_registered',
                                'registration_response_sha256', 'registered_from', 'theory_registration_id',
                                'theory_date_registered', 'theory_response_sha256', 'suite_registration_id',
                                'suite_date_registered', 'suite_run_start_sha256', 'manifest_sha256',
                                'manifest_found_in_registration', 'read_utc', 'confirmed_by_author', 'confirmed_utc'))


def stage_root() -> Path:
    """The stage the executing packages sit in: the directory that holds p21base and p21lane."""
    return Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(text: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RuntimeError('Duplicate metadata field: ' + key)
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(RuntimeError('Non-finite metadata')))


def _read(path: Path):
    return _parse(path.read_text('utf-8'))


def _moment(value) -> datetime:
    """A time written in ISO form with its zone. Anything else raises ValueError."""
    if not isinstance(value, str):
        raise ValueError('a time must be text')
    when = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if when.tzinfo is None:
        raise ValueError('a time must state its zone')
    return when


def _read_log(path: Path, fields: frozenset, label: str) -> list[dict]:
    """Every line that is not blank must be one complete entry with exactly the registered fields, seq counting 1, 2,
    3 in the order written, a time that states its zone and is not in the future, two different lower-case SHA-256
    hashes and author_approved set to true. Anything else refuses and names the line. A log that is absent is empty."""
    if not path.exists():
        return []
    try:
        text = path.read_text('utf-8')
    except (OSError, ValueError):
        raise RuntimeError('The ' + label + ' cannot be read as UTF-8 text')
    entries = []
    for number, line in enumerate(text.split('\n'), 1):
        if not line.strip():
            continue
        place = label[0].upper() + label[1:] + ' line ' + str(number)
        try:
            entry = _parse(line)
        except (ValueError, RuntimeError):
            entry = None
        if not isinstance(entry, dict) or set(entry) != fields:
            raise RuntimeError(place + ' is not one JSON object with exactly the registered fields')
        seq = entry['seq']
        if not isinstance(seq, int) or isinstance(seq, bool) or seq != len(entries) + 1:
            raise RuntimeError(place + ' is out of sequence: entries are numbered 1, 2, 3 in the order they are written')
        try:
            when = _moment(entry['utc'])
        except ValueError:
            raise RuntimeError(place + ' needs a time that states its zone')
        if when > datetime.now(timezone.utc):
            raise RuntimeError(place + ' is dated in the future')
        for name in ('file', 'unchanged_statement'):
            if not isinstance(entry[name], str) or not entry[name].strip():
                raise RuntimeError(place + ' needs text in its field ' + name)
        hashes = ['replaced_sha256', 'new_sha256'] + (['error_trace_sha256'] if 'error_trace_sha256' in fields else [])
        for name in hashes:
            if not isinstance(entry[name], str) or not re.fullmatch('[0-9a-f]{64}', entry[name]):
                raise RuntimeError(place + ' needs a SHA-256 hash in lower-case hexadecimal in its field ' + name)
        if entry['replaced_sha256'] == entry['new_sha256']:
            raise RuntimeError(place + ' records no change: its two hashes are equal')
        if entry['author_approved'] is not True:
            raise RuntimeError(place + ' is not approved by the author')
        entries.append({**entry, 'line': number})
    return entries


def read_repair_log(path: str | Path) -> list[dict]:
    """The defect-repair log as a list of entries in the order written, each with its line number."""
    entries = _read_log(Path(path), REPAIR_FIELDS, 'repair log')
    for entry in entries:
        if not isinstance(entry['failing_case'], str) or not entry['failing_case'].strip():
            raise RuntimeError('Repair log line ' + str(entry['line']) + ' needs text in its field failing_case')
    return entries


def read_look_repair_log(path: str | Path) -> list[dict]:
    """The look-failure repair log of section 7.4, read like the repair log."""
    entries = _read_log(Path(path), LOOK_REPAIR_FIELDS, 'look repair log')
    for entry in entries:
        if entry['look'] not in (1, 2) or isinstance(entry['look'], bool):
            raise RuntimeError('Look repair log line ' + str(entry['line']) + ' names no registered look')
        if not isinstance(entry['public_sets_show_defect'], bool):
            raise RuntimeError('Look repair log line ' + str(entry['line']) + ' must say whether the public sets show the defect')
    return entries


def repair_chains(entries: list[dict], registered: dict, fixed: set, start: dict | None = None) -> dict[str, str]:
    """For each repaired file, the hash on which its chain of logged repairs ends.

    A file's first entry must replace its registered hash (or, for a later log, the hash the earlier log ended on),
    and each later entry must replace the hash that the entry before it produced. An entry refuses when it names a file
    the registration does not list, or one that is never repairable, or when it does not continue its file's chain.
    """
    last = dict(start or {})
    for entry in entries:
        rel = entry['file']
        place = 'Repair entry at line ' + str(entry['line'])
        if rel not in registered:
            raise RuntimeError(place + ' names a file the registration does not list: ' + rel)
        if rel in fixed:
            raise RuntimeError(place + ' names a file that is never repairable: ' + rel)
        if entry['replaced_sha256'] != last.get(rel, registered[rel]):
            raise RuntimeError(place + ' does not continue the repair chain of ' + rel)
        last[rel] = entry['new_sha256']
    return last


def freeze_time(freeze_path: Path) -> datetime | None:
    if not freeze_path.exists():
        return None
    try:
        return _moment(_read(freeze_path)['created_utc'])
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        raise RuntimeError('The runtime freeze record has no readable time; repairs cannot be dated against it')


def refuse_repairs_after_freeze(entries: list[dict], freeze_path: str | Path) -> None:
    """Once the runtime freeze exists, a defect repair dated after it refuses. The freeze records its time to the
    second, so a repair's time is compared to the whole second."""
    frozen = freeze_time(Path(freeze_path))
    if not entries or frozen is None:
        return
    for entry in entries:
        if _moment(entry['utc']).replace(microsecond=0) > frozen:
            raise RuntimeError('Repair log line ' + str(entry['line'])
                               + ' is dated after the runtime freeze; nothing changes after the freeze')


def check_look_repairs(entries: list[dict], workspace: Path) -> None:
    """The exception of section 7.4: each look repair names one of the four analysis files, is dated after the
    runtime freeze, and answers a recorded look failure whose error trace has the hash it names."""
    frozen = freeze_time(workspace / 'freeze.json')
    for entry in entries:
        place = 'Look repair log line ' + str(entry['line'])
        if entry['file'] not in LOOK_REPAIRABLE:
            raise RuntimeError(place + ' names a file the look-failure rule does not reach: ' + entry['file'])
        if frozen is None or _moment(entry['utc']).replace(microsecond=0) <= frozen:
            raise RuntimeError(place + ' is not dated after the runtime freeze; before it, repairs go to the repair log')
        failure = workspace / 'looks' / ('look%d_failure.json' % entry['look'])
        if not failure.is_file():
            raise RuntimeError(place + ' answers no recorded failure of look ' + str(entry['look']))
        record = _read(failure)
        if record.get('error_trace_sha256') != entry['error_trace_sha256']:
            raise RuntimeError(place + ' names another error trace than the recorded look failure')
        if _moment(record.get('utc')).replace(microsecond=0) > _moment(entry['utc']):
            raise RuntimeError(place + ' is dated before the look failure it answers')


def _never_repairable(rel: str) -> bool:
    """The manifest itself and every registered configuration file hold registered choices only."""
    return rel == MANIFEST or rel.startswith('configs/')


def registry_get(url: str, timeout: float = 60.0) -> tuple[dict, str]:
    """GET one public record of the registry's interface, read without a key and with a neutral client name; returns
    the parsed record and the SHA-256 of the response bytes."""
    request = urllib.request.Request(url, headers={'Accept': 'application/vnd.api+json',
                                                   'User-Agent': REGISTRY_USER_AGENT}, method='GET')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(5_000_001)
    if len(raw) > 5_000_000:
        raise RuntimeError('the registry response exceeds its byte envelope')
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def registration_record(identifier: str) -> dict:
    """The fields of a registration record that the attestation, `init` and the run read (section 4.9)."""
    if not isinstance(identifier, str) or not re.fullmatch('[a-z0-9]{5}', identifier):
        raise RuntimeError('a registration identifier is the five characters of its OSF address')
    body, sha = registry_get(REGISTRY + identifier + '/')
    data = body.get('data') if isinstance(body, dict) else None
    if not isinstance(data, dict) or data.get('id') != identifier:
        raise RuntimeError('the registry returned another record than ' + identifier)
    attributes = data.get('attributes') if isinstance(data.get('attributes'), dict) else {}
    relationships = data.get('relationships') if isinstance(data.get('relationships'), dict) else {}
    registered_from = None
    origin = relationships.get('registered_from')
    if isinstance(origin, dict):
        link = origin.get('data')
        if isinstance(link, dict) and isinstance(link.get('id'), str):
            registered_from = link['id']
        else:
            href = ((origin.get('links') or {}).get('related') or {}).get('href') if isinstance(origin.get('links'), dict) else None
            found = re.search(r'/nodes/([a-z0-9]{5})/?$', href or '')
            registered_from = found.group(1) if found else None
    return {'identifier': identifier, 'date_registered': attributes.get('date_registered'),
            'public': attributes.get('public'), 'pending_registration_approval': attributes.get('pending_registration_approval'),
            'pending_embargo_approval': attributes.get('pending_embargo_approval'),
            'pending_withdrawal': attributes.get('pending_withdrawal'), 'withdrawn': attributes.get('withdrawn'),
            'archiving': attributes.get('archiving'), 'title': attributes.get('title'),
            'registered_from': registered_from, 'response_sha256': sha}


def registration_problems(record: dict) -> list[str]:
    """Why a registration record does not read REGISTERED: public and approved, not withdrawn, not archiving."""
    problems = []
    if record.get('public') is not True:
        problems.append('not public')
    for name in ('pending_registration_approval', 'pending_embargo_approval', 'pending_withdrawal', 'withdrawn', 'archiving'):
        if record.get(name) is True:
            problems.append(name.replace('_', ' '))
    if not isinstance(record.get('date_registered'), str) or not record['date_registered'].strip():
        problems.append('no date_registered')
    return problems


def registration_file_hashes(identifier: str, limit: int = 2000) -> set[str]:
    """The SHA-256 of every file the registration's own storage holds, read from the registry without a key."""
    found: set[str] = set()
    queue = [REGISTRY + identifier + '/files/osfstorage/']
    seen = 0
    while queue and seen < limit:
        url = queue.pop(0)
        body, _ = registry_get(url)
        for item in body.get('data') or []:
            seen += 1
            attributes = item.get('attributes') or {}
            if attributes.get('kind') == 'folder':
                related = (((item.get('relationships') or {}).get('files') or {}).get('links') or {}).get('related') or {}
                href = related.get('href') if isinstance(related, dict) else related
                if isinstance(href, str):
                    queue.append(href)
            else:
                hashes = ((attributes.get('extra') or {}).get('hashes') or {})
                if isinstance(hashes.get('sha256'), str):
                    found.add(hashes['sha256'].lower())
        following = (body.get('links') or {}).get('next')
        if isinstance(following, str):
            queue.append(following)
    return found


def attest(workspace: str | Path, registration_id: str, config: dict, *, confirm=input, show=print) -> dict:
    """Step 1 of section 11.3: read both registrations from the registry's public interface, show what was read and
    ask the author to confirm it; then write the attestation. No value of the derivation is typed: the unit's
    identifier given here is checked against the record it names (its source component and its storage, which must
    hold this manifest), and 1.103's identifier is the configuration's."""
    workspace = Path(workspace).resolve()
    path = workspace / ATTESTATION
    if path.exists():
        raise RuntimeError('The attestation exists; it is written once and never replaced')
    manifest_path = stage_root() / MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError('The stage holds no registration manifest')
    manifest_sha = _sha(manifest_path)
    unit = registration_record(registration_id)
    theory = registration_record(config['theory_registration'])
    problems = [('this unit', p) for p in registration_problems(unit)] + [('1.103', p) for p in registration_problems(theory)]
    component = config.get('unit_component')
    if not component:
        problems.append(('configuration', 'the unit component is not named'))
    elif unit['registered_from'] != component:
        problems.append(('this unit', 'registered from ' + str(unit['registered_from']) + ', not from the unit component ' + component))
    suite = {'suite_registration_id': None, 'suite_date_registered': None, 'suite_run_start_sha256': None}
    if config.get('suite_registration'):
        record = registration_record(config['suite_registration'])
        problems += [('the suite', p) for p in registration_problems(record)]
        start = workspace / 'suite_run_start.json'
        if not start.is_file():
            problems.append(('the suite', 'no run-start record in the workspace'))
        else:
            lanes = _read(start).get('lanes')
            if not isinstance(lanes, list) or STUDY_ID not in lanes:
                problems.append(('the suite', 'the run-start record does not name this lane'))
            suite = {'suite_registration_id': record['identifier'], 'suite_date_registered': record['date_registered'],
                     'suite_run_start_sha256': _sha(start)}
    try:
        held = registration_file_hashes(registration_id)
    except Exception as exc:
        held = set()
        problems.append(('this unit', 'its files could not be read: ' + type(exc).__name__))
    if manifest_sha not in held:
        problems.append(('this unit', 'its files do not hold this stage\'s registration manifest ' + manifest_sha))
    if problems:
        raise RuntimeError('STOP: the attestation cannot be written: ' + '; '.join(who + ': ' + why for who, why in problems))
    read_utc = datetime.now(timezone.utc).isoformat(timespec='seconds')
    record = {'study_id': STUDY_ID, 'registration_id': unit['identifier'],
              'registration_url': 'https://osf.io/' + unit['identifier'] + '/', 'date_registered': unit['date_registered'],
              'registration_response_sha256': unit['response_sha256'], 'registered_from': unit['registered_from'],
              'theory_registration_id': theory['identifier'], 'theory_date_registered': theory['date_registered'],
              'theory_response_sha256': theory['response_sha256'], **suite, 'manifest_sha256': manifest_sha,
              'manifest_found_in_registration': True, 'read_utc': read_utc}
    show('Read from the registry at ' + read_utc + ':')
    show('  this unit: ' + unit['identifier'] + ', registered ' + str(unit['date_registered']) + ', from ' + str(unit['registered_from'])
         + ', public and approved; title: ' + str(unit['title']))
    show('  1.103: ' + theory['identifier'] + ', registered ' + str(theory['date_registered']) + ', public and approved')
    if suite['suite_registration_id']:
        show('  the suite: ' + suite['suite_registration_id'] + ', registered ' + str(suite['suite_date_registered']))
    show('  the registration holds this stage\'s manifest, sha256 ' + manifest_sha)
    answer = confirm('Type yes if these are the registrations you clicked: ')
    if str(answer).strip().lower() != 'yes':
        raise RuntimeError('STOP: the author did not confirm what was read; nothing was written')
    record['confirmed_by_author'] = True
    record['confirmed_utc'] = datetime.now(timezone.utc).isoformat(timespec='seconds')
    data = json.dumps(record, indent=2, ensure_ascii=False).encode('utf-8') + b'\n'
    with path.open('xb') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())
    return record


def read_attestation(workspace: Path) -> dict:
    path = workspace / ATTESTATION
    if not path.is_file():
        raise RuntimeError('STOP: no attestation; run attest after both registrations read REGISTERED')
    record = _read(path)
    if not isinstance(record, dict) or set(record) != ATTESTATION_FIELDS:
        raise RuntimeError('The attestation does not hold exactly its registered fields')
    if record['study_id'] != STUDY_ID or record['confirmed_by_author'] is not True or record['manifest_found_in_registration'] is not True:
        raise RuntimeError('The attestation is not this study\'s, or the author did not confirm it')
    if not re.fullmatch(r'https://osf\.io/[a-z0-9]{5}/', record.get('registration_url') or ''):
        raise RuntimeError('An actual OSF registration address is required')
    for name in ('read_utc', 'confirmed_utc'):
        when = _moment(record[name])
        if when > datetime.now(timezone.utc):
            raise RuntimeError('The attestation is dated in the future')
    return record


def require_registered(workspace: str | Path | None = None) -> Path:
    """The gate every command passes first. It checks that the author named this workspace, that the attestation is
    his and confirmed, that the stage's manifest is the registered one, that every registered file holds its registered
    bytes or the last logged repair, and that no Python file of the lane's packages is unlisted."""
    value = os.environ.get(WORKSPACE_ENV)
    if not value:
        raise RuntimeError('STOP: author-published OSF registrations required before any study import or command; '
                           'the bench prints the command that names the workspace')
    named = Path(value).resolve()
    if workspace is not None and Path(workspace).resolve() != named:
        raise RuntimeError('The workspace named in ' + WORKSPACE_ENV + ' is another workspace')
    record = read_attestation(named)
    root = stage_root()
    manifest_path = root / MANIFEST
    if not manifest_path.is_file() or _sha(manifest_path) != record['manifest_sha256']:
        raise RuntimeError('The stage\'s manifest is not the registered one')
    manifest = _read(manifest_path)
    files = manifest.get('files')
    if manifest.get('study_id') != STUDY_ID or not isinstance(files, dict) or not files:
        raise RuntimeError('The registered manifest is not this study\'s, or it is empty')
    if CONFIG_REL not in files:
        raise RuntimeError('The registered configuration is not listed in the manifest')
    fixed = {rel for rel in files if _never_repairable(rel)}
    entries = read_repair_log(named / REPAIR_LOG)
    repaired = repair_chains(entries, files, fixed) if entries else {}
    if entries:
        refuse_repairs_after_freeze(entries, named / 'freeze.json')
    look_entries = read_look_repair_log(named / LOOK_REPAIR_LOG)
    if look_entries:
        check_look_repairs(look_entries, named)
        repaired = repair_chains(look_entries, files, fixed, start=repaired)
    for rel, expected in files.items():
        relative = Path(rel)
        target = (root / relative).resolve()
        if relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(root):
            raise RuntimeError('Unsafe registered file path')
        actual = _sha(target) if target.is_file() else None
        if actual is None or (actual != expected and actual != repaired.get(rel)):
            raise RuntimeError(('Source bytes equal neither the registered bytes nor the last logged repair: '
                                if rel in repaired else 'Registered source bytes differ: ') + rel)
    for package in PACKAGES:
        for path in sorted((root / package).rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            if rel not in files:
                raise RuntimeError('A Python file of the lane\'s packages is not in the registered manifest: ' + rel)
    return named


def load_config(workspace: Path) -> dict:
    """The registered configuration, after the gate has checked its bytes."""
    require_registered(workspace)
    config = _read(stage_root() / CONFIG_REL)
    if config.get('study_id') != STUDY_ID:
        raise RuntimeError('The registered configuration is not this study\'s')
    return config
