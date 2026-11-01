#!/usr/bin/env python3
"""Operator-run, one-incident recovery addendum. No provider-generation calls."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile

UNIT = Path(__file__).resolve().parents[3]
MANIFEST = '013d7008aa92a7867385d20f15e6754988e1d83d20eebc6096b20b9e5505e026'
STUDY = '41-D-COUNTED-v0.1'
REMOTE_ID = 'req_c76fb04d25bb4d74b142cc3c503589fb'
REJECTED_UTC = '2026-10-27T23:27:13+00:00'
ENGINE = 'gpt-6-luna'
NODE = 'zwnc5'
CAUSE = 'unit83_ra01_operator_authorised_unchanged_request_retry'
HOLD_STATES = ('request_rejected', 'identity_drift', 'pricing_tier_mismatch',
               'usage_missing', 'reservation_exceeded')
FIELDS = {'utc', 'request_id', 'cause', 'action', 'reconciled_charge_usd',
          'written_without_reading_any_score'}
PACKAGE_FILES = ('RECOVERY-AMENDMENT-01.md', 'recovery_amendment.py', 'resume-unit83.zsh',
                 'test_recovery_amendment.py', 'VALIDATION.md')


class Stop(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Stop(message)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unique(items):
    out = {}
    for key, value in items:
        require(key not in out, 'Duplicate JSON key; stop for review.')
        out[key] = value
    return out


def load(path):
    return json.loads(Path(path).read_text('utf-8'), object_pairs_hook=unique)


def write_once(path, data):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == data, 'Existing amendment artifact differs: ' + path.name)
        return
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def save_once(path, value):
    write_once(path, canonical(value) + b'\n')


def append(path, data):
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with path.open('rb') as handle:
            handle.seek(-1, 2)
            require(handle.read() == b'\n', 'Append-only file lacks final newline: ' + path.name)
    with path.open('ab') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def recovery_entries(path):
    entries = []
    if not path.exists():
        return entries
    for line in path.read_text('utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line, object_pairs_hook=unique)
        require(isinstance(r, dict) and set(r) == FIELDS, 'Invalid recovery-log fields.')
        require(r['written_without_reading_any_score'] is True, 'Invalid recovery attestation.')
        require(r['action'] in ('reissue', 'abandon_study'), 'Invalid recovery action.')
        require(all(isinstance(r[k], str) and r[k].strip() for k in ('utc', 'request_id', 'cause')), 'Invalid recovery strings.')
        require(datetime.fromisoformat(r['utc'].replace('Z', '+00:00')).tzinfo is not None, 'Undated recovery entry.')
        require(finite(r['reconciled_charge_usd']) or
                (r['action'] == 'abandon_study' and r['reconciled_charge_usd'] is None), 'Invalid recovery charge.')
        require(r['action'] != 'abandon_study', 'The study was abandoned; this helper cannot resume it.')
        entries.append(r)
    return entries


class Recovery:
    def __init__(self, unit=UNIT, package=None, expected_manifest=MANIFEST):
        self.unit = Path(unit)
        self.root = self.unit / 'workspace'
        self.stage = self.unit / 'stage'
        self.package = Path(package) if package else Path(__file__).resolve().parent
        self.expected_manifest = expected_manifest
        self.log = self.root / 'private/recovery_log.jsonl'
        self.incident_path = self.package / 'INCIDENT-BEFORE-RESUME.json'
        self.capsule = self.package / 'AMENDMENT-CAPSULE.zip'
        self.deposit_path = self.package / 'DEPOSIT-RECEIPT.json'

    @contextmanager
    def locked(self):
        path = self.root / 'private/STUDY.lock'
        require(path.is_file(), 'The study lock is missing.')
        with path.open('r+b') as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise Stop('Another study process is running. Do not start a second run.') from exc
            yield

    def check_package(self):
        hashes = load(self.package / 'CODE-SHA256.json')
        require(set(hashes) == set(PACKAGE_FILES), 'Amendment package manifest differs.')
        for name, expected in hashes.items():
            require(sha((self.package / name).read_bytes()) == expected, 'Amendment file changed: ' + name)

    def frozen(self):
        self.check_package()
        manifest_path = self.stage / 'registration_manifest.json'
        require(sha(manifest_path.read_bytes()) == self.expected_manifest, 'Registered manifest differs.')
        manifest = load(manifest_path)
        for rel, expected in manifest['files'].items():
            path = (self.stage / rel).resolve()
            require(path.is_relative_to(self.stage.resolve()), 'Unsafe registered path.')
            require(sha(path.read_bytes()) == expected, 'Registered source changed: ' + rel)
        record = load(self.root / 'freeze.json')
        require(record.get('study_id') == STUDY, 'Wrong frozen study.')
        for rel, expected in record['files'].items():
            base = self.stage if rel.startswith('stage/') else self.root
            path = (base / (rel[6:] if rel.startswith('stage/') else rel)).resolve()
            require(path.is_relative_to(base.resolve()), 'Unsafe freeze path.')
            require(sha(path.read_bytes()) == expected, 'Frozen bytes changed: ' + rel)
        env = load(self.root / 'environment.json')
        require(env['python'].split()[0] == sys.version.split()[0], 'Use the Python version recorded at the freeze.')
        attestation = load(self.root / 'registration_approval.json')
        require(attestation.get('registration_id') == 'mn73w' and attestation.get('registered_from') == NODE,
                'Registration/source component mismatch.')
        return sha((self.root / 'freeze.json').read_bytes())

    def target(self, recorded_id=None):
        entries = recovery_entries(self.log)
        released = {r['request_id'] for r in entries}
        dbpath = self.root / 'private/requests.sqlite'
        with sqlite3.connect(dbpath.as_uri() + '?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            if recorded_id:
                rows = db.execute('SELECT * FROM calls WHERE id=?', (recorded_id,)).fetchall()
            else:
                rows = db.execute("SELECT * FROM calls WHERE state='request_rejected' AND created=?",
                                  (REJECTED_UTC,)).fetchall()
            matches = []
            for raw in rows:
                row = dict(raw)
                require(re.fullmatch(r'[0-9a-f]{64}', row['id']) is not None, 'Malformed journal request ID.')
                folder = self.root / 'private/calls' / row['id']
                result = load(folder / 'result.json')
                if result.get('headers', {}).get('x-request-id') == REMOTE_ID:
                    matches.append((row, result, folder))
            require(len(matches) == 1, 'Exactly one matching original rejection is required.')
            row, result, folder = matches[0]
            marks = ','.join('?' for _ in HOLD_STATES)
            held = db.execute(f"SELECT id FROM calls WHERE state<>'reserved' AND (charge IS NULL OR state IN ({marks}))",
                              HOLD_STATES).fetchall()
            require(all(r['id'] == row['id'] or r['id'] in released for r in held),
                    'Another unresolved hold exists; this amendment cannot release it.')
        require(row['provider'] == STUDY + '/openai' and row['created'] == REJECTED_UTC, 'Wrong provider or timestamp.')
        require(row['state'] in ('request_rejected', 'held_released'), 'The original request is in an unexpected state.')
        require(result.get('status') == 'http_failure' and result.get('http_status') == 400 and
                result.get('error_type') == 'invalid_request_error' and result.get('error_code') == 'invalid_prompt' and
                result.get('failure_class') == 'hold' and result.get('requested_model') == ENGINE and
                result.get('request_id') == row['id'], 'The rejection does not match the authorised incident.')
        error = load(folder / 'http_error.txt').get('error', {})
        require(error.get('code') == 'invalid_prompt' and
                'flagged as potentially violating our usage policy' in error.get('message', ''), 'The recorded error differs.')
        detail = json.loads(row['detail'], object_pairs_hook=unique)
        original = detail['reservation_detail']
        request = load(folder / 'request.json')
        digest = sha(canonical(request))
        require(digest == original['request_sha256'], 'Saved request does not match its journal hash.')
        config = load(self.stage / 'configs/p21.json')
        require(original['engine'] == ENGINE and result['engine'] == ENGINE, 'Wrong engine.')
        for key, value in config['engines'][ENGINE]['request_fields'].items():
            require(request.get(key) == value, 'Saved request setting differs: ' + key)
        require(request.get('model') == ENGINE and request.get('service_tier') == 'flex', 'Model or tier differs.')
        computed_id = sha(canonical({'study': STUDY, 'engine': ENGINE, 'role': row['role'],
                                    'slot': row['slot'], 'body': request}))
        require(computed_id == row['id'], 'Full request identity does not match the journal.')
        require(finite(row['reservation']) and (row['charge'] is None or finite(row['charge'])), 'Invalid journal amount.')
        snapshot = {'request_id': row['id'], 'provider_request_id': REMOTE_ID,
                    'rejected_utc': REJECTED_UTC, 'engine': ENGINE,
                    'original_request_sha256': digest, 'result_sha256': sha((folder / 'result.json').read_bytes()),
                    'http_error_sha256': sha((folder / 'http_error.txt').read_bytes()),
                    'role': row['role'], 'slot': row['slot'], 'base_slot': original['base_slot'],
                    'attempt': original['attempt']}
        return snapshot, row, entries

    def prepare(self):
        with self.locked():
            frozen_sha = self.frozen()
            if self.incident_path.exists():
                incident = load(self.incident_path)
                snapshot, _, _ = self.target(incident['evidence']['request_id'])
                require(snapshot == incident['evidence'] and frozen_sha == incident['freeze_sha256'],
                        'Original incident evidence has changed.')
            else:
                state = load(self.root / 'lane_state.json')
                require(not any(k in state.get('done', {}) for k in ('look', 'report')),
                        'The first look/report has already completed; this amendment was prepared for a pre-look hold.')
                snapshot, row, entries = self.target()
                require(not any(r['request_id'] == row['id'] or r['cause'].startswith(CAUSE) for r in entries),
                        'This incident already has a recovery authorisation; do not create another.')
                incident = {'amendment': 'Unit83-RA01', 'prepared_utc': now(), 'registration': 'mn73w',
                            'component': NODE, 'freeze_sha256': frozen_sha,
                            'registration_manifest_sha256': self.expected_manifest,
                            'code_manifest_sha256': sha((self.package / 'CODE-SHA256.json').read_bytes()),
                            'status': 'operator-authorised post-registration recovery deviation; not original preregistration',
                            'no_study_scores_or_answers_inspected_by_helper': True, 'evidence': snapshot}
                save_once(self.incident_path, incident)
            names = (*PACKAGE_FILES, 'CODE-SHA256.json', self.incident_path.name)
            if not self.capsule.exists():
                with zipfile.ZipFile(self.capsule, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
                    for name in names:
                        archive.writestr(name, (self.package / name).read_bytes())
            self.check_capsule()
            print('Amendment prepared; original request and frozen source hashes verified. No hold released.')

    def check_capsule(self):
        names = (*PACKAGE_FILES, 'CODE-SHA256.json', self.incident_path.name)
        with zipfile.ZipFile(self.capsule) as archive:
            require(sorted(archive.namelist()) == sorted(names), 'Unexpected amendment capsule contents.')
            for name in names:
                require(archive.read(name) == (self.package / name).read_bytes(), 'Capsule content mismatch: ' + name)
        return sha(self.capsule.read_bytes())

    def deposit(self, requester=None):
        with self.locked():
            self.frozen()
            digest = self.check_capsule()
            token = os.environ.get('OSF_COMPONENT_TOKEN')
            require(bool(token), 'OSF_COMPONENT_TOKEN is absent. Load it from Keychain in your Terminal.')
            request = requester or (lambda url, **kw: osf_request(url, token, **kw))
            node = json.loads(request(f'https://api.osf.io/v2/nodes/{NODE}/'))
            component_public = node.get('data', {}).get('attributes', {}).get('public')
            require(isinstance(component_public, bool), 'OSF did not report the component visibility.')
            filename = 'unit83-recovery-amendment-01-' + digest + '.zip'
            matches = []
            url = f'https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/'
            for _ in range(200):
                require(is_osf(url), 'Unexpected OSF listing URL.')
                body = json.loads(request(url))
                for item in body.get('data') or []:
                    if item.get('attributes', {}).get('name') == filename:
                        matches.append(item)
                url = body.get('links', {}).get('next')
                if not url:
                    break
            else:
                raise Stop('OSF listing exceeded the pagination limit.')
            require(len(matches) <= 1, 'Duplicate named amendment deposits; review before continuing.')
            if matches:
                item = matches[0]
            else:
                url = f'https://files.osf.io/v1/resources/{NODE}/providers/osfstorage/?kind=file&name=' + urllib.parse.quote(filename)
                item = json.loads(request(url, data=self.capsule.read_bytes(), method='PUT')).get('data', {})
            download = item.get('links', {}).get('download')
            require(isinstance(download, str), 'OSF did not provide a download link.')
            require(sha(request(download)) == digest, 'OSF downloaded bytes do not match the amendment capsule.')
            if self.deposit_path.exists():
                receipt = load(self.deposit_path)
                require(receipt['capsule_sha256'] == digest and receipt['component'] == NODE,
                        'Existing deposit receipt differs.')
            else:
                receipt = {'amendment': 'Unit83-RA01', 'component': NODE, 'component_public': component_public,
                           'uploaded_and_read_back_utc': now(), 'capsule_sha256': digest,
                           'osf_file_id': item.get('id'), 'download_url': download,
                           'osf_name': filename, 'registration_revision_approved': False}
                save_once(self.deposit_path, receipt)
            print('OSF addendum deposited and downloaded with matching SHA-256. No hold released.')
            print('Component visibility: ' + ('public' if component_public else 'restricted; this is not a public registration update'))

    def apply(self):
        with self.locked():
            frozen_sha = self.frozen()
            digest = self.check_capsule()
            deposit = load(self.deposit_path)
            require(deposit.get('capsule_sha256') == digest and deposit.get('component') == NODE and
                    isinstance(deposit.get('component_public'), bool), 'Verified deposit receipt with visibility is required.')
            incident = load(self.incident_path)
            snapshot, row, entries = self.target(incident['evidence']['request_id'])
            require(snapshot == incident['evidence'] and frozen_sha == incident['freeze_sha256'], 'Evidence changed before release.')
            cause = CAUSE + ';capsule_sha256=' + digest
            relevant = [r for r in entries if r['request_id'] == row['id'] or r['cause'].startswith(CAUSE)]
            if relevant:
                require(len(relevant) == 1 and relevant[0]['cause'] == cause and relevant[0]['request_id'] == row['id'],
                        'Conflicting or duplicate recovery entries; stop for review.')
                entry = relevant[0]
                require(entry['action'] == 'reissue' and entry['reconciled_charge_usd'] >= max(row['reservation'], row['charge'] or 0),
                        'Recovery action or conservative accounting differs.')
            else:
                require(row['state'] == 'request_rejected', 'Only the original rejected attempt may be released.')
                state = load(self.root / 'lane_state.json')
                require(not any(k in state.get('done', {}) for k in ('look', 'report')), 'A look/report has completed before release.')
                entry = {'utc': now(), 'request_id': row['id'], 'cause': cause, 'action': 'reissue',
                         'reconciled_charge_usd': max(row['reservation'], row['charge'] or 0),
                         'written_without_reading_any_score': True}
                append(self.log, canonical(entry) + b'\n')
            deviation = {'utc': entry['utc'], 'kind': 'post_registration_recovery_amendment',
                         'amendment': 'Unit83-RA01', 'cause': cause, 'action': 'reissue',
                         'request_id': row['id'], 'provider_request_id': REMOTE_ID,
                         'capsule_sha256': digest, 'osf_component': NODE, 'original_protocol_unchanged': False,
                         'frozen_study_source_unchanged': True, 'maximum_policy_hold_releases': 1}
            path = self.root / 'deviation_log.jsonl'
            existing = [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []
            matches = [r for r in existing if r.get('kind') == deviation['kind'] and r.get('amendment') == 'Unit83-RA01']
            require(not matches or matches == [deviation], 'Conflicting amendment deviation record.')
            if not matches:
                append(path, canonical(deviation) + b'\n')
            defect = self.unit / 'run/DEFECT-LOG.md'
            marker = 'Unit83-RA01 capsule ' + digest
            text = defect.read_text() if defect.exists() else ''
            if marker not in text:
                append(defect, ('\n## ' + entry['utc'] + ' — ' + marker + '\n\n'
                               'Operator-authorised post-registration recovery amendment: one unchanged-request '
                               'reissue authorisation for OpenAI ' + REMOTE_ID + '. Original frozen source, '
                               'request, model and statistical rules unchanged; recovery procedure amended. '
                               'Addendum read back from OSF component zwnc5 before release; visibility recorded in deposit receipt. '
                               'No study scores or generated answers inspected. Amended-protocol disclosure '
                               'must accompany any final report. Not an original-protocol pass.\n').encode())
            save_once(self.package / 'APPLIED.json', {'recovery_entry': entry, 'capsule_sha256': digest,
                                                     'audit_complete': True})
            print('One incident-specific recovery authorisation recorded. No provider-generation call was made.')
            print('A new policy rejection is NOT authorised for automatic release by this amendment.')

    def disclose(self):
        with self.locked():
            self.check_package()
            require(load(self.package / 'APPLIED.json')['audit_complete'] is True, 'Amendment application is incomplete.')
            state = load(self.root / 'lane_state.json')
            require(state.get('done', {}).get('report', {}).get('status') == 'done', 'Report is not complete.')
            require((self.root / 'report/REPORT.md').is_file(), 'Completed report is missing.')
            deposit = load(self.deposit_path)
            text = ('# Mandatory companion to the Unit 83 report\n\n'
                    'This run continued under the operator-authorised post-registration Recovery Amendment 01 '
                    'of 28 October 2026. One OpenAI HTTP 400 invalid_prompt policy rejection was authorised '
                    'for an unchanged-request retry. The original rejection and subsequent attempts remain '
                    'journalled. No determination that the rejection was a false positive is made.\n\n'
                    'The frozen prompts, model, source and statistical rules were unchanged, but the recovery '
                    'procedure was amended after collection started. Any original outcome wording in REPORT.md '
                    'must be read with this disclosure; the run must not be presented as an unchanged execution '
                    'of the preregistration. Treat the result as amended-protocol/preliminary evidence pending '
                    'assessment of the deviation and independent replication.\n\n'
                    'Addendum component: https://osf.io/zwnc5/\n\n'
                    'Component visibility at deposit: ' + ('public' if deposit['component_public'] else 'restricted') + '. '
                    'The original sharing settings were preserved. Include the addendum in the public results package.\n\n'
                    'Capsule SHA-256: ' + deposit['capsule_sha256'] + '\n\n'
                    'The full addendum and implementation are preserved under '
                    'unit/run/amendments/2026-10-28-recovery-01/. Include that directory, the incident, '
                    'deposit and application receipts, and this disclosure in the results package. '
                    'Deposit of the addendum is not OSF approval of a revised registration.\n')
            write_once(self.root / 'report/RECOVERY-AMENDMENT-DISCLOSURE.md', text.encode())
            print('Mandatory recovery-amendment disclosure saved beside the completed report.')


def is_osf(url):
    parts = urllib.parse.urlsplit(url)
    host = (parts.hostname or '').lower()
    return parts.scheme == 'https' and (host == 'osf.io' or host.endswith('.osf.io')) and not parts.username and not parts.password


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        require(urllib.parse.urlsplit(newurl).scheme == 'https', 'Non-HTTPS redirect refused.')
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and not is_osf(newurl):
            for store in (new.headers, new.unredirected_hdrs):
                for name in list(store):
                    if name.lower() == 'authorization':
                        del store[name]
        return new


def osf_request(url, token, data=None, method='GET'):
    parts = urllib.parse.urlsplit(url)
    require(parts.scheme == 'https' and not parts.username and not parts.password, 'Unsafe download address.')
    require(method == 'GET' or is_osf(url), 'Uploads are restricted to OSF.')
    headers = {'User-Agent': 'unit83-ra01-deposit'}
    if is_osf(url):
        headers['Authorization'] = 'Bearer ' + token
    if data is not None:
        headers['Content-Type'] = 'application/octet-stream'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.build_opener(SafeRedirect()).open(req, timeout=45) as response:
        body = response.read(20_000_001)
    require(len(body) <= 20_000_000, 'OSF response exceeds the metadata/capsule limit.')
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'deposit', 'apply', 'disclose'))
    args = parser.parse_args()
    try:
        getattr(Recovery(), args.command)()
        return 0
    except Stop as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
    except urllib.error.HTTPError as exc:
        print('STOP: OSF returned HTTP %d. No new provider call was made. Preserve files and report this code.' % exc.code, file=sys.stderr)
    except Exception as exc:
        print('STOP: amendment operation failed (' + type(exc).__name__ + '). Preserve files for review.', file=sys.stderr)
    return 1


if __name__ == '__main__':
    sys.exit(main())
