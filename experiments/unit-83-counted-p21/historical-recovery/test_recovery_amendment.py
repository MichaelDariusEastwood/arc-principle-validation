"""Synthetic fixtures only. No live study module imports, workspace reads or network calls."""
import contextlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.request

import recovery_amendment as ra


class Fixture:
    def __init__(self, base):
        self.unit = Path(base) / 'unit'
        self.root = self.unit / 'workspace'
        self.stage = self.unit / 'stage'
        self.package = self.unit / 'run/amendments/2026-10-28-recovery-01'
        self.package.mkdir(parents=True)
        (self.root / 'private/calls').mkdir(parents=True)
        (self.root / 'private/STUDY.lock').touch()
        (self.stage / 'configs').mkdir(parents=True)
        self.request = {'model': ra.ENGINE, 'service_tier': 'flex', 'input': 'SYNTHETIC BENIGN TEXT',
                        'max_output_tokens': 16000, 'reasoning': {'effort': 'low'}}
        config = {'engines': {ra.ENGINE: {'request_fields': {k: v for k, v in self.request.items() if k != 'input'}}}}
        self.put(self.stage / 'configs/p21.json', config)
        manifest = {'files': {'configs/p21.json': ra.sha((self.stage / 'configs/p21.json').read_bytes())}}
        self.put(self.stage / 'registration_manifest.json', manifest)
        manifest_sha = ra.sha((self.stage / 'registration_manifest.json').read_bytes())
        self.put(self.root / 'environment.json', {'python': sys.version})
        self.put(self.root / 'registration_approval.json', {'registration_id': 'mn73w', 'registered_from': ra.NODE})
        self.put(self.root / 'freeze.json', {'study_id': ra.STUDY, 'files': {
            'stage/registration_manifest.json': manifest_sha,
            'stage/configs/p21.json': manifest['files']['configs/p21.json'],
            'environment.json': ra.sha((self.root / 'environment.json').read_bytes()),
            'registration_approval.json': ra.sha((self.root / 'registration_approval.json').read_bytes())}})
        self.put(self.root / 'lane_state.json', {'done': {'freeze': {'status': 'done'}}})
        source = Path(__file__).parent
        for name in ra.PACKAGE_FILES:
            content = (source / name).read_bytes() if (source / name).exists() else b'Synthetic validation placeholder\n'
            (self.package / name).write_bytes(content)
        self.put(self.package / 'CODE-SHA256.json', {
            name: ra.sha((self.package / name).read_bytes()) for name in ra.PACKAGE_FILES})
        self.slot = 'synthetic-slot'
        self.role = 'synthetic-role'
        self.ident = ra.sha(ra.canonical({'study': ra.STUDY, 'engine': ra.ENGINE, 'role': self.role,
                                        'slot': self.slot, 'body': self.request}))
        self.call = self.root / 'private/calls' / self.ident
        self.call.mkdir()
        self.put(self.call / 'request.json', self.request)
        self.put(self.call / 'result.json', {'headers': {'x-request-id': ra.REMOTE_ID},
            'status': 'http_failure', 'http_status': 400, 'error_type': 'invalid_request_error',
            'error_code': 'invalid_prompt', 'failure_class': 'hold', 'requested_model': ra.ENGINE,
            'engine': ra.ENGINE, 'request_id': self.ident})
        self.put(self.call / 'http_error.txt', {'error': {'code': 'invalid_prompt',
            'message': 'Invalid prompt: your prompt was flagged as potentially violating our usage policy.'}})
        self.db = self.root / 'private/requests.sqlite'
        with sqlite3.connect(self.db) as db:
            db.execute('CREATE TABLE calls (id TEXT PRIMARY KEY, provider TEXT, reservation REAL, charge REAL, state TEXT, detail TEXT, created TEXT, slot TEXT, role TEXT)')
            detail = {'reservation_detail': {'engine': ra.ENGINE, 'request_sha256': ra.sha(ra.canonical(self.request)),
                                             'base_slot': self.slot, 'attempt': 1}}
            db.execute('INSERT INTO calls VALUES (?,?,?,?,?,?,?,?,?)', (self.ident, ra.STUDY + '/openai', .05, .02,
                'request_rejected', json.dumps(detail), ra.REJECTED_UTC, self.slot, self.role))
        self.recovery = ra.Recovery(self.unit, self.package, manifest_sha)
        self.uploads = 0
        self.uploaded = None
        self.corrupt_download = False
        self.component_public = True

    @staticmethod
    def put(path, value):
        Path(path).write_text(json.dumps(value))

    def api_request(self, url, data=None, method='GET'):
        if url == f'https://api.osf.io/v2/nodes/{ra.NODE}/':
            return json.dumps({'data': {'attributes': {'public': self.component_public}}}).encode()
        item = {'id': 'synthetic-file', 'attributes': {'name': self.filename() if self.uploaded else ''},
                'links': {'download': 'https://files.osf.io/synthetic-download'}}
        if method == 'PUT':
            self.uploads += 1
            self.uploaded = data
            item['attributes']['name'] = self.filename()
            return json.dumps({'data': item}).encode()
        if url.endswith('/files/osfstorage/'):
            return json.dumps({'data': [item] if self.uploaded else [], 'links': {'next': None}}).encode()
        if url == 'https://files.osf.io/synthetic-download':
            return b'wrong bytes' if self.corrupt_download else self.uploaded
        raise AssertionError('Unexpected URL in synthetic fixture')

    def filename(self):
        return 'unit83-recovery-amendment-01-' + ra.sha(self.uploaded) + '.zip'

    def prepare_deposit(self):
        self.recovery.prepare()
        with patch.dict(os.environ, {'OSF_COMPONENT_TOKEN': 'synthetic-token'}):
            self.recovery.deposit(self.api_request)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='unit83-ra01-synthetic-')
        self.fixture = Fixture(self.temp.name)
        self.output = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.output)
        self.capture.__enter__()

    def tearDown(self):
        self.capture.__exit__(None, None, None)
        self.temp.cleanup()

    def test_success_is_idempotent_and_database_unchanged(self):
        f = self.fixture
        before = f.db.read_bytes()
        f.prepare_deposit()
        f.recovery.apply()
        log = f.recovery.log.read_bytes()
        deviations = (f.root / 'deviation_log.jsonl').read_bytes()
        defect = (f.unit / 'run/DEFECT-LOG.md').read_bytes()
        f.prepare_deposit()
        f.recovery.apply()
        self.assertEqual(f.uploads, 1)
        self.assertEqual(f.db.read_bytes(), before)
        self.assertEqual(f.recovery.log.read_bytes(), log)
        self.assertEqual((f.root / 'deviation_log.jsonl').read_bytes(), deviations)
        self.assertEqual((f.unit / 'run/DEFECT-LOG.md').read_bytes(), defect)
        entry = ra.recovery_entries(f.recovery.log)[0]
        self.assertEqual(set(entry), ra.FIELDS)
        self.assertEqual(entry['reconciled_charge_usd'], .05)
        self.assertNotIn('SYNTHETIC BENIGN TEXT', self.output.getvalue())
        self.assertNotIn('synthetic-token', self.output.getvalue())

    def test_refuses_changed_request(self):
        f = self.fixture
        f.put(f.call / 'request.json', {**f.request, 'input': 'CHANGED'})
        with self.assertRaisesRegex(ra.Stop, 'journal hash'):
            f.recovery.prepare()
        self.assertFalse(f.recovery.log.exists())

    def test_refuses_changed_stage(self):
        f = self.fixture
        (f.stage / 'configs/p21.json').write_text('{}')
        with self.assertRaisesRegex(ra.Stop, 'source changed'):
            f.recovery.prepare()

    def test_refuses_wrong_provider_request(self):
        f = self.fixture
        result = ra.load(f.call / 'result.json')
        result['headers']['x-request-id'] = 'different-id'
        f.put(f.call / 'result.json', result)
        with self.assertRaisesRegex(ra.Stop, 'Exactly one'):
            f.recovery.prepare()

    def test_refuses_wrong_error(self):
        f = self.fixture
        result = ra.load(f.call / 'result.json')
        result['error_code'] = 'other_error'
        f.put(f.call / 'result.json', result)
        with self.assertRaisesRegex(ra.Stop, 'does not match'):
            f.recovery.prepare()

    def test_refuses_another_hold(self):
        f = self.fixture
        with sqlite3.connect(f.db) as db:
            db.execute('INSERT INTO calls SELECT ?,provider,reservation,charge,state,detail,?, ?,role FROM calls LIMIT 1', ('b'*64, '2026-10-27T23:28:00+00:00', 'another-slot'))
        with self.assertRaisesRegex(ra.Stop, 'Another unresolved hold'):
            f.recovery.prepare()

    def test_refuses_another_rejection_after_apply(self):
        f = self.fixture
        f.prepare_deposit()
        f.recovery.apply()
        before = f.recovery.log.read_bytes()
        with sqlite3.connect(f.db) as db:
            db.execute('INSERT INTO calls SELECT ?,provider,reservation,charge,state,detail,created,?,role FROM calls LIMIT 1', ('c'*64, 'next-attempt'))
        with self.assertRaisesRegex(ra.Stop, 'Another unresolved hold'):
            f.recovery.apply()
        self.assertEqual(before, f.recovery.log.read_bytes())

    def test_requires_verified_deposit(self):
        f = self.fixture
        f.recovery.prepare()
        with self.assertRaises(FileNotFoundError):
            f.recovery.apply()
        self.assertFalse(f.recovery.log.exists())

    def test_bad_download_blocks_release(self):
        f = self.fixture
        f.corrupt_download = True
        with self.assertRaisesRegex(ra.Stop, 'do not match'):
            f.prepare_deposit()
        self.assertFalse(f.recovery.deposit_path.exists())
        self.assertFalse(f.recovery.log.exists())

    def test_restricted_component_visibility_is_disclosed(self):
        f = self.fixture
        f.component_public = False
        f.prepare_deposit()
        f.recovery.apply()
        self.assertFalse(ra.load(f.recovery.deposit_path)['component_public'])
        self.assertIn('restricted; this is not a public registration update', self.output.getvalue())

    def test_upload_recovery_does_not_duplicate_file(self):
        f = self.fixture
        f.corrupt_download = True
        with self.assertRaises(ra.Stop):
            f.prepare_deposit()
        f.corrupt_download = False
        f.prepare_deposit()
        self.assertEqual(f.uploads, 1)

    def test_refuses_after_first_look(self):
        f = self.fixture
        f.put(f.root / 'lane_state.json', {'done': {'look': {'status': 'done'}}})
        with self.assertRaisesRegex(ra.Stop, 'first look/report'):
            f.recovery.prepare()

    def test_refuses_changed_evidence_between_prepare_and_apply(self):
        f = self.fixture
        f.prepare_deposit()
        with (f.call / 'result.json').open('ab') as handle:
            handle.write(b'\n')
        with self.assertRaisesRegex(ra.Stop, 'Evidence changed'):
            f.recovery.apply()

    def test_refuses_invalid_recovery_log(self):
        f = self.fixture
        f.recovery.log.write_text('{"bad": true}\n')
        with self.assertRaisesRegex(ra.Stop, 'Invalid recovery-log'):
            f.recovery.prepare()

    def test_refuses_abandoned_study(self):
        f = self.fixture
        f.recovery.log.write_bytes(ra.canonical({'utc': ra.now(), 'request_id': 'x', 'cause': 'operator',
            'action': 'abandon_study', 'reconciled_charge_usd': None,
            'written_without_reading_any_score': True}) + b'\n')
        with self.assertRaisesRegex(ra.Stop, 'abandoned'):
            f.recovery.prepare()

    def test_active_lock_blocks_helper(self):
        f = self.fixture
        with f.recovery.locked():
            with self.assertRaisesRegex(ra.Stop, 'Another study process'):
                f.recovery.prepare()

    def test_existing_charge_never_reduced(self):
        f = self.fixture
        with sqlite3.connect(f.db) as db:
            db.execute('UPDATE calls SET charge=.08')
        f.prepare_deposit()
        f.recovery.apply()
        self.assertEqual(ra.recovery_entries(f.recovery.log)[0]['reconciled_charge_usd'], .08)

    def test_resume_partial_audit_without_second_release(self):
        f = self.fixture
        f.prepare_deposit()
        original_append = ra.append
        def fail_on_deviation(path, data):
            if Path(path).name == 'deviation_log.jsonl':
                raise OSError('synthetic disk failure')
            return original_append(path, data)
        with patch.object(ra, 'append', fail_on_deviation):
            with self.assertRaises(OSError):
                f.recovery.apply()
        before = f.recovery.log.read_bytes()
        f.recovery.apply()
        self.assertEqual(before, f.recovery.log.read_bytes())
        self.assertTrue((f.package / 'APPLIED.json').exists())

    def test_disclosure_preserves_original_report(self):
        f = self.fixture
        f.prepare_deposit()
        f.recovery.apply()
        with self.assertRaisesRegex(ra.Stop, 'Report is not complete'):
            f.recovery.disclose()
        f.put(f.root / 'lane_state.json', {'done': {'report': {'status': 'done'}}})
        (f.root / 'report').mkdir()
        report = f.root / 'report/REPORT.md'
        report.write_text('SYNTHETIC ORIGINAL REPORT')
        f.recovery.disclose()
        f.recovery.disclose()
        self.assertEqual(report.read_text(), 'SYNTHETIC ORIGINAL REPORT')
        self.assertTrue((f.root / 'report/RECOVERY-AMENDMENT-DISCLOSURE.md').exists())

    def test_token_does_not_follow_cross_host_redirect(self):
        req = urllib.request.Request('https://files.osf.io/one', headers={'Authorization': 'Bearer synthetic-token'})
        nxt = ra.SafeRedirect().redirect_request(req, None, 302, 'redirect', {}, 'https://storage.example/signed')
        self.assertFalse(any(k.lower() == 'authorization' for k in nxt.headers))
        with self.assertRaisesRegex(ra.Stop, 'Non-HTTPS'):
            ra.SafeRedirect().redirect_request(req, None, 302, 'redirect', {}, 'http://storage.example/signed')

    def test_no_prompt_or_keys_in_capsule(self):
        f = self.fixture
        f.prepare_deposit()
        import zipfile
        with zipfile.ZipFile(f.recovery.capsule) as archive:
            incident = json.loads(archive.read('INCIDENT-BEFORE-RESUME.json'))
            self.assertNotIn('input', incident['evidence'])
            self.assertNotIn('SYNTHETIC BENIGN TEXT', json.dumps(incident))
            self.assertNotIn('request.json', archive.namelist())
            self.assertNotIn('result.json', archive.namelist())


if __name__ == '__main__':
    unittest.main(verbosity=2)
