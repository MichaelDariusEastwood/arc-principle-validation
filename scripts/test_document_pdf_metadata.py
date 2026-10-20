"""Document-tool regression checks; no study or experimental code is imported.

Discovery: the existing website metadata stamper and canonical document checker
were inspected; this covers lost export metadata, identity drift and safe repair.
The release-profile cases cover the changed-file check: a frozen registration passes
under its own profile with every document-export check kept, and nothing else does.
"""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from pypdf import PdfReader, PdfWriter
from document_pdf_metadata import stamp, state, identity_issues, metadata_findings
from check_document_release import findings, ROOT


class DocumentMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root / 'document.pdf'
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        writer.add_metadata({'/CreationDate': "D:20260101120000+00'00'", '/Producer': 'document check'})
        writer.write(self.path)
        self.fields = dict(title='Document title', authors=['Michael Darius Eastwood'],
                           subject='A document metadata check.', keywords=['document'],
                           doi='https://doi.org/10.17605/OSF.IO/WXPCE',
                           url='https://example.org/document', version='2.4', dates=['2026-10-08'])

    def tearDown(self):
        self.tmp.cleanup()

    def test_render_without_metadata_is_refused(self):
        ok, missing = state(self.path, self.fields)
        self.assertFalse(ok)
        self.assertIn('/Author', missing)
        self.assertIn('XMP', missing)

    def test_repaired_document_cannot_drop_metadata_record(self):
        self.assertTrue(metadata_findings({'metadata_revision': {'date': '2026-10-08'}}))

    def test_required_documents_cannot_remove_both_metadata_fields(self):
        rows = json.loads((ROOT / 'papers/publication-manifest.json').read_text())['documents']
        for row in rows:
            if row['slug'] not in {'hari-treaty-paper', 'hari-treaty-draft-instruments'}:
                continue
            with self.subTest(slug=row['slug']):
                row = copy.deepcopy(row)
                row.pop('pdf_metadata', None)
                row.pop('metadata_revision', None)
                self.assertIn('repaired document has lost its required metadata record', findings(row))

    def test_empty_subject_clears_stale_nonempty_subject(self):
        stamp(self.path, self.fields)
        empty = {**self.fields, 'subject': ''}
        self.assertIn('/Subject', state(self.path, empty)[1])
        stamp(self.path, empty)
        self.assertEqual(PdfReader(self.path).metadata.get('/Subject'), '')
        self.assertEqual(state(self.path, empty), (True, []))
        self.assertFalse(stamp(self.path, empty)['changed'])

    def test_append_only_repair_is_idempotent(self):
        before = self.path.read_bytes()
        report = stamp(self.path, self.fields)
        self.assertTrue(report['append_only'])
        self.assertTrue(self.path.read_bytes().startswith(before))
        self.assertEqual(len(PdfReader(self.path).pages), 1)
        repaired = self.path.read_bytes()
        self.assertFalse(stamp(self.path, self.fields)['changed'])
        self.assertEqual(self.path.read_bytes(), repaired)

    def test_stale_version_is_detected(self):
        stamp(self.path, self.fields)
        next_fields = {**self.fields, 'version': '2.5'}
        self.assertEqual(state(self.path, next_fields), (False, ['XMP']))

    def test_wrong_author_cannot_be_stamped_as_current(self):
        self.root.joinpath('citation.cff').write_text("title: 'Document title'\nversion: '2.4'\nurl: 'https://example.org/document'\ndate-released: '2026-10-08'\nauthors:\n  - family-names: 'Eastwood'\n    given-names: 'Michael Darius'\npreferred-citation:\n  doi: '10.17605/OSF.IO/WXPCE'\n  date-published: '2026-10-08'\nmessage: 'Earlier reference 10.17605/OSF.IO/WXPCE'\n")
        row = {'title': self.fields['title'], 'version': '2.4', 'doi': self.fields['doi'],
               'files': {'cff': {'path': 'citation.cff'}}, 'pdf_metadata': copy.deepcopy(self.fields)}
        self.assertEqual(identity_issues(row, self.root), [])
        row['pdf_metadata']['authors'] = ['Someone Else']
        self.assertIn('metadata author disagrees with citation', identity_issues(row, self.root))

    def test_conflicting_primary_doi_is_not_rescued_by_prose(self):
        self.test_wrong_author_cannot_be_stamped_as_current()
        p = self.root / 'citation.cff'
        p.write_text(p.read_text().replace("  doi: '10.17605/OSF.IO/WXPCE'", "  doi: '10.17605/OSF.IO/67MX8'"))
        row = {k: self.fields[k] for k in ('title', 'version', 'doi')}
        row.update(files={'cff': {'path': 'citation.cff'}}, pdf_metadata=self.fields)
        self.assertIn('metadata DOI disagrees with authoritative citation DOI', identity_issues(row, self.root))

    def test_wrong_original_date_is_rejected(self):
        self.test_wrong_author_cannot_be_stamped_as_current()
        row = {k: self.fields[k] for k in ('title', 'version', 'doi')}
        fields = copy.deepcopy(self.fields)
        fields['dates'] = ['2025-10-08', '2026-10-08']
        row.update(files={'cff': {'path': 'citation.cff'}}, pdf_metadata=fields)
        self.assertIn('metadata original publication date disagrees with preferred citation', identity_issues(row, self.root))


class ReleaseProfileTests(unittest.TestCase):
    """A frozen registration keeps its own name and every release check; no other representation passes."""

    HTML = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Registration</title></head>'
            '<body><main><h1>Registration</h1><p>Registered text.</p></main></body></html>\n')
    CFF = "cff-version: 1.2.0\nversion: '{}'\ntitle: 'Registration'\n"
    PDF_PATH = 'papers/Registration/Registration.pdf'

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'papers/Registration').mkdir(parents=True)
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        pdf = io.BytesIO()
        writer.write(pdf)
        self.row = {'slug': 'registration', 'folder': 'Registration', 'title': 'Registration',
                    'version': '1.103', 'version_date': '15 October 2026',
                    'doi': 'https://doi.org/10.17605/OSF.IO/P8CKQ', 'representation': 'frozen-registration',
                    'files': {'html': self.put('Registration.html', self.HTML.encode('utf-8')),
                              'pdf': self.put('Registration.pdf', pdf.getvalue()),
                              'txt': self.put('Registration.txt', b'Registered text. ' * 100),
                              'cff': self.put('Registration.cff', self.CFF.format('1.103').encode('utf-8'))}}

    def tearDown(self):
        self.tmp.cleanup()

    def put(self, name, data):
        """Write one fixture file and return its manifest entry with the correct hash."""
        rel = 'papers/Registration/' + name
        (self.root / rel).write_bytes(data)
        return {'path': rel, 'sha256': hashlib.sha256(data).hexdigest()}

    def test_frozen_registration_with_correct_bytes_passes(self):
        self.assertEqual(findings(self.row, self.root), [])

    def test_frozen_registration_with_wrong_hash_fails(self):
        row = copy.deepcopy(self.row)
        row['files']['pdf']['sha256'] = '0' * 64
        self.assertEqual(findings(row, self.root), ['hash differs from manifest: ' + self.PDF_PATH])

    def test_unknown_representation_fails(self):
        for representation in ['conversion-pending', 'registration', 'frozen registration', 'Frozen-Registration',
                               'frozen-registration ', 'frozen-registration-v2', '', None, ['frozen-registration']]:
            with self.subTest(representation=representation):
                row = dict(self.row, representation=representation)
                self.assertEqual(findings(row, self.root), ['official document export still required'])
        unlabelled = {k: v for k, v in self.row.items() if k != 'representation'}
        self.assertEqual(findings(unlabelled, self.root), ['official document export still required'])

    def test_document_export_row_behaves_as_before(self):
        row = dict(self.row, representation='document-export')
        self.assertEqual(findings(row, self.root), [])
        row = copy.deepcopy(row)
        row['files']['pdf']['sha256'] = '0' * 64
        self.assertEqual(findings(row, self.root), ['hash differs from manifest: ' + self.PDF_PATH])

    def test_frozen_registration_keeps_every_document_export_check(self):
        page = lambda extra: self.HTML.replace('<main>', extra + '<main>').encode('utf-8')
        fields = dict(title='Registration', authors=['Michael Darius Eastwood'], subject='',
                      keywords=['registration'], doi='https://doi.org/10.17605/OSF.IO/WXPCE',
                      url='https://example.org/registration', version='1.103', dates=['2026-09-08', '2026-10-15'])
        defects = {
            'missing required format: cff': lambda row: row['files'].pop('cff'),
            'manifest path escapes repository': lambda row: row['files']['txt'].update(path='../Registration.txt'),
            'missing file: papers/Registration/Absent.txt':
                lambda row: row['files']['txt'].update(path='papers/Registration/Absent.txt'),
            'hash differs from manifest: papers/Registration/Registration.txt':
                lambda row: row['files']['txt'].update(sha256='0' * 64),
            'PDF signature is missing': lambda row: row['files'].update(pdf=self.put('Unsigned.pdf', b'not a PDF')),
            'website navigation: site-nav':
                lambda row: row['files'].update(html=self.put('Navigation.html', page('<nav class="site-nav"></nav>'))),
            'local-only asset or private path':
                lambda row: row['files'].update(html=self.put('Local.html', page('<img src="http://localhost/f.png" alt="">'))),
            'executable remote script in a document export':
                lambda row: row['files'].update(html=self.put('Script.html', page('<script src="https://example.org/s.js"></script>'))),
            'citation version disagrees with manifest':
                lambda row: row['files'].update(cff=self.put('Older.cff', self.CFF.format('1.102').encode('utf-8'))),
            'text companion below recorded full-text floor':
                lambda row: row['files'].update(txt=self.put('Short.txt', b'Registered text.')),
            'repaired document has lost its required metadata record':
                lambda row: row.update(metadata_revision={'date': '2026-10-20'}),
            'metadata doi disagrees with document manifest': lambda row: row.update(pdf_metadata=fields),
        }
        for expected, spoil in defects.items():
            with self.subTest(defect=expected):
                row = copy.deepcopy(self.row)
                spoil(row)
                frozen = findings(row, self.root)
                self.assertIn(expected, frozen)
                self.assertEqual(frozen, findings(dict(row, representation='document-export'), self.root))

    def test_published_registration_keeps_its_profile_and_is_checked_by_default(self):
        rows = json.loads((ROOT / 'papers/publication-manifest.json').read_text())['documents']
        row = next(r for r in rows if r['slug'] == 'arc-theory-predictions-registration')
        self.assertEqual(row['representation'], 'frozen-registration')
        self.assertEqual(findings(row), [])
        run = subprocess.run([sys.executable, str(ROOT / 'scripts/check_document_release.py')],
                             capture_output=True, text=True, check=False)
        report = json.loads(run.stdout)
        self.assertIn(row['slug'], report['checked'])
        self.assertNotIn(row['slug'], report['remaining_export_conversions'])


if __name__ == '__main__':
    unittest.main()
