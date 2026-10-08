"""Document-tool regression checks; no study or experimental code is imported.

Discovery: the existing website metadata stamper and canonical document checker
were inspected; this covers lost export metadata, identity drift and safe repair.
"""
import copy
from pathlib import Path
import tempfile
import unittest
from pypdf import PdfReader, PdfWriter
from document_pdf_metadata import stamp, state, identity_issues, metadata_findings


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


if __name__ == '__main__':
    unittest.main()
