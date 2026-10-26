"""Release-tool regressions for alias files; no study code or research data is imported.

Discovery, 21 October 2026: the alias check extends check_paper_records.py, whose module docstring records the
eden_capabilities_index search. These cases cover the HRIH drift of 15 October 2026, when the primary text and
citation files moved to version 3.52 and both legacy-named alias files kept the earlier text.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from check_paper_records import alias_findings, MANIFEST, ROOT

ALIAS = 'papers/Paper-Z/Legacy-Name.txt'
PRIMARY = 'papers/Paper-Z/Paper-Z.txt'


class AliasRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'papers/Paper-Z').mkdir(parents=True)
        (self.root / PRIMARY).write_text('Version: 3.52\n', encoding='utf-8')
        (self.root / ALIAS).write_text('Version: 3.52\n', encoding='utf-8')
        self.row = {'slug': 'paper-z', 'folder': 'Paper-Z',
                    'files': {'txt': {'path': PRIMARY, 'sha256': '0' * 64}},
                    'aliases': [{'path': ALIAS, 'primary': 'txt'}]}

    def tearDown(self):
        self.tmp.cleanup()

    def findings(self, row=None):
        (self.root / MANIFEST).write_text(json.dumps({'documents': [row or self.row]}), encoding='utf-8')
        return alias_findings(self.root)

    def test_alias_carrying_its_primary_bytes_passes(self):
        self.assertEqual(self.findings(), [])

    def test_primary_moved_and_alias_left_at_the_earlier_text_is_refused(self):
        (self.root / ALIAS).write_text('Version: 3.51\n', encoding='utf-8')
        self.assertEqual(self.findings(), [
            ALIAS + ': differs from its primary ' + PRIMARY + "; copy the primary file's bytes to it"])

    def test_missing_alias_file_is_refused(self):
        (self.root / ALIAS).unlink()
        self.assertEqual(self.findings(), [
            ALIAS + ': listed as an alias of ' + PRIMARY + ', but no such file exists'])

    def test_alias_of_a_format_the_row_does_not_declare_is_refused(self):
        self.row['aliases'][0]['primary'] = 'cff'
        self.assertIn('which the row does not declare', ' '.join(self.findings()))

    def test_alias_must_be_a_separate_file_of_the_same_kind_in_its_own_folder(self):
        for path in ('papers/Other/Legacy-Name.txt', 'papers/Paper-Z/old/Legacy-Name.txt',
                     PRIMARY, 'papers/Paper-Z/Legacy-Name.cff'):
            with self.subTest(path=path):
                row = copy.deepcopy(self.row)
                row['aliases'][0]['path'] = path
                self.assertIn('must be a separate .txt file directly in papers/Paper-Z/', ' '.join(self.findings(row)))

    def test_malformed_or_repeated_alias_entries_are_refused(self):
        cases = [([], 'aliases must be a non-empty list'),
                 (ALIAS, 'aliases must be a non-empty list'),
                 ([{'path': ALIAS}], 'without a "path" and a "primary" format'),
                 ([{'path': ALIAS, 'primary': 'txt'}, {'path': ALIAS, 'primary': 'txt'}], 'lists the alias ' + ALIAS + ' twice')]
        for aliases, expected in cases:
            with self.subTest(aliases=aliases):
                row = copy.deepcopy(self.row)
                row['aliases'] = aliases
                self.assertIn(expected, ' '.join(self.findings(row)))

    def test_repository_legacy_names_stay_listed_and_carry_their_primary_bytes(self):
        rows = {r['slug']: r for r in json.loads((ROOT / MANIFEST).read_text(encoding='utf-8'))['documents']}
        for slug in ('paper-xi-convergence', 'paper-c-pnp', 'hrih-paper'):
            with self.subTest(slug=slug):
                self.assertEqual(sorted(a['primary'] for a in rows[slug].get('aliases', [])), ['cff', 'txt'])
        self.assertEqual(alias_findings(ROOT), [])


if __name__ == '__main__':
    unittest.main()
