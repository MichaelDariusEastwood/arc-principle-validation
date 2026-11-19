"""Regression checks for released machine and Markdown registration companions."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from check_document_release import registration_companion_findings


class RegistrationCompanionsTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.folder = 'Example'
        prefix = f'papers/{self.folder}/{self.folder}.'
        (self.root / 'papers' / self.folder).mkdir(parents=True)
        self.row = dict(slug='arc-theory-predictions-registration', folder=self.folder, version='1.2',
                        files={k:dict(path=prefix+k, sha256='a'*64) for k in ('pdf','txt','md','json')})
        keys=(2,4,7,9,12,14,17,19,21,24,26,29,31,33)
        fields=[]
        for k in keys:
            text = 'P1 · first\nP2 · second' if k == 24 else f'Body for {k}.'
            fields.append(dict(key=f'343-{k}',title=f'Field {k}',text=text,characters=len(text)))
        self.record=dict(propositions=2, versions=[dict(version='1.2',sha256='a'*64)], fields=fields)
        self.official='\n\n'.join(f['title']+'\n\n'+f['text'] for f in fields)+'\n'
        self.md='# Record\n\nExample. Registration v1.2.\n\n'+self.official
        self.write('txt',self.official)
        self.write('md',self.md)
        self.write_machine()

    def write(self, ext, text):
        data=text.encode()
        (self.root / self.row['files'][ext]['path']).write_bytes(data)
        self.row['files'][ext]['sha256']=hashlib.sha256(data).hexdigest()

    def write_machine(self):
        self.write('json',json.dumps(self.record))

    def issues(self):
        return registration_companion_findings(self.row,self.root)

    def test_current_fields_and_count(self):
        self.assertEqual(self.issues(),[])

    def test_retired_count_with_fresh_checksum(self):
        self.record['propositions']=1
        self.write_machine()
        self.assertIn('machine proposition count disagrees with the numbered Predictions field',self.issues())

    def test_older_markdown_header_or_missing_text(self):
        self.write('md',self.md.replace('Registration v1.2','Registration v1.1'))
        self.assertIn('registration Markdown introduction identifies a different edition',self.issues())
        self.write('md',self.md.replace('Body for 33.','old'))
        self.assertIn('registration Markdown or length differs for 343-33',self.issues())

    def test_incomplete_field_even_with_consistent_reported_length(self):
        field=self.record['fields'][-1]
        field['text']=field['text'][:-1]
        field['characters']=len(field['text'])
        self.write_machine()
        self.assertIn('machine field bodies differ from the official text',self.issues())

    def test_missing_companion_and_stale_pdf(self):
        del self.row['files']['md']
        self.assertIn('missing declared registration companion: md',self.issues())
        self.write_machine()
        self.row['files']['md']=dict(path='papers/Example/Example.md',sha256=hashlib.sha256(self.md.encode()).hexdigest())
        self.record['versions'][0]['sha256']='b'*64
        self.write_machine()
        self.assertIn('machine version history disagrees with the release PDF',self.issues())


if __name__ == '__main__':
    unittest.main()
