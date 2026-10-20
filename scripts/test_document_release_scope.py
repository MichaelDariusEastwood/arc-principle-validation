"""Release-tool regressions only; no study code or research data."""
import unittest

from check_document_release import affected_documents


class ChangedDocumentScopeTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"slug": "legacy", "folder": "Legacy",
             "representation": "conversion-pending",
             "files": {"pdf": {"path": "papers/Legacy/Legacy.pdf"}}},
            {"slug": "frozen", "folder": "Frozen",
             "representation": "frozen-registration",
             "files": {"md": {"path": "papers/Frozen/README.md"}}},
        ]

    def test_readme_only_correction_does_not_request_an_unrelated_edition(self):
        self.assertEqual(affected_documents(self.rows, [
            "README.md", "papers/Legacy/README.md"]), set())

    def test_canonical_pdf_still_selects_pending_document(self):
        self.assertEqual(affected_documents(self.rows, [
            "papers/Legacy/Legacy.pdf"]), {"legacy"})

    def test_other_text_and_source_files_keep_existing_conservative_scope(self):
        for name in ("Legacy.html", "Legacy.txt", "Legacy.cff", "Legacy.md",
                     "Legacy.json", "Legacy.tex", "code/check.py"):
            with self.subTest(name=name):
                self.assertEqual(affected_documents(self.rows, [
                    "papers/Legacy/" + name]), {"legacy"})

    def test_manifest_declared_readme_cannot_escape_a_release_check(self):
        self.assertEqual(affected_documents(self.rows, [
            "papers/Frozen/README.md"]), {"frozen"})

    def test_readme_plus_pdf_still_checks_the_paper(self):
        self.assertEqual(affected_documents(self.rows, [
            "papers/Legacy/README.md", "papers/Legacy/Legacy.pdf"]), {"legacy"})

    def test_neighbouring_folder_prefix_does_not_match(self):
        self.assertEqual(affected_documents(self.rows, [
            "papers/Legacy-Other/Legacy.pdf"]), set())

    def test_case_variant_is_not_silently_exempted(self):
        self.assertEqual(affected_documents(self.rows, [
            "papers/Legacy/readme.md"]), {"legacy"})

    def test_multiple_documents_keep_their_own_scope(self):
        self.assertEqual(affected_documents(self.rows, [
            "papers/Legacy/README.md", "papers/Frozen/README.md"]), {"frozen"})


if __name__ == "__main__":
    unittest.main()
