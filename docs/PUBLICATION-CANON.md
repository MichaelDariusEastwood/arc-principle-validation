# Released research documents

The canonical released documents live in this repository's `papers/` directory.
Each document keeps one folder and one filename per format. Its Git history records
each released version. Private working sources remain private.

The document HTML and PDF are a pair from the same export. The HTML contains the
paper or instrument, its figures, references and contents, without website navigation,
menus or other website controls. The text companion holds the full document text;
the citation file carries that document's version and date.

The website has its own presentation pages and receives the released PDF bytes.
OSF receives the document HTML and that same PDF, retaining the oldest file identity
and version history. A website page is never an OSF export merely because it is HTML.

`papers/publication-manifest.json` records versions, files, hashes and representations.
Rows awaiting document exports are explicitly marked and cannot pass a release check.
Existing public copies remain available while replacements are prepared. Importing a
release does not close any outstanding scientific, legal or source review.

Before a document release or deposit, run:

```
python3 scripts/check_document_release.py --slug <document-slug>
```

This verifies recorded bytes, formats, citation versions, the full-text floor and HTML
representation. It does not prove the science or replace visual PDF review. It imports
no study code and performs no experiment.

Published registrations remain frozen. A text correction requires a successor version
approved and registered by the author. Historical files and licence grants retain their
history. T2 prepares documents and deposits; T4 reviews and merges them and carries
matching files to the website.

## Document metadata before release

Before local checks, provide the document-only dependencies with `python3 -m pip install -r scripts/requirements-documents.txt` in the chosen environment. The hosted document workflow installs the same pinned dependencies.

After exporting a document, apply its recorded metadata with `python3 scripts/document_pdf_metadata.py --slug <document-slug> --apply`. Check the citation identity first; a metadata revision must not relabel an old document. This tool refuses changes to existing content objects and preserves the original render dates. Record the resulting PDF hash in `papers/publication-manifest.json`, then run `python3 scripts/check_document_release.py`. Its default checks all current document exports; a lost HARI metadata record must be restored before release.

The metadata-only HARI repair of 8 October 2026 keeps Paper 2.4 and Instruments 2.3 at their original visible dates. Both documents retain their working-draft status and outstanding review. The website and OSF must receive these canonical PDF bytes; neither mirror independently stamps or re-renders them.
