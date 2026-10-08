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
