# Zenodo Batch Upload Guide — ARC / Eden Research Programme

## What to upload to Zenodo
Each of the 22+ papers in `papers/` should be uploaded as a separate Zenodo record.
Each record gets its own DOI. All records link to the parent OSF project (DOI: 10.17605/OSF.IO/6C5XB).

## Per-paper upload steps (repeat for each)
1. Go to https://zenodo.org/uploads/new
2. Title: [Paper Title from PUBLICATION-MANIFEST.json]
3. Authors: Eastwood, Michael Darius
4. Description: [First paragraph of paper]
5. Upload files: the .html + .pdf + CITATION.cff from the paper's directory
6. License: Creative Commons Attribution 4.0 (CC-BY-4.0) for papers
7. Related identifiers: cites 10.17605/OSF.IO/6C5XB (the parent OSF project)
8. Publication date: 2026-01-02 (book publication) or 2026-07-05 (for papers published after)
9. Keywords: AI alignment, AI safety, Eden Protocol, ARC Principle, recursive intelligence, alignment scaling
10. Communities: ai-alignment, artificial-intelligence
11. Publish

## Batch strategy
- Papers I-IX: published as a coherent suite (same date, linked)
- Paper X-XI + HRIH + Paper C: published individually
- Eden Engineering + Eden Vision: published together
- PNP evidence bundle: registered as supporting materials (cite-only)

## OSF registration
- OSF project already exists: DOI 10.17605/OSF.IO/6C5XB
- Register each new Zenodo DOI as a related work on the OSF project
- Or: upload paper PDFs directly to OSF as components of the main project

## After publication
- Update PAPER_PUBLICATION_STATUS.md with DOIs
- Update CANONICAL_INDEX.md with DOIs
- Add DOIs to website paper pages (meta tags)
- Regenerate website mirrors with DOI links
