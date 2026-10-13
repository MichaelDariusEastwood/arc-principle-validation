#!/usr/bin/env python3
"""Check the papers index and the PDF checksum record against the released files.

docs/paper-pdf-sha256sums.txt must list every papers/<Folder>/<file>.pdf with its current SHA-256, so that
`shasum -a 256 -c docs/paper-pdf-sha256sums.txt` passes. papers/README.md must link every folder in
papers/publication-manifest.json and state each versioned document's manifest version and date. Records move
with the files: a release that changes a PDF or a version updates both records in the same pull request.

Why: on 12 October 2026 five checksum lines failed and the index named three superseded versions and omitted
both HARI Treaty documents, because releases from 8 to 10 October changed the files without these two records.

Discovery, 12 October 2026: eden_capabilities_index.py search 'paper pdf sha256sums checksum record papers index
version check github mirror' searched 4,275 items in 10 kinds; no exact match, 463 near misses. The nearest, the
website's check-paper-version-drift.py and check-paper-file-versions.py, read the website's own surfaces;
check_document_release.py verifies the manifest's hashes, not these two records. No study code is imported or run.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
SUMS = 'docs/paper-pdf-sha256sums.txt'
INDEX = 'papers/README.md'
MANIFEST = 'papers/publication-manifest.json'
MONTHS = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
          'September', 'October', 'November', 'December')


def checksum_findings(root=ROOT):
    out, listed = [], {}
    for number, line in enumerate((root / SUMS).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        digest, separator, rel = line.partition('  ')
        if not separator or not re.fullmatch(r'[0-9a-f]{64}', digest):
            out.append(f'{SUMS} line {number}: not "<sha256>  <path>"')
            continue
        if rel in listed:
            out.append(f'{SUMS}: {rel} is listed twice')
        listed[rel] = digest
    present = sorted(p.relative_to(root).as_posix() for p in root.glob('papers/*/*.pdf'))
    for rel in present:
        actual = hashlib.sha256((root / rel).read_bytes()).hexdigest()
        if rel not in listed:
            out.append(f'{SUMS}: {rel} is not listed; add the line: {actual}  {rel}')
        elif listed[rel] != actual:
            out.append(f'{SUMS}: {rel} is listed as {listed[rel][:12]}; replace the line with: {actual}  {rel}')
    for rel in sorted(set(listed) - set(present)):
        out.append(f'{SUMS}: {rel} is listed but no such PDF exists')
    return out


def index_date(long_date):
    day, month, year = long_date.split()
    if month not in MONTHS or not day.isdigit() or not year.isdigit():
        raise ValueError(long_date)
    return f'{int(day)} {month[:3]} {year}'


def index_findings(root=ROOT):
    out = []
    lines = [l for l in (root / INDEX).read_text(encoding='utf-8').splitlines() if l.startswith('|')]
    for row in json.loads((root / MANIFEST).read_text(encoding='utf-8'))['documents']:
        link = '](' + row['folder'] + '/)'
        rows = [l for l in lines if link in l]
        if not rows:
            out.append(f'{INDEX}: no table row links {row["folder"]}/')
            continue
        if not row.get('version'):
            continue
        try:
            expected = f'v{row["version"]} (revised {index_date(row["version_date"])})'
        except (AttributeError, ValueError):
            out.append(f'{MANIFEST}: {row["slug"]} version_date is not "D Month YYYY"')
            continue
        if not any('| ' + expected + ' |' in l for l in rows):
            out.append(f'{INDEX}: the {row["folder"]} row should state "{expected}", as the manifest records')
    return out


def main():
    problems = checksum_findings() + index_findings()
    for problem in problems:
        print(problem)
    print(json.dumps({'checksum_record': SUMS, 'index': INDEX, 'problems': len(problems)}))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
