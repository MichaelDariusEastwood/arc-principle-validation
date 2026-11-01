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

Aliases: a manifest row may list files kept under an earlier name so that old links keep working. Each entry
gives the alias's path and the format of its primary file in the row's files, and the alias must carry that
primary file's bytes exactly. Why: on 15 October 2026 the HRIH 3.52 release (commit 61e19be) updated
HRIH-Paper.txt and HRIH-Paper.cff but left both legacy-named alias files at the earlier text, and no check
noticed until the census of 21 October 2026.

Discovery, 21 October 2026: eden_capabilities_index.py search 'alias file primary bytes identical legacy name
check manifest' searched 4,347 items in 10 kinds; no exact match, 459 near misses. The website's
canonical_pdf_errors compares its own download routes with this manifest; check_document_release.py hashes only
the files a row declares. Neither reads an alias file, so the alias check extends this record check.
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


def alias_findings(root=ROOT):
    """Every alias a manifest row lists must carry its primary file's bytes (see Aliases above).

    This extends the record check; the discover-first search of 21 October 2026 is recorded in the module docstring.
    """
    out = []
    for row in json.loads((root / MANIFEST).read_text(encoding='utf-8'))['documents']:
        if 'aliases' not in row:
            continue
        slug, aliases = row.get('slug'), row['aliases']
        if not isinstance(aliases, list) or not aliases:
            out.append(f'{MANIFEST}: {slug} aliases must be a non-empty list')
            continue
        files = row.get('files') or {}
        declared = {item.get('path') for item in files.values() if isinstance(item, dict)}
        prefix = 'papers/' + row['folder'] + '/'
        seen = set()
        for alias in aliases:
            if (not isinstance(alias, dict) or not isinstance(alias.get('path'), str)
                    or not isinstance(alias.get('primary'), str)):
                out.append(f'{MANIFEST}: {slug} has an alias entry without a "path" and a "primary" format')
                continue
            rel, fmt = alias['path'], alias['primary']
            if rel in seen:
                out.append(f'{MANIFEST}: {slug} lists the alias {rel} twice')
                continue
            seen.add(rel)
            primary = files.get(fmt, {}).get('path') if isinstance(files.get(fmt), dict) else None
            if not primary:
                out.append(f'{MANIFEST}: {slug} alias {rel} names the primary format "{fmt}", which the row does not declare')
                continue
            if (not rel.startswith(prefix) or '/' in rel[len(prefix):] or rel in declared
                    or Path(rel).suffix != Path(primary).suffix):
                out.append(f'{MANIFEST}: {slug} alias {rel} must be a separate {Path(primary).suffix} file directly in {prefix}')
                continue
            if not (root / rel).is_file():
                out.append(f'{rel}: listed as an alias of {primary}, but no such file exists')
            elif not (root / primary).is_file():
                out.append(f'{primary}: the primary of the alias {rel} does not exist')
            elif (root / rel).read_bytes() != (root / primary).read_bytes():
                out.append(f'{rel}: differs from its primary {primary}; copy the primary file\'s bytes to it')
    return out


def readme_title_findings(root=ROOT):
    """A current document-export README's Full title follows its manifest title.

    A catalogue name can be retained explicitly as a former title. It must not
    remain labelled Full title after the manuscript's next edition changes it.
    """
    out = []
    for row in json.loads((root / MANIFEST).read_text(encoding='utf-8'))['documents']:
        if row.get('representation') != 'document-export':
            continue
        rel = 'papers/' + row['folder'] + '/README.md'
        path = root / rel
        if not path.is_file():
            continue
        titles = re.findall(r'^\*\*Full title:\*\* (.+)$', path.read_text(encoding='utf-8'), re.M)
        if len(titles) > 1 or (titles and titles[0] != row.get('title')):
            out.append(f'{rel}: Full title must equal the current manuscript title in {MANIFEST}')
    return out


def main():
    problems = checksum_findings() + index_findings() + alias_findings() + readme_title_findings()
    for problem in problems:
        print(problem)
    # The alias count shows that the alias check, which extends the record check, has run.
    aliases = sum(len(row['aliases']) for row in json.loads((ROOT / MANIFEST).read_text(encoding='utf-8'))['documents']
                  if isinstance(row.get('aliases'), list))
    print(json.dumps({'checksum_record': SUMS, 'index': INDEX, 'aliases': aliases, 'problems': len(problems)}))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
