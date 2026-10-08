#!/usr/bin/env python3
"""Check document representations without importing or running study code.

Discovery, 8 October 2026: eden_capabilities_index searched 4,265 items in 10 kinds;
no exact match, 990 near misses. Inspected generate-research-pdfs.ts, especially
writeExportHtml: it produces print HTML but does not guard this repository's releases.
This check consumes those exports and verifies the published document manifest.
Discovery follow-up: eden_capabilities_index.py search covered changed-file validation;
the existing renderer was read above. No existing repository export-release gate was found.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class Representation(HTMLParser):
    def __init__(self):
        super().__init__()
        self.problems = []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        furniture = set(a.get('class', '').split()) & {
            'site-nav', 'site-nav-wrap', 'eden-crumb', 'lang-switch',
            'ia-research-bridge', 'site-footer'}
        if furniture or (tag == 'nav' and a.get('aria-label', '').lower() == 'primary'):
            self.problems.append('website navigation: ' + ', '.join(sorted(furniture or {tag})))
        for key in ['src', 'href']:
            if a.get(key, '').startswith(('file:', '/Users/', 'http://localhost', 'http://127.0.0.1')):
                self.problems.append('local-only asset or private path')
        if tag == 'script' and a.get('src'):
            self.problems.append('executable remote script in a document export')


def findings(row, root=ROOT):
    if row.get('representation') != 'document-export':
        return ['official document export still required']
    out = []
    for ext in ['html', 'pdf', 'txt', 'cff']:
        entry = row['files'].get(ext)
        if not entry:
            out.append('missing required format: ' + ext)
            continue
        path = root / entry['path']
        if not path.resolve().is_relative_to(root.resolve()):
            out.append('manifest path escapes repository')
            continue
        if not path.is_file():
            out.append('missing file: ' + entry['path'])
            continue
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry['sha256']:
            out.append('hash differs from manifest: ' + entry['path'])
        if ext == 'pdf' and not data.startswith(b'%PDF-'):
            out.append('PDF signature is missing')
        if ext == 'html':
            parser = Representation()
            parser.feed(data.decode('utf-8'))
            out.extend(parser.problems)
        if ext == 'cff':
            version = re.search(r'^version:\s*[\x27\"]?([^\x27\"\n]+)', data.decode('utf-8'), re.M)
            if not version or version.group(1).strip() != row['version']:
                out.append('citation version disagrees with manifest')
        if ext == 'txt' and len(data) < row.get('minimum_text_bytes', 1000):
            out.append('text companion below recorded full-text floor')
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', action='append')
    ap.add_argument('--changed-since')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    rows = json.loads((ROOT / 'papers/publication-manifest.json').read_text())['documents']
    selected = set(args.slug or [])
    unlisted = []
    if args.all:
        selected.update(r['slug'] for r in rows)
    elif args.changed_since:
        changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only',
                                           args.changed_since, 'HEAD', '--', 'papers/'], text=True).splitlines()
        for row in rows:
            if any(p.startswith('papers/' + row['folder'] + '/') for p in changed):
                selected.add(row['slug'])
        folders = {r['folder'] for r in rows}
        unlisted = [p for p in changed if len(p.split('/')) == 3
                    and Path(p).suffix in {'.html', '.pdf', '.txt', '.cff'}
                    and p.split('/')[1] not in folders]
    elif not selected:
        selected.update(r['slug'] for r in rows if r['representation'] == 'document-export')
    known = {r['slug'] for r in rows}
    errors = {s: ['unknown document'] for s in selected - known}
    if unlisted:
        errors['unlisted documents'] = unlisted
    for row in rows:
        if row['slug'] in selected:
            found = findings(row)
            if found:
                errors[row['slug']] = found
    print(json.dumps({'checked': sorted(selected), 'errors': errors,
                      'remaining_export_conversions': [r['slug'] for r in rows if r['representation'] != 'document-export']}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
