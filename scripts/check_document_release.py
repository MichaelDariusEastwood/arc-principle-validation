#!/usr/bin/env python3
"""Check document representations without importing or running study code.

Discovery, 8 October 2026: eden_capabilities_index searched 4,265 items in 10 kinds;
no exact match, 990 near misses. Inspected generate-research-pdfs.ts, especially
writeExportHtml: it produces print HTML but does not guard this repository's releases.
This check consumes those exports and verifies the published document manifest.
Discovery follow-up: eden_capabilities_index.py search covered changed-file validation;
the existing renderer was read above. No existing repository export-release gate was found.
Discovery, 20 October 2026: eden_capabilities_index.py search 'frozen registration release
profile changed-file document release check' searched 4,341 items in 10 kinds; no exact
match, 628 near misses, the nearest being this checker, which the release profiles extend.
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
# Release profiles: the representations under which a manifest row can pass. A row is checked only when its
# representation is exactly one of these names; any other value, conversion-pending among them, is refused.
# Each profile runs every check in findings(): the four required formats, the recorded hashes, the PDF
# signature, the export layout of the HTML, the citation version, the full-text floor and the fixed document
# identities. A published registration keeps its own profile, frozen-registration, because its text is frozen
# and a correction is a successor version; it is never relabelled as an ordinary paper in order to pass.
RELEASE_PROFILES = ('document-export', 'frozen-registration')


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


def registration_companion_findings(row, root):
    """Check both reader downloads against the registered full text and its date record."""
    if row.get('slug') != 'arc-theory-predictions-registration':
        return []
    prefix = 'papers/{0}/{0}.'.format(row['folder'])
    out = []
    for ext in ('md', 'json'):
        entry = row['files'].get(ext, {})
        path = root / (prefix + ext)
        if entry.get('path') != prefix + ext or not path.is_file():
            out.append('missing declared registration companion: ' + ext)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != entry.get('sha256'):
            out.append('registration companion checksum differs: ' + ext)
    if out:
        return out
    try:
        record = json.loads((root / (prefix + 'json')).read_text(encoding='utf-8'))
        markdown = (root / (prefix + 'md')).read_text(encoding='utf-8')
        official = (root / row['files']['txt']['path']).read_text(encoding='utf-8')
        fields = record['fields']
        # Immutable field IDs of OSF's Theory-Based Predictions schema, in registry order.
        keys = [f'343-{i}' for i in (2, 4, 7, 9, 12, 14, 17, 19, 21, 24, 26, 29, 31, 33)]
        if [field['key'] for field in fields] != keys:
            return ['registration field identity or order differs from the approved schema']
        registered = '\n\n'.join(field['title'] + '\n\n' + field['text'] for field in fields)
        if not official.rstrip('\n').endswith(registered):
            out.append('machine field bodies differ from the official text')
        position = 0
        for field in fields:
            body = field['text']
            start = markdown.find(body, position)
            if not body or len(body) != field['characters'] or start < 0:
                out.append('registration Markdown or length differs for ' + field['key'])
                break
            position = start + len(body)
        predictions = next(field['text'] for field in fields if field['key'] == '343-24')
        numbers = [int(n) for n in re.findall(r'^P(\d+)\s*·', predictions, re.M)]
        if (not numbers or numbers != list(range(1, len(numbers) + 1))
                or type(record.get('propositions')) is not int or record['propositions'] != len(numbers)):
            out.append('machine proposition count disagrees with the numbered Predictions field')
        version = row['version']
        latest = record['versions'][-1]
        if latest['version'] != version or latest['sha256'] != row['files']['pdf']['sha256']:
            out.append('machine version history disagrees with the release PDF')
        headline = markdown.partition('\n\n')[2].split('\n', 1)[0]
        if not re.search(r'\bRegistration v' + re.escape(version) + r'(?:[.]?(?:\s|$))', headline):
            out.append('registration Markdown introduction identifies a different edition')
    except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
        out.append('invalid registration companion: ' + type(error).__name__)
    return out


def findings(row, root=ROOT):
    if row.get('representation') not in RELEASE_PROFILES:
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
    out.extend(registration_companion_findings(row, root))
    # Required document identities are fixed in the checker, not removable row flags.
    try:
        from document_pdf_metadata import metadata_findings
        out.extend(metadata_findings(row, root))
    except Exception as exc:
        out.append('metadata verification failed: ' + type(exc).__name__ + ': ' + str(exc))
    return sorted(set(out))


def affected_documents(rows, changed):
    """README-only maintenance is not a new document edition.

    Preserve the existing conservative folder rule for every other file. A
    README explicitly named by a publication manifest still counts as an
    artefact; a filename alone cannot exempt a declared release file.
    """
    selected = set()
    for row in rows:
        declared = {item['path'] for item in row.get('files', {}).values()
                    if isinstance(item, dict) and 'path' in item}
        prefix = 'papers/' + row['folder'] + '/'
        if any(path.startswith(prefix) and
               (Path(path).name != 'README.md' or path in declared)
               for path in changed):
            selected.add(row['slug'])
    return selected


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
        selected.update(affected_documents(rows, changed))
        folders = {r['folder'] for r in rows}
        unlisted = [p for p in changed if len(p.split('/')) == 3
                    and Path(p).suffix in {'.html', '.pdf', '.txt', '.cff'}
                    and p.split('/')[1] not in folders]
    elif not selected:
        selected.update(r['slug'] for r in rows if r['representation'] in RELEASE_PROFILES)
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
                      'remaining_export_conversions': [r['slug'] for r in rows if r['representation'] not in RELEASE_PROFILES]}, indent=2))
    return bool(errors)


if __name__ == '__main__':
    sys.exit(main())
