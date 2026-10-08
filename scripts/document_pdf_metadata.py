#!/usr/bin/env python3
"""Canonical document PDF metadata, preserving rendered content.

The Info and XMP format is shared with the existing publication checker.
Only explicitly recorded document metadata is applied. No study code is imported.
Discovery: eden_capabilities_index searched 4,267 items across 10 kinds, no exact
match and 240 near misses. The existing website stamp-pdf-metadata.py was read;
its Info/XMP routines are reused here with canonical-source and content checks.
Discovery: eden_capabilities_index.py search 'canonical PDF metadata lost record
validation' searched 4,268 items in 10 kinds, with 234 near misses. The existing
canonical module and website stamper were inspected and reused for this repair.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import yaml
from xml.sax.saxutils import escape
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

ROOT = Path(__file__).resolve().parents[1]

XMP_NS = (
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:pdf="http://ns.adobe.com/pdf/1.3/" '
    'xmlns:xmp="http://ns.adobe.com/xap/1.0/" '
    'xmlns:xmpMM="http://ns.adobe.com/xap/1.0/mm/"'
)

def info_dict(f):
    d = {
        "/Title": f["title"],
        "/Author": ", ".join(f["authors"]),
        "/Keywords": "; ".join(f["keywords"]),
    }
    if f["subject"]:
        d["/Subject"] = f["subject"]
    return d

def xmp_packet(f):
    def alt(tag, value):
        return (f"<dc:{tag}><rdf:Alt><rdf:li xml:lang=\"x-default\">"
                f"{escape(value)}</rdf:li></rdf:Alt></dc:{tag}>")

    parts = [alt("title", f["title"])]
    parts.append("<dc:creator><rdf:Seq>"
                 + "".join(f"<rdf:li>{escape(a)}</rdf:li>" for a in f["authors"])
                 + "</rdf:Seq></dc:creator>")
    if f["subject"]:
        parts.append(alt("description", f["subject"]))
    parts.append("<dc:subject><rdf:Bag>"
                 + "".join(f"<rdf:li>{escape(k)}</rdf:li>" for k in f["keywords"])
                 + "</rdf:Bag></dc:subject>")
    if f["dates"]:
        parts.append("<dc:date><rdf:Seq>"
                     + "".join(f"<rdf:li>{escape(d)}</rdf:li>" for d in f["dates"])
                     + "</rdf:Seq></dc:date>")
    if f["doi"]:
        parts.append(f"<dc:identifier>{escape(f['doi'])}</dc:identifier>")
    if f["url"]:
        parts.append(f"<dc:source>{escape(f['url'])}</dc:source>")
    parts.append(f"<pdf:Keywords>{escape('; '.join(f['keywords']))}</pdf:Keywords>")
    if f["version"]:
        parts.append(f"<xmpMM:VersionID>{escape(f['version'])}</xmpMM:VersionID>")
    body = "".join(parts)
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        f'  <rdf:Description rdf:about="" {XMP_NS}>\n'
        f'   {body}\n'
        '  </rdf:Description>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )

def current_xmp(reader):
    meta = reader.root_object.get("/Metadata")
    if meta is None:
        return None
    try:
        return meta.get_object().get_data().decode("utf-8")
    except Exception:
        return None

def state(path, f):
    """(ok, missing) for one PDF: ok when Info and XMP already say exactly this."""
    reader = PdfReader(path)
    have = reader.metadata or {}
    missing = [k for k, v in info_dict(f).items() if (have.get(k) or "") != v]
    if current_xmp(reader) != xmp_packet(f):
        missing.append("XMP")
    return (not missing), missing


def identity_issues(row, root=ROOT):
    f = row.get('pdf_metadata')
    if not f:
        return []
    issues = []
    for key in ('title', 'version', 'doi'):
        if f.get(key) != row.get(key):
            issues.append('metadata ' + key + ' disagrees with document manifest')
    cff = yaml.safe_load((root / row['files']['cff']['path']).read_text())
    if not isinstance(cff, dict):
        return issues + ['citation is not a mapping']
    for key, expected in [('title', f['title']), ('version', f['version']),
                          ('url', f['url']), ('date-released', f['dates'][-1])]:
        if str(cff.get(key, '')) != expected:
            issues.append('metadata ' + key + ' disagrees with citation')
    authors = [' '.join(str(a.get(k, '')).strip() for k in ('given-names', 'family-names')).strip()
               if not a.get('name') else str(a['name']) for a in cff.get('authors', [])]
    if not authors or f['authors'] != authors:
        issues.append('metadata author disagrees with citation')
    preferred = cff.get('preferred-citation') or {}
    normalise_doi = lambda d: str(d).removeprefix('https://doi.org/').removeprefix('http://doi.org/').casefold()
    primary_dois = [value for value in (preferred.get('doi'), cff.get('doi')) if value]
    if not primary_dois or any(normalise_doi(d) != normalise_doi(f['doi']) for d in primary_dois):
        issues.append('metadata DOI disagrees with authoritative citation DOI')
    if not f.get('dates') or str(preferred.get('date-published', '')) != f['dates'][0]:
        issues.append('metadata original publication date disagrees with preferred citation')
    return issues


def metadata_findings(row, root=ROOT):
    if not row.get('pdf_metadata'):
        return ['repaired document has lost its required metadata record'] if row.get('metadata_revision') else []
    problems = identity_issues(row, root)
    ok, missing = state(root / row['files']['pdf']['path'], row['pdf_metadata'])
    if not ok:
        problems.append('PDF metadata missing or stale: ' + ', '.join(missing))
    return problems


def content_fingerprint(reader):
    """Digest every existing PDF object except document metadata.

    Incremental writing preserves object numbers. Include the catalogue with only
    Metadata removed; page content, fonts, images, annotations, outlines and tagged
    structure are all compared, independently of any renderer or text extractor.
    """
    from io import BytesIO
    from pypdf.generic import IndirectObject, DictionaryObject
    info = reader.trailer.raw_get('/Info') if '/Info' in reader.trailer else None
    meta = reader.root_object.raw_get('/Metadata') if '/Metadata' in reader.root_object else None
    excluded = {x.idnum for x in (info, meta) if isinstance(x, IndirectObject)}
    identifiers = {(i, gen) for gen, values in reader.xref.items()
                   for i in values if i and gen != 65535}
    identifiers.update((i, 0) for i in reader.xref_objStm)
    objects = {}
    root_id = reader.trailer.raw_get('/Root').idnum
    for number, generation in sorted(identifiers):
        if number in excluded:
            continue
        obj = reader.get_object(IndirectObject(number, generation, reader))
        if isinstance(obj, dict) and obj.get('/Type') == '/XRef':
            continue
        if number == root_id:
            obj = DictionaryObject({k: v for k, v in obj.items() if k != '/Metadata'})
        buffer = BytesIO()
        obj.write_to_stream(buffer)
        objects[f'{number}:{generation}'] = hashlib.sha256(buffer.getvalue()).hexdigest()
    return objects


def stamp(path, fields):
    """Append only metadata; refuse any change to an existing content object."""
    path = Path(path)
    if state(path, fields)[0]:
        return {'changed': False, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    original = path.read_bytes()
    before = PdfReader(path)
    before_objects = content_fingerprint(before)
    original_dates = {k: str((before.metadata or {}).get(k, ''))
                      for k in ('/CreationDate', '/ModDate', '/Creator', '/Producer')}
    writer = PdfWriter(str(path), incremental=True)
    writer.add_metadata(info_dict(fields))
    stream = DecodedStreamObject()
    stream.set_data(xmp_packet(fields).encode('utf-8'))
    stream[NameObject('/Type')] = NameObject('/Metadata')
    stream[NameObject('/Subtype')] = NameObject('/XML')
    writer._root_object[NameObject('/Metadata')] = writer._add_object(stream)
    temporary = path.with_suffix(path.suffix + '.metadata-tmp')
    try:
        writer.write(temporary)
        updated = temporary.read_bytes()
        if not updated.startswith(original):
            raise ValueError('PDF update is not append-only')
        after = PdfReader(temporary)
        after_objects = content_fingerprint(after)
        if any(after_objects.get(k) != v for k, v in before_objects.items()):
            raise ValueError('A pre-existing content object changed')
        if len(before.pages) != len(after.pages):
            raise ValueError('PDF page count changed')
        if original_dates != {k: str((after.metadata or {}).get(k, '')) for k in original_dates}:
            raise ValueError('Original render dates or producer changed')
        if not state(temporary, fields)[0]:
            raise ValueError('Written metadata did not verify')
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {'changed': True, 'previous_sha256': hashlib.sha256(original).hexdigest(),
            'sha256': hashlib.sha256(updated).hexdigest(), 'pages': len(after.pages),
            'unchanged_content_objects': len(before_objects), 'append_only': True,
            'render_metadata_unchanged': True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--slug', action='append', required=True)
    ap.add_argument('--apply', action='store_true', help='apply recorded metadata; otherwise check only')
    ap.add_argument('--receipt', type=Path)
    args = ap.parse_args()
    manifest = json.loads((ROOT / 'papers/publication-manifest.json').read_text())
    selected = [r for r in manifest['documents'] if r['slug'] in args.slug]
    if {r['slug'] for r in selected} != set(args.slug):
        ap.error('unknown document slug')
    reports = []
    for row in selected:
        if not row.get('pdf_metadata'):
            ap.error(row['slug'] + ': no approved metadata record')
        issues = identity_issues(row)
        if issues:
            ap.error(row['slug'] + ': ' + '; '.join(issues))
        report = stamp(ROOT / row['files']['pdf']['path'], row['pdf_metadata']) if args.apply else {'errors': metadata_findings(row)}
        reports.append({'slug': row['slug'], **report})
    payload = json.dumps(reports, indent=2) + '\n'
    if args.receipt:
        args.receipt.write_text(payload)
    print(payload)
    return int(any(r.get('errors') for r in reports))


if __name__ == '__main__':
    sys.exit(main())
