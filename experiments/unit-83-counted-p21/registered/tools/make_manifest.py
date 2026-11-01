#!/usr/bin/env python3
"""Make the registration manifest by command (41-D-COUNTED, section 12, packages A and H).

It lists every file of the stage that the registration fixes by hash: every file of the two packages (p21base and
p21lane), the configuration, the authored tests and the tools, each with its SHA-256, and writes
registration_manifest.json in the stage. It reads files as bytes and hashes them; it imports nothing of the lane and
executes nothing of it, so it is a build tool and not a study command. It is run once, when the build is closed after
its static review and before the registration's files are deposited; the manifest's own hash then goes into the
registration, and the gate (p21base/registration_gate.py) refuses every command whose stage differs from it.

Usage, from anywhere:  python3 tools/make_manifest.py [--check]
  --check   compare the stage with the manifest that exists and print every difference; write nothing.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

STUDY_ID = '41-D-COUNTED-v0.1'
MANIFEST = 'registration_manifest.json'
FOLDERS = ('p21base', 'p21lane', 'configs', 'tests', 'tools')
SUFFIXES = ('.py', '.json')


def stage_root():
    return Path(__file__).resolve().parents[1]


def listed_files(root):
    """Every file the manifest lists: under the five folders, Python and JSON files, no cache, sorted."""
    out = []
    for folder in FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if not path.is_file() or path.suffix not in SUFFIXES:
                continue
            if '__pycache__' in path.parts or path.name.startswith('.'):
                continue
            out.append(path.relative_to(root).as_posix())
    return out


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    files = {rel: sha256(root / rel) for rel in listed_files(root)}
    if 'configs/p21.json' not in files:
        raise SystemExit('the configuration configs/p21.json is missing from the stage')
    return {'study_id': STUDY_ID, 'created_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'made_by': 'tools/make_manifest.py', 'files': files}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    root = stage_root()
    manifest = build(root)
    path = root / MANIFEST
    if args.check:
        if not path.exists():
            print('no manifest exists')
            return 1
        old = json.loads(path.read_text('utf-8'))['files']
        new = manifest['files']
        differences = sorted(set(old) ^ set(new)) + sorted(rel for rel in set(old) & set(new) if old[rel] != new[rel])
        for rel in differences:
            print('differs: ' + rel)
        print('manifest ' + ('matches the stage' if not differences else 'differs in %d files' % len(differences)))
        return 1 if differences else 0
    if path.exists():
        raise SystemExit('a manifest exists; it is never overwritten (check it with --check)')
    data = json.dumps(manifest, indent=2, ensure_ascii=False).encode('utf-8') + b'\n'
    with path.open('xb') as handle:
        handle.write(data)
    print('wrote %s, %d files, sha256 %s' % (MANIFEST, len(manifest['files']), hashlib.sha256(data).hexdigest()))
    return 0


if __name__ == '__main__':
    sys.exit(main())
