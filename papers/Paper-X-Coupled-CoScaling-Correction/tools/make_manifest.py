#!/usr/bin/env python3
"""Write MANIFEST.json: SHA-256 of every tracked artefact in the Paper X tree.

Deterministic and self-excluding (it never hashes MANIFEST.json itself). Run from the
paper root: ``python3 tools/make_manifest.py``. Lets a reviewer or archive verify that the
PDF, code, and results match what was committed.
"""
import hashlib, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = (".html", ".pdf", ".py", ".md", ".json", ".txt", ".png")
EXTRA = ("Makefile", "Dockerfile", "requirements.txt")
SKIP_DIRS = {"__pycache__", ".git", ".ipynb_checkpoints", ".pytest_cache", "node_modules", ".vscode"}
SKIP_FILES = {"MANIFEST.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    entries = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn in SKIP_FILES:
                continue
            if not (fn.endswith(EXTS) or fn in EXTRA):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            entries[rel] = {"sha256": sha256(full), "bytes": os.path.getsize(full)}
    manifest = {
        "paper": "The Coupled Co-Scaling Law (Paper X)",
        "note": "SHA-256 of each artefact. Regenerate with tools/make_manifest.py.",
        "n_files": len(entries),
        "files": {k: entries[k] for k in sorted(entries)},
    }
    out = os.path.join(ROOT, "MANIFEST.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print(f"wrote {out}: {len(entries)} files hashed")


if __name__ == "__main__":
    main()
