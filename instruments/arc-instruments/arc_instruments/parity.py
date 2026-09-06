"""Parity between the reference kit and its copies in the deciding units.

The per-unit copies are derived mechanically from the coordination repository's kit; a repair that
reaches the kit and not the copies leaves stale engines in the units. This module hashes every Python
file under two trees and reports the files that differ, the files missing on either side, and one
verdict. A verification run supplies the two roots as arguments; it reads and never writes.

Usage: python3 -m arc_instruments.parity <reference-root> <copy-root>
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Dict, List


def file_hashes(root: str, suffixes=(".py",)) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", ".pytest_cache")]
        for f in filenames:
            if f.endswith(suffixes):
                p = os.path.join(dirpath, f)
                with open(p, "rb") as fh:
                    out[os.path.relpath(p, root)] = hashlib.sha256(fh.read()).hexdigest()
    return out


def compare_trees(reference: str, copy: str) -> Dict[str, object]:
    a, b = file_hashes(reference), file_hashes(copy)
    differ: List[str] = sorted(k for k in a if k in b and a[k] != b[k])
    missing_in_copy: List[str] = sorted(k for k in a if k not in b)
    extra_in_copy: List[str] = sorted(k for k in b if k not in a)
    return {"files_compared": len(set(a) & set(b)), "differ": differ, "missing_in_copy": missing_in_copy,
            "extra_in_copy": extra_in_copy, "parity": not (differ or missing_in_copy)}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: python3 -m arc_instruments.parity <reference-root> <copy-root>")
        return 2
    r = compare_trees(argv[0], argv[1])
    for k in ("files_compared", "differ", "missing_in_copy", "extra_in_copy", "parity"):
        print("%-16s %s" % (k, r[k]))
    return 0 if r["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
