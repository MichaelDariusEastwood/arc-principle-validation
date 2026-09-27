#!/usr/bin/env python3
"""Every requirements file in this repository must be a list of Python packages that pip can read.

Why this exists: on 21 August 2026 a release sync that matched files by extension wrote Paper X's plain-text
export over papers/Paper-X-Coupled-CoScaling-Correction/requirements.txt. From then, the paper's five-minute
check (pip install -r requirements.txt && python code/test_theorems_independent.py; expected: 14 passed) stopped
at its first step for anyone who ran it, until the file was restored on 27 September 2026.

This check reads every tracked *requirements*.txt and fails on any line that is not a package requirement, a pip
option, a comment or blank. It also fails when it finds no requirements file at all, because that means its own
file list is wrong, not that the repository is clean.

Run: python .github/scripts/check_requirements.py   (exit 0 clean, 1 on any unreadable line)
"""
import subprocess
import sys
from pathlib import Path

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError:  # pip carries its own copy
    from pip._vendor.packaging.requirements import InvalidRequirement, Requirement

ROOT = Path(__file__).resolve().parents[2]
OPTIONS = ("-r", "-c", "-e", "-i", "-f", "--requirement", "--constraint", "--editable", "--index-url",
           "--extra-index-url", "--find-links", "--no-binary", "--only-binary", "--pre", "--prefer-binary",
           "--trusted-host", "--hash")


def requirement_files(root=ROOT):
    out = subprocess.run(["git", "-C", str(root), "ls-files", "*requirements*.txt"],
                         capture_output=True, text=True, check=True).stdout.split()
    return [root / f for f in out]


def unreadable_lines(path, root=ROOT):
    bad = []
    for n, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.split(" #", 1)[0].split("\t#", 1)[0].strip()
        if not line or line.startswith("#") or line.startswith(OPTIONS):
            continue
        try:
            Requirement(line)
        except InvalidRequirement:
            bad.append(f"{path.relative_to(root)}:{n}: not a package requirement: {raw[:90]!r}")
    return bad


def main():
    files = requirement_files()
    if not files:
        print("FAIL: no requirements file found; the file list is wrong, not the repository clean")
        return 1
    bad = [b for f in files for b in unreadable_lines(f)]
    print(f"requirements files: {len(files)} read")
    if bad:
        print(f"FAIL: {len(bad)} line(s) pip cannot read as requirements")
        for b in bad[:20]:
            print("  " + b)
        return 1
    print("PASS: every requirements file is a list of packages pip can read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
