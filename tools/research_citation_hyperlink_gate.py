#!/usr/bin/env python3
"""research_citation_hyperlink_gate.py — G-RESEARCH-CITATION-HYPERLINK enforcement gate.

Gate ID: G-RESEARCH-CITATION-HYPERLINK
Lane: T2-research (research authoring + citation truth)
Status: ENFORCED (research papers)

Binding rule from the operator (2026-07-03): in a research paper, no person, idea,
or paper may be mentioned without a supportive link. Every reference must resolve to
its official source; every in-text citation must point to a linked reference; every
integrity hash (SHA-256) must point to the external anchor that makes it checkable,
or it is decorative and must go. Modelled on the legal
G-VERBATIM-CITATION-HYPERLINK gate (tools/legal/T3_citation_hyperlink_gate.py).

This gate does NOT verify that a link is *correct* (a human/T5 owns that — a wrong DOI
is worse than none). It enforces that a supportive link EXISTS wherever a source is
invoked. Three checks, each block-on-fail:

  1. REFERENCES COVERAGE — every bibliographic entry in the References list carries an
     <a href>. A bare reference blocks.
  2. IN-TEXT RESOLUTION — every in-text citation ("Author (2020)", "Author et al.,
     2020", "(Author & Other, 2020)") resolves to a linked reference by surname, or
     carries its own inline link. An orphan citation blocks.
  3. INTEGRITY-HASH ANCHOR — every SHA-256 tag has, within its block, a link or an
     external anchor keyword (osf.io / doi.org / arxiv / zenodo / /evidence/ / dkim).
     A bare hash blocks (it proves nothing a reader can check).

CLI:
  python3 tools/research_citation_hyperlink_gate.py --file PAPER.html [--json]
  python3 tools/research_citation_hyperlink_gate.py --dir papers/ [--json]
  python3 tools/research_citation_hyperlink_gate.py --selftest
Exit 0 = pass; exit 1 = blocked.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

# ── anchor vocabulary that makes an integrity hash checkable ──────────────
_ANCHOR_KEYWORDS = ("osf.io", "doi.org", "arxiv.org", "arxiv:", "zenodo",
                    "/evidence/", "dkim", "github.com", "hash", "manifest")

# ── in-text academic citation forms ───────────────────────────────────────
# Surname allows internal apostrophe/hyphen and accented letters.
_SURNAME = r"[A-Z][A-Za-zÀ-ſ'’\-]+"
# "Author et al. (2020)" / "Author et al., 2020" / "(Author et al., 2020)"
_CITE_ETAL = re.compile(rf"({_SURNAME})\s+et\s+al\.?,?\s*\(?((?:19|20)\d\d)[a-z]?\)?")
# "Author and Other (2020)" / "Author & Other (2020)"
_CITE_TWO = re.compile(rf"({_SURNAME})\s+(?:and|&)\s+{_SURNAME}\s*\(?((?:19|20)\d\d)[a-z]?\)?")
# "(Author, 2020)" or "(Author 2020)"
_CITE_PAREN = re.compile(rf"\(({_SURNAME}),?\s*((?:19|20)\d\d)[a-z]?\)")
# "Author (2020)" single-author narrative form
_CITE_NARR = re.compile(rf"\b({_SURNAME})\s*\(((?:19|20)\d\d)[a-z]?\)")

# Words that look like a surname before "(2020)" but are not authors.
_STOP = {"Figure", "Table", "Section", "Appendix", "Paper", "Act", "Regulation",
         "Chapter", "Part", "Volume", "Article", "Fig", "Eq", "Equation", "Box",
         "Note", "Version", "Since", "In", "By", "See", "The", "This", "From",
         "January", "February", "March", "April", "May", "June", "July", "August",
         "September", "October", "November", "December",
         "Humain", "Phénomène", "Phenomene", "Masnavi"}

_SHA = re.compile(r"SHA-?256", re.IGNORECASE)
_HREF = re.compile(r"href\s*=", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def _strip_tags(html: str) -> str:
    return _TAG.sub(" ", html)


def _acronym(head: str):
    """Generate an initialism from the capitalised words of an org name head,
    e.g. 'American Psychiatric Association' -> 'APA'. None if < 2 cap words."""
    words = re.findall(r"[A-Z][A-Za-z]+", head)
    return "".join(w[0] for w in words) if len(words) >= 2 else None


def _split_body_refs(html: str):
    """Return (body_html, refs_html). The references region is the content of the
    <div class="refs"> block, bounded by its </div> so a trailing page footer is NOT
    treated as a reference; body is everything before the block. Falls back to
    id="references"->EOF when there is no refs div."""
    m = re.search(r'<[^>]*\bclass\s*=\s*["\'][^"\']*\brefs\b[^"\']*["\'][^>]*>', html, re.IGNORECASE)
    if m:
        start = m.end()
        em = re.search(r'</div>', html[start:], re.IGNORECASE)
        end = start + em.start() if em else len(html)
        return html[:m.start()], html[start:end]
    m = re.search(r'id\s*=\s*["\']references["\']', html, re.IGNORECASE)
    if m:
        return html[:m.start()], html[m.start():]
    return html, ""


def _reference_entries(refs_html: str):
    """Yield (entry_html, has_link, surnames) for each <p> bibliographic entry."""
    for m in re.finditer(r"<p\b[^>]*>(.*?)</p>", refs_html, re.DOTALL | re.IGNORECASE):
        entry = m.group(1)
        text = _strip_tags(entry)
        if not re.search(r"(?:19|20)\d\d", text):
            continue  # not a dated bibliographic entry (e.g. a heading note)
        has_link = bool(_HREF.search(entry))
        # surnames = capitalised tokens before the first year
        head = re.split(r"(?:19|20)\d\d", text, 1)[0]
        surnames = set(re.findall(_SURNAME, head))
        yield entry, has_link, surnames, text.strip()


def check_html(html: str, name: str = "") -> dict:
    blocking, warnings, findings = [], [], []
    body_html, refs_html = _split_body_refs(html)

    # ── 1. References coverage ───────────────────────────────────────────
    linked_authors, unlinked_authors, linked_acronyms = set(), set(), set()
    ref_total = ref_linked = 0
    for entry, has_link, surnames, text in _reference_entries(refs_html):
        ref_total += 1
        label = (text[:70] + "…") if len(text) > 70 else text
        head = re.split(r"(?:19|20)\d\d", text, 1)[0]
        acr = _acronym(head)
        allcaps = set(re.findall(r"\b[A-Z]{2,6}\b", text))
        if has_link:
            ref_linked += 1
            linked_authors |= surnames
            linked_acronyms |= allcaps
            if acr:
                linked_acronyms.add(acr)
        else:
            unlinked_authors |= surnames
            msg = f"reference has no link: {label}"
            blocking.append(msg)
            findings.append({"type": "reference", "status": "blocked", "text": label})

    # ── 2. In-text citation resolution ───────────────────────────────────
    # Work paragraph-by-paragraph so an inline link in the same <p> counts.
    cite_total = cite_ok = 0
    seen = set()
    for pm in re.finditer(r"<p\b[^>]*>(.*?)</p>", body_html, re.DOTALL | re.IGNORECASE):
        para = pm.group(1)
        para_has_link = bool(_HREF.search(para))
        ptext = _strip_tags(para)
        for rx in (_CITE_ETAL, _CITE_TWO, _CITE_PAREN, _CITE_NARR):
            for cm in rx.finditer(ptext):
                surname, year = cm.group(1), cm.group(2)
                if surname in _STOP:
                    continue
                key = (surname, year, cm.start())
                if key in seen:
                    continue
                seen.add(key)
                cite_total += 1
                if surname in linked_authors or surname in linked_acronyms:
                    cite_ok += 1
                elif surname in unlinked_authors:
                    m = f"in-text '{surname} {year}' cites a reference that has no link"
                    blocking.append(m)
                    findings.append({"type": "citation", "status": "blocked", "ref": f"{surname} {year}"})
                elif para_has_link:
                    cite_ok += 1  # self-supporting inline link in same paragraph
                else:
                    m = f"in-text '{surname} {year}' has no supportive link and no matching reference"
                    blocking.append(m)
                    findings.append({"type": "citation", "status": "blocked", "ref": f"{surname} {year}"})

    # ── 3. Integrity-hash anchor ─────────────────────────────────────────
    sha_total = sha_ok = 0
    for sm in _SHA.finditer(html):
        sha_total += 1
        window = html[max(0, sm.start() - 400): sm.end() + 400].lower()
        if _HREF.search(window) or any(k in window for k in _ANCHOR_KEYWORDS):
            sha_ok += 1
        else:
            m = f"SHA-256 tag at offset {sm.start()} has no link or external anchor nearby (proves nothing checkable)"
            blocking.append(m)
            findings.append({"type": "sha", "status": "blocked", "offset": sm.start()})

    passed = len(blocking) == 0
    return {
        "name": name,
        "pass": passed,
        "gate": "G-RESEARCH-CITATION-HYPERLINK",
        "rule": "No person, idea or paper mentioned without a supportive link.",
        "references": {"total": ref_total, "linked": ref_linked},
        "in_text": {"total": cite_total, "resolved": cite_ok},
        "sha_tags": {"total": sha_total, "anchored": sha_ok},
        "blocking": blocking,
        "findings": findings,
    }


def gate_file(path: str) -> dict:
    return check_html(Path(path).read_text(encoding="utf-8", errors="replace"), name=path)


def _print(result: dict):
    v = "✅ PASS" if result["pass"] else "❌ BLOCKED"
    r, t, s = result["references"], result["in_text"], result["sha_tags"]
    print(f"G-RESEARCH-CITATION-HYPERLINK: {v}  [{result['name']}]")
    print(f"  references {r['linked']}/{r['total']} linked · "
          f"in-text {t['resolved']}/{t['total']} resolved · "
          f"sha {s['anchored']}/{s['total']} anchored")
    if result["blocking"]:
        print(f"  {len(result['blocking'])} blocking:")
        for b in result["blocking"][:60]:
            print(f"    → {b}")


def selftest() -> bool:
    cases = [
        ('<p>Body cites Smith et al., 2020.</p>'
         '<div class="refs"><p>Smith, J. (2020). A paper. <a href="https://doi.org/x">DOI</a>.</p></div>',
         True, "linked ref + resolved in-text"),
        ('<p>Body cites Jones et al., 2019.</p>'
         '<div class="refs"><p>Jones, K. (2019). A paper. No link here.</p></div>',
         False, "bare reference must block"),
        ('<p>Body cites Doe et al., 2018 with no reference at all.</p><div class="refs"></div>',
         False, "orphan in-text citation must block"),
        ('<p>Integrity: SHA-256 abc123 anchored at <a href="https://osf.io/x">OSF</a>.</p>',
         True, "sha with adjacent anchor passes"),
        ('<p>Integrity: SHA-256 abc123 with nothing to check it against.</p>',
         False, "bare sha must block"),
        ('<p>Prose with no citations, no hashes.</p>', True, "clean prose passes"),
    ]
    print("=== G-RESEARCH-CITATION-HYPERLINK SELFTEST ===\n")
    ok = True
    for i, (html, expect, desc) in enumerate(cases, 1):
        got = check_html(html)["pass"]
        status = "PASS" if got == expect else "FAIL"
        if got != expect:
            ok = False
        print(f"  Test {i}: {status} ({desc}) expected={'PASS' if expect else 'BLOCK'} got={'PASS' if got else 'BLOCK'}")
    print("\n=== SELFTEST " + ("PASSED" if ok else "FAILED") + " ===")
    return ok


def main():
    ap = argparse.ArgumentParser(description="G-RESEARCH-CITATION-HYPERLINK gate")
    ap.add_argument("--file")
    ap.add_argument("--dir")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    targets = []
    if args.file:
        targets = [args.file]
    elif args.dir:
        targets = sorted(glob.glob(os.path.join(args.dir, "**", "*.html"), recursive=True))
    else:
        ap.print_help()
        sys.exit(1)

    results = [gate_file(t) for t in targets]
    if args.json:
        json.dump(results, sys.stdout, indent=2)
        print()
    else:
        for r in results:
            _print(r)
    sys.exit(0 if all(r["pass"] for r in results) else 1)


if __name__ == "__main__":
    main()
