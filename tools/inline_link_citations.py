#!/usr/bin/env python3
"""inline_link_citations.py — wrap every in-text citation in an inline hyperlink,
using ONLY the source URLs already present in the paper's own References list.

No fabrication: a citation is linked only when its author surname (or an org acronym
derived from the reference) maps to a reference that itself carries a verified link.
Citations already inside an <a> are skipped; the References block is never touched.

  python3 tools/inline_link_citations.py --dry  papers/.../PAPER.html   # report only
  python3 tools/inline_link_citations.py        papers/.../PAPER.html   # rewrite in place

Handles APA narrative ("Author (2020)", "Author and Other (2020)", "Author et al.
(2020)") and parenthetical ("(Author, 2020)", "(Author et al., 2020)") forms. Multi-
work parentheticals and bracket-span styles are left for review (reported as unmatched).
"""
import re
import sys
from pathlib import Path

SUR = r"[A-Z][A-Za-zÀ-ſ'’\-]+"
STOP = {"Figure", "Table", "Section", "Appendix", "Paper", "Act", "Chapter", "Part",
        "Volume", "Article", "Fig", "Eq", "Equation", "Box", "Note", "Version", "Since",
        "In", "By", "See", "The", "This", "From", "Empirically", "Survey", "January",
        "February", "March", "April", "May", "June", "July", "August", "September",
        "October", "November", "December",
        "Jan", "Feb", "Mar", "Apr", "Jun", "Jul", "Aug", "Sep", "Sept", "Oct", "Nov", "Dec",
        "Humain", "Phenomene", "Masnavi",
        "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"}

CITE = re.compile(
    r"(?P<narr>(?P<n1>" + SUR + r")(?:\s+et\s+al\.?|\s+(?:and|&)\s+" + SUR + r")?"
    r"\s*\((?:19|20)\d\d[a-z]?\))"
    r"|"
    r"(?P<paren>\((?P<n2>" + SUR + r")(?:\s+et\s+al\.?|\s+(?:and|&)\s+" + SUR + r")?"
    r",?\s*(?:19|20)\d\d[a-z]?\))"
)


def refs_region(html):
    m = re.search(r'<[^>]*class="[^"]*\brefs\b[^"]*"[^>]*>', html)
    if not m:
        return len(html), len(html)
    start = m.end()
    em = re.search(r'</div>', html[start:])
    return m.start(), start + (em.start() if em else 0)


def build_map(refs_html):
    mp = {}
    for pm in re.finditer(r'<p\b[^>]*>(.*?)</p>', refs_html, re.S):
        entry = pm.group(1)
        hrefs = re.findall(r'href="([^"]+)"', entry)
        if not hrefs:
            continue
        url = hrefs[0]
        text = re.sub(r'<[^>]+>', ' ', entry)
        head = re.split(r'(?:19|20)\d\d', text, 1)[0]
        for s in re.findall(SUR, head):
            mp.setdefault(s, url)
        words = re.findall(r"[A-Z][A-Za-z]+", head)
        if len(words) >= 2:
            mp.setdefault("".join(w[0] for w in words), url)
        for ac in re.findall(r"\b[A-Z]{2,6}\b", text):
            mp.setdefault(ac, url)
    return mp


def wrap_text(text, mp, stats):
    out, last = [], 0
    for m in CITE.finditer(text):
        name = m.group('n1') or m.group('n2')
        full = m.group(0)
        out.append(text[last:m.start()])
        if name and name not in STOP and name in mp:
            out.append('<a href="%s">%s</a>' % (mp[name], full))
            stats["linked"] += 1
        else:
            out.append(full)
            if name and name not in STOP:
                stats["unmatched"].add(name)
        last = m.end()
    out.append(text[last:])
    return ''.join(out)


def process_body(body, mp, stats):
    out, pos, depth = [], 0, 0
    for tm in re.finditer(r'<[^>]+>', body):
        seg = body[pos:tm.start()]
        out.append(wrap_text(seg, mp, stats) if depth == 0 else seg)
        tag = tm.group(0)
        out.append(tag)
        low = tag.lower()
        if re.match(r'<a[\s>]', low):
            depth += 1
        elif low.startswith('</a'):
            depth = max(0, depth - 1)
        pos = tm.end()
    tail = body[pos:]
    out.append(wrap_text(tail, mp, stats) if depth == 0 else tail)
    return ''.join(out)


def main():
    dry = '--dry' in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    for path in paths:
        html = Path(path).read_text(encoding='utf-8')
        b0, r1 = refs_region(html)
        mp = build_map(html[b0:r1])
        stats = {"linked": 0, "unmatched": set()}
        new_body = process_body(html[:b0], mp, stats)
        new_html = new_body + html[b0:]
        print("%s: linked %d | unmatched %s"
              % (Path(path).name, stats["linked"], sorted(stats["unmatched"])[:25]))
        if not dry and new_html != html:
            Path(path).write_text(new_html, encoding='utf-8')
            print("  written.")


if __name__ == '__main__':
    main()
