#!/usr/bin/env python3
"""anti_crank_gate.py - G-ANTI-CRANK: lint papers against the reviewer heuristics
that code a manuscript as crankery (crackpot-index patterns), context-aware.

BLOCKING: revolution-vocabulary (revolutionary, groundbreaking, unprecedented,
paradigm shift); certainty overclaims (undeniable, irrefutable, indisputable,
absolute proof, cannot be wrong); self-adjacent genius comparison (the author
likened to Einstein/Galileo/Newton/Tesla/Copernicus/Darwin); persecution
vocabulary (suppressed/establishment-blocks/they-don't-want) EXCLUDING technical
suppression (error/bias/drift/noise/thermalisation/evaluator).
ADVISORY (reported, non-blocking): exclamation marks in prose outside quotes;
runs of 3+ shouted words.

  python3 tools/anti_crank_gate.py --dir papers   |  --file X.html  |  --selftest
Exit 0 pass / 1 blocked.
"""
import re, sys, glob

REV = re.compile(r"\b(revolutionar\w*|ground-?breaking|unprecedented|paradigm[- ]shift\w*)\b", re.I)
CERT = re.compile(r"\b(undeniabl\w*|irrefutabl\w*|indisputabl\w*|absolute proof|definitively proven|cannot be wrong)\b", re.I)
GENIUS = re.compile(r"\b(like|as|than|a modern|the next|another)\s+(an? )?(Einstein|Galileo|Newton|Tesla|Copernicus|Darwin)\b|\b(I|the author|Eastwood)\b[^.]{0,80}\b(Einstein|Galileo|Tesla|Copernicus)\b", re.I)
PERSEC = re.compile(r"\bsuppress(ed|ion|es)?\b|\bestablishment (refuses|ignores|blocks)|they don'?t want|silenced by", re.I)
TECH = re.compile(r"error|bias|drift|noise|thermal|evaluator|entropy|decoherence|hallucination|ethic|alignment|instruct|reasoning|suppression[- ](condition|aware|vulnerabilit|recovery|pressure)|resist", re.I)
BANG = re.compile(r"[a-z]!(?=\s|$)")
QUOTEISH = re.compile(r"[\"“”‘’']")

def check(path):
    h = open(path, encoding="utf-8").read()
    h = re.sub(r"<!--.*?-->", "", h, flags=re.S)
    body = re.sub(r"<[^>]+>", " ", h)
    blocking, advisory = [], []
    for m in REV.finditer(body):
        blocking.append(("revolution-word", m.group(0)))
    for m in CERT.finditer(body):
        blocking.append(("certainty-overclaim", m.group(0)))
    for m in GENIUS.finditer(body):
        ctx = body[max(0,m.start()-60):m.end()+60]
        if "Newton 2" in ctx or "journal" in ctx.lower(): continue
        blocking.append(("genius-comparison", re.sub(r"\s+"," ",ctx)[:100]))
    GRIEVE = re.compile(r"\b(my work|my papers|my research|the author|blocked me|ignored (me|my)|refus\w+ to (review|publish)|silenced|censor)\b", re.I)
    for m in PERSEC.finditer(body):
        ctx = body[max(0,m.start()-80):m.end()+80]
        if not GRIEVE.search(ctx): continue
        blocking.append(("persecution", re.sub(r"\s+"," ",ctx)[:100]))
    for m in BANG.finditer(body):
        ctx = body[max(0,m.start()-70):m.end()+10]
        if QUOTEISH.search(ctx): continue
        advisory.append(("exclamation", re.sub(r"\s+"," ",ctx)[:90]))
    return blocking, advisory

def main():
    if "--selftest" in sys.argv:
        import tempfile, os
        good = "<p>The correction requirement is falsifiable. Error suppression scales.</p>"
        bad = "<p>This revolutionary result is irrefutable, like Einstein I was suppressed by the establishment refuses.</p>"
        for name, txt, expect_block in [("good", good, False), ("bad", bad, True)]:
            f = tempfile.NamedTemporaryFile("w", suffix=".html", delete=False); f.write(txt); f.close()
            b, a = check(f.name); os.unlink(f.name)
            ok = bool(b) == expect_block
            print(f"  {name}: {'PASS' if ok else 'FAIL'} ({len(b)} blocking)")
            if not ok: sys.exit(1)
        print("=== SELFTEST PASSED ==="); return
    targets = []
    if "--file" in sys.argv: targets = [sys.argv[sys.argv.index("--file")+1]]
    elif "--dir" in sys.argv: targets = sorted(glob.glob(sys.argv[sys.argv.index("--dir")+1] + "/*/*.html"))
    bad = 0
    for t in targets:
        b, a = check(t)
        if b:
            bad += 1
            print(f"BLOCKED {t}")
            for k, v in b[:6]: print(f"   -> [{k}] {v}")
        elif a:
            print(f"pass (advisory {len(a)}) {t.split('/')[-1]}")
        else:
            print(f"pass  {t.split('/')[-1]}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
