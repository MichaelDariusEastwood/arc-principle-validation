#!/usr/bin/env python3
"""Truth-gate remediation tool (T2-research, 2026-07-05).

Conservative, DRY-RUN by default. Applies the two clearest mechanical fixes toward
truth-gate-zero WITHOUT touching correct text:

  1. wrong-book-date : "Published Jan 6, 2026" (arrival mislabelled) -> "Published 2 January 2026 (author copy arrived 6 January)"
                       ONLY where not already caveated with 'arrived'/'arrival'/'2 Jan'.
  2. unqualified-78  : bare "78%" referring to the Anthropic alignment-faking figure
                       -> "78% in the RL-training condition (12% baseline)"
                       ONLY where (a) the sentence mentions alignment/faking/Anthropic AND
                       (b) it is NOT already qualified within the sentence.

Usage:
  python3 truthgate_remediate.py <root>            # dry-run: prints proposed edits, writes nothing
  python3 truthgate_remediate.py <root> --apply    # apply (run ONLY on a clean checkout)

LAW 3: never touches slugs/DOIs/filenames/SHAs; only prose. Reviewable diff before commit.
"""
import re, sys, os, glob

APPLY = '--apply' in sys.argv
roots = [a for a in sys.argv[1:] if not a.startswith('--')] or ['.']

# --- rule 1: book date ---
BOOKDATE = re.compile(r'[Pp]ublished:?\s+Jan(?:uary)?\s+6,?\s+2026')
def bookdate_ok(sentence):
    # skip if already caveated in the same sentence
    return not re.search(r'arriv|2 Jan|Jan(?:uary)? 2\b|02 Jan', sentence, re.I)

# --- rule 2: unqualified 78% ---
BARE78 = re.compile(r'(?<![\d.])78%')
def is_anthropic_78(sentence):
    return bool(re.search(r'align|faking|Anthropic|RL|compliance', sentence, re.I))
def already_qualified(sentence):
    return bool(re.search(r'RL-training|reinforcement|RL condition|12% baseline|baseline', sentence, re.I))

def sentence_around(text, pos):
    s = text.rfind('.', 0, pos); e = text.find('.', pos)
    return text[(s+1 if s>=0 else 0):(e if e>=0 else len(text))]

def process(path):
    try: t = open(path, encoding='utf-8', errors='replace').read()
    except Exception: return []
    edits = []
    # book date
    for m in BOOKDATE.finditer(t):
        sent = sentence_around(t, m.start())
        if bookdate_ok(sent):
            edits.append(('book-date', m.group(0), 'Published 2 January 2026 (author copy arrived 6 January)', m.start()))
    # 78%
    for m in BARE78.finditer(t):
        sent = sentence_around(t, m.start())
        if is_anthropic_78(sent) and not already_qualified(sent):
            edits.append(('unqualified-78', '78%', '78% in the RL-training condition (12% baseline)', m.start()))
    return edits

total = 0; files_touched = 0
for root in roots:
    for f in glob.glob(os.path.join(root, '**', '*'), recursive=True):
        if not os.path.isfile(f) or not f.endswith(('.md', '.html', '.txt')): continue
        if any(s in f for s in ('_fetched-', '/strategy/', '_superseded/', 'website-private-archive/', '.git/')): continue
        edits = process(f)
        if not edits: continue
        files_touched += 1
        rel = f.split('pm/research/')[-1] if 'pm/research/' in f else f
        print(f"\n{rel}  ({len(edits)} edit(s))")
        for kind, old, new, pos in edits[:6]:
            print(f"    [{kind}] {old!r} -> {new!r}")
        total += len(edits)
        if APPLY:
            t = open(f, encoding='utf-8', errors='replace').read()
            # apply book-date first (specific), then 78% (only Anthropic-context, sentence-checked)
            t = BOOKDATE.sub(lambda m: 'Published 2 January 2026 (author copy arrived 6 January)'
                             if bookdate_ok(sentence_around(t, m.start())) else m.group(0), t)
            def sub78(m):
                sent = sentence_around(t, m.start())
                return '78% in the RL-training condition (12% baseline)' if (is_anthropic_78(sent) and not already_qualified(sent)) else '78%'
            t = BARE78.sub(sub78, t)
            open(f, 'w', encoding='utf-8').write(t)

print(f"\n{'APPLIED' if APPLY else 'DRY-RUN'}: {total} proposed edit(s) across {files_touched} file(s).")
print("Review the diff before committing. Run without --apply first." if APPLY else "Re-run with --apply on a CLEAN checkout to write.")
