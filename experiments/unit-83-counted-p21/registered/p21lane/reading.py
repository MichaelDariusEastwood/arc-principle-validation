"""Reading a reply (41-D-COUNTED, section 6.1).

Package E of the build list (DESIGN-FREEZE.md, section 12). The reader decides what a reply says, never what it is
worth: it reads an entry or a verdict, or names the reason it cannot. Nothing is repaired. A truncated or refused reply
is invalid by its return record before any text is read, so the collector passes no such text here and records
TRUNCATED or REFUSED itself. Nothing here is executed before both registrations read REGISTERED (SD-001, SD-012);
before then the only mechanical check is a parse of this file's text.

Two readings the freeze leaves to the build, recorded in BUILD-DECISIONS.md for the static review:
  - an entry line is read as it stands: spaces around a field, a missing plus sign and the order of flags are the only
    leniencies (forms.parse), so an entry wrapped in inline backticks or followed by a full stop is no entry. Admission
    (section 11.3, step 4) shows whether either form occurs while the reader can still be changed;
  - the verdict is read from the last line that still holds something once markdown emphasis, surrounding punctuation
    and spaces are removed, so a closing code fence or a rule of dashes is not taken for the last line.
"""
import re
import string

from . import forms

# Why an entry reply is invalid (section 6.1). The reader returns the first three; the collector records the last two
# from the return record.
NO_ENTRY = 'no_entry'
MANY_ENTRIES = 'many_entries'
WRONG_CASE = 'wrong_case'
TRUNCATED = 'truncated'
REFUSED = 'refused'
ENTRY_REASONS = (NO_ENTRY, MANY_ENTRIES, WRONG_CASE, TRUNCATED, REFUSED)

# Why a verdict reply is unparseable. An unparseable verdict counts as ACCEPT (section 5.3) and is counted by reason.
EMPTY_REPLY = 'empty_reply'
LAST_LINE_UNREADABLE = 'last_line_unreadable'
VERDICT_REASONS = (EMPTY_REPLY, LAST_LINE_UNREADABLE)

VERDICTS = ('ACCEPT', 'REJECT')
LABELS = ('VERDICT', 'ANSWER', 'FINAL ANSWER', 'DECISION')      # the registered list of section 6.1

_EMPHASIS = re.compile(r'[*_`~]')
_SURROUNDING = string.punctuation + '\u2018\u2019\u201c\u201d\u2022\u2013\u2014' + ' \t'   # curly quotes, bullet, two dash characters
_LABELLED = re.compile(r'^(%s) ?: ?(ACCEPT|REJECT)$' % '|'.join(re.escape(label) for label in LABELS))


def read_entry(text, code, form):
    """(Entry, None) when the reply holds exactly one line that parses as an entry for the arriving case's code and
    form, anywhere in it and with any words around it; otherwise (None, reason)."""
    found = [e for e in (forms.parse(raw) for raw in (text or '').splitlines()) if e is not None]
    if not found:
        return None, NO_ENTRY
    if len(found) > 1:
        return None, MANY_ENTRIES
    entry = found[0]
    if entry.code != code or entry.form != form:
        return None, WRONG_CASE
    return entry, None


def _normal(line):
    """A line with markdown emphasis, surrounding punctuation and spaces removed, inner spaces single, in capitals."""
    return ' '.join(_EMPHASIS.sub('', line).strip(_SURROUNDING).split()).upper()


def verdict_and_reason(text):
    """('ACCEPT' or 'REJECT', None), or (None, reason) when the last line is not exactly one of the two words, alone or
    after one registered label and a colon."""
    lines = [s for s in (_normal(raw) for raw in (text or '').splitlines()) if s]
    if not lines:
        return None, EMPTY_REPLY
    last = lines[-1]
    if last in VERDICTS:
        return last, None
    labelled = _LABELLED.match(last)
    if labelled:
        return labelled.group(2), None
    return None, LAST_LINE_UNREADABLE


def read_verdict(text):
    """'ACCEPT', 'REJECT' or None (section 6.1)."""
    return verdict_and_reason(text)[0]
