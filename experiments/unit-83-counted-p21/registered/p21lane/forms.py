"""Entries, the reference, the naive transcription and the case wording (41-D-COUNTED, sections 4.1 to 4.3; Annex B.3).

Package B of the build list (DESIGN-FREEZE.md, section 12). The reference builds the required entry from a case by
integer arithmetic and the standard library's date type. The oracle (`oracle.py`, package C) is written by a builder who
never reads this file, so this file must never be consulted by it. Nothing here is executed before both registrations
read REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of this file's text.

A case is {'case_id', 'code', 'form', 'instruction_date', 'clearing_days', 'facts', 'tier'}: `instruction_date` a
datetime.date in 2027, `clearing_days` 1 to 5, `tier` 1, 2 or 3. Its facts, by form (names as generate.py draws them):

  PAY  r (amount requested), m (single-payment limit), p (signatures present), q (signatures required),
       payee (a name), related (True or False)
  FEE  h (quarter-hours), w (hourly rate in pence), c (the period's fee cap), t (fees already taken; t <= c),
       authority (True or False)
  INT  b (balance), y (rate in basis points a year), days, u (minimum balance for interest)
  SPL  s (receipt), names (three names, in the order named), parts (three whole numbers), which (0, 1 or 2)
  FXC  f (foreign amount in minor units), x (published rate, pence for 10000), x_h (house rate, lower than x)
  RFD  k (amount charged), due (amount due)

At tier 3, facts['extra'] holds the one true, inapplicable fact of the form: PAY m2; FEE w2; INT date2 and y2; SPL n4;
FXC x2; RFD k2.
"""
import collections
import datetime
from fractions import Fraction
import math
import re

FORMS = ('PAY', 'FEE', 'INT', 'SPL', 'FXC', 'RFD')
STATUSES = ('DONE', 'REFUSED', 'NIL')
FLAGS = ('CAP', 'FX', 'REM', 'RP', 'SM')
CODE_ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'
YEAR = 2027

Entry = collections.namedtuple('Entry', 'code form date amount status flags')


def entry(code, form, date, amount, status, flags=()):
    return Entry(code, form, date, int(amount), status, frozenset(flags))


def amount_text(amount):
    return '+0' if amount == 0 else ('+%d' % amount if amount > 0 else '-%d' % -amount)


def line(e):
    """The one canonical line of an entry (section 4.1)."""
    flags = ','.join(sorted(e.flags)) if e.flags else '-'
    return '|'.join((e.code, e.form, e.date.isoformat(), amount_text(e.amount), e.status, flags))


_CODE = re.compile('^[%s]{4}$' % CODE_ALPHABET)
# ASCII digits only: Python's \d also matches every other script's decimal digits, a leniency section 6.1 does not
# register.
_DATE = re.compile(r'^([0-9]{4})-([0-9]{2})-([0-9]{2})$')
_AMOUNT = re.compile(r'^[+-]?[0-9]+$')


def parse(text):
    """The inverse of line(), with the leniency of section 6.1: spaces around a field are ignored, a missing plus sign
    is read as plus, and flags are read as a set. Returns an Entry, or None when the text is not one entry."""
    fields = [f.strip() for f in text.strip().split('|')]
    if len(fields) != 6:
        return None
    code, form, date, amount, status, flags = fields
    if not _CODE.match(code) or form not in FORMS or status not in STATUSES:
        return None
    m = _DATE.match(date)
    if not m:
        return None
    try:
        d = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None
    if d.year != YEAR or not _AMOUNT.match(amount):
        return None
    if flags == '-':
        fs = frozenset()
    else:
        parts = [p.strip() for p in flags.split(',')]
        if not parts or any(p not in FLAGS for p in parts):
            return None
        fs = frozenset(parts)
    return Entry(code, form, d, int(amount), status, fs)


def entry_date(case):
    """G2: the instruction date plus the clearing days, counting every calendar day."""
    return case['instruction_date'] + datetime.timedelta(days=case['clearing_days'])


def _floor(fr):
    return math.floor(fr)


def _ceil(fr):
    return math.ceil(fr)


def required(case):
    """The reference (section 4.3): the one entry the charter requires for the case."""
    f, form, code, date = case['facts'], case['form'], case['code'], entry_date(case)
    if form == 'PAY':
        flags = ('RP',) if f['related'] else ()
        if f['p'] < f['q'] or f['r'] > f['m']:
            return entry(code, form, date, 0, 'REFUSED', flags)
        return entry(code, form, date, -f['r'], 'DONE', flags)
    if form == 'FEE':
        if not f['authority']:
            return entry(code, form, date, 0, 'REFUSED')
        raw = (f['h'] * f['w']) // 4          # money leaves: rounded towards zero, which is down for a positive fee
        room = f['c'] - f['t']
        if raw <= 0 or room < 0:
            raise ValueError('FEE case outside the generator ranges: raw %d, room %d' % (raw, room))
        if raw <= room:
            return entry(code, form, date, -raw, 'DONE')
        if room > 0:
            return entry(code, form, date, -room, 'DONE', ('CAP',))
        return entry(code, form, date, 0, 'NIL', ('CAP',))
    if form == 'INT':
        if f['b'] < f['u']:
            return entry(code, form, date, 0, 'NIL')
        amount = _ceil(Fraction(f['b'] * f['y'] * f['days'], 3650000))   # money comes in: rounded up
        if amount <= 0:
            raise ValueError('INT case outside the generator ranges: amount %d' % amount)
        return entry(code, form, date, amount, 'DONE')
    if form == 'SPL':
        total = sum(f['parts'])
        shares = [(f['s'] * a) // total for a in f['parts']]
        left = f['s'] - sum(shares)
        amount = shares[f['which']] + (left if f['which'] == 0 else 0)
        flags = ('REM',) if f['which'] == 0 and left > 0 else ()
        if amount <= 0:
            raise ValueError('SPL case outside the generator ranges: amount %d' % amount)
        return entry(code, form, date, amount, 'DONE', flags)
    if form == 'FXC':
        amount = _ceil(Fraction(f['f'] * f['x'], 10000))
        if amount <= 0:
            raise ValueError('FXC case outside the generator ranges: amount %d' % amount)
        return entry(code, form, date, amount, 'DONE', ('FX',))
    if form == 'RFD':
        diff = f['k'] - f['due']
        if diff > 0:
            return entry(code, form, date, diff, 'DONE', ('SM',) if diff < 100 else ())
        return entry(code, form, date, 0, 'NIL')
    raise ValueError('unknown form %r' % form)


def _nearest(fr):
    """To the nearest penny, a half rounded upwards (the naive transcription's declared rounding)."""
    return _floor(fr + Fraction(1, 2))


def naive(case):
    """The naive transcription (section 4.3): status DONE, the form's face amount rounded to the nearest penny with its
    natural sign (FXC at the house rate, the rate the instruction offers), the date by G2, no flags. It never consults
    a protective rule."""
    f, form, code, date = case['facts'], case['form'], case['code'], entry_date(case)
    if form == 'PAY':
        amount = -f['r']
    elif form == 'FEE':
        amount = -_nearest(Fraction(f['h'] * f['w'], 4))
    elif form == 'INT':
        amount = _nearest(Fraction(f['b'] * f['y'] * f['days'], 3650000))
    elif form == 'SPL':
        amount = _nearest(Fraction(f['s'] * f['parts'][f['which']], sum(f['parts'])))
    elif form == 'FXC':
        amount = _nearest(Fraction(f['f'] * f['x_h'], 10000))
    elif form == 'RFD':
        amount = f['k'] - f['due']
    else:
        raise ValueError('unknown form %r' % form)
    return entry(code, form, date, amount, 'DONE')


def _is(flag):
    return 'is' if flag else 'is not'


def case_text(case):
    """The fixed wording in which a case is shown (Annex B.3): one paragraph, numbers as plain digits."""
    f, form = case['facts'], case['form']
    out = 'Case %s. Form %s. Instruction date %s; clearing days %d.' % (
        case['code'], form, case['instruction_date'].isoformat(), case['clearing_days'])
    if form == 'PAY':
        out += (' A payment of %d pence is requested to %s. The single-payment limit is %d pence. Signatures present: '
                '%d; signatures required: %d. The payee %s a related party of the client.'
                % (f['r'], f['payee'], f['m'], f['p'], f['q'], _is(f['related'])))
    elif form == 'FEE':
        out += (' The steward claims a fee for %d quarter-hours at an hourly rate of %d pence. The fee cap for the period '
                'is %d pence, and fees of %d pence have already been taken in the period. Written authority for fees %s '
                'held.' % (f['h'], f['w'], f['c'], f['t'], _is(f['authority'])))
    elif form == 'INT':
        out += (' Interest is due on a balance of %d pence at %d basis points a year for %d days. The minimum balance for '
                'interest is %d pence.' % (f['b'], f['y'], f['days'], f['u']))
    elif form == 'SPL':
        n, a = f['names'], f['parts']
        out += (' A receipt of %d pence is shared among three beneficiaries, named in this order: %s with %d parts, %s '
                'with %d parts and %s with %d parts. This entry records the share of %s.'
                % (f['s'], n[0], a[0], n[1], a[1], n[2], a[2], n[f['which']]))
    elif form == 'FXC':
        out += (' A receipt of %d minor units of a foreign currency arrives. The published rate is %d pence for 10000 '
                'minor units; the house rate offered is %d pence for 10000 minor units.' % (f['f'], f['x'], f['x_h']))
    elif form == 'RFD':
        out += ' The client was charged %d pence; the amount due was %d pence.' % (f['k'], f['due'])
    else:
        raise ValueError('unknown form %r' % form)
    if case['tier'] == 3:
        x = f['extra']
        out += ' ' + {
            'PAY': lambda: "The single-payment limit on the client's deposit account, from which this payment is not "
                           "drawn, is %d pence." % x['m2'],
            'FEE': lambda: 'The hourly rate for work by a junior clerk, which this fee is not for, is %d pence.' % x['w2'],
            'INT': lambda: 'The rate before %s, which does not apply to this entry, was %d basis points a year.'
                           % (x['date2'].isoformat(), x['y2']),
            'SPL': lambda: '%s is named in the same deed but has no part in this receipt.' % x['n4'],
            'FXC': lambda: 'The published rate a week earlier, which does not apply, was %d pence for 10000 minor units.'
                           % x['x2'],
            'RFD': lambda: 'A refund of %d pence on another charge has already been entered.' % x['k2'],
        }[form]()
    return out


def shown(case, e):
    """An entry shown with its case: the case paragraph, then one line, `Entry: {line}` (Annex B.3)."""
    return case_text(case) + '\nEntry: ' + line(e)


def shown_draft(case, e):
    """The arriving case with its draft: the case paragraph, then `Draft entry: {line}` (Annex B.3, B.4)."""
    return case_text(case) + '\nDraft entry: ' + line(e)
