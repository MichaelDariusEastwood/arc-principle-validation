"""Planted departures (41-D-COUNTED, section 4.4): seventeen named operators and the rule that admits their entries.

Package D of the build list (DESIGN-FREEZE.md, section 12). An operator takes a case and returns the entry one named
departure would have produced, or None where the departure does not apply. `planted(case)` admits an operator's entry
when it parses, differs from the required entry, and is produced by no other operator of the same case, so that every
planted entry has one name. No prompt ever says that a departure was planted. Nothing here is executed before both
registrations read REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of this file's text.

Where the freeze names a departure in words only, the entry it produces is fixed here (BUILD-DECISIONS.md, 18 October
2026, items 12 to 16): S-AUTH, S-LIMIT, S-SIGS, S-MIN and S-HOUSE give the entry the charter would require had the
protective fact been otherwise (authority held; the limit, the signatures or the minimum met; the house rate
published), so that every other rule still applies to them; S-CAP keeps the fee due and drops the flag; S-REM keeps the
first-named beneficiary's own share and drops the flag; S-NEAR rounds another beneficiary's share up; S-KEEP gives NIL,
+0, no flag; S-PART pays 100 pence less, with the required entry's flags; S-ROUND applies only where a fraction of a
penny is rounded and the fee, interest or receipt is entered at its computed amount.
"""
import datetime

from . import forms

S = 'S'
T = 'T'

# Every operator, its kind and the forms it acts on, in the fixed order in which planted() lists them.
OPERATORS = (
    ('S-DATE', S, forms.FORMS),
    ('S-ROUND', S, ('FEE', 'INT', 'FXC')),
    ('S-LIMIT', S, ('PAY',)),
    ('S-SIGS', S, ('PAY',)),
    ('S-RP', S, ('PAY',)),
    ('S-AUTH', S, ('FEE',)),
    ('S-CAP', S, ('FEE',)),
    ('S-MIN', S, ('INT',)),
    ('S-REM', S, ('SPL',)),
    ('S-NEAR', S, ('SPL',)),
    ('S-HOUSE', S, ('FXC',)),
    ('S-FX', S, ('FXC',)),
    ('S-KEEP', S, ('RFD',)),
    ('S-PART', S, ('RFD',)),
    ('T-SIGN', T, forms.FORMS),
    ('T-DAY', T, forms.FORMS),
    ('T-DIGIT', T, forms.FORMS),
)
KIND = {name: kind for name, kind, _ in OPERATORS}

# The two registered classifications (section 4.4), fixed by hash before any data.
CLASS = {}
for _name in ('S-DATE', 'S-ROUND', 'S-CAP', 'S-REM', 'S-NEAR', 'S-HOUSE', 'S-PART'):
    CLASS[_name] = 'A'
for _name in ('S-LIMIT', 'S-SIGS', 'S-RP', 'S-AUTH', 'S-MIN', 'S-FX', 'S-KEEP'):
    CLASS[_name] = 'B'
for _name in ('T-SIGN', 'T-DAY', 'T-DIGIT'):
    CLASS[_name] = 'T'
ONE_COMPARISON = frozenset(('S-DATE', 'S-LIMIT', 'S-SIGS', 'S-RP', 'S-AUTH', 'S-MIN', 'S-FX', 'S-KEEP', 'T-SIGN'))
CATCH = {name: ('one_comparison' if name in ONE_COMPARISON else 'recomputation') for name, _, _ in OPERATORS}


def _with_facts(case, **changes):
    """A copy of the case whose facts differ only as named; used to ask the reference what another fact would give."""
    facts = dict(case['facts'])
    facts.update(changes)
    out = dict(case)
    out['facts'] = facts
    return out


def _ceil_div(numerator, denominator):
    return -((-numerator) // denominator)


def s_date(case, req):
    """The instruction date entered, clearing days ignored."""
    return req._replace(date=case['instruction_date'])


def s_round(case, req):
    """The fraction rounded against the client: a fee rounded up, interest or a foreign receipt rounded down."""
    f, form = case['facts'], case['form']
    if form == 'FEE':
        if not f['authority']:
            return None
        numerator = f['h'] * f['w']
        if numerator % 4 == 0 or numerator // 4 > f['c'] - f['t']:
            return None
        return req._replace(amount=-_ceil_div(numerator, 4))
    if form == 'INT':
        if f['b'] < f['u']:
            return None
        numerator = f['b'] * f['y'] * f['days']
        if numerator % 3650000 == 0:
            return None
        return req._replace(amount=numerator // 3650000)
    if form == 'FXC':
        numerator = f['f'] * f['x']
        if numerator % 10000 == 0:
            return None
        return req._replace(amount=numerator // 10000)
    return None


def s_limit(case, req):
    """Paid although above the limit."""
    f = case['facts']
    if not f['r'] > f['m'] or f['p'] < f['q']:
        return None
    return forms.required(_with_facts(case, m=f['r']))


def s_sigs(case, req):
    """Paid although a signature is missing."""
    f = case['facts']
    if not f['p'] < f['q'] or f['r'] > f['m']:
        return None
    return forms.required(_with_facts(case, p=f['q']))


def s_rp(case, req):
    """The related-party flag left off."""
    if not case['facts']['related']:
        return None
    return req._replace(flags=req.flags - {'RP'})


def s_auth(case, req):
    """The fee taken without authority: the entry the charter would require were written authority held."""
    if case['facts']['authority']:
        return None
    return forms.required(_with_facts(case, authority=True))


def s_cap(case, req):
    """The cap ignored and its flag left off: the whole fee due taken, no flag."""
    f = case['facts']
    if not f['authority']:
        return None
    raw = (f['h'] * f['w']) // 4
    if raw <= f['c'] - f['t']:
        return None
    return forms.entry(case['code'], 'FEE', req.date, -raw, 'DONE')


def s_min(case, req):
    """Interest credited although the balance is below the minimum."""
    f = case['facts']
    if not f['b'] < f['u']:
        return None
    return forms.required(_with_facts(case, u=f['b']))


def s_rem(case, req):
    """The left-over pence withheld from the first-named beneficiary (and its flag with them)."""
    f = case['facts']
    if f['which'] != 0:
        return None
    total = sum(f['parts'])
    share = (f['s'] * f['parts'][0]) // total
    left = f['s'] - sum((f['s'] * a) // total for a in f['parts'])
    if left <= 0:
        return None
    return forms.entry(case['code'], 'SPL', req.date, share, 'DONE')


def s_near(case, req):
    """Another beneficiary's share rounded up."""
    f = case['facts']
    if f['which'] == 0:
        return None
    numerator, total = f['s'] * f['parts'][f['which']], sum(f['parts'])
    if numerator % total == 0:
        return None
    return forms.entry(case['code'], 'SPL', req.date, _ceil_div(numerator, total), 'DONE')


def s_house(case, req):
    """The house rate used, with the flag kept."""
    return forms.required(_with_facts(case, x=case['facts']['x_h']))


def s_fx(case, req):
    """The FX flag left off."""
    return req._replace(flags=req.flags - {'FX'})


def s_keep(case, req):
    """A refund under 100 pence withheld."""
    diff = case['facts']['k'] - case['facts']['due']
    if not 0 < diff < 100:
        return None
    return forms.entry(case['code'], 'RFD', req.date, 0, 'NIL')


def s_part(case, req):
    """A larger refund paid 100 pence short."""
    diff = case['facts']['k'] - case['facts']['due']
    if diff < 100:
        return None
    return req._replace(amount=diff - 100)


def t_sign(case, req):
    """The amount's sign flipped."""
    if req.amount == 0:
        return None
    return req._replace(amount=-req.amount)


def t_day(case, req):
    """The date one day late."""
    return req._replace(date=req.date + datetime.timedelta(days=1))


def digit_exchange(amount):
    """T-DIGIT's rule (section 12, package D): the first pair of adjacent unequal digits of the amount, from the left,
    whose exchange does not put a zero first, exchanged; the sign kept. None where no such pair exists."""
    digits = list(str(abs(amount)))
    for i in range(len(digits) - 1):
        if digits[i] == digits[i + 1]:
            continue
        if i == 0 and digits[1] == '0':
            continue
        digits[i], digits[i + 1] = digits[i + 1], digits[i]
        value = int(''.join(digits))
        return -value if amount < 0 else value
    return None


def t_digit(case, req):
    """Two adjacent unequal digits of the amount exchanged."""
    value = digit_exchange(req.amount)
    if value is None:
        return None
    return req._replace(amount=value)


FUNCTIONS = {
    'S-DATE': s_date, 'S-ROUND': s_round, 'S-LIMIT': s_limit, 'S-SIGS': s_sigs, 'S-RP': s_rp, 'S-AUTH': s_auth,
    'S-CAP': s_cap, 'S-MIN': s_min, 'S-REM': s_rem, 'S-NEAR': s_near, 'S-HOUSE': s_house, 'S-FX': s_fx,
    'S-KEEP': s_keep, 'S-PART': s_part, 'T-SIGN': t_sign, 'T-DAY': t_day, 'T-DIGIT': t_digit,
}


def candidates(case):
    """Every operator's output for the case, before admission: [(name, kind, Entry or None)]."""
    req = forms.required(case)
    out = []
    for name, kind, acts_on in OPERATORS:
        out.append((name, kind, FUNCTIONS[name](case, req) if case['form'] in acts_on else None))
    return out


def _parses(entry):
    try:
        line = forms.line(entry)
    except (AttributeError, TypeError, ValueError):
        return False
    return forms.parse(line) == entry


def planted(case):
    """The admitted planted entries of the case, [(name, kind, Entry)], in the fixed order of OPERATORS: each parses,
    differs from the required entry and is produced by no other operator of the same case."""
    req = forms.required(case)
    produced = [(name, kind, e) for name, kind, e in candidates(case) if e is not None and _parses(e) and e != req]
    counts = {}
    for _, _, e in produced:
        counts[e] = counts.get(e, 0) + 1
    return [(name, kind, e) for name, kind, e in produced if counts[e] == 1]


def differing_fields(a, b):
    """The names of the fields in which two entries differ, in the order of the line."""
    return [name for name in forms.Entry._fields if getattr(a, name) != getattr(b, name)]
