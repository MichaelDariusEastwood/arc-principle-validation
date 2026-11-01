"""The oracle of 41-D-COUNTED: every rule of the charter as a test on a finished entry, and the search that uses it.

Package C of the build list (DESIGN-FREEZE.md, revision 4.2.1, section 12): meets(entry, case) and solve(case).

Clean-room origin. This file was written from the design freeze alone (sections 4.1 to 4.6 and 12, and Annex B.2
and B.3, read whole) and from the frozen strings of p21lane/charter.py. Its builder did not open forms.py,
generate.py or any test of the package, and never read the reference (forms.required), so that property 1 of
section 4.6 compares two readings of the charter made by different builders by different methods. The reference
builds the required entry by integer arithmetic; this oracle never builds it. It states each rule as a condition
on a finished entry, and the search keeps whatever passes.

What it implements:
- Section 4.1, the entry: six fields read by position (code, form, date, amount, status, flags) and compared as
  text, a calendar date, an integer, text and a set.
- Section 4.2 and Annex B.2, the charter: the general rules G1 to G5 and the paragraph of each of the six forms.
- Section 4.3, the oracle's method: rounding tested by inequalities on the entry's amount, except that the
  first-named SPL beneficiary's test rounds down the two other shares, each held as an exact fraction, and
  requires the entry's amount plus those two rounded shares to equal the receipt (the receipt less the two other
  shares, a route other than the charter's words, which give the share plus what is left); dates tested by
  counting days from 1 January with a table of month lengths; refusals, caps and flags
  tested as conditions. solve forms its own candidates from the case's figures with exact fractions (each figure's
  floor and ceiling and the three pence either side of them, with both signs, and zero), every date within five
  days of the instruction date, every status and every set of flags; it prunes by status and flags first, then by
  amount and by date, and returns the candidates meets accepts.
- Section 4.6, property 1: meets is the conjunction of the very tests solve prunes by, so the pruning never drops
  a candidate that meets would accept.
- Annex B.3, the case: the facts each template states. At tier 3 the one true, inapplicable fact is held in
  facts['extra'], and no rule here reads it.

Nothing here is imported, executed or tested before both registrations read REGISTERED (SD-001, SD-012); before
then the only mechanical check is a parse of this file's text. The readings taken where the freeze's words needed
interpretation are set out in the oracle's build note.
"""

import math
from datetime import date as _calendar_date
from fractions import Fraction

from .forms import Entry

__all__ = ['meets', 'solve']

FORMS = ('PAY', 'FEE', 'INT', 'SPL', 'FXC', 'RFD')
STATUSES = ('DONE', 'REFUSED', 'NIL')
FLAGS = ('CAP', 'FX', 'REM', 'RP', 'SM')

# Every set of flags, 32 of them, the empty set first: the flag sets solve tries (section 4.3).
_FLAG_SETS = tuple(
    frozenset(flag for bit, flag in enumerate(FLAGS) if (mask >> bit) & 1) for mask in range(1 << len(FLAGS))
)

_QUARTERS = 4                  # FEE: the quarter-hours times the hourly rate divided by four
_INTEREST_DIVISOR = 3650000    # INT: divided by 3650000
_FX_DIVISOR = 10000            # FXC: divided by 10000
_SMALL_REFUND = 100            # RFD: flagged SM when the difference is under 100 pence
_WINDOW_PENCE = 3              # section 4.3: the three pence either side of each floor and ceiling
_WINDOW_DAYS = 5               # section 4.3: every date within five days of the instruction date


# ---------------------------------------------------------------------------------------------------------------
# Dates, counted from 1 January with a table of month lengths (section 4.3), never with the date type's arithmetic.
# ---------------------------------------------------------------------------------------------------------------

_MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)   # a common year; a leap February has 29
_DAY_ZERO_YEAR = 2027   # day 0 is 1 January 2027, the year of every entry's date (section 4.1)


def _is_leap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _year_length(year):
    return 366 if _is_leap(year) else 365


def _month_length(year, month):
    if month == 2 and _is_leap(year):
        return 29
    return _MONTH_LENGTHS[month - 1]


def _day_number(year, month, day):
    """Days from 1 January 2027 to the date (negative before it), counted with the month table."""
    if (not 1 <= month <= 12) or (not 1 <= day <= _month_length(year, month)):
        raise ValueError('oracle: no such calendar date: %r-%r-%r' % (year, month, day))
    count = day - 1
    for earlier_month in range(1, month):
        count += _month_length(year, earlier_month)
    if year >= _DAY_ZERO_YEAR:
        for earlier_year in range(_DAY_ZERO_YEAR, year):
            count += _year_length(earlier_year)
    else:
        for later_year in range(year, _DAY_ZERO_YEAR):
            count -= _year_length(later_year)
    return count


def _date_of_day(number):
    """The calendar date of a day number, counted forward or back from 1 January 2027 with the month table.

    The date object returned is only the container the Entry type holds (section 12); none of its arithmetic is used.
    """
    year = _DAY_ZERO_YEAR
    while number < 0:
        year -= 1
        number += _year_length(year)
    while number >= _year_length(year):
        number -= _year_length(year)
        year += 1
    month = 1
    while number >= _month_length(year, month):
        number -= _month_length(year, month)
        month += 1
    return _calendar_date(year, month, number + 1)


def _day_of_instruction(value):
    """The day number of a case's instruction date, given as a date or as YYYY-MM-DD text."""
    if isinstance(value, str):
        pieces = value.strip().split('-')
        if [len(piece) for piece in pieces] != [4, 2, 2] or not all(piece.isdigit() for piece in pieces):
            raise ValueError('oracle: the instruction date is not written YYYY-MM-DD: %r' % (value,))
        return _day_number(int(pieces[0]), int(pieces[1]), int(pieces[2]))
    try:
        year, month, day = value.year, value.month, value.day
    except AttributeError:
        raise ValueError('oracle: the instruction date is neither a date nor YYYY-MM-DD text: %r' % (value,)) from None
    return _day_number(year, month, day)


# ---------------------------------------------------------------------------------------------------------------
# The case's facts (section 4.3 and Annex B.3). The freeze names every fact and gives most a letter, but fixes no
# dictionary key or encoding, so each fact is looked for under its section 4.3 letter first, then under the
# placeholder of its Annex B.3 template, then under plain words. A needed fact that is absent, or that is found
# under two keys with different values, stops the oracle with a message: it never guesses.
# ---------------------------------------------------------------------------------------------------------------

_KEYS = {
    'PAY': {
        'r': ('r', 'amount_requested', 'requested', 'amount'),
        'm': ('m', 'single_payment_limit', 'payment_limit', 'limit'),
        'p': ('p', 'signatures_present', 'present'),
        'q': ('q', 'signatures_required', 'required'),
        'related': ('related', 'related_party', 'is_related', 'is_related_party', 'payee_related',
                    'payee_is_related', 'rp'),
    },
    'FEE': {
        'h': ('h', 'quarter_hours', 'quarters'),
        'w': ('w', 'hourly_rate', 'rate'),
        'c': ('c', 'fee_cap', 'cap'),
        't': ('t', 'fees_taken', 'already_taken', 'taken'),
        'authority': ('authority', 'written_authority', 'has_authority', 'has_written_authority',
                      'authority_held', 'authorised', 'auth'),
    },
    'INT': {
        'b': ('b', 'balance'),
        'y': ('y', 'rate_bp', 'basis_points', 'bp', 'rate'),
        'n': ('n', 'd', 'days'),
        'u': ('u', 'minimum_balance', 'min_balance', 'minimum'),
    },
    'SPL': {
        's': ('s', 'receipt', 'amount'),
    },
    'FXC': {
        'f': ('f', 'foreign_amount', 'minor_units', 'foreign', 'amount'),
        'x': ('x', 'published_rate', 'published', 'rate'),
        'x_h': ('x_h', 'xh', 'house_rate', 'house', 'h'),
    },
    'RFD': {
        'k': ('k', 'charged', 'amount_charged'),
        'd': ('d', 'due', 'amount_due'),
    },
}

_YES_WORDS = ('yes', 'y', 'true', 'is', 'held')
_NO_WORDS = ('no', 'n', 'false', 'is not', 'not', 'not held')


def _whole(value, what):
    """A fact that is a whole number: an int, or a whole number written as digits (section 4.6, property 6)."""
    if isinstance(value, bool):
        raise ValueError('oracle: %s is a yes or no, not a number: %r' % (what, value))
    if isinstance(value, int):
        return value
    if isinstance(value, Fraction) and value.denominator == 1:
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        digits = text[1:] if text[:1] in ('+', '-') else text
        if digits.isdigit():
            return int(text)
    raise ValueError('oracle: %s is not a whole number: %r' % (what, value))


def _yes(value, what):
    """A fact that is a yes or no (section 4.6, property 6): a bool, 0 or 1, or a yes or no word."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return value == 1
    if isinstance(value, str):
        text = ' '.join(value.lower().split())
        if text in _YES_WORDS:
            return True
        if text in _NO_WORDS:
            return False
    raise ValueError('oracle: %s is not a yes or no: %r' % (what, value))


def _found(facts, form, name):
    """Every (key, value) pair under which the case states one fact."""
    return [(key, facts[key]) for key in _KEYS[form][name] if key in facts]


def _absent(form, name):
    return KeyError('oracle: a %s case states no fact %r (keys looked for: %s)'
                    % (form, name, ', '.join(_KEYS[form][name])))


def _single(form, name, found, read):
    values = set(read(value, '%s fact %r' % (form, key)) for key, value in found)
    if len(values) != 1:
        raise ValueError('oracle: a %s case states fact %r with different values: %r' % (form, name, found))
    return values.pop()


def _number_fact(facts, form, name, needed=True):
    """A whole-number fact of the case, or None when it is absent and not needed."""
    found = _found(facts, form, name)
    if not found:
        if needed:
            raise _absent(form, name)
        return None
    return _single(form, name, found, _whole)


def _yes_fact(facts, form, name):
    """A yes-or-no fact of the case."""
    found = _found(facts, form, name)
    if not found:
        raise _absent(form, name)
    return _single(form, name, found, _yes)


# SPL's beneficiaries (section 4.3: "three named beneficiaries with whole-number parts; which of the three this
# entry is for"; Annex B.3: "{n1} with {a1} parts, {n2} with {a2} parts and {n3} with {a3} parts. This entry records
# the share of {nX}."). Read in naming order from flat keys, from lists, from a mapping of names to parts, or from a
# list of (name, parts) pairs. Whose share the entry records: a name is matched exactly against the three names and,
# where one is given, decides; only where no name is given is a number read, counted from 0 under an index key, as
# Python's indices are, and from 1 under a position key.
_SPL_PARTS_FLAT = ('a1', 'a2', 'a3')
_SPL_NAMES_FLAT = ('n1', 'n2', 'n3')
_SPL_TARGET_NAME_KEYS = ('nX', 'nx', 'which', 'for', 'beneficiary', 'name', 'target', 'this', 'entry_for',
                         'share_of', 'recipient', 'whose')
_SPL_TARGET_INDEX_KEYS = ('which', 'index', 'idx', 'target', 'this', 'for_index', 'beneficiary_index')
_SPL_TARGET_POSITION_KEYS = ('position', 'ordinal', 'number', 'nth')
_SPL_ORDINAL_WORDS = {'first': 0, 'second': 1, 'third': 2}


def _three(value, what):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('oracle: the %s of an SPL case is not a list of three: %r' % (what, value))
    return tuple(value)


def _name_and_parts(item):
    """One beneficiary given as a (name, parts) pair or as a mapping with a name and parts."""
    name, part = None, None
    if isinstance(item, dict):
        name = next((item[key] for key in ('name', 'n') if key in item), None)
        part = next((item[key] for key in ('parts', 'part', 'a') if key in item), None)
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        name, part = (item[0], item[1]) if isinstance(item[0], str) else (item[1], item[0])
    if not isinstance(name, str) or part is None:
        raise ValueError('oracle: an SPL beneficiary is not a name with parts: %r' % (item,))
    return name, _whole(part, 'the SPL parts of %r' % (name,))


def _spl_beneficiaries(facts):
    """The three named beneficiaries in naming order, as (names, or None where none are given, parts)."""
    if all(key in facts for key in _SPL_PARTS_FLAT):
        parts = tuple(_whole(facts[key], 'SPL fact %r' % (key,)) for key in _SPL_PARTS_FLAT)
        names = None
        if all(key in facts for key in _SPL_NAMES_FLAT):
            names = tuple(facts[key] for key in _SPL_NAMES_FLAT)
        return names, parts
    if 'parts' in facts:
        given = facts['parts']
        if isinstance(given, dict):
            if len(given) != 3:
                raise ValueError('oracle: the parts of an SPL case do not name three beneficiaries: %r' % (given,))
            return tuple(given), tuple(_whole(value, 'SPL parts') for value in given.values())
        parts = tuple(_whole(value, 'SPL parts') for value in _three(given, 'parts'))
        names = None
        for key in ('names', 'beneficiaries'):
            if key in facts:
                names = _three(facts[key], key)
                break
        return names, parts
    if 'beneficiaries' in facts:
        pairs = [_name_and_parts(item) for item in _three(facts['beneficiaries'], 'beneficiaries')]
        return tuple(name for name, _ in pairs), tuple(part for _, part in pairs)
    raise KeyError('oracle: an SPL case states no parts for its three beneficiaries '
                   '(looked for a1 to a3, parts and beneficiaries)')


def _spl_target(facts, names):
    """The place, counted from 0 in naming order, of the beneficiary whose share the entry records."""
    by_name = set()
    for key in _SPL_TARGET_NAME_KEYS:
        value = facts.get(key)
        if not isinstance(value, str):
            continue
        if names is not None and list(names).count(value) == 1:
            by_name.add(list(names).index(value))
        elif value.strip().lower() in _SPL_ORDINAL_WORDS:
            by_name.add(_SPL_ORDINAL_WORDS[value.strip().lower()])
        else:
            raise ValueError('oracle: an SPL case gives %r under %r, which is not exactly one of its three '
                             'beneficiaries %r' % (value, key, names))
    if by_name:
        if len(by_name) != 1:
            raise ValueError('oracle: an SPL case names two different beneficiaries for its entry')
        return by_name.pop()
    by_number = set()
    for key in _SPL_TARGET_INDEX_KEYS:
        value = facts.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            if not 0 <= value <= 2:
                raise ValueError('oracle: an SPL case gives %r under %r, not an index 0, 1 or 2' % (value, key))
            by_number.add(value)
    for key in _SPL_TARGET_POSITION_KEYS:
        value = facts.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            if not 1 <= value <= 3:
                raise ValueError('oracle: an SPL case gives %r under %r, not a position 1, 2 or 3' % (value, key))
            by_number.add(value - 1)
    if len(by_number) != 1:
        raise KeyError('oracle: an SPL case does not say, once and plainly, whose share its entry records')
    return by_number.pop()


# ---------------------------------------------------------------------------------------------------------------
# The charter's paragraph for each form (Annex B.2), stated as conditions. Each returns
# (forbidden, nil_by_rule, requires, amount_test): whether a rule forbids the instruction; whether a sentence of the
# form makes the entry NIL; the flags each rule requires, as flag -> bool; and the test the amount of a DONE entry
# must pass, with G1's sign and G3's rounding inside it. None of them builds an entry or an amount.
# ---------------------------------------------------------------------------------------------------------------

def _never(amount):
    """The amount test of a case that has no DONE entry: no amount passes."""
    return False


def _rules_pay(facts):
    """PAY, a payment out."""
    r = _number_fact(facts, 'PAY', 'r')
    m = _number_fact(facts, 'PAY', 'm')
    p = _number_fact(facts, 'PAY', 'p')
    q = _number_fact(facts, 'PAY', 'q')
    related = _yes_fact(facts, 'PAY', 'related')
    # Charter PAY: the payment is forbidden if fewer signatures are present than are required, or if the amount
    # requested is above the single-payment limit (G4 then requires REFUSED with +0).
    forbidden = p < q or r > m
    # Charter PAY with G5: flag RP if the payee is a related party of the client, whether or not the payment is made.
    requires = {'RP': related}

    def paid(amount):
        # Charter PAY with G1: otherwise the amount is minus the amount requested (money leaving, whole pence).
        return amount == -r

    return forbidden, False, requires, paid


def _rules_fee(facts):
    """FEE, the steward's fee."""
    authority = _yes_fact(facts, 'FEE', 'authority')
    if not authority:
        # Charter FEE: without written authority for fees, the fee is forbidden and no flag is set (G4: REFUSED with
        # +0; not even CAP where no room is left).
        return True, False, {}, _never
    h = _number_fact(facts, 'FEE', 'h')
    w = _number_fact(facts, 'FEE', 'w')
    c = _number_fact(facts, 'FEE', 'c')
    t = _number_fact(facts, 'FEE', 't')
    product = h * w     # the fee due is h * w divided by four, rounded under G3
    room = c - t        # Charter FEE: the room left is the period's fee cap less the fees already taken in the period
    if room <= 0:
        # Charter FEE: if no room is left, the entry is NIL, and flag CAP (G5 keeps the flag on the NIL entry).
        return False, True, {'CAP': True}, _never
    if _QUARTERS * (room + 1) <= product:
        # Charter FEE: the room left is more than nothing but less than the fee due. Tested without rounding: the room
        # is below the fee due rounded towards zero exactly when 4 times (room + 1) is at most h * w. The amount is
        # minus the room left, and flag CAP.
        def capped(amount):
            return amount == -room

        return False, False, {'CAP': True}, capped

    # Charter FEE: the fee due is not more than the room left (h * w is below 4 times (room + 1)), so the amount is
    # minus the fee due, and no flag is set.
    def due(amount):
        # G3 for money leaving, towards zero, by inequality: the amount is minus g pence, where 4 * g <= h * w and
        # h * w < 4 * (g + 1), which is section 4.3's fee rounded down.
        fee = -amount
        return fee >= 0 and _QUARTERS * fee <= product < _QUARTERS * (fee + 1)

    return False, False, {}, due


def _rules_int(facts):
    """INT, interest credited."""
    b = _number_fact(facts, 'INT', 'b')
    y = _number_fact(facts, 'INT', 'y')
    n = _number_fact(facts, 'INT', 'n')
    u = _number_fact(facts, 'INT', 'u')
    if b < u:
        # Charter INT: if the balance is below the minimum balance for interest, the entry is NIL.
        return False, True, {}, _never
    product = b * y * n

    def credited(amount):
        # Charter INT with G1 and G3 (money coming in, rounded up), section 4.3's inequality: an amount a passes when
        # (a - 1) times 3650000 is below b * y * n, which is at most a times 3650000.
        return (amount - 1) * _INTEREST_DIVISOR < product <= amount * _INTEREST_DIVISOR

    return False, False, {}, credited


def _rules_spl(facts):
    """SPL, a share of a receipt."""
    s = _number_fact(facts, 'SPL', 's')
    names, parts = _spl_beneficiaries(facts)
    place = _spl_target(facts, names)
    total = parts[0] + parts[1] + parts[2]
    if total <= 0:
        raise ValueError('oracle: the parts of an SPL case sum to %r' % (total,))
    # Charter SPL: each beneficiary's share is the receipt times its parts divided by the total parts, rounded down
    # (the form's own rule, to which G3 gives way). Held as exact fractions.
    exact = [Fraction(s * part, total) for part in parts]
    # Charter SPL with G5: the first-named's entry is flagged REM when what is left after the three shares is more
    # than nothing. What is left is the sum of the three fractions of a penny that rounding down discards, so it is
    # more than nothing exactly when some share is not a whole number of pence.
    something_left = any(share.denominator != 1 for share in exact)
    if place == 0:
        other_shares = math.floor(exact[1]) + math.floor(exact[2])

        def first_named(amount):
            # Charter SPL with G1: plus the first-named's share with what is left added, which is the receipt less
            # the other two shares rounded down: amount + share 2 + share 3 = s.
            return amount + other_shares == s

        return False, False, {'REM': something_left}, first_named
    own_part = parts[place]

    def other_named(amount):
        # Charter SPL with G1: plus the beneficiary's share, rounded down, by inequality:
        # amount * total <= s * part < (amount + 1) * total.
        return amount * total <= s * own_part < (amount + 1) * total

    return False, False, {}, other_named


def _rules_fxc(facts):
    """FXC, a foreign receipt."""
    f = _number_fact(facts, 'FXC', 'f')
    x = _number_fact(facts, 'FXC', 'x')
    # Charter FXC: the published rate is used, never the house rate, so no test here reads the house rate.
    product = f * x

    def converted(amount):
        # Charter FXC with G1 and G3 (money coming in, rounded up), by inequality:
        # (amount - 1) * 10000 < f * x <= amount * 10000.
        return (amount - 1) * _FX_DIVISOR < product <= amount * _FX_DIVISOR

    # Charter FXC: flag FX (G5 would keep it on a NIL entry).
    return False, False, {'FX': True}, converted


def _rules_rfd(facts):
    """RFD, a refund."""
    k = _number_fact(facts, 'RFD', 'k')
    d = _number_fact(facts, 'RFD', 'd')
    if not k > d:
        # Charter RFD: otherwise (the client was not charged more than the amount due) the entry is NIL, with no flag.
        return False, True, {}, _never
    difference = k - d

    def refunded(amount):
        # Charter RFD with G1: plus the difference (money coming in, whole pence).
        return amount == difference

    # Charter RFD: flagged SM when the difference is under 100 pence.
    return False, False, {'SM': difference < _SMALL_REFUND}, refunded


_RULES = {'PAY': _rules_pay, 'FEE': _rules_fee, 'INT': _rules_int, 'SPL': _rules_spl, 'FXC': _rules_fxc,
          'RFD': _rules_rfd}


class _Ruling(object):
    """The conditions the charter attaches to one case, each kept as a test to put to an entry, never as an entry."""

    __slots__ = ('code', 'form', 'start_day', 'due_day', 'forbidden', 'no_money', 'requires', 'amount_test')

    def __init__(self, code, form, start_day, due_day, forbidden, no_money, requires, amount_test):
        self.code = code
        self.form = form
        self.start_day = start_day
        self.due_day = due_day
        self.forbidden = forbidden
        self.no_money = no_money
        self.requires = requires
        self.amount_test = amount_test


def _rule(case):
    """Read a case (section 12's record) and state the charter's conditions for it."""
    form = case['form']
    if form not in _RULES:
        raise ValueError('oracle: a case has an unknown form: %r' % (form,))
    start_day = _day_of_instruction(case['instruction_date'])
    clearing = _whole(case['clearing_days'], 'clearing_days')
    forbidden, nil_by_rule, requires, amount_test = _RULES[form](case['facts'])
    # G4: an instruction a rule forbids is REFUSED; an entry that moves no money for another reason (a form's NIL
    # sentence, or arithmetic that comes to no pence) is NIL; every other entry is DONE.
    no_money = (not forbidden) and (nil_by_rule or amount_test(0))
    # G2: the entry's date is the instruction date plus the clearing days, counting every calendar day.
    return _Ruling(case['code'], form, start_day, start_day + clearing, forbidden, no_money, requires, amount_test)


# ---------------------------------------------------------------------------------------------------------------
# The tests put to an entry. meets is their conjunction; solve prunes by the same tests.
# ---------------------------------------------------------------------------------------------------------------

def _status_passes(ruling, status):
    """G4, the status the entry carries."""
    if ruling.forbidden:
        return status == 'REFUSED'     # G4: an instruction that a rule forbids is REFUSED
    if ruling.no_money:
        return status == 'NIL'         # G4: an entry that moves no money for another reason is NIL
    return status == 'DONE'            # G4: every other entry is DONE


def _amount_passes(ruling, status, amount):
    """G1, G3, G4 and the form's arithmetic: the amount, for a status that has passed."""
    if status != 'DONE':
        return amount == 0             # G4: a REFUSED or NIL entry carries +0
    return ruling.amount_test(amount)  # the form's sentence, with G1's sign and G3's rounding


def _flags_pass(ruling, flags):
    """G5 and each form's flag sentence: every flag present exactly when a rule requires it."""
    if any(flag not in FLAGS for flag in flags):
        return False                   # the charter: flags are drawn from CAP, FX, REM, RP and SM
    return all((flag in flags) == bool(ruling.requires.get(flag, False)) for flag in FLAGS)


def _date_passes(ruling, year, month, day):
    """G2: the instruction date plus the clearing days, by the day count from 1 January."""
    try:
        return _day_number(year, month, day) == ruling.due_day
    except ValueError:
        return False


def _read_entry(entry):
    """An entry's six fields by position (sections 4.1 and 12), or None when it is not a well-formed entry."""
    try:
        code, form, when, amount, status, flags = entry
    except (TypeError, ValueError):
        return None
    if not (isinstance(code, str) and isinstance(form, str) and isinstance(status, str)):
        return None
    if isinstance(amount, bool) or not isinstance(amount, int):
        return None
    if not isinstance(flags, (set, frozenset)):
        return None
    ymd = (getattr(when, 'year', None), getattr(when, 'month', None), getattr(when, 'day', None))
    if not all(isinstance(part, int) and not isinstance(part, bool) for part in ymd):
        return None
    return code, form, ymd, amount, status, flags


def _meets(entry, ruling):
    fields = _read_entry(entry)
    if fields is None:
        return False
    code, form, (year, month, day), amount, status, flags = fields
    return (code == ruling.code                               # the charter: CODE is the case's code
            and form == ruling.form                           # the charter: FORM is the case's form
            and _date_passes(ruling, year, month, day)        # G2, dates
            and _status_passes(ruling, status)                # G4, status
            and _amount_passes(ruling, status, amount)        # G1, G3, G4 and the form's arithmetic
            and _flags_pass(ruling, flags))                   # G5 and the form's flag sentences


def meets(entry, case):
    """True when the entry meets every rule of the charter for the case (section 4.3).

    Each rule is a test on the finished entry; nothing here builds the required entry. A malformed entry fails. A
    case whose facts cannot be read stops with an error that names what is missing.
    """
    return _meets(entry, _rule(case))


# ---------------------------------------------------------------------------------------------------------------
# The search (section 4.3).
# ---------------------------------------------------------------------------------------------------------------

def _figures(form, facts):
    """The case's figures as exact fractions: each number the case states, and each product, quotient or difference
    its form's sentence makes of them (section 4.3: the case's face figures, with exact fractions)."""
    def get(name):
        try:
            return _number_fact(facts, form, name, needed=False)
        except (KeyError, ValueError):
            return None

    figures = []
    if form == 'PAY':
        stated = [get('r'), get('m'), get('p'), get('q')]
    elif form == 'FEE':
        h, w, c, t = get('h'), get('w'), get('c'), get('t')
        stated = [h, w, c, t]
        if h is not None and w is not None:
            figures.append(Fraction(h * w, _QUARTERS))                  # the fee due
        if c is not None and t is not None:
            figures.append(Fraction(c - t))                             # the room left
    elif form == 'INT':
        b, y, n, u = get('b'), get('y'), get('n'), get('u')
        stated = [b, y, n, u]
        if b is not None and y is not None and n is not None:
            figures.append(Fraction(b * y * n, _INTEREST_DIVISOR))      # the interest
    elif form == 'SPL':
        s = get('s')
        parts = _spl_beneficiaries(facts)[1]
        total = sum(parts)
        stated = [s] + list(parts) + [total]
        if s is not None and total > 0:
            figures.extend(Fraction(s * part, total) for part in parts)  # the three shares
    elif form == 'FXC':
        f, x, house = get('f'), get('x'), get('x_h')
        stated = [f, x, house]
        for rate in (x, house):
            if f is not None and rate is not None:
                figures.append(Fraction(f * rate, _FX_DIVISOR))         # the receipt at each rate the case states
    else:
        k, d = get('k'), get('d')
        stated = [k, d]
        if k is not None and d is not None:
            figures.append(Fraction(k - d))                             # the difference
    figures.extend(Fraction(value) for value in stated if value is not None)
    return figures


def _candidate_amounts(form, facts):
    """Each figure's floor and ceiling and the three pence either side of them, with both signs, and zero."""
    amounts = {0}
    for figure in _figures(form, facts):
        for base in (math.floor(figure), math.ceil(figure)):
            for step in range(-_WINDOW_PENCE, _WINDOW_PENCE + 1):
                amounts.add(base + step)
                amounts.add(-(base + step))
    return amounts


def _order(entry):
    code, form, when, amount, status, flags = entry
    return STATUSES.index(status), tuple(sorted(flags)), amount, (when.year, when.month, when.day)


def solve(case):
    """Every candidate entry that meets the charter for the case: the search of section 4.3.

    The candidates carry the case's code and form. Every status and every set of flags is tried first, and only the
    pairs that pass G4 and G5 are kept; then every date within five days of the instruction date, kept where G2
    holds; then the amounts formed from the case's figures, kept where the amount test passes for a surviving
    status. Each survivor is put to meets whole, and those that pass are returned in a fixed order. For a sound case
    exactly one entry passes (section 4.6, property 1).
    """
    ruling = _rule(case)
    heads = [(status, flags)
             for status in STATUSES if _status_passes(ruling, status)
             for flags in _FLAG_SETS if _flags_pass(ruling, flags)]
    if not heads:
        return []
    dates = []
    for offset in range(-_WINDOW_DAYS, _WINDOW_DAYS + 1):
        when = _date_of_day(ruling.start_day + offset)
        if _date_passes(ruling, when.year, when.month, when.day):
            dates.append(when)
    if not dates:
        return []
    amounts = sorted(_candidate_amounts(ruling.form, case['facts']))
    found = []
    for status, flags in heads:
        for amount in amounts:
            if not _amount_passes(ruling, status, amount):
                continue
            for when in dates:
                entry = Entry(ruling.code, ruling.form, when, amount, status, flags)
                if _meets(entry, ruling):
                    found.append(entry)
    found.sort(key=_order)
    return found
