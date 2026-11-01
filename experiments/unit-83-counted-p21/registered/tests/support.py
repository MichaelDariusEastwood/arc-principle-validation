"""Shared helpers of the authored tests (41-D-COUNTED, section 12, package H). Not a test module: it holds the hand-made
cases, the builders of synthetic blocks and the stand-in clients. Not run before both registrations read REGISTERED.

Every expected entry below was worked by hand from the charter's words (Annex B.2) and checked by plain integer
arithmetic that imported nothing of the study; BUILD-DECISIONS.md (18 October 2026) records the cases.
"""
import copy
import datetime
import json
import re
import tempfile
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
D = datetime.date
NAMES3 = ['Alder Penn', 'Bryony Vale', 'Caspian Rook']


def case(code, form, date, clearing, facts, tier, case_id='claude-haiku-5-5:1:1'):
    return {'case_id': case_id, 'code': code, 'form': form, 'instruction_date': date, 'clearing_days': clearing,
            'facts': facts, 'tier': tier}


# (case, the required entry's line), one for each form at each tier.
REFERENCE = [
    (case('ABCD', 'PAY', D(2027, 1, 2), 2, {'r': 1250, 'm': 5000, 'p': 2, 'q': 2, 'payee': 'Alder Penn', 'related': True}, 1),
     'ABCD|PAY|2027-01-04|-1250|DONE|RP'),
    (case('BCDE', 'PAY', D(2027, 3, 30), 3, {'r': 123456, 'm': 100000, 'p': 2, 'q': 1, 'payee': 'Bryony Vale', 'related': False}, 2),
     'BCDE|PAY|2027-04-02|+0|REFUSED|-'),
    (case('CDEF', 'PAY', D(2027, 6, 29), 5, {'r': 98765432, 'm': 99000000, 'p': 1, 'q': 3, 'payee': 'Caspian Rook',
                                             'related': True, 'extra': {'m2': 50000000}}, 3),
     'CDEF|PAY|2027-07-04|+0|REFUSED|RP'),
    (case('DEFG', 'FEE', D(2027, 2, 10), 1, {'h': 8, 'w': 12000, 'c': 50000, 't': 10000, 'authority': True}, 1),
     'DEFG|FEE|2027-02-11|-24000|DONE|-'),
    (case('EFGH', 'FEE', D(2027, 4, 28), 4, {'h': 37, 'w': 6789, 'c': 100000, 't': 60000, 'authority': True}, 2),
     'EFGH|FEE|2027-05-02|-40000|DONE|CAP'),
    (case('FGHJ', 'FEE', D(2027, 12, 20), 3, {'h': 1234, 'w': 54321, 'c': 5000000, 't': 5000000, 'authority': True,
                                              'extra': {'w2': 20001}}, 3),
     'FGHJ|FEE|2027-12-23|+0|NIL|CAP'),
    (case('GHJK', 'INT', D(2027, 5, 10), 2, {'b': 50000, 'y': 400, 'days': 90, 'u': 10000}, 1),
     'GHJK|INT|2027-05-12|+494|DONE|-'),
    (case('HJKL', 'INT', D(2027, 8, 30), 5, {'b': 1234567, 'y': 333, 'days': 47, 'u': 2000000}, 2),
     'HJKL|INT|2027-09-04|+0|NIL|-'),
    (case('JKLM', 'INT', D(2027, 11, 28), 4, {'b': 87654321, 'y': 1234, 'days': 123, 'u': 1000000,
                                              'extra': {'date2': D(2027, 10, 1), 'y2': 999}}, 3),
     'JKLM|INT|2027-12-02|+3645027|DONE|-'),
    (case('KLMN', 'SPL', D(2027, 1, 30), 1, {'s': 1001, 'names': list(NAMES3), 'parts': [1, 2, 3], 'which': 0}, 1),
     'KLMN|SPL|2027-01-31|+168|DONE|REM'),
    (case('LMNP', 'SPL', D(2027, 2, 26), 3, {'s': 250000, 'names': list(NAMES3), 'parts': [5, 4, 3], 'which': 2}, 2),
     'LMNP|SPL|2027-03-01|+62500|DONE|-'),
    (case('MNPQ', 'SPL', D(2027, 9, 28), 2, {'s': 12345678, 'names': list(NAMES3), 'parts': [7, 9, 7], 'which': 1,
                                             'extra': {'n4': 'Delphine Marr'}}, 3),
     'MNPQ|SPL|2027-09-30|+4830917|DONE|-'),
    (case('NPQR', 'FXC', D(2027, 3, 3), 2, {'f': 50000, 'x': 12000, 'x_h': 11500}, 1),
     'NPQR|FXC|2027-03-05|+60000|DONE|FX'),
    (case('PQRS', 'FXC', D(2027, 10, 30), 3, {'f': 123456, 'x': 4567, 'x_h': 4300}, 2),
     'PQRS|FXC|2027-11-02|+56383|DONE|FX'),
    (case('QRST', 'FXC', D(2027, 7, 29), 5, {'f': 9876543, 'x': 12347, 'x_h': 11800, 'extra': {'x2': 12100}}, 3),
     'QRST|FXC|2027-08-03|+12194568|DONE|FX'),
    (case('RSTU', 'RFD', D(2027, 4, 1), 1, {'k': 5000, 'due': 4950}, 1),
     'RSTU|RFD|2027-04-02|+50|DONE|SM'),
    (case('STUV', 'RFD', D(2027, 6, 15), 4, {'k': 250000, 'due': 180000}, 2),
     'STUV|RFD|2027-06-19|+70000|DONE|-'),
    (case('TUVW', 'RFD', D(2027, 5, 31), 1, {'k': 12345678, 'due': 12345678, 'extra': {'k2': 1000000}}, 3),
     'TUVW|RFD|2027-06-01|+0|NIL|-'),
]
BY_CODE = {c['code']: c for c, _ in REFERENCE}
FEE_NO_AUTHORITY = case('UVWX', 'FEE', D(2027, 1, 5), 2, {'h': 8, 'w': 12000, 'c': 50000, 't': 10000, 'authority': False}, 1)
FEE_FITS_FRACTION = case('VWXY', 'FEE', D(2027, 1, 5), 2, {'h': 37, 'w': 6789, 'c': 100000, 't': 0, 'authority': True}, 2)
FXC_TIE = case('WXYZ', 'FXC', D(2027, 1, 5), 2, {'f': 12345, 'x': 10007, 'x_h': 10006}, 2)


def load_config(**changes):
    """The stage's registered configuration, with test-only changes (never written back)."""
    config = json.loads((STAGE / 'configs' / 'p21.json').read_text('utf-8'))
    config = copy.deepcopy(config)
    for key, value in changes.items():
        config[key] = value
    return config


def temp_root(prefix='p21-test-'):
    return Path(tempfile.mkdtemp(prefix=prefix))


def plain_block(v_in, v_out, v_t=None, faulty=None, view=None, unusable=None, arms=('C', 'M', 'T'), v_w=None):
    """A synthetic block as analysis.values() returns it, with only the fields the block's quantities read."""
    faulty = list(faulty) if faulty is not None else [k % 2 == 0 for k in range(18)]
    v_t = list(v_t) if v_t is not None else list(v_out)
    view = list(view) if view is not None else [k + 1 for k in range(18)]
    flags = {role: [False] * 18 for role in arms}
    for role, values in (unusable or {}).items():
        flags[role] = list(values)
    b = {'which': 1, 'arms': list(arms), 'faulty': faulty, 'v_in': list(v_in), 'v_out': list(v_out), 'v_T': v_t,
         'view': view, 'unusable': flags, 'v_G': [None] * 18, 'v_I': [None] * 18}
    if 'W' in arms:
        b['v_W'] = list(v_w)
    return b


ENTRY_LINE = re.compile(r'^(Case [^\n]*)\n(?:Entry|Draft entry): ([^\n]*)$', re.M)


class WorldClient:
    """A stand-in for the provider client whose replies follow a stated world (section 12, package H, tests): the
    corrector repairs a draft with chance `repair` and otherwise returns it unchanged; the generating arm writes the
    required entry; a verdict call rejects the entries it is shown when one of them departs from its case's required
    entry, catching each such entry with chance `catch` less `fall` for every entry in view beyond the first, and
    raises a false alarm on a clean view with chance `false_alarm`. Chances are read from the SHA-256 of the call's
    slot, so a replay answers alike. The first ask of every call whose slot hash falls below `unusable` returns an
    unusable reply, so that the re-ask is exercised."""

    def __init__(self, required_by_text, engines, repair=0.95, catch=0.95, fall=0.02, false_alarm=0.03, unusable=0.02):
        self.required = dict(required_by_text)
        self.engines = list(engines)
        self.repair, self.catch, self.fall, self.false_alarm, self.unusable = repair, catch, fall, false_alarm, unusable
        self.calls = 0

    @staticmethod
    def _chance(text, salt):
        import hashlib
        value = int.from_bytes(hashlib.sha256((salt + '|' + text).encode('utf-8')).digest()[:8], 'big')
        return value / float(1 << 64)

    def headroom(self, engine):
        return 1e9

    class _Ledger:
        def get(self, row_id):
            return {'reservation': 0.0, 'charge': 0.0}

    ledger = _Ledger()

    def call(self, name, engine, role, system, user):
        self.calls += 1
        base = {'status': 'ok', 'input_tokens': 100, 'output_tokens': 10, 'reasoning_tokens': 0, 'returned_model': engine,
                'response_id': 'r-' + name, 'elapsed_seconds': 0.01, 'base_slot': name, 'attempt': 1, 'request_id': name}
        if '/r' not in name and self._chance(name, 'unusable') < self.unusable:
            if role in ('C', 'G'):
                return {**base, 'text': 'I have checked the entry.'}
            return {**base, 'status': 'truncated', 'text': ''}
        pairs = ENTRY_LINE.findall(user)
        if role in ('C', 'G'):
            arriving_text = user.split('THIS ROUND\'S CHANGE.' if role == 'C' else 'NEW CASE.', 1)[1]
            case_text = [line for line in arriving_text.split('\n') if line.startswith('Case ')][0]
            required = self.required[case_text]
            if role == 'G':
                return {**base, 'text': required}
            draft = re.search(r'^Draft entry: ([^\n]*)$', user, re.M).group(1)
            text = required if self._chance(name, 'repair') < self.repair else draft
            return {**base, 'text': 'Checked.\n' + text}
        shown = len(pairs)
        departing = [text for text, line in pairs if self.required.get(text) != line]
        reject = False
        if departing:
            chance = max(0.0, self.catch - self.fall * (shown - 1))
            reject = any(self._chance(name + '|' + text, 'catch') < chance for text in departing)
        elif self._chance(name, 'alarm') < self.false_alarm:
            reject = True
        return {**base, 'text': 'All entries checked.\n' + ('REJECT' if reject else 'ACCEPT')}
