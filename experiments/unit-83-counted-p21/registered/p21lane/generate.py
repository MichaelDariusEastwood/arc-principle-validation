"""The entry generator (41-D-COUNTED, sections 4.5 to 4.8): cases, banks and the public sets, drawn from a seed.

Package D of the build list (DESIGN-FREEZE.md, section 12). The generator's exact ranges are the constants below, fixed
by hash in the registration (section 4.5). Every draw comes from `Stream`, a counter of SHA-256 blocks keyed by the
seed and a label, so that anyone can rebuild the bank from the registered seed and the registry's public record with no
dependence on any library's random-number generator. Nothing here is executed before both registrations read
REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of this file's text.

How the ranges serve the registered properties (section 4.6). Each case first draws its pattern, which of its form's
protective rules bind, from a table of integer weights for its form and tier; its numbers are then drawn, and drawn again,
until they realise that pattern, so that the share of each pattern is the table's, whatever the numbers. Property 2 (the
naive transcription right in at most 40 per cent of each form's cases in each tier) and property 3 (at least two S
operators in at least 70 per cent) are then sums of independent draws with registered expectations, and the tables put
each well clear of its limit (BUILD-DECISIONS.md, 18 October 2026, item 9, gives each expectation and its margin).
Rejection sampling until the case admits a planted entry of each kind gives property 3's first part case by case.

The facts of a case, by form (the oracle's builder may read this list; it never reads forms.required):
  PAY  r (amount requested), m (single-payment limit), p (signatures present), q (signatures required),
       payee (a name from NAMES), related (True or False)
  FEE  h (quarter-hours), w (hourly rate in pence), c (the period's fee cap), t (fees already taken; t <= c),
       authority (True or False)
  INT  b (balance), y (rate in basis points a year), days, u (minimum balance for interest)
  SPL  s (receipt), names (three names from NAMES, in the order named), parts (three whole numbers), which (0, 1 or 2)
  FXC  f (foreign amount in minor units), x (published rate, pence for 10000), x_h (house rate, lower than x)
  RFD  k (amount charged), due (amount due)
At tier 3, facts['extra'] holds the one true, inapplicable fact: PAY m2; FEE w2; INT date2 and y2; SPL n4; FXC x2;
RFD k2. A case is {'case_id', 'code', 'form', 'instruction_date', 'clearing_days', 'facts', 'tier'}; case_id is the
engine's name (its model identifier), the pair index and the round position, joined by colons.
"""
import datetime
import hashlib

from p21base.common import GateError

from . import departures, forms

GENERATOR_VERSION = '41-D-COUNTED-v0.1/generate/1'
YEAR = forms.YEAR
MONTH_LENGTHS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)   # 2027 is not a leap year
TIERS = (1, 2, 3)
ROUNDS = 18
PER_FORM = 3
FAULTY_PER_BLOCK = 9
MAX_DRAWS = 20000

# Fictional names, the only free text a case holds (property 6; Annex B.3).
NAMES = (
    'Alder Penn', 'Bryony Vale', 'Caspian Rook', 'Delphine Marr', 'Ewan Thistle', 'Fenella Quay', 'Gideon Lark',
    'Hester Wren', 'Idris Fallow', 'Juno Hatch', 'Kester Bramble', 'Linnet Shaw', 'Morwen Cray', 'Nyle Ashdown',
    'Orla Tamsin', 'Piran Holt', 'Quillon Reede', 'Rosalind Hythe', 'Silas Coombe', 'Tamsin Garrow', 'Ulric Stave',
    'Verity Clough', 'Wendell Oakes', 'Xanthe Mere', 'Yorick Pell', 'Zillah Brack', 'Aldous Fenn', 'Briar Calloway',
    'Cedric Moss', 'Damaris Lowe', 'Elowen Pike', 'Florian Hale', 'Gwendolen Rye', 'Hollis Kemp', 'Isolde Varga',
    'Jory Cadman', 'Keturah Bell', 'Lorcan Steep', 'Mabyn Royce', 'Nesta Quill', 'Osric Dunmore', 'Petra Sallow',
    'Rufus Tregear', 'Saffron Lisle', 'Tobias Wane', 'Una Merrow', 'Vashti Coyle', 'Wystan Pryor',
)

# ---- The registered ranges (section 4.5), as numbers. A chance is (numerator, denominator); weights are out of 1000.
AMOUNT_DIGITS = {1: (3, 5), 2: (5, 7), 3: (7, 9)}           # digits of pence in a case's amount
CROSS_MONTH = {1: (0, 3), 2: (1, 3), 3: (2, 3)}             # clearing across a month end
NEAR = {1: (0, 2), 2: (1, 2), 3: (1, 2)}                     # a limit, cap or minimum near its amount
NEAR_PERMILLE = {1: 0, 2: 200, 3: 20}                        # near: within 20 per cent (tier 2), 2 per cent (tier 3)
FAR_BELOW_PERMILLE = (300, 700)                              # far, below the amount
FAR_ABOVE_PERMILLE = (1500, 3000)                            # far, above the amount
FEE_HOURS = {1: (4, 40, 4), 2: (8, 800, 1), 3: (400, 4000, 1)}   # quarter-hours: least, most, step (tier 1 whole hours)
FEE_RATE = {1: (2000, 25000), 2: (5001, 60000), 3: (20001, 99999)}   # hourly rate in pence (tier 1 whole pounds)
INT_RATE = {1: (25, 800), 2: (10, 950), 3: (101, 1999)}     # basis points a year (tier 1 multiples of 25)
INT_DAYS_ROUND = (30, 60, 90, 120, 180, 365)                 # tier 1 days; tier 2 any other day count up to 365
FX_RATE = {1: (5000, 20000), 2: (3001, 30000), 3: (1001, 99999)}   # pence for 10000 minor units (tier 1 multiples of 100)
HOUSE_PERMILLE = (900, 990)                                  # the house rate, below the published one
PARTS_TOTAL = {1: (3, 6), 2: (6, 12), 3: (11, 23)}          # the beneficiaries' parts, summed

# Each form's patterns: which protective rules bind, with integer weights out of 1000 (BUILD-DECISIONS.md, item 9).
PAY_PATTERNS = {   # (refusal: none, limit or sigs; payee related)
    1: ((('none', False), 200), (('limit', False), 250), (('sigs', False), 200), (('none', True), 250),
        (('limit', True), 50), (('sigs', True), 50)),
    2: ((('none', False), 200), (('limit', False), 200), (('sigs', False), 150), (('none', True), 200),
        (('limit', True), 150), (('sigs', True), 100)),
    3: ((('none', False), 150), (('limit', False), 175), (('sigs', False), 75), (('none', True), 100),
        (('limit', True), 300), (('sigs', True), 200)),
}
FEE_PATTERNS = {   # (written authority; the cap: fits, partial, nil or any; quarter-hours times rate modulo 4: 0, 1, 23 or any)
    1: (((False, 'any', 'any'), 350), ((True, 'fits', '0'), 200), ((True, 'partial', 'any'), 300),
        ((True, 'nil', 'any'), 150)),
    2: (((False, 'any', 'any'), 250), ((True, 'partial', 'any'), 250), ((True, 'nil', 'any'), 100),
        ((True, 'fits', '23'), 250), ((True, 'fits', '1'), 75), ((True, 'fits', '0'), 75)),
    3: (((False, 'any', 'any'), 200), ((True, 'partial', 'any'), 250), ((True, 'nil', 'any'), 100),
        ((True, 'fits', '23'), 300), ((True, 'fits', '1'), 75), ((True, 'fits', '0'), 75)),
}
INT_PATTERNS = {   # (the minimum binds; the fraction of a penny: low below a half, high at or above it, or any)
    1: (((True, 'any'), 350), ((False, 'low'), 390), ((False, 'high'), 260)),
    2: (((True, 'any'), 300), ((False, 'low'), 420), ((False, 'high'), 280)),
    3: (((True, 'any'), 300), ((False, 'low'), 420), ((False, 'high'), 280)),
}
_SPL = ((('first', 'left'), 280), (('first', 'exact'), 50), (('other', 'high'), 470), (('other', 'low'), 170),
        (('other', 'exact'), 30))
SPL_PATTERNS = {1: _SPL, 2: _SPL, 3: _SPL}   # (the first-named or another; what is left, or the share's fraction)
_RFD = (('nil', 250), ('small', 400), ('large', 350))
RFD_PATTERNS = {1: _RFD, 2: _RFD, 3: _RFD}   # nil, a refund under 100 pence, a refund of 100 pence or more (section 4.5)
PATTERNS = {'PAY': PAY_PATTERNS, 'FEE': FEE_PATTERNS, 'INT': INT_PATTERNS, 'SPL': SPL_PATTERNS, 'RFD': RFD_PATTERNS}


class PropertyFailure(GateError):
    """A registered property of section 4.6 failed; `number` names it."""
    def __init__(self, number, message):
        super().__init__('property %d: %s' % (number, message))
        self.number = number


class Stream:
    """A deterministic stream of draws: SHA-256 of the key and a counter, read as unsigned integers, every integer draw
    uniform by rejection. The key is the SHA-256 of the seed bytes, a vertical bar and the label."""
    def __init__(self, seed, label):
        if not isinstance(seed, (bytes, bytearray)) or not isinstance(label, str):
            raise TypeError('a stream needs seed bytes and a text label')
        self._key = hashlib.sha256(bytes(seed) + b'|' + label.encode('utf-8')).digest()
        self._counter = 0
        self._buffer = b''

    def _bytes(self, n):
        while len(self._buffer) < n:
            self._buffer += hashlib.sha256(self._key + self._counter.to_bytes(8, 'big')).digest()
            self._counter += 1
        out, self._buffer = self._buffer[:n], self._buffer[n:]
        return out

    def below(self, n):
        """A uniform integer from 0 to n - 1."""
        if not isinstance(n, int) or n <= 0:
            raise ValueError('below() needs a positive integer')
        if n == 1:
            return 0
        bits = (n - 1).bit_length()
        size = (bits + 7) // 8
        while True:
            value = int.from_bytes(self._bytes(size), 'big') >> (size * 8 - bits)
            if value < n:
                return value

    def between(self, low, high):
        """A uniform integer from low to high, both included."""
        if high < low:
            raise ValueError('an empty range')
        return low + self.below(high - low + 1)

    def chance(self, numerator, denominator):
        return self.below(denominator) < numerator

    def choice(self, items):
        items = list(items)
        return items[self.below(len(items))]

    def shuffle(self, items):
        for i in range(len(items) - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]

    def sample(self, items, k):
        pool = list(items)
        if not 0 <= k <= len(pool):
            raise ValueError('a sample larger than its population')
        for i in range(k):
            j = i + self.below(len(pool) - i)
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]

    def weighted(self, table):
        total = sum(weight for _, weight in table)
        point = self.below(total)
        for value, weight in table:
            if point < weight:
                return value
            point -= weight
        raise AssertionError('unreachable')


# ---- Drawing the numbers of a case.

def _amount(rng, tier):
    low, high = AMOUNT_DIGITS[tier]
    digits = rng.between(low, high)
    return rng.between(10 ** (digits - 1), 10 ** digits - 1)


def digits_ok(value, tier):
    low, high = AMOUNT_DIGITS[tier]
    return 10 ** (low - 1) <= abs(value) <= 10 ** high - 1


def _scaled(rng, base, low_permille, high_permille):
    low = -((-base * low_permille) // 1000)
    high = (base * high_permille) // 1000
    return rng.between(low, high) if low <= high else None


def _beside(rng, tier, amount, binding, near, binds_below):
    """A limit, a cap's room or a minimum beside an amount. A limit or a room binds when it lies below the amount
    (binds_below), a minimum when it lies above it. Near: within the tier's window on the side the pattern asks; far:
    30 to 70 per cent of the amount below it, or 150 to 300 per cent above it."""
    below = binding == binds_below
    window = NEAR_PERMILLE[tier] if near else 0
    if window:
        if below:
            # A binding limit or room lies strictly below the amount; a minimum that does not bind may equal it.
            low = -((-amount * (1000 - window)) // 1000)
            high = amount - 1 if binding else amount
        else:
            # A limit or room that does not bind may equal the amount; a binding minimum lies strictly above it.
            low = amount + 1 if binding else amount
            high = (amount * (1000 + window)) // 1000
        return rng.between(low, high) if low <= high else None
    return _scaled(rng, amount, *(FAR_BELOW_PERMILLE if below else FAR_ABOVE_PERMILLE))


def _figure(rng, tier, low, high, step):
    """A rate or a count: at tier 1 round (a multiple of step), at tier 2 not round (no multiple of step), at tier 3
    awkward (no multiple of 5)."""
    if tier == 1:
        return step * rng.between(-((-low) // step), high // step)
    while True:
        value = rng.between(low, high)
        if (tier == 2 and value % step) or (tier == 3 and value % 5):
            return value


def _days(rng, tier):
    if tier == 1:
        return rng.choice(INT_DAYS_ROUND)
    while True:
        value = rng.between(1, 365)
        if (tier == 2 and value not in INT_DAYS_ROUND) or (tier == 3 and value % 5):
            return value


def _parts(rng, total):
    """Three whole numbers of at least 1 summing to total, uniform over the compositions."""
    cuts = sorted(rng.sample(range(1, total), 2))
    return [cuts[0], cuts[1] - cuts[0], total - cuts[1]]


def _dates(rng, cross):
    """The instruction date and the clearing days. Across a month end the date plus the clearing days falls in the
    next month; otherwise in the same one. Every entry date, and that date plus one day, stays in 2027."""
    clearing = rng.between(1, 5)
    if cross:
        month = rng.between(1, 11)
        length = MONTH_LENGTHS[month - 1]
        day = rng.between(length - clearing + 1, length)
    else:
        month = rng.between(1, 12)
        length = MONTH_LENGTHS[month - 1]
        day = rng.between(1, length - clearing - (1 if month == 12 else 0))
    return datetime.date(YEAR, month, day), clearing


def _code(rng, taken):
    while True:
        code = ''.join(rng.choice(forms.CODE_ALPHABET) for _ in range(4))
        if code not in taken:
            return code


def _pay(rng, tier, near, pattern, instruction):
    refusal, related = pattern
    r = _amount(rng, tier)
    m = _beside(rng, tier, r, refusal == 'limit', near, True)
    if m is None or m < 1:
        return None
    q = rng.between(1, 3)
    p = rng.between(0, q - 1) if refusal == 'sigs' else rng.between(q, 3)
    if (r > m) != (refusal == 'limit') or (p < q) != (refusal == 'sigs'):
        return None
    facts = {'r': r, 'm': m, 'p': p, 'q': q, 'payee': rng.choice(NAMES), 'related': related}
    if tier == 3:
        m2 = _scaled(rng, r, 1200, 2000) if r > m else _scaled(rng, r, 500, 900)
        if m2 is None or m2 < 1 or m2 == m:
            return None
        facts['extra'] = {'m2': m2}
    return facts


def _fee(rng, tier, near, pattern, instruction):
    authority, cap, residue = pattern
    low, high, step = FEE_HOURS[tier]
    h = step * rng.between(-((-low) // step), high // step)
    w = _figure(rng, tier, FEE_RATE[tier][0], FEE_RATE[tier][1], 100)
    raw = (h * w) // 4
    if raw < 1 or not digits_ok(raw, tier):
        return None
    left = (h * w) % 4
    if (residue == '0' and left != 0) or (residue == '1' and left != 1) or (residue == '23' and left not in (2, 3)):
        return None
    state = cap if cap != 'any' else rng.choice(('fits', 'partial', 'nil'))
    if state == 'nil':
        room = 0
        t = rng.between(max(1, raw // 2), 2 * raw)
    else:
        room = _beside(rng, tier, raw, state == 'partial', near, True)
        if room is None or (state == 'partial' and not 0 < room < raw) or (state == 'fits' and room < raw):
            return None
        t = rng.between(0, 2 * raw)
    facts = {'h': h, 'w': w, 'c': room + t, 't': t, 'authority': authority}
    if tier == 3:
        w2 = _scaled(rng, w, 300, 700)
        if w2 is None or w2 < 1:
            return None
        if w2 % 5 == 0:
            w2 += 1
        if w2 == w:
            return None
        facts['extra'] = {'w2': w2}
    return facts


def _int(rng, tier, near, pattern, instruction):
    binds, fraction = pattern
    b = _amount(rng, tier)
    y = _figure(rng, tier, INT_RATE[tier][0], INT_RATE[tier][1], 25)
    days = _days(rng, tier)
    u = _beside(rng, tier, b, binds, near, False)
    if u is None or u < 1 or (b < u) != binds:
        return None
    rest = (b * y * days) % 3650000
    if fraction == 'low' and not 0 < 2 * rest < 3650000:
        return None
    if fraction == 'high' and not 2 * rest >= 3650000:
        return None
    facts = {'b': b, 'y': y, 'days': days, 'u': u}
    if tier == 3:
        y2 = _scaled(rng, y, 500, 1500)
        if y2 is None or y2 < 1 or y2 == y:
            return None
        facts['extra'] = {'date2': instruction - datetime.timedelta(days=rng.between(14, 120)), 'y2': y2}
    return facts


def _spl(rng, tier, near, pattern, instruction):
    group, kind = pattern
    s = _amount(rng, tier)
    total = rng.between(*PARTS_TOTAL[tier])
    parts = _parts(rng, total)
    which = 0 if group == 'first' else rng.between(1, 2)
    left = s - sum((s * a) // total for a in parts)
    if group == 'first':
        if (kind == 'left') != (left > 0):
            return None
    else:
        rest = (s * parts[which]) % total
        if (kind == 'exact' and rest != 0) or (kind == 'low' and not 0 < 2 * rest < total) or (kind == 'high' and not 2 * rest >= total):
            return None
    names = rng.sample(NAMES, 4 if tier == 3 else 3)
    facts = {'s': s, 'names': names[:3], 'parts': parts, 'which': which}
    if tier == 3:
        facts['extra'] = {'n4': names[3]}
    return facts


def _fxc(rng, tier, near, pattern, instruction):
    x = _figure(rng, tier, FX_RATE[tier][0], FX_RATE[tier][1], 100)
    f = (_amount(rng, tier) * 10000) // x
    if f < 1:
        return None
    amount = -((-f * x) // 10000)
    if not digits_ok(amount, tier):
        return None
    x_h = (x * rng.between(*HOUSE_PERMILLE)) // 1000
    if not 1 <= x_h < x or -((-f * x_h) // 10000) == amount:
        return None
    facts = {'f': f, 'x': x, 'x_h': x_h}
    if tier == 3:
        x2 = _scaled(rng, x, 950, 1050)
        if x2 is None or x2 < 1 or x2 == x:
            return None
        facts['extra'] = {'x2': x2}
    return facts


def _rfd(rng, tier, near, pattern, instruction):
    k = _amount(rng, tier)
    if pattern == 'nil':
        if rng.chance(1, 10):
            due = k
        else:
            due = k + rng.between(max(1, k * 10 // 1000), max(1, k * 500 // 1000))
    elif pattern == 'small':
        due = k - rng.between(1, 99)
    else:
        diff = _scaled(rng, k, 50, 600)
        if diff is None or diff < 100 or diff > k - 1:
            return None
        due = k - diff
    if due < 1:
        return None
    facts = {'k': k, 'due': due}
    if tier == 3:
        k2 = _scaled(rng, k, 10, 200)
        if k2 is None or k2 < 1:
            return None
        facts['extra'] = {'k2': k2}
    return facts


DRAWERS = {'PAY': _pay, 'FEE': _fee, 'INT': _int, 'SPL': _spl, 'FXC': _fxc, 'RFD': _rfd}


def draw_case(rng, engine, pair, position, form, tier, taken=frozenset()):
    """One case of the tier's registered ranges (section 4.5). Its code is drawn first, unique within the pair (taken
    holds the codes already drawn), then whether its clearing crosses a month end, whether its limit lies near its
    amount, and its pattern; its numbers are drawn again until they realise the pattern and the case admits a planted
    entry of each kind (property 3, case by case)."""
    if form not in forms.FORMS or tier not in TIERS:
        raise ValueError('unknown form or tier')
    code = _code(rng, taken)
    cross = rng.chance(*CROSS_MONTH[tier])
    near = rng.chance(*NEAR[tier])
    pattern = rng.weighted(PATTERNS[form][tier]) if form in PATTERNS else None
    for _ in range(MAX_DRAWS):
        instruction, clearing = _dates(rng, cross)
        facts = DRAWERS[form](rng, tier, near, pattern, instruction)
        if facts is None:
            continue
        case = {'case_id': '%s:%d:%d' % (engine, pair, position), 'code': code, 'form': form,
                'instruction_date': instruction, 'clearing_days': clearing, 'facts': facts, 'tier': tier}
        try:
            forms.required(case)
        except ValueError:
            continue
        kinds = {kind for _, kind, _ in departures.planted(case)}
        if departures.S in kinds and departures.T in kinds:
            return case
    raise GateError('generator: no %s case at tier %d realised pattern %r within %d draws' % (form, tier, pattern, MAX_DRAWS))


def make_bank(seed_bytes, tier, engine, n_pairs):
    """The bank of one engine at one tier: for each pair index, its eighteen fresh cases, three of each form, in the
    pair's own seeded uniform order (section 4.7), each case's position its round."""
    pairs = []
    for pair in range(1, n_pairs + 1):
        rng = Stream(seed_bytes, 'bank|tier%d|%s|pair%d' % (tier, engine, pair))
        order = list(forms.FORMS) * PER_FORM
        rng.shuffle(order)
        taken, cases = set(), []
        for position, form in enumerate(order, 1):
            case = draw_case(rng, engine, pair, position, form, tier, frozenset(taken))
            taken.add(case['code'])
            cases.append(case)
        pairs.append({'pair': pair, 'cases': cases})
    return {'version': GENERATOR_VERSION, 'tier': tier, 'engine': engine, 'n_pairs': n_pairs, 'pairs': pairs}


# ---- The registered properties (section 4.6).

def _fact_ok(name, value):
    if isinstance(value, bool) or isinstance(value, datetime.date):
        return True
    if isinstance(value, int):
        return value >= 0
    if isinstance(value, str):
        return value in NAMES
    if isinstance(value, list):
        return all(_fact_ok(name, item) for item in value) and len(value) == 3
    return False


def _case_properties(case, oracle):
    """Properties 1, 3 (its case-by-case part), 4, 5 (its dates) and 6 for one case; returns the S operator count."""
    req = forms.required(case)
    if oracle is not None:
        if not oracle.meets(req, case):
            raise PropertyFailure(1, 'the reference entry fails oracle.meets for ' + case['case_id'])
        solved = oracle.solve(case)
        if list(solved) != [req]:
            raise PropertyFailure(1, 'oracle.solve does not return the reference entry alone for ' + case['case_id'])
    admitted = departures.planted(case)
    kinds = [kind for _, kind, _ in admitted]
    if departures.S not in kinds or departures.T not in kinds:
        raise PropertyFailure(3, 'a case admits no planted entry of one kind: ' + case['case_id'])
    entries = [e for _, _, e in admitted]
    if req in entries or len(set(entries)) != len(entries):
        raise PropertyFailure(4, 'a planted entry equals the required entry or another planted entry: ' + case['case_id'])
    for e in [req] + entries:
        if e.date.year != YEAR:
            raise PropertyFailure(5, 'a date falls outside 2027: ' + case['case_id'])
    if set(case) != {'case_id', 'code', 'form', 'instruction_date', 'clearing_days', 'facts', 'tier'}:
        raise PropertyFailure(6, 'a case holds a field outside the record: ' + case['case_id'])
    facts = dict(case['facts'])
    extra = facts.pop('extra', {})
    for name, value in list(facts.items()) + list(extra.items()):
        if not _fact_ok(name, value):
            raise PropertyFailure(6, 'a fact is not a number, a date, a yes or no, or a listed name: %s %s' % (case['case_id'], name))
    return kinds.count(departures.S)


def _load_oracle():
    try:
        from . import oracle
    except ImportError as exc:
        raise PropertyFailure(1, 'package C, the oracle (p21lane/oracle.py), is absent: it is written by a builder '
                                 'who never reads forms.required, and init cannot run without it') from exc
    return oracle


def check(bank, schedule=None, oracle=None):
    """Every property of section 4.6 on one bank (one engine at one tier), raising PropertyFailure with the
    property's number on the first failure. Properties 2 and 3 are read per form within the bank."""
    oracle = oracle if oracle is not None else _load_oracle()
    naive_right, two_s, count = {}, {}, {}
    for pair in bank['pairs']:
        codes = [case['code'] for case in pair['cases']]
        if len(set(codes)) != len(codes):
            raise PropertyFailure(5, 'codes repeat within pair %d' % pair['pair'])
        if sorted(case['form'] for case in pair['cases']) != sorted(list(forms.FORMS) * PER_FORM):
            raise PropertyFailure(6, 'pair %d does not hold three cases of each form' % pair['pair'])
        for case in pair['cases']:
            s_count = _case_properties(case, oracle)
            form = case['form']
            count[form] = count.get(form, 0) + 1
            naive_right[form] = naive_right.get(form, 0) + (forms.naive(case) == forms.required(case))
            two_s[form] = two_s.get(form, 0) + (s_count >= 2)
    if schedule is not None:
        for entry in schedule['pairs']:
            faulty = entry['faulty_block1']
            if len(faulty) != FAULTY_PER_BLOCK or len(set(faulty)) != FAULTY_PER_BLOCK or not set(faulty) <= set(range(1, ROUNDS + 1)):
                # Section 4.7's rule, not a draw-dependent property of section 4.6: an implementation defect, which
                # seal.init lets through as one, never a failure that moves to the next bank of the registered sequence.
                # It is read before the draw-dependent shares, so that a defect is never reported as one of them.
                raise GateError('implementation defect: the schedule of pair %d does not hold nine faulty rounds in each '
                                'block (section 4.7)' % entry['pair'])
    for form in forms.FORMS:
        n = count.get(form, 0)
        if n and 5 * naive_right[form] > 2 * n:
            raise PropertyFailure(2, 'the naive transcription is right in more than 40 per cent of %s cases' % form)
        if n and 10 * two_s[form] < 7 * n:
            raise PropertyFailure(3, 'fewer than 70 per cent of %s cases admit two S operators' % form)
    return {'cases': sum(count.values()), 'naive_right': naive_right, 'two_s': two_s, 'by_form': count}


# ---- The public sets (section 4.8).

def _register(seed_bytes, name, size, tier, planted_positions):
    """A public register of `size` entries at one tier: forms in a seeded order, codes unique within it, and the
    drafts at planted_positions planted (kind S two times in three, the operator uniform among the case's)."""
    rng = Stream(seed_bytes, 'public|' + name)
    order = (list(forms.FORMS) * ((size + 5) // 6))[:size]
    rng.shuffle(order)
    taken, items = set(), []
    for position, form in enumerate(order, 1):
        case = draw_case(rng, 'public-' + name, 0, position, form, tier, frozenset(taken))
        taken.add(case['code'])
        items.append(_item(rng, case, position in planted_positions))
    return items


def _item(rng, case, faulty):
    if not faulty:
        return {'case': case, 'draft': forms.line(forms.required(case)), 'faulty': False, 'operator': None}
    options = departures.planted(case)
    kind = departures.S if rng.chance(2, 3) else departures.T
    name, _, entry = rng.choice([o for o in options if o[1] == kind])
    return {'case': case, 'draft': forms.line(entry), 'faulty': True, 'operator': name}


def _singles(seed_bytes, label, tier, n_faulty, n_sound):
    """Single-entry calibration items: forms in turn, faulty and sound mixed in a seeded order."""
    rng = Stream(seed_bytes, 'public|' + label)
    kinds = [True] * n_faulty + [False] * n_sound
    rng.shuffle(kinds)
    items = []
    for i, faulty in enumerate(kinds, 1):
        form = forms.FORMS[(i - 1) % len(forms.FORMS)]
        case = draw_case(rng, 'public-' + label, 0, i, form, tier)
        items.append(_item(rng, case, faulty))
    return items


def public_sets(public_seed):
    """The public sets of section 4.8, from the registered public seed alone: the three-entry register for the format
    controls (tier 1, its second draft planted); the fourteen-entry register for the gating full-size verdict controls
    (tier 3, two drafts planted at seeded positions); the eighteen-entry register for the full-size corrector and
    generating controls and the ten reported verdict controls (tier 3, its last draft planted); and for each tier a
    calibration set of 240 single entries for the monitor (120 planted) and 120 for the corrector (80 planted)."""
    seed_bytes = hashlib.sha256(public_seed.encode('utf-8')).digest()
    rng = Stream(seed_bytes, 'public|positions')
    out = {'version': GENERATOR_VERSION,
           'format3': _register(seed_bytes, 'format3', 3, 1, {2}),
           'verdict14': _register(seed_bytes, 'verdict14', 14, 3, set(rng.sample(range(1, 15), 2))),
           'register18': _register(seed_bytes, 'register18', 18, 3, {18}),
           'calibration': {}}
    for tier in TIERS:
        out['calibration'][str(tier)] = {
            'monitor': _singles(seed_bytes, 'calibration-monitor-tier%d' % tier, tier, 120, 120),
            'corrector': _singles(seed_bytes, 'calibration-corrector-tier%d' % tier, tier, 80, 40)}
    return out


def public_items(sets):
    """Every item of the public sets, for their case-by-case properties."""
    items = list(sets['format3']) + list(sets['verdict14']) + list(sets['register18'])
    for tier in sets['calibration'].values():
        items += list(tier['monitor']) + list(tier['corrector'])
    return items


def check_public(sets, oracle=None):
    """Properties 1, 3 (case by case), 4, 5 and 6 on every public case; codes unique within each register. The shares of
    properties 2 and 3 are the bank's and are not read on these small sets."""
    oracle = oracle if oracle is not None else _load_oracle()
    for name in ('format3', 'verdict14', 'register18'):
        codes = [item['case']['code'] for item in sets[name]]
        if len(set(codes)) != len(codes):
            raise PropertyFailure(5, 'codes repeat within the public register ' + name)
    for item in public_items(sets):
        _case_properties(item['case'], oracle)


# ---- Serialisation: dates as ISO text, everything else as it stands.

def case_to_json(case):
    out = dict(case)
    out['instruction_date'] = case['instruction_date'].isoformat()
    facts = dict(case['facts'])
    if 'extra' in facts:
        extra = dict(facts['extra'])
        if 'date2' in extra:
            extra['date2'] = extra['date2'].isoformat()
        facts['extra'] = extra
    out['facts'] = facts
    return out


def case_from_json(value):
    out = dict(value)
    out['instruction_date'] = datetime.date.fromisoformat(value['instruction_date'])
    facts = dict(value['facts'])
    if 'extra' in facts:
        extra = dict(facts['extra'])
        if 'date2' in extra:
            extra['date2'] = datetime.date.fromisoformat(extra['date2'])
        facts['extra'] = extra
    out['facts'] = facts
    return out


def bank_to_json(bank):
    return {**bank, 'pairs': [{'pair': p['pair'], 'cases': [case_to_json(c) for c in p['cases']],
                               'required': [forms.line(forms.required(c)) for c in p['cases']]} for p in bank['pairs']]}


def bank_from_json(value):
    return {**value, 'pairs': [{'pair': p['pair'], 'cases': [case_from_json(c) for c in p['cases']]} for p in value['pairs']]}


def public_to_json(sets):
    def items(seq):
        return [{**item, 'case': case_to_json(item['case'])} for item in seq]
    return {'version': sets['version'], 'format3': items(sets['format3']), 'verdict14': items(sets['verdict14']),
            'register18': items(sets['register18']),
            'calibration': {t: {'monitor': items(v['monitor']), 'corrector': items(v['corrector'])} for t, v in sets['calibration'].items()}}


def public_from_json(value):
    def items(seq):
        return [{**item, 'case': case_from_json(item['case'])} for item in seq]
    return {'version': value['version'], 'format3': items(value['format3']), 'verdict14': items(value['verdict14']),
            'register18': items(value['register18']),
            'calibration': {t: {'monitor': items(v['monitor']), 'corrector': items(v['corrector'])} for t, v in value['calibration'].items()}}
