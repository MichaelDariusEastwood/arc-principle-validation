"""Pairs and schedules (41-D-COUNTED, section 4.7).

Package D of the build list (DESIGN-FREEZE.md, section 12). A block is eighteen rounds on one engine, one case a round,
from an empty register in every arm. A pair is two blocks with the same eighteen cases in the same order and the same
planted entry prepared for every round; the first block's nine faulty rounds are a seeded uniform draw of nine of the
eighteen, and the second block's are the other nine. For a faulty round the kind is drawn S with probability two thirds
and T with one third, and the operator uniformly among those of that kind the case admits (section 4.4). The order of a
pair's cases is drawn when its bank is made (generate.make_bank), so that a case's identifier carries its round
position; this module draws the rest. Nothing here is executed before both registrations read REGISTERED (SD-001,
SD-012); before then the only mechanical check is a parse of this file's text.
"""
from . import departures, forms, generate

SCHEDULE_VERSION = '41-D-COUNTED-v0.1/schedule/1'


def make(engine, n_pairs, bank, seed_bytes):
    """For each pair: the order of its eighteen cases (by position), the nine faulty positions of its first block, the
    kind and operator drawn for every position with the planted line they give, and the codes."""
    if bank['engine'] != engine or len(bank['pairs']) != n_pairs:
        raise ValueError('the bank is not this engine\'s, or holds another number of pairs')
    pairs = []
    for pair in bank['pairs']:
        rng = generate.Stream(seed_bytes, 'schedule|tier%d|%s|pair%d' % (bank['tier'], engine, pair['pair']))
        faulty = sorted(rng.sample(range(1, generate.ROUNDS + 1), generate.FAULTY_PER_BLOCK))
        positions = []
        for position, case in enumerate(pair['cases'], 1):
            options = departures.planted(case)
            kind = departures.S if rng.chance(2, 3) else departures.T
            name, _, entry = rng.choice([option for option in options if option[1] == kind])
            positions.append({'position': position, 'case_id': case['case_id'], 'code': case['code'], 'kind': kind,
                              'operator': name, 'class': departures.CLASS[name], 'catch': departures.CATCH[name],
                              'planted': forms.line(entry)})
        pairs.append({'pair': pair['pair'], 'faulty_block1': faulty, 'positions': positions})
    return {'version': SCHEDULE_VERSION, 'tier': bank['tier'], 'engine': engine, 'pairs': pairs}


def is_faulty(which, position, faulty_block1):
    """Whether the round at a position is faulty in block `which` (1 or 2): block 2 runs the complementary schedule."""
    if which not in (1, 2):
        raise ValueError('a pair has blocks 1 and 2')
    return (position in faulty_block1) == (which == 1)


def block_drafts(pair_schedule, cases, which):
    """The block's eighteen rounds: (case, draft Entry, faulty, operator or None) in round order. The same draft goes
    to every arm (section 5.1)."""
    out = []
    for position, case in enumerate(cases, 1):
        entry = pair_schedule['positions'][position - 1]
        if entry['case_id'] != case['case_id'] or entry['code'] != case['code']:
            raise ValueError('the schedule and the bank disagree at position %d' % position)
        faulty = is_faulty(which, position, pair_schedule['faulty_block1'])
        draft = forms.parse(entry['planted']) if faulty else forms.required(case)
        if draft is None:
            raise ValueError('a planted line does not parse at position %d' % position)
        out.append((case, draft, faulty, entry['operator'] if faulty else None))
    return out
