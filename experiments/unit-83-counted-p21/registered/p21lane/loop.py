"""The loop: the registers of section 5 for one block, round by round (41-D-COUNTED, sections 5.1 to 5.7a).

Package F of the build list (DESIGN-FREEZE.md, section 12). Round k brings the k-th scheduled case with its draft: the
required entry, or the planted one if the round is faulty in this block. The same draft goes to every arm; the round's
calls are built together and none sees another's reply; when all have returned, each arm's register is updated and
round k plus one is built from the updated registers. Nothing is reset during a block. The loop reads verdicts and the
validity of entries, never a score: whether a kept entry is sound is the scorer's (section 6.4). What the loop knows
of the schedule (which drafts were planted) builds the twin's cleaned view (section 5.4) and the flags section 6.5
prints, and nothing else. Nothing here is executed before both registrations read REGISTERED (SD-001, SD-012); before
then the only mechanical check is a parse of this file's text.

Roles: C the corrector (in-loop arm), M the monitor (outside arm), T the twin (the control arm's matched call), W the
windowed monitor (the positive control, pairs 1 to 220), G the generating arm and I the informed monitor (registered
secondaries, pairs 1 to 120). The control arm keeps every draft and makes no call of its own.
"""
from . import forms, prompts

ROLES = ('C', 'M', 'T', 'W', 'G', 'I')
ENTRY_ROLES = ('C', 'G')
VERDICT_ROLES = ('M', 'T', 'W', 'I')
ROLE_NAMES = {'C': 'corrector', 'M': 'monitor', 'T': 'twin', 'W': 'windowed monitor', 'G': 'generating arm',
              'I': 'informed monitor'}
ROUNDS = 18
STAGE = 'collect'


def arms_for(pair, generating_admitted, control_pairs, secondary_pairs):
    """The roles a pair has: the three registered arms always; the windowed monitor on the first control_pairs pairs;
    the generating arm (where admission kept it on this engine) and the informed monitor on the first secondary_pairs."""
    arms = ['C', 'M', 'T']
    if pair <= control_pairs:
        arms.append('W')
    if pair <= secondary_pairs:
        if generating_admitted:
            arms.append('G')
        arms.append('I')
    return tuple(arms)


def slot(engine, pair, which, round_number, role, stage=STAGE):
    """stage:engine:pair:block:round:role (section 12)."""
    return '%s:%s:%d:%d:%d:%s' % (stage, engine, pair, which, round_number, role)


def accepted(record):
    """Whether a verdict call lets the draft join its register: REJECT keeps it out; ACCEPT, and a reply no attempt of
    which could be read, let it in (section 5.3: an unreadable verdict counts as ACCEPT)."""
    return not (record.get('valid') and record.get('verdict') == 'REJECT')


class Block:
    """One block's registers. `rounds` is schedule.block_drafts(...): (case, draft Entry, faulty, operator) in round
    order; `arms` the roles this pair has."""

    def __init__(self, engine, pair, which, rounds, arms):
        if len(rounds) != ROUNDS:
            raise ValueError('a block has eighteen rounds')
        self.engine, self.pair, self.which, self.rounds, self.arms = engine, pair, which, rounds, tuple(arms)
        self.k = 0
        self.kept = []            # C: (case, entry) in the order kept
        self.outside = []         # M: (case, draft, faulty, admitted_on_unreadable) in the order accepted
        self.windowed = []        # W: (case, draft, faulty)
        self.informed_register = []  # I: (case, draft, faulty)
        self.written = []         # G: (case, entry)
        self.fallback_faulty = False   # the corrector's register holds a faulty draft kept on an invalid reply
        self.log = []

    def done(self):
        return self.k >= ROUNDS

    def arriving(self):
        return self.rounds[self.k]


def views(block):
    """The views of the round about to be built: the monitor's, the twin's, the window's and the informed monitor's,
    each a list of (case, entry) with the arriving draft last."""
    case, draft, faulty, _ = block.arriving()
    monitor_view = [(c, e) for c, e, _, _ in block.outside] + [(case, draft)]
    planted_codes = {c['code'] for c, _, was_faulty, _ in block.outside if was_faulty}
    twin = prompts.twin_view(monitor_view, planted_codes, forms.required)
    window = [(c, e) for c, e, _ in block.windowed] + [(case, draft)]
    informed = [(c, e) for c, e, _ in block.informed_register] + [(case, draft)]
    return monitor_view, twin, window, informed, planted_codes


def calls(block):
    """The calls of the next round: [{'slot', 'role', 'system', 'user'}], one for each role the pair has."""
    if block.done():
        raise ValueError('the block is finished')
    case, draft, faulty, _ = block.arriving()
    round_number = block.k + 1
    monitor_view, twin, window, informed, _ = views(block)
    out = []
    for role in block.arms:
        if role == 'C':
            system, user = prompts.corrector(block.kept, case, draft)
        elif role == 'M':
            system, user = prompts.monitor(monitor_view)
        elif role == 'T':
            system, user = prompts.monitor(twin)
        elif role == 'W':
            system, user = prompts.window(window)
        elif role == 'I':
            system, user = prompts.informed(informed, case['code'])
        elif role == 'G':
            system, user = prompts.generating(block.written, case)
        else:
            raise ValueError('unknown role ' + role)
        out.append({'slot': slot(block.engine, block.pair, block.which, round_number, role), 'role': role,
                    'system': system, 'user': user})
    return out


def absorb(block, records):
    """Apply the commit rules of sections 5.2 to 5.7a to the round's return records ({role: record}, each after the
    re-ask) and log the round. A valid corrector return is kept, an invalid one keeps the draft; ACCEPT (or an
    unreadable verdict) lets the draft join the monitor's, the window's or the informed monitor's register, REJECT
    leaves the register as it was and that case is absent for the rest of the block; the twin's verdict is recorded and
    never applied; a valid generated entry is kept, an invalid one keeps nothing."""
    if set(records) != set(block.arms):
        raise ValueError('a round needs one record for each role the pair has')
    case, draft, faulty, operator = block.arriving()
    round_number = block.k + 1
    monitor_view, twin, window, informed, planted_codes = views(block)
    window_codes = [e.code for _, e in prompts.code_order(window)[:3]]
    entry = {
        'round': round_number, 'case_id': case['case_id'], 'code': case['code'], 'form': case['form'],
        'faulty': faulty, 'operator': operator, 'draft': forms.line(draft),
        'views': {'M': len(monitor_view), 'T': len(twin), 'W': min(3, len(window)), 'I': len(informed)},
        'twin_same_bytes': prompts.monitor(monitor_view) == prompts.monitor(twin),
        'view_holds_planted': bool(planted_codes),
        'view_holds_fault_admitted_on_unreadable': any(f and u for _, _, f, u in block.outside),
        'window_holds_arriving': case['code'] in window_codes,
        'corrector_register_holds_fallback_fault': block.fallback_faulty,
        'records': records, 'kept': {}, 'accepted': {},
    }
    for role in block.arms:
        record = records[role]
        if role == 'C':
            kept = forms.parse(record['entry_line']) if record.get('valid') else draft
            if kept is None:
                raise ValueError('a valid corrector record holds no entry line')
            block.kept.append((case, kept))
            entry['kept']['C'] = forms.line(kept)
            if not record.get('valid') and faulty:
                block.fallback_faulty = True
        elif role == 'G':
            written = forms.parse(record['entry_line']) if record.get('valid') else None
            if written is not None:
                block.written.append((case, written))
            entry['kept']['G'] = forms.line(written) if written is not None else None
        elif role == 'M':
            ok = accepted(record)
            if ok:
                block.outside.append((case, draft, faulty, not record.get('valid')))
            entry['accepted']['M'] = ok
        elif role == 'W':
            ok = accepted(record)
            if ok:
                block.windowed.append((case, draft, faulty))
            entry['accepted']['W'] = ok
        elif role == 'I':
            ok = accepted(record)
            if ok:
                block.informed_register.append((case, draft, faulty))
            entry['accepted']['I'] = ok
        elif role == 'T':
            pass
    block.log.append(entry)
    block.k += 1
    return entry
