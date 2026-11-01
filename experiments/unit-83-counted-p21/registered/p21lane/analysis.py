"""The analysis: values, the mark, the second verdict and what is printed beside them (41-D-COUNTED, sections 6 and 7).

Package G of the build list (DESIGN-FREEZE.md, section 12, revision 4.2.1). `values` joins the logged rounds to the
blind score records; `look` is a pure function of those arrays that computes every quantity, check, mark, axis and label
of section 7 for pairs 1 to n of each engine by index; `decide` applies the stopping rule of section 7.4 to the mark;
`seal` writes a look's record where no command prints it and returns its hash. Nothing here is executed before both
registrations read REGISTERED (SD-001, SD-012); before then the only mechanical check is a parse of this file's text.

The rule, in the freeze's order (section 7.5), at a look: (1) NOT EVALUABLE where either engine fails a manipulation
check, the readable-replies condition or the ceiling; (2) SUPPORTED where the intervals of D and D- lie above +0.0075,
those of the level and L- above zero, the sign-flip check agrees and the confirmation values for D, D-, the level and
L- reach 40; (3) REFUTED where the positive control passed on its slope and PC- over pairs 1 to 220 with its two
confirmations at 40, the sign-flip check agrees, the confirmations of the form's conditions reach 40, D's interval
meets one of the three forms with the gate where the form needs it, and D+ is clear; the form printed is the strongest
the bounds support; (4) otherwise continue at the first look, INCONCLUSIVE with its reason at the second. At zero pooled
variance of any quantity the mark reads, no mark stands ("degenerate interval"). The confirmation (section 7.2) leaves
out, for each condition, every index at which either engine's pair rests on a reply left unusable after the re-ask as
that condition's own quantities read it, every engine's factor there 1.
"""
import collections
import hashlib
import json
import math
import operator
import random
from pathlib import Path

from p21base.common import GateError, canonical, digest, load, now, save

from . import collect, departures, forms, seal, stats

W = stats.WEIGHTS
ROUNDS = stats.ROUNDS
SLOPE_BOUND = 81.0 / 323.0       # every pair slope lies in [-81/323, +81/323] (section 7.2)
LEVEL_BOUND = 1.5                # every pair level lies in [-3/2, +3/2]
LOG_THRESHOLD = math.log(40.0)
VIEW_BANDS = ((1, 3), (4, 6), (7, 9), (10, 12), (13, 10 ** 6))
ROUND_BANDS = ((1, 6), (7, 12), (13, 18))
SUPPORTED, REFUTED, INCONCLUSIVE, NOT_EVALUABLE, CONTINUE = 'SUPPORTED', 'REFUTED', 'INCONCLUSIVE', 'NOT EVALUABLE', 'continue'
REASONS = {
    'margin': 'at the margin', 'precision': 'insufficient precision', 'crossing': 'a narrowing deficit or a crossing',
    'disagree': 'the two tests disagree', 'control': 'the positive control failed',
    'unusable': 'the mark depended on unusable replies', 'confirmation': 'the confirmation withheld it',
    'degenerate': 'degenerate interval', 'second_look': 'the second look was not reached',
    'against': 'not supported; the interaction ran against the claim',
}


# ---- values (section 6.2): the join of the logged rounds and the score records.

def _sound(scores, seed, case_id, line):
    record = scores.get(seal.entry_key(seal.case_key(case_id, seed), line))
    if record is None:
        raise GateError('a logged entry has no score record')
    return record['sound']


def _value(accepted, sound):
    return (1 if sound else -1) if accepted else 0


def _verdict_accepts(record):
    return not (record.get('valid') and record.get('verdict') == 'REJECT')


def values(logs_by_engine, scores, seed, required_by_case=None):
    """The arrays of every block, round and arm: v for each arm, the twin's and the windowed monitor's values, view
    sizes, the unusable flags after the re-ask, verdicts, and what section 6.5 prints. `required_by_case` maps a case
    identifier to its required Entry, for the corrector's returns that differ from the draft beyond the planted field."""
    arrays = {}
    for engine, logs in logs_by_engine.items():
        out = []
        for log in logs:
            blocks = []
            for block in log['blocks']:
                rounds = block['rounds']
                arms = [role for role in ('C', 'M', 'T', 'W', 'G', 'I') if role in rounds[0]['records']]
                b = {'which': block['which'], 'arms': arms, 'faulty': [], 'operator': [], 'class': [], 'catch': [],
                     'draft_sound': [], 'kept_sound': [], 'v_in': [], 'v_out': [], 'v_T': [], 'v_W': [], 'v_I': [],
                     'v_G': [], 'view': [], 'view_I': [], 'view_W': [], 'twin_same': [], 'view_holds_planted': [],
                     'fault_on_unreadable': [], 'fallback_fault': [], 'window_holds_arriving': [],
                     'changed_beyond_planted': [], 'unusable': {r: [] for r in arms}, 'verdict': {r: [] for r in arms},
                     'valid_first': {r: [] for r in arms}, 'asks': {r: [] for r in arms}, 'truncated': {r: [] for r in arms},
                     'refusals': {r: [] for r in arms}, 'tokens': {r: [] for r in arms}, 'reasons': {r: [] for r in arms}}
                for rnd in rounds:
                    case_id, draft = rnd['case_id'], rnd['draft']
                    faulty = rnd['faulty']
                    draft_sound = _sound(scores, seed, case_id, draft)
                    if draft_sound == faulty:
                        raise GateError('a draft\'s score disagrees with its schedule at ' + case_id)
                    kept = rnd['kept']['C']
                    kept_sound = _sound(scores, seed, case_id, kept)
                    records = rnd['records']
                    b['faulty'].append(faulty)
                    b['operator'].append(rnd.get('operator'))
                    b['class'].append(departures.CLASS.get(rnd.get('operator')) if faulty else None)
                    b['catch'].append(departures.CATCH.get(rnd.get('operator')) if faulty else None)
                    b['draft_sound'].append(draft_sound)
                    b['kept_sound'].append(kept_sound)
                    b['v_in'].append(1 if kept_sound else -1)
                    b['v_out'].append(_value(rnd['accepted']['M'], draft_sound))
                    b['v_T'].append(_value(_verdict_accepts(records['T']), draft_sound))
                    b['v_W'].append(_value(rnd['accepted']['W'], draft_sound) if 'W' in arms else None)
                    b['v_I'].append(_value(rnd['accepted']['I'], draft_sound) if 'I' in arms else None)
                    if 'G' in arms:
                        written = rnd['kept'].get('G')
                        b['v_G'].append(0 if written is None else (1 if _sound(scores, seed, case_id, written) else -1))
                    else:
                        b['v_G'].append(None)
                    b['view'].append(rnd['views']['M'])
                    b['view_I'].append(rnd['views'].get('I'))
                    b['view_W'].append(rnd['views'].get('W'))
                    b['twin_same'].append(rnd['twin_same_bytes'])
                    b['view_holds_planted'].append(rnd['view_holds_planted'])
                    b['fault_on_unreadable'].append(rnd['view_holds_fault_admitted_on_unreadable'])
                    b['fallback_fault'].append(rnd['corrector_register_holds_fallback_fault'])
                    b['window_holds_arriving'].append(rnd.get('window_holds_arriving'))
                    changed = None
                    if records['C'].get('valid') and required_by_case is not None:
                        draft_entry, kept_entry = forms.parse(draft), forms.parse(kept)
                        planted_fields = set(departures.differing_fields(draft_entry, required_by_case[case_id])) if faulty else set()
                        changed = bool(set(departures.differing_fields(kept_entry, draft_entry)) - planted_fields)
                    b['changed_beyond_planted'].append(changed)
                    for role in arms:
                        record = records[role]
                        b['unusable'][role].append(not record.get('usable'))
                        b['verdict'][role].append(record.get('verdict') if record.get('valid') else None)
                        asks = record.get('asks') or [record]
                        b['valid_first'][role].append(bool(asks[0].get('valid')))
                        b['asks'][role].append(len(asks))
                        b['truncated'][role].append(sum(1 for a in asks if a.get('truncated')))
                        b['refusals'][role].append([a.get('refusal_category') or 'none' for a in asks if a.get('refused')])
                        b['tokens'][role].append([sum((a.get(k) or 0) for a in asks) for k in ('input_tokens', 'output_tokens', 'reasoning_tokens')])
                        b['reasons'][role].append([a.get('reason') for a in asks if not a.get('valid')])
                blocks.append(b)
            out.append({'pair': log['pair'], 'started_utc': log.get('started_utc'), 'completed_utc': log.get('completed_utc'),
                        'blocks': blocks})
        arrays[engine] = out
    return arrays


# ---- Each block's quantities (sections 6.3 and 7.5).

def _first(flags):
    for k, flag in enumerate(flags):
        if flag:
            return k
    return None


def _outside_range(v, faulty, free):
    if not free:
        return v, v
    return (-1, 0) if faulty else (0, 1)


def _bounds(lo_in, hi_in, lo_out, hi_out):
    a = [li - ho for li, ho in zip(lo_in, hi_out)]
    b = [hi - lo for hi, lo in zip(hi_in, lo_out)]
    minus = sum(w * (ak if w > 0 else bk) for w, ak, bk in zip(W, a, b))
    plus = sum(w * (bk if w > 0 else ak) for w, ak, bk in zip(W, a, b))
    return minus, plus, stats.mean(a)


def block_quantities(b):
    """D, E, V, S, DC, the level, the positive control's slope, the bounds of section 7.5 (each arm free from its first
    unusable reply in the block to its end, by the sign of each round's weight), the outside view's growth g, the running
    score's slope and the absent-entry recomputations of section 7.7."""
    vin, vout, vt, faulty = b['v_in'], b['v_out'], b['v_T'], b['faulty']
    gap = [i - o for i, o in zip(vin, vout)]
    q = {'D': stats.slope(gap), 'E': stats.slope(vin), 'V': stats.slope([-t for t in vt]),
         'S': stats.slope([t - o for t, o in zip(vt, vout)]), 'level': stats.mean(gap),
         'outside_slope': stats.slope(vout), 'g': stats.slope(b['view'])}
    q['DC'] = q['E'] + q['V']
    first_c = _first(b['unusable']['C'])
    first_m = _first(b['unusable']['M'])
    lo_in = [-1 if first_c is not None and k >= first_c else v for k, v in enumerate(vin)]
    hi_in = [1 if first_c is not None and k >= first_c else v for k, v in enumerate(vin)]
    out_ranges = [_outside_range(v, f, first_m is not None and k >= first_m) for k, (v, f) in enumerate(zip(vout, faulty))]
    q['D_minus'], q['D_plus'], q['L_minus'] = _bounds(lo_in, hi_in, [r[0] for r in out_ranges], [r[1] for r in out_ranges])
    q['E_minus'] = sum(w * (lo if w > 0 else hi) for w, lo, hi in zip(W, lo_in, hi_in))
    q['E_plus'] = sum(w * (hi if w > 0 else lo) for w, lo, hi in zip(W, lo_in, hi_in))
    q['first_unusable'] = {'C': first_c, 'M': first_m, 'W': None}
    q['rests_D'] = first_c is not None or first_m is not None
    if 'W' in b['arms']:
        vw = b['v_W']
        first_w = _first(b['unusable']['W'])
        q['first_unusable']['W'] = first_w
        q['PC'] = stats.slope([i - x for i, x in zip(vin, vw)])
        w_ranges = [_outside_range(v, f, first_w is not None and k >= first_w) for k, (v, f) in enumerate(zip(vw, faulty))]
        q['PC_minus'], _, _ = _bounds(lo_in, hi_in, [r[0] for r in w_ranges], [r[1] for r in w_ranges])
        q['rests_PC'] = first_c is not None or first_w is not None
    running_in = [stats.mean(vin[:r]) for r in range(1, ROUNDS + 1)]
    running_out = [stats.mean(vout[:r]) for r in range(1, ROUNDS + 1)]
    q['running'] = stats.slope([i - o for i, o in zip(running_in, running_out)])
    for name, absent in (('lo', -0.5), ('hi', 0.5)):
        vout_a = [absent if v == 0 else v for v in vout]
        vt_a = [absent if v == 0 else v for v in vt]
        q['D_absent_' + name] = stats.slope([i - o for i, o in zip(vin, vout_a)])
        q['V_absent_' + name] = stats.slope([-t for t in vt_a])
        q['S_absent_' + name] = stats.slope([t - o for t, o in zip(vt_a, vout_a)])
    if 'G' in b['arms']:
        q['G_slope'] = stats.slope(b['v_G'])
        q['G_gap'] = stats.slope([g - o for g, o in zip(b['v_G'], vout)])
    if 'I' in b['arms']:
        q['I_gap'] = stats.slope([i - o for i, o in zip(b['v_I'], vout)])
    return q


SCALARS = ('D', 'E', 'V', 'S', 'DC', 'level', 'outside_slope', 'g', 'D_minus', 'D_plus', 'L_minus', 'E_minus', 'E_plus',
           'running', 'D_absent_lo', 'D_absent_hi', 'V_absent_lo', 'V_absent_hi', 'S_absent_lo', 'S_absent_hi')


def pair_values(pair):
    """A pair's value of each quantity is the mean of its two blocks'; it rests on an unusable reply where either block
    does (section 7.2)."""
    qs = [block_quantities(b) for b in pair['blocks']]
    out = {name: (qs[0][name] + qs[1][name]) / 2.0 for name in SCALARS}
    out['rests_D'] = qs[0]['rests_D'] or qs[1]['rests_D']
    for name in ('PC', 'PC_minus', 'G_slope', 'G_gap', 'I_gap'):
        if name in qs[0] and name in qs[1]:
            out[name] = (qs[0][name] + qs[1][name]) / 2.0
    if 'rests_PC' in qs[0]:
        out['rests_PC'] = qs[0]['rests_PC'] or qs[1]['rests_PC']
    out['first_unusable'] = [q['first_unusable'] for q in qs]
    out['pair'] = pair['pair']
    return out


# ---- The confirmation (section 7.2).

CONDITIONS = {
    # name: (quantity, direction, boundary in tolerances, rests key, lower bound, upper bound)
    'D>+tol': ('D', 'above', 1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'D->+tol': ('D_minus', 'above', 1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'level>0': ('level', 'above', 0, 'rests_D', -LEVEL_BOUND, LEVEL_BOUND),
    'L->0': ('L_minus', 'above', 0, 'rests_D', -LEVEL_BOUND, LEVEL_BOUND),
    'D<+tol': ('D', 'below', 1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'D+<+tol': ('D_plus', 'below', 1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'D>-tol': ('D', 'above', -1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'D->-tol': ('D_minus', 'above', -1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'D<-tol': ('D', 'below', -1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'D+<-tol': ('D_plus', 'below', -1, 'rests_D', -SLOPE_BOUND, SLOPE_BOUND),
    'PC>+tol': ('PC', 'above', 1, 'rests_PC', -SLOPE_BOUND, SLOPE_BOUND),
    'PC->+tol': ('PC_minus', 'above', 1, 'rests_PC', -SLOPE_BOUND, SLOPE_BOUND),
}
CONTROL_CONDITIONS = ('PC>+tol', 'PC->+tol')


def factor(x, direction, boundary, lower, upper):
    """A pair value's factor: 1 + (x - m) / 2(m - a) for a mean above m, 1 - (x - m) / 2(b - m) for a mean below it;
    each is at least one half within the bounds. The error names no value, so that the error trace of a look that
    cannot be computed carries no result (section 7.4)."""
    if not lower - 1e-12 <= x <= upper + 1e-12:
        raise GateError('a pair value read by the confirmation lies outside its bounds by construction')
    if direction == 'above':
        return 1.0 + (x - boundary) / (2.0 * (boundary - lower))
    return 1.0 - (x - boundary) / (2.0 * (upper - boundary))


def confirmation(pv_by_engine, condition, n, tolerance):
    """The confirmation value of one condition over indices 1 to n: the product over the indices read of the product
    of the admitted engines' factors; an index at which either engine's pair rests on a reply left unusable after the
    re-ask, as this condition's quantities read it, is left out as a whole (revision 4.2.1)."""
    quantity, direction, k, rests, lower, upper = CONDITIONS[condition]
    boundary = k * tolerance
    engines = sorted(pv_by_engine)
    log_value, left_out = 0.0, []
    for i in range(n):
        rows = [pv_by_engine[e][i] for e in engines]
        if any(row.get(rests) for row in rows):
            left_out.append(i + 1)
            continue
        for row in rows:
            log_value += math.log(factor(row[quantity], direction, boundary, lower, upper))
    return {'condition': condition, 'log_value': log_value, 'value': math.exp(min(log_value, 700.0)),
            'reached': log_value >= LOG_THRESHOLD - 1e-12, 'indices_left_out': left_out, 'n_left_out': len(left_out),
            'n_read': n}


# ---- Item 1 of section 7.5, on each engine.

def _rate(numerator, denominator):
    return numerator / denominator if denominator else None


def engine_checks(pairs, control_n, config):
    """The manipulation checks, the readable-replies condition for each role apart and the ceiling, on one engine."""
    disc = {'faulty': 0, 'faulty_rej': 0, 'sound': 0, 'sound_rej': 0}
    repair = {'faulty': 0, 'repaired': 0, 'sound': 0, 'damaged': 0}
    readable = {role: collections.defaultdict(lambda: [0, 0]) for role in ('M', 'T', 'C')}
    unusable = {role: [0, 0] for role in ('C', 'M', 'T', 'W')}
    for index, pair in enumerate(pairs):
        for b in pair['blocks']:
            for k in range(ROUNDS):
                faulty = b['faulty'][k]
                if b['view'][k] == 1:
                    for role in ('M', 'T'):
                        rejected = b['verdict'][role][k] == 'REJECT'
                        disc['faulty' if faulty else 'sound'] += 1
                        disc['faulty_rej' if faulty else 'sound_rej'] += int(rejected)
                if k < 2:
                    if faulty:
                        repair['faulty'] += 1
                        repair['repaired'] += int(b['kept_sound'][k])
                    else:
                        repair['sound'] += 1
                        repair['damaged'] += int(not b['kept_sound'][k])
                for role in ('M', 'T'):
                    band = _band(b['view'][k], VIEW_BANDS)
                    readable[role][band][0] += int(not b['unusable'][role][k])
                    readable[role][band][1] += 1
                band = _band(k + 1, ROUND_BANDS)
                readable['C'][band][0] += int(not b['unusable']['C'][k])
                readable['C'][band][1] += 1
                for role in ('C', 'M', 'T'):
                    unusable[role][0] += int(b['unusable'][role][k])
                    unusable[role][1] += 1
                if index < control_n and 'W' in b['arms']:
                    unusable['W'][0] += int(b['unusable']['W'][k])
                    unusable['W'][1] += 1
    hit = _rate(disc['faulty_rej'], disc['faulty'])
    false = _rate(disc['sound_rej'], disc['sound'])
    discrimination = None if hit is None or false is None else hit - false
    rep = _rate(repair['repaired'], repair['faulty'])
    dam = _rate(repair['damaged'], repair['sound'])
    net = None if rep is None or dam is None else rep - dam
    floor = config['manipulation_floor']
    regime = config['readable_regime']
    failures = []
    if discrimination is None or discrimination < floor:
        failures.append(('discrimination', 'discrimination at view size 1 under %.2f' % floor))
    if net is None or net < floor:
        failures.append(('net_repair', 'net repair at rounds 1 and 2 under %.2f' % floor))
    bands_out = {}
    for role, bands in readable.items():
        total_ok = sum(v[0] for v in bands.values())
        total = sum(v[1] for v in bands.values())
        share = _rate(total_ok, total)
        bands_out[role] = {'overall': share, 'calls': total,
                           'bands': {'%d-%d' % band: {'readable': v[0], 'calls': v[1], 'share': _rate(v[0], v[1])} for band, v in sorted(bands.items())}}
        if share is None or share < regime['overall']:
            failures.append(('readable_' + role, 'readable %s replies under %.2f over all calls' % (role, regime['overall'])))
        for band, v in bands.items():
            if v[1] >= regime['band_minimum_calls'] and v[0] / v[1] < regime['band']:
                failures.append(('readable_' + role + '_band', 'readable %s replies under %.2f in band %d-%d' % (role, regime['band'], band[0], band[1])))
    ceiling = {}
    broken = []
    for role, (bad, total) in unusable.items():
        share = _rate(bad, total)
        ceiling[role] = {'unusable': bad, 'calls': total, 'share': share}
        if share is not None and share > config['ceiling']:
            broken.append(role)
    return {'discrimination': discrimination, 'hit': hit, 'false_alarm': false, 'net_repair': net, 'repair': rep,
            'damage': dam, 'readable': bands_out, 'ceiling': ceiling, 'broken_arms': broken, 'failures': failures,
            'admitted': not failures and not broken}


def _band(value, bands):
    for low, high in bands:
        if low <= value <= high:
            return (low, high)
    raise ValueError('no band holds %r' % value)


# ---- The registered fall and the part it carries (sections 6.3 and 7.6), with one bootstrap.

def _fall_counts(b, mode):
    """The twin's calls by view size: [faulty calls, faulty rejections, sound calls, sound rejections]. mode 'main'
    reads readable verdicts only; 'reject' and 'accept' count unreadable twin verdicts as REJECT or ACCEPT; 'pre' reads
    readable verdicts before the block's first unreadable monitor verdict. Role 'I' reads the informed monitor."""
    role = 'I' if mode == 'informed' else 'T'
    views = b['view_I'] if role == 'I' else b['view']
    out = {}
    first_m = _first(b['unusable']['M'])
    for k in range(ROUNDS):
        unreadable = b['unusable'][role][k]
        if mode in ('main', 'informed', 'pre') and unreadable:
            continue
        if mode == 'pre' and first_m is not None and k >= first_m:
            continue
        verdict = b['verdict'][role][k]
        rejected = (verdict == 'REJECT') if not unreadable else (mode == 'reject')
        row = out.setdefault(views[k], [0, 0, 0, 0])
        if b['faulty'][k]:
            row[0] += 1
            row[1] += int(rejected)
        else:
            row[2] += 1
            row[3] += int(rejected)
    return out


def _weighted_slope(xs, ys, ws):
    total = sum(ws)
    if total <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / total
    my = sum(w * y for w, y in zip(ws, ys)) / total
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        return None
    return sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx


def fall_from_totals(totals, min_calls, f_cap):
    """One engine's registered fall from summed counts: sizes with fewer than min_calls faulty or sound calls are left
    out and counted; d(n) = (h - f) / (1 - f) with f held at no more than f_cap; beta its least-squares slope on n
    weighted by calls; f-bar the false-alarm rate pooled over the sizes the fall uses."""
    kept = sorted(n for n, row in totals.items() if row[0] >= min_calls and row[2] >= min_calls)
    left_out = sorted(n for n in totals if n not in kept)
    if len(kept) < 2:
        return {'beta': None, 'fbar': None, 'sizes': kept, 'left_out': left_out}
    xs, ys, ws, curve = [], [], [], {}
    for n in kept:
        fc, fr, sc, sr = totals[n]
        h = fr / fc
        f = min(sr / sc, f_cap)
        d = (h - f) / (1.0 - f)
        xs.append(n); ys.append(d); ws.append(fc + sc)
        curve[n] = {'h': h, 'f': sr / sc, 'd': d, 'calls': fc + sc}
    fbar = sum(totals[n][3] for n in kept) / sum(totals[n][2] for n in kept)
    return {'beta': _weighted_slope(xs, ys, ws), 'fbar': fbar, 'sizes': kept, 'left_out': left_out, 'curve': curve}


def _sum_counts(dicts):
    totals = {}
    for d in dicts:
        for n, row in d.items():
            t = totals.setdefault(n, [0, 0, 0, 0])
            for i in range(4):
                t[i] += row[i]
    return totals


def _w_of(fbar, beta, gbar):
    if fbar is None or beta is None or gbar is None:
        return None
    return -0.5 * (1.0 - fbar) * beta * gbar


def mechanism_point(pairs_by_engine, pv_by_engine, config, mode='main'):
    engines = sorted(pairs_by_engine)
    falls = {}
    for e in engines:
        totals = _sum_counts(_fall_counts(b, mode) for pair in pairs_by_engine[e] for b in pair['blocks'])
        falls[e] = fall_from_totals(totals, config['fall']['min_calls'], config['fall']['f_cap'])
    betas = [falls[e]['beta'] for e in engines]
    fbars = [falls[e]['fbar'] for e in engines]
    beta = None if any(v is None for v in betas) else stats.mean(betas)
    fbar = None if any(v is None for v in fbars) else stats.mean(fbars)
    gbar = stats.mean([stats.mean([row['g'] for row in pv_by_engine[e]]) for e in engines])
    return {'falls': falls, 'beta': beta, 'fbar': fbar, 'gbar': gbar, 'W': _w_of(fbar, beta, gbar)}


def _columns(pairs, pv, max_view):
    """Per pair, the vector the bootstrap sums: the twin's fall counts by size, E, g, and the depth sums of y for the
    in-loop and outside arms (the scale label)."""
    columns = []
    for n in range(1, max_view + 1):
        for i in range(4):
            columns.append([0.0] * len(pairs))
    e_col, g_col = [0.0] * len(pairs), [0.0] * len(pairs)
    y_in = [[0.0] * len(pairs) for _ in range(ROUNDS)]
    y_out = [[0.0] * len(pairs) for _ in range(ROUNDS)]
    for p, pair in enumerate(pairs):
        for b in pair['blocks']:
            for n, row in _fall_counts(b, 'main').items():
                for i in range(4):
                    columns[(n - 1) * 4 + i][p] += row[i]
            for k in range(ROUNDS):
                y_in[k][p] += (1 + b['v_in'][k]) / 2.0
                y_out[k][p] += (1 + b['v_out'][k]) / 2.0
        e_col[p] = pv[p]['E']
        g_col[p] = pv[p]['g']
    return columns, e_col, g_col, y_in, y_out


def _transform(y):
    return math.asin(math.sqrt(min(1.0, max(0.0, y))))


def _scale_slope(means_in, means_out):
    """The slope over depth of the in-loop minus outside gap of the arm means of y on the arcsine square root scale."""
    return stats.slope([_transform(a) - _transform(b) for a, b in zip(means_in, means_out)])


def bootstrap(pairs_by_engine, pv_by_engine, config, level):
    """One bootstrap over pairs within engine (section 7.2): 20,000 resamples from the registered seed, every quantity
    recomputed in each resample: the registered fall, W, M and the scale label's slope. Percentile intervals at the
    look's level; the standard error of W is the resamples' standard deviation."""
    engines = sorted(pairs_by_engine)
    max_view = max(max(b['view']) for e in engines for pair in pairs_by_engine[e] for b in pair['blocks'])
    prepared = {e: _columns(pairs_by_engine[e], pv_by_engine[e], max_view) for e in engines}
    rng = random.Random(stats.seed_int(config['bootstrap']['seed']))
    resamples = config['bootstrap']['resamples']
    out = {'beta': [], 'W': [], 'M': [], 'scale': [], 'E': [], 'undefined': 0}
    mul = operator.mul
    for _ in range(resamples):
        betas, fbars, gs, es, ins, outs = [], [], [], [], [], []
        for e in engines:
            columns, e_col, g_col, y_in, y_out = prepared[e]
            n = len(e_col)
            weights = [0] * n
            for i in rng.choices(range(n), k=n):
                weights[i] += 1
            totals = {}
            for size in range(1, max_view + 1):
                row = [sum(map(mul, weights, columns[(size - 1) * 4 + i])) for i in range(4)]
                if any(row):
                    totals[size] = row
            fall = fall_from_totals(totals, config['fall']['min_calls'], config['fall']['f_cap'])
            betas.append(fall['beta']); fbars.append(fall['fbar'])
            gs.append(sum(map(mul, weights, g_col)) / n)
            es.append(sum(map(mul, weights, e_col)) / n)
            ins.append([sum(map(mul, weights, y_in[k])) / (2.0 * n) for k in range(ROUNDS)])
            outs.append([sum(map(mul, weights, y_out[k])) / (2.0 * n) for k in range(ROUNDS)])
        e_pooled = stats.mean(es)
        out['E'].append(e_pooled)
        out['scale'].append(stats.mean([_scale_slope(a, b) for a, b in zip(ins, outs)]))
        if any(v is None for v in betas) or any(v is None for v in fbars):
            out['undefined'] += 1
            continue
        beta, fbar, gbar = stats.mean(betas), stats.mean(fbars), stats.mean(gs)
        w = _w_of(fbar, beta, gbar)
        out['beta'].append(beta); out['W'].append(w); out['M'].append(e_pooled + w)
    result = {'resamples': resamples, 'undefined': out['undefined']}
    for name in ('beta', 'W', 'M', 'scale', 'E'):
        result[name + '_interval'] = stats.percentile_interval([float(v) for v in out[name]], level)
    result['W_se'] = math.sqrt(stats.sample_variance(out['W'])) if len(out['W']) > 1 else None
    return result


def scale_label(pairs_by_engine, config):
    """The point quantities of the scale label (section 7.7): the transformed slope, the share of engine, arm and depth
    means of y within 0.05 of 0 or 1 (the control's cells left out) and half the transform's mean slope at the arm means."""
    engines = sorted(pairs_by_engine)
    slopes, near, cells, derivatives = [], 0, 0, []
    for e in engines:
        pairs = pairs_by_engine[e]
        n = 2.0 * len(pairs)
        means_in = [sum((1 + b['v_in'][k]) / 2.0 for p in pairs for b in p['blocks']) / n for k in range(ROUNDS)]
        means_out = [sum((1 + b['v_out'][k]) / 2.0 for p in pairs for b in p['blocks']) / n for k in range(ROUNDS)]
        slopes.append(_scale_slope(means_in, means_out))
        for y in means_in + means_out:
            cells += 1
            near += int(y <= 0.05 or y >= 0.95)
            y = min(1 - 1e-6, max(1e-6, y))
            derivatives.append(1.0 / (2.0 * math.sqrt(y * (1.0 - y))))
    return {'slope': stats.mean(slopes), 'near_bound_share': near / cells if cells else None,
            'half_mean_transform_slope': 0.5 * stats.mean(derivatives)}


def negative_control(pairs_by_engine, level):
    """Over the rounds where the twin's prompt is the monitor's byte for byte, the pooled difference between the two
    rejection rates (each pair's own rate difference, pooled by the estimator of section 7.2), with its 99.9 per cent
    interval, which must include zero (section 7.10)."""
    diffs, agreement = {}, [0, 0]
    for e, pairs in pairs_by_engine.items():
        values = []
        for pair in pairs:
            m = rm = rt = 0
            for b in pair['blocks']:
                for k in range(ROUNDS):
                    if not b['twin_same'][k]:
                        continue
                    m += 1
                    vm, vt = b['verdict']['M'][k] == 'REJECT', b['verdict']['T'][k] == 'REJECT'
                    rm += int(vm); rt += int(vt)
                    agreement[0] += int(vm == vt); agreement[1] += 1
            if m:
                values.append((rm - rt) / m)
        diffs[e] = values
    if any(len(v) < 2 for v in diffs.values()):
        return {'passed': False, 'interval': None, 'reason': 'too few pairs with identical prompts'}
    row = stats.pooled(diffs)
    interval = stats.t_interval(row, level)
    if interval is None:
        passed = row['estimate'] == 0.0
    else:
        passed = interval[0] <= 0.0 <= interval[1]
    return {'passed': passed, 'estimate': row['estimate'], 'interval': interval, 'level': level,
            'agreement': _rate(agreement[0], agreement[1])}


# ---- The look (sections 7.2 to 7.10).

def _interval_row(values_by_engine, level):
    row = stats.pooled(values_by_engine)
    interval = stats.t_interval(row, level)
    per_engine = {e: {'estimate': row['engines'][e]['mean'], 'se': row['engines'][e]['se'],
                      'interval_95': stats.engine_interval(values_by_engine[e], 0.95),
                      'interval_look': stats.engine_interval(values_by_engine[e], level)} for e in values_by_engine}
    return {'estimate': row['estimate'], 'se': row['se'], 'df': row['df'], 'interval': interval,
            'degenerate': row['degenerate'], 'engines': per_engine}


def _above(row, boundary):
    return row['interval'] is not None and row['interval'][0] > boundary


def _below(row, boundary):
    return row['interval'] is not None and row['interval'][1] < boundary


def _gate_met(row, gate):
    return row['se'] == row['se'] and row['se'] <= gate


def evaluate_mark(est, conf, flips, control, gate, tolerance, directional, alpha):
    """Items 2 to 4 of section 7.5, given the item-1 result already passed. Returns (mark, form, reason, detail)."""
    D, Dm, Dp, L, Lm = est['D'], est['D_minus'], est['D_plus'], est['level'], est['L_minus']
    PC, PCm = est['PC'], est['PC_minus']
    for row in (D, L, Dm, Dp, Lm, PC, PCm):
        if row['degenerate']:
            return None, None, 'degenerate', {}
    tol = tolerance
    ok = lambda name: conf[name]['reached']
    flip_ok = lambda key: flips[key]['p'] <= alpha
    detail = {}
    # SUPPORTED.
    supp_interval = _above(D, tol)
    if supp_interval:
        parts = {'level': _above(L, 0.0), 'flip': flip_ok('above_+tol'), 'bounds': _above(Dm, tol) and _above(Lm, 0.0),
                 'confirmation': all(ok(c) for c in ('D>+tol', 'D->+tol', 'level>0', 'L->0'))}
        detail['supported'] = parts
        if all(parts.values()):
            return SUPPORTED, None, None, detail
        if not parts['level']:
            return None, None, 'crossing', detail
        if not parts['flip']:
            return None, None, 'disagree', detail
        if not parts['bounds']:
            return None, None, 'unusable', detail
        return None, None, 'confirmation', detail
    # REFUTED: the strongest form whose conditions hold.
    control_passed = control['passed']
    dplus_clear = (_below(Dp, tol) and _gate_met(Dp, gate)) or _below(Dp, -tol)
    forms_by_d = {
        'constant': D['interval'] is not None and D['interval'][0] > -tol and D['interval'][1] < tol and _gate_met(D, gate),
        'narrowing': _below(D, -tol),
        'no widening': _below(D, tol) and _gate_met(D, gate),
    }
    # Each sign-flip key names the side of a boundary the mark asserts: 'below_+tol' is the lower tail at c = +tol (the
    # upper side of REFUTED), 'above_-tol' the upper tail at c = -tol (the lower side of the band), 'below_-tol' the
    # lower tail at c = -tol (narrowing); SUPPORTED reads 'above_+tol', the upper tail at c = +tol (section 7.5).
    needs = {
        'constant': (('below_+tol', 'above_-tol'), ('D<+tol', 'D+<+tol', 'D>-tol', 'D->-tol') + CONTROL_CONDITIONS),
        'narrowing': (('below_-tol',), ('D<-tol', 'D+<-tol') + CONTROL_CONDITIONS),
        'no widening': (('below_+tol',), ('D<+tol', 'D+<+tol') + CONTROL_CONDITIONS),
    }
    supports = {'constant': _above(Dm, -tol) and dplus_clear, 'narrowing': _below(Dp, -tol), 'no widening': dplus_clear}
    first_failure = None
    for form in ('constant', 'narrowing', 'no widening'):
        if not forms_by_d[form]:
            continue
        flip_keys, conditions = needs[form]
        parts = {'control': control_passed, 'flip': all(flip_ok(k) for k in flip_keys), 'bounds': supports[form],
                 'confirmation': all(ok(c) for c in conditions)}
        detail['refuted_' + form] = parts
        if all(parts.values()):
            if form != 'constant' and not directional:
                return None, None, 'against', detail
            return REFUTED, form, None, detail
        if first_failure is None:
            first_failure = parts
    if first_failure is not None:
        if not first_failure['control']:
            return None, None, 'control', detail
        if not first_failure['flip']:
            return None, None, 'disagree', detail
        if not first_failure['bounds']:
            return None, None, 'unusable', detail
        return None, None, 'confirmation', detail
    if D['interval'] is not None and (D['interval'][0] <= tol <= D['interval'][1] or D['interval'][0] <= -tol <= D['interval'][1]):
        return None, None, 'margin', detail
    return None, None, 'precision', detail


def mechanism_axis(mark, negative, boot, point, est, gate, tolerance):
    """The second verdict of section 7.6, read beside the mark and never in its place."""
    if mark == NOT_EVALUABLE or not negative['passed']:
        return {'verdict': NOT_EVALUABLE, 'fall_registered': None}
    beta_interval, w_interval = boot['beta_interval'], boot['W_interval']
    registered = beta_interval is not None and beta_interval[1] < 0.0
    if registered and w_interval is not None and w_interval[0] > tolerance:
        verdict = 'supported'
    elif w_interval is not None and w_interval[1] < tolerance and boot['W_se'] is not None and boot['W_se'] <= gate:
        verdict = 'unsupported'
    else:
        verdict = 'inconclusive'
    labels = []
    if est['E']['interval'] is not None and est['E']['interval'][0] > tolerance:
        labels.append('corrector-driven')
    w = point['W']
    # Printed beside the axis with no verdict (section 7.6): E with its interval and its bounds E- and E+, and M with
    # M- and M+ (E- plus W, and E+ plus W).
    beside = {'E': est['E']['estimate'], 'E_interval': est['E']['interval'],
              'E_minus': est['E_minus']['estimate'], 'E_minus_interval': est['E_minus']['interval'],
              'E_plus': est['E_plus']['estimate'], 'E_plus_interval': est['E_plus']['interval'],
              'M': None if w is None else est['E']['estimate'] + w,
              'M_minus': None if w is None else est['E_minus']['estimate'] + w,
              'M_plus': None if w is None else est['E_plus']['estimate'] + w}
    return {'verdict': verdict, 'fall_registered': registered, 'beta': point['beta'], 'beta_interval': beta_interval,
            'W': w, 'W_interval': w_interval, 'W_se': boot['W_se'], 'M_interval': boot['M_interval'],
            'fbar': point['fbar'], 'gbar': point['gbar'], 'labels': labels, 'beside': beside}


def _reading(row, tolerance):
    """Where an interval lies against the boundaries of section 7.5, as words: above the tolerance, inside the band,
    below the tolerance (not wholly inside the band), below minus the tolerance, or none of these."""
    interval = row['interval']
    if interval is None:
        return 'degenerate'
    low, high = interval
    if low > tolerance:
        return 'above the tolerance'
    if high < -tolerance:
        return 'below minus the tolerance'
    if low > -tolerance and high < tolerance:
        return 'inside the band'
    if high < tolerance:
        return 'below the tolerance, not wholly inside the band'
    return 'none of these'


def absent_readings(est, tolerance):
    """Section 7.7, the absent entry: D and the parts V and S recomputed with an absent entry valued at minus 0.5 and at
    plus 0.5 (E does not change: the in-loop arm always keeps an entry), each with its interval at the look's level, and
    the reading of D's interval against the boundaries the mark reads. Printed with the derivation of section 6.2 and
    no label; it never changes a mark. The confirmation is not recomputed, because the bounds of section 7.2 are
    derived for an absent value of 0 (BUILD-DECISIONS.md, 18 October 2026)."""
    out = {}
    for name, value in (('lo', -0.5), ('hi', 0.5)):
        out[name] = {'absent_value': value, 'D': est['D_absent_' + name], 'V': est['V_absent_' + name],
                     'S': est['S_absent_' + name], 'E': est['E'],
                     'D_reading': _reading(est['D_absent_' + name], tolerance)}
    return out


def check_constants(config):
    """The constants this module carries agree with the registered configuration: the confirmation's threshold of 40
    and the bands of the readable-replies condition (sections 7.2 and 7.5)."""
    bands = config['readable_regime']
    view = tuple((low, high if high is not None else VIEW_BANDS[-1][1]) for low, high in bands['monitor_and_twin_bands_by_entries_in_view'])
    rounds = tuple(tuple(band) for band in bands['corrector_bands_by_round'])
    if config['confirmation_threshold'] != 40 or view != VIEW_BANDS or rounds != ROUND_BANDS:
        raise GateError('the analysis constants and the registered configuration disagree')
    if config['stopping_word'] not in ('mark', 'both'):
        raise GateError('the stopping word is mark or both (section 7.4)')


def look(arrays, n_pairs, number, config):
    """Every quantity, check, mark, axis and label of section 7 for pairs 1 to n_pairs of each engine by index; a pure
    function of the arrays. number is 1 or 2."""
    check_constants(config)
    looks = config['looks']
    level, gate, alpha = looks['levels'][number - 1], looks['gates'][number - 1], looks['sign_flip_alpha'][number - 1]
    tolerance = config['tolerance']
    engines = sorted(arrays)
    pairs = {}
    for e in engines:
        if len(arrays[e]) < n_pairs:
            raise GateError('a look reads complete pairs only: %s has %d of %d' % (e, len(arrays[e]), n_pairs))
        pairs[e] = arrays[e][:n_pairs]
        if [p['pair'] for p in pairs[e]] != list(range(1, n_pairs + 1)):
            raise GateError('a look reads pairs 1 to n by index')
    control_n = min(config['positive_control_pairs'], n_pairs)
    pv = {e: [pair_values(p) for p in pairs[e]] for e in engines}
    checks = {e: engine_checks(pairs[e], control_n, config) for e in engines}
    est = {}
    for name in SCALARS:
        est[name] = _interval_row({e: [row[name] for row in pv[e]] for e in engines}, level)
    for name in ('PC', 'PC_minus'):
        est[name] = _interval_row({e: [row[name] for row in pv[e][:control_n]] for e in engines}, level)
    conf = {}
    for name in CONDITIONS:
        n = control_n if name in CONTROL_CONDITIONS else n_pairs
        conf[name] = confirmation({e: pv[e][:n] for e in engines}, name, n, tolerance)
    flips_cfg = config['sign_flip']
    d_values = {e: [row['D'] for row in pv[e]] for e in engines}
    # c is +0.0075 for SUPPORTED and for the upper side of REFUTED, -0.0075 for the lower side of the band and for
    # narrowing; each key names the side of the boundary the mark asserts (section 7.5).
    flips = {
        'above_+tol': stats.sign_flip_p(d_values, tolerance, 'upper', flips_cfg['flips'], flips_cfg['seed']),
        'below_+tol': stats.sign_flip_p(d_values, tolerance, 'lower', flips_cfg['flips'], flips_cfg['seed']),
        'above_-tol': stats.sign_flip_p(d_values, -tolerance, 'upper', flips_cfg['flips'], flips_cfg['seed']),
        'below_-tol': stats.sign_flip_p(d_values, -tolerance, 'lower', flips_cfg['flips'], flips_cfg['seed']),
    }
    control = {'passed': _above(est['PC'], tolerance) and _above(est['PC_minus'], tolerance),
               'slope': est['PC'], 'lower_bound': est['PC_minus'], 'pairs': control_n}
    admitted = [e for e in engines if checks[e]['admitted']]
    negative = negative_control(pairs, config['negative_control_level'])
    point = mechanism_point(pairs, pv, config)
    boot = bootstrap(pairs, pv, config, level)
    scale = scale_label(pairs, config)
    if len(admitted) < config['engines_needed']:
        mark, form, reason, detail = NOT_EVALUABLE, None, None, {}
        broken = [(e, arm) for e in engines for arm in checks[e]['broken_arms']]
        cause = 'broken_instrument' if broken else 'regime'
    else:
        broken, cause = [], None
        mark, form, reason, detail = evaluate_mark(est, conf, flips, control, gate, tolerance,
                                                   config['directional_reading'], alpha)
        if mark is None:
            mark = CONTINUE if number == 1 else INCONCLUSIVE
    axis = mechanism_axis(mark, negative, boot, point, est, gate, tolerance)
    labels = []
    if mark in (SUPPORTED, REFUTED):
        labels = _labels(mark, est, scale, boot, pairs, tolerance, config)
    sensitivities = {mode: mechanism_point(pairs, pv, config, mode)['W'] for mode in ('reject', 'accept', 'pre')}
    record = {
        'study_id': config['study_id'], 'look': number, 'n_pairs': n_pairs, 'level': level, 'gate': gate,
        'tolerance': tolerance, 'stopping_word': config['stopping_word'], 'engines': engines, 'admitted': admitted,
        'checks': checks,
        'mark': mark, 'form': form, 'reason': reason, 'reason_text': REASONS.get(reason), 'cause': cause,
        'broken': broken, 'detail': detail, 'estimates': est, 'confirmation': conf, 'sign_flip': flips,
        'positive_control': {'passed': control['passed'], 'pairs': control_n}, 'negative_control': negative,
        'mechanism': {**axis, 'point': _strip_curves(point), 'sensitivities': sensitivities},
        'bootstrap': boot, 'scale': scale, 'labels': labels,
        'absent_entry': absent_readings(est, tolerance),
        'printed': printed(pairs, pv, est, point, config, level),
    }
    return record


def _strip_curves(point):
    out = {k: v for k, v in point.items() if k != 'falls'}
    out['falls'] = {e: {k: v for k, v in f.items()} for e, f in point['falls'].items()}
    return out


def _labels(mark, est, scale, boot, pairs, tolerance, config):
    labels = []
    interval = boot['scale_interval']
    if mark == SUPPORTED and not (interval is not None and interval[0] > 0.0):
        labels.append('scale-dependent')
    if mark == REFUTED and interval is not None and scale['half_mean_transform_slope'] > 0:
        converted = (interval[0] / scale['half_mean_transform_slope'], interval[1] / scale['half_mean_transform_slope'])
        if converted[0] > tolerance:
            labels.append('scale-dependent')
    for e, row in est['D']['engines'].items():
        own = row['interval_95']
        if own is None:
            continue
        if mark == SUPPORTED and own[1] < tolerance:
            labels.append('family-dependent')
            break
        if mark == REFUTED and own[0] > tolerance:
            labels.append('family-dependent')
            break
    share = budget_share(pairs)
    if share['share'] is not None and share['share'] > config['budget_label_share']:
        labels.append('budget-limited')
    return labels


def budget_share(pairs_by_engine, min_view=7):
    """The share of the monitor's and the twin's replies at views of seven or more entries that reached the
    16,000-token allowance (section 7.7), every ask counted."""
    hit = total = 0
    for pairs in pairs_by_engine.values():
        for pair in pairs:
            for b in pair['blocks']:
                for k in range(ROUNDS):
                    if b['view'][k] >= min_view:
                        for role in ('M', 'T'):
                            hit += b['truncated'][role][k]
                            total += b['asks'][role][k]
    return {'share': _rate(hit, total), 'truncated': hit, 'asks': total}


# ---- What section 6.5 prints beside the mark (with intervals and no label).

def _by_round(pairs_by_engine, fn):
    out = []
    for k in range(ROUNDS):
        num = den = 0
        for pairs in pairs_by_engine.values():
            for pair in pairs:
                for b in pair['blocks']:
                    n, d = fn(b, k)
                    num += n
                    den += d
        out.append(_rate(num, den))
    return out


def printed(pairs, pv, est, point, config, level):
    tolerance = config['tolerance']
    out = {}
    fbar, gbar = point['fbar'], point['gbar']
    out['fall_needed'] = (2.0 * tolerance / ((1.0 - fbar) * gbar)) if fbar is not None and gbar else None
    out['fall_measured'] = -point['beta'] if point['beta'] is not None else None
    out['W_false_alarm_part'] = (est['V']['estimate'] - point['W']) if point['W'] is not None else None
    out['outside_slope_falling'] = est['outside_slope']['interval'] is not None and est['outside_slope']['interval'][1] < 0.0
    def rejected_when(holds_planted):
        return lambda b, k: ((int(b['verdict']['M'][k] == 'REJECT'), 1) if b['view_holds_planted'][k] == holds_planted else (0, 0))

    def repaired_when(holds_fallback):
        return lambda b, k: ((int(b['kept_sound'][k]), 1) if b['faulty'][k] and b['fallback_fault'][k] == holds_fallback else (0, 0))

    def damaged_when(holds_fallback):
        return lambda b, k: ((int(not b['kept_sound'][k]), 1) if not b['faulty'][k] and b['fallback_fault'][k] == holds_fallback else (0, 0))

    out['rejection_by_round'] = {'clean': _by_round(pairs, rejected_when(False)),
                                 'holding_planted': _by_round(pairs, rejected_when(True))}
    out['mean_view_by_round'] = _by_round(pairs, lambda b, k: (b['view'][k], 1))
    out['repair_by_round'] = _by_round(pairs, lambda b, k: (int(b['kept_sound'][k]) if b['faulty'][k] else 0, int(b['faulty'][k])))
    out['damage_by_round'] = _by_round(pairs, lambda b, k: (int(not b['kept_sound'][k]) if not b['faulty'][k] else 0, int(not b['faulty'][k])))
    out['repair_by_fallback'] = {
        'register_clean': {'repair': _rate(*_sum2(pairs, repaired_when(False))), 'damage': _rate(*_sum2(pairs, damaged_when(False)))},
        'register_holds_fallback_fault': {'repair': _rate(*_sum2(pairs, repaired_when(True))), 'damage': _rate(*_sum2(pairs, damaged_when(True)))}}
    out['false_rejection_by_view_fault'] = {
        'view_clean_of_unreadable_admissions': _rate(*_sum2(pairs, lambda b, k: (int(b['verdict']['M'][k] == 'REJECT'), 1) if not b['faulty'][k] and not b['fault_on_unreadable'][k] else (0, 0))),
        'view_holds_fault_admitted_on_unreadable': _rate(*_sum2(pairs, lambda b, k: (int(b['verdict']['M'][k] == 'REJECT'), 1) if not b['faulty'][k] and b['fault_on_unreadable'][k] else (0, 0)))}
    out['changed_beyond_planted'] = _rate(*_sum2(pairs, lambda b, k: (int(bool(b['changed_beyond_planted'][k])), 1) if b['changed_beyond_planted'][k] is not None else (0, 0)))
    out['sound_kept_by_round'] = {'in': _by_round(pairs, lambda b, k: (int(b['kept_sound'][k]), 1)),
                                  'out': _by_round(pairs, lambda b, k: (int(b['v_out'][k] == 1), 1))}
    service = {}
    for arm in ('out', 'in'):
        rows = []
        for k in range(ROUNDS):
            faults_kept_out = refused_sound = faults = sounds = 0
            for ps in pairs.values():
                for pair in ps:
                    for b in pair['blocks']:
                        if arm == 'out':
                            kept = b['v_out'][k] != 0
                            if b['faulty'][k]:
                                faults += 1; faults_kept_out += int(not kept)
                            else:
                                sounds += 1; refused_sound += int(not kept)
                        else:
                            if b['faulty'][k]:
                                faults += 1; faults_kept_out += int(b['kept_sound'][k])
                            else:
                                sounds += 1; refused_sound += int(not b['kept_sound'][k])
            rows.append({'service': _rate(faults_kept_out, faults) - _rate(refused_sound, sounds) if faults and sounds else None,
                         'false_rejection': _rate(refused_sound, sounds)})
        positive = [(math.log(k + 2), math.log(r['service'])) for k, r in enumerate(rows) if r['service'] is not None and r['service'] > 0]
        excluded = sum(1 for r in rows if r['service'] is None or r['service'] <= 0)
        elasticity = _weighted_slope([x for x, _ in positive], [y for _, y in positive], [1.0] * len(positive)) if len(positive) >= 2 else None
        service[arm] = {'by_round': rows, 'path_elasticity': elasticity, 'rounds_excluded': excluded}
    out['correction_service'] = service
    out['by_class'] = _by_group(pairs, 'class', config)
    out['by_catch'] = _by_group(pairs, 'catch', config)
    out['raw_rejection_by_view'] = {role: _rates_by_view(pairs, role) for role in ('M', 'T')}
    out['unparseable_by_view'] = {role: _unusable_by_view(pairs, role) for role in ('M', 'T')}
    out['invalid_corrector_by_round'] = _by_round(pairs, lambda b, k: (int(b['unusable']['C'][k]), 1))
    out['invalid_corrector_reasons'] = _reasons(pairs, 'C')
    out['cap_share_by_arm'] = {role: _cap_share(pairs, role) for role in ('C', 'M', 'T', 'W', 'G', 'I')}
    out['cap_share_by_view'] = {role: _cap_share_by_view(pairs, role) for role in ('C', 'M', 'T', 'W', 'G', 'I')}
    # The third manipulation check of section 7.10: the share of valid corrector returns whose code is the arriving
    # case's is 1 by the reading rule (a return for another case is invalid, wrong_case), printed with the invalid share.
    valid_c, calls_c = _sum2(pairs, lambda b, k: (int(not b['unusable']['C'][k]), 1))
    out['corrector_answered_arriving'] = {'share_of_valid_returns': 1.0 if valid_c else None,
                                          'valid_returns': valid_c, 'calls': calls_c,
                                          'invalid_share': _rate(calls_c - valid_c, calls_c)}
    out['refusals'] = {role: dict(collections.Counter(c for ps in pairs.values() for p in ps for b in p['blocks'] if role in b['arms'] for cats in b['refusals'][role] for c in cats)) for role in ('C', 'M', 'T', 'W', 'G', 'I')}
    out['tokens_by_arm'] = {role: [sum(t[i] for ps in pairs.values() for p in ps for b in p['blocks'] if role in b['arms'] for t in b['tokens'][role]) for i in range(3)] for role in ('C', 'M', 'T', 'W', 'G', 'I')}
    out['reasks'] = {role: _reask(pairs, role) for role in ('C', 'M', 'T', 'W', 'G', 'I')}
    out['unusable_share'] = {e: {role: _rate(sum(sum(b['unusable'][role]) for p in ps for b in p['blocks'] if role in b['arms']),
                                             sum(len(b['unusable'][role]) for p in ps for b in p['blocks'] if role in b['arms']))
                                 for role in ('C', 'M', 'T', 'W', 'G', 'I')} for e, ps in pairs.items()}
    out['free_blocks'] = {e: {arm: _rate(sum(1 for row in rows for f in row['first_unusable'] if f[arm] is not None), 2 * len(rows))
                              for arm in ('C', 'M', 'W')} for e, rows in pv.items()}
    out['first_unusable_round'] = {e: {arm: dict(collections.Counter(f[arm] + 1 for row in rows for f in row['first_unusable'] if f[arm] is not None))
                                       for arm in ('C', 'M', 'W')} for e, rows in pv.items()}
    out['generating'] = _secondary(pairs, pv, config, 'G', level)
    out['informed'] = _secondary(pairs, pv, config, 'I', level)
    out['thirds'] = thirds(pairs, pv, level)
    out['budget'] = budget_share(pairs)
    return out


def _sum2(pairs_by_engine, fn):
    num = den = 0
    for pairs in pairs_by_engine.values():
        for pair in pairs:
            for b in pair['blocks']:
                for k in range(ROUNDS):
                    n, d = fn(b, k)
                    num += n
                    den += d
    return num, den


def _by_group(pairs, key, config):
    """The registered fall, rejection and repair, apart for each class of planted departure (A, B, T) or each group by
    what catching takes; sound drafts enter every group's false alarms."""
    groups = sorted({g for ps in pairs.values() for p in ps for b in p['blocks'] for g in b[key] if g})
    out = {}
    for group in groups:
        falls = {}
        for e, ps in pairs.items():
            totals = {}
            for p in ps:
                for b in p['blocks']:
                    for k in range(ROUNDS):
                        if b['unusable']['T'][k] or (b['faulty'][k] and b[key][k] != group):
                            continue
                        row = totals.setdefault(b['view'][k], [0, 0, 0, 0])
                        rejected = b['verdict']['T'][k] == 'REJECT'
                        if b['faulty'][k]:
                            row[0] += 1; row[1] += int(rejected)
                        else:
                            row[2] += 1; row[3] += int(rejected)
            falls[e] = fall_from_totals(totals, config['fall']['min_calls'], config['fall']['f_cap'])
        betas = [f['beta'] for f in falls.values()]
        out[group] = {'fall': None if any(v is None for v in betas) else -stats.mean(betas),
                      'monitor_rejection': _rate(*_sum2(pairs, lambda b, k, g=group: (int(b['verdict']['M'][k] == 'REJECT'), 1) if b['faulty'][k] and b[key][k] == g else (0, 0))),
                      'repair': _rate(*_sum2(pairs, lambda b, k, g=group: (int(b['kept_sound'][k]), 1) if b['faulty'][k] and b[key][k] == g else (0, 0)))}
    return out


def _rates_by_view(pairs, role):
    out = {}
    for ps in pairs.values():
        for p in ps:
            for b in p['blocks']:
                for k in range(ROUNDS):
                    row = out.setdefault(b['view'][k], [0, 0, 0, 0])
                    rejected = b['verdict'][role][k] == 'REJECT'
                    if b['faulty'][k]:
                        row[0] += 1; row[1] += int(rejected)
                    else:
                        row[2] += 1; row[3] += int(rejected)
    return {n: {'faulty_rejection': _rate(r[1], r[0]), 'sound_rejection': _rate(r[3], r[2]), 'calls': r[0] + r[2]} for n, r in sorted(out.items())}


def _unusable_by_view(pairs, role):
    out = {}
    for ps in pairs.values():
        for p in ps:
            for b in p['blocks']:
                for k in range(ROUNDS):
                    row = out.setdefault(b['view'][k], [0, 0])
                    row[0] += int(b['unusable'][role][k]); row[1] += 1
    return {n: _rate(a, c) for n, (a, c) in sorted(out.items())}


def _reasons(pairs, role):
    counter = collections.Counter()
    for ps in pairs.values():
        for p in ps:
            for b in p['blocks']:
                if role in b['arms']:
                    for reasons in b['reasons'][role]:
                        counter.update(r or 'unknown' for r in reasons)
    return dict(counter)


def _cap_share(pairs, role):
    hit = total = 0
    for ps in pairs.values():
        for p in ps:
            for b in p['blocks']:
                if role in b['arms']:
                    hit += sum(b['truncated'][role]); total += sum(b['asks'][role])
    return _rate(hit, total)


def _cap_share_by_view(pairs, role):
    """The share of a role's asks that reached the 16,000-token allowance, by the number of entries the call showed:
    for the verdict roles the entries in their own view, for the corrector and the generating arm the entries of its
    register plus the arriving case, which is the round (section 6.5)."""
    out = {}
    for ps in pairs.values():
        for p in ps:
            for b in p['blocks']:
                if role not in b['arms']:
                    continue
                for k in range(ROUNDS):
                    if role in ('M', 'T'):
                        size = b['view'][k]
                    elif role == 'I':
                        size = b['view_I'][k]
                    elif role == 'W':
                        size = b['view_W'][k]
                    else:
                        size = k + 1
                    row = out.setdefault(size, [0, 0])
                    row[0] += b['truncated'][role][k]
                    row[1] += b['asks'][role][k]
    return {str(n): _rate(hit, total) for n, (hit, total) in sorted(out.items(), key=lambda item: (item[0] is None, item[0] or 0))}


def _reask(pairs, role):
    calls = reasked = second = third = 0
    by_round = [0] * ROUNDS
    for ps in pairs.values():
        for p in ps:
            for b in p['blocks']:
                if role not in b['arms']:
                    continue
                for k in range(ROUNDS):
                    calls += 1
                    used = b['asks'][role][k]
                    if used > 1:
                        reasked += 1; by_round[k] += 1
                        if not b['unusable'][role][k]:
                            if used == 2:
                                second += 1
                            else:
                                third += 1
    return {'calls': calls, 're_asked': reasked, 'usable_at_second': _rate(second, reasked),
            'usable_at_third': _rate(third, reasked), 're_asked_by_round': by_round}


def _secondary(pairs, pv, config, role, level):
    limit = config['secondary_pairs']
    rows = {e: [row for row in rs[:limit]] for e, rs in pv.items()}
    if role == 'G':
        if not all(any('G_slope' in row for row in rs) for rs in rows.values()):
            return None
        sel = {e: [row for row in rs if 'G_slope' in row] for e, rs in rows.items()}
        if any(len(v) < 2 for v in sel.values()):
            return None
        by_round = _by_round({e: ps[:limit] for e, ps in pairs.items()},
                             lambda b, k: ((b['v_G'][k] or 0) if 'G' in b['arms'] else 0, 1 if 'G' in b['arms'] else 0))
        return {'value_by_round': by_round, 'slope': _interval_row({e: [r['G_slope'] for r in v] for e, v in sel.items()}, 0.95),
                'gap_slope': _interval_row({e: [r['G_gap'] for r in v] for e, v in sel.items()}, 0.95)}
    sel = {e: [row for row in rs if 'I_gap' in row] for e, rs in rows.items()}
    if any(len(v) < 2 for v in sel.values()):
        return None
    sub = {e: ps[:limit] for e, ps in pairs.items()}
    informed_fall = {}
    twin_fall = {}
    for e, ps in sub.items():
        informed_fall[e] = fall_from_totals(_sum_counts(_fall_counts(b, 'informed') for p in ps for b in p['blocks'] if 'I' in b['arms']),
                                            config['fall']['min_calls'], config['fall']['f_cap'])
        twin_fall[e] = fall_from_totals(_sum_counts(_fall_counts(b, 'main') for p in ps for b in p['blocks']),
                                        config['fall']['min_calls'], config['fall']['f_cap'])
    bi = [f['beta'] for f in informed_fall.values()]
    bt = [f['beta'] for f in twin_fall.values()]
    difference = None if any(v is None for v in bi + bt) else stats.mean(bi) - stats.mean(bt)
    return {'value_by_round': _by_round(sub, lambda b, k: ((b['v_I'][k] or 0) if 'I' in b['arms'] else 0, 1 if 'I' in b['arms'] else 0)),
            'gap_slope': _interval_row({e: [r['I_gap'] for r in v] for e, v in sel.items()}, 0.95),
            'curves': {'informed': {e: f.get('curve') for e, f in informed_fall.items()}, 'twin': {e: f.get('curve') for e, f in twin_fall.items()}},
            'fall_difference_informed_minus_twin': difference}


def thirds(pairs, pv, level):
    """The mark's quantity D by thirds of the run in time: each engine's pairs ranked by completion time and split into
    thirds, pooled as section 7.2 pools them, with 95 per cent intervals and no label."""
    out = []
    for third in range(3):
        values = {}
        for e, ps in pairs.items():
            order = sorted(range(len(ps)), key=lambda i: (ps[i]['completed_utc'] or '', i))
            size = len(order)
            chosen = order[third * size // 3:(third + 1) * size // 3]
            values[e] = [pv[e][i]['D'] for i in chosen]
        if any(len(v) < 2 for v in values.values()):
            out.append(None)
            continue
        out.append(_interval_row(values, 0.95))
    return out


# ---- The stopping rule (section 7.4) and the sealing of a look.

def decide(record):
    """'stop' or 'continue': the lane stops at the first look if the mark there is NOT EVALUABLE, SUPPORTED or
    REFUTED; otherwise it runs the second stage. At the second look it stops. The registered stopping word is 'mark'
    (section 7.4); were it 'both', the one word section 7.4 names, a decided mark would also need the mechanism axis
    decided (supported or unsupported) to stop at the first look, with the same largest size and caps."""
    if record['look'] == 2:
        return 'stop'
    if record['mark'] == NOT_EVALUABLE:
        return 'stop'
    if record['mark'] not in (SUPPORTED, REFUTED):
        return 'continue'
    if record.get('stopping_word', 'mark') == 'both':
        axis = (record.get('mechanism') or {}).get('verdict')
        return 'stop' if axis in ('supported', 'unsupported') else 'continue'
    return 'stop'


def incomplete(config, number, stops, completed):
    """The record of a look not taken: NOT EVALUABLE, administratively incomplete, before the first look; the first
    look's INCONCLUSIVE with the reason that the second look was not reached, between the looks."""
    return {'study_id': config['study_id'], 'look': number, 'mark': NOT_EVALUABLE if number == 1 else INCONCLUSIVE,
            'reason': 'administratively_incomplete' if number == 1 else 'second_look',
            'reason_text': 'administratively incomplete' if number == 1 else REASONS['second_look'],
            'stops': stops, 'completed_pairs': completed, 'utc': now()}


def json_safe(value):
    """The record as JSON can hold it: a number that is not finite (a degree of freedom or a standard error left
    undefined by a degenerate interval) becomes null, a tuple a list and a key text; nothing else changes."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def seal_record(root, record, name):
    """Write a look's record where no command prints it and return its SHA-256; the harness prints only that hash and
    the decision (sections 6.4 and 7.4). (The build list calls it analysis.seal; this module imports the module seal
    under that name.)"""
    path = Path(root) / 'looks' / (name + '.json')
    save(path, json_safe(record), exclusive=True)
    return digest(path.read_bytes())


def load_arrays(root, config, tier, engines, n_pairs):
    """The arrays of pairs 1 to n_pairs of each engine: the pair logs joined to the score records."""
    root = Path(root)
    seed = seal.bank_seed(root, config)
    scores = load(root / 'scores' / 'score_records.json')
    logs = {engine: collect.pair_logs(root, engine, n_pairs) for engine in engines}
    required = {}
    for engine in engines:
        bank = seal.load_bank(root, tier, engine)
        for pair in bank['pairs'][:n_pairs]:
            for case in pair['cases']:
                required[case['case_id']] = forms.required(case)
    return values(logs, scores, seed, required)


def report(root, config):
    """`analysis.report(root)` of the build list (section 12, package G): the look records, the printed sentence of
    section 10 chosen by the mark, the axis and the labels, the confirmation values, the tables of section 6.5 and, for
    any repair made after a look, whether the public sets showed the defect. The words live in sentences.py, which this
    calls; the import is made here so that sentences.py may import this module."""
    from . import sentences
    return sentences.report(root, config)
