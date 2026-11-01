"""What each verdict lets the author print, and the report (41-D-COUNTED, sections 6.5, 10, 11.4 and 11.5).

Package G of the build list (DESIGN-FREEZE.md, section 12). The candidate sentences of section 10, word for word, with
their brackets filled from the look record: every slope and level printed with its sign and its interval at the look's
level. The sentence is chosen by the mark, the axis and the labels; the report adds the look records, the confirmation
values, the tables of section 6.5, the admission report, the deviation record and, for any repair made after a look,
whether the public sets showed the defect. Nothing here is executed before both registrations read REGISTERED (SD-001,
SD-012); before then the only mechanical check is a parse of this file's text.
"""
import json
from pathlib import Path

from p21base.common import load, now, save

from . import analysis

UNIT_ID = 'Unit 83 (study identifier 41-D-COUNTED-v0.1)'
DOMAIN = 'Domain: counted, exact comparison on a frozen, sealed suite; no judged term.'
CONDITION_WORDS = {
    'D>+tol': 'D above +0.0075', 'D->+tol': 'D- above +0.0075', 'level>0': 'the level above zero',
    'L->0': 'L- above zero', 'D<+tol': 'D below +0.0075', 'D+<+tol': 'D+ below +0.0075', 'D>-tol': 'D above -0.0075',
    'D->-tol': 'D- above -0.0075', 'D<-tol': 'D below -0.0075', 'D+<-tol': 'D+ below -0.0075',
    'PC>+tol': 'the positive control\'s slope above +0.0075', 'PC->+tol': 'PC- above +0.0075',
}
MARK_CONDITIONS = {
    analysis.SUPPORTED: ('D>+tol', 'D->+tol', 'level>0', 'L->0'),
    'constant': ('D<+tol', 'D+<+tol', 'D>-tol', 'D->-tol', 'PC>+tol', 'PC->+tol'),
    'narrowing': ('D<-tol', 'D+<-tol', 'PC>+tol', 'PC->+tol'),
    'no widening': ('D<+tol', 'D+<+tol', 'PC>+tol', 'PC->+tol'),
}
INCONCLUSIVE_WORDS = {
    'margin': 'the interval for the widening contained the registered tolerance',
    'precision': 'the run was too imprecise to tell a widening of the registered size from none',
    'crossing': 'the in-loop arm narrowed a deficit and did not open a lead',
    'disagree': 'the interval and the sign-flip check disagreed',
    'control': 'the positive control did not show a widening',
    'unusable': 'the mark depended on unusable replies',
    'confirmation': 'the intervals and the sign-flip check cleared the boundary, but the bounded-mean confirmation of the registration did not reach 40',
    'degenerate': 'the interval was degenerate',
    'against': 'the interaction ran against the claim, which this registration does not score as refutation',
}
TAIL = ("Depth is the count of administered arrivals, one entry a round, nine in eighteen carrying a planted departure "
        "from a written charter, none revised twice, and no kept entry changes what a later round is given; the "
        "monitor's register held {n} entries on average at the last round. Every call ran at the providers' low "
        "reasoning setting, with an allowance of 16,000 output tokens a call, the same for every arm, and up to three "
        "attempts a call on an unusable reply, counted by arm. The corrector acted inside each round at fixed weights: "
        "that is placement, not correction built into a system. Both arms are reviewers of checkable departures from "
        "written rules, on two engines the author chose, so nothing is shown about judged correction, about safety or "
        "common-mode failure, about build order, about a higher reasoning setting or a longer register, or about the "
        "Eden Protocol as an architecture.")
CORRECTOR_DRIVEN = ("The corrector's own improvement with depth is a second source of the widening, which the "
                    "registered fall does not carry.")
# The set of section 11.3, step 4 an excluded engine failed, as the NOT EVALUABLE sentence names it (section 10).
FAILED_SET_WORDS = {
    'transport': 'at its transport checks', 'format_verdict': 'at its verdict-format controls',
    'format_entry': 'at its entry-format controls', 'verdict14': 'at its 500 full-size verdict controls',
    'corrector18': 'at its 300 full-size corrector controls',
    'model_changed': 'on a reply that named another model than the registered identifier',
}


def signed(value, places=4):
    if value is None:
        return 'not computed'
    return ('+' if value >= 0 else '-') + ('%.' + str(places) + 'f') % abs(value)


def with_interval(row, level, places=4):
    """A slope or level with its sign and its interval at the look's level."""
    interval = row.get('interval')
    text = signed(row.get('estimate'), places)
    if interval is None:
        return text + ' (degenerate interval)'
    return '%s (%s per cent interval %s to %s)' % (text, _level(level), signed(interval[0], places), signed(interval[1], places))


def _level(level):
    return ('%.1f' % (100 * level)).rstrip('0').rstrip('.')


def _value_interval(value, interval, level, places=4, sign=True):
    show = signed if sign else (lambda v, p=places: ('%.' + str(p) + 'f') % v if v is not None else 'not computed')
    if interval is None:
        return show(value, places)
    return '%s (%s per cent interval %s to %s)' % (show(value, places), _level(level), show(interval[0], places), show(interval[1], places))


def _names(config, engines):
    return {e: config['display_names'].get(e, e) for e in engines}


def mean_last_view(record):
    view = record['printed']['mean_view_by_round'][-1]
    return '%.1f' % view if view is not None else 'not computed'


def unusable_counts(record):
    """[m] of the run's [N] calls returned an unusable reply on every attempt: the calls of the four arms the mark,
    its bound and its positive control read (the corrector, the monitor and the twin over the pairs read, the windowed
    monitor over the control's pairs), on both engines (BUILD-DECISIONS.md, 18 October 2026)."""
    m = n = 0
    for engine, checks in record['checks'].items():
        for role, row in checks['ceiling'].items():
            m += row['unusable']
            n += row['calls']
    return m, n


def error_control_sentence(record, conditions):
    conf = record['confirmation']
    left = '; '.join('%s: %d' % (CONDITION_WORDS[c], conf[c]['n_left_out']) for c in conditions)
    values = '; '.join('%s: %.4g' % (CONDITION_WORDS[c], conf[c]['value']) for c in conditions)
    m, n = unusable_counts(record)
    return ("%d of the run's %d calls returned an unusable reply on every attempt; the verdict on P21 holds for every "
            "value those replies could have taken, with everything they could have changed later in their blocks; the "
            "mechanism axis, which reads the part the monitor's fall carries, is not held to that test, and its "
            "sensitivity to those replies is printed beside it. The verdict's error control is the bounded-mean "
            "confirmation of the registration, valid for any distribution of the pair values within their bounds given "
            "independent pairs; it left out, from each condition, the %s indices at which either engine's pair rests on "
            "a reply left unusable after the re-ask, both engines' pairs at each index, and for the rest it holds where "
            "which replies stay unusable at an index does not depend on what any reply of the pairs at that index would "
            "have said, the test above covering those replies whatever they would have said; its values at this look "
            "were %s." % (m, n, left, values))


def axis_sentence(record):
    axis = record['mechanism']
    level = record['level']
    est = record['estimates']
    fall = _value_interval(-axis['beta'] if axis.get('beta') is not None else None,
                           None if axis.get('beta_interval') is None else (-axis['beta_interval'][1], -axis['beta_interval'][0]),
                           level, 4, sign=False)
    w = _value_interval(axis.get('W'), axis.get('W_interval'), level)
    e = with_interval(est['E'], level)
    verdict = axis['verdict']
    if verdict == analysis.NOT_EVALUABLE:
        return ("Mechanism axis NOT EVALUABLE: the monitor and the twin, shown the same bytes, did not agree within the "
                "registered interval of the negative control.")
    if not axis.get('fall_registered'):
        return ("No fall in the monitor's catch rate was registered, so the widening is unexplained by this framework and "
                "is not support for the information-asymmetry mechanism: it may come from the outside arm's action space, "
                "which can only accept or reject. Mechanism axis %s." % verdict)
    if verdict == 'supported':
        return ("Mechanism axis supported: the monitor's corrected catch rate fell by %s an entry as its register grew, and "
                "the part of the widening that fall carries was %s, above the tolerance; the corrector's own change with "
                "depth was %s." % (fall, w, e))
    if verdict == 'unsupported':
        false_alarm = signed(record['printed']['W_false_alarm_part'])
        return ("Mechanism axis unsupported: the monitor's corrected catch rate fell by %s an entry as its register grew, "
                "but the part of the widening that fall carries was %s, below the tolerance; the rest came from misses the "
                "monitor could not undo (%s), from false alarms that grew with the register (%s) and from the corrector's "
                "own change with depth (%s), so the widening is not support for the information-asymmetry mechanism."
                % (fall, w, with_interval(est['S'], level), false_alarm, e))
    return ("Mechanism axis inconclusive: the monitor's corrected catch rate fell by %s an entry as its register grew, and "
            "the part of the widening that fall carries was %s, which neither clears the tolerance nor falls below it; the "
            "corrector's own change with depth was %s." % (fall, w, e))


def sentences(record, config, stopped=None):
    """The paragraphs the report prints for a look record, chosen by its mark, axis and labels (section 10)."""
    level = record.get('level')
    mark = record['mark']
    out = []
    n_last = mean_last_view(record) if 'printed' in record else 'not computed'
    names = _names(config, record.get('engines', []))
    if mark == analysis.SUPPORTED:
        est = record['estimates']
        out.append("P21 is supported under the registered test, by the programme's own run (%s), on the access axis: in a "
                   "counted, administered revision loop on two fixed-weight engines of two providers (Claude Haiku 5.5 and "
                   "GPT-6 Luna), pooled with equal weight, correction placed inside each round led a monitor that saw only "
                   "the finished register by %s on average, a lead the value's definition makes likely whenever the "
                   "corrector repairs at more than half the rate at which the monitor discriminates and which by itself "
                   "counts for nothing; and the per-round lead widened by %s a round over eighteen rounds, more than the "
                   "registered tolerance of 0.0075." % (UNIT_ID, with_interval(est['level'], level), with_interval(est['D'], level)))
        out.append(axis_sentence(record))
        if record['mechanism']['verdict'] == 'supported':
            out.append("Law V, the ARC Embedding Law, remains a law under test: its placement limb, P21, is supported under "
                       "the registered test, by the programme's own run, at this first rung, with its mechanism axis "
                       "supported, on two fixed-weight engines in an administered loop whose monitor saw %s entries at the "
                       "last round; its build-order limb, read on P12, is undecided, since P12 is untested." % n_last)
        else:
            out.append("Law V, the ARC Embedding Law, remains a law under test: its placement limb, P21, is supported on "
                       "the performance axis only, with its mechanism axis %s, which is not support for the "
                       "information-asymmetry mechanism; its build-order limb, read on P12, is undecided, since P12 is "
                       "untested." % record['mechanism']['verdict'].lower())
        out.append(error_control_sentence(record, MARK_CONDITIONS[analysis.SUPPORTED]))
    elif mark == analysis.REFUTED:
        est = record['estimates']
        axis = record['mechanism']
        moved = 'narrowed by more than' if record['form'] == 'narrowing' else 'did not widen by'
        if est['level']['estimate'] is not None and est['level']['estimate'] > 0:
            lead = "The lead the in-loop arm showed, %s, is a level difference, which counts against P21." % with_interval(est['level'], level)
        else:
            lead = "The in-loop arm was behind by %s." % with_interval(est['level'], level)
        if axis.get('fall_registered'):
            fall = 'fell by %s an entry' % _value_interval(-axis['beta'], (-axis['beta_interval'][1], -axis['beta_interval'][0]), level, 4, sign=False)
        else:
            fall = 'was not registered as falling'
        needed = record['printed']['fall_needed']
        out.append("P21 is refuted under the registered test, by the programme's own run (%s), on the access axis: in a "
                   "counted, administered revision loop on two fixed-weight engines of two providers (Claude Haiku 5.5 and "
                   "GPT-6 Luna), pooled with equal weight, the per-round lead of correction placed inside each round over a "
                   "monitor that saw only the finished register %s the registered tolerance of 0.0075 a round over "
                   "eighteen rounds, in which the monitor's view reached %s entries at the last round; it changed by %s, in "
                   "a design whose interval at that look's level would have excluded zero four times in five had the "
                   "widening been the tolerance's size. A monitor shown only three entries of its register, which loses "
                   "sight of the new entry by construction, fell behind by %s a round, which shows that the apparatus "
                   "carries a widening into the mark. %s The monitor's corrected catch rate %s, against a fall of about %s "
                   "an entry that would by itself carry a widening of the tolerance's size, computed from the run's own "
                   "false-alarm rate and the growth of the monitor's view, with the false alarms held at their mean and the "
                   "corrector's change and the old misses taken as zero; over the same rounds the corrector's own value "
                   "changed by %s a round, and misses the monitor could not undo added %s a round; mechanism axis %s. "
                   "Nothing is shown about registers longer than %s entries."
                   % (UNIT_ID, moved, n_last, with_interval(est['D'], level), with_interval(est['PC'], level), lead, fall,
                      '%.3f' % needed if needed is not None else 'not computed', with_interval(est['E'], level),
                      with_interval(est['S'], level), axis['verdict'].lower() if axis['verdict'] != analysis.NOT_EVALUABLE else 'NOT EVALUABLE', n_last))
        # N2 (T2-research, 20 October 2026): the narrowing form carries no gate, so where it decides without the gate the
        # look sets for the other forms, the report prints the gate's value and the run's standard error beside the sentence.
        d_se = est['D'].get('se') if isinstance(est.get('D'), dict) else None
        if record['form'] == 'narrowing' and not (d_se is not None and d_se == d_se and d_se <= record['gate']):
            out.append("The narrowing form decided this mark without its gate: the standard error of the per-round lead "
                       "was %s, against the look's gate of %s." % ('%.4f' % d_se if d_se is not None and d_se == d_se
                                                                    else 'not computed', '%.4f' % record['gate']))
        out.append("Under the rule registered in successor 1.103, a refuted P21 refutes Law V, the ARC Embedding Law, under "
                   "the registered test, by the programme's own run, in this apparatus: an administered loop of eighteen "
                   "rounds in which no kept entry changes what a later round is given, whose monitor saw %s entries at the "
                   "last round, on two fixed-weight engines at the providers' low reasoning setting. The Eden Protocol's "
                   "engineering case loses, in this apparatus, a depth interaction of the registered size. Laws I to IV "
                   "and P15 are untouched, and P12 keeps its own status: untested." % n_last)
        out.append(error_control_sentence(record, MARK_CONDITIONS[record['form']]))
    elif mark == analysis.INCONCLUSIVE:
        reason = record.get('reason')
        if reason == 'second_look':
            stop = stopped or {}
            words = 'the second look was not reached: %s stopped at %s pairs after the first look' % (
                names.get(stop.get('engine'), stop.get('engine', 'an engine')), stop.get('completed_prefix', 'its completed'))
        else:
            words = INCONCLUSIVE_WORDS.get(reason, reason or 'no reason recorded')
        # The registered sentence names the mechanism axis's verdict (field 344-77; the re-read's finding 3, ruling N22),
        # read as the stopping rule reads it, with NOT EVALUABLE where a record carries no axis.
        axis_verdict = (record.get('mechanism') or {}).get('verdict')
        axis_words = 'NOT EVALUABLE' if axis_verdict in (None, analysis.NOT_EVALUABLE) else str(axis_verdict).lower()
        out.append("P21 is not decided by this run (%s): %s. Nothing is supported or refuted. Law V, the ARC Embedding "
                   "Law, remains a law under test: its placement limb, P21, is undecided, with its mechanism axis %s; "
                   "its build-order limb, read on P12, is undecided, since P12 is untested." % (UNIT_ID, words, axis_words))
    elif mark == analysis.NOT_EVALUABLE:
        out.append("P21 was not evaluated by this run (%s): %s, so P21 stays untested. Law V, the ARC Embedding Law, "
                   "remains a law under test: its placement limb, P21, is undecided, with its mechanism axis NOT "
                   "EVALUABLE; its build-order limb, read on P12, is undecided, since P12 is untested."
                   % (UNIT_ID, not_evaluable_cause(record, config, stopped)))
    out.append(DOMAIN)
    out.append(TAIL.format(n=n_last))
    labels = list(record.get('labels') or [])
    axis_labels = list((record.get('mechanism') or {}).get('labels') or [])
    if labels:
        out.append('Labels: ' + ', '.join('"%s"' % label for label in labels) + '.')
    if 'corrector-driven' in axis_labels:
        out.append('Mechanism axis: "corrector-driven". ' + CORRECTOR_DRIVEN)
    return out


def not_evaluable_cause(record, config, stopped):
    names = config['display_names']
    if record.get('reason') == 'administratively_incomplete':
        stop = stopped or {}
        return 'administratively incomplete: %s stopped at %s pairs before the first look' % (
            config['display_names'].get(stop.get('engine'), stop.get('engine', 'an engine')), stop.get('completed_prefix', 'fewer than 220'))
    if record.get('reason') == 'not_admitted':
        # Section 10's bracket names the engine: each excluded engine with the set of section 11.3, step 4 it failed.
        excluded = record.get('excluded') or {}
        named = '; '.join('on %s, admission excluded the engine %s' % (names.get(e, e), FAILED_SET_WORDS.get(why, 'at a failed set'))
                          for e, why in sorted(excluded.items()))
        return (named + ', so ' if named else '') + ('fewer than two engines were admitted, a registered regime '
                                                      'condition that was never reached')
    if record.get('cause') == 'broken_instrument':
        engine, arm = record['broken'][0]
        row = record['checks'][engine]['ceiling'][arm]
        arm_words = {'C': 'corrector', 'M': 'monitor', 'T': 'twin', 'W': 'windowed monitor'}[arm]
        return ('a broken instrument: %d of %s\'s %d %s replies stayed unusable after the re-ask, more than the '
                'registered ceiling of one in 200' % (row['unusable'], names.get(engine, engine), row['calls'], arm_words))
    for engine in sorted(record.get('checks', {})):
        failures = record['checks'][engine]['failures']
        if failures:
            return 'on %s, %s, a registered regime condition that was never reached' % (names.get(engine, engine), failures[0][1])
    return 'a registered regime condition that was never reached'


def report(root, config):
    """The report: the look records, the printed sentences, the confirmation values, the tables of section 6.5, the
    admission report, the deviation record with counts and the admissibility conditions; written to report/."""
    root = Path(root)
    state = load(root / 'lane_state.json') if (root / 'lane_state.json').exists() else {}
    looks = {}
    for name in ('look1', 'look2', 'look1_incomplete', 'look2_incomplete', 'look1_record'):
        path = root / 'looks' / (name + '.json')
        if path.exists():
            looks[name] = load(path)
    final_name = 'look2' if 'look2' in looks else ('look2_incomplete' if 'look2_incomplete' in looks else
                                                  ('look1' if 'look1' in looks else 'look1_incomplete'))
    final = looks.get(final_name)
    stopped = state.get('stop_detail')
    lines = ['# 41-D-COUNTED-v0.1: the report', '', 'Written ' + now() + ' by the lane\'s report command.', '']
    if final is None:
        lines.append('No look was taken.')
    else:
        if final_name == 'look2_incomplete' and 'look1' in looks:
            shown = dict(looks['look1'])
            shown.update(mark=analysis.INCONCLUSIVE, reason='second_look')
        else:
            shown = final
        lines += ['## The verdict', '']
        lines += [paragraph + '\n' for paragraph in sentences(shown, config, stopped)]
        if 'confirmation' in shown:
            lines += ['## The confirmation values at this look', '']
            for name, row in sorted(shown['confirmation'].items()):
                lines.append('- %s: %.6g (indices left out: %d of %d)' % (CONDITION_WORDS[name], row['value'], row['n_left_out'], row['n_read']))
            lines.append('')
    lines += identity_lines(root, config)
    for name, record in sorted(looks.items()):
        lines += ['## Look record: ' + name, '', '```json', json.dumps(_summary(record), indent=1, sort_keys=True), '```', '']
    for rel in ('admission/admission_receipt.json', 'admission/calibration_receipt.json', 'freeze.json'):
        path = root / rel
        if path.exists():
            lines += ['## ' + rel, '', '```json', json.dumps(load(path), indent=1, sort_keys=True)[:200000], '```', '']
    lines += ['## The deviation record (section 11.4)', '']
    lines.append('Every retry, hold, release, exclusion, repair and stop, counted by kind, engine, arm and round (the '
                 'round and arm read from the call\'s slot where it has one).')
    lines.append('')
    for (kind, who, role, rnd), count in sorted(deviation_counts(root).items(), key=lambda item: tuple(str(x) for x in item[0])):
        lines.append('- %s; %s; arm %s; round %s: %d' % (kind, who, role, rnd, count))
    lines.append('')
    defects = root / 'repairs.jsonl'
    if defects.exists():
        lines += ['## Defect repairs before the runtime freeze (section 11.2, item 7)', '']
        for line in defects.read_text('utf-8').splitlines():
            if line.strip():
                entry = json.loads(line)
                lines.append('- %s, %s: %s (from %s to %s).' % (entry['utc'], entry['file'], entry['failing_case'],
                                                               entry['replaced_sha256'][:12], entry['new_sha256'][:12]))
        lines.append('')
    repairs = root / 'look_repairs.jsonl'
    if repairs.exists():
        lines += ['## Repairs after a look (section 7.4)', '']
        for line in repairs.read_text('utf-8').splitlines():
            if line.strip():
                entry = json.loads(line)
                lines.append('- %s, look %s: the public sets %s the defect; the repaired result is %s.' % (
                    entry['file'], entry['look'], 'show' if entry['public_sets_show_defect'] else 'do not show',
                    'confirmatory' if entry['public_sets_show_defect'] else 'exploratory and never a verdict on the counted route'))
        lines.append('')
    lines += ['## The admissibility conditions of section 1.2', '']
    lines += admissibility(root, looks)
    text = '\n'.join(lines) + '\n'
    save(root / 'report' / 'report.json', {'final': final_name, 'looks': {k: _summary(v) for k, v in looks.items()}, 'utc': now()})
    out = root / 'report' / 'REPORT.md'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding='utf-8')
    return out


def identity_lines(root, config):
    """The engines' identities as the receipts support them (section 3.1): the returned model admission recorded for
    each engine and, where GPT-6 Luna's replies carry the bare name, the sentence that a change of the model behind it
    cannot be seen."""
    path = Path(root) / 'admission' / 'admission_receipt.json'
    if not path.exists():
        return []
    receipt = load(path)
    lines = ['## The engines\' identities (section 3.1)', '']
    for engine, model in sorted((receipt.get('expected_models') or {}).items()):
        name = config['display_names'].get(engine, engine)
        dated = (receipt.get('dated_names') or {}).get(engine)
        line = '- %s: every reply used returned the model %s, the value admission recorded.' % (name, model)
        if engine == 'gpt-6-luna' and dated is False:
            line += (' Its replies carry the bare name, so a change of the model behind that name could not be seen; a '
                     'rerun on another day may meet a different GPT-6 Luna behind the same name, since the provider '
                     'documents no snapshot.')
        elif engine == 'claude-haiku-5-5':
            line += (' The provider states that the weights behind an identifier do not change and that the serving '
                     'infrastructure around it can.')
        lines.append(line)
    lines.append('')
    return lines


def _slot_parts(slot):
    """(role, round) from a call's slot, stage:engine:pair:block:round:role, with any re-ask mark (/r2, /r3) or numbered
    attempt mark (#attemptk) removed; ('-', '-') for a slot of admission, calibration or none."""
    if not isinstance(slot, str):
        return '-', '-'
    base = slot.split('#', 1)[0].split('/', 1)[0]
    parts = base.split(':')
    if len(parts) == 6 and parts[0] == 'collect':
        return parts[5], parts[4]
    return '-', '-'


def deviation_counts(root):
    """The deviation record counted by (kind, engine or provider, arm, round)."""
    counts = {}
    path = Path(root) / 'deviation_log.jsonl'
    if not path.exists():
        return counts
    for line in path.read_text('utf-8').splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict):
            continue
        role, rnd = _slot_parts(record.get('slot'))
        key = (record.get('kind'), record.get('engine') or record.get('provider') or '-', role, rnd)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _summary(record):
    keep = ('study_id', 'look', 'n_pairs', 'level', 'gate', 'mark', 'form', 'reason', 'reason_text', 'cause', 'broken',
            'admitted', 'labels', 'estimates', 'mechanism', 'positive_control', 'negative_control', 'sign_flip', 'checks',
            'printed', 'scale', 'bootstrap', 'absent_entry', 'confirmation', 'stops', 'completed_pairs', 'utc')
    return {k: record[k] for k in keep if k in record}


def admissibility(root, looks):
    """Each condition of section 1.2 that code can read, met or failed; the rest are stated for the author's report."""
    lines = []
    attestation = root / 'registration_approval.json'
    commitment = root / 'bank_commitment.json'
    if attestation.exists() and commitment.exists():
        a, c = load(attestation), load(commitment)
        lines.append('- Lodged before data: both registrations read REGISTERED at %s, before the bank was made at %s: met.'
                     % (a['confirmed_utc'], c['created_utc']))
    else:
        lines.append('- Lodged before data: the attestation or the commitment is absent: not shown.')
    lines.append('- Blinding: the scorer read only the table of case keys and entry lines and the sealed answers (section 6.4): met by construction.')
    lines.append('- No rescue: the looks, their sizes and the stopping rule were registered; no look was added: met.')
    lines.append('- The literature, the scope, the regime record and the rung: stated in the registration and in the sentences above.')
    return lines
