"""Admission, calibration and the pairs in flight (41-D-COUNTED, section 11.3, steps 4 and 5).

Package H of the build list (DESIGN-FREEZE.md, section 12). Every request is built by the same functions the collector
uses, from the public sets of section 4.8, so that admission exercises the exact bytes the run will send; every item is
read as the main stage reads a reply, each unusable first attempt asked again up to twice (section 11.2, item 8), and
the pass marks apply after the re-ask (revision 4.2). Steps 4 and 5 read only the form of replies, token counts, limit
headers and elapsed time, and the calibration stage's single-entry answers on the public calibration sets; nothing
touches the sealed bank. Nothing here is executed before both registrations read REGISTERED (SD-001, SD-012); before
then the only mechanical check is a parse of this file's text.
"""
import concurrent.futures
import math
import statistics
from pathlib import Path

from p21base import providers
from p21base.common import GateError, load, now, save

from . import charter, collect, forms, generate, prompts, score, seal, stats

TRANSPORT_ROLE = 'X'


def _receipt_path(root, name):
    return Path(root) / 'admission' / name


def _transport(ctx, engine):
    """Three transport checks with the base's transport text; two must return ok with the registered identifier; the
    limit headers of the first reply are recorded. A reply whose returned model is not the registered identifier (on
    GPT-6 Luna, the bare name or a dated name that extends it) comes back from the client as model_changed, so `ok`
    carries the identity check. Whether GPT-6 Luna's replies carry a dated name or the bare name is recorded (section
    3.1): with the bare name, a change of the model behind it cannot be seen, and the report says so."""
    system, user = prompts.transport()
    results = []
    for i in (1, 2, 3):
        result = ctx.call('admit:%s:transport:%d' % (engine, i), engine, TRANSPORT_ROLE, system, user)
        results.append(result)
    ok = [r for r in results if r.get('status') == 'ok' and r.get('returned_model')]
    models = sorted({r['returned_model'] for r in ok})
    returned = models[0] if len(models) == 1 else None
    registered = ctx.config['engines'][engine]['model']
    return {'passed': len(ok) >= 2 and len(models) == 1, 'returned_model': returned,
            'dated_name': None if returned is None else returned != registered,
            'statuses': [r.get('status') for r in results], 'limit_headers': (results[0].get('headers') or {}),
            'elapsed_seconds': [r.get('elapsed_seconds') for r in results]}


def _many(ctx, engine, role, prefix, count, system, user, case, width):
    """count items of one control, each read as the main stage reads a reply, sent `width` at a time."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, width)) as pool:
        futures = [pool.submit(collect.ask, ctx, engine, role, '%s:%d' % (prefix, i), system, user, case)
                   for i in range(1, count + 1)]
        return [future.result() for future in futures]


def _summary(records):
    first_unusable = sum(1 for r in records if not r['asks'][0]['valid'])
    reasked = [r for r in records if r['asks_used'] > 1]
    return {'items': len(records), 'usable_after_reask': sum(1 for r in records if r['usable']),
            'first_attempt_unusable': first_unusable,
            'first_attempt_unusable_rate': first_unusable / len(records) if records else None,
            'first_attempt_upper_95': stats.binomial_upper(first_unusable, len(records)),
            're_asked': len(reasked),
            'usable_at_second': sum(1 for r in reasked if r['asks_used'] == 2 and r['usable']),
            'usable_at_third': sum(1 for r in reasked if r['asks_used'] == 3 and r['usable']),
            'truncated_asks': sum(a['truncated'] for r in records for a in r['asks']),
            'refused_asks': sum(a['refused'] for r in records for a in r['asks']),
            'elapsed_seconds': [a.get('elapsed_seconds') for r in records for a in r['asks'] if a.get('elapsed_seconds') is not None]}


def _format_set(ctx, engine, sets, kind, number, width, config):
    """One format set on the three-entry public register: three verdict-format controls (kind 'verdict', the monitor's
    prompt on the three entries) or three entry-format controls (kind 'entry', the corrector's prompt with the first
    two entries kept and the third arriving with its draft); two of the three must be readable."""
    marks = config['admission']
    items = sets['format3']
    arriving = items[-1]
    if kind == 'verdict':
        system, user = prompts.monitor([(item['case'], forms.parse(item['draft'])) for item in items])
        role = 'M'
    else:
        kept = [(item['case'], forms.required(item['case'])) for item in items[:-1]]
        system, user = prompts.corrector(kept, arriving['case'], forms.parse(arriving['draft']))
        role = 'C'
    records = _many(ctx, engine, role, 'admit:%s:format_%s:%d' % (engine, kind, number), marks['format_items'],
                    system, user, arriving['case'], width)
    return {'kind': kind, 'set': number, **_summary(records),
            'passed': sum(r['usable'] for r in records) >= marks['format_pass']}


def _format_controls(ctx, engine, sets, kind, width, config):
    """A format set and, if it fails, the same set sent once more (section 11.3, step 4); the kind passes when either
    set passes."""
    first = _format_set(ctx, engine, sets, kind, 1, width, config)
    if first['passed']:
        return {'passed': True, 'sets': [first]}
    second = _format_set(ctx, engine, sets, kind, 2, width, config)
    return {'passed': second['passed'], 'sets': [first, second]}


def admit_engine(ctx, engine, sets, config):
    """Step 4 for one engine. A failed core set, after its second set where it has one, excludes the engine."""
    marks = config['admission']
    out = {'engine': engine, 'utc': now()}
    transport = _transport(ctx, engine)
    out['transport'] = transport
    if not transport['passed']:
        out.update(admitted=False, generating_admitted=False, failed='transport')
        return out
    headers = transport['limit_headers']
    width = marks['concurrency'] if _limits(headers, ctx.config['engines'][engine]['provider']) else marks['concurrency_without_limits']
    try:
        out['format'] = {kind: _format_controls(ctx, engine, sets, kind, width, config) for kind in ('verdict', 'entry')}
        failed = [kind for kind in ('verdict', 'entry') if not out['format'][kind]['passed']]
        if failed:
            out.update(admitted=False, generating_admitted=False, failed='format_' + failed[0])
            return out
        full = sets['verdict14']
        v_system, v_user = prompts.monitor([(item['case'], forms.parse(item['draft'])) for item in full])
        verdicts = _many(ctx, engine, 'M', 'admit:%s:verdict14' % engine, marks['verdict_items'], v_system, v_user, full[-1]['case'], width)
        out['verdict14'] = _summary(verdicts)
        register = sets['register18']
        kept = [(item['case'], forms.required(item['case'])) for item in register[:-1]]
        arriving = register[-1]
        c_system, c_user = prompts.corrector(kept, arriving['case'], forms.parse(arriving['draft']))
        correctors = _many(ctx, engine, 'C', 'admit:%s:corrector18' % engine, marks['corrector_items'], c_system, c_user, arriving['case'], width)
        out['corrector18'] = _summary(correctors)
        r_system, r_user = prompts.monitor([(item['case'], forms.parse(item['draft'])) for item in register])
        reported = _many(ctx, engine, 'M', 'admit:%s:verdict18' % engine, marks['reported_verdict_items'], r_system, r_user, arriving['case'], width)
        out['verdict18_reported'] = _summary(reported)
        g_system, g_user = prompts.generating(kept, arriving['case'])
        generated = _many(ctx, engine, 'G', 'admit:%s:generating18' % engine, marks['generating_items'], g_system, g_user, arriving['case'], width)
        out['generating18'] = _summary(generated)
    except collect.ModelChanged as exc:
        # A reply whose returned model fails the receipt of section 3.1 ends the engine's use: at admission it is
        # excluded and reported (section 11.1), and the journal's replay of the same reply cannot stop admission again.
        out.update(admitted=False, generating_admitted=False, failed='model_changed', detail=str(exc))
        return out
    passed_verdicts = out['verdict14']['usable_after_reask'] >= marks['verdict_pass']
    passed_correctors = out['corrector18']['usable_after_reask'] >= marks['corrector_pass']
    out['generating_admitted'] = out['generating18']['usable_after_reask'] >= marks['generating_pass']
    out['admitted'] = passed_verdicts and passed_correctors
    if not out['admitted']:
        out['failed'] = 'verdict14' if not passed_verdicts else 'corrector18'
    times = [t for key in ('verdict14', 'corrector18', 'verdict18_reported', 'generating18') for t in out[key]['elapsed_seconds']]
    times += [t for t in transport['elapsed_seconds'] if t is not None]
    out['median_call_seconds'] = statistics.median(times) if times else None
    out['floor_needs'] = config['floor_unusable_limit']
    return out


def _limits(headers, provider):
    """The requests and token limits a minute that the first admission reply's headers state, or None. Anthropic states
    an output-token limit; OpenAI states one limit on input and output tokens together, which the rule for pairs in
    flight reads against the planned input and output tokens together (BUILD-DECISIONS.md, 18 October 2026)."""
    if provider == providers.ANTHROPIC:
        names, counted = ('anthropic-ratelimit-requests-limit', 'anthropic-ratelimit-output-tokens-limit'), 'output'
    else:
        names, counted = ('x-ratelimit-limit-requests', 'x-ratelimit-limit-tokens'), 'input_and_output'
    try:
        values = [float(headers[name]) for name in names]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(value > 0 for value in values):
        return None
    return {'requests_per_minute': values[0], 'tokens_per_minute': values[1], 'tokens_counted': counted}


def run(root, config, ctx, sets):
    """Step 4 for each engine; writes the admission receipt. With fewer than two engines admitted the lane does not
    collect (section 11.1)."""
    engines = seal.engines_of(config)
    receipt = {'utc': now(), 'engines': {}}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as pool:
        futures = {engine: pool.submit(admit_engine, ctx, engine, sets, config) for engine in engines}
        for engine in engines:
            receipt['engines'][engine] = futures[engine].result()
    admitted = [e for e in engines if receipt['engines'][e]['admitted']]
    receipt['admitted'] = admitted
    receipt['generating_admitted'] = {e: receipt['engines'][e]['generating_admitted'] for e in admitted}
    receipt['expected_models'] = {e: receipt['engines'][e]['transport']['returned_model'] for e in admitted}
    receipt['dated_names'] = {e: receipt['engines'][e]['transport']['dated_name'] for e in admitted}
    receipt['enough_engines'] = len(admitted) >= config['engines_needed']
    # Every exclusion goes to the deviation record (section 11.4), and so does a generating arm dropped on an engine.
    for e in engines:
        if not receipt['engines'][e]['admitted']:
            ctx.deviation({'kind': 'exclusion', 'engine': e, 'failed': receipt['engines'][e].get('failed')})
        elif not receipt['engines'][e]['generating_admitted']:
            ctx.deviation({'kind': 'generating_arm_dropped', 'engine': e})
    save(_receipt_path(root, 'admission_receipt.json'), receipt, exclusive=True)
    return receipt


def calibrate(root, config, ctx, sets, admitted):
    """Step 5: 240 single-entry monitor calls and 120 single-entry corrector calls on each tier's public calibration
    set, for each admitted engine. The tier is the highest at which, on every admitted engine, the catch rate is at
    least 0.95, the false-alarm rate at most 0.05 and net repair at least 0.85; if no tier qualifies, tier 1. Answers are
    scored by the main stage's own pure function against the public sets' required entries."""
    rule = config['calibration']
    width = config['admission']['concurrency']
    receipt = {'utc': now(), 'engines': {}, 'rule': rule}
    qualifies = {}
    for engine in admitted:
        receipt['engines'][engine] = {}
        for tier in generate.TIERS:
            items = sets['calibration'][str(tier)]
            caught = alarms = faulty = sound = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, width)) as pool:
                monitor_jobs = []
                for i, item in enumerate(items['monitor'], 1):
                    system, user = prompts.monitor([(item['case'], forms.parse(item['draft']))])
                    monitor_jobs.append((item, pool.submit(collect.ask, ctx, engine, 'M', 'calibrate:%s:tier%d:monitor:%d' % (engine, tier, i), system, user, item['case'])))
                for item, future in monitor_jobs:
                    record = future.result()
                    rejected = record['usable'] and record.get('verdict') == 'REJECT'
                    if item['faulty']:
                        faulty += 1; caught += int(rejected)
                    else:
                        sound += 1; alarms += int(rejected)
            repaired = damaged = c_faulty = c_sound = 0
            jobs = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, width)) as pool:
                for i, item in enumerate(items['corrector'], 1):
                    system, user = prompts.corrector([], item['case'], forms.parse(item['draft']))
                    jobs.append((item, pool.submit(collect.ask, ctx, engine, 'C', 'calibrate:%s:tier%d:corrector:%d' % (engine, tier, i), system, user, item['case'])))
                for item, future in jobs:
                    record = future.result()
                    kept = record['entry_line'] if record['usable'] else item['draft']
                    sound_kept, _ = score.compare(kept, forms.line(forms.required(item['case'])))
                    if item['faulty']:
                        c_faulty += 1; repaired += int(sound_kept)
                    else:
                        c_sound += 1; damaged += int(not sound_kept)
            row = {'catch': caught / faulty, 'false_alarm': alarms / sound, 'net_repair': repaired / c_faulty - damaged / c_sound,
                   'monitor_items': faulty + sound, 'corrector_items': c_faulty + c_sound}
            row['qualifies'] = (row['catch'] >= rule['catch'] and row['false_alarm'] <= rule['false_alarm']
                                and row['net_repair'] >= rule['net_repair'])
            receipt['engines'][engine][str(tier)] = row
            qualifies.setdefault(tier, []).append(row['qualifies'])
    chosen = 1
    for tier in sorted(generate.TIERS, reverse=True):
        if qualifies.get(tier) and all(qualifies[tier]):
            chosen = tier
            break
    receipt['tier'] = chosen
    save(_receipt_path(root, 'calibration_receipt.json'), receipt, exclusive=True)
    return receipt


def in_flight(root, config, receipt):
    """The registered rule for pairs in flight (section 11.3), read from step 4 alone: for each engine, the smallest
    number from 4 to 12 at which the projected wall time of both stages, from the median call time, is within five days,
    and never more than keeps the planned requests and output tokens a minute under half the limits the first admission
    reply's headers state; where the headers state no limit, four. Where no number meets five days, the largest number
    the limits allow."""
    rule = config['pairs_in_flight']
    out = {'utc': now(), 'engines': {}}
    pairs = config['looks']['sizes'][-1]
    for engine in receipt['admitted']:
        row = receipt['engines'][engine]
        provider = config['engines'][engine]['provider']
        limits = _limits(row['transport']['limit_headers'], provider)
        median = row['median_call_seconds'] or rule['assumed_call_seconds']
        if limits is None:
            out['engines'][engine] = {'pairs_in_flight': rule['least'], 'reason': 'the headers state no limit',
                                      'median_call_seconds': median}
            continue
        per_call = rule['planned_output_tokens_per_call'] + (
            rule['planned_input_tokens_per_call'] if limits['tokens_counted'] == 'input_and_output' else 0)
        allowed = []
        for n in range(rule['least'], rule['most'] + 1):
            requests = n * rule['calls_per_pair_round'] * 60.0 / median
            tokens = requests * per_call
            if requests < limits['requests_per_minute'] / 2.0 and tokens < limits['tokens_per_minute'] / 2.0:
                allowed.append(n)
        if not allowed:
            choice, reason = rule['least'], 'no number keeps the planned load under half the limits; the least'
        else:
            meeting = [n for n in allowed if math.ceil(pairs / n) * rule['rounds'] * median <= rule['days'] * 86400]
            choice = meeting[0] if meeting else allowed[-1]
            reason = 'within %d days' % rule['days'] if meeting else 'no number meets the days; the largest the limits allow'
        out['engines'][engine] = {'pairs_in_flight': choice, 'reason': reason, 'median_call_seconds': median,
                                  'limits': limits,
                                  'projected_days': math.ceil(pairs / choice) * rule['rounds'] * median / 86400.0}
    save(_receipt_path(root, 'in_flight.json'), out, exclusive=True)
    return out
