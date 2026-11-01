"""The collector (41-D-COUNTED, sections 3.4, 5, 11.1 and 11.2).

Package F of the build list (DESIGN-FREEZE.md, section 12). For each engine, pairs run in index order with the number
in flight that the runtime freeze fixed (section 11.3); both blocks of a pair together; a round's calls sent together.
Nothing here is executed before both registrations read REGISTERED (SD-001, SD-012); before then the only mechanical
check is a parse of this file's text.

The return record. Every provider result is judged before any text is read (`judge`): a reply whose returned model
differs from the admitted one ends that engine's collection; a truncated reply and a refusal are recorded as such, with
reason `truncated` or `refused`, and no text of theirs is passed to the reader; only a result whose status is `ok` (or
an empty reply that carried output tokens) is read by the rules of section 6.1. An unusable reply is asked again, the
same request to the same model, up to twice more (section 11.2, item 8); each ask is its own journalled request with
its own numbered transport attempts; the first usable reply is used; only a call whose three asks were all unusable
falls back to section 5's rule for its role.

Money (section 11.1). A pair starts only when this lane's journal can reserve its worst case inside the lane's cap for
that provider, counting settled charges, the reservations still open and, for every pair of the engine still in flight,
its whole worst case (`can_start`), so that no pair is left half done by a cap (section 3.3, item 2); if it cannot, the
collector waits for the pairs in flight to settle and asks again, and the provider starts no further pair only when
none of its pairs is in flight and the answer is still no; the other provider continues. Counting a pair in flight at
its whole worst case on top of what its calls have already settled only delays a start; with none in flight the rule is
exactly the lenient reading of revision 4.2 (BUILD-DECISIONS.md, 18 October 2026). Pairs still run in index order, so
the completed pairs are a prefix of the seeded order whatever they cost.

Stops and holds (sections 3.1, 3.4 and 11.2). A changed model identifier, a posted price change (`stop_at`) and the cap
end an engine's collection at its last completed pair, and are recorded in engine_stops.jsonl; no pair after a model
change is used. A transport-class hold holds its provider; the collector releases it itself after one hour with a
fixed recovery entry, at most six times in a UTC day for each provider, and a seventh holds the provider until the
author's `recover`. Every other hold stops new calls on that provider until `recover`, and leaves the other running.

Resume. Restartable at any point: a completed pair is read from its log, and a pair in progress is replayed from its
first round, every call answered from the journal and the saved replies (the client returns a settled attempt's saved
result and issues nothing), so that the registers are rebuilt exactly as they were run.
"""
import concurrent.futures
import datetime
import json
import threading
import time
import traceback
from pathlib import Path

from p21base import providers
from p21base.common import GateError, digest, load, now, save
from p21base.ledger import RECOVERY_LOG

from . import forms, loop, reading, schedule, seal

STOP_CAP = 'cap'
STOP_MODEL = 'model_changed'
STOP_PRICE = 'price_change'
STOP_HELD = 'held'
STOP_ERROR = 'error'
# The study abandoned by the author's recover (section 11.2, item 6): a stop before a look, like the three above
# (section 7.4). The lane reads it from the recovery log before any client is built (lane._stage).
STOP_ABANDONED = 'abandoned'
PERMANENT_STOPS = (STOP_CAP, STOP_MODEL, STOP_PRICE, STOP_ABANDONED)
ENGINE_STOPS = 'engine_stops.jsonl'
DEVIATIONS = 'deviation_log.jsonl'
HOLD_RELEASE_SECONDS = 3600
RELEASES_A_DAY = 6
AUTO_RELEASE_CAUSE = 'transport_hold_auto_release'
CALLS_PER_PAIR_ROUND = 2 * len(loop.ROLES)


_DEVIATION_LOCK = threading.Lock()


def append_deviation(root, record):
    """Append one entry to the deviation record of section 11.4 as it happens: every retry, hold, release, exclusion,
    repair and stop, one JSON object a line. The provider client's events (retries and holds) come here too."""
    line = json.dumps({'utc': now(), **record}, ensure_ascii=False, sort_keys=True)
    with _DEVIATION_LOCK:
        with (Path(root) / DEVIATIONS).open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')


def read_deviations(root):
    """Every entry of the deviation record that reads as a JSON object, in the order written."""
    path = Path(root) / DEVIATIONS
    if not path.exists():
        return []
    out = []
    for line in path.read_text('utf-8').splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


class ModelChanged(GateError):
    """A reply's returned model differs from the admitted value: that engine's collection ends (section 3.1)."""


class Abandoned(GateError):
    """The engine stopped while this pair was in flight; the pair is not used."""


def judge(result, engine, role, case):
    """The return record of one ask (section 12): the provider's result judged before any text is read."""
    status = result.get('status')
    record = {'slot': result.get('base_slot'), 'engine': engine, 'role': role, 'status': status, 'valid': False,
              'reason': None, 'input_tokens': result.get('input_tokens'), 'output_tokens': result.get('output_tokens'),
              'reasoning_tokens': result.get('reasoning_tokens'), 'returned_model': result.get('returned_model'),
              'response_id': result.get('response_id'), 'elapsed_seconds': result.get('elapsed_seconds'),
              'transport_attempts': result.get('attempt'), 'truncated': status == providers.TRUNCATED,
              'refused': status == providers.REFUSED, 'refusal_category': result.get('refusal_category'),
              'request_id': result.get('request_id')}
    if status == providers.MODEL_CHANGED:
        raise ModelChanged('the returned model of %s was %r' % (engine, result.get('returned_model')))
    if status in ('ok', 'empty_response'):
        text = result.get('text') if status == 'ok' else ''
        if role in loop.ENTRY_ROLES:
            entry, reason = reading.read_entry(text, case['code'], case['form'])
            record.update(valid=entry is not None, reason=reason, entry_line=forms.line(entry) if entry is not None else None)
        else:
            verdict, reason = reading.verdict_and_reason(text)
            record.update(valid=verdict is not None, reason=reason, verdict=verdict)
    elif status == providers.TRUNCATED:
        record['reason'] = reading.TRUNCATED
    elif status == providers.REFUSED:
        record['reason'] = reading.REFUSED
    else:
        # A finish no prompt can produce (a stop sequence, a tool call): recorded with its word, and no text read.
        record['reason'] = 'abnormal_finish'
    if role in loop.ENTRY_ROLES and 'entry_line' not in record:
        record['entry_line'] = None
    if role in loop.VERDICT_ROLES and 'verdict' not in record:
        record['verdict'] = None
    return record


def ask(ctx, engine, role, slot, system, user, case):
    """One call under the re-ask of section 11.2, item 8: up to three asks of the same request; the first usable reply
    is used. The final record carries every ask."""
    asks = []
    for number in (1, 2, 3):
        name = slot if number == 1 else slot + '/r%d' % number
        result = ctx.call(name, engine, role, system, user)
        record = judge(result, engine, role, case)
        asks.append(record)
        if record['valid']:
            break
        if number < 3:
            ctx.deviation({'kind': 're_ask', 'engine': engine, 'slot': name, 'reason': record['reason']})
    final = dict(asks[-1])
    final.update(slot=slot, asks=asks, asks_used=len(asks), usable=asks[-1]['valid'],
                 unusable_after_reask=not asks[-1]['valid'])
    return final


class Context:
    """What the collector's threads share for one run of a stage: the client, the stop state of each engine, the
    transport-hold manager and the append-only records."""

    def __init__(self, root, config, client, tier, in_flight, generating, clock=time.time):
        self.root, self.config, self.client, self.tier = Path(root), config, client, tier
        self.in_flight, self.generating, self.clock = dict(in_flight), dict(generating), clock
        self.lock = threading.RLock()
        self.cond = threading.Condition(self.lock)
        self.stops = {}
        self.capped = set()
        self.holds = {}
        for record in read_engine_stops(self.root):
            if record['reason'] in PERMANENT_STOPS:
                self.stops[record['engine']] = record['reason']
        # The re-asks already in the deviation record: a pair resumed from the journal replays its asks, and each
        # re-ask is recorded once (section 11.4).
        self.recorded_re_asks = {record.get('slot') for record in read_deviations(self.root) if record.get('kind') == 're_ask'}

    # Records
    def deviation(self, record):
        if record.get('kind') == 're_ask':
            with self.lock:
                if record.get('slot') in self.recorded_re_asks:
                    return
                self.recorded_re_asks.add(record.get('slot'))
        append_deviation(self.root, record)

    def stop(self, engine, reason, detail=''):
        with self.lock:
            if engine in self.stops and self.stops[engine] in PERMANENT_STOPS:
                return
            self.stops[engine] = reason
            completed = completed_prefix(self.root, engine)
            record = {'utc': now(), 'engine': engine, 'reason': reason, 'detail': detail, 'completed_prefix': completed}
            with (self.root / ENGINE_STOPS).open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + '\n')
            self.deviation({'kind': 'stop', **record})
            self.cond.notify_all()

    def stopped(self, engine):
        with self.lock:
            return self.stops.get(engine)

    # Calls, with the transport-hold release of section 11.2, item 6
    def call(self, name, engine, role, system, user):
        provider = self.config['engines'][engine]['provider']
        while True:
            self.wait_while_held(provider, engine)
            reason = self.stopped(engine)
            if reason in (STOP_MODEL, STOP_PRICE, STOP_HELD, STOP_ERROR):
                raise Abandoned('%s stopped (%s)' % (engine, reason))
            try:
                return self.client.call(name, engine, role, system, user)
            except providers.TransportHold as hold:
                self.release_transport_hold(engine, hold)

    def wait_while_held(self, provider, engine):
        """A transport hold stops new calls on its provider and leaves the other running (sections 11.1 and 11.2,
        items 1 and 6; package F): a call made while the provider is held waits for the harness's release, and where
        the hold stays until the author's recover (a seventh in a UTC day) it is not made. A wait ends at once when the
        engine stops (stop notifies), so that its call is abandoned without waiting out the hour."""
        with self.cond:
            while True:
                state = self.holds.get(provider)
                if state is None or (state['since'] is None and not state['refused']):
                    return
                if self.stops.get(engine) in (STOP_MODEL, STOP_PRICE, STOP_HELD, STOP_ERROR):
                    return
                if state['refused']:
                    raise providers.ProviderHeld('the transport hold on %s stays until the author\'s recover' % provider)
                self.cond.wait(timeout=60.0)

    def _releases_today(self, provider):
        today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
        return sum(1 for record in read_deviations(self.root)
                   if record.get('kind') == 'auto_release' and record.get('provider') == provider
                   and str(record.get('utc', '')).startswith(today))

    def release_transport_hold(self, engine, hold):
        """Hold the provider; after one hour one thread writes a fixed recovery entry for every attempt held, and every
        waiting call is issued again. A seventh release in a UTC day is not made: the provider stays held."""
        provider = self.config['engines'][engine]['provider']
        with self.cond:
            state = self.holds.setdefault(provider, {'since': None, 'rows': set(), 'generation': 0, 'refused': False})
            if state['refused']:
                raise providers.ProviderHeld('the transport hold on %s was released six times today; the author\'s recover releases it' % provider)
            if state['since'] is None:
                state['since'] = self.clock()
                self.deviation({'kind': 'provider_held', 'provider': provider, 'engine': engine, 'request_id': hold.row_id})
            state['rows'].add(hold.row_id)
            generation = state['generation']
            while state['generation'] == generation and not state['refused']:
                remaining = state['since'] + HOLD_RELEASE_SECONDS - self.clock()
                if remaining <= 0:
                    break
                self.cond.wait(timeout=min(remaining, 60.0))
            if state['refused']:
                raise providers.ProviderHeld('the transport hold on %s stays until the author\'s recover' % provider)
            if state['generation'] != generation:
                return
            if self._releases_today(provider) >= RELEASES_A_DAY:
                state['refused'] = True
                self.stop(engine, STOP_HELD, 'six automatic releases today')
                self.cond.notify_all()
                raise providers.ProviderHeld('the transport hold on %s was released six times today' % provider)
            for row_id in sorted(state['rows']):
                row = self.client.ledger.get(row_id)
                entry = {'utc': now(), 'request_id': row_id, 'cause': AUTO_RELEASE_CAUSE, 'action': 'reissue',
                         'reconciled_charge_usd': row['reservation'] if row else 0.0,
                         'written_without_reading_any_score': True}
                with (self.root / 'private' / RECOVERY_LOG).open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(entry, sort_keys=True) + '\n')
            self.deviation({'kind': 'auto_release', 'provider': provider, 'rows': sorted(state['rows'])})
            state.update(since=None, rows=set(), generation=generation + 1)
            self.cond.notify_all()


def read_engine_stops(root):
    path = Path(root) / ENGINE_STOPS
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text('utf-8').splitlines() if line.strip()]


def pair_path(root, engine, pair):
    return Path(root) / 'pairs' / engine / ('%04d.json' % pair)


def completed_prefix(root, engine):
    """The length of the run of completed pairs from pair 1 in index order."""
    n = 0
    while pair_path(root, engine, n + 1).exists():
        n += 1
    return n


def worst_case(config, engine, pair):
    """The pair's worst case (section 3.4): every call at the 16,000-token cap, the input at twice plan, the adapter's
    margin and three attempts reserved for every call, by the arms the pair has."""
    table = config['engines'][engine]['pair_worst_case_usd']
    if pair <= config['secondary_pairs']:
        return table['with_secondaries']
    if pair <= config['positive_control_pairs']:
        return table['with_window']
    return table['core']


def can_start(ctx, engine, pair, in_flight=()):
    """Whether the journal can reserve the pair's worst case inside this lane's cap for the provider (section 11.1):
    the cap less the settled charges and open reservations the journal holds (its headroom), less the whole worst case
    of every pair of this engine still in flight, must cover the new pair's worst case. With no pair in flight it is the
    journal's headroom against the pair's worst case alone."""
    held = sum(worst_case(ctx.config, engine, other) for other in in_flight)
    return ctx.client.headroom(engine) - held >= worst_case(ctx.config, engine, pair) - 1e-9


def run_pair(ctx, engine, pair, bank_pair, pair_schedule, arms, pool):
    """Both blocks of one pair, round by round, each round's calls sent together; returns the pair's log."""
    started = now()
    blocks = [loop.Block(engine, pair, which, schedule.block_drafts(pair_schedule, bank_pair['cases'], which), arms)
              for which in (1, 2)]
    for _ in range(loop.ROUNDS):
        jobs = []
        for block in blocks:
            case = block.arriving()[0]
            for call in loop.calls(block):
                future = pool.submit(ask, ctx, engine, call['role'], call['slot'], call['system'], call['user'], case)
                jobs.append((block, call['role'], future))
        concurrent.futures.wait([future for _, _, future in jobs])
        for _, _, future in jobs:
            error = future.exception()
            if error is not None:
                raise error
        for block in blocks:
            loop.absorb(block, {role: future.result() for b, role, future in jobs if b is block})
        if ctx.stopped(engine) == STOP_MODEL:
            raise Abandoned('the model changed while pair %d was in flight' % pair)
    return {'study_id': ctx.config['study_id'], 'engine': engine, 'pair': pair, 'tier': ctx.tier, 'arms': list(arms),
            'started_utc': started, 'completed_utc': now(),
            'blocks': [{'which': block.which, 'rounds': block.log} for block in blocks]}


def _complete(ctx, engine, pair, log):
    with ctx.lock:
        if ctx.stopped(engine) == STOP_MODEL:
            return False
        path = pair_path(ctx.root, engine, pair)
        if not path.exists():
            save(path, log, exclusive=True)
        return True


def engine_worker(ctx, engine, first, last):
    """One engine's stage: pairs first to last in index order, up to the number in flight; the per-provider stop."""
    bank = seal.load_bank(ctx.root, ctx.tier, engine)
    plan = seal.load_schedule(ctx.root, ctx.tier, engine)
    width = int(ctx.in_flight[engine])
    pending = [p for p in range(first, last + 1) if not pair_path(ctx.root, engine, p).exists()]
    index = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=width) as pairs, \
            concurrent.futures.ThreadPoolExecutor(max_workers=width * CALLS_PER_PAIR_ROUND) as calls:
        running = {}
        while True:
            while (ctx.stopped(engine) is None and engine not in ctx.capped and index < len(pending)
                   and len(running) < width):
                pair = pending[index]
                if not can_start(ctx, engine, pair, list(running.values())):
                    if running:
                        break
                    ctx.stop(engine, STOP_CAP, 'the worst case of pair %d cannot be reserved with no pair in flight' % pair)
                    break
                arms = loop.arms_for(pair, ctx.generating.get(engine, True), ctx.config['positive_control_pairs'],
                                     ctx.config['secondary_pairs'])
                future = pairs.submit(run_pair, ctx, engine, pair, bank['pairs'][pair - 1], plan['pairs'][pair - 1], arms, calls)
                running[future] = pair
                index += 1
            if not running:
                if engine in ctx.capped and ctx.stopped(engine) is None:
                    ctx.stop(engine, STOP_CAP, 'a reservation was refused at the cap')
                break
            done, _ = concurrent.futures.wait(list(running), return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                pair = running.pop(future)
                try:
                    _complete(ctx, engine, pair, future.result())
                except providers.CapReached:
                    ctx.capped.add(engine)
                    ctx.deviation({'kind': 'cap_reservation_refused', 'engine': engine, 'pair': pair})
                except ModelChanged as exc:
                    ctx.stop(engine, STOP_MODEL, str(exc))
                except providers.EngineStopped as exc:
                    ctx.stop(engine, STOP_PRICE, str(exc))
                except providers.ProviderHeld as exc:
                    ctx.stop(engine, STOP_HELD, str(exc))
                except Abandoned:
                    pass
                except Exception as exc:
                    ctx.stop(engine, STOP_ERROR, type(exc).__name__ + ': ' + str(exc) + '\n' + traceback.format_exc(limit=8))
    return {'engine': engine, 'completed_prefix': completed_prefix(ctx.root, engine), 'stopped': ctx.stopped(engine),
            'target': last}


def stage(ctx, first, last):
    """A collection stage (section 11.3, steps 7 and 9): every admitted engine at once, each in its own thread."""
    engines = list(ctx.in_flight)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines)) as pool:
        futures = {pool.submit(engine_worker, ctx, engine, first, last): engine for engine in engines}
        for future in concurrent.futures.as_completed(futures):
            engine = futures[future]
            try:
                results[engine] = future.result()
            except Exception as exc:
                ctx.stop(engine, STOP_ERROR, type(exc).__name__ + ': ' + str(exc))
                results[engine] = {'engine': engine, 'completed_prefix': completed_prefix(ctx.root, engine),
                                   'stopped': STOP_ERROR, 'target': last}
    status = {'utc': now(), 'first': first, 'last': last, 'engines': results}
    save(ctx.root / 'collect' / ('stage_%d_%d.json' % (first, last)), status)
    return status


def write_stop_at(root, date_text):
    """`stop_at` (section 3.4): each engine's collection ends at its last completed pair before the date a posted
    price change takes effect. The client refuses every call from that date."""
    try:
        when = datetime.date.fromisoformat(date_text)
    except (TypeError, ValueError):
        raise GateError('stop_at needs a date written YYYY-MM-DD')
    path = Path(root) / 'stop_at.json'
    if path.exists():
        earlier = load(path).get('date')
        if earlier and earlier <= when.isoformat():
            return load(path)
    record = {'date': when.isoformat(), 'written_utc': now(), 'cause': 'posted price change'}
    save(path, record)
    return record


def pair_logs(root, engine, n_pairs):
    """The logs of pairs 1 to n_pairs of an engine, in index order; raises if one is missing."""
    out = []
    for pair in range(1, n_pairs + 1):
        path = pair_path(root, engine, pair)
        if not path.exists():
            raise GateError('pair %d of %s is not complete' % (pair, engine))
        out.append(load(path))
    return out
