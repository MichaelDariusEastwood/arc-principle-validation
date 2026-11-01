"""Authored cases for the collector (41-D-COUNTED, sections 3.4, 5, 11.1 and 11.2): the return record judged before any
text is read, the re-ask, the start rule that leaves no pair half done, and a stand-in run of the provider client at
the run's concurrency with injected 429s, time-outs, empty replies and an interruption. Not run before both
registrations read REGISTERED."""
import concurrent.futures
import io
import json
import os
import socket
import threading
import urllib.error

from p21base import providers
from p21base.common import canonical, digest
from p21lane import collect, reading

from support import BY_CODE, load_config, temp_root

CASE = BY_CODE['ABCD']
GOOD = 'ABCD|PAY|2027-01-04|-1250|DONE|RP'


def result(status='ok', text=GOOD, **extra):
    return {'status': status, 'text': text, 'input_tokens': 100, 'output_tokens': 10, 'reasoning_tokens': 0,
            'returned_model': 'claude-haiku-5-5', 'response_id': 'r', 'elapsed_seconds': 0.1, 'base_slot': 's',
            'attempt': 1, 'request_id': 'q', **extra}


def test_the_return_record_is_judged_before_any_text_is_read():
    ok = collect.judge(result(), 'claude-haiku-5-5', 'C', CASE)
    assert ok['valid'] and ok['entry_line'] == GOOD and ok['reason'] is None
    truncated = collect.judge(result(status=providers.TRUNCATED), 'claude-haiku-5-5', 'C', CASE)
    assert not truncated['valid'] and truncated['reason'] == reading.TRUNCATED and truncated['entry_line'] is None
    assert truncated['truncated'] is True
    refused = collect.judge(result(status=providers.REFUSED, text='REJECT', refusal_category='cyber'), 'claude-haiku-5-5', 'M', CASE)
    assert not refused['valid'] and refused['reason'] == reading.REFUSED and refused['verdict'] is None
    assert refused['refusal_category'] == 'cyber'
    empty = collect.judge(result(status='empty_response', text=GOOD), 'claude-haiku-5-5', 'C', CASE)
    assert not empty['valid'] and empty['reason'] == reading.NO_ENTRY
    odd = collect.judge(result(status='pause_turn'), 'claude-haiku-5-5', 'C', CASE)
    assert not odd['valid'] and odd['reason'] == 'abnormal_finish'
    assert 'text' not in ok and 'text' not in truncated
    try:
        collect.judge(result(status=providers.MODEL_CHANGED), 'claude-haiku-5-5', 'C', CASE)
    except collect.ModelChanged:
        pass
    else:
        raise AssertionError('a changed model did not end the engine\'s collection')


class ScriptedContext:
    def __init__(self, results):
        self.results = list(results)
        self.names = []
        self.deviations = []

    def call(self, name, engine, role, system, user):
        self.names.append(name)
        return self.results.pop(0)

    def deviation(self, record):
        self.deviations.append(record)


def test_the_re_ask_up_to_twice():
    ctx = ScriptedContext([result(status=providers.TRUNCATED), result(text='No entry here.'), result()])
    record = collect.ask(ctx, 'claude-haiku-5-5', 'C', 'collect:e:1:1:3:C', 'sys', 'user', CASE)
    assert ctx.names == ['collect:e:1:1:3:C', 'collect:e:1:1:3:C/r2', 'collect:e:1:1:3:C/r3']
    assert record['usable'] and record['asks_used'] == 3 and record['slot'] == 'collect:e:1:1:3:C'
    assert [d['reason'] for d in ctx.deviations] == [reading.TRUNCATED, reading.NO_ENTRY]
    ctx = ScriptedContext([result(text='VERDICT: REJECT')])
    record = collect.ask(ctx, 'claude-haiku-5-5', 'M', 'collect:e:1:1:3:M', 'sys', 'user', CASE)
    assert record['usable'] and record['verdict'] == 'REJECT' and record['asks_used'] == 1
    ctx = ScriptedContext([result(text='maybe')] * 3)
    record = collect.ask(ctx, 'claude-haiku-5-5', 'M', 'collect:e:1:1:3:M', 'sys', 'user', CASE)
    assert record['unusable_after_reask'] and not record['usable'] and record['asks_used'] == 3


class HeadroomContext:
    def __init__(self, headroom):
        self.config = load_config()
        self.client = type('C', (), {'headroom': lambda self_, engine: headroom})()


def test_the_start_rule_counts_every_pair_in_flight():
    ctx = HeadroomContext(10.0)
    engine = 'claude-haiku-5-5'
    assert collect.worst_case(ctx.config, engine, 1) == 5.72
    assert collect.worst_case(ctx.config, engine, 121) == 3.81
    assert collect.worst_case(ctx.config, engine, 221) == 2.87
    assert collect.can_start(ctx, engine, 1)
    assert not collect.can_start(ctx, engine, 2, [1])                 # 10.00 - 5.72 < 5.72
    assert collect.can_start(ctx, engine, 300, [250, 251])            # 10.00 - 2 x 2.87 >= 2.87
    assert not collect.can_start(ctx, engine, 300, [250, 251, 252])
    assert not collect.can_start(HeadroomContext(2.0), engine, 300)  # with none in flight the provider stops for good


def test_the_completed_prefix_and_stop_at():
    root = temp_root()
    for pair in (1, 2, 4):
        path = collect.pair_path(root, 'gpt-6-luna', pair)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    assert collect.completed_prefix(root, 'gpt-6-luna') == 2
    assert collect.write_stop_at(root, '2026-12-10')['date'] == '2026-12-10'
    assert collect.write_stop_at(root, '2026-12-20')['date'] == '2026-12-10'     # the earlier date stands
    assert collect.write_stop_at(root, '2026-12-05')['date'] == '2026-12-05'


# ---- A stand-in run of the provider client: the real client and journal, a stand-in network.

class Response:
    def __init__(self, body):
        self.raw = json.dumps(body).encode('utf-8')
        self.headers = {'request-id': 'req-1', 'anthropic-ratelimit-requests-limit': '1000'}

    def read(self, n=-1):
        return self.raw

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def anthropic_body(text='READY', output_tokens=10):
    return {'id': 'msg_1', 'model': 'claude-haiku-5-5', 'stop_reason': 'end_turn',
            'content': [{'type': 'thinking', 'thinking': 'hidden'}, {'type': 'text', 'text': text}] if text else [],
            'usage': {'input_tokens': 120, 'output_tokens': output_tokens, 'service_tier': 'standard',
                      'output_tokens_details': {'thinking_tokens': 5}}}


class StandInNetwork:
    """Answers each request by a script: a list of outcomes for the n-th time a given body is seen."""
    def __init__(self, script):
        self.script = script
        self.seen = {}
        self.lock = threading.Lock()

    def open(self, request, timeout=None):
        body = request.data
        with self.lock:
            n = self.seen.get(body, 0)
            self.seen[body] = n + 1
        outcome = self.script(body, n)
        if outcome == '429':
            raise urllib.error.HTTPError(request.full_url, 429, 'Too Many Requests', {'retry-after': '1'},
                                         io.BytesIO(b'{"error": {"type": "rate_limit_error"}}'))
        if outcome == 'timeout':
            raise socket.timeout('timed out')
        if outcome == 'empty':
            return Response(anthropic_body(text='', output_tokens=0))
        return Response(anthropic_body())


def stand_in_client(root, script):
    config = load_config(pricing_valid_through='2999-12-31')
    os.environ['ANTHROPIC_API_KEY'] = 'stand-in-key-not-a-real-one'
    client = providers.Client(root, config, live=True, sleep=lambda seconds: None)
    client.opener = StandInNetwork(script)
    return client, config


def rows(client):
    with client.ledger.lock:
        return client.ledger.db.execute('SELECT slot, charge, reservation, state FROM calls ORDER BY slot').fetchall()


def test_a_request_through_429_a_time_out_and_an_empty_reply():
    root = temp_root()
    outcomes = ['429', 'timeout', 'empty', 'ok']
    client, _ = stand_in_client(root, lambda body, n: outcomes[n])
    try:
        record = client.call('collect:claude-haiku-5-5:1:1:1:C', 'claude-haiku-5-5', 'C', 'system', 'user')
        assert record['status'] == 'ok' and record['attempt'] == 4 and record['text'] == 'READY'
        journal = {slot: (charge, reservation, state) for slot, charge, reservation, state in rows(client)}
        first = journal['collect:claude-haiku-5-5:1:1:1:C']
        second = journal['collect:claude-haiku-5-5:1:1:1:C#attempt2']
        third = journal['collect:claude-haiku-5-5:1:1:1:C#attempt3']
        fourth = journal['collect:claude-haiku-5-5:1:1:1:C#attempt4']
        assert first[0] == 0.0                                            # a 429 is uncharged
        assert abs(second[0] - second[1]) < 1e-12                         # a time-out is charged at its reservation
        assert abs(third[0] - 120 * 0.10 / 1e6) < 1e-12                   # an empty reply: from its usage
        assert abs(fourth[0] - (120 * 0.10 + 10 * 0.50) / 1e6) < 1e-12
        # A replay answers from the journal and issues nothing.
        seen = dict(client.opener.seen)
        again = client.call('collect:claude-haiku-5-5:1:1:1:C', 'claude-haiku-5-5', 'C', 'system', 'user')
        assert again['response_id'] == record['response_id'] and client.opener.seen == seen
    finally:
        client.close()


def test_an_interrupted_attempt_is_charged_in_full_and_issued_again():
    root = temp_root()
    client, config = stand_in_client(root, lambda body, n: 'ok')
    try:
        spec = config['engines']['claude-haiku-5-5']
        pcfg = config['providers']['anthropic']
        body = providers.payload('anthropic', 'system', 'user', spec['request_fields'])
        ident = digest(canonical({'study': config['study_id'], 'engine': 'claude-haiku-5-5', 'role': 'C', 'slot': 'x2', 'body': body}))
        _, reservation = providers.reserve_amount(pcfg, 'anthropic', body)
        client.ledger.reserve(ident, spec['budget_key'], reservation, {'slot': 'x2', 'role': 'C'})
        record = client.call('x2', 'claude-haiku-5-5', 'C', 'system', 'user')
        assert record['status'] == 'ok' and record['attempt'] == 2
        journal = {slot: (charge, res) for slot, charge, res, _ in rows(client)}
        assert abs(journal['x2'][0] - journal['x2'][1]) < 1e-12
    finally:
        client.close()


def test_one_study_lock_for_the_journal():
    root = temp_root()
    client, _ = stand_in_client(root, lambda body, n: 'ok')
    try:
        try:
            stand_in_client(root, lambda body, n: 'ok')
        except Exception as exc:
            assert 'lock' in str(exc)
        else:
            raise AssertionError('a second client took the journal')
    finally:
        client.close()


def test_the_run_concurrency_with_injected_failures():
    root = temp_root()

    def script(body, n):
        mark = digest(body)[0]
        if n == 0 and mark in '0123':
            return '429'
        if n == 0 and mark in '4567':
            return 'timeout'
        if n == 0 and mark in '89':
            return 'empty'
        return 'ok'

    client, _ = stand_in_client(root, script)
    try:
        slots = ['collect:claude-haiku-5-5:%d:1:1:C' % i for i in range(1, 49)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            results = list(pool.map(lambda s: client.call(s, 'claude-haiku-5-5', 'C', 'system', s), slots))
        assert all(r['status'] == 'ok' for r in results)
        journal = rows(client)
        assert len({row[0] for row in journal}) == len(journal)
        assert all(row[3] != 'reserved' for row in journal)
        assert client.headroom('claude-haiku-5-5') < 108.33
    finally:
        client.close()
