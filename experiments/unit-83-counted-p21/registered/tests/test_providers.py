"""Authored cases for the carried base's adapters (41-D-COUNTED, section 3.1 and package A): the registered request
fields, the reading of a reply into a result record with hand-made bodies, and the failure and settlement classes of
section 11.2. Not run before both registrations read REGISTERED."""
from p21base import providers
from p21base.common import GateError

from support import load_config


def test_the_request_body_holds_the_registered_fields_only():
    config = load_config()
    fields = config['engines']['claude-haiku-5-5']['request_fields']
    body = providers.payload('anthropic', 'SYSTEM', 'USER', fields)
    assert body['system'] == 'SYSTEM' and body['messages'] == [{'role': 'user', 'content': 'USER'}]
    assert body['max_tokens'] == 16000 and body['output_config'] == {'effort': 'low'} and body['thinking'] == {'type': 'adaptive'}
    assert 'temperature' not in body and 'top_p' not in body and 'cache_control' not in body and 'tools' not in body
    body = providers.payload('openai', 'SYSTEM', 'USER', config['engines']['gpt-6-luna']['request_fields'])
    assert body['instructions'] == 'SYSTEM' and body['input'] == [{'role': 'user', 'content': 'USER'}]
    assert body['service_tier'] == 'flex' and body['store'] is False and body['max_output_tokens'] == 16000
    for bad in (dict(fields, temperature=0.0), dict(fields, max_tokens=8000)):
        try:
            providers.payload('anthropic', 'SYSTEM', 'USER', bad)
        except GateError:
            pass
        else:
            raise AssertionError('a field outside the registered ones was sent')


def anthropic(stop='end_turn', content=None, tier='standard', created=0, **usage):
    body = {'id': 'msg_1', 'model': 'claude-haiku-5-5', 'stop_reason': stop,
            'content': content if content is not None else [{'type': 'thinking', 'thinking': 'x'}, {'type': 'text', 'text': 'ACCEPT'}],
            'usage': {'input_tokens': 100, 'cache_creation_input_tokens': created, 'cache_read_input_tokens': 0,
                      'output_tokens': 40, 'service_tier': tier, 'output_tokens_details': {'thinking_tokens': 30}}}
    body['usage'].update(usage)
    return body


def test_reading_an_anthropic_reply():
    ok = providers.normalise('anthropic', anthropic())
    assert ok['status'] == 'ok' and ok['text'] == 'ACCEPT' and ok['output_tokens'] == 40 and ok['reasoning_tokens'] == 30
    assert ok['input_tokens'] == 100 and ok['returned_model'] == 'claude-haiku-5-5'
    assert providers.normalise('anthropic', anthropic(stop='max_tokens'))['status'] == providers.TRUNCATED
    refused = providers.normalise('anthropic', dict(anthropic(stop='refusal', content=[]), stop_details={'category': 'cyber'}))
    assert refused['status'] == providers.REFUSED and refused['refusal_category'] == 'cyber' and refused['input_tokens'] == 100
    assert providers.normalise('anthropic', anthropic(tier='priority'))['status'] == 'pricing_tier_mismatch'
    assert providers.normalise('anthropic', anthropic(created=500))['status'] == 'pricing_tier_mismatch'
    assert providers.normalise('anthropic', anthropic(tier=None))['status'] == 'ok'
    assert providers.normalise('anthropic', anthropic(content=[], output_tokens=0))['status'] == 'empty_transport'


def openai(status='completed', parts=None, tier='flex', **extra):
    body = {'id': 'resp_1', 'model': 'gpt-6-luna', 'status': status, 'service_tier': tier,
            'output': [{'type': 'reasoning'}, {'type': 'message', 'content': parts if parts is not None else [{'type': 'output_text', 'text': 'REJECT'}]}],
            'usage': {'input_tokens': 90, 'output_tokens': 25, 'output_tokens_details': {'reasoning_tokens': 20}}}
    body.update(extra)
    return body


def test_reading_an_openai_reply():
    ok = providers.normalise('openai', openai())
    assert ok['status'] == 'ok' and ok['text'] == 'REJECT' and ok['reasoning_tokens'] == 20
    cut = providers.normalise('openai', openai(status='incomplete', incomplete_details={'reason': 'max_output_tokens'}))
    assert cut['status'] == providers.TRUNCATED
    filtered = providers.normalise('openai', openai(status='incomplete', incomplete_details={'reason': 'content_filter'}))
    assert filtered['status'] == providers.REFUSED
    assert providers.normalise('openai', openai(parts=[{'type': 'refusal', 'refusal': 'no'}]))['status'] == providers.REFUSED
    assert providers.normalise('openai', openai(tier='default'))['status'] == 'pricing_tier_mismatch'


def test_the_failure_and_settlement_classes():
    assert providers.failure_class({'status': 'http_failure', 'http_status': 429}) == 'transport'
    assert providers.failure_class({'status': 'http_failure', 'http_status': 503}) == 'transport'
    assert providers.failure_class({'status': 'http_failure', 'http_status': 400}) == 'hold'
    assert providers.failure_class({'status': 'http_failure', 'http_status': 401}) == 'hold'
    assert providers.failure_class({'status': 'transport_unknown'}) == 'transport'
    assert providers.failure_class({'status': 'pricing_tier_mismatch'}) == 'hold'
    assert providers.failure_class({'status': providers.TRUNCATED}) is None
    assert providers.settlement_class({'status': 'http_failure', 'http_status': 429}) == 'uncharged'
    assert providers.settlement_class({'status': 'http_failure', 'http_status': 529}) == 'uncharged'
    assert providers.settlement_class({'status': 'http_failure', 'http_status': 502, 'error_body_json': True}) == 'uncharged'
    assert providers.settlement_class({'status': 'http_failure', 'http_status': 502, 'error_body_json': False}) == 'reservation'
    assert providers.settlement_class({'status': 'transport_unknown'}) == 'reservation'
    assert providers.settlement_class({'status': 'empty_transport', 'input_tokens': 10, 'output_tokens': 0}) == 'usage'
    assert providers.attempt_slot('s', 1) == 's' and providers.attempt_slot('s', 3) == 's#attempt3'
