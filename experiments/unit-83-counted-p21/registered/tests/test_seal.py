"""Authored cases for the sealing rule (41-D-COUNTED, sections 4.9 and 6.4). Not run before registration."""
import hashlib

from p21base.common import GateError
from p21lane import seal

from support import load_config


def test_the_bank_seed_derivation():
    assert seal.derive('seed', 'abcde', '2026-11-03T05:00:00.123456') == hashlib.sha256(b'seed|abcde|2026-11-03T05:00:00.123456').digest()
    assert seal.derive('seed', 'abcde', 'when', 0) == hashlib.sha256(b'seed|abcde|when').digest()
    assert seal.derive('seed', 'abcde', 'when', 2) == hashlib.sha256(b'seed|abcde|when|2').digest()
    for bad in (('', 'abcde', 'when'), ('seed', '', 'when'), ('seed', 'abcde', '')):
        try:
            seal.derive(*bad)
        except GateError:
            pass
        else:
            raise AssertionError('an empty input was accepted')
    try:
        seal.derive('seed', 'abcde', 'when', -1)
    except GateError:
        pass
    else:
        raise AssertionError('a negative counter was accepted')


def test_the_keys_the_scorer_sees_carry_no_label():
    seed = b'\x01\x02'
    key = seal.case_key('gpt-6-luna:12:7', seed)
    assert key == hashlib.sha256(b'gpt-6-luna:12:7|0102').hexdigest()
    assert 'gpt' not in key and len(key) == 64
    assert seal.entry_key(key, 'ABCD|PAY|2027-01-04|-1250|DONE|RP') == hashlib.sha256(
        (key + '|ABCD|PAY|2027-01-04|-1250|DONE|RP').encode('utf-8')).hexdigest()


def test_the_engines_in_their_registered_order():
    assert seal.engines_of(load_config()) == ['claude-haiku-5-5', 'gpt-6-luna']
