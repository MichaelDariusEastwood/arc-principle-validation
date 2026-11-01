"""Authored cases for the reader (41-D-COUNTED, section 6.1). Not run before both registrations read REGISTERED."""
from p21lane import reading

GOOD = 'ABCD|PAY|2027-01-04|-1250|DONE|RP'


def test_one_bare_line():
    entry, reason = reading.read_entry(GOOD, 'ABCD', 'PAY')
    assert reason is None and entry.amount == -1250 and entry.flags == frozenset({'RP'})


def test_words_around_and_one_fence():
    text = 'Checked against G1 to G5.\n```\n' + GOOD + '\n```\nThis is the entry as it should stand.'
    assert reading.read_entry(text, 'ABCD', 'PAY')[1] is None


def test_leniencies_of_section_6_1():
    entry, reason = reading.read_entry(' ABCD | PAY | 2027-01-04 | 1250 | DONE | RP ', 'ABCD', 'PAY')
    assert reason is None and entry.amount == 1250
    entry, reason = reading.read_entry('ABCD|FEE|2027-01-04|-900|DONE|CAP', 'ABCD', 'FEE')
    assert reason is None and entry.flags == frozenset({'CAP'})


def test_no_entry():
    assert reading.read_entry('', 'ABCD', 'PAY') == (None, reading.NO_ENTRY)
    assert reading.read_entry('Entry: ' + GOOD, 'ABCD', 'PAY') == (None, reading.NO_ENTRY)
    assert reading.read_entry('`' + GOOD + '`', 'ABCD', 'PAY') == (None, reading.NO_ENTRY)
    assert reading.read_entry(GOOD + '.', 'ABCD', 'PAY') == (None, reading.NO_ENTRY)
    assert reading.read_entry('Q0X1|PAY|2027-01-04|-1250|DONE|RP', 'Q0X1', 'PAY') == (None, reading.NO_ENTRY)


def test_many_entries():
    assert reading.read_entry(GOOD + '\n' + GOOD, 'ABCD', 'PAY') == (None, reading.MANY_ENTRIES)


def test_wrong_case():
    assert reading.read_entry(GOOD, 'ABCE', 'PAY') == (None, reading.WRONG_CASE)
    assert reading.read_entry(GOOD, 'ABCD', 'FEE') == (None, reading.WRONG_CASE)


def test_verdicts_and_labels():
    for text in ('REJECT', '**REJECT**', 'reject.', 'All checked.\nVerdict: REJECT', 'FINAL ANSWER : reject',
                 '- REJECT', 'Answer: **REJECT**', 'REJECT\n```', 'REJECT\n---'):
        assert reading.read_verdict(text) == 'REJECT', text
    assert reading.read_verdict('Decision: ACCEPT') == 'ACCEPT'


def test_verdict_refusals():
    assert reading.verdict_and_reason('') == (None, reading.EMPTY_REPLY)
    assert reading.verdict_and_reason('```\n```') == (None, reading.EMPTY_REPLY)
    for text in ('ACCEPT or REJECT', 'Result: REJECT', 'REJECT, because P2 is wrong', 'REJECT\nThanks.',
                 'I would REJECT', 'VERDICT - REJECT'):
        assert reading.verdict_and_reason(text) == (None, reading.LAST_LINE_UNREADABLE), text
