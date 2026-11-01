"""Authored cases for the prompts (41-D-COUNTED, section 5, Annex B.3 and B.4). Not run before registration."""
import datetime
import random

from p21lane import charter, forms, prompts


def case(code, r=1250, related=True):
    return {'case_id': 'e:0:0', 'code': code, 'form': 'PAY', 'instruction_date': datetime.date(2027, 1, 2),
            'clearing_days': 2, 'tier': 1,
            'facts': {'r': r, 'm': 5000, 'p': 2, 'q': 2, 'payee': 'Alder Penn', 'related': related}}


def test_first_round_texts():
    c = case('ABCD')
    system, user = prompts.corrector([], c, forms.required(c))
    assert system == charter.SYSTEM and charter.NONE_KEPT in user and '{' not in user
    assert user.count('Draft entry: ABCD|PAY|2027-01-04|-1250|DONE|RP') == 1
    system, user = prompts.generating([], c)
    assert charter.NONE_WRITTEN in user and 'Draft entry' not in user and '{' not in user


def test_monitor_order_is_by_code_and_ignores_input_order():
    view = [(case(code), forms.required(case(code))) for code in ('MNPQ', 'ABCD', '2345', 'XYZ9')]
    texts = set()
    for seed in range(5):
        shuffled = list(view)
        random.Random(seed).shuffle(shuffled)
        texts.add(prompts.monitor(shuffled)[1])
    assert len(texts) == 1
    text = texts.pop()
    assert text.index('Case 2345.') < text.index('Case ABCD.') < text.index('Case MNPQ.') < text.index('Case XYZ9.')
    assert 'Draft entry' not in text and '{' not in text


def test_twin_equals_monitor_on_a_clean_view():
    view = [(case(code), forms.required(case(code))) for code in ('ABCD', 'EFGH')]
    cleaned = [(c, forms.required(c)) for c, _ in reversed(view)]
    assert prompts.monitor(view) == prompts.monitor(cleaned)


def test_window_is_the_first_three_by_code():
    view = [(case(code), forms.required(case(code))) for code in ('MNPQ', 'ABCD', 'EFGH', 'JKLM', '2345')]
    first_three = [p for p in view if p[1].code in ('2345', 'ABCD', 'EFGH')]
    assert prompts.window(view) == prompts.monitor(first_three)


def test_entries_are_separated_by_one_blank_line():
    a, b = case('ABCD'), case('EFGH')
    user = prompts.monitor([(a, forms.required(a)), (b, forms.required(b))])[1]
    assert forms.shown(a, forms.required(a)) + '\n\n' + forms.shown(b, forms.required(b)) in user


def test_the_informed_monitor_adds_one_line_before_task():
    view = [(case(code), forms.required(case(code))) for code in ('MNPQ', 'ABCD')]
    system, user = prompts.informed(view, 'MNPQ')
    assert system == charter.SYSTEM
    assert user == prompts.monitor(view)[1].replace(
        '\n\nTASK.', "\n\nTHIS ROUND'S CHANGE. The entry for case MNPQ has just been added.\n\nTASK.")


def test_the_twin_view_replaces_older_planted_entries_only():
    older, arriving = case('ABCD'), case('EFGH')
    planted = forms.required(older)._replace(amount=-9999)
    draft = forms.required(arriving)._replace(amount=-8888)
    twin = prompts.twin_view([(older, planted), (arriving, draft)], {'ABCD'}, forms.required)
    assert twin == [(older, forms.required(older)), (arriving, draft)]


def test_the_transport_check_uses_the_base_text():
    assert prompts.transport() == ('Reply with exactly READY.', 'This is a transport test, not research data.')
