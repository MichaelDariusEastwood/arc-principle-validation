"""Authored cases for the generator, the schedules and the public sets (41-D-COUNTED, sections 4.5 to 4.8). Property 1
needs the oracle of package C, written by another builder; where these cases call generate.check they pass a stand-in
oracle that reads forms.required, so they test the other properties and never stand for property 1. Not run before
both registrations read REGISTERED."""
import datetime
import math

from p21lane import departures, forms, generate, schedule

SEED = b'test-seed-not-the-registered-one'


class StandInOracle:
    """Reads forms.required; never a substitute for package C, which init requires."""
    @staticmethod
    def meets(entry, case):
        return entry == forms.required(case)

    @staticmethod
    def solve(case):
        return [forms.required(case)]


def test_the_stream_is_determined_by_its_seed_and_label():
    a, b, c = generate.Stream(SEED, 'x'), generate.Stream(SEED, 'x'), generate.Stream(SEED, 'y')
    first = [a.below(1000) for _ in range(50)]
    assert first == [b.below(1000) for _ in range(50)]
    assert first != [c.below(1000) for _ in range(50)]
    s = generate.Stream(SEED, 'z')
    for n in (1, 2, 3, 7, 32, 1000, 10 ** 9):
        assert all(0 <= s.below(n) < n for _ in range(20))
    picked = s.sample(range(1, 19), 9)
    assert len(set(picked)) == 9 and set(picked) <= set(range(1, 19))
    items = list(range(18))
    s.shuffle(items)
    assert sorted(items) == list(range(18))


def _naive_right(form, pattern):
    if form == 'PAY':
        return pattern == ('none', False)
    if form == 'FEE':
        return pattern[0] and pattern[1] == 'fits' and pattern[2] in ('0', '1')
    if form == 'INT':
        return pattern == (False, 'high')
    if form == 'SPL':
        return pattern in (('first', 'exact'), ('other', 'low'), ('other', 'exact'))
    return pattern == 'large'


def _s_operators(form, pattern):
    if form == 'PAY':
        return 1 + (pattern[0] != 'none') + bool(pattern[1])
    if form == 'FEE':
        authority, cap, residue = pattern
        if not authority or cap in ('partial', 'nil'):
            return 2
        return 1 + (residue != '0')
    if form == 'INT':
        return 2
    if form == 'SPL':
        return 1 + (pattern in (('first', 'left'), ('other', 'high'), ('other', 'low')))
    return 1 + (pattern != 'nil')


def test_the_pattern_tables_give_the_expectations_of_build_decision_9():
    naive = {'PAY': {1: 0.20, 2: 0.20, 3: 0.15}, 'FEE': {1: 0.20, 2: 0.15, 3: 0.15}, 'INT': {1: 0.26, 2: 0.28, 3: 0.28},
             'SPL': {1: 0.25, 2: 0.25, 3: 0.25}, 'RFD': {1: 0.35, 2: 0.35, 3: 0.35}}
    two_s = {'PAY': {1: 0.80, 2: 0.80, 3: 0.85}, 'FEE': {1: 0.80, 2: 0.925, 3: 0.925}, 'INT': {1: 1.0, 2: 1.0, 3: 1.0},
             'SPL': {1: 0.92, 2: 0.92, 3: 0.92}, 'RFD': {1: 0.75, 2: 0.75, 3: 0.75}}
    for form, tables in generate.PATTERNS.items():
        for tier, table in tables.items():
            total = sum(weight for _, weight in table)
            assert total == 1000
            right = sum(weight for pattern, weight in table if _naive_right(form, pattern)) / total
            two = sum(weight for pattern, weight in table if _s_operators(form, pattern) >= 2) / total
            assert math.isclose(right, naive[form][tier]) and right <= 0.35 + 1e-12, (form, tier, right)
            assert math.isclose(two, two_s[form][tier]) and two >= 0.75 - 1e-12, (form, tier, two)


def test_a_bank_holds_eighteen_fresh_cases_a_pair():
    for tier in (1, 2, 3):
        bank = generate.make_bank(SEED, tier, 'claude-haiku-5-5', 2)
        assert bank['tier'] == tier and len(bank['pairs']) == 2
        for pair in bank['pairs']:
            cases = pair['cases']
            assert sorted(c['form'] for c in cases) == sorted(list(forms.FORMS) * 3)
            assert len({c['code'] for c in cases}) == 18
            for position, c in enumerate(cases, 1):
                assert c['case_id'] == 'claude-haiku-5-5:%d:%d' % (pair['pair'], position)
                entry = forms.required(c)
                assert entry.date.year == 2027 and (entry.date + datetime.timedelta(days=1)).year == 2027
                assert ('extra' in c['facts']) == (tier == 3)
                kinds = {k for _, k, _ in departures.planted(c)}
                assert kinds == {departures.S, departures.T}
    again = generate.make_bank(SEED, 2, 'claude-haiku-5-5', 2)
    assert generate.bank_to_json(again) == generate.bank_to_json(generate.make_bank(SEED, 2, 'claude-haiku-5-5', 2))


def test_each_engine_has_its_own_pairs():
    a = generate.make_bank(SEED, 2, 'claude-haiku-5-5', 1)
    b = generate.make_bank(SEED, 2, 'gpt-6-luna', 1)
    assert [c['code'] for c in a['pairs'][0]['cases']] != [c['code'] for c in b['pairs'][0]['cases']]


def test_schedules_are_complementary_and_planted_lines_depart():
    bank = generate.make_bank(SEED, 2, 'gpt-6-luna', 3)
    plan = schedule.make('gpt-6-luna', 3, bank, SEED)
    for pair, entry in zip(bank['pairs'], plan['pairs']):
        faulty = entry['faulty_block1']
        assert len(faulty) == 9 and len(set(faulty)) == 9 and set(faulty) <= set(range(1, 19))
        one = schedule.block_drafts(entry, pair['cases'], 1)
        two = schedule.block_drafts(entry, pair['cases'], 2)
        for (c1, d1, f1, _), (c2, d2, f2, _) in zip(one, two):
            assert c1 is c2 and f1 != f2
            planted = d1 if f1 else d2
            assert planted != forms.required(c1) and planted in [e for _, _, e in departures.planted(c1)]
        assert sum(f for _, _, f, _ in one) == 9 and sum(f for _, _, f, _ in two) == 9


def test_check_passes_a_sound_bank_with_a_stand_in_oracle():
    # Two pairs give each form six cases, and the shares of properties 2 and 3, registered for a bank's 1,980 cases a
    # form, fail by chance alone in most draws of that size; so only those two draw-dependent properties may refuse
    # here, and properties 1, 4, 5 and 6 and the schedule's rule must pass.
    bank = generate.make_bank(SEED, 1, 'claude-haiku-5-5', 2)
    plan = schedule.make('claude-haiku-5-5', 2, bank, SEED)
    try:
        counts = generate.check(bank, plan, StandInOracle)
    except generate.PropertyFailure as failure:
        assert failure.number in (2, 3), str(failure)
        return
    assert counts['cases'] == 36 and all(n == 6 for n in counts['by_form'].values())


def test_check_names_the_property_that_fails():
    bank = generate.make_bank(SEED, 1, 'claude-haiku-5-5', 1)
    bank['pairs'][0]['cases'][1]['code'] = bank['pairs'][0]['cases'][0]['code']
    try:
        generate.check(bank, None, StandInOracle)
    except generate.PropertyFailure as failure:
        assert failure.number == 5
    else:
        raise AssertionError('a repeated code passed the check')


def test_a_broken_schedule_is_an_implementation_defect():
    # Section 4.7's rule, not a draw-dependent property: never a PropertyFailure that init would answer with the next
    # bank of the registered sequence.
    bank = generate.make_bank(SEED, 1, 'claude-haiku-5-5', 1)
    plan = schedule.make('claude-haiku-5-5', 1, bank, SEED)
    plan['pairs'][0]['faulty_block1'] = plan['pairs'][0]['faulty_block1'][:8]
    try:
        generate.check(bank, plan, StandInOracle)
    except generate.PropertyFailure:
        raise AssertionError('a broken schedule was read as a property of the draw')
    except Exception as exc:
        assert 'implementation defect' in str(exc)
    else:
        raise AssertionError('a broken schedule passed the check')


def test_the_public_sets():
    sets = generate.public_sets('test-public-seed')
    assert len(sets['format3']) == 3 and [i['faulty'] for i in sets['format3']] == [False, True, False]
    assert len(sets['verdict14']) == 14 and sum(i['faulty'] for i in sets['verdict14']) == 2
    assert len(sets['register18']) == 18 and sets['register18'][-1]['faulty'] and sum(i['faulty'] for i in sets['register18']) == 1
    for tier in ('1', '2', '3'):
        monitor, corrector = sets['calibration'][tier]['monitor'], sets['calibration'][tier]['corrector']
        assert len(monitor) == 240 and sum(i['faulty'] for i in monitor) == 120
        assert len(corrector) == 120 and sum(i['faulty'] for i in corrector) == 80
    generate.check_public(sets, StandInOracle)
    assert generate.public_from_json(generate.public_to_json(sets))['register18'][0]['case'] == sets['register18'][0]['case']


def test_serialisation_keeps_every_field():
    bank = generate.make_bank(SEED, 3, 'gpt-6-luna', 1)
    for c in bank['pairs'][0]['cases']:
        assert generate.case_from_json(generate.case_to_json(c)) == c
