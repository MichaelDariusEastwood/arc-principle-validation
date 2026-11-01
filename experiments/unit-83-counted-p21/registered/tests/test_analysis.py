"""Authored cases for the analysis (41-D-COUNTED, sections 6.3 and 7): the exact split, the worst-case bound on a block
worked by hand, the bounded-mean confirmation with its index-level exclusion (revision 4.2.1), the rule at each
boundary, the stopping rule, item 1's checks and the second verdict. Not run before both registrations read
REGISTERED."""
import math

from p21lane import analysis

from support import load_config, plain_block

TOL = 0.0075
GATE = 0.00206
ALPHA = 0.0025


def close(a, b, tol=1e-12):
    return abs(a - b) <= tol


def test_the_three_part_split_is_exact():
    v_in = [1, 1, -1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1]
    v_out = [0, 1, -1, 1, 0, 1, 1, 0, 1, -1, 1, 0, 1, -1, 1, 1, 0, 1]
    v_t = [0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, -1, 1, 1, 0, 1]
    q = analysis.block_quantities(plain_block(v_in, v_out, v_t))
    assert close(q['D'], q['E'] + q['V'] + q['S'])
    assert close(q['DC'], q['E'] + q['V'])
    assert close(q['level'], sum(i - o for i, o in zip(v_in, v_out)) / 18.0)


def test_the_worst_case_bound_on_a_worked_block():
    # The corrector's first invalid reply is at round 17, on a faulty draft it kept; rounds 17 and 18 are free for the
    # in-loop arm (minus 1 or plus 1); the monitor rejected every draft. Weights (2k - 19) / 969: rounds 17 and 18 carry
    # 15/969 and 17/969. D = -30/969; D- = -2(15 + 17)/969 = -64/969; D+ = 0; L- = (16 - 2)/18.
    v_in = [1] * 18
    v_in[16] = -1
    faulty = [False] * 18
    faulty[16] = True
    invalid = [False] * 18
    invalid[16] = True
    q = analysis.block_quantities(plain_block(v_in, [0] * 18, faulty=faulty, unusable={'C': invalid}))
    assert close(q['D'], -30 / 969.0)
    assert close(q['D_minus'], -64 / 969.0) and close(q['D_plus'], 0.0)
    assert close(q['L_minus'], 14 / 18.0)
    assert close(q['E_minus'], -64 / 969.0) and close(q['E_plus'], 0.0)
    assert q['first_unusable']['C'] == 16 and q['rests_D']


def test_with_every_reply_usable_the_bounds_are_the_values():
    q = analysis.block_quantities(plain_block([1] * 18, [0, 1] * 9))
    assert close(q['D_minus'], q['D']) and close(q['D_plus'], q['D']) and close(q['L_minus'], q['level'])
    assert not q['rests_D']


def test_a_free_monitor_takes_its_action_space():
    # The monitor's first unreadable verdict is at round 1: every round's outside value may be minus 1 or 0 on a faulty
    # draft and 0 or plus 1 on a sound one; the corrector keeps a sound entry in every round.
    faulty = [k % 2 == 0 for k in range(18)]
    unreadable = [True] + [False] * 17
    v_out = [-1 if f else 1 for f in faulty]
    q = analysis.block_quantities(plain_block([1] * 18, v_out, faulty=faulty, unusable={'M': unreadable}))
    a = [1 - (0 if f else 1) for f in faulty]
    b = [1 - (-1 if f else 0) for f in faulty]
    w = analysis.W
    assert close(q['D_minus'], sum(wk * (ak if wk > 0 else bk) for wk, ak, bk in zip(w, a, b)))
    assert close(q['D_plus'], sum(wk * (bk if wk > 0 else ak) for wk, ak, bk in zip(w, a, b)))
    assert close(q['L_minus'], sum(a) / 18.0)


def test_the_pair_bounds_and_the_factor():
    sb = analysis.SLOPE_BOUND
    assert close(sb, 81.0 / 323.0)
    assert close(analysis.factor(TOL, 'above', TOL, -sb, sb), 1.0)
    assert close(analysis.factor(-sb, 'above', TOL, -sb, sb), 0.5)
    assert close(analysis.factor(sb, 'below', TOL, -sb, sb), 0.5)
    try:
        analysis.factor(sb + 0.01, 'above', TOL, -sb, sb)
    except Exception as exc:
        assert 'outside its bounds' in str(exc) and str(sb + 0.01) not in str(exc)
    else:
        raise AssertionError('a value beyond its bounds was accepted')


def test_the_confirmation_leaves_out_the_whole_index():
    def rows(flags, pc_flags=None):
        return [{'D': 0.1, 'rests_D': f, 'PC': 0.1, 'rests_PC': (pc_flags or [False] * len(flags))[i]} for i, f in enumerate(flags)]
    pv = {'a': rows([False, False, False]), 'b': rows([False, True, False])}
    result = analysis.confirmation(pv, 'D>+tol', 3, TOL)
    f = 1.0 + (0.1 - TOL) / (2.0 * (TOL + 81.0 / 323.0))
    assert result['indices_left_out'] == [2] and result['n_left_out'] == 1 and result['n_read'] == 3
    assert close(result['log_value'], 4 * math.log(f))
    # A pair resting on a reply that only the positive control reads leaves its index out of the control's conditions.
    pv = {'a': rows([False, False], [False, True]), 'b': rows([False, False])}
    assert analysis.confirmation(pv, 'D>+tol', 2, TOL)['n_left_out'] == 0
    assert analysis.confirmation(pv, 'PC>+tol', 2, TOL)['indices_left_out'] == [2]


def test_the_confirmation_reaches_40_only_with_enough_evidence():
    # log 40 / log(1.17907...) is about 22.4 factors.
    reached = analysis.confirmation({'a': [{'D': 0.1, 'rests_D': False}] * 12, 'b': [{'D': 0.1, 'rests_D': False}] * 12},
                                    'D>+tol', 12, TOL)
    short = analysis.confirmation({'a': [{'D': 0.1, 'rests_D': False}] * 11, 'b': [{'D': 0.1, 'rests_D': False}] * 11},
                                  'D>+tol', 11, TOL)
    assert reached['reached'] and not short['reached']


def row(low, high, se=0.001, degenerate=False):
    return {'estimate': (low + high) / 2.0, 'se': se, 'df': 200.0, 'interval': None if degenerate else (low, high),
            'degenerate': degenerate, 'engines': {}}


def supported_estimates():
    return {'D': row(0.009, 0.02), 'D_minus': row(0.008, 0.02), 'D_plus': row(0.009, 0.021), 'level': row(0.1, 0.2),
            'L_minus': row(0.05, 0.2), 'PC': row(0.02, 0.03), 'PC_minus': row(0.019, 0.03)}


def conf(missing=()):
    return {name: {'reached': name not in missing} for name in analysis.CONDITIONS}


def flips(*agreeing):
    return {key: {'p': 0.001 if key in agreeing else 0.5} for key in ('above_+tol', 'below_+tol', 'above_-tol', 'below_-tol')}


PASSED = {'passed': True}


def evaluate(est, confirmation=None, flip=None, control=PASSED, directional=True):
    return analysis.evaluate_mark(est, confirmation or conf(), flip or flips(), control, GATE, TOL, directional, ALPHA)


def test_supported_and_each_reason_it_is_withheld():
    est = supported_estimates()
    assert evaluate(est, flip=flips('above_+tol'))[:3] == (analysis.SUPPORTED, None, None)
    assert evaluate(est, conf(missing=('L->0',)), flips('above_+tol'))[2] == 'confirmation'
    assert evaluate(est)[2] == 'disagree'
    bound = dict(est, D_minus=row(0.007, 0.02))
    assert evaluate(bound, flip=flips('above_+tol'))[2] == 'unusable'
    crossing = dict(est, level=row(-0.01, 0.2))
    assert evaluate(crossing, flip=flips('above_+tol'))[2] == 'crossing'


def refuted_estimates(d, d_minus, d_plus, se=0.001):
    return {'D': row(*d, se=se), 'D_minus': row(*d_minus, se=se), 'D_plus': row(*d_plus, se=se), 'level': row(0.1, 0.2),
            'L_minus': row(0.05, 0.2), 'PC': row(0.02, 0.03), 'PC_minus': row(0.019, 0.03)}


def test_refuted_in_each_form():
    constant = refuted_estimates((-0.005, 0.005), (-0.006, 0.005), (-0.005, 0.006))
    assert evaluate(constant, flip=flips('below_+tol', 'above_-tol'))[:2] == (analysis.REFUTED, 'constant')
    widening = refuted_estimates((-0.012, 0.004), (-0.013, 0.003), (-0.011, 0.005))
    assert evaluate(widening, flip=flips('below_+tol'))[:2] == (analysis.REFUTED, 'no widening')
    narrowing = refuted_estimates((-0.03, -0.01), (-0.031, -0.011), (-0.029, -0.009), se=0.005)
    assert evaluate(narrowing, flip=flips('below_-tol'))[:2] == (analysis.REFUTED, 'narrowing')
    # D- not clear of minus the tolerance: the bounds support only "no widening".
    weaker = refuted_estimates((-0.005, 0.005), (-0.009, 0.005), (-0.005, 0.006))
    assert evaluate(weaker, flip=flips('below_+tol'))[:2] == (analysis.REFUTED, 'no widening')


def test_each_reason_a_refutation_is_withheld():
    constant = refuted_estimates((-0.005, 0.005), (-0.006, 0.005), (-0.005, 0.006))
    assert evaluate(constant, flip=flips('below_+tol', 'above_-tol'), control={'passed': False})[2] == 'control'
    assert evaluate(constant, conf(missing=('PC->+tol',)), flips('below_+tol', 'above_-tol'))[2] == 'confirmation'
    narrowing = refuted_estimates((-0.03, -0.01), (-0.031, -0.011), (-0.02, -0.005), se=0.005)
    assert evaluate(narrowing, flip=flips('below_-tol'))[2] == 'unusable'
    widening = refuted_estimates((-0.012, 0.004), (-0.013, 0.003), (-0.011, 0.005))
    assert evaluate(widening, flip=flips('below_+tol'), directional=False)[2] == 'against'


def test_margin_precision_and_the_degenerate_interval():
    assert evaluate(refuted_estimates((0.005, 0.009), (0.004, 0.009), (0.005, 0.01)))[2] == 'margin'
    assert evaluate(refuted_estimates((-0.006, 0.006), (-0.006, 0.006), (-0.006, 0.006), se=0.003))[2] == 'precision'
    est = supported_estimates()
    est['D'] = row(0.0, 0.0, degenerate=True)
    assert evaluate(est)[2] == 'degenerate'


def test_the_stopping_rule():
    assert analysis.decide({'look': 1, 'mark': analysis.SUPPORTED}) == 'stop'
    assert analysis.decide({'look': 1, 'mark': analysis.REFUTED}) == 'stop'
    assert analysis.decide({'look': 1, 'mark': analysis.NOT_EVALUABLE}) == 'stop'
    assert analysis.decide({'look': 1, 'mark': analysis.CONTINUE}) == 'continue'
    assert analysis.decide({'look': 2, 'mark': analysis.INCONCLUSIVE}) == 'stop'
    # The one registered word of section 7.4 that would stop only on both verdicts (not the frozen word, 'mark').
    both = {'look': 1, 'mark': analysis.SUPPORTED, 'stopping_word': 'both'}
    assert analysis.decide(dict(both, mechanism={'verdict': 'inconclusive'})) == 'continue'
    assert analysis.decide(dict(both, mechanism={'verdict': 'unsupported'})) == 'stop'
    assert analysis.decide(dict(both, mark=analysis.NOT_EVALUABLE)) == 'stop'
    assert load_config()['stopping_word'] == 'mark'
    config = load_config()
    assert analysis.incomplete(config, 1, [], {})['mark'] == analysis.NOT_EVALUABLE
    second = analysis.incomplete(config, 2, [], {})
    assert second['mark'] == analysis.INCONCLUSIVE and second['reason'] == 'second_look'


def checks_pair(unusable_m=()):
    blocks = []
    for which in (1, 2):
        faulty = [(k % 2 == 0) == (which == 1) for k in range(18)]
        verdict = ['REJECT' if f else 'ACCEPT' for f in faulty]
        unusable = [k in unusable_m for k in range(18)] if which == 1 else [False] * 18
        blocks.append({'arms': ['C', 'M', 'T'], 'faulty': faulty, 'view': [k + 1 for k in range(18)],
                       'verdict': {'M': [None if u else v for u, v in zip(unusable, verdict)], 'T': list(verdict)},
                       'kept_sound': [True] * 18,
                       'unusable': {'C': [False] * 18, 'M': unusable, 'T': [False] * 18}})
    return {'blocks': blocks}


def test_item_one_on_each_engine_and_the_ceiling():
    config = load_config()
    good = analysis.engine_checks([checks_pair() for _ in range(10)], 10, config)
    assert good['admitted'] and close(good['discrimination'], 1.0) and close(good['net_repair'], 1.0)
    # Four of the monitor's 360 replies unusable after the re-ask: 0.011 of the arm, above the ceiling of 0.005.
    pairs = [checks_pair((5, 9) if i < 2 else ()) for i in range(10)]
    broken = analysis.engine_checks(pairs, 10, config)
    assert broken['broken_arms'] == ['M'] and not broken['admitted']
    assert close(broken['ceiling']['M']['share'], 4 / 360.0)


def test_the_second_verdict():
    est = {'E': row(-0.001, 0.001), 'E_minus': row(-0.002, 0.001), 'E_plus': row(-0.001, 0.002)}
    point = {'beta': -0.05, 'W': 0.012, 'fbar': 0.03, 'gbar': 0.5}
    boot = {'beta_interval': (-0.07, -0.03), 'W_interval': (0.009, 0.015), 'W_se': 0.0015, 'M_interval': (0.008, 0.016)}
    ok = {'passed': True}
    axis = analysis.mechanism_axis(analysis.SUPPORTED, ok, boot, point, est, GATE, TOL)
    assert axis['verdict'] == 'supported' and axis['fall_registered'] and not axis['labels']
    assert close(axis['beside']['M_minus'], est['E_minus']['estimate'] + 0.012)
    assert analysis.mechanism_axis(analysis.SUPPORTED, ok, dict(boot, W_interval=(0.001, 0.006)), point, est, GATE, TOL)['verdict'] == 'unsupported'
    assert analysis.mechanism_axis(analysis.SUPPORTED, ok, dict(boot, W_interval=(0.001, 0.006), W_se=0.003), point, est, GATE, TOL)['verdict'] == 'inconclusive'
    assert analysis.mechanism_axis(analysis.SUPPORTED, ok, dict(boot, W_interval=(0.005, 0.009)), point, est, GATE, TOL)['verdict'] == 'inconclusive'
    no_fall = analysis.mechanism_axis(analysis.SUPPORTED, ok, dict(boot, beta_interval=(-0.07, 0.01)), point, est, GATE, TOL)
    assert no_fall['verdict'] == 'inconclusive' and not no_fall['fall_registered']
    assert analysis.mechanism_axis(analysis.SUPPORTED, {'passed': False}, boot, point, est, GATE, TOL)['verdict'] == analysis.NOT_EVALUABLE
    assert analysis.mechanism_axis(analysis.NOT_EVALUABLE, ok, boot, point, est, GATE, TOL)['verdict'] == analysis.NOT_EVALUABLE
    driven = dict(est, E=row(0.008, 0.012))
    assert analysis.mechanism_axis(analysis.SUPPORTED, ok, boot, point, driven, GATE, TOL)['labels'] == ['corrector-driven']


def test_the_registered_fall_from_counts():
    # Sizes 1 and 2 with 20 or more faulty and sound calls each; size 3 left out with too few.
    totals = {1: [40, 38, 40, 2], 2: [40, 34, 40, 2], 3: [5, 3, 5, 0]}
    fall = analysis.fall_from_totals(totals, 20, 0.98)
    d1 = (38 / 40.0 - 2 / 40.0) / (1 - 2 / 40.0)
    d2 = (34 / 40.0 - 2 / 40.0) / (1 - 2 / 40.0)
    assert fall['sizes'] == [1, 2] and fall['left_out'] == [3]
    assert close(fall['beta'], d2 - d1) and close(fall['fbar'], 4 / 80.0)
    assert close(analysis._w_of(0.05, -0.02, 0.5), 0.5 * 0.95 * 0.02 * 0.5)


def test_the_constants_agree_with_the_configuration():
    analysis.check_constants(load_config())
