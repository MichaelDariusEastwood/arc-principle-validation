"""The Monte Carlo interval on a reported rate: the arithmetic a reader put to the programme on
5 September 2026, reproduced exactly, and the two counts kept apart."""
import math

import pytest

from arc_instruments import mc_interval as MC


def test_zero_in_four_hundred_has_the_stated_one_sided_upper_bound():
    # 1 - 0.05 ** (1/400) = 0.00746
    up = MC.exact_upper(0, 400, 0.95)
    assert abs(up - (1 - 0.05 ** (1 / 400))) < 1e-6
    assert abs(up - 0.00746) < 5e-5


def test_twenty_in_four_hundred_is_five_per_cent_with_an_upper_bound_near_seven():
    up = MC.exact_upper(20, 400, 0.95)
    assert abs(up - 0.0718) < 1e-3, up
    assert 20 / 400 == 0.05 and up > 0.05


def test_the_binomial_cdf_matches_a_direct_sum():
    n, p = 12, 0.3
    for k in range(-1, n + 2):
        direct = sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(0, min(k, n) + 1)) if k >= 0 else 0.0
        assert abs(MC.binom_cdf(k, n, p) - direct) < 1e-12


def test_exact_two_sided_brackets_the_rate_and_contains_the_truth_in_most_worlds():
    lo, hi = MC.clopper_pearson(38, 40)
    assert lo < 0.95 < hi
    assert lo > 0.8 and hi < 1.0
    lo0, hi0 = MC.clopper_pearson(0, 40)
    assert lo0 == 0.0 and 0.05 < hi0 < 0.1


def test_wilson_and_exact_agree_to_the_first_decimal_at_moderate_n():
    for k, n in ((5, 40), (20, 40), (35, 40)):
        el, eh = MC.clopper_pearson(k, n)
        wl, wh = MC.wilson(k, n)
        assert abs(el - wl) < 0.05 and abs(eh - wh) < 0.05


def test_the_two_counts_are_named_and_the_inner_count_does_not_narrow_the_interval():
    a = MC.RateWithUncertainty(successes=40, outer_repetitions=40, inner_resamples=40)
    b = MC.RateWithUncertainty(successes=40, outer_repetitions=40, inner_resamples=4000)
    assert a.as_dict()["exact_two_sided"] == b.as_dict()["exact_two_sided"]
    assert "inner resamples" in a.render() and "40 of 40" in a.render()
    assert a.as_dict()["exact_one_sided_lower"] > 0.9, "40 of 40 still leaves a lower bound below 1"
