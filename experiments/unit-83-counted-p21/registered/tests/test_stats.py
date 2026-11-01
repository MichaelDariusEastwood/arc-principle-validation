"""Authored cases for the numerical helpers (41-D-COUNTED, sections 7.2 to 7.5): the slope's weights, the quantiles
against published table values (section 11.3, step 2), the pooled estimator, the exact binomial bound and the sign-flip
check. Not run before both registrations read REGISTERED."""
import math

from p21lane import stats

from support import load_config


def close(a, b, tol):
    return abs(a - b) <= tol


def test_the_slope_weights():
    assert all(close(w, (2 * k - 19) / 969.0, 1e-15) for k, w in enumerate(stats.WEIGHTS, 1))
    assert all(w < 0 for w in stats.WEIGHTS[:9]) and all(w > 0 for w in stats.WEIGHTS[9:])
    assert close(stats.slope(list(range(1, 19))), 1.0, 1e-12)
    assert close(stats.slope([3.0] * 18), 0.0, 1e-12)


def test_normal_quantiles_against_published_values():
    for p, z in ((0.975, 1.959964), (0.995, 2.575829), (0.9975, 2.807034), (0.9775, 2.004654), (0.999, 3.090232),
                 (0.9995, 3.290527)):
        assert close(stats.normal_quantile(p), z, 5e-7), p
        assert close(stats.normal_quantile(1 - p), -z, 5e-7), p


def test_t_quantiles_against_published_table_values():
    for p, df, t in ((0.975, 1, 12.706), (0.975, 2, 4.303), (0.975, 5, 2.571), (0.975, 10, 2.228), (0.975, 30, 2.042),
                     (0.995, 10, 3.169), (0.995, 30, 2.750), (0.9995, 10, 4.587)):
        assert close(stats.t_quantile(p, df), t, 6e-4), (p, df)
    assert close(stats.t_quantile(0.975, 1e9), 1.959964, 1e-6)
    assert close(stats.t_cdf(stats.t_quantile(0.9, 7.3), 7.3), 0.9, 1e-10)


def test_the_precision_gates_follow_their_formula():
    looks = load_config()['looks']
    for level, gate in zip(looks['levels'], looks['gates']):
        z = stats.normal_quantile((1 + level) / 2.0)
        assert round(0.0075 / (z + 0.8416), 5) == gate


def test_the_pooled_estimator():
    row = stats.pooled({'a': [0.0, 2.0], 'b': [1.0, 1.0, 4.0]})
    assert close(row['estimate'], 1.5, 1e-12) and close(row['variance'], 0.5, 1e-12)
    assert close(row['df'], 8.0 / 3.0, 1e-12) and not row['degenerate']
    interval = stats.t_interval(row, 0.95)
    half = stats.t_quantile(0.975, 8.0 / 3.0) * math.sqrt(0.5)
    assert close(interval[0], 1.5 - half, 1e-9) and close(interval[1], 1.5 + half, 1e-9)
    flat = stats.pooled({'a': [1.0, 1.0], 'b': [2.0, 2.0]})
    assert flat['degenerate'] and stats.t_interval(flat, 0.95) is None


def test_the_exact_binomial_upper_bound():
    assert close(stats.binomial_upper(0, 500), 0.0059736, 1e-6)
    assert close(stats.binomial_upper(0, 300), 0.0099361, 1e-6)
    upper = stats.binomial_upper(3, 500)
    assert close(stats.binomial_cdf(3, 500, upper), 0.05, 1e-6)


def test_percentiles_by_linear_interpolation():
    assert stats.percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5
    assert stats.percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0 and stats.percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    low, high = stats.percentile_interval([float(v) for v in range(101)], 0.9)
    assert close(low, 5.0, 1e-9) and close(high, 95.0, 1e-9)


def test_the_sign_flip_check():
    far = {'a': [0.05] * 40, 'b': [0.06] * 40}
    upper = stats.sign_flip_p(far, 0.0075, 'upper', 2000, 'seed')
    assert upper['p'] == 1.0 / 2001 and close(upper['observed'], 0.0475, 1e-12)
    lower = stats.sign_flip_p(far, 0.0075, 'lower', 2000, 'seed')
    assert lower['p'] == 1.0
    assert stats.sign_flip_p(far, 0.0075, 'upper', 500, 'seed') == stats.sign_flip_p(far, 0.0075, 'upper', 500, 'seed')
