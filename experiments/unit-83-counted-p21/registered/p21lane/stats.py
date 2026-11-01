"""Numerical helpers for the analysis (41-D-COUNTED, sections 7.2 to 7.10), standard library only.

Package G of the build list (DESIGN-FREEZE.md, section 12). The t and normal quantiles are hand-written for the
standard library and are checked at step 2 of section 11.3 against published table values (tests/test_stats.py). Every
function is pure. Nothing here is executed before both registrations read REGISTERED (SD-001, SD-012); before then the
only mechanical check is a parse of this file's text.
"""
import hashlib
import math
import random

ROUNDS = 18


def slope_weights(rounds=ROUNDS):
    """The least-squares weights on k = 1 to rounds: w_k = (k - mean) / sum of squared deviations; at eighteen rounds
    (2k - 19) / 969, negative in rounds 1 to 9 and positive in rounds 10 to 18, none zero."""
    centre = (rounds + 1) / 2.0
    total = sum((k - centre) ** 2 for k in range(1, rounds + 1))
    return [(k - centre) / total for k in range(1, rounds + 1)]


WEIGHTS = slope_weights()


def slope(values, weights=WEIGHTS):
    return sum(w * v for w, v in zip(weights, values))


def mean(values):
    values = list(values)
    if not values:
        raise ValueError('the mean of nothing')
    return math.fsum(values) / len(values)


def sample_variance(values):
    values = list(values)
    n = len(values)
    if n < 2:
        return float('nan')
    m = mean(values)
    return math.fsum((v - m) ** 2 for v in values) / (n - 1)


# ---- The normal distribution.

def normal_cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def normal_quantile(p):
    """The inverse of the standard normal distribution: Acklam's rational approximation, refined by two Newton steps
    on math.erfc (absolute error far below 1e-12 in the region used)."""
    if not 0.0 < p < 1.0:
        raise ValueError('a probability strictly between 0 and 1')
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00)
    low = 0.02425
    if p < low:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= 1 - low:
        q = p - 0.5
        r = q * q
        x = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    for _ in range(2):
        error = normal_cdf(x) - p
        x -= error * math.sqrt(2 * math.pi) * math.exp(x * x / 2.0)
    return x


# ---- The t distribution, through the regularised incomplete beta function.

def _betacf(a, b, x):
    """The continued fraction of the incomplete beta function, by the modified Lentz method."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 10000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        step = d * c
        h *= step
        if abs(step - 1.0) < 1e-15:
            return h
    raise ArithmeticError('the incomplete beta continued fraction did not converge')


def betainc(a, b, x):
    """The regularised incomplete beta function I_x(a, b)."""
    if not 0.0 <= x <= 1.0:
        raise ValueError('x lies in [0, 1]')
    if x == 0.0 or x == 1.0:
        return x
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t, df):
    """The distribution function of Student's t at real degrees of freedom df > 0."""
    if df <= 0:
        raise ValueError('degrees of freedom must be positive')
    if math.isinf(df):
        return normal_cdf(t)
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t >= 0 else tail


def t_quantile(p, df):
    """The inverse of t_cdf, by bracketing and bisection to a relative precision of about 1e-13."""
    if not 0.0 < p < 1.0:
        raise ValueError('a probability strictly between 0 and 1')
    if math.isinf(df) or df > 1e7:
        return normal_quantile(p)
    if p < 0.5:
        return -t_quantile(1.0 - p, df)
    if p == 0.5:
        return 0.0
    low, high = 0.0, max(1.0, normal_quantile(p))
    while t_cdf(high, df) < p:
        low, high = high, high * 2.0
        if high > 1e12:
            raise ArithmeticError('no t quantile found')
    for _ in range(200):
        middle = (low + high) / 2.0
        if t_cdf(middle, df) < p:
            low = middle
        else:
            high = middle
        if high - low <= 1e-13 * max(1.0, high):
            break
    return (low + high) / 2.0


def two_sided_quantile(level, df=float('inf')):
    return t_quantile((1.0 + level) / 2.0, df)


# ---- The pooled estimator of section 7.2.

def pooled(values_by_engine):
    """Equal-weight pooling over engines: each engine's mean over its pair values, with variance the sample variance
    over the pair count; the pooled mean the equal-weight mean, its variance the sum of the engine variances over the
    square of the number of engines, its degrees of freedom by Satterthwaite's formula."""
    engines = sorted(values_by_engine)
    if not engines:
        raise ValueError('no engine to pool')
    rows = {}
    for engine in engines:
        values = list(values_by_engine[engine])
        n = len(values)
        m = mean(values)
        var_mean = sample_variance(values) / n if n >= 2 else float('nan')
        rows[engine] = {'n': n, 'mean': m, 'var_mean': var_mean, 'se': math.sqrt(var_mean) if var_mean == var_mean and var_mean >= 0 else float('nan')}
    j = len(engines)
    estimate = math.fsum(rows[e]['mean'] for e in engines) / j
    variance = math.fsum(rows[e]['var_mean'] for e in engines) / (j * j)
    denominator = math.fsum((rows[e]['var_mean'] ** 2) / (rows[e]['n'] - 1) for e in engines if rows[e]['n'] > 1)
    total = math.fsum(rows[e]['var_mean'] for e in engines)
    df = (total * total) / denominator if denominator > 0 else float('nan')
    return {'estimate': estimate, 'variance': variance, 'se': math.sqrt(variance) if variance >= 0 else float('nan'),
            'df': df, 'engines': rows, 'degenerate': not variance > 0}


def t_interval(pooled_row, level):
    """The two-sided t interval at the look's level, or None where the pooled variance is zero (degenerate)."""
    if pooled_row['degenerate'] or not pooled_row['df'] > 0:
        return None
    half = two_sided_quantile(level, pooled_row['df']) * pooled_row['se']
    return (pooled_row['estimate'] - half, pooled_row['estimate'] + half)


def engine_interval(values, level):
    """One engine's own t interval at a level, with n - 1 degrees of freedom; None if degenerate."""
    values = list(values)
    n = len(values)
    if n < 2:
        return None
    var = sample_variance(values)
    if not var > 0:
        return None
    half = two_sided_quantile(level, n - 1) * math.sqrt(var / n)
    m = mean(values)
    return (m - half, m + half)


def percentile(sorted_values, q):
    """The q quantile of sorted values by linear interpolation between order statistics (Hyndman and Fan's type 7)."""
    n = len(sorted_values)
    if n == 0:
        return float('nan')
    position = (n - 1) * q
    below = int(math.floor(position))
    above = min(below + 1, n - 1)
    weight = position - below
    return sorted_values[below] * (1 - weight) + sorted_values[above] * weight


def percentile_interval(values, level):
    finite = sorted(v for v in values if isinstance(v, float) and math.isfinite(v))
    if not finite:
        return None
    return (percentile(finite, (1 - level) / 2.0), percentile(finite, (1 + level) / 2.0))


def seed_int(text):
    """A registered seed written as text, read as an integer by SHA-256."""
    return int.from_bytes(hashlib.sha256(text.encode('utf-8')).digest(), 'big')


# ---- Exact binomial bounds, for admission's report (section 11.3, step 4).

def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    logp, logq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(0, k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * logp + (n - i) * logq)
    return min(1.0, total)


def binomial_upper(k, n, confidence=0.95):
    """The exact one-sided upper confidence bound for a rate seen k times in n (Clopper and Pearson); with k = 0 it is
    1 minus (1 - confidence) to the power 1/n: 0.0060 at n = 500 and 0.0099 at n = 300."""
    if n <= 0:
        return 1.0
    if k >= n:
        return 1.0
    alpha = 1.0 - confidence
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    low, high = k / n, 1.0
    for _ in range(200):
        middle = (low + high) / 2.0
        if binomial_cdf(k, n, middle) > alpha:
            low = middle
        else:
            high = middle
    return high


# ---- The sign-flip check of section 7.5.

def _chunk_tables(values):
    """For each run of eight values, the 256 signed sums: a bit set gives +value, a bit clear gives -value."""
    tables = []
    for start in range(0, len(values), 8):
        chunk = values[start:start + 8]
        table = []
        for byte in range(256):
            total = 0.0
            for bit, value in enumerate(chunk):
                total += value if (byte >> bit) & 1 else -value
            table.append(total)
        tables.append(table)
    return tables


def sign_flip_p(values_by_engine, constant, direction, flips, seed_text):
    """The one-sided Monte Carlo p-value of the equal-weight statistic on the pair values less `constant`, each pair's
    sign flipped independently within its engine: (1 + the flips at least as extreme) / (1 + flips). direction 'upper'
    tests a mean above the constant, 'lower' a mean below it."""
    if direction not in ('upper', 'lower'):
        raise ValueError('direction is upper or lower')
    engines = sorted(values_by_engine)
    centred = {e: [v - constant for v in values_by_engine[e]] for e in engines}
    observed = math.fsum(mean(centred[e]) for e in engines) / len(engines)
    tables = {e: _chunk_tables(centred[e]) for e in engines}
    sizes = {e: len(centred[e]) for e in engines}
    rng = random.Random(seed_int(seed_text))
    extreme = 0
    tolerance = 1e-12
    for _ in range(flips):
        total = 0.0
        for e in engines:
            table = tables[e]
            raw = rng.getrandbits(8 * len(table)).to_bytes(len(table), 'little')
            total += sum(table[i][raw[i]] for i in range(len(table))) / sizes[e]
        statistic = total / len(engines)
        if direction == 'upper' and statistic >= observed - tolerance:
            extreme += 1
        elif direction == 'lower' and statistic <= observed + tolerance:
            extreme += 1
    return {'observed': observed, 'p': (1 + extreme) / (1 + flips), 'flips': flips, 'constant': constant,
            'direction': direction}
