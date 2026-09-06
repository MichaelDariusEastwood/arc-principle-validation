import numpy as np

from arc_instruments import form_discrimination as fd


def test_ladder_span():
    r = fd.ladder(7, 1.5)
    assert len(r) == 7 and abs(r[0] - 1.0) < 1e-12 and abs(r[-1] - 10 ** 1.5) < 1e-9


def test_power_and_exponential_recovered_at_registered_precision():
    truths = {
        "power a=1.5": ("power", lambda r: fd.power(r, 1.0, 1.5)),
        "exponential b=0.06": ("exponential", lambda r: fd.expo(r, 1.0, 0.06)),
    }
    m = fd.confusion_matrix(points=7, decades=1.5, noise=0.02, reps=30, truths=truths, seed=3)
    rec = fd.recovery_rates(m)
    assert rec["power a=1.5"] >= 0.9
    assert rec["exponential b=0.06"] >= 0.9
    assert 0.0 <= fd.false_selection_rate(m) <= 0.1


def test_matrix_rows_sum_to_one():
    truths = {"power a=0.5": ("power", lambda r: fd.power(r, 1.0, 0.5))}
    m = fd.confusion_matrix(points=7, decades=1.5, noise=0.05, reps=12, truths=truths, seed=5)
    row = m["power a=0.5"]
    assert abs(sum(v for k, v in row.items() if k != "_true_family") - 1.0) < 1e-9


def test_rival_detection_rates_present():
    truths = {"saturating K=15": ("saturating", lambda r: fd.satur(r, 40.0, 15.0))}
    m = fd.confusion_matrix(points=9, decades=2.0, noise=0.02, reps=12, truths=truths, seed=7)
    d = fd.rival_detection_rates(m)
    assert "saturating K=15" in d and d["saturating K=15"] >= 0.8
