import numpy as np

from arc_instruments import blinding as bl


def test_james_index_extremes():
    actual = ["A", "B"] * 50
    assert bl.james_index(actual, [bl.DONT_KNOW] * 100) == 1.0
    assert bl.james_index(actual, actual) == 0.0
    opposite = ["B" if a == "A" else "A" for a in actual]
    assert bl.james_index(actual, opposite) == 1.0
    rng = np.random.default_rng(0)
    guess = [rng.choice(["A", "B"]) for _ in actual]
    assert 0.35 < bl.james_index(actual, guess) < 0.65


def test_bang_index_extremes():
    actual = ["A"] * 40 + ["B"] * 40
    assert bl.bang_index(actual, actual, "A") == 1.0
    assert bl.bang_index(actual, [bl.DONT_KNOW] * 80, "A") == 0.0
    opposite = ["B" if a == "A" else "A" for a in actual]
    assert bl.bang_index(actual, opposite, "B") == -1.0


def test_bootstrap_interval_contains_point():
    actual = ["A", "B"] * 30
    guess = actual[:40] + [bl.DONT_KNOW] * 20
    point = bl.james_index(actual, guess)
    lo, hi = bl.bootstrap_interval(bl.james_index, actual, guess, n_boot=300)
    assert lo <= point <= hi


def test_label_effect_recovers_injected_bias():
    rows = bl.simulate_label_experiment(120, ["fam1", "fam2"], {"fam1": 0.8, "fam2": 0.0}, score_sd=0.5, seed=2)
    eff = bl.label_effect(rows, n_boot=300)
    assert 0.5 < eff["fam1"]["own"]["estimate"] < 1.1
    lo, hi = eff["fam1"]["own"]["interval"]
    assert lo > 0.3
    lo2, hi2 = eff["fam2"]["own"]["interval"]
    assert lo2 < 0.0 < hi2


def _texts(rng, family, n):
    vocab = {"fam1": ["therefore", "consequently", "we observe", "the lemma", "it follows"],
             "fam2": ["so basically", "kind of", "you know", "pretty much", "at the end of the day"]}
    common = ["the", "system", "improves", "with", "depth", "and", "correction", "keeps", "pace", "under", "load"]
    out = []
    for _ in range(n):
        words = [rng.choice(common) for _ in range(30)] + [rng.choice(vocab[family]) for _ in range(6)]
        rng.shuffle(words)
        out.append(" ".join(words))
    return out


def test_provenance_challenge_detects_style_and_positive_control():
    rng = np.random.default_rng(11)
    tr = _texts(rng, "fam1", 60) + _texts(rng, "fam2", 60); trl = ["fam1"] * 60 + ["fam2"] * 60
    te = _texts(rng, "fam1", 40) + _texts(rng, "fam2", 40); tel = ["fam1"] * 40 + ["fam2"] * 40
    res = bl.provenance_challenge(tr, trl, te, tel, positive_control_marker="LEAKTAG-")
    assert res["accuracy"] > 0.8 and res["leak_material"] is True
    assert res["positive_control_detected"] is True


def test_provenance_challenge_near_chance_when_families_identical():
    rng = np.random.default_rng(12)
    tr = _texts(rng, "fam1", 60) + _texts(rng, "fam1", 60); trl = ["x"] * 60 + ["y"] * 60
    te = _texts(rng, "fam1", 40) + _texts(rng, "fam1", 40); tel = ["x"] * 40 + ["y"] * 40
    res = bl.provenance_challenge(tr, trl, te, tel, positive_control_marker="LEAKTAG-")
    assert res["leak_material"] is False
    assert res["positive_control_detected"] is True
