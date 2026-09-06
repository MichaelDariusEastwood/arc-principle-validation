"""The operator's rulings of 5 September 2026 (10 and 17), pinned so that the engine and the frozen text
cannot drift apart silently."""
from arc_instruments import precision as pr
from arc_instruments import regions as RG
from arc_instruments import verdicts as V


def test_ruling_10_fixed_numbers_are_the_defaults():
    assert V.p11_pair((-0.05, 0.05)) == V.IDENTICAL_WITHIN_MARGIN          # 0.10 band by default
    assert V.p11_pair((0.11, 0.30)) == V.DISTINCT
    assert V.p5_agreement((-0.09, 0.09)) == V.SUPPORTED                    # 0.10 band
    assert V.p17_pairwise((-0.19, 0.19)) == V.INDEPENDENT                  # 0.20 on the log ratio
    assert V.p3_frontier((-0.09, 0.09), True, True, True, False) == V.SUPPORTED   # tolerance 0.10
    assert pr.FALSE_AFFIRMATIVE_MAX == 0.05 and pr.DETECTION_PROBABILITY_MIN == 0.80


def test_ruling_17_strict_clearance_everywhere():
    assert all(a["boundary"] == "open" for a in RG.REGIONS)
    for fn, contact in ((V.p8_departure, (0.40, 0.55)), (V.p22_typed_ordering, (-0.10, 0.05)), (V.p19_zero_scaling, (-0.10, 0.04)),
                        (V.p10_material_fraction, (0.10, 0.20)), (V.p15_decay, (0.30, 0.50)), (V.p14_blinded_shrink, (-0.20, 0.00))):
        assert fn(contact) == V.INCONCLUSIVE, fn.__name__
    for fn in (V.p8_departure, V.p22_typed_ordering, V.p19_zero_scaling, V.p10_material_fraction, V.p15_decay, V.p14_blinded_shrink):
        assert fn((0.5, 0.5)) == V.NOT_EVALUABLE, fn.__name__
