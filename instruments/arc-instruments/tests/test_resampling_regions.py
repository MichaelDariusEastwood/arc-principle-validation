import importlib.util
import json
import os

import numpy as np
import pytest

from arc_instruments import regions as RG
from arc_instruments import resampling as rs
from arc_instruments import verdicts as V


def test_cluster_bootstrap_is_wider_than_row_bootstrap_under_clustering():
    rng = np.random.default_rng(5)
    n_clusters, per = 20, 10
    effects = rng.normal(0.0, 1.0, n_clusters)
    values = np.concatenate([effects[c] + rng.normal(0.0, 0.3, per) for c in range(n_clusters)])
    ids = np.repeat(np.arange(n_clusters), per)
    clo, chi = rs.cluster_bootstrap(np.mean, values, ids, n_boot=400, seed=1)
    rlo, rhi = rs.row_bootstrap(np.mean, values, n_boot=400, seed=1)
    assert (chi - clo) > 1.5 * (rhi - rlo)
    plo, phi = rs.paired_history_bootstrap(np.mean, values, ids, n_boot=400, seed=1)
    assert abs((phi - plo) - (chi - clo)) < 1e-12


def test_cluster_indices_cover_whole_clusters():
    rng = np.random.default_rng(2)
    ids = np.array([0, 0, 1, 1, 2, 2])
    idx = rs.cluster_bootstrap_indices(ids, rng)
    assert idx.size == 6
    for c in np.unique(ids[idx]):
        assert np.sum(ids[idx] == c) % 2 == 0


def test_regions_export_round_trips():
    text = RG.to_json()
    back = json.loads(text)
    assert [a["name"] for a in back] == [a["name"] for a in RG.REGIONS]
    assert RG.axis("P8.exponent")["boundary"] == "open"


class _TableReader:
    """A minimal independent reader of the exported table, with the semantics the retired coordination
    harness used (5 September 2026): an interval receives a region when it lies wholly inside it, "open"
    boundaries make a contact fall outside, overlapping regions are a table defect, and "requires" flags
    gate a verdict. It exists so that the table and the engine are checked against each other by a
    second implementation, not by the engine reading its own export."""

    INF = float("inf")

    @staticmethod
    def _inside(lo, hi, rlo, rhi, closed):
        rlo = -_TableReader.INF if rlo is None else float(rlo)
        rhi = _TableReader.INF if rhi is None else float(rhi)
        if closed:
            return lo >= rlo and hi <= rhi
        left = lo > rlo or rlo == -_TableReader.INF
        right = hi < rhi or rhi == _TableReader.INF
        return left and right

    def evaluate(self, axis, lo, hi, flags=None):
        flags = set(flags or [])
        closed = axis.get("boundary", "closed") == "closed"
        containing = [n for n, (a, b) in axis["regions"].items() if self._inside(lo, hi, a, b, closed)]
        if len(containing) > 1:
            raise ValueError("table defect on %s: %r" % (axis["name"], containing))
        verdict = containing[0] if containing else V.INCONCLUSIVE
        need = set(axis.get("requires", {}).get(verdict, []))
        if need - flags:
            verdict = V.INCONCLUSIVE
        return {"verdict": verdict, "proposition_verdict": axis.get("aggregate", {}).get(verdict, verdict)}

    def check_table(self, axes, samples=2001):
        problems = []
        for axis in axes:
            finite = [float(b) for r in axis["regions"].values() for b in r if b is not None]
            if not finite:
                continue
            lo, hi = min(finite) - 1.0, max(finite) + 1.0
            closed = axis.get("boundary", "closed") == "closed"
            for i in range(samples):
                x = lo + i * (hi - lo) / (samples - 1)
                hits = [n for n, (a, b) in axis["regions"].items() if self._inside(x, x, a, b, closed)]
                if len(hits) > 1:
                    problems.append("%s: point %.4f lies in %r (overlap)" % (axis["name"], x, hits))
                    break
        return problems


def _load_harness():
    return _TableReader()


ENGINE = {
    "P8.exponent": lambda iv: V.p8_departure(iv),
    "P8.unity": lambda iv: V.p8_clears_unity(iv),
    "P22.checkable_minus_judged": lambda iv: V.p22_typed_ordering(iv),
    "P19.alignment_elasticity": lambda iv: V.p19_zero_scaling(iv),
    "P19.legacy_half": lambda iv: V.p19_keep_pace(iv),
    "P10.fraction_not_surviving": lambda iv: V.p10_material_fraction(iv),
    "P15.fraction_retained": lambda iv: V.p15_decay(iv),
    "P17.pair_log_ratio": lambda iv: V.p17_pairwise(iv),
    "P14.blinded_minus_unblinded": lambda iv: V.p14_blinded_shrink(iv),
    "P6.gamma_same_class": lambda iv: V.p6_survey([iv], replicated=True),
}
NORMALISE = {V.INSUFFICIENT_PRECISION: V.INCONCLUSIVE}


def test_engine_and_harness_agree_on_one_table():
    harness = _load_harness()
    if harness is None:
        pytest.skip("the coordination repository's harness is not on this tree; the estate copies skip this cross-check")
    problems = harness.check_table(RG.REGIONS)
    assert not any("overlap" in p for p in problems), problems
    rng = np.random.default_rng(20260905)
    checked = 0
    for name, fn in ENGINE.items():
        axis = RG.axis(name)
        for _ in range(300):
            a, b = sorted(rng.uniform(-1.0, 2.0, 2))
            if b - a < 1e-9:
                continue
            got_h = harness.evaluate(axis, a, b, ["replicated"])["verdict"]
            got_e = NORMALISE.get(fn((a, b)), fn((a, b)))
            assert got_h == got_e, (name, a, b, got_h, got_e)
            checked += 1
    # the audit's counterexamples through both readers
    for name, iv, expect in [("P8.exponent", (0.35, 0.45), V.INCONCLUSIVE), ("P8.exponent", (0.48, 0.52), V.EQUIVALENT_TO_NULL),
                             ("P22.checkable_minus_judged", (-0.15, -0.05), V.INCONCLUSIVE), ("P8.exponent", (0.20, 0.40), V.INCONCLUSIVE)]:
        assert harness.evaluate(RG.axis(name), iv[0], iv[1], ["replicated"])["verdict"] == expect
        assert ENGINE[name](iv) == expect
    assert checked > 2000


def test_strict_clearance_and_points():
    assert V.p8_departure((0.40, 0.55)) == V.INCONCLUSIVE
    assert V.p8_departure((0.61, 0.90)) == V.ABOVE_NULL
    assert V.p8_departure((0.60, 0.60)) == V.NOT_EVALUABLE
    assert V.p22_typed_ordering((-0.30, -0.10)) == V.INCONCLUSIVE
    assert V.p22_typed_ordering((-0.30, -0.12)) == V.JUDGED_HIGHER
    assert V.p12_build_order((-0.30, -0.12), True) == V.SUPPORTED
    assert V.p12_build_order((0.12, 0.30), True) == V.SUPPORTED
    assert V.p12_build_order((-0.05, 0.05), True) == V.REFUTED
    assert V.p14_blinded_shrink((0.0, 0.2)) == V.INCONCLUSIVE
    assert V.p14_blinded_shrink((0.01, 0.2)) == V.REFUTED
    assert V.p6_survey([(0.45, 0.55)]) == V.INCONCLUSIVE
    assert V.p6_survey([(0.30, 0.49)]) == V.NO_COUNTEREXAMPLE
    assert V.p6_survey([(0.52, 0.60)], replicated=False) == V.AWAITING_REPLICATION
    assert V.region_verdict((0.45, 0.60), 0.5, 0.10, "A", "B", "E", closed=True) == "E"
    assert V.region_verdict((0.45, 0.60), 0.5, 0.10, "A", "B", "E") == V.INCONCLUSIVE


def test_relative_keep_pace_and_eclipse_ratio():
    assert V.p19_keep_pace_relative((0.15, 0.25)) == V.OUT_SCALES_DRIFT      # 0.3 against drift 0.1
    assert V.p19_keep_pace_relative((-0.25, -0.15)) == V.LAGS_DRIFT          # 0.6 against drift 0.8
    assert V.p19_keep_pace_relative((-0.05, 0.05)) == V.INCONCLUSIVE
    v, iv = V.eclipse_ratio((0.40, 0.60), (-0.05, 0.30))
    assert v == V.NOT_EVALUABLE and iv is None
    v, iv = V.eclipse_ratio((0.45, 0.55), (0.48, 0.52))
    assert v == V.INCONCLUSIVE and iv[0] < 0.90 and iv[1] > 1.10   # two decent exponent intervals give an inconclusive ratio
    v, iv = V.eclipse_ratio((0.49, 0.51), (0.49, 0.51))
    assert v == V.NEAR_ONE and iv[0] > 0.95 and iv[1] < 1.05
    v, iv = V.eclipse_ratio((0.70, 0.80), (0.30, 0.40))
    assert v == V.ABOVE_ONE


def test_p20_power_gate_replaces_geometry():
    adv = {"reciprocal": (0.15, 0.40), "constant": (0.20, 0.50)}
    assert V.p20_form([0.2, 0.3, 0.4], adv, margin=0.10) == V.NOT_EVALUABLE               # no calibrated power: no geometric substitute
    assert V.p20_form([0.2, 0.3, 0.4], adv, margin=0.10, discriminating_power=0.95) == V.SUPPORTED
    assert V.p20_form([0.2, 0.3, 0.4], adv, margin=0.10, discriminating_power=0.50) == V.NOT_DISCRIMINATING
    nested_tie = {"reciprocal": (0.15, 0.40), "burden_intensity": (-0.05, 0.05)}
    assert V.p20_form([0.2, 0.4], nested_tie, margin=0.10, discriminating_power=0.9, nested={"burden_intensity"}) == V.SUPPORTED
    assert V.p20_form([0.2, 0.4], nested_tie, margin=0.10, discriminating_power=0.9) == V.INCONCLUSIVE
    nested_wins = {"reciprocal": (0.15, 0.40), "burden_intensity": (-0.40, -0.15)}
    assert V.p20_form([0.2, 0.4], nested_wins, margin=0.10, discriminating_power=0.9, nested={"burden_intensity"}) == V.REFUTED
