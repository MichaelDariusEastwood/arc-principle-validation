"""The spending controller: it must bound concurrent dispatch and a retry storm, and it must never be
able to change a scientific rule."""
import threading

import pytest

from arc_runner import budget as B


def test_a_reservation_is_held_while_in_flight_so_concurrency_cannot_spend_it_twice():
    c = B.BudgetController(B.Allowance(limit_gbp=10.0))
    t = c.reserve("call-a", 6.0)
    assert c.held_gbp == 6.0 and c.available_gbp == 4.0
    with pytest.raises(B.BudgetExhausted):
        c.reserve("call-b", 6.0)
    c.settle(t, 1.0)                      # the actual charge is far below the conservative maximum
    assert c.committed_gbp == 1.0 and c.held_gbp == 0.0


def test_a_retry_storm_exhausts_the_allowance_and_not_the_account():
    c = B.BudgetController(B.Allowance(limit_gbp=5.0, max_retries_per_operation=2))
    spent, refused = 0.0, 0
    for _ in range(50):
        if not c.retry_allowed("op"):
            refused += 1
            continue
        try:
            t = c.reserve("op", 2.0)
        except B.BudgetExhausted:
            refused += 1
            continue
        c.settle(t, 2.0)
        spent += 2.0
    assert spent <= 5.0 and refused > 0
    assert c.available_gbp >= 0.0


def test_concurrent_dispatch_never_exceeds_the_allowance():
    c = B.BudgetController(B.Allowance(limit_gbp=20.0))
    ok, blocked = [], []

    def worker(i):
        try:
            t = c.reserve("call-%d" % i, 3.0)
            c.settle(t, 3.0)
            ok.append(i)
        except B.BudgetExhausted:
            blocked.append(i)

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert c.committed_gbp <= 20.0
    assert len(ok) == 6 and len(blocked) == 14


def test_an_unapproved_model_is_refused_before_any_money_moves():
    c = B.BudgetController(B.Allowance(limit_gbp=100.0, approved_models=["budget-a", "budget-b"]))
    with pytest.raises(B.PremiumFallbackRefused):
        c.reserve("call", 1.0, model="frontier-top")
    assert c.committed_gbp == 0.0 and c.held_gbp == 0.0


def test_exhaustion_says_the_science_does_not_move():
    c = B.BudgetController(B.Allowance(limit_gbp=1.0))
    with pytest.raises(B.BudgetExhausted) as e:
        c.reserve("call", 2.0)
    msg = str(e.value).lower()
    assert "observations preserved" in msg
    assert "interrupted-run" in msg
    assert "margin do not move" in msg


def test_the_report_names_what_the_allowance_covers_and_refuses_to_claim_a_guarantee():
    c = B.BudgetController(B.pilot_allowance())
    r = c.report()
    assert r["allowance_gbp"] == 50.0
    assert r["includes"]["vendor_inference"] is True and r["includes"]["tax"] is False
    assert "not an absolute spending guarantee" in r["guarantee"]
    assert "unpaid_human_hours" in r


def test_the_controller_exposes_no_way_to_change_a_scientific_rule():
    names = [n for n in dir(B.BudgetController) if not n.startswith("_")]
    for banned in ("margin", "depth", "replicates", "verdict", "window"):
        assert not any(banned in n for n in names), (banned, names)


# --------------------------------------------------------------------------------------------------
# The ceiling is consulted and not only recorded (finding A9)
#
# The approved figure was required by the deciding gate, written into the manifest, and read by
# nothing: no run reserved against it, so it bounded the record and not the spending. These cases are
# about the object that closes that, and about the halt being the registered one.
# --------------------------------------------------------------------------------------------------

class _CountingSystem:
    """A system that answers every call. It costs nothing, which is the point: what stops the third
    call here is the allowance and not the provider."""
    name = "counting"

    def __init__(self):
        self.calls = 0

    def revise(self, artefact, retained, task, rng):
        self.calls += 1
        return dict(artefact, rounds=int(artefact.get("rounds", 0)) + 1)

    def metadata(self):
        return {"adapter": self.name, "calls": self.calls}


def test_a_metered_adapter_reserves_before_each_call_and_halts_rather_than_overspending():
    inner = _CountingSystem()
    c = B.BudgetController(B.Allowance(limit_gbp=1.0))
    ad = B.MeteredAdapter(inner, c, estimated_max_call_gbp=0.4)
    art = {"rounds": 0}
    art = ad.revise(art, {"fraction": 1.0}, "task", None)
    art = ad.revise(art, {"fraction": 1.0}, "task", None)
    assert inner.calls == 2 and c.committed_gbp == pytest.approx(0.8)
    with pytest.raises(B.BudgetExhausted):
        ad.revise(art, {"fraction": 1.0}, "task", None)
    assert inner.calls == 2, "the third call was never dispatched, which is what a ceiling means"
    assert "observations preserved" in c.halted_reason


def test_a_metered_adapter_returns_its_inner_artefact_untouched_and_reports_the_ledger():
    inner = _CountingSystem()
    ad = B.MeteredAdapter(inner, B.BudgetController(B.Allowance(limit_gbp=10.0)), 0.5)
    out = ad.revise({"rounds": 3, "text": "x"}, {"fraction": 0.5}, "task", None)
    assert out == {"rounds": 4, "text": "x"}
    md = ad.metadata()
    assert md["adapter"] == "counting" and md["calls"] == 1
    assert md["budget"]["committed_gbp"] == pytest.approx(0.5)
    assert md["budget"]["estimated_max_call_gbp"] == 0.5
    assert "not an absolute spending guarantee" in md["budget"]["guarantee"]


def test_a_call_that_raised_returns_its_hold_rather_than_stopping_the_run_twice():
    class Fails:
        name = "fails"

        def revise(self, *a, **k):
            raise RuntimeError("the endpoint refused")

    c = B.BudgetController(B.Allowance(limit_gbp=1.0))
    ad = B.MeteredAdapter(Fails(), c, 0.4)
    with pytest.raises(RuntimeError):
        ad.revise({}, {}, "task", None)
    assert c.held_gbp == 0.0 and c.committed_gbp == 0.0


def test_a_meter_that_reserves_nothing_is_refused_at_construction():
    c = B.BudgetController(B.Allowance(limit_gbp=1.0))
    for figure in (0.0, -1.0):
        with pytest.raises(ValueError):
            B.MeteredAdapter(_CountingSystem(), c, figure)
