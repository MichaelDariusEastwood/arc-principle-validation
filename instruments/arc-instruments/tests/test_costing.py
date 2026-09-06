from arc_instruments import costing as co


def test_call_counts_follow_the_designs():
    c = co.call_counts(co.CHEAP)
    assert c["P5 calibration (two routes, negative controls)"]["generation"] == int(3 * 27 * 3 * 1.8)
    assert c["P5 held-out and P1 titration (depth and breadth arms)"]["generation"] == 3 * 2 * 32 * 2
    assert c["Stage A scorer channel (label experiment)"]["judge"] == 300 * 3 * 3 * 2


def test_intensive_costs_more_than_cheap_and_evaluation_dominates():
    tc, ti = co.token_totals(co.CHEAP), co.token_totals(co.INTENSIVE)
    assert ti["total_in"] > 10 * tc["total_in"]
    ev = ti["stages"]["P5 calibration (two routes, negative controls)"]
    assert ev["tokens_in"] > 0.4 * ti["total_in"]              # the capability battery is the cost lever
    assert co.cost(co.CHEAP)["budget_api"] < 100
    assert co.cost(co.INTENSIVE)["frontier_top"] > co.cost(co.INTENSIVE)["budget_api"]


def test_report_renders():
    r = co.report(co.CHEAP, "cheap")
    assert "Total" in r and "mixed portfolio" in r


def test_flagship_profiles_fit_the_small_budget_and_the_battery_reading_does_not():
    assert co.gbp(co.lineage_mix_cost(co.P5_FLAGSHIP_LADDER)) < 300
    assert co.gbp(co.lineage_mix_cost(co.P16_FORECAST_SECONDARY)) < 100
    assert co.gbp(co.lineage_mix_cost(co.P5_FLAGSHIP_BATTERY_625)) > 1000     # two per cent capability precision by battery
    assert co.cost(co.CHEAP)["budget_api"] / co.USD_PER_GBP < 50                # the pilot on the budget tier alone
    assert "Pounds" in co.flagship_menu()


def test_runner_faithful_p5_rounds_count_one_bank_per_lineage():
    from arc_instruments import costing as C
    r = C.runner_p5_rounds(states=5, fractions=5, reps=16, control_fraction=0.2, lineages=3, heldout_per_lineage=1, replicates=4, window=128, cal_depth=4)
    assert r["bank"] == 5 * 5 * 16 * 3
    assert r["controls"] == 5 * 5 * 3 * 3            # round(16 * 0.2) = 3 control cells per (state, fraction)
    assert r["calibration"] == 3 * 4
    assert r["sealed_window"] == 3 * 4 * 124
    assert r["total"] == r["bank"] + r["controls"] + r["calibration"] + r["sealed_window"]


def test_runner_faithful_p16_rounds_and_costs_are_consistent():
    from arc_instruments import costing as C
    r = C.runner_p16_rounds()
    assert r["arm_systems"] == 21 and r["total"] == 27 * 96
    usd = C.rounds_cost_usd(r["total"], "budget_api")
    assert abs(usd - 27 * 96 * (6000 * 0.5 + 3000 * 2.0) / 1e6) < 1e-9
    assert C.rounds_cost_mix_gbp(r["total"]) > 0


def test_the_p16_manifest_counts_controls_as_arms_and_matches_its_protocol():
    from arc_instruments import design_manifest as DM
    m = DM.p16_manifest()
    # five dose arms plus a sham plus a baseline is SEVEN arms, not six: the controls are arms.
    assert m["arm_count"] == 7 and m["arm_systems"] == 21
    assert [a["arm"] for a in m["arms"]][-2:] == ["sham", "baseline"]
    # the numbers the execution protocol states in prose
    DM.assert_matches(m, arm_count=7, arm_systems=21, locating_systems=6, horizon=96,
                      generation_rounds=2592, gbp_three_lineage_mix=49.0)


def test_the_p5_manifest_matches_its_protocol_and_a_mismatch_is_named():
    from arc_instruments import design_manifest as DM
    import pytest as _pt
    m = DM.p5_manifest()
    DM.assert_matches(m, states=5, fractions=5, reps_per_cell=16, heldout_systems=3, replicates=4,
                      window=128, generation_rounds=2925, gbp_three_lineage_mix=55.0)
    with _pt.raises(AssertionError) as e:
        DM.assert_matches(m, arm_count=6)
    assert "manifest says" in str(e.value)


def test_the_manifest_hash_moves_with_the_design_and_not_with_the_prices():
    from arc_instruments import design_manifest as DM
    a = DM.p16_manifest()
    b = DM.p16_manifest(systems_per_arm=6)
    assert a["sha256"] != b["sha256"], "a changed design must change the hash"
    assert "prices_are_placeholders" in a["cost"]
    assert "cost" not in DM.render(a).split("generation rounds")[0]
