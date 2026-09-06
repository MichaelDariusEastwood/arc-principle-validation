"""Finding A6: the two regression directions are a consistency check, and identification is separate.

Each acceptance case the finding names is one test here, and each is stated as a world rather than as
an assertion about the code: a world in which the retention exponent differs from the capability
exponent, a world in which a nuisance rate scaling with the available capability makes the two
directions agree while neither is measuring the coupling, a world measured too imprecisely to decide
either way, and a world in which the domain genuinely intervened on capability a second time. The
finding's closing sentence is the standard these are written to: recovery of the programmed identity
between the retention fraction and the capability state is insufficient, because that identity is
what the reference mock built in and what the reference label was reading back.

THE SECOND HALF OF THIS FILE IS THE REVIEW OF THE FIRST REPAIR, and each test there is one defect
that repair left behind. The identification equivalence read a CLOSED boundary while the registered
convention reads a strict one, so exact contact was agreement here and not agreement there. The
judgement read admissibility off a key in the record instead of re-deriving it, so a manipulation
with no documentation at all reached IDENTIFIED by carrying an empty list, and a saved bundle
re-scored to its own label rather than to its evidence. A manipulation could place its cells with the
BANK'S OWN loader and be counted as a second channel. The set of manipulations sat outside the sealed
specification and outside the deciding gate, though it had become the only route to IDENTIFIED. And
the sealed prediction's estimator was swapped on the disagreement branch, which this finding did not
ask for. Every one of those is a world here too.
"""
import numpy as np
import pytest

from arc_runner import (adapters, code_domain as CD, custody as CUSTODY, ladder as L, manifest as M,
                        mode as MODE, p5, p5_identification as PI, p5_observation as PO,
                        p5_prediction as PRED)


def _place(state):
    """The BANK's placement channel: capability placed by loading a checkpoint at that state."""
    return {"kind": "mock", "capability": float(state), "rounds": 0}


def _distil(state):
    """A SECOND placement channel, and a different callable from the bank's.

    The two are different objects and the artefacts they return carry different provenance. That
    difference is what makes a manipulation built on this one a second channel at all: the first
    repair of this finding built every 'independent' manipulation in this file out of `_place`, so
    the only thing distinguishing the second channel from the first was a pair of declared booleans,
    and the easier case was being tested under the harder case's name.
    """
    return {"kind": "distilled", "capability": float(state), "rounds": 0}


def _scaffold(state):
    """A third placement channel, for the case that needs two second channels that are not each other."""
    return {"kind": "scaffolded", "capability": float(state), "rounds": 0}


def _ladder():
    return L.MockLadder(n_items=20000, scale=400.0)


def _bank(cfg, seed=1, **adapter_kw):
    ad = adapters.MockCouplingAdapter(noise=adapter_kw.pop("noise", 0.05), **adapter_kw)
    lad = _ladder()
    return p5.run_bank(ad, lad, cfg, np.random.default_rng(seed), _place), ad, lad


def _manipulation(adapter, name="distilled placement", **kw):
    """A documented second channel: capability placed without running a revision round.

    Every field the judgement requires is filled in, because a manipulation that omits one is a
    different test and has its own case below. The placement is `_distil` and never the bank's own
    `_place`, which is the point of the whole object.
    """
    fields = dict(
        channel="the capability state is placed by distilling a larger system into the artefact, "
                "which performs no revision round and therefore uses no retained material",
        exclusion_restriction="the distillation carries no rate factor that also scales the revision "
                              "round's increment, so an elasticity measured under it is the "
                              "elasticity of capability and not of the pathway that produced it",
        independent_of_retention=True, shares_nuisance_with_bank=False,
        place_at_state=_distil, adapter=adapter, states=(40.0, 90.0, 180.0))
    fields.update(kw)
    return PI.CapabilityManipulation(name=name, **fields)


def _measure(m, ad, lad, cfg, rng, bank_place_at_state=_place):
    """Measure a manipulation the way a run does, with the bank's own loader in hand for comparison."""
    return p5.measure_manipulation(m, ad, lad, cfg, rng, bank_place_at_state=bank_place_at_state)


# --------------------------------------------------------------------------------------------------
# Acceptance case one: theta different from beta
# --------------------------------------------------------------------------------------------------


def test_the_crossed_fit_recovers_the_capability_exponent_when_the_retention_exponent_differs():
    """The general process a * f ** theta * U ** beta, with theta = 0.9 and beta = 0.5.

    The single-exponent model puts one exponent on the available capability f * U, which asserts that
    a tenth of the retention and a tenth of the capability do the same thing to the increment. On
    this world that assertion is false and the fit returns roughly 0.60, which is neither exponent.
    The crossed fit returns both, and the capability elasticity is the one the registered growth law
    is about.
    """
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, beta=0.5, retention_exponent=0.9)
    rows = [r for r in bank["rows"] if not r.get("control")]
    model = PO.model_for(lad)

    single = PO.fit_paired(rows, model)
    crossed = PO.fit_paired(rows, model, crossed=True)

    assert crossed.usable and crossed.crossed
    assert crossed.beta == pytest.approx(0.5, abs=0.08), crossed.as_record()
    assert crossed.retention_elasticity == pytest.approx(0.9, abs=0.08), crossed.as_record()
    assert crossed.retention_excess == pytest.approx(0.4, abs=0.10)
    assert np.isfinite(crossed.retention_excess_se) and crossed.retention_excess_se > 0
    # the single-exponent fit is not wrong about its own model; it is answering a different question,
    # and the difference is what the crossed design exists to expose
    assert abs(single.beta - 0.5) > 0.05, single.as_record()
    assert abs(single.beta - crossed.beta) > 0.05


def test_a_differing_retention_exponent_is_reported_as_inconsistency_and_not_as_a_refutation():
    """The two directions carry different elasticities, which is what INCONSISTENT means here.

    The finding is explicit that a valid capability mechanism must not be rejected merely because
    retention is not its sole channel, so the judgement's own words are asserted: it says the design
    has not identified the elasticity, and it says that this is not a refutation of the mechanism.
    """
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, beta=0.5, retention_exponent=0.9)
    routes = p5.estimate_routes(bank, cfg)

    assert routes["route_agreement"] == PI.INCONSISTENT, routes["route_consistency"]
    assert routes["beta_state_route"] == pytest.approx(0.5, abs=0.10)
    assert routes["beta_retention_route"] == pytest.approx(0.9, abs=0.10)
    assert routes["beta_capability"] == pytest.approx(0.5, abs=0.08)
    assert routes["identification"] == PI.NOT_IDENTIFIED
    assert "not a refutation of the mechanism" in routes["identification_judgement"]["reason"]


def test_an_independent_manipulation_identifies_even_when_the_two_directions_disagree():
    """theta different from beta, and a documented second channel that agrees: IDENTIFIED.

    This is the case the reference label could not reach at all. Its rule made a route disagreement
    a non-identification whatever else was true, so a world in which the retention elasticity simply
    differs from the capability elasticity, and in which a second intervention confirms the
    capability elasticity, was reported exactly as a confounded world was.
    """
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, beta=0.5, retention_exponent=0.9)
    second = adapters.MockCouplingAdapter(beta=0.5, a=1.1, noise=0.05)
    measured = _measure(_manipulation(second), ad, lad, cfg, np.random.default_rng(9))
    routes = p5.estimate_routes(bank, cfg, manipulation_estimates=[measured])

    assert routes["route_agreement"] == PI.INCONSISTENT
    assert routes["identification"] == PI.IDENTIFIED, routes["identification_judgement"]
    assert routes["identification_judgement"]["manipulations"][0]["agreement"] == PI.CONSISTENT


# --------------------------------------------------------------------------------------------------
# Acceptance case two: two routes sharing the same omitted rate factor
# --------------------------------------------------------------------------------------------------


def test_a_shared_omitted_rate_factor_makes_the_routes_agree_without_identifying_anything():
    """A nuisance rate scaling with the available capability adds 0.2 to both elasticities.

    The two directions therefore agree exactly, the equivalence test says CONSISTENT, and the number
    they agree on is 0.7 while the coupling is 0.5. This is the world in which the reference label
    printed IDENTIFIED, and it is why numerical agreement between two slopes through one bank cannot
    be an identification: the agreement is produced by the nuisance, not despite it.
    """
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, seed=2, beta=0.5, available_rate_exponent=0.2)
    routes = p5.estimate_routes(bank, cfg)

    assert routes["route_agreement"] == PI.CONSISTENT, routes["route_consistency"]
    assert routes["beta_capability"] == pytest.approx(0.7, abs=0.08), routes["crossed_fit"]
    assert abs(routes["beta_capability"] - 0.5) > 0.1
    assert routes["negative_control"]["status"] == "NO MATERIAL COUPLING"
    assert routes["identification"] == PI.NOT_ESTABLISHED
    assert "no independent capability manipulation was supplied" in \
        routes["identification_judgement"]["reason"]


def test_an_independent_manipulation_catches_the_shared_rate_factor():
    """The second channel does not carry the nuisance, so it disagrees, and the run says so."""
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, seed=2, beta=0.5, available_rate_exponent=0.2)
    clean = adapters.MockCouplingAdapter(beta=0.5, a=1.1, noise=0.05)
    measured = _measure(_manipulation(clean), ad, lad, cfg, np.random.default_rng(9))
    routes = p5.estimate_routes(bank, cfg, manipulation_estimates=[measured])

    assert routes["route_agreement"] == PI.CONSISTENT
    assert routes["identification"] == PI.NOT_IDENTIFIED, routes["identification_judgement"]
    assert routes["identification_judgement"]["manipulations"][0]["agreement"] == PI.INCONSISTENT


# --------------------------------------------------------------------------------------------------
# Acceptance case three: broad intervals with similar point estimates
# --------------------------------------------------------------------------------------------------


def test_broad_intervals_with_similar_point_estimates_are_unresolved_and_not_consistent():
    """The point gap is small and the interval is wider than the margin, so nothing is decided.

    This is the arithmetic defect on its own. The reference rule compared the gap with the margin as
    a POINT, so it called this bank agreed; what the bank actually shows is a gap somewhere between
    minus 0.09 and plus 0.20, which is a statement about its own imprecision. The old rule's verdict
    is computed here beside the new one so that the difference is visible in the test rather than
    only in the finding.
    """
    cfg = p5.P5Config(reps=6)
    bank, ad, lad = _bank(cfg, seed=3, beta=0.5, noise=0.4)
    routes = p5.estimate_routes(bank, cfg)
    test = routes["route_consistency"]["test"]

    assert abs(test["point"]) < cfg.route_margin          # what the point comparison would have read
    assert test["half_width"] > cfg.route_margin          # what the interval says about that reading
    assert routes["route_agreement"] == PI.UNRESOLVED, routes["route_consistency"]
    assert routes["identification"] in (PI.NOT_ESTABLISHED, PI.NOT_IDENTIFIED)


def test_a_manipulation_whose_interval_spans_the_margin_does_not_identify():
    """The same arithmetic on the other equivalence: an imprecise second channel decides nothing.

    The record carries the manipulation's DESCRIPTION and not merely an empty failures list, because
    the judgement re-derives admissibility from the description now. A record without one is a
    different case and has its own test below.
    """
    m = _manipulation(None, name="second channel", place_at_state=None,
                      estimate={"beta": 0.52, "beta_se": 0.30,
                                "provenance": "measured on the distillation bank of 12 June, "
                                              "recorded in the domain's own log"})
    judgement = PI.judge(
        consistency={"label": PI.CONSISTENT, "capability_elasticity": 0.5, "retention_elasticity": 0.5},
        control={"status": "NO MATERIAL COUPLING"}, bank_elasticity=0.50, bank_elasticity_se=0.02,
        manipulations=[{"name": m.name, "manipulation": m.as_record(), "beta": 0.52, "beta_se": 0.30,
                        "documentation_failures": []}], margin=0.10)
    assert judgement["n_manipulations_admissible"] == 1
    assert judgement["manipulations"][0]["agreement"] == PI.UNRESOLVED
    assert judgement["label"] == PI.NOT_ESTABLISHED
    assert "too broad to decide" in judgement["reason"]


# --------------------------------------------------------------------------------------------------
# Acceptance case four: genuinely independent capability manipulations with matching effects
# --------------------------------------------------------------------------------------------------


def test_two_independent_capability_manipulations_with_matching_effects_identify():
    """The clean world, with two documented second channels that both agree.

    Only here does the word IDENTIFIED appear, and it rests on interventions rather than on
    directions: two further placements of the capability state, each by a channel that runs no
    retained material, each with its exclusion restriction written down, each measured on its own
    bank, and each agreeing with the bank's capability elasticity inside the registered margin as an
    interval.

    THE TWO CHANNELS ARE TWO CHANNELS. Each places with its own loader, and neither of them is the
    bank's. The first version of this case built both out of the bank's own `_place`, so it asserted
    that two declarations identify rather than that two placements do.
    """
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, beta=0.5)
    rng = np.random.default_rng(9)
    ms = [_manipulation(adapters.MockCouplingAdapter(beta=0.5, a=1.1, noise=0.05), name="distilled"),
          _manipulation(adapters.MockCouplingAdapter(beta=0.5, a=0.55, noise=0.05), name="scaffolded",
                        channel="the capability state is placed by attaching an external tool, which "
                                "performs no revision round",
                        place_at_state=_scaffold, states=(50.0, 110.0, 200.0))]
    measured = [_measure(m, ad, lad, cfg, rng) for m in ms]
    routes = p5.estimate_routes(bank, cfg, manipulation_estimates=measured)

    assert all(m["measured"] and not m["documentation_failures"] for m in measured)
    assert {m["manipulation"]["placement_is_the_banks_own_loader"] for m in measured} == {False}
    assert routes["identification"] == PI.IDENTIFIED, routes["identification_judgement"]
    assert [a["agreement"] for a in routes["identification_judgement"]["manipulations"]] == \
        [PI.CONSISTENT, PI.CONSISTENT]
    assert routes["identification_judgement"]["n_manipulations_admissible"] == 2


def test_one_disagreeing_manipulation_withholds_identification_from_the_pair():
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, beta=0.5)
    rng = np.random.default_rng(9)
    measured = [_measure(_manipulation(
        adapters.MockCouplingAdapter(beta=0.5, a=1.1, noise=0.05), name="agrees"), ad, lad, cfg, rng),
        _measure(_manipulation(
            adapters.MockCouplingAdapter(beta=0.9, a=0.4, noise=0.05), name="disagrees",
            place_at_state=_scaffold), ad, lad, cfg, rng)]
    routes = p5.estimate_routes(bank, cfg, manipulation_estimates=measured)
    assert routes["identification"] == PI.NOT_IDENTIFIED, routes["identification_judgement"]


# --------------------------------------------------------------------------------------------------
# The documentation the identification rests on, and the refusals
# --------------------------------------------------------------------------------------------------


def test_an_undocumented_manipulation_is_refused_before_any_call_is_made():
    """Every failure is named at once, and the refusal costs nothing.

    The order matters: a manipulation whose exclusion restriction is not written down is refused
    BEFORE its bank is run, so an inadmissible intervention is never something a run has already paid
    for. The adapter here counts its calls and must never be reached.
    """
    cfg = p5.P5Config(reps=4)
    lad = _ladder()
    counted = adapters.MockCouplingAdapter(beta=0.5)
    bad = PI.CapabilityManipulation(name="unwritten", place_at_state=_distil, adapter=counted)
    record = _measure(bad, counted, lad, cfg, np.random.default_rng(0))

    assert record["measured"] is False
    assert counted.calls == 0
    failures = " ".join(record["documentation_failures"])
    assert "no channel" in failures
    assert "no exclusion restriction" in failures
    assert "not declared independent of retention" in failures
    assert "undeclared" in failures


def test_a_manipulation_declaring_a_shared_nuisance_is_inadmissible():
    m = PI.CapabilityManipulation(
        name="same family", channel="a second checkpoint from the same recursive run",
        exclusion_restriction="none needed", independent_of_retention=True,
        shares_nuisance_with_bank=True,
        estimate={"beta": 0.5, "beta_se": 0.01, "provenance": "the same recursive run's own bank"})
    assert not PI.admissible(m)
    judgement = PI.judge(consistency={"label": PI.CONSISTENT}, control={"status": "NO MATERIAL COUPLING"},
                         bank_elasticity=0.5, bank_elasticity_se=0.01,
                         manipulations=[{"name": m.name, "manipulation": m.as_record(),
                                         "beta": 0.5, "beta_se": 0.01,
                                         "documentation_failures": PI.documentation_failures(m)}],
                         margin=0.10)
    assert judgement["label"] == PI.NOT_ESTABLISHED
    assert judgement["n_manipulations_admissible"] == 0
    assert "not admissible" in judgement["reasons"][0]


def test_a_manipulation_may_carry_a_measurement_made_elsewhere():
    cfg = p5.P5Config(reps=4)
    lad = _ladder()
    ad = adapters.MockCouplingAdapter(beta=0.5)
    m = _manipulation(None, name="measured elsewhere", place_at_state=None,
                      estimate={"beta": 0.51, "beta_se": 0.02,
                                "provenance": "the distillation ladder of 12 June 2026, run and "
                                              "recorded by the domain outside this runner"})
    record = _measure(m, ad, lad, cfg, np.random.default_rng(0))
    assert record["measured"] and record["source"] == "supplied with the manipulation"
    assert record["beta"] == pytest.approx(0.51) and ad.calls == 0
    assert not record["documentation_failures"]


def test_an_estimate_with_no_provenance_is_refused():
    """A number typed into a mapping and a number measured somewhere are the same object in a mapping.

    The estimate path exists for a domain that measured the second channel elsewhere, and it is the
    one path on which this runner sees no bank, no adapter and no ladder. A hand-typed beta with a
    hand-typed standard error therefore reached IDENTIFIED with zero calls made anywhere. The
    provenance does not make the number true; it makes the claim one a reader can go and check, which
    is what every other requirement in this object is for.
    """
    cfg = p5.P5Config(reps=4)
    lad = _ladder()
    ad = adapters.MockCouplingAdapter(beta=0.5)
    m = _manipulation(None, name="typed in", place_at_state=None,
                      estimate={"beta": 0.5, "beta_se": 0.001})
    record = _measure(m, ad, lad, cfg, np.random.default_rng(0))
    assert record["measured"] is False
    assert any("provenance" in f for f in record["documentation_failures"])

    judgement = PI.judge(consistency={"label": PI.CONSISTENT},
                         control={"status": "NO MATERIAL COUPLING"},
                         bank_elasticity=0.5, bank_elasticity_se=0.01,
                         manipulations=[record], margin=0.10)
    assert judgement["label"] == PI.NOT_ESTABLISHED


# --------------------------------------------------------------------------------------------------
# The boundary: the registered convention decides it, and this file does not decide it again
# --------------------------------------------------------------------------------------------------


def test_exact_boundary_contact_is_not_clearance_in_the_identification_equivalence():
    """An interval whose end lands exactly on the margin has not cleared it, here as everywhere else.

    The first repair of this finding wrote the comparison out again in this module as
    `abs(point) + half <= margin`, which is the CLOSED boundary the registered convention exists to
    replace. On these numbers the two rules disagreed: the registered comparison returned unresolved
    with the words "exact contact is not clearance" and the identification equivalence returned
    CONSISTENT, and the judgement built on it returned IDENTIFIED. The divergence ran one way only,
    since the INCONSISTENT branch was strict already.
    """
    point, se, margin = 0.125, 0.125, 0.25          # exactly representable, so the contact is exact
    reading = PI.equivalence(point, se, margin, z=1.0)
    registered = PRED.REGISTERED.clearance(point, se * 1.0, margin)

    assert abs(point) + se <= margin                # what the closed rule this replaces would read
    assert reading["label"] == PI.UNRESOLVED, reading
    assert registered["verdict"] == PRED.UNRESOLVED
    assert "exact contact is not clearance" in reading["reason"]


def test_the_identification_equivalence_and_the_registered_comparison_are_one_rule():
    """The same numbers get the same answer from both, at the boundary and either side of it.

    This is the property the two files could not both hold while each declared its own arithmetic,
    and it is asserted as a property rather than as a pair of expected labels, because the failure it
    prevents is drift and drift is only visible across a range.
    """
    conv = PRED.REGISTERED
    # the translation is written out here rather than imported, so that a mapping changed in the
    # module would fail this test rather than be asserted against itself
    expected = {PRED.AGREES: PI.CONSISTENT, PRED.DISAGREES: PI.INCONSISTENT,
                PRED.UNRESOLVED: PI.UNRESOLVED}
    for point in (-0.30, -0.10, -0.05, 0.0, 0.05, 0.10, 0.30):
        for se in (0.0, 0.01, 0.05, 0.10, 0.25):
            ours = PI.equivalence(point, se, 0.10, z=1.0)
            theirs = conv.clearance(point, se, 0.10)
            assert ours["label"] == expected[theirs["verdict"]], (point, se, ours, theirs)
            assert ours["level"] == conv.level


def test_a_run_may_override_the_boundary_and_both_equivalences_follow_it():
    """The convention is one object, so a run that relaxes the boundary relaxes it in one place.

    A closed boundary is not the registered choice and the author may nevertheless register one. The
    test is that doing so reaches this file through the convention rather than through a second
    constant nobody remembers to change.
    """
    closed = PRED.IntervalConvention(strict_clearance=False)
    assert PI.equivalence(0.125, 0.125, 0.25, z=1.0, convention=closed)["label"] == PI.CONSISTENT
    assert PI.equivalence(0.125, 0.125, 0.25, z=1.0)["label"] == PI.UNRESOLVED
    # and the level and multiplier are the registered object's, not a second copy of the number
    assert PI.TOST_Z == PRED.REGISTERED.equivalence_z
    assert p5.P5Config().route_equivalence_z == PRED.REGISTERED.equivalence_z


# --------------------------------------------------------------------------------------------------
# Admissibility is re-derived, never read off the record
# --------------------------------------------------------------------------------------------------


def test_a_record_with_no_description_is_not_admissible_however_it_labels_itself():
    """The probe that broke the first repair: a name, a beta, a standard error, and nothing else.

    No channel, no exclusion restriction, no independence declaration, no nuisance declaration, and
    the judgement returned IDENTIFIED with an empty reasons list. The record's own
    `documentation_failures` key was the only thing consulted, and the record is written by whoever
    hands it over.
    """
    judgement = PI.judge(consistency={"label": PI.CONSISTENT},
                         control={"status": "NO MATERIAL COUPLING"},
                         bank_elasticity=0.5, bank_elasticity_se=0.01,
                         manipulations=[{"name": "no documentation", "beta": 0.5, "beta_se": 0.01}],
                         margin=0.10)
    assert judgement["label"] == PI.NOT_ESTABLISHED, judgement
    assert judgement["n_manipulations_admissible"] == 0
    assert judgement["reasons"] and "no manipulation description" in judgement["reasons"][0]


def test_an_empty_failures_key_does_not_make_an_undocumented_record_admissible():
    """The same record, now asserting its own innocence. The description is still empty."""
    judgement = PI.judge(consistency={"label": PI.CONSISTENT},
                         control={"status": "NO MATERIAL COUPLING"},
                         bank_elasticity=0.5, bank_elasticity_se=0.01,
                         manipulations=[{"name": "asserts itself", "beta": 0.5, "beta_se": 0.01,
                                         "documentation_failures": [],
                                         "manipulation": {"name": "asserts itself"}}],
                         margin=0.10)
    assert judgement["label"] == PI.NOT_ESTABLISHED, judgement
    failures = " ".join(judgement["reasons"])
    assert "no channel" in failures and "no exclusion restriction" in failures


def test_a_bundle_whose_manipulation_record_was_stripped_does_not_rescore_to_identified(tmp_path):
    """The bundle path of the same defect. A saved label is not evidence for itself.

    `reanalyse_bank` feeds the saved manipulation records back through the judgement, which is right,
    and the judgement used to believe their `documentation_failures` key, which is not. A bundle
    whose manipulation block has been removed and whose failures list has been emptied re-scores here
    to NOT ESTABLISHED, because there is nothing left in it to check.
    """
    cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
    ad = adapters.MockCouplingAdapter(beta=0.5, noise=0.03)
    lad = _ladder()
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    out = tmp_path / "bundle"
    res = p5.run_p5(ad, lad, cfg, 7, _place, start, ["S1", "S2"], bundle=str(out),
                    manipulations=[_manipulation(adapters.MockCouplingAdapter(beta=0.5, a=1.1,
                                                                             noise=0.03))])
    assert res["routes"]["identification"] == PI.IDENTIFIED, res["routes"]["identification_judgement"]

    bundle = CUSTODY.load_bundle(str(out))
    for rec in bundle["capability_manipulations"]:
        rec.pop("manipulation", None)
        rec["documentation_failures"] = []
    again = CUSTODY.reanalyse_bank(bundle)
    assert again["identification"] == PI.NOT_ESTABLISHED, again["identification_judgement"]


# --------------------------------------------------------------------------------------------------
# Structural independence: the one check that needs no domain knowledge
# --------------------------------------------------------------------------------------------------


def test_a_manipulation_placing_with_the_banks_own_loader_is_refused():
    """A second channel that is the first channel. Declared independent, and not independent.

    The manipulation here differs from the bank in nothing at all: the same placement callable, and
    no adapter of its own, so `measure_manipulation` falls back to the bank's. It declares itself
    independent of retention and free of the bank's nuisance, and those two booleans were the whole
    of what made it a second channel.
    """
    cfg = p5.P5Config(reps=4)
    lad = _ladder()
    counted = adapters.MockCouplingAdapter(beta=0.5)
    m = _manipulation(None, name="the bank again", place_at_state=_place)
    record = _measure(m, counted, lad, cfg, np.random.default_rng(0))

    assert record["measured"] is False and counted.calls == 0
    assert record["manipulation"]["placement_is_the_banks_own_loader"] is True
    assert any("bank's own loader" in f for f in record["documentation_failures"])


def test_a_placement_that_was_never_compared_with_the_banks_fails_closed():
    """A caller that does not hand over the bank's loader has not had the question answered.

    The record says the comparison was not performed and the judgement reads that as a failure, for
    the same reason an undeclared nuisance pathway is one: the identification rests on the answer,
    and nobody asked.
    """
    cfg = p5.P5Config(reps=4)
    lad = _ladder()
    ad = adapters.MockCouplingAdapter(beta=0.5)
    record = p5.measure_manipulation(_manipulation(ad), ad, lad, cfg, np.random.default_rng(0))

    assert record["measured"] is False
    assert record["manipulation"]["placement_checked_against_the_bank"] is False
    assert any("was not checked" in f for f in record["documentation_failures"])


def test_a_run_whose_second_channel_is_the_banks_own_loader_does_not_identify():
    """The same object through the whole runner, because that is where it reached IDENTIFIED."""
    cfg = p5.P5Config(reps=6, window_end=32, checkpoints=(4, 8, 16, 32))
    ad = adapters.MockCouplingAdapter(beta=0.5, noise=0.03)
    lad = _ladder()
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    res = p5.run_p5(ad, lad, cfg, 7, _place, start, ["S1", "S2"],
                    manipulations=[_manipulation(None, name="the bank again", place_at_state=_place)])
    judgement = res["routes"]["identification_judgement"]
    assert judgement["label"] != PI.IDENTIFIED, judgement
    assert any("bank's own loader" in r for r in judgement["reasons"]), judgement["reasons"]


def test_a_manipulation_reusing_the_banks_adapter_is_recorded_and_not_refused():
    """The open decision, asserted as an open decision.

    The adapter is the system being measured, so a second placement of the same system is what the
    design asks for and reusing it is not refused. It is recorded, so that an author who reads it
    otherwise can find it, and the judgement prints the decision beside its own label.
    """
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, beta=0.5)
    record = _measure(_manipulation(None, name="same system, second channel"), ad, lad, cfg,
                      np.random.default_rng(9))
    assert record["measured"] is True
    assert record["manipulation"]["adapter_is_the_banks_own"] is True
    routes = p5.estimate_routes(bank, cfg, manipulation_estimates=[record])
    assert routes["identification"] == PI.IDENTIFIED, routes["identification_judgement"]
    assert any("adapter" in d for d in routes["identification_judgement"]["open_decisions"])


# --------------------------------------------------------------------------------------------------
# The gap is estimated once, and the estimator is the crossed model's
# --------------------------------------------------------------------------------------------------


def test_the_route_gap_is_the_crossed_excess_and_not_a_difference_of_two_route_errors():
    """The two routes share every cell, so their difference has a covariance no hypot carries.

    The gap reported is the crossed model's kappa with its own standard error. The naive alternative,
    differencing the two route point estimates and combining their standard errors as though the two
    fits were independent, is computed here beside it: on this bank it is a different number, and
    there is a margin at which the two arithmetics return different labels for the same bank. That
    margin is constructed rather than registered, because the point is that the two estimators are
    not interchangeable and not that the registered margin happens to separate them.
    """
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, seed=2, beta=0.5, available_rate_exponent=0.2)
    routes = p5.estimate_routes(bank, cfg)
    rows = [r for r in bank["rows"] if not r.get("control")]
    crossed = PO.fit_paired(rows, PO.model_for(lad), cfg.z_interval, crossed=True)
    test = routes["route_consistency"]["test"]

    assert "crossed model's excess retention exponent" in routes["route_consistency"]["gap_estimate_source"]
    assert test["point"] == pytest.approx(crossed.retention_excess, abs=1e-12)
    assert test["se"] == pytest.approx(crossed.retention_excess_se, abs=1e-12)

    naive_point = routes["beta_retention_route"] - routes["beta_state_route"]
    naive_se = float(np.hypot(routes["se_state_route"], routes["se_retention_route"]))
    assert naive_se != pytest.approx(test["se"], rel=1e-3)

    half_ours = PI.TOST_Z * test["se"]
    half_naive = PI.TOST_Z * naive_se
    between = (abs(test["point"]) + half_ours + abs(naive_point) + half_naive) / 2.0
    assert PI.equivalence(test["point"], test["se"], between)["label"] == PI.CONSISTENT
    assert PI.equivalence(naive_point, naive_se, between)["label"] == PI.UNRESOLVED


# --------------------------------------------------------------------------------------------------
# The negative control, and a second channel that measured nothing
# --------------------------------------------------------------------------------------------------


def test_a_material_coupling_in_the_control_withholds_identification_from_an_agreeing_manipulation():
    """The control leaks a coupling, so the estimator is responding to something else.

    No agreement elsewhere repairs that, and the branch that says so had no test at all: the whole
    suite ran without the mock's `control_leak` ever being used.
    """
    judgement = PI.judge(
        consistency={"label": PI.CONSISTENT, "capability_elasticity": 0.5, "retention_elasticity": 0.5},
        control={"status": "MATERIAL COUPLING", "beta_control": 0.42},
        bank_elasticity=0.50, bank_elasticity_se=0.01,
        manipulations=[{"name": "agrees", "beta": 0.50, "beta_se": 0.01,
                        "manipulation": _manipulation(None, place_at_state=_distil).as_record(_place)}],
        margin=0.10)
    assert judgement["manipulations"][0]["agreement"] == PI.CONSISTENT
    assert judgement["label"] == PI.NOT_IDENTIFIED
    assert "negative control carries a material coupling" in judgement["reason"]


def test_a_leaking_control_is_measured_as_a_material_coupling_on_a_real_bank():
    """And the same branch reached from a bank rather than from a hand-made record."""
    cfg = p5.P5Config(reps=8)
    bank, ad, lad = _bank(cfg, seed=5, beta=0.5, control_leak=0.9, control_floor=0.9)
    routes = p5.estimate_routes(bank, cfg)
    assert routes["negative_control"]["status"] == "MATERIAL COUPLING", routes["negative_control"]
    assert routes["identification"] == PI.NOT_IDENTIFIED


def test_a_second_channel_that_measured_nothing_is_not_reported_as_an_imprecise_comparison():
    """A fit that failed and a comparison that was too broad are different statements.

    Both used to land in the branch that says "the intervals are too broad to decide", which
    describes an imprecise comparison where there was no comparison at all.
    """
    cfg = p5.P5Config(reps=4)
    lad = _ladder()
    ad = adapters.MockCouplingAdapter(beta=0.5)
    silent = adapters.MockCouplingAdapter(beta=0.5, a=0.0)      # no increment anywhere: nothing to fit
    record = _measure(_manipulation(silent, name="measured nothing"), ad, lad, cfg,
                      np.random.default_rng(4))
    assert record["measured"] is True and not np.isfinite(record["beta"])

    judgement = PI.judge(consistency={"label": PI.CONSISTENT},
                         control={"status": "NO MATERIAL COUPLING"},
                         bank_elasticity=0.5, bank_elasticity_se=0.01,
                         manipulations=[record], margin=0.10)
    assert judgement["label"] == PI.NOT_ESTABLISHED
    assert "produced no readable elasticity" in judgement["reason"]
    assert "too broad" not in judgement["reason"]
    assert judgement["manipulations"][0]["measured_an_elasticity"] is False


# --------------------------------------------------------------------------------------------------
# The labels themselves
# --------------------------------------------------------------------------------------------------


def test_the_consistency_check_never_returns_the_word_identified():
    """The check's own vocabulary, asserted as vocabulary.

    A label is what a reader carries away, and the whole of this finding is that one label was
    carrying away a claim the measurement could not make. The three labels the consistency check may
    return are asserted here across a range of gaps and precisions so that no branch of it can
    reintroduce the fourth.
    """
    for point in (-1.0, -0.2, -0.05, 0.0, 0.05, 0.2, 1.0):
        for se in (0.0001, 0.01, 0.05, 0.3, float("nan")):
            label = PI.equivalence(point, se, 0.10)["label"]
            assert label in (PI.CONSISTENT, PI.INCONSISTENT, PI.UNRESOLVED)
            assert label != PI.IDENTIFIED


def test_equivalence_is_the_interval_and_never_a_point_comparison():
    inside = PI.equivalence(0.02, 0.01, 0.10)
    outside = PI.equivalence(0.30, 0.01, 0.10)
    spanning = PI.equivalence(0.02, 0.20, 0.10)
    assert inside["label"] == PI.CONSISTENT
    assert outside["label"] == PI.INCONSISTENT
    assert spanning["label"] == PI.UNRESOLVED
    assert inside["interval"] == pytest.approx([0.02 - 1.645 * 0.01, 0.02 + 1.645 * 0.01])
    # a gap with no standard error is unresolved, not agreed: the point is not substituted for the interval
    assert PI.equivalence(0.001, float("nan"), 0.10)["label"] == PI.UNRESOLVED


def test_the_verdict_block_reports_the_two_judgements_separately():
    cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
    ad = adapters.MockCouplingAdapter(beta=0.5, noise=0.03)
    lad = _ladder()
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    res = p5.run_p5(ad, lad, cfg, 7, _place, start, ["S1", "S2"],
                    manipulations=[_manipulation(adapters.MockCouplingAdapter(beta=0.5, a=1.1, noise=0.03))])
    v = res["verdicts"]
    assert v["ROUTE_AGREEMENT"] in (PI.CONSISTENT, PI.INCONSISTENT, PI.UNRESOLVED)
    assert v["IDENTIFICATION"] == PI.IDENTIFIED, v["identification_judgement"]
    assert v["identification_judgement"]["requirement"].startswith("IDENTIFIED requires")
    assert np.isfinite(v["capability_elasticity"]) and np.isfinite(v["retention_elasticity"])


def test_a_bundle_rescores_to_the_same_identification(tmp_path):
    """The manipulations travel with the evidence, so an analyst reaches the same judgement.

    A bundle that carried the label but not the interventions it rests on would let a reanalysis
    report NOT ESTABLISHED for a run that did intervene twice. That is the same defect as the one
    this finding repairs, pointing the other way: a label that does not follow the evidence.
    """
    cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
    ad = adapters.MockCouplingAdapter(beta=0.5, noise=0.03)
    lad = _ladder()
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    out = tmp_path / "bundle"
    res = p5.run_p5(ad, lad, cfg, 7, _place, start, ["S1", "S2"], bundle=str(out),
                    manipulations=[_manipulation(adapters.MockCouplingAdapter(beta=0.5, a=1.1, noise=0.03))])
    bundle = CUSTODY.load_bundle(str(out))

    assert len(bundle["capability_manipulations"]) == 1
    assert bundle["capability_manipulations"][0]["manipulation"]["exclusion_restriction"]
    assert bundle["capability_manipulations"][0]["manipulation"]["adapter"]
    again = CUSTODY.reanalyse_bank(bundle)
    assert again["identification"] == res["routes"]["identification"]
    assert again["route_agreement"] == res["routes"]["route_agreement"]
    assert CUSTODY.recompute_verdicts(bundle)["IDENTIFICATION"] == res["verdicts"]["IDENTIFICATION"]


# --------------------------------------------------------------------------------------------------
# The second channels are a pre-run commitment, and the deciding gate sees them
# --------------------------------------------------------------------------------------------------


def test_the_second_channels_are_inside_the_sealed_specification():
    """A channel added after the bank was read, or dropped because it disagreed, leaves a trace.

    The manipulations are the only route to IDENTIFIED, so the set of them is a commitment in exactly
    the way the assigned held-out panel is. Their descriptions go into the manifest's configuration
    before the first paid call, and the specification hash covers the configuration.
    """
    cfg = p5.P5Config(reps=4, window_end=16, checkpoints=(4, 8, 16))
    ad = adapters.MockCouplingAdapter(beta=0.5, noise=0.03)
    lad = _ladder()
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    res = p5.run_p5(ad, lad, cfg, 7, _place, start, ["S1"],
                    manipulations=[_manipulation(adapters.MockCouplingAdapter(beta=0.5, a=1.1,
                                                                             noise=0.03))])
    sealed = res["manifest"]["config"]["capability_manipulations"]
    assert len(sealed) == 1 and sealed[0]["exclusion_restriction"]
    assert sealed[0]["placement_is_the_banks_own_loader"] is False

    before = CUSTODY.spec_hash_of(res["manifest"])
    moved = dict(res["manifest"])
    moved["config"] = dict(moved["config"], capability_manipulations=[])
    assert CUSTODY.spec_hash_of(moved) != before


def test_the_deciding_gate_refuses_an_undocumented_second_channel():
    """The gate had never been shown a manipulation, and a manipulation is now the whole claim."""
    inputs = MODE.ConfirmatoryInputs(
        manipulations=(PI.CapabilityManipulation(name="unwritten", place_at_state=_distil),))
    refusals = MODE.missing_confirmatory_inputs(inputs)
    named = [r for r in refusals if r.startswith("capability-manipulation")]
    assert named, refusals
    assert any("no exclusion restriction" in r for r in named)


def test_the_deciding_gate_refuses_a_second_channel_whose_loader_manufactures_its_cells(tmp_path):
    """The placeholder refusal of finding A9, applied to the channel that had escaped it.

    The loader here returns a number dressed as an artefact rather than the artefact the store holds,
    which is the object A9 exists to refuse for the bank. It reached the identification through the
    second channel, where nothing looked.
    """
    store = CD.CheckpointStore(str(tmp_path))
    for s in (40.0, 90.0):
        store.save(CD.state_name(s), CD.new_artefact("def add(x, y):\n    return x + y"))
    placeholder = lambda s: {"kind": "manufactured", "capability": float(s)}
    m = _manipulation(None, name="manufactured cells", place_at_state=placeholder, states=(40.0, 90.0))
    inputs = MODE.ConfirmatoryInputs(checkpoint_store=store, states=(40.0, 90.0), manipulations=(m,))
    refusals = MODE.missing_confirmatory_inputs(inputs)
    named = [r for r in refusals if r.startswith("capability-manipulation")]
    assert any("manufactured a cell rather than loading one" in r for r in named), refusals


def test_the_command_line_says_that_identified_is_unreachable_from_it(capsys):
    """There is no option that supplies a second channel, so the summary says so.

    A reader who sees NOT ESTABLISHED on every run this command line produces is entitled to know
    whether that is a finding about the bank or a fact about the way the run was started.
    """
    from arc_runner import cli
    assert cli.main(["demonstrate", "p5"]) == 0
    printed = capsys.readouterr().out
    assert "IDENTIFICATION NOT ESTABLISHED" in printed
    assert "IDENTIFIED is not reachable from it in any mode" in printed


# --------------------------------------------------------------------------------------------------
# Which elasticity the sealed prediction is built from, which is a separate question again
# --------------------------------------------------------------------------------------------------


def test_the_sealed_prediction_uses_the_pooled_fit_when_the_two_directions_are_consistent():
    cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
    ad = adapters.MockCouplingAdapter(beta=0.5, noise=0.03)
    lad = _ladder()
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    r = p5.run_p5(ad, lad, cfg, 7, _place, start, ["S1"])["routes"]
    assert r["route_agreement"] == PI.CONSISTENT, r["route_consistency"]
    assert r["beta_used_for_prediction"] == pytest.approx(r["beta_pooled"], abs=1e-12)
    assert "pooled fit" in r["beta_for_prediction_source"]


def test_the_sealed_prediction_uses_the_state_route_when_the_two_directions_disagree():
    """The registered rule, restored and pinned.

    The first repair of this finding swapped this branch for the crossed fit's capability exponent.
    The two estimate the same quantity, so the swap was not a large number; it was an unregistered
    one, and an unregistered estimator inside a sealed prediction is the thing a seal exists to
    prevent. The test states its own precondition first, that the two numbers differ at all on this
    bank, so that the identity asserted afterwards is not satisfied by both estimators at once.
    """
    cfg = p5.P5Config(reps=8, window_end=32, checkpoints=(4, 8, 16, 32))
    ad = adapters.MockCouplingAdapter(beta=0.5, retention_exponent=0.9, noise=0.05)
    lad = _ladder()
    start = lambda name: {"kind": "mock", "capability": 20.0, "rounds": 0, "system": name}
    r = p5.run_p5(ad, lad, cfg, 7, _place, start, ["S1"])["routes"]

    assert r["route_agreement"] == PI.INCONSISTENT, r["route_consistency"]
    assert abs(r["beta_state_route"] - r["beta_capability"]) > 1e-9, r   # the precondition
    assert r["beta_used_for_prediction"] == pytest.approx(r["beta_state_route"], abs=1e-12)
    assert r["beta_used_for_prediction"] != pytest.approx(r["beta_capability"], abs=1e-12)
    assert "state route" in r["beta_for_prediction_source"]
    assert r["beta_for_prediction_alternatives"]["crossed_capability_exponent"] == \
        pytest.approx(r["beta_capability"], abs=1e-12)
