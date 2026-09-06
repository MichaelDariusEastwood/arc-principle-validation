"""Mutation tests: five deliberate mistakes that the fixtures must catch (the consolidated package's
requirement of 5 September 2026). Each mutant is a small reimplementation of a rule with one error; the
fixture must return different verdicts for the mutant and the engine, and the engine's verdict must be
the contract's. A suite that cannot tell a mutant from the engine is not evidence of anything."""
from arc_instruments import balance as bl
from arc_instruments import conversion as cv
from arc_instruments import verdicts as V


def mutant_closed_boundary(iv):
    return V.region_verdict(iv, 0.5, 0.10, V.ABOVE_NULL, V.BELOW_NULL, V.EQUIVALENT_TO_NULL, closed=True)


def mutant_unresolved_as_losses(cells):
    wins = sum(1 for c in cells if c == "power")
    losses = len(cells) - wins
    if wins * 2 > len(cells):
        return V.SUPPORTED
    if losses * 2 > len(cells):
        return V.REFUTED
    return V.INCONCLUSIVE


def mutant_drop_direct_term(direct, cross, allocation):
    return cross * allocation


def mutant_wrong_sign_balance(alpha, chi):
    return alpha * (1.0 - chi) - 1.0


def mutant_default_for_unavailable(gamma_values, adv, margin):
    g = list(gamma_values)
    if not (any(x <= 0.4 for x in g) and any(x >= 0.6 for x in g)):
        return V.NOT_DISCRIMINATING
    return V.p20_form(g, adv, margin, discriminating_power=0.9)


def test_open_swapped_for_closed_is_caught():
    contact = (0.40, 0.55)
    assert mutant_closed_boundary(contact) == V.EQUIVALENT_TO_NULL
    assert V.p8_departure(contact) == V.INCONCLUSIVE


def test_unresolved_cells_as_failures_is_caught():
    cells = [V.INCONCLUSIVE] * 8
    assert mutant_unresolved_as_losses(cells) == V.REFUTED
    assert V.p1_form(cells, 0.05, 0.10) == V.INCONCLUSIVE


def test_dropped_direct_term_is_caught():
    assert abs(mutant_drop_direct_term(-0.4, 0.8, 1.0) - 0.8) < 1e-12
    assert abs(cv.target_axis_elasticity(-0.4, 0.8, 1.0) - 0.4) < 1e-12


def test_wrong_sign_is_caught():
    assert mutant_wrong_sign_balance(3.0, 0.5) > 0
    assert bl.balance_elasticity(3.0, 0.5) < 0


def test_default_for_unavailable_value_is_caught():
    adv = {"reciprocal": (0.15, 0.40)}
    assert mutant_default_for_unavailable([0.2, 0.4], adv, 0.10) == V.NOT_DISCRIMINATING
    assert V.p20_form([0.2, 0.4], adv, 0.10) == V.NOT_EVALUABLE
