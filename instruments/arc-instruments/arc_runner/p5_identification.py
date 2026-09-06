"""P5's identification: what the two regression directions settle, and what only an intervention can.

WHY THIS FILE EXISTS (finding A6). The runner's bank crosses capability states with retention
fractions and fits two regression directions through it: the response to the STATE at fixed
retention, and the response to RETENTION at fixed state. When those two numbers came out within a
fixed distance of each other, and the negative control showed nothing, the run printed IDENTIFIED.

Three separate things were wrong with that, and they are three separate repairs.

THE FIRST IS WHAT THE TWO DIRECTIONS ARE. They are not two interventions. Nothing was manipulated
twice: one bank was measured once, and two slopes were taken through it in two directions. The
charter's identification asks for two ways of CHANGING CAPABILITY, one of which does not go through
recursion, so that agreement between them says the elasticity belongs to capability rather than to
the pathway that produced it. Two slopes through one bank cannot say that, however closely they
agree, because they share every cell, every artefact, every read and every nuisance those carry.

THE SECOND IS WHAT AGREEMENT BETWEEN THEM MEANS. Under the general process a * f ** theta * U ** beta
the state direction estimates beta and the retention direction estimates theta, so the gap between
them estimates theta - beta and their agreement is a test of theta = beta. That is a real and useful
test, and it is not identification of beta: the capability elasticity remains identified when the two
differ, by the crossed fit that estimates both, which is why this module refuses to read a route
disagreement as a refutation of the mechanism. It reads it as what it is, which is that this bank
carries two elasticities and nothing in it chooses between them. And agreement is not sufficient
either: a nuisance rate that scales with the AVAILABLE capability adds the same amount to both
elasticities, so the two directions agree exactly while both are measuring beta plus the nuisance.
Agreement without identification is a world, not a hypothetical, and it is simulated in the tests.

THE THIRD IS THE ARITHMETIC OF THE COMPARISON. The gap was compared with a fixed margin as a POINT.
A point comparison declares agreement from an imprecise bank, because an estimate with a standard
error of a third and a gap of a hundredth is a bank that measured nothing and reported agreement.
The comparison here is a two one-sided tests equivalence: the interval on the gap must lie wholly
inside the margin to be called consistent, wholly outside to be called inconsistent, and anything
else is unresolved and says so. The gap is estimated ONCE, as the crossed model's excess retention
exponent, rather than differenced between two fits that share every cell and therefore have a
covariance that no difference of two standard errors carries.

AND THE BOUNDARY IS THE REGISTERED ONE, WHICH IT WAS NOT (a defect in the first repair of this
finding, found in review). The equivalence written here read `abs(point) + half <= margin`, which is
a CLOSED boundary, while `arc_runner.p5_prediction.IntervalConvention` is the registered convention
and reads strict clearance: an interval whose end lands exactly on the margin has not cleared it.
On a gap of 0.125 with a half width of 0.125 against a margin of 0.25 the registered rule returned
UNRESOLVED and this file returned CONSISTENT, and the divergence ran one way only, since the
INCONSISTENT branch was strict already. Every equivalence in this file now goes through the
convention object itself, so the two cannot say different things about the same numbers, and the
level and the multiplier are read from it rather than declared here a second time.

WHAT THIS MODULE RETURNS, AND WHAT IT REFUSES TO RETURN. `route_consistency` returns CONSISTENT,
INCONSISTENT or UNRESOLVED and never the word IDENTIFIED. `judge` returns the identification
judgement, and returns IDENTIFIED only when the domain supplied a capability manipulation that does
not run through retention, documented its exclusion restriction, and that manipulation's own estimate
of the capability elasticity agrees with the bank's inside the same equivalence. With no such
manipulation the judgement is NOT ESTABLISHED, which is a statement about the design and not about
the mechanism.

AND THE JUDGEMENT RE-DERIVES ADMISSIBILITY RATHER THAN TRUSTING A LABEL (the second defect of the
first repair). `judge` used to read a `documentation_failures` key off each record and treat an empty
list as documentation. A record is a mapping a caller supplies and a bundle carries, so a manipulation
with no channel, no exclusion restriction and no declarations at all reached IDENTIFIED by carrying an
empty list, and a saved bundle re-scored to the label it was written with rather than to the evidence
under it. Every requirement is now re-derived from the manipulation's own DESCRIPTION at the moment
the word is produced, by `failures_in_record`, and a record carrying no description at all is refused
for that reason: a claim that cannot be checked is not a claim that passed.

STRUCTURAL INDEPENDENCE, WHERE IT COSTS NOTHING TO CHECK IT. Whether a second channel truly avoids
the bank's nuisance is a domain declaration and this module says so. One case needs no domain
knowledge: a manipulation whose placement callable IS the bank's own placement callable is the same
intervention twice, whatever it declares, and it is refused as such. Whether the check was performed
at all is recorded, and an unperformed check on a manipulation that ran its own bank fails closed,
for the same reason an undeclared nuisance answer does: the identification rests on the answer, and
the question was not asked.

OPEN DECISIONS, NAMED RATHER THAN TAKEN. Three choices belong to the author and the conservative
reading is implemented here and reported in every record. The equivalence level and boundary are the
registered convention's, being the ninety per cent two-sided interval with strict clearance, read
from `arc_runner.p5_prediction.REGISTERED`; a different registered level or boundary belongs in that
one object and reaches this file from it. The manipulation's agreement is tested against the same
`route_margin` as the route gap, because the registration names one margin on the exponent scale and
inventing a second one here would be this module choosing a threshold; if the author wants a separate
identification margin it belongs in the configuration beside that one. And an inconsistent route pair
with no manipulation is reported as NOT IDENTIFIED rather than NOT ESTABLISHED, because two
elasticities with nothing to choose between them is a bank that has failed to identify one, and
because the design battery requires non-identification to be DECLARED under a capability-dependent
nuisance rate rather than left silent.

A fourth choice is recorded and NOT taken: whether a manipulation that reuses the bank's own model
adapter can be a second channel. The adapter is the system being measured, and a second placement of
the same system is what the design asks for, so reusing it is not refused; it is recorded in every
manipulation record as `adapter_is_the_banks_own` so that a reader can see it and an author can
decide otherwise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np

from . import p5_prediction as PRED

CONSISTENT = "CONSISTENT"
INCONSISTENT = "INCONSISTENT"
UNRESOLVED = "UNRESOLVED"

IDENTIFIED = "IDENTIFIED"
NOT_IDENTIFIED = "NOT IDENTIFIED"
NOT_ESTABLISHED = "NOT ESTABLISHED"
INCONCLUSIVE = "INCONCLUSIVE"

# The registered convention's multiplier, read from the one object that declares it and kept here
# under its old name because callers and tests have imported `PI.TOST_Z` since the first repair. It
# is an ALIAS and never a second declaration: a number written down twice is a number that can drift,
# and this file and `arc_runner.p5_prediction` had already begun citing each other for it.
TOST_Z = PRED.REGISTERED.equivalence_z

# What the three labels mean, mapped from the registered convention's own vocabulary so that the
# boundary rule cannot differ between the final comparison and this one.
_FROM_CLEARANCE = {PRED.AGREES: CONSISTENT, PRED.DISAGREES: INCONSISTENT, PRED.UNRESOLVED: UNRESOLVED}


def _number(value: Any) -> float:
    """A float or a not-a-number, with no truthiness anywhere near it."""
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def equivalence(point: float, se: float, margin: float, z: Optional[float] = None,
                convention: Optional[PRED.IntervalConvention] = None) -> Dict[str, Any]:
    """Two one-sided tests on one difference, reported as an interval and a label.

    CONSISTENT when the interval lies wholly inside the margin, INCONSISTENT when it lies wholly
    outside, UNRESOLVED otherwise, which includes every case where there is no interval at all. A
    point comparison would call the last of those consistent whenever the point happened to be small,
    which is the defect finding A6 names: the label would then be a statement about the bank's
    imprecision rather than about the quantity.

    THE BOUNDARY IS NOT DECIDED HERE. The half width and the margin are handed to the registered
    convention's own `clearance`, and its three verdicts are renamed to this file's three labels.
    Writing the comparison out again here is what let a closed boundary live in this file while the
    registered convention read a strict one, so the comparison is not written out again.
    """
    conv = convention if convention is not None else PRED.REGISTERED
    zz = float(conv.equivalence_z if z is None else z)
    m = float(margin)
    if not np.isfinite(point):
        return {"label": UNRESOLVED, "point": float("nan"), "se": float("nan"),
                "half_width": float("nan"), "margin": m,
                "reason": "the gap has no estimate on this bank"}
    if not np.isfinite(se):
        return {"label": UNRESOLVED, "point": float(point), "se": float("nan"),
                "half_width": float("inf"), "margin": m,
                "reason": "the gap has a point estimate and no standard error, so no interval can be "
                          "read; a point comparison is refused here rather than substituted"}
    half = float(abs(zz) * se)
    reading = conv.clearance(point, half, m)
    return {"label": _FROM_CLEARANCE[reading["verdict"]], "point": float(point), "se": float(se),
            "half_width": half, "margin": m,
            "interval": [float(point - half), float(point + half)],
            "level": conv.level, "boundary": reading.get("boundary"),
            "boundary_rule": "the registered interval convention decides the boundary; this file "
                             "does not write the comparison out a second time",
            "reason": reading["reason"]}


def route_consistency(crossed_fit: Any, state_route: Mapping[str, Any],
                      retention_route: Mapping[str, Any], margin: float,
                      z: Optional[float] = None,
                      convention: Optional[PRED.IntervalConvention] = None) -> Dict[str, Any]:
    """The numerical consistency of the two regression directions. Never an identification.

    The gap is the crossed model's excess retention exponent, kappa = theta - beta, because that is
    the population quantity the two directions differ by and it is estimated once with its own
    standard error. The difference of the two route point estimates is reported beside it as the
    descriptive number it is: the two routes are fitted to overlapping cells, so no interval can be
    built by combining their standard errors as though they were independent, and this module will
    not manufacture one.
    """
    b1 = float(state_route.get("beta", float("nan")))
    b2 = float(retention_route.get("beta", float("nan")))
    observed = abs(b1 - b2) if np.isfinite(b1) and np.isfinite(b2) else float("nan")
    usable = bool(getattr(crossed_fit, "usable", False))
    if usable:
        test = equivalence(getattr(crossed_fit, "retention_excess", float("nan")),
                           getattr(crossed_fit, "retention_excess_se", float("nan")), margin, z,
                           convention)
        source = ("the crossed model's excess retention exponent, estimated once from every cell "
                  "with its own standard error")
    else:
        test = {"label": UNRESOLVED, "point": float("nan"), "se": float("nan"),
                "half_width": float("nan"), "margin": float(margin),
                "reason": "the crossed model did not fit this bank (%s), so the gap has no interval "
                          "and the routes are not compared as a point"
                          % getattr(crossed_fit, "adequacy", "no fit")}
        source = "none: the crossed model did not fit"
    return {"label": test["label"], "test": test, "gap_estimate_source": source,
            "observed_route_difference": observed,
            "beta_state_route": b1, "beta_retention_route": b2,
            "capability_elasticity": float(getattr(crossed_fit, "beta", float("nan"))),
            "retention_elasticity": float(getattr(crossed_fit, "retention_elasticity", float("nan"))),
            "what_this_is": "a numerical consistency check between two regression directions through "
                            "one bank. It tests whether the elasticity in the retention fraction and "
                            "the elasticity in the capability state are the same number. It is not a "
                            "second intervention and it is never an identification.",
            "what_this_is_not": "agreement here is consistent with a nuisance rate that scales with "
                                "the available capability, which adds equally to both elasticities; "
                                "disagreement here is consistent with a retention elasticity that "
                                "genuinely differs from the capability elasticity, in which case the "
                                "capability elasticity is still estimated by the crossed fit."}


@dataclass(frozen=True)
class CapabilityManipulation:
    """A second way of placing the capability state, supplied by the domain.

    This is the object the charter's identification needs and the bank cannot contain: an
    intervention that moves capability through a channel that is not retention, so that the
    elasticity measured under it and the elasticity measured under the bank's own placement are two
    estimates of one quantity produced by two different causes. Agreement between them is evidence
    about the estimand; agreement between two slopes through one bank is not.

    `exclusion_restriction` is required text and is not decoration. Every identification claim in
    this design rests on one, being the assumption that the second channel does not itself carry the
    nuisance the first one might, and a claim whose assumption is not written down cannot be argued
    with. A manipulation that does not state it is refused as undocumented rather than used.

    `place_at_state` is the domain's second loader, run as its own bank. `estimate` is the
    alternative for a domain that measured the elasticity under the second channel elsewhere and
    carries the number and its standard error; one or the other is required and neither is inferred.
    An estimate must also carry a `provenance`, because a number typed into a mapping and a number
    measured somewhere are indistinguishable in the mapping and only one of them is evidence.

    `shares_nuisance_with_bank` is declared and not guessed. An undeclared answer fails closed to
    undocumented, because the whole content of the exclusion restriction is that the answer is no,
    and a manipulation that has not been asked the question has not answered it.
    """

    name: str
    channel: str = ""
    exclusion_restriction: str = ""
    independent_of_retention: bool = False
    place_at_state: Optional[Callable[[float], Any]] = None
    adapter: Any = None
    states: Optional[Sequence[float]] = None
    estimate: Optional[Mapping[str, Any]] = None
    shares_nuisance_with_bank: Optional[bool] = None

    def as_record(self, bank_place_at_state: Any = None, bank_adapter: Any = None) -> Dict[str, Any]:
        """What travels with the measurement, into the run record and into the evidence bundle.

        The two bank arguments are the structural comparison, and they are arguments rather than
        fields because they belong to the run and not to the manipulation. Where they are not
        supplied the record says the comparison was not performed, which `failures_in_record` reads
        as a failure for a manipulation that ran its own bank: an identification whose independence
        was never checked has not been checked, and silence about it is the same fail-closed case as
        an undeclared nuisance pathway.
        """
        own_loader: Optional[bool] = None
        if bank_place_at_state is not None and self.place_at_state is not None:
            own_loader = self.place_at_state is bank_place_at_state
        own_adapter: Optional[bool] = None
        if bank_adapter is not None:
            own_adapter = (self.adapter is None) or (self.adapter is bank_adapter)
        est = dict(self.estimate) if self.estimate else None
        return {"name": self.name, "channel": self.channel,
                "exclusion_restriction": self.exclusion_restriction,
                "independent_of_retention": bool(self.independent_of_retention),
                "shares_nuisance_with_bank": self.shares_nuisance_with_bank,
                "measured_by": ("its own bank" if self.place_at_state is not None else
                                "an estimate supplied with the manipulation" if self.estimate else
                                "nothing"),
                "states": list(self.states) if self.states else None,
                # WHICH SYSTEM ANSWERED THE SECOND CHANNEL, NAMED IN THE RECORD. A manipulation is now
                # the only route to IDENTIFIED, and a record that does not say which adapter measured
                # it cannot show that a mock did not.
                "adapter": (None if self.adapter is None else
                            str(getattr(self.adapter, "name", type(self.adapter).__name__))),
                "adapter_is_the_banks_own": own_adapter,
                "placement_checked_against_the_bank": bank_place_at_state is not None,
                "placement_is_the_banks_own_loader": own_loader,
                "estimate_provenance": (str(est.get("provenance") or "") if est else ""),
                "estimate": est}


def failures_in_record(record: Optional[Mapping[str, Any]]) -> List[str]:
    """Every requirement a manipulation's DESCRIPTION fails, re-derived from the description itself.

    This is the function the judgement calls, and it is the reason the judgement no longer trusts a
    `documentation_failures` key. That key is written by whoever built the record: a caller passing a
    hand-made mapping, or a bundle written by an earlier run. A record whose key said `[]` and whose
    description said nothing at all reached IDENTIFIED, which is the defect of this finding arriving
    through the door the repair opened. The requirements are checked here against the description
    every time the word is produced, so a record can be believed only about what it actually states.
    """
    if not isinstance(record, Mapping) or not record:
        return ["no manipulation description travels with this record, so nothing about it can be "
                "checked: an identification that rests on a claim nobody can read is not an "
                "identification"]
    out: List[str] = []
    if not str(record.get("name") or "").strip():
        out.append("no name")
    if not str(record.get("channel") or "").strip():
        out.append("no channel: how this manipulation places the capability state is not stated")
    if not str(record.get("exclusion_restriction") or "").strip():
        out.append("no exclusion restriction: the assumption the identification would rest on is not "
                   "written down")
    if not record.get("independent_of_retention"):
        out.append("not declared independent of retention: a second lever that runs through the same "
                   "channel as the first is the same intervention twice")
    shares = record.get("shares_nuisance_with_bank")
    if shares is None:
        out.append("shares_nuisance_with_bank is undeclared, which fails closed: the exclusion "
                   "restriction says the answer is no, and it has not been asked")
    elif shares:
        out.append("declared to share a nuisance pathway with the bank, so agreement between the two "
                   "is expected under the nuisance as well as under the mechanism")
    measured_by = record.get("measured_by")
    if measured_by == "its own bank":
        # THE ONE STRUCTURAL CHECK THAT NEEDS NO DOMAIN KNOWLEDGE. Everything else about independence
        # is the domain's declaration and is treated as one. Whether the second channel's placement
        # callable is the FIRST channel's placement callable is a fact about the run, and a
        # manipulation that places its cells with the bank's own loader has run the bank twice.
        own = record.get("placement_is_the_banks_own_loader")
        if own is True:
            out.append("this manipulation places the capability state with the bank's own loader, so "
                       "it is the same placement channel run a second time and not a second channel")
        elif own is None:
            out.append("whether this manipulation's placement is the bank's own loader was not "
                       "checked, which fails closed: the identification rests on the two channels "
                       "being different and the question was not asked")
    elif measured_by == "an estimate supplied with the manipulation":
        if not str(record.get("estimate_provenance") or "").strip():
            out.append("the supplied estimate carries no provenance: where and how the second "
                       "channel's elasticity was measured is not written down, so a typed number and "
                       "a measurement are the same object here")
    else:
        out.append("no placement loader and no supplied estimate, so this manipulation measures "
                   "nothing")
    return out


def documentation_failures(m: CapabilityManipulation, bank_place_at_state: Any = None,
                           bank_adapter: Any = None) -> List[str]:
    """Every requirement this manipulation fails, named at once rather than one at a time.

    It is the same function the judgement uses, applied to this manipulation's own record, so that a
    refusal before the bank runs and a refusal at the moment the word is produced cannot disagree.
    """
    return failures_in_record(m.as_record(bank_place_at_state, bank_adapter))


def admissible(m: CapabilityManipulation, bank_place_at_state: Any = None,
               bank_adapter: Any = None) -> bool:
    return not documentation_failures(m, bank_place_at_state, bank_adapter)


def _record_failures(r: Mapping[str, Any]) -> List[str]:
    """The failures of one measured record: what the measurement reported, and what the description
    itself says, with the second re-derived rather than believed. Order is preserved and duplicates
    are dropped, so a failure named by both appears once."""
    out: List[str] = []
    for f in list(r.get("documentation_failures") or []) + failures_in_record(r.get("manipulation")):
        if f not in out:
            out.append(f)
    return out


def judge(*, consistency: Mapping[str, Any], control: Optional[Mapping[str, Any]],
          bank_elasticity: float, bank_elasticity_se: float,
          manipulations: Sequence[Mapping[str, Any]] = (), margin: float = 0.10,
          z: Optional[float] = None, bank_usable: bool = True,
          convention: Optional[PRED.IntervalConvention] = None) -> Dict[str, Any]:
    """The identification judgement, as a separate object from the numerical consistency check.

    Each `manipulations` entry is a measured manipulation record: the manipulation's own description,
    its documentation failures if any, and its estimate of the capability elasticity with a standard
    error. The comparison with the bank's elasticity is an equivalence test on the difference, and
    the two estimates are independent because they come from different cells placed by different
    causes, which is the whole point of the second manipulation and the reason the difference has a
    standard error that can be written down at all.

    ADMISSIBILITY IS RE-DERIVED HERE and never read off the record's own label. See
    `failures_in_record`: the first version of this judgement trusted a `documentation_failures` key,
    and a record carrying an empty list and no description at all reached IDENTIFIED.
    """
    reasons: List[str] = []
    admissible_records = []
    for r in manipulations:
        failures = _record_failures(r)
        if failures:
            reasons.append("manipulation %r is not admissible: %s"
                           % (r.get("name"), "; ".join(failures)))
        else:
            admissible_records.append(r)
    agreements = []
    for r in admissible_records:
        # A manipulation whose estimate is exactly zero is an estimate, not a missing value, so the
        # test is `is None` and finiteness and never truthiness: a coupling of zero is one of the
        # worlds this design has to be able to report.
        est, se = _number(r.get("beta")), _number(r.get("beta_se"))
        diff = (float(est) - float(bank_elasticity)
                if np.isfinite(est) and np.isfinite(bank_elasticity) else float("nan"))
        combined = (float(np.hypot(se, float(bank_elasticity_se)))
                    if np.isfinite(se) and np.isfinite(bank_elasticity_se) else float("nan"))
        test = equivalence(diff, combined, margin, z, convention)
        agreements.append({"name": r.get("name"), "agreement": test["label"], "test": test,
                           "manipulation_beta": float(est), "manipulation_beta_se": float(se),
                           # WHETHER THIS SECOND CHANNEL MEASURED ANYTHING AT ALL, kept apart from
                           # whether it agreed. A manipulation whose own fit failed has no elasticity
                           # to compare, and reporting that as an equivalence too broad to decide
                           # describes an imprecise comparison where there was no comparison.
                           "measured_an_elasticity": bool(np.isfinite(est) and np.isfinite(se)),
                           "measurement_reason": r.get("reason"),
                           "bank_beta": float(bank_elasticity), "bank_beta_se": float(bank_elasticity_se),
                           "manipulation": r.get("manipulation"), "fit": r.get("fit")})
    control_status = (control or {}).get("status")
    unmeasured = [a for a in agreements if not a["measured_an_elasticity"]]
    label, why = NOT_ESTABLISHED, ""
    if not bank_usable:
        label = INCONCLUSIVE
        why = ("the bank carries no readable capability elasticity, so there is nothing for a second "
               "manipulation to agree or disagree with")
    elif control_status == "MATERIAL COUPLING":
        label = NOT_IDENTIFIED
        why = ("the negative control carries a material coupling, so the estimator is responding to "
               "something other than the mechanism and no agreement elsewhere repairs that")
    elif any(a["agreement"] == INCONSISTENT for a in agreements):
        label = NOT_IDENTIFIED
        why = ("an independent capability manipulation disagrees with the bank beyond the margin, so "
               "the two channels are not measuring one elasticity")
    elif agreements and all(a["agreement"] == CONSISTENT for a in agreements):
        if control_status == "NO MATERIAL COUPLING":
            label = IDENTIFIED
            why = ("an independent capability manipulation, documented and not running through "
                   "retention, agrees with the bank inside the equivalence, and the negative control "
                   "shows no material coupling")
        else:
            label = NOT_ESTABLISHED
            why = ("the independent manipulation agrees, and the negative control is %s, so the "
                   "condition that the estimator responds to nothing else is not established"
                   % (control_status or "not reported"))
    elif unmeasured:
        label = NOT_ESTABLISHED
        why = ("the independent capability manipulation(s) %s produced no readable elasticity of "
               "their own (%s), so there was nothing to compare with the bank. This is a second "
               "channel that did not measure, and not a comparison that was too imprecise to decide."
               % (", ".join(repr(a["name"]) for a in unmeasured),
                  "; ".join(str(a.get("measurement_reason") or "no reason recorded")
                            for a in unmeasured)))
    elif agreements:
        label = NOT_ESTABLISHED
        why = ("the independent capability manipulation neither agrees nor disagrees inside the "
               "equivalence: the intervals are too broad to decide, which is a statement about the "
               "precision of this design and not about the mechanism")
    elif consistency.get("label") == INCONSISTENT:
        label = NOT_IDENTIFIED
        why = ("no independent capability manipulation was supplied, and the two regression "
               "directions of the bank carry different elasticities (retention %.3f against "
               "capability %.3f), so this bank holds two readings and nothing in it chooses between "
               "them. This is a failure of the design to identify the elasticity and not a "
               "refutation of the mechanism: a capability mechanism whose channel is not retention "
               "alone produces exactly this."
               % (consistency.get("retention_elasticity", float("nan")),
                  consistency.get("capability_elasticity", float("nan"))))
    else:
        label = NOT_ESTABLISHED
        why = ("no independent capability manipulation was supplied. The two regression directions "
               "of one bank are two slopes through the same cells and their agreement is a test of "
               "whether the retention and capability elasticities are one number, which is not the "
               "same question as whether the capability elasticity is identified.")
    conv = convention if convention is not None else PRED.REGISTERED
    return {"label": label, "reason": why, "reasons": reasons,
            "route_consistency": consistency.get("label"),
            "negative_control_status": control_status,
            "manipulations": agreements,
            "n_manipulations_supplied": len(manipulations),
            "n_manipulations_admissible": len(admissible_records),
            "admissibility_source": "re-derived from each manipulation's own description at the "
                                    "moment this judgement was made, never read off a record's own "
                                    "documentation_failures key",
            "interval_convention": conv.as_record(),
            "requirement": "IDENTIFIED requires an independent capability manipulation that does not "
                           "run through retention, a written exclusion restriction, an estimate of "
                           "the capability elasticity from that manipulation agreeing with the "
                           "bank's inside the registered margin as an interval, and a negative "
                           "control showing no material coupling. Numerical agreement between the "
                           "bank's two regression directions is none of those things.",
            "open_decisions": [
                "the equivalence level and the boundary rule are the registered interval "
                "convention's, read from arc_runner.p5_prediction.REGISTERED, being the ninety per "
                "cent two-sided interval with strict clearance; a different registered level or "
                "boundary belongs in that one object and reaches this judgement from it",
                "the manipulation's agreement is tested against the same exponent-scale margin as "
                "the route gap, because the registration names one margin and a second one invented "
                "here would be this module choosing a threshold",
                "an inconsistent route pair with no manipulation is reported NOT IDENTIFIED rather "
                "than NOT ESTABLISHED, being the reading that does not claim identification and the "
                "one the design battery's nuisance-rate adversary requires to be declared",
                "a manipulation that reuses the bank's own model adapter is recorded and not "
                "refused, because the adapter is the system being measured and a second placement of "
                "the same system is what the design asks for; the record carries "
                "adapter_is_the_banks_own so that an author who reads it otherwise can see it"]}
