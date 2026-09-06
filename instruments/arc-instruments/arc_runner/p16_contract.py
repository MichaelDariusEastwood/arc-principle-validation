"""Candidate P16 component adjudication, for review before registration.

This does not replace the governing charter or its estimator. It requires explicit
margins and already estimated joint intervals, never fabricates missing alpha,
capacity, temporal evidence or assay validity. All thresholds come from its caller.
The output is contract compatibility, NOT empirical support for the theory.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Mapping, Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float
    def __post_init__(self):
        if not math.isfinite(self.lo) or not math.isfinite(self.hi) or self.lo > self.hi:
            raise ValueError("interval endpoints must be finite and ordered")
    def minus(self, other): return Interval(self.lo-other.hi, self.hi-other.lo)

@dataclass(frozen=True)
class UnitPrediction:
    unit_id: str
    role: str  # above, below, sham, baseline
    alpha: Interval  # independently measured target range sealed before outcomes
    delta: Interval  # predicted elasticity, with prediction uncertainty
    event_depth: Optional[Interval]
    required_horizon: int
    def __post_init__(self):
        if not self.unit_id or self.role not in {"above", "below", "sham", "baseline"}:
            raise ValueError("invalid experimental unit")
        if not isinstance(self.required_horizon, int) or self.required_horizon < 1:
            raise ValueError("a positive integer informative horizon is required")
        if self.alpha.lo <= 0: raise ValueError("alpha range must be positive for this candidate design")
        if self.role == "above" and self.event_depth is None:
            raise ValueError("above-boundary units need a sealed timing interval")
        if self.event_depth and (self.event_depth.lo < 0 or self.event_depth.hi > self.required_horizon):
            raise ValueError("timing prediction lies outside the informative horizon")

@dataclass(frozen=True)
class Contract:
    protocol_sha256: str
    units: Tuple[UnitPrediction, ...]
    line_slope: Interval
    line_zero: Interval
    delta_margin: float
    slope_margin: float
    zero_margin: float
    alpha_margin: float
    timing_margin: float
    sign_guard: float
    required_fraction: float  # strict majority if the approved specification supplies 0.5
    interval_method_id: str
    def __post_init__(self):
        if len(self.protocol_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.protocol_sha256):
            raise ValueError("the reviewed protocol digest is required")
        if not self.units or len({u.unit_id for u in self.units}) != len(self.units):
            raise ValueError("the assignment universe must be nonempty and unique")
        if {u.role for u in self.units} != {"above", "below", "sham", "baseline"}:
            raise ValueError("above, below, sham and baseline units are required")
        margins=(self.delta_margin,self.slope_margin,self.zero_margin,self.alpha_margin,self.timing_margin)
        if any(not math.isfinite(v) or v <= 0 for v in margins):
            raise ValueError("all equivalence margins must be supplied and positive")
        if not math.isfinite(self.sign_guard) or self.sign_guard < 0:
            raise ValueError("invalid sign guard")
        if not 0 <= self.required_fraction < 1 or not self.interval_method_id:
            raise ValueError("missing aggregation or interval specification")
        if self.line_slope.hi >= 0: raise ValueError("the predicted balance line must be falling")
        for u in self.units:
            if u.role == "above" and u.delta.hi >= -self.sign_guard:
                raise ValueError("above prediction is not negative beyond the guard")
            if u.role == "below" and u.delta.lo <= self.sign_guard:
                raise ValueError("below prediction is not positive beyond the guard")

@dataclass(frozen=True)
class Observation:
    alpha: Optional[Interval]
    delta: Optional[Interval]
    event_depth: Optional[Interval]
    observed_horizon: int
    censored: bool
    measurement_valid: bool
    interval_method_id: str
    def __post_init__(self):
        if not isinstance(self.observed_horizon, int) or self.observed_horizon < 0:
            raise ValueError("invalid observed horizon")
        if type(self.censored) is not bool or type(self.measurement_valid) is not bool:
            raise ValueError("measurement and censoring states must be Boolean")
        if self.event_depth and (self.event_depth.lo < 0 or self.event_depth.hi > self.observed_horizon):
            raise ValueError("observed event interval extends beyond observation")

@dataclass(frozen=True)
class LineObservation:
    slope: Optional[Interval]
    zero: Optional[Interval]
    interval_method_id: str
    uses_independent_measured_alpha: bool


def equivalent(difference: Optional[Interval], margin: float):
    """Strict boundaries; a boundary touch or partial overlap is unresolved."""
    if not math.isfinite(margin) or margin <= 0: raise ValueError("invalid margin")
    if difference is None: return "UNRESOLVED"
    if difference.lo > -margin and difference.hi < margin: return "MATCHED"
    if difference.lo > margin or difference.hi < -margin: return "ADVERSE"
    return "UNRESOLVED"


def log_ratio_elasticity(depths, correction_service, offered_burden):
    """Descriptive slope of log(Q/W) on log R. No interval or stability inference.

    Q and W must use independently justified units and the same exposure clock;
    positive finite arrays alone cannot certify that measurement requirement.
    """
    R,Q,W=(np.asarray(x,dtype=float) for x in (depths,correction_service,offered_burden))
    if R.ndim != 1 or Q.shape != R.shape or W.shape != R.shape or len(R)<3:
        raise ValueError("three matched one-dimensional observations are required")
    if not all(np.all(np.isfinite(x)&(x>0)) for x in (R,Q,W)) or not np.all(np.diff(R)>0):
        raise ValueError("depth must increase and observations must be finite and positive")
    return float(np.polyfit(np.log(R),np.log(Q)-np.log(W),1)[0])


def _unit(pred, obs, cfg):
    if obs is None: return {"result":"UNRESOLVED", "reason":"missing assigned unit"}
    if (not obs.measurement_valid or obs.censored or obs.observed_horizon < pred.required_horizon
        or obs.interval_method_id != cfg.interval_method_id or obs.alpha is None or obs.delta is None):
        return {"result":"UNRESOLVED", "reason":"measurement, horizon, censoring or interval gate"}
    parts={"alpha":equivalent(obs.alpha.minus(pred.alpha),cfg.alpha_margin),
           "delta":equivalent(obs.delta.minus(pred.delta),cfg.delta_margin)}
    if pred.role == "above":
        parts["sign"]="MATCHED" if obs.delta.hi < -cfg.sign_guard else (
            "ADVERSE" if obs.delta.lo > cfg.sign_guard else "UNRESOLVED")
        parts["timing"]=equivalent(obs.event_depth.minus(pred.event_depth) if obs.event_depth else None,
                                    cfg.timing_margin)
        # A non-alarm has no timing estimate and cannot become a refutation by itself.
    elif pred.role == "below":
        parts["sign"]="MATCHED" if obs.delta.lo > cfg.sign_guard else (
            "ADVERSE" if obs.delta.hi < -cfg.sign_guard else "UNRESOLVED")
    result="ADVERSE" if "ADVERSE" in parts.values() else (
        "MATCHED" if all(v=="MATCHED" for v in parts.values()) else "UNRESOLVED")
    return {"result":result,"components":parts}


def adjudicate(cfg: Contract, observations: Mapping[str, Observation], line: LineObservation):
    expected={u.unit_id for u in cfg.units}
    if set(observations)-expected: raise ValueError("unassigned observations cannot enter the result")
    rows={u.unit_id:_unit(u,observations.get(u.unit_id),cfg) for u in cfg.units}
    groups={}
    for role in ("above","below","sham","baseline"):
        vals=[rows[u.unit_id]["result"] for u in cfg.units if u.role==role]
        # Missing/invalid units stay in their original denominator.
        groups[role]={"assigned":len(vals),"matched":vals.count("MATCHED"),
                      "adverse":vals.count("ADVERSE"),"unresolved":vals.count("UNRESOLVED")}
    if line.interval_method_id != cfg.interval_method_id or not line.uses_independent_measured_alpha:
        line_parts={"slope":"UNRESOLVED","zero":"UNRESOLVED"}
    else:
        line_parts={"slope":equivalent(line.slope.minus(cfg.line_slope) if line.slope else None,cfg.slope_margin),
                    "zero":equivalent(line.zero.minus(cfg.line_zero) if line.zero else None,cfg.zero_margin)}
    controls_bad=any(groups[r]["adverse"] > cfg.required_fraction*groups[r]["assigned"] for r in ("sham","baseline"))
    all_matched=all(g["matched"] > cfg.required_fraction*g["assigned"] for g in groups.values())
    adverse=any(groups[r]["adverse"] > cfg.required_fraction*groups[r]["assigned"] for r in ("above","below"))
    if controls_bad: result="NOT SPECIFIC"
    elif "ADVERSE" in line_parts.values() or adverse: result="ADVERSE TO CANDIDATE CONTRACT"
    elif all_matched and all(v=="MATCHED" for v in line_parts.values()): result="MATCHES CANDIDATE CONTRACT"
    else: result="UNRESOLVED"
    return {"contract_result":result, "empirical_verdict":"NOT TESTED", "protocol_sha256":cfg.protocol_sha256,
            "per_unit":rows,"groups":groups,"line":line_parts,
            "limitations":["candidate contract requires author approval against the governing charter",
                "input intervals and measurement validity require independent assay review",
                "local component adjudication does not establish prospective commitment or finite-horizon safety"]}
