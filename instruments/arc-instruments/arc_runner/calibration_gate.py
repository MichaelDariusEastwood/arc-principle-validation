"""Validate a frozen P16 operating-characteristic record, not a calibration claim in prose.

The shipped 0.052 point estimate and legacy run-pattern battery are development evidence.
This gate requires whole-procedure errors, including both verdict directions, in each declared
world. Counts and provenance still need independent review: schema validation cannot authenticate
an investigator's results or establish coverage of every possible data-generating process.
"""
from dataclasses import asdict
import math
import re

from scipy.stats import beta

from arc_instruments.sealing import sha256_of

SCHEMA = "arc-p16-complete-procedure-calibration-v1"
REQUIRED_WORLDS = frozenset(("null", "positive_control", "wrong_location", "wrong_timing",
                             "wrong_slope", "failed_delivery", "correlated_noise"))


def configuration_digest(cfg):
    config = asdict(cfg)
    config.pop("alarm_rate_null_calibration", None)
    return sha256_of(config)


def refusals(cfg, quantity=None):
    """No defaults for evidential limits; the registration supplies them and the underlying counts."""
    from .custody import package_code_identity
    from .observation import BALANCE_ELASTICITY, SERVICE_RATIO, LOG_SERVICE_RATIO

    record = getattr(cfg, "alarm_rate_null_calibration", None)
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        return ["calibration: missing complete-procedure calibration; the default 0.052 is development-only"]
    out = []
    if record.get("configuration_sha256") != configuration_digest(cfg):
        out.append("calibration: configuration differs from the calibrated procedure")
    if record.get("code_sha256") != package_code_identity():
        out.append("calibration: implementation differs from the calibrated procedure")
    if record.get("endpoint") != "P16" or record.get("unit") != "whole_run":
        out.append("calibration: calibrate the final P16 wrapper, not per-look alarms or run_pattern")
    for field in ("registration", "evidence_uri", "reviewed_by"):
        if not isinstance(record.get(field), str) or not record[field].strip():
            out.append("calibration: missing " + field)
    if not re.fullmatch(r"[0-9a-f]{64}", str(record.get("evidence_sha256", ""))):
        out.append("calibration: missing evidence digest")
    try:
        confidence = float(record["simultaneous_confidence"])
        limit = float(record["registered_error_limit"])
        null = float(cfg.alarm_rate_null)
        if any(isinstance(v, bool) for v in (record["simultaneous_confidence"],
                                             record["registered_error_limit"], cfg.alarm_rate_null)):
            raise ValueError
        if not (0.5 < confidence < 1 and 0 < limit < 1 and 0 < null < 1):
            raise ValueError
    except (KeyError, ValueError, TypeError, OverflowError):
        return out + ["calibration: invalid confidence, registered error limit or per-arm null rate"]
    rows = record.get("worlds")
    if not isinstance(rows, list) or not rows:
        return out + ["calibration: no whole-procedure worlds"]
    quantities = record.get("observation_quantities")
    allowed = {BALANCE_ELASTICITY, SERVICE_RATIO, LOG_SERVICE_RATIO}
    if (not isinstance(quantities, list) or not quantities
            or any(not isinstance(q, str) or q not in allowed for q in quantities)
            or len(set(quantities)) != len(quantities)):
        return out + ["calibration: invalid observation quantities"]
    if quantity is not None and quantity not in quantities:
        out.append("calibration: this observation quantity was not calibrated")
    expected = {(q, w) for q in quantities for w in REQUIRED_WORLDS}
    seen = set()
    # Simultaneous one-sided exact bounds over both error directions and every declared world.
    # No estimated success rate, rounded percentage or zero observed errors substitutes for this.
    tail = (1 - confidence) / (2 * len(rows))
    for row in rows:
        if not isinstance(row, dict):
            out.append("calibration: malformed world record")
            continue
        key = (row.get("quantity"), row.get("world"))
        if any(not isinstance(v, str) for v in key) or key in seen:
            out.append("calibration: invalid or duplicate world identity")
            continue
        seen.add(key)
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("world_spec_sha256", ""))):
            out.append("calibration: missing frozen world specification digest")
        n = row.get("independent_runs")
        if type(n) is not int or n <= 0:
            out.append("calibration: independent_runs must be a positive integer")
            continue
        for field in ("false_supports", "false_refutations"):
            k = row.get(field)
            if type(k) is not int or not 0 <= k <= n:
                out.append("calibration: invalid " + field)
                continue
            upper = 1.0 if k == n else float(beta.ppf(1 - tail, k + 1, n - k))
            if not math.isfinite(upper) or upper > limit:
                out.append("calibration: %s %s %s upper bound %.6g exceeds registered limit %.6g"
                           % (*key, field, upper, limit))
    if not expected.issubset(seen) or any(q not in quantities for q, _ in seen):
        out.append("calibration: missing required worlds or undeclared observation quantity")
    return out
