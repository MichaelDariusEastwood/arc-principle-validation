"""The canonical decision regions, exported in the format the coordination repository's harness reads.

One table, two readers. The verdict engine (verdicts.py) is the source; the synthetic-interval harness
is the table checker and command line. Region names are the engine's own verdict strings so that the harness's axis verdict and
the engine's function return the same string on the same interval; "aggregate" maps the axis verdict to
the proposition-level label the registration scores; "requires" names the flags a verdict needs (the
fresh-data repetition). Every axis is "open": a boundary contact is unscored, which the harness reports
as INCONCLUSIVE, matching the engine's strict-clearance rule.

The boundary convention was ruled on 5 September 2026 (ruling 17): open everywhere. The numbers follow
ruling 10 of the same day. The applying session still confirms each number against the frozen
registration text; where the text differs, the text governs and this table is corrected in the same
commit.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from . import verdicts as V

REGIONS: List[Dict[str, object]] = [
    {"name": "P8.exponent", "boundary": "open",
     "regions": {V.ABOVE_NULL: [0.60, None], V.BELOW_NULL: [None, 0.40], V.EQUIVALENT_TO_NULL: [0.40, 0.60]},
     "aggregate": {V.ABOVE_NULL: V.SUPPORTED, V.BELOW_NULL: V.REFUTED, V.EQUIVALENT_TO_NULL: V.REFUTED}},
    {"name": "P8.unity", "boundary": "open",
     "regions": {V.CLEARS_UNITY: [1.0, None], V.DOES_NOT_CLEAR_UNITY: [None, 1.0]},
     "aggregate": {}},
    {"name": "P22.checkable_minus_judged", "boundary": "open",
     "regions": {V.CHECKABLE_HIGHER: [0.10, None], V.JUDGED_HIGHER: [None, -0.10], V.PRACTICALLY_EQUAL: [-0.10, 0.10]},
     "aggregate": {V.CHECKABLE_HIGHER: V.SUPPORTED, V.JUDGED_HIGHER: V.REFUTED, V.PRACTICALLY_EQUAL: V.REFUTED},
     "requires": {V.CHECKABLE_HIGHER: ["replicated"], V.JUDGED_HIGHER: ["replicated"], V.PRACTICALLY_EQUAL: ["replicated"]}},
    {"name": "P19.alignment_elasticity", "boundary": "open",
     "regions": {V.MATERIALLY_POSITIVE: [0.10, None], V.MATERIALLY_NEGATIVE: [None, -0.10], V.EQUIVALENT_TO_ZERO: [-0.10, 0.10]},
     "aggregate": {V.MATERIALLY_POSITIVE: V.REFUTED, V.MATERIALLY_NEGATIVE: V.REFUTED, V.EQUIVALENT_TO_ZERO: V.SUPPORTED}},
    {"name": "P19.legacy_half", "boundary": "open",
     "regions": {V.WHOLLY_ABOVE_ONE_HALF: [0.5, None], V.WHOLLY_BELOW_ONE_HALF: [None, 0.5]},
     "aggregate": {}},
    {"name": "P12.change_in_leverage", "boundary": "open",
     "regions": {"ABOVE": [0.10, None], "BELOW": [None, -0.10], "EQUIVALENT": [-0.10, 0.10]},
     "aggregate": {"ABOVE": V.SUPPORTED, "BELOW": V.SUPPORTED, "EQUIVALENT": V.REFUTED}},
    {"name": "P10.fraction_not_surviving", "boundary": "open",
     "regions": {V.SUPPORTED: [0.10, None], V.REFUTED: [None, 0.10]},
     "aggregate": {}},
    {"name": "P15.fraction_retained", "boundary": "open",
     "regions": {V.SUPPORTED: [None, 0.50], V.REFUTED: [0.50, None]},
     "aggregate": {}},
    {"name": "P17.pair_log_ratio", "boundary": "open",
     "regions": {V.INDEPENDENT: [-0.20, 0.20], V.CORRELATED: [0.20, None], V.ANTI_CORRELATED: [None, -0.20]},
     "aggregate": {V.INDEPENDENT: V.SUPPORTED, V.CORRELATED: V.REFUTED, V.ANTI_CORRELATED: V.REFUTED}},
    {"name": "P5.predicted_minus_fitted", "boundary": "open",
     "regions": {V.SUPPORTED: [-0.10, 0.10], "ABOVE": [0.10, None], "BELOW": [None, -0.10]},
     "aggregate": {"ABOVE": V.REFUTED, "BELOW": V.REFUTED}},
    {"name": "P11.difference", "boundary": "open",
     "regions": {V.IDENTICAL_WITHIN_MARGIN: [-0.10, 0.10], "DISTINCT_ABOVE": [0.10, None], "DISTINCT_BELOW": [None, -0.10]},
     "aggregate": {V.IDENTICAL_WITHIN_MARGIN: V.REFUTED, "DISTINCT_ABOVE": V.SUPPORTED, "DISTINCT_BELOW": V.SUPPORTED}},
    {"name": "P6.gamma_same_class", "boundary": "open",
     "regions": {V.REFUTED: [0.50, None], V.NO_COUNTEREXAMPLE: [None, 0.50]},
     "aggregate": {},
     "requires": {V.REFUTED: ["replicated"]}},
    {"name": "P14.blinded_minus_unblinded", "boundary": "open",
     "regions": {V.SUPPORTED: [None, 0.0], V.REFUTED: [0.0, None]},
     "aggregate": {}},
]


def axis(name: str) -> Dict[str, object]:
    for a in REGIONS:
        if a["name"] == name:
            return a
    raise KeyError(name)


def to_json(path: Optional[str] = None, indent: int = 1) -> str:
    text = json.dumps(REGIONS, indent=indent)
    if path:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return text
