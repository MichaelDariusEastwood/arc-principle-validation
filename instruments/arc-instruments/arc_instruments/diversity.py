"""Successive-output diversity as a diagnostic, and delivery as the validity gate for sequential arms.

Second edition (5 September 2026). The first edition marked a sequential arm invalid when its outputs
copied their predecessors, reasoning that reading outputs rather than scores made the gate
outcome-independent. That reasoning was wrong: outputs are post-intervention observations, and a
correctly delivered recursive loop can stagnate, converge or repeat an already optimal answer. Excluding
such runs selects on the behaviour under study. Validity is now decided on delivery alone (the previous
artefact and revision-policy state reached the next round, as the execution trace records), copying is
reported as a diagnostic beside the outcome, and an arm with no eligible runs is NOT EVALUABLE rather
than valid by default.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set

NO_OBSERVATIONS = "NO OBSERVATIONS"


def char_ngrams(text: str, n: int = 4) -> Set[str]:
    t = " ".join(text.split()).lower()
    if len(t) < n:
        return {t} if t else set()
    return {t[i:i + n] for i in range(len(t) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def successive_similarity(outputs: Sequence[str], n: int = 4) -> List[float]:
    """Similarity of output k to output k-1, for k = 1..len-1."""
    grams = [char_ngrams(o, n) for o in outputs]
    return [jaccard(grams[i - 1], grams[i]) for i in range(1, len(grams))]


def copy_rate(outputs: Sequence[str], threshold: float = 0.95, n: int = 4) -> float:
    sims = successive_similarity(outputs, n)
    if not sims:
        return 0.0
    return sum(1 for s in sims if s >= threshold) / len(sims)


def diagnostics(arms: Dict[str, Sequence[Sequence[str]]], threshold: float = 0.95, n: int = 4) -> Dict[str, Dict[str, object]]:
    """Per-arm copy diagnostics: mean copy rate and the number of runs read. Never a validity decision."""
    out: Dict[str, Dict[str, object]] = {}
    for arm, runs in arms.items():
        rates = [copy_rate(seq, threshold, n) for seq in runs if len(seq) >= 2]
        out[arm] = {"mean_copy_rate": (sum(rates) / len(rates) if rates else None), "n_runs": len(rates),
                    "state": (NO_OBSERVATIONS if not rates else "observed")}
    return out


def delivery_gate(delivery: Dict[str, Sequence[bool]]) -> Dict[str, Dict[str, object]]:
    """arms: arm name -> per-run flags from the execution trace that the prior artefact and policy
    state reached the next round. An arm is valid where every run delivered its state; a run that did not
    is an undelivered run and is reported as such; an arm with no runs is NOT EVALUABLE. No output, score
    or novelty enters this decision."""
    out: Dict[str, Dict[str, object]] = {}
    for arm, flags in delivery.items():
        flags = list(flags)
        if not flags:
            out[arm] = {"valid": None, "state": NO_OBSERVATIONS, "n_runs": 0, "undelivered_runs": 0}
            continue
        undelivered = sum(1 for f in flags if not f)
        out[arm] = {"valid": bool(undelivered == 0), "state": "delivered" if undelivered == 0 else "undelivered runs present",
                    "n_runs": len(flags), "undelivered_runs": undelivered}
    return out


def validity_gate(*args, **kwargs):
    """Retired in the second edition: copying is a diagnostic (diagnostics) and validity is delivery
    (delivery_gate). Calling this raises so that no caller silently keeps the outcome-dependent gate."""
    raise RuntimeError("validity_gate was retired on 5 September 2026: use diagnostics() for copy rates and delivery_gate() for validity")
