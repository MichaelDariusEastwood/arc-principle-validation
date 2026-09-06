"""The revision loop: rounds, retention control, checkpoints, and a ladder reading at each.

A trajectory is a sequence of rounds. At each round the system sees its task and a retained fraction
of its own prior output, revises, and the ladder is read. Retention is a uniform random subsample and
never a top-ranked one, because retaining the best of a set confounds how much was retained with how
good it was. A checkpoint is a full copy of the artefact at a depth, and it is what the held-out
continuation is generated from after the seal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adapters import ModelAdapter
from .ladder import Ladder, LadderResult
from .sampling import ReadUncertainty, read_uncertainty


def retain(artefact: Dict[str, Any], fraction: float, rng: np.random.Generator) -> Dict[str, Any]:
    """The context the next round may see. For text, a uniform random subsample of lines; for the
    mock, the fraction itself, which the mock's growth equation consumes directly."""
    if not np.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("retention fraction must be finite and in [0, 1]")
    out = {"fraction": float(fraction)}
    text = artefact.get("text")
    if isinstance(text, str) and text:
        lines = text.split("\n")
        keep = int(round(len(lines) * fraction))
        idx = sorted(rng.choice(len(lines), size=min(keep, len(lines)), replace=False))
        out["text"] = "\n".join(lines[i] for i in idx)
    return out


@dataclass
class Checkpoint:
    depth: int
    artefact: Dict[str, Any]
    reading: LadderResult
    # WHAT THIS READING'S OWN SAMPLING ERROR WAS (finding A7). The reading travelled without it, so
    # P5's calibration window entered the sealed prediction's interval as though every score in it
    # were exact, and the starting state, which is one of those scores, never moved in the interval
    # at all. The uncertainty is computed where the reads happen, under the ladder's declared
    # sampling unit, because that is the only place the individual reads still exist: the checkpoint
    # keeps their average. It is optional so that a Checkpoint built by an older caller, or by a test
    # that has no ladder in hand, is still a Checkpoint.
    read_uncertainty: Optional[ReadUncertainty] = None

    @property
    def read_sd(self) -> float:
        """The sampling error of this checkpoint's score, or zero where none was recorded. Zero is
        the fail-closed reading for a MISSING uncertainty in the same sense it is for a deterministic
        ladder: it claims no precision the reading did not have, and a caller that needs to know
        whether an uncertainty was recorded at all asks for the object."""
        u = self.read_uncertainty
        return float(u.sd) if u is not None and np.isfinite(u.sd) else 0.0


@dataclass
class Trajectory:
    system: str
    checkpoints: List[Checkpoint] = field(default_factory=list)

    def depths(self) -> List[int]:
        return [c.depth for c in self.checkpoints]

    def scores(self) -> List[float]:
        return [c.reading.score for c in self.checkpoints]

    def read_sds(self) -> List[float]:
        """The sampling error of each checkpoint's score, in the order `scores` returns them."""
        return [c.read_sd for c in self.checkpoints]

    def at(self, depth: int) -> Optional[Checkpoint]:
        for c in self.checkpoints:
            if c.depth == depth:
                return c
        return None


def run_round(adapter: ModelAdapter, artefact: Dict[str, Any], fraction: float, task: str,
              rng: np.random.Generator, control: Optional[str] = None) -> Dict[str, Any]:
    """One round. `control="unusable_retention"` marks a registered negative-control cell: the retained
    material is handed over in a form the round cannot use (for text, the lines are shuffled across
    an unrelated artefact by the caller; for the mock, the adapter honours the flag)."""
    retained = retain(artefact, fraction, rng)
    if control:
        retained["control"] = control
    return adapter.revise(artefact, retained, task, rng)


def _read(ladder: Ladder, art: Dict[str, Any], rng: np.random.Generator, reads: int,
          context: Optional[str] = None) -> Tuple[LadderResult, ReadUncertainty]:
    """One checkpoint reading, averaged over `reads` ladder passes. The calibrated rate is the binding
    precision of the whole P5 comparison, and it is read from a handful of early checkpoints, so those
    are read more times than the rest.

    Each individual pass is recorded when the ladder is keeping a read log, not the average of them:
    the evidence bundle carries what was counted, and the averaging is an analysis step a later reader
    is entitled to redo."""
    rs = []
    for _ in range(max(1, int(reads))):
        r = ladder.score(art, rng)
        # The artefact goes with the record so that the count can be tied to the thing it counted:
        # a read log holding a subset size and a pass count alone cannot be traced to any reading.
        ladder.record_read(r, context, art)
        rs.append(r)
    if len(rs) == 1:
        return rs[0], read_uncertainty(ladder, rs)
    # The pool the reads were drawn from travels with the average, because the finite population
    # correction needs the denominator the reads actually had and a ladder reconfigured later would
    # supply another one. The average lost it before finding A7 went looking for it.
    #
    # AND THE AVERAGE IS NOT ROUNDED BACK TO A WHOLE COUNT. Rounding a mean of several reads to the
    # nearest integer throws away exactly the information the extra reads were paid for, so a mean
    # of 17.4 and a mean of 17.6 became the same reading. The result field is a float for this
    # reason, and every consumer of it already treats it as one.
    return (LadderResult(passes=float(np.mean([r.passes for r in rs])), n_items=rs[0].n_items,
                         at_ceiling=any(r.at_ceiling for r in rs),
                         population_size=rs[0].population_size),
            read_uncertainty(ladder, rs))


def run_trajectory(adapter: ModelAdapter, ladder: Ladder, start: Dict[str, Any], task: str,
                   depths: Sequence[int], fraction: float, rng: np.random.Generator,
                   system: str = "system", start_depth: int = 0, reads: int = 1) -> Trajectory:
    """Run from `start` (a checkpointed artefact at `start_depth`) through every round up to
    max(depths), reading the ladder at each depth in `depths`, `reads` times per checkpoint."""
    traj = Trajectory(system=system)
    art = dict(start)
    if start_depth in depths:
        res, unc = _read(ladder, art, rng, reads, "%s@%d" % (system, start_depth))
        traj.checkpoints.append(Checkpoint(start_depth, dict(art), res, unc))
    for d in range(start_depth + 1, max(depths) + 1):
        art = run_round(adapter, art, fraction, task, rng)
        if d in depths:
            res, unc = _read(ladder, art, rng, reads, "%s@%d" % (system, d))
            traj.checkpoints.append(Checkpoint(d, dict(art), res, unc))
    return traj


def loglog_slope(depths: Sequence[float], scores: Sequence[float]) -> float:
    """The fitted growth exponent: OLS slope of log score on log depth over estimable points."""
    x = np.log(np.asarray(depths, float)); y = np.log(np.maximum(np.asarray(scores, float), 1e-9))
    ok = np.isfinite(x) & np.isfinite(y) & (np.asarray(depths) > 0)
    if ok.sum() < 2:
        return float("nan")
    return float(np.polyfit(x[ok], y[ok], 1)[0])
