"""Sealed predictions and held-out partitions.

A prediction is sealed when its canonical bytes and their hash exist before the data that could decide
it are opened, and a later reveal can be checked against that hash by anyone. This module produces the
sealed record (canonical JSON, SHA-256, UTC time, the hash of the frozen specification it was made
under) and verifies a reveal against it; it partitions identifiers into calibration and held-out sets by
a seeded rule and seals the held-out list so that the split cannot move after the outcomes are seen.
The independent timestamp (an OpenTimestamps proof of the sealed record) and the custody of the
held-out data by a party outside the programme are acts outside this code; this module only makes
them checkable.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_of(payload: Any) -> str:
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def seal(payload: Dict[str, Any], spec_sha256: str, sealed_by: str, note: str = "") -> Dict[str, Any]:
    """Return a sealed record for a prediction payload made under a frozen specification."""
    return {
        "sealed_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sealed_by": sealed_by,
        "spec_sha256": spec_sha256,
        "payload_sha256": sha256_of(payload),
        "payload_bytes": len(canonical_bytes(payload)),
        "note": note,
        "independent_timestamp": None,
    }


def verify(sealed_record: Dict[str, Any], revealed_payload: Dict[str, Any], spec_sha256: Optional[str] = None) -> Dict[str, Any]:
    """Check a revealed payload against its seal, and the specification hash where supplied."""
    got = sha256_of(revealed_payload)
    ok_payload = got == sealed_record.get("payload_sha256")
    ok_spec = True if spec_sha256 is None else (spec_sha256 == sealed_record.get("spec_sha256"))
    return {"payload_matches": ok_payload, "spec_matches": ok_spec, "verified": bool(ok_payload and ok_spec),
            "revealed_sha256": got, "sealed_sha256": sealed_record.get("payload_sha256"),
            "independent_timestamp_present": sealed_record.get("independent_timestamp") is not None}


def holdout_partition(identifiers: Sequence[str], holdout_fraction: float, seed: int) -> Dict[str, Any]:
    """Seeded split of identifiers into calibration and held-out sets, with the held-out list sealed."""
    ids = sorted(set(map(str, identifiers)))
    if not ids:
        raise ValueError("no identifiers to partition")
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout fraction must lie strictly between zero and one")
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(len(ids)))
    n_hold = max(1, int(round(holdout_fraction * len(ids))))
    hold = sorted(ids[i] for i in perm[:n_hold])
    cal = sorted(ids[i] for i in perm[n_hold:])
    return {"seed": seed, "holdout_fraction": holdout_fraction, "calibration": cal, "holdout": hold,
            "holdout_sha256": sha256_of(hold), "n_calibration": len(cal), "n_holdout": len(hold)}


def write_sealed(path: str, record: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(record, indent=1, sort_keys=True) + "\n")


def read_sealed(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
