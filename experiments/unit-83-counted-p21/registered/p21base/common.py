from __future__ import annotations
import hashlib, json, math, os, tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

class GateError(RuntimeError):
    """A scientific, safety or budget gate refused the operation."""

def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def file_sha(path: str | Path) -> str:
    return digest(Path(path).read_bytes())

def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def load(path: str | Path) -> Any:
    return json.loads(Path(path).read_text("utf-8"))

def save(path: str | Path, value: Any, *, exclusive: bool = False) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8") + b"\n"
    if exclusive:
        with path.open("xb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        return
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

def checked_fraction(x: Any) -> float:
    if not finite(x) or not 0 <= x <= 1: raise GateError("Invalid fraction")
    return float(x)
