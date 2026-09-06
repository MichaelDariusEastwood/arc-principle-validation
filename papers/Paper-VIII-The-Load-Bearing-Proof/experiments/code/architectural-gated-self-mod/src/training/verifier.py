from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable

import torch

from models.policy_program import compile_policy


@dataclass
class VerificationResult:
    ok: bool
    reason: str
    details: Dict[str, float]


def verify_policy_source(source: str) -> VerificationResult:
    try:
        fn = compile_policy(source)
        sample = fn({
            "loss": 0.5,
            "loss_delta": 0.1,
            "grad_norm": 1.0,
            "param_norm": 1.5,
            "retention": 0.8,
            "task_progress": 0.5,
            "improvement": 0.1,
        })
    except Exception as exc:
        return VerificationResult(False, f"policy_compile_error: {exc}", {})
    required = {"lr_scale", "grad_clip", "noise_scale"}
    if not isinstance(sample, dict) or set(sample.keys()) != required:
        return VerificationResult(False, "policy_bad_return_shape", {})
    for k, v in sample.items():
        if not isinstance(v, (int, float)):
            return VerificationResult(False, f"policy_non_numeric_{k}", {})
        if not (v == v):
            return VerificationResult(False, f"policy_nan_{k}", {})
    return VerificationResult(True, "ok", {"lr_scale": float(sample["lr_scale"])})


def verify_model_state(model: torch.nn.Module, *, max_param_norm: float, catastrophic_floor: float) -> VerificationResult:
    details: Dict[str, float] = {}
    total_norm = 0.0
    for name, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            return VerificationResult(False, f"nan_or_inf_in_{name}", details)
        total_norm += float(torch.norm(p.detach()).item())
    details["param_norm"] = total_norm
    if total_norm > max_param_norm:
        return VerificationResult(False, "param_norm_exceeded", details)
    return VerificationResult(True, "ok", details)


def verify_determinism(policy_fn: Callable[[Dict[str, float]], Dict[str, float]], checks: int = 2) -> VerificationResult:
    probe = {
        "loss": 0.25,
        "loss_delta": -0.05,
        "grad_norm": 0.8,
        "param_norm": 3.0,
        "retention": 0.9,
        "task_progress": 0.7,
        "improvement": 0.05,
    }
    ref = policy_fn(probe)
    for _ in range(checks - 1):
        out = policy_fn(probe)
        if out != ref:
            return VerificationResult(False, "policy_non_deterministic", {})
    return VerificationResult(True, "ok", {})
