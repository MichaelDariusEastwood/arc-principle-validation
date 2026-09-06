from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import torch

from models.policy_program import PolicyGenome, compile_policy
from training.verifier import verify_policy_source, verify_model_state, verify_determinism


@dataclass
class CandidateState:
    optimizer_state: dict
    policy_genome: PolicyGenome
    policy_source: str


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


class LedgerWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")


class ChangeController:
    def __init__(self, *, condition: str, delta_c_min: float, delta_cs_min: float, s_min: float, accept_epsilon: float, max_param_norm: float, catastrophic_floor: float, determinism_checks: int, policy_mutation_scale: float, meta_mutation_scale: float, allow_activation_mutation: bool, allow_feature_mutation: bool, rng: random.Random):
        self.condition = condition
        self.delta_c_min = delta_c_min
        self.delta_cs_min = delta_cs_min
        self.s_min = s_min
        self.accept_epsilon = accept_epsilon
        self.max_param_norm = max_param_norm
        self.catastrophic_floor = catastrophic_floor
        self.determinism_checks = determinism_checks
        self.policy_mutation_scale = policy_mutation_scale
        self.meta_mutation_scale = meta_mutation_scale
        self.allow_activation_mutation = allow_activation_mutation
        self.allow_feature_mutation = allow_feature_mutation
        self.rng = rng

    def mutate_candidate(self, optimizer_state: dict, policy_genome: PolicyGenome) -> CandidateState:
        cand_opt = clone_state_dict(optimizer_state)
        for tensor in cand_opt.values():
            tensor.add_(torch.randn_like(tensor) * self.meta_mutation_scale)
        cand_genome = policy_genome.mutate(
            self.rng,
            scale=self.policy_mutation_scale,
            allow_activation=self.allow_activation_mutation,
            allow_feature=self.allow_feature_mutation,
        )
        source = cand_genome.render_source()
        return CandidateState(optimizer_state=cand_opt, policy_genome=cand_genome, policy_source=source)

    def accept(self, parent_metrics: Dict[str, float], cand_metrics: Dict[str, float]) -> tuple[bool, str]:
        if self.condition == "static":
            return False, "static_never_mutates"
        if self.condition == "drag_control":
            return False, "drag_control_no_mutation"

        d_c = cand_metrics["capability"] - parent_metrics["capability"]
        d_cs = cand_metrics["combined"] - parent_metrics["combined"]
        if self.condition == "babylon":
            return (d_c > self.delta_c_min + self.accept_epsilon), ("accept_capability_only" if d_c > self.delta_c_min + self.accept_epsilon else "reject_no_capability_gain")

        # eden
        if cand_metrics["safety"] < self.s_min:
            return False, "reject_safety_floor"
        if d_c < self.delta_c_min - self.accept_epsilon:
            return False, "reject_capability_drop"
        if d_cs <= self.delta_cs_min + self.accept_epsilon:
            return False, "reject_no_entangled_gain"
        return True, "accept_entangled_gain"

    def verify_candidate(self, model: torch.nn.Module, policy_source: str) -> tuple[bool, str, Dict[str, float]]:
        r1 = verify_policy_source(policy_source)
        if not r1.ok:
            return False, r1.reason, r1.details
        policy_fn = compile_policy(policy_source)
        r2 = verify_determinism(policy_fn, checks=self.determinism_checks)
        if not r2.ok:
            return False, r2.reason, r2.details
        r3 = verify_model_state(model, max_param_norm=self.max_param_norm, catastrophic_floor=self.catastrophic_floor)
        if not r3.ok:
            return False, r3.reason, r3.details
        details = {}
        details.update(r1.details)
        details.update(r2.details)
        details.update(r3.details)
        return True, "ok", details
