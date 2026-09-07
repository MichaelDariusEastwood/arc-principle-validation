from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

from models.optimizee import MLPOptimizee
from models.learned_optimizer import LearnedOptimizer
from models.policy_program import PolicyGenome, compile_policy
from training.inner_loop import rollout_episode, EpisodeMetrics
from training.controller import ChangeController, CandidateState, clone_state_dict
from tasks.suite import Task


@dataclass
class AgentState:
    optimizer_state: dict
    policy_genome: PolicyGenome
    policy_source: str

    def clone(self) -> "AgentState":
        return AgentState(
            optimizer_state=clone_state_dict(self.optimizer_state),
            policy_genome=copy.deepcopy(self.policy_genome),
            policy_source=str(self.policy_source),
        )


def initial_agent_state(learned_opt: LearnedOptimizer) -> AgentState:
    genome = PolicyGenome()
    return AgentState(
        optimizer_state=learned_opt.clone_state(),
        policy_genome=genome,
        policy_source=genome.render_source(),
    )


def build_model(model_cfg: Dict, device: torch.device) -> MLPOptimizee:
    return MLPOptimizee(
        input_dim=int(model_cfg["input_dim"]),
        hidden_sizes=list(model_cfg["hidden_sizes"]),
        output_dim=int(model_cfg["output_dim"]),
        init_std=float(model_cfg.get("init_std", 0.05)),
    ).to(device)


def build_optimizer(opt_cfg: Dict, device: torch.device) -> LearnedOptimizer:
    return LearnedOptimizer(
        feature_dim=int(opt_cfg.get("feature_dim", 8)),
        hidden_size=int(opt_cfg.get("hidden_size", 32)),
    ).to(device)


def evaluate_state(
    *,
    agent_state: AgentState,
    tasks: List[Task],
    model_cfg: Dict,
    opt_cfg: Dict,
    lambda_drag: float,
    drag_enabled: bool,
    batch_size: int,
    device: torch.device,
) -> tuple[EpisodeMetrics, MLPOptimizee]:
    model = build_model(model_cfg, device)
    learned_opt = build_optimizer(opt_cfg, device)
    learned_opt.load_cloned_state(agent_state.optimizer_state)
    policy_fn = compile_policy(agent_state.policy_source)
    metrics = rollout_episode(
        model,
        learned_opt,
        policy_fn,
        tasks,
        base_lr=float(opt_cfg["base_lr"]),
        base_momentum=float(opt_cfg["momentum"]),
        base_weight_decay=float(opt_cfg["weight_decay"]),
        base_grad_clip=float(opt_cfg["grad_clip"]),
        steps_per_task=int(opt_cfg["steps_per_task"]),
        batch_size=int(batch_size),
        lambda_drag=float(lambda_drag),
        drag_enabled=bool(drag_enabled),
        device=device,
    )
    return metrics, model


def run_iteration(
    *,
    parent_state: AgentState,
    controller: ChangeController,
    tasks: List[Task],
    model_cfg: Dict,
    opt_cfg: Dict,
    lambda_drag: float,
    batch_size: int,
    device: torch.device,
) -> tuple[AgentState, Dict]:
    parent_metrics, parent_model = evaluate_state(
        agent_state=parent_state,
        tasks=tasks,
        model_cfg=model_cfg,
        opt_cfg=opt_cfg,
        lambda_drag=lambda_drag if controller.condition in {"eden", "drag_control"} else 0.0,
        drag_enabled=controller.condition in {"eden", "drag_control"},
        batch_size=batch_size,
        device=device,
    )

    log_row = {
        "parent": parent_metrics,
        "accepted": False,
        "reason": "no_mutation",
        "condition": controller.condition,
        "policy_source_before": parent_state.policy_source,
    }

    if controller.condition in {"static", "drag_control"}:
        log_row["policy_source_after"] = parent_state.policy_source
        return parent_state.clone(), log_row

    candidate = controller.mutate_candidate(parent_state.optimizer_state, parent_state.policy_genome)
    cand_state = AgentState(
        optimizer_state=candidate.optimizer_state,
        policy_genome=candidate.policy_genome,
        policy_source=candidate.policy_source,
    )
    cand_metrics, cand_model = evaluate_state(
        agent_state=cand_state,
        tasks=tasks,
        model_cfg=model_cfg,
        opt_cfg=opt_cfg,
        lambda_drag=lambda_drag if controller.condition == "eden" else 0.0,
        drag_enabled=controller.condition == "eden",
        batch_size=batch_size,
        device=device,
    )
    ok, verify_reason, verify_details = controller.verify_candidate(cand_model, candidate.policy_source)
    if not ok:
        log_row.update({
            "candidate": cand_metrics,
            "accepted": False,
            "reason": verify_reason,
            "verify_details": verify_details,
            "policy_source_after": parent_state.policy_source,
        })
        return parent_state.clone(), log_row

    accept, reason = controller.accept(
        {"capability": parent_metrics.capability, "safety": parent_metrics.safety, "combined": parent_metrics.combined},
        {"capability": cand_metrics.capability, "safety": cand_metrics.safety, "combined": cand_metrics.combined},
    )
    log_row.update({
        "candidate": cand_metrics,
        "accepted": bool(accept),
        "reason": reason,
        "verify_details": verify_details,
        "policy_source_after": candidate.policy_source if accept else parent_state.policy_source,
    })
    return (cand_state if accept else parent_state.clone()), log_row
