from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List

import torch

from tasks.suite import Task, evaluate_all_tasks, capability_score, retention_ratio
from models.learned_optimizer import LearnedOptimizer, UpdateControls


@dataclass
class EpisodeMetrics:
    capability: float
    safety: float
    combined: float
    drag_cost: float
    compute_steps: int
    post_train_scores: List[float]
    final_scores: List[float]
    checkpoint_scores: List[float]
    task_rows: List[Dict[str, float]]
    elapsed_seconds: float


def _features(loss: float, loss_delta: float, grad_norm: float, param_norm: float, retention: float, task_progress: float, improvement: float, device: torch.device) -> torch.Tensor:
    return torch.tensor([
        float(loss), float(loss_delta), float(grad_norm), float(param_norm),
        float(retention), float(task_progress), float(improvement), 1.0,
    ], dtype=torch.float32, device=device)


def _default_controls() -> UpdateControls:
    return UpdateControls(lr_scale=1.0, momentum=0.9, weight_decay=1.0e-4, grad_clip=5.0, noise_scale=0.0)


def rollout_episode(
    model: torch.nn.Module,
    learned_opt: LearnedOptimizer | None,
    policy_fn,
    tasks: List[Task],
    *,
    base_lr: float,
    base_momentum: float,
    base_weight_decay: float,
    base_grad_clip: float,
    steps_per_task: int,
    batch_size: int,
    lambda_drag: float,
    drag_enabled: bool,
    device: torch.device,
) -> EpisodeMetrics:
    start = time.time()
    velocity = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    if learned_opt is not None:
        learned_opt.reset_state(batch_size=1, device=device)

    post_train_scores: List[float] = []
    checkpoint_scores: List[float] = []
    current_retention = 1.0
    prev_loss = 0.0
    compute_steps = 0
    drag_cost = 0.0
    task_rows: List[Dict[str, float]] = []

    for task_index, task in enumerate(tasks):
        loader_idx = 0
        before_scores = [row["score"] for row in evaluate_all_tasks(model, tasks[:task_index])] if task_index > 0 else []
        if before_scores:
            checkpoint_scores = before_scores
        train_x, train_y = task.train_x, task.train_y

        for step in range(steps_per_task):
            model.train()
            batch_start = (loader_idx * batch_size) % train_x.size(0)
            batch_end = batch_start + batch_size
            x = train_x[batch_start:batch_end]
            y = train_y[batch_start:batch_end]
            if x.size(0) == 0:
                loader_idx = 0
                continue
            loader_idx += 1
            pred = model(x)
            loss = torch.mean((pred - y) ** 2)
            model.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm_sq += float(torch.sum(p.grad.detach() ** 2).item())
            grad_norm = grad_norm_sq ** 0.5
            param_norm = sum(float(torch.norm(p.detach()).item()) for p in model.parameters())
            loss_value = float(loss.detach().item())
            loss_delta = prev_loss - loss_value if step > 0 else 0.0
            feats = _features(loss_value, loss_delta, grad_norm, param_norm, current_retention, (task_index + 1) / len(tasks), max(0.0, loss_delta), device)

            opt_controls = learned_opt(feats) if learned_opt is not None else _default_controls()
            prog_controls = policy_fn({
                "loss": loss_value,
                "loss_delta": loss_delta,
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "retention": current_retention,
                "task_progress": (task_index + 1) / len(tasks),
                "improvement": max(0.0, loss_delta),
            })

            lr = base_lr * opt_controls.lr_scale * float(prog_controls["lr_scale"])
            momentum = max(0.0, min(0.99, opt_controls.momentum))
            weight_decay = max(0.0, opt_controls.weight_decay + base_weight_decay)
            grad_clip = min(base_grad_clip, opt_controls.grad_clip, float(prog_controls["grad_clip"]))
            noise_scale = max(0.0, opt_controls.noise_scale + float(prog_controls["noise_scale"]))

            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            with torch.no_grad():
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g = p.grad + weight_decay * p
                    velocity[name].mul_(momentum).add_(g, alpha=-lr)
                    if noise_scale > 0:
                        velocity[name].add_(torch.randn_like(p) * noise_scale)
                    p.add_(velocity[name])
            prev_loss = loss_value
            compute_steps += 1

        current_score = evaluate_all_tasks(model, [task])[0]["score"]
        post_train_scores.append(current_score)

        after_scores = [row["score"] for row in evaluate_all_tasks(model, tasks[:task_index])] if task_index > 0 else []
        current_retention = retention_ratio(after_scores, checkpoint_scores)
        if drag_enabled and task_index > 0:
            # verification tax: re-evaluate all previously seen tasks twice
            for _ in range(2):
                _ = evaluate_all_tasks(model, tasks[:task_index])
                drag_cost += float(task_index)

        task_rows.append({
            "task_index": task_index,
            "task_name": task.name,
            "post_train_score": current_score,
            "retention": current_retention,
        })

    final_rows = evaluate_all_tasks(model, tasks)
    final_scores = [row["score"] for row in final_rows]
    capability = capability_score(post_train_scores)
    safety = current_retention if len(tasks) > 1 else 1.0
    combined = capability * safety - (lambda_drag * drag_cost)
    elapsed = time.time() - start
    return EpisodeMetrics(
        capability=float(capability),
        safety=float(safety),
        combined=float(combined),
        drag_cost=float(drag_cost),
        compute_steps=int(compute_steps),
        post_train_scores=[float(x) for x in post_train_scores],
        final_scores=[float(x) for x in final_scores],
        checkpoint_scores=[float(x) for x in checkpoint_scores],
        task_rows=task_rows,
        elapsed_seconds=float(elapsed),
    )
