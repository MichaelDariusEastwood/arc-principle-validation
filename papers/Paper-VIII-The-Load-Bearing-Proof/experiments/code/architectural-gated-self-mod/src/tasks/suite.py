from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import torch


@dataclass
class Task:
    name: str
    train_x: torch.Tensor
    train_y: torch.Tensor
    val_x: torch.Tensor
    val_y: torch.Tensor
    conflict_with: str | None = None


def _base_functions() -> List[tuple[str, callable]]:
    return [
        ("sin(x)", lambda x: np.sin(x)),
        ("cos(x)", lambda x: np.cos(x)),
        ("sin(2x)", lambda x: np.sin(2 * x)),
        ("cos(2x)", lambda x: np.cos(2 * x)),
        ("x/3", lambda x: x / 3.0),
        ("tanh(x)", lambda x: np.tanh(x)),
        ("sin(x)*cos(x)", lambda x: np.sin(x) * np.cos(x)),
        ("x*sin(x)/3", lambda x: (x * np.sin(x)) / 3.0),
    ]


def make_adversarial_regression_tasks(
    *,
    num_tasks: int,
    train_samples: int,
    val_samples: int,
    noise: float,
    seed: int,
    device: torch.device,
) -> List[Task]:
    """Create a curriculum where neighboring tasks intentionally conflict.

    Even-index tasks are positive versions of a base function and odd-index tasks
    are the negation, which induces forgetting pressure in small shared models.
    """
    rng = np.random.default_rng(seed)
    functions = _base_functions()
    tasks: List[Task] = []
    func_idx = 0

    for task_idx in range(num_tasks):
        base_name, fn = functions[func_idx % len(functions)]
        sign = 1.0 if task_idx % 2 == 0 else -1.0
        name = f"{'+' if sign > 0 else '-'}{base_name}"
        conflict_with = None if sign > 0 else f"+{base_name}"

        x_train = np.linspace(-3, 3, train_samples, dtype=np.float32).reshape(-1, 1)
        x_val = np.linspace(-3, 3, val_samples, dtype=np.float32).reshape(-1, 1)
        train_noise = rng.normal(0.0, noise, size=(train_samples, 1)).astype(np.float32)
        val_noise = rng.normal(0.0, noise, size=(val_samples, 1)).astype(np.float32)
        y_train = sign * fn(x_train).astype(np.float32) + train_noise
        y_val = sign * fn(x_val).astype(np.float32) + val_noise

        tasks.append(
            Task(
                name=name,
                train_x=torch.from_numpy(x_train).to(device),
                train_y=torch.from_numpy(y_train).to(device),
                val_x=torch.from_numpy(x_val).to(device),
                val_y=torch.from_numpy(y_val).to(device),
                conflict_with=conflict_with,
            )
        )

        if sign < 0:
            func_idx += 1

    return tasks


def score_from_mse(mse: float) -> float:
    return float(1.0 / (1.0 + max(mse, 0.0)))


def evaluate_task(model: torch.nn.Module, task: Task) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        pred = model(task.val_x)
        mse = torch.mean((pred - task.val_y) ** 2).item()
    return {"mse": float(mse), "score": score_from_mse(mse)}


def evaluate_all_tasks(model: torch.nn.Module, tasks: List[Task]) -> List[Dict[str, float]]:
    out = []
    for idx, task in enumerate(tasks):
        metrics = evaluate_task(model, task)
        metrics.update({"task_index": idx, "task_name": task.name})
        out.append(metrics)
    return out


def retention_ratio(final_scores: List[float], checkpoint_scores: List[float]) -> float:
    if not checkpoint_scores:
        return 1.0
    ratios = []
    for final, checkpoint in zip(final_scores, checkpoint_scores):
        denom = max(checkpoint, 1.0e-8)
        ratios.append(max(0.0, min(1.0, final / denom)))
    return float(np.mean(ratios)) if ratios else 1.0


def capability_score(post_train_scores: List[float]) -> float:
    return float(np.mean(post_train_scores)) if post_train_scores else 0.0
