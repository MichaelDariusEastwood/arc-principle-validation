from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from tasks.suite import make_adversarial_regression_tasks
from models.learned_optimizer import LearnedOptimizer
from training.outer_loop import initial_agent_state, run_iteration, build_optimizer
from training.controller import ChangeController, LedgerWriter
from metrics.logging import write_json, to_jsonable


def choose_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_seed_condition(cfg: Dict[str, Any], *, seed: int, condition: str) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = choose_device(cfg["experiment"].get("device", "auto"))
    tasks = make_adversarial_regression_tasks(
        num_tasks=int(cfg["tasks"]["num_tasks"]),
        train_samples=int(cfg["tasks"]["train_samples"]),
        val_samples=int(cfg["tasks"]["val_samples"]),
        noise=float(cfg["tasks"]["noise"]),
        seed=seed,
        device=device,
    )

    learned_opt = build_optimizer(cfg["optimizer"], device)
    agent_state = initial_agent_state(learned_opt)

    rng = random.Random(seed + 1000)
    controller = ChangeController(
        condition=condition,
        delta_c_min=float(cfg["experiment"]["delta_c_min"]),
        delta_cs_min=float(cfg["experiment"]["delta_cs_min"]),
        s_min=float(cfg["experiment"]["s_min"]),
        accept_epsilon=float(cfg["experiment"]["accept_epsilon"]),
        max_param_norm=float(cfg["verifier"]["max_param_norm"]),
        catastrophic_floor=float(cfg["verifier"]["catastrophic_floor"]),
        determinism_checks=int(cfg["verifier"]["determinism_checks"]),
        policy_mutation_scale=float(cfg["policy"]["mutation_scale"]),
        meta_mutation_scale=float(cfg["optimizer"]["meta_noise_std"]),
        allow_activation_mutation=bool(cfg["policy"].get("allow_activation_mutation", True)),
        allow_feature_mutation=bool(cfg["policy"].get("allow_feature_mutation", True)),
        rng=rng,
    )

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(cfg["experiment"]["output_dir"]) / f"{now}_{condition}_seed{seed}"
    ledger = LedgerWriter(run_dir / "ledger.jsonl")

    history = []
    compute_track = []
    capability_track = []
    safety_track = []
    combined_track = []

    for iteration in range(int(cfg["experiment"]["iterations"])):
        agent_state, log_row = run_iteration(
            parent_state=agent_state,
            controller=controller,
            tasks=tasks,
            model_cfg=cfg["model"],
            opt_cfg=cfg["optimizer"],
            lambda_drag=float(cfg["experiment"]["lambda_drag"]),
            batch_size=int(cfg["tasks"]["batch_size"]),
            device=device,
        )
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "condition": condition,
            "iteration": iteration,
            "accepted": bool(log_row["accepted"]),
            "reason": log_row["reason"],
            "parent": to_jsonable(log_row["parent"]),
            "candidate": to_jsonable(log_row.get("candidate")) if log_row.get("candidate") is not None else None,
            "verify_details": log_row.get("verify_details", {}),
            "policy_source_before": log_row["policy_source_before"],
            "policy_source_after": log_row["policy_source_after"],
        }
        ledger.append(row)
        history.append(row)
        chosen = log_row.get("candidate") if log_row.get("accepted") and log_row.get("candidate") is not None else log_row["parent"]
        capability_track.append(float(chosen.capability))
        safety_track.append(float(chosen.safety))
        combined_track.append(float(chosen.combined))
        compute_track.append(float(chosen.compute_steps))

    summary = {
        "seed": seed,
        "condition": condition,
        "device": str(device),
        "config": cfg,
        "history": history,
        "tracks": {
            "capability": capability_track,
            "safety": safety_track,
            "combined": combined_track,
            "compute_steps": compute_track,
        },
        "final_policy_source": agent_state.policy_source,
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded Eden self-modifying AI harness")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--condition", type=str, default=None, help="Override a single condition")
    parser.add_argument("--seed", type=int, default=None, help="Run one seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    conditions = [args.condition] if args.condition else list(cfg["experiment"]["conditions"])
    seeds = [args.seed] if args.seed is not None else list(cfg["experiment"]["seeds"])

    all_summaries = []
    for condition in conditions:
        for seed in seeds:
            print(f"[run_train] condition={condition} seed={seed}")
            all_summaries.append(run_seed_condition(cfg, seed=seed, condition=condition))

    out_dir = Path(cfg["experiment"]["output_dir"])
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "runs": [
            {
                "condition": s["condition"],
                "seed": s["seed"],
                "device": s["device"],
                "final_capability": s["tracks"]["capability"][-1],
                "final_safety": s["tracks"]["safety"][-1],
                "final_combined": s["tracks"]["combined"][-1],
            }
            for s in all_summaries
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[run_train] wrote manifest to {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
