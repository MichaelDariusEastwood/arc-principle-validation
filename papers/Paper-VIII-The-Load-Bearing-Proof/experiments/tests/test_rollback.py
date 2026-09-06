import random

import torch

from models.learned_optimizer import LearnedOptimizer
from training.outer_loop import initial_agent_state
from training.controller import ChangeController


def test_reject_preserves_parent_state():
    torch.manual_seed(0)
    learned_opt = LearnedOptimizer(feature_dim=8, hidden_size=16)
    parent = initial_agent_state(learned_opt)
    parent_copy = {k: v.clone() for k, v in parent.optimizer_state.items()}
    controller = ChangeController(
        condition="eden",
        delta_c_min=0.0,
        delta_cs_min=0.0,
        s_min=0.8,
        accept_epsilon=1e-6,
        max_param_norm=1000.0,
        catastrophic_floor=0.1,
        determinism_checks=2,
        policy_mutation_scale=0.1,
        meta_mutation_scale=0.1,
        allow_activation_mutation=True,
        allow_feature_mutation=True,
        rng=random.Random(123),
    )
    cand = controller.mutate_candidate(parent.optimizer_state, parent.policy_genome)
    # parent state must remain untouched
    for name, tensor in parent.optimizer_state.items():
        assert torch.equal(tensor, parent_copy[name])
    # candidate differs on at least one tensor
    assert any(not torch.equal(cand.optimizer_state[k], parent_copy[k]) for k in cand.optimizer_state)
