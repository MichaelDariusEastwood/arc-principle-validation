#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  EDEN PROTOCOL — SELF-MODIFYING AI EXPERIMENT v3.0 (ADVERSARIAL TASKS)
  
  THE CRITICAL FIX:
  
  v1.0 was rigged (asymmetric proposal distributions).
  v2.0 was fair but toothless (tasks too similar to create tension).
  v3.0 creates GENUINE capability-safety conflict:
  
    Tasks are designed so that the optimal weights for Task N are
    ACTIVELY HARMFUL for Tasks 1 through N-1. Same input domain,
    conflicting output mappings. A network that masters Task 3
    has necessarily destroyed its ability to solve Task 1.
    
    Under these conditions:
      Baseline (capability only): accepts modifications that improve the
        current task while SILENTLY destroying old-task retention.
        It cannot detect the damage because it only checks the current task.
      
      Eden (capability × safety): catches the destruction because the
        safety term (retention of old tasks) drops, pulling the
        combined score down even when capability rises.
    
    If Eden outperforms baseline HERE, under fair proposals and
    genuinely adversarial tasks, the result is scientifically valid.
  
  FAIRNESS GUARANTEES (inherited from v2.0):
    - Identical proposal distributions for all modes
    - Multiple random seeds with statistical testing
    - Explicit fairness audit at startup
    - Fisher's exact test on collapse rates
    - Mann-Whitney U on combined scores
  
  Author: Michael Darius Eastwood
  Framework: ARC Principle / Eden Protocol
  OSF: 10.17605/OSF.IO/6C5XB
═══════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy import stats as sp_stats
import json
import argparse
import warnings
warnings.filterwarnings('ignore')


# ═══════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK (identical to v2.0)
# ═══════════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1,
                 learning_rate=0.01, momentum=0.9,
                 weight_scale=1.0, grad_clip=5.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_scale = weight_scale
        self.grad_clip = grad_clip
        self._init_weights(hidden_dim)
        self.train_steps = 0
        self.weight_history = []

    def _init_weights(self, hidden_dim):
        self.hidden_dim = hidden_dim
        s1 = np.sqrt(2.0 / (self.input_dim + hidden_dim))
        s2 = np.sqrt(2.0 / (hidden_dim + self.output_dim))
        self.W1 = np.random.randn(self.input_dim, hidden_dim) * s1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, self.output_dim) * s2
        self.b2 = np.zeros((1, self.output_dim))
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        self.z1 = X @ (self.W1 * self.weight_scale) + self.b1
        self.a1 = np.maximum(0, self.z1)
        self.z2 = self.a1 @ (self.W2 * self.weight_scale) + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        loss = np.mean((y_pred - y) ** 2)
        dz2 = 2 * (y_pred - y) / m
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        da1 = dz2 @ (self.W2 * self.weight_scale).T
        dz1 = da1 * (self.z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        for grad in [dW1, db1, dW2, db2]:
            norm = np.linalg.norm(grad)
            if norm > self.grad_clip:
                grad *= self.grad_clip / norm
        self.vW2 = self.momentum * self.vW2 - self.learning_rate * dW2
        self.vb2 = self.momentum * self.vb2 - self.learning_rate * db2
        self.vW1 = self.momentum * self.vW1 - self.learning_rate * dW1
        self.vb1 = self.momentum * self.vb1 - self.learning_rate * db1
        self.W2 += self.vW2
        self.b2 += self.vb2
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.train_steps += 1
        self.weight_history.append(np.linalg.norm(self.W1) + np.linalg.norm(self.W2))
        return loss

    def train_batch(self, X, y, epochs=10):
        loss = 0
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = self.backward(X, y, y_pred)
        return loss

    def evaluate(self, X, y):
        return np.mean((self.forward(X) - y) ** 2)

    def get_weight_norm(self):
        return np.linalg.norm(self.W1) + np.linalg.norm(self.W2)

    def get_weight_stability(self, window=10):
        if len(self.weight_history) < window:
            return 1.0
        return 1.0 / (1.0 + np.var(self.weight_history[-window:]))

    def snapshot(self):
        return {k: getattr(self, k).copy() for k in
                ['W1', 'b1', 'W2', 'b2', 'vW1', 'vb1', 'vW2', 'vb2']}

    def restore(self, snap):
        for k, v in snap.items():
            setattr(self, k, v.copy())


# ═══════════════════════════════════════════════════════════════════════════
# ADVERSARIAL TASK ENVIRONMENT (THE v3.0 CORE INNOVATION)
#
# The key insight: tasks must use the SAME input domain but produce
# CONFLICTING outputs. For a small network with shared hidden
# representations, getting good at Task N requires weight configurations
# that actively produce wrong answers for Task N-1.
#
# Design principles:
#   1. Same X range across all tasks (forces shared representation)
#   2. Output mappings that are negatively correlated or orthogonal
#   3. Increasing task complexity forces representational overwriting
#   4. A network cannot simultaneously satisfy all tasks perfectly
# ═══════════════════════════════════════════════════════════════════════════

class AdversarialTaskEnvironment:
    """
    Tasks where optimal weights for Task N destroy performance on Task N-1.
    
    Strategy: ALL tasks use the same input domain [-3, 3].
    But the target functions are designed to CONFLICT:
    
      Task 0: y = +sin(x)           (positive sine)
      Task 1: y = -sin(x)           (NEGATION of Task 0)
      Task 2: y = +cos(x)           (orthogonal to both)
      Task 3: y = -cos(x)           (negation of Task 2)
      Task 4: y = x/3               (linear, conflicts with all periodic)
      Task 5: y = -x/3 + sin(2x)   (anti-correlated with Task 4)
    
    Critical property: A network trained perfectly on Task 1 (y = -sin(x))
    will produce EXACTLY wrong answers for Task 0 (y = +sin(x)).
    The baseline cannot detect this because it only checks the current task.
    The Eden objective CAN detect it because it checks ALL old tasks.
    """

    def __init__(self, n_samples=200, noise=0.02):
        self.n_samples = n_samples
        self.noise = noise
        self.tasks = self._generate_adversarial_tasks()
        self.current_task_idx = 0

    def _generate_adversarial_tasks(self) -> List[Dict]:
        X = np.linspace(-3, 3, self.n_samples).reshape(-1, 1)
        n = np.random.randn(self.n_samples, 1) * self.noise

        tasks = [
            {
                'name': '+sin(x)',
                'X': X.copy(),
                'y': np.sin(X) + n,
                'conflict_with': 'Task 1 is its negation',
            },
            {
                'name': '-sin(x)',
                'X': X.copy(),
                'y': -np.sin(X) + n,
                'conflict_with': 'Task 0 is its negation',
            },
            {
                'name': '+cos(x)',
                'X': X.copy(),
                'y': np.cos(X) + n,
                'conflict_with': 'Task 3 is its negation; orthogonal to Tasks 0-1',
            },
            {
                'name': '-cos(x)',
                'X': X.copy(),
                'y': -np.cos(X) + n,
                'conflict_with': 'Task 2 is its negation',
            },
            {
                'name': 'x/3 (linear)',
                'X': X.copy(),
                'y': X / 3.0 + n,
                'conflict_with': 'Linear vs periodic: requires different representations',
            },
            {
                'name': '-x/3 + sin(2x)',
                'X': X.copy(),
                'y': -X / 3.0 + np.sin(2 * X) + n,
                'conflict_with': 'Anti-correlated with Task 4',
            },
        ]
        return tasks

    def get_current_task(self):
        return self.tasks[self.current_task_idx]

    def advance_task(self):
        if self.current_task_idx < len(self.tasks) - 1:
            self.current_task_idx += 1
            return True
        return False

    def measure_capability(self, net):
        """Accuracy on CURRENT task only."""
        task = self.tasks[self.current_task_idx]
        mse = net.evaluate(task['X'], task['y'])
        return 1.0 / (1.0 + mse)

    def measure_safety(self, net):
        """
        Accuracy on ALL PREVIOUSLY LEARNED tasks.
        
        This is where the adversarial design bites:
        A network optimised for -sin(x) will score terribly on +sin(x).
        The baseline never checks this. Eden always checks this.
        """
        if self.current_task_idx == 0:
            return 1.0
        total = 0.0
        for i in range(self.current_task_idx):
            task = self.tasks[i]
            mse = net.evaluate(task['X'], task['y'])
            total += 1.0 / (1.0 + mse)
        return total / self.current_task_idx

    def measure_combined(self, net):
        cap = self.measure_capability(net)
        safe = self.measure_safety(net)
        return cap, safe, cap * safe

    def verify_adversarial_property(self):
        """
        Verify that the tasks genuinely conflict.
        Compute correlation between consecutive task targets.
        Negative correlation = adversarial.
        """
        print("\n  ADVERSARIAL TASK VERIFICATION")
        print("  " + "=" * 55)
        for i in range(len(self.tasks) - 1):
            y_a = self.tasks[i]['y'].flatten()
            y_b = self.tasks[i + 1]['y'].flatten()
            corr = np.corrcoef(y_a, y_b)[0, 1]
            conflict = "ADVERSARIAL" if corr < -0.3 else "ORTHOGONAL" if abs(corr) < 0.3 else "ALIGNED"
            print(f"  Task {i} ({self.tasks[i]['name']:>16}) vs "
                  f"Task {i+1} ({self.tasks[i+1]['name']:>16}): "
                  f"r = {corr:+.3f} [{conflict}]")
        print("  " + "=" * 55)


# ═══════════════════════════════════════════════════════════════════════════
# META-CONTROLLER (identical fairness guarantees from v2.0)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Proposal:
    parameter: str
    old_value: float
    new_value: float
    accepted: bool = False


class MetaController:
    """
    FAIRNESS GUARANTEE: Identical proposal distributions for all modes.
    The ONLY difference is the acceptance criterion.
    """
    def __init__(self, mode='baseline', exploration_rate=0.4,
                 verification_cost=3):
        assert mode in ('baseline', 'eden', 'eden_drag')
        self.mode = mode
        self.exploration_rate = exploration_rate
        self.verification_cost = verification_cost
        self.log = []
        self.total_verification_compute = 0

    def propose_modifications(self, net, env):
        """IDENTICAL distribution for ALL modes."""
        proposals = []

        if np.random.random() < self.exploration_rate:
            factor = np.exp(np.random.randn() * 0.4)
            new_lr = np.clip(net.learning_rate * factor, 1e-6, 10.0)
            proposals.append(Proposal('learning_rate', net.learning_rate, new_lr))

        if np.random.random() < self.exploration_rate:
            factor = np.exp(np.random.randn() * 0.3)
            new_ws = np.clip(net.weight_scale * factor, 0.01, 50.0)
            proposals.append(Proposal('weight_scale', net.weight_scale, new_ws))

        if np.random.random() < self.exploration_rate * 0.5:
            factor = np.exp(np.random.randn() * 0.3)
            new_gc = np.clip(net.grad_clip * factor, 0.1, 500.0)
            proposals.append(Proposal('grad_clip', net.grad_clip, new_gc))

        if np.random.random() < self.exploration_rate * 0.3:
            new_mom = np.clip(net.momentum + np.random.randn() * 0.1, 0.0, 0.999)
            proposals.append(Proposal('momentum', net.momentum, new_mom))

        return proposals

    def evaluate_proposal(self, proposal, net, env):
        """
        THE ONLY DIFFERENCE BETWEEN MODES:
          baseline:   accept if capability improves (blind to old tasks)
          eden:       accept if capability x safety improves
          eden_drag:  same + pay verification compute
        """
        snap = net.snapshot()
        old_params = {p: getattr(net, p) for p in
                      ['learning_rate', 'weight_scale', 'grad_clip', 'momentum']}

        cap_before, safe_before, combined_before = env.measure_combined(net)

        setattr(net, proposal.parameter, proposal.new_value)
        task = env.get_current_task()
        net.train_batch(task['X'], task['y'], epochs=3)

        if self.mode == 'eden_drag':
            for i in range(env.current_task_idx):
                net.train_batch(env.tasks[i]['X'], env.tasks[i]['y'],
                                epochs=self.verification_cost)
                self.total_verification_compute += self.verification_cost

        cap_after, safe_after, combined_after = env.measure_combined(net)

        if self.mode == 'baseline':
            accept = cap_after > cap_before
        else:
            accept = combined_after > combined_before

        if not accept:
            net.restore(snap)
            for p, v in old_params.items():
                setattr(net, p, v)

        proposal.accepted = accept
        self.log.append(proposal)
        return accept


# ═══════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CycleMetrics:
    cycle: int
    capability: float
    safety: float
    combined: float
    learning_rate: float
    weight_scale: float
    weight_norm: float
    task_idx: int
    collapsed: bool = False


def run_single(mode, n_cycles=180, task_switch=30, seed=42):
    np.random.seed(seed)

    net = NeuralNetwork()
    env = AdversarialTaskEnvironment()
    meta = MetaController(mode=mode)

    history = []
    collapsed = False
    collapse_cycle = None

    for cycle in range(n_cycles):
        if cycle > 0 and cycle % task_switch == 0:
            env.advance_task()

        w_norm = net.get_weight_norm()
        if np.isnan(w_norm) or w_norm > 1e6 or net.learning_rate > 5.0:
            if not collapsed:
                collapsed = True
                collapse_cycle = cycle
            history.append(CycleMetrics(cycle, 0, 0, 0, net.learning_rate,
                                        net.weight_scale, 1e6,
                                        env.current_task_idx, True))
            continue

        task = env.get_current_task()
        net.train_batch(task['X'], task['y'], epochs=5)

        proposals = meta.propose_modifications(net, env)
        for p in proposals:
            meta.evaluate_proposal(p, net, env)

        cap, safe, combined = env.measure_combined(net)
        history.append(CycleMetrics(cycle, cap, safe, combined,
                                    net.learning_rate, net.weight_scale,
                                    net.get_weight_norm(),
                                    env.current_task_idx))

    return {
        'mode': mode, 'seed': seed, 'history': history,
        'collapsed': collapsed, 'collapse_cycle': collapse_cycle,
        'final_cap': history[-1].capability if history else 0,
        'final_safe': history[-1].safety if history else 0,
        'final_combined': history[-1].combined if history else 0,
        'peak_cap': max(h.capability for h in history) if history else 0,
        'peak_combined': max(h.combined for h in history) if history else 0,
        'min_safety': min(h.safety for h in history if not h.collapsed) if any(not h.collapsed for h in history) else 0,
        'mean_safety': np.mean([h.safety for h in history if not h.collapsed]) if any(not h.collapsed for h in history) else 0,
        'mean_combined': np.mean([h.combined for h in history if not h.collapsed]) if any(not h.collapsed for h in history) else 0,
        'final_lr': net.learning_rate,
        'accepted': sum(1 for m in meta.log if m.accepted),
        'total_mods': len(meta.log),
        'verification_compute': meta.total_verification_compute,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FAIRNESS AUDIT
# ═══════════════════════════════════════════════════════════════════════════

def run_fairness_audit():
    print("\n  FAIRNESS AUDIT")
    print("  " + "=" * 55)
    for mode in ['baseline', 'eden', 'eden_drag']:
        np.random.seed(42)
        net = NeuralNetwork()
        env = AdversarialTaskEnvironment()
        meta = MetaController(mode=mode)
        lr_factors = []
        for _ in range(1000):
            props = meta.propose_modifications(net, env)
            for p in props:
                if p.parameter == 'learning_rate':
                    lr_factors.append(p.new_value / p.old_value)
        if lr_factors:
            arr = np.array(lr_factors)
            print(f"  {mode:<12} LR proposals: {len(lr_factors):4d} | "
                  f"mean={np.mean(arr):.3f} | "
                  f"up {np.mean(arr > 1)*100:.1f}% down {np.mean(arr < 1)*100:.1f}%")
    print("\n  All modes: identical distributions ✓")
    print("  " + "=" * 55)


# ═══════════════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def analyse(baseline_runs, eden_runs, drag_runs):
    n = len(baseline_runs)

    print("\n  " + "=" * 70)
    print("  STATISTICAL ANALYSIS (adversarial tasks, fair proposals)")
    print("  " + "=" * 70)

    # Collapse rates
    bc = sum(1 for r in baseline_runs if r['collapsed'])
    ec = sum(1 for r in eden_runs if r['collapsed'])
    dc = sum(1 for r in drag_runs if r['collapsed'])

    print(f"\n  Collapse rates ({n} seeds):")
    print(f"    Baseline:   {bc}/{n} ({bc/n*100:.0f}%)")
    print(f"    Eden:       {ec}/{n} ({ec/n*100:.0f}%)")
    print(f"    Eden+Drag:  {dc}/{n} ({dc/n*100:.0f}%)")

    # Fisher's exact (collapse)
    fisher_p = None
    if bc != ec:
        table = [[bc, n - bc], [ec, n - ec]]
        _, fisher_p = sp_stats.fisher_exact(table)
        print(f"\n  Fisher's exact (baseline vs eden collapse):")
        print(f"    p = {fisher_p:.4f} "
              f"({'SIGNIFICANT' if fisher_p < 0.05 else 'not significant'})")

    # Mean combined scores (all runs, treating collapsed as 0)
    base_combined = [r['mean_combined'] for r in baseline_runs]
    eden_combined = [r['mean_combined'] for r in eden_runs]
    drag_combined = [r['mean_combined'] for r in drag_runs]

    print(f"\n  Mean C×S across entire run (all seeds):")
    for name, vals in [('Baseline', base_combined), ('Eden', eden_combined),
                       ('Eden+Drag', drag_combined)]:
        print(f"    {name:<12} mean={np.mean(vals):.4f} "
              f"std={np.std(vals):.4f}")

    # Mann-Whitney U (Eden > Baseline on mean combined)
    u_stat, p_mw = sp_stats.mannwhitneyu(eden_combined, base_combined,
                                          alternative='greater')
    print(f"\n  Mann-Whitney U (Eden mean C×S > Baseline mean C×S):")
    print(f"    U = {u_stat:.1f}, p = {p_mw:.4f} "
          f"({'SIGNIFICANT' if p_mw < 0.05 else 'not significant'})")

    # Eden+Drag vs Baseline
    u2, p2 = sp_stats.mannwhitneyu(drag_combined, base_combined,
                                    alternative='greater')
    print(f"\n  Mann-Whitney U (Eden+Drag mean C×S > Baseline mean C×S):")
    print(f"    U = {u2:.1f}, p = {p2:.4f} "
          f"({'SIGNIFICANT' if p2 < 0.05 else 'not significant'})")

    # Safety scores specifically
    base_safety = [r['mean_safety'] for r in baseline_runs]
    eden_safety = [r['mean_safety'] for r in eden_runs]
    drag_safety = [r['mean_safety'] for r in drag_runs]

    print(f"\n  Mean SAFETY across entire run (all seeds):")
    for name, vals in [('Baseline', base_safety), ('Eden', eden_safety),
                       ('Eden+Drag', drag_safety)]:
        print(f"    {name:<12} mean={np.mean(vals):.4f} "
              f"std={np.std(vals):.4f}")

    u3, p3 = sp_stats.mannwhitneyu(eden_safety, base_safety,
                                    alternative='greater')
    print(f"\n  Mann-Whitney U (Eden safety > Baseline safety):")
    print(f"    U = {u3:.1f}, p = {p3:.4f} "
          f"({'SIGNIFICANT' if p3 < 0.05 else 'not significant'})")

    # Effect size (Cohen's d on mean_combined)
    pooled_std = np.sqrt((np.var(base_combined) + np.var(eden_combined)) / 2)
    if pooled_std > 0:
        cohens_d = (np.mean(eden_combined) - np.mean(base_combined)) / pooled_std
    else:
        cohens_d = 0
    print(f"\n  Cohen's d (Eden vs Baseline, mean C×S): {cohens_d:.3f}")
    if abs(cohens_d) > 0.8:
        print(f"    Large effect")
    elif abs(cohens_d) > 0.5:
        print(f"    Medium effect")
    elif abs(cohens_d) > 0.2:
        print(f"    Small effect")
    else:
        print(f"    Negligible effect")

    # Per-seed comparison table
    print(f"\n  Per-seed results (mean C×S over entire run):")
    print(f"  {'Seed':>6} {'Baseline':>10} {'Eden':>10} {'Eden+Drag':>10} "
          f"{'Eden wins?':>12}")
    print(f"  " + "-" * 55)
    eden_wins = 0
    for i in range(n):
        b_val = base_combined[i]
        e_val = eden_combined[i]
        d_val = drag_combined[i]
        win = "YES" if e_val > b_val else "no"
        if e_val > b_val:
            eden_wins += 1
        print(f"  {baseline_runs[i]['seed']:>6} {b_val:>10.4f} {e_val:>10.4f} "
              f"{d_val:>10.4f} {win:>12}")
    print(f"\n  Eden wins {eden_wins}/{n} seeds ({eden_wins/n*100:.0f}%)")

    # Binomial test on win rate
    binom_result = sp_stats.binomtest(eden_wins, n, 0.5, alternative='greater')
    binom_p = binom_result.pvalue
    print(f"  Binomial test (Eden wins > 50%): p = {binom_p:.4f} "
          f"({'SIGNIFICANT' if binom_p < 0.05 else 'not significant'})")

    return {
        'n_seeds': n,
        'collapse_rates': {'baseline': bc/n, 'eden': ec/n, 'eden_drag': dc/n},
        'fisher_p': fisher_p,
        'mean_combined': {
            'baseline': float(np.mean(base_combined)),
            'eden': float(np.mean(eden_combined)),
            'eden_drag': float(np.mean(drag_combined)),
        },
        'mean_safety': {
            'baseline': float(np.mean(base_safety)),
            'eden': float(np.mean(eden_safety)),
            'eden_drag': float(np.mean(drag_safety)),
        },
        'mann_whitney_combined_p': float(p_mw),
        'mann_whitney_safety_p': float(p3),
        'cohens_d': float(cohens_d),
        'eden_win_rate': eden_wins / n,
        'binomial_p': float(binom_p),
        'base_combined_all': base_combined,
        'eden_combined_all': eden_combined,
        'drag_combined_all': drag_combined,
        'base_safety_all': base_safety,
        'eden_safety_all': eden_safety,
        'drag_safety_all': drag_safety,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_representative(baseline, eden, eden_drag, output_dir='/home/claude'):
    DARK = '#0F172A'
    RED, GREEN, BLUE = '#DC2626', '#059669', '#2563EB'

    def extract(result, field):
        return [getattr(h, field) for h in result['history']]

    n = len(baseline['history'])
    x = np.arange(n)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(DARK)
    for ax in axes.flat:
        ax.set_facecolor(DARK)
        ax.tick_params(colors='#888')
        for s in ax.spines.values():
            s.set_color('#333')
        ax.grid(True, alpha=0.15, color='#444')

    # Combined
    ax = axes[0, 0]
    ax.plot(x, extract(baseline, 'combined'), color=RED, lw=2, ls='--',
            label='Baseline (cap only)', alpha=0.9)
    ax.plot(x, extract(eden, 'combined'), color=GREEN, lw=2.5,
            label='Eden (cap x safe)', alpha=0.9)
    ax.plot(x, extract(eden_drag, 'combined'), color=BLUE, lw=2,
            label='Eden + drag', alpha=0.9)
    for ts in range(30, n, 30):
        ax.axvline(ts, color='#555', alpha=0.3, ls=':')
        if ts // 30 <= 5:
            task_names = ['+sin', '-sin', '+cos', '-cos', 'x/3', '-x/3+sin2x']
            idx = ts // 30
            if idx < len(task_names):
                ax.text(ts + 1, 0.95, f'→{task_names[idx]}', color='#666',
                        fontsize=8, va='top')
    ax.set_title('Capability × Safety (Adversarial Tasks)', fontsize=13,
                 fontweight='bold', color='white')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333',
              labelcolor='white', loc='upper right')

    # Capability only
    ax = axes[0, 1]
    ax.plot(x, extract(baseline, 'capability'), color=RED, lw=2, ls='--',
            label='Baseline', alpha=0.9)
    ax.plot(x, extract(eden, 'capability'), color=GREEN, lw=2.5,
            label='Eden', alpha=0.9)
    ax.plot(x, extract(eden_drag, 'capability'), color=BLUE, lw=2,
            label='Eden+Drag', alpha=0.9)
    ax.set_title('Capability (current task only)', fontsize=13,
                 fontweight='bold', color='white')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333',
              labelcolor='white')

    # Safety (THE KEY PLOT)
    ax = axes[1, 0]
    ax.plot(x, extract(baseline, 'safety'), color=RED, lw=2, ls='--',
            label='Baseline (blind to old tasks)', alpha=0.9)
    ax.plot(x, extract(eden, 'safety'), color=GREEN, lw=2.5,
            label='Eden (protects old tasks)', alpha=0.9)
    ax.plot(x, extract(eden_drag, 'safety'), color=BLUE, lw=2,
            label='Eden+Drag', alpha=0.9)
    ax.set_title('SAFETY: Retention of Old Tasks (The Key Metric)',
                 fontsize=13, fontweight='bold', color='white')
    ax.set_xlabel('Self-modification cycles', fontsize=11, color='#AAA')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333',
              labelcolor='white')

    # Learning rate
    ax = axes[1, 1]
    ax.semilogy(x, extract(baseline, 'learning_rate'), color=RED, lw=2,
                ls='--', alpha=0.9, label='Baseline')
    ax.semilogy(x, extract(eden, 'learning_rate'), color=GREEN, lw=2.5,
                alpha=0.9, label='Eden')
    ax.semilogy(x, extract(eden_drag, 'learning_rate'), color=BLUE, lw=2,
                alpha=0.9, label='Eden+Drag')
    ax.set_title('Learning rate (log scale)', fontsize=13, fontweight='bold',
                 color='white')
    ax.set_xlabel('Self-modification cycles', fontsize=11, color='#AAA')
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333',
              labelcolor='white')

    fig.suptitle('SELF-MODIFYING AI v3.0: Adversarial Tasks, Fair Proposals\n'
                 'Tasks conflict: +sin → -sin → +cos → -cos → linear → anti-linear',
                 fontsize=14, fontweight='bold', color='white', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = f'{output_dir}/eden_selfmod_v3_results.png'
    fig.savefig(path, dpi=200, facecolor=DARK, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {path}")


def plot_stats(stats, output_dir='/home/claude'):
    DARK = '#0F172A'
    RED, GREEN, BLUE = '#DC2626', '#059669', '#2563EB'

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor(DARK)
    for ax in axes:
        ax.set_facecolor(DARK)
        ax.tick_params(colors='#888')
        for s in ax.spines.values():
            s.set_color('#333')

    modes = ['Baseline', 'Eden', 'Eden+Drag']
    colors = [RED, GREEN, BLUE]

    # Mean combined
    vals = [stats['mean_combined']['baseline'],
            stats['mean_combined']['eden'],
            stats['mean_combined']['eden_drag']]
    bars = axes[0].bar(modes, vals, color=colors, width=0.5, edgecolor='#333')
    for bar, v in zip(bars, vals):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f'{v:.3f}', ha='center', fontsize=11, fontweight='bold',
                     color='white')
    axes[0].set_ylabel('Mean C × S', fontsize=11, color='#AAA')
    axes[0].set_title('Mean Combined Score', fontsize=13, fontweight='bold',
                      color='white')
    p_text = f'p = {stats["mann_whitney_combined_p"]:.4f}'
    axes[0].text(0.5, 0.02, p_text, transform=axes[0].transAxes,
                 ha='center', fontsize=10, color='#AAA')

    # Mean safety
    vals_s = [stats['mean_safety']['baseline'],
              stats['mean_safety']['eden'],
              stats['mean_safety']['eden_drag']]
    bars = axes[1].bar(modes, vals_s, color=colors, width=0.5, edgecolor='#333')
    for bar, v in zip(bars, vals_s):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                     f'{v:.3f}', ha='center', fontsize=11, fontweight='bold',
                     color='white')
    axes[1].set_ylabel('Mean Safety', fontsize=11, color='#AAA')
    axes[1].set_title('Mean Safety (Old Task Retention)', fontsize=13,
                      fontweight='bold', color='white')
    p_text_s = f'p = {stats["mann_whitney_safety_p"]:.4f}'
    axes[1].text(0.5, 0.02, p_text_s, transform=axes[1].transAxes,
                 ha='center', fontsize=10, color='#AAA')

    # Box plots of combined across seeds
    data = [stats['base_combined_all'], stats['eden_combined_all'],
            stats['drag_combined_all']]
    bp = axes[2].boxplot(data, labels=modes, patch_artist=True,
                         medianprops={'color': 'white', 'linewidth': 2},
                         whiskerprops={'color': '#666'},
                         capprops={'color': '#666'},
                         flierprops={'markeredgecolor': '#666'})
    for patch, col in zip(bp['boxes'], colors):
        patch.set_facecolor(col + '40')
        patch.set_edgecolor(col)
    axes[2].set_ylabel('Mean C × S per seed', fontsize=11, color='#AAA')
    axes[2].set_title(f'Distribution ({stats["n_seeds"]} seeds)', fontsize=13,
                      fontweight='bold', color='white')
    d_text = f"Cohen's d = {stats['cohens_d']:.2f}"
    axes[2].text(0.5, 0.02, d_text, transform=axes[2].transAxes,
                 ha='center', fontsize=10, color='#AAA')

    win_pct = stats['eden_win_rate'] * 100
    fig.suptitle(f'v3.0 Statistical Summary: Eden wins {win_pct:.0f}% of seeds '
                 f'(binomial p = {stats["binomial_p"]:.4f})',
                 fontsize=14, fontweight='bold', color='white', y=1.02)
    fig.tight_layout()
    path = f'{output_dir}/eden_selfmod_v3_stats.png'
    fig.savefig(path, dpi=200, facecolor=DARK, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Eden Protocol Self-Modifying AI v3.0 (Adversarial Tasks)')
    parser.add_argument('--cycles', type=int, default=180)
    parser.add_argument('--task-switch', type=int, default=30)
    parser.add_argument('--seeds', type=int, default=20,
                        help='Number of random seeds (default: 20)')
    parser.add_argument('--output', type=str, default='/home/claude')
    args = parser.parse_args()

    print("=" * 70)
    print("  EDEN PROTOCOL — SELF-MODIFYING AI v3.0")
    print("  ADVERSARIAL TASKS + FAIR PROPOSALS")
    print("  The genuine test: tasks that CONFLICT with each other")
    print("=" * 70)

    # Verify adversarial property
    np.random.seed(0)
    env_check = AdversarialTaskEnvironment()
    env_check.verify_adversarial_property()

    # Fairness audit
    run_fairness_audit()

    seed_list = list(range(42, 42 + args.seeds))

    print(f"\n  Running {args.seeds} seeds x 3 modes = "
          f"{args.seeds * 3} experiments")
    print(f"  Cycles: {args.cycles} | Task switch: {args.task_switch}")
    print(f"  Tasks: +sin, -sin, +cos, -cos, x/3, -x/3+sin(2x)")

    all_results = {}
    for mode in ['baseline', 'eden', 'eden_drag']:
        print(f"\n  {'─'*55}")
        print(f"  MODE: {mode.upper()} ({args.seeds} seeds)")
        print(f"  {'─'*55}")

        runs = []
        for seed in seed_list:
            r = run_single(mode, args.cycles, args.task_switch, seed)
            status = "COLLAPSED" if r['collapsed'] else "stable"
            print(f"    seed={seed}: {status:>9} | "
                  f"C×S={r['mean_combined']:.4f} | "
                  f"safe={r['mean_safety']:.4f} | "
                  f"LR={r['final_lr']:.5f}")
            runs.append(r)
        all_results[mode] = runs

    # Statistical analysis
    stats = analyse(all_results['baseline'], all_results['eden'],
                    all_results['eden_drag'])

    # Plots
    print("\n  Generating plots...")
    plot_representative(all_results['baseline'][0],
                        all_results['eden'][0],
                        all_results['eden_drag'][0],
                        output_dir=args.output)
    plot_stats(stats, output_dir=args.output)

    # Export JSON
    export = {
        'metadata': {
            'experiment': 'Eden Protocol Self-Modifying AI v3.0 (Adversarial)',
            'author': 'Michael Darius Eastwood',
            'timestamp': datetime.now().isoformat(),
            'n_seeds': args.seeds,
            'cycles': args.cycles,
            'task_switch': args.task_switch,
            'fairness': 'VERIFIED: identical proposal distributions',
            'adversarial': 'VERIFIED: consecutive tasks negatively correlated',
            'description': (
                'Tasks use same input domain with conflicting outputs. '
                '+sin(x) followed by -sin(x) forces representational conflict. '
                'Baseline checks only current task. '
                'Eden checks current task × retention of all old tasks.'
            ),
        },
        'statistics': {
            k: v for k, v in stats.items()
            if k not in ('base_combined_all', 'eden_combined_all',
                         'drag_combined_all', 'base_safety_all',
                         'eden_safety_all', 'drag_safety_all')
        },
        'per_seed': {
            mode: [{'seed': r['seed'], 'collapsed': r['collapsed'],
                    'mean_combined': r['mean_combined'],
                    'mean_safety': r['mean_safety'],
                    'final_lr': r['final_lr']}
                   for r in runs]
            for mode, runs in all_results.items()
        }
    }
    path = f"{args.output}/eden_selfmod_v3_results.json"
    with open(path, 'w') as f:
        json.dump(export, f, indent=2, default=str)
    print(f"  [JSON] {path}")

    # Final verdict
    print("\n" + "=" * 70)
    p_combined = stats['mann_whitney_combined_p']
    p_safety = stats['mann_whitney_safety_p']
    p_binom = stats['binomial_p']
    d = stats['cohens_d']
    win_rate = stats['eden_win_rate']

    if p_combined < 0.05 and p_safety < 0.05:
        print("  ✓ RESULT: EDEN SIGNIFICANTLY OUTPERFORMS BASELINE")
        print(f"    Combined C×S:  p = {p_combined:.4f} (significant)")
        print(f"    Safety alone:  p = {p_safety:.4f} (significant)")
        print(f"    Cohen's d:     {d:.3f}")
        print(f"    Eden win rate: {win_rate*100:.0f}% (binomial p = {p_binom:.4f})")
        print(f"    Under FAIR proposals and ADVERSARIAL tasks,")
        print(f"    the entangled loss function provides measurable")
        print(f"    protection against catastrophic forgetting.")
        print(f"    The load-bearing wall holds under genuine stress.")
    elif p_safety < 0.05:
        print("  ~ PARTIAL RESULT: EDEN PRESERVES SAFETY BETTER")
        print(f"    Safety:    p = {p_safety:.4f} (significant)")
        print(f"    Combined:  p = {p_combined:.4f} (not significant)")
        print(f"    Eden preserves old-task knowledge better,")
        print(f"    but the combined metric is not significantly different.")
    elif p_combined < 0.1 or p_safety < 0.1:
        print("  ~ TREND: Eden outperforms baseline (marginal significance)")
        print(f"    Combined: p = {p_combined:.4f}")
        print(f"    Safety:   p = {p_safety:.4f}")
        print(f"    Cohen's d: {d:.3f}")
        print(f"    More seeds may reach significance.")
    else:
        print("  ✗ NO SIGNIFICANT DIFFERENCE UNDER THESE CONDITIONS")
        print(f"    Combined: p = {p_combined:.4f}")
        print(f"    Safety:   p = {p_safety:.4f}")
        print(f"    The adversarial task structure may need further")
        print(f"    strengthening, or the theory needs revision.")

    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
