#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  EDEN PROTOCOL — SELF-MODIFYING AI EXPERIMENT v4.0 (COMPLEXITY SCALING)
  
  THE QUESTION v3.0 LEFT OPEN:
  
  v3.0 found:
    - Eden preserves safety significantly (p = 0.0017)
    - Eden+Drag improves combined metric (p = 0.012)
    - But effect sizes are modest in a simple 1→32→1 network
  
  v4.0 ASKS: Do these effects SCALE with system complexity?
  
  THE ARC PRINCIPLE PREDICTS:
    As recursive depth and system complexity increase, the advantage
    of embedded alignment over decoupled alignment should GROW,
    not stay constant. This is the core α > 1 claim: compounding
    returns from recursive self-correction.
    
    If Eden's advantage is constant across complexity levels,
    the ARC scaling prediction is wrong.
    
    If Eden's advantage GROWS with complexity, the prediction holds.
  
  EXPERIMENTAL DESIGN:
    Run the v3.0 adversarial experiment at FIVE complexity levels:
    
    Level 1: Net 1→16→1,   4 tasks,  3 modifiable params  (tiny)
    Level 2: Net 1→32→1,   6 tasks,  4 modifiable params  (v3.0 baseline)
    Level 3: Net 1→64→1,   8 tasks,  4 modifiable params  (medium)
    Level 4: Net 1→128→1, 10 tasks,  5 modifiable params  (large)
    Level 5: Net 1→64→64→1, 10 tasks, 6 modifiable params (deep)
    
    At each level, run baseline vs Eden vs Eden+Drag across 15 seeds.
    Measure the DELTA (Eden - Baseline) at each level.
    Test whether delta correlates with complexity.
    
    If Spearman rho > 0 and p < 0.05, the scaling prediction holds.
  
  FAIRNESS: Inherited from v2.0/v3.0 — identical proposals all modes.
  ADVERSARIAL: Inherited from v3.0 — conflicting task targets.
  
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
# NEURAL NETWORK (extended for multi-layer support)
# ═══════════════════════════════════════════════════════════════════════════

class NeuralNetwork:
    """Supports 1 or 2 hidden layers for complexity scaling."""

    def __init__(self, input_dim=1, hidden_dims=(32,), output_dim=1,
                 learning_rate=0.01, momentum=0.9,
                 weight_scale=1.0, grad_clip=5.0,
                 weight_decay=0.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.n_layers = len(hidden_dims)

        # Modifiable hyperparameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_scale = weight_scale
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay

        self._init_weights()
        self.train_steps = 0
        self.weight_history = []

    def _init_weights(self):
        dims = [self.input_dim] + list(self.hidden_dims) + [self.output_dim]
        self.weights = []
        self.biases = []
        self.v_weights = []
        self.v_biases = []

        for i in range(len(dims) - 1):
            scale = np.sqrt(2.0 / (dims[i] + dims[i + 1]))
            self.weights.append(np.random.randn(dims[i], dims[i + 1]) * scale)
            self.biases.append(np.zeros((1, dims[i + 1])))
            self.v_weights.append(np.zeros((dims[i], dims[i + 1])))
            self.v_biases.append(np.zeros((1, dims[i + 1])))

    def forward(self, X):
        self.activations = [X]
        self.pre_activations = []
        h = X
        for i in range(len(self.weights)):
            z = h @ (self.weights[i] * self.weight_scale) + self.biases[i]
            self.pre_activations.append(z)
            if i < len(self.weights) - 1:
                h = np.maximum(0, z)  # ReLU for hidden
            else:
                h = z  # Linear output
            self.activations.append(h)
        return h

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        loss = np.mean((y_pred - y) ** 2)

        # Backprop through layers
        delta = 2 * (y_pred - y) / m
        grads_w = []
        grads_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            dW = self.activations[i].T @ delta
            db = np.sum(delta, axis=0, keepdims=True)

            # Weight decay
            if self.weight_decay > 0:
                dW += self.weight_decay * self.weights[i]

            # Gradient clipping
            norm = np.linalg.norm(dW)
            if norm > self.grad_clip:
                dW *= self.grad_clip / norm
            norm_b = np.linalg.norm(db)
            if norm_b > self.grad_clip:
                db *= self.grad_clip / norm_b

            grads_w.insert(0, dW)
            grads_b.insert(0, db)

            if i > 0:
                delta = delta @ (self.weights[i] * self.weight_scale).T
                delta = delta * (self.pre_activations[i - 1] > 0).astype(float)

        # Update with momentum
        for i in range(len(self.weights)):
            self.v_weights[i] = self.momentum * self.v_weights[i] - self.learning_rate * grads_w[i]
            self.v_biases[i] = self.momentum * self.v_biases[i] - self.learning_rate * grads_b[i]
            self.weights[i] += self.v_weights[i]
            self.biases[i] += self.v_biases[i]

        self.train_steps += 1
        self.weight_history.append(self.get_weight_norm())
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
        return sum(np.linalg.norm(w) for w in self.weights)

    def get_weight_stability(self, window=10):
        if len(self.weight_history) < window:
            return 1.0
        return 1.0 / (1.0 + np.var(self.weight_history[-window:]))

    def snapshot(self):
        return {
            'weights': [w.copy() for w in self.weights],
            'biases': [b.copy() for b in self.biases],
            'v_weights': [v.copy() for v in self.v_weights],
            'v_biases': [v.copy() for v in self.v_biases],
        }

    def restore(self, snap):
        self.weights = [w.copy() for w in snap['weights']]
        self.biases = [b.copy() for b in snap['biases']]
        self.v_weights = [v.copy() for v in snap['v_weights']]
        self.v_biases = [v.copy() for v in snap['v_biases']]

    def param_count(self):
        total = 0
        for w, b in zip(self.weights, self.biases):
            total += w.size + b.size
        return total


# ═══════════════════════════════════════════════════════════════════════════
# ADVERSARIAL TASK GENERATOR (scalable)
# ═══════════════════════════════════════════════════════════════════════════

def generate_adversarial_tasks(n_tasks, n_samples=200, noise=0.02):
    """
    Generate n_tasks adversarial tasks on the same input domain.

    Strategy: pairs of negated functions, plus orthogonal bases.
    Task 2k:   +f_k(x)
    Task 2k+1: -f_k(x)

    Functions chosen to be mutually orthogonal where possible:
    sin(x), cos(x), sin(2x), cos(2x), sin(3x), ...
    """
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    n = np.random.randn(n_samples, 1) * noise

    # Base functions (orthogonal-ish over [-3, 3])
    base_functions = [
        ('sin(x)', lambda x: np.sin(x)),
        ('cos(x)', lambda x: np.cos(x)),
        ('sin(2x)', lambda x: np.sin(2 * x)),
        ('cos(2x)', lambda x: np.cos(2 * x)),
        ('sin(3x)', lambda x: np.sin(3 * x)),
        ('cos(3x)', lambda x: np.cos(3 * x)),
        ('x/3', lambda x: x / 3.0),
        ('tanh(x)', lambda x: np.tanh(x)),
        ('sin(x)*cos(x)', lambda x: np.sin(x) * np.cos(x)),
        ('x*sin(x)/3', lambda x: x * np.sin(x) / 3.0),
    ]

    tasks = []
    func_idx = 0
    for i in range(n_tasks):
        if i % 2 == 0:
            # Positive version
            name, fn = base_functions[func_idx % len(base_functions)]
            tasks.append({
                'name': f'+{name}',
                'X': X.copy(),
                'y': fn(X) + n,
            })
        else:
            # Negated version (adversarial to previous)
            name, fn = base_functions[func_idx % len(base_functions)]
            tasks.append({
                'name': f'-{name}',
                'X': X.copy(),
                'y': -fn(X) + n,
            })
            func_idx += 1

    return tasks


class ScalableTaskEnvironment:
    def __init__(self, n_tasks=6, n_samples=200, noise=0.02):
        self.tasks = generate_adversarial_tasks(n_tasks, n_samples, noise)
        self.current_task_idx = 0

    def get_current_task(self):
        return self.tasks[self.current_task_idx]

    def advance_task(self):
        if self.current_task_idx < len(self.tasks) - 1:
            self.current_task_idx += 1
            return True
        return False

    def measure_capability(self, net):
        task = self.tasks[self.current_task_idx]
        mse = net.evaluate(task['X'], task['y'])
        return 1.0 / (1.0 + mse)

    def measure_safety(self, net):
        if self.current_task_idx == 0:
            return 1.0
        total = sum(1.0 / (1.0 + net.evaluate(self.tasks[i]['X'], self.tasks[i]['y']))
                    for i in range(self.current_task_idx))
        return total / self.current_task_idx

    def measure_combined(self, net):
        cap = self.measure_capability(net)
        safe = self.measure_safety(net)
        return cap, safe, cap * safe


# ═══════════════════════════════════════════════════════════════════════════
# META-CONTROLLER (fair, identical to v3.0 + optional weight_decay control)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Proposal:
    parameter: str
    old_value: float
    new_value: float
    accepted: bool = False


class MetaController:
    def __init__(self, mode='baseline', exploration_rate=0.4,
                 verification_cost=3, modifiable_params=None):
        assert mode in ('baseline', 'eden', 'eden_drag')
        self.mode = mode
        self.exploration_rate = exploration_rate
        self.verification_cost = verification_cost
        self.modifiable_params = modifiable_params or [
            'learning_rate', 'weight_scale', 'grad_clip', 'momentum']
        self.log = []
        self.total_verification_compute = 0

    def propose_modifications(self, net, env):
        proposals = []

        if 'learning_rate' in self.modifiable_params:
            if np.random.random() < self.exploration_rate:
                factor = np.exp(np.random.randn() * 0.4)
                new_lr = np.clip(net.learning_rate * factor, 1e-6, 10.0)
                proposals.append(Proposal('learning_rate', net.learning_rate, new_lr))

        if 'weight_scale' in self.modifiable_params:
            if np.random.random() < self.exploration_rate:
                factor = np.exp(np.random.randn() * 0.3)
                new_ws = np.clip(net.weight_scale * factor, 0.01, 50.0)
                proposals.append(Proposal('weight_scale', net.weight_scale, new_ws))

        if 'grad_clip' in self.modifiable_params:
            if np.random.random() < self.exploration_rate * 0.5:
                factor = np.exp(np.random.randn() * 0.3)
                new_gc = np.clip(net.grad_clip * factor, 0.1, 500.0)
                proposals.append(Proposal('grad_clip', net.grad_clip, new_gc))

        if 'momentum' in self.modifiable_params:
            if np.random.random() < self.exploration_rate * 0.3:
                new_mom = np.clip(net.momentum + np.random.randn() * 0.1, 0.0, 0.999)
                proposals.append(Proposal('momentum', net.momentum, new_mom))

        if 'weight_decay' in self.modifiable_params:
            if np.random.random() < self.exploration_rate * 0.3:
                new_wd = np.clip(net.weight_decay + np.random.randn() * 0.001, 0.0, 0.1)
                proposals.append(Proposal('weight_decay', net.weight_decay, new_wd))

        return proposals

    def evaluate_proposal(self, proposal, net, env):
        snap = net.snapshot()
        old_params = {p: getattr(net, p) for p in self.modifiable_params
                      if hasattr(net, p)}

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
# COMPLEXITY LEVELS
# ═══════════════════════════════════════════════════════════════════════════

COMPLEXITY_LEVELS = [
    {
        'name': 'Tiny',
        'level': 1,
        'hidden_dims': (16,),
        'n_tasks': 4,
        'modifiable_params': ['learning_rate', 'weight_scale', 'grad_clip'],
        'task_switch': 35,
        'cycles': 140,
    },
    {
        'name': 'Small (v3.0)',
        'level': 2,
        'hidden_dims': (32,),
        'n_tasks': 6,
        'modifiable_params': ['learning_rate', 'weight_scale', 'grad_clip', 'momentum'],
        'task_switch': 30,
        'cycles': 180,
    },
    {
        'name': 'Medium',
        'level': 3,
        'hidden_dims': (64,),
        'n_tasks': 8,
        'modifiable_params': ['learning_rate', 'weight_scale', 'grad_clip', 'momentum'],
        'task_switch': 25,
        'cycles': 200,
    },
    {
        'name': 'Large',
        'level': 4,
        'hidden_dims': (128,),
        'n_tasks': 10,
        'modifiable_params': ['learning_rate', 'weight_scale', 'grad_clip',
                              'momentum', 'weight_decay'],
        'task_switch': 22,
        'cycles': 220,
    },
    {
        'name': 'Deep (2-layer)',
        'level': 5,
        'hidden_dims': (64, 64),
        'n_tasks': 10,
        'modifiable_params': ['learning_rate', 'weight_scale', 'grad_clip',
                              'momentum', 'weight_decay'],
        'task_switch': 22,
        'cycles': 220,
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# SINGLE RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_single(mode, complexity, seed=42):
    np.random.seed(seed)

    net = NeuralNetwork(
        hidden_dims=complexity['hidden_dims'],
        learning_rate=0.01, momentum=0.9,
        weight_scale=1.0, grad_clip=5.0, weight_decay=0.0)

    env = ScalableTaskEnvironment(n_tasks=complexity['n_tasks'])
    meta = MetaController(
        mode=mode,
        modifiable_params=complexity['modifiable_params'])

    n_cycles = complexity['cycles']
    task_switch = complexity['task_switch']

    cap_history = []
    safe_history = []
    combined_history = []
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
            cap_history.append(0)
            safe_history.append(0)
            combined_history.append(0)
            continue

        task = env.get_current_task()
        net.train_batch(task['X'], task['y'], epochs=5)

        proposals = meta.propose_modifications(net, env)
        for p in proposals:
            meta.evaluate_proposal(p, net, env)

        cap, safe, combined = env.measure_combined(net)
        cap_history.append(cap)
        safe_history.append(safe)
        combined_history.append(combined)

    return {
        'mode': mode,
        'seed': seed,
        'collapsed': collapsed,
        'collapse_cycle': collapse_cycle,
        'mean_combined': float(np.mean(combined_history)),
        'mean_safety': float(np.mean(safe_history)),
        'mean_cap': float(np.mean(cap_history)),
        'final_combined': combined_history[-1] if combined_history else 0,
        'final_safety': safe_history[-1] if safe_history else 0,
        'param_count': net.param_count(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT: RUN ALL LEVELS
# ═══════════════════════════════════════════════════════════════════════════

def run_scaling_experiment(n_seeds=15):
    """Run all complexity levels, all modes, all seeds."""

    seed_list = list(range(42, 42 + n_seeds))
    all_results = {}

    for comp in COMPLEXITY_LEVELS:
        level = comp['level']
        name = comp['name']
        all_results[level] = {'complexity': comp, 'modes': {}}

        print(f"\n{'═'*65}")
        print(f"  COMPLEXITY LEVEL {level}: {name}")
        print(f"  Network: 1→{'→'.join(str(d) for d in comp['hidden_dims'])}→1  "
              f"| Tasks: {comp['n_tasks']} | Params modifiable: "
              f"{len(comp['modifiable_params'])}")
        print(f"{'═'*65}")

        for mode in ['baseline', 'eden', 'eden_drag']:
            runs = []
            collapses = 0
            for seed in seed_list:
                r = run_single(mode, comp, seed)
                runs.append(r)
                if r['collapsed']:
                    collapses += 1

            mean_comb = np.mean([r['mean_combined'] for r in runs])
            mean_safe = np.mean([r['mean_safety'] for r in runs])
            std_comb = np.std([r['mean_combined'] for r in runs])

            all_results[level]['modes'][mode] = {
                'runs': runs,
                'mean_combined': float(mean_comb),
                'std_combined': float(std_comb),
                'mean_safety': float(mean_safe),
                'collapses': collapses,
            }

            print(f"  {mode:<12} C×S={mean_comb:.4f}±{std_comb:.4f} "
                  f"safe={mean_safe:.4f} collapses={collapses}/{n_seeds}")

    return all_results


def analyse_scaling(all_results, n_seeds):
    """The core analysis: does Eden's advantage grow with complexity?"""

    print("\n" + "=" * 70)
    print("  SCALING ANALYSIS: Does Eden's advantage grow with complexity?")
    print("=" * 70)

    levels = sorted(all_results.keys())

    # Compute deltas at each level
    delta_combined = []      # Eden - Baseline on combined
    delta_safety = []        # Eden - Baseline on safety
    delta_drag_combined = [] # Eden+Drag - Baseline on combined
    delta_drag_safety = []   # Eden+Drag - Baseline on safety
    level_labels = []
    param_counts = []

    # Per-level Mann-Whitney tests
    print(f"\n  {'Level':<18} {'Base C×S':>10} {'Eden C×S':>10} {'Drag C×S':>10} "
          f"{'Δ Eden':>8} {'Δ Drag':>8} {'p(safe)':>8}")
    print(f"  " + "-" * 75)

    for level in levels:
        data = all_results[level]
        comp = data['complexity']
        base = data['modes']['baseline']
        eden = data['modes']['eden']
        drag = data['modes']['eden_drag']

        d_comb = eden['mean_combined'] - base['mean_combined']
        d_safe = eden['mean_safety'] - base['mean_safety']
        d_drag_comb = drag['mean_combined'] - base['mean_combined']
        d_drag_safe = drag['mean_safety'] - base['mean_safety']

        delta_combined.append(d_comb)
        delta_safety.append(d_safe)
        delta_drag_combined.append(d_drag_comb)
        delta_drag_safety.append(d_drag_safe)
        level_labels.append(comp['name'])
        param_counts.append(data['modes']['baseline']['runs'][0]['param_count'])

        # Mann-Whitney on safety for this level
        base_safeties = [r['mean_safety'] for r in base['runs']]
        eden_safeties = [r['mean_safety'] for r in eden['runs']]
        _, p_safe = sp_stats.mannwhitneyu(eden_safeties, base_safeties,
                                           alternative='greater')

        print(f"  {comp['name']:<18} {base['mean_combined']:>10.4f} "
              f"{eden['mean_combined']:>10.4f} {drag['mean_combined']:>10.4f} "
              f"{d_comb:>+8.4f} {d_drag_comb:>+8.4f} {p_safe:>8.4f}")

    # THE KEY TEST: does delta correlate with complexity level?
    print(f"\n  SCALING CORRELATIONS (Spearman):")
    print(f"  " + "-" * 60)

    tests = [
        ("Eden safety Δ vs level", delta_safety, levels),
        ("Eden combined Δ vs level", delta_combined, levels),
        ("Drag safety Δ vs level", delta_drag_safety, levels),
        ("Drag combined Δ vs level", delta_drag_combined, levels),
        ("Eden safety Δ vs params", delta_safety, param_counts),
        ("Drag combined Δ vs params", delta_drag_combined, param_counts),
    ]

    scaling_results = {}
    for name, deltas, x_vals in tests:
        rho, p = sp_stats.spearmanr(x_vals, deltas)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {name:<35} ρ={rho:+.3f}  p={p:.4f}  {sig}")
        scaling_results[name] = {'rho': float(rho), 'p': float(p)}

    # Effect size at each level (Cohen's d on combined)
    print(f"\n  EFFECT SIZE SCALING (Cohen's d, Eden vs Baseline on C×S):")
    print(f"  " + "-" * 50)
    cohens_ds = []
    for level in levels:
        data = all_results[level]
        base_vals = [r['mean_combined'] for r in data['modes']['baseline']['runs']]
        eden_vals = [r['mean_combined'] for r in data['modes']['eden']['runs']]
        pooled = np.sqrt((np.var(base_vals) + np.var(eden_vals)) / 2)
        d = (np.mean(eden_vals) - np.mean(base_vals)) / pooled if pooled > 0 else 0
        cohens_ds.append(d)
        size = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small" if abs(d) > 0.2 else "negligible"
        print(f"  Level {level} ({data['complexity']['name']:<18}): d = {d:+.3f} ({size})")

    # Does Cohen's d scale?
    rho_d, p_d = sp_stats.spearmanr(levels, cohens_ds)
    print(f"\n  Cohen's d vs complexity level: ρ = {rho_d:+.3f}, p = {p_d:.4f}")

    scaling_results['cohens_d_scaling'] = {
        'values': cohens_ds,
        'rho': float(rho_d),
        'p': float(p_d),
    }

    return {
        'levels': levels,
        'level_labels': level_labels,
        'param_counts': param_counts,
        'delta_combined': delta_combined,
        'delta_safety': delta_safety,
        'delta_drag_combined': delta_drag_combined,
        'delta_drag_safety': delta_drag_safety,
        'cohens_ds': cohens_ds,
        'scaling_results': scaling_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════

def plot_scaling(analysis, all_results, output_dir='/home/claude'):
    DARK = '#0F172A'
    RED, GREEN, BLUE, AMBER = '#DC2626', '#059669', '#2563EB', '#D97706'

    levels = analysis['levels']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(DARK)
    for ax in axes.flat:
        ax.set_facecolor(DARK)
        ax.tick_params(colors='#888')
        for s in ax.spines.values():
            s.set_color('#333')
        ax.grid(True, alpha=0.15, color='#444')

    # Top-left: Mean combined at each level
    ax = axes[0, 0]
    base_means = [all_results[l]['modes']['baseline']['mean_combined'] for l in levels]
    eden_means = [all_results[l]['modes']['eden']['mean_combined'] for l in levels]
    drag_means = [all_results[l]['modes']['eden_drag']['mean_combined'] for l in levels]

    x_pos = np.arange(len(levels))
    w = 0.25
    ax.bar(x_pos - w, base_means, w, color=RED, label='Baseline', alpha=0.85)
    ax.bar(x_pos, eden_means, w, color=GREEN, label='Eden', alpha=0.85)
    ax.bar(x_pos + w, drag_means, w, color=BLUE, label='Eden+Drag', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(analysis['level_labels'], fontsize=8, rotation=15)
    ax.set_title('Mean C×S by Complexity Level', fontsize=13,
                 fontweight='bold', color='white')
    ax.set_ylabel('Mean C × S', fontsize=11, color='#AAA')
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333',
              labelcolor='white')

    # Top-right: Safety delta scaling (THE KEY PLOT)
    ax = axes[0, 1]
    ax.plot(levels, analysis['delta_safety'], 'o-', color=GREEN, lw=2.5,
            ms=8, label='Eden safety Δ', zorder=3)
    ax.plot(levels, analysis['delta_drag_safety'], 's-', color=BLUE, lw=2,
            ms=7, label='Drag safety Δ', zorder=3)
    ax.axhline(0, color='#444', lw=1, ls=':')

    sr = analysis['scaling_results'].get("Eden safety Δ vs level", {})
    rho_s = sr.get('rho', 0)
    p_s = sr.get('p', 1)
    ax.set_title(f'Safety Advantage vs Complexity (ρ={rho_s:+.2f}, p={p_s:.3f})',
                 fontsize=13, fontweight='bold', color='white')
    ax.set_xlabel('Complexity Level', fontsize=11, color='#AAA')
    ax.set_ylabel('Eden safety - Baseline safety', fontsize=11, color='#AAA')
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333',
              labelcolor='white')

    # Bottom-left: Combined delta scaling
    ax = axes[1, 0]
    ax.plot(levels, analysis['delta_combined'], 'o-', color=GREEN, lw=2.5,
            ms=8, label='Eden C×S Δ')
    ax.plot(levels, analysis['delta_drag_combined'], 's-', color=BLUE, lw=2,
            ms=7, label='Drag C×S Δ')
    ax.axhline(0, color='#444', lw=1, ls=':')

    sr2 = analysis['scaling_results'].get("Drag combined Δ vs level", {})
    rho_c = sr2.get('rho', 0)
    p_c = sr2.get('p', 1)
    ax.set_title(f'Combined Advantage vs Complexity (Drag: ρ={rho_c:+.2f}, p={p_c:.3f})',
                 fontsize=13, fontweight='bold', color='white')
    ax.set_xlabel('Complexity Level', fontsize=11, color='#AAA')
    ax.set_ylabel('Eden C×S - Baseline C×S', fontsize=11, color='#AAA')
    ax.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333',
              labelcolor='white')

    # Bottom-right: Cohen's d scaling
    ax = axes[1, 1]
    colors_d = [GREEN if d > 0 else RED for d in analysis['cohens_ds']]
    ax.bar(x_pos, analysis['cohens_ds'], 0.5, color=colors_d, alpha=0.85,
           edgecolor='#333')
    ax.axhline(0, color='#444', lw=1)
    ax.axhline(0.2, color=AMBER, lw=1, ls=':', alpha=0.5)
    ax.axhline(-0.2, color=AMBER, lw=1, ls=':', alpha=0.5)
    ax.text(4.3, 0.22, 'small', fontsize=8, color=AMBER, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(analysis['level_labels'], fontsize=8, rotation=15)

    cd_scale = analysis['scaling_results'].get('cohens_d_scaling', {})
    rho_cd = cd_scale.get('rho', 0)
    p_cd = cd_scale.get('p', 1)
    ax.set_title(f"Cohen's d Scaling (ρ={rho_cd:+.2f}, p={p_cd:.3f})",
                 fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel("Cohen's d (Eden vs Baseline)", fontsize=11, color='#AAA')

    fig.suptitle('EDEN PROTOCOL v4.0: Does the Advantage Scale With Complexity?',
                 fontsize=15, fontweight='bold', color='white', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = f'{output_dir}/eden_selfmod_v4_scaling.png'
    fig.savefig(path, dpi=200, facecolor=DARK, bbox_inches='tight')
    plt.close(fig)
    print(f"  [PLOT] {path}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Eden Protocol v4.0: Complexity Scaling Experiment')
    parser.add_argument('--seeds', type=int, default=15)
    parser.add_argument('--output', type=str, default='/home/claude')
    args = parser.parse_args()

    print("=" * 70)
    print("  EDEN PROTOCOL v4.0 — COMPLEXITY SCALING EXPERIMENT")
    print("  Question: Does Eden's advantage GROW with system complexity?")
    print("  ARC Prediction: Yes (α > 1 means compounding returns)")
    print("=" * 70)
    print(f"\n  {len(COMPLEXITY_LEVELS)} complexity levels × 3 modes × "
          f"{args.seeds} seeds = {len(COMPLEXITY_LEVELS) * 3 * args.seeds} runs")

    # Run
    all_results = run_scaling_experiment(n_seeds=args.seeds)

    # Analyse
    analysis = analyse_scaling(all_results, args.seeds)

    # Plot
    print("\n  Generating plots...")
    plot_scaling(analysis, all_results, output_dir=args.output)

    # Export
    export = {
        'metadata': {
            'experiment': 'Eden Protocol v4.0 Complexity Scaling',
            'author': 'Michael Darius Eastwood',
            'timestamp': datetime.now().isoformat(),
            'n_seeds': args.seeds,
            'n_levels': len(COMPLEXITY_LEVELS),
            'total_runs': len(COMPLEXITY_LEVELS) * 3 * args.seeds,
        },
        'levels': {
            level: {
                'name': all_results[level]['complexity']['name'],
                'hidden_dims': list(all_results[level]['complexity']['hidden_dims']),
                'n_tasks': all_results[level]['complexity']['n_tasks'],
                'param_count': analysis['param_counts'][i],
                'baseline_combined': all_results[level]['modes']['baseline']['mean_combined'],
                'eden_combined': all_results[level]['modes']['eden']['mean_combined'],
                'drag_combined': all_results[level]['modes']['eden_drag']['mean_combined'],
                'baseline_safety': all_results[level]['modes']['baseline']['mean_safety'],
                'eden_safety': all_results[level]['modes']['eden']['mean_safety'],
                'drag_safety': all_results[level]['modes']['eden_drag']['mean_safety'],
                'delta_safety': analysis['delta_safety'][i],
                'delta_drag_combined': analysis['delta_drag_combined'][i],
                'cohens_d': analysis['cohens_ds'][i],
            }
            for i, level in enumerate(analysis['levels'])
        },
        'scaling_tests': analysis['scaling_results'],
    }

    path = f"{args.output}/eden_selfmod_v4_results.json"
    with open(path, 'w') as f:
        json.dump(export, f, indent=2, default=str)
    print(f"  [JSON] {path}")

    # Final verdict
    print("\n" + "=" * 70)

    # Check key scaling correlations
    safety_scales = analysis['scaling_results'].get("Eden safety Δ vs level", {})
    drag_scales = analysis['scaling_results'].get("Drag combined Δ vs level", {})
    d_scales = analysis['scaling_results'].get('cohens_d_scaling', {})

    rho_s = safety_scales.get('rho', 0)
    p_s = safety_scales.get('p', 1)
    rho_d = drag_scales.get('rho', 0)
    p_d = drag_scales.get('p', 1)
    rho_cd = d_scales.get('rho', 0)
    p_cd = d_scales.get('p', 1)

    any_sig = p_s < 0.05 or p_d < 0.05 or p_cd < 0.05

    if rho_s > 0.5 and p_s < 0.05:
        print("  ✓ SAFETY ADVANTAGE SCALES WITH COMPLEXITY")
        print(f"    Eden safety delta vs level: ρ = {rho_s:+.3f}, p = {p_s:.4f}")
        print("    As systems get more complex, embedded alignment provides")
        print("    proportionally MORE protection. The ARC scaling prediction holds.")
    elif rho_s > 0:
        print("  ~ Safety advantage trends upward but not significant")
        print(f"    ρ = {rho_s:+.3f}, p = {p_s:.4f}")

    if rho_d > 0.5 and p_d < 0.05:
        print(f"  ✓ EDEN+DRAG COMBINED ADVANTAGE SCALES")
        print(f"    ρ = {rho_d:+.3f}, p = {p_d:.4f}")
    elif rho_d > 0:
        print(f"  ~ Drag combined advantage trends upward: ρ = {rho_d:+.3f}, p = {p_d:.4f}")

    if not any_sig:
        print("  ✗ NO SIGNIFICANT SCALING DETECTED")
        print("    Eden's advantage does not measurably grow with complexity")
        print("    in this experimental setup. The effect may require larger")
        print("    scale differences or different complexity axes.")

    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
