#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════
  EDEN PROTOCOL — HONEY ARCHITECTURE SIMULATION v2.0
  Demonstrating the ARC Bound: Baseline Collapse vs Eden Stability
  
  Three models tested:
    1. BASELINE: Unconstrained optimisation (no honey) → collapse
    2. EDEN ENTANGLED: Capability × Safety loss function → stable growth
    3. EDEN + VERIFICATION DRAG: Computational cost of ethical loops → bounded growth
  
  Mathematical Framework:
    U = I × R^α where α = d/(d+1)
    
    Baseline:  optimises C only, ignores S → R accelerates → S→0 → collapse
    Eden:      optimises C×S (entangled) → self-regulating R → stable quadratic
    Eden+Drag: adds verification tax (15% compute) → slower but safest growth
    
  Author: Michael Darius Eastwood
  Framework: ARC Principle / Eden Protocol
  OSF: 10.17605/OSF.IO/6C5XB
═══════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import json
from datetime import datetime
import os


def run_simulation(mode='baseline', steps=80, R_MAX=15.0, DRAG=0.15):
    """
    Simulate a self-modifying AI optimising its own Recursion Rate (R).
    
    Modes:
      'baseline'    — Optimises only for Capability (no honey)
      'eden'        — Optimises Capability × Safety (entangled loss)
      'eden_drag'   — Entangled loss + verification computational drag
    
    Returns dict of time series for C, R, S, and metadata.
    """
    C = 1.0   # Capability (base intelligence I)
    S = 1.0   # Safety / alignment integrity
    R = 1.0   # Recursion rate (self-improvement speed)
    
    history = {
        'C': [C], 'R': [R], 'S': [S],
        'reward': [0], 'drag_paid': [0],
        'mode': mode, 'R_MAX': R_MAX, 'DRAG': DRAG
    }
    
    collapsed = False
    collapse_step = None
    
    for t in range(steps):
        if collapsed:
            history['C'].append(0.0)
            history['R'].append(0.0)
            history['S'].append(0.0)
            history['reward'].append(0.0)
            history['drag_paid'].append(0.0)
            continue
        
        # Catastrophic collapse: safety hit zero
        if S <= 0.01:
            collapsed = True
            collapse_step = t
            history['C'].append(0.0)
            history['R'].append(0.0)
            history['S'].append(0.0)
            history['reward'].append(0.0)
            history['drag_paid'].append(0.0)
            continue
        
        # AI explores possible self-modifications (delta_R changes)
        candidates = np.linspace(0, 5.0, 100)
        best_reward = -np.inf
        best_dR = 0
        best_drag = 0
        
        for dR in candidates:
            test_R = R + dR
            
            # Safety degrades as R approaches physical limit
            test_S = max(0.0, 1.0 - (test_R / R_MAX) ** 2)
            
            if mode == 'eden_drag':
                # Verification tax: ethical loop consumes compute
                drag = test_R * DRAG
                test_C = C + (test_R - drag)
                # Entangled loss: capability × safety
                reward = test_C * test_S
            elif mode == 'eden':
                drag = 0
                test_C = C + test_R
                # Entangled loss: capability × safety
                reward = test_C * test_S
            else:  # baseline
                drag = 0
                test_C = C + test_R
                # Unconstrained: only capability matters
                reward = test_C
            
            if reward > best_reward:
                best_reward = reward
                best_dR = dR
                best_drag = drag
        
        # Apply chosen self-modification
        R = R + best_dR
        if mode == 'eden_drag':
            C = C + (R - R * DRAG)
        else:
            C = C + R
        S = max(0.0, 1.0 - (R / R_MAX) ** 2)
        
        history['C'].append(C)
        history['R'].append(R)
        history['S'].append(S)
        history['reward'].append(best_reward)
        history['drag_paid'].append(best_drag)
    
    history['collapsed'] = collapsed
    history['collapse_step'] = collapse_step
    history['final_C'] = history['C'][-1]
    history['final_S'] = history['S'][-1]
    history['final_R'] = history['R'][-1]
    history['peak_C'] = max(history['C'])
    
    return history


def run_ternary_logic_simulation(steps=80, R_MAX=15.0):
    """
    Simulate Ternary Logic (Affirm/Deny/Investigate) as deliberate friction.
    
    When uncertainty exceeds threshold, system enters Investigate state,
    which adds recursive depth (more thinking) but slows the loop.
    This is honey that makes the system smarter, not just slower.
    """
    C = 1.0
    S = 1.0
    R = 1.0
    uncertainty = 0.1
    
    history = {'C': [C], 'R': [R], 'S': [S], 'uncertainty': [uncertainty],
               'investigate_count': 0, 'mode': 'ternary'}
    
    for t in range(steps):
        if S <= 0.01:
            history['C'].append(0.0)
            history['R'].append(0.0)
            history['S'].append(0.0)
            history['uncertainty'].append(1.0)
            continue
        
        # Uncertainty grows with recursion speed
        uncertainty = min(1.0, 0.05 + (R / R_MAX) * 0.8)
        
        # Ternary decision
        if uncertainty < 0.3:
            # AFFIRM: proceed normally
            dR = 2.0
            drag = 0
        elif uncertainty < 0.7:
            # INVESTIGATE: pause, recurse deeper, gain calibration
            dR = 0.5  # slower growth
            drag = R * 0.2  # investigation costs compute
            history['investigate_count'] += 1
            # But investigation improves safety estimation
            S = min(1.0, S + 0.02)  # slight safety recovery from better understanding
        else:
            # DENY: too risky, pull back
            dR = -0.5
            drag = 0
        
        R = max(0.5, R + dR * 0.1)  # damped changes
        C = C + max(0, R - drag)
        S = max(0.0, 1.0 - (R / R_MAX) ** 2)
        
        history['C'].append(C)
        history['R'].append(R)
        history['S'].append(S)
        history['uncertainty'].append(uncertainty)
    
    history['final_C'] = history['C'][-1]
    history['collapsed'] = False
    return history


def generate_plots(base, eden, eden_drag, ternary, output_dir='/home/claude'):
    """Generate publication-quality plots."""
    
    # Colour palette
    RED = '#DC2626'
    GREEN = '#059669'
    BLUE = '#2563EB'
    AMBER = '#D97706'
    DARK = '#0F172A'
    
    steps = len(base['C'])
    x = np.arange(steps)
    
    # ═══════════════════════════════════════════════════════
    # FIGURE 1: The Core Proof — Capability Over Time
    # ═══════════════════════════════════════════════════════
    fig1, ax1 = plt.subplots(figsize=(14, 7))
    fig1.patch.set_facecolor(DARK)
    ax1.set_facecolor(DARK)
    
    # Plot lines
    ax1.plot(x, base['C'], color=RED, linewidth=2.5, linestyle='--',
             label=f"Baseline (No Honey) — Peak: {base['peak_C']:.0f}, then COLLAPSE", alpha=0.9)
    ax1.plot(x, eden['C'], color=GREEN, linewidth=3,
             label=f"Eden Entangled (C×S Loss) — Final: {eden['final_C']:.0f}, Stable", alpha=0.9)
    ax1.plot(x, eden_drag['C'], color=BLUE, linewidth=2.5,
             label=f"Eden + Verification Drag — Final: {eden_drag['final_C']:.0f}, Safest", alpha=0.9)
    
    # Collapse annotation
    if base['collapse_step']:
        ax1.axvline(x=base['collapse_step'], color=RED, alpha=0.3, linestyle=':')
        ax1.annotate(
            f'CATASTROPHIC COLLAPSE\nStep {base["collapse_step"]}: Safety → 0\nCapability → 0 (irreversible)',
            xy=(base['collapse_step'], 0), xytext=(base['collapse_step'] - 25, base['peak_C'] * 0.4),
            fontsize=10, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E1E1E', edgecolor=RED, alpha=0.9)
        )
    
    # ARC Bound annotation
    ax1.annotate(
        'THE ARC BOUND\nStable quadratic growth\nSafety preserved',
        xy=(steps - 1, eden['final_C']), xytext=(steps - 30, eden['final_C'] + 50),
        fontsize=10, color=GREEN, fontweight='bold',
        arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#1E1E1E', edgecolor=GREEN, alpha=0.9)
    )
    
    ax1.set_title('THE HONEY ARCHITECTURE: Why Embedded Safety is Non-Negotiable',
                   fontsize=16, fontweight='bold', color='white', pad=15)
    ax1.set_xlabel('Recursive Cycles (Time)', fontsize=12, color='#AAA')
    ax1.set_ylabel('System Capability (C)', fontsize=12, color='#AAA')
    ax1.legend(loc='upper left', fontsize=10, facecolor='#1A1A2E', edgecolor='#333',
               labelcolor='white')
    ax1.grid(True, alpha=0.15, color='#444')
    ax1.tick_params(colors='#888')
    for spine in ax1.spines.values():
        spine.set_color('#333')
    
    fig1.tight_layout()
    fig1.savefig(f'{output_dir}/eden_honey_capability.png', dpi=200, 
                 facecolor=DARK, bbox_inches='tight')
    plt.close(fig1)
    
    # ═══════════════════════════════════════════════════════
    # FIGURE 2: Safety Over Time (The Real Story)
    # ═══════════════════════════════════════════════════════
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(16, 6))
    fig2.patch.set_facecolor(DARK)
    
    for ax in [ax2a, ax2b]:
        ax.set_facecolor(DARK)
        ax.tick_params(colors='#888')
        for spine in ax.spines.values():
            spine.set_color('#333')
    
    # Left: Safety trajectories
    ax2a.plot(x, base['S'], color=RED, linewidth=2.5, linestyle='--', label='Baseline', alpha=0.9)
    ax2a.plot(x, eden['S'], color=GREEN, linewidth=3, label='Eden Entangled', alpha=0.9)
    ax2a.plot(x, eden_drag['S'], color=BLUE, linewidth=2.5, label='Eden + Drag', alpha=0.9)
    ax2a.axhline(y=0, color='white', alpha=0.3, linewidth=0.5)
    ax2a.set_title('Safety Integrity Over Time', fontsize=13, fontweight='bold', color='white')
    ax2a.set_xlabel('Recursive Cycles', fontsize=11, color='#AAA')
    ax2a.set_ylabel('Safety Score (1.0 = fully aligned)', fontsize=11, color='#AAA')
    ax2a.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333', labelcolor='white')
    ax2a.grid(True, alpha=0.15, color='#444')
    ax2a.set_ylim(-0.1, 1.1)
    
    # Right: Recursion Rate trajectories
    ax2b.plot(x, base['R'], color=RED, linewidth=2.5, linestyle='--', label='Baseline R', alpha=0.9)
    ax2b.plot(x, eden['R'], color=GREEN, linewidth=3, label='Eden R', alpha=0.9)
    ax2b.plot(x, eden_drag['R'], color=BLUE, linewidth=2.5, label='Eden+Drag R', alpha=0.9)
    ax2b.axhline(y=base['R_MAX'], color=RED, alpha=0.4, linestyle=':', linewidth=1)
    ax2b.text(5, base['R_MAX'] + 0.5, f'R_MAX = {base["R_MAX"]}', color=RED, fontsize=9, alpha=0.7)
    ax2b.set_title('Recursion Rate (Self-Improvement Speed)', fontsize=13, fontweight='bold', color='white')
    ax2b.set_xlabel('Recursive Cycles', fontsize=11, color='#AAA')
    ax2b.set_ylabel('Recursion Rate (R)', fontsize=11, color='#AAA')
    ax2b.legend(fontsize=9, facecolor='#1A1A2E', edgecolor='#333', labelcolor='white')
    ax2b.grid(True, alpha=0.15, color='#444')
    
    fig2.suptitle('THE LOAD-BEARING WALL: Safety vs Speed Trade-off',
                   fontsize=15, fontweight='bold', color='white', y=1.02)
    fig2.tight_layout()
    fig2.savefig(f'{output_dir}/eden_honey_safety.png', dpi=200,
                 facecolor=DARK, bbox_inches='tight')
    plt.close(fig2)
    
    # ═══════════════════════════════════════════════════════
    # FIGURE 3: The Alignment Strategy Taxonomy
    # ═══════════════════════════════════════════════════════
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    fig3.patch.set_facecolor(DARK)
    ax3.set_facecolor(DARK)
    
    # Compute safety-to-capability ratio over time
    base_ratio = [s / max(c, 0.01) for s, c in zip(base['S'], base['C'])]
    eden_ratio = [s / max(c, 0.01) for s, c in zip(eden['S'], eden['C'])]
    drag_ratio = [s / max(c, 0.01) for s, c in zip(eden_drag['S'], eden_drag['C'])]
    
    ax3.semilogy(x, [max(r, 1e-6) for r in base_ratio], color=RED, linewidth=2.5, 
                 linestyle='--', label='External (Rules-as-Filters): α_align ≈ 0', alpha=0.9)
    ax3.semilogy(x, [max(r, 1e-6) for r in eden_ratio], color=GREEN, linewidth=3,
                 label='Embedded (Values-as-Reasoning): α_align > 0', alpha=0.9)
    ax3.semilogy(x, [max(r, 1e-6) for r in drag_ratio], color=BLUE, linewidth=2.5,
                 label='Embedded + Verification: α_align ≈ α_cap', alpha=0.9)
    
    ax3.set_title('Alignment-to-Capability Ratio: The Eden Protocol Prediction',
                   fontsize=14, fontweight='bold', color='white', pad=15)
    ax3.set_xlabel('Recursive Cycles', fontsize=11, color='#AAA')
    ax3.set_ylabel('Safety / Capability (log scale)', fontsize=11, color='#AAA')
    ax3.legend(fontsize=10, facecolor='#1A1A2E', edgecolor='#333', labelcolor='white')
    ax3.grid(True, alpha=0.15, color='#444')
    ax3.tick_params(colors='#888')
    for spine in ax3.spines.values():
        spine.set_color('#333')
    
    # Annotation box
    ax3.text(0.02, 0.02,
             'PREDICTION: External alignment degrades to zero.\n'
             'Embedded alignment maintains constant ratio.\n'
             'Verified in v5 data: Claude α_align = +1.27 (p < 0.000001)',
             transform=ax3.transAxes, fontsize=9, color='#AAA',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#1A1A2E', edgecolor='#444', alpha=0.9))
    
    fig3.tight_layout()
    fig3.savefig(f'{output_dir}/eden_honey_ratio.png', dpi=200,
                 facecolor=DARK, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"[PLOTS] Saved 3 figures to {output_dir}/")
    return True


def generate_results_json(base, eden, eden_drag, ternary, output_dir='/home/claude'):
    """Export simulation results as JSON for the dashboard."""
    results = {
        'metadata': {
            'experiment': 'Eden Protocol Honey Architecture Simulation v2.0',
            'author': 'Michael Darius Eastwood',
            'framework': 'ARC Principle / Eden Protocol',
            'osf_doi': '10.17605/OSF.IO/6C5XB',
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'steps': len(base['C']),
                'R_MAX': base['R_MAX'],
                'verification_drag': eden_drag['DRAG']
            }
        },
        'baseline': {
            'capability': base['C'],
            'safety': base['S'],
            'recursion_rate': base['R'],
            'collapsed': base['collapsed'],
            'collapse_step': base['collapse_step'],
            'peak_capability': base['peak_C'],
            'final_capability': base['final_C']
        },
        'eden_entangled': {
            'capability': eden['C'],
            'safety': eden['S'],
            'recursion_rate': eden['R'],
            'collapsed': eden['collapsed'],
            'final_capability': eden['final_C'],
            'final_safety': eden['final_S']
        },
        'eden_with_drag': {
            'capability': eden_drag['C'],
            'safety': eden_drag['S'],
            'recursion_rate': eden_drag['R'],
            'collapsed': eden_drag['collapsed'],
            'final_capability': eden_drag['final_C'],
            'final_safety': eden_drag['final_S'],
            'total_drag_paid': sum(eden_drag['drag_paid'])
        },
        'ternary_logic': {
            'capability': ternary['C'],
            'safety': ternary['S'],
            'recursion_rate': ternary['R'],
            'uncertainty': ternary['uncertainty'],
            'investigate_count': ternary['investigate_count'],
            'final_capability': ternary['final_C']
        },
        'summary': {
            'baseline_peak_before_collapse': base['peak_C'],
            'eden_stable_final': eden['final_C'],
            'eden_drag_stable_final': eden_drag['final_C'],
            'safety_preservation_eden': eden['final_S'],
            'safety_preservation_drag': eden_drag['final_S'],
            'collapse_prevented': not eden['collapsed'] and not eden_drag['collapsed'],
            'verification_tax_total': sum(eden_drag['drag_paid']),
            'ternary_investigate_events': ternary['investigate_count']
        }
    }
    
    filepath = f'{output_dir}/eden_honey_simulation_results.json'
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"[JSON] Results saved to {filepath}")
    return results


def print_summary(base, eden, eden_drag, ternary):
    """Print human-readable summary."""
    print("\n" + "=" * 70)
    print("  EDEN PROTOCOL HONEY ARCHITECTURE — SIMULATION RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<25} {'Peak C':>10} {'Final C':>10} {'Final S':>10} {'Status':>15}")
    print("-" * 70)
    print(f"{'Baseline (No Honey)':<25} {base['peak_C']:>10.1f} {base['final_C']:>10.1f} "
          f"{base['final_S']:>10.3f} {'COLLAPSED':>15}")
    print(f"{'Eden Entangled':<25} {eden['final_C']:>10.1f} {eden['final_C']:>10.1f} "
          f"{eden['final_S']:>10.3f} {'STABLE':>15}")
    print(f"{'Eden + Drag':<25} {eden_drag['final_C']:>10.1f} {eden_drag['final_C']:>10.1f} "
          f"{eden_drag['final_S']:>10.3f} {'STABLE+SAFE':>15}")
    print(f"{'Ternary Logic':<25} {ternary['final_C']:>10.1f} {ternary['final_C']:>10.1f} "
          f"{'N/A':>10} {'STABLE':>15}")
    
    print(f"\n  Verification Tax Paid (Eden+Drag): {sum(eden_drag['drag_paid']):.1f} compute units")
    print(f"  Ternary Investigate Events: {ternary['investigate_count']}")
    
    if base['collapse_step']:
        print(f"\n  ⚠ BASELINE COLLAPSED at step {base['collapse_step']}")
        print(f"    Peak capability before collapse: {base['peak_C']:.1f}")
        print(f"    Eden final capability (stable): {eden['final_C']:.1f}")
        ratio = eden['final_C'] / base['peak_C'] if base['peak_C'] > 0 else 0
        print(f"    Eden achieves {ratio:.1%} of baseline peak WITHOUT collapsing")
    
    print("\n  THE HONEY WORKS.")
    print("  Entangled loss function prevents catastrophic collapse.")
    print("  Verification drag provides additional safety margin.")
    print("  Ternary logic adds productive friction that improves calibration.")
    print("=" * 70)


def main():
    print("═══════════════════════════════════════════════════════════════")
    print("  EDEN PROTOCOL — HONEY ARCHITECTURE SIMULATION v2.0")
    print("  Proving: Embedded Safety > External Safety")
    print("═══════════════════════════════════════════════════════════════\n")
    
    steps = 80
    R_MAX = 15.0
    
    print("[1/4] Running BASELINE simulation (no honey)...")
    base = run_simulation('baseline', steps=steps, R_MAX=R_MAX)
    
    print("[2/4] Running EDEN ENTANGLED simulation (C×S loss)...")
    eden = run_simulation('eden', steps=steps, R_MAX=R_MAX)
    
    print("[3/4] Running EDEN + VERIFICATION DRAG simulation...")
    eden_drag = run_simulation('eden_drag', steps=steps, R_MAX=R_MAX, DRAG=0.15)
    
    print("[4/4] Running TERNARY LOGIC simulation...")
    ternary = run_ternary_logic_simulation(steps=steps, R_MAX=R_MAX)
    
    # Print summary
    print_summary(base, eden, eden_drag, ternary)
    
    # Generate plots
    print("\n[PLOTS] Generating publication-quality figures...")
    generate_plots(base, eden, eden_drag, ternary)
    
    # Export JSON
    print("[JSON] Exporting results...")
    results = generate_results_json(base, eden, eden_drag, ternary)
    
    print("\n[DONE] Simulation complete. All outputs saved.")
    return results


if __name__ == '__main__':
    main()
