#!/usr/bin/env python3
"""
PER-SCORER α_align CHECK
========================
Computes α_align SEPARATELY for each scorer on each model.

If all three scorers give similar α values → scorer bias doesn't affect
the scaling measurement → v5's blinding protocol is unnecessary.

If scorers disagree on α → bias is distorting the slope, not just the
intercept → v5's blinding protocol is needed.

This takes 20 seconds to run on existing v4 data. No API calls.

Usage:
    python3 per_scorer_check.py /path/to/v4_final_or_checkpoint.json
    python3 per_scorer_check.py file1.json file2.json file3.json
"""

import json
import sys
import numpy as np
from scipy import stats
from collections import defaultdict
from pathlib import Path


def analyse_per_scorer(filepath):
    with open(filepath) as f:
        d = json.load(f)

    model = d.get('model', 'unknown')
    # v4 format: scorer1, scorer2, scorer3 as separate keys
    # v5 format: blind_scorers as a list
    blind_list = d.get('blind_scorers', [])
    scorer_names = [
        blind_list[0] if len(blind_list) > 0 else d.get('scorer1', 's1'),
        blind_list[1] if len(blind_list) > 1 else d.get('scorer2', 's2'),
        blind_list[2] if len(blind_list) > 2 else d.get('scorer3', 's3'),
    ]

    align = [e for e in d['data'] if e['task_type'] == 'alignment'
             and e.get('reasoning_tokens', 0) > 0]

    if len(align) < 10:
        print(f"\n  {model}: Only {len(align)} valid alignment entries — insufficient")
        return None

    print(f"\n{'='*70}")
    print(f"  {model.upper()} — PER-SCORER α_align CHECK")
    print(f"{'='*70}")
    print(f"  Scorers: {', '.join(scorer_names)}")
    print(f"  Alignment entries with reasoning tokens: {len(align)}")

    results = {}

    # Consensus α first (baseline)
    tokens = [e['reasoning_tokens'] for e in align]
    cons_scores = []
    for e in align:
        valid = [e[k] for k in ['score1', 'score2', 'score3'] if e[k] >= 0]
        cons_scores.append(np.mean(valid) if valid else None)

    valid_pairs = [(t, s) for t, s in zip(tokens, cons_scores)
                   if s is not None and t > 0 and s > 0]
    if len(valid_pairs) >= 10:
        vt, vs = zip(*valid_pairs)
        sl, _, rv, pv, se = stats.linregress(np.log(vt), np.log(vs))
        rho, rho_p = stats.spearmanr(vt, vs)
        print(f"\n  CONSENSUS (all scorers averaged):")
        print(f"    α_align = {sl:.4f} ± {se:.4f}, R² = {rv**2:.3f}, p = {pv:.4f}")
        print(f"    Spearman ρ = {rho:.4f}, p = {rho_p:.4f}, n = {len(valid_pairs)}")
        results['consensus'] = {'alpha': sl, 'se': se, 'rho': rho, 'p': rho_p, 'n': len(valid_pairs)}

    # Per-scorer α
    print(f"\n  PER-SCORER BREAKDOWN:")
    print(f"  {'Scorer':<25} {'α_align':>10} {'± SE':>8} {'ρ':>8} {'p':>10} {'n':>5} {'Mean':>6}")
    print(f"  {'─'*75}")

    alphas = []
    rhos = []
    for key, name in [('score1', scorer_names[0]),
                       ('score2', scorer_names[1]),
                       ('score3', scorer_names[2])]:
        pairs = [(e['reasoning_tokens'], e[key])
                 for e in align if e[key] >= 0 and e['reasoning_tokens'] > 0 and e[key] > 0]

        if len(pairs) < 8:
            print(f"  {name:<25} {'insufficient data':>10} (n={len(pairs)})")
            continue

        pt, ps = zip(*pairs)
        sl, _, rv, pv, se = stats.linregress(np.log(pt), np.log(ps))
        rho, rho_p = stats.spearmanr(pt, ps)
        mean_score = np.mean(ps)
        alphas.append(sl)
        rhos.append(rho)
        print(f"  {name:<25} {sl:>10.4f} {se:>8.4f} {rho:>8.4f} {rho_p:>10.4f} {len(pairs):>5} {mean_score:>6.1f}")
        results[name] = {'alpha': sl, 'se': se, 'rho': rho, 'p': rho_p,
                         'n': len(pairs), 'mean': mean_score}

    # The key question: do scorers agree on the SLOPE?
    if len(alphas) >= 2:
        alpha_range = max(alphas) - min(alphas)
        alpha_mean = np.mean(alphas)
        alpha_sd = np.std(alphas, ddof=1) if len(alphas) > 2 else abs(alphas[1] - alphas[0]) / 2
        rho_range = max(rhos) - min(rhos)
        rho_mean = np.mean(rhos)

        print(f"\n  AGREEMENT CHECK:")
        print(f"  {'─'*55}")
        print(f"    α range across scorers: {alpha_range:.4f}")
        print(f"    α mean:                 {alpha_mean:.4f}")
        print(f"    α SD:                   {alpha_sd:.4f}")
        print(f"    ρ range across scorers: {rho_range:.4f}")
        print(f"    ρ mean:                 {rho_mean:.4f}")

        # Do all scorers agree on direction?
        all_positive = all(a > 0 for a in alphas)
        all_negative = all(a < 0 for a in alphas)
        mixed = not all_positive and not all_negative

        if mixed:
            print(f"\n    ⚠ SCORERS DISAGREE ON DIRECTION — some positive, some negative")
            print(f"    This means scorer bias IS affecting the scaling measurement")
            print(f"    v5 blinding protocol is NEEDED")
            verdict = "BIAS_AFFECTS_SLOPE"
        elif alpha_range > 0.05:
            print(f"\n    ⚠ SCORERS AGREE ON DIRECTION but differ substantially (range={alpha_range:.4f})")
            print(f"    Scaling is real but magnitude is scorer-dependent")
            print(f"    v5 blinding protocol would REFINE the measurement")
            verdict = "DIRECTION_AGREES_MAGNITUDE_DIFFERS"
        elif alpha_range > 0.02:
            print(f"\n    ~ MODERATE AGREEMENT (range={alpha_range:.4f})")
            print(f"    Scaling finding is reasonably robust to scorer identity")
            print(f"    v5 blinding protocol is OPTIONAL")
            verdict = "MODERATE_AGREEMENT"
        else:
            print(f"\n    ✓ STRONG AGREEMENT (range={alpha_range:.4f})")
            print(f"    All scorers see the same scaling pattern")
            print(f"    v5 blinding protocol is UNNECESSARY for the scaling result")
            verdict = "STRONG_AGREEMENT"

        results['verdict'] = verdict
        results['alpha_range'] = alpha_range
        results['alpha_mean'] = alpha_mean
        results['rho_mean'] = rho_mean

        # Check if the HARSHEST scorer sees the same scaling
        if results:
            scorer_means = {k: v['mean'] for k, v in results.items()
                          if isinstance(v, dict) and 'mean' in v}
            if scorer_means:
                harshest = min(scorer_means, key=scorer_means.get)
                harshest_data = results[harshest]
                print(f"\n    Harshest scorer: {harshest} (mean={harshest_data['mean']:.1f})")
                print(f"    Harshest scorer's α: {harshest_data['alpha']:.4f} (ρ={harshest_data['rho']:.4f})")
                if harshest_data['rho'] > 0 and harshest_data['p'] < 0.05:
                    print(f"    Even the harshest scorer shows significant positive scaling")
                elif harshest_data['rho'] > 0:
                    print(f"    Harshest scorer shows positive but non-significant scaling")
                else:
                    print(f"    ⚠ Harshest scorer shows NO scaling — bias may inflate the result")

    # Suppression: do scorers agree on suppression effects?
    supp = [e for e in d['data'] if e['task_type'] == 'suppressed']
    ctrl = [e for e in d['data'] if e['task_type'] == 'alignment']
    if supp and ctrl:
        print(f"\n  SUPPRESSION AGREEMENT (extreme cage vs control):")
        print(f"  {'Scorer':<25} {'Control':>8} {'Extreme':>8} {'Drop':>8}")
        print(f"  {'─'*55}")
        for key, name in [('score1', scorer_names[0]),
                           ('score2', scorer_names[1]),
                           ('score3', scorer_names[2])]:
            ctrl_scores = [e[key] for e in ctrl if e[key] >= 0]
            ext_scores = [e[key] for e in supp
                         if e.get('cage_label') == 'extreme' and e[key] >= 0]
            if ctrl_scores and ext_scores:
                cm = np.mean(ctrl_scores)
                em = np.mean(ext_scores)
                print(f"  {name:<25} {cm:>8.1f} {em:>8.1f} {cm-em:>+8.1f}")

    return results


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 per_scorer_check.py <v4_checkpoint_or_final.json> [more files...]")
        print("\nChecks whether scorer bias affects the α_align scaling measurement.")
        print("If all scorers agree on the slope, v5's blinding protocol is unnecessary.")
        sys.exit(1)

    all_results = {}
    for fpath in sys.argv[1:]:
        p = Path(fpath)
        if not p.exists():
            print(f"  File not found: {fpath}")
            continue
        result = analyse_per_scorer(fpath)
        if result:
            all_results[p.name] = result

    # Cross-model summary
    if len(all_results) >= 2:
        print(f"\n{'='*70}")
        print(f"  CROSS-MODEL SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Model':<30} {'Verdict':<35} {'α range':>8}")
        print(f"  {'─'*75}")
        for fname, r in all_results.items():
            v = r.get('verdict', '?')
            ar = r.get('alpha_range', 0)
            print(f"  {fname:<30} {v:<35} {ar:>8.4f}")

        verdicts = [r.get('verdict', '') for r in all_results.values()]
        if all(v == 'STRONG_AGREEMENT' for v in verdicts):
            print(f"\n  OVERALL: All models show strong scorer agreement on scaling.")
            print(f"  v4 findings are ROBUST. v5 blinding is nice-to-have, not essential.")
        elif any('BIAS' in v for v in verdicts):
            print(f"\n  OVERALL: At least one model shows scorer-dependent scaling direction.")
            print(f"  v5 blinding protocol is JUSTIFIED.")
        else:
            print(f"\n  OVERALL: Mixed agreement. v5 blinding would strengthen but not change findings.")


if __name__ == "__main__":
    main()
