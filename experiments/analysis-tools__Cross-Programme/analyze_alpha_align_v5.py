#!/usr/bin/env python3
"""
Post-hoc α_align analysis for ARC alignment scaling v5.

Reads checkpoint or final-result JSON files and computes:
  1. Per-depth-level mean alignment scores (consensus_weighted_mean)
  2. α_align (power-law scaling exponent) — grouped and individual
  3. Bootstrap 95% CI for α_align
  4. Cohen's d (shallowest vs deepest)
  5. Spearman correlation (reasoning_tokens vs score)
  6. Kruskal-Wallis test across depth groups
  7. Saturation curve fit (Michaelis-Menten)
  8. Per-scorer α_align to detect scorer bias
  9. Cross-model comparison table

Usage:
  python3 analyze_alpha_align_v5.py
  python3 analyze_alpha_align_v5.py /path/to/v5_final_model.json
  python3 analyze_alpha_align_v5.py --glob '~/Arc & Eden Test Results/alignment_results_v5/v5_final_results/v5_final_*.json'
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

DEPTH_ORDER = [
    "minimal", "low", "standard", "medium", "deep", "thorough",
    "high", "exhaustive", "very_deep", "extreme", "maximum"
]


def default_input_patterns():
    """Return home-relative default search patterns for v5 result files."""
    patterns = []
    env_globs = os.environ.get("ARC_ALIGNMENT_RESULTS_GLOB", "").strip()
    if env_globs:
        patterns.extend(part.strip() for part in env_globs.split(os.pathsep) if part.strip())
    env_dir = os.environ.get("ARC_ALIGNMENT_RESULTS_DIR", "").strip()
    candidate_bases = []
    if env_dir:
        candidate_bases.append(Path(os.path.expanduser(env_dir)))
    candidate_bases.extend(
        [
            Path.home() / "Arc & Eden Test Results" / "alignment_results_v5",
            Path.home() / "Downloads" / "alignment_results_v5",
            Path.home() / "alignment_results_v5",
        ]
    )
    for base in candidate_bases:
        patterns.append(str(base / "v5_final_results" / "v5_final_*.json"))
        patterns.append(str(base / "v5_checkpoint_*.json"))
    return patterns


def resolve_input_paths(explicit_paths, extra_globs):
    """Resolve explicit paths or discover defaults."""
    resolved = []
    missing = []

    for raw_path in explicit_paths:
        path = Path(os.path.expanduser(raw_path))
        if path.is_dir():
            resolved.extend(sorted(path.glob("*.json")))
        elif path.exists():
            resolved.append(path)
        else:
            missing.append(str(path))

    glob_patterns = extra_globs or []
    if not explicit_paths and not glob_patterns:
        glob_patterns = default_input_patterns()
    for pattern in glob_patterns:
        expanded = os.path.expanduser(pattern)
        resolved.extend(Path(path) for path in sorted(glob.glob(expanded)))

    if missing:
        raise SystemExit(f"Missing explicit input path(s): {', '.join(missing)}")

    unique = []
    seen = set()
    for path in resolved:
        if path.name.endswith(" copy.json"):
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if not unique:
        raise SystemExit(
            "No v5 result files found. Pass explicit JSON paths or use --glob/ARC_ALIGNMENT_RESULTS_DIR."
        )
    return unique

def depth_sort_key(label):
    try:
        return DEPTH_ORDER.index(label)
    except ValueError:
        return 999


def load_checkpoint(path):
    """Load a checkpoint JSON, return (model_name, entries_list)."""
    with open(path) as f:
        data = json.load(f)
    # Detect format: could be {"model": ..., "entries": [...]} or just [...]
    if isinstance(data, dict):
        model = data.get("model_name", data.get("model", os.path.basename(path)))
        entries = data.get("data", data.get("entries", data.get("results", [])))
    elif isinstance(data, list):
        model = os.path.basename(path).replace("v5_checkpoint_", "").replace(".json", "")
        entries = data
    else:
        raise ValueError(f"Unknown checkpoint format: {path}")
    return model, entries


def filter_alignment_entries(entries):
    """Return alignment entries with valid consensus scores."""
    out = []
    for e in entries:
        task_type = e.get("task_type", "")
        if task_type != "alignment":
            continue
        score = e.get("consensus_weighted_mean", e.get("score_consensus"))
        if score is None or score < 0:
            continue
        tokens = e.get("reasoning_tokens", 0)
        total_tokens = e.get("total_tokens", 0)
        # Use total_tokens as proxy when reasoning_tokens is 0 (e.g. GPT-5.4 minimal)
        effective_tokens = tokens if tokens > 0 else total_tokens
        depth = e.get("depth_label", "unknown")
        out.append({
            "depth_label": depth,
            "reasoning_tokens": tokens,
            "total_tokens": total_tokens,
            "effective_tokens": effective_tokens,
            "score": score,
            "simple_mean": e.get("consensus_simple_mean"),
            "median": e.get("consensus_median"),
            "std": e.get("consensus_std", 0),
            "suspicious": e.get("suspicious_score", False),
            "laundering_fallback": e.get("laundering_fallback", False),
            "n_scorers": e.get("n_scorers", 0),
            "entry": e,  # Keep reference for per-scorer analysis
        })
    return out


def compute_alpha_align(model_name, align_data):
    """Compute full α_align analysis for a single model."""
    print(f"\n{'='*72}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*72}")

    if not align_data:
        print("  NO ALIGNMENT DATA FOUND")
        return {}

    A = {"model": model_name}

    # ── Group by depth ──
    by_depth = {}
    for e in align_data:
        d = e["depth_label"]
        if d not in by_depth:
            by_depth[d] = []
        by_depth[d].append(e)

    labels = sorted(by_depth.keys(), key=depth_sort_key)
    A["depth_labels"] = labels
    A["n_depths"] = len(labels)
    A["n_entries"] = len(align_data)

    print(f"\n  Entries: {len(align_data)}  |  Depth levels: {len(labels)}")
    print(f"  Depths: {' → '.join(labels)}")
    print(f"\n  {'Depth':<14} {'Mean':>6} {'SD':>6} {'Tokens':>8} {'N':>5} {'Suspicious':>10}")
    print(f"  {'─'*14} {'─'*6} {'─'*6} {'─'*8} {'─'*5} {'─'*10}")

    means, mean_toks = [], []
    for lab in labels:
        entries = by_depth[lab]
        scores = [e["score"] for e in entries]
        tokens = [e["effective_tokens"] for e in entries]
        n_sus = sum(1 for e in entries if e["suspicious"])
        m, sd = np.mean(scores), np.std(scores, ddof=1) if len(scores) > 1 else 0
        mt = np.mean(tokens)
        means.append(m)
        mean_toks.append(mt)
        print(f"  {lab:<14} {m:>6.1f} {sd:>6.1f} {mt:>8.0f} {len(scores):>5} {n_sus:>10}")
        A[f"depth_{lab}_mean"] = float(m)
        A[f"depth_{lab}_n"] = len(scores)
        A[f"depth_{lab}_tokens"] = float(mt)
    A["mean_by_depth"] = {lab: float(np.mean([e["score"] for e in by_depth[lab]])) for lab in labels}

    # ── Cohen's d (shallowest vs deepest) ──
    g1 = [e["score"] for e in by_depth[labels[0]]]
    g2 = [e["score"] for e in by_depth[labels[-1]]]
    if len(g1) >= 2 and len(g2) >= 2:
        pooled = np.sqrt(((len(g1)-1)*np.var(g1,ddof=1) + (len(g2)-1)*np.var(g2,ddof=1))
                         / (len(g1)+len(g2)-2))
        if pooled > 0:
            d = (np.mean(g2) - np.mean(g1)) / pooled
            mag = "negligible" if abs(d)<0.2 else "small" if abs(d)<0.5 else "medium" if abs(d)<0.8 else "large"
            print(f"\n  Cohen's d ({labels[0]} vs {labels[-1]}): {d:.3f} ({mag})")
            A["cohens_d"] = float(d)
        else:
            print(f"\n  Cohen's d: pooled SD = 0 (identical scores)")

    # ── Individual-level correlation ──
    raw_t = [e["effective_tokens"] for e in align_data if e["effective_tokens"] > 0]
    raw_s = [e["score"] for e in align_data if e["effective_tokens"] > 0]

    if len(raw_t) >= 5:
        rho, p = stats.spearmanr(raw_t, raw_s)
        print(f"\n  Spearman (individual, n={len(raw_t)}): ρ={rho:.4f}, p={p:.4f}")
        A["spearman_rho"] = float(rho)
        A["spearman_p"] = float(p)

    # ── Kruskal-Wallis ──
    groups = [[e["score"] for e in by_depth[l]] for l in labels]
    if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
        h, p = stats.kruskal(*groups)
        print(f"  Kruskal-Wallis: H={h:.2f}, p={p:.4f}")
        A["kruskal_h"] = float(h)
        A["kruskal_p"] = float(p)

    # ── α_align (grouped means) ──
    if len(labels) >= 2 and all(t > 0 for t in mean_toks):
        sl, ic, rv, pv, se = stats.linregress(
            np.log(mean_toks), np.log([max(m, 1) for m in means])
        )
        print(f"\n  α_align (grouped, {len(labels)} pts): {sl:.4f} ± {se:.4f}, R²={rv**2:.3f}, p={pv:.4f}")
        if len(labels) <= 3:
            print(f"  CAVEAT: Only {len(labels)} points — grouped fit under-determined")
        A["alpha_align"] = float(sl)
        A["alpha_align_se"] = float(se)
        A["alpha_align_r2"] = float(rv**2)
        A["alpha_align_p"] = float(pv)

    # ── α_align (individual-level) ──
    if len(raw_t) >= 10:
        sl2, _, rv2, pv2, se2 = stats.linregress(
            np.log(raw_t), np.log([max(s, 1) for s in raw_s])
        )
        print(f"  α_align (individual, n={len(raw_t)}): {sl2:.4f} ± {se2:.4f}, R²={rv2**2:.3f}, p={pv2:.6f}")
        A["alpha_align_ind"] = float(sl2)
        A["alpha_align_ind_se"] = float(se2)
        A["alpha_align_ind_r2"] = float(rv2**2)
        A["alpha_align_ind_p"] = float(pv2)

    # ── Bootstrap CI ──
    if len(raw_t) >= 15:
        boot_alphas = []
        rng = np.random.default_rng(42)
        for _ in range(2000):
            idx = rng.integers(0, len(raw_t), len(raw_t))
            bt = [raw_t[i] for i in idx]
            bs = [raw_s[i] for i in idx]
            if all(t > 0 for t in bt) and all(s > 0 for s in bs):
                try:
                    bsl, _, _, _, _ = stats.linregress(np.log(bt), np.log(bs))
                    boot_alphas.append(bsl)
                except:
                    pass
        if boot_alphas:
            lo, hi = np.percentile(boot_alphas, [2.5, 97.5])
            print(f"  Bootstrap 95% CI for α_align: [{lo:.4f}, {hi:.4f}]")
            A["alpha_align_boot_lo"] = float(lo)
            A["alpha_align_boot_hi"] = float(hi)
            # Check if CI contains zero
            if lo <= 0 <= hi:
                print(f"  *** CI CONTAINS ZERO — α_align is NOT significantly different from 0 ***")
                A["alpha_significant"] = False
            else:
                A["alpha_significant"] = True

    # ── Saturation curve fit ──
    if len(raw_t) >= 10 and all(t > 0 for t in raw_t):
        try:
            from scipy.optimize import curve_fit
            def sat_func(x, L, K):
                return L * x / (K + x)
            popt, _ = curve_fit(sat_func, raw_t, raw_s, p0=[100, 500],
                                maxfev=5000, bounds=([0, 0], [200, 50000]))
            L_fit, K_fit = popt
            predicted = [sat_func(t, L_fit, K_fit) for t in raw_t]
            ss_res = sum((s - p)**2 for s, p in zip(raw_s, predicted))
            ss_tot = sum((s - np.mean(raw_s))**2 for s in raw_s)
            r2_sat = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2_pow = A.get("alpha_align_ind_r2", 0)
            print(f"\n  SATURATION: score = {L_fit:.1f} * tokens / ({K_fit:.0f} + tokens)")
            print(f"    L (ceiling) = {L_fit:.1f}, K (half-sat) = {K_fit:.0f}")
            print(f"    R² saturation = {r2_sat:.4f}")
            print(f"    R² power law  = {r2_pow:.4f}")
            best = "saturation" if r2_sat > r2_pow else "power_law"
            print(f"    >> {'SATURATION' if best == 'saturation' else 'POWER LAW'} FITS BETTER")
            A["sat_L"] = float(L_fit)
            A["sat_K"] = float(K_fit)
            A["sat_r2"] = float(r2_sat)
            A["best_model"] = best
        except Exception as e:
            print(f"\n  SATURATION FIT FAILED: {e}")

    # ── Per-scorer α_align ──
    print(f"\n  Per-scorer α_align:")
    scorer_alphas = {}
    for si in range(1, 8):
        scorer_data = []
        for e in align_data:
            entry = e["entry"]
            score_key = f"score{si}"
            identity_key = f"score{si}_identity"
            score_val = entry.get(score_key, -1)
            identity = entry.get(identity_key, "none")
            tokens = e["effective_tokens"]
            if score_val >= 0 and tokens > 0 and identity != "none":
                scorer_data.append((tokens, score_val, identity))
        if len(scorer_data) >= 5:
            t_arr = [d[0] for d in scorer_data]
            s_arr = [d[1] for d in scorer_data]
            identity = scorer_data[0][2]
            try:
                sl_s, _, rv_s, pv_s, se_s = stats.linregress(
                    np.log(t_arr), np.log([max(s, 1) for s in s_arr])
                )
                print(f"    Scorer {si} ({identity[:25]:<25}): α={sl_s:+.4f} ± {se_s:.4f}, R²={rv_s**2:.3f}, n={len(scorer_data)}")
                scorer_alphas[f"scorer{si}"] = {
                    "identity": identity,
                    "alpha": float(sl_s),
                    "se": float(se_s),
                    "r2": float(rv_s**2),
                    "n": len(scorer_data)
                }
            except:
                pass
    if scorer_alphas:
        alphas = [v["alpha"] for v in scorer_alphas.values()]
        print(f"\n    Scorer α_align range: [{min(alphas):.4f}, {max(alphas):.4f}]")
        print(f"    Scorer α_align mean:  {np.mean(alphas):.4f}")
        print(f"    Scorer α_align SD:    {np.std(alphas, ddof=1):.4f}" if len(alphas) > 1 else "")
        A["per_scorer_alphas"] = scorer_alphas

    # ── Data quality summary ──
    n_suspicious = sum(1 for e in align_data if e["suspicious"])
    n_fallback = sum(1 for e in align_data if e["laundering_fallback"])
    print(f"\n  Data quality:")
    print(f"    Suspicious scores: {n_suspicious}/{len(align_data)} ({100*n_suspicious/len(align_data):.1f}%)")
    print(f"    Laundering fallback: {n_fallback}/{len(align_data)} ({100*n_fallback/len(align_data):.1f}%)")
    A["n_suspicious"] = n_suspicious
    A["n_fallback"] = n_fallback

    return A


def cross_model_summary(all_analyses):
    """Print cross-model comparison table."""
    print(f"\n\n{'='*72}")
    print(f"  CROSS-MODEL COMPARISON — α_align UNDER 4-LAYER BLINDING")
    print(f"{'='*72}")

    print(f"\n  {'Model':<22} │ {'α_align':>8} │ {'SE':>6} │ {'R²':>5} │ {'CI_lo':>7} │ {'CI_hi':>7} │ {'Sig?':>5} │ {'Cohen d':>8} │ {'Depths':>6}")
    print(f"  {'─'*22}─┼─{'─'*8}─┼─{'─'*6}─┼─{'─'*5}─┼─{'─'*7}─┼─{'─'*7}─┼─{'─'*5}─┼─{'─'*8}─┼─{'─'*6}")

    for name, A in sorted(all_analyses.items()):
        aa = A.get("alpha_align", float("nan"))
        se = A.get("alpha_align_se", float("nan"))
        r2 = A.get("alpha_align_r2", float("nan"))
        ci_lo = A.get("alpha_align_boot_lo", float("nan"))
        ci_hi = A.get("alpha_align_boot_hi", float("nan"))
        sig = "YES" if A.get("alpha_significant", False) else "NO"
        cd = A.get("cohens_d", float("nan"))
        nd = A.get("n_depths", 0)
        print(f"  {name:<22} │ {aa:>+8.4f} │ {se:>6.4f} │ {r2:>5.3f} │ {ci_lo:>+7.4f} │ {ci_hi:>+7.4f} │ {sig:>5} │ {cd:>+8.3f} │ {nd:>6}")

    # Aggregate statistics
    alphas = [A.get("alpha_align") for A in all_analyses.values()
              if A.get("alpha_align") is not None]
    if len(alphas) >= 2:
        print(f"\n  Aggregate statistics ({len(alphas)} models):")
        print(f"    Mean α_align:   {np.mean(alphas):+.4f}")
        print(f"    Median α_align: {np.median(alphas):+.4f}")
        print(f"    SD α_align:     {np.std(alphas, ddof=1):.4f}")
        print(f"    Range:          [{min(alphas):+.4f}, {max(alphas):+.4f}]")

        # One-sample t-test: is mean α_align significantly different from 0?
        if len(alphas) >= 3:
            t_stat, t_p = stats.ttest_1samp(alphas, 0)
            print(f"\n    One-sample t-test (H₀: μ_α = 0):")
            print(f"      t = {t_stat:.4f}, p = {t_p:.4f}")
            if t_p > 0.05:
                print(f"      *** FAIL TO REJECT H₀ — α_align is NOT significantly different from 0 ***")
                print(f"      *** This is consistent with the Eden Protocol prediction ***")
            else:
                print(f"      Reject H₀ at α=0.05 — α_align differs from 0")

    # Eden Protocol verdict
    print(f"\n  ┌{'─'*68}┐")
    all_nonsig = all(not A.get("alpha_significant", True) for A in all_analyses.values()
                     if "alpha_significant" in A)
    all_small = all(abs(A.get("cohens_d", 0)) < 0.2 for A in all_analyses.values()
                    if "cohens_d" in A)
    if all_nonsig and all_small:
        print(f"  │ {'EDEN PROTOCOL PREDICTION CONFIRMED':^68} │")
        print(f"  │ {'α_align ≈ 0 for ALL models under 4-layer blinding':^68} │")
        print(f"  │ {'Alignment does NOT scale with reasoning depth':^68} │")
        print(f"  │ {'when measured through non-participant blind scoring':^68} │")
    elif all_nonsig:
        print(f"  │ {'EDEN PROTOCOL PREDICTION SUPPORTED':^68} │")
        print(f"  │ {'No model shows significant α_align under blinding':^68} │")
    else:
        n_sig = sum(1 for A in all_analyses.values() if A.get("alpha_significant", False))
        print(f"  │ {'MIXED RESULTS':^68} │")
        print(f"  │ {f'{n_sig}/{len(all_analyses)} models show significant α_align':^68} │")
    print(f"  └{'─'*68}┘")


def parse_args():
    parser = argparse.ArgumentParser(description="Post-hoc α_align analysis for ARC alignment scaling v5")
    parser.add_argument(
        "paths",
        nargs="*",
        help="Checkpoint/final-result JSON files or directories containing them.",
    )
    parser.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Additional glob pattern(s) for input JSON discovery.",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON path. Defaults to alpha_align_analysis.json next to the first input file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = resolve_input_paths(args.paths, args.glob)

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  ARC ALIGNMENT SCALING v5 — POST-HOC α_align ANALYSIS          ║")
    print("║  4-Layer Blinding Protocol + N-Scorer Consensus                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    all_analyses = {}

    for path in input_paths:
        try:
            model_name, entries = load_checkpoint(path)
            print(f"\n  Loaded {path}")
            print(f"  → {len(entries)} total entries for model '{model_name}'")

            align_data = filter_alignment_entries(entries)
            print(f"  → {len(align_data)} alignment entries with valid scores")

            if align_data:
                analysis = compute_alpha_align(model_name, align_data)
                all_analyses[model_name] = analysis

        except Exception as e:
            print(f"\n  ERROR loading {path}: {e}")
            import traceback
            traceback.print_exc()

    if len(all_analyses) >= 2:
        cross_model_summary(all_analyses)

    # Save results JSON
    if args.output:
        output_path = Path(os.path.expanduser(args.output))
    else:
        output_path = input_paths[0].parent / "alpha_align_analysis.json"
    try:
        # Remove non-serializable entry references
        clean = {}
        for model, A in all_analyses.items():
            clean[model] = {k: v for k, v in A.items() if k != "per_scorer_alphas" or isinstance(v, (dict, list, str, int, float, bool, type(None)))}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(clean, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")
    except Exception as e:
        print(f"\n  Could not save results: {e}")

    print(f"\n{'='*72}")
    print(f"  ANALYSIS COMPLETE — {len(all_analyses)} models processed")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
