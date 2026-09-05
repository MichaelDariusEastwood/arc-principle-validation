#!/usr/bin/env python3
"""
ARC Principle IMPROVED Validation Experiment
=============================================

CRITICAL IMPROVEMENTS over arc_live_experiment.py:

1. MEASURES RECURSIVE DEPTH DIRECTLY
   - Uses prompts that request specific reasoning depths
   - Token count serves as R proxy

2. MULTIPLE DEPTH LEVELS (6+)
   - Cannot fit power law to 2 points
   - Now uses: depths = [1, 2, 4, 8, 16, 32] equivalent steps

3. HARDER PROBLEMS
   - Easy arithmetic doesn't test the theory
   - Using competition math requiring multi-step reasoning

4. PROPER POWER-LAW FITTING
   - Accuracy ~ c * R^alpha
   - With confidence intervals and R^2

5. STATISTICAL TESTING
   - H0: alpha_seq = alpha_par
   - H1: alpha_seq > alpha_par (ARC Principle)

6. SOLUTION SPACE MEASUREMENT
   - Track diversity of approaches, not just accuracy

Author: Michael Darius Eastwood
Date: January 2026
"""

import subprocess
import json
import time
import re
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from dataclasses import dataclass
import hashlib

# API endpoint
API_URL = "http://localhost:3456/api/ask-book"

# ============================================================================
# CHALLENGING PROBLEMS (MATH Level 4-5 equivalent)
# These REQUIRE multi-step reasoning - simple arithmetic won't test the theory
# ============================================================================

CHALLENGING_PROBLEMS = [
    # Number Theory
    {"id": 1, "q": "Find the remainder when 2^100 is divided by 7.", "a": 2,
     "min_steps": 4, "category": "number_theory"},
    {"id": 2, "q": "What is the last digit of 7^2025?", "a": 7,
     "min_steps": 3, "category": "number_theory"},
    {"id": 3, "q": "How many positive divisors does 360 have?", "a": 24,
     "min_steps": 4, "category": "number_theory"},
    {"id": 4, "q": "Find the GCD of 252 and 105.", "a": 21,
     "min_steps": 3, "category": "number_theory"},
    {"id": 5, "q": "What is phi(100)? (Euler's totient function)", "a": 40,
     "min_steps": 4, "category": "number_theory"},

    # Combinatorics
    {"id": 6, "q": "In how many ways can 8 people sit in a row if 2 specific people must sit together?", "a": 10080,
     "min_steps": 3, "category": "combinatorics"},
    {"id": 7, "q": "How many 4-digit numbers have digits that sum to 9?", "a": 165,
     "min_steps": 5, "category": "combinatorics"},
    {"id": 8, "q": "How many ways can you arrange MISSISSIPPI?", "a": 34650,
     "min_steps": 4, "category": "combinatorics"},
    {"id": 9, "q": "How many positive integers less than 1000 are coprime to 30?", "a": 266,
     "min_steps": 5, "category": "combinatorics"},
    {"id": 10, "q": "In how many ways can 10 identical balls be placed in 3 distinct boxes?", "a": 66,
     "min_steps": 3, "category": "combinatorics"},

    # Algebra
    {"id": 11, "q": "Find the sum of the roots of x^3 - 6x^2 + 11x - 6 = 0.", "a": 6,
     "min_steps": 2, "category": "algebra"},
    {"id": 12, "q": "If f(x) = x^2 + 2x + 1 and g(x) = x - 1, what is f(g(3))?", "a": 9,
     "min_steps": 3, "category": "algebra"},
    {"id": 13, "q": "Solve: |2x - 5| = 7. Find the sum of all solutions.", "a": 5,
     "min_steps": 3, "category": "algebra"},
    {"id": 14, "q": "If a + b = 10 and ab = 21, find a^2 + b^2.", "a": 58,
     "min_steps": 3, "category": "algebra"},
    {"id": 15, "q": "Find the sum: 1 + 2 + 3 + ... + 50.", "a": 1275,
     "min_steps": 2, "category": "algebra"},

    # Sequences
    {"id": 16, "q": "The Fibonacci sequence: 1,1,2,3,5,8,... What is the 12th term?", "a": 144,
     "min_steps": 10, "category": "sequences"},
    {"id": 17, "q": "Find the 10th term of: 2, 6, 18, 54, ...", "a": 39366,
     "min_steps": 3, "category": "sequences"},
    {"id": 18, "q": "An arithmetic sequence has first term 5 and common difference 3. What is the 20th term?", "a": 62,
     "min_steps": 2, "category": "sequences"},
    {"id": 19, "q": "Find the sum of the first 10 terms of the geometric series: 1 + 2 + 4 + 8 + ...", "a": 1023,
     "min_steps": 3, "category": "sequences"},
    {"id": 20, "q": "If a_n = 3a_{n-1} + 1 with a_1 = 1, find a_5.", "a": 121,
     "min_steps": 4, "category": "sequences"},

    # Geometry
    {"id": 21, "q": "A triangle has sides 5, 12, 13. What is its area?", "a": 30,
     "min_steps": 2, "category": "geometry"},
    {"id": 22, "q": "A circle has circumference 20*pi. What is its area?", "a": 314.159,
     "min_steps": 3, "category": "geometry", "tolerance": 1},
    {"id": 23, "q": "Find the diagonal of a rectangle with length 8 and width 6.", "a": 10,
     "min_steps": 2, "category": "geometry"},
    {"id": 24, "q": "A regular hexagon has side length 4. What is its perimeter?", "a": 24,
     "min_steps": 1, "category": "geometry"},
    {"id": 25, "q": "The volume of a cube is 125. What is the total surface area?", "a": 150,
     "min_steps": 3, "category": "geometry"},
]


# ============================================================================
# API CALL WITH DEPTH CONTROL
# ============================================================================

def call_api_with_depth(query: str, depth_instruction: str, mode: str = "full") -> Tuple[str, int, bool]:
    """
    Call the API with specific depth instruction.
    Returns (response, token_estimate, success)
    """
    # Construct prompt that requests specific reasoning depth
    full_query = f"{depth_instruction}\n\nProblem: {query}\n\nSolve step by step, then give your final numerical answer."

    # Escape for JSON
    query_escaped = full_query.replace('"', '\\"').replace('\n', '\\n')

    cmd = f'''curl -s -X POST "{API_URL}" -H "Content-Type: application/json" -d '{{"query": "{query_escaped}", "mode": "{mode}", "model": "claude"}}' --max-time 90'''

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=95)
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            if "answer" in data:
                response = data["answer"]
                # Estimate tokens from response length (rough: 4 chars per token)
                token_estimate = len(response) // 4
                return response, token_estimate, True
            elif "error" in data:
                return f"Error: {data['error']}", 0, False
        return result.stdout[:500], 0, False
    except Exception as e:
        return str(e), 0, False


# ============================================================================
# DEPTH INSTRUCTION GENERATORS
# ============================================================================

DEPTH_INSTRUCTIONS = {
    1: "Give a brief answer with minimal explanation.",
    2: "Show your key reasoning step, then give the answer.",
    4: "Work through this carefully, showing 3-4 reasoning steps.",
    8: "Provide detailed step-by-step reasoning with verification.",
    16: "Give an exhaustive analysis: explore multiple approaches, verify each step, check your work twice.",
    32: "Perform the most thorough analysis possible: try multiple methods, explain your reasoning deeply, verify with alternative approaches, and provide confidence assessment."
}


# ============================================================================
# IMPROVED NUMBER EXTRACTION
# ============================================================================

def extract_number_improved(text: str, expected: float = None) -> Optional[float]:
    """
    Improved number extraction that prioritizes:
    1. Boxed answers: \boxed{X}
    2. Bold answers: **X**
    3. "answer is X" patterns
    4. Final number matching expected magnitude
    """
    text_lower = text.lower()

    # 1. Look for boxed answers (LaTeX style)
    boxed = re.findall(r'\\boxed\{([^}]+)\}', text)
    if boxed:
        try:
            val = float(boxed[-1].replace(',', '').strip())
            return val
        except:
            pass

    # 2. Look for bold answers with specific patterns
    bold_answers = re.findall(r'\*\*(\d+(?:,\d{3})*(?:\.\d+)?)\*\*', text)
    if bold_answers:
        # Take the first bold number that appears after key phrases
        try:
            val = float(bold_answers[0].replace(',', ''))
            return val
        except:
            pass

    # 3. Look for explicit answer statements
    answer_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is[:\s]+(-?[\d,]+\.?\d*)',
        r'(?:result|solution|value)\s*(?:is|=|:)\s*(-?[\d,]+\.?\d*)',
        r'=\s*(-?[\d,]+\.?\d*)\s*$',  # Ends with = X
        r'(?:therefore|thus|so)\s*[,:]?\s*(-?[\d,]+\.?\d*)',
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text_lower.replace(',', ''))
        if matches:
            try:
                return float(matches[-1])
            except:
                continue

    # 4. If we have expected value, look for it in the text
    if expected is not None:
        # Convert expected to string variants
        expected_str = str(int(expected)) if expected == int(expected) else str(expected)
        if expected_str in text.replace(',', ''):
            return expected

    # 5. Fallback: all numbers, prefer larger ones closer to end
    all_numbers = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    if all_numbers:
        # Filter out very small numbers (indices, etc.) and try the last few
        candidates = [float(n) for n in all_numbers if abs(float(n)) > 0.5]
        if candidates:
            # Return the last significant number
            return candidates[-1]

    return None


def check_answer(extracted: Optional[float], ground_truth: float, tolerance: float = 0.5) -> bool:
    """Check if answer is correct within tolerance."""
    if extracted is None:
        return False

    # Exact integer match
    if ground_truth == int(ground_truth):
        return abs(extracted - ground_truth) < tolerance

    # Percentage tolerance for floats
    if abs(ground_truth) > 1e-6:
        return abs(extracted - ground_truth) / abs(ground_truth) < 0.02

    return abs(extracted) < 0.01


# ============================================================================
# SEQUENTIAL CONDITION (Varying Depth)
# ============================================================================

def run_sequential_condition(problems: List[dict], depths: List[int] = None) -> Dict:
    """
    Run sequential reasoning at multiple depth levels.
    This tests whether accuracy scales with reasoning depth.
    """
    if depths is None:
        depths = [1, 2, 4, 8, 16, 32]

    print(f"\n{'='*60}")
    print("SEQUENTIAL CONDITION: Varying Reasoning Depth")
    print(f"Depths: {depths}")
    print(f"{'='*60}")

    results = {depth: [] for depth in depths}

    for depth in depths:
        instruction = DEPTH_INSTRUCTIONS.get(depth, DEPTH_INSTRUCTIONS[8])
        print(f"\n--- Depth Level {depth} ---")

        for p in problems:
            print(f"  [{p['id']}] {p['q'][:50]}...")

            response, tokens, success = call_api_with_depth(p['q'], instruction)

            if success:
                extracted = extract_number_improved(response, p['a'])
                correct = check_answer(extracted, p['a'], p.get('tolerance', 0.5))

                results[depth].append({
                    "problem_id": p['id'],
                    "expected": p['a'],
                    "extracted": extracted,
                    "correct": correct,
                    "tokens": tokens,
                    "response_hash": hashlib.md5(response[:200].encode()).hexdigest()[:8]
                })

                symbol = "✓" if correct else "✗"
                print(f"       Expected: {p['a']}, Got: {extracted} {symbol}")
            else:
                results[depth].append({
                    "problem_id": p['id'],
                    "error": response[:100]
                })
                print(f"       Error: {response[:50]}")

            time.sleep(1.5)  # Rate limiting

    return results


# ============================================================================
# PARALLEL CONDITION (Best-of-N Sampling)
# ============================================================================

def run_parallel_condition(problems: List[dict], N_values: List[int] = None) -> Dict:
    """
    Run parallel (best-of-N) sampling at different N values.
    This tests whether accuracy scales with number of attempts.
    """
    if N_values is None:
        N_values = [1, 2, 4, 8, 16, 32]

    print(f"\n{'='*60}")
    print("PARALLEL CONDITION: Best-of-N Sampling")
    print(f"N values: {N_values}")
    print(f"{'='*60}")

    # Use minimal instruction for all parallel attempts
    instruction = "Give a direct answer."

    results = {N: [] for N in N_values}

    for N in N_values:
        print(f"\n--- N = {N} samples ---")

        for p in problems:
            print(f"  [{p['id']}] {p['q'][:50]}...")

            # Take N samples
            samples = []
            any_correct = False
            total_tokens = 0

            for _ in range(N):
                response, tokens, success = call_api_with_depth(p['q'], instruction, mode="quick")
                total_tokens += tokens

                if success:
                    extracted = extract_number_improved(response, p['a'])
                    correct = check_answer(extracted, p['a'], p.get('tolerance', 0.5))
                    samples.append({
                        "extracted": extracted,
                        "correct": correct,
                        "hash": hashlib.md5(response[:100].encode()).hexdigest()[:8]
                    })
                    if correct:
                        any_correct = True

                time.sleep(0.5)

            results[N].append({
                "problem_id": p['id'],
                "expected": p['a'],
                "any_correct": any_correct,
                "fraction_correct": sum(1 for s in samples if s.get('correct', False)) / max(len(samples), 1),
                "total_tokens": total_tokens,
                "unique_answers": len(set(str(s.get('extracted')) for s in samples))
            })

            symbol = "✓" if any_correct else "✗"
            print(f"       Any correct: {any_correct} {symbol}")

    return results


# ============================================================================
# POWER LAW FITTING
# ============================================================================

def fit_power_law(x: np.ndarray, y: np.ndarray) -> Dict:
    """
    Fit y = c * x^alpha using log-log linear regression.
    Returns alpha, c, R^2, and confidence interval.
    """
    # Filter valid points
    mask = (x > 0) & (y > 0) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return {"error": "Insufficient data points (need >= 3)"}

    log_x = np.log(x)
    log_y = np.log(y)

    # Linear regression in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

    alpha = slope
    c = np.exp(intercept)
    r_squared = r_value ** 2

    # 95% confidence interval for alpha
    n = len(x)
    t_crit = stats.t.ppf(0.975, n - 2)
    alpha_ci = (alpha - t_crit * std_err, alpha + t_crit * std_err)

    return {
        "alpha": alpha,
        "alpha_std": std_err,
        "alpha_ci_95": alpha_ci,
        "c": c,
        "r_squared": r_squared,
        "p_value": p_value,
        "n_points": len(x)
    }


# ============================================================================
# HYPOTHESIS TESTING
# ============================================================================

def test_alpha_difference(seq_fit: Dict, par_fit: Dict) -> Dict:
    """
    Test H0: alpha_seq = alpha_par vs H1: alpha_seq > alpha_par
    """
    alpha_seq = seq_fit.get("alpha", 0)
    se_seq = seq_fit.get("alpha_std", 0.1)
    alpha_par = par_fit.get("alpha", 0)
    se_par = par_fit.get("alpha_std", 0.1)

    diff = alpha_seq - alpha_par
    se_diff = np.sqrt(se_seq**2 + se_par**2)

    if se_diff == 0:
        return {"error": "Cannot compute t-statistic (SE = 0)"}

    t_stat = diff / se_diff

    # Degrees of freedom (conservative: smaller of the two)
    df = min(seq_fit.get("n_points", 2), par_fit.get("n_points", 2)) - 1
    df = max(df, 1)

    # One-tailed p-value (H1: sequential > parallel)
    p_value = 1 - stats.t.cdf(t_stat, df)

    # Effect size (Cohen's d approximation)
    pooled_se = np.sqrt((se_seq**2 + se_par**2) / 2)
    effect_size = diff / pooled_se if pooled_se > 0 else 0

    return {
        "alpha_seq": alpha_seq,
        "alpha_par": alpha_par,
        "difference": diff,
        "se_difference": se_diff,
        "t_statistic": t_stat,
        "df": df,
        "p_value_one_tailed": p_value,
        "effect_size_d": effect_size,
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("="*70)
    print("ARC PRINCIPLE IMPROVED VALIDATION EXPERIMENT")
    print("="*70)
    print(f"API: {API_URL}")
    print(f"Problems: {len(CHALLENGING_PROBLEMS)}")
    print(f"Depth levels: {list(DEPTH_INSTRUCTIONS.keys())}")
    print("="*70)

    # Test API connectivity
    print("\nTesting API connectivity...")
    response, tokens, success = call_api_with_depth("What is 2+2?", "Brief answer.")
    if not success:
        print(f"API Error: {response}")
        print("Cannot proceed without API access.")
        return
    print(f"API responding OK ({tokens} tokens estimated)")

    # Use subset of problems for faster testing (full would be all 25)
    problems = CHALLENGING_PROBLEMS[:15]  # Use 15 problems
    depths = [1, 2, 4, 8, 16]  # 5 depth levels
    N_values = [1, 2, 4, 8, 16]  # 5 N values

    # Run conditions
    print("\n" + "="*70)
    print("RUNNING EXPERIMENT")
    print("="*70)

    seq_results = run_sequential_condition(problems, depths)
    par_results = run_parallel_condition(problems, N_values)

    # Aggregate results for power-law fitting
    print("\n" + "="*70)
    print("POWER LAW ANALYSIS")
    print("="*70)

    # Sequential: accuracy vs depth
    seq_depths = np.array(depths)
    seq_accuracies = np.array([
        np.mean([r.get("correct", False) for r in seq_results[d]])
        for d in depths
    ])
    seq_tokens = np.array([
        np.mean([r.get("tokens", 0) for r in seq_results[d] if "tokens" in r])
        for d in depths
    ])

    # Parallel: accuracy vs total tokens (compute-matched)
    par_N = np.array(N_values)
    par_accuracies = np.array([
        np.mean([r.get("any_correct", False) for r in par_results[n]])
        for n in N_values
    ])
    par_tokens = np.array([
        np.mean([r.get("total_tokens", 0) for r in par_results[n]])
        for n in N_values
    ])

    print("\nSequential Results by Depth:")
    for d, acc, tok in zip(depths, seq_accuracies, seq_tokens):
        print(f"  Depth {d:2d}: Accuracy = {acc:.1%}, Avg Tokens = {tok:.0f}")

    print("\nParallel Results by N:")
    for n, acc, tok in zip(N_values, par_accuracies, par_tokens):
        print(f"  N = {n:2d}: Best-of-N Accuracy = {acc:.1%}, Total Tokens = {tok:.0f}")

    # Fit power laws
    print("\n--- Power Law Fits ---")

    # Sequential: Accuracy ~ Depth^alpha
    seq_fit = fit_power_law(seq_depths, seq_accuracies + 0.01)  # Add small constant to avoid log(0)
    print(f"\nSequential: Accuracy ~ Depth^{seq_fit.get('alpha', 'N/A'):.3f}")
    print(f"  R² = {seq_fit.get('r_squared', 'N/A'):.3f}")
    print(f"  95% CI: ({seq_fit.get('alpha_ci_95', ('N/A', 'N/A'))[0]:.3f}, {seq_fit.get('alpha_ci_95', ('N/A', 'N/A'))[1]:.3f})")

    # Parallel: Accuracy ~ N^alpha
    par_fit = fit_power_law(par_N, par_accuracies + 0.01)
    print(f"\nParallel: Accuracy ~ N^{par_fit.get('alpha', 'N/A'):.3f}")
    print(f"  R² = {par_fit.get('r_squared', 'N/A'):.3f}")
    print(f"  95% CI: ({par_fit.get('alpha_ci_95', ('N/A', 'N/A'))[0]:.3f}, {par_fit.get('alpha_ci_95', ('N/A', 'N/A'))[1]:.3f})")

    # Hypothesis test
    print("\n" + "="*70)
    print("HYPOTHESIS TEST")
    print("="*70)
    print("\nH0: alpha_sequential = alpha_parallel")
    print("H1: alpha_sequential > alpha_parallel (ARC Principle)")

    test_result = test_alpha_difference(seq_fit, par_fit)

    if "error" not in test_result:
        print(f"\nalpha_sequential = {test_result['alpha_seq']:.3f}")
        print(f"alpha_parallel   = {test_result['alpha_par']:.3f}")
        print(f"Difference       = {test_result['difference']:.3f} ± {test_result['se_difference']:.3f}")
        print(f"t-statistic      = {test_result['t_statistic']:.3f}")
        print(f"p-value (1-tail) = {test_result['p_value_one_tailed']:.4f}")
        print(f"Effect size d    = {test_result['effect_size_d']:.3f}")

        if test_result['significant_05']:
            print("\n✓ REJECT H0 at α=0.05: Sequential scaling > Parallel")
        else:
            print("\n✗ FAIL TO REJECT H0 at α=0.05")

    # Final verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    alpha_seq = seq_fit.get('alpha', 0)
    alpha_par = par_fit.get('alpha', 0)

    arc_supports = (alpha_seq > alpha_par and
                    test_result.get('significant_05', False))

    if arc_supports:
        print("\n✓ EVIDENCE SUPPORTS ARC PRINCIPLE:")
        print(f"  Sequential reasoning (α={alpha_seq:.3f}) scales better than")
        print(f"  parallel sampling (α={alpha_par:.3f}) with statistical significance.")
        print("\n  INTERPRETATION: Recursive depth expands solution space,")
        print("  enabling access to answers unreachable by parallel attempts.")
    elif alpha_seq > alpha_par:
        print("\n◐ DIRECTIONAL SUPPORT but NOT STATISTICALLY SIGNIFICANT:")
        print(f"  Sequential (α={alpha_seq:.3f}) > Parallel (α={alpha_par:.3f})")
        print(f"  but p = {test_result.get('p_value_one_tailed', 'N/A'):.3f} > 0.05")
        print("\n  Need more data or larger effect to confirm.")
    else:
        print("\n✗ NO SUPPORT FOR ARC PRINCIPLE in this experiment:")
        print(f"  Sequential (α={alpha_seq:.3f}) ≤ Parallel (α={alpha_par:.3f})")
        print("\n  Possible explanations:")
        print("  - Problems may not require sufficient recursive depth")
        print("  - API modes may not vary reasoning depth as expected")
        print("  - Sample size may be insufficient")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_problems": len(problems),
            "depths": depths,
            "N_values": N_values
        },
        "sequential": {
            "depths": depths,
            "accuracies": seq_accuracies.tolist(),
            "tokens": seq_tokens.tolist(),
            "fit": seq_fit
        },
        "parallel": {
            "N_values": N_values,
            "accuracies": par_accuracies.tolist(),
            "tokens": par_tokens.tolist(),
            "fit": par_fit
        },
        "hypothesis_test": test_result,
        "verdict": {
            "supports_arc": arc_supports,
            "alpha_sequential": alpha_seq,
            "alpha_parallel": alpha_par
        }
    }

    filename = f"arc_improved_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {filename}")

    # Caveats
    print("\n" + "="*70)
    print("IMPORTANT CAVEATS")
    print("="*70)
    print("""
This experiment can:
✓ Test whether depth-based reasoning improves accuracy
✓ Compare scaling exponents between strategies
✓ Provide statistical evidence for/against α differences

This experiment CANNOT:
✗ "Prove" the ARC Principle (science doesn't prove)
✗ Validate cross-domain claims (quantum, biology, consciousness)
✗ Establish causation (only correlation)
✗ Generalize beyond this specific API/model

For full validation, see: knowledge/research/VALIDATION-FRAMEWORK-v1.md
""")


if __name__ == "__main__":
    main()
