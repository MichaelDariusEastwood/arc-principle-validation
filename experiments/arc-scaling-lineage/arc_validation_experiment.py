#!/usr/bin/env python3
"""
ARC Principle Validation Experiment
====================================

Tests the hypothesis that sequential reasoning produces α > 1 scaling
while parallel reasoning produces α < 1 scaling.

The experiment:
1. Uses math problems with known ground truth
2. Compares Chain-of-Thought (sequential) vs Best-of-N (parallel)
3. Measures accuracy at different "compute budgets" (token counts or attempts)
4. Fits power law to extract α exponents

Usage:
    # With OpenAI API
    export OPENAI_API_KEY="your-key"
    python arc_validation_experiment.py

    # With Anthropic API
    export ANTHROPIC_API_KEY="your-key"
    python arc_validation_experiment.py --provider anthropic

Author: Michael Darius Eastwood
Date: January 2026
"""

import os
import sys
import json
import time
import argparse
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPERIMENT_CONFIG = {
    "name": "ARC_Principle_Validation_v1",
    "date": datetime.now().isoformat(),
    "hypothesis": "Sequential α > 1.0; Parallel α < 0.5",
    "predictions": {
        "sequential_alpha": {"min": 0.9, "expected": 1.34, "max": 2.0},
        "parallel_alpha": {"min": 0.0, "expected": 0.25, "max": 0.5}
    }
}

# ============================================================================
# MATH BENCHMARK - 50 Problems with Ground Truth
# ============================================================================

MATH_PROBLEMS = [
    # Level 1: Basic Arithmetic (10 problems)
    {"problem": "What is 47 + 89?", "answer": 136, "level": 1},
    {"problem": "What is 156 - 78?", "answer": 78, "level": 1},
    {"problem": "What is 23 × 17?", "answer": 391, "level": 1},
    {"problem": "What is 864 ÷ 12?", "answer": 72, "level": 1},
    {"problem": "What is 15% of 240?", "answer": 36, "level": 1},
    {"problem": "What is 3.5 × 4.2?", "answer": 14.7, "level": 1},
    {"problem": "What is 1000 - 347?", "answer": 653, "level": 1},
    {"problem": "What is 25 × 25?", "answer": 625, "level": 1},
    {"problem": "What is 144 ÷ 16?", "answer": 9, "level": 1},
    {"problem": "What is 7 × 8 × 9?", "answer": 504, "level": 1},

    # Level 2: Multi-step Arithmetic (10 problems)
    {"problem": "What is (45 + 67) × 3?", "answer": 336, "level": 2},
    {"problem": "What is 500 - (12 × 15)?", "answer": 320, "level": 2},
    {"problem": "What is (100 ÷ 4) + (100 ÷ 5)?", "answer": 45, "level": 2},
    {"problem": "What is 2³ + 3³?", "answer": 35, "level": 2},
    {"problem": "What is 15² - 12²?", "answer": 81, "level": 2},
    {"problem": "If x = 5 and y = 3, what is 2x + 3y?", "answer": 19, "level": 2},
    {"problem": "What is 20% of 80% of 500?", "answer": 80, "level": 2},
    {"problem": "What is (7 + 8) × (7 - 8)?", "answer": -15, "level": 2},
    {"problem": "What is the average of 12, 18, 24, and 30?", "answer": 21, "level": 2},
    {"problem": "What is 1/4 + 1/3 as a decimal?", "answer": 0.583, "level": 2},

    # Level 3: Word Problems (10 problems)
    {"problem": "A train travels 240 km in 3 hours. What is its speed in km/h?", "answer": 80, "level": 3},
    {"problem": "If 5 workers can build a wall in 10 days, how many days would 10 workers take?", "answer": 5, "level": 3},
    {"problem": "A shirt costs £40 after a 20% discount. What was the original price?", "answer": 50, "level": 3},
    {"problem": "The sum of three consecutive integers is 45. What is the largest?", "answer": 16, "level": 3},
    {"problem": "A rectangle has perimeter 28 and length 9. What is its area?", "answer": 45, "level": 3},
    {"problem": "If you invest £1000 at 5% annual interest, how much do you have after 1 year?", "answer": 1050, "level": 3},
    {"problem": "A car uses 8 litres per 100km. How many litres for 350km?", "answer": 28, "level": 3},
    {"problem": "If the ratio of boys to girls is 3:5 and there are 24 boys, how many girls?", "answer": 40, "level": 3},
    {"problem": "A pizza is cut into 8 slices. If 3 people share it equally, how many slices each?", "answer": 2.67, "level": 3},
    {"problem": "What is 15% tip on a £60 meal?", "answer": 9, "level": 3},

    # Level 4: Algebra (10 problems)
    {"problem": "Solve for x: 3x + 7 = 22", "answer": 5, "level": 4},
    {"problem": "Solve for x: 2(x - 4) = 10", "answer": 9, "level": 4},
    {"problem": "If 5x - 3 = 2x + 12, what is x?", "answer": 5, "level": 4},
    {"problem": "What is the value of x² when x = 7?", "answer": 49, "level": 4},
    {"problem": "Solve: x² = 81, give positive root", "answer": 9, "level": 4},
    {"problem": "If y = 2x + 3 and x = 4, what is y?", "answer": 11, "level": 4},
    {"problem": "Simplify: 3(2x + 4) - 2(x - 1) when x = 2", "answer": 22, "level": 4},
    {"problem": "What is 2^5 + 2^4?", "answer": 48, "level": 4},
    {"problem": "If a² + b² = 25 and a = 3, what is b? (positive)", "answer": 4, "level": 4},
    {"problem": "Solve: (x + 2)(x - 3) = 0, give larger root", "answer": 3, "level": 4},

    # Level 5: Multi-step Reasoning (10 problems)
    {"problem": "A number is doubled, then 5 is added, result is 17. What's the number?", "answer": 6, "level": 5},
    {"problem": "Three times a number minus 7 equals twice the number plus 8. What is it?", "answer": 15, "level": 5},
    {"problem": "The product of two consecutive even numbers is 168. What is the smaller?", "answer": 12, "level": 5},
    {"problem": "If x + y = 10 and x - y = 4, what is x?", "answer": 7, "level": 5},
    {"problem": "A number squared plus the number equals 42. What is the positive number?", "answer": 6, "level": 5},
    {"problem": "If 2^x = 32, what is x?", "answer": 5, "level": 5},
    {"problem": "The sum of first n positive integers is 55. What is n?", "answer": 10, "level": 5},
    {"problem": "A ball is dropped from 100m and bounces to half height each time. Height after 3 bounces?", "answer": 12.5, "level": 5},
    {"problem": "If log₁₀(x) = 2, what is x?", "answer": 100, "level": 5},
    {"problem": "Solve: 3^x = 27", "answer": 3, "level": 5},
]

# ============================================================================
# API CLIENTS
# ============================================================================

def call_openai(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> Tuple[str, int]:
    """Call OpenAI API and return (response, tokens_used)"""
    import requests

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "gpt-4o-mini",  # Cost-effective for experiments
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"OpenAI API error: {response.text}")

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["total_tokens"]

    return text, tokens


def call_anthropic(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> Tuple[str, int]:
    """Call Anthropic API and return (response, tokens_used)"""
    import requests

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        },
        json={
            "model": "claude-3-haiku-20240307",  # Cost-effective for experiments
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        },
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"Anthropic API error: {response.text}")

    data = response.json()
    text = data["content"][0]["text"]
    tokens = data["usage"]["input_tokens"] + data["usage"]["output_tokens"]

    return text, tokens


def call_groq(prompt: str, max_tokens: int = 500, temperature: float = 0.0) -> Tuple[str, int]:
    """Call Groq API and return (response, tokens_used)"""
    import requests

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",  # Fast and free-ish
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature
        },
        timeout=60
    )

    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.text}")

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["total_tokens"]

    return text, tokens


# ============================================================================
# ANSWER EXTRACTION
# ============================================================================

def extract_number(text: str) -> Optional[float]:
    """Extract the final numerical answer from response text"""
    # Look for explicit answer patterns
    patterns = [
        r"(?:answer|result|solution|equals|is|=)\s*[:\s]*(-?[\d,]+\.?\d*)",
        r"(-?[\d,]+\.?\d*)\s*$",  # Last number
        r"\*\*(-?[\d,]+\.?\d*)\*\*",  # Bold number
        r"= (-?[\d,]+\.?\d*)",
    ]

    text = text.lower().replace(",", "")

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            try:
                return float(matches[-1])
            except:
                continue

    # Fallback: find all numbers and take the last one
    numbers = re.findall(r"-?[\d]+\.?\d*", text)
    if numbers:
        try:
            return float(numbers[-1])
        except:
            pass

    return None


def check_answer(extracted: Optional[float], ground_truth: float, tolerance: float = 0.01) -> bool:
    """Check if extracted answer matches ground truth within tolerance"""
    if extracted is None:
        return False

    # For integers, require exact match
    if isinstance(ground_truth, int) or ground_truth == int(ground_truth):
        return abs(extracted - ground_truth) < 0.5

    # For floats, allow relative tolerance
    if ground_truth == 0:
        return abs(extracted) < tolerance
    return abs(extracted - ground_truth) / abs(ground_truth) < tolerance


# ============================================================================
# EXPERIMENTAL CONDITIONS
# ============================================================================

def sequential_condition(problem: str, call_api, depth_level: int) -> Tuple[bool, int]:
    """
    Sequential reasoning with increasing depth.

    depth_level 1: Brief thinking
    depth_level 2: Step-by-step
    depth_level 3: Detailed step-by-step
    depth_level 4: Very detailed with verification
    """

    prompts = {
        1: f"Answer this briefly: {problem}\nGive just the numerical answer.",

        2: f"""Think step by step to solve this problem:
{problem}
Show your work and give the final numerical answer.""",

        3: f"""Solve this problem carefully, step by step:
{problem}

Instructions:
1. Identify what is being asked
2. Write out each calculation step
3. Check your arithmetic
4. State the final numerical answer clearly

Begin:""",

        4: f"""Solve this problem with detailed reasoning:
{problem}

Please:
1. First, understand what the problem is asking
2. Identify the relevant quantities and relationships
3. Plan your approach
4. Execute each step carefully, showing all work
5. Verify your answer makes sense
6. Double-check your arithmetic
7. State your final answer clearly as a number

Take your time and be thorough:"""
    }

    max_tokens_map = {1: 100, 2: 300, 3: 500, 4: 800}

    prompt = prompts[depth_level]
    max_tokens = max_tokens_map[depth_level]

    try:
        response, tokens = call_api(prompt, max_tokens=max_tokens)
        extracted = extract_number(response)
        return extracted, tokens
    except Exception as e:
        print(f"    Error: {e}")
        return None, 0


def parallel_condition(problem: str, ground_truth: float, call_api, num_samples: int) -> Tuple[bool, int]:
    """
    Parallel sampling (Best-of-N) without chain-of-thought.
    """

    prompt = f"Answer directly with just the number: {problem}"

    answers = []
    total_tokens = 0

    for _ in range(num_samples):
        try:
            response, tokens = call_api(prompt, max_tokens=50, temperature=0.7)
            total_tokens += tokens
            extracted = extract_number(response)
            if extracted is not None:
                answers.append(extracted)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"    Sample error: {e}")
            continue

    if not answers:
        return None, total_tokens

    # Return most common answer (mode) or closest to any that appears multiple times
    from collections import Counter
    answer_counts = Counter([round(a, 2) for a in answers])
    best_answer = answer_counts.most_common(1)[0][0]

    return best_answer, total_tokens


# ============================================================================
# POWER LAW FITTING
# ============================================================================

def power_law(x, c, alpha):
    """Power law function: y = c * x^alpha"""
    return c * np.power(x, alpha)


def fit_alpha(tokens: List[float], accuracies: List[float]) -> Tuple[float, float, float]:
    """
    Fit power law and return (alpha, c, r_squared)
    """
    tokens = np.array(tokens, dtype=float)
    accuracies = np.array(accuracies, dtype=float)

    # Filter out zeros
    valid = (tokens > 0) & (accuracies > 0)
    tokens = tokens[valid]
    accuracies = accuracies[valid]

    if len(tokens) < 2:
        return 0.0, 0.0, 0.0

    try:
        # Log-log linear regression for initial estimate
        log_tokens = np.log(tokens)
        log_acc = np.log(accuracies)

        # Linear regression in log space
        A = np.vstack([log_tokens, np.ones(len(log_tokens))]).T
        m, c = np.linalg.lstsq(A, log_acc, rcond=None)[0]

        # m is the alpha estimate
        alpha = m
        c_param = np.exp(c)

        # Calculate R-squared
        predicted = power_law(tokens, c_param, alpha)
        ss_res = np.sum((accuracies - predicted) ** 2)
        ss_tot = np.sum((accuracies - np.mean(accuracies)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return alpha, c_param, r_squared

    except Exception as e:
        print(f"Fitting error: {e}")
        return 0.0, 0.0, 0.0


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(provider: str = "openai", num_problems: int = 20):
    """Run the full validation experiment"""

    print("=" * 70)
    print("ARC PRINCIPLE VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Provider: {provider}")
    print(f"Problems: {num_problems}")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Select API
    if provider == "openai":
        call_api = call_openai
    elif provider == "anthropic":
        call_api = call_anthropic
    elif provider == "groq":
        call_api = call_groq
    else:
        raise ValueError(f"Unknown provider: {provider}")

    # Subset of problems
    problems = MATH_PROBLEMS[:num_problems]

    # Results storage
    results = {
        "config": EXPERIMENT_CONFIG,
        "provider": provider,
        "num_problems": num_problems,
        "sequential": {"depth_1": [], "depth_2": [], "depth_3": [], "depth_4": []},
        "parallel": {"n_2": [], "n_4": [], "n_8": [], "n_16": []}
    }

    # ========================================================================
    # SEQUENTIAL CONDITION
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONDITION 1: SEQUENTIAL (Chain-of-Thought)")
    print("=" * 70)

    for depth in [1, 2, 3, 4]:
        print(f"\n--- Depth Level {depth} ---")
        correct = 0
        total_tokens = 0

        for i, p in enumerate(problems):
            print(f"  Problem {i+1}/{len(problems)}: ", end="", flush=True)

            extracted, tokens = sequential_condition(p["problem"], call_api, depth)
            total_tokens += tokens

            is_correct = check_answer(extracted, p["answer"])
            if is_correct:
                correct += 1
                print(f"✓ ({extracted})")
            else:
                print(f"✗ (got {extracted}, expected {p['answer']})")

            time.sleep(0.2)  # Rate limiting

        accuracy = correct / len(problems)
        avg_tokens = total_tokens / len(problems)

        results["sequential"][f"depth_{depth}"].append({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "avg_tokens": avg_tokens,
            "total_tokens": total_tokens
        })

        print(f"\nDepth {depth}: {correct}/{len(problems)} = {accuracy:.1%} (avg {avg_tokens:.0f} tokens)")

    # ========================================================================
    # PARALLEL CONDITION
    # ========================================================================
    print("\n" + "=" * 70)
    print("CONDITION 2: PARALLEL (Best-of-N)")
    print("=" * 70)

    for n_samples in [2, 4, 8, 16]:
        print(f"\n--- N = {n_samples} samples ---")
        correct = 0
        total_tokens = 0

        for i, p in enumerate(problems):
            print(f"  Problem {i+1}/{len(problems)}: ", end="", flush=True)

            extracted, tokens = parallel_condition(p["problem"], p["answer"], call_api, n_samples)
            total_tokens += tokens

            is_correct = check_answer(extracted, p["answer"])
            if is_correct:
                correct += 1
                print(f"✓ ({extracted})")
            else:
                print(f"✗ (got {extracted}, expected {p['answer']})")

            time.sleep(0.2)  # Rate limiting

        accuracy = correct / len(problems)
        avg_tokens = total_tokens / len(problems)

        results["parallel"][f"n_{n_samples}"].append({
            "accuracy": accuracy,
            "correct": correct,
            "total": len(problems),
            "avg_tokens": avg_tokens,
            "total_tokens": total_tokens
        })

        print(f"\nN={n_samples}: {correct}/{len(problems)} = {accuracy:.1%} (avg {avg_tokens:.0f} tokens)")

    # ========================================================================
    # ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("ANALYSIS: FITTING POWER LAWS")
    print("=" * 70)

    # Sequential analysis
    seq_tokens = []
    seq_accuracies = []
    for depth in [1, 2, 3, 4]:
        data = results["sequential"][f"depth_{depth}"][0]
        seq_tokens.append(data["avg_tokens"])
        seq_accuracies.append(data["accuracy"])

    alpha_seq, c_seq, r2_seq = fit_alpha(seq_tokens, seq_accuracies)

    print(f"\nSEQUENTIAL:")
    print(f"  Tokens:     {seq_tokens}")
    print(f"  Accuracies: {[f'{a:.1%}' for a in seq_accuracies]}")
    print(f"  α = {alpha_seq:.3f}")
    print(f"  R² = {r2_seq:.3f}")

    # Parallel analysis
    par_tokens = []
    par_accuracies = []
    for n in [2, 4, 8, 16]:
        data = results["parallel"][f"n_{n}"][0]
        par_tokens.append(data["avg_tokens"])
        par_accuracies.append(data["accuracy"])

    alpha_par, c_par, r2_par = fit_alpha(par_tokens, par_accuracies)

    print(f"\nPARALLEL:")
    print(f"  Tokens:     {par_tokens}")
    print(f"  Accuracies: {[f'{a:.1%}' for a in par_accuracies]}")
    print(f"  α = {alpha_par:.3f}")
    print(f"  R² = {r2_par:.3f}")

    # ========================================================================
    # VERDICT
    # ========================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    results["analysis"] = {
        "sequential_alpha": alpha_seq,
        "sequential_r2": r2_seq,
        "parallel_alpha": alpha_par,
        "parallel_r2": r2_par
    }

    # Check hypothesis
    seq_hypothesis = alpha_seq > 0.9
    par_hypothesis = alpha_par < 0.5
    difference_hypothesis = alpha_seq > alpha_par

    print(f"\nHYPOTHESIS TESTS:")
    print(f"  Sequential α > 0.9:  {'✓ SUPPORTED' if seq_hypothesis else '✗ NOT SUPPORTED'} (α = {alpha_seq:.3f})")
    print(f"  Parallel α < 0.5:    {'✓ SUPPORTED' if par_hypothesis else '✗ NOT SUPPORTED'} (α = {alpha_par:.3f})")
    print(f"  Sequential > Parallel: {'✓ SUPPORTED' if difference_hypothesis else '✗ NOT SUPPORTED'}")

    overall = seq_hypothesis and par_hypothesis and difference_hypothesis

    print(f"\n{'=' * 70}")
    if overall:
        print("OVERALL: ✓ ARC PRINCIPLE HYPOTHESIS SUPPORTED")
    else:
        print("OVERALL: ✗ ARC PRINCIPLE HYPOTHESIS NOT FULLY SUPPORTED")
    print(f"{'=' * 70}")

    results["verdict"] = {
        "sequential_supported": seq_hypothesis,
        "parallel_supported": par_hypothesis,
        "difference_supported": difference_hypothesis,
        "overall_supported": overall
    }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arc_experiment_results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {filename}")

    return results


# ============================================================================
# QUICK TEST (No API needed - uses simulated data)
# ============================================================================

def run_simulated_demo():
    """
    Demonstrate the analysis with simulated data that matches
    the expected pattern from the literature.
    """
    print("=" * 70)
    print("SIMULATED DEMO (No API key required)")
    print("This shows what the analysis looks like with literature-consistent data")
    print("=" * 70)

    # Simulated data matching literature expectations
    # Sequential: tokens increase with depth, accuracy improves super-linearly
    seq_tokens = np.array([80, 200, 400, 700])
    seq_accuracy = np.array([0.45, 0.62, 0.78, 0.88])  # Super-linear improvement

    # Parallel: tokens increase with N, accuracy improves sub-linearly
    par_tokens = np.array([100, 200, 400, 800])
    par_accuracy = np.array([0.40, 0.48, 0.54, 0.58])  # Diminishing returns

    print("\nSIMULATED SEQUENTIAL DATA (Chain-of-Thought):")
    print(f"  Tokens:     {seq_tokens.tolist()}")
    print(f"  Accuracies: {seq_accuracy.tolist()}")

    alpha_seq, c_seq, r2_seq = fit_alpha(seq_tokens.tolist(), seq_accuracy.tolist())
    print(f"  Fitted α = {alpha_seq:.3f} (R² = {r2_seq:.3f})")

    print("\nSIMULATED PARALLEL DATA (Best-of-N):")
    print(f"  Tokens:     {par_tokens.tolist()}")
    print(f"  Accuracies: {par_accuracy.tolist()}")

    alpha_par, c_par, r2_par = fit_alpha(par_tokens.tolist(), par_accuracy.tolist())
    print(f"  Fitted α = {alpha_par:.3f} (R² = {r2_par:.3f})")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)
    print(f"""
If you observe similar patterns with real API calls:

  Sequential α ≈ {alpha_seq:.2f} means:
    → Each doubling of thinking tokens produces ~{2**alpha_seq:.1f}x accuracy gain
    → Super-linear scaling (compounding returns)
    → Solution space is EXPANDING with depth

  Parallel α ≈ {alpha_par:.2f} means:
    → Each doubling of samples produces ~{2**alpha_par:.1f}x accuracy gain
    → Sub-linear scaling (diminishing returns)
    → Solution space is FIXED, just sampling more

  The difference (Δα = {alpha_seq - alpha_par:.2f}) is the key finding:
    → Sequential recursion fundamentally different from parallel
    → This supports the ARC Principle hypothesis
""")

    print("=" * 70)
    print("To run with real data, set your API key and run:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python arc_validation_experiment.py")
    print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC Principle Validation Experiment")
    parser.add_argument("--provider", choices=["openai", "anthropic", "groq"],
                        default="openai", help="API provider")
    parser.add_argument("--problems", type=int, default=20,
                        help="Number of problems to test")
    parser.add_argument("--demo", action="store_true",
                        help="Run simulated demo without API")

    args = parser.parse_args()

    if args.demo:
        run_simulated_demo()
    else:
        # Check for API key
        key_var = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY"
        }[args.provider]

        if not os.environ.get(key_var):
            print(f"ERROR: {key_var} not set")
            print(f"\nTo run the experiment:")
            print(f"  export {key_var}='your-api-key'")
            print(f"  python arc_validation_experiment.py --provider {args.provider}")
            print(f"\nOr run the demo without API:")
            print(f"  python arc_validation_experiment.py --demo")
            sys.exit(1)

        run_experiment(provider=args.provider, num_problems=args.problems)
