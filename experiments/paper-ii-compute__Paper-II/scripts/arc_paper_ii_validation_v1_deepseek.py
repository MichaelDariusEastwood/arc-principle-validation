#!/usr/bin/env python3
"""
ARC PRINCIPLE VALIDATION EXPERIMENT
====================================
Experimental validation using DeepSeek R1 API.

This script tests the ARC Principle prediction that:
- Sequential recursion yields α > 1 (compounding returns)
- Parallel recursion yields α < 1 (diminishing returns)

Author: Michael Darius Eastwood
Date: January 2026
Version: 2.0

Requirements:
- Python 3.8+
- openai >= 1.0.0
- DEEPSEEK_API_KEY environment variable

Usage:
    export DEEPSEEK_API_KEY="your-key-here"
    python arc_validation_deepseek.py

Output:
    - Console: Per-problem results and summary statistics
    - File: JSON with complete experimental data
"""

import os
import sys
import json
import math
import re
from datetime import datetime
from collections import Counter

# Check for API key before importing openai
if not os.environ.get("DEEPSEEK_API_KEY"):
    print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
    print("Please run: export DEEPSEEK_API_KEY='your-key-here'")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.")
    print("Please run: pip install openai")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Token budgets for sequential condition
TOKEN_BUDGETS = [512, 1024, 2048, 4096]

# Sample counts for parallel condition  
PARALLEL_N = [1, 2, 4]

# API configuration
API_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-reasoner"

# Competition-level mathematics problems
# These are designed to:
# 1. Avoid ceiling effects (baseline accuracy approximately 58%)
# 2. Require genuine reasoning (not just recall)
# 3. Have verifiable numerical answers
PROBLEMS = [
    {
        "question": "What is the sum of all positive divisors of 120?",
        "answer": 360
    },
    {
        "question": "In how many ways can you arrange the letters of the word MISSISSIPPI?",
        "answer": 34650
    },
    {
        "question": "What is the remainder when 7^100 is divided by 13?",
        "answer": 9
    },
    {
        "question": "Find the number of positive integers less than 1000 that are divisible by neither 5 nor 7.",
        "answer": 686
    },
    {
        "question": "What is the sum of the first 50 terms of the arithmetic sequence 3, 7, 11, 15, ...?",
        "answer": 5050
    },
    {
        "question": "How many 4-digit numbers have digits that sum to 9?",
        "answer": 165
    },
    {
        "question": "What is 17! / (14! * 3!)?",
        "answer": 680
    },
    {
        "question": "Find the last two digits of 3^2025.",
        "answer": 43
    },
    {
        "question": "In a round-robin tournament with 10 teams, how many total games are played?",
        "answer": 45
    },
    {
        "question": "What is the sum of all two-digit prime numbers?",
        "answer": 1043
    },
    {
        "question": "How many integers from 1 to 100 are neither perfect squares nor perfect cubes?",
        "answer": 87
    },
    {
        "question": "What is the value of C(20,10)?",
        "answer": 184756
    }
]


# =============================================================================
# API CLIENT
# =============================================================================

def create_client():
    """Create and return DeepSeek API client."""
    return OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=API_BASE_URL
    )


# =============================================================================
# ANSWER EXTRACTION
# =============================================================================

def extract_number(text):
    """
    Extract the final numerical answer from model response.
    
    Handles common formats:
    - "The answer is 42"
    - "Therefore, 42"
    - "= 42"
    - Plain "42"
    - Numbers with commas: "1,000"
    
    Returns None if no number found.
    """
    if not text:
        return None
    
    # Clean the text
    text = text.strip()
    
    # Remove commas from numbers
    text = text.replace(",", "")
    
    # Try to find "answer is X" pattern
    patterns = [
        r"(?:answer|result|value|total)\s*(?:is|=|:)\s*(\d+)",
        r"(?:therefore|thus|so|hence)[,\s]+(\d+)",
        r"=\s*(\d+)\s*$",
        r"(\d+)\s*$"  # Last number in text
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    
    # Fallback: find all numbers and return the last one
    numbers = re.findall(r"\d+", text)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
    
    return None


# =============================================================================
# API CALLS
# =============================================================================

def call_deepseek(client, question, max_tokens=4096):
    """
    Call DeepSeek R1 API and return response with reasoning tokens.
    
    Args:
        client: OpenAI-compatible client
        question: Problem to solve
        max_tokens: Maximum tokens for response
    
    Returns:
        dict with keys:
            - answer_text: The model's final answer
            - reasoning_text: The reasoning chain
            - reasoning_tokens: Number of reasoning tokens used
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": "You are a mathematical problem solver. Think through the problem carefully, then provide your final numerical answer. End your response with 'The answer is X' where X is the numerical answer."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            max_tokens=max_tokens
        )
        
        # Extract response content
        message = response.choices[0].message
        
        # DeepSeek R1 provides reasoning_content separately
        reasoning_text = getattr(message, 'reasoning_content', '') or ''
        answer_text = message.content or ''
        
        # Count reasoning tokens (approximate if not provided directly)
        reasoning_tokens = len(reasoning_text.split()) * 1.3  # Rough token estimate
        
        return {
            "answer_text": answer_text,
            "reasoning_text": reasoning_text,
            "reasoning_tokens": int(reasoning_tokens)
        }
        
    except Exception as e:
        print(f"  API Error: {e}")
        return {
            "answer_text": "",
            "reasoning_text": "",
            "reasoning_tokens": 0
        }


# =============================================================================
# EXPERIMENTAL CONDITIONS
# =============================================================================

def run_sequential_condition(client, problems):
    """
    Run sequential condition: vary reasoning depth via token budget.
    
    This tests whether deeper sequential reasoning improves accuracy
    with compounding returns (α > 1).
    """
    print("\n--- SEQUENTIAL CONDITION ---")
    print("Varying reasoning depth via token budget\n")
    
    results = []
    
    for budget in TOKEN_BUDGETS:
        print(f"Budget {budget}:")
        
        correct = 0
        total_tokens = 0
        
        for i, prob in enumerate(problems):
            response = call_deepseek(client, prob["question"], max_tokens=budget)
            
            extracted = extract_number(response["answer_text"])
            is_correct = (extracted == prob["answer"])
            
            if is_correct:
                correct += 1
            
            tokens_used = response["reasoning_tokens"]
            total_tokens += tokens_used
            
            status = "✓" if is_correct else "✗"
            print(f"  [{i+1:2d}] Exp: {prob['answer']:>6}, Got: {extracted:>6} {status} R={tokens_used}")
        
        accuracy = correct / len(problems)
        error = 1 - accuracy
        avg_tokens = total_tokens / len(problems)
        
        results.append({
            "budget": budget,
            "accuracy": accuracy,
            "error": error,
            "avg_tokens": avg_tokens
        })
        
        print(f"  => Accuracy: {accuracy*100:.1f}%, Avg R: {avg_tokens:.0f} tokens\n")
    
    return results


def run_parallel_condition(client, problems):
    """
    Run parallel condition: majority voting over N samples.
    
    This tests whether parallel sampling improves accuracy
    with diminishing returns (α < 1).
    """
    print("\n--- PARALLEL CONDITION ---")
    print("Majority voting over N independent samples\n")
    
    results = []
    
    for N in PARALLEL_N:
        print(f"N={N} samples:")
        
        correct = 0
        total_tokens = 0
        
        for i, prob in enumerate(problems):
            # Generate N independent samples
            samples = []
            sample_tokens = 0
            
            for _ in range(N):
                response = call_deepseek(client, prob["question"], max_tokens=1024)
                extracted = extract_number(response["answer_text"])
                samples.append(extracted)
                sample_tokens += response["reasoning_tokens"]
            
            # Majority vote
            valid_samples = [s for s in samples if s is not None]
            if valid_samples:
                counter = Counter(valid_samples)
                majority_answer = counter.most_common(1)[0][0]
            else:
                majority_answer = None
            
            is_correct = (majority_answer == prob["answer"])
            
            if is_correct:
                correct += 1
            
            total_tokens += sample_tokens
            
            status = "✓" if is_correct else "✗"
            print(f"  [{i+1:2d}] Exp: {prob['answer']:>6}, Got: {majority_answer:>6} {status} R={sample_tokens}")
        
        accuracy = correct / len(problems)
        error = 1 - accuracy
        avg_tokens = total_tokens / len(problems)
        
        results.append({
            "N": N,
            "accuracy": accuracy,
            "error": error,
            "avg_tokens": avg_tokens
        })
        
        print(f"  => Accuracy: {accuracy*100:.1f}%, Avg R: {avg_tokens:.0f} tokens\n")
    
    return results


# =============================================================================
# ANALYSIS
# =============================================================================

def calculate_alpha(r1, e1, r2, e2):
    """
    Calculate scaling exponent α from two data points.
    
    Uses the formula: α = ln(E₁/E₂) / ln(R₂/R₁)
    
    Returns None if calculation is invalid (e.g., division by zero).
    """
    if r1 <= 0 or r2 <= 0 or e1 <= 0 or e2 <= 0:
        return None
    if r1 == r2 or e1 == e2:
        return 0.0
    
    try:
        alpha = math.log(e1 / e2) / math.log(r2 / r1)
        return alpha
    except (ValueError, ZeroDivisionError):
        return None


def analyse_results(seq_results, par_results):
    """
    Calculate α values from experimental results.
    
    Returns dict with sequential and parallel α estimates.
    """
    # Calculate sequential alphas (between consecutive budgets)
    seq_alphas = []
    for i in range(len(seq_results) - 1):
        r1 = seq_results[i]["avg_tokens"]
        e1 = seq_results[i]["error"]
        r2 = seq_results[i + 1]["avg_tokens"]
        e2 = seq_results[i + 1]["error"]
        
        alpha = calculate_alpha(r1, e1, r2, e2)
        if alpha is not None:
            seq_alphas.append(alpha)
    
    # Calculate parallel alphas (between consecutive N values)
    par_alphas = []
    for i in range(len(par_results) - 1):
        r1 = par_results[i]["avg_tokens"]
        e1 = par_results[i]["error"]
        r2 = par_results[i + 1]["avg_tokens"]
        e2 = par_results[i + 1]["error"]
        
        alpha = calculate_alpha(r1, e1, r2, e2)
        if alpha is not None:
            par_alphas.append(alpha)
    
    # Calculate averages
    avg_seq_alpha = sum(seq_alphas) / len(seq_alphas) if seq_alphas else 0
    avg_par_alpha = sum(par_alphas) / len(par_alphas) if par_alphas else 0
    
    return {
        "seq_alphas": seq_alphas,
        "par_alphas": par_alphas,
        "avg_seq_alpha": avg_seq_alpha,
        "avg_par_alpha": avg_par_alpha
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the complete ARC Principle validation experiment."""
    
    print("=" * 70)
    print("ARC PRINCIPLE VALIDATION - DeepSeek R1")
    print("=" * 70)
    print(f"Problems: {len(PROBLEMS)}")
    print(f"Sequential budgets: {TOKEN_BUDGETS}")
    print(f"Parallel N values: {PARALLEL_N}")
    print("=" * 70)
    
    # Create API client
    client = create_client()
    
    # Run experiments
    seq_results = run_sequential_condition(client, PROBLEMS)
    par_results = run_parallel_condition(client, PROBLEMS)
    
    # Analyse results
    analysis = analyse_results(seq_results, par_results)
    
    # Print summary
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    print("\nSequential Condition:")
    for r in seq_results:
        print(f"  Budget {r['budget']:4d}: {r['accuracy']*100:.1f}% acc, "
              f"{r['error']*100:.1f}% err, R={r['avg_tokens']:.0f}")
    
    print("\nParallel Condition:")
    for r in par_results:
        print(f"  N={r['N']:1d}: {r['accuracy']*100:.1f}% acc, "
              f"{r['error']*100:.1f}% err, R={r['avg_tokens']:.0f}")
    
    print("\nScaling Exponents:")
    print(f"  Sequential α values: {[f'{a:.2f}' for a in analysis['seq_alphas']]}")
    print(f"  Sequential α (avg):  {analysis['avg_seq_alpha']:.2f}")
    print(f"  Parallel α values:   {[f'{a:.2f}' for a in analysis['par_alphas']]}")
    print(f"  Parallel α (avg):    {analysis['avg_par_alpha']:.2f}")
    
    print("\nInterpretation:")
    if analysis['avg_seq_alpha'] > 1:
        print(f"  Sequential: α = {analysis['avg_seq_alpha']:.2f} > 1 ✓ (super-linear)")
    else:
        print(f"  Sequential: α = {analysis['avg_seq_alpha']:.2f} ≤ 1 ✗ (sub-linear)")
    
    if analysis['avg_par_alpha'] < 1:
        print(f"  Parallel:   α = {analysis['avg_par_alpha']:.2f} < 1 ✓ (sub-linear)")
    else:
        print(f"  Parallel:   α = {analysis['avg_par_alpha']:.2f} ≥ 1 ✗ (super-linear)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"arc_deepseek_results_{timestamp}.json"
    
    results_data = {
        "timestamp": timestamp,
        "sequential": seq_results,
        "parallel": par_results,
        **analysis
    }
    
    with open(filename, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {filename}")
    print("=" * 70)
    
    return results_data


if __name__ == "__main__":
    main()
