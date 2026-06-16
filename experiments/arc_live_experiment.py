#!/usr/bin/env python3
"""
ARC Principle LIVE Experiment
==============================

Uses the deployed ask-book API to test sequential vs parallel reasoning.

This experiment:
1. Sends math problems to the live API
2. Compares detailed (sequential) vs quick (parallel-like) responses
3. Measures accuracy and response characteristics
"""

import subprocess
import json
import time
import re
from typing import Optional, Tuple
import numpy as np

# API endpoint - use local dev server
API_URL = "http://localhost:3456/api/ask-book"

# Test problems with ground truth
PROBLEMS = [
    {"q": "What is 47 + 89?", "a": 136},
    {"q": "What is 23 × 17?", "a": 391},
    {"q": "What is (45 + 67) × 3?", "a": 336},
    {"q": "Solve: 3x + 7 = 22. What is x?", "a": 5},
    {"q": "A train travels 240 km in 3 hours. What is the speed in km/h?", "a": 80},
    {"q": "What is 2³ + 3³?", "a": 35},
    {"q": "If 5 workers build a wall in 10 days, how many days for 10 workers?",  "a": 5},
    {"q": "What is 15% of 240?", "a": 36},
    {"q": "Three consecutive integers sum to 45. What is the largest?", "a": 16},
    {"q": "Solve: 2(x - 4) = 10. What is x?", "a": 9},
]

def call_api(query: str, mode: str = "quick", model: str = "claude") -> Tuple[str, bool]:
    """Call the ask-book API and return (response, success)"""

    # Escape the query for JSON
    query_escaped = query.replace('"', '\\"').replace('\n', '\\n')

    cmd = f'''curl -s -X POST "{API_URL}" -H "Content-Type: application/json" -d '{{"query": "{query_escaped}", "mode": "{mode}", "model": "{model}"}}'  --max-time 60'''

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=65)
        if result.returncode == 0 and result.stdout:
            try:
                data = json.loads(result.stdout)
                if "answer" in data:
                    return data["answer"], True
                elif "response" in data:
                    return data["response"], True
                elif "error" in data:
                    return f"API Error: {data['error']}", False
                else:
                    return result.stdout[:500], False
            except json.JSONDecodeError:
                return result.stdout[:500], False
        else:
            return f"Request failed: {result.stderr[:200]}", False
    except subprocess.TimeoutExpired:
        return "Timeout", False
    except Exception as e:
        return str(e), False


def extract_number(text: str) -> Optional[float]:
    """Extract numerical answer from response"""
    text = text.lower()

    # Look for explicit answer patterns
    patterns = [
        r"(?:answer|result|equals|is|=)\s*[:\s]*(-?[\d,]+\.?\d*)",
        r"\*\*(-?[\d,]+\.?\d*)\*\*",
        r"(-?[\d,]+\.?\d*)\s*(?:km/h|days|workers|units)?(?:\s|$|\.)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text.replace(",", ""))
        if matches:
            try:
                return float(matches[-1])
            except:
                continue

    # Fallback: all numbers, take last significant one
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        # Filter out very small numbers (like years, indices)
        significant = [float(n) for n in numbers if abs(float(n)) > 0.01]
        if significant:
            return significant[-1]

    return None


def check_answer(extracted: Optional[float], ground_truth: float) -> bool:
    """Check if answer matches within tolerance"""
    if extracted is None:
        return False

    if ground_truth == int(ground_truth):
        return abs(extracted - ground_truth) < 0.5

    if ground_truth == 0:
        return abs(extracted) < 0.01

    return abs(extracted - ground_truth) / abs(ground_truth) < 0.05


def run_condition(problems: list, mode: str, model: str, description: str) -> dict:
    """Run a single experimental condition"""
    print(f"\n{'='*60}")
    print(f"CONDITION: {description}")
    print(f"Mode: {mode}, Model: {model}")
    print(f"{'='*60}")

    correct = 0
    results = []

    for i, p in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {p['q']}")
        print(f"  Expected: {p['a']}")

        response, success = call_api(p['q'], mode=mode, model=model)

        if success:
            extracted = extract_number(response)
            is_correct = check_answer(extracted, p['a'])

            if is_correct:
                correct += 1
                print(f"  Got: {extracted} ✓ CORRECT")
            else:
                print(f"  Got: {extracted} ✗ WRONG")
                print(f"  Response snippet: {response[:150]}...")

            results.append({
                "question": p['q'],
                "expected": p['a'],
                "extracted": extracted,
                "correct": is_correct,
                "response_length": len(response)
            })
        else:
            print(f"  API Error: {response[:100]}")
            results.append({
                "question": p['q'],
                "expected": p['a'],
                "extracted": None,
                "correct": False,
                "error": response[:200]
            })

        time.sleep(1)  # Rate limiting

    accuracy = correct / len(problems)
    avg_response_len = np.mean([r.get("response_length", 0) for r in results if "response_length" in r])

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{len(problems)} = {accuracy:.1%}")
    print(f"Avg response length: {avg_response_len:.0f} chars")
    print(f"{'='*60}")

    return {
        "mode": mode,
        "model": model,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(problems),
        "avg_response_length": avg_response_len,
        "results": results
    }


def main():
    print("="*70)
    print("ARC PRINCIPLE LIVE VALIDATION EXPERIMENT")
    print("="*70)
    print(f"API: {API_URL}")
    print(f"Problems: {len(PROBLEMS)}")
    print("="*70)

    # Test API connectivity first
    print("\nTesting API connectivity...")
    test_response, test_success = call_api("What is 2+2?", mode="quick", model="claude")
    if not test_success:
        print(f"API Error: {test_response}")
        print("\nCannot proceed without API access.")
        return
    print(f"API responding: {test_response[:100]}...")

    # Run conditions
    results = {}

    # Condition 1: Quick mode (like parallel - minimal reasoning)
    results["quick_claude"] = run_condition(
        PROBLEMS,
        mode="quick",
        model="claude",
        description="QUICK MODE (Minimal reasoning, like parallel)"
    )

    # Condition 2: Full mode (like sequential - detailed reasoning)
    results["full_claude"] = run_condition(
        PROBLEMS,
        mode="full",
        model="claude",
        description="FULL MODE (Detailed reasoning, like sequential)"
    )

    # Analysis
    print("\n" + "="*70)
    print("COMPARISON ANALYSIS")
    print("="*70)

    quick_acc = results["quick_claude"]["accuracy"]
    full_acc = results["full_claude"]["accuracy"]
    quick_len = results["quick_claude"]["avg_response_length"]
    full_len = results["full_claude"]["avg_response_length"]

    print(f"\nQUICK (parallel-like):")
    print(f"  Accuracy: {quick_acc:.1%}")
    print(f"  Avg response: {quick_len:.0f} chars")

    print(f"\nFULL (sequential-like):")
    print(f"  Accuracy: {full_acc:.1%}")
    print(f"  Avg response: {full_len:.0f} chars")

    print(f"\nDIFFERENCE:")
    print(f"  Accuracy improvement: {full_acc - quick_acc:+.1%}")
    print(f"  Response length ratio: {full_len/quick_len:.1f}x")

    # Efficiency calculation
    if quick_len > 0 and full_len > 0:
        quick_efficiency = quick_acc / quick_len * 1000
        full_efficiency = full_acc / full_len * 1000

        print(f"\nEFFICIENCY (accuracy per 1000 chars):")
        print(f"  Quick: {quick_efficiency:.3f}")
        print(f"  Full: {full_efficiency:.3f}")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)

    if full_acc > quick_acc:
        improvement = (full_acc - quick_acc) / quick_acc * 100 if quick_acc > 0 else float('inf')
        print(f"\n✓ SEQUENTIAL (full) OUTPERFORMS PARALLEL (quick)")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"\nThis supports the ARC Principle hypothesis:")
        print(f"  Sequential reasoning expands the solution space,")
        print(f"  enabling access to correct answers that quick")
        print(f"  responses cannot reach.")
    else:
        print(f"\n✗ No significant difference observed")
        print(f"  This could indicate:")
        print(f"  - Problems too easy (solution space already accessible)")
        print(f"  - Mode differences not creating sufficient depth variation")

    # Save results
    import json
    from datetime import datetime

    filename = f"arc_live_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {filename}")


if __name__ == "__main__":
    main()
