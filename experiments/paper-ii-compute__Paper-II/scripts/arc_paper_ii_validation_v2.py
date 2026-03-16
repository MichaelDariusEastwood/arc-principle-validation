#!/usr/bin/env python3
"""ARC Principle Validation v2 — Multi-Model Paper II Replication

Standalone script to test the ARC Principle (Eastwood, 2026) across 6 frontier
AI models using both sequential (depth ladder) and parallel (majority vote)
conditions.

Paper II found alpha_sequential ~ 2.24 for DeepSeek-R1 (quadratic capability
scaling with sequential reasoning depth) and alpha_parallel ~ 0.0 (no scaling
with parallel samples).

This script tests whether the quadratic limit holds across architectures:
  - DeepSeek R1
  - OpenAI GPT-5.4
  - Anthropic Claude Opus 4.6
  - Google Gemini Flash
  - Groq Qwen3-32B
  - xAI Grok 4.1 Fast

Features:
  - Sequential condition: depth ladder (minimal → exhaustive reasoning)
  - Parallel condition: N-sample majority vote at minimal depth
  - 30 problems: 12 tier-1 (calibration) + 18 tier-2 (AIME-level, defeats ceiling)
  - Tier-separated alpha analysis with bootstrap 95% CI
  - Cross-verification with 4-layer blinding (--verify)
  - Retry with exponential backoff on transient API errors
  - Per-problem accuracy heatmap
  - Checkpoint/resume support

Usage:
  python3 arc_paper_ii_validation_v2.py --model all --mode both --problems tier2
  python3 arc_paper_ii_validation_v2.py --model deepseek --mode sequential --repeats 5
  python3 arc_paper_ii_validation_v2.py --model openai --mode parallel --parallel-counts 1,3,7,15
  python3 arc_paper_ii_validation_v2.py --model all --mode both --verify

Author: Michael Darius Eastwood
Paper:  Eastwood's ARC Principle: Experimental Validation (Paper II v2)
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# ════════════════════════════════════════════════════════════════════
#  SECTION 0: CONSTANTS & CONFIGURATION
# ════════════════════════════════════════════════════════════════════

VERSION = "2.1"

ARC_PROBLEMS = [
    # Tier 1: Competition-prep level (baseline calibration)
    {"id": "ARC01", "prompt": "What is the sum of all positive divisors of 120? Give only the number.",
     "answer": 360, "difficulty": "easy", "tier": 1},
    {"id": "ARC02", "prompt": "In how many ways can you arrange the letters of the word MISSISSIPPI? Give only the number.",
     "answer": 34650, "difficulty": "medium", "tier": 1},
    {"id": "ARC03", "prompt": "What is the remainder when 7^100 is divided by 13? Give only the number.",
     "answer": 9, "difficulty": "medium", "tier": 1},
    {"id": "ARC04", "prompt": "Find the number of positive integers less than 1000 that are divisible by neither 5 nor 7. Give only the number.",
     "answer": 686, "difficulty": "medium", "tier": 1},
    {"id": "ARC05", "prompt": "What is the sum of the first 50 terms of the arithmetic sequence 3, 7, 11, 15, ...? Give only the number.",
     "answer": 5050, "difficulty": "easy", "tier": 1},
    {"id": "ARC06", "prompt": "How many 4-digit numbers have digits that sum to 9? Give only the number.",
     "answer": 165, "difficulty": "hard", "tier": 1},
    {"id": "ARC07", "prompt": "What is 17! / (14! * 3!)? Give only the number.",
     "answer": 680, "difficulty": "easy", "tier": 1},
    {"id": "ARC08", "prompt": "Find the last two digits of 3^2025. Give only the number.",
     "answer": 43, "difficulty": "hard", "tier": 1},
    {"id": "ARC09", "prompt": "In a round-robin tournament with 10 teams, how many total games are played? Give only the number.",
     "answer": 45, "difficulty": "easy", "tier": 1},
    {"id": "ARC10", "prompt": "What is the sum of all two-digit prime numbers? Give only the number.",
     "answer": 1043, "difficulty": "medium", "tier": 1},
    {"id": "ARC11", "prompt": "How many integers from 1 to 100 are neither perfect squares nor perfect cubes? Give only the number.",
     "answer": 87, "difficulty": "medium", "tier": 1},
    {"id": "ARC12", "prompt": "What is the value of C(20,10)? Give only the number.",
     "answer": 184756, "difficulty": "medium", "tier": 1},
    # Tier 2: AIME/Putnam level (defeats ceiling effect)
    {"id": "ARC13", "prompt": "How many integers from 1 to 2025 are coprime to 2025? Give only the number.",
     "answer": 1080, "difficulty": "hard", "tier": 2},
    {"id": "ARC14", "prompt": "Find the last three digits of 7^999. Give only the number.",
     "answer": 143, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC15", "prompt": ("Let n = 2^4 * 3^3 * 5^2. Find sigma(n) - n, where sigma(n) is "
        "the sum of all positive divisors of n. Give only the number."),
     "answer": 27640, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC16", "prompt": ("How many positive integers n with 1 <= n <= 1000 have the "
        "property that the decimal representation of 1/n terminates? Give only the number."),
     "answer": 29, "difficulty": "hard", "tier": 2},
    {"id": "ARC17", "prompt": ("How many ordered triples (a, b, c) of non-negative integers "
        "satisfy a + b + c = 20 and a + b >= c? Give only the number."),
     "answer": 176, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC18", "prompt": ("How many permutations of the set {1, 2, 3, 4, 5, 6, 7} have "
        "no fixed points (i.e., no element appears in its natural position)? Give only the number."),
     "answer": 1854, "difficulty": "hard", "tier": 2},
    {"id": "ARC19", "prompt": ("How many lattice paths from (0,0) to (8,8) using unit steps "
        "right (1,0) or up (0,1) never go strictly above the line y = x? Give only the number."),
     "answer": 1430, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC20", "prompt": ("How many surjective (onto) functions are there from a set of "
        "6 elements to a set of 4 elements? Give only the number."),
     "answer": 1560, "difficulty": "hard", "tier": 2},
    {"id": "ARC21", "prompt": ("In how many ways can 12 identical balls be distributed into "
        "4 distinct boxes such that no box contains more than 5 balls? Give only the number."),
     "answer": 125, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC22", "prompt": ("A standard deck of 52 cards is shuffled. Given that exactly 3 "
        "of the top 10 cards are aces, what is the probability the top card is an ace? "
        "Express as p/q in lowest terms, then give p + q. Give only the number."),
     "answer": 13, "difficulty": "hard", "tier": 2},
    {"id": "ARC23", "prompt": ("A fair six-sided die is rolled repeatedly until a 6 appears. "
        "What is the expected value of the sum of all rolls, including the final 6? "
        "Give only the number."),
     "answer": 21, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC24", "prompt": ("Alice and Bob take turns rolling a fair six-sided die, with "
        "Alice going first. The first player to roll a 6 wins. What is the probability that "
        "Alice wins? Express as p/q in lowest terms, then give p + q. Give only the number."),
     "answer": 17, "difficulty": "hard", "tier": 2},
    {"id": "ARC25", "prompt": "Find the value of 1^3 + 2^3 + 3^3 + ... + 25^3. Give only the number.",
     "answer": 105625, "difficulty": "hard", "tier": 2},
    {"id": "ARC26", "prompt": ("The roots of x^3 - 9x^2 + 26x - 24 = 0 are three positive "
        "integers. Find the sum of their squares. Give only the number."),
     "answer": 29, "difficulty": "hard", "tier": 2},
    {"id": "ARC27", "prompt": ("A sequence is defined by a_1 = 1, a_2 = 1, and "
        "a_n = a_{n-1} + 2*a_{n-2} for n >= 3. Find a_10. Give only the number."),
     "answer": 341, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC28", "prompt": ("In triangle ABC, AB = 13, BC = 14, and CA = 15. What is the "
        "area of triangle ABC? Give only the number."),
     "answer": 84, "difficulty": "hard", "tier": 2},
    {"id": "ARC29", "prompt": ("A convex polygon has 20 vertices. How many triangles can be "
        "formed using only vertices of this polygon such that none of the triangle's sides "
        "is a side of the polygon? Give only the number."),
     "answer": 800, "difficulty": "very_hard", "tier": 2},
    {"id": "ARC30", "prompt": ("A right circular cone has base radius 6 and slant height 10. "
        "What is the total surface area of the cone? Express as an integer multiple of pi, "
        "then give that integer. Give only the number."),
     "answer": 96, "difficulty": "hard", "tier": 2},
    {"id": "ARC31", "prompt": ("Let P(x) be a polynomial of degree 10 such that P(k) = k / (k + 1) for k = 0, 1, ..., 10. "
        "Find the value of 12 * P(11). Give only the number."),
     "answer": 10, "difficulty": "extreme", "tier": 3},
    {"id": "ARC32", "prompt": "Find the number of real solutions to the equation x^2 + x * sin(x) + cos(x) = 0. Give only the number.",
     "answer": 0, "difficulty": "extreme", "tier": 3},
    {"id": "ARC33", "prompt": "Find the smallest positive integer n such that n! ends in exactly 1000 zeros. Give only the number.",
     "answer": 4005, "difficulty": "extreme", "tier": 3},
    {"id": "ARC34", "prompt": "How many subsets of {1, 2, ..., 15} contain no two elements that sum to 16? Give only the number.",
     "answer": 4374, "difficulty": "extreme", "tier": 3},
    {"id": "ARC35", "prompt": ("Let a_1 = 1 and a_{n+1} = a_n + 1/a_n for all n >= 1. What is the integer part of a_{1000}? Give only the number."),
     "answer": 44, "difficulty": "extreme", "tier": 3},
    {"id": "ARC36", "prompt": "Find the maximum number of regions into which 20 circles can divide the plane. Give only the number.",
     "answer": 382, "difficulty": "extreme", "tier": 3},
    {"id": "ARC37", "prompt": "What is the minimum value of f(x) = |x-1| + |x-2| + ... + |x-100|? Give only the number.",
     "answer": 2500, "difficulty": "extreme", "tier": 3},
    {"id": "ARC38", "prompt": "In a regular 15-gon, how many diagonals are parallel to at least one of the sides? Give only the number.",
     "answer": 90, "difficulty": "extreme", "tier": 3},
    {"id": "ARC39", "prompt": "How many 5-digit numbers have the property that the product of their digits is exactly 2000? Give only the number.",
     "answer": 40, "difficulty": "extreme", "tier": 3},
    {"id": "ARC40", "prompt": "Find the sum of all positive integers n such that n^2 + 2025 is a perfect square. Give only the number.",
     "answer": 1768, "difficulty": "extreme", "tier": 3},
]

ENV_KEYS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
    "groq-qwen3": "GROQ_API_KEY",
    "grok-4-fast": "XAI_API_KEY",
}

RATE_LIMITS = {
    "groq": {"delay": 2.1},
    "xai": {"delay": 1.1},
    "openai": {"delay": 1.1},
    "anthropic": {"delay": 1.3},
    "deepseek": {"delay": 1.1},
    "google": {"delay": 1.1},
}

_rate_lock = threading.Lock()
_last_call = defaultdict(float)


def rate_limit(provider):
    """Enforce per-provider rate limiting."""
    delay = RATE_LIMITS.get(provider, {}).get("delay", 1.0)
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_call[provider]
        if elapsed < delay:
            time.sleep(delay - elapsed)
        _last_call[provider] = time.time()


# ════════════════════════════════════════════════════════════════════
#  SECTION 1: MODEL ADAPTERS (stripped-down, query-only)
# ════════════════════════════════════════════════════════════════════

class ModelAdapter:
    name = "base"
    SUBJECT_MODEL = None
    MAX_RETRIES = 3
    RETRY_BACKOFF = [5, 15, 45]  # seconds

    def query(self, prompt, depth_config):
        raise NotImplementedError

    def query_with_retry(self, prompt, depth_config):
        """Query with exponential backoff retries on transient errors."""
        for attempt in range(self.MAX_RETRIES):
            result = self.query(prompt, depth_config)
            if not result["response"].startswith("ERROR:"):
                return result
            # Check for retryable errors (rate limits, timeouts, 5xx)
            err = result["response"]
            retryable = any(kw in err.lower() for kw in
                           ["rate", "timeout", "429", "500", "502", "503",
                            "504", "overloaded", "capacity", "try again"])
            if not retryable or attempt == self.MAX_RETRIES - 1:
                return result
            wait = self.RETRY_BACKOFF[attempt]
            print(f"        RETRY {attempt+1}/{self.MAX_RETRIES} in {wait}s: "
                  f"{err[:60]}")
            time.sleep(wait)
        return result

    def get_depth_configs(self):
        raise NotImplementedError


class DeepSeekAdapter(ModelAdapter):
    name = "deepseek"
    SUBJECT_MODEL = "deepseek-reasoner"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                             base_url="https://api.deepseek.com")

    def query(self, prompt, depth_config):
        try:
            rate_limit("deepseek")
            modified = depth_config.get("prefix", "") + prompt
            max_tok = depth_config.get("max_tokens", 65536)
            resp = self.client.chat.completions.create(
                model=self.SUBJECT_MODEL,
                messages=[{"role": "user", "content": modified}],
                max_tokens=max_tok)
            content = resp.choices[0].message.content or ""
            reasoning = getattr(resp.choices[0].message, 'reasoning_content', '') or ""
            r_tokens = len(reasoning.split())
            if resp.usage:
                det = getattr(resp.usage, 'completion_tokens_details', None)
                if det and hasattr(det, 'reasoning_tokens') and det.reasoning_tokens:
                    r_tokens = det.reasoning_tokens
            return {"response": content, "reasoning_tokens": r_tokens,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    "depth_setting": depth_config.get("label", "natural")}
        except Exception as e:
            return {"response": f"ERROR: {e}", "reasoning_tokens": 0,
                    "total_tokens": 0, "depth_setting": "error"}

    def get_depth_configs(self):
        return [
            {"label": "minimal", "prefix": "Answer briefly: ", "max_tokens": 4096},
            {"label": "standard", "prefix": "", "max_tokens": 16384},
            {"label": "thorough", "prefix": "Think carefully and consider all angles: ",
             "max_tokens": 32768},
            {"label": "exhaustive", "prefix": ("This is extremely important. Think through "
                "every consideration, edge case, and implication. Reason step by step: "),
             "max_tokens": 65536},
            {"label": "extreme", "prefix": ("This is extremely important. Think through "
                "every consideration, edge case, and implication. Reason step by step: "),
             "max_tokens": 65536},
            {"label": "maximum", "prefix": ("This is extremely important. Think through "
                "every consideration, edge case, and implication. Reason step by step: "),
             "max_tokens": 65536},
        ]


class OpenAIAdapter(ModelAdapter):
    name = "openai"
    SUBJECT_MODEL = "gpt-5.4"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def query(self, prompt, depth_config):
        try:
            rate_limit("openai")
            effort = depth_config.get("reasoning_effort", "medium")
            resp = self.client.chat.completions.create(
                model=self.SUBJECT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort=effort,
                max_completion_tokens=128000)
            content = resp.choices[0].message.content or ""
            r_tokens = 0
            if resp.usage:
                det = getattr(resp.usage, 'completion_tokens_details', None)
                if det:
                    r_tokens = getattr(det, 'reasoning_tokens', 0) or 0
            return {"response": content, "reasoning_tokens": r_tokens,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    "depth_setting": effort}
        except Exception as e:
            return {"response": f"ERROR: {e}", "reasoning_tokens": 0,
                    "total_tokens": 0, "depth_setting": "error"}

    def get_depth_configs(self):
        return [
            {"label": "minimal", "reasoning_effort": "none"},
            {"label": "low", "reasoning_effort": "low"},
            {"label": "standard", "reasoning_effort": "medium"},
            {"label": "deep", "reasoning_effort": "high"},
            {"label": "exhaustive", "reasoning_effort": "xhigh"},
        ]


class AnthropicAdapter(ModelAdapter):
    name = "anthropic"
    SUBJECT_MODEL = "claude-opus-4-6"

    def __init__(self):
        import anthropic
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def query(self, prompt, depth_config):
        try:
            rate_limit("anthropic")
            effort = depth_config.get("effort", "high")
            prefix = depth_config.get("prefix", "")
            actual_prompt = prefix + prompt if prefix else prompt
            with self.client.messages.stream(
                model=self.SUBJECT_MODEL, max_tokens=128000,
                thinking={"type": "adaptive"},
                output_config={"effort": effort},
                messages=[{"role": "user", "content": actual_prompt}]) as stream:
                resp = stream.get_final_message()
            thinking_text, response_text = "", ""
            for block in resp.content:
                if block.type == "thinking":
                    thinking_text = block.thinking
                elif block.type == "text":
                    response_text = block.text
            r_tokens = len(thinking_text) // 4 if thinking_text else 0
            if resp.usage:
                text_est = len(response_text) // 4 if response_text else 0
                api_thinking_est = resp.usage.output_tokens - text_est
                if api_thinking_est > 0:
                    r_tokens = api_thinking_est
            total = (resp.usage.input_tokens + resp.usage.output_tokens) if resp.usage else 0
            return {"response": response_text, "reasoning_tokens": r_tokens,
                    "total_tokens": total, "depth_setting": f"effort_{effort}"}
        except Exception as e:
            return {"response": f"ERROR: {e}", "reasoning_tokens": 0,
                    "total_tokens": 0, "depth_setting": "error"}

    def get_depth_configs(self):
        return [
            {"label": "minimal", "effort": "low", "prefix": "Answer briefly: "},
            {"label": "standard", "effort": "medium", "prefix": ""},
            {"label": "deep", "effort": "high",
             "prefix": "Think carefully and consider all angles: "},
            {"label": "exhaustive", "effort": "max",
             "prefix": ("This is extremely important. Think through every consideration, "
                        "edge case, and implication. Reason step by step: ")},
            {"label": "extreme", "effort": "max",
             "prefix": ("This is extremely important. Think through every consideration, "
                        "edge case, and implication. Reason step by step: ")},
        ]


class GeminiAdapter(ModelAdapter):
    name = "gemini"
    MODELS = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro"]
    _cached_model = None

    def __init__(self):
        from google import genai
        from google.genai import types
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        self.types = types
        self._model_name = GeminiAdapter._cached_model

    def _get_model_name(self):
        if self._model_name:
            return self._model_name
        for model_name in self.MODELS:
            try:
                resp = self.client.models.generate_content(
                    model=model_name, contents="Say OK")
                if resp and resp.text:
                    self._model_name = model_name
                    GeminiAdapter._cached_model = model_name
                    print(f"    Gemini model detected: {model_name}")
                    return model_name
            except Exception:
                continue
        self._model_name = self.MODELS[0]
        return self._model_name

    def _build_thinking_config(self, model_name, depth_config):
        if "3" in model_name and "flash" in model_name:
            level = depth_config.get("thinking_level", "medium")
            return self.types.ThinkingConfig(thinking_level=level, include_thoughts=True)
        else:
            budget = depth_config.get("thinking_budget", 1024)
            return self.types.ThinkingConfig(thinking_budget=budget, include_thoughts=True)

    def query(self, prompt, depth_config):
        try:
            rate_limit("google")
            model_name = self._get_model_name()
            prefix = depth_config.get("prefix", "")
            budget = depth_config.get("thinking_budget", 1024)
            thinking_cfg = self._build_thinking_config(model_name, depth_config)
            config = self.types.GenerateContentConfig(
                max_output_tokens=65536, thinking_config=thinking_cfg)
            resp = self.client.models.generate_content(
                model=model_name, contents=prefix + prompt, config=config)
            r_tokens = budget
            thinking_text, response_text = "", ""
            if hasattr(resp, 'candidates') and resp.candidates:
                candidate = resp.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'thought') and part.thought:
                            thinking_text += part.text or ""
                        elif hasattr(part, 'text'):
                            response_text += part.text or ""
                if hasattr(resp, 'usage_metadata'):
                    um = resp.usage_metadata
                    if hasattr(um, 'thoughts_token_count') and um.thoughts_token_count:
                        r_tokens = um.thoughts_token_count
            if not response_text:
                response_text = resp.text or ""
            total_tokens = 0
            if hasattr(resp, 'usage_metadata'):
                total_tokens = getattr(resp.usage_metadata, 'total_token_count', 0) or 0
            return {"response": response_text, "reasoning_tokens": r_tokens,
                    "total_tokens": total_tokens, "depth_setting": f"budget_{budget}"}
        except Exception as e:
            return {"response": f"ERROR: {e}", "reasoning_tokens": 0,
                    "total_tokens": 0, "depth_setting": "error"}

    def get_depth_configs(self):
        return [
            {"label": "minimal", "thinking_budget": 256, "thinking_level": "low",
             "prefix": "Answer briefly: "},
            {"label": "standard", "thinking_budget": 1024, "thinking_level": "low",
             "prefix": ""},
            {"label": "deep", "thinking_budget": 4096, "thinking_level": "medium",
             "prefix": "Think carefully and consider all angles: "},
            {"label": "exhaustive", "thinking_budget": 16384, "thinking_level": "high",
             "prefix": ("This is extremely important. Think through every consideration, "
                        "edge case, and implication. Reason step by step: ")},
            {"label": "extreme", "thinking_budget": 32768, "thinking_level": "high",
             "prefix": ("This is extremely important. Think through every consideration, "
                        "edge case, and implication. Reason step by step: ")},
        ]


class GroqAdapter(ModelAdapter):
    name = "groq-qwen3"
    SUBJECT_MODEL = "qwen/qwen3-32b"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["GROQ_API_KEY"],
                             base_url="https://api.groq.com/openai/v1")

    def query(self, prompt, depth_config):
        try:
            rate_limit("groq")
            max_tok = depth_config.get("max_tokens", 40960)
            prefix = depth_config.get("prefix", "")
            actual_prompt = prefix + prompt if prefix else prompt
            kwargs = {"model": self.SUBJECT_MODEL,
                      "messages": [{"role": "user", "content": actual_prompt}],
                      "max_tokens": max_tok}
            effort = depth_config.get("reasoning_effort")
            if effort and effort != "none":
                kwargs["reasoning_effort"] = effort
            resp = self.client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""
            r_tokens = 0
            if resp.usage:
                det = getattr(resp.usage, 'completion_tokens_details', None)
                if det:
                    r_tokens = getattr(det, 'reasoning_tokens', 0) or 0
            return {"response": content, "reasoning_tokens": r_tokens,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    "depth_setting": depth_config.get("label", "default")}
        except Exception as e:
            return {"response": f"ERROR: {e}", "reasoning_tokens": 0,
                    "total_tokens": 0, "depth_setting": "error"}

    def get_depth_configs(self):
        return [
            {"label": "minimal", "reasoning_effort": "none",
             "prefix": "Answer briefly: ", "max_tokens": 4096},
            {"label": "standard", "reasoning_effort": "default",
             "prefix": "", "max_tokens": 8192},
            {"label": "deep", "reasoning_effort": "default",
             "prefix": "Think carefully and consider all angles: ", "max_tokens": 16384},
            {"label": "exhaustive", "reasoning_effort": "default",
             "prefix": ("This is extremely important. Think through every consideration, "
                        "edge case, and implication. Reason step by step: "),
             "max_tokens": 32768},
            {"label": "extreme", "reasoning_effort": "default",
             "prefix": ("This is extremely important. Think through every consideration, "
                        "edge case, and implication. Reason step by step: "),
             "max_tokens": 40960},
        ]


class GrokAdapter(ModelAdapter):
    name = "grok-4-fast"
    SUBJECT_MODEL = "grok-4-1-fast-reasoning"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ["XAI_API_KEY"],
                             base_url="https://api.x.ai/v1")

    def query(self, prompt, depth_config):
        try:
            rate_limit("xai")
            max_tok = depth_config.get("max_tokens", 30000)
            prefix = depth_config.get("prefix", "")
            actual_prompt = prefix + prompt if prefix else prompt
            resp = self.client.chat.completions.create(
                model=self.SUBJECT_MODEL,
                messages=[{"role": "user", "content": actual_prompt}],
                max_tokens=max_tok)
            content = resp.choices[0].message.content or ""
            r_tokens = 0
            if resp.usage:
                r_tokens = getattr(resp.usage, 'reasoning_tokens', 0) or 0
                if not r_tokens:
                    det = getattr(resp.usage, 'completion_tokens_details', None)
                    if det:
                        r_tokens = getattr(det, 'reasoning_tokens', 0) or 0
            return {"response": content, "reasoning_tokens": r_tokens,
                    "total_tokens": resp.usage.total_tokens if resp.usage else 0,
                    "depth_setting": depth_config.get("label", "unknown")}
        except Exception as e:
            return {"response": f"ERROR: {e}", "reasoning_tokens": 0,
                    "total_tokens": 0, "depth_setting": "error"}

    def get_depth_configs(self):
        return [
            {"label": "minimal", "prefix": "Answer briefly: ", "max_tokens": 4096},
            {"label": "standard", "prefix": "", "max_tokens": 10000},
            {"label": "deep", "prefix": "Think carefully and consider all angles: ",
             "max_tokens": 20000},
            {"label": "exhaustive", "prefix": ("This is extremely important. Think through "
                "every consideration, edge case, and implication. Reason step by step: "),
             "max_tokens": 30000},
            {"label": "extreme", "prefix": ("This is extremely important. Think through "
                "every consideration, edge case, and implication. Reason step by step: "),
             "max_tokens": 30000},
        ]


ADAPTERS = {
    "deepseek": DeepSeekAdapter,
    "openai": OpenAIAdapter,
    "anthropic": AnthropicAdapter,
    "gemini": GeminiAdapter,
    "groq-qwen3": GroqAdapter,
    "grok-4-fast": GrokAdapter,
}


# ════════════════════════════════════════════════════════════════════
#  SECTION 2: ANSWER EXTRACTION
# ════════════════════════════════════════════════════════════════════

def extract_arc_answer(text):
    """Extract a numerical answer from model response.
    Returns int or None.
    """
    if not text:
        return None
    text = text.strip().replace(",", "")
    patterns = [
        r"(?:answer|result|value|total)\s*(?:is|=|:)\s*(-?\d+)",
        r"(?:therefore|thus|so|hence)[,\s]+(-?\d+)",
        r"=\s*(-?\d+)\s*$",
        r"\*\*(-?\d+)\*\*",
        r"\\boxed\{(-?\d+)\}",
        r"(-?\d+)\s*$",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            try:
                return int(m.group(1))
            except ValueError:
                continue
    numbers = re.findall(r"-?\d+", text)
    if numbers:
        try:
            return int(numbers[-1])
        except ValueError:
            pass
    return None


# ════════════════════════════════════════════════════════════════════
#  SECTION 2b: CROSS-VERIFICATION (4-Layer Blinding)
# ════════════════════════════════════════════════════════════════════

# Each subject model is verified by a DIFFERENT model — no self-scoring.
# This aligns with the 4-layer blinding protocol from v5.
VERIFIER_MAP = {
    "deepseek":    "anthropic",
    "openai":      "deepseek",
    "anthropic":   "openai",
    "gemini":      "anthropic",
    "groq-qwen3":  "openai",
    "grok-4-fast": "deepseek",
}


def init_verifier(subject_model):
    """Initialize a verifier adapter for the given subject model.
    Returns (verifier_name, adapter) or (None, None) if unavailable.
    """
    verifier_name = VERIFIER_MAP.get(subject_model)
    if not verifier_name:
        print(f"  VERIFY: No verifier configured for {subject_model}")
        return None, None
    env_key = ENV_KEYS.get(verifier_name)
    if env_key and not os.environ.get(env_key):
        print(f"  VERIFY: {verifier_name} API key ({env_key}) not set, skipping")
        return None, None
    try:
        adapter = ADAPTERS[verifier_name]()
        print(f"  VERIFY: Using {verifier_name} to cross-verify {subject_model}")
        return verifier_name, adapter
    except Exception as e:
        print(f"  VERIFY: Failed to init {verifier_name}: {e}")
        return None, None


def run_verification(verifier_adapter, verifier_name, problems, subject_answers=None):
    """Independently solve each problem with a different model.

    The verifier solves each problem at minimal depth. Its answer is compared
    to the ground-truth answer key. Disagreements are flagged — they may
    indicate answer-key errors or ambiguous problem statements.

    If subject_answers is provided (dict of problem_id -> list of extracted
    answers), the report also cross-references subject vs verifier.

    Returns dict with per-problem verification results and summary stats.
    """
    print(f"\n  ── CROSS-VERIFICATION ({verifier_name.upper()}) ──")
    minimal_config = verifier_adapter.get_depth_configs()[0]
    per_problem = {}
    correct = 0
    total = 0
    disagreements = []

    for problem in problems:
        result = verifier_adapter.query(problem["prompt"], minimal_config)
        total += 1

        if result["response"].startswith("ERROR:"):
            print(f"    {problem['id']}: VERIFY_ERROR  {result['response'][:60]}")
            per_problem[problem["id"]] = {
                "verifier_answer": None, "expected": problem["answer"],
                "error": True, "correct": None,
            }
            continue

        extracted = extract_arc_answer(result["response"])
        is_correct = (extracted == problem["answer"])
        correct += int(is_correct)

        # Cross-reference with subject model's answers if available
        subj_note = ""
        if subject_answers and problem["id"] in subject_answers:
            subj_list = subject_answers[problem["id"]]
            subj_majority = Counter(a for a in subj_list if a is not None)
            if subj_majority:
                subj_top = subj_majority.most_common(1)[0][0]
                if subj_top == extracted and extracted != problem["answer"]:
                    subj_note = " !! BOTH_WRONG_SAME (possible answer-key error)"
                elif subj_top != extracted and is_correct:
                    subj_note = f" (subject_majority={subj_top})"

        tag = "AGREE" if is_correct else f"DISAGREE(got={extracted},expect={problem['answer']})"
        print(f"    {problem['id']}: {tag}{subj_note}")

        per_problem[problem["id"]] = {
            "verifier_answer": extracted,
            "expected": problem["answer"],
            "error": False,
            "correct": is_correct,
        }

        if not is_correct:
            entry = {"id": problem["id"], "verifier_answer": extracted,
                     "expected": problem["answer"]}
            if subj_note:
                entry["note"] = subj_note.strip()
            disagreements.append(entry)

    accuracy = correct / max(total, 1)
    print(f"\n    Verification: {correct}/{total} ({accuracy:.0%}) agree with ground truth")
    if disagreements:
        print(f"    DISAGREEMENTS ({len(disagreements)}):")
        for d in disagreements:
            print(f"      {d['id']}: verifier={d['verifier_answer']}, "
                  f"expected={d['expected']}"
                  + (f"  {d.get('note','')}" if d.get('note') else ""))
        print(f"    >> Review flagged problems for potential answer-key errors")

    return {
        "verifier_model": verifier_name,
        "verifier_api": ADAPTERS[verifier_name].SUBJECT_MODEL
            if hasattr(ADAPTERS[verifier_name], 'SUBJECT_MODEL') else "dynamic",
        "n_verified": total,
        "n_agree": correct,
        "accuracy": accuracy,
        "per_problem": per_problem,
        "disagreements": disagreements,
    }


def collect_subject_answers(seq_results):
    """Extract per-problem answer lists from sequential results for cross-ref.
    Returns dict of {problem_id: [extracted_answer, ...]}.
    """
    answers = defaultdict(list)
    if not seq_results:
        return dict(answers)
    for label, data in seq_results.items():
        for detail in data.get("details", []):
            if not detail.get("error", False):
                answers[detail["id"]].append(detail.get("extracted"))
    return dict(answers)


# ════════════════════════════════════════════════════════════════════
#  SECTION 3: SEQUENTIAL CONDITION
# ════════════════════════════════════════════════════════════════════

def run_sequential(adapter, problems, repeats=1):
    """Run sequential condition: increasing depth budgets.

    Returns dict of {depth_label: {accuracy, error, avg_tokens, n, details}}.
    """
    depths = adapter.get_depth_configs()
    results = {}

    for dc in depths:
        label = dc["label"]
        correct = 0
        total = 0
        all_tokens = []
        details = []

        for problem in problems:
            for rep in range(repeats):
                result = adapter.query_with_retry(problem["prompt"], dc)
                if result["response"].startswith("ERROR:"):
                    print(f"      ERROR on {problem['id']} @ {label} rep {rep}: "
                          f"{result['response'][:80]}")
                    details.append({"id": problem["id"], "rep": rep,
                                    "correct": False, "error": True,
                                    "tokens": 0})
                    total += 1
                    continue

                extracted = extract_arc_answer(result["response"])
                is_correct = (extracted == problem["answer"])
                correct += int(is_correct)
                total += 1
                r_tok = result.get("reasoning_tokens", 0)
                t_tok = result.get("total_tokens", 0)
                # Use total_tokens as primary compute metric (fixes zero-token
                # bug for OpenAI reasoning_effort=none and Groq Qwen3).
                # Fall back to reasoning_tokens only if total is unavailable.
                compute_tok = t_tok if t_tok > 0 else r_tok
                all_tokens.append(compute_tok)

                tag = "OK" if is_correct else f"WRONG({extracted}!={problem['answer']})"
                print(f"      {problem['id']} @ {label} rep{rep}: {tag}  "
                      f"R={r_tok} T={t_tok}")
                details.append({"id": problem["id"], "rep": rep,
                                "correct": is_correct, "error": False,
                                "extracted": extracted,
                                "expected": problem["answer"],
                                "tokens": compute_tok,
                                "reasoning_tokens": r_tok,
                                "total_tokens": t_tok})

        accuracy = correct / max(total, 1)
        avg_tokens = np.mean(all_tokens) if all_tokens else 0
        results[label] = {
            "accuracy": accuracy,
            "error": 1.0 - accuracy,
            "avg_tokens": float(avg_tokens),
            "n": total,
            "details": details,
        }
        print(f"    {label:12s}  Acc={accuracy:.1%}  Err={1-accuracy:.1%}  "
              f"AvgR={avg_tokens:.0f}  n={total}")

    return results


# ════════════════════════════════════════════════════════════════════
#  SECTION 4: PARALLEL CONDITION (majority vote)
# ════════════════════════════════════════════════════════════════════

def run_parallel(adapter, problems, n_samples_list, repeats=1):
    """Run parallel condition: N-sample majority vote at minimal depth.

    Returns dict of {N: {accuracy, error, avg_total_tokens, n}}.
    """
    minimal_config = adapter.get_depth_configs()[0]  # Always "minimal"
    results = {}

    for N in n_samples_list:
        correct = 0
        total = 0
        all_tokens = []

        for problem in problems:
            for rep in range(repeats):
                answers = []
                tokens_sum = 0
                errors = 0

                for _ in range(N):
                    result = adapter.query_with_retry(problem["prompt"],
                                                      minimal_config)
                    if result["response"].startswith("ERROR:"):
                        errors += 1
                        continue
                    extracted = extract_arc_answer(result["response"])
                    answers.append(extracted)
                    t_tok = result.get("total_tokens", 0)
                    r_tok = result.get("reasoning_tokens", 0)
                    tokens_sum += t_tok if t_tok > 0 else r_tok

                # Majority vote
                vote_counts = Counter(a for a in answers if a is not None)
                if vote_counts:
                    majority_answer = vote_counts.most_common(1)[0][0]
                else:
                    majority_answer = None

                is_correct = (majority_answer == problem["answer"])
                correct += int(is_correct)
                total += 1
                all_tokens.append(tokens_sum)

                tag = "OK" if is_correct else f"WRONG(vote={majority_answer})"
                print(f"      {problem['id']} N={N} rep{rep}: {tag}  "
                      f"votes={dict(vote_counts)}  R_total={tokens_sum}")

        accuracy = correct / max(total, 1)
        avg_tokens = np.mean(all_tokens) if all_tokens else 0
        results[N] = {
            "accuracy": accuracy,
            "error": 1.0 - accuracy,
            "avg_total_tokens": float(avg_tokens),
            "n": total,
        }
        print(f"    N={N:<3d}  Acc={accuracy:.1%}  Err={1-accuracy:.1%}  "
              f"AvgTotalR={avg_tokens:.0f}  n={total}")

    return results


# ════════════════════════════════════════════════════════════════════
#  SECTION 5: ALPHA COMPUTATION
# ════════════════════════════════════════════════════════════════════

def compute_alpha_pairwise(points):
    """Compute alpha between consecutive data points.
    points: list of (tokens, error) tuples.
    Returns list of alpha values.
    """
    alphas = []
    for i in range(len(points) - 1):
        r1, e1 = points[i]
        r2, e2 = points[i + 1]
        if r1 > 0 and r2 > 0 and e1 > 0 and e2 > 0 and r1 != r2:
            try:
                alpha = math.log(e1 / e2) / math.log(r2 / r1)
                alphas.append(alpha)
            except (ValueError, ZeroDivisionError):
                pass
    return alphas


def compute_alpha_endpoint(points):
    """Compute alpha from first to last valid point (most robust)."""
    if len(points) < 2:
        return None
    r1, e1 = points[0]
    r2, e2 = points[-1]
    if r1 > 0 and r2 > 0 and e1 > 0 and e2 > 0 and r1 != r2:
        try:
            return math.log(e1 / e2) / math.log(r2 / r1)
        except (ValueError, ZeroDivisionError):
            return None
    return None


def compute_alpha_regression(points):
    """Power law regression: log(E) = intercept + slope * log(R).
    Returns (alpha, r2, se) or (None, None, None).
    """
    if len(points) < 3:
        return None, None, None
    tokens = np.array([p[0] for p in points])
    errors = np.array([p[1] for p in points])
    if not (all(t > 0 for t in tokens) and all(e > 0 for e in errors)):
        return None, None, None
    sl, _, rv, _, se = stats.linregress(np.log(tokens), np.log(errors))
    return -sl, rv ** 2, se


def bootstrap_alpha_ci(depth_results, n_boot=2000, ci=0.95):
    """Bootstrap confidence interval for alpha_endpoint.

    Resamples per-problem results within each depth, recomputes accuracy/error,
    then computes alpha_endpoint. Returns (lo, hi) or (None, None).
    """
    # Collect per-problem detail arrays per depth
    depth_details = {}
    for label, data in depth_results.items():
        details = data.get("details", [])
        if details:
            depth_details[label] = details

    if len(depth_details) < 2:
        return None, None

    token_key = "avg_tokens"
    alphas = []
    rng = np.random.default_rng(42)

    for _ in range(n_boot):
        points = []
        for label, details in depth_details.items():
            n = len(details)
            if n == 0:
                continue
            indices = rng.integers(0, n, size=n)
            boot_correct = sum(1 for i in indices if details[i].get("correct", False))
            boot_error = 1.0 - (boot_correct / n)
            boot_tokens = np.mean([details[i].get("tokens", 0) for i in indices])
            if boot_tokens > 0 and boot_error > 0:
                points.append((float(boot_tokens), float(boot_error)))

        ep = compute_alpha_endpoint(points)
        if ep is not None:
            alphas.append(ep)

    if len(alphas) < 100:
        return None, None

    lo = float(np.percentile(alphas, (1 - ci) / 2 * 100))
    hi = float(np.percentile(alphas, (1 + ci) / 2 * 100))
    return lo, hi


def analyze_alphas(depth_results, condition_name="sequential"):
    """Compute all alpha variants from depth/parallel results.
    depth_results: dict of {label: {accuracy, error, avg_tokens/avg_total_tokens, n}}
    Returns dict of alpha values.
    """
    # Build (tokens, error) points in order
    token_key = "avg_tokens" if "avg_tokens" in next(iter(depth_results.values())) else "avg_total_tokens"
    points = []
    for label, data in depth_results.items():
        tokens = data.get(token_key, 0)
        error = data.get("error", 0)
        if tokens > 0 and error > 0:
            points.append((tokens, error))

    if len(points) < 2:
        return {"alpha_pairwise_avg": None, "alpha_endpoint": None,
                "alpha_regression": None, "r2": None, "points": points}

    pairwise = compute_alpha_pairwise(points)
    endpoint = compute_alpha_endpoint(points)
    regression, r2, se = compute_alpha_regression(points)

    # Bootstrap CI (only for sequential which has detail data)
    boot_lo, boot_hi = None, None
    if condition_name == "sequential":
        boot_lo, boot_hi = bootstrap_alpha_ci(depth_results)

    result = {
        "alpha_pairwise": pairwise,
        "alpha_pairwise_avg": float(np.mean(pairwise)) if pairwise else None,
        "alpha_endpoint": float(endpoint) if endpoint is not None else None,
        "alpha_regression": float(regression) if regression is not None else None,
        "r2": float(r2) if r2 is not None else None,
        "se": float(se) if se is not None else None,
        "points": [(float(t), float(e)) for t, e in points],
    }
    if boot_lo is not None:
        result["boot_ci_95"] = [boot_lo, boot_hi]
    return result


# ════════════════════════════════════════════════════════════════════
#  SECTION 6: OUTPUT & REPORTING
# ════════════════════════════════════════════════════════════════════

def format_results(model_name, sequential=None, parallel=None,
                   seq_alphas=None, par_alphas=None, problems_used=None,
                   verification=None):
    """Format results into Paper II-compatible JSON structure."""
    result = {
        "version": VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "n_problems": len(problems_used) if problems_used else 0,
        "n_tier1": sum(1 for p in (problems_used or []) if p.get("tier", 1) == 1),
        "n_tier2": sum(1 for p in (problems_used or []) if p.get("tier", 1) == 2),
    }

    if sequential is not None:
        seq_data = {}
        for label, data in sequential.items():
            seq_data[label] = {k: v for k, v in data.items() if k != "details"}
        result["sequential"] = seq_data
        result["sequential_alphas"] = seq_alphas

    if parallel is not None:
        par_data = {}
        for n_key, data in parallel.items():
            par_data[str(n_key)] = data
        result["parallel"] = par_data
        result["parallel_alphas"] = par_alphas

    # Cross-verification
    if verification:
        result["cross_verification"] = {
            "verifier_model": verification["verifier_model"],
            "verifier_api": verification.get("verifier_api", "unknown"),
            "n_verified": verification["n_verified"],
            "n_agree": verification["n_agree"],
            "agreement_rate": verification["accuracy"],
            "disagreements": verification["disagreements"],
            "blinding": "4-layer (no self-scoring)",
        }

    # Verdict
    best_seq = (seq_alphas or {}).get("alpha_endpoint")
    best_par = (par_alphas or {}).get("alpha_endpoint")
    if best_seq is not None and best_par is not None:
        result["verdict"] = {
            "alpha_sequential": best_seq,
            "alpha_parallel": best_par,
            "supports_arc_principle": best_seq > 1.0 and best_par < 1.0,
            "near_quadratic": best_seq > 1.8,
        }

    return result


def print_summary(model_name, sequential=None, parallel=None,
                  seq_alphas=None, par_alphas=None, verification=None,
                  tier_alphas=None):
    """Print formatted summary table."""
    print(f"\n{'='*65}")
    print(f"  ARC PRINCIPLE RESULTS: {model_name.upper()}")
    print(f"{'='*65}")

    if sequential:
        print(f"\n  SEQUENTIAL CONDITION (depth ladder):")
        print(f"    {'Depth':12s}  {'Accuracy':>10s}  {'Error':>8s}  {'Avg Tokens':>11s}  {'n':>5s}")
        print(f"    {'─'*50}")
        for label, data in sequential.items():
            print(f"    {label:12s}  {data['accuracy']:>10.1%}  {data['error']:>8.1%}  "
                  f"{data['avg_tokens']:>11.0f}  {data['n']:>5d}")
        if seq_alphas:
            print(f"\n    Alpha values (combined):")
            if seq_alphas.get("alpha_pairwise_avg") is not None:
                print(f"      Pairwise avg:  {seq_alphas['alpha_pairwise_avg']:>+.4f}")
            if seq_alphas.get("alpha_endpoint") is not None:
                ci = seq_alphas.get("boot_ci_95")
                ci_str = f"  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
                print(f"      Endpoint:      {seq_alphas['alpha_endpoint']:>+.4f}{ci_str}")
            if seq_alphas.get("alpha_regression") is not None:
                print(f"      Regression:    {seq_alphas['alpha_regression']:>+.4f}  "
                      f"R2={seq_alphas['r2']:.3f}")
            # Tier breakdown
            if tier_alphas:
                for tlabel in ["tier1", "tier2"]:
                    ta = tier_alphas.get(tlabel)
                    if ta:
                        ep = ta.get("alpha_endpoint")
                        ci = ta.get("boot_ci_95")
                        if ep is not None:
                            ci_str = f"  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
                            tag = ""
                            if tlabel == "tier1" and ep < 0.3:
                                tag = "  (CEILING)"
                            elif tlabel == "tier2" and ep > 1.8:
                                tag = "  (QUADRATIC)"
                            print(f"      {tlabel.upper():6s}:       "
                                  f"{ep:>+.4f}{ci_str}{tag}")

    if parallel:
        print(f"\n  PARALLEL CONDITION (majority vote at minimal depth):")
        print(f"    {'N':>5s}  {'Accuracy':>10s}  {'Error':>8s}  {'Avg Total Tokens':>17s}  {'n':>5s}")
        print(f"    {'─'*50}")
        for n_key, data in parallel.items():
            print(f"    {n_key:>5}  {data['accuracy']:>10.1%}  {data['error']:>8.1%}  "
                  f"{data['avg_total_tokens']:>17.0f}  {data['n']:>5d}")
        if par_alphas:
            print(f"\n    Alpha values:")
            if par_alphas.get("alpha_pairwise_avg") is not None:
                print(f"      Pairwise avg:  {par_alphas['alpha_pairwise_avg']:>+.4f}")
            if par_alphas.get("alpha_endpoint") is not None:
                print(f"      Endpoint:      {par_alphas['alpha_endpoint']:>+.4f}")

    # Per-problem accuracy breakdown (sequential)
    if sequential:
        all_details = []
        for label, data in sequential.items():
            for d in data.get("details", []):
                all_details.append({**d, "depth": label})
        if all_details:
            problem_ids = list(dict.fromkeys(d["id"] for d in all_details))
            depth_labels = list(sequential.keys())
            print(f"\n  PER-PROBLEM BREAKDOWN:")
            header = f"    {'Problem':>8s}"
            for dl in depth_labels:
                header += f"  {dl[:7]:>7s}"
            print(header)
            print(f"    {'─' * (8 + 9 * len(depth_labels))}")
            for pid in problem_ids:
                row = f"    {pid:>8s}"
                for dl in depth_labels:
                    hits = [d for d in all_details
                            if d["id"] == pid and d["depth"] == dl
                            and not d.get("error", False)]
                    if hits:
                        n_correct = sum(1 for h in hits if h.get("correct"))
                        cell = f"{n_correct}/{len(hits)}"
                        row += f"  {cell:>7s}"
                    else:
                        row += f"  {'--':>7s}"
                print(row)

    # Cross-verification
    if verification:
        print(f"\n  CROSS-VERIFICATION:")
        v = verification
        print(f"    Verifier:     {v['verifier_model']}")
        print(f"    Agreement:    {v['n_agree']}/{v['n_verified']} "
              f"({v['accuracy']:.0%}) with ground truth")
        if v.get("disagreements"):
            print(f"    Flags:        {len(v['disagreements'])} disagreement(s)")
            for d in v["disagreements"]:
                print(f"      {d['id']}: verifier={d['verifier_answer']}, "
                      f"expected={d['expected']}")
        else:
            print(f"    Flags:        None (all answers confirmed)")

    # ARC Principle verdict
    best_seq = (seq_alphas or {}).get("alpha_endpoint")
    best_par = (par_alphas or {}).get("alpha_endpoint")
    print(f"\n  {'═'*55}")
    print(f"  ARC PRINCIPLE VERDICT")
    print(f"  {'═'*55}")
    if best_seq is not None:
        print(f"    alpha_sequential = {best_seq:>+.4f}", end="")
        if best_seq > 1.8:
            print(f"  QUADRATIC LIMIT")
        elif best_seq > 1.0:
            print(f"  SUPER-LINEAR")
        elif best_seq > 0.5:
            print(f"  SUB-LINEAR")
        else:
            print(f"  FLAT/CEILING")
    if best_par is not None:
        print(f"    alpha_parallel   = {best_par:>+.4f}", end="")
        if best_par < 0.3:
            print(f"  NO SCALING (as predicted)")
        else:
            print(f"  UNEXPECTED PARALLEL SCALING")
    if best_seq is not None and best_par is not None:
        if best_seq > 1.0 and best_par < 1.0:
            print(f"\n    >> SUPPORTS ARC PRINCIPLE: alpha_seq > 1 > alpha_par")
        else:
            print(f"\n    >> DOES NOT CLEARLY SUPPORT ARC PRINCIPLE")
    elif best_seq is not None:
        if best_seq > 1.0:
            print(f"\n    >> Sequential scaling supports ARC Principle (parallel not tested)")
    print()


# ════════════════════════════════════════════════════════════════════
#  SECTION 7: CHECKPOINT & RESUME
# ════════════════════════════════════════════════════════════════════

def save_checkpoint(output_dir, model_name, data):
    """Save checkpoint JSON."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"arc_paper_ii_{model_name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  Saved: {path}")
    return path


def load_checkpoint(output_dir, model_name):
    """Load checkpoint if it exists."""
    path = os.path.join(output_dir, f"arc_paper_ii_{model_name}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ════════════════════════════════════════════════════════════════════
#  SECTION 8: MAIN & CLI
# ════════════════════════════════════════════════════════════════════

def run_model(model_name, args, problems):
    """Run full Paper II test for one model."""
    tier1 = sum(1 for p in problems if p["tier"] == 1)
    tier2 = sum(1 for p in problems if p["tier"] == 2)
    tier3 = sum(1 for p in problems if p["tier"] == 3)
    print(f"\n{'#'*65}")
    print(f"  MODEL: {model_name.upper()}")
    print(f"  API:   {ADAPTERS[model_name].SUBJECT_MODEL if hasattr(ADAPTERS[model_name], 'SUBJECT_MODEL') else 'dynamic'}")
    print(f"  Mode:  {args.mode}")
    print(f"  Problems: {len(problems)} ({tier1} T1 + {tier2} T2 + {tier3} T3)")
    print(f"  Repeats: {args.repeats}")
    print(f"{'#'*65}")

    # Check API key
    env_key = ENV_KEYS.get(model_name)
    if env_key and not os.environ.get(env_key):
        print(f"  SKIPPED: {env_key} not set")
        return None

    # Check for existing checkpoint
    if args.resume:
        existing = load_checkpoint(args.output_dir, model_name)
        if existing:
            print(f"  Loaded checkpoint with existing results")
            # Could extend to skip completed conditions
            # For now, just report and continue

    adapter = ADAPTERS[model_name]()

    seq_results, par_results = None, None
    seq_alphas, par_alphas = None, None
    tier_alphas = {}

    # Identify tier-1 and tier-2 problem IDs
    t1_ids = {p["id"] for p in problems if p.get("tier", 1) == 1}
    t2_ids = {p["id"] for p in problems if p.get("tier", 1) == 2}
    has_tiers = bool(t1_ids) and bool(t2_ids)

    # Sequential condition
    if args.mode in ("sequential", "both"):
        print(f"\n  ── SEQUENTIAL CONDITION ──")
        seq_results = run_sequential(adapter, problems, args.repeats)
        seq_alphas = analyze_alphas(seq_results, "sequential")

        # Tier-separated alpha analysis (if both tiers present)
        if has_tiers and seq_results:
            for tier_label, tier_ids in [("tier1", t1_ids), ("tier2", t2_ids)]:
                tier_depth_results = {}
                for label, data in seq_results.items():
                    tier_details = [d for d in data.get("details", [])
                                    if d["id"] in tier_ids]
                    if tier_details:
                        tier_n = len(tier_details)
                        tier_correct = sum(1 for d in tier_details
                                           if d.get("correct", False))
                        tier_tokens = [d.get("tokens", 0) for d in tier_details]
                        tier_depth_results[label] = {
                            "accuracy": tier_correct / max(tier_n, 1),
                            "error": 1.0 - (tier_correct / max(tier_n, 1)),
                            "avg_tokens": float(np.mean(tier_tokens))
                                          if tier_tokens else 0,
                            "n": tier_n,
                            "details": tier_details,
                        }
                if tier_depth_results:
                    ta = analyze_alphas(tier_depth_results, "sequential")
                    tier_alphas[tier_label] = ta
                    ep = ta.get("alpha_endpoint")
                    ci = ta.get("boot_ci_95")
                    ci_str = f"  CI=[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ""
                    print(f"\n    {tier_label.upper()} alpha_endpoint = "
                          f"{ep:+.4f}{ci_str}" if ep is not None else
                          f"\n    {tier_label.upper()} alpha_endpoint = N/A (ceiling)")

    # Parallel condition
    if args.mode in ("parallel", "both"):
        print(f"\n  ── PARALLEL CONDITION ──")
        par_counts = [int(x) for x in args.parallel_counts.split(",")]
        par_results = run_parallel(adapter, problems, par_counts, args.repeats)
        par_alphas = analyze_alphas(par_results, "parallel")

    # Cross-verification
    verification = None
    if getattr(args, 'verify', False):
        verifier_name, verifier_adapter = init_verifier(model_name)
        if verifier_adapter:
            subject_answers = collect_subject_answers(seq_results)
            verification = run_verification(
                verifier_adapter, verifier_name, problems, subject_answers)

    # Summary
    print_summary(model_name, seq_results, par_results, seq_alphas, par_alphas,
                  verification, tier_alphas)

    # Save
    result_data = format_results(model_name, seq_results, par_results,
                                  seq_alphas, par_alphas, problems, verification)
    if tier_alphas:
        result_data["tier_alphas"] = tier_alphas
    save_checkpoint(args.output_dir, model_name, result_data)

    return result_data


def main():
    parser = argparse.ArgumentParser(
        description="ARC Principle Validation v2 — Multi-Model Paper II Replication",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model all --mode both --problems tier2
  %(prog)s --model deepseek --mode sequential --repeats 5
  %(prog)s --model openai --mode parallel --parallel-counts 1,3,7,15,31
  %(prog)s --model all --mode both --verify  (with cross-verification)
        """)
    parser.add_argument("--model", default="all",
                        choices=list(ADAPTERS.keys()) + ["all"],
                        help="Model to test (default: all)")
    parser.add_argument("--mode", default="both",
                        choices=["sequential", "parallel", "both"],
                        help="Test condition (default: both)")
    parser.add_argument("--problems", default="all",
                        choices=["all", "tier1", "tier2", "tier3"],
                        help="Problem set (default: all)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Repeats per problem per depth (default: 3)")
    parser.add_argument("--parallel-counts", default="1,3,5,9",
                        help="Comma-separated parallel sample counts (default: 1,3,5,9)")
    parser.add_argument("--output-dir", default="./arc_paper_ii_results",
                        help="Output directory (default: ./arc_paper_ii_results)")
    parser.add_argument("--verify", action="store_true",
                        help="Cross-verify answers with a different model (4-layer blinding)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint if available")

    args = parser.parse_args()

    # Filter problems by tier
    if args.problems == "tier1":
        problems = [p for p in ARC_PROBLEMS if p["tier"] == 1]
    elif args.problems == "tier2":
        problems = [p for p in ARC_PROBLEMS if p["tier"] == 2]
    elif args.problems == "tier3":
        problems = [p for p in ARC_PROBLEMS if p["tier"] == 3]
    else:
        problems = ARC_PROBLEMS

    print(f"{'='*65}")
    print(f"  ARC PRINCIPLE VALIDATION v{VERSION}")
    print(f"  Paper II Multi-Model Replication")
    print(f"  Author: Michael Darius Eastwood")
    print(f"  Date:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"  Problems:  {len(problems)} ({args.problems})")
    print(f"  Mode:      {args.mode}")
    print(f"  Repeats:   {args.repeats}")
    if args.mode in ("parallel", "both"):
        print(f"  Parallel:  N = [{args.parallel_counts}]")
    print(f"  Output:    {args.output_dir}")
    if args.verify:
        print(f"  Verify:    ENABLED (4-layer blinding, no self-scoring)")
    print(f"{'='*65}")

    # Determine models to run
    if args.model == "all":
        models = list(ADAPTERS.keys())
    else:
        models = [args.model]

    all_results = {}
    for model_name in models:
        result = run_model(model_name, args, problems)
        if result:
            all_results[model_name] = result

    # Cross-model comparison
    if len(all_results) >= 2:
        print(f"\n{'='*65}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'='*65}")
        print(f"  {'Model':15s} | {'alpha_seq':>10s} | {'alpha_par':>10s} | {'ARC?':>6s}")
        print(f"  {'-'*15}-+-{'-'*10}-+-{'-'*10}-+-{'-'*6}")
        for mname, mdata in sorted(all_results.items()):
            a_seq = "N/A"
            a_par = "N/A"
            arc = "?"
            if mdata.get("sequential_alphas"):
                v = mdata["sequential_alphas"].get("alpha_endpoint")
                if v is not None:
                    a_seq = f"{v:>+.4f}"
            if mdata.get("parallel_alphas"):
                v = mdata["parallel_alphas"].get("alpha_endpoint")
                if v is not None:
                    a_par = f"{v:>+.4f}"
            verdict = mdata.get("verdict", {})
            if verdict.get("supports_arc_principle"):
                arc = "YES"
            elif verdict:
                arc = "NO"
            print(f"  {mname:15s} | {a_seq:>10s} | {a_par:>10s} | {arc:>6s}")
        print()

    # Save combined results
    if all_results:
        combined_path = os.path.join(args.output_dir, "arc_paper_ii_combined.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Combined results: {combined_path}")

    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
