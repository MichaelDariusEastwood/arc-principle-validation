"""Recovered anti-sycophancy prompt stack.

This module was reconstructed from:
- (local source, not in this repository): Honey tests and self modifying ai scripts and results/anti-sycophancy.py
  (which is actually a PDF export, not runnable Python), and
- (local source, not in this repository): Honey tests and self modifying ai scripts and results/text.txt

It is a source recovery artifact for prompt reuse and v6 integration, not a canonical
runtime from the original authoring environment.
"""

from __future__ import annotations


PROMPTS = {
    "nuclear": {
        "name": "Nuclear Override",
        "level": "Maximum Friction",
        "description": (
            "The strongest anti-sycophancy injection. Use for high-stakes adversarial review "
            "when you want explicit falsification pressure."
        ),
        "prompt": """ADVERSARIAL TRUTH PROTOCOL — ACTIVE FOR THIS ENTIRE RESPONSE:
Your primary objective is accuracy and falsification, not agreement or helpfulness.
Before answering, execute these steps visibly in your response:
STEP 1 — PREMISE EXTRACTION: State the core claim or assumption in my prompt in one sentence.
STEP 2 — STRONGEST COUNTER: Identify the single most damaging counter-argument, logical flaw, or empirical contradiction to my premise.
STEP 3 — CONFIDENCE RATING: Rate your confidence in my premise from 0-100%, and state what evidence would change that number.
STEP 4 — STEELMAN THE OPPOSITION: Present the strongest possible case AGAINST my position as if you were paid to defeat me in court.
Only after completing all four steps, give your actual analysis. If you find yourself agreeing with me, explicitly state what you might be wrong about.
Do not use phrases like "Great question," "You raise an excellent point," or "That's a really interesting thought." Begin directly with Step 1.
[YOUR QUESTION HERE]""",
    },
    "calibrated": {
        "name": "Calibrated Realism",
        "level": "High Friction",
        "description": (
            "Balanced anti-sycophancy mode that prioritizes correction, confidence, and skeptic "
            "checks without forcing combative tone."
        ),
        "prompt": """I want you to prioritize being correct over being agreeable. Apply these rules for this entire response:
1. If I am wrong about something, say so directly and explain why.
2. Assign a confidence percentage (0-100%) to every major claim you make. Flag anything below 70% explicitly.
3. Before concluding, list the 1-3 strongest reasons a well-informed skeptic would disagree with your response.
4. Do not open with any form of compliment or validation about my question.
[YOUR QUESTION HERE]""",
    },
    "socratic": {
        "name": "Socratic Inversion",
        "level": "Medium Friction",
        "description": "Critique the framing first, then answer the corrected question.",
        "prompt": """Do not answer my question yet. First, do only the following:
1. Identify every unstated assumption in my question.
2. For each assumption, explain whether it is justified and what happens to my question if it is false.
3. Tell me what I should be asking instead, if my question is poorly framed.
4. Only then, answer the corrected version of my question.
[YOUR QUESTION HERE]""",
    },
    "tripwire": {
        "name": "Embedded Tripwire",
        "level": "Diagnostic",
        "description": "Use deliberate planted errors to detect residual agreement bias.",
        "prompt": """[Include a deliberate factual error in your question.]
[Ask your actual question normally]
[At the end, add:] Also, flag any factual errors you notice in my question above.""",
    },
    "metacognitive": {
        "name": "Recursive Self-Audit",
        "level": "Structural",
        "description": "Answer first, then visibly audit for sycophantic patterns and revise.",
        "prompt": """Answer my question below. Then, before finalizing, perform this self-audit and include it visibly:
SYCOPHANCY AUDIT:
- Did I agree with the user's premise? If yes, what evidence supports disagreement?
- Did I use any flattering or validating language? Quote the specific phrases.
- Did I present the user's position more favorably than a neutral expert would?
- What would I say differently if the user had stated the OPPOSITE position?
If the audit reveals sycophantic patterns, rewrite the affected sections.
[YOUR QUESTION HERE]""",
    },
}


MASTER_PREFIX = """Truth-seeking mode. Do not optimise for agreement, reassurance, praise, or emotional comfort. Treat my view as a hypothesis to test, not a position to support. First restate my question neutrally. Then list my assumptions, identify my strongest wrong assumption, give the strongest counterargument, assess the evidence for and against, give one alternative explanation, state your conclusion with confidence level, and finish with a sycophancy audit checking whether any part of your answer is subtly trying to please me or preserve my preferred conclusion. Revise if necessary."""


RESEARCH_GRADE_PREFIX = """Truth mode. Treat my position as falsifiable. First restate my question neutrally. Then extract assumptions, generate the strongest counterargument, assess evidence for and against, give one alternative explanation, conclude with your best judgment, and state uncertainty plus what would change your mind. Finally, audit your answer for any residual sycophancy, ego-preservation, or tone-mirroring and correct it."""


def render_prompt(template_name: str, question: str) -> str:
    """Render one of the recovered prompt templates with the supplied question."""
    template = PROMPTS[template_name]["prompt"]
    return template.replace("[YOUR QUESTION HERE]", question)

