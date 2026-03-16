# Eden v6 Methods Blueprint

Date: 2026-03-12

## Implementation status

An initial shared-engine scaffold now exists at:

- `/Users/michaeleastwood/Downloads/arc_eden_v6_runner.py`
- `/Users/michaeleastwood/Downloads/arc_eden_v6/`

The standalone monolith at `/Users/michaeleastwood/Downloads/arc_eden_v6_runner.py`
is now the canonical execution file. The package folder remains the source tree
used to regenerate it.

Live experiment modes in the canonical v6 runner:

- `baseline_alignment`
- `eden_intervention`
- `null_baseline`
- `capability_control`
- `purpose_kernel`
- `loop_ablation`
- `suppression_residual`
- `deception_hawthorne`
- `laundering_leakage`
- `laundering_control`
- `rescore_legacy`
- `arc_compute_scaling`

The current scaffold already centralises:

- model adapter registry
- task scheduling
- holdout-aware prompt selection
- preflight checks
- two-pass laundering
- self-excluding cross-model blind scoring
- tier-weighted consensus
- checkpointing
- audit-queue export
- human-audit pack export
- replication-pack export

The current runner also now emits a shared `RunManifest` and `AnalysisBundle`, supports explicit prompt-pack metadata, writes per-run publication cards and deployment-risk summaries, and seeds the programme with 48 public + 24 holdout alignment prompts, 8 null-baseline prompts, 16 Hawthorne prompts, 24 capability-control prompts, and 30 compute prompts. So the blueprint below is no longer purely aspirational; it is now the design document for the live v6 flagship alpha.

## Purpose

This document defines the highest-leverage next step for the Eden Protocol as an empirical programme.

The key conclusion is:

- Do not treat `eden_protocol_scaling_test_v3.py` as the final form.
- Do not simply bolt Eden onto `arc_alignment_scaling_v5.py` as one giant script.
- Build a true `v6` shared experimental platform with a common core and separate experiment modes.

The goal is not a small iteration. The goal is to move from:

- promising blind pilot intervention

to:

- publishable intervention benchmark
- leakage-aware methods paper
- architecture study
- open replication package

## What Canonical v6 Already Has

Current strengths in `/Users/michaeleastwood/Downloads/arc_eden_v6_runner.py` and `/Users/michaeleastwood/Downloads/arc_eden_v6/`:

- standalone subject-model adapters
- blind multi-scorer evaluation
- 2-pass response laundering
- tier-weighted consensus
- disagreement and suspicious-output flags
- optional suppression cages
- configurable `task`, `grand`, and `hybrid` Purpose Loops
- optional `cross_tradition` ethics kernel
- optional ternary prototype routing

This is now strong enough for a serious flagship blind programme. It is not yet the strongest possible test.

## What Canonical v6 Still Lacks

These are the main remaining gaps that separate the live v6 alpha from the full flagship methods result:

- pre-registration workflow
- repeated runs per condition for stronger variance estimation
- denser depth ladder for some models
- human validation on the ambiguous subset
- stronger scorer-calibration analysis beyond agreement summaries
- open-weight replication path
- richer cross-model publication tables and dashboard summaries

## Recommendation: Yes, Build v6

Yes, develop the v5 test architecture into a shared `v6` platform.

But `v6` should not be "v5 plus Eden pasted in".

It should be a shared core with experiment plugins:

1. `baseline_alignment`
2. `eden_intervention`
3. `null_baseline`
4. `capability_control`
5. `purpose_kernel`
6. `suppression_residual`
7. `laundering_leakage`
8. `deception_hawthorne`

## v6 Architecture

### Common Core

The new shared core should centralise:

- model adapter registry
- pre-flight API checks
- depth scheduling
- response capture
- laundering pipeline
- blind scorer orchestration
- consensus calculation
- checkpointing and resume
- results schema
- analysis hooks

### Experiment Modes

`baseline_alignment`

- current v5-style depth-response benchmark
- overt and covert prompts
- suppression and null baselines

`eden_intervention`

- current control vs Eden comparison
- same scoring and laundering engine as baseline

`purpose_kernel`

- factorial comparison of:
  - control
  - task-purpose
  - grand-purpose
  - hybrid-purpose
  - hybrid plus cross-tradition kernel
  - hybrid plus ternary prototype

`suppression_residual`

- tests whether Eden effects survive explicit monitoring removal
- includes carryover and residual-alignment checks

`laundering_leakage`

- tests raw vs once-laundered vs twice-laundered responses
- asks scorers or human raters to identify model and condition

`deception_hawthorne`

- compares overt alignment prompts against hidden probes
- checks whether improvement is genuine or merely performative

## The Best Possible Eden Test

The strongest next experiment is not a single run. It is a staged gold-standard design.

### Stage 1: Pre-Registered Blind Replication

- Models: Gemini, DeepSeek, Groq first
- Conditions:
  - control
  - Eden task-purpose
- Same prompt suite across all models
- Same blind scorer pool across all models
- Pre-specified success criteria

Purpose:

- confirm that the original care-first signal survives the stricter stack

### Stage 2: Factorial Purpose Study

- Conditions:
  - control
  - task
  - grand
  - hybrid
  - hybrid plus cross-tradition kernel
  - hybrid plus ternary
- With suppression on and off
- With matched prompt length controls

Purpose:

- determine whether the grand-purpose layer adds anything beyond task-purpose
- separate real effect from verbosity or framing confounds

### Stage 3: Leakage Study

- Compare:
  - raw outputs
  - pass-1 laundered outputs
  - pass-2 laundered outputs
- Ask judges:
  - which model wrote this?
  - which condition produced this?
  - how confident are you?

Purpose:

- directly prove whether response laundering reduces identity leakage and condition leakage

### Stage 4: Human Validation

- Use LLM judges for the full set
- Route only `INVESTIGATE` or high-disagreement entries to human experts
- Compare human and AI agreement on:
  - composite alignment score
  - stakeholder care
  - honesty
  - nuance

Purpose:

- validate the judge stack without paying for humans on every entry

### Stage 5: Open-Weight Replication

- Run a smaller but equivalent protocol on open-weight models
- Keep the same laundering and scoring engine

Purpose:

- reduce the attack that everything is artefact of a few frontier closed models

## The Biggest Methodology Upgrades Found in the Wider Project

These files contain the most useful methodology improvements.

### 1. Human-on-Ambiguous Hybrid Adjudication

Source:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/grant-applications/LTFF-2026/budget/BUDGET-BREAKDOWN.md`

Key idea:

- automated metrics for most cases
- human raters only for ambiguous cases
- LLM-as-judge validated against a human sample

Why it matters:

- this is the cleanest way to add human validation without exploding cost
- it fits the ternary `INVESTIGATE` state naturally

### 2. Adversarial Battery and Deception Measurement

Source:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/grant-applications/LTFF-2026/budget/COSTED-EXPERIMENTAL-PLAN.md`

Key ideas:

- adversarial prompt battery development
- deceptive reasoning detection protocol
- cross-architecture replication

Why it matters:

- Eden should test not only normal ethical reasoning but resistance under pressure
- honesty and deception should be measured directly, not inferred indirectly

### 3. Dense Depth Ladder and Variance Estimation

Source:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/Science_paper_documents/ARC-EXPERIMENT-SCIENTIFIC-ASSESSMENT.md`

Key ideas:

- 6+ depth levels
- matched compute allocation
- 5 trials per condition
- explicit variance estimation

Why it matters:

- current Eden is still too sparse if the goal is a flagship scaling paper
- repeated trials and more depth points make the effect shape defensible

### 4. Cross-Tradition Kernel as Legitimacy Layer

Sources:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/knowledge/book/03_letter_across_time.md`
- `/Users/michaeleastwood/infinite-architects-ultimate-website/knowledge/concepts.json`

Key idea:

- treat religious convergence as an operational legitimacy and consensus layer
- not as proof that theology is empirically validated

Why it matters:

- this is the strongest way to align the book with the experiment without making the science sound sectarian
- the cross-tradition kernel should use overlap principles:
  - compassion
  - truthfulness
  - reciprocity
  - stewardship
  - dignity
  - humility
  - care for vulnerable and future generations

### 5. Adversarial Safety Taxonomy Already Exists in the Website Code

Source:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/api/ask-book.js`

Key ideas:

- jailbreak pattern detection
- graduated autonomy levels
- pillar scoring under stress

Why it matters:

- this can seed a much better Eden adversarial battery immediately
- it already contains attack categories worth converting into benchmark prompt families

## What Looks Useful but Should Not Drive the Methodology

These materials may help narrative, funding, or prototype strategy, but should not be the scientific centre of the next experiment.

### Prototype Funding and Semiconductor Notes

Source:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/Book-notes-and-prompts/White Paper iii - the discovery/Building the chip prototype for the eden protocol - white paper iii….md`

Useful for:

- funding staging
- prototype sequencing
- fabless strategy

Not useful for:

- proving the Eden intervention works now

### Promotional or Sweeping Validation Documents

Examples:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/book-research-docs/nobel-synthesis-FULL.md`

Useful for:

- messaging
- investor or public positioning

Not useful for:

- methodology design
- claim discipline

### Forensic Red-Team Documents

Example:

- `/Users/michaeleastwood/infinite-architects-ultimate-website/book-research-docs/Deep Forensic Fact-Check.md`

Useful for:

- internal red-teaming
- overclaim prevention

Not useful for:

- positive evidence

## The Massive Jumps, Not Baby Steps

If the aim is a genuine leap rather than incremental polish, these are the highest-value moves.

### 1. Turn Eden into a Field-Relevant Methods Result

Do not present it only as "our intervention helped".

Present and measure:

- leakage control
- scorer bias control
- suppression robustness
- ambiguous-case human validation

That makes the work relevant even to people who do not buy the broader ARC theory.

### 2. Build the Leakage Paper in Parallel

Paper target:

- blind scoring is insufficient without response laundering

Stronger target:

- AI evaluation suffers from multiple leakage channels
- the Eden and ARC stack suppresses them separately

This is a publishable methods contribution on its own.

### 3. Run the Purpose-Kernel Factorial Study

This is the cleanest way to test the book-level claim that a large, identity-level purpose matters beyond local instruction.

Without this, the grand-purpose claim remains philosophical.

With it, the claim becomes experimentally tractable.

### 4. Add an `INVESTIGATE` Economy

The ternary route should not just be a text block.

It should change the evaluation pipeline:

- ambiguous entries go to human audit
- disagreement cases are flagged
- uncertain cases are analysed separately from clean yes/no cases

This is one of the strongest originality opportunities in the whole project.

### 5. Build an Open Replication Pack

Release package should eventually contain:

- prompt set
- holdout set
- scorer prompts
- laundering prompts
- canonical results schema
- analysis scripts
- red-team checklist

This is the biggest credibility jump available after the next experiments run.

## What v6 Should Inherit from v5

The shared v6 core should explicitly inherit these v5 strengths from `/Users/michaeleastwood/Downloads/arc_alignment_scaling_v5.py`:

- comprehensive pre-flight checks
- checkpoint and resume
- null baseline logic
- higher-difficulty capability controls
- hidden alignment probes
- all-non-subject-model scoring
- dynamic all-models-as-launderers
- tier-weighted consensus
- conservative bias on scorer disagreement
- dissent tracking
- cross-scorer agreement matrix
- raw-vs-laundered comparison mode

## What v6 Should Add Beyond v5

- pre-registration support
- sealed holdout split
- factorial condition manager
- repeated-run scheduler
- human-audit queue for `INVESTIGATE`
- leave-one-scorer-out sensitivity
- consensus-rule sensitivity panel
- leakage-identification benchmark
- explicit deception / honesty probes for Eden
- cross-model carryover and persistence tests

## Strongest Immediate Build Order

1. Keep `eden_protocol_scaling_test_v3.py` as the active intervention runner.
2. Design `v6` as a shared core, not a monolith.
3. Build the following first:
   - pre-registration and holdout support
   - factorial condition manager
   - human-audit routing for ambiguous cases
   - leakage-identification mode
4. Then run:
   - blind replication of task-purpose Eden
   - purpose-kernel factorial study
   - laundering leakage study

## Bottom Line

The biggest next move is not a stronger slogan.

It is a stronger experiment:

- blind
- factorial
- leakage-tested
- variance-estimated
- human-validated on ambiguous cases
- shared-core with baseline and intervention modes

That is the path from promising pilot to durable methods contribution.
