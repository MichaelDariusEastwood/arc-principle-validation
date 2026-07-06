# OSF deposit pack - copy-paste descriptions for posting Paper X

> Purpose: everything needed to post *The Coupled Co-Scaling Law* and its
> apparatus to OSF professionally. Each block is labelled with the OSF field it
> belongs in. Honest throughout: states the priority and the verified soundness,
> and states plainly that empirical confirmation on real systems is still open.

## Dating (do this, it is the strongest and most honest option)
- **Document date stays 26 June 2026.** Do not alter the masthead. It is the priority timestamp and is corroborated by dated GitHub commits.
- **OSF deposit date is 29 June 2026** (today), recorded automatically by OSF. Do not backdate it.
- State **both** in the description. Four corroborating timestamps (book copyright 8 Dec 2024; book published 2 Jan 2026; paper 26 June 2026; OSF deposit 29 June 2026; plus the GitHub commit hash) are stronger priority evidence than any single date.

## Recommended structure (simple + professional)
One **public OSF Project**. Either upload a snapshot of the paper folder to OSF
Storage, or connect the GitHub repo via the OSF **GitHub add-on** (keeps it in
sync; recommended) and additionally upload the **PDF** to OSF Storage so the
archival copy is independent of GitHub. Components are optional; descriptions for
four of them are provided below if you want the tidier layout.

---

## BLOCK 1 - PROJECT TITLE  (paste into: Title)

```
The Coupled Co-Scaling Law: A Falsifiable Threshold Criterion for the Stability of Recursive Self-Improvement
```

---

## BLOCK 2 - PROJECT DESCRIPTION  (paste into: Description)

```
This project archives the working paper "The Coupled Co-Scaling Law" together with its complete computational apparatus: a runnable verification harness, independent theorem test suites, a pre-registered real-model experiment protocol with its harness, and a SHA-256 integrity manifest.

Most work on recursive self-improvement treats it as a problem of speed: capability may grow explosively and outrun supervision, so the proposed lever is to cap the growth rate. From a minimal dynamical model, this paper derives that the operative lever is different. Stability is set not by the growth rate but by a single inequality between two scaling exponents: the rate at which a misalignment-correcting force strengthens with capability (beta) must exceed the rate at which drift accelerates with capability (k). The criterion is beta > k.

Its sharpest consequence addresses the central fear of the field: a hard takeoff, even a genuine finite-time intelligence explosion, drives the modelled misalignment fraction to zero if and only if beta > k, and the speed of the explosion does not change that asymptotic verdict. The criterion shares the threshold form of the quantum error-correction sub-threshold condition, offered as a falsifiable hypothesis (the model's suppression law is power-law, not exponential). The closed-form predictions are checked by a runnable harness, and the theorem statements are independently re-derived in a standalone test suite.

Scope and honesty. The results hold within the stated minimal model. The decisive empirical question, whether real self-improving systems satisfy the criterion, remains the open problem; a pre-registered protocol and a runnable real-model harness are included to make that test possible. The paper has been hardened against three independent adversarial reviews.

Author: Michael Darius Eastwood (independent researcher; ARC/Eden research programme). Working paper dated 26 June 2026; deposited to OSF 29 June 2026. Code and full commit history: https://github.com/MichaelDariusEastwood/arc-principle-validation
```

---

## BLOCK 3 - TAGS / KEYWORDS  (paste into: Tags, one at a time or comma-separated)

```
AI safety, AI alignment, recursive self-improvement, intelligence explosion, hard takeoff, scalable oversight, scaling laws, dynamical systems, Lyapunov stability, threshold criterion, quantum error correction analogy, falsifiability, AI governance, control theory, co-scaling
```

---

## BLOCK 4 - CATEGORY & SUBJECTS

- **Category** (dropdown): `Project`
- **Subjects** (OSF/bepress taxonomy - add these paths):
  - Physical Sciences and Mathematics → Computer Sciences → Artificial Intelligence and Robotics
  - Physical Sciences and Mathematics → Computer Sciences → Theory and Algorithms
  - Physical Sciences and Mathematics → Applied Mathematics → Dynamic Systems

---

## BLOCK 5 - LICENSE

- Recommended: **CC-BY 4.0** (Creative Commons Attribution). It maximises reach while making attribution a licence condition, which protects your priority. Select it in OSF under "Add a license".

---

## BLOCK 6 - WIKI HOME PAGE  (paste into: Wiki, the "home" page)

```markdown
# The Coupled Co-Scaling Law

A falsifiable threshold criterion for the stability of recursive self-improvement:
**a misalignment-correcting force must out-scale drift-acceleration, i.e. `beta > k`.**

**Author:** Michael Darius Eastwood (independent; ARC/Eden research programme)
**Working paper dated:** 26 June 2026 · **Deposited to OSF:** 29 June 2026
**Code (full commit history):** https://github.com/MichaelDariusEastwood/arc-principle-validation

## The claim, precisely
From a minimal dynamical model, stability under recursive self-improvement is set
not by the growth rate but by an inequality between two scaling exponents:
correction strength scales as capability^beta, drift acceleration as capability^k,
and the system is alignment-stable iff `beta > k`. A finite-time intelligence
explosion drives the modelled misalignment fraction to zero iff `beta > k`, and the
explosion's speed does not change that asymptotic verdict.

## What is original (priority)
1. the dimensionless control parameter `rho = gamma1 r / A`;
2. the sharpened stability criterion `beta > k` under accelerating self-improvement;
3. the Hard-Takeoff Depth-Regularity Theorem;
4. identification of the compounding drift channel as the locus of genuine divergence.
The underlying conceptual thesis was first set out in the book *Infinite Architects*
(copyright deposited 8 December 2024; published 2 January 2026); the formal,
measurable results here are the 2026 form of that thesis.

## What is NOT claimed (honesty)
- Not a proof of AI alignment.
- Not an empirically confirmed law: the decisive test, whether real self-improving
  systems satisfy the criterion, is the open problem. A pre-registered protocol and
  a runnable real-model harness are provided to make that test possible.
- The quantum-error-correction correspondence is offered as a falsifiable hypothesis
  (power-law suppression, not exponential).

## Standing
The proofs are independently re-derived in a standalone test suite, and the paper
has been hardened against three independent adversarial reviews.

## Contents
- `Paper-X-Coupled-CoScaling-Correction.pdf` / `.html` - the paper
- `CLAIMS.md` - itemised claim ledger
- `code/` - verification harness + theorem test suites
- `experiments/` - real-model protocol, harness, and the zero-context launch runbook
- `results/` - verdict tables and run records
- `MANIFEST.json` - SHA-256 of every file (integrity)
- `NEGATIVE_RESULTS.md`, `SECURITY.md`, `REPRODUCIBILITY.md`

## How to cite
> Eastwood, M. D. (2026). *The Coupled Co-Scaling Law: A Falsifiable Threshold
> Criterion for the Stability of Recursive Self-Improvement.* ARC/Eden research
> programme. OSF. https://doi.org/10.17605/OSF.IO/6C5XB
```

---

## BLOCK 7 - COMPONENT DESCRIPTIONS  (optional; paste into each Component's Description)

**Component "Paper & Claims"**
```
The paper (PDF and HTML) and the itemised claim ledger (CLAIMS.md). States the beta > k criterion, the Hard-Takeoff Depth-Regularity Theorem, the minimal-model scope, and the priority disclosure. Dated 26 June 2026.
```

**Component "Verification Code & Tests"**
```
Deterministic verification harness (experiment_coscaling.py) that checks the closed-form predictions against the model, plus two pytest suites: a regression suite and an independent theorem re-derivation (no harness import) that re-proves the corrected theorem statements from scratch. Offline; no API calls.
```

**Component "Real-Model Experiment: Protocol & Harness"**
```
Pre-registered protocol (PROTOCOL_V2.md, hypotheses H1-H4) and a real-model harness for testing whether the co-scaling mechanism appears on a live model: three task families, a sham-extra-compute control, fused static-plus-blind-panel misalignment scoring, and matched-pair bootstrap CIs. Includes RUN_REAL_MODEL_EXPERIMENT.md, a zero-context launch runbook. The harness executes model-generated code and must be run in an isolated sandbox (SECURITY.md). No confirmatory run is included; this is the instrument, not a result.
```

**Component "Results & Integrity"**
```
Verdict tables from the deterministic checks, the v1 mechanism-probe record, and MANIFEST.json (SHA-256 of every artefact). Real-model results are labelled for exactly what they support; selftest outputs are marked NOT DATA.
```

---

## BLOCK 8 - HOW TO CITE  (paste into: Description footer or Wiki)

```
Eastwood, M. D. (2026). The Coupled Co-Scaling Law: A Falsifiable Threshold Criterion for the Stability of Recursive Self-Improvement. ARC/Eden research programme. OSF. https://doi.org/10.17605/OSF.IO/6C5XB
```

---

## BLOCK 9 - FILE / FOLDER UPLOAD MAP

Upload the contents of `papers/Paper-X-Coupled-CoScaling-Correction/`:

| Upload | What it is |
|---|---|
| `Paper-X-Coupled-CoScaling-Correction.pdf` | the paper (archival; upload to OSF Storage even if you also link GitHub) |
| `Paper-X-Coupled-CoScaling-Correction.html` | the paper (source) |
| `README.md` | repository overview |
| `CLAIMS.md` | itemised claim ledger |
| `code/` | verification harness + test suites |
| `experiments/` | protocols, real-model harness, launch runbook |
| `results/` | verdict tables + run records |
| `figures/` | publication figures |
| `MANIFEST.json` | SHA-256 integrity manifest |
| `NEGATIVE_RESULTS.md`, `SECURITY.md`, `REPRODUCIBILITY.md`, `REVIEW_OUTREACH.md` | honesty + reproducibility docs |
| `requirements.txt`, `Makefile`, `Dockerfile` | reproducible environment |

Tip: the cleanest route is OSF's **GitHub add-on** (Project → Add-ons → GitHub) to
link `arc-principle-validation`, plus a direct upload of the **PDF** to OSF Storage
so the archived paper does not depend on the repo staying public.

---

## BLOCK 10 - POSTING CHECKLIST

1. Create / open the public Project (node 6C5XB).
2. Paste Title (Block 1) and Description (Block 2).
3. Add Tags (Block 3), Category + Subjects (Block 4), License CC-BY 4.0 (Block 5).
4. Paste the Wiki home page (Block 6).
5. Connect GitHub via the add-on and/or upload files (Block 9); always upload the PDF.
6. (Optional) create the four components and paste their descriptions (Block 7).
7. Add yourself as contributor; add your ORCID if you have one.
8. Confirm the project is Public, then "Create DOI" if not already minted (6C5XB).
9. Leave the document's 26 June 2026 date untouched; OSF will stamp the 29 June 2026 deposit automatically.
```
```
