# External review plan - Paper X

You do not need a PhD or academic rank to get this reviewed. You need to make **one
narrow claim** cheap for a competent person to inspect. Do not ask anyone to "review the
ARC/Eden programme" - ask them to find a fatal flaw in a single theorem or in the protocol.

## Split the review into five lanes (one expert each)

| Lane | What to review | The narrow ask |
|---|---|---|
| Control theory / dynamical systems | Theorems 1-4, the depth-clock proof, β > k | "Under the stated ODE assumptions, is the β > k criterion correctly derived?" |
| Linear systems / non-normal dynamics | Theorem 5, spectral abscissa, transient growth | "Is the spectral-stability claim right, and is the non-normal-transient caveat adequate?" |
| Statistics / stochastic processes | Theorem 6, the OU tail bound | "Is the OU local-Gaussian framing honest, and what breaks it?" |
| ML evaluations / AI safety | the real-model protocol, blinding, reward-hack task design | "Would the β/k estimator give an interpretable measurement if run across six models?" |
| Quantum information | the QEC analogy language **only** | "Is the threshold-form analogy misleading? (It is explicitly not a QEC theorem.)" |

Approach those whose work is *closest*, not the most famous first. The paper already
cites the nearest neighbours: scalable-oversight scaling laws (Engels et al. 2025),
corrigibility (Christiano), model-collapse (Shumailov et al. 2024). A control theorist
familiar with gain scheduling / small-gain is ideal for lanes 1-2.

## The email (narrow, no endorsement ask)

> **Subject:** Request for adversarial technical review of a β > k stability theorem
>
> Dear Dr [Name],
>
> I am an independent researcher preparing a short technical note on a minimal dynamical
> model of recursive self-improvement. The central claim is narrow: under the stated ODE
> assumptions, the misalignment fraction is asymptotically controlled iff the correction
> exponent β exceeds the drift-acceleration exponent k.
>
> I am not asking for endorsement. I am asking whether you can find a fatal flaw, in one
> specific part: [Theorem 3 / the vector extension / the estimator protocol]. I can offer
> [honorarium] for a short written technical review, and I would publish your critique and
> my response, unless you prefer anonymity.
>
> The paper, code, red-team report, and a claim-status ledger (which separates model
> theorem, internal verification, synthetic estimator validation, and a real-model pilot)
> are public: [links].
>
> Respectfully,
> Michael Darius Eastwood

Attach, at most: (1) a 2-page mathematical note, (2) the theorem-proof appendix, (3) the
code link, (4) `results/redteam.md`, (5) a one-page "What this does not claim" (the Scope
and Claim-Status boxes). Do **not** lead with the book, the cosmic framing, or priority.

## Where to post

- **arXiv** (cs.AI / cs.LG / cs.SY / eess.SY) - needs an endorser; ask one narrowly
  ("is this in scope and coherent?", not "do you endorse it?"). arXiv is a moderated
  preprint server, not peer review.
- **OpenReview / workshop** - NeurIPS / ICML / ICLR workshops on AI safety, scalable
  oversight, evaluations, agents, governance. Main-conference acceptance is unlikely
  before real-model evidence.
- **Control venues** - IEEE CDC / ACC / IFAC workshops, once the theorem polish lands.
- **Alignment Forum / LessWrong** - strong adversarial AI-safety critique (not formal
  academic validation).
- **OSF preprint** - already the canonical home; keep it the source of record.
- **GitHub** - open "Adversarial review requested" issues with theorem-specific labels.

## After review

Publish a **response-to-review** document even if the reviews are critical - that is the
single strongest credibility signal an independent researcher can give. The red-team
ledger (`results/redteam.md`) and this repository's correction history already model the
posture: find serious defects, fix them in the open, invite more.
