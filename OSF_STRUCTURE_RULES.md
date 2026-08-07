# OSF structure rules · standing canon

**Operator directive, 8 August 2026: every paper has its own component. If a paper does not have
one, create it.** These rules are binding on every lane and every automated upload path.

---

## 1 · The rule, and the mechanical reason for it

**Every paper in the programme has its own OSF component, and every preregistration registers from
the component of the paper it belongs to.**

**The reason is not tidiness.** An OSF registration is a **frozen snapshot of the node it is made
from**. Register forty-two preregistrations from `6c5xb` and you get forty-two frozen copies of the
entire programme. A reviewer opening the thirty-seventh finds the whole estate inside it, and the
one study they came to check is buried.

Register Paper IV.d's follow-on from Paper IV.d's component and the snapshot contains Paper IV.d's
materials. **That is what a reader expects when they follow a DOI from a paper to its
preregistration, and anything else costs the reader time for no gain.**

## 2 · The structure, as it stands

`6c5xb` **The ARC Principle** is the top-level project. Every paper is a **child component** of it,
`category: project`, public. Twenty-seven children as of 8 August 2026.

| registration prefix | component | node |
|---|---|---|
| `paper-i--` | Paper I | `b6n27` |
| `paper-ii--` | Paper II | `8fjma` |
| `paper-iii--` | Paper III | `hqcgf` |
| `paper-iv-a--` | Paper IV.a | `mb9r6` |
| `paper-iv-b--` | Paper IV.b | `a7r56` |
| `paper-iv-c--` | Paper IV.c | `j3q2e` |
| `paper-iv-d--` | Paper IV.d | `2s3e6` |
| `paper-v--` | Paper V | `kzeya` |
| `paper-vi--` | Paper VI | `8ez2n` |
| `paper-vii--` | Paper VII | `x6wa7` |
| `paper-viii--` | Paper VIII | `7yj4e` |
| `paper-ix--` | Paper IX | `k7ruz` |
| `paper-x--` | Paper X | `bse2q` |
| `paper-xi--` | Paper XI | `dc9gw` |
| **`paper-xii--`** | **Paper XII: Public Benchmark Rescoring** | **`3tzp7`** created 8 Aug 2026 |
| `paper-c--` | Paper C | `rbhdp` |
| `paper-hrih--` | HRIH | `uydxq` |
| `paper-foundational--` | Cauchy Framework, Foundational | `y7qgd` |
| `paper-origin-scaling--` | On the Origin of Scaling Laws | `xzy9u` |
| `paper-eden-vision--` | Eden Protocol: Philosophical Vision | `9m3dg` |
| **`study-*`, `arc-align-instrument-*`** | **ARC-Eden Study Series** | **`awz3h`** created 8 Aug 2026 |

## 3 · Studies share one component. A study promoted to a paper does not.

The `study-*` and `arc-align-instrument-*` registrations are **studies, not papers**, and they
share `awz3h`.

**Twenty-three separate components would be twenty-three nodes of metadata to configure, twenty
three chances to misconfigure contributors, and it would turn a manual review of the registration
into a review of the registration and its node.** The rule is per paper, and a study is not a paper.

**When a study becomes a paper it gets its own component**, and its next registration moves there.
The earlier registration stays where it was filed: a registration's node is part of its permanent
record and moving it is not available.

## 4 · What must never happen

**A registration must never be filed from a node that is not in the table above.** An upload path
that cannot resolve a registration to a node **fails loudly and stops**. It does not fall back to
`6c5xb`, and it does not pick something plausible.

**A draft on the wrong node is the single hardest error for a human reviewer to catch**, because
the registration content reads correctly and only the parent is wrong. By the time it is submitted
the snapshot is permanent.

**`awjr4` Eden Protocol: Engineering Specification is private and was withdrawn on 14 July 2026
pending patent and disclosure review. Nothing registers from it.**

## 5 · Two registrations are held from filing entirely

`study-p--pre-training-placement` and `study-q--component-level-ablation` both disclose the
**compute-matched sham** measurement method, which the 5 August 2026 patent assessment identified
as the only asset that is both non-public and plausibly patentable.

**Operator directive: disclose nothing patentable.** So they have a home in `awz3h` and they do not
upload. **Forty-two green, forty upload.**

## 6 · Draft only. A human submits.

**Operator directive: the AI creates and perfects drafts. A human reviews and submits.**

Creating a draft and populating it are reversible. **Submission is not:** a registration cannot be
edited, and withdrawing leaves a tombstone and loses the date, which is the entire reason for
filing. The submit path must not exist in the upload code at all, and `G-OSF-DRAFT-ONLY` enforces
that at build time.

**Never delete anything from the OSF account.** No `DELETE` verb in any tool that touches it.

## 7 · Creating a component, for the next lane that has to

Match the existing pattern rather than inventing one. `POST /v2/nodes/6c5xb/children/` with
`category: project` and `public: true`, and **check the title against the existing children first**,
because a duplicate component is a second home for the same paper and there is no way to tell which
one a future registration should have used.
