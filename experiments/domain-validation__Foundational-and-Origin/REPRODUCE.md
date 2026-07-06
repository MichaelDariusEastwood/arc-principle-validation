# Reproduce — Domain Validation (Foundational & Origin papers)

> **Provenance note (T2-research, 2026-07-05):** Assembled from this directory's
> committed scripts and pre-computed `results_*.txt` outputs. This is the source
> validation programme for the Foundational and On-the-Origin papers; Paper VII's
> suite was later factored out of here (see `cauchy-unification__Paper-VII/README.md`).
> Written to close a reproducibility gap (this dir had result files but no run doc).
> HOLD for T5 verification.

## What this is
The cross-domain empirical validation harness for the ARC Principle's prediction
α = d/(d+1): predicted scaling exponents against measured exponents across physics,
biology, and cosmology domains. Outputs are the `results_*.txt` files already in
this directory.

## Environment
Python 3 with **NumPy**, **SciPy**, and **matplotlib** (a few scripts also use
`dataclasses`, standard library on Python 3.7+). No network access required.

## Entry-points (each is a standalone `python3 <script>` run)
| Script | What it does |
|--------|--------------|
| `arc_1d_prediction_test.py` | The 1D prediction test (α at d=1 → 1/2). |
| `arc_20_domain_universal_test.py` | 20-domain blind structured-prediction comparison (legacy/exploratory — permissive scoring; provenance only). |
| `arc_physics_domains_test.py` | Physics-domain exponent predictions vs measured. |
| `arc_acoustic_time_crystal_test.py` · `arc_real_time_crystal_test.py` | Time-crystal validations (synthetic + real experimental data). |
| `arc_einstein_verification.py` | E=mc² / chain-reaction consistency check. |
| `arc_rigorous_validation.py` | Rigorous validation v2.0 (methodological fixes applied). |
| `arc_definitive_test.py` | Cross-domain blind prediction test. |
| `arc_section7_breakthrough.py` | Section-7 breakthrough contributions. |
| `arc_unified_paradigm_test.py` | Unified-paradigm synthesis across regimes. |
| `arc_universal_proof.py` | Universal proof from Cauchy functional equations + max entropy. |
| `arc_complete_test_suite.py` | Runs all computationally feasible tests. |
| `prove_IxR_equals_complexity_v2.py` | Numerical proof: Intelligence × Recursion ↔ complexity (corrected). |
| `generate_canonical_results.py` | Builds the canonical results map + verification snapshots. |
| `generate_phase0_evidence_pack.py` | Generates the bounded Phase-0 evidence pack. |

## To reproduce
1. **Env:** `python3 -m pip install numpy scipy matplotlib`
2. **Run the full suite:** `python3 arc_complete_test_suite.py` (or any single script above).
3. **Compare** the printed output against the committed `results_*.txt` of the same name (e.g. `arc_rigorous_validation.py` → `results_arc_rigorous_validation.txt`).
4. **Rebuild the canonical map:** `python3 generate_canonical_results.py`.

## Honest scope (LAW 1)
- `arc_20_domain_universal_test.py` is **legacy/exploratory** (permissive "within-tolerance" scoring) — retained for provenance, not a canonical headline. The strict canonical successor lives in `cauchy-unification__Paper-VII/` (`arc_50_domain_universal_test.py`).
- Several results are **mathematical/numerical verifications of proven theorems** (Cauchy equations, α=d/(d+1)) — "Supporting" evidence tier, not independent empirical confirmation.
- Measured exponents cited in these tests should be reconciled to one source per domain (see the metabolic-exponent reconciliation note, 2026-07-05).

## Outputs
The `results_*.txt` files in this directory (one per script). No files are overwritten by re-running unless you redirect output.
