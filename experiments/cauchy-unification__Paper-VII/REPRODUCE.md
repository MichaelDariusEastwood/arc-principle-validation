# Reproduce - Cauchy Unification (Paper VII)

> **Provenance note (T2-research, 2026-07-05):** This directory's full reproduction
> instructions already live in **`README.md`** (scripts table, results→script map,
> data/preregistration assets, exact run commands, evidence tiers, and honest caveats).
> This `REPRODUCE.md` exists only so the reproduction doc is discoverable under the
> conventional filename used across the other experiment directories — it does not
> duplicate or supersede the README. **Read `README.md` in this directory.**

## Quick start
```bash
# Current canonical tiered suite (primary):
python3 scripts/arc_50_domain_universal_test.py
# Null / negative controls:
python3 scripts/run_null_controls.py
# Cauchy no-go theorem + mathematical foundation:
python3 scripts/arc_rigorous_validation.py
python3 scripts/arc_universal_proof.py
```
Requires Python 3 + NumPy + SciPy. (R replication `cross_validate_fits.R` needs `Rscript`, currently a documented local blocker - see `results/cross_library_replication_status.md`.)

## Canonical headline (per README, honest scope)
- Legacy baseline-20 under the strict runner: **15/20** empirical family matches (p = 1.67e-4).
- Expanded empirical cohort: **19/25** empirical family matches (p = 1.56e-5).
- 50-domain suite: tiered (published-exponents direct 13/13; analytic identities 6/6), **not** one blended p-value.
- The legacy `arc_20_domain_universal_test.py` uses a permissive "R² within 0.05" rule and is **retained for provenance only** - not the canonical headline.

Full detail, evidence tiers, and the pre-registration status caveats: **`README.md`**.
