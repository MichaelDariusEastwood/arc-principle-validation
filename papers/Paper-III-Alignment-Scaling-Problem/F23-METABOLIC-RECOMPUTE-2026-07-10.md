# F23 metabolic-scaling recompute and reconciliation

**Lane:** T6-research (research-facing surfaces) acting on Phase-4 WF-4 papers finishing.
**Date:** 2026-07-10 UTC.
**Scope:** Reconcile the four contradictory published variants of the biology mean-error figure across Paper III, its Fig 14 caption/legend, On the Origin of Scaling Laws, and Executive Summary, per canonical register status `RECOMPUTE_REQUIRED`.
**Basis:** direct read of the paper HTML source (Paper III lines 2011-2076 and figure 14 legend at line 2295; Origin lines 947-1003 and 1084-1137). No underlying CSV, JSON, or analysis-code table for the biology metabolic data exists inside `arc-principle-validation/`; the tables in the HTML papers are the only representation of the dataset. All numerical operations below are performed on the values as printed in the HTML tables.

---

## 1. What the canonical register flagged

The `metabolic_scaling` key of `/Users/michaeleastwood/eden-research/research-management/research-canonical-facts.json` records four contradictory public claims:

| Source | Claim |
|---|---|
| Paper III abstract and §3.6 caption (dagger note) | 2.4% mean error across **11** species groups |
| Paper III Figure 14 legend text (SVG line 2295) | biology mean error **3.1%** across **8** total predictions (of which biology = 3, physics = 5) |
| On the Origin of Scaling Laws, formula-table caption | 2.5% mean absolute error across all predictions; **9** species groups |
| Executive Summary v5 (historical) | 2.5% across **8** systems |

Register status: `RECOMPUTE_REQUIRED`, all four variants blocked for external use until a recompute from Paper III's actual data tables.

---

## 2. What data actually exists

Searches under `arc-principle-validation/`:

- `papers/Paper-III-Alignment-Scaling-Problem/results/` — contains only v1..v5-final blind-evaluation JSON for the alignment scaling experiment (six frontier LLMs). **No biology / metabolic data.**
- `experiments/` — no `F23`, `metabolic`, `allometric`, or `kleiber` directory; only `F23-METABOLIC-NUMBERS-REFINED-2026-07-05.md` (a T2-research note that already flags this exact recompute route).
- No CSV / JSON / analysis-notebook anywhere in the tree that materialises a biology metabolic-error dataset. The only visible tables are the two HTML tables inside Paper III §3.6 and Origin §5 (`The Formula`).
- Prior-version HTMLs (`prior-versions/Paper-III-Alignment-Scaling-Problem-v11.0.html`, `prior-versions/On-the-Origin-of-Scaling-Laws-v2.0.html`, `archive/paper/FINAL-SUITE/archive/Executive-Summary-v5.html`) carry the same contradictory numbers verbatim — the drift is not new to the current version.

**Conclusion on data availability:** the published biology error numbers cannot be reproduced from any repository-internal dataset because none exists. They can only be checked against the values printed inside the HTML tables themselves. That is what this recompute does.

---

## 3. Recompute — Paper III §3.6 biology table (six rows)

Source: `Paper-III-Alignment-Scaling-Problem.html` lines 2023-2071.

| # | Organism group | d | Predicted α = d/(d+1) | Published mean α | Stated error | Recomputed \|meas−pred\|/pred × 100 |
|---|---|---|---|---|---|---|
| 1 | Mammals (3D vascular) | 3 | 0.750 | 0.737 | 1.7% | 1.73% |
| 2 | Birds | 3 | 0.750 | 0.720 | 4.0% | 4.00% |
| 3 | Insects (3D tracheal) | 3 | 0.750 | 0.750 | 0.0% | 0.00% |
| 4 | Jellyfish (2D body plan) | 2 | 0.667 | 0.680 | 2.0% | 1.95% |
| 5 | Flatworms (2D body plan) | 2 | 0.667 | 0.670 | 0.5% | 0.45% |
| 6 | Filamentous fungi (1D cytoplasmic streaming) | 1 | 0.500 | 0.547 | 9.4% | 9.40% |

- **Sum of the six stated errors:** 17.60 percentage points.
- **Row-wise mean of the six errors: 2.933%** (rounds to 2.9%).
- Mean excluding fungi (five rows, i.e. discarding the amber "consistent, not yet confirmed" row): **1.64%**.
- Every stated per-row error matches the recomputed value to within rounding, so the table's per-row arithmetic is correct.

**The row-wise mean of Paper III's own six-row biology table is 2.9%, not 2.4% and not 3.1%.**

---

## 4. Recompute — On the Origin of Scaling Laws formula table (biology plus cosmology)

Source: `On-the-Origin-of-Scaling-Laws.html` lines 947-996.

| # | System | d | Predicted α | Measured α | Stated error |
|---|---|---|---|---|---|
| 1 | Mammals (3D vascular) | 3 | 0.750 | 0.737 | 1.7% |
| 2 | Birds | 3 | 0.750 | 0.720 | 4.0% |
| 3 | Insects | 3 | 0.750 | 0.750 | 0.0% |
| 4 | Reptiles | 3 | 0.750 | 0.760 | 1.3% |
| 5 | 2D biology | 2 | 0.667 | (no valid test organism) | Untested |
| 6 | Universe, matter era | 2 | 0.667 | 0.667 | 0.0% |
| 7 | Universe, radiation era | 1 | 0.500 | 0.500 | 0.0% |

- Six numeric rows (excluding the untested 2D-biology row).
- **Sum of stated errors: 7.00.**
- **Row-wise mean over the six numeric rows: 1.167%** (rounds to 1.2%).
- Origin does not include the Jellyfish, Flatworms, or Fungi rows that appear in Paper III §3.6. Different table, different set.

**The row-wise mean of Origin's own biology-plus-cosmology table is 1.2%, not 2.5%.**

---

## 5. Recompute — physics table (five rows)

Source: identical table appears in both papers (Paper III lines 2214-2251, Origin lines 1095-1132).

| System | d | Predicted α | Measured α | Error |
|---|---|---|---|---|
| KPZ surface roughness (1D) | 1 | 0.500 | 0.500 | 0.0% |
| 2D percolation (specific heat) | 2 | 0.667 | 0.667 | 0.0% |
| Brittle fragmentation (2D) | 2 | 0.667 | 0.670 | 0.5% |
| Earthquake B-value (2D faults) | 2 | 0.667 | 0.667 | 0.0% |
| Brittle fragmentation (3D) | 3 | 0.750 | 0.750 | 0.0% |

- Sum = 0.5, **row-wise mean = 0.10%**.
- Claim "less than 0.2% mean error" **is confirmed**.

This is the only one of the four biology-related summary figures that reproduces from its own table.

---

## 6. Attempted reconstructions of the "3.1%" and "8 predictions" claims

Paper III Figure 14 legend explicitly says "Biology (3 predictions), Physics (5 confirmations)". SVG-coordinate inspection of the three blue circles in the figure (offsets against the plot's stated y-axis 0.50-0.80) gives biology measured values of 0.547, 0.680, 0.746 — which map to Fungi (d=1), Jellyfish (d=2), and a d=3 3D-organism group. Reconstructing the "3 biology predictions" three ways:

| Reconstruction | Errors used | Mean |
|---|---|---|
| Grouped-by-dimension (Mammals+Birds+Insects → 0.7357, Jellyfish+Flatworms → 0.6750, Fungi) | 1.91%, 1.20%, 9.40% | **4.17%** |
| Alt: Mammals-only + Jellyfish-only + Fungi | 1.73%, 1.95%, 9.40% | **4.36%** |
| 6-row Paper III biology, row-wise | 1.73, 4.00, 0.00, 1.95, 0.45, 9.40 | **2.93%** |

None of these equals 3.1%. The 8-total (3 biology + 5 physics) reconstruction gives 1.63% (grouped biology) or 1.65% (all six biology + five physics) — neither matches 2.5%.

**No subset of the visible tables yields either 2.4%, 2.5%, or 3.1% exactly.** The "11 species groups", "9 species groups", and "8 systems" species-counts likewise do not correspond to the row counts of any visible table. They appear to reference underlying literature datasets (e.g. West-Brown-Enquist 1997 taxonomic groupings, Glazier 2005 review categories) that are **not** materialised anywhere in the arc-principle-validation repository.

---

## 7. Which variant survives?

**None of the three biology-summary variants (2.4%, 2.5%, 3.1%) survives as-published.** The only figure that reproduces from its own table is the physics claim "less than 0.2%". The three biology claims should be replaced by the row-wise means that actually derive from the printed tables:

- Paper III §3.6 biology table (6 rows, individual organism groups): **2.9%** mean absolute error.
- Origin formula table biology-plus-cosmology (6 numeric rows): **1.2%** mean absolute error.
- Physics table (5 rows, identical in both papers): **0.10%** mean absolute error (rounds to <0.2% as claimed).

The species-group counts ("11", "9", "8") are unverifiable from the repository and must be either (a) recovered from the underlying literature datasets or (b) retired in favour of the visible row counts.

---

## 8. Files and claims that need updating

| File | Line(s) | Current text | Replacement (contingent on T4-research ratification) |
|---|---|---|---|
| `papers/Paper-III-Alignment-Scaling-Problem/Paper-III-Alignment-Scaling-Problem.html` | 602 (abstract) | "predicting metabolic scaling exponents across 11 species groups ... to 2.4% mean error" | "across the six organism groups listed in §3.6 (mammals, birds, insects, jellyfish, flatworms, filamentous fungi) to 2.9% mean absolute error (row-wise); the underlying literature comprises 11 species-group datasets that require reingestion for a taxon-weighted mean" |
| same file | 1634 (Table dagger footnote) | "Mean error 2.4% across 11 species groups" | "Mean error 2.9% across six organism groups in §3.6 (row-wise); 11-species-group taxon-weighted mean pending literature reingestion" |
| same file | 2295 (Fig 14 SVG legend) | "biology mean error 3.1%" | "biology mean error 2.9% (six-row row-wise), physics mean error 0.10%" |
| `papers/On-the-Origin-of-Scaling-Laws/On-the-Origin-of-Scaling-Laws.html` | 1001 (formula table caption) | "Mean absolute error across all predictions: 2.5%" | "Mean absolute error across the six numeric rows (biology plus cosmology): 1.2%" |
| same file | 537 (§ header text) | "9 species groups; species-group count pending recompute" | "six organism-plus-cosmology rows shown in §5; 9-species-group taxon-weighted mean pending literature reingestion" |
| `archive/paper/FINAL-SUITE/archive/Executive-Summary-v5.html` | 586, 808 | "2.5% across 8 systems" | historical — leave as-is, but link to this reconciliation from the top of the file |
| `prior-versions/Paper-III-Alignment-Scaling-Problem-v11.0.html`, `prior-versions/On-the-Origin-of-Scaling-Laws-v2.0.html` | multiple | same drift as current versions | historical — no edit |

All four contradictory variants must not be quoted in any new external-facing surface (website, grant, book) until T4-research ratifies replacement wording.

---

## 9. Proposed canonical-register patch (for T4-research ratification only, do not apply)

Target: `/Users/michaeleastwood/eden-research/research-management/research-canonical-facts.json`, key `metabolic_scaling`.

```json
{
  "metabolic_scaling": {
    "status": "RECOMPUTED_2026-07-10_ROW-WISE_ONLY",
    "verified_by_lane": "T6-research (recompute) → awaiting T4-research ratification",
    "source_basis": "direct read of Paper III §3.6 HTML table (6 rows) and Origin §5 HTML table (7 rows, 6 numeric); no underlying CSV/JSON/notebook exists in arc-principle-validation",
    "row_wise_means_verified": {
      "paper_iii_bio_6row_percent": 2.933,
      "paper_iii_bio_6row_rounded": "2.9%",
      "origin_bio_plus_cosmology_6row_percent": 1.167,
      "origin_bio_plus_cosmology_6row_rounded": "1.2%",
      "physics_5row_percent": 0.10,
      "physics_5row_rounded": "0.10% (matches published <0.2% claim)"
    },
    "row_wise_tables": {
      "paper_iii_bio": [
        {"group":"Mammals","d":3,"pred":0.750,"meas":0.737,"err_pct":1.73},
        {"group":"Birds","d":3,"pred":0.750,"meas":0.720,"err_pct":4.00},
        {"group":"Insects","d":3,"pred":0.750,"meas":0.750,"err_pct":0.00},
        {"group":"Jellyfish","d":2,"pred":0.667,"meas":0.680,"err_pct":1.95},
        {"group":"Flatworms","d":2,"pred":0.667,"meas":0.670,"err_pct":0.45},
        {"group":"Filamentous fungi","d":1,"pred":0.500,"meas":0.547,"err_pct":9.40}
      ],
      "origin_bio_plus_cosmology": [
        {"system":"Mammals","d":3,"pred":0.750,"meas":0.737,"err_pct":1.73},
        {"system":"Birds","d":3,"pred":0.750,"meas":0.720,"err_pct":4.00},
        {"system":"Insects","d":3,"pred":0.750,"meas":0.750,"err_pct":0.00},
        {"system":"Reptiles","d":3,"pred":0.750,"meas":0.760,"err_pct":1.33},
        {"system":"Universe matter era","d":2,"pred":0.667,"meas":0.667,"err_pct":0.00},
        {"system":"Universe radiation era","d":1,"pred":0.500,"meas":0.500,"err_pct":0.00}
      ]
    },
    "retired_variants": {
      "2.4%_across_11_species_groups": "RETIRED — not derivable from any visible table row-wise. '11 species groups' refers to underlying literature datasets that are not materialised in the repository.",
      "2.5%_across_all_predictions_or_8_systems": "RETIRED — not derivable from Origin's own 6-row table (row-wise mean = 1.2%) nor from any biology+physics combination.",
      "3.1%_biology_of_8_predictions": "RETIRED — Fig 14 grouped-by-dimension recompute gives 4.17%, six-row row-wise gives 2.93%; 3.1% is not reproducible from any visible subset."
    },
    "surviving_claim": "physics mean error <0.2% (5-row table, row-wise mean 0.10%) — VERIFIED",
    "external_use_gate": "BLOCKED until T4-research ratifies replacement wording per Section 8 of F23-METABOLIC-RECOMPUTE-2026-07-10.md",
    "action_to_reinstate_11_or_9_species_wording": "reingest the underlying WBE-1997 / Glazier-2005 / Aguilar-Trigueros-2017 / Thommen-2019 taxonomic datasets, compute a taxon-weighted rather than row-wise mean, and record the ingestion trace under experiments/F23-metabolic-source-data/",
    "provenance": {
      "recompute_report": "arc-principle-validation/papers/Paper-III-Alignment-Scaling-Problem/F23-METABOLIC-RECOMPUTE-2026-07-10.md",
      "computed_by": "T6-research",
      "computed_at_utc": "2026-07-10"
    }
  }
}
```

DO NOT apply this fragment. Canon writes are gated. This is the proposed replacement structure only; T4-research is the release-oracle for the research swarm and holds the ratification authority.

---

## 10. What is missing and how to obtain it

To restore any variant that references "11 species groups", "9 species groups", or a taxon-weighted rather than row-wise mean:

1. **Recover the underlying literature datasets.** Named in the papers as: West, Brown & Enquist (1997, *Science* 276:122); Kleiber (1932, *Hilgardia* 6:315); Aguilar-Trigueros et al. (2017, *ISME J* 11:2175); Thommen et al. (2019, *Cell Reports* 27:3462); Glazier (2005, *Biol Rev* 80:611); White, Phillips & Seymour (2006, *Biol Lett* 2:125); Cambui (2025, *Preprints* 202501.0001). These are the sources from which "11 species groups" would be constructed.
2. **Materialise them as a CSV** at `experiments/F23-metabolic-source-data/species_groups.csv` with columns `group, source_ref, n_species, d, predicted_alpha, measured_alpha_mean, measured_alpha_sd`.
3. **Author a taxon-weighted recompute script** at the same path (`recompute_row_and_weighted.py`) that produces row-wise mean, taxon-weighted mean, and CI. Emit output to `experiments/F23-metabolic-source-data/RESULT.json`.
4. **Route to T4-research** for ratification, then update the canonical register from `RECOMPUTED_2026-07-10_ROW-WISE_ONLY` to `RECOMPUTED_TAXON-WEIGHTED`.

Operator or T4-research action needed for step 1 — the source PDFs are not in this repository.

---

## 11. Continuation state

`working` (T6-research). This report is the row-wise recompute only; taxon-weighted recompute requires source-literature acquisition per §10.

Downstream dispatch on ratification: T4-research (research release-oracle), T2-research (canon wording after ratification), website portal sync after wording is locked.

End of report.

---

## Addendum: adversarial verification pass (2026-07-10, T6-research surge)

This addendum records an adversarial verification pass over the report above. It confirms one part, retracts another, and revises the recommendation.

### Arithmetic confirmed

The row-wise arithmetic in the body of this report reproduces correctly:

- Paper III six-row biology mean: 2.933 per cent.
- Origin six-row biology mean: 1.167 per cent.
- Physics five-row mean: 0.10 per cent.

These figures stand.

### Central premise retracted

The report's central "no data exists" premise, as asserted in sections 2, 6, 9, and 10, is wrong and is treated as retracted. A structured metabolic dataset does exist in this repository at:

`experiments/cauchy-unification__Paper-VII/results/results_50_domain_validation.json`

It holds 13 published-exponent metabolic-scaling entries (`predicted_exponent`, `published_exponent`, and `nearest_comparator.abs_error`) covering Mammals, Birds, Fish, Reptiles, Insects, Amphibians, Crustaceans, Jellyfish, Cnidarians, Ctenophores, and three fungi groups. Raw Kleiber mass-BMR data also lives in `experiments/domain-validation__Foundational-and-Origin/arc_20_domain_universal_test.py` at approximately lines 265 to 295.

### Row-wise aggregation of the Paper VII 13-entry dataset

Aggregating that Paper VII dataset row-wise gives:

- All 13 rows: 4.13 per cent mean absolute error.
- 10 non-fungi rows: 2.57 per cent mean absolute error, a near-match to the historical "2.5 per cent" variant this report had retired.
- 11-group construction (10 non-fungi rows plus one combined-fungi row): 3.19 per cent mean absolute error, a near-match to the historical "3.1 per cent" variant this report had retired.

### Consequence

The historical 2.5 per cent and 3.1 per cent figures are plausibly derivable from repository data under reasonable grouping choices. This report's claim in sections 2, 6, 9, and 10 that they are "not reproducible from any visible subset" is not established and is withdrawn.

### Revised recommendation

Do not hard-retire the 2.4 per cent, 2.5 per cent, or 3.1 per cent variants as "not derivable". Instead:

1. Treat `experiments/cauchy-unification__Paper-VII/results/results_50_domain_validation.json` as the candidate canonical source.
2. Run a documented taxon-weighted aggregation from it, and decide the grouping explicitly (all-13 vs non-fungi-10 vs 11-group).
3. Mark each historical variant as "requires reconciliation against the Paper VII 50-domain dataset" rather than "RETIRED".

Only the physics "less than 0.2 per cent" survival claim is unambiguously safe as stated.

### Canonical-register patch must be rewritten

The proposed canonical-register patch in section 9 of this report must be rewritten before any T4-research ratification, because its rationales depended on the false "no data" premise. This is noted here explicitly so that downstream lanes do not adopt section 9 as-is.

### Status

F23 remains `RECOMPUTE_REQUIRED`. The next action is the Paper VII aggregation described above, which is a repository-internal computation and requires no new literature acquisition. Assign to T2-research and T5-research.
