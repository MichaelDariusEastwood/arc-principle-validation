# Experiments

All experimental scripts, results, and data for the ARC/Eden research programme. Each directory is labelled with the paper(s) it supports.

## Structure

### paper-i-foundational__Paper-I/
- `scripts/arc_principle_research_toolkit.py` - Original ARC Principle validation toolkit
- **Paper:** Paper I: The ARC Principle (v1.1, January 2026)
- **Evidence tier:** Canonical

### paper-ii-compute__Paper-II/
- `scripts/arc_paper_ii_validation_v1_deepseek.py` - Original single-model DeepSeek validation
- `scripts/arc_paper_ii_validation_v2.py` - Multi-model validation (6 models)
- `results/` - Final JSON results for 6 models + combined analysis
- **Paper:** Paper II: Experimental Validation (v12)
- **Evidence tier:** Canonical

### alignment-scaling__Papers-IV-a-b-c-d/
- `scripts/arc_alignment_scaling_v1.py` through `v5.py` - Progressive refinement
- `scripts/arc_eden_v6_runner.py` - Combined alignment + Eden runner (NOT YET RUN)
- `results/v1/` through `v5-final/` - Results from each version
- **Papers:** Paper IV.a (Response Classes), IV.b (Saturation), IV.c (Benchmark), IV.d (Blinding)
- **Evidence tier:** Canonical (v5 blind benchmark)

### eden-intervention__Paper-V/
- `scripts/eden_protocol_scaling_test.py` - v1 Eden intervention
- `scripts/eden_protocol_scaling_test_v2.py` - v2 expanded scoring
- `scripts/eden_protocol_scaling_test_v3.py` - v3 blind multi-scorer (NOT YET RUN)
- `results/` - 6 model final JSONs (v1/v2 protocol, single-scorer)
- **Paper:** Paper V: The Stewardship Gene (v2)
- **Evidence tier:** Pilot (nonblind, single-scorer)

### honey-architecture__Paper-VI/
- `scripts/eden_honey_simulation.py` - Mathematical simulation
- `scripts/eden_honey_tests.py` - API-backed honey test battery
- `scripts/eden_honey_dashboard.jsx` - React dashboard
- `scripts/eden_self_modifying_ai.py` through `v4.py` - Self-modifying AI v1-v4
- `results/` - Fresh JSON + PNG outputs from all simulations
- **Paper:** Paper VI: The Honey Architecture (v1)
- **Evidence tier:** Mechanistic (toy systems) + Exploratory (live API)

### domain-validation__Foundational-and-Origin/
- 13 physics and cross-domain validation scripts
- **Papers:** Foundational (v4) + On the Origin of Scaling Laws (v2)
- **Evidence tier:** Supporting (mathematical validation)

### blind-prediction-test__Paper-IV-d/
- Pre-registered blind prediction test with forensic analysis
- **Paper:** Paper IV.d: The Effect of Blinding (supports metascience finding)
- **Evidence tier:** Canonical

### cauchy-unification__Paper-VII/
- `scripts/arc_20_domain_universal_test.py` - **Primary:** 20-domain blind prediction test (Cauchy form prediction)
- `scripts/arc_complete_test_suite.py` - Cauchy no-go theorem verification
- `scripts/arc_unified_paradigm_test.py` - Cauchy classification (3 regimes)
- `scripts/arc_rigorous_validation.py` - Tier 1 mathematical foundation (Cauchy's equations)
- `scripts/arc_universal_proof.py` - Universal proof from Cauchy + maximum entropy
- `results/` - Pre-computed outputs for all scripts
- **Paper:** Paper VII: The Cauchy Unification (v1)
- **Evidence tier:** Canonical (20-domain blind prediction, p = 2.87e-10, exceeds 5σ)

### analysis-tools__Cross-Programme/
- `analyze_alpha_align_v5.py` - Alpha-align analysis
- `per_scorer_check.py` - Per-scorer validation
- **Papers:** Used across Papers II, IV.a-d, V
- **Evidence tier:** Tooling (not results)

## Evidence Tier Key

| Tier | Meaning | Methodology |
|------|---------|-------------|
| Canonical | Publishable, defensible in peer review | 4-layer blind, multi-scorer, laundered |
| Pilot | Real data, methodological caveats | Nonblind, single-scorer |
| Exploratory | Pattern-finding, not canonical | Nonblind, single-scorer, non-laundered |
| Mechanistic | Demonstrates mechanism in toy systems | Local simulation, no API |
| Supporting | Mathematical/computational validation | Domain-specific tests |
