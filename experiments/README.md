# Experiments

All experimental scripts, results, and data for the ARC/Eden research programme.

## Structure

### paper-i-foundational/
- `scripts/arc_principle_research_toolkit.py` - Original ARC Principle validation toolkit
- Papers: Foundational v3-v4

### paper-ii-compute/
- `scripts/arc_paper_ii_validation_v1_deepseek.py` - Original single-model DeepSeek validation
- `scripts/arc_paper_ii_validation_v2.py` - Multi-model validation (DeepSeek, Gemini, Grok, Groq, OpenAI)
- `results/` - Final JSON results for 6 models + combined analysis
- Papers: Paper II v12

### alignment-scaling/
- `scripts/arc_alignment_scaling_v1.py` through `v5.py` - Progressive refinement of alignment scaling benchmark
- `scripts/arc_eden_v6_runner.py` - Combined alignment + Eden runner (not yet run)
- `results/v1/` through `v5-final/` - Results from each version
- Papers: Papers IV.a, IV.b, IV.c, IV.d

### eden-intervention/
- `scripts/eden_protocol_scaling_test.py` - v1 Eden intervention test
- `scripts/eden_protocol_scaling_test_v2.py` - v2 with expanded scoring
- `scripts/eden_protocol_scaling_test_v3.py` - v3 with blind multi-scorer evaluation (NOT YET RUN)
- `results/` - 6 model final JSONs (v1/v2 protocol, single-scorer, pilot-grade)
- Papers: Paper V

### honey-architecture/
- `scripts/eden_honey_simulation.py` - Mathematical simulation of honey architecture
- `scripts/eden_honey_tests.py` - Comprehensive API-backed honey test battery
- `scripts/eden_honey_dashboard.jsx` - React dashboard for honey results
- `scripts/eden_self_modifying_ai.py` - v1 self-modifying AI experiment
- `scripts/eden_self_modifying_ai_v2.py` - v2 multi-seed robustness (fair test)
- `scripts/eden_self_modifying_ai_v3.py` - v3 adversarial conflicting tasks
- `scripts/eden_self_modifying_ai_v4.py` - v4 complexity scaling
- `results/` - Fresh JSON + PNG outputs from all simulations (generated 16 March 2026)
- Papers: Paper VI

### domain-validation/
- 13 physics and cross-domain validation scripts testing ARC Principle predictions
- Covers: 1D prediction, 20-domain universal test, acoustic time crystals, Einstein verification, and more

### blind-prediction-test/
- Pre-registered blind prediction test with forensic analysis

### analysis-tools/
- `analyze_alpha_align_v5.py` - Alpha-align analysis tool
- `per_scorer_check.py` - Per-scorer validation

## Key Notes

- The v5-final alignment results are the canonical dataset (6 models, 4-layer blind evaluation)
- Eden intervention results are pilot-grade (single-scorer, nonblind) until v3 is run
- Honey architecture results are toy-system simulations, not frontier-model evidence
- Paper II has two scripts: v1 (DeepSeek only) and v2 (multi-model)
- The v6 runner and Eden v3 test have not been run yet
