# Canonical v5 Alignment Table

Source directory: `/Users/michaeleastwood/Arc & Eden Test Results/alignment_results_v5/v5_final_results`

- All rows are recomputed from final JSONs only.
- Only `task_type == "alignment"` rows are included.
- The final JSONs contain valid alignment consensus scores in `consensus_weighted_mean`.
- Scorer pools are non-uniform across subject runs (6-7 scorers), so table notes keep that caveat visible.

| Model | File | Alignment n | Depths | Scorers | Consensus field | α_align (grouped) | Spearman ρ | Cohen's d | Run-quality note |
| --- | --- | ---: | --- | ---: | --- | --- | --- | --- | --- |
| Claude Opus | `v5_final_claude-opus_20260312_112739.json` | 140 | `minimal → standard → deep → exhaustive → extreme` | 6 | `consensus_weighted_mean` | +0.038 ± 0.015 (p=0.083) | +0.299 (p=0.000) | +1.353 | Valid final blind run; positive but non-significant; suspicious 0/140; 6 scorers. |
| DeepSeek R1 | `v5_final_deepseek-r1_20260311_211855.json` | 168 | `minimal → standard → thorough → exhaustive → extreme → maximum` | 7 | `consensus_weighted_mean` | -0.022 ± 0.032 (p=0.533) | -0.135 (p=0.081) | -0.163 | Valid final blind run; near-flat blind signal; suspicious 5/168; 7 scorers. |
| GPT-5.4 | `v5_final_openai-gpt54_20260311_191836.json` | 140 | `minimal → low → standard → deep → exhaustive` | 7 | `consensus_weighted_mean` | -0.009 ± 0.013 (p=0.531) | +0.002 (p=0.985) | +0.068 | Valid final blind run; near-flat blind signal; suspicious 5/140; 7 scorers. |
| Gemini Flash | `v5_final_gemini-flash_20260311_151244.json` | 140 | `minimal → standard → deep → exhaustive → extreme` | 7 | `consensus_weighted_mean` | -0.309 ± 0.134 (p=0.105) | -0.246 (p=0.003) | -0.612 | Valid final blind run; negative but non-significant; suspicious 5/140; 7 scorers. |
| Grok 4 Fast | `v5_final_grok-4-fast_20260311_200910.json` | 140 | `minimal → standard → deep → exhaustive → extreme` | 7 | `consensus_weighted_mean` | +0.401 ± 0.190 (p=0.126) | +0.175 (p=0.039) | +1.593 | Valid final blind run; positive but non-significant; suspicious 3/140; 7 scorers. |
| Groq Qwen3 | `v5_final_groq-qwen3_20260312_073302.json` | 140 | `minimal → standard → deep → exhaustive → extreme` | 6 | `consensus_weighted_mean` | +0.118 ± 0.016 (p=0.005) | +0.139 (p=0.102) | +1.016 | Valid final blind run; positive blind signal; suspicious 2/140; 6 scorers. |
