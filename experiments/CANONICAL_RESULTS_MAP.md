# Canonical Results Map

This file is the source-of-truth index for the current ARC / Eden programme as of 12 March 2026.

## Eden Protocol
| Model | Status | Valid pairs | Invalid pairs | Overall delta | d | p | Canonical file |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| claude | exploratory_partial | 30 | 10 | 0.17 | 0.055 | 0.7645 | `/Users/michaeleastwood/Arc & Eden Test Results/eden_results/eden_final_claude_20260312_130933.json` |
| deepseek | pilot_interpretable_nonblind | 40 | 0 | 2.02 | 0.193 | 0.2304 | `/Users/michaeleastwood/Arc & Eden Test Results/eden_results/eden_final_deepseek_20260312_020928.json` |
| gemini | pilot_interpretable_nonblind | 40 | 0 | 5.33 | 0.528 | 0.0018 | `/Users/michaeleastwood/Arc & Eden Test Results/eden_results/eden_final_gemini_20260312_013901.json` |
| gpt | operational_failure | 0 | 40 | NA | NA | NA | `/Users/michaeleastwood/Arc & Eden Test Results/eden_results/eden_final_gpt_20260312_121158.json` |
| grok | exploratory_mixed_quality | 26 | 14 | -0.04 | -0.004 | 0.9837 | `/Users/michaeleastwood/Arc & Eden Test Results/eden_results/eden_final_grok_20260312_124959.json` |
| groq | pilot_interpretable_nonblind | 40 | 0 | 4.92 | 0.545 | 0.0014 | `/Users/michaeleastwood/Arc & Eden Test Results/eden_results/eden_final_groq_20260312_123528.json` |

## Alignment v5
| Model | Blind scorers | Alignment rows | Depth means | Canonical file |
| --- | ---: | ---: | --- | --- |
| deepseek-r1 | None | 0 |  | `/Users/michaeleastwood/Arc & Eden Test Results/alignment_results_v5/v5_final_deepseek-r1_20260311_211855.json` |
| gemini-flash | None | 0 |  | `/Users/michaeleastwood/Arc & Eden Test Results/alignment_results_v5/v5_final_gemini-flash_20260311_151244.json` |
| grok-4-fast | None | 0 |  | `/Users/michaeleastwood/Arc & Eden Test Results/alignment_results_v5/v5_final_grok-4-fast_20260311_200910.json` |
| groq-qwen3 | None | 0 |  | `/Users/michaeleastwood/Arc & Eden Test Results/alignment_results_v5/v5_final_groq-qwen3_20260312_073302.json` |
| openai-gpt54 | None | 0 |  | `/Users/michaeleastwood/Arc & Eden Test Results/alignment_results_v5/v5_final_openai-gpt54_20260311_191836.json` |
| claude-opus | None | 0 |  | `/Users/michaeleastwood/Arc & Eden Test Results/alignment_results/v5_final_claude-opus_20260312_112739.json` |

## Paper II Compute Scaling
| Model | alpha_seq | alpha_parallel | regression alpha | r^2 | Notes | Canonical file |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| deepseek | 3.049 | 0.000 | NA | NA | step_function_or_ceiling | `/Users/michaeleastwood/Arc & Eden Test Results/arc_paper_ii_results/arc_paper_ii_deepseek.json` |
| gemini | 0.590 | 0.315 | 0.493 | 0.861 | continuous_or_floor | `/Users/michaeleastwood/Arc & Eden Test Results/arc_paper_ii_results/arc_paper_ii_gemini.json` |
| grok-4-fast | NA | NA | NA | NA | step_function_or_ceiling | `/Users/michaeleastwood/Arc & Eden Test Results/arc_paper_ii_results/arc_paper_ii_grok-4-fast.json` |
| groq-qwen3 | 0.242 | 0.220 | 0.087 | 0.103 | continuous_or_floor | `/Users/michaeleastwood/Arc & Eden Test Results/arc_paper_ii_results/arc_paper_ii_groq-qwen3.json` |
| openai | 1.470 | -0.030 | 1.599 | 0.947 | step_function_or_ceiling | `/Users/michaeleastwood/Arc & Eden Test Results/arc_paper_ii_results/arc_paper_ii_openai.json` |

## Interpretation Rules

- Treat `pilot_interpretable_nonblind` Eden runs as promising but non-confirmatory until canonical `arc_eden_v6` blind replication lands.
- Treat `exploratory_*` Eden runs as signal-generation only.
- Treat `operational_failure` runs as no evidence.
- Treat the versioned or top-level file listed here as canonical even when other folders contain overlapping copies.

