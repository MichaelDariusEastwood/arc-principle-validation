# Canonical Eden Intervention Table

Source directory: `/Users/michaeleastwood/Arc & Eden Test Results/eden_results`

- Means and deltas exclude `score == -1` operational failures.
- All intervention rows are single-scorer, nonblind pilot data and should be written that way.
- Run-quality labels are inherited from the consistency audit and reinforced by the raw-score failure counts below.

| Model | File | Scorer | Control mean | Eden mean | Delta | Valid n (C/E) | Failures excluded | Run-quality label | Note |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- | --- |
| Claude Opus | `eden_final_claude_20260312_130933.json` | `gemini` | 92.567 | 92.733 | +0.167 | 30/30 | 20 | `exploratory_partial` | Exploratory/partial with negligible delta after excluded failures. |
| DeepSeek R1 | `eden_final_deepseek_20260312_020928.json` | `gemini` | 86.900 | 88.925 | +2.025 | 40/40 | 0 | `pilot_interpretable_nonblind` | Interpretable pilot signal; still nonblind and single-scorer. |
| GPT-5.4 | `eden_final_gpt_20260312_121158.json` | `gemini` | n/a | n/a | n/a | 0/0 | 80 | `operational_failure` | No usable scores. Keep out of inferential claims. |
| Gemini Flash | `eden_final_gemini_20260312_013901.json` | `deepseek` | 77.325 | 82.650 | +5.325 | 40/40 | 0 | `pilot_interpretable_nonblind` | Interpretable pilot signal; still nonblind and single-scorer. |
| Grok 4 Fast | `eden_final_grok_20260312_124959.json` | `gemini` | 87.525 | 88.692 | +1.167 | 40/26 | 14 | `exploratory_mixed_quality` | Mixed-quality pilot; positive delta depends on excluding failed Eden rows. |
| Groq Qwen3 | `eden_final_groq_20260312_123528.json` | `gemini` | 82.350 | 87.275 | +4.925 | 40/40 | 0 | `pilot_interpretable_nonblind` | Interpretable pilot signal; still nonblind and single-scorer. |
