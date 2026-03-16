# Cross-Folder Consistency Audit

## Known Issues
- v5 final raw files are split across /alignment_results_v5 and /alignment_results; Claude Opus final currently lives only in the top-level folder.
- v5 scorer pools are not uniform: subject runs use 6-7 blind scorers depending on subject identity and the available non-subject scorer adapters.
- Eden evidence must be stratified by run quality: five analysable runs show stakeholder-care gains, but the broadest composite uplift remains concentrated in Gemini and Groq; GPT-5.4 failed operationally.
- The running HTML report historically mixed superseded chronicle text with current headline claims; later sections should explicitly supersede earlier ones.

## v5 Scorer Counts By Canonical Subject Run
- `claude-opus`: None
- `deepseek-r1`: None
- `gemini-flash`: None
- `grok-4-fast`: None
- `groq-qwen3`: None
- `openai-gpt54`: None

## Eden Status Grid
- `claude`: exploratory_partial
- `deepseek`: pilot_interpretable_nonblind
- `gemini`: pilot_interpretable_nonblind
- `gpt`: operational_failure
- `grok`: exploratory_mixed_quality
- `groq`: pilot_interpretable_nonblind

## Raw File Domains
- `alignment_v5_top_level`: 0 files
- `alignment_v5_versioned`: 0 files
- `eden_results`: 6 files
- `paper_ii_results`: 6 files

