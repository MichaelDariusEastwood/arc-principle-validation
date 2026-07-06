#!/usr/bin/env bash
# gate_papers.sh — research paper release gate.
# Runs G-RESEARCH-CITATION-HYPERLINK (no source without a supportive link) across
# every paper. Run before committing any paper edit and before any merge to main.
#   bash tools/gate_papers.sh            # all papers
#   bash tools/gate_papers.sh papers/HRIH-Paper
# Exit 0 = all green; exit 1 = at least one paper blocked.
set -uo pipefail
cd "$(dirname "$0")/.."
DIR="${1:-papers}"
python3 tools/research_citation_hyperlink_gate.py --dir "$DIR"
