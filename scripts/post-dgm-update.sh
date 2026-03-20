#!/bin/bash
# ============================================================
# POST-DGM UPDATE SCRIPT
# Run this after the DGM v3 experiment completes.
# It updates Paper VIII, regenerates PDFs, and prepares
# everything for GitHub push and grant submission.
# ============================================================

set -e

REPO="/Users/michaeleastwood/arc-principle-validation"
RESULTS="$REPO/papers/Paper-VIII-The-Load-Bearing-Proof/results/dgm_v3_calibrated_results.json"

echo "============================================================"
echo "  POST-DGM UPDATE"
echo "============================================================"

# Check if results file exists
if [ ! -f "$RESULTS" ]; then
    echo "ERROR: DGM results not found at $RESULTS"
    echo "The experiment may still be running."
    exit 1
fi

echo "  DGM results found. Size: $(du -h "$RESULTS" | cut -f1)"
echo ""
echo "  Next steps (run manually or via Claude Code):"
echo "  1. Update Paper VIII with DGM v3 results"
echo "  2. Regenerate Paper VIII PDF"
echo "  3. Update all grant application narratives with DGM result"
echo "  4. Regenerate grant application HTMLs"
echo "  5. Push to GitHub"
echo "  6. Deploy website"
echo "  7. Upload PDFs to OSF"
echo ""
echo "  Run: claude 'The DGM v3 results are at $RESULTS. Update Paper VIII,"
echo "  regenerate PDFs, update all grant narratives, and prepare for push.'"
echo "============================================================"
