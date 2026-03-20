#!/bin/bash
# ============================================================
# PAPER PDF EXPORT SCRIPT - Gold Standard
# ============================================================
# This script exports all papers to properly formatted PDFs.
# No headers, no footers, no URLs, no dates, no orphaned titles.
# Run this ONCE after any paper update. It handles everything.
#
# Usage: ./scripts/export-pdfs.sh
# Or:    ./scripts/export-pdfs.sh Paper-VIII-The-Load-Bearing-Proof
#
# See VERSION-CONTROL-STANDARDS.md Section 11.8 for full documentation.
# ============================================================

set -e

REPO_ROOT="/Users/michaeleastwood/arc-principle-validation"
PAPERS_DIR="$REPO_ROOT/papers"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if [ ! -f "$CHROME" ]; then
    echo "ERROR: Google Chrome not found at $CHROME"
    exit 1
fi

if [ ! -d "$PAPERS_DIR" ]; then
    echo "ERROR: Papers directory not found at $PAPERS_DIR"
    exit 1
fi

# If a specific paper name is given, only export that one
if [ -n "$1" ]; then
    PAPER_DIRS=("$PAPERS_DIR/$1")
    if [ ! -d "${PAPER_DIRS[0]}" ]; then
        echo "ERROR: Paper directory $1 not found in $PAPERS_DIR"
        exit 1
    fi
else
    # Find all paper directories that contain an HTML file
    PAPER_DIRS=()
    for dir in "$PAPERS_DIR"/*/; do
        dirname=$(basename "$dir")
        if [ -f "$dir/$dirname.html" ]; then
            PAPER_DIRS+=("$dir")
        fi
    done
fi

echo ""
echo "============================================================"
echo "  PAPER PDF EXPORT"
echo "============================================================"
echo "  Papers directory: $PAPERS_DIR"
echo "  Papers to export: ${#PAPER_DIRS[@]}"
echo "  Chrome: $(basename "$CHROME")"
echo "============================================================"
echo ""

for paper_dir in "${PAPER_DIRS[@]}"; do
    dirname=$(basename "$paper_dir")
    html_file="$paper_dir/$dirname.html"
    pdf_file="$paper_dir/$dirname.pdf"

    if [ ! -f "$html_file" ]; then
        echo "  SKIPPED: $dirname (no $dirname.html found)"
        continue
    fi

    echo "  Exporting: $dirname.html -> $dirname.pdf"

    "$CHROME" \
        --headless=new \
        --disable-gpu \
        --no-sandbox \
        --run-all-compositor-stages-before-draw \
        --virtual-time-budget=10000 \
        --print-to-pdf="$pdf_file" \
        --print-to-pdf-no-header \
        "file://$html_file" 2>/dev/null

    if [ -f "$pdf_file" ]; then
        size=$(du -h "$pdf_file" | cut -f1)
        echo "    Done ($size)"
    else
        echo "    FAILED"
    fi
done

echo ""
echo "============================================================"
echo "  EXPORT COMPLETE"
echo "  PDFs ready for GitHub and OSF upload."
echo "  HTMLs ready for website upload."
echo "============================================================"
echo ""
