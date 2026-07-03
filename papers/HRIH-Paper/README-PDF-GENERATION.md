# HRIH Paper — PDF Generation

The HTML paper is complete at `Hyperspace-Recursive-Intelligence-Hypothesis.html`.

PDF generation requires Chrome headless (same as all other papers — see 
export-pdfs.sh in the repo root). MathJax equations require a browser
engine to render properly.

To generate the PDF:
```bash
cd /Users/michaeleastwood/arc-principle-validation
# Same method used for all 20 other papers:
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=papers/HRIH-Paper/Hyperspace-Recursive-Intelligence-Hypothesis.pdf \
  papers/HRIH-Paper/Hyperspace-Recursive-Intelligence-Hypothesis.html
```

Or run: `bash scripts/export-pdfs.sh` (if it exists in the repo).

After generation, the paper will be the 21st paper with HTML+PDF.
