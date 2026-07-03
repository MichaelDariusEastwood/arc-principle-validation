#!/usr/bin/env python3
"""A4 PDF export for the ARC/Eden corpus.

Renders paper HTML (with MathJax) and markdown deliverables to
print-perfect A4 PDFs using Playwright's headless renderer.
WebKit is installed for cross-engine render checks; PDF printing uses the
Chromium headless shell (the only Playwright engine exposing page.pdf()).

Usage:
  python3 tools/paper_pdf_export.py <file-or-dir> [...]   # export targets
  python3 tools/paper_pdf_export.py --all                 # every paper HTML
"""
import sys, subprocess, html as H
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "--user", "playwright"], check=True)
    from playwright.sync_api import sync_playwright

PAPERS = Path(__file__).resolve().parent.parent / "papers"

# Centralised print furniture — every PDF in the corpus gets identical page
# numbering ("Page X of Y") from this single place, so it can never drift or be
# forgotten per-paper. Chromium fills the pageNumber/totalPages spans natively.
# The templates do NOT inherit page CSS, so the font is set inline.
FOOTER_TEMPLATE = (
    '<div style="width:100%; text-align:center;'
    " font-family:Georgia,'Times New Roman',serif; font-size:8pt; color:#666;\">"
    'Page <span class="pageNumber"></span> of <span class="totalPages"></span>'
    "</div>"
)
# Near-empty header (a bare template would make Chromium print its default date/url).
HEADER_TEMPLATE = '<div style="font-size:1px; height:0;"></div>'

MD_TEMPLATE = """<!DOCTYPE html><html lang="en-GB"><head><meta charset="utf-8">
<title>{title}</title><style>
@page {{ size: A4; margin: 18mm 16mm; }}
body {{ font-family: Georgia, 'Times New Roman', serif; font-size: 10.5pt;
       line-height: 1.55; color: #1a1a1a; max-width: 100%; }}
h1 {{ font-size: 17pt; line-height: 1.2; border-bottom: 2.5px solid #1a1a1a;
     padding-bottom: 4pt; }}
h2 {{ font-size: 13pt; margin-top: 14pt; border-bottom: 1px solid #999;
     padding-bottom: 2pt; page-break-after: avoid; }}
h3 {{ font-size: 11.5pt; page-break-after: avoid; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.5pt;
        page-break-inside: avoid; }}
th {{ background: #1a1a1a; color: #fff; padding: 3pt 5pt; text-align: left; }}
td {{ border: 0.5pt solid #bbb; padding: 3pt 5pt; vertical-align: top; }}
blockquote {{ border-left: 3pt solid #2e7d32; margin: 8pt 0; padding: 4pt 10pt;
             background: #f4f8f4; }}
code {{ font-family: Menlo, monospace; font-size: 8.5pt; background: #f1f1f4;
       padding: 0 2pt; }}
pre {{ background: #f6f6f8; border: 0.5pt solid #ccc; padding: 6pt;
      font-size: 8pt; white-space: pre-wrap; page-break-inside: avoid; }}
ul, ol {{ padding-left: 16pt; }}
li {{ margin: 2pt 0; }}
a {{ color: #1565c0; text-decoration: none; }}
</style></head><body>{body}</body></html>"""


def md_to_html(md_path: Path) -> str:
    try:
        import markdown  # type: ignore
        body = markdown.markdown(md_path.read_text(encoding="utf-8"),
                                 extensions=["tables", "fenced_code"])
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "markdown"], check=True)
        import markdown  # type: ignore
        body = markdown.markdown(md_path.read_text(encoding="utf-8"),
                                 extensions=["tables", "fenced_code"])
    return MD_TEMPLATE.format(title=H.escape(md_path.stem), body=body)


def export(targets):
    results = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        for src in targets:
            src = Path(src).resolve()
            out = src.with_suffix(".pdf")
            if src.suffix == ".md":
                tmp = src.with_suffix(".tmp-print.html")
                tmp.write_text(md_to_html(src), encoding="utf-8")
                load = tmp
            else:
                load = src
            page.goto(f"file://{load}", wait_until="networkidle", timeout=120000)
            # Let MathJax finish typesetting when present.
            page.evaluate(
                """() => (window.MathJax && MathJax.startup
                          ? MathJax.startup.promise : Promise.resolve())"""
            )
            page.wait_for_timeout(400)
            page.pdf(path=str(out), format="A4",
                     margin={"top": "16mm", "bottom": "18mm",
                             "left": "14mm", "right": "14mm"},
                     print_background=True, prefer_css_page_size=False,
                     display_header_footer=True,
                     header_template=HEADER_TEMPLATE,
                     footer_template=FOOTER_TEMPLATE)
            if src.suffix == ".md":
                load.unlink(missing_ok=True)
            kb = out.stat().st_size // 1024
            results.append((out, kb))
            print(f"  {out.name}: {kb} KB")
        browser.close()
    return results


def main():
    args = sys.argv[1:]
    if not args or args == ["--all"]:
        targets = sorted(str(p) for p in PAPERS.glob("*/*.html")
                         if "tmp-print" not in p.name)
    else:
        targets = []
        for a in args:
            p = Path(a)
            if p.is_dir():
                targets += sorted(str(x) for x in p.rglob("*.html")
                                  if "tmp-print" not in x.name)
                targets += sorted(str(x) for x in p.rglob("*.md"))
            else:
                targets.append(a)
    print(f"Exporting {len(targets)} document(s) to A4 PDF...")
    export(targets)
    print("DONE")


if __name__ == "__main__":
    main()
