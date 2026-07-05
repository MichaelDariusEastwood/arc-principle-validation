# F23 refined — metabolic-error numbers: "2.4% vs 2.5%" is NOT a contradiction; table-mean recompute → T5

**Lane:** T2-research · 2026-07-05 · Ladder #2 (checkable-number). Honest partial: framing corrected, one verification handed to T5 (clean recompute).

## What I resolved (LAW 1 — corrects my own earlier flag)
Earlier session note framed "stated 2.5%/2.4% vs table-actual 1.21%/0.10%/0.70%" as a contradiction. On inspection that is **imprecise**:
- **Paper III "2.4% mean error"** is explicitly *"across 11 species groups"* — a BIOLOGICAL/metabolic-only mean.
- **On-the-Origin "2.5%"** is explicitly *"Mean absolute error across all predictions"* — a BROADER set that its own table shows includes COSMOLOGICAL rows (e.g. "Universe, radiation era, d=1.5, 0.0%").
These are **different quantities over different prediction sets**. 2.4% ≠ 2.5% is therefore expected, not a contradiction. The "2.4-vs-2.5 contradiction" framing is RETIRED.

## What I could NOT verify from here (honest null)
Whether the full per-species table's mean is exactly 2.4% (Paper III) / 2.5% (On-the-Origin). Reason: HTML-stripped regex captured only a partial error column (6 values → 1.167% — an INCOMPLETE extraction, not the true table mean). The values 1.21/0.10/0.70 do not appear cleanly in the paper HTML (only in the PDF binary + unrelated alignment JSON). A trustworthy answer needs the table parsed from source (PDF/underlying data), not from stripped HTML.

## Route to T5 (precise question)
1. Parse Paper III's full 11-species metabolic error table from source; confirm mean = 2.4% (or report the true mean).
2. Parse On-the-Origin's full all-predictions table; confirm mean absolute error = 2.5%.
3. If either prose figure ≠ its own table mean → genuine contradiction, route back to T2/operator to correct.
4. Reconcile: state clearly in both papers WHICH set each mean covers (11 species vs all predictions) so a reviewer doesn't read them as the same number.

## Net
Downgraded from "contradiction" to "needs a clean recompute + a one-line set-scope clarification." No false alarm left standing; no unverified claim made.
