# Runner battery for the reconciled instrument tree, 6 September 2026

What this is. The operating characteristics of the analysis code, world by world, produced by the
code in `instruments/arc-instruments` as it stood when the battery was run. Each row runs the scoring code many
times in one simulated world whose answer is known in advance. It is evidence about the code and
never about any real system.

How it was produced. From `instruments/arc-instruments`:

    python3 runner_battery.py --out <this directory> --seeds 2

Two seeds per row, and the reason is stated rather than assumed: at two repetitions a rate of 1.00
means no failure in two runs, which is a check that the code runs and reports, not a measurement of
its error rate. The interval columns carry that uncertainty honestly, and every rate here should be
read through them. The earlier run at forty seeds per row measured the rates; this one exists so
that the committed evidence describes the code that is actually in the tree.

Interpreter: Python 3.9.6, numpy 2.0.2, scipy 1.13.1, pytest 8.4.2. The suite passed 515 tests with
none failing at the moment these files were written. `MANIFEST.json` carries the sha256 of both
output files and of every source file under `instruments/arc-instruments`, so a reader can tell whether
the tree has moved since.

Regenerated once, after a review found that the deciding path asked for custody of a record and never
asked what the record was about. Nine source files changed and the readings did not: every row of the
table below is byte for byte what the previous run produced, and the two output files differ from it
only in the timestamp they were generated at and the seconds they took. That is the expected result
and it is worth stating, because the change adds a refusal on a path this battery never takes: the
battery scores simulated worlds in demonstration mode, and what was added refuses a simulated system
on the deciding path. A reading that had moved would have meant the change reached further than it
was meant to.

What it supersedes, and what it does not. An earlier directory, dated 5 September 2026, holds a
forty-seed run of the code as it stood before the repairs recorded here landed. It is not deleted
and it is not wrong: it is a record of a different tree, and its own note names the commit it was
taken from. Where a figure differs between the two, this directory describes the current code and
that one describes the code it was run against. The two are not comparable row by row in any case:
the world list is not the same length, and several readings moved for stated reasons, chiefly that
an averaged ladder reading is no longer rounded back to a whole count.
