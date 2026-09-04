# Early ARC scaling scripts, superseded

Four standalone scripts written before the experiments were organised per paper. They are
kept because they are the record of how the first scaling measurements were made, and
because deleting the earlier attempt and keeping only the one that worked is the thing a
reader should be able to check against.

**Nothing here is current.** No result in any paper rests on these scripts. The measurements
that papers cite live in the per-paper folders listed in `../README.md`.

## The lineage, oldest first

| File | Lines | What it was |
|------|-------|-------------|
| `arc_live_experiment.py` | 275 | the first live run, against the deployed ask-book API, testing sequential against parallel reasoning |
| `arc_validation_experiment.py` | 723 | the same hypothesis rebuilt as a standalone validation, testing whether sequential reasoning produces a scaling exponent above one |
| `arc_improved_experiment.py` | 648 | a revision of the live run, whose own docstring lists what it changed |
| `arc_rigorous_experiment.py` | 760 | the last of the four, rebuilt against a stated external methodology |

The adjectives in three of these filenames are the authors' own from the time. They are
not a ranking, they carry no evidential weight, and the file called rigorous is not
thereby more reliable than the others. The names are left as they were written rather
than tidied, because renaming a file to something more flattering than its history is the
opposite of what this repository is for.

## Why they are not in a paper folder

Each of them predates the split into per-paper experiments, and none maps cleanly onto a
single document. Filing them under one paper would assert a lineage that does not exist.
