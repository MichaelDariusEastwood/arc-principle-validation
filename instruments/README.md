# Instruments

Reference code for confirmatory runs that have not been made, and the recorded operating
characteristics of that code. **Nothing in this folder is a result about any real system.**
The `experiments/` tree holds the runs that produced the programme's results; this folder holds
the instruments those results' successors would be decided by.

```
instruments/
├── README.md                              this file
├── arc-instruments/                       the kit: decision rules, design simulators, the runner
└── evidence/
    └── 2026-09-06-runner-battery-a1-a9/   what the kit's analysis code does, world by world
```

## What the kit is

`arc-instruments/` is executable decision rules for the twenty-two registered propositions of the
theory-level preregistration, design-sensitivity simulators for the deciding experiments (form
discrimination, ceiling-chain precision, coupling identification, burden identification, corrector
dependence, blinding integrity, and the P5 and P16 designs), sealed-prediction and held-out
partition tooling, and a runner that carries the P5 and P16 charters as code.

Pure Python 3.9 with numpy and scipy. The kit makes no network call of its own; a provider adapter
is something a caller supplies.

**The registration governs.** Where this code and the registration disagree, the tests are the place
the disagreement becomes visible, and one of the two is corrected. A number in a Python file is a
number a program needs in order to run, and never the author's registered choice.

This is reference code and a design instrument. It is not a validated deciding instrument and it is
not empirical evidence.

## Running the suite

From `instruments/arc-instruments`:

```bash
python3 -m pip install -r requirements-test.txt
python3 -m pytest -q
```

**515 tests pass with none failing**, in five to nine minutes, on Python 3.9.6 with numpy 2.0.2,
scipy 1.13.1 and pytest 8.4.2. Those pins name what was actually run rather than what is newest: a
pin that has not been run is a claim.

## The four execution modes

The kind of run is said once, explicitly, and the gate fails closed. Before these modes existed, the
only thing separating a rehearsal from a deciding run was a pair of flags and a line of text a
reader might not have read.

| Mode | Runs against | What it may claim |
|---|---|---|
| `DEMONSTRATION` | the simulated system | a recovery of a test case whose answer the test already knows: evidence about the pipeline, never about any real system |
| `SMOKE` | a real provider, with the reference pool that was not written for the study | that the wiring works, and nothing else |
| `PILOT` | a real provider | sizing for the instrument. It is never scored |
| `CONFIRMATORY` | a real provider, under the registered design | the deciding path, and the only mode that may be scored at proposition level |

`CONFIRMATORY` refuses to begin unless every input the registration names is present **before the
first provider call**: a domain ladder that is not the reference smoke pool and that declares which
population its readings sample; a checkpoint store holding every state the bank will place; a
resolved configuration carrying every registered quantity its verdict will be read with; a spending
controller that reserves before each dispatch; and an anchoring service that will attest the seal. A
missing input stops the run with a named refusal listing every requirement that failed rather than
only the first, so that a setup is fixed once.

Where a registered number is unset, the deciding path refuses rather than defaulting in its own
favour, and it refuses a value labelled a candidate as well, because a value present is not a value
registered. The gate never supplies one: a number the runner filled in would be the runner's choice
wearing the registration's name.

A refusal is not a verdict, and it is not a reason to run in a weaker mode instead. **A confirmatory
run that cannot start has produced no evidence at all.**

## The deciding path refuses without a released apparatus

A simulated system may demonstrate the whole apparatus and may not decide a proposition. The
question is asked twice, and the two catch different runs.

- `arc_runner.mode` asks before the first paid call, where a refusal costs nothing, and names every
  system the run holds including the held-out panel's.
- `arc_runner.manifest.require_released_apparatus` asks again at proposition level, where a bundle
  re-scored from disk long afterwards is read.

The declaration is the caller's own statement about the caller's own apparatus, in the way the
anchoring service and the prior-inspection attestation are. It is written into the confirmatory
inputs before the seal, so it sits inside the sealed specification hash: a record edited afterwards
to say the simulator was a real system fails custody rather than passing the release check.

The one anchor shipped here is a mock, labelled a mock in its identifier, its label and a boolean
field, and refused at proposition level. A mock receipt is honest in a demonstration and a forgery
in a deciding run.

## The battery evidence

`evidence/2026-09-06-runner-battery-a1-a9/` holds the analysis code's operating characteristics,
world by world, produced from the kit directory by:

```bash
python3 runner_battery.py --out <that directory> --seeds 2
```

Each row runs the scoring code many times inside one simulated world whose answer is known in
advance. Two seeds per row is a check that the code runs and reports, not a measurement of a rate,
and the interval columns carry that uncertainty rather than hiding it. Read every rate through them.

`MANIFEST.json` pins the sha256 of both output files and of the eighty Python source files as the
battery was run, and `run_from` names the directory it was run from at that time.

**Nine of those eighty hashes differ here, and seventy-one agree.** Publication removed internal
cross-references from nine files: `arc_instruments/costing.py`, `arc_instruments/parity.py`,
`arc_instruments/regions.py`, `arc_runner/code_domain.py`, `arc_runner/ladder.py`,
`arc_runner/p16.py`, `arc_runner/p16_contract.py`, `arc_runner/p5.py` and
`arc_runner/p5_observation.py`. In seven of them the rewritten text is comment and docstring only.
In `arc_instruments/costing.py` it is one human-facing string literal, the pilot row label in the
table `flagship_menu()` returns. In `arc_runner/p16_contract.py` it is the module docstring and one
entry of the `limitations` list the adjudication returns. Those two functions therefore return one
phrase of prose that differs from the manifest run. No logic, control flow, numeric constant or data
value changed in any of the nine, no test asserts either string, and the suite passes the same 515
tests in this home. The manifest's two pinned output files match byte for byte, so the battery
result is the one the manifest records. The manifest is left exactly as the run produced it, because
an evidence record that is edited after the fact is no longer a record.

## The decisions this code did not make

A register of the working choices the code made and did not register is held privately with the
registration masters and is not published here. It records, for each question, what the question is,
what the code does today with a file and a symbol so it can be found, and what changes if the author
rules otherwise. It carries one hundred and fifteen conservative readings, deduplicated by substance
across four rounds, and four items that need an author ruling before any deciding run. It contains
no recommendation, and where two readings are both defensible it says so rather than preferring one.

Every number in this kit sits under that register until the author rules. Nothing in the code is
registered by being in the code.
