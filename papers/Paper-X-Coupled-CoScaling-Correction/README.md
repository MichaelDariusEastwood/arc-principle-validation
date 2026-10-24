# Paper X — Correction That Out-Scales Drift

Michael Darius Eastwood · The ARC Theory · Law II, the ARC Co-Scaling Law

## Current mathematical correction — 19 October 2026

[**Corrected Theorems 1–7 and their proofs**](THEOREMS.md) are the mathematical
specification for the revised independent tests. This is a targeted correction
of the theorem hypotheses and their extensions, not a new empirical result or
a revision of priority dates. It preserves the positive-gain-drift result:

$$\gamma_1>0,\quad\gamma_2=\gamma_3=0
\quad\Longrightarrow\quad d\to0\ \text{iff}\ \beta>k.$$

The hard-takeoff extension permits non-negative level drift when k>0 and
retains the necessity condition only with positive gain drift. Zero gain drift,
constant versus time-varying compounding, left-null matrix projection, stochastic
marginal versus pathwise risk, and finite-capacity trajectories are now stated
separately. The proof of each corrected result is included in `THEOREMS.md`.

The manuscript mirrors (PDF, HTML, text and citation files) are now version 1.16
of 21 October 2026, the edition typeset in LaTeX and the manuscript release that
carries these corrections: its Appendix B.1 lists each change, and its Appendix A
proves each theorem as now stated. The mirror at the baseline of this review was
v1.14; the PDF supplied for the review was v1.15, which was held before deposit
and is superseded by v1.16. The experiment outputs remain unchanged.
The old README is preserved byte-for-byte as
[`archive/README-before-2026-10-19.md`](archive/README-before-2026-10-19.md)
for provenance, not as a statement of current results.

## Run the independent checks

```bash
git clone https://github.com/MichaelDariusEastwood/arc-principle-validation.git
cd arc-principle-validation
git switch fix/paper-x-theorems-20261019
cd papers/Paper-X-Coupled-CoScaling-Correction
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q code/test_theorems_independent.py
```

The repair branch starts from
`040ad474fa0dcea14f3ec53521906acabfc2c670`. Checking out that original commit
will intentionally reproduce the old files, not this correction. Pin the repair
commit for archival reproduction. After merge, the tests are also available on
`main`; do not infer a merge from this README.

The suite independently implements the equations and does not import
`code/experiment_coscaling.py`. It checks closed forms, the original (C,D)
equations against the depth clock, full-matrix left/right-null counterexamples,
non-normal transients, changing-coefficient counterexamples, OU moments,
finite-capacity trajectories, and invalid solver outputs. Finite numerical
checks do not prove asymptotic theorems or validate real AI systems.

The existing `.github/workflows/five-minute-checks.yml` is enabled and runs this
command on Python 3.11 and 3.13. It uses pytest's exit status rather than matching
a hard-coded pass count. Local results and their scope are recorded in
[`verification/2026-10-19.md`](verification/2026-10-19.md); CI results are a
separate record and must be checked on the pull request.

## Evidence and scope

The historical ODE harness is internal verification, not a real-system test.
The archived model pilots do not establish the co-scaling exponents. This
mathematical correction neither reruns nor rehabilitates those pilots and does
not change their withdrawn or provisional evidentiary status. Theorems concern
a modelled normalised index, not a demonstrated guarantee of AI safety.

The paper's deposit is [OSF BSE2Q](https://doi.org/10.17605/OSF.IO/BSE2Q).
The original publication record, research framing and historical file inventory
remain available in the manuscript mirrors and the archived README; statements
there that conflict with `THEOREMS.md` are superseded for this test specification.

## LaTeX source

[`Paper-X-Coupled-CoScaling-Correction-latex-source.tar.gz`](Paper-X-Coupled-CoScaling-Correction-latex-source.tar.gz): the LaTeX source package of version 1.16 (21 October 2026), the source files, bibliography and figures of the PDF of record; sha256 `2206d6c1bdf861ec9b7e8509614c7c2c27c2ada432d1445ed32b236216b0225d`.
