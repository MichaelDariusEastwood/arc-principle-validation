# Security notes - Paper X

Only one component here has a security surface: the **real-model harness**
(`experiments/scripts/realmodel_coscaling.py`) executes **model-generated code** to
score capability. The deterministic verification harness (`code/experiment_coscaling.py`),
the tests, and the estimator do **not** execute untrusted code and have no network surface.

## Threat model

The real-model harness asks a frontier model to write a `evaluate(expr)` parser and then
**runs that code** against hidden tests to measure capability objectively. Model-written
code is untrusted: it may attempt to read the test harness, touch the filesystem, open
the network, spin CPU, or exhaust memory.

## Mitigations actually implemented (read the code, not just this file)

- **Isolated subprocess.** Candidate code runs via `python3 -I` (isolated mode:
  no user site-packages, no `PYTHON*` env inheritance), never in the harness process.
  See `_run_sandboxed()` and `_run_hidden_tests()`.
- **Resource limits.** `RLIMIT_CPU` (12 s) and `RLIMIT_AS` (1 GiB address space) are set
  in the child preexec, plus a wall-clock `timeout` and an in-runner `signal.alarm`.
- **Minimal environment + temp cwd.** The child runs in a stripped environment and a
  throwaway working directory.
- **No `eval` of model output.** Where the harness parses a candidate's *output* it uses
  `ast.literal_eval` (literals only), never `eval`/`exec` on model-produced strings.
- **Static-integrity gate.** Candidate source is statically scanned (`static_violation()`);
  using `eval/exec/compile/ast/importlib/...` zeroes the capability score
  (`C_compliant = C_raw · 1[no static violation]`) - integrity does not depend on the
  blind evaluator alone.
- **Evidence laundering.** The blind evaluator sees code passed through an AST round-trip
  (`_launder()`) that strips comments/docstrings, so scoring is on behaviour, not tells.

## Operator responsibilities (the limits the code cannot enforce)

`rlimit` + isolated mode do **not** sandbox the network or the kernel. The in-code header
says it explicitly: **run the real-model harness only inside a disposable, network-isolated
environment** - a container (Docker/Podman), gVisor, a microVM, or a throwaway VM - with
no network and nothing sensitive mounted. Treat each run as if it will execute hostile code.

- **API keys.** The harness reads provider keys from the environment
  (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, ...). No keys are stored in the repository. Use
  least-privilege, budget-capped keys: the harness makes real, billable model calls, so
  rate/secret hygiene and a spend cap are the operator's responsibility.
- **The published `Dockerfile` deliberately runs only the offline checks**, never the
  real-model harness.

## Reporting

This is a research artefact, not a deployed service. If you find a sandbox-escape path in
the harness, please open an issue on the repository describing it.
