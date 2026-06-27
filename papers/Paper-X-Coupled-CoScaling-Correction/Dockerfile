# Paper X - The Coupled Co-Scaling Law: reproducible verification environment.
#
# This image runs ONLY the deterministic, offline checks: the verification harness,
# both pytest suites, and the synthetic beta/k estimator. It does NOT run the
# real-model harness (realmodel_coscaling.py), which makes live API calls and must
# be run in a disposable, network-isolated sandbox with your own keys (see SECURITY.md).
#
#   docker build -t paperx .
#   docker run --rm paperx              # -> verify + tests + estimator
#
FROM python:3.11-slim

WORKDIR /paperx

# Dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Source, tests, scripts.
COPY code/ ./code/
COPY experiments/ ./experiments/
COPY Makefile .

# Default: prove code-matches-maths and run both test suites + the synthetic estimator.
# (No network; no real-model calls.)
CMD ["sh", "-c", "cd code && python experiment_coscaling.py && python -m pytest -q && cd ../experiments/scripts && python estimate_exponents.py --selftest"]
