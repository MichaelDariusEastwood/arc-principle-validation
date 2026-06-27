# Paper X - The Coupled Co-Scaling Law: reproducibility targets.
# Usage: make <target>.  `make all` runs install + verify + test.
#
# What each target proves (read honestly):
#   verify            -> the numerical harness matches the closed-form theorems
#                        (code-vs-maths; NOT a test against real AI)
#   test              -> the in-tree regression suite passes
#   test-independent  -> a from-scratch suite (no harness import) confirms the
#                        corrected theorems, incl. the Theorem-4 equality case and
#                        non-normal transient growth
#   estimator         -> the beta/k estimator recovers known exponents on synthetic data
# None of these test the model against a real self-improving system; that is the
# open empirical problem (PROTOCOL.md, paper section 8).

PY ?= python3
CODE := code
SCRIPTS := experiments/scripts

.PHONY: all install verify test test-independent test-all estimator selftest manifest clean

all: install verify test-all

install:
	$(PY) -m pip install -r requirements.txt

verify:
	cd $(CODE) && $(PY) experiment_coscaling.py

test:
	cd $(CODE) && $(PY) -m pytest test_coscaling.py -q

test-independent:
	cd $(CODE) && $(PY) -m pytest test_theorems_independent.py -q

test-all:
	cd $(CODE) && $(PY) -m pytest -q

estimator:
	cd $(SCRIPTS) && $(PY) estimate_exponents.py --selftest

selftest:
	cd $(SCRIPTS) && $(PY) realmodel_coscaling.py --selftest --rounds 4 --seeds 2

manifest:
	$(PY) tools/make_manifest.py

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
