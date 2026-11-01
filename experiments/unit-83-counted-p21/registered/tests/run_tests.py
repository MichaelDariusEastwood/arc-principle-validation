#!/usr/bin/env python3
"""The runner of the authored tests, standard library only (41-D-COUNTED, section 11.3, step 2).

The author's machine has the CPython 3.13 already installed there and no third-party package (section 3.5), so the
tests are plain functions named test_* with bare assertions, and this runner finds and runs them. It calls the
registration gate first: nothing of the study is imported or run before both registrations read REGISTERED and the
attestation exists (SD-001, SD-012). Step 2 of section 11.3 is this command, printed by the bench:

  cd <stage> && P21LANE_WORKSPACE=<workspace> python3 tests/run_tests.py [name ...]

A failed test is a logged defect repair before the runtime freeze (section 11.2, item 7), then the case run again.
"""
import importlib
import sys
import time
import traceback
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
TESTS = STAGE / 'tests'


def main(argv=None):
    sys.path.insert(0, str(STAGE))
    sys.path.insert(0, str(TESTS))
    from p21base import registration_gate as gate
    gate.require_registered()
    names = list(argv if argv is not None else sys.argv[1:])
    modules = sorted(path.stem for path in TESTS.glob('test_*.py'))
    if names:
        modules = [m for m in modules if m in names]
    passed = failed = 0
    started = time.time()
    for name in modules:
        module = importlib.import_module(name)
        for attr in sorted(vars(module)):
            function = vars(module)[attr]
            if not attr.startswith('test_') or not callable(function) or getattr(function, '__module__', None) != name:
                continue
            try:
                function()
            except Exception:
                failed += 1
                print('FAIL %s.%s' % (name, attr))
                traceback.print_exc()
            else:
                passed += 1
                print('pass %s.%s' % (name, attr))
    print('%d passed, %d failed, in %.1f seconds' % (passed, failed, time.time() - started))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
