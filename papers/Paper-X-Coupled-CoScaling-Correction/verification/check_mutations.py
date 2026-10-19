"""Targeted fault injection: each deliberate regression must make pytest fail.
Run: python verification/check_mutations.py. Never modifies the real suite.
"""
from pathlib import Path
import json
import subprocess
import sys
import tempfile

root = Path(__file__).resolve().parents[1]
source = (root / 'code/test_theorems_independent.py').read_text()
mutations = [
    ('wrong_dilution_sign', '(q + 1.0 - gamma3) * y[0]', '(q - 1.0 - gamma3) * y[0]',
     'test_theorem2_three_regimes'),
    ('discard_level_forcing', 'level - (q + 1.0 - gamma3)', '0.0 - (q + 1.0 - gamma3)',
     'test_theorem3_zero_gain_nonzero_level'),
    ('omit_growth_exponent', '(beta - k) * log_C', 'beta * log_C',
     'test_theorem2_three_regimes'),
    ('accept_failed_solver', 'if not sol.success or sol.status != 0:', 'if False:',
     'test_solver_guard_rejects'),
    ('double_ou_innovation_variance',
     'innovation = sigma * np.sqrt(-np.expm1(-2 * kappa * dt) / (2 * kappa))',
     'innovation = sigma * np.sqrt(-np.expm1(-2 * kappa * dt) / kappa)',
     'test_theorem6_ou_stationary_variance'),
]
results = []
for name, old, new, selector in mutations:
    if source.count(old) != 1:
        raise RuntimeError(f'{name}: mutation target is not unique')
    with tempfile.TemporaryDirectory(prefix='paper-x-mutation-') as tmp:
        path = Path(tmp) / 'test_mutant.py'
        path.write_text(source.replace(old, new, 1))
        run = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(path), '-k', selector],
                             capture_output=True, text=True, timeout=35)
        if run.returncode != 1:
            raise RuntimeError(f'{name}: expected assertion failure (1), got {run.returncode}\n{run.stdout}\n{run.stderr}')
        summary = run.stdout.strip().splitlines()[-1]
        results.append({'mutation': name, 'exit_code': run.returncode, 'summary': summary})
        print(name + ': ' + summary, flush=True)
(root / 'verification/mutation-results.json').write_text(json.dumps(results, indent=2) + '\n')
