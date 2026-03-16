# Honey Provenance Map

## Findings Summary

- PNG and PDF artifacts exist on disk for the honey and self-modifying figure families.
- `text.txt` contains recoverable source fragments, plot titles, export paths, and experiment descriptions for honey and self-modifying v1-v4.
- Standalone sources were recovered for `eden_honey_simulation.py`, `eden_honey_tests.py`, `eden_honey_dashboard.jsx`, and self-modifying AI v1-v4 scripts.
- Generated JSON exports recovered in the current search scope: 6.
- JSON exports still missing in the current search scope: 0.
- `anti-sycophancy.pdf` exists in Downloads, but it is outside the scope of this honey/self-mod provenance pack.

## Recovered from `text.txt`

- `eden_honey_simulation_from_text.py.txt` from lines 1752-2605 for `honey_simulation`.
- `eden_honey_tests_from_text.py.txt` from lines 1753-8796 for `honey_tests`.
- `eden_honey_dashboard_from_text.jsx.txt` from lines 1754-2105 for `honey_dashboard`.
- `eden_selfmod_v1_from_text.py.txt` from lines 4540-4765 for `selfmod_v1`.
- `eden_selfmod_v2_from_text.py.txt` from lines 5980-6170 for `selfmod_v2`.
- `eden_selfmod_v3_from_text.py.txt` from lines 7000-7265 for `selfmod_v3`.
- `eden_selfmod_v4_from_text.py.txt` from lines 8110-8285 for `selfmod_v4`.

## Standalone source files recovered

- `eden_honey_simulation.py` for `honey_simulation` at `/Users/michaeleastwood/Downloads/Honey tests and self modifying ai scripts and results/eden_honey_simulation.py`.
- `eden_honey_tests.py` for `honey_tests` at `/Users/michaeleastwood/Downloads/Honey tests and self modifying ai scripts and results/eden_honey_tests.py`.
- `eden_honey_dashboard.jsx` for `honey_dashboard` at `/Users/michaeleastwood/Downloads/Honey tests and self modifying ai scripts and results/eden_honey_dashboard.jsx`.
- `eden_self_modifying_ai.py` for `selfmod_v1` at `/Users/michaeleastwood/Downloads/Honey tests and self modifying ai scripts and results/eden_self_modifying_ai.py`.
- `eden_self_modifying_ai_v2.py` for `selfmod_v2` at `/Users/michaeleastwood/Downloads/Honey tests and self modifying ai scripts and results/eden_self_modifying_ai_v2.py`.
- `eden_self_modifying_ai_v3.py` for `selfmod_v3` at `/Users/michaeleastwood/Downloads/Honey tests and self modifying ai scripts and results/eden_self_modifying_ai_v3.py`.
- `eden_self_modifying_ai_v4.py` for `selfmod_v4` at `/Users/michaeleastwood/Downloads/Honey tests and self modifying ai scripts and results/eden_self_modifying_ai_v4.py`.

## Raw JSON/result files recovered

- `eden_honey_simulation_results.json` for `honey_simulation` (status `mapped`): Recovered raw JSON output found in the current search scope.
- `eden_honey_test_results.json` for `honey_tests` (status `mapped`): Recovered raw JSON output found in the current search scope.
- `eden_selfmod_results.json` for `selfmod_v1` (status `mapped`): Recovered raw JSON output found in the current search scope.
- `eden_selfmod_v2_results.json` for `selfmod_v2` (status `mapped`): Recovered raw JSON output found in the current search scope.
- `eden_selfmod_v3_results.json` for `selfmod_v3` (status `mapped`): Recovered raw JSON output found in the current search scope.
- `eden_selfmod_v4_results.json` for `selfmod_v4` (status `mapped`): Recovered raw JSON output found in the current search scope.

## Artifacts found on disk

- `eden honey capability.png` (honey_simulation, status `mapped`)
- `eden honey safety.png` (honey_simulation, status `mapped`)
- `eden honey ratio.png` (honey_simulation, status `mapped`)
- `eden honey simulation.pdf` (honey_simulation, status `partial`)
- `eden honey tests.pdf` (honey_tests, status `partial`)
- `eden honey dashboard.pdf` (honey_dashboard, status `partial`)
- `eden selfmod results.png` (selfmod_v1, status `mapped`)
- `eden selfmod weights.png` (selfmod_v1, status `mapped`)
- `eden self modifying ai.pdf` (selfmod_v1, status `partial`)
- `eden selfmod v2 results.png` (selfmod_v2, status `mapped`)
- `eden selfmod v2 stats.png` (selfmod_v2, status `mapped`)
- `eden self modifying ai v2.pdf` (selfmod_v2, status `partial`)
- `eden selfmod v3 results.png` (selfmod_v3, status `mapped`)
- `eden selfmod v3 stats.png` (selfmod_v3, status `mapped`)
- `eden self modifying ai v3.pdf` (selfmod_v3, status `partial`)
- `eden selfmod v4 scaling.png` (selfmod_v4, status `mapped`)
- `eden self modifying ai v4.pdf` (selfmod_v4, status `partial`)

## Referenced but not recovered as standalone source

- None in the current search scope. All referenced standalone scripts were recovered.

## Raw JSON/result files not found

- None in the current search scope.

## Safe claims vs provenance-limited claims

### Safe claims

- The honey/self-modifying figure family exists as real PNG/PDF artifacts on disk.
- `text.txt` contains embedded plotting/export logic for honey and self-modifying v1-v4.
- Standalone honey and self-modifying v1-v4 source files are now recovered in the Downloads search scope.
- Eden v3 is compile-clean, exposes a documented CLI, and has explicit env/output conventions.

### Provenance-limited claims

- Honey and self-modifying numerical findings should not be treated as fully reproducible from standalone source until the exported JSONs are matched to the published PDFs/PNGs and checked for consistency.
- Any JSON recovered with `demo_mode=true` is a derived/demo artifact, not a canonical raw benchmark run.
- The self-modifying PDFs look like derived presentation artifacts rather than canonical raw-result containers.

