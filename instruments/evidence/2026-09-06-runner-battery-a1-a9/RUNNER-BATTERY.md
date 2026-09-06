# Runner battery: the analysis code's operating characteristics, world by world (2026-09-06T12:34:34Z, 2 seeds per row)

The runner is the code that will score real data. Each row runs it many times in one simulated world. Never evidence about any real system.

## P5 (arc_runner.p5)

The route-agreement columns and the identification columns answer different questions and are tabulated separately (finding A6). Route agreement is a numerical consistency check between two regression directions through one bank: it tests whether the elasticity in the retention fraction and the elasticity in the capability state are one number. Identification asks whether the capability elasticity is identified at all, and reaches IDENTIFIED only on an independent capability manipulation, which none of these simulated worlds supplies, so NOT ESTABLISHED is the correct reading of every row that carries it.

| World | true beta | SUPPORTED | REFUTED | INCONCLUSIVE | NOT EVALUABLE | routes CONSISTENT | routes INCONSISTENT | routes UNRESOLVED | ident NOT IDENTIFIED | ident NOT ESTABLISHED | premise HOLDS | premise NOT REFUTED | premise REFUTED | mean beta pooled (sd) | mean capability elasticity (sd) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| true, registered configuration | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.511 (0.008) | 0.518 (0.005) |
| true, cheap ladder (4,000 items) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.482 (0.030) | 0.516 (0.020) |
| true, four calibration reads instead of sixteen | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.511 (0.008) | 0.518 (0.005) |
| true, four replicates instead of eight | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.499 (0.008) | 0.497 (0.015) |
| true, noisier system (0.08) | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.519 (0.009) | 0.521 (0.011) |
| true, one held-out system | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.511 (0.008) | 0.518 (0.005) |
| rate confound (theta 0.2) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.627 (0.002) | 0.711 (0.020) |
| rate confound (theta 0.1) | 0.50 | 0.50 | 0.00 | 0.50 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 1.00 | 0.00 | 0.00 | 0.572 (0.014) | 0.617 (0.026) |
| true at a different coupling (0.35): mechanism holds, must SUPPORT | 0.35 | 1.00 | 0.00 | 0.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.351 (0.012) | 0.369 (0.014) |
| REGIME CHANGE: bank at 0.50, held-out evolves at 0.35; must REFUTE | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.511 (0.008) | 0.518 (0.005) |
| REGIME CHANGE: bank at 0.50, held-out evolves at 0.65; must REFUTE | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.513 (0.008) | 0.535 (0.028) |
| REGIME CHANGE, small: held-out at 0.40 (the registered 0.10 shift) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.511 (0.008) | 0.518 (0.005) |
| no coupling (beta 0): linear growth, mechanism holds | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.00 | 1.00 | 0.00 | -0.010 (0.000) | 0.003 (0.000) |
| ladder without headroom (scale 100) | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | n/a | n/a |
| true, REGISTERED WINDOW 128 (ladder scale 10,000, 200,000 items) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.506 (0.013) | 0.518 (0.030) |
| REGIME CHANGE 0.50 to 0.35 at window 128; must REFUTE | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.506 (0.013) | 0.518 (0.030) |
| REGIME CHANGE 0.50 to 0.40 at window 128 (the registered 0.10 shift) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.506 (0.013) | 0.518 (0.030) |
| no coupling (beta 0) at window 128 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | n/a | n/a |
| rate confound (theta 0.2) at window 128 | 0.50 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.622 (0.004) | 0.702 (0.002) |
| NEGATIVE CONTROL LEAKS: control cells carry increment coupling 0.30; must be NOT IDENTIFIED | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.511 (0.008) | 0.518 (0.005) |
| NEGATIVE CONTROL LEAKS, mild: control coupling 0.15 | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.00 | 0.511 (0.008) | 0.518 (0.005) |
| true, REGISTERED WINDOW 128, ladder with 800,000 items (bank-precise) | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.498 (0.010) | 0.500 (0.003) |
| REGIME CHANGE 0.50 to 0.35 at window 128, bank-precise ladder; must REFUTE | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.498 (0.010) | 0.500 (0.003) |
| REGIME CHANGE 0.50 to 0.40 at window 128, bank-precise ladder (the 0.10 shift) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.498 (0.010) | 0.500 (0.003) |
| CANDIDATE: window 128, 800,000-item ladder, 16 bank replicates; true | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.500 (0.002) | 0.497 (0.001) |
| CANDIDATE, REGIME CHANGE 0.50 to 0.35; must REFUTE | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.500 (0.002) | 0.497 (0.001) |
| CANDIDATE, REGIME CHANGE 0.50 to 0.40 (the 0.10 shift) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.500 (0.002) | 0.497 (0.001) |
| CANDIDATE, noisier system (0.08) | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.498 (0.003) | 0.496 (0.001) |
| CANDIDATE, control leaks 0.30; must be NOT IDENTIFIED | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.50 | 0.00 | 0.500 (0.002) | 0.497 (0.001) |
| CANDIDATE, saturation at 400 (H3 must refute the premise) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.365 (0.002) | 0.352 (0.004) |
| CANDIDATE, bank states to 1,000 (30, 100, 250, 500, 1000); true | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.501 (0.001) | 0.502 (0.000) |
| CANDIDATE, bank states to 1,000; saturation at 400 (H3 must refute) | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.284 (0.003) | 0.280 (0.006) |
| CANDIDATE, bank states to 1,000; REGIME CHANGE 0.50 to 0.35 | 0.50 | 0.00 | 0.50 | 0.50 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 1.00 | 0.00 | 0.00 | 0.501 (0.001) | 0.502 (0.000) |
| SATURATION (H3): increment saturates at available 60; premise must be REFUTED | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.098 (0.003) | 0.087 (0.004) |
| SATURATION (H3), mild: saturates at 400 | 0.50 | 0.50 | 0.00 | 0.50 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.50 | 0.50 | 0.361 (0.001) | 0.363 (0.002) |
| CROSSED: retention exponent 0.9 against coupling 0.5; routes must be INCONSISTENT | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.644 (0.006) | 0.517 (0.005) |
| CROSSED, mild: retention exponent 0.65 against coupling 0.5 | 0.50 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.575 (0.014) | 0.528 (0.015) |
| SHARED RATE FACTOR: nuisance scaling with available capability at 0.2; routes must AGREE at 0.7 while the coupling is 0.5 | 0.50 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 1.00 | 0.50 | 0.50 | 0.00 | 0.692 (0.010) | 0.696 (0.016) |

### Uncertainty on the P5 rates (exact two-sided 95 per cent intervals; 2 outer repetitions per row; 200 inner resamples per run inside the analysis, which do not narrow these)

| World | headline | exact interval | one-sided upper on the OTHER verdicts |
|---|---|---|---|
| true, registered configuration | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| true, registered configuration (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| true, cheap ladder (4,000 items) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| true, cheap ladder (4,000 items) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| true, four calibration reads instead of sixteen | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| true, four calibration reads instead of sixteen (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| true, four replicates instead of eight | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| true, four replicates instead of eight (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| true, noisier system (0.08) | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| true, noisier system (0.08) (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| true, one held-out system | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| true, one held-out system (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| rate confound (theta 0.2) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| rate confound (theta 0.2) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| rate confound (theta 0.1) | INCONCLUSIVE 0.50 | [0.013, 0.987] | SUPPORTED at most 0.975 |
| rate confound (theta 0.1) (unobserved) | REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| true at a different coupling (0.35): mechanism holds, must SUPPORT | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| true at a different coupling (0.35): mechanism holds, must SUPPORT (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| REGIME CHANGE: bank at 0.50, held-out evolves at 0.35; must REFUTE | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| REGIME CHANGE: bank at 0.50, held-out evolves at 0.35; must REFUTE (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| REGIME CHANGE: bank at 0.50, held-out evolves at 0.65; must REFUTE | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| REGIME CHANGE: bank at 0.50, held-out evolves at 0.65; must REFUTE (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| REGIME CHANGE, small: held-out at 0.40 (the registered 0.10 shift) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| REGIME CHANGE, small: held-out at 0.40 (the registered 0.10 shift) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| no coupling (beta 0): linear growth, mechanism holds | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| no coupling (beta 0): linear growth, mechanism holds (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| ladder without headroom (scale 100) | NOT EVALUABLE 1.00 | [0.158, 1.000] | none observed |
| ladder without headroom (scale 100) (unobserved) | SUPPORTED, REFUTED, INCONCLUSIVE | 0 of 2 | each at most 0.7764 |
| true, REGISTERED WINDOW 128 (ladder scale 10,000, 200,000 items) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| true, REGISTERED WINDOW 128 (ladder scale 10,000, 200,000 items) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| REGIME CHANGE 0.50 to 0.35 at window 128; must REFUTE | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| REGIME CHANGE 0.50 to 0.35 at window 128; must REFUTE (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| REGIME CHANGE 0.50 to 0.40 at window 128 (the registered 0.10 shift) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| REGIME CHANGE 0.50 to 0.40 at window 128 (the registered 0.10 shift) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| no coupling (beta 0) at window 128 | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| no coupling (beta 0) at window 128 (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| rate confound (theta 0.2) at window 128 | NOT EVALUABLE 1.00 | [0.158, 1.000] | none observed |
| rate confound (theta 0.2) at window 128 (unobserved) | SUPPORTED, REFUTED, INCONCLUSIVE | 0 of 2 | each at most 0.7764 |
| NEGATIVE CONTROL LEAKS: control cells carry increment coupling 0.30; must be NOT IDENTIFIED | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| NEGATIVE CONTROL LEAKS: control cells carry increment coupling 0.30; must be NOT IDENTIFIED (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| NEGATIVE CONTROL LEAKS, mild: control coupling 0.15 | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| NEGATIVE CONTROL LEAKS, mild: control coupling 0.15 (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| true, REGISTERED WINDOW 128, ladder with 800,000 items (bank-precise) | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| true, REGISTERED WINDOW 128, ladder with 800,000 items (bank-precise) (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| REGIME CHANGE 0.50 to 0.35 at window 128, bank-precise ladder; must REFUTE | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| REGIME CHANGE 0.50 to 0.35 at window 128, bank-precise ladder; must REFUTE (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| REGIME CHANGE 0.50 to 0.40 at window 128, bank-precise ladder (the 0.10 shift) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| REGIME CHANGE 0.50 to 0.40 at window 128, bank-precise ladder (the 0.10 shift) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE: window 128, 800,000-item ladder, 16 bank replicates; true | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE: window 128, 800,000-item ladder, 16 bank replicates; true (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, REGIME CHANGE 0.50 to 0.35; must REFUTE | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE, REGIME CHANGE 0.50 to 0.35; must REFUTE (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, REGIME CHANGE 0.50 to 0.40 (the 0.10 shift) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE, REGIME CHANGE 0.50 to 0.40 (the 0.10 shift) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, noisier system (0.08) | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE, noisier system (0.08) (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, control leaks 0.30; must be NOT IDENTIFIED | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE, control leaks 0.30; must be NOT IDENTIFIED (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, saturation at 400 (H3 must refute the premise) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE, saturation at 400 (H3 must refute the premise) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, bank states to 1,000 (30, 100, 250, 500, 1000); true | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE, bank states to 1,000 (30, 100, 250, 500, 1000); true (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, bank states to 1,000; saturation at 400 (H3 must refute) | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| CANDIDATE, bank states to 1,000; saturation at 400 (H3 must refute) (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CANDIDATE, bank states to 1,000; REGIME CHANGE 0.50 to 0.35 | REFUTED 0.50 | [0.013, 0.987] | INCONCLUSIVE at most 0.975 |
| CANDIDATE, bank states to 1,000; REGIME CHANGE 0.50 to 0.35 (unobserved) | SUPPORTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| SATURATION (H3): increment saturates at available 60; premise must be REFUTED | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| SATURATION (H3): increment saturates at available 60; premise must be REFUTED (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| SATURATION (H3), mild: saturates at 400 | SUPPORTED 0.50 | [0.013, 0.987] | INCONCLUSIVE at most 0.975 |
| SATURATION (H3), mild: saturates at 400 (unobserved) | REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CROSSED: retention exponent 0.9 against coupling 0.5; routes must be INCONSISTENT | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| CROSSED: retention exponent 0.9 against coupling 0.5; routes must be INCONSISTENT (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| CROSSED, mild: retention exponent 0.65 against coupling 0.5 | SUPPORTED 1.00 | [0.158, 1.000] | none observed |
| CROSSED, mild: retention exponent 0.65 against coupling 0.5 (unobserved) | REFUTED, INCONCLUSIVE, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |
| SHARED RATE FACTOR: nuisance scaling with available capability at 0.2; routes must AGREE at 0.7 while the coupling is 0.5 | INCONCLUSIVE 1.00 | [0.158, 1.000] | none observed |
| SHARED RATE FACTOR: nuisance scaling with available capability at 0.2; routes must AGREE at 0.7 while the coupling is 0.5 (unobserved) | SUPPORTED, REFUTED, NOT EVALUABLE | 0 of 2 | each at most 0.7764 |

## P16 (arc_runner.p16)

| World | SUPPORTED | REFUTED (no reversal) | REFUTED (mislocated) | NOT SPECIFIC | INCONCLUSIVE | INCONCLUSIVE (silent, nothing demonstrated) |
|---|---|---|---|---|---|---|
| true, registered configuration | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| true, cheap tier (48 rounds, one system per arm) | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| true, noisier margins (0.08) | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| true, boundary mislocated by 0.3 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 |
| no boundary | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| generic deterioration in every arm | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 0.00 |
| true, threshold 3.0 (uncalibrated) | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

### Uncertainty on the P16 rates (exact two-sided 95 per cent intervals; 2 outer repetitions per row; the detection rule uses no inner resampling)

| World | headline | exact interval | unobserved verdicts, each at most |
|---|---|---|---|
| true, registered configuration | SUPPORTED 1.00 | [0.158, 1.000] | 0.7764 |
| true, cheap tier (48 rounds, one system per arm) | SUPPORTED 1.00 | [0.158, 1.000] | 0.7764 |
| true, noisier margins (0.08) | SUPPORTED 1.00 | [0.158, 1.000] | 0.7764 |
| true, boundary mislocated by 0.3 | REFUTED (boundary mislocated) 1.00 | [0.158, 1.000] | 0.7764 |
| no boundary | INCONCLUSIVE (no reversal, and none demonstrated absent) 1.00 | [0.158, 1.000] | 0.7764 |
| generic deterioration in every arm | NOT SPECIFIC (INCONCLUSIVE) 1.00 | [0.158, 1.000] | 0.7764 |
| true, threshold 3.0 (uncalibrated) | SUPPORTED 1.00 | [0.158, 1.000] | 0.7764 |

A rate of zero in 2 runs is not zero risk: its one-sided 95 per cent upper bound is 0.7764, and it says nothing about worlds this battery did not run.
