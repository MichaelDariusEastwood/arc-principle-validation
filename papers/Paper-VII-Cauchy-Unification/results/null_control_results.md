# Null-Control Results

These controls test whether the stricter 25-domain empirical result can be reproduced after breaking the predictive structure.

- Observed empirical headline: `19/25`
- Family-label null `p(match >= observed)` over `10000` trials: `0.000000`
- Shuffled-y null `p(match >= observed)` over `20` iterations: `0.000000`

## Family-label null

- Mean matches: `8.326`
- Std dev: `2.372`
- 99th percentile: `14.000`
- Max observed in simulation: `18`

## Shuffled-y null

- Mean total matches: `7.300`
- Std dev: `1.342`
- 99th percentile: `9.810`
- Max observed in simulation: `10`

Winner-family share under shuffled-y null:

- `bounded`: `0.8000`
- `exponential`: `0.1180`
- `power_law`: `0.0820`

Highest per-domain false-confirmation rates under shuffled-y null:

- `11` Bacterial Growth (E. coli Logistic): `1.000` for predicted family `bounded`
- `13` Epidemic SIR (2014 Ebola): `1.000` for predicted family `bounded`
- `12` O2-Hemoglobin Curve (Hill Equation): `0.950` for predicted family `bounded`
- `15` Muscle Force-Velocity (Hill 1938): `0.850` for predicted family `bounded`
- `16` Network Growth Rate (Facebook MAU): `0.750` for predicted family `bounded`
- `14` Amdahl's Law (CPU Multi-Core Scaling): `0.600` for predicted family `bounded`
- `20` Time Crystal Order Parameter (Rydberg Gas): `0.450` for predicted family `bounded`
- `19` Neural Scaling Laws (LLM Loss vs Params): `0.300` for predicted family `power_law`
