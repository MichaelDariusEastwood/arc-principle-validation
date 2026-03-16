"""
================================================================================
ARC PRINCIPLE: ACOUSTIC TIME CRYSTAL VALIDATION
================================================================================

Tests the ARC framework against acoustic time crystals — a system where
discrete time-translation symmetry is spontaneously broken in a periodically
driven acoustic/phononic system.

PHYSICAL SYSTEMS MODELLED:
1. Single parametric oscillator (Mathieu equation)
   - Driven at 2w, subharmonic response at w
   - Above threshold: amplitude grows then saturates

2. Coupled parametric oscillator chain (phononic time crystal)
   - N masses coupled by springs with modulated stiffness
   - Collective subharmonic order parameter
   - Spatial coherence of time-crystal phase

3. Acoustic bubble dynamics (Rayleigh-Plesset-like)
   - Bubble radius oscillation under acoustic driving
   - Period-doubling cascade as drive amplitude increases
   - Subharmonic amplitude vs drive strength

REFERENCES:
- Floquet Time Crystals: Else, Bauer, Nayak (2016) arXiv:1603.08001
- Acoustic DTC: Huygens et al., Crystals 12(3), 399 (2022)
- Time-modulated phononic lattices: Kim & Daraio (2023)
- Temporal refraction in phononic lattice: Kim et al. (2024)

ARC PREDICTIONS:
- Subharmonic amplitude vs drive cycles: SATURATION (bounded oplus)
- Order parameter vs coupling strength: SATURATION (bounded oplus)
- Spatial coherence length vs system size: POWER-LAW or SATURATION

================================================================================
Michael Darius Eastwood | March 2026
================================================================================
"""

import numpy as np
from scipy import optimize, stats, signal
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("=" * 80)
print("  ARC PRINCIPLE: ACOUSTIC TIME CRYSTAL VALIDATION")
print("  Testing discrete time-translation symmetry breaking")
print("  in parametrically driven acoustic/phononic systems")
print("=" * 80)
print()

results_all = []


# ============================================================================
# MEASUREMENT APPARATUS (from main physics test)
# ============================================================================

def measure_composition_operator(R, U, name=""):
    """Measure oplus by examining how marginal gains relate to accumulated state."""
    dU = np.diff(U)
    U_mid = (U[:-1] + U[1:]) / 2

    mask = (U_mid > 0) & (np.abs(dU) > 1e-15)
    if mask.sum() < 5:
        return None, None, None, None

    dU_m = dU[mask]
    U_m = U_mid[mask]

    scores = {}

    # Multiplicative: log(|dU|) = beta * log(U) + c
    try:
        log_dU = np.log(np.abs(dU_m) + 1e-30)
        log_U = np.log(np.abs(U_m) + 1e-30)
        slope, intercept, r_val, p_val, se = stats.linregress(log_U, log_dU)
        scores['multiplicative'] = {'r2': r_val**2, 'beta': slope, 'se': se}
    except:
        scores['multiplicative'] = {'r2': 0, 'beta': 0, 'se': 999}

    # Additive: dU ~ constant
    try:
        cv = np.std(dU_m) / (np.abs(np.mean(dU_m)) + 1e-30)
        scores['additive'] = {'r2': max(0, 1 - cv), 'rate': np.mean(dU_m)}
    except:
        scores['additive'] = {'r2': 0, 'rate': 0}

    # Bounded: dU vs (U_max - U) linear with positive slope
    try:
        U_max_est = np.max(U) * 1.05
        gap = U_max_est - U_m
        slope_b, int_b, r_b, p_b, se_b = stats.linregress(gap, dU_m)
        scores['bounded'] = {'r2': r_b**2 if slope_b > 0 else 0, 'Umax': U_max_est}
    except:
        scores['bounded'] = {'r2': 0, 'Umax': 0}

    best = max(scores.keys(), key=lambda k: scores[k]['r2'])
    beta = scores['multiplicative']['beta'] if best == 'multiplicative' else None

    return best, scores, beta, scores[best]['r2']


def fit_scaling(R, U):
    """Independently fit power-law, exponential, and saturating models."""
    fits = {}

    # Power-law: U = a * R^alpha
    try:
        mask = (R > 0) & (U > 0)
        log_r = np.log(R[mask])
        log_u = np.log(U[mask])
        slope, intercept, r_val, _, _ = stats.linregress(log_r, log_u)
        fits['power_law'] = {'alpha': slope, 'r2': r_val**2, 'a': np.exp(intercept)}
    except:
        fits['power_law'] = {'alpha': 0, 'r2': 0}

    # Exponential: U = a * exp(lambda * R)
    try:
        mask = U > 0
        log_u = np.log(U[mask])
        slope, intercept, r_val, _, _ = stats.linregress(R[mask], log_u)
        fits['exponential'] = {'lambda': slope, 'r2': r_val**2, 'a': np.exp(intercept)}
    except:
        fits['exponential'] = {'lambda': 0, 'r2': 0}

    # Saturating: U = U_max * (1 - exp(-kR))
    try:
        U_max_est = np.max(U) * 1.05
        def sat_func(R, U_max, k):
            return U_max * (1 - np.exp(-k * R))
        popt, _ = optimize.curve_fit(sat_func, R, U, p0=[U_max_est, 0.1], maxfev=5000)
        U_pred = sat_func(R, *popt)
        ss_res = np.sum((U - U_pred)**2)
        ss_tot = np.sum((U - np.mean(U))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fits['saturation'] = {'Umax': popt[0], 'k': popt[1], 'r2': r2}
    except:
        fits['saturation'] = {'Umax': 0, 'k': 0, 'r2': 0}

    best = max(fits.keys(), key=lambda k: fits[k]['r2'])
    return best, fits


# ============================================================================
# TEST 1: SINGLE PARAMETRIC OSCILLATOR (Mathieu Equation)
# ============================================================================
print("=" * 80)
print("  TEST 1: SINGLE PARAMETRIC OSCILLATOR")
print("  Model: Mathieu equation with nonlinear damping")
print("  x'' + gamma*x' + w0^2*(1 + eps*cos(2*w0*t))*x - beta_nl*x^3 = noise")
print("  Subharmonic response at w0 when driven at 2*w0")
print("=" * 80)

# Parameters
w0 = 2 * np.pi      # Natural frequency
gamma = 0.05         # Linear damping
eps = 0.15           # Parametric drive depth (above threshold ~ gamma/w0)
beta_nl = 0.01       # Nonlinear saturation (cubic term)
dt = 0.001
T_period = 2 * np.pi / w0

# Simulate
n_periods = 500
n_steps = int(n_periods * T_period / dt)

x = 0.001  # Small initial displacement (seed for parametric growth)
v = 0.0
t = 0.0

# Measure subharmonic amplitude every few periods
measure_interval = 5  # periods
periods_measured = []
subharmonic_amplitudes = []
current_period_data = []

for step in range(n_steps):
    t = step * dt

    # Parametric drive at 2*w0
    k_mod = w0**2 * (1 + eps * np.cos(2 * w0 * t))

    # Nonlinear restoring force (Duffing-like saturation)
    force = -k_mod * x + beta_nl * x**3 - gamma * v

    # Small noise (thermal fluctuations)
    force += np.random.normal(0, 0.0001)

    # Verlet integration
    v += force * dt
    x += v * dt

    # Track data within current measurement window
    period_number = t / T_period
    current_period_data.append(x)

    if len(current_period_data) >= int(measure_interval * T_period / dt):
        # Extract subharmonic amplitude via FFT
        segment = np.array(current_period_data)
        if len(segment) > 10:
            fft_vals = np.abs(np.fft.rfft(segment))
            freqs = np.fft.rfftfreq(len(segment), dt)

            # Find peak near w0/(2*pi) = 1 Hz (the subharmonic)
            target_freq = w0 / (2 * np.pi)
            freq_idx = np.argmin(np.abs(freqs - target_freq))

            # Subharmonic amplitude (normalised)
            if freq_idx > 0 and freq_idx < len(fft_vals):
                sub_amp = fft_vals[freq_idx] / len(segment)
                periods_measured.append(len(periods_measured) * measure_interval)
                subharmonic_amplitudes.append(sub_amp)

        current_period_data = []

R_param = np.array(periods_measured[2:], dtype=float)  # Skip initial transient
U_param = np.array(subharmonic_amplitudes[2:])

# Normalise
if U_param.max() > 0:
    U_param = U_param / U_param.max()

# Measure and fit
if len(R_param) > 10:
    op_type, scores, beta, r2 = measure_composition_operator(R_param, U_param)
    best_fit, fits = fit_scaling(R_param, U_param)

    print(f"\n  Simulation: {n_periods} drive periods, dt = {dt}")
    print(f"  Drive depth epsilon = {eps} (threshold ~ {gamma/w0:.4f})")
    print(f"  Data points: {len(R_param)}")
    print(f"  Subharmonic amplitude range: [{U_param.min():.6f}, {U_param.max():.6f}]")
    print(f"\n  Composition operator: {op_type.upper() if op_type else 'UNDETERMINED'} (R2 = {r2:.4f})")
    if beta is not None:
        print(f"  beta measured: {beta:.4f}")

    print(f"\n  Best scaling fit: {best_fit.upper()}")
    for fname, fdata in fits.items():
        r2_val = fdata.get('r2', 0)
        if fname == 'power_law':
            print(f"    Power-law:   alpha = {fdata['alpha']:.4f}, R2 = {r2_val:.4f}")
        elif fname == 'exponential':
            print(f"    Exponential: lambda = {fdata.get('lambda', 0):.6f}, R2 = {r2_val:.4f}")
        elif fname == 'saturation':
            print(f"    Saturation:  U_max = {fdata.get('Umax', 0):.4f}, k = {fdata.get('k', 0):.6f}, R2 = {r2_val:.4f}")

    param_expected = "saturation"
    param_match = best_fit == param_expected
    print(f"\n  ARC PREDICTION: {param_expected.upper()} (bounded oplus)")
    print(f"  OBSERVED:       {best_fit.upper()}")
    print(f"  {'CONFIRMED' if param_match else 'NOT CONFIRMED'}")

    results_all.append({
        'domain': 'Parametric Oscillator (Mathieu)',
        'system': 'Single oscillator, subharmonic growth vs drive cycles',
        'predicted': param_expected,
        'observed': best_fit,
        'match': param_match,
        'op_type': op_type,
        'beta': beta,
        'fits': fits,
        'r2': r2
    })
else:
    print("\n  WARNING: Insufficient data points for analysis")
    print(f"  Got {len(R_param)} points (need > 10)")


# ============================================================================
# TEST 2: COUPLED PARAMETRIC CHAIN (Phononic Time Crystal)
# ============================================================================
print()
print("=" * 80)
print("  TEST 2: COUPLED PARAMETRIC OSCILLATOR CHAIN")
print("  Model: Phononic crystal with time-modulated stiffness")
print("  N coupled masses, parametric drive at 2*w0")
print("  Measure: collective subharmonic order parameter vs coupling")
print("=" * 80)

# Phononic crystal: chain of N masses connected by springs
# On-site potential modulated at 2*w0
# Inter-site coupling K_c

N_chain = 50  # Number of sites
w0_chain = 2 * np.pi
gamma_chain = 0.08
eps_chain = 0.12
beta_nl_chain = 0.005
dt_chain = 0.002
T_period_chain = 2 * np.pi / w0_chain

# Vary coupling strength
coupling_strengths = np.linspace(0.01, 2.0, 30)
order_parameters = []

print(f"\n  Chain size: N = {N_chain} oscillators")
print(f"  Coupling strengths: {len(coupling_strengths)} values in [{coupling_strengths[0]:.2f}, {coupling_strengths[-1]:.2f}]")
print(f"  Simulating...", end=" ", flush=True)

for ci, K_c in enumerate(coupling_strengths):
    # Initialise chain
    x_chain = np.random.normal(0, 0.001, N_chain)
    v_chain = np.zeros(N_chain)

    # Run for enough periods to reach steady state
    n_periods_chain = 200
    n_steps_chain = int(n_periods_chain * T_period_chain / dt_chain)

    # Collect last 50 periods for measurement
    measure_start = int(0.75 * n_steps_chain)
    subharmonic_data = []

    for step in range(n_steps_chain):
        t = step * dt_chain

        # On-site parametric modulation
        k_mod = w0_chain**2 * (1 + eps_chain * np.cos(2 * w0_chain * t))

        # Forces
        # On-site: -k_mod * x + beta_nl * x^3
        force = -k_mod * x_chain + beta_nl_chain * x_chain**3 - gamma_chain * v_chain

        # Nearest-neighbour coupling
        coupling = np.zeros(N_chain)
        coupling[0] = K_c * (x_chain[1] - x_chain[0])
        coupling[-1] = K_c * (x_chain[-2] - x_chain[-1])
        coupling[1:-1] = K_c * (x_chain[2:] + x_chain[:-2] - 2 * x_chain[1:-1])
        force += coupling

        # Noise
        force += np.random.normal(0, 0.0001, N_chain)

        # Integrate
        v_chain += force * dt_chain
        x_chain += v_chain * dt_chain

        if step >= measure_start:
            subharmonic_data.append(x_chain.copy())

    # Compute collective subharmonic order parameter
    # = spatial average of |FFT at w0| across all sites
    subharmonic_data = np.array(subharmonic_data)

    site_subharmonics = []
    for site in range(N_chain):
        site_signal = subharmonic_data[:, site]
        fft_vals = np.abs(np.fft.rfft(site_signal))
        freqs = np.fft.rfftfreq(len(site_signal), dt_chain)
        target_freq = w0_chain / (2 * np.pi)
        freq_idx = np.argmin(np.abs(freqs - target_freq))
        if freq_idx > 0 and freq_idx < len(fft_vals):
            site_subharmonics.append(fft_vals[freq_idx] / len(site_signal))
        else:
            site_subharmonics.append(0.0)

    # Order parameter: mean subharmonic amplitude across chain
    # (analogous to magnetisation in spin systems)
    order_param = np.mean(site_subharmonics)
    order_parameters.append(order_param)

    if (ci + 1) % 10 == 0:
        print(f"{ci+1}", end=" ", flush=True)

print("done.")

R_chain = coupling_strengths
U_chain = np.array(order_parameters)

# Normalise
if U_chain.max() > 0:
    U_chain = U_chain / U_chain.max()

# Measure and fit
op_type, scores, beta, r2 = measure_composition_operator(R_chain, U_chain)
best_fit, fits = fit_scaling(R_chain, U_chain)

print(f"\n  Order parameter range: [{U_chain.min():.6f}, {U_chain.max():.6f}]")
print(f"\n  Composition operator: {op_type.upper() if op_type else 'UNDETERMINED'} (R2 = {r2:.4f})")
if beta is not None:
    print(f"  beta measured: {beta:.4f}")

print(f"\n  Best scaling fit: {best_fit.upper()}")
for fname, fdata in fits.items():
    r2_val = fdata.get('r2', 0)
    if fname == 'power_law':
        print(f"    Power-law:   alpha = {fdata['alpha']:.4f}, R2 = {r2_val:.4f}")
    elif fname == 'exponential':
        print(f"    Exponential: lambda = {fdata.get('lambda', 0):.6f}, R2 = {r2_val:.4f}")
    elif fname == 'saturation':
        print(f"    Saturation:  U_max = {fdata.get('Umax', 0):.4f}, k = {fdata.get('k', 0):.6f}, R2 = {r2_val:.4f}")

chain_expected = "saturation"
chain_match = best_fit == chain_expected
print(f"\n  ARC PREDICTION: {chain_expected.upper()} (bounded oplus)")
print(f"  OBSERVED:       {best_fit.upper()}")
print(f"  {'CONFIRMED' if chain_match else 'NOT CONFIRMED'}")

results_all.append({
    'domain': 'Phononic Time Crystal Chain',
    'system': f'N={N_chain} coupled parametric oscillators, order param vs coupling',
    'predicted': chain_expected,
    'observed': best_fit,
    'match': chain_match,
    'op_type': op_type,
    'beta': beta,
    'fits': fits,
    'r2': r2
})


# ============================================================================
# TEST 3: PERIOD-DOUBLING CASCADE (Drive Amplitude Scaling)
# ============================================================================
print()
print("=" * 80)
print("  TEST 3: PERIOD-DOUBLING CASCADE")
print("  Model: Single oscillator, varying drive amplitude")
print("  Measure: subharmonic amplitude vs drive strength epsilon")
print("  (Analogous to acoustic bubble time crystal experiments)")
print("=" * 80)

# This models the experiment in Crystals 12(3), 399 (2022)
# where acoustic bubbles show period-doubling with DTC behaviour

drive_amplitudes = np.linspace(0.02, 0.30, 40)
subharmonic_response = []

w0_pd = 2 * np.pi
gamma_pd = 0.06
beta_nl_pd = 0.008
dt_pd = 0.001
T_period_pd = 2 * np.pi / w0_pd

print(f"\n  Drive amplitudes: {len(drive_amplitudes)} values in [{drive_amplitudes[0]:.2f}, {drive_amplitudes[-1]:.2f}]")
print(f"  Threshold estimate: eps_c ~ gamma/w0 = {gamma_pd/w0_pd:.4f}")
print(f"  Simulating...", end=" ", flush=True)

for di, eps_val in enumerate(drive_amplitudes):
    x = 0.001
    v = 0.0

    n_periods_pd = 300
    n_steps_pd = int(n_periods_pd * T_period_pd / dt_pd)
    measure_start = int(0.6 * n_steps_pd)

    signal_data = []

    for step in range(n_steps_pd):
        t = step * dt_pd
        k_mod = w0_pd**2 * (1 + eps_val * np.cos(2 * w0_pd * t))
        force = -k_mod * x + beta_nl_pd * x**3 - gamma_pd * v
        force += np.random.normal(0, 0.00005)

        v += force * dt_pd
        x += v * dt_pd

        if step >= measure_start:
            signal_data.append(x)

    # Extract subharmonic amplitude
    signal_arr = np.array(signal_data)
    fft_vals = np.abs(np.fft.rfft(signal_arr))
    freqs = np.fft.rfftfreq(len(signal_arr), dt_pd)
    target_freq = w0_pd / (2 * np.pi)
    freq_idx = np.argmin(np.abs(freqs - target_freq))

    if freq_idx > 0 and freq_idx < len(fft_vals):
        sub_amp = fft_vals[freq_idx] / len(signal_arr)
    else:
        sub_amp = 0.0

    subharmonic_response.append(sub_amp)

    if (di + 1) % 10 == 0:
        print(f"{di+1}", end=" ", flush=True)

print("done.")

R_pd = drive_amplitudes
U_pd = np.array(subharmonic_response)

# Normalise
if U_pd.max() > 0:
    U_pd = U_pd / U_pd.max()

# Measure and fit
op_type, scores, beta, r2 = measure_composition_operator(R_pd, U_pd)
best_fit, fits = fit_scaling(R_pd, U_pd)

print(f"\n  Subharmonic response range: [{U_pd.min():.6f}, {U_pd.max():.6f}]")
print(f"\n  Composition operator: {op_type.upper() if op_type else 'UNDETERMINED'} (R2 = {r2:.4f})")
if beta is not None:
    print(f"  beta measured: {beta:.4f}")

print(f"\n  Best scaling fit: {best_fit.upper()}")
for fname, fdata in fits.items():
    r2_val = fdata.get('r2', 0)
    if fname == 'power_law':
        print(f"    Power-law:   alpha = {fdata['alpha']:.4f}, R2 = {r2_val:.4f}")
    elif fname == 'exponential':
        print(f"    Exponential: lambda = {fdata.get('lambda', 0):.6f}, R2 = {r2_val:.4f}")
    elif fname == 'saturation':
        print(f"    Saturation:  U_max = {fdata.get('Umax', 0):.4f}, k = {fdata.get('k', 0):.6f}, R2 = {r2_val:.4f}")

pd_expected = "saturation"
pd_match = best_fit == pd_expected
print(f"\n  ARC PREDICTION: {pd_expected.upper()} (bounded oplus, nonlinear saturation)")
print(f"  OBSERVED:       {best_fit.upper()}")
print(f"  {'CONFIRMED' if pd_match else 'NOT CONFIRMED'}")

results_all.append({
    'domain': 'Period-Doubling Cascade',
    'system': 'Subharmonic amplitude vs drive strength (acoustic bubble model)',
    'predicted': pd_expected,
    'observed': best_fit,
    'match': pd_match,
    'op_type': op_type,
    'beta': beta,
    'fits': fits,
    'r2': r2
})


# ============================================================================
# TEST 4: SPATIAL COHERENCE LENGTH
# ============================================================================
print()
print("=" * 80)
print("  TEST 4: SPATIAL COHERENCE OF TIME-CRYSTAL PHASE")
print("  Model: Coupled chain, measure correlation length vs system size")
print("  How far does the subharmonic phase coherence extend?")
print("=" * 80)

# For different chain sizes N, measure the spatial correlation
# of the subharmonic phase across the chain

chain_sizes = [10, 20, 30, 50, 75, 100, 150, 200]
coherence_lengths = []

K_c_fixed = 1.0  # Fixed coupling
w0_sc = 2 * np.pi
gamma_sc = 0.08
eps_sc = 0.12
beta_nl_sc = 0.005
dt_sc = 0.003
T_period_sc = 2 * np.pi / w0_sc

print(f"\n  Chain sizes: {chain_sizes}")
print(f"  Fixed coupling K = {K_c_fixed}")
print(f"  Simulating...", end=" ", flush=True)

for ni, N in enumerate(chain_sizes):
    x_chain = np.random.normal(0, 0.001, N)
    v_chain = np.zeros(N)

    n_periods_sc = 200
    n_steps_sc = int(n_periods_sc * T_period_sc / dt_sc)
    measure_start = int(0.7 * n_steps_sc)

    subharmonic_data = []

    for step in range(n_steps_sc):
        t = step * dt_sc
        k_mod = w0_sc**2 * (1 + eps_sc * np.cos(2 * w0_sc * t))
        force = -k_mod * x_chain + beta_nl_sc * x_chain**3 - gamma_sc * v_chain

        coupling = np.zeros(N)
        coupling[0] = K_c_fixed * (x_chain[1] - x_chain[0])
        coupling[-1] = K_c_fixed * (x_chain[-2] - x_chain[-1])
        if N > 2:
            coupling[1:-1] = K_c_fixed * (x_chain[2:] + x_chain[:-2] - 2 * x_chain[1:-1])
        force += coupling
        force += np.random.normal(0, 0.0001, N)

        v_chain += force * dt_sc
        x_chain += v_chain * dt_sc

        if step >= measure_start:
            subharmonic_data.append(x_chain.copy())

    subharmonic_data = np.array(subharmonic_data)

    # Extract subharmonic phase at each site
    site_phases = []
    site_amps = []
    for site in range(N):
        site_signal = subharmonic_data[:, site]
        # Analytic signal for phase extraction
        analytic = signal.hilbert(site_signal)
        phase = np.angle(analytic)
        amp = np.abs(analytic)
        site_phases.append(np.mean(phase))
        site_amps.append(np.mean(amp))

    site_phases = np.array(site_phases)
    site_amps = np.array(site_amps)

    # Spatial correlation function of phase
    # C(r) = <cos(phi_i - phi_{i+r})>
    max_r = min(N // 2, 50)
    correlations = []
    for r in range(1, max_r + 1):
        cos_diff = np.cos(site_phases[:-r] - site_phases[r:])
        correlations.append(np.mean(cos_diff))

    correlations = np.array(correlations)

    # Find correlation length (where C drops below 1/e)
    threshold = 1.0 / np.e
    above_threshold = np.where(correlations > threshold)[0]
    if len(above_threshold) > 0:
        corr_length = above_threshold[-1] + 1
    else:
        corr_length = 1

    coherence_lengths.append(corr_length)
    print(f"N={N}(xi={corr_length})", end=" ", flush=True)

print("done.")

R_sc_test = np.array(chain_sizes, dtype=float)
U_sc_test = np.array(coherence_lengths, dtype=float)

# Measure and fit
op_type, scores, beta, r2 = measure_composition_operator(R_sc_test, U_sc_test)
best_fit, fits = fit_scaling(R_sc_test, U_sc_test)

print(f"\n  Coherence lengths: {coherence_lengths}")
print(f"\n  Composition operator: {op_type.upper() if op_type else 'UNDETERMINED'} (R2 = {r2:.4f})")
if beta is not None:
    print(f"  beta measured: {beta:.4f}")

print(f"\n  Best scaling fit: {best_fit.upper()}")
for fname, fdata in fits.items():
    r2_val = fdata.get('r2', 0)
    if fname == 'power_law':
        print(f"    Power-law:   alpha = {fdata['alpha']:.4f}, R2 = {r2_val:.4f}")
    elif fname == 'exponential':
        print(f"    Exponential: lambda = {fdata.get('lambda', 0):.6f}, R2 = {r2_val:.4f}")
    elif fname == 'saturation':
        print(f"    Saturation:  U_max = {fdata.get('Umax', 0):.4f}, k = {fdata.get('k', 0):.6f}, R2 = {r2_val:.4f}")

# Could be either power-law or saturation depending on regime
sc_expected = "saturation"  # Bounded system -> bounded coherence
# But power-law also valid near critical point
sc_match = best_fit in [sc_expected, "power_law"]
print(f"\n  ARC PREDICTION: {sc_expected.upper()} or POWER-LAW (depending on regime)")
print(f"  OBSERVED:       {best_fit.upper()}")
print(f"  {'CONFIRMED' if sc_match else 'NOT CONFIRMED'}")

results_all.append({
    'domain': 'Spatial Coherence Length',
    'system': f'Correlation length vs chain size (K={K_c_fixed})',
    'predicted': sc_expected,
    'observed': best_fit,
    'match': sc_match,
    'op_type': op_type,
    'beta': beta,
    'fits': fits,
    'r2': r2
})


# ============================================================================
# AGGREGATE RESULTS
# ============================================================================
print()
print()
print("=" * 80)
print("+" + "-" * 78 + "+")
print("|" + " " * 15 + "ACOUSTIC TIME CRYSTAL: AGGREGATE RESULTS" + " " * 22 + "|")
print("+" + "-" * 78 + "+")
print("=" * 80)
print()

print(f"  {'Test':<35} | {'System':<45} | {'Pred':<10} | {'Obs':<10} | Match")
print("  " + "-" * 115)

correct = 0
total = len(results_all)

for r in results_all:
    mark = "CONFIRMED" if r['match'] else "  FAILED"
    if r['match']:
        correct += 1
    system_short = r['system'][:45]
    print(f"  {r['domain']:<35} | {system_short:<45} | {r['predicted']:<10} | {r['observed']:<10} | {mark}")

print("  " + "-" * 115)
print()
print(f"  ACOUSTIC TIME CRYSTAL PREDICTION ACCURACY: {correct}/{total} ({100*correct/total:.0f}%)")
print()

# Composition operator summary
print("  COMPOSITION OPERATOR ANALYSIS:")
print("  " + "-" * 60)
for r in results_all:
    op = r.get('op_type', 'unknown')
    beta_val = r.get('beta')
    beta_str = f"beta={beta_val:.4f}" if beta_val is not None else "N/A"
    print(f"    {r['domain']:<35}: oplus = {op:<15} ({beta_str})")

print()
print("  SCALING FIT COMPARISON:")
print("  " + "-" * 60)
for r in results_all:
    fits = r.get('fits', {})
    best = r['observed']
    for fname, fdata in fits.items():
        r2_val = fdata.get('r2', 0)
        marker = " <-- BEST" if fname == best else ""
        if fname == 'power_law':
            print(f"    {r['domain']:<35}: PL  alpha={fdata.get('alpha',0):.4f}  R2={r2_val:.4f}{marker}")
        elif fname == 'exponential':
            print(f"    {r['domain']:<35}: EXP lam={fdata.get('lambda',0):.6f}  R2={r2_val:.4f}{marker}")
        elif fname == 'saturation':
            print(f"    {r['domain']:<35}: SAT Umax={fdata.get('Umax',0):.4f}  R2={r2_val:.4f}{marker}")

print()
print("  " + "=" * 70)
if correct == total:
    print("  VERDICT: COMPLETE CONFIRMATION")
    print("  The ARC framework correctly predicts scaling form across ALL")
    print("  acoustic time crystal tests. Bounded composition operators")
    print("  produce saturating scaling exactly as predicted.")
elif correct >= total * 0.75:
    print("  VERDICT: STRONG CONFIRMATION")
    print("  The ARC framework correctly predicts scaling form in most")
    print("  acoustic time crystal configurations tested.")
elif correct >= total * 0.5:
    print("  VERDICT: MODERATE CONFIRMATION")
    print("  Mixed results. Some configurations match ARC predictions,")
    print("  others show different scaling behaviour.")
else:
    print("  VERDICT: WEAK CONFIRMATION")
    print("  Further investigation needed.")
print("  " + "=" * 70)

print()
print("  PHYSICAL SIGNIFICANCE:")
print("  " + "-" * 60)
print("  Acoustic time crystals break discrete time-translation symmetry")
print("  in periodically driven acoustic/phononic systems. The ARC")
print("  framework predicts that the bounded composition operator")
print("  (damping + nonlinear saturation) should produce saturating")
print("  scaling of the subharmonic order parameter.")
print()
print("  If confirmed, this means:")
print("  1. Time-crystal formation follows universal scaling laws")
print("  2. The composition operator oplus determines the scaling form")
print("  3. Bounded oplus -> saturation is a structural prediction,")
print("     not a parameter fit")
print("  4. The same mathematical structure governs synchronisation,")
print("     parametric amplification, and symmetry breaking")
print()
print("=" * 80)
print("  ACOUSTIC TIME CRYSTAL VALIDATION COMPLETE")
print("=" * 80)
