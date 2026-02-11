"""
================================================================================
THE DEFINITIVE BLIND PREDICTION TEST
================================================================================

This is the test that determines whether the ARC Principle is a discovery.

PROTOCOL (Gold Standard for Scientific Validation):

    1. Choose systems where β and α can be measured INDEPENDENTLY
    2. Measure β from PROCESS data (how gains accumulate)
    3. Predict α = 1/(1-β) BEFORE looking at outcome data
    4. Measure α from OUTCOME data (how performance scales)
    5. Compare prediction to measurement
    6. Calculate statistical significance

CRITICAL REQUIREMENT:
    β must be measured from DIFFERENT data than α.
    Using the same data would be circular.

SYSTEMS TESTED:
    1. Barabási-Albert Networks (preferential attachment → degree distribution)
    2. Gradient Descent with Momentum (velocity accumulation → loss scaling)
    3. Belief Propagation Decoder (message strength → error scaling)
    4. Coupled Oscillator Synchronization (coupling → order parameter)
    5. Evolutionary Algorithm (fitness accumulation → convergence)

If predictions match across multiple independent systems → VALIDATED DISCOVERY
If predictions systematically fail → FALSIFIED

================================================================================
"""

import numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(2026)  # For reproducibility

print("=" * 80)
print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║           THE DEFINITIVE BLIND PREDICTION TEST                              ║")
print("║                                                                              ║")
print("║           Testing: α = 1/(1-β) as a Scientific Law                          ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")
print("=" * 80)
print()

results = []

# ============================================================================
# SYSTEM 1: BARABÁSI-ALBERT NETWORK
# ============================================================================
#
# This is the cleanest test because:
# - β is set by the PHYSICS of preferential attachment
# - α is measured from the EMERGENT degree distribution
# - These are genuinely independent aspects
#
# The theoretical prediction: if P(attach) ∝ k^β, then P(k) ∝ k^(-γ) where γ = 1 + 1/β
# In ARC notation: α_degree = 1 + 1/β = 1 + α_scaling, so α_scaling = 1/β
# Actually, let's be more careful: the degree distribution exponent γ = 3 for linear PA (β=1)
# For sub-linear PA with P ∝ k^β, the exponent is γ = 1 + 1/β
#
# But we need to map this to the ARC framework carefully.
# In BA networks, the "capability" is the degree k, and "recursion" is network age.
# The growth rate of degree for node i is: dk_i/dt ∝ k_i^β (preferential attachment)
# This IS the Bernoulli ODE! So β_PA directly maps to β_ARC.
# The solution gives k_i(t) ∝ t^(1/(1-β)) for a node born at time 0.
# So α_ARC = 1/(1-β) should describe how degree grows with time for individual nodes.

print("━" * 80)
print("  SYSTEM 1: BARABÁSI-ALBERT NETWORK")
print("  Physics: Preferential attachment with varying exponent")
print("━" * 80)
print()

def run_ba_network_test(beta_attachment, n_nodes=5000, m=3, n_trials=10):
    """
    Run BA network with preferential attachment P(k) ∝ k^β

    STEP 1: Generate network with known β (this is the "physics")
    STEP 2: Measure β from attachment statistics (PROCESS measurement)
    STEP 3: Predict α = 1/(1-β)
    STEP 4: Measure α from how early nodes' degrees grow (OUTCOME measurement)
    STEP 5: Compare
    """

    alpha_measurements = []
    beta_measurements = []

    for trial in range(n_trials):
        # Initialize with m+1 fully connected nodes
        degrees = np.ones(m + 1) * m
        birth_times = np.zeros(m + 1)

        # Track degree growth for first 50 nodes (excluding seed)
        tracked_nodes = list(range(m + 1, min(m + 51, n_nodes)))
        degree_history = {i: [] for i in tracked_nodes}
        time_history = []

        # Also track attachment events to measure β
        attachment_events = []  # (target_degree, was_chosen)

        for t in range(m + 1, n_nodes):
            # Calculate attachment probabilities: P(i) ∝ k_i^β
            probs = degrees ** beta_attachment
            probs = probs / probs.sum()

            # Record attachment statistics for β measurement
            # We'll measure: log(P_chosen) vs log(k) to get β

            # Choose m targets
            targets = np.random.choice(len(degrees), size=m, replace=False, p=probs)

            # Record for β measurement (sample of events)
            if t % 100 == 0:  # Sample every 100 steps
                for i, (k, p) in enumerate(zip(degrees, probs)):
                    was_chosen = i in targets
                    attachment_events.append((k, p, was_chosen))

            # Update degrees
            degrees[targets] += 1
            degrees = np.append(degrees, m)
            birth_times = np.append(birth_times, t)

            # Track degree growth
            for node in tracked_nodes:
                if node < len(degrees):
                    degree_history[node].append(degrees[node])
            time_history.append(t)

        # STEP 2: MEASURE β FROM ATTACHMENT STATISTICS
        # β determines how attachment probability scales with degree
        # log(P) = log(c) + β*log(k)
        attachment_events = np.array(attachment_events)
        if len(attachment_events) > 100:
            k_vals = attachment_events[:, 0]
            p_vals = attachment_events[:, 1]
            valid = (k_vals > 0) & (p_vals > 0)
            if np.sum(valid) > 50:
                log_k = np.log(k_vals[valid])
                log_p = np.log(p_vals[valid])
                slope_beta, _, r_beta, _, se_beta = stats.linregress(log_k, log_p)
                beta_measured = slope_beta
                beta_measurements.append(beta_measured)

        # STEP 4: MEASURE α FROM DEGREE GROWTH
        # For a node born at time t_birth, degree k(t) ∝ (t - t_birth)^α
        # where α = 1/(1-β)
        alpha_trials = []
        for node in tracked_nodes[:20]:  # Use first 20 tracked nodes
            if node in degree_history and len(degree_history[node]) > 100:
                k_series = np.array(degree_history[node])
                t_series = np.array(time_history[:len(k_series)]) - node  # Time since birth

                valid = (t_series > 10) & (k_series > m)  # Skip early transient
                if np.sum(valid) > 50:
                    log_t = np.log(t_series[valid])
                    log_k = np.log(k_series[valid])
                    slope_alpha, _, r_alpha, _, _ = stats.linregress(log_t, log_k)
                    if r_alpha**2 > 0.8:  # Only use good fits
                        alpha_trials.append(slope_alpha)

        if len(alpha_trials) > 5:
            alpha_measurements.append(np.median(alpha_trials))

    return np.array(beta_measurements), np.array(alpha_measurements)

print("  Running blind prediction test...")
print()

# Test with β = 0.7 (sub-linear preferential attachment)
beta_true = 0.7
beta_measured_arr, alpha_measured_arr = run_ba_network_test(beta_true, n_nodes=3000, n_trials=15)

if len(beta_measured_arr) > 3 and len(alpha_measured_arr) > 3:
    beta_mean = np.mean(beta_measured_arr)
    beta_std = np.std(beta_measured_arr)
    alpha_mean = np.mean(alpha_measured_arr)
    alpha_std = np.std(alpha_measured_arr)

    # STEP 3: PREDICT α from measured β
    alpha_predicted = 1.0 / (1.0 - beta_mean)

    # STEP 5: COMPARE
    prediction_error = abs(alpha_predicted - alpha_mean)
    relative_error = prediction_error / alpha_mean * 100

    # Statistical test: is prediction within uncertainty?
    combined_uncertainty = np.sqrt(alpha_std**2 + (beta_std / (1-beta_mean)**2)**2)
    z_score = prediction_error / combined_uncertainty if combined_uncertainty > 0 else 0

    print(f"  MEASUREMENT RESULTS (β_true = {beta_true}):")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"  β measured from attachment statistics: {beta_mean:.4f} ± {beta_std:.4f}")
    print(f"  α predicted = 1/(1-β):                 {alpha_predicted:.4f}")
    print(f"  α measured from degree growth:         {alpha_mean:.4f} ± {alpha_std:.4f}")
    print(f"  ")
    print(f"  Prediction error:                      {prediction_error:.4f} ({relative_error:.1f}%)")
    print(f"  Z-score:                               {z_score:.2f}")
    print(f"  Prediction within 2σ:                  {'✓ YES' if z_score < 2 else '✗ NO'}")
    print()

    results.append({
        'system': 'Barabási-Albert Network',
        'beta_measured': beta_mean,
        'beta_std': beta_std,
        'alpha_predicted': alpha_predicted,
        'alpha_measured': alpha_mean,
        'alpha_std': alpha_std,
        'error_pct': relative_error,
        'z_score': z_score,
        'validated': z_score < 2
    })
else:
    print("  Insufficient data for statistical analysis")
    print()


# ============================================================================
# SYSTEM 2: GRADIENT DESCENT WITH MOMENTUM
# ============================================================================
#
# Physics: The momentum accumulation follows dv/dt = -η∇L + μv
# The "capability" is inverse loss: U = 1/L
# β can be measured from how velocity magnitude grows relative to gradient
# α can be measured from how loss decreases with iteration

print()
print("━" * 80)
print("  SYSTEM 2: GRADIENT DESCENT WITH MOMENTUM")
print("  Physics: Momentum accumulation on Rosenbrock function")
print("━" * 80)
print()

def run_gradient_descent_test(n_trials=20):
    """
    STEP 1: Run gradient descent with momentum
    STEP 2: Measure β from how momentum builds (PROCESS)
    STEP 3: Predict α
    STEP 4: Measure α from loss curve (OUTCOME)
    """

    # Rosenbrock function
    a, b = 1.0, 100.0

    def loss(x, y):
        return (a - x)**2 + b * (y - x**2)**2

    def grad(x, y):
        dLdx = -2*(a - x) - 4*b*x*(y - x**2)
        dLdy = 2*b*(y - x**2)
        return np.array([dLdx, dLdy])

    beta_measurements = []
    alpha_measurements = []

    for trial in range(n_trials):
        # Random starting point
        x, y = np.random.uniform(-2, 2), np.random.uniform(-1, 3)
        lr = 0.0001
        momentum = 0.9
        v = np.array([0.0, 0.0])

        loss_history = []
        velocity_history = []
        grad_history = []

        for t in range(2000):
            L = loss(x, y)
            g = grad(x, y)

            loss_history.append(L)
            velocity_history.append(np.linalg.norm(v))
            grad_history.append(np.linalg.norm(g))

            v = momentum * v - lr * g
            x += v[0]
            y += v[1]

            if L < 1e-10:
                break

        loss_history = np.array(loss_history)
        velocity_history = np.array(velocity_history)
        grad_history = np.array(grad_history)

        # STEP 2: Measure β from momentum dynamics
        # In the recursive framework: dv/dt ≈ v^β where higher β means faster accumulation
        # We measure: log(dv) vs log(v)
        if len(velocity_history) > 200:
            dv = np.diff(velocity_history)
            v_mid = (velocity_history[:-1] + velocity_history[1:]) / 2

            # Use middle section where dynamics are clean
            start, end = 100, min(800, len(dv) - 100)
            valid = (v_mid[start:end] > 1e-6) & (np.abs(dv[start:end]) > 1e-10)

            if np.sum(valid) > 50:
                log_v = np.log(v_mid[start:end][valid])
                log_dv = np.log(np.abs(dv[start:end][valid]))

                slope, _, r, _, _ = stats.linregress(log_v, log_dv)
                if r**2 > 0.3:  # Reasonable fit
                    # Clamp to valid range
                    beta_eff = np.clip(slope, 0.1, 0.95)
                    beta_measurements.append(beta_eff)

        # STEP 4: Measure α from loss curve
        # U = 1/L should scale as t^α
        if len(loss_history) > 200:
            t_range = np.arange(100, min(1000, len(loss_history)))
            U = 1.0 / (loss_history[t_range] + 1e-15)

            valid = (U > 1) & (U < 1e10)
            if np.sum(valid) > 50:
                log_t = np.log(t_range[valid])
                log_U = np.log(U[valid])

                slope, _, r, _, _ = stats.linregress(log_t, log_U)
                if r**2 > 0.5 and slope > 0:
                    alpha_measurements.append(slope)

    return np.array(beta_measurements), np.array(alpha_measurements)

print("  Running blind prediction test...")
print()

beta_gd, alpha_gd = run_gradient_descent_test(n_trials=30)

if len(beta_gd) > 5 and len(alpha_gd) > 5:
    beta_mean = np.mean(beta_gd)
    beta_std = np.std(beta_gd)
    alpha_mean = np.mean(alpha_gd)
    alpha_std = np.std(alpha_gd)

    alpha_predicted = 1.0 / (1.0 - beta_mean) if beta_mean < 0.99 else 100

    prediction_error = abs(alpha_predicted - alpha_mean)
    relative_error = prediction_error / alpha_mean * 100 if alpha_mean > 0 else 0

    combined_uncertainty = np.sqrt(alpha_std**2 + (beta_std / max((1-beta_mean)**2, 0.01))**2)
    z_score = prediction_error / combined_uncertainty if combined_uncertainty > 0 else 0

    print(f"  MEASUREMENT RESULTS:")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"  β measured from momentum dynamics:     {beta_mean:.4f} ± {beta_std:.4f}")
    print(f"  α predicted = 1/(1-β):                 {alpha_predicted:.4f}")
    print(f"  α measured from loss scaling:          {alpha_mean:.4f} ± {alpha_std:.4f}")
    print(f"  ")
    print(f"  Prediction error:                      {prediction_error:.4f} ({relative_error:.1f}%)")
    print(f"  Z-score:                               {z_score:.2f}")
    print(f"  Prediction within 2σ:                  {'✓ YES' if z_score < 2 else '✗ NO'}")
    print()

    results.append({
        'system': 'Gradient Descent + Momentum',
        'beta_measured': beta_mean,
        'beta_std': beta_std,
        'alpha_predicted': alpha_predicted,
        'alpha_measured': alpha_mean,
        'alpha_std': alpha_std,
        'error_pct': relative_error,
        'z_score': z_score,
        'validated': z_score < 2
    })


# ============================================================================
# SYSTEM 3: KURAMOTO OSCILLATOR SYNCHRONIZATION
# ============================================================================
#
# Physics: Coupled oscillators synchronize through collective coupling
# β measures how coupling strength scales with current synchronization
# α measures how order parameter grows with time

print()
print("━" * 80)
print("  SYSTEM 3: KURAMOTO OSCILLATOR SYNCHRONIZATION")
print("  Physics: Coupled phase oscillators")
print("━" * 80)
print()

def run_kuramoto_test(n_trials=20):
    """
    STEP 1: Run Kuramoto model with varying coupling
    STEP 2: Measure β from how effective coupling scales with order parameter
    STEP 3: Predict α
    STEP 4: Measure α from order parameter growth
    """

    beta_measurements = []
    alpha_measurements = []

    for trial in range(n_trials):
        N = 100
        # Natural frequencies from Lorentzian distribution
        omega = np.random.standard_cauchy(N) * 0.3
        omega = np.clip(omega, -2, 2)

        K = 2.5  # Coupling strength (above critical)
        dt = 0.02
        theta = np.random.uniform(0, 2*np.pi, N)

        order_history = []
        dorder_history = []

        for t in range(1500):
            # Calculate order parameter
            z = np.mean(np.exp(1j * theta))
            r = np.abs(z)
            order_history.append(r)

            if t > 0:
                dorder_history.append(r - order_history[-2])

            # Kuramoto dynamics
            psi = np.angle(z)
            dtheta = omega + K * r * np.sin(psi - theta)
            theta = (theta + dtheta * dt) % (2 * np.pi)

        order_history = np.array(order_history)
        dorder_history = np.array(dorder_history)

        # STEP 2: Measure β from dr/dt vs r relationship
        # The Kuramoto model has dr/dt ≈ f(r) where the shape determines β
        if len(dorder_history) > 200:
            r_mid = order_history[1:]
            dr = dorder_history

            # Focus on growth phase (not saturated)
            valid = (r_mid > 0.1) & (r_mid < 0.8) & (dr > 0)

            if np.sum(valid) > 30:
                log_r = np.log(r_mid[valid])
                log_dr = np.log(dr[valid] + 1e-10)

                slope, _, r_corr, _, _ = stats.linregress(log_r, log_dr)
                if r_corr**2 > 0.2:
                    beta_eff = np.clip(slope, 0.1, 0.95)
                    beta_measurements.append(beta_eff)

        # STEP 4: Measure α from order parameter growth
        # During growth phase, r(t) ∝ t^α
        growth_start = np.argmax(order_history > 0.1)
        growth_end = np.argmax(order_history > 0.8)

        if growth_end > growth_start + 50:
            t_range = np.arange(growth_start, growth_end)
            r_range = order_history[t_range]

            valid = r_range > 0.05
            if np.sum(valid) > 30:
                log_t = np.log(t_range[valid] - growth_start + 1)
                log_r = np.log(r_range[valid])

                slope, _, r_corr, _, _ = stats.linregress(log_t, log_r)
                if r_corr**2 > 0.5 and slope > 0:
                    alpha_measurements.append(slope)

    return np.array(beta_measurements), np.array(alpha_measurements)

print("  Running blind prediction test...")
print()

beta_kur, alpha_kur = run_kuramoto_test(n_trials=30)

if len(beta_kur) > 5 and len(alpha_kur) > 5:
    beta_mean = np.mean(beta_kur)
    beta_std = np.std(beta_kur)
    alpha_mean = np.mean(alpha_kur)
    alpha_std = np.std(alpha_kur)

    alpha_predicted = 1.0 / (1.0 - beta_mean) if beta_mean < 0.99 else 100

    prediction_error = abs(alpha_predicted - alpha_mean)
    relative_error = prediction_error / alpha_mean * 100 if alpha_mean > 0 else 0

    combined_uncertainty = np.sqrt(alpha_std**2 + (beta_std / max((1-beta_mean)**2, 0.01))**2)
    z_score = prediction_error / combined_uncertainty if combined_uncertainty > 0 else 0

    print(f"  MEASUREMENT RESULTS:")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"  β measured from coupling dynamics:     {beta_mean:.4f} ± {beta_std:.4f}")
    print(f"  α predicted = 1/(1-β):                 {alpha_predicted:.4f}")
    print(f"  α measured from order growth:          {alpha_mean:.4f} ± {alpha_std:.4f}")
    print(f"  ")
    print(f"  Prediction error:                      {prediction_error:.4f} ({relative_error:.1f}%)")
    print(f"  Z-score:                               {z_score:.2f}")
    print(f"  Prediction within 2σ:                  {'✓ YES' if z_score < 2 else '✗ NO'}")
    print()

    results.append({
        'system': 'Kuramoto Oscillators',
        'beta_measured': beta_mean,
        'beta_std': beta_std,
        'alpha_predicted': alpha_predicted,
        'alpha_measured': alpha_mean,
        'alpha_std': alpha_std,
        'error_pct': relative_error,
        'z_score': z_score,
        'validated': z_score < 2
    })


# ============================================================================
# SYSTEM 4: LDPC BELIEF PROPAGATION DECODER
# ============================================================================
#
# Physics: Message passing on factor graph
# β measures how message reliability builds
# α measures how bit error rate decreases

print()
print("━" * 80)
print("  SYSTEM 4: LDPC BELIEF PROPAGATION DECODER")
print("  Physics: Message passing on sparse graph")
print("━" * 80)
print()

def run_ldpc_test(n_trials=20):
    """
    Simplified LDPC-like decoder simulation
    """

    beta_measurements = []
    alpha_measurements = []

    for trial in range(n_trials):
        # Simple regular LDPC-like structure
        n_bits = 100
        n_checks = 50

        # Initial LLRs (log-likelihood ratios) - noisy channel
        snr = 2.0
        transmitted = np.random.randint(0, 2, n_bits)
        noise = np.random.randn(n_bits) / snr
        received_llr = (2 * transmitted - 1) * 2 * snr + noise

        # Random sparse parity check connections
        H = np.zeros((n_checks, n_bits))
        for c in range(n_checks):
            connected = np.random.choice(n_bits, size=6, replace=False)
            H[c, connected] = 1

        # Belief propagation
        llr = received_llr.copy()
        reliability_history = []
        error_rate_history = []

        for iteration in range(50):
            # Record metrics
            decisions = (llr > 0).astype(int)
            error_rate = np.mean(decisions != transmitted)
            reliability = np.mean(np.abs(llr))

            reliability_history.append(reliability)
            error_rate_history.append(error_rate)

            # Check-to-variable messages (simplified)
            new_llr = received_llr.copy()
            for c in range(n_checks):
                connected = np.where(H[c] > 0)[0]
                for v in connected:
                    others = [o for o in connected if o != v]
                    if len(others) > 0:
                        # Product of tanh(llr/2) for others
                        prod = np.prod(np.tanh(llr[others] / 2))
                        msg = 2 * np.arctanh(np.clip(prod, -0.999, 0.999))
                        new_llr[v] += 0.3 * msg

            llr = new_llr

        reliability_history = np.array(reliability_history)
        error_rate_history = np.array(error_rate_history)

        # STEP 2: Measure β from reliability growth
        if len(reliability_history) > 10:
            dr = np.diff(reliability_history)
            r_mid = (reliability_history[:-1] + reliability_history[1:]) / 2

            valid = (r_mid > 0.5) & (dr > 0)
            if np.sum(valid) > 5:
                log_r = np.log(r_mid[valid])
                log_dr = np.log(dr[valid] + 1e-10)

                slope, _, r_corr, _, _ = stats.linregress(log_r, log_dr)
                if r_corr**2 > 0.1:
                    beta_eff = np.clip(slope, 0.1, 0.95)
                    beta_measurements.append(beta_eff)

        # STEP 4: Measure α from error rate decay
        # Error rate ∝ t^(-α) means U = 1/error ∝ t^α
        valid_err = error_rate_history > 1e-6
        if np.sum(valid_err) > 10:
            t_range = np.arange(len(error_rate_history))[valid_err]
            U = 1.0 / error_rate_history[valid_err]

            valid = (t_range > 2) & (U < 1e6)
            if np.sum(valid) > 5:
                log_t = np.log(t_range[valid] + 1)
                log_U = np.log(U[valid])

                slope, _, r_corr, _, _ = stats.linregress(log_t, log_U)
                if r_corr**2 > 0.3 and slope > 0:
                    alpha_measurements.append(slope)

    return np.array(beta_measurements), np.array(alpha_measurements)

print("  Running blind prediction test...")
print()

beta_ldpc, alpha_ldpc = run_ldpc_test(n_trials=30)

if len(beta_ldpc) > 5 and len(alpha_ldpc) > 5:
    beta_mean = np.mean(beta_ldpc)
    beta_std = np.std(beta_ldpc)
    alpha_mean = np.mean(alpha_ldpc)
    alpha_std = np.std(alpha_ldpc)

    alpha_predicted = 1.0 / (1.0 - beta_mean) if beta_mean < 0.99 else 100

    prediction_error = abs(alpha_predicted - alpha_mean)
    relative_error = prediction_error / alpha_mean * 100 if alpha_mean > 0 else 0

    combined_uncertainty = np.sqrt(alpha_std**2 + (beta_std / max((1-beta_mean)**2, 0.01))**2)
    z_score = prediction_error / combined_uncertainty if combined_uncertainty > 0 else 0

    print(f"  MEASUREMENT RESULTS:")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"  β measured from reliability growth:    {beta_mean:.4f} ± {beta_std:.4f}")
    print(f"  α predicted = 1/(1-β):                 {alpha_predicted:.4f}")
    print(f"  α measured from error decay:           {alpha_mean:.4f} ± {alpha_std:.4f}")
    print(f"  ")
    print(f"  Prediction error:                      {prediction_error:.4f} ({relative_error:.1f}%)")
    print(f"  Z-score:                               {z_score:.2f}")
    print(f"  Prediction within 2σ:                  {'✓ YES' if z_score < 2 else '✗ NO'}")
    print()

    results.append({
        'system': 'LDPC Belief Propagation',
        'beta_measured': beta_mean,
        'beta_std': beta_std,
        'alpha_predicted': alpha_predicted,
        'alpha_measured': alpha_mean,
        'alpha_std': alpha_std,
        'error_pct': relative_error,
        'z_score': z_score,
        'validated': z_score < 2
    })


# ============================================================================
# SYSTEM 5: EVOLUTIONARY ALGORITHM
# ============================================================================
#
# Physics: Fitness-proportional selection
# β measures how fitness accumulates through selection
# α measures how best fitness grows over generations

print()
print("━" * 80)
print("  SYSTEM 5: EVOLUTIONARY ALGORITHM")
print("  Physics: Fitness-proportional selection on NK landscape")
print("━" * 80)
print()

def run_evolution_test(n_trials=20):
    """
    Evolutionary algorithm on rugged fitness landscape
    """

    beta_measurements = []
    alpha_measurements = []

    for trial in range(n_trials):
        n_genes = 30
        pop_size = 100
        n_generations = 200

        # Random NK-like fitness function
        fitness_table = np.random.rand(n_genes, 4)  # 2-bit interactions

        def fitness(genome):
            f = 0
            for i in range(n_genes):
                j = (i + 1) % n_genes
                idx = genome[i] * 2 + genome[j]
                f += fitness_table[i, idx]
            return f / n_genes

        # Initialize population
        population = np.random.randint(0, 2, (pop_size, n_genes))
        fitnesses = np.array([fitness(ind) for ind in population])

        fitness_history = []
        dfitness_history = []

        for gen in range(n_generations):
            best_fitness = np.max(fitnesses)
            mean_fitness = np.mean(fitnesses)
            fitness_history.append(best_fitness)

            if gen > 0:
                dfitness_history.append(best_fitness - fitness_history[-2])

            # Selection (fitness-proportional)
            probs = fitnesses / fitnesses.sum()
            parents_idx = np.random.choice(pop_size, size=pop_size, p=probs)
            parents = population[parents_idx]

            # Crossover and mutation
            new_pop = []
            for i in range(0, pop_size, 2):
                p1, p2 = parents[i], parents[min(i+1, pop_size-1)]
                crossover_point = np.random.randint(1, n_genes)
                c1 = np.concatenate([p1[:crossover_point], p2[crossover_point:]])
                c2 = np.concatenate([p2[:crossover_point], p1[crossover_point:]])

                # Mutation
                for c in [c1, c2]:
                    if np.random.rand() < 0.1:
                        mut_point = np.random.randint(n_genes)
                        c[mut_point] = 1 - c[mut_point]

                new_pop.extend([c1, c2])

            population = np.array(new_pop[:pop_size])
            fitnesses = np.array([fitness(ind) for ind in population])

        fitness_history = np.array(fitness_history)
        dfitness_history = np.array(dfitness_history)

        # STEP 2: Measure β from fitness dynamics
        # How does df/dt scale with f?
        if len(dfitness_history) > 50:
            f_mid = fitness_history[1:]
            df = dfitness_history

            valid = (f_mid > 0.3) & (df > 0)
            if np.sum(valid) > 20:
                log_f = np.log(f_mid[valid])
                log_df = np.log(df[valid] + 1e-10)

                slope, _, r_corr, _, _ = stats.linregress(log_f, log_df)
                if r_corr**2 > 0.1:
                    beta_eff = np.clip(slope, 0.1, 0.95)
                    beta_measurements.append(beta_eff)

        # STEP 4: Measure α from fitness growth
        # Best fitness as function of generation
        valid_range = (fitness_history > 0.3) & (fitness_history < 0.95)
        if np.sum(valid_range) > 20:
            t_range = np.arange(len(fitness_history))[valid_range]
            f_range = fitness_history[valid_range]

            log_t = np.log(t_range + 1)
            log_f = np.log(f_range)

            slope, _, r_corr, _, _ = stats.linregress(log_t, log_f)
            if r_corr**2 > 0.3 and slope > 0:
                alpha_measurements.append(slope)

    return np.array(beta_measurements), np.array(alpha_measurements)

print("  Running blind prediction test...")
print()

beta_evo, alpha_evo = run_evolution_test(n_trials=30)

if len(beta_evo) > 5 and len(alpha_evo) > 5:
    beta_mean = np.mean(beta_evo)
    beta_std = np.std(beta_evo)
    alpha_mean = np.mean(alpha_evo)
    alpha_std = np.std(alpha_evo)

    alpha_predicted = 1.0 / (1.0 - beta_mean) if beta_mean < 0.99 else 100

    prediction_error = abs(alpha_predicted - alpha_mean)
    relative_error = prediction_error / alpha_mean * 100 if alpha_mean > 0 else 0

    combined_uncertainty = np.sqrt(alpha_std**2 + (beta_std / max((1-beta_mean)**2, 0.01))**2)
    z_score = prediction_error / combined_uncertainty if combined_uncertainty > 0 else 0

    print(f"  MEASUREMENT RESULTS:")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"  β measured from selection dynamics:    {beta_mean:.4f} ± {beta_std:.4f}")
    print(f"  α predicted = 1/(1-β):                 {alpha_predicted:.4f}")
    print(f"  α measured from fitness growth:        {alpha_mean:.4f} ± {alpha_std:.4f}")
    print(f"  ")
    print(f"  Prediction error:                      {prediction_error:.4f} ({relative_error:.1f}%)")
    print(f"  Z-score:                               {z_score:.2f}")
    print(f"  Prediction within 2σ:                  {'✓ YES' if z_score < 2 else '✗ NO'}")
    print()

    results.append({
        'system': 'Evolutionary Algorithm',
        'beta_measured': beta_mean,
        'beta_std': beta_std,
        'alpha_predicted': alpha_predicted,
        'alpha_measured': alpha_mean,
        'alpha_std': alpha_std,
        'error_pct': relative_error,
        'z_score': z_score,
        'validated': z_score < 2
    })


# ============================================================================
# AGGREGATE ANALYSIS
# ============================================================================

print()
print("=" * 80)
print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║                    AGGREGATE RESULTS                                         ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")
print("=" * 80)
print()

if len(results) > 0:
    print(f"  {'System':<30s} │ {'β_meas':>8s} {'α_pred':>8s} {'α_meas':>8s} │ {'Error%':>8s} {'Z':>6s} │ {'Valid':>6s}")
    print("  " + "─" * 90)

    validated_count = 0
    for r in results:
        valid_str = '✓' if r['validated'] else '✗'
        print(f"  {r['system']:<30s} │ {r['beta_measured']:8.4f} {r['alpha_predicted']:8.4f} {r['alpha_measured']:8.4f} │ {r['error_pct']:8.1f} {r['z_score']:6.2f} │ {valid_str:>6s}")
        if r['validated']:
            validated_count += 1

    print("  " + "─" * 90)
    print()

    # Statistical summary
    alpha_pred_all = np.array([r['alpha_predicted'] for r in results])
    alpha_meas_all = np.array([r['alpha_measured'] for r in results])

    correlation = np.corrcoef(alpha_pred_all, alpha_meas_all)[0, 1]
    slope, intercept, r_val, p_val, se = stats.linregress(alpha_pred_all, alpha_meas_all)

    print(f"  AGGREGATE STATISTICS:")
    print(f"  ────────────────────────────────────────────────────────────────")
    print(f"  Systems tested:                        {len(results)}")
    print(f"  Predictions validated (Z < 2):         {validated_count}/{len(results)} ({100*validated_count/len(results):.0f}%)")
    print(f"  ")
    print(f"  Correlation (α_predicted vs α_measured): {correlation:.4f}")
    print(f"  Linear fit slope:                      {slope:.4f} (perfect = 1.0)")
    print(f"  Linear fit R²:                         {r_val**2:.4f}")
    print(f"  p-value:                               {p_val:.2e}")
    print()

    # The verdict
    print("  ╔═══════════════════════════════════════════════════════════════════════╗")
    if validated_count >= len(results) * 0.6 and correlation > 0.7:
        print("  ║  VERDICT: PREDICTIONS VALIDATED                                      ║")
        print("  ║                                                                       ║")
        print("  ║  The relationship α = 1/(1-β) successfully predicted scaling         ║")
        print("  ║  exponents from independently measured coupling parameters.           ║")
        print("  ║                                                                       ║")
        print("  ║  This constitutes BLIND PREDICTION across multiple systems.          ║")
        status = "VALIDATED"
    elif validated_count >= len(results) * 0.4:
        print("  ║  VERDICT: PARTIAL SUPPORT                                            ║")
        print("  ║                                                                       ║")
        print("  ║  Some predictions validated, others diverged.                        ║")
        print("  ║  The framework may apply in some domains but not universally.        ║")
        status = "PARTIAL"
    else:
        print("  ║  VERDICT: PREDICTIONS FAILED                                         ║")
        print("  ║                                                                       ║")
        print("  ║  Measured α values do not match predictions from measured β.         ║")
        print("  ║  The relationship α = 1/(1-β) is not supported by this test.         ║")
        status = "FAILED"
    print("  ╚═══════════════════════════════════════════════════════════════════════╝")
    print()


# ============================================================================
# VISUALIZATION
# ============================================================================

if len(results) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: α_predicted vs α_measured
    ax1 = axes[0]
    alpha_pred = [r['alpha_predicted'] for r in results]
    alpha_meas = [r['alpha_measured'] for r in results]
    alpha_err = [r['alpha_std'] for r in results]
    colors = ['green' if r['validated'] else 'red' for r in results]

    ax1.errorbar(alpha_pred, alpha_meas, yerr=alpha_err, fmt='none', ecolor='grey', alpha=0.5)
    ax1.scatter(alpha_pred, alpha_meas, c=colors, s=150, edgecolors='black', zorder=3)

    # Perfect prediction line
    lim = max(max(alpha_pred), max(alpha_meas)) * 1.1
    ax1.plot([0, lim], [0, lim], 'k--', linewidth=2, label='Perfect prediction')

    # Fit line
    ax1.plot([0, lim], [intercept, intercept + slope * lim], 'b-', alpha=0.5,
             label=f'Fit: R²={r_val**2:.3f}')

    for i, r in enumerate(results):
        ax1.annotate(r['system'].split()[0], (alpha_pred[i], alpha_meas[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax1.set_xlabel('α predicted = 1/(1-β)', fontsize=12)
    ax1.set_ylabel('α measured', fontsize=12)
    ax1.set_title('BLIND PREDICTION TEST\nα_predicted vs α_measured', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.set_xlim(0, lim)
    ax1.set_ylim(0, lim)

    # Panel 2: Z-scores
    ax2 = axes[1]
    systems = [r['system'].split()[0] for r in results]
    z_scores = [r['z_score'] for r in results]
    colors = ['green' if z < 2 else 'red' for z in z_scores]

    bars = ax2.bar(systems, z_scores, color=colors, edgecolor='black')
    ax2.axhline(y=2, color='red', linestyle='--', linewidth=2, label='2σ threshold')
    ax2.set_ylabel('Z-score', fontsize=12)
    ax2.set_title('PREDICTION ACCURACY\n(Z < 2 = validated)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)

    # Panel 3: Summary
    ax3 = axes[2]
    ax3.axis('off')

    summary_text = f"""
    BLIND PREDICTION TEST RESULTS
    ══════════════════════════════════════

    Systems tested:     {len(results)}
    Validated (Z < 2):  {validated_count}/{len(results)} ({100*validated_count/len(results):.0f}%)

    Correlation:        {correlation:.4f}
    R² (pred vs meas):  {r_val**2:.4f}
    p-value:            {p_val:.2e}

    ══════════════════════════════════════

    VERDICT: {status}

    The relationship α = 1/(1-β) was tested
    by measuring β from PROCESS data and
    α from OUTCOME data independently.

    {"Predictions matched measurements." if status == "VALIDATED" else "Results are mixed." if status == "PARTIAL" else "Predictions did not match."}
    """

    ax3.text(0.1, 0.9, summary_text, fontsize=11, fontfamily='monospace',
             verticalalignment='top', transform=ax3.transAxes,
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', edgecolor='black'))

    plt.tight_layout()
    plt.savefig('/Users/michaeleastwood/Downloads/BLIND_PREDICTION_RESULTS.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Figure saved: /Users/michaeleastwood/Downloads/BLIND_PREDICTION_RESULTS.png")


print()
print("=" * 80)
print("  BLIND PREDICTION TEST COMPLETE")
print("=" * 80)
