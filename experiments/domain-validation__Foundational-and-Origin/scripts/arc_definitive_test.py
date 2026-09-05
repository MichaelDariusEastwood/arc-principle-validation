"""
============================================================================
THE ARC FRAMEWORK: DEFINITIVE CROSS-DOMAIN BLIND PREDICTION TEST
============================================================================

THE TEST:
  For each of 8 recursive systems spanning 5 domains:
    1. Run the system at multiple recursive depths
    2. Measure ONLY the composition operator ⊕ (marginal gain vs accumulated)
    3. From ⊕ alone, PREDICT the scaling function (power-law/exp/saturation)
    4. From ⊕ alone, PREDICT α or λ
    5. INDEPENDENTLY fit the actual U vs R curve
    6. Compare blind prediction against observed reality

  If the composition operator correctly predicts the scaling function
  BEFORE the scaling curve is fitted, the framework's central claim
  (Theorem 1) is validated empirically.

  This is the test specified in Foundational Paper §5.1:
  "Measure only its composition operator ⊕. The measured ⊕ will predict
   whether the system follows power-law, exponential, or saturating
   scaling BEFORE the full U vs R curve is measured."

SYSTEMS TESTED:
  1. Gradient descent with momentum (optimisation)
  2. PageRank iterative computation (network science)
  3. Evolutionary algorithm (biology/computation)
  4. Iterative image denoising / deblurring (signal processing)
  5. Newton's method for root finding (numerical analysis)
  6. Simulated annealing (statistical physics)
  7. Coupled oscillator synchronisation (physics)
  8. Error-correcting code with iterative decoding (information theory)

============================================================================
"""

import numpy as np
from scipy import optimize, stats
from scipy.signal import convolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)  # Reproducibility

# ============================================================================
# CORE ARC MEASUREMENT APPARATUS
# These functions know NOTHING about the specific systems.
# They take raw (R, U) data and extract ⊕, predict scaling, measure α.
# ============================================================================

class ARCMeasurementApparatus:
    """
    Implements the measurement protocol from Appendix B of the
    Foundational Paper. Takes raw capability-vs-depth data and:
      1. Measures the composition operator
      2. Classifies it (multiplicative/additive/bounded)
      3. Predicts scaling function and exponent
      4. Independently fits the actual curve
      5. Compares prediction vs observation
    """

    def __init__(self, name):
        self.name = name
        self.results = {}

    def measure_composition_operator(self, R, U):
        """
        Step 1: Measure ⊕ by examining how marginal gains relate
        to accumulated capability.

        Plot ΔU(r) vs U(r). The functional form reveals ⊕:
          - ΔU ∝ U^β  →  multiplicative (power-law scaling)
          - ΔU ∝ const →  additive (exponential scaling)
          - ΔU ∝ (U_max - U) →  bounded (saturation)
        """
        # Compute marginal gains
        delta_U = np.diff(U)
        U_accum = U[:-1]

        # Remove near-zero or negative gains (noise)
        valid = (delta_U > 1e-15) & (U_accum > 1e-15)
        if np.sum(valid) < 4:
            # Try with absolute values for decreasing-error systems
            delta_U_abs = np.abs(np.diff(U))
            valid = (delta_U_abs > 1e-15) & (U_accum > 1e-15)
            if np.sum(valid) < 4:
                return None
            delta_U = delta_U_abs

        dU = delta_U[valid]
        Ua = U_accum[valid]

        self.results['marginal_gains'] = dU
        self.results['accumulated'] = Ua

        return dU, Ua

    def classify_operator(self, dU, Ua):
        """
        Step 2: Classify ⊕ by fitting three models to (Ua, dU).

        Model A (Multiplicative): log(dU) = β·log(Ua) + const
        Model B (Additive/Constant): dU = const (slope ≈ 0 in log-log)
        Model C (Bounded): dU = k·(U_max - Ua)
        """
        log_dU = np.log(dU)
        log_Ua = np.log(Ua)

        residuals = {}

        # Model A: Power-law marginal gains → multiplicative ⊕
        try:
            slope_A, intercept_A, r_A, p_A, se_A = stats.linregress(log_Ua, log_dU)
            pred_A = intercept_A + slope_A * log_Ua
            res_A = np.mean((log_dU - pred_A)**2)
            residuals['multiplicative'] = res_A
            self.results['beta_measured'] = slope_A
            self.results['r_squared_mult'] = r_A**2
        except:
            residuals['multiplicative'] = np.inf
            self.results['beta_measured'] = np.nan

        # Model B: Constant marginal gains → additive ⊕
        try:
            mean_dU = np.mean(log_dU)
            res_B = np.mean((log_dU - mean_dU)**2)
            residuals['additive'] = res_B
        except:
            residuals['additive'] = np.inf

        # Model C: Linear decay in marginal gains → bounded ⊕
        try:
            # dU should decrease linearly with Ua if bounded
            slope_C, intercept_C, r_C, p_C, se_C = stats.linregress(Ua, dU)
            pred_C = intercept_C + slope_C * Ua
            # Use relative residuals for fair comparison
            res_C_raw = np.mean((dU - pred_C)**2)
            # Convert to log-space for comparable residuals
            pred_C_log = np.log(np.maximum(pred_C, 1e-15))
            res_C = np.mean((log_dU - pred_C_log)**2) if np.all(pred_C > 0) else np.inf
            residuals['bounded'] = res_C
            self.results['slope_bounded'] = slope_C

            if slope_C < 0 and intercept_C > 0:
                self.results['U_max_estimated'] = -intercept_C / slope_C
        except:
            residuals['bounded'] = np.inf

        # Classify
        best = min(residuals, key=residuals.get)
        self.results['operator_class'] = best
        self.results['classification_residuals'] = residuals
        self.results['classification_confidence'] = (
            sorted(residuals.values())[1] / max(residuals[best], 1e-15)
            if len(residuals) > 1 else 1.0
        )

        return best

    def predict_scaling(self):
        """
        Step 3: From the classified ⊕, PREDICT the scaling function
        and exponent WITHOUT looking at the U vs R curve.

        Multiplicative ⊕ with β → predicts U ∝ R^α where α = 1/(1-β)
        Additive ⊕ → predicts U ∝ e^(λR)
        Bounded ⊕ → predicts U → U_max (saturation)
        """
        op_class = self.results['operator_class']

        if op_class == 'multiplicative':
            beta = self.results['beta_measured']
            if beta < 1.0 and beta > -10:
                alpha_predicted = 1.0 / (1.0 - beta) if beta < 0.99 else 100
            else:
                alpha_predicted = np.nan
            self.results['predicted_form'] = 'power_law'
            self.results['alpha_predicted'] = alpha_predicted
            self.results['beta_for_prediction'] = beta

        elif op_class == 'additive':
            self.results['predicted_form'] = 'exponential'
            self.results['alpha_predicted'] = np.nan  # λ instead

        elif op_class == 'bounded':
            self.results['predicted_form'] = 'saturation'
            self.results['alpha_predicted'] = np.nan

        return self.results['predicted_form']

    def fit_actual_scaling(self, R, U):
        """
        Step 4: INDEPENDENTLY fit the actual U vs R curve to all three
        functional forms. Select best fit. This is the "ground truth"
        against which the blind prediction is compared.
        """
        R = np.array(R, dtype=float)
        U = np.array(U, dtype=float)

        # Normalise: I = U(R=R_min)
        I_base = U[0]
        U_norm = U / max(I_base, 1e-15)
        R_norm = R / R[0] if R[0] > 0 else R

        fit_results = {}

        # Fit A: Power law  U/I = (R/R0)^α
        try:
            valid = (R_norm > 0) & (U_norm > 0)
            log_R = np.log(R_norm[valid])
            log_U = np.log(U_norm[valid])
            # Remove first point (it's (0,0) in log space)
            if len(log_R) > 2:
                mask = np.isfinite(log_R) & np.isfinite(log_U) & (log_R > 0)
                if np.sum(mask) >= 2:
                    slope, intercept, r_val, p_val, se = stats.linregress(
                        log_R[mask], log_U[mask])
                    pred = intercept + slope * log_R[mask]
                    res = np.mean((log_U[mask] - pred)**2)
                    fit_results['power_law'] = {
                        'alpha': slope, 'residual': res,
                        'r_squared': r_val**2
                    }
        except:
            pass

        # Fit B: Exponential  U/I = e^(λ(R - R0))
        try:
            valid = U_norm > 0
            log_U = np.log(U_norm[valid])
            R_shifted = R[valid] - R[0]
            if len(R_shifted) >= 3:
                mask = np.isfinite(log_U) & np.isfinite(R_shifted)
                if np.sum(mask) >= 2:
                    slope, intercept, r_val, p_val, se = stats.linregress(
                        R_shifted[mask], log_U[mask])
                    pred = intercept + slope * R_shifted[mask]
                    res = np.mean((log_U[mask] - pred)**2)
                    fit_results['exponential'] = {
                        'lambda': slope, 'residual': res,
                        'r_squared': r_val**2
                    }
        except:
            pass

        # Fit C: Saturation  U = U_max(1 - e^(-kR))
        try:
            U_max_est = np.max(U) * 1.1
            def sat_model(R_val, U_max, k_sat):
                return U_max * (1 - np.exp(-k_sat * R_val))
            popt, _ = optimize.curve_fit(sat_model, R, U,
                                        p0=[U_max_est, 0.1],
                                        bounds=([U[-1]*0.5, 1e-6],
                                               [U[-1]*10, 100]),
                                        maxfev=5000)
            pred = sat_model(R, *popt)
            res = np.mean((np.log(np.maximum(U, 1e-15)) -
                          np.log(np.maximum(pred, 1e-15)))**2)
            fit_results['saturation'] = {
                'U_max': popt[0], 'k': popt[1], 'residual': res
            }
        except:
            pass

        # Also fit logarithmic: U/I = 1 + k·ln(R/R0)
        try:
            valid = R_norm > 0
            log_R = np.log(R_norm[valid])
            mask = np.isfinite(log_R)
            if np.sum(mask) >= 2:
                slope, intercept, r_val, p_val, se = stats.linregress(
                    log_R[mask], U_norm[valid][mask])
                pred = intercept + slope * log_R[mask]
                res = np.mean((U_norm[valid][mask] - pred)**2)
                # Convert to log-space residual for fair comparison
                log_pred = np.log(np.maximum(pred, 1e-15))
                log_actual = np.log(np.maximum(U_norm[valid][mask], 1e-15))
                res_log = np.mean((log_actual - log_pred)**2)
                fit_results['logarithmic'] = {
                    'k': slope, 'residual': res_log,
                    'r_squared': r_val**2
                }
        except:
            pass

        if not fit_results:
            self.results['observed_form'] = 'undetermined'
            self.results['alpha_observed'] = np.nan
            return 'undetermined'

        # Select best fit
        best_form = min(fit_results, key=lambda k: fit_results[k]['residual'])
        self.results['observed_form'] = best_form
        self.results['fit_results'] = fit_results

        if best_form == 'power_law':
            self.results['alpha_observed'] = fit_results['power_law']['alpha']
        elif best_form == 'exponential':
            self.results['alpha_observed'] = np.nan
            self.results['lambda_observed'] = fit_results['exponential']['lambda']
        elif best_form == 'saturation':
            self.results['alpha_observed'] = np.nan
            self.results['Umax_observed'] = fit_results['saturation']['U_max']
        elif best_form == 'logarithmic':
            self.results['alpha_observed'] = np.nan

        return best_form

    def check_five_properties(self, R, U):
        """
        Step 5: Check all five ARC properties.
        """
        properties = {}

        # P1: Threshold behaviour - is there a transition from ~linear to super-linear?
        if len(R) > 4:
            mid = len(R) // 2
            # Slope in first half vs second half (log-log)
            try:
                R_pos = R[R > 0]
                U_pos = U[:len(R_pos)]
                log_R = np.log(R_pos)
                log_U = np.log(np.maximum(U_pos, 1e-15))

                s1, _, _, _, _ = stats.linregress(log_R[:mid], log_U[:mid])
                s2, _, _, _, _ = stats.linregress(log_R[mid:], log_U[mid:])
                properties['threshold'] = abs(s2 - s1) > 0.1
                properties['threshold_detail'] = f'slope change: {s1:.2f} → {s2:.2f}'
            except:
                properties['threshold'] = False
                properties['threshold_detail'] = 'insufficient data'

        # P2: Recursive depth dependence - does U increase with R?
        properties['depth_dependence'] = U[-1] > U[0] * 1.01
        properties['depth_ratio'] = U[-1] / max(U[0], 1e-15)

        # P3: Base quality dependence - is I > 0 required?
        properties['base_quality'] = U[0] > 0
        properties['I_value'] = U[0]

        # P4: Multiplicative I×R interaction
        # Test: does doubling I double U at all R?
        # We can only check separability: U(R)/U(1) should be independent of I
        # Since we have one run, check if U/I is monotonically increasing
        U_normalised = U / max(U[0], 1e-15)
        properties['multiplicative'] = np.all(np.diff(U_normalised) >= -0.01 * U_normalised[:-1])

        # P5: Regime boundaries - does growth eventually slow?
        if len(U) > 3:
            second_diff = np.diff(np.diff(U))
            # Check if acceleration decreases in later stages
            late_accel = np.mean(second_diff[len(second_diff)//2:])
            early_accel = np.mean(second_diff[:len(second_diff)//2])
            properties['regime_boundary'] = late_accel < early_accel
            properties['boundary_detail'] = f'accel: {early_accel:.3e} → {late_accel:.3e}'
        else:
            properties['regime_boundary'] = False

        self.results['five_properties'] = properties
        return properties

    def evaluate(self):
        """
        Step 6: Compare blind prediction against observation.
        """
        predicted = self.results.get('predicted_form', 'unknown')
        observed = self.results.get('observed_form', 'unknown')

        # Form match
        form_match = predicted == observed
        # Also accept power_law prediction matching logarithmic observation
        # (logarithmic is power-law with α→0)
        if predicted == 'power_law' and observed == 'logarithmic':
            alpha_pred = self.results.get('alpha_predicted', np.nan)
            if not np.isnan(alpha_pred) and alpha_pred < 0.5:
                form_match = True

        self.results['form_match'] = form_match

        # Alpha match (if both are power-law)
        if predicted == 'power_law' and observed == 'power_law':
            alpha_pred = self.results.get('alpha_predicted', np.nan)
            alpha_obs = self.results.get('alpha_observed', np.nan)
            if not np.isnan(alpha_pred) and not np.isnan(alpha_obs):
                self.results['alpha_error'] = abs(alpha_pred - alpha_obs)
                self.results['alpha_error_pct'] = (
                    abs(alpha_pred - alpha_obs) / max(abs(alpha_obs), 0.01) * 100
                )
            else:
                self.results['alpha_error'] = np.nan
        else:
            self.results['alpha_error'] = np.nan

        return form_match


# ============================================================================
# THE EIGHT TEST SYSTEMS
# Each returns (R_array, U_array) where R is recursive depth and U is
# measured capability. The apparatus has never seen these systems.
# ============================================================================

def system_1_gradient_descent_momentum():
    """
    Gradient descent WITH MOMENTUM on a non-convex function.
    Momentum creates recursive feedback: each step uses accumulated
    velocity from all previous steps.

    Output feeds back as input (velocity carries forward).
    Structured asymmetry: the loss landscape has specific curvature.
    """
    # Rosenbrock function: f(x,y) = (a-x)^2 + b(y-x^2)^2
    a, b = 1.0, 100.0

    def loss(x, y):
        return (a - x)**2 + b * (y - x**2)**2

    def grad(x, y):
        dx = -2*(a - x) - 4*b*x*(y - x**2)
        dy = 2*b*(y - x**2)
        return np.array([dx, dy])

    x, y = -1.5, 1.5  # Starting point
    lr = 0.001
    momentum = 0.9
    v = np.array([0.0, 0.0])

    # Run and record capability = 1/loss (higher = better)
    max_depth = 200
    R_vals = list(range(0, max_depth + 1, 1))
    U_vals = []

    for r in range(max_depth + 1):
        current_loss = loss(x, y)
        U_vals.append(1.0 / (current_loss + 1e-10))  # Capability = inverse loss

        g = grad(x, y)
        v = momentum * v - lr * g  # RECURSIVE: velocity accumulates
        x += v[0]
        y += v[1]

    # Subsample for cleaner measurement
    step = 5
    R_out = np.array(R_vals[::step])
    U_out = np.array(U_vals[::step])
    return R_out, U_out


def system_2_pagerank():
    """
    PageRank iterative computation on a scale-free network.
    Each iteration refines page scores using the ENTIRE accumulated
    link structure. Output of iteration n becomes input for n+1.

    Recursive: each iteration operates on full accumulated state.
    Structured asymmetry: the network has heterogeneous degree distribution.
    """
    N = 500  # Nodes
    # Generate scale-free network (Barabási-Albert)
    m = 3  # Edges per new node
    adj = np.zeros((N, N))
    degrees = np.zeros(N)

    # Start with complete graph on m+1 nodes
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            adj[i, j] = adj[j, i] = 1
            degrees[i] += 1
            degrees[j] += 1

    # Preferential attachment
    for new_node in range(m + 1, N):
        probs = degrees[:new_node] / max(degrees[:new_node].sum(), 1)
        targets = np.random.choice(new_node, size=m, replace=False, p=probs)
        for t in targets:
            adj[new_node, t] = adj[t, new_node] = 1
            degrees[new_node] += 1
            degrees[t] += 1

    # Transition matrix
    out_degree = adj.sum(axis=1)
    out_degree[out_degree == 0] = 1
    M = (adj / out_degree[:, None]).T

    damping = 0.85
    teleport = (1 - damping) / N * np.ones(N)

    # PageRank iterations
    pr = np.ones(N) / N  # Uniform start
    max_iter = 80

    R_vals = []
    U_vals = []

    for r in range(max_iter):
        R_vals.append(r + 1)
        # Capability = information content (KL divergence from uniform)
        pr_safe = np.maximum(pr, 1e-15)
        uniform = np.ones(N) / N
        kl_div = np.sum(pr_safe * np.log(pr_safe / uniform))
        U_vals.append(kl_div + 1.0)  # Shift to positive

        pr_new = damping * M @ pr + teleport  # RECURSIVE
        pr = pr_new / pr_new.sum()

    return np.array(R_vals), np.array(U_vals)


def system_3_evolutionary_algorithm():
    """
    Evolutionary algorithm with elitism on a rugged fitness landscape.
    Each generation builds on the BEST of the previous generation.
    Mutations explore; selection preserves accumulated gains.

    Recursive: each generation's starting population = previous output.
    Structured asymmetry: the fitness landscape has specific structure.
    """
    # NK fitness landscape (tunable ruggedness)
    N_genes = 20
    K = 4  # Epistatic interactions (ruggedness)

    # Generate fitness contributions
    fitness_table = {}
    for i in range(N_genes):
        neighbours = [(i + j + 1) % N_genes for j in range(K)]
        for bits in range(2**(K+1)):
            fitness_table[(i, bits)] = np.random.random()

    def fitness(genome):
        total = 0
        for i in range(N_genes):
            neighbours = [(i + j + 1) % N_genes for j in range(K)]
            bits = genome[i]
            for n in neighbours:
                bits = bits * 2 + genome[n]
            total += fitness_table.get((i, bits % (2**(K+1))), 0)
        return total / N_genes

    # Evolution
    pop_size = 100
    n_generations = 150
    mutation_rate = 0.05
    elite_frac = 0.1

    population = [np.random.randint(0, 2, N_genes) for _ in range(pop_size)]

    R_vals = []
    U_vals = []

    for gen in range(n_generations):
        fitnesses = np.array([fitness(ind) for ind in population])
        R_vals.append(gen + 1)
        U_vals.append(np.max(fitnesses))  # Best fitness = capability

        # Selection (tournament)
        new_pop = []
        # Elitism
        elite_n = max(int(pop_size * elite_frac), 1)
        elite_idx = np.argsort(fitnesses)[-elite_n:]
        for idx in elite_idx:
            new_pop.append(population[idx].copy())

        while len(new_pop) < pop_size:
            # Tournament selection
            t1, t2 = np.random.choice(pop_size, 2, replace=False)
            parent = population[t1] if fitnesses[t1] > fitnesses[t2] else population[t2]
            child = parent.copy()
            # Mutation
            for i in range(N_genes):
                if np.random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            new_pop.append(child)

        population = new_pop

    return np.array(R_vals), np.array(U_vals)


def system_4_iterative_denoising():
    """
    Iterative denoising of a structured signal.
    Each iteration uses the current best estimate to refine the next.
    Output feeds directly back as input.

    Recursive: each denoising step operates on accumulated estimate.
    Structured asymmetry: the true signal has specific structure.
    """
    # True signal: sum of sinusoids
    N = 1000
    t = np.linspace(0, 4 * np.pi, N)
    true_signal = np.sin(t) + 0.5 * np.sin(3 * t) + 0.3 * np.sin(7 * t)

    # Add heavy noise
    noise_level = 2.0
    noisy = true_signal + noise_level * np.random.randn(N)

    # Iterative Wiener-like denoising
    max_iter = 60
    estimate = noisy.copy()
    kernel_size = 15
    kernel = np.ones(kernel_size) / kernel_size

    R_vals = []
    U_vals = []

    for r in range(max_iter):
        R_vals.append(r + 1)

        # Capability = SNR (signal-to-noise ratio in dB)
        residual = estimate - true_signal
        signal_power = np.mean(true_signal**2)
        noise_power = np.mean(residual**2)
        snr = 10 * np.log10(signal_power / max(noise_power, 1e-15))
        U_vals.append(snr + 30)  # Shift positive

        # RECURSIVE denoising step:
        # Use current estimate to create adaptive filter
        smoothed = np.convolve(estimate, kernel, mode='same')

        # Adaptive: weight towards smoothed where estimate is far from mean
        deviation = np.abs(estimate - np.mean(estimate))
        weight = np.clip(deviation / (np.std(estimate) + 1e-10), 0, 1)

        estimate = weight * smoothed + (1 - weight) * estimate
        # Small regularisation toward original data
        estimate = 0.95 * estimate + 0.05 * noisy

    return np.array(R_vals), np.array(U_vals)


def system_5_newton_method():
    """
    Newton's method for finding roots of a polynomial.
    Each iteration uses the current estimate AND its derivative
    to compute the next estimate. Quadratic convergence = recursive.

    Recursive: x_{n+1} = x_n - f(x_n)/f'(x_n)
    Structured asymmetry: the function has specific curvature.
    """
    # f(x) = x^3 - 2x - 5 (has root near x ≈ 2.0946)
    def f(x):
        return x**3 - 2*x - 5

    def f_prime(x):
        return 3*x**2 - 2

    x = 4.0  # Starting far from root
    max_iter = 25

    R_vals = []
    U_vals = []

    for r in range(max_iter):
        R_vals.append(r + 1)
        error = abs(f(x))
        # Capability = precision = 1/error (higher = better)
        U_vals.append(1.0 / max(error, 1e-16))

        if abs(f_prime(x)) > 1e-15:
            x = x - f(x) / f_prime(x)  # RECURSIVE
        else:
            break

    return np.array(R_vals), np.array(U_vals)


def system_6_simulated_annealing():
    """
    Simulated annealing on a multi-modal energy landscape.
    Temperature schedule creates a transition from exploration to
    exploitation. The cooling schedule itself is recursive: each
    temperature depends on the previous.

    Bounded by the global minimum energy.
    """
    # 2D energy landscape with multiple minima
    def energy(x, y):
        return (np.sin(3*x) * np.cos(3*y) +
                0.5 * np.sin(5*x + y) +
                0.3 * np.cos(7*y - 2*x) +
                0.1 * (x**2 + y**2))

    x, y = 3.0, 3.0
    T = 10.0
    cooling = 0.97
    best_E = energy(x, y)
    max_iter = 300

    R_vals = []
    U_vals = []

    for r in range(max_iter):
        R_vals.append(r + 1)
        U_vals.append(-best_E + 5)  # Capability = -energy (shifted positive)

        # Propose move
        dx = np.random.normal(0, T * 0.1)
        dy = np.random.normal(0, T * 0.1)
        new_E = energy(x + dx, y + dy)

        # Accept/reject (Metropolis)
        dE = new_E - energy(x, y)
        if dE < 0 or np.random.random() < np.exp(-dE / max(T, 1e-10)):
            x += dx
            y += dy
            if new_E < best_E:
                best_E = new_E

        T *= cooling  # RECURSIVE cooling

    return np.array(R_vals[::3]), np.array(U_vals[::3])


def system_7_coupled_oscillators():
    """
    Kuramoto model: coupled oscillators synchronising.
    Each oscillator's phase at t+1 depends on ALL other oscillators' phases at t.
    Synchronisation = recursive collective computation.

    Recursive: global order parameter at step n feeds back to all oscillators.
    Structured asymmetry: oscillators have different natural frequencies.
    """
    N_osc = 50
    # Natural frequencies (structured asymmetry: Lorentzian distribution)
    omega = np.random.standard_cauchy(N_osc) * 0.5
    omega = np.clip(omega, -5, 5)

    K_coupling = 2.0  # Coupling strength (above critical)
    dt = 0.05
    max_steps = 400
    theta = np.random.uniform(0, 2 * np.pi, N_osc)

    R_vals = []
    U_vals = []

    for step in range(0, max_steps, 4):
        R_vals.append(step + 1)
        # Order parameter r = |<e^{iθ}>|
        order = abs(np.mean(np.exp(1j * theta)))
        U_vals.append(order)

        # Advance 4 timesteps
        for _ in range(4):
            # Mean field coupling (RECURSIVE: all-to-all feedback)
            mean_field = np.mean(np.sin(theta[None, :] - theta[:, None]), axis=1)
            dtheta = omega + K_coupling * mean_field
            theta += dtheta * dt
            theta = theta % (2 * np.pi)

    return np.array(R_vals), np.array(U_vals)


def system_8_iterative_decoding():
    """
    Iterative belief propagation decoding of an LDPC code.
    Simplified model: each iteration refines bit probability estimates
    using accumulated evidence from connected check nodes.

    This is the closest analogue to quantum error correction in
    classical information theory.

    Recursive: each decoding iteration uses full accumulated belief state.
    Structured asymmetry: the code has specific parity check structure.
    """
    # Simplified LDPC-like iterative decoding
    N_bits = 100
    N_checks = 50

    # Parity check connections (each check connects to ~6 bits)
    H = np.zeros((N_checks, N_bits))
    for i in range(N_checks):
        connected = np.random.choice(N_bits, size=6, replace=False)
        H[i, connected] = 1

    # True codeword (all zeros for simplicity)
    codeword = np.zeros(N_bits)

    # Channel: BSC with error probability p
    p_error = 0.08
    received = codeword.copy()
    errors = np.random.random(N_bits) < p_error
    received[errors] = 1 - received[errors]

    # Initial log-likelihood ratios
    p_0 = 1 - p_error
    p_1 = p_error
    llr_channel = np.log(p_0 / p_1) * (1 - 2 * received)

    # Iterative decoding
    max_iter = 40
    llr = llr_channel.copy()

    R_vals = []
    U_vals = []

    for iteration in range(max_iter):
        R_vals.append(iteration + 1)

        # Capability = fraction of bits with correct hard decision
        decisions = (llr < 0).astype(float)
        bit_error_rate = np.mean(decisions != codeword)
        # Capability = 1 / BER (higher = better)
        U_vals.append(1.0 / max(bit_error_rate, 1e-6))

        # RECURSIVE belief propagation step
        # Check-to-variable messages
        new_llr = llr_channel.copy()
        for j in range(N_checks):
            connected_bits = np.where(H[j] > 0)[0]
            for bit in connected_bits:
                others = [b for b in connected_bits if b != bit]
                if others:
                    # Product of tanh(llr/2) for other bits
                    product = np.prod(np.tanh(llr[others] / 2))
                    product = np.clip(product, -0.9999, 0.9999)
                    check_msg = 2 * np.arctanh(product)
                    new_llr[bit] += check_msg

        llr = new_llr  # RECURSIVE UPDATE

    return np.array(R_vals), np.array(U_vals)


# ============================================================================
# RUN ALL TESTS
# ============================================================================

systems = {
    'Gradient Descent\n(Momentum)': system_1_gradient_descent_momentum,
    'PageRank\n(Network)': system_2_pagerank,
    'Evolutionary\nAlgorithm': system_3_evolutionary_algorithm,
    'Iterative\nDenoising': system_4_iterative_denoising,
    'Newton\'s\nMethod': system_5_newton_method,
    'Simulated\nAnnealing': system_6_simulated_annealing,
    'Coupled\nOscillators': system_7_coupled_oscillators,
    'LDPC Iterative\nDecoding': system_8_iterative_decoding,
}

print("=" * 72)
print("THE ARC FRAMEWORK: DEFINITIVE CROSS-DOMAIN BLIND PREDICTION TEST")
print("=" * 72)
print()
print("Protocol: For each system, measure ⊕ FIRST, predict scaling BLIND,")
print("          then compare against independently fitted scaling curve.")
print()
print("=" * 72)

all_results = {}

for name, system_fn in systems.items():
    clean_name = name.replace('\n', ' ')
    print(f"\n{'─' * 72}")
    print(f"SYSTEM: {clean_name}")
    print(f"{'─' * 72}")

    # Generate data
    R, U = system_fn()
    print(f"  Data: {len(R)} points, R ∈ [{R[0]}, {R[-1]}], "
          f"U ∈ [{U[0]:.4f}, {U[-1]:.4f}]")

    # Create measurement apparatus
    apparatus = ARCMeasurementApparatus(clean_name)

    # STEP 1: Measure composition operator (BLIND - no curve fitting yet)
    comp = apparatus.measure_composition_operator(R, U)
    if comp is None:
        print(f"  ⚠ Could not measure composition operator")
        all_results[name] = apparatus.results
        continue
    dU, Ua = comp

    # STEP 2: Classify operator
    op_class = apparatus.classify_operator(dU, Ua)
    conf = apparatus.results['classification_confidence']
    beta = apparatus.results.get('beta_measured', np.nan)
    print(f"  ⊕ Classification: {op_class.upper()} "
          f"(confidence: {conf:.1f}x)")
    if not np.isnan(beta):
        print(f"  β measured: {beta:.4f}")

    # STEP 3: BLIND PREDICTION
    predicted_form = apparatus.predict_scaling()
    alpha_pred = apparatus.results.get('alpha_predicted', np.nan)
    print(f"  ★ BLIND PREDICTION: {predicted_form.upper()}", end='')
    if not np.isnan(alpha_pred):
        print(f" with α = {alpha_pred:.3f}")
    else:
        print()

    # STEP 4: INDEPENDENTLY fit actual curve (the "answer")
    observed_form = apparatus.fit_actual_scaling(R, U)
    alpha_obs = apparatus.results.get('alpha_observed', np.nan)
    print(f"  ► OBSERVED:         {observed_form.upper()}", end='')
    if not np.isnan(alpha_obs):
        print(f" with α = {alpha_obs:.3f}")
    else:
        print()

    # STEP 5: Check five properties
    props = apparatus.check_five_properties(R, U)
    n_props = sum([
        props.get('threshold', False),
        props.get('depth_dependence', False),
        props.get('base_quality', False),
        props.get('multiplicative', False),
        props.get('regime_boundary', False),
    ])

    # STEP 6: Evaluate
    match = apparatus.evaluate()
    alpha_err = apparatus.results.get('alpha_error', np.nan)

    print()
    if match:
        print(f"  ✓ FORM PREDICTION: CORRECT")
    else:
        print(f"  ✗ FORM PREDICTION: INCORRECT "
              f"(predicted {predicted_form}, observed {observed_form})")

    if not np.isnan(alpha_err):
        print(f"  α error: {alpha_err:.3f} "
              f"({apparatus.results.get('alpha_error_pct', 0):.1f}%)")
    elif not np.isnan(alpha_pred) and not np.isnan(alpha_obs):
        print(f"  α predicted: {alpha_pred:.3f}, observed: {alpha_obs:.3f}")

    print(f"  Five properties: {n_props}/5 present")

    apparatus.results['n_properties'] = n_props
    all_results[name] = apparatus.results


# ============================================================================
# COMPREHENSIVE RESULTS TABLE
# ============================================================================

print("\n\n")
print("=" * 100)
print("╔════════════════════════════════════════════════════════════════════════════════════════════════╗")
print("║                    RESULTS: BLIND PREDICTION TEST ACROSS 8 SYSTEMS                          ║")
print("╚════════════════════════════════════════════════════════════════════════════════════════════════╝")
print("=" * 100)

# Summary table
print(f"\n{'System':<25s} | {'⊕ Type':<15s} | {'Predicted':<12s} | "
      f"{'Observed':<12s} | {'Match':>5s} | {'α pred':>7s} | "
      f"{'α obs':>7s} | {'α err':>7s} | {'Props':>5s}")
print("─" * 110)

correct_form = 0
correct_alpha = 0
total_alpha = 0
total = 0
total_properties = 0

for name, res in all_results.items():
    clean = name.replace('\n', ' ')
    op = res.get('operator_class', '?')
    pred = res.get('predicted_form', '?')
    obs = res.get('observed_form', '?')
    match = res.get('form_match', False)
    a_pred = res.get('alpha_predicted', np.nan)
    a_obs = res.get('alpha_observed', np.nan)
    a_err = res.get('alpha_error', np.nan)
    n_props = res.get('n_properties', 0)

    total += 1
    if match:
        correct_form += 1
    if not np.isnan(a_err):
        total_alpha += 1
        if a_err < 0.5:
            correct_alpha += 1
    total_properties += n_props

    match_str = "  ✓" if match else "  ✗"
    a_pred_str = f"{a_pred:7.3f}" if not np.isnan(a_pred) else "    N/A"
    a_obs_str = f"{a_obs:7.3f}" if not np.isnan(a_obs) else "    N/A"
    a_err_str = f"{a_err:7.3f}" if not np.isnan(a_err) else "    N/A"

    print(f"  {clean:<23s} | {op:<15s} | {pred:<12s} | "
          f"{obs:<12s} | {match_str:>5s} | {a_pred_str:>7s} | "
          f"{a_obs_str:>7s} | {a_err_str:>7s} | {n_props:>3d}/5")

print("─" * 110)
print()

# ============================================================================
# FINAL SCORING
# ============================================================================

print("╔════════════════════════════════════════════════════════════════════════╗")
print("║                        FINAL SCORING                                ║")
print("╚════════════════════════════════════════════════════════════════════════╝")
print()
print(f"  FUNCTIONAL FORM PREDICTION:")
print(f"    Correct: {correct_form}/{total} ({100*correct_form/total:.0f}%)")
print(f"    The composition operator predicted the correct scaling")
print(f"    function in {correct_form} of {total} systems tested.")
print()

if total_alpha > 0:
    print(f"  EXPONENT (α) PREDICTION:")
    print(f"    Within 0.5 of observed: {correct_alpha}/{total_alpha} "
          f"({100*correct_alpha/total_alpha:.0f}%)")
    print()

avg_props = total_properties / total
print(f"  FIVE UNIVERSAL PROPERTIES:")
print(f"    Average: {avg_props:.1f}/5 properties present per system")
print(f"    Total: {total_properties}/{total * 5} "
      f"({100*total_properties/(total*5):.0f}%)")
print()

# Overall assessment
print(f"  ═══════════════════════════════════════════════════════════════")
if correct_form >= total * 0.75:
    print(f"  ASSESSMENT: The composition operator is a RELIABLE predictor")
    print(f"  of scaling function form. The central claim of Theorem 1 is")
    print(f"  SUPPORTED by this {total}-system cross-domain test.")
elif correct_form >= total * 0.5:
    print(f"  ASSESSMENT: The composition operator is a PARTIAL predictor")
    print(f"  of scaling function form. The framework shows promise but")
    print(f"  has significant failure modes requiring investigation.")
else:
    print(f"  ASSESSMENT: The composition operator FAILS to reliably predict")
    print(f"  scaling function form. Theorem 1 is NOT supported by this test.")

print()

# ============================================================================
# DETAILED INTERPRETATION
# ============================================================================

print("=" * 72)
print("DETAILED INTERPRETATION")
print("=" * 72)
print()

for name, res in all_results.items():
    clean = name.replace('\n', ' ')
    match = res.get('form_match', False)
    if not match:
        pred = res.get('predicted_form', '?')
        obs = res.get('observed_form', '?')
        print(f"  MISMATCH: {clean}")
        print(f"    Predicted: {pred}, Observed: {obs}")
        beta = res.get('beta_measured', np.nan)
        if not np.isnan(beta):
            print(f"    β measured: {beta:.4f}")
        resids = res.get('classification_residuals', {})
        if resids:
            for form, r in sorted(resids.items(), key=lambda x: x[1]):
                print(f"      {form}: residual = {r:.6f}")
        print()

# ============================================================================
# GENERATE COMPREHENSIVE FIGURE
# ============================================================================

fig2, axes = plt.subplots(4, 4, figsize=(24, 24))
fig2.suptitle('ARC Framework: Blind Prediction Test Across 8 Recursive Systems',
              fontsize=16, fontweight='bold', y=0.98)

# Left column: U vs R curves with fits
# Right column: Composition operator (dU vs U_accumulated)

system_list = list(systems.items())
for idx, (name, system_fn) in enumerate(system_list):
    row = idx // 2
    col_base = (idx % 2) * 2

    R, U = system_fn()
    res = all_results[name]

    # Panel A: U vs R
    ax_a = axes[row, col_base]
    ax_a.plot(R, U, 'b-', linewidth=1.5, alpha=0.8, label='Data')

    # Show predicted vs observed form
    pred = res.get('predicted_form', '?')
    obs = res.get('observed_form', '?')
    match = res.get('form_match', False)
    color = 'green' if match else 'red'
    symbol = '✓' if match else '✗'

    clean_name = name.replace('\n', ' ')
    ax_a.set_title(f'{clean_name}\n{symbol} Pred: {pred} | Obs: {obs}',
                   fontsize=9, color=color, fontweight='bold')
    ax_a.set_xlabel('R (recursive depth)', fontsize=8)
    ax_a.set_ylabel('U (capability)', fontsize=8)
    ax_a.tick_params(labelsize=7)

    # Panel B: Composition operator
    ax_b = axes[row, col_base + 1]
    dU = res.get('marginal_gains', None)
    Ua = res.get('accumulated', None)

    if dU is not None and Ua is not None:
        valid = (dU > 0) & (Ua > 0)
        if np.sum(valid) > 0:
            ax_b.loglog(Ua[valid], dU[valid], 'ko', markersize=3, alpha=0.6)

            # Show classification
            op = res.get('operator_class', '?')
            beta = res.get('beta_measured', np.nan)
            beta_str = f', β={beta:.2f}' if not np.isnan(beta) else ''
            ax_b.set_title(f'⊕: {op}{beta_str}', fontsize=9, fontweight='bold')

            # Overlay best fit line
            if not np.isnan(beta) and op == 'multiplicative':
                x_fit = np.linspace(np.log(Ua[valid].min()),
                                   np.log(Ua[valid].max()), 50)
                intercept = np.mean(np.log(dU[valid]) - beta * np.log(Ua[valid]))
                y_fit = np.exp(intercept + beta * x_fit)
                ax_b.loglog(np.exp(x_fit), y_fit, 'r-', linewidth=2,
                           label=f'β = {beta:.2f}')
                ax_b.legend(fontsize=7)

    ax_b.set_xlabel('U_accumulated', fontsize=8)
    ax_b.set_ylabel('ΔU (marginal gain)', fontsize=8)
    ax_b.tick_params(labelsize=7)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('/home/claude/arc_blind_prediction_test.png',
            dpi=150, bbox_inches='tight', facecolor='white')
print("\nFigure saved: arc_blind_prediction_test.png")

# ============================================================================
# FINAL SUMMARY FIGURE: THE SCORECARD
# ============================================================================

fig3, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')

# Create summary scorecard
cell_text = []
for name, res in all_results.items():
    clean = name.replace('\n', ' ')
    op = res.get('operator_class', '?')
    pred = res.get('predicted_form', '?')
    obs = res.get('observed_form', '?')
    match = '✓' if res.get('form_match', False) else '✗'
    a_pred = res.get('alpha_predicted', np.nan)
    a_obs = res.get('alpha_observed', np.nan)
    n_props = res.get('n_properties', 0)

    cell_text.append([
        clean,
        op,
        pred,
        obs,
        match,
        f'{a_pred:.2f}' if not np.isnan(a_pred) else 'N/A',
        f'{a_obs:.2f}' if not np.isnan(a_obs) else 'N/A',
        f'{n_props}/5'
    ])

col_labels = ['System', '⊕ Type', 'Predicted', 'Observed', 'Match',
              'α predicted', 'α observed', 'Properties']

table = ax.table(cellText=cell_text, colLabels=col_labels,
                loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 2.0)

# Colour the match column
for i, row in enumerate(cell_text):
    match_cell = table[i + 1, 4]
    if row[4] == '✓':
        match_cell.set_facecolor('#c8e6c9')
    else:
        match_cell.set_facecolor('#ffcdd2')

ax.set_title(f'ARC Blind Prediction Scorecard: {correct_form}/{total} Correct '
             f'({100*correct_form/total:.0f}%)\n'
             f'Average properties: {avg_props:.1f}/5',
             fontsize=14, fontweight='bold', pad=20)

plt.savefig('/home/claude/arc_scorecard.png', dpi=150,
            bbox_inches='tight', facecolor='white')
print("Scorecard saved: arc_scorecard.png")

print("\n" + "=" * 72)
print("TEST COMPLETE")
print("=" * 72)
