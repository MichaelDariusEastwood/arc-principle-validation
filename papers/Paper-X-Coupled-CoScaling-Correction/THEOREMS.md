# Paper X — corrected theorem statements and proofs

**Correction date:** 19 October 2026. **Basis:** repository commit
`040ad474fa0dcea14f3ec53521906acabfc2c670`, together with the mathematical
statements of Paper X v1.15 supplied for review. **Scope:** a mathematical
correction, not a new empirical result, a new law, or a change to priority dates.

This document supplies replacement statements and proofs for Theorems 1–7 and
the time-varying-exponent remark. It supersedes the corresponding mathematical
statements in the historical manuscript mirrors for the purposes of the revised
independent test suite. The original numbered results are retained; this is not
a separate theory. The historical HTML/PDF files and experiment outputs have not
been silently rewritten. A new published manuscript edition must incorporate
these replacements; creation of this file does not update a website or OSF deposit.

## Model, domain and meanings

Let C0, A0, b > 0; let beta and k be real constants; and let gamma1, gamma2,
gamma3 and d0 be non-negative. The scalar model is

$$\dot C=bC^{1+k},\qquad A=A_0 C^\beta,\qquad r=bC^k,$$
$$\dot D=\gamma_1\dot C+\gamma_2 C+\gamma_3 rD-AD,\qquad D(0)=C_0d_0.$$

C has its maximal positive domain: 0 <= t < t_star when k > 0, where
`t_star = 1/(k b C0^k)`; otherwise 0 <= t < infinity. Define

$$d=D/C,\quad \tau=\log(C/C_0),\quad s=\beta-k,\quad
q=A/r=q_0 e^{s\tau},\quad q_0=(A_0/b)C_0^s.$$

Capability tends to infinity and tau tends to infinity at the end of that
maximal domain. Direct differentiation, without an equilibrium approximation, gives

$$\dot d=\gamma_1r+\gamma_2-[A+(1-\gamma_3)r]d,$$
$$d'=\gamma_1+\frac{\gamma_2}{bC_0^k}e^{-k\tau}
       -[q_0e^{s\tau}+1-\gamma_3]d. \tag{D}$$

A prime denotes a depth derivative, not a wall-clock derivative. An
*instantaneous equilibrium* is the zero of the current right-hand side; it is
not generally a solution. *Vanishing relative misalignment*, *boundedness*,
*homogeneous stability*, *absolute misalignment*, and *safety* are different
claims. Here d is a non-negative normalised index; the equations do not enforce
d <= 1 outside the gain-only invariant region. No result is a safety guarantee.

## Theorem 1 — autonomous gain-only transient

For constant A,r >= 0 with kappa=A+r > 0 and gamma2=gamma3=0,

$$d(t)=d_*+(d_0-d_*)e^{-\kappa t},\qquad
 d_*={\gamma_1r\over A+r}.$$

The equilibrium is globally exponentially attracting with rate kappa. When
A=r=0 instead, d(t)=d0: there is no unique attracting equilibrium.

**Proof.** Multiply `d_dot + kappa d = gamma1 r` by `exp(kappa t)` and integrate.
The difference of any two solutions is their initial difference times
`exp(-kappa t)`. These constant-coefficient cases are local/autonomous
specialisations; holding r constant does **not** mean holding C constant.

## Theorem 2 — gain-only bound, limits, and the positive-drift hypothesis

Let gamma2=gamma3=0. For non-negative locally integrable A(t),r(t),

$$0\le d(t)\le\max(d_0,\gamma_1)$$

on every interval on which the solution exists. Under the power laws above,

$$\lim_{\tau\to\infty}d(\tau)=
\begin{cases}
0,&\beta>k,\\
\gamma_1/(1+A_0/b),&\beta=k,\\
\gamma_1,&\beta<k.
\end{cases}$$

Consequently, **for gamma1 > 0**,

$$d\longrightarrow0\quad\Longleftrightarrow\quad\beta>k.$$

For gamma1=0 the necessity statement does not apply: d tends to zero for
**every** beta,k. More precisely,

$$d(\tau)=d_0\exp[-\tau-Q(\tau)],\quad
Q(\tau)=\begin{cases}q_0(e^{s\tau}-1)/s,&s\ne0,\\q_0\tau,&s=0.\end{cases}$$

For positive gamma1 and s>0 there is also a leading-order trajectory relation,
not just an instantaneous-equilibrium relation:

$$d(\tau)\sim{\gamma_1\over q(\tau)}
={\gamma_1b\over A_0}C^{k-\beta}. \tag{L}$$

**Proof.** At d=0 the derivative is non-negative; at any d >= gamma1 it is
non-positive. Scalar comparison proves invariance (including locally integrable
coefficients via the integral solution). In depth, `d'=gamma1-(1+q)d`.
When q tends to zero, for every epsilon>0 the eventual solutions lie between
solutions with decay 1 and 1+epsilon; their limiting equilibria squeeze d to
gamma1. Constant q gives the middle row exactly. If q tends to infinity,
eventually `d' <= gamma1-M d` for any M>0, proving limsup d <= gamma1/M
and hence convergence to zero. Zero forcing gives the displayed exact solution.
For (L), put z=q d/gamma1. Then
`z'=q-(q+1-s)z`. For any epsilon>0, sufficiently large q makes the vector field
point into `[1-epsilon,1+epsilon]` at both boundaries, with a restoring magnitude
bounded away from zero outside it. Every non-negative solution enters and remains
in an arbitrarily small such interval. Thus z tends to one.

## Theorem 3 — hard-takeoff depth regularity

Let k>0 and gamma3=0, allowing gamma2 >= 0. Then

$$C(t)=C_0(1-t/t_*)^{-1/k},\qquad
\tau(t)=-k^{-1}\log(1-t/t_*).$$

The clock change is smooth and strictly increasing for t<t_star and maps that
half-open domain onto tau>=0. It does not extend the model past t_star.
Equation (D) has bounded coefficients on each compact depth interval. For
gamma1>0 its limits are exactly the three limits in Theorem 2; hence vanishing
d is equivalent to beta>k. If gamma1=0, d tends to zero for every beta, including
when gamma2>0. A convenient bound for all these additive cases is

$$0\le d(\tau)\le\max\left(d_0,\gamma_1+\frac{\gamma_2}{bC_0^k}\right).$$

**Proof.** Integrate the capability equation and divide the fraction equation by
r>0. The extra forcing is `L exp(-k tau)` with L=gamma2/(b C0^k), which vanishes.
The difference between the forced solution and its gamma2=0 counterpart, with
the same initial state, is at most

$$L\int_0^\tau e^{-(\tau-u)}e^{-ku}\,du\longrightarrow0,$$

since `1+q>=1`. Theorem 2 therefore supplies the limits. The same comparison
with forcing at most gamma1+L supplies the bound. In particular, for
`gamma1=0,gamma2=1,A0=b=C0=beta=k=1,d0=0.05`,
`d=exp(-tau)-0.95 exp(-2 tau)` tends to zero although beta=k.

Only the **regime classification** of this additive positive-gain-drift model is
independent of b. The gap at beta=k, finite-depth paths, intervention time, and
compounding prefactor thresholds do depend on b. “Regular” means locally regular
at finite tau, not uniformly bounded coefficients at infinity, exact zero drift,
or operational control of an explosion.

## Theorem 4 — compounding: autonomous and non-autonomous results separated

### 4(a) Constant coefficients

For `d_dot=iota-kappa_eff d`, with constant iota>=0 and real kappa_eff,

$$d(t)=\begin{cases}
d_0e^{-\kappa_{eff}t}
+\iota(1-e^{-\kappa_{eff}t})/\kappa_{eff},&\kappa_{eff}\ne0,\\
d_0+\iota t,&\kappa_{eff}=0.
\end{cases}$$

When `kappa_eff=A+(1-gamma3)r`, A>0, the homogeneous system is exponentially
stable exactly when `rho_prop=(gamma3-1)r/A < 1`. With iota>0, equality produces
linear growth in **wall-clock time** and a negative kappa_eff produces
exponential growth. With iota=0, the origin stays invariant even if kappa_eff<0;
at equality every initial state is constant. With iota=0,kappa_eff<0,d0>0,
growth is exponential. If A=0 use kappa_eff directly; do not divide by A.

**Proof.** Integrating factor, as in Theorem 1, with the zero-decay case taken
separately. Divergence statements concern the non-negative scalar domain.

### 4(b) Time-varying coefficients

For locally integrable iota>=0 and kappa_eff, define
`K(t)=integral_0^t kappa_eff(u) du`. The exact solution is

$$d(t)=e^{-K(t)}d_0+
\int_0^t e^{-[K(t)-K(u)]}\iota(u)\,du. \tag{V}$$

Homogeneous decay from a fixed initial time is equivalent to K(t) tending to
positive infinity at the end of the domain. Forced boundedness also requires
control of the convolution in (V); it does not follow from pointwise positive
decay, or even from K tending to infinity. On an infinite domain, a sufficient
condition is kappa_eff>=epsilon>0 and bounded iota. More generally a propagator
bound `exp[-(K(t)-K(u))] <= M exp[-a(t-u)]` gives
`d(t) <= M exp(-at)d0 + M sup(iota)/a`.

**Proof.** Multiply by exp(K) and integrate. All assertions follow from (V).
For example `d_dot=1-d/(1+t),d0=0` gives
`d=(t+t^2/2)/(1+t)`: positive decay and K=log(1+t) do not bound the forced path.
Crossing rho_prop=1 at a single instant or for a finite interval is not a
criterion for eventual divergence. This replaces any pointwise iff applied to
the non-autonomous model.

### 4(c) Power-law gain-only classification

Let gamma2=0 and **gamma1>0**. Write a=1-gamma3, s=beta-k, and
h=q0+a. The following are limits/rates of the **trajectory in depth**:

| Regime | Trajectory as tau tends to infinity |
|---|---|
| s>0, any fixed gamma3 | `d ~ gamma1/q -> 0` |
| s=0, h>0 | `d -> gamma1/h` |
| s=0, h=0 | `d=d0+gamma1 tau` |
| s=0, h<0 | exponential growth at rate -h in tau |
| s<0, gamma3<1 | `d -> gamma1/(1-gamma3)` |
| s<0, gamma3=1 | `d=gamma1 tau+O(1)` |
| s<0, gamma3>1 | `d ~ B exp((gamma3-1)tau)`, with B>0 |

Here q0=A0/b when s=0. Accordingly beta=k is bounded only when
`A0>(gamma3-1)b` in the supercritical compounding case. For beta<k that case
eventually diverges, even if early correction is strong. The subcritical floor
`gamma1/(1-gamma3)` applies only when beta<k, not for every exponent.

**Proof.** Equation (D) becomes `d'=gamma1-(q+a)d`. For s>0 the same z argument
as Theorem 2 gives `z'=q-(q+a-s)z` and z->1. For s=0 use 4(a) in tau. For s<0,
q tends to zero and is integrable. If a>0 use the squeeze proof of Theorem 2.
If a=0, the integrating factor exp(Q) approaches a finite positive constant;
its difference from that limit is integrable, yielding `d=gamma1 tau+O(1)`.
If a<0, put c=-a>0. The formula becomes

$$d=e^{c\tau-Q(\tau)}\left[d_0+
\gamma_1\int_0^\tau e^{-cu+Q(u)}\,du\right].$$

Q has a finite limit and the bracket approaches a finite positive value, proving
the last row. When gamma1=0 use instead the exact homogeneous solution
`d=d0 exp[-a tau-Q(tau)]`, including all zero-state exceptions.

At gamma3=1,beta<k, the instantaneous equilibrium grows as C^(k-beta),
but the trajectory grows as log C. For k>0 it is unbounded as t approaches
t_star from below, though finite at every t<t_star. At beta=k,h=0, linear
growth in tau is also logarithmic in capability, not linear in wall-clock time.

## Theorem 5 — constant-matrix stability and left-null projections

For constant real M and constant f in `x_dot=f-Mx`, a unique globally
exponentially attracting equilibrium exists exactly when

$$\min_i\operatorname{Re}\lambda_i(M)>0
\quad\Longleftrightarrow\quad
\alpha(-M)<0,$$

where the **spectral abscissa** is conventionally
`alpha(B)=max_i Re lambda_i(B)`, not the minimum. For
`M=A+(1-gamma3)r I`, the sufficient condition
`lambda_min((A+A^T)/2)>(gamma3-1)r` also implies monotone decay of the
Euclidean norm of `x-x_star`. Positive stability alone need not prevent transient
amplification. Neither assertion about a fixed equilibrium silently applies to
time-varying matrices or to a moving target.

**Proof.** The deviation is `exp(-Mt)(x0-x_star)`; Jordan blocks decay exactly
under the displayed spectral condition. With H=(M+M^T)/2,
`d||x-x_star||^2/dt=-2(x-x_star)^T H(x-x_star)`, giving the stronger norm bound.
For time-varying M, a uniform Hermitian lower bound H(t)>=mu I, mu>0,
controls homogeneous norm decay, whereas frozen positive spectra alone do not.
For instance switching for unit intervals between
`M1=[[1,-4],[0,1]]` and `M2=M1^T` gives a period propagator
`exp(-M2)exp(-M1)` with spectral radius greater than two although both frozen
matrices have eigenvalues 1,1.

**Null projection, replacement of the right-null limb.** In gain-only depth,

$$\mathbf d'=\gamma_1\mathbf c-(1-\gamma_3)\mathbf d
              -r^{-1}\mathbf A(\tau)\mathbf d.$$

For a **fixed** unit vector v satisfying `v^T A(tau)=0` at every depth,
y=v^T d obeys exactly

$$y'=\gamma_1v^T\mathbf c-(1-\gamma_3)y.$$

Its solution is the affine formula of 4(a) in tau. For gamma3<1 it tends to
`gamma1(v^T c)/(1-gamma3)`; for gamma3=1 it is linear; for gamma3>1 it generally
has an exponentially growing mode. In the non-negative scalar-projection case,
positive forcing or positive initial y ensures divergence in the last case;
with signed projections, exceptional cancelling initial data must not be excluded
without an assumption. A projected value need not be a non-negative fraction.

**Proof.** Left-multiply the full vector equation by v^T. The removed term is
`v^T A d`, so a right-null condition `A v=0` is not sufficient. For symmetric
A the two conditions agree. If v changes with depth there is an additional
`v'^T d` term. Example: `A=[[0,1],[0,2]],v=(1,0)` is right-null but the first
component still contains `-d2/r`. The revised test integrates the full two-state
system before projecting, rather than assuming the scalar reduction.

This theorem does not identify physical placement of a corrector with its
scaling exponent, nor establish that positive definiteness of A is necessary.

## Theorem 6 — OU marginal law, with its limits

Let t be wall-clock time and consider the real-valued SDE

$$\mathrm dX=(\iota-\kappa X)\mathrm dt+\sigma\mathrm dW_t,$$

with constant kappa>0, iota real, sigma>0. Put mu=iota/kappa. For Gaussian
initial state of mean m0 and variance v0, independent of future Brownian increments,

$$m(t)=\mu+(m_0-\mu)e^{-\kappa t},\qquad
v(t)=v_0e^{-2\kappa t}+{\sigma^2\over2\kappa}(1-e^{-2\kappa t}).$$

The unique stationary law is `N(mu,sigma^2/(2 kappa))`. At a **single time
in stationarity**,

$$\Pr(X>d_{crit})=\overline\Phi\left(
{d_{crit}-\mu\over \sigma/\sqrt{2\kappa}}\right).$$

For deterministic initial data use v0=0 and the finite-time mean/variance,
not the stationary law at t=0. Sigma=0 gives a deterministic, degenerate law.

**Proof.** Multiply the SDE by exp(kappa t). Brownian stochastic integration
has mean zero and variance obtained by integrating
`sigma^2 exp(-2 kappa(t-u))`; the Gaussian law follows. Moment equations
`m_dot=iota-kappa m`, `v_dot=sigma^2-2kappa v` give an independent verification.

In the model iota=gamma1 r and kappa=A+r. The Gaussian probability is not a
first-passage probability `P(sup_{0<=u<=T} X(u)>dcrit)`, an infinite-horizon
no-failure guarantee, or a certificate for a physically bounded fraction. An
unrestricted OU process has support outside [0,1]. A local fraction approximation
needs boundary-distance and small-noise checks; reflecting, absorbing, jump or
multiplicative-noise models need separate derivations.

For a *family of frozen-coefficient models*, with beta>k and
sigma=sigma0 C^eta, mu scales as C^(k-beta) and variance as C^(2eta-beta).
At fixed positive dcrit, beta>k and beta>2eta are sufficient for the stationary
single-time tail to vanish. Beta>k alone does not control arbitrarily scaling
noise. These are frozen-law comparisons, not a stationary distribution for a
time-varying process. With fixed noise, `variance/mu^2` scales as C^(beta-2k),
so the local near-zero Gaussian interpretation can deteriorate.

Under the deterministic clock change d tau=r(t) dt the noise coefficient is
sigma/sqrt(r), while drift coefficients divide by r. Omitting this Brownian
scaling would change the process. No stochastic hard-takeoff safety conclusion
is imported from the constant-coefficient theorem.

## Theorem 7 — finite-capacity equilibrium and conditional safe interval

Let beta>k>0, Amax,Cs,b>0, and gamma2=gamma3=0. Set

$$A(C)=A_{max}{C^\beta\over C^\beta+C_s^\beta},\quad q=A/(bC^k).$$

Then q tends to zero at both endpoints C->0 and C->infinity and has its unique
maximum at

$$C_{opt}=C_s\left({\beta-k\over k}\right)^{1/\beta},$$
$$q_{max}={A_{max}\over bC_s^k}{k\over\beta}
\left({\beta-k\over k}\right)^{(\beta-k)/\beta}.$$

The *instantaneous-equilibrium* minimum is m=gamma1/(1+qmax), attained at Copt.
It is not necessarily the trajectory's minimum, or the capability at which a
trajectory is safest. For any d0>=0 and tau>=tau0,

$$d(\tau)\ge m+(d_0-m)e^{-(1+q_{max})(\tau-\tau_0)}.$$

Thus d0>=m prevents a downward crossing of m, while d0=0 may start below m.
For gamma1>0 the actual trajectory tends to gamma1 at large capability: bounded
capacity does not give vanishing relative misalignment. This is not an assertion
that any particular positive floor is intrinsically unsafe. If gamma1=0,
d tends to zero, so an unqualified “no indefinite stability” assertion is false.

At k=0 with beta>0, q tends to Amax/b and d tends to
`gamma1/(1+Amax/b)`. If Amax=a0 C^beta_cap, with beta>0,Cs fixed, the ultimate
ratio is asymptotic to `(a0/b) C^(beta_cap-k)`. The three limits in Theorem 2
hold with beta_cap replacing beta; its necessity iff still requires gamma1>0.

**Proof.** The logarithmic derivative of q is
`beta Cs^beta/(C^beta+Cs^beta)-k`, which decreases strictly from beta-k to -k.
Solving for its zero and substituting gives Copt and qmax. Since d>=0,
`d' >= gamma1-(1+qmax)d`; scalar comparison proves the lower bound for every
non-negative initial state. The limits follow from the actual depth equation
and its asymptotic coefficient, as in Theorem 2, not from assuming exact
tracking of a moving equilibrium.

For a prescribed model-index threshold 0<dcrit<gamma1, suppose q>=
`gamma1/dcrit-1` throughout an interval and d at the interval's start is at most
dcrit. Then d remains at most dcrit on that interval: the vector field at the
upper boundary is non-positive. Both the threshold and the initial-state
condition are required. A large qmax alone does not certify a safe trajectory.

## Replacement for Remark 4 — time-varying exponents

For positive, locally absolutely continuous A and r along an increasing
capability path, define *local logarithmic slopes*
`beta(tau)=d log A/dtau`, `k(tau)=d log r/dtau`. Then

$$q(\tau)=q(0)\exp\left(\int_0^\tau[\beta(u)-k(u)]\,du\right).$$

Do not use this formula by substituting varying powers in `C^beta(C)` without
the derivative terms that substitution entails. In the gain-only model, q->infinity
is **sufficient** for d->0. Thus divergence of the cumulative local margin to
positive infinity is sufficient, and an eventual strictly positive lower bound
on that margin is sufficient. Necessity is not claimed for arbitrary varying
coefficients (narrow, recurring dips can make pointwise q fail to tend to infinity
without preventing d from tending to zero).

The exact necessary-and-sufficient statement for gamma1>0 is instead

$$d(\tau)\to0\ \Longleftrightarrow\
\int_0^\tau \exp\left[-\int_u^\tau(1+q(v))\,dv\right]\,du\to0.$$

**Proof.** Apply (V) in depth. The homogeneous term is at most d0 exp(-tau),
so it vanishes; the non-negative forced term is gamma1 times the displayed
kernel integral. For sufficiency of q->infinity use the comparison proof of
Theorem 2. Eventual monotonicity of q is one additional condition under which
q->infinity also becomes necessary: otherwise q has a finite limit and the
limiting forced equilibrium is positive.

## Relative versus absolute misalignment — a corollary, not a safety law

Under the positive-gain-drift assumptions of (L), and **within beta>k**,

$$D=Cd\sim{\gamma_1b\over A_0}C^{1+k-\beta}.$$

Accordingly absolute D diverges for k<beta<k+1, tends to gamma1 b/A0 at
beta=k+1, and vanishes for beta>k+1. This does not refute the relative-index
criterion or equate D with actual harm. It prevents the relative-index theorem
from being described as a guarantee of vanishing absolute damage. Neither
inequality establishes survival, welfare or a catastrophic-risk bound.

## Verification and publication boundaries

`code/test_theorems_independent.py` is an independently implemented numerical
regression suite, with closed forms, original-equation/depth-clock comparisons,
full-matrix projection checks, moment equations, an independent-path OU ensemble,
and finite-capacity optimisation/trajectory checks. Failed, truncated,
event-terminated and non-finite solver outputs are rejected. Assertions about
asymptotic rates sample finite domains; the proofs above, not a finite test run,
carry the infinite-limit claims. Statistical envelopes are fixed in advance of
the recorded runs. The suite does not import the original harness.

Historical pilots, their limitations and their withdrawn evidentiary status are
not changed by this mathematical correction. No external-versus-embedded or
same-class-versus-cross-class superiority follows merely from these equations.
No mathematical or empirical claim is certified “perfect”.

Implementation references: SciPy's primary documentation for
[solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html)
and [matrix exponentials](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.expm.html).
The mathematical arguments in this correction are derived explicitly above.
