---
title: "Neural ODE: Neural Ordinary Differential Equations"
paper:
  title: "Neural Ordinary Differential Equations"
  authors: "Ricky T. Q. Chen et al."
  venue: "NeurIPS 2018"
  arxiv: "1806.07366"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "stochastic-modeling"
order: 1
tags: [neural-ode, adjoint-method, continuous-depth, normalizing-flows, continuous-time, time-series, latent-variable, instantaneous-change-of-variables]
date: 2018-07-01
status: draft
summary: "Replace a stack of residual layers with a learned vector field integrated by a black-box ODE solver, and train it with the adjoint method so that memory does not grow with depth."
---

## Abstract

A residual block updates its hidden state by a small additive step, which is Euler integration of an unnamed flow. This paper takes the limit seriously: a network defines the *time derivative* of the state, and an off-the-shelf solver produces the output. The technical core is a way to differentiate through that solver without looking inside it — the gradients are themselves the solution of a second ODE run backwards in time, so memory does not grow with the number of solver steps. Three uses follow: a continuous-depth classifier, a normalizing flow whose log-density change is a trace instead of a log-determinant, and a latent-variable time-series model that accepts observations at arbitrary real times. The experiments (MNIST, 2-D densities, synthetic spirals) are proofs of concept, and most headline numbers come from a single run.

**Keywords:** neural ODE, adjoint sensitivity method, continuous normalizing flow, instantaneous change of variables, latent ODE, irregular time series

## 1 Introduction

Residual nets, RNN decoders and normalizing flows all share the update $h_{t+1} = h_t + f(h_t, \theta_t)$, which earlier work had already read as an Euler discretisation of a continuous transformation. The question here is what is gained by dropping the discretisation and handing the initial value problem to a numerical solver.

Four answers are offered. *Memory*: if gradients can be obtained without storing the forward pass, training memory is constant in "depth" — the bottleneck that limits how deep a residual net can be trained. *Adaptive computation*: modern solvers monitor their own local error and choose step sizes, so compute follows the difficulty of each input, and the tolerance can be loosened after training for cheap inference. *Flows*: in continuous time the change-of-variables formula collapses from a determinant to a trace, which removes the architectural contortions (triangular Jacobians, single-unit layers, fixed dimension splits) that discrete flows use to keep the determinant cheap. *Time series*: a state defined at every real-valued time does not need observations binned onto a grid.

All four hinge on the same obstacle: training. Backpropagating through each internal operation of the solver costs memory proportional to the number of steps, accumulates numerical error in the gradient, and is awkward for implicit solvers, which run an inner nonlinear solve at every step. The paper's contribution is to avoid that entirely.

## 2 Background

The model is an initial value problem. With input $h(0)$, the output is $h(T)$ where

$$
\frac{dh(t)}{dt} = f(h(t), t, \theta)
\tag{1}
$$

and $f$ is a neural network whose parameters $\theta$ are shared across all $t$ — unlike [ResNet](/blog/resnet/), where each block has its own weights, so "depth" here buys integration time rather than parameters. Existence and uniqueness follow from Picard's theorem when $f$ is uniformly Lipschitz in the state and continuous in time, which holds for finite weights with tanh or ReLU nonlinearities. Uniqueness is what makes the map $h(0) \mapsto h(T)$ invertible for free: two trajectories cannot cross, so running the solver backwards recovers the input. This is also the reason [Score-SDE](/blog/score-sde/)'s probability-flow ODE and [Flow Matching](/blog/flow-matching/) can treat a learned vector field as a bijection without constraining its architecture.

## 3 Method

> **Key idea.** Treat the ODE solver as an opaque function. The gradients of the loss with respect to the initial state and the parameters are themselves the solution of another ODE, so one extra backwards solve replaces backpropagation through the solver's internals.

### 3.1 The adjoint, derived

Let the loss be $L(z(t_1))$ with $z(t_1) = \mathrm{ODESolve}(z(t_0), f, t_0, t_1, \theta)$, and define the adjoint $a(t) = \partial L / \partial z(t)$: the sensitivity of the final loss to a perturbation of the state at time $t$. Write the flow over a short interval as $z(t + \varepsilon) = T_\varepsilon(z(t), t)$. The ordinary chain rule links two instants,

$$
a(t) = a(t+\varepsilon)\, \frac{\partial T_\varepsilon(z(t), t)}{\partial z(t)} ,
\tag{2}
$$

which is exactly the discrete backprop recursion $\partial L/\partial h_t = (\partial L/\partial h_{t+1})(\partial h_{t+1}/\partial h_t)$ written in continuous time. Expanding $T_\varepsilon(z) = z + \varepsilon f(z, t, \theta) + O(\varepsilon^2)$ gives $\partial T_\varepsilon/\partial z = I + \varepsilon\, \partial f/\partial z + O(\varepsilon^2)$; substituting into the definition of the derivative,

$$
\frac{da(t)}{dt} = \lim_{\varepsilon \to 0^+} \frac{a(t+\varepsilon) - a(t+\varepsilon)\big(I + \varepsilon\, \partial f/\partial z + O(\varepsilon^2)\big)}{\varepsilon} = -a(t)^{\top} \frac{\partial f(z(t), t, \theta)}{\partial z} .
\tag{3}
$$

The $O(\varepsilon^2)$ terms vanish in the limit, so <mark>(3) is exact, not an approximation — every error in an adjoint gradient is numerical, introduced by the solver, never by the formula.</mark> Starting from $a(t_1)$, which ordinary backprop through the loss head supplies, integrating (3) back to $t_0$ gives $\partial L/\partial z(t_0)$.

### 3.2 One augmented state for the parameters and the times

The parameter gradient comes from the same machinery rather than a separate derivation. Treat $\theta$ and $t$ as extra state variables with trivial dynamics, $d\theta/dt = 0$ and $dt/dt = 1$, and stack them: $z_{\text{aug}} = [z, \theta, t]$. The Jacobian of the augmented dynamics has $\partial f/\partial z$, $\partial f/\partial \theta$, $\partial f/\partial t$ in its first block row and zeros elsewhere, so applying (3) to the augmented system splits into three scalar-free equations. Integrating the $\theta$ component from the terminal condition $a_\theta(t_1) = 0$ gives

$$
\frac{dL}{d\theta} = -\int_{t_1}^{t_0} a(t)^{\top} \frac{\partial f(z(t), t, \theta)}{\partial \theta}\, dt ,
\qquad
\frac{dL}{dt_1} = a(t_1)^{\top} f(z(t_1), t_1, \theta) .
\tag{4}
$$

Both integrands are vector–Jacobian products, which reverse-mode autodiff evaluates at roughly the cost of one call to $f$ — no Jacobian is ever materialised. The state $z(t)$ that they need is not stored: it is re-integrated backwards from $z(t_1)$ alongside $a$ and the running parameter gradient, all concatenated into one augmented state. <mark>One backwards solver call therefore returns every gradient, with memory independent of the number of solver steps.</mark> This is the one step that is *not* exact in practice: the backwards-reconstructed $z(t)$ can drift from the forward trajectory, and re-integration, not the adjoint identity, is where the method can silently lose accuracy. When the loss touches intermediate times, the backward solve is broken at each observation and the adjoint receives a jump of $\partial L/\partial z(t_i)$ there.

### 3.3 Intuition: the one-dimensional linear flow

Take $D = 1$, $f(z, t, \theta) = \theta z$, and $L = z(t_1)$, so $z(t_1) = z_0 e^{\theta \Delta}$ with $\Delta = t_1 - t_0$. The adjoint equation reads $da/dt = -\theta a$ with $a(t_1) = 1$, whose backwards solution is $a(t) = e^{\theta(t_1 - t)}$. Plugging into (4),

$$
\frac{dL}{d\theta} = \int_{t_0}^{t_1} a(t)\, z(t)\, dt = \int_{t_0}^{t_1} e^{\theta(t_1-t)} z_0 e^{\theta(t - t_0)}\, dt = \Delta\, z_0 e^{\theta \Delta} = \Delta\, z(t_1),
\tag{5}
$$

which is what differentiating the closed form $z_0 e^{\theta\Delta}$ gives directly. The integrand happens to be constant in $t$ here; the general lesson is that the adjoint carries sensitivity backwards while the state carries information forwards, and the parameter gradient is their product accumulated along the path.

### 3.4 Continuous normalizing flows

A discrete flow $z_1 = f(z_0)$ changes log-density by $-\log|\det \partial f/\partial z_0|$, and that determinant costs $O(D^3)$ in the dimension or the hidden width. Theorem 1 of the paper shows that if the state follows $dz/dt = f(z(t), t)$ with $f$ uniformly Lipschitz in $z$ and continuous in $t$, then

$$
\frac{\partial \log p(z(t))}{\partial t} = -\mathrm{tr}\!\left(\frac{\partial f}{\partial z(t)}\right) .
\tag{6}
$$

The proof takes the $\varepsilon \to 0$ limit of the discrete formula, uses L'Hôpital's rule and Jacobi's formula for the derivative of a determinant, and ends with the same Taylor expansion of $T_\varepsilon$ as in 3.1. <mark>A log-determinant is replaced by a trace, and $f$ need not be bijective by construction</mark>, since uniqueness of solutions makes the whole transformation invertible. Appendix A.2 identifies (6) as the Liouville equation — Fokker–Planck with zero diffusion — evaluated *along a particle's trajectory* rather than at a fixed point in space, which is why it becomes a $(D{+}1)$-dimensional ODE instead of a PDE needing a grid exponential in $D$. That reframing is what later lets [Score-SDE](/blog/score-sde/) compute exact likelihoods for a diffusion model.

Because the trace is linear, dynamics written as a sum $\sum_{n=1}^{M} f_n(z)$ have a log-density derivative that is the sum of the traces, so a "wide" flow costs $O(M)$ where a discrete one costs $O(M^3)$ — the reason standard normalizing flows stack many single-unit layers. In every experiment the authors use the planar form $f(z) = u\,h(w^{\top} z + b)$, whose Jacobian is an outer product, so its trace is an inner product $u^{\top} \partial h/\partial z$ and costs nothing. They add a learned gate $\sigma_n(t) \in (0,1)$ per unit to make the dynamics time-dependent, and call the result a continuous normalizing flow (CNF). The reverse map costs about as much as the forward one, so the same model trains by maximum likelihood and then samples.

### 3.5 Latent ODE for time series

Each series is summarised by an initial latent state; a shared, time-invariant vector field carries it forward:

$$
z_{t_0} \sim p(z_{t_0}), \quad z_{t_1},\dots,z_{t_N} = \mathrm{ODESolve}(z_{t_0}, f, \theta_f, t_0,\dots,t_N), \quad x_{t_i} \sim p(x \mid z_{t_i}, \theta_x) .
\tag{7}
$$

Training is a VAE: an RNN reads the observations in reverse time order and outputs $q_\phi(z_{t_0} \mid x_{t_1..t_N})$, and Appendix E maximises $\sum_i \log p(x_{t_i}\mid z_{t_i},\theta_x) + \log p(z_{t_0}) - \log q_\phi(z_{t_0}\mid\cdot)$ with $p(z_{t_0}) = \mathcal{N}(0, I)$. Because $f$ is time-invariant, any point on the trajectory determines the whole of it, so extrapolation is just solving further. The computation graph is [Fig. 6 in the paper](https://arxiv.org/pdf/1806.07366#page=6). A Poisson likelihood for the observation times, with intensity $\lambda(z(t))$, can be added; its integral is appended to the ODE state so the trajectory and the likelihood come from one solver call. [Latent ODE](/blog/latent-ode/) replaces the RNN encoder here and takes this further.

### 3.6 Algorithm

```
# forward
z_T = ODESolve(z_0, f, t_0, t_1, theta)          # solver internals not recorded

# backward: one augmented solve, no stored activations
s_1 = [ z_T , dL/dz_T , zeros_like(theta) ]
def aug(s, t):
    z, a, _ = s
    return [ f(z, t, theta),
             -vjp(f, z, a) wrt z,                # = -a^T df/dz
             -vjp(f, z, a) wrt theta ]           # = -a^T df/dtheta
[z_0_hat, dL/dz_0, dL/dtheta] = ODESolve(s_1, aug, t_1, t_0, theta)

# with a loss at intermediate times t_N > ... > t_1:
for i = N .. 1:
    solve aug backwards over [t_i, t_{i-1}]
    a <- a + dL/dz(t_{i-1})                      # jump at each observation
```

## 4 Implementation notes

| Setting | As reported |
|---|---|
| Solver (classification) | implicit Adams (LSODE / VODE) via `scipy.integrate` |
| Gradient path | adjoint implemented in `autograd`; $f$ and its derivatives evaluated on GPU in TensorFlow, called from the Fortran solver from Python |
| Tolerance, sequence models | 1.5e-8 (solver default) |
| Tolerance, classification / density | 1e-3 / 1e-5, reported as not degrading performance |
| MNIST body | downsample twice, then 6 residual blocks, replaced by one ODESolve module |
| CNF density matching | CNF: Adam, 10,000 iterations; discrete NF: RMSprop, 500,000 iterations |
| CNF max-likelihood | 64 hidden units (CNF) versus 64 stacked one-unit layers (NF) |
| Latent ODE (spirals) | 4-D latent; RNN encoder, 25 units; $f$ and decoder each one hidden layer, 20 units |
| Spiral data | 1,000 noisy 2-D spirals, 100 equally spaced steps, $n \in \{30, 50, 100\}$ points subsampled without replacement |
| Optimiser, LR, batch size (MNIST) | not stated |
| Compute / wall-clock | not stated |

Two details are easy to get wrong when reproducing. First, an implicit solver was chosen *because* direct backpropagation through it is hard, so the RK-Net baseline is not the same solver with a different gradient rule — it is a different integrator. Second, the tolerance is not a free accuracy knob: it is also the numerical accuracy of the gradient, and the backward tolerance must be set separately from the forward one. A public PyTorch implementation with GPU solvers (`torchdiffeq`) is referenced by the paper.

## 5 Experiments

**Supervised learning.** $L$ is the number of layers in the ResNet and $\tilde{L}$ the number of function evaluations (NFE) that the solver chooses per forward pass.

| Model | Test error | # Params | Memory | Time |
|---|---|---|---|---|
| 1-layer MLP (LeCun et al., 1998) | 1.60% | 0.24 M | – | – |
| ResNet | 0.41% | 0.60 M | $O(L)$ | $O(L)$ |
| RK-Net | 0.47% | 0.22 M | $O(\tilde{L})$ | $O(\tilde{L})$ |
| **ODE-Net** | 0.42% | 0.22 M | $O(1)$ | $O(\tilde{L})$ |

**Latent ODE on spirals.** RMSE is measured on 100 time points *beyond* those used for training, so this table scores extrapolation, not reconstruction. The baseline is a 25-unit RNN trained on negative Gaussian log-likelihood, also tried with the time gap appended to its input.

| Predictive RMSE | 30/100 | 50/100 | 100/100 |
|---|---|---|---|
| RNN | 0.3937 | 0.3202 | 0.1813 |
| **Latent ODE** | **0.1642** | **0.1502** | **0.1346** |

**Claim by claim.**

- *Constant memory in depth.* Supported by derivation, not by measurement: the memory column of Table 1 is asymptotic notation, and no memory benchmark against the ResNet or RK-Net is reported.
- *Accuracy matches a residual net.* Table 1 gives 0.42% against 0.41% on MNIST. <mark>One dataset, one run, no error bars — a 0.01-point gap is well inside seed noise, so "comparable" is the safe reading and nothing stronger is established.</mark>
- *Explicit accuracy–speed dial.* The best-supported claim: [Fig. 3 in the paper](https://arxiv.org/pdf/1806.07366#page=4) shows numerical error tracking the requested tolerance and forward time proportional to NFE, both over several orders of magnitude.
- *The adjoint is cheaper than backprop through the solver.* Backward NFE is about half of forward NFE (Fig. 3c). This is a proxy: NFE is not wall-clock, and the comparison against RK-Net is inferred rather than timed.
- *Depth is emergent.* NFE rises over training (Fig. 3d), read as the dynamics becoming more complex. Suggestive, and also a warning — <mark>cost grows during training and the modeller does not control it.</mark>
- *CNFs are at least as expressive as discrete flows at $K = M$.* Figure 4d shows lower loss on 2-D targets, and maximum-likelihood training transports mass smoothly where the discrete flow struggles on two moons. Confounded: the two models were trained with different optimisers and a 50× difference in iteration budget, so expressivity and trainability are not separated.
- *Better on irregular time series.* Table 2, one synthetic dataset, one seed, no error bars; the gap is widest with the fewest observations (0.1642 vs 0.3937 at 30/100), which is at least the predicted direction, and the latent space separates clockwise from counter-clockwise spirals.
- *Parameter efficiency.* Not claimed in the text — the acknowledgements thank a reader "for pointing out an unsupported claim about parameter efficiency" — yet the 0.22 M / 0.60 M column survives in Table 1. Read that column as a description of two unmatched architectures, not as a result.

## 6 Limitations

**Stated by the authors.** Minibatching concatenates batch elements into one combined ODE of dimension $D \times K$, so error control is shared and could in principle require $K$ times more evaluations, though they did not observe this. Uniqueness requires $f$ to be Lipschitz, which finite weights and tanh/ReLU give. Tolerances must be chosen for both passes. Re-integrating $z(t)$ backwards can diverge from the forward trajectory; checkpointing is the suggested remedy, and the check that it was not needed was informal ("we informally checked" that reversing many CNF layers recovered the initial states).

**My reading.** No wall-clock comparison anywhere, which is what "time $O(\tilde{L})$" actually costs. The trace in (6) is only cheap for the planar dynamics used here; for a general $f$ in $D$ dimensions an exact trace needs $D$ backward passes, so the linear-in-width argument does not transfer to a deep dynamics network — the gap that stochastic trace estimation later fills. Nothing is run beyond MNIST, 2-D toy densities and synthetic spirals, and no result is repeated across seeds. The time-series model uses a time-invariant $f$, so it cannot represent a non-stationary system. And the latent trajectory is deterministic given $z_{t_0}$: all randomness lives in the initial state and the observation noise, which is the wrong shape for any system that is continually perturbed — the limitation that motivates the rest of this series.

## 7 Extensions

**What was built on this.** [Latent ODE](/blog/latent-ode/) replaces the RNN recognition network with an ODE-RNN and tests the time-series model on real data. [Stochastic Adjoint](/blog/scalable-sde-gradients/) extends the construction to SDEs, where backwards reconstruction is much harder because Brownian paths must be reproduced; [Neural JSDE](/blog/neural-jump-sde/) adds jumps at event times, and [SDE-GAN](/blog/neural-sde-gan/) trains continuous-time generators adversarially. On the generative side, [Score-SDE](/blog/score-sde/)'s probability-flow ODE is equation (6) applied to a diffusion's marginals and [DDIM](/blog/ddim/) discretises that same deterministic flow, while [Rectified Flow](/blog/rectified-flow/) and [Flow Matching](/blog/flow-matching/) keep the ODE but train the field by regression, avoiding the trace and the solver at training time. FFJORD (Hutchinson's estimator for an unbiased stochastic trace) and Augmented Neural ODEs (extra state dimensions for flows a non-crossing trajectory cannot express) answer the two limitations above (from general knowledge, unverified).

**Open problems.** How to control NFE rather than watch it grow; how to regularise the learned vector field so that the solver stays cheap without losing accuracy; how to get an exact trace in high dimension; how to bound the error the backwards reconstruction of $z(t)$ introduces into the gradient, rather than checking it informally.

**Research directions.** *These are ideas, not results — none has been run.*

1. **Penalise the cost of the solver.** Hypothesis: adding $\|f(z,t)\|^2$ or $\|\partial f/\partial z\|_F^2$ to the loss lowers NFE at matched test error, because the solver's step size is set by the stiffness of the learned field. Data: MNIST plus the spiral set. Baseline: the same ODE-Net without the penalty, compared at matched tolerance. Metric: test error against total NFE and wall-clock, not against epochs. Failure mode: the penalty suppresses the curvature the model needs, and error rises before NFE falls; the diagnostic is whether the error–NFE frontier moves at all or merely slides along itself.
2. **Continuous-time density for event-driven market data.** Hypothesis: a CNF over trade-to-trade log returns, conditioned on a latent ODE state driven by irregular arrival times, fits the tails of intraday returns better than any model that first bins to a fixed grid. Data: an irregular per-trade series with timestamps. Baseline: a GARCH or HAR model on 5-minute bins and a discrete flow on the same bins. Metric: out-of-sample log-likelihood per event and coverage of the 1% / 99% quantiles. Failure mode: the deterministic latent path cannot carry ongoing shocks, so the flow absorbs all randomness into the marginal — expect calibrated unconditional tails but no volatility persistence, visible as autocorrelation in squared residuals the model fails to reproduce. That failure is the argument for latent SDEs rather than latent ODEs, and it is exactly what [Stochastic Adjoint](/blog/scalable-sde-gradients/) makes trainable.

## 8 Takeaways

- A network can define a vector field instead of a layer; the solver decides how many evaluations are needed, which gives an explicit accuracy–speed dial but removes control over cost.
- The adjoint identity is exact; only re-integrating the state backwards and the solver tolerance introduce error. That is where reproduction problems come from, not from the derivation.
- In continuous time the density change of a flow is a trace, not a determinant, so invertible generative models stop needing architectural tricks — the identity that later underwrites exact likelihoods for diffusion models.
- The evidence is thin by modern standards: one image dataset, toy densities, synthetic spirals, no seeds, no wall-clock. This is a paper about a building block, and it should be read as one.
- <mark>For financial time series the relevant part is Section 5: trades, quotes and corporate events arrive at irregular times, and a continuous-time latent state avoids binning them.</mark> The deterministic latent path is equally relevant as a warning, since price dynamics are driven by ongoing noise rather than fixed by an initial condition — which is why this note is the first stop in a series that ends up at SDEs.

## References

1. R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud. *Neural Ordinary Differential Equations.* NeurIPS 2018. arXiv:1806.07366.
2. K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. arXiv:1512.03385.
3. D. J. Rezende, S. Mohamed. *Variational Inference with Normalizing Flows.* ICML 2015. arXiv:1505.05770.
4. L. S. Pontryagin et al. *The Mathematical Theory of Optimal Processes.* 1962 (adjoint sensitivity method).
5. Y. Rubanova, R. T. Q. Chen, D. Duvenaud. *Latent ODEs for Irregularly-Sampled Time Series.* NeurIPS 2019. arXiv:1907.03907 (next note in this series).
