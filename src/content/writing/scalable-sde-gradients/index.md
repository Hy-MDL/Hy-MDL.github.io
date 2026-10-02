---
title: "Stochastic Adjoint: Scalable Gradients for Stochastic Differential Equations"
paper: { title: "Scalable Gradients for Stochastic Differential Equations", authors: "Xuechen Li et al.", venue: "AISTATS 2020", arxiv: "2001.01328", license: "arxiv.org/licenses/nonexclusive-distrib/1.0/" }
series: "stochastic-modeling"
order: 3
tags: [neural-sde, adjoint-method, stratonovich, brownian-tree, latent-sde, variational-inference, girsanov, torchsde]
date: 2020-01-01
status: draft
summary: "Extends the constant-memory adjoint trick of neural ODEs to SDEs by running a Stratonovich SDE backwards in time, rebuilds the forward noise from a single seed with a virtual Brownian tree, and uses the result to train latent SDEs by variational inference."
---

## Abstract

Neural ODEs became practical because the adjoint sensitivity method returns parameter gradients by solving a second ODE backwards, so nothing from the forward pass needs to be kept. This paper does the same for SDEs. In Stratonovich form the inverse of the stochastic flow solves a backward SDE whose coefficients differ from the forward ones only by a sign, and differentiating that inverse flow gives an adjoint process built entirely from vector–Jacobian products. Because the backward solve must see the same Brownian path as the forward solve, the authors add a data structure — the virtual Brownian tree — that regenerates the path at any query time from a single seed at $O(1)$ memory. They then place an SDE prior and an SDE posterior inside a variational autoencoder, the latent SDE, whose whole objective is differentiated with one forward and one backward solve. On a 50-dimensional motion capture benchmark it reaches a test MSE of 4.03, the best among the compared dynamics models.

**Keywords:** neural SDE, adjoint sensitivity, Stratonovich integral, stochastic flow of diffeomorphisms, virtual Brownian tree, latent SDE, Girsanov, variational inference

## 1 Introduction

Fitting an SDE with neural drift and diffusion requires the gradient of a path functional with respect to many parameters, and before this paper there were two routes, both scaling badly. The **pathwise** or forward-sensitivity approach propagates a Jacobian of size (parameters)² or (parameters)×(states) alongside the state, so its time cost grows with the total dimension $D$ of states plus parameters. **Backpropagating through the operations of the solver** — the approach Giles and Glasserman introduced for fast Monte Carlo Greeks in the LIBOR market model — is fast per step but stores every intermediate quantity, so memory grows with the number of steps $L$.

The paper's complexity table sets the target. Counting both memory and time in units of one evaluation of drift and diffusion, <mark>forward pathwise costs $O(1)$ memory and $O(LD)$ time, backprop through the solver costs $O(L)$ in both, and the stochastic adjoint costs $O(1)$ memory and $O(L\log L)$ time</mark>. The extra $\log L$ is the price of reconstructing noise instead of storing it.

Two obstacles stand in the way. The first is mathematical: Itô integrals use a left-endpoint rule that privileges the forward direction of time, so a naive reversal does not retrace the forward path. The second is computational: an adaptive backward solver queries the noise at times the forward solver never visited, so a cache of increments is not enough.

## 2 Background

For the ODE adjoint — augmenting the state with $\partial \mathcal{L}/\partial z_t$ and $\partial\mathcal{L}/\partial\theta$, then integrating that system in reverse — see [Neural ODE](/blog/neural-ode/); for the variational-autoencoder framing with a latent trajectory and an encoder that reads the observations, see [Latent ODE](/blog/latent-ode/). This note assumes both.

An Itô SDE for a state $Z_t \in \mathbb{R}^d$ driven by an $m$-dimensional Wiener process is

$$
Z_T = z_0 + \int_0^T b(Z_t, t)\,dt + \sum_{i=1}^m \int_0^T \sigma_i(Z_t, t)\,dW^{(i)}_t, \tag{1}
$$

with drift $b$ and diffusion columns $\sigma_i$. The Stratonovich integral $\circ\,dW_t$ replaces each cell's left endpoint $Y_{t_{k-1}}$ by the average $(Y_{t_k}+Y_{t_{k-1}})/2$; the two integrals differ by a finite-variation correction, so Itô and Stratonovich models convert into each other. Throughout, $b$ and $\sigma$ are assumed to have infinitely many bounded state derivatives and bounded first time derivatives ($b,\sigma\in C_b^{\infty,1}$), which buys existence, uniqueness and smooth dependence on the starting point.

The **backward Wiener process** is $\check W_t = W_t - W_T$, adapted to the backward filtration $\{\mathcal{F}_{t,T}\}$, with the backward Stratonovich integral defined over a partition running from $T$ down to $0$. Kunita's theorem then says that (1), read in Stratonovich form, generates a stochastic flow of diffeomorphisms $\Phi_{s,t}$, and that the inverse flow $\check\Psi_{s,t} = \Phi_{s,t}^{-1}$ solves

$$
\check\Psi_{s,t}(z) = z - \int_s^t b(\check\Psi_{u,t}(z), u)\,du - \int_s^t \sigma(\check\Psi_{u,t}(z), u)\circ d\check W_u. \tag{2}
$$

Forward and backward coefficients differ by a sign and nothing else. <mark>This sign symmetry is exactly what the Stratonovich convention buys</mark>: for an Itô SDE, negating drift and diffusion and integrating from the end state gives a visibly wrong reconstruction ([Fig. 2 in the paper](https://arxiv.org/pdf/2001.01328#page=3)).

## 3 Method

> **Key idea.** In Stratonovich form the inverse flow is itself an SDE with the same coefficients up to sign. Differentiate that inverse flow once and you get a linear backward SDE for $\partial\mathcal{L}/\partial Z_t$ whose every term is a vector–Jacobian product — the ODE adjoint with an extra diffusion line and a time-reversed copy of the *same* Brownian path.

### 3.1 From the inverse flow to the adjoint

Let $\mathcal{L}(Z_T)$ be a scalar loss on the terminal state and $J_{s,t}(z) = \nabla\check\Psi_{s,t}(z)$ the Jacobian of the inverse flow. Differentiating (2) under the stochastic integral — legitimate because the coefficients are $C_b^{\infty,1}$ — gives a linear backward SDE for $J$, and the Stratonovich form of Itô's lemma applied to $K_{s,t} = J_{s,t}^{-1}$ flips the signs:

$$
K_{s,t}(z) = I_d + \int_s^t K_{r,t}(z)\,\nabla b(\check\Psi_{r,t}(z), r)\,dr + \int_s^t K_{r,t}(z)\,\nabla\sigma(\check\Psi_{r,t}(z), r)\circ d\check W_r. \tag{3}
$$

$K$ is the Jacobian of the *forward* map evaluated along the reconstructed path, $K_{s,t}(z) = \nabla\Phi_{s,t}(\check\Psi_{s,t}(z))$. Left-multiplying by the constant row vector $\nabla\mathcal{L}(z)$ and setting $\check A_{s,t}(z) = \nabla\mathcal{L}(z) K_{s,t}(z)$ turns (3) into the system the algorithm actually solves:

$$
\check A_{s,t}(z) = \nabla \mathcal{L}(z) + \int_s^t \check A_{r,t}(z)\,\nabla b(\check\Psi_{r,t}(z), r)\,dr + \int_s^t \check A_{r,t}(z)\,\nabla \sigma(\check\Psi_{r,t}(z), r)\circ d\check W_r, \tag{4}
$$

solved jointly with (2). The drift term is a vector–Jacobian product with $\nabla b$, the diffusion term the same with $\nabla\sigma$; neither Jacobian is ever formed, and $\check\Psi$ is recomputed rather than stored, which is where the constant memory comes from. Every step so far is exact.

**The adaptedness problem.** Lemma 3.1 of the paper fixes the endpoint $z$ and treats it as deterministic. In training the endpoint is $Z_T = \Phi_{0,T}(z_0)$, a function of the whole noise path, so $\check A$ is no longer adapted to the backward filtration and is not the solution of a backward SDE in the usual sense. The authors sidestep this rather than patch it: since (4) admits a strong solution, there is a deterministic measurable map $F$ — an Itô map — with $\check A_{0,T}(z) = F(z, W_\cdot)$, and likewise a map $G$ for the forward flow. Then <mark>the gradient is the plain composition $A_{0,T}(z) = F(G(z,W_\cdot), W_\cdot)$ for $\mathbb{P}$-almost every noise realisation</mark> — a statement about two deterministic functions of one sample path, so no filtration is involved.

**Parameters.** As in neural ODEs, $\theta$ joins the state with zero drift and zero diffusion; its adjoint accumulates $\check A^z_s \,\partial b/\partial\theta$ and $\check A^z_s\,\partial\sigma/\partial\theta$. The pseudocode then differs from the ODE adjoint in three places: an augmented diffusion function, a Brownian sample replicated across the three blocks of the augmented state, and a sign flip in time ([Fig. 1 in the paper](https://arxiv.org/pdf/2001.01328#page=3)).

### 3.2 Convergence of the discretised algorithm

With numerical schemes $F_h$, $G_h$ on a mesh $h = T/L$, the algorithm returns $F_h(G_h(z,W_\cdot),W_\cdot)$. Theorem 3.3 gives two sufficient conditions: (i) both schemes converge in probability to their continuous maps as $h\to 0$, and (ii) for every $M>0$, $\sup_{|z|\le M}|F_h(z,W_\cdot) - F(z,W_\cdot)| \to 0$ in probability. Condition (i) is standard — Itô–Taylor schemes converge pathwise from a fixed start. Condition (ii) asks for convergence *uniform over starting points*, which the numerics literature does not study because it fixes the initial condition, and which is needed here precisely because the backward solver's starting point $G_h(z,W_\cdot)$ is random and moves with $h$.

The authors prove (ii) for Euler–Maruyama, applying a mean-square bound of order $h^{2p_2-1}$ (with $p_2=1.0$) to the augmented process $(F,\nabla_z F)$ and a Sobolev inequality to turn a weighted $L^2$ bound into a uniform one. <mark>The proof is written for $d=1$</mark>; extending it needs an $L^p$ estimate with $p>d$ whose dependence on the starting point is explicit, which the authors say is not available. No rates are proved, and no other scheme is covered.

### 3.3 Why diagonal noise matters

For general matrix-valued $\sigma$, strong order above $1/2$ needs iterated integrals $\int_0^t\int_0^s dW^{(i)}_u dW^{(j)}_s$ with $i\ne j$ — Lévy areas — which are expensive to simulate. Appendix 9.5 gives the escape. If the forward SDE has diagonal noise ($m=d$, each $\nabla\sigma_i$ non-zero only at entry $(i,i)$), the augmented backward state $(\check Z_t, \check A^z_t, \check A^\theta_t)$ has a $(2d+p)\times d$ diffusion matrix that is *not* diagonal, yet a case check over its three row blocks shows it satisfies the commutativity condition

$$
\sum_{i=1}^{d}\Sigma_{i,j_2}(x)\,\frac{\partial \Sigma_{k,j_1}(x)}{\partial x_i} = \sum_{i=1}^{d}\Sigma_{i,j_1}(x)\,\frac{\partial \Sigma_{k,j_2}(x)}{\partial x_i}. \tag{5}
$$

Under (5) the iterated integrals appear only in the symmetric combination $\int\!\int dW^{(i)}dW^{(j)} + \int\!\int dW^{(j)}dW^{(i)} = \Delta W^{(i)}\Delta W^{(j)}$, which is free. So <mark>Milstein and stochastic Runge–Kutta schemes of strong order 1.0 are available whenever the forward model has diagonal noise</mark>, with a Milstein backward step whose number of `vjp` calls is independent of $d$.

### 3.4 Virtual Brownian tree

Lévy's Brownian bridge says that given $w_s$ at $t_s$ and $w_e$ at $t_e$, the value at an interior time $t$ is Gaussian:

$$
W_t \mid w_s, w_e \sim \mathcal{N}\!\left(\frac{(t_e - t)\,w_s + (t - t_s)\,w_e}{t_e - t_s},\; \frac{(t_e - t)(t - t_s)}{t_e - t_s}\, I_d\right). \tag{6}
$$

Applying (6) recursively at midpoints builds a Brownian tree. Storing it is expensive; the trick is to make every node reproducible. Each node carries a key from a splittable PRNG, a child's key is derived deterministically from its parent's, and a node's sample is a deterministic function of its key and its two bracketing values. Querying time $t$ means bisecting towards $t$ and stopping when the midpoint is within tolerance $\epsilon$. Nothing is kept but the initial seed, so <mark>memory is $O(1)$ and a query costs $O(\log(1/\epsilon))$</mark>. With a fixed-step solver of $L$ steps the required tolerance scales as $1/L$, which is the $\log L$ in the complexity table.

### 3.5 Latent SDEs and the path-space KL

Maximising likelihood directly over SDE parameters overfits and drives the diffusion to zero, so the authors train by variational inference. A prior and an approximate posterior share a starting point and a diffusion but differ in drift:

$$
d\tilde Z_t = h_\theta(\tilde Z_t,t)\,dt + \sigma(\tilde Z_t,t)\,dW_t, \qquad dZ_t = h_\phi(Z_t,t)\,dt + \sigma(Z_t,t)\,dW_t. \tag{7}
$$

The shared $\sigma$ is not cosmetic. If $u$ solves $\sigma(z,t)u(z,t) = h_\phi(z,t)-h_\theta(z,t)$ and satisfies Novikov's condition, Girsanov's theorem gives a measure $Q$, with density $M_T$, under which the posterior process has the prior's law. Path measures with different diffusion coefficients are mutually singular and their KL is infinite, so the shared $\sigma$ is what keeps the bound finite. Taking logs, applying Jensen, and using that $\int_0^\cdot u^\top dW$ is a zero-mean martingale gives

$$
\log p(x_{t_1},\dots,x_{t_N}\mid\theta) \;\ge\; \mathbb{E}_{Z}\!\left[\sum_{i=1}^N \log p(x_{t_i}\mid z_{t_i}) - \int_0^T \tfrac{1}{2}\,|u(z_t, t)|^2\,dt\right], \tag{8}
$$

with the expectation over posterior paths. The first term is reconstruction, the second the path-space KL, and the drift mismatch enters only through $u$. Computationally the KL is one extra scalar state with drift $\tfrac12|u|^2$ and zero diffusion; Appendix 9.7 shows its own adjoint is constant, so neither that state nor its adjoint is simulated backwards, and the augmented system still commutes. The whole objective costs one forward and one backward solve, and an encoder emitting the posterior's initial distribution and a context vector completes a VAE whose latent variable is an entire path.

### 3.6 Intuition: the 1-D linear case

Take $dZ_t = aZ_t\,dt + bZ_t\circ dW_t$ with $Z_0=z_0$. Because Stratonovich calculus obeys the ordinary chain rule, the solution is exactly $Z_t = z_0\exp(at + bW_t)$, with no $-b^2/2$ correction, so $\partial\mathcal{L}/\partial z_0 = \mathcal{L}'(Z_T)\exp(aT+bW_T)$.

Now run (4). Here $\nabla b = a$ and $\nabla\sigma = b$ are constants, so the adjoint solves the same linear equation backwards and accumulates exactly $\exp(aT+bW_T)$. For the parameter $a$ it picks up $\check A^a_0 = \int_0^T \check A^z_s Z_s\,ds$ with $\check A^z_s = \mathcal{L}'(Z_T)Z_T/Z_s$; the $Z_s$ cancel and $\check A^a_0 = \mathcal{L}'(Z_T)\,Z_T\,T = \mathcal{L}'(Z_T)\,\partial Z_T/\partial a$. Exact — but only because the same $W_T$ appears on both passes. Feed the backward solve a fresh Brownian sample and the factor becomes $\exp(aT+bW'_T)$: plausible-looking and completely wrong. That is the whole justification for the Brownian tree. Under Itô the same equation solves to $z_0\exp((a-b^2/2)t+bW_t)$, and reversing with negated coefficients misses the drift by $b^2$ — Figure 2 of the paper in one line.

### 3.7 Algorithm

```text
# forward
seed  ~ PRNG                                  # the only noise state kept
W     = VirtualBrownianTree(seed, [0,T], eps)
z_T, kl = sdeint([z_0, 0], drift_aug, diff_aug, W, 0 -> T)
loss  = -sum_i log p(x_i | z_{t_i}) + kl

# backward: one solve, state = [z, a_z, a_theta]
a_z     = dloss/dz_T
a_theta = 0
def drift_aug([z,a_z,a_th], t):        # note the reversed time argument
    v = b(z,-t); return [-v, vjp(a_z,v,z), vjp(a_z,v,theta)]
def diff_aug([z,a_z,a_th], t):
    s = sigma(z,-t); return [-s, vjp(a_z,s,z), vjp(a_z,s,theta)]
def noise(t):  return replicate(-W(-t), 3)   # SAME path, reversed
[z_0', a_z, a_theta] = sdeint([z_T,a_z,0], drift_aug, diff_aug, noise, -T -> 0)
return a_z, a_theta                           # dL/dz_0, dL/dtheta
```

Sampling from a trained latent SDE needs no backward pass: draw $z_0$ from the learned prior over initial states, draw a fresh Brownian path, solve the prior SDE forward, decode.

## 4 Implementation notes

The `torchsde` release implements several solvers in PyTorch with adaptive stepping driven by a PI controller and wraps the adjoint in a `torch.autograd.Function` subclass, so any SDE written as an `nn.Module` gets gradients. Appendix 9.13 prints the core of that class; the detail easiest to miss is that the backward solve runs *segment by segment* between observation times, with each $\partial\mathcal{L}/\partial z_{t_i}$ added to the adjoint at its boundary rather than all at once.

| Setting | Geometric BM | Stochastic Lorenz | Motion capture |
|---|---|---|---|
| Ground truth | $\mu=1$, $\sigma=0.5$, $x_0=0.1+\varepsilon$, $\varepsilon\sim\mathcal N(0,0.03^2)$ | $\sigma=10$, $\rho=28$, $\beta=8/3$, noise $0.15$ per dim | 23 walking sequences, subject 35, 50-D |
| Data | 1024 series, $\Delta t = 0.02$ on $[0,1]$ | 1024 series, $\Delta t = 0.025$ on $[0,1]$ | 16 train / 3 val / 4 test |
| Observation noise | $\mathcal N(0, 0.01^2)$ | $\mathcal N(0, 0.01^2)$, after per-dim normalisation | Gaussian, as in ODE²VAE |
| Encoder | 1-layer GRU, 100 units, run backwards | same | MLP over the first 3 frames |
| Drifts | MLP, 1 hidden layer, 100 units | same | posterior drift also takes a 3-D context |
| Diffusion | MLP, 100 units, sigmoid output | 4 scalar nets, one per latent dim | several scalar nets (diagonal noise) |
| Latent dim | 4 | 4 | 6 |
| Solver step | fixed $0.01$ | fixed | fixed, $1/5$ of the smallest observation gap |
| Optimiser | Adam, lr $0.01$, decay $0.999$ per iteration | same | same |
| KL schedule | linear anneal over first 50 iterations | same | anneal over first 200 iterations, optional |
| KL weight | not stated | not stated | tuned over $\{1, 0.1, 0.01, 0.001\}$ |
| Training length | not stated | not stated | at most 400 iterations |

Two numbers are worth keeping. The mocap latent SDE has **11,605** parameters against ODE²VAE's **12,157** and the authors' latent ODE baseline's **10,573**, so the comparison is roughly parameter-matched. And every activation is softplus, with a sigmoid on the diffusion output to keep it positive — the diffusion being the component most likely to blow up or collapse. Left unspecified: training length on the synthetic datasets, batch size everywhere, wall-clock cost of any experiment, and the tree tolerance $\epsilon$ (which never entered the reported runs at all).

## 5 Experiments

### 5.1 Gradient accuracy

Three test SDEs with closed-form solutions, each duplicated to 10 dimensions with per-dimension parameters drawn from a Gaussian and squashed through a sigmoid. Example 1 is geometric Brownian motion; example 2 is $dX_t = -p^2\sin(X_t)\cos^3(X_t)\,dt + p\cos^2(X_t)\,dW_t$, whose Itô solution is $\arctan(pW_t+\tan X_0)$ (a drift of $-\tfrac{p^2}{2}\sin X_t\cos^3X_t$ would not give this solution); example 3 is a time-inhomogeneous linear SDE. The Milstein adjoint gradient is compared to the analytic one.

Error falls with step size and with the adaptive solver's absolute tolerance on all three problems; function-evaluation counts run far above the ODE case; and on **two of the three** problems the fixed-step Milstein adjoint is more time-efficient at equal error than backpropagating through Milstein or Euler ([Fig. 5 in the paper](https://arxiv.org/pdf/2001.01328#page=8)). Euler backprop is cheaper per step but hits a worse error floor; Milstein backprop must differentiate the scheme's own higher-order terms.

### 5.2 Synthetic and motion capture

On 1-D geometric Brownian motion and the 3-D stochastic Lorenz attractor the posterior reconstructs the data and the learned prior is not degenerate: for Lorenz the prior gives bimodal samples in both latent and data space ([Fig. 6 in the paper](https://arxiv.org/pdf/2001.01328#page=9)), which a latent ODE with a unimodal Gaussian initial posterior cannot do without a multi-modal decoder. On geometric Brownian motion the authors flag a failure of their own: most of the prior's uncertainty ends up in the initial latent state rather than the diffusion.

Motion capture follows the ODE²VAE protocol — encode the first three frames, predict the remaining 297, report test MSE over 50 samples with a 95% $t$-interval.

| Method | Test MSE (297 future frames) |
|---|---|
| DTSBN-S † | 34.86 ± 0.02 |
| npODE † | 22.96 |
| NeuralODE † | 22.49 ± 0.88 |
| ODE²VAE † | 10.06 ± 1.4 |
| ODE²VAE-KL † | 8.09 ± 1.95 |
| Latent ODE | 5.98 ± 0.28 |
| **Latent SDE (this work)** | **4.03 ± 0.20** |

† taken by the authors from the ODE²VAE paper; the last two rows are their own runs.

### 5.3 Claim by claim

- *"The gradient is the composition $F\circ G$."* Theorem 3.2, proved, and the honest formulation: the adjoint is not an adapted backward SDE. Solid.
- *"The discretised algorithm converges."* Theorem 3.3, conditional on an assumption verified only for Euler–Maruyama and only for $d=1$. Figure 5(a) supports convergence empirically on one test problem; nothing supports a rate.
- *"$O(1)$ memory."* Argued, not demonstrated. <mark>The reported experiments used a Brownian implementation that stores all intermediate queries</mark>; the tree helped only for larger GPU batches. The measured claim is weaker: memory reduced "by 1/2-1/3" against backprop through the solver.
- *"High-order solvers are usable."* Supported by Appendix 9.5 and the Milstein timing plots, but only under diagonal noise.
- *"Competitive on real data."* One table, one dataset, 297-frame horizon, roughly parameter-matched, and the margin over the latent ODE (4.03 vs 5.98) exceeds either interval. Attributing that margin to the path-space KL rests on one qualitative remark — removing the KL penalty improved training error and worsened validation error — with no ablation numbers.

## 6 Limitations

**Stated by the authors.**

- Convergence rates for numerical gradients under general schemes are unknown, and whether the results survive under weak rather than strong error is open.
- Condition (ii) is verified only for Euler–Maruyama; extending it is called technically non-trivial and deferred.
- Gradient variance is not addressed; control variates and antithetic paths are named as future work.
- The bottleneck is the sequential solve, and NFE counts run far above the ODE case.
- On geometric Brownian motion the learned prior puts most of its uncertainty in the initial state, which the authors call an interpretability problem.

**My reading.**

- The $O(1)$-memory headline and the experiments are disjoint. Nothing measures the tree's overhead against stored noise, so the trade-off between $\log L$ queries and $O(L)$ storage is unquantified.
- The high-order path needs diagonal noise, which excludes most multivariate financial models: correlated Brownian drivers are the norm. Reaching diagonal noise through a state-independent mixing matrix is a real restriction on the model class.
- The tree tolerance $\epsilon$ interacts with solver adaptivity in a way nobody analyses — the noise the backward solver sees is itself an approximation, and that error is absent from the convergence theorem.
- The evaluation is one small real dataset with a three-frame encoder, inheriting five table rows from another paper, so protocol differences are assumed away rather than checked.
- The paper's printed solution to test Example 1 interchanges $\alpha$ and $\beta$ relative to its own SDE: for $dX_t=\alpha X_t\,dt+\beta X_t\,dW_t$ the Itô solution is $X_0\exp((\alpha-\beta^2/2)t+\beta W_t)$, not the printed $X_0\exp((\beta-\alpha^2/2)t+\alpha W_t)$. A transcription slip, but worth knowing if you reimplement the benchmark.

## 7 Extensions

**What was built on this.** The direct successor is [SDE-GAN](/blog/neural-sde-gan/), which reuses `torchsde` but abandons the adjoint: gradient-penalty training needs a double backward, and Kidger et al. report that a double *continuous* adjoint is too inaccurate at moderate step sizes to train with, so they differentiate the solver's internals instead — the sharpest practical boundary anyone has drawn around this method. [Neural JSDE](/blog/neural-jump-sde/) extends latent continuous-time dynamics to jumps; [Sig-SDE](/blog/sig-sdes/) and [Neural SDEs for pricing and hedging](/blog/neural-sde-pricing-hedging/) pursue calibration with signatures and payoff matching. The reverse-time SDE of [Score-SDE](/blog/score-sde/) is a different object — a reversal in law, not a pathwise inverse flow — and reading the two together sharpens exactly the contrast this paper's Figure 2 draws. Later, Kidger, Foster, Li and Lyons (NeurIPS 2021) introduced the Brownian Interval and the algebraically reversible Heun solver, refining the noise and memory story further.

**Open problems.** Rates for the composed estimator; condition (ii) beyond Euler–Maruyama and $d=1$; variance reduction for adjoint gradients; high-order schemes without diagonal noise; whether a rough-path formulation, which the authors raise themselves, removes the adaptedness problem rather than merely avoiding it.

**Research directions.** *These are ideas, not results — none has been run.*

1. **Latent SDE with a stationary prior for intraday returns.** *Hypothesis:* constraining the prior drift to a mean-reverting form with a learned diffusion gives better multi-step return distributions than a free prior, because the KL term then regularises towards a process with the right long-run behaviour. *Data:* minute-bar returns for a liquid equity index, split by year. *Baseline:* the same latent SDE with a free MLP prior drift, and a latent ODE. *Metric:* CRPS at 5/30/120 minutes, plus the decay of autocorrelation in absolute returns. *Likely failure mode:* what happened on geometric Brownian motion — the model dumps its uncertainty into the initial latent state and learns a near-deterministic drift, so the diffusion never has to be right.
2. **Adjoint variance as a diagnostic.** *Hypothesis:* the variance of $\check A_{0,T}$ across Brownian samples signals where a fitted SDE is misspecified, concentrating where drift and diffusion disagree about the local scale. *Data:* the paper's three closed-form test SDEs, where the exact gradient is known, then a fitted stochastic volatility model. *Baseline:* analytic gradient variance where available. *Metric:* correlation between adjoint variance and pointwise misspecification error. *Likely failure mode:* the variance is dominated by the terminal $\nabla\mathcal{L}$ and carries no local information.
3. **Diagonal-noise reparameterisation for correlated drivers.** *Hypothesis:* a learned state-independent mixing matrix on top of a diagonal-noise latent SDE recovers correlated observed dynamics while preserving commutativity, so Milstein stays available. *Data:* a multi-asset basket with strong empirical correlation. *Baseline:* the same model with full matrix diffusion and Euler–Maruyama. *Metric:* gradient error against a very fine reference solve at matched wall-clock. *Likely failure mode:* the mixing matrix absorbs the drift structure too, and the model degenerates to a factor model with noise bolted on.

## 8 Takeaways

- Write the SDE in Stratonovich form and the neural ODE adjoint carries over almost unchanged: negate the coefficients, propagate vector–Jacobian products, reuse the noise. The sign symmetry of (2) is the whole trick.
- The backward pass must see the *same* Brownian sample, not the same distribution. The virtual Brownian tree regenerates it from one seed at $O(1)$ memory and $O(\log(1/\epsilon))$ per query — and the paper's own experiments did not need it.
- Sharing the diffusion between prior and posterior is what makes the path-space KL finite, via Girsanov; the resulting term is one extra scalar state with zero diffusion and it behaves as a regulariser, not merely as a bound.
- The theory is honest about its gaps: the adjoint is a composition of Itô maps rather than an adapted backward SDE, and the discretisation theorem rests on a uniformity condition proved only for Euler–Maruyama in one dimension.
- For financial time series this is the plumbing under any learned continuous-time model, and the prior SDE is the natural place to put domain structure. The costs are a sequential solve, noisy gradients, and a diagonal-noise assumption if you want strong order 1.0. The paper names derivative pricing as a target but runs no financial experiment.

## References

1. Li, X., Wong, T.-K. L., Chen, R. T. Q., Duvenaud, D. *Scalable Gradients for Stochastic Differential Equations.* AISTATS 2020. arXiv:2001.01328.
2. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. *Neural Ordinary Differential Equations.* NeurIPS 2018.
3. Kunita, H. *Stochastic Flows and Stochastic Differential Equations.* Cambridge University Press, 1990.
4. Giles, M., Glasserman, P. *Smoking Adjoints: Fast Monte Carlo Greeks.* Risk, 2006.
5. Yıldız, Ç., Heinonen, M., Lähdesmäki, H. *ODE²VAE: Deep Generative Second Order ODEs with Bayesian Neural Networks.* NeurIPS 2019.
6. Kidger, P., Foster, J., Li, X., Oberhauser, H., Lyons, T. *Neural SDEs as Infinite-Dimensional GANs.* ICML 2021. arXiv:2102.03657.
7. Kidger, P., Foster, J., Li, X., Lyons, T. *Efficient and Accurate Gradients for Neural SDEs.* NeurIPS 2021.
