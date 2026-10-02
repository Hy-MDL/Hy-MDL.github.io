---
title: "DDIM: Denoising Diffusion Implicit Models"
paper:
  title: "Denoising Diffusion Implicit Models"
  authors: "Jiaming Song et al."
  venue: "ICLR 2021"
  arxiv: "2010.02502"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "score-to-flow"
order: 4
tags: [diffusion, sampling, ddim, non-markovian, probability-flow-ode, deterministic-sampling, generative-models]
date: 2020-10-01
status: draft
summary: "The DDPM loss constrains only the noisy marginals, so a whole family of non-Markovian forward processes shares it; the deterministic member of that family samples a pretrained DDPM in 20–100 steps and turns x_T into a usable latent code."
---

## Abstract

DDPMs need on the order of a thousand sequential network calls per sample. This paper shows that the cost is a property of the *sampler*, not of the trained network. The DDPM objective only ever evaluates the network on samples drawn from the marginals $q(x_t \mid x_0)$; it never touches the joint $q(x_{1:T}\mid x_0)$. Any inference process with the same marginals — Markovian or not — is therefore trained by the same loss. The authors construct such a family, indexed by a per-step noise level $\sigma_t$, whose reverse updates interpolate between DDPM's stochastic step and a fully deterministic map from noise to data (the DDIM). Because the construction also works on a subsequence of timesteps, a model trained with $T=1000$ can be sampled in tens of steps with no retraining, and the deterministic limit turns out to be an Euler discretisation of a probability-flow ODE.

**Keywords:** diffusion models, fast sampling, non-Markovian inference, implicit generative models, neural ODE, latent interpolation

## 1 Introduction

Iterative denoising models — [DDPM](/blog/ddpm/) and [NCSN](/blog/ncsn/) — reached GAN-level image quality without adversarial training, and paid for it entirely at inference time. The generative chain is built to approximate the reverse of a long forward diffusion, so each of its steps has to be simulated in sequence. The paper quantifies the gap on a single Nvidia 2080 Ti: about 20 hours to draw 50k images of size $32\times32$ from a DDPM, under a minute for a GAN, and close to 1000 hours for 50k images at $256\times256$.

Why not just take fewer DDPM steps? Because the Gaussian form of each reverse conditional is only accurate when the step is small — that is precisely the argument the DDPM paper uses to justify $T=1000$. Coarsening the same chain attacks the assumption the chain rests on. The move here is different: rather than approximate the reverse of the *same* forward process more crudely, replace the forward process with one whose reverse is short, while leaving the training objective bit-for-bit unchanged so that existing checkpoints stay valid. <mark>The contribution is a change of inference model with zero change to training.</mark>

## 2 Background

A notation warning, because it causes real confusion when reading DDIM next to DDPM: this paper writes $\alpha_t$ for what DDPM calls $\bar\alpha_t$, the cumulative signal level. DDPM's per-step quantities are recovered as $\alpha_t^{\mathrm{DDPM}}=\alpha_t/\alpha_{t-1}$ and $\beta_t=1-\alpha_t/\alpha_{t-1}$ (Appendix C.2); the justification is that the schedule becomes one sequence $\alpha_{1:T}$ instead of a derived hierarchy. Everything below uses the paper's convention, with $\alpha_0:=1$. The forward marginal is

$$
x_t = \sqrt{\alpha_t}\,x_0 + \sqrt{1-\alpha_t}\,\epsilon,
\qquad \epsilon\sim\mathcal N(0,I),
\tag{1}
$$

so $x_t$ is a fixed linear blend of a data point and one Gaussian draw. The training loss, for a vector of positive weights $\gamma$, is

$$
L_\gamma(\epsilon_\theta)=\sum_{t=1}^{T}\gamma_t\,
\mathbb E_{x_0,\epsilon_t}\Big[\big\|\epsilon_\theta^{(t)}\big(\sqrt{\alpha_t}x_0+\sqrt{1-\alpha_t}\,\epsilon_t\big)-\epsilon_t\big\|_2^2\Big].
\tag{2}
$$

Each term asks the network to recover the noise mixed into one clean sample; $\gamma_t$ says how much timestep $t$ counts, and DDPM trains with $\gamma=\mathbf 1$. The fact that drives the whole paper is visible by inspection: Eq. (2) draws $x_t$ straight from $x_0$ through Eq. (1). It never samples a trajectory.

## 3 Method

> **Key idea.** The DDPM objective pins down the marginals $q(x_t\mid x_0)$ and nothing else. Many joints share those marginals, including non-Markovian ones whose reverse process is deterministic and whose chain can skip timesteps. One trained network therefore serves an entire family of samplers, selected at test time.

### 3.1 A family of inference processes

The authors define the inference process *backwards*, conditioned on $x_0$, which is the only way to control the marginals directly. Fix $q_\sigma(x_T\mid x_0)=\mathcal N(\sqrt{\alpha_T}x_0,(1-\alpha_T)I)$ and, for $t>1$, set

$$
q_\sigma(x_{t-1}\mid x_t,x_0)=\mathcal N\!\Big(
\underbrace{\sqrt{\alpha_{t-1}}\,x_0}_{\text{signal at } t-1}
+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\cdot\frac{x_t-\sqrt{\alpha_t}\,x_0}{\sqrt{1-\alpha_t}}}_{\text{reuse of the noise already in } x_t},
\;\sigma_t^2 I\Big).
\tag{3}
$$

The fraction is exactly the unit-variance noise vector that produced $x_t$ from $x_0$ under Eq. (1); the coefficient in front of it is whatever is left of the $t-1$ noise budget after reserving $\sigma_t^2$ for fresh randomness. Lemma 1 verifies by induction from $t=T$ downwards (one Gaussian marginalisation, Bishop eq. 2.115) that $q_\sigma(x_{t-1}\mid x_0)=\mathcal N(\sqrt{\alpha_{t-1}}x_0,(1-\alpha_{t-1})I)$ — the mean contributions cancel and the two variance terms sum to $1-\alpha_{t-1}$. This is exact, not an approximation, and holds for every $\sigma\in\mathbb R^T_{\ge0}$ with $\sigma_t^2\le 1-\alpha_{t-1}$.

Bayes' rule then gives the implied forward transition $q_\sigma(x_t\mid x_{t-1},x_0)$, which conditions on $x_0$ as well as $x_{t-1}$: the process is no longer a Markov chain, and no longer a diffusion. The paper never uses the closed form of that transition — it is produced only to show the construction is a coherent inference model.

### 3.2 Why the DDPM loss still trains it

The generative step mirrors Eq. (3) with the unknown $x_0$ replaced by the network's estimate of it,

$$
f_\theta^{(t)}(x_t)=\frac{x_t-\sqrt{1-\alpha_t}\,\epsilon_\theta^{(t)}(x_t)}{\sqrt{\alpha_t}},
\tag{4}
$$

which is Eq. (1) solved for $x_0$. Define $p_\theta^{(t)}(x_{t-1}\mid x_t)=q_\sigma(x_{t-1}\mid x_t,f_\theta^{(t)}(x_t))$, with a Gaussian of variance $\sigma_1^2 I$ at $t=1$ so the model has support everywhere.

Theorem 1 says: for all $\sigma>0$ there are weights $\gamma$ and a constant $C$ with $J_\sigma=L_\gamma+C$, where $J_\sigma$ is this generative process's variational objective. Each KL term in the ELBO is between two Gaussians of the *same* variance $\sigma_t^2 I$, so it collapses to a squared distance between means, which by Eq. (4) reduces to $\|\epsilon-\epsilon^{(t)}_\theta(x_t)\|^2$ scaled by $1/(2d\sigma_t^2\alpha_t)$ with $d$ the data dimension. Hence <mark>$\gamma_t = 1/(2d\sigma_t^2\alpha_t)$: changing $\sigma$ only re-weights the per-timestep terms of the same loss.</mark>

The final step is the one to watch. If the networks $\epsilon_\theta^{(t)}$ do not share parameters across $t$, each term is minimised independently and the minimiser does not depend on $\gamma$ at all — so $L_{\mathbf 1}$ is a valid surrogate for every $J_\sigma$. <mark>Real models do share one network across all $t$, so the equivalence is exact only in an idealisation the experiments do not satisfy</mark>; in practice the authors simply load pretrained DDPM checkpoints and report that it works.

### 3.3 The sampling rule, $\eta$, and the DDIM limit

Substituting Eq. (4) into Eq. (3) gives the update

$$
x_{t-1}=\underbrace{\sqrt{\alpha_{t-1}}\,f_\theta^{(t)}(x_t)}_{\text{predicted } x_0,\ \text{rescaled}}
+\underbrace{\sqrt{1-\alpha_{t-1}-\sigma_t^2}\;\epsilon_\theta^{(t)}(x_t)}_{\text{direction back toward } x_t}
+\underbrace{\sigma_t\,\epsilon_t}_{\text{fresh noise}} .
\tag{5}
$$

Read it as: jump all the way to the currently predicted clean image, then walk back out along the predicted noise direction to the noise level appropriate for $t-1$, then optionally add new randomness. Experiments parameterise the noise level by one scalar $\eta$ over the sampling subsequence $\tau$:

$$
\sigma_{\tau_i}(\eta)=\eta\,\sqrt{\frac{1-\alpha_{\tau_{i-1}}}{1-\alpha_{\tau_i}}}\,\sqrt{1-\frac{\alpha_{\tau_i}}{\alpha_{\tau_{i-1}}}}.
\tag{6}
$$

At $\eta=1$ the implied forward process becomes Markovian again and Eq. (5) is exactly DDPM ancestral sampling with the smaller of DDPM's two variance choices. <mark>At $\eta=0$ the noise term vanishes and $x_0$ becomes a deterministic function of $x_T$ — an implicit probabilistic model trained with a denoising loss</mark>, which is what the name records. Note that $\eta=0$ falls outside Theorem 1 ($\sigma>0$); the paper justifies it in a footnote as a limit that can be approached by making $\sigma_t$ small.

A fifth setting appears in the tables, written $\hat\sigma_{\tau_i}=\sqrt{1-\alpha_{\tau_i}/\alpha_{\tau_{i-1}}}$: DDPM's *larger* variance, used by Ho et al. only for their CIFAR-10 samples. Appendix D.3 makes explicit what it does — it keeps the deterministic coefficients of the $\eta=1$ update but substitutes a bigger noise scale, so <mark>$\hat\sigma$ is not a member of the $\sigma$-family at all and does not preserve the marginals when steps are skipped.</mark> That is a stronger statement than "it is noisier", and it predicts the collapse seen in the table.

### 3.4 Skipping timesteps

Because the loss depends only on marginals, the same argument runs on an increasing subsequence $\tau=(\tau_1,\dots,\tau_S)$ of $\{1,\dots,T\}$ with $\tau_S=T$. Appendix C.1 writes out the inference factorisation: the visited variables $\{x_{\tau_i}\}$ and $x_0$ form a chain, while the skipped ones hang off $x_0$ in a star and contribute KL terms to the objective without ever being sampled at test time. The generative process then runs Eq. (5) over reversed $\tau$, replacing $\alpha_{t-1}$ with $\alpha_{\tau_{i-1}}$. $S$ becomes a pure test-time knob. [Fig. 2 in the paper](https://arxiv.org/pdf/2010.02502#page=5) draws the case $\tau=[1,3]$.

### 3.5 The ODE limit

Rewriting Eq. (5) at $\eta=0$ in terms of $x_t/\sqrt{\alpha_t}$ gives

$$
\frac{x_{t-\Delta t}}{\sqrt{\alpha_{t-\Delta t}}}=\frac{x_t}{\sqrt{\alpha_t}}
+\Big(\sqrt{\tfrac{1-\alpha_{t-\Delta t}}{\alpha_{t-\Delta t}}}-\sqrt{\tfrac{1-\alpha_t}{\alpha_t}}\Big)\,\epsilon_\theta^{(t)}(x_t),
\tag{7}
$$

which is visibly an Euler step. Setting $\bar x=x/\sqrt\alpha$ and $\sigma=\sqrt{1-\alpha}/\sqrt\alpha$ turns it into

$$
d\bar x(t)=\epsilon_\theta^{(t)}\!\Big(\frac{\bar x(t)}{\sqrt{\sigma^2+1}}\Big)\,d\sigma(t).
\tag{8}
$$

Proposition 1 shows that, at the optimum, this ODE is the probability-flow ODE of the variance-exploding SDE of the concurrent [Score-SDE](/blog/score-sde/) paper; the proof is the change of variables above plus the identity $\nabla_{\bar x}\log p_t=-\epsilon_\theta/\sigma$ relating denoiser to score. Worth flagging: the body text on p. 6 calls the same object "a continuous-time analog of DDPM" while Proposition 1 says variance-exploding — the labels are reconciled only by the reparameterisation, and the sentence reads as an inconsistency.

The ODEs coincide; the *discretisations* do not. <mark>DDIM takes Euler steps with respect to $d\sigma$, whereas the probability-flow sampler takes them with respect to $dt$</mark>, and the two agree only when consecutive $\alpha$ are close — which the paper argues is exactly what fails when $S$ is small. Running Eq. (8) forwards also encodes $x_0$ into $x_T$, which is where the reconstruction experiment comes from.

### 3.6 Intuition: the exactly solvable case

Here is a small calculation of my own that explains the ordering of the table before any image is generated. Take 1-D data $x_0\sim\mathcal N(0,1)$ and write $\alpha_t=\cos^2\theta_t$, so $\theta$ increases with $t$ from $0$. Then every marginal is $\mathcal N(0,1)$ and the optimal denoiser is linear: $\epsilon_\theta(x_t)=\mathbb E[\epsilon\mid x_t]=\sin\theta_t\,x_t$, so $f_\theta(x_t)=\cos\theta_t\,x_t$.

Feed that into Eq. (5). At $\eta=0$,

$$
x_{t-1}=\big(\cos\theta_{t-1}\cos\theta_t+\sin\theta_{t-1}\sin\theta_t\big)x_t=\cos(\theta_t-\theta_{t-1})\,x_t ,
$$

a pure contraction. Over $S$ uniform steps spanning a total angle $\Theta$, the output variance is $\prod\cos^2\Delta\theta_i\approx 1-\Theta^2/S$: the deterministic sampler is *under*-dispersed, with a variance deficit that decays like $1/S$. Repeating the algebra at $\eta=1$ gives a per-step variance of $1-(\sin^2\theta_t-\sin^2\theta_{t-1})^2/(\cos^2\theta_{t-1}\sin^2\theta_t)\approx 1-4\Delta\theta^2$ — the same $1/S$ law with four times the constant. And $\hat\sigma$ is exactly right here: its mean coefficient is $\cos\theta_t/\cos\theta_{t-1}$, which is the true correlation between $x_{t-1}$ and $x_t$, and its variance $1-\alpha_t/\alpha_{t-1}$ is exactly the residual, so it preserves $\mathcal N(0,1)$ at any step size.

So the toy model reproduces two of the paper's three orderings — less noise means less accumulated discretisation error at small $S$ — and gets the third backwards. That mismatch is informative: $\hat\sigma$ is the *exact* posterior variance for Gaussian data and an over-estimate for image data, where the leftover noise it injects has to be denoised away by steps that no longer exist. The paper reaches the same explanation empirically, via FID's sensitivity to residual noise.

### 3.7 Algorithm

Training is unchanged from DDPM and is repeated only to make the point that nothing in it depends on $\tau$ or $\eta$:

```
# Training (identical to DDPM, gamma = 1)
repeat:
    x0 ~ data
    t  ~ Uniform{1..T}
    eps ~ N(0, I)
    xt  = sqrt(alpha[t]) * x0 + sqrt(1 - alpha[t]) * eps
    take a gradient step on || eps_theta(xt, t) - eps ||^2

# Sampling (choose tau and eta at test time; alpha[0] := 1)
tau = increasing subsequence of 1..T, length S, tau[S] = T
x   = sample from N(0, I)                      # this is x_{tau_S}
for i = S down to 1:
    a_cur, a_prev = alpha[tau[i]], alpha[tau[i-1]]
    e     = eps_theta(x, tau[i])
    x0hat = (x - sqrt(1 - a_cur) * e) / sqrt(a_cur)
    sig   = eta * sqrt((1 - a_prev) / (1 - a_cur)) * sqrt(1 - a_cur / a_prev)
    z     = sample from N(0, I) if i > 1 else 0
    x     = sqrt(a_prev) * x0hat + sqrt(1 - a_prev - sig^2) * e + sig * z
return x

# Encoding (eta = 0 only): run the same loop with i ascending,
# swapping the roles of a_cur and a_prev.
```

## 4 Implementation notes

| Item | As reported |
|---|---|
| Datasets | CIFAR-10 $32\times32$ (unconditional), CelebA $64\times64$, LSUN Bedroom and Church $256\times256$ |
| Training objective | $L_{\mathbf 1}$ (Eq. 2 with $\gamma=\mathbf 1$), $T=1000$ |
| $\alpha$ schedule | the heuristic of Ho et al.; the specific $\beta$ range is not restated here |
| Network | U-Net over a Wide ResNet backbone, as in DDPM |
| CelebA model | five feature-map resolutions, $64\times64$ down to $4\times4$; StyleGAN-repo preprocessing of the original (non-HQ) CelebA |
| Checkpoints | pretrained DDPM checkpoints for CIFAR-10, Bedroom, Church; CelebA trained by the authors |
| Subsequence $\tau$ | linear $\tau_i=\lfloor ci\rfloor$ or quadratic $\tau_i=\lfloor ci^2\rfloor$, $c$ chosen so the last element is near $T$; quadratic for CIFAR-10, linear for the rest |
| Noise setting | $\eta\in\{0,0.2,0.5,1.0\}$ via Eq. (6), plus the non-family $\hat\sigma$ |
| Hardware | one Nvidia 2080 Ti for the timing figure |
| Optimiser, LR, batch size, iterations | **not stated** for the CelebA model |
| Number of samples for FID | **not stated**; the timing figure uses 50k |

Things that are easy to get wrong when reproducing. (i) The $\alpha$ convention — feeding a DDPM codebase's per-step $\alpha_t$ into Eq. (5) silently produces nonsense. (ii) $\alpha_0:=1$, so the final step has $\sqrt{1-\alpha_{\tau_0}-\sigma^2}=0$ and lands directly on the predicted $x_0$; an off-by-one in $\tau$ shows up as a permanently noisy output. (iii) $\tau_S=T$ — starting the chain below the top noise level mismatches the prior. (iv) Eq. (6) uses $\alpha_{\tau_{i-1}}$, not $\alpha_{\tau_i - 1}$. (v) The $\hat\sigma$ variant keeps the $\eta=1$ deterministic coefficients, so it cannot be obtained by setting $\eta>1$.

## 5 Experiments

Setup: one model per dataset, trained once with $T=1000$ and $L_{\mathbf 1}$; only $\tau$ (hence $S$) and $\eta$ vary at test time. Metric is FID throughout, lower better.

**Table 1, FID on CIFAR-10 ($32\times32$) and CelebA ($64\times64$).**

| Sampler | CIFAR $S=10$ | 20 | 50 | 100 | 1000 | CelebA $S=10$ | 20 | 50 | 100 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| **$\eta=0.0$ (DDIM)** | **13.36** | **6.84** | **4.67** | **4.16** | 4.04 | **17.33** | **13.73** | **9.17** | **6.53** | 3.51 |
| $\eta=0.2$ | 14.04 | 7.11 | 4.77 | 4.25 | 4.09 | 17.66 | 14.11 | 9.51 | 6.79 | 3.64 |
| $\eta=0.5$ | 16.66 | 8.35 | 5.25 | 4.46 | 4.29 | 19.86 | 16.06 | 11.01 | 8.09 | 4.28 |
| $\eta=1.0$ (DDPM) | 41.07 | 18.36 | 8.01 | 5.78 | 4.73 | 33.12 | 26.03 | 18.48 | 13.93 | 5.98 |
| $\hat\sigma$ (larger variance) | 367.43 | 133.37 | 32.72 | 9.99 | **3.17** | 299.71 | 183.83 | 71.71 | 45.20 | **3.26** |

**Table 3 (appendix), FID on LSUN at $256\times256$.** For reference, 1000-step DDPM scores 6.36 on Bedroom and 7.89 on Church.

| Sampler | Bedroom $S=10$ | 20 | 50 | 100 | Church $S=10$ | 20 | 50 | 100 |
|---|---|---|---|---|---|---|---|---|
| **DDIM ($\eta=0$)** | **16.95** | **8.89** | **6.75** | 6.62 | **19.45** | **12.47** | **10.84** | 10.58 |
| DDPM ($\eta=1$) | 42.78 | 22.77 | 10.81 | **6.81** | 51.56 | 23.37 | 11.16 | **8.27** |

**Table 2, reconstruction error** (per-dimension MSE on the CIFAR-10 test set, encode then decode with the same $S$): 0.014 at $S=10$, 0.0065 at 20, 0.0023 at 50, 0.0009 at 100, 0.0004 at 200, and 0.0001 at 500 and 1000.

### Claim by claim

*"DDIM samples 10× to 50× faster."* Supported on CIFAR-10 and CelebA by Table 1 plus the linear time-versus-steps plot ([Fig. 4](https://arxiv.org/pdf/2010.02502#page=8)): 100-step DDIM scores 4.16 against 4.04 at 1000 steps, and 20-step DDIM on CelebA (13.73) matches 100-step DDPM (13.93). Two caveats. The factor is stated as "10× to 50×" in the abstract and §5.1 but "10× to 100×" in the introduction; the tables support the smaller claim. And <mark>on LSUN Church it fails outright: DDIM plateaus near 10.6 while DDPM reaches 8.27 at 100 steps and 7.89 at 1000</mark> — reported in the appendix without comment.

*"Less stochasticity is better at small budgets."* Strongly supported: FID is monotone in $\eta$ at every $S$ on both datasets, and the gap widens as $S$ shrinks (13.36 vs 41.07 at $S=10$). The $\hat\sigma$ row matches §3.3's reading that it stops being marginal-preserving once steps are skipped.

*"At full budget, determinism is not free."* <mark>At $S=1000$ the deterministic sampler loses to $\hat\sigma$ on both datasets (4.04 vs 3.17; 3.51 vs 3.26)</mark> — the recommendation is budget-dependent, not universal.

*"Samples are consistent in $x_T$."* [Fig. 5](https://arxiv.org/pdf/2010.02502#page=8) shows one $x_T$ decoded at several $S$ keeping its high-level content — qualitative only, with no metric, no failure rate, and a small panel behind the claim that 20 steps already look like 1000.

*"$x_T$ behaves like a GAN latent."* Qualitatively by spherical-linear interpolation ([Fig. 6](https://arxiv.org/pdf/2010.02502#page=8), formula in Appendix D.5), quantitatively by Table 2: <mark>round-trip error falls from 0.014 to 0.0001 as $S$ grows — flow-like invertibility from a model never trained to be invertible.</mark> Nothing beyond MSE is measured: no downstream use of the code, no attribute editing, no comparison with a GAN latent space.

## 6 Limitations

**Stated by the authors.**
- Theorem 1 requires $\sigma>0$ and no parameter sharing across $t$; DDIM ($\sigma=0$) is handled in a footnote as an approachable limit.
- Training with a different (or continuous) number of forward steps than the sampler uses is proposed but left as future work.
- Better ODE integrators — the authors name multistep methods such as Adams–Bashforth — are raised as an obvious next step and not tried.
- The non-Gaussian version of the construction (a multinomial forward process, Appendix A) is derived but never evaluated.

**My reading.**
- The theory concerns minimisers of a loss; with a shared network it licenses *reusing* a checkpoint but says nothing about how sampler mismatch interacts with finite capacity.
- Evaluation is FID-only on four image datasets, and FID is a poor instrument for exactly the risk $\eta=0$ runs: removing sampling noise narrows the output distribution (§3.6 gives an explicit variance deficit), and no recall, coverage or likelihood number is reported to check it.
- The LSUN Church result shows the ranking is dataset-dependent in a way the paper never investigates; with only $\eta\in\{0,1\}$ reported there, an intermediate $\eta$ is untested.
- $\tau$ spacing is chosen per dataset by trying both options, so part of the small-$S$ gain is hyperparameter search rather than method.
- No likelihoods, despite §3.5 setting up the [Neural ODE](/blog/neural-ode/) machinery that would make them computable.

## 7 Extensions

**What was built on this.** The $\eta$ knob and the ODE reading both became standard. [Score-SDE](/blog/score-sde/) supplies the continuous-time framework Proposition 1 connects to; [Improved DDPM](/blog/improved-ddpm/) attacks the same few-step regime from the training side by learning the reverse variances; [EDM](/blog/edm/) pushes "sampling is ODE integration" to its conclusion, re-deriving schedule, preconditioning and solver together. Deterministic sampling is what makes [Latent Diffusion](/blog/latent-diffusion/) and [Classifier-Free Guidance](/blog/classifier-free-guidance/) practical at 20–50 steps, and DDIM inversion is the standard route to editing a real image with a pretrained model. [Consistency Models](/blog/consistency-models/) can be read as learning this ODE's solution map directly. Dedicated high-order solvers — DPM-Solver, PNDM — followed shortly after.

**Open problems.**
- What is the right step-size *measure*? The paper shows that stepping in $\sigma$ beats stepping in $t$ at small $S$ but offers no principle for choosing the parameterisation, and picks $\tau$ by trying two options.
- Why does the optimal $\eta$ move with the budget, and can it be set from a computable quantity (score error, curvature) rather than a grid search?
- What does the deterministic map cost in distributional terms? There is no measurement of diversity loss anywhere in the paper.
- The non-Markovian construction is fully general in $\sigma$, but only the one-parameter $\eta$ slice is ever explored; per-timestep $\sigma_t$ is a free vector with $T$ entries.

**Research directions.**

*These are ideas, not results — none has been run.*

1. **Budget-aware $\eta$ schedules.** *Hypothesis:* the best $\eta$ is not constant in $t$ — early, high-noise steps tolerate stochasticity and late steps do not — so a fitted $\eta(t)$ beats any constant $\eta$ at $S\le 20$. *Data:* CIFAR-10 as a sanity check, then daily returns of a liquid equity index. *Baseline:* the best constant $\eta$ per budget from a grid. *Metric:* FID for images; for returns, moment errors, the tail index, and the discriminative/predictive scores standard in time-series generation. *Failure mode:* the gain sits inside the grid search's noise, or the schedule rediscovers $\eta\equiv0$.

2. **Does determinism cost tail mass?** *Hypothesis:* at small $S$, $\eta=0$ yields a narrower output distribution than $\eta=1$ and the deficit concentrates in the tails — the defect §3.6 exhibits in 1-D. *Data:* a diffusion model fitted to multivariate daily returns, as in [Diffusion-TS](/blog/diffusion-ts/) or [TSDiff](/blog/tsdiff/). *Baseline:* the same checkpoint at $\eta=1$, $S=1000$. *Metric:* variance ratio against training data plus VaR/ES error at 1% and 5%, across $S\in\{10,20,50,100\}$. *Failure mode:* the model's own tail misspecification swamps the sampler's; a pilot on synthetic heavy-tailed data would catch that first.

3. **The latent as a controlled-experiment device.** *Hypothesis:* since $x_T$ is a genuine code and the map deterministic, conditional generators can be compared under a fixed latent — a paired rather than unpaired comparison, with much lower Monte-Carlo error. *Data:* a conditional path generator in the style of [CSDI](/blog/csdi/), conditioned on observed history. *Baseline:* the same comparison with independent stochastic sampling. *Metric:* variance of the estimated conditioning effect at fixed sample count. *Failure mode:* the latent is not conditioning-invariant, so the same $x_T$ means different things under different conditions and the pairing buys nothing.

## 8 Takeaways

- The DDPM loss constrains the marginals $q(x_t\mid x_0)$ and nothing else; the joint inference process is a design choice that can be made *after* training.
- One scalar $\eta$ interpolates between DDPM ($\eta=1$) and a deterministic implicit model ($\eta=0$). Less noise wins when steps are scarce, by a margin that grows as the budget shrinks — but $\hat\sigma$ still wins at $S=1000$, and on LSUN Church DDPM wins from 100 steps up.
- The deterministic sampler is an Euler discretisation, in $\sigma$ rather than $t$, of an ODE equal to the VE probability-flow ODE — the hinge between the diffusion and flow halves of this series.
- A fixed $x_T$ is a usable latent code: consistent across step counts, interpolable, invertible to a per-dimension MSE of $10^{-4}$ — none of which the objective asked for.
- The theory is exact only without parameter sharing across timesteps, and the paper's own $\hat\sigma$ variant is not a member of the family it derives. Both gaps are bridged empirically, and both are worth remembering when transplanting the method.
- For financial time series the immediate value is computational: scenario generation needs many paths, and a 10–50× cheaper sampler from an unchanged model is directly useful, with the deterministic noise-to-path map as a bonus for controlled comparison. Whether removing sampling noise also removes tail mass is left entirely open, and it is the question that matters most for risk work.

## References

1. Song, J., Meng, C., Ermon, S. *Denoising Diffusion Implicit Models.* ICLR 2021. arXiv:2010.02502.
2. Ho, J., Jain, A., Abbeel, P. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. arXiv:2006.11239.
3. Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., Poole, B. *Score-Based Generative Modeling through Stochastic Differential Equations.* ICLR 2021. arXiv:2011.13456.
4. Song, Y., Ermon, S. *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS 2019. arXiv:1907.05600.
5. Chen, R. T. Q., Rubanova, Y., Bettencourt, J., Duvenaud, D. *Neural Ordinary Differential Equations.* NeurIPS 2018. arXiv:1806.07366.
6. Jolicoeur-Martineau, A., Piché-Taillefer, R., Tachet des Combes, R., Mitliagkas, I. *Adversarial score matching and improved sampling for image generation.* 2020. (Cited by the paper for FID's sensitivity to residual noise.)
