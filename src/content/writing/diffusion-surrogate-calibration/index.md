---
title: "DBS: Diffusion-Based Surrogate Modeling and Multi-Fidelity Calibration"
paper:
  title: "Diffusion-Based Surrogate Modeling and Multi-Fidelity Calibration"
  authors: "Naichen Shi, Hao Yan, Shenghan Guo, Raed Al Kontar"
  venue: "arXiv 2024"
  arxiv: "2407.17720"
  license: "creativecommons.org/licenses/by-nc-nd/4.0/"
series: "surrogates-bo"
order: 6
tags: [surrogate-modeling, multi-fidelity, model-calibration, diffusion, conditional-diffusion, guidance, physics-informed, uncertainty-quantification, kennedy-ohagan]
date: 2024-07-01
status: draft
summary: "A conditional diffusion model is used as the calibration layer on top of physics simulators: a cheap simulation is fed to the denoiser as an input, and an expensive one, when it happens to exist, steers sampling through a Bayes-rule guidance term, so the costly simulator never has to be run for the whole training set."
---

## Abstract

Simulators encode physics but are biased; field data are faithful but scarce. Kennedy–O'Hagan (KOH) calibration reconciles them with a Gaussian process on the simulator–reality gap, which breaks down for $128\times128$ images. This paper swaps the GP for a conditional diffusion model and splits simulators by cost. A *cheap* simulation, available everywhere, becomes an extra input channel of the denoiser. An *expensive* one, available only sometimes, never touches training: at sampling time its hand-designed log-likelihood gradient is added to the learned score through Bayes' rule. A Wasserstein bound links sample quality to the error of the score network and of that guidance gradient. The method is tested on a 2-D buoyant-plume simulation and on infrared video of a NIST metal 3-D printing benchmark.

**Keywords:** surrogate modelling · multi-fidelity simulation · output calibration · conditional diffusion · posterior-sampling guidance · uncertainty quantification

## 1 Introduction

The target is the future state of a physical system — a buoyancy field ten time units ahead, or the next frames of a thermal video — given past observations, simulators that are fast but crude, and simulators that are better but too slow to run on everything.

**Physics-informed and physics-constrained networks** write an ODE/PDE into the loss, which presumes the equation is right. In the printing case the only available physics is an idealised heat equation for the melt pool and an optical-flow model for the spatter, both of which the authors call oversimplified, so the bias would move into the network. **GP-based calibration and multi-fidelity surrogates** — KOH and its descendants — handle bias and carry uncertainty, but cost grows quadratically to cubically in the data size and they need a structural guess about how fidelities correlate. On a 16k-dimensional field the paper's KOH baseline barely moves the simulator output.

Conditioning a diffusion model on every simulator output fails on cost: the expensive simulator would have to run on every training trajectory. <mark>The paper's central design decision is to use cheap simulations as inputs at training time and expensive simulations only as likelihoods at sampling time</mark>, so the two are decoupled, and the costly simulator can be run for just the cases that matter.

## 2 Background

**Diffusion and DDIM.** Training follows [DDPM](/blog/ddpm/): corrupt the target with Gaussian noise at level $t$ and regress the noise. Sampling follows the deterministic [DDIM](/blog/ddim/) update, whose continuous limit is the probability-flow ODE of the variance-preserving SDE in [Score-SDE](/blog/score-sde/). The only fact needed below is that an $\epsilon$-predicting network is a scaled score estimate, $\epsilon_\theta \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log p(x_t\mid\cdot)$.

**Conditioning vs guidance.** Side information $c$ either enters the network at training time (as in [CSDI](/blog/csdi/) or [TSDiff](/blog/tsdiff/)), or steers a pretrained model at sampling time through $\nabla\log p(c\mid x_t)$ — the logic of [classifier guidance](/blog/diffusion-beats-gans/) and of diffusion posterior sampling (DPS), which the paper follows. DBS uses the first route for the cheap simulator and the second for the expensive one.

**KOH calibration.** In output-correction form, reality is simulator output plus a GP-distributed discrepancy; full KOH also infers the simulator's parameters. DBS does not (Section 6).

## 3 Method

> **Key idea.** Train one conditional denoiser on cheap physics. At test time, if an expensive simulation is available, turn it into a likelihood and add its gradient to the score — no retraining, and no expensive runs over the training set.

Notation used throughout: $x$ is the target state at the prediction time (the paper's $x_{0,s^*}$), $h$ the observed history, $c_1$ the cheap simulation output, $c_2$ the expensive one, and $x_t=\sqrt{\bar\alpha_t}\,x+\sqrt{1-\bar\alpha_t}\,\epsilon$ the noised target with $\alpha_t=1-\beta_t$, $\bar\alpha_t=\prod_{\tau\le t}\alpha_\tau$.

### 3.1 Cheap physics as an input channel

The training set is augmented so that each trajectory carries its cheap-simulator output, and the noise-prediction loss simply takes it as one more input:

$$
\min_\theta\;
\mathbb E_{(x,h,c_1),\,t,\,\epsilon}
\Big\|\,\epsilon-\epsilon_\theta\big(\sqrt{\bar\alpha_t}\,x+\sqrt{1-\bar\alpha_t}\,\epsilon,\;t,\;h,\;c_1\big)\Big\|^2 .
\tag{1}
$$

The first argument is the corrupted target, $t$ the noise level, and $h,c_1$ are concatenated channel-wise. At the optimum $\epsilon_\theta$ gives the score of $p(x_t\mid h,c_1)$: ordinary conditional diffusion, novel only in what is conditioned on.

Sampling uses the paper's form of the deterministic DDIM step,

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}+\sqrt{\alpha_t-\bar\alpha_t}}\;\epsilon_\theta(x_t,t,h,c_1)\right).
\tag{2}
$$

The odd-looking coefficient is exact, not a heuristic. Substituting $\bar\alpha_{t-1}=\bar\alpha_t/\alpha_t$ into the textbook DDIM step $x_{t-1}=\sqrt{\bar\alpha_{t-1}}\hat x_0+\sqrt{1-\bar\alpha_{t-1}}\,\epsilon_\theta$ gives the coefficient $\sqrt{1-\bar\alpha_t}-\sqrt{\alpha_t-\bar\alpha_t}$. Rationalising that difference gives (2). For small $\beta_t$ it tends to $\beta_t/(2\sqrt{1-\bar\alpha_t})$, which makes (2) an Euler step of the reverse ODE.

### 3.2 Expensive physics as a likelihood

When $c_2$ exists, the target distribution becomes $p(x\mid h,c_1,c_2)$, and the reverse ODE needs the score of $p(x_t\mid h,c_1,c_2)$. Bayes' rule splits it *exactly*:

$$
\nabla_{x_t}\log p(x_t\mid h,c_1,c_2)
=\underbrace{\nabla_{x_t}\log p(x_t\mid h,c_1)}_{\text{trained denoiser}}
+\underbrace{\nabla_{x_t}\log p(c_2\mid x_t,h,c_1)}_{\text{guidance } g}.
\tag{3}
$$

The first term is what (1) learned. The second has no closed form, because $c_2$ relates to the *clean* $x$, not to $x_t$. Writing $p(c_2\mid x_t)=\int p(c_2\mid x)\,p(x\mid x_t)\,dx$, Appendix A makes two approximations.

1. **Point-mass posterior.** Following DPS, $p(x\mid x_t,c_1)$ is replaced by a delta at its Tweedie mean,
   $\hat x_0=\big(x_t-\sqrt{1-\bar\alpha_t}\,\epsilon_\theta(x_t,t,h,c_1)\big)/\sqrt{\bar\alpha_t}$.
2. **No backpropagation through the network.** The chain rule needs $\partial\hat x_0/\partial x_t$, which would mean differentiating the U-Net with respect to a $128\times128$ input. The paper keeps only the explicit term, $\partial\hat x_0/\partial x_t\approx I/\sqrt{\bar\alpha_t}$, and treats $\epsilon_\theta$ as constant in $x_t$.

Together:

$$
g(x_t)\;=\;\frac{1}{\sqrt{\bar\alpha_t}}\;\nabla_{\hat x_0}\log p\big(c_2\mid \hat x_0,c_1\big)
\;\approx\;\nabla_{x_t}\log p(c_2\mid x_t,h,c_1).
\tag{4}
$$

The gradient is taken at the denoised guess and rescaled back to the noisy coordinates. <mark>The second approximation is what separates DBS from standard DPS, which differentiates through the network (from general knowledge, unverified): it trades the guidance gradient's fidelity for memory</mark>. Section 3.4 shows what it costs.

The likelihood is left to the user. The recipe (Section III-D): assume $c_2$ independent of $c_1$ given $x$, and use an energy model,

$$
p(c_2\mid x)\;\propto\;\exp\!\big(-\gamma\,E(c_2,x)\big),
\tag{5}
$$

where $E$ is small when simulation and state agree and $\gamma$ is a confidence temperature. The normaliser vanishes under the gradient, so only $E$ needs to be differentiable. Combining (2)–(5) gives the guided step

$$
x_{t-1}=\frac{1}{\sqrt{\alpha_t}}\left(x_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}+\sqrt{\alpha_t-\bar\alpha_t}}\,\epsilon_\theta(x_t,t,h,c_1)+(1-\alpha_t)\,g(x_t)\right).
\tag{6}
$$

The first term is the variance-preserving drift, the second pulls toward what the cheap-physics-aware denoiser finds plausible, the third toward agreement with the expensive simulator. Two remarks of mine: matching (6) to the ODE with score (3) gives a guidance weight of $\beta_t/2$, not $\beta_t$ (the continuous form the paper itself analyses in Appendix E, its Eq. (31), also carries $\beta_t/2$); and the fluid experiment drops the $(1-\alpha_t)$ factor entirely for numerical stability. <mark>The guidance strength in practice is therefore a tuned constant, not the Bayes-rule weight the derivation produces</mark>, and the sampler that was run is not exactly the one Theorem 1 analyses.

### 3.3 What is guaranteed

In the continuous-time limit, under the regularity conditions of Kwon et al. plus Lipschitz-type bounds $L_\epsilon,L_g$ on the network and on $g$, Theorem 1 states

$$
W_2\big(p(x\mid h,c_1,c_2),\,q_\theta\big)=
O\!\Big(\sqrt{\mathcal L_1+\mathcal L_2}\;+\;W_2\big(p(x_T\mid h,c_1,c_2),\,\mathcal N(0,I)\big)\Big),
\tag{7}
$$

where $\mathcal L_1$ is the $\beta_t$-weighted squared error of the rescaled denoiser against the true conditional score and $\mathcal L_2$ the same for $g$; without $c_2$, $\mathcal L_2$ drops out. The proof is a contraction argument on two continuity equations, with factors like $\exp\!\big(T(1+L_\epsilon+L_g)/2\big)$ inside the $O(\cdot)$. <mark>The bound is honest but conditional: it says nothing about how large $\mathcal L_2$ is under the two approximations of (4)</mark>, which are the only genuinely new part of the method.

### 3.4 Intuition: one Gaussian dimension

Take scalar $x\sim\mathcal N(m,s^2)$ given $c_1$, and the quadratic energy $E=(c_2-x)^2$, so $c_2$ is a Gaussian observation of $x$ with variance $1/(2\gamma)$. Everything is then exact (this calculation is mine, not the paper's). The denoising posterior is Gaussian with

$$
\mathbb E[x\mid x_t]=m+\frac{\sqrt{\bar\alpha_t}\,s^2}{\bar\alpha_t s^2+1-\bar\alpha_t}\big(x_t-\sqrt{\bar\alpha_t}\,m\big),
\qquad
\operatorname{Var}[x\mid x_t]=\frac{s^2(1-\bar\alpha_t)}{\bar\alpha_t s^2+1-\bar\alpha_t},
\tag{8}
$$

and the exact guidance term is

$$
\nabla_{x_t}\log p(c_2\mid x_t)=
\frac{c_2-\hat x_0}{\tfrac{1}{2\gamma}+\operatorname{Var}[x\mid x_t]}\cdot
\frac{\sqrt{\bar\alpha_t}\,s^2}{\bar\alpha_t s^2+1-\bar\alpha_t}.
\tag{9}
$$

Equation (4) deletes $\operatorname{Var}[x\mid x_t]$ from the denominator (point mass) and replaces the true Jacobian, the second factor, by $1/\sqrt{\bar\alpha_t}$ (no backprop). The Jacobian ratio, approximate over exact, is $1+(1-\bar\alpha_t)/(\bar\alpha_t s^2)$; for unit-variance data, $1/\bar\alpha_t$ — 2 at $\bar\alpha_t=0.5$, 10 at $\bar\alpha_t=0.1$, 1 as $t\to0$. Both shortcuts push the same way: <mark>the guidance is too strong at high noise and correct only near the end of sampling</mark>. With $\gamma=0.01$ the variance deletion hardly matters ($1/(2\gamma)=50$ against a posterior variance of at most 1); the Jacobian factor does. A small $\gamma$ is, in effect, damping the over-strong early pull by hand.

### 3.5 Algorithm

```text
TRAIN  (cheap physics only)
  input: pairs (x, h, c1) for all N trajectories
  repeat for B epochs:
      draw (x, h, c1), t ~ U{1..T}, eps ~ N(0, I)
      x_t  <- sqrt(abar_t) * x + sqrt(1 - abar_t) * eps
      take a gradient step on || eps - eps_theta(x_t, t, h, c1) ||^2

SAMPLE  (one draw; repeat with fresh x_T for uncertainty)
  input: eps_theta, h, c1, optional c2, energy E, temperature gamma
  x_T ~ N(0, I)
  for t = T, ..., 1:
      e     <- eps_theta(x_t, t, h, c1)
      k_t   <- (1 - alpha_t) / ( sqrt(1 - abar_t) + sqrt(alpha_t - abar_t) )
      step  <- x_t - k_t * e                          # DDIM, Eq. (2)
      if c2 available:
          x0hat <- (x_t - sqrt(1 - abar_t) * e) / sqrt(abar_t)
          g     <- -(gamma / sqrt(abar_t)) * grad_{x0hat} E(c2, x0hat)   # no backprop through eps_theta
          step  <- step + w_t * g                     # paper: w_t = 1 - alpha_t; experiments: w_t = 1
      x_{t-1} <- step / sqrt(alpha_t)
  return x_0
```

The mean and pixel-wise spread over repeated draws give the point prediction and its uncertainty map.

## 4 Implementation notes

| Item | As reported |
|---|---|
| Denoiser | U-Net from the `denoising-diffusion-pytorch` code base; conv + self-attention blocks, $128\times128$ down to $16\times16$ with up to 1024 channels; sinusoidal time embedding injected by addition and multiplication |
| Conditioning | history and $c_1$ concatenated channel-wise after a "preprocess" block (Fig. 1); how the $32\times32$ $c_1$ is brought to $128\times128$ is **not stated** |
| Optimiser / epochs | Adam; 200 epochs (fluid), 100 epochs (printing) |
| Learning rate, batch size, $T$, $\beta$ schedule, number of sampling steps | **not stated** |
| Guidance | $\gamma=0.01$ in the fluid case; $(1-\alpha_t)$ weight on $g$ removed; $\gamma$ for the printing case **not stated** |
| Fluid data | Boussinesq plumes, FluidSim/FluidFFT on $128\times128$; random initial buoyancy and vorticity, $s=0\to10$; $N=6880$ runs, 90/10 split |
| Fluid fidelities | $c_1$: same solver on $32\times32$ (buoyancy + vorticity at $s^*$); $c_2$: $64\times64$ buoyancy |
| Printing data | NIST AM-Bench 2018 bridge part (624 layers), infrared camera at 1800 fps; first 50 layers cut into 10-frame clips; 2 context frames, last 5 predicted; 80/20 split; frame resolution **not stated** |
| Printing fidelities | $c_1$: Green's-function solution of a 2-D heat equation driven by the G-code laser path, four parameters $(\kappa_x,\kappa_y,\rho,C_n)$ fitted by least squares with Adam; $c_2$: a per-pixel spatter velocity field used through a semi-Lagrangian warp |
| Uncertainty | 40 independent samples, pixel-wise standard deviation |
| Hardware, wall-clock times of simulators and sampler | **not stated** |
| Code | `github.com/UMDataScienceLab/MGDM` (per the paper) |

Things that are easy to get wrong when reproducing:

- **The fluid likelihood compares the two fields at $32\times32$.** Eq. (16) average-pools $c_2$ by 2 and the sample by 4. The expensive simulation therefore constrains only the coarse content of the prediction, at the same resolution as $c_1$.
- **How the printing flow field $v$ is obtained is not described** — the text calls it "estimated" on test frames. If it comes from the frames being predicted, it leaks target information; this bears directly on the consistency-score result.
- **Notation is inconsistent** ($c_{s,1}$ vs $c_{1,s}$; $\alpha_t$ and $\bar\alpha_t$ collapse in places). Use (2) as derived above.
- The row "With $c_{s,2}$" in Tables I–II is DBS with *both* simulators (Table III repeats the numbers).

## 5 Experiments

**Fluid system.** Predict the $128\times128$ buoyancy at $s^*=10$ from the initial field. Baselines: KOH (GPyTorch, independent multi-output GP on the residual), KOH on a VAE latent, a physics-constrained VAE, a U-Net regressor (NN), DDIM without physics (S-DDIM), DiT-B/4 from scratch with $c_1$, and latent diffusion with $c_1$. In [Fig. 2 in the paper](https://arxiv.org/pdf/2407.17720#page=9), S-DDIM draws sharp swirls in the wrong places, $c_1$ fixes their placement, and DiT and LDM are noisy or washed out.

*Table I — fluid test set, mean (std). MSE in units of $10^{-3}$; SSIM as printed (apparently $\times100$).*

| Method | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| KOH | 1.33 (0.04) | 29.8 (0.1) | 99.986 (0.001) | 0.416 (0.003) |
| KOH + VAE | 1.58 (0.07) | 29.4 (0.2) | 99.982 (0.001) | 0.301 (0.007) |
| PCVAE | 2.66 (0.12) | 26.7 (0.2) | 99.966 (0.001) | 0.338 (0.004) |
| NN | 1.33 (0.06) | 31.2 (0.3) | 99.987 (0.001) | 0.292 (0.008) |
| S-DDIM | 3.69 (0.2) | 25.5 (0.2) | 99.955 (0.001) | 0.401 (0.002) |
| LDM | 2.42 (0.05) | 26.6 (0.1) | 99.984 (0.001) | 0.335 (0.006) |
| DiT | 1.50 (0.07) | 30.5 (0.2) | 99.990 (0.001) | 0.456 (0.005) |
| DBS + $c_1$ | 1.14 (0.08) | 33.2 (0.4) | 99.992 (0.001) | 0.287 (0.009) |
| **DBS + $c_1$ + $c_2$** | **1.00** (0.05) | **33.6** (0.3) | **99.993** (0.001) | **0.228** (0.006) |

*Table III (appendix ablation) — fluid, which simulator matters.*

| Variant | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|
| S-DDIM | 3.69 (0.2) | 25.5 (0.2) | 99.955 (0.001) | 0.401 (0.002) |
| DBS, $c_1$ only | 1.14 (0.08) | 33.2 (0.4) | 99.992 (0.001) | 0.287 (0.009) |
| DBS, $c_2$ guidance only (denoiser without $c_1$) | 1.12 (0.06) | 32.1 (0.3) | 99.992 (0.001) | **0.216** (0.006) |
| DBS, $c_1$ + $c_2$ | **1.00** (0.05) | **33.6** (0.3) | **99.993** (0.001) | 0.228 (0.006) |

**Metal printing.** Predict thermal frames; the extra metric is a consistency score (CS): the masked squared residual of warping the previous frame by the spatter flow, divided by the masked previous frame and summed over frames. Rows of samples are in [Fig. 4](https://arxiv.org/pdf/2407.17720#page=11), and the benchmark predictions are in [Fig. 5](https://arxiv.org/pdf/2407.17720#page=12): NN is over-dispersed, while KOH and KOH+VAE show no spatter.

*Table II — printing test set, mean (std).*

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | CS ↓ |
|---|---|---|---|---|
| KOH | **25.1** (0.4) | 99.976 (0.003) | 0.33 (0.02) | 0.28 (0.03) |
| KOH + VAE | 20.7 (0.6) | 99.899 (0.003) | 0.27 (0.01) | 0.28 (0.02) |
| NN | 24.0 (0.2) | 99.960 (0.002) | 0.24 (0.01) | **0.064** (0.008) |
| S-DDIM | 21.3 (0.2) | 99.924 (0.004) | 0.166 (0.002) | 1.43 (0.03) |
| DBS + $c_1$ | 24.3 (0.2) | **99.977** (0.006) | 0.140 (0.002) | 1.41 (0.02) |
| DBS + $c_1$ + $c_2$ | 24.2 (0.2) | 99.976 (0.007) | **0.136** (0.005) | **0.064** (0.005) |

### Claim by claim

*"Cheap physics as input improves the surrogate."* Strongly supported: fluid MSE 3.69 → 1.14 and PSNR 25.5 → 33.2; printing PSNR 21.3 → 24.3 and LPIPS 0.166 → 0.140. The appendix feature maps ([Fig. 10](https://arxiv.org/pdf/2407.17720#page=19)) add a qualitative story: the latent features of the $c_1$-conditioned U-Net are less noisy and concentrate where buoyancy changes sharply.

*"The expensive simulation adds further, without retraining."* Supported on fluid, weakly on printing. In Table I the clear gain is LPIPS (0.287 → 0.228); MSE moves by about two reported standard deviations (1.14 → 1.00) and PSNR by about one (33.2 → 33.6). On printing, PSNR and SSIM do not move; only CS collapses (1.41 → 0.064). <mark>That score is the same warp-and-mask residual the guidance energy minimises, so the headline printing gain is partly a metric the sampler was told to optimise.</mark> The NN baseline, whose frames the paper itself calls over-dispersed, reaches the same 0.064, so a low CS does not by itself mean realistic spatter; the CS of the ground-truth frames is not reported.

*"More physics, monotonically better."* Only partly: in the ablation, $c_2$ guidance alone has the best LPIPS of any variant (0.216). Adding $c_1$ helps pixel metrics and costs perceptual quality — unmentioned in the main text.

*"DBS beats GP-based calibration in high dimensions."* Yes on fluid, on all four metrics; on printing KOH keeps the best PSNR (25.1). The KOH baseline is a batch-independent multi-output GP on a $16{,}384$-dimensional residual — no parameter calibration, no spatial correlation — so this says more about a naive GP than about calibration.

*"Inference cost is independent of training-set size."* True by construction, but no method or simulator is timed, so the cheap/expensive split itself is never priced.

*"Built-in uncertainty quantification."* Shown only as a picture ([Fig. 6](https://arxiv.org/pdf/2407.17720#page=12)): DBS spread is high on spatter and low on the melt pool, S-DDIM is unsure where the melt pool is, KOH's variance is featureless. No coverage or CRPS is reported, and Section 3.4 suggests guided samples will be under-dispersed.

The ± values are described only as the standard deviation "on the test set"; whether that is a spread over test cases or over repeated runs is not stated, and no seeds are mentioned.

## 6 Limitations

**Stated by the authors**

- Simulator parameters are fixed or pre-fitted by least squares; folding parameter calibration into the diffusion model is future work.
- $p(c_2\mid x,c_1)$ must be designed with domain knowledge (Section III-D).
- The printing simulators are knowingly oversimplified; DiT is data-starved at about 6k samples; LDM's pre-trained image autoencoder probably suffers domain shift (their hypothesis).

**My reading**

- This is *output* correction, not KOH calibration: no posterior over physical parameters.
- Every training trajectory needs a cheap simulation. In the fluid study the "observations" are a finer simulator, so the corrected bias is pure discretisation error — the friendliest kind.
- The guidance (4) over-pulls at high noise, the implemented weight is ad hoc, $\gamma$ is never varied, and the theorem covers none of it.
- Missing baselines: full-Jacobian DPS, a denoiser trained on $c_2$ where it exists, and a GP with a spatial kernel or basis reduction.
- Despite the "surrogate" label, nothing is optimised through the model — no design loop, no acquisition function.

## 7 Extensions

**What was built on this.** No follow-up is in the collection. The closest relatives cited in the PDF are CoCoGen (PDE constraints during reverse diffusion), Shu et al. (PDE residuals as denoiser input), GenCFD, and PhysGen (refining simulator-rendered video). Here, the guidance traces to [classifier guidance](/blog/diffusion-beats-gans/), the sampler to [DDIM](/blog/ddim/), and the bound to the probability-flow view of [Score-SDE](/blog/score-sde/). Whether later work adopted the cheap-input / expensive-guidance split I could not check offline.

**Open problems.**
- How large is $\mathcal L_2$ under the Tweedie and no-Jacobian shortcuts, and can a cheap correction (a scalar rescaling such as the factor in Section 3.4, or a Jacobian–vector product) recover most of the gap?
- When is a fidelity level better used as an input than as a likelihood? The ablation hints that guidance alone can beat input conditioning on perceptual quality.
- Joint inference over simulator parameters and the state, as in full KOH, with the diffusion model as the discrepancy prior.
- Choosing *which* cases deserve the expensive simulation, the multi-fidelity design question that a Bayesian-optimisation reader would ask first, is untouched.

**Research directions.**

*These are ideas, not results — none has been run.*

1. **Calibrated guidance weight.** *Hypothesis:* rescaling $g$ by the Gaussian-case Jacobian ratio of Section 3.4 (per-pixel data variance as $s^2$) recovers most of full-Jacobian DPS at DBS's memory cost. *Data:* the paper's Boussinesq setup, via its public code. *Baseline:* DBS as published; DPS with backprop through the U-Net. *Metric:* MSE, LPIPS, peak GPU memory, time per sample. *Likely failure mode:* re-tuning $\gamma$ absorbs the whole difference — itself evidence that the weight is one free knob.

2. **Does guidance shrink uncertainty below calibration?** *Hypothesis:* pixel-wise 90% intervals from DBS + $c_1$ cover near nominal; adding $c_2$ guidance lowers coverage through the point-mass step. *Data:* fluid test set, 100+ draws per case. *Baseline:* KOH intervals; an NN ensemble. *Metric:* coverage, interval width, CRPS. *Likely failure mode:* coverage is driven mainly by the (unstated) number of sampling steps.

3. **Cheap model as default, expensive one as guidance, for return scenarios.** The owner's [TailFlow](/research/tailflow/) page describes a conditional DDPM for 10-day multi-asset returns, conditioned on the EWMA volatility state and whitened so that the untrained network defaults to a Gaussian copula with Student-$t$ margins — the network learns departures from an analytic model, as DBS's denoiser does from $c_1$. The page reports a 13–21% under-statement of 10-day risk, attributed to missing volatility clustering inside the window, and names day-by-day generation or conditioning on a latent volatility path as the obvious fix. *Hypothesis:* a DBS-style energy on each sample's 10-day realised variance, targeted at a costlier multi-step volatility forecast, reduces that under-statement without retraining. *Data:* TailFlow's synthetic regime-switching Student-$t$ market with known 10-day VaR/ES. *Baseline:* unguided TailFlow and filtered historical simulation. *Metric:* 10-day ES error against truth and the FZ0 score, with 1-day ES error not allowed to degrade. *Likely failure mode:* DBS guidance is per-sample, sensible when $c_2$ simulates *this* outcome; a volatility forecast describes a distribution, so per-path guidance would pull every scenario toward the forecast mean and thin the tails it was meant to restore — the energy may need to act on a batch statistic. (The owner's [exchange-queueing](/projects/exchange-queueing/) page has the same cheap/expensive pairing — Erlang-C says about one node where trace-driven simulation needs about 48 — but with a scalar output, not a field.)

## 8 Takeaways

- The useful idea is architectural: <mark>simulators that are cheap enough to run everywhere become denoiser inputs; simulators that are not become likelihood terms at sampling time</mark>, so expensive runs scale with the number of predictions, not with the training set.
- The Bayes split (3) is exact; the Tweedie point mass, dropped Jacobian, energy model and hand-set weight are approximations, and in 1-D with unit-variance data the dropped Jacobian alone over-weights guidance by $1/\bar\alpha_t$, before the point-mass step adds more.
- Against GP-based KOH on a 16k-pixel field the diffusion surrogate wins clearly on fluid (MSE 1.00 vs 1.33, LPIPS 0.228 vs 0.416); on printing video the win is perceptual, not pixel-wise.
- The bound leaves the guidance error — the novel part — unquantified, and the sampler run is not the one analysed.
- Uncertainty maps look sensible but are uncalibrated; test coverage before using them for decisions.
- For financial scenarios the transferable half is the cheap analytic model as the denoiser's default, which TailFlow already does its own way; the other half needs a per-scenario target, which finance rarely has.

## References

1. Shi, N., Yan, H., Guo, S., Al Kontar, R. *Diffusion-Based Surrogate Modeling and Multi-Fidelity Calibration.* arXiv:2407.17720, 2024 (v2 2025; the PDF carries a "TASE" running header). License CC BY-NC-ND 4.0.
2. Kennedy, M. C., O'Hagan, A. *Bayesian calibration of computer models.* JRSS-B 63(3), 2001.
3. Chung, H., Kim, J., McCann, M. T., Klasky, M. L., Ye, J. C. *Diffusion posterior sampling for general noisy inverse problems.* ICLR 2023.
4. Song, J., Meng, C., Ermon, S. *Denoising Diffusion Implicit Models.* arXiv:2010.02502, 2020 (as cited in the PDF).
5. Kwon, D., Fan, Y., Lee, K. *Score-based generative modeling secretly minimizes the Wasserstein distance.* NeurIPS 2022.
