---
title: "DBS: Diffusion-Based Surrogate Modeling and Multi-Fidelity Calibration"
paper:
  title: "Diffusion-Based Surrogate Modeling and Multi-Fidelity Calibration"
  authors: "Naichen Shi, Hao Yan, Shenghan Guo, Raed Al Kontar"
  venue: "IEEE Transactions on Automation Science and Engineering, 2025"
  arxiv: "2407.17720"
  license: "creativecommons.org/licenses/by-nc-nd/4.0/"
series: "surrogates-bo"
order: 6
tags: [surrogate-modeling, multi-fidelity, model-calibration, diffusion, conditional-diffusion, guidance, physics-informed, uncertainty-quantification, kennedy-ohagan]
date: 2024-07-01
status: draft
summary: "What I took from it: cheap simulations become inputs to the denoiser and expensive ones become likelihood terms at sampling time, so expensive runs scale with the number of predictions rather than the size of the training set. The Bayes split behind it is exact; the guidance shortcuts are not, and in a one-dimensional check they over-weight the expensive simulation at high noise."
---
## Why I read it

I have used diffusion models as scenario generators ([TailFlow](/research/tailflow/), [CASE](/research/diffusion-scenarios/)) and I am now building a physics simulator of battery degradation. This paper is where the two meet: a diffusion model used as the calibration layer on top of simulators of different cost. I wanted to understand how it decides what each simulator is for.

## The problem, in one paragraph

Physics simulators encode the physics but are biased; field observations are faithful but scarce. The classical fix, Kennedy–O'Hagan (KOH) calibration, puts a Gaussian process on the gap between simulator and reality, which does not scale to a $128\times128$ field. Conditioning a diffusion model on every simulator would scale, but the expensive simulator would then have to run on every training case. That cost is the real constraint, and the paper is designed around it.

## The idea, as I understand it

There are two roles, and the whole method follows from keeping them apart:

- **The cheap simulation is context.** It is available for every case, so it becomes an extra input channel of the denoiser during training. This is ordinary conditional diffusion, new only in what is conditioned on.
- **The expensive simulation is guidance.** It exists only for some cases, so it never touches training. At sampling time, Bayes' rule splits the score exactly:
$$
\nabla_{x_t}\log p(x_t\mid h,c_1,c_2)=\underbrace{\nabla_{x_t}\log p(x_t\mid h,c_1)}_{\text{trained denoiser}}+\underbrace{\nabla_{x_t}\log p(c_2\mid x_t,h,c_1)}_{\text{guidance from the expensive run}},
$$
where $h$ is the observed history, $c_1$ the cheap output, $c_2$ the expensive one and $x_t$ the noised state. The second term is evaluated through a user-designed energy $E(c_2,x)$, at the denoised estimate of the state.

> **My comment.** In CASE everything the engine knows enters as trained context, through 37 condition channels, which is why one frozen checkpoint can be re-conditioned each morning. Information that exists only on some decision dates, such as a desk's hand-written stress view, could not enter that way, and this split says it should enter as guidance instead. I would be careful with the weight, though: the high-noise steps that my check below shows are over-weighted are, I suspect, also where a scenario's overall volatility level gets set.

<mark>Cheap simulation as context, expensive simulation as guidance: that division is what lets the expensive runs scale with the number of predictions rather than with the training set.</mark> A Wasserstein bound ties sample quality to the error of the denoiser and of the guidance term.

## My check: what the guidance shortcuts do

The second term has no closed form, so the paper makes two approximations. It treats the denoising posterior as a point mass at its mean, and it does not differentiate through the network. To see what they cost, I worked out the case where everything is Gaussian and exact. Take one dimension, $x\sim\mathcal N(m,s^2)$ given $c_1$, and the quadratic energy $E=(c_2-x)^2$. Comparing the shortcut with the exact gradient, the dropped Jacobian alone over-weights the guidance by a factor $1+(1-\bar\alpha_t)/(\bar\alpha_t s^2)$, which is $1/\bar\alpha_t$ for unit-variance data. That is 2 at $\bar\alpha_t=0.5$ and 10 at $\bar\alpha_t=0.1$, falling to 1 as sampling ends. The point-mass step pushes the same way.

<mark>So the guidance is too strong at high noise and right only near the end of sampling.</mark> In practice the paper also changes the weight: the fluid experiment drops the $(1-\alpha_t)$ factor for stability. The guidance strength actually used is a tuned constant rather than the Bayes-rule weight, and the bound does not cover it.

## What the results show

Fluid case: predicting a $128\times128$ buoyancy field ten time units ahead, with a $32\times32$ run of the same solver as the cheap simulator and a $64\times64$ run as the expensive one.

| fluid, test set | MSE ($\times10^{-3}$) | LPIPS |
|---|---|---|
| KOH (Gaussian process on the gap) | 1.33 | 0.416 |
| diffusion, no physics | 3.69 | 0.401 |
| DBS, cheap simulation as input | 1.14 | 0.287 |
| DBS, expensive guidance only | 1.12 | **0.216** |
| DBS, both | **1.00** | 0.228 |

Metal 3-D printing, predicting infrared frames: KOH keeps the best PSNR (25.1 against 24.2 for DBS with both simulators). The expensive guidance moves one metric, a consistency score, from 1.41 to 0.064.

How I read it:

- **Cheap physics as input clearly helps.** On the fluid case, MSE drops from 3.69 to 1.14; on printing, PSNR rises from 21.3 to 24.3.
- **The expensive guidance adds mostly perceptual quality.** On the fluid case, LPIPS improves from 0.287 to 0.228, while MSE moves by about two reported standard deviations. On printing, the only metric that moves is the consistency score, which is the same warp residual the guidance energy minimises. A plain neural network reaches the same 0.064 with frames the paper itself calls over-dispersed, so a low score alone does not mean realistic spatter.
- **More physics is not monotonically better.** Guidance alone has the best LPIPS of all variants (0.216); adding the cheap input helps pixel error and costs perceptual quality.
- **The win over Gaussian processes is real but against a naive baseline.** The KOH baseline is an independent multi-output GP on 16,384 pixels, with no spatial correlation and no parameter calibration.

## Where I am not convinced

- **This is output correction, not KOH calibration.** There is no posterior over the simulator's physical parameters.
- **The fluid "observations" are a finer simulator,** so the bias being corrected is discretisation error, the friendliest kind of bias.
- **The novel part is unquantified.** The bound leaves the guidance error open, the implemented weight is ad hoc, and the energy temperature is never varied.
- **The cost split is never priced.** No simulator or sampler is timed, although cost is the motivation.
- **Uncertainty is shown as maps only.** There is no coverage or CRPS, and the analysis above suggests guided samples will be under-dispersed.

## What I take from it

- **The architectural rule is what I will reuse:** what is cheap enough to run everywhere becomes context; what is not becomes a likelihood, spent only where it matters.
- **My battery runs show the assumption that needs care.** DBS assumes the cheap simulation exists for every case. In my [battery degradation simulator](/research/battery-degradation-sim/), PyBaMM's reduced model (SPMe) failed at 2–3C charging and I dropped it. That is exactly the fast-charging region where the decision is made, so the cheap channel was missing where it mattered most. A practical version needs to know where the cheap fidelity is valid.
- **And where the expensive run should go.** In the same simulator, a Gaussian-process surrogate chose charging rates that met a 60 °C limit at 15 and 25 °C, but at 45 °C the simulated peak reached 60.7 °C. The temperature response has a kink the coarse grid could not resolve. That is the kind of place where expensive guidance, spent near the constraint boundary, would earn its cost. The question of which cases deserve the expensive simulation is left open in the paper, and it looks like the real design problem to me.
- **Uncertainty must be tested before it drives a decision.** My exchange-queue study made the same point: a posterior that promised 97% service-level coverage delivered 84%.

## What I would try next

*Ideas, not results.*

1. **A calibrated guidance weight.** Rescale the guidance by the Jacobian ratio from the Gaussian check above, and compare it with full posterior-sampling guidance that differentiates through the network. Measure error, memory and time on the paper's public fluid code. If re-tuning the energy temperature absorbs the whole difference, that itself shows the weight is one free knob.
2. **Choose where the expensive simulation runs.** Spend expensive runs where they most reduce a decision loss, such as near the battery temperature limit, instead of uniformly, and compare the decision quality at equal simulation budget.

## References

1. N. Shi, H. Yan, S. Guo, R. Al Kontar. *Diffusion-Based Surrogate Modeling and Multi-Fidelity Calibration.* IEEE Transactions on Automation Science and Engineering, 2025. doi:10.1109/TASE.2025.3582171. arXiv:2407.17720. Code: github.com/UMDataScienceLab/MGDM.
2. M. C. Kennedy, A. O'Hagan. *Bayesian calibration of computer models.* JRSS-B 63(3), 2001.
3. H. Chung, J. Kim, M. T. McCann, M. L. Klasky, J. C. Ye. *Diffusion posterior sampling for general noisy inverse problems.* ICLR 2023.
4. J. Song, C. Meng, S. Ermon. *Denoising Diffusion Implicit Models.* ICLR 2021.
5. D. Kwon, Y. Fan, K. Lee. *Score-based generative modeling secretly minimizes the Wasserstein distance.* NeurIPS 2022.
