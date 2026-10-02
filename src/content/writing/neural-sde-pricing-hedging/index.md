---
title: "Neural SDEs: Robust pricing and hedging via neural SDEs"
paper: { title: "Robust pricing and hedging via neural SDEs", authors: "Patryk Gierjatowicz et al.", venue: "arXiv 2020 (q-fin.MF)", arxiv: "2007.04154", license: "arxiv.org/licenses/nonexclusive-distrib/1.0/" }
series: "stochastic-modeling"
order: 10
tags: [neural-sde, model-uncertainty, robust-finance, calibration, local-volatility, control-variate, hedging, generative-model]
date: 2020-08-01
status: draft
summary: "Drift and diffusion of an arbitrage-free SDE are neural networks calibrated to vanilla prices; minimising and maximising an exotic's price over all such calibrated models gives data-driven robust price bounds, with a learned hedge doubling as a variance-reducing control variate."
---

## Abstract

Gierjatowicz, Sabate-Vidales, Šiška, Szpruch and Žurič keep the classical form of a financial model, an Itô SDE under a risk-neutral measure, but let neural networks play the role of its coefficients. Such a neural SDE is heavily over-parametrised, so many different parameter settings reproduce the same liquid option prices while disagreeing on illiquid ones. The authors turn that ambiguity into a feature: they search, within the calibrated family, for the lowest and highest price of a given exotic. Training is a Monte Carlo stochastic-gradient scheme in which a second network learns a hedging strategy that acts as a control variate; the paper proves that this variance reduction also shrinks the bias of the gradient estimator. Experiments calibrate neural local-volatility and local-stochastic-volatility models to Heston-generated vanilla prices and bound a lookback option.

**Keywords:** neural SDE, model uncertainty, robust price bounds, calibration, local volatility, local stochastic volatility, control variate, hedging, tamed Euler scheme, augmented Lagrangian

## 1 Introduction

Classical modelling proceeds by hand: collect stylised facts, write down a parsimonious SDE, calibrate a handful of parameters. Neural-network calibration, from Hernandez onward, removed much of that cost, but only for a *fixed* parametric model. It did nothing about the more serious question of which model to use, or how wrong the chosen one might be.

Robust finance answers that question in principle. Let $\mathcal{M}$ be all martingale measures that reprice the liquid instruments exactly; then an illiquid payoff $\Psi$ has the price interval $\big(\inf_{\mathbb{Q}\in\mathcal{M}}\mathbb{E}^{\mathbb{Q}}[\Psi],\ \sup_{\mathbb{Q}\in\mathcal{M}}\mathbb{E}^{\mathbb{Q}}[\Psi]\big)$. The trouble is twofold: without further structure the interval is often too wide to trade on, and the extremal measures are not explicit models that one can simulate, use under the real-world measure, or plug into a hedging engine.

The paper's position sits between the two. <mark>Keep a strong structural prior (a diffusion with the correct risk-neutral drift) but let the data choose the coefficients</mark>, and compute bounds over that restricted but still very large class. Concurrent work by Cuchiero, Khosrawi and Teichmann on a network leverage function is a special case.

## 2 Background

Let $\Phi_1,\dots,\Phi_M$ be discounted payoffs of liquid options with arbitrage-free market prices $p(\Phi_i)$. A parametric model with law $\mathbb{Q}(\theta)$ is perfectly calibrated if $\mathbb{E}^{\mathbb{Q}(\theta)}[\Phi_i]=p(\Phi_i)$ for all $i$.

Two classical facts frame the experiments. Dupire's result says a continuum of call prices in strike and maturity pins down a unique local volatility function. With finitely many quotes that uniqueness is lost, and the usual fix, interpolating the surface before applying Dupire's formula, is itself an unacknowledged modelling choice. Second, the martingale representation theorem: for square-integrable $\Phi$ there is an adapted $Z$ with $\mathbb{E}[\Phi\mid\mathcal{F}_0]=\Phi-\int_0^T Z_s\,dW_s$, so subtracting the right stochastic integral removes all variance from a Monte Carlo estimate.

## 3 Method

> **Key idea.** Over-parametrisation means calibration has many solutions. Do not pick one and hope; steer the optimiser to the calibrated solutions that make the exotic cheapest and dearest, and report that interval. Learn the hedge at the same time: it is the variance reduction that makes training work.

### 3.1 The neural SDE

The state $X^\theta=(S^\theta,V^\theta)$ splits into traded assets $S$ and non-traded factors $V$, driven by an $n$-dimensional Brownian motion $W$ under $\mathbb{Q}$, with constant rate $r$:

$$
dS^\theta_t = rS^\theta_t\,dt + \sigma^S(t,X^\theta_t,\theta)\,dW_t, \qquad
dV^\theta_t = b^V(t,X^\theta_t,\theta)\,dt + \sigma^V(t,X^\theta_t,\theta)\,dW_t. \tag{1}
$$

Here $\sigma^S$, $b^V$, $\sigma^V$ are feed-forward networks with weights $\theta$. Because the drift of $S$ is fixed at $rS$, the discounted price is a (local) martingale for every $\theta$: <mark>absence of arbitrage is built into the architecture, not learned</mark>. A real-world version follows by adding a third network $\zeta$ as market price of risk and applying Girsanov, so that a statistic under $\mathbb{P}(\theta)$ is just another expectation under $\mathbb{Q}(\theta)$ weighted by a density; calibrating to historical moments is then formally the same task as calibrating to option prices. The solution map from Brownian path to $X^\theta$ is adapted, so the neural SDE is a generative model given by a causal transport of Wiener measure.

### 3.2 Calibration and robust bounds

With a convex loss $\ell$ (squared error in practice), calibration and bounding are

$$
\theta^*\in\arg\min_\theta \sum_{i=1}^M \ell\big(\mathbb{E}^{\mathbb{Q}(\theta)}[\Phi_i],\,p(\Phi_i)\big), \tag{2}
$$

$$
\theta^{l,*}\in\arg\min_\theta \mathbb{E}^{\mathbb{Q}(\theta)}[\Psi],\quad
\theta^{u,*}\in\arg\max_\theta \mathbb{E}^{\mathbb{Q}(\theta)}[\Psi]
\quad\text{s.t.}\quad \sum_{i=1}^M \ell\big(\mathbb{E}^{\mathbb{Q}(\theta)}[\Phi_i],\,p(\Phi_i)\big)=0. \tag{3}
$$

The constraint is handled by an augmented Lagrangian: minimise $f(\theta)+\lambda h(\theta)+c\,h(\theta)^2$ with $f$ the exotic price and $h$ the calibration error, and every 50 steps update $\lambda\leftarrow\lambda+c\cdot\text{MSE}$, $c\leftarrow 2c$.

### 3.3 Hedging strategy as control variate

A second network $\bar h$ with weights $\xi$ takes time and the (stopped) path and outputs a position in the discounted asset $\bar S^\theta_t=e^{-rt}S^\theta_t$. It is trained, for fixed $\theta$, by

$$
\bar\xi^*\in\arg\min_{\bar\xi}\ \mathrm{Var}\Big[\Phi(X^\theta)-\int_0^T \bar h\big(s,(X^\theta_{s\wedge t})_{t},\bar\xi\big)\,d\bar S^\theta_s\Big]. \tag{4}
$$

Integrating against $\bar S$ instead of $W$ gives a tradable strategy, at the price of less variance reduction. Training alternates Adam epochs on $\theta$ (with $\xi$ frozen) and on $\xi$ (with $\theta$ frozen). Paths are simulated with a tamed Euler scheme, which divides each increment by $1+|\cdot|\sqrt{\Delta t}$ so that a network diffusion with super-linear growth cannot blow up the moments.

### 3.4 Why variance reduction fixes the gradient

Objective (2) has the expectation *inside* $\ell$, so a minibatch gradient $\partial_\theta h^N$ is biased. For squared loss the paper shows

$$
\big|\mathbb{E}[\partial_\theta h^N(\theta)]-\partial_\theta h(\theta)\big|
\le \frac{2}{N}\big(\mathrm{Var}[\Phi^{cv}(X^\theta)]\big)^{1/2}\big(\mathrm{Var}[\partial_\theta\Phi(X^\theta)]\big)^{1/2}, \tag{5}
$$

with $N$ the number of paths and $\Phi^{cv}$ the hedged payoff. <mark>A better hedge therefore means a less biased calibration gradient</mark>, not merely a less noisy one.

### 3.5 Randomised training across maturities

Coefficients use one network per maturity bucket $[T_{i-1},T_i]$. A path-dependent $\Psi$ couples all buckets, and back-propagating through all of them is memory-hungry. The authors instead sample one bucket uniformly per step and differentiate only its weights, rescaling by the number of buckets. They prove the resulting gradient is unbiased (Theorem 4.1); memory stays constant in the number of networks.

## 4 Experiments

**Setup.** "Market" data are call prices from a Heston model, maturities 2, 4, …, 12 months, typically 21 strikes in $[0.8,1.2]$. The exotic is a lookback paying $\max_{t\le T}S_t-S_T$. Two neural SDEs are tested: local volatility, $dS=rS\,dt+\sigma(t,S,\theta)S\,dW$, and a local-stochastic-volatility model with a second, non-traded factor and correlation $\rho$. Each coefficient network has 4 hidden layers of 50 ReLU units with a softplus on volatility outputs; hedge networks have 3 layers of 20. Simulation uses 96 time steps per year, $4\times10^4$ paths per gradient step, and $4\times10^5$ antithetic paths for evaluation. Every calibration is repeated from 10 initialisations.

**Calibration quality.** For the LV model, price MSE is about $10^{-9}$ at 6 months and $10^{-8}$ at 12 months when fitting only vanillas, and about $10^{-8}$ under the bound-seeking objectives; implied volatility errors are in basis points ([Fig. 5.2](https://arxiv.org/pdf/2007.04154#page=20)). With the control variate the calibration RMSE falls markedly faster per epoch than without ([Fig. 5.6](https://arxiv.org/pdf/2007.04154#page=22)). The learned lookback hedge at 6 months leaves residual variance $1.6\times10^{-3}$.

**Bounds.** The table collects the 12-month lookback price for the LV neural SDE, initialisation 1, Lagrange-multiplier variant, from the paper's Tables 1–4. The Heston generator itself prices the lookback at 0.174.

| Strikes per maturity | Lower bound | Unconstrained | Upper bound | Width |
|---|---|---|---|---|
| 11 (0.9–1.1) | 0.154 | 0.178 | 0.214 | 0.060 |
| 21 (0.8–1.2) | 0.153 | 0.182 | 0.205 | 0.052 |
| 31 (0.7–1.3) | 0.172 | 0.184 | 0.197 | 0.025 |
| **41 (0.6–1.4)** | **0.172** | **0.183** | **0.193** | **0.021** |

(Widths are my subtraction.) <mark>More quotes shrink the interval</mark>, consistent with the conjecture that the Dupire-unique model is recovered in the limit. At 2 months all variants agree to the third decimal; the ambiguity grows with maturity. The LSV model shows similarly visible ranges ([Fig. 5.3](https://arxiv.org/pdf/2007.04154#page=21)). Notably, <mark>even plain re-initialisation of an unconstrained fit moves the exotic price</mark>, so the training algorithm is part of the model.

## 5 Discussion

**Strengths.** The framing is the contribution: model risk stops being a footnote and becomes a computed interval attached to an explicit, simulable model. The martingale constraint is structural. Theorem 3.2 gives a rare, clean reason why a particular variance-reduction device helps optimisation, and the randomised-maturity gradient is a practical trick with a proof.

**Weaknesses.** The bounds are over what SGD can reach inside one architecture, not over all calibrated diffusions. They are inner estimates of the robust interval with no certificate, and the tables show the two constrained solvers and the two seeds disagreeing noticeably at long maturities. Everything is synthetic and single-asset; the target is a Heston surface, so there is no bid–ask noise, no arbitrage in the quotes, and no test of the real-world calibration that the introduction advertises. The "exact calibration" constraint is met only up to MSE $\sim10^{-8}$, and how much slack that gives the exotic is not quantified. Path-dependent coefficients and non-Brownian noise are deferred.

**Not shown.** No comparison with martingale-optimal-transport bounds on the same data, which would show how much the diffusion prior tightens things; no out-of-sample hedging P&L under a model different from the one the hedge was trained on.

## 6 Takeaways

- A neural SDE is a classical arbitrage-free diffusion whose coefficients are networks; interpretability lives in the structure, not in the weights.
- Non-uniqueness of calibration is exploited: minimise and maximise the exotic price subject to fitting the vanillas, and quote the interval.
- The hedging network doubles as a control variate, and lower variance provably lowers the bias of the nested-expectation gradient.
- More market quotes narrow the bounds (width 0.060 to 0.021 for the 12-month lookback as strikes go from 11 to 41); seed and optimiser still matter.
- For diffusion-style generative modelling of financial series this is directly relevant: it is a learned SDE generator with a hard no-arbitrage constraint, trained on option prices instead of a likelihood or score, and its measure-change construction suggests how one generator could be fitted jointly to historical paths under $\mathbb{P}$ and option prices under $\mathbb{Q}$. The paper sketches that joint fit but does not run it.

## References

- P. Gierjatowicz, M. Sabate-Vidales, D. Šiška, L. Szpruch, Ž. Žurič. *Robust pricing and hedging via neural SDEs.* arXiv:2007.04154, 2020.
- C. Cuchiero, W. Khosrawi, J. Teichmann. *A generative adversarial network approach to calibration of local stochastic volatility models.* arXiv:2005.02505, 2020.
- A. Hernandez. *Model calibration with neural networks.* Risk, 2016.
- B. Dupire. *Pricing with a smile.* Risk, 1994.
- M. Beiglböck, P. Henry-Labordère, F. Penkner. *Model-independent bounds for option prices: a mass transport approach.* Finance and Stochastics, 2013.
