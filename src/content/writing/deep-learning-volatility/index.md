---
title: "Deep Learning Volatility: A deep neural network perspective on pricing and calibration in (rough) volatility models"
paper: { title: "Deep Learning Volatility: A deep neural network perspective on pricing and calibration in (rough) volatility models", authors: "Blanka Horvath et al.", venue: "arXiv 2019 (q-fin.MF)", arxiv: "1901.09647", license: "arxiv.org/licenses/nonexclusive-distrib/1.0/" }
series: "stochastic-modeling"
order: 8
tags: [rough-volatility, rough-bergomi, calibration, implied-volatility, surrogate-model, monte-carlo, option-pricing]
date: 2019-02-01
status: draft
summary: "A small feed-forward network learns the map from stochastic-volatility parameters to a whole implied-volatility grid offline, so that calibrating rough Bergomi becomes a millisecond deterministic least-squares problem."
---

## Abstract

Horvath, Muguruza and Tomas attack the practical obstacle that keeps rough volatility models off trading desks: every price needs a slow Monte Carlo run, and calibration needs thousands of prices. Their answer is to split calibration in two. Offline, a neural network is trained on synthetic data to reproduce the map from model parameters to an 8 × 11 grid of implied volatilities, treated like the pixels of an image. Online, a standard optimiser fits the parameters by evaluating that network instead of the simulator. For rough Bergomi and one-factor Bergomi with piecewise-constant forward variance curves, the surrogate stays within Monte Carlo accuracy on average, evaluates a full surface in tens of microseconds, and calibrates to about ten years of SPX smiles.

**Keywords:** rough volatility, rough Bergomi, implied volatility surface, model calibration, neural network surrogate, Levenberg–Marquardt, Monte Carlo pricing

## 1 Introduction

The authors open with an uncomfortable observation: models become popular because they are fast, not because they are right. SABR owes its position to an asymptotic formula, Heston to Fourier pricing, Black–Scholes to a closed form. Rough volatility models sit at the other extreme. The fractional kernel destroys the Markov property, so PDE methods are unavailable and one is left with Monte Carlo. A calibration loop that calls a half-second pricer thousands of times is not usable in production.

Earlier neural approaches, starting with Hernandez, learned the *inverse* map directly: market quotes in, model parameters out. That is fast, but it fuses pricing and calibration into one opaque object. The paper argues for the opposite design. <mark>Learn only the forward pricing map, and leave calibration as an explicit deterministic optimisation on top of it.</mark> The training data is then unlimited (it comes from the simulator), the target is a smooth function, and risk managers still face the same model they already understand.

## 2 Background

**Calibration.** A model $\mathcal{M}(\theta)$ with parameters $\theta \in \Theta \subset \mathbb{R}^n$ is fitted to market quotes $P^{MKT}(\zeta)$ of contracts $\zeta$ by

$$
\hat\theta = \arg\min_{\theta\in\Theta}\; \delta\big(\tilde P(\mathcal{M}(\theta),\zeta),\; P^{MKT}(\zeta)\big), \tag{1}
$$

where $\delta$ is a distance and $\tilde P$ is a numerical approximation of the true pricing map, since a closed form rarely exists. Every optimiser spends its time evaluating $\tilde P$ at trial parameters, so the cost per evaluation is the bottleneck.

**Rough Bergomi.** The main test case has log-price $X$ and variance $V$ given by

$$
dX_t = -\tfrac12 V_t\,dt + \sqrt{V_t}\,dW_t, \qquad
V_t = \xi_0(t)\,\mathcal{E}\!\Big(\sqrt{2H}\,\nu\int_0^t (t-s)^{H-1/2}\,dZ_s\Big), \tag{2}
$$

with Hurst parameter $H\in(0,1)$, volatility-of-volatility $\nu>0$, Brownian motions $W,Z$ with correlation $\rho$, stochastic exponential $\mathcal{E}$, and initial forward variance curve $\xi_0(\cdot)$. To make $\theta$ finite-dimensional the curve is taken piecewise constant between option maturities. The one-factor Bergomi model replaces the power kernel with $\eta\,e^{-\beta(t-s)}$; Heston and SABR are also discussed.

## 3 Method

> **Key idea.** Treat the simulator as a teacher. Spend the Monte Carlo budget once, offline, to train a network that outputs the entire implied-volatility surface for any parameter vector; afterwards calibration is an ordinary least-squares fit against a function that costs microseconds and has exact gradients.

```mermaid
flowchart LR
  A[Sample parameters θ] --> B[Monte Carlo pricer]
  B --> C[8 x 11 implied vol grid]
  A --> D[Network F]
  C --> D
  D -->|offline, once| E[Trained surrogate]
  E --> F[Levenberg-Marquardt on market smiles]
  F -->|online, ms| G[Calibrated θ]
```

### 3.1 Two-step calibration

Problem (1) is rewritten as

$$
\text{(i) learn } \tilde F(\theta) \approx \tilde P(\mathcal{M}(\theta),\zeta), \qquad
\text{(ii) calibrate } \hat\theta = \arg\min_{\theta\in\Theta} \delta\big(\tilde F(\theta),\, P^{MKT}(\zeta)\big). \tag{3}
$$

The accuracy target is modest: if the simulator is within $O(\epsilon)$ of the true price, the network only needs to be within $O(\epsilon)$ of the simulator.

### 3.2 The surface as an image

The design choice the authors stress most is the output format. Instead of feeding $(\theta, T, k)$ and predicting one volatility, the network takes $\theta$ alone and returns all values on a fixed grid $\Delta=\{(T_i,k_j)\}$ of $n=8$ maturities and $m=11$ strikes. With $F^*(\theta)_{ij}=\sigma_{BS}^{\mathcal{M}(\theta)}(T_i,k_j)$ the Black–Scholes implied volatility under the model, the weights solve

$$
\hat w = \arg\min_{w}\sum_{u=1}^{N_{train}}\sum_{i=1}^{n}\sum_{j=1}^{m}\big(F(\theta_u,w)_{ij}-F^*(\theta_u)_{ij}\big)^2 . \tag{4}
$$

Three arguments are given for this grid-based training. Neighbouring grid points share information, as pixels do. <mark>Two different parameter vectors are much less likely to agree on 88 points than on one, which helps the inverse problem stay well posed.</mark> And since the same simulated paths price every strike and maturity, refining the grid costs almost no extra simulation. Off-grid points are handled by standard arbitrage-free interpolation, as with market quotes.

### 3.3 Architecture and training

The network is deliberately small: four hidden layers of 30 units, ELU activations, linear output of size 88. For the 11-parameter Bergomi-type models this is 6,808 weights. ELU is chosen over ReLU for a specific reason: the approximation theorem of Hornik, Stinchcombe and White for derivatives requires a smooth activation, and the calibration step relies on $\nabla_\theta \tilde F \approx \nabla_\theta \tilde P$. Inputs are rescaled to $[-1,1]$, outputs are standardised, batch size is 32, and training runs up to 200 epochs with early stopping.

### 3.4 The calibration step

With $\tilde F$ fixed, the online problem is

$$
\hat\theta=\arg\min_{\theta\in\Theta}\sum_{i,j}\big(\tilde F(\theta)_{ij}-\sigma^{MKT}_{BS}(T_i,k_j)\big)^2. \tag{5}
$$

Gradient-based solvers (Levenberg–Marquardt, BFGS, L-BFGS-B, SLSQP) are compared with gradient-free ones (Nelder–Mead, COBYLA, differential evolution). The former are fast but local; the latter are global but scale badly with the number of parameters. Levenberg–Marquardt is adopted as the default.

## 4 Experiments

**Setup.** For each model 80,000 parameter vectors are drawn uniformly, e.g. for rough Bergomi $\xi_0\in[0.01,0.16]^8$, $\nu\in[0.5,4]$, $\rho\in[-0.95,-0.1]$, $H\in[0.025,0.5]$; 68,000 are used for training and 12,000 for testing. Each surface is priced with 60,000 Monte Carlo paths and a spot-martingale control variate, on strikes 0.5 to 1.5 and maturities 0.1 to 2.0 years.

**Speed.** The paper's timing table, per full surface on CPU:

| Evaluation | Time per full surface |
|---|---|
| Monte Carlo, 1-factor Bergomi | 300,000 µs |
| Monte Carlo, rough Bergomi | 500,000 µs |
| **Network, surface** | **30.9 µs** |
| Network, surface gradient | 113 µs |
| Speed-up, network vs. Monte Carlo | 9,000–16,000× |

**Accuracy.** Heat maps of relative error over the grid ([Fig. 6 in the paper](https://arxiv.org/pdf/1901.09647#page=20)) show that <mark>the mean relative error between network and Monte Carlo is well under 0.5% everywhere, with standard deviation below 1%</mark>, comparable to the Monte Carlo confidence band itself. The maximum error, however, reaches 25% at some grid points.

**Synthetic calibration.** Calibrating with Levenberg–Marquardt to test-set surfaces, the 99% quantile of the surface RMSE is below 1% for both models.

**SPX history.** Rough Bergomi with five forward-variance levels is calibrated daily to SPX smiles from January 2010 to March 2019, on maturities of 1 to 12 months and strikes 0.85 to 1.25. <mark>The calibrated $H$ stays below one half and mostly in $[0.1, 0.15]$</mark>, in line with earlier time-series estimates. Against a brute-force Monte Carlo calibration the RMSE gap is below 0.2% most of the time ([Fig. 12](https://arxiv.org/pdf/1901.09647#page=25)), and the network fit is sometimes better, which the authors attribute to exact network gradients versus finite differences on a noisy pricer. Differential evolution beats Levenberg–Marquardt on the surrogate, which they read as a hint that <mark>the network's first derivatives may not be accurate enough</mark>, and leave open.

**Exotics and model recognition.** Replacing the strike axis by a barrier axis, the same recipe prices digital down-and-in and down-and-out options under rough Bergomi with average absolute error under 10 bp. A closing proof of concept trains a classifier to recognise which model generated a surface.

## 5 Discussion

**Strengths.** The division of labour is the lasting contribution. Because the network only replaces a deterministic function of parameters, its error can be audited against the simulator on as many points as one likes, and nothing about the model's hedging or risk interpretation changes. Handling a piecewise-constant forward variance curve is what makes the method usable on real term structures. Code is released.

**Weaknesses.** The error plots are reported over the 68,000 *training* parameter sets according to their captions; out-of-sample generalisation is asserted in the text but I could not find matching test-set heat maps. A 25% worst-case error is large, and the paper does not say where in parameter space it occurs. The surrogate is tied to one grid and one parameter box; a market regime outside $[0.01,0.16]$ forward variance requires retraining. Nothing enforces absence of static arbitrage in the output surface. The offline cost, 80,000 surfaces at roughly half a second each, is not discussed.

**Not shown.** No comparison with the direct inverse-map approach on the same data, no noise-robustness study for market quotes, and no analysis of parameter identifiability beyond the CDF plots.

## 6 Takeaways

- Separate pricing from calibration: learn the forward map offline, optimise online. The network is a fast numerical scheme for a known model, not a new model.
- Predicting the whole surface on a fixed grid reuses simulated paths and is better conditioned to invert.
- Smooth activations matter when the surrogate's gradients drive the optimiser; even so, derivative accuracy is the weakest link reported.
- With a 9,000–16,000× speed-up, rough Bergomi with a term structure of forward variance can be recalibrated daily across a decade of SPX data, and the implied roughness agrees with historical estimates.
- For generative modelling of financial series the relevance is indirect but concrete: the surrogate pattern (expensive stochastic simulator as teacher, cheap network as student, evaluated against the teacher's own Monte Carlo error) applies equally to a learned path generator whose option prices must be matched to a surface. The paper itself learns no dynamics; the SDE is fixed.

## References

- B. Horvath, A. Muguruza, M. Tomas. *Deep Learning Volatility: A deep neural network perspective on pricing and calibration in (rough) volatility models.* arXiv:1901.09647, 2019.
- C. Bayer, P. Friz, J. Gatheral. *Pricing under rough volatility.* Quantitative Finance, 2015.
- J. Gatheral, T. Jaisson, M. Rosenbaum. *Volatility is rough.* Cited in the paper as the source of the historical estimate of $H$.
- A. Hernandez. *Model calibration with neural networks.* Risk, 2016.
- C. Bayer, B. Stemper. *Deep calibration of rough stochastic volatility models.* arXiv:1810.03399, 2018.
