---
title: "PIML for battery diagnostics: Physics-Informed Machine Learning for Battery Degradation Diagnostics — A Comparison of State-of-the-Art Methods"
paper:
  title: "Physics-Informed Machine Learning for Battery Degradation Diagnostics: A Comparison of State-of-the-Art Methods"
  authors: "Sina Navidi et al."
  venue: "arXiv preprint 2024"
  arxiv: "2404.04429"
  license: "creativecommons.org/licenses/by-nc-nd/4.0/"
series: "ee-timeseries"
order: 6
tags: [battery, degradation-diagnostics, physics-informed-ml, pinn, co-kriging, delta-learning, data-augmentation, extrapolation]
date: 2024-05-01
status: draft
summary: "Four ways of injecting a half-cell physics model into small ML models are compared on the task of predicting late-life electrode degradation from about seven months of test data."
---

## Abstract

Capacity fade tells you that a lithium-ion cell has aged, not which component aged. This paper targets the component level: the active mass left on each electrode and the remaining lithium inventory, estimated from a slow-rate incremental capacity curve. The catch is that the models may only train on early-life measurements, yet they are scored on data collected years later. The authors use a half-cell open-circuit-voltage model as the physics source and compare four ways of combining it with lightweight learners: a physics-informed neural network (PINN), multi-fidelity co-kriging, delta learning with an elastic net, and plain data augmentation. On 24 implantable-grade cells aged for more than four years, <mark>every physics-informed variant beats its purely data-driven counterpart on late-life error</mark>, with the largest gain going to the neural network. The paper closes with a qualitative guide to choosing among them.

**Keywords:** lithium-ion battery, degradation modes, half-cell model, physics-informed neural network, co-kriging, delta learning, extrapolation

## 1 Introduction

Standard health indicators (capacity, internal resistance) are cell-level quantities. Two cells with the same capacity loss can have very different internal damage, so the diagnostics literature tracks three degradation modes instead: loss of active material at the positive electrode (LAM$_{PE}$), at the negative electrode (LAM$_{NE}$), and loss of lithium inventory (LLI). Measuring them directly means dismantling the cell.

Non-destructive alternatives have their own costs. Fitting a physical model to voltage curves is iterative and partly manual; pure ML regressors need labelled late-life data, which takes years to collect. Thelen et al. showed that small models trained on early-life plus simulated data can predict late-life modes, given some foreknowledge of where the cells will end up. The present paper asks the broader question: <mark>given the same physics model and the same scarce data, which integration strategy extrapolates best, and at what engineering cost?</mark>

## 2 Background

**Half-cell model.** At very low current (C/50 here), the full-cell voltage curve is approximately the difference between the positive and negative electrode open-circuit curves. Four parameters place those two curves: the active masses $m_p$ and $m_n$ stretch each electrode's capacity axis ($Q_p = m_p q_p$, $Q_n = m_n q_n$, with $q$ the specific capacity), and two slippage terms $\delta_p$, $\delta_n$ shift them horizontally. Cell capacity $Q$ follows from clipping the result to the usable voltage window, and the lithium inventory indicator is

$$
\mathrm{LII} = Q_p - (\delta_p - \delta_n). \tag{1}
$$

The model plays two roles. Fitted to each reference performance test (RPT), it supplies the "ground truth" labels $(Q, m_p, m_n, \mathrm{LII})$. Run forward over a parameter grid, it generates synthetic $dQ/dV(V)$ curves with labels, including heavily degraded states.

**Data.** 24 lithium-cobalt-oxide/graphite cells, six groups of four, varying discharge rate (C/24, C/10, C/3), temperature (37 °C, 55 °C), and upper cutoff voltage (4.075 V, 4.175 V). A few cells were disassembled partway through to check the fitted active masses.

## 3 Method

> **Key idea.** All four methods see the same two information sources, a small set of early-life experimental curves and a large set of half-cell simulations. They differ only in *where* the physics enters: the loss function (PINN), the prior over functions (co-kriging), a bias-correction stage (delta learning), or just the training set (augmentation).

Every model maps a 100-point $dQ/dV(V)$ curve to the four health parameters.

### 3.1 PINN

Instead of predicting health parameters directly, a two-hidden-layer network (30 units, 2274 weights) predicts the half-cell parameters $\hat y = [\hat m_p, \hat m_n, \hat\delta_p, \hat\delta_n]$. A differentiable surrogate $f_{hc}$ of the half-cell model then maps these to $\hat U = [\hat Q, \widehat{\mathrm{LII}}]$ and to the voltages of the two dominant $dQ/dV$ peaks. The loss is

$$
\mathcal{L}_{total} = \underbrace{\mathrm{MSE}(\hat y, y)}_{\mathcal{L}_1} + \lambda_1 \underbrace{\mathrm{MSE}(\hat U, U)}_{\mathcal{L}_2} + \lambda_2 \underbrace{\mathrm{MSE}(\hat V_{peak}^{sim}, V_{peak}^{exp})}_{\mathcal{L}_3}, \tag{2}
$$

with $\lambda_1,\lambda_2$ set empirically. $\mathcal{L}_2$ ties the end-points of the reconstructed curve to physics; $\mathcal{L}_3$ constrains the active masses, because peak positions are sensitive to $m_p$ and $m_n$. A surrogate replaces the half-cell code so that gradients flow through the physics terms. Training also includes simulated samples from the most-degraded 20% of the parameter range. At test time the network runs alone.

### 3.2 Co-kriging

Simulation is treated as a low-fidelity process $f_L$ and experiment as a high-fidelity one $f_H$, linked autoregressively:

$$
f_H(x) = \rho\, f_L(x) + f_\Delta(x), \tag{3}
$$

where $\rho$ is a scalar regression coefficient and $f_\Delta$ a Gaussian process for the discrepancy. One GPR is fit to the simulations, and a second to the residuals at the experimental inputs. Both use a Matérn kernel with $\nu = 3/2$. Unlike the others, this method yields a predictive variance for free.

### 3.3 Delta learning with elastic net

The same two-stage structure with deterministic linear models: one elastic net trained on simulations over the whole lifetime, a second trained on early-life data to learn the first one's bias. <mark>The bet is that a bias learned in the lightly aged region carries over to the heavily aged one.</mark>

### 3.4 Data augmentation

Simulated samples from the whole parameter space are appended to the early-life set and one elastic net is trained on the union. [Fig. 5 in the paper](https://arxiv.org/pdf/2404.04429#page=10) shows the four pipelines side by side.

## 4 Experiments

**Setup.** Training uses the first 10 RPTs per cell, about seven months of aging; testing uses everything afterwards, out to more than four years. Four-fold cross-validation holds out one cell per group in each fold, giving 180 training points for baselines, 212 for PINN and augmentation, 964 for delta learning, and 192–200 test points. Each baseline is the same learner without simulation data or physics terms; runs are repeated ten times. The metric is a relative error pooled over folds,

$$
\mathrm{RMSPE}_t = \sqrt{\frac{1}{\sum_k N_k}\sum_{k=1}^{4}\sum_{i=1}^{N_k}\left(\frac{\hat y_{t,i}-y_{t,i}}{y_{t,i}}\right)^2}\times 100\%, \tag{4}
$$

for health parameter $t$ with $N_k$ test points in fold $k$.

**Main comparison.** The paper reports these numbers as labelled bars in [Fig. 8](https://arxiv.org/pdf/2404.04429#page=19); retyped below (mean late-life RMSPE, %; lowest per column in bold).

| Model | $Q$ | $m_p$ | $m_n$ | LII |
|---|---|---|---|---|
| Baseline NN | 5.72 | 14.07 | 16.17 | 7.27 |
| PINN | 1.31 | 4.52 | **5.56** | 1.51 |
| Baseline kriging (GPR) | 0.93 | 9.58 | 9.85 | 2.13 |
| Co-kriging | 1.37 | 4.29 | 8.68 | 1.74 |
| Baseline elastic net | 1.02 | 5.49 | 7.06 | 3.87 |
| Delta learning (elastic net) | **0.55** | 5.04 | 6.63 | 2.13 |
| Data augmentation (elastic net) | 0.77 | **3.68** | 6.59 | **1.45** |

Three readings. First, <mark>the PINN gains the most over its own baseline, cutting error by roughly a factor of three to five on every parameter</mark>, and its spread across runs is much narrower. Second, no method wins everywhere: the elastic-net variants are best on $Q$, $m_p$ and LII, the PINN only on $m_n$. Third, <mark>the plain elastic net is already a strong baseline, ahead of the unconstrained network on all four targets</mark>. Active masses are harder than $Q$ and LII for every model.

**Sample size and extrapolation.** With 5, 10 or 15 RPTs per cell (about 3, 8 and 12 months), most models improve from 90 to 180 points and little beyond; the PINN is nearly flat. For one 55 °C cell, the unconstrained network's error grows with degradation depth while the PINN's stays small ([Fig. 10](https://arxiv.org/pdf/2404.04429#page=23)).

**Sensitivity.** Dropping any single loss term, or using one alone, raises PINN error; equal weights are best for $Q$, $m_n$ and LII. A crossed ablation (network with or without the physics loss, with or without simulated samples) shows that <mark>augmentation alone helps the baseline network only marginally, while removing augmentation from the PINN hurts only marginally: the loss function does the work.</mark>

## 5 Discussion

**Strengths.** Same input, targets and folds, and each method is matched against its own physics-free twin, so the gain from physics is separated from the gain from the learner. The split is temporal, so this tests extrapolation. The loss-versus-augmentation ablation is the most reusable result.

**Weaknesses.** The labels are themselves half-cell fits, so "truth" and "physics prior" come from the same model; agreement is partly built in, and destructive validation covers only a few cells. The PINN is compared with a weak baseline, and in absolute terms it loses to augmented elastic net on three of four targets. The prose is also kinder to co-kriging than the figure: the text describes capacity error as improved, while Fig. 8 shows 1.37% against 0.93% for plain kriging. The qualitative "High/Medium/Low" table is opinion, not measurement; the PINN's "high" generalization is not tested on unseen operating conditions, since every fold contains cells from all six groups.

**Not shown.** Calibration of the co-kriging variance, runtime numbers, and any statistical test on the differences.

## 6 Takeaways

- Where physics enters matters more than whether simulated data is present: a constraint in the loss changed the network's extrapolation far more than extra synthetic samples did.
- With a few hundred points, regularised linear models plus simulated data are hard to beat; the PINN's headline improvement is relative to a poor starting point.
- Two-stage correction is cheap, and the GP version adds uncertainty estimates, but it assumes early-life bias persists into late life.
- When labels are produced by the same physical model used as the prior, reported accuracy measures consistency with that model, not necessarily with the cell.

## References

1. S. Navidi, A. Thelen, T. Li, C. Hu. *Physics-Informed Machine Learning for Battery Degradation Diagnostics: A Comparison of State-of-the-Art Methods.* arXiv:2404.04429, 2024.
2. A. Thelen et al. *Integrating physics-based modeling and machine learning for degradation diagnostics of lithium-ion batteries.* Energy Storage Materials, 50:668–695, 2022.
3. C. R. Birkl et al. *Degradation diagnostics for lithium ion cells.* Journal of Power Sources, 341:373–386, 2017.
4. G. E. Karniadakis et al. *Physics-informed machine learning.* Nature Reviews Physics, 3(6):422–440, 2021.
