---
title: "Heteroscedastic ER-SAA: Heteroscedasticity-aware residuals-based contextual stochastic optimization"
paper:
  title: "Heteroscedasticity-aware residuals-based contextual stochastic optimization"
  authors: "Rohit Kannan, Güzin Bayraksan, James R. Luedtke"
  venue: "arXiv 2021"
  arxiv: "2101.03139"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "simulation-selection"
order: 6
tags: [contextual-stochastic-optimization, heteroscedasticity, residuals, variance-function-estimation, sample-average-approximation, large-deviations, garch, scenario-generation]
date: 2021-01-01
status: draft
summary: "What I took from it: if the noise scale depends on the covariates, standardise the residuals by a learned scale before reusing them, and rescale them at the new covariate. The guarantees go through, but now the scale model's inverse has to be learned well too, and the conditional distribution may differ across covariates only in location and scale."
---

## Why I read it

This short note is where Kannan, Bayraksan and Luedtke's residual framework meets the feature that dominates financial data: noise whose size depends on the state. My question across projects is what happens to a decision's guarantee when the model feeding it is learned. In CASE, a conditional diffusion transformer, the learned object carrying the state-dependent spread is the generator itself. Here it is an explicit scale function $Q(x)$. Seeing exactly what the theory needs from $\hat Q$ tells me what I would have to show about a generator's conditional spread.

## The problem, in one paragraph

The [ER-SAA](/blog/data-driven-saa-covariates/) and [ER-DRO](/blog/residuals-dro-covariates/) frameworks reuse residuals $y^i-\hat f_n(x^i)$ as samples of the noise, which is valid only when the noise is independent of $X$. When demand variability depends on season, or wind variability on location, a residual from a calm covariate is the wrong size for a volatile one. <mark>Reusing raw residuals then puts the same spread on every new covariate</mark>, too narrow in volatile states and too wide in calm ones.

## The idea, as I understand it

- **Location–scale model.** Assume $Y=f^*(X)+Q^*(X)\varepsilon$, with $Q^*(x)\succ0$ the square root of the conditional error covariance and $\varepsilon$ independent of $X$.
- **Standardise, then rescale.** Estimate $\hat f_n$ and $\hat Q_n$, standardise $\hat\varepsilon^i_n=[\hat Q_n(x^i)]^{-1}(y^i-\hat f_n(x^i))$, and solve
$$
\hat v^{ER}_n(x)=\min_{z\in Z}\ \frac1n\sum_{i=1}^n c\Big(z,\ \mathrm{proj}_{\mathcal Y}\big(\hat f_n(x)+\hat Q_n(x)\,\hat\varepsilon^i_n\big)\Big).
$$
The same scenarios can centre an ER-DRO ambiguity set.
- **Reduce everything to one term.** Theorems 1–3 show that consistency, an $O_p(n^{-r/2})$ rate, and an exponential finite-sample bound for the decision each follow once the mean deviation $\frac1n\sum_i\|\tilde\varepsilon^i_n(x)\|$ between ER and full-information scenarios has the same property. They assume $c(z,\cdot)$ is Lipschitz uniformly in $z$, plus the matching SAA conditions.
- **Bound that term.** By repeated Cauchy–Schwarz (inequality (8)), the mean deviation is at most
$$
\|\hat f_n(x)-f^*(x)\|+\|\hat Q_n(x)-Q^*(x)\|\,\tfrac1n\textstyle\sum_i\|\varepsilon^i\|+(\text{error in }\hat Q_n^{-1}\text{ on the data})+(\hat f_n\text{ error on the data, weighted by }\hat Q_n^{-1}).
$$
So the conditions are: $\hat f_n$ and $\hat Q_n$ consistent at $x$; the empirical $L^2$ errors of $\hat f_n$ and of $\hat Q_n^{-1}$ on the training points going to zero; and LLNs for $\|Q^*(X)\|^4$, $\|Q^*(X)^{-1}\|^2$ and $\|\varepsilon\|^4$ (Assumptions 5–7). The finite-sample version (Theorem 7) needs light tails: for i.i.d. data, exponential moments of $\|Q^*(X)^{-1}\|^p$ for some $p>2$, and of $\|Q^*(X)\|^p$ and $\|\varepsilon\|^p$ for some $p>4$.

The part I did not expect is that <mark>the inverse of the scale model matters as much as the model itself</mark>. Residuals are divided by $\hat Q_n(x^i)$, so a scale model that is too small anywhere on the training data inflates the noise samples drawn from that point.

## What the results show

There are no experiments in this note. It is a theory note whose contribution is the bound above, together with a survey of estimators that meet its assumptions. For the mean, those are OLS and feasible weighted least squares (Assumptions 7–8 with $r=1$), the Lasso and heteroscedasticity-adapted variants, adaptive Huber regression (large-deviation bounds even for heavy-tailed errors), and kNN or kernel regression with $r=O(1)/d_x$. For the scale, they are diagonal parametric forms such as $q_j(X)^2=\exp(\sigma_j+\theta_j^\top X)$, fitted by regressing squared residuals, nonparametric local-polynomial or local-likelihood smoothers, and GARCH quasi-likelihood for time series. The authors say the finite-sample condition on $\hat Q_n$ is "typically harder to verify" and leave it to future work.

The empirical test came later, in the Operations Research version of ER-SAA, which absorbed this note. There, heteroscedasticity-aware ER-SAA beat the version that ignores heteroscedasticity in several cases, especially at large $n$ and under severe heteroscedasticity, with both OLS and kNN as the mean model (Figs. 5–6 of arXiv:2207.13554). It came with the caveat that the scale's parametric form was assumed known.

## Where I am not convinced

- **Location–scale is a strong assumption.** Only the centre and spread of the conditional distribution may depend on $X$; the shape of $\varepsilon$ is fixed. A footnote allows invertible maps $Y=m^*(X,\varepsilon)$, but nothing is developed for them. In markets, skew and tail thickness change with the regime, not just volatility.
- **Diagonal $Q$ in every worked example.** The theory allows full matrices, but correlation that changes with the state, which is what breaks diversification in a crisis, is not treated.
- **Time series is cited, not analysed.** GARCH and AR-ARCH estimators are listed as meeting the regression assumptions, but the SAA side is stated for i.i.d. or mixing data without a worked dependent example.
- **No numbers of its own,** and the later experiments assume the true parametric family for $Q^*$.
- **Fourth moments throughout.** The Cauchy–Schwarz route asks for $\mathbb E\|\varepsilon\|^4<\infty$ even for consistency. The authors give an alternative bound with milder moment conditions, at the price of requiring $Q^*$ and $\hat Q_n$ to be uniformly invertible.

## What I take from it

- **Filtered historical simulation is a close cousin.** Standardising returns by a GARCH volatility and rescaling them at today's volatility is heteroscedastic ER-SAA with $Q$ from GARCH, moved to time series, a setting the note cites but does not analyse. In [TailFlow](/research/tailflow/), filtered HS had 28.0% 1-day 99% ES error against the diffusion model's 16.2%, but a 10-day ES bias of only +1.7% against the diffusion model's −13.3% at the 95% level. The structured residual method kept the multi-day spread right where the generative model under-stated it. This note suggests why that can be defensible: if the location–scale model holds, the guarantees depend only on getting $f$, $Q$ and $Q^{-1}$ right.
- **A testable claim about the generator.** A conditional diffusion model learns centre, spread and shape together. If the data really are location–scale, it is spending capacity on shape that does not change. If they are not, residual methods are misspecified in a way this theory cannot see. Which case holds is checkable: standardise CASE's generated scenarios by their own conditional mean and spread, then test whether what is left depends on the conditioning state.
- **The inverse-scale condition has a generative analogue.** Errors in calm states are amplified in the residual method. For a generator, the equivalent risk is a conditional spread that collapses in rarely seen states. Whether that explains my diffusion model's 10-day under-statement is a hypothesis I have not tested.

## What I would try next

*Ideas, not results.*

1. **Hybrid scenario generator.** Use a GARCH or learned $\hat Q(x)$ to standardise returns, train the diffusion model only on the standardised residual's conditional shape, and rescale. Compare ES error and CVaR-constrained regret against plain diffusion and plain filtered HS on TailFlow's synthetic market.
2. **Shape-dependence test.** On real returns, test whether volatility-standardised residuals still depend on the regime (skew, tail index), to see whether the location–scale assumption is adequate for my portfolios before choosing between the two families.

## References

- R. Kannan, G. Bayraksan, J. R. Luedtke. *Heteroscedasticity-aware residuals-based contextual stochastic optimization.* arXiv:2101.03139, 2021 (technical report; its analysis was folded into arXiv:2207.13554).
- R. Kannan, G. Bayraksan, J. R. Luedtke. *Technical Note — Data-Driven Sample Average Approximation with Covariate Information.* Operations Research, 2025.
- J. P. Romano, M. Wolf. *Resurrecting weighted least squares.* Journal of Econometrics 197(1), 2017.
- M. Davidian, R. J. Carroll. *Variance function estimation.* JASA 82(400), 1987.
