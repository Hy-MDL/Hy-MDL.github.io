---
title: "ER-SAA: Data-Driven Sample Average Approximation with Covariate Information"
paper:
  title: "Technical Note — Data-Driven Sample Average Approximation with Covariate Information"
  authors: "Rohit Kannan, Güzin Bayraksan, James R. Luedtke"
  venue: "Operations Research (2025)"
  arxiv: "2207.13554"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "simulation-selection"
order: 5
tags: [contextual-stochastic-optimization, sample-average-approximation, covariates, residuals, jackknife, large-deviations, regression, scenario-generation]
date: 2022-07-01
status: draft
summary: "What I took from it: the simplest conditional scenario generator is a point prediction plus the model's own residuals, and for a cost that is Lipschitz in the uncertainty its decision error is bounded by two regression errors, one at the new covariate and one averaged over the training points. That bound is the cleanest answer I have seen to how much of a decision's guarantee survives a learned model."
---

## Why I read it

This is the paper Kannan, Bayraksan and Luedtke's covariate line starts from. The question I keep returning to in my own projects is what happens to a decision's guarantee when the model feeding it is learned. In [Input-Uncertainty Select](/research/input-uncertainty-select/) the learned piece was an input distribution; in [TailFlow](/research/tailflow/) and CASE it is a conditional generative model. This paper asks the same question for the plainest possible learned model, a regression, and answers it with theorems.

## The problem, in one paragraph

We observe $n$ pairs $(x^i,y^i)$ of covariates and uncertain parameters, then see a new covariate $x$ and must solve $\min_{z\in Z}\mathbb E[c(z,Y)\mid X=x]$. The authors assume $Y=f^*(X)+Q^*(X)\varepsilon$ with $\varepsilon$ independent of $X$, so the conditional distribution is the error distribution shifted by $f^*(x)$ and scaled by $Q^*(x)$. Neither is known. Reweighting methods (kNN or forest weights on past $y^i$, as in Bertsimas and Kallus) avoid a model but converge slowly in the covariate dimension; plugging in a point forecast ignores uncertainty altogether. <mark>The question is what you lose by learning $f^*$ and reusing its residuals as the noise.</mark>

## The idea, as I understand it

- **Prediction plus residuals.** Fit $\hat f_n$ (and, if heteroscedastic, $\hat Q_n$), form residuals $\hat\varepsilon^i_n=[\hat Q_n(x^i)]^{-1}(y^i-\hat f_n(x^i))$, and solve the empirical residuals-based SAA
$$
\hat v^{ER}_n(x)=\min_{z\in Z}\ \frac1n\sum_{i=1}^n c\Big(z,\ \mathrm{proj}_{\mathcal Y}\big(\hat f_n(x)+\hat Q_n(x)\,\hat\varepsilon^i_n\big)\Big).
$$
It costs the same as naive SAA plus one regression.
- **The bound that carries everything.** Compare each scenario with the full-information one, $f^*(x)+\varepsilon^i$. In the homoscedastic case the average gap is at most
$$
\frac1n\sum_{i=1}^n\|\tilde\varepsilon^i_n(x)\|\ \le\ \|\hat f_n(x)-f^*(x)\|+\frac1n\sum_{i=1}^n\|\hat f_n(x^i)-f^*(x^i)\|,
$$
and if $c(z,\cdot)$ is Lipschitz uniformly in $z$ (Assumption 1), the objective gap is that times the Lipschitz constant. Every theorem is this lemma plus a standard SAA result.
- **Leave-one-out residuals.** In-sample residuals are too small when the model over-fits. J-SAA uses jackknife residuals around $\hat f_n(x)$; J+-SAA uses each leave-one-out model's own prediction, as in jackknife+ prediction intervals.

The guarantees come in three grades. Consistency (Theorem 5) needs Lipschitz cost, a uniform LLN for the full-information SAA, fourth-moment LLNs, and consistent regression at $x$ and in empirical $L^2$ on the training points. The rate is $O_p(n^{-\alpha/2})$, with $\alpha=1$ for OLS or Lasso and $\alpha=O(1)/d_x$ for kNN or forests (appendix Theorem 13). Finite-sample: exponential bounds on the regression error give $P(\mathrm{dist}(\hat z^{ER}_n(x),S^*(x))\ge\eta)\le Q(\eta,x)e^{-\gamma(n,\eta,x)}$ (Theorem 8). For a two-stage LP with sub-Gaussian errors and OLS, Proposition 9 makes it concrete:
$$
n\ \ge\ n^*+\frac{O(1)\,\sigma^2 d_y}{\kappa^2}\Big(\log\frac{O(1)}{\delta}+d_x\Big),
$$
where $n^*$ is the usual SAA sample size; for the Lasso, $d_x$ becomes $s\log d_x$ for sparsity $s$. <mark>The price of learning appears as an additive sample-size term, linear in $d_x$ for OLS and logarithmic for the Lasso.</mark>

## What the results show

The testbed is a two-stage resource allocation LP with 20 resources and 30 customer types. Demand depends on 3 of $d_x$ covariates through powers $p\in\{0.5,1,2\}$, so a linear model is misspecified unless $p=1$, with Gaussian errors ($\sigma=5$ by default) and optional heteroscedasticity. Each setting has 100 replications, reported as box plots of a 99% upper confidence bound on the optimality gap.

| comparison | what the paper reports | source |
|---|---|---|
| naive SAA, no covariates, $n=10{,}100$ | median UCB about 11%, 5% and 26% for $p=1, 0.5, 2$ | Section 4 text |
| ER-SAA+OLS vs kNN reweighting, $p=0.5$ | OLS still better at the largest $n=100(d_x+1)$ | Fig. 2 |
| same, $p=2$ | kNN better only for $n\ge 80$ at $d_x=3$, and not at larger $d_x$ | Fig. 2 |
| ER-SAA+OLS vs point prediction | ER-SAA better in every case | Fig. 2 |
| J-SAA vs ER-SAA, $n\in\{1.3,1.5,2\}(d_x+1)$ | smaller 75th and 95th percentiles, mostly at small $n$ | Fig. 3 |
| raising $\sigma$ from 5 to 20 | ER-SAA+OLS needs more data; kNN unaffected or slightly better | Fig. 4 |

How I read it:

- **Structure beats flexibility at these sample sizes,** even when the structure is wrong. A misspecified linear model with residuals beats a consistent nonparametric method across most of the grid.
- **Residuals are worth more than the point forecast.** Point prediction loses everywhere, which is the expected cost of ignoring spread in a recourse problem.
- **The jackknife helps the tails of the distribution, not the median.** It matters where OLS over-fits, which is exactly where in-sample residuals are too optimistic.

## Where I am not convinced

- **One synthetic family.** Every experiment uses the same LP and the same data generator, and misspecification is only the exponent $p$.
- **Figures, not tables.** Apart from the naive-SAA medians, the comparisons are read off box plots.
- **Light tails are doing real work.** Assumption 8 can fail for heavy-tailed errors, as the authors say, and the errors are Gaussian throughout. Return data, where I would use this, are not.
- **The misspecified limit is characterised, not bounded.** Remark 1 says the solution converges to the solution of a different problem, with no bound on how far off that is.
- **Guarantees are pointwise in $x$.** Theorem 6 averages over $X$, but nothing is uniform over the covariate a decision maker actually faces.
- **Baselines are narrow.** Only kNN reweighting and point prediction; no forest weights, no decision-focused training.

## What I take from it

- **A template for my own question.** Decision error is at most Lipschitz constant times the scenario error, and scenario error splits into error at $x$ and error on the data. For a generative scenario model the matching quantity is a Wasserstein distance between generated and true conditionals: $|\mathbb E_P c-\mathbb E_{Q}c|\le L\,W_1(P,Q)$. CASE scores its scenarios by the VaR/ES reserve they imply, and ES is Lipschitz in the loss, so the same argument applies.
- **A baseline I have been missing.** "Factor regression plus its residuals" is the obvious competitor to a conditional diffusion model as a scenario generator. If it wins at my sample sizes, as the misspecified linear model does here, the generative model has to earn its place.
- **In-sample residuals flatter the model.** That is the same direction as my bootstrap under-stating CVaR error by 1.2x to 16x in Input-Uncertainty Select. The jackknife is a cheap correction worth trying there.

## What I would try next

*Ideas, not results.*

1. **Residuals vs generator.** On TailFlow's synthetic market, where true VaR/ES are known, compare ER-SAA and J+-SAA scenarios from a linear factor model with conditional diffusion scenarios. Measure reserve error and the CVaR-constrained decision's regret at matched $n$.
2. **Check how tight the bound is.** For each generator, estimate $W_1$ between generated and true conditional returns and plot it against realised regret, to see whether the Lipschitz bound is tight enough to guide a choice between generators.

## References

- R. Kannan, G. Bayraksan, J. R. Luedtke. *Technical Note — Data-Driven Sample Average Approximation with Covariate Information.* Operations Research, 2025. arXiv:2207.13554 (first circulated on Optimization Online, July 2020; this version absorbs arXiv:2101.03139).
- D. Bertsimas, N. Kallus. *From predictive to prescriptive analytics.* Management Science 66(3), 2020.
- G.-Y. Ban, J. Gallien, A. J. Mersereau. *Dynamic procurement of new products with covariate information: the residual tree method.* M&SOM 21(4), 2019.
- R. F. Barber, E. J. Candès, A. Ramdas, R. J. Tibshirani. *Predictive inference with the jackknife+.* Annals of Statistics 49(1), 2021.
