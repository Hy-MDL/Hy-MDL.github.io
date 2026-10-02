---
title: "ER-DRO: Residuals-based distributionally robust optimization with covariate information"
paper:
  title: "Residuals-based distributionally robust optimization with covariate information"
  authors: "Rohit Kannan, Güzin Bayraksan, James R. Luedtke"
  venue: "Mathematical Programming 207 (2024) 369–425"
  arxiv: "2012.01088"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "simulation-selection"
order: 4
tags: [distributionally-robust-optimization, wasserstein, contextual-stochastic-optimization, covariates, residuals, cvar, portfolio-optimization, cross-validation]
date: 2020-12-01
status: draft
summary: "What I took from it: put the ambiguity ball around the residual scenarios and make its radius the sum of two parts, one for the regression error and one for ordinary sampling error. The certificate follows in a few lines. In practice the radius comes from cross-validation, and the useful finding is that the cross-validation has to use the covariates, or the ball shrinks too fast."
---

## Why I read it

This is the robust half of Kannan, Bayraksan and Luedtke's covariate work. My own question is what happens to a decision's guarantee when the model feeding it is learned. The [ER-SAA paper](/blog/data-driven-saa-covariates/) answers it asymptotically. This one asks for a finite-sample certificate: a number $\hat v$ such that the true cost of the chosen decision is below $\hat v$ with probability $1-\alpha$. That is the shape of guarantee a VaR/ES reserve should have. Its experiment is also a mean–CVaR portfolio, which is my domain.

## The problem, in one paragraph

Same setting as ER-SAA: $Y=f^*(X)+\varepsilon$ with $\varepsilon$ independent of $X$ (homoscedastic here), $n$ joint observations, a new covariate $x$, and the goal $\min_z\mathbb E[c(z,Y)\mid X=x]$. With small $n$, ER-SAA is optimistic: its optimal value under-estimates the true cost of its own decision. The classical fix is DRO around the empirical distribution. <mark>Here no one has samples from the conditional distribution at $x$, only scenarios built from a fitted model</mark>, so the usual ways of setting the radius do not apply directly.

## The idea, as I understand it

- **Centre the ball on the residual scenarios.** Let $\hat P^{ER}_n(x)=\frac1n\sum_i\delta_{\mathrm{proj}_{\mathcal Y}(\hat f_n(x)+\hat\varepsilon^i_n)}$ and solve
$$
\hat v^{DRO}_n(x)=\min_{z\in Z}\ \sup_{Q\in\hat{\mathcal P}_n(x)}\ \mathbb E_{Y\sim Q}[c(z,Y)].
$$
The ambiguity set can be a $p$-Wasserstein ball, a sample-robust set (each scenario moves within a radius), or a reweighting of the same scenarios (CVaR or phi-divergence).
- **Two-part radius.** By the triangle inequality, $d_{W,p}(\hat P^{ER}_n(x),P_{Y\mid X=x})$ is at most the power-mean scenario error plus the distance between the full-information empirical distribution and the truth. So the radius is
$$
\zeta_n(\alpha,x)=\underbrace{\kappa^{(1)}_{p,n}(\alpha,x)}_{\text{regression error}}+\underbrace{\kappa^{(2)}_{p,n}(\alpha)}_{\text{sampling error}} .
$$
- **Certificate (Theorem 7).** If the errors are i.i.d. with $\mathbb E[\exp(\|\varepsilon\|^a)]<\infty$ for some $a>p$, and the regression error at $x$ and on the training points has a finite-sample tail bound (Assumption 2), then
$$
P\big(g(\hat z^{DRO}_n(x);x)\le\hat v^{DRO}_n(x)\big)\ge 1-\alpha .
$$
Consistency, an $O_p(\zeta_n)$ rate, an $L^q$ rate averaged over $X$, and an exponential solution bound follow under further Lipschitz or smoothness conditions. For sample-robust and reweighting sets, the rate matches ER-SAA's $O_p(n^{-r/2})$ if the radius shrinks fast enough (Theorem 17).

The authors are open about the catch (Remark 3). The sampling part shrinks like $n^{-\min\{p/d_y,1/2\}}$ up to logs, so <mark>the theoretical radius suffers the curse of dimensionality even when the regression is parametric</mark>. In the experiment $d_y=10$, which is why the radius actually used comes from cross-validation. The ball also covers only the residuals, not the uncertainty in the regression coefficients.

## What the results show

The testbed is a mean–CVaR portfolio: 10 assets, $\beta=0.8$, $\rho=10$. Returns are linear in powers $\theta\in\{0.5,1,2\}$ of 3 relevant covariates out of $d_x\in\{3,10,100\}$, with Gaussian idiosyncratic noise plus a common Gaussian shock. Prediction is linear (OLS, Lasso or Ridge). Each setting has 50 data replications × 20 covariate draws = 1,000 99%-UCBs on the optimality gap, using 20,000 conditional samples. The radius is picked from 28 candidates.

| comparison | what the figures show | source |
|---|---|---|
| Wasserstein (W) vs sample-robust (S) vs Hellinger (H), OLS | W and S similar; H, which cannot leave the scenario support, adds little at small $d_x$ | Fig. 1 |
| W+OLS vs ER-SAA+OLS, small $n$ | W better in almost all cases; the exception is Algorithm 2 at $\theta=2$, large $n$ | Figs. 1–2 |
| radius by CV ignoring covariates (Alg. 1) | radius goes to zero quickly; worst of the three | Figs. 2, 7 |
| covariate-independent CV (Alg. 2) vs covariate-dependent CV (Alg. 3) | Alg. 2 better at small $n$ and $d_x$, Alg. 3 at larger | Figs. 2, 6 |
| optimal values against true cost, $d_x=100$ | ER-SAA optimistic; W pessimistic, and at $\theta=2$ the pessimism does not vanish because the CV radius stays positive | Fig. 4 |

How I read it:

- **The robust layer mostly repairs small-sample optimism and mild misspecification.** At $\theta=0.5$, W+OLS largely removes the misspecification penalty. At $\theta=2$, only the covariate-dependent radius does.
- **Use the covariates in the cross-validation.** Algorithm 1 borrows the standard no-covariate tuning and is beaten by both covariate-aware versions. This is the paper's most practical result.
- **Benefits fade with $n$,** as expected. The authors frame ER-DRO and ER-SAA as complements, not competitors.

## Where I am not convinced

- **The certificate is never measured.** Theorem 7 holds for the theoretical radius; the experiments use a cross-validated one. Figure 4 shows pessimism on average, not a coverage frequency against $1-\alpha$.
- **Box plots only.** No table gives a number I can compare against.
- **Light tails are assumed exactly where finance is heavy-tailed.** The certificate needs an exponential moment of order above $p$, and the simulated returns are Gaussian. A mean–CVaR portfolio is the case where tails matter most.
- **Homoscedastic errors.** The heteroscedastic version is deferred to a [separate note](/blog/heteroscedastic-residuals-saa/) with no experiments of its own.
- **Tractability is narrow.** Wasserstein and sample-robust ER-DRO on two-stage problems is NP-hard in general. The portfolio is single-stage, so it avoids the issue.
- A small slip: Section 7 says "Algorithm 3 provides a better-performing alternative to Algorithm 3", which should read Algorithm 2.

## What I take from it

- **Separate the model's error from the sampling error.** I can carry the two-part radius over to a generative scenario model: one term for how far the generator's conditional is from the truth, one for finite scenario count. In CASE, the VaR/ES reserve is computed on generated scenarios. A Wasserstein ball around them, with the generator term estimated from held-out data, would turn the reserve into a certificate instead of a point estimate.
- **The coefficient gap matches what I found.** ER-DRO deliberately leaves out uncertainty in the fitted model. In [TailFlow](/research/tailflow/), what lowered the chance of an infeasible portfolio from 0.405 to 0.155 was an ensemble over input models, which is exactly the coefficient-level uncertainty this paper leaves out. The two kinds of robustness are complementary, and I have only tested one.
- **Tune the radius at the decision, not globally.** The covariate-dependent tuning only pays at larger $n$, a cost a generative model may be able to reduce, since it can produce extra conditional scenarios for the validation step.

## What I would try next

*Ideas, not results.*

1. **Measure the certificate.** On TailFlow's synthetic market, where true conditional ES is known, run Wasserstein DRO around (a) linear-factor residual scenarios and (b) diffusion scenarios. Record how often the true CVaR cost lies below the DRO value at nominal $1-\alpha$, with both the cross-validated and a held-out-calibrated radius.
2. **Add the coefficient layer.** Combine the residual ball with an ensemble over fitted models (worst case over members, ball within each), and check whether coverage improves on what each layer gives alone.

## References

- R. Kannan, G. Bayraksan, J. R. Luedtke. *Residuals-based distributionally robust optimization with covariate information.* Mathematical Programming 207, 369–425, 2024. arXiv:2012.01088.
- P. Mohajerin Esfahani, D. Kuhn. *Data-driven distributionally robust optimization using the Wasserstein metric.* Mathematical Programming 171, 2018.
- D. Bertsimas, S. Shtern, B. Sturt. *Two-stage sample robust optimization.* Operations Research 70(1), 2022.
- N. Fournier, A. Guillin. *On the rate of convergence in Wasserstein distance of the empirical measure.* Probability Theory and Related Fields 162, 2015.
