---
title: "Submodel uncertainty: Quantifying and Attributing Submodel Uncertainty in Stochastic Simulation Models and Digital Twins"
paper:
  title: "Quantifying and Attributing Submodel Uncertainty in Stochastic Simulation Models and Digital Twins"
  authors: "Mohammadmahdi Ghasemloo, David J. Eckman, Yaxian Li"
  venue: "arXiv 2026"
  arxiv: "2602.16099"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "simulation-selection"
order: 3
tags: [input-uncertainty, submodel-uncertainty, digital-twins, bootstrap, bayesian-model-averaging, variance-decomposition, regression-trees, simulation]
date: 2026-02-01
status: draft
summary: "What I took from it: input uncertainty is only one case of a wider problem, since simulations now contain learned routing rules, surrogates and generative pieces, each with its own estimation error. Resampling every learned piece gives honest intervals and a tree-based split of the blame, but coverage still falls short and the attribution has no guarantee."
---

## Why I read it

My research question is how much of a decision's guarantee survives when part of the model is learned. The input-uncertainty literature answers it for input distributions. This paper widens it to every learned piece inside a simulation, which is the situation my [TailFlow](/research/tailflow/) and [learned-prior](/research/model-uncertainty-priors/) work is in. It is the closest 2026 paper to what I want to do.

## The problem, in one paragraph

Simulations increasingly replace real subprocesses with fitted ones: input distributions, but also machine-learned routing or decision logic, optimisation proxies, learned policies and generative models. The authors call each of these a submodel and its estimation error submodel uncertainty. Input-uncertainty methods cover only the input distributions. The usual confidence interval ignores this epistemic error entirely and can under-cover. And existing tools do not say which submodel the uncertainty comes from, so a modeller cannot tell where more data would help most.

## The idea, as I understand it

- **Split the variance.** By the law of total variance, the variance of the simulation estimate is a submodel (epistemic) part plus a simulation-noise part:
$$
\mathrm{Var}\big(\bar Y(\hat S)\big)=\underbrace{\mathrm{Var}\big(\mathbb E[\bar Y(\hat S)\mid\hat S]\big)}_{\text{submodel}}+\underbrace{\mathbb E\big[\mathrm{Var}(\bar Y(\hat S)\mid\hat S)\big]}_{\text{simulation noise}} .
$$
- **Resample every learned piece.** Generate plausible versions of each submodel by bootstrapping its training data or by sampling its posterior. Combine them in a stacked Latin hypercube design, run each configuration, and take quantiles of the configuration means as the interval.
- **Blame by tree.** Fit a regression tree with each submodel as a categorical feature, and score each submodel by how much variance its splits explain. The residual is the simulation-noise share.
- **Digital twins.** Repeat this for each observed state and average, and estimate the twin's bias against observed data.

<mark>The useful shift for me is treating a learned routing rule or a generative model as the same kind of object as an input distribution: something estimated, with uncertainty that has to travel to the output.</mark>

> **My comment.** The catch for generative submodels is how the plausible versions are drawn. In TailFlow I used an ensemble of 11 re-trained models as a posterior over input models, and its spread turned out too narrow to read as a calibrated uncertainty: it made the decision safer without being an honest sample of the epistemic distribution this decomposition needs.

## What the results show

| synthetic model, 90% intervals | coverage | width | source |
|---|---|---|---|
| no epistemic uncertainty | 61.0% | 20.45 | Table 1 |
| input uncertainty only | 82.0% | 36.14 | Table 1 |
| all submodel uncertainty | 87.0% | 38.60 | Table 1 |

- **Accounting for learned pieces matters.** Ignoring them gives 61% coverage on a 90% interval. Adding input uncertainty brings it to 82%, and all submodels to 87%.
- **Even the full version under-covers.** 87% is below 90%. The authors blame bias, which is not corrected; with 100 macro-replications, 87 against 90 is also within noise.
- **The attribution is plausible but weakly separating.** On the toy model the ranking agrees with an ANOVA, but the scores are close: 0.25 for the most important piece and 0.20 for one the ANOVA finds insignificant.
- **In a contact-centre digital twin,** the frequentist and Bayesian versions disagree on which routing submodel matters most: 0.096 against 0.210 for the expert-side routing.

> **My comment.** I read the 87% as the optimistic end. Bootstrapping a submodel's training data resamples around the fitted model, not around the truth, and in my input-uncertainty study the bootstrap standard deviation under-stated the actual true-minus-fitted CVaR error by 2.5–16× whenever the input model was misspecified.

## Where I am not convinced

- **No theorems.** Neither the interval's validity nor the importance scores' consistency is proved.
- **Missing comparisons.** There is no Sobol or Shapley-based attribution and no comparison with existing input-uncertainty methods beyond "input uncertainty only".
- **The twin experiment is tiny.** One simulated day, five versions per submodel, no macro-replications, and the "real system" is itself a simulation.
- **Cost is discussed, not measured.** Retraining every submodel by bootstrap is the dominant cost and is never timed.

## What I take from it

- **The question I care about is now named.** "Submodel uncertainty" is a better frame for my TailFlow study than input uncertainty: the diffusion model there is a learned submodel feeding a CVaR-constrained selection, and its error is what flipped decisions.
- **Generative submodels make resampling cheap or expensive.** Bootstrapping a diffusion model means retraining it many times. Posterior draws, ensembles or the density of a flow could replace that, which is where my learned-prior work, where the flow's density was the key, connects.
- **Attribution should be about decisions, not variance.** For selection, what matters is which submodel can flip the chosen system, not which one adds the most variance to an output.

## What I would try next

*Ideas, not results.*

1. **Attribute decision risk, not variance.** In TailFlow's synthetic market, treat the generator and the volatility-state filter as submodels and attribute the probability of an infeasible choice, not the output variance, to each.
2. **Cheap generative versions.** Replace bootstrap retraining of a flow input model with draws from an ensemble or a posterior over its weights, and check whether interval coverage holds at a fraction of the cost.

## References

- M. Ghasemloo, D. J. Eckman, Y. Li. *Quantifying and Attributing Submodel Uncertainty in Stochastic Simulation Models and Digital Twins.* arXiv:2602.16099, 2026.
- R. R. Barton. *Tutorial: Input uncertainty in output analysis.* Proceedings of the Winter Simulation Conference, 2012.
- H. Lam. *Advanced tutorial: Input uncertainty and robust analysis in stochastic simulation.* Proceedings of the Winter Simulation Conference, 2016.
- B. L. Nelson. *Foundations and Methods of Stochastic Simulation.* Springer, 2013.
