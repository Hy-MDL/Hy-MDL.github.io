---
title: "BRF: Feasibility Determination for Subjective Probability Constraints"
paper:
  title: "Feasibility Determination for Subjective Probability Constraints"
  authors: "Taehoon Kim, Sigrún Andradóttir, Seong-Hee Kim, Yuwei Zhou"
  venue: "arXiv 2026"
  arxiv: "2605.27237"
  license: "creativecommons.org/licenses/by/4.0/"
series: "simulation-selection"
order: 1
tags: [feasibility-determination, ranking-and-selection, probability-constraints, bernoulli, indifference-zone, random-walk, common-random-numbers, simulation]
date: 2026-05-01
status: draft
summary: "What I took from it: when a constraint is a probability, every observation is a coin flip, and comparing it with a dummy coin flip at the threshold turns feasibility into a gambler's-ruin walk that needs no variance estimate and no batching. One sample path then serves every threshold the decision maker wants to try."
---

## Why I read it

Ranking & selection and feasibility determination are the part of simulation optimisation I want to work in, and Seong-Hee Kim's group builds the procedures with statistical guarantees that I keep using as a reference. I also have a concrete use. In my [battery simulator](/research/battery-degradation-sim/), the constraints I care about are probabilities: the chance that peak temperature exceeds a limit, or that capacity falls below 80% before a given cycle.

## The problem, in one paragraph

There are $k$ simulated systems, and each has constraints of the form $p=\Pr(\text{event})\le h$, so every observation is Bernoulli. The constraints are "subjective": the decision maker may try several thresholds $h$ per constraint, all at once or added later. Most valid feasibility procedures assume normal data, so on Bernoulli data they average observations into batch means first. <mark>Batches large enough to look normal waste observations, especially when $p$ is small</mark>, and that is exactly the regime of safety constraints.

> **My comment.** One practical snag for my own use: my battery simulator is deterministic for a given protocol and parameter set, so there is no Bernoulli observation to draw until I decide where the randomness comes from (cell-to-cell parameter spread, ambient noise, or posterior draws of uncertain parameters). That choice would define what "probability of exceeding 60 °C" even means.

## The idea, as I understand it

- **Compare with a dummy coin.** For each threshold $h$, draw a dummy Bernoulli($h$) variable alongside every real observation and track the running sum of the differences. It is a simple random walk. Declare the system feasible when the walk falls to $-H$, and infeasible when it rises to $+H$.
- **Gambler's ruin gives the error rate.** The probability of absorbing at the wrong boundary is known in closed form. With an indifference zone set as an odds ratio $\theta>1$ rather than a difference, the walk needs
$$
\frac{1}{1+\theta^{H}}\le\beta
$$
for error $\beta$, so $H$ is chosen up front. No variance estimate, no initial sample and no batching are needed.
- **One path, many thresholds.** All thresholds on a constraint share one uniform random number per step, so the dummy coins are monotone in $h$. A single stream of observations then decides every threshold, and only the two hardest thresholds per constraint cost any error budget.

I find the odds-ratio indifference zone the quietly important choice. <mark>It narrows near 0 and 1, which is where probability constraints for rare events live</mark>, and a difference-based zone would treat 0.01 versus 0.02 the same as 0.50 versus 0.51.

## What the results show

| comparison | result | source |
|---|---|---|
| one system, $p=0.15$, $\theta=1.2$: expected stopping time | theory 1,133.3 against simulation 1,141.1 | Table 1 |
| against the batch-means procedure, eight values of $p$, $\theta=1.2$ | new procedure meets the target in 8 of 8 cases with about 1,850 observations; batch means with batch size 1 meets it in 3 of 8, and needs batch size 100 and about 2,786 observations to reach 8 of 8 | Table 2 |
| cost of batch means at matched accuracy | roughly 1.5 to 9.3 times the observations | Section 5.2 |
| (s,S) inventory, 77 systems | multi-pass heuristic 116,402 observations; single-pass procedure 352,507; batch means 4,707,590 | Table 5 |

How I read it:

- **The validity result is clean.** The batch-means procedure under-covers when $p$ is small and batches are short; the new procedure meets its target throughout.
- **The large savings come from the heuristics.** Passes that add thresholds later have no proof, and their savings depend on the order in which thresholds are tried: about 20% in one configuration and none in another. One 100-system example is, in the paper's own words, an extreme case built to favour them.
- **Common random numbers neither helped nor hurt** in the inventory example.

> **My comment.** Before using this on battery constraints I would add the walk to rs-lab and run the same check I ran for every procedure there, the nominal level verified over thousands of macro-replications, including the small-p settings where batch means fails.

## Where I am not convinced

- **The heuristics are thinly validated.** One small configuration plus the examples, with no adversarial test near very small $p$.
- **The least-favourable configuration is asserted,** described as "expected to be" the slippage configuration.
- **The stopping time is unbounded.** Only its expectation is known.
- **Only one baseline.** There is no Bayesian or OCBA-style feasibility method and no sequential probability ratio test.
- **Conservatism is not measured.** Bonferroni splitting across constraints and systems gives coverage of 0.997 to 1.000 in several tables, which suggests budget is being left on the table.

## What I take from it

- **Use the structure of the data instead of forcing normality.** The whole gain comes from treating Bernoulli data as Bernoulli. My queue study had a similar lesson in a different form: an arrival model that ignored burstiness looked fine on average and failed on coverage.
- **A certified final screen for my battery work.** A surrogate can suggest charging protocols, but a statement like "peak temperature above 60 °C with probability at most 1%" should come from a procedure like this. In my simulator, a Gaussian-process surrogate predicted a 45 °C protocol would stay under the limit and the simulation reached 60.7 °C. That is the failure a certified check exists for.
- **Learned models could order the thresholds.** The heuristics' savings depend on threshold order, which is something a learned model could predict.

## What I would try next

*Ideas, not results.*

1. **Battery safety feasibility.** Treat charging protocols as systems and $\Pr(T>T_{\max})$ and $\Pr(\text{capacity}<80\%\text{ by cycle }N)$ as constraints, with a surrogate choosing the threshold order. Compare observations used against uniform ordering.
2. **Input uncertainty.** Drive the simulator with a learned input model, mix the walk over posterior draws of that model, and check whether "only two thresholds per constraint cost error budget" survives the extra uncertainty.

## References

- T. Kim, S. Andradóttir, S.-H. Kim, Y. Zhou. *Feasibility Determination for Subjective Probability Constraints.* arXiv:2605.27237, 2026. A single-constraint version appeared at WSC 2024.
- Y. Zhou, S. Andradóttir, S.-H. Kim. *Finding feasible systems for subjective constraints using recycled observations.* INFORMS Journal on Computing, 2022.
- H. M. Taylor, S. Karlin. *An Introduction to Stochastic Modeling.* Academic Press, 1994.
