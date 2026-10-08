---
title: "Root-finding calibration: Root Finding and Metamodeling for Rapid and Robust Computer Model Calibration"
paper:
  title: "Root Finding and Metamodeling for Rapid and Robust Computer Model Calibration"
  authors: "Yongseok Jeon, Sara Shashaani"
  venue: "arXiv 2026"
  arxiv: "2603.23790"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "simulation-selection"
order: 2
tags: [model-calibration, root-finding, stochastic-kriging, bayesian-optimization, acquisition-functions, sample-average-approximation, digital-twins]
date: 2026-03-01
status: draft
summary: "What I took from it: squaring the calibration error throws away its sign, and the sign says which side of the answer you are on. Averaging the residuals into one signed number and searching for its root lets two evaluations of opposite sign bracket the answer, at the price of a bias that is small only when the residuals agree across outputs."
---

## Why I read it

Calibration is the step between a simulator and a decision: before you optimise anything with a model, its parameters have to match the real system. I am about to face this with my [battery simulator](/research/battery-degradation-sim/), whose degradation parameters come from a published cell, not mine. This paper, from Sara Shashaani's group, attacks calibration cost with one simple observation.

## The problem, in one paragraph

Calibration chooses the parameters $\theta$ of a black-box model $h(X;\theta)$ to minimise the mean squared discrepancy between model and real outputs. It is expensive when the data are large or the model is slow. The real system is noisy, so a calibration can over-fit, and many outputs make it hard to see how each parameter matters. Standard surrogate-based approaches, including Bayesian optimisation, minimise the squared loss, and <mark>squaring throws away the sign of the error, which is the information that says which way to move</mark>.

## The idea, as I understand it

- **Keep the sign.** Average the residuals across the output dimensions into one signed scalar $S(\theta)$ and look for a root of its expectation:
$$
\tilde f(\theta)=\mathbb E\big[S(\theta)\big]=0,\qquad S(\theta)=\frac{1}{m_y}\sum_{i=1}^{m_y}\big[Y-h(X;\theta)\big]_i .
$$
Squaring this averaged residual gives a lower bound on the original objective.
- **Bracket the root.** Any two evaluated points with opposite signs define a box that must contain a root. The search keeps shrinking to the smallest such box.
- **Search with kriging.** A Gaussian-process model of the signed residual is searched with three new root-finding versions of the usual acquisition functions: lower confidence bound, probability of improvement and expected improvement. With noisy simulators, stochastic kriging turns "opposite signs" into a probability.

> **My comment.** My battery GP surrogate failed at exactly such a boundary. At 45 °C it said a charge rate would keep the cell under the 60 °C limit, and the simulator reached 60.7 °C; a probability of being on the wrong side of the constraint, rather than a point prediction, is what that decision needed.

The theory states the price clearly. The root-finding answer is close to the true calibration only up to a bias that cannot be removed, roughly the variance of $S$ and the disagreement between outputs near the root. <mark>The method is fast exactly when the outputs err in the same direction, and can find spurious roots when positive and negative residuals cancel</mark>.

## What the results show

The examples are a 2-D test function, an M/M/1 queue (one unknown rate, 100 sojourn times as outputs) and a stochastic SIR epidemic, with 100 macro-replications and at most 10 sequential evaluations. There are no result tables. Reading the curves, root finding with stochastic kriging and box shrinking reaches a lower post-evaluated calibration error than minimisation for all three acquisitions in all three examples. The sampled points concentrate along the zero contour instead of scattering.

What this supports, in my reading: with very small budgets, in one or two dimensions, on problems with a clean single sign change, the sign is worth a lot. It does not yet say how the method scales.

> **My comment.** The M/M/1 example is the friendliest case possible: one rate, and every sojourn time moves the same way when it changes. The queue in my exchange-queueing reanalysis has bursty batch arrivals that a Poisson model cannot carry, and there I would expect a fitted rate to over-predict short sojourns and under-predict long ones, which is exactly the cancellation across outputs that the bias term warns about.

## Where I am not convinced

- **No real data.** The abstract mentions data-driven and physics-based examples, but all three are simulated.
- **No numbers or timings.** The "significant computational gains" rest on curves with wide, overlapping bands.
- **Low dimension only.** At most two parameters. The box search compares pairs of points and will likely scale poorly.
- **The failure case is untested.** No experiment has residuals that change sign across outputs, which is where the bias bites.
- **Missing baselines.** There is no comparison with derivative-free root finders such as the group's own ASTRO-DF line, with Kennedy–O'Hagan calibration, or with stochastic bisection.

## What I take from it

- **Look for the sign in my own problems.** Battery calibration against voltage curves is a good case: a monotone change in a diffusion coefficient usually shifts the whole curve one way, so the residuals agree across time points and the bias should be small.
- **The generative-model version is suggested but not done.** The paper notes that the draws used by stochastic kriging could come from a fitted generative model. That is my research direction: calibrate against draws from a learned input model, and add that model's uncertainty to the noise the search sees.
- **Calibration is a selection problem at the end.** Deciding whether a pair of points really brackets a root is a feasibility question, which links this paper to the [feasibility procedure](/blog/feasibility-subjective-probability/) I read next to it.

## What I would try next

*Ideas, not results.*

1. **Battery parameter calibration.** Calibrate solid-phase diffusivity and a reaction-rate constant of a PyBaMM model against discharge curves, and compare root finding with squared-loss BO at equal simulation budget.
2. **A designed failure.** Construct outputs whose residuals change sign across dimensions, and measure how often the method converges to a spurious root, as a check on the bias term.

## References

- Y. Jeon, S. Shashaani. *Root Finding and Metamodeling for Rapid and Robust Computer Model Calibration.* arXiv:2603.23790, 2026.
- B. Ankenman, B. L. Nelson, J. Staum. *Stochastic kriging for simulation metamodeling.* Operations Research 58(2), 2010.
- M. C. Kennedy, A. O'Hagan. *Bayesian calibration of computer models.* JRSS-B 63(3), 2001.
