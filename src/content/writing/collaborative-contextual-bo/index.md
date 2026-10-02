---
title: "CCBO: Collaborative Contextual Bayesian Optimization"
paper:
  title: "Collaborative Contextual Bayesian Optimization"
  authors: "Chih-Yu Chang, Qiyuan Chen, Tianhan Gao, David Fenning, Chinedum Okwudire, Neil Dasgupta, Wei Lu, Raed Al Kontar"
  venue: "arXiv 2026"
  arxiv: "2604.18912"
  license: "creativecommons.org/licenses/by/4.0/"
series: "surrogates-bo"
order: 8
tags: [bayesian-optimization, contextual-bo, collaborative-bo, federated-learning, thompson-sampling, random-fourier-features, regret, manufacturing]
date: 2026-04-01
status: draft
summary: "What I took from it: several clients learn a whole map from context to best design, borrowing a pooled model early through a decaying coin flip and running their own Thompson sampling later. The guarantee is the familiar no-harm kind, the experiments use a schedule the theorem does not cover, and the real open question is what to borrow when clients genuinely differ."
---

## Why I read it

This is the newest paper in the Al Kontar group's collaborative Bayesian optimisation (BO) line, after [consensus BO](/blog/consensus-bayesian-optimization/) and [LLINBO](/blog/llinbo/). I wanted to see what changes when each client needs not one optimum but a whole policy, the best design for every operating condition.

## The problem, in one paragraph

In contextual BO with controllable contexts, a client can choose the context it tests, for example a material thickness, and has to learn the map $x^*(c)=\arg\max_x f(x,c)$ over all contexts. That means exploring across contexts as well as within each one, which the authors call substantially more resource-hungry than ordinary BO. Existing contextual methods learn from one client's data only. Existing collaborative and federated BO methods look for one optimum per client and do not handle variation across contexts. The paper claims to be the first to bring contextual BO into a collaborative setting.

## The idea, as I understand it

Each of $K$ clients fits its own Gaussian process (GP) over design and context. At every round a coin with probability $p_t$ decides the mode.

- **Collaborative mode.** Average all clients' posterior means into $\bar\mu$. Find the context where the client's own recommended design and the pooled model's recommended design disagree most, measured under $\bar\mu$, and play the pooled model's design there.
- **Independent mode.** Apply the same "largest disagreement" rule to a Thompson sample from the client's own GP. This is the multi-task Thompson sampling of Char et al.

$p_t$ starts at 1 and decays, so the client moves from borrowing to deciding for itself. Two variants matter for practice. One starts from peers' archived posterior means, the offline mode. The other shares ridge-fitted random-Fourier-feature weights instead of posterior means, for privacy.

What I like is the choice of context. <mark>The client spends an experiment where its own belief and the group's belief disagree most about what to do</mark>, which is a natural way to turn other clients' knowledge into an exploration signal.

## What the theory says

- **Regret.** If $\sum_t t\,p_t<\infty$, the last collaborative round is finite almost surely, after which the method is plain multi-task Thompson sampling. Expected regret is therefore $O(\sqrt{\gamma_T T})$ (Theorem 1).
- **Communication.** With $\sum_t p_t=O(\sqrt T)$, the communication cost is bounded with high probability (Theorem 2).

So, like consensus BO and LLINBO, <mark>the guarantee says collaboration cannot break convergence; it says nothing about how much it helps</mark>. The collaborative rounds are counted as worst-case regret.

## What the results show

There are no result tables, only regret curves with 95% bands: ten clients, ten replications, at most four dimensions, plus a hot-rolling case with ten clients and thirty replications.

- **Identical and mildly different clients.** On test functions CCBO has the lowest regret, but on the 2-D Ackley case the bands overlap.
- **Clients differ only by a small shift.** "Heterogeneous" means inputs shifted by at most 0.05 on a $[0,1]$ scale. That is mild, and the paper says strong heterogeneity is open.
- **More clients help, up to a point.** Two and five clients are worse and noisier, and fifteen is only slightly better than ten.
- **Privacy is cheap but not measured.** Sharing random-feature weights is slightly worse early and comparable later. No attack or formal privacy guarantee is evaluated.
- **The hot-rolling case runs on a neural-network stand-in** trained on 10,000 simulator samples, not on the simulator itself, and the stand-in's accuracy is not stated.

## Where I am not convinced

- **The experiments use a schedule the theorem excludes.** Every run uses $p_t=1/\sqrt t$, for which $\sum_t t\,p_t$ diverges, so the guaranteed regime is never tested.
- **The baselines are weak.** Federated Thompson sampling is paired with random context selection. There is no pooled multi-task GP and no contextual version of consensus BO.
- **The offline mode is described but never evaluated,** although starting from peers' archives is the most practical use.
- **No numbers.** Final regret, wall-clock time and tests are not reported.

## What I take from it

- **Borrowing should come with an exit, again.** This is the fourth paper from the group I have read with the same shape: borrow while data are scarce, then hand control back. Here the exit is a coin whose bias decays on a clock.
- **What to borrow is the open problem.** Averaging posterior means works when clients are near-copies. For my [battery simulator](/research/battery-degradation-sim/), where cells or protocols are the clients and C-rate or temperature is the context, the clients differ a lot. A learned prior from peers' archives, the untested offline mode, would be the version I would try first.
- **The disagreement rule is reusable.** Choosing where to experiment by where two models disagree about the best action is a cheap acquisition idea I could use in my learned-prior work.

## What I would try next

*Ideas, not results.*

1. **Offline mode on genuinely different clients.** Use archived posteriors from battery protocols at other temperatures as the starting $\bar\mu$, and measure how much bias the borrowed prior brings when clients are not shifted copies.
2. **Earned trust instead of a clock.** Let $p_t$ fall faster for a client whose own model already agrees with the pool, and test whether that keeps the benefit under stronger heterogeneity.

## References

- C.-Y. Chang, Q. Chen, T. Gao, D. Fenning, C. Okwudire, N. Dasgupta, W. Lu, R. Al Kontar. *Collaborative Contextual Bayesian Optimization.* arXiv:2604.18912, 2026. Code: github.com/cchihyu/Collaborative-Contextual-Bayesian-Optimization.
- I. Char, Y. Chung, W. Neiswanger, K. Kandasamy, A. O. Nelson, M. Boyer, E. Kolemen, J. Schneider. *Offline contextual Bayesian optimization.* NeurIPS 2019.
- Z. Dai, B. K. H. Low, P. Jaillet. *Federated Bayesian optimization via Thompson sampling.* NeurIPS 2020.
