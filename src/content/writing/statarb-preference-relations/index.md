---
title: "Preference-relation StatArb: Statistical arbitrage portfolio construction based on preference relations"
paper:
  title: "Statistical arbitrage portfolio construction based on preference relations"
  authors: "Fredi Šarić et al."
  venue: "Expert Systems with Applications, 2023, 121906"
  arxiv: "2310.08284"
  license: "creativecommons.org/licenses/by-nc-nd/4.0/"
series: "eswa-finance"
order: 4
tags: [statistical-arbitrage, pairs-trading, preference-relations, potential-method, graph, portfolio-construction]
date: 2023-10-01
status: draft
summary: "Pairwise spread signals over hundreds of stocks contradict each other; projecting them onto a single utility vector by least squares gives a consistent ranking, and the resulting long-short portfolio earns 14.03% a year on 553 US stocks at 0.1% costs."
---

## Abstract

Pairs trading produces one signal per pair, and with hundreds of stocks the signals stop agreeing: A looks cheap against B, B against C, yet A looks expensive against C. This paper treats the full set of pairwise signals as a noisy preference graph and uses the potential method, a multiple-criteria decision-making tool, to find the closest graph that is internally consistent. The solution is a per-stock utility whose differences reproduce the pairwise preferences as well as possible in the least-squares sense, and it has a closed form. Graph transformations then keep only strong, extreme signals, weights are set in proportion to utility, and a holding rule called the momentum decorator cuts turnover. On daily data for 553 US stocks from 1990 to 2019, with costs of 0.1% and next-day-open execution, performance rises with the size of the universe, from roughly zero at 50 stocks to 14.03% annualised excess return on the full set.

**Keywords:** statistical arbitrage, pairs trading, preference relations, potential method, multiple-criteria decision making, long-short portfolios

## 1 Introduction

The textbook pairs trade (Gatev et al., 2006) finds two stocks whose normalised prices moved together, waits until the spread opens up, then buys the laggard and shorts the leader. Later studies cited by the authors report that its profitability has declined in developed markets.

One answer is to go multivariate. Jacobs and Weber (2015) attribute pairs profits mainly to slow diffusion of common information across many securities, which suggests looking at all stocks together. Existing multivariate approaches each carry a burden: Huck (2009, 2010) needs return forecasts before ranking, and the designs of Zhao et al. assume multivariate cointegration.

The authors take a different angle. <mark>They leave the pair signal arbitrary and focus entirely on the aggregation step: how to turn a set of mutually contradictory pairwise signals into one coherent long-short portfolio.</mark> The pair rule in the experiments is kept deliberately crude.

## 2 Background

A pairs strategy can be written as a relation $M(s_i,s_j)\in\{-1,0,1\}$ on the set of securities $S=\{s_1,\dots,s_N\}$: go long $s_i$ and short $s_j$, the reverse, or do nothing. A portfolio can honour all signals at once only if $M$ is a **preference relation**, meaning irreflexive, asymmetric and transitive. Transitivity is what real data violates, so the task is to find the preference relation closest to the raw $M$.

Multiple-criteria decision making usually solves this on a real-valued relaxation, a preference function $\rho(s_i,s_j)$ that carries intensity as well as direction. The potential method (Čaklović, 2012) is one such technique, first applied to statistical arbitrage by the same group (Mrčela et al., 2017).

## 3 Method

> **Key idea.** Assume every consistent set of pairwise preferences comes from a hidden per-stock utility, $\rho^*(s_i,s_j)=u^*(s_i)-u^*(s_j)$. Then cleaning up contradictory pair signals is a least-squares projection with a closed-form answer, and the utility itself gives both the ranking and the portfolio weights.

```mermaid
flowchart LR
  P["Pairwise z-scored log spreads (all pairs)"] --> U["Potential method: utility u per stock"]
  U --> R["Consistent preference graph"]
  R --> T["Edge thresholding (kappa)"]
  T --> V["Prune intermediate vertices: bipartite graph"]
  V --> K["Keep top n and bottom m by utility"]
  K --> W["Utility-proportional weights"]
  W --> H["Momentum decorator: hold until utility changes sign"]
```

### 3.1 The potential method

The defining condition is

$$
\rho^*(s_i,s_j) = u^*(s_i) - u^*(s_j) \tag{1}
$$

where $u^*:S\to\mathbb{R}$ is a latent utility. Irreflexivity, antisymmetry and transitivity all follow from Eq. (1). Stacking all pairs gives $B u^* = \rho^*$, with $B$ the $\binom{N}{2}\times N$ incidence matrix of the complete directed graph on the securities. Given a raw, inconsistent signal vector $\rho$, the method solves

$$
u^* = \arg\min_{u'} \lVert B u' - \rho\rVert_2^2 \quad \text{s.t.}\ \sum_i u'_i = 0 \tag{2}
$$

where the zero-sum constraint pins down the otherwise free additive constant. Because $B^\top B$ is the Laplacian of a complete graph, the solution collapses to

$$
u^* = \frac{1}{N} B^\top \rho, \qquad \rho^* = B u^* \tag{3}
$$

It is worth unpacking Eq. (3), although the paper does not phrase it this way: row $i$ of $B^\top\rho$ adds up stock $i$'s signal against every other stock, so <mark>the utility of a stock is simply its average pairwise signal against the whole universe</mark>. That also explains the paper's Theorem 1: if $\rho=\rho^*+\epsilon$ with zero-mean noise, the estimator is unbiased, and if the noise is uncorrelated across pairs its variance shrinks as $N$ grows.

### 3.2 Preference-preserving graph transformations

Taking the sign of $\rho^*$ would put almost every stock into a position. Three transformations, each shown to keep the graph a valid preference graph, are applied in order.

**Edge thresholding** removes weak edges:

$$
\rho^*_\kappa(s_i,s_j) = \begin{cases}\rho^*(s_i,s_j) & \text{if } |\rho^*(s_i,s_j)|\ge\kappa\\ 0 & \text{otherwise}\end{cases} \tag{4}
$$

**Vertex pruning** keeps only sources (zero in-degree, preferred to everything they touch) and sinks (zero out-degree); stocks in the middle of a path are dropped because a stronger signal exists at the ends. The graph becomes bipartite. **Final selection** keeps the top $n$ and bottom $m$ vertices by utility.

### 3.3 Weights and the momentum decorator

Within each leg, weights are proportional to absolute utility:

$$
w(s_i) = \pm\,\frac{|u^*(s_i)|}{\sum_{s_j\in T}|u^*(s_j)|} \tag{5}
$$

with the plus sign and $T=T_L$ for the long leg, the minus sign and $T=T_S$ for the short leg, and zero otherwise, so the portfolio is zero-investment. Daily re-selection churns the book, so the **momentum decorator** keeps an open position until the stock's utility changes sign or hits zero.

### 3.4 The pair signal used in the experiments

$$
\rho(s_i,s_j)^{(t)} = \frac{c^{(t)}_{i,j}-\mu^{(t)}_{i,j}}{\sigma^{(t)}_{i,j}}, \qquad c^{(t)}_{i,j}=\log\frac{p^{(t)}_i}{p^{(t)}_j} \tag{6}
$$

where $p^{(t)}_i$ is the price of stock $i$, and $\mu, \sigma$ are the mean and standard deviation of the log spread over the previous $T$ days, excluding day $t$ itself. Standardising the deviation makes one $\kappa$ fit all pairs.

## 4 Experiments

**Setup.** Daily prices of $N=553$ US stocks, 1990 to 2019. Lookback 60 days, $\kappa=3.0$, $m=n=20$, transaction costs 0.1%, orders filled at the next day's open, daily rebalancing. Robustness to the choice of universe is tested with a **security bootstrap**: $B=100$ random subsets of 50 and of 250 stocks, same parameters throughout.

Annualised excess returns of the long-short portfolio with utility-proportional weights and the momentum decorator (Table 2 of the paper; median over bootstrap subsets, 2.5% and 97.5% percentiles in brackets):

| Universe | Mean | Std. dev. | t-statistic |
|---|---|---|---|
| 50 stocks | -0.54% (-5.4%, 7.05%) | 21.18% (18.61%, 24.76%) | -0.14 (-1.31, 1.87) |
| 250 stocks | 8.95% (5.14%, 11.49%) | 17.24% (16.35%, 18.21%) | 2.74 (1.53, 3.52) |
| **553 stocks (full set, single run)** | **14.03%** | 18.95% | **3.94** |

<mark>Returns rise steadily with the number of securities, from about zero at 50 stocks to 14.03% at 553</mark>. At 250 stocks the whole bootstrap interval is already positive.

**Variants.** Four combinations are compared: equal (EW) or utility-proportional (UP) weights, with or without the decorator (w/ M). UP beats EW on mean return in every configuration ([Fig. 3 in the paper](https://arxiv.org/pdf/2310.08284#page=12)). <mark>The decorator lowers volatility and sharply lowers turnover in all cases, but on the full universe it does not raise the mean return.</mark> Holding periods stretch from a few days to as long as 23 days ([Fig. 2](https://arxiv.org/pdf/2310.08284#page=11), [Fig. 5](https://arxiv.org/pdf/2310.08284#page=13)).

**Factor exposure.** Fama-French five-factor regressions on the full universe, $n=7037$ daily observations (long-short rows of Table 3 of the paper; the paper marks significance at the 0.05 level in boldface, which did not survive my text extraction, so none is marked here):

| Method | $\alpha$ | $\beta_{MKT}$ | $\beta_{SMB}$ | $\beta_{HML}$ | $\beta_{RMW}$ | $\beta_{CMA}$ | adj. $R^2$ |
|---|---|---|---|---|---|---|---|
| EW | 0.045 | 0.254 | 0.030 | 0.204 | -0.134 | -0.329 | 0.05 |
| **UP** | **0.063** | 0.254 | 0.022 | 0.237 | -0.170 | -0.331 | 0.04 |
| EW w/ M | 0.023 | 0.248 | 0.025 | 0.225 | -0.083 | -0.290 | 0.09 |
| UP w/ M | 0.039 | 0.260 | 0.012 | 0.208 | -0.084 | -0.279 | 0.09 |

<mark>The factors explain at most 9% of the long-short variance.</mark> In separate regressions of the two legs, the long leg has a clearly positive alpha (0.045 to 0.071 depending on the variant), while the short leg's alpha is near zero (-0.002 to 0.010).

## 5 Discussion

**Strengths.** The problem is well posed and the solution is cheap: a closed form that scales to any universe and accepts any pair signal. The security bootstrap is a useful device, because it separates "the method works" from "the method works on this list of tickers".

**Weaknesses and open questions.**

- Once Eq. (3) is read as an average, with the simple z-score signal the ranking is close to a cross-sectional sort on how far each stock has drifted from the rest over 60 days. The paper includes no plain short-term reversal or Gatev-style top-pairs baseline, so it is impossible to say how much the graph machinery adds over a one-line sort.
- The market beta of the long-short book is about 0.25 in every variant. Low $R^2$ is not the same as neutrality, and the authors concede that zero-investment does not imply market-neutral.
- Theorem 1 needs noise that is uncorrelated across pairs. Pairs $(i,j)$ and $(i,k)$ share stock $i$, so an idiosyncratic shock to $i$ enters $N-1$ signals at once. The variance-reduction argument is therefore optimistic for exactly the noise source that matters most.
- The sign convention is never spelled out. As defined in Eq. (6), a large positive value means $s_i$ is rich relative to $s_j$, which in a convergence trade argues for shorting $s_i$; yet high-utility stocks go long.
- The universe is described only as roughly 500 S&P 500 stocks over three decades. Constituent selection is not discussed, so survivorship bias cannot be ruled out.
- Reporting is thin for a trading paper: no Sharpe ratio, drawdowns, sub-periods or borrowing costs, and the headline 14.03% is a single path with no interval. The units of $\alpha$ in Table 3 are not stated.

## 6 Takeaways

- Contradictory pairwise signals can be reconciled by projecting them onto a utility vector; on a complete graph this is an average of each stock's signals against all others.
- The gain from a larger universe is the paper's most robust empirical finding, supported by bootstrap intervals at 50 and 250 stocks.
- Without a simple reversal baseline, the evidence shows viability, not superiority over simpler alternatives.
- Relevance to diffusion or stochastic modelling of financial series is limited. The one transferable item is the evaluation idea: bootstrapping over asset subsets is a cheap way to test whether any cross-sectional model, generative or not, depends on a particular universe.

## References

1. Šarić, F., Begušić, S., Merćep, A., Kostanjčar, Z. "Statistical arbitrage portfolio construction based on preference relations." arXiv:2310.08284, 2023.
2. Gatev, E., Goetzmann, W. N., Rouwenhorst, K. G. "Pairs trading: Performance of a relative-value arbitrage rule." Review of Financial Studies, 19(3), 2006.
3. Čaklović, L. "Measure of inconsistency for the potential method." Lecture Notes in Computer Science, vol. 7647, 2012.
4. Mrčela, L., Merćep, A., Begušić, S., Kostanjčar, Z. "Portfolio optimization using preference relation based on statistical arbitrage." SST 2017.
5. Jacobs, H., Weber, M. "On the determinants of pairs trading profitability." Journal of Financial Markets, 23, 2015.
