---
title: "Battery Protocol Select — choosing a fast-charging protocol when the early-life prediction is uncertain"
slug: battery-protocol-select
category: "Simulation Optimization"
summary: "My first battery study, on the 124-cell Severson et al. fast-charging dataset. An early-life predictor that reports its own uncertainty stands in for the months-long cycle-life test, and the question is whether that uncertainty helps decide which charging protocols to keep testing. After 100 cycles a 10-feature Gaussian process reaches RMSE 118 / 225 cycles on the two standard test sets, against 138 / 196 for the one-feature baseline it reproduces. For a single pick from candidates pooled across the three batches, ranking by the probability of being best gains nothing over the point prediction (0.862 vs 0.861). Used to stop tests early, the same uncertainty does pay. Eliminating candidates whose probability is below α matches the 100-cycle pick of five candidates at 65 % of the testing. With the survivors run to end of life, it finds the longest-lived of five cells in 0.971 of draws at 41 % of the cheapest always-correct cost. It drops the true best about three times as often as α promises (0.029 at α = 0.01), and a surrogate that is too confident drops it in 0.359 of draws."
period: "2026.10 –"
status: "Ongoing experiment"
stack: [Python, NumPy, SciPy, pandas, pyarrow, h5py, Matplotlib, multiprocessing, pytest]
tags: [battery, cycle-life, early-prediction, fast-charging, gaussian-process, uncertainty-quantification, calibration, surrogate-model, ranking-and-selection, sequential-elimination]
metrics:
  - { label: "Early-life predictor after 100 cycles", value: "RMSE 118 / 225 cycles", note: "10-feature ARD GP on the primary / secondary test cells; the one-feature linear baseline gives 138 / 196, reproducing the published variance model; the GP's 90 % intervals cover 0.88 / 0.85" }
  - { label: "One-shot pick: P(best) vs point prediction", value: "0.862 vs 0.861", note: "k = 5 candidates pooled across batches, after 100 cycles, 4,000 draws; uncertainty adds nothing to a single pick here (within one batch at k = 10 it does help: 0.612 vs 0.568)" }
  - { label: "Sequential elimination, pick at 100 cycles", value: "same PCS at 65 % of the cost", note: "k = 5, α = 0.001: PCS 0.861 at 327 cycles, against 0.861 at 500 for testing all five; at k = 10 it costs 52 % but PCS falls from 0.830 to 0.819" }
  - { label: "Eliminate, then run survivors to end of life", value: "PCS 0.971 at 41 % of the cost", note: "k = 5, α = 0.01: 1,207 cycles against 2,976 for the cheapest always-correct rule; true best dropped in 0.029 of draws, about three times α" }
  - { label: "Overconfident surrogate (variance × 0.01)", value: "PCS 0.971 → 0.642", note: "P(true best eliminated) rises from 0.029 to 0.359 while the cost falls to 112 cycles" }
code: "projects/battery-protocol-select"
order: 13
kind: research
scope: personal
thumb: "/projects/battery-protocol-select/media/thumb.jpg"
---

## Abstract

A new fast-charging protocol is validated by cycling cells until they reach end of life. In the Severson et
al. (2019) dataset used here, that takes between 148 and 2,237 cycles per cell. Early-life prediction exists to
shorten this test. I ask a narrower question than how accurate the predictor is: **if the predictor reports its
own uncertainty, can that uncertainty decide which candidate protocols are still worth testing, and what happens
when it is wrong?**

On the 124 LFP/graphite cells, a one-feature linear model on $\log_{10}\operatorname{var}\Delta Q$ reproduces
the published variance model, with RMSE 138 and 196 cycles on the primary and secondary test sets. A 10-feature
Gaussian process reaches 118 and 225 cycles, and its 90 % intervals cover 0.88 and 0.85 of the primary and secondary test cells. The GP is then
cross-fitted, leaving out one group of charging policies at a time, and used as a surrogate in a
ranking-and-selection experiment with $k$ = 5 or 10 candidate cells and 4,000 draws.

- **One pick:** when candidates are drawn from all three batches, choosing by the probability of being best
  is no better than choosing by the point prediction: the largest gap, after 20 cycles with ten candidates, is
  0.018, and the bootstrap intervals overlap. Inside a single batch it does
  better at $k$ = 10 (0.612 against 0.568 after 100 cycles).
- **Adaptive testing:** the probability does help when it decides which tests to stop. After every block of 20
  cycles, candidates whose probability of being best falls below α are dropped. At α = 0.001 this matches the
  PCS of testing all five candidates for 100 cycles (0.861 vs 0.861) at 65 % of the cost. With ten candidates
  it costs 52 % but loses some PCS (0.819 vs 0.830).
- **Survivors to end of life:** at α = 0.01, running the remaining candidates to end of life gives PCS 0.971
  ($k$ = 5) and 0.922 ($k$ = 10). That costs 41 % and 31 % of the cheapest rule that is always correct.
- **Weak spots:** the nominal α is not a guarantee: the best cell is dropped in 0.029 and 0.078 of draws at
  α = 0.01, about three and eight times α. Inside a single test campaign the problem is harder: PCS 0.899 for
  five candidates.
- **Calibration:** shrinking the predictive variance by a factor of 100 cuts PCS from 0.971 to 0.642. The
  uncertainty is useful only to the extent that it is calibrated.

## 1 Introduction

<figure class="gif">
  <img src="/projects/diffusion-bridge/media/bridge.gif" alt="Lithium entering an electrode particle by diffusion, the Brownian walks behind it, and a diffusion model blurring a ring into noise and bringing it back" loading="lazy">
  <figcaption>One heat equation three times: lithium diffusing into an electrode particle (Fick), the Brownian walks behind it, and a diffusion model blurring a ring into noise and bringing it back.</figcaption>
</figure>


I trained as an electrical engineer before moving to industrial engineering. Batteries are the part of my
current field where that background helps most directly, because the raw material is current and voltage
traces from a cycler. For a first study in the area I did not want to build one more cycle-life predictor. My
reading notes already cover several:
- [Severson-style early prediction with a GP](/blog/battery-cycle-life-early-prediction/)
- [BatteryML](/blog/batteryml/)
- [BatteryLife](/blog/batterylife-benchmark/)
- [BatLiNet](/blog/battery-lifetime-diverse-aging/)
- [an interpretable pipeline on this same dataset](/blog/battery-cycle-life-ml-and-more/)

What those papers mostly leave open is **what the prediction is for**. The prediction feeds a decision.
Severson et al. built early prediction to shorten protocol testing, and Attia et al. (2020, *Nature*) then put
it inside a closed loop that searched over fast-charging protocols. A decision built on a prediction depends
on more than the point estimate. "This cell will last about 900 cycles" can justify stopping its test only if
the model also says how far off 900 might be, and only if that statement is right.

The question is the one behind my other studies. **If a model carries the uncertainty that the problem
actually has, can it act as a surrogate for the expensive evaluation, so that the decision is controlled
through the surrogate and adapted as data arrive?**
- In the [queueing study](/projects/exchange-queueing/), the expensive evaluation is a queue simulator. A
  surrogate re-solves a capacity decision every minute.
- Here the expensive evaluation is a physical life test. The surrogate is a Gaussian process with a
  predictive variance on log cycle life. The decision is which candidate protocols keep cycling, revisited
  every 20 cycles.

That makes the problem ranking and selection: pick the best of $k$ alternatives while spending as little
evaluation as possible. [rs-lab](/research/rs-lab/) builds that machinery with simulation outputs as the noisy
observations. [Input-Uncertainty Select](/research/input-uncertainty-select/) asks how much of its guarantee
survives an input model that was itself estimated. In this study the noisy observation becomes a prediction
error.

This is my first study in battery research, and I am ready to learn the field properly: the electrochemistry
behind the features as much as the data. I am looking forward to carrying the selection and surrogate ideas
of my earlier projects into a new domain. This study is built on those projects, and I intend to do my best
with the next steps listed in §5.

Three experiments:
- **E1** measures how accurate and how well calibrated the early-life predictor is after 20 to 100 cycles.
- **E2** uses it to select a protocol and counts the testing it saves.
- **E3** deliberately miscalibrates it to see what the selection loses.

## 2 Data and method

### 2.1 Data

The dataset is Severson et al., *Nature Energy* 4:383–391 (2019): 124 A123 APR18650M1A LFP/graphite cells in
three batches, fast-charged under 68 distinct published policies. I used the Hugging Face mirror
`bsebench-org/severson-2019` at a fixed revision.
- **Integrity.** All 124 parquet files (2,561,700,761 bytes) match their SHA-256 hashes.
- **Streaming.** Each file was downloaded, reduced to compact per-cycle features and a few Q(V) curves, then
  deleted. No raw time series are kept.

Three findings shaped the pipeline:

1. **Units.** The `current_A` column holds the original C-rate, not amperes: charge steps read 3.6 and
   discharge reads −4.0. Capacity is therefore 1.1 Ah × ∫I dt. With that correction the cycle-2 discharge
   capacity lies between 1.043 and 1.096 Ah across the three batches.
2. **Labels.** In batches 1 and 3 the parquet's own coulomb count never reaches the 0.88 Ah end-of-life
   threshold before the recording stops, and in batch 3 it runs above the original discharge capacity late in
   life. Labels are therefore the original `cycle_life`
   field of the `.mat` files. I read only their `summary` arrays from the raw mirror, by HTTP range requests:
   about 12.9 MB of the 8.27 GB, with nothing stored. The cross-checks agree:
   - Batch 2's coulomb-count life matches the label within 1 cycle for all 43 cells.
   - In batches 1 and 3, the last recorded cycle + 1 equals the label for 36 of 41 and 40 of 40 cells.
3. **Q(V) curves.** The parquet-derived discharge curves Q(V) match the original `Qdlin` arrays on nine spot-checked
   cells: the worst RMS difference is 0.00054 Ah, and $\log_{10}\operatorname{var}\Delta Q$ differs by at most
   0.012.

Cycle life ranges from 148 to 2,237 cycles, median 736.5.

| batch | cells | cycle life (min–max) |
|---|---|---|
| 1 | 41 | 534–2,237 |
| 2 | 43 | 148–713 |
| 3 | 40 | 541–1,935 |

Batch 2 is uniformly short-lived. The policy inferred from each cell's charge-current profile agrees with the
published policy for 0.976, 0.907 and 0.900 of cells in the three batches. The standard split comes from
Severson's released code: 41 training cells, 43 primary-test cells and 40 secondary-test cells (all of batch 3).

![Figure 1 — All 124 cells in grey; in colour the shortest-, median- and longest-lived cell (148, 742 and 2,237 cycles). (a) Discharge capacity to end of life (dashed: 0.88 Ah). (b) The 100th minus the 10th discharge curve, ΔQ₁₀₀₋₁₀(V). (c) Cycle life against log₁₀ var ΔQ, the published signature, with a least-squares line.](./figs/data_overview.png)

### 2.2 The early-life predictor

All models predict the natural log of cycle life, defined as cycles to 0.88 Ah. The key signal is the change in
the discharge curve between an early cycle and cycle $c$:

$$
\Delta Q_{c}(V) \;=\; Q_{c}(V) - Q_{10}(V),\qquad V\in[2.0,\,3.5]\ \text{V on 1,000 points}.
\tag{1}
$$

- **Baseline.** Ordinary least squares on $\log_{10}\operatorname{var}\Delta Q_c$, with Student-t prediction
  intervals. This is the one-feature version of the published variance model; with a single feature the
  elastic-net penalty makes no difference.
- **GP, var ΔQ only.** The same single feature in a Gaussian process.
- **GP, 10 features.** Severson's full-model features: var and min of ΔQ, the slope and intercept of the
  capacity fade, $Q_d(2)$, $\max Q_d - Q_d(2)$, the average charge time over cycles 2–6, the summed maximum
  temperature, the minimum internal resistance and its change.

The kernel works on standardised inputs and has three parts:

$$
k(x,x') \;=\; s_{\text{lin}}^{2}\,x^{\top}x' \;+\; s_f^{2}\,\text{Matérn}_{5/2}^{\text{ARD}}(x,x') \;+\; s_n^{2}\,\mathbb{1}[x=x'].
\tag{2}
$$

The hyper-parameters are type-II MAP estimates. The linear part lets the GP extrapolate like the published
linear models, and the Matérn part absorbs non-linearity. The configuration (feature set, noise floor,
linear part on or off) was chosen by 5-fold CV of the predictive log score **inside the training split
only**. The chosen setting is all ten features, noise floor 0.2 and the linear part on. Scalar features come
from the original summary arrays after a glitch filter, which set 40 discharge-capacity, 124 temperature and
349 charge-time values to missing. Both the Q(V) fallback and the glitch filter use only cycles at or before the
decision cycle: a missing or invalid discharge curve is replaced by the nearest earlier valid cycle (at most
three back), and a summary value is flagged by comparing it with the median of the previous eight cycles.

The surrogate used for selection must give honest predictions for every cell. It is therefore
**cross-fitted**: the 124 cells are split into 8 folds by published charging policy, and each fold is
predicted by a model that never saw any cell of its policies. Every cross-fitted model is refitted
separately at each $c \in \{20, 40, 60, 80, 100\}$.

### 2.3 Selection with an uncertain surrogate

Each draw has $k$ candidates: one random cell from each of $k$ distinct published policies. Its observed life
$L_i$ is the truth. At epoch $c_b$ (cycles 20, 40, …, 100) the surrogate gives a mean $\mu_i$ and standard
deviation $\sigma_i$ on log life. A candidate that has already died has its life revealed ($\sigma_i = 0$).
Among the surviving set $S_b$, each candidate's probability of being best is

$$
P_i(c_b) \;=\; \Pr\Bigl(\ell_i = \max_{j\in S_b}\ell_j\Bigr),\qquad
\ell_j \sim \mathcal N\bigl(\mu_j(c_b),\, s\,\sigma_j^{2}(c_b)\bigr)\ \text{independently},
\tag{3}
$$

estimated with 2,000 Monte Carlo samples. Here $s = 1$ except in E3. The shortest life in the data is 148
cycles, so no candidate dies before the last epoch at 100 cycles and the reveal step never fires there. Four kinds of rule are compared:

- **(a) point** — test every candidate for $c$ cycles, then pick $\arg\max_i \mu_i(c)$.
- **(b) P(best)** — test every candidate for $c$ cycles, then pick $\arg\max_i P_i(c)$.
- **(c) sequential elimination**, in the spirit of KN. After each block,

  $$
  S_{b} \;=\; \{\, i\in S_{b-1} : P_i(c_b)\ge\alpha \,\}\ \cup\ \{\arg\max_i P_i(c_b)\},
  \tag{4}
  $$

  and testing stops as soon as one candidate is left. If more than one survives at cycle 100, the rule
  either picks the top P(best) ("pick at 100") or keeps cycling the survivors until all but one have died
  ("survivors to end of life"). The second version is exact among the survivors.
- **(d) oracles** — run every candidate to end of life, or stop as soon as all but one have died. The second
  is the cheapest rule that is always correct.

**Scoring.**
- **Cost** is the total number of cycles run over all candidates. A cell stops costing once it dies or is
  dropped.
- **PCS** is the probability that the chosen cell is the longest-lived.
- **Regret** is the best life minus the chosen life, in cycles.
- **Intervals** are percentile bootstraps (1,000 resamples) over the 4,000 draws. They measure Monte Carlo
  error for these 124 cells and this fitted surrogate. They do not include the uncertainty of having only 124
  cells: the draws reuse the same cells many times.

## 3 Experiments

**E1. Predictor.**
- **Standard split:** the three models at $c$ = 20 … 100, scored by RMSE and MAPE in cycles, coverage of the
  90 % and 95 % intervals, and the predictive log score.
- **Cross-fit:** the same models, cross-fitted on all 124 cells.

**E2. Selection.**
- **Rules:** (a)–(d) for $k \in \{5, 10\}$, with α ∈ {0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2} for rule (c).
- **Main surrogate:** the cross-fitted 10-feature GP.
- **Robustness checks:**
  - the cross-fitted OLS baseline as the surrogate;
  - the GP fitted on the standard training split, with candidates drawn from the 83 test cells;
  - candidates drawn from a single batch. This is closer to one real test campaign, since policies and
    batches are confounded.

**E3. Calibration.**
- **Variance scaling:** the surrogate's predictive variance is multiplied by $s \in \{0.01, 0.03, 0.1, 0.3, 1,
  3\}$.
- **Rules re-run:** P(best) after 100 cycles and elimination at α = 0.01 and 0.05.

**Compute.**
- **Hardware:** the lab server `felabworkstation`, 64 cores, CPU only, at most 16 worker processes. The GP
  fits involve at most 124 cells, so no GPU was needed.
- **Run times:** E2 and E3 ran in 218 s. E1 took 53 s, the CV choice of GP configuration 52 s, and the
  Q(V) validation 119 s. Every result file records its host and wall time.
- **Tests:** 22 pytest checks pass on the server. They cover feature extraction on a synthetic cell, the
  selection rules on hand-built cases, and the GP and OLS fits.

## 4 Results

### 4.1 E1 — the predictor: better on the primary set, worse on batch 3, and overconfident

| c = 100, standard split | RMSE (cycles) primary / secondary | MAPE % primary / secondary | 90 % PI coverage primary / secondary |
|---|---|---|---|
| OLS on var ΔQ (baseline) | 138 / 196 | 14.8 / 11.4 | 0.977 / 0.925 |
| GP, var ΔQ only | 137 / 196 | 13.4 / 11.3 | 0.930 / 0.925 |
| GP, 10 features | 118 / 225 | 10.1 / 11.8 | 0.884 / 0.850 |

The baseline reproduces the published variance model: 138 and 196 cycles. These are recomputed from the data,
not copied from the paper. The 10-feature GP has the lowest MAPE on the primary test set (10.1 % against 14.8 %
for the baseline). Excluding the single 148-cycle cell, its primary-test MAPE is 7.9 %. On the secondary set
(batch 3) it is the worst of the three models on both measures: MAPE 11.8 % against 11.4 % and 11.3 %, and
RMSE 225 against 196. Pooled over all 83 test cells its RMSE is 178 cycles, against 168 for the baseline. This
fits the [BatteryML](/blog/batteryml/) finding that hand-crafted features in a linear model are hard to beat on
single-chemistry LFP data. It also matches the weak secondary-test result discussed in the [tutorial on this
dataset](/blog/battery-cycle-life-ml-and-more/).

The GP also overfits. Its training RMSE is 38 cycles, about a third of its primary-test error. Its noise term sits on the
chosen floor of 0.2, which is the top of the CV grid. A low error reported on the fitting data, as in the
[student GP replication](/blog/battery-cycle-life-early-prediction/), says little about new cells.

| all 83 test cells | c = 20 | c = 40 | c = 60 | c = 80 | c = 100 |
|---|---|---|---|---|---|
| OLS: RMSE (cycles) | 432 | 229 | 230 | 215 | 168 |
| OLS: 90 % coverage | 0.880 | 0.940 | 0.928 | 0.928 | 0.952 |
| 10-feature GP: RMSE (cycles) | 353 | 201 | 250 | 197 | 178 |
| 10-feature GP: 90 % coverage | 0.602 | 0.831 | 0.904 | 0.855 | 0.867 |

![Figure 2 — (a) The 10-feature GP after 100 cycles: predicted against observed cycle life with 90 % intervals, for training, primary-test and secondary-test cells. (b) RMSE on the 83 test cells against the number of cycles available to the predictor. (c) Coverage of the nominal 90 % interval against cycles, including the cross-fitted GP used as the selection surrogate.](./figs/predictor.png)

Extra cycles buy accuracy unevenly. The GP's error falls from 353 to 201 cycles between 20 and 40 cycles, rises to
250 at 60, and only falls to 178 at 100. At 20 cycles its intervals cover only 0.602, so they are far too narrow. The
cross-fitted version, which is the one E2 uses, is closer to the nominal 0.90 at every $c$ except 60 (0.919 against
0.904), though it still under-covers at 20 cycles:

| cross-fitted 10-feature GP, all 124 cells | c = 20 | c = 40 | c = 60 | c = 80 | c = 100 |
|---|---|---|---|---|---|
| RMSE (cycles) | 300 | 215 | 239 | 160 | 137 |
| MAPE % | 22.8 | 14.9 | 11.0 | 9.9 | 9.3 |
| 90 % coverage | 0.798 | 0.855 | 0.919 | 0.895 | 0.919 |

The cross-fitted OLS at $c$ = 100 gives RMSE 150, MAPE 13.5 % and coverage 0.944.

Figure 3 shows what an interval means for a single cell. The two cells are both from batch 3: among test cells
that lived 650 to 1,300 cycles, they are the one the GP predicts best and the one it misses by most. For the first,
the GP sees 100 cycles and predicts 857 cycles; the cell lasts 858. For the second, it predicts 751 with an upper
bound of 937, and the cell lasts 1,002. The linear baseline, with one feature, predicts 887 and is closer. The
interval is narrow and wrong at once. Section 4.3 shows what such overconfidence does to a selection rule.

![Figure 3 — Two batch-3 test cells after 100 cycles. Crosses: the discharge capacity the predictor sees. Dotted: the rest of the cell's life. Black dashed: the true end of life, where capacity reaches 0.88 Ah (grey). Purple band and line: the 10-feature GP's 90 % interval and median. Orange dashed: the linear baseline. Left, the cell the GP predicts best; right, the one it misses by most.](./figs/early_prediction.png)

### 4.2 E2 — selection: the uncertainty helps only when it decides what to stop testing

| rule ($k$ = 5) | PCS [95 % CI] | regret (cycles) | cost (cycles) | P(best dropped) |
|---|---|---|---|---|
| (a) point prediction after 20 cycles | 0.591 [0.576, 0.606] | 134 | 100 | — |
| (a) point prediction after 100 cycles | 0.861 [0.851, 0.872] | 10 | 500 | — |
| (b) max P(best) after 100 cycles | 0.862 [0.851, 0.872] | 10 | 500 | — |
| (c) eliminate at α = 0.001, pick at 100 | 0.861 [0.850, 0.871] | 10 | 327 | 0.004 |
| (c) eliminate at α = 0.01, pick at 100 | 0.851 [0.841, 0.862] | 13 | 269 | 0.029 |
| (c) eliminate at α = 0.01, survivors to end of life | 0.971 [0.965, 0.976] | 4 | 1,207 | 0.029 |
| (c) eliminate at α = 0.001, survivors to end of life | 0.996 [0.994, 0.998] | 0.9 | 1,613 | 0.004 |
| (d) run until all but one die | 1 | 0 | 2,976 | — |
| (d) run all to end of life | 1 | 0 | 3,303 | — |

| rule ($k$ = 10) | PCS [95 % CI] | regret (cycles) | cost (cycles) | P(best dropped) |
|---|---|---|---|---|
| (a) point prediction after 20 cycles | 0.368 [0.354, 0.383] | 291 | 200 | — |
| (a) point prediction after 100 cycles | 0.830 [0.819, 0.842] | 18 | 1,000 | — |
| (b) max P(best) after 100 cycles | 0.831 [0.820, 0.843] | 19 | 1,000 | — |
| (c) eliminate at α = 0.001, pick at 100 | 0.819 [0.808, 0.831] | 22 | 518 | 0.021 |
| (c) eliminate at α = 0.01, survivors to end of life | 0.922 [0.913, 0.930] | 13 | 1,915 | 0.078 |
| (c) eliminate at α = 0.001, survivors to end of life | 0.980 [0.976, 0.985] | 3 | 2,634 | 0.020 |
| (d) run until all but one die | 1 | 0 | 6,252 | — |
| (d) run all to end of life | 1 | 0 | 6,602 | — |

![Figure 4 — Probability of correct selection (a, b) and regret (c, d) against total testing cost, for k = 5 and k = 10 candidates. Point prediction and max P(best) are traced over c = 20 … 100 cycles. The sequential rules are traced over α from 0.2 to 0.001. Stars and crosses mark the two always-correct oracles. Error bars are bootstrap 95 % intervals.](./figs/selection_frontier.png)

1. **For a single pick from the pooled cells, the uncertainty adds nothing (a negative result).** Max P(best)
   and the point argmax cannot be told apart at any $c$: 0.862 against 0.861 at 100 cycles for $k$ = 5, and
   0.831 against 0.830 for $k$ = 10. The largest gap is 0.018, after 20 cycles at $k$ = 10 (0.386 against
   0.368), and the two bootstrap intervals still overlap there. If all candidates are tested equally long, the ranking of the means
   already carries the decision. This does not hold inside a single batch (see Robustness below).
2. **When the testing is adaptive, it saves cycles.** Dropping candidates whose P(best) falls below
   α = 0.001 reaches the same PCS as testing everyone for 100 cycles at $k$ = 5 (0.861 against 0.861) with
   327 cycles instead of 500 (65 %). At $k$ = 10 it uses 518 instead of 1,000 cycles (52 %) but gives up
   some PCS: 0.819 [0.808, 0.831] against 0.830 [0.819, 0.842]. Run the survivors to end of
   life, and PCS reaches 0.971 for $k$ = 5 at 1,207 cycles, which is 41 % of the 2,976 cycles of the cheapest
   always-correct rule. For $k$ = 10 it reaches 0.922 at 1,915 cycles, or 31 % of 6,252. The larger the
   candidate set, the larger the relative saving, but also the larger the PCS shortfall.
3. **The nominal α is not a guarantee.** At α = 0.01 the true best is eliminated in 0.029 ($k$ = 5) and
   0.078 ($k$ = 10) of draws, about three and eight times α. At α = 0.05 the figures are 0.099 and 0.155.
   There are four likely causes:
   - the rule looks at the data five times;
   - P(best) treats candidates as independent;
   - at early epochs the surrogate is overconfident (coverage 0.798 at 20 cycles);
   - P(best) is a Monte Carlo estimate from 2,000 samples, so at α = 0.001 a candidate is dropped when it
     wins at most one sample, and sampling noise alone can drop a candidate whose true probability is
     above α.

**Robustness.** For every surrogate and pool, elimination with survivors run to end of life beats the 100-cycle
point pick on PCS at well under the cheapest always-correct cost, but how much PCS it keeps depends on the pool.
The one-shot negative result does not hold everywhere: inside one batch, max P(best) beats the point prediction at $k$ = 10 after 100 cycles. A single test
campaign is also clearly harder:

| surrogate / pool | point after 100 cycles, PCS | eliminate α = 0.01, survivors to end of life: PCS · cost | cheapest always-correct cost |
|---|---|---|---|
| cross-fitted GP, all 124 cells ($k$ = 5) | 0.861 | 0.971 · 1,207 | 2,976 |
| cross-fitted OLS, all 124 cells ($k$ = 5) | 0.766 | 0.992 · 1,841 | 2,976 |
| GP on the standard training split, 83 test cells ($k$ = 5) | 0.806 | 0.958 · 1,306 | 3,298 |
| cross-fitted GP, candidates from one batch ($k$ = 5) | 0.674 | 0.899 · 2,000 | 3,702 |
| cross-fitted GP, candidates from one batch ($k$ = 10) | 0.568 | 0.766 · 2,338 | 6,641 |

- **OLS surrogate.** It is less accurate, so its one-shot pick is worse. Its wider intervals make the
  elimination more conservative: PCS is higher but the cost is higher too.
- **Single batch.** Inside one batch, at $k$ = 10, the true best is dropped in 0.234 of draws at α = 0.01.
  Here the one-shot P(best) pick does better than the point prediction after 100 cycles: PCS 0.612
  [0.596, 0.627] against 0.568 [0.552, 0.583]. A likely reason is that when the candidates'
  means are close, their different predictive variances change which one looks best.
  Much of the easy part of the pooled problem is telling short-lived batch 2 apart from batches 1 and 3. The
  within-batch rows are the more honest measure of one real campaign.

### 4.3 E3 — miscalibration: an overconfident surrogate stops early, cheaply and wrongly

At $s$ = 1 the cross-fitted intervals cover 0.919 at 100 cycles. Scaling the variance down makes the
surrogate overconfident: at $s$ = 0.1 the nominal 90 % interval covers 0.500, and at $s$ = 0.01 only 0.169.

| $k$ = 5, eliminate at α = 0.01, survivors to end of life | s = 0.01 | s = 0.03 | s = 0.1 | s = 0.3 | s = 1 | s = 3 |
|---|---|---|---|---|---|---|
| 90 % coverage at 100 cycles | 0.169 | 0.274 | 0.500 | 0.726 | 0.919 | 0.976 |
| PCS | 0.642 | 0.698 | 0.781 | 0.874 | 0.971 | 1.000 |
| P(true best eliminated) | 0.359 | 0.303 | 0.220 | 0.126 | 0.029 | 0.001 |
| cost (cycles) | 112 | 131 | 237 | 551 | 1,207 | 1,940 |

![Figure 5 — (a) Coverage of the nominal 90 % interval after 20 and 100 cycles as the predictive variance is scaled by s. (b) PCS, (c) probability that the true best is eliminated, and (d) testing cost, against s, for elimination at α = 0.01 (survivors to end of life), at α = 0.05 (pick at 100 and survivors to end of life), and for the P(best) pick after 100 cycles. Solid lines k = 5, dashed k = 10.](./figs/calibration_effect.png)

- **The adaptive rule depends on calibration.** An overconfident surrogate stops after 112 cycles in total,
  barely more than one block for each of five candidates. It discards the true best in 0.359 of draws, so
  the low cost is bought with wrong decisions. At $k$ = 10 the effect is larger: PCS falls from 0.922 to
  0.459, and P(true best eliminated) rises from 0.078 to 0.542.
- **Underconfidence wastes cycles.** At $s$ = 3, PCS is 1.000 at 1,940 cycles, 1.6 times the cost at
  $s$ = 1.
- **The one-shot pick ignores calibration.** P(best) after 100 cycles stays between 0.860 and 0.863 for
  $k$ = 5 across all $s$. Rescaling every variance by the same factor does not reorder the means, so it
  barely changes which candidate looks best.

This is the same failure that appears in my other studies. In the [queueing
study](/projects/exchange-queueing/), the surrogate controller promised a service level it did not deliver.
In [model-uncertainty-priors](/research/model-uncertainty-priors/), the GAN-based prior understated model
uncertainty. In each case a decision is controlled through the model's uncertainty, and it is only as good as
the calibration of that uncertainty.

## 5 Limitations & next steps

**Limitations**
- **The alternatives are cells, not protocols.** 43 of the 68 policies have a single cell, so PCS here means
  picking the best *cell*. Cell-to-cell spread within a protocol is large: the nine 4.8C(80%) cells last from
  461 cycles (batch 2) to 1,836 cycles (batch 3). A protocol decision would need replicate cells and a
  posterior on each protocol's mean.
- **Batch is confounded with protocol.** Batch 2 is uniformly short-lived. The pooled numbers are therefore
  partly a batch-identification problem, which is why the single-batch rows are reported beside them.
- **P(best) assumes independence.** In (3) the candidates' predictive distributions are independent. Within a
  cross-fitting fold they are jointly Gaussian and correlated, and the joint posterior is the correct object.
- **The surrogate is not updated during the selection.** Each epoch uses the model trained for that number of
  cycles. A candidate's own early cycles enter only through its features, not through a Bayesian update.
- **α is a tuning knob, not a guarantee.** With any of the GP surrogates, the measured rate of dropping
  the true best exceeds α at every α tried, from 0.001 to 0.2; at α = 0.01 it runs from 0.029 to 0.234
  depending on the pool and $k$. Only the wider-interval OLS surrogate stays
  below α, and only at $k$ = 5 with α ≤ 0.01 (0.0088 at α = 0.01). Repeated looks and the independence
  assumption probably both contribute; a KN-style correction for multiple looks was not applied.
- **The 10-feature GP overfits.** Its training RMSE is 38 cycles, and its noise term sits at the top of the CV
  grid. Its standard-split intervals cover 0.88 and 0.85, not 0.90.
- **The glitch filter came after a first run.** It was added after two primary-test cells with capacity
  glitches in the original summaries produced very large errors. The GP configuration was still chosen only
  by CV inside the training split, and the first run is not reported as a result.
- **Licence.** The mirror's card says CC-BY-4.0. One per-cell sidecar says "restricted_with_permission". Both
  are recorded in `results/data_manifest.json`.
- **One dataset, one chemistry.** All 124 cells are the same LFP cell type, cycled in one laboratory.

**Next steps**
- **Recalibrate before selecting.** Fit the variance scale, or a conformal correction, on the cross-fit, so
  that the rule's α means what it says. Then test whether the measured elimination rate falls to α.
- **Use a joint posterior, updated online.** Use the GP's joint predictive distribution, and refit as each
  candidate's cycles arrive. This turns the control rule from "drop below α" into a proper sequential
  Bayesian decision, closer to the KN and OCBA procedures in [rs-lab](/research/rs-lab/).
- **Select protocols, not cells.** Pool the replicate cells of each policy into a protocol-level posterior.
  Then ask the question at the protocol level, which is the decision Attia et al.'s closed loop actually
  makes.
- **Check transfer to other datasets.** Move to other datasets from the [BatteryLife](/blog/batterylife-benchmark/)
  and [BatteryML](/blog/batteryml/) collections, to see whether a surrogate calibrated on one chemistry stays
  calibrated on another.
