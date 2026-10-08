---
title: "TDQN: An Application of Deep Reinforcement Learning to Algorithmic Trading"
paper:
  title: "An Application of Deep Reinforcement Learning to Algorithmic Trading"
  authors: "Thibaut Théate et al."
  venue: "Expert Systems with Applications, vol. 173, 2021, 114632"
  arxiv: "2004.06627"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "eswa-finance"
order: 5
tags: [reinforcement-learning, dqn, algorithmic-trading, sharpe-ratio, transaction-costs, evaluation]
date: 2020-04-01
status: draft
summary: "A double-DQN agent that flips between fully long and fully short on a single stock averages a Sharpe ratio of 0.404 across 30 stocks in 2018-2019, against 0.369 for buy and hold, and the paper is unusually candid about variance and overfitting."
---

## Abstract

The paper casts single-stock daily trading as a reinforcement learning problem and proposes TDQN, a Deep Q-Network adapted to it. The agent sees a window of open-high-low-close-volume data plus its current position, chooses between two actions (go fully long or fully short), and is rewarded with the daily portfolio return after a proportional trading cost. Since no simulator of the market exists, training relies on replaying the one historical price path under many different action sequences, with data augmentation and heavy regularisation. The second contribution is an evaluation protocol: a fixed testbench of 30 stocks and index funds across regions and sectors, one common set of hyperparameters, four classical benchmark strategies, and the Sharpe ratio as the main score. On the 2018-2019 test period TDQN beats the active benchmarks clearly and buy and hold narrowly. The authors report high run-to-run variance and visible overfitting on some stocks.

**Keywords:** deep reinforcement learning, DQN, algorithmic trading, trading costs, Sharpe ratio, performance assessment

## 1 Introduction

Most machine learning work on trading forecasts prices and leaves the trading rule as an afterthought. Reinforcement learning promises to learn the decision rule directly, with costs and position constraints inside the problem. The authors point at two obstacles that set markets apart from the games where deep RL succeeded: <mark>the environment is highly stochastic and very poorly observable</mark>, since the agent sees a sliver of what moves prices.

They are equally concerned with how results are reported. Their review argues that trading papers pick their own instrument, period and cost assumption, and that backtest overfitting is widespread (Bailey et al., 2014). The stated ambition is therefore twofold: a DRL trading policy, and a less biased way of judging any trading policy.

## 2 Background

**Q-learning and DQN.** An RL agent seeks a policy maximising the expected discounted return

$$
R = \sum_{t=0}^{\infty}\gamma^t r_t \tag{1}
$$

where $r_t$ is the reward at step $t$ and $\gamma\in[0,1]$ the discount factor. DQN (Mnih et al., 2013, 2015) approximates the action-value function $Q(s,a)$ with a neural network of parameters $\theta$, learns off-policy from a replay memory, and stabilises training with a slowly updated target network $\theta^-$. Double DQN (van Hasselt et al., 2015) reduces the overestimation of Q-values by selecting the next action with one network and evaluating it with the other.

**Objective.** The score the authors care about is the annualised Sharpe ratio,

$$
S_r = \frac{\mathbb{E}[R_s - R_f]}{\sqrt{\operatorname{var}[R_s - R_f]}} \simeq \frac{\mathbb{E}[R_s]}{\sqrt{\operatorname{var}[R_s]}} \tag{2}
$$

with $R_s$ the strategy return and $R_f$ the risk-free return, taken as negligible; daily values are scaled by $\sqrt{252}$.

## 3 Method

> **Key idea.** Shrink the trading problem until DQN can handle it: a two-action agent (all-in long or all-in short) on one stock, trained by replaying the historical path under alternative action sequences, with trading costs inside the environment so that the policy itself learns when a position change is worth paying for.

### 3.1 Observations, actions, rewards

The paper reduces the observation to

$$
o_t = \left\{ \{p^O_{t'}, p^H_{t'}, p^L_{t'}, p^C_{t'}, V_{t'}\}_{t'=t-\tau}^{t},\; P_t \right\} \tag{3}
$$

where the $p$ terms are daily open, high, low and close prices, $V$ is volume, $\tau$ the history length, and $P_t$ the current position (long or short).

The portfolio is cash $v^c_t$ plus $n_t$ shares, with $n_t$ allowed to be negative. Buying $Q_t$ shares at price $p_t$ updates cash as

$$
v^c_{t+1} = v^c_t - Q_t\,p_t - C\,|Q_t|\,p_t \tag{4}
$$

where $C$ is a proportional cost, a heuristic lumping together fees, spread, market impact and timing; it is set to 0.1%. Cash must stay non-negative, and when short it must satisfy

$$
v^c_{t+1} \ge -n_{t+1}\,p_t\,(1+\epsilon)(1+C) \tag{5}
$$

so that the agent can buy back its borrowed shares even after an adverse daily move of size $\epsilon$, a bound the agent assumes in advance. These constraints define an interval of admissible $Q_t$, which the paper then collapses to two actions: $Q^{\mathrm{Long}}_t$ converts as much cash as possible into shares, and $Q^{\mathrm{Short}}_t$ mirrors that holding on the short side, clipped to the lower bound implied by Eq. (5). Repeating the previous action trades nothing. There is no flat position.

The reward is the daily return of the portfolio value $v_t$,

$$
r_t = \frac{v_{t+1}-v_t}{v_t} \tag{6}
$$

The authors acknowledge that <mark>maximising discounted daily returns is only a relaxation of the Sharpe objective they evaluate on</mark>, and leave closing that gap to future work.

### 3.2 Training on artificial trajectories

Only one real trajectory exists per stock. New ones are made by keeping the historical prices and changing the agent's actions, which is valid as long as the agent is too small to move the market. Because there are just two actions, a simple exploration trick is available: at every step the opposite action is also executed on a copy of the environment and both transitions are stored in the replay memory.

> **My comment.** Replaying one path under different actions is valid only because the agent is a price-taker, and the moment I ask where richer trajectories could come from I am back at TailFlow's question: a generator used as the input model for a decision, whose errors flow into the choice. I wonder whether a policy trained on generated paths would mostly learn the generator's defects, for example the volatility clustering that CASE's paths still under-build.

### 3.3 Changes relative to vanilla DQN

The update target is the double-DQN one,

$$
y_i = r_i + \gamma\, Q\!\left(o_{i+1},\ \arg\max_{a\in\mathcal{A}} Q(o_{i+1},a;\theta);\ \theta^-\right) \tag{7}
$$

with $y_i=r_i$ at terminal states. The remaining changes are a catalogue of stabilisers: a feed-forward network with Leaky ReLU in place of the CNN, Adam in place of RMSProp, Huber loss, gradient clipping, Xavier initialisation, batch normalisation, and dropout, L2 and early stopping against what the authors describe as a strong tendency to overfit. Inputs are low-pass filtered, converted to daily changes and normalised; augmentation uses signal shifting, filtering and added noise. The DQN schematic is [Fig. 3 in the paper](https://arxiv.org/pdf/2004.06627#page=8).

## 4 Experiments

**Testbench.** 30 instruments: five index trackers and 25 stocks from US, European and Asian markets across technology, finance, energy, automotive and food. Training covers 2012-01-01 to 2017-12-31 and testing 2018-01-01 to 2019-12-31. One agent is trained per stock, hyperparameters are shared by all 30, and weights are frozen during the test. Benchmarks are buy and hold (B&H), sell and hold (S&H), trend following with moving averages (TF) and mean reversion with moving averages (MR), all using the same inputs and actions. Starting capital is \$100,000, and costs are 0.1%.

Sharpe ratios from Table 6 of the paper, where TDQN values are expected performance averaged over 50 training runs:

| Strategy | Average over 30 stocks | Apple (AAPL) | Tesla (TSLA) |
|---|---|---|---|
| B&H | 0.369 | 1.239 | 0.508 |
| S&H | -0.202 | -1.593 | -0.154 |
| TF | -0.331 | 1.178 | -0.987 |
| MR | -0.056 | -0.609 | 0.358 |
| **TDQN** | **0.404** | **1.424** | **0.621** |

<mark>TDQN averages 0.404 against 0.369 for buy and hold, while both active benchmarks average below zero.</mark> Going through the 30 rows of that table myself, TDQN is above B&H on 15 instruments, below on 12 and identical on 3 (the Dow Jones, S&P 500 and NASDAQ trackers, where the numbers match to three decimals). The authors read the ties as the agent learning to go passive when active trading looks too uncertain.

**Apple, a favourable case.** In one typical run (Table 4 of the paper) TDQN reaches a Sharpe of 1.484 with a profit of \$100,288, against 1.239 and \$79,823 for B&H; its maximum drawdown is 17.31% against 38.51%. See [Fig. 7](https://arxiv.org/pdf/2004.06627#page=13).

**Tesla, a deliberately unfavourable case.** The typical run (Table 5) has a Sharpe of 0.261, a profit of just \$98 and a maximum drawdown lasting 331 days, below B&H at 0.508. Yet the 50-run expectation in Table 6 is 0.621. <mark>The authors use this gap to show that a single trained agent can land far from the expected performance</mark>, and the training curve sits well above the test curve, which they read as overfitting ([Fig. 10](https://arxiv.org/pdf/2004.06627#page=14)).

**Trading costs.** With costs of 0%, 0.1% and 0.2% on Apple, the agent trades less and less, and ends up passive when costs become too high ([Fig. 11](https://arxiv.org/pdf/2004.06627#page=17)). The authors treat this as the main argument for RL over a forecast-then-trade pipeline.

## 5 Discussion

**Strengths.** The problem formalisation is careful. The evaluation protocol is the better half of the paper: a fixed and diverse testbench, shared hyperparameters, costs included, expected performance over 50 runs instead of a best seed, a failure case shown on purpose, and public code. The paper itself says TDQN "only barely" surpasses B&H.

**Weaknesses.**

- The margin is small and carries no uncertainty estimate. A difference of 0.035 in average Sharpe comes with no standard error, and the test window is two years. As a rough guide of my own, not the paper's, the sampling error of an annualised Sharpe ratio over $T$ years is on the order of $1/\sqrt{T}$, about 0.7 here, per instrument.
- One regime only. The test period was mostly bullish, by the authors' own account, and 2008 is excluded on purpose to avoid regime shifts that would hurt training. Robustness to regime change is therefore untested.
- Benchmarks are all classical. No forecasting-based or other RL baseline is included, by design, so the paper cannot say whether DQN is a good choice among learning methods.
- The policy is all-or-nothing. With no flat or fractional position the agent cannot express uncertainty through size; the only way to reduce risk is to stop switching.
- The ablation is missing. Roughly ten modifications to DQN are listed as helpful "experimentally", with no table showing what each contributes. The main text also does not state the discount factor finally used, and its discussion of $\gamma$ points both ways: long-horizon agents should trade less, yet a small $\gamma$ is also said to reduce trading frequency.

> **My comment.** Read as ranking and selection, this is choosing the best of five strategies from two years of daily data, with a 0.035 Sharpe gap. In rs-lab I would state that as a probability of correct selection, and I doubt two years per instrument comes anywhere near the sample a procedure with a 0.95 guarantee would ask for.

**What is not shown.** Per-stock variance across the 50 runs, turnover, and borrowing costs for the short side.

## 6 Takeaways

- TDQN is a heavily regularised double DQN with two actions, long or short, trained by replaying one historical path under alternative decisions.
- Against buy and hold the result is essentially a draw (0.404 vs 0.369 average Sharpe over two bullish years); against simple trend-following and mean-reversion rules it is a clear win.
- The lasting contribution is methodological: a fixed testbench, shared hyperparameters, costs included, and expected instead of best-run performance.
- Putting costs in the environment makes trading frequency an output of learning; the agent turning passive as costs rise is the cleanest behavioural result in the paper.
- Relevance to stochastic modelling of financial series: the binding constraint here is a single historical trajectory, which the authors work around with shifting, filtering and noise. A generative model of price paths is an obvious candidate for supplying richer training trajectories, and the authors' own last suggestion, learning return distributions instead of expectations, points the same way. The paper tests neither.

## References

1. Théate, T., Ernst, D. "An Application of Deep Reinforcement Learning to Algorithmic Trading." arXiv:2004.06627, 2020.
2. Mnih, V., et al. "Human-Level Control through Deep Reinforcement Learning." Nature, 518, 2015.
3. van Hasselt, H., Guez, A., Silver, D. "Deep Reinforcement Learning with Double Q-Learning." arXiv:1509.06461, 2015.
4. Moody, J., Saffell, M. "Learning to Trade via Direct Reinforcement." IEEE Transactions on Neural Networks, 12(4), 2001.
5. Bailey, D. H., Borwein, J. M., López de Prado, M., Zhu, Q. J. "Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance." Notices of the AMS, 2014.
