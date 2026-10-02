---
title: "PreBit: A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin"
paper: { title: "PreBit - A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin", authors: "Yanzhao Zou, Dorien Herremans", venue: "Expert Systems with Applications, vol. 233, 2023, 120838", arxiv: "2206.00648", license: "creativecommons.org/licenses/by-nc-nd/4.0/" }
series: "eswa-finance"
order: 1
tags: [bitcoin, twitter, finbert, multimodal, svm, extreme-events, backtesting, class-imbalance]
date: 2022-06-01
status: draft
summary: "A late-fusion SVM over a technical-indicator SVM and a FinBERT-embedding CNN flags next-day Bitcoin moves of 2% or 5%; tweets help on upward moves, but the price-only model is the better trader."
---

## Abstract

Zou and Herremans ask whether the text of ordinary Bitcoin tweets, rather than a sentiment score distilled from it, helps predict large next-day price moves. They collect up to 5,000 tweets per day from 2015 to mid-2021, embed them with FinBERT, and classify each day with a CNN. A separate SVM reads five days of candlestick data, technical indicators and the prices of Ethereum and gold. A third, tiny SVM fuses the two output probabilities. The target is binary: does tomorrow's high (or low) move at least 2% or 5% away from today's close. On a one-year test set the fusion model improves the positive-class F1 for upward moves, makes little difference for downward moves, and statistical significance is not established. A long-only backtest shows lower drawdown than holding Bitcoin, with the caveat that the price-only SVM performs at least as well as the fusion variants.

**Keywords:** Bitcoin, extreme price movement, Twitter, FinBERT, sentence embeddings, late fusion, SVM, focal loss, backtesting

## 1 Introduction

Bitcoin has a large retail following and no physical demand anchoring its value, so the authors expect its price to be unusually sensitive to public conversation.

Most earlier work with social media first reduces text to a sentiment score or to counts of posts and comments. The authors see two problems. General-purpose sentiment tools mishandle financial vocabulary such as "hold", "bull" or "chart", and a scalar score throws away the context. <mark>The paper's premise is that the full tweet content, embedded with a finance-tuned language model, carries information that a sentiment score discards.</mark> They found only one earlier study that fed text representations directly to a Bitcoin predictor, and it used bag-of-words features.

A second choice is to frame prediction as detection of extreme moves, not regression of price. A binary "up 5% tomorrow" signal can be plugged into a trading rule without further processing.

## 2 Background

**FinBERT** is BERT further trained on financial text, including a Reuters news corpus. It returns a 768-dimensional vector per input passage.

**Focal loss** reweights cross-entropy to cope with class imbalance:

$$
\mathrm{FL}(p_t) = -\alpha_t\,(1 - p_t)^{\gamma}\,\log p_t, \tag{1}
$$

where $p_t$ is the predicted probability of the true class, $\alpha_t \in [0,1]$ upweights the minority class, and $\gamma \ge 0$ shrinks the contribution of examples the model already classifies confidently.

**Labels.** For a threshold $\theta \in \{\pm 2\%, \pm 5\%\}$, a day is positive if the next day's high (for up tasks) or low (for down tasks) crosses $\theta$ relative to the close. The 5% tasks have roughly one positive per five negatives; the 2% tasks are close to 2:3.

## 3 Method

> **Key idea.** Keep the modalities separate until the last step. One model sees only prices, one sees only tweet embeddings, and a two-input SVM decides how much to trust each probability.

```mermaid
flowchart LR
  T["up to 5,000 tweets per day"] --> E["FinBERT, 362 x 768 matrix"]
  E --> C["Twitter CNN"]
  P["5 days x 19 price features"] --> S["TA SVM (RBF)"]
  C -->|"p_twitter"| F["Fusion SVM"]
  S -->|"p_TA"| F
  F --> Y["extreme move tomorrow?"]
```

The authors' own diagram is [Fig. 5 in the paper](https://arxiv.org/pdf/2206.00648#page=11).

### 3.1 Tweets to a fixed-size input

Tweets containing "Bitcoin" were scraped backwards from midnight GMT each day, 9,435,437 in total. They are lower-cased, stripped of URLs, "@" and "#" symbols and non-English characters. Embedding every tweet separately would be expensive, so a day's tweets are concatenated and cut into slices of 200 tokens with a 50-token overlap. Each slice gives one 768-dimensional FinBERT vector; a day becomes an $n \times 768$ matrix, zero-padded to $362 \times 768$, the largest $n$ in the data.

### 3.2 Twitter CNN

Two architectures are tried. The *parallel* CNN follows Kim's sentence classifier: filters spanning 3, 4 and 5 consecutive slices across the whole embedding width, global max pooling over each feature map, concatenation, two dense layers and a softmax. The *sequential* CNN treats the matrix as an image and stacks three 2-D convolutions with $5\times5$, $4\times4$ and $3\times3$ kernels in LeNet style. The parallel model has 2.6 million parameters and the sequential one 7.6 million. Both are trained with either cross-entropy or focal loss, with the choice and the values of $\alpha$, $\gamma$ made on a 10% validation split.

### 3.3 TA SVM

The price model uses 19 daily features: OHLCV, five moving averages, MACD, 20-day standard deviation, two Bollinger bands, high-low spread, a binary moving-average indicator, and Ethereum and gold prices. Price-linked features are expressed relative to the previous close,

$$
\tilde f_t = \frac{f_t - C_{t-1}}{C_{t-1}}, \tag{2}
$$

with $C_{t-1}$ the previous day's Bitcoin close; volume, Ethereum and gold are converted to their own day-on-day percentage change, and the volatility-type features are divided by $C_{t-1}$. Five days are concatenated into a 95-dimensional vector and classified by an RBF-kernel SVM whose $C$ and $\gamma$ come from a grid search with 4-fold cross-validation on F1.

### 3.4 Fusion and thresholding

The fusion SVM takes $x = (p_{\text{TA}}, p_{\text{Twitter}})$, a sigmoid of the TA SVM's decision value and the CNN's softmax output. Its own decision score is normally compared with 0.5. Because precision matters more than recall for a trader who only wants high-conviction entries, the authors also raise the cut-off $\tau$ to 0.95 and 0.99 after training, which changes only which test days are flagged.

## 4 Experiments

**Setup.** 2,337 daily observations from 1 January 2015 to 31 May 2021. The last 365 days (from 1 June 2020) form the test set and include both a strong rally and a sharp decline. Two random baselines, uniform and class-stratified, are simulated 1,000 times to give 95% intervals.

**Task Up 5%** (60 positives among 365 test days):

| Model | Precision (T) | Recall (T) | F1 (T) | Weighted F1 | Accuracy % |
|---|---|---|---|---|---|
| TA SVM | 0.32 | 0.22 | 0.26 | 0.78 | 71.23 |
| Twitter CNN (parallel) | 0.20 | 0.47 | 0.28 | 0.65 | 59.22 |
| Twitter CNN (sequential) | 0.18 | 0.95 | 0.22 | 0.24 | 26.10 |
| Fusion (parallel) | 0.31 | 0.48 | 0.37 | 0.76 | 73.42 |
| **Fusion (sequential)** | 0.31 | 0.50 | **0.38** | 0.76 | **73.70** |
| Stratified baseline | 0.16 | 0.16 | 0.16 | 0.72 | 72.49 |

<mark>Fusion roughly doubles the recall of the price-only model on 5% up-moves (0.22 to 0.50) at the same precision.</mark> On Up 2% the parallel fusion reaches positive F1 0.61 and accuracy 64.38% against 0.54 and 61.91% for the TA SVM. On the down tasks the gain disappears: for Down 5% the sequential fusion is identical to the TA SVM in every reported metric (F1 0.33, accuracy about 81.4%). The Twitter CNNs alone are weak; the sequential CNN on Up 5% flags almost everything as positive. Its F1 of 0.22 is as printed; its precision and recall imply about 0.30, and the paper prints 0.30 as that row's negative-class F1, so the two look swapped. <mark>The authors report that a Diebold–Mariano test did not reject the null hypothesis</mark>, and note that the stratified baseline's upper interval exceeds the fusion model's accuracy on Up 5%.

**Backtest** on Up 5% signals: buy at the close when flagged, sell at the next close, no fees, no slippage. Full test year:

| Strategy | Profit % | Sortino | Sharpe | Max drawdown % | Win % | Trades |
|---|---|---|---|---|---|---|
| Buy and hold | 249.3 | 3.28 | 2.08 | 45.5 | – | – |
| 7/21-day MA cross | 199.7 | 3.60 | 2.17 | 37.7 | 40.0 | 10 |
| **TA SVM** | 60.4 | **3.62** | 1.56 | 16.0 | 58.0 | 31 |
| Fusion, $\tau = 0.5$ | 1.4 | 0.26 | 0.18 | 12.4 | 56.0 | 50 |
| Fusion, $\tau = 0.95$ | 49.9 | 3.02 | 1.39 | 14.9 | 55.6 | 36 |
| Fusion, $\tau = 0.99$ | 56.6 | 3.45 | 1.49 | 16.0 | 56.6 | 30 |

In the bear segment (about the last 50 days) buy-and-hold loses 40.5% while the TA SVM and the $\tau = 0.99$ fusion both gain 32.0% on 7 trades. In the bull segment the $\tau = 0.95$ fusion has the best Sortino among the model strategies (5.97) with 8.3% drawdown. The price path and the bear-period trades are shown in [Figs. 10–11 of the paper](https://arxiv.org/pdf/2206.00648#page=24).

## 5 Discussion

**Strengths.** The dataset and code are public, which is rare in this literature. The evaluation is more careful than usual: a chronological split, per-class metrics, simulated random baselines with intervals, an honest report of a failed significance test, and separate bull and bear backtests.

**Weaknesses.** The headline claim needs qualifying. Text helps the classification metrics on up-moves only, and <mark>in trading the price-only SVM matches or beats every fusion variant over the full year</mark>; the default-threshold fusion earns 1.4%. At $\tau = 0.99$ the fusion model nearly reproduces the TA SVM's trades, which suggests the threshold is mostly filtering out what the Twitter branch added. Cut-offs of 0.95 and 0.99 are examined on the test year itself, not chosen on validation data. As far as Algorithm 3 describes it, the fusion SVM is fitted on probabilities that the base models produce for their own training days; out-of-fold probabilities are not mentioned, so the fusion stage may see over-confident inputs. The model-selection paragraph quotes an F1 of 0.97 for the SVM on Up 5% while the test table gives 0.26; the first number is presumably in-sample, but the text does not say. Finally, 60 positives, 7 bear-market trades and zero transaction costs are thin ground for a profitability claim, as the authors themselves acknowledge.

**Not shown.** A sentiment-score baseline on the same data, which is the comparison the motivation calls for; rolling or multi-period evaluation; results with trading fees.

## 6 Takeaways

- Late fusion of two probabilities is a cheap way to combine text and prices with only about two thousand daily samples.
- Full-text FinBERT embeddings of mass tweets add recall on upward extreme days; no benefit is visible on downward days.
- Classification gains did not carry over to the backtest: the technical-indicator SVM is the strongest model strategy on risk-adjusted terms.
- For stochastic modelling of financial series the relevant lesson is indirect: extreme moves are rare events (about 16% of days at the 5% level here), and point classifiers struggle with them. A generative model that produces a calibrated next-day distribution would deliver tail probabilities for any threshold at once, and text embeddings like these are a natural conditioning signal to test. The paper itself does not attempt this.

## References

1. Y. Zou, D. Herremans. *PreBit – A multimodal model with Twitter FinBERT embeddings for extreme price movement prediction of Bitcoin*. Expert Systems with Applications, vol. 233. arXiv:2206.00648.
2. D. Araci. *FinBERT: Financial sentiment analysis with pre-trained language models*. 2019.
3. Y. Kim. *Convolutional neural networks for sentence classification*. 2014.
4. T.-Y. Lin et al. *Focal loss for dense object detection*. 2017.
5. C. Lamon, E. Nielsen, E. Redondo. *Cryptocurrency price prediction using news and social media sentiment*. 2017.
