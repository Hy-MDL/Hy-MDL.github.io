---
title: "Rough volatility: Volatility is rough"
paper: { title: "Volatility is rough", authors: "Jim Gatheral et al.", venue: "arXiv 2014 (q-fin.ST)", arxiv: "1410.3394", license: "arxiv.org/licenses/nonexclusive-distrib/1.0/" }
series: "stochastic-modeling"
order: 7
tags: [rough-volatility, fractional-brownian-motion, hurst-exponent, realized-variance, long-memory, volatility-forecasting, rfsv, fractional-ornstein-uhlenbeck]
date: 2014-11-01
status: draft
summary: "Measuring how the moments of log-volatility increments scale with the lag gives one exponent near 0.1 for every asset and every moment order; that single measurement selects fractional Brownian motion with H < 1/2 as the driver, yields the RFSV model, explains why long-memory tests fire on data that has no long memory, and gives a one-parameter forecast of realized variance that beats HAR."
---

## Abstract

Gatheral, Jaisson and Rosenbaum ask how smooth the volatility path of a financial asset actually is, and answer with a measurement rather than a model fit. From daily volatility proxies built out of high-frequency data they compute the $q$-th absolute moment of log-volatility increments at lag $\Delta$ and regress it on $\Delta$ in logs. The relation is a clean power law, the exponent is proportional to $q$ with a constant near $0.1$, and the increments are close to Gaussian. Those two facts force fractional Brownian motion with Hurst exponent well below one half as the driver, packaged here as the Rough Fractional Stochastic Volatility (RFSV) model: a fractional Ornstein-Uhlenbeck log-volatility whose reversion is too slow to see. It reproduces the empirical autocovariance in the right coordinates, makes standard long-memory tests report memory that is not there, and gives a one-parameter forecast of realized variance that beats AR and HAR at every horizon tested.

**Keywords:** rough volatility, fractional Brownian motion, Hurst exponent, realized variance, fractional Ornstein-Uhlenbeck, monofractal scaling, spurious long memory, volatility forecasting, Hawkes processes

## 1 Introduction

Continuous-time finance writes the log-price as $dY_t = \mu_t\,dt + \sigma_t\,dW_t$, and the modelling choice that matters is the volatility $\sigma_t$. Before this paper the standard menu offered two levels of path regularity and nothing between: the essentially smooth volatility of Black-Scholes and Dupire's local volatility, and the Brownian-driven volatility of Hull-White, Heston and SABR, exactly as irregular as Brownian motion and no more. A third option existed but pointed the wrong way: Comte and Renault modelled log-volatility with a fractional Brownian motion (fBM) and chose $H > 1/2$, because that produces long memory — then an accepted stylized fact, traced to Ding-Granger-Engle and to Andersen and Bollerslev.

The paper attacks this from two sides. First, options. The at-the-money skew $\psi(\tau) = \partial_k \sigma_{BS}(k,\tau)\big|_{k=0}$ is well described by a power law in time to expiry; the authors fit $\psi(\tau) = A\,\tau^{-0.4}$ to SPX closing quotes on 20 June 2013 ([Fig. 1.2](https://arxiv.org/pdf/1410.3394#page=5)), where conventional stochastic volatility gives a skew flat for small $\tau$ and then decaying as a sum of exponentials. Fukasawa had shown that fBM-driven volatility generates $\psi(\tau) \sim \tau^{H-1/2}$ for small $\tau$, so matching a skew that blows up at short expiry needs $H$ near zero; with $H > 1/2$ the skew *increases* with expiry, which no equity market does. <mark>The same asymptotic that makes fBM attractive to Comte and Renault rules out their sign of $H - 1/2$.</mark> It also removes the standard argument that an exploding short-dated smile implies jumps.

Second, and the actual subject of the paper: the realized volatility series itself says $H$ is small — a statement about the physical measure, obtained without touching an option price.

## 2 Background

A fractional Brownian motion $W^H$ with Hurst parameter $H \in (0,1)$ is a centered self-similar Gaussian process with stationary increments whose absolute moments scale exactly:

$$
\mathbb{E}\big[\,|W^H_{t+\Delta} - W^H_t|^q\,\big] = K_q\,\Delta^{qH}, \qquad q > 0, \tag{1}
$$

where $K_q$ is the $q$-th absolute moment of a standard normal variable. For $H = 1/2$ this is ordinary Brownian motion. Sample paths are Hölder continuous of any order below $H$, so smaller $H$ means a rougher path. For $H > 1/2$ the increments are positively correlated, $\mathrm{Cov}[W^H_1, W^H_k - W^H_{k-1}]$ decays like $k^{2H-2}$, and the sum over $k$ diverges — long memory. In fBM, regularity and memory are the same parameter, which is why picking $H$ to get memory also picks the skew, and why getting one right meant getting the other wrong.

Nothing here needs option-pricing machinery — the calibration side is covered in [Deep Learning Volatility](/blog/deep-learning-volatility/) and [Deep calibration](/blog/deep-calibration-rough-vol/) — or any learned dynamics. The model is four parameters wide and is fitted by reading slopes off log-log plots.

## 3 Data, scaling and the RFSV model

> **Key idea.** Do not assume a volatility model and then estimate its parameters. Measure how the moments of log-volatility increments scale with the lag. One exponent fits every moment order and every asset, it is about $0.1$, and that single number selects fBM with $H \ll 1/2$ as the driver — the model is a consequence, not an input.

### 3.1 Data and volatility proxies

Spot volatility is not observable, so the authors use two daily proxies and keep the difference between them in view throughout. For DAX and Bund futures they have tick data covering 1,248 days (13 May 2010 to 1 August 2014, most liquid maturity each day) and estimate integrated variance from 10 to 11 am London time with the noise-robust "model with uncertainty zones" estimator; a one-hour window is short next to the multi-day lags of interest, so this is close to a spot value. For the S&P and NASDAQ they take Oxford-Man 5-minute realized variance over the whole trading day, 3,540 days from 3 January 2000 to 31 March 2014. The second proxy averages over eight hours rather than one, and integration smooths, so it should bias the measured regularity upward. The authors say so in advance and then measure the bias twice — by simulation, and analytically in Appendix C.

### 3.2 The scaling measurement

For lag $\Delta$ and order $q$, define the sample moment

$$
m(q,\Delta) = \frac{1}{N}\sum_{k=1}^{N} \big|\log \sigma_{k\Delta} - \log \sigma_{(k-1)\Delta}\big|^q ,
\qquad N = \lfloor T/\Delta \rfloor . \tag{2}
$$

For a given $\Delta$ several such sums exist depending on the starting offset; the reported $m(q,\Delta)$ averages them. The working assumption is that $N^{q s_q} m(q,\Delta)$ converges as $\Delta \to 0$, which puts the volatility path in the Besov space $B^{s_q}_{q,\infty}$ and in no smoother one — so $s_q$ is regularity measured in $L^q$, and if $\log\sigma$ were fBM then $s_q = H$ for every $q$.

The estimation is two nested regressions. Regress $\log m(q,\Delta)$ on $\log\Delta$ for $\Delta = 1,\dots,30$ days and $q \in \{0.5, 1, 1.5, 2, 3\}$; each $q$ gives a slope $\zeta_q$, so $m(q,\Delta) \propto \Delta^{\zeta_q}$ ([Fig. 2.1-2.2](https://arxiv.org/pdf/1410.3394#page=9)). Then regress $\zeta_q$ on $q$. <mark>The second regression is the load-bearing one: if $\zeta_q$ is linear in $q$, a single exponent describes all moment orders — monofractal scaling, exactly the signature (1) of fBM.</mark> It is linear, with $\zeta_q \approx qH$ and

<mark>$H = 0.125$ (DAX), $0.082$ (Bund), $0.142$ (S&P), $0.139$ (NASDAQ)</mark> — bond futures and equity indices, tick-level and five-minute proxies, all in the same narrow band. Across all 21 Oxford-Man indices the estimates $\zeta_q/q$ run from $0.071$ (MXX) to $0.158$ (SSMI); by halves of each series, $0.059$ to $0.206$, systematically higher in the half containing the 2008 crisis. The authors describe this as "between 0.06 and 0.20", which their own Table B.2 marginally exceeds at both ends. The $\zeta_q$ curves are slightly concave; simulated fBM paths of the same length are too, so the authors charge it to sample size.

The distributional half of the evidence is the histogram: increments at lags of 1, 5, 25 and 125 days are close to Gaussian, and — the sharper check — <mark>the one-day normal fit rescaled by $\Delta^H$ lands on top of the fitted normals at longer lags</mark> ([Fig. 2.7](https://arxiv.org/pdf/1410.3394#page=13)). Scaling and Gaussianity are verified together, not separately.

### 3.3 From scaling to the model

Gaussian increments obeying (1) suggest the simplest thing that can produce them:

$$
\log \sigma_{t+\Delta} - \log \sigma_t = \nu\,\big(W^H_{t+\Delta} - W^H_t\big),
\qquad\text{equivalently}\qquad
\sigma_t = \sigma\exp\{\nu W^H_t\}, \tag{3}
$$

where $\nu$ sets the size of a one-day move and $H$ how that size grows with the lag. This process is not stationary, which is unattractive both mathematically and at very long horizons, so the authors replace fBM by a stationary fractional Ornstein-Uhlenbeck process $X$ and set $\sigma_t = \exp(X_t)$:

$$
dX_t = \nu\,dW^H_t - \alpha\,(X_t - m)\,dt ,
\qquad
X_t = \nu\!\int_{-\infty}^{t} e^{-\alpha(t-s)}\,dW^H_s + m , \tag{4}
$$

where $m$ is the long-run mean of log-volatility, $\alpha > 0$ the reversion speed, and the stochastic integral is pathwise Riemann-Stieltjes. The decisive choice is $\alpha \ll 1/T$ for the observation window $T$. Proposition 3.1 is exact: as $\alpha \to 0$, $\mathbb{E}\big[\sup_{t\le T}|X^\alpha_t - X^\alpha_0 - \nu W^H_t|\big] \to 0$, so the *increment* of $X$ from the start of the window converges uniformly in expectation to $\nu W^H$; Corollary 3.1 then transfers the scaling law (1) to $X$. Formally RFSV is the Comte-Renault FSV model with both parameter choices inverted — $H < 1/2$ with $\alpha \ll 1/T$ instead of $H > 1/2$ with $\alpha \gg 1/T$.

Two consequences follow, both exact in the $\alpha\to 0$ limit and both testable. The first is the autocovariance:

$$
\mathrm{Cov}\big[\log\sigma_t, \log\sigma_{t+\Delta}\big] = \mathrm{Var}[\log\sigma_t] - \tfrac{1}{2}\,\nu^2\,\Delta^{2H} + o(1) . \tag{5}
$$

The correct abscissa is therefore $\Delta^{2H}$, not $\Delta$ and not $\log\Delta$: plotted that way the S&P log-volatility autocovariance is a straight line at $H = 0.14$ ([Fig. 3.1](https://arxiv.org/pdf/1410.3394#page=17)). At the lags considered $\mathrm{Var}[\log\sigma_t]$ dominates the second term, so the decay is gentle. Second, since $X$ is Gaussian, $\mathbb{E}[\sigma_t\sigma_{t+\Delta}]$ is an exponential of (5), so $\log \mathbb{E}[\sigma_t\sigma_{t+\Delta}]$ is again linear in $\Delta^{2H}$ ([Fig. 3.2](https://arxiv.org/pdf/1410.3394#page=18)) and <mark>the autocovariance of volatility is not a power law — the log-log plot is visibly curved, in the data and in the model</mark> ([Fig. 3.3](https://arxiv.org/pdf/1410.3394#page=19)).

The same closed form kills the alternative directly. Since $m(2,\Delta) = 2(\mathrm{Var}[\log\sigma_t] - \mathrm{Cov}[\log\sigma_t,\log\sigma_{t+\Delta}])$, the exact fOU autocovariance gives $m(2,\Delta)$ for any $(H,\alpha)$. Plotted for the FSV estimates of Chronopoulou and Viens ($H = 0.53$) with $\alpha = 0.5$, it rises steeply and then flattens at lag $\approx 1/\alpha$; the empirical points do neither ([Fig. 3.4](https://arxiv.org/pdf/1410.3394#page=20)).

### 3.4 Intuition: what $\alpha \ll 1/T$ buys, and why roughness is visible

The stationary variance of (4) is $\mathrm{Var}[\log\sigma_t] = H(2H-1)\nu^2\alpha^{-2H}\Gamma(2H-1)$, positive for $H<1/2$ (both $H(2H-1)$ and $\Gamma(2H-1)$ are negative) and divergent as $\alpha \to 0$. That is the whole trick in one line: the level of log-volatility becomes arbitrarily diffuse while every *increment* keeps the fixed law $\mathcal{N}(0,\nu^2\Delta^{2H})$. Over a finite window you see a fBM with an unknown offset; over geological time the process is still stationary. At the paper's $\alpha = 5\times10^{-4}$ per day the reversion scale is about 2,000 days, longer than either dataset.

Roughness itself is easiest to feel as a ratio. Take $\nu = 0.3$, $H = 0.14$. A one-day log-volatility move has standard deviation $0.30$; a 20-day move has $\nu\,20^{0.14} = 0.46$, where a Brownian driver with the same daily size would give $0.3\sqrt{20} = 1.34$. Short-horizon variation is large relative to long-horizon variation — volatility jitters violently day to day yet wanders slowly over a month, the pattern practitioners call mean reversion.

Because $H$ is small, $\Delta^H$ grows very slowly, and the authors exploit that. The model's rescaled volatility path on $[0,\Delta]$ is approximately a geometric fBM with coefficient $\nu\Delta^H$; between one day and five years (1,250 trading days) that coefficient multiplies by only $1250^{0.14} \approx 2.7$. <mark>A volatility path over one day therefore looks statistically like one over a decade — the fractal appearance of real realized-variance series</mark> ([Fig. 3.6](https://arxiv.org/pdf/1410.3394#page=22)).

### 3.5 Algorithm

```text
ESTIMATE H  (per asset)
  input: daily spot-volatility proxies sigma[1..T]
  for q in {0.5, 1, 1.5, 2, 3}:
      for Delta in 1..30:
          m[q,Delta] = mean over starting offsets of
                       mean_k |log sigma[k*Delta] - log sigma[(k-1)*Delta]|^q
      zeta[q] = slope of OLS( log m[q,Delta] ~ log Delta )
  H     = slope of OLS( zeta[q] ~ q )          # through the origin
  nu^2  = exp( intercept of the q = 2 regression )

SIMULATE RFSV on [0,T], step d
  W = fBM(H) path on the grid          # spectral method
  X[0] = m
  for n = 0,1,...:
      X[n+1] = X[n] + nu*(W[n+1]-W[n]) + alpha*d*(m - X[n])
  sigma = exp(X)

FORECAST log-variance Delta days ahead      # one parameter: H
  w(u)  = cos(H*pi)/pi * 1/((u+1) * u^(H+1/2))
  pred  = Riemann sum over past days u > 0 of w(u) * log sigma^2[t - Delta*u]

FORECAST variance
  c     = Gamma(3/2 - H) / (Gamma(H + 1/2) * Gamma(2 - 2H))
  pred2 = exp( pred + 2*c*nu^2*Delta^(2H) )   # lognormal correction
```

The forecast rests on the conditional expectation of fBM with $H<1/2$ (Nuzman and Poor), which together with $\log\sigma_t^2 \approx 2\nu W^H_t + C$ gives

$$
\mathbb{E}\big[\log\sigma^2_{t+\Delta}\,\big|\,\mathcal{F}_t\big]
= \frac{\cos(H\pi)}{\pi}\,\Delta^{H+1/2}\int_{-\infty}^{t}\frac{\log\sigma_s^2}{(t-s+\Delta)\,(t-s)^{H+1/2}}\,ds . \tag{6}
$$

The constants $2\nu$ and $C$ cancel, which is why $H$ is the only parameter. Substituting $s = t - \Delta u$ removes $\Delta$ from the kernel, leaving $\tfrac{\cos(H\pi)}{\pi}\int_0^\infty \log\sigma^2_{t-\Delta u}\,\big[(u+1)u^{H+1/2}\big]^{-1}du$: weights depend on the past only through *how many forecast horizons ago* it was. Truncating at $u = 1$ costs $\varepsilon = 0.35$ of the kernel mass, so <mark>to forecast $\Delta$ days ahead you mostly need the last $\Delta$ days — the relevant window is linear in the horizon</mark>, a rule practitioners already use. As $H \to 0$ the kernel reduces to the multifractal-random-walk predictor of Duchon, Robert and Vargas.

## 4 Implementation notes

| Item | As reported |
|---|---|
| Proxy A (DAX, Bund) | integrated variance 10-11 am, "uncertainty zones" estimator; 1,248 days |
| Proxy B (indices) | Oxford-Man 5-min realized variance, whole day; 3,540 days |
| Scaling regression | $\Delta = 1..30$ days, $q \in \{0.5,1,1.5,2,3\}$, offsets averaged |
| Simulation length | 2,000 days |
| Simulated parameters | $H = 0.14$, $\nu = 0.3$, $m = X_0 = -5$, $\alpha = 5\times10^{-4}$ |
| fBM simulation | spectral method, 40,000,000 points (20,000 per day), Euler step $\delta = 1/20000$ |
| Price simulation | $P_{(n+1)\delta} - P_{n\delta} = P_{n\delta}\,\sigma_{n\delta}\sqrt{\delta}\,U_n$, $U_n$ i.i.d. standard normal |
| Microstructure | uncertainty-zones model, tick value $5\times10^{-4}$, $\eta = 0.25$ |
| Long-memory test 2 | fractional differencing order $d = 0.4$, Bartlett bands |
| Forecast benchmarks | AR(5), AR(10) via Yule-Walker on a rolling 500-day window; HAR(3) by OLS |
| Forecast evaluation | $k$ from 500 to $N-\Delta$; $H$ estimated once per asset on the full sample |
| Compute | not stated |
| Code | not released |

Details that are easy to get wrong. The second regression must go through the origin in $q$, or the reported $H$ is not the slope the theory predicts. The intercept of the $q=2$ regression, not its slope, carries $\nu^2$, and it feeds the lognormal correction in the variance forecast. The forecast kernel has an integrable singularity at $s \to t$ and a heavy tail at $s \to -\infty$; the paper says only that a Riemann sum is used, so the discretisation near $u=0$ and the truncation radius are both unspecified. The evaluation treats the proxies as the true $\sigma^2$ on both sides of the error, for RFSV and the baselines alike. Appendix C is worth reading first: in a tractable surrogate where $v_t=\sigma_t^2$ itself is fBM-driven, smoothing over a window of length $\delta$ turns $m(2,\Delta)$ into $\alpha^2\Delta^{2H}f(\delta/\Delta)$ with $f$ rising to 1, inflating the log-log slope and deflating the intercept — the same direction and size the RFSV simulation shows.

## 5 Experiments

### 5.1 Recovering the estimator's bias

Running the whole pipeline on 2,000 simulated days with true $H = 0.14$: the one-hour uncertainty-zones proxy returns $H \approx 0.16$, the eight-hour realized-variance proxy $H \approx 0.18$, and in both cases $\zeta_q$ is still linear in $q$. The fSS calculation of Appendix C gives $0.161$ and $0.184$ for window lengths $1/24$ and $1/3$ of a day against a true $0.140$, with the fitted $\nu$ falling from $0.300$ to $0.263$ and $0.230$. <mark>Both routes say the same thing: the index estimates near $0.14$ are upper bounds, and the true exponent is probably nearer $0.1$.</mark>

### 5.2 Spurious long memory

Two classical procedures are run on S&P data and on a simulated RFSV path of 3,500 days — the length of the real sample. The first tracks how $V(t) = \mathrm{Var}[\int_0^t \sigma_s^2\,ds]$ grows with $t$; if $V(t)\sim t^{2-\gamma}$ then the framework of Andersen et al. reads off an autocorrelation decaying like $t^{-\gamma}$. Data and simulation both give a log-log slope of $1.86$, hence $\gamma = 0.14$ and a verdict of long memory. The second fractionally differences log-volatility with $d = 0.4$ and inspects the residual autocorrelation for whiteness; again the two panels are hard to tell apart. <mark>A model with no long memory passes both tests, and returns the same memory parameter the literature reports for real data.</mark>

### 5.3 Forecasting

The score is $P$, the mean squared forecast error divided by the sample variance of log-variance, so lower is better and $P=1$ is the unconditional mean. Table 5.1 (log-variance, all 15 cells):

| Index, horizon | AR(5) | AR(10) | HAR(3) | RFSV |
|---|---|---|---|---|
| SPX, $\Delta = 1$ | 0.317 | 0.318 | 0.314 | **0.313** |
| SPX, $\Delta = 5$ | 0.459 | 0.449 | 0.437 | **0.426** |
| SPX, $\Delta = 20$ | 0.764 | 0.694 | 0.656 | **0.606** |
| FTSE, $\Delta = 1$ | 0.230 | 0.229 | 0.225 | **0.223** |
| FTSE, $\Delta = 5$ | 0.357 | 0.344 | 0.337 | **0.320** |
| FTSE, $\Delta = 20$ | 0.651 | 0.571 | 0.541 | **0.472** |
| Nikkei, $\Delta = 1$ | 0.357 | 0.358 | 0.351 | **0.345** |
| Nikkei, $\Delta = 5$ | 0.553 | 0.533 | 0.513 | **0.504** |
| Nikkei, $\Delta = 20$ | 0.875 | 0.795 | 0.746 | **0.714** |
| DAX, $\Delta = 1$ | 0.237 | 0.238 | 0.234 | **0.231** |
| DAX, $\Delta = 5$ | 0.372 | 0.362 | 0.350 | **0.339** |
| DAX, $\Delta = 20$ | 0.661 | 0.590 | 0.550 | **0.498** |
| CAC40, $\Delta = 1$ | 0.244 | 0.244 | 0.241 | **0.238** |
| CAC40, $\Delta = 5$ | 0.378 | 0.373 | 0.366 | **0.350** |
| CAC40, $\Delta = 20$ | 0.669 | 0.613 | 0.598 | **0.522** |

Table 5.2 repeats the exercise on variance rather than log-variance, with the lognormal correction $2c\nu^2\Delta^{2H}$ applied inside the exponential:

| Index, horizon | AR(5) | AR(10) | HAR(3) | RFSV |
|---|---|---|---|---|
| SPX, $\Delta = 1$ | 0.520 | 0.566 | 0.489 | **0.475** |
| SPX, $\Delta = 20$ | 1.070 | 1.010 | 1.036 | **0.903** |
| FTSE, $\Delta = 1$ | 0.612 | 0.621 | 0.582 | **0.567** |
| FTSE, $\Delta = 20$ | 1.046 | 0.984 | 0.935 | **0.874** |
| Nikkei, $\Delta = 1$ | 0.554 | 0.579 | **0.504** | 0.505 |
| Nikkei, $\Delta = 20$ | 1.097 | 1.046 | 1.011 | **0.964** |
| DAX, $\Delta = 1$ | 0.439 | 0.448 | 0.399 | **0.386** |
| DAX, $\Delta = 20$ | 0.931 | 0.850 | 0.816 | **0.746** |
| CAC40, $\Delta = 1$ | 0.533 | 0.542 | 0.470 | **0.465** |
| CAC40, $\Delta = 20$ | 0.982 | 0.952 | 0.912 | **0.828** |

Claim by claim. *RFSV beats AR and HAR* — supported: all 15 cells of Table 5.1 and 14 of 15 in Table 5.2, losing to HAR by $0.001$ on the Nikkei at one day. *Especially at longer horizons* — supported, and the clearest signal in the paper: at $\Delta=1$ the margin over HAR is $0.001$–$0.006$, at $\Delta=20$ it is $0.032$–$0.076$, and the ordering AR $<$ HAR $<$ RFSV is monotone in how much long-range structure each predictor encodes. *With one parameter* — true by construction, and that parameter does not depend on the horizon, whereas AR and HAR coefficients must be re-estimated per horizon. What is missing is any standard error: fifteen wins out of fifteen is hard to get by chance, but nothing quantifies the sampling variability of a $0.001$ gap and no Diebold-Mariano-style test appears. *Rough at any reasonable time scale* — supported down to one day only; below that it is extrapolation. *The microstructure explanation* — Section 6 sketches rather than demonstrates that Hawkes order flow with $\|\varphi\|_1 \approx 1$ and a power-law kernel has an integrated-fractional scaling limit with $H < 1/2$; both empirical regularities it invokes are cited from other papers and nothing is estimated here.

## 6 Limitations

**Stated by the authors.** Spot volatility is proxied, never observed, and whole-day realized variance biases $H$ upward. The forecasting section explicitly assumes volatility is perfectly observed. The concavity of $\zeta_q$ is called a finite-sample artefact — checked against simulated fBM, but still an assumption. Long memory in the asymptotic sense is declared unanswerable rather than answered: covariances at very large lags are never measurable, so the claim is only that the autocovariance is not a power law at observable scales. Option pricing is deferred; consistency with the volatility surface is argued through Fukasawa's asymptotics only.

**My reading.** The scaling evidence is visual regression on log-log plots with no confidence intervals, and there is no formal test against the obvious competitor: a smoother latent volatility observed with estimation noise, which would also inflate short-lag increments and mimic a small $H$. The simulation study checks the estimator's bias under the model's own assumptions but never under that alternative. The two-way split is a weak probe for time variation, and the crisis-half estimates are uniformly higher — if $H$ moves with the regime, a constant-$H$ forecast is misspecified exactly when forecasting matters. $\alpha$ and $m$ are never estimated; $\alpha$ is simply chosen small, and since the model on $[0,T]$ is designed to be insensitive to $\alpha$, the data cannot identify it. There is no leverage effect and no jumps — the anticorrelation between price and volatility innovations is promised for the sequel. Finally, the one-day gains are of order $0.003$ in a ratio near $0.3$; the headline is the 20-day column.

## 7 Extensions

**What was built on this.** The direct descendant is rough Bergomi (Bayer, Friz and Gatheral), which takes $H<1/2$ under the pricing measure and fits the surface; its calibration cost is exactly what [Deep Learning Volatility](/blog/deep-learning-volatility/) removes with a neural surrogate, and [Deep calibration](/blog/deep-calibration-rough-vol/) attacks from the inverse-map side. Rough Heston (El Euch and Rosenbaum) restores an affine structure and a characteristic function, built on the Hawkes limit theorems cited in Section 6. A later literature argues that measurement error alone can manufacture apparent roughness (from general knowledge, unverified). From the data-driven side, [Quant GANs](/blog/quant-gans/) and [SigCWGAN](/blog/conditional-sig-wgan/) target the same stylized facts, and [Neural SDEs](/blog/neural-sde-pricing-hedging/) parameterises the dynamics directly.

**Open problems.** Is $H$ constant across assets and across time? The two-half split says probably not, and offers no estimator with a standard error to settle it. What does the proxy do below the daily scale, where "rough at any reasonable time scale" is untested? How should $\alpha$ and $m$ be identified when the model is built to be insensitive to them? And what is the right joint law of price and volatility innovations, absent here?

**Research directions.** *These are ideas, not results — none has been run.*

1. **Roughness as an evaluation metric for generative models.** Hypothesis: a learned path generator matched on returns reproduces fat tails and clustering while producing log-volatility with the wrong $H$. Data: Oxford-Man realized variance against sampled paths from a diffusion or GAN simulator. Baseline: the same generator scored only on marginals and autocorrelation. Metric: the double regression of Section 3.2 on generated paths — $\hat H$, linearity of $\zeta_q$ in $q$, straightness of (5) in $\Delta^{2H}$. Failure mode: the estimator's own upward bias swamps the comparison unless generated paths go through the identical proxy construction, so the pipeline and not the latent path must be compared.
2. **Identification test against smooth-plus-noise.** Hypothesis: monofractal scaling with $H \approx 0.1$ cannot be produced by Heston-type volatility seen through realized-variance estimation error. Data: simulated Heston paths pushed through the same 5-minute estimator over the same 3,540 days. Baseline: the RFSV simulation of Section 3.4. Metric: the distribution of $\hat H$ over repetitions, and whether $\zeta_q$ stays linear in $q$ or bends. Failure mode: the answer depends entirely on the assumed microstructure noise, so the experiment bounds the alternative rather than excluding it.
3. **Rough-consistent conditioning for a diffusion forecaster.** Hypothesis: feeding a score-based time-series model the kernel of (6) as an explicit conditioning feature helps multi-horizon variance forecasts more than lengthening its context does. Data: the five indices of Table 5.1. Baselines: HAR, RFSV alone, and an unconditioned forecaster such as [TSDiff](/blog/tsdiff/). Metric: the ratio $P$ at $\Delta \in \{1,5,20\}$ plus predictive-interval calibration. Failure mode: the feature is nearly linear in the recent path, so a long-context model learns it anyway and the gain vanishes.

## 8 Takeaways

- The method is the contribution as much as the model: two nested log-log regressions, no model assumed, one exponent that holds across moment orders, assets and asset classes.
- Log-volatility increments show monofractal scaling with $H$ roughly $0.06$–$0.2$ and are close to Gaussian; after correcting for proxy smoothing the true exponent is nearer $0.1$. Volatility paths are far rougher than Brownian motion.
- RFSV is a fractional Ornstein-Uhlenbeck log-volatility with $H<1/2$ and reversion so slow that on any practical window it is fBM. The stationarity is a formality; $\alpha$ is not identifiable.
- Long memory in the power-law sense is not supported. Two standard procedures report it on simulated RFSV paths that provably lack it, returning the same parameters as on real data — which explains how the stylized fact became one.
- A one-parameter kernel forecast from fBM theory beats AR and HAR on five indices, decisively at 20 days and marginally at one, with no significance testing.
- For generative modelling of financial series this is a cheap, concrete target: the scaling law (2) and the $\Delta^{2H}$ autocovariance (5) are two numbers any simulator can be scored on, and low-dimensional Markovian volatility SDEs with Brownian noise cannot reproduce them — a sharp test of whether a learned model captured volatility dynamics or only its marginal.

## References

1. J. Gatheral, T. Jaisson, M. Rosenbaum. *Volatility is rough.* arXiv:1410.3394, 2014.
2. F. Comte, E. Renault. *Long memory in continuous-time stochastic volatility models.* Mathematical Finance, 1998.
3. P. Cheridito, H. Kawaguchi, M. Maejima. *Fractional Ornstein-Uhlenbeck processes.* Electronic Journal of Probability, 2003.
4. M. Fukasawa. *Asymptotic analysis for stochastic volatility: martingale expansion.* Finance and Stochastics, 2011.
5. C. J. Nuzman, H. V. Poor. *Linear estimation of self-similar processes via Lamperti's transformation.* Journal of Applied Probability, 2000.
6. F. Corsi. *A simple approximate long-memory model of realized volatility.* Journal of Financial Econometrics, 2009.
7. T. Jaisson, M. Rosenbaum. *Limit theorems for nearly unstable Hawkes processes.* Annals of Applied Probability, to appear (as cited in the paper).
