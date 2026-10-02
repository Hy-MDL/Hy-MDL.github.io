---
title: "NCSNv2: Improved Techniques for Training Score-Based Generative Models"
paper:
  title: "Improved Techniques for Training Score-Based Generative Models"
  authors: "Yang Song, Stefano Ermon"
  venue: "NeurIPS 2020"
  arxiv: "2006.09011"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "score-to-flow"
order: 2
tags: [score-based-models, langevin-dynamics, noise-schedule, concentration-of-measure, ema, tweedie, ncsn]
date: 2020-06-01
status: draft
summary: "Five rules, each derived on a solvable surrogate, that replace NCSN's hand-picked noise ladder and sampler settings with quantities computed from the data — and take score-based generation from 32x32 to 256x256."
---

## Abstract

The first noise-conditional score network worked at $32\times32$ and fell apart above it. This follow-up asks which of its hand-set constants were responsible and replaces each with a number computable from the dataset. Two surrogates carry the argument: the empirical distribution smoothed by $\sigma_1$, which fixes how large the first noise level must be for the sampler to move between training points at all; and a single isotropic Gaussian in $\mathbb{R}^D$, which — because a high-dimensional Gaussian is a thin spherical shell — fixes how closely the levels must be spaced, forces the ladder to be geometric rather than assuming it, and yields the Langevin step size in closed form. Two engineering changes complete the set: conditioning on noise by dividing an unconditional network's output by $\sigma$, and sampling from an exponential moving average of the weights. NCSNv2 lowers CIFAR-10 FID from 25.32 to 10.87 and produces coherent samples up to FFHQ $256\times256$ with no per-dataset tuning. The gains, however, depend on an added Tweedie denoising step, and the high-resolution results — the actual selling point — carry no quantitative metric.

**Keywords:** score-based generative models, annealed Langevin dynamics, noise schedule, concentration of measure, geometric progression, exponential moving average, Tweedie's formula, HYPE

## 1 Introduction

[NCSN](/blog/ncsn/) trains one network on $L$ Gaussian-perturbed copies of the data and samples by running Langevin dynamics down the ladder. Its constants — $\sigma_1=1$, $L=10$, $T=100$ steps per level, $\epsilon=2\times10^{-5}$ — were found by hand on $32\times32$ images, and the paper offered no procedure for changing them. What happens when you do not change them is documented here: at CelebA $64\times64$ the samples keep a consistent global structure but acquire obvious colour artifacts, and <mark>beyond $96\times96$ NCSN fails completely, in both structure and colour</mark>.

Four design choices are named as culprits: (i) the set of noise scales $\{\sigma_i\}$, (ii) how $\sigma$ enters the network, (iii) the step-size parameter $\epsilon$, and (iv) the number of Langevin steps $T$ per level. The strategy is uniform throughout — replace the data distribution by something simple enough to solve exactly, solve it, transfer the answer — and it is also the paper's main weakness: every rule is only as good as its surrogate.

## 2 Background

The score of a density is $\nabla_x\log p(x)$. In this paper's convention Langevin dynamics iterates
$$
x_t = x_{t-1} + \alpha\,\nabla_x \log p(x_{t-1}) + \sqrt{2\alpha}\,z_t,\qquad z_t\sim\mathcal N(0,I),
\tag{1}
$$
with step size $\alpha$. This is *not* NCSN's convention, which writes drift $\epsilon/2$ and noise $\sqrt\epsilon$; the two $\epsilon$ symbols differ by a factor of two, and the $\epsilon$ values in this paper's tables belong to Eq. (1). Given $\sigma_1>\dots>\sigma_L$ and perturbed densities $p_{\sigma_i}$, the NCSN objective is
$$
\frac{1}{2L}\sum_{i=1}^{L}\mathbb E_{p_{\text{data}}(x)}\;\mathbb E_{\tilde x\sim\mathcal N(x,\sigma_i^2 I)}\Big\lVert \sigma_i\, s_\theta(\tilde x,\sigma_i)+\frac{\tilde x-x}{\sigma_i}\Big\rVert_2^2 ,
\tag{2}
$$
already written with the $\lambda(\sigma)=\sigma^2$ weighting folded in, and annealed Langevin dynamics runs $T$ steps of Eq. (1) at each level with $\alpha_i=\epsilon\,\sigma_i^2/\sigma_L^2$, passing the last iterate on. See the [NCSN note](/blog/ncsn/) for where Eq. (2) comes from and why Gaussian perturbation is needed in the first place.

**The denoising step.** After this work's first public release, Jolicoeur-Martineau et al. observed — as had Saremi & Hyvärinen, Li et al. and Kadkhodaie & Simoncelli in other settings — that returning
$$
x_{\text{out}} = x_T + \sigma_L^2\, s_\theta(x_T,\sigma_L)
\tag{3}
$$
instead of $x_T$ improves FID substantially without changing how the samples look. By Tweedie's formula, for $y=x+\sigma\varepsilon$ the posterior mean satisfies $\mathbb E[x\mid y]=y+\sigma^2\nabla_y\log p_\sigma(y)$, so Eq. (3) is exactly the posterior-mean estimate of the clean image and strips the residual $\mathcal N(0,\sigma_L^2I)$ that the sampler necessarily carries. The paper's main results are reported with it and its appendix keeps the originals without it — a distinction that turns out to matter enormously. (The paper's Algorithm 1 writes this as $x_T+\sigma_T^2 s_\theta(x_T,\sigma_T)$; $T$ is the step count, not a level index, so read $\sigma_L$.)

## 3 Method

> **Key idea.** Every hand-tuned constant in NCSN is replaced by a closed-form calculation on a tractable surrogate: the empirical distribution smoothed by $\sigma_1$ fixes the top of the ladder, and a single isotropic Gaussian in $\mathbb{R}^D$ fixes the spacing of the levels and the sampler. The resulting rules need only the data dimension, pairwise distances, and a compute budget.

### 3.1 Technique 1 — the largest noise scale

Approximate $p_{\text{data}}$ by its empirical distribution on $\{x^{(1)},\dots,x^{(N)}\}$. Then the perturbed density at the top of the ladder is an $N$-component mixture, $\hat p_{\sigma_1}(x)=\frac1N\sum_i p^{(i)}(x)$ with $p^{(i)}=\mathcal N(x\mid x^{(i)},\sigma_1^2I)$, and differentiating the log of that sum gives a responsibility-weighted average of component scores,
$$
\nabla_x\log \hat p_{\sigma_1}(x)=\sum_{i=1}^{N} r^{(i)}(x)\,\nabla_x\log p^{(i)}(x),\qquad r^{(i)}(x)=\frac{p^{(i)}(x)}{\sum_k p^{(k)}(x)} .
\tag{4}
$$
Every term is exact so far. Now ask what a chain currently sitting in component $i$ knows about component $j$: it knows only $\mathbb E_{p^{(i)}}[r^{(j)}]$, because that is the weight component $j$ carries in the drift the chain feels. The bound is
$$
\mathbb E_{p^{(i)}(x)}\big[r^{(j)}(x)\big]\;\le\;\frac12\exp\!\Big(-\frac{\lVert x^{(i)}-x^{(j)}\rVert_2^2}{8\sigma_1^2}\Big).
\tag{5}
$$
The proof drops all components but $i$ and $j$ from the denominator, uses the geometric-mean/harmonic-mean inequality to replace $2/(1/p^{(i)}+1/p^{(j)})$ by $\sqrt{p^{(i)}p^{(j)}}$, and integrates the resulting Gaussian — a genuine upper bound, though a loose one for $N\gg2$. The shape is what matters: the weight decays like $\exp(-d^2/8\sigma_1^2)$ in the distance $d$. If $\sigma_1\ll d$, component $j$ is invisible and <mark>Langevin dynamics behaves as though the rest of the dataset did not exist</mark> — the diversity failure, quantified.

Hence **Technique 1: set $\sigma_1$ to the maximum Euclidean distance between pairs of training points.** On CIFAR-10 the median pairwise distance is about 18, so NCSN's $\sigma_1=1$ makes the bound smaller than $10^{-17}$; the rule gives $\sigma_1=50$. The controlled test removes the network entirely — annealed Langevin with *exact* scores for a 10,000-component mixture centred on CIFAR-10 test images gives average pairwise sample distance 10.12 at $\sigma_1=1$, against 18.65 at $\sigma_1=50$ and 17.78 for the data ([Fig. 2 in the paper](https://arxiv.org/pdf/2006.09011#page=4)). Isolating the sampler this way is what makes Technique 1 convincing where the others are merely plausible.

### 3.2 Technique 2 — spacing and number of levels

Now take a dataset of one point, so $p_{\sigma_i}=\mathcal N(0,\sigma_i^2I)$ in $\mathbb R^D$. In hyperspherical coordinates $p_{\sigma_i}(x)=p(\varphi)\,p_{\sigma_i}(r)$, and the angular part $p(\varphi)$ is uniform and identical at every level — so levels differ only in their radial law. That law concentrates: with $r=\lVert x\rVert_2$,
$$
p(r)=\frac{1}{2^{D/2-1}\Gamma(D/2)}\frac{r^{D-1}}{\sigma^{D}}\exp\!\Big(\!-\frac{r^2}{2\sigma^2}\Big),
\qquad r-\sqrt D\sigma \;\xrightarrow{d}\; \mathcal N(0,\sigma^2/2)\ \ (D\to\infty).
\tag{6}
$$
The limit follows from the CLT applied to $\lVert x\rVert_2^2/D$ plus the delta method. For image dimensions it is accurate enough to treat each $p_{\sigma_i}(r)$ as $\mathcal N(\sqrt D\sigma_i,\;\sigma_i^2/2)$ outright: <mark>a high-dimensional Gaussian is not a ball but a thin shell of radius $\sqrt D\sigma$ and thickness $\sigma/\sqrt2$</mark>, whose relative thickness $1/\sqrt{2D}$ is about 1.3% at $D=3072$.

Two adjacent levels must therefore have overlapping shells, or the sampler arrives at level $i$ carrying points that level $i$'s score network never saw. Requiring $p_{\sigma_i}$ to place a fixed probability $C$ on the three-sigma radial interval of $p_{\sigma_{i-1}}$ gives, with $\gamma_i=\sigma_{i-1}/\sigma_i$,
$$
\Phi\big(\sqrt{2D}(\gamma_i-1)+3\gamma_i\big)-\Phi\big(\sqrt{2D}(\gamma_i-1)-3\gamma_i\big)=C ,
\tag{7}
$$
$\Phi$ the standard normal CDF. Because the left side depends on $i$ only through $\gamma_i$, fixing $C$ forces $\gamma_2=\dots=\gamma_L$: <mark>the geometric progression falls out of the analysis instead of being assumed</mark>, which is the most satisfying step in the paper. The authors take $C\approx0.5$ as the compromise between quality ($C\to1$) and cost, giving $L=232$ on CIFAR-10 and $L=2311$ on FFHQ $256^2$.

Eq. (7) also has a legible asymptotic. With $\gamma\approx1$ the second term is $\Phi(u-3)$ and the first is essentially 1, where $u=\sqrt{2D}(\gamma-1)$; solving $1-\Phi(u-3)=C$ at $C\approx0.5$ needs $u\approx3$, so
$$
\gamma \approx 1+\frac{3}{\sqrt{2D}},\qquad L \approx 1+\frac{\sqrt{2D}}{3}\,\log\frac{\sigma_1}{\sigma_L}.
\tag{8}
$$
So the number of levels grows like $\sqrt D$, times a log factor that itself grows because Technique 1 makes $\sigma_1$ scale with the data diameter. That is the paper's central structural claim in one line, and it predicts the reported numbers well: going from CIFAR-10 ($D=3072$) to FFHQ ($D=196{,}608$) multiplies $\sqrt D$ by 8 and $\log(\sigma_1/\sigma_L)$ by 1.23, predicting $L\approx232\times8\times1.23\approx2280$ against the reported 2311. (Eq. (8) and this check are mine, not the paper's.)

### 3.3 Technique 3 — noise conditioning

NCSN carried a separate set of normalization scales and biases per noise level, so conditioning memory grew linearly in $L$ — survivable at $L=10$, absurd at $L=2311$, and impossible if the network has no normalization layers. The one-point surrogate already says what the conditioning should do: for $p_\sigma=\mathcal N(0,\sigma^2I)$ the score is $-x/\sigma^2$ and $\mathbb E\lVert\nabla_x\log p_\sigma(x)\rVert_2\approx\sqrt D/\sigma$, matching the $1/\sigma$ norm scaling observed empirically in trained networks. So set
$$
s_\theta(x,\sigma)=s_\theta(x)/\sigma
\tag{9}
$$
with $s_\theta$ unconditional. The paper's justification for not letting the network learn this is practical: $\sigma_1$ and $\sigma_L$ differ by three to four orders of magnitude, which is hard for a network to produce by itself. Training loss curves match the original conditioning ([Fig. 3 in the paper](https://arxiv.org/pdf/2006.09011#page=5)) — though note the comparison is between two models that both use Techniques 1, 2, 4 and 5, so it isolates the conditioning cleanly but says nothing about $L=10$. The form extends to a continuum of $\sigma$, which is what makes [Score-SDE](/blog/score-sde/) possible, and [EDM](/blog/edm/) later generalizes it into a full preconditioning scheme.

### 3.4 Technique 4 — step size and steps per level

Same one-point surrogate. At level $i$ the score is $-x/\sigma_i^2$, so Eq. (1) is a linear recursion,
$$
x_t = \Big(1-\frac{\alpha}{\sigma_i^2}\Big)x_{t-1}+\sqrt{2\alpha}\,z_t
\quad\Longrightarrow\quad
\operatorname{Var}[x_t]=\Big(1-\frac{\alpha}{\sigma_i^2}\Big)^{2}\operatorname{Var}[x_{t-1}]+2\alpha I .
\tag{10}
$$
The iterate stays Gaussian and centred, so only the variance matters. Subtracting the fixed point $v=2\alpha/(1-(1-\alpha/\sigma_i^2)^2)$ makes the recursion homogeneous, and with $\operatorname{Var}[x_0]=\sigma_{i-1}^2I$ — the chain starts from the previous level — $T$ steps give $x_T\sim\mathcal N(0,s_T^2I)$ with
$$
\frac{s_T^2}{\sigma_i^2}=\Big(1-\frac{\epsilon}{\sigma_L^2}\Big)^{2T}\Big(\gamma^2-b\Big)+b,
\qquad
b=\frac{2\epsilon/\sigma_L^2}{1-\big(1-\epsilon/\sigma_L^2\big)^{2}} ,
\tag{11}
$$
after substituting $\alpha=\epsilon\sigma_i^2/\sigma_L^2$, which makes $\alpha/\sigma_i^2=\epsilon/\sigma_L^2$ and so removes $i$ from the contraction factor entirely. Every step here is exact for the surrogate. Two consequences: with a geometric ladder the ratio is <mark>the same at every level and carries no explicit dependence on $D$</mark>, and $s_T^2/\sigma_i^2=1$ is exactly the statement that $T$ steps were enough to forget the previous level and equilibrate at this one.

Hence **Technique 4: fix $T$ from the compute budget (so $T\cdot L$ is a few thousand), then grid-search $\epsilon$ to bring Eq. (11) closest to 1.** Grid search rather than gradient descent, because the objective is badly behaved near $\epsilon=2\sigma_L^2$, where the contraction factor $|1-\epsilon/\sigma_L^2|$ reaches 1 and the chain stops converging at all.

### 3.5 Technique 5 — exponential moving average

The training loss falls smoothly, but FID measured during training swings widely, and samples from one checkpoint share a colour cast that drifts between checkpoints. Keeping a shadow copy $\theta'\leftarrow m\theta'+(1-m)\theta_i$ with $m=0.999$ and sampling from it <mark>stabilizes FID and removes the colour shift</mark> ([Fig. 4 in the paper](https://arxiv.org/pdf/2006.09011#page=7)). The one technique with no derivation — pure observation — and the one every subsequent diffusion paper adopted without discussion.

### 3.6 Do the published schedules actually satisfy the rules?

The rules are cheap enough to check, and checking them is a good way to see what they mean. Taking $\sigma_L=0.01$ (retained from NCSN) and the $\sigma_1,L,T,\epsilon$ of the paper's Table 4, I evaluated Eq. (7) and Eq. (11) directly. These numbers are my computation, not the paper's.

| Schedule | $\gamma$ | $C$ from Eq. (7) | $s_T^2/\sigma_i^2$ from Eq. (11) | $L\cdot T$ |
|---|---|---|---|---|
| NCSN, CIFAR-10 $32^2$ | 1.6681 | $0$ (to machine precision) | 1.111 | 1,000 |
| NCSNv2, CIFAR-10 $32^2$ | 1.0376 | 0.567 | 1.056 | 1,160 |
| NCSNv2, CelebA $64^2$ | 1.0184 | 0.567 | 1.031 | 2,500 |
| NCSNv2, LSUN church $96^2$ | 1.0122 | 0.566 | 1.025 | 3,152 |
| NCSNv2, LSUN bedroom/tower $128^2$ | 1.0091 | 0.567 | 1.017 | 3,258 |
| NCSNv2, FFHQ $256^2$ | 1.0045 | 0.567 | 1.009 | 6,933 |

Three readings. All five NCSNv2 ladders land on $C=0.566$–$0.567$, so Technique 2 was applied uniformly and "$C\approx0.5$" in practice means 0.57. The original NCSN ladder gives $C=0$ to double precision — adjacent shells with literally no overlap, the concrete form of "NCSN fails above $32^2$". Less flatteringly, the original ladder's $s_T^2/\sigma_i^2=1.111$ is not catastrophic: NCSNv2's 1.009–1.056 is better, but the gain from Technique 4 alone is modest. Technique 2 is the decisive change; Technique 4 is what makes it affordable, by cutting $T$ from 100 to 3–5.

### 3.7 Algorithm

```text
CONFIGURE  (all four numbers computed, none tuned)
  sigma_1 = max_{i,j} ||x_i - x_j||_2          # subsample 10k points if N > 60k
  sigma_L = 0.01
  solve  Phi(sqrt(2D)(g-1)+3g) - Phi(sqrt(2D)(g-1)-3g) = 0.5   for g
  L       = 1 + round( log(sigma_1/sigma_L) / log(g) )
  T       = budget / L                          # T*L a few thousand
  eps     = argmin_eps | Eq.(11)(eps, T, g) - 1 |      # grid search

TRAIN
  repeat:
      x ~ data;  i ~ Uniform{1..L};  z ~ N(0,I)
      xt   = x + sigma_i * z
      loss = 0.5 * || s(xt) + z ||^2             # Technique 3: sigma_i cancels out of Eq.(2)
      theta  <- Adam step
      theta' <- 0.999*theta' + 0.001*theta       # EMA shadow copy

SAMPLE  (annealed Langevin, EMA weights, with denoising)
  x <- Uniform([0,1]^D)
  for i = 1..L:
      a = eps * sigma_i^2 / sigma_L^2
      repeat T times:
          z ~ N(0,I)
          x <- x + a * s'(x)/sigma_i + sqrt(2a) * z      # note: a, not a/2
  return x + sigma_L * s'(x)                             # Tweedie denoise
```

Note what the training loop lost: under Technique 3 the objective is $\tfrac12\lVert s_\theta(\tilde x)+z\rVert^2$ with no $\sigma$ in it at all. The network is a plain noise predictor and the entire schedule lives in the data pipeline and the sampler.

## 4 Implementation notes

**Schedules produced by the rules (Table 4 of the paper).**

| Model | Dataset | $\sigma_1$ | $L$ | $T$ | $\epsilon$ | Batch | Iterations |
|---|---|---|---|---|---|---|---|
| NCSN | CIFAR-10 $32^2$ | 1 | 10 | 100 | 2e-5 | 128 | 300k |
| NCSN | CelebA $64^2$ | 1 | 10 | 100 | 2e-5 | 128 | 210k |
| NCSN | LSUN church $96^2$ | 1 | 10 | 100 | 2e-5 | 128 | 200k |
| NCSN | LSUN bedroom $128^2$ | 1 | 10 | 100 | 2e-5 | 64 | 150k |
| **NCSNv2** | CIFAR-10 $32^2$ | 50 | 232 | 5 | 6.2e-6 | 128 | 300k |
| **NCSNv2** | CelebA $64^2$ | 90 | 500 | 5 | 3.3e-6 | 128 | 210k |
| **NCSNv2** | LSUN church $96^2$ | 140 | 788 | 4 | 4.9e-6 | 128 | 200k |
| **NCSNv2** | LSUN bedroom/tower $128^2$ | 190 | 1086 | 3 | 1.8e-6 | 128 | 150k |
| **NCSNv2** | FFHQ $256^2$ | 348 | 2311 | 3 | 0.9e-7 | 32 | 80k |

| Item | As reported |
|---|---|
| Backbone | RefineNet, as in NCSN, with three changes for v2: CondInstanceNorm++ reduced to one class ("InstanceNorm++"), average pooling reverted to max pooling, normalization removed inside RefineBlocks |
| Depth / width | $32^2$–$64^2$: 4 RefineNet cascades, 128→256 filters; $96^2$–$128^2$: 5 cascades, 128→512; $256^2$: 6 cascades, 128→512. Dilation 2 then 4 in the last two residual stages. ELU throughout |
| Optimizer | Adam; lr $10^{-3}$ without Technique 3, $10^{-4}$ with it (otherwise the loss explodes); Adam's own $\varepsilon$ = $10^{-3}$ for FFHQ, $10^{-8}$ elsewhere |
| $\sigma_1$ estimation | exact max pairwise distance, or the max over a random subsample of 10,000 points when $N>60{,}000$ |
| EMA momentum | $m=0.999$ (a separate $m=0.9$ is used only to smooth the plotted loss curves in Fig. 3) |
| Model selection | FID on 1,000 samples every 5,000 iterations; lowest-FID checkpoint used for full FID, Inception and HYPE |
| FID protocol | CIFAR-10: samples against training statistics. CelebA: 10,000 samples against the test set |
| Preprocessing | CelebA center crop $140^2$ then resize $64^2$; LSUN resize shortest side then center crop; FFHQ resized $1024^2\to256^2$ with a 63k/7k split; random horizontal flip everywhere |
| Compute | V100s; CIFAR-10 22 h on 2 GPUs, CelebA 29 h on 4, LSUN 52 h on 8, FFHQ 41 h on 8 |
| Sampling cost | 2 min (CIFAR-10) to 50 min (FFHQ) per batch on the same hardware |
| Stability | training at $128^2$ and $256^2$ reported unstable near convergence; attributed to Adam, with AMSGrad suggested but not used |
| Interpolation | interpolate the *injected Langevin noise* $\{z_{ij}\}$ spherically, $\cos\phi\,z^{(1)}+\sin\phi\,z^{(2)}$, from a shared $x_0$ — not a latent-space walk |
| Not stated | FID or any metric for LSUN and FFHQ; number of samples behind the full CIFAR-10 FID; sensitivity to $C$ |

**Easy to get wrong when reproducing.**
1. **Which Langevin convention.** $\epsilon$ in Table 4 belongs to Eq. (1) (drift $\alpha$, noise $\sqrt{2\alpha}$). Plugging it into NCSN's $\epsilon/2$–$\sqrt\epsilon$ form halves every step.
2. **The denoising step is not cosmetic.** Table 1 below shows it moving CIFAR-10 FID from 31.75 to 10.87 on identical samples. Report FID with and without it or the number means nothing.
3. **$\sigma_1$ from the maximum, not the median.** With the CIFAR-10 median (≈18) instead of the max (50) the ladder is a third too short at the top; with an outlier-contaminated dataset the max may be far too large.
4. **Lower the learning rate with Technique 3.** $10^{-3}$ is reported to blow up once the $1/\sigma$ rescaling is in the forward pass, because the effective output scale at small $\sigma$ is now $100\times$ larger.
5. **$\epsilon$ must stay below $2\sigma_L^2$.** Outside that range the contraction factor in Eq. (11) has modulus $\ge1$ and the inner chain diverges; the grid search must be bounded accordingly.

## 5 Experiments

**Setup.** CIFAR-10 $32^2$, CelebA $64^2$, LSUN church_outdoor $96^2$, LSUN bedroom and tower $128^2$, FFHQ $256^2$. Only the first two are evaluated quantitatively.

**Main results (Table 1 of the paper).**

| Model | CIFAR-10 Inception ↑ | CIFAR-10 FID ↓ | CelebA $64^2$ FID ↓ |
|---|---|---|---|
| PixelCNN | 4.60 | 65.93 | – |
| IGEBM | 6.02 | 40.58 | – |
| WGAN-GP | 7.86 ± .07 | 36.4 | – |
| SNGAN | 8.22 ± .05 | 21.7 | – |
| NCSN (w/o denoising) | **8.87 ± .12** | 25.32 | 26.89 |
| NCSN (w/ denoising) | 7.32 ± .12 | 29.8 | 25.30 |
| NCSNv2 (w/o denoising) | 8.73 ± .13 | 31.75 | 28.86 |
| **NCSNv2 (w/ denoising)** | 8.40 ± .07 | **10.87** | **10.23** |

**Human evaluation (Table 5, HYPE$_\infty$ on CelebA $64^2$, higher is better).**

| Model | HYPE$_\infty$ (%) | Fakes taken for real (%) | Reals taken for fake (%) |
|---|---|---|---|
| StyleGAN (truncated) | 50.7 | 62.2 | 39.3 |
| ProgressiveGAN | 40.3 | 46.2 | 34.4 |
| **NCSNv2** | **37.3** | 49.8 | 24.8 |
| NCSN | 19.8 | 22.3 | 17.3 |
| BEGAN | 10.0 | 6.2 | 13.8 |
| WGAN-GP | 3.8 | 1.7 | 5.9 |

**Claim-by-claim.**

- *"NCSNv2 substantially improves FID."* Supported — but only with denoising. <mark>Without it NCSNv2 is worse than NCSN on both datasets (31.75 vs 25.32; 28.86 vs 26.89) and never beats the original Inception score of 8.87</mark>. Two changes are confounded in the headline number.
- *"FID misranks these models."* The authors' defence, well made: HYPE$_\infty$ reverses the without-denoising ranking, 37.3 against 19.8. The breakdown is informative — NCSNv2's fakes fool raters 49.8% of the time against NCSN's 22.3%, but its *real* images are misjudged more often too (24.8% vs 17.3%), which hints the two arms were not seen under identical conditions. One study, one dataset, one resolution, a third-party service.
- *"Technique 1 governs diversity."* The strongest evidence here, because the exact-score mixture experiment removes the network entirely (10.12 vs 18.65, against 17.78 for data). The caveat is that the surrogate *is* the model in that test, so it validates the analysis rather than the trained system.
- *"Geometric spacing is derived, not assumed."* Genuinely forced by the structure of Eq. (7). But $C\approx0.5$ and the three-sigma interval are choices with no ablation — nothing shows what happens at $C=0.2$ or $0.9$, the one knob a practitioner would reach for.
- *"Technique 3 matches the original conditioning."* Fig. 3 shows matching training losses and better samples by inspection. Since Eq. (9) is a reparameterization of the same objective, matching losses is close to the minimum one could ask of it.
- *"All five techniques together are best."* The weakest claim. Groups $\{5\}$, $\{1,2,4\}$, $\{3\}$ are removed successively; each improves on vanilla NCSN, but FID is <mark>not monotone in the number of techniques</mark> and the combination is defended by visual inspection ([Fig. 5 in the paper](https://arxiv.org/pdf/2006.09011#page=7)). Consistent with the paper's own position on FID — and still an argument settled by eye.
- *"Scales to $256\times256$."* Samples and interpolations at $96^2$–$256^2$ where NCSN produces nothing recognizable, plus nearest-neighbour panels and train/test curves against memorization. All qualitative: <mark>the headline result has no number attached to it</mark>.

## 6 Limitations

**Stated by the authors.**
- FID and Inception have known issues and can be sensitive to imperceptible noise; the paper says its own numbers should be read with caution.
- The FID ablation is not monotone in the number of techniques; the combination is defended visually.
- Training near convergence is unstable at $128^2$ and $256^2$, attributed to Adam.
- Non-Gaussian perturbation kernels are named as future work, as is a theory of sample quality for score-based models.

**My reading.**
- The surrogates are crude in one specific direction: a one-point dataset has no estimation error, so Techniques 2 and 4 say nothing about how the *network's* error interacts with step size or spacing — precisely the regime where a real sampler fails. Technique 1's surrogate at least has $N$ components, which is why it is the best-supported rule.
- $C\approx0.5$ and the three-sigma interval are free parameters smuggled in to eliminate two others. The method is less tuning-free than advertised; it has moved the tuning somewhere dimension-independent, which is better but not the same thing.
- Quantitative evidence exists only for $32^2$ and $64^2$. The datasets that justify the title carry no metric.
- Technique 1 rests on a *maximum* over pairs, the least robust statistic available, and the concession to that — subsampling 10,000 points when $N>60{,}000$ — shrinks the max by chance rather than by design, making $\sigma_1$ depend on $N$ in a way no rule describes.
- Sampling still costs 1,160 network evaluations on CIFAR-10 and 6,933 on FFHQ, *more* than NCSN's 1,000. This paper makes high-resolution generation work; it does not make it cheap. That goes to [DDIM](/blog/ddim/) and [EDM](/blog/edm/).
- No likelihoods, and still no use of the training loss to compare models — the capability NCSN advertised and neither paper exercises.

## 7 Extensions

**What was built on this.** [Score-SDE](/blog/score-sde/) is the direct successor: Technique 2's $L\to\infty$ limit is the Variance Exploding SDE, Technique 3's $s_\theta(x)/\sigma$ is what makes a continuum of noise levels trainable, the predictor-corrector sampler generalizes annealed Langevin, and a probability-flow ODE supplies the likelihoods missing here. [EDM](/blog/edm/) revisits all five choices as design axes, replacing Eq. (9) with a general preconditioner and $C$ with a tunable $\sigma$ distribution; [Improved DDPM](/blog/improved-ddpm/) makes the parallel argument on the [DDPM](/blog/ddpm/) side that the schedule, not the architecture, is the lever. EMA and Tweedie denoising became defaults everywhere, including [ADM](/blog/diffusion-beats-gans/) and [LDM](/blog/latent-diffusion/) — though LDM's answer to $L\propto\sqrt D$ is simply to shrink $D$.

**Open problems the paper leaves.**
- How should $C$ be chosen, and is there a quality/compute frontier in $C$ rather than a single recommended value?
- What is the right analogue of Techniques 1 and 2 for a non-Gaussian kernel, where "shell thickness" and "pairwise distance" may not be the governing quantities?
- Can $\sigma_1$ be set robustly? A maximum over pairs is a worst case; nothing in the analysis requires the *maximum* rather than a high quantile of the distance distribution.
- Is there any bound connecting score-estimation error per level to the final sample distribution? Every rule here assumes the score is exact.
- Why does EMA help? No mechanism is offered for the colour-shift artifact it removes.

**Research directions.** *These are ideas, not results — none has been run.*

1. **A robust Technique 1 for heavy-tailed data.** *Hypothesis:* on a return panel the maximum pairwise distance is set by two or three crisis days, so Technique 1 gives a $\sigma_1$ far larger than the bulk geometry needs, inflating $L$ (which grows with $\log\sigma_1$ by Eq. (8)) without improving coverage; a 99th percentile of pairwise distances should match sample diversity at materially smaller $L$. *Data:* daily returns of a fixed equity universe, $D$ in the low hundreds, spanning at least one crisis. *Baseline:* Technique 1 verbatim. *Metric:* average pairwise distance of samples against data (the paper's own diversity measure), $L$ at fixed $C$, and tail coverage on a held-out crisis period. *Likely failure mode:* the quantile version under-covers exactly the crisis days one wants generated — in which case those extra levels are the price of the tail, and that trade-off is the result.
2. **Does the shell argument survive a factor structure?** *Hypothesis:* Technique 2 assumes $p_\sigma$ is isotropic, so the radial law concentrates at $\sqrt D\sigma$ with relative width $1/\sqrt{2D}$; for data with strong low-rank structure the perturbed density is an anisotropic ellipsoid, its radius concentrates less tightly, and the effective dimension in Eq. (8) should be the participation ratio $(\sum_k\lambda_k)^2/\sum_k\lambda_k^2$ rather than $D$ — cutting $L$ by the ratio of the two. *Data:* synthetic Gaussians with controlled eigenvalue decay (known answer), then a return panel whose decay is measured. *Baseline:* $L$ from Eq. (7) with ambient $D$. *Metric:* per-level radial overlap measured empirically, and sample quality at matched compute. *Likely failure mode:* the participation ratio summarizes only second moments, and heavy tails can break the shell argument at any $D$.
3. **Separating the denoising step from the techniques.** *Hypothesis:* the reported gain is two independent effects — Tweedie denoising removing $\sigma_L$-scale noise that FID punishes disproportionately, and the techniques improving the samples — and the second is smaller than the headline suggests. *Data:* LSUN church $96^2$, where NCSN is reported to fail outright. *Baseline:* NCSN + denoising versus NCSNv2 without denoising. *Metric:* FID plus a human or CLIP-based preference test, since the paper's own position is that FID misranks. *Likely failure mode:* NCSN's failure at $96^2$ is structural, making the comparison degenerate — which would itself settle the question in the paper's favour.

## 8 Takeaways

- Set the largest noise level to the scale of the data's diameter. Below that, Eq. (5) makes every distant training point's weight exponentially small and the sampler cannot move between them: diversity collapse, with a formula attached.
- In high dimension a Gaussian is a thin shell, so adjacent levels must overlap radially. That one fact forces geometric spacing, makes $L$ grow like $\sqrt D\log(\sigma_1/\sigma_L)$, and explains why NCSN's ten levels give zero overlap at $32^2$ — §3.6 makes it concrete.
- Conditioning on noise can be as simple as dividing by $\sigma$, which drops the memory cost of $L$ to zero and opens the door to continuous noise levels — the step [Score-SDE](/blog/score-sde/) needs.
- Use EMA weights for sampling and denoise the final iterate with Tweedie's formula. Both are now standard; the second moves FID so much that a reported FID which does not say whether it was applied is uninterpretable.
- Read the table with the denoising column covered: NCSNv2's raw samples score *worse* than NCSN's on FID and no better on Inception, and the high-resolution claim rests on pictures. The techniques are convincing; the quantitative case is thinner than the abstract implies.
- For financial time series the rules transfer in form, needing only $D$, pairwise distances and a budget — but Technique 1 is a maximum over pairs, which with heavy-tailed returns is set by a handful of crisis days. A quantile is the obvious substitute, and the shell argument's $D$ probably wants to be an effective dimension once returns have a factor structure. Both are my reading; the paper tests images only.

## References

1. Y. Song, S. Ermon. *Improved Techniques for Training Score-Based Generative Models.* NeurIPS 2020. arXiv:2006.09011.
2. Y. Song, S. Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution.* NeurIPS 2019. arXiv:1907.05600.
3. A. Jolicoeur-Martineau, R. Piché-Taillefer, I. Mitliagkas, R. Tachet des Combes. *Adversarial Score Matching and Improved Sampling for Image Generation.* arXiv:2009.05475, 2020.
4. B. Efron. *Tweedie's Formula and Selection Bias.* JASA 106(496):1602–1614, 2011.
5. S. Zhou et al. *HYPE: A Benchmark for Human Eye Perceptual Evaluation of Generative Models.* NeurIPS 2019.
6. G. Lin, A. Milan, C. Shen, I. Reid. *RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation.* CVPR 2017.
