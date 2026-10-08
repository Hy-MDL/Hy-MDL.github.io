---
title: "Normalizing Flows for Probabilistic Modeling and Inference"
paper:
  title: "Normalizing Flows for Probabilistic Modeling and Inference"
  authors: "George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, Balaji Lakshminarayanan"
  venue: "JMLR 22 (2021) 1-64"
  arxiv: "1912.02762"
  license: "creativecommons.org/licenses/by/4.0/"
series: "normalizing-flows"
order: 8
tags: [normalizing-flows, survey, universality, conditioner-transformer, residual-flows, topological-constraints, variational-inference, likelihood-free-inference]
date: 2019-12-01
status: draft
summary: "The unifying account: every autoregressive flow is a conditioner paired with a transformer, universality follows from the chain rule plus conditional CDFs, coupling layers are the K=2 end of a spectrum whose universality at sub-linear depth is an open problem, and a diffeomorphism cannot change the topology of its input."
---

## Abstract

This is the review that made the flow literature a field rather than a pile of architectures. Its organising move is a decomposition: an autoregressive flow is $z_i'=\tau(z_i;h_i)$ with $h_i=c_i(z_{<i})$, a strictly monotone *transformer* $\tau$ whose parameters are supplied by a *conditioner* $c_i$ — and these are independent choices, so affine/spline/integral transformers cross with recurrent/masked/coupling conditioners to generate every model in the literature. Around that sit four results worth knowing cold. Universality: the conditional-CDF map has a triangular Jacobian whose determinant *is* the density, so it pushes any positive density to a uniform on $(0,1)^D$ — autoregressive flows can represent anything, in principle. The coupling/autoregressive spectrum: split into $K$ parts, invert at $O(K)$ times the cost of evaluating, with $K=2$ and $K=D$ the familiar ends; a single coupling layer is *not* universal however expressive its network, $D$ of them are, and whether fewer than $O(D)$ suffice is open. The KL duality: forward KL in $x$-space equals reverse KL in $u$-space, where the flow-induced $p_u^*$ is fitted to the base, and vice versa, which is why maximum likelihood and variational inference are the same optimisation seen from two ends. And the topological constraint: a diffeomorphism preserves topology, so a flow whose base has one mode cannot produce a target with two without leaving mass in between.

**Keywords:** normalizing flows, conditioner and transformer, universal approximation, coupling layers, residual flows, topological constraints, likelihood-free inference

## 1 Introduction

The framing is statistical rather than generative: *the search for well-specified probabilistic models — models that correctly describe the processes that produce data — is one of the enduring ideals of the statistical sciences*, and flows are one tool for getting richer descriptions. The image the title comes from is deliberate: *pushing a simple density through a series of transformations to produce a richer, potentially more multi-modal distribution — like a fluid flowing through a set of tubes*.

The review dates from the moment the field needed one. By late 2019 there were coupling flows, autoregressive flows in both directions, planar and Sylvester flows, invertible ResNets, continuous-time flows, spline transformers, and a growing pile of one-off constructions, with no shared vocabulary and a lot of duplicated results. This paper supplies the vocabulary, and much of what people now say about flows — "conditioner", "transformer", "one-pass in which direction" — is said in these words because of this paper.

Two reasons it is the right place to stop reading architectures and start thinking. First, the decomposition is *generative of the literature*: once you see conditioner × transformer, [NICE](/blog/nice/), [Real NVP](/blog/realnvp/), Glow, [MAF](/blog/maf/), [IAF](/blog/iaf/) and [Neural Spline Flows](/blog/neural-spline-flows/) are six cells of a table, and the empty cells are visible. Second, §6 is a catalogue of uses that has nothing to do with images: importance sampling, MCMC, variational inference, likelihood-free inference. That section is the bridge from this series to the [sequential Monte Carlo](/blog/bootstrap-filter/) one, and it is where flows stop being a generative-modelling curiosity and become an inference tool.

## 2 Background: what a flow is, stated carefully

$x=T(u)$ with $u\sim p_u(u)$, $T$ a diffeomorphism — invertible, with $T$ and $T^{-1}$ both differentiable — which forces $u$ to be $D$-dimensional too. Then

$$
p_x(x)=p_u(u)\,\lvert\det J_T(u)\rvert^{-1}\ \ \text{with}\ u=T^{-1}(x),
\qquad\text{equivalently}\qquad
p_x(x)=p_u\bigl(T^{-1}(x)\bigr)\,\lvert\det J_{T^{-1}}(x)\rvert. \tag{1}
$$

The volume reading is the one to internalise: $\lvert\det J_T(u)\rvert\approx\mathrm{Vol}(dx)/\mathrm{Vol}(du)$, and since the probability mass in $dx$ equals that in $du$, expansion means lower density and contraction means higher.

A small terminological correction that is more than pedantry. The review declines to call $p_u$ a "prior" or $u$ a "latent variable": *upon observing $x$, the corresponding $u=T^{-1}(x)$ is uniquely determined and thus no longer 'latent'.* A flow is not a latent-variable model. There is no posterior over $u$, no inference gap, nothing to integrate out. That is precisely the property that makes a flow usable where a VAE or a GAN is not — and the reason the word "prior" keeps causing confusion when people compare flows to GANs.

Composition: $(T_2\circ T_1)^{-1}=T_1^{-1}\circ T_2^{-1}$ and $\det J_{T_2\circ T_1}(u)=\det J_{T_2}(T_1(u))\cdot\det J_{T_1}(u)$. Inverses reverse, log-determinants add. Everything else is engineering.

## 3 The four results

> **Key idea.** Almost every design question about flows reduces to one of four things: can it represent the target (universality), which direction is cheap (conditioner choice), what is being minimised (KL duality), and is the target even reachable (topology).

### 3.1 Universality, constructively

Assume $p_x(x)>0$ everywhere and that all conditional CDFs are differentiable. Factor by the chain rule, $p_x(x)=\prod_i p_x(x_i\mid x_{<i})$, and define

$$
z_i=F_i(x_i,x_{<i})=\int_{-\infty}^{x_i}p_x(x_i'\mid x_{<i})\,dx_i'=\Pr(x_i'\le x_i\mid x_{<i}). \tag{2}
$$

Each $F_i(\cdot,x_{<i}):\mathbb{R}\to(0,1)$ is invertible because its derivative $\partial F_i/\partial x_i=p_x(x_i\mid x_{<i})$ is positive. $z_i$ does not depend on $x_j$ for $j>i$, so $J_F$ is lower triangular and

$$
\det J_F(x)=\prod_{i=1}^{D}\frac{\partial F_i}{\partial x_i}=\prod_{i=1}^{D}p_x(x_i\mid x_{<i})=p_x(x). \tag{3}
$$

**The Jacobian determinant of this map is the density itself.** So by (1),

$$
p_z(z)=p_x(x)\lvert\det J_F(x)\rvert^{-1}=p_x(x)\,p_x(x)^{-1}=1, \tag{4}
$$

and $z$ is uniform on $(0,1)^D$. Build the analogous $G$ from $p_u$'s conditional CDFs and $T=F^{-1}\circ G$ turns any such $p_u$ into any such $p_x$.

This is the cleanest proof in the flow literature and it is four lines. It also does two jobs at once: it establishes universality, and it *explains why triangular Jacobians are the right structure to build flows out of* — they are not a computational convenience grafted on, they are what the universal construction has.

The caveat is immediate and the review states it: *this is just a statement of representational power and makes no guarantees about the flow's behavior in practice.* Universality at infinite width and depth says nothing about what a 10-layer flow trained on a few thousand points will do.

> **My comment.** This is why I would rather grade a learned prior against an exact answer than argue from universality. In the factor-selection setup the hyper-g/n posterior over all $2^{10}$ factor subsets is computable, so the flow can be scored by its total-variation distance from it (0.0159), which says something universality never could.

### 3.2 Conditioner and transformer

$$
z_i'=\tau(z_i;h_i),\qquad h_i=c_i(z_{<i}). \tag{5}
$$

$\tau$ is strictly monotone in $z_i$, hence invertible, and parameterised by $h_i$; $c_i$ is *any* function at all — it need not be a bijection, its only constraint being that it sees only indices below $i$. The Jacobian is lower triangular with the transformer derivatives on the diagonal, so

$$
\log\lvert\det J_{f_\phi}(z)\rvert=\sum_{i=1}^{D}\log\left\lvert\frac{\partial\tau}{\partial z_i}(z_i;h_i)\right\rvert, \tag{6}
$$

in $O(D)$, *and the lower-triangular part of the Jacobian is irrelevant* — a sentence worth remembering, because everything expensive in the model lives there and never enters the loss.

The asymmetry: forward, every $h_i$ and $z_i'$ can be computed in parallel; inverse, all $z_{<i}$ must be known before $h_i$. Which direction you make cheap is the MAF/IAF decision, and the review notes both conventions are in use.

**Transformers.** Affine, $\tau=\alpha_iz_i+\beta_i$ with $\alpha_i=\exp\tilde\alpha_i$ for guaranteed invertibility and $\log\lvert\det J\rvert=\sum_i\tilde\alpha_i$ — simple, and the review is explicit that its limited expressiveness is why flows built on it need depth. Then combination-based (mixture CDFs, as in Flow++), integral-based (NAF, SOS, unconstrained monotone networks), and spline-based ([NSF](/blog/neural-spline-flows/)).

**Conditioners.** Recurrent — parameter sharing through an RNN state, but *it turns an inherently parallel computation into a sequential one*, $O(D)$ steps, so rarely used. Masked — MADE, one network, one pass, all $h_i$ at once. Coupling — the aggressive-masking special case.

That these are *independent choices* is the review's most useful organising claim: *any type of transformer can be paired up with any type of conditioner*.

### 3.3 The $K$-part spectrum

A coupling layer sets $(h_1,\dots,h_d)$ to constants and $(h_{d+1},\dots,h_D)=F(z_{\le d})$, giving the Jacobian

$$
J_{f_\phi}=\begin{bmatrix}I&0\\A&\mathrm{D}\end{bmatrix} \tag{7}
$$

with $\mathrm{D}$ diagonal. Coupling and fully autoregressive are the two ends of one axis: *one can split the input into $K$ parts and transform the $k$-th part elementwise as a function of parts $1$ to $k-1$, with $K=2$ corresponding to a coupling layer and $K=D$ to a fully autoregressive flow*, and *inverting the transformation will be $O(K)$ times more expensive than evaluating it, hence $K$ could be chosen based on the computational requirements of the task*. I have never seen anyone actually tune $K$, and it is a free dial sitting in plain sight.

The expressiveness statement is sharp and is the single most useful fact in the review:

- **A single coupling layer is not a universal approximator**, *regardless of how expressive the function $F$ is*. It cannot represent an arbitrary autoregressive transformation.
- **A composition of $D$ coupling layers is**, provided the $i$-th has $d=i-1$ — because then the $i$-th layer can express any $z_i'=\tau(z_i;c_i(z_{<i}))$ and the composition is fully autoregressive.
- But that construction needs $D$ sequential layers in both directions, so it buys nothing over a masked autoregressive flow.
- **Open problem:** *whether it's possible to obtain a universal approximator by composing strictly fewer than $O(D)$ coupling layers.*

That last line is the honest state of the theory. Everyone builds deep coupling flows and observes that they work; nobody can say why a number of layers independent of $D$ should be enough.

### 3.4 Residual flows

$z'=z+g_\phi(z)$. Two routes to invertibility:

**Contractive.** If $g_\phi$ is contractive — $\delta(F(z_A),F(z_B))\le L\,\delta(z_A,z_B)$ with $L<1$ — the Banach fixed-point theorem gives a unique fixed point reachable by iteration from any starting point, and the inverse is computed by that iteration. This is i-ResNet and Residual Flows. The price: *contractive residual flows typically require iterative algorithms for sampling and density evaluation*, so neither direction is a single pass.

**Matrix determinant lemma.** For $A$ invertible $D\times D$ and $V,W$ of size $D\times M$,

$$
\det\bigl(A+VW^{\mathsf T}\bigr)=\det\bigl(I+W^{\mathsf T}A^{-1}V\bigr)\det A, \tag{8}
$$

which for diagonal $A$ turns an $O(D^3+D^2M)$ determinant into $O(M^3+DM^2)$, preferable when $M<D$. Planar flow, $z'=z+v\sigma(w^{\mathsf T}z+b)$, is the $M=1$ case — *expanding/contracting the space in the direction perpendicular to the hyperplane $w^{\mathsf T}z+b=0$* — and Sylvester flows are the general $M$.

So the taxonomy is: **autoregressive** flows (triangular Jacobian, one direction cheap), **linear** flows (full Jacobian, cheap by decomposition — PLU, QR, Householder — used mainly for permuting between autoregressive layers), **residual** flows (full Jacobian, cheap by contraction or by the determinant lemma), and **continuous-time** flows (trace instead of determinant).

### 3.5 The KL duality

Appendix A proves both directions in four lines each by change of variables:

$$
D_{\mathrm{KL}}\bigl[p_x^*\,\|\,p_x\bigr]=D_{\mathrm{KL}}\bigl[p_u^*\,\|\,p_u\bigr],
\qquad
D_{\mathrm{KL}}\bigl[p_x\,\|\,p_x^*\bigr]=D_{\mathrm{KL}}\bigl[p_u\,\|\,p_u^*\bigr], \tag{9}
$$

where $p_u^*$ is the distribution induced by running the inverse flow on the target. Forward KL in data space — maximum likelihood, mass-covering, needs samples from the target — equals a reverse KL in base space, in which the induced $p_u^*$ is fitted to the base $p_u$. Reverse KL in data space — variational, mode-seeking, needs the unnormalised target density — equals a forward KL in base space. This is the general form of the MAF/IAF equivalence, and it explains why "train a density estimator" and "fit a variational posterior" are the same procedure read in opposite directions.

The practical corollary: which KL you can use is decided by what you have. Samples from the target $\Rightarrow$ forward. An evaluable unnormalised density $\Rightarrow$ reverse. That single fact determines whether a flow is a density estimator or a proposal.

### 3.6 Topology, the constraint nobody designs around

Because $T$ is a diffeomorphism, $\mathcal{U}$ and $\mathcal{X}$ must be homeomorphic — and so must *every intermediate space* $\mathcal{Z}_k$ along the flow. Consequences:

- A flow cannot map $\mathbb{R}^D$ to the sphere $S^D$, or $\mathbb{R}^D$ to $\mathbb{R}^{D'}$ for $D\ne D'$.
- A continuous-time flow cannot implement $T:\mathbb{R}\to\mathbb{R}$ with $T(1)=-1$ and $T(-1)=1$, because the trajectories would have to cross (Dupont et al.). Augmenting with $\rho$ auxiliary dimensions bypasses it, but then *density evaluation in the original $D$-dimensional space can no longer be done analytically*, which *removes a benefit of flows that makes them an attractive modeling choice in the first place*.
- **If the target has disconnected modes, the base must have the same number.** *If the base density has fewer modes than the target (which will likely be the case in practice), the flow will be forced to assign a non-zero amount of probability mass to the 'empty' space between the disconnected modes.* True for every flow, not just continuous ones.

This is the deepest limitation in the review and the least acted on. A standard normal base has one mode. Real data — regimes in a return series, classes in an image set, failure modes in a sensor panel — often does not. Every flow in this series is spending capacity making the bridges between modes as thin as it can afford, and none of them can make them vanish. The fixes are latent-variable flows (RAD, Cornish et al.) that index a *mixture* of flows, and they give up analytic $p_x(x)$ for a variational bound — which is to say they give up the reason you wanted a flow.

> **My comment.** For daily returns I suspect this bites less than for images, since a regime-switching return distribution is a scale mixture around one centre and its support is connected. Where I would expect a bridge is a regime-level quantity, for example 20-day realised volatility in a two-regime market like TailFlow's synthetic one, which should be close to bimodal; that is the place I would measure the inter-mode mass.

## 4 Practical notes the review collects

- **Depth is the norm.** *Implementing a flow often amounts to composing as many transformations as computation and memory will allow* — Glow's architecture uses *as many as 320 sub-transformations distributed across 40 GPUs*.
- **Normalisation.** Batch norm, with fixed batch statistics, is a composition of two affine maps and therefore a legitimate flow layer with a diagonal Jacobian,
  $$
  \mathrm{BN}(z)=\alpha\frac{z-\hat\mu}{\sqrt{\hat\sigma^2+\varepsilon}}+\beta,
  \qquad
  \det J_{\mathrm{BN}}(z)=\prod_{i=1}^{D}\frac{\alpha_i}{\sqrt{\hat\sigma_i^2+\varepsilon}}.
  $$
- **Continuous-time flows** (§4) are covered with the Euler-discretisation equivalence to residual flows and the adjoint method — the [FFJORD](/blog/ffjord/) machinery, derived rather than cited.
- **Generalisations** (§5) is the part most people skip and the part that ages best: the general probability-transformation formula for non-diffeomorphic maps, piecewise-invertible transformations and mixtures of flows, flows for discrete variables, flows on Riemannian manifolds, and equivariant flows for symmetric densities.
- **Linear flow parameterisations** (Appendix B): PLU, with $L,U$ constrained to positive diagonals for invertibility and an $O(D)$ determinant; QR and Householder alternatives.

## 5 Applications, and what they establish

§6 is a catalogue, and its value is in showing how few of the uses are generative modelling.

**Density estimation and generation** (§6.1). The familiar case; the review's contribution here is mostly to point out that the forward-KL objective is mass-covering, which is why likelihood-trained models produce blurry-but-diverse samples.

**Importance and rejection sampling** (§6.2.1). A flow makes a proposal you can both sample and score, which is exactly the two operations importance sampling needs. This is the cleanest argument for flows outside generative modelling and it is one sentence in a 64-page review.

**MCMC** (§6.2.2). The historically striking observation: *the application of flows in MCMC precedes the appearance of flows in deep learning by at least a few decades*, because **Hamiltonian Monte Carlo is a flow**. On phase space $(\eta,v)$ with $H(\eta,v)=-\log p(\eta,v)$, HMC proposes $(\eta',v')=T(\eta,v)$ where $T$ is the continuous-time flow of

$$
\frac{d(\eta,v)}{dt}=\left(\frac{\partial H}{\partial v},-\frac{\partial H}{\partial\eta}\right) \tag{10}
$$

followed by momentum negation. That flow is *volume-preserving*, so its absolute Jacobian determinant is 1 everywhere, which together with the negation makes the proposal symmetric and cancels it from the Metropolis–Hastings ratio. Thirty years of HMC is a normalizing flow chosen precisely so its Jacobian determinant is trivial. A-NICE-MC replaces the Hamiltonian flow with a learned one.

**Variational inference** (§6.2.3) and **likelihood-free inference** (§6.2.4). The latter is where this group's own research programme lives: a conditional flow learns the likelihood $p(x\mid\theta)$ or the posterior $p(\theta\mid x)$ directly from simulator output, which is the only option when the simulator has no tractable density.

**Representation learning** (§6.3): hybrid discriminative-generative classification, and reinforcement learning where policies and models are parameterised by MAFs.

**What the review does not do.** It is not an empirical comparison. There is no table benchmarking flow classes against each other, no reproduction of published numbers, no recommendation grounded in measurement. Every comparative claim is analytical — cost, expressiveness, invertibility — and for the practical question "which flow should I use on my data", the review answers only with the trade-off structure. That is a legitimate choice for a review and it is worth being explicit that a reader looking for "use X" will not find it.

## 6 Limitations

**Stated by the authors.** That universality is representational and says nothing about finite-depth, finite-sample behaviour. That *more study of the theoretical properties of flows is also needed*, specifically *understanding their approximation capabilities for finite sample and finite depth settings*. That sub-$O(D)$ universality of coupling flows is open. That topological constraints are bypassed only at the cost of analytic densities.

**My reading.**

- **The empirical gap.** The review is analytically complete and empirically silent. It never says how much the affine-versus-spline transformer choice is worth in nats, or at what dimension coupling stops competing with autoregressive, because it never measures anything.
- **Tails are not discussed.** For a review written for statisticians, the absence of any treatment of tail behaviour — what happens to $p_x$ far from the data, how the base distribution's tail propagates through $T$, whether flows extrapolate — is a real omission. The base distribution's tail is, in most constructions, the model's tail.
- **The 2021 vintage shows in one place.** The simulation-free continuous-time methods ([Flow Matching](/blog/flow-matching/), Rectified Flow, Stochastic Interpolants) arrived after publication, and §4's treatment of continuous flows is entirely maximum-likelihood-with-a-solver. The conceptual apparatus survives — a probability path and a velocity field are still the objects — but the computational picture it paints is now the historical one.
- **Model selection is absent.** Nothing on how to compare two fitted flows, on marginal likelihood, on priors over flow parameters, or on any Bayesian treatment of the flow itself. Given the review's framing around well-specified probabilistic models, that is the missing chapter.
- **The $K$-part spectrum is stated and never explored**, here or anywhere since as far as I know.

> **My comment.** My model-uncertainty project sits in this missing chapter from an unusual side: the flow is not a model being compared but the prior that turns factor subsets into posterior probabilities. Its density made it controllable, and that cut both ways: Bayesian optimisation over its shape barely moved the held-out pricing error (0.25135 to 0.25045) while the held-out posterior entropy fell from 0.783 to 0.321. A flexible prior lets you tune the uncertainty measure itself, a hazard I would want any Bayesian treatment of flows to name.

## 7 Extensions

**What was built on this.** The conditioner/transformer vocabulary went straight into the software: `nflows`, `normflows`, Pyro's and TensorFlow Probability's flow modules all expose those two objects as separate classes, which is this review's decomposition compiled. The simulation-based-inference programme it summarises in §6.2.4 became `sbi` and a substantial applied literature in cosmology, neuroscience and epidemiology. The topological-constraint discussion in §5.5 is the reference for the mixture-of-flows and augmented-flow lines. And the open problem in §3.3 — universality of coupling flows at sub-linear depth — is still, to my knowledge, the sharpest unanswered theoretical question about the models everyone actually uses.

**Open problems the review names.** Sub-$O(D)$ universality for coupling flows. Finite-depth, finite-sample approximation guarantees. More flexible transformations at tractable cost, which the review calls *a core issue for some time*.

**Research directions.** *These are ideas, not results — none has been run.*

1. **Tune the $K$ nobody tunes.** Hypothesis: on tabular data there is an interior optimum in $K$ — the number of parts in the $K$-way generalisation of coupling — trading expressiveness against the $O(K)$ inverse cost, and it sits nearer 2 than $D$ for low data-per-dimension and nearer $D$ for high, mirroring the data-per-dimension thesis of [Neural Spline Flows](/blog/neural-spline-flows/). Data: the five UCI density benchmarks at several subsample sizes. Baseline: $K=2$ (coupling) and $K=D$ (masked autoregressive) with an otherwise identical spline transformer. Metric: test log-likelihood and inverse-pass wall-clock, as a frontier over $K\in\{2,4,8,\dots,D\}$. Likely failure mode: a $K$-way masked conditioner is fiddly enough to implement that the comparison ends up measuring implementation quality rather than $K$.
2. **Make the disconnected-modes penalty measurable.** Hypothesis: the probability mass a unimodal-base flow is forced to place between disconnected target modes is a predictable function of the mode separation and the flow's depth, and it is large enough at realistic depths to matter for anomaly detection — where that spurious inter-mode mass is exactly a false negative. Data: Gaussian mixtures in $\mathbb{R}^D$ with controlled separation and $D$; then a two-regime financial return series. Baseline: a mixture-of-flows model with a matched component count, and a Gaussian mixture. Metric: model mass assigned to a ball in the empty region, versus separation and depth; and AUROC for detecting points placed there. Likely failure mode: at high $D$ the "empty region" has negligible volume under any reasonable metric and the effect disappears, which would itself be worth stating as a dimension-dependent reassurance.
3. **Flows as SMC proposals, using the review's own §6.2.1 argument.** Hypothesis: a conditional flow trained by forward KL on simulator output gives a proposal whose importance weights have lower variance than the bootstrap proposal in a nonlinear state-space model, and the KL duality of §3.5 predicts *which* KL to train with — forward when simulated state/observation pairs are available, reverse when only the unnormalised optimal proposal is. Data: a simulated stochastic-volatility model with known parameters, then a daily index series. Baseline: bootstrap proposal, locally optimal Gaussian proposal, [auxiliary particle filter](/blog/auxiliary-particle-filter/). Metric: effective sample size per step and variance of the log-likelihood estimate at fixed particle count. Likely failure mode: forward-KL training makes the proposal mass-covering, which is good for weight variance, but the flow's Gaussian tails under-cover the rare states where weights are largest — the failure the review does not discuss because it never discusses tails.

## 8 Takeaways

- A flow is not a latent-variable model. $u=T^{-1}(x)$ is determined, not inferred, and that is the whole reason a flow gives you $p(x)$ where a VAE gives you a bound and a GAN gives you nothing.
- Universality is four lines: the conditional-CDF map has a triangular Jacobian whose determinant equals the density, so it maps any positive density to a uniform. Triangular structure is what the universal construction *has*, not a convenience bolted on.
- Every autoregressive flow is a conditioner and a transformer, chosen independently. Learn those two words and the literature becomes a table with visible empty cells.
- Coupling and fully autoregressive are the $K=2$ and $K=D$ ends of one spectrum, with the inverse costing $O(K)$ times the forward. A single coupling layer is provably not universal; $D$ of them are; anything in between is an open problem.
- Forward KL in data space equals reverse KL in base space, and reverse equals forward. Maximum likelihood and variational inference are one optimisation read from two ends, and which one you can run is decided by whether you have samples or an unnormalised density.
- A flow preserves topology. A unimodal base cannot produce a genuinely disconnected target; the model will put mass where there is none, and the standard fixes cost you the analytic density.
- The applications that matter least resemble image generation: proposals for importance sampling and SMC, MCMC kernels, variational posteriors, and likelihood-free inference. HMC was a volume-preserving flow decades before anyone called it one.
- The review is analytical throughout and measures nothing. For "which flow for my data", it gives you the trade-off structure and leaves the experiment to you.

## References

1. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., Lakshminarayanan, B. *Normalizing Flows for Probabilistic Modeling and Inference.* JMLR 22(57):1-64, 2021. arXiv:1912.02762.
2. Hyvärinen, A., Pajunen, P. *Nonlinear Independent Component Analysis: Existence and Uniqueness Results.* Neural Networks 12(3), 1999.
3. Dupont, E., Doucet, A., Teh, Y. W. *Augmented Neural ODEs.* NeurIPS 2019.
4. Cornish, R., Caterini, A. L., Deligiannidis, G., Doucet, A. *Relaxing Bijectivity Constraints with Continuously Indexed Normalising Flows.* ICML 2020.
5. Behrmann, J., Grathwohl, W., Chen, R. T. Q., Duvenaud, D., Jacobsen, J.-H. *Invertible Residual Networks.* ICML 2019.
6. Rezende, D. J., Mohamed, S. *Variational Inference with Normalizing Flows.* ICML 2015.
7. Neal, R. M. *MCMC using Hamiltonian Dynamics.* Handbook of Markov Chain Monte Carlo, 2011.
8. Durkan, C., Bekasov, A., Murray, I., Papamakarios, G. *Neural Spline Flows.* NeurIPS 2019.
