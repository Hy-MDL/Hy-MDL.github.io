---
title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
paper:
  title: "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
  authors: "Ben Mildenhall et al."
  venue: "ECCV 2020"
  arxiv: "2003.08934"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "vision"
order: 9
tags: [nerf, view-synthesis, volume-rendering, implicit-representation, positional-encoding]
date: 2020-04-01
status: draft
summary: "A single MLP maps a 3D point and a viewing direction to density and colour, and classical volume rendering turns that field into images that can be fit to posed photographs by gradient descent."
---

## Abstract

NeRF stores one static scene inside the weights of a plain fully-connected network. The network takes a 3D position and a viewing direction and returns a volume density and an RGB colour; images are produced by integrating those outputs along camera rays with the standard volume rendering equation. Since that integral is differentiable, the only supervision needed is a set of photographs with known camera poses. Two additions make the idea work in practice: a sinusoidal encoding of the inputs, which lets the MLP express fine detail, and a coarse-to-fine sampling scheme that spends ray samples where the scene actually is. On synthetic and real captures it beats Neural Volumes, SRN and LLFF on almost every metric, while the whole scene fits in about 5 MB of weights. The price is time: one to two days of optimisation per scene on a V100.

**Keywords:** view synthesis, neural radiance field, volume rendering, positional encoding, hierarchical sampling, implicit scene representation

## 1 Introduction

View synthesis asks for new images of a scene given a handful of photographs of it. Before this paper, the options were roughly two. Mesh-based pipelines can be optimised through differentiable rasterisers or path tracers, but the loss landscape is poorly behaved and they need a template mesh with fixed topology, which real scenes do not come with. Discrete volumetric methods (voxel grids, multiplane images) optimise well and handle complicated shapes, yet their cost grows with resolution because finer images require a finer 3D grid.

A third line, coordinate-based MLPs that encode shape as a signed distance or occupancy function, avoids the grid entirely. The authors point out that these had only been shown on simple, low-complexity shapes and produced oversmoothed renderings. NeRF keeps the continuous-MLP idea but changes what the network represents: not a surface, but a *volume* with view-dependent emission. <mark>The claim is that this is the first continuous neural scene representation able to render high-resolution, photorealistic views of real scenes from ordinary RGB captures.</mark>

## 2 Background

Two pieces of older machinery carry the method.

**Volume rendering.** A participating medium is described by a density $\sigma(\mathbf{x})$, read here as the differential probability that a ray stops at an infinitesimal particle at $\mathbf{x}$, and an emitted colour. The colour seen along a ray is an integral of emission weighted by how much light survives to that depth. The discrete quadrature the paper uses comes from Max's 1995 review of optical models, and it is identical to front-to-back alpha compositing.

**Spectral bias.** Rahaman et al. showed that deep networks fit low-frequency components first and struggle with high-frequency ones, and that lifting inputs through high-frequency functions helps. NeRF leans on this result directly.

## 3 Method

> **Key idea.** Do not predict geometry. Predict a field of density and direction-dependent colour with an MLP, render it with a differentiable integral along each ray, and let the photometric error on the input photos shape the field.

The pipeline is sketched in [Fig. 2 in the paper](https://arxiv.org/pdf/2003.08934#page=5): sample points on a ray, query the MLP, composite, compare with the ground-truth pixel.

### 3.1 The radiance field

The scene is a function

$$
F_\Theta : (\mathbf{x}, \mathbf{d}) \mapsto (\mathbf{c}, \sigma) \tag{1}
$$

where $\mathbf{x}\in\mathbb{R}^3$ is position, $\mathbf{d}$ a unit viewing direction, $\mathbf{c}$ RGB colour and $\sigma$ density. To keep geometry consistent across views, <mark>density depends on position only, while colour may depend on both position and direction</mark>. Architecturally, eight 256-wide ReLU layers process $\mathbf{x}$ and emit $\sigma$ plus a 256-d feature; that feature is concatenated with $\mathbf{d}$ and passed through one more 128-wide layer to give colour.

### 3.2 Volume rendering

For a ray $\mathbf{r}(t)=\mathbf{o}+t\mathbf{d}$ between bounds $t_n$ and $t_f$, the expected colour is

$$
C(\mathbf{r})=\int_{t_n}^{t_f} T(t)\,\sigma(\mathbf{r}(t))\,\mathbf{c}(\mathbf{r}(t),\mathbf{d})\,dt,\qquad T(t)=\exp\!\Big(-\!\int_{t_n}^{t}\sigma(\mathbf{r}(s))\,ds\Big) \tag{2}
$$

with $T(t)$ the transmittance, i.e. the probability the ray reaches $t$ unobstructed. The interval is cut into $N$ equal bins and one sample $t_i$ is drawn uniformly inside each bin. Because the draws change every iteration, the MLP ends up being queried at continuous locations over training rather than on a fixed lattice. The estimator is

$$
\hat C(\mathbf{r})=\sum_{i=1}^{N} T_i\,\big(1-e^{-\sigma_i\delta_i}\big)\,\mathbf{c}_i,\qquad T_i=\exp\!\Big(-\sum_{j<i}\sigma_j\delta_j\Big) \tag{3}
$$

where $\delta_i=t_{i+1}-t_i$. Setting $\alpha_i = 1-e^{-\sigma_i\delta_i}$ recovers ordinary alpha compositing.

### 3.3 Positional encoding

Feeding raw coordinates gives blurry results. Each scalar input $p$ is instead expanded as

$$
\gamma(p)=\big(\sin(2^0\pi p),\cos(2^0\pi p),\dots,\sin(2^{L-1}\pi p),\cos(2^{L-1}\pi p)\big) \tag{4}
$$

with $L=10$ for the components of $\mathbf{x}$ and $L=4$ for $\mathbf{d}$. The authors note the same form appears in Transformers, but for a different reason: there it injects token order, here it makes a continuous function easier to fit at high frequency.

### 3.4 Hierarchical sampling

Uniform sampling wastes queries on empty or occluded space. NeRF trains two networks. The coarse one is evaluated at $N_c$ stratified samples, and its compositing weights $w_i=T_i(1-e^{-\sigma_i\delta_i})$ are normalised into a piecewise-constant density along the ray. $N_f$ further points are drawn from that density by inverse-transform sampling, and the fine network is evaluated on all $N_c+N_f$ points. The loss sums squared colour error for both passes over a batch of rays $\mathcal{R}$:

$$
\mathcal{L}=\sum_{\mathbf{r}\in\mathcal{R}}\Big[\lVert \hat C_c(\mathbf{r})-C(\mathbf{r})\rVert_2^2+\lVert \hat C_f(\mathbf{r})-C(\mathbf{r})\rVert_2^2\Big] \tag{5}
$$

The coarse term is kept so that its weights remain a useful proposal distribution. Settings: 4096 rays per batch, $N_c=64$, $N_f=128$, Adam with learning rate decaying from $5\times10^{-4}$ to $5\times10^{-5}$, 100–300k iterations.

## 4 Experiments

Three datasets: the four simple Lambertian objects of DeepVoxels (512×512, 479 training views), a new path-traced set of eight objects with complex geometry and non-Lambertian materials (800×800, 100 training and 200 test views), and eight real forward-facing phone captures (20–62 images each at 1008×756, one eighth held out). Camera poses for real data come from COLMAP. Baselines are Neural Volumes (NV), Scene Representation Networks (SRN) and Local Light Field Fusion (LLFF).

| Method | Diffuse Synth. PSNR↑ / SSIM↑ / LPIPS↓ | Realistic Synth. PSNR↑ / SSIM↑ / LPIPS↓ | Real Fwd-Facing PSNR↑ / SSIM↑ / LPIPS↓ |
|---|---|---|---|
| SRN | 33.20 / 0.963 / 0.073 | 22.26 / 0.846 / 0.170 | 22.84 / 0.668 / 0.378 |
| NV | 29.62 / 0.929 / 0.099 | 26.05 / 0.893 / 0.160 | – |
| LLFF | 34.38 / 0.985 / 0.048 | 24.88 / 0.911 / 0.114 | 24.13 / 0.798 / **0.212** |
| **NeRF** | **40.15 / 0.991 / 0.023** | **31.01 / 0.947 / 0.081** | **26.50 / 0.811** / 0.250 |

<mark>NeRF wins every column except LPIPS on real forward-facing scenes, where LLFF is slightly better</mark>; the authors argue the video results still favour NeRF because LLFF blends separate per-view representations and flickers between them. The gap is largest on the realistic synthetic set, about 5 dB over the best baseline, which is where fine geometry such as the ship rigging and Lego treads is hard for grids ([Fig. 5 in the paper](https://arxiv.org/pdf/2003.08934#page=11)).

Ablations on the realistic synthetic set (full model 31.01 dB):

| Variant | PSNR↑ |
|---|---|
| No PE, no view dependence, no hierarchy | 26.67 |
| No view dependence | 27.66 |
| No positional encoding | 28.77 |
| No hierarchical sampling | 30.06 |
| 25 input images | 27.78 |
| 50 input images | 29.79 |
| $L=5$ / $L=15$ | 30.59 / 30.81 |

<mark>Positional encoding and view dependence matter most; hierarchical sampling is a smaller, efficiency-oriented gain.</mark> Going from $L=10$ to $L=15$ does not help, which the authors attribute to $2^L$ exceeding the highest frequency present in the images. The visual effect of the first two is shown in [Fig. 4 in the paper](https://arxiv.org/pdf/2003.08934#page=7): without direction input the specular highlight on the bulldozer tread disappears, and without encoding the whole object is smoothed.

On storage, <mark>the network weights take about 5 MB per scene, against more than 15 GB for LLFF's per-view grids on one synthetic scene</mark>, a ratio the paper puts at 3000×.

## 5 Discussion

**Strengths.** The design is small: one MLP, one integral, one L2 loss, no 3D supervision. The separation of density (position only) from colour (position and direction) is a clean inductive bias that gives multi-view consistency almost for free. The ablation table is honest and makes clear which parts carry the result.

**Weaknesses.** Cost is the obvious one. Each scene needs its own optimisation of roughly one to two days on a V100, and every pixel at test time requires 64 + 192 network evaluations across the two passes. LLFF, by comparison, processes a small capture in under ten minutes. The method also assumes a static scene, fixed lighting and accurate poses; nothing is learned across scenes, so there is no prior to fall back on when views are sparse (performance drops by more than 3 dB going from 100 to 25 images).

**Not shown.** Per-frame rendering time is not reported. There is no study of robustness to pose error, no unbounded 360° real scenes, and no analysis of the recovered geometry itself, only of rendered colour. The authors themselves flag interpretability: with a grid one can reason about failure modes, with MLP weights one cannot.

## 6 Takeaways

- A radiance field, not a surface, is the right thing to regress with a coordinate MLP: volumes are forgiving to optimise and the rendering integral is differentiable end to end.
- Sinusoidal input features are what turn a blurry fit into a sharp one; this is the spectral-bias fix that later became standard for coordinate networks.
- The coarse network acts as a learned proposal distribution for where to integrate, a form of importance-driven discretisation rather than Monte Carlo estimation.
- Compactness and quality were bought with compute. The follow-up literature on grids, hashes and Gaussians is largely about getting that time back.
- For financial time-series modelling the link is indirect but real in two places: Eq. (2) has the same structure as an intensity model, with $\sigma$ as a hazard rate and $T$ as a survival probability, and Fourier-feature encodings of a continuous coordinate are the same device used to embed time or noise level in diffusion networks. Nothing else in the paper transfers directly.

## References

- Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., Ng, R. *NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.* ECCV 2020. arXiv:2003.08934.
- Mildenhall, B. et al. *Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines.* SIGGRAPH 2019.
- Sitzmann, V., Zollhöfer, M., Wetzstein, G. *Scene Representation Networks.* NeurIPS 2019.
- Lombardi, S. et al. *Neural Volumes: Learning Dynamic Renderable Volumes from Images.* SIGGRAPH 2019.
- Rahaman, N. et al. *On the Spectral Bias of Neural Networks.* ICML 2018 (as cited in the paper).
