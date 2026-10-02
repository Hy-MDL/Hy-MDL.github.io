---
title: "CLIP: Learning Transferable Visual Models From Natural Language Supervision"
paper:
  title: "Learning Transferable Visual Models From Natural Language Supervision"
  authors: "Alec Radford et al."
  venue: "ICML 2021"
  arxiv: "2103.00020"
  license: "arxiv.org/licenses/nonexclusive-distrib/1.0/"
series: "vision"
order: 5
tags: [clip, contrastive-learning, vision-language, zero-shot, prompt-engineering, distribution-shift, representation-learning]
date: 2021-04-01
status: draft
summary: "Training an image encoder and a text encoder to match 400 million web image-caption pairs yields a model whose classifier can be written in plain English, reaching ResNet-50-level ImageNet accuracy with no ImageNet labels and much better robustness under distribution shift."
---

## Abstract

Image classifiers are normally trained against a closed list of categories, so every new concept needs new labels. CLIP replaces the label list with free-form text: an image encoder and a text encoder are trained jointly, from scratch, to decide which caption in a batch belongs to which image, using 400 million image-text pairs gathered from the web. After pre-training, a classifier for any dataset is obtained by embedding the class names as sentences, with no gradient step. Evaluated on more than 30 datasets, this zero-shot classifier is often on par with supervised baselines, and on ImageNet it equals the original ResNet-50 without touching the 1.28 million training images. The features are also strong under linear probing and far more stable under natural distribution shift than ImageNet-trained models.

**Keywords:** contrastive pre-training, natural language supervision, zero-shot transfer, prompt ensembling, linear probe, effective robustness

## 1 Introduction

In NLP, task-agnostic pre-training on raw web text lets one model serve many tasks without task-specific heads. Vision in 2021 still leaned on crowd-labelled datasets such as ImageNet. Learning image representations from accompanying text had been tried for two decades with unconvincing results: the closest zero-shot predecessor, Visual N-Grams, reached only 11.5% on ImageNet. What worked instead was weak supervision (Instagram hashtags, JFT-300M) with a fixed label vocabulary and a static softmax head, which cannot express a new concept at test time.

The authors argue that the real gap between the two camps is scale: accelerator-years on up to billions of images versus accelerator-days on one or two hundred thousand (VirTex, ICMLM, ConVIRT). CLIP is a simplified ConVIRT pushed to web scale, and the paper studies what happens when you do that.

## 2 Background

Existing paired datasets were too small: MS-COCO and Visual Genome have around 100,000 photos each, and YFCC100M shrinks to about 15 million images once only English natural-language titles or descriptions are kept. The authors therefore built WIT (WebImageText): 400 million pairs collected by searching for text containing one of 500,000 queries, with at most 20,000 pairs per query to keep the concept distribution roughly balanced.

The objective is not new: it is the multi-class N-pair loss, better known as InfoNCE, which ConVIRT had already applied to medical image-text pairs.

## 3 Method

> **Key idea.** Do not try to predict the words of a caption. Only ask which whole caption goes with which image inside a batch. That easier proxy task is far cheaper to learn, and because the text encoder stays in the loop, it can later turn any sentence into the weights of a classifier.

### 3.1 Why contrastive

The first attempt was a captioning model: a CNN plus a 63M-parameter transformer language model. It learned ImageNet concepts three times more slowly than a baseline that predicts a bag-of-words encoding of the same text. <mark>Keeping the bag-of-words text encoder but swapping the predictive loss for a contrastive one gave a further 4x gain in zero-shot ImageNet efficiency</mark> ([Fig. 2 in the paper](https://arxiv.org/pdf/2103.00020#page=3) plots accuracy against images processed for the three objectives).

### 3.2 Objective

For a batch of $N$ pairs, let $f_i$ and $g_j$ be the image and text encoder outputs, $W_I, W_T$ linear projections into a shared space, and

$$
u_i = \frac{W_I f_i}{\lVert W_I f_i \rVert}, \qquad v_j = \frac{W_T g_j}{\lVert W_T g_j \rVert}, \qquad s_{ij} = e^{t}\, u_i^\top v_j . \tag{1}
$$

Here $s_{ij}$ is a cosine similarity scaled by a learned log-temperature $t$. The loss is a symmetric cross-entropy over rows and columns of the $N \times N$ similarity matrix:

$$
\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{e^{s_{ii}}}{\sum_{j} e^{s_{ij}}} + \log\frac{e^{s_{ii}}}{\sum_{j} e^{s_{ji}}}\right]. \tag{2}
$$

The $N$ diagonal entries are the true pairs and the $N^2 - N$ off-diagonal ones act as negatives. Compared with ConVIRT the recipe is stripped down: no pre-trained initialisation, a linear projection head, a random square crop as the only augmentation, and a learned temperature (initialised to the equivalent of 0.07, with the logit scale clipped at 100).

### 3.3 Zero-shot classifier

For a dataset with $K$ classes, each class name is placed in a template such as "A photo of a {label}." and embedded to get $v_1,\dots,v_K$. The prediction for image $x$ is

$$
p(y = k \mid x) = \frac{\exp\!\left(e^{t}\, u(x)^\top v_k\right)}{\sum_{k'=1}^{K}\exp\!\left(e^{t}\, u(x)^\top v_{k'}\right)} . \tag{3}
$$

This is multinomial logistic regression with normalised inputs and weights and no bias; <mark>the text encoder plays the role of a hypernetwork that writes the classifier weights from a description</mark>. The weights are computed once and cached.

### 3.4 Models and training

The image tower is either a modified ResNet (with attention pooling instead of global average pooling) or a Vision Transformer. The text tower is a 12-layer, 512-wide transformer whose end-of-sequence activation is the text feature. Eight models were trained: five ResNets (RN50, RN101, RN50x4/x16/x64) and three ViTs (B/32, B/16, L/14), all for 32 epochs with a batch of 32,768. The largest ResNet took 18 days on 592 V100s; ViT-L/14 took 12 days on 256 V100s and was then run for one more epoch at 336 pixels. That last model is the default "CLIP" in the results.

## 4 Experiments

**Zero-shot versus prior work.** The only comparable earlier system is Visual N-Grams (Table 1 of the paper):

| Model | aYahoo | ImageNet | SUN |
|---|---|---|---|
| Visual N-Grams | 72.4 | 11.5 | 23.0 |
| **CLIP (ViT-L/14@336px)** | **98.4** | **76.2** | **58.5** |

The authors stress that this is context, not a controlled comparison. <mark>76.2% zero-shot top-1 equals the original ResNet-50, and top-5 reaches 95%.</mark>

**Prompts matter.** Bare class names are ambiguous ("boxer" in Oxford-IIIT Pets is a dog) and unlike the full sentences seen in training. The default template adds 1.3% on ImageNet, an ensemble of 80 prompts averaged in embedding space adds another 3.5%, and across 36 datasets the two tricks together are worth almost 5 points, roughly what 4x more compute buys.

**Against a supervised baseline.** Compared with logistic regression on ResNet-50 features, zero-shot CLIP wins on 16 of 27 datasets. The gains are largest on Stanford Cars (+28.9), Country211 (+23.2) and Food101 (+22.5); the losses are largest on EuroSAT (−37.1), KITTI Distance (−34.0) and PatchCamelyon (−19.5). Specialised or abstract tasks are clearly weak.

**Few-shot and data efficiency.** <mark>Zero-shot CLIP matches a 4-shot linear probe on its own features</mark> and is close to the best 16-shot probe among public models (BiT-M). The estimated number of labels per class needed to match zero-shot ranges from under 1 to 184, with median 5.4. Zero-shot still trails a fully supervised probe on the same features by 10 to 25 points on most datasets, and its error follows a log-log linear trend over a 44x range of compute.

**Linear probes.** <mark>On the 12-dataset Kornblith suite the best CLIP model beats the previous best (Noisy Student EfficientNet-L2) by 2.6% on average, and by 5% on the broader 27-dataset suite, winning on 21 of 27.</mark> CLIP ViTs are about 3x more compute-efficient than CLIP ResNets.

**Robustness.** On seven natural distribution shifts of ImageNet, <mark>zero-shot CLIP closes up to 75% of the gap between in-distribution and shifted accuracy</mark>. Against a ResNet-101 with the same 76.2% ImageNet score, CLIP obtains 70.1 vs 64.3 on ImageNetV2, 88.9 vs 37.7 on ImageNet-R, 72.3 vs 32.6 on ObjectNet, 60.2 vs 25.2 on ImageNet Sketch and 77.1 vs 2.7 on ImageNet-A. <mark>Fitting a linear classifier on ImageNet raises ImageNet accuracy by 9.2 points to 85.4% yet slightly lowers average accuracy under shift</mark>, and the robustness advantage fades steadily from 0-shot to fully supervised.

## 5 Discussion

**Strengths.** The method is very simple, and the paper's weight is in its evaluation: dozens of datasets, 66 reference models, and a per-dataset breakdown that shows failures as prominently as wins. The robustness experiment is the most thought-provoking part, since it separates "has good features" from "was fitted to this distribution".

**Weaknesses.** The headline zero-shot comparison is against a linear probe on a ResNet-50, a modest baseline, and zero-shot remains well under CLIP's own supervised probe. The paper describes WIT only briefly, so the main ingredient is hard to inspect. Prompts were tuned per dataset, which blurs the meaning of "zero-shot", and the compute is out of reach for most labs.

**Not shown in the part I read.** Through Section 3 there is no ablation over dataset or batch size, and no controlled test of whether robustness comes from language supervision, data diversity, or simply not training on ImageNet; the authors say they lack confident answers. Data overlap, bias and limitations are treated in later sections that this note does not cover.

## 6 Takeaways

- A batch-level matching loss between two encoders, plus enough paired data, is sufficient to get a classifier that is programmed with sentences instead of labels.
- Choosing the objective by training efficiency was decisive: contrastive matching beat caption generation by roughly an order of magnitude (3x then 4x) in the small-scale comparison.
- Prompt wording and ensembling are part of the model, worth almost 5 points on average at no inference cost once cached.
- Adapting to one distribution buys in-distribution accuracy and gives back robustness; zero-shot and few-shot evaluation over broad suites gives a truer picture of capability.
- For financial time-series diffusion the link is indirect: the symmetric InfoNCE loss is a candidate for aligning time-series windows with paired text (news, filings) to build conditioning embeddings, and the robustness result is a reminder that fitting tightly to one regime may gain nothing out of regime.

## References

1. A. Radford, J. W. Kim, C. Hallacy, et al. *Learning Transferable Visual Models From Natural Language Supervision.* arXiv:2103.00020, 2021.
2. Y. Zhang, H. Jiang, Y. Miura, C. D. Manning, C. P. Langlotz. *Contrastive Learning of Medical Visual Representations from Paired Images and Text (ConVIRT).* 2020.
3. A. Li, A. Jabri, A. Joulin, L. van der Maaten. *Learning Visual N-Grams from Web Data.* 2017.
4. A. van den Oord, Y. Li, O. Vinyals. *Representation Learning with Contrastive Predictive Coding (InfoNCE).* 2018.
5. R. Taori, A. Dave, V. Shankar, N. Carlini, B. Recht, L. Schmidt. *Measuring Robustness to Natural Distribution Shift in Image Classification.* 2020.
