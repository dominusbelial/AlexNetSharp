---
id: logprob
title: logprob
summary: Measure the synthetic label probability and ranking errors.
paper-section: 3 / Multinomial logistic regression
citations: layers/layers-145.cfg:305; src/layer.cu:1825; src/layer_kernels.cu:106
---

This instance has 0 output values per image and 0 parameter slots. Effective inputs in order: labels, probs.

Configuration: coeff=1; gpu=0; quantB=default; quantF=default; topk=5.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: This is an evaluation measurement for one artificial label, not evidence of classification quality or training.

[type-cost.logreg](card:type-cost.logreg) · [labels](card:labels) · [probs](card:probs)

Formula: NLL = -log(p[label]); top-k error = 1 - clamp((k-B)/E,0,1), B = larger probabilities, E = equal probabilities
