---
id: type-cost.logreg
title: cost.logreg
summary: Measure the synthetic label probability and ranking errors.
paper-section: 3 / Multinomial logistic regression
citations: src/layer.cu:1825; src/layer_kernels.cu:106
---

The cost reads labels and probabilities. It reports negative log probability, top-1 error and top-5 error; ties receive fractional credit.

Why it matters: This is an evaluation measurement for one artificial label, not evidence of classification quality or training.

[logprob](card:logprob)

Formula: NLL = -log(p[label]); top-k error = 1 - clamp((k-B)/E,0,1), B = larger probabilities, E = equal probabilities
