---
id: type-softmax
title: softmax
summary: Turn the 1000 logits into class probabilities.
paper-section: 3 / 1000-way softmax
citations: src/layer.cu:1020; src/layer.cu:1026
---

Subtract the largest logit, exponentiate, and divide by the total. Class labels are shown only as indices because this slice has no ImageNet class map.

Why it matters: Untrained probabilities have no predictive meaning. Subtracting the maximum avoids overflow for finite logits.

[probs](card:probs)

Formula: p[c] = exp(z[c]-max(z)) / sum_j exp(z[j]-max(z))
