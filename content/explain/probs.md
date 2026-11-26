---
id: probs
title: probs
summary: Turn the 1000 logits into class probabilities.
paper-section: 3 / 1000-way softmax
citations: layers/layers-145.cfg:301; src/layer.cu:1020; src/layer.cu:1026
---

This instance has 1,000 output values per image and 0 parameter slots. Effective inputs in order: fc1000.

Configuration: gpu=0; neuron=; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: Untrained probabilities have no predictive meaning. Subtracting the maximum avoids overflow for finite logits.

[type-softmax](card:type-softmax) · [fc1000](card:fc1000) · [logprob](card:logprob)

Formula: p[c] = exp(z[c]-max(z)) / sum_j exp(z[j]-max(z))
