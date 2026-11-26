---
id: type-fc
title: fc
summary: Combine every input feature with learned weighted sums.
paper-section: 3.5 / Fully connected layers
citations: src/layer.cu:640; src/layer.cu:644; layers/layers-145.cfg:277
---

Fully connected layers keep one weight matrix per input. The bias is added after the first product, before later products. The four 2048 nodes have separate following ReLUs.

Why it matters: Cross-tower fan-in joins both halves. fc2048bb intentionally reads hs1b before hs1a; its first weight matrix belongs to hs1b.

[fc2048a](card:fc2048a) · [fc2048b](card:fc2048b) · [fc2048ba](card:fc2048ba) · [fc2048bb](card:fc2048bb) · [fc1000](card:fc1000)

Formula: y = (X[0] W[0] + bias) + X[1] W[1]; reduction follows declared input order
