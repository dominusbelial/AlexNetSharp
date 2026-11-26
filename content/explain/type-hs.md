---
id: type-hs
title: hs
summary: Evaluation multiplies the response by the keep probability.
paper-section: 4.2 / Dropout
citations: src/layer.cu:1127; src/layer.cu:1133
---

Training would use an unscaled Bernoulli mask. This replay is evaluation mode and makes no random dropout draws.

Why it matters: The original convention uses keep multiplication during evaluation, so omitting this node would change every later response.

[hs1a](card:hs1a) · [hs1b](card:hs1b) · [hs2a](card:hs2a) · [hs2b](card:hs2b)

Formula: y = keep * x (evaluation); y = Bernoulli(keep) * x (training)
