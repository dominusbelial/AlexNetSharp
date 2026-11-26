---
id: hs1b
title: hs1b
summary: Evaluation multiplies the response by the keep probability.
paper-section: 4.2 / Dropout
citations: layers/layers-145.cfg:261; src/layer.cu:1127; src/layer.cu:1133
---

This instance has 2,048 output values per image and 0 parameter slots. Effective inputs in order: fc2048b_neuron.

Configuration: enable=True; gpu=1; keep=0.5; neuron=; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: The original convention uses keep multiplication during evaluation, so omitting this node would change every later response.

[type-hs](card:type-hs) · [fc2048b_neuron](card:fc2048b_neuron) · [fc2048ba](card:fc2048ba) · [fc2048bb](card:fc2048bb)

Formula: y = keep * x (evaluation); y = Bernoulli(keep) * x (training)
