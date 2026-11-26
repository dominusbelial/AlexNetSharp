---
id: fc2048a_neuron
title: fc2048a_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:238; layer.py:627; include/neuron.cuh:163
---

This instance has 2,048 output values per image and 0 parameter slots. Effective inputs in order: fc2048a.

Configuration: gpu=0; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after fc2048a.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [fc2048a](card:fc2048a) · [hs1a](card:hs1a)

Formula: y = x < 0 ? 0 : x
