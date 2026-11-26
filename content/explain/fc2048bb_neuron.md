---
id: fc2048bb_neuron
title: fc2048bb_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:275; layer.py:627; include/neuron.cuh:163
---

This instance has 2,048 output values per image and 0 parameter slots. Effective inputs in order: fc2048bb.

Configuration: gpu=1; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after fc2048bb.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [fc2048bb](card:fc2048bb) · [hs2b](card:hs2b)

Formula: y = x < 0 ? 0 : x
