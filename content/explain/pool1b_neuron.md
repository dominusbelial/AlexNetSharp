---
id: pool1b_neuron
title: pool1b_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:56; layer.py:627; include/neuron.cuh:163
---

This instance has 34,992 output values per image and 0 parameter slots. Effective inputs in order: pool1b.

Configuration: gpu=1; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after pool1b.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [pool1b](card:pool1b) · [conv2b](card:conv2b)

Formula: y = x < 0 ? 0 : x
