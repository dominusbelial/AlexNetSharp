---
id: pool1a_neuron
title: pool1a_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:47; layer.py:627; include/neuron.cuh:163
---

This instance has 34,992 output values per image and 0 parameter slots. Effective inputs in order: pool1a.

Configuration: gpu=0; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after pool1a.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [pool1a](card:pool1a) · [conv2a](card:conv2a)

Formula: y = x < 0 ? 0 : x
