---
id: conv3b_neuron
title: conv3b_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:149; layer.py:627; include/neuron.cuh:163
---

This instance has 32,448 output values per image and 0 parameter slots. Effective inputs in order: conv3b.

Configuration: gpu=1; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after conv3b.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [conv3b](card:conv3b) · [conv4b](card:conv4b)

Formula: y = x < 0 ? 0 : x
