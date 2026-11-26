---
id: conv2b_neuron
title: conv2b_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:80; layer.py:627; include/neuron.cuh:163
---

This instance has 93,312 output values per image and 0 parameter slots. Effective inputs in order: conv2b.

Configuration: gpu=1; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after conv2b.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [conv2b](card:conv2b) · [rnorm2b](card:rnorm2b)

Formula: y = x < 0 ? 0 : x
