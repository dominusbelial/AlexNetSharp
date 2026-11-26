---
id: conv2a_neuron
title: conv2a_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:65; layer.py:627; include/neuron.cuh:163
---

This instance has 93,312 output values per image and 0 parameter slots. Effective inputs in order: conv2a.

Configuration: gpu=0; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after conv2a.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [conv2a](card:conv2a) · [rnorm2a](card:rnorm2a)

Formula: y = x < 0 ? 0 : x
