---
id: pool3a_neuron
title: pool3a_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:220; layer.py:627; include/neuron.cuh:163
---

This instance has 4,608 output values per image and 0 parameter slots. Effective inputs in order: pool3a.

Configuration: gpu=0; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after pool3a. This is the pool after conv5, often called pool5; the config names it pool3.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [pool3a](card:pool3a) · [fc2048a](card:fc2048a) · [fc2048b](card:fc2048b)

Formula: y = x < 0 ? 0 : x
