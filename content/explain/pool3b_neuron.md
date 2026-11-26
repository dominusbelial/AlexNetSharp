---
id: pool3b_neuron
title: pool3b_neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layers/layers-145.cfg:229; layer.py:627; include/neuron.cuh:163
---

This instance has 4,608 output values per image and 0 parameter slots. Effective inputs in order: pool3b.

Configuration: gpu=1; neuron=relu; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the explicit ReLU after pool3b. This is the pool after conv5, often called pool5; the config names it pool3.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[type-neuron](card:type-neuron) · [pool3b](card:pool3b) · [fc2048a](card:fc2048a) · [fc2048b](card:fc2048b)

Formula: y = x < 0 ? 0 : x
