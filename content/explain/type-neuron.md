---
id: type-neuron
title: neuron
summary: A separate ReLU removes negative responses.
paper-section: 3.1 / ReLU nonlinearity
citations: layer.py:627; include/neuron.cuh:163
---

The original parser detaches neuron=relu into an explicit layer and redirects downstream inputs to it. ReLU keeps positive values and maps negative values to zero.

Why it matters: Showing this node preserves the exact order of convolution, normalization, pooling and activation.

[pool1a_neuron](card:pool1a_neuron) · [pool1b_neuron](card:pool1b_neuron) · [conv2a_neuron](card:conv2a_neuron) · [conv2b_neuron](card:conv2b_neuron) · [conv3a_neuron](card:conv3a_neuron) · [conv3b_neuron](card:conv3b_neuron) · [conv4a_neuron](card:conv4a_neuron) · [conv4b_neuron](card:conv4b_neuron) · [pool3a_neuron](card:pool3a_neuron) · [pool3b_neuron](card:pool3b_neuron) · [fc2048a_neuron](card:fc2048a_neuron) · [fc2048b_neuron](card:fc2048b_neuron) · [fc2048ba_neuron](card:fc2048ba_neuron) · [fc2048bb_neuron](card:fc2048bb_neuron)

Formula: y = x < 0 ? 0 : x
