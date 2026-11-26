---
id: type-conv
title: conv
summary: Shared filters scan local windows to build feature maps.
paper-section: 3.5 / Overall architecture
citations: layer.py:1004; layer.py:1041; src/layer.cu:829
---

Each output channel uses its own small RGB or feature-space filter at every location. Multiple inputs keep independent weight matrices in input order.

Why it matters: Ceiling arithmetic preserves a partial final window. Conv1 has no immediate ReLU; signed responses reach normalization.

[conv1a](card:conv1a) · [conv1b](card:conv1b) · [conv2a](card:conv2a) · [conv2b](card:conv2b) · [conv3a](card:conv3a) · [conv3b](card:conv3b) · [conv4a](card:conv4a) · [conv4b](card:conv4b) · [conv5a](card:conv5a) · [conv5b](card:conv5b)

Formula: y[f,y,x] = bias[f] + sum_inputs sum_c,ky,kx W[c,ky,kx,f] * input[c,y*stride+start+ky,x*stride+start+kx]; outside input = 0
