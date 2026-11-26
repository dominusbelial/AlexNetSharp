---
id: conv2a
title: conv2a
summary: Shared filters scan local windows to build feature maps.
paper-section: 3.5 / Overall architecture
citations: layers/layers-145.cfg:65; layer.py:1004; layer.py:1041; src/layer.cu:829
---

This instance has 93,312 output values per image and 153,728 parameter slots. Effective inputs in order: pool1a_neuron.

Configuration: channels=[48]; epsB=0.02; epsW=[1e-05]; filterSize=[5]; filters=[128]; gpu=0; groups=[1]; initB=1; initBFunc=; initCFunc=; initW=[0.01]; initWFunc=; momB=0.9; momW=[0.9]; neuron=relu; padding=[-2]; partialSum=3; quantB=default; quantF=default; randSparse=[False]; schedB=default; schedW=default; sharedBiases=True; stride=[1]; superEps=0; superMom=0; wball=[0]; wc=[0.0005]; wcNormMax=[0]; wcNormMin=[0]; weightSource=[''].

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: Ceiling arithmetic preserves a partial final window. Conv1 has no immediate ReLU; signed responses reach normalization.

[type-conv](card:type-conv) · [pool1a_neuron](card:pool1a_neuron) · [conv2a_neuron](card:conv2a_neuron)

Formula: y[f,y,x] = bias[f] + sum_inputs sum_c,ky,kx W[c,ky,kx,f] * input[c,y*stride+start+ky,x*stride+start+kx]; outside input = 0
