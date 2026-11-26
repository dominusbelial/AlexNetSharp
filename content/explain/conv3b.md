---
id: conv3b
title: conv3b
summary: Shared filters scan local windows to build feature maps.
paper-section: 3.5 / Overall architecture
citations: layers/layers-145.cfg:149; layer.py:1004; layer.py:1041; src/layer.cu:829
---

This instance has 32,448 output values per image and 442,560 parameter slots. Effective inputs in order: pool2a, pool2b.

Configuration: channels=[128, 128]; epsB=0.02; epsW=[1e-05, 1e-05]; filterSize=[3, 3]; filters=[192, 192]; gpu=1; groups=[1, 1]; initB=0; initBFunc=; initCFunc=; initW=[0.03, 0.03]; initWFunc=; momB=0.9; momW=[0.9, 0.9]; neuron=relu; padding=[-1, -1]; partialSum=13; quantB=default; quantF=default; randSparse=[False, False]; schedB=default; schedW=default; sharedBiases=True; stride=[1, 1]; superEps=0; superMom=0; wball=[0, 0]; wc=[0.0005, 0.0005]; wcNormMax=[0, 0]; wcNormMin=[0, 0]; weightSource=['', ''].

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: Ceiling arithmetic preserves a partial final window. Conv1 has no immediate ReLU; signed responses reach normalization.

[type-conv](card:type-conv) · [pool2a](card:pool2a) · [pool2b](card:pool2b) · [conv3b_neuron](card:conv3b_neuron)

Formula: y[f,y,x] = bias[f] + sum_inputs sum_c,ky,kx W[c,ky,kx,f] * input[c,y*stride+start+ky,x*stride+start+kx]; outside input = 0
