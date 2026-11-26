---
id: conv1b
title: conv1b
summary: Shared filters scan local windows to build feature maps.
paper-section: 3.5 / Overall architecture
citations: layers/layers-145.cfg:22; layer.py:1004; layer.py:1041; src/layer.cu:829
---

This instance has 145,200 output values per image and 17,472 parameter slots. Effective inputs in order: data.

Configuration: channels=[3]; epsB=0; epsW=[0]; filterSize=[11]; filters=[48]; gpu=1; groups=[1]; initB=0; initBFunc=; initCFunc=; initW=[0.01]; initWFunc=; momB=0.9; momW=[0.9]; neuron=; padding=[0]; partialSum=5; quantB=default; quantF=default; randSparse=[False]; schedB=default; schedW=default; sharedBiases=True; stride=[4]; superEps=0; superMom=0; wball=[0]; wc=[0.0005]; wcNormMax=[0]; wcNormMin=[0]; weightSource=[''].

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: Ceiling arithmetic preserves a partial final window. Conv1 has no immediate ReLU; signed responses reach normalization.

[type-conv](card:type-conv) · [data](card:data) · [rnorm1b](card:rnorm1b)

Formula: y[f,y,x] = bias[f] + sum_inputs sum_c,ky,kx W[c,ky,kx,f] * input[c,y*stride+start+ky,x*stride+start+kx]; outside input = 0
