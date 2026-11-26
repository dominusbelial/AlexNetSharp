---
id: conv4a
title: conv4a
summary: Shared filters scan local windows to build feature maps.
paper-section: 3.5 / Overall architecture
citations: layers/layers-145.cfg:163; layer.py:1004; layer.py:1041; src/layer.cu:829
---

This instance has 32,448 output values per image and 331,968 parameter slots. Effective inputs in order: conv3a_neuron.

Configuration: channels=[192]; epsB=0.02; epsW=[1e-05]; filterSize=[3]; filters=[192]; gpu=0; groups=[1]; initB=1; initBFunc=; initCFunc=; initW=[0.03]; initWFunc=; momB=0.9; momW=[0.9]; neuron=relu; padding=[-1]; partialSum=13; quantB=default; quantF=default; randSparse=[False]; schedB=default; schedW=default; sharedBiases=True; stride=[1]; superEps=0; superMom=0; wball=[0]; wc=[0.0005]; wcNormMax=[0]; wcNormMin=[0]; weightSource=[''].

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: Ceiling arithmetic preserves a partial final window. Conv1 has no immediate ReLU; signed responses reach normalization.

[type-conv](card:type-conv) · [conv3a_neuron](card:conv3a_neuron) · [conv4a_neuron](card:conv4a_neuron)

Formula: y[f,y,x] = bias[f] + sum_inputs sum_c,ky,kx W[c,ky,kx,f] * input[c,y*stride+start+ky,x*stride+start+kx]; outside input = 0
