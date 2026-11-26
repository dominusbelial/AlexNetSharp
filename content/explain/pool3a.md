---
id: pool3a
title: pool3a
summary: Keep the strongest response in each local window.
paper-section: 3.4 / Overlapping pooling
citations: layers/layers-145.cfg:220; layer.py:1164; src/layer.cu:1327
---

This instance has 4,608 output values per image and 0 parameter slots. Effective inputs in order: conv5a.

Configuration: channels=128; gpu=0; neuron=relu; outputsX=0; pool=max; quantB=default; quantF=default; sizeX=3; start=0; stride=2.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot. This is the pool after conv5, often called pool5; the config names it pool3.

Why it matters: It reduces spatial size while allowing small position changes. Pool1 and pool3 have separate following ReLUs; pool2 does not.

[type-pool](card:type-pool) · [conv5a](card:conv5a) · [pool3a_neuron](card:pool3a_neuron)

Formula: y[c,y,x] = max(input[c, clipped(start+y*stride .. +sizeX-1), clipped(start+x*stride .. +sizeX-1)]); seed = -2e38
