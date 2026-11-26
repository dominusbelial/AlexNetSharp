---
id: pool1b
title: pool1b
summary: Keep the strongest response in each local window.
paper-section: 3.4 / Overlapping pooling
citations: layers/layers-145.cfg:56; layer.py:1164; src/layer.cu:1327
---

This instance has 34,992 output values per image and 0 parameter slots. Effective inputs in order: rnorm1b.

Configuration: channels=48; gpu=1; neuron=relu; outputsX=0; pool=max; quantB=default; quantF=default; sizeX=3; start=0; stride=2.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: It reduces spatial size while allowing small position changes. Pool1 and pool3 have separate following ReLUs; pool2 does not.

[type-pool](card:type-pool) · [rnorm1b](card:rnorm1b) · [pool1b_neuron](card:pool1b_neuron)

Formula: y[c,y,x] = max(input[c, clipped(start+y*stride .. +sizeX-1), clipped(start+x*stride .. +sizeX-1)]); seed = -2e38
