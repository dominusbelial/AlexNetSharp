---
id: pool1a
title: pool1a
summary: Keep the strongest response in each local window.
paper-section: 3.4 / Overlapping pooling
citations: layers/layers-145.cfg:47; layer.py:1164; src/layer.cu:1327
---

This instance has 34,992 output values per image and 0 parameter slots. Effective inputs in order: rnorm1a.

Configuration: channels=48; gpu=0; neuron=relu; outputsX=0; pool=max; quantB=default; quantF=default; sizeX=3; start=0; stride=2.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: It reduces spatial size while allowing small position changes. Pool1 and pool3 have separate following ReLUs; pool2 does not.

[type-pool](card:type-pool) · [rnorm1a](card:rnorm1a) · [pool1a_neuron](card:pool1a_neuron)

Formula: y[c,y,x] = max(input[c, clipped(start+y*stride .. +sizeX-1), clipped(start+x*stride .. +sizeX-1)]); seed = -2e38
