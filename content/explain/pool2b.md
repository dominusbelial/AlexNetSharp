---
id: pool2b
title: pool2b
summary: Keep the strongest response in each local window.
paper-section: 3.4 / Overlapping pooling
citations: layers/layers-145.cfg:127; layer.py:1164; src/layer.cu:1327
---

This instance has 21,632 output values per image and 0 parameter slots. Effective inputs in order: cnorm2b.

Configuration: channels=128; gpu=1; neuron=; outputsX=0; pool=max; quantB=default; quantF=default; sizeX=3; start=0; stride=2.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: It reduces spatial size while allowing small position changes. Pool1 and pool3 have separate following ReLUs; pool2 does not.

[type-pool](card:type-pool) · [cnorm2b](card:cnorm2b) · [conv3a](card:conv3a) · [conv3b](card:conv3b)

Formula: y[c,y,x] = max(input[c, clipped(start+y*stride .. +sizeX-1), clipped(start+x*stride .. +sizeX-1)]); seed = -2e38
