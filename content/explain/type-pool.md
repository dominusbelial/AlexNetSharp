---
id: type-pool
title: pool
summary: Keep the strongest response in each local window.
paper-section: 3.4 / Overlapping pooling
citations: layer.py:1164; src/layer.cu:1327
---

Max pooling takes the largest real input in a clipped window, separately for each channel. In this configuration the windows overlap.

Why it matters: It reduces spatial size while allowing small position changes. Pool1 and pool3 have separate following ReLUs; pool2 does not.

[pool1a](card:pool1a) · [pool1b](card:pool1b) · [pool2a](card:pool2a) · [pool2b](card:pool2b) · [pool3a](card:pool3a) · [pool3b](card:pool3b)

Formula: y[c,y,x] = max(input[c, clipped(start+y*stride .. +sizeX-1), clipped(start+x*stride .. +sizeX-1)]); seed = -2e38
