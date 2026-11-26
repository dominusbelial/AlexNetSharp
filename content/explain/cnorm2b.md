---
id: cnorm2b
title: cnorm2b
summary: Normalize a spatial neighbourhood within one channel.
paper-section: 3.3 / Code-specific extra stage
citations: layers/layers-145.cfg:113; layer.py:1183; src/layer.cu:1569
---

This instance has 93,312 output values per image and 0 parameter slots. Effective inputs in order: rnorm2b.

Configuration: blocked=False; channels=128; gpu=1; neuron=; pow=0.75; quantB=default; quantF=default; scale=4e-05; size=5.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: The released 145 configuration adds this stage after conv2 normalization although the paper does not describe it.

[type-rnorm](card:type-rnorm) · [rnorm2b](card:rnorm2b) · [pool2b](card:pool2b)

Formula: denom = 1 + (rawScale/size²) * sum_spatial_window x²; y = x * denom^(-pow)
