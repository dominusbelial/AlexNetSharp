---
id: type-rnorm
title: rnorm
summary: Normalize a spatial neighbourhood within one channel.
paper-section: 3.3 / Code-specific extra stage
citations: layer.py:1183; src/layer.cu:1569
---

Spatial response normalization gathers a square neighbourhood in the same channel. It is a different operation from cross-map normalization.

Why it matters: The released 145 configuration adds this stage after conv2 normalization although the paper does not describe it.

[cnorm2a](card:cnorm2a) · [cnorm2b](card:cnorm2b)

Formula: denom = 1 + (rawScale/size²) * sum_spatial_window x²; y = x * denom^(-pow)
