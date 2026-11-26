---
id: rnorm1b
title: rnorm1b
summary: Nearby channels compete at the same image position.
paper-section: 3.3 / Local response normalization
citations: layers/layers-145.cfg:41; layer.py:1183; src/layer.cu:1589
---

This instance has 145,200 output values per image and 0 parameter slots. Effective inputs in order: conv1b.

Configuration: blocked=False; channels=48; gpu=1; minDiv=2; neuron=; pow=0.75; quantB=default; quantF=default; scale=2e-05; size=5.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: This operates on signed conv1 outputs before pooling and ReLU. Negative responses still contribute their squares.

[type-cmrnorm](card:type-cmrnorm) · [conv1b](card:conv1b) · [pool1b](card:pool1b)

Formula: denom = minDiv + (rawScale/size) * sum_channel_window x²; y = x * denom^(-pow)
