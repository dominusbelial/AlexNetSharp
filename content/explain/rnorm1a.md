---
id: rnorm1a
title: rnorm1a
summary: Nearby channels compete at the same image position.
paper-section: 3.3 / Local response normalization
citations: layers/layers-145.cfg:35; layer.py:1183; src/layer.cu:1589
---

This instance has 145,200 output values per image and 0 parameter slots. Effective inputs in order: conv1a.

Configuration: blocked=False; channels=48; gpu=0; minDiv=2; neuron=; pow=0.75; quantB=default; quantF=default; scale=2e-05; size=5.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: This operates on signed conv1 outputs before pooling and ReLU. Negative responses still contribute their squares.

[type-cmrnorm](card:type-cmrnorm) · [conv1a](card:conv1a) · [pool1a](card:pool1a)

Formula: denom = minDiv + (rawScale/size) * sum_channel_window x²; y = x * denom^(-pow)
