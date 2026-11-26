---
id: rnorm2b
title: rnorm2b
summary: Nearby channels compete at the same image position.
paper-section: 3.3 / Local response normalization
citations: layers/layers-145.cfg:101; layer.py:1183; src/layer.cu:1589
---

This instance has 93,312 output values per image and 0 parameter slots. Effective inputs in order: conv2b_neuron.

Configuration: blocked=False; channels=128; gpu=1; minDiv=2; neuron=; pow=0.75; quantB=default; quantF=default; scale=2e-05; size=5.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: This operates on signed conv1 outputs before pooling and ReLU. Negative responses still contribute their squares.

[type-cmrnorm](card:type-cmrnorm) · [conv2b_neuron](card:conv2b_neuron) · [cnorm2b](card:cnorm2b)

Formula: denom = minDiv + (rawScale/size) * sum_channel_window x²; y = x * denom^(-pow)
