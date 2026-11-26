---
id: type-cmrnorm
title: cmrnorm
summary: Nearby channels compete at the same image position.
paper-section: 3.3 / Local response normalization
citations: layer.py:1183; src/layer.cu:1589
---

Cross-map response normalization divides each response by a power of the squared responses in a channel window. Its configured base is 2.

Why it matters: This operates on signed conv1 outputs before pooling and ReLU. Negative responses still contribute their squares.

[rnorm1a](card:rnorm1a) · [rnorm1b](card:rnorm1b) · [rnorm2a](card:rnorm2a) · [rnorm2b](card:rnorm2b)

Formula: denom = minDiv + (rawScale/size) * sum_channel_window x²; y = x * denom^(-pow)
