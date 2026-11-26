---
id: labels
title: labels
summary: The image or class index entering the graph.
paper-section: 2 / Dataset
citations: layers/layers-145.cfg:5; layer.py:1099; convdata_jpeg.py:73
---

This instance has 1 output values per image and 0 parameter slots. Effective inputs in order: none.

Configuration: dataIdx=1; gpu=-1; quantB=default; quantF=default.

Live shape, activation statistics and parameter values are supplied by the loaded snapshot.

Why it matters: Both towers see the same crop. A synthetic label is only a cost demonstration.

[type-data](card:type-data) · [logprob](card:logprob)

Formula: x[c,y,x] = RGB[c,y,x] - mean[c]; label = 0
