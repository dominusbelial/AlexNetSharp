---
id: type-data
title: data
summary: The image or class index entering the graph.
paper-section: 2 / Dataset
citations: layer.py:1099; convdata_jpeg.py:73
---

Image data is planar RGB. The labels stream is a class index. This replay uses a real CC0 Labrador photograph and synthetic label 0.

Why it matters: Both towers see the same crop. A synthetic label is only a cost demonstration.

[data](card:data) · [labels](card:labels)

Formula: x[c,y,x] = RGB[c,y,x] - mean[c]; label = 0
