# Experiment manifest — layers-145 (readable version of `experiment.json`)

Source of truth: `docs/spec/experiment.json` (every value cited `path:line` into the original release and checked by
`tests/AlexNetSharp.Tests`). Paths below are relative to the original tree (`computerhistory/AlexNet-Source-Code`
`2f8b81f`).

## What the parameter file is

`layers/layer-params-145.cfg` is **not a recipe**. Its values are the state of the file after the last hand edit, and its
comments (`:165-203`) log three runs:

| Run | What happened | Result |
|---|---|---|
| **run1** (`:165-182`) | Fresh start. epsW 0.01 → 0.001 at epoch 25 → 0.0001 at 51; conv1 → 0.00001 and colour noise 0.1 → 0 at 67; conv1 frozen (0) at 72; rest → 0.00001 at 79; killed at 91. Paused at 36 and 58, moved hosts. | logprob 1.741, **40.66 % top-1 / 18.21 % top-5 error** (`:182`, views not stated) |
| **run1-continued** (`:192-203`) | Same checkpoint, rate pushed back up to 0.001 (conv1 still 0), colour noise back to 0.1; 0.0001 at 107; 0.00001 and noise 0 at 124; killed at 135. | logprob 1.726, **40.25 % / 17.99 %**, multiview (`:202-203`) |
| **2012-full** (`:184-190`) | Separate run; 0.01 → 0.001 at 19 → 0.0001 at 47; log ends at 61. | none recorded |

The file's current values (conv1 epsW 0, others 0.00001) are run1's **final** state, not a starting point.

## Per-layer rates

All 15 weighted layers: momW = momB = 0.9, wc = 0.0005, wball = 0, epsB = 0.02 (conv1: 0.0). Multi-input layers
(`conv3a/b`, `fc2048a/b/ba/bb`, `fc1000`) have one epsW/momW/wc/wball value **per input**, in input order
(`layer.py:728-737`). There is no automatic schedule (`schedW` default, `layer.py:739`): every change was a manual edit
between restarts. A layer with epsW 0 is not updated at all, so momentum and weight decay stop too
(`src/weights.cu:148`).

## Data and evaluation

- Minibatch 128 (`convnet.py:177`). Epochs: default 50000, i.e. run until stopped by hand (`convnet.py:196`).
- Crop: `inner = img_size − 2·crop_border` (`convdata_jpeg.py:73`); default border 4 (`convnet.py:182`), but the paper's
  224 from 256 implies 16 — open (ledger `OQ-LEDGER-1`).
- Mean: per-pixel mean image from `batches.meta`, centre-cropped and subtracted (`convdata_jpeg.py:95`, `:210`).
- Training crop: one random offset in [0, 2·border] per axis and one 50 % flip **per chunk of 32 images**
  (`convdata_jpeg.py:77`, `:239`, `:244`).
- Colour noise: per image `(randn(3) × PCA stdevs) · eigvecsᵀ` (`:65`), added × coefficient to the mean-subtracted crop,
  then the crop is divided by `1 + coefficient` (`:177-179`). Training only.
- Test: centre crop; multiview = 4 corners + centre, each mirrored = 10 views (`:75`, `:226`), probabilities averaged.

## Proposed fresh run

Replay **run1**: it is the only run that starts from scratch, it has three rate drops in about 90 epochs, and its result
matches the paper's single-CNN 40.7 % / 18.2 %.

- Start: epsW 0.01 on every weighted layer (conv1 included), epsB 0.02, momentum 0.9, wc 0.0005, colour noise 0.1,
  minibatch 128, crop 256 → 224.
- Schedule: epoch 25 → 0.001; 51 → 0.0001; 79 → 0.00001; stop at 91. The conv1 steps at 67 (0.00001, noise 0) and 72
  (freeze) are marked optional.
- Two variants: **literal** (every run1 transition at its recorded epoch) and **plateau** (the paper's rule: divide by 10
  when validation error stops improving; record the epochs used). The original epochs were hand decisions on plateaus, so
  a run with a different data order may plateau elsewhere.

## Uncertainties

| Id | Topic | Detail |
|---|---|---|
| OQ-EXP-1 | Starting rates | 0.01 and colour noise 0.1 are inferred from the first "from" values. epsB 0.02 is never changed in the log; whether bias rates were also lowered is unknown. |
| OQ-EXP-2 | Pauses | run1 paused at 36 and 58 and moved hosts; RNG state and data order across restarts are unrecorded. |
| OQ-EXP-3 | Views of run1's result | `:182` is unlabelled; assumed single-view (the later result at `:202` is labelled multiview). |
| OQ-EXP-4 | Missing framework | `convnet.py` imports `gpumodel`, `options`, `data`, `util` (`convnet.py:4-7`), none in the release: epoch loop, test frequency and checkpoint behaviour are unknown. |
| OQ-EXP-5 | Dataset preparation | `batches.meta` (img_size, data_mean, color_pca, batch_size) and the batch-building script are absent; resize, source size and PCA statistics must be reconstructed and recorded. |
| OQ-EXP-6 | "2012-full" | Probably train+val or another variant; no result; not the target. |
| OQ-EXP-7 | `run4.sh` | Resumes `layers-120-4gpu` checkpoints (`run4.sh:3`); context only. |
