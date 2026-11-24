# AlexNetSharp — the real 2012 AlexNet, trained in C#, explored live in the browser

## Goal

Run the **real thing**: the network, layer maths, training procedure and data of the 2012 ImageNet winner, taken from
the original source release (`computerhistory/AlexNet-Source-Code`, BSD-style, Google 2023). Render the process in a
**web build** (MonoGame API via KNI/BlazorGL, as in the AllegianceCore browser client; no desktop client) and make every
part **easy to explore and explained on screen**. Speed is not the goal, but throughput decides feasibility, so it is
measured early (story 15).

Reviewed by Codex on 2026-11-24 (`docs/reviews/CODEX_PLAN_REVIEW.md`); its corrections are folded in below.

## Which experiment ("the real thing", precisely)

- The network is `layers/layers-145.cfg` + `layers/layer-params-145.cfg`: the paper's topology (two towers,
  48/128/192/192/128 filters, cross-tower conv3 and FC, 4096-4096-1000). It differs from the paper's prose in places,
  e.g. an extra **spatial** `rnorm` after conv2's cross-map norm (`layers-145.cfg:95`, `:107`) and the conv1 order
  conv → cmrnorm → pool(`neuron=relu`). **The code is the specification; differences are kept and shown, never
  "corrected".**
- `layer-params-145.cfg` records several historical runs and hand interventions (conv1 later frozen, colour noise
  changed, `:165`). `run4.sh` belongs to `layers-120-4gpu`, not 145. Story 1 writes an **experiment manifest**: the
  chosen starting rates, every epoch transition, and each historical uncertainty with its source line.
- Target result: the config comments record ≈40.66% top-1 / 18.21% top-5 **error** for one run and 40.25% / 17.99%
  multiview (`layer-params-145.cfg:180`). Compared only with a matching dataset, checkpoint and evaluation protocol.

## Faithfulness rules (from the original source)

- Shapes: the original parser's **ceiling** arithmetic (224 → 55 with a partial boundary window, `layer.py:1004`).
- Normalisation: `cmrnorm` (cross-map) and `rnorm` (spatial) are different; the parser divides scale by `size` resp.
  `size²` (`layer.py:1179`); `cmrnorm` has `minDiv`/`blocked` semantics (`layer.cu:1583`). PyTorch LRN defaults are
  **not** an oracle for these.
- Dropout (`hs`): unscaled random mask in training, multiply by `keep` at evaluation (`layer.cu:1121`).
- Init: Gaussian std from the config (0.03 for conv3–5), bias 0 unless `initB` (`layer.py:807`, `:832`); multi-input
  layers keep separate weight matrices in input order (`fc2048bb` reads `hs1b,hs1a`).
- Optimiser: batch averaging, momentum, separate bias rates, weight decay exactly as `weights.cu:145`.
- Convolution kernels (cudaconv2 `filter_acts/weight_acts/img_acts`) are **not** in the release (`Makefile-distrib:16`).
  Story 1 locates the matching version (cuda-convnet / cuda-convnet2) and records the evidence of equivalence, including
  boundary-window behaviour.
- Data: resize short side to 256 then centre crop (paper); mean image subtraction, RGB planar, crop/flip choices shared
  in chunks of 32 (`convdata_jpeg.py:95`, `:237`); PCA noise added then divided by `1 + color_noise_coeff` (`:154`);
  ten-view test = 5 positions × mirror, average **probabilities** (`:221`). Preprocessing settings (interpolation, JPEG
  encoder, class map, mean/PCA, checksums) are part of the experiment and recorded.

## Verification

1. Cited original logic for every behaviour.
2. A separately written **scalar reference** (plain loops) for every layer; optimised code must match it.
3. **Hand-computed golden cases**: boundary windows, normalisation edges, pool ties, multi-input gradient accumulation,
   fixed dropout masks, stable softmax loss; forward, input/weight/bias gradients and **two or more updates**.
4. Numerical gradient checks.
5. A small full graph with injected weights, inputs and masks compared layer by layer with the references.
6. Deterministic augmentation fixtures; uninterrupted vs checkpoint/resume identical (optimiser, RNG, data order).
7. Overfit a fixed tiny dataset; compare loss/gradients/updates over many batches with the reference.

What this proves: agreement with an independently reconstructed specification, **not** bitwise historical parity (no
CUDA original or historical checkpoint exists to compare against). The app says so.

## Explorable by design

Every visible thing is hoverable and opens an **explain card**: what it is in plain words, its live numbers, why it
matters, the paper section, and the original `file:line`. Hover = one line; click = pinned card (several can be pinned
and compared); "deeper" = the formula with live values. Cards link to related things.

Also (from the review): receptive field and contributing pixels of a selected activation; one gradient animated through
fan-out and accumulation; train vs eval (dropout, augmentation) side by side; an experiment timeline (schedule changes,
uncertainties, checkpoint provenance); a replayable sample with its crop and RNG seed; a guided tour photo → … → loss →
backprop → update. Explanations are data (`content/explain/*.md`) with citations.

## Architecture (web)

- `AlexNetSharp.Core` — no graphics dependency: `.cfg` parser, shape inference, layers (scalar reference + optimised),
  optimiser, data pipeline, checkpoints. Runs in .NET (trainer) and in WebAssembly (browser inference).
- `AlexNetSharp.Trainer` — headless .NET process that does the real training (on a server/lane; ImageNet cannot be
  trained in a browser). Publishes **sampled** snapshots (one tracked example, filters, gradient statistics, curves) at
  a bounded rate and a pause/snapshot mode; snapshot cost is measured, not assumed free. Writes a run log + checkpoints.
- `AlexNetSharp.Web` — the browser app (KNI BlazorGL, MonoGame API): 3D layer stack, filter grids, gradient flow,
  dashboards, explain cards, tour. Two sources:
  - **live**: WebSocket to a running trainer;
  - **replay**: recorded run logs + checkpoints served as static files, so a public page works with no trainer running.
  It also runs **inference in the browser** on a downloaded checkpoint for the visitor's own photo (FP16 weights ≈120 MB,
  loaded on demand).
- `AlexNetSharp.Tests` — references, goldens, gradient checks, parser/shape, web hit-testing and card citations.

## Data (not in git)

- `data/` — ImageNet ILSVRC-2012 (≈150 GB JPEGs; Domingo downloads with his own account). A 256×256 uint8 cache would
  be ≈236 GB; keep the JPEGs, optionally a documented resized-JPEG cache, stream with bounded buffers.
- A fixed, stratified **100-class subset** for integration, curves and throughput (1000 outputs kept, or a labelled
  100-output variant).
- `.scratch/` — oracle venv (CPU PyTorch, test-only), scratch outputs. `checkpoints/` — training state.

## Feasibility

Forward ≈0.72 G multiply-adds per image; forward + backward ≈4.3 GFLOP. 1.2M images × 90 epochs ≈ 4.6×10¹⁷ FLOP →
≈538 / 108 / 11 days at a sustained 10 / 50 / 500 GFLOP/s. Weights + gradients + momentum ≈720 MB; activations for a
batch of 128 run to several GB. Measure the real step (story 15) before committing to the full run; an ILGPU/OpenCL
backend on the Arc B50 is the fallback if CPU throughput is too low.

## Stories (delegated, one at a time, Claude reviews each)

1. Specification ledger + experiment manifest + cudaconv2 version evidence (machine-checked completeness).
2. Config parser (defaults, lists, errors; golden parse of both 145 files).
3. Shape inference (ceiling, fan-in; full-network shape fixtures).
4. Tensor/layout primitives (round trips, accumulation).
5. Scalar convolution (hand fixtures incl. partial edges and gradients).
6. Optimised convolution (randomised parity with scalar; benchmark).
7. Pooling + ReLU (edges, ties, backward).
8. Both normalisations (independent forward/backward + gradient checks).
9. FC + multi-input fan-in.
10. Dropout, softmax, loss (fixed masks, train/eval, stable loss).
11. Optimiser + initialisation (seeded stats, multi-step update fixtures).
12. Data preparation (fixed images, labels, mean/PCA, checksums).
13. Augmentation + ten-view evaluation.
14. Trainer + checkpoints (tiny-set overfit, exact deterministic resume).
15. Full-graph integration + sustained throughput and memory benchmark.
16. Web slice: KNI browser app, replay source, layer stack + explain cards (automated hit-testing, card citations).
17. Live source (trainer WebSocket) + in-browser inference on a checkpoint.
18. 100-class subset pilot (curves, throughput, memory, restart report).
19. Full-run readiness (frozen manifest, disk/RAM, validation protocol, documented parity gaps).
