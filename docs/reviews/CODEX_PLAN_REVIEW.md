# Codex review of the AlexNetSharp plan (2026-11-24, read-only, original-source citations)

## Assessment

**The plan should first choose a specific, documented experiment to reproduce.** `layers-145.cfg` has the recognizable AlexNet topology, but it is not identical to the paper’s described network, and `run4.sh` does not supply its training schedule. The released code supports a strong reconstruction; it does not by itself establish every detail of the historical winning run.

### 1. Faithfulness

- **Network:** `layers-145` has the expected tower widths and cross-tower connections, but adds a *spatial* `rnorm` after conv2’s cross-map normalization ([layers-145.cfg:95](layers/layers-145.cfg:95), [layers-145.cfg:107](layers/layers-145.cfg:107)). The [paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) describes cross-map normalization, not that extra stage. Also, conv1 has no ReLU in its section: normalization precedes pooling, whose `neuron=relu` is applied there ([layers-145.cfg:9](layers/layers-145.cfg:9), [layers-145.cfg:35](layers/layers-145.cfg:35), [layers-145.cfg:47](layers/layers-145.cfg:47)). Verify the framework’s inserted-neuron ordering; don’t silently rewrite it as conventional conv→ReLU→LRN.

- **224 versus 227:** The design’s assertion that code and paper disagree on the 55×55 shape is too strong. The original parser uses **ceiling**, yielding 55 from 224, with a partial boundary window ([layer.py:1004](layer.py:1004)). Confirm the missing kernel’s boundary treatment; standard floor-based convolution on 227 is not automatically equivalent. Crop size depends on metadata and `crop-border`; the default border is 4, not evidence of 224 ([convnet.py:182](convnet.py:182), [convdata_jpeg.py:71](convdata_jpeg.py:71)).

- **Normalization:** `cmrnorm` crosses channels; `rnorm` is spatial. The parser divides configured scale by `size` or `size²` respectively ([layer.py:1179](layer.py:1179)); `cmrnorm` also has `minDiv` and `blocked` semantics ([layer.cu:1583](src/layer.cu:1583)). PyTorch LRN defaults are not an oracle for these choices. The actual cudaconv2 sources are absent from this checkout, despite the build expecting them ([Makefile-distrib:16](Makefile-distrib:16)). Locate and version the matching dependency before claiming kernel parity; a later cuda-convnet2 version requires evidence of equivalence.

- **Dropout and initialization:** `hs` uses an unscaled random mask in training and multiplies by keep probability in evaluation ([layer.cu:1121](src/layer.cu:1121)); modern inverted dropout changes intermediate activations. Weights use Gaussian standard deviations from the config, including 0.03 for conv3–5; default bias is zero, with explicit ones elsewhere ([layer.py:807](layer.py:807), [layer.py:832](layer.py:832)). Preserve separate matrices and their input order, especially `fc2048bb`.

- **Schedule:** [run4.sh:3](run4.sh:3) resumes `layers-120-4gpu` checkpoints; its auto files are not a fresh `145` recipe. The `145` parameter file instead records multiple runs and interventions, including frozen conv1 and changed color noise ([layer-params-145.cfg:165](layers/layer-params-145.cfg:165)). Its current zero conv1 learning rate is **not** a sensible initial setting. Specify the chosen starting rates, historical uncertainties and exact epoch transitions as an experiment manifest.

- **Data:** The JPEG provider consumes already prepared images; the [paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) specifies resize-short-side to 256 then center crop. The provider uses a centered mean image, RGB planar values and shared crop/flip choices in chunks of 32 ([convdata_jpeg.py:95](convdata_jpeg.py:95), [convdata_jpeg.py:237](convdata_jpeg.py:237)). PCA noise is added *then the centered image is divided by* `1 + color_noise_coeff` ([convdata_jpeg.py:154](convdata_jpeg.py:154)). Ten-view testing uses five positions and mirrors ([convdata_jpeg.py:221](convdata_jpeg.py:221)); average **probabilities**, then score. `avg-valid.py` additionally averages prediction files from different models, not just crops ([avg-valid.py:27](avg-valid.py:27)).

The design’s 40.7%/18.2% are **error rates**, not accuracies. The config comments record approximately 40.664%/18.210% for one run and 40.25%/17.994% for a later multiview result ([layer-params-145.cfg:180](layers/layer-params-145.cfg:180)). Dataset, checkpoint and evaluation protocol must match before comparison.

### 2. Verification

The three proposed checks are necessary, but can agree on the same wrong interpretation. Add:

- Tiny, hand-computed golden cases for boundary windows, normalization edges, pool ties, multi-input gradient accumulation, dropout masks and softmax loss. Compare forward, input/weight/bias gradients and **two or more updates**, including batch averaging, momentum, bias rates and decay ([weights.cu:145](src/weights.cu:145)).
- A separately written scalar reference, plus fixtures from the matching original kernels if obtainable. This repo’s [cpuCNN.cu:10](src/cpuCNN.cu:10) implements softmax-tree operations, **not** a CPU reference for AlexNet convolution and training.
- Deterministic augmentation fixtures and a complete small graph compared layer-by-layer with injected weights, inputs and masks. Test uninterrupted versus checkpoint/resume, including optimizer, RNG, data order and schedule.
- Overfit a fixed tiny dataset; then compare loss, gradients and updates over many identical batches against the independent reference, with stated tolerances.

Without original CUDA outputs or trusted historical checkpoints, this establishes **implementation and trajectory agreement with an independently reconstructed specification**, not proof of bitwise or historical training parity. Floating-point order and random streams will diverge.

### 3. Feasibility

A rough full AlexNet forward pass is ~0.72 billion multiply-adds; forward plus two backward passes is roughly **4.3 billion FLOPs/image**. At 1.2 million images × 90 epochs, that is ~**4.6 × 10¹⁷ FLOPs**, excluding normalization, augmentation and validation. Sustained end-to-end throughput of 10, 50 or 500 GFLOP/s implies approximately **538, 108 or 11 days**, respectively. These are scenarios, not predictions: CPU model, SIMD, memory bandwidth and implementation dominate. Benchmark the actual full training step early; 24 cores alone cannot determine runtime.

~60 million FP32 parameters are ~240 MB; weights, gradients and momentum alone are ~720 MB. One batch of 128 conv1 activations at 96×55×55 is ~149 MB; retained activations, gradients and convolution workspace can bring the process into **several GB**, potentially much more with naïve im2col. Measure peak RSS.

A 256×256 RGB uint8 cache for 1.2 million images is ~236 GB decimal; FP32 is ~944 GB. Retain original JPEGs, optionally use a documented resized JPEG cache, and stream/decode with bounded buffers. Record interpolation, JPEG encoder, class mapping, mean/PCA metadata and checksums: preprocessing is part of the experiment.

A fixed, stratified **100-class subset is worthwhile** for integration, learning curves and throughput; preserve 1000 outputs for the closest structural test or explicitly label a 100-output variant. Neither proves full-ImageNet accuracy.

### 4. Explorability

Show receptive field and contributing pixels for a selected activation; animate one numerical gradient through fan-out and accumulation; compare train/eval dropout and augmentation side by side. Add an experiment timeline with schedule changes, uncertainty/source links, checkpoint provenance, and a replayable sample including its crop and RNG seed. Provide sampled statistics and a pause/snapshot mode: copying every activation several times per second can materially slow training, contrary to the design’s guarantee.

### 5. Ordered stories and gates

Each story should have a bounded brief, fixtures, acceptance command and reviewer sign-off; the complete work is larger than these individual slices.

1. **Specification ledger:** cited graph, unresolved kernel/data assumptions; gate: machine-readable completeness check.
2. **Config parser:** defaults, lists and errors; gate: golden parse of both 145 files.
3. **Shape inference:** ceiling and graph fan-in; gate: boundary and full-network shape fixtures.
4. **Tensor/layout primitives:** gate: indexed round trips and accumulation cases.
5. **Scalar convolution:** gate: hand fixtures including partial edges and gradients.
6. **Optimized convolution:** gate: randomized parity with scalar reference and benchmark.
7. **Pooling/ReLU:** gate: edges, ties and backward fixtures.
8. **Both normalizations:** gate: independently derived forward/backward and gradient checks.
9. **FC and fan-in:** gate: two-input forward/gradient parity.
10. **Dropout, softmax, loss:** gate: fixed-mask train/eval and stable-loss fixtures.
11. **Optimizer and initialization:** gate: seeded statistics and explicit multi-step update fixtures.
12. **Data preparation:** gate: fixed images, labels, mean/PCA and checksum fixtures.
13. **Augmentation/evaluation:** gate: injected random choices and ten-view probability fixtures.
14. **Trainer/checkpoints:** gate: tiny-set overfit and exact deterministic resume.
15. **Full graph integration:** gate: layer-by-layer oracle and sustained batch benchmark.
16. **Visualization slice:** gate: automated hit-testing, card content/citations and snapshot consistency.
17. **Subset pilot:** gate: reproducible learning curves, throughput, memory and restart report.
18. **Full-run readiness:** gate: frozen manifest, disk/RAM estimates, validation protocol and documented remaining parity gaps.

### 6. Direct disagreements

Do not call `145` plus `run4.sh` **the** paper experiment. Do not make full ImageNet training contingent on only per-layer gradient checks, or promise that rendering cannot perturb training. Finally, “speed is not a goal” cannot mean ignoring throughput: a correct implementation that needs years changes the project’s feasibility. Measure a representative step before committing to the full run.