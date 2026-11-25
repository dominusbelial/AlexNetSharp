# Specification Ledger — `layers/layers-145.cfg` + `layers/layer-params-145.cfg` (`ANX-1`)

Machine-checked companion: `docs/spec/ledger.json` (upstream `computerhistory/AlexNet-Source-Code` commit `2f8b81f`).
Kernel-level evidence for omitted `cudaconv2` sources: `docs/spec/CUDACONV2.md`.

---

## 1. Critical Faithfulness Rules (Codex Review & Original Source)

1. **Ceiling shape arithmetic & partial boundary window (`224 -> 55`):**
   - `LocalLayerParser.parse` (`layer.py:1004`) computes `modulesX = 1 + int(ceil((2 * padding + imgSize - filterSize) / float(stride)))` and negates `padding` to `paddingStart = -padding` (`layer.py:1041`).
   - `PoolLayerParser.parse` (`layer.py:1164`) computes `outputsX = int(ceil((imgSize - start - sizeX) / float(stride))) + 1` when `outputsX <= 0`.
   - On a `224x224` input crop, `conv1a` / `conv1b` (`filterSize=11, padding=0, stride=4`) yield `1 + ceil(213 / 4.0) = 55`. Module index `54` spans `[216..226]`, reading pixels `216..223` and **zero-padding 3 pixels on the right and bottom only** (`CUDACONV2.md` §2).

2. **Inserted-neuron detachment & `conv1` / `conv5` ReLU ordering:**
   - Any non-`neuron` section declaring `neuron=<spec>` causes `LayerParser.detach_neuron_layers` (`layer.py:272-275`, `:627-646`) to synthesize a separate `type=neuron` layer `<name>_neuron` and rewrite every downstream consumer's `inputs` from `<name>` to `<name>_neuron`.
   - **`conv1a` / `conv1b` (`layers/layers-145.cfg:9, 22`) have NO `neuron=relu`:** cross-map normalization `rnorm1a/b` runs on raw signed `conv1` activations (`layers/layers-145.cfg:35, 41`), followed by `pool1a/b` (`max`, `:47, 56`), whose `neuron=relu` (`:54, 63`) runs *after* pooling (`conv -> cmrnorm -> pool -> relu`). Because `x_f^2` in `cmrnorm` includes negative pre-ReLU responses, this is **not** mathematically equivalent to `conv -> relu -> cmrnorm -> pool`.
   - **`conv5a` / `conv5b` (`layers/layers-145.cfg:192, 206`) also omit `neuron=relu`**, deferring `neuron=relu` to `pool3a/b` (`:227, 236`). Since `max_w(max(0, x)) == max(0, max_w(x))`, `conv5 -> pool3 -> relu` matches `conv5 -> relu -> pool3` while computing ReLU on 36 pixels/channel instead of 169.

3. **Two distinct normalizations (`cmrnorm` vs `rnorm`) & extra spatial stage after `conv2`:**
   - `NormLayerParser.add_params` (`layer.py:1179-1187`) divides configured `scale` by **`size`** for `cmrnorm` (`CROSSMAP_RESPONSE_NORM`) and by **`size**2`** for spatial `rnorm` (`RESPONSE_NORM`).
   - `CrossMapResponseNormLayer` (`src/layer.cu:1583-1594`) parses `blocked` (default `False`, `layer.py:1193`) and `minDiv` (default `1.0`, `layer.py:1186`; configured to **`2`** on `rnorm1a/b` and `rnorm2a/b` at `layers/layer-params-145.cfg:140, 145, 150, 155`):
     `denom[c, y, x] = minDiv + (scale / size) * sum_{f=max(0, c - size/2)}^{min(C, c - size/2 + size) - 1} x[f, y, x]^2`, `y = x * denom^(-pow)`.
   - **Extra spatial normalization not in the NIPS 2012 paper:** after `rnorm2a` / `rnorm2b` (`layers/layers-145.cfg:95, 101`), `layers-145.cfg` inserts `[cnorm2a]` and `[cnorm2b]` of `type=rnorm` (`layers/layers-145.cfg:107-118`, `layers/layer-params-145.cfg:157-163`, `scale=0.001 -> 0.001/25 = 0.00004`, `pow=0.75`). `ResponseNormLayer` (`src/layer.cu:1560-1574`) does **not** use `minDiv`; its denominator is `1.0f + (scale / size^2) * sum_{(y',x') in W_5x5(y,x)} x[c, y', x']^2`.

4. **Unscaled `hs` dropout (`HiddenSexLayer`):**
   - `HiddenSexLayer` (`src/layer.cu:1121-1153`, `layer.py:689-709`) multiplies by an **unscaled** Bernoulli mask `u < keep` (`keep=0.5`) when `_enable && passType == PASS_TRAIN`, and multiplies activations/gradients by `keep` (`0.5`) during evaluation (or whenever `_enable == false`).

5. **Weight/bias initialization & multi-input matrix order (`fc2048bb`):**
   - `WeightLayerParser.make_weights` (`layer.py:807`) allocates one independent `float32` matrix per input in declared `inputs` order: `initW[i] * nr.randn(rows[i], cols[i])` (`order='C'` shape `(filterChannels*filterSize^2, filters)` for `conv`, `layer.py:1072`; `order='F'` shape `(numInputs[i], outputs)` for `fc`, `layer.py:883`).
   - **Asymmetric input order:** `[fc2048ba]` reads `inputs=hs1a,hs1b` (`layers/layers-145.cfg:268`), whereas `[fc2048bb]` reads **`inputs=hs1b,hs1a`** (`layers/layers-145.cfg:277`) so each GPU places its local input at index `0`.
   - Biases initialize to `initB * ones(...)` (`layer.py:824`), where `initB` defaults to `0` (`layer.py:833`): `initB=0` (default) on `conv1a/b`, `conv3a/b`, `fc1000`; explicit `initB=1` on `conv2a/b`, `conv4a/b`, `conv5a/b`, `fc2048a/b`, `fc2048ba/bb`.
   - **Parser quirk (`layer.py:832` vs `:113`):** `WeightLayerParser.parse` passes float `default=0.01` to `safe_get_float_list(name, 'initW', default=0.01)`, which attempts `0.01.split(',')` and raises `LayerParsingError` if `initW` is omitted on an `fc` layer (`LocalLayerParser.parse` overrides with `default=None` at `layer.py:986`).

6. **Negative-gradient sign convention & optimizer update (`src/weights.cu:145-198`):**
   - `kLogregSoftmaxGrad` (`src/layer_kernels.cu:464`) computes **`coeff * ((label == c) - p_c)`**, i.e. the **negative** loss derivative `-dE/dz`.
   - With `g = -(1 / numCases) * sum_n dE_n/dW` (`src/layer.cu:654, 668, 846, 861`) and `superEps <= 0`, `Weights::update` (`src/weights.cu:146-198`) updates:
     `V <- momW * V + epsW * g - wc * epsW * W`, then `W <- W + V`.
   - Biases (`src/layer.cu:565`) always pass `wc = 0, wball = 0, useGrad = true`: `V_b <- momB * V_b + epsB * g_b`, `B <- B + V_b`.
   - Unshared `FCLayer` (`src/convnet.cu:444`, `useGrad = false`) folds `momW` and `epsW / numCases` directly into `_weightsInc` during `FCLayer::bpropWeights` (`src/layer.cu:668-671`), after which `Weights::update` subtracts `wc * epsW * W` and adds `_weightsInc` into `_weights` (algebraically identical).

---

## 2. Layer & Neuron Type Reference

All non-data layers also inherit `neuron=""` (`layer.py:374`), `gpu=inputLayers[0]['gpu']` (`layer.py:378`), and `quantF=default`, `quantB=default` (`layer.py:177-178`).

| Type | Inputs | Required `.cfg` / `params.cfg` keys | Parser defaults when absent (`layer.py`) | Forward & Backward Summary | Key Citations |
| --- | --- | --- | --- | --- | --- |
| `data` | 0 | `dataIdx` | `gpu=-1` (`:223`), `quantF="default"`, `quantB="default"` (`:177-178`) | **Fwd:** Copies host minibatch `dataIdx` (`0` images, `1` labels) to consumer GPUs (`startFprop`). **Bwd:** None (`isGradProducer() == false`). | `layer.py:1092-1102`, `src/layer.cu:1167-1266` |
| `conv` | `>= 1` | `inputs`, `channels`, `filters`, `filterSize`, `initW`, `partialSum`; `epsW`, `epsB`, `momW`, `momB` | `padding=[0]*N` (negated at `:1041`), `stride=[1]*N`, `groups=[1]*N`, `randSparse=[False]*N`, `initB=0`, `initWFunc=""`, `initBFunc=""`, `initCFunc=""`, `weightSource=['']*N`, `sharedBiases=True` (`:1064`), `wc=[0.0]*N`, `wball=[0.0]*N`, `wcNormMin=[0.0]*N`, `wcNormMax=[0.0]*N`, `superEps=0.0`, `superMom=0.0`, `schedW="default"`, `schedB="default"` | **Fwd:** Sums `convFilterActs` across inputs onto `modulesX^2` (`1+ceil((2*pad+img-filt)/stride)`) with zero-padded out-of-bounds pixels; adds `sharedBiases`. **Bwd:** Bias sum (`1/numCases`), `convWeightActs` reduced over `modules/partialSum` (`1/numCases`), `convImgActs` to gradConsumers. | `layer.py:974-1076`, `src/layer.cu:815-897` |
| `cmrnorm` | 1 | `inputs`, `channels`, `size`; `scale`, `pow` | `blocked=False` (`:1193`), `minDiv=1.0` (`:1186`); divides `scale /= size` (`:1183`) | **Fwd:** Cross-channel window `[max(0, c-size/2), min(C, c-size/2+size))` (when `!blocked`): `denom = minDiv + (scale/size)*sum_f x_f^2`, `y = x * denom^(-pow)`. **Bwd:** `convResponseNormCrossMapUndo` using cached `_denoms`. | `layer.py:1170-1216`, `src/layer.cu:1583-1594` |
| `rnorm` | 1 | `inputs`, `channels`, `size`; `scale`, `pow` | `blocked=False` (`:1193`); divides `scale /= size**2` (`:1183`); no `minDiv` | **Fwd:** Within-channel clamped `size x size` spatial window: `denom = 1.0 + (scale/size^2)*sum x^2`, `y = x * denom^(-pow)`. **Bwd:** `convResponseNormUndo` using cached `_denoms`. | `layer.py:1170-1216`, `src/layer.cu:1560-1574` |
| `pool` | 1 | `inputs`, `pool`, `channels`, `sizeX`, `stride` | `start=0` (`:1139`), `outputsX=0` -> `ceil((imgSize-start-sizeX)/stride)+1` (`:1164`), `doMax=False` (only if `pool=='rand'`) | **Fwd (`max`):** Clamped `sizeX x sizeX` window max (`convLocalPool<MaxPooler>`). **Bwd (`max`):** `convLocalMaxUndo` routes full upstream gradient to every input pixel matching window max (`img == ma`, including all ties). | `layer.py:1125-1168`, `src/layer.cu:1273-1334` |
| `fc` | `>= 1` | `inputs`, `outputs`, `initW` (quirk `:832`); `epsW`, `epsB`, `momW`, `momB` | `initB=0` (`:833`), `initWFunc=""`, `initBFunc=""`, `weightSource=['']*N`, `wc=[0.0]*N`, `wball=[0.0]*N`, `superEps=0.0`, `superMom=0.0`, `schedW="default"`, `schedB="default"` | **Fwd:** Sums `X[i] * W[i]` in declared input order (`W[i]` F-order `numInputs[i] x outputs`), adds `1 x outputs` bias row once. **Bwd:** `dX[i] += dY * W[i]^T`, `db = (1/numCases)*sum dY`, `_weightsInc[i] = momW*V + (epsW/numCases)*(X[i]^T * dY)`. | `layer.py:873-886`, `src/layer.cu:634-675`, `src/convnet.cu:444` |
| `hs` | 1 | `inputs`, `keep`; `enable` | None beyond base `LayerWithInputParser` | **Fwd:** Unscaled binary mask `(u < keep) * x` when `enable && PASS_TRAIN`; `keep * x` otherwise. **Bwd:** Same `_sexMask` in `PASS_TRAIN`, or `keep` otherwise. | `layer.py:689-709`, `src/layer.cu:1121-1153` |
| `softmax` | 1 | `inputs` | None beyond base `LayerWithInputParser` | **Fwd:** `p_c = exp(z_c - max_k z_k) / sum_j exp(z_j - max_k z_k)`. **Bwd:** Fuses `coeff * ((label == c) - p_c)` when sole same-GPU consumer is `cost.logreg`; else `p_c * sum_j dy_j * ((j == c) - p_j)`. | `layer.py:1104-1112`, `src/layer.cu:1016-1042`, `src/layer_kernels.cu:464` |
| `cost.logreg` | 2 (`labels, probs`) | `inputs`, `coeff` | `topk=1` (`:1257`; `neuron` deleted at `:1225`) | **Fwd:** Accumulates multiview `probs` when testing; computes `[-sum log p_y, N - top1, N - topk]` with fractional tie splitting (`computeMultiSoftmaxCost` when `topk > 1`). **Bwd:** Delegates fused logit grad to single-consumer same-GPU `softmax` (`doWork == false`). | `layer.py:1250-1271`, `src/layer.cu:1800-1858`, `src/layer_kernels.cu:96-119` |
| `neuron` (`relu`) | 1 | `inputs`, `neuron` | Detached as `<src>_neuron` (`:630`); `relu` takes 0 params (`usesActs=True, usesInputs=False`, `:1400`) | **Fwd (`relu`):** `y = (x < 0.0f ? 0.0f : x)`. **Bwd (`relu`):** `dx = dy * (y > 0.0f)` (accumulated if `scaleTargets > 0`). | `layer.py:594-653, 1400`, `src/layer.cu:483-510`, `src/neuron.cu:36`, `include/neuron.cuh:146-176` |

---

## 3. Section-by-Section Ledger (`layers/layers-145.cfg`)

*(All weighted layers also inherit defaulted `initWFunc=""`, `initBFunc=""`, `weightSource=['']*N`, `superEps=0`, `superMom=0`, `schedW="default"`, `schedB="default"`, `quantF="default"`, `quantB="default"`, plus `initCFunc=""`, `wcNormMin=0`, `wcNormMax=0` on `conv`; see `docs/spec/ledger.json` for the exhaustive per-section parameter dictionaries.)*

### 3.1 Data & Convolutional Stages (Sections 1–24)

| # | `[section]` (`cfg` / `params` line) | Type (GPU) | Declared `inputs` -> Post-detach inputs | Output shape (`outputs`) | Explicit parameters & notable parser defaults | Forward / Backward notes & implementation citations |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `data` (`:1` / —) | `data` (`-1` def) | `[]` -> `[]` | `3x224x224` (`150528`) | `dataIdx=0` | Host RGB planar crop stream -> GPU 0 & 1 (`layer.py:1096`, `src/layer.cu:1185`). |
| 2 | `labels` (`:5` / —) | `data` (`-1` def) | `[]` -> `[]` | `1` (`1`) | `dataIdx=1` | Host class-index stream -> GPU 0 (`layer.py:1098`, `src/layer.cu:1167`). |
| 3 | `conv1a` (`:9` / `:1`) | `conv` (`0`) | `[data]` -> `[data]` | `48x55x55` (`145200`), `W:(363,48)`, `b:(48,1)` | `channels=3, filters=48, padding=0, stride=4, filterSize=11, initW=0.01, partialSum=5, sharedBiases=1, gpu=0`; **def:** `initB=0, groups=1, randSparse=0, neuron=""`; **params:** `epsW=0.000, epsB=0.0, momW=0.9, momB=0.9, wc=0.0005, wball=0` | Ceiling `224->55` (3-px right/bottom zero pad); **no ReLU here**; frozen `epsW=0` in snapshot (`src/layer.cu:829, 869`). |
| 4 | `conv1b` (`:22` / `:9`) | `conv` (`1`) | `[data]` -> `[data]` | `48x55x55` (`145200`), `W:(363,48)`, `b:(48,1)` | Same as `conv1a` with `gpu=1` | Tower-B `conv1` on GPU 1 (`src/layer.cu:823, 857`). |
| 5 | `rnorm1a` (`:35` / `:137`) | `cmrnorm` (`0` def) | `[conv1a]` -> `[conv1a]` | `48x55x55` (`145200`) | `channels=48, size=5`; **def:** `blocked=False, neuron="", gpu=0`; **params:** `scale=0.0001` (`->0.00002`), `pow=0.75, minDiv=2` | Cross-map norm on raw signed `conv1a`, `denom = 2 + 2e-5 * sum_5 x^2` (`layer.py:1183`, `src/layer.cu:1589`). |
| 6 | `rnorm1b` (`:41` / `:142`) | `cmrnorm` (`1` def) | `[conv1b]` -> `[conv1b]` | `48x55x55` (`145200`) | Same as `rnorm1a` with inherited `gpu=1` | Tower-B cross-map norm (`src/layer.cu:1593`). |
| 7 | `pool1a` (`:47` / —) | `pool` (`0` def) | `[rnorm1a]` -> `[rnorm1a]` | `48x27x27` (`34992`) | `pool=max, sizeX=3, stride=2, channels=48, neuron=relu`; **def:** `start=0, outputsX=0->27, gpu=0` | Max-pool `55->27`, then detached `pool1a_neuron` (`relu`) (`src/layer.cu:1327`, `include/neuron.cuh:163`). |
| 8 | `pool1b` (`:56` / —) | `pool` (`1` def) | `[rnorm1b]` -> `[rnorm1b]` | `48x27x27` (`34992`) | Same as `pool1a` with inherited `gpu=1` | Max-pool `55->27`, then detached `pool1b_neuron` (`relu`) (`src/layer.cu:1333`). |
| 9 | `conv2a` (`:65` / `:17`) | `conv` (`0`) | `[pool1a]` -> `[pool1a_neuron]` | `128x27x27` (`93312`), `W:(1200,128)`, `b:(128,1)` | `filters=128, padding=2 (->-2), stride=1, filterSize=5, channels=48, initW=0.01, initB=1, partialSum=3, sharedBiases=1, neuron=relu, gpu=0`; **def:** `groups=1, randSparse=0`; **params:** `epsW=1e-5, epsB=0.02, momW=0.9, momB=0.9, wc=0.0005, wball=0` | Within-tower `5x5` conv + `initB=1` + detached `conv2a_neuron` (`relu`) (`src/layer.cu:829, 894`). |
| 10 | `conv2b` (`:80` / `:25`) | `conv` (`1`) | `[pool1b]` -> `[pool1b_neuron]` | `128x27x27` (`93312`), `W:(1200,128)`, `b:(128,1)` | Same as `conv2a` with `gpu=1` | Tower-B `5x5` conv + detached `conv2b_neuron` (`relu`) (`src/layer.cu:823`). |
| 11 | `rnorm2a` (`:95` / `:147`) | `cmrnorm` (`0` def) | `[conv2a]` -> `[conv2a_neuron]` | `128x27x27` (`93312`) | `channels=128, size=5`; **def:** `blocked=False, gpu=0`; **params:** `scale=0.0001 (->0.00002), pow=0.75, minDiv=2` | Cross-map norm after `conv2a_neuron` (`src/layer.cu:1589`). |
| 12 | `rnorm2b` (`:101` / `:152`) | `cmrnorm` (`1` def) | `[conv2b]` -> `[conv2b_neuron]` | `128x27x27` (`93312`) | Same as `rnorm2a` with inherited `gpu=1` | Tower-B cross-map norm after `conv2b_neuron` (`src/layer.cu:1593`). |
| 13 | `cnorm2a` (`:107` / `:157`) | `rnorm` (`0` def) | `[rnorm2a]` -> `[rnorm2a]` | `128x27x27` (`93312`) | `channels=128, size=5`; **def:** `blocked=False, gpu=0`; **params:** `scale=0.001 (->0.00004), pow=0.75` (no `minDiv`) | Extra `5x5` **spatial** response norm (`denom = 1 + 4e-5 * sum_{5x5} x^2`) (`layer.py:1183`, `src/layer.cu:1569`). |
| 14 | `cnorm2b` (`:113` / `:161`) | `rnorm` (`1` def) | `[rnorm2b]` -> `[rnorm2b]` | `128x27x27` (`93312`) | Same as `cnorm2a` with inherited `gpu=1` | Tower-B extra `5x5` spatial response norm (`src/layer.cu:1573`). |
| 15 | `pool2a` (`:119` / —) | `pool` (`0` def) | `[cnorm2a]` -> `[cnorm2a]` | `128x13x13` (`21632`) | `pool=max, sizeX=3, stride=2, channels=128`; **def:** `start=0, outputsX=0->13, neuron="", gpu=0` | Max-pool `27->13` (no ReLU); fans out to `conv3a` & `conv3b` (`src/layer.cu:1327`). |
| 16 | `pool2b` (`:127` / —) | `pool` (`1` def) | `[cnorm2b]` -> `[cnorm2b]` | `128x13x13` (`21632`) | Same as `pool2a` with inherited `gpu=1` | Tower-B max-pool `27->13`; fans out to `conv3a` & `conv3b` (`src/layer.cu:1333`). |
| 17 | `conv3a` (`:135` / `:33`) | `conv` (`0`) | `[pool2a, pool2b]` -> `[pool2a, pool2b]` | `192x13x13` (`32448`), `W:2x(1152,192)`, `b:(192,1)` | `filters=192,192, padding=1,1 (->-1,-1), stride=1,1, filterSize=3,3, channels=128,128, initW=0.03,0.03, partialSum=13, sharedBiases=1, neuron=relu, gpu=0`; **def:** `initB=0, groups=1,1, randSparse=0,0`; **params:** `epsW=1e-5,1e-5, epsB=0.02, momW=0.9,0.9, momB=0.9, wc=5e-4,5e-4, wball=0,0` | Cross-tower `3x3` conv (`initB=0` def, `initW=0.03`) + detached `conv3a_neuron` (`relu`) (`src/layer.cu:829`). |
| 18 | `conv3b` (`:149` / `:41`) | `conv` (`1`) | `[pool2a, pool2b]` -> `[pool2a, pool2b]` | `192x13x13` (`32448`), `W:2x(1152,192)`, `b:(192,1)` | Same as `conv3a` with `gpu=1` | Cross-tower `3x3` conv on GPU 1 + detached `conv3b_neuron` (`relu`) (`src/layer.cu:857`). |
| 19 | `conv4a` (`:163` / `:49`) | `conv` (`0` def) | `[conv3a]` -> `[conv3a_neuron]` | `192x13x13` (`32448`), `W:(1728,192)`, `b:(192,1)` | `filters=192, padding=1 (->-1), stride=1, filterSize=3, channels=192, neuron=relu, initW=0.03, initB=1, partialSum=13, sharedBiases=1`; **def:** `gpu=0, groups=1, randSparse=0`; **params:** `epsW=1e-5, epsB=0.02, momW=0.9, momB=0.9, wc=5e-4, wball=0` | Within-tower `3x3` conv on inherited GPU 0 + detached `conv4a_neuron` (`layer.py:378`, `src/layer.cu:894`). |
| 20 | `conv4b` (`:177` / `:57`) | `conv` (`1` def) | `[conv3b]` -> `[conv3b_neuron]` | `192x13x13` (`32448`), `W:(1728,192)`, `b:(192,1)` | Same as `conv4a` with inherited `gpu=1` | Within-tower `3x3` conv on inherited GPU 1 + detached `conv4b_neuron` (`src/layer.cu:823`). |
| 21 | `conv5a` (`:192` / `:65`) | `conv` (`0` def) | `[conv4a]` -> `[conv4a_neuron]` | `128x13x13` (`21632`), `W:(1728,128)`, `b:(128,1)` | `filters=128, padding=1 (->-1), stride=1, filterSize=3, channels=192, initW=0.03, initB=1, partialSum=13, groups=1, randSparse=0`; **def:** `sharedBiases=True` (`:1064`), `neuron="", gpu=0`; **params:** `epsW=1e-5, epsB=0.02, momW=0.9, momB=0.9, wc=5e-4, wball=0` | Within-tower `3x3` conv; `sharedBiases=True` by default; ReLU deferred to `pool3a` (`src/layer.cu:829`). |
| 22 | `conv5b` (`:206` / `:73`) | `conv` (`1` def) | `[conv4b]` -> `[conv4b_neuron]` | `128x13x13` (`21632`), `W:(1728,128)`, `b:(128,1)` | Same as `conv5a` with inherited `gpu=1` | Tower-B `conv5`; ReLU deferred to `pool3b` (`src/layer.cu:881`). |
| 23 | `pool3a` (`:220` / —) | `pool` (`0` def) | `[conv5a]` -> `[conv5a]` | `128x6x6` (`4608`) | `pool=max, sizeX=3, stride=2, channels=128, neuron=relu`; **def:** `start=0, outputsX=0->6, gpu=0` | Max-pool `13->6` + detached `pool3a_neuron` (`relu`); feeds `fc2048a` & `fc2048b` (`src/layer.cu:1327`). |
| 24 | `pool3b` (`:229` / —) | `pool` (`1` def) | `[conv5b]` -> `[conv5b]` | `128x6x6` (`4608`) | Same as `pool3a` with inherited `gpu=1` | Max-pool `13->6` + detached `pool3b_neuron` (`relu`); feeds `fc2048a` & `fc2048b` (`src/layer.cu:1333`). |

---

### 3.2 Classifier, Dropout & Cost Stages (Sections 25–35)

| # | `[section]` (`cfg` / `params` line) | Type (GPU) | Declared `inputs` -> Post-detach inputs | Output shape (`outputs`) | Explicit parameters & notable parser defaults | Forward / Backward notes & implementation citations |
| --- | --- | --- | --- | --- | --- | --- |
| 25 | `fc2048a` (`:238` / `:81`) | `fc` (`0`) | `[pool3a, pool3b]` -> `[pool3a_neuron, pool3b_neuron]` | `2048` (`2048`), `W:2x(4608,2048)` F, `b:(1,2048)` | `outputs=2048, initW=0.01,0.01, initB=1, neuron=relu, gpu=0`; **params:** `epsW=1e-5,1e-5, epsB=0.02, momW=0.9,0.9, momB=0.9, wc=5e-4,5e-4, wball=0,0` | Cross-tower FC1 half A + detached `fc2048a_neuron` (`relu`) (`layer.py:877`, `src/layer.cu:640, 659`). |
| 26 | `fc2048b` (`:247` / `:89`) | `fc` (`1`) | `[pool3a, pool3b]` -> `[pool3a_neuron, pool3b_neuron]` | `2048` (`2048`), `W:2x(4608,2048)` F, `b:(1,2048)` | Same as `fc2048a` with `gpu=1` | Cross-tower FC1 half B + detached `fc2048b_neuron` (`relu`) (`src/layer.cu:647`). |
| 27 | `hs1a` (`:256` / `:125`) | `hs` (`0` def) | `[fc2048a]` -> `[fc2048a_neuron]` | `2048` (`2048`) | `keep=0.5`; **def:** `neuron="", gpu=0`; **params:** `enable=true` | Unscaled dropout (`u < 0.5` in `PASS_TRAIN`, `* 0.5` in eval); feeds `fc2048ba[0]` & `fc2048bb[1]` (`src/layer.cu:1126`). |
| 28 | `hs1b` (`:261` / `:131`) | `hs` (`1` def) | `[fc2048b]` -> `[fc2048b_neuron]` | `2048` (`2048`) | `keep=0.5`; **def:** `neuron="", gpu=1`; **params:** `enable=true` | Unscaled dropout on GPU 1; feeds `fc2048bb[0]` & `fc2048ba[1]` (`src/layer.cu:1137`). |
| 29 | `fc2048ba` (`:266` / `:97`) | `fc` (`0`) | `[hs1a, hs1b]` -> `[hs1a, hs1b]` | `2048` (`2048`), `W:2x(2048,2048)` F, `b:(1,2048)` | `outputs=2048, initW=0.01,0.01, initB=1, neuron=relu, gpu=0`; **params:** `epsW=1e-5,1e-5, epsB=0.02, momW=0.9,0.9, momB=0.9, wc=5e-4,5e-4, wball=0,0` | Cross-tower FC2 half A (`inputs=hs1a,hs1b`) + detached `fc2048ba_neuron` (`relu`) (`src/layer.cu:640`). |
| 30 | `fc2048bb` (`:275` / `:105`) | `fc` (`1`) | **`[hs1b, hs1a]`** -> **`[hs1b, hs1a]`** | `2048` (`2048`), `W:2x(2048,2048)` F, `b:(1,2048)` | Same as `fc2048ba` with `gpu=1` and **reversed input order `hs1b,hs1a`** | `W[0]` multiplies `hs1b`, `W[1]` multiplies `hs1a`; detached `fc2048bb_neuron` (`relu`) (`layers/layers-145.cfg:277`, `src/layer.cu:659`). |
| 31 | `hs2a` (`:284` / `:128`) | `hs` (`0` def) | `[fc2048ba]` -> `[fc2048ba_neuron]` | `2048` (`2048`) | `keep=0.5`; **def:** `neuron="", gpu=0`; **params:** `enable=true` | Unscaled dropout feeding `fc1000` input `0` (`src/layer.cu:1126`). |
| 32 | `hs2b` (`:289` / `:134`) | `hs` (`1` def) | `[fc2048bb]` -> `[fc2048bb_neuron]` | `2048` (`2048`) | `keep=0.5`; **def:** `neuron="", gpu=1`; **params:** `enable=true` | Unscaled dropout on GPU 1 feeding `fc1000` input `1` (`src/layer.cu:1137`). |
| 33 | `fc1000` (`:294` / `:113`) | `fc` (`0`) | `[hs2a, hs2b]` -> `[hs2a, hs2b]` | `1000` (`1000`), `W:2x(2048,1000)` F, `b:(1,1000)` | `outputs=1000, initW=0.01,0.01, gpu=0`; **def:** **`initB=0`**, `neuron=""`; **params:** `epsW=1e-5,1e-5, epsB=0.02, momW=0.9,0.9, momB=0.9, wc=5e-4,5e-4, wball=0,0` | 1000-way linear logits (`initB=0` default, no ReLU) (`layer.py:833`, `src/layer.cu:640`). |
| 34 | `probs` (`:301` / —) | `softmax` (`0` def) | `[fc1000]` -> `[fc1000]` | `1000` (`1000`) | **def:** `neuron="", gpu=0` | Stable softmax; backward emits fused `coeff * ((label == c) - p_c)` (`src/layer.cu:1020, 1038`, `src/layer_kernels.cu:464`). |
| 35 | `logprob` (`:305` / `:121`) | `cost.logreg` (`0`) | `[labels, probs]` -> `[labels, probs]` | `0` (`_costv`: 3 scalars) | `gpu=0`; **params:** `coeff=1, topk=5` (default `1` at `layer.py:1257`) | Multiview prob accumulator + `[-sum log p_y, N - top1, N - top5]` via `computeMultiSoftmaxCost` (`src/layer.cu:1804-1858`). |

### 3.3 Detached Neuron Layers (`layer.py:272, 627`)

`LayerParser.detach_neuron_layers` synthesizes **14** `type=neuron` (`relu`) layers in section order:
`pool1a_neuron`, `pool1b_neuron`, `conv2a_neuron`, `conv2b_neuron`, `conv3a_neuron`, `conv3b_neuron`, `conv4a_neuron`,
`conv4b_neuron`, `pool3a_neuron`, `pool3b_neuron`, `fc2048a_neuron`, `fc2048b_neuron`, `fc2048ba_neuron`, `fc2048bb_neuron`.

Total trainable parameters across the 15 weighted sections (22 weight matrices + 15 bias vectors):
- Convolutional weights + biases: `2*(363*48 + 48) + 2*(1200*128 + 128) + 2*(2*1152*192 + 192) + 2*(1728*192 + 192) + 2*(1728*128 + 128) = 2,334,080`.
- Fully-connected weights + biases: `2*(2*4608*2048 + 2048) + 2*(2*2048*2048 + 2048) + (2*2048*1000 + 1000) = 58,631,144`.
- **Grand total:** **`60,965,224` FP32 parameters** (`~232.56 MiB` per copy).

ANX-3 correction: the original displayed convolution subtotal omitted its 1,376 biases despite including them
in the formula. Weight entries alone total 60,954,656; biases total 10,568 (`layer.py:883-884,1070-1073`).
Every shape field for both 224 and 227 crops, including detached neurons and partial-window flags, is recorded
in each layer's `shape.inference_by_inner_size` in `ledger.json` and machine checked.

---

## 4. Open Questions (Unresolvable from `AlexNet-Source-Code` Alone)

1. **`OQ-LEDGER-1` — Exact `batches.meta` image size & `--crop-border` CLI flag (`224` vs `227`):**
   `DataLayerParser` (`layer.py:1099`) reads `data` dimensions from `JPEGCroppedImageNetDP` (`convdata_jpeg.py:73`:
   `inner_size = img_size - 2 * crop_border`, default `crop_border=4` at `convnet.py:182`). Neither `batches.meta` nor
   the `layers-145` launch command is in the tree. Under ceiling arithmetic (`layer.py:1004`), any `inner_size` in
   `{224, 225, 226, 227}` produces `55x55` `conv1` outputs; `224` matches the NIPS 2012 paper (`256` with `crop_border=16`,
   or pre-cropped `232` with default `crop_border=4`) and exercises the 3-pixel right/bottom zero-padded partial window.
2. **`OQ-LEDGER-2` — Omitted September 2012 `cudaconv2` / `nvmatrix` source snapshot:**
   `Makefile-distrib:16` and `build.sh:14` copy external `cudaconv2` and `nvmatrix` trees omitted from this repo. As
   proven in `docs/spec/CUDACONV2.md`, public `cuda-convnet` (r561) and `cuda-convnet2` jointly determine every kernel
   used in `layers-145.cfg`, though the exact intermediate `.cu` files (and unused `convLocalRandomPool` / `convQuantizeHalf`)
   were never published.
3. **`OQ-LEDGER-3` — Late-run frozen `conv1a/b` rates in `layers/layer-params-145.cfg`:**
   Committed `layers/layer-params-145.cfg:2-3, 10-11` has `epsW=0.000, epsB=0.0` on `conv1a/b` (from the epoch-72 manual
   intervention at `:179`), which makes `gradConsumer=False` (`layer.py:744`) if parsed without restoring initial rates.
4. **`OQ-LEDGER-4` — Host/GPU RNG seeds & CUDA reduction order:**
   Unseeded `numpy.random.randn` (`layer.py:807`), device CURAND `randomizeUniform` (`src/layer.cu:1129`), and warp/block
   FP32 summation order preclude bitwise historical trajectory identity.
