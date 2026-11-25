# cudaconv2 Version & Kernel Evidence (`ANX-1`)

## Why this document exists

`computerhistory/AlexNet-Source-Code` (commit `2f8b81f`) omits the `cudaconv2` and `nvmatrix` kernel sources that the
build expects:

- `Makefile-distrib:3` includes `-I./include/cudaconv2 -I./include/nvmatrix`, and `Makefile-distrib:16-17` compiles
  `src/cudaconv2/*.cu` and `include/cudaconv2/*.cuh`.
- `build.sh:4,14-15` copies `conv_util.cu`, `filter_acts.cu`, `weight_acts.cu`, `img_acts.cu`, `conv_util.cuh`, and
  `cudaconv2.cuh` from `$NVCONV2_K20X_INCLUDE/..`.
- `src/layer.cu:33` includes `<cudaconv2.cuh>` and makes 45 calls across 33 external `cudaconv2` / `conv_util` routines
  (plus `src/quantizer.cu:60,64` calling `convQuantizeHalf` / `convDequantizeHalf` when half-precision cross-GPU
  quantisation is enabled).

Both public upstream candidates by Alex Krizhevsky were fetched read-only into `.scratch/cudaconv2-candidates/` (not
committed) and compared function-by-function against `src/layer.cu`.

---

## Candidate sources & provenance

| Candidate | Source URL | Revision / Archive digest | Layout of cudaconv headers & sources |
| --- | --- | --- | --- |
| **Candidate A — `cuda-convnet` (Google Code archive)** | `https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/cuda-convnet/source-archive.zip` | Archive SHA-256: `b55bd30ceb2c69c9540ae7e7c2f6f6cd075551071c351e57e3b7a5d7bb10f819` (SVN `trunk` rev `562`, last change rev `561` on `2012-07-25T06:13:58.079541Z`, paths below relative to `cuda-convnet/trunk/`) | `include/cudaconv2/{cudaconv2.cuh,conv_util.cuh}`, `src/cudaconv2/{filter_acts.cu,weight_acts.cu,img_acts.cu,conv_util.cu}` (matches `Makefile-distrib:3,16-17`) |
| **Candidate B — `cuda-convnet2` (GitHub)** | `https://github.com/akrizhevsky/cuda-convnet2` | Git `HEAD` commit: `3238bf0367f63eb370e897b9e5714794cb67ddc2` (40 commits; initial commit `a3bfc4a`, 2014 Kepler release; paths below relative to repo root) | `cudaconv3/include/{cudaconv2.cuh,conv_util.cuh}`, `cudaconv3/src/{filter_acts.cu,weight_acts.cu,img_acts.cu,conv_util.cu}` |

---

## Verdict

**Neither public candidate matches `src/layer.cu` 100% identically.** The September 2012 `AlexNet-Source-Code` tree sits
chronologically and structurally between Candidate A (25 July 2012) and Candidate B (2014):

1. **Candidate A (`cuda-convnet` Google Code archive, r561/r562) is the primary match (30 / 33 functions match
   verbatim)** in C++ signature, parameter order, 1D `partialSum` semantics, random-sparse (`*Sparse`) support,
   non-template `AvgPooler()`, and `_denoms` caching. Only **three** calls in `src/layer.cu` post-date Candidate A:
   - `convResponseNormCrossMap` (`src/layer.cu:1589`) passes **9 arguments** (`*_inputs[0], _denoms, getActs(), _channels, _size, _scale, _pow, _minDiv, _blocked`), adding `float minDiv` to Candidate A's 8-argument overload (`include/cudaconv2/conv_util.cuh:68-69`).
   - `convReflectHorizontal` (`src/layer.cu:1492,1496`, unused in `layers-145.cfg`) is absent from Candidate A and present in Candidate B (`cudaconv3/include/conv_util.cuh:66`, `cudaconv3/src/conv_util.cu:74`).
   - `convLocalRandomPool` (`src/layer.cu:1350`, `pool=rand` when `doMax=0`, unused in `layers-145.cfg`) is absent from **both** Candidate A and Candidate B (an unreleased stochastic-pooling kernel, alongside `convQuantizeHalf` / `convDequantizeHalf` in `src/quantizer.cu:60,64`).
2. **Candidate B (`cuda-convnet2` / `cudaconv3`) is a later Kepler rewrite that breaks 17 / 33 call signatures and
   changes `convWeightActs` `partialSum` semantics incompatibly:**
   - Removes all 6 `*Sparse` convolution/local kernels and `convTICA` / `convTICAGrad`.
   - Changes `AvgPooler` to `template<bool sum> class AvgPooler` (`cudaconv3/include/conv_util.cuh:71-72`), breaking `AvgPooler()` at `src/layer.cu:1307,1607`, and adds `bool sum` to `convLocalAvgUndo` (`cudaconv3/include/conv_util.cuh:33-35`).
   - Adds `float minDiv` to `convResponseNorm` and `convContrastNorm` (`cudaconv3/include/conv_util.cuh:39,42`).
   - Removes `NVMatrix& denoms` from `convResponseNormCrossMap` and `convResponseNormCrossMapUndo` (`cudaconv3/include/conv_util.cuh:60-65`) so denominators are recomputed in backward; however, `cudaconv3/src/conv_util.cu:2890,2898-2899` still contains Krizhevsky's commented-out `denoms.resize(images)` and `printf` showing `(images, meanDiffs, denoms, target, ..., minDiv, blocked)` — direct fossil evidence of the September 2012 9-argument signature called at `src/layer.cu:1589`.
   - **Critical runtime incompatibility in `convWeightActs`:** Candidate B keeps the 14-argument C++ prototype (`cudaconv3/include/cudaconv2.cuh:58-60`) but reinterprets parameter 12 from 1D `int partialSum` (divisor of `numModules = modulesX * modulesY`) to 2D `int sumWidth` (`cudaconv3/src/weight_acts.cu:2090-2095`, where the old `assert(numModules % partialSum == 0)` is commented out). For `conv1a` (`modules = 55×55 = 3025`, `partialSum = 5`), `src/layer.cu:875` reshapes `_weightGradTmp` to `_modules / _partialSum = 605` rows, which matches Candidate A (`src/cudaconv2/weight_acts.cu:982,1011-1013`) and conflicts with Candidate B (`DIVUP(55, 5)^2 = 121` rows).

**Consequence for `layers-145.cfg`:** Every kernel exercised by `layers-145.cfg` (`convFilterActs`, `convWeightActs`,
`convImgActs`, `convLocalPool<MaxPooler>`, `convLocalMaxUndo`, `convResponseNorm`, `convResponseNormUndo`,
`convResponseNormCrossMap`, `convResponseNormCrossMapUndo`) is completely determined with zero ambiguity: Candidate A
supplies the exact convolution, pooling, spatial-norm, and cross-map-norm backward semantics, while Candidate B's
`kFCNorm` (`cudaconv3/src/conv_util.cu:1161-1210`) confirms how `minDiv` enters the cross-map denominator
(`denom = minDiv + addScale * sum`).

---

## Signature comparison for every call in `src/layer.cu`

Legend:
- **Match** — exact parameter list and semantics match `src/layer.cu`.
- **ABI-only (semantic break)** — compiles with the same parameter types, but parameter meaning changed incompatibly.
- **Signature mismatch** — function exists, but parameter count or template signature differs.
- **Absent** — function does not exist in the candidate.

| Function / Type | `src/layer.cu` call site(s) | Used in `145`? | Candidate A (`cuda-convnet` trunk) `file:line` | Candidate B (`cuda-convnet2`) `file:line` |
| --- | --- | --- | --- | --- |
| `convFilterActs` (12 args) | `src/layer.cu:829` | **Yes** | **Match** — `include/cudaconv2/cudaconv2.cuh:39`, `src/cudaconv2/filter_acts.cu:1188` | **Match** — `cudaconv3/include/cudaconv2.cuh:29`, `cudaconv3/src/filter_acts.cu:2254` |
| `convFilterActsSparse` (14 args) | `src/layer.cu:825` | No (`randSparse=0`) | **Match** — `include/cudaconv2/cudaconv2.cuh:83`, `src/cudaconv2/filter_acts.cu:1424` | **Absent** |
| `convWeightActs` (14 args) | `src/layer.cu:869` | **Yes** | **Match** (`partialSum` divides `modules`) — `include/cudaconv2/cudaconv2.cuh:67`, `src/cudaconv2/weight_acts.cu:960` | **ABI-only (semantic break: 2D `sumWidth`)** — `cudaconv3/include/cudaconv2.cuh:57`, `cudaconv3/src/weight_acts.cu:2720` |
| `convWeightActsSparse` (16 args) | `src/layer.cu:865` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:99`, `src/cudaconv2/weight_acts.cu:1226` | **Absent** |
| `convImgActs` (12 args) | `src/layer.cu:894` | **Yes** (`conv2..5`) | **Match** — `include/cudaconv2/cudaconv2.cuh:54`, `src/cudaconv2/img_acts.cu:1897` | **Match** — `cudaconv3/include/cudaconv2.cuh:44`, `cudaconv3/src/img_acts.cu:2687` |
| `convImgActsSparse` (14 args) | `src/layer.cu:885` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:112`, `src/cudaconv2/img_acts.cu:2565` | **Absent** |
| `localFilterActs` (12 args) | `src/layer.cu:952` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:47`, `src/cudaconv2/filter_acts.cu:1201` | **Match** — `cudaconv3/include/cudaconv2.cuh:37`, `cudaconv3/src/filter_acts.cu:2267` |
| `localFilterActsSparse` (14 args) | `src/layer.cu:948` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:88`, `src/cudaconv2/filter_acts.cu:1431` | **Absent** |
| `localWeightActs` (13 args) | `src/layer.cu:987` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:76`, `src/cudaconv2/weight_acts.cu:971` | **Match** — `cudaconv3/include/cudaconv2.cuh:66`, `cudaconv3/src/weight_acts.cu:2731` |
| `localWeightActsSparse` (15 args) | `src/layer.cu:983` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:106`, `src/cudaconv2/weight_acts.cu:1240` | **Absent** |
| `localImgActs` (12 args) | `src/layer.cu:998` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:60`, `src/cudaconv2/img_acts.cu:1908` | **Match** — `cudaconv3/include/cudaconv2.cuh:50`, `cudaconv3/src/img_acts.cu:2698` |
| `localImgActsSparse` (14 args) | `src/layer.cu:994` | No | **Match** — `include/cudaconv2/cudaconv2.cuh:118`, `src/cudaconv2/img_acts.cu:2578` | **Absent** |
| `normalizeLocalWeights` (3 args) | `src/layer.cu:1006` | No | **Match** — `include/cudaconv2/conv_util.cuh:61`, `src/cudaconv2/conv_util.cu:83` | **Match** — `cudaconv3/include/conv_util.cuh:57`, `cudaconv3/src/conv_util.cu:205` |
| `convLocalPool` + `MaxPooler` (8 args) | `src/layer.cu:1327,1348` | **Yes** | **Match** — `include/cudaconv2/conv_util.cuh:84,327` | **Match** — `cudaconv3/include/conv_util.cuh:85,450` |
| `convLocalPool` + `MaxAbsPooler` (8 args) | `src/layer.cu:1325` | No | **Match** — `include/cudaconv2/conv_util.cuh:97,327` | **Match** — `cudaconv3/include/conv_util.cuh:98,450` |
| `convLocalPool` + `AvgPooler` (8 args) | `src/layer.cu:1307,1607` | No | **Match** — `include/cudaconv2/conv_util.cuh:71,327` | **Signature mismatch** (`AvgPooler<bool sum>` requires template arg) — `cudaconv3/include/conv_util.cuh:71-72,450` |
| `convLocalMaxUndo` (10 args) | `src/layer.cu:1333,1356` | **Yes** | **Match** — `include/cudaconv2/conv_util.cuh:40`, `src/cudaconv2/conv_util.cu:1869` | **Match** — `cudaconv3/include/conv_util.cuh:36`, `cudaconv3/src/conv_util.cu:2075` |
| `convLocalAvgUndo` (9 args) | `src/layer.cu:1311` | No | **Match** — `include/cudaconv2/conv_util.cuh:37`, `src/cudaconv2/conv_util.cu:1968` | **Signature mismatch** (10 args, adds `bool sum`) — `cudaconv3/include/conv_util.cuh:33`, `cudaconv3/src/conv_util.cu:2174` |
| `convLocalRandomPool` (7 args) | `src/layer.cu:1350` | No | **Absent** | **Absent** |
| `convCrop` (6 args) | `src/layer.cu:1413,1417,1425` | No | **Match** — `include/cudaconv2/conv_util.cuh:60`, `src/cudaconv2/conv_util.cu:2593` | **Match** — `cudaconv3/include/conv_util.cuh:56`, `cudaconv3/src/conv_util.cu:2823` |
| `convResizeBilinear` (5 args) | `src/layer.cu:1415,1512` | No | **Match** — `include/cudaconv2/conv_util.cuh:57`, `src/cudaconv2/conv_util.cu:2416` | **Match** — `cudaconv3/include/conv_util.cuh:53`, `cudaconv3/src/conv_util.cu:2646` |
| `convBedOfNails` (8 args) | `src/layer.cu:1448` | No | **Match** — `include/cudaconv2/conv_util.cuh:52`, `src/cudaconv2/conv_util.cu:665` | **Match** — `cudaconv3/include/conv_util.cuh:48`, `cudaconv3/src/conv_util.cu:789` |
| `convBedOfNailsUndo` (8 args) | `src/layer.cu:1452` | No | **Match** — `include/cudaconv2/conv_util.cuh:54`, `src/cudaconv2/conv_util.cu:670` | **Match** — `cudaconv3/include/conv_util.cuh:50`, `cudaconv3/src/conv_util.cu:794` |
| `convGaussianBlur` (7 args) | `src/layer.cu:1466,1467,1472,1473` | No | **Match** — `include/cudaconv2/conv_util.cuh:50`, `src/cudaconv2/conv_util.cu:682` | **Match** — `cudaconv3/include/conv_util.cuh:46`, `cudaconv3/src/conv_util.cu:806` |
| `convReflectHorizontal` (3 args) | `src/layer.cu:1492,1496` | No | **Absent** | **Match** — `cudaconv3/include/conv_util.cuh:66`, `cudaconv3/src/conv_util.cu:74` |
| `convRGBToYUV` (2 args) | `src/layer.cu:1529` | No | **Match** — `include/cudaconv2/conv_util.cuh:58`, `src/cudaconv2/conv_util.cu:2470` | **Match** — `cudaconv3/include/conv_util.cuh:54`, `cudaconv3/src/conv_util.cu:2700` |
| `convRGBToLAB` (3 args) | `src/layer.cu:1547` | No | **Match** — `include/cudaconv2/conv_util.cuh:59`, `src/cudaconv2/conv_util.cu:2516` | **Match** — `cudaconv3/include/conv_util.cuh:55`, `cudaconv3/src/conv_util.cu:2746` |
| `convResponseNorm` (7 args) | `src/layer.cu:1569` | **Yes** (`cnorm2a/b`) | **Match** — `include/cudaconv2/conv_util.cuh:43`, `src/cudaconv2/conv_util.cu:2064` | **Signature mismatch** (8 args, adds `float minDiv`) — `cudaconv3/include/conv_util.cuh:39`, `cudaconv3/src/conv_util.cu:2317` |
| `convResponseNormUndo` (11 args) | `src/layer.cu:1573` | **Yes** (`cnorm2a/b`) | **Match** — `include/cudaconv2/conv_util.cuh:44`, `src/cudaconv2/conv_util.cu:2228` | **Match** — `cudaconv3/include/conv_util.cuh:40`, `cudaconv3/src/conv_util.cu:2482` |
| `convResponseNormCrossMap` (9 args) | `src/layer.cu:1589` | **Yes** (`rnorm1/2`) | **Signature mismatch** (8 args: has `denoms`, lacks `minDiv`) — `include/cudaconv2/conv_util.cuh:68`, `src/cudaconv2/conv_util.cu:2892` | **Signature mismatch** (8 args: has `minDiv`, drops `denoms`; fossil comment at `cudaconv3/src/conv_util.cu:2890,2898`) — `cudaconv3/include/conv_util.cuh:64`, `cudaconv3/src/conv_util.cu:3018` |
| `convResponseNormCrossMapUndo` (12 args) | `src/layer.cu:1593` | **Yes** (`rnorm1/2`) | **Match** (`denoms` already incorporates `minDiv`) — `include/cudaconv2/conv_util.cuh:66`, `src/cudaconv2/conv_util.cu:2801` | **Signature mismatch** (drops `denoms`, adds `float minDiv`) — `cudaconv3/include/conv_util.cuh:60`, `cudaconv3/src/conv_util.cu:2934` |
| `convContrastNorm` (8 args) | `src/layer.cu:1610` | No | **Match** — `include/cudaconv2/conv_util.cuh:46`, `src/cudaconv2/conv_util.cu:2074` | **Signature mismatch** (9 args, adds `float minDiv`) — `cudaconv3/include/conv_util.cuh:42`, `cudaconv3/src/conv_util.cu:2327` |
| `convContrastNormUndo` (11 args) | `src/layer.cu:1615` | No | **Match** — `include/cudaconv2/conv_util.cuh:47`, `src/cudaconv2/conv_util.cu:2214` | **Match** — `cudaconv3/include/conv_util.cuh:43`, `cudaconv3/src/conv_util.cu:2468` |
| `convTICA` (6 args) | `src/layer.cu:1995` | No | **Match** — `include/cudaconv2/conv_util.cuh:63`, `src/cudaconv2/conv_util.cu:2694` | **Absent** |
| `convTICAGrad` (7 args) | `src/layer.cu:2001` | No | **Match** — `include/cudaconv2/conv_util.cuh:62`, `src/cudaconv2/conv_util.cu:2643` | **Absent** |

*(In addition, `src/quantizer.cu:60,64` calls `convQuantizeHalf` and `convDequantizeHalf`, which are absent from both
candidates and unused when `quantF=default`, `quantB=default`.)*

---

## Verified kernel semantics required by `layers-145.cfg`

### 1. Tensor layout & `paddingStart` convention
- Convolutional/pooling/normalisation activations use row-major `(channels * imgPixels, numCases)` where rows are ordered
  `(channel, y, x)` with `x` fastest within each pixel plane and `case` fastest in memory (`src/cudaconv2/filter_acts.cu:1209`,
  `include/cudaconv2/conv_util.cuh:121-122`).
- Convolution weights have shape `(filterChannels * filterPixels, numFilters)` in C-order (`layer.py:1072`,
  `src/cudaconv2/filter_acts.cu:1210`).
- `layer.py:1041` negates `padding` (`dic['padding'][i] = -dic['padding'][i]`), so `padding=2` in `.cfg` arrives at
  `convFilterActs` / `convWeightActs` / `convImgActs` as `paddingStart = -2`.

### 2. Boundary / partial-window handling (`224 -> 55` ceiling arithmetic)
- `layer.py:1004` computes `modulesX = 1 + int(ceil((2 * padding + imgSize - filterSize) / float(stride)))`.
- In `filter_acts.cu` (`src/cudaconv2/filter_acts.cu:129-154` in Candidate A; same logic in Candidate B), module
  `(my, mx)` samples image coordinates `x = paddingStart + mx * moduleStride + fx`, `y = paddingStart + my * moduleStride + fy`.
  Any `(x, y)` outside `[0, imgSize)` loads `0.0f` (`else { // Padding ... shImages[...] = 0; }`).
- `weight_acts.cu:1017-1020` asserts `paddingStart <= 0` and `paddingStart + (numModulesX - 1) * moduleStride + filterSize >= imgSizeX`,
  which is why `layer.py:1004` uses `ceil`: every input pixel must be visited if the last stride step does not land
  flush with `imgSize`.
- On `conv1a` / `conv1b` (`imgSize=224`, `filterSize=11`, `padding=0 -> paddingStart=0`, `stride=4`), `modulesX = 55`.
  Module index `54` starts at `54 * 4 = 216` and spans `216..226`, reading real pixels `216..223` and **implicitly
  zero-padding 3 pixels on the right and bottom (`224, 225, 226`)**, with **0 padding on the left and top**. This is
  equivalent to asymmetric zero-padding `(left=0, top=0, right=3, bottom=3)` on a `224×224` crop, **not** symmetric
  padding and **not** cropping a `227×227` image.
- `weight_acts.cu` and `img_acts.cu` (`src/cudaconv2/img_acts.cu:92-113`) apply the exact adjoint of this zero-padded
  partial window.

### 3. `partialSum` in `convWeightActs`
- In Candidate A (`src/cudaconv2/weight_acts.cu:982,1011-1013`) and `src/layer.cu:860-878`, `partialSum` divides
  `numModules = modulesX * modulesX` (`partialSum == 0` means `numModules`, i.e. full reduction in one pass).
- `convWeightActs` writes `(numModules / partialSum, filterChannels * filterPixels, numFilters)` into `_weightGradTmp`,
  scaled by `scaleWGrad = 1.0f / numCases` (training) or `1.0f` (gradient check), and `src/layer.cu:875-877` sums over
  axis 0 into `_weights[inpIdx].getGrad()`. Mathematically this is just a parallel reduction partitioning of the exact
  sum over all `(my, mx)` module positions (up to FP32 summation order).

### 4. Sparse / grouped connections
- `convFilterActs` / `convWeightActs` / `convImgActs` take `numGroups`; when `groups > 1` and `randSparse == 0`, channels
  and filters are partitioned into `numGroups` contiguous blocks (`filterChannels = channels / groups`,
  `filtersPerGroup = numFilters / groups`). In `layers-145.cfg`, every conv section uses `groups=1` and `randSparse=0`
  (cross-GPU tower splitting is expressed as explicit separate sections `conv1a`/`conv1b`, etc., and two-input fan-in on
  `conv3a`/`conv3b`).

### 5. Max pooling boundary clamping & tie-breaking (`convLocalPool`, `convLocalMaxUndo`)
- Forward (`include/cudaconv2/conv_util.cuh:162-166` in Candidate A; `cudaconv3/include/conv_util.cuh:162-166` in
  Candidate B): for output `(oy, ox)`, `startPx = startX + o * strideX`, clamped to
  `[max(0, startPx), min(imgSize, startPx + subsX))` (no artificial zeros inserted into the max window). In
  `layers-145.cfg`, `55 -> 27`, `27 -> 13`, and `13 -> 6` all tile cleanly with `sizeX=3, start=0, stride=2`.
- Backward (`src/cudaconv2/conv_util.cu:1547-1560` in Candidate A; `cudaconv3/src/conv_util.cu:1753-1766` in
  Candidate B): for each input pixel `img`, iterates over all overlapping pooled outputs `outputIdx` and accumulates
  `prod += (img == ma) * mg`. **If multiple pixels inside a pooling window tie for the maximum `ma`, every tied pixel
  receives the full upstream gradient `mg` (not divided by tie count, and not first-winner-only).**

### 6. Spatial response normalisation (`rnorm` -> `convResponseNorm` / `convResponseNormUndo`)
- Used by `[cnorm2a]` and `[cnorm2b]` (`type=rnorm`, `size=5`, configured `scale=0.001` -> parser divides by `size^2 = 25`
  to pass `_scale = 0.00004`, `_pow = 0.75`; `layer.py:1183`).
- Forward (`src/cudaconv2/conv_util.cu:758-795, 947-1014, 2064`): within each channel independently, sums `x[c, y', x']^2`
  over `y' in [max(0, y - sizeX/2), min(imgSize, y - sizeX/2 + sizeX))` and `x' in [max(0, x - sizeX/2), min(imgSize, x - sizeX/2 + sizeX))`
  (integer division `sizeX/2`, so `size=5` gives `[y-2, y+3)` clamped to `[0, imgSize)`), sets
  `denom = 1.0f + addScale * sum`, and outputs `y = x * denom^(-powScale)` (`minDiv` is **not** used by `type=rnorm`; the
  additive constant is `1.0f`).
- Backward (`src/cudaconv2/conv_util.cu:1831-1850, 2228`): for input `(c, y, x)`,
  `dx[c, y, x] = outGrad[c, y, x] * denom[c, y, x]^(-powScale) + x[c, y, x] * sum_{(y',x') : (y,x) in W(y',x')} (-2 * addScale * powScale * acts[c, y', x'] * outGrad[c, y', x'] / denom[c, y', x'])`.

### 7. Cross-map response normalisation (`cmrnorm` -> `convResponseNormCrossMap` / `convResponseNormCrossMapUndo`)
- Used by `[rnorm1a]`, `[rnorm1b]`, `[rnorm2a]`, `[rnorm2b]` (`type=cmrnorm`, `size=5`, `blocked=False` default,
  configured `scale=0.0001` -> parser divides by `size = 5` to pass `_scale = 0.00002`, `_pow = 0.75`, `_minDiv = 2.0`;
  `layer.py:1183-1193`, `layers/layer-params-145.cfg:137-155`).
- Forward (`cudaconv3/src/conv_util.cu:1190-1208` + `src/cudaconv2/conv_util.cu:2767-2786`): at each spatial pixel
  `(y, x)` and channel `c`:
  - `startF = blocked ? (c / sizeF) * sizeF : -sizeF / 2 + c` (integer division `sizeF / 2`, so for `sizeF=5, blocked=false`, `startF = c - 2`);
  - `loopStartF = blocked ? startF : max(0, startF)`, `loopEndF = min(numFilters, startF + sizeF)`;
  - `denom[c, y, x] = minDiv + addScale * sum_{f = loopStartF}^{loopEndF - 1} x[f, y, x]^2` (**note `minDiv = 2.0` in `layers-145`, not `1.0`!**);
  - `acts[c, y, x] = x[c, y, x] * denom[c, y, x]^(-powScale)`.
- Backward (`src/cudaconv2/conv_util.cu:2801-2889`): uses cached `_denoms` (which already includes `minDiv`). First sets
  `tmp[f] = -2 * addScale * powScale * acts[f] * outGrad[f] / denom[f]`, then for each channel `c` sums `tmp[f]` over
  all channels `f` whose forward window included `c` (for `blocked=false` and odd `sizeF=5`, `f in [max(0, c - (sizeF - 1)/2), min(numFilters, c + sizeF/2 + 1))`; in general `f` such that `c in [loopStartF(f), loopEndF(f))`):
  `dx[c] = outGrad[c] * denom[c]^(-powScale) + x[c] * sum_{f : c in W(f)} tmp[f]`.
