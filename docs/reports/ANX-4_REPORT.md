# ANX-4 — Tensor and layout primitives

Started: 2026-11-25T17:21:35Z. Base: `464ce57`. Report created before implementation.
Finished: 2026-11-25T17:34:45Z (13.2 min elapsed). All five brief items and all gates passed.
No commits or pushes. Paths in source citations are upstream-relative.

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. Tensor, views, index helpers and parameter containers | done | Six files in `src/AlexNetSharp.Core/Tensors/`; index/view/weight tests and complete 224/227 allocations passed in the initial integrated run. | 17:21–17:29 UTC (~8 min, including source review and first proof) |
| 2. NCHW bridge and RGB crop import | done | `LayoutBridge.cs`; hand interleaving, four odd-size round trips, RGB coordinates/mean and untouched-case tests passed. | 17:21–17:29 UTC (~8 min, including source review and first proof) |
| 3. Accumulation and graph fan-out | done | `FloatMatrix.AddProduct/Add`, `GradientBuffer`; NaN/product/fan-out, all six gradient fan-outs in both arrival orders and remote-only merge proofs pass; source-derived table below. | 17:21–17:31 UTC (~10 min, including source review and remote merge fixture) |
| 4. Tests | done | 30 tensor groups + 101 existing parser/shape tests and spec checks pass in each of three fresh processes; see final timings below. | 17:25–17:31 UTC (~6 min) |
| 5. Report and final gates | done | Build 0 warnings/errors; all 131 tests + spec checks pass 3×; original-tree, LF/path/whitespace and diff checks pass. Nothing unfinished. | 17:21–17:34 UTC (~13.2 min; report kept current throughout, including final verification) |

Times are elapsed intervals and overlap: source review, implementation and fixtures were interleaved.
First integrated proofs passed at 17:29 UTC; final proof timings follow below. Items 1–4 include their initial successful proof; the final repeat gate and report time are charged to item 5.

## Source provenance and fidelity

Read `DESIGN.md`, `docs/spec/LEDGER.md`, `docs/spec/CUDACONV2.md` (especially §1–3),
and `docs/reports/ANX-3_REPORT.md` before implementation. Original files were resolved
through `ALEXNET_ORIGINAL_SRC` in `local.env`, read without modification, and checked
against a pre-implementation size/mtime/SHA-256 baseline of 274 files.

The original release omits NVMatrix and cudaconv2 sources. Candidate A in
`CUDACONV2.md` was downloaded into ignored `.scratch/cudaconv2-candidates/` and read
without editing. Its archive SHA-256 matches ANX-1:
`b55bd30ceb2c69c9540ae7e7c2f6f6cd075551071c351e57e3b7a5d7bb10f819`.
Below, **A:** identifies paths relative to Candidate A's `cuda-convnet/trunk/`.
References without that prefix identify the original 2012 tree. Candidate evidence
is identified explicitly; this does not claim access to the missing September snapshot
or execution of the historical CUDA binary.

No ANX-2 parser, ANX-3 shape inference, golden fixture, ledger or manifest behavior changed.
The test entry point registers the tensor suite and prints its count and duration.
All arithmetic is scalar float32 loops. No layer forward/backward math, optimizer,
initialization, image decoder, web code, SIMD or optimization was added.

## Layout table per buffer kind

Let `N` be cases, `C` channels, `H,W` plane dimensions, `O` outputs, `F` filters,
and `K = filterChannels * filterHeight * filterWidth`. C-order means row-major;
F-order means column-major. Offsets are in float elements, not bytes.

| Buffer kind | Logical rows × columns / layout | Flat offset or ordering | Source evidence |
| --- | --- | --- | --- |
| Canonical data/image activations | `(C*H*W, N)`, C | `(((c*H)+y)*W+x)*N+n` | A:`src/cudaconv2/filter_acts.cu:1209-1213`; A:`include/cudaconv2/conv_util.cuh:121-122`; `src/layer.cu:1223-1226` |
| Conv, max-pool, spatial/cross-map norm output | `(O,N)`, C; `O=C*H*W` | Rows channel, y, x; case fastest | `src/layer.cu:769,1319,1560,1583`; same A kernel layout above |
| FC and softmax native output | `(N,O)`, F (`isTrans=true`) | `output*N+case`, the same bytes as canonical `(O,N)` C | `src/layer.cu:635,1017,152-155`; A:`src/nvmatrix/nvmatrix.cu:530-550` |
| All neuron native outputs, including image neurons | `(N,O)`, F | Same case-fastest allocation; image feature row remains channel, y, x | `src/layer.cu:483-484,152-155` |
| hs/dropout output | `(O,N)`, C (`isTrans=false`) | Same physical order as its FC/neuron input, explicit view change | `src/layer.cu:1121,152-155` |
| Labels | `(1,N)`, C at data layer | One float label per case | `convdata_jpeg.py:268-269,199`; `src/layer.cu:1167,1209-1226` |
| Conv weights, per input | `(K,F)`, C | `((channel*filterHeight+y)*filterWidth+x)*F+filter` | `layer.py:797-807,1072`; A:`src/cudaconv2/filter_acts.cu:1210` |
| FC weights, per input | `(numInputs[i],O)`, F | `output*numInputs[i]+inputFeature` | `layer.py:883` |
| Conv shared bias | `(F,1)`, C | One entry per filter; broadcast across pixels/cases | `layer.py:1069,1073`; `src/layer.cu:833-837` |
| Conv unshared bias (supported by shape allocation) | `(F*modules,1)`, C | One entry per output feature | `layer.py:1069,1073`; `src/layer.cu:838-839` |
| FC bias | `(1,O)`, F | One entry per output | `layer.py:884`; `src/layer.cu:642-643` |
| Activity gradient | Target producer's view, made compatible with the current consumer's `_trans` | Same dimensions/storage map as the corresponding activations | `src/layer.cu:272-279,287-295` |
| One-case view | `(C*H*W,1)`, strided C | Offset `case`, row stride `N` | Derived directly from activation formula above |
| One channel plane for one case | `(H,W)`, strided C | Offset `channel*H*W*N+case`, row stride `W*N`, column stride `N` | Derived directly from activation formula above |
| Logreg cost public output | No activation (`outputs=0`) | `LayerBuffers.Activations=null`; original uses private per-case cost scratch and scalar cost values | `layer.py:187,1261-1273`; `src/layer.cu:1825-1836` |

For all 49 effective 145 layers, the native transposed set is precisely the five
FC layers (`fc2048a`, `fc2048b`, `fc2048ba`, `fc2048bb`, `fc1000`), `probs`, and
the 14 detached neurons. The remaining 29 use `_trans=false`, including the cost
(which has no public activation output). This includes the four hs layers, even
though their inputs come from transposed neuron outputs. Tests verify every layer's
native dimensions and layout, not just its element count. An activation's current
view in the original can change again when its next consumer calls `transpose(_trans)`;
the table records each layer's own native view, not a permanent runtime flag.

`FloatMatrix` exposes `Rows`, `Cols`, `Layout`, `IsTrans`, `LeadingDimension`,
`FollowingDimension`, `RowStride`, `ColumnStride`, `StorageOffset`, `ElementCount`
and `IsContiguous`. The leading dimension is columns for C-order, rows for F-order;
the following dimension is the other axis (A:`include/nvmatrix/nvmatrix.cuh:147-160`).
Actual view strides are separate from these logical dimensions. `TransposeView()`
swaps dimensions/strides and layout without copying or mutating the original object;
`Slice`, `CaseView`, and `ChannelPlaneView` share storage. `GetFlatOffset` is an
absolute offset in the entire backing `Storage`, including the view's offset.

`ActivationLayout` maps coordinates to canonical row/column and absolute flat offset,
and implements both inverses. It validates each axis so an invalid x/case cannot
silently address the next row. `ActivationTensor.Matrix` is always canonical C-order;
request its transposed view explicitly. `LayerBuffers.Allocate` explicitly selects
the layer's native view from ANX-3 `LayerShape`. Weight buffers preserve ANX-3 shapes
and input names/order; no transpose, reshape or resize is hidden in arithmetic.

`GetNumCases(layerTrans, weightUpdateFrequency, minibatchSize)` matches
`src/layer.cu:337-340`: cases are rows for a transposed layer and columns otherwise;
when frequency differs from 1, the original uses frequency times minibatch size,
not the number of cases currently in the matrix. Both branches are tested.

## NCHW and single-crop bridge

`FromNchw` and `ToNchw` copy between `[case][channel][y][x]` flat arrays and canonical
case-fastest matrices. Import/export does no floating-point arithmetic, preserving
float32 bit patterns, including signed zero and a NaN payload.

`ImportRgbCrop` accepts exactly one already cropped `224×224×3` interleaved RGB byte
image and a selected case in a `(150528,N)` tensor. RGB order follows PIL's
`convert('RGB')`, then `swapaxes(0,2).swapaxes(1,2).flatten()`
(`convdata_jpeg.py:34-41`); the provider reshapes to `(cases,3,H,W)` before cropping
(`:222`). The helper keeps 0–255 intensity units. It optionally subtracts an
already cropped planar mean (`:95,207-210`); it does not divide by 255, decode,
resize, choose crops, flip or add PCA noise. The optional empty mean selects raw
intensities; preprocessing policy remains a caller decision.

The provider returns `cropped.T` (`convdata_jpeg.py:199`), an F-order host view with
logical `(features,cases)` dimensions. This is not yet the kernel's C-order buffer:
`DataLayer::copyData` detects `dataMatrix.isTrans()` and calls `flipTrans` when
copying to each GPU (`src/layer.cu:1215-1226`). That operation changes storage order
while preserving logical dimensions (A:`src/nvmatrix/nvmatrix.cu:554-573`). The import
helper writes the resulting GPU/data-layer layout directly. The 3-case crop fixture
checks distinct RGB and asymmetric coordinates, planar mean subtraction and that
other cases stay untouched.

## Accumulation and NaN semantics

`AddProduct(a,b,scaleThis,scaleAB)` computes `scaleThis*target + scaleAB*(a*b)`.
The original calls it for FC forward (`src/layer.cu:641`) and input gradient
(`:649`), with separate weight matrices in input order (`layer.py:797-807`).
The scalar implementation accepts either explicit output layout and preallocated
shapes, extending NVMatrix's column-major product target without changing the
logical result. It does not reproduce NVMatrix's automatic resizing/layout changes.
Target/operand backing aliases are rejected; partial overlapping Add views are
also rejected, rather than silently corrupting inputs. Exact same-index Add aliases
are supported. These are explicit CPU API boundaries, not extra parser restrictions.

Crucially, zero target scale discards existing NaNs. Candidate A's AddProduct
branches at A:`src/nvmatrix/nvmatrix.cu:221-225` to `rightMult`, which passes
SGEMM beta=0 at `:190-205`. The [cuBLAS GEMM contract](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
allows C to be invalid input when beta is zero. `Add` similarly takes source-only
scaling at A:`src/nvmatrix/nvmatrix.cu:583-586`, rather than evaluating `0*target`.
Tests explicitly assert finite results after a NaN-filled target is overwritten,
and NaN preservation with nonzero target scale. They do not use tolerance comparisons
that could accidentally accept NaN.

GEMM alpha=0 skips product operands; Add's source-only scaling still computes
`source*scaleSource` (A:`include/nvmatrix/nvmatrix_operators.cuh:155-163`), so a NaN
source scaled by zero remains NaN. Both distinct behaviors have fixtures.

`GradientBuffer.BeginPass` resets the contribution count without clearing storage,
matching `Layer::reset` (`src/layer.cu:329-334`). `NextScaleTargets` is 0 before the
first successful write, 1 afterwards. `Add` and `AddProduct` return the coefficient
actually used. Failed shape/alias checks do not advance the counter. The hand
single-buffer fan-out is `[1,2;3,4]` followed by `[10,20;30,40]`, yielding
`[11,22;33,44]`, with coefficients 0 then 1; a new pass overwrites that old sum.

### Where the original chooses the coefficient

- Forward inputs: the shared activation target is written first with 0; remaining
  inputs use 1. Without sharing, input 0 uses 0 and later inputs use 1
  (`src/layer.cu:157-165`). Conv/FC add the bias only on that first write
  (`:642-643,833-839`). All multi-input 145 weight layers retain matrix input order,
  including `fc2048bb`'s `hs1b,hs1a` (`layers/layers-145.cfg:277`).
- Backward consumers: both nonshared and shared target paths choose
  `previous.getRcvdBInputs(currentDevice)>0 ? 1 : 0`, then increment the count
  (`src/layer.cu:285-295`). Counts and buffers are per producer/device
  (`:358-367,394-405,448-454`). `NeuronLayer` routes zero to overwrite and positive
  to add (`:490-505`; ReLU paths `include/neuron.cuh:152-157`). Conv, FC, pool,
  norms and hs pass/branch on this same coefficient (`src/layer.cu:894-895,649,1333,1573,1593,1137-1151`).
- Cross-GPU merge: when there is no local contribution, the first remote buffer
  copies with target scale 0; otherwise it adds with 1 (`src/layer.cu:178,202-212`
  for `USE_PD`; `:228-251` for the other path). Consumer arrival order is not fixed
  (`:370-377`). The report does not invent a global first consumer.
- Weight/bias reductions use update counts instead: bias scale is
  `getNumUpdates()>0` (`src/layer.cu:655,847`); conv direct gradients do likewise
  unless using partial sums, whose temporary target always overwrites before the
  counted final sum (`:860-877`). FC gradient scale uses update count; its direct
  momentum-increment path uses momentum on the first training write and 1 later
  (`:663-671`). Those optimizer paths are documented, not implemented here.

## Complete 145 fan-out table

Input indices below are zero based. Neuron detachment changes `pool3a/b`'s effective
consumers to originate from `pool3a/b_neuron` (`layer.py:630-645`). GPU inheritance is
`layer.py:378,634`. The six gradient fan-outs each have one consumer on GPU0 and one
on GPU1; therefore **each consumer's local scaleTargets is 0**, and the producer's
merge adds the remote result with **1**. This differs from the equivalent scalar
single-buffer sequence 0 then 1, which is tested separately.

| Producer (own GPU) | Consumer / input index (GPU) | Consumer scaleTargets | Producer merge | Connectivity citations |
| --- | --- | --- | --- | --- |
| data (host) | conv1a / 0 (0); conv1b / 0 (1) | No gradient write to data | None | `layers/layers-145.cfg:11,24`; no trainable ancestor, `src/layer.cu:286,409-419` |
| pool2a (0) | conv3a / 0 (0); conv3b / 0 (1) | 0; 0 in separate buffers | Add remote with 1 | `layers/layers-145.cfg:137,147,151,161`; dispatch `src/layer.cu:287-295` |
| pool2b (1) | conv3a / 1 (0); conv3b / 1 (1) | 0; 0 in separate buffers | Add remote with 1 | `layers/layers-145.cfg:137,147,151,161`; dispatch `src/layer.cu:287-295` |
| pool3a_neuron (0) | fc2048a / 0 (0); fc2048b / 0 (1) | 0; 0 in separate buffers | Add remote with 1 | `layers/layers-145.cfg:240,245,249,254`; detach `layer.py:630-645` |
| pool3b_neuron (1) | fc2048a / 1 (0); fc2048b / 1 (1) | 0; 0 in separate buffers | Add remote with 1 | `layers/layers-145.cfg:240,245,249,254`; detach `layer.py:630-645` |
| hs1a (0) | fc2048ba / 0 (0); fc2048bb / 1 (1) | 0; 0 in separate buffers | Add remote with 1 | `layers/layers-145.cfg:268,273,277,282` |
| hs1b (1) | fc2048ba / 1 (0); fc2048bb / 0 (1) | 0; 0 in separate buffers | Add remote with 1 | `layers/layers-145.cfg:268,273,277,282` |

`GraphFanOut` enumerates the parsed effective graph and verifies this complete producer
set, consumer identities and devices; each of the six gradient fan-outs is exercised
in both consumer arrival orders with separate buffers and a summed result. A separate
remote-only merge tests the case of `hs2b` on GPU1, whose sole consumer `fc1000` is on
GPU0 (`layers/layers-145.cfg:289-299`): consumer scale 0, then merge/copy with 0.
The CPU primitives model these accumulators; this story does not add GPU scheduling.

All other actual backward consumer writes in 145 are sole-consumer first writes with
scaleTargets=0. Two qualifications matter: the stored params freeze conv1a/b
(`layers/layer-params-145.cfg:2-3,10-11`), so their data/norm/pool1/neuron ancestry has
no gradient consumer (`layer.py:744`; recursive ancestry `src/layer.cu:409-419`) and
conv2 does not write input gradients to `pool1a/b_neuron`. Restoring initial conv1
rates as the manifest describes enables those additional sole-consumer writes,
still with 0. Also, logreg invokes the `probs` backward edge with 0 but computes no
probability gradient in this same-GPU, sole-softmax-consumer case
(`src/layer.cu:1845-1856`); `probs` instead writes the fused logit gradient to
`fc1000` with scale 0 (`:1032-1040`). No NaN or initialization policy is inferred
from an unwritten probability-gradient scratch buffer.

## Test inventory and allocation proof

| Tensor groups | Count | Proof |
| --- | ---: | --- |
| Hand indices | 2 | Four corners, last channel/case, 13×13 and 55×55 planes; both inverses and matrix physical offsets |
| Shared views | 2 | Rectangular C/F matrices, transpose/slices, absolute strides, case/plane mutation visibility and isolation, getNumCases |
| NCHW | 5 | Explicit two-case hand interleaving; four odd/singleton shapes, every coordinate and bit-preserving round trip |
| RGB | 1 | Full 224×224 three-channel crop into a selected case, RGB units/order, mean and untouched case |
| Weights | 2 | All 18 rows of a 2-channel 3×3 filter by hand and explicit filter-fastest bytes; FC F-order bytes |
| AddProduct | 10 | Eight input/output layout combinations with signed hand values, transposed/strided views, scaled overwrite/add, zero/NaN behavior |
| Add | 1 | Mixed layouts, overwrite/add/general scales, NaN target/source behavior |
| Gradient accumulation | 4 | Single-buffer fan-out, product fan-out, remote-only merge, all graph fan-outs/devices in both arrival orders |
| Validation | 1 | Invalid axes, overflow, dimensions, aliasing and crop/mean lengths; failed gradient write preserves count |
| Complete network allocations | 2 | Every 224/227 layer activation/weight/bias array allocated and checked against ANX-3 independent fixtures |
| Total new tensor groups | **30** | Plus 16 parser and 85 shape tests = **131** tests, plus existing spec checks |

Each full-network allocation retains all arrays simultaneously for a batch of 3,
checks every native activation dimension/order and buffer end, all input weight names
and shapes, all biases, actual allocated lengths, and independent parameter totals.
There are 49 layer entries: 48 public activation matrices, 22 weight matrices and
15 bias vectors. No public activation is allocated for the cost. Counts agree with
`tests/fixtures/layers-145.shapes.json` at both crop sizes.

Weights: **60,954,656**; biases: **10,568**; total: **60,965,224** parameters,
**243,860,896 bytes** of float32 parameter storage. Activations for three cases are
**6,026,739** elements at 224 and **6,038,916** at 227. The per-case 224 sum is
`150528 + 1 + 4*145200 + 4*34992 + 8*93312 + 2*21632 + 8*32448 + 2*21632 + 4*4608 + 12*2048 + 2*1000 = 2008913`;
227 changes only the data layer by `154587-150528=4059`. The activation totals are
hand constants in the tensor tests in addition to per-layer fixture comparisons.

`fc2048bb` additionally checks names `hs1b,hs1a` and verifies that modifying W[0]
does not change W[1], even though both shapes are equal. Parameter slots are zero
allocated; random/bias initialization and sharing (unused by 145) remain later work.
This is an allocation/layout proof, not historical floating-point trajectory parity.

## Final verification

Final build: exit **0**, **0 warnings / 0 errors**. MSBuild elapsed **00:00:01.58**;
whole build process wall time **1.7169 s**. `NUGET_HTTP_CACHE_PATH`, NuGet
packages, CLI state and temp directory were explicitly set under `.scratch/`.

Each row below is a separate new OS process executing the compiled test assembly,
without rebuilding. PIDs are distinct and recorded with complete output in the ignored logs.

| Run | Exit | Tensor groups / time | Entire suite time | Process wall time |
| --- | --- | --- | --- | --- |
| 1 | 0 | 30 / 45 ms | 131 tests + spec / 162 ms | 0.1812 s |
| 2 | 0 | 30 / 36 ms | 131 tests + spec / 154 ms | 0.1718 s |
| 3 | 0 | 30 / 45 ms | 131 tests + spec / 161 ms | 0.1803 s |

Each run also passed the existing spec check: **35 sections, 10 layer types,
15 weighted layers, 569 parser parameters, 455 source citations and 1,470 shape
fields**. The executable harness uses no external test-framework packages.

Logs: `.scratch/anx4/final-build.log`, `final-tests-1.log`, `final-tests-2.log`,
`final-tests-3.log`, `final-results.json`, and `integrity.log` (all in that directory).
The initial integrated run had 29 tensor groups; the final suite includes the added
remote-only merge fixture and has 30. No implementation test failures occurred;
hand fixture totals were checked before the first suite run.

Integrity: all **274 original files** retain their baseline file set, sizes, mtimes
and SHA-256 digests. All **9 changed/new deliverable files** use LF with final
newlines, contain no trailing whitespace, literal home paths or secret assignments;
`git diff --check` passes. Authored files, candidate downloads, build/cache/temp
artifacts and logs stayed inside this repository; no files outside the repo were
edited. No original files changed. No stash, commits or pushes were performed.
All brief items are done and nothing is unfinished.


Reproduction from repository root, keeping caches and CLI state inside the repo:

```sh
mkdir -p .scratch/tmp
export DOTNET_CLI_HOME="$PWD/.scratch/dotnet-home"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
export DOTNET_GENERATE_ASPNET_CERTIFICATE=false
export DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE=true
export NUGET_PACKAGES="$PWD/.scratch/nuget-packages"
export NUGET_HTTP_CACHE_PATH="$PWD/.scratch/nuget-http-cache"
export TMPDIR="$PWD/.scratch/tmp"
"$HOME/.dotnet/dotnet" build --disable-build-servers
for run in 1 2 3; do
    "$HOME/.dotnet/dotnet" tests/AlexNetSharp.Tests/bin/Debug/net10.0/AlexNetSharp.Tests.dll || exit 1
done
```
