# ANX-5 Report — Scalar convolution

Started: 2026-11-25T18:04Z (approximately; first clock reading 18:06:08Z). Base: `1fb7c6a`. Finished: 2026-11-25T18:15Z (~11 min elapsed). All five items and gates passed; no commits or pushes.

## Brief items

| Item | Status | Evidence | Time |
| --- | --- | --- | --- |
| 1. Scalar convolution | done | Seven independent fixtures, numerical gradients and contract proofs passed; `ScalarConvolution.cs`. | ~18:04–18:11 UTC (7 min) |
| 2. Independent fixtures | done | Seven exact-rational fixtures, both partialSum modes and all gradients pass. | ~18:08–18:11 UTC (3 min) |
| 3. Numerical gradients | done | 3,824 float64 finite differences passed, worst absolute error 5.58786e-8. | ~18:09–18:11 UTC (2 min) |
| 4. Real geometry | done | All ten 145 layers, two cases each, adjoint/Euler and actual conv1 tap proofs passed. | ~18:09–18:13 UTC (4 min) |
| 5. Report and gates | done | Both builds 0 warnings/errors; 177 tests plus spec checks pass 3× in fresh processes; provenance, regeneration and hygiene checks pass. | ~18:04–18:15 UTC (11 min, including ongoing report; final gates ~18:13–18:15) |

## Findings while implementing

The brief requests partial windows for 5×5 / filter 3 / stride 2 / padding 0 and
15×15 / filter 11 / stride 4. Both tile exactly: `1+ceil((5-3)/2)=2` and
`1+ceil((15-11)/4)=2`. The exact requested sizes are retained; additional 6×6
and 16×16 fixtures exercise actual ceiling partial windows. With padding 1 the
5×5 case has 3 modules and symmetric configured padding, rather than extra
ceiling padding. This is a contradiction in the brief examples, not a defect
in `CUDACONV2.md` or the original parser (`layer.py:1004`).

Candidate A downloaded under ignored `.scratch/anx5/`; archive SHA-256 matches
`b55bd30ceb2c69c9540ae7e7c2f6f6cd075551071c351e57e3b7a5d7bb10f819`.
Original tree baseline: 274 files (size, mtime and SHA-256).

## Source semantics and API

Read `DESIGN.md`, `CUDACONV2.md` (sections 1–3 and convolution signature rows),
ANX-3 and ANX-4 reports before implementation. Unprefixed citations below refer
to the original release; `A:` refers to the verified archive's
`cuda-convnet/trunk/`. The archive is evidence for missing kernels, not a claim
that the September binary was executed. No candidate source is part of this change.

| Behavior | Scalar implementation | Original and candidate evidence |
| --- | --- | --- |
| Shapes and windows | `1+ceil((image-2*paddingStart-filter)/stride)`; sample `paddingStart+module*stride+tap`; out-of-range value 0 | `layer.py:1004,1041`; A:`src/cudaconv2/filter_acts.cu:129-154` |
| Layout | Existing ANX-4 `FloatMatrix`; input/output `(channel*y*x,case)` C-order, W `(channel*fy*fx,filter)` C-order, bias `(filter,1)` | `layer.py:1072-1073`; A:`src/cudaconv2/filter_acts.cu:1209-1213` |
| Multi-input | Separate W per declared input; first input overwrites and adds bias, later inputs accumulate | `src/layer.cu:157-165,823-842`; `layer.py:797-807` |
| Shared bias | Broadcast one value per filter over modules/cases, only when target scale is zero | `src/layer.cu:833-839`; `layer.py:1069,1073` |
| Input adjoint | Sum all output windows containing each input pixel, including partial edge windows; never batch-average | `src/layer.cu:881-895`; A:`src/cudaconv2/img_acts.cu:92-113` |
| Input eligibility | Caller chooses whether to call `BackwardInput`; real checks skip conv1 data gradients | `src/layer.cu:285-295,409-419`; `CUDACONV2.md` conv signature rows |
| Weight gradient | Training `1/numCases`, gradient-check 1; weight update count represented by explicit target scale 0 or 1 | `src/layer.cu:857-877`; A:`src/cudaconv2/weight_acts.cu:186-196` |
| Partial sum | 0 means all modules; positive divisor partitions flattened row-major modules into consecutive chunks; scale each chunk, then sum chunks | `src/layer.cu:860-877`; A:`src/cudaconv2/weight_acts.cu:76,112-115,186-196,599-601,982,1011-1020` |
| Bias gradient | Sum modules then cases per filter, use same training/GC scale and target scale | `src/layer.cu:844-854`; A:`src/nvmatrix/nvmatrix.cu:896-901` |
| Deferred updates | Optional normalization case count accepts `FloatMatrix.GetNumCases` virtual batch count | `src/layer.cu:337-340` |
| Zero target scale | Overwrite without reading stale targets/NaNs; applies to all four operations | A:`src/cudaconv2/weight_acts.cu:180-196`; A:`src/nvmatrix/nvmatrix.cu:896-901`; ANX-4 convention |
| Group/sparse support | Reject either before execution with “unsupported in this port” and layer name; all 145 convs use 1/false | `CUDACONV2.md` section 4; A:`src/cudaconv2/filter_acts.cu:243-258`; original `layers/layers-145.cfg` |

`ScalarConvolution.FromLayer` takes the existing parsed layer and ANX-3 shape;
`LayerParameterBuffers.Allocate` supplies its existing ANX-4 parameter matrices.
Low-level construction supports small independent fixtures and varying input geometry.
`Forward` handles the complete fan-in; `ForwardInput`, `BackwardInput`,
`BackwardWeights`, and `BackwardBiases` expose explicit target scaling.
The graph will decide gradient eligibility and accumulation counts in later stories.
The stored historical rates freeze conv1 ancestry (ANX-4 report); the brief's
conv2 input-adjoint checks represent the manifest's initially trainable conv1 setup.

All arithmetic is scalar float32. Forward reduces channel, filter y, filter x;
input gradients reduce module y, module x, filter. Weight gradients reduce
module then case inside each consecutive chunk, multiply each completed chunk
by the scale, and reduce chunks in ascending order. Bias gradients reduce
module then case. Chunking changes float32 association: for a 3×3 image, 1×1
filter and unit upstream gradient, values `[1e8,1,1,-1e8,1,1,0,0,0]` yield
2 with partialSum 0 and 0 with partialSum 3; both are explicitly asserted.
The final CPU chunk reduction is sequential, whereas original NVMatrix uses GPU
reduction trees. Historical bitwise parity is not claimed. The partition and
scaling stages match the original; ANX-6 must match this scalar reference's stated order.

Unshared biases are also supported and tested (already allocated by ANX-4).
Canonical strided row slices preserve offsets/strides and have sentinel tests;
transposed/noncanonical matrices and backing aliases are rejected before writes.
No ANX-2..4 implementation or layout was changed.

## Fixture and numerical proof inventory

`tests/fixtures/make_conv_fixture.py` independently constructs a sparse linear
operator from input-pixel/output-window membership using exact rational arithmetic.
It applies the operator and its transpose and derives weight/bias coefficients;
it never calls the C# code. Regenerate `conv-cases.json` with Python's standard
library only. Expected tensors contain exact halves/quarters and small integers,
so all fixture comparisons require exact equality and reject NaN/Infinity.

| Fixture | Image/channels | Filter/stride/config padding | Filters/cases | Module width |
| --- | --- | --- | --- | --- |
| requested-5x5-full | 5 / 1 | 3 / 2 / 0 | 1 / 1 | 2 |
| requested-5x5-pad1 | 5 / 1 | 3 / 2 / 1 (`paddingStart=-1`) | 1 / 1 | 3 |
| ceiling-6x6-partial | 6 / 1 | 3 / 2 / 0 | 1 / 1 | 3 |
| two-channels-filters-cases | 4 / 2 | 3 / 2 / 0 | 2 / 2 | 2 |
| requested-15x15-k11-s4-full | 15 / 1 | 11 / 4 / 0 | 1 / 1 | 2 |
| mini-conv1-16x16-partial | 16 / 1 | 11 / 4 / 0 | 1 / 1 | 3 |
| two-inputs-distinct-geometry | 6 / 1 and 3 / 2 | 3 / 2 / 0 and 1 / 1 / 0 | 2 / 2 | 3 |

Each fixture runs partialSum 0 and a divisor (module width), forward, input,
weight and bias gradients; every gradient checks target scales 0 and 1,
weight/bias check training and GC scaling. Forward checks full/explicit fan-in
and scale 1 suppressing repeated bias; all zero-scale writes start from NaN.
Additional contracts cover cancellation order, virtual-batch normalization,
unshared bias, named group/sparse errors, invalid partial sums, shape/layout/alias
errors, and strided buffers with untouched sentinels.

Numerical checks use seeds 7301–7304: 6×6 / C2 / K3 / S2 / N2;
4×4 / C1 / K3 / S2 / padding1 / N3; 16×16 / C1 / K11 / S4 / N2;
and two inputs (4×4 / C1 / K3 / S2, 2×2 / C2 / K1 / S1), N2.
They cover every input, weight and shared-bias variable, partialSum 0/divisor,
and both training and GC objective scaling. A separately traversed float64
input-centric scatter defines `loss = dot(forward, upstream)`. Variables are
copied into double arrays before perturbation. Central differences use
`epsilon=1e-5`; acceptance is `abs(error) <= 2e-5*(1+abs(numeric))`.
This allows float32 accumulation error while double central differences avoid
float32 perturbation cancellation. All 3,824 checks passed; worst observed
absolute error was 5.58786e-8, well below the declared tolerance.

Initial build caught two missing column indices in the boundary test itself;
these were fixed before the first integrated run. Rebuild passed with 0 warnings
and 0 errors. No original-spec deviation was found; `CUDACONV2.md` remains accurate.

The first full Debug run passed all 177 tests plus spec checks in 68.040 s.
A separate rounding witness checks scale-before-combine: for N=3, chunks
containing `[1e8, -1e8+8, 1]` (other values zero), partialSum 3 gives
2.3333332538604736, whereas partialSum 0 gives 3. This guards the placement
of the training scale independently of the mathematical-gradient tolerance.
Final repeated Release proofs and repository integrity checks passed, as recorded below.

## Real-geometry proof and timings

All ten convolution layers are constructed from the parsed 145 definitions and
ANX-3 shapes, with ANX-4 parameter allocations, two cases per layer and fixed
seeds 14500–14509. Both inputs of conv3a/b are exercised. All outputs, input
gradients (conv2–5), weight gradients and bias gradients have their expected
shapes and finite values. Input-adjoint and weight Euler dot-product identities
use float64 dot accumulation and tolerance `5e-4*(1+abs(expected))`, allowing
long float32 reductions. The seven tiny fixtures and independent numerical
objective supply the independent correctness oracle; these real-size identities
add scale/layout and adjoint checks.

Both conv1 towers additionally execute the actual convolution with one-hot taps
and coordinate-encoded images. Each of the 11 final-window columns/rows reads
216..223 then 0,0,0, proving the right and bottom boundaries by actual values.
Geometry assertions also check start 216 and coordinates 216..226. Conv1 data
gradients are deliberately omitted; weight and bias backward passes still run.

Timings below are Release run 1, milliseconds. “Weights + proof” includes weight
gradients and an extra per-input forward for adjoint/Euler verification. Total
also includes bias reduction and other checks; these are informational scalar
reference measurements, not an ANX-6 throughput benchmark.

| Layer | Output C×H×W, N=2 | Inputs | Forward | Input gradient | Weights + proof | Total |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| conv1a | 48x55x55 | 1 | 154 | omitted (data) | 373 | 528 |
| conv1b | 48x55x55 | 1 | 152 | omitted (data) | 355 | 507 |
| conv2a | 128x27x27 | 1 | 353 | 261 | 767 | 1382 |
| conv2b | 128x27x27 | 1 | 355 | 258 | 768 | 1381 |
| conv3a | 192x13x13 | 2 | 242 | 157 | 547 | 946 |
| conv3b | 192x13x13 | 2 | 243 | 155 | 547 | 945 |
| conv4a | 192x13x13 | 1 | 182 | 117 | 413 | 712 |
| conv4b | 192x13x13 | 1 | 183 | 116 | 413 | 713 |
| conv5a | 128x13x13 | 1 | 120 | 77 | 272 | 470 |
| conv5b | 128x13x13 | 1 | 121 | 77 | 272 | 470 |

## Final gates

`~/.dotnet/dotnet build`: exit 0, **0 warnings / 0 errors**, MSBuild 1.53 s.
`~/.dotnet/dotnet build -c Release`: exit 0, **0 warnings / 0 errors**, MSBuild 1.34 s.
Final suite runs execute the compiled Release test assembly without rebuilding.
CLI state, packages, HTTP cache and temporary output were explicitly directed to
`.scratch/` (`NUGET_HTTP_CACHE_PATH` included); authored/build output stays in this
repository. No command wrote to the original tree or another source tree.

| Fresh process | Exit | Total tests | Suite time | Process wall time |
| --- | --- | --- | --- | --- |
| 1 | 0 | 177 + spec checks | 8.662 s | 8.687 s |
| 2 | 0 | 177 + spec checks | 8.714 s | 8.738 s |
| 3 | 0 | 177 + spec checks | 8.766 s | 8.840 s |

Total final process wall time: 26.266 s. Each suite retains all existing 131 tests
and adds 46 groups: 14 fixture/partialSum combinations, 16 numerical combinations,
5 contract groups and 11 real-geometry/boundary groups. Each process rechecks
569 parsed parameters, 1,470 shape fields and 455 source citations. Final PIDs
are distinct; full output and process timing records are in ignored
`.scratch/anx5/final-run-*.log` and `final-runs.json`.

Integrity checks passed:

- All 274 original-tree files match the pre-implementation baseline by relative
  name, byte size, nanosecond mtime and SHA-256; no additions or removals.
- Candidate A archive digest matches ANX-1; all extracted source bytes equal the
  archive. Archive and extracted files are read-only under ignored `.scratch/`.
- Re-running `make_conv_fixture.py` yields byte-identical JSON.
- `git diff --check` passes; all six changed/new deliverable files use LF and
  contain no local home paths, environment paths, hostnames, IPs or secrets.
- No CUDA source is among tracked or unignored new files; git index is unchanged.
  No commits, pushes, stashes or process-killing commands were performed.
- ANX-2..4 source and fixtures are unchanged; their existing tests pass in each run.

The only brief correction is the two exactly tiling miniature geometries explained
above; both requested cases and additional true partial-window cases are proven.
Groups/sparse remain explicitly unsupported as permitted by the brief. There is
no unfinished item. Historical CUDA bitwise parity and optimized performance
remain outside this scalar-reference story.
