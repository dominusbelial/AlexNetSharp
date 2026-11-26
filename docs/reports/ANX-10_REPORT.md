# ANX-10 — Dropout, softmax and logistic-regression cost

Started: 2026-11-26T01:07Z. Base: `8e2ecef`. Report created before implementation.
All six items and required gates passed. Finished at approximately 01:26 UTC (~19 min elapsed). All citations are upstream-relative.

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. Scalar reference | done | Dropout/softmax/logreg/multiview fixture, gradient, contract and real-geometry tests passed in initial Debug and full Release runs. | ~01:07–01:20 UTC (13 min, source review through first full proof) |
| 2. Optimized path | done | Independent flat-stride case kernels; raw-bit parity at 1/2/24/all/repeated-all; full general backward N=128 passed in Release. | ~01:09–01:20 UTC (11 min) |
| 3. Independent hand fixtures | done | Independent Python fixtures passed on both engines, including all requested singular/tie/multiview cases; exact reduction-order witness and both RNG endpoints passed in final gate runs. | ~01:12–01:24 UTC (12 min, including final witness/endpoint proofs) |
| 4. Numerical gradients | done | 186 independent float64 central differences passed; worst absolute error 3.80130628e-8. | ~01:14–01:18 UTC (4 min) |
| 5. Real geometry and timing | done | Four hs/probs/logprob batch-128 proofs, parsed fused dispatch, actual FC-pair fan-out and batch-128 benchmark passed. | ~01:14–01:24 UTC (10 min, including full-mode proofs and timing) |
| 6. Report and gates | done | Debug/Release builds clean; 683 tests + spec 3× Debug, 710 full Release; benchmark and original/candidate/fixture/hygiene audits passed. | ~01:07–01:26 UTC (19 min, report throughout; final gates/audit ~01:22–01:26) |

## Source audit

The original tree was baselined before implementation (274 files, sizes, nanosecond
mtimes and SHA-256). Candidate A's archive matches the specified SHA-256
`b55bd30ceb2c69c9540ae7e7c2f6f6cd075551071c351e57e3b7a5d7bb10f819`;
helpers were extracted read-only under ignored `.scratch/anx10/`.
No candidate code or local environment path is in deliverables.

Key findings verified directly: disabled hs still scales by keep; dropout adds
for any nonzero scaleTargets, whereas softmax/logreg add only for exactly 1.
The general softmax kernel uses an ascending j loop separately for every c,
then multiplies by p_c. This is algebraically the usual Jacobian product but has
a distinct float32 reduction order. The fused path emits negative loss gradients
without batch averaging. For the softmax and cost class-axis reductions, A's
`kDumbAggCols` seeds from the first class, then uses `new > old ? new : old`,
not the generic finite max seed used by other reduction helpers.

## Verified semantics and boundaries

All arithmetic buffers are float32. Canonical dropout/cost matrices are
`(features-or-classes,cases)` C-order; softmax is native `(cases,classes)`
F-order. `TransposeView()` bridges these without copying, as in ANX-4.
Strided views and nonzero offsets are supported; softmax/cost output backing
aliases are rejected, while dropout permits exact in-place index mappings.
No graph executor, optimiser, data/crop pipeline, web or GPU code is added.

| Behavior | Implemented contract | Source evidence |
| --- | --- | --- |
| Dropout placement | Four 2048-wide layers follow detached FC ReLUs; hs itself changes to C-order. | `layer.py:627-645,689-709`; `src/layer.cu:1121,152-155`; `layers/layers-145.cfg:256-292` |
| Training mask | One independent draw for every feature/case; strict `u<keep`, saved binary `_sexMask`, unscaled multiplication in forward and backward. Zero times NaN/inf remains NaN. | `src/layer.cu:1127-1143`; A:`src/nvmatrix/nvmatrix.cu:326-330,623-624,830-835`; A:`include/nvmatrix/nvmatrix_operators.cuh:111-120,235-239` |
| Disabled / non-training hs | Multiply by keep for both directions, including disabled training, test, GC, feature generation and multiview. No RNG draw. | `src/layer.cu:1127-1134,1138-1151`; pass constants `include/util.cuh:55-67` |
| hs target accumulation | Any nonzero scaleTargets adds; zero never reads the old target. Training uses product+target, other modes target+scaled-upstream, preserving the original expression order. | `src/layer.cu:1139-1151`; `include/neuron.cuh:36-54` |
| Softmax forward | Per case: max starts from class 0; classes ascend; subtract max, MathF.Exp for all classes, sum starting at class 0, divide. Max uses `new>old?new:old`, retaining the first tied maximum. | `src/layer.cu:1020-1026`; A:`src/nvmatrix/nvmatrix.cu:700-716`; A:`include/nvmatrix/nvmatrix_kernels.cuh:544-556`; A:`include/nvmatrix/nvmatrix_operators.cuh:305-313` |
| Float math / singularities | Original exponential is **`__expf`**, log is **`__logf`**, division is **`__fdividef`**. Port uses MathF.Exp/MathF.Log/ordinary float division. Finite large positive/negative logits are stable; all -inf, +inf maxima, or any NaN lead to NaN probabilities. No special one-hot replacement. | A:`include/nvmatrix/nvmatrix_operators.cuh:34-52,228-232`; `src/layer_kernels.cu:106,396,439` |
| General softmax backward | For each output class c, seed v=+0; ascending j: `v += dy_j * (indicator(j==c)-p_j)`; then `v *= p_c`. Keep this O(C²) order instead of reassociating into dy-dot(dy,p). | `src/layer_kernels.cu:479-498` |
| Fused condition | `_doLogregGrad` (default true), exactly one consumer, that consumer type `cost.logreg`, same device. The route captures its coeff and reads its first input as labels. | `src/layer.cu:1017,1034-1040` |
| Fused sign / averaging | `coeff*(indicator(label==c)-p_c)` is the **negative** derivative of positive NLL. No `1/N` here. Parameter callers later divide by N (or eps/N); no optimiser is implemented here. | `src/layer_kernels.cu:456-469`; `src/layer.cu:654,664-671`; `src/weights.cu:145-152` |
| Cost's independent skip test | Cost skips its probability-gradient write if producer has at most one consumer, is softmax and is on the same device. This test does not inspect `_doLogregGrad`. Forced general softmax needs explicit probability upstream. Otherwise cost computes coeff*indicator/p for every class, including 0/0 on nonlabel zeros. | `src/layer.cu:1845-1856`; `src/layer_kernels.cu:429-445` |
| Softmax/cost target accumulation | Add old target only for `scaleTargets==1`; every other coefficient overwrites. This differs from hs's nonzero condition. | `src/layer.cu:1038-1040,1855`; `src/layer_kernels.cu:440-444,465-469,492-496` |
| Cost outputs | Unweighted `[-sum(log p_label), N-sum(top1 credit), N-sum(topk credit)]`. Logs are natural float32. No clipping/safelog: label p=0 gives log=-inf and NLL=+inf. Coeff affects backward, not the reported cost. | `src/layer.cu:1825-1836`; `src/layer_kernels.cu:96-119,387-418`; unused-for-logreg safelog `include/layer_kernels.cuh:40-42` |
| Fractional ties | B=#classes strictly above labelp, E=#classes equal to labelp. Top1 credit is 1/E if labelp=maxp, else 0. Topk credit: slots=k-B; 0 when slots<=0, 1 when E<=slots, otherwise slots/E. topk=1 reuses top1. | `src/layer_kernels.cu:108-117,410-417`; `src/layer.cu:1827-1836` |
| NaN cost input | Retain literal comparisons, not an assumed probability validity repair. A NaN labelp gives NaN log, top1 credit 0; for k>1 B=E=0 gives topk credit 1. | `src/layer_kernels.cu:103-117` |
| Multiview ownership | Start copies first view and count=1; continue sums and increments count; only end multiplies saved sum by float32(1.0/count) then computes. Intermediate calls retain previous cost; a new start overwrites prior average or incomplete sum. Start+End handles one view. | `src/layer.cu:1804-1837`; flags `include/util.cuh:59-67` |
| Case totals | Ascending case float32 sums, then loss negation in float32; error counts subtract the widened totals in double. At N=128 this follows A's host path; batches >4096 use a different historical GPU reduction tree, deliberately not reproduced. | `src/layer.cu:1833-1836`; A:`src/nvmatrix/nvmatrix.cu:936-979`; A:`include/nvmatrix/nvmatrix_kernels.cuh:60`; A:`src/common/matrix.cpp:563-564,615-621` |

The independent tiny float32 rounding witness has p=[.125,.25,.625] and
upstream=[-55865.36328125,9107.64453125,68786.7421875]. Original-order gradient
is [-11768.8515625,-7294.45263671875,19063.3046875]; the algebraically simplified
O(C) variant changes all three float32 answers. Both engines require exact fixture
values. For regular positive probabilities the general and fused loss paths
agree within float32 tolerance; at zero label probability fusion stays finite
while division followed by general backward yields NaNs.

## RNG parity gap and saved state

The historical CUDA RNG stream **cannot be reproduced**. This is a documented
parity gap, not a bug: A stores per-device CURAND thread states, invokes
`curand_uniform` and advances them according to its launch pattern
(A:`src/nvmatrix/nvmatrix.cu:246-254,271-283`;
A:`include/nvmatrix/nvmatrix_kernels.cuh:613-617`). No historical linked CUDA
binary/checkpoint is available for trajectory comparison. Intrinsic rounding,
possible CUDA multiply/add contraction and large-batch tree sums are further
historical floating-point limits. CPU scalar/optimized bit parity is required.

`IDropoutMaskSource` writes binary masks; `FixedDropoutMaskSource` copies injected
values. The default training building block `SeededDropoutMaskSource` is a
counter-addressed SplitMix64 finalizer, specified completely in its XML comment:

- index = drawOffset + feature*cases + case, modulo 2^64;
- z = seed + 0x9E3779B97F4A7C15*(index+1), modulo 2^64;
- z = (z xor (z>>30))*0xBF58476D1CE4E5B9, modulo 2^64;
- z = (z xor (z>>27))*0x94D049BB133111EB, modulo 2^64;
- z = z xor (z>>31); u=(top24bits(z)+1)*2^-24, exactly representable in (0,1];
- mask = u<keep; advance drawOffset by the number of elements after each fill.

The independent Python oracle pins ordinary seeds, cursor wraparound and both
uniform endpoints. Strict equality drops the value; u=1 can drop even with
keep=1, matching the source's strict predicate. Retain seed and drawOffset for
future checkpoint work (ANX-14); this story adds no checkpoint format. Each layer
should own its source/stream. Sources and layer forward state are sequentially
owned, not concurrently callable. Worker count changes only execution scheduling.

Both dropout engines remember the most recent completed forward's mode, shape and
mask. Backward reuses them without drawing, including repeated backward calls;
a new training forward refreshes the mask. Non-training/disabled calls never
advance RNG. This CPU API rejects backward before forward or with a different
shape. The multiview API requires Start after End and consistent dimensions;
original code relied on its caller rather than checking these invalid sequences.
Label matrices follow the original truncation-to-int semantics for finite values;
unsafe nonfinite/out-of-range indices are rejected before writes. Positive
matrix dimensions and topK>=1 are explicit CPU API boundaries, not changes to
the existing parser. No keep-range restriction was added.

## Proof inventory and tolerances

`tests/fixtures/make_dropout_softmax_cost_fixture.py` uses only Python's standard
library, independent of the C# implementation. Main oracles use float64 or exact
binary fractions; the one explicitly named rounding witness uses struct-rounded
float32 operations to pin the source reduction. Strict JSON strings represent
NaN/±infinity. Comparisons explicitly check NaN and infinity, avoiding the existing
harness's permissive NaN tolerance trap.

- 19 dropout fixtures on both engines: all-keep/all-drop/mixed across train,
  eval, disabled train, GC, feature generation and multiview; special NaN/inf
  mask multiplication. Full forward/backward and target scales 0/1 are checked.
- Six softmax fixtures on both engines: large ±logits, equal logits, mixed/all
  -inf, NaN first/middle, +inf, and logits below the generic finite max seed.
  General backward, overwrite/add and NaN target behavior are checked.
- Two fused fixtures on both engines: regular multi-case coeff=.75 agreement
  with probability-backward+general-softmax, and zero labelp stability; all
  probability-gradient elements, including nonlabel zeros, are checked.
- Five cost fixtures on both engines: top-5 boundary tie (two above, six tied,
  three slots => credit .5), tied maxima, topk=1, uniform probabilities, zeros,
  NaN/negative inputs, multiple cases and labels 0/999 at 1000 classes.
- Two multiview fixtures on both engines, with two and ten distinct views:
  intermediate deferral, final mean probabilities/cost, preserved source,
  fresh-start reset, retained prior cost and one-view Start|End.
- Five independent RNG fixtures, including both endpoints and cursor wrapping;
  one exact float32 general-backward rounding witness on both engines.
- 24 randomized geometries: cases 1/2/3/5/17/128, classes 1..17, finite and
  NaN/inf/zero inputs, coefficient 0/1/-.75/.3 and target scales 0/1/.5/-1.
  All operations (including ten-view sums/averages) match raw bits at
  1/2/24/all/repeated-all workers. Inputs are checked unchanged.
- Contract groups cover exact/partial aliases, strided C/F views with sentinels,
  labels/layouts/shapes/worker counts, state lifecycle, no-draw modes, mask
  reuse, per-op scale conditions, every fused-condition component and actual
  dispatcher routing from parsed 145 definitions, with the flag on/off.

Dropout finite hand results and RNG values require exact equality; the rounding
witness requires exact float32 equality. Other float64 fixture results use
`abs(error)<=2e-6*(1+abs(expected))` for float32 exp/log/division and sum rounding.
NaNs and infinities require the expected kind/sign. Scalar/optimized and thread
comparisons require **identical raw bits**, including NaN representations and
signed zero; no tolerance.

Numerical seeds 100201/100202/100203, cases 2/3 and classes 3/4/5, cover every
logit coordinate for general softmax under dot(dy,p) and fused softmax+positive
NLL, plus every fixed-mask dropout input. Central differences perturb independent
**float64** copies, epsilon=1e-5; the objectives use Math.Exp/Math.Log, never the
port kernels. Both engines pass all **186 comparisons** at
`3e-6*(1+abs(numeric))`, worst absolute error **3.80130628e-8**. Fused gradients
are compared against **negative coeff times** the numerical positive-loss
derivative. Ties/zeros and singular logits are excluded from numerical checks:
classification tie metrics are nonsmooth and log/zero divisions are singular;
these behaviors have separate literal fixtures instead. Random logits are
moderate, distinct and have positive probabilities, so truncation/float32
perturbation error cannot undermine the double oracle.

## Real geometry, fan-out and execution

All four parsed hs layers execute full 2048×128 train forward/backward on scalar
and optimized engines at 1/2/all workers, with matching seed/masks, NaN overwrite,
and evaluation forward. Separate 2048×128 mask proofs explicitly compare 1 and
24 workers, successive fills, and resume from the same cursor. `probs` executes
1000×128 stable forward, `logprob` topk=5 cost, and fused backward at
1/2/24/all workers, labels 0/999, and coeff=1. The parsed sole local
`probs→logprob` edge selects and executes the fused dispatcher; the opt-out route
executes general backward with explicit upstream. A repeated-batch check proves
no accidental division by 128.

Actual `fc2048ba` and `fc2048bb` dX contributions feed both hs1 producers with the
ANX-9 input slots (ba: hs1a,hs1b; bb: hs1b,hs1a). Both consumers use scale=0
in separate per-device buffers; the producer merges local+remote with scale=1,
then applies its saved dropout mask. Both arrival orders pass, including sampled
independent scalar FC dots and every merged/masked element. For hs1a local=ba;
for hs1b local=bb. This preserves the ANX-4/9 per-device rule, not an invented
fixed global consumer arrival order (`src/layer.cu:287-295,358-377`).
Default uses N=3 for these actual FC products; full mode uses N=128. hs2a/b
have only fc1000 as consumer; hs2b's remote-only merge remains covered by the
unchanged ANX-4/9 proofs.

The O(C²*N) 1000-class general backward is behind `--full-scalar-parity`, where
all 128,000 gradient elements match scalar at 1/2/all workers. The initial full
Release proof (before the final extra rounding/endpoint fixture assertions)
passed **706 tests plus spec checks in 42.917 s**. Default initial Debug passed
**679 in 11.977 s**. Final gate counts/timings follow below.

Optimized kernels use flat backing arrays and explicit strides, parallelizing
**only cases**. Every reduction has fixed class or case order and separate
multiply/add; no atomics, FMA, shared accumulators, horizontal SIMD reductions,
or scheduling-dependent sums. Arithmetic loops are independent from the scalar
reference; only validation/state/dispatch contracts are shared. No operation is
skipped. This gives useful acceleration on this geometry without changing the
original general gradient's rounding order.

Dropout retains 4*features*cases bytes of mask storage, reused at the same shape.
Softmax forward/backward needs no large temporary beyond caller targets. Cost
allocates three N-element float arrays for per-case logs/credits (12*N bytes),
then serially reduces them. Multiview retains one class×case float sum/average,
replacing it on Start; cost allocation is the same. Small scheduling/read-only
wrapper objects are additional; workers do not replicate matrices. Reference
and optimized kernels share these bounds. Future net assembly owns lifetime,
remote scheduling and buffer reuse.

## Specification reconciliation

No ANX-2..9 implementation, test assertion or machine-readable fixture/spec was
changed. Program.cs only registers the new suite and opt-in benchmark. README
adds story status and reproduction. LEDGER.md now explicitly qualifies disabled
training, `_doLogregGrad`, operation-specific scale predicates, unweighted costs,
no log clipping, exact fractional ties and multiview sum-then-average lifecycle.
CUDACONV2.md §9 adds the complete individual helper-signature audit and precision
boundaries. These are evidence-backed clarifications/corrections of omissions;
ANX-4 layout/fan-out and ANX-7 neuron placement remain valid. No relevant helper
signature mismatch was found; ANX-8's unrelated minDiv mismatch remains recorded.


## Batch-128 timings

Release SDK 10.0.403, 24 logical processors/workers (Intel Core Ultra 9 285HX,
as recorded by earlier geometry reports on this machine). Two warmups followed
by three samples per operation; medians in milliseconds. Operands/targets are
preallocated; internal allocations, RNG, mask validation, scheduling and writes
are included. No builds/tests/other benchmark ran concurrently. Samples and
configuration are in ignored `.scratch/anx10/benchmark.json`.

| Layer/op | Scalar ms | Optimized ms |
| --- | ---: | ---: |
| hs1a train forward (RNG included) | 3.815 | 3.191 |
| hs1a backward | 0.501 | 1.155 |
| hs1a eval | 0.323 | 0.519 |
| hs1b train forward (RNG included) | 3.713 | 2.930 |
| hs1b backward | 0.496 | 0.960 |
| hs1b eval | 0.322 | 0.700 |
| hs2a train forward (RNG included) | 3.711 | 2.883 |
| hs2a backward | 0.498 | 0.723 |
| hs2a eval | 0.322 | 0.475 |
| hs2b train forward (RNG included) | 3.717 | 2.731 |
| hs2b backward | 0.498 | 1.013 |
| hs2b eval | 0.321 | 0.566 |
| probs forward | 0.714 | 0.840 |
| probs general backward | 191.665 | 17.557 |
| probs fused backward | 0.249 | 0.584 |
| logprob forward top5 | 0.635 | 0.336 |
| logprob probability backward | 0.250 | 0.391 |
| logprob ten-view accumulation + cost | 3.371 | 4.926 |

Parallel scheduling helps the expensive O(C²) general backward substantially,
and helps mask generation and top-5 cost. Several small operations are slower
with all workers (e.g. fused backward and evaluation); their scalar work is
already below one millisecond. Both paths are implemented for every operation;
none was skipped. Ten-view accumulation is memory/scheduling dominated here.
These are informational microbenchmarks, not sustained trainer throughput;
JIT/PGO, GC, mixed CPU cores and scheduling affect results.

## Final gates and integrity

Debug and Release builds both report **0 warnings / 0 errors**. MSBuild elapsed
1.67 s / 2.53 s (process 1.787 s / 2.641 s). SDK invoked from its installed
location, with CLI home, packages, NuGet HTTP cache, TMPDIR and all XDG runtime,
cache/config/data directories redirected under ignored `.scratch/anx10/`.

| Fresh process | PID | Exit | Tests plus spec | Suite seconds | Process seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Debug 1 | 303 | 0 | 683 | 12.530 | 12.607 |
| Debug 2 | 340 | 0 | 683 | 12.206 | 12.280 |
| Debug 3 | 378 | 0 | 683 | 12.002 | 12.066 |
| Full Release | 415 | 0 | 710 | 42.141 | 42.203 |

Each run checked 35 sections, 10 layer types, 15 weighted layers, 569 parser
parameters, 455 original citations and 1,470 linked shape fields. The new suite
has **119 default / 120 full-mode groups**. No prior assertion was removed or
weakened. Default Debug remains well under 20 seconds. Logs are
`.scratch/anx10/build-debug.log`, `build-release.log`, `default-debug-1..3.log`,
`full-release.log`, `benchmark.log`, with distinct process IDs and exit codes in
`gates.json`. That directory also holds raw timing and integrity evidence.

Integrity audit:

- All **274 original files** retain identical relative names, sizes, nanosecond
  mtimes and SHA-256; no additions/removals. No original source changed.
- Candidate archive SHA-256 matches the specified digest; all extracted helper
  bytes equal their archive entries and have write permissions removed. The
  original has no nvmatrix implementation tree, so A is explicitly cited.
- Regenerating the independent fixture is byte-identical; the digest is recorded
  in `.scratch/anx10/integrity.json`. No candidate source/archive or environment
  path appears among deliverables.
- Earlier implementation/tests/fixtures and machine-readable specifications are
  byte-unchanged except Program.cs registration. Only README and the two narrative
  specs are edited alongside this story's new files.
- All **20 deliverable files** use LF, final newlines and no trailing whitespace;
  private-path/hostname/IP scans and `git diff --check` pass. Nothing is staged.
- Task-authored files, downloads, builds, caches and temporary/runtime artifacts
  stayed in this repository; no outside file was edited. No commits, pushes,
  stash or process-killing commands were used. Changes remain in the working tree.

Reproduction from repository root (redirect CLI/package/runtime/temp state under
scratch as above; use the installed SDK):

```sh
dotnet build --disable-build-servers
dotnet build -c Release --disable-build-servers
# Run the compiled Debug assembly in three fresh processes:
dotnet tests/AlexNetSharp.Tests/bin/Debug/net10.0/AlexNetSharp.Tests.dll
dotnet tests/AlexNetSharp.Tests/bin/Debug/net10.0/AlexNetSharp.Tests.dll
dotnet tests/AlexNetSharp.Tests/bin/Debug/net10.0/AlexNetSharp.Tests.dll
dotnet tests/AlexNetSharp.Tests/bin/Release/net10.0/AlexNetSharp.Tests.dll --full-scalar-parity
dotnet tests/AlexNetSharp.Tests/bin/Release/net10.0/AlexNetSharp.Tests.dll --benchmark-dropout-softmax-cost --dsc-output .scratch/dropout-softmax-cost-benchmark.json
python3 tests/fixtures/make_dropout_softmax_cost_fixture.py
```

Benchmark optional flags: `--dsc-threads <positive integer>` and
`--dsc-output <path>`. Nothing is unfinished. Historical RNG/intrinsic/CUDA
rounding parity is explicitly outside the available source/execution evidence;
all requested CPU implementation and proof items are complete.
