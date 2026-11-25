# ANX-6 — Optimised CPU convolution

Started 2026-11-25 at approximately 19:19 UTC; completed at 19:29 UTC (~10 min).
Base `8673c97`. Report was the first deliverable created and was updated during
implementation and verification. All seven items and gates passed. The scalar
oracle is unchanged. Changes remain in the working tree; no commits or pushes.

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. Optimised convolution | done | Managed SIMD forward and all gradients; same contract and validation; exact fixtures, numerical/contract and full real-size parity passed. | 19:19–19:25 UTC (~6 min) |
| 2. Determinism | done | Raw float bits at 1, 2 and 24 threads, repeated 24; randomized and all ten real-layer workloads passed. | 19:22–19:25 UTC (~3 min) |
| 3. Bounded memory | done | Explicit fixed panels and worker counts; formulas and per-layer table below; all layers executed at N=128. Peak 10.125 MiB. | 19:21–19:26 UTC (~5 min) |
| 4. Scalar parity | done | 66 randomized + 3 panel/SIMD cases; all seven exact fixtures; all ten real layers at N=2; zero tolerance, all four operations. | 19:22–19:25 UTC (~3 min) |
| 5. Benchmark | done | All ten layers, N=128, warmup + 3 samples/op, batch-2 scalar comparisons, CPU/thread/FLOP details and projections below. | 19:23–19:26 UTC (~3 min) |
| 6. Fast default suite | done | Three fresh default Debug runs: 5.149 / 4.970 / 5.176 s including build/startup, each 282 tests + spec checks. All ANX-5 assertions retained in documented modes. | 19:21–19:27 UTC (~6 min) |
| 7. Report and gates | done | Debug/Release builds 0 warnings/errors; default passes 3x; full Release passes; original and scalar unchanged; final integrity checks passed. | 19:19–19:29 UTC (~10 min) |

Intervals are approximate elapsed wall time, including the item's successful proof;
they overlap because implementation, source review and verification were interleaved.

## Progress evidence

- 19:23 UTC: first Debug suite passed, 282 tests plus spec checks, 4.758 s.
  Both engines passed all seven exact fixtures, numerical and contract checks.
  Initial optimized random/determinism proof passed 70 groups with tolerance 0.
- Initial Release benchmark (one warmup and one sample/op) completed all ten
  layers at N=128 and scalar/optimized N=2. Preliminary conv total 7.293 s,
  64.56 GFLOP/s. This exposed repeated coordinate arithmetic in weight gradients;
  an additional bounded per-worker image panel now amortizes it over filters.
  The subsequent full parity proof and final three-sample measurements passed,
  as recorded below.

- Full Release parity passed after the image-panel optimization: 303 tests plus
  spec checks, 18.202 s. Every element in all ten real layers at N=2 matched
  raw float bits; optimized thread counts 1, 2 and 24 and repeated 24 passed.

## Design and accumulation order

`OptimizedConvolution` has the scalar API plus optional `maxDegreeOfParallelism`
(default -1, the runtime processor count). `FromLayer` also accepts it. Canonical
strided rows, offsets, backing-alias rejection, geometry validation, named group/
sparse rejection, unshared biases and validation before any fan-in write are
retained. Validation is deliberately duplicated to leave `ScalarConvolution.cs`
byte-for-byte unchanged; there is no shared arithmetic with the oracle. The test
adapter runs the existing ANX-5 assertions against either implementation.

Forward packs a K-by-64 panel of flattened (module,case) positions, then performs
managed `Vector<float>` dot products across independent positions. A fixed number
of workers owns disjoint cyclic tiles and reuses one panel each. Sequential input
fan-in adds bias on the first write only. Input gradients assign independent
input rows to workers and vectorize contiguous cases. Weight gradients transpose
at most 4096 upstream (module,case) positions to filter-fastest panels. Each worker
packs that position span for one weight row at a time, reusing coordinates across
all filter vectors. Chunk and total arrays retain every weight accumulator across
panel boundaries. Bias gradients parallelize independent bias rows.

This is a tiled managed matrix-product kernel, with direct input adjoints. It
uses no native libraries, unsafe code, atomics or horizontal SIMD reductions.
Separate multiply and add operations deliberately preserve scalar rounding;
FMA could change rounding and is not used. This trades some possible
throughput for a stronger oracle comparison and a simple determinism guarantee.

| Operation | Fixed reduction order | Independent parallel/SIMD dimension |
| --- | --- | --- |
| Forward | channel, filter y, filter x; then bias; inputs in declared order | output tile / (module,case) lanes |
| Input gradient | module y, module x, filter | input row / case lanes |
| Weight gradient | consecutive partialSum chunk; module, case inside chunk; scale each completed chunk; sum chunks ascending | weight row / filter lanes |
| Bias gradient | module, case; then scale | bias row |

When partialSum=0 the one chunk covers all modules. Panel boundaries do not
reset or scale accumulators. `normalizationCases` and gradientCheck use the same
scale calculation as the scalar reference. scaleTargets=0 discards the old target
without reading it; 1 adds once after the completed reduction. These stages match
`src/layer.cu:823-895`, especially `src/layer.cu:869` and `:875-877` for scale-before-chunk-combine, and
A:`src/cudaconv2/filter_acts.cu:129-154` for zero padding. Candidate A was fetched
read-only into ignored scratch storage and its archive SHA-256 verified against
`docs/spec/CUDACONV2.md`. No candidate source is included in deliverables.

## Parity and determinism proof

Tolerance is **zero**, stronger than a floating-point error allowance: accumulation
order does not differ from the oracle. All comparisons use raw float bits, including
retained NaN targets in scale-1 tests. The seven independently generated rational
fixtures remain exact on both paths (14 fixture/partialSum groups per engine).
Both engines also run all 3,824 independent finite-difference checks (the ANX-5
numeric tolerance is unchanged), cancellation/scale-before-combine witnesses,
virtual-batch, unshared bias, named rejection, alias/layout and sentinel contracts.
No ANX-5 behavior or proven bug needed correction.

Random seeds 61000–61065 cover 66 geometries, filter sizes 1..11, valid strides 1..4,
image widths filterSize+1..filterSize+8, paddingStart 0/-1/-2, channels 1..5, filters 1..13,
cases 1..5, exact and partial windows, partialSum 0 and module-width divisors.
Every third case has a second input with different geometry. All outputs of
forward/explicit forward-input, input/weight/bias gradients are compared, with
scaleTargets 0/1, NaN discard/retention, training/GC and virtual normalization.
Seeds 69001–69003 additionally cover a 12675-position reduction split across
4096-position panels and 195-position partial chunks, N=17 SIMD tails, and N=128.
Shared and unshared bias paths are both exercised.

Each workload executes at 1, 2, all 24 threads, and again at 24; every output bit
is compared to the scalar oracle and thus to the other thread counts. The opt-in
mode adds all ten real 145 layers at N=2, seeds 69570–69579: all forward, input
(conv2–5), weight and bias elements, both conv3 inputs. It also retains every
original full-size ANX-5 scalar Euler/adjoint/shape/finite assertion. The default
runs those real-size assertions on the optimized path. Both towers retain the
full actual convolution proof of 216..223 plus three zeros, separately for right
and bottom and all 11 taps; no boundary assertion was removed or reduced.

## Workspace bounds

Let K=channels*filterSize² for one input, F=filters, M=modules, N=cases,
W=min(requested workers, relevant independent work count), T=min(4096,M*N).
Forward live float payload is `4*K*64*min(workers,ceil(M*N/64))` bytes.
Weight-gradient live float payload is `4*(T*F + 2*K*F + T*min(workers,K))` bytes:
upstream panel, chunk+total accumulators and per-worker image panels. Input and
bias gradients allocate no float workspace. Matrix operands and result storage,
small array/object headers and runtime task scheduling are excluded. Arrays are
per call, become collectible on return, and are not retained in a pool; GC heap
reservation/temporary dead arrays can exceed live workspace. Fan-in inputs run
sequentially, so the peak is the maximum single-input workspace, not their sum.
These formulas are bounded independently of batch size once panels fill. They
also explicitly expose the linear worker-count cost. Batch-128 tables below use
24 workers. There is no full-batch im2col allocation.

| Layer | Forward MiB | Weight MiB | Input/bias MiB | Peak MiB |
| --- | ---: | ---: | ---: | ---: |
| conv1a | 2.127 | 1.258 | 0 | 2.127 |
| conv1b | 2.127 | 1.258 | 0 | 2.127 |
| conv2a | 7.031 | 3.547 | 0 | 7.031 |
| conv2b | 7.031 | 3.547 | 0 | 7.031 |
| conv3a | 6.750 | 5.062 | 0 | 6.750 |
| conv3b | 6.750 | 5.062 | 0 | 6.750 |
| conv4a | 10.125 | 5.906 | 0 | 10.125 |
| conv4b | 10.125 | 5.906 | 0 | 10.125 |
| conv5a | 10.125 | 4.062 | 0 | 10.125 |
| conv5b | 10.125 | 4.062 | 0 | 10.125 |

Conv1 full-batch im2col would be 536.169 MiB (562,214,400 bytes) per input;
its actual peak live workspace is 2.127 MiB at 24 workers. The largest layer peak
is 10.125 MiB. The N=128 benchmark executes these bounds at the requested batch
size, including both inputs of conv3, without full-batch lowering.

## Batch-128 benchmark

CPU: Intel Core Ultra 9 285HX; 24 logical processors, 24 requested workers,
8 float SIMD lanes (AVX2), .NET 10.0.403 SDK, Release. No concurrent benchmark or
test process ran during final measurements. One untimed warmup per operation,
then three samples; tables use medians. All allocations, panel packing, task
scheduling and final writes are timed. Small batches are independently warmed.
These are isolated operation microbenchmarks, not a sustained full trainer run;
JIT tiering, mixed performance/efficiency cores, thermals and sub-millisecond
scheduling noise can affect results. Raw sample arrays are in the ignored
`.scratch/anx6/benchmark-final.json` artifact. Benchmark mode is opt-in and
its JSON export records CPU model only, never machine identity.

FLOP formula: forward and weights use `2*N*M*F*sum(K_i)` (multiply and add counted
separately, including zero padding). Input gradients use
`2*N*F*sum(channels_i*valid spatial taps_i)`, counting only in-image taps visited
by the direct adjoint; enumerate modules and filter coordinates to count taps.
Bias uses `N*M*F` reduction additions. Small bias adds, normalization/chunk-combine
operations, target additions and address arithmetic are excluded from these
conventional dominant arithmetic counts. conv1 data gradients are omitted;
weight and bias gradients for both conv1 towers are included (manifest's initial
trainable setting, rather than the later frozen historical rates).

Each operation cell is **milliseconds / achieved GFLOP/s**.

| Layer | Forward | Input gradient | Weight gradient | Bias gradient |
| --- | ---: | ---: | ---: | ---: |
| conv1a | 121.682 / 110.89 | omitted (data) | 288.213 / 46.82 | 2.269 / 8.19 |
| conv1b | 112.341 / 120.11 | omitted (data) | 301.567 / 44.74 | 2.028 / 9.16 |
| conv2a | 210.139 / 136.41 | 228.906 / 114.34 | 341.323 / 83.98 | 1.393 / 8.57 |
| conv2b | 202.531 / 141.54 | 237.512 / 110.20 | 323.830 / 88.52 | 1.214 / 9.84 |
| conv3a | 141.485 / 135.27 | 129.351 / 133.17 | 197.872 / 96.72 | 0.736 / 5.64 |
| conv3b | 141.237 / 135.51 | 131.433 / 131.06 | 206.017 / 92.90 | 0.534 / 7.78 |
| conv4a | 103.435 / 138.77 | 94.129 / 137.25 | 132.193 / 108.58 | 0.657 / 6.32 |
| conv4b | 104.959 / 136.76 | 93.202 / 138.62 | 132.798 / 108.09 | 0.576 / 7.21 |
| conv5a | 75.128 / 127.37 | 61.653 / 139.70 | 102.994 / 92.91 | 0.398 / 6.96 |
| conv5b | 78.433 / 122.01 | 60.998 / 141.20 | 99.122 / 96.54 | 0.256 / 10.83 |

### Scalar comparison at batch 2

Cells give **scalar ms / optimized ms / speedup** using three-sample medians,
same CPU and 24 optimized workers. Tiny bias workloads can be slower in the
parallel path; the measured costs, including these regressions, are retained.

| Layer | Forward | Input gradient | Weight gradient | Bias gradient |
| --- | ---: | ---: | ---: | ---: |
| conv1a | 154.892 / 4.234 / 36.59x | omitted (data) | 203.717 / 5.566 / 36.60x | 0.234 / 0.672 / 0.35x |
| conv1b | 156.258 / 3.632 / 43.03x | omitted (data) | 202.729 / 5.425 / 37.37x | 0.230 / 0.066 / 3.49x |
| conv2a | 368.108 / 6.885 / 53.47x | 169.856 / 27.745 / 6.12x | 418.069 / 6.139 / 68.10x | 0.152 / 0.107 / 1.43x |
| conv2b | 355.579 / 6.851 / 51.90x | 170.684 / 27.374 / 6.24x | 417.427 / 6.256 / 66.72x | 0.155 / 0.036 / 4.37x |
| conv3a | 257.856 / 9.349 / 27.58x | 104.680 / 21.301 / 4.91x | 302.559 / 5.478 / 55.23x | 0.068 / 0.027 / 2.48x |
| conv3b | 259.344 / 10.539 / 24.61x | 104.968 / 19.011 / 5.52x | 299.347 / 5.328 / 56.18x | 0.063 / 0.036 / 1.73x |
| conv4a | 184.985 / 6.968 / 26.55x | 78.605 / 13.482 / 5.83x | 227.690 / 3.730 / 61.04x | 0.057 / 0.024 / 2.35x |
| conv4b | 194.365 / 5.840 / 33.28x | 78.828 / 12.939 / 6.09x | 225.654 / 4.104 / 54.99x | 0.057 / 0.039 / 1.44x |
| conv5a | 122.065 / 3.941 / 30.97x | 52.666 / 9.084 / 5.80x | 152.052 / 2.677 / 56.81x | 0.040 / 0.025 / 1.62x |
| conv5b | 131.967 / 3.809 / 34.65x | 52.758 / 10.841 / 4.87x | 150.715 / 3.121 / 48.29x | 0.039 / 0.022 / 1.81x |

The non-bias batch-2 operations speed up **4.87–68.10x**.
The coordinate-reuse change reduced the preliminary convolution step from 7.293 s
(one-sample exploratory run) to 4.465 s (final three-sample medians).
This is directional evidence rather than a controlled isolated speedup estimate.

### Training projections and feasibility

Sum of per-operation medians, all ten conv layers: **4.465 s per batch 128**,
**470.830 GFLOP**, aggregate **105.46 GFLOP/s**, or 28.67 images/s
for convolution alone. Forward, input adjoint, weights and bias are executed
sequentially in this projection, with both tower layers included.

`DESIGN.md` assumes 4.3 GFLOP/image for full forward+backward. At N=128 that is
550.4 GFLOP; the measured convolution arithmetic is 85.54% of that
approximate full-step count. Assuming the remaining operations sustain the same
aggregate rate gives **5.219 s per full training step**, of
which 4.465 s (85.54%) is convolution.
For 1.2M images * 90 epochs / 128 = 843750 steps, convolution alone costs
**43.60 days**, and that same-rate full-step extrapolation costs
**50.97 days**. These exclude data preparation, snapshots,
validation, optimizer/memory traffic and throughput changes in a complete graph;
there is no implemented trainer to measure yet. The convolution-only number is
a time floor under these measurements; the full-step number is an assumption,
not a prediction of achieved end-to-end speed.

CPU execution is practical for development and smaller experiments, but a roughly
51-day arithmetic-only full run is expensive. **The GPU fallback is likely needed
for a practical full 90-epoch ImageNet run.** Story 15 still needs to measure the
complete sustained step before choosing the backend; this story adds no GPU code.

## Test modes and final gates

```sh
dotnet build
dotnet build -c Release
dotnet run --project tests/AlexNetSharp.Tests
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --full-scalar-parity
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-conv
```

The executable used for gates was the installed SDK's `dotnet`. NuGet HTTP cache
(`NUGET_HTTP_CACHE_PATH`), packages, CLI state and temporary files were explicitly
redirected under ignored `.scratch/`; authored files and build outputs stay in the
repository. No command wrote outside this repository. Original sources were
opened read-only. Benchmark mode bypasses the default gate and accepts optional
`--conv-threads`, `--conv-repeats`, and `--conv-output` arguments, documented in
README. Default repeats are three; default workers are the runtime CPU count.

| Mode | Assertions run |
| --- | --- |
| Default Debug or Release | Existing 131 parser/shape/tensor tests + spec checks; ANX-5 exact, numerical and contract groups on both engines (70 groups); optimized real-size Euler/adjoint/shape/finite and full boundary proof (11); randomized/panel/SIMD parity+determinism and direct validation (70). Total 282. |
| `--full-scalar-parity` | Everything above; original ANX-5 full-size scalar and boundary assertions (11 additional), full elementwise scalar parity and thread determinism on all ten real layers (10 additional). Total 303. |
| `--benchmark-conv` (Release) | Separate timed all-layer N=128 and scalar-vs-optimized N=2 measurements; no default gate assertions are replaced by benchmark timing. |

No ANX-5 assertion was deleted. Its 177-test suite is fully represented in the
opt-in mode; the default replaces its full-size scalar checks with the same
assertions on the optimized implementation. Both engines' miniature checks remain
in the default. Spec checks still verify 569 parser parameters, 1470 shape fields
and 455 original source citations on every suite run.

Final builds: Debug **0 warnings / 0 errors**, MSBuild 2.37 s (process 2.488 s);
Release **0 warnings / 0 errors**, MSBuild 0.58 s (process 0.705 s).

| Fresh process | Exit | Tests + spec checks | Suite time | Process wall time including build/startup |
| --- | --- | --- | ---: | ---: |
| default-debug-1 | 0 | 282 | 4.152 s | 5.149 s |
| default-debug-2 | 0 | 282 | 4.017 s | 4.970 s |
| default-debug-3 | 0 | 282 | 4.202 s | 5.176 s |
| full-release | 0 | 303 | 18.442 s | 19.465 s |

All three default invocations used the requested command in fresh processes and
finished comfortably below the 20 s target. Full Release parity passed twice
(first 18.202 s suite, final 18.442 s suite). Gate logs and distinct process IDs
are preserved in ignored `.scratch/anx6/gates.json` and the matching log files.

Final integrity proof:

- All 274 original-tree files match their pre-implementation relative names,
  sizes, nanosecond mtimes and SHA-256 hashes, with no additions/removals.
- Candidate A archive SHA-256 matches the specification; all three extracted
  kernels equal archive bytes and have write permissions removed.
- `ScalarConvolution.cs` is byte-identical to HEAD. ANX-2..4 implementation,
  golden fixtures and specification files are unchanged.
- `git diff --check` passes, and no changes are staged. All eight deliverable
  files use LF, contain no home/environment paths, hostnames, IPs or secrets,
  and include no copied CUDA/candidate source. Scratch assets are ignored.
- No original or external tree was modified; no commits, pushes, stashes or
  process-killing commands were performed.

No brief item is unfinished. There was no ANX-5 specification deviation or bug
requiring a correction. Historical CUDA bitwise parity is still outside the
available evidence; the proof here is bitwise parity with the scalar CPU oracle.
