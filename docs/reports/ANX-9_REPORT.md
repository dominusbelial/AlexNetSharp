# ANX-9 — Fully connected multi-input layers

Started: 2026-11-26T00:05:58Z. Base: `c9a8d95`. Report created before implementation.
Finished: 2026-11-26T00:20:01Z (~14 min elapsed).
All six brief items and verification gates passed. Changes remain in the working tree; no commits or pushes.

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. Scalar reference | done | Initial Debug: all fixtures, contracts, numerical and sampled real checks passed | 00:06–00:14 UTC (~8 min) |
| 2. Optimized deterministic path | done | Initial Debug: 36 randomized raw-bit parity groups at 1/2/24/repeated-24 workers passed | 00:08–00:14 UTC (~6 min) |
| 3. Independent hand fixtures | done | Nine float64 fixtures × two engines and independent fan-out fixture passed | 00:09–00:14 UTC (~5 min) |
| 4. Numerical gradients | done | 510 float64 central-difference comparisons; worst error 3.81698593e-8 | 00:10–00:14 UTC (~4 min) |
| 5. Real geometry and benchmark | done | Every-element N=3/9 Release parity at all worker counts; actual pool3/hs1 fan-out; five-layer N=128 timing passed | 00:12–00:16 UTC (~4 min) |
| 6. Report/spec | done | Source/matrix signature audit, coefficient/placement tables, spec additions, fixtures, tolerances and benchmark documented; integrity audit passed | 00:06–00:20 UTC (~14 min, throughout) |
| Build/test/integrity gates | done | Final Debug/Release builds 0 warnings/errors; three fresh Debug processes, full Release and integrity audit passed | 00:17–00:20 UTC (~3 min) |

## Progress

00:14 UTC: initial Debug build: 0 warnings/errors. Integrated default suite:
564 tests + specification checks in 10.746 s (12.618 s process including build).
Full Release checks and benchmark pending. Initial NuGet setup attempted its
migration mutex in a read-only runtime directory and failed without writing;
redirecting XDG_RUNTIME_DIR under scratch resolved it. All CLI/package/cache/temp
state is now explicitly repository-local.

## Provenance and scope

Read DESIGN.md, LEDGER.md's FC row and 145 rows 25–33, CUDACONV2.md,
and ANX-4/6/7/8 reports before implementation. Original sources were resolved
through ignored local.env and opened read-only. A pre-change baseline records
relative names, sizes, nanosecond mtimes and SHA-256 for 246 original files
(excluding .git). The original has no nvmatrix tree. Candidate A was fetched
into ignored `.scratch/anx9/candidate-a/`, its archive SHA-256 verified as
`b55bd30ceb2c69c9540ae7e7c2f6f6cd075551071c351e57e3b7a5d7bb10f819`,
and the needed matrix sources extracted and made read-only. **A:** citations
are relative to its `cuda-convnet/trunk/`; other citations are original-relative.
No candidate source is a deliverable.

No ANX-2..8 arithmetic, parser, shapes, machine-readable specs or existing
fixtures changed. Only registration in the test entry point changed there.
No optimizer, initialization, dropout, softmax/loss, graph assembly, data loader,
web or GPU code was added. FC applies no activation. Runtime FromLayer rejects
weightSource/sharing with a layer-named “unsupported in this port” error on
both engines; parsing remains compatible and 145 uses no shared FC weights.

## Source semantics

The FC parser requires positive outputs and builds one F-order `(numInputs[i],O)`
weight matrix per declared input, plus F-order `(1,O)` bias
(`layer.py:880-884`). initB defaults to zero (`:833`); the four FC2048 layers
explicitly set 1 and fc1000 leaves the default. The original initW default quirk
is retained: `:832` supplies the scalar 0.01 to a list reader whose `:113` calls
`.split(',')`, so omission raises a parsing error rather than initializing with
0.01. All five 145 FC sections explicitly supply two initW entries. Initializing
those values is outside this story; kernels use supplied parameter buffers.

| Behaviour | Port contract | Evidence |
| --- | --- | --- |
| Input/weight pairing | One W per input slot, in declared order; fc2048bb's slot 0 is hs1b, slot 1 hs1a | `layer.py:797-807,883`; `layers/layers-145.cfg:268,277` |
| Layout | Native X=(N,K), Y/dY=(N,O) F; W/dW=(K,O) F; bias/db=(1,O) F. Canonical (feature,N) C uses explicit TransposeView, same bytes | `src/layer.cu:152-155,635`; `layer.py:883-884`; A:`src/nvmatrix/nvmatrix.cu:530-550` |
| Forward first input | `acts.addProduct(X[0],W[0],0,1)` then `acts.addVector(b)` | `src/layer.cu:157-165,640-644`; A:`src/nvmatrix/nvmatrix.cu:221-236,652-657` |
| Remaining inputs | `acts.addProduct(X[i],W[i],1,1)`, ascending i, with no further bias addition | `src/layer.cu:162-165,641-644` |
| Bias placement | After the **first product**, before product 1; literal float32 witness `(16777216+1)+(-16777216)=0`, whereas bias-last gives 1 | `src/layer.cu:641-644`; executable contract test |
| Explicit input operation | scaleTargets=0 overwrites and adds bias; any nonzero scale scales old target and adds only the product. Whole Forward uses 0 then 1 | `src/layer.cu:640-644`; A:`src/nvmatrix/nvmatrix.cu:221-236` |
| Input gradient | `target.addProduct(dY,W[i].getTranspose(),scaleTargets,1)`, no batch averaging | `src/layer.cu:647-650`; A:`src/nvmatrix/nvmatrix.cu:530-532,221-236` |
| Weight gradient | `target=beta*target+alpha*(X^T*dY)`, completed product scaled once | `src/layer.cu:659-671`; A:`src/nvmatrix/nvmatrix.cu:189-205,221-236` |
| Bias gradient | `getGrad().addSum(dY,0,beta,alpha)`; axis 0 is cases in native F view | `src/layer.cu:653-656`; A:`src/nvmatrix/nvmatrix.cu:700-724,896-905` |
| Zero target coefficient | Never reads old NaNs when beta=0, including negative zero beta | A:`src/nvmatrix/nvmatrix.cu:222-224,204-205,896-900`; ANX-4 contract |
| Zero product coefficient | GEMM alpha=0 skips operands and returns +0 product. Bias sum still evaluates source, so zero alpha times NaN is NaN | ANX-4 BLAS contract; A:`src/nvmatrix/nvmatrix.cu:896-900`; A:`include/nvmatrix/nvmatrix_operators.cuh:281-290` |
| Signed upstream | Kernels accept the supplied gradient; no sign flip or implicit averaging. Original logreg supplies negative loss derivative | `src/layer_kernels.cu:464`; `src/layer.cu:649,666,671` |
| Validation/ownership | Explicit native shapes/layouts, offsets and F-strided slices; target backing aliases rejected before writes; all fan-in inputs validated before the first product | CPU API boundary, preserving ANX-4 layout rules |

All FC matrix signatures were checked individually, rather than assuming Candidate
A is the linked library. `addProduct(a,b,float,float)` matches A header :210;
`rightMult(b,float,target)` matches :212; `addVector(vec)` matches :362;
`getTranspose()` matches :412; `addSum(a,int,float,float)` matches :373;
`sum(int,target)` and `sum(int)` exist at :374–375 but are not directly called
by FC. Header path: A:`include/nvmatrix/nvmatrix.cuh`. Implementations and
coefficient order are cited in the table and CUDACONV2.md §8. **No FC signature
mismatch was found.** ANX-8's missing minDiv mismatch remains relevant elsewhere.

There is a newly documented **semantic caveat**: A's `_aggregate` checks physical
width>1 (`A:src/nvmatrix/nvmatrix.cu:723`); at width=1 it calls `copy(target)`
(`:801-803`), bypassing alpha/beta and the +0 reduction seed. Thus its N=1 FC
bias helper does not implement the advertised scaling contract. The requested
port contract is honored for all N, including 1. It cannot be established whether
the missing September library retained or fixed this candidate edge case.

A's ordinary bias sums use parallel CUDA reduction trees
(`A:include/nvmatrix/nvmatrix_kernels.cuh:272-332`), and its GEMMs use cuBLAS.
The port intentionally fixes ascending CPU reductions; it proves scalar/SIMD
bit parity, not historical CUDA bitwise parity. Keeping alpha/beta inside the
kernels preserves the original caller's algebra and scaling placement without
promising unavailable library-internal rounding equivalence.

## Coefficients ANX-11 must supply

Let U be the parameter's pre-write update count and N=`getNumCases(dY)`.
N is actual cases when weightUpdateFreq=1, otherwise frequency×minibatchSize
(`src/layer.cu:337-340`). Calculate these float32 coefficients with the original
operations, including **eps/N division**, not eps times a separately rounded
reciprocal. Do not scale terms inside the reduction.

| Destination / mode | alpha | beta | Source |
| --- | --- | --- | --- |
| Unshared weight increment, training, useGrad=false | `getEps(progress)/N` | U=0: `momW`; U>0: 1 | `src/convnet.cu:444`; `src/layer.cu:668-671` |
| Weight gradient, training, useGrad=true | `1.0f/N` | U=0: 0; U>0: 1 | `src/layer.cu:663-666` |
| Either weight destination, PASS_GC | 1 | U=0: 0; U>0: 1 | `src/layer.cu:664-670` |
| Bias gradient, training | `1.0f/N` | U=0: 0; U>0: 1 | `src/layer.cu:653-656` |
| Bias gradient, PASS_GC | 1 | U=0: 0; U>0: 1 | same |

Biases always use a separate gradient buffer (`src/layer.cu:565`); epsB/momB are
applied later by Weights::update (`src/weights.cu:151-152`), not in this bias
reduction. With superEps>0 the constructor forces useGrad (`src/layer.cu:544`);
sharing also affects that path and is unsupported here. Update counts advance
only after successful gradient calls (`src/layer.cu:581-590`). Learning-rate
base>0 guards also belong to the caller. Weight decay/wball/schedules and applying
increments remain ANX-11 work (`src/weights.cu:146-198`, `src/layer.cu:677`).
These APIs implement only the ordered product/reduction and final scaling.

## Five FC layers and separate neurons

All rows have two independently weighted inputs, both of the indicated width.
Neuron insertion and consumer rewriting are `layer.py:627-645`; FC itself never
calls a neuron (`src/layer.cu:640-644`).

| Layer / GPU | Effective input order | K → O per input | initB | Following operation | Config citation |
| --- | --- | --- | --- | --- | --- |
| fc2048a / 0 | pool3a_neuron, pool3b_neuron | 4608 → 2048 | 1 | fc2048a_neuron (relu) → hs1a | `layers/layers-145.cfg:238-259` |
| fc2048b / 1 | pool3a_neuron, pool3b_neuron | 4608 → 2048 | 1 | fc2048b_neuron (relu) → hs1b | `:247-264` |
| fc2048ba / 0 | hs1a, hs1b | 2048 → 2048 | 1 | fc2048ba_neuron (relu) → hs2a | `:266-287` |
| fc2048bb / 1 | **hs1b, hs1a** | 2048 → 2048 | 1 | fc2048bb_neuron (relu) → hs2b | `:275-292` |
| fc1000 / 0 | hs2a, hs2b | 2048 → 1000 | 0 (default) | probs (softmax), no inserted neuron | `:294-303`; `layer.py:833` |

## Fan-out and consumer order

The original chooses beta from the producer's **per-device** received count
(`src/layer.cu:287-295,358-367`), reset each pass (`:329-334`). There is no
fixed global consumer-arrival order: device insertion is randomized (`:370-377`).
For these four fan-outs each GPU has one consumer, so both independently use
beta=0. Once all contributions arrive, an existing local result receives the
remote result with beta=1 (`:178,202-212,228-251`), regardless of arrival order.
A CPU single-buffer consumer sequence uses 0 then 1; both orders are tested too.

| Producer / GPU | Local FC / input slot | Remote FC / input slot | Producer summation |
| --- | --- | --- | --- |
| pool3a_neuron / 0 | fc2048a / 0 | fc2048b / 0 | a contribution + b contribution |
| pool3b_neuron / 1 | fc2048b / 1 | fc2048a / 1 | b contribution + a contribution |
| hs1a / 0 | fc2048ba / 0 | fc2048bb / 1 | ba contribution + bb contribution |
| hs1b / 1 | fc2048bb / 0 | fc2048ba / 1 | bb contribution + ba contribution |

Connectivity: `layers/layers-145.cfg:240,245,249,254,268,273,277,282`,
with pool3 neuron detachment from `layer.py:630-645`. hs2a/b each have only
fc1000 as consumer; hs2b's remote-only producer first copies with beta=0.
Existing ANX-4 already proves the remote-only merge; the FC suite additionally
asserts this graph edge. Actual-size N=3 FC backward contributions for every
pool3/hs1 producer are checked against sampled scalar dot products, then merged
in local-plus-remote order for both arrival sequences from NaN-filled buffers.
The independent tiny fixture separately proves actual FC overwrite-then-add
calls with two different W/dY pairs; coefficients 0 then 1 reproduce its oracle.

## Independent fixtures and numerical checks

`tests/fixtures/make_fc_fixture.py` is standard-library Python, independent of C#.
Its float64 oracle evaluates full tiny dense products, gradients and fan-out.
Finite operands and coefficients are binary fractions, and finite fixture sums
fit exactly in float32; finite fixture equality therefore uses zero tolerance.
NaN requires IsNaN; infinities require matching sign. Strict JSON strings encode
NaN, ±infinity and -0. Dedicated literal contracts compare signed-zero raw bits.

Nine fixtures, each on both engines:

- One input, width 3, O=5, N=1.
- Two inputs with widths 3/5, 9/11, 5/3, 3/5; O=9/11/3/9; N=2/3/4/5.
- hs-order-ab and hs-order-ba, widths 3/3, N=3, O=5: swap input values while
  keeping weight slots fixed, checking independently different outputs.
- Signed zero, negative, NaN, ±infinity in N=5 input/upstream; ordinary coefficients.
- The same special values with alpha=0, beta=0.5: GEMM discards NaN products,
  while bias reduction retains source NaNs.

Each fixture checks full forward, explicit ordered input calls, bias once,
input gradients overwrite/add, target NaN discard/retention at scales 0/1,
weight/bias gradients with alpha=0.375 and beta=-0.25 (special alpha=0 case
above), and source preservation. Independent tiny fan-out vectors are also in
fc-cases.json. Strided F views at N=17 exercise offsets, SIMD and tails with
untouched sentinels on all outputs. Bad shapes/layouts/indices/counts and
exact/partial backing aliases are rejected. A bad later fan-in input cannot
partially write the output. Negative FC output proves no implicit ReLU.
The literal 2^24 cancellation witness proves bias placement and float32 rounding.
Scaled-zero kernels require -0 bits; zero-alpha GEMM requires +0 even from NaN.

Numerical seeds 92001/92002/92003, two inputs K=3/7, O=5, N=2/3/4:
all input, weight and bias coordinates are perturbed in mutable **float64** copies.
The independently evaluated loss is dot(dY, sum_i X_i W_i + b).
Central differences use epsilon=1e-5; float32 gradients from both engines must
satisfy `abs(error) <= 2e-6*(1+abs(numeric))`. All **510 comparisons** pass;
worst absolute error **3.81698593e-8**. No float32 perturbation rounding affects
the double finite-difference oracle.

## Optimized path, determinism and workspace

Both engines use +0-seeded float32 sums. Forward reduces k ascending, inserts
bias after input 0, then completes remaining inputs in declared order. dX reduces
output features ascending; dW/db reduce cases ascending. Alpha and beta apply
after a completed reduction, with separate multiply/add and no FMA.

Optimized forward parallelizes output features and vectorizes independent cases;
tails use the same scalar arithmetic. dX owns disjoint input-feature SIMD blocks
and vectorizes input features, broadcasting each upstream value. dW packs X
once into a case-major transpose and vectorizes independent input features;
workers own output columns. Bias gradients parallelize independent outputs.
There are no shared accumulators, atomics or horizontal SIMD reductions. Layer
instances retain no mutable workspace. The implementations share geometry and
validation, not arithmetic loops.

Workspace float payload (excluding preallocated operands/targets and small task
objects): forward/dX/db **0 bytes**; dW **4*K*N bytes** for one packed input.
This bound is input-size proportional, independent of worker count and output
width, with no full weight-sized scratch or per-worker matrix replication. Inputs
are processed sequentially. At N=128 peak is **2.250 MiB** per fc2048a/b and
**1.000 MiB** per fc2048ba/bb/fc1000. The caller retains all outputs; allocation
is included in benchmark timings. Dead arrays and GC heap reservation can exceed
live workspace.

Seeds 91000–91035 cover one, two and three inputs, widths 1–23, outputs 1–21,
N=1/2/3/4/5/8/9/17/128 on this machine; includes zero signs, NaN/inf,
alpha=0/-0/0.3/-0.75 and beta=0/-0/1/0.5/-1. All forward, explicit-input,
dX/dW/db bits match the scalar reference at workers **1, 2, 24, repeated 24**.
No tolerance, including NaN representations. This also proves thread-count
independence at SIMD widths and tails.

All five layers are constructed through parsed definitions and inferred shapes;
exact input order, initB, shape and detached-neuron edges are asserted. Default
N=3 executes optimized forward/all gradients and checks sampled scalar dots
across cases/features plus finite checks over entire tensors. Full mode adds
**every-element** scalar comparisons at N=3 and N=9 for all five layers,
workers 1/2/all/repeated-all, including dX addition and nontrivial dW/db scaling.
The full mode passed once in Release: **590 tests + spec checks in 41.008 s**
(41.441 s process, no build). Default initial Debug: **564 tests in 10.746 s**.

## Specification changes

LEDGER.md now explicitly places bias between products, lists caller coefficients
for gradient versus increment buffers, and distinguishes per-device fan-out from
a single-buffer policy. CUDACONV2.md §8 adds the individual NVMatrix signature
audit, CUDA reduction association and the N=1 scaling caveat. Neither prior
ANX-4 layouts nor its fan-out rules required correction. No ANX-2..8 behavior
was changed. Historical linked-library rounding is an evidence limitation,
not claimed historical bit parity.


## Batch-128 benchmark

Release .NET SDK 10.0.403, Intel Core Ultra 9 285HX, 24 logical processors /
workers, 8 float SIMD lanes. Two untimed warmups then three samples per operation;
medians below include allocation, packing, scheduling and final writes. Each
input/weight-gradient measurement includes both inputs. No benchmark/test/build
ran concurrently. Raw arrays and shapes are in ignored
`.scratch/anx9/benchmark.json`. Times are informational layer microbenchmarks;
JIT/PGO, mixed CPU cores, scheduling, thermal state and GC affect them. No
sustained trainer throughput is inferred.

Each cell is **milliseconds / achieved GFLOP/s**. Dominant product FLOPs for
forward, dX and dW are `2*N*O*sum(K_i)` each; bias uses `N*O` additions.
Bias broadcast, alpha/beta scaling and target addition are excluded from these
conventional counts. Workspace includes one input transpose, never both at once.

| Layer | Forward | Input gradient | Weight gradient | Bias gradient |
| --- | ---: | ---: | ---: | ---: |
| fc2048a | 38.646 / 125.03 | 76.704 / 62.99 | 35.172 / 137.38 | 0.480 / 0.55 |
| fc2048b | 38.345 / 126.01 | 75.282 / 64.18 | 37.928 / 127.40 | 0.205 / 1.28 |
| fc2048ba | 17.493 / 122.76 | 54.658 / 39.29 | 18.790 / 114.29 | 0.053 / 4.95 |
| fc2048bb | 17.018 / 126.19 | 53.852 / 39.88 | 20.189 / 106.37 | 0.043 / 6.14 |
| fc1000 | 10.687 / 98.12 | 21.007 / 49.91 | 11.300 / 92.80 | 0.030 / 4.31 |

Reproduction (CLI state, packages, HTTP cache and temporary/runtime state should
be redirected to repository scratch; set NUGET_HTTP_CACHE_PATH under `.scratch/`):

```sh
dotnet build --disable-build-servers
dotnet build -c Release --disable-build-servers
dotnet run --project tests/AlexNetSharp.Tests
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --full-scalar-parity
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-fc --fc-output .scratch/fc-benchmark.json
python3 tests/fixtures/make_fc_fixture.py
```

Benchmark optional flags: `--fc-threads <positive integer>`,
`--fc-repeats <positive integer>` (default 3), `--fc-output <path>`.


## Final gates and integrity

Final requested builds used the installed SDK with build servers disabled.
Both Debug and Release report **0 warnings / 0 errors**. Debug MSBuild 1.66 s
(process 1.787 s); Release MSBuild 1.59 s (process 1.722 s). All CLI, packages,
NuGet HTTP cache, XDG runtime/cache/config/data and temporary state were redirected
under ignored `.scratch/anx9/`. Exact gate logs and distinct process IDs are
recorded there; the combined summary is `.scratch/anx9/gates.json`.

| Fresh process | Exit | Tests + spec checks | Suite seconds | Process seconds, build/startup included |
| --- | --- | --- | ---: | ---: |
| default-debug-1 | 0 | 564 | 10.711 | 11.717 |
| default-debug-2 | 0 | 564 | 10.866 | 11.859 |
| default-debug-3 | 0 | 564 | 10.840 | 11.834 |
| full-release | 0 | 590 | 40.921 | 41.911 |

All default Debug runs finish below 20 s, including build/startup. Final full
Release includes every new complete scalar check and all pre-existing full-mode
assertions. The additional zero-sign tests and negative-zero beta coverage passed
in these final runs. No existing test or assertion was removed or weakened.
Both modes also check 569 parser fields, 1,470 shape fields and 455 original
citations. There are 78 new default FC groups and 83 full-mode FC groups.

Final integrity audit (`.scratch/anx9/integrity.json`):

- All **246 original files** retain identical relative names, sizes, nanosecond
  mtimes and SHA-256, with no additions/removals. No original source changed.
- Candidate archive hash matches the specification; all five extracted matrix
  files equal their archive bytes, and archive/files have write permissions
  removed. No candidate source, archive or environment path is in deliverables.
- Fixture regeneration is byte-identical; SHA-256
  `73f0f7876ab8bc91a8b7e6812c067045311fd22791d61684c4fa72fbbb67e51a`.
- Previous tracked implementation/tests/fixtures and machine-readable specs are
  unchanged except the test entry-point registration. Other modified prior files
  are only README and the two documented narrative specs.
- All **13 deliverables** use LF and final newlines. Private-path, hostname and
  IP scans pass; `git diff --check` passes; nothing is staged. No commit, push,
  stash or process-killing command was used.
- Authored files, downloads, builds, caches and temporary/runtime artifacts stayed
  in this repository. No file outside the repository was changed by task work.
  The initial runtime-directory attempt failed read-only; redirecting XDG runtime
  state allowed the work to complete within the permitted tree.

Nothing is unfinished. Historical GPU bit parity is outside the available
linked-library/execution evidence; all requested CPU implementation and proof
work is complete, with the candidate's N=1 semantic caveat explicitly recorded.
