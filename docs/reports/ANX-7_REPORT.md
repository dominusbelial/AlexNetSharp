# ANX-7 — Pooling and ReLU

Started approximately 20:36 UTC on 2026-11-25; this report was created first
at 20:36:11 UTC. Base `fc34e9a`. Changes remain in the working tree, without
commits or pushes. Finished at approximately 20:50 UTC (~14 min elapsed).
All six brief items and all gates are complete.

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. Scalar reference | done | `ScalarPool` and `ScalarRelu`; independent fixtures, numerical gradients and contracts passed. | ~20:36–20:42 UTC (6 min) |
| 2. Optimised path | done | Independent managed SIMD/parallel implementations; raw-bit parity at 1/2/24/repeated-24 passed. | ~20:37–20:42 UTC (5 min) |
| 3. Independent hand fixtures | done | 21 pooling cases and ReLU special values passed on both engines. | ~20:38–20:42 UTC (4 min) |
| 4. Numerical gradients | done | 1,760 float64 central differences passed; worst absolute error 1.7687853e-8. | ~20:39–20:42 UTC (3 min) |
| 5. Real geometry and timings | done | All six pools at N=3, every last-window tap, all 14 detached ReLUs; batch-128 timings completed. | ~20:40–20:44 UTC (4 min) |
| 6. Report and gates | done | Both builds clean; default Debug 3× and full Release parity passed; original/candidate/fixture/hygiene integrity checks passed. | ~20:36–20:50 UTC (14 min, report throughout; final gates/audit ~20:44–20:50) |

Intervals overlap because source review, implementation and verification were interleaved.
An item is marked done only after its corresponding executable proof passed.

## Provenance and scope

Read `DESIGN.md`, the pool/neuron ledger rows, `CUDACONV2.md`, and ANX-3..6
reports. Original paths were resolved through ignored `local.env`; citations below
are upstream-relative. The original tree was opened read-only and baselined
at 20:36:31 UTC (274 files; relative paths, size, nanosecond mtime and SHA-256).

Candidate A was fetched into ignored `.scratch/anx7/`, verified against ANX-1's
archive SHA-256 `b55bd30ceb2c69c9540ae7e7c2f6f6cd075551071c351e57e3b7a5d7bb10f819`,
and made read-only. **A:** citations are relative to its `cuda-convnet/trunk/`.
Only the two pool kernel files and the operator header were extracted; no candidate
source is included in deliverables. This proves reconstructed semantics and CPU
scalar/optimized parity; the missing September CUDA snapshot was not executed.

No ANX-2..6 implementation or assertion is changed. The existing test entry point
registers this suite and a separate informational benchmark. No graph executor,
normalization, FC math, dropout, optimizer, data loading, web or GPU code is added.
`maxabs`, `rand` and unknown pool kinds throw an error naming the layer and saying
“unsupported in this port.” Average pooling was cheap enough to include on both paths.

## Semantics verified against the sources

| Behavior | Port contract | Original / candidate evidence |
| --- | --- | --- |
| Pool construction and calls | Pool does its operation alone. Max forward receives input, geometry and MaxPooler; backward receives original input, upstream and saved pool output. Average needs no saved matrices. | `src/layer.cu:1273-1295,1303-1333`; A:`include/cudaconv2/conv_util.cuh:327-342`; A:`src/cudaconv2/conv_util.cu:1869-1895,1968-1987` |
| Shape | Default start 0; width `1+ceil((image-start-sizeX)/stride)` unless positive explicit outputsX overrides it. | `layer.py:1139-1141,1163-1165` |
| Layout | Pool `(channel*y*x,case)` C-order. ReLU accepts the explicit ANX-4 native transposed `(case,feature)` F-order view and canonical view. No implicit conversion. | A:`include/cudaconv2/conv_util.cuh:120-122,145-150`; `src/layer.cu:483-484,150-155` |
| Edge handling | Visit y then x in `[max(0,start+o*stride), min(image,start+o*stride+sizeX))`. No padded zero values. | A:`include/cudaconv2/conv_util.cuh:161-175` |
| Maximum | Seed is finite `-2e38f`, then numeric `fmaxf`; forward always overwrites. | A:`include/cudaconv2/conv_util.cuh:84-94,157,174,186` |
| Max ties / overlap | Every input equal to a window's output receives its full upstream gradient. Sum overlapping outputs in ascending output y,x order. No argmax index or tie division. | `src/layer.cu:1333`; A:`src/cudaconv2/conv_util.cu:1512-1515,1536-1559` |
| Max mask multiplication | Evaluate `float(input==max)*upstream` even for false matches. A NaN upstream therefore contaminates every input in that window. | A:`src/cudaconv2/conv_util.cu:1555-1559` |
| Average | Sum real pixels / clipped count, not sizeX squared. Backward adds upstream multiplied by float32 reciprocal of each clipped area. | `src/layer.cu:1307,1311`; A:`include/cudaconv2/conv_util.cuh:71-80,165-186`; A:`src/cudaconv2/conv_util.cu:1430-1446` |
| Pool accumulation | Activity gradients have no batch averaging. scaleTargets 0 overwrites without reading the target; nonzero scales multiply the old target once after the sum. Output scale is fixed at 1 by the layer. | `src/layer.cu:1311,1333`; A:`src/cudaconv2/conv_util.cu:1454-1470,1566-1582,1903-1916` |
| ReLU identity | `relu` creates ReluNeuron; forward uses `x<0 ? +0 : x`, retaining negative zero and NaN. | `src/neuron.cu:36-37`; `include/neuron.cuh:146-164` |
| ReLU mask | Backward uses **saved output**, not original input: upstream times `(output>0)`. Zero, negative zero and NaN outputs have false masks. Multiplication preserves NaN upstream even with false masks. | `layer.py:1400`; `include/neuron.cuh:152-170` |
| ReLU accumulation | `scaleTargets>0` means add derived gradient to old target; it is a boolean, not an arbitrary multiplier. Nonpositive values overwrite. Fan-out chooses 0 for first contribution and 1 subsequently. | `src/layer.cu:490-505,287-295`; `include/neuron.cuh:41-42,94-101,156-157` |
| ReLU memory sharing | Elementwise forward/backward can run in place. The intended sharing guard requires a disposable equal-size sole-consumer input, but fresh 145 parsing never enters it (dictionary-key iteration). Max-pool saved output must survive post-pool ReLU. | `include/neuron.cuh:67-68`; `layer.py:338-354,651,831,977,1145-1147`; `src/layer.cu:150,508-509` |

`PoolGeometry`, `ScalarPool.FromLayer` and `OptimizedPool.FromLayer` use existing
parsed parameters and ANX-3 shapes. The positive explicit-output override is retained,
including empty windows. Invalid/nonpositive runtime dimensions are rejected; shape
inference's historical pathological negative widths are not executable tensors.
Size is 1..image and stride 1..size. CUDA's channel launch divisibility restrictions
are unnecessary for CPU execution, allowing tiny independent fixtures.

Pools validate all matrices before writing, require canonical C-order, preserve row
strides/offsets and reject shared target backing storage. ReLU supports both layouts,
mixed layouts and strided views; exact same-index aliases work and differently mapped
shared storage is rejected. API operands and outputs are preallocated, as in ANX-5/6.
The caller retains the pool output separately and explicitly requests transposed views
for native neuron execution; real-geometry tests exercise that sequence.

### NaN, infinity and signed zero

The pool's accumulator starts numeric, so NaN input never makes a max output NaN.
An all-NaN, empty, all-negative-infinity or entirely-below-seed window returns
`-2e38f`. With finite upstream, a NaN input compares unequal and gets zero. Even
non-winning pixels get NaN from NaN upstream because the mask multiplies rather than
branches. Mixed zero signs choose +0; all -0 retains -0. Both signs compare equal
in backward, so every zero tie receives the full gradient.

NVIDIA's [CUDA math API, fmaxf](https://docs.nvidia.com/cuda/archive/8.0/cuda-math-api/group__CUDA__MATH__SINGLE.html)
documents numeric selection when one argument is NaN. [PTX max](https://docs.nvidia.com/cuda/pdf/ptx_isa_8.7.pdf)
documents +0 greater than -0. Those contracts explain the candidate's library call;
they are not historical execution evidence. ReLU's ternary retains -0 and NaN;
its output comparison is false for both, so finite upstream gives signed zero
according to multiplication. NaN upstream stays NaN. An overwritten NaN target is
never read; accumulated NaN targets remain NaN on both operations.

### Every 145 neuron= placement

Each row is a separate inserted `<layer>_neuron`, applied **after** its source
operation. All downstream inputs are rewritten to that neuron. This is parser
behavior, not an activation performed inside pool/conv/FC
(`layer.py:272-275,295-298,627-645`; `src/layer.cu:508-509`).

| Layer | Neuron | Where applied | Citation in `layers/layers-145.cfg` |
| --- | --- | --- | --- |
| pool1a | relu | conv1a → cmrnorm rnorm1a → pool1a → pool1a_neuron → conv2a | `:35-54,67` |
| pool1b | relu | conv1b → cmrnorm rnorm1b → pool1b → pool1b_neuron → conv2b | `:41-63,82` |
| conv2a | relu | After conv2a, before rnorm2a | `:65-77,95-97` |
| conv2b | relu | After conv2b, before rnorm2b | `:80-92,101-103` |
| conv3a | relu | After conv3a, before conv4a | `:135-146,165` |
| conv3b | relu | After conv3b, before conv4b | `:149-160,179` |
| conv4a | relu | After conv4a, before conv5a | `:163-171,194` |
| conv4b | relu | After conv4b, before conv5b | `:177-185,208` |
| pool3a | relu | conv5a → pool3a → pool3a_neuron → both fc2048a/b | `:220-227,240,249` |
| pool3b | relu | conv5b → pool3b → pool3b_neuron → both fc2048a/b | `:229-236,240,249` |
| fc2048a | relu | After fc2048a, before hs1a | `:238-244,258` |
| fc2048b | relu | After fc2048b, before hs1b | `:247-253,263` |
| fc2048ba | relu | After fc2048ba, before hs2a | `:266-272,286` |
| fc2048bb | relu | After fc2048bb, before hs2b | `:275-281,291` |

Pool2a/b have no neuron key. Conv1a/b and conv5a/b have no neuron key either;
their activation is deferred to pool1 and pool3 respectively. The original parser
marks pool usesActs/usesInputs true (`layer.py:1146-1147` even compares literal
strings for avg); thus post-max-pool forward must preserve its pre-ReLU output.
Conv/FC set usesActs false (`layer.py:831,977`), which would make their
sole-consumer detached ReLUs eligible under the intended sharing condition.
However, the actual optimizer at `layer.py:345` iterates `for ll in layers`,
where `layers` is still the dictionary passed at `:298`. Its `'inputs' in ll`
tests string **keys**; none of the 49 actual names contains that substring.
The sum is therefore zero, never one. Both targets stay at their initialized
-1 (`layer.py:154-156`), for forward and backward. **A fresh 145 parse does not
select in-place ReLU automatically.** An isolated execution of the actual
`:338-354` method against all 49 fixture names/shapes, even with every metadata
flag made permissive, confirmed all targets remain -1 (14 detached neurons;
ignored `.scratch/anx7/sharing-probe.json`). This is a source-code quirk, not
an inferred fix; no existing parser behavior was changed. The C++ primitive and
this port still support explicit in-place execution, proven by both layout tests.

## Independent fixtures and numerical proof

`tests/fixtures/make_pool_relu_fixture.py` uses only Python's standard library.
It constructs membership sets by enumerating the entire image and testing window
membership, then scatters gradients output-first. It never invokes C# and does
not use the input-centric undo bounds. Explicit binary32 rounding makes expected
averages reproducible. Strings encode NaN, infinities and -0 in strict JSON.

21 pooling cases in `pool-relu-cases.json`:

- 5×5, 6×6, 7×7, size 3 / stride 2 / start 0: widths 2,3,3. Only 6×6 has a
  partial default final window; 5/7 tile fully. Explicit width 2 on 6×6 omits
  that edge. Start 1 on 5/7 adds partial edges; start -1 covers top/left clipping.
- Two channels / three cases with distinct values; negative-only windows.
- Ties within one window and shared between overlapping windows, mixed zero
  signs, all-negative-zero, mixed NaN, all-NaN, values below the seed/negative
  infinity, NaN upstream on non-winners, and an explicitly empty final window.
- Average at all three requested sizes with C=2/N=3, plus negative start clipping.

Every case runs both engines, forward from NaN targets, backward overwrite from
NaN, addition to finite targets, and retention of NaN targets at scale 1. Average
also runs backward with null input/output caches. Fixture comparisons require exact
bits for numeric values including signed zero; expected NaNs require IsNaN rather
than a payload convention. Separate engine parity compares **all** raw bits,
including actual NaN representations.

ReLU fixture: negative, -0, +0, positive, NaN and both infinities, with signed and
NaN upstream. Repeated values cover SIMD lanes and tails at N=17 in C and F
layouts, both engines, overwrite/add, NaN target retention, in-place operations,
and mutated saved outputs proving the mask source. Contracts also prove the
boolean positive-scale behavior at 0.5 and overwrite at -1, named unsupported
pools, dimensions/layout/alias failures, and sentinel preservation in strided views.

Max and ReLU finite differences use fixed seeds 7701–7703, sizes 5/6/7, C=2/N=2,
and a clipped negative start on 7. Distinct shuffled exactly representable inputs
are separated by 1 and offset by 0.25, ensuring no ties or zeros. Double-precision
copies are perturbed by ±1e-5. The independent double loss enumerates input-set
membership and computes dot(max, upstream); ReLU uses its double scalar function.
Ties and zero are excluded because they are nondifferentiable, and central
differences would not test the historical all-ties routing / zero-mask choice.
Acceptance is `abs(error) <= 2e-6*(1+abs(numeric))`, allowing float32 activity
arithmetic; all 1,760 comparisons passed, worst error **1.7687853e-8**.

## Optimized parity, determinism and real geometry

OptimizedPool assigns independent output rows (forward) or input rows (backward)
to workers and vectorizes contiguous cases. Each lane visits windows/taps in the
scalar order. SIMD numeric max explicitly ignores NaNs and handles both zero
signs. Average uses the same float32 add/divide or reciprocal/multiply sequence.
OptimizedRelu vectorizes each contiguous dimension for either layout, with a
scalar fallback for mixed strides. No atomics, horizontal reductions, FMA,
native dependency or full tensor workspace is used. Parallel scheduling/closures
allocate small runtime objects; no float scratch arrays are allocated by the ops.

Seeds 77100–77135 cover image widths 3..9, windows 1..3, stride 1..window,
start -1/0/1, C=1..3, and N=1/3/8/17/33/128. Both max and average, targets
scales 0/1/0.5, NaNs, infinities, ties, signed zeros and SIMD tails are checked.
Each workload uses workers 1,2,24,24 and compares every output bit to the scalar
oracle. ReLU also checks fixed operands across these worker counts. The tolerance
is **zero**, including NaNs. The finite-target fixture tests complement the
random parity's alternating finite/NaN targets, so scale-1 arithmetic is exercised independently.

All six 145 pools run N=3 (all channels/cases) with seeded inputs and upstreams,
scalar and optimized forward/backward. The last window is proven by making each
of its nine taps the unique maximum in turn, applying upstream 7 only to that
output and checking exact input-gradient membership. A second witness makes every
outside pixel larger than every inside pixel, proving the forward window excludes
all outsiders; all tied inside pixels then prove the full backward membership.
Both spatial axes are tested.

| Layers | Channels | Width | Last y/x window, inclusive | Output values per case |
| --- | ---: | --- | --- | ---: |
| pool1a/b | 48 | 55 → 27 | 52..54 | 34,992 |
| pool2a/b | 128 | 27 → 13 | 24..26 | 21,632 |
| pool3a/b | 128 | 13 → 6 | 10..12 | 4,608 |

The default suite also verifies all 14 detached names and consumer rewrites,
no pool2 neuron, negative pool outputs retained until a separate ReLU, and the
post-pool ReLU backward followed by max undo. Native F-order transpose views
are used for this actual neuron call sequence. The opt-in full mode adds the
same real pool operands at 1/2/24/repeated-24 workers, retaining all ANX-5/6
full-size convolution proofs unchanged.

## Batch-128 informational timing

Release .NET SDK 10.0.403, Intel Core Ultra 9 285HX, 24 logical processors /
requested workers, AVX2 (8 float lanes). Three untimed warmups then five samples
per operation; medians below are milliseconds. Calls include scheduling and final
writes; operands/outputs are preallocated. Runs were sequential, with no concurrent
suite. Raw arrays are in ignored `.scratch/anx7/benchmark-final.json`.
These are isolated operation timings; tiered JIT/PGO, worker startup, thermals and
mixed CPU cores cause noise (especially the first pool1a operation). No sustained
training-throughput claim is inferred. The slower first row is retained.

| Layer | Scalar forward | Optimized forward | Scalar backward | Optimized backward | ReLU forward | ReLU backward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pool1a | 160.482 | 9.222 | 398.905 | 12.681 | 1.635 | 2.068 |
| pool1b | 161.634 | 1.741 | 401.156 | 3.966 | 0.448 | 0.498 |
| pool2a | 100.597 | 0.862 | 252.361 | 2.415 | — | — |
| pool2b | 99.822 | 0.910 | 251.425 | 3.142 | — | — |
| pool3a | 21.472 | 0.153 | 55.942 | 0.454 | 0.029 | 0.034 |
| pool3b | 21.360 | 0.149 | 55.970 | 0.483 | 0.034 | 0.033 |

ReLU timings use separate saved output/gradient buffers and occur only where the
145 config inserts them. Average is implemented and verified but not benchmarked
because the 145 network uses max exclusively.

## Specification changes

No contradictory pooling/ReLU arithmetic rule was found in `LEDGER.md` or `CUDACONV2.md`.
The fresh-parser sharing quirk above was previously undocumented; the ledger now
records it with the literal source lines.
Their pool summaries were expanded with the previously omitted finite seed,
NaN handling, mask multiplication and clipped-average denominator, with candidate
citations. No machine-checked ledger/manifest/shape fields changed. The original
parser's literal-string avg cache flags are recorded above, without changing the
existing parser. Those flags do not affect average kernel arithmetic.

## Verification gates

Both final builds passed with **0 warnings / 0 errors**: Debug MSBuild 1.53 s
(process 1.652 s), Release MSBuild 1.59 s (process 1.701 s). Each suite invocation
below used the installed SDK and a fresh OS process. All runs exited zero.

| Fresh process | Tests plus specification checks | Suite seconds | Process seconds including build/startup |
| --- | ---: | ---: | ---: |
| default-debug-1 | 380 | 5.672 | 6.635 |
| default-debug-2 | 380 | 5.651 | 6.584 |
| default-debug-3 | 380 | 5.682 | 6.662 |
| full-release | 401 | 19.197 | 20.152 |

Default remains well below 20 seconds, including build/startup. Its 98 new groups
are 42 pooling-fixture engine cases, four ReLU layout/engine cases, six numerical
groups, 36 randomized geometries, three contract groups and seven real-geometry /
placement groups. Full mode retains ANX-5/6's 303 groups and these 98, with
additional real pool thread-count assertions inside the real-geometry groups.
Each run also verifies 35 sections, 10 types, 15 weighted layers, 569 parser
parameters, 1,470 shape fields and 455 original citations. No existing assertion
was deleted or weakened. Gate logs and distinct process IDs are in ignored
`.scratch/anx7/gates.json` and matching logs.

The final repetition followed stronger boundary exclusion and mixed-stride ReLU
checks; both repetitions passed. Initial integrated Debug proof was 5.998 s.
The final source audit additionally executed the isolated original sharing method,
with its findings recorded above; it required no arithmetic implementation change.

Reproduction (from repository root; CLI state, packages, NuGet HTTP cache and
scratch/temp outputs stay inside the repository):

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
"$HOME/.dotnet/dotnet" build -c Release --disable-build-servers
for run in 1 2 3; do
    "$HOME/.dotnet/dotnet" run --project tests/AlexNetSharp.Tests || exit 1
done
"$HOME/.dotnet/dotnet" run -c Release --project tests/AlexNetSharp.Tests -- --full-scalar-parity
"$HOME/.dotnet/dotnet" run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-pool --pool-output .scratch/pool-benchmark.json
python3 tests/fixtures/make_pool_relu_fixture.py
```

Final integrity proof (ignored `.scratch/anx7/integrity.json`):

- All **274 original files** retain their relative paths, sizes, nanosecond mtimes
  and SHA-256 digests, with no additions/removals.
- Candidate A archive digest matches the brief; all extracted bytes match the
  archive, and both archive and extracted files have write permissions removed.
- Independent fixture regeneration is byte-identical, SHA-256
  `45ac5dcc296346a3e4018e42122b3ca2f3e68791deedfd20d6b3846ea1193442`.
- Every previously tracked source/test/fixture and machine-readable specification
  is byte-identical to HEAD except test entry-point registration. ANX-2..6 code
  and behavior remain intact. Documentation additions are described above.
- All **14 deliverable files** use LF with final newlines; no local home/original
  paths, hostnames, IPs, secrets, CUDA headers/sources or candidate archives are
  present. `git diff --check` passes and the index is unchanged.
- All authored files, caches, downloads, build outputs and temporary artifacts
  stayed in this repository. No file outside the repository was intentionally
  changed; no original-tree file changed. No commits, pushes, stashes or process
  killing commands were used.

Nothing is unfinished. Historical CUDA bitwise execution parity remains outside
the available evidence; CPU optimized/scalar parity is proven at zero tolerance.
