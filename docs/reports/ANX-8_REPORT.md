# ANX-8 — Response normalisation

Started 2026-11-25 at 23:25 UTC; report created first at 23:25:14 UTC.
Base `983cc38`. Changes remain in the working tree, without commits or pushes.
Finished at 23:37 UTC (~12 min). All six brief items and all gates passed.

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. Scalar reference | done | `ScalarNorm`: both forward/backward kernels, cache and accumulation contracts; 24 independent fixtures and numerical checks passed. | 23:25–23:31 UTC (~6 min) |
| 2. Optimised path | done | Independent `OptimizedNorm`, SIMD across cases and parallel rows; zero-tolerance raw-bit parity at 1/2/24/repeated-24 passed. | 23:26–23:31 UTC (~5 min) |
| 3. Independent fixtures | done | Standard-library float64 generator, 24 cases on both engines, NaN discard/retention and finite accumulation passed. | 23:28–23:31 UTC (~3 min) |
| 4. Numerical gradients | done | 300 coordinates per engine, 600 float32 comparisons with float64 central differences; double Jacobian also checked independently. | 23:29–23:31 UTC (~2 min) |
| 5. Real geometry and timings | done | All six 145 layers at N=3/default and N=17/full, all worker counts and accumulation pass; batch-128 scalar/optimized timing completed. | 23:30–23:36 UTC (~6 min) |
| 6. Report and spec corrections | done | Semantics/citations, tolerances, fixture list, timings and all narrative spec corrections documented; final integrity audit passes. | 23:25–23:37 UTC (~12 min, throughout) |
| Build/test gates | done | Debug/Release: 0 warnings/errors; fresh Debug 3×: 486 tests; full Release: 507 tests; original and deliverable hygiene checks pass. | 23:35–23:37 UTC (~2 min) |

Intervals overlap: source reading, implementation, fixtures and tests were interleaved.
Times include each item's initial successful proof; final repeat gates are charged
below to the report/gate work.

## Provenance and scope

Read `DESIGN.md`, the norm rows and full 145 table in `LEDGER.md`, `CUDACONV2.md`
§6–7, and ANX-4/6/7 reports before implementation. Original paths were resolved
through ignored `local.env` and opened read-only. A baseline records relative
names, size, nanosecond mtime and SHA-256 for 246 original files excluding `.git`.

Candidate A was fetched read-only into ignored `.scratch/cudaconv2-candidates/`.
Archive SHA-256 matches ANX-1:
`b55bd30ceb2c69c9540ae7e7c2f6f6cd075551071c351e57e3b7a5d7bb10f819`.
Only `conv_util.cu/.cuh` were extracted, byte-checked and made read-only.
**A:** citations mean paths relative to `cuda-convnet/trunk/`; unprefixed source
citations refer to the original tree. Candidate B's one kernel file was read at
revision `3238bf0367f63eb370e897b9e5714794cb67ddc2`, using the URL/revision in
`CUDACONV2.md`, and made read-only in scratch. **B:** means that revision's paths.
No candidate source is part of the deliverables.

No ANX-2..7 implementation, assertion, machine-readable ledger, manifest or golden
fixture changes. Only the test entry point registers the new suite/benchmark.
No net assembly, trainer, FC, dropout, softmax, optimiser, data loading, web or GPU.
Both fixed blocked channel groups and unblocked windows were cheap to implement.

## Semantics verified against the sources

Let `a` be the **already parser-divided** float32 scale, `b` the positive config
power, `d` the denominator cache, `z` the saved norm output and `g` upstream.
Forward computes `d = base + a*sum(x*x)`, then `z = x*MathF.Pow(d,-b)`.
Backward first computes `k = (-2*a)*b` and `t = (k*g)*z/d` independently at every
output. It then gathers `t` over every forward window containing this input:
`dx = x*sum(t) + g*MathF.Pow(d,-b)`. Parentheses/order here describe the actual
float32 operations. Target scaling happens once after this completed gradient.

| Behaviour | Port contract | Original / candidate evidence |
| --- | --- | --- |
| Parser scale | cmrnorm `rawScale/size`, rnorm `rawScale/(size*size)` in double; cast to float once when constructing a runtime layer | `layer.py:1179-1184`; `src/layer.cu:1564-1565` |
| Power/defaults | Config pow remains positive 0.75, exponent passed to pow is negative. cmrnorm minDiv default 1, configured 2. blocked defaults false for both, spatial ignores it | `layer.py:1184-1193`; `src/layer.cu:1560-1593` |
| Layer calls/cache | Both forward kernels fill `_denoms`; backward receives it, original input, saved output and upstream. Output gradient scale is 1 | `src/layer.cu:1568-1573,1588-1593` |
| Layout | Canonical C-order `(channels*height*width,cases)`; row order channel,y,x, case fastest; preallocated buffers; no implicit transpose | `src/layer.cu:1560,1583`; A:`src/cudaconv2/conv_util.cu:1041-1058,1629-1654,2801-2822` |
| Spatial forward | Within the same channel: each coordinate interval `[p-size/2,p-size/2+size)` clipped to the image. Visit y then x, starting sum at +0. Base is always 1 | A:`src/cudaconv2/conv_util.cu:758-804,842-889,947-1013,2064-2065` |
| Cross-map forward | At fixed pixel: channels `[c-size/2,c-size/2+size)`, clipped at both ends, ascending channel order, sum seed +0 | A:`src/cudaconv2/conv_util.cu:1039-1086,2742-2781,2892-2893` |
| minDiv provenance | A hardcodes base 1; original passes minDiv, B confirms `minDiv + a*sum`. Port uses configured minDiv only for cmrnorm | A:`src/cudaconv2/conv_util.cu:1084`; `src/layer.cu:1589`; B:`cudaconv3/src/conv_util.cu:1206,2890,2898-2899` |
| Blocked cross-map | Window starts at `(c/size)*size`, ends at min(C,start+size); reverse uses the same group, including a clipped final group | A:`src/cudaconv2/conv_util.cu:1068-1070,1142-1144` |
| Backward preliminaries | `t = ((-2*a)*b*g)*z/d`. Uses cached d, never re-sums input squares. CUDA overwrites z; this API preserves it with separate temporary storage | A:`src/cudaconv2/conv_util.cu:1593-1605,2226-2255,2799-2828` |
| Reverse windows | Interval `[p+size/2-size+1,p+size/2+1)` clipped to the channel/image bounds. For odd size 5 it agrees with forward; even sizes have different starts. Sum ascending channel or y then x | A:`src/cudaconv2/conv_util.cu:1142-1163,1643-1691,1769-1834` |
| Backward final/accumulation | `x*sum(t)+g*pow(d,-b)`; scaleTargets=0 skips old target, otherwise `scaleTargets*old+dx`. No batch averaging | `src/layer.cu:1573,1593,287-295`; A:`src/cudaconv2/conv_util.cu:1160-1175,1684-1707` |
| Power/division implementation | A explicitly uses `__powf` and `__fdividef`; port uses `MathF.Pow` and ordinary float32 division, separate multiply/add, no FMA | A:`src/cudaconv2/conv_util.cu:804,889,1013,1086,1163,1601,1690` |

The candidate's explicit intrinsics, rather than an assumed compiler fast-math
flag, are the evidence for its approximate pow/division behaviour. The port proves
CPU scalar/optimized bit parity, not execution parity with the missing September
CUDA snapshot. Cross-map minDiv is a reconstruction supported by original A/B
sources, not a claim that A has the original's nine-argument forward ABI.

`FromLayer` uses ANX-2 typed parameters and ANX-3 dimensions, verifies matching
name/type/input/output shapes, and performs no scale division a second time.
Runtime constructors permit channels and planes smaller than the window and omit
CUDA channel divisibility limits so the requested tiny fixtures are executable.
The config parser's historical size/divisibility restrictions remain intact.
Spatial MinDiv and Blocked arguments are ignored, matching the original layer.
Invalid geometry/kind/thread counts, incorrect dimensions/layout and shared target
backing storage are rejected before writes. Named unsupported-kind errors say
“unsupported in this port.” Canonical row-strided views and offsets work; exact or
partial shared backing targets are conservatively rejected. Inputs, saved output,
upstream and cache remain untouched, proven with raw-bit snapshots and sentinels.

## NaN, infinity, zero and cache behaviour

No operand is masked or skipped because its value is zero. NaN in an input makes
every denominator whose forward window includes it NaN, then backward can spread
it through reverse windows via the saved intermediate. NaN upstream similarly
propagates through t and every gathered window. The independent fixtures check the
entire arrays, including unaffected channels/pixels and cases. Infinity produces
infinite denominators: finite neighbours times inverse-power zero yield signed
zero; infinity times zero yields NaN. Float32 square overflow intentionally yields
an infinite denominator even when a float64 calculation would remain finite.
The explicit overflow contract checks finite 1e30 and infinite inputs separately.

All-zero input gives base denominator and signed-zero forward output. Backward
retains its direct scaled gradient. Forward overwrites NaN output/cache buffers;
backward scaleTargets 0 overwrites NaN targets without reading them, while 1 retains
NaN. Arbitrary nonzero scaleTargets (0.5 and -1) is tested against the scalar path.
An injected denominator of 16 with zero input yields exactly 0.125 upstream for
pow 0.75, proving backward consumes the supplied cache. Separately changing saved
output from 0 to 1 proves that cache is consumed rather than reconstructed too.

## Independent fixtures and numerical proof

`tests/fixtures/make_norm_fixture.py` uses only Python's standard library and
float64 arithmetic. It enumerates membership by inspecting all input coordinates,
then scatters each output's Jacobian contributions, rather than deriving the
production input-centric reverse bounds. It never imports or invokes the C# code.
Nonfinite values and negative zero are strings in strict JSON. Fixture generation
is deterministic; final regeneration/hash evidence is recorded below.

24 cases in `norm-cases.json`, each run against both engines:

- Unblocked channel counts 3, 5 and 7 with size 5, clipping both ends, N=3;
  minDiv 1 versus 2; multiple spatial pixels; even size 4; blocked size 3 with
  seven channels, exercising the last partial group.
- Spatial 1×1, 2×2 and 6×6, C=2/N=3, size 5: every edge and corner; even size 4
  on 6×6; spatial minDiv=99/blocked=true proves both ignored settings.
- Both norms: all signed zeros, large signed finite values, NaN, +infinity,
  -infinity and NaN upstream in one case only. Ordinary cases contain negative
  inputs and zeros, representing raw pre-ReLU conv1 outputs.
- Every case: forward/cache from NaN, backward overwrite from NaN, accumulation
  into finite 3, accumulation into NaN, and source preservation.

Finite fixture comparisons retain the JSON oracle's double precision and use
`abs(error) <= 4e-6*(1+abs(expected))`; NaNs require IsNaN, infinities and signed
forward zeros require exact bits. Float64 sums and transcendental powers cannot
in general equal ordered float32 arithmetic: inputs/parameters are rounded to
float32, each square/add/multiply rounds, and MathF.Pow returns float32. The
absolute component controls cancellation near zero, relative component controls
large denominators. Special overflow is tested separately with explicit expected
IEEE categories, because a float64 oracle does not overflow at 1e30 squared.
Optimized/scalar comparisons use **zero tolerance, raw bits**, including NaNs.

Numerical checks use fixed seeds 82000, 82112, 82400, 82428 and 82572,
N=2, scale=0.125 to make cross terms substantial, power=0.75 and cmrnorm minDiv=2.
Geometries: cmrnorm C7/2×2/size5, rnorm C2/6×6/size5, cmrnorm C7/1×1/size4,
rnorm C1/6×6/size4 and blocked cmrnorm C7/1×1/size3. Every coordinate is perturbed,
including first/last channels and every edge/corner. The independent double loss
enumerates forward membership; central differences use epsilon 1e-5. A separate
output-scattered double Jacobian must agree within `2e-8*(1+abs(numeric))`;
float32 backward on each engine must agree within `4e-6*(1+abs(numeric))`.
All **600 comparisons** pass (300 coordinates per engine): worst float32 error
**1.53097574e-7**, worst independent float64 error **2.79810397e-10**.

## Optimized implementation and real geometry

Optimized forward owns output rows, backward owns input rows, and SIMD lanes own
independent cases. Small ordered window tables store channels or plane indices;
window reductions retain every scalar addition in ascending order. There are no
sliding sums, horizontal SIMD reductions, atomics or shared accumulators. Pow is
scalar per lane; vector multiplication/division retain the same operation order.
Backward creates one float32 Rows×Cases temporary; each parallel pass completes
before the next begins. No mutable layer state depends on a prior call.

Workspace payload: forward needs no float workspace. Backward allocates
`4*C*H*W*N` bytes, preserving saved outputs rather than overwriting them. For
N=128 this is 70.898 MiB per conv1 norm and 45.563 MiB per conv2/spatial norm.
Ordered tables retain at most `2*C*size` integer entries for cross-map, or
`2*H*W*size²` for spatial, plus arrays/headers. They are batch/thread independent;
no duplicated per-worker float buffer is allocated. This port intentionally uses
more live memory than CUDA's destructive activation reuse; no historical buffer
sharing policy is inferred from a layer primitive.

36 random seeds 81000–81035 cover both kinds, widths/channels below/equal/above
windows, sizes 1–7 (odd/even), blocked groups, N=1/2/7/8/9/17/128 on this runtime,
minDiv 1/2, scales 0.03/0.00002, powers 0/0.5/0.75/1/1.25, negative inputs,
signed zeros, NaN/inf and overflow. Scales 0/1/0.5/-1 apply to target accumulation.
All denominator/output/input-gradient bits are compared at threads 1, 2, all 24,
and repeated 24. Strided fixtures include N=17 to cover both SIMD lanes and tail.

All six 145 norms are created from the parsed config and inferred shapes:

| Layers | Geometry | Configured scale → runtime scale | Power | Base |
| --- | --- | --- | --- | --- |
| rnorm1a/b | 48×55×55 | 0.0001 → 0.00002 | 0.75 | 2 |
| rnorm2a/b | 128×27×27 | 0.0001 → 0.00002 | 0.75 | 2 |
| cnorm2a/b | 128×27×27 | 0.001 → 0.00004 | 0.75 | 1 |

Default mode compares full scalar/optimized output, denominator and backward at
N=3 with finite checks. Full mode adds N=17 at 1/2/all/repeated-all with explicit
accumulation checks. Both towers assert every effective graph edge in
`conv1 → cmrnorm → pool1 → detached relu` and
`conv2 → detached relu → cmrnorm → rnorm(cnorm2) → pool2`, including absence of a
conv1 neuron. Config/runtime values and exact shape element counts are asserted.

## Specification corrections

- `CUDACONV2.md` §7 now cites A's actual `kFCNorm` body, not merely its host
  dispatch. The old cudaconv3 reference was relevant for minDiv, but could not
  stand in for A's denominator cache. B is explicitly supplementary evidence.
- The verdict's former “zero ambiguity” phrasing now identifies a source-based
  reconstruction; the missing September kernel has not been executed.
- §6–7 and `LEDGER.md` now state CUDA's destructive saved-output preliminaries,
  the actual gradient-before-output multiplication order, reverse bounds for even
  windows, blocked partial groups, spatial ignoring blocked, intrinsic pow/division
  use and this port's preserving-buffer/MathF choices.
- No configured 145 behaviour or ANX-2..7 bug required an implementation fix.
  Machine-readable ledger/manifest and previous fixtures remain unchanged.

## Progress and final verification

23:31 UTC: initial Debug build passed with 0 warnings/errors and the integrated
suite passed 486 tests + spec checks in 7.087 s. Initial Release build also passed
with 0 warnings/errors. Subsequently widened randomized pow coverage and full-size
N=17 checks; all final builds and fresh-process gates then passed. Results below.


## Batch-128 timings

CPU: Intel Core Ultra 9 285HX, 24 logical processors / workers, SIMD 8 float lanes,
.NET SDK 10.0.403, Release. Separate benchmark process; no concurrent test or build
workload during measurements. Each operation has three untimed warmups and five
timed samples; median milliseconds below include allocation and task scheduling.
Constructor/window-table setup is outside the timed kernel call; denominator cache
is a preallocated output. Backward timing includes its temporary allocation.
Raw arrays are in ignored `.scratch/anx8/benchmark.json`. These isolated layer
microbenchmarks do not predict sustained trainer throughput; JIT tiering, mixed CPU
cores, thermals and allocation/GC noise can differ between the two towers.

| Layer | Scalar forward ms | Optimized forward ms | Scalar backward ms | Optimized backward ms |
| --- | ---: | ---: | ---: | ---: |
| rnorm1a | 256.906 | 19.694 | 303.842 | 18.172 |
| rnorm1b | 258.568 | 8.458 | 305.266 | 17.759 |
| rnorm2a | 167.091 | 4.453 | 185.048 | 11.356 |
| rnorm2b | 167.882 | 4.731 | 189.000 | 11.818 |
| cnorm2a | 399.009 | 8.326 | 444.374 | 12.986 |
| cnorm2b | 398.803 | 6.699 | 446.537 | 13.136 |

The optimized path is retained: measured forward speedup 13.0–59.5×, backward
16.0–34.2× over the scalar loop at N=128. First-tower forward timing is higher
than its identical second-tower geometry; both raw measurements are retained.
Timing is informational, not a performance assertion in the default gate.

```sh
dotnet build
dotnet build -c Release
dotnet run --project tests/AlexNetSharp.Tests
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --full-scalar-parity
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-norm --norm-output .scratch/norm-benchmark.json
```

## Final gates and integrity

The installed `dotnet` executable ran the requested builds and test commands.
CLI state, NuGet packages/HTTP cache (`NUGET_HTTP_CACHE_PATH`) and temporary files
were redirected under ignored `.scratch/anx8/`. Both final builds report
**0 warnings / 0 errors**: Debug MSBuild 1.60 s (process 1.722 s), Release MSBuild
0.57 s (process 0.691 s). Distinct process IDs and exact logs/results are in
ignored `.scratch/anx8/gates.json` and matching log files.

| Fresh process | Exit | Tests + spec checks | Suite seconds | Process wall seconds, build/startup included |
| --- | --- | --- | ---: | ---: |
| default-debug-1 | 0 | 486 | 6.962 | 7.980 |
| default-debug-2 | 0 | 486 | 7.126 | 8.131 |
| default-debug-3 | 0 | 486 | 7.060 | 8.010 |
| full-release | 0 | 507 | 22.967 | 23.916 |

All three default Debug processes remain comfortably below the 20 s target.
Default includes every previous check and 106 new norm groups. Full mode includes
all existing scalar convolution/pool checks plus N=17 norms at every worker count
and accumulation. Both modes verify 569 parser fields, 1470 shape fields and
455 original citations. All 600 norm numerical comparisons have the same errors
reported above in Debug and Release.

Final audit:

- All 246 baselined original files (excluding `.git`) retain identical relative
  names, sizes, nanosecond mtimes and SHA-256 hashes, with no additions/removals.
  No original-tree files or Git metadata were written.
- Candidate A archive hash matches the specified digest; both extracted files
  equal archive bytes and have write permissions removed. Candidate B is also
  read-only. All candidate bytes remain in ignored scratch.
- Fixture regeneration is byte-identical; SHA-256
  `578f23f85f81d5e94c37377e301c2ca1e77a8e0d9dcac4f3bcd30fcc3b2d3fc2`.
- Every pre-existing tracked source file, fixture and machine-readable spec is
  byte-identical to HEAD, except the documented test entry-point registration.
  The only other prior-file edits are README and the two narrative specs.
- All 12 deliverable files use LF and end with a newline, with no home/environment
  paths, hostnames, IPs, secrets or copied candidate source. `git diff --check`
  passes and no changes are staged.
- Authored files, downloads, build outputs, CLI/package caches and temporary
  artifacts stayed in this repository. No outside file was intentionally changed;
  original-tree content/mtime integrity is independently checked as above. No
  commits, pushes, stashes or process-killing commands were used.

Nothing is unfinished. Historical CUDA bitwise parity is outside the available
source/execution evidence; CPU scalar/optimized bit parity is proven at zero
tolerance for the tested workloads and worker counts.
