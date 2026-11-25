# ANX-3 Report — Original parser shape inference

Started: 2026-11-25T14:07:17Z. Base: `cbc09a0`. Finished: 2026-11-25T14:19:02Z (11.8 min elapsed). No commits or pushes.

## Brief items

| Item | Status | Evidence | Time |
| --- | --- | --- | --- |
| 1. Shape inference | done | `Shapes/ShapeInference.cs`, `NetworkShapes.cs`; both full-network cases and 78 hand boundary/error cases pass, including every 145 type. | 14:07–14:16 UTC (~9 min, including source review and diagnostic fixes) |
| 2. Full-network fixtures and hand comparison | done | Independent integer oracle, 49-layer fixtures for 224/227; full comparison and numeric parameter assertions pass. Tables below. | 14:09–14:17 UTC (~8 min) |
| 3. Boundary and error fixtures | done | `shape-cases.json`: 33 numeric cases and 45 exact original-message cases pass. Inventory below. | 14:10–14:17 UTC (~7 min) |
| 4. Ledger link | done | 1,470 fields across 49 layers × 2 crop sizes; 4 negative checks detect missing/changed declared and detached fields; existing spec checks pass. No ledger gaps. | 14:12–14:17 UTC (~5 min) |
| 5. Report and verification gates | done | Build 0 warnings/errors; all 101 tests + spec checks pass in 3 fresh processes; source integrity, LF/path and diff checks pass. Nothing unfinished. | Report kept current from 14:07; final report/gates 14:16–14:19 UTC (~3.0 min) |

## Source inspection

References use upstream-relative paths only. Original source is read-only and resolved through `local.env`.
Data dimensions are provider inputs (`layer.py:1099`, `convdata_jpeg.py:73,99-102`); 224 is a default,
not a recovered historical fact (`OQ-LEDGER-1`).

## Findings during verification

First build passed with 0 warnings / 0 errors (2.39 s); the first integrated shape/spec check linked
1,470 fields and 455 citations. Its independent total-parameter assertion exposed a pre-existing
arithmetic error in `LEDGER.md`: the convolution subtotal 2,332,704 counted weights alone, despite
the displayed formula including 1,376 biases. Correct subtotal: 2,334,080. Correct network total:
60,965,224, not 60,963,848. Corrected the prose and retained independent numeric assertions.
Proven ANX-2 diagnostic bugs were corrected after this first run: its early filter-count equality
check emitted a truncated message without the input dimensions. Original `layer.py:1011-1012`
checks filter counts together with spatial widths, reporting every `CxHxW`. The equality check
now runs in shape inference, where the required dimensions exist. Valid parses and the ANX-2
golden fixture are unchanged. New fixtures cover both unequal widths and unequal filter counts.
Likewise, pool `sizeX` and spatial-norm `size` were checked against a minimum alone, so zero
reported "greater than or equal to 1" instead of the original "in the range 1-imgSize"
(`layer.py:1148,1207`). These checks now run in shapes with both bounds; pool stride follows
sizeX there (`layer.py:1149`). New fixtures assert the original full messages. This is a
validation-phase change for invalid configurations only; configuration decoding remains ANX-2's.


## API and shape semantics

`ShapeInference.Infer(NetworkDefinition, DataDimensions?)` returns `NetworkShapes.Layers` in the
ANX-2 parser order: 35 file-order declared sections, then 14 detached neurons in source-section order.
Dependencies are resolved before consumers without reordering that result. Effective input names and
`num_inputs` retain input order. `DataDimensions` accepts arbitrary provider dimensions keyed by `dataIdx`
and the class count; `DataDimensions.Jpeg(innerSize: 224, classes: 1000)` is the default. A 227 crop
has 154,587 image values, compared with 150,528 at 224. There is no graphics dependency or tensor allocation.

As in the original parser, `img_size` and `channels` describe image **inputs**, so convolution fields
are lists (one per input); norm/pool lists have one element. Data, FC, hs, softmax, cost and neuron
layers have no such original fields: empty arrays mean inapplicable. `modules_x`, `modules` apply
to convolution; `outputs_x` to pooling; otherwise they are null. Output image size is `modules_x`
for conv, `outputs_x` for pool, and unchanged `img_size` for norms. A detached neuron preserves
its input output count. Cost has `outputs=0` (`layer.py:187,1261-1273`), not an activation tensor.
Weights are `[rows, columns]` in `inputs` order; biases are `[rows, columns]` or an empty array.
Counts are parameter slots, with no batch dimension. The slot-count properties do not deduplicate
shared weights; layers-145 has no sharing.

The per-input partial flag is derived from the last window's exclusive right/bottom bound. For
conv: `(modulesX-1)*stride + filterSize > imgSize - 2*paddingStart`; for pool:
`start + (outputsX-1)*stride + sizeX > imgSize`, with a positive output width. This excludes
ordinary symmetric configured padding from the flag. Config padding is already negated by ANX-2;
the engine uses `paddingStart` directly, never negates the stored config again (`layer.py:1041`).
These are shape calculations only, with no forward/backward or connectivity/initialization functions.

## Hand comparison with the paper and ledger

Using `1 + ceil((image + 2*padding - filter)/stride)` (`layer.py:1004`):

| Check | Hand calculation | Result |
| --- | --- | --- |
| conv1a, 224 | `1+ceil((224-11)/4)=55`; last starts `54*4=216`, ends 226 inclusive | `48×55×55=145200`; partial, 3 extra pixels at right/bottom |
| conv1a, 227 | `1+ceil((227-11)/4)=55`; last ends 226 inclusive | Same count; full last window |
| pool1a | `1+ceil((55-3)/2)=27` | `48×27×27=34992`; no partial |
| conv2a | `1+ceil((27+4-5)/1)=27` | `128×27×27=93312`; W `48*25 × 128` |
| pool2a | `1+ceil((27-3)/2)=13` | `128×13×13=21632` |
| conv3a | Both inputs: `1+ceil((13+2-3)/1)=13` | `192×13×13=32448`; two W `128*9 × 192` |
| conv5a | `1+ceil((13+2-3)/1)=13` | `128×13×13=21632`; W `192*9 × 128` |
| pool3a | `1+ceil((13-3)/2)=6` | `128×6×6=4608` |
| fc2048a | Two inputs of `128*6*6=4608` | W `[4608,2048]` for each tower, in input order |
| fc2048bb | `hs1b,hs1a`, each 2048 | W[0] for hs1b, W[1] for hs1a: each `[2048,2048]` |
| fc1000 | Two 2048 inputs, config outputs 1000 | Two W `[2048,1000]`; output 1000 |

The [2012 paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf),
§3.5 and Figure 2, supports the two-tower connectivity, channel counts and classifier widths.
Its Figure 2 first-layer neuron total is 253,440, whereas this parser gives 290,400 across both towers:
the explicit ceiling formula in the released code controls this implementation. The ledger retains
that distinction and OQ-LEDGER-1; 224 is a documented default, not a recovered launch command.

The complete output table follows. Image output notation is `channels×height×width`; vector outputs
have one number. Counts and W shapes are identical at both crops except the data count. Inputs in
this table are the effective, detached-neuron names. Input image sizes differ only for conv1a/b
(224 versus 227); all later image dimensions are identical.

| Layer | Effective inputs (matrix order) | Output dimensions at 224 / 227 | Outputs at 224 / 227 | W per input | Partial at 224 / 227 |
| --- | --- | --- | --- | --- | --- |
| data | — | 3×224×224 / 3×227×227 | 150528 / 154587 | — | — |
| labels | — | 1 | 1 | — | — |
| conv1a | data | 48×55×55 | 145200 | 363×48 | yes / no |
| conv1b | data | 48×55×55 | 145200 | 363×48 | yes / no |
| rnorm1a | conv1a | 48×55×55 | 145200 | — | — |
| rnorm1b | conv1b | 48×55×55 | 145200 | — | — |
| pool1a | rnorm1a | 48×27×27 | 34992 | — | no / no |
| pool1b | rnorm1b | 48×27×27 | 34992 | — | no / no |
| conv2a | pool1a_neuron | 128×27×27 | 93312 | 1200×128 | no / no |
| conv2b | pool1b_neuron | 128×27×27 | 93312 | 1200×128 | no / no |
| rnorm2a | conv2a_neuron | 128×27×27 | 93312 | — | — |
| rnorm2b | conv2b_neuron | 128×27×27 | 93312 | — | — |
| cnorm2a | rnorm2a | 128×27×27 | 93312 | — | — |
| cnorm2b | rnorm2b | 128×27×27 | 93312 | — | — |
| pool2a | cnorm2a | 128×13×13 | 21632 | — | no / no |
| pool2b | cnorm2b | 128×13×13 | 21632 | — | no / no |
| conv3a | pool2a, pool2b | 192×13×13 | 32448 | 1152×192, 1152×192 | no / no |
| conv3b | pool2a, pool2b | 192×13×13 | 32448 | 1152×192, 1152×192 | no / no |
| conv4a | conv3a_neuron | 192×13×13 | 32448 | 1728×192 | no / no |
| conv4b | conv3b_neuron | 192×13×13 | 32448 | 1728×192 | no / no |
| conv5a | conv4a_neuron | 128×13×13 | 21632 | 1728×128 | no / no |
| conv5b | conv4b_neuron | 128×13×13 | 21632 | 1728×128 | no / no |
| pool3a | conv5a | 128×6×6 | 4608 | — | no / no |
| pool3b | conv5b | 128×6×6 | 4608 | — | no / no |
| fc2048a | pool3a_neuron, pool3b_neuron | 2048 | 2048 | 4608×2048, 4608×2048 | — |
| fc2048b | pool3a_neuron, pool3b_neuron | 2048 | 2048 | 4608×2048, 4608×2048 | — |
| hs1a | fc2048a_neuron | 2048 | 2048 | — | — |
| hs1b | fc2048b_neuron | 2048 | 2048 | — | — |
| fc2048ba | hs1a, hs1b | 2048 | 2048 | 2048×2048, 2048×2048 | — |
| fc2048bb | hs1b, hs1a | 2048 | 2048 | 2048×2048, 2048×2048 | — |
| hs2a | fc2048ba_neuron | 2048 | 2048 | — | — |
| hs2b | fc2048bb_neuron | 2048 | 2048 | — | — |
| fc1000 | hs2a, hs2b | 1000 | 1000 | 2048×1000, 2048×1000 | — |
| probs | fc1000 | 1000 | 1000 | — | — |
| logprob | labels, probs | — (cost) | 0 | — | — |
| pool1a_neuron | pool1a | 48×27×27 | 34992 | — | — |
| pool1b_neuron | pool1b | 48×27×27 | 34992 | — | — |
| conv2a_neuron | conv2a | 128×27×27 | 93312 | — | — |
| conv2b_neuron | conv2b | 128×27×27 | 93312 | — | — |
| conv3a_neuron | conv3a | 192×13×13 | 32448 | — | — |
| conv3b_neuron | conv3b | 192×13×13 | 32448 | — | — |
| conv4a_neuron | conv4a | 192×13×13 | 32448 | — | — |
| conv4b_neuron | conv4b | 192×13×13 | 32448 | — | — |
| pool3a_neuron | pool3a | 128×6×6 | 4608 | — | — |
| pool3b_neuron | pool3b | 128×6×6 | 4608 | — | — |
| fc2048a_neuron | fc2048a | 2048 | 2048 | — | — |
| fc2048b_neuron | fc2048b | 2048 | 2048 | — | — |
| fc2048ba_neuron | fc2048ba | 2048 | 2048 | — | — |
| fc2048bb_neuron | fc2048bb | 2048 | 2048 | — | — |


## Parameter count

| Stage | Weight entries | Bias entries | Total |
| --- | ---: | ---: | ---: |
| Conv1 (both towers) | 34,848 | 96 | 34,944 |
| Conv2 (both towers) | 307,200 | 256 | 307,456 |
| Conv3 (both towers, two matrices each) | 884,736 | 384 | 885,120 |
| Conv4 (both towers) | 663,552 | 384 | 663,936 |
| Conv5 (both towers) | 442,368 | 256 | 442,624 |
| All convolution | 2,332,704 | 1,376 | 2,334,080 |
| FC1 (4 matrices of 4608×2048) | 37,748,736 | 4,096 | 37,752,832 |
| FC2 (4 matrices of 2048×2048) | 16,777,216 | 4,096 | 16,781,312 |
| FC1000 (2 matrices of 2048×1000) | 4,096,000 | 1,000 | 4,097,000 |
| All FC | 58,621,952 | 9,192 | 58,631,144 |
| Network | **60,954,656** | **10,568** | **60,965,224** |

The weight-only count is 60.954656 million. The total exceeds the paper's approximate 60 million
by 965,224 (about 1.61%), consistent with rounded prose rather than a topology change. Biases
contribute only 10,568; they do not account for that rounding difference. The extra spatial norms,
neuron detachment and hs layers have no trainable weights. Both crop sizes yield the same
pool3 sizes and therefore the same FC parameter counts. There are 22 weight matrices and 15
bias vectors. The independent arithmetic is tested, including the corrected ANX-1 subtotal.

## Boundary/error fixture inventory

Every numeric fixture is hand-sized; each error fixture asserts the **entire** message, not just a
substring. 33 successful cases + 45 errors = 78 fixtures. Their `citation` fields identify original
source checks. The test runs config parsing and then inference, so checks already owned by ANX-2
remain exercised as part of the pipeline.

| Original checks | Fixture coverage |
| --- | --- |
| Conv stride/filter/padding/channels/groups ranges (`layer.py:992-1001`) | Zero stride/filter/channels/groups, negative padding, zero computed imgSize, list cardinality |
| Conv square-image predicate (`:1020-1021`) | 72-dimensional non-square input; 4-value/3-channel Python-2 predicate quirk retained |
| Conv output equality (`:1011-1012`) | Unequal spatial widths and unequal filter counts; full dimension diagnostics |
| Channels <4 or divisible by4 (`:1022-1023`) | Reject 5 channels |
| Filter ≤ image+2padding (`:1024-1025`) | Exact-bound filter and oversized filter |
| Group/sparse restrictions (`:1027-1039`) | groups=1 with sparse; channels/filterChannels, filterChannels/4, groups*filterChannels/channels, channels/(4*groups), filters/(16*groups); valid grouped, sparse explicit/default cases |
| Conv ceiling/padding/partialSum/bias shape (`:1004,1041,1066-1073`) | Ceiling vs floor differing/agreement, stride1 and2, symmetric padding and partial padding, valid/invalid partialSum, unshared biases |
| Pool ranges and kind (`:1148-1160`) | sizeX 0/too-large, stride 0/too-large, negative outputsX, channels0, invalid pool; all 4 accepted kinds |
| Pool gradient ancestry (`:1158-1159`, `:293,314,830`) | Frozen weighted ancestor still triggers channel/16 check; direct data does not; valid16-channel case |
| Pool square-image check and dimensions (`:1161-1165`) | Non-square, ceiling vs floor, nonzero/negative start, explicit output override and overrun; start past image retains original negative computed width squared |
| Cross-map norm (`:1200-1204`) | Channels range, lower/upper window bounds, channel/16 check, square-image check, blocked shape preservation |
| Spatial norm (`:1207-1214`) | Lower/upper window bounds, channels range, channel/4 restriction, non-square, preserved output and Python-2 small-input predicate |
| FC (`:882-884`) | Positive output range, weight/bias dimensions, asymmetric two-input order, full-network fan-in |
| Shared weight dimensions (`:802-804`) | Mismatched source matrix |
| Neuron/hs/softmax (`:649,705,1110`) | Exact dimension preservation; hs keep is not range checked by original |
| Cost and base input checks (`:368-370,1263-1270`) | Zero-output input, nonscalar labels, non-softmax probs, wrong number of classes; valid cost has outputs0 |

The separate `channels % groups` check (`layer.py:1037`) cannot fail independently: for groups>1
it is implied by the preceding `channels % (4*groups)` check; for groups1 it always passes.
This branch is proven algebraically and covered by the valid grouped case. No new restriction
is invented for pool `start` or hs `keep`. Checks on runtime arrays returned by custom initializer
functions (`layer.py:779-819,953-961`) require executing initialization and are outside this
shape-only story; the 145 config uses no such functions. No random connectivity is generated.

## Ledger linkage and independent fixture generation

`shape.inference_by_inner_size` in each ledger section (and each detached-neuron `shape`) stores
all 15 emitted fields at both crop sizes, including identity, input order, empty arrays and nulls.
The checker serializes actual inferred shapes and compares every field, rejects unrecorded fields,
and also compares ANX-1's existing output/weight/bias fields at 224. Four mutation checks remove
or change `outputs` on both a declared and a detached layer and verify rejection. Existing source
citation checks still run; the 10 added citations cover original arithmetic and the derived flag.
There are **no ledger gaps**.

`tests/fixtures/make_shape_fixture.py` independently computes the full fixture from ANX-2's parsed
golden using integer `isqrt` and `-(-n//stride)` ceiling; the C# engine uses `Math.Sqrt` and
`Math.Ceiling`. It never calls the implementation. Normal tests reparse the original 145 configs,
recompute all shapes and compare every fixture field under fr-FR culture. The asymmetric FC
boundary case detects swapped inputs even though full-network FC input sizes are equal.

Item intervals above overlap because source review, fixtures, implementation and ledger checks
were interleaved; durations are elapsed task intervals, not additive billed work.


## Final verification

Final build: **0 warnings, 0 errors**, exit 0, MSBuild elapsed **0.79 s**.
All tests run as the existing executable test harness; there are no external test-framework packages.
Each of the following is a separate fresh OS process, running the compiled test assembly with no build.

| Run | Exit | Suite time | Process wall time | Results |
| --- | --- | --- | --- | --- |
| 1 | 0 | 119 ms | 0.136 s | 16 parser + 85 shape checks, spec and shape ledger checks passed |
| 2 | 0 | 115 ms | 0.133 s | 16 parser + 85 shape checks, spec and shape ledger checks passed |
| 3 | 0 | 118 ms | 0.136 s | 16 parser + 85 shape checks, spec and shape ledger checks passed |

The 85 shape checks comprise 2 complete-network cases, 1 hand-comparison/default-input check,
78 boundary/error fixtures and 4 ledger mutation checks. Combined with the existing 16 ANX-2
checks, **101 tests pass per process**. Each spec run checks 35 sections, 10 layer types,
15 weighted layers, 569 parser parameters, 455 source citations and 1,470 emitted shape fields.

Reproduction from repository root (all caches and CLI state remain inside the repository):

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

Final logs: `.scratch/anx3/final-build.log`, `final-tests-1.log`, `final-tests-2.log`,
`final-tests-3.log` (the latter three in the same directory). Independent Python fixture
regeneration produced byte-for-byte identical JSON (SHA-256
`6ddc0b3d36577b9e58fa74085028e31a952be690566f56a681df3e9c82f83a2e`).
`git diff --check` passed. All 12 changed/new deliverable files use LF and contain no literal
home path. The original tree's **274 files** match their pre-implementation size, mtime and
SHA-256 baseline; no original file was modified. Intentional file writes and retained build/cache
artifacts stayed in this repository; no files outside it were edited. No stash, commits or pushes.
All five brief items and all verification gates are complete; nothing is unfinished.
