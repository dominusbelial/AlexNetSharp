# VIZ-1 — Explorable browser replay

Report created before implementation on 2026-11-26 UTC (2026-11-25 local date).
Base: `ca9f46c`. Work remains in the working tree; no commits or pushes.
Original-tree baseline recorded at 01:29:35 UTC. All brief items and gates passed.
Finished at approximately 01:53 UTC (~25 minutes including reading, implementation and verification).

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. KNI BlazorGL web app | done | Solution builds cleanly; trimmed browser startup and banner passed at both viewports; Core unchanged. | ~01:31–01:49 UTC, 18 min |
| 2. Snapshot format and producer | done | 49 layers, actual optimized forward, 8,235,834 raw bytes; byte determinism and direct Core comparison passed. | ~01:29–01:43 UTC, 14 min |
| 3. Layer stack and pure layout | done | All boxes, input edges, cross-tower colors, explicit neurons, reversed slots; projection/ray/nonoverlap tests pass. | ~01:31–01:49 UTC, 18 min |
| 4. Explain cards | done | 59 Markdown cards, all instance/type/link checks, 184 original citations, multiple browser-pinned cards and deeper formula passed. | ~01:33–01:47 UTC, 14 min |
| 5. Activations, filters and receptive fields | done | Runtime textures; actual atlas pixel pick, numeric edge range, GL pixel checks and 96 RGB filter color check passed. | ~01:33–01:49 UTC, 16 min |
| 6. Tests | done | 702 groups + spec checks in three fresh Debug processes; 729 full Release; direct/determinism and skip-branch proofs pass. | ~01:36–01:53 UTC, 17 min |
| 7. Headless browser proof | done | Clean trimmed publish; both viewports, zero console errors, actual GL box/activation/filter/RF pixels and interactions pass. | ~01:40–01:52 UTC, 12 min |
| 8. Report and gates | done | Clean builds, fresh-process test/producer gates, original/Core/index checks and two bounded screenshots. | ~01:28–01:53 UTC, 25 min, throughout |

Times are approximate overlapping work intervals, including diagnosis and verification, not additive billing.

## Architecture

`AlexNetSharp.Web` is a .NET 10 Blazor WebAssembly app, KNI BlazorGL 4.3.9001,
`Microsoft.AspNetCore.Components.WebAssembly` 10.0.10, `PublishTrimmed=true`,
`TrimMode=partial`. Explicit GameFactory, InputFactory and GraphicsFactory
registration survives trimming. The game starts from Blazor after first render;
JavaScript supplies browser animation, pointer/keyboard events and sizing. HTML
renders text, tooltips, numeric controls and independent pinned explain cards.
`StackGame` uses BasicEffect for 3D boxes/edges and SpriteBatch for runtime textures.
No MGCB, XNB, content pipeline or downloaded font/media dependency exists.

`AlexNetSharp.Replay` references graphics-free Core and contains the immutable
snapshot DTOs/source-generated JSON context, validation, placeholder producer,
System.Numerics layout/camera/picking, receptive fields, atlas geometry and card
parser. The console producer references this library. Web references Core directly
as required; no Core implementation, result, fixture or machine-readable prior spec
changed. The browser consumes replay bytes; it does not execute inference.
`ISnapshotSource` separates replay acquisition from visualization. `ReplaySource`
is its only implementation. `IAudioLayer` is an empty future stub.

Two browser integration bugs were found by proofs and fixed: focusing an embedded
canvas must prevent automatic page scroll before mouse-up; KNI presentation metadata
must match the actual canvas dimensions before viewport-to-GL conversion (browser
window/focus events otherwise shift rendered pixels). The game uses one viewport
and maps the layer stack into the top half through its projection. A canonical JSON
round-trip check also caught culture-sensitive SortedDictionary reconstruction;
parameter keys are explicitly restored to ordinal order after deserialization.
These are new web/replay implementation fixes, not changes to ANX-2..10.

## Snapshot, placeholders and provenance

Full schema/extension documentation: [REPLAY.md](../spec/REPLAY.md).
The committed configs in `layers/` are byte-identical upstream 145 configs, with
an upstream licence and provenance notice. ANX-2 parsing and ANX-3 shape inference
produce all 35 declared and 14 detached layers. Ordered, recursive input-first
execution calls optimized convolution, pooling, ReLU, both normalizations, FC,
dropout, softmax and cost. The four dropout layers run **eval (×keep)**. The cost
uses synthetic label index 0; top-5 predictions have class indices, not invented
class names or claims about accuracy.

**Placeholder, not ANX-11 initialisation:** seeded SplitMix64 uniforms and
Box-Muller Gaussian samples with each input's configured `initW`, biases from
configured `initB`. Separate multi-input matrices and input order are preserved.
This is not historical NumPy/CUDA RNG parity, optimizer work or training.

**Not the ANX-12 pipeline:** resize short side to 256 with nearest-neighbour
sampling, centre crop to 224×224, subtract RGB constants (123,117,104), no /255.
The single committed sample is a deterministic 256×256 synthetic RGB colour study,
196,623 bytes, CC0-1.0, created here with a documented pixel formula. There is no
external photo or uncertain licence. See [sample provenance](../../content/samples/PROVENANCE.md).
The replay carries the image checksum and restores these means to show the actual
preprocessed crop, rather than displaying a different source image.

`manifest.json` schema version 1 records step=0, config/params/image SHA-256, seed,
placeholder recipes, type/layout/mode, per-layer identity/shape/tower/ordered inputs/
consumers/neuron placement/parameters/local geometry/statistics, top-5 indices and
three cost metrics. `activations.bin` is **float32 little-endian**, channel/y/x
order, every layer's one-image output and **96×3×11×11 conv1 filters**. Cost has
zero output elements and a zero-length blob; its metrics are in the manifest.
Each blob has offset/count/checksum, and validation checks contiguity, dimensions,
finite values, graph edges/cycles, required constructor fields and version/mode.
The combined raw files are bounded below **10,000,000 bytes**.

The direct proof independently invokes the optimized Core forward path at two
workers and compares every activation's raw float bits, all conv1 filters and cost
with the serialized replay. Two repeated productions are byte-identical. Final
gates additionally compare two fresh producer processes and the committed files.
Determinism is proven on this runtime/platform; no cross-platform transcendental
or historical CUDA rounding identity is asserted.

Future trainers can append immutable step directories through a run index. Later
versions can add curves, gradient statistics, checkpoint/RNG identifiers and more
filter blobs. A live adapter can implement the same source boundary. Those writers,
transports and fields are documented extension points, not implemented features.

## Layout, cards and pixel rules

The pure layout assigns 2.15-unit topological stages and towers at y=±2.3. Data
is at the first end; labels are at the output end, near cost. Box extents use
log₂(C/H) scale, as specified numerically in REPLAY.md. An orthographic camera fits
all layers; mouse drag or arrows orbit, shift/right drag or WASD pan, wheel or +/−
zoom, and R resets. Picking inverts the same projection/view matrices and performs
nearest positive slab-ray intersections against world boxes. Screen coordinates
for browser automation come directly from this class, without hard-coded box
locations. HTML labels identify all 14 `*_neuron` boxes. Amber cross-tower edges
and numbered input-slot offsets retain all graph inputs, with an explicit
`fc2048bb: 0=hs1b, 1=hs1a` annotation/card.

There are 49 instance cards and 10 type cards in `content/explain/*.md`, each with
front matter, summary, paper section, original file:line citations, significance,
formula and related-card links. Instance cards combine this content with actual
snapshot shape, parameter count/values and min/max/mean/std. Hover gives a tooltip;
click pins independently closable cards; deeper shows formulas and live substitutions
where meaningful. Original citations are tested against `ALEXNET_ORIGINAL_SRC`;
when absent the new card check prints an explicit skip. No candidate A citations
are needed by these cards, and no candidate source was added.

Conv/pool/norm selection builds per-channel runtime texture atlases. The color map
runs from midnight blue to mint, min/max **per channel**, constant channels at the
midpoint. The conv1 filter atlas uses RGB, min/max across the three channels **per
filter**. Nearest sampling preserves map pixels. Numeric channel/x/y inputs permit
precise selection on small screens, alongside actual atlas pixel picking.

Receptive fields are structural potential dependencies: recurse backwards through
stride/start/kernel, clip at every intermediate tensor, and union input branches.
Spatial normalization expands the range; cross-map normalization/ReLU preserve
spatial coordinates. Both unclipped nominal and actual contributing inclusive
input bounds are displayed. This includes potential pool-window contributors,
not only this example's winning pixels. The yellow outline is drawn on the exact
input crop. Original `pool3a/b` is the pool after conv5, often called pool5.

| Selected unit | Nominal x/y range | Contributing input range |
| --- | --- | --- |
| conv1 (0,0) | 0..10 | 0..10 |
| conv1 (54,54) | 216..226 | 216..223 (partial padding) |
| pool1 (0,0) | 0..18 | 0..18 |
| pool1 (26,26) | 208..226 | 208..223 |
| conv2 (0,0) | -16..34 | 0..34 |
| conv2 (13,13) | 88..138 | 88..138 |
| pool after conv5 (2,2) | -16..210 | 0..210 |
| pool after conv5 (5,5) | 80..306 | 80..223 |

The pool-after-conv5 field includes the extra 5×5 spatial norm: nominal width 227,
stride 32. Tests independently enumerate actual index dependency sets using typed
Core parameters and compare them with these hand-computed bounds.

## Tests and gates

New default tests: 19 groups, including all 49 world boxes, both projected viewport
sizes, known rays, moved-camera picking, no overlaps, every input edge, cross-tower
and reversed input slots; eight hand/independent receptive-field cases; all 59 cards,
links and 184 original citations; required-field/version/checksum/truncation/size
rejection; canonical ordinal JSON round trip; activation statistics/top-5; input,
activation/filter atlas geometry and actual pixel picking. Existing tests are
unchanged except registration/project reference. The producer/direct-forward proof
is explicit `--snapshot-proof`, outside the fast default suite.

Both whole-solution builds passed with **0 warnings / 0 errors**: Debug 3.941 s,
Release 3.219 s process time. Gate logs are in `.scratch/viz1`; the reusable runner
is `tools/viz1-gates.py`. All invocations below are fresh OS processes, exit 0.

| Gate | PID | Suite time | Process time |
| --- | ---: | ---: | ---: |
| Default Debug 1, 702 groups + spec | 655 | 12.143 s | 12.212 s |
| Default Debug 2, 702 groups + spec | 691 | 12.247 s | 12.301 s |
| Default Debug 3, 702 groups + spec | 726 | 12.339 s | 12.408 s |
| Full scalar parity Release, 729 groups + spec | 761 | 42.705 s | 42.765 s |
| Snapshot determinism/direct-Core proof | 797 | 5.353 s | 5.400 s |
| Fresh producer 1 | 810 | — | 2.051 s |
| Fresh producer 2 | 819 | — | 2.026 s |

Default remains comfortably below 20 s. New default replay tests take 0.122–0.126 s.
Each main run also verifies all 569 parser parameters, 1,470 shape fields and 455
existing original citations, alongside the 184 card citations. The absent-env card
skip path was also executed separately; the original Core suite still resolves its
checkout through the pre-existing local settings, while the new citation check
prints its explicit skip and the suite passes.

Two fresh console producers and the committed snapshot match exactly:

| File | Raw bytes | SHA-256 |
| --- | ---: | --- |
| manifest.json | 60,790 | `c5acc380b137ce33ca2c25384c01ffd0729ce09f0f65427a40754db5346b736d` |
| activations.bin | 8,175,044 | `bd4e4756dc8dee355f5dcf7649bc2146615916151233e563fef0562e14fc777d` |

## Headless browser proof

`tools/viz1-browser-proof.cjs` passed against a **clean** trimmed Release publish
(the script deletes its own prior publish output first, avoiding stale fingerprinted
assets in the size count). It creates a loopback-only server on an OS-selected port
and fresh browser contexts, with all profiles/caches/temp directories under scratch.
It closes the server and browser after the proof; no external hosting/deployment.

Both 1280×800 and 390×844 assert startup without console/runtime/HTTP errors,
49 layout-derived positions, the explicit untrained/parity banner, real mouse hover
on conv1, real click pinning and a second coexisting card, deeper live values,
actual channel-0 pixel (3,4) picking, numeric edge selection with x=216..223,
GL activation texture colors and yellow receptive-field outline, camera keyboard
motion/reset and the selected 3D box's exact green pixel after reset, and RGB filter
texture colors. The test does not substitute DOM layer buttons for box clicks or
call the pin handler directly. All samples are from the real KNI canvas.

| Viewport | Cold first frame | Console errors | Mint activation pixels | Yellow RF pixels | Distinct RGB filter colors | Screenshot bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1280×800 | 1,147.6 ms | 0 | 59,096 | 52 | 11,586 | 356,326 |
| 390×844 | 1,083.3 ms | 0 | 19,967 | 20 | 9,819 | 232,396 |

Load time is navigation-to-first-real-frame on local loopback with a fresh browser
context, not an Internet bandwidth claim. The app consumes the complete replay;
there is no server inference or fake activation data. Chrome uses headless software
WebGL for this proof. Published size is the sum of physical files, including the
standard compressed variants and debug/runtime files shipped by publish:

- Whole clean publish: **28,127,486 bytes**.
- Static `wwwroot`: **27,116,160 bytes**.
- Published replay directory: **8,248,914 bytes**, including compressed manifest variants.
- Raw snapshot: **8,235,834 bytes**, below the 10 MB bound.

Raw proof metrics: `.scratch/viz1/browser-proof.json`; publish output and browser
logs are under the same ignored directory. Two full-page screenshots, both under
400 KB, were copied into the report assets (no other screenshots are deliverables):
[desktop baseline](VIZ-1_assets/desktop.png). The historical mobile screenshot
was retired in VIZ-2 to keep the repository within its four-screenshot media
budget; see the clearly identified [current VIZ-2 phone view](VIZ-2_assets/web-phone.jpg).
Full-page images include scrollable content; browser viewport dimensions remain
those specified by the brief. On mobile, multiple pinned cards share a horizontal
scroll area.

## Integrity

All **274 original files** retain identical relative names, sizes, nanosecond mtimes
and SHA-256, with no additions/removals. Config copies match their original bytes;
inherited whitespace/comments are deliberately retained. Core, all earlier layer
implementations and fixtures/specs are unchanged. Only test registration/project
reference, the solution and README were edited among prior deliverables.
No candidate CUDA source/archive or read-only reference-application path appears
in deliverables. The original licence accompanies the copied configs.

All authored text is LF; no personal home paths, machine hostnames, IP literals or
secrets occur in deliverables. `git diff --check` passes. The only image/media
assets are the 196,623-byte synthetic input and the two screenshots above. Replay
float data is not media. Downloads, SDK/NuGet/npm caches, builds, browser profiles
and temporary files stayed under repository scratch; no outside file was edited
by task work. The original tree is independently byte/metadata checked. No commits,
pushes, stash, broad process-killing commands or staging operations were used;
the git index remains unchanged. Nothing in the brief is unfinished.

## Reproduction

All writable SDK, NuGet, npm, runtime, browser profile and temporary state is placed
under ignored `.scratch/`. The build/test runner does this itself:

```sh
python3 tools/viz1-gates.py
npm --cache .scratch/npm-cache install --prefix .scratch/npm playwright
node tools/viz1-browser-proof.cjs
```

The browser script publishes Release with trimming, serves static output on an
OS-selected loopback port, launches headless Chrome, saves screenshots/metrics under
`.scratch/viz1`, and closes its own server/browser. It is not part of default tests.
The gates runner records each process PID, exit and elapsed time in `gates.json`.

## Known gaps

No training, optimizer/ANX-11 initialization, ANX-12 data pipeline, trainer, live
source, in-browser inference, trained checkpoint, ImageNet data/class map, guided
tour or audio implementation. These are explicitly outside this slice. There is
only one immutable replay step. All weights/preprocessing are labelled placeholders;
a structural RF is not a saliency/gradient explanation. Small fitted graphs can be
zoomed or explored through the layer directory and numeric pixel controls.
