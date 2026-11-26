# VIZ-2 — Photo replay, perspective stack and shared desktop renderer

Started 2026-11-26 at approximately 02:57 UTC (2026-11-25 local date), base `109ea17`.
This report was created before implementation and updated during the work.
All seven items and required gates passed. Changes remain in the working tree;
no commits, staging, pushes or stash operations.

## Brief items

| Item | Status | Evidence | Elapsed time |
| --- | --- | --- | --- |
| 1. Real photo | done | CC0 Labrador photo, 144,377 bytes; provenance/hash test; managed producer; regenerated manifest; deterministic/direct-Core proof and clean regeneration passed. | ~14 min, 02:57–03:11 UTC |
| 2. Real 3D view | done | Perspective and face shading in both hosts; all 49 default centre picks exact; all corners fit; moved-camera, reset and limits checks; actual browser/desktop GL pixels passed. | ~13 min, 02:58–03:11 UTC |
| 3. Phone layout | done | Three-column vertical snake, content-sized canvas, persistent tap-to-name; 390×844 proof: 64.18% width, no overflow, real touch orbit/pinch and outside page scroll passed. | ~10 min, 03:02–03:12 UTC |
| 4. Shared viz and desktop | done | Platform-neutral Viz project; Web and SDL2 Desktop both use it. In-canvas Markdown cards/font, activations, filters, RF and hover shown. Linux execution and win-x64 publish passed. | ~16 min, 02:59–03:15 UTC |
| 5. Desktop proof | done | Opt-in clean-state driver; six PNGs and JSON receipt; offscreen attempted, display fallback exit 0 in 1.369 s; text/RF/stack/filter pixels asserted. | ~14 min, 03:01–03:15 UTC |
| 6. Tests | done | 706 default tests + spec checks, three fresh processes <12.3 s; 733 full Release; snapshot/direct comparison; absent-payload test branch and clean solution generation passed. | ~12 min, 03:02–03:14 UTC |
| 7. Report and gates | done | Build/test/visual receipts, four total bounded screenshots, provenance, sizes/timing, integrity and limitations below. | ~19 min, throughout |

Intervals are approximate, overlap and include diagnosis/verification; they are not additive.
The final desktop timing instrumentation was also rebuilt and exercised by the clean desktop proof.

## Photo and snapshot

The sample is [Portrait of a labrador retriever](https://commons.wikimedia.org/wiki/File:Portrait_of_a_labrador_retriever.jpg),
by **Dktue**, photographed 20 July 2019, explicitly licensed **CC0-1.0** on Commons.
The unmodified 960×640 thumbnail is 144,377 bytes, below 200,000 bytes.
[PROVENANCE.md](../../content/samples/PROVENANCE.md) records the source/download URLs,
author, licence link, transformation and SHA-256:

`8e04f295a5a2e907dfe9924b5e2dbee594c0942e8ecca7556fd8d64aa1818777`

Labrador retriever is an ImageNet subject, but **these weights are untrained and the
shown top-5 predictions are meaningless**. The label remains synthetic index 0;
cost is not an accuracy measurement for this dog. Both hosts say the predictions
are meaningless. The synthetic image is removed because no test needs it.

The console producer alone references **StbImageSharp 2.30.16**, a managed C#
MIT/public-domain decoder, with no native decoder or ImageSharp dependency. The
placeholder forward implementation moved from Replay into that console assembly.
The published browser proof explicitly checks that neither the decoder nor the
producer assembly is shipped in `_framework`. Both hosts reconstruct the exact
224² crop from stored input floats; neither decodes the JPEG at runtime.

Preprocessing remains explicitly **not ANX-12**: nearest-neighbour resize of the
short side to 256, centre crop 224, subtract RGB constants (123,117,104), no /255.
Weights remain the VIZ-1 SplitMix64/Box-Muller placeholders, not ANX-11 initialization.
No Core implementation, fixture or earlier numerical specification changed.

The generated payload remains ignored/untracked. Shared `tools/GenerateReplay.targets`
lets either host build regenerate it. The producer compares its freshly generated
manifest byte for byte with the tracked manifest **before writing**; an intentional
recipe/photo change requires `--update-manifest`. Generation is serialized across
platform targets. Tests independently enforce the same pinned-manifest comparison.

| File | Raw bytes | SHA-256 |
| --- | ---: | --- |
| manifest.json | 60,809 | `24c6f87cf1f16f5f89a1ba81fd560b6b8b556e27a75babb5ed3e805f37c5f53b` |
| activations.bin | 8,175,044 | `d873f8203592228b5fbe3d2df5c4bd2f4393afebb5a17ecbe0b6befac6a2a22e` |

Combined raw snapshot: **8,235,853 bytes**, below 10,000,000. Two fresh producers
and repeated in-process productions matched these bytes. All 49 blobs, conv1
filters and cost match direct optimized Core execution at two workers.

## Camera, layout and touch

The perspective vertical field of view is **.76 radians** (~43.5°). The default
landscape camera uses yaw **−.28**, pitch **−.55** radians, with an aspect-fitted
distance and padding that keeps every box corner inside the default view. Distinct
face brightness (.42–1.0) makes box fronts, sides and tops read as solids.
Hover is warm cream, selection mint, ReLUs purple, tower B blue; amber fan-in edges
and reversed input slots remain visible. Reset restores angle, target and zoom.

World box extents explicitly encode **width, height and channels**:

- X = `.45 + .075 log₂(W)`;
- Y = `.45 + .12 log₂(H)`;
- Z = `.4 + .2 log₂(C)`.

Each log argument is clamped to at least 1. This logarithmic display scale preserves
visibility of vectors and large channel counts; it is not a linear tensor volume.
Landscape topological spacing remains 2.15 units, towers at y=±2.3. The scale is
stated in the Web hint, desktop footer and [schema/layout documentation](../spec/REPLAY.md).

Yaw is bounded to ±1.05, pitch to [−1.05,.8], zoom to [.6,4]; pan is bounded to the
scene extent. Mouse/touch movement changes the camera continuously, without snaps.
Rendering and picking use identical perspective matrices; picking inverts them,
unprojects a ray and finds the nearest positive axis-aligned box intersection.
Tests cover all 49 exact centre picks in both default layouts, known world rays,
nonoverlap, all projected corners, moved-camera picking, perspective, reset and limits.

At scene widths ≤600 px, topological stages snake through **three columns**,
spaced 6.2 units horizontally and 3.3 vertically, with towers side by side. The
portrait camera defaults to yaw −.16, pitch −.24 with a 32-unit fitted extent.
Canvas height follows its width: approximately **1.22 × width + 210 px** for maps
and photo, rounded to pixels. At 390 px, the actual canvas is 362×652 and the stack
occupies 442 px vertically. A readable, persistent **“Tap a box · name”** strip
replaces tiny projected labels; tapping also pins its full explanation below.
Landscape labels and layout remain available.

One finger orbits; two fingers pinch zoom. Pointer capture and cancellation handle
finger lifecycle; only the canvas has `touch-action:none`. KNI's legacy window
`touchstart/move/end` listeners unconditionally cancel events, including
noncancelable touchend events. Because this host uses pointer events, passive
listeners stop their bubble path at the document, leaving browser default scrolling
available. The proof asserts both camera changes without canvas-induced page scroll
and an actual successful touch scroll **outside** the canvas, with no console errors.

## Project structure and desktop UI

| Project | Responsibility |
| --- | --- |
| Core | Existing graphics-free parser/shapes/layer mathematics; unchanged. |
| Replay | Graphics-free snapshot format/validation, RF tracing, Markdown card parser and source interface. |
| Viz | Shared KNI `StackGame`, face shading, geometry, perspective camera, picking, keyboard mappings, atlas/pixel geometry, texture creation and card plain-text conversion. Only `nkast.Xna.Framework*` **4.3.9001** packages; no platform package. |
| Web | BlazorGL **4.3.9001**, HTTP replay source, browser pointer/touch glue and HTML card/label UI. |
| Desktop | net10.0 SDL2/GL **4.3.9001** host, native mouse/keyboard input and in-canvas text/cards with `FontStashSharp.Kni` **1.6.2**. |
| Snapshot tool | JPEG decoding, placeholder forward and snapshot production; build-time only for either host. |

Desktop accepts an optional replay-directory argument, defaults to Web's
`wwwroot/replay` in the repository, and loads the same validated bytes from disk.
Its build generates a missing default payload, like Web. There is no trainer or
inference in either UI. Published standalone copies can use the explicit path
argument (or put the replay in a `replay` directory next to the executable).

The desktop draws the shared stack, activation atlas, 96 RGB conv1 filters, selected
pixel and RF outline. Hover shows the name and summary; click pins independent
cards. The panel reads the same `content/explain/*.md` files, including live shape,
parameter count, statistics, body, formula, related links and citations. It scrolls,
links pin related cards, x closes a card, Tab toggles deeper text, F toggles filters;
the camera mouse gestures and arrow/WASD/+−/R keys match Web.

Text uses bundled, unmodified **DejaVu Sans**, 759,720 bytes, through FontStashSharp.
Its permissive Bitstream Vera licence, public-domain DejaVu changes, notices,
source and file hash are in [content/fonts/LICENSE.txt](../../content/fonts/LICENSE.txt).
There is exactly one font file. No `.xnb`, MGCB or content pipeline is used.

## Verification

`tools/viz2-gates.py` puts NuGet packages/HTTP cache, CLI, temporary and XDG state
under `.scratch/`. Both whole-solution Debug and Release builds passed with
**0 warnings / 0 errors**. Final builds also passed after the desktop first-frame
measurement was added. The three default runs are fresh OS processes; no test
process reuse or exclusions were used.

| Gate | Process time | Result |
| --- | ---: | --- |
| Solution Debug | 3.177 s | 0 warnings / 0 errors |
| Solution Release | 3.339 s | 0 warnings / 0 errors |
| Default Debug 1 | 12.236 s | 706 tests + spec checks |
| Default Debug 2 | 12.052 s | 706 tests + spec checks |
| Default Debug 3 | 12.028 s | 706 tests + spec checks |
| Full scalar parity Release | 42.572 s | 733 tests + spec checks |
| Snapshot determinism/direct-Core | 5.285 s | Passed |
| Fresh producer 1 / 2 | 2.027 / 2.036 s | Identical manifest/payload |
| Test absent-payload regeneration | 3.880 s | 23 VIZ groups, 184 citations, pinned manifest preserved |
| Whole solution with payload deleted and Web obj/bin removed | 10.946 s | Both platform generation targets; 0 warnings/errors; pinned manifest preserved |

The new default tests check camera/layout/limits/picking, source photo CC0/PD licence
line and SHA-256, file bounds, shared card link labels, bundled font licence/hash,
and the earlier replay/RF/card/atlas checks. The default VIZ slice is ~.19 s when
the payload exists. Core's 569 parameter links, 1,470 shape fields and 455 original
citations remain checked, alongside 184 explain-card citations.

Receipts: [build/test gates](VIZ-2_assets/gates.json),
[determinism](VIZ-2_assets/determinism.json),
[clean regeneration](VIZ-2_assets/clean-regeneration.json).
Detailed logs remain under ignored `.scratch/viz2`.

## Clean browser proof

`node tools/viz1-browser-proof.cjs --clean` passed after deleting `activations.bin`
and Web `obj/bin`, then making a clean trimmed Release publish. It checked the
pinned manifest, decoder/producer exclusion, actual GL pixels, all 49 projected
positions, exact hover/clicks, two coexisting cards, deeper values, atlas pixel
(0,3,4), numeric edge selection x=216..223, selected box after reset, filters,
keyboard motion, no horizontal overflow, and mobile touch behavior.

| Viewport | Cold first frame | Stack width / canvas | Activation mint pixels | RF yellow pixels | Filter colors | Console errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1280×800 | 1,125.4 ms | 73.27% | 59,310 | 52 | 11,586 | 0 |
| 390×844 | 1,070.6 ms | **64.18%** | 23,416 | 20 | 9,819 | 0 |

Cold first-frame time is browser navigation to the first real KNI draw on loopback,
with a fresh context and headless software WebGL. It is not an Internet download
estimate. [Browser receipt](VIZ-2_assets/browser-proof.json).

## Clean desktop proof

`python3 tools/viz2-desktop-proof.py` deletes the generated payload and Web obj/bin,
builds Desktop Release, then launches `--proof <out-dir>`.
`SDL_VIDEODRIVER=offscreen` was tried first and failed in 0.216 s inside KNI's
`OGL.GetString`/GL-extension initialization. **Offscreen GL is unavailable through
this backend on this machine.** The driver then used the real display with an
**1100×720** window for **1.369 s**, well below 30 s, and exited **0**.
The display log contains Mesa driver-fallback diagnostics; the rendered output
and all pixel assertions passed. No hardware-acceleration claim is made.

The fixed sequence captures **default, hover conv1a, pinned card, pixel (0,3,4),
orbit**, plus **filters**. Six original PNGs and a JSON receipt are written under
`.scratch/viz2/desktop`. The first complete desktop frame was **348.8 ms** after
process start, including runtime startup, disk snapshot load/validation, textures
and font setup. This is a local measurement, not a general performance promise.

Each frame has >369,000 non-background pixels and ≥7,862 colored stack pixels;
the pinned-card frames have **728 actual glyph pixels** in the summary text region.
All activation frames contain **56 yellow RF-outline pixels within the photo**.
The filter frame contains **11,586 distinct colors**. Hover resolves `conv1a`,
selection is (channel=0,x=3,y=4), and the orbit camera differs from default.
The driver asserts these values/thresholds and exits only after all six frames.
[Desktop proof receipt](VIZ-2_assets/desktop-proof.json).

## Published sizes and screenshots

Physical file totals include standard compressed variants/debug/runtime files for
Web, and copied dependency/native files/content for Desktop. Desktop publishes are
framework-dependent; they exclude the installed .NET runtime and the separately
shared 8,235,853-byte raw replay. Windows publishing was verified on Linux;
Windows execution was not available on this machine.

| Output | Physical bytes |
| --- | ---: |
| Whole clean Web publish | 28,105,742 |
| Web wwwroot | 27,087,116 |
| Web published replay (including compressed manifests) | 8,248,986 |
| Desktop portable publish | 16,308,058 |
| Desktop win-x64 publish | 5,887,310 |

[Desktop size receipt](VIZ-2_assets/desktop-published-sizes.json).

Three current report screenshots, each <400,000 bytes:

- [Web desktop](VIZ-2_assets/web-desktop.jpg): 230,080 bytes.
- [Web phone](VIZ-2_assets/web-phone.jpg): 194,312 bytes.
- [Desktop app](VIZ-2_assets/desktop-app.jpg): 140,168 bytes.

Web captures are full-page JPEGs from the proof; viewport dimensions remain those
in the table. The desktop report JPEG is a compressed copy of the actual proof PNG.
One historical [VIZ-1 desktop baseline](VIZ-1_assets/desktop.png), 356,326 bytes,
remains for comparison. The old VIZ-1 mobile PNG was retired and its report link
updated, keeping **four screenshots total in the repository**. The only other
media are the one CC0 photo and one permissively licensed font.

## Integrity, reproduction and remaining limits

All **274 original-source files** retain the same relative names, sizes,
nanosecond mtimes and SHA-256 as the pre-work baseline. Core, earlier fixtures,
ledger and experiment manifest are unchanged. No candidate source was downloaded
or added. The git index is unchanged; the replay payload remains ignored/untracked.
Authored text has LF endings, and no private home paths, machine identifiers,
IP literals or secrets were added. NuGet/npm/CLI/browser temporary state stayed under `.scratch/`; normal build
outputs stayed under their repository projects. No file outside this repository
was edited.
`git diff --check` passes. Only child PIDs started by proof scripts are stopped;
no broad process-killing commands were used.

```sh
python3 tools/viz2-gates.py
npm --cache .scratch/npm-cache install --prefix .scratch/npm playwright
node tools/viz1-browser-proof.cjs --clean
python3 tools/viz2-desktop-proof.py
```

For interactive runs, use `dotnet run --project src/AlexNetSharp.Web` or
`dotnet run --project src/AlexNetSharp.Desktop` with writable NuGet/CLI caches
configured under scratch as in the runners. Both hosts regenerate a missing default
payload. See README for path arguments and controls.

Known limits: Windows was cross-published but not executed; SDL offscreen GL fails
locally and the supported display fallback is proven. Perspective can occlude boxes
at user-chosen orbit angles; nearest-ray picking remains correct. This remains one
untrained replay with placeholder preprocessing and a synthetic label, not training,
a trained classifier, live source, trainer, in-browser inference, guided tour or
audio. Structural RF bounds are potential contributors, not saliency. These are
explicit scope/validation limits; **no brief item is unfinished**.
