# Replay schema v1

The producer is `tools/AlexNetSharp.Snapshot`. It reads the committed, unmodified
145 configs using the ANX-2 parser, infers ANX-3 shapes, and calls the optimized
ANX-6..10 layers. Core remains unchanged and graphics-free. No historical CUDA
execution or trained checkpoint is claimed.

## Files and validation

Each immutable step directory contains `manifest.json` and `activations.bin`.
JSON is UTF-8, camelCase, generated with `ReplayJson` for trimmed WebAssembly.
`ReplaySnapshot.Validate` is the executable schema contract: version, dtype,
layout, mode, graph consistency/acyclicity, shape/counts, contiguous blob offsets,
SHA-256, finite float values and the combined size bound. Unknown schema versions
are rejected. The committed example totals less than 10,000,000 bytes. Neither
weights beyond conv1 nor a batch dimension are downloaded.

Header fields: `schemaVersion=1`, `step=0`, config/params SHA-256, `seed`, labelled
weight recipe, `dtype=float32-le`, `layout=channel,y,x`, `mode=eval`, image provenance,
`layers`, `conv1Filters`, `top5`, `cost`, and informational `extensions`.

Each layer has name, original type, `[C,H,W]` shape (vectors `[outputs,1,1]`, cost
`[0,1,1]`), original device/tower `-1/0/1`, ordered effective inputs, consumers,
neuron placement, parameter-slot count, all parsed parameters as invariant text,
local spatial `kernel/stride/start`, min/max/mean/population standard deviation/count,
and an activation blob descriptor. Cost has no activation output: its zero-length
blob still has a checksum; its three metrics live in `cost`. The synthetic label is
index 0, with no semantic class mapping. Top-5 indices sort by probability descending,
then index ascending. `cost` = negative log likelihood, top-1 error, top-5 error.

Blob descriptors contain `file`, byte `offset`, element `count` and SHA-256 over
exact bytes. Blobs are float32 little-endian, canonical channel/y/x order, no batch
axis. Manifest layer order is parser order (35 declared + 14 detached), independent
of recursive topological execution. The final blob is the conv1 filter grid's
96×3×11×11 values, tower A filters then tower B, filter/channel/y/x order.

## Placeholder choices

Weights: **placeholder, not ANX-11 initialisation**. One SplitMix64 stream, initial
state seed 145, generates two 53-bit open uniforms per Box-Muller cosine sample.
Every weight slot in each allocated backing matrix gets a Gaussian with that
input's configured `initW`; biases get configured `initB`. A fixed recursive
input-first traversal, starting in parser layer order, determines stream ownership.
Separate matrices and the reversed `fc2048bb` order are retained. The exact formula
is in `PlaceholderForward.Gaussian`; this is not NumPy/CUDA RNG parity. Byte identity
is proven across fresh producer invocations on the current .NET runtime/platform;
transcendental implementations may differ on other runtimes/platforms.

Preprocessing: **not the ANX-12 pipeline**. P6 8-bit RGB, short side resized to 256
with nearest-neighbour samples, centre crop 224, subtract constants RGB (123,117,104),
no division by 255. See `content/samples/PROVENANCE.md`: synthetic CC0 image, no
external photograph or licence ambiguity. Image SHA-256 is recorded in the manifest.
The displayed input restores these means from the exact saved input activation.
Dropout evaluation multiplies by keep, without drawing masks. All layers execute
real optimized Core forward operations; the cost is calculated for label 0.

## Extension points

`ISnapshotSource.LoadAsync` supplies the same immutable snapshot to the web view;
`ReplaySource` is the only implemented source. A future run index can append entries
`{step, manifest}` pointing to new directories, without overwriting previous steps.
Trainer iteration is `step`; the schema can add optional scalar curves, per-layer
gradient statistics, additional named filter blob descriptors, checkpoint identifiers
and RNG metadata in a subsequent version. Readers must explicitly support that
version; version 1 rejects unknown versions. A WebSocket adapter can implement the
same source boundary. No run-index writer, trainer, live transport or inference
engine is implemented. `IAudioLayer` is an empty interface for future work.

## Layout and receptive fields

`LayerLayout` and `OrbitCamera` use System.Numerics only. Layers occupy topological
stages 2.15 units apart, towers at y=±2.3, data at the first centre, labels at the
output end, cost at the last stage. Box extents are x=`.8+.025 log2(C)`,
y=`.5+.095 log2(H)`, z=`.35+.08 log2(C)` with each argument clamped to at least 1.
Orthographic projection fits the whole graph; orbit, pan and zoom share the same
matrices as inverse-projection slab-ray picking. Cross-tower edges are amber;
input slots have distinct offsets; cards explicitly show reversed FC fan-in.

Receptive fields trace a selected spatial interval backwards: multiply by stride,
add start, extend by kernel−1, clip to each preceding tensor, union input branches.
Spatial norms expand the interval; cross-map norms and neurons keep its spatial
coordinates. This is the structural potential receptive field, not just max-pool
winners or nonzero ReLU paths. The unclipped nominal range is also shown. Only
image-like conv/pool/norm layers offer pixel views; FC/cost cards remain inspectable.
`pool3a/b` in the original config is the pool after conv5 (often called pool5).

Activation maps are runtime `Texture2D.SetData` atlases, low midnight-blue to high
mint, independent min/max for each channel (constant channel = midpoint). RGB
filters use one min/max across all three channels of each filter. Both use nearest
sampling. Clicking the grid resolves channel and pixel through the same atlas
geometry; a yellow rectangle marks the clipped input range. Numeric controls offer
precise selection when maps are small. Text and cards are HTML; there is no content
pipeline, XNB or MGCB asset.
