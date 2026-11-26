# AlexNetSharp

The real 2012 AlexNet (the ImageNet ILSVRC-2012 winner), reimplemented in C# from the original source release and
made explorable in the browser.

The specification is the original code itself: `layers-145.cfg` / `layer-params-145.cfg` and the cuda-convnet sources
published by the Computer History Museum ([computerhistory/AlexNet-Source-Code](https://github.com/computerhistory/AlexNet-Source-Code)).
Every behaviour is cited to the original `file:line`, and differences from the paper are kept and shown, never
"corrected". See [DESIGN.md](DESIGN.md) for the goal, the faithfulness rules and the story plan.

## Status

Early. Done so far:

| Story | What | Report |
|---|---|---|
| ANX-1 | Specification ledger (35 layer sections, 445 machine-checked citations), experiment manifest, cuda-convnet version evidence | [ANX-1](docs/reports/ANX-1_REPORT.md) |
| ANX-2 | Config parser with the original's defaults and error messages; golden parse of the 145 network | [ANX-2](docs/reports/ANX-2_REPORT.md) |
| ANX-3 | Shape inference with the original's ceiling arithmetic (224 → 55 with a partial boundary window) | [ANX-3](docs/reports/ANX-3_REPORT.md) |
| ANX-4 | Tensor layouts, parameter buffers (60,965,224 parameters) and the original's overwrite-vs-accumulate gradient semantics | [ANX-4](docs/reports/ANX-4_REPORT.md) |
| ANX-5 | Scalar reference convolution (forward, input/weight/bias gradients, `partialSum`), exact-rational fixtures, 3,824 finite-difference checks, conv1's partial edge window proven on real pixels | [ANX-5](docs/reports/ANX-5_REPORT.md) |
| ANX-6 | Managed SIMD CPU convolution, bitwise scalar parity and thread determinism, bounded workspace, batch-128 benchmark | [ANX-6](docs/reports/ANX-6_REPORT.md) |
| ANX-7 | Scalar and SIMD max/average pooling and ReLU, edge/tie/NaN fixtures, numerical gradients and batch-128 timings | [ANX-7](docs/reports/ANX-7_REPORT.md) |
| ANX-8 | Scalar and SIMD spatial/cross-map response normalisation, cached-denominator gradients, independent float64 fixtures, numerical checks and batch-128 timings | [ANX-8](docs/reports/ANX-8_REPORT.md) |
| ANX-9 | Scalar and SIMD multi-input fully connected layers, ordered bias/fan-in, alpha/beta gradients, fan-out and batch-128 timings | [ANX-9](docs/reports/ANX-9_REPORT.md) |

| ANX-10 | Unscaled dropout with injectable deterministic masks, stable softmax, fused/general loss gradients, fractional top-k ties and multiview cost | [ANX-10](docs/reports/ANX-10_REPORT.md) |

Next: optimiser/initialisation, data pipeline, trainer, and web explorer.

## Layout

- `src/AlexNetSharp.Core` — parser, shapes, (later) layers, optimiser, data pipeline; no graphics dependency.
- `tests/AlexNetSharp.Tests` — spec checker, parser/shape tests and fixtures.
- `docs/spec` — the ledger and experiment manifest; `docs/reports` — one report per story.

## Running the checks

The tests read the original release from a local checkout. Point `ALEXNET_ORIGINAL_SRC` at it (environment variable,
or a `local.env` file in the repo root containing `ALEXNET_ORIGINAL_SRC=<path>`), then:

```sh
dotnet build
dotnet run --project tests/AlexNetSharp.Tests
```

The default Debug suite runs the exact fixtures, numerical checks and contracts on both convolution engines,
randomized bitwise parity/determinism, and optimized real-size checks. Full-size scalar assertions and elementwise
parity for all ten 145 convolution layers are opt-in:

```sh
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --full-scalar-parity
```

The convolution benchmark is separate from the default gate. It measures all ten layers at batch 128, compares
scalar and optimized operations at batch 2, and prints training-run projections:

```sh
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-conv
# Optional: --conv-threads 2 --conv-repeats 3 --conv-output .scratch/conv-benchmark.json
```

Pooling and ReLU fixtures, numerical gradients, thread determinism and all six 145 pool
shapes run in the default suite. The full mode additionally repeats real pool parity
at every tested thread count. Batch-128 pool timings (including scalar comparisons
and the four post-pool ReLUs) run separately:

```sh
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-pool
# Optional: --pool-output .scratch/pool-benchmark.json
```

Normalisation checks include both kernels, clamped odd/even windows, blocked
cross-map groups, special values, numerical gradients and all six 145 geometries
at N=3. Full scalar parity additionally checks N=17 (SIMD lanes and tail) at
1, 2 and all worker counts. Batch-128 timings are informational and opt-in:

```sh
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-norm
# Optional: --norm-output .scratch/norm-benchmark.json
```

Fully connected checks include independent one/two-input fixtures, reversed input
order, alpha/beta gradients, NaN handling, numerical differences, thread determinism
and all five 145 shapes at N=3. Default runs optimized kernels with sampled scalar
checks; `--full-scalar-parity` compares every element at N=3 and N=9 with
1/2/all/repeated-all workers. Both modes verify actual pool3/hs1 fan-out and the
separate FC ReLU placement. Batch-128 timing is opt-in:

```sh
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-fc
# Optional: --fc-threads 2 --fc-repeats 3 --fc-output .scratch/fc-benchmark.json
python3 tests/fixtures/make_fc_fixture.py
```

Dropout, softmax and logistic-regression cost checks include independent fixtures,
float64 gradients, RNG and worker-count determinism, all four 2048-wide dropout
layers and 1000-class batch-128 softmax/fused gradients/top-5 cost. Dropout uses
unscaled training masks and `* keep` evaluation, including disabled training.
The port RNG does not reproduce the original CUDA stream. The full parity mode
also runs batch-128 general softmax backward and actual FC-pair fan-out.

```sh
dotnet run -c Release --project tests/AlexNetSharp.Tests -- --benchmark-dropout-softmax-cost
# Optional: --dsc-threads 2 --dsc-output .scratch/dropout-softmax-cost-benchmark.json
python3 tests/fixtures/make_dropout_softmax_cost_fixture.py
```

ImageNet data is not part of this repository.

## License

MIT for this repository's code and documents (see [LICENSE](LICENSE)). The original AlexNet source release is
© 2023 Google LLC under its own BSD-style license; this repository does not copy it, it cites it.
