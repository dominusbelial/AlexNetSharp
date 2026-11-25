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

Next: tensor layouts, then each layer type against hand-computed goldens, the trainer, and the web explorer.

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

ImageNet data is not part of this repository.

## License

MIT for this repository's code and documents (see [LICENSE](LICENSE)). The original AlexNet source release is
© 2023 Google LLC under its own BSD-style license; this repository does not copy it, it cites it.
