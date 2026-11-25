# ANX-1 Report — Specification ledger, experiment manifest and cudaconv2 version evidence

- **Started (UTC):** 2026-11-24T23:32:15Z (ANX-1B resumed 2026-11-25T00:56:20Z)
- **Base commit:** `7d37a67`
- **Upstream reference:** `computerhistory/AlexNet-Source-Code` commit `2f8b81f` (resolved via `ALEXNET_ORIGINAL_SRC`)

## Brief items

| Item | Status | Evidence |
| --- | --- | --- |
| 1. Ledger (`docs/spec/ledger.json`, `docs/spec/LEDGER.md`) | done | Authored all 35 `[section]` entries, 10 `layer_types`, `relu` neuron type, 14 detached `_neuron` layers, faithfulness rules, optimizer rule, and 4 open questions; 348 citations verified against `ALEXNET_ORIGINAL_SRC` |
| 2. Experiment manifest (`docs/spec/experiment.json`, `docs/spec/EXPERIMENT.md`) | done | Written by Claude after the Carbon budget stop: recorded rates for all 15 weighted layers, the three logged runs with every transition and result cited, data options, proposed fresh run (run1 replay, literal/plateau variants), 7 uncertainties; 97 citations verified |
| 3. cudaconv2 version evidence (`docs/spec/CUDACONV2.md`) | done | Fetched both candidates read-only into `.scratch/cudaconv2-candidates/`; recorded URLs, SHA-256 (`b55bd30c...`), commit (`3238bf03...`), 33-function comparison table, and verified kernel semantics in `docs/spec/CUDACONV2.md` |
| 4. Machine check (`tests/AlexNetSharp.Tests`) | done | Checker (ANX-1B) covers sections, types, citations+tokens, per-layer rate coverage; `SPEC CHECK PASSED: 35 sections, 10 layer types, 15 weighted layers, 445 total citations verified.` 3× in fresh processes; failure proof below |
| 5. Report (`docs/reports/ANX-1_REPORT.md`) | done | This file |

## Item timings

| Item | Duration |
| --- | --- |
| Setup & initial inspection | ~5 min (23:32–23:37 UTC) |
| 1. Ledger | ~18 min (00:56–01:14 UTC) |
| 2. Experiment manifest | ~25 min (Claude, after the budget stop) |
| 3. cudaconv2 version evidence | ~8 min (23:37–23:45 UTC) |
| 4. Machine check & verification gates | checker by ANX-1B; gate + failure proof ~10 min (Claude) |

## Gate

- `dotnet build`: 0 warnings / 0 errors.
- Spec test 3× in fresh processes: passed (445 citations).
- Nothing outside the repo changed; the original tree is untouched.

## Failure proof

A copy of the ledger with `rnorm2b` removed and one citation moved by a line (`--ledger <tmp>/bad-ledger.json`), exit 1:

```
SPEC CHECK FAILED (2 error(s)):
  - ledger.faithfulness_rules.unscaled_hs_dropout.citations[0] (src/layer.cu:1122): token "HiddenSexLayer::HiddenSexLayer" not found on line 1122 (actual: "_enable = pyDictGetInt(paramsDict, "enable");").
  - ledger.sections: missing entry for layers-145.cfg section '[rnorm2b]'.
```

While writing the manifest it also caught three wrong line numbers in the first draft.

## Cost and history

- ANX-1 (Gemini Carbon): item 3 and the report skeleton; stopped when one reply hit the output-token limit. About $1.9.
- ANX-1B (Carbon, continued from the WIP with small writes): items 1 and the checker; stopped by the $3.70 budget
  watchdog. About $2.6.
- Claude: item 2, gate, failure proof, report.

## Open questions

Ledger (`docs/spec/ledger.json`): OQ-LEDGER-1 crop width 224 vs 227; OQ-LEDGER-2 missing cudaconv2/nvmatrix sources
(see `CUDACONV2.md`); OQ-LEDGER-3 frozen conv1 rates vs gradient-consumer status; OQ-LEDGER-4 RNG streams and CUDA
summation order. Manifest (`docs/spec/experiment.json`): OQ-EXP-1 … OQ-EXP-7 (starting rates, pauses, views of the run1
result, missing framework modules, dataset preparation, "2012-full", `run4.sh`).
