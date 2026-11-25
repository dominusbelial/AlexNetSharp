# ANX-2 Report — Config parser: layers-145.cfg + layer-params-145.cfg into a typed, defaulted model

- **Started (UTC):** 2026-11-25T02:01:01Z
- **Base commit:** `4f936a6`
- **Upstream reference:** `computerhistory/AlexNet-Source-Code` (resolved via `ALEXNET_ORIGINAL_SRC`)

## Brief items

| Item | Status | Evidence |
| --- | --- | --- |
| 1. Parser (`src/AlexNetSharp.Core/Config/`) | done | INI reader (SafeConfigParser semantics: lower-cased option names, `%(x)s` interpolation, `[DEFAULT]`, comments, continuations) and typed model `NetworkDefinition`; only the 145 layer/neuron types; detached neuron layers as in `layer.py`; no shape inference. Build 0 warnings / 0 errors. |
| 2. Golden parse (`tests/fixtures/layers-145.parsed.json`) | done | Fixture written from both 145 files; `GoldenParseTests` re-parses and compares. Hand comparison below. |
| 3. Error tests | done | `ConfigParserErrorTests`: 13 tests covering all 11 rules; messages match `layer.py` wording (e.g. `:102` required parameter, `:289` no type given, `:305` is unused). |
| 4. Spec link (`tests/AlexNetSharp.Tests`) | done | `SpecLinkChecks`: 569 emitted parameters all found in their ledger entries, no ledger gaps; the ANX-1 spec check still passes. |
| 5. Report | done | Completed by Claude at review: the lane run stopped at its $6 budget before updating this table. |

## Item timings

| Item | Duration |
| --- | --- |
| Setup & spec/source inspection | ~32 min (02:01–02:33 UTC) |
| 1. Parser (`src/AlexNetSharp.Core/Config/`) | ~34 min (02:33–03:07 UTC, implementation complete) |
| 2–4. Golden, error tests, spec link | ~1 h 10 min (03:07–04:41 UTC; budget stop) |

## Hand comparison (Claude review)

Every parameter of all 35 ledger sections (569 values, including conv1a, conv3a, rnorm2a, cnorm2a, hs1a, fc2048bb,
fc1000 and logprob) was compared with the fixture, together with the effective input order. All match. The only
differences from the ledger's configured `value` are the values the original parser stores, which the ledger records
as `parser_stored`: normalisation `scale` divided by `size` (cmrnorm) or `size²` (rnorm) (`layer.py:1179`), and
conv `padding` negated (`layer.py:1041`). The fixture uses the stored values. Inputs follow the detached-neuron
rewrite (e.g. `rnorm2a` reads `conv2a_neuron`); `fc2048bb` keeps `hs1b,hs1a`.

## Test results

`SPEC CHECK PASSED: 35 sections, 10 layer types, 15 weighted layers, 569 parser params linked, 445 total citations
verified.` and `CONFIG PARSER TESTS PASSED: 3 golden/smoke tests + 13 error/INI rule tests (16 total)`, 3× in fresh
processes (Claude, at review).

## Cost

Gemini 4 Carbon, $6 cap: budget stop at guarded $6.03 (provider-reported $4.16) after 77 calls.
