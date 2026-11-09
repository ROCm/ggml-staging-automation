# Benchmarking

By default, CI runs the packaged HRX suites before benchmarks in the same GPU
job and stops if a suite fails. The [CI validation guide](../ci/README.md)
describes tests-only and benchmarks-only execution. These workflow modes are
separate from the per-model `test-modes` described below.

When benchmarks run, PR CI defaults to the `single` tier defined in the model
manifest. The cumulative `small` tier covers
Lemonade's smaller-model suite; `all` covers every ordinary model. Automated submodule
bumps use `all`.

Apply `ci:single-model`, `ci:small-models`, or `ci:all-models` to start CI with
that coverage; the highest selected tier wins. Manual CI runs also offer all
three tiers. Models in `select-only` are excluded from every cumulative tier,
including `all`, and cannot be enabled by a coverage label. To run them, supply
a JSON array of exact manifest IDs in the manual CI `model_ids` input (or the
runner's `--model-ids` option). A non-empty list replaces tier coverage; an empty
list keeps the selected tier. Unknown or duplicate IDs fail before downloads.
For example, `--model-ids '["flint-q3-k-m","flint-iq2-xxs"]'` selects only those
two models. The `select-only` tier itself cannot be requested as bulk coverage.

The runner and workflow use a 60 GiB disk budget (64,424,509,440 bytes), with a
2 GiB reserve and a further cap at available disk space. This is a binary GiB
budget, not decimal GB; the `--max-disk-gib` option retains the existing units.

## Lemonade release and test modes

The [release pin](../../../benchmarks/hrx/lemonade_release.json) selects a
checksum-verified released Lemonade package. CI benchmarks the same verified
GGUF through the HRX and regular llama.cpp Vulkan recipes using the CI-built
server executable. The [model manifest](../../../benchmarks/hrx/model_manifest.json)
selects weights, tiers, test modes and expectations; `name` also names the local
model directory.

See [run_lemonade_benchmark.py](run_lemonade_benchmark.py) for the setup mechanisms:
its function docstrings explain private daemon state, local-path registrations,
recipe options, executable overrides and runtime verification.

`test-modes` maps benchmark IDs to independent cases. A benchmark omitted from
that mapping uses `["default"]` on both backends. Setting
`"lemonade-benchmark": ["mtp:off", "mtp:on"]` produces two comparisons:
HRX/off versus Vulkan/off and HRX/on versus Vulkan/on. Off supplies
`--spec-type none`; on uses Lemonade's automatic MTP defaults. Both use normal
EOS termination, 4096-token contexts, the existing short/long-output scenarios,
one warmup and three measured runs. The model stays loaded within each scenario
so measured runs reuse kernels compiled during warmup; server prompt caching is
disabled so each run processes the full prompt. The next scenario loads a fresh
model. Perplexity remains one comparison per model, with both existing
prefill-like and decode-like regimes; it does not exercise MTP.

Speculative counters and verification of actual drafting are deferred. A passing
on-case establishes successful generation with Lemonade defaults, not that MTP
was enabled. Reports label these as requested settings.

Exceptions are independent for each model, check, backend and mode. For example:

```json
{
  "test-modes": {
    "lemonade-benchmark": ["mtp:off", "mtp:on"],
    "perplexity": ["default"]
  },
  "expected_results": {
    "lemonade-benchmark": {
      "hrx": {"mtp:on": "report-only", "mtp:off": "report-only"}
    },
    "perplexity": {
      "hrx": {"default": "fail"}
    }
  }
}
```

Omitted expectations mean `pass`. Use `default` for ordinary models and
perplexity, and explicit `mtp:off`/`mtp:on` keys for expanded Lemonade cases. `fail`
requires failure (a success is a CI-failing XPASS); `report-only` retains the
measured outcome without gating. No exception is inherited by the other mode
or backend. Unsupported benchmark/mode combinations, duplicate modes and
expectations for unselected modes are rejected. Modes are independent cases,
not a Cartesian product. Perplexity currently supports only `default`.
Infrastructure and malformed-artifact errors remain fatal.

The pinned Lemonade release predates automatic HRX MTP enablement. Updating the
release pin to one supporting HRX MTP needs no special version-dependent
enablement code. Qwen3.5 generation and perplexity use the default pass
expectation. Backend correctness fixes must also be present in the CI-produced
llama-server.

## Architecture-gallery batch 1

All ten variants from [batch 1](https://github.com/ROCm/ggml-staging-automation/issues/129)
are in the ordinary `all` tier, including the models that needed no backend
changes. Their immutable Hugging Face revisions, filenames, sizes and SHA-256
checksums are in the model manifest. Generation requests `mtp:off` because the
survey qualified ordinary text inference, not speculative decoding or vision.

| Model ID | Quantization | HRX expectation |
|---|---|---|
| `ornith-1.5-9b` | Q4_K_M | Pass |
| `ornith-1.5-35b-a3b` | Q4_K_M | Pass |
| `qwen3.5-9b` | UD-Q4_K_XL | Pass |
| `gemma-4-e4b-it` | Q4_K_M | Pass |
| `gemma-4-e2b-it` | Q4_K_M | Pass |
| `qwen3-coder-30b-a3b-instruct` | Q4_K_M | Pass |
| `qwen3.6-35b-a3b` | UD-Q3_K_M | Pass |
| `mellum2.1-12b-a2.5b-thinking` | Q4_K_M | Pass |
| `gemma-4-26b-a4b-it` | UD-Q4_K_M | Pass |
| `muse-glimmer-30b` | UD-Q4_K_XL | Report only |

Muse's HRX generation and perplexity remain report-only while the
[weight-residency work](https://github.com/ROCm/ggml-staging-automation/issues/122)
following [llama.cpp PR #160](https://github.com/AMD-Ecosystem/llama.cpp/pull/160)
lands. Survey bead `lemonade-ws-qiur` records a 24 GiB failure and a gfx1151
pass with 29.69 GB of resident weight-cache allocations; `lemonade-ws-p6dm`
tracks the remaining layout work. Both success and failure are reported without
requiring a failure. Vulkan stays gating, as do models without a current failure
exception. Existing Qwen3.8-27B and Mistral-Small-3.2-24B exceptions are retained.

The **Batch 1 model coverage** workflow builds the checked-in source revisions
and runs these ten models on the existing gfx1151 runner through the normal
generation and perplexity jobs. It runs on pushes to
`users/astgeorg/batch1-model-ci` and supports manual dispatch. Artifacts include
both backends' results, debug logs, and exact source revisions. This focused run
uses the existing CI profiles: 4096-token generation contexts and the
512-token prefill-like/decode-like perplexity regimes. It does not repeat the
survey's full-corpus context-2048 qualification.

The survey's Gemma variants have unusually high absolute perplexity on both
backends. CI preserves the measurements and applies the normal relative
HRX/Vulkan check; passing that check does not establish expected model quality.
The backend pin includes the merged Gemma scale fusion from PR #183, which
removed the scale REPEAT CPU fallback recorded in the earlier survey.
