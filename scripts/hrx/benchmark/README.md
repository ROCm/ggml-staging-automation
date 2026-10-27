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
HRX/off versus Vulkan/off and HRX/on versus Vulkan/on. Off supplies `--spec-type none`; on supplies no MTP
flag and uses Lemonade's automatic defaults. Both use normal EOS termination,
4096-token contexts, the existing short/long-output scenarios, one warmup and
three measured runs. Perplexity remains one comparison per model, with both
existing prefill-like and decode-like regimes; it does not exercise MTP.

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
