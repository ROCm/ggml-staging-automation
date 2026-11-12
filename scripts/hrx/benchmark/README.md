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

Long-context work is opt-in through `include_long_context` on the CI workflow
or `--include-long-context` on `run_batched_benchmark.py`. This runs two
additional checks without changing the short perplexity cases: streaming
teacher-forced perplexity at 32K, 64K, 128K and 256K tokens, and a separate
`llama-bench` prefill/decode sweep at those contexts. The PPL evaluator scores
the final 1024 corpus continuation tokens after an exact `N - 1024` prefix,
computing full-vocabulary likelihood one token at a time. Contexts above a
GGUF's declared training context retain an unsupported diagnostic and count as
failed checks, as do contexts that exhaust the selected device. An expected
failure must be declared for that check, backend and context; otherwise it
fails CI. The evaluator records model-context, model-execution and
resource-exhaustion failures separately from infrastructure errors. Only the
first three can match an explicit expected failure; unavailable devices,
invalid artifacts and incomplete comparisons remain fatal, including for
report-only models.

The context sweep runs `pp N` and `tg 256 @ d (N - 256)` as separate
measurements. llama-bench's depth is prefetched outside the timed decode
section, so the decode result starts with populated KV. It uses the default
workload warmup and three measured repetitions. `--contexts`,
`--decode-tokens`, and `--repetitions` on `run_context_sweep.py` allow local
overrides; `--context-lengths` and `--scored-tokens` configure the PPL worker.
Raw commands, logs, model/corpus provenance, and llama-bench output fields are
kept in the JSON artifact and backend logs.

Long-check expectations use their own `long-context-perplexity` and
`context-sweep` keys; exceptions configured for short `perplexity` do not
carry over. Use a decimal context length as a mode when declaring a
context-specific exception, and include that mode in `test-modes`; omitted
contexts continue to expect a pass. Infrastructure failures and unavailable
PPL comparisons are not excused by a model XFAIL. HRX coverage above 32K depends on the llama.cpp submodule including
the graph-resource, 256K attention-bound and long-context accumulation fixes
from [#185](https://github.com/AMD-Ecosystem/llama.cpp/pull/185),
[#193](https://github.com/AMD-Ecosystem/llama.cpp/pull/193) and
[#194](https://github.com/AMD-Ecosystem/llama.cpp/pull/194). The reports record
the exact submodule revisions so a source-pin update can be reviewed
separately from benchmark automation changes.

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
perplexity, explicit `mtp:off`/`mtp:on` keys for expanded Lemonade cases, and
decimal token counts for long-context checks. `fail`
requires failure (a success is a CI-failing XPASS); `report-only` retains the
measured outcome without gating. No exception is inherited by the other mode
or backend. Unsupported benchmark/mode combinations, duplicate modes and
expectations for unselected modes are rejected. Modes are independent cases,
not a Cartesian product. Short perplexity supports only `default`.
Infrastructure and malformed-artifact errors remain fatal.

The pinned Lemonade release predates automatic HRX MTP enablement. Updating the
release pin to one supporting HRX MTP needs no special version-dependent
enablement code. Qwen3.5 generation and perplexity use the default pass
expectation. Backend correctness fixes must also be present in the CI-produced
llama-server.
