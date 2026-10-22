# ggml-staging-automation

Temporary staging ground for AMD experimental ggml project branches; projects
here will be upstreamed or removed. This is not a stable distribution channel
and should not be relied on as one.

## Building

This repository currently bootstraps Linux development builds for the HRX-enabled
`llama.cpp` submodule. The build uses a pinned TheRock ROCm artifact run, builds
the embedded `hrx-system` submodule, then builds and installs `llama.cpp` with
the HRX backend enabled.

Initialize the submodules first:

```bash
git submodule update --init hrx-system llama.cpp
```

The ROCm artifact pin lives in `rocm-version.json`:

```json
{
  "release_type": "nightly",
  "run_id": "<TheRock run id>"
}
```

`release_type` defaults to `nightly`, but `run_id` must be an exact TheRock run
id. The scripts intentionally do not fall back to a floating latest build. They
fetch the repository's required TheRock artifact closure, including the HIP
headers/tooling needed to compile HRX device kernels.

Install the Python packages used by the TheRock fetch helper:

```bash
python3 -m pip install --upgrade -r requirements.txt
```

Run the full local Release build:

```bash
python3 scripts/hrx/build/build_all.py
```

The default layout is:

```text
build/rocm-root
build/downloads
build/hrx-system-build
build/hrx-system-install
build/llama.cpp-build
build/llama.cpp-install
```

The individual steps are also available for incremental development:

```bash
python3 scripts/hrx/build/fetch_rocm.py
python3 scripts/hrx/build/build_hrx_system.py
python3 scripts/hrx/build/build_llama_cpp.py
python3 scripts/hrx/build/validate_install.py
```

The default build type is `Release` for both `hrx-system` and `llama.cpp`. The
llama.cpp build enables CPU, HRX, and optionally Vulkan for `gfx1100`, `gfx1151`, and
`gfx1201`. When `GGML_HRX_BUNDLE_RUNTIME_LIBS` is enabled by the script, HRX, Loom,
and the required shared ROCm runtime libraries are copied next to the HRX backend
in the build and install trees with `$ORIGIN` RPATHs. ROCm sysdeps are preserved
under an adjacent `rocm_sysdeps/lib` directory.

Windows support is intentionally not implemented yet. The scripts and CMake
layout keep runtime libraries adjacent so the later Windows flow can use the
same basic packaging model with DLL copying instead of ELF RPATHs.

## CI model coverage

PR CI defaults to `single` (Qwen3-0.6B). The cumulative `small` tier covers
Lemonade's smaller-model suite; `all` covers every model. Automated submodule
bumps use `all`.

Apply `ci:single-model`, `ci:small-models`, or `ci:all-models` to start CI with
that coverage; the highest selected tier wins. Manual CI runs also offer all
three tiers.

## Releases

The `Release` workflow (`.github/workflows/release.yml`) runs nightly and can
also be dispatched manually. The published archive
`llama-<version>-bin-manylinux-hrx-x64.tar.gz` contains llama.cpp release with
HRX and Vulkan backends enabled. The required HRX/ROCm/Vulkan runtime libraries
ship alongside the binaries, so no ROCm install is needed on the target machine.

### Lemonade release and MTP comparisons

`benchmarks/hrx/lemonade_release.json` pins the released Ubuntu embeddable
package and its SHA-256. CI verifies the archive and records the pin in
`metadata.json`; it does not build Lemonade. Each private daemon uses the
CI-produced llama-server through `hrx.hrx_bin` and `llamacpp.vulkan_bin`, with
model and executable fetching disabled. Model registrations for `llamacpp-hrx`
and `llamacpp` share the same checksum-verified GGUF via snapshot-cache links.
Manifest names remain independent of Lemonade's `user.*` registration namespace.

Without `mtp`, a model gets an ordinary Lemonade benchmark on each backend.
Adding `"mtp": ["off", "on"]` produces two comparisons: HRX/off versus Vulkan/off,
and HRX/on versus Vulkan/on. Off supplies `--spec-type none`; on supplies no MTP
flag and must demonstrate automatic drafting. Both use normal EOS termination,
4096-token contexts, the existing short/long-output scenarios, one warmup and
three measured runs. Perplexity remains one comparison per model, with both
existing prefill-like and decode-like regimes; it does not exercise MTP.

A loopback HTTP relay records the existing benchmark completion responses,
including warmups, without generating additional requests. Reports show drafted
and accepted tokens and acceptance rate. These counters are diagnostics, not
numerical-correctness checks. An on-case with zero drafted tokens fails; there
is no acceptance-rate threshold. The relay adds client-side latency, so throughput
comparisons use backend timings rather than end-to-end client latency.

Exceptions are independent for each model, check, backend and mode. For example:

```json
{
  "mtp": ["off", "on"],
  "expected_results": {
    "lemonade-benchmark": {
      "hrx": {"on": "fail", "off": "report-only"}
    },
    "perplexity": {
      "hrx": {"default": "fail"}
    }
  }
}
```

Omitted expectations mean `pass`. Use `default` for ordinary models and
perplexity, and explicit `off`/`on` keys for expanded Lemonade cases. `fail`
requires failure (a success is a CI-failing XPASS); `report-only` retains the
measured outcome without gating. No exception is inherited by the other mode
or backend. Infrastructure and malformed-artifact errors remain fatal.

The pinned Lemonade release predates automatic HRX MTP enablement. The explicit
Qwen3.5 HRX/on exception records this missing capability. When updating the
release pin to one supporting HRX MTP, remove that exception; no special
version-dependent enablement code is needed. Backend correctness fixes must
also be present in the CI-produced llama-server.
