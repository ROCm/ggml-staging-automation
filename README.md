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

PR CI defaults to the `single` tier (Qwen3-0.6B). Tiers are cumulative:
`small` contains the existing smaller-model Lemonade suite (currently 25
models, including some larger than 1 GB); `all` includes every model.
Both Lemonade benchmarking and perplexity run on HRX and Vulkan.

Apply `ci:single-model`, `ci:small-models`, or `ci:all-models` to request
coverage explicitly. Adding a coverage label starts CI; the highest selected
tier wins and remains selected on subsequent PR updates. Unrelated labels do
not start builds or benchmarks. Removing a coverage label takes effect on
the next normal PR update. Automated submodule bump PRs request `all`.
Manual CI runs offer all three tiers and default to `single`.

Unlabeled PRs run builds and benchmarks only when their complete diff changes
submodule pointers (`llama.cpp`, `hrx-system`), `.gitmodules`,
`requirements.txt`, `rocm-version.json`, scripts under
`scripts/hrx/{build,benchmark,package,ci}/`, files under `benchmarks/hrx/`,
or the `ci.yml`, `build_llama_cpp_linux.yml`, or `release_test_bench.yml`
workflows. Documentation, security tooling (including Zizmor), and
bump/release orchestration alone do not trigger these jobs. Coverage labels
and manual dispatch override this filter. A documentation-only update to a
PR with earlier relevant changes still runs CI.

New commits and coverage-label requests cancel superseded CI on the same PR.
There are no push-to-main CI builds or benchmarks, and releases do not run
GPU benchmarks. CPU security checks and release publishing retain their own
triggers.

## Releases

The `Release` workflow (`.github/workflows/release.yml`) runs nightly and can
also be dispatched manually. The published archive
`llama-<version>-bin-manylinux-hrx-x64.tar.gz` contains llama.cpp release with
HRX and Vulkan backends enabled. The required HRX/ROCm/Vulkan runtime libraries
ship alongside the binaries, so no ROCm install is needed on the target machine.
