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
python3 scripts/hrx/build_all.py
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
python3 scripts/hrx/fetch_rocm.py
python3 scripts/hrx/build_hrx_system.py
python3 scripts/hrx/build_llama_cpp.py
python3 scripts/hrx/validate_install.py
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

## CI

The `CI` workflow (`.github/workflows/ci.yml`) runs on pull requests and pushes
to `main`. It builds the release package, extracts its `llama-server`, and runs
the following HRX-only Lemonade benchmark on `gfx1151`:

```bash
lemonade bench --backend hrx --auto-pull --output benchmark.json Qwen3-30B-A3B-Instruct-2507-GGUF
```

The workflow builds `AaronStGeorge/lemonade@hrx-integration` without its web
application and configures it to use the extracted same-run `llama-server`.
Executable downloads are disabled, so Lemonade cannot fall back to the bundled
HRX executable. Lemonade's current model registry, benchmark scenarios, and
other user-facing defaults are intentionally left unpinned. Long-context and
Vulkan workloads are not part of this benchmark.

A successful run uploads `lemonade-bench-gfx1151` for 90 days. The artifact has
two files at its root:

- `benchmark.json` is Lemonade's untouched output and the stable input for
  future benchmark comparisons.
- `metadata.json` records its schema version and the exact `llama.cpp`,
  `hrx-system`, and Lemonade commits.

Failed scenarios, malformed output, or any nonzero `failed_runs` prevent this
baseline artifact from being published. The daemon and benchmark logs are
uploaded in a separate `lemonade-bench-debug-logs-gfx1151` failure artifact
instead. The build continues to produce the existing test package with one-day
retention for compatibility, although the benchmark workflow no longer
downloads or executes its op tests.

## Releases

The `Release` workflow (`.github/workflows/release.yml`) runs nightly and can
also be dispatched manually. The published archive
`llama-<version>-bin-manylinux-hrx-x64.tar.gz` contains llama.cpp release with
HRX and Vulkan backends enabled. The required HRX/ROCm/Vulkan runtime libraries
ship alongside the binaries, so no ROCm install is needed on the target machine.
