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

## Benchmark model tiers

Benchmark models are grouped into cumulative tiers in
`scripts/hrx/lemonade_model_manifest.json`. The root `tiers` array orders tiers
from the smallest to the most comprehensive. Each model's `tier` is its minimum
tier, so selecting a tier includes models assigned to that tier and every tier
below it.

The `smoke` tier benchmarks `qwen3-30b-a3b-instruct-2507` and `llama-3.1-8b`.
The `full` tier adds every other model in the manifest. CI selects tiers by
event:

- Pull requests use `smoke`.
- Pushes to `main` use `full`.
- Manual CI runs offer a `smoke` or `full` choice and default to `full`.
- Release runs use `full`; benchmark failures remain nonblocking for publishing.

## Releases

The `Release` workflow (`.github/workflows/release.yml`) runs nightly and can
also be dispatched manually. The published archive
`llama-<version>-bin-manylinux-hrx-x64.tar.gz` contains llama.cpp release with
HRX and Vulkan backends enabled. The required HRX/ROCm/Vulkan runtime libraries
ship alongside the binaries, so no ROCm install is needed on the target machine.
