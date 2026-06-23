# ggml-staging-release

Release management repo for AMD staging ggml project branches (prior to upstreaming).

## Building

This repository currently bootstraps Linux development builds for the HRX-enabled
`llama.cpp` submodule. The build uses a pinned TheRock ROCm artifact run, builds
the embedded `hrx-system` submodule, then builds and installs `llama.cpp` with
the HRX backend enabled.

Initialize the submodules first:

```bash
git submodule update --init hrx-system llama.cpp
```

Set the ROCm artifact pin in `rocm-version.json`:

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
llama.cpp build enables CPU, Vulkan, and HRX for `gfx1100`, `gfx1151`, and
`gfx1201`. When `GGML_HRX_EMBED_ROCM_LIBS` is enabled by the script, HRX, Loom,
and the required shared ROCm runtime libraries are copied next to the HRX backend
in the build and install trees with `$ORIGIN` RPATHs. ROCm sysdeps are preserved
under an adjacent `rocm_sysdeps/lib` directory.

Windows support is intentionally not implemented yet. The scripts and CMake
layout keep runtime libraries adjacent so the later Windows flow can use the
same basic packaging model with DLL copying instead of ELF RPATHs.
