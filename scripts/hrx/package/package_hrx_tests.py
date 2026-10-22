#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Package the four HRX suites with the runtime from the same release build."""

import json
import subprocess
from pathlib import Path, PurePosixPath

from build_artifacts_from_llama_cpp_install import (
    REPO_ROOT, PackageEntry, create_package, glob_install_tree,
)

SUITES = {
    "test-hrx-buffer": ("hrx-buffer", 1),
    "hrx-backend-test": ("hrx-backend", 4),
    "test-hrx-loom-jit": ("hrx-loom-jit", 2),
    "test-hrx-ops": ("hrx", 4),
}


def main():
    build = REPO_ROOT / "build"
    staging = build / "hrx-test-metadata"
    staging.mkdir(exist_ok=True)
    entries = glob_install_tree(build / "llama.cpp-install", "lib/**/*.so*")
    ctest = []
    for name, (prefix, workers) in SUITES.items():
        binary = build / "llama.cpp-build" / "bin" / name
        if not binary.is_file():
            raise SystemExit(f"Missing HRX test executable: {binary}")
        entries.append(PackageEntry(PurePosixPath("bin") / name, binary))
        ctest.append(
            f'add_test({name} "${{CMAKE_CURRENT_LIST_DIR}}/bin/{name}" '
            f'--{prefix}-test-jobs {workers} --{prefix}-test-filter . --{prefix}-test-output all)\n'
            f'set_tests_properties({name} PROPERTIES TIMEOUT 900 '
            f'FAIL_REGULAR_EXPRESSION "test skipped: no device")\n'
        )
    (staging / "CTestTestfile.cmake").write_text("".join(ctest))
    revisions = {
        name: subprocess.check_output(["git", "-C", str(REPO_ROOT / directory),
                                       "rev-parse", "HEAD"], text=True).strip()
        for name, directory in (("automation", "."), ("llama_cpp", "llama.cpp"),
                                ("hrx_system", "hrx-system"))
    }
    (staging / "metadata.json").write_text(json.dumps({
        "revisions": revisions, "internal_workers": {k: v[1] for k, v in SUITES.items()},
    }, indent=2) + "\n")
    for name in ("CTestTestfile.cmake", "metadata.json"):
        entries.append(PackageEntry(PurePosixPath(name), staging / name))
    create_package(package_file=build / "artifacts" / "hrx-tests.tar.gz",
                   package_root=PurePosixPath("hrx-tests"), entries=entries)


if __name__ == "__main__":
    main()
