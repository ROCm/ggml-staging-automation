#!/usr/bin/env python3
"""Configure, build, and install llama.cpp tests for release overlay packaging."""

from __future__ import annotations

import argparse

from build_llama_cpp import build_llama_cpp
from hrx_build import add_common_path_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_path_args(parser)
    parser.add_argument("--build-type", default="Release")
    parser.add_argument("--no-install", action="store_true")
    parser.add_argument("--backend-dl", action="store_true")
    args, extra_cmake_args = parser.parse_known_args()

    build_llama_cpp(
        rocm_root=args.rocm_root,
        hrx_install=args.hrx_install_dir,
        build_dir=args.llama_test_build_dir,
        install_dir=args.llama_test_install_dir,
        build_type=args.build_type,
        target="all",
        install=not args.no_install,
        ggml_build_tests=False,
        llama_build_tests=True,
        llama_tests_install=True,
        build_examples=False,
        backend_dl=args.backend_dl,
        extra_cmake_args=extra_cmake_args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
