#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Find the benchmark artifact from the latest completed main CI run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import sys
from typing import Callable
from urllib.parse import urlencode

from github_actions_api import GitHubAPIError, gha_send_request, gha_set_output


JsonRequest = Callable[[str], object]


@dataclass(frozen=True)
class MainBenchmarkArtifact:
    run_id: int
    run_url: str
    artifact_id: int


def discover_main_benchmark(
    repository: str,
    workflow: str,
    branch: str,
    artifact_name: str,
    *,
    send_request: JsonRequest = gha_send_request,
) -> MainBenchmarkArtifact | None:
    """Return a usable artifact from the latest completed workflow run."""
    runs_query = urlencode(
        {"branch": branch, "status": "completed", "per_page": 1}
    )
    runs_response = send_request(
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{workflow}/runs?{runs_query}"
    )
    runs = runs_response["workflow_runs"]
    if not runs or runs[0]["conclusion"] != "success":
        return None

    run = runs[0]
    artifacts_query = urlencode({"name": artifact_name, "per_page": 100})
    artifacts_response = send_request(
        f"https://api.github.com/repos/{repository}/actions/runs/{run['id']}"
        f"/artifacts?{artifacts_query}"
    )
    artifact = next(
        (
            candidate
            for candidate in artifacts_response["artifacts"]
            if candidate["name"] == artifact_name and not candidate["expired"]
        ),
        None,
    )
    if artifact is None:
        return None

    return MainBenchmarkArtifact(
        run_id=run["id"],
        run_url=run["html_url"],
        artifact_id=artifact["id"],
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args(argv)

    try:
        artifact = discover_main_benchmark(
            args.repository,
            args.workflow,
            args.branch,
            args.artifact,
        )
    except (GitHubAPIError, KeyError, TypeError) as exc:
        print(f"Could not discover main benchmark artifact: {exc}", file=sys.stderr)
        return 1

    if artifact is None:
        print("The latest completed main CI run has no usable benchmark artifact.")
        return 0

    gha_set_output(
        {
            "run_id": str(artifact.run_id),
            "run_url": artifact.run_url,
            "artifact_id": str(artifact.artifact_id),
        }
    )
    print(
        f"Using benchmark artifact {artifact.artifact_id} from "
        f"{artifact.run_url}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
