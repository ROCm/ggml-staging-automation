# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build_tools" / "github_actions"))

from find_main_benchmark import discover_main_benchmark  # noqa: E402


def workflow_run(
    run_id: int,
    conclusion: str,
    created_at: str,
) -> dict:
    return {
        "id": run_id,
        "html_url": f"https://github.example/actions/runs/{run_id}",
        "status": "completed",
        "conclusion": conclusion,
        "head_branch": "main",
        "created_at": created_at,
    }


def artifact(
    artifact_id: int,
    name: str = "lemonade-bench-gfx1151",
    *,
    expired: bool = False,
) -> dict:
    return {
        "id": artifact_id,
        "name": name,
        "expired": expired,
        "created_at": "2026-08-12T02:30:00Z",
    }


class MainBenchmarkDiscoveryTest(unittest.TestCase):
    def discover(self, runs: list[dict], artifacts: list[dict] | None):
        requested_urls = []

        def send_request(url: str):
            requested_urls.append(url)
            if "/actions/workflows/" in url:
                return {"workflow_runs": runs}
            if "/artifacts?" in url:
                if artifacts is None:
                    self.fail("Artifacts were queried for an unusable workflow run")
                return {"artifacts": artifacts}
            self.fail(f"Unexpected GitHub API request: {url}")

        result = discover_main_benchmark(
            "ROCm/ggml-staging-automation",
            "ci.yml",
            "main",
            "lemonade-bench-gfx1151",
            send_request=send_request,
        )
        return result, requested_urls

    def test_newest_completed_success_uses_its_exact_unexpired_artifact(self) -> None:
        older = workflow_run(100, "success", "2026-08-10T00:00:00Z")
        newest = workflow_run(200, "success", "2026-08-12T00:00:00Z")

        result, urls = self.discover(
            [newest, older],
            [artifact(301, "different"), artifact(302)],
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.run_id, 200)
        self.assertEqual(result.artifact_id, 302)
        self.assertEqual(
            result.run_url, "https://github.example/actions/runs/200"
        )
        self.assertIn("actions/workflows/ci.yml/runs?", urls[0])
        self.assertIn("branch=main", urls[0])
        self.assertIn("status=completed", urls[0])
        self.assertIn("per_page=1", urls[0])
        self.assertIn("actions/runs/200/artifacts?", urls[1])

    def test_failed_newest_completed_run_does_not_fall_back(self) -> None:
        older_success = workflow_run(100, "success", "2026-08-10T00:00:00Z")
        newest_failure = workflow_run(200, "failure", "2026-08-12T00:00:00Z")

        result, urls = self.discover([newest_failure, older_success], None)

        self.assertIsNone(result)
        self.assertEqual(len(urls), 1)

    def test_successful_newest_run_without_named_artifact_is_unavailable(self) -> None:
        newest = workflow_run(200, "success", "2026-08-12T00:00:00Z")

        result, urls = self.discover([newest], [artifact(301, "different")])

        self.assertIsNone(result)
        self.assertEqual(len(urls), 2)

    def test_successful_newest_run_with_expired_artifact_is_unavailable(self) -> None:
        newest = workflow_run(200, "success", "2026-08-12T00:00:00Z")

        result, urls = self.discover([newest], [artifact(302, expired=True)])

        self.assertIsNone(result)
        self.assertEqual(len(urls), 2)


if __name__ == "__main__":
    unittest.main()
