#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Decide whether the submodule-bump automation may force-push its branch.

The bump workflow keeps one rolling PR by recreating
``users/automation/bump-submodules`` as ``main`` plus a single bot commit and
force-pushing it (peter-evans/create-pull-request).  Without a guard, a fix a
human pushed to that branch would be silently force-pushed away on the next
scheduled run.  This script pauses the automation while the branch carries any
commit that is not the automation's own; bumping resumes on its own once the
branch is gone (``delete-branch: true`` removes it when the PR merges).

The automation also pauses while the open bump PR's CI has concluded in
failure: a failing bump needs a human to look at it, and force-pushing new
pins would discard the evidence and restart CI as if nothing were wrong.

A commit counts as the automation's own when its author email equals
``BOT_EMAIL``, the identity the workflow defines once and commits with on
its create-pull-request step.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys


def gh_api(path: str) -> dict | list:
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def failed_check_runs(repository: str, branch: str) -> tuple[int | None, list[str]]:
    """Return the open bump PR's number and the names of its failed checks."""
    owner = repository.split("/")[0]
    pulls = gh_api(f"repos/{repository}/pulls?head={owner}:{branch}&state=open")
    if not pulls:
        return None, []

    head_sha = pulls[0]["head"]["sha"]
    runs = gh_api(f"repos/{repository}/commits/{head_sha}/check-runs?per_page=100")
    failed = [
        run["name"]
        for run in runs["check_runs"]
        if run["conclusion"] in ("failure", "timed_out")
    ]
    return pulls[0]["number"], failed


def report(proceed: bool, message: str, *details: str) -> int:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"proceed={str(proceed).lower()}\n")

    print("### Bump branch check\n")
    print(message)
    if details:
        print()
        for line in details:
            print(f"- {line}")
    return 0


def main() -> int:
    repository = os.environ["GITHUB_REPOSITORY"]
    branch = os.environ["BUMP_BRANCH"]
    bot_email = os.environ["BOT_EMAIL"]

    compare = subprocess.run(
        ["gh", "api", f"repos/{repository}/compare/main...{branch}"],
        capture_output=True,
        text=True,
    )
    compare_failed = compare.returncode != 0
    branch_is_absent = compare_failed and "HTTP 404" in compare.stderr
    compare_failed_for_other_reason = compare_failed and not branch_is_absent
    if compare_failed_for_other_reason:
        sys.stderr.write(compare.stderr)
        return compare.returncode

    if branch_is_absent:
        return report(
            True, f"Branch `{branch}` does not exist; proceeding with the bump."
        )

    commits = json.loads(compare.stdout)["commits"]
    human_commits = [
        commit
        for commit in commits
        if commit["commit"]["author"]["email"] != bot_email
    ]
    if human_commits:
        return report(
            False,
            f"A human commit is holding the bump: branch `{branch}` is left "
            "untouched until its PR is merged or the branch is deleted.",
            *(
                f"`{commit['sha'][:12]}` "
                f"{commit['commit']['message'].splitlines()[0]}"
                for commit in human_commits
            ),
        )

    pr_number, failed_checks = failed_check_runs(repository, branch)
    if failed_checks:
        return report(
            False,
            f"Failing CI is holding the bump: PR #{pr_number} needs attention. "
            "The branch is left untouched until its checks pass, a fix is "
            "pushed, or the PR is closed.",
            *(f"`{name}` failed" for name in failed_checks),
        )

    return report(
        True,
        f"Branch `{branch}` carries only automation commits and no failing "
        "checks; proceeding with the bump.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
