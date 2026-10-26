# Packaged HRX tests

The regular build compiles the four HRX suites. The artifact packager produces
both the release archive and a separate `hrx-tests.tar.gz` containing those
executables, matching runtime libraries, source revisions, and a relocatable
CTest manifest. Scheduled release builds also produce the test artifact;
the published release archive does not contain tests.

CI runs tests first in the existing benchmark GPU job. A failing suite stops
benchmark setup and execution. The manual `validation_mode` input selects
`tests-and-benchmarks` (the default), `tests-only`, or `benchmarks-only`.
Tests-only skips Lemonade setup, model/corpus downloads, and benchmark reports.
The existing model tier and benchmark selection apply when benchmarks run.
All modes use the checked-in submodule revisions.

The Python runner only invokes CTest and returns its status. GPU visibility is
inherited from the remote runner. The package supplies its runtime library path,
internal worker counts (buffer 1, backend 4, JIT 2, ops 4), a 900-second per-suite
timeout, and a rule that fails missing-device skips. CTest runs two suites at a
time and streams verbose output. The workflow uploads source metadata, JUnit,
and CTest logs even when tests fail.

To run an extracted package locally, first select a supported GPU as appropriate
for the machine, then run:

```sh
python3 scripts/hrx/ci/run_hrx_tests.py /path/to/hrx-tests
```
