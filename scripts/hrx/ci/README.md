# Packaged HRX tests

CI builds a separate `hrx-tests` artifact containing the four HRX suites and
runtime libraries from the same build. The gfx1151 job pins a supported KFD
device with `ROCR_VISIBLE_DEVICES`, checks executable dependencies, then runs
CTest with two concurrent suites. Internal workers are buffer 1, backend 4,
Loom JIT 2, and ops 4; each suite has a 900-second timeout. Failures remain CI
failures while CTest collects the remaining suites. `hrx-test-results-gfx1151`
contains exact source revisions, GPU properties, commands, wall time, verbose
per-case output, CTest logs, and JUnit results.

The manual CI input `test_backend_ref` selects an explicit backend revision
for a qualification run and skips model benchmarks. Leave it empty for the
normal submodule pin. Qualification results must be reported with that revision;
they do not qualify the normal pin. The release workflow still excludes tests.
