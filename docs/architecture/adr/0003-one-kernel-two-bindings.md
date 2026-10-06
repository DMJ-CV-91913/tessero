# 0003. One C kernel, two bindings

- Status: Accepted
- Date: 2026-09-26

## Context

FFI has a fixed per-call cost of about 9 µs and cannot overload operators or
array access for PHP classes. Applications that make many small calls, such as
per-request pricing or scoring, need lower overhead, and users want
`$a + $b` and `$a['1:, 0']`. The design brief asked for a full Zend extension.
Writing a second numerical engine would double the code to test and let the
two drift apart.

## Decision

The extension (`ext/`) compiles the **same** C sources as the FFI library.
`ext/libtessero/` is a copy kept identical by `tools/sync-ext.sh`, and CI fails
on drift. The extension adds only the PHP object layer: conversions, handlers,
the iterator, JSON and serialisation. Its kernel symbols are hidden (visibility
plus a linker version script), so both bindings can be loaded in one process.
The FFI package's `NDArray` inherits operators from the extension's `Operand`
class when it is loaded.

## Consequences

- Algorithms are implemented and tested once. The extension parity suite
  replays the FFI fixtures to prove the bindings agree.
- The extension's call overhead is about 0.3 µs, the same as a plain PHP function call.
- The extension brings back a build per PHP version for users who choose it.
  It is optional, and PIE handles the build.
- The extension's API is a subset of the FFI package's (no LAPACK, multi-axis
  reductions or `.npy` I/O yet). The gap is documented and on the roadmap.
