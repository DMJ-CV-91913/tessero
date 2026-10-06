# 0001. FFI-first distribution with prebuilt native kernels

- Status: Accepted
- Date: 2026-09-26

## Context

Earlier PHP numerics projects took one of two routes. Pure-PHP libraries are
portable but store each number as a 16-byte zval and run about 30× slower than
NumPy on element-wise work. C extensions are fast, but need a build per PHP
version, OS and thread-safety mode. That burden has ended several projects when
their maintainers moved on. Since PHP 7.4, FFI can call C libraries without an
extension, and it ships with PHP.

## Decision

The primary distribution is a Composer package that loads a prebuilt C
library (`libtessero`) through FFI. The library is built by CI for six platforms
and shipped in `lib/<os>-<arch>/`. Users install with `composer require` and
need no compiler. The secure `ffi.enable=preload` default is supported through
a preload script rather than asking users to enable FFI everywhere.

## Consequences

- Installation matches every other Composer package. There is no per-PHP-version build.
- One call costs about 9 µs, so the API must avoid per-element calls. Every
  operation is whole-array.
- FFI defects in PHP affect us. Two were found and worked around
  ([Upstream bugs](../../project/upstream-bugs.md)).
- The FPM story needs preload configuration, which is documented and
  covered by an end-to-end test.
- For workloads of many small calls, a native extension is still valuable
  ([ADR-0003](0003-one-kernel-two-bindings.md)).

## Alternatives considered

- **Extension only**: fastest, but brings back the build-matrix burden. It is
  now offered as an optional second binding.
- **Pure PHP**: rejected on performance and memory grounds.
