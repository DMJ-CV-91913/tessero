# 0009. A ufunc engine in the extension, and memory-mapped arrays outside the budget

- Status: Accepted, amended by [0010](0010-shared-kernels-and-vector-math.md)
- Date: 2026-09-27

## Context

The brief for this release asked for NumPy-style universal functions
(`Math::sin($matrix)`, SIMD-friendly loops, broadcasting, output buffers) and
memory-mapped arrays (`NDArray::memmap()`, `munmap` on destruction, `flush()`
with `MS_ASYNC`/`MS_SYNC`). The extension already had element-wise kernels
for about 50 operations, behind `NDArray` methods and operators.

## Decision

1. **Two ufunc families, one interface.** `Tessero\Ext\Math` exposes the
   existing kernel operations (so `Math::sin($a)` and `$a->sin()` are
   bit-identical by construction), plus new functions implemented as tables of
   typed inner loops, `loop(args, n, steps)`, in the style of NumPy's ufunc
   loops. One engine does type resolution, casting, broadcasting (zero
   strides), dimension coalescing and threading for all loop ufuncs.
2. **Loops are written to vectorise.** Each inner loop has a contiguous path
   over `restrict` pointers with a direct call and `omp simd`, compiled for
   AVX-512/AVX2/baseline with `target_clones`. Tessero does not use
   `-ffast-math`, so libm calls are not vectorised. Accuracy and NaN semantics
   take precedence over the extra speed.
3. **`out` writes directly when possible.** The element-wise cores gained an
   optional destination. It is used directly when dtype and shape match and no
   input partially overlaps it (exact in-place aliasing is allowed). Otherwise
   the engine computes into a temporary and casts into `out` (same-kind casts
   only).
4. **Memmaps are a second kind of owner.** A root array owns either a
   `tsr_alloc` block or a `tsr_map`. Views reference the root in both cases, so
   the existing ownership rules (and their tests) cover mappings unchanged.
5. **Mapped bytes are outside the memory budget** and reported separately
   (`memory_mapped`). The budget exists to stop anonymous memory from reaching
   the OOM killer. File-backed pages are evictable, and counting them would make
   large mappings impossible.
6. **The mapping's file descriptor stays open** until release, so
   `flush(sync: true)` can `fsync` (required for durable file-size changes).
7. **Read-only is enforced twice:** a flag checked on every write path, and
   `PROT_READ` pages.

## Consequences

- 74 functions with NumPy's broadcasting and dtype results, checked against
  NumPy/SciPy (64 fixture cases) and identical across thread counts.
- A write into a read-only map is a PHP exception, not a crash. Truncating a
  mapped file from outside still raises SIGBUS. This is inherent to mmap and is
  documented.
- The FFI package did not get the loop ufuncs or memmap in this step. ADR 0010
  moved both into `csrc/`, which gave the FFI package the same functions.

## Alternatives considered

- **Function-pointer-per-element ufuncs** (an indirect call per element) would be
  simpler but cannot vectorise. They were rejected in favour of typed inner
  loops called once per run.
- **`emalloc`-style accounting for mappings** was rejected (see 5).
- **Closing the descriptor right after `mmap()`** (POSIX allows it) was
  rejected because `fsync` needs it (see 6).
