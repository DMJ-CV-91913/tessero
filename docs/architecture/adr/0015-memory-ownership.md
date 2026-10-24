# 0015. Memory ownership: the layer that allocates memory frees it

- Status: Accepted
- Date: 2026-10-24
- Amends: 0004 (memory budget), 0011 (registry results), 0014 (ABI)

## Context

Tessero has four layers that allocate memory: the C kernel (`libtessero`, its own counted allocator
`tsr_alloc` on top of `aligned_alloc` / `_aligned_malloc`), the vendored C and C++ code inside it (C library
`malloc`, C++ `new`, LAPACKE's internal workspaces), the Zend extension (the Zend allocator: `emalloc` per request,
`pemalloc(..., 1)` per process) and the PHP FFI binding (`FFI::new`, managed by PHP unless created with
`owned = false`). Native consumers (Talario, Kinetix) link `libtessero` directly and receive kernel memory too.

Memory freed by a different allocator than the one that produced it is undefined behaviour, and in this stack it
fails differently per platform: on Linux `free()` of a `tsr_alloc` block happens to work (glibc accepts
`aligned_alloc` memory) but silently corrupts the memory budget (`tsr_allocated`); on Windows the same call crashes
(`_aligned_malloc` memory needs `_aligned_free`); `efree()` of kernel memory corrupts the Zend heap of the request.
The 2026-10 fuzzing found one such defect (`signal.stft` released `tsr_alloc` blocks with `free()`), and an audit
of the result teardown found a second (the extension did not release kind 2, string, results on the failure path
of a multi-result routine).

## Decision

**The golden rule: the layer that allocates memory owns its deallocation, with the matching function and size.**

This is DJs Golden Rule(TM), made binding for Tessero and Talario by the project owner on 2026-10-24 in the
words "The layer that allocates the memory must own its deallocation." The principle is rooted in established
practice in native and PHP extension development; the mark (a trademark of Douglas Johnson, unregistered) names
the projects' adoption and enforcement of it, which this ADR specifies.

### The contract per ABI call (for native consumers and both bindings)

| Memory handed out by | Released by | Never |
|---|---|---|
| `tsr_alloc(bytes)`, `tsr_calloc(bytes)` | `tsr_free(p, bytes)`, the same `bytes` | `free`, `efree`, `FFI::free`, a different size |
| `tsr_alloc_n(n, elsize, pad)` | `tsr_free(p, n*elsize + pad)` | |
| `tsr_fn_routine` result kind 2 (string) and kind 3 (array) | `tsr_free(arr.data, bytes)`, or `tsr_fn_results_free(&r, 1)` | |
| `tsr_fn_routine` result kind 6 (sequence: `arr.ndim` 1, `arr.shape[0]` items, `arr.data` holds them, `bytes` its size) | each item as above, then `tsr_free(arr.data, bytes)`; `tsr_fn_results_free(&r, 1)` does both | freeing the block before its items |
| `tsr_fn_routine` failure (negative return) | nothing: the kernel has already released partial results; every slot is kind 0 | |
| `tsr_mmap_open(..., &handle, &data)` | `tsr_mmap_close(handle)`, once, after the last use of `data` | `tsr_free`/`free` of `handle` or `data` |
| strings returned by the header (`tsr_version`, `tsr_fn_name`, `tsr_fn_error`, `tsr_mmap_error`, `tsr_ufunc_name`, ...) | nobody: static or thread-local | any free |
| caller memory passed in (arrays, strings, sequences, RNG state) | the caller | kept by the kernel past the call |

`tsr_fn_results_free(tsr_result *res, int64_t n)` is new in TESSERO_0.2 (additive; `tsr_abi_version()` stays 2).
It is the kernel's own teardown, so native consumers and both bindings release results exactly as the kernel
built them.

### Layers

- **Kernel.** Memory that crosses the ABI comes from `tsr_alloc`. The C library allocator is used for scratch
  that is allocated and freed in the same function (LAPACK workspaces, temporaries) and never reaches a result.
  No kernel code, thread-local or cache keeps a caller pointer after the call returns (thread-locals that carry
  per-call context are restored or cleared before return).
- **Threads.** OpenMP worker threads run kernel code only. They never touch Zend memory or any Zend API (Zend's
  allocator is per request and single threaded): there is no Zend token under `csrc/`, no `#pragma omp` in the
  extension, and no callback in `tessero.h` through which a worker could re-enter PHP. Workers read and write
  caller buffers only during the synchronous call. Kernel errors go to thread-local buffers (`tsr_fn_error`,
  `tsr_mmap_error`) that the calling thread reads after the call; no parallel region sets an error or prints.
  A future callback in `tessero.h` must run on the calling thread only and say so ("ADR 0015: calling thread").
- **Extension.** Request memory comes from `emalloc`/`ecalloc` and is freed with `efree` on every path, including
  `goto done` error paths. Kernel memory adopted by an `NDArray` is released by its `free_obj` handler with
  `tsr_free(block, block_bytes)`; a view holds a counted reference to its root object (never a second owner of the
  block), so the block lives until the last view is gone and is freed once. Clone copies; unserialize refuses an
  initialised object. Persistent memory (bindings, arginfo, method tables, interned class names, the method hash)
  is allocated at MINIT with `pemalloc(..., 1)`, recorded (`tsr_pkeep`) and returned at MSHUTDOWN with
  `pefree(..., 1)`; no request memory is cached in module globals across requests.
- **FFI binding.** Kernel memory is held by exactly one `Tessero\Native\Buffer`, which frees it in `__destruct`
  with `tsr_free($address, $bytes)` (or `tsr_mmap_close` for a mapping). Between `tsr_fn_routine` returning and a
  Buffer adopting a result, `Registry` releases every result not yet adopted on every exception path. Memory PHP
  allocates (`FFI::new`) never reaches `tsr_free`; the only unmanaged `FFI::new(..., false)` buffers (string
  arguments) are freed with `FFI::free` in a `finally`. Buffers are never cloned or serialized.

### Gates

1. **Runtime: `-DTSR_OWNERSHIP_CHECK`.** `tsr_alloc` places a 64-byte header before the pointer it hands out (the
   pointer stays 64-byte aligned) and records every live block with its size. Then `free()`/`efree()` of a kernel
   pointer is an invalid free that ASan and the C library abort on, and `tsr_free` aborts with "libtessero
   ownership violation" on a pointer `tsr_alloc` did not return, a double free, or a size other than the one
   allocated. `tsr_ownership_check()` / `tsr_feature("ownership_check")` report the mode. It is on in
   `make sanitize`, the fuzz builds and the ASan CI jobs (C tests, phpt, parity); release builds compile none of
   it (identical code).
2. **Static: `tools/check-ownership.sh`** (bash + perl) in CI, with `tools/check-ownership-selftest.sh` proving every
   rule fires: K1 counted-allocator pointer to `free()` or C-library pointer to `tsr_free` within a function; K2
   C-library memory stored into a result; K3 Zend tokens under `csrc/`; E1 raw C allocator in the extension; E2
   `efree` of kernel memory; E3 `#pragma omp` in the extension; H1 an unmarked callback in `tessero.h`; P1
   `FFI::free` of kernel memory or of managed `FFI::new` memory; P2 `tsr_free` outside the two PHP owners.
3. **Tests.** C death tests for each violation class under the check; `tsr_fn_results_free` and failing-routine
   tests; PHP tests in both bindings (`tests/Unit/OwnershipTest.php`, `ext/tests/020-ownership.phpt`): views
   outlive a dropped parent and drop in either order, results are released on success and failure, and a
   10000-iteration request loop returns `memoryInUse()` to its baseline.

## Consequences

- Native consumers have one documented way to release every kind of kernel memory, and a teardown function so
  they do not re-implement it.
- A violation on a tested path fails CI on every platform the ASan jobs run on, not only on Windows at run time.
- The check costs a lock and a hash lookup per allocation in instrumented builds only.

## Alternatives considered

- **Converting every kernel `malloc` to `tsr_alloc`.** Rejected: scratch that never leaves a function obeys the
  rule already, and the conversion would touch about 1500 sites for no ownership gain.
- **A header (magic) without a registry.** Rejected: checking a foreign pointer's header reads memory in front of
  it (an out-of-bounds read under ASan), and a double free cannot be told from reuse.
