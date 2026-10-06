# Memory-safety review (2026-09)

A review of both bindings, the C kernel and the manuals against the usual
failure modes of native code called from PHP: the checklist PHP FFI guides
give, plus the problems specific to arrays, memory maps and long-lived
workers. Each finding was fixed with a regression test, or is recorded below
as a residual risk.

## Checklist

| Practice | Tessero | Status |
|---|---|---|
| Parse the C header once, not per request; preload under FPM | `Library::ffi()` loads once per process (`FFI::cdef`). Under FPM, `resources/preload.php` binds the scope with `FFI::load`, and request code only calls `FFI::scope()`. The header is macro-free and FFI-parseable by design. | ✔ |
| Pair every native allocation with a destructor | `Native\Buffer` is the only owner of kernel memory. `__destruct` frees it (`tsr_free`) or unmaps it (`tsr_mmap_close`). The extension frees in `free_obj`, which runs after all views are gone. | ✔ |
| Prevent double frees | `released` flag; `Buffer` cannot be cloned; the map handle is cleared on close. The extension frees exactly once, in `free_obj`, and clears the map pointer on release. | ✔ (hardened, see F-4) |
| Pointers must not outlive their owner | `ptr()` is always called on a named variable in the statement that calls C. A search found no temporary chains like `$x->copy()->ptr()`. `ptr()`/`meta()` document the contract. | ✔ (documented) |
| Validate pointer lifetimes with sentinel state | every accessor checks `released`; LAPACK pointers go through the same check | ✔ (F-4) |
| `FFI::string` copies; PHP strings passed to C live only for the call | `FFI::string` is used only to copy results out; no C function keeps a PHP string | ✔ |
| Opaque handles instead of guessed struct layouts | the memmap handle is `void*`; the one shared struct (`tsr_array`) is declared in the same header the kernel is compiled from | ✔ |
| Header and binary must match | `tsr_version()` is checked against `Tessero::VERSION` at load | ✔ (F-5) |
| "Paranoid C": validate arguments, return error codes | every kernel returns a negative code on bad input; the bindings map codes to typed exceptions; untrusted-text parsers are fuzzed | ✔ |
| Debug with ASan and core dumps | the C suite, the extension suites and the fuzzers run under ASan + UBSan in CI; troubleshooting documents core dumps, `gdb` and ASan builds of both bindings | ✔ |
| Memory outside `memory_limit` must be bounded | the per-process native budget throws before allocating; `memory_in_use` and `memory_mapped` are exported for monitoring; the Laravel provider resets settings per request and job | ✔ |
| Long-running workers must not leak | soak harness: native and mapped bytes back at baseline after **every** iteration, plus heap, RSS and descriptor trends | ✔ (5.5 h nightly; 24–72 h per release) |

## Findings fixed in this review

| ID | Severity | Where | Problem | Fix and test |
|---|---|---|---|---|
| F-1 | High | extension `zeros/empty/…`, `fromBytes`, `__unserialize`; C `tsr_array_reshape` | Element counts were multiplied without overflow checks. A shape such as `[2^61, 8]` wraps to 0 elements: a 64-byte block was then described as a huge array, and any access read or wrote out of bounds. Serialized payloads from a cache or queue, or `fromBytes` with an attacker-chosen shape, could trigger it. Negative dimensions whose product was positive were also accepted by `unserialize`. | overflow-checked counts (`tsr_shape_size`, `checked_count`), rejected before allocating; `014-memory-safety.phpt`, C `test_size_overflow` |
| F-2 | High | FFI `NDArray` constructor, `empty/zeros`, `fromBytes`, `fromFlat`, `reshape`, `__unserialize` | Same class of bug in PHP: `(int) array_product()` silently becomes a float and then an arbitrary int on overflow, and negative dimensions passed `fromBytes`/`unserialize`. | `NDArray::checkedSize()` everywhere a shape becomes a size; `Buffer::allocate` refuses negative sizes; `MemorySafetyTest` |
| F-3 | High | FFI `Generator::shuffle` | Shuffled in place without checking writability. On a read-only memory map (`PROT_READ`) the C write would crash the process. | writability check; `testReadOnlyMemoryIsNeverWrittenFromC` |
| F-4 | Medium | FFI `Buffer::flush`, `Linalg` | After release (possible during PHP's shutdown destructor pass), `flush()` would pass a freed mapping handle to C, and `Linalg` built LAPACK pointers from the raw address without the release check. | handle cleared on close; `flush` checks `released`; `Linalg::d()` goes through `ptr()`; `testReleasedBuffersAreNeverUsed` |
| F-5 | Medium | FFI loading | A binary from another version (a stale `TESSERO_LIB`) loaded silently if its symbols happened to match. A changed struct layout would then corrupt memory instead of failing. | version check at load, with an actionable message |
| F-6 | Medium | `resources/preload.php` | The generated FFI header, which names the library to `dlopen`, was written to the shared temp directory under a predictable name and reused if it already existed. A local user could plant a header pointing at another library, loaded at FPM start. | `tempnam` (exclusive, 0600), deleted after `FFI::load`; `tests/e2e/preload.sh` |
| F-7 | Medium | C slice parser (found by fuzzing) | A blank item read past its comma; huge steps overflowed the count and stride arithmetic (also in the PHP slicer). | bounded blank skipping, overflow-free count and stride; C and PHP regressions |
| F-8 | Low | FFI `out:` arguments | Direct writes into an `out` array that partially overlapped an input produced wrong values. This is not a memory-safety issue, but it is the same aliasing class. | overlap check, temporary; `MathTest::testOverlappingOutputGoesThroughATemporary` |
| F-9 | Medium | extension `sort_impl`; FFI `NDArray::sort`, `CsrMatrix` | The return codes of `tsr_sort`/`tsr_argsort` (and three sparse kernels in FFI) were ignored. When the budget ran out mid-sort, the caller got an unsorted array or zero indices and no error. This is not memory corruption, but it is a silent wrong result. Found while generating the coverage inventory. | every call checked; the partial result is freed before throwing; budget cases in `MemorySafetyTest` and `014-memory-safety.phpt` |

## Residual risks and how they are handled

- **SIGBUS on memory maps.** If another process truncates a mapped file,
  or a network filesystem disappears, the next access raises SIGBUS. No
  library can prevent this. The guide says to map only files that are not
  shrunk, and troubleshooting explains the signal.
- **`fork()` after OpenMP.** libgomp does not survive a fork once its thread
  pool exists. Documented, with a runbook. The safe default is `threads = 1`.
- **Raw pointers are public.** `NDArray::ptr()`, `meta()` and `buffer()` exist
  for advanced integrations. They are documented as valid only while the
  array is referenced.
- **Windows.** The memory-map code for Win32 has not been compiled yet, and the
  kernel's atomics need MSVC C11 atomics support. This is gated on a Windows
  CI runner ([roadmap](roadmap.md)).
- **Coverage-guided fuzzing** runs in CI (libFuzzer). The review environment
  could only run the standalone mutation driver: about 13 million inputs,
  clean after F-7.
- **Soak length.** Runs so far: 30 minutes per backend, about 1.26 million
  mixed iterations, all criteria met. The 24–72 h runs are a release gate
  ([Soak testing](../operations/soak-testing.md)).

## Verification after the fixes

- C: 270 checks, optimised, ASan + UBSan, and Clang.
- FFI, extension and cross-backend PHP suites: 1 214 pass (26 FFI-only skipped
  under the extension), plus 706 FFI tests in each JIT mode (off, function, tracing).
- Extension: 14 phpt files, including `014-memory-safety.phpt`.
- Preload end-to-end.
- Laravel bridge on both backends.
- 59 documentation examples.
