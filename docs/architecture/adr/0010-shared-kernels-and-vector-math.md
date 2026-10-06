# 0010. Shared kernels for both bindings, and vector exp/log written in-house

- Status: Accepted
- Date: 2026-09-27
- Amends: [0009](0009-ufunc-engine-and-memmap.md) (the ufunc loops and the mapping code move from the extension into libtessero)

## Context

ADR 0009 put the loop ufuncs and the memory-mapping code in the extension.
That left the FFI package without `Math`, `memmap` or `.npy` memory maps, and
each gap would have needed a second implementation in PHP. The roadmap also
called for fuzzing the code that reads untrusted input, and for closing the
largest speed gap with NumPy: `exp`/`log` were 4–8× slower, because glibc's
versions are scalar.

## Decision

1. **Everything that decides a result lives in libtessero.**
   - `csrc/src/ufunc.c`: the loop tables, type resolution (`tsr_ufunc_resolve`)
     and the threaded loop runner (`tsr_ufunc`).
   - `csrc/src/mmap.c`: memory-map planning (`tsr_mmap_plan`, pure arithmetic),
     mapping, flush and unmap (`tsr_mmap_*`), and the `.npy` header parser and
     writer (`tsr_npy_*`).

   The bindings keep argument parsing, PHP-level checks (open_basedir, stream
   wrappers), object lifetimes and exception mapping. Each binding is a thin
   layer over the same code, so results are identical by construction, and a
   cross-backend test checks them byte for byte.
2. **Pure functions for fuzzing.** Planning and parsing are separated from I/O
   (`tsr_mmap_plan`, `tsr_npy_header`, `tsr_array_slice`, `tsr_array_json`), so
   libFuzzer can drive them millions of times per minute with strong oracles.
3. **Vector exp/log from fdlibm's algorithms rather than SLEEF.**
   - `csrc/src/vmath.h` implements `exp` and `log` branch-free: special cases
     are blended in with bit masks, so GCC and Clang vectorise the kernels'
     loops for SSE2, AVX2 and AVX-512.
   - It keeps fdlibm's argument reduction and minimax coefficients. Measured
     error is below 1 ulp against long-double references over 2×10^7 points,
     with C99 special values.
   - float32 goes through the float64 kernel and rounds once.
   - The kernel is compiled with `-fno-trapping-math`. That flag only lets the
     compiler if-convert selects; it changes no result.

## Consequences

- The FFI package gains `Tessero\Math` (74 ufuncs), `NDArray::memmap`,
  `openMemmap`, `load($path, $mmapMode)` and `save`. The extension gains
  `.npy` load, save and `openMemmap`, plus multi-axis reductions with
  `keepdims`.
- The ABI grows by the `tsr_ufunc_*`, `tsr_mmap_*`, `tsr_npy_*`, `tsr_rfft` and
  `tsr_irfft` functions and the error code −9 (`TSR_EIO`). Prebuilt binaries
  and headers must match; `ext/libtessero` is synced by `tools/sync-ext.sh` as
  before.
- `exp` on 10^6 float64 went from 9.3 ms to 1.2 ms (NumPy with AVX-512:
  0.76 ms), and `log` to 1.7 ms. Results can differ from glibc's in the last
  bit. Parity tolerances (1e-12 relative) already allow for this.
- Vectorised `sin`/`cos`/`tan` are not done yet. They need a Payne–Hanek
  reduction for large arguments to stay under 1 ulp. They are the next step
  on the roadmap.

## Alternatives considered

- **SLEEF.** It is the most complete vector libm, and a good choice where it
  can be vendored. It could not be fetched in this environment, and it would
  add a large dependency to the self-contained `ext/` tree that PIE builds.
  `vmath.h` has an interface a SLEEF backend could replace later.
- **`-ffast-math` / libmvec.** Rejected as in ADR 0009: it breaks NaN handling
  and summation guarantees.
- **Reimplement ufuncs and memmap in PHP for the FFI package.** Two
  implementations to keep in step. Element-wise PHP loops over FFI are orders
  of magnitude slower.
