# Security and production-readiness review, October 2026

Status: first remediation tranche landed on branch `harden/production-gates`; backlog below.

Prompted by the October 2026 PHP language-layer vulnerability wave. Seven reviewers covered the Zend
extension, the C kernel, the FFI binding, numerical standards (IEEE 754, reproducibility), build and supply
chain, the ML-platform contract, and the website (API, MCP, LLM surfaces). An outside advisory proposing a
separate pointer-registry library was also assessed (see the end).

## Headline finding: CI was not gating anything

Every `ci` workflow run on record failed: the runners never installed the LAPACKE headers, so each build job
stopped at `#include <lapacke.h>` and every downstream job (parity, extension, ext under ASan, preload,
release) was skipped. The sanitizer and fuzz jobs existed but had not run successfully. Separately,
`ext/libtessero` had drifted from `csrc` (missing `repro_f32.c`, `selfcheck.c`, the FITPACK sources and the
canonical-FMA matmul), so a clean-clone extension build could not link.

## Fixed in this tranche

Memory safety (each has a regression test that runs under ASan + UBSan):

| Area | Defect | Fix |
|---|---|---|
| kernel `np.einsum` | long or repeated explicit output overflowed fixed stack arrays | bounded appends, repeated labels refused |
| kernel `np.histogramdd` / `histogram2d` | bin-count product wrapped, under-allocated counts, heap overflow | checked product; bins capped at 2^40; non-finite bins refused before the cast |
| kernel integer `floor_divide` / `mod` / `divmod` | `MIN / -1` raised SIGFPE (kills the PHP worker) | `y == -1` answered without dividing (NumPy results) |
| kernel integer `negative` / `abs` / `square` | signed overflow (UB) | wrapping arithmetic, as NumPy |
| kernel routine dispatch | a short argument list let routines read past `args` | missing slots padded with "absent" |
| kernel CSR kernels and Krylov solvers | decreasing or negative `indptr`, out-of-range columns read/wrote out of bounds | structure validated per call |
| ext `NDArray::eye` | extreme `k` overflowed index math, heap write out of bounds | empty diagonals refused before the loop |
| ext `NDArray::where` | non-numeric operand produced dtype -1, out-of-bounds table read | refused; `tsr_ndarray_new` rejects invalid dtypes |
| ext nested-array scan | self-referencing array recursed until stack overflow | depth capped at 32 |
| ext `arange` | non-finite length cast to int64 (UB) | checked before the cast |
| ext `multivariateNormal` | `dim * dim` allocation before validating `cov` | rows validated first, `safe_emalloc` |
| FFI `Native\Buffer` | `unserialize()` gadget: forged payload made `__destruct` free any address | `__serialize`/`__unserialize` refuse |
| FFI `NDArray` constructor | public constructor accepted strides/offset reaching past the buffer | view extent checked against the buffer |
| FFI `CsrMatrix::fromArrays` | `indptr` not validated | must start at 0 and never decrease |
| FFI `Linalg::solveTriangular` | short `b` let LAPACK write past the heap block | row count checked |

Numerical correctness: `logsumexp` with `+inf` returned NaN (SciPy returns inf); fixed.

Build and CI:

* Compiler and linker hardening (OpenSSF guide), probed per toolchain in `csrc/Makefile` (`HARDEN=0` to
  disable) and `ext/config.m4` (`--disable-tessero-harden`): `_FORTIFY_SOURCE=3`, stack protector, stack-clash
  protection, CET (`-fcf-protection=full`) or BTI/PAC, zero-initialised automatics, RELRO, BIND_NOW,
  non-executable stack. Verified on the built `.so`: RELRO, BIND_NOW, NX stack, canary and FORTIFY symbols,
  IBT+SHSTK property note. None of these flags changes floating-point semantics.
* The extension now compiles `repro_f32.c` with `-ffp-contract=off -fno-fast-math`, as the Makefile always
  did (it previously inherited contraction `fast`), and links FITPACK and BLAS explicitly.
* CI: LAPACKE headers installed; `permissions: contents: read` by default; `persist-credentials: false`;
  readelf hardening assertions on `libtessero.so` and `tessero.so`; extension parity under ASan; a
  `repro` job proving bit-identical output across scalar/AVX2 and 1/4 threads on x86-64 and aarch64; the
  release job now needs fuzz, repro and both parity jobs, and ships `LICENSE` and `THIRD_PARTY_NOTICES.md`.
  macOS and Windows are tier 2 (no LAPACK linkage yet) and no longer block the pipeline.

## CI status after this tranche

Now green on GitHub: Linux x86-64, aarch64 and musl builds; C tests under ASan + UBSan; libFuzzer; the
`ext/libtessero` sync check; reproducible f32 on both ISAs; FPM-style preload; the reference Docker image.

The downstream jobs ran for the first time ever and expose a second layer of pre-existing problems, none caused
by these changes (the reference host passes the same suites):
* FFI parity, the PHP matrix, docs and bench on `ubuntu-22.04`: that runner's LAPACK/OpenBLAS fails
  `eigvals` ("illegal value in argument 10"), `tf2zpk` root solves, `eig_banded`, `subspace_angles`. Move these
  jobs to the runner image whose LAPACK matches the reference host, or pin OpenBLAS.
* `tessero/laravel unit tests`: the job never enables FFI (`ffi.enable=1`) or installs OpenBLAS.
* `ext-tessero under ASan + UBSan`: `make test` runs without loading the module ("Required extension missing").
* macOS and Windows: no LAPACK linkage (tier 2, non-blocking). `macos-13` was retired; the leg now uses
  `macos-15-intel` (it had left every `ci` run on `main` queued forever).

## PHP security floor (2026-10-24)

Owner decision: Tessero supports PHP 8.2 and newer, but only releases that carry the 2026-10-21 security fixes:
8.2.35, 8.3.36, 8.4.27, 8.5.12 or newer on each branch. Enforced in three layers, kept in step by
`tests/Unit/PhpSecurityTest.php`:

- install: the `php` constraint `~8.2.35 || ~8.3.36 || ~8.4.27 || ^8.5.12` in `composer.json`,
  `laravel/composer.json` and `ext/composer.json` (PIE);
- FFI: `Tessero\Native\Library::phpPatched()` runs before the library loads (`LibraryUnavailable` otherwise);
- extension: a compile-time `#error` against unpatched headers, and a MINIT check of the running PHP (patch
  releases of a branch share the extension API). On an unpatched PHP the extension logs an `E_CORE_WARNING` and
  registers no classes instead of failing MINIT, which would be a fatal startup error for the whole SAPI.

The reference Docker image pins `php:8.4.27` by digest.

Supported range: PHP 8.2 to 8.5 only. PHP 8.6 is a pre-release and is refused like an unpatched build (Composer
constraint tops out below 8.6, and both run-time checks say so); a new minor is added once it is stable and CI covers
it. Retiring 8.2 and 8.3 is tracked as TSR-208.

PHP informational advisory 2026-10-23 (CVE-2026-115560, full advisory with the next regular release): a residual RCE
vector needs `session.serialize_handler` switched to `php` at run time with `ini_set()` before `session_start()`, with
session upload progress enabled. Tessero does not use sessions, and the website uses Laravel's database session
driver, not native PHP sessions. Mitigation as php.net advises: `session.upload_progress.enabled = 0`, now set in the
reference image (`docker/tessero.ini`) and recommended for every production PHP-FPM until the next PHP release.

## Input sanitization gate (2026-10-24)

Owner directive: sanitize inputs before code executes. Every registry entry point now runs a structural gate
before dispatch (`csrc/src/fn_gate.c`, [ADR 0016](../architecture/adr/0016-input-sanitization-gate.md)): argument
kinds and flag bits, strings, array dtype, rank, dimensions, element-count and stride-extent overflow, NULL data,
elements before their block, alignment, outputs that would alias through a zero stride, and bounded sequences. No
routine runs on input it refuses, for the PHP bindings and for native callers alike. `fuzz/fuzz_abi.c` sends every
structure it classifies invalid to a random routine and fails if the call is accepted.

## Backlog (prioritised)

P0
* Fix the second-layer CI jobs above, then merge; cut a release only once CI is green.
* Add the verbatim f2c permission notice next to `f2c.h`; SHA-pin third-party actions (`softprops`,
  `shivammathur`, `msys2`) and add Dependabot for actions and composer.

P1
* `tsr_selfcheck` is never called and its gemm probe uses exactly representable inputs, so it cannot detect a
  non-FMA build. Use full-mantissa inputs, prove sensitivity in CI, and call it at MINIT / FFI load.
* Runtime FP-environment gate (MXCSR FTZ/DAZ/rounding on x86, FPCR on aarch64), checked on every OpenMP
  thread before bit-exact routines.
* Float-to-int casts of NaN/Inf/out-of-range values (cast loops, reductions stored to int) are UB: one
  saturating helper with NumPy's results.
* Cap OpenBLAS threads in the extension (`tessero.blas_threads`, default 1 outside the CLI); today only the
  FFI path caps them, so php-fpm workers oversubscribe the CPU.
* `-fvisibility=hidden` plus a version script for `libtessero.so`, a versioned soname, `tsr_abi_version()` /
  feature query, and an ABI diff in CI (Talario and Kinetix link the C ABI directly).
* Native memory budget derived from `memory_limit` by default; route the ~1000 raw `malloc` sites through a
  counted, checked allocator (`tsr_alloc_n`).
* Document `tsr_matmul` (OpenBLAS, tolerance) versus `tsr_gemm` (canonical, bit-exact) and publish a per-op
  numerical-guarantees table.
* PHP readers (`src/Io/*`): decompression caps, declared-count checks before loops, `readsav` record-pointer
  loop, stream-wrapper refusal; `TESSERO_LIB`/`TESSERO_BLAS` read with `local_only` and absolute paths only.
* New fuzz targets: einsum subscripts, histogram bins, text I/O, `.npy` header writer, NDArray
  `__unserialize`/`fromBytes`, sparse constructors.

P2
* SBOM (CycloneDX), build provenance attestation, cosign-signed checksums, CodeQL, OpenSSF Scorecard.
* Nightly TSan for the OpenMP kernels, Valgrind over the release-flag build, gcov coverage.
* Complex division (Smith's algorithm), float `floor_divide` via `npy_divmod`, NaN-propagating `norm(inf)`,
  signed-zero checks in the parity comparator, documented `long double` platform dependence.
* Thread-local error state in the C++ distribution layer; LAPACK 32-bit dimension guards.

## The outside advisory ("omni-panel")

Adopted where it maps onto Tessero's real boundaries: buffer-extent validation (the NDArray constructor check
above), checked 64-bit size arithmetic (already central in `tsr_shape_size` and `tsr_ndarray_new`, extended
here to histogram bins and CSR structure), the 64-byte alignment guarantee (already provided by `tsr_alloc`
and asserted by the C tests), determinism across threads and SIMD levels (the new `repro` CI job), and
attack-payload regression tests run under ASan in CI.

Not adopted as written: a separate `libtessero_security.so` with a fixed 1024-entry global pointer registry.
It would add another FFI surface, it is not thread-safe, it accepts any pointer that falls inside any
registered block (not the block it belongs to), its uint32 dimension limit would break large tensors, and its
64-byte alignment requirement rejects ordinary `FFI::new` buffers. Tessero already tracks every allocation by
ownership (`Native\Buffer`, the extension's object handlers), which is the right place for these checks.

## Memory ownership (ADR 0015)

Owner directive: the layer that allocates memory must own its deallocation. The contract per ABI call, the rules
per layer (kernel, OpenMP workers, extension, FFI binding) and the gates are in
[ADR 0015](../architecture/adr/0015-memory-ownership.md). In short:

* Runtime gate `-DTSR_OWNERSHIP_CHECK` (sanitizer, fuzz and ASan CI builds): `free()` of a kernel block is an
  invalid free, and `tsr_free` aborts on a foreign pointer, a double free or a wrong size. The full parity suite
  (1845 functions, both bindings), the C tests, phpt and the PHP unit suites run clean under it.
* Static gate `tools/check-ownership.sh` (with a self-test that plants one violation per rule) in CI. Run against
  the tree before the fuzz fixes it reports the `signal.stft` defect (free() of `tsr_alloc` blocks) at all four sites.
* Defects fixed: the extension did not release kind 2 (string) routine results on the failure path of a
  multi-result routine; `tsr_fn_routine` returned before initialising its result slots on argument errors; a
  thread-local in the BiasedUrn distributions kept the caller's generator after the call.
* New ABI (additive, TESSERO_0.2): `tsr_fn_results_free`, `tsr_ownership_check`.
