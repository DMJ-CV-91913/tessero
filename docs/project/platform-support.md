# Platform support

Which platforms the native kernel (`libtessero`) and the Zend extension are built and tested on, and what each
tier promises. The C ABI rules (exported symbols, soname, `tsr_abi_version`, `tsr_feature`) are in
[ADR 0014](../architecture/adr/0014-abi-stability.md).

## Tiers

| Tier | Platform (`lib/<platform>/`) | Toolchain in CI | Linear algebra | Promise |
|---|---|---|---|---|
| 1 | `linux-x86_64` (glibc >= 2.35) | GCC, Ubuntu 22.04 | LAPACKE + OpenBLAS | Built and tested on every change; a failure blocks the merge. Parity gate on both backends. |
| 1 | `linux-aarch64` (glibc >= 2.35) | GCC, Ubuntu 22.04 arm64 | LAPACKE + OpenBLAS | As above. |
| 1 | `linux-musl-x86_64` (Alpine 3.20) | GCC, musl | LAPACKE + OpenBLAS (as `libblas`) | As above. |
| 2 | `darwin-arm64`, `darwin-x86_64` | Apple clang | not linked yet | Built in CI, failures do not block. `linalg.*` needs the platform LAPACK (Accelerate), which is tracked work. |
| 2 | `windows-x86_64` | MSYS2 / MinGW | not linked yet | As above; needs an OpenBLAS/LAPACKE build for MSVC/MinGW. |

A tier-2 platform becomes tier 1 when its LAPACK linkage lands and the parity gate passes on it.

## What differs by platform

- **Exported symbols.** On Linux (GNU ld, gold, lld) a version script exports exactly the `tsr_` functions of
  `include/tessero.h` in the version node `TESSERO_0.2`; `make -C csrc abi-check` verifies it. On macOS the
  hidden-by-default build gives the same set without a version node. The MinGW build exports every non-static
  function (no `__declspec(dllexport)` annotations yet).
- **Soname.** Linux: `libtessero.so` with soname `libtessero.so.<abi>` and a `libtessero.so.<abi>` symlink next to
  it. macOS: install name `@rpath/libtessero.dylib`. Windows: `tessero.dll`.
- **SIMD dispatch.** x86-64 glibc builds with GCC >= 12 compile the hot loops for x86-64-v4, v3 and baseline and
  pick one at load time (ifunc). Other platforms use the build's `-march`. Results are identical either way.
- **CPU count.** On Linux the thread clamp of `tsr_set_threads` honours the affinity mask and the cgroup CPU quota.
  Elsewhere it uses the CPU count OpenMP reports.
- **BLAS threads.** `tsr_set_blas_threads` / `tessero.blas_threads` control OpenBLAS (or MKL) wherever the BLAS
  is found by name in the process. On Windows it currently reports that no controllable BLAS is loaded.
- **Fork handling** (`pthread_atfork`) is POSIX-only; Windows has no fork.

## Runtime requirements

- Linux: glibc >= 2.35 or musl, `liblapacke` and a BLAS (`libblas.so.3`, OpenBLAS recommended), `libgomp`.
- The C++ runtime is linked statically into `libtessero` and kept private to it.
- PHP 8.2+ for the FFI binding; the extension builds against the PHP headers of the target version (NTS and ZTS).
