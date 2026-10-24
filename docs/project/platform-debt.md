# Platform debt register

Policy: every feature is built and proven on Linux first (tier 1), then adopted on macOS and Windows (tier 2) in
the same change where feasible. When macOS or Windows hits a roadblock, the Linux work ships and the blocker is
logged here. Tier-2 CI jobs run on every change and stay visible (their failures are reported, not hidden); they
do not block a release until promoted. An entry is closed only when the matching CI job is green.

Status values: open, in progress, closed.

| ID | Platform | Area | Issue | Plan | Status |
|---|---|---|---|---|---|
| PD-1 | macOS | build | No LAPACK linkage: the build stopped at `#include <lapacke.h>` (Accelerate has LAPACK but no LAPACKE). | Homebrew OpenBLAS (LAPACK + LAPACKE + cblas), `OPENBLAS_PREFIX` override, rpath (PR #56). | in progress |
| PD-2 | Windows (MinGW) | build | No LAPACK linkage; POSIX `getline()` missing in `np_io.c`. | MSYS2 `mingw-w64-x86_64-openblas`, portable `tsr_getline` (PR #56). | in progress |
| PD-3 | Windows (MinGW) | runtime | `tessero.dll` depends on MinGW runtime DLLs (OpenBLAS, gfortran, winpthread). Windows `LoadLibrary` does not search the DLL's own directory for its dependencies. | CI bundles the DLLs next to `tessero.dll`. The FFI loader (`src/Native/Library.php`) should add the library directory with `AddDllDirectory`/`SetDllDirectory` (or load with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`) so users do not have to edit PATH. | open |
| PD-4 | Windows (MSVC) | extension | `ext/config.w32` builds the kernel with MSVC but links no LAPACKE/OpenBLAS, and the f2c FITPACK sources are untested under MSVC. | Link a prebuilt OpenBLAS for MSVC (vcpkg or the official OpenBLAS Windows release); build with the PHP Windows SDK in CI. | open |
| PD-5 | macOS | distribution | The dylib's rpath points at the Homebrew prefix, so a prebuilt binary needs Homebrew OpenBLAS at the same path. | Bundle OpenBLAS into the release (delocate-style) or document the Homebrew requirement. | open |
| PD-6 | macOS, Windows | performance | Built without OpenMP (`OPENMP ?=` empty on both), so element-wise kernels and MDP sweeps are serial; Apple clang has no `-fopenmp` without libomp. | libomp from Homebrew on macOS; MinGW `-fopenmp` on Windows; verify thread-count determinism there. | open |
| PD-7 | macOS (Apple clang) | performance | `target_clones` SIMD multiversioning needs GCC >= 12, so macOS builds take the portable path. | Evaluate per-function `__attribute__((target))` dispatch for clang, or accept the portable path on Apple silicon. | open |
| PD-8 | Windows | numerics | f2c FITPACK uses `typedef long int integer;`: 32-bit on Windows (LLP64), 64-bit on Linux/macOS. Large problems behave differently, and the 0-ULP claim is only proven on Linux. | Verify FITPACK parity on Windows in CI; consider `-DINTEGER_STAR_8`-style typedef to a fixed 32-bit int everywhere (f2c's own contract is 32-bit). | open |
| PD-9 | Windows (MSVC) | numerics | `long double` is 64-bit under MSVC (80-bit x87 on Linux x86-64, 128-bit on aarch64 Linux), so `stats_fn_tests2.cpp` and `dist_cont3.cpp` give different last bits. | Documented in the numerical-guarantees doc; replace with double-double where parity requires it. | open |
| PD-10 | macOS, Windows | CI | Parity fixtures (FFI and ext) have never run on macOS or Windows, so tier-2 numerical behaviour is unverified. | Run the FFI parity suite on the tier-2 legs once PD-1/PD-2 build (report-only until green). | open |
