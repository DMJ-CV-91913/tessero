# Gap analysis: implementation status

Each gap from the PHP scientific-computing gap analysis, what was built, and
the evidence. "Verified here" means run on the reference machine (Linux x86-64,
2 cores, PHP 8.4.21, NumPy 2.4.4, SciPy 1.17.1). "CI" means configured in
`.github/workflows/ci.yml` but not yet executed, because the reference machine
cannot reach GitHub. Status as of 0.2.0.

For the current function-by-function comparison with NumPy and SciPy, see the
[NumPy/SciPy gap analysis](numpy-scipy-gap-analysis.md). For the remaining work, see the
[open issues](open-issues.md).

## Roadmap items

| # | Gap | Acceptance target | Status | Evidence |
|---|-----|-------------------|--------|----------|
| 1 | Native kernel library and allocator | 1M-element add ≤ 2.0 ms (was 5.0 ms) | **Met**: 0.9–1.6 ms across runs (NumPy 1.0–1.3 ms on the same VM) | `bench/run.php`; 173 C checks clean under ASan+UBSan |
| 2 | LAPACK layer | numpy.linalg to 1e-10 on fixtures | **Met** at 1e-9 relative on every fixture; most agree to ~1e-15 | 23 linalg parity cases: solve, inv, det, slogdet, cholesky, qr (reduced/complete), eigh, eig, svd, pinv, lstsq, lu, solve_triangular, norms, cond, rank, stacked det |
| 3 | Indexing and broadcasting engine | Passes a port of NumPy's indexing cases | **Partly met**: slicing now parsed in C (`tsr_array_slice`) and shared by both bindings; 300 NumPy-generated slice/stride/broadcast fuzz cases pass on both; NumPy's own `test_indexing.py` has not been ported | `tests/Parity/FuzzTest.php`, `tests/Ext/ExtParityTest.php` |
| 4 | Zero-compile distribution | `composer require` on 6 platforms, no compiler | **Built, partly verified**: linux-x86_64 binary built and tested; FPM preload path verified under a non-CLI SAPI with `ffi.enable=preload`. The other 5 platform builds are CI jobs not yet run. OpenBLAS is not bundled: it comes from the OS or SciPy's wheel | `tests/e2e/preload.sh`, `bin/tessero doctor`, Makefile targets for glibc/musl/macOS/MinGW |
| 5 | Correctness and performance infrastructure | Parity, fuzzing, benchmark gates per commit | **Built**: parity generator + 258 cases, 300 fuzz cases, 22 gated benchmarks, sanitizer jobs (C and extension), JIT-mode matrix, docs-example checks. Runs locally; CI not yet executed | `tests/`, `bench/`, `.github/workflows/ci.yml` |

## After the top five

| Gap | Status |
|-----|--------|
| FFT | **Done**, own implementation rather than pocketfft (mixed radix 2/3/4/5 + generic + Bluestein). Matches numpy.fft to 1e-10 for 14 lengths including primes 17, 97, 127, 1031. About 2× slower than pocketfft. |
| Random generators | **Done**: PCG64 + SeedSequence + ziggurat normal + Lemire integers + shuffle, bit-identical to NumPy for random, uniform, normal, integers (32- and 64-bit paths), choice with p, permutation (41 stream cases, 6 seeds). exponential is inversion-based (not stream-identical); the other 35+ NumPy distributions are not implemented. |
| Sparse CSR + iterative solvers | **Done** (core): CSR from dense/triplets, matvec, matmat, transpose, Jacobi-preconditioned CG and BiCGSTAB; matches scipy.sparse. No direct sparse solver (spsolve), no CSC/COO classes, no sparse-sparse products. |
| minimize / root / curve_fit | **Done**: BFGS (SciPy's algorithm and line search), L-BFGS-B (projected L-BFGS, same optimum, different iterates), Nelder-Mead (same iterates as SciPy), brentq (bit-identical root and call count), newton/secant, vector root (damped Newton), least squares (Levenberg-Marquardt) and curve_fit. No trust-region methods, no constraints beyond bounds. |
| linprog / milp | **Done (0.2)**: dense two-phase simplex and branch and bound in C; SciPy conventions for bounds, status and marginals; 14 LP + 10 MILP parity cases against HiGHS. Dense only: large sparse models need an external solver. |
| Markov decision processes | **Done (0.2)**: value, policy, modified policy iteration and finite horizon in C on CSR transitions; OpenMP; 4 parity models. |
| Native extension | **Done (0.2)**: `ext-tessero` with operators, slicing, iteration, JSON, serialisation, solvers; 10 phpt + 439 parity cases, ASan-clean. Replaces the operator-only `tessero_ops`. |
| Laravel integration | **Done (0.2)**: `tessero/laravel` (provider, facade, casts, validation rule, Octane-safe settings). Unit-tested; the provider inside a real Laravel app is a CI job not yet run. |
| .npy / .npz | **Done**: v1/v2 read/write, C and Fortran order, numpy-written files read back exactly. Arrow not done. |

## Audit findings (section 1)

| Finding | Resolution |
|---------|-----------|
| Native memory invisible to `memory_limit` | `tsr_alloc` counts every byte; `Tessero::setMemoryBudget()` makes over-budget allocations throw `MemoryError` (tested). |
| Views outliving their parent buffer | Every view holds its `Buffer`; memory is freed in `Buffer::__destruct` exactly once; `Buffer` is uncloneable (tested: view survives parent, allocation count returns to baseline). |
| FFI::new zero-fill tax | `tsr_alloc` does not zero; `tsr_calloc` only for `zeros()`. |
| No slice syntax | `$a['1:10:2, ::-1']`, `$a['..., None, 0']`, `slice()` with ints/strings/null. |
| No operator overloading | `ext-tessero` (0.2): `do_operation` on its own arrays, and on the FFI `NDArray` through `Tessero\Ext\Operand`. Core works without it. |
| Per-element FFI access is slow | No element loops in PHP; all operations are whole-array calls. |
| Tracing JIT miscompile | Reproducer that fails 398/400 under `opcache.jit=tracing` (`laravel-sciphp/tools/jit-repro/`); report drafted in [Upstream bugs](upstream-bugs.md). Not yet reduced to a minimal script. |
| (new) Function-JIT defect | Found by the parity suite during this work: Nelder-Mead's loop counter corrupted under `opcache.jit=function`. Worked around; reproducer and draft report in [Upstream bugs](upstream-bugs.md) (issue 4). |
| FFI copy defect | Root cause isolated: pointer-arithmetic temporaries break later arithmetic on the source pointer. 10-line reproducer and draft report; a second FFI defect (void* → integer cast) found and reported the same way. Tessero never does CData arithmetic. |
| SciPHP: float64 only | 7 dtypes with NumPy 2 promotion. |
| SciPHP: broadcasting only for add | General broadcasting for every binary op, `where`, assignment. |
| SciPHP: no LAPACK/FFT/RNG | See above. |
| SciPHP: FPM falls back to pure PHP | Tessero driver in SciPHP (`SCIPHP_DRIVER=auto|tessero`) + preload script: native speed in FPM with `ffi.enable=preload`. |
| SciPHP: HTTP layer coupled to Laravel | Tessero is framework-free; SciPHP is now the Laravel layer on top. |
| SciPHP: Laravel feature tests never run | **Still open**: Packagist is unreachable from the build machine, so Laravel itself cannot be installed here. |
| Independent review | **Still open**: needs a reviewer other than the author. |

## Hardware utilisation (section 2.3)

| Item | Status |
|------|--------|
| SIMD for element-wise and reductions | GCC `target_clones` (AVX-512 / AVX2 / baseline, chosen by ifunc at load) on x86-64 glibc; compiler flags elsewhere. Transcendentals (`exp`, `log`, trig) still use scalar libm: 4–8× slower than NumPy's AVX-512 kernels. |
| Thread policy | Opt-in OpenMP (0.2) for large element-wise kernels and MDP sweeps, deterministic across thread counts; default 1 thread; OpenBLAS pinned to 1 thread in non-CLI SAPIs. |
| GPU | Not started, as the analysis recommended. |

## Known shortfalls to fix next

1. Vectorised transcendental functions (SLEEF or glibc libmvec) — `exp` is the largest remaining gap.
2. Sort: 3–6× slower than NumPy's AVX-512 quicksort; argsort is faster than NumPy's stable argsort.
3. FFT: ~2× pocketfft; real-input FFT still runs a full complex transform.
4. Run the CI matrix (5 more platforms, PHP 8.2/8.3/8.5, macOS/Windows toolchains have not compiled this code yet).
5. Complex linear algebra, complex `matmul`, more random distributions, `spsolve`.
