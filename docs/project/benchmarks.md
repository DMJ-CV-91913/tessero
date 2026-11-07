# Benchmarks

Tessero's two backends (the FFI backend `Tessero\*` and the native extension `Tessero\Ext\*`) share one C
kernel. Dense linear algebra routes through OpenBLAS on both. The numbers below are reproducible; absolute
GFLOPS depend on the host and on how many threads OpenBLAS is given, so the headline figure is a **controlled
ratio measured back-to-back in the same process**, which is independent of machine load.

## Matrix multiply: OpenBLAS vs the portable fallback

The kernel ships a portable, cache-blocked `gemm` for environments without a BLAS library. `tsr_matmul`'s
float path now calls `cblas_dgemm`/`cblas_sgemm` (OpenBLAS) directly when available, with the blocked kernel
kept as the fallback (and for integer/overflow cases). The speedup from using OpenBLAS, measured by running
both paths on the same 1024×1024 double-precision matrices in one process (median of 3, Intel Core i7‑4790,
Haswell, 4 cores / 8 threads):

| OpenBLAS threads | fallback blocked gemm | OpenBLAS `cblas_dgemm` | speedup |
|---|---|---|---|
| 1 | 4.7 GFLOPS | 42.2 GFLOPS | **9.0×** |
| 4 | 4.6 GFLOPS | 78.2 GFLOPS | **16.9×** |

The fallback is single-threaded (flat ~4.6 GFLOPS); OpenBLAS scales with cores, so the ratio grows from ~9×
(single-threaded) to ~17× (4 threads). With this change the native extension backend matches the FFI backend
on matmul instead of trailing it.

Reproduce:

```sh
gcc bench/gemm_openblas.c -O3 -std=gnu11 -mavx2 -fopenmp -o /tmp/gemm_bench -lblas -lm
OPENBLAS_NUM_THREADS=4 /tmp/gemm_bench 1024
```

## Backend comparison (PHP level)

`bench/backends.php` times the two Tessero backends on matmul, elementwise ufuncs and reductions. On the same
host (OpenBLAS ×4), FFI and ext are within noise of each other on all operations; memory-bound ops (`add`,
`sum`) are limited by bandwidth, compute-bound `matmul` by OpenBLAS.

## Tessero in a PHP app vs NumPy in a Python app

The same cases, timed the way an application sees them: `bench/run.php` in PHP (FFI backend) against
`bench/numpy_bench.py` in Python, median of repeated runs, one thread for both (`OPENBLAS_NUM_THREADS=1`). Intel Core
i7-4790 (Haswell, AVX2), PHP 8.5.12, NumPy 2.4.4 / SciPy 1.17.1. A ratio below 1 means Tessero is faster.

| case | Tessero ms | NumPy ms | ratio |
|---|---|---|---|
| exp 1M / log 1M | 4.18 / 4.32 | 5.53 / 5.21 | 0.76 / 0.83 |
| PHP list -> array 1M (Python list -> array) | 21.1 | 34.3 | 0.61 |
| add, multiply, sqrt, transpose, boolean mask 1M | | | 0.93 - 1.04 |
| rng normal 1M | 10.8 | 11.5 | 0.95 |
| matmul 600x600 / svd 300x300 | 9.01 / 22.3 | 8.94 / 21.9 | 1.01 / 1.02 |
| solve / cholesky 600x600 (separate run) | 5.19 / 4.55 | 5.70 / 6.05 | 0.91 / 0.75 |
| solve / cholesky 2000x2000 (separate run) | 140 / 101 | 143 / 101 | 0.98 / 1.00 |
| array -> PHP list 1M | 28.9 | 24.6 | 1.18 |
| fft 2^20 (real input, as the bench passes it) | 49.5 | 52.4 | 0.94 |
| fft / ifft 2^20 complex input | 51.0 / 50.6 | 47.4 / 48.3 | 1.08 / 1.05 |
| var / std 1M | 1.1 | 2.9 | 0.38 |
| sum 1M | 0.48 | 0.35 | 1.38 |
| rfft / irfft 2^20 | 34.0 / 28.0 | 25.7 / 27.1 | 1.32 / 1.03 |
| fft 1,000,003 (prime, Bluestein) | 468 | 261 | 1.80 |
| sort 1M (re-timed with `bench/sort_bench.*` after `sort()` became the AVX2 sort) | 17.1 | 13.7 | 1.25 |

The fft, rfft and var rows were measured in a second run (`fft` and `rfft` with and without the copy removal, `var`
with and without its row buffer); the other rows come from the full bench run. Geometric mean over the 21 shared
cases: **1.10x** NumPy's time after the first round (1.21x before the plan cache, the scalar-parameter draw loop, the
single-pass mean and the AVX2 min/max), before the FFT copy removal. Outside the bench set: `mean` 1M 0.39 ms (NumPy
0.34), `max` 1M 0.33 ms (0.26), `exponential` 1M 5.4 ms (6.45).

A 2^20 complex FFT spends about 35 ms in butterfly arithmetic and 16 ms in the recursion's strided reads of its input
(a sequential copy of the same 16 MB takes 1.6 ms); the time around the kernel call is now under a millisecond.

Reductions to a scalar on small arrays (FFI backend, per call): `sum` of 10 values 1.9 us (NumPy 1.0), `max` 1.9 us
(1.1), `mean` 1.6 us (2.7), `var` 1.7 us (9.1).

Bluestein lengths: a repeated 1,000,003-point `fft` takes about 220 ms (NumPy 260) under the CLI's 256 MiB FFT cache;
under a web SAPI's 32 MiB default the plan (about 112 MB) is rebuilt per call, about 490 ms.

Known gaps, largest first: `cholesky` of small matrices (100x100: 0.16 ms against 0.05, the triangle masking), `rfft` (1.3x),
and `sum` / `max` of 1M values (1.1-1.25x).

Reproduce:

```sh
OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 bench/run.php --json --no-gates > tessero.json
OPENBLAS_NUM_THREADS=1 python3 bench/numpy_bench.py --json > numpy.json
```

## Tessero in a PHP app vs SciPy in a Python app

The same machine and conditions as the NumPy table: Intel Core i7-4790 (Haswell, AVX2), one thread
(`OPENBLAS_NUM_THREADS=1`), PHP 8.5.12 (FFI backend) against Python with SciPy 1.17.1 / NumPy 2.4.4, median of
repeated runs (the optimize cases repeat each call 200 times). Both sides build bit-identical inputs from the same
seeds: Tessero's generator reproduces NumPy's stream. The optimize cases use each environment's natural code: a
PHP closure for the objective in Tessero, `scipy.optimize.rosen` (vectorised NumPy) in SciPy. Measured 2026-11-05 on
0.3.0. A ratio below 1 means Tessero is faster.

| case | Tessero ms | SciPy ms | ratio |
|---|---|---|---|
| optimize.minimize rosen BFGS n=5 | 0.333 | 10.3 | 0.03 |
| optimize.minimize_scalar brent | 0.00285 | 0.0331 | 0.09 |
| optimize.brentq | 0.00287 | 0.00965 | 0.30 |
| optimize.linprog 200x200 dense (SciPy: HiGHS) | 9.14 | 25.7 | 0.36 |
| special.logsumexp 1M | 8.11 | 21.4 | 0.38 |
| stats.zscore 1M | 10.1 | 14.3 | 0.70 |
| stats.describe 1M | 31.1 | 41.2 | 0.76 |
| stats.ttest_ind 100k vs 100k | 1.18 | 1.45 | 0.81 |
| stats.pearsonr 1M | 25.4 | 30.6 | 0.83 |
| linalg.solve 600x600 | 6.2 | 6.63 | 0.94 |
| special.jv(1.5, x) 1M | 751 | 779 | 0.96 |
| linalg.svd 300x300 | 22.4 | 22.1 | 1.01 |
| linalg.inv 600x600 | 18.2 | 17.8 | 1.03 |
| linalg.eigh 300x300 | 12.2 | 10.8 | 1.12 |
| stats.norm.cdf 1M | 47.2 | 40.6 | 1.16 |
| spatial.distance.pdist 2000x3 | 8.54 | 7.25 | 1.18 |
| spatial.distance.cdist 1000x1000x3 | 4.29 | 3.59 | 1.20 |
| linalg.lu_factor 600x600 | 6.54 | 5.37 | 1.22 |
| signal.convolve 10k * 101 | 0.355 | 0.282 | 1.26 |
| linalg.det 600x600 | 6.34 | 4.94 | 1.28 |
| special.erf 1M | 25.2 | 18.9 | 1.33 |
| special.expit 1M | 10.4 | 7.84 | 1.33 |
| special.gamma 1M | 24.4 | 17.0 | 1.44 |
| signal.lfilter order 2, 1M | 9.2 | 4.5 | 2.04 |
| ndimage.uniform_filter 1000x1000 | 31.5 | 12.7 | 2.48 |
| ndimage.gaussian_filter 1000x1000 | 68.1 | 16.8 | 4.05 |
| signal.fftconvolve 100k * 1001 | 250 | 3.51 | 71.4 |

Both sides return the same answers: `linprog` reaches the same optimum to 12 digits (HiGHS takes 83 iterations,
Tessero's simplex 109); BFGS stops at the same point after the same 25 iterations (183 function evaluations against
SciPy's 180), though Tessero reports a precision-loss status where SciPy reports success, an open discrepancy.
Known gaps: `signal.fftconvolve` transforms at the exact length (101,000 = 2^3 5^3 101 here, a Bluestein transform)
where SciPy pads to a fast length; the `ndimage` filters and `lfilter` are scalar loops; `lu_factor` and `det` still
pay LAPACKE's row-major copies (the fix used for `solve` applies); `linalg.expm` does not yet handle matrices with
complex eigenvalues (a general random matrix raises), so it is not in the table.

Reproduce: the scripts are in `bench/` (`scipy_bench.php`, `scipy_bench.py`), run under the same conditions.

## Choosing a sort: sort(), sortSimd(), sortRuns()

Three sorts with the same result (C: `tsr_sort`, `tsr_sort_simd`, `tsr_sort_runs`; PHP: `NDArray::sort()`,
`sortSimd()`, `sortRuns()`, on both backends). All three order the values by the same total order (NaN last, -0.0
before +0.0) and by comparison only, so the output is byte-identical whatever the algorithm, the CPU's SIMD level,
the thread count, the alignment of the input or the CPU's denormals-are-zero mode; the test suite checks this for
every dtype, including infinities, NaN payloads, subnormals and the integer extremes. `sort()` is the default and
takes the AVX2 quicksort of `sortSimd()` where the CPU has AVX2 (the radix sort elsewhere); the choice is purely about
speed:

| function | algorithm | strengths | weaknesses |
|---|---|---|---|
| `sort()` | the default: `sortSimd()` where the CPU has AVX2; otherwise an LSD radix sort on 64-bit keys in a scratch buffer | the fastest available on the machine, without choosing | -- |
| `sortSimd()` | AVX2 quicksort in place at the data's own width (8 float32/int32 or 4 float64/int64 values per vector), in-register sorting networks for the small pieces, no scratch buffer | the fastest of the three on almost every input | needs AVX2 (without it, it runs the radix sort); time depends on the input |
| `sortRuns()` | natural (run-adaptive) merge sort | mostly sorted data (values appended to a sorted array, a few edited values), also without AVX2 | 2-3x slower than the others on random data |

Milliseconds, float64, one thread, median of repeated runs through the C library (same machine as above, while
another benchmark was running, so treat small differences as noise; the fastest is in bold):

| data, n = 1,000,000 | sort | sortSimd | sortRuns |
|---|---|---|---|
| random | 32.0 | **16.1** | 75.7 |
| 10 distinct values | 20.9 | **5.1** | 41.0 |
| sorted / reversed | 25.0 / 25.5 | **1.5 / 1.9** | 3.0 / 3.7 |
| sorted, 1% of values moved | 29.4 | **14.3** | 26.1 |
| sorted, 1% random values appended | 31.3 | **14.8** | 20.3 |
| interleaved (two sorted sequences) | 23.9 | **13.8** | 33.8 |
| integer-valued doubles (1000 values) | 20.0 | **9.5** | 57.5 |

| data, n = 100,000 | sort | sortSimd | sortRuns |
|---|---|---|---|
| random | 3.55 | **1.30** | 5.85 |
| 10 distinct values | 1.85 | **0.40** | 2.81 |
| sorted / reversed | 2.13 / 2.52 | **0.09 / 0.12** | 0.21 / 0.25 |
| sorted, 1% of values moved | 2.55 | **1.13** | 1.40 |
| sorted, 1% random values appended | 2.69 | 1.20 | **1.06** |

Against NumPy, 1,000,000 random values, one thread (NumPy 2.4.4 `np.sort`, which on this AVX2 machine dispatches to
Intel's x86-simd-sort vector quicksort):

| dtype | NumPy | sort | sortSimd |
|---|---|---|---|
| float64 | **14.1** | 31.9 | 15.9 |
| float32 | **7.5** | 33.6 | 7.8 |
| int64 | 17.1 | 24.3 | **16.7** |
| int32 | **6.8** | 24.0 | 7.4 |

`sortSimd()` also uses the threads set with `Tessero::setThreads()` (extension: `Engine::setMaxThreads()` or
the `tessero.threads` ini setting) for arrays of 131,072 values or more; NumPy's sort is single-threaded. The
partitions near the top of the sort become tasks for the other threads, and the pivots do not depend on the thread
count, so the output is the one-thread output, byte for byte (tested at 1 and 4 threads). Random values, same
machine (4 cores, 8 hardware threads; a separate benchmark occupied one core during these runs):

| dtype, 1,000,000 values | NumPy | sortSimd 1 thread | 2 threads | 4 threads |
|---|---|---|---|---|
| float64 | 14.1 | 15.9 | 9.0 | **5.7** |
| float32 | 7.5 | 7.9 | 4.5 | **3.4** |
| int64 | 17.1 | 16.7 | 9.4 | **6.5** |
| int32 | 6.8 | 7.5 | 4.1 | **2.8** |

| dtype, 10,000,000 values | NumPy | sortSimd 1 thread | 4 threads |
|---|---|---|---|
| float64 | 176 | 194 | **90** |
| float32 | 92 | 98 | **48** |
| int64 | 206 | 200 | **86** |
| int32 | 84 | 90 | **41** |

At 1M values four threads sort 2.2-2.6x faster than NumPy; at 10M the gain is about 2x, as the sort becomes
limited by memory bandwidth. Eight threads (hyper-threads) add nothing on this machine.

How the three work, why their outputs are identical, and the risk register with its tests:
[sorting design](sort-design.md).

Rule of thumb: `sort()`; `sortRuns()` for mostly sorted data where AVX2 is missing. The `sort` columns in the tables
above are the radix sort, measured before `sort()` became the AVX2 sort; it is now the path of CPUs without AVX2.

## Full sweep

Every function family timed against NumPy/SciPy on the same inputs, single-threaded (2026-11-06); all the tables,
the gaps that remain and how to reproduce: [speed sweep](speed-sweep.md).

| tier | timed on both sides | geomean Tessero / reference | Tessero faster |
|---|---|---|---|
| `scipy.special` element-wise, 100,000 values | 212 functions | 1.05 | 31 |
| heavier calls at realistic sizes (re-timed after the fixes) | 84 cases | 1.07 | 39 |
| per-call latency, every parity fixture call | 1,015 functions | 0.62 | 615 |

## Correctness, not just speed

Every performance figure sits on top of verified correctness, generated from recorded test runs against real
NumPy 2.4.4 / SciPy 1.17.1 — see [`docs/project/_generated/metrics.md`](_generated/metrics.md): **100.0% of the
in-scope SciPy surface and NumPy core on both backends**, 238k+ values checked per backend, cross-checked by an
independent arbitrary-precision BCMath oracle (`tools/parity/run-oracle.php`) and composed-pipeline scenario
tests (`tools/parity/run-scenarios.php`).
