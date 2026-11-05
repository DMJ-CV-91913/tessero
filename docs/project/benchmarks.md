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
| solve / cholesky 600x600 | 6.65 / 4.66 | 5.79 / 4.02 | 1.15 / 1.16 |
| array -> PHP list 1M | 28.9 | 24.6 | 1.18 |
| fft 2^20 (real input, as the bench passes it) | 49.5 | 52.4 | 0.94 |
| fft / ifft 2^20 complex input | 51.0 / 50.6 | 47.4 / 48.3 | 1.08 / 1.05 |
| var / std 1M | 1.1 | 2.9 | 0.38 |
| sum 1M | 0.48 | 0.35 | 1.38 |
| rfft / irfft 2^20 | 34.0 / 28.0 | 25.7 / 27.1 | 1.32 / 1.03 |
| fft 1,000,003 (prime, Bluestein) | 468 | 261 | 1.80 |
| sort 1M | 31.4 | 13.6 | 2.31 |

The fft, rfft and var rows were measured in a second run (`fft` and `rfft` with and without the copy removal, `var`
with and without its row buffer); the other rows come from the full bench run. Geometric mean over the 21 shared
cases: **1.10x** NumPy's time after the first round (1.21x before the plan cache, the scalar-parameter draw loop, the
single-pass mean and the AVX2 min/max), before the FFT copy removal. Outside the bench set: `mean` 1M 0.39 ms (NumPy
0.34), `max` 1M 0.33 ms (0.26), `exponential` 1M 5.4 ms (6.45).

A 2^20 complex FFT spends about 35 ms in butterfly arithmetic and 16 ms in the recursion's strided reads of its input
(a sequential copy of the same 16 MB takes 1.6 ms); the time around the kernel call is now under a millisecond.

Reductions to a scalar on small arrays (FFI backend, per call): `sum` of 10 values 1.9 us (NumPy 1.0), `max` 1.9 us
(1.1), `mean` 1.6 us (2.7), `var` 1.7 us (9.1).

Known gaps, largest first: `sort` (radix sort with a fixed cost of about 26 ms even on sorted input, against NumPy's
AVX2 quicksort), Bluestein lengths (their plans, about 112 MB at 1,000,003 points, exceed the 64 MiB FFT cache: 490 ms
per call against NumPy's 260; with a 256 MiB cap a repeated call takes about 220 ms), `rfft` (1.3x), and `sum` / `max`
of 1M values (1.1-1.25x).

Reproduce:

```sh
OPENBLAS_NUM_THREADS=1 php -d ffi.enable=1 bench/run.php --json --no-gates > tessero.json
OPENBLAS_NUM_THREADS=1 python3 bench/numpy_bench.py --json > numpy.json
```

## Correctness, not just speed

Every performance figure sits on top of verified correctness, generated from recorded test runs against real
NumPy 2.4.4 / SciPy 1.17.1 — see [`docs/project/_generated/metrics.md`](_generated/metrics.md): **100.0% of the
in-scope SciPy surface and NumPy core on both backends**, 238k+ values checked per backend, cross-checked by an
independent arbitrary-precision BCMath oracle (`tools/parity/run-oracle.php`) and composed-pipeline scenario
tests (`tools/parity/run-scenarios.php`).
