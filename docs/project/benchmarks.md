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

## Correctness, not just speed

Every performance figure sits on top of verified correctness, generated from recorded test runs against real
NumPy 2.4.4 / SciPy 1.17.1 — see [`docs/project/_generated/metrics.md`](_generated/metrics.md): **100.0% of the
in-scope SciPy surface and NumPy core on both backends**, 238k+ values checked per backend, cross-checked by an
independent arbitrary-precision BCMath oracle (`tools/parity/run-oracle.php`) and composed-pipeline scenario
tests (`tools/parity/run-scenarios.php`).
