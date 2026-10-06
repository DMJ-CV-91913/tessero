# Roadmap

What stands between Tessero and a credible 1.0, and what comes after it, in
priority order. Estimates are in **focused engineering sessions**: one
session is a working day of implementation, testing and documentation.
Items move to the [changelog](changelog.md) when they ship, and each row's
status is kept current. The beta work items, with IDs and acceptance criteria, are
in the [open issues](open-issues.md). Function-level coverage is in the
[NumPy/SciPy gap analysis](numpy-scipy-gap-analysis.md).

## 1. Access and people: the real gate to 1.0

These need accounts, hardware or people rather than code. Nothing else on
this page makes a 1.0 credible without them.

| Item | Why | Needs | Status |
|---|---|---|---|
| Public repository with CI running | Everything is verified on one Linux x86-64 host with PHP 8.4 | a GitHub organisation; CI minutes (the free tier is enough) | **open** |
| Packagist + PIE registration | `composer require` / `pie install` only work from registered packages | Packagist account; subtree splits for `tessero/laravel` and `tessero/tessero-ext` | **open** |
| Windows build of the extension and the kernel | The memory-map code (`csrc/src/mmap.c`) is written against the Win32 API (`CreateFileMapping`, `MapViewOfFile`, `Interlocked*`) but has never been compiled | a Windows runner (CI) | **open** |
| Production pilot | Tests prove correctness, not fitness under real load | one real workload (for example an hourly × node price cube with memmap, an LP dispatch model) | **open** |
| Independent review | One author so far | a reviewer for `csrc/` and `ext/` (C and Zend API experience); a security review before 1.0 | **open**, the main use of first sponsor funds |
| Second maintainer | Single-maintainer numerics projects in PHP have not survived | recruitment through the launch posts | **open** |

## 2. Engineering to 1.0

| # | Item | Estimate | Status |
|---|---|---:|---|
| 2.1 | **Backend parity.** Loop ufuncs and `Math` in the FFI package, `memmap` in the FFI package, `.npy` load/save/memmap in the extension, multi-axis reductions with `keepdims` in the extension | 2 | **done**: loops, mapping and `.npy` parsing moved into the kernel ([ADR 0010](../architecture/adr/0010-shared-kernels-and-vector-math.md)); a cross-backend test requires identical bytes |
| 2.2 | **Linear algebra in the extension**: load LAPACKE at runtime as the FFI package does; `solve inv det cholesky eigh svd qr lstsq` | 1–2 | planned (until then, use the FFI package alongside the extension) |
| 2.3 | **Coverage-guided fuzzing** (libFuzzer) of the slice parser, `.npy` header parser, memmap planning and the JSON writer, with a corpus in the repository and CI jobs | 1 | **done**: 4 targets; 60 s per push and 30 min nightly per target. Two slice-parser bugs found and fixed. |
| 2.4 | **Property-based tests**: algebraic identities and round trips on random shapes, dtypes and strides, across both backends | 1 | **done**: `PropertyTest` (5 properties × 40 cases, ×2000 nightly) and `BackendParityTest` (> 3 000 cross-backend comparisons) |
| 2.5 | **Soak tests**: a harness that runs mixed workloads for hours in one process and fails if native memory, mapped bytes, heap, RSS or descriptors drift | 0.5 (+ wall-clock) | **done** (`tools/soak.php`, 5.5 h nightly per backend); 24–72 h runs are a release-checklist step, not yet performed |
| 2.6 | **Vectorised transcendentals** | 2 | **exp/log done**: branch-free fdlibm-based kernels, < 1 ulp; `exp` 7.6× faster (1.6× NumPy with AVX-512), `log` 1.7× NumPy. **Next**: `sin`/`cos`/`tan` with a Payne–Hanek reduction (1 session) |
| 2.7 | **Real-input FFT** of half length | 0.5 | **done**: `tsr_rfft`/`tsr_irfft`, about 1.6× faster than the complex FFT of the same data. The complex FFT is still ~2× slower than NumPy's pocketfft (2.7b: radix-4/split-radix, 1–2 sessions) |
| 2.8 | **Faster sort** | 1–2 | **partly done**: MSD+LSD hybrid radix with parallel buckets; 1M float64 is 30 ms against NumPy's 7.5 ms (AVX-512 quicksort). **Next**: a SIMD sorting network for small buckets |
| 2.9 | **Sparse direct solver** (`spsolve`): sparse LU with fill-reducing ordering, BSD-licensed | 2–3 | planned |
| 2.10 | **HiGHS backend** (MIT) for large sparse LP/MILP, loaded at runtime when installed, same result objects as `linprog`/`milp` | 1–2 | planned; needs HiGHS available to build and test against |
| 2.11 | API review and freeze, deprecation policy, 1.0 docs pass | 1 | after the above |

Remaining to 1.0: about **9–13 sessions** of engineering, in parallel with
section 1.

## 3. After 1.0: the most useful additions from the Python ecosystem

Only the libraries that PHP lacks, that fit Tessero's C kernel, and that
matter for the planning, forecasting, pricing and data workloads PHP
applications actually run. Each would be checked against the Python
original with generated fixtures, as the rest of Tessero is.

| Priority | From Python | What Tessero would offer | Estimate |
|---:|---|---|---:|
| 1 | **scipy.stats** | continuous and discrete distributions (pdf, cdf, ppf, rvs), descriptive statistics, quantiles, histograms, the common hypothesis tests | 4–5 |
| 2 | **scipy.interpolate** | 1-D/2-D interpolation, cubic and PCHIP splines, smoothing splines: curve shaping, filling gaps in time series | 2–3 |
| 3 | **PuLP / Pyomo-style modelling + HiGHS** | build LP/MILP models with variables and expressions instead of matrices; solve with the built-in simplex or HiGHS | 3–4 |
| 4 | **pyarrow (Parquet / Arrow IPC)** | read and write columnar files with zero-copy into arrays: interoperability with data lakes and Python pipelines | 3–4 |
| 5 | **statsmodels (time series + OLS/GLM)** | regression with standard errors and p-values; ARIMA/SARIMAX, exponential smoothing, seasonal decomposition | 5–6 |
| 6 | **scipy.signal** | convolution, FIR/IIR filters, Welch spectra, resampling, peak finding (builds on the FFT) | 2–3 |

Deliberately left out: a scikit-learn clone (Rubix ML already serves PHP;
Tessero should become a fast backend for it instead), model training in general
(run Python-trained models through ONNX Runtime), and GPU support.

## Status of earlier items

The platform, native extension, solvers, Laravel package, documentation, ufuncs
and memory-mapped arrays are done; see the [changelog](changelog.md) and the
[gap-analysis status](gap-closure.md).
