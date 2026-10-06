# Numerical methods

This page records the algorithm behind each numerical kernel, why it was chosen,
and how its accuracy is checked. Solvers (LP, MILP, MDP, sparse iterative,
optimisation) are on [Solvers](solvers.md).

## Summation and moments

| Operation | Method | Reason |
|---|---|---|
| `sum`, `mean` | pairwise summation: runs of ≤ 128 elements summed with 8 accumulators, halves combined recursively | error O(log n · ε) instead of O(n · ε); the same approach as NumPy, so results agree to a few ulps |
| `mean`, `var`, `std` | two passes, as NumPy: mean = pairwise sum / n, then the pairwise sum of squared deviations | stable when mean ≫ spread (unlike E[x²]−E[x]²) and bit-identical to NumPy on contiguous rows; a one-pass Welford update, used before 0.3, drifted by an ulp |
| integer `sum`/`prod` | int64 accumulation with two's-complement wrap | NumPy semantics; no undefined behaviour |
| `cumsum`/`cumprod` | sequential | NumPy semantics (cumulative sums are not pairwise in NumPy either) |
| reductions along a non-last axis | `tsr_reduce_mid` accumulates whole rows (outer × len × inner) | contiguous inner loops instead of strided gathers |

## Sorting

Floats are mapped to order-preserving unsigned keys (the sign-flip trick).
NaNs are moved to the end first, as NumPy orders them.

- **`sort`** (values; stability is invisible) uses an MSD + LSD hybrid radix
  sort.
  - One most-significant 11-bit pass scatters the keys into up to 2048
    buckets. A constant digit is skipped and the next one is used.
  - Each bucket is then small enough for cache and is finished by 8-bit LSD
    passes (constant digits skipped), or by insertion sort below 32 keys.
  - Buckets are independent, so with `threads > 1` they are sorted in
    parallel.
- **`argsort`** uses a stable LSD radix sort with 11-bit digits (3 passes for
  32-bit keys, 6 for 64-bit), so it is stable like NumPy's `kind='stable'`.
  −0.0 and +0.0 keep their input order.

Trade-off: O(n) and branch-free, but memory-bound. On 1M random doubles `sort`
takes about 30 ms single-threaded against NumPy's 7.5 ms with its AVX-512
quicksort. A SIMD sorting network for the per-bucket step is on the roadmap.
`argsort` is on par with NumPy's stable argsort.

## Vectorised exp and log

glibc's `exp` and `log` are accurate but scalar. `csrc/src/vmath.h` has
branch-free versions that the compiler vectorises (SSE2, AVX2, AVX-512 via
`target_clones`) without `-ffast-math`:

- **exp**: Cody–Waite reduction x = k ln2 + r, fdlibm's rational minimax
  approximation on |r| ≤ ln2/2, and scaling by 2^k in two steps so gradual
  underflow and the top of the range are exact.
- **log**: the input is split as m · 2^k with m in [√2/2, √2) using integer
  bit operations. Then fdlibm's s = f/(2+f) series with its seven minimax
  coefficients; both of fdlibm's final formulas are computed and blended.
  Subnormal inputs are pre-scaled by 2^54.
- Special values (NaN, ±∞, ±0, overflow, underflow, negative arguments) are
  blended in at the end with bit masks, so there is no branch in the loop.
- float32 uses the float64 kernel and rounds once.

Measured error is below 1 ulp (0.90 for exp, 0.88 for log, over 2×10^7
points against long double). The C tests check this on every build. Results
can differ from glibc's in the last bit, as NumPy's SIMD versions do.

## FFT

Iterative mixed-radix Cooley–Tukey with dedicated radix-2, 3, 4 and 5
butterflies and a generic odd-radix butterfly. Twiddles are precomputed per
level into contiguous tables. Lengths with a prime factor above 5 that would
make the generic butterfly quadratic use Bluestein's chirp-z algorithm on a
power-of-two convolution, so every length is O(n log n).

Accuracy: twiddles are computed directly as `cos(2πk/n)`, `sin(2πk/n)` for
every k (no recurrence), and the C tests compare against a long-double
direct DFT. Parity with `numpy.fft` is 1e-10 relative on 14 lengths
including primes.

Real input (`rfft`, `irfft`): for even n the n samples are packed as n/2
complex values, transformed with one complex FFT of half the length, and
split into the spectra of the even and odd samples with one twiddle per
output bin. The twiddles come from a first-octant table by symmetry. The
result matches the full complex transform to 1e-11 in the C tests and
`numpy.fft.rfft` to 1e-15 on the parity cases. Odd n uses the complex
transform.

## Random numbers

| Component | Algorithm | Matches NumPy |
|---|---|---|
| Seeding | SeedSequence (hash-mixing of entropy words into the 256-bit state) | bit-exact |
| Generator | PCG64 (128-bit LCG, XSL-RR output) | bit-exact |
| `random` | 53-bit mantissa from the top bits (`(x >> 11) * 2⁻⁵³`) | bit-exact |
| `normal` | ziggurat with NumPy's 256-layer tables (`ziggurat.h`) | bit-exact |
| `integers` | Lemire's nearly-divisionless method, 32- and 64-bit paths as NumPy selects them | bit-exact |
| `choice(p=…)` | CDF + binary search on `random()` draws | bit-exact |
| `permutation`, `shuffle` | Fisher–Yates with NumPy's bounded integer draws | bit-exact |
| `exponential` | inversion `−scale · log1p(−u)` | **not** stream-identical (NumPy uses a ziggurat) |

41 stream cases across 6 seeds are compared value by value against NumPy.

## Matrix products

With OpenBLAS: `dgemm`/`sgemm` (and batched loops for stacked inputs).
Without it: a cache-blocked kernel with FMA for float types and wrapping
arithmetic for int64.

## Linear algebra

LAPACK via LAPACKE row-major entry points: `dgesv` (solve), `dgetrf`/`dgetri`
(inv, det via LU), `dpotrf` (Cholesky), `dsyevd` (eigh), `dgeev` (eig),
`dgesdd` (svd), `dgeqrf`/`dorgqr` (qr), `dgelsd` (lstsq, SVD-based, robust to
rank deficiency), `dtrtrs` (triangular solves). Stacked inputs loop over
matrices in C-contiguous copies.

## Floating-point environment

Kernels are compiled with `-O3 -fno-math-errno` and **without**
`-ffast-math`. Floating-point semantics (NaN propagation, signed zeros,
IEEE rounding) are preserved, and the compiler may not reassociate sums, which
would break the pairwise-summation guarantees and cross-thread determinism.
GCC's default contraction mode under `-std=gnu11` (`-ffp-contract=fast`) lets
the AVX2 and AVX-512 clones fuse a multiply and an add into one FMA. This is why
the last bit of some results can differ between CPU generations (see
[Troubleshooting](../operations/troubleshooting.md#results-differ-slightly-between-machines)).
On any one machine, results are deterministic.

## JSON number formatting

Floats are written in the shortest form that round-trips: 15 significant
digits are tried first, then 16, then 17, and the first that parses back to the
same double is used. This matches PHP's `serialize_precision = -1` and
Python's `repr`. Integer-valued floats get a fast integer path. The extension
uses PHP's own `php_gcvt` so its output is byte-identical to `json_encode`.
