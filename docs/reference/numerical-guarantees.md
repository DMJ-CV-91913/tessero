# Numerical guarantees

What Tessero promises about the bits it produces, in three tiers, and the semantics of the edge cases
(integer overflow, float to int conversion, NaN, signed zero) that every tier shares. The reference for
"correct" is NumPy 2.4.4 and SciPy 1.17.1, the versions the parity gate pins.

## The three tiers

### Tier A: bit-exact everywhere

The same inputs give the same output bits on every supported CPU (x86-64 baseline, AVX2, AVX-512, ARM64
NEON/SVE), every SIMD clone, every thread count and every conforming compiler, given a canonical
floating-point environment (see below). This holds by construction: these functions use only the
correctly rounded IEEE operations (`+ - *` and a round-once `fmaf`/`fma`), in one fixed evaluation order,
with no reduction whose order depends on SIMD width or threads.

- The reproducible-f32 primitives and composites: `tsr_repro_{rsqrt,exp2,exp,erf,recip,tanh}_f32`,
  `tsr_repro_{sigmoid,gelu_tanh,softmax,rmsnorm,layernorm}_f32` and the backward composites. Their
  definitions and SHA-256 goldens are in `csrc/repro/GOLDENS.md`; CPU and GPU (Vulkan) agree bit for bit.
- `tsr_gemm` (F32/F64), `tsr_axpy` and `tsr_dot_f64`: the accumulation is an explicit round-once FMA in a
  fixed i/p/j order (`TSR_RFMA`), so compiler contraction settings cannot change it.

The guarantee is enforced, not assumed:

- `tsr_selfcheck()` runs a known-answer GEMM (8 x 67 x 8, full-mantissa inputs, exponents 2^-8..2^8, both
  signs) through the dispatched `tsr_gemm` and a no-clone reference and compares both with a frozen digest.
  The inputs are chosen so that a non-FMA build, a reordered sum, a double accumulator and OpenBLAS each
  produce a different digest (the C tests prove this against deliberately wrong variants).
- `tsr_fpenv_check()` reads the FP control register (x86-64 MXCSR, aarch64 FPCR) and `fegetround()` on the
  calling thread and on every OpenMP worker. The extension runs the thread-local check at load and warns
  (it does not refuse to load); `Tessero\Ext\Engine::selfcheck()` runs the full check plus the KAT.
  `tsr_fpenv_canonicalize()` is the explicit repair; it is never applied silently.

A canonical environment means: round to nearest even, no flush-to-zero (FTZ), no denormals-are-zero (DAZ),
all FP exceptions masked. The usual way to lose it is linking anything built with `-ffast-math` (its
startup code sets FTZ/DAZ for the whole process).

### Tier B: deterministic per build and machine

Everything else. The same binary on the same CPU, with the same libm, the same BLAS/LAPACK and the same
thread settings, gives the same bits on every run: no kernel reduction depends on scheduling (parallel
element-wise loops split into fixed static chunks, and reductions do not split across threads).
Across machines or builds the low bits may differ, because:

- libm transcendentals (`exp`, `log`, `sin`, `pow`, ...) are only faithfully rounded and differ between
  glibc, musl, macOS and Windows runtimes, and between glibc versions.
- `tsr_matmul` (the `matmul` / `dot` / `@` path) calls OpenBLAS `cblas_dgemm` / `cblas_sgemm` whenever every
  dimension fits a C `int`. OpenBLAS picks its kernel and blocking per CPU, so its results are only
  tolerance-equal across machines. Use `tsr_gemm` when bit-exactness is required: it is the canonical,
  tier A product (and slower).
- LAPACK routines (`numpy.linalg`, `scipy.linalg`) come from the system LAPACK/OpenBLAS: tolerance-equal
  across implementations, and decompositions with sign or ordering freedom (eigenvectors, SVD) are
  canonicalised where NumPy's own output is canonical, otherwise compared by invariants.
- Supported LAPACK is 3.11 or newer. LAPACK 3.10 (Ubuntu 22.04) runs safely; four functions differ there in
  eigenvector sign or last digits (see the installation guide), and CI checks that nothing else does.
- Code without explicit FMA may be contracted differently by different compilers or targets (GCC
  contracts `a * b + c` by default on targets with FMA instructions).
- `long double` is 80-bit x87 extended precision on x86-64 Linux, 128-bit quad on aarch64 Linux and plain
  `double` on MSVC and Apple ARM64. The few places that use it (the Kolmogorov-Smirnov distribution in
  `csrc/cxx/stats_fn_tests2.cpp`, a high-order recurrence in `csrc/cxx/dist_cont3.cpp`, test references in
  `csrc/src/vmath.h`) follow SciPy, which has the same platform dependence.

### Tier C: tolerance against NumPy / SciPy

Every function in the parity gate (`tools/parity/run-tests.php`) matches pinned NumPy 2.4.4 / SciPy
1.17.1 within the tolerances recorded per function in the fixtures (many are exact). A tolerance is never
loosened to make a test pass.

## Edge-case semantics (all tiers)

### Integer arithmetic wraps

Integer `+ - *`, negation, `abs`, `square`, cumulative sums/products and integer sums/products wrap
around in two's complement, as NumPy integers do (`abs(INT64_MIN) == INT64_MIN`). Integer floor division
and remainder by zero give 0; `MIN // -1` gives the wrapped `MIN` and `MIN % -1` gives 0 (C would trap).
Integer sums and products accumulate in int64.

### Float to integer conversion saturates like x86-64

C leaves the conversion of NaN, infinity and out-of-range values undefined. Tessero defines it as the
value NumPy produces on x86-64 (the `cvttsd2si` "integer indefinite"):

| source value | to int64 | to int32 | to uint8 |
|---|---|---|---|
| in range | truncation toward zero | truncation toward zero | low byte of the int32 truncation (300.0 -> 44, -1.0 -> 255) |
| NaN, +-Inf, out of range | INT64_MIN | INT32_MIN | 0 (the low byte of INT32_MIN) |

`astype`, complex-to-integer (real part), reductions stored to an integer dtype, and float-valued integer
arguments all use these helpers (`tsr_f2i64`, `tsr_f2i32`, `tsr_f2u8` in `csrc/src/internal.h`). On other
architectures NumPy itself follows the hardware (ARM64 saturates to the type's limits and maps NaN to 0),
so Tessero is deliberately the x86-64 NumPy answer everywhere.

### NaN propagates

- Element-wise arithmetic and `maximum`/`minimum` propagate NaN (`fmax`/`fmin` ignore it, as in NumPy).
- `min`, `max`, `argmin`, `argmax` reductions return the first NaN (its index for arg-reductions).
- `norm(x, ord=inf)` and `ord=-inf` (vector, matrix and `axis=` forms) return NaN when x contains NaN,
  like `abs(x).max()`; earlier versions skipped it.
- `logsumexp` returns NaN if any element is NaN (also when the others are all `-inf`), `-inf` for an
  empty or all `-inf` slice, and `inf` when the maximum is `+inf`.
- `nan_policy='omit'` on paired-sample statistics (`spearmanrho`, `kendalltau`, `pointbiserialr`,
  `chatterjeexi`) drops a position when either sample has NaN there; samples of different lengths are an
  error.

### Division

- Float `floor_divide` and `remainder` follow NumPy's `npy_divmod`: the remainder has the sign of the
  divisor (a zero remainder is `+0.0` or `-0.0` by the divisor's sign), the quotient is computed from
  `(x - fmod(x, y)) / y` and snapped, so `1.0 // 0.1 == 9.0` (not `floor(1.0 / 0.1) == 10`) and
  `x == (x // y) * y + x % y` up to rounding, as in NumPy. Division by zero gives `x / 0` (inf or nan)
  for the quotient and NaN for the remainder; `inf // y` is NaN.
- Complex division uses NumPy's Smith-style algorithm (scale by the larger component of the divisor), so
  it neither overflows for large divisors nor underflows for small ones:
  `(1+1j) / (1e200+1e200j) == 1e-200+0j`.

### Signed zero

`-0.0` is preserved through arithmetic, negation and copies, `1 / -0.0 == -inf`, and zero results of
`remainder` / `floor_divide` take the signs NumPy gives them. Comparisons treat `-0.0 == 0.0`.

### float32 accumulation

float32 reductions keep NumPy's float32 result dtype, but the accumulator depends on the path:

- A reduction over the last axis, over a non-contiguous axis, or over the whole array (`tsr_reduce`)
  accumulates float32 data in **double** (pairwise summation for `sum`, a double running product for
  `prod`) and rounds once to float32 at the end. This is usually more accurate than NumPy, which
  accumulates pairwise in float32, so the low bits can differ from NumPy within the parity tolerance.
- A reduction over a non-last axis of a contiguous array (`tsr_reduce_mid`) combines whole rows in
  **float32**, sequentially, as NumPy does for that layout.
- `mean`, `var` and `std` accumulate in double. `cumsum` / `cumprod` of float32 accumulate in float32.

### LAPACK dimension limit

LAPACK takes 32-bit integers on the usual LP64 builds. Every LAPACK-routed function refuses (with an
error naming the dimension) a matrix dimension above 2,147,483,647 instead of letting it wrap.
