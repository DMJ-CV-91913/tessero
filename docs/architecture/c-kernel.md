# The C kernel (libtessero)

`csrc/` is a dependency-free C11 library, about 3 600 lines. It builds as a
shared library for the FFI package (`make -C csrc`) and is compiled into the
extension from the verified copy in `ext/libtessero/`.

## Files

| File | Contents |
|---|---|
| `include/tessero.h` | the complete public ABI: declarations only, no macros or includes, parseable by `FFI::cdef` |
| `src/internal.h` | error enum, dtype sizes, the N-operand iterator, wrap-around integer macros, `TSR_PAR_MIN` |
| `src/ops.h` | op codes for binary/unary/reduction kernels (mirrored in `Native\Abi`) |
| `src/alloc.c` | allocator, budget and counters, thread setting, SIMD probe, version |
| `src/array.c` | `tsr_array` metadata: init, broadcasting, slice-string parser, transpose, reshape-as-view, element access, strided binary op, JSON writer |
| `src/elementwise.c` | binary, unary, compare, where, copy/cast for 7 dtypes; contiguous, scalar and strided variants; OpenMP |
| `src/vmath.h` | branch-free, vectorisable `exp`/`log` (fdlibm algorithms, < 1 ulp) used by the unary kernels |
| `src/ufunc.c` | loop ufuncs (cbrt, erf, gamma, logaddexp, fmax, fmod, …): typed inner loops, `tsr_ufunc_resolve`, threaded runner; shared by both bindings |
| `src/mmap.c` | memory-map planning (pure, fuzzed), mapping/flush/unmap on POSIX and Win32, `.npy` header parser and writer |
| `src/reduce.c` | reductions (pairwise sum), moments (NumPy's two-pass mean/variance), `reduce_mid` (non-last axis), cumulative sum/product |
| `src/index.c` | take/put, mask selection, nonzero |
| `src/sort.c` | MSD+LSD hybrid radix sort (parallel buckets) and stable LSD argsort for every dtype |
| `src/matmul.c` | blocked matrix product (fallback when no BLAS) |
| `src/fft.c` | mixed-radix complex FFT + Bluestein; half-length real FFT (`tsr_rfft`, `tsr_irfft`) |
| `src/rng.c`, `src/ziggurat.h` | PCG64, SeedSequence, ziggurat normals (NumPy's tables), Lemire integers, shuffle |
| `src/sparse.c` | CSR from dense/triplets, matvec/matmat, transpose, CG, BiCGSTAB |
| `src/lp.c` | two-phase simplex (`tsr_linprog`), branch and bound (`tsr_milp`) |
| `src/mdp.c` | Bellman backup, value/policy/modified-policy iteration, policy evaluation, finite horizon |
| `tests/test_main.c` | 265 checks, run optimised and under ASan + UBSan |
| `fuzz/` | libFuzzer targets (slice parser, `.npy` header, memmap planning, JSON writer), seed corpora, a standalone mutation driver |

## Array metadata

```c
typedef struct tsr_array {
    void   *data;          /* start of the memory block */
    int64_t offset;        /* byte offset of element [0,…,0] */
    int32_t ndim;          /* 0..32 */
    int32_t dtype;         /* 0 f64, 1 f32, 2 i64, 3 i32, 4 u8, 5 bool, 6 c128 */
    int64_t shape[32];
    int64_t strides[32];   /* bytes; may be 0 (broadcast) or negative (reversed) */
} tsr_array;
```

The extension embeds one `tsr_array` in every PHP object. The FFI package
keeps shape and strides in PHP and builds a `tsr_array` on demand
(`NDArray::meta()`) for the functions that take one: slicing, JSON, strided binary ops.

## The iterator

Every element-wise kernel goes through `tsr_iter`, which takes up to four
operands (e.g. `out = where(cond, a, b)`) with their own strides and:

1. drops dimensions of length 1;
2. merges adjacent dimensions whenever every operand is contiguous across
   them (`stride[d-1] == stride[d] * shape[d]` for all operands).

A C-contiguous array of any shape therefore becomes **one** flat inner loop,
and a transposed or sliced one becomes the smallest number of strided runs.
The inner loop is specialised for the common cases: all operands contiguous,
one operand a broadcast scalar (stride 0), or general strides.

## SIMD

On x86-64 glibc with GCC, the hot loops carry
`__attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))`
(the `TSR_CLONES` macro in `internal.h`). The compiler emits three versions,
and the dynamic loader's ifunc resolver picks one when the library loads,
based on the CPU. The same binary runs on any x86-64 CPU and uses AVX-512
(v4) or AVX2+FMA (v3) where available. `tsr_simd_level()` reports the choice
(`Tessero::info()['simd']`). Elsewhere (musl, macOS, Windows, ARM, Clang), the
platform's compiler flags choose the instruction set: NEON is the aarch64
baseline. `make portable` or `-DTSR_NO_CLONES` disables the clones.


## Threads

`#pragma omp parallel for schedule(static)` splits element-wise loops of at
least `TSR_PAR_MIN` = 131 072 elements, and MDP backups over more than 4 096
state-action pairs, into one contiguous chunk per thread. With static
scheduling and no cross-thread arithmetic, results are identical for any
thread count. Reductions, sort and FFT stay serial on purpose (fixed
summation order).

`tsr_set_threads(n)` stores the count in an atomic. It is clamped to
[1, 4 × `omp_get_num_procs()`], and is always 1 without OpenMP. Each parallel
region passes `num_threads(threads)`, so the global OpenMP state is never
modified.

## Integer semantics

Signed integer overflow is undefined behaviour in C, but defined
(two's-complement wrap) in NumPy. The kernel computes signed integer `+ - *`,
sums, products and integer matrix products in unsigned arithmetic
(`TSR_WADD/WSUB/WMUL`), which matches NumPy exactly and stays clean under UBSan.

## Error handling

No kernel aborts, prints or longjmps. Every failure is a negative return
code ([Error codes](../reference/error-codes.md)). Allocation failures unwind
what was allocated. Tests check that the byte counter returns to its starting
value after failures.

## Portability

| Platform | Compiler | Notes |
|---|---|---|
| Linux glibc x86-64/aarch64 | GCC ≥ 9, Clang ≥ 12 | reference platform; huge-page hints via `madvise` |
| Linux musl (Alpine) | GCC | no ifunc on musl: baseline instruction set unless you add `-march` |
| macOS arm64/x86-64 | Apple Clang | no OpenMP by default |
| Windows x86-64 | MinGW-w64 | `_aligned_malloc`; no OpenMP |

Only Linux glibc x86-64 has been built and tested in the reference
environment. The other rows are CI jobs that have not run yet (see the
[roadmap](../project/roadmap.md)).
