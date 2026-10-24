/* Shared internals of libtessero (not part of the ABI). */
#ifndef TSR_INTERNAL_H
#define TSR_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
/* libtessero is built hidden-by-default (csrc/Makefile VIS): the public ABI is exactly what tessero.h declares */
#if defined(TSR_EXPORT_ABI) && defined(__GNUC__)
#  pragma GCC visibility push(default)
#endif
#include "../include/tessero.h"
#if defined(TSR_EXPORT_ABI) && defined(__GNUC__)
#  pragma GCC visibility pop
#endif

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

#define TSR_MAXDIM 32

enum { TSR_F64 = 0, TSR_F32 = 1, TSR_I64 = 2, TSR_I32 = 3, TSR_U8 = 4, TSR_BOOL = 5, TSR_C128 = 6, TSR_NDTYPES = 7 };
enum { TSR_OK = 0, TSR_EARG = -1, TSR_EINDEX = -2, TSR_ENOMEM = -3, TSR_ETYPE = -4, TSR_EDIM = -5, TSR_ESHAPE = -6, TSR_ECONTIG = -7, TSR_ECONVERGE = -8, TSR_EIO = -9 };

/* Parallel element-wise loops only above this many elements per call (OpenMP builds). */
#define TSR_PAR_MIN ((int64_t)1 << 17)
int tsr_get_threads(void);

/*
 * Function multi-versioning: on x86-64 glibc, hot loops are compiled for
 * x86-64-v4 (AVX-512), v3 (AVX2/FMA) and baseline, and the dynamic loader
 * picks the best clone for the running CPU (ifunc). Elsewhere (musl,
 * macOS, Windows, ARM) the platform build flags choose the ISA instead.
 *
 * GCC >= 12 is REQUIRED for the clones: GCC 11 (the Ubuntu 22.04 LTS default)
 * cannot emit the ifunc dispatcher for `static` multiversioned functions and
 * fails the build with "no dispatcher found for the versioning attributes".
 * On GCC 11 (or anything else) TSR_CLONES expands to nothing, so the functions
 * compile as a single version using the build's -march; results are IDENTICAL
 * (determinism is by construction, not by which SIMD clone runs), only the
 * auto-dispatched SIMD speedup is lost. Install gcc-12+ and build with CC=gcc-12
 * to get the multiversioned fast paths. Define TSR_NO_CLONES to force-disable.
 */
#if defined(__x86_64__) && defined(__GLIBC__) && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) && !defined(TSR_NO_CLONES)
#  define TSR_CLONES __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#  define TSR_CLONES
#endif

/* Two's-complement wrap-around, as NumPy integers behave (signed overflow is undefined in C). */
#define TSR_WADD(T, a, b) ((T)((uint64_t)(int64_t)(a) + (uint64_t)(int64_t)(b)))
#define TSR_WSUB(T, a, b) ((T)((uint64_t)(int64_t)(a) - (uint64_t)(int64_t)(b)))
#define TSR_WMUL(T, a, b) ((T)((uint64_t)(int64_t)(a) * (uint64_t)(int64_t)(b)))

/*
 * Round-once fused multiply-add: a*b + c with a SINGLE rounding, by construction.
 * Explicit fmaf/fma make the result bit-identical across compiler version, the
 * -ffp-contract setting, which SIMD clone runs, and the target ISA (a non-FMA CPU
 * uses the correctly-rounded software FMA and gets the SAME bits). This is what
 * makes the gemm / axpy / dot kernels canonical, the same discipline repro_f32.c
 * applies to the transcendentals. Relying on the compiler to contract c + a*b into
 * an FMA is the footgun: GCC 14 and GCC 12 disagreed and silently emitted a
 * two-round, non-canonical kernel. T is the element type (float or double).
 */
#define TSR_RFMA(T, a, b, c) (_Generic((T)0, float: fmaf, double: fma, default: fma)((a), (b), (c)))

/* Element count of a shape, or -1 if negative or if the count (or count * 16 bytes) would overflow int64.
   Every size that becomes an allocation or a loop bound goes through this. */
static inline int64_t tsr_shape_size(int32_t ndim, const int64_t *shape)
{
    int64_t n = 1;
    for (int32_t d = 0; d < ndim; d++) {
        if (shape[d] < 0) return -1;
        if (shape[d] != 0 && n > (INT64_MAX / 16) / shape[d]) return -1;
        n *= shape[d];
    }
    return n;
}

/* the machine's physical memory in bytes, 0 when unknown (alloc.c) */
int64_t tsr_physical_bytes(void);
/* 1 when `bytes` more can be allocated in total: within the budget left, or the machine's RAM without a budget */
static inline int tsr_fits(double bytes)
{
    const int64_t budget = tsr_budget();
    if (budget > 0) return bytes <= (double)(budget - tsr_allocated());
    const int64_t phys = tsr_physical_bytes();
    return phys <= 0 || bytes <= (double)phys;
}

static inline int64_t tsr_itemsize(int dtype)
{
    static const int64_t sizes[TSR_NDTYPES] = {8, 4, 8, 4, 1, 1, 16};
    return (dtype >= 0 && dtype < TSR_NDTYPES) ? sizes[dtype] : 0;
}

/*
 * N-operand strided iterator. Size-1 dimensions are dropped and adjacent
 * dimensions are merged whenever every operand is contiguous across them,
 * so a C-contiguous array of any shape becomes one flat inner loop.
 */
typedef struct {
    int nd;          /* dimensions after coalescing (>= 1 unless empty) */
    int nop;
    int64_t size;    /* total elements */
    int64_t shape[TSR_MAXDIM];
    int64_t st[4][TSR_MAXDIM];
} tsr_iter;

static inline int tsr_iter_init(tsr_iter *it, int nop, int32_t ndim, const int64_t *shape, const int64_t *const *strides)
{
    if (ndim < 0 || ndim > TSR_MAXDIM || nop < 1 || nop > 4) {
        return ndim > TSR_MAXDIM ? TSR_EDIM : TSR_EARG;
    }
    it->nop = nop;
    it->nd = 0;
    it->size = tsr_shape_size(ndim, shape);
    if (it->size < 0) return TSR_EARG;              /* negative dimension or an element count that overflows */
    if (it->size == 0) return TSR_OK;
    for (int d = 0; d < ndim; d++) {
        if (shape[d] == 1) continue;
        int k = it->nd;
        if (k > 0) {
            int merge = 1;
            for (int op = 0; op < nop; op++) {
                if (it->st[op][k - 1] != strides[op][d] * shape[d]) { merge = 0; break; }
            }
            if (merge) {
                it->shape[k - 1] *= shape[d];
                for (int op = 0; op < nop; op++) it->st[op][k - 1] = strides[op][d];
                continue;
            }
        }
        it->shape[k] = shape[d];
        for (int op = 0; op < nop; op++) it->st[op][k] = strides[op][d];
        it->nd++;
    }
    if (it->nd == 0) { /* scalar */
        it->nd = 1;
        it->shape[0] = 1;
        for (int op = 0; op < nop; op++) it->st[op][0] = 0;
    }
    return TSR_OK;
}

/*
 * Walk every outer position; call BODY with pointer array p[] positioned at
 * the start of an inner run of length n (inner strides it.st[op][nd-1]).
 */
#define TSR_ITER_OUTER(it, base, BODY)                                              \
    do {                                                                            \
        int64_t idx_[TSR_MAXDIM] = {0};                                             \
        char *p[4];                                                                 \
        for (int o_ = 0; o_ < (it).nop; o_++) p[o_] = (char *)(base)[o_];         \
        int64_t n = (it).shape[(it).nd - 1];                                        \
        int64_t outer_ = (it).size / n;                                             \
        for (int64_t r_ = 0; r_ < outer_; r_++) {                                   \
            BODY;                                                                   \
            for (int d_ = (it).nd - 2; d_ >= 0; d_--) {                             \
                if (++idx_[d_] < (it).shape[d_]) {                                  \
                    for (int o_ = 0; o_ < (it).nop; o_++) p[o_] += (it).st[o_][d_]; \
                    break;                                                          \
                }                                                                   \
                idx_[d_] = 0;                                                       \
                for (int o_ = 0; o_ < (it).nop; o_++)                               \
                    p[o_] -= (it).st[o_][d_] * ((it).shape[d_] - 1);                \
            }                                                                       \
        }                                                                           \
    } while (0)

/* numpy bit-generator view of a PCG64 state (rng.c), for the numpy/random distributions */
struct bitgen;
void tsr_pcg64_bitgen(uint64_t *state, struct bitgen *bg);

/*
 * Checked allocation sizes (ADR 0014). A size that is the product of two INDEPENDENT lengths (n*n, rows*cols,
 * count*itemsize) goes through these instead of a bare multiply: a negative operand or an int64 overflow gives
 * -1 / NULL, so a huge argument becomes a MemoryError instead of a short buffer.
 *   tsr_count_mul(a, b)        a*b, or -1
 *   tsr_bytes_n(n, el, pad)    n*el + pad, or -1
 *   tsr_alloc_n(n, el, pad)    counted tsr_alloc() of tsr_bytes_n() bytes (tessero.h); free it with
 *                              tsr_free(p, tsr_bytes_n(n, el, pad))
 *   tsr_scratch_n(n, el)       raw malloc of n*el bytes (at least 1) for scratch released with free()
 *   tsr_scratch0_n(n, el)      the zero-filled (calloc) variant
 * The raw helpers are NOT counted against the budget (see ADR 0014 for the counted-allocator plan).
 * tsr_effective_cpus (alloc.c): CPUs this process may use (affinity mask and cgroup quota), cached, >= 1.
 */
#include <stdlib.h>
int tsr_effective_cpus(void);
static inline int tsr_mul_overflows(int64_t a, int64_t b, int64_t *out)
{
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(a, b, out);
#else
    if (a != 0 && b != 0 && (a > INT64_MAX / b)) return 1;   /* operands are non-negative here */
    *out = a * b;
    return 0;
#endif
}
static inline int64_t tsr_count_mul(int64_t a, int64_t b)
{
    int64_t r;
    if (a < 0 || b < 0 || tsr_mul_overflows(a, b, &r)) return -1;
    return r;
}
static inline int64_t tsr_bytes_n(int64_t n, int64_t elsize, int64_t pad)
{
    const int64_t b = tsr_count_mul(n, elsize);
    if (b < 0 || pad < 0 || b > INT64_MAX - pad) return -1;
    return b + pad;
}
static inline void *tsr_scratch_n(int64_t n, int64_t elsize)
{
    const int64_t b = tsr_bytes_n(n, elsize, 0);
    if (b < 0 || (uint64_t)b > (uint64_t)SIZE_MAX) return NULL;
    return malloc(b > 0 ? (size_t)b : 1);
}
static inline void *tsr_scratch0_n(int64_t n, int64_t elsize)
{
    const int64_t b = tsr_bytes_n(n, elsize, 0);
    if (b < 0 || (uint64_t)b > (uint64_t)SIZE_MAX) return NULL;
    return b > 0 ? calloc((size_t)n, (size_t)elsize) : calloc(1, 1);
}

#endif
