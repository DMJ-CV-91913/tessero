/* Shared internals of libtessero (not part of the ABI). */
#ifndef TSR_INTERNAL_H
#define TSR_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include "../include/tessero.h"

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
 */
#if defined(__x86_64__) && defined(__GLIBC__) && defined(__GNUC__) && !defined(__clang__) && !defined(TSR_NO_CLONES)
#  define TSR_CLONES __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#  define TSR_CLONES
#endif

/* Two's-complement wrap-around, as NumPy integers behave (signed overflow is undefined in C). */
#define TSR_WADD(T, a, b) ((T)((uint64_t)(int64_t)(a) + (uint64_t)(int64_t)(b)))
#define TSR_WSUB(T, a, b) ((T)((uint64_t)(int64_t)(a) - (uint64_t)(int64_t)(b)))
#define TSR_WMUL(T, a, b) ((T)((uint64_t)(int64_t)(a) * (uint64_t)(int64_t)(b)))

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

#endif
