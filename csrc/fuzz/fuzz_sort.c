/*
 * Fuzz target: the three sorts of sort.c (docs/project/sort-design.md). tsr_sort (radix), tsr_sort_simd (AVX2
 * quicksort, multithreaded for large arrays) and tsr_sort_runs (natural merge sort) must return the same bytes for
 * every input.
 *
 * Input: byte 0 = dtype (f64, f32, i64, i32, u8, bool), byte 1 = flags: bits 0, 5 and 6 all set tile the elements
 * past TSR_PAR_MIN and set 4 threads (the parallel path where OpenMP is built in; one input in eight, as a tiled
 * input costs about a thousand small ones), bit 1 runs the sorts under denormals-are-zero /
 * flush-to-zero (x86-64), bits 2-4 cap the quicksort's recursion depth at 1..7 (the heapsort fallback); the rest =
 * the elements' bytes. Oracle: the three outputs are identical and ordered: non-decreasing in the total order of
 * the design (-0.0 before +0.0), NaNs only at the end, as the canonical NaN (arrays of fewer than two elements are
 * returned untouched, payload included).
 */
#include "../src/internal.h"
#include <math.h>
#include <stdlib.h>

static const int fz_dt[6] = {TSR_F64, TSR_F32, TSR_I64, TSR_I32, TSR_U8, TSR_BOOL};

/* the order key of element i (floats: the signed-integer image of the bits); *nan for NaN */
static int64_t fz_key(int dt, const unsigned char *b, int64_t i, int *nan)
{
    *nan = 0;
    switch (dt) {
    case TSR_F64: { int64_t u; memcpy(&u, b + i * 8, 8); *nan = (u & INT64_MAX) > 0x7ff0000000000000LL; return u ^ ((u >> 63) & INT64_MAX); }
    case TSR_F32: { int32_t u; memcpy(&u, b + i * 4, 4); *nan = (u & INT32_MAX) > 0x7f800000; return u ^ ((u >> 31) & INT32_MAX); }
    case TSR_I64: { int64_t v; memcpy(&v, b + i * 8, 8); return v; }
    case TSR_I32: { int32_t v; memcpy(&v, b + i * 4, 4); return v; }
    default: return b[i];
    }
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size)
{
    if (size < 2) return 0;
    const int dt = fz_dt[data[0] % 6];
    const int flags = data[1];
    const int64_t isz = tsr_itemsize(dt);
    int64_t n = (int64_t)(size - 2) / isz;
    if (n == 0) return 0;
    const int big = (flags & 0x61) == 0x61;
    const int64_t total = big ? TSR_PAR_MIN + 7 : n;
    unsigned char *src = malloc((size_t)(total * isz)), *a = malloc((size_t)(total * isz)),
                  *b = malloc((size_t)(total * isz)), *c = malloc((size_t)(total * isz));
    for (int64_t i = 0; i < total; i++) {                       /* tiled: element i is input element i % n */
        memcpy(src + i * isz, data + 2 + (i % n) * isz, (size_t)isz);
        if (dt == TSR_BOOL) src[i] &= 1;                        /* bools are 0/1 bytes */
    }
    n = total;
    memcpy(a, src, (size_t)(n * isz)); memcpy(b, src, (size_t)(n * isz)); memcpy(c, src, (size_t)(n * isz));

    const int saved = tsr_get_threads();
    tsr_set_threads(big ? 4 : 1);
    tsr_sort_test_depth = (flags >> 2) & 7;
#if defined(__x86_64__) && defined(__GNUC__)
    const unsigned int csr = __builtin_ia32_stmxcsr();
    if (flags & 2) __builtin_ia32_ldmxcsr(csr | 0x8040);
#endif
    const int ra = tsr_sort(dt, n, a), rb = tsr_sort_simd(dt, n, b), rc = tsr_sort_runs(dt, n, c);
#if defined(__x86_64__) && defined(__GNUC__)
    __builtin_ia32_ldmxcsr(csr);
#endif
    tsr_sort_test_depth = 0;
    tsr_set_threads(saved);

    if (ra != TSR_OK || rb != TSR_OK || rc != TSR_OK) abort();  /* no allocation can fail at these sizes */
    if (memcmp(a, b, (size_t)(n * isz)) || memcmp(a, c, (size_t)(n * isz))) abort();
    int64_t prev = 0;
    int seen = 0, in_nans = 0;
    for (int64_t i = 0; i < n; i++) {
        int nan;
        const int64_t k = fz_key(dt, a, i, &nan);
        if (nan) {
            in_nans = 1;
            const int canonical = dt == TSR_F64 ? !memcmp(a + i * 8, &(double){NAN}, 8) : !memcmp(a + i * 4, &(float){NAN}, 4);
            if (!canonical && n >= 2) abort();                  /* fewer than 2 elements: returned untouched */
            continue;
        }
        if (in_nans) abort();                                   /* a number after a NaN */
        if (seen && k < prev) abort();
        prev = k; seen = 1;
    }
    free(src); free(a); free(b); free(c);
    return 0;
}
