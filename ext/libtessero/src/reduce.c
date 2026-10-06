/* Axis reductions, moments, sorting. */
#include <math.h>
#include <stdlib.h>
#include "internal.h"
#include "ops.h"

/* Pairwise summation (as NumPy): O(log n) error growth instead of O(n). */
#define PAIRWISE(NAME, T)                                                                  \
    static double NAME(const char *p, int64_t n, int64_t s)                                \
    {                                                                                      \
        if (n < 8) {                                                                       \
            double r = 0.0;                                                                \
            for (int64_t i = 0; i < n; i++) r += (double)*(const T *)(p + i * s);          \
            return r;                                                                      \
        }                                                                                  \
        if (n <= 128) {                                                                    \
            double r[8];                                                                   \
            for (int j = 0; j < 8; j++) r[j] = (double)*(const T *)(p + j * s);            \
            int64_t i;                                                                     \
            for (i = 8; i + 8 <= n; i += 8)                                                \
                for (int j = 0; j < 8; j++) r[j] += (double)*(const T *)(p + (i + j) * s); \
            double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7])); \
            for (; i < n; i++) res += (double)*(const T *)(p + i * s);                     \
            return res;                                                                    \
        }                                                                                  \
        int64_t h = (n / 2) & ~(int64_t)7;                                                 \
        return NAME(p, h, s) + NAME(p + h * s, n - h, s);                                  \
    }
PAIRWISE(psum_f64, double)
PAIRWISE(psum_f32, float)
PAIRWISE(psum_i64, int64_t)
PAIRWISE(psum_i32, int32_t)
PAIRWISE(psum_u8, uint8_t)

/* NumPy's pairwise sum of a contiguous float64 run (shared with the registry functions) */
double tsr_psum(const double *x, int64_t n) { return psum_f64((const char *)x, n, 8); }

static void store(int dt, char *o, double d, int64_t i, int use_int)
{
    switch (dt) {
    case TSR_F64: *(double *)o = use_int ? (double)i : d; break;
    case TSR_F32: *(float *)o = (float)(use_int ? (double)i : d); break;
    case TSR_I64: *(int64_t *)o = use_int ? i : (int64_t)d; break;
    case TSR_I32: *(int32_t *)o = (int32_t)(use_int ? i : (int64_t)d); break;
    case TSR_U8: case TSR_BOOL: *(uint8_t *)o = (uint8_t)(use_int ? i : (int64_t)d); break;
    default: break;
    }
}

#define REDUCE_FLOAT(NAME, T, PSUM)                                                         \
    static int NAME(int op, int64_t L, const char *p, int64_t s, int odt, char *o)          \
    {                                                                                       \
        switch (op) {                                                                       \
        case R_SUM: store(odt, o, PSUM(p, L, s), 0, 0); return TSR_OK;                      \
        case R_PROD: { double r = 1; for (int64_t i = 0; i < L; i++) r *= *(const T *)(p + i * s); store(odt, o, r, 0, 0); return TSR_OK; } \
        case R_MIN: case R_MAX: case R_ARGMIN: case R_ARGMAX: {                             \
            if (L == 0) return TSR_EARG;                                                    \
            int want_min = (op == R_MIN || op == R_ARGMIN);                                 \
            T best = *(const T *)p; int64_t bi = 0;                                         \
            for (int64_t i = 1; i < L && best == best; i++) { /* first NaN wins, as NumPy */ \
                T v = *(const T *)(p + i * s);                                              \
                if (v != v || (want_min ? v < best : v > best)) { best = v; bi = i; }       \
            }                                                                               \
            if (op == R_ARGMIN || op == R_ARGMAX) store(odt, o, 0, bi, 1); else store(odt, o, (double)best, 0, 0); \
            return TSR_OK;                                                                  \
        }                                                                                   \
        case R_ANY: { int r = 0; for (int64_t i = 0; i < L && !r; i++) r = *(const T *)(p + i * s) != 0; *(uint8_t *)o = (uint8_t)r; return TSR_OK; } \
        case R_ALL: { int r = 1; for (int64_t i = 0; i < L && r; i++) r = *(const T *)(p + i * s) != 0; *(uint8_t *)o = (uint8_t)r; return TSR_OK; } \
        default: return TSR_EARG;                                                           \
        }                                                                                   \
    }
REDUCE_FLOAT(red_f64, double, psum_f64)
REDUCE_FLOAT(red_f32, float, psum_f32)

#define REDUCE_INT(NAME, T)                                                                 \
    static int NAME(int op, int64_t L, const char *p, int64_t s, int odt, char *o)          \
    {                                                                                       \
        switch (op) {                                                                       \
        case R_SUM: { int64_t r = 0; for (int64_t i = 0; i < L; i++) r = TSR_WADD(int64_t, r, *(const T *)(p + i * s)); \
                      if (odt == TSR_F64 || odt == TSR_F32) store(odt, o, (double)r, 0, 0); else store(odt, o, 0, r, 1); return TSR_OK; } \
        case R_PROD: { int64_t r = 1; for (int64_t i = 0; i < L; i++) r = TSR_WMUL(int64_t, r, *(const T *)(p + i * s)); \
                      if (odt == TSR_F64 || odt == TSR_F32) store(odt, o, (double)r, 0, 0); else store(odt, o, 0, r, 1); return TSR_OK; } \
        case R_MIN: case R_MAX: case R_ARGMIN: case R_ARGMAX: {                             \
            if (L == 0) return TSR_EARG;                                                    \
            int want_min = (op == R_MIN || op == R_ARGMIN);                                 \
            T best = *(const T *)p; int64_t bi = 0;                                         \
            for (int64_t i = 1; i < L; i++) {                                               \
                T v = *(const T *)(p + i * s);                                              \
                if (want_min ? v < best : v > best) { best = v; bi = i; }                   \
            }                                                                               \
            if (op == R_ARGMIN || op == R_ARGMAX) store(odt, o, 0, bi, 1); else store(odt, o, 0, (int64_t)best, 1); \
            return TSR_OK;                                                                  \
        }                                                                                   \
        case R_ANY: { int r = 0; for (int64_t i = 0; i < L && !r; i++) r = *(const T *)(p + i * s) != 0; *(uint8_t *)o = (uint8_t)r; return TSR_OK; } \
        case R_ALL: { int r = 1; for (int64_t i = 0; i < L && r; i++) r = *(const T *)(p + i * s) != 0; *(uint8_t *)o = (uint8_t)r; return TSR_OK; } \
        default: return TSR_EARG;                                                           \
        }                                                                                   \
    }
REDUCE_INT(red_i64, int64_t)
REDUCE_INT(red_i32, int32_t)
REDUCE_INT(red_u8, uint8_t)

typedef int (*red_fn)(int, int64_t, const char *, int64_t, int, char *);
static const red_fn RED[TSR_NDTYPES] = {red_f64, red_f32, red_i64, red_i32, red_u8, red_u8, NULL};

/* Split an array into (non-axis dims) x (axis) for a 2-operand iterator over the outputs. */
static int prepare(int32_t ndim, const int64_t *shape, const int64_t *sin, int32_t axis, int64_t *oshape,
                   int64_t *ostrides_in, int32_t *ond)
{
    if (axis < 0 || axis >= ndim) return TSR_EARG;
    int k = 0;
    for (int d = 0; d < ndim; d++) {
        if (d == axis) continue;
        oshape[k] = shape[d];
        ostrides_in[k] = sin[d];
        k++;
    }
    *ond = k;
    return TSR_OK;
}

int tsr_reduce(int op, int dtype, int32_t ndim, const int64_t *shape, const void *in, const int64_t *sin,
               int32_t axis, int out_dtype, void *out, const int64_t *sout)
{
    if (dtype < 0 || dtype >= TSR_NDTYPES || RED[dtype] == NULL || ndim > TSR_MAXDIM) return dtype == TSR_C128 ? TSR_ETYPE : TSR_EARG;
    int64_t oshape[TSR_MAXDIM], ostr[TSR_MAXDIM];
    int32_t ond;
    int rc = prepare(ndim, shape, sin, axis, oshape, ostr, &ond);
    if (rc != TSR_OK) return rc;
    int64_t L = shape[axis], sL = sin[axis];
    if (L == 0 && (op == R_MIN || op == R_MAX || op == R_ARGMIN || op == R_ARGMAX)) return TSR_EARG;

    const int64_t *st[2] = {ostr, sout};
    tsr_iter it;
    if (ond == 0) { /* 1-D input: single output */
        return RED[dtype](op, L, (const char *)in, sL, out_dtype, (char *)out);
    }
    rc = tsr_iter_init(&it, 2, ond, oshape, st);
    if (rc != TSR_OK || it.size == 0) return rc;
    const void *base[2] = {in, out};
    int nd = it.nd;
    int err = TSR_OK;
    TSR_ITER_OUTER(it, base, {
        for (int64_t i = 0; i < n && err == TSR_OK; i++)
            err = RED[dtype](op, L, p[0] + i * it.st[0][nd - 1], sL, out_dtype, p[1] + i * it.st[1][nd - 1]);
        if (err != TSR_OK) break;
    });
    return err;
}

/*
 * Mean and M2 (sum of squared deviations) along an axis, in float64, computed
 * the way numpy.mean / numpy.var do: mean = pairwise_sum(x) / n, then
 * M2 = pairwise_sum((x - mean)^2). Two passes over each row, but the results
 * match NumPy's (a one-pass Welford update drifts by an ulp or two: the mean
 * of 24 ones and 24 zeros came out as 0.5000000000000001).
 */
int tsr_moments(int dtype, int32_t ndim, const int64_t *shape, const void *in, const int64_t *sin, int32_t axis,
                double *mean, const int64_t *smean, double *m2, const int64_t *sm2)
{
    if (dtype == TSR_C128 || tsr_itemsize(dtype) == 0 || ndim > TSR_MAXDIM) return TSR_ETYPE;
    int64_t oshape[TSR_MAXDIM], ostr[TSR_MAXDIM];
    int32_t ond;
    int rc = prepare(ndim, shape, sin, axis, oshape, ostr, &ond);
    if (rc != TSR_OK) return rc;
    int64_t L = shape[axis], sL = sin[axis];
    double *dev = L > 0 ? (double *)tsr_alloc(L * 8) : NULL;   /* squared deviations of one row */
    if (L > 0 && !dev) return TSR_ENOMEM;

#define TWOPASS(T, PSUM)                                                            \
    for (int64_t i = 0; i < n; i++) {                                               \
        const char *q = p[0] + i * it.st[0][nd - 1];                                \
        double mu = 0, M = 0;                                                       \
        if (L > 0) {                                                                \
            mu = PSUM(q, L, sL) / (double)L;                                        \
            for (int64_t j = 0; j < L; j++) {                                       \
                const double d = (double)*(const T *)(q + j * sL) - mu;             \
                dev[j] = d * d;                                                     \
            }                                                                       \
            M = psum_f64((const char *)dev, L, 8);                                  \
        }                                                                           \
        *(double *)(p[1] + i * it.st[1][nd - 1]) = mu;                              \
        *(double *)(p[2] + i * it.st[2][nd - 1]) = M;                               \
    }

    const int64_t *st[3] = {ostr, smean, sm2};
    tsr_iter it;
    int64_t unit_shape = 1, zst[3] = {0, 0, 0};
    const int64_t *zs[3] = {&zst[0], &zst[1], &zst[2]};
    if (ond == 0) rc = tsr_iter_init(&it, 3, 1, &unit_shape, zs);
    else rc = tsr_iter_init(&it, 3, ond, oshape, st);
    if (rc != TSR_OK || it.size == 0) { if (dev) tsr_free(dev, L * 8); return rc; }
    const void *base[3] = {in, mean, m2};
    int nd = it.nd;
    TSR_ITER_OUTER(it, base, {
        switch (dtype) {
        case TSR_F64: TWOPASS(double, psum_f64) break;
        case TSR_F32: TWOPASS(float, psum_f32) break;
        case TSR_I64: TWOPASS(int64_t, psum_i64) break;
        case TSR_I32: TWOPASS(int32_t, psum_i32) break;
        default: TWOPASS(uint8_t, psum_u8) break;
        }
    });
    if (dev) tsr_free(dev, L * 8);
    return TSR_OK;
#undef TWOPASS
}

/* Cumulative sum (op 0) or product (op 1) along the middle axis of a contiguous
   (outer, len, inner) block. float64/float32 keep their type; integer and bool
   inputs accumulate into int64 (numpy's default for cumsum/cumprod). */
#define CUMULATE(TI, TO, ADD, MUL)                                                                \
    for (int64_t o = 0; o < outer; o++) {                                               \
        const TI *src = (const TI *)in + o * len * inner;                               \
        TO *dst = (TO *)out + o * len * inner;                                          \
        for (int64_t j = 0; j < inner; j++) dst[j] = (TO)src[j];                        \
        for (int64_t i = 1; i < len; i++) {                                             \
            const TI *s = src + i * inner;                                              \
            TO *d = dst + i * inner, *prev = dst + (i - 1) * inner;                     \
            if (op == 0) for (int64_t j = 0; j < inner; j++) d[j] = ADD(TO, prev[j], (TO)s[j]); \
            else for (int64_t j = 0; j < inner; j++) d[j] = MUL(TO, prev[j], (TO)s[j]); \
        }                                                                               \
    }

#define FADD(T, a, b) ((a) + (b))
#define FMUL(T, a, b) ((a) * (b))

TSR_CLONES int tsr_cumulative(int op, int dtype, int64_t outer, int64_t len, int64_t inner, const void *in, void *out)
{
    if ((op != 0 && op != 1) || outer < 0 || len < 0 || inner < 0) return TSR_EARG;
    switch (dtype) {
    case TSR_F64: CUMULATE(double, double, FADD, FMUL); return TSR_OK;
    case TSR_F32: CUMULATE(float, float, FADD, FMUL); return TSR_OK;
    case TSR_I64: CUMULATE(int64_t, int64_t, TSR_WADD, TSR_WMUL); return TSR_OK;
    case TSR_I32: CUMULATE(int32_t, int64_t, TSR_WADD, TSR_WMUL); return TSR_OK;
    case TSR_U8: case TSR_BOOL: CUMULATE(uint8_t, int64_t, TSR_WADD, TSR_WMUL); return TSR_OK;
    default: return TSR_ETYPE;
    }
}

/*
 * Reduction over the middle axis of a contiguous (outer, len, inner) block with
 * inner > 1: whole rows are combined element-wise, so the loop is unit-stride
 * and vectorises (NumPy reduces a non-last axis the same way, sequentially).
 * SUM/PROD of integers accumulate in int64; MIN/MAX propagate NaN.
 */
#define MID(TI, TO, INIT, STEP)                                                         \
    for (int64_t o = 0; o < outer; o++) {                                               \
        const TI *src = (const TI *)in + o * len * inner;                               \
        TO *acc = (TO *)out + o * inner;                                                \
        if (len == 0) { for (int64_t j = 0; j < inner; j++) acc[j] = (TO)(INIT); continue; } \
        for (int64_t j = 0; j < inner; j++) acc[j] = (TO)src[j];                        \
        for (int64_t i = 1; i < len; i++) {                                             \
            const TI *row = src + i * inner;                                            \
            for (int64_t j = 0; j < inner; j++) { TO a = acc[j]; TO x = (TO)row[j]; acc[j] = (STEP); } \
        }                                                                               \
    }

#define MID_ALL(TI, TO)                                                                 \
    switch (op) {                                                                       \
    case R_SUM: MID(TI, TO, 0, a + x); return TSR_OK;                                   \
    case R_PROD: MID(TI, TO, 1, a * x); return TSR_OK;                                  \
    default: return TSR_EARG;                                                           \
    }
#define MID_ALL_INT(TI, TO)                                                             \
    switch (op) {                                                                       \
    case R_SUM: MID(TI, TO, 0, TSR_WADD(TO, a, x)); return TSR_OK;                      \
    case R_PROD: MID(TI, TO, 1, TSR_WMUL(TO, a, x)); return TSR_OK;                     \
    default: return TSR_EARG;                                                           \
    }
#define MID_MINMAX(T)                                                                   \
    switch (op) {                                                                       \
    case R_MIN: MID(T, T, 0, (a != a || a <= x) ? a : x); return TSR_OK;                \
    case R_MAX: MID(T, T, 0, (a != a || a >= x) ? a : x); return TSR_OK;                \
    default: break;                                                                     \
    }

TSR_CLONES int tsr_reduce_mid(int op, int dtype, int64_t outer, int64_t len, int64_t inner, const void *in, void *out)
{
    if (outer < 0 || len < 0 || inner < 0) return TSR_EARG;
    if ((op == R_MIN || op == R_MAX) && len == 0) return TSR_EARG;
    switch (dtype) {
    case TSR_F64: MID_MINMAX(double) MID_ALL(double, double)
    case TSR_F32: MID_MINMAX(float) MID_ALL(float, float)
    case TSR_I64: MID_MINMAX(int64_t) MID_ALL_INT(int64_t, int64_t)
    case TSR_I32: MID_MINMAX(int32_t) MID_ALL_INT(int32_t, int64_t)
    case TSR_U8: case TSR_BOOL: MID_MINMAX(uint8_t) MID_ALL_INT(uint8_t, int64_t)
    default: return TSR_ETYPE;
    }
}
