/*
 * Shared helpers for the registry fuzz targets (fuzz_einsum, fuzz_hist, fuzz_textio, fuzz_dtype,
 * fuzz_routines): a byte reader that turns the fuzz input into numbers, strings and small arrays,
 * tsr_arg builders whose storage is exact-size heap blocks (so ASan sees any overread), and the
 * shared oracle: every result is well formed, freed exactly once, and the iteration leaves
 * tsr_allocated() where it found it (no leaked kernel allocations, also on error paths).
 *
 * Everything is deterministic: the same input bytes always build the same call.
 */
#ifndef TSR_FUZZ_COMMON_H
#define TSR_FUZZ_COMMON_H

#include "../src/internal.h"
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- byte reader: reads past the end yield zeros, so every input maps to some call ---- */
typedef struct { const uint8_t *p; size_t n; } fz_in;

static inline uint8_t fz_u8(fz_in *in)
{
    if (!in->n) return 0;
    in->n--;
    return *in->p++;
}

static inline uint32_t fz_u32(fz_in *in)
{
    uint32_t v = 0;
    for (int i = 0; i < 4; i++) v |= (uint32_t)fz_u8(in) << (8 * i);
    return v;
}

static inline uint64_t fz_u64(fz_in *in)
{
    const uint64_t lo = fz_u32(in);           /* separate statements: argument order is unspecified */
    return lo | (uint64_t)fz_u32(in) << 32;
}

/* a double: mostly from a palette of edge values (NaN, infinities, huge, subnormal, int-like, size-like),
   sometimes small integers, sometimes raw bits */
static inline double fz_double(fz_in *in)
{
    static const double P[] = {
        0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 3.0, 10.0, -3.5, 1e-300, 4.9e-324, DBL_MIN, DBL_MAX, -DBL_MAX,
        1e308, 1e18, 9.2e18, 9.3e18, 4e9, 1048576.0, 2147483648.0, 9007199254740993.0, 1e-9, 100.0,
    };
    const uint8_t sel = fz_u8(in);
    if (sel < 32) {
        switch (sel & 3) {
        case 0: return NAN;
        case 1: return INFINITY;
        case 2: return -INFINITY;
        default: return -NAN;
        }
    }
    if (sel < 128) return P[sel % (sizeof P / sizeof P[0])];
    if (sel < 224) return (double)((int)fz_u8(in) - 64);
    uint64_t bits = fz_u64(in);
    double d;
    memcpy(&d, &bits, 8);
    return d;
}

/* a NUL-terminated string of at most max bytes (max < cap), copied into buf */
static inline const char *fz_str(fz_in *in, char *buf, size_t cap, size_t max)
{
    size_t len = fz_u8(in) % (max + 1);
    if (len >= cap) len = cap - 1;
    for (size_t i = 0; i < len; i++) buf[i] = (char)fz_u8(in);
    buf[len] = '\0';
    return buf;
}

/* ---- per-iteration storage: exact-size malloc blocks freed at the end of the iteration ---- */
#define FZ_POOL 64
typedef struct { void *blk[FZ_POOL]; int n; } fz_pool;

static inline void *fz_take(fz_pool *pool, size_t bytes)
{
    if (pool->n >= FZ_POOL) return NULL;
    void *p = malloc(bytes ? bytes : 1);
    if (!p) abort();
    pool->blk[pool->n++] = p;
    return p;
}

static inline void fz_release(fz_pool *pool)
{
    for (int i = 0; i < pool->n; i++) free(pool->blk[i]);
    pool->n = 0;
}

/* ---- tsr_arg builders ---- */
static inline tsr_arg fz_arg_none(void) { tsr_arg a; memset(&a, 0, sizeof a); return a; }

static inline tsr_arg fz_arg_num(double v)
{
    tsr_arg a = fz_arg_none();
    a.kind = 1;
    a.num = v;
    return a;
}

static inline tsr_arg fz_arg_int(int64_t v)
{
    tsr_arg a = fz_arg_num((double)v);
    a.flags = 1;
    a.ival = v;
    return a;
}

static inline tsr_arg fz_arg_bool(int v)
{
    tsr_arg a = fz_arg_none();
    a.kind = 4;
    a.num = v ? 1.0 : 0.0;
    return a;
}

static inline tsr_arg fz_arg_str(const char *s)
{
    tsr_arg a = fz_arg_none();
    a.kind = 2;
    a.str = s;
    return a;
}

static inline int64_t fz_itemsize(int dt)
{
    static const int64_t Z[] = {8, 4, 8, 4, 1, 1, 16};
    return dt >= 0 && dt < 7 ? Z[dt] : 8;
}

/* a C-contiguous array of the given dtype and shape, values from the reader. Returns kind 0 when the pool
   is exhausted or the shape is refused. */
static inline tsr_arg fz_arg_array(fz_in *in, fz_pool *pool, int dt, int32_t nd, const int64_t *shape)
{
    tsr_arg a = fz_arg_none();
    int64_t count = 1;
    for (int d = 0; d < nd; d++) count *= shape[d];
    const int64_t isz = fz_itemsize(dt);
    uint8_t *data = (uint8_t *)fz_take(pool, (size_t)(count * isz));
    if (!data) return a;
    for (int64_t i = 0; i < count; i++) {
        const double v = fz_double(in);
        switch (dt) {
        case 0: memcpy(data + i * 8, &v, 8); break;
        case 1: { float f = (float)(isfinite(v) && fabs(v) > FLT_MAX ? (v > 0 ? INFINITY : -INFINITY) : v); memcpy(data + i * 4, &f, 4); break; }
        case 2: { int64_t x = (int64_t)fz_u64(in); memcpy(data + i * 8, &x, 8); break; }
        case 3: { int32_t x = (int32_t)fz_u32(in); memcpy(data + i * 4, &x, 4); break; }
        case 4: data[i] = fz_u8(in); break;
        case 5: data[i] = fz_u8(in) & 1; break;
        default: { double im = fz_double(in); memcpy(data + i * 16, &v, 8); memcpy(data + i * 16 + 8, &im, 8); break; }
        }
    }
    if (tsr_array_init(&a.arr, dt, nd, shape, data) != 0) return fz_arg_none();
    a.kind = 3;
    return a;
}

/* a small f64 array whose rank (0..maxnd) and dims (0..maxdim) come from the reader */
static inline tsr_arg fz_arg_small(fz_in *in, fz_pool *pool, int dt, int maxnd, int maxdim)
{
    int64_t shape[32];
    const int32_t nd = (int32_t)(fz_u8(in) % (maxnd + 1));
    for (int d = 0; d < nd; d++) shape[d] = fz_u8(in) % (maxdim + 1);
    return fz_arg_array(in, pool, dt, nd, shape);
}

/* ---- oracle ---- */
static inline void fz_fail(const char *what, const char *routine)
{
    fprintf(stderr, "FUZZ ORACLE FAILURE: %s (%s)\n", what, routine ? routine : "?");
    abort();
}

/* validate and free one result (recursively for sequences) */
static inline void fz_result_free(tsr_result *r, const char *routine, int depth)
{
    if (depth > 4) fz_fail("result sequence nested too deep", routine);
    switch (r->kind) {
    case 0: case 1: case 4: case 5: break;
    case 2:                                                 /* string (dtype names): NUL-terminated, length in shape[0] */
        if (!r->arr.data || r->arr.shape[0] < 0 || r->bytes <= r->arr.shape[0] ||
            memchr(r->arr.data, 0, (size_t)r->bytes) != (char *)r->arr.data + r->arr.shape[0])
            fz_fail("bad string result", routine);
        tsr_free(r->arr.data, r->bytes);
        break;
    case 3:
        if (r->arr.ndim < 0 || r->arr.ndim > 32) fz_fail("result ndim outside 0..32", routine);
        if (r->arr.dtype < 0 || r->arr.dtype >= 7) fz_fail("result dtype invalid", routine);
        {
            int64_t count = 1;
            for (int d = 0; d < r->arr.ndim; d++) {
                if (r->arr.shape[d] < 0) fz_fail("negative result dimension", routine);
                count *= r->arr.shape[d];
            }
            if (count && (!r->arr.data || r->bytes < count * fz_itemsize(r->arr.dtype)))
                fz_fail("result block smaller than its shape", routine);
            if (count) {                                    /* touch the last element: ASan checks the block */
                volatile uint8_t last = ((const uint8_t *)r->arr.data)[r->arr.offset + count * fz_itemsize(r->arr.dtype) - 1];
                (void)last;
            }
        }
        if (r->arr.data) tsr_free(r->arr.data, r->bytes);
        break;
    case 6: {
        tsr_result *e = (tsr_result *)r->arr.data;
        const int64_t k = r->arr.shape[0];
        if (k < 0 || (k && !e) || (k && r->bytes < k * (int64_t)sizeof(tsr_result))) fz_fail("bad sequence result", routine);
        for (int64_t i = 0; i < k; i++) fz_result_free(&e[i], routine, depth + 1);
        if (e) tsr_free(e, r->bytes);
        break;
    }
    default: fz_fail("unknown result kind", routine);
    }
    memset(r, 0, sizeof *r);
}

/* call a routine by id: free the results, check the kernel allocation counter came back to where it was */
static inline int fz_call(int id, int nargs, const tsr_arg *args)
{
    if (id < 0) return -1;
    const char *name = tsr_fn_name(id);
    tsr_result res[8];
    memset(res, 0, sizeof res);
    const int64_t before = tsr_allocated();
    const int rc = tsr_fn_routine(id, nargs, args, 8, res);
    if (rc > 0) fz_fail("positive return code", name);
    if (rc == 0) for (int k = 0; k < 8; k++) fz_result_free(&res[k], name, 0);
    else for (int k = 0; k < 8; k++) if (res[k].kind != 0 && res[k].arr.data) fz_fail("result left on failure", name);
    if (tsr_allocated() != before) {
        fprintf(stderr, "leak: tsr_allocated %lld -> %lld after %s rc=%d (%s)\n", (long long)before,
                (long long)tsr_allocated(), name, rc, tsr_fn_error());
        fz_fail("kernel allocation leaked", name);
    }
    return rc;
}

/* one-time setup: a memory budget so size-like arguments fail cleanly instead of exhausting the machine */
static inline void fz_init(void)
{
    static int done;
    if (done) return;
    done = 1;
    tsr_set_budget((int64_t)256 << 20);
}

#endif
