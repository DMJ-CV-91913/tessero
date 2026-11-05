/*
 * Complex FFT of any length.
 *
 *  - n factored into 4, 2, 3, 5 and then any remaining primes;
 *  - recursive decimation-in-time (out-of-place, stride on input), with
 *    dedicated radix-2/3/4/5 butterflies and a generic O(p^2) butterfly;
 *  - if a prime factor exceeds TSR_FFT_MAX_RADIX the whole length goes
 *    through Bluestein's chirp-z algorithm on a power-of-two convolution,
 *    so primes such as 1_000_003 stay O(n log n).
 *
 * Forward: X[k] = sum x[j] exp(-2 pi i jk/n). Inverse uses +i and scales by 1/n
 * (numpy's default "backward" normalisation). Data are interleaved re, im.
 */
#include "internal.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <pthread.h>
#endif

#define TSR_FFT_MAX_RADIX 61

typedef struct { double re, im; } cpx;

/* A plan is read-only once built, so one cached plan serves every caller and thread at once; the generic
   butterfly's scratch and Bluestein's work buffers are per call. */
typedef struct {
    int64_t n, maxp;
    int nf;
    int cached;       /* memory from malloc (a plan held by the cache) rather than tsr_alloc */
    int64_t factors[64];
    cpx *tw;          /* tw[k] = exp(sign * 2 pi i k / n); kept only when a generic (p > 5) butterfly reads it */
    cpx *ltw[64];     /* per-level contiguous twiddles for radix 2-5: ltw[k*(p-1) + q-1] = tw[q*k*fstride] */
    int64_t ltw_len[64];
} plan_t;

/* Memory of an object the cache may hold comes from malloc: the cache keeps it past the call, so it must not count
   against the tsr_alloc budget as if it were an array the caller still owns. The cache is capped
   (TSR_FFT_CACHE_BYTES); an object too large for it is built with tsr_alloc and freed after the call, as before. */
static void *pl_alloc(int cached, int64_t bytes) { return cached ? malloc((size_t)bytes) : tsr_alloc(bytes); }
static void pl_free(int cached, void *p, int64_t bytes)
{
    if (!p) return;
    if (cached) free(p);
    else tsr_free(p, bytes);
}

static inline cpx cmul(cpx a, cpx b) { return (cpx){a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
static inline cpx cadd(cpx a, cpx b) { return (cpx){a.re + b.re, a.im + b.im}; }
static inline cpx csub(cpx a, cpx b) { return (cpx){a.re - b.re, a.im - b.im}; }

static int factorize(int64_t n, int64_t *f, int64_t *maxp)
{
    int nf = 0;
    *maxp = 1;
    while (n % 4 == 0) { f[nf++] = 4; n /= 4; }
    while (n % 2 == 0) { f[nf++] = 2; n /= 2; }
    for (int64_t p = 3; p * p <= n; p += 2)
        while (n % p == 0) { f[nf++] = p; n /= p; if (p > *maxp) *maxp = p; }
    if (n > 1) { f[nf++] = n; if (n > *maxp) *maxp = n; }
    for (int i = 0; i < nf; i++) if (f[i] > *maxp) *maxp = f[i];
    return nf;
}

static void twiddles(cpx *tw, int64_t n, double sign)
{
    for (int64_t k = 0; k < n; k++) {
        /* reduce the angle to the first octant-free form for accuracy: use k/n directly */
        double a = sign * 2.0 * M_PI * (double)k / (double)n;
        tw[k].re = cos(a);
        tw[k].im = sin(a);
    }
}

static void butterfly(const plan_t *pl, const cpx *lt, cpx *out, int64_t fstride, int64_t m, int64_t p, int inverse,
                      cpx *scratch)
{
    const cpx *tw = pl->tw;
    if (p == 2) {
        for (int64_t k = 0; k < m; k++) {
            cpx t = cmul(out[k + m], lt[k]);
            out[k + m] = csub(out[k], t);
            out[k] = cadd(out[k], t);
        }
    } else if (p == 4) {
        for (int64_t k = 0; k < m; k++) {
            cpx a0 = out[k];
            cpx a1 = cmul(out[k + m], lt[3 * k]);
            cpx a2 = cmul(out[k + 2 * m], lt[3 * k + 1]);
            cpx a3 = cmul(out[k + 3 * m], lt[3 * k + 2]);
            cpx s02 = cadd(a0, a2), d02 = csub(a0, a2);
            cpx s13 = cadd(a1, a3), d13 = csub(a1, a3);
            /* multiply d13 by -i (forward) or +i (inverse) */
            cpx r = inverse ? (cpx){-d13.im, d13.re} : (cpx){d13.im, -d13.re};
            out[k] = cadd(s02, s13);
            out[k + 2 * m] = csub(s02, s13);
            out[k + m] = cadd(d02, r);
            out[k + 3 * m] = csub(d02, r);
        }
    } else if (p == 3) {
        const double s3 = (inverse ? 1.0 : -1.0) * 0.86602540378443864676;
        for (int64_t k = 0; k < m; k++) {
            cpx a0 = out[k];
            cpx a1 = cmul(out[k + m], lt[2 * k]);
            cpx a2 = cmul(out[k + 2 * m], lt[2 * k + 1]);
            cpx s = cadd(a1, a2), d = csub(a1, a2);
            cpx c = {a0.re - 0.5 * s.re, a0.im - 0.5 * s.im};
            cpx rot = {-s3 * d.im, s3 * d.re};
            out[k] = cadd(a0, s);
            out[k + m] = cadd(c, rot);
            out[k + 2 * m] = csub(c, rot);
        }
    } else if (p == 5) {
        const double c1 = 0.30901699437494742410, c2 = -0.80901699437494742410;
        const double sg = inverse ? 1.0 : -1.0;
        const double s1 = sg * 0.95105651629515357212, s2 = sg * 0.58778525229247312917;
        for (int64_t k = 0; k < m; k++) {
            cpx a0 = out[k];
            cpx a1 = cmul(out[k + m], lt[4 * k]);
            cpx a2 = cmul(out[k + 2 * m], lt[4 * k + 1]);
            cpx a3 = cmul(out[k + 3 * m], lt[4 * k + 2]);
            cpx a4 = cmul(out[k + 4 * m], lt[4 * k + 3]);
            cpx t1 = cadd(a1, a4), t2 = cadd(a2, a3), t3 = csub(a1, a4), t4 = csub(a2, a3);
            cpx b1 = {a0.re + c1 * t1.re + c2 * t2.re, a0.im + c1 * t1.im + c2 * t2.im};
            cpx b2 = {a0.re + c2 * t1.re + c1 * t2.re, a0.im + c2 * t1.im + c1 * t2.im};
            /* i * (s1 t3 + s2 t4) and i * (s2 t3 - s1 t4) */
            cpx r1 = {-(s1 * t3.im + s2 * t4.im), s1 * t3.re + s2 * t4.re};
            cpx r2 = {-(s2 * t3.im - s1 * t4.im), s2 * t3.re - s1 * t4.re};
            out[k] = cadd(a0, cadd(t1, t2));
            out[k + m] = cadd(b1, r1);
            out[k + 4 * m] = csub(b1, r1);
            out[k + 2 * m] = cadd(b2, r2);
            out[k + 3 * m] = csub(b2, r2);
        }
    } else {
        cpx *t = scratch;
        const int64_t n = pl->n;
        for (int64_t k = 0; k < m; k++) {
            for (int64_t q = 0; q < p; q++) t[q] = cmul(out[k + q * m], tw[(q * k * fstride) % n]);
            for (int64_t s = 0; s < p; s++) {
                cpx acc = t[0];
                int64_t step = s * m * fstride, idx = 0;
                for (int64_t q = 1; q < p; q++) {
                    idx += step;
                    if (idx >= n) idx -= n * (idx / n);
                    acc = cadd(acc, cmul(t[q], tw[idx]));
                }
                out[k + s * m] = acc;
            }
        }
    }
}

static void rec(const plan_t *pl, cpx *out, const cpx *in, int64_t fstride, int64_t istride, int fi, int inverse,
                cpx *scratch)
{
    int64_t p = pl->factors[fi];
    int64_t m = 1;
    for (int j = fi + 1; j < pl->nf; j++) m *= pl->factors[j];
    if (m == 1) {
        for (int64_t q = 0; q < p; q++) out[q] = in[q * istride];
    } else {
        for (int64_t q = 0; q < p; q++)
            rec(pl, out + q * m, in + q * istride, fstride * p, istride * p, fi + 1, inverse, scratch);
    }
    butterfly(pl, pl->ltw[fi], out, fstride, m, p, inverse, scratch);
}

/* rec for real input: the leaf reads x[q * istride] as (x, 0.0), exactly what converting the input to complex first
   would hold, so the transform is the same bit for bit without that conversion pass */
static void rec_real(const plan_t *pl, cpx *out, const double *in, int64_t fstride, int64_t istride, int fi, int inverse,
                     cpx *scratch)
{
    int64_t p = pl->factors[fi];
    int64_t m = 1;
    for (int j = fi + 1; j < pl->nf; j++) m *= pl->factors[j];
    if (m == 1) {
        for (int64_t q = 0; q < p; q++) out[q] = (cpx){in[q * istride], 0.0};
    } else {
        for (int64_t q = 0; q < p; q++)
            rec_real(pl, out + q * m, in + q * istride, fstride * p, istride * p, fi + 1, inverse, scratch);
    }
    butterfly(pl, pl->ltw[fi], out, fstride, m, p, inverse, scratch);
}

static void plan_free(plan_t *pl)
{
    for (int fi = 0; fi < pl->nf; fi++) pl_free(pl->cached, pl->ltw[fi], pl->ltw_len[fi] * (int64_t)sizeof(cpx));
    pl_free(pl->cached, pl->tw, pl->n * (int64_t)sizeof(cpx));
}

static int plan_init(plan_t *pl, int64_t n, int inverse, int cached)
{
    pl->n = n;
    pl->cached = cached;
    pl->nf = factorize(n, pl->factors, &pl->maxp);
    for (int fi = 0; fi < pl->nf; fi++) { pl->ltw[fi] = NULL; pl->ltw_len[fi] = 0; }
    pl->tw = (cpx *)pl_alloc(cached, n * (int64_t)sizeof(cpx));
    if (!pl->tw) return TSR_ENOMEM;
    twiddles(pl->tw, n, inverse ? 1.0 : -1.0);
    int64_t fstride = 1;
    for (int fi = 0; fi < pl->nf; fi++) {
        int64_t p = pl->factors[fi];
        int64_t m = n / (fstride * p);
        if (p <= 5 && m > 0) {
            int64_t len = m * (p - 1);
            cpx *lt = (cpx *)pl_alloc(cached, len * (int64_t)sizeof(cpx));
            if (!lt) { plan_free(pl); return TSR_ENOMEM; }
            for (int64_t k = 0; k < m; k++)
                for (int64_t q = 1; q < p; q++) lt[k * (p - 1) + q - 1] = pl->tw[q * k * fstride];
            pl->ltw[fi] = lt;
            pl->ltw_len[fi] = len;
        }
        fstride *= p;
    }
    /* radix 2-5 butterflies read only the per-level tables: drop the full one (half the plan's memory) */
    if (pl->maxp <= 5) { pl_free(cached, pl->tw, n * (int64_t)sizeof(cpx)); pl->tw = NULL; }
    return TSR_OK;
}

/* bytes a plan of length n keeps after plan_init */
static int64_t plan_bytes(int64_t n)
{
    int64_t f[64], maxp;
    const int nf = factorize(n, f, &maxp);
    int64_t count = maxp > 5 ? n : 0, fstride = 1;
    for (int fi = 0; fi < nf; fi++) {
        const int64_t p = f[fi], m = n / (fstride * p);
        if (p <= 5 && m > 0) count += m * (p - 1);
        fstride *= p;
    }
    return count * (int64_t)sizeof(cpx);
}

/* Unnormalised transform of one row via the mixed-radix plan (in != out); scratch holds maxp + 1 values. */
static void run_plan(const plan_t *pl, const cpx *in, cpx *out, int inverse, cpx *scratch)
{
    if (pl->n == 1) { out[0] = in[0]; return; }
    rec(pl, out, in, 1, 1, 0, inverse, scratch);
}

static void run_plan_real(const plan_t *pl, const double *in, cpx *out, int inverse, cpx *scratch)
{
    if (pl->n == 1) { out[0] = (cpx){in[0], 0.0}; return; }
    rec_real(pl, out, in, 1, 1, 0, inverse, scratch);
}

/* Bluestein: X_k = w_k * sum_j (x_j w_j) conj(w_{k-j}), w_k = exp(-/+ i pi k^2 / n). Read-only once built; the two
   length-M work buffers are the caller's. M is a power of two, so the plans never use the generic butterfly. */
typedef struct {
    int64_t n, M;
    int cached;
    plan_t fwd, inv;
    cpx *w, *bf;      /* w: n chirp values; bf: the transformed conjugate chirp, M values */
} blue_t;

static int64_t blue_len(int64_t n)
{
    int64_t M = 1;
    while (M < 2 * n - 1) M <<= 1;
    return M;
}

static int64_t blue_bytes(int64_t n)
{
    const int64_t M = blue_len(n);
    return 2 * plan_bytes(M) + (n + M) * (int64_t)sizeof(cpx);
}

static int blue_init(blue_t *bl, int64_t n, int inverse, int cached)
{
    const int64_t M = blue_len(n);
    bl->n = n;
    bl->M = M;
    bl->cached = cached;
    if (plan_init(&bl->fwd, M, 0, cached) != TSR_OK) return TSR_ENOMEM;
    if (plan_init(&bl->inv, M, 1, cached) != TSR_OK) { plan_free(&bl->fwd); return TSR_ENOMEM; }
    cpx *mem = (cpx *)pl_alloc(cached, (n + M) * (int64_t)sizeof(cpx));
    cpx *b = (cpx *)tsr_alloc(M * (int64_t)sizeof(cpx));
    if (!mem || !b) {
        pl_free(cached, mem, (n + M) * (int64_t)sizeof(cpx));
        if (b) tsr_free(b, M * (int64_t)sizeof(cpx));
        plan_free(&bl->fwd);
        plan_free(&bl->inv);
        return TSR_ENOMEM;
    }
    bl->w = mem;
    bl->bf = mem + n;
    const double sign = inverse ? 1.0 : -1.0;
    for (int64_t k = 0; k < n; k++) {
        /* k^2 mod 2n keeps the angle small and exact */
        uint64_t kk = (uint64_t)k;
        uint64_t k2 = (uint64_t)(((unsigned __int128)kk * kk) % (unsigned __int128)(2 * (uint64_t)n));
        double a = sign * M_PI * (double)k2 / (double)n;
        bl->w[k] = (cpx){cos(a), sin(a)};
    }
    memset(b, 0, (size_t)M * sizeof(cpx));
    b[0] = (cpx){bl->w[0].re, -bl->w[0].im};
    for (int64_t k = 1; k < n; k++) {
        cpx c = {bl->w[k].re, -bl->w[k].im};
        b[k] = c;
        b[M - k] = c;
    }
    cpx scratch[8];
    run_plan(&bl->fwd, b, bl->bf, 0, scratch);
    tsr_free(b, M * (int64_t)sizeof(cpx));
    return TSR_OK;
}

static void blue_run(const blue_t *bl, const cpx *in, cpx *out, cpx *a, cpx *b)
{
    const int64_t n = bl->n, M = bl->M;
    cpx scratch[8];
    for (int64_t k = 0; k < n; k++) b[k] = cmul(in[k], bl->w[k]);
    memset(b + n, 0, (size_t)(M - n) * sizeof(cpx));
    run_plan(&bl->fwd, b, a, 0, scratch);
    for (int64_t k = 0; k < M; k++) a[k] = cmul(a[k], bl->bf[k]);
    run_plan(&bl->inv, a, b, 1, scratch);
    const double s = 1.0 / (double)M;
    for (int64_t k = 0; k < n; k++) {
        cpx v = {b[k].re * s, b[k].im * s};
        out[k] = cmul(v, bl->w[k]);
    }
}

static void blue_free(blue_t *bl)
{
    plan_free(&bl->fwd);
    plan_free(&bl->inv);
    pl_free(bl->cached, bl->w, (bl->n + bl->M) * (int64_t)sizeof(cpx));
}

/*
 * Twiddles w[k] = exp(sign * 2 pi i k / n), k = 0 .. n/2, from a table of the
 * first octant (n/8 + 1 sin/cos pairs, 1/4 of the memory of a full table) by
 * symmetry when 8 divides n; otherwise the table holds every k.
 */
typedef struct { cpx *t; int64_t n, len; int oct, cached; } tw_t;

static int64_t tw_bytes(int64_t n) { return (n % 8 == 0 ? n / 8 + 1 : n / 2 + 1) * (int64_t)sizeof(cpx); }

static int tw_init(tw_t *w, int64_t n, int cached)
{
    w->n = n;
    w->oct = n % 8 == 0;
    w->len = w->oct ? n / 8 + 1 : n / 2 + 1;
    w->cached = cached;
    w->t = (cpx *)pl_alloc(cached, w->len * (int64_t)sizeof(cpx));
    if (!w->t) return TSR_ENOMEM;
    for (int64_t k = 0; k < w->len; k++) {
        const double a = 2.0 * M_PI * (double)k / (double)n;
        w->t[k] = (cpx){cos(a), sin(a)};
    }
    return TSR_OK;
}

static void tw_free(tw_t *w) { pl_free(w->cached, w->t, w->len * (int64_t)sizeof(cpx)); }

/* ---------------------------------------------------------------- plan cache
 * NumPy (pocketfft) keeps recent plans; rebuilding one costs n cos/sin pairs, more than the transform itself at
 * 2^20 points (about 54 of 93 ms). Plans, Bluestein chirps and real-FFT twiddles are cached by (kind, n, direction)
 * up to TSR_FFT_CACHE_BYTES in all, least recently used out first. An entry is read-only and reference-counted: one
 * in use is never evicted, and a lookup that finds the cache full of busy entries builds a private copy. The cached
 * values are the ones a fresh build computes, so results are bit-identical with or without the cache. */
#define TSR_FFT_CACHE_SLOTS 16
#define TSR_FFT_CACHE_BYTES ((int64_t)64 << 20)

enum { K_PLAN, K_BLUE, K_RTW };

typedef struct {
    void *obj;        /* plan_t, blue_t or tw_t; NULL for a free slot */
    int kind, inverse, refs;
    int64_t n, bytes;
    uint64_t used;
} fft_slot;

static fft_slot g_fft_slots[TSR_FFT_CACHE_SLOTS];
static int64_t g_fft_bytes;
static uint64_t g_fft_clock;
#ifdef _WIN32
static SRWLOCK g_fft_lock = SRWLOCK_INIT;
#  define FFT_LOCK() AcquireSRWLockExclusive(&g_fft_lock)
#  define FFT_UNLOCK() ReleaseSRWLockExclusive(&g_fft_lock)
#else
static pthread_mutex_t g_fft_lock = PTHREAD_MUTEX_INITIALIZER;
#  define FFT_LOCK() pthread_mutex_lock(&g_fft_lock)
#  define FFT_UNLOCK() pthread_mutex_unlock(&g_fft_lock)
#endif

typedef struct { void *obj; int kind; fft_slot *slot; } fft_ref;   /* slot NULL: a private copy */

static int64_t obj_bytes(int kind, int64_t n)
{
    return kind == K_PLAN ? plan_bytes(n) : kind == K_BLUE ? blue_bytes(n) : tw_bytes(n);
}

static void obj_destroy(int kind, void *obj)
{
    if (kind == K_PLAN) plan_free((plan_t *)obj);
    else if (kind == K_BLUE) blue_free((blue_t *)obj);
    else tw_free((tw_t *)obj);
    free(obj);
}

static void *obj_build(int kind, int64_t n, int inverse, int cached, int *rc)
{
    void *obj = malloc(kind == K_PLAN ? sizeof(plan_t) : kind == K_BLUE ? sizeof(blue_t) : sizeof(tw_t));
    if (!obj) { *rc = TSR_ENOMEM; return NULL; }
    *rc = kind == K_PLAN ? plan_init((plan_t *)obj, n, inverse, cached)
        : kind == K_BLUE ? blue_init((blue_t *)obj, n, inverse, cached)
                         : tw_init((tw_t *)obj, n, cached);
    if (*rc != TSR_OK) { free(obj); return NULL; }
    return obj;
}

static fft_slot *fft_find(int kind, int64_t n, int inverse)
{
    for (int i = 0; i < TSR_FFT_CACHE_SLOTS; i++) {
        fft_slot *s = &g_fft_slots[i];
        if (s->obj && s->kind == kind && s->n == n && s->inverse == inverse) return s;
    }
    return NULL;
}

static int fft_acquire(int kind, int64_t n, int inverse, fft_ref *r)
{
    r->kind = kind;
    r->slot = NULL;
    FFT_LOCK();
    fft_slot *s = fft_find(kind, n, inverse);
    if (s) {
        s->refs++;
        s->used = ++g_fft_clock;
        r->obj = s->obj;
        r->slot = s;
        FFT_UNLOCK();
        return TSR_OK;
    }
    FFT_UNLOCK();
    /* build outside the lock: other lengths stay available meanwhile */
    const int64_t bytes = obj_bytes(kind, n);
    const int cacheable = bytes <= TSR_FFT_CACHE_BYTES;
    int rc;
    r->obj = obj_build(kind, n, inverse, cacheable, &rc);
    if (!r->obj) return rc;
    if (!cacheable) return TSR_OK;
    FFT_LOCK();
    if (!fft_find(kind, n, inverse)) {   /* another thread may have cached this length meanwhile: keep ours private */
        for (;;) {
            fft_slot *empty = NULL, *lru = NULL;
            for (int i = 0; i < TSR_FFT_CACHE_SLOTS; i++) {
                fft_slot *c = &g_fft_slots[i];
                if (!c->obj) { if (!empty) empty = c; }
                else if (c->refs == 0 && (!lru || c->used < lru->used)) lru = c;
            }
            if (empty && g_fft_bytes + bytes <= TSR_FFT_CACHE_BYTES) {
                *empty = (fft_slot){r->obj, kind, inverse, 1, n, bytes, ++g_fft_clock};
                g_fft_bytes += bytes;
                r->slot = empty;
                break;
            }
            if (!lru) break;               /* everything cached is in use: this one stays private */
            obj_destroy(lru->kind, lru->obj);
            g_fft_bytes -= lru->bytes;
            lru->obj = NULL;
        }
    }
    FFT_UNLOCK();
    return TSR_OK;
}

static void fft_release(fft_ref *r)
{
    if (!r->slot) { obj_destroy(r->kind, r->obj); return; }
    FFT_LOCK();
    r->slot->refs--;
    FFT_UNLOCK();
}

/* rows x n complex (or real, real_in) -> rows x n complex. The input is staged row by row through a buffer only when
   it overlaps the output (an in-place transform) or a real row feeds Bluestein; otherwise the plan reads it directly,
   which saves a pass over the data (and, for real input, the conversion to complex the caller would do). */
static int fft_rows(int64_t n, int64_t rows, int inverse, const double *in, int real_in, double *out)
{
    if (n < 1 || rows < 0 || !in || !out) return TSR_EARG;
    if (rows == 0) return TSR_OK;
    int64_t f[64], maxp;
    factorize(n, f, &maxp);
    const int use_blue = maxp > TSR_FFT_MAX_RADIX;
    const double scale = inverse ? 1.0 / (double)n : 1.0;
    const uintptr_t ib = (uintptr_t)in, ie = ib + (uintptr_t)(rows * n * (real_in ? 1 : 2)) * sizeof(double);
    const uintptr_t ob = (uintptr_t)out, oe = ob + (uintptr_t)(rows * n * 2) * sizeof(double);
    const int stage = (ib < oe && ob < ie) || (use_blue && real_in);

    fft_ref ref;
    int rc = fft_acquire(use_blue ? K_BLUE : K_PLAN, n, inverse, &ref);
    if (rc != TSR_OK) return rc;
    /* per-call memory: the row staging buffer, the generic butterfly's scratch and, for Bluestein, its two
       length-M work buffers */
    const int64_t M = use_blue ? ((const blue_t *)ref.obj)->M : 0;
    const int64_t nstage = stage ? n : 0, nscr = use_blue ? 0 : maxp + 1;
    const int64_t work = (nstage + nscr + 2 * M) * (int64_t)sizeof(cpx);
    cpx *tmp = work > 0 ? (cpx *)tsr_alloc(work) : NULL;
    if (work > 0 && !tmp) { fft_release(&ref); return TSR_ENOMEM; }
    cpx *scratch = tmp + nstage, *wa = scratch + nscr, *wb = wa + M;
    for (int64_t r = 0; r < rows; r++) {
        cpx *o = (cpx *)(out + 2 * n * r);
        const cpx *src = real_in ? NULL : (const cpx *)(in + 2 * n * r);
        if (stage) {
            if (real_in) { const double *x = in + n * r; for (int64_t i = 0; i < n; i++) tmp[i] = (cpx){x[i], 0.0}; }
            else memcpy(tmp, src, (size_t)n * sizeof(cpx));
            src = tmp;
        }
        if (use_blue) blue_run((const blue_t *)ref.obj, src, o, wa, wb);
        else if (real_in && !stage) run_plan_real((const plan_t *)ref.obj, in + n * r, o, inverse, scratch);
        else run_plan((const plan_t *)ref.obj, src, o, inverse, scratch);
    }
    if (tmp) tsr_free(tmp, work);
    fft_release(&ref);
    if (rc == TSR_OK && inverse) {
        for (int64_t i = 0; i < 2 * n * rows; i++) out[i] *= scale;
    }
    return rc;
}

int tsr_fft(int64_t n, int64_t rows, int inverse, const double *in, double *out)
{
    return fft_rows(n, rows, inverse, in, 0, out);
}

int tsr_fft_real(int64_t n, int64_t rows, int inverse, const double *in, double *out)
{
    return fft_rows(n, rows, inverse, in, 1, out);
}

/*
 * Real-input FFT (numpy.fft.rfft / irfft) at half the cost of a complex FFT.
 *
 * Even n: the n real samples are packed as m = n/2 complex values
 * z[j] = x[2j] + i x[2j+1], transformed with one complex FFT of length m, and
 * split into the spectra of the even and odd samples:
 *   E[k] = (Z[k] + conj(Z[m-k])) / 2,   O[k] = (Z[k] - conj(Z[m-k])) / 2i,
 *   X[k] = E[k] + W^k O[k],  W = exp(-2 pi i / n),  k = 0 .. m.
 * irfft runs the same steps backwards (imaginary parts of the DC and Nyquist
 * bins are ignored, as numpy does). Odd n uses the complex FFT.
 */
static inline cpx tw_get(const tw_t *w, int64_t k, double sign)
{
    cpx r;
    if (!w->oct) r = w->t[k];
    else {
        const int64_t q = w->n / 4, o = w->n / 8, h = w->n / 2;
        if (k <= o) r = w->t[k];
        else if (k <= q) { const cpx v = w->t[q - k]; r = (cpx){v.im, v.re}; }          /* pi/2 - phi */
        else if (k <= q + o) { const cpx v = w->t[k - q]; r = (cpx){-v.im, v.re}; }     /* pi/2 + phi */
        else { const cpx v = w->t[h - k]; r = (cpx){-v.re, v.im}; }                     /* pi - phi */
    }
    r.im *= sign;
    return r;
}

int tsr_rfft(int64_t n, int64_t rows, const double *in, double *out)
{
    if (n < 1 || rows < 0 || !in || !out) return TSR_EARG;
    if (rows == 0) return TSR_OK;
    const int64_t h = n / 2 + 1;
    if (n % 2 == 1) {
        cpx *buf = (cpx *)tsr_alloc(n * rows * (int64_t)sizeof(cpx));
        if (!buf) return TSR_ENOMEM;
        int rc = tsr_fft_real(n, rows, 0, in, (double *)buf);
        if (rc == TSR_OK)
            for (int64_t r = 0; r < rows; r++) memcpy(out + 2 * h * r, buf + n * r, (size_t)h * sizeof(cpx));
        tsr_free(buf, n * rows * (int64_t)sizeof(cpx));
        return rc;
    }
    const int64_t m = n / 2;
    cpx *Z = (cpx *)tsr_alloc(m * rows * (int64_t)sizeof(cpx));
    if (!Z) return TSR_ENOMEM;
    fft_ref wref;
    if (fft_acquire(K_RTW, n, 0, &wref) != TSR_OK) { tsr_free(Z, m * rows * (int64_t)sizeof(cpx)); return TSR_ENOMEM; }
    const tw_t W = *(const tw_t *)wref.obj;
    /* the input read as m complex values (x0 + i x1, x2 + i x3, ...), transformed straight into Z */
    int rc = tsr_fft(m, rows, 0, in, (double *)Z);
    if (rc == TSR_OK) {
        for (int64_t r = 0; r < rows; r++) {
            const cpx *z = Z + m * r;
            cpx *X = (cpx *)(out + 2 * h * r);
            for (int64_t k = 0; k <= m; k++) {
                const cpx a = z[k % m], b = z[(m - k) % m];                 /* Z[k], Z[m-k] */
                const double er = 0.5 * (a.re + b.re), ei = 0.5 * (a.im - b.im);  /* E = (a + conj b)/2 */
                const double orr = 0.5 * (a.im + b.im), oi = -0.5 * (a.re - b.re); /* O = (a - conj b)/(2i) */
                const cpx wk = tw_get(&W, k, -1.0);
                const double c = wk.re, s = wk.im;
                X[k].re = er + (c * orr - s * oi);
                X[k].im = ei + (c * oi + s * orr);
            }
        }
    }
    fft_release(&wref);
    tsr_free(Z, m * rows * (int64_t)sizeof(cpx));
    return rc;
}

int tsr_irfft(int64_t n, int64_t rows, const double *in, double *out)
{
    if (n < 1 || rows < 0 || !in || !out) return TSR_EARG;
    if (rows == 0) return TSR_OK;
    const int64_t h = n / 2 + 1;
    if (n % 2 == 1) {
        /* rebuild the Hermitian spectrum and run the complex inverse */
        cpx *buf = (cpx *)tsr_alloc(n * rows * (int64_t)sizeof(cpx));
        if (!buf) return TSR_ENOMEM;
        for (int64_t r = 0; r < rows; r++) {
            const cpx *X = (const cpx *)(in + 2 * h * r);
            cpx *f = buf + n * r;
            f[0] = (cpx){X[0].re, 0.0};
            for (int64_t k = 1; k < h; k++) { f[k] = X[k]; f[n - k] = (cpx){X[k].re, -X[k].im}; }
        }
        int rc = tsr_fft(n, rows, 1, (double *)buf, (double *)buf);
        if (rc == TSR_OK) for (int64_t i = 0; i < n * rows; i++) out[i] = buf[i].re;
        tsr_free(buf, n * rows * (int64_t)sizeof(cpx));
        return rc;
    }
    const int64_t m = n / 2;
    cpx *Z = (cpx *)tsr_alloc(m * rows * (int64_t)sizeof(cpx));
    if (!Z) return TSR_ENOMEM;
    fft_ref wref;
    if (fft_acquire(K_RTW, n, 0, &wref) != TSR_OK) { tsr_free(Z, m * rows * (int64_t)sizeof(cpx)); return TSR_ENOMEM; }
    const tw_t W = *(const tw_t *)wref.obj;
    for (int64_t r = 0; r < rows; r++) {
        const cpx *X = (const cpx *)(in + 2 * h * r);
        cpx *z = Z + m * r;
        for (int64_t k = 0; k < m; k++) {
            cpx a = X[k], b = X[m - k];                                    /* X[k], X[m-k] */
            if (k == 0) { a.im = 0.0; b.im = 0.0; }                         /* DC and Nyquist are real */
            const double er = 0.5 * (a.re + b.re), ei = 0.5 * (a.im - b.im);      /* E = (a + conj b)/2 */
            const double dr = 0.5 * (a.re - b.re), di = 0.5 * (a.im + b.im);      /* (a - conj b)/2 = W^k O */
            const cpx wk = tw_get(&W, k, 1.0);                             /* O = W^-k (a - conj b)/2 */
            const double c = wk.re, s = wk.im;
            const double orr = c * dr - s * di, oi = c * di + s * dr;
            z[k].re = er - oi;                                              /* Z = E + i O */
            z[k].im = ei + orr;
        }
    }
    /* m complex results are the n real outputs (interleaved re, im = x[2j], x[2j+1]): transform straight into out */
    int rc = tsr_fft(m, rows, 1, (double *)Z, out);
    fft_release(&wref);
    tsr_free(Z, m * rows * (int64_t)sizeof(cpx));
    return rc;
}
