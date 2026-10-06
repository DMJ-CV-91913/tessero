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
#include <string.h>

#define TSR_FFT_MAX_RADIX 61

typedef struct { double re, im; } cpx;

typedef struct {
    int64_t n;
    int nf;
    int64_t factors[64];
    cpx *tw;          /* tw[k] = exp(sign * 2 pi i k / n) */
    cpx *scratch;     /* generic butterfly scratch, size max radix */
    cpx *ltw[64];     /* per-level contiguous twiddles for radix 2-5: ltw[k*(p-1) + q-1] = tw[q*k*fstride] */
    int64_t ltw_len[64];
} plan_t;

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

static void butterfly(const plan_t *pl, const cpx *lt, cpx *out, int64_t fstride, int64_t m, int64_t p, int inverse)
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
        cpx *t = pl->scratch;
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

static void rec(const plan_t *pl, cpx *out, const cpx *in, int64_t fstride, int64_t istride, int fi, int inverse)
{
    int64_t p = pl->factors[fi];
    int64_t m = 1;
    for (int j = fi + 1; j < pl->nf; j++) m *= pl->factors[j];
    if (m == 1) {
        for (int64_t q = 0; q < p; q++) out[q] = in[q * istride];
    } else {
        for (int64_t q = 0; q < p; q++)
            rec(pl, out + q * m, in + q * istride, fstride * p, istride * p, fi + 1, inverse);
    }
    butterfly(pl, pl->ltw[fi], out, fstride, m, p, inverse);
}

static int plan_init(plan_t *pl, int64_t n, int inverse, int64_t *maxp)
{
    pl->n = n;
    pl->nf = factorize(n, pl->factors, maxp);
    pl->tw = (cpx *)tsr_alloc(n * (int64_t)sizeof(cpx));
    pl->scratch = (cpx *)tsr_alloc((*maxp + 1) * (int64_t)sizeof(cpx));
    if (!pl->tw || !pl->scratch) {
        if (pl->tw) tsr_free(pl->tw, n * (int64_t)sizeof(cpx));
        if (pl->scratch) tsr_free(pl->scratch, (*maxp + 1) * (int64_t)sizeof(cpx));
        return TSR_ENOMEM;
    }
    twiddles(pl->tw, n, inverse ? 1.0 : -1.0);
    int64_t fstride = 1;
    for (int fi = 0; fi < pl->nf; fi++) {
        int64_t p = pl->factors[fi];
        int64_t m = n / (fstride * p);
        pl->ltw[fi] = NULL;
        pl->ltw_len[fi] = 0;
        if (p <= 5 && m > 0) {
            int64_t len = m * (p - 1);
            cpx *lt = (cpx *)tsr_alloc(len * (int64_t)sizeof(cpx));
            if (!lt) {
                for (int j = 0; j < fi; j++) if (pl->ltw[j]) tsr_free(pl->ltw[j], pl->ltw_len[j] * (int64_t)sizeof(cpx));
                tsr_free(pl->tw, n * (int64_t)sizeof(cpx));
                tsr_free(pl->scratch, (*maxp + 1) * (int64_t)sizeof(cpx));
                return TSR_ENOMEM;
            }
            for (int64_t k = 0; k < m; k++)
                for (int64_t q = 1; q < p; q++) lt[k * (p - 1) + q - 1] = pl->tw[q * k * fstride];
            pl->ltw[fi] = lt;
            pl->ltw_len[fi] = len;
        }
        fstride *= p;
    }
    return TSR_OK;
}

static void plan_free(plan_t *pl, int64_t maxp)
{
    for (int fi = 0; fi < pl->nf; fi++)
        if (pl->ltw[fi]) tsr_free(pl->ltw[fi], pl->ltw_len[fi] * (int64_t)sizeof(cpx));
    tsr_free(pl->tw, pl->n * (int64_t)sizeof(cpx));
    tsr_free(pl->scratch, (maxp + 1) * (int64_t)sizeof(cpx));
}

/* Unnormalised transform of one row via the mixed-radix plan (in != out). */
static void run_plan(const plan_t *pl, const cpx *in, cpx *out, int inverse)
{
    if (pl->n == 1) { out[0] = in[0]; return; }
    rec(pl, out, in, 1, 1, 0, inverse);
}

/* Bluestein: X_k = w_k * sum_j (x_j w_j) conj(w_{k-j}), w_k = exp(-/+ i pi k^2 / n). */
typedef struct {
    int64_t n, M, maxp;
    plan_t fwd, inv;
    cpx *w, *bf, *a, *b;
} blue_t;

static int blue_init(blue_t *bl, int64_t n, int inverse)
{
    int64_t M = 1;
    while (M < 2 * n - 1) M <<= 1;
    bl->n = n;
    bl->M = M;
    int64_t mp;
    if (plan_init(&bl->fwd, M, 0, &mp) != TSR_OK) return TSR_ENOMEM;
    if (plan_init(&bl->inv, M, 1, &bl->maxp) != TSR_OK) { plan_free(&bl->fwd, mp); return TSR_ENOMEM; }
    bl->maxp = mp;
    int64_t bytes = (n + 3 * M) * (int64_t)sizeof(cpx);
    cpx *mem = (cpx *)tsr_alloc(bytes);
    if (!mem) { plan_free(&bl->fwd, mp); plan_free(&bl->inv, mp); return TSR_ENOMEM; }
    bl->w = mem;
    bl->bf = mem + n;
    bl->a = mem + n + M;
    bl->b = mem + n + 2 * M;
    const double sign = inverse ? 1.0 : -1.0;
    for (int64_t k = 0; k < n; k++) {
        /* k^2 mod 2n keeps the angle small and exact */
        uint64_t kk = (uint64_t)k;
        uint64_t k2 = (uint64_t)(((unsigned __int128)kk * kk) % (unsigned __int128)(2 * (uint64_t)n));
        double a = sign * M_PI * (double)k2 / (double)n;
        bl->w[k] = (cpx){cos(a), sin(a)};
    }
    memset(bl->b, 0, (size_t)M * sizeof(cpx));
    bl->b[0] = (cpx){bl->w[0].re, -bl->w[0].im};
    for (int64_t k = 1; k < n; k++) {
        cpx c = {bl->w[k].re, -bl->w[k].im};
        bl->b[k] = c;
        bl->b[M - k] = c;
    }
    run_plan(&bl->fwd, bl->b, bl->bf, 0);
    return TSR_OK;
}

static void blue_run(blue_t *bl, const cpx *in, cpx *out)
{
    const int64_t n = bl->n, M = bl->M;
    for (int64_t k = 0; k < n; k++) bl->b[k] = cmul(in[k], bl->w[k]);
    memset(bl->b + n, 0, (size_t)(M - n) * sizeof(cpx));
    run_plan(&bl->fwd, bl->b, bl->a, 0);
    for (int64_t k = 0; k < M; k++) bl->a[k] = cmul(bl->a[k], bl->bf[k]);
    run_plan(&bl->inv, bl->a, bl->b, 1);
    const double s = 1.0 / (double)M;
    for (int64_t k = 0; k < n; k++) {
        cpx v = {bl->b[k].re * s, bl->b[k].im * s};
        out[k] = cmul(v, bl->w[k]);
    }
}

static void blue_free(blue_t *bl)
{
    plan_free(&bl->fwd, bl->maxp);
    plan_free(&bl->inv, bl->maxp);
    tsr_free(bl->w, (bl->n + 3 * bl->M) * (int64_t)sizeof(cpx));
}

int tsr_fft(int64_t n, int64_t rows, int inverse, const double *in, double *out)
{
    if (n < 1 || rows < 0 || !in || !out) return TSR_EARG;
    if (rows == 0) return TSR_OK;
    int64_t f[64], maxp;
    factorize(n, f, &maxp);
    const int use_blue = maxp > TSR_FFT_MAX_RADIX;
    const double scale = inverse ? 1.0 / (double)n : 1.0;

    /* input may alias output: stage each row through a temporary */
    cpx *tmp = (cpx *)tsr_alloc(n * (int64_t)sizeof(cpx));
    if (!tmp) return TSR_ENOMEM;
    int rc = TSR_OK;
    if (use_blue) {
        blue_t bl;
        if ((rc = blue_init(&bl, n, inverse)) == TSR_OK) {
            for (int64_t r = 0; r < rows; r++) {
                memcpy(tmp, in + 2 * n * r, (size_t)n * sizeof(cpx));
                blue_run(&bl, tmp, (cpx *)(out + 2 * n * r));
            }
            blue_free(&bl);
        }
    } else {
        plan_t pl;
        int64_t mp;
        if ((rc = plan_init(&pl, n, inverse, &mp)) == TSR_OK) {
            for (int64_t r = 0; r < rows; r++) {
                memcpy(tmp, in + 2 * n * r, (size_t)n * sizeof(cpx));
                run_plan(&pl, tmp, (cpx *)(out + 2 * n * r), inverse);
            }
            plan_free(&pl, mp);
        }
    }
    tsr_free(tmp, n * (int64_t)sizeof(cpx));
    if (rc == TSR_OK && inverse) {
        for (int64_t i = 0; i < 2 * n * rows; i++) out[i] *= scale;
    }
    return rc;
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
/*
 * Twiddles w[k] = exp(sign * 2 pi i k / n), k = 0 .. n/2, from a table of the
 * first octant (n/8 + 1 sin/cos pairs, 1/4 of the memory of a full table) by
 * symmetry when 8 divides n; otherwise the table holds every k.
 */
typedef struct { cpx *t; int64_t n, len; int oct; } tw_t;

static int tw_init(tw_t *w, int64_t n)
{
    w->n = n;
    w->oct = n % 8 == 0;
    w->len = w->oct ? n / 8 + 1 : n / 2 + 1;
    w->t = (cpx *)tsr_alloc(w->len * (int64_t)sizeof(cpx));
    if (!w->t) return TSR_ENOMEM;
    for (int64_t k = 0; k < w->len; k++) {
        const double a = 2.0 * M_PI * (double)k / (double)n;
        w->t[k] = (cpx){cos(a), sin(a)};
    }
    return TSR_OK;
}

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

static void tw_free(tw_t *w) { tsr_free(w->t, w->len * (int64_t)sizeof(cpx)); }

int tsr_rfft(int64_t n, int64_t rows, const double *in, double *out)
{
    if (n < 1 || rows < 0 || !in || !out) return TSR_EARG;
    if (rows == 0) return TSR_OK;
    const int64_t h = n / 2 + 1;
    if (n % 2 == 1) {
        cpx *buf = (cpx *)tsr_alloc(n * rows * (int64_t)sizeof(cpx));
        if (!buf) return TSR_ENOMEM;
        for (int64_t i = 0; i < n * rows; i++) buf[i] = (cpx){in[i], 0.0};
        int rc = tsr_fft(n, rows, 0, (double *)buf, (double *)buf);
        if (rc == TSR_OK)
            for (int64_t r = 0; r < rows; r++) memcpy(out + 2 * h * r, buf + n * r, (size_t)h * sizeof(cpx));
        tsr_free(buf, n * rows * (int64_t)sizeof(cpx));
        return rc;
    }
    const int64_t m = n / 2;
    cpx *Z = (cpx *)tsr_alloc(m * rows * (int64_t)sizeof(cpx));
    if (!Z) return TSR_ENOMEM;
    memcpy(Z, in, (size_t)(n * rows) * sizeof(double));              /* (x0 + i x1, x2 + i x3, ...) */
    tw_t W;
    if (tw_init(&W, n) != TSR_OK) { tsr_free(Z, m * rows * (int64_t)sizeof(cpx)); return TSR_ENOMEM; }
    int rc = tsr_fft(m, rows, 0, (double *)Z, (double *)Z);
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
    tw_free(&W);
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
    tw_t W;
    if (tw_init(&W, n) != TSR_OK) { tsr_free(Z, m * rows * (int64_t)sizeof(cpx)); return TSR_ENOMEM; }
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
    int rc = tsr_fft(m, rows, 1, (double *)Z, (double *)Z);
    if (rc == TSR_OK) memcpy(out, Z, (size_t)(n * rows) * sizeof(double));
    tw_free(&W);
    tsr_free(Z, m * rows * (int64_t)sizeof(cpx));
    return rc;
}
