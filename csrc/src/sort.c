/*
 * Sorting: radix sorts on order-preserving integer keys (MSD + LSD hybrid for
 * values, stable LSD for argsort).
 *
 * Floats are mapped to unsigned keys whose integer order equals the numeric
 * order (flip all bits of negatives, set the sign bit of positives); NaNs are
 * moved to the end first, as NumPy orders them. Radix sort is O(n), stable,
 * and skips byte positions where every key agrees, so int arrays with a small
 * range take one or two passes. argsort is stable (NumPy's kind='stable'):
 * -0.0 and +0.0 compare equal and keep their input order.
 */
#include "internal.h"
#include <math.h>
#include <stdlib.h>

static inline uint64_t key_f64(double v, int zero_equal)
{
    uint64_t u;
    if (zero_equal && v == 0.0) v = 0.0;
    memcpy(&u, &v, 8);
    return (u >> 63) ? ~u : (u | 0x8000000000000000ULL);
}
static inline double unkey_f64(uint64_t k)
{
    uint64_t u = (k >> 63) ? (k & 0x7fffffffffffffffULL) : ~k;
    double v;
    memcpy(&v, &u, 8);
    return v;
}
static inline uint64_t key_f32(float v, int zero_equal)
{
    uint32_t u;
    if (zero_equal && v == 0.0f) v = 0.0f;
    memcpy(&u, &v, 4);
    return (u >> 31) ? (uint32_t)~u : (u | 0x80000000u);
}
static inline float unkey_f32(uint64_t k)
{
    uint32_t kk = (uint32_t)k;
    uint32_t u = (kk >> 31) ? (kk & 0x7fffffffu) : ~kk;
    float v;
    memcpy(&v, &u, 4);
    return v;
}

/* Stable LSD radix sort of keys (and optional payload) over the low `bits` bits, 11-bit digits
   (6 passes for 64-bit keys, 3 for 32-bit), skipping digits on which every key agrees. */
#define RBITS 11
#define RSIZE (1 << RBITS)
#define RMASK (RSIZE - 1)
static int radix(uint64_t *keys, int64_t *payload, int64_t n, int bytes)
{
    if (n < 2) return TSR_OK;
    const int passes = (bytes * 8 + RBITS - 1) / RBITS;
    uint64_t *k2 = (uint64_t *)tsr_alloc(n * 8);
    int64_t *p2 = payload ? (int64_t *)tsr_alloc(n * 8) : NULL;
    int64_t *hist = (int64_t *)calloc((size_t)passes * RSIZE, sizeof(int64_t));
    if (!k2 || (payload && !p2) || !hist) {
        if (k2) tsr_free(k2, n * 8);
        if (p2) tsr_free(p2, n * 8);
        free(hist);
        return TSR_ENOMEM;
    }
    for (int64_t i = 0; i < n; i++) {
        uint64_t k = keys[i];
        for (int b = 0; b < passes; b++) hist[b * RSIZE + ((k >> (RBITS * b)) & RMASK)]++;
    }
    uint64_t *src = keys, *dst = k2;
    int64_t *psrc = payload, *pdst = p2;
    for (int b = 0; b < passes; b++) {
        int64_t *h = hist + b * RSIZE;
        int trivial = 0;
        for (int d = 0; d < RSIZE; d++) if (h[d] == n) { trivial = 1; break; }
        if (trivial) continue;
        int64_t sum = 0;
        for (int d = 0; d < RSIZE; d++) { int64_t c = h[d]; h[d] = sum; sum += c; }
        const int shift = RBITS * b;
        if (psrc) {
            for (int64_t i = 0; i < n; i++) {
                int64_t pos = h[(src[i] >> shift) & RMASK]++;
                dst[pos] = src[i];
                pdst[pos] = psrc[i];
            }
            int64_t *pt = psrc; psrc = pdst; pdst = pt;
        } else {
            for (int64_t i = 0; i < n; i++) dst[h[(src[i] >> shift) & RMASK]++] = src[i];
        }
        uint64_t *t = src; src = dst; dst = t;
    }
    if (src != keys) {
        memcpy(keys, src, (size_t)n * 8);
        if (payload) memcpy(payload, psrc, (size_t)n * 8);
    }
    tsr_free(k2, n * 8);
    if (p2) tsr_free(p2, n * 8);
    free(hist);
    return TSR_OK;
}

/*
 * Value sort (tsr_sort): stability does not matter, so a cache-friendly hybrid
 * beats plain LSD. One most-significant-digit pass (11 bits) scatters the keys
 * into up to 2048 buckets; each bucket, now small enough for L1/L2, is sorted
 * by LSD passes over 8-bit digits (skipping digits on which it agrees), or by
 * insertion sort below 32 keys. Buckets are independent, so with threads > 1
 * they are sorted in parallel. About 25% faster than 11-bit LSD single-threaded
 * on random float64, and it scales with threads.
 */
static void ins_sort(uint64_t *a, int64_t n)
{
    for (int64_t i = 1; i < n; i++) {
        const uint64_t k = a[i];
        int64_t j = i - 1;
        while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
        a[j + 1] = k;
    }
}

static void lsd8(uint64_t *a, uint64_t *s, int64_t n, int hibit)
{
    const int passes = (hibit + 7) / 8;
    int64_t h[8][256];
    memset(h, 0, sizeof(int64_t) * 256 * (size_t)passes);
    for (int64_t i = 0; i < n; i++) {
        const uint64_t k = a[i];
        for (int p = 0; p < passes; p++) h[p][(k >> (8 * p)) & 255]++;
    }
    uint64_t *src = a, *dst = s;
    for (int p = 0; p < passes; p++) {
        int64_t *hp = h[p];
        const int sh = 8 * p;
        if (hp[(src[0] >> sh) & 255] == n) continue;
        int64_t sum = 0;
        for (int d = 0; d < 256; d++) { const int64_t c = hp[d]; hp[d] = sum; sum += c; }
        for (int64_t i = 0; i < n; i++) dst[hp[(src[i] >> sh) & 255]++] = src[i];
        uint64_t *t = src; src = dst; dst = t;
    }
    if (src != a) memcpy(a, src, (size_t)n * 8);
}

#define MSD_BITS 11
static void msd_sort(uint64_t *a, uint64_t *s, int64_t n, int hibit, int threads)
{
    if (n <= 32) { ins_sort(a, n); return; }
    if (hibit <= 0) return;                                       /* all remaining keys are equal */
    if (n <= 4096 || hibit <= 16) { lsd8(a, s, n, hibit); return; }
    const int lo = hibit > MSD_BITS ? hibit - MSD_BITS : 0, w = hibit - lo, nb = 1 << w;
    const uint64_t mask = ((uint64_t)1 << w) - 1;
    int64_t cnt[1 << MSD_BITS], off[1 << MSD_BITS], pos[1 << MSD_BITS];
    memset(cnt, 0, sizeof(int64_t) * (size_t)nb);
    for (int64_t i = 0; i < n; i++) cnt[(a[i] >> lo) & mask]++;
    int64_t sum = 0;
    for (int d = 0; d < nb; d++) {
        if (cnt[d] == n) { msd_sort(a, s, n, lo, threads); return; }   /* this digit is constant: skip it */
        off[d] = pos[d] = sum;
        sum += cnt[d];
    }
    for (int64_t i = 0; i < n; i++) s[pos[(a[i] >> lo) & mask]++] = a[i];
    memcpy(a, s, (size_t)n * 8);
    if (threads < 2 || n < TSR_PAR_MIN) threads = 1;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16) if (threads > 1)
#endif
    for (int d = 0; d < nb; d++)
        if (cnt[d] > 1) msd_sort(a + off[d], s + off[d], cnt[d], lo, 1);
}

static int value_sort(uint64_t *keys, int64_t n)
{
    if (n < 2) return TSR_OK;
    uint64_t *s = (uint64_t *)tsr_alloc(n * 8);
    if (!s) return TSR_ENOMEM;
    msd_sort(keys, s, n, 64, tsr_get_threads());
    tsr_free(s, n * 8);
    return TSR_OK;
}

int tsr_sort(int dtype, int64_t n, void *data)
{
    if (n < 2) return TSR_OK;
    if (dtype == TSR_U8 || dtype == TSR_BOOL) {           /* counting sort */
        int64_t c[256] = {0};
        uint8_t *d = (uint8_t *)data;
        for (int64_t i = 0; i < n; i++) c[d[i]]++;
        int64_t k = 0;
        for (int v = 0; v < 256; v++) for (int64_t j = 0; j < c[v]; j++) d[k++] = (uint8_t)v;
        return TSR_OK;
    }
    uint64_t *keys = (uint64_t *)tsr_alloc(n * 8);
    if (!keys) return TSR_ENOMEM;
    int64_t m = 0, nans = 0;
    int rc;
    switch (dtype) {
    case TSR_F64: {
        double *d = (double *)data;
        for (int64_t i = 0; i < n; i++) { if (isnan(d[i])) nans++; else keys[m++] = key_f64(d[i], 0); }
        rc = value_sort(keys, m);
        for (int64_t i = 0; i < m; i++) d[i] = unkey_f64(keys[i]);
        for (int64_t i = m; i < n; i++) d[i] = NAN;
        break;
    }
    case TSR_F32: {
        float *d = (float *)data;
        for (int64_t i = 0; i < n; i++) { if (isnan(d[i])) nans++; else keys[m++] = key_f32(d[i], 0); }
        rc = value_sort(keys, m);
        for (int64_t i = 0; i < m; i++) d[i] = unkey_f32(keys[i]);
        for (int64_t i = m; i < n; i++) d[i] = NAN;
        break;
    }
    case TSR_I64: {
        int64_t *d = (int64_t *)data;
        for (int64_t i = 0; i < n; i++) keys[i] = (uint64_t)d[i] ^ 0x8000000000000000ULL;
        rc = value_sort(keys, n);
        for (int64_t i = 0; i < n; i++) d[i] = (int64_t)(keys[i] ^ 0x8000000000000000ULL);
        break;
    }
    case TSR_I32: {
        int32_t *d = (int32_t *)data;
        for (int64_t i = 0; i < n; i++) keys[i] = (uint32_t)d[i] ^ 0x80000000u;
        rc = value_sort(keys, n);
        for (int64_t i = 0; i < n; i++) d[i] = (int32_t)((uint32_t)keys[i] ^ 0x80000000u);
        break;
    }
    default:
        rc = TSR_ETYPE;
    }
    (void)nans;
    tsr_free(keys, n * 8);
    return rc;
}

int tsr_argsort(int dtype, int64_t n, const void *data, int64_t *out)
{
    if (n <= 0) return TSR_OK;
    uint64_t *keys = (uint64_t *)tsr_alloc(n * 8);
    if (!keys) return TSR_ENOMEM;
    int64_t m = 0, tail = n;
    int bytes = 8;
    /* NaN positions go to the end in input order; the rest are radix-sorted with their indices */
    switch (dtype) {
    case TSR_F64: {
        const double *d = (const double *)data;
        int64_t nn = 0;
        for (int64_t i = 0; i < n; i++) if (isnan(d[i])) nn++;
        tail = n - nn;
        int64_t t = tail;
        for (int64_t i = 0; i < n; i++) {
            if (isnan(d[i])) out[t++] = i;
            else { keys[m] = key_f64(d[i], 1); out[m++] = i; }
        }
        break;
    }
    case TSR_F32: {
        const float *d = (const float *)data;
        int64_t nn = 0;
        for (int64_t i = 0; i < n; i++) if (isnan(d[i])) nn++;
        tail = n - nn;
        int64_t t = tail;
        for (int64_t i = 0; i < n; i++) {
            if (isnan(d[i])) out[t++] = i;
            else { keys[m] = key_f32(d[i], 1); out[m++] = i; }
        }
        bytes = 4;
        break;
    }
    case TSR_I64: {
        const int64_t *d = (const int64_t *)data;
        for (int64_t i = 0; i < n; i++) { keys[i] = (uint64_t)d[i] ^ 0x8000000000000000ULL; out[i] = i; }
        m = n;
        break;
    }
    case TSR_I32: {
        const int32_t *d = (const int32_t *)data;
        for (int64_t i = 0; i < n; i++) { keys[i] = (uint32_t)d[i] ^ 0x80000000u; out[i] = i; }
        m = n;
        bytes = 4;
        break;
    }
    case TSR_U8: case TSR_BOOL: {
        const uint8_t *d = (const uint8_t *)data;
        for (int64_t i = 0; i < n; i++) { keys[i] = d[i]; out[i] = i; }
        m = n;
        bytes = 1;
        break;
    }
    default:
        tsr_free(keys, n * 8);
        return TSR_ETYPE;
    }
    int rc = radix(keys, out, m, bytes);
    tsr_free(keys, n * 8);
    return rc;
}
