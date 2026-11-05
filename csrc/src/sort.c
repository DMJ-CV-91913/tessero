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

/*
 * tsr_sort_simd's sorter: an AVX2 quicksort of the same keys tsr_sort radix-sorts, so the two return the same bits.
 * Keys are compared as int64 (the order-preserving uint64 key with its top bit flipped). Partition: four vectors
 * (16 keys) per step from the end with less free space, lanes below the pivot moved left by a permutation table
 * and stored from both ends, 16 keys held back at each end so a full store always fits, branchless scalar tails.
 * Pivot: median of nine. Keys equal to a minimum pivot are split off in one more pass (many duplicates stay linear);
 * a depth limit falls back to heapsort. Leaves of up to 16 keys: Batcher's network. Already ascending input
 * returns at once and non-increasing input is reversed. Faster than the radix sort on random, duplicate-heavy and
 * sorted data, close to it on large nearly sorted or interleaved data; its time depends on the input and on AVX2,
 * so it is a separate entry point and the radix sort stays the default (docs/project/benchmarks.md).
 */
#if defined(__GNUC__) && defined(__x86_64__)                      /* GCC and clang; MSVC uses the radix sort */
#include <immintrin.h>
#define TSR_HAVE_AVX2_SORT 1
/* scratch arrays filled before use: no -ftrivial-auto-var-init zero fill per call (csrc/Makefile HARDEN) */
#if defined(__has_attribute)
#  if __has_attribute(uninitialized)
#    define TSR_NOINIT __attribute__((uninitialized))
#  endif
#endif
#ifndef TSR_NOINIT
#  define TSR_NOINIT
#endif
#define QS_LEAF 16
/* for each 4-bit lane mask (lanes below the pivot), the 32-bit-element permutation that moves those lanes first */
static const int32_t qs_perm[16][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {2, 3, 0, 1, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {4, 5, 0, 1, 2, 3, 6, 7}, {0, 1, 4, 5, 2, 3, 6, 7}, {2, 3, 4, 5, 0, 1, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {6, 7, 0, 1, 2, 3, 4, 5}, {0, 1, 6, 7, 2, 3, 4, 5}, {2, 3, 6, 7, 0, 1, 4, 5}, {0, 1, 2, 3, 6, 7, 4, 5},
    {4, 5, 6, 7, 0, 1, 2, 3}, {0, 1, 4, 5, 6, 7, 2, 3}, {2, 3, 4, 5, 6, 7, 0, 1}, {0, 1, 2, 3, 4, 5, 6, 7},
};

static void qs_sift(int64_t *a, int64_t n, int64_t i)
{
    for (;;) {
        int64_t c = 2 * i + 1;
        if (c >= n) return;
        if (c + 1 < n && a[c + 1] > a[c]) c++;
        if (a[i] >= a[c]) return;
        const int64_t t = a[i]; a[i] = a[c]; a[c] = t; i = c;
    }
}
static void qs_heapsort(int64_t *a, int64_t n)
{
    for (int64_t i = n / 2 - 1; i >= 0; i--) qs_sift(a, n, i);
    for (int64_t e = n - 1; e > 0; e--) { const int64_t t = a[0]; a[0] = a[e]; a[e] = t; qs_sift(a, e, 0); }
}

__attribute__((target("avx2"))) static inline void qs_store(__m256i v, __m256i pv, int64_t *a, int64_t *lw, int64_t *rw)
{
    const int m = _mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpgt_epi64(pv, v)));
    const __m256i w = _mm256_permutevar8x32_epi32(v, _mm256_loadu_si256((const __m256i *)qs_perm[m]));
    _mm256_storeu_si256((__m256i *)(a + *lw), w);
    _mm256_storeu_si256((__m256i *)(a + *rw - 4), w);
    *lw += __builtin_popcount(m);
    *rw -= 4 - __builtin_popcount(m);
}

/* keys < q to the left, in place; returns how many */
__attribute__((target("avx2"))) static int64_t qs_partition(int64_t *a, int64_t n, int64_t q)
{
    if (n < 64) {
        TSR_NOINIT int64_t buf[64];
        int64_t lw = 0, rw = n;
        memcpy(buf, a, (size_t)n * 8);
        for (int64_t i = 0; i < n; i++) { const int64_t x = buf[i], lt = x < q; a[lw] = x; a[rw - 1] = x; lw += lt; rw -= 1 - lt; }
        return lw;
    }
    const __m256i pv = _mm256_set1_epi64x(q);
    TSR_NOINIT int64_t hold[32];
    memcpy(hold, a, 16 * 8);
    memcpy(hold + 16, a + n - 16, 16 * 8);
    int64_t lr = 16, rr = n - 16, lw = 0, rw = n;
    while (rr - lr >= 16) {
        const int64_t *src;
        if (lr - lw <= rw - rr) { src = a + lr; lr += 16; } else { rr -= 16; src = a + rr; }
        const __m256i v0 = _mm256_loadu_si256((const __m256i *)src), v1 = _mm256_loadu_si256((const __m256i *)(src + 4));
        const __m256i v2 = _mm256_loadu_si256((const __m256i *)(src + 8)), v3 = _mm256_loadu_si256((const __m256i *)(src + 12));
        qs_store(v0, pv, a, &lw, &rw); qs_store(v1, pv, a, &lw, &rw);
        qs_store(v2, pv, a, &lw, &rw); qs_store(v3, pv, a, &lw, &rw);
    }
    while (rr - lr >= 4) {                                     /* free space >= 32 + read: single vectors are safe */
        __m256i v;
        if (lr - lw <= rw - rr) { v = _mm256_loadu_si256((const __m256i *)(a + lr)); lr += 4; }
        else { rr -= 4; v = _mm256_loadu_si256((const __m256i *)(a + rr)); }
        qs_store(v, pv, a, &lw, &rw);
    }
    TSR_NOINIT int64_t buf[36];
    int nb = 0;
    for (int64_t i = lr; i < rr; i++) buf[nb++] = a[i];
    for (int i = 0; i < 32; i++) buf[nb++] = hold[i];
    for (int i = 0; i < nb; i++) { const int64_t x = buf[i], lt = x < q; a[lw] = x; a[rw - 1] = x; lw += lt; rw -= 1 - lt; }
    return lw;
}

static inline int64_t qs_med3(int64_t x, int64_t y, int64_t z)
{
    return x < y ? (y < z ? y : (x < z ? z : x)) : (x < z ? x : (y < z ? z : y));
}

#define QS_CAS(x, y) do { const int64_t a_ = v[x], b_ = v[y]; v[x] = a_ < b_ ? a_ : b_; v[y] = a_ < b_ ? b_ : a_; } while (0)
/* up to 16 keys: Batcher's odd-even merge network (63 compare-exchanges, branchless), padded with INT64_MAX */
static void qs_leaf(int64_t *a, int64_t n)
{
    if (n < 2) return;
    if (n <= 6) {
        for (int64_t i = 1; i < n; i++) {
            const int64_t k = a[i];
            int64_t j = i - 1;
            while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
            a[j + 1] = k;
        }
        return;
    }
    TSR_NOINIT int64_t v[16];
    for (int64_t i = 0; i < n; i++) v[i] = a[i];
    for (int64_t i = n; i < 16; i++) v[i] = INT64_MAX;
    QS_CAS(0, 1); QS_CAS(2, 3); QS_CAS(4, 5); QS_CAS(6, 7); QS_CAS(8, 9); QS_CAS(10, 11); QS_CAS(12, 13); QS_CAS(14, 15);
    QS_CAS(0, 2); QS_CAS(1, 3); QS_CAS(4, 6); QS_CAS(5, 7); QS_CAS(8, 10); QS_CAS(9, 11); QS_CAS(12, 14); QS_CAS(13, 15);
    QS_CAS(1, 2); QS_CAS(5, 6); QS_CAS(9, 10); QS_CAS(13, 14);
    QS_CAS(0, 4); QS_CAS(1, 5); QS_CAS(2, 6); QS_CAS(3, 7); QS_CAS(8, 12); QS_CAS(9, 13); QS_CAS(10, 14); QS_CAS(11, 15);
    QS_CAS(2, 4); QS_CAS(3, 5); QS_CAS(10, 12); QS_CAS(11, 13);
    QS_CAS(1, 2); QS_CAS(3, 4); QS_CAS(5, 6); QS_CAS(9, 10); QS_CAS(11, 12); QS_CAS(13, 14);
    QS_CAS(0, 8); QS_CAS(1, 9); QS_CAS(2, 10); QS_CAS(3, 11); QS_CAS(4, 12); QS_CAS(5, 13); QS_CAS(6, 14); QS_CAS(7, 15);
    QS_CAS(4, 8); QS_CAS(5, 9); QS_CAS(6, 10); QS_CAS(7, 11);
    QS_CAS(2, 4); QS_CAS(3, 5); QS_CAS(6, 8); QS_CAS(7, 9); QS_CAS(10, 12); QS_CAS(11, 13);
    QS_CAS(1, 2); QS_CAS(3, 4); QS_CAS(5, 6); QS_CAS(7, 8); QS_CAS(9, 10); QS_CAS(11, 12); QS_CAS(13, 14);
    for (int64_t i = 0; i < n; i++) a[i] = v[i];
}
#undef QS_CAS

static void qs_rec(int64_t *a, int64_t n, int depth)
{
    while (n > QS_LEAF) {
        if (depth-- <= 0) { qs_heapsort(a, n); return; }
        const int64_t p = qs_med3(qs_med3(a[0], a[n / 8], a[n / 4]), qs_med3(a[3 * n / 8], a[n / 2], a[5 * n / 8]),
                                  qs_med3(a[3 * n / 4], a[7 * n / 8], a[n - 1]));
        int64_t m = qs_partition(a, n, p);
        if (m == 0) {                                          /* p is the minimum: split off every key equal to it */
            m = p == INT64_MAX ? n : qs_partition(a, n, p + 1);
            a += m; n -= m;
            continue;
        }
        if (m < n - m) { qs_rec(a, m, depth); a += m; n -= m; }
        else { qs_rec(a + m, n - m, depth); n = m; }
    }
    qs_leaf(a, n);
}

/* sort uint64 keys (as tsr_sort's value_sort does) with the AVX2 quicksort */
static int simd_value_sort(uint64_t *keys, int64_t n)
{
    if (n < 2) return TSR_OK;
    for (int64_t i = 0; i < n; i++) keys[i] ^= 0x8000000000000000ULL;        /* unsigned order -> signed order */
    int64_t *a = (int64_t *)keys;
    int64_t i = 1;
    while (i < n && a[i - 1] <= a[i]) i++;
    if (i < n) {
        int64_t j = 1;
        while (i == 1 && j < n && a[j - 1] >= a[j]) j++;
        if (i == 1 && j >= n) {                                 /* non-increasing: reverse */
            for (int64_t l = 0, r = n - 1; l < r; l++, r--) { const int64_t t = a[l]; a[l] = a[r]; a[r] = t; }
        } else {
            int depth = 4;
            for (int64_t t = n; t > 1; t >>= 1) depth += 2;
            qs_rec(a, n, depth);
        }
    }
    for (int64_t k = 0; k < n; k++) keys[k] ^= 0x8000000000000000ULL;
    return TSR_OK;
}
#endif

/*
 * tsr_sort_runs's sorter: a natural (run-adaptive) merge sort of the same keys. Ascending runs are kept and
 * descending runs reversed; runs shorter than RUN_MIN are extended by insertion sort (as Timsort does), then
 * neighbouring runs are merged bottom-up through a scratch buffer, a pair already in order being copied as is.
 * Sorted input costs one scan, data with a few runs costs about one pass per doubling of the run length, so
 * nearly sorted and appended-to data is fast; random data degrades to an n log n merge sort, slower than the
 * other two sorters.
 */
#define RUN_MIN 32
static int runs_value_sort(uint64_t *keys, int64_t n)
{
    if (n < 2) return TSR_OK;
    /* run boundaries: starts[r] .. starts[r + 1] */
    int64_t cap = n / RUN_MIN + 2, nr = 0;
    int64_t *starts = (int64_t *)tsr_alloc(cap * 8);
    if (!starts) return TSR_ENOMEM;
    for (int64_t i = 0; i < n;) {
        int64_t j = i + 1;
        if (j < n && keys[j] < keys[i]) {                      /* strictly descending: extend, then reverse */
            while (j + 1 < n && keys[j + 1] < keys[j]) j++;
            for (int64_t l = i, r = j; l < r; l++, r--) { const uint64_t t = keys[l]; keys[l] = keys[r]; keys[r] = t; }
            j++;
        } else {
            while (j < n && keys[j - 1] <= keys[j]) j++;
        }
        if (j - i < RUN_MIN && j < n) {                         /* short run: extend to RUN_MIN by insertion */
            const int64_t end = i + RUN_MIN < n ? i + RUN_MIN : n;
            for (int64_t k = j; k < end; k++) {
                const uint64_t x = keys[k];
                int64_t p = k - 1;
                while (p >= i && keys[p] > x) { keys[p + 1] = keys[p]; p--; }
                keys[p + 1] = x;
            }
            j = end;
        }
        if (nr + 1 >= cap) {                                     /* cannot happen (runs are >= RUN_MIN but the last) */
            tsr_free(starts, cap * 8);
            return TSR_EARG;
        }
        starts[nr++] = i;
        i = j;
    }
    starts[nr] = n;
    if (nr == 1) { tsr_free(starts, cap * 8); return TSR_OK; }
    uint64_t *buf = (uint64_t *)tsr_alloc(n * 8);
    if (!buf) { tsr_free(starts, cap * 8); return TSR_ENOMEM; }
    uint64_t *src = keys, *dst = buf;
    while (nr > 1) {                                             /* one bottom-up pass merges run pairs */
        int64_t w = 0;
        for (int64_t r = 0; r < nr; r += 2) {
            const int64_t lo = starts[r], mid = starts[r + 1], hi = r + 1 < nr ? starts[r + 2] : mid;
            if (r + 1 >= nr || src[mid - 1] <= src[mid]) {       /* odd run out, or already in order: copy */
                memcpy(dst + lo, src + lo, (size_t)(hi - lo) * 8);
            } else {
                int64_t a = lo, b = mid, o = lo;
                while (a < mid && b < hi) dst[o++] = src[b] < src[a] ? src[b++] : src[a++];
                while (a < mid) dst[o++] = src[a++];
                while (b < hi) dst[o++] = src[b++];
            }
            starts[w++] = lo;
        }
        starts[w] = n;
        nr = w;
        uint64_t *t = src; src = dst; dst = t;
    }
    if (src != keys) memcpy(keys, src, (size_t)n * 8);
    tsr_free(buf, n * 8);
    tsr_free(starts, cap * 8);
    return TSR_OK;
}

enum { SORT_RADIX = 0, SORT_SIMD = 1, SORT_RUNS = 2 };

static int sort_keys(int algo, uint64_t *keys, int64_t n)
{
#ifdef TSR_HAVE_AVX2_SORT
    if (algo == SORT_SIMD) return simd_value_sort(keys, n);
#endif
    if (algo == SORT_RUNS) return runs_value_sort(keys, n);
    return value_sort(keys, n);
}

static int sort_impl(int dtype, int64_t n, void *data, int algo);

int tsr_sort(int dtype, int64_t n, void *data)
{
    return sort_impl(dtype, n, data, SORT_RADIX);
}

int tsr_sort_simd(int dtype, int64_t n, void *data)
{
#ifdef TSR_HAVE_AVX2_SORT
    if (__builtin_cpu_supports("avx2")) return sort_impl(dtype, n, data, SORT_SIMD);
#endif
    return sort_impl(dtype, n, data, SORT_RADIX);                 /* no AVX2: the radix sort, same result */
}

int tsr_sort_runs(int dtype, int64_t n, void *data)
{
    return sort_impl(dtype, n, data, SORT_RUNS);
}

/* the keys, NaN handling and output are shared, so tsr_sort and tsr_sort_simd return the same bits */
static int sort_impl(int dtype, int64_t n, void *data, int algo)
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
        rc = sort_keys(algo, keys, m);
        for (int64_t i = 0; i < m; i++) d[i] = unkey_f64(keys[i]);
        for (int64_t i = m; i < n; i++) d[i] = NAN;
        break;
    }
    case TSR_F32: {
        float *d = (float *)data;
        for (int64_t i = 0; i < n; i++) { if (isnan(d[i])) nans++; else keys[m++] = key_f32(d[i], 0); }
        rc = sort_keys(algo, keys, m);
        for (int64_t i = 0; i < m; i++) d[i] = unkey_f32(keys[i]);
        for (int64_t i = m; i < n; i++) d[i] = NAN;
        break;
    }
    case TSR_I64: {
        int64_t *d = (int64_t *)data;
        for (int64_t i = 0; i < n; i++) keys[i] = (uint64_t)d[i] ^ 0x8000000000000000ULL;
        rc = sort_keys(algo, keys, n);
        for (int64_t i = 0; i < n; i++) d[i] = (int64_t)(keys[i] ^ 0x8000000000000000ULL);
        break;
    }
    case TSR_I32: {
        int32_t *d = (int32_t *)data;
        for (int64_t i = 0; i < n; i++) keys[i] = (uint32_t)d[i] ^ 0x80000000u;
        rc = sort_keys(algo, keys, n);
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
