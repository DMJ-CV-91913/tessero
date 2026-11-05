/*
 * AVX2 quicksort of signed integer keys, included by sort.c once per key width with
 *   QS_T      key type (a may_alias int64 or int32 typedef: the keys live in the caller's float or int buffer)
 *   QS_N      name suffix (64 / 32)
 *   QS_L      keys per 256-bit vector (4 / 8)
 *   QS_MAX    largest key
 *   QS_SET1, QS_CMPGT, QS_MASK, QS_PERM   the width's broadcast, compare, lane mask and left-pack permutation
 * Partition: four vectors (4 * QS_L keys) per step from the end with less free space, the lanes below the pivot
 * moved left by a permutation and stored from both ends; 4 * QS_L keys are held back at each end so a full store
 * always fits, and the tails are branchless scalar loops. Pivot: median of nine. Keys equal to a minimum pivot are
 * split off in one more pass (many duplicates stay linear); a depth limit falls back to heapsort. Leaves of up to
 * 16 keys: Batcher's network. Comparison only, so the result is the sorted keys whatever the path taken.
 */
#define QS_CAT_(a, b) a##b
#define QS_CAT(a, b) QS_CAT_(a, b)
#define QS_FN(name) QS_CAT(name, QS_N)

static void QS_FN(qs_sift)(QS_T *a, int64_t n, int64_t i)
{
    for (;;) {
        int64_t c = 2 * i + 1;
        if (c >= n) return;
        if (c + 1 < n && a[c + 1] > a[c]) c++;
        if (a[i] >= a[c]) return;
        const QS_T t = a[i]; a[i] = a[c]; a[c] = t; i = c;
    }
}
static void QS_FN(qs_heapsort)(QS_T *a, int64_t n)
{
    for (int64_t i = n / 2 - 1; i >= 0; i--) QS_FN(qs_sift)(a, n, i);
    for (int64_t e = n - 1; e > 0; e--) { const QS_T t = a[0]; a[0] = a[e]; a[e] = t; QS_FN(qs_sift)(a, e, 0); }
}

__attribute__((target("avx2"))) static inline void QS_FN(qs_store)(__m256i v, __m256i pv, QS_T *a, int64_t *lw, int64_t *rw)
{
    const int m = QS_MASK(QS_CMPGT(pv, v));
    const __m256i w = _mm256_permutevar8x32_epi32(v, QS_PERM(m));
    _mm256_storeu_si256((__m256i *)(a + *lw), w);
    _mm256_storeu_si256((__m256i *)(a + *rw - QS_L), w);
    *lw += __builtin_popcount(m);
    *rw -= QS_L - __builtin_popcount(m);
}

/* keys < q to the left, in place; returns how many */
__attribute__((target("avx2"))) static int64_t QS_FN(qs_partition)(QS_T *a, int64_t n, QS_T q)
{
    if (n < 16 * QS_L) {
        TSR_NOINIT QS_T buf[16 * QS_L];
        int64_t lw = 0, rw = n;
        memcpy(buf, a, (size_t)n * sizeof(QS_T));
        for (int64_t i = 0; i < n; i++) { const QS_T x = buf[i]; const int64_t lt = x < q; a[lw] = x; a[rw - 1] = x; lw += lt; rw -= 1 - lt; }
        return lw;
    }
    const __m256i pv = QS_SET1(q);
    TSR_NOINIT QS_T hold[8 * QS_L];
    memcpy(hold, a, 4 * QS_L * sizeof(QS_T));
    memcpy(hold + 4 * QS_L, a + n - 4 * QS_L, 4 * QS_L * sizeof(QS_T));
    int64_t lr = 4 * QS_L, rr = n - 4 * QS_L, lw = 0, rw = n;
    while (rr - lr >= 4 * QS_L) {
        const QS_T *src;
        if (lr - lw <= rw - rr) { src = a + lr; lr += 4 * QS_L; } else { rr -= 4 * QS_L; src = a + rr; }
        const __m256i v0 = _mm256_loadu_si256((const __m256i *)src), v1 = _mm256_loadu_si256((const __m256i *)(src + QS_L));
        const __m256i v2 = _mm256_loadu_si256((const __m256i *)(src + 2 * QS_L)), v3 = _mm256_loadu_si256((const __m256i *)(src + 3 * QS_L));
        QS_FN(qs_store)(v0, pv, a, &lw, &rw); QS_FN(qs_store)(v1, pv, a, &lw, &rw);
        QS_FN(qs_store)(v2, pv, a, &lw, &rw); QS_FN(qs_store)(v3, pv, a, &lw, &rw);
    }
    while (rr - lr >= QS_L) {                                  /* free space >= 8 vectors + read: single vectors fit */
        __m256i v;
        if (lr - lw <= rw - rr) { v = _mm256_loadu_si256((const __m256i *)(a + lr)); lr += QS_L; }
        else { rr -= QS_L; v = _mm256_loadu_si256((const __m256i *)(a + rr)); }
        QS_FN(qs_store)(v, pv, a, &lw, &rw);
    }
    TSR_NOINIT QS_T buf[9 * QS_L];
    int nb = 0;
    for (int64_t i = lr; i < rr; i++) buf[nb++] = a[i];
    for (int i = 0; i < 8 * QS_L; i++) buf[nb++] = hold[i];
    for (int i = 0; i < nb; i++) { const QS_T x = buf[i]; const int64_t lt = x < q; a[lw] = x; a[rw - 1] = x; lw += lt; rw -= 1 - lt; }
    return lw;
}

static inline QS_T QS_FN(qs_med3)(QS_T x, QS_T y, QS_T z)
{
    return x < y ? (y < z ? y : (x < z ? z : x)) : (x < z ? x : (y < z ? z : y));
}

#define QS_CAS(x, y) do { const QS_T a_ = v[x], b_ = v[y]; v[x] = a_ < b_ ? a_ : b_; v[y] = a_ < b_ ? b_ : a_; } while (0)
/* up to 16 keys: Batcher's odd-even merge network (63 compare-exchanges, branchless), padded with QS_MAX */
static void QS_FN(qs_leaf)(QS_T *a, int64_t n)
{
    if (n < 2) return;
    if (n <= 6) {
        for (int64_t i = 1; i < n; i++) {
            const QS_T k = a[i];
            int64_t j = i - 1;
            while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }
            a[j + 1] = k;
        }
        return;
    }
    TSR_NOINIT QS_T v[16];
    for (int64_t i = 0; i < n; i++) v[i] = a[i];
    for (int64_t i = n; i < 16; i++) v[i] = QS_MAX;
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

static void QS_FN(qs_rec)(QS_T *a, int64_t n, int depth)
{
    while (n > 16) {
        if (depth-- <= 0) { QS_FN(qs_heapsort)(a, n); return; }
        const QS_T p = QS_FN(qs_med3)(QS_FN(qs_med3)(a[0], a[n / 8], a[n / 4]), QS_FN(qs_med3)(a[3 * n / 8], a[n / 2], a[5 * n / 8]),
                                      QS_FN(qs_med3)(a[3 * n / 4], a[7 * n / 8], a[n - 1]));
        int64_t m = QS_FN(qs_partition)(a, n, p);
        if (m == 0) {                                          /* p is the minimum: split off every key equal to it */
            m = p == QS_MAX ? n : QS_FN(qs_partition)(a, n, (QS_T)(p + 1));
            a += m; n -= m;
            continue;
        }
        if (m < n - m) { QS_FN(qs_rec)(a, m, depth); a += m; n -= m; }
        else { QS_FN(qs_rec)(a + m, n - m, depth); n = m; }
    }
    QS_FN(qs_leaf)(a, n);
}

/* ascending input returns at once and non-increasing input is reversed (equal keys are interchangeable) */
static void QS_FN(qs_sort)(QS_T *a, int64_t n)
{
    if (n < 2) return;
    int64_t i = 1;
    while (i < n && a[i - 1] <= a[i]) i++;
    if (i >= n) return;
    if (i == 1) {
        int64_t j = 1;
        while (j < n && a[j - 1] >= a[j]) j++;
        if (j >= n) {
            for (int64_t l = 0, r = n - 1; l < r; l++, r--) { const QS_T t = a[l]; a[l] = a[r]; a[r] = t; }
            return;
        }
    }
    int depth = 4;
    for (int64_t t = n; t > 1; t >>= 1) depth += 2;
    QS_FN(qs_rec)(a, n, depth);
}

#undef QS_FN
#undef QS_CAT
#undef QS_CAT_
