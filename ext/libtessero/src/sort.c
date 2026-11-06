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
#ifdef _OPENMP
#  include <omp.h>
#endif

/* scratch arrays filled before use: no -ftrivial-auto-var-init zero fill per call (csrc/Makefile HARDEN) */
#if defined(__has_attribute)
#  if __has_attribute(uninitialized)
#    define TSR_NOINIT __attribute__((uninitialized))
#  endif
#endif
#ifndef TSR_NOINIT
#  define TSR_NOINIT
#endif

int tsr_sort_test_depth = 0;                                  /* internal.h: tests only */

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

/* The lsd8 histogram and msd_sort's per-level count tables live in a workspace allocated with the scratch buffer,
   not on the stack: a thread's stack can be small (musl: 128 KB for pthreads and OpenMP workers), and on glibc the
   static TLS of every loaded library is carved out of it (about 66 KB in a process with OpenBLAS), so 16 KB tables
   per recursion level overflowed it (R15 in docs/project/sort-design.md). Each thread has RADIX_TABLES tables of
   RADIX_TAB counts: [0] for lsd8 (8 passes x 256), [1 + L] for MSD level L. Level L sees at most 64 - 11 L key bits
   and only levels above 16 bits run MSD, so L <= 4. */
#define RADIX_TAB 2048
#define RADIX_TABLES 6
#define RADIX_STRIDE (RADIX_TABLES * RADIX_TAB + 72)   /* per thread; the pad keeps the regions off each other's cache sets */

static void lsd8(uint64_t *restrict a, uint64_t *restrict s, int64_t n, int hibit, int64_t (*restrict h)[256])
{
    const int passes = (hibit + 7) / 8;
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
/* ws: this thread's tables (at level 0, the base of every thread's). One table per level holds the counts, then
   the bucket starts, and after the scatter the bucket ends (bucket d is [end[d-1], end[d])). A constant digit
   loops instead of recursing. Only level 0 runs threads; a child on team thread t takes the tables at
   ws + t x RADIX_STRIDE. */
static void msd_sort(uint64_t *restrict a, uint64_t *restrict s, int64_t n, int hibit, int threads, int64_t *restrict ws, int level)
{
    int64_t *restrict const end = ws + (size_t)(1 + level) * RADIX_TAB;   /* a, s and the tables never overlap */
    int lo, nb;
    uint64_t mask;
    for (;;) {
        if (n <= 32) { ins_sort(a, n); return; }
        if (hibit <= 0) return;                                   /* all remaining keys are equal */
        if (n <= 4096 || hibit <= 16 || 1 + level >= RADIX_TABLES) { lsd8(a, s, n, hibit, (int64_t (*)[256])ws); return; }
        lo = hibit > MSD_BITS ? hibit - MSD_BITS : 0;
        nb = 1 << (hibit - lo);
        mask = ((uint64_t)1 << (hibit - lo)) - 1;
        memset(end, 0, sizeof(int64_t) * (size_t)nb);
        for (int64_t i = 0; i < n; i++) end[(a[i] >> lo) & mask]++;
        int constant = 0;
        for (int d = 0; d < nb; d++) if (end[d] == n) { constant = 1; break; }
        if (!constant) break;
        hibit = lo;                                               /* this digit is constant: skip it */
    }
    int64_t sum = 0;
    for (int d = 0; d < nb; d++) { const int64_t c = end[d]; end[d] = sum; sum += c; }
    for (int64_t i = 0; i < n; i++) s[end[(a[i] >> lo) & mask]++] = a[i];
    memcpy(a, s, (size_t)n * 8);
    if (threads < 2 || n < TSR_PAR_MIN) threads = 1;
#if defined(_OPENMP)
#pragma omp parallel for num_threads(threads) schedule(dynamic, 16) if (threads > 1)
#endif
    for (int d = 0; d < nb; d++) {
        const int64_t b = d ? end[d - 1] : 0, c = end[d] - b;
#if defined(_OPENMP)
        int64_t *const mine = ws + (size_t)omp_get_thread_num() * RADIX_STRIDE;
#else
        int64_t *const mine = ws;
#endif
        if (c > 1) msd_sort(a + b, s + b, c, lo, 1, mine, level + 1);
    }
}

/* up to 4096 keys msd_sort would go straight to lsd8 and never recurse: the one table stays on the stack (a single
   16 KB frame, kept out of the callers' frames) without the hardening zero fill, which cost small sorts 30% */
#if defined(__GNUC__)
__attribute__((noinline))
#elif defined(_MSC_VER)
__declspec(noinline)
#endif
static int value_sort_small(uint64_t *keys, int64_t n)
{
    int64_t h[8][256] TSR_NOINIT;
    if (n <= 32) { ins_sort(keys, n); return TSR_OK; }
    uint64_t *s = (uint64_t *)tsr_alloc(n * 8);
    if (!s) return TSR_ENOMEM;
    lsd8(keys, s, n, 64, h);
    tsr_free(s, n * 8);
    return TSR_OK;
}

static int value_sort(uint64_t *keys, int64_t n)
{
    if (n < 2) return TSR_OK;
    if (n <= 4096) return value_sort_small(keys, n);
    const int threads = tsr_get_threads();
    const int64_t team = threads > 1 && n >= TSR_PAR_MIN ? threads : 1;
    const int64_t words = team * RADIX_STRIDE;                    /* the tables first, then the scratch keys */
    int64_t *ws = (int64_t *)tsr_alloc((words + n) * 8);
    if (!ws) return TSR_ENOMEM;
    msd_sort(keys, (uint64_t *)(ws + words), n, 64, threads, ws, 0);
    tsr_free(ws, (words + n) * 8);
    return TSR_OK;
}

/*
 * tsr_sort_simd's sorter: an AVX2 quicksort (sort_qs.h) run in place on the caller's buffer at the data's own
 * width, 8 keys per vector for float32/int32 and 4 for float64/int64, with in-register sorting networks for the
 * leaves (sort_net.h, generated by tools/gen-sort-net.php). No scratch buffer, no allocation.
 *  - int64/int32 are sorted as they are.
 *  - float32 is turned in place into int32 keys with tsr_sort's order (u ^ ((u >> 31) & 0x7fffffff): negatives get
 *    their magnitude bits flipped, so -0.0 sorts before +0.0); the map is its own inverse and is undone after.
 *  - float64 is sorted as doubles (AVX2 compares and min/max doubles in one instruction, int64 in three to five).
 *    Those compares see -0.0 == +0.0, so the negative zeros are counted first and the zero run is rewritten
 *    after the sort, -0.0 before +0.0. Under denormals-are-zero / flush-to-zero (MXCSR) the compares would see
 *    subnormals as zero, so float64 then takes the integer-key route of float32.
 *  - NaNs are dropped in the first pass and written back at the end as tsr_sort writes them.
 * Every step is a bijection of the bit patterns or restores them exactly, so the output is byte-identical to
 * tsr_sort. Faster than the radix sort on random, duplicate-heavy, sorted and nearly sorted data; its time depends
 * on the input and on AVX2, so it is a separate entry point and the radix sort stays the default
 * (docs/project/benchmarks.md).
 */
#if defined(__GNUC__) && defined(__x86_64__)                      /* GCC and clang; MSVC uses the radix sort */
#include <immintrin.h>
#define TSR_HAVE_AVX2_SORT 1
/* the keys overlay the caller's float or integer data */
typedef int64_t qs_i64 __attribute__((__may_alias__));
typedef int32_t qs_i32 __attribute__((__may_alias__));

/* for each 4-bit lane mask (lanes below the pivot), the 32-bit-element permutation that moves those lanes first */
static const int32_t qs_perm64[16][8] = {
    {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {2, 3, 0, 1, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {4, 5, 0, 1, 2, 3, 6, 7}, {0, 1, 4, 5, 2, 3, 6, 7}, {2, 3, 4, 5, 0, 1, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7},
    {6, 7, 0, 1, 2, 3, 4, 5}, {0, 1, 6, 7, 2, 3, 4, 5}, {2, 3, 6, 7, 0, 1, 4, 5}, {0, 1, 2, 3, 6, 7, 4, 5},
    {4, 5, 6, 7, 0, 1, 2, 3}, {0, 1, 4, 5, 6, 7, 2, 3}, {2, 3, 4, 5, 6, 7, 0, 1}, {0, 1, 2, 3, 4, 5, 6, 7},
};
/* the same for 8 lanes, one nibble per output lane (lane j's source is nibble j): set lanes first, then the rest */
static const uint32_t qs_perm32[256] = {
    0x76543210, 0x76543210, 0x76543201, 0x76543210, 0x76543102, 0x76543120, 0x76543021, 0x76543210,
    0x76542103, 0x76542130, 0x76542031, 0x76542310, 0x76541032, 0x76541320, 0x76540321, 0x76543210,
    0x76532104, 0x76532140, 0x76532041, 0x76532410, 0x76531042, 0x76531420, 0x76530421, 0x76534210,
    0x76521043, 0x76521430, 0x76520431, 0x76524310, 0x76510432, 0x76514320, 0x76504321, 0x76543210,
    0x76432105, 0x76432150, 0x76432051, 0x76432510, 0x76431052, 0x76431520, 0x76430521, 0x76435210,
    0x76421053, 0x76421530, 0x76420531, 0x76425310, 0x76410532, 0x76415320, 0x76405321, 0x76453210,
    0x76321054, 0x76321540, 0x76320541, 0x76325410, 0x76310542, 0x76315420, 0x76305421, 0x76354210,
    0x76210543, 0x76215430, 0x76205431, 0x76254310, 0x76105432, 0x76154320, 0x76054321, 0x76543210,
    0x75432106, 0x75432160, 0x75432061, 0x75432610, 0x75431062, 0x75431620, 0x75430621, 0x75436210,
    0x75421063, 0x75421630, 0x75420631, 0x75426310, 0x75410632, 0x75416320, 0x75406321, 0x75463210,
    0x75321064, 0x75321640, 0x75320641, 0x75326410, 0x75310642, 0x75316420, 0x75306421, 0x75364210,
    0x75210643, 0x75216430, 0x75206431, 0x75264310, 0x75106432, 0x75164320, 0x75064321, 0x75643210,
    0x74321065, 0x74321650, 0x74320651, 0x74326510, 0x74310652, 0x74316520, 0x74306521, 0x74365210,
    0x74210653, 0x74216530, 0x74206531, 0x74265310, 0x74106532, 0x74165320, 0x74065321, 0x74653210,
    0x73210654, 0x73216540, 0x73206541, 0x73265410, 0x73106542, 0x73165420, 0x73065421, 0x73654210,
    0x72106543, 0x72165430, 0x72065431, 0x72654310, 0x71065432, 0x71654320, 0x70654321, 0x76543210,
    0x65432107, 0x65432170, 0x65432071, 0x65432710, 0x65431072, 0x65431720, 0x65430721, 0x65437210,
    0x65421073, 0x65421730, 0x65420731, 0x65427310, 0x65410732, 0x65417320, 0x65407321, 0x65473210,
    0x65321074, 0x65321740, 0x65320741, 0x65327410, 0x65310742, 0x65317420, 0x65307421, 0x65374210,
    0x65210743, 0x65217430, 0x65207431, 0x65274310, 0x65107432, 0x65174320, 0x65074321, 0x65743210,
    0x64321075, 0x64321750, 0x64320751, 0x64327510, 0x64310752, 0x64317520, 0x64307521, 0x64375210,
    0x64210753, 0x64217530, 0x64207531, 0x64275310, 0x64107532, 0x64175320, 0x64075321, 0x64753210,
    0x63210754, 0x63217540, 0x63207541, 0x63275410, 0x63107542, 0x63175420, 0x63075421, 0x63754210,
    0x62107543, 0x62175430, 0x62075431, 0x62754310, 0x61075432, 0x61754320, 0x60754321, 0x67543210,
    0x54321076, 0x54321760, 0x54320761, 0x54327610, 0x54310762, 0x54317620, 0x54307621, 0x54376210,
    0x54210763, 0x54217630, 0x54207631, 0x54276310, 0x54107632, 0x54176320, 0x54076321, 0x54763210,
    0x53210764, 0x53217640, 0x53207641, 0x53276410, 0x53107642, 0x53176420, 0x53076421, 0x53764210,
    0x52107643, 0x52176430, 0x52076431, 0x52764310, 0x51076432, 0x51764320, 0x50764321, 0x57643210,
    0x43210765, 0x43217650, 0x43207651, 0x43276510, 0x43107652, 0x43176520, 0x43076521, 0x43765210,
    0x42107653, 0x42176530, 0x42076531, 0x42765310, 0x41076532, 0x41765320, 0x40765321, 0x47653210,
    0x32107654, 0x32176540, 0x32076541, 0x32765410, 0x31076542, 0x31765420, 0x30765421, 0x37654210,
    0x21076543, 0x21765430, 0x20765431, 0x27654310, 0x10765432, 0x17654320, 0x07654321, 0x76543210,
};

#include "sort_net.h"

/* quicksort leaves: insertion sort for a handful of keys, else the smallest network that holds them */
/* parallel sort: tasks down to parts of n / (threads * QS_PAR_SPLIT) keys, at least QS_PAR_MINPART */
#define QS_PAR_SPLIT 16
#define QS_PAR_MINPART ((int64_t)1 << 14)

#define QS_INSERTION(T)                                                                     \
    for (int64_t i = 1; i < n; i++) {                                                       \
        const T k = a[i];                                                                   \
        int64_t j = i - 1;                                                                  \
        while (j >= 0 && a[j] > k) { a[j + 1] = a[j]; j--; }                                \
        a[j + 1] = k;                                                                       \
    }
__attribute__((target("avx2"))) static void qs_leaf64(qs_i64 *a, int64_t n)
{
    if (n <= 4) { QS_INSERTION(qs_i64) }
    else if (n <= 16) qs_net64_4(a, n);
    else if (n <= 32) qs_net64_8(a, n);
    else qs_net64_16(a, n);
}
__attribute__((target("avx2"))) static void qs_leaf32(qs_i32 *a, int64_t n)
{
    if (n <= 8) { QS_INSERTION(qs_i32) }
    else if (n <= 64) qs_net32_8(a, n);
    else qs_net32_16(a, n);
}
__attribute__((target("avx2"))) static void qs_leafd(double *a, int64_t n)
{
    if (n <= 4) { QS_INSERTION(double) }
    else if (n <= 16) qs_netd_4(a, n);
    else if (n <= 32) qs_netd_8(a, n);
    else qs_netd_16(a, n);
}
#undef QS_INSERTION

#define QS_T qs_i64
#define QS_N 64
#define QS_L 4
#define QS_MAX INT64_MAX
#define QS_SET1(q) _mm256_set1_epi64x(q)
#define QS_CMPGT(a, b) _mm256_cmpgt_epi64(a, b)
#define QS_MASK(m) _mm256_movemask_pd(_mm256_castsi256_pd(m))
#define QS_PERM(m) _mm256_loadu_si256((const __m256i *)qs_perm64[m])
#define QS_LEAFMAX 64
#define QS_NEXT(p) ((QS_T)((p) + 1))
#include "sort_qs.h"
#undef QS_T
#undef QS_N
#undef QS_L
#undef QS_MAX
#undef QS_SET1
#undef QS_CMPGT
#undef QS_MASK
#undef QS_PERM
#undef QS_LEAFMAX
#undef QS_NEXT

#define QS_T qs_i32
#define QS_N 32
#define QS_L 8
#define QS_MAX INT32_MAX
#define QS_SET1(q) _mm256_set1_epi32(q)
#define QS_CMPGT(a, b) _mm256_cmpgt_epi32(a, b)
#define QS_MASK(m) _mm256_movemask_ps(_mm256_castsi256_ps(m))
#define QS_PERM(m) _mm256_srlv_epi32(_mm256_set1_epi32((int)qs_perm32[m]), _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28))
#define QS_LEAFMAX 128
#define QS_NEXT(p) ((QS_T)((p) + 1))
#include "sort_qs.h"
#undef QS_T
#undef QS_N
#undef QS_L
#undef QS_MAX
#undef QS_SET1
#undef QS_CMPGT
#undef QS_MASK
#undef QS_PERM
#undef QS_LEAFMAX
#undef QS_NEXT

/* float64 without NaNs, sorted as doubles: one-instruction compares and min/max (AVX2 has no 64-bit integer min/max).
   -0.0 and +0.0 compare equal here; simd_sort_inplace restores their signs. */
#define QS_T double
#define QS_N d
#define QS_L 4
#define QS_MAX INFINITY
#define QS_SET1(q) _mm256_castpd_si256(_mm256_set1_pd(q))
#define QS_CMPGT(a, b) _mm256_castpd_si256(_mm256_cmp_pd(_mm256_castsi256_pd(b), _mm256_castsi256_pd(a), _CMP_LT_OQ))
#define QS_MASK(m) _mm256_movemask_pd(_mm256_castsi256_pd(m))
#define QS_PERM(m) _mm256_loadu_si256((const __m256i *)qs_perm64[m])
#define QS_LEAFMAX 64
#define QS_NEXT(p) nextafter(p, INFINITY)
#include "sort_qs.h"
#undef QS_T
#undef QS_N
#undef QS_L
#undef QS_MAX
#undef QS_SET1
#undef QS_CMPGT
#undef QS_MASK
#undef QS_PERM
#undef QS_LEAFMAX
#undef QS_NEXT

/* float <-> key maps, a vector at a time. to_keys drops NaNs (compacting in place: the store at k + m never passes
   the vector just loaded, since m <= i); a vector holding a NaN is done one key at a time. The map is its own
   inverse: u ^ (sign(u) ? magnitude bits : 0). */
__attribute__((target("avx2"))) static int64_t f64_to_keys(qs_i64 *k, int64_t n)
{
    const __m256i mag = _mm256_set1_epi64x(INT64_MAX), inf = _mm256_set1_epi64x(0x7ff0000000000000LL);
    const __m256i zero = _mm256_setzero_si256();
    int64_t m = 0, i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i u = _mm256_loadu_si256((const __m256i *)(k + i));
        const __m256i nan = _mm256_cmpgt_epi64(_mm256_and_si256(u, mag), inf);
        if (_mm256_testz_si256(nan, nan)) {
            _mm256_storeu_si256((__m256i *)(k + m), _mm256_xor_si256(u, _mm256_and_si256(_mm256_cmpgt_epi64(zero, u), mag)));
            m += 4;
            continue;
        }
        for (int64_t j = i; j < i + 4; j++) {
            const int64_t x = k[j];
            if ((x & INT64_MAX) <= 0x7ff0000000000000LL) k[m++] = x ^ ((x >> 63) & INT64_MAX);
        }
    }
    for (; i < n; i++) {
        const int64_t x = k[i];
        if ((x & INT64_MAX) <= 0x7ff0000000000000LL) k[m++] = x ^ ((x >> 63) & INT64_MAX);
    }
    return m;
}
__attribute__((target("avx2"))) static void f64_from_keys(qs_i64 *k, int64_t m)
{
    const __m256i mag = _mm256_set1_epi64x(INT64_MAX), zero = _mm256_setzero_si256();
    int64_t i = 0;
    for (; i + 4 <= m; i += 4) {
        const __m256i u = _mm256_loadu_si256((const __m256i *)(k + i));
        _mm256_storeu_si256((__m256i *)(k + i), _mm256_xor_si256(u, _mm256_and_si256(_mm256_cmpgt_epi64(zero, u), mag)));
    }
    for (; i < m; i++) k[i] ^= (k[i] >> 63) & INT64_MAX;
}
__attribute__((target("avx2"))) static int64_t f32_to_keys(qs_i32 *k, int64_t n)
{
    const __m256i mag = _mm256_set1_epi32(INT32_MAX), inf = _mm256_set1_epi32(0x7f800000);
    int64_t m = 0, i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i u = _mm256_loadu_si256((const __m256i *)(k + i));
        const __m256i nan = _mm256_cmpgt_epi32(_mm256_and_si256(u, mag), inf);
        if (_mm256_testz_si256(nan, nan)) {
            _mm256_storeu_si256((__m256i *)(k + m), _mm256_xor_si256(u, _mm256_and_si256(_mm256_srai_epi32(u, 31), mag)));
            m += 8;
            continue;
        }
        for (int64_t j = i; j < i + 8; j++) {
            const int32_t x = k[j];
            if ((x & INT32_MAX) <= 0x7f800000) k[m++] = x ^ ((x >> 31) & INT32_MAX);
        }
    }
    for (; i < n; i++) {
        const int32_t x = k[i];
        if ((x & INT32_MAX) <= 0x7f800000) k[m++] = x ^ ((x >> 31) & INT32_MAX);
    }
    return m;
}
__attribute__((target("avx2"))) static void f32_from_keys(qs_i32 *k, int64_t m)
{
    const __m256i mag = _mm256_set1_epi32(INT32_MAX);
    int64_t i = 0;
    for (; i + 8 <= m; i += 8) {
        const __m256i u = _mm256_loadu_si256((const __m256i *)(k + i));
        _mm256_storeu_si256((__m256i *)(k + i), _mm256_xor_si256(u, _mm256_and_si256(_mm256_srai_epi32(u, 31), mag)));
    }
    for (; i < m; i++) k[i] ^= (k[i] >> 31) & INT32_MAX;
}

/* drop the NaNs of d in place (compacting as f64_to_keys does) and count the negative zeros */
__attribute__((target("avx2"))) static int64_t f64_drop_nans(double *d, int64_t n, int64_t *nneg)
{
    const qs_i64 *k = (const qs_i64 *)d;
    const __m256i mag = _mm256_set1_epi64x(INT64_MAX), inf = _mm256_set1_epi64x(0x7ff0000000000000LL);
    const __m256i nz = _mm256_set1_epi64x(INT64_MIN);
    int64_t m = 0, i = 0, neg = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i u = _mm256_loadu_si256((const __m256i *)(k + i));
        const __m256i nan = _mm256_cmpgt_epi64(_mm256_and_si256(u, mag), inf);
        if (_mm256_testz_si256(nan, nan)) {
            neg += __builtin_popcount(_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(u, nz))));
            _mm256_storeu_si256((__m256i *)(d + m), u);
            m += 4;
            continue;
        }
        for (int64_t j = i; j < i + 4; j++) {
            const int64_t x = k[j];
            if ((x & INT64_MAX) > 0x7ff0000000000000LL) continue;
            neg += x == INT64_MIN;
            memcpy(d + m++, &x, 8);
        }
    }
    for (; i < n; i++) {
        const int64_t x = k[i];
        if ((x & INT64_MAX) > 0x7ff0000000000000LL) continue;
        neg += x == INT64_MIN;
        memcpy(d + m++, &x, 8);
    }
    *nneg = neg;
    return m;
}

__attribute__((target("avx2"))) static int simd_sort_inplace(int dtype, int64_t n, void *data)
{
    const int threads = tsr_get_threads();
    switch (dtype) {
    case TSR_I64: qs_sort64((qs_i64 *)data, n, threads); return TSR_OK;
    case TSR_I32: qs_sort32((qs_i32 *)data, n, threads); return TSR_OK;
    case TSR_F64: {
        double *d = (double *)data;
        int64_t m;
        if (_mm_getcsr() & 0x8040) {
            /* denormals-are-zero (or flush-to-zero) mode: the double compares would see subnormals as zero, so sort
               the integer keys instead */
            qs_i64 *k = (qs_i64 *)data;
            m = f64_to_keys(k, n);
            qs_sort64(k, m, threads);
            f64_from_keys(k, m);
        } else {
            int64_t nneg = 0;
            m = f64_drop_nans(d, n, &nneg);
            qs_sortd(d, m, threads);
            /* the zeros are one run [lo, hi), signs mixed by min/max: rewrite it as nneg times -0.0, then +0.0 */
            int64_t lo = 0, hi = m;
            while (lo < hi) { const int64_t mid = lo + (hi - lo) / 2; if (d[mid] < 0.0) lo = mid + 1; else hi = mid; }
            hi = m;
            for (int64_t l = lo; l < hi;) { const int64_t mid = l + (hi - l) / 2; if (d[mid] > 0.0) hi = mid; else l = mid + 1; }
            for (int64_t i = lo; i < hi; i++) d[i] = i - lo < nneg ? -0.0 : 0.0;
        }
        for (int64_t i = m; i < n; i++) d[i] = NAN;
        return TSR_OK;
    }
    case TSR_F32: {
        qs_i32 *k = (qs_i32 *)data;
        const int64_t m = f32_to_keys(k, n);
        qs_sort32(k, m, threads);
        f32_from_keys(k, m);
        float *d = (float *)data;
        for (int64_t i = m; i < n; i++) d[i] = NAN;
        return TSR_OK;
    }
    default:
        return TSR_ETYPE;                                                          /* not reached: see tsr_sort_simd */
    }
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

enum { SORT_RADIX = 0, SORT_RUNS = 1 };

static int sort_keys(int algo, uint64_t *keys, int64_t n)
{
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
    if (n >= 2 && (dtype == TSR_F64 || dtype == TSR_F32 || dtype == TSR_I64 || dtype == TSR_I32) && __builtin_cpu_supports("avx2"))
        return simd_sort_inplace(dtype, n, data);
#endif
    return sort_impl(dtype, n, data, SORT_RADIX);   /* no AVX2, or uint8/bool (counting sort): the same result */
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
