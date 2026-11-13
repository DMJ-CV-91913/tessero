/* libtessero C unit tests. Run: make test && make sanitize */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../src/internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#if !defined(_WIN32)
#include <pthread.h>
#endif
#include "../src/ops.h"
#include "../src/vmath.h"

static int failures = 0, checks = 0;
#define CHECK(cond, ...)                                                   \
    do {                                                                   \
        checks++;                                                          \
        if (!(cond)) {                                                     \
            failures++;                                                    \
            fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);           \
            fprintf(stderr, __VA_ARGS__);                                  \
            fprintf(stderr, "\n");                                         \
        }                                                                  \
    } while (0)
#define CLOSE(a, b, tol) (fabs((a) - (b)) <= (tol) * (1.0 + fabs(b)))

/* the platform's temporary directory: TEMP/TMP on Windows (there is no /tmp), TMPDIR or /tmp elsewhere */
static const char *test_tmpdir(void)
{
#ifdef _WIN32
    const char *d = getenv("TEMP");
    if (!d || !*d) d = getenv("TMP");
    if (!d || !*d) d = ".";
#else
    const char *d = getenv("TMPDIR");
    if (!d || !*d) d = "/tmp";
#endif
    return d;
}

static void test_alloc(void)
{
    int64_t before = tsr_allocated();
    void *p = tsr_alloc(1000);
    CHECK(p && ((uintptr_t)p % 64) == 0, "64-byte alignment");
    CHECK(tsr_allocated() == before + 1000, "accounting");
    tsr_free(p, 1000);
    CHECK(tsr_allocated() == before, "free accounting");
    tsr_set_budget(before + 4096);
    CHECK(tsr_alloc(1 << 20) == NULL, "budget refuses");
    void *q = tsr_alloc(1024);
    CHECK(q != NULL, "budget allows small");
    tsr_free(q, 1024);
    tsr_set_budget(0);
    double *z = (double *)tsr_calloc(80);
    CHECK(z && z[9] == 0.0, "calloc zero");
    tsr_free(z, 80);

    /* large-block reuse: a freed block of 1 MiB or more is kept and handed out again (not in ownership/ASan builds) */
    {
        const int64_t saved = tsr_alloc_cache_limit(), big = (int64_t)3 << 20;
        tsr_set_alloc_cache_limit((int64_t)16 << 20);
        CHECK(tsr_alloc_cache_limit() == (int64_t)16 << 20, "alloc cache limit set");
        const int64_t base = tsr_alloc_cache_bytes(), was = tsr_allocated();
        char *b1 = (char *)tsr_alloc(big);
        CHECK(b1 && ((uintptr_t)b1 % 64) == 0 && tsr_allocated() == was + big, "big alloc accounting");
        tsr_free(b1, big);
        CHECK(tsr_allocated() == was, "kept block not counted as allocated");
        if (!TSR_REUSE) {
            CHECK(tsr_alloc_cache_bytes() == 0, "no reuse in the ownership-check and ASan builds");
        } else {
            CHECK(tsr_alloc_cache_bytes() == base + big, "freed block kept (%lld)", (long long)tsr_alloc_cache_bytes());
            char *b2 = (char *)tsr_alloc(big - 4096);              /* within a quarter of its size: the same block */
            CHECK(b2 == b1 && tsr_alloc_cache_bytes() == base, "kept block reused");
            if (b2) memset(b2, 0x5a, (size_t)(big - 4096));             /* old data for the calloc check below */
            tsr_free(b2, big - 4096);
            char *b3 = (char *)tsr_alloc(big / 2);                 /* far smaller: a new block, the kept one stays */
            CHECK(b3 && b3 != b1, "no reuse for a much smaller request");
            tsr_free(b3, big / 2);
            double *zc = (double *)tsr_calloc(big - 4096);         /* the dirty block again, zeroed by tsr_calloc */
            CHECK(zc == (double *)b1 && zc[0] == 0.0 && zc[(big - 4096) / 8 - 1] == 0.0, "calloc zeroes a reused block");
            tsr_free(zc, big - 4096);
            tsr_set_alloc_cache_limit(big);                        /* lowering the limit releases the oldest */
            CHECK(tsr_alloc_cache_bytes() <= big, "lower limit trims (%lld)", (long long)tsr_alloc_cache_bytes());
        }
        tsr_set_alloc_cache_limit(0);
        CHECK(tsr_alloc_cache_bytes() == 0, "limit 0 releases every kept block");
        char *b4 = (char *)tsr_alloc(big);
        tsr_free(b4, big);
        CHECK(tsr_alloc_cache_bytes() == 0, "nothing kept at limit 0");
        tsr_set_alloc_cache_limit(-5);
        CHECK(tsr_alloc_cache_limit() == 0, "negative limit clamps to 0");
        tsr_set_alloc_cache_limit(saved);
        CHECK(tsr_alloc_cache_limit() == saved, "alloc cache limit restored");
    }
    CHECK(tsr_simd_level() >= 0, "simd level");
    CHECK(strlen(tsr_version()) > 0, "version");
}

static void test_binary(void)
{
    /* contiguous 2x3 + broadcast row (stride 0 on axis 0) */
    double a[6] = {1, 2, 3, 4, 5, 6}, b[3] = {10, 20, 30}, o[6];
    int64_t shape[2] = {2, 3}, sa[2] = {24, 8}, sb[2] = {0, 8}, so[2] = {24, 8};
    CHECK(tsr_binary(OP_ADD, TSR_F64, 2, shape, a, sa, b, sb, o, so) == 0, "add rc");
    CHECK(o[0] == 11 && o[5] == 36, "broadcast add %g %g", o[0], o[5]);

    /* transposed view: a^T (3x2) * 2 via scalar stride 0 */
    double two = 2.0, t[6];
    int64_t tshape[2] = {3, 2}, ta[2] = {8, 24}, ts[2] = {0, 0}, to[2] = {16, 8};
    tsr_binary(OP_MUL, TSR_F64, 2, tshape, a, ta, &two, ts, t, to);
    CHECK(t[0] == 2 && t[1] == 8 && t[5] == 12, "transposed mul");

    /* reversed stride */
    double r[6];
    int64_t n6[1] = {6}, rev[1] = {-8}, c8[1] = {8};
    tsr_binary(OP_SUB, TSR_F64, 1, n6, a + 5, rev, a, c8, r, c8);
    CHECK(r[0] == 5 && r[5] == -5, "negative stride");

    /* python-style modulo and floordiv */
    double x[2] = {-7, 7}, y[2] = {2, -2}, m[2], fd[2];
    int64_t n2[1] = {2};
    tsr_binary(OP_MOD, TSR_F64, 1, n2, x, c8, y, c8, m, c8);
    tsr_binary(OP_FLOORDIV, TSR_F64, 1, n2, x, c8, y, c8, fd, c8);
    CHECK(m[0] == 1 && m[1] == -1 && fd[0] == -4 && fd[1] == -4, "mod/floordiv %g %g %g %g", m[0], m[1], fd[0], fd[1]);
    int64_t xi[2] = {-7, 7}, yi[2] = {2, -2}, mi[2];
    tsr_binary(OP_MOD, TSR_I64, 1, n2, xi, c8, yi, c8, mi, c8);
    CHECK(mi[0] == 1 && mi[1] == -1, "int mod");
    CHECK(tsr_binary(OP_DIV, TSR_I64, 1, n2, xi, c8, yi, c8, mi, c8) == TSR_ETYPE, "int true-div refused");

    /* NaN-propagating max, comparisons to bool */
    double nanv[2] = {NAN, 1}, one[2] = {0, 2}, mx[2];
    tsr_binary(OP_MAX, TSR_F64, 1, n2, nanv, c8, one, c8, mx, c8);
    CHECK(isnan(mx[0]) && mx[1] == 2, "maximum propagates NaN");
    uint8_t cmp[6];
    int64_t s1[2] = {3, 1};
    tsr_binary(OP_GT, TSR_F64, 2, shape, a, sa, b, sb, cmp, s1);
    CHECK(cmp[0] == 0 && cmp[5] == 0, "gt");

    /* complex multiply */
    double c1[2] = {1, 2}, c2[2] = {3, -1}, cc[2];
    int64_t n1[1] = {1}, s16[1] = {16};
    tsr_binary(OP_MUL, TSR_C128, 1, n1, c1, s16, c2, s16, cc, s16);
    CHECK(cc[0] == 5 && cc[1] == 5, "complex mul");

    /* large contiguous add exercises the SIMD clones */
    int64_t N = 100003;
    double *A = malloc(N * 8), *B = malloc(N * 8), *C = malloc(N * 8);
    for (int64_t i = 0; i < N; i++) { A[i] = i; B[i] = 2.0 * i; }
    int64_t nN[1] = {N};
    tsr_binary(OP_ADD, TSR_F64, 1, nN, A, c8, B, c8, C, c8);
    CHECK(C[N - 1] == 3.0 * (N - 1), "large add");
    free(A); free(B); free(C);

    /* clip: one pass with the bits of maximum then minimum (NaN, -0 and +0, inf, a NaN bound, strided, int32) */
    double cx[8] = {-1.0, 0.5, NAN, -0.0, 0.0, 3.0, 2.0, -INFINITY}, p2[8], p1[8];
    double bounds[3][2] = {{0.0, 2.0}, {-0.0, NAN}, {NAN, 1.0}};
    int64_t n8[1] = {8}, n4[1] = {4}, z0[1] = {0}, s16c[1] = {16};
    for (int k = 0; k < 3; k++) {
        tsr_binary(OP_MAX, TSR_F64, 1, n8, cx, c8, &bounds[k][0], z0, p2, c8);
        tsr_binary(OP_MIN, TSR_F64, 1, n8, p2, c8, &bounds[k][1], z0, p2, c8);
        memset(p1, 0x5a, sizeof p1);
        CHECK(tsr_clip(TSR_F64, 1, n8, cx, c8, &bounds[k][0], &bounds[k][1], p1, c8) == 0 &&
              memcmp(p1, p2, sizeof p1) == 0, "clip f64 %d", k);
    }
    tsr_binary(OP_MAX, TSR_F64, 1, n4, cx, s16c, &bounds[0][0], z0, p2, c8);
    tsr_binary(OP_MIN, TSR_F64, 1, n4, p2, c8, &bounds[0][1], z0, p2, c8);
    CHECK(tsr_clip(TSR_F64, 1, n4, cx, s16c, &bounds[0][0], &bounds[0][1], p1, c8) == 0 &&
          memcmp(p1, p2, 4 * sizeof(double)) == 0, "clip strided");
    int32_t ix[4] = {-5, 3, 10, 7}, ilo = 0, ihi = 7, io[4];
    int64_t c4[1] = {4};
    CHECK(tsr_clip(TSR_I32, 1, n4, ix, c4, &ilo, &ihi, io, c4) == 0 && io[0] == 0 && io[1] == 3 && io[2] == 7 && io[3] == 7,
          "clip int32");
    uint8_t bx[4] = {0, 1, 1, 0}, blo = 0, bhi = 1, bo[4];
    int64_t sb1[1] = {1};
    CHECK(tsr_clip(TSR_BOOL, 1, n4, bx, sb1, &blo, &bhi, bo, sb1) == TSR_ETYPE, "clip bool refused");

    /* too many dims */
    int64_t big[33]; for (int i = 0; i < 33; i++) big[i] = 1;
    CHECK(tsr_binary(OP_ADD, TSR_F64, 33, big, a, big, a, big, o, big) == TSR_EDIM, "dim cap");
}

static void test_unary_cast(void)
{
    double a[4] = {-1.5, 0, 2.25, NAN}, o[4];
    int64_t n4[1] = {4}, c8[1] = {8}, c1[1] = {1};
    tsr_unary(U_ABS, TSR_F64, 1, n4, a, c8, o, c8);
    CHECK(o[0] == 1.5 && o[2] == 2.25, "abs");
    uint8_t m[4];
    tsr_unary(U_ISNAN, TSR_F64, 1, n4, a, c8, m, c1);
    CHECK(m[3] == 1 && m[0] == 0, "isnan");
    int32_t i32[4];
    int64_t c4[1] = {4};
    double b[4] = {-1.7, 0.2, 2.9, 7};
    tsr_copy(TSR_F64, TSR_I32, 1, n4, b, c8, i32, c4);
    CHECK(i32[0] == -1 && i32[2] == 2 && i32[3] == 7, "cast truncates toward zero");
    double cplx[4] = {3, 4, 0, -1}, mag[2];
    int64_t n2[1] = {2}, s16[1] = {16};
    tsr_unary(U_CABS, TSR_C128, 1, n2, cplx, s16, mag, c8);
    CHECK(mag[0] == 5 && mag[1] == 1, "cabs");
    double f[3];
    tsr_arange(TSR_F64, 3, 1.0, 0.5, f);
    CHECK(f[2] == 2.0, "arange");
    double v = 7.0;
    tsr_fill(TSR_F64, 3, f, &v);
    CHECK(f[0] == 7 && f[2] == 7, "fill");
}

static void test_reduce(void)
{
    /* pairwise summation: 1e7 x 0.1 */
    int64_t N = 10000000;
    double *x = malloc(N * 8);
    for (int64_t i = 0; i < N; i++) x[i] = 0.1;
    int64_t shape[1] = {N}, c8[1] = {8};
    double s = 0;
    int64_t so[1] = {0};
    tsr_reduce(R_SUM, TSR_F64, 1, shape, x, c8, 0, TSR_F64, &s, so);
    CHECK(fabs(s - 1e6) < 1e-6, "pairwise sum error %.3e", s - 1e6);
    free(x);

    double a[6] = {3, 1, 4, 1, 5, 9};
    int64_t sh[2] = {2, 3}, sa[2] = {24, 8};
    double rows[2], cols[3];
    int64_t s1[1] = {8};
    tsr_reduce(R_SUM, TSR_F64, 2, sh, a, sa, 1, TSR_F64, rows, s1);
    tsr_reduce(R_MAX, TSR_F64, 2, sh, a, sa, 0, TSR_F64, cols, s1);
    CHECK(rows[0] == 8 && rows[1] == 15, "sum axis 1");
    CHECK(cols[0] == 3 && cols[2] == 9, "max axis 0");
    int64_t am[2];
    tsr_reduce(R_ARGMIN, TSR_F64, 2, sh, a, sa, 1, TSR_I64, am, s1);
    CHECK(am[0] == 1 && am[1] == 0, "argmin first occurrence");

    double mean[1], m2[1];
    int64_t n6[1] = {6}, z[1] = {0};
    tsr_moments(TSR_F64, 1, n6, a, c8, 0, mean, z, m2, z);
    CHECK(CLOSE(mean[0], 23.0 / 6.0, 1e-15), "mean");
    double var = 0; for (int i = 0; i < 6; i++) var += (a[i] - 23.0 / 6.0) * (a[i] - 23.0 / 6.0);
    CHECK(CLOSE(m2[0], var, 1e-13), "welford m2 %g %g", m2[0], var);

    double mid[3];
    tsr_reduce_mid(R_SUM, TSR_F64, 1, 2, 3, a, mid);
    CHECK(mid[0] == 4 && mid[1] == 6 && mid[2] == 13, "reduce_mid sum axis 0");
    double nanrow[4] = {1, NAN, 3, 0};
    tsr_reduce_mid(R_MAX, TSR_F64, 1, 2, 2, nanrow, mid);
    CHECK(mid[0] == 3 && isnan(mid[1]), "reduce_mid max propagates NaN");
    double tsrc[6] = {1, 2, 3, 4, 5, 6}, tdst[6];
    int64_t tshape2[2] = {3, 2}, tin[2] = {8, 24}, tout[2] = {16, 8};
    tsr_copy(TSR_F64, TSR_F64, 2, tshape2, tsrc, tin, tdst, tout);
    CHECK(tdst[0] == 1 && tdst[1] == 4 && tdst[2] == 2 && tdst[5] == 6, "tiled transpose copy");
    double cs[6];
    tsr_cumulative(0, TSR_F64, 2, 3, 1, a, cs);
    CHECK(cs[2] == 8 && cs[3] == 1 && cs[5] == 15, "cumsum rows");
    int32_t ci[4] = {1, 2, 3, 4};
    int64_t cp[4];
    tsr_cumulative(1, TSR_I32, 1, 2, 2, ci, cp);
    CHECK(cp[2] == 3 && cp[3] == 8, "cumprod axis0 -> int64");
    double w[5] = {3, NAN, 1, 2, 1};
    int64_t idx[5];
    tsr_argsort(TSR_F64, 5, w, idx);
    CHECK(idx[0] == 2 && idx[1] == 4 && idx[4] == 1, "stable argsort, NaN last");
    tsr_sort(TSR_F64, 5, w);
    CHECK(w[0] == 1 && w[3] == 3 && isnan(w[4]), "sort NaN last");
}

static void test_index(void)
{
    double a[6] = {0, 1, 2, 3, 4, 5}, o[4];
    int64_t idx[2] = {2, -1};
    /* a as 2x3, take columns {2, -1} along axis 1: outer=2, axis_len=3, inner=1 */
    CHECK(tsr_take(8, 2, 3, 1, a, idx, 2, o) == 0, "take rc");
    CHECK(o[0] == 2 && o[1] == 2 && o[2] == 5 && o[3] == 5, "take values");
    int64_t bad[1] = {3};
    CHECK(tsr_take(8, 2, 3, 1, a, bad, 1, o) == TSR_EINDEX, "take bounds");
    uint8_t mask[6] = {1, 0, 1, 0, 0, 1};
    CHECK(tsr_count_true(6, mask) == 3, "count");
    double c[3];
    tsr_compress(8, 6, a, mask, c);
    CHECK(c[2] == 5, "compress");
    double v = -1;
    tsr_mask_assign(8, 6, a, mask, &v, 1);
    CHECK(a[0] == -1 && a[1] == 1 && a[5] == -1, "mask assign");
}

static void test_matmul(void)
{
    int64_t m = 67, n = 45, k = 131;
    double *A = malloc(m * k * 8), *B = malloc(k * n * 8), *C = malloc(m * n * 8);
    for (int64_t i = 0; i < m * k; i++) A[i] = sin((double)i);
    for (int64_t i = 0; i < k * n; i++) B[i] = cos((double)i);
    tsr_matmul(TSR_F64, m, n, k, A, k, B, n, C, n);
    double err = 0;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            double ref = 0;
            for (int64_t p = 0; p < k; p++) ref += A[i * k + p] * B[p * n + j];
            err = fmax(err, fabs(ref - C[i * n + j]));
        }
    CHECK(err < 1e-12, "matmul err %.3e", err);
    CHECK(fabs(tsr_dot_f64(k, A, 1, A, 1) - C[0] * 0 - tsr_dot_f64(k, A, 1, A, 1)) == 0, "dot");
    free(A); free(B); free(C);
}

/* reference row-major GEMM: C = alpha*op(A)*op(B) + beta*C, op by transpose flag */
static double gemm_ref(int ta, int tb, int64_t m, int64_t n, int64_t k, double alpha,
                       const double *A, int64_t lda, const double *B, int64_t ldb,
                       double beta, const double *C, int64_t i, int64_t j)
{
    double acc = 0;
    for (int64_t p = 0; p < k; p++) {
        double aip = ta ? A[p * lda + i] : A[i * lda + p];
        double bpj = tb ? B[j * ldb + p] : B[p * ldb + j];
        acc += aip * bpj;
    }
    return alpha * acc + beta * (C ? C[i * n + j] : 0);
}

static void test_gemm(void)
{
    int64_t m = 17, n = 23, k = 29;
    double *A = malloc(m * k * 8), *AT = malloc(k * m * 8), *B = malloc(k * n * 8), *BT = malloc(n * k * 8), *C = malloc(m * n * 8);
    for (int64_t i = 0; i < m * k; i++) A[i] = sin(0.3 * i + 1);
    for (int64_t i = 0; i < k * n; i++) B[i] = cos(0.2 * i + 2);
    for (int64_t i = 0; i < m; i++) for (int64_t p = 0; p < k; p++) AT[p * m + i] = A[i * k + p]; /* k x m */
    for (int64_t p = 0; p < k; p++) for (int64_t j = 0; j < n; j++) BT[j * k + p] = B[p * n + j]; /* n x k */

    /* NN, alpha=1, beta=0 (plain forward) */
    tsr_gemm(TSR_F64, 0, 0, m, n, k, 1.0, A, k, B, n, 0.0, C, n);
    double err = 0;
    for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++)
        err = fmax(err, fabs(C[i * n + j] - gemm_ref(0, 0, m, n, k, 1.0, A, k, B, n, 0.0, NULL, i, j)));
    CHECK(err < 1e-12, "gemm NN err %.3e", err);

    /* op(A^T) via transa: AT is k x m, op(AT)=A; must equal the NN result */
    double *C2 = malloc(m * n * 8);
    tsr_gemm(TSR_F64, 1, 0, m, n, k, 1.0, AT, m, B, n, 0.0, C2, n);
    err = 0; for (int64_t i = 0; i < m * n; i++) err = fmax(err, fabs(C2[i] - C[i]));
    CHECK(err < 1e-12, "gemm TN matches NN err %.3e", err);

    /* op(B^T) via transb: BT is n x k, op(BT)=B; must equal the NN result */
    tsr_gemm(TSR_F64, 0, 1, m, n, k, 1.0, A, k, BT, k, 0.0, C2, n);
    err = 0; for (int64_t i = 0; i < m * n; i++) err = fmax(err, fabs(C2[i] - C[i]));
    CHECK(err < 1e-12, "gemm NT matches NN err %.3e", err);

    /* NT with beta = 1 and beta = 3 (the backward-gradient form): bit-identical to the same call on an explicit B */
    {
        double *C3 = malloc(m * n * 8), *C4 = malloc(m * n * 8);
        for (int bi = 0; bi < 2; bi++) {
            const double bv = bi ? 3.0 : 1.0;
            for (int64_t i = 0; i < m * n; i++) C3[i] = C4[i] = sin(0.37 * i + 0.2);
            tsr_gemm(TSR_F64, 0, 1, m, n, k, 1.0, A, k, BT, k, bv, C3, n);
            tsr_gemm(TSR_F64, 0, 0, m, n, k, 1.0, A, k, B, n, bv, C4, n);
            CHECK(memcmp(C3, C4, (size_t)(m * n) * 8) == 0, "gemm NT (packed B) bit-identical to NN, beta %.0f", bv);
        }
        free(C3); free(C4);
    }

    /* TT: both transposed */
    tsr_gemm(TSR_F64, 1, 1, m, n, k, 1.0, AT, m, BT, k, 0.0, C2, n);
    err = 0; for (int64_t i = 0; i < m * n; i++) err = fmax(err, fabs(C2[i] - C[i]));
    CHECK(err < 1e-12, "gemm TT matches NN err %.3e", err);

    /* alpha/beta accumulate: C <- 2*A@B + 3*C, starting from the current C (= A@B) */
    for (int64_t i = 0; i < m * n; i++) C2[i] = C[i];
    tsr_gemm(TSR_F64, 0, 0, m, n, k, 2.0, A, k, B, n, 3.0, C2, n);
    err = 0; for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++)
        err = fmax(err, fabs(C2[i * n + j] - (2.0 * C[i * n + j] + 3.0 * C[i * n + j])));
    CHECK(err < 1e-11, "gemm alpha/beta err %.3e", err);

    /* CBLAS constants (111 NoTrans / 112 Trans) accepted like 0/1 */
    tsr_gemm(TSR_F64, 112, 111, m, n, k, 1.0, AT, m, B, n, 0.0, C2, n);
    err = 0; for (int64_t i = 0; i < m * n; i++) err = fmax(err, fabs(C2[i] - C[i]));
    CHECK(err < 1e-12, "gemm CBLAS flag aliases err %.3e", err);

    /* float32 path */
    float *Af = malloc(m * k * 4), *Bf = malloc(k * n * 4), *Cf = malloc(m * n * 4);
    for (int64_t i = 0; i < m * k; i++) Af[i] = (float)A[i];
    for (int64_t i = 0; i < k * n; i++) Bf[i] = (float)B[i];
    tsr_gemm(TSR_F32, 0, 0, m, n, k, 1.0, Af, k, Bf, n, 0.0, Cf, n);
    err = 0; for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++)
        err = fmax(err, fabs((double)Cf[i * n + j] - gemm_ref(0, 0, m, n, k, 1.0, A, k, B, n, 0.0, NULL, i, j)));
    CHECK(err < 1e-3, "gemm f32 err %.3e", err);

    CHECK(tsr_gemm(TSR_I64, 0, 0, m, n, k, 1.0, A, k, B, n, 0.0, C, n) == TSR_ETYPE, "gemm rejects int");

    /* Series N9-1: the threaded tsr_gemm gives the same bits at 1, 2, 3, 4 and 8 threads (rows split in fixed blocks;
       every element keeps its fma chain). 96 x 80 x 192 exceeds the threading threshold; all four transpose cases. */
    {
        const int64_t tm = 96, tn = 80, tk = 192;
        float *ta_ = malloc(tm * tk * 4), *tb_ = malloc(tk * tn * 4), *ref = malloc(tm * tn * 4), *got = malloc(tm * tn * 4);
        double *dta = malloc(tm * tk * 8), *dtb = malloc(tk * tn * 8), *dref = malloc(tm * tn * 8), *dgot = malloc(tm * tn * 8);
        for (int64_t i = 0; i < tm * tk; i++) { ta_[i] = (float)sin(0.37 * i + 0.5); dta[i] = sin(0.37 * i + 0.5); }
        for (int64_t i = 0; i < tk * tn; i++) { tb_[i] = (float)cos(0.29 * i + 1.5); dtb[i] = cos(0.29 * i + 1.5); }
        const int saved = tsr_get_threads();
        int same = 1;
        for (int tr = 0; tr < 4; tr++) {
            const int xa = tr & 1, xb = (tr >> 1) & 1;
            const int64_t la = xa ? tm : tk, lb = xb ? tk : tn;
            tsr_set_threads(1);
            tsr_gemm(TSR_F32, xa, xb, tm, tn, tk, 1.0, ta_, la, tb_, lb, 0.0, ref, tn);
            tsr_gemm(TSR_F64, xa, xb, tm, tn, tk, 1.0, dta, la, dtb, lb, 0.0, dref, tn);
            const int counts[4] = { 2, 3, 4, 8 };
            for (int q = 0; q < 4; q++) {
                tsr_set_threads(counts[q]);
                for (int64_t i = 0; i < tm * tn; i++) { got[i] = 1.0f; dgot[i] = 1.0; }
                tsr_gemm(TSR_F32, xa, xb, tm, tn, tk, 1.0, ta_, la, tb_, lb, 0.0, got, tn);
                tsr_gemm(TSR_F64, xa, xb, tm, tn, tk, 1.0, dta, la, dtb, lb, 0.0, dgot, tn);
                if (memcmp(ref, got, tm * tn * 4) != 0 || memcmp(dref, dgot, tm * tn * 8) != 0) same = 0;
            }
        }
        tsr_set_threads(saved);
        CHECK(same, "gemm bit-identical at 1, 2, 3, 4, 8 threads (f32 and f64, all transposes)");
        free(ta_); free(tb_); free(ref); free(got); free(dta); free(dtb); free(dref); free(dgot);
    }

    free(A); free(AT); free(B); free(BT); free(C); free(C2); free(Af); free(Bf); free(Cf);
}

static void test_axpy(void)
{
    const int64_t n = 50;
    double *x = malloc(n * 8), *y = malloc(n * 8), *ref = malloc(n * 8);
    for (int64_t i = 0; i < n; i++) { x[i] = sin(0.4 * i + 1); y[i] = cos(0.3 * i); }

    /* f64 unit stride, alpha = -0.1 (an SGD step W -= lr*dW) */
    for (int64_t i = 0; i < n; i++) ref[i] = y[i] + (-0.1) * x[i];
    tsr_axpy(TSR_F64, n, -0.1, x, 1, y, 1);
    double err = 0; for (int64_t i = 0; i < n; i++) err = fmax(err, fabs(y[i] - ref[i]));
    CHECK(err < 1e-14, "axpy f64 unit err %.3e", err);

    /* alpha = 0 is a no-op */
    double *y0 = malloc(n * 8); for (int64_t i = 0; i < n; i++) y0[i] = y[i];
    tsr_axpy(TSR_F64, n, 0.0, x, 1, y, 1);
    err = 0; for (int64_t i = 0; i < n; i++) err = fmax(err, fabs(y[i] - y0[i]));
    CHECK(err == 0.0, "axpy alpha=0 no-op");

    /* non-unit stride: y[2i] += 3*x[3i], for i in 0..k-1 */
    for (int64_t i = 0; i < n; i++) { x[i] = (double)(i + 1); y[i] = 0.0; }
    const int64_t k = 10;
    tsr_axpy(TSR_F64, k, 3.0, x, 3, y, 2);
    err = 0; for (int64_t i = 0; i < k; i++) err = fmax(err, fabs(y[2 * i] - 3.0 * x[3 * i]));
    CHECK(err == 0.0, "axpy strided");

    /* negative stride (reference-BLAS): y walked forward, x from the far end */
    for (int64_t i = 0; i < n; i++) { x[i] = (double)i; y[i] = 0.0; }
    tsr_axpy(TSR_F64, 5, 1.0, x, -1, y, 1);   /* y[i] += x[(5-1-i)] = x[4-i] */
    err = 0; for (int64_t i = 0; i < 5; i++) err = fmax(err, fabs(y[i] - x[4 - i]));
    CHECK(err == 0.0, "axpy negative incx");

    /* f32 path */
    float *xf = malloc(n * 4), *yf = malloc(n * 4);
    for (int64_t i = 0; i < n; i++) { xf[i] = (float)x[i]; yf[i] = 1.0f; }
    tsr_axpy(TSR_F32, n, 2.5, xf, 1, yf, 1);
    err = 0; for (int64_t i = 0; i < n; i++) err = fmax(err, fabs((double)yf[i] - (1.0 + 2.5 * x[i])));
    CHECK(err < 1e-4, "axpy f32 err %.3e", err);

    CHECK(tsr_axpy(TSR_I64, n, 1.0, x, 1, y, 1) == TSR_ETYPE, "axpy rejects int");
    CHECK(tsr_axpy(TSR_F64, 0, 1.0, x, 1, y, 1) == TSR_OK, "axpy n=0 no-op ok");
    CHECK(tsr_axpy(TSR_F64, -1, 1.0, x, 1, y, 1) == TSR_EARG, "axpy n<0 rejected");

    free(x); free(y); free(ref); free(y0); free(xf); free(yf);
}

static void test_fft(void)
{
    const int64_t sizes[] = {1, 2, 3, 4, 5, 6, 7, 8, 12, 15, 16, 30, 49, 97, 101, 128, 210, 1000, 1009, 1024, 4096};
    for (size_t t = 0; t < sizeof(sizes) / sizeof(sizes[0]); t++) {
        int64_t n = sizes[t];
        double *x = malloc(2 * n * 8), *X = malloc(2 * n * 8), *y = malloc(2 * n * 8);
        for (int64_t i = 0; i < n; i++) { x[2 * i] = sin(0.37 * i) + 0.1 * i; x[2 * i + 1] = cos(1.3 * i); }
        CHECK(tsr_fft(n, 1, 0, x, X) == 0, "fft rc n=%ld", (long)n);
        double err = 0, scale = 0;
        for (int64_t k = 0; k < n; k++) {
            long double re = 0, im = 0;
            for (int64_t j = 0; j < n; j++) {
                long double a = -2.0L * 3.14159265358979323846264338327950288L * (long double)((j * k) % n) / n;
                re += x[2 * j] * cosl(a) - x[2 * j + 1] * sinl(a);
                im += x[2 * j] * sinl(a) + x[2 * j + 1] * cosl(a);
            }
            err = fmax(err, fmax(fabs((double)re - X[2 * k]), fabs((double)im - X[2 * k + 1])));
            scale = fmax(scale, fabs((double)re));
        }
        CHECK(err <= 1e-12 * (1 + scale) * log2((double)n + 1), "fft n=%ld err %.3e", (long)n, err);
        tsr_fft(n, 1, 1, X, y);
        double rt = 0;
        for (int64_t i = 0; i < 2 * n; i++) rt = fmax(rt, fabs(y[i] - x[i]));
        CHECK(rt < 1e-11 * (1 + n * 0.01), "ifft roundtrip n=%ld err %.3e", (long)n, rt);
        free(x); free(X); free(y);
    }
    /* in-place, several rows */
    double r[16];
    for (int i = 0; i < 16; i++) r[i] = i;
    tsr_fft(4, 2, 0, r, r);
    CHECK(r[0] == 12 && r[1] == 16 && r[8] == 44 && r[9] == 48, "in-place multi-row");

    /* plan cache: same bits cached, uncached and real-input; the limit bounds what is held */
    {
        const int64_t n = 1000, saved = tsr_fft_cache_limit();
        double *x = malloc(2 * n * 8), *xr = malloc(n * 8), *a = malloc(2 * n * 8), *b = malloc(2 * n * 8), *c = malloc(2 * n * 8);
        for (int64_t i = 0; i < n; i++) { xr[i] = sin(0.37 * i) + 0.1 * i; x[2 * i] = xr[i]; x[2 * i + 1] = 0.0; }
        tsr_set_fft_cache_limit(0);
        CHECK(tsr_fft_cache_bytes() == 0, "fft cache emptied at limit 0 (%lld)", (long long)tsr_fft_cache_bytes());
        tsr_fft(n, 1, 0, x, a);                                   /* uncached */
        CHECK(tsr_fft_cache_bytes() == 0, "nothing cached at limit 0");
        tsr_set_fft_cache_limit((int64_t)64 << 20);
        tsr_fft(n, 1, 0, x, b);                                   /* builds and caches */
        CHECK(tsr_fft_cache_bytes() > 0, "plan cached");
        tsr_fft_real(n, 1, 0, xr, c);                              /* cached plan, real input */
        CHECK(memcmp(a, b, (size_t)(2 * n * 8)) == 0 && memcmp(a, c, (size_t)(2 * n * 8)) == 0, "fft cached == uncached == real input");
        tsr_set_fft_cache_limit(-5);
        CHECK(tsr_fft_cache_limit() == 0 && tsr_fft_cache_bytes() == 0, "negative limit clamps to 0 and frees");
        tsr_set_fft_cache_limit(saved);
        CHECK(tsr_fft_cache_limit() == saved, "limit restored");
        free(x); free(xr); free(a); free(b); free(c);
    }
}

static void test_rng(void)
{
    /* numpy.random.default_rng(12345) */
    uint32_t ent[1] = {12345};
    uint64_t w[4];
    tsr_seed_sequence(ent, 1, w, 4);
    CHECK(w[0] == 0xb5ae6482a03d837cULL && w[3] == 0x3ebb0f96a013fd73ULL, "SeedSequence %016llx", (unsigned long long)w[0]);
    uint64_t st[6];
    tsr_pcg64_seed(st, w[0], w[1], w[2], w[3]);
    double u[3];
    tsr_pcg64_random(st, 3, u);
    CHECK(u[0] == 0x1.d1958c6d97fe0p-3 && u[1] == 0x1.445c4c5727930p-2 && u[2] == 0x1.984049046889dp-1,
          "random() matches numpy %a", u[0]);

    tsr_pcg64_seed(st, w[0], w[1], w[2], w[3]);
    int64_t ints[5], big[2];
    tsr_pcg64_integers(st, 5, 0, 10, ints);
    tsr_pcg64_integers(st, 2, -5, 1LL << 40, big);
    CHECK(ints[0] == 6 && ints[1] == 2 && ints[2] == 7 && ints[3] == 3 && ints[4] == 2, "integers small");
    CHECK(big[0] == 743549873826LL && big[1] == 430029498618LL, "integers 64-bit %lld", (long long)big[0]);

    tsr_pcg64_seed(st, w[0], w[1], w[2], w[3]);
    double nv[4];
    tsr_pcg64_normal(st, 4, 1.5, 2.0, nv);
    CHECK(nv[0] == -0x1.58ff985d98e88p+0 && nv[1] == 0x1.01c1daa758324p+2 && nv[2] == -0x1.ee3b00a69acf0p-3 &&
              nv[3] == 0x1.f69b4a9fd53c4p-1,
          "normal matches numpy %a", nv[0]);

    /* numpy.random.default_rng(12345).permutation(10) */
    tsr_pcg64_seed(st, w[0], w[1], w[2], w[3]);
    int64_t perm[10];
    for (int i = 0; i < 10; i++) perm[i] = i;
    tsr_pcg64_shuffle(st, 10, 8, perm);
    CHECK(perm[0] == 4 && perm[9] == 5, "permutation matches numpy (%ld ... %ld)", (long)perm[0], (long)perm[9]);

    /* distribution sanity on 1e6 normals */
    int64_t N = 1000000;
    double *z = malloc(N * 8), s = 0, s2 = 0;
    tsr_pcg64_normal(st, N, 0, 1, z);
    for (int64_t i = 0; i < N; i++) { s += z[i]; s2 += z[i] * z[i]; }
    CHECK(fabs(s / N) < 0.005 && fabs(s2 / N - 1) < 0.01, "normal moments");
    free(z);
}

static void test_sparse(void)
{
    /* 1-D Poisson matrix, SPD */
    int64_t n = 200, nnz = 3 * n - 2;
    int64_t *ip = malloc((n + 1) * 8), *ix = malloc(nnz * 8);
    double *d = malloc(nnz * 8), *b = malloc(n * 8), *x = calloc(n, 8), *y = malloc(n * 8);
    int64_t p = 0;
    ip[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        if (i > 0) { ix[p] = i - 1; d[p++] = -1; }
        ix[p] = i; d[p++] = 2;
        if (i < n - 1) { ix[p] = i + 1; d[p++] = -1; }
        ip[i + 1] = p;
        b[i] = 1.0;
    }
    double res;
    int64_t it = tsr_csr_cg(n, ip, ix, d, b, x, 1e-10, 0, 1000, &res);
    CHECK(it > 0, "cg converged (%ld)", (long)it);
    tsr_csr_matvec(n, ip, ix, d, x, y);
    double err = 0; for (int64_t i = 0; i < n; i++) err = fmax(err, fabs(y[i] - 1));
    CHECK(err < 1e-7, "cg residual %.3e", err);

    memset(x, 0, n * 8);
    it = tsr_csr_bicgstab(n, ip, ix, d, b, x, 1e-10, 0, 2000, &res);
    tsr_csr_matvec(n, ip, ix, d, x, y);
    err = 0; for (int64_t i = 0; i < n; i++) err = fmax(err, fabs(y[i] - 1));
    CHECK(it > 0 && err < 1e-7, "bicgstab it=%ld residual %.3e", (long)it, err);

    double dense[6] = {0, 1, 0, 2, 0, 3};
    CHECK(tsr_dense_nnz(2, 3, dense) == 3, "nnz");
    int64_t cip[3], cix[3], tip[4], tix[3];
    double cd[3], td[3];
    tsr_dense_to_csr(2, 3, dense, cip, cix, cd);
    CHECK(cip[1] == 1 && cix[1] == 0 && cd[2] == 3, "dense->csr");
    tsr_csr_transpose(2, 3, cip, cix, cd, tip, tix, td);
    CHECK(tip[1] == 1 && tip[2] == 2 && tip[3] == 3 && td[0] == 2 && td[1] == 1, "transpose");
    double B[6] = {1, 2, 3, 4, 5, 6}, Cm[4];
    tsr_csr_matmat(2, 2, cip, cix, cd, B, Cm); /* [0 1 0;2 0 3] * [1 2;3 4;5 6] */
    CHECK(Cm[0] == 3 && Cm[1] == 4 && Cm[2] == 17 && Cm[3] == 22, "csr matmat");
    free(ip); free(ix); free(d); free(b); free(x); free(y);
}

static void test_array_meta(void)
{
    double buf[24];
    for (int i = 0; i < 24; i++) buf[i] = i;
    tsr_array a, v, t, r;
    int64_t sh[3] = {2, 3, 4};
    CHECK(tsr_array_init(&a, TSR_F64, 3, sh, buf) == 0 && a.strides[0] == 96 && a.strides[2] == 8, "init strides");
    CHECK(tsr_array_slice(&a, "1, ::-1, 1:4:2", &v) == 0, "slice rc");
    CHECK(v.ndim == 2 && v.shape[0] == 3 && v.shape[1] == 2, "slice shape %d %lld %lld", v.ndim, (long long)v.shape[0], (long long)v.shape[1]);
    int64_t ix[2] = {0, 1};
    CHECK(*(double *)tsr_array_at(&v, ix) == 12 + 8 + 3, "slice element %g", *(double *)tsr_array_at(&v, ix));
    CHECK(tsr_array_slice(&a, "..., None, -1", &v) == 0 && v.ndim == 3 && v.shape[2] == 1 && v.strides[2] == 0, "ellipsis + newaxis");
    CHECK(tsr_array_slice(&a, "5", &v) == TSR_EINDEX, "index out of range");
    CHECK(tsr_array_slice(&a, "1:x", &v) == TSR_EARG, "syntax error");
    CHECK(tsr_array_slice(&a, "::0", &v) == TSR_EARG, "zero step");
    CHECK(tsr_array_slice(&a, "0,0,0,0", &v) == TSR_EINDEX, "too many indices");
    CHECK(tsr_array_slice(&a, "1:0", &v) == 0 && v.shape[0] == 0, "empty slice");
    tsr_array_transpose(&a, NULL, &t);
    CHECK(t.shape[0] == 4 && t.strides[0] == 8 && !tsr_array_is_contiguous(&t), "transpose view");
    CHECK(tsr_array_reshape(&t, 1, (int64_t[]){-1}, &r) == TSR_ECONTIG, "reshape needs copy");
    CHECK(tsr_array_reshape(&a, 2, (int64_t[]){-1, 6}, &r) == 0 && r.shape[0] == 4, "reshape -1");
    CHECK(tsr_array_reshape(&a, 2, (int64_t[]){5, 5}, &r) == TSR_ESHAPE, "reshape mismatch");

    int32_t ond;
    int64_t osh[32];
    CHECK(tsr_broadcast_shape(2, (int64_t[]){3, 1}, 1, (int64_t[]){4}, &ond, osh) == 0 && ond == 2 && osh[0] == 3 && osh[1] == 4, "broadcast shape");
    CHECK(tsr_broadcast_shape(1, (int64_t[]){3}, 1, (int64_t[]){4}, &ond, osh) == TSR_ESHAPE, "broadcast mismatch");

    double row[4] = {10, 20, 30, 40}, res[12];
    tsr_array ar, out, m;
    tsr_array_init(&ar, TSR_F64, 1, (int64_t[]){4}, row);
    tsr_array_init(&m, TSR_F64, 2, (int64_t[]){3, 4}, buf);
    tsr_array_init(&out, TSR_F64, 2, (int64_t[]){3, 4}, res);
    CHECK(tsr_array_binary(OP_ADD, &m, &ar, &out) == 0 && res[0] == 10 && res[11] == 51, "array binary broadcast");

    char js[256];
    tsr_array small;
    double sv[4] = {1, 0.1, NAN, -2.5e-300};
    tsr_array_init(&small, TSR_F64, 2, (int64_t[]){2, 2}, sv);
    int64_t need = tsr_array_json(&small, NULL, 0);
    int64_t got = tsr_array_json(&small, js, sizeof(js));
    js[got] = 0;
    CHECK(need == got && strcmp(js, "[[1.0,0.1],[null,-2.5e-300]]") == 0, "json %s", js);
    int64_t iv[3] = {-1, 0, 7};
    tsr_array ia;
    tsr_array_init(&ia, TSR_I64, 1, (int64_t[]){3}, iv);
    got = tsr_array_json(&ia, js, sizeof(js));
    js[got] = 0;
    CHECK(strcmp(js, "[-1,0,7]") == 0, "json ints %s", js);
    tsr_array rev;
    tsr_array_slice(&ia, "::-1", &rev);
    got = tsr_array_json(&rev, js, sizeof(js));
    js[got] = 0;
    CHECK(strcmp(js, "[7,0,-1]") == 0, "json reversed view %s", js);
    CHECK(tsr_array_json(&ia, js, 3) == 8, "json reports needed size when truncated");
}

static void test_threads(void)
{
    int64_t N = 1 << 20;
    double *A = malloc(N * 8), *B = malloc(N * 8), *C1 = malloc(N * 8), *C2 = malloc(N * 8);
    for (int64_t i = 0; i < N; i++) { A[i] = sin((double)i); B[i] = cos((double)i); }
    int64_t n1[1] = {N}, c8[1] = {8};
    tsr_set_threads(1);
    tsr_binary(OP_HYPOT, TSR_F64, 1, n1, A, c8, B, c8, C1, c8);
    tsr_set_threads(4);
    tsr_binary(OP_HYPOT, TSR_F64, 1, n1, A, c8, B, c8, C2, c8);
    CHECK(memcmp(C1, C2, N * 8) == 0, "threaded result identical to serial");
    tsr_unary(U_EXP, TSR_F64, 1, n1, A, c8, C2, c8);
    tsr_set_threads(1);
    tsr_unary(U_EXP, TSR_F64, 1, n1, A, c8, C1, c8);
    CHECK(memcmp(C1, C2, N * 8) == 0, "threaded unary identical");
    CHECK(tsr_get_threads() == 1, "threads reset");
    free(A); free(B); free(C1); free(C2);
}

static void test_lp(void)
{
    /* max 3x + 2y s.t. x + y <= 4, x + 3y <= 7, x <= 3  -> x=3, y=1, obj 11; duals -2, 0, -1 (non-degenerate) */
    double c[2] = {-3, -2}, A[6] = {1, 1, 1, 3, 1, 0}, b[3] = {4, 7, 3}, x[2], fun, y[3], red[2];
    int64_t nit;
    int st = tsr_linprog(2, c, 3, A, b, 0, NULL, NULL, NULL, NULL, 0, 0, x, &fun, y, NULL, red, &nit);
    CHECK(st == 0 && fabs(x[0] - 3) < 1e-9 && fabs(x[1] - 1) < 1e-9 && fabs(fun + 11) < 1e-9, "lp basic st=%d x=%g,%g", st, x[0], x[1]);
    CHECK(fabs(y[0] + 2) < 1e-9 && fabs(y[1]) < 1e-9 && fabs(y[2] + 1) < 1e-9, "lp duals %g %g %g", y[0], y[1], y[2]);
    /* equality + free + bounded variables: min x0 + 2x1 - x2, x0 + x1 + x2 = 5, x0 - x2 >= -1, -2<=x1<=2, x2 free <= 10 */
    double c2[3] = {1, 2, -1}, Aeq[3] = {1, 1, 1}, beq[1] = {5}, Aub[3] = {-1, 0, 1}, bub[1] = {1};
    double lb[3] = {0, -2, -INFINITY}, ub[3] = {INFINITY, 2, 10}, x3[3], ye[1], yu[1];
    st = tsr_linprog(3, c2, 1, Aub, bub, 1, Aeq, beq, lb, ub, 0, 0, x3, &fun, yu, ye, NULL, &nit);
    /* optimum: x1 = -2, then x0 + x2 = 7 with x2 <= x0 + 1 -> x2 = 4, x0 = 3: 3 - 4 - 4 = -5 */
    CHECK(st == 0 && fabs(fun + 5) < 1e-9 && fabs(x3[1] + 2) < 1e-9, "lp mixed bounds st=%d fun=%g x=%g,%g,%g", st, fun, x3[0], x3[1], x3[2]);
    double dual_obj = beq[0] * ye[0] + bub[0] * yu[0] + (-2) * (c2[1] - (ye[0] * 1 + yu[0] * 0));
    CHECK(fabs(dual_obj - fun) < 1e-9, "strong duality %g vs %g", dual_obj, fun);
    /* infeasible and unbounded */
    double ci[1] = {1}, Ai[1] = {1}, bi[1] = {-1};
    CHECK(tsr_linprog(1, ci, 1, Ai, bi, 0, NULL, NULL, NULL, NULL, 0, 0, x, &fun, NULL, NULL, NULL, NULL) == 2, "infeasible");
    double cu[1] = {-1};
    CHECK(tsr_linprog(1, cu, 0, NULL, NULL, 0, NULL, NULL, NULL, NULL, 0, 0, x, &fun, NULL, NULL, NULL, NULL) == 3, "unbounded");
    {
        double cn[2] = {1.0, NAN}, An[2] = {1.0, INFINITY}, bn[1] = {1.0}, xn[2], fn;
        CHECK(tsr_linprog(2, cn, 0, NULL, NULL, 0, NULL, NULL, NULL, NULL, 0, 0, xn, &fn, NULL, NULL, NULL, NULL) == -1, "linprog rejects NaN in c");
        double cf[2] = {1.0, 1.0};
        CHECK(tsr_linprog(2, cf, 1, An, bn, 0, NULL, NULL, NULL, NULL, 0, 0, xn, &fn, NULL, NULL, NULL, NULL) == -1, "linprog rejects inf in A");
        CHECK(tsr_milp(2, cn, 0, NULL, NULL, 0, NULL, NULL, NULL, NULL, NULL, 0, 0, 0, xn, &fn, NULL, NULL) == -1, "milp rejects NaN in c");
        /* moments follow numpy: pairwise sum / n, then pairwise sum of squared deviations */
        double half[48], mu, m2v;
        for (int k = 0; k < 48; k++) half[k] = k < 24 ? 1.0 : 0.0;
        int64_t hs[1] = {48}, hst[1] = {8}, z1[1] = {0};
        CHECK(tsr_moments(0, 1, hs, half, hst, 0, &mu, z1, &m2v, z1) == 0 && mu == 0.5 && m2v == 12.0, "mean of 24 ones and 24 zeros is exactly 0.5");
        /* the tableau counts against the memory budget; milp reports the failure instead of pruning */
        enum { NB = 60 };
        static double cb2[NB], Ab2[NB * NB], bb2[NB], xb2[NB];
        for (int j = 0; j < NB; j++) { cb2[j] = -1.0 - j % 3; bb2[j] = 10.0 + j; for (int k = 0; k < NB; k++) Ab2[j * NB + k] = (j == k) ? 2.0 : 0.1; }
        int64_t before = tsr_allocated();
        tsr_set_budget(before + 4096);
        CHECK(tsr_linprog(NB, cb2, NB, Ab2, bb2, 0, NULL, NULL, NULL, NULL, 0, 0, xb2, &fn, NULL, NULL, NULL, NULL) == -3, "linprog tableau respects the budget");
        uint8_t ib2[NB];
        memset(ib2, 1, sizeof ib2);
        CHECK(tsr_milp(NB, cb2, NB, Ab2, bb2, 0, NULL, NULL, NULL, NULL, ib2, 0, 0, 0, xb2, &fn, NULL, NULL) == -3, "milp reports out-of-budget");
        tsr_set_budget(0);
        CHECK(tsr_allocated() == before, "no leak after budget failures");
        CHECK(tsr_linprog(NB, cb2, NB, Ab2, bb2, 0, NULL, NULL, NULL, NULL, 0, 0, xb2, &fn, NULL, NULL, NULL, NULL) == 0, "same LP solves without budget");
    }
    /* degenerate (Beale-style cycling example) must terminate */
    double cb[4] = {-0.75, 150, -0.02, 6}, Ab[12] = {0.25, -60, -0.04, 9, 0.5, -90, -0.02, 3, 0, 0, 1, 0}, bb[3] = {0, 0, 1}, xb[4];
    st = tsr_linprog(4, cb, 3, Ab, bb, 0, NULL, NULL, NULL, NULL, 0, 0, xb, &fun, NULL, NULL, NULL, &nit);
    CHECK(st == 0 && fabs(fun + 0.05) < 1e-9, "Beale cycling example st=%d fun=%g", st, fun);

    /* 0/1 knapsack: values 10,13,7,8 weights 5,6,4,3 cap 10 -> best 21 (items 0? ) */
    double kc[4] = {-10, -13, -7, -8}, kA[4] = {5, 6, 4, 3}, kb[1] = {10}, klb[4] = {0, 0, 0, 0}, kub[4] = {1, 1, 1, 1}, kx[4], bound;
    uint8_t integ[4] = {1, 1, 1, 1};
    int64_t nodes;
    st = tsr_milp(4, kc, 1, kA, kb, 0, NULL, NULL, klb, kub, integ, 0, 0, 0, kx, &fun, &nodes, &bound);
    CHECK(st == 0 && fabs(fun + 21) < 1e-9, "knapsack st=%d fun=%g", st, fun);
    /* integer infeasible: 2x = 1 */
    double ic[1] = {1}, iA[1] = {2}, ib[1] = {1};
    uint8_t one[1] = {1};
    st = tsr_milp(1, ic, 0, NULL, NULL, 1, iA, ib, NULL, NULL, one, 0, 0, 0, kx, &fun, &nodes, &bound);
    CHECK(st == 2, "integer infeasible st=%d", st);
}

static void test_mdp(void)
{
    /* 2 states, 2 actions. a0: stay, a1: move. R(s0)=(0, 0), R(s1)=(1, 0). Optimal: s0 move, s1 stay.
       V(s1) = 1/(1-g), V(s0) = g V(s1). */
    const double g = 0.9;
    int64_t indptr[5] = {0, 1, 2, 3, 4}, indices[4] = {0, 1, 1, 0};
    double data[4] = {1, 1, 1, 1}, R[4] = {0, 0, 1, 0}, V[2] = {0, 0};
    int64_t pol[2], it;
    double delta;
    int st = tsr_mdp_value_iteration(2, 2, indptr, indices, data, R, g, 1e-8, 0, V, pol, &it, &delta);
    CHECK(st == 0 && pol[0] == 1 && pol[1] == 0, "VI policy %lld %lld", (long long)pol[0], (long long)pol[1]);
    CHECK(fabs(V[1] - 10) < 1e-6 && fabs(V[0] - 9) < 1e-6, "VI values %g %g", V[0], V[1]);
    int64_t p2[2] = {0, 1};
    double V2[2];
    st = tsr_mdp_policy_iteration(2, 2, indptr, indices, data, R, g, 0, 0, 0, V2, p2, &it);
    CHECK(st == 0 && p2[0] == 1 && p2[1] == 0 && fabs(V2[1] - 10) < 1e-12 && fabs(V2[0] - 9) < 1e-12, "PI exact");
    int64_t p3[2] = {0, 0};
    double V3[2];
    st = tsr_mdp_policy_iteration(2, 2, indptr, indices, data, R, g, 20, 1e-10, 0, V3, p3, &it);
    CHECK(st == 0 && p3[0] == 1 && p3[1] == 0, "modified PI");
    double Vf[3 * 2];
    int64_t pf[2 * 2];
    tsr_mdp_finite_horizon(2, 2, indptr, indices, data, R, 1.0, 2, NULL, Vf, pf);
    CHECK(Vf[0] == 1 && Vf[1] == 2 && Vf[2] == 0 && Vf[3] == 1 && pf[0] == 1 && pf[1] == 0, "finite horizon");
    double Q[4], Vo[2];
    tsr_mdp_bellman(2, 2, indptr, indices, data, R, g, V2, Q, Vo, pol);
    CHECK(fabs(Q[1] - 9) < 1e-12 && fabs(Q[0] - 8.1) < 1e-12, "bellman Q");
    int64_t bad[5] = {0, 1, 2, 3, 4}, badi[4] = {0, 1, 5, 0};
    CHECK(tsr_mdp_value_iteration(2, 2, bad, badi, data, R, g, 0, 0, V, pol, &it, &delta) == TSR_EINDEX, "bad index");
}

static void test_ufunc(void)
{
    CHECK(tsr_ufunc_count() == 24, "ufunc count %d", tsr_ufunc_count());
    const int cb = tsr_ufunc_find("cbrt"), fm = tsr_ufunc_find("fmod"), fx = tsr_ufunc_find("fmax"), tr = tsr_ufunc_find("trunc");
    CHECK(cb >= 0 && fm >= 0 && fx >= 0 && tr >= 0 && tsr_ufunc_find("nope") < 0 && tsr_ufunc_find(NULL) < 0, "find");
    CHECK(tsr_ufunc_nin(cb) == 1 && tsr_ufunc_nin(fm) == 2 && tsr_ufunc_name(-1) == NULL && tsr_ufunc_name(cb)[0] == 'c', "meta");
    int ld, od, op = -1;
    CHECK(tsr_ufunc_resolve(cb, TSR_I32, &ld, &od, &op) == 0 && ld == TSR_F64 && od == TSR_F64, "int -> float64 loop");
    CHECK(tsr_ufunc_resolve(cb, TSR_F32, &ld, &od, &op) == 0 && ld == TSR_F32, "float32 loop");
    CHECK(tsr_ufunc_resolve(tr, TSR_I64, &ld, &od, &op) == 1 && od == TSR_I64, "identity");
    CHECK(tsr_ufunc_resolve(fx, TSR_I32, &ld, &od, &op) == 2 && op == OP_MAX, "native fallback");
    CHECK(tsr_ufunc_resolve(fm, TSR_BOOL, &ld, &od, &op) == 0 && ld == TSR_I64, "bool fmod -> int64");
    CHECK(tsr_ufunc_resolve(fm, TSR_U8, &ld, &od, &op) == 0 && ld == TSR_U8, "u8 fmod");
    CHECK(tsr_ufunc_resolve(cb, TSR_C128, &ld, &od, &op) == TSR_ETYPE, "complex refused");

    double x[4] = {-27, 8, 0.0, -0.0}, y[4];
    /* libm at run time: GCC would fold cbrt(constant) with MPFR, correctly rounded, unlike glibc */
    volatile double v27 = 27.0, v8 = 8.0;
    const double cb27 = cbrt(v27), cbm27 = cbrt(-v27), cb8 = cbrt(v8);
    int64_t n4[1] = {4}, c8[1] = {8}, z[1] = {0};
    CHECK(tsr_ufunc(cb, TSR_F64, 1, n4, x, c8, NULL, NULL, y, c8) == 0 && y[0] == cbm27 && y[1] == cb8, "cbrt is libm's (numpy.cbrt, ADR 0012) %.17g", y[0]);
    CHECK(tsr_ufunc(cb, TSR_I64, 1, n4, x, c8, NULL, NULL, y, c8) == TSR_ETYPE, "no int loop");
    int64_t a[4] = {7, -7, 5, INT64_MIN}, b = -1, r[4];
    CHECK(tsr_ufunc(fm, TSR_I64, 1, n4, a, c8, &b, z, r, c8) == 0 && r[0] == 0 && r[3] == 0, "fmod int MIN %% -1");
    int64_t zero = 0;
    tsr_ufunc(fm, TSR_I64, 1, n4, a, c8, &zero, z, r, c8);
    CHECK(r[0] == 0 && r[1] == 0, "fmod by zero");
    double p[4] = {1, NAN, 3, NAN}, q[4] = {NAN, 2, 1, NAN}, m[4];
    tsr_ufunc(fx, TSR_F64, 1, n4, p, c8, q, c8, m, c8);
    CHECK(m[0] == 1 && m[1] == 2 && m[2] == 3 && isnan(m[3]), "fmax ignores NaN");
    /* in place, 0-d, strided */
    tsr_ufunc(cb, TSR_F64, 1, n4, x, c8, NULL, NULL, x, c8);
    CHECK(x[0] == cbm27 && signbit(x[3]), "in place");
    double s = 64, so;
    CHECK(tsr_ufunc(cb, TSR_F64, 0, NULL, &s, NULL, NULL, NULL, &so, NULL) == 0 && so == 4, "0-d");
    double w[6] = {1, 0, 8, 0, 27, 0}, wo[3];
    int64_t n3[1] = {3}, s16[1] = {16};
    tsr_ufunc(cb, TSR_F64, 1, n3, w, s16, NULL, NULL, wo, c8);
    CHECK(wo[0] == 1 && wo[1] == cb8 && wo[2] == cb27, "strided");
    CHECK(tsr_ufunc(99, TSR_F64, 1, n4, x, c8, NULL, NULL, y, c8) == TSR_EARG, "bad id");
    CHECK(tsr_ufunc(fm, TSR_F64, 1, n4, x, c8, NULL, NULL, y, c8) == TSR_EARG, "missing b");
    /* threads: identical results */
    const int64_t N = (int64_t)1 << 18;
    double *big = malloc(sizeof(double) * N), *o1 = malloc(sizeof(double) * N), *o2 = malloc(sizeof(double) * N);
    for (int64_t i = 0; i < N; i++) big[i] = (double)i / 1000.0 - 100.0;
    int64_t nb[1] = {N};
    const int t0 = tsr_get_threads();
    tsr_set_threads(1);
    tsr_ufunc(tsr_ufunc_find("erf"), TSR_F64, 1, nb, big, c8, NULL, NULL, o1, c8);
    tsr_set_threads(4);
    tsr_ufunc(tsr_ufunc_find("erf"), TSR_F64, 1, nb, big, c8, NULL, NULL, o2, c8);
    tsr_set_threads(t0);
    CHECK(memcmp(o1, o2, sizeof(double) * N) == 0, "thread-count independent");
    free(big); free(o1); free(o2);
}

static void test_mmap_npy(void)
{
    int32_t nd;
    int64_t sh[32], bytes, start, len, ns;
    nd = 2; sh[0] = 3; sh[1] = 4;
    CHECK(tsr_mmap_plan(TSR_F64, 1, &nd, sh, 4096 + 8, 0, 4096, &bytes, &start, &len, &ns) == 0
          && bytes == 96 && start == 4096 && len == 104 && ns == 4096 + 8 + 96, "plan r+ extends");
    nd = 2;
    CHECK(tsr_mmap_plan(TSR_F64, 0, &nd, sh, 0, 10, 4096, &bytes, &start, &len, &ns) == TSR_ESHAPE, "plan r too small");
    nd = -1;
    CHECK(tsr_mmap_plan(TSR_I32, 0, &nd, sh, 4, 44, 4096, &bytes, &start, &len, &ns) == 0 && nd == 1 && sh[0] == 10, "plan derive 1-D");
    nd = -1;
    CHECK(tsr_mmap_plan(TSR_I32, 0, &nd, sh, 4, 42, 4096, &bytes, &start, &len, &ns) == TSR_ESHAPE && strlen(tsr_mmap_error()) > 0, "plan ragged");
    nd = -1;
    CHECK(tsr_mmap_plan(TSR_F64, 2, &nd, sh, 0, 100, 4096, &bytes, &start, &len, &ns) == TSR_EARG, "w+ needs shape");
    nd = 1; sh[0] = 3;
    CHECK(tsr_mmap_plan(TSR_F64, 0, &nd, sh, 3, 100, 4096, &bytes, &start, &len, &ns) == TSR_EARG, "misaligned offset");
    nd = 2; sh[0] = INT64_MAX / 2; sh[1] = 4;
    CHECK(tsr_mmap_plan(TSR_F64, 1, &nd, sh, 0, 0, 4096, &bytes, &start, &len, &ns) == TSR_ENOMEM, "overflow");
    nd = 1; sh[0] = 0;
    CHECK(tsr_mmap_plan(TSR_F64, 1, &nd, sh, 0, 0, 4096, &bytes, &start, &len, &ns) == TSR_ESHAPE, "empty");

    /* open / write / reopen / flush */
    char path[1024];
    snprintf(path, sizeof path, "%s/tsr_mmap_test_XXXXXX", test_tmpdir());
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp");
    if (fd >= 0) close(fd);
    void *h = NULL, *data = NULL;
    nd = 2; sh[0] = 100; sh[1] = 50;
    const int64_t before = tsr_mmap_bytes();
    CHECK(tsr_mmap_open(path, 2, TSR_F64, &nd, sh, 128, &h, &data) == 0 && h && data, "open w+ %s", tsr_mmap_error());
    if (h) {
        double *d = data;
        for (int i = 0; i < 5000; i++) d[i] = i * 0.5;
        CHECK(tsr_mmap_bytes() > before && tsr_mmap_length(h) >= 40000, "mapped bytes counted");
        CHECK(tsr_mmap_flush(h, 1) == 0, "flush sync");
        tsr_mmap_close(h);
        CHECK(tsr_mmap_bytes() == before, "mapped bytes released");
    }
    nd = -1;
    CHECK(tsr_mmap_open(path, 0, TSR_F64, &nd, sh, 128, &h, &data) == 0 && nd == 1 && sh[0] == 5000
          && ((double *)data)[4999] == 4999 * 0.5, "reopen r");
    if (h) tsr_mmap_close(h);
    nd = -1;
    CHECK(tsr_mmap_open(path, 3, TSR_F64, &nd, sh, 128, &h, &data) == 0, "open c");
    if (h) { ((double *)data)[0] = 42; tsr_mmap_close(h); }
    nd = -1;
    tsr_mmap_open(path, 0, TSR_F64, &nd, sh, 128, &h, &data);
    CHECK(h && ((double *)data)[0] == 0.0, "copy-on-write leaves the file");
    if (h) tsr_mmap_close(h);
    unlink(path);
    CHECK(tsr_mmap_open(path, 0, TSR_F64, &nd, sh, 0, &h, &data) == TSR_EIO && h == NULL, "missing file: %s", tsr_mmap_error());
    CHECK(tsr_mmap_open("php://memory", 0, TSR_F64, &nd, sh, 0, &h, &data) == TSR_EARG, "wrapper refused");
    CHECK(tsr_mmap_open(test_tmpdir(), 0, TSR_F64, &nd, sh, 0, &h, &data) != 0, "directory refused");

    /* .npy header round trip */
    char hb[256];
    int64_t shp[3] = {2, 3, 4};
    const int64_t hl = tsr_npy_write_header(TSR_F32, 3, shp, 0, hb, sizeof(hb));
    CHECK(hl == 128 && hl % 64 == 0 && hb[hl - 1] == '\n', "write header %lld", (long long)hl);
    int dt, fo;
    int64_t off;
    CHECK(tsr_npy_header(hb, hl, &dt, &nd, sh, &fo, &off) == 0 && dt == TSR_F32 && nd == 3 && sh[2] == 4 && fo == 0 && off == hl, "parse header");
    int64_t one[1] = {7};
    const int64_t h1 = tsr_npy_write_header(TSR_BOOL, 1, one, 1, hb, sizeof(hb));
    CHECK(tsr_npy_header(hb, h1, &dt, &nd, sh, &fo, &off) == 0 && dt == TSR_BOOL && nd == 1 && sh[0] == 7 && fo == 1, "1-D bool fortran");
    CHECK(tsr_npy_write_header(TSR_F64, 0, NULL, 0, NULL, 0) == 128, "size query");
    const char *np = "\x93NUMPY\x01\x00\x46\x00{'descr': '<f8', 'fortran_order': False, 'shape': (), }                      \n";
    CHECK(tsr_npy_header(np, 80, &dt, &nd, sh, &fo, &off) == 0 && nd == 0 && off == 80, "numpy 0-d header");
    const char *be = "\x93NUMPY\x01\x00\x46\x00{'descr': '>f8', 'fortran_order': False, 'shape': (), }                      \n";
    CHECK(tsr_npy_header(be, 80, &dt, &nd, sh, &fo, &off) == TSR_ETYPE, "big-endian refused");
    const char *bad = "\x93NUMPY\x01\x00\x46\x00{'descr': '<f8', 'fortran_order': False, 'shape': (1,}                      \n";
    CHECK(tsr_npy_header(bad, 80, &dt, &nd, sh, &fo, &off) == TSR_EARG, "malformed");
    CHECK(tsr_npy_header(np, 40, &dt, &nd, sh, &fo, &off) == TSR_EARG, "truncated");
}

static void test_slice_blanks(void)
{
    /* blank items and blanks around commas (fuzzer finding: a blank item read past its comma) */
    double buf[12];
    int64_t sh[2] = {3, 4};
    tsr_array m, v;
    tsr_array_init(&m, TSR_F64, 2, sh, buf);
    CHECK(tsr_array_slice(&m, "1 , 2", &v) == 0 && v.ndim == 0 && v.offset == 48, "blanks around comma");
    CHECK(tsr_array_slice(&m, " 1:3 , ::2 ", &v) == 0 && v.shape[0] == 2 && v.shape[1] == 2, "blanks in slices");
    CHECK(tsr_array_slice(&m, "0, ,1", &v) == TSR_EARG, "blank item refused");
    CHECK(tsr_array_slice(&m, "0,   ", &v) == TSR_EARG, "trailing blank item refused");
    CHECK(tsr_array_slice(&m, "  ", &v) == TSR_EARG, "blank spec refused");
    /* fuzzer finding: huge steps overflowed count and stride arithmetic */
    CHECK(tsr_array_slice(&m, "::4611686018427387904", &v) == 0 && v.shape[0] == 1 && v.shape[1] == 4, "huge step");
    CHECK(tsr_array_slice(&m, "::-4611686018427387904, 1::4611686018427387904", &v) == 0 && v.shape[0] == 1 && v.shape[1] == 1
          && v.offset == 2 * 32 + 8, "huge negative step");
}

/* double-double arithmetic for an exp/log reference that does not depend on long double: long double is plain
   double on Apple ARM64 and MSVC, and Valgrind evaluates x87 code at double precision, so there a long double
   "reference" is itself rounded to double and every result looks exactly 0 or 1 ulp off */
typedef struct { double hi, lo; } dd_t;
static dd_t dd_two_sum(double a, double b) { const double s = a + b, bb = s - a; return (dd_t){s, (a - (s - bb)) + (b - bb)}; }
static dd_t dd_add(dd_t a, dd_t b) { const dd_t s = dd_two_sum(a.hi, b.hi); return dd_two_sum(s.hi, s.lo + a.lo + b.lo); }
static dd_t dd_mul(dd_t a, dd_t b)
{
    const double p = a.hi * b.hi, e = fma(a.hi, b.hi, -p) + (a.hi * b.lo + a.lo * b.hi);
    return dd_two_sum(p, e);
}
static dd_t dd_div_d(dd_t a, double d) { const double q = a.hi / d; return dd_two_sum(q, (fma(-q, d, a.hi) + a.lo) / d); }
/* e^x = 2^k * m, m a double-double near [0.7, 1.42] (relative error about 2^-100): x = k ln2 + r, Taylor in r */
static dd_t dd_exp(double x, int *k)
{
    const double ln2_hi = 0x1.62e42fefa39efp-1, ln2_lo = 0x1.abc9e3b39803fp-56;
    const double kd = nearbyint(x / ln2_hi);
    const double p = kd * ln2_hi, pe = fma(kd, ln2_hi, -p), q = kd * ln2_lo, qe = fma(kd, ln2_lo, -q);
    dd_t r = dd_add(dd_two_sum(x, -p), (dd_t){-pe, 0});
    r = dd_add(r, (dd_t){-q, -qe});
    dd_t s = {1, 0};
    for (int n = 27; n >= 1; n--) s = dd_add((dd_t){1, 0}, dd_div_d(dd_mul(r, s), n));
    *k = (int)kd;
    return s;
}
/* error of e = exp(x) in ulps of e, against dd_exp (e normal) */
static double dd_exp_err(double x, double e)
{
    int k;
    const dd_t m = dd_exp(x, &k);
    const double u = ldexp(nextafter(e, INFINITY) - e, -k);
    return fabs((ldexp(e, -k) - m.hi) - m.lo) / u;
}
/* error of l = log(y) in ulps of l: one Newton step on e^t = y from libm's log, with e^t in double-double */
static double dd_log_err(double y, double l)
{
    const double t0 = log(y);
    int k;
    const dd_t m = dd_exp(t0, &k);
    const double ys = ldexp(y, -k), c = ((ys - m.hi) - m.lo) / m.hi;   /* exact scaling; ys - m.hi is exact */
    const double u = nextafter(fabs(l), INFINITY) - fabs(l);
    return fabs((l - t0) - c) / u;
}
/* 1 when long double arithmetic really carries more precision than double, as evaluated here */
static int long_double_is_wider(void)
{
    volatile long double one = 1.0L, tiny = 0x1p-60L;
    return one + tiny != one;
}

/* tsr_fmaf_ro (fmaf_ro.h) is the correctly rounded fmaf used where the C runtime's rounds twice: check it against
   fmaf wherever fmaf is the correct one, on inputs built so the exact a*b + c sits next to a float halfway point,
   which is where a second rounding goes wrong (the naive (float)(double sum) must be caught out at least once) */
static void test_fmaf_ro(void)
{
    uint64_t st = 0x9E3779B97F4A7C15ull;
    int bad = 0, naive_wrong = 0;
    for (int i = 0; i < 400000; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        const uint32_t uc = ((uint32_t)(127 + (int)((st >> 8) % 41) - 20) << 23) | (uint32_t)(st & 0x7FFFFFu) | ((uint32_t)(st >> 63) << 31);
        float c;
        memcpy(&c, &uc, 4);
        /* a*b is +-ulp(c)/2 up to the rounding of b: the exact sum lies just off a float midpoint */
        const double ulp = (double)nextafterf(fabsf(c), INFINITY) - (double)fabsf(c);
        const float a = (float)(1.0 + ldexp((double)((st >> 20) & 0xFFFFu), -23));
        float b = (float)(ulp / 2 / a);
        if ((st >> 40) & 1u) b = -b;
        const double p = (double)a * (double)b;
        const float want = fmaf(a, b, c), got = tsr_fmaf_ro(a, b, c), naive = (float)(p + (double)c);
        if (memcmp(&want, &got, 4) != 0 && !(isnan(want) && isnan(got))) bad++;
        if (memcmp(&want, &naive, 4) != 0) naive_wrong++;
    }
    const float z = tsr_fmaf_ro(-0.0f, 1.0f, 0.0f), nz = tsr_fmaf_ro(-0.0f, 1.0f, -0.0f);
    if (!TSR_FMAF_IS_RO) {   /* fmaf is the C library's: the reference */
        CHECK(bad == 0, "tsr_fmaf_ro == fmaf on near-halfway inputs (%d differ)", bad);
        CHECK(naive_wrong > 0, "the near-halfway inputs do catch a double-rounding fmaf (%d)", naive_wrong);
    }
    CHECK(z == 0 && !signbit(z) && nz == 0 && signbit(nz), "tsr_fmaf_ro signed zeros");
    CHECK(isinf(tsr_fmaf_ro(INFINITY, 2.0f, 1.0f)) && isnan(tsr_fmaf_ro(INFINITY, 0.0f, 1.0f)), "tsr_fmaf_ro inf and nan");
    CHECK(tsr_fmaf_ro(0x1.000002p0f, 0x1.000002p0f, -0x1.000004p0f) == 0x1p-46f, "tsr_fmaf_ro keeps the low product bits");
    /* double fma: the product of 1 + 2^-52 with itself needs 106 bits; an fma built on a 64-bit-mantissa long
       double loses the 2^-104 term and returns 0 */
    volatile double d = 0x1.0000000000001p0, e = -0x1.0000000000002p0;
    CHECK(fma(d, d, e) == 0x1p-104, "fma is correctly rounded (%a)", fma(d, d, e));
}

static void test_vmath(void)
{
    /* vectorised exp/log (vmath.h): < 1 ulp, against long double references where long double is wider, and
       always against the double-double reference (which must agree with long double where both exist) */
    const int wide = long_double_is_wider();
    double dd_e = 0, dd_l = 0, agree = 0;
    double worst_e = 0, worst_l = 0;
    for (int i = 0; i < 200000; i++) {
        const double x = -745.0 + 1455.0 * i / 200000.0 + 1e-7 * (i % 13);
        const double e = tsr_exp(x);
        const long double re = expl((long double)x);
        if (isfinite(e) && e > 0x1p-1022) {
            const double u = nextafter(e, INFINITY) - e;
            const double err = (double)(fabsl((long double)e - re) / u), derr = dd_exp_err(x, e);
            if (err > worst_e) worst_e = err;
            if (derr > dd_e) dd_e = derr;
            if (wide && fabs(err - derr) > agree) agree = fabs(err - derr);
        }
        const double y = ldexp(1.0 + (i % 997) / 997.0, (i % 2098) - 1074);
        const double l = tsr_log(y);
        const long double rl = logl((long double)y);
        if (l != 0) {
            const double u = nextafter(fabs(l), INFINITY) - fabs(l);
            const double err = (double)(fabsl((long double)l - rl) / u), derr = dd_log_err(y, l);
            if (err > worst_l) worst_l = err;
            if (derr > dd_l) dd_l = derr;
            if (wide && fabs(err - derr) > agree) agree = fabs(err - derr);
        }
    }
    CHECK(dd_e < 1.0 && dd_l < 1.0, "exp/log max error %.3f / %.3f ulp (double-double reference)", dd_e, dd_l);
    if (wide) {
        CHECK(worst_e < 1.0 && worst_l < 1.0, "exp/log max error %.3f / %.3f ulp (long double reference)", worst_e, worst_l);
        CHECK(agree < 0.01, "double-double and long double references agree (max difference %.2g ulp)", agree);
    }
    CHECK(tsr_exp(0) == 1 && tsr_exp(-INFINITY) == 0 && isinf(tsr_exp(INFINITY)) && isnan(tsr_exp(NAN)), "exp specials");
    CHECK(isinf(tsr_exp(709.79)) && tsr_exp(-745.14) == 0 && tsr_exp(-745.13) > 0, "exp overflow/underflow");
    CHECK(tsr_log(1) == 0 && isinf(tsr_log(0)) && tsr_log(0) < 0 && isnan(tsr_log(-1)) && isinf(tsr_log(INFINITY)), "log specials");
    CHECK(tsr_log(4.9406564584124654e-324) == log(4.9406564584124654e-324), "log of the smallest subnormal");
    CHECK(tsr_expf(1.0f) == expf(1.0f) && tsr_logf(10.0f) == logf(10.0f), "float32 variants");
    /* the kernels use them: a contiguous and a strided run give the same values */
    double in[8] = {-1, 0, 0.5, 1, 2, 10, 100, -700}, o1[8], o2[4];
    int64_t n8[1] = {8}, n4[1] = {4}, c8[1] = {8}, s16[1] = {16};
    tsr_unary(U_EXP, TSR_F64, 1, n8, in, c8, o1, c8);
    tsr_unary(U_EXP, TSR_F64, 1, n4, in, s16, o2, c8);
    CHECK(o1[0] == tsr_exp(-1) && o1[7] == tsr_exp(-700) && o2[1] == o1[2] && o2[3] == o1[6], "tsr_unary uses vmath");
}

static void test_rfft(void)
{
    const int64_t sizes[] = {1, 2, 3, 4, 5, 6, 8, 12, 15, 16, 30, 97, 128, 1000};
    double worst = 0, worst_inv = 0;
    for (size_t si = 0; si < sizeof(sizes) / sizeof(sizes[0]); si++) {
        const int64_t n = sizes[si], h = n / 2 + 1, rows = 3;
        double *x = malloc(sizeof(double) * n * rows), *c = malloc(sizeof(double) * 2 * n * rows);
        double *X = malloc(sizeof(double) * 2 * h * rows), *back = malloc(sizeof(double) * n * rows);
        for (int64_t i = 0; i < n * rows; i++) { x[i] = sin(0.7 * i) + 0.1 * (i % 5); c[2 * i] = x[i]; c[2 * i + 1] = 0; }
        tsr_fft(n, rows, 0, c, c);
        CHECK(tsr_rfft(n, rows, x, X) == 0, "rfft rc n=%lld", (long long)n);
        for (int64_t r = 0; r < rows; r++)
            for (int64_t k = 0; k < h; k++) {
                const double d = hypot(X[2 * (h * r + k)] - c[2 * (n * r + k)], X[2 * (h * r + k) + 1] - c[2 * (n * r + k) + 1]);
                if (d > worst) worst = d;
            }
        CHECK(tsr_irfft(n, rows, X, back) == 0, "irfft rc");
        for (int64_t i = 0; i < n * rows; i++) if (fabs(back[i] - x[i]) > worst_inv) worst_inv = fabs(back[i] - x[i]);
        free(x); free(c); free(X); free(back);
    }
    CHECK(worst < 1e-11 && worst_inv < 1e-12, "rfft matches the complex FFT (%.2g) and inverts (%.2g)", worst, worst_inv);
}

/* Regressions for the 2026-10 security review: each input used to crash the process, write out of bounds or
   hit undefined behaviour. They run under ASan + UBSan in `make sanitize`. */
static tsr_arg arg_str(const char *s) { tsr_arg a; memset(&a, 0, sizeof a); a.kind = 2; a.str = s; return a; }
static tsr_arg arg_num(double v) { tsr_arg a; memset(&a, 0, sizeof a); a.kind = 1; a.num = v; return a; }
static tsr_arg arg_arr(double *d, int32_t nd, const int64_t *shape)
{
    tsr_arg a; memset(&a, 0, sizeof a); a.kind = 3; tsr_array_init(&a.arr, TSR_F64, nd, shape, d); return a;
}

static void test_security_regressions(void)
{
    int64_t n2[1] = {2}, c8[1] = {8}, c4[1] = {4};
    /* INT_MIN // -1 and INT_MIN % -1 trapped with SIGFPE; NumPy wraps the quotient and gives remainder 0 */
    int64_t x64[2] = {INT64_MIN, 7}, m1[2] = {-1, -1}, q64[2], r64[2];
    CHECK(tsr_binary(OP_FLOORDIV, TSR_I64, 1, n2, x64, c8, m1, c8, q64, c8) == 0 && q64[0] == INT64_MIN && q64[1] == -7,
          "int64 MIN // -1 wraps");
    CHECK(tsr_binary(OP_MOD, TSR_I64, 1, n2, x64, c8, m1, c8, r64, c8) == 0 && r64[0] == 0 && r64[1] == 0, "int64 MIN %% -1 == 0");
    int32_t x32[2] = {INT32_MIN, 9}, m32[2] = {-1, -1}, q32[2];
    CHECK(tsr_binary(OP_FLOORDIV, TSR_I32, 1, n2, x32, c4, m32, c4, q32, c4) == 0 && q32[0] == INT32_MIN && q32[1] == -9,
          "int32 MIN // -1 wraps");
    /* negation, abs and square of integers wrap like NumPy instead of overflowing (UB) */
    int64_t neg[2], ab[2], sq[2], big[2] = {INT64_MIN, 3037000500LL};
    tsr_unary(U_NEG, TSR_I64, 1, n2, big, c8, neg, c8);
    tsr_unary(U_ABS, TSR_I64, 1, n2, big, c8, ab, c8);
    tsr_unary(U_SQUARE, TSR_I64, 1, n2, big, c8, sq, c8);
    CHECK(neg[0] == INT64_MIN && ab[0] == INT64_MIN && ab[1] == 3037000500LL, "int64 neg/abs wrap");
    CHECK(sq[1] == (int64_t)((uint64_t)3037000500LL * (uint64_t)3037000500LL), "int64 square wraps");

    /* einsum: a long or repeated explicit output overflowed fixed stack arrays */
    double v[3] = {1, 2, 3};
    int64_t sh3[1] = {3};
    char sub[300] = "a->";
    for (int i = 3; i < 280; i++) sub[i] = 'a';
    sub[280] = 0;
    int id = tsr_fn_find("np.einsum");
    tsr_arg ea[2] = {arg_str(sub), arg_arr(v, 1, sh3)};
    tsr_result er[1];
    CHECK(id >= 0 && tsr_fn_routine(id, 2, ea, 1, er) == TSR_EARG, "einsum long output refused");
    tsr_arg eb[2] = {arg_str("a->aa"), arg_arr(v, 1, sh3)};
    CHECK(tsr_fn_routine(id, 2, eb, 1, er) == TSR_EARG, "einsum repeated output label refused");
    tsr_arg ec[2] = {arg_str("a->a"), arg_arr(v, 1, sh3)};
    if (tsr_fn_routine(id, 2, ec, 1, er) == 0) {
        CHECK(((double *)er[0].arr.data)[2] == 3, "einsum identity still works");
        tsr_free(er[0].arr.data, er[0].bytes);
    } else CHECK(0, "einsum identity failed: %s", tsr_fn_error());
    /* a zero factor must not hide a NaN of a later operand (numpy: 0 * nan = nan); and a matrix product through the
       strided inner loop equals the plain triple loop, bit for bit (same summation order) */
    {
        double z2[2] = {0.0, 1.0}, n2[2] = {NAN, 2.0};
        int64_t s2[1] = {2};
        tsr_arg en[3] = {arg_str("i,i"), arg_arr(z2, 1, s2), arg_arr(n2, 1, s2)};
        tsr_result eo[1];
        const int rcn = tsr_fn_routine(id, 3, en, 1, eo);
        CHECK(rcn == 0 && isnan(eo[0].kind == 1 ? eo[0].num : ((double *)eo[0].arr.data)[0]), "einsum('i,i', [0, 1], [nan, 2]) is nan");
        if (rcn == 0 && eo[0].kind == 3) tsr_free(eo[0].arr.data, eo[0].bytes);
        enum { EM = 7, EK = 9, EN = 5 };
        double ea2[EM * EK], eb2[EK * EN], ref[EM * EN];
        for (int i = 0; i < EM * EK; i++) ea2[i] = sin(1.0 + i) * 3.0;
        for (int i = 0; i < EK * EN; i++) eb2[i] = cos(2.0 + i) * 5.0;
        for (int i = 0; i < EM; i++) for (int k = 0; k < EN; k++) { double s = 0.0; for (int j = 0; j < EK; j++) s += ea2[i * EK + j] * eb2[j * EN + k]; ref[i * EN + k] = s; }
        int64_t sa2[2] = {EM, EK}, sb2[2] = {EK, EN};
        tsr_arg em[3] = {arg_str("ij,jk->ik"), arg_arr(ea2, 2, sa2), arg_arr(eb2, 2, sb2)};
        const int rcm = tsr_fn_routine(id, 3, em, 1, eo);
        CHECK(rcm == 0 && !memcmp(eo[0].arr.data, ref, sizeof ref), "einsum ij,jk->ik == the ascending-j triple loop, bit for bit");
        if (rcm == 0) tsr_free(eo[0].arr.data, eo[0].bytes);
    }

    /* histogramdd: 32 dimensions of 2^20 bins wrapped the bin-count product and under-allocated the counts */
    double sample[32];
    for (int i = 0; i < 32; i++) sample[i] = i;
    int64_t sh32[2] = {1, 32};
    int hd = tsr_fn_find("np.histogramdd");
    tsr_arg ha[2] = {arg_arr(sample, 2, sh32), arg_num(1048576.0)};
    tsr_result hr[2];
    CHECK(hd >= 0 && tsr_fn_routine(hd, 2, ha, 2, hr) == TSR_EARG, "histogramdd bin-count overflow refused");
    tsr_arg hb[2] = {arg_arr(sample, 2, sh32), arg_num(NAN)};
    CHECK(tsr_fn_routine(hd, 2, hb, 2, hr) == TSR_EARG, "histogramdd NaN bins refused");
    int h2 = tsr_fn_find("np.histogram2d");
    tsr_arg h2a[3] = {arg_arr(v, 1, sh3), arg_arr(v, 1, sh3), arg_num(4e9)};
    tsr_result h2r[3];
    CHECK(h2 >= 0 && tsr_fn_routine(h2, 3, h2a, 3, h2r) == TSR_EARG, "histogram2d bin-count overflow refused");

    /* a routine called with fewer arguments than it declares must not read past the argument array */
    tsr_arg one[1] = {arg_arr(sample, 2, sh32)};
    const int rc = tsr_fn_routine(hd, 1, one, 2, hr);
    if (rc == 0) { tsr_result *e = (tsr_result *)hr[1].arr.data;
                   tsr_free(hr[0].arr.data, hr[0].bytes);
                   for (int64_t k = 0; k < hr[1].arr.shape[0]; k++) tsr_free(e[k].arr.data, e[k].bytes);
                   tsr_free(hr[1].arr.data, hr[1].bytes); }
    CHECK(rc == 0 || rc == TSR_EARG, "short argument list handled (rc %d)", rc);

    /* CSR kernels refuse an indptr that decreases or does not start at 0 */
    int64_t bad_ip[3] = {0, 5, 2}, neg_ip[3] = {-4, 1, 2}, ix[2] = {0, 1};
    double dd[2] = {1, 1}, xx[2] = {1, 1}, yy[2];
    CHECK(tsr_csr_matvec(2, bad_ip, ix, dd, xx, yy) == TSR_EINDEX, "csr matvec decreasing indptr refused");
    CHECK(tsr_csr_matvec(2, neg_ip, ix, dd, xx, yy) == TSR_EINDEX, "csr matvec negative indptr refused");
    int64_t ok_ip[3] = {0, 1, 2}, far_ix[2] = {0, 9};
    double res = 0;
    CHECK(tsr_csr_cg(2, ok_ip, far_ix, dd, xx, yy, 1e-10, 0, 10, &res) == TSR_EINDEX, "csr cg column outside matrix refused");
}

/* Regressions for the argument-validation findings of the fuzzing campaign (csrc/fuzz/README.md, batch A): each
   input used to read or write out of bounds, hit undefined behaviour (an unchecked float->int cast, a signed
   overflow) or run unbounded. They run under ASan + UBSan in `make sanitize`. */
static tsr_arg arg_int(int64_t v) { tsr_arg a = arg_num((double)v); a.flags = 1; a.ival = v; return a; }
static tsr_arg arg_none(void) { tsr_arg a; memset(&a, 0, sizeof a); return a; }
static void fa_free(tsr_result *r, int n)
{
    for (int k = 0; k < n; k++)
        if ((r[k].kind == 3 || r[k].kind == 2) && r[k].arr.data) tsr_free(r[k].arr.data, r[k].bytes);
}
/* call routine `name`; results stay in r (nres slots, zeroed first) for the caller to inspect and fa_free */
static int fa_call(const char *name, int nargs, const tsr_arg *a, tsr_result *r, int nres)
{
    memset(r, 0, sizeof(tsr_result) * (size_t)nres);
    const int id = tsr_fn_find(name);
    if (id < 0) { fprintf(stderr, "no routine %s\n", name); return -999; }
    return tsr_fn_routine(id, nargs, a, nres, r);
}
static double fa_num(const tsr_result *r) { return r->kind == 3 ? ((const double *)r->arr.data)[0] : r->num; }

static void test_fuzz_batch_a(void)
{
    tsr_result r[8];
    int rc;
    double e0[1] = {0}, one[1] = {1.0}, three[3] = {1, 2, 3}, five[1] = {5.0}, nanv[2] = {NAN, 2.0};
    int64_t s0[1] = {0}, s1[1] = {1}, s2[1] = {2}, s3[1] = {3};

    /* F2: ellip_harm / _2 / ellip_normal with p < 1 indexed z[-1]; SciPy returns NaN */
    tsr_arg eh[5] = {arg_num(5), arg_num(8), arg_int(1), arg_int(0), arg_num(2.5)};
    rc = fa_call("special.ellip_harm", 5, eh, r, 1);
    CHECK(rc == 0 && isnan(r[0].num), "F2 ellip_harm(p=0) is NaN (rc %d)", rc);
    rc = fa_call("special.ellip_harm_2", 5, eh, r, 1);
    CHECK(rc == 0 && isnan(r[0].num), "F2 ellip_harm_2(p=0) is NaN (rc %d)", rc);
    rc = fa_call("special.ellip_normal", 4, eh, r, 1);
    CHECK(rc == 0 && isnan(r[0].num), "F2 ellip_normal(p=0) is NaN (rc %d)", rc);
    eh[3] = arg_num(NAN);
    rc = fa_call("special.ellip_harm", 5, eh, r, 1);
    CHECK(rc == 0 && isnan(r[0].num), "F2 ellip_harm(p=NaN) is NaN (rc %d)", rc);
    eh[3] = arg_int(1);
    rc = fa_call("special.ellip_harm", 5, eh, r, 1);
    CHECK(rc == 0 && isfinite(r[0].num), "F2 ellip_harm(p=1) still evaluates (rc %d)", rc);

    /* F5: a label on a 0-length axis and a longer axis was accepted (0 meant "unset") and read out of bounds */
    tsr_arg ei[3] = {arg_str("i,i"), arg_arr(e0, 1, s0), arg_arr(three, 1, s3)};
    CHECK(fa_call("np.einsum", 3, ei, r, 1) == TSR_EARG, "F5 einsum 0-length vs 3 refused");
    ei[2] = arg_arr(one, 1, s1);
    rc = fa_call("np.einsum", 3, ei, r, 1);
    CHECK(rc == 0 && fa_num(&r[0]) == 0.0, "F5 einsum 0-length vs 1 broadcasts to 0 (rc %d)", rc);
    fa_free(r, 1);
    tsr_arg eo[3] = {arg_str("i,j->ij"), arg_arr(e0, 1, s0), arg_arr(three, 1, s3)};
    rc = fa_call("np.einsum", 3, eo, r, 1);
    CHECK(rc == 0 && r[0].kind == 3 && r[0].arr.shape[0] == 0 && r[0].arr.shape[1] == 3, "F5 einsum outer with a 0-length axis (rc %d)", rc);
    fa_free(r, 1);
    /* F8: dots that are not one "..." per term were misparsed (olab left unset), and a short output "." read past
       the string */
    double cube[8] = {0}; int64_t s222[3] = {2, 2, 2};
    tsr_arg ed[2] = {arg_str("a.b.c"), arg_arr(cube, 3, s222)};
    CHECK(fa_call("np.einsum", 2, ed, r, 1) == TSR_EARG, "F8 einsum 'a.b.c' refused");
    ed[0] = arg_str("..a.");
    CHECK(fa_call("np.einsum", 2, ed, r, 1) == TSR_EARG, "F8 einsum '..a.' refused");
    tsr_arg eu[2] = {arg_str("a->."), arg_arr(three, 1, s3)};
    CHECK(fa_call("np.einsum", 2, eu, r, 1) == TSR_EARG, "F8 einsum output '.' refused");
    eu[0] = arg_str("...a->....");
    CHECK(fa_call("np.einsum", 2, eu, r, 1) == TSR_EARG, "F8 einsum output '....' refused");
    eu[0] = arg_str("...a->...");
    rc = fa_call("np.einsum", 2, eu, r, 1);
    CHECK(rc == 0 && fa_num(&r[0]) == 6.0, "einsum '...a->...' still sums (rc %d)", rc);
    fa_free(r, 1);

    /* F23: an order ~1e18 indexed the fixed [256] coefficient buffer; SciPy returns at most 251 coefficients */
    tsr_arg mc[2] = {arg_num(1e18), arg_num(2.0)};
    CHECK(fa_call("special.mathieu_odd_coef", 2, mc, r, 1) == TSR_EARG, "F23 mathieu_odd_coef(1e18) refused (int overflow)");
    mc[0] = arg_int(1000); mc[1] = arg_num(1.0);
    rc = fa_call("special.mathieu_even_coef", 2, mc, r, 1);
    CHECK(rc == 0 && r[0].arr.shape[0] == 251, "F23 mathieu_even_coef(1000) has 251 coefficients (rc %d)", rc);
    fa_free(r, 1);
    mc[1] = arg_num(NAN);
    CHECK(fa_call("special.mathieu_even_coef", 2, mc, r, 1) == TSR_EARG, "F23 mathieu_even_coef(q=NaN) refused");

    /* F20: pbvv_seq repeated the recurrence per order (quadratic: 2^20 ran for minutes); v=1e300 cast to int */
    tsr_arg pv[2] = {arg_num(1048576.0), arg_num(1.0)};
    rc = fa_call("special.pbvv_seq", 2, pv, r, 2);
    CHECK(rc == 0 && r[0].arr.shape[0] == 1048577, "F20 pbvv_seq(2^20) in one recurrence (rc %d)", rc);
    fa_free(r, 2);
    pv[0] = arg_num(1e300);
    CHECK(fa_call("special.pbvv_seq", 2, pv, r, 2) == TSR_EARG, "F20 pbvv_seq(1e300) refused");
    pv[0] = arg_num(NAN);
    CHECK(fa_call("special.pbdv_seq", 2, pv, r, 2) == TSR_EARG, "F20 pbdv_seq(NaN) refused");
    pv[0] = arg_num(-5.5);
    rc = fa_call("special.pbvv_seq", 2, pv, r, 2);
    CHECK(rc == 0 && r[0].arr.shape[0] == 2 && CLOSE(((double *)r[0].arr.data)[0], 0.22003509, 1e-7) &&
          CLOSE(((double *)r[0].arr.data)[1], -0.54176921, 1e-7), "pbvv_seq(-5.5, 1) matches scipy (rc %d)", rc);
    fa_free(r, 2);
    /* the fixed [260] zero buffer of erf_zeros / fresnel*_zeros overflowed for nt > 260 */
    tsr_arg ez[1] = {arg_int(300)};
    rc = fa_call("special.erf_zeros", 1, ez, r, 1);
    CHECK(rc == 0 && r[0].arr.shape[0] == 300, "erf_zeros(300) (rc %d)", rc);
    fa_free(r, 1);
    tsr_arg nc[1] = {arg_int(10000000)};
    CHECK(fa_call("np.newton_cotes", 1, nc, r, 1) == TSR_EARG, "newton_cotes(1e7) refused before the (rn+1)^2 allocation");

    /* F13: an empty sample made stat_osm offset a null pointer; N is an integer >= 0 as numpy.linspace needs */
    tsr_arg bn[4] = {arg_arr(e0, 1, s0), arg_num(-2), arg_num(2), arg_int(3)};
    rc = fa_call("stats.boxcox_normplot", 4, bn, r, 2);
    CHECK(rc == 0 && r[0].arr.shape[0] == 0, "F13 boxcox_normplot([]) is empty (rc %d)", rc);
    fa_free(r, 2);
    rc = fa_call("stats.yeojohnson_normplot", 4, bn, r, 2);
    CHECK(rc == 0 && r[0].arr.shape[0] == 0, "F13 yeojohnson_normplot([]) is empty (rc %d)", rc);
    fa_free(r, 2);
    rc = fa_call("stats.probplot", 1, bn, r, 5);
    CHECK(rc == 0 && r[0].arr.shape[0] == 0 && r[4].num == 0.0 && isnan(r[2].num), "F13 probplot([]) is (nan, nan, 0) (rc %d)", rc);
    fa_free(r, 5);
    bn[0] = arg_arr(three, 1, s3); bn[3] = arg_num(NAN);
    CHECK(fa_call("stats.boxcox_normplot", 4, bn, r, 2) == TSR_ETYPE, "F13 boxcox_normplot N=NaN refused");
    bn[3] = arg_int(-1);
    CHECK(fa_call("stats.boxcox_normplot", 4, bn, r, 2) == TSR_EARG, "F13 boxcox_normplot N=-1 refused");

    /* F18: ecdf never advanced past a NaN (a 2-D fuzz input ran for minutes); SciPy wants 1-D, NaN-free */
    double sq4[4] = {1, 2, 3, 4}; int64_t s22[2] = {2, 2};
    tsr_arg ec[1] = {arg_arr(sq4, 2, s22)};
    CHECK(fa_call("stats.ecdf", 1, ec, r, 2) == TSR_EARG, "F18 ecdf 2-D refused");
    ec[0] = arg_arr(nanv, 1, s2);
    CHECK(fa_call("stats.ecdf", 1, ec, r, 2) == TSR_EARG, "F18 ecdf NaN refused");

    /* F34: f_obs of length 1 against an empty f_exp read the empty side; NumPy broadcasting gives length 0 */
    tsr_arg cs[4] = {arg_arr(five, 1, s1), arg_arr(e0, 1, s0), arg_int(0), arg_none()};
    rc = fa_call("stats.chisquare", 4, cs, r, 2);
    CHECK(rc == 0 && fa_num(&r[0]) == 0.0 && isnan(fa_num(&r[1])), "F34 chisquare([5], f_exp=[]) is (0, nan) (rc %d)", rc);
    fa_free(r, 2);

    /* F15 / F36 / F40: a 0-d input indexed shape[-1] / stride[-1] */
    double sc3[1] = {3.0};
    tsr_arg mf[2] = {arg_arr(sc3, 0, NULL), arg_int(1)};
    rc = fa_call("ndimage.maximum_filter", 2, mf, r, 1);
    CHECK(rc == 0 && r[0].arr.ndim == 0 && fa_num(&r[0]) == 3.0, "F15 maximum_filter of a 0-d input (rc %d)", rc);
    fa_free(r, 1);
    mf[1] = arg_int(3);
    rc = fa_call("ndimage.morphological_laplace", 2, mf, r, 1);
    CHECK(rc == 0 && fa_num(&r[0]) == 0.0, "F15 morphological_laplace of a 0-d input (rc %d)", rc);
    fa_free(r, 1);
    tsr_arg mk[2] = {arg_arr(sc3, 0, NULL), arg_arr(sc3, 0, NULL)};
    rc = fa_call("distance.minkowski_distance", 2, mk, r, 1);
    CHECK(rc == 0 && fa_num(&r[0]) == 0.0, "F36 minkowski_distance of 0-d inputs (rc %d)", rc);
    double ot[2] = {1.0, 2.0}, oz[1] = {1.0};
    tsr_arg mk2[2] = {arg_arr(oz, 0, NULL), arg_arr(ot, 1, s2)};
    rc = fa_call("distance.minkowski_distance", 2, mk2, r, 1);
    CHECK(rc == 0 && fa_num(&r[0]) == 1.0, "F36 minkowski_distance 0-d against (2,) broadcasts (rc %d)", rc);
    double m23[6] = {0}; int64_t s23[2] = {2, 3}, s33[2] = {3, 3};
    double m33[9] = {0};
    tsr_arg mk3[2] = {arg_arr(m23, 2, s23), arg_arr(m33, 2, s33)};
    CHECK(fa_call("distance.minkowski_distance", 2, mk3, r, 1) == TSR_EARG, "minkowski_distance (2,3) vs (3,3) refused (read past y)");
    tsr_arg fh[3] = {arg_arr(sc3, 0, NULL), arg_num(0.1), arg_num(0.0)};
    CHECK(fa_call("scipyfft.fht", 3, fh, r, 3) == TSR_EDIM, "F40 fht of a 0-d input refused");

    /* F11: a NaN label was cast to int64 (find_objects); it is background */
    tsr_arg fo[1] = {arg_arr(nanv, 1, s2)};
    rc = fa_call("ndimage.find_objects", 1, fo, r, 1);
    CHECK(rc == 0 && r[0].arr.shape[0] == 2 && ((double *)r[0].arr.data)[2] == 1.0 && ((double *)r[0].arr.data)[3] == 2.0,
          "F11 find_objects with a NaN label (rc %d)", rc);
    fa_free(r, 1);
    /* labels / masks shorter than the input were read past their end */
    tsr_arg ls[2] = {arg_arr(three, 1, s3), arg_arr(ot, 1, s2)};
    CHECK(fa_call("ndimage.sum", 2, ls, r, 1) == TSR_EARG, "ndimage.sum with short labels refused");
    tsr_arg bp[3] = {arg_arr(sq4, 2, s22), arg_none(), arg_arr(ot, 1, s2)};
    CHECK(fa_call("ndimage.binary_propagation", 3, bp, r, 1) == TSR_EARG, "binary_propagation with a short mask refused");
    double neg4[4] = {-5, 5, 0, 0}, mk4[4] = {1, 0, 0, 2};
    tsr_arg ws[2] = {arg_arr(neg4, 2, s22), arg_arr(mk4, 2, s22)};
    CHECK(fa_call("ndimage.watershed_ift", 2, ws, r, 1) == TSR_ETYPE, "watershed_ift with a negative input refused");
    tsr_arg g1[2] = {arg_arr(three, 1, s3), arg_num(NAN)};
    CHECK(fa_call("ndimage.gaussian_filter1d", 2, g1, r, 1) == TSR_EARG, "gaussian_filter1d sigma=NaN refused");
    tsr_arg u1[2] = {arg_arr(three, 1, s3), arg_int((int64_t)1 << 62)};
    CHECK(fa_call("ndimage.uniform_filter1d", 2, u1, r, 1) == TSR_EARG, "uniform_filter1d size 2^62 refused");
    tsr_arg mo[5] = {arg_arr(three, 1, s3), arg_int(3), arg_str("reflect"), arg_num(0), arg_num(9e18)};
    CHECK(fa_call("ndimage.minimum_filter", 5, mo, r, 1) == TSR_EARG, "minimum_filter origin 9e18 refused");
    tsr_arg zm[2] = {arg_arr(three, 1, s3), arg_num(NAN)};
    CHECK(fa_call("ndimage.zoom", 2, zm, r, 1) == TSR_EARG, "zoom NaN refused");
    double idm[1] = {1.0};
    tsr_arg af[4] = {arg_arr(three, 1, s3), arg_arr(idm, 1, s1), arg_num(0), arg_arr(ot, 1, s2)};
    CHECK(fa_call("ndimage.affine_transform", 4, af, r, 1) == TSR_EARG, "affine_transform output_shape of the wrong length refused");

    /* map_coordinates with (ndim,) coordinates: a scalar result whose 0-d block leaked */
    double mcoord[1] = {1.0};
    tsr_arg mcs[3] = {arg_arr(three, 1, s3), arg_arr(mcoord, 1, s1), arg_int(1)};
    rc = fa_call("ndimage.map_coordinates", 3, mcs, r, 1);
    CHECK(rc == 0 && r[0].kind == 1 && r[0].num == 2.0, "map_coordinates scalar result (rc %d, no 0-d block leaked)", rc);
    fa_free(r, 1);

    /* F7 / F9: histogram bin counts: (bins + 1) * 8 wrapped; an infinite estimator quotient was cast to int64 */
    tsr_arg hb[2] = {arg_arr(three, 1, s3), arg_int((int64_t)1 << 60)};
    CHECK(fa_call("np.histogram", 2, hb, r, 2) == TSR_EARG, "F7 histogram bins=2^60 refused");
    CHECK(fa_call("np.histogram_bin_edges", 2, hb, r, 1) == TSR_EARG, "F7 histogram_bin_edges bins=2^60 refused");
    double spread[8] = {-1e308, 0, 0, 0, 0, 1e308, 1, 2}; int64_t s8[1] = {8};
    tsr_arg he[2] = {arg_arr(spread, 1, s8), arg_str("fd")};
    CHECK(fa_call("np.histogram_bin_edges", 2, he, r, 1) == TSR_EARG, "F9 histogram 'fd' over an infinite spread refused");

    /* F31 / F33: sparse diagonal constructors built a dense m*n matrix (the product overflowed) */
    tsr_arg ey[1] = {arg_num(4e9)};
    CHECK(fa_call("sparse.eye", 1, ey, r, 4) == TSR_ENOMEM, "F31 sparse.eye(4e9) refused (int32 indices)");
    ey[0] = arg_num(NAN);
    CHECK(fa_call("sparse.eye", 1, ey, r, 4) == TSR_EARG, "F31 sparse.eye(NaN) refused");
    ey[0] = arg_int(100000);
    rc = fa_call("sparse.eye_array", 1, ey, r, 4);
    CHECK(rc == 0 && r[0].arr.shape[0] == 100000 && r[2].arr.shape[0] == 100001, "sparse.eye_array(1e5) without a dense build (rc %d)", rc);
    fa_free(r, 4);
    tsr_arg ek[3] = {arg_int(3), arg_none(), arg_int(1)};
    rc = fa_call("sparse.eye", 3, ek, r, 4);
    CHECK(rc == 0 && r[0].arr.shape[0] == 2 && ((int32_t *)r[1].arr.data)[0] == 1 && ((int32_t *)r[1].arr.data)[1] == 2 &&
          ((int32_t *)r[2].arr.data)[3] == 2, "sparse.eye(3, k=1) structure (rc %d)", rc);
    fa_free(r, 4);
    double d12[2] = {1, 2}, d34[2] = {3, 4}, big2[2] = {4e9, 4e9}, off0[1] = {0}, off00[2] = {0, 0}, offn[1] = {NAN};
    tsr_arg dl[2] = {arg_arr(d12, 1, s2), arg_arr(d34, 1, s2)};
    tsr_arg seq; memset(&seq, 0, sizeof seq); seq.kind = 5; seq.items = dl; seq.count = 1;
    tsr_arg dg[3] = {seq, arg_arr(off0, 1, s1), arg_arr(big2, 1, s2)};
    CHECK(fa_call("sparse.diags", 3, dg, r, 4) == TSR_ENOMEM, "F33 sparse.diags shape (4e9, 4e9) refused");
    dg[1] = arg_arr(offn, 1, s1); dg[2] = arg_none();
    CHECK(fa_call("sparse.diags", 3, dg, r, 4) == TSR_EARG, "F33 sparse.diags offset NaN refused");
    dg[1] = arg_arr(off00, 1, s2);
    CHECK(fa_call("sparse.diags", 2, dg, r, 4) == TSR_EARG, "sparse.diags with more offsets than diagonals refused");
    seq.count = 2; dg[0] = seq;
    rc = fa_call("sparse.diags", 2, dg, r, 4);
    CHECK(rc == 0 && r[0].arr.shape[0] == 2 && ((double *)r[0].arr.data)[0] == 4.0 && ((double *)r[0].arr.data)[1] == 6.0,
          "sparse.diags repeated offsets still sum (rc %d)", rc);
    fa_free(r, 4);
    double sd[6] = {1, 2, 3, 4, 5, 6};
    tsr_arg sp[4] = {arg_arr(sd, 2, s23), arg_arr(off0, 1, s1), arg_int(3), arg_int(3)};
    CHECK(fa_call("sparse.spdiags", 4, sp, r, 4) == TSR_EARG, "sparse.spdiags with fewer offsets than rows refused");

    /* F1 / F10 / F6: text and binary file readers */
    char path[1024];
    snprintf(path, sizeof path, "%s/tsr_fuzz_a_XXXXXX", test_tmpdir());   /* no /tmp on Windows */
    int fd = mkstemp(path);
    CHECK(fd >= 0, "mkstemp");
    if (fd >= 0) {
        const char txt[] = "1 300 -3 2.7";
        CHECK(write(fd, txt, sizeof txt - 1) == (ssize_t)(sizeof txt - 1), "write");
        close(fd);
        tsr_arg ff[4] = {arg_str(path), arg_str("uint8"), arg_int(3), arg_str(" ")};
        rc = fa_call("np.fromfile", 4, ff, r, 1);
        CHECK(rc == 0 && r[0].arr.shape[0] == 3 && ((uint8_t *)r[0].arr.data)[1] == 44 && ((uint8_t *)r[0].arr.data)[2] == 253,
              "F1 fromfile text uint8 parses integers and wraps (rc %d)", rc);
        fa_free(r, 1);
        ff[2] = arg_int(-1);
        CHECK(fa_call("np.fromfile", 4, ff, r, 1) == TSR_EARG, "F1 fromfile text int '2.7' refused (unmatched data)");
        tsr_arg fb[5] = {arg_str(path), arg_str("uint8"), arg_int(-1), arg_none(), arg_num(NAN)};
        CHECK(fa_call("np.fromfile", 5, fb, r, 1) == TSR_ETYPE, "F10 fromfile offset NaN refused");
        fb[4] = arg_int(INT64_MAX);
        CHECK(fa_call("np.fromfile", 5, fb, r, 1) == TSR_EARG, "F10 fromfile offset past the end refused");
        fb[4] = arg_int(-9223372036854775807LL);
        CHECK(fa_call("np.fromfile", 5, fb, r, 1) == TSR_EARG, "F10 fromfile negative offset refused");
        FILE *f = fopen(path, "w");
        if (f) fclose(f);                                  /* empty file */
        tsr_arg gf[1] = {arg_str(path)};
        rc = fa_call("np.genfromtxt", 1, gf, r, 1);
        CHECK(rc == 0, "F6 genfromtxt of an empty file (rc %d)", rc);
        fa_free(r, 1);
        rc = fa_call("np.loadtxt", 1, gf, r, 1);
        CHECK(rc == 0, "F6 loadtxt of an empty file (rc %d)", rc);
        fa_free(r, 1);
        unlink(path);
    }

    /* CI libFuzzer: np.linalg.svd of a matrix with NaN / infinity hung inside dgesdd; NumPy: "SVD did not converge" */
    double sv[4] = {1.0, NAN, 2.0, INFINITY};
    tsr_arg svd_a[1] = {arg_arr(sv, 2, s22)};
    CHECK(fa_call("linalg.svd", 1, svd_a, r, 3) == TSR_EARG, "svd of a non-finite matrix refused (no dgesdd hang)");
    CHECK(fa_call("slinalg.svd", 1, svd_a, r, 3) == TSR_EARG, "scipy svd of a non-finite matrix refused");
    double lb[2] = {1.0, 2.0};
    tsr_arg ls_a[2] = {arg_arr(sv, 2, s22), arg_arr(lb, 1, s2)};
    CHECK(fa_call("linalg.lstsq", 2, ls_a, r, 4) == TSR_EARG, "lstsq with a non-finite matrix refused (no dgelsd hang)");

    /* F4: the .npy writer refuses a header the reader refuses (byte count beyond int64) */
    int64_t hs1[1] = {INT64_MAX - 7}, hs2[2] = {(int64_t)1 << 62, 4}, hs3[2] = {(int64_t)1 << 60, 8};
    char hdr[256];
    CHECK(tsr_npy_write_header(TSR_F64, 1, hs1, 0, NULL, 0) < 0, "F4 npy header for a 9223372036854775800 dim refused");
    CHECK(tsr_npy_write_header(TSR_U8, 2, hs2, 0, hdr, sizeof hdr) < 0, "F4 npy header for 2^64 elements refused");
    CHECK(tsr_npy_write_header(TSR_F64, 2, hs3, 0, hdr, sizeof hdr) < 0, "F4 npy header for 2^66 bytes refused");
    int64_t ok2[2] = {(int64_t)1 << 30, 8};
    CHECK(tsr_npy_write_header(TSR_F64, 2, ok2, 0, hdr, sizeof hdr) > 0, "npy header for 2^33 elements still written");

    /* CI fuzz: sph_harm_y cast its orders to int unchecked, and xsf negated m = INT_MIN (signed overflow);
       lambertw cast its branch index unchecked. Orders out of reach give NaN; a non-finite k is refused. */
    tsr_arg sy[4] = {arg_num(-2147483648.0), arg_num(-2147483648.0), arg_num(0.5), arg_num(0.25)};
    rc = fa_call("special.sph_harm_y", 4, sy, r, 1);
    CHECK(rc == 0 && isnan(((double *)r[0].arr.data)[0]), "sph_harm_y with order INT_MIN is NaN (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    sy[0] = arg_num(NAN); sy[1] = arg_num(1e300);
    rc = fa_call("special.sph_harm_y", 4, sy, r, 1);
    CHECK(rc == 0 && isnan(((double *)r[0].arr.data)[0]), "sph_harm_y with NaN/huge orders is NaN (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    sy[0] = arg_int(2); sy[1] = arg_int(1);
    rc = fa_call("special.sph_harm_y", 4, sy, r, 1);
    CHECK(rc == 0 && !isnan(((double *)r[0].arr.data)[0]), "sph_harm_y(2, 1, ...) still computed (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    /* xsf's int recurrence coefficients overflow beyond |n| = 645 (CI fuzz, legendre.h:584): 645 computes, 646 is NaN */
    sy[0] = arg_int(645);
    rc = fa_call("special.sph_harm_y", 4, sy, r, 1);
    CHECK(rc == 0 && !isnan(((double *)r[0].arr.data)[0]), "sph_harm_y(645, 1, ...) computed (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    sy[0] = arg_int(646);
    rc = fa_call("special.sph_harm_y", 4, sy, r, 1);
    CHECK(rc == 0 && isnan(((double *)r[0].arr.data)[0]), "sph_harm_y(646, 1, ...) is NaN, not an int overflow (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    /* CI fuzz (np_shape.c:687): np.roll with a scalar shift and axis=() read axes[0] uninitialised and indexed with it.
       NumPy broadcasts the shift against zero axes: nothing rolls and the result is a copy */
    double rl[3] = {1, 2, 3};
    int64_t rls[1] = {3};
    tsr_arg noax; memset(&noax, 0, sizeof noax); noax.kind = 5; noax.count = 0;
    tsr_arg ra[3] = {arg_arr(rl, 1, rls), arg_int(1), noax};
    rc = fa_call("np.roll", 3, ra, r, 1);
    CHECK(rc == 0 && r[0].arr.shape[0] == 3 && ((double *)r[0].arr.data)[0] == 1 && ((double *)r[0].arr.data)[2] == 3,
          "np.roll(a, 1, axis=()) is an unrolled copy (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    /* CI fuzz timeout: invhilbert is O(n^3) (three O(n) binomials per entry); n beyond 700 is refused */
    tsr_arg ih[1] = {arg_int(701)};
    CHECK(fa_call("slinalg.invhilbert", 1, ih, r, 1) == TSR_EARG, "invhilbert(701) refused (work out of reach)");
    ih[0] = arg_int(3);
    rc = fa_call("slinalg.invhilbert", 1, ih, r, 1);
    CHECK(rc == 0 && r[0].arr.shape[0] == 3 && ((double *)r[0].arr.data)[0] == 9.0 && ((double *)r[0].arr.data)[1] == -36.0,
          "invhilbert(3) still exact: [[9, -36, 30], ...] (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    /* CI fuzz timeout (main): sparse.diags with a huge offset inferred a 2^30-row shape and scanned every row for
       one entry. Only the rows a diagonal crosses are scanned now; the empty rows still appear in indptr. */
    {
        double dv2[2] = {1, 2}, offm3[1] = {-3}, offbig[1] = {16777216.0};
        tsr_arg dsq; memset(&dsq, 0, sizeof dsq); dsq.kind = 5; dsq.items = dl; dsq.count = 1;
        dl[0] = arg_arr(dv2, 1, s2);
        tsr_arg dd[2] = {dsq, arg_arr(offm3, 1, s1)};
        rc = fa_call("sparse.diags", 2, dd, r, 4);
        const int32_t *ip = rc == 0 ? (const int32_t *)r[2].arr.data : NULL;
        CHECK(rc == 0 && r[2].arr.shape[0] == 6 && ip[0] == 0 && ip[3] == 0 && ip[4] == 1 && ip[5] == 2 &&
              ((int32_t *)r[1].arr.data)[0] == 0 && ((int32_t *)r[1].arr.data)[1] == 1,
              "sparse.diags([1, 2], -3): leading empty rows in indptr, entries at (3,0), (4,1) (rc %d)", rc);
        if (rc == 0) fa_free(r, 4);
        dd[1] = arg_arr(offbig, 1, s1);
        rc = fa_call("sparse.diags", 2, dd, r, 4);
        ip = rc == 0 ? (const int32_t *)r[2].arr.data : NULL;
        CHECK(rc == 0 && r[2].arr.shape[0] == 16777219 && ip[1] == 1 && ip[2] == 2 && ip[16777218] == 2 &&
              ((int32_t *)r[1].arr.data)[1] == 16777217,
              "sparse.diags([1, 2], 2^24): trailing empty rows in indptr, entries at (0,2^24), (1,2^24+1) (rc %d)", rc);
        if (rc == 0) fa_free(r, 4);
    }
    /* CI fuzz (main): a NaN matrix reaching LAPACKE_dgeev corrupted the caller's heap on LAPACKE's NaN-check path,
       before it returned -5. dgeev/dgees are now never called with a non-finite matrix; the routine reports the
       same failure. Under the sanitizer + ownership-check build any heap damage here aborts the test. */
    double ev_nan[4] = {1.0, NAN, 2.0, 3.0};
    int64_t ev_sh[2] = {2, 2};
    tsr_arg evg[1] = {arg_arr(ev_nan, 2, ev_sh)};
    CHECK(fa_call("linalg.eigvals", 1, evg, r, 1) < 0, "linalg.eigvals of a NaN matrix fails cleanly (no LAPACKE call)");
    CHECK(fa_call("linalg.eig", 1, evg, r, 2) < 0, "linalg.eig of a NaN matrix fails cleanly (no LAPACKE call)");
    double ev_ok[4] = {2.0, 0.0, 0.0, 3.0};
    evg[0] = arg_arr(ev_ok, 2, ev_sh);
    rc = fa_call("linalg.eigvals", 1, evg, r, 1);
    CHECK(rc == 0 && ((double *)r[0].arr.data)[0] == 2.0 && ((double *)r[0].arr.data)[1] == 3.0, "linalg.eigvals of a finite matrix unchanged (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    tsr_arg lw[2] = {arg_num(1.0), arg_num(NAN)};
    CHECK(fa_call("special.lambertw", 2, lw, r, 1) < 0, "lambertw with k = NaN refused");
    lw[1] = arg_num(1e300);
    CHECK(fa_call("special.lambertw", 2, lw, r, 1) < 0, "lambertw with k = 1e300 refused");
}

static void test_size_overflow(void)
{
    /* shapes whose element count wraps around int64 must never describe a small buffer */
    double buf[8];
    tsr_array a, v;
    int64_t big[2] = {(int64_t)1 << 62, 4}, neg[2] = {-1, -8}, one[1] = {0};
    CHECK(tsr_array_init(&a, TSR_F64, 2, big, buf) == TSR_EARG, "init refuses an overflowing shape");
    CHECK(tsr_array_init(&a, TSR_F64, 2, neg, buf) == TSR_EARG, "init refuses negative dimensions");
    tsr_array_init(&a, TSR_F64, 1, one, buf);                          /* size 0 */
    CHECK(tsr_array_reshape(&a, 2, big, &v) != TSR_OK, "reshape: 2^62 x 4 is not 0 elements");
    int64_t s1[2] = {(int64_t)1 << 40, 1}, s2[2] = {1, (int64_t)1 << 40}, out[2];
    int32_t nd;
    CHECK(tsr_broadcast_shape(2, s1, 2, s2, &nd, out) == TSR_ESHAPE, "broadcast refuses a 2^80-element result");
    tsr_iter it;
    int64_t st0[2] = {0, 0};
    const int64_t *sts[1] = {st0};
    int64_t huge[2] = {(int64_t)1 << 40, (int64_t)1 << 40};
    CHECK(tsr_iter_init(&it, 1, 2, huge, sts) == TSR_EARG, "iterator refuses an overflowing size");
}

/* ---- ABI, capability negotiation and runtime settings (ADR 0014) ---- */
#include <pthread.h>
#include <stdatomic.h>
#if !defined(_WIN32)
#include <sys/wait.h>
#endif
#include "../include/tessero_version.h"

#define ABI_DM_MOMENT 11                  /* DM_MOMENT in src/fn.h */
typedef struct { int id; int bad; int calls; _Atomic int *done; } abi_dist_job;

/* skewnorm.moment(order, a=1) over n elements; the element n/2 has the unsupported order 21 when `fail` is set */
static int abi_moment(int id, int64_t n, int fail, double *out)
{
    double *order = malloc((size_t)n * sizeof(double)), a = 1.0, loc = 0.0, scale = 1.0;
    for (int64_t i = 0; i < n; i++) order[i] = (fail && i == n / 2) ? 21.0 : 2.0;
    void *data[5] = {order, &a, &loc, &scale, out};
    int64_t shape[1] = {n}, strides[5] = {8, 0, 0, 0, 8};
    const int rc = tsr_fn_ufunc(id, ABI_DM_MOMENT, TSR_F64, 1, shape, 5, data, strides);
    free(order);
    return rc;
}

/* long calls whose middle element fails: each must report its own failure */
static void *abi_dist_failing(void *p)
{
    abi_dist_job *j = (abi_dist_job *)p;
    const int64_t n = (int64_t)1 << 16;
    double *out = malloc((size_t)n * sizeof(double));
    for (int it = 0; it < 12; it++, j->calls++)
        if (abi_moment(j->id, n, 1, out) != TSR_EARG || !strstr(tsr_fn_error(), "skewnorm")) j->bad++;
    free(out);
    atomic_store(j->done, 1);
    return NULL;
}

/* short successful calls meanwhile: none may pick up (or clear) the other thread's failure */
static void *abi_dist_ok(void *p)
{
    abi_dist_job *j = (abi_dist_job *)p;
    double out[4];
    while (!atomic_load(j->done) || j->calls == 0) {
        j->calls++;
        if (abi_moment(j->id, 4, 0, out) != TSR_OK || !isfinite(out[0])) j->bad++;
    }
    return NULL;
}

/* free every array (and nested sequence) a routine returned */
static void free_results(tsr_result *r, int n)
{
    for (int i = 0; i < n; i++) {
        if (r[i].kind == 6) {
            tsr_result *e = (tsr_result *)r[i].arr.data;
            free_results(e, (int)r[i].arr.shape[0]);
        }
        if ((r[i].kind == 3 || r[i].kind == 6) && r[i].arr.data) tsr_free(r[i].arr.data, r[i].bytes);
    }
}

void tsr_special_yn(const void *, const double *, double *);
void tsr_special_yn__int(const void *, const double *, double *);
void tsr_special_pbdv(const void *, const double *, double *);
void tsr_special_pbvv(const void *, const double *, double *);
void tsr_special_betainccinv(const void *, const double *, double *);
void tsr_special_smirnov(const void *, const double *, double *);

/* Bugs found by the CI matrix (2026-10): an old LAPACKE, sanitizers, Valgrind and an older compiler */
static void test_ci_regressions(void)
{
    /* LAPACKE < 3.11 (Ubuntu 22.04's 3.10, OpenBLAS 0.3.20) rejects ldvl/ldvr < n in row-major dgeev even when
       the vectors are not wanted: every companion-matrix root finder failed for degree >= 2 */
    double p3[3] = {1, -3, 2}, c3[3] = {2, -3, 1}, ch[3] = {0, 0, 1};
    int64_t s3[1] = {3};
    tsr_result r[3];
    struct { const char *fn; double *coef; double want0, want1; } rt[] = {
        {"np.roots", p3, 1, 2}, {"npoly.polyroots", c3, 1, 2}, {"npoly.chebroots", ch, -0.70710678118654757, 0.70710678118654757}};
    for (int k = 0; k < 3; k++) {
        const int id = tsr_fn_find(rt[k].fn);
        tsr_arg a[1] = {arg_arr(rt[k].coef, 1, s3)};
        const int rc = id >= 0 ? tsr_fn_routine(id, 1, a, 1, r) : -1;
        CHECK(rc == 0, "%s rc %d: %s", rt[k].fn, rc, tsr_fn_error());
        if (rc == 0) {
            const double *o = (const double *)r[0].arr.data;
            const int cplx = r[0].arr.dtype == TSR_C128;
            CHECK(r[0].arr.shape[0] == 2 && CLOSE(o[0], rt[k].want0, 1e-12) && CLOSE(o[cplx ? 2 : 1], rt[k].want1, 1e-12),
                  "%s roots %g %g", rt[k].fn, o[0], o[cplx ? 2 : 1]);
            free_results(r, 1);
        }
    }
    double b2[2] = {1, 2}, a3[3] = {1, 3, 2};   /* (s + 2) / ((s + 1)(s + 2)) */
    int64_t s2[1] = {2};
    const int tz = tsr_fn_find("signal.tf2zpk");
    tsr_arg za[2] = {arg_arr(b2, 1, s2), arg_arr(a3, 1, s3)};
    const int rz = tz >= 0 ? tsr_fn_routine(tz, 2, za, 3, r) : -1;
    CHECK(rz == 0 && r[1].arr.shape[0] == 2, "tf2zpk poles rc %d: %s", rz, tsr_fn_error());
    if (rz == 0) free_results(r, 3);
    double A[4] = {0, 1, -2, -3}, B[2] = {0, 1}, C[2] = {1, 0}, D[1] = {0};
    int64_t s22[2] = {2, 2}, s21[2] = {2, 1}, s12[2] = {1, 2}, s11[2] = {1, 1};
    const int st = tsr_fn_find("signal.ss2tf");
    tsr_arg sa[4] = {arg_arr(A, 2, s22), arg_arr(B, 2, s21), arg_arr(C, 2, s12), arg_arr(D, 2, s11)};
    const int rs = st >= 0 ? tsr_fn_routine(st, 4, sa, 2, r) : -1;
    CHECK(rs == 0, "ss2tf rc %d: %s", rs, tsr_fn_error());
    if (rs == 0) {
        const double *den = (const double *)r[1].arr.data;
        CHECK(CLOSE(den[0], 1.0, 1e-12) && CLOSE(den[1], 3.0, 1e-12) && CLOSE(den[2], 2.0, 1e-12), "ss2tf den %g %g %g", den[0], den[1], den[2]);
        free_results(r, 2);
    }

    /* pbdv / pbvv sized a work array from abs((int)v) + 2: an infinite or huge order was undefined behaviour and
       asked calloc for ~2^64 bytes (ASan aborts). SciPy's calloc fails and it returns NaN, NaN. */
    const double vs[] = {INFINITY, -INFINITY, 3e9, -3e9, 2147483646.0, NAN};
    for (int k = 0; k < 6; k++) {
        double in[2] = {vs[k], 1.0}, o[2] = {0, 0}, q[2] = {0, 0};
        tsr_special_pbdv(NULL, in, o);
        tsr_special_pbvv(NULL, in, q);
        CHECK(isnan(o[0]) && isnan(o[1]) && isnan(q[0]) && isnan(q[1]), "pbdv/pbvv(%g, 1) = %g %g %g %g", vs[k], o[0], o[1], q[0], q[1]);
    }
    double pin[2] = {2.5, 1.0}, po[2];
    tsr_special_pbdv(NULL, pin, po);
    CHECK(po[0] == -0.7476376353845996 && po[1] == 1.6629868944926967, "pbdv(2.5, 1) unchanged %.17g %.17g", po[0], po[1]);
    const int ps = tsr_fn_find("special.pbdv_seq");
    tsr_arg pa[2] = {arg_num(INFINITY), arg_num(1.0)};
    CHECK(ps >= 0 && tsr_fn_routine(ps, 2, pa, 2, r) == TSR_EARG, "pbdv_seq(inf) refused like int(inf)");

    /* yn of an order outside int: the (int)n cast was undefined (a gcc-11 build returned Y_2(0.5) = -5.44); SciPy's
       x86-64 build returns -inf for 0 <= x < inf and NaN otherwise */
    const double ns[] = {INFINITY, -INFINITY, 3e9, -3e9, -2147483648.0, 1e300};
    for (int k = 0; k < 6; k++) {
        double y1[2] = {ns[k], 0.5}, y2[2] = {ns[k], 0.0}, y3[2] = {ns[k], -1.0}, y4[2] = {ns[k], INFINITY}, o1, o2, o3, o4;
        tsr_special_yn(NULL, y1, &o1); tsr_special_yn(NULL, y2, &o2); tsr_special_yn(NULL, y3, &o3); tsr_special_yn(NULL, y4, &o4);
        CHECK(o1 == -INFINITY && o2 == -INFINITY && isnan(o3) && isnan(o4), "yn(%g, x) = %g %g %g %g", ns[k], o1, o2, o3, o4);
    }
    double y5[2] = {5.0, 0.5}, y6[2] = {-5.0, 100.0}, o5, o6, i5, i6;
    tsr_special_yn(NULL, y5, &o5); tsr_special_yn(NULL, y6, &o6);
    tsr_special_yn__int(NULL, y5, &i5); tsr_special_yn__int(NULL, y6, &i6);
    CHECK(o5 == i5 && o6 == i6, "yn of an integral double order is the integer-order kernel %.17g %.17g", o5, o6);
    /* yn is tier B (built on libm sin/cos/log, docs/reference/numerical-guarantees.md), so the bits are pinned per
       platform (PD-20): glibc, musl and MinGW agree; Apple's libm lands 1 ulp away in one value or the other */
#if defined(__APPLE__) && defined(__aarch64__)
    CHECK(o5 == -7946.3014788074743 && o6 == 0.029480196281661417, "yn finite orders unchanged %.17g %.17g", o5, o6);
#elif defined(__APPLE__)
    CHECK(o5 == -7946.3014788074752 && o6 == 0.029480196281661424, "yn finite orders unchanged %.17g %.17g", o5, o6);
#else
    /* glibc, musl and MinGW */
    CHECK(o5 == -7946.3014788074752 && o6 == 0.029480196281661417, "yn finite orders unchanged %.17g %.17g", o5, o6);
#endif
    double sm[2] = {INFINITY, 0.5}, so;
    tsr_special_smirnov(NULL, sm, &so);
    CHECK(isnan(so), "smirnov(inf, 0.5) = %g (SciPy: NaN)", so);

    /* windows.tukey with alpha >= 1 allocated its result, then let the Hann window allocate another over it (the
       first leaked; found by the extension's parity run under ASan). The final allocation check catches a leak. */
    const int tk = tsr_fn_find("windows.tukey");
    tsr_arg ta[2] = {arg_num(8), arg_num(1.0)};
    const int rt2 = tk >= 0 ? tsr_fn_routine(tk, 2, ta, 1, r) : -1;
    CHECK(rt2 == 0 && r[0].arr.shape[0] == 8 && ((const double *)r[0].arr.data)[0] == 0.0, "tukey(8, 1.0) is Hann (rc %d)", rt2);
    if (rt2 == 0) free_results(r, 1);

    /* betainccinv: Boost's evaluation_error path (Temme root finder) formatted its message through iostreams,
       which crashed under Valgrind; the exception and SciPy's catch stay, the value is unchanged */
    double bi[3] = {1e-10, 5, 1e-300}, bo;
    tsr_special_betainccinv(NULL, bi, &bo);
    double bj[3] = {2, 3, 0.4}, bp;
    tsr_special_betainccinv(NULL, bj, &bp);
    CHECK(bp == 0.44450000208376739, "betainccinv(2, 3, 0.4) = %.17g", bp);
    CHECK(isnan(bo), "betainccinv(1e-10, 5, 1e-300) = %g (SciPy: NaN)", bo);
}

/* Fuzzer findings, batch B (csrc/fuzz/README.md: numpy.polynomial, np.polyfit/roots and scipy.signal): each input
   below used to read or write out of bounds, hit a float->int cast with undefined behaviour, leak a kernel
   allocation or run for minutes. Every call must now return cleanly (an error where NumPy/SciPy raise) and leave
   no tessero allocation behind. */
static int fzb_call(const char *name, int nargs, tsr_arg *args, int nres, tsr_result *res)
{
    const int id = tsr_fn_find(name);
    if (id < 0) return 1;                                 /* a positive code never comes from a routine */
    const int64_t before = tsr_allocated();
    const int rc = tsr_fn_routine(id, nargs, args, nres, res);
    if (rc == 0) free_results(res, nres);
    if (tsr_allocated() != before) return 2;              /* leaked a kernel allocation */
    return rc;
}


static tsr_arg arg_arr_dt(void *d, int dtype, int32_t nd, const int64_t *shape)
{
    tsr_arg a; memset(&a, 0, sizeof a); a.kind = 3; tsr_array_init(&a.arr, dtype, nd, shape, d); return a;
}

static void test_fuzz_batch_b(void)
{
    tsr_result r[4];
    double one[8] = {1, 2, 3, 4, 5, 6, 7, 8}, nanv[4] = {NAN, NAN, NAN, NAN}, dummy = 0;
    int64_t s0[1] = {0}, s1[1] = {1}, s2[1] = {2}, s3[1] = {3}, s4[1] = {4};

    /* npoly.* (F16 F17 F19 F25 F28 F29 F32): an empty coefficient series indexed c[0], c[-1] or c[n-2], or sized
       calloc(2n-1) = calloc(-1); NumPy raises "Coefficient array is empty" (IndexError where it indexes first) */
    tsr_arg e0 = arg_arr(&dummy, 1, s0), c3 = arg_arr(one, 1, s3);
    CHECK(fzb_call("npoly.chebint", 2, (tsr_arg[]){e0, arg_int(1)}, 1, r) == TSR_EINDEX, "F16 chebint([]) is an IndexError");
    CHECK(fzb_call("npoly.hermeint", 2, (tsr_arg[]){e0, arg_int(1)}, 1, r) == TSR_EINDEX, "F29 hermeint([]) is an IndexError");
    CHECK(fzb_call("npoly.chebmulx", 1, (tsr_arg[]){e0}, 1, r) == TSR_EARG, "F17 chebmulx([]) refused");
    CHECK(fzb_call("npoly.lag2poly", 1, (tsr_arg[]){e0}, 1, r) == TSR_EARG, "F19 lag2poly([]) refused");
    CHECK(fzb_call("npoly.herm2poly", 1, (tsr_arg[]){e0}, 1, r) == TSR_EARG, "F32 herm2poly([]) refused");
    CHECK(fzb_call("npoly.chebdiv", 2, (tsr_arg[]){c3, e0}, 2, r) == TSR_EARG, "F25 chebdiv(c, []) refused");
    CHECK(fzb_call("npoly.chebmul", 2, (tsr_arg[]){e0, c3}, 1, r) == TSR_EARG, "F28 chebmul([], c) refused");
    CHECK(fzb_call("npoly.polypow", 2, (tsr_arg[]){e0, arg_int(2)}, 1, r) == TSR_EARG, "polypow([]) refused (was calloc(-1))");
    CHECK(fzb_call("npoly.chebval", 2, (tsr_arg[]){c3, e0}, 1, r) == TSR_EINDEX, "chebval(x, []) is an IndexError");
    /* an order or degree is an integer within reach: NaN / a fraction is NumPy's TypeError, a huge one is refused */
    CHECK(fzb_call("npoly.chebint", 2, (tsr_arg[]){c3, arg_num(NAN)}, 1, r) == TSR_ETYPE, "chebint(m=NaN) is a TypeError");
    CHECK(fzb_call("npoly.chebint", 2, (tsr_arg[]){c3, arg_num(2.5)}, 1, r) == TSR_ETYPE, "chebint(m=2.5) is a TypeError");
    CHECK(fzb_call("npoly.chebint", 2, (tsr_arg[]){c3, arg_num(1e9)}, 1, r) == TSR_EARG, "chebint(m=1e9) refused, not run");
    CHECK(fzb_call("npoly.legvander", 2, (tsr_arg[]){c3, arg_num(1e300)}, 1, r) == TSR_ETYPE, "legvander(deg=1e300): no float cast");
    CHECK(fzb_call("npoly.legvander", 2, (tsr_arg[]){c3, arg_int(INT64_MAX - 2)}, 1, r) == TSR_ENOMEM, "legvander(deg=2^63-3): shape refused");
    CHECK(fzb_call("npoly.leggauss", 1, (tsr_arg[]){arg_int(1048576)}, 2, r) == TSR_ENOMEM, "leggauss(2^20): the 2^20 x 2^20 companion refused");
    const int id_der = tsr_fn_find("npoly.chebder");
    if (id_der >= 0 && tsr_fn_routine(id_der, 2, (tsr_arg[]){c3, arg_int(INT64_MAX)}, 1, r) == 0) {   /* was INT64_MAX steps */
        CHECK(r[0].arr.shape[0] == 1 && ((const double *)r[0].arr.data)[0] == 0.0, "chebder(c, m=2^63-1) = [0]");
        free_results(r, 1);
    } else CHECK(0, "chebder(c, m=2^63-1) failed: %s", tsr_fn_error());
    const int id_int = tsr_fn_find("npoly.chebint");
    if (id_int >= 0 && tsr_fn_routine(id_int, 2, (tsr_arg[]){c3, arg_int(2)}, 1, r) == 0) {
        CHECK(r[0].arr.shape[0] == 5, "chebint([1,2,3], 2) has 5 terms");
        free_results(r, 1);
    } else CHECK(0, "chebint([1,2,3], 2) failed: %s", tsr_fn_error());
    /* 2-D / 3-D forms: x and y of different lengths were read in lock-step, a one-element deg read deg[1] */
    double two[2] = {1, 2};
    CHECK(fzb_call("npoly.chebval2d", 3, (tsr_arg[]){arg_arr(two, 1, s2), arg_arr(one, 1, s1), arg_arr(one, 2, (int64_t[]){2, 2})}, 1, r) == TSR_EARG,
          "chebval2d with x, y of different shapes refused");
    CHECK(fzb_call("npoly.chebvander2d", 3, (tsr_arg[]){arg_arr(two, 1, s2), arg_arr(two, 1, s2), arg_arr(two, 1, s1)}, 1, r) == TSR_EARG,
          "chebvander2d with one degree refused");
    CHECK(fzb_call("npoly.polyvander2d", 3, (tsr_arg[]){arg_arr(two, 1, s2), arg_arr(two, 1, s2), arg_arr(nanv, 1, s2)}, 1, r) == TSR_ETYPE,
          "polyvander2d with NaN degrees refused");
    /* np.polyfit: NaN deg was cast to int64, an empty x reached LAPACK */
    CHECK(fzb_call("np.polyfit", 3, (tsr_arg[]){c3, c3, arg_num(NAN)}, 1, r) == TSR_EARG, "np.polyfit(deg=NaN) refused");
    CHECK(fzb_call("np.polyfit", 3, (tsr_arg[]){e0, e0, arg_int(1)}, 1, r) == TSR_ETYPE, "np.polyfit on empty x is a TypeError");

    /* zpk transforms (F12 F14 F22 F37): more zeros than poles made the degree negative and the zero buffer too
       small; SciPy: "Improper transfer function. Must have at least as many poles as zeros." */
    tsr_arg kz = arg_num(1.0), p1 = arg_arr(one, 1, s1);
    CHECK(fzb_call("signal.lp2bp_zpk", 3, (tsr_arg[]){c3, p1, kz}, 3, r) == TSR_EARG, "F12 lp2bp_zpk improper refused");
    CHECK(fzb_call("signal.lp2hp_zpk", 3, (tsr_arg[]){c3, p1, kz}, 3, r) == TSR_EARG, "F14 lp2hp_zpk improper refused");
    CHECK(fzb_call("signal.lp2bs_zpk", 3, (tsr_arg[]){c3, p1, kz}, 3, r) == TSR_EARG, "F22 lp2bs_zpk improper refused");
    CHECK(fzb_call("signal.bilinear_zpk", 4, (tsr_arg[]){c3, p1, kz, arg_num(2.0)}, 3, r) == TSR_EARG, "F37 bilinear_zpk improper refused");
    CHECK(fzb_call("signal.lp2lp_zpk", 3, (tsr_arg[]){c3, p1, kz}, 3, r) == TSR_EARG, "lp2lp_zpk improper refused");
    /* zpk2sos (F30): an N-d root array was walked with its first stride only (out of bounds); it is now read in C
       order through every stride, so a (2, 1) zero array means the same as [-1, -1] */
    double zz[2] = {-1, -1}, pp[2] = {0.5, 0.5};
    const int id_sos = tsr_fn_find("signal.zpk2sos");
    if (id_sos >= 0 && tsr_fn_routine(id_sos, 3, (tsr_arg[]){arg_arr(zz, 2, (int64_t[]){2, 1}), arg_arr(pp, 1, s2), kz}, 1, r) == 0) {
        const double *o = (const double *)r[0].arr.data;
        CHECK(r[0].arr.shape[0] == 1 && o[0] == 1 && o[1] == 2 && o[2] == 1 && o[3] == 1 && o[4] == -1 && o[5] == 0.25,
              "F30 zpk2sos of a (2,1) zero array");
        free_results(r, 1);
    } else CHECK(0, "F30 zpk2sos failed: %s", tsr_fn_error());
    CHECK(fzb_call("signal.zpk2sos", 3, (tsr_arg[]){arg_arr(nanv, 3, (int64_t[]){1, 2, 2}), arg_arr(nanv, 1, s4), kz}, 1, r) <= 0, "zpk2sos of NaN roots is clean");
    /* lp2lp (F21): an empty denominator was read at a[0] by sig_normalize */
    double b3[3] = {NAN, 1, 2};
    CHECK(fzb_call("signal.lp2lp", 2, (tsr_arg[]){arg_arr(b3, 1, s3), arg_arr(&dummy, 3, (int64_t[]){2, 2, 0})}, 2, r) == TSR_EARG, "F21 lp2lp with empty a refused");
    CHECK(fzb_call("signal.lp2bp", 2, (tsr_arg[]){arg_arr(b3, 1, s3), e0}, 2, r) == TSR_EARG, "lp2bp with empty a refused (was calloc(-1))");
    /* iirdesign (F35): wp/ws were read as raw doubles at data[0], an empty complex array past its end */
    CHECK(fzb_call("signal.iirdesign", 4, (tsr_arg[]){arg_arr(one, 1, s4), arg_arr_dt(&dummy, TSR_C128, 3, (int64_t[]){0, 1, 0}), arg_num(1), arg_num(40)}, 2, r) == TSR_EARG,
          "F35 iirdesign with an empty ws refused");
    /* dbode (F38): a 0-d denominator was indexed at shape[0] - 1 */
    double big = 1048576.0;
    CHECK(fzb_call("signal.dbode", 3, (tsr_arg[]){arg_arr(two, 1, s2), arg_arr(&big, 0, NULL), arg_num(NAN)}, 3, r) == TSR_EARG, "F38 dbode with a 0-d den refused");
    CHECK(fzb_call("signal.dbode", 2, (tsr_arg[]){arg_arr(two, 1, s2), e0}, 3, r) == TSR_EARG, "dbode with an empty den refused");
    /* welch / csd / stft (F39): a negative or NaN segment length reached calloc(count * size) */
    double x4[4] = {64, 9.3e18, 2, 3};
    tsr_arg x4a = arg_arr(x4, 1, s4), none; memset(&none, 0, sizeof none);
    CHECK(fzb_call("signal.welch", 4, (tsr_arg[]){x4a, none, none, arg_int(-3)}, 2, r) == TSR_EARG, "F39 welch(nperseg=-3) refused");
    CHECK(fzb_call("signal.welch", 4, (tsr_arg[]){x4a, none, none, arg_num(NAN)}, 2, r) == TSR_EARG, "welch(nperseg=NaN) refused");
    CHECK(fzb_call("signal.welch", 6, (tsr_arg[]){x4a, none, none, arg_int(4), none, arg_int(2)}, 2, r) == TSR_EARG, "welch(nfft < nperseg) refused");
    CHECK(fzb_call("signal.welch", 5, (tsr_arg[]){x4a, none, none, arg_int(4), arg_num(-INFINITY)}, 2, r) == TSR_EARG, "welch(noverlap=-inf) refused");
    CHECK(fzb_call("signal.csd", 5, (tsr_arg[]){x4a, x4a, none, none, arg_int(-3)}, 2, r) == TSR_EARG, "csd(nperseg=-3) refused");
    CHECK(fzb_call("signal.spectrogram", 4, (tsr_arg[]){x4a, none, none, arg_num(1e300)}, 3, r) <= 0, "spectrogram(nperseg=1e300) clean");
    /* stft (F3): the input copy (a tsr_alloc block) was released with free(), leaking 64 bytes on every call */
    CHECK(fzb_call("signal.stft", 1, (tsr_arg[]){x4a}, 3, r) == 0, "F3 stft(x) succeeds without a leak");
    CHECK(fzb_call("signal.stft", 6, (tsr_arg[]){x4a, none, none, arg_int(4), none, arg_num(-1)}, 3, r) == TSR_EARG, "stft(nfft=-1) refused");
    /* peak_widths / peak_prominences (F26): a peak index outside x (4e9) was dereferenced */
    double pk[2] = {4000000000.0, 3}, xn[1] = {NAN}, rel = -64;
    CHECK(fzb_call("signal.peak_widths", 3, (tsr_arg[]){arg_arr(xn, 1, s1), arg_arr(pk, 1, s2), arg_arr(&rel, 0, NULL)}, 4, r) == TSR_EARG, "F26 peak_widths bad peak refused");
    CHECK(fzb_call("signal.peak_prominences", 2, (tsr_arg[]){arg_arr(xn, 1, s1), arg_arr(pk, 1, s2)}, 3, r) == TSR_EARG, "peak_prominences bad peak refused");
    double xp[3] = {0, 1, 0}, p_ok = 1;
    const int id_pp = tsr_fn_find("signal.peak_prominences");
    if (id_pp >= 0 && tsr_fn_routine(id_pp, 2, (tsr_arg[]){arg_arr(xp, 1, s3), arg_arr(&p_ok, 1, s1)}, 3, r) == 0) {
        CHECK(((const double *)r[0].arr.data)[0] == 1.0, "peak_prominences([0,1,0], [1]) = 1");
        free_results(r, 3);
    } else CHECK(0, "peak_prominences valid call failed: %s", tsr_fn_error());
    /* windows: chebwin (F27) was an O(M^2) cosine sum (minutes at M = 65536); tukey (F24, the alpha >= 1 path the
       bugfix stream fixed) and every window took M through an unchecked cast */
    const int id_cw = tsr_fn_find("windows.chebwin");
    if (id_cw >= 0 && tsr_fn_routine(id_cw, 2, (tsr_arg[]){arg_int(65536), arg_num(100)}, 1, r) == 0) {
        const double *w = (const double *)r[0].arr.data; double mx = 0;
        for (int64_t i = 0; i < 65536; i++) if (w[i] > mx) mx = w[i];
        CHECK(r[0].arr.shape[0] == 65536 && mx == 1.0 && fabs(w[1] - w[65534]) < 1e-12, "F27 chebwin(65536) by FFT");
        free_results(r, 1);
    } else CHECK(0, "F27 chebwin(65536) failed: %s", tsr_fn_error());
    CHECK(fzb_call("windows.tukey", 3, (tsr_arg[]){arg_int(4), arg_int(1), arg_arr(one, 3, (int64_t[]){3, 1, 1})}, 1, r) <= 0, "F24 tukey(4, 1, sym=array) clean");
    CHECK(fzb_call("windows.tukey", 2, (tsr_arg[]){arg_num(INFINITY), arg_num(0.5)}, 1, r) == TSR_EARG, "tukey(M=inf) refused");
    CHECK(fzb_call("windows.tukey", 2, (tsr_arg[]){arg_int(8), arg_num(NAN)}, 1, r) == TSR_EARG, "tukey(alpha=NaN) refused (SciPy: int(nan))");
    CHECK(fzb_call("windows.hann", 1, (tsr_arg[]){arg_num(9.3e18)}, 1, r) == TSR_EARG, "hann(M=9.3e18) refused");
    /* the systemic float->int cast of numeric arguments: NaN / inf / 1e300 now take the routine's error path */
    CHECK(fzb_call("signal.freqz", 3, (tsr_arg[]){c3, c3, arg_num(NAN)}, 2, r) == TSR_EARG, "freqz(worN=NaN) refused");
    CHECK(fzb_call("signal.butter", 2, (tsr_arg[]){arg_num(INFINITY), arg_num(0.5)}, 2, r) == TSR_EARG, "butter(N=inf) refused");
    CHECK(fzb_call("signal.savgol_coeffs", 2, (tsr_arg[]){arg_num(1e300), arg_int(2)}, 1, r) == TSR_EARG, "savgol_coeffs(1e300) refused");
    CHECK(fzb_call("signal.correlation_lags", 2, (tsr_arg[]){arg_num(NAN), arg_int(3)}, 1, r) == TSR_EARG, "correlation_lags(NaN) refused");
    CHECK(fzb_call("signal.envelope", 3, (tsr_arg[]){x4a, arg_int(1), arg_int(9)}, 1, r) == TSR_EARG, "envelope band outside -n//2..(n+1)//2 refused");
    double taps[1] = {40};
    CHECK(fzb_call("signal.max_len_seq", 4, (tsr_arg[]){arg_int(4), none, none, arg_arr(taps, 1, s1)}, 2, r) == TSR_EARG, "max_len_seq tap > nbits refused");
    CHECK(fzb_call("signal.resample_poly", 3, (tsr_arg[]){x4a, arg_num(1e18), arg_int(1)}, 1, r) < 0, "resample_poly(up=1e18) refused");
    double fq[2] = {0, 1}, gn[2] = {1, 0};
    CHECK(fzb_call("signal.firwin2", 4, (tsr_arg[]){arg_int(9), arg_arr(fq, 1, s2), arg_arr(gn, 1, s2), arg_int(3)}, 1, r) == TSR_EARG, "firwin2(nfreqs <= numtaps) refused");
    double far[2] = {1e300, -INFINITY};
    const int id_cs = tsr_fn_find("signal.cspline1d_eval");
    if (id_cs >= 0 && tsr_fn_routine(id_cs, 2, (tsr_arg[]){c3, arg_arr(far, 1, s2)}, 1, r) == 0) {   /* the mirror loop never ended */
        CHECK(isfinite(((const double *)r[0].arr.data)[0]) && isnan(((const double *)r[0].arr.data)[1]), "cspline1d_eval at 1e300 / -inf terminates");
        free_results(r, 1);
    } else CHECK(0, "cspline1d_eval far points failed: %s", tsr_fn_error());

    /* found by this batch's own campaign (fuzz/corpus/routines/N*): more of the same class in scipy.signal */
    CHECK(fzb_call("signal.cspline1d", 1, (tsr_arg[]){e0}, 1, r) == 0, "N1 cspline1d([]) is empty (wrote out[-1])");
    CHECK(fzb_call("signal.symiirorder1", 3, (tsr_arg[]){e0, arg_num(0.5), arg_num(0.1)}, 1, r) == TSR_EARG, "symiirorder1([]) refused (wrote out[-1])");
    CHECK(fzb_call("signal.symiirorder2", 3, (tsr_arg[]){arg_arr(one, 1, s1), arg_num(0.5), arg_num(0.1)}, 1, r) == TSR_EARG, "N2 symiirorder2 of one sample refused (read x[1])");
    CHECK(fzb_call("signal.envelope", 4, (tsr_arg[]){x4a, none, none, arg_int(64)}, 1, r) == 0, "N3 envelope(n_out > len(z)) pads the spectrum");
    CHECK(fzb_call("windows.taylor", 2, (tsr_arg[]){arg_int(24), arg_num(9.3e18)}, 1, r) == TSR_EARG, "N5 taylor(nbar=9.3e18) refused");
    CHECK(fzb_call("signal.medfilt", 2, (tsr_arg[]){x4a, arg_arr((double[]){INFINITY}, 1, s1)}, 1, r) == TSR_EARG, "N6 medfilt(kernel_size=[inf]) refused");
    const int id_mf = tsr_fn_find("signal.medfilt");
    double mx5[5] = {5, -1, 3, 9, 2};
    if (id_mf >= 0 && tsr_fn_routine(id_mf, 2, (tsr_arg[]){arg_arr(mx5, 1, (int64_t[]){5}), arg_int(65537)}, 1, r) == 0) {   /* the box is all padding but x */
        CHECK(((const double *)r[0].arr.data)[2] == 0.0, "medfilt with a kernel far larger than x");
        free_results(r, 1);
    } else CHECK(0, "medfilt huge kernel failed: %s", tsr_fn_error());
    if (id_mf >= 0 && tsr_fn_routine(id_mf, 2, (tsr_arg[]){arg_arr(mx5, 1, (int64_t[]){5}), arg_int(3)}, 1, r) == 0) {
        const double *o = (const double *)r[0].arr.data;
        CHECK(o[0] == 0 && o[1] == 3 && o[2] == 3 && o[3] == 3 && o[4] == 2, "medfilt([5,-1,3,9,2], 3) = [0,3,3,3,2]");
        free_results(r, 1);
    } else CHECK(0, "medfilt 3 failed: %s", tsr_fn_error());
    double rb[2] = {4000000000.0, 3}, rd = NAN, rb1[2] = {0.1, 0.1}, rd1 = 1;
    CHECK(fzb_call("signal.remez", 3, (tsr_arg[]){arg_int(1), arg_arr(rb, 1, s2), arg_arr(&rd, 0, NULL)}, 1, r) == TSR_EARG, "N7 remez with edges beyond fs/2 refused");
    CHECK(fzb_call("signal.remez", 3, (tsr_arg[]){arg_int(1), arg_arr(rb1, 1, s2), arg_arr(&rd1, 1, s1)}, 1, r) <= 0, "remez with a one-point grid is clean");
    uint8_t h8[2] = {0, 0};
    double xf[5] = {166, 3, 1e-300, 3, 1e-9};
    CHECK(fzb_call("signal.find_peaks", 2, (tsr_arg[]){arg_arr(xf, 1, (int64_t[]){5}), arg_arr_dt(h8, TSR_U8, 1, s2)}, 1, r) == 0, "N8 find_peaks(height=uint8 array) reads bytes");
    double bf[1] = {1};
    CHECK(fzb_call("signal.filtfilt", 3, (tsr_arg[]){e0, arg_arr(bf, 1, s1), arg_arr(one, 1, (int64_t[]){8})}, 1, r) == TSR_EARG, "N10 filtfilt with empty b refused");
    CHECK(fzb_call("signal.lfilter_zi", 2, (tsr_arg[]){e0, arg_arr(bf, 1, s1)}, 1, r) == TSR_EARG, "lfilter_zi with empty b refused");
    CHECK(fzb_call("signal.convolve", 3, (tsr_arg[]){arg_arr(two, 1, s2), e0, arg_str("same")}, 1, r) == TSR_EARG, "N11 convolve(a, [], 'same') refused");
    CHECK(fzb_call("signal.correlate", 3, (tsr_arg[]){x4a, e0, arg_str("same")}, 1, r) == TSR_EARG, "N12 correlate(x, [], same) refused (read past the empty full result)");
    CHECK(fzb_call("signal.correlate", 3, (tsr_arg[]){e0, x4a, arg_str("same")}, 1, r) == 0, "correlate([], x, same) is empty (SciPy)");
    CHECK(fzb_call("signal.findfreqs", 3, (tsr_arg[]){arg_arr(one, 1, s2), e0, arg_int(4)}, 1, r) <= 0, "findfreqs with no poles does not leak");
    /* dpss (N9): above 1024 points only the Kmax tapers are computed (it was an O(M^3) full eigensolve) */
    const int id_dp = tsr_fn_find("windows.dpss");
    if (id_dp >= 0 && tsr_fn_routine(id_dp, 3, (tsr_arg[]){arg_int(1500), arg_num(2.5), arg_int(2)}, 1, r) == 0) {
        const double *w = (const double *)r[0].arr.data;
        CHECK(r[0].arr.shape[0] == 2 && fabs(w[0] - 0.00017167929445718752) < 1e-12 && fabs(w[750] - 0.04529204787020111) < 1e-12 &&
              fabs(w[1500 + 100] - 0.006608579188112873) < 1e-12 && fabs(w[1500 + 1499] + 0.0012259428754950801) < 1e-12,
              "dpss(1500, 2.5, 2) matches SciPy");
        free_results(r, 1);
    } else CHECK(0, "dpss(1500) failed: %s", tsr_fn_error());
    if (id_dp >= 0 && tsr_fn_routine(id_dp, 2, (tsr_arg[]){arg_int(4096), arg_num(1.0)}, 1, r) == 0) {
        const double *w = (const double *)r[0].arr.data;
        CHECK(fabs(w[0] - 0.24011681985880415) < 1e-10 && fabs(w[2048] - 0.9999999403953588) < 1e-12, "N9 dpss(4096, 1) matches SciPy");
        free_results(r, 1);
    } else CHECK(0, "dpss(4096) failed: %s", tsr_fn_error());
    CHECK(fzb_call("windows.dpss", 3, (tsr_arg[]){arg_int(4096), arg_num(17), arg_int(4096)}, 1, r) == TSR_EARG, "N13 dpss(4096, 17, Kmax=4096) refused, not run for minutes");
}

/* fuzz batch C: interpolate.splantider/splder/insert/spalde/NdPPoly/NdBSpline and the hierarchy routines trusted
   their (t, c, k) sizes and linkage ids. Each case was an out-of-bounds access, a stack overflow, a cast of an
   unchecked double or an uninitialized output; each now runs (sizes SciPy accepts) or refuses as SciPy does. */
static int fzc_run(const char *name, int nargs, tsr_arg *args, int nres, tsr_result *res)
{
    const int id = tsr_fn_find(name);
    return id < 0 ? 1 : tsr_fn_routine(id, nargs, args, nres, res);   /* caller frees on 0 */
}

static void test_fuzz_batch_c(void)
{
    tsr_result r[3];
    const int64_t before = tsr_allocated();
    /* ---- splantider (A-N1): a c longer than len(t)+2n+2 was copied into a buffer of that size (heap write), and a k
       with no core left read cnew[-k] (heap read before the block) */
    double t4[4] = {0, 1, 2, 3}, c50[50], one = 2.0;
    for (int i = 0; i < 50; i++) c50[i] = i + 1;
    int64_t s4[1] = {4}, s50[1] = {50}, s1[1] = {1}, s2[1] = {2};
    tsr_arg ta = arg_arr(t4, 1, s4);
    if (fzc_run("interpolate.splantider", 4, (tsr_arg[]){ta, arg_arr(c50, 1, s50), arg_int(0), arg_int(1)}, 3, r) == 0) {
        const double *cc = (const double *)r[1].arr.data;
        CHECK(r[0].arr.shape[0] == 6 && r[1].arr.shape[0] == 6 && r[2].ival == 1 && cc[0] == 0.0 && cc[1] == 1.0 && cc[3] == 6.0,
              "A-N1 splantider with len(c) > len(t) uses c[:len(t)-k-1] (SciPy)");
        free_results(r, 3);
    } else CHECK(0, "A-N1 splantider long c failed: %s", tsr_fn_error());
    CHECK(fzb_call("interpolate.splantider", 3, (tsr_arg[]){ta, arg_arr(c50, 1, s50), arg_int(7)}, 3, r) == TSR_EINDEX,
          "A-N1 splantider with k >= len(t)-1 is SciPy's IndexError (was cnew[-k])");
    CHECK(fzb_call("interpolate.splantider", 3, (tsr_arg[]){ta, arg_arr(c50, 1, s2), arg_int(0)}, 3, r) == TSR_EARG,
          "splantider: c shorter than the core does not broadcast");
    CHECK(fzb_call("interpolate.splantider", 3, (tsr_arg[]){ta, arg_arr(&one, 1, s1), arg_int(0)}, 3, r) == 0,
          "splantider: a one-element c broadcasts");
    CHECK(fzb_call("interpolate.splantider", 3, (tsr_arg[]){ta, arg_arr(c50, 1, s4), arg_num(NAN)}, 3, r) == TSR_ETYPE, "splantider(k=NaN) refused, no cast");
    CHECK(fzb_call("interpolate.splantider", 3, (tsr_arg[]){ta, arg_arr(c50, 1, s4), arg_num(1.5)}, 3, r) == TSR_ETYPE, "splantider(k=1.5) is a TypeError");
    CHECK(fzb_call("interpolate.splantider", 3, (tsr_arg[]){ta, arg_arr(c50, 1, s4), arg_int(-1)}, 3, r) == TSR_EARG, "splantider(k=-1) refused");
    CHECK(fzb_call("interpolate.splantider", 4, (tsr_arg[]){ta, arg_arr(c50, 1, s4), arg_int(0), arg_num(1e300)}, 3, r) == TSR_EARG,
          "splantider(n=1e300) refused, no cast");
    CHECK(fzb_call("interpolate.splantider", 4, (tsr_arg[]){ta, arg_arr(c50, 1, s4), arg_int(0), arg_int(INT32_MAX)}, 3, r) == TSR_EARG,
          "splantider(n=2^31-1): the work bound refuses it");
    /* splder: n = 0 returns (t, c, k) as given; a negative n is splantider (SciPy); repeated knots raise */
    double tc[8] = {0, 0, 0, 0, 1, 1, 1, 1}, cc4[8] = {1, 2, 3, 4, 0, 0, 0, 0};
    int64_t s8[1] = {8};
    tsr_arg tca = arg_arr(tc, 1, s8), cca = arg_arr(cc4, 1, s8);
    if (fzc_run("interpolate.splder", 4, (tsr_arg[]){tca, arg_arr(cc4, 1, s4), arg_int(3), arg_int(0)}, 3, r) == 0) {
        CHECK(r[1].arr.shape[0] == 4 && r[2].ival == 3, "splder(n=0) returns c unchanged (length 4)");
        free_results(r, 3);
    } else CHECK(0, "splder(n=0) failed: %s", tsr_fn_error());
    if (fzc_run("interpolate.splder", 4, (tsr_arg[]){tca, cca, arg_int(3), arg_int(-1)}, 3, r) == 0) {
        CHECK(r[0].arr.shape[0] == 10 && r[2].ival == 4, "splder(n=-1) is splantider(n=1)");
        free_results(r, 3);
    } else CHECK(0, "splder(n=-1) failed: %s", tsr_fn_error());
    double trep[8] = {0, 0, 0, 0, 0, 1, 1, 1};
    CHECK(fzb_call("interpolate.splder", 3, (tsr_arg[]){arg_arr(trep, 1, s8), cca, arg_int(3)}, 3, r) == TSR_EARG,
          "splder with internal repeated knots raises (SciPy's FloatingPointError -> ValueError)");
    CHECK(fzb_call("interpolate.splder", 4, (tsr_arg[]){arg_arr(tc, 1, s4), cca, arg_int(3), arg_int(2)}, 3, r) == TSR_EARG,
          "splder with too few knots for n passes refused");
    /* insert: x at the right end of the base interval found no interval and used l = 0 (ccur[-k]) */
    if (fzc_run("interpolate.insert", 4, (tsr_arg[]){arg_num(1.0), tca, cca, arg_int(3)}, 3, r) == 0) {
        CHECK(r[0].arr.shape[0] == 9, "insert at the right end of the base interval");
        free_results(r, 3);
    } else CHECK(0, "insert(x=t[n-k-1]) failed: %s", tsr_fn_error());
    CHECK(fzb_call("interpolate.insert", 4, (tsr_arg[]){arg_num(2.0), tca, cca, arg_int(3)}, 3, r) == TSR_EARG, "insert outside the base interval: Invalid input data");
    CHECK(fzb_call("interpolate.insert", 4, (tsr_arg[]){arg_num(0.5), tca, cca, arg_int(9)}, 3, r) == TSR_EARG, "insert with k too large for t refused");
    /* spalde and NdBSpline evaluated through a d[8] stack buffer: k >= 8 overflowed it */
    double t20[20], c10[10], xs[3] = {0.0, 0.5, 1.0};
    for (int i = 0; i < 20; i++) t20[i] = i < 10 ? 0.0 : 1.0;
    for (int i = 0; i < 10; i++) c10[i] = 1.0;
    int64_t s20[1] = {20}, s10[1] = {10}, s3[1] = {3};
    if (fzc_run("interpolate.spalde", 4, (tsr_arg[]){arg_arr(t20, 1, s20), arg_arr(c10, 1, s10), arg_int(9), arg_arr(xs, 1, s3)}, 1, r) == 0) {
        const double *d = (const double *)r[0].arr.data;
        CHECK(r[0].arr.shape[1] == 10 && fabs(d[0] - 1.0) < 1e-12 && fabs(d[1]) < 1e-9, "spalde with k = 9 (constant spline: f = 1, f' = 0)");
        free_results(r, 1);
    } else CHECK(0, "spalde k=9 failed: %s", tsr_fn_error());
    double xo[1] = {2.0};
    CHECK(fzb_call("interpolate.spalde", 4, (tsr_arg[]){tca, cca, arg_int(3), arg_arr(xo, 1, s1)}, 1, r) == TSR_EARG, "spalde outside t(k)..t(n-k-1) raises");
    CHECK(fzb_call("interpolate.spalde", 4, (tsr_arg[]){arg_arr(tc, 1, s4), cca, arg_int(3), arg_arr(xs, 1, s3)}, 1, r) == TSR_EARG, "spalde with too few knots refused");
    double cxy[100], xi3[6] = {0.5, 0.5, 0.5, 0.2, 0.2, 0.2};
    for (int i = 0; i < 100; i++) cxy[i] = 1.0;
    int64_t sc[2] = {10, 10}, sxi2[2] = {3, 2}, sxi3[2] = {2, 3};
    tsr_arg t20a = arg_arr(t20, 1, s20);
    if (fzc_run("interpolate.NdBSpline", 6, (tsr_arg[]){t20a, t20a, arg_arr(cxy, 2, sc), arg_int(9), arg_int(9), arg_arr(xi3, 2, sxi2)}, 1, r) == 0) {
        CHECK(fabs(((const double *)r[0].arr.data)[0] - 1.0) < 1e-12, "NdBSpline with kx = ky = 9");
        free_results(r, 1);
    } else CHECK(0, "NdBSpline k=9 failed: %s", tsr_fn_error());
    CHECK(fzb_call("interpolate.NdBSpline", 6, (tsr_arg[]){t20a, t20a, arg_arr(cxy, 2, sc), arg_int(9), arg_int(9), arg_arr(xi3, 2, sxi3)}, 1, r) == TSR_EARG,
          "NdBSpline with xi of shape (n, 3) refused (read xi[2t+1] of another row)");
    CHECK(fzb_call("interpolate.NdBSpline", 6, (tsr_arg[]){ta, t20a, arg_arr(cxy, 2, sc), arg_int(9), arg_int(9), arg_arr(xi3, 2, sxi2)}, 1, r) == TSR_EARG,
          "NdBSpline with a knot vector shorter than c.shape[0] + k + 1 refused");
    double c4d[4] = {1, 1, 1, 1}, xb2[2] = {0, 1};
    int64_t s4d[4] = {1, 1, 2, 2}, sxi1[2] = {6, 1};
    CHECK(fzb_call("interpolate.NdPPoly", 4, (tsr_arg[]){arg_arr(c4d, 4, s4d), arg_arr(xb2, 1, s2), arg_arr(xb2, 1, s2), arg_arr(xi3, 2, sxi2)}, 1, r) == TSR_EARG,
          "NdPPoly with too few breakpoints refused (read xb[mx])");
    CHECK(fzb_call("interpolate.NdPPoly", 4, (tsr_arg[]){arg_arr(c4d, 4, s4d), arg_arr(t4, 1, s3), arg_arr(t4, 1, s3), arg_arr(xi3, 2, sxi1)}, 1, r) == TSR_EARG,
          "NdPPoly with xi of shape (n, 1) refused");

    /* ---- hierarchy (A-N2): ids naming clusters not yet formed, repeated or non-finite indexed Z, R and buffers out
       of bounds; SciPy's _is_valid_linkage refuses them with its messages */
    int64_t z2[2] = {2, 4}, z1[2] = {1, 4}, z0[2] = {0, 4}, r1[2] = {1, 4};
    double Zok[8] = {0, 1, 1, 2, 2, 3, 2, 3}, Zbefore[8] = {0, 5, 1, 2, 1, 2, 2, 3}, Zrep[8] = {0, 1, 1, 2, 0, 1, 2, 2};
    double Znan[8] = {0, NAN, 1, 2, 2, 3, 2, 3}, Zneg[8] = {0, 1, -1, 2, 2, 3, 2, 3};
    tsr_arg zok = arg_arr(Zok, 2, z2);
    CHECK(fzb_call("hierarchy.fcluster", 2, (tsr_arg[]){arg_arr(Zbefore, 2, z2), arg_num(1.0)}, 1, r) == TSR_EARG,
          "A-N2 fcluster with a cluster used before it is formed refused (incon_collect read past Z)");
    CHECK(fzb_call("hierarchy.fcluster", 2, (tsr_arg[]){arg_arr(Zrep, 2, z2), arg_num(1.0)}, 1, r) == TSR_EARG, "fcluster with a cluster used twice refused");
    CHECK(fzb_call("hierarchy.fcluster", 2, (tsr_arg[]){arg_arr(Znan, 2, z2), arg_num(1.0)}, 1, r) == TSR_EARG, "fcluster with a NaN id refused, no cast");
    CHECK(fzb_call("hierarchy.inconsistent", 1, (tsr_arg[]){arg_arr(Zneg, 2, z2)}, 1, r) == TSR_EARG, "inconsistent with a negative distance refused");
    CHECK(fzb_call("hierarchy.cophenet", 1, (tsr_arg[]){arg_arr(Zok, 2, z0)}, 1, r) == TSR_EARG, "cophenet on an empty Z refused");
    double Zone[4] = {0, 7, 1, 2};
    CHECK(fzb_call("hierarchy.leaves_list", 1, (tsr_arg[]){arg_arr(Zone, 2, z1)}, 1, r) == TSR_EARG, "a one-row Z naming cluster 7 refused");
    if (fzc_run("hierarchy.fcluster", 3, (tsr_arg[]){zok, arg_num(0.5), arg_str("distance")}, 1, r) == 0) {
        const int64_t *T = (const int64_t *)r[0].arr.data;
        CHECK(r[0].arr.shape[0] == 3 && T[0] != T[1] && T[1] != T[2] && T[0] != T[2], "fcluster distance 0.5 on a valid Z: three clusters");
        free_results(r, 1);
    } else CHECK(0, "fcluster valid failed: %s", tsr_fn_error());
    double mc1[1] = {1.0}, R1[4] = {1, 0, 1, 0}, Rneg[8] = {1, -1, 1, 0, 1, 0, 1, 0};
    CHECK(fzb_call("hierarchy.fcluster", 6, (tsr_arg[]){zok, arg_num(1.0), arg_str("monocrit"), arg_int(2), arg_none(), arg_arr(mc1, 1, s1)}, 1, r) == TSR_EARG,
          "fcluster monocrit shorter than Z refused (read past it)");
    CHECK(fzb_call("hierarchy.fcluster", 5, (tsr_arg[]){zok, arg_num(1.0), arg_str("inconsistent"), arg_int(2), arg_arr(R1, 2, r1)}, 1, r) == TSR_EARG,
          "fcluster with an R of fewer rows than Z refused");
    CHECK(fzb_call("hierarchy.fcluster", 5, (tsr_arg[]){zok, arg_num(1.0), arg_str("inconsistent"), arg_int(2), arg_arr(Rneg, 2, z2)}, 1, r) == TSR_EARG,
          "fcluster with a negative standard deviation in R refused");
    CHECK(fzb_call("hierarchy.fcluster", 4, (tsr_arg[]){zok, arg_num(1.0), arg_str("inconsistent"), arg_num(NAN)}, 1, r) == TSR_EARG, "fcluster depth=NaN refused, no cast");
    CHECK(fzb_call("hierarchy.fcluster", 4, (tsr_arg[]){zok, arg_num(1.0), arg_str("inconsistent"), arg_int(0)}, 1, r) == 0, "fcluster depth=0 runs (SciPy: the node alone)");
    CHECK(fzb_call("hierarchy.fcluster", 3, (tsr_arg[]){zok, arg_num(2.5), arg_str("maxclust")}, 1, r) == TSR_ETYPE, "fcluster maxclust t=2.5 is a TypeError");
    CHECK(fzb_call("hierarchy.fcluster", 3, (tsr_arg[]){zok, arg_num(NAN), arg_str("maxclust_monocrit")}, 1, r) == TSR_EARG ||
          fzb_call("hierarchy.fcluster", 3, (tsr_arg[]){zok, arg_num(NAN), arg_str("maxclust_monocrit")}, 1, r) == TSR_ETYPE, "fcluster maxclust_monocrit t=NaN refused");
    CHECK(fzb_call("hierarchy.fcluster", 3, (tsr_arg[]){zok, arg_num(1.0), arg_str("bogus")}, 1, r) == TSR_EARG, "fcluster invalid criterion");
    CHECK(fzb_call("hierarchy.maxinconsts", 2, (tsr_arg[]){zok, arg_arr(R1, 2, r1)}, 2, r) == TSR_EARG, "maxinconsts with R of fewer rows refused (read past R)");
    CHECK(fzb_call("hierarchy.maxRstat", 3, (tsr_arg[]){zok, arg_arr(Zok, 2, z2), arg_num(1.0)}, 3, r) == TSR_ETYPE, "maxRstat with a float i is a TypeError");
    double Tneg[3] = {-1, -1, 5}, Tfrac[3] = {1, 1.5, 2};
    if (fzc_run("hierarchy.leaders", 2, (tsr_arg[]){zok, arg_arr(Tneg, 1, s3)}, 2, r) == 0) {
        CHECK(r[0].arr.shape[0] == 2, "leaders with label -1 (was the internal 'mixed' marker)");
        free_results(r, 2);
    } else CHECK(0, "leaders with -1 labels failed: %s", tsr_fn_error());
    CHECK(fzb_call("hierarchy.leaders", 2, (tsr_arg[]){zok, arg_arr(Tfrac, 1, s3)}, 2, r) == TSR_ETYPE, "leaders with a non-integer label is a TypeError");
    double I1[3] = {-5, -5, 3}, I2[3] = {7, 7, -2}, I3[3] = {7, 8, -2};
    if (fzc_run("hierarchy.is_isomorphic", 2, (tsr_arg[]){arg_arr(I1, 1, s3), arg_arr(I2, 1, s3)}, 2, r) == 0) {
        CHECK(r[0].num == 1.0, "is_isomorphic with negative labels (indexed d1[-5])");
        free_results(r, 2);
    } else CHECK(0, "is_isomorphic failed: %s", tsr_fn_error());
    if (fzc_run("hierarchy.is_isomorphic", 2, (tsr_arg[]){arg_arr(I1, 1, s3), arg_arr(I3, 1, s3)}, 2, r) == 0) {
        CHECK(r[0].num == 0.0, "is_isomorphic: a split label is not isomorphic");
        free_results(r, 2);
    } else CHECK(0, "is_isomorphic (2) failed: %s", tsr_fn_error());
    CHECK(fzb_call("hierarchy.is_isomorphic", 2, (tsr_arg[]){arg_arr(I1, 1, s3), arg_arr(I1, 1, s2)}, 2, r) == TSR_EARG, "is_isomorphic lengths differ: SciPy's ValueError");
    double M99[6] = {1, 2, 1, 99, 3, 2};
    int64_t m3[2] = {2, 3};
    CHECK(fzb_call("hierarchy.from_mlab_linkage", 1, (tsr_arg[]){arg_arr(M99, 2, m3)}, 1, r) == TSR_EARG, "from_mlab_linkage with id 99 refused (indexed sz[98])");
    double X1[2] = {0, 1}, ynan[3] = {1, NAN, 2};
    int64_t x1[2] = {1, 2};
    CHECK(fzb_call("hierarchy.linkage", 1, (tsr_arg[]){arg_arr(X1, 2, x1)}, 1, r) == TSR_EARG, "linkage of one observation refused (was a (-1, 4) Z)");
    CHECK(fzb_call("hierarchy.linkage", 1, (tsr_arg[]){arg_arr(X1, 1, (int64_t[]){0})}, 1, r) == TSR_EARG, "linkage of an empty condensed matrix refused");
    CHECK(fzb_call("hierarchy.linkage", 1, (tsr_arg[]){arg_arr(ynan, 1, s3)}, 1, r) == TSR_EARG, "linkage with a NaN distance refused (SciPy)");
    CHECK(fzb_call("hierarchy.fclusterdata", 2, (tsr_arg[]){arg_arr(X1, 2, x1), arg_num(1.0)}, 1, r) == TSR_EARG, "fclusterdata of one observation refused");
    double X3[6] = {0, 0, 1, 0, 5, 5};
    int64_t x3[2] = {3, 2};
    CHECK(fzb_call("hierarchy.fclusterdata", 3, (tsr_arg[]){arg_arr(X3, 2, x3), arg_num(1.0), arg_str("monocrit")}, 1, r) == TSR_EARG,
          "fclusterdata criterion=monocrit refused (read a NULL monocrit)");
    CHECK(fzb_call("hierarchy.cut_tree", 3, (tsr_arg[]){zok, arg_int(2), arg_arr(Zok, 1, s1)}, 1, r) == TSR_EARG, "cut_tree with both n_clusters and height refused");
    CHECK(fzb_call("hierarchy.cut_tree", 2, (tsr_arg[]){zok, arg_num(NAN)}, 1, r) == 0, "cut_tree n_clusters=NaN runs (searchsorted puts NaN last), no cast");
    /* a 3000-merge chain at a depth beyond it: the inconsistency statistic walks it iteratively */
    enum { CH = 3000 };
    double *Zc = (double *)malloc(sizeof(double) * 4 * CH);
    if (Zc) {
        for (int i = 0; i < CH; i++) { Zc[i * 4] = i == 0 ? 0 : CH + i; Zc[i * 4 + 1] = i + 1; Zc[i * 4 + 2] = i + 1; Zc[i * 4 + 3] = i + 2; }
        if (fzc_run("hierarchy.inconsistent", 2, (tsr_arg[]){arg_arr(Zc, 2, (int64_t[]){CH, 4}), arg_int(INT32_MAX)}, 1, r) == 0) {
            const double *R = (const double *)r[0].arr.data;
            CHECK(R[(CH - 1) * 4 + 2] == CH && R[2] == 1, "inconsistent on a 3000-deep chain counts every merge below the root");
            free_results(r, 1);
        } else CHECK(0, "inconsistent chain failed: %s", tsr_fn_error());
        free(Zc);
    }

    /* C-N1 / C-N2 (batch C campaign): PPoly and BPoly took x and the shape of c unchecked. A short x was read past
       its end (pp_interval, x[i+1]); m = 0 made pp_interval return -1; an empty first axis of c wrote binom[0]
       into a 0-byte block. SciPy's _PPolyBase refusals now come first. */
    {
        double pc[4] = {2.0, 1.0, 3.0, 4.0}, px2[2] = {0.0, 1.0}, px3[3] = {0.0, 1.0, 2.0}, pxn[3] = {0.0, 2.0, 1.0};
        double pxnan[2] = {0.0, NAN}, pq[1] = {0.5};
        const int64_t s21[2] = {2, 1}, s12[2] = {1, 2}, s01[2] = {0, 1}, s22[2] = {2, 2}, n1[1] = {1}, n2[1] = {2}, n3[1] = {3};
        const tsr_arg q = arg_arr(pq, 1, n1);
        CHECK(fzb_call("interpolate.BPoly", 3, (tsr_arg[]){arg_arr(pc, 2, s12), arg_arr(px2, 1, n2), q}, 1, r) == TSR_EARG,
              "C-N1 BPoly with 2 intervals in c but 2 breakpoints refused (read past x)");
        CHECK(fzb_call("interpolate.BPoly", 3, (tsr_arg[]){arg_arr(pc, 2, s01), arg_arr(px2, 1, n2), q}, 1, r) == TSR_EARG,
              "C-N2 BPoly with an empty coefficient axis refused (write past binom)");
        CHECK(fzb_call("interpolate.PPoly", 3, (tsr_arg[]){arg_arr(pc, 2, s21), arg_arr(px2, 1, n1), q}, 1, r) == TSR_EARG,
              "PPoly with one breakpoint refused");
        CHECK(fzb_call("interpolate.PPoly", 3, (tsr_arg[]){arg_arr(pc, 2, s22), arg_arr(pxn, 1, n3), q}, 1, r) == TSR_EARG,
              "PPoly with non-monotonic x refused");
        CHECK(fzb_call("interpolate.PPoly", 3, (tsr_arg[]){arg_arr(pc, 2, s21), arg_arr(pxnan, 1, n2), q}, 1, r) == TSR_EARG,
              "PPoly with a NaN breakpoint refused");
        if (fzc_run("interpolate.PPoly", 3, (tsr_arg[]){arg_arr(pc, 2, s21), arg_arr(px2, 1, n2), q}, 1, r) == 0) {
            CHECK(((const double *)r[0].arr.data)[0] == 2.0, "PPoly([[2],[1]], [0,1])(0.5) = 2");
            free_results(r, 1);
        } else CHECK(0, "PPoly valid failed: %s", tsr_fn_error());
        /* C-N3 (batch C campaign): a one-row Z naming the same leaf twice reached cophenet's pair index -1 */
        double zsame[4] = {0.0, 0.0, 1.0, 2.0};
        CHECK(fzb_call("hierarchy.cophenet", 1, (tsr_arg[]){arg_arr(zsame, 2, (int64_t[]){1, 4})}, 1, r) == TSR_EARG,
              "C-N3 cophenet on a one-row Z joining leaf 0 with itself refused (wrote out[-1])");
        if (fzc_run("interpolate.BPoly", 3,(tsr_arg[]){arg_arr(pc, 2, s22), arg_arr(px3, 1, n3), q}, 1, r) == 0) {
            CHECK(((const double *)r[0].arr.data)[0] == 2.5, "BPoly([[2,1],[3,4]], [0,1,2])(0.5) = 2*(1-s) + 3*s = 2.5");
            free_results(r, 1);
        } else CHECK(0, "BPoly valid failed: %s", tsr_fn_error());
    }
    CHECK(tsr_allocated() == before, "fuzz batch C leaves no kernel allocation behind (%lld bytes)", (long long)(tsr_allocated() - before));
}

/* batch D: csgraph node indices and CSR arrays (C-O1, the CI csg_bfs_pred overflow), the BiasedUrn recursion's
   pointer arithmetic, and the specfun int overflows reached through hyperu and the spheroidal radial functions.
   Under `make sanitize` the calls below would stop at the old undefined behaviour. */
static int bd_ufunc(const char *name, int method, int nin, int nout, double *in, int64_t n, double *out)
{
    const int id = tsr_fn_find(name);
    if (id < 0) return -999;
    void *data[16]; int64_t strides[16], shape[1] = {n};
    for (int k = 0; k < nin; k++) { data[k] = in + k * n; strides[k] = 8; }
    for (int k = 0; k < nout; k++) { data[nin + k] = out + k * n; strides[nin + k] = 8; }
    return tsr_fn_ufunc(id, method, 0, 1, shape, nin + nout, data, strides);
}

static void test_fuzz_batch_d(void)
{
    tsr_result r[4];
    const int64_t before = tsr_allocated();
    double g[9] = {0, 1, 0, 0, 0, 1, 1, 0, 0};                     /* 3-node cycle */
    int64_t s33[2] = {3, 3}, s13[2] = {1, 3}, s3[1] = {3}, s2[1] = {2};
    const tsr_arg G = arg_arr(g, 2, s33);
    const char *starts[3] = {"csgraph.breadth_first_order", "csgraph.depth_first_order", "csgraph.breadth_first_tree"};
    for (int k = 0; k < 3; k++) {
        tsr_arg a[2] = {G, arg_int(5)};
        CHECK(fa_call(starts[k], 2, a, r, 2) == TSR_EARG, "%s with i_start past the nodes refused", starts[k]);
        a[1] = arg_int(-1);
        CHECK(fa_call(starts[k], 2, a, r, 2) == TSR_EARG, "%s with a negative i_start refused", starts[k]);
        a[1] = arg_num(NAN);
        CHECK(fa_call(starts[k], 2, a, r, 2) < 0, "%s with i_start = NaN refused", starts[k]);
        a[1] = arg_int(2);
        const int rc = fa_call(starts[k], 2, a, r, 2);
        CHECK(rc == 0, "%s from node 2 still works (rc %d)", starts[k], rc);
        if (rc == 0) fa_free(r, 2);
    }
    tsr_arg dft[2] = {G, arg_int(3)};
    CHECK(fa_call("csgraph.depth_first_tree", 2, dft, r, 1) == TSR_EARG, "depth_first_tree with i_start = n refused");

    /* csgraph_to_dense: indptr past the data, decreasing, not starting at 0; n_nodes beyond n*n */
    double dat[2] = {1, 2}, ind[2] = {0, 1}, ptr_bad[4] = {0, 1, 9, 9}, ptr_dec[4] = {0, 2, 1, 2}, ptr_off[4] = {1, 1, 2, 2};
    int64_t s4[1] = {4};
    tsr_arg cd[4] = {arg_arr(dat, 1, s2), arg_arr(ind, 1, s2), arg_arr(ptr_bad, 1, s4), arg_int(3)};
    CHECK(fa_call("csgraph.csgraph_to_dense", 4, cd, r, 1) == TSR_EARG, "csgraph_to_dense with indptr past the data refused");
    cd[2] = arg_arr(ptr_dec, 1, s4);
    CHECK(fa_call("csgraph.csgraph_to_dense", 4, cd, r, 1) == TSR_EARG, "csgraph_to_dense with a decreasing indptr refused");
    cd[2] = arg_arr(ptr_off, 1, s4);
    CHECK(fa_call("csgraph.csgraph_to_dense", 4, cd, r, 1) == TSR_EARG, "csgraph_to_dense with indptr[0] != 0 refused");
    cd[3] = arg_num(1e19);
    CHECK(fa_call("csgraph.csgraph_to_dense", 4, cd, r, 1) < 0, "csgraph_to_dense with a huge n_nodes refused");
    double ptr_ok[4] = {0, 1, 2, 2};
    cd[2] = arg_arr(ptr_ok, 1, s4); cd[3] = arg_int(3);
    int rc = fa_call("csgraph.csgraph_to_dense", 4, cd, r, 1);
    CHECK(rc == 0 && ((double *)r[0].arr.data)[0] == 1.0 && ((double *)r[0].arr.data)[4] == 2.0, "csgraph_to_dense still builds (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);

    /* predecessors that do not match the graph were read past their end */
    double p13[3] = {-9999, 0, 1}, p2[2] = {-9999, 0};
    tsr_arg cdm[2] = {G, arg_arr(p13, 2, s13)};
    CHECK(fa_call("csgraph.construct_dist_matrix", 2, cdm, r, 1) == TSR_EARG, "construct_dist_matrix with 1x3 predecessors for 3 nodes refused");
    tsr_arg rp[2] = {G, arg_arr(p2, 1, s2)};
    CHECK(fa_call("csgraph.reconstruct_path", 2, rp, r, 1) == TSR_EARG, "reconstruct_path with 2 predecessors for 3 nodes refused");
    rp[1] = arg_arr(p13, 1, s3);
    rc = fa_call("csgraph.reconstruct_path", 2, rp, r, 1);
    CHECK(rc == 0, "reconstruct_path with matching predecessors still works (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);

    /* maximum_flow / yen: no float->int cast of NaN or non-integral capacities and indices */
    double cap[9] = {0, 3, 0, 0, 0, 2, 0, 0, 0};
    tsr_arg mf[3] = {arg_arr(cap, 2, s33), arg_int(0), arg_int(2)};
    rc = fa_call("csgraph.maximum_flow", 3, mf, r, 2);
    CHECK(rc == 0 && fa_num(&r[0]) == 2.0, "maximum_flow 0->2 is 2 (rc %d)", rc);
    if (rc == 0) fa_free(r, 2);
    mf[1] = arg_num(NAN);
    CHECK(fa_call("csgraph.maximum_flow", 3, mf, r, 2) < 0, "maximum_flow with source = NaN refused");
    cap[1] = 1.5; mf[1] = arg_int(0);
    CHECK(fa_call("csgraph.maximum_flow", 3, mf, r, 2) == TSR_EARG, "maximum_flow with a non-integral capacity refused");
    cap[1] = NAN;
    CHECK(fa_call("csgraph.maximum_flow", 3, mf, r, 2) == TSR_EARG, "maximum_flow with a NaN capacity refused");
    tsr_arg ye[4] = {G, arg_int(0), arg_int(2), arg_num(NAN)};
    CHECK(fa_call("csgraph.yen", 4, ye, r, 2) < 0, "yen with K = NaN refused");

    /* BiasedUrn (Wallenius): pmf through the recursion (lower limit moving) and cdf through the table method.
       On x86-64 the values are the pre-fix bits (dist_disc.cpp now indexes instead of stepping pointers below the
       buffer). aarch64 compilers contract a*b+c into one FMA (the kernel is built with -ffp-contract=off only for
       the reproducible-f32 and numpy-random code), so two values differ there in the last bits; Linux and macOS
       aarch64 agree, and the aarch64 pin guards against drift (no pre-fix aarch64 bits were recorded). */
    enum { W = 4 };
    double win[6 * W], wout[W];
    for (int i = 0; i < W; i++) { win[i] = 2 + i; win[W + i] = 60; win[2 * W + i] = 30; win[3 * W + i] = 25; win[4 * W + i] = 0.1 + i; win[5 * W + i] = 0; }
    rc = bd_ufunc("stats.nchypergeom_wallenius", 0, 6, 1, win, W, wout);
#if defined(__aarch64__)
    const double wexp0 = 0x1.6f3fc1ff1f474p-3, wexp3 = 0x1.cfdd2c2779e1dp-35;
#else
    const double wexp0 = 0x1.6f3fc1ff1f476p-3, wexp3 = 0x1.cfdd2c2779e1ep-35;
#endif
    CHECK(rc == 0 && wout[0] == wexp0 && wout[1] == 0x1.2cb5830c138e2p-23 && wout[2] == 0x1.203a553569e89p-31
          && wout[3] == wexp3, "nchypergeom_wallenius.pmf through the recursion keeps its bits (rc %d, %a %a %a %a)", rc, wout[0], wout[1], wout[2], wout[3]);
    double wsum = 0, wsx[6 * 26], wso[26];
    for (int i = 0; i <= 25; i++) { wsx[i] = i; wsx[26 + i] = 60; wsx[52 + i] = 30; wsx[78 + i] = 25; wsx[104 + i] = 2.5; wsx[130 + i] = 0; }
    rc = bd_ufunc("stats.nchypergeom_wallenius", 0, 6, 1, wsx, 26, wso);
    for (int i = 0; i <= 25; i++) wsum += wso[i];
    CHECK(rc == 0 && fabs(wsum - 1.0) < 1e-9, "nchypergeom_wallenius pmf sums to 1 over its support (%.17g)", wsum);
    rc = bd_ufunc("stats.nchypergeom_wallenius", 2, 6, 1, wsx, 26, wso);
    CHECK(rc == 0 && fabs(wso[25] - 1.0) < 1e-9 && wso[10] <= wso[20], "nchypergeom_wallenius cdf (table method) (rc %d)", rc);

    /* specfun: chgubi's digit estimate (hyperu with huge a) and the rmn1/rmn2l work size (c = -inf) were signed
       int overflows on (int) of a non-finite value; the results keep SciPy's x86-64 bits */
    double hin[3 * 3] = {1e300, 1e200, 1, 2, 1, 1, 1, 2, 0.5}, hout[3];
    rc = bd_ufunc("special.hyperu", 0, 3, 1, hin, 3, hout);
    CHECK(rc == 0 && isnan(hout[0]) && isnan(hout[1]) && hout[2] == 0x1.d887be0f4bed8p-1, "hyperu with huge a is NaN, hyperu(1, 1, 0.5) keeps its bits (rc %d)", rc);
    double pin[4 * 2] = {0, 1, 1, 3, -INFINITY, 2.5, 1.5, 1.2}, pout[2 * 2];
    const char *rad[4] = {"special.pro_rad1", "special.pro_rad2", "special.obl_rad1", "special.obl_rad2"};
    /* c = -inf is outside the domain. SciPy's x86-64 result is (0, nan), an artifact of x86's conversion of a NaN to
       int (the parity fixtures record it, so x86-64 keeps it); on aarch64 the second-kind radial functions give
       (nan, nan) for the same input. Pinned per architecture, like yn (platform-debt PD-20). */
    for (int k = 0; k < 4; k++) {
        rc = bd_ufunc(rad[k], 0, 4, 2, pin, 2, pout);
#if defined(__aarch64__)
        const int rad_ok = (k == 1 || k == 3) ? isnan(pout[0]) && isnan(pout[2]) : pout[0] == 0.0 && isnan(pout[2]);
#else
        const int rad_ok = pout[0] == 0.0 && isnan(pout[2]);
#endif
        CHECK(rc == 0 && rad_ok, "%s at c = -inf keeps its pinned value (rc %d; got %a %a %a %a)", rad[k], rc, pout[0], pout[1], pout[2], pout[3]);
    }

    /* min_weight_full_bipartite_matching: NaN/inf weights broke the potentials and the unmatched p[j] = 0 wrote at
       index -1 (fuzz D-N1); an infeasible graph returned a "matching" through missing edges */
    double bw[4] = {1, 2, 3, 4};
    tsr_arg bm[1] = {arg_arr(bw, 2, (int64_t[]){2, 2})};
    rc = fa_call("csgraph.min_weight_full_bipartite_matching", 1, bm, r, 2);
    CHECK(rc == 0 && ((int64_t *)r[1].arr.data)[0] == 0 && ((int64_t *)r[1].arr.data)[1] == 1, "bipartite matching of [[1,2],[3,4]] (rc %d)", rc);
    if (rc == 0) fa_free(r, 2);
    bw[1] = NAN;
    CHECK(fa_call("csgraph.min_weight_full_bipartite_matching", 1, bm, r, 2) == TSR_EARG, "bipartite matching with a NaN weight refused");
    bw[1] = INFINITY;
    CHECK(fa_call("csgraph.min_weight_full_bipartite_matching", 1, bm, r, 2) == TSR_EARG, "bipartite matching with an infinite weight refused");
    double bz[4] = {1, 2, 0, 0};                                         /* row 1 has no edge */
    bm[0] = arg_arr(bz, 2, (int64_t[]){2, 2});
    CHECK(fa_call("csgraph.min_weight_full_bipartite_matching", 1, bm, r, 2) == TSR_EARG, "bipartite matching without a full matching refused");

    /* invpascal: O(n^4) symmetric / O(n^3) triangular work timed out the CI fuzz run (seed 1); now one binomial row
       per k with sl_binom's exact arithmetic (same bits) and a bound on the symmetric work */
    tsr_arg ip[2] = {arg_int(8), arg_str("symmetric")};
    rc = fa_call("slinalg.invpascal", 2, ip, r, 1);
    CHECK(rc == 0 && ((double *)r[0].arr.data)[3 * 8 + 5] == 0x1.b08p+9 && ((double *)r[0].arr.data)[5 * 8 + 2] == -0x1.0e8p+9,
          "invpascal(8) symmetric keeps its values (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    ip[1] = arg_str("upper");
    rc = fa_call("slinalg.invpascal", 2, ip, r, 1);
    /* scipy's default exact=True result is int64: its zeros are plain zeros (exact=False would give -0.0 where
       (-1)^(i+j) < 0; Tessero computes the exact=True values, as the parity fixtures check) */
    CHECK(rc == 0 && ((double *)r[0].arr.data)[3 * 8 + 5] == 0x1.4p+3 && ((double *)r[0].arr.data)[5 * 8 + 2] == 0.0
          && !signbit(((double *)r[0].arr.data)[5 * 8 + 2]), "invpascal(8) upper keeps its values, +0.0 below the diagonal (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    ip[0] = arg_int(1861); ip[1] = arg_str("symmetric");
    CHECK(fa_call("slinalg.invpascal", 2, ip, r, 1) == TSR_EARG, "invpascal(1861) symmetric refused (work bound)");
    ip[0] = arg_int(3000); ip[1] = arg_str("lower");
    rc = fa_call("slinalg.invpascal", 2, ip, r, 1);
    CHECK(rc == 0 && ((double *)r[0].arr.data)[1 * 3000 + 0] == -1.0, "invpascal(3000) lower is computed (O(n^2)) (rc %d)", rc);
    if (rc == 0) fa_free(r, 1);
    CHECK(tsr_allocated() == before, "batch D leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* tsr_unary_find / tsr_binary_find: every name maps to its code and back; the codes are the ones of ops.h */
static void test_op_names(void)
{
    int ok = tsr_unary_find("exp") == U_EXP && tsr_unary_find("tanh") == U_TANH && tsr_unary_find("conjugate") == U_CONJ
          && tsr_unary_find("abs") == U_ABS && tsr_unary_find("conj") == U_CONJ && tsr_unary_find("nope") == -1 && tsr_unary_find(NULL) == -1
          && tsr_binary_find("add") == OP_ADD && tsr_binary_find("floor_divide") == OP_FLOORDIV && tsr_binary_find("true_divide") == OP_DIV
          && tsr_binary_find("remainder") == OP_MOD && tsr_binary_find("nope") == -1 && tsr_unary_name(-1) == NULL && tsr_binary_name(20) == NULL;
    for (int op = 0; op <= U_CONJ; op++) ok = ok && tsr_unary_name(op) && tsr_unary_find(tsr_unary_name(op)) == op;
    for (int op = 0; op <= OP_FLOORDIV; op++) ok = ok && tsr_binary_name(op) && tsr_binary_find(tsr_binary_name(op)) == op;
    ok = ok && tsr_unary_name(U_CONJ + 1) == NULL;
    CHECK(ok, "op code names: tsr_unary_find / tsr_binary_find and back");
}

/* tsr_cholesky / tsr_solve / tsr_slogdet run the registry routines: the same bytes as calling them through the registry */
static void test_linalg_typed(void)
{
    enum { N = 6, K = 3 };
    double m[N * N], spd[N * N], b[N * K], l[N * N], x[N * K], sgn, lad;
    for (int i = 0; i < N * N; i++) m[i] = sin(0.7 * i + 1.0);
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        double s = i == j ? (double)N : 0.0;
        for (int k = 0; k < N; k++) s += m[i * N + k] * m[j * N + k];
        spd[i * N + j] = s;
    }
    for (int i = 0; i < N * K; i++) b[i] = cos(0.3 * i);
    const int64_t before = tsr_allocated();
    int64_t sa[2] = {N, N}, sb[2] = {N, K};
    tsr_result r[2];
    int ok = tsr_cholesky(N, spd, l) == TSR_OK;
    tsr_arg ca[1] = {arg_arr(spd, 2, sa)};
    ok = ok && tsr_fn_routine(tsr_fn_find("linalg.cholesky"), 1, ca, 1, r) == 0 && !memcmp(r[0].arr.data, l, sizeof l);
    tsr_fn_results_free(r, 1);
    CHECK(ok, "tsr_cholesky == linalg.cholesky, byte for byte");
    ok = tsr_solve(N, K, spd, b, x) == TSR_OK;
    tsr_arg sa2[2] = {arg_arr(spd, 2, sa), arg_arr(b, 2, sb)};
    ok = ok && tsr_fn_routine(tsr_fn_find("linalg.solve"), 2, sa2, 1, r) == 0 && !memcmp(r[0].arr.data, x, sizeof x);
    tsr_fn_results_free(r, 1);
    CHECK(ok, "tsr_solve == linalg.solve, byte for byte");
    ok = tsr_slogdet(N, m, &sgn, &lad) == TSR_OK;
    tsr_arg da[1] = {arg_arr(m, 2, sa)};
    ok = ok && tsr_fn_routine(tsr_fn_find("linalg.slogdet"), 1, da, 2, r) == 0 && r[0].num == sgn && r[1].num == lad;
    tsr_fn_results_free(r, 2);
    CHECK(ok, "tsr_slogdet == linalg.slogdet (sign %g, log|det| %g)", sgn, lad);
    double notpd[4] = {1.0, 2.0, 2.0, 1.0}, l2[4];
    CHECK(tsr_cholesky(2, notpd, l2) == TSR_EARG, "tsr_cholesky refuses a matrix that is not positive definite");
    CHECK(tsr_allocated() == before, "the typed linalg calls leave no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* the iterative solvers on a CSR matrix (the group data, indices, indptr, shape) return what they return on the same
   matrix dense: the CSR product sums the same products, without the dense one's exact zeros. Malformed CSR groups are
   refused before any use. */
static void test_splinalg_csr(void)
{
    enum { N = 60 };
    static const char *const solvers[] = {"splinalg.cg", "splinalg.bicgstab", "splinalg.bicg", "splinalg.cgs",
                                          "splinalg.minres", "splinalg.gmres", "splinalg.qmr", "splinalg.tfqmr"};
    double dense[N * N], b[N], data[N * 7];
    int64_t ind[N * 7], ptr[N + 1], shp[2] = {N, N}, nnz = 0;
    uint64_t st = 777;
    memset(dense, 0, sizeof dense);
    for (int i = 0; i < N; i++) {                         /* symmetric, diagonally dominant: every solver converges */
        for (int d = 1; d <= 3; d++) {
            const int j = (i + d * 7) % N;
            if (j == i) continue;
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            const double v = -0.5 * (double)(st >> 11) / 9007199254740992.0;
            dense[i * N + j] = dense[j * N + i] = v;
        }
    }
    for (int i = 0; i < N; i++) {
        double s = 1.0;
        for (int j = 0; j < N; j++) s += fabs(dense[i * N + j]);
        dense[i * N + i] = s;
        b[i] = sin(0.37 * i) + 0.1;
    }
    ptr[0] = 0;
    for (int i = 0; i < N; i++) {                         /* CSR of the same matrix, columns ascending */
        for (int j = 0; j < N; j++) if (dense[i * N + j] != 0.0) { data[nnz] = dense[i * N + j]; ind[nnz] = j; nnz++; }
        ptr[i + 1] = nnz;
    }
    const int64_t before = tsr_allocated();
    int64_t sd[2] = {N, N}, sb[1] = {N}, sn[1] = {nnz}, sp[1] = {N + 1}, s2[1] = {2};
    int same = 0, ran = 0;
    char why[256] = "";                                   /* the first solver that differs, and how (a failure's evidence) */
    for (size_t s = 0; s < sizeof solvers / sizeof solvers[0]; s++) {
        const int id = tsr_fn_find(solvers[s]);
        if (id < 0) continue;
        tsr_arg items[4] = {arg_arr(data, 1, sn), arg_arr_dt(ind, TSR_I64, 1, sn), arg_arr_dt(ptr, TSR_I64, 1, sp), arg_arr_dt(shp, TSR_I64, 1, s2)};
        tsr_arg grp; memset(&grp, 0, sizeof grp); grp.kind = 5; grp.count = 4; grp.items = items;
        tsr_arg ad[4] = {arg_arr(dense, 2, sd), arg_arr(b, 1, sb), {0}, arg_num(1e-10)};
        tsr_arg ac[4] = {grp, arg_arr(b, 1, sb), {0}, arg_num(1e-10)};
        tsr_result rd[2], rc2[2];
        ran++;
        const int rcd = tsr_fn_routine(id, 4, ad, 2, rd);
        if (rcd != 0) {
            if (!why[0]) snprintf(why, sizeof why, "; first: %s dense rc %d (%s)", solvers[s], rcd, tsr_fn_error());
            continue;
        }
        const int rcc = tsr_fn_routine(id, 4, ac, 2, rc2);
        if (rcc == 0) {
            const double *xd = rd[0].arr.data, *xc = rc2[0].arr.data;
            int eq = rd[0].arr.shape[0] == N && rc2[0].arr.shape[0] == N;
            double maxdiff = 0.0;
            for (int i = 0; eq && i < N; i++) maxdiff = fmax(maxdiff, fabs(xd[i] - xc[i]));
            for (int i = 0; eq && i < N; i++) eq = xd[i] == xc[i];   /* equal values; only a zero's sign may differ */
            double res = 0.0;                               /* and a solution: |A x - b| small */
            for (int i = 0; i < N && rc2[0].arr.shape[0] == N; i++) { double r = -b[i]; for (int j = 0; j < N; j++) r += dense[i * N + j] * xc[j]; res = fmax(res, fabs(r)); }
            same += eq && res < 1e-6;
            if (!(eq && res < 1e-6) && !why[0])
                snprintf(why, sizeof why, "; first: %s max |dense - csr| %.3g, residual %.3g", solvers[s], maxdiff, res);
            tsr_fn_results_free(rc2, 2);
        } else if (!why[0]) snprintf(why, sizeof why, "; first: %s csr rc %d (%s)", solvers[s], rcc, tsr_fn_error());
        tsr_fn_results_free(rd, 2);
    }
    CHECK(ran == 8 && same == ran, "iterative solvers on CSR == on the same matrix dense (%d of %d%s)", same, ran, why);

    /* malformed CSR groups: refused, nothing leaked */
    const int id = tsr_fn_find("splinalg.cg");
    int refused = 0, trials = 0;
    for (int forge = 0; forge < 7; forge++) {
        int64_t i2[N * 7], p2[N + 1], sh2[2] = {N, N};
        double fi[N * 7];
        memcpy(i2, ind, (size_t)nnz * 8); memcpy(p2, ptr, sizeof p2);
        tsr_arg items[4] = {arg_arr(data, 1, sn), arg_arr_dt(i2, TSR_I64, 1, sn), arg_arr_dt(p2, TSR_I64, 1, sp), arg_arr_dt(sh2, TSR_I64, 1, s2)};
        tsr_arg grp; memset(&grp, 0, sizeof grp); grp.kind = 5; grp.count = 4; grp.items = items;
        if (forge == 0) i2[5] = N;                                     /* a column index out of range */
        if (forge == 1) { p2[10] = p2[11] + 1; }                      /* indptr decreasing */
        if (forge == 2) p2[N] = nnz + 1;                               /* indptr past the data */
        if (forge == 3) sh2[1] = N + 1;                                /* not square */
        if (forge == 4) grp.count = 3;                                 /* three parts */
        if (forge == 5) { for (int64_t k = 0; k < nnz; k++) fi[k] = (double)ind[k] + 0.5; items[1] = arg_arr(fi, 1, sn); }   /* non-integer */
        if (forge == 6) p2[0] = 1;                                     /* indptr not from 0 */
        tsr_arg ac[2] = {grp, arg_arr(b, 1, sb)};
        tsr_result r[2];
        trials++;
        refused += tsr_fn_routine(id, 2, ac, 2, r) == TSR_EARG;
    }
    CHECK(refused == trials, "the iterative solvers refuse malformed CSR matrices (%d of %d refused)", refused, trials);
    CHECK(tsr_allocated() == before, "the CSR solvers leave no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* spsolve on CSR: the banded LU (or the dense fallback for a wide band) solves A x = b to rounding, matches the dense
   solve, handles several right-hand sides, and reports a singular matrix */
static void spsolve_csr_case(int64_t n, int band, int nrhs, int singular, int *ok, int *refused)
{
    const int id = tsr_fn_find("splinalg.spsolve");
    double *data = malloc((size_t)(n * 9) * 8), *dense = n <= 400 ? calloc((size_t)(n * n), 8) : NULL;
    int64_t *ind = malloc((size_t)(n * 9) * 8), *ptr = malloc((size_t)(n + 1) * 8), nnz = 0, shp[2] = {n, n};
    double *b = malloc((size_t)(n * nrhs) * 8);
    uint64_t st = 99 + (uint64_t)n;
    ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        const int64_t offs[5] = {-band, -1, 0, 1, band};
        for (int o = 0; o < 5; o++) {
            const int64_t j = band > 2 && (o == 0 || o == 4) ? (i + offs[o] * 7 + n) % n : i + offs[o];   /* band > 2: wide, scattered */
            if (j < 0 || j >= n || (o == 0 && band <= 1) || (o == 4 && band <= 1)) continue;
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            double v = o == 2 ? 6.0 : -1.0 + 0.5 * (double)(st >> 11) / 9007199254740992.0;
            if (singular && i == n - 1) v = 0.0;                      /* the last row all zero */
            data[nnz] = v; ind[nnz] = j; nnz++;
            if (dense) dense[i * n + j] += v;
        }
        ptr[i + 1] = nnz;
    }
    for (int64_t i = 0; i < n * nrhs; i++) b[i] = cos(0.1 * i);
    int64_t sn[1] = {nnz}, sp[1] = {n + 1}, s2[1] = {2}, sb1[1] = {n}, sb2[2] = {n, nrhs};
    tsr_arg items[4] = {arg_arr(data, 1, sn), arg_arr_dt(ind, TSR_I64, 1, sn), arg_arr_dt(ptr, TSR_I64, 1, sp), arg_arr_dt(shp, TSR_I64, 1, s2)};
    tsr_arg grp; memset(&grp, 0, sizeof grp); grp.kind = 5; grp.count = 4; grp.items = items;
    tsr_arg a[2] = {grp, nrhs == 1 ? arg_arr(b, 1, sb1) : arg_arr(b, 2, sb2)};
    tsr_result r[1];
    const int rc = tsr_fn_routine(id, 2, a, 1, r);
    if (singular) { *refused += rc == TSR_EARG; if (rc == 0) tsr_fn_results_free(r, 1); }
    else if (rc == 0) {
        const double *x = r[0].arr.data;
        double res = 0.0, scale = 0.0;                                /* max |A x - b| over the columns */
        for (int c = 0; c < nrhs; c++)
            for (int64_t i = 0; i < n; i++) {
                double s = -b[i * nrhs + c];
                for (int64_t q = ptr[i]; q < ptr[i + 1]; q++) s += data[q] * x[ind[q] * nrhs + c];
                res = fmax(res, fabs(s)); scale = fmax(scale, fabs(b[i * nrhs + c]));
            }
        int good = res <= 1e-12 * scale;
        if (good && dense) {                                          /* and the dense solve's answer */
            int64_t sd[2] = {n, n};
            tsr_arg ad[2] = {arg_arr(dense, 2, sd), a[1]};
            tsr_result rd[1];
            if (tsr_fn_routine(id, 2, ad, 1, rd) == 0) {
                const double *xd = rd[0].arr.data;
                for (int64_t i = 0; good && i < n * nrhs; i++) good = fabs(xd[i] - x[i]) <= 1e-12 * (1.0 + fabs(xd[i]));
                tsr_fn_results_free(rd, 1);
            } else good = 0;
        }
        *ok += good;
        tsr_fn_results_free(r, 1);
    }
    free(data); free(dense); free(ind); free(ptr); free(b);
}
static void test_spsolve_csr(void)
{
    const int64_t before = tsr_allocated();
    int ok = 0, refused = 0;
    spsolve_csr_case(2000, 1, 1, 0, &ok, &refused);                  /* tridiagonal, banded LU */
    spsolve_csr_case(300, 1, 3, 0, &ok, &refused);                   /* tridiagonal, 3 right-hand sides, vs dense */
    spsolve_csr_case(300, 2, 1, 0, &ok, &refused);                   /* pentadiagonal, vs dense */
    spsolve_csr_case(120, 9, 2, 0, &ok, &refused);                   /* scattered entries: the band is too wide, dense */
    spsolve_csr_case(50, 1, 1, 1, &ok, &refused);                    /* singular */
    CHECK(ok == 4 && refused == 1, "spsolve on CSR: banded and dense-fallback solves to rounding, == the dense solve, singular refused (%d of 4, %d of 1)", ok, refused);
    CHECK(tsr_allocated() == before, "spsolve on CSR leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* eigsh on CSR (thick-restart Lanczos; SM densified) == eigsh on the same matrix dense, for every `which`, on a 2-D
   Poisson matrix (eigenvalues known, many close) and a random symmetric indefinite one; with and without v0 */
static int eigsh_csr_agree(int64_t n, const double *dense, int64_t k, const char *which, int with_v0)
{
    const int id = tsr_fn_find("splinalg.eigsh");
    double *data = malloc((size_t)(n * n) * 8), *v0 = malloc((size_t)n * 8);
    int64_t *ind = malloc((size_t)(n * n) * 8), *ptr = malloc((size_t)(n + 1) * 8), nnz = 0, shp[2] = {n, n};
    ptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) if (dense[i * n + j] != 0.0) { data[nnz] = dense[i * n + j]; ind[nnz] = j; nnz++; }
        ptr[i + 1] = nnz;
        v0[i] = 1.0 + cos(0.9 * (double)i);
    }
    int64_t sd[2] = {n, n}, sn[1] = {nnz}, sp[1] = {n + 1}, s2[1] = {2}, sv[1] = {n};
    tsr_arg items[4] = {arg_arr(data, 1, sn), arg_arr_dt(ind, TSR_I64, 1, sn), arg_arr_dt(ptr, TSR_I64, 1, sp), arg_arr_dt(shp, TSR_I64, 1, s2)};
    tsr_arg grp; memset(&grp, 0, sizeof grp); grp.kind = 5; grp.count = 4; grp.items = items;
    tsr_arg w; memset(&w, 0, sizeof w); w.kind = 2; w.str = which;
    tsr_arg none; memset(&none, 0, sizeof none);
    tsr_arg ad[3] = {arg_arr((double *)dense, 2, sd), arg_num((double)k), w};
    tsr_arg ac[5] = {grp, arg_num((double)k), w, none, with_v0 ? arg_arr(v0, 1, sv) : none};
    tsr_result rd[1], rc2[1];
    int ok = 0;
    if (tsr_fn_routine(id, 3, ad, 1, rd) == 0) {
        if (tsr_fn_routine(id, 5, ac, 1, rc2) == 0) {
            const double *ld = rd[0].arr.data, *lc = rc2[0].arr.data;
            ok = rd[0].arr.shape[0] == k && rc2[0].arr.shape[0] == k;
            for (int64_t i = 0; ok && i < k; i++) ok = fabs(ld[i] - lc[i]) <= 1e-9 * (1.0 + fabs(ld[i]));
            tsr_fn_results_free(rc2, 1);
        }
        tsr_fn_results_free(rd, 1);
    }
    free(data); free(v0); free(ind); free(ptr);
    return ok;
}
static void test_eigsh_csr(void)
{
    enum { G = 15, NP = G * G, NR = 200 };
    double *pois = calloc((size_t)(NP * NP), 8), *rnd = calloc((size_t)(NR * NR), 8);
    for (int r = 0; r < G; r++)                           /* 2-D Poisson, 5-point stencil */
        for (int c = 0; c < G; c++) {
            const int i = r * G + c;
            pois[i * NP + i] = 4.0;
            if (c > 0) pois[i * NP + i - 1] = -1.0;
            if (c + 1 < G) pois[i * NP + i + 1] = -1.0;
            if (r > 0) pois[i * NP + i - G] = -1.0;
            if (r + 1 < G) pois[i * NP + i + G] = -1.0;
        }
    uint64_t st = 2026;
    for (int i = 0; i < NR; i++) {                        /* random symmetric, indefinite, about 6 entries a row */
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        rnd[i * NR + i] = (double)(st >> 11) / 9007199254740992.0 * 8.0 - 4.0;
        for (int d = 0; d < 3; d++) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            const int j = (int)(st % NR);
            if (j == i) continue;
            const double v = (double)(st >> 11) / 9007199254740992.0 - 0.5;
            rnd[i * NR + j] += v; rnd[j * NR + i] += v;
        }
    }
    const int64_t before = tsr_allocated();
    static const char *const whiches[] = {"LA", "SA", "LM", "BE", "SM"};
    int ok = 0, cases = 0;
    for (int wi = 0; wi < 5; wi++) {
        cases += 3;
        ok += eigsh_csr_agree(NP, pois, 5, whiches[wi], 0);
        ok += eigsh_csr_agree(NR, rnd, 4, whiches[wi], 0);
        ok += eigsh_csr_agree(NR, rnd, 6, whiches[wi], 1);
    }
    CHECK(ok == cases, "eigsh on CSR (Lanczos) == eigsh dense: LA/SA/LM/BE/SM, Poisson and indefinite, with and without v0 (%d of %d)", ok, cases);
    CHECK(tsr_allocated() == before, "eigsh on CSR leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
    free(pois); free(rnd);
}

/* hierarchy.linkage == the full-scan agglomeration it replaced (kept here as the reference), byte for byte: every
   method, random distances and small-integer distances full of ties, condensed and observation input */
static double lt_lw(int method, double dik, double djk, double dij, double ni, double nj, double nk)
{
    switch (method) {
    case 0: return dik < djk ? dik : djk;
    case 1: return dik > djk ? dik : djk;
    case 2: return (ni * dik + nj * djk) / (ni + nj);
    case 3: return (dik + djk) / 2.0;
    case 4: return ((ni + nk) * dik + (nj + nk) * djk - nk * dij) / (ni + nj + nk);
    case 5: return (ni * dik + nj * djk) / (ni + nj) - ni * nj * dij / ((ni + nj) * (ni + nj));
    default: return dik / 2.0 + djk / 2.0 - dij / 4.0;
    }
}
static void lt_reference(double *W, int64_t n, int method, double *Z)
{
    const int sq = method >= 4;
    int64_t *act = malloc((size_t)n * 8), *sz = malloc((size_t)n * 8), *cid = malloc((size_t)n * 8);
    for (int64_t i = 0; i < n; i++) { act[i] = i; sz[i] = 1; cid[i] = i; }
    int64_t na = n, nxt = n;
    for (int64_t step = 0; step < n - 1; step++) {
        double best = INFINITY; int64_t bi = 0, bj = 1;
        for (int64_t a = 0; a < na; a++)
            for (int64_t b = a + 1; b < na; b++) { double d = W[act[a] * n + act[b]]; if (d < best) { best = d; bi = a; bj = b; } }
        int64_t A = act[bi], B = act[bj]; double dij = best;
        int64_t c1 = cid[A] < cid[B] ? cid[A] : cid[B], c2 = cid[A] < cid[B] ? cid[B] : cid[A];
        Z[step * 4 + 0] = (double)c1; Z[step * 4 + 1] = (double)c2;
        Z[step * 4 + 2] = sq ? sqrt(dij) : dij; Z[step * 4 + 3] = (double)(sz[A] + sz[B]);
        for (int64_t a = 0; a < na; a++) { int64_t k = act[a]; if (k == A || k == B) continue;
            double nv = lt_lw(method, W[A * n + k], W[B * n + k], dij, (double)sz[A], (double)sz[B], (double)sz[k]);
            W[A * n + k] = W[k * n + A] = nv; }
        sz[A] += sz[B]; cid[A] = nxt++;
        for (int64_t a = bj; a < na - 1; a++) act[a] = act[a + 1];
        na--;
    }
    free(act); free(sz); free(cid);
}
static void test_linkage_fast(void)
{
    static const char *const methods[] = {"single", "complete", "average", "weighted", "ward", "centroid", "median"};
    const int id = tsr_fn_find("hierarchy.linkage");
    const int64_t before = tsr_allocated();
    uint64_t st = 31337;
    int bad = 0, cases = 0;
    for (int kind = 0; kind < 3; kind++) {
        const int64_t n = kind == 2 ? 300 : 120, L = n * (n - 1) / 2, dim = 3;
        double *y = malloc((size_t)L * 8), *obs = malloc((size_t)(n * dim) * 8), *W = malloc((size_t)(n * n) * 8), *Zr = malloc((size_t)((n - 1) * 4) * 8);
        for (int64_t i = 0; i < n * dim; i++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; obs[i] = kind == 1 ? (double)(st % 4) : (double)(st >> 11) / 9007199254740992.0; }
        int64_t q = 0;
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = i + 1; j < n; j++) {
                st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                y[q++] = kind == 1 ? (double)(1 + st % 5) : (double)(st >> 11) / 9007199254740992.0;   /* kind 1: many ties */
            }
        for (int mi = 0; mi < 7; mi++) {
            const int sq = mi >= 4;
            for (int input = 0; input < 2; input++) {    /* condensed y, then the observations (Euclidean) */
                q = 0;
                for (int64_t i = 0; i < n; i++) {
                    W[i * n + i] = INFINITY;
                    for (int64_t j = i + 1; j < n; j++) {
                        double d;
                        if (input == 0) d = y[q++];
                        else { double s = 0.0; for (int64_t c = 0; c < dim; c++) { const double df = obs[i * dim + c] - obs[j * dim + c]; s += df * df; } d = sq ? s : sqrt(s); }
                        if (input == 0 && sq) d *= d;
                        W[i * n + j] = W[j * n + i] = d;
                    }
                }
                lt_reference(W, n, mi, Zr);
                int64_t sy[1] = {L}, so[2] = {n, dim};
                tsr_arg m; memset(&m, 0, sizeof m); m.kind = 2; m.str = methods[mi];
                tsr_arg a[2] = {input == 0 ? arg_arr(y, 1, sy) : arg_arr(obs, 2, so), m};
                tsr_result r[1];
                cases++;
                if (tsr_fn_routine(id, 2, a, 1, r) != 0) { bad++; continue; }
                bad += memcmp(r[0].arr.data, Zr, (size_t)((n - 1) * 4) * 8) != 0;
                tsr_fn_results_free(r, 1);
            }
        }
        free(y); free(obs); free(W); free(Zr);
    }
    CHECK(bad == 0, "hierarchy.linkage == the full-scan reference, byte for byte, 7 methods x ties x 2 inputs (%d of %d differ)", bad, cases);
    CHECK(tsr_allocated() == before, "linkage leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* distance.convex_hull_2d and distance.delaunay_2d, checked by their definitions: the hull walk (counter-clockwise
   from the lowest x, then y, every point on or inside every edge, strict turns, so collinear points dropped), and
   the Delaunay triangulation (no point inside any triangle's circumcircle, no degenerate triangle, 2n - 2 - h
   triangles by Euler, no edge in more than two triangles); collinear input gives no triangles */
static int64_t *g2t_call(const char *fn, double *P, int64_t n, int64_t *rows, int64_t *cols, tsr_result *r)
{
    int64_t sp[2] = {n, 2};
    tsr_arg a[1] = {arg_arr(P, 2, sp)};
    if (tsr_fn_routine(tsr_fn_find(fn), 1, a, 1, r) != 0) return NULL;
    *rows = r[0].arr.shape[0];
    *cols = r[0].arr.ndim == 2 ? r[0].arr.shape[1] : 1;
    return r[0].arr.data;
}
/* the geometric references live in test_geom_ref.c, built with contraction off like spatial_distance.c (why: there) */
double g2t_orient(const double *P, int64_t a, int64_t b, int64_t c);   /* > 0: counter-clockwise */
double g2t_incircle(double ax, double ay, double bx, double by, double cx, double cy, const double *P, int64_t p);
double kd_ref_dist(const double *P, const double *Q, int64_t m, int64_t i, int64_t q);
void fd_normal_ref(const double *r, const double *rh, const double *h, int m, int np, double *J, double *A, double *g);
static void test_geom2d(void)
{
    const int64_t before = tsr_allocated();
    uint64_t st = 4711;
    enum { N = 2000 };
    double *P = malloc((size_t)N * 2 * 8);
    for (int i = 0; i < N * 2; i++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; P[i] = (double)(st >> 11) / 9007199254740992.0; }
    for (int i = 0; i < 10; i++) { P[2 * (N - 1 - i)] = P[2 * i]; P[2 * (N - 1 - i) + 1] = P[2 * i + 1]; }   /* duplicates */
    /* the hull */
    tsr_result r[1];
    int64_t h, one;
    int64_t *hv = g2t_call("distance.convex_hull_2d", P, N, &h, &one, r);
    int ok = hv != NULL && h >= 3;
    for (int64_t q = 0; ok && q < h; q++) {
        const int64_t a = hv[q], b = hv[(q + 1) % h], c = hv[(q + 2) % h];
        ok = g2t_orient(P, a, b, c) > 0.0;                /* strict turns */
        for (int64_t p = 0; ok && p < N; p++) ok = g2t_orient(P, a, b, p) >= 0.0;   /* everything inside */
    }
    for (int64_t p = 0; ok && p < N; p++)                 /* the start: the lowest x, then y */
        ok = P[2 * hv[0]] < P[2 * p] || (P[2 * hv[0]] == P[2 * p] && P[2 * hv[0] + 1] <= P[2 * p + 1]);
    CHECK(ok, "convex_hull_2d: a counter-clockwise walk from the lowest point, strict turns, every point inside (%lld vertices)", (long long)h);
    const int64_t hull_n = h;
    if (hv) tsr_fn_results_free(r, 1);
    /* the triangulation */
    int64_t nt, three;
    int64_t *T = g2t_call("distance.delaunay_2d", P, N, &nt, &three, r);
    const int64_t distinct = N - 10;                      /* the duplicates are skipped */
    ok = T != NULL && three == 3 && nt == 2 * distinct - 2 - hull_n;
    int empty = 1, nondeg = 1;
    for (int64_t t = 0; ok && t < nt; t++) {
        int64_t a = T[3 * t], b = T[3 * t + 1], c = T[3 * t + 2];
        double o = g2t_orient(P, a, b, c);
        if (o == 0.0) { nondeg = 0; break; }
        if (o < 0.0) { const int64_t x = b; b = c; c = x; }
        const double ax = P[2 * a], ay = P[2 * a + 1], bx = P[2 * b], by = P[2 * b + 1], cx = P[2 * c], cy = P[2 * c + 1];
        for (int64_t p = 0; p < N && empty; p++) {
            if (p == a || p == b || p == c) continue;
            if (g2t_incircle(ax, ay, bx, by, cx, cy, P, p) > 1e-12) empty = 0;   /* strictly inside the circumcircle */
        }
    }
    int shared = 1;                                       /* every edge in at most two triangles */
    if (ok) {
        int64_t *ek = malloc((size_t)(nt * 3) * 8);
        for (int64_t t = 0; t < nt; t++)
            for (int e = 0; e < 3; e++) {
                const int64_t u = T[3 * t + e], v = T[3 * t + (e + 1) % 3];
                ek[3 * t + e] = (u < v ? u : v) * N + (u < v ? v : u);
            }
        for (int64_t i = 1; i < nt * 3; i++) {             /* insertion sort is fine for the test size */
            const int64_t k = ek[i]; int64_t j = i - 1;
            while (j >= 0 && ek[j] > k) { ek[j + 1] = ek[j]; j--; }
            ek[j + 1] = k;
        }
        for (int64_t i = 2; i < nt * 3; i++) if (ek[i] == ek[i - 2]) { shared = 0; break; }
        free(ek);
    }
    CHECK(ok && empty && nondeg && shared, "delaunay_2d: 2n-2-h triangles (%lld), empty circumcircles, none degenerate, edges in at most two", (long long)nt);
    if (T) tsr_fn_results_free(r, 1);
    /* collinear points and too few points: no triangles; the hull of collinear points has fewer than 3 vertices */
    double L[20];
    for (int i = 0; i < 10; i++) { L[2 * i] = (double)i; L[2 * i + 1] = 2.0 * (double)i + 1.0; }
    int64_t *T2 = g2t_call("distance.delaunay_2d", L, 10, &nt, &three, r);
    ok = T2 != NULL && nt == 0;
    if (T2) tsr_fn_results_free(r, 1);
    int64_t *H2 = g2t_call("distance.convex_hull_2d", L, 10, &h, &one, r);
    ok = ok && H2 != NULL && h < 3;
    if (H2) tsr_fn_results_free(r, 1);
    int64_t *T3 = g2t_call("distance.delaunay_2d", L, 2, &nt, &three, r);
    ok = ok && T3 != NULL && nt == 0;
    if (T3) tsr_fn_results_free(r, 1);
    double bad[4] = {0.0, NAN, 1.0, 1.0};
    ok = ok && g2t_call("distance.delaunay_2d", bad, 2, &nt, &three, r) == NULL;
    CHECK(ok, "delaunay_2d / convex_hull_2d on collinear, too few and non-finite points");
    free(P);
    CHECK(tsr_allocated() == before, "the 2-D geometry routines leave no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* linalg.fd_normal == its definition, exactly: J[k][j] = (rh[j][k] - r[k]) / h[j], A = J^T J and g = J^T r summed in
   the order of k, zero Jacobian entries (rows where rh equals r) skipped; a mismatched rh is refused */
/* ndimage.label == a flood fill from each component's first pixel in row-major order (the reference below): the
   same labels and count, on random masks of several densities and shapes, every symmetric 3x3 structure family
   (4-, 8-connected, one axis, the diagonals, none), float64 (NaN is foreground) and bool input, and a one-sided
   structure */
static int32_t lab_ref(const uint8_t *fg, int64_t m, int64_t n, const double *st, int32_t *lab)
{
    int nb[8][2], nn = 0;
    for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++)
        if ((di || dj) && st[(di + 1) * 3 + dj + 1] != 0.0) { nb[nn][0] = di; nb[nn][1] = dj; nn++; }
    int64_t *stk = malloc((size_t)(m * n + 1) * 8);
    int32_t num = 0;
    memset(lab, 0, (size_t)(m * n) * 4);
    for (int64_t p = 0; p < m * n; p++) {
        if (!fg[p] || lab[p]) continue;
        int64_t sp = 0; stk[sp++] = p; lab[p] = ++num;
        while (sp) {
            const int64_t q = stk[--sp];
            for (int k = 0; k < nn; k++) {
                const int64_t i = q / n + nb[k][0], j = q % n + nb[k][1];
                if (i < 0 || i >= m || j < 0 || j >= n || !fg[i * n + j] || lab[i * n + j]) continue;
                lab[i * n + j] = num; stk[sp++] = i * n + j;
            }
        }
    }
    free(stk);
    return num;
}
static void test_label_two_pass(void)
{
    const int id = tsr_fn_find("ndimage.label");
    const int64_t before = tsr_allocated();
    static const double sts[7][9] = {
        {0, 1, 0, 1, 0, 1, 0, 1, 0}, {1, 1, 1, 1, 1, 1, 1, 1, 1}, {0, 0, 0, 1, 1, 1, 0, 0, 0}, {0, 1, 0, 0, 1, 0, 0, 1, 0},
        {1, 0, 0, 0, 1, 0, 0, 0, 1}, {1, 0, 1, 0, 0, 0, 1, 0, 1}, {0, 0, 0, 0, 1, 0, 0, 0, 0}};
    static const double one_sided[9] = {0, 1, 0, 0, 0, 1, 0, 0, 0};
    static const int64_t shapes[5][2] = {{1, 1}, {1, 300}, {300, 1}, {37, 53}, {200, 211}};
    uint64_t rs = 777;
    int bad = 0, cases = 0;
    for (int sh = 0; sh < 5; sh++)
        for (int dens = 0; dens < 4; dens++) {
            const int64_t m = shapes[sh][0], n = shapes[sh][1], N = m * n, sz[2] = {m, n}, s33[2] = {3, 3};
            double *x = malloc((size_t)N * 8); uint8_t *fg = malloc((size_t)N), *xb = malloc((size_t)N);
            int32_t *ref = malloc((size_t)N * 4);
            for (int64_t p = 0; p < N; p++) {
                rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                const uint64_t v = rs % 100;
                fg[p] = (int)v < (int[]){10, 45, 60, 90}[dens];
                x[p] = !fg[p] ? (v % 2 ? 0.0 : -0.0) : (v % 7 == 0 ? NAN : (double)v - 50.5);
                xb[p] = fg[p];
            }
            for (int s = 0; s <= 8; s++) {
                const double *st = s < 7 ? sts[s] : s == 7 ? sts[0] : one_sided;
                tsr_arg a[2] = {arg_arr(x, 2, sz), arg_arr((double *)st, 2, s33)};
                if (s == 7) a[0] = arg_arr_dt(xb, TSR_BOOL, 2, sz);
                tsr_result r[2];
                const int32_t nref = lab_ref(fg, m, n, st, ref);
                cases++;
                if (tsr_fn_routine(id, 2, a, 2, r) != 0) { bad++; continue; }
                if (r[0].arr.dtype != TSR_I32 || (int32_t)r[1].ival != nref || memcmp(r[0].arr.data, ref, (size_t)N * 4) != 0) bad++;
                tsr_fn_results_free(r, 2);
            }
            free(x); free(fg); free(xb); free(ref);
        }
    CHECK(id >= 0 && bad == 0, "ndimage.label: the flood fill's labels and count on %d cases (%d differ)", cases, bad);
    CHECK(tsr_allocated() == before, "ndimage.label leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* scipyfft.fft_axis == a transposed copy through tsr_fft / tsr_fft_real and back, bit for bit: every (outer, len,
   inner) split of several shapes (radix 2-5, a generic radix, Bluestein lengths, len 1), real and complex input,
   forward and inverse */
static void test_fft_axis(void)
{
    const int id = tsr_fn_find("scipyfft.fft_axis");
    const int64_t before = tsr_allocated();
    static const int64_t shapes[6][3] = {{3, 8, 5}, {2, 67, 3}, {1, 1, 4}, {5, 13, 1}, {4, 60, 33}, {1, 128, 2}};
    uint64_t rs = 31;
    int bad = 0, cases = 0;
    for (int si = 0; si < 6; si++) {
        const int64_t *sh = shapes[si], N = sh[0] * sh[1] * sh[2];
        double *x = malloc((size_t)N * 16), *t = malloc((size_t)N * 16), *f = malloc((size_t)N * 16), *ref = malloc((size_t)N * 16);
        for (int64_t i = 0; i < 2 * N; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; x[i] = (double)(rs >> 11) / 9007199254740992.0 - 0.5; }
        if (N > 9) { x[3] = NAN; x[8] = -INFINITY; x[9] = -0.0; }
        for (int ax = 0; ax < 3; ax++)
            for (int cplx = 0; cplx < 2; cplx++)
                for (int inv = 0; inv < 2; inv++) {
                    if (!cplx && inv) continue;
                    int64_t outer = 1, inner = 1;
                    for (int d = 0; d < ax; d++) outer *= sh[d];
                    for (int d = ax + 1; d < 3; d++) inner *= sh[d];
                    const int64_t len = sh[ax], w = cplx ? 2 : 1;
                    /* reference: (outer, inner, len) rows, transformed, put back */
                    for (int64_t o = 0; o < outer; o++) for (int64_t i = 0; i < len; i++) for (int64_t c = 0; c < inner; c++)
                        for (int64_t k = 0; k < w; k++) t[((o * inner + c) * len + i) * w + k] = x[((o * len + i) * inner + c) * w + k];
                    const int rc = cplx ? tsr_fft(len, outer * inner, inv, t, f) : tsr_fft_real(len, outer * inner, 0, t, f);
                    for (int64_t o = 0; o < outer; o++) for (int64_t i = 0; i < len; i++) for (int64_t c = 0; c < inner; c++)
                        for (int k = 0; k < 2; k++) ref[((o * len + i) * inner + c) * 2 + k] = f[((o * inner + c) * len + i) * 2 + k];
                    tsr_arg a[3]; memset(a, 0, sizeof a);
                    a[0] = arg_arr_dt(x, cplx ? TSR_C128 : TSR_F64, 3, sh);
                    a[1].kind = 1; a[1].num = ax; a[1].ival = ax; a[1].flags = 1;
                    a[2].kind = 4; a[2].num = inv;
                    tsr_result r[1];
                    cases++;
                    if (rc != 0 || tsr_fn_routine(id, 3, a, 1, r) != 0) { bad++; continue; }
                    if (r[0].arr.dtype != TSR_C128 || memcmp(r[0].arr.data, ref, (size_t)N * 16) != 0) bad++;
                    tsr_fn_results_free(r, 1);
                }
        free(x); free(t); free(f); free(ref);
    }
    CHECK(id >= 0 && bad == 0, "scipyfft.fft_axis: a transposed tsr_fft, bit for bit, on %d cases (%d differ)", cases, bad);
    CHECK(tsr_allocated() == before, "scipyfft.fft_axis leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* signal.fftconvolve on large inputs goes through the FFT (conv_fft_worth): it must agree with the direct sum, which
   correlate with the kernel reversed computes, to rounding, in every mode, in 1-D and 2-D; small inputs keep the
   direct sum and so agree bit for bit. */
static void test_fftconvolve_fft(void)
{
    static const int64_t cases[3][4] = {{1, 3000, 1, 1000}, {120, 130, 40, 30}, {1, 10, 1, 3}};
    static const char *modes[3] = {"full", "same", "valid"};
    uint64_t rs = 77;
    for (int ci = 0; ci < 3; ci++) {
        const int nd = cases[ci][0] == 1 ? 1 : 2;
        int64_t sa[2], sv[2];
        if (nd == 1) { sa[0] = cases[ci][1]; sv[0] = cases[ci][3]; }
        else { sa[0] = cases[ci][0]; sa[1] = cases[ci][1]; sv[0] = cases[ci][2]; sv[1] = cases[ci][3]; }
        const int64_t na = nd == 1 ? sa[0] : sa[0] * sa[1], nv = nd == 1 ? sv[0] : sv[0] * sv[1];
        double *a = malloc((size_t)na * sizeof(double)), *v = malloc((size_t)nv * sizeof(double)), *vr = malloc((size_t)nv * sizeof(double));
        for (int64_t i = 0; i < na; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; a[i] = (double)(rs >> 11) / 9007199254740992.0 - 0.3; }
        for (int64_t i = 0; i < nv; i++) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; v[i] = (double)(rs >> 11) / 9007199254740992.0 - 0.6; }
        for (int64_t i = 0; i < nv; i++) vr[i] = v[nv - 1 - i];
        for (int m = 0; m < 3; m++) {
            tsr_arg f[3] = {arg_arr(a, nd, sa), arg_arr(v, nd, sv), arg_str(modes[m])};
            tsr_arg c[3] = {arg_arr(a, nd, sa), arg_arr(vr, nd, sv), arg_str(modes[m])};
            tsr_result rf[1], rc_[1];
            const int e1 = fa_call("signal.fftconvolve", 3, f, rf, 1), e2 = fa_call("signal.correlate", 3, c, rc_, 1);
            int64_t n = 0;
            double scale = 0.0, err = 0.0;
            int same = e1 == 0 && e2 == 0 && rf[0].arr.ndim == rc_[0].arr.ndim;
            if (same) {
                n = 1;
                for (int d = 0; d < rf[0].arr.ndim; d++) { same &= rf[0].arr.shape[d] == rc_[0].arr.shape[d]; n *= rf[0].arr.shape[d]; }
            }
            if (same) {
                const double *x = rf[0].arr.data, *y = rc_[0].arr.data;
                for (int64_t i = 0; i < n; i++) { if (fabs(y[i]) > scale) scale = fabs(y[i]); if (fabs(x[i] - y[i]) > err) err = fabs(x[i] - y[i]); }
            }
            if (ci < 2) CHECK(same && err <= 1e-12 * scale, "fftconvolve FFT path == direct sum to rounding (%s, case %d: %.3g of %.3g)", modes[m], ci, err, scale);
            else CHECK(same && n > 0 && memcmp(rf[0].arr.data, rc_[0].arr.data, (size_t)n * sizeof(double)) == 0, "fftconvolve small inputs: the direct sum, bit for bit (%s)", modes[m]);
            fa_free(rf, 1); fa_free(rc_, 1);
        }
        free(a); free(v); free(vr);
    }
}

/* the Kelvin zeros by SciPy's KLVNZO: SciPy 1.17.1's values, and an nt past the functions' range ends with an error
   instead of Newton steps on NaN that never stop */
static void test_kelvin_zeros(void)
{
    tsr_result r[8];
    tsr_arg n3[1] = {arg_int(3)};
    int rc = fa_call("special.ber_zeros", 1, n3, r, 1);
    const double *z = rc == 0 ? (const double *)r[0].arr.data : NULL;
    CHECK(z && fabs(z[0] - 2.8489178207951396) <= 1e-14 * 2.85 && fabs(z[1] - 7.238829447632408) <= 1e-14 * 7.24
            && fabs(z[2] - 11.673963549647077) <= 1e-14 * 11.7, "ber_zeros(3) == SciPy's (rc %d)", rc);
    fa_free(r, 1);
    rc = fa_call("special.kei_zeros", 1, n3, r, 1);
    z = rc == 0 ? (const double *)r[0].arr.data : NULL;
    CHECK(z && fabs(z[0] - 3.9146676068432655) <= 1e-14 * 3.92 && fabs(z[2] - 12.782557148597336) <= 1e-14 * 12.8,
          "kei_zeros(3) == SciPy's (rc %d)", rc);
    fa_free(r, 1);
    tsr_arg big[1] = {arg_int(5000)};
    rc = fa_call("special.ber_zeros", 1, big, r, 1);
    CHECK(rc == TSR_EARG, "ber_zeros(5000) past the range ends with an error (rc %d)", rc);
    fa_free(r, 1);
}

/* the Bessel and Airy zeros by SciPy's JYZO / CYZO / AIRYZO: SciPy 1.17.1's values to the last digits */
static int zeros_match(const char *name, int nargs, const tsr_arg *a, int nres, int slot, const double *ref, int n)
{
    tsr_result r[4];
    const int rc = fa_call(name, nargs, a, r, nres);
    int ok = rc == 0 && r[slot].kind == 3 && r[slot].arr.shape[0] == n;
    const int cplx = ok && r[slot].arr.dtype == TSR_C128;
    for (int i = 0; ok && i < n; i++) {
        const double v = ((const double *)r[slot].arr.data)[cplx ? 2 * i : i];
        ok = fabs(v - ref[i]) <= 1e-14 * fabs(ref[i]) + (cplx ? 5e-9 : 0.0);
        if (cplx) ok = ok && ((const double *)r[slot].arr.data)[2 * i + 1] == 0.0;
    }
    fa_free(r, nres);
    return ok;
}
static void test_bessel_airy_zeros(void)
{
    tsr_arg n0[2] = {arg_int(0), arg_int(3)}, n1[2] = {arg_int(1), arg_int(2)}, n0b[2] = {arg_int(0), arg_int(2)},
            n50[2] = {arg_int(50), arg_int(2)}, t2[1] = {arg_int(2)};
    CHECK(zeros_match("special.jn_zeros", 2, n0, 1, 0, (const double[]){2.4048255576957724, 5.520078110286311, 8.653727912911013}, 3), "jn_zeros(0, 3) == SciPy's");
    CHECK(zeros_match("special.jnp_zeros", 2, n1, 1, 0, (const double[]){1.8411837813406595, 5.3314427735250325}, 2), "jnp_zeros(1, 2) == SciPy's");
    CHECK(zeros_match("special.yn_zeros", 2, n1, 1, 0, (const double[]){2.197141326031017, 5.429681040794135}, 2), "yn_zeros(1, 2) == SciPy's");
    CHECK(zeros_match("special.ynp_zeros", 2, n0b, 1, 0, (const double[]){2.197141326031017, 5.429681040794135}, 2), "ynp_zeros(0, 2) == SciPy's");
    CHECK(zeros_match("special.jn_zeros", 2, n50, 1, 0, (const double[]){57.116899160119175, 62.80769876483536}, 2), "jn_zeros(50, 2) == SciPy's");
    CHECK(zeros_match("special.jnyn_zeros", 2, n1, 4, 2, (const double[]){2.197141326031017, 5.429681040794135}, 2), "jnyn_zeros(1, 2)[2] == yn_zeros(1, 2)");
    CHECK(zeros_match("special.ai_zeros", 1, t2, 4, 0, (const double[]){-2.3381074104597674, -4.08794944413097}, 2)
          && zeros_match("special.ai_zeros", 1, t2, 4, 1, (const double[]){-1.0187929716474715, -3.2481975821798375}, 2)
          && zeros_match("special.ai_zeros", 1, t2, 4, 2, (const double[]){0.5356566560156998, -0.4190154780325641}, 2)
          && zeros_match("special.ai_zeros", 1, t2, 4, 3, (const double[]){0.7012108227206915, -0.8031113696548628}, 2), "ai_zeros(2) == SciPy's");
    CHECK(zeros_match("special.bi_zeros", 1, t2, 4, 0, (const double[]){-1.173713222709127, -3.271093302836352}, 2)
          && zeros_match("special.bi_zeros", 1, t2, 4, 3, (const double[]){0.6019578879762394, -0.7603101414928012}, 2), "bi_zeros(2) == SciPy's");
    CHECK(zeros_match("special.y0_zeros", 1, t2, 2, 0, (const double[]){0.89357697, 3.95767842}, 2)
          && zeros_match("special.y1p_zeros", 1, t2, 2, 1, (const double[]){0.41672993, -0.30317374}, 2), "y0_zeros(2), y1p_zeros(2) == SciPy's (complex128, imaginary 0)");
    tsr_arg huge[2] = {arg_int(0), arg_int(10000000)};
    tsr_result r[1];
    CHECK(fa_call("special.jn_zeros", 2, huge, r, 1) == TSR_EARG, "jn_zeros(0, 1e7) refused before the work");
    fa_free(r, 1);
}

static void test_fd_normal(void)
{
    const int64_t before = tsr_allocated();
    enum { M = 257, NP = 3 };
    double r[M], rh[NP * M], h[NP] = {1.5e-8, -2.25e-8, 3e-8};
    uint64_t st = 99;
    for (int k = 0; k < M; k++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        r[k] = (double)(st >> 11) / 9007199254740992.0 - 0.5;
        for (int j = 0; j < NP; j++) rh[j * M + k] = k % 7 == j ? r[k] : r[k] + h[j] * cos(0.1 * k + j);
    }
    double J[M * NP], A[NP * NP] = {0}, g[NP] = {0};
    fd_normal_ref(r, rh, h, M, NP, J, A, g);              /* test_geom_ref.c: contraction off, as the PHP loop rounds */
    int64_t sr[1] = {M}, srh[2] = {NP, M}, sh[1] = {NP};
    tsr_arg a[3] = {arg_arr(r, 1, sr), arg_arr(rh, 2, srh), arg_arr(h, 1, sh)};
    tsr_result res[3];
    int ok = tsr_fn_routine(tsr_fn_find("linalg.fd_normal"), 3, a, 3, res) == 0;
    if (ok) {
        ok = res[0].arr.shape[0] == NP && res[0].arr.shape[1] == NP && res[1].arr.shape[0] == NP
             && res[2].arr.shape[0] == M && res[2].arr.shape[1] == NP
             && memcmp(res[0].arr.data, A, sizeof A) == 0 && memcmp(res[1].arr.data, g, sizeof g) == 0
             && memcmp(res[2].arr.data, J, sizeof J) == 0;
        tsr_fn_results_free(res, 3);
    }
    int64_t bad[2] = {NP, M - 1};
    a[1] = arg_arr(rh, 2, bad);
    ok = ok && tsr_fn_routine(tsr_fn_find("linalg.fd_normal"), 3, a, 3, res) != 0;
    CHECK(ok, "fd_normal: the forward-difference Jacobian and J^T J, J^T r as defined, bit for bit; a wrong rh refused");
    CHECK(tsr_allocated() == before, "fd_normal leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* distance.kdtree_query == an exhaustive search, exactly: the same distances (sqrt of the axis sum in order), the same
   indices and order (nearest first, ties by index), on random points, integer grids full of equal distances,
   duplicates, queries on data points, k = 1, 7 and k > n */
static int kd_cmp(const void *a, const void *b)
{
    const double *x = a, *y = b;
    return x[0] < y[0] ? -1 : x[0] > y[0] ? 1 : (x[1] < y[1] ? -1 : x[1] > y[1]);
}
static void test_kdtree(void)
{
    const int id = tsr_fn_find("distance.kdtree_query"), bid = tsr_fn_find("distance.kdtree_build");
    CHECK(id >= 0 && bid >= 0, "distance.kdtree_query and distance.kdtree_build are registered");
    if (id < 0 || bid < 0) return;
    const int64_t before = tsr_allocated();
    uint64_t st = 4242;
    int bad = 0, cases = 0, reuse_bad = 0;
    for (int shape = 0; shape < 3; shape++) {
        const int64_t n = shape == 2 ? 40 : 3000, m = shape == 1 ? 2 : 3, nq = 200;
        double *P = malloc((size_t)(n * m) * 8), *Q = malloc((size_t)(nq * m) * 8), *pairs = malloc((size_t)n * 16);
        for (int64_t i = 0; i < n * m; i++) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            P[i] = shape == 0 ? (double)(st >> 11) / 9007199254740992.0 : (double)(st % 9);   /* grids: many ties */
        }
        for (int64_t i = 0; i < 20; i++) memcpy(P + (n - 1 - i) * m, P + i * m, (size_t)m * 8);   /* duplicates */
        for (int64_t i = 0; i < nq * m; i++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; Q[i] = shape == 0 ? (double)(st >> 11) / 9007199254740992.0 : (double)(st % 10) - 0.5 * (st % 2); }
        memcpy(Q, P + 5 * m, (size_t)m * 8);                                                     /* a query on a point */
        static const int64_t ks[] = {1, 7, 100};
        for (int kc = 0; kc < 3; kc++) {
            const int64_t k = ks[kc], kk = k < n ? k : n;
            int64_t sp[2] = {n, m}, sq[2] = {nq, m};
            tsr_arg a[3] = {arg_arr(P, 2, sp), arg_arr(Q, 2, sq), arg_num((double)k)};
            tsr_result r[2];
            cases++;
            if (tsr_fn_routine(id, 3, a, 2, r) != 0) { bad++; continue; }
            const double *d = r[0].arr.data;
            const int64_t *ix = r[1].arr.data;
            int ok = r[0].arr.shape[1] == kk;
            for (int64_t q = 0; q < nq && ok; q++) {
                for (int64_t i = 0; i < n; i++) { pairs[2 * i] = kd_ref_dist(P, Q, m, i, q); pairs[2 * i + 1] = (double)i; }
                qsort(pairs, (size_t)n, 16, kd_cmp);
                for (int64_t t = 0; t < kk; t++)
                    if (memcmp(&d[q * kk + t], &pairs[2 * t], 8) || ix[q * kk + t] != (int64_t)pairs[2 * t + 1]) { ok = 0; break; }
            }
            bad += !ok;
            /* the same query on the tree of kdtree_build: the same bytes */
            tsr_result tb[3];
            if (tsr_fn_routine(bid, 1, a, 3, tb) != 0) reuse_bad++;
            else {
                tsr_arg a2[6] = {arg_arr(P, 2, sp), arg_arr(Q, 2, sq), arg_num((double)k),
                                 arg_arr_dt(tb[0].arr.data, TSR_I64, 1, tb[0].arr.shape), arg_arr_dt(tb[1].arr.data, TSR_I64, 2, tb[1].arr.shape),
                                 arg_arr(tb[2].arr.data, 2, tb[2].arr.shape)};
                tsr_result r2[2];
                if (tsr_fn_routine(id, 6, a2, 2, r2) != 0) reuse_bad++;
                else {
                    reuse_bad += memcmp(r2[0].arr.data, r[0].arr.data, (size_t)(nq * kk) * 8) || memcmp(r2[1].arr.data, r[1].arr.data, (size_t)(nq * kk) * 8);
                    tsr_fn_results_free(r2, 2);
                }
                tsr_fn_results_free(tb, 3);
            }
            tsr_fn_results_free(r, 2);
        }
        free(P); free(Q); free(pairs);
    }
    CHECK(bad == 0, "kdtree_query == exhaustive search: distances, indices and order (%d of %d cases differ)", bad, cases);
    CHECK(reuse_bad == 0, "kdtree_query on the tree of kdtree_build: the same distances and indices (%d of %d differ)", reuse_bad, cases);

    /* kdtree_build's tree, then forged trees. The search checks each node and index it reaches: with k = N it reaches
       all of them and every forgery is refused; with k = 1 a forgery off its path may go unnoticed, but nothing is read
       out of bounds (the sanitizer build runs this) and the call either succeeds or is refused. Nothing leaked. */
    enum { N = 100, M = 2 };
    double P[N * M], Q0[M] = {0.5, 0.5};
    for (int i = 0; i < N * M; i++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; P[i] = (double)(st >> 11) / 9007199254740992.0; }
    int64_t sp[2] = {N, M}, s1[2] = {1, M};
    tsr_arg ab[1] = {arg_arr(P, 2, sp)};
    tsr_result t[3];
    const int built = tsr_fn_routine(bid, 1, ab, 3, t) == 0;
    CHECK(built && t[0].arr.shape[0] == N && t[1].arr.shape[0] >= 3 && t[1].arr.shape[1] == 4 && t[2].arr.shape[1] == 2 * M,
          "kdtree_build returns idx (n), nodes (nn x 4) and box (nn x 2m)");
    if (built) {
        const int64_t nn = t[1].arr.shape[0];
        int64_t *idx = malloc(N * 8), *nodes = malloc((size_t)nn * 32);
        double *box = malloc((size_t)nn * 2 * M * 8), fidx[N];
        int refused = 0, trials = 0, near_ok = 0;
        for (int forge = 0; forge < 14; forge++) {
            const int all = forge < 7;
            memcpy(idx, t[0].arr.data, N * 8); memcpy(nodes, t[1].arr.data, (size_t)nn * 32); memcpy(box, t[2].arr.data, (size_t)nn * 2 * M * 8);
            int64_t sidx[1] = {N}, snodes[2] = {nn, 4}, sbox[2] = {nn, 2 * M}, sboxbad[2] = {nn, 2 * M - 1};
            int nargs = 6;
            tsr_arg af[6] = {arg_arr(P, 2, sp), arg_arr(Q0, 2, s1), arg_num(all ? (double)N : 1.0), arg_arr_dt(idx, TSR_I64, 1, sidx),
                             arg_arr_dt(nodes, TSR_I64, 2, snodes), arg_arr(box, 2, sbox)};
            const int f = forge % 7;
            if (f == 0) idx[7] = N;                                            /* index out of range */
            if (f == 1) nodes[4 * 1 + 1] += 1;                                 /* the root's left child split off the median */
            if (f == 2) nodes[2] = 0;                                          /* the root its own child: a cycle */
            if (f == 3) { int64_t l = nodes[2]; while (nodes[4 * l + 2] >= 0) l = nodes[4 * l + 2];        /* a leaf turned inner */
                          nodes[4 * l + 1] = nodes[4 * l] + 1; nodes[4 * l + 2] = nodes[4 * l + 3] = nn - 1; }
            if (f == 4) af[5] = arg_arr(box, 2, sboxbad);                      /* the box of the wrong width */
            if (f == 5) nargs = 5;                                             /* idx and nodes without box */
            if (f == 6) { for (int i = 0; i < N; i++) fidx[i] = (double)idx[i] + 0.25; af[3] = arg_arr(fidx, 1, sidx); }   /* non-integer */
            tsr_result rf[2];
            const int rc = tsr_fn_routine(id, nargs, af, 2, rf);
            if (all) { trials++; refused += rc == TSR_EARG; }
            else near_ok += rc == TSR_EARG || (rc == 0 && rf[1].arr.shape[0] == 1);
            if (rc == 0) tsr_fn_results_free(rf, 2);
        }
        CHECK(refused == trials, "kdtree_query refuses forged trees when the search reaches every node (%d of %d refused)", refused, trials);
        CHECK(near_ok == 7, "kdtree_query with k = 1 on forged trees: refused or a result, never out of bounds (%d of 7)", near_ok);
        /* a tree of finite data searched over data that is not finite: refused when the search reaches the point */
        double Pn[N * M];
        memcpy(Pn, P, sizeof Pn);
        Pn[2 * 11 + 1] = NAN;
        int64_t sidx[1] = {N}, snodes[2] = {nn, 4}, sbox[2] = {nn, 2 * M};
        tsr_arg ag[6] = {arg_arr(Pn, 2, sp), arg_arr(Q0, 2, s1), arg_num((double)N), arg_arr_dt(t[0].arr.data, TSR_I64, 1, sidx),
                         arg_arr_dt(t[1].arr.data, TSR_I64, 2, snodes), arg_arr(t[2].arr.data, 2, sbox)};
        tsr_result rg[2];
        CHECK(tsr_fn_routine(id, 6, ag, 2, rg) == TSR_EARG, "kdtree_query on a tree whose data is not finite: refused");
        free(idx); free(nodes); free(box);
        tsr_fn_results_free(t, 3);
    }
    double nanpt[3] = {0.0, NAN, 1.0};
    int64_t sn[2] = {1, 3};
    tsr_arg an[2] = {arg_arr(nanpt, 2, sn), arg_arr(nanpt, 2, sn)};
    tsr_result rn[3];
    CHECK(tsr_fn_routine(id, 2, an, 2, rn) == TSR_EARG && tsr_fn_routine(bid, 1, an, 3, rn) == TSR_EARG, "kdtree_query and kdtree_build refuse non-finite data");
    CHECK(tsr_allocated() == before, "kdtree_query leaves no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* scipy.fft dct / idct / dst / idst above DCT_DIRECT_MAX run through a real FFT (Makhoul): they must agree with the
   direct summations at sizes around the switch, odd, even and prime, and stay exact enough for a round trip at 1M
   points (where the direct sums would take hours). The reference must be accurate where long double is double
   (macOS arm64, MSVC): the integer phase is reduced modulo the period 4N before it becomes an angle (an angle of
   5e4 rad alone carries 1e-11 of error in double), and the sum is compensated (Neumaier). */
static long double trig_phase(int sine, int64_t m, int64_t N)   /* sin or cos of pi m / (2N) */
{
    const long double pi = 3.141592653589793238462643383279502884L;
    const long double t = pi * (long double)(m % (4 * N)) / (2.0L * N);
    return sine ? sinl(t) : cosl(t);
}
static void trig_add(long double *s, long double *c, long double v)
{
    const long double t = *s + v;
    *c += fabsl(*s) >= fabsl(v) ? (*s - t) + v : (v - t) + *s;
    *s = t;
}
static long double trig_ref(int which, int type, const double *x, int64_t N, int64_t k)
{
    long double s = 0.0L, c = 0.0L;
    if (which == 0 && type == 2) for (int64_t n = 0; n < N; n++) trig_add(&s, &c, 2.0L * x[n] * trig_phase(0, (2 * n + 1) * k, N));
    if (which == 0 && type == 3) { s = x[0]; for (int64_t n = 1; n < N; n++) trig_add(&s, &c, 2.0L * x[n] * trig_phase(0, n * (2 * k + 1), N)); }
    if (which == 1 && type == 2) for (int64_t n = 0; n < N; n++) trig_add(&s, &c, 2.0L * x[n] * trig_phase(1, (2 * n + 1) * (k + 1), N));
    if (which == 1 && type == 3) { s = (k & 1) ? -x[N - 1] : x[N - 1]; for (int64_t n = 0; n < N - 1; n++) trig_add(&s, &c, 2.0L * x[n] * trig_phase(1, (n + 1) * (2 * k + 1), N)); }
    return s + c;
}
static void test_fft_trig(void)
{
    static const int64_t sizes[] = {127, 128, 129, 130, 131, 200, 255, 256, 257, 1000, 1023, 4096, 4099};
    static const char *names[2][2] = {{"scipyfft.dct", "scipyfft.idct"}, {"scipyfft.dst", "scipyfft.idst"}};
    const int64_t before = tsr_allocated();
    uint64_t st = 777;
    double *x = malloc(4099 * 8);
    int bad = 0, cases = 0;
    double worst = 0.0;
    for (size_t si = 0; si < sizeof sizes / sizeof sizes[0]; si++) {
        const int64_t N = sizes[si];
        for (int64_t i = 0; i < N; i++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; x[i] = (double)(st >> 11) / 9007199254740992.0 - 0.5; }
        int64_t sh[1] = {N};
        for (int which = 0; which < 2; which++)
            for (int inv = 0; inv < 2; inv++)
                for (int type = 2; type <= 3; type++) {
                    const int id = tsr_fn_find(names[which][inv]);
                    tsr_arg a[2] = {arg_arr(x, 1, sh), arg_num((double)type)};
                    tsr_result r[1];
                    cases++;
                    if (id < 0 || tsr_fn_routine(id, 2, a, 1, r) != 0) { bad++; continue; }
                    const double *y = r[0].arr.data;
                    const int eff = inv ? (type == 2 ? 3 : 2) : type;
                    const long double sc = inv ? 1.0L / (2.0L * N) : 1.0L;
                    long double mx = 0, err = 0;
                    for (int64_t k = 0; k < N; k++) {
                        const long double ref = trig_ref(which, eff, x, N, k) * sc;
                        if (fabsl(ref) > mx) mx = fabsl(ref);
                        if (fabsl(y[k] - ref) > err) err = fabsl(y[k] - ref);
                    }
                    const double rel = (double)(err / (mx > 0 ? mx : 1));
                    if (rel > worst) worst = rel;
                    if (rel > 1e-13) bad++;
                    tsr_free(r[0].arr.data, r[0].bytes);
                }
    }
    CHECK(bad == 0, "dct/idct/dst/idst types 2-3 == direct sums around the FFT switch (%d of %d differ, worst %.2g)", bad, cases, worst);

    /* 1M points: idct(dct(x)) == x and idst(dst(x)) == x */
    const int64_t M = 1 << 20;
    double *big = malloc((size_t)M * 8);
    for (int64_t i = 0; i < M; i++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; big[i] = (double)(st >> 11) / 9007199254740992.0 - 0.5; }
    int64_t shm[1] = {M};
    double rt = 0.0;
    int ok = 1;
    for (int which = 0; which < 2; which++) {
        tsr_arg a[2] = {arg_arr(big, 1, shm), arg_num(2.0)};
        tsr_result f[1], g[1];
        if (tsr_fn_routine(tsr_fn_find(names[which][0]), 2, a, 1, f) != 0) { ok = 0; continue; }
        tsr_arg b[2] = {arg_arr(f[0].arr.data, 1, shm), arg_num(2.0)};
        if (tsr_fn_routine(tsr_fn_find(names[which][1]), 2, b, 1, g) != 0) { ok = 0; tsr_free(f[0].arr.data, f[0].bytes); continue; }
        const double *z = g[0].arr.data;
        for (int64_t i = 0; i < M; i++) if (fabs(z[i] - big[i]) > rt) rt = fabs(z[i] - big[i]);
        tsr_free(f[0].arr.data, f[0].bytes); tsr_free(g[0].arr.data, g[0].bytes);
    }
    CHECK(ok && rt < 1e-12, "1M-point dct/idct and dst/idst round trips (max error %.2g)", rt);

    /* dctn over a 200 x 150 array: rows then columns, each above the switch */
    enum { R = 200, C = 150 };
    double *m2 = malloc(R * C * 8), *tmp = malloc(R * C * 8);
    for (int i = 0; i < R * C; i++) { st ^= st << 13; st ^= st >> 7; st ^= st << 17; m2[i] = (double)(st >> 11) / 9007199254740992.0 - 0.5; }
    for (int i = 0; i < R; i++) for (int k = 0; k < C; k++) tmp[i * C + k] = (double)trig_ref(0, 2, m2 + i * C, C, k);
    double colv[R], err2 = 0.0, mx2 = 0.0;
    int64_t sh2[2] = {R, C};
    tsr_arg a2[2] = {arg_arr(m2, 2, sh2), arg_num(2.0)};
    tsr_result r2[1];
    const int rc2 = tsr_fn_routine(tsr_fn_find("scipyfft.dctn"), 2, a2, 1, r2);
    if (rc2 == 0) {
        const double *y2 = r2[0].arr.data;
        for (int k = 0; k < C; k++) {
            for (int i = 0; i < R; i++) colv[i] = tmp[i * C + k];
            for (int j = 0; j < R; j++) {
                const double ref = (double)trig_ref(0, 2, colv, R, j);
                if (fabs(ref) > mx2) mx2 = fabs(ref);
                if (fabs(y2[j * C + k] - ref) > err2) err2 = fabs(y2[j * C + k] - ref);
            }
        }
        tsr_free(r2[0].arr.data, r2[0].bytes);
    }
    CHECK(rc2 == 0 && err2 <= 1e-13 * mx2, "dctn 200 x 150 == separable direct sums (relative error %.2g)", mx2 > 0 ? err2 / mx2 : err2);
    free(x); free(big); free(m2); free(tmp);
    CHECK(tsr_allocated() == before, "the trig transforms leave no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

/* tsr_sort with the radix sort forced (the path of machines without AVX2; tsr_sort itself is the AVX2 sort where
   available): the comparisons below keep checking all three algorithms */
static int radix_sort(int dtype, int64_t n, void *data)
{
    tsr_sort_test_radix = 1;
    const int rc = tsr_sort(dtype, n, data);
    tsr_sort_test_radix = 0;
    return rc;
}

/* tsr_sort, tsr_sort_simd and tsr_sort_runs return the same bytes: every dtype, NaN, -0.0/+0.0, duplicates, sorted,
   reversed and interleaved data, +-inf, NaN payloads and signs, subnormals, the integer extremes, sizes 0..300,
   100000 and 300000 (the parallel sort), element-aligned starts that are not 32-byte aligned (the AVX2 loads are
   unaligned), 1 and 4 threads */
static void test_sort_variants(void)
{
    {   /* tsr_sort is tsr_sort_simd; the radix sort is reached through the test switch only */
        double x[5] = {3, -1, 2, NAN, -0.0}, y[5] = {3, -1, 2, NAN, -0.0};
        CHECK(tsr_sort(TSR_F64, 5, x) == 0 && tsr_sort_simd(TSR_F64, 5, y) == 0 && !memcmp(x, y, sizeof x), "tsr_sort == tsr_sort_simd");
    }
    static const int dts[] = {TSR_F64, TSR_F32, TSR_I64, TSR_I32, TSR_U8};
    const int saved = tsr_get_threads();
    uint64_t st = 0x9E3779B97F4A7C15ULL;
    int bad = 0, cases = 0;
    const int64_t big = 300000;                                      /* > TSR_PAR_MIN: the parallel path */
    unsigned char *src = malloc((size_t)(big + 8) * 8), *a = malloc((size_t)(big + 8) * 8),
                  *b = malloc((size_t)(big + 8) * 8), *c = malloc((size_t)(big + 8) * 8);
    for (int threads = 1; threads <= 4; threads += 3) {
        tsr_set_threads(threads);
        for (size_t d = 0; d < sizeof dts / sizeof dts[0]; d++) {
            const int dt = dts[d];
            const int64_t isz = dt == TSR_F64 || dt == TSR_I64 ? 8 : dt == TSR_U8 ? 1 : 4;
            for (int64_t n = 0; n <= big; n = n < 300 ? n + 1 : (n == 300 ? 100000 : (n < big ? big : big + 1))) {
                for (int pattern = 0; pattern < 7; pattern++) {
                    if (n > 300 && (pattern == 4 || pattern == 5)) continue;
                    for (int64_t i = 0; i < n; i++) {
                        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
                        double v = (double)(int64_t)(st >> 40) - 8388608.0;                     /* random */
                        if (pattern == 1) v = (double)(st % 5) - 2.0;                           /* duplicates */
                        if (pattern == 2) v = (double)i;                                        /* sorted */
                        if (pattern == 3) v = (double)(n - i);                                  /* reversed */
                        if (pattern == 4) v = i % 2 ? (double)i : (double)(n - i);              /* interleaved */
                        if (pattern == 5) v = i % 7 == 0 ? (i % 2 ? -0.0 : 0.0) : (i % 11 == 0 ? NAN : v);
                        unsigned char *p = src + 8 + i * isz;                                   /* +8: element-aligned, not 32-byte aligned */
                        if (dt == TSR_F64) memcpy(p, &v, 8);
                        else if (dt == TSR_F32) { float f = (float)v; memcpy(p, &f, 4); }
                        else if (dt == TSR_I64) { int64_t x = isnan(v) ? 7 : (int64_t)v; memcpy(p, &x, 8); }
                        else if (dt == TSR_I32) { int32_t x = isnan(v) ? 7 : (int32_t)v; memcpy(p, &x, 4); }
                        else { uint8_t x = isnan(v) ? 7 : (uint8_t)((int64_t)v & 255); memcpy(p, &x, 1); }
                        if (pattern == 6) {                                                    /* extremes, as raw bits */
                            static const uint64_t e64[] = {0x7ff0000000000000ULL, 0xfff0000000000000ULL, 0x7ff8000000000000ULL,
                                0xfff8000000000001ULL, 0x7ff0000000000001ULL, 0x0000000000000001ULL, 0x8000000000000001ULL,
                                0x7fefffffffffffffULL, 0xffefffffffffffffULL, 0x8000000000000000ULL, 0, 0x3ff0000000000000ULL,
                                0x7fffffffffffffffULL, 0xffffffffffffffffULL};
                            static const uint32_t e32[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7f800001u, 1u,
                                0x80000001u, 0x7f7fffffu, 0xff7fffffu, 0x80000000u, 0, 0x3f800000u, 0x7fffffffu, 0xffffffffu};
                            const int64_t r = (int64_t)(st % 14);
                            if (dt == TSR_F64 || dt == TSR_I64) memcpy(p, &e64[r], 8);  /* as ints: both extremes, -1, 0 */
                            else if (dt == TSR_F32 || dt == TSR_I32) memcpy(p, &e32[r], 4);
                        }
                    }
                    memcpy(a + 8, src + 8, (size_t)(n * isz)); memcpy(b + 8, src + 8, (size_t)(n * isz)); memcpy(c + 8, src + 8, (size_t)(n * isz));
                    const int ra = radix_sort(dt, n, a + 8), rb = tsr_sort_simd(dt, n, b + 8), rc = tsr_sort_runs(dt, n, c + 8);
                    cases++;
                    if (ra || rb || rc || memcmp(a + 8, b + 8, (size_t)(n * isz)) || memcmp(a + 8, c + 8, (size_t)(n * isz))) bad++;
                }
            }
        }
    }
    tsr_set_threads(saved);
    free(src); free(a); free(b); free(c);
    CHECK(bad == 0, "tsr_sort == tsr_sort_simd == tsr_sort_runs, byte for byte (%d of %d cases differ)", bad, cases);
}

/* the AVX2 sort compares float64 values directly unless the CPU treats subnormals as zero (MXCSR DAZ/FTZ): then it
   must fall back to integer keys and still match tsr_sort byte for byte, subnormals and signed zeros included */
static void test_sort_daz(void)
{
#if defined(__x86_64__) && defined(__GNUC__)
    enum { N = 4000 };
    static double src[N], a[N], b[N];
    static float sf[N], af[N], bf[N];
    uint64_t st = 12345;
    for (int i = 0; i < N; i++) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        uint64_t u = st % 3 == 0 ? (st >> 12) : st;                                   /* many subnormals */
        if (i % 17 == 0) u = i % 2 ? 0x8000000000000000ULL : 0;
        if (i % 29 == 0) u = 0x7ff8000000000000ULL;
        memcpy(&src[i], &u, 8);
        uint32_t v = st % 3 == 0 ? (uint32_t)(st >> 41) : (uint32_t)st;
        if (i % 17 == 0) v = i % 2 ? 0x80000000u : 0;
        if (i % 29 == 0) v = 0x7fc00000u;
        memcpy(&sf[i], &v, 4);
    }
    int bad = 0;
    for (int mode = 0; mode < 2; mode++) {
        memcpy(a, src, sizeof a); memcpy(b, src, sizeof b); memcpy(af, sf, sizeof af); memcpy(bf, sf, sizeof bf);
        const unsigned csr = __builtin_ia32_stmxcsr();
        if (mode) __builtin_ia32_ldmxcsr(csr | 0x8040);
        const int r = radix_sort(TSR_F64, N, a) | tsr_sort_simd(TSR_F64, N, b) | radix_sort(TSR_F32, N, af) | tsr_sort_simd(TSR_F32, N, bf);
        __builtin_ia32_ldmxcsr(csr);
        if (r || memcmp(a, b, sizeof a) || memcmp(af, bf, sizeof af)) bad++;
    }
    CHECK(bad == 0, "tsr_sort_simd == tsr_sort on subnormals and signed zeros, with and without DAZ/FTZ (%d modes differ)", bad);
#endif
}

/* ---- sort hardening: the risk register of docs/project/sort-design.md, one block per risk ----
   Oracle: qsort over the documented total order (NaN last, written as the canonical NaN; -0.0 before +0.0), so the
   radix sort is checked too, not trusted. Every case runs tsr_sort, tsr_sort_simd and tsr_sort_runs on copies. */
static int64_t th_k64(uint64_t u) { const int64_t s = (int64_t)u; return s ^ ((s >> 63) & INT64_MAX); }
static int32_t th_k32(uint32_t u) { const int32_t s = (int32_t)u; return s ^ ((s >> 31) & INT32_MAX); }
static int th_cmp_f64(const void *x, const void *y)
{
    uint64_t a, b; memcpy(&a, x, 8); memcpy(&b, y, 8);
    const int64_t p = th_k64(a), q = th_k64(b);
    return p < q ? -1 : p > q;
}
static int th_cmp_f32(const void *x, const void *y)
{
    uint32_t a, b; memcpy(&a, x, 4); memcpy(&b, y, 4);
    const int32_t p = th_k32(a), q = th_k32(b);
    return p < q ? -1 : p > q;
}
static int th_cmp_i64(const void *x, const void *y) { int64_t a, b; memcpy(&a, x, 8); memcpy(&b, y, 8); return a < b ? -1 : a > b; }
static int th_cmp_i32(const void *x, const void *y) { int32_t a, b; memcpy(&a, x, 4); memcpy(&b, y, 4); return a < b ? -1 : a > b; }
static int64_t th_isz(int dt) { return dt == TSR_F64 || dt == TSR_I64 ? 8 : 4; }

static void th_oracle(int dt, int64_t n, unsigned char *b)
{
    const int64_t isz = th_isz(dt);
    int64_t m = 0;
    if (dt == TSR_F64 || dt == TSR_F32) {                      /* NaNs out, then back at the end as NAN */
        for (int64_t i = 0; i < n; i++) {
            int nan;
            if (dt == TSR_F64) { double v; memcpy(&v, b + i * 8, 8); nan = isnan(v); }
            else { float v; memcpy(&v, b + i * 4, 4); nan = isnan(v); }
            if (!nan) { memmove(b + m * isz, b + i * isz, (size_t)isz); m++; }
        }
    } else m = n;
    qsort(b, (size_t)m, (size_t)isz, dt == TSR_F64 ? th_cmp_f64 : dt == TSR_F32 ? th_cmp_f32 : dt == TSR_I64 ? th_cmp_i64 : th_cmp_i32);
    for (int64_t i = m; i < n; i++) {
        if (dt == TSR_F64) { const double v = NAN; memcpy(b + i * 8, &v, 8); }
        else { const float v = NAN; memcpy(b + i * 4, &v, 4); }
    }
}

/* 1 when the three sorts all return the oracle's bytes for src */
static int th_agree(int dt, int64_t n, const unsigned char *src)
{
    const size_t bytes = (size_t)(n * th_isz(dt)) + 1;
    unsigned char *o = malloc(bytes), *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    memcpy(o, src, bytes - 1); memcpy(a, src, bytes - 1); memcpy(b, src, bytes - 1); memcpy(c, src, bytes - 1);
    th_oracle(dt, n, o);
    const int rc = radix_sort(dt, n, a) | tsr_sort_simd(dt, n, b) | tsr_sort_runs(dt, n, c);
    const int ok = rc == 0 && !memcmp(o, a, bytes - 1) && !memcmp(o, b, bytes - 1) && !memcmp(o, c, bytes - 1);
    free(o); free(a); free(b); free(c);
    return ok;
}

static uint64_t th_rng(uint64_t *s) { *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s; }

/* store an integer-valued pattern value in dtype dt (floats get the same value as a double / float) */
static void th_put(int dt, unsigned char *b, int64_t i, int64_t v)
{
    if (dt == TSR_F64) { const double x = (double)v; memcpy(b + i * 8, &x, 8); }
    else if (dt == TSR_F32) { const float x = (float)v; memcpy(b + i * 4, &x, 4); }
    else if (dt == TSR_I64) memcpy(b + i * 8, &v, 8);
    else { const int32_t x = (int32_t)v; memcpy(b + i * 4, &x, 4); }
}
/* the dtype's smallest / largest value (floats: -inf / +inf, or -max / +max when alt) */
static void th_put_extreme(int dt, unsigned char *b, int64_t i, int top, int alt)
{
    if (dt == TSR_F64) { const double x = alt ? (top ? 1.7976931348623157e308 : -1.7976931348623157e308) : (top ? INFINITY : -INFINITY); memcpy(b + i * 8, &x, 8); }
    else if (dt == TSR_F32) { const float x = alt ? (top ? 3.40282347e38f : -3.40282347e38f) : (top ? INFINITY : -INFINITY); memcpy(b + i * 4, &x, 4); }
    else if (dt == TSR_I64) { const int64_t x = top ? INT64_MAX : INT64_MIN; memcpy(b + i * 8, &x, 8); }
    else { const int32_t x = top ? INT32_MAX : INT32_MIN; memcpy(b + i * 4, &x, 4); }
}

enum { TH_RANDOM, TH_BINARY, TH_FEW, TH_ORGAN, TH_SAW7, TH_SAW64, TH_SAWN8, TH_ONE_MIN, TH_ONE_MAX, TH_MOSTLY_MIN,
       TH_MOSTLY_MAX, TH_MINMAX, TH_SORTED_LAST_MIN, TH_REV_DUP, TH_SAMPLE_EQUAL, TH_FLOAT_EDGE, TH_ZEROS, TH_ZERO_SUB,
       TH_NPATTERNS };

static void th_fill(int dt, int64_t n, int pat, unsigned char *b, uint64_t *s)
{
    const int fl = dt == TSR_F64 || dt == TSR_F32;
    const int64_t per = n / 8 > 0 ? n / 8 : 1;
    const uint64_t pick = th_rng(s);
    for (int64_t i = 0; i < n; i++) {
        const uint64_t r = th_rng(s);
        switch (pat) {
        case TH_RANDOM: th_put(dt, b, i, (int64_t)(r >> 34) - (1LL << 29)); break;
        case TH_BINARY: th_put(dt, b, i, (int64_t)(r & 1)); break;
        case TH_FEW: th_put(dt, b, i, (int64_t)(r % 3) - 1); break;
        case TH_ORGAN: th_put(dt, b, i, i < n / 2 ? i : n - i); break;
        case TH_SAW7: th_put(dt, b, i, i % 7); break;
        case TH_SAW64: th_put(dt, b, i, i % 64); break;
        case TH_SAWN8: th_put(dt, b, i, i % per); break;              /* every median-of-9 sample is equal */
        case TH_ONE_MIN: case TH_ONE_MAX: th_put(dt, b, i, 5); break;
        case TH_MOSTLY_MIN: if (r % 10) th_put_extreme(dt, b, i, 0, pick & 1); else th_put(dt, b, i, (int64_t)(r >> 40)); break;
        case TH_MOSTLY_MAX: if (r % 10) th_put_extreme(dt, b, i, 1, pick & 1); else th_put(dt, b, i, (int64_t)(r >> 40)); break;
        case TH_MINMAX: th_put_extreme(dt, b, i, (int)(i & 1), (int)(pick & 1)); break;
        case TH_SORTED_LAST_MIN: th_put(dt, b, i, i == n - 1 ? -1 : i); break;
        case TH_REV_DUP: th_put(dt, b, i, (n - i) / 3); break;
        case TH_SAMPLE_EQUAL: th_put(dt, b, i, (int64_t)(r >> 40)); break;
        case TH_FLOAT_EDGE: case TH_ZEROS: case TH_ZERO_SUB: {
            if (!fl) { th_put(dt, b, i, (int64_t)(r % 5) - 2); break; }
            /* the values where the float path has special cases: QS_NEXT steps, the zero run, NaN, infinities */
            static const uint64_t e64[] = {0x8000000000000001ULL, 0x8000000000000000ULL, 0, 1, 0x7fefffffffffffffULL,
                0xffefffffffffffffULL, 0xfff0000000000000ULL, 0x7ff0000000000000ULL, 0x0010000000000000ULL,
                0x7ff8000000000000ULL, 0xfff8000000000005ULL, 0x800fffffffffffffULL};
            static const uint32_t e32[] = {0x80000001u, 0x80000000u, 0, 1, 0x7f7fffffu, 0xff7fffffu, 0xff800000u,
                0x7f800000u, 0x00800000u, 0x7fc00000u, 0xffc00005u, 0x807fffffu};
            int k;
            if (pat == TH_FLOAT_EDGE) k = r % 10 < 8 ? (int)(pick % 12) : (int)(r % 12);      /* 80% one edge value */
            else if (pat == TH_ZEROS) k = (int)(r & 1);                                        /* -0.0 / +0.0 only */
            else k = (int)(r % 6 == 5 ? 9 : r % 4);                                            /* zeros, +-tiny, NaN */
            if (dt == TSR_F64) memcpy(b + i * 8, &e64[k], 8); else memcpy(b + i * 4, &e32[k], 4);
            break;
        }
        }
    }
    if (n > 0 && pat == TH_ONE_MIN) th_put_extreme(dt, b, (int64_t)(pick % (uint64_t)n), 0, 0);
    if (n > 0 && pat == TH_ONE_MAX) th_put_extreme(dt, b, (int64_t)(pick % (uint64_t)n), 1, 0);
    if (n > 0 && pat == TH_SAMPLE_EQUAL)                            /* the nine pivot samples hold one value */
        for (int k = 0; k <= 8; k++) th_put(dt, b, k < 8 ? k * n / 8 : n - 1, 42);
}

static void test_sort_hardening(void)
{
    static const int dts[] = {TSR_F64, TSR_F32, TSR_I64, TSR_I32};
    const int saved = tsr_get_threads();
    const int64_t big = 300000;
    unsigned char *src = malloc((size_t)big * 8);
    uint64_t st = 0xD1B54A32D192ED03ULL;
    const int64_t before = tsr_allocated();

    /* 1. every size through every leaf network and partition boundary (16 / 32 / 64 / 128 keys, the 16 * L scalar
          partition, the 4 * L hold-back): n = 0 .. 1100, random, 0/1 and three-valued data */
    int bad = 0, cases = 0;
    tsr_set_threads(1);
    for (size_t d = 0; d < 4; d++)
        for (int64_t n = 0; n <= 1100; n++)
            for (int pat = TH_RANDOM; pat <= TH_FEW; pat++) {
                th_fill(dts[d], n, pat, src, &st);
                cases++; bad += !th_agree(dts[d], n, src);
            }
    CHECK(bad == 0, "sorts == oracle at every size 0..1100 (%d of %d cases differ)", bad, cases);

    /* 2. the leaf networks by the 0-1 principle: a comparator network sorts every input if it sorts every 0/1
          input; exhaustive for n <= 16 (the whole 16-key network and the start of the 32/64-key ones) */
    bad = 0; cases = 0;
    for (size_t d = 0; d < 4; d++) {
        const int dt = dts[d];
        const int64_t isz = th_isz(dt);
        for (int n = 1; n <= 16; n++)
            for (uint32_t m = 0; m < (1u << n); m++) {
                int ones = 0;
                for (int i = 0; i < n; i++) { th_put(dt, src, i, (m >> i) & 1); ones += (m >> i) & 1; }
                if (tsr_sort_simd(dt, n, src)) { bad++; continue; }
                unsigned char z[8], o[8];
                th_put(dt, z, 0, 0); th_put(dt, o, 0, 1);
                for (int i = 0; i < n; i++) if (memcmp(src + i * isz, i < n - ones ? z : o, (size_t)isz)) { bad++; break; }
                cases++;
            }
    }
    CHECK(bad == 0, "sortSimd sorts every 0/1 input of up to 16 keys, all dtypes (%d of %d differ)", bad, cases);

    /* 3. structured and adversarial data: equal pivot samples, extremes (QS_MAX / QS_NEXT edges), minimum-pivot
          splitting, organ pipes, sawtooths, float edge values, zero runs; serial and parallel sizes */
    bad = 0; cases = 0;
    for (size_t d = 0; d < 4; d++)
        for (int pat = TH_ORGAN; pat < TH_NPATTERNS; pat++)
            for (int sz = 0; sz < 3; sz++) {
                const int64_t n = sz == 0 ? 1000 : sz == 1 ? 65537 : big;
                tsr_set_threads(sz == 2 ? 4 : 1);
                th_fill(dts[d], n, pat, src, &st);
                cases++;
                if (!th_agree(dts[d], n, src)) { bad++; fprintf(stderr, "  sort pattern %d dtype %d n %lld differs\n", pat, dts[d], (long long)n); }
            }
    CHECK(bad == 0, "sorts == oracle on structured, adversarial and float-edge data (%d of %d cases differ)", bad, cases);

    /* 4. the heapsort fallback (normally reached only by adversarial inputs): recursion depth capped */
    bad = 0; cases = 0;
    for (int depth = 1; depth <= 8; depth += 3) {
        tsr_sort_test_depth = depth;
        for (size_t d = 0; d < 4; d++)
            for (int sz = 0; sz < 3; sz++)
                for (int pat = TH_RANDOM; pat <= TH_ORGAN; pat++) {
                    const int64_t n = sz == 0 ? 200 : sz == 1 ? 5000 : big;
                    tsr_set_threads(sz == 2 ? 4 : 1);
                    th_fill(dts[d], n, pat, src, &st);
                    cases++; bad += !th_agree(dts[d], n, src);
                }
    }
    tsr_sort_test_depth = 0;
    CHECK(bad == 0, "sorts == oracle with the heapsort fallback forced at depth 1, 4, 7 (%d of %d differ)", bad, cases);

#if defined(_OPENMP) && defined(__x86_64__) && defined(__GNUC__)
    /* 5. MXCSR is per thread: workers left in denormals-are-zero mode while the caller is not (the caller picks the
          double path, so every worker must compare as the caller does) */
    {
        tsr_set_threads(4);
        const unsigned int csr = __builtin_ia32_stmxcsr();
#pragma omp parallel num_threads(4)
        { if (omp_get_thread_num() != 0) __builtin_ia32_ldmxcsr(csr | 0x8040); }
        bad = 0;
        for (int rep = 0; rep < 3; rep++) {
            th_fill(TSR_F64, big, TH_ZERO_SUB, src, &st);
            for (int64_t i = 0; i < big; i += 3) { const uint64_t u = th_rng(&st) >> 13; memcpy(src + i * 8, &u, 8); }  /* subnormals */
            bad += !th_agree(TSR_F64, big, src);
        }
#pragma omp parallel num_threads(4)
        { __builtin_ia32_ldmxcsr(csr); }
        CHECK(bad == 0, "parallel float64 sort ignores worker threads' DAZ/FTZ state (%d of 3 differ)", bad);
    }

    /* 6. the application's own threads: concurrent sorts of different arrays (OpenMP region: the sort's own region is
          nested, so it runs serially; nothing is shared, so each result must still be exact) */
    {
        tsr_set_threads(4);
        enum { K = 4 };
        int fails = 0;
#pragma omp parallel for num_threads(K) reduction(+ : fails) schedule(static)
        for (int k = 0; k < K; k++) {
            unsigned char *mine = malloc((size_t)big * 8);
            uint64_t s2 = 0x1234567ULL + (uint64_t)k;
            th_fill(dts[k % 4], big, TH_RANDOM, mine, &s2);
            fails += !th_agree(dts[k % 4], big, mine);
            free(mine);
        }
        CHECK(fails == 0, "concurrent sorts from an application's OpenMP threads are exact (%d of %d differ)", fails, (int)K);
    }
#endif
    tsr_set_threads(saved);
    free(src);
    CHECK(tsr_allocated() == before, "the sorts leave no allocation behind (%lld)", (long long)(tsr_allocated() - before));
}

#if !defined(_WIN32)
/* 7. the application's own (non-OpenMP) threads, each calling the multithreaded sort at once: every caller starts its
      own OpenMP team (oversubscribed, but exact) */
struct th_job { int dt; uint64_t seed; int ok; };
static void *th_worker(void *p)
{
    struct th_job *j = p;
    unsigned char *mine = malloc((size_t)300000 * 8);
    th_fill(j->dt, 300000, TH_RANDOM, mine, &j->seed);
    j->ok = th_agree(j->dt, 300000, mine);
    th_fill(j->dt, 300000, TH_ZERO_SUB, mine, &j->seed);
    j->ok &= th_agree(j->dt, 300000, mine);
    free(mine);
    return NULL;
}
static void test_sort_app_threads(void)
{
    const int saved = tsr_get_threads();
    tsr_set_threads(4);
    enum { K = 4 };
    pthread_t t[K];
    struct th_job jobs[K];
    for (int k = 0; k < K; k++) {
        jobs[k] = (struct th_job){ (int[]){TSR_F64, TSR_F32, TSR_I64, TSR_I32}[k], 0xABCDEFULL + (uint64_t)k, 0 };
        pthread_create(&t[k], NULL, th_worker, &jobs[k]);
    }
    int ok = 0;
    for (int k = 0; k < K; k++) { pthread_join(t[k], NULL); ok += jobs[k].ok; }
    tsr_set_threads(saved);
    CHECK(ok == K, "concurrent multithreaded sorts from %d application pthreads are exact (%d ok)", (int)K, ok);
}

/* 8. small thread stacks: 128 KB is musl's default for pthreads and its OpenMP workers, and on glibc the static TLS
      of the loaded libraries comes out of it (about 66 KB here). The radix sort once kept 49 KB of tables per level
      on the stack and overflowed it. The keys have one random bit in each 11-bit MSD digit
      (bits 20, 31, 42, 53, 62) over 20 random low bits, so every level splits in two buckets above 4096 keys and
      the radix sort recurses to its full depth on the calling thread. */
static void *th_small_worker(void *p)
{
    struct th_job *j = p;
    enum { M = 300000 };
    int64_t *v = malloc((size_t)M * 8);
    for (int64_t i = 0; i < M; i++) {
        const uint64_t r = th_rng(&j->seed);
        uint64_t k = r & 0xFFFFF;
        for (int b = 0; b < 5; b++) k |= ((r >> (20 + b)) & 1) << (int[]){20, 31, 42, 53, 62}[b];
        v[i] = (int64_t)k;
    }
    j->ok = th_agree(TSR_I64, M, (const unsigned char *)v);
    free(v);
    return NULL;
}
static void test_sort_small_stacks(void)
{
    const int saved = tsr_get_threads();
    int ok = 1;
    for (int threads = 1; threads <= 4; threads += 3) {
        tsr_set_threads(threads);
        pthread_attr_t at;
        pthread_attr_init(&at);
        pthread_attr_setstacksize(&at, 128 * 1024);
        pthread_t t;
        struct th_job job = { TSR_I64, 0x5EED5EEDULL + (uint64_t)threads, 0 };
        ok = ok && pthread_create(&t, &at, th_small_worker, &job) == 0 && pthread_join(t, NULL) == 0 && job.ok;
        pthread_attr_destroy(&at);
    }
    tsr_set_threads(saved);
    CHECK(ok, "the sorts run exact on a 128 KB thread stack at the radix sort's full recursion depth (1 and 4 threads)");
}
#endif

static void test_abi_runtime(void)
{    /* version and capability negotiation: one version source */
    CHECK(tsr_abi_version() == TSR_ABI_VERSION && TSR_ABI_VERSION == 2, "abi version %d", tsr_abi_version());
    CHECK(strcmp(tsr_version(), TSR_VERSION_STRING) == 0, "tsr_version %s", tsr_version());
    {
        /* "unknown", or 40 lowercase hex digits optionally followed by "-dirty" */
        const char *c = tsr_build_commit();
        int hex = 0;
        while (c[hex] && ((c[hex] >= '0' && c[hex] <= '9') || (c[hex] >= 'a' && c[hex] <= 'f'))) hex++;
        CHECK(strcmp(c, "unknown") == 0 || (hex == 40 && (c[40] == '\0' || strcmp(c + 40, "-dirty") == 0)),
              "tsr_build_commit %s", c);
    }
    CHECK(tsr_feature("repro_f32") == 1 && tsr_feature("gemm_canonical") == 1 && tsr_feature("lapack") == 1, "features present");
    CHECK(tsr_feature("openmp") == tsr_openmp(), "openmp feature matches tsr_openmp");
    const int fpenv = tsr_feature("fpenv_check");
    CHECK(fpenv == 0 || fpenv == 1, "fpenv_check answers 0/1");
    CHECK(tsr_feature("no_such_feature") == 0 && tsr_feature("") == 0 && tsr_feature(NULL) == 0, "unknown features are 0");

    /* checked allocation sizes: products of independent lengths never wrap */
    CHECK(tsr_count_mul((int64_t)1 << 32, (int64_t)1 << 31) == -1, "count_mul overflow");
    CHECK(tsr_count_mul(-1, 8) == -1 && tsr_count_mul(0, INT64_MAX) == 0 && tsr_count_mul(3, 7) == 21, "count_mul values");
    CHECK(tsr_bytes_n(INT64_MAX / 8, 8, 64) == -1 && tsr_bytes_n(10, 8, 64) == 144, "bytes_n pad overflow / value");
    CHECK(tsr_scratch_n((int64_t)1 << 62, 16) == NULL && tsr_scratch0_n(-5, 8) == NULL, "scratch refuses overflow");
    void *s0 = tsr_scratch0_n(0, 8);
    CHECK(s0 != NULL, "scratch of 0 elements is a valid block");
    free(s0);
    const int64_t before = tsr_allocated();
    CHECK(tsr_alloc_n((int64_t)1 << 62, 16, 0) == NULL && tsr_alloc_n(-1, 8, 0) == NULL, "alloc_n refuses overflow");
    CHECK(tsr_allocated() == before, "a refused alloc_n leaves the accounting alone");
    void *an = tsr_alloc_n(100, 8, 24);
    CHECK(an && tsr_allocated() == before + 824, "alloc_n counted: %ld", (long)(tsr_allocated() - before));
    tsr_free(an, tsr_bytes_n(100, 8, 24));

    /* cluster.hierarchy: the n*n workspace is counted, so a budget refuses it (it used to bypass the budget) */
    const int64_t nobs = 1000;
    double *obs = malloc((size_t)nobs * 2 * sizeof(double));
    for (int64_t i = 0; i < nobs * 2; i++) obs[i] = (double)((i * 7919) % 1009);
    int64_t osh[2] = {nobs, 2};
    const int lk = tsr_fn_find("hierarchy.linkage");
    tsr_arg la[1] = {arg_arr(obs, 2, osh)};
    tsr_result lr[1];
    tsr_set_budget(tsr_allocated() + (1 << 20));                  /* 1 MiB: far below the 8 MB workspace */
    CHECK(lk >= 0 && tsr_fn_routine(lk, 1, la, 1, lr) == TSR_ENOMEM, "linkage workspace counted against the budget");
    tsr_set_budget(0);
    int64_t small[2] = {50, 2};
    tsr_arg ls[1] = {arg_arr(obs, 2, small)};
    if (tsr_fn_routine(lk, 1, ls, 1, lr) == TSR_OK) {
        CHECK(lr[0].arr.shape[0] == 49 && lr[0].arr.shape[1] == 4, "linkage still works");
        tsr_free(lr[0].arr.data, lr[0].bytes);
    } else CHECK(0, "linkage failed: %s", tsr_fn_error());
    /* a condensed distance vector whose length is not k*(k-1)/2 sized a workspace larger than it and read past it */
    int64_t len5[1] = {5}, len6[1] = {6};
    tsr_arg l5[1] = {arg_arr(obs, 1, len5)}, l6[1] = {arg_arr(obs, 1, len6)};
    CHECK(tsr_fn_routine(lk, 1, l5, 1, lr) == TSR_EARG, "linkage refuses a condensed length of 5");
    if (tsr_fn_routine(lk, 1, l6, 1, lr) == TSR_OK) {
        CHECK(lr[0].arr.shape[0] == 3, "linkage of 4 points from 6 distances");
        tsr_free(lr[0].arr.data, lr[0].bytes);
    } else CHECK(0, "linkage of 6 distances failed: %s", tsr_fn_error());
    free(obs);

    /* signal: zpk2tf (now checked) through butter, and a high order beyond its stack scratch */
    const int bt = tsr_fn_find("signal.butter");
    for (int N = 4; N <= 64; N += 60) {
        tsr_arg ba[2] = {arg_num((double)N), arg_num(0.25)};
        tsr_result br[2];
        if (bt >= 0 && tsr_fn_routine(bt, 2, ba, 2, br) == TSR_OK) {
            CHECK(br[0].arr.shape[0] == N + 1 && br[1].arr.shape[0] == N + 1 && ((double *)br[1].arr.data)[0] == 1.0,
                  "butter(%d) shape and a[0]", N);
            tsr_free(br[0].arr.data, br[0].bytes);
            tsr_free(br[1].arr.data, br[1].bytes);
        } else CHECK(0, "butter(%d) failed: %s", N, tsr_fn_error());
    }

    /* BLAS pool: controllable exactly when a BLAS with a thread API is loaded */
    CHECK(tsr_set_blas_threads(0) == TSR_EARG, "blas threads < 1 refused");
    if (tsr_feature("openblas_threads")) {
        const int prev = tsr_get_blas_threads();
        CHECK(tsr_set_blas_threads(1) == TSR_OK && tsr_get_blas_threads() == 1, "blas threads set to 1");
        if (prev > 0) tsr_set_blas_threads(prev);
    } else {
        CHECK(tsr_set_blas_threads(1) == TSR_ETYPE && tsr_get_blas_threads() == 0, "no BLAS: refused, unknown size");
    }

    /* thread clamp follows the CPUs this process may use (affinity / cgroup quota) */
    const int cpus = tsr_effective_cpus();
    CHECK(cpus >= 1, "effective cpus %d", cpus);
#if defined(_SC_NPROCESSORS_ONLN)
    CHECK(cpus <= sysconf(_SC_NPROCESSORS_ONLN), "cpus %d within the online CPUs", cpus);
#endif
    tsr_set_threads(1 << 30);
    CHECK(tsr_get_threads() == (tsr_openmp() ? cpus * 4 : 1), "threads clamped to 4 x effective cpus (%d)", tsr_get_threads());

    /* repro_f32 regions follow tsr_set_threads, and stay bit-identical across thread counts */
    const int64_t RN = 1 << 16;
    float *rx = malloc(RN * sizeof(float)), *r1 = malloc(RN * sizeof(float)), *r4 = malloc(RN * sizeof(float));
    for (int64_t i = 0; i < RN; i++) rx[i] = (float)((double)(i - RN / 2) / 1024.0);
    tsr_set_threads(1);
    tsr_repro_exp_f32(RN, rx, r1);
    tsr_set_threads(4);
    tsr_repro_exp_f32(RN, rx, r4);
    CHECK(memcmp(r1, r4, RN * sizeof(float)) == 0, "repro exp identical at 1 and 4 threads");
    tsr_repro_erf_f32(RN, rx, r4);
    tsr_set_threads(1);
    tsr_repro_erf_f32(RN, rx, r1);
    CHECK(memcmp(r1, r4, RN * sizeof(float)) == 0, "repro erf identical at 1 and 4 threads");
    free(rx); free(r1); free(r4);

    /* fork: a child of a process that computed with threads runs serial until set again; the parent keeps its count
       (POSIX only: Windows has no fork; platform-debt PD-15) */
#if !defined(_WIN32)
    /* the parent really runs a threaded region first (above TSR_PAR_MIN), so the OpenMP runtime's worker pool exists
       at fork time; the child must run the same kernel serially and finish (an alarm turns a deadlock into a failure) */
    const int64_t fn_[1] = {(int64_t)1 << 18}, fs8[1] = {8};
    double *fa = (double *)malloc(sizeof(double) * (size_t)fn_[0]), *fo = (double *)malloc(sizeof(double) * (size_t)fn_[0]);
    for (int64_t i = 0; i < fn_[0]; i++) fa[i] = (double)(i % 7) * 0.25;
    tsr_set_threads(4);
    tsr_unary(U_EXP, TSR_F64, 1, fn_, fa, fs8, fo, fs8);       /* parallel region in the parent */
    const double fwant = fo[(int64_t)1 << 17];                 /* the threaded result: the serial child must match */
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) {
        alarm(20);
        if (tsr_get_threads() != 1) _exit(3);
        fo[(int64_t)1 << 17] = 0.0;
        if (tsr_unary(U_EXP, TSR_F64, 1, fn_, fa, fs8, fo, fs8) != 0) _exit(4);
        _exit(fo[(int64_t)1 << 17] == fwant ? 0 : 5);
    }
    int status = -1;
    CHECK(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "forked child of a threaded parent runs a kernel serially, same bits, no deadlock (status %d)", status);
    CHECK(tsr_get_threads() == (tsr_openmp() ? 4 : 1), "parent keeps its thread count");
    tsr_set_threads(1);
    free(fa); free(fo);
#endif

    /* distribution errors are per calling thread: a failing call on one thread never leaks into the other */
    const int sk = tsr_fn_find("stats.skewnorm");
    CHECK(sk >= 0, "stats.skewnorm registered");
    if (sk >= 0) {
        _Atomic int done = 0;
        abi_dist_job fail_job = {sk, 0, 0, &done}, ok_job = {sk, 0, 0, &done};
        pthread_t ta, tb;
        pthread_create(&tb, NULL, abi_dist_ok, &ok_job);
        pthread_create(&ta, NULL, abi_dist_failing, &fail_job);
        pthread_join(ta, NULL);
        pthread_join(tb, NULL);
        CHECK(fail_job.bad == 0, "failing thread saw its own error every time (%d of %d missed)", fail_job.bad, fail_job.calls);
        CHECK(ok_job.bad == 0, "succeeding thread never saw the other thread's error (%d of %d leaked)", ok_job.bad, ok_job.calls);
    }
}

/* Regressions for the 2026-10 numerical-correctness review (docs/reference/numerical-guarantees.md). Every
   expected value below is NumPy 2.4.4 / SciPy 1.17.1 output on x86-64. Deterministic: fixed inputs, no timing,
   and the OpenMP part asserts only on the OR over all workers. */
#include <fenv.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__x86_64__)
static uint32_t nr_csr_get(void) { uint32_t c; __asm__ volatile("stmxcsr %0" : "=m"(c)); return c; }
static void nr_csr_set(uint32_t c) { __asm__ volatile("ldmxcsr %0" : : "m"(c)); }
#endif
static int nr_bits_eq(double a, double b) { return memcmp(&a, &b, sizeof a) == 0; }
static int nr_bits_eqf(float a, float b) { return memcmp(&a, &b, sizeof a) == 0; }
static double nr_routine_num(const char *name, int nargs, const tsr_arg *args)
{
    tsr_result r[1];
    const int id = tsr_fn_find(name);
    if (id < 0 || tsr_fn_routine(id, nargs, args, 1, r) != 0) return -12345.0;
    double v = r[0].kind == 3 ? ((double *)r[0].arr.data)[0] : r[0].num;
    if (r[0].kind == 3) tsr_free(r[0].arr.data, r[0].bytes);
    return v;
}

static void test_numerics_review(void)
{
    /* (1) selfcheck sensitivity: full-mantissa KAT. The canonical digest must not survive a two-rounding
       mul+add or a reordered k sum (the old k/64 KAT was exact in f32 and could not see either). */
    const uint64_t d0 = tsr_selfcheck_digest(0), d1 = tsr_selfcheck_digest(1);
    CHECK(d0 == tsr_selfcheck_digest(0), "KAT digest is deterministic");
    CHECK(tsr_selfcheck_canonical() == 0 || d0 == tsr_selfcheck_canonical(), "KAT reference == frozen %016llx", (unsigned long long)d0);
    CHECK(d1 == d0, "dispatched tsr_gemm == fmaf reference");
    CHECK(tsr_selfcheck_digest(2) != d0, "KAT detects a non-FMA (mul then add) build");
    CHECK(tsr_selfcheck_digest(3) != d0, "KAT detects a reordered reduction");
    CHECK(tsr_selfcheck_digest(99) == 0, "unknown KAT variant");
    CHECK(tsr_selfcheck() == 0, "selfcheck canonical (rc %d)", tsr_selfcheck());

    /* (2) FP environment gate */
    char why[256];
    CHECK(tsr_fpenv_check(why, sizeof why) == 0 && strcmp(why, "canonical") == 0, "fpenv canonical: %s", why);
    CHECK(tsr_fpenv_check_thread(NULL, 0) == 0, "fpenv thread canonical");
    fesetround(FE_UPWARD);
    const int up = tsr_fpenv_check(why, sizeof why);
    const int sc_up = tsr_selfcheck();
    fesetround(FE_TONEAREST);
    CHECK((up & 16) && strstr(why, "fegetround") != NULL, "fpenv sees round-upward (%d: %s)", up, why);
    CHECK(sc_up == -6, "selfcheck refuses a non-canonical FP environment (rc %d)", sc_up);
    int fp_modelled = 1;   /* 0 under an emulator that ignores FP control writes (Valgrind keeps only the rounding mode) */
#if defined(__x86_64__)
    const uint32_t csr0 = nr_csr_get();
    nr_csr_set(csr0 | (1u << 15) | (1u << 6));                         /* FTZ + DAZ, what -ffast-math sets */
    fp_modelled = (nr_csr_get() & ((1u << 15) | (1u << 6))) == ((1u << 15) | (1u << 6));
    if (fp_modelled) {
        const int ftz = tsr_fpenv_check(why, sizeof why);
        CHECK(ftz == 3 && strstr(why, "FTZ") && strstr(why, "DAZ"), "fpenv names FTZ/DAZ (%d: %s)", ftz, why);
        nr_csr_set(csr0 & ~(1u << 9));                                 /* unmask divide-by-zero */
        CHECK(tsr_fpenv_check_thread(why, sizeof why) == 8 && strstr(why, "masks"), "fpenv sees unmasked traps: %s", why);
        nr_csr_set(csr0 | (1u << 15));
        CHECK(tsr_fpenv_canonicalize() == 0 && nr_csr_get() == csr0, "canonicalize restores MXCSR");
    } else {
        nr_csr_set(csr0);
        fprintf(stderr, "note: MXCSR FTZ/DAZ/mask writes do not take effect here (an emulator such as Valgrind): "
                        "FTZ/DAZ/trap checks not applicable\n");
    }
#endif
#ifdef _OPENMP
    const int th = tsr_get_threads();
    tsr_set_threads(4);
#pragma omp parallel num_threads(4)
    {
#if defined(__x86_64__)
        if (omp_get_thread_num() != 0) nr_csr_set(nr_csr_get() | (1u << 15));
#else
        if (omp_get_thread_num() != 0) fesetround(FE_UPWARD);
#endif
    }
    const int wk = tsr_fpenv_check(why, sizeof why);
    CHECK(tsr_fpenv_check_thread(NULL, 0) == 0, "caller unaffected by worker state");
    /* does a worker's change persist into the next parallel region? libgomp keeps each worker's MXCSR/FPCR;
       libomp loads the caller's FP control into every worker at each region start (KMP_INHERIT_FP_CONTROL), so a
       worker cannot drift there. The gate must report exactly what the library's own parallel loops would see. */
    int drift = 0;
#pragma omp parallel num_threads(4) reduction(| : drift)
    {
#if defined(__x86_64__)
        if (omp_get_thread_num() != 0) drift |= (int)((nr_csr_get() >> 15) & 1u);
#else
        if (omp_get_thread_num() != 0) drift |= fegetround() != FE_TONEAREST;
#endif
    }
    CHECK(((wk & 64) != 0) == (drift != 0), "fpenv reports OpenMP worker drift exactly when it persists (%d: %s; drift %d)", wk, why, drift);
#if defined(__GNUC__) && !defined(__clang__)
    CHECK(!fp_modelled || drift, "libgomp keeps worker FP state, so the worker check is exercised (drift %d)", drift);
#endif
    CHECK(tsr_fpenv_canonicalize() == 0, "canonicalize repairs the workers");
    tsr_set_threads(th);
#endif
    CHECK(tsr_fpenv_check(NULL, 0) == 0, "fpenv canonical again");

    /* (3) float -> int casts: NumPy's x86-64 results, defined where the plain C cast is undefined behaviour */
    double fv[10] = {NAN, INFINITY, -INFINITY, 1e300, -1.0, 300.0, -2.5, 9223372036854775808.0, 1e10, 2147483648.0};
    int64_t n10[1] = {10}, s8[1] = {8}, s4[1] = {4}, s1[1] = {1};
    int64_t i64[10]; int32_t i32[10]; uint8_t u8[10];
    tsr_copy(TSR_F64, TSR_I64, 1, n10, fv, s8, i64, s8);
    tsr_copy(TSR_F64, TSR_I32, 1, n10, fv, s8, i32, s4);
    tsr_copy(TSR_F64, TSR_U8, 1, n10, fv, s8, u8, s1);
    CHECK(i64[0] == INT64_MIN && i64[1] == INT64_MIN && i64[2] == INT64_MIN && i64[3] == INT64_MIN && i64[4] == -1
          && i64[5] == 300 && i64[6] == -2 && i64[7] == INT64_MIN && i64[8] == 10000000000LL && i64[9] == 2147483648LL,
          "f64 -> int64 saturates like cvttsd2si");
    CHECK(i32[0] == INT32_MIN && i32[3] == INT32_MIN && i32[5] == 300 && i32[6] == -2 && i32[8] == INT32_MIN
          && i32[9] == INT32_MIN, "f64 -> int32 saturates like cvttsd2si");
    CHECK(u8[0] == 0 && u8[4] == 255 && u8[5] == 44 && u8[6] == 254 && u8[8] == 0, "f64 -> uint8 via int32 as NumPy");
    float ff[3] = {NAN, 3e9f, -300.0f};
    tsr_copy(TSR_F32, TSR_I32, 1, (int64_t[]){3}, ff, s4, i32, s4);
    tsr_copy(TSR_F32, TSR_U8, 1, (int64_t[]){3}, ff, s4, u8, s1);
    CHECK(i32[0] == INT32_MIN && i32[1] == INT32_MIN && i32[2] == -300 && u8[2] == 212, "f32 -> int32/uint8");
    double cz[2] = {NAN, 1.0};
    tsr_copy(TSR_C128, TSR_I64, 1, s1, cz, (int64_t[]){16}, i64, s8);
    CHECK(i64[0] == INT64_MIN, "complex(NaN) -> int64");
    double big2[2] = {1e300, 1e300};
    int64_t rsum = 0;
    CHECK(tsr_reduce(R_SUM, TSR_F64, 1, (int64_t[]){2}, big2, s8, 0, TSR_I64, &rsum, s8) == 0 && rsum == INT64_MIN,
          "float sum stored to int64 saturates");
    tsr_arange(TSR_I64, 2, NAN, 1.0, i64);
    CHECK(i64[0] == INT64_MIN && i64[1] == INT64_MIN + 1, "arange from a NaN start is defined");
    /* the packed (SSE) contiguous paths agree with the scalar helpers, NaN / out-of-range lanes included */
    double vd[37]; float vf[37]; int32_t vi[37]; uint8_t vu[37]; int okv = 1;
    for (int i = 0; i < 37; i++) { vd[i] = i % 5 == 0 ? NAN : i % 7 == 0 ? 3e9 : i % 11 == 0 ? -1e300 : (i - 18) * 37.75; vf[i] = (float)vd[i]; }
    tsr_copy(TSR_F64, TSR_I32, 1, (int64_t[]){37}, vd, s8, vi, s4);
    tsr_copy(TSR_F64, TSR_U8, 1, (int64_t[]){37}, vd, s8, vu, s1);
    for (int i = 0; i < 37; i++) okv &= vi[i] == tsr_f2i32(vd[i]) && vu[i] == tsr_f2u8(vd[i]);
    tsr_copy(TSR_F32, TSR_I32, 1, (int64_t[]){37}, vf, s4, vi, s4);
    tsr_copy(TSR_F32, TSR_U8, 1, (int64_t[]){37}, vf, s4, vu, s1);
    for (int i = 0; i < 37; i++) okv &= vi[i] == tsr_f2i32f(vf[i]) && vu[i] == tsr_f2u8f(vf[i]);
    CHECK(okv && vi[0] == INT32_MIN && vu[19] == (uint8_t)(int32_t)37.75, "packed casts == scalar helpers");

    /* (4) complex division: Smith's method (NumPy) instead of |y|^2, which overflowed */
    double cx[4] = {1, 1, 1, 1}, cy[4] = {1e200, 1e200, 1e-200, 1e-200}, cq[4];
    int64_t s16[1] = {16}, n2[1] = {2};
    tsr_binary(OP_DIV, TSR_C128, 1, n2, cx, s16, cy, s16, cq, s16);
    CHECK(CLOSE(cq[0], 1e-200, 1e-15) && cq[1] == 0.0, "(1+1j)/(1e200+1e200j) = %g%+gj", cq[0], cq[1]);
    CHECK(CLOSE(cq[2], 1e200, 1e-15) && cq[3] == 0.0, "(1+1j)/(1e-200+1e-200j) = %g%+gj", cq[2], cq[3]);

    /* (5) float floor_divide / remainder = npy_divmod: 1.0 // 0.1 == 9, signs, zero divisor, infinities */
    double fa[9] = {1.0, -1.0, -2.0, 2.0, 1.0, 0.0, 1.0, -1.0, INFINITY};
    double fb[9] = {0.1, 0.1, 1.0, -1.0, 0.0, 0.0, INFINITY, INFINITY, 1.0};
    double fq[9], fm[9];
    int64_t n9[1] = {9};
    tsr_binary(OP_FLOORDIV, TSR_F64, 1, n9, fa, s8, fb, s8, fq, s8);
    tsr_binary(OP_MOD, TSR_F64, 1, n9, fa, s8, fb, s8, fm, s8);
    CHECK(fq[0] == 9.0 && nr_bits_eq(fm[0], 0x1.9999999999996p-4), "1.0 // 0.1 = %g, %% = %a", fq[0], fm[0]);
    CHECK(fq[1] == -10.0 && fm[1] == 0x1p-54, "-1.0 // 0.1 = %g, %% = %a", fq[1], fm[1]);
    CHECK(fq[2] == -2.0 && nr_bits_eq(fm[2], 0.0) && fq[3] == -2.0 && nr_bits_eq(fm[3], -0.0), "zero remainder takes the divisor's sign");
    CHECK(fq[4] == INFINITY && isnan(fm[4]) && isnan(fq[5]) && isnan(fm[5]), "division by zero: inf/nan, remainder nan");
    CHECK(nr_bits_eq(fq[6], 0.0) && fm[6] == 1.0 && fq[7] == -1.0 && fm[7] == INFINITY, "infinite divisor");
    CHECK(isnan(fq[8]) && isnan(fm[8]), "inf // 1 is nan");
    for (int i = 0; i < 4; i++) CHECK(CLOSE(fq[i] * fb[i] + fm[i], fa[i], 1e-15), "x == (x // y) * y + x %% y at %d", i);
    float ga[3] = {1.0f, -7.5f, 0.0f}, gb[3] = {0.1f, 2.0f, -3.0f}, gq[3], gm[3];
    tsr_binary(OP_FLOORDIV, TSR_F32, 1, (int64_t[]){3}, ga, s4, gb, s4, gq, s4);
    tsr_binary(OP_MOD, TSR_F32, 1, (int64_t[]){3}, ga, s4, gb, s4, gm, s4);
    CHECK(gq[0] == 9.0f && nr_bits_eqf(gm[0], 0x1.999996p-4f) && gq[1] == -4.0f && gm[1] == 0.5f
          && nr_bits_eqf(gq[2], -0.0f) && nr_bits_eqf(gm[2], -0.0f), "float32 floor_divide / remainder");

    /* (6) norm(ord=+-inf) propagates NaN like abs(x).max() / .min() */
    double nv[3] = {1.0, NAN, 3.0};
    int64_t sh3[1] = {3};
    CHECK(isnan(nr_routine_num("linalg.norm", 2, (tsr_arg[]){arg_arr(nv, 1, sh3), arg_num(INFINITY)})), "norm inf NaN");
    CHECK(isnan(nr_routine_num("linalg.norm", 2, (tsr_arg[]){arg_arr(nv, 1, sh3), arg_num(-INFINITY)})), "norm -inf NaN");
    tsr_arg vn[4] = {arg_arr(nv, 1, sh3), {0}, {0}, arg_num(INFINITY)};
    CHECK(isnan(nr_routine_num("linalg.vector_norm", 4, vn)), "vector_norm inf NaN");
    double mv[4] = {1.0, 2.0, NAN, 0.5};
    int64_t sh22[2] = {2, 2};
    CHECK(isnan(nr_routine_num("linalg.norm", 2, (tsr_arg[]){arg_arr(mv, 2, sh22), arg_num(INFINITY)})), "matrix norm inf NaN");
    CHECK(isnan(nr_routine_num("linalg.norm", 3, (tsr_arg[]){arg_arr(mv, 2, sh22), arg_num(-INFINITY), arg_num(0)})), "axis norm -inf NaN");
    CHECK(nr_routine_num("linalg.norm", 2, (tsr_arg[]){arg_arr((double[]){1, -4, 3}, 1, sh3), arg_num(INFINITY)}) == 4.0, "norm inf finite");

    /* (7) LAPACK lapack_int guard: a 2^31 x 0 matrix holds no data but its dimension would wrap in (lapack_int) */
    int64_t huge[2] = {(int64_t)1 << 31, 0};
    double dummy = 0;
    tsr_result lr[2];
    const int qr = tsr_fn_find("linalg.qr");
    CHECK(qr >= 0 && tsr_fn_routine(qr, 1, (tsr_arg[]){arg_arr(&dummy, 2, huge)}, 2, lr) == TSR_EARG
          && strstr(tsr_fn_error(), "LAPACK") != NULL, "qr refuses a dimension above INT32_MAX: %s", tsr_fn_error());
    CHECK(tsr_fn_routine(tsr_fn_find("linalg.norm"), 2, (tsr_arg[]){arg_arr(&dummy, 2, huge), arg_num(2)}, 1, lr) == TSR_EARG,
          "norm(ord=2) SVD path guarded");
    CHECK(tsr_fn_routine(tsr_fn_find("splinalg.svds"), 1, (tsr_arg[]){arg_arr(&dummy, 2, huge)}, 1, lr) == TSR_EARG,
          "svds guarded");
    double id2[4] = {2, 0, 0, 2};
    CHECK(nr_routine_num("linalg.det", 1, (tsr_arg[]){arg_arr(id2, 2, sh22)}) == 4.0, "guarded det still works");

    /* (8) nan_policy='omit' on paired samples of unequal length is refused before compaction reads past y */
    const int sp = tsr_fn_find("stats.spearmanrho");
    double px[5] = {1, 2, NAN, 4, 5}, py[5] = {2, 1, 4, NAN, 3}, ro = 0, po = 0;
    tsr_array pin[2], pout[2];
    tsr_array_init(&pin[0], TSR_F64, 1, (int64_t[]){5}, px);
    tsr_array_init(&pin[1], TSR_F64, 1, (int64_t[]){3}, py);
    tsr_array_init(&pout[0], TSR_F64, 0, s1, &ro);
    tsr_array_init(&pout[1], TSR_F64, 0, s1, &po);
    const uint64_t axm[2] = {1, 1};
    const double omit[2] = {0, 1};
    CHECK(sp >= 0 && tsr_fn_gufunc(sp, 2, pin, axm, 0, omit, 2, pout) == TSR_EARG, "paired omit refuses unequal lengths");
    tsr_array_init(&pin[1], TSR_F64, 1, (int64_t[]){5}, py);
    CHECK(tsr_fn_gufunc(sp, 2, pin, axm, 0, omit, 2, pout) == 0 && CLOSE(ro, 0.5, 1e-14), "paired omit drops NaN pairs (rho %g)", ro);

    /* (9) logsumexp: scipy gives NaN for any NaN (also beside -inf only) and -inf for an all -inf row */
    CHECK(isnan(nr_routine_num("special.logsumexp", 1, (tsr_arg[]){arg_arr((double[]){NAN, -INFINITY}, 1, (int64_t[]){2})})), "logsumexp [nan,-inf]");
    CHECK(isnan(nr_routine_num("special.logsumexp", 1, (tsr_arg[]){arg_arr((double[]){NAN, NAN}, 1, (int64_t[]){2})})), "logsumexp [nan,nan]");
    CHECK(nr_routine_num("special.logsumexp", 1, (tsr_arg[]){arg_arr((double[]){-INFINITY, -INFINITY}, 1, (int64_t[]){2})}) == -INFINITY, "logsumexp all -inf");
    CHECK(nr_routine_num("special.logsumexp", 1, (tsr_arg[]){arg_arr((double[]){INFINITY, -INFINITY}, 1, (int64_t[]){2})}) == INFINITY, "logsumexp [inf,-inf]");
    CHECK(CLOSE(nr_routine_num("special.logsumexp", 1, (tsr_arg[]){arg_arr((double[]){1, 2}, 1, (int64_t[]){2})}), 2.3132616875182228, 1e-15), "logsumexp [1,2] finite path");
    double lm[4] = {NAN, -INFINITY, -INFINITY, -INFINITY};
    tsr_result lres[1];
    tsr_arg ax1 = arg_num(1); ax1.flags = 1; ax1.ival = 1;
    if (tsr_fn_routine(tsr_fn_find("special.logsumexp"), 2, (tsr_arg[]){arg_arr(lm, 2, sh22), ax1}, 1, lres) == 0) {
        const double *o = (const double *)lres[0].arr.data;
        CHECK(isnan(o[0]) && o[1] == -INFINITY, "logsumexp axis=1 rows: %g %g", o[0], o[1]);
        tsr_free(lres[0].arr.data, lres[0].bytes);
    } else CHECK(0, "logsumexp axis=1 failed: %s", tsr_fn_error());
}

/* Fuzz, 2026-11: the solvers take k right-hand sides from b's shape[1] and read n*k values from their copy of b.
   A b with a trailing zero-length axis, (2, 8, 0), holds none, and a 0-d b's shape[1] is not part of its shape:
   matmul_toeplitz read past its copy (heap-buffer-overflow under ASan). Each is now refused with an error. */
void fn_results_free(tsr_result *res, int n);   /* fn.h */
static int rhs_call(const char *name, tsr_arg *args, int nargs)
{
    tsr_result r[4];
    const int rc = tsr_fn_routine(tsr_fn_find(name), nargs, args, 4, r);
    if (rc >= 0) fn_results_free(r, 4);
    return rc;
}

static void test_rhs_shape_regressions(void)
{
    double m2[4] = {4.0, 1.0, 1.0, 3.0}, c2[2] = {2.0, 0.5}, piv[2] = {0.0, 1.0}, one[1] = {1.0};
    const int64_t s22[2] = {2, 2}, s2[1] = {2}, s1[1] = {1}, s280[3] = {2, 8, 0}, s02[2] = {0, 2};
    const tsr_arg A = arg_arr(m2, 2, s22), C = arg_arr(c2, 1, s2), B = arg_arr(NULL, 3, s280), I1 = arg_arr(one, 1, s1);
    tsr_arg pair[2] = {A, B};                             /* multi_dot([A, B]) */
    tsr_arg seq; memset(&seq, 0, sizeof seq); seq.kind = 5; seq.count = 2; seq.items = pair;
    /* every routine the zero-length-axis sweep (fuzz/shape_sweep.c) found reading past a block, one call each with
       an array of shape (2, 8, 0) where its shape promises 2 x 8 values */
    struct { const char *name; tsr_arg args[3]; int n; const char *expect; } cases[] = {
        {"linalg.lstsq", {A, B}, 2, "fewer than"},
        {"linalg.multi_dot", {seq}, 1, "fewer than"},
        {"slinalg.lstsq", {A, B}, 2, "fewer than"},
        {"slinalg.cho_solve", {A, arg_int(0), B}, 3, "fewer than"},
        {"slinalg.cho_solve_banded", {A, arg_int(0), B}, 3, "fewer than"},
        {"slinalg.solveh_banded", {A, B}, 2, "fewer than"},
        {"slinalg.solve_triangular", {A, B}, 2, "fewer than"},
        {"slinalg.lu_solve", {A, arg_arr(piv, 1, s2), B}, 3, "fewer than"},
        {"slinalg.toeplitz", {B}, 1, "fewer than"},
        {"slinalg.solve_toeplitz", {C, B}, 2, "fewer than"},
        {"slinalg.solve_circulant", {C, B}, 2, "fewer than"},
        {"slinalg.matmul_toeplitz", {C, B}, 2, "fewer than"},
        {"signal.lfiltic", {B, I1, I1}, 3, "fewer than"},
        {"cluster.kmeans2", {A, arg_arr(NULL, 2, s02)}, 2, "at least one centroid"},
        {"splinalg.cg", {A, B}, 2, "fewer than"},
        {"splinalg.bicg", {A, B}, 2, "fewer than"},
        {"splinalg.bicgstab", {A, B}, 2, "fewer than"},
        {"splinalg.cgs", {A, B}, 2, "fewer than"},
        {"splinalg.minres", {A, B}, 2, "fewer than"},
        {"splinalg.gmres", {A, B}, 2, "fewer than"},
        {"splinalg.lgmres", {A, B}, 2, "fewer than"},
        {"splinalg.gcrotmk", {A, B}, 2, "fewer than"},
        {"splinalg.qmr", {A, B}, 2, "fewer than"},
        {"splinalg.tfqmr", {A, B}, 2, "fewer than"},
        {"splinalg.lsqr", {A, B}, 2, "fewer than"},
        {"splinalg.lsmr", {A, B}, 2, "fewer than"},
        {"splinalg.expm_multiply", {A, B}, 2, "fewer than"},
        {"splinalg.spsolve", {A, B}, 2, "fewer than"},
        {"splinalg.spsolve_triangular", {A, B}, 2, "fewer than"},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        const int rc = rhs_call(cases[i].name, cases[i].args, cases[i].n);
        CHECK(rc == TSR_EARG && strstr(tsr_fn_error(), cases[i].expect), "%s refuses a (2, 8, 0) argument (rc %d: %s)",
              cases[i].name, rc, tsr_fn_error());
    }
    /* the fuzzer's input: a 0-d x whose shape[0] (outside its shape) matches n, and shape[1] = 8 */
    tsr_arg x0 = arg_arr(one, 0, NULL);
    x0.arr.shape[0] = 2;
    x0.arr.shape[1] = 8;
    tsr_arg mt[2] = {C, x0};
    const int rc = rhs_call("slinalg.matmul_toeplitz", mt, 2);
    CHECK(rc == TSR_EARG && strstr(tsr_fn_error(), "x holds 1 values"), "matmul_toeplitz refuses a 0-d x (rc %d: %s)", rc, tsr_fn_error());
    /* no right-hand side, b (2, 0): an empty (2, 0) x, nothing read (it copied 2 values from the empty b) */
    const int64_t s20[2] = {2, 0};
    tsr_arg st[2] = {A, arg_arr(NULL, 2, s20)};
    tsr_result r0[1];
    const int rc0 = tsr_fn_routine(tsr_fn_find("slinalg.solve_triangular"), 2, st, 1, r0);
    CHECK(rc0 == 0 && r0[0].arr.ndim == 2 && r0[0].arr.shape[0] == 2 && r0[0].arr.shape[1] == 0,
          "solve_triangular with b of shape (2, 0) gives an empty (2, 0) x (rc %d: %s)", rc0, tsr_fn_error());
    if (rc0 == 0) fn_results_free(r0, 1);
    /* lu_solve: piv shorter than n, and a pivot outside [0, n) (dgetrs would swap with a row past x) */
    double piv_bad[2] = {0.0, 7.0}, b1_rhs[2] = {1.0, 2.0};
    tsr_arg lu1[3] = {A, arg_arr(NULL, 2, s20), arg_arr(b1_rhs, 1, s2)};
    tsr_arg lu2[3] = {A, arg_arr(piv_bad, 1, s2), arg_arr(b1_rhs, 1, s2)};
    int rcl = rhs_call("slinalg.lu_solve", lu1, 3);
    CHECK(rcl == TSR_EARG && strstr(tsr_fn_error(), "piv holds 0 values"), "lu_solve refuses a short piv (rc %d: %s)", rcl, tsr_fn_error());
    rcl = rhs_call("slinalg.lu_solve", lu2, 3);
    CHECK(rcl == TSR_EARG && strstr(tsr_fn_error(), "not a row index"), "lu_solve refuses a pivot outside [0, n) (rc %d: %s)", rcl, tsr_fn_error());
    /* valid right-hand sides still solve: (2,) and (2, 1) */
    const int64_t s21[2] = {2, 1};
    double b1[2] = {1.0, 2.0};
    tsr_arg ok[2] = {C, arg_arr(b1, 1, s2)};
    CHECK(rhs_call("slinalg.matmul_toeplitz", ok, 2) == 0, "matmul_toeplitz with x of shape (2,)");
    ok[1] = arg_arr(b1, 2, s21);
    CHECK(rhs_call("slinalg.solve_toeplitz", ok, 2) == 0, "solve_toeplitz with b of shape (2, 1)");
}

/* The registry's input gate (fn_gate.c, ADR 0016): every tsr_fn_* entry point refuses structurally invalid
   input before any routine, loop or core reads it, and still accepts the edge cases that are valid (empty
   arrays with no data, reversed views whose offset points at the end of the block, broadcast strides). */
void fn_results_free(tsr_result *res, int n);   /* fn.h */
int fn_gate_array(const tsr_array *a, int with_data, const char *what, int k);   /* fn.h */
static int fn_gate_array_for_test(const tsr_array *a) { return fn_gate_array(a, 2, "output", 0); }
static int gate_routine(int id, tsr_arg a)
{
    tsr_result r[4];
    const int rc = tsr_fn_routine(id, 1, &a, 4, r);
    if (rc >= 0) fn_results_free(r, 4);
    return rc;
}

static void test_arg_gate(void)
{
    const int uq = tsr_fn_find("np.unique");
    CHECK(uq >= 0, "np.unique is registered");
    double v[4] = {3.0, 1.0, 2.0, 1.0};
    const int64_t s4[1] = {4};
    tsr_arg ok = arg_arr(v, 1, s4);
    CHECK(gate_routine(uq, ok) == 0, "a valid array passes the gate");

    tsr_arg a = ok; a.kind = 9;
    CHECK(gate_routine(uq, a) == TSR_EARG && strstr(tsr_fn_error(), "argument 0"), "unknown kind (%s)", tsr_fn_error());
    a = arg_num(1.0); a.flags = 4;
    CHECK(gate_routine(uq, a) == TSR_EARG, "undefined flag bits on a number");
    a = arg_int(3); a.num = 4.0;
    CHECK(gate_routine(uq, a) == TSR_EARG, "integer and double disagree");
    a = arg_str("x"); a.str = NULL;
    CHECK(gate_routine(uq, a) == TSR_EARG, "NULL string");
    a = arg_none(); a.kind = 4; a.flags = 1;
    CHECK(gate_routine(uq, a) == TSR_EARG, "flag bits on a bool");
    a = ok; a.flags = 1;
    CHECK(gate_routine(uq, a) == TSR_EARG, "undefined flag bit on an array");
    a = ok; a.arr.dtype = 99;
    CHECK(gate_routine(uq, a) == TSR_ETYPE, "unknown dtype");
    a = ok; a.arr.ndim = 33;
    CHECK(gate_routine(uq, a) == TSR_EDIM, "ndim above 32");
    a = ok; a.arr.ndim = -1;
    CHECK(gate_routine(uq, a) == TSR_EARG, "negative ndim");
    a = ok; a.arr.shape[0] = -4;
    CHECK(gate_routine(uq, a) == TSR_EARG, "negative dimension");
    a = ok; a.arr.ndim = 2; a.arr.shape[0] = (int64_t)1 << 40; a.arr.shape[1] = (int64_t)1 << 40; a.arr.strides[1] = 0;
    CHECK(gate_routine(uq, a) == TSR_EARG, "element count overflow");
    a = ok; a.arr.data = NULL;
    CHECK(gate_routine(uq, a) == TSR_EARG, "NULL data with elements");
    a = ok; a.arr.strides[0] = -8;
    CHECK(gate_routine(uq, a) == TSR_EARG && strstr(tsr_fn_error(), "before the start"), "reversed view reaching before its block (%s)", tsr_fn_error());
    a = ok; a.arr.offset = 3;
    CHECK(gate_routine(uq, a) == TSR_EARG && strstr(tsr_fn_error(), "aligned"), "misaligned float64 elements");
    a = ok; a.arr.strides[0] = INT64_MAX / 2;
    CHECK(gate_routine(uq, a) == TSR_EARG, "stride extent overflow");
    a = ok; a.arr.offset = INT64_MAX - 8;
    CHECK(gate_routine(uq, a) == TSR_EARG, "offset overflow");
    tsr_arg nest[5];
    for (int k = 0; k < 5; k++) { nest[k] = arg_none(); nest[k].kind = 5; nest[k].count = 1; nest[k].items = k < 4 ? &nest[k + 1] : &ok; }
    CHECK(gate_routine(uq, nest[0]) == TSR_EARG, "sequences nested deeper than 4");
    a = arg_none(); a.kind = 5; a.count = 2; a.items = NULL;
    CHECK(gate_routine(uq, a) == TSR_EARG, "sequence items NULL");
    a.count = -1; a.items = &ok;
    CHECK(gate_routine(uq, a) == TSR_EARG, "negative sequence length");
    tsr_result r[2];
    CHECK(tsr_fn_routine(uq, -1, &ok, 2, r) == TSR_EARG, "negative argument count");

    /* valid edge cases still pass */
    const int64_t s0[1] = {0};
    a = arg_arr(NULL, 1, s0);
    CHECK(gate_routine(uq, a) == 0, "an empty array may have no data");
    a = ok; a.arr.offset = 24; a.arr.strides[0] = -8;            /* v[::-1] */
    tsr_result rr[4];
    int rc = tsr_fn_routine(uq, 1, &a, 4, rr);
    CHECK(rc == 0 && rr[0].kind == 3 && rr[0].arr.shape[0] == 3, "a reversed view with its offset at the end passes (%d)", rc);
    if (rc == 0) fn_results_free(rr, 4);

    /* hyp1f1 at x = +inf returns (NaN) instead of looping ~10^16 times in Boost's final rescaling (SciPy hangs) */
    {
        const int h1 = tsr_fn_find("special.hyp1f1");
        double hv[8] = {0.5, 1.5, INFINITY, 0.0, 3.7, -1.5, INFINITY, 0.0};
        void *hp[4] = {&hv[0], &hv[1], &hv[2], &hv[3]}, *hq[4] = {&hv[4], &hv[5], &hv[6], &hv[7]};
        CHECK(h1 >= 0 && tsr_fn_ufunc(h1, 0, TSR_F64, 0, NULL, 4, hp, NULL) == 0 && isnan(hv[3]), "hyp1f1(0.5, 1.5, inf) returns");
        CHECK(h1 >= 0 && tsr_fn_ufunc(h1, 0, TSR_F64, 0, NULL, 4, hq, NULL) == 0 && isinf(hv[7]) && hv[7] > 0, "hyp1f1(3.7, -1.5, inf) returns (inf)");
    }

    /* ufunc: the strided operand set */
    const int erf = tsr_fn_find("special.erf");
    double x[4] = {0.0, 0.5, 1.0, 2.0}, y[4];
    void *data[2] = {x, y};
    int64_t shape[1] = {4}, strides[2] = {8, 8};
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 1, shape, 2, data, strides) == 0, "ufunc: valid operands pass");
    shape[0] = -4;
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 1, shape, 2, data, strides) == TSR_EARG, "ufunc: negative dimension");
    shape[0] = 4; strides[0] = INT64_MAX / 2;
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 1, shape, 2, data, strides) == TSR_EARG, "ufunc: stride overflow");
    strides[0] = 0;                                               /* broadcast input: valid */
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 1, shape, 2, data, strides) == 0, "ufunc: a broadcast (stride 0) input passes");
    int64_t big[2] = {(int64_t)1 << 40, (int64_t)1 << 40}, st2[4] = {0, 0, 0, 0};
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 2, big, 2, data, st2) == TSR_EARG, "ufunc: element count overflow");
    /* an empty operand set visits no element, so the gate accepts any stride; the element loop must not do
       stride arithmetic on it either (fuzz_abi: INT64_MAX * 8 in the dimension-merge test, signed overflow) */
    int64_t zsh[2] = {0, 8}, zst[4] = {8, INT64_MAX, 8, INT64_MAX};
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 2, zsh, 2, data, zst) == 0, "ufunc: a zero-length axis with an extreme stride is a no-op");
    /* operands are loaded through typed pointers: misaligned steps or first elements are refused (fuzz_abi: rvs) */
    int64_t mst[2] = {4, 8};
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 1, shape, 2, data, mst) == TSR_EARG && strstr(tsr_fn_error(), "aligned"),
          "ufunc: a misaligned float64 step (%s)", tsr_fn_error());
    void *mdata[2] = {(char *)x + 1, y};
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 0, NULL, 2, mdata, NULL) == TSR_EARG, "ufunc: a misaligned 0-d operand");
    /* outputs may not alias through a zero stride (write races, unbounded work); inputs may (broadcast) */
    int64_t ost[2] = {8, 0};
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 1, shape, 2, data, ost) == TSR_EARG && strstr(tsr_fn_error(), "alias"),
          "ufunc: a zero-stride output is refused (%s)", tsr_fn_error());
    int64_t one4[1] = {1}, ost1[2] = {8, 0};
    CHECK(tsr_fn_ufunc(erf, 0, TSR_F64, 1, one4, 2, data, ost1) == 0, "ufunc: a zero stride on a length-1 output axis is fine");

    /* gufunc: input and output arrays */
    const int med = tsr_fn_find("np.median");
    tsr_array in, out;
    tsr_array_init(&in, TSR_F64, 1, s4, v);
    double m = 0.0;
    tsr_array_init(&out, TSR_F64, 0, NULL, &m);
    const uint64_t axm[1] = {1};
    CHECK(tsr_fn_gufunc(med, 1, &in, axm, 0, NULL, 1, &out) == 0 && m == 1.5, "gufunc: valid arrays pass (median %g)", m);
    tsr_array bad = in; bad.dtype = 42;
    CHECK(tsr_fn_gufunc(med, 1, &bad, axm, 0, NULL, 1, &out) == TSR_ETYPE, "gufunc: unknown input dtype");
    bad = in; bad.shape[0] = -1;
    CHECK(tsr_fn_gufunc(med, 1, &bad, axm, 0, NULL, 1, &out) == TSR_EARG, "gufunc: negative input dimension");
    tsr_array bo = out; bo.offset = 4;
    CHECK(tsr_fn_gufunc(med, 1, &in, axm, 0, NULL, 1, &bo) == TSR_EARG, "gufunc: misaligned output");
    bo = out; bo.data = NULL;
    CHECK(tsr_fn_gufunc(med, 1, &in, axm, 0, NULL, 1, &bo) == TSR_EARG, "gufunc: NULL output data");
    {
        tsr_array ob;                                             /* a length-3 output with stride 0 */
        const int64_t s3[1] = {3};
        double m3 = 0.0;
        tsr_array_init(&ob, TSR_F64, 1, s3, &m3);
        ob.strides[0] = 0;
        CHECK(fn_gate_array_for_test(&ob) == TSR_EARG, "gufunc: a zero-stride output array is refused");
    }
    CHECK(tsr_fn_gufunc(med, 2, &in, axm, 0, NULL, 1, &out) == TSR_EARG, "gufunc: wrong input count");
    /* empty core slices: two loop iterations over an input with no elements, NULL data and a negative offset (all
       valid for an empty array). The gather must form no address from NULL (fuzz_abi; the sanitize build checks it). */
    tsr_array em;
    const int64_t s20[2] = {2, 0};
    tsr_array_init(&em, TSR_F64, 2, s20, NULL);
    em.offset = -16;
    em.strides[0] = INT64_MIN;                                   /* any stride is valid on an empty array */
    double m2[2] = {0.0, 0.0};
    const int64_t s2[1] = {2};
    tsr_array out2;
    tsr_array_init(&out2, TSR_F64, 1, s2, m2);
    const uint64_t ax1[1] = {2};
    const int rce = tsr_fn_gufunc(med, 1, &em, ax1, 0, NULL, 1, &out2);
    CHECK(rce == 0 || rce == TSR_EARG, "gufunc: median over empty slices of a NULL-data array returns cleanly (%d)", rce);
    int32_t ond[1];
    int64_t osh[32];
    bad = in; bad.ndim = 40;
    CHECK(tsr_fn_gufunc_shape(med, 1, &bad, axm, 0, NULL, ond, osh) == TSR_EDIM, "gufunc_shape: ndim above 32");
    bad = in; bad.data = NULL;
    CHECK(tsr_fn_gufunc_shape(med, 1, &bad, axm, 0, NULL, ond, osh) == 0, "gufunc_shape reads no data");

    /* random and rvs */
    const int sn = tsr_fn_find("random.standard_normal");
    uint64_t st[6];
    tsr_pcg64_seed(st, 0, 7, 0, 1);
    double g[4];
    void *gd[1] = {g};
    int64_t gs[1] = {4}, gst[1] = {8};
    CHECK(tsr_fn_random(sn, st, NULL, 1, gs, 1, gd, gst, TSR_F64) == 0, "random: valid output passes");
    CHECK(tsr_fn_random(sn, st, NULL, 1, gs, 1, gd, gst, 7) == TSR_ETYPE, "random: unknown output dtype");
    CHECK(tsr_fn_random(sn, st, NULL, 1, gs, 0, gd, gst, TSR_F64) == TSR_EARG, "random: no operands");
    gs[0] = -2;
    CHECK(tsr_fn_random(sn, st, NULL, 1, gs, 1, gd, gst, TSR_F64) == TSR_EARG, "random: negative dimension");
    gs[0] = (int64_t)1 << 40; gst[0] = 0;                         /* 2^40 draws into one slot: refused, not run */
    CHECK(tsr_fn_random(sn, st, NULL, 1, gs, 1, gd, gst, TSR_F64) == TSR_EARG, "random: a zero-stride output is refused");
    gs[0] = 4; gst[0] = 8;
    const int norm = tsr_fn_find("stats.norm");
    double loc = 0.0, scale = 1.0, z[4];
    void *nd[3] = {&loc, &scale, z};
    int64_t ns[1] = {4}, nst[3] = {0, 0, 8};
    CHECK(tsr_fn_rvs(norm, st, 1, ns, 3, nd, nst) == 0, "rvs: valid operands pass");
    nd[1] = NULL;
    CHECK(tsr_fn_rvs(norm, st, 1, ns, 3, nd, nst) == TSR_EARG, "rvs: NULL operand");
    nd[1] = &scale;
    ns[0] = INT64_MAX / 16;                                       /* the scratch size (np + 1) * n * 8 overflowed */
    nst[2] = 8;                                                   /* a real output step: the gate passes, the scratch size overflows */
    CHECK(tsr_fn_rvs(norm, st, 1, ns, 3, nd, nst) == TSR_ENOMEM, "rvs: scratch size overflow is refused");
    CHECK(tsr_allocated() == 0, "the gate leaks nothing (%lld bytes)", (long long)tsr_allocated());
}

/* Memory ownership (ADR 0015): the layer that allocates memory frees it. The teardown of routine results is
   the kernel's own (tsr_fn_results_free), every failed tsr_fn_routine leaves its result slots empty, and in an
   ownership-checked build (make sanitize) each kind of violation is a hard, reported failure. */
#if !defined(_WIN32)
#  include <sys/wait.h>
/* run `violate` in a child with stderr captured; 1 when the child died (did not exit 0), and the captured
   text in `msg` */
static int own_child_dies(void (*violate)(void), char *msg, size_t cap)
{
    int fd[2];
    if (pipe(fd) != 0) return 0;
    fflush(NULL);
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(fd[1], 2);
        close(fd[0]);
        violate();
        _exit(0);                                    /* survived: the violation went unnoticed */
    }
    close(fd[1]);
    size_t n = 0;
    ssize_t r;
    while (n + 1 < cap && (r = read(fd[0], msg + n, cap - 1 - n)) > 0) n += (size_t)r;
    msg[n] = '\0';
    char sink[256];
    while (read(fd[0], sink, sizeof sink) > 0) {
    }
    close(fd[0]);
    int status = 0;
    if (pid < 0 || waitpid(pid, &status, 0) != pid) return 0;
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}
static void own_free_libc(void) { void *p = tsr_alloc(100); free(p); }                         /* (i) */
static void own_free_foreign(void) { void *p = malloc(100); tsr_free(p, 100); }               /* (ii) */
static void own_free_wrong_size(void) { void *p = tsr_alloc(100); tsr_free(p, 64); }          /* (iii) */
static void own_free_twice(void) { void *p = tsr_alloc(100); tsr_free(p, 100); tsr_free(p, 100); }   /* (iv) */
#endif

static void test_ownership(void)
{
    const int64_t base = tsr_allocated();
    /* a kind 6 sequence holding a kind 2 string and a kind 3 array, as a routine would return it */
    tsr_result r[2];
    memset(r, 0, sizeof r);
    tsr_result *items = tsr_calloc(2 * (int64_t)sizeof(tsr_result));
    CHECK(items != NULL, "sequence block");
    if (items) {
        r[0].kind = 6; r[0].bytes = 2 * (int64_t)sizeof(tsr_result); r[0].arr.data = items;
        r[0].arr.ndim = 1; r[0].arr.shape[0] = 2;
        items[0].kind = 2; items[0].bytes = 8; items[0].arr.data = tsr_alloc(8);
        if (items[0].arr.data) memcpy(items[0].arr.data, "float64", 8);
        items[1].kind = 3; items[1].bytes = 64; items[1].arr.data = tsr_alloc(64);
        r[1].kind = 2; r[1].bytes = 4; r[1].arr.data = tsr_alloc(4);
        CHECK(tsr_allocated() > base, "results allocated");
        tsr_fn_results_free(r, 2);
        CHECK(tsr_allocated() == base && r[0].kind == 0 && r[1].kind == 0,
              "tsr_fn_results_free releases strings, arrays and sequences with their items (%lld bytes left)",
              (long long)(tsr_allocated() - base));
        tsr_fn_results_free(r, 2);                    /* reset to kind 0: a second call is a no-op */
        CHECK(tsr_allocated() == base, "tsr_fn_results_free twice is a no-op");
    }
    tsr_fn_results_free(NULL, 3);

    /* a failing routine leaves every slot empty (also an unknown id, which fails before any work) */
    tsr_result g[3];
    memset(g, 0x5a, sizeof g);
    CHECK(tsr_fn_routine(-1, 0, NULL, 3, g) < 0 && g[0].kind == 0 && g[1].kind == 0 && g[2].kind == 0,
          "an unknown routine id fails with empty results");
    const int tk = tsr_fn_find("windows.tukey");
    if (tk >= 0) {
        tsr_arg ta[2];
        memset(ta, 0, sizeof ta);
        ta[0].kind = 1; ta[0].num = -3.0; ta[1].kind = 1; ta[1].num = 0.5;
        memset(g, 0x5a, sizeof g);
        const int rc = tsr_fn_routine(tk, 2, ta, 1, g);
        CHECK(rc < 0 && g[0].kind == 0 && tsr_allocated() == base,
              "a routine that fails frees its partial results itself (rc %d, %lld bytes left)", rc,
              (long long)(tsr_allocated() - base));
        tsr_fn_results_free(g, 1);
    }

    CHECK(tsr_feature("ownership_check") == tsr_ownership_check(), "feature(\"ownership_check\") matches the build");
#if !defined(_WIN32)
    if (tsr_ownership_check()) {
        char msg[4096];
        CHECK(own_child_dies(own_free_libc, msg, sizeof msg), "free() of a tsr_alloc block is caught");
        CHECK(own_child_dies(own_free_foreign, msg, sizeof msg) && strstr(msg, "ownership violation") &&
              strstr(msg, "did not return"), "tsr_free of a malloc block is caught: %.200s", msg);
        CHECK(own_child_dies(own_free_wrong_size, msg, sizeof msg) && strstr(msg, "byte count"),
              "tsr_free with the wrong size is caught: %.200s", msg);
        CHECK(own_child_dies(own_free_twice, msg, sizeof msg) && strstr(msg, "already freed"),
              "a double tsr_free is caught: %.200s", msg);
    }
#endif
}


/* Opt-in GPU-matching profile (2026-10): flush scope and the subnormal / NaN helpers. */
static uint32_t fp_bits(float v) { uint32_t b; memcpy(&b, &v, sizeof b); return b; }
static float fp_from(uint32_t b) { float v; memcpy(&v, &b, sizeof v); return v; }
static void test_fp_flush_profile(void)
{
    CHECK(tsr_feature("fp_flush_profile") == 1, "feature fp_flush_profile");
    float s[7] = {fp_from(0x000AE398u), fp_from(0x800AE398u), 1.0f, 0.0f, fp_from(0x80000000u),
                  fp_from(0x00800000u), fp_from(0x00000001u)};
    CHECK(tsr_flush_subnormal_f32(s, 7) == 3, "flush_subnormal counts 3");
    CHECK(fp_bits(s[0]) == 0u && fp_bits(s[1]) == 0x80000000u && fp_bits(s[6]) == 0u, "subnormals -> signed zero");
    CHECK(s[2] == 1.0f && fp_bits(s[5]) == 0x00800000u && fp_bits(s[4]) == 0x80000000u, "normals and zeros untouched");
    CHECK(tsr_flush_subnormal_f32(NULL, 3) == 0 && tsr_flush_subnormal_f32(s, 0) == 0, "flush_subnormal degenerate");

    float q[6] = {1.0f, fp_from(0x7FC00001u), fp_from(0xFFC00000u), fp_from(0x7F800001u),
                  fp_from(0x7F800000u), fp_from(0x7FC00000u)};
    CHECK(tsr_nan_first_f32(q, 6) == 1, "nan_first finds index 1");
    CHECK(tsr_canon_nan_f32(q, 6) == 3, "canon_nan rewrites 3 non-canonical NaNs");
    CHECK(fp_bits(q[1]) == 0x7FC00000u && fp_bits(q[2]) == 0x7FC00000u && fp_bits(q[3]) == 0x7FC00000u,
          "NaN payloads, sign and signalling NaN -> 0x7FC00000");
    CHECK(fp_bits(q[4]) == 0x7F800000u && q[0] == 1.0f, "infinity and normals untouched");
    CHECK(tsr_nan_first_f32(q, 1) == -1 && tsr_nan_first_f32(NULL, 4) == -1, "nan_first none / NULL");

    {   /* f64 twins */
        const uint64_t sub = 0x000123456789ABCDull, nsub = 0x800123456789ABCDull, dmin = 0x0010000000000000ull;
        const uint64_t q1 = 0x7FF8000000000001ull, q2 = 0xFFF8000000000000ull, sn = 0x7FF0000000000001ull, inf = 0x7FF0000000000000ull;
        uint64_t raw[6] = {sub, nsub, dmin, q1, q2, sn}; double d[6]; memcpy(d, raw, sizeof d);
        CHECK(tsr_flush_subnormal_f64(d, 6) == 2, "f64 flush counts 2");
        memcpy(raw, d, sizeof raw);
        CHECK(raw[0] == 0u && raw[1] == 0x8000000000000000ull && raw[2] == dmin, "f64 subnormals -> signed zero, min normal kept");
        CHECK(tsr_nan_first_f64(d, 6) == 3, "f64 nan_first finds index 3");
        CHECK(tsr_canon_nan_f64(d, 6) == 3, "f64 canon_nan rewrites 3");
        memcpy(raw, d, sizeof raw);
        CHECK(raw[3] == 0x7FF8000000000000ull && raw[4] == 0x7FF8000000000000ull && raw[5] == 0x7FF8000000000000ull, "f64 NaN -> 0x7FF8000000000000");
        double one_inf; memcpy(&one_inf, &inf, sizeof one_inf);
        CHECK(tsr_canon_nan_f64(&one_inf, 1) == 0 && tsr_nan_first_f64(&one_inf, 1) == -1, "f64 infinity untouched");
    }
    uint64_t saved = 0;
    CHECK(tsr_fpenv_flush_enter(NULL) == -1, "flush_enter rejects NULL");
#if defined(__x86_64__) || defined(__aarch64__)
    /* volatile stores pin each multiply inside its scope: C does not model the FP control word, so an
       optimiser may otherwise compute a pure product after tsr_fpenv_flush_leave */
    volatile float tiny = fp_from(0x000AE398u), one = 1.0f, in_scope, after;
    CHECK(tsr_fpenv_flush_enter(&saved) == 0, "flush_enter");
    in_scope = tiny * one;
    const int env_in = tsr_fpenv_check_thread(NULL, 0);
    CHECK(tsr_fpenv_flush_leave(saved) == 0, "flush_leave");
    after = tiny * one;
    if (env_in != 0) {   /* 0 only under an emulator that ignores FP control writes */
        CHECK(fp_bits(in_scope) == 0u, "inside the scope a subnormal product flushes to zero (%08x)", fp_bits(in_scope));
        CHECK(env_in & 1, "fpenv reports flush-to-zero inside the scope (%d)", env_in);
    }
    CHECK(fp_bits(after) == 0x000AE398u, "after leave the subnormal is preserved again (%08x)", fp_bits(after));
    CHECK(tsr_fpenv_check_thread(NULL, 0) == 0, "fpenv canonical again after leave");
#endif
}

int main(void)
{
    test_alloc();
    test_binary();
    test_unary_cast();
    test_reduce();
    test_index();
    test_matmul();
    test_gemm();
    test_axpy();
    test_fft();
    test_rng();
    test_sparse();
    test_array_meta();
    test_threads();
    test_lp();
    test_mdp();
    test_ufunc();
    test_mmap_npy();
    test_slice_blanks();
    test_fmaf_ro();
    test_vmath();
    test_rfft();
    test_size_overflow();
    test_numerics_review();
    test_fp_flush_profile();
    test_security_regressions();
    test_fuzz_batch_a();
    test_fuzz_batch_b();
    test_fuzz_batch_c();
    test_fuzz_batch_d();
    test_ci_regressions();
    test_abi_runtime();
    test_fft_trig();
    test_kdtree();
    test_splinalg_csr();
    test_spsolve_csr();
    test_eigsh_csr();
    test_linkage_fast();
    test_geom2d();
    test_fd_normal();
    test_label_two_pass();
    test_fft_axis();
    test_fftconvolve_fft();
    test_kelvin_zeros();
    test_bessel_airy_zeros();
    test_linalg_typed();
    test_op_names();
    test_sort_variants();
    test_sort_daz();
    test_sort_hardening();
#if !defined(_WIN32)
    test_sort_app_threads();
    test_sort_small_stacks();
#endif
    test_arg_gate();
    test_rhs_shape_regressions();
    test_ownership();
    CHECK(tsr_allocated() == 0, "no leaked tessero allocations (%ld bytes)", (long)tsr_allocated());
    printf("libtessero %s simd=%d: %d checks, %d failures\n", tsr_version(), tsr_simd_level(), checks, failures);
    return failures ? 1 : 0;
}
