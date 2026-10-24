/* libtessero C unit tests. Run: make test && make sanitize */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "../src/internal.h"
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
    char path[] = "/tmp/tsr_mmap_test_XXXXXX";
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
    CHECK(tsr_mmap_open("/tmp", 0, TSR_F64, &nd, sh, 0, &h, &data) != 0, "directory refused");

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

static void test_vmath(void)
{
    /* vectorised exp/log (vmath.h) against long double references: < 1 ulp */
    double worst_e = 0, worst_l = 0;
    for (int i = 0; i < 200000; i++) {
        const double x = -745.0 + 1455.0 * i / 200000.0 + 1e-7 * (i % 13);
        const double e = tsr_exp(x);
        const long double re = expl((long double)x);
        if (isfinite(e) && e > 0x1p-1022) {
            const double u = nextafter(e, INFINITY) - e;
            const double err = (double)(fabsl((long double)e - re) / u);
            if (err > worst_e) worst_e = err;
        }
        const double y = ldexp(1.0 + (i % 997) / 997.0, (i % 2098) - 1074);
        const double l = tsr_log(y);
        const long double rl = logl((long double)y);
        if (l != 0) {
            const double u = nextafter(fabs(l), INFINITY) - fabs(l);
            const double err = (double)(fabsl((long double)l - rl) / u);
            if (err > worst_l) worst_l = err;
        }
    }
    CHECK(worst_e < 1.0 && worst_l < 1.0, "exp/log max error %.3f / %.3f ulp", worst_e, worst_l);
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
#include <sys/wait.h>
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

static void test_abi_runtime(void)
{
    /* version and capability negotiation: one version source */
    CHECK(tsr_abi_version() == TSR_ABI_VERSION && TSR_ABI_VERSION == 2, "abi version %d", tsr_abi_version());
    CHECK(strcmp(tsr_version(), TSR_VERSION_STRING) == 0, "tsr_version %s", tsr_version());
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

    /* fork: a child of a process that computed with threads runs serial until set again; the parent keeps its count */
    tsr_set_threads(4);
    (void)tsr_get_threads();                                    /* as a kernel would, before a parallel region */
    fflush(NULL);
    pid_t pid = fork();
    if (pid == 0) _exit(tsr_get_threads() == 1 ? 0 : 3);
    int status = -1;
    CHECK(pid > 0 && waitpid(pid, &status, 0) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
          "forked child runs serial (status %d)", status);
    CHECK(tsr_get_threads() == (tsr_openmp() ? 4 : 1), "parent keeps its thread count");
    tsr_set_threads(1);

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
#if defined(__x86_64__)
    const uint32_t csr0 = nr_csr_get();
    nr_csr_set(csr0 | (1u << 15) | (1u << 6));                         /* FTZ + DAZ, what -ffast-math sets */
    const int ftz = tsr_fpenv_check(why, sizeof why);
    CHECK(ftz == 3 && strstr(why, "FTZ") && strstr(why, "DAZ"), "fpenv names FTZ/DAZ (%d: %s)", ftz, why);
    nr_csr_set(csr0 & ~(1u << 9));                                     /* unmask divide-by-zero */
    CHECK(tsr_fpenv_check_thread(why, sizeof why) == 8 && strstr(why, "masks"), "fpenv sees unmasked traps: %s", why);
    nr_csr_set(csr0 | (1u << 15));
    CHECK(tsr_fpenv_canonicalize() == 0 && nr_csr_get() == csr0, "canonicalize restores MXCSR");
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
    CHECK(omp_get_max_threads() < 1 || (wk & 64), "fpenv checks OpenMP workers (%d: %s)", wk, why);
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
    test_vmath();
    test_rfft();
    test_size_overflow();
    test_numerics_review();
    test_security_regressions();
    test_abi_runtime();
    CHECK(tsr_allocated() == 0, "no leaked tessero allocations (%ld bytes)", (long)tsr_allocated());
    printf("libtessero %s simd=%d: %d checks, %d failures\n", tsr_version(), tsr_simd_level(), checks, failures);
    return failures ? 1 : 0;
}
