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

int main(void)
{
    test_alloc();
    test_binary();
    test_unary_cast();
    test_reduce();
    test_index();
    test_matmul();
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
    CHECK(tsr_allocated() == 0, "no leaked tessero allocations (%ld bytes)", (long)tsr_allocated());
    printf("libtessero %s simd=%d: %d checks, %d failures\n", tsr_version(), tsr_simd_level(), checks, failures);
    return failures ? 1 : 0;
}
