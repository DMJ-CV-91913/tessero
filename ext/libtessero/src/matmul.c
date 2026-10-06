/* Cache-blocked matrix product used when no BLAS library is available. */
#include "internal.h"

#define BLOCK 64

#define MATMUL(NAME, T, FMA)                                                                         \
    TSR_CLONES static void NAME(int64_t m, int64_t n, int64_t k, const T *a, int64_t lda,       \
                                const T *b, int64_t ldb, T *c, int64_t ldc)                     \
    {                                                                                           \
        for (int64_t i = 0; i < m; i++) memset(c + i * ldc, 0, (size_t)n * sizeof(T));          \
        for (int64_t kk = 0; kk < k; kk += BLOCK)                                               \
            for (int64_t jj = 0; jj < n; jj += BLOCK)                                           \
                for (int64_t i = 0; i < m; i++) {                                               \
                    T *ci = c + i * ldc;                                                        \
                    int64_t kmax = kk + BLOCK < k ? kk + BLOCK : k;                             \
                    int64_t jmax = jj + BLOCK < n ? jj + BLOCK : n;                             \
                    for (int64_t p = kk; p < kmax; p++) {                                       \
                        T aip = a[i * lda + p];                                                 \
                        const T *bp = b + p * ldb;                                              \
                        for (int64_t j = jj; j < jmax; j++) ci[j] = FMA(T, ci[j], aip, bp[j]);               \
                    }                                                                           \
                }                                                                               \
    }

#define FFMA(T, c, a, b) ((c) + (a) * (b))
#define IFMA(T, c, a, b) TSR_WADD(T, c, TSR_WMUL(T, a, b))
MATMUL(mm_f64, double, FFMA)
MATMUL(mm_f32, float, FFMA)
MATMUL(mm_i64, int64_t, IFMA)

int tsr_matmul(int dtype, int64_t m, int64_t n, int64_t k, const void *a, int64_t lda, const void *b, int64_t ldb,
               void *c, int64_t ldc)
{
    switch (dtype) {
    case TSR_F64: mm_f64(m, n, k, a, lda, b, ldb, c, ldc); return TSR_OK;
    case TSR_F32: mm_f32(m, n, k, a, lda, b, ldb, c, ldc); return TSR_OK;
    case TSR_I64: mm_i64(m, n, k, a, lda, b, ldb, c, ldc); return TSR_OK;
    default: return TSR_ETYPE;
    }
}

/* General matrix multiply C = alpha * op(A) * op(B) + beta * C, row-major, matching the BLAS xgemm contract so
   this fallback is interchangeable with cblas_{s,d}gemm. op(A) is m x k, op(B) is k x n, C is m x n. A is read
   as m x k (lda >= k) when not transposed or k x m (lda >= m) when transposed; B as k x n (ldb >= n) or n x k
   (ldb >= k). When beta == 0, C is overwritten (its prior contents, which may be uninitialised, are not read). */
#define GEMM(NAME, T)                                                                                    \
    TSR_CLONES static void NAME(int ta, int tb, int64_t m, int64_t n, int64_t k,                         \
                                T alpha, const T *a, int64_t lda, const T *b, int64_t ldb,               \
                                T beta, T *c, int64_t ldc)                                               \
    {                                                                                                    \
        for (int64_t i = 0; i < m; i++) {                                                                \
            T *ci = c + i * ldc;                                                                         \
            if (beta == (T)0) { for (int64_t j = 0; j < n; j++) ci[j] = (T)0; }                          \
            else if (beta != (T)1) { for (int64_t j = 0; j < n; j++) ci[j] = beta * ci[j]; }             \
            for (int64_t p = 0; p < k; p++) {                                                            \
                T s = alpha * (ta ? a[p * lda + i] : a[i * lda + p]);                                    \
                if (tb) { for (int64_t j = 0; j < n; j++) ci[j] += s * b[j * ldb + p]; }                 \
                else { const T *bp = b + p * ldb; for (int64_t j = 0; j < n; j++) ci[j] += s * bp[j]; }  \
            }                                                                                            \
        }                                                                                                \
    }
GEMM(gemm_f64, double)
GEMM(gemm_f32, float)

int tsr_gemm(int dtype, int transa, int transb, int64_t m, int64_t n, int64_t k,
             double alpha, const void *a, int64_t lda, const void *b, int64_t ldb,
             double beta, void *c, int64_t ldc)
{
    const int ta = transa != 0 && transa != 111;   /* 0 or CBLAS NoTrans(111) -> no transpose; else transpose */
    const int tb = transb != 0 && transb != 111;
    switch (dtype) {
    case TSR_F64: gemm_f64(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc); return TSR_OK;
    case TSR_F32: gemm_f32(ta, tb, m, n, k, (float)alpha, a, lda, b, ldb, (float)beta, c, ldc); return TSR_OK;
    default: return TSR_ETYPE;
    }
}

TSR_CLONES double tsr_dot_f64(int64_t n, const double *x, int64_t incx, const double *y, int64_t incy)
{
    double s[4] = {0, 0, 0, 0};
    int64_t i = 0;
    if (incx == 1 && incy == 1) {
        for (; i + 4 <= n; i += 4)
            for (int j = 0; j < 4; j++) s[j] += x[i + j] * y[i + j];
    }
    double r = (s[0] + s[1]) + (s[2] + s[3]);
    for (; i < n; i++) r += x[i * incx] * y[i * incy];
    return r;
}
