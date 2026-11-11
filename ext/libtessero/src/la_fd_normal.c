/*
 * linalg.fd_normal, on its own so it builds with multiply-add contraction off (csrc/Makefile NOCONTRACT_C): its
 * sums must round each product separately, as Optimize\LeastSquares's PHP loop does, which it reproduces bit for
 * bit (tests/Unit/NumericsTest: the kernel and the PHP loop give the same iterations). aarch64 GCC and Apple clang
 * fuse g[i] += ji * r[k] into one multiply-add by default; x86-64 builds without -mfma never do. Moved unchanged
 * from np_linalg.c, whose registry table still lists it.
 */
#include "fn.h"

#include <string.h>

/* linalg.fd_normal(r, rh, h) -> A, g, J: the forward-difference Jacobian J[k][j] = (rh[j][k] - r[k]) / h[j] of the
   residuals r (m) from the residuals rh (n, m) at the steps h (n), and the normal equations A = J^T J, g = J^T r,
   each sum in the order of k with zero Jacobian entries skipped (as Optimize\LeastSquares's PHP loop did), for
   its Levenberg-Marquardt iterations. */
int r_fd_normal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 3 || args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 2
        || args[2].kind != 3 || args[2].arr.ndim != 1) {
        fn_set_error("fd_normal: r (m), rh (n, m) and h (n) must be arrays"); return TSR_EARG;
    }
    const int64_t m = args[0].arr.shape[0], n = args[2].arr.shape[0];
    if (args[1].arr.shape[0] != n || args[1].arr.shape[1] != m) {
        fn_set_error("fd_normal: rh must be (%lld, %lld)", (long long)n, (long long)m); return TSR_EARG;
    }
    int64_t cr, crh, ch;
    double *r = mat_f64(&args[0], "r", &cr); if (!r) return TSR_ENOMEM;
    double *rh = mat_f64(&args[1], "rh", &crh); if (!rh) { fn_free_doubles(r, cr); return TSR_ENOMEM; }
    double *h = mat_f64(&args[2], "h", &ch); if (!h) { fn_free_doubles(r, cr); fn_free_doubles(rh, crh); return TSR_ENOMEM; }
    int64_t sa[2] = {n, n}, sg[1] = {n}, sj[2] = {m, n};
    double *A = (double *)fn_result_array(&res[0], TSR_F64, 2, sa);
    double *g = A ? (double *)fn_result_array(&res[1], TSR_F64, 1, sg) : NULL;
    double *J = g ? (double *)fn_result_array(&res[2], TSR_F64, 2, sj) : NULL;
    if (!J) { fn_free_doubles(r, cr); fn_free_doubles(rh, crh); fn_free_doubles(h, ch); return TSR_ENOMEM; }
    for (int64_t k = 0; k < m; k++)
        for (int64_t j = 0; j < n; j++) J[k * n + j] = (rh[j * m + k] - r[k]) / h[j];
    memset(A, 0, sizeof(double) * (size_t)(n * n));
    memset(g, 0, sizeof(double) * (size_t)n);
    for (int64_t k = 0; k < m; k++) {
        const double *row = J + k * n;
        for (int64_t i = 0; i < n; i++) {
            const double ji = row[i];
            if (ji == 0.0) continue;
            g[i] += ji * r[k];
            for (int64_t j = i; j < n; j++) A[i * n + j] += ji * row[j];
        }
    }
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < i; j++) A[i * n + j] = A[j * n + i];
    fn_free_doubles(r, cr); fn_free_doubles(rh, crh); fn_free_doubles(h, ch);
    return TSR_OK;
}
