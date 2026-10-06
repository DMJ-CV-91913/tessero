/*
 * Linear algebra on row-major LAPACKE (float64), shared by both bindings through the registry (ADR 0011):
 * Tessero\Linalg (FFI) and Tessero\Ext\Linalg (extension). Each routine works on a private contiguous float64
 * copy and loops over the leading dimensions of stacked (..., M, M) inputs, exactly as NumPy's gufunc-based
 * numpy.linalg does. A singular or non-positive-definite matrix is reported through fn_set_error and a negative
 * return (the bindings raise; NumPy raises numpy.linalg.LinAlgError, and the parity runner accepts any raise
 * where NumPy raises).
 *
 * Semantics follow numpy.linalg (numpy/linalg/_linalg.py) and mirror the earlier FFI implementation in
 * src/Linalg/Linalg.php: solve (dgesv), inv (dgetrf+dgetri), det/slogdet (dgetrf), cholesky (dpotrf, lower).
 */
#include "fn.h"

#include <float.h>
#include <lapacke.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef TSR_MAXDIM
#define TSR_MAXDIM 32
#endif

/* the last two dimensions must be square; returns M and the number of stacked matrices (product of the rest) */
static int square_batch(const tsr_array *a, int64_t *m, int64_t *batch)
{
    if (a->ndim < 2) {
        fn_set_error("%d-dimensional array given. Array must be at least two-dimensional", (int)a->ndim);
        return TSR_EARG;
    }
    if (a->shape[a->ndim - 1] != a->shape[a->ndim - 2]) {
        fn_set_error("Last 2 dimensions of the array must be square");
        return TSR_EARG;
    }
    *m = a->shape[a->ndim - 1];
    int64_t b = 1;
    for (int d = 0; d < a->ndim - 2; d++) b *= a->shape[d];
    *batch = b;
    return TSR_OK;
}

/* a routine argument as a contiguous float64 copy; message and error on a non-array */
static double *mat_f64(const tsr_arg *a, const char *what, int64_t *n)
{
    if (a->kind != 3) { fn_set_error("%s must be an array", what); return NULL; }
    double *p = fn_arg_doubles(a, n);
    if (!p) fn_set_error("%s: out of memory", what);
    return p;
}

/* LU factorisation (row-major) of one M x M matrix in place; sign of the permutation and sum of log|diag|.
   Returns 1 on success, 0 if the matrix is singular (U has a zero pivot). */
static int lu_slogdet(double *a, int64_t m, double *sign, double *logabs)
{
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(m > 0 ? m : 1));
    if (!ipiv) return -1;
    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)m, a, (lapack_int)m, ipiv);
    if (info < 0) { free(ipiv); return -2; }
    if (info > 0) { free(ipiv); *sign = 0.0; *logabs = -INFINITY; return 0; }
    double s = 1.0, lg = 0.0;
    for (int64_t i = 0; i < m; i++) {
        if (ipiv[i] != (lapack_int)(i + 1)) s = -s;
        const double d = a[i * m + i];
        if (d < 0) s = -s;
        lg += log(fabs(d));
    }
    free(ipiv);
    *sign = s;
    *logabs = lg;
    return 1;
}

/* det / slogdet(a): stacked; det returns sign*exp(logabs), slogdet returns (sign, logabsdet). ctx: 1 = slogdet */
static int r_det(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nargs; (void)nres;
    const int slog = ctx && *(const int *)ctx;
    const tsr_arg *arg = &args[0];
    if (arg->kind != 3) { fn_set_error("det: input must be an array"); return TSR_ETYPE; }
    int64_t m, batch;
    int rc = square_batch(&arg->arr, &m, &batch);
    if (rc < 0) return rc;
    int64_t na;
    double *a = mat_f64(arg, "a", &na);
    if (!a) return TSR_ENOMEM;
    const int32_t nd = arg->arr.ndim - 2;                 /* leading dims -> result shape */
    int64_t sh[TSR_MAXDIM];
    for (int d = 0; d < nd; d++) sh[d] = arg->arr.shape[d];
    double *sgn = NULL, *lg = NULL, *dt = NULL;
    if (slog) {
        tsr_result *seq = res;                            /* two named outputs: sign, logabsdet */
        sgn = (double *)fn_result_array(&seq[0], TSR_F64, nd, sh);
        lg = (double *)fn_result_array(&seq[1], TSR_F64, nd, sh);
        if (!sgn || !lg) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    } else {
        dt = (double *)fn_result_array(&res[0], TSR_F64, nd, sh);
        if (!dt) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    }
    for (int64_t i = 0; i < batch; i++) {
        double s, l;
        int r = lu_slogdet(a + i * m * m, m, &s, &l);
        if (r == -1) { fn_free_doubles(a, na); return TSR_ENOMEM; }
        if (r == -2) { fn_free_doubles(a, na); fn_set_error("linalg: illegal value in dgetrf"); return TSR_EARG; }
        if (slog) { sgn[i] = s; lg[i] = l; }
        else dt[i] = s * exp(l);
    }
    fn_free_doubles(a, na);
    if (nd == 0) {                                        /* 2-D input -> scalar results */
        if (slog) {
            const double s = sgn[0], l = lg[0];
            fn_results_free(res, 2);
            fn_result_num(&res[0], s);
            fn_result_num(&res[1], l);
        } else {
            const double v = dt[0];
            fn_results_free(res, 1);
            fn_result_num(&res[0], v);
        }
    }
    return TSR_OK;
}
static const int SLOGDET = 1;

/* inv(a): stacked inverse via dgetrf + dgetri */
static int r_inv(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    const tsr_arg *arg = &args[0];
    if (arg->kind != 3) { fn_set_error("inv: input must be an array"); return TSR_ETYPE; }
    int64_t m, batch;
    int rc = square_batch(&arg->arr, &m, &batch);
    if (rc < 0) return rc;
    int64_t na;
    double *a = mat_f64(arg, "a", &na);
    if (!a) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, arg->arr.ndim, arg->arr.shape);
    if (!out) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(m > 0 ? m : 1));
    if (!ipiv) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int err = TSR_OK;
    for (int64_t i = 0; i < batch; i++) {
        double *mm = a + i * m * m;
        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)m, mm, (lapack_int)m, ipiv);
        if (info == 0) info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, (lapack_int)m, mm, (lapack_int)m, ipiv);
        if (info > 0) { fn_set_error("Singular matrix"); err = TSR_EARG; break; }
        if (info < 0) { fn_set_error("linalg inv: illegal argument %d", (int)info); err = TSR_EARG; break; }
        memcpy(out + i * m * m, mm, sizeof(double) * (size_t)(m * m));
    }
    free(ipiv);
    fn_free_doubles(a, na);
    return err;
}

/* cholesky(a, upper=False): lower (or upper) triangular factor L with A = L L^T, stacked (dpotrf) */
static int r_cholesky(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *arg = &args[0];
    if (arg->kind != 3) { fn_set_error("cholesky: input must be an array"); return TSR_ETYPE; }
    int upper = 0;
    if (nargs > 1 && args[1].kind == 4) upper = args[1].num != 0;
    else if (nargs > 1 && args[1].kind == 1) upper = args[1].num != 0 || args[1].ival != 0;
    int64_t m, batch;
    int rc = square_batch(&arg->arr, &m, &batch);
    if (rc < 0) return rc;
    int64_t na;
    double *a = mat_f64(arg, "a", &na);
    if (!a) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, arg->arr.ndim, arg->arr.shape);
    if (!out) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int err = TSR_OK;
    for (int64_t i = 0; i < batch; i++) {
        double *mm = a + i * m * m;
        lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, upper ? 'U' : 'L', (lapack_int)m, mm, (lapack_int)m);
        if (info > 0) { fn_set_error("Matrix is not positive definite"); err = TSR_EARG; break; }
        if (info < 0) { fn_set_error("linalg cholesky: illegal argument %d", (int)info); err = TSR_EARG; break; }
        double *o = out + i * m * m;                      /* keep the requested triangle, zero the other */
        for (int64_t r = 0; r < m; r++)
            for (int64_t c = 0; c < m; c++)
                o[r * m + c] = upper ? (c >= r ? mm[r * m + c] : 0.0) : (c <= r ? mm[r * m + c] : 0.0);
    }
    fn_free_doubles(a, na);
    return err;
}

/* solve(a, b): a is (..., M, M); b is (M,) or (..., M) (a vector per stack) or (..., M, K) (matrices). dgesv. */
static int r_solve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("solve: a and b must be arrays"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr, *B = &args[1].arr;
    int64_t m, batch;
    int rc = square_batch(A, &m, &batch);
    if (rc < 0) return rc;

    /* NumPy 2.0 semantics: b is a single vector iff it is 1-D (shape (M,), broadcast over a's stack); otherwise
       it is a stack of matrices (..., M, K). The old "b.ndim == a.ndim-1 is a stack of vectors" case was removed. */
    int shared_vec = 0, vector = 0;
    int64_t k = 1;
    int32_t rnd;
    int64_t rsh[TSR_MAXDIM];
    if (B->ndim == 1) {
        if (B->shape[0] != m) { fn_set_error("solve: b has wrong length"); return TSR_EARG; }
        shared_vec = 1; vector = 1;
        rnd = A->ndim - 1;
        for (int d = 0; d < rnd; d++) rsh[d] = A->shape[d];
    } else if (B->ndim == A->ndim) {
        if (B->shape[B->ndim - 2] != m) { fn_set_error("solve: b has wrong number of rows"); return TSR_EARG; }
        for (int d = 0; d < B->ndim - 2; d++)
            if (B->shape[d] != A->shape[d]) { fn_set_error("solve: a and b have mismatched batch dimensions"); return TSR_EARG; }
        k = B->shape[B->ndim - 1];
        rnd = A->ndim;
        for (int d = 0; d < rnd; d++) rsh[d] = A->shape[d];
        rsh[rnd - 1] = k;
    } else {
        fn_set_error("solve: b must have 1, a.ndim-1 or a.ndim dimensions");
        return TSR_EARG;
    }

    int64_t na, nb;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "b", &nb);
    if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *x = (double *)fn_result_array(&res[0], TSR_F64, rnd, rsh);
    if (!x) { fn_free_doubles(a, na); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(m > 0 ? m : 1));
    if (!ipiv) { fn_free_doubles(a, na); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    const int64_t rhs = vector ? 1 : k;
    int err = TSR_OK;
    for (int64_t i = 0; i < batch; i++) {
        double *ai = a + i * m * m;
        double *xi = x + i * m * rhs;
        const double *bi = shared_vec ? b : b + i * m * rhs;
        memcpy(xi, bi, sizeof(double) * (size_t)(m * rhs));   /* dgesv solves in place in the RHS/output buffer */
        lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)rhs, ai, (lapack_int)m,
                                        ipiv, xi, (lapack_int)rhs);
        if (info > 0) { fn_set_error("Singular matrix"); err = TSR_EARG; break; }
        if (info < 0) { fn_set_error("linalg solve: illegal argument %d", (int)info); err = TSR_EARG; break; }
    }
    free(ipiv);
    fn_free_doubles(a, na);
    fn_free_doubles(b, nb);
    return err;
}

/* singular values of one m x n matrix (row-major buffer, destroyed); s holds min(m,n) values. Computed with
   dgesdd job 'N' in COL-major on the same bytes, i.e. on A^T (n x m): the singular values are the same, and
   this avoids a LAPACKE row-major dgesdd('N') leading-dimension check ("wrong parameter 11"). */
static lapack_int svd_values(double *a, int64_t m, int64_t n, double *s)
{
    double dummy = 0.0;
    return LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'N', (lapack_int)n, (lapack_int)m, a, (lapack_int)n, s,
                          &dummy, 1, &dummy, 1);
}

/* svdvals(a): singular values (numpy.linalg.svdvals), 2-D */
static int r_svdvals(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("svdvals: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2) { fn_set_error("svdvals: input must be two-dimensional"); return TSR_EARG; }
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    const int64_t sh[1] = {k};
    double *s = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!s) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (k > 0) {
        lapack_int info = svd_values(a, m, n, s);
        if (info != 0) { fn_free_doubles(a, na); fn_set_error("svdvals: the algorithm did not converge"); return TSR_EARG; }
    }
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* matrix_rank(a, tol=None): number of singular values above tol (default max(s)*max(M,N)*eps) */
static int r_matrix_rank(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("matrix_rank: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2) { fn_set_error("matrix_rank: input must be two-dimensional"); return TSR_EARG; }
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    int have_tol = 0;
    double tol = 0.0;
    if (nargs > 1 && args[1].kind == 1) { have_tol = 1; tol = args[1].flags & 1 ? (double)args[1].ival : args[1].num; }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    int64_t rank = 0;
    if (k > 0) {
        double *s = (double *)malloc(sizeof(double) * (size_t)k);
        if (!s) { fn_free_doubles(a, na); return TSR_ENOMEM; }
        lapack_int info = svd_values(a, m, n, s);
        if (info != 0) { free(s); fn_free_doubles(a, na); fn_set_error("matrix_rank: the algorithm did not converge"); return TSR_EARG; }
        double smax = 0.0;
        for (int64_t i = 0; i < k; i++) if (s[i] > smax) smax = s[i];
        if (!have_tol) tol = smax * (double)(m > n ? m : n) * 2.220446049250313e-16;
        for (int64_t i = 0; i < k; i++) if (s[i] > tol) rank++;
        free(s);
    }
    fn_free_doubles(a, na);
    fn_result_int(&res[0], rank);
    return TSR_OK;
}

/* cond(a): 2-norm condition number, max(s)/min(s) (numpy.linalg.cond, p=None) */
static int r_cond(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("cond: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2) { fn_set_error("cond: input must be two-dimensional"); return TSR_EARG; }
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double c = FN_INF;
    if (k > 0) {
        double *s = (double *)malloc(sizeof(double) * (size_t)k);
        if (!s) { fn_free_doubles(a, na); return TSR_ENOMEM; }
        lapack_int info = svd_values(a, m, n, s);
        if (info != 0) { free(s); fn_free_doubles(a, na); fn_set_error("cond: the algorithm did not converge"); return TSR_EARG; }
        double smax = 0.0, smin = FN_INF;
        for (int64_t i = 0; i < k; i++) { if (s[i] > smax) smax = s[i]; if (s[i] < smin) smin = s[i]; }
        c = smin == 0.0 ? FN_INF : smax / smin;
        free(s);
    }
    fn_free_doubles(a, na);
    fn_result_num(&res[0], c);
    return TSR_OK;
}

/* pinv(a, rcond=1e-15): Moore-Penrose pseudo-inverse via SVD (unique; sign conventions cancel). 2-D. */
static int r_pinv(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("pinv: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2) { fn_set_error("pinv: input must be two-dimensional"); return TSR_EARG; }
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    double rcond = 1e-15;
    if (nargs > 1 && args[1].kind == 1) rcond = args[1].flags & 1 ? (double)args[1].ival : args[1].num;
    const int64_t sh[2] = {n, m};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    if (!out) return TSR_ENOMEM;
    if (k == 0) return TSR_OK;                          /* empty -> zeros(n, m) */
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *u = (double *)malloc(sizeof(double) * (size_t)(m * k));
    double *s = (double *)malloc(sizeof(double) * (size_t)k);
    double *vt = (double *)malloc(sizeof(double) * (size_t)(k * n));
    if (!u || !s || !vt) { free(u); free(s); free(vt); fn_free_doubles(a, na); return TSR_ENOMEM; }
    lapack_int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', (lapack_int)m, (lapack_int)n, a, (lapack_int)n,
                                     s, u, (lapack_int)k, vt, (lapack_int)n);
    if (info != 0) { free(u); free(s); free(vt); fn_free_doubles(a, na); fn_set_error("pinv: SVD did not converge"); return TSR_EARG; }
    double smax = 0.0;
    for (int64_t i = 0; i < k; i++) if (s[i] > smax) smax = s[i];
    const double cut = rcond * smax;
    /* pinv[i,j] = sum_l vt[l,i] * (1/s[l] if s[l] > cut else 0) * u[j,l]  ->  (n x m) */
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < m; j++) {
            double acc = 0.0;
            for (int64_t l = 0; l < k; l++)
                if (s[l] > cut) acc += vt[l * n + i] * (1.0 / s[l]) * u[j * k + l];
            out[i * m + j] = acc;
        }
    free(u); free(s); free(vt);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* eigvalsh(a, UPLO='L'): eigenvalues of a symmetric matrix, ascending, stacked (dsyevd, values only) */
static int r_eigvalsh(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("eigvalsh: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    int64_t nn, batch;
    int rc = square_batch(A, &nn, &batch);
    if (rc < 0) return rc;
    char uplo = 'L';
    if (nargs > 1 && args[1].kind == 2 && args[1].str && (args[1].str[0] == 'U' || args[1].str[0] == 'u')) uplo = 'U';
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    int64_t sh[TSR_MAXDIM];
    for (int d = 0; d < A->ndim - 1; d++) sh[d] = A->shape[d];
    double *w = (double *)fn_result_array(&res[0], TSR_F64, A->ndim - 1, sh);
    if (!w) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int err = TSR_OK;
    for (int64_t i = 0; i < batch && nn > 0; i++) {
        lapack_int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', uplo, (lapack_int)nn, a + i * nn * nn, (lapack_int)nn, w + i * nn);
        if (info != 0) { fn_set_error("eigvalsh: the eigenvalue algorithm did not converge"); err = TSR_EARG; break; }
    }
    fn_free_doubles(a, na);
    return err;
}

/* lstsq(a, b, rcond=None): least squares via dgelsd -> (x, residuals, rank, singular_values). 2-D a. */
static int r_lstsq(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("lstsq: a and b must be arrays"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr, *B = &args[1].arr;
    if (A->ndim != 2) { fn_set_error("lstsq: a must be two-dimensional"); return TSR_EARG; }
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    const int vector = B->ndim == 1;
    if (B->shape[0] != m) { fn_set_error("lstsq: b must have as many rows as a"); return TSR_EARG; }
    const int64_t nrhs = vector ? 1 : B->shape[1];
    double rcond = 2.220446049250313e-16 * (double)(m > n ? m : n);
    if (nargs > 2 && args[2].kind == 1) rcond = args[2].flags & 1 ? (double)args[2].ival : args[2].num;
    int64_t na, nb;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *bsrc = mat_f64(&args[1], "b", &nb);
    if (!bsrc) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    const int64_t rows = m > n ? m : n;                 /* dgelsd needs b sized max(m,n) x nrhs */
    double *bb = (double *)calloc((size_t)(rows * nrhs > 0 ? rows * nrhs : 1), sizeof(double));
    double *s = (double *)calloc((size_t)(k > 0 ? k : 1), sizeof(double));
    if (!bb || !s) { free(bb); free(s); fn_free_doubles(a, na); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < nrhs; j++) bb[i * nrhs + j] = bsrc[i * nrhs + j];
    lapack_int rank = 0, info = 0;
    if (k > 0 && nrhs > 0)
        info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, (lapack_int)nrhs, a, (lapack_int)n,
                              bb, (lapack_int)nrhs, s, rcond, &rank);
    if (info != 0) { free(bb); free(s); fn_free_doubles(a, na); fn_free_doubles(bsrc, nb); fn_set_error("lstsq: SVD did not converge"); return TSR_EARG; }
    /* results: x (n,) or (n, nrhs); residuals; rank; singular_values (k,) */
    tsr_result *seq = res;
    int64_t xsh1[1] = {n}, xsh2[2] = {n, nrhs};
    double *x = (double *)(vector ? fn_result_array(&seq[0], TSR_F64, 1, xsh1) : fn_result_array(&seq[0], TSR_F64, 2, xsh2));
    if (!x) { free(bb); free(s); fn_free_doubles(a, na); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < nrhs; j++) x[i * nrhs + j] = bb[i * nrhs + j];
    /* residuals: (nrhs,) sum of squares of the tail rows when rank == n and m > n, else empty */
    int64_t rres = (rank == (lapack_int)n && m > n) ? nrhs : 0;
    int64_t rsh[1] = {rres};
    double *resd = (double *)fn_result_array(&seq[1], TSR_F64, 1, rsh);
    if (!resd) { free(bb); free(s); fn_free_doubles(a, na); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t j = 0; j < rres; j++) {
        double acc = 0.0;
        for (int64_t i = n; i < m; i++) { const double v = bb[i * nrhs + j]; acc += v * v; }
        resd[j] = acc;
    }
    fn_result_int(&seq[2], k > 0 ? rank : 0);
    int64_t ssh[1] = {k};
    double *sv = (double *)fn_result_array(&seq[3], TSR_F64, 1, ssh);
    if (!sv) { free(bb); free(s); fn_free_doubles(a, na); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < k; i++) sv[i] = s[i];
    free(bb); free(s);
    fn_free_doubles(a, na);
    fn_free_doubles(bsrc, nb);
    return TSR_OK;
}

/* Canonicalise decomposition signs so factor results are comparable across LAPACK builds (they are unique only
   up to the sign of each singular/eigen vector). For each of the first `ncol` columns of u (row stride ustride,
   m rows), flip its sign so the largest-magnitude entry is non-negative (numpy/scikit-learn svd_flip); if v is
   given, flip the matching row of v (row length vn). Used for svd (u=U, v=Vh) and eigh (u=eigenvectors). */
static void sign_canon(double *u, int64_t m, int64_t ustride, int64_t ncol, double *v, int64_t vn)
{
    for (int64_t j = 0; j < ncol; j++) {
        int64_t am = 0;
        double best = -1.0;
        for (int64_t i = 0; i < m; i++) { const double a = fabs(u[i * ustride + j]); if (a > best) { best = a; am = i; } }
        if (u[am * ustride + j] < 0.0) {
            for (int64_t i = 0; i < m; i++) u[i * ustride + j] = -u[i * ustride + j];
            if (v) for (int64_t c = 0; c < vn; c++) v[j * vn + c] = -v[j * vn + c];
        }
    }
}

/* eigh(a, UPLO='L'): eigenvalues (ascending) and eigenvectors of a symmetric matrix, stacked (dsyevd 'V').
   Eigenvector columns are sign-canonicalised. Returns (eigenvalues, eigenvectors). */
static int r_eigh(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("eigh: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    int64_t n, batch;
    int rc = square_batch(A, &n, &batch);
    if (rc < 0) return rc;
    char uplo = 'L';
    if (nargs > 1 && args[1].kind == 2 && args[1].str && (args[1].str[0] == 'U' || args[1].str[0] == 'u')) uplo = 'U';
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);            /* becomes the eigenvectors */
    if (!a) return TSR_ENOMEM;
    int64_t wsh[TSR_MAXDIM], vsh[TSR_MAXDIM];
    for (int d = 0; d < A->ndim - 1; d++) wsh[d] = A->shape[d];
    for (int d = 0; d < A->ndim; d++) vsh[d] = A->shape[d];
    double *w = (double *)fn_result_array(&res[0], TSR_F64, A->ndim - 1, wsh);
    double *v = (double *)fn_result_array(&res[1], TSR_F64, A->ndim, vsh);
    if (!w || !v) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int err = TSR_OK;
    for (int64_t b = 0; b < batch && n > 0; b++) {
        double *ab = a + b * n * n;
        lapack_int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', uplo, (lapack_int)n, ab, (lapack_int)n, w + b * n);
        if (info != 0) { fn_set_error("eigh: the eigenvalue algorithm did not converge"); err = TSR_EARG; break; }
        sign_canon(ab, n, n, n, NULL, 0);
        memcpy(v + b * n * n, ab, sizeof(double) * (size_t)(n * n));
    }
    fn_free_doubles(a, na);
    return err;
}

/* pinvh(a): Moore-Penrose pseudo-inverse of a real symmetric matrix via its eigendecomposition
   (scipy.linalg.pinvh). pinvh = sum_k (1/w_k) v_k v_k^T over eigenvalues above SciPy's default cutoff
   (rtol = n*eps, cutoff = rtol*max|w|). The result is unique (sign of v_k cancels). 2-D. */
static int r_pinvh(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("pinvh: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2 || A->shape[0] != A->shape[1]) { fn_set_error("pinvh: input must be a square matrix"); return TSR_EARG; }
    const int64_t n = A->shape[0];
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);               /* overwritten with eigenvectors */
    if (!a) return TSR_ENOMEM;
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (n > 0) {
        double *w = (double *)malloc(sizeof(double) * (size_t)n);
        if (!w) { fn_free_doubles(a, na); return TSR_ENOMEM; }
        lapack_int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', (lapack_int)n, a, (lapack_int)n, w);
        if (info != 0) { free(w); fn_free_doubles(a, na); fn_set_error("pinvh: the eigenvalue algorithm did not converge"); return TSR_EARG; }
        double wmax = 0.0;
        for (int64_t k = 0; k < n; k++) { const double aw = fabs(w[k]); if (aw > wmax) wmax = aw; }
        const double cutoff = (double)n * DBL_EPSILON * wmax;
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) {
                double s = 0.0;
                for (int64_t k = 0; k < n; k++)
                    if (fabs(w[k]) > cutoff) s += (a[i * n + k] * a[j * n + k]) / w[k];   /* column k = k-th eigenvector */
                out[i * n + j] = s;
            }
        free(w);
    }
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* svd(a, full_matrices=True): A = U diag(s) Vh (dgesdd). U columns / Vh rows are sign-canonicalised together.
   Returns (U, S, Vh). 2-D. */
static int r_svd(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("svd: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2) { fn_set_error("svd: input must be two-dimensional"); return TSR_EARG; }
    int full = 1;
    if (nargs > 1 && args[1].kind == 4) full = args[1].num != 0;
    else if (nargs > 1 && args[1].kind == 1) full = args[1].num != 0 || args[1].ival != 0;
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    const int64_t uc = full ? m : k, vr = full ? n : k;
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    int64_t ush[2] = {m, uc}, ssh[1] = {k}, vsh[2] = {vr, n};
    double *u = (double *)fn_result_array(&res[0], TSR_F64, 2, ush);
    double *s = (double *)fn_result_array(&res[1], TSR_F64, 1, ssh);
    double *vt = (double *)fn_result_array(&res[2], TSR_F64, 2, vsh);
    if (!u || !s || !vt) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (k > 0) {
        lapack_int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, full ? 'A' : 'S', (lapack_int)m, (lapack_int)n, a, (lapack_int)n,
                                         s, u, (lapack_int)uc, vt, (lapack_int)n);
        if (info != 0) { fn_free_doubles(a, na); fn_set_error("svd: the algorithm did not converge"); return TSR_EARG; }
        sign_canon(u, m, uc, k, vt, n);                 /* flip the k singular directions (u col j <-> vt row j) */
    }
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* qr(a, mode='reduced'): A = Q R with R upper-triangular and a non-negative diagonal (unique). dgeqrf + dorgqr.
   modes 'reduced' (Q m x k, R k x n) and 'complete' (Q m x m, R m x n). Returns (Q, R). 2-D. */
static int r_qr(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("qr: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2) { fn_set_error("qr: input must be two-dimensional"); return TSR_EARG; }
    int complete = 0;
    if (nargs > 1 && args[1].kind == 2 && args[1].str) {
        const char *md = args[1].str;                    /* numpy: reduced/complete; scipy: economic/full */
        if (!strcmp(md, "complete") || !strcmp(md, "full")) complete = 1;
        else if (strcmp(md, "reduced") != 0 && strcmp(md, "economic") != 0) { fn_set_error("qr: mode must be 'reduced'/'complete' (numpy) or 'economic'/'full' (scipy)"); return TSR_EARG; }
    }
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    const int64_t qc = complete ? m : k, rr = complete ? m : k;
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);             /* m x n, holds R + reflectors after dgeqrf */
    if (!a) return TSR_ENOMEM;
    double *tau = (double *)malloc(sizeof(double) * (size_t)(k > 0 ? k : 1));
    if (!tau) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int64_t qsh[2] = {m, qc}, rsh[2] = {rr, n};
    double *q = (double *)fn_result_array(&res[0], TSR_F64, 2, qsh);
    double *r = (double *)fn_result_array(&res[1], TSR_F64, 2, rsh);
    if (!q || !r) { free(tau); fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (k > 0) {
        lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, a, (lapack_int)n, tau);
        if (info != 0) { free(tau); fn_free_doubles(a, na); fn_set_error("qr: dgeqrf failed"); return TSR_EARG; }
        /* R = upper triangle of the m x n factor, rows 0..rr-1 */
        for (int64_t i = 0; i < rr; i++)
            for (int64_t j = 0; j < n; j++) r[i * n + j] = j >= i ? a[i * n + j] : 0.0;
        /* Q: copy the reflector part into an m x qc buffer, then dorgqr */
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < qc; j++) q[i * qc + j] = j < n ? a[i * n + j] : 0.0;
        lapack_int info2 = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)qc, (lapack_int)k, q, (lapack_int)qc, tau);
        if (info2 != 0) { free(tau); fn_free_doubles(a, na); fn_set_error("qr: dorgqr failed"); return TSR_EARG; }
        /* canonicalise: make R's diagonal non-negative (flip Q column i and R row i where R[i,i] < 0) */
        for (int64_t i = 0; i < k && i < rr; i++)
            if (r[i * n + i] < 0.0) {
                for (int64_t j = 0; j < n; j++) r[i * n + j] = -r[i * n + j];
                for (int64_t rrow = 0; rrow < m; rrow++) q[rrow * qc + i] = -q[rrow * qc + i];
            }
    }
    free(tau);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* ---- general (non-symmetric) eigenproblem: eig / eigvals via dgeev ---- */
/* order eigenvalues by (real, imag) so the LAPACK-dependent order is canonical on every build */
static int cmp_eval(const void *pa, const void *pb)
{
    const double *a = (const double *)pa, *b = (const double *)pb;
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    if (a[1] < b[1]) return -1;
    if (a[1] > b[1]) return 1;
    return 0;
}

/* eigvals(a): eigenvalues of a general matrix (dgeev), sorted by (real, imag). float64 if all real, else
   complex128 (interleaved re, im). 2-D. */
static int r_eigvals(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nargs; (void)nres;
    const int force_complex = ctx && *(const int *)ctx;   /* scipy.linalg.eigvals is always complex */
    if (args[0].kind != 3) { fn_set_error("eigvals: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    int64_t n, batch;
    int rc = square_batch(A, &n, &batch);
    if (rc < 0) return rc;
    if (A->ndim != 2) { fn_set_error("eigvals: input must be two-dimensional"); return TSR_EARG; }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *wr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *wi = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *ev = (double *)malloc(sizeof(double) * (size_t)(2 * (n > 0 ? n : 1)));
    if (!wr || !wi || !ev) { free(wr); free(wi); free(ev); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int all_real = 1;
    if (n > 0) {
        double d = 0.0;
        lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)n, a, (lapack_int)n, wr, wi, &d, 1, &d, 1);
        if (info != 0) { free(wr); free(wi); free(ev); fn_free_doubles(a, na); fn_set_error("eigvals: the QR algorithm failed to converge"); return TSR_EARG; }
        for (int64_t i = 0; i < n; i++) { ev[2 * i] = wr[i]; ev[2 * i + 1] = wi[i]; if (wi[i] != 0.0) all_real = 0; }
        qsort(ev, (size_t)n, 2 * sizeof(double), cmp_eval);
    }
    if (force_complex) all_real = 0;
    int64_t sh[1] = {n};
    if (all_real) {
        double *o = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
        if (!o) { free(wr); free(wi); free(ev); fn_free_doubles(a, na); return TSR_ENOMEM; }
        for (int64_t i = 0; i < n; i++) o[i] = ev[2 * i];
    } else {
        double *o = (double *)fn_result_array(&res[0], TSR_C128, 1, sh);
        if (!o) { free(wr); free(wi); free(ev); fn_free_doubles(a, na); return TSR_ENOMEM; }
        for (int64_t i = 0; i < n; i++) { o[2 * i] = ev[2 * i]; o[2 * i + 1] = ev[2 * i + 1]; }
    }
    free(wr); free(wi); free(ev);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* eig(a): eigenvalues and right eigenvectors of a general matrix (dgeev). Eigenvalues sorted by (real, imag),
   eigenvectors reordered to match and phase-canonicalised (largest-magnitude component made real, positive).
   float64 when all eigenvalues are real, else complex128. 2-D. Returns (eigenvalues, eigenvectors). */
static int r_eig(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nargs; (void)nres;
    const int force_complex = ctx && *(const int *)ctx;   /* scipy.linalg.eig is always complex */
    if (args[0].kind != 3) { fn_set_error("eig: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    int64_t n, batch;
    int rc = square_batch(A, &n, &batch);
    if (rc < 0) return rc;
    if (A->ndim != 2) { fn_set_error("eig: input must be two-dimensional"); return TSR_EARG; }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *wr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *wi = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *vr = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    /* complex eigenvectors Vc[i*n+j] = (re, im) interleaved as [.. , 2]; and a permutation of columns */
    double *vc = (double *)malloc(sizeof(double) * (size_t)(2 * (n * n > 0 ? n * n : 1)));
    int64_t *ord = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    if (!wr || !wi || !vr || !vc || !ord) { free(wr); free(wi); free(vr); free(vc); free(ord); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int all_real = 1;
    if (n > 0) {
        double d = 0.0;
        lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V', (lapack_int)n, a, (lapack_int)n, wr, wi, &d, 1, vr, (lapack_int)n);
        if (info != 0) { free(wr); free(wi); free(vr); free(vc); free(ord); fn_free_doubles(a, na); fn_set_error("eig: the QR algorithm failed to converge"); return TSR_EARG; }
        for (int64_t i = 0; i < n; i++) if (wi[i] != 0.0) all_real = 0;
        if (force_complex) all_real = 0;
        /* assemble complex eigenvectors from dgeev's packed real storage (conjugate pairs) */
        for (int64_t j = 0; j < n; j++) {
            if (wi[j] == 0.0) {
                for (int64_t i = 0; i < n; i++) { vc[2 * (i * n + j)] = vr[i * n + j]; vc[2 * (i * n + j) + 1] = 0.0; }
            } else if (wi[j] > 0.0 && j + 1 < n) {
                for (int64_t i = 0; i < n; i++) {
                    const double re = vr[i * n + j], im = vr[i * n + j + 1];
                    vc[2 * (i * n + j)] = re;     vc[2 * (i * n + j) + 1] = im;       /* eigenvector j   = re + i*im */
                    vc[2 * (i * n + j + 1)] = re; vc[2 * (i * n + j + 1) + 1] = -im;  /* eigenvector j+1 = re - i*im */
                }
                j++;
            }
        }
        /* order columns by (wr, wi) */
        double *key = (double *)malloc(sizeof(double) * (size_t)(2 * n));
        for (int64_t i = 0; i < n; i++) { key[2 * i] = wr[i]; key[2 * i + 1] = wi[i]; ord[i] = i; }
        for (int64_t i = 1; i < n; i++) {                    /* insertion sort of ord by (wr, wi) */
            int64_t o = ord[i]; double kr = key[2 * o], ki = key[2 * o + 1]; int64_t p = i - 1;
            while (p >= 0 && (key[2 * ord[p]] > kr || (key[2 * ord[p]] == kr && key[2 * ord[p] + 1] > ki))) { ord[p + 1] = ord[p]; p--; }
            ord[p + 1] = o;
        }
        free(key);
    }
    int64_t wsh[1] = {n}, vsh[2] = {n, n};
    double *w = (double *)fn_result_array(&res[0], all_real ? TSR_F64 : TSR_C128, 1, wsh);
    double *v = (double *)fn_result_array(&res[1], all_real ? TSR_F64 : TSR_C128, 2, vsh);
    if (!w || !v) { free(wr); free(wi); free(vr); free(vc); free(ord); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t c = 0; c < n; c++) {
        const int64_t j = ord[c];
        if (all_real) w[c] = wr[j]; else { w[2 * c] = wr[j]; w[2 * c + 1] = wi[j]; }
        /* phase-canonicalise column j: unit factor making its largest-|.| entry real and positive */
        int64_t km = 0; double best = -1.0;
        for (int64_t i = 0; i < n; i++) { const double m2 = vc[2 * (i * n + j)] * vc[2 * (i * n + j)] + vc[2 * (i * n + j) + 1] * vc[2 * (i * n + j) + 1]; if (m2 > best) { best = m2; km = i; } }
        double fr = 1.0, fi = 0.0;
        const double mr = vc[2 * (km * n + j)], mi = vc[2 * (km * n + j) + 1], mag = sqrt(mr * mr + mi * mi);
        if (mag > 0) { fr = mr / mag; fi = -mi / mag; }      /* factor = conj(v[km]) / |v[km]| */
        for (int64_t i = 0; i < n; i++) {
            const double re = vc[2 * (i * n + j)], im = vc[2 * (i * n + j) + 1];
            const double nr = re * fr - im * fi, ni = re * fi + im * fr;
            if (all_real) v[i * n + c] = nr; else { v[2 * (i * n + c)] = nr; v[2 * (i * n + c) + 1] = ni; }
        }
    }
    free(wr); free(wi); free(vr); free(vc); free(ord);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* ---- small argument helpers (norm) ---- */
static const tsr_arg *la_arg(const tsr_arg *args, int nargs, int k)
{
    static const tsr_arg none = {0};
    return k < nargs ? &args[k] : &none;
}
static int la_int(const tsr_arg *a, int64_t *out)
{
    if (a->kind == 1 && (a->flags & 1)) { *out = a->ival; return TSR_OK; }
    if (a->kind == 1) { *out = (int64_t)a->num; return TSR_OK; }
    if (a->kind == 4) { *out = a->num != 0; return TSR_OK; }
    fn_set_error("axis must be an integer"); return TSR_EARG;
}

/* ---- norm: parse the ord argument ---- */
enum { ORD_NONE, ORD_NUM, ORD_FRO, ORD_NUC, ORD_PINF, ORD_NINF };
static int parse_ord(const tsr_arg *a, int *kind, double *p)
{
    *kind = ORD_NONE;
    *p = 0.0;
    if (a->kind == 0) return TSR_OK;
    if (a->kind == 1) { *kind = ORD_NUM; *p = a->flags & 1 ? (double)a->ival : a->num;
        if (*p == FN_INF) *kind = ORD_PINF; else if (*p == -FN_INF) *kind = ORD_NINF; return TSR_OK; }
    if (a->kind == 2 && a->str) {
        if (!strcmp(a->str, "fro")) { *kind = ORD_FRO; return TSR_OK; }
        if (!strcmp(a->str, "nuc")) { *kind = ORD_NUC; return TSR_OK; }
        if (!strcmp(a->str, "inf")) { *kind = ORD_PINF; return TSR_OK; }
        if (!strcmp(a->str, "-inf")) { *kind = ORD_NINF; return TSR_OK; }
    }
    fn_set_error("norm: invalid ord");
    return TSR_EARG;
}

/* norm(x, ord=None, axis=None, keepdims=False): vector and matrix norms (numpy.linalg.norm), axis None or int. */
static int r_norm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("norm: input must be an array"); return TSR_ETYPE; }
    const tsr_array *X = &args[0].arr;
    int okind, rc;
    double p;
    if ((rc = parse_ord(la_arg(args, nargs, 1), &okind, &p)) < 0) return rc;
    const tsr_arg *axarg = la_arg(args, nargs, 2);
    const int has_axis = axarg->kind != 0;
    int64_t na;
    double *x = fn_arg_doubles(&args[0], &na);
    const int nd = X->ndim;
    const int64_t total = tsr_shape_size(nd, X->shape);
    if (!x && total > 0) { fn_set_error("norm: out of memory"); return TSR_ENOMEM; }

    if (!has_axis) {
        double out;
        if (okind == ORD_NONE || (nd == 2 && okind == ORD_FRO) || nd == 1 || nd == 0) {
            if (nd <= 1 && okind == ORD_NUM) {           /* vector p-norm on a 1-D (or scalar) array */
                double acc = 0.0;
                if (p == 0.0) { int64_t c = 0; for (int64_t i = 0; i < total; i++) if (x[i] != 0.0) c++; out = (double)c; }
                else { for (int64_t i = 0; i < total; i++) acc += pow(fabs(x[i]), p); out = pow(acc, 1.0 / p); }
            } else if (nd <= 1 && okind == ORD_PINF) { out = 0.0; for (int64_t i = 0; i < total; i++) if (fabs(x[i]) > out) out = fabs(x[i]); }
            else if (nd <= 1 && okind == ORD_NINF) { out = FN_INF; for (int64_t i = 0; i < total; i++) if (fabs(x[i]) < out) out = fabs(x[i]); if (total == 0) out = 0.0; }
            else { double acc = 0.0; for (int64_t i = 0; i < total; i++) acc += x[i] * x[i]; out = sqrt(acc); }  /* frobenius / 2-norm */
        } else if (nd == 2) {
            const int64_t m = X->shape[0], n = X->shape[1];
            if (okind == ORD_NUC || (okind == ORD_NUM && (p == 2.0 || p == -2.0))) {
                const int64_t k = m < n ? m : n;
                double *s = (double *)malloc(sizeof(double) * (size_t)(k > 0 ? k : 1));
                double *xc = (double *)malloc(sizeof(double) * (size_t)(total > 0 ? total : 1));
                if (!s || !xc) { free(s); free(xc); fn_free_doubles(x, na); return TSR_ENOMEM; }
                memcpy(xc, x, sizeof(double) * (size_t)total);
                lapack_int info = k > 0 ? svd_values(xc, m, n, s) : 0;
                if (info != 0) { free(s); free(xc); fn_free_doubles(x, na); fn_set_error("norm: SVD did not converge"); return TSR_EARG; }
                if (okind == ORD_NUC) { out = 0.0; for (int64_t i = 0; i < k; i++) out += s[i]; }
                else if (p == 2.0) { out = 0.0; for (int64_t i = 0; i < k; i++) if (s[i] > out) out = s[i]; }
                else { out = FN_INF; for (int64_t i = 0; i < k; i++) if (s[i] < out) out = s[i]; if (k == 0) out = 0.0; }
                free(s); free(xc);
            } else if (okind == ORD_NUM && (p == 1.0 || p == -1.0)) {   /* column sums */
                out = p > 0 ? 0.0 : FN_INF;
                for (int64_t j = 0; j < n; j++) { double c = 0.0; for (int64_t i = 0; i < m; i++) c += fabs(x[i * n + j]); if (p > 0 ? c > out : c < out) out = c; }
            } else if (okind == ORD_PINF || okind == ORD_NINF) {         /* row sums */
                out = okind == ORD_PINF ? 0.0 : FN_INF;
                for (int64_t i = 0; i < m; i++) { double r = 0.0; for (int64_t j = 0; j < n; j++) r += fabs(x[i * n + j]); if (okind == ORD_PINF ? r > out : r < out) out = r; }
            } else { fn_free_doubles(x, na); fn_set_error("norm: invalid ord for a matrix"); return TSR_EARG; }
        } else { fn_free_doubles(x, na); fn_set_error("norm: improper number of dimensions to norm"); return TSR_EARG; }
        fn_free_doubles(x, na);
        fn_result_num(&res[0], out);
        return TSR_OK;
    }

    /* axis is an int: vector norm along that axis (keepdims optional) */
    int64_t axv;
    if ((rc = la_int(axarg, &axv)) < 0) { fn_free_doubles(x, na); return rc; }
    if (axv < -(int64_t)nd || axv >= nd) { fn_free_doubles(x, na); fn_set_error("axis %lld is out of bounds for array of dimension %d", (long long)axv, nd); return TSR_EARG; }
    const int ax = (int)(axv < 0 ? axv + nd : axv);
    int keep = 0;
    const tsr_arg *ka = la_arg(args, nargs, 3);
    if (ka->kind == 4) keep = ka->num != 0; else if (ka->kind == 1) keep = ka->ival != 0 || ka->num != 0;
    int64_t osh[TSR_MAXDIM];
    int32_t ond = 0;
    for (int d = 0; d < nd; d++) { if (d == ax) { if (keep) osh[ond++] = 1; } else osh[ond++] = X->shape[d]; }
    const int64_t alen = X->shape[ax];
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < ax; d++) outer *= X->shape[d];
    for (int d = ax + 1; d < nd; d++) inner *= X->shape[d];
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    if (!out) { fn_free_doubles(x, na); return TSR_ENOMEM; }
    for (int64_t o = 0; o < outer; o++)
        for (int64_t in = 0; in < inner; in++) {
            double mx = 0.0, mn = FN_INF; int64_t cnt = 0;
            for (int64_t t = 0; t < alen; t++) {
                const double v = fabs(x[(o * alen + t) * inner + in]);
                if (v > mx) mx = v;
                if (v < mn) mn = v;
                if (v != 0.0) cnt++;
            }
            double r;
            if (okind == ORD_PINF) r = mx;
            else if (okind == ORD_NINF) r = alen ? mn : 0.0;
            else if (okind == ORD_NUM && p == 0.0) r = (double)cnt;
            else if (okind == ORD_NUM && p == 1.0) { r = 0.0; for (int64_t t = 0; t < alen; t++) r += fabs(x[(o * alen + t) * inner + in]); }
            else if (okind == ORD_NUM && p != 2.0) { r = 0.0; for (int64_t t = 0; t < alen; t++) r += pow(fabs(x[(o * alen + t) * inner + in]), p); r = pow(r, 1.0 / p); }
            else { double a2 = 0.0; for (int64_t t = 0; t < alen; t++) { const double v = x[(o * alen + t) * inner + in]; a2 += v * v; } r = sqrt(a2); }
            out[o * inner + in] = r;
        }
    fn_free_doubles(x, na);
    return TSR_OK;
}

/* multi_dot(*arrays): the chained matrix product A0 @ A1 @ ... @ An-1 (numpy.linalg.multi_dot). Computed
   left-to-right (numpy picks the cheapest parenthesisation; the result is the same up to float rounding, within
   the linalg tolerance). The first array may be 1-D (a row) and the last 1-D (a column), as in NumPy. */
static int r_multi_dot(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items = args; int64_t k = nargs;
    if (nargs == 1 && args[0].kind == 5) { items = args[0].items; k = args[0].count; }
    if (k < 2) { fn_set_error("multi_dot: at least two arrays are required"); return TSR_EARG; }
    for (int64_t i = 0; i < k; i++) if (items[i].kind != 3) { fn_set_error("multi_dot: every argument must be an array"); return TSR_EARG; }
    const int sq_first = items[0].arr.ndim == 1;            /* 1-D first -> row vector */
    const int sq_last = items[k - 1].arr.ndim == 1;         /* 1-D last  -> column vector */
    int64_t na;
    double *a0 = fn_arg_doubles(&items[0], &na);            /* tsr-owned */
    if (!a0) return TSR_ENOMEM;
    int64_t ar = sq_first ? 1 : items[0].arr.shape[0];
    int64_t ac = sq_first ? items[0].arr.shape[0] : items[0].arr.shape[1];
    double *acc = (double *)malloc(sizeof(double) * (size_t)(ar * ac > 0 ? ar * ac : 1));   /* malloc-owned copy */
    if (!acc) { fn_free_doubles(a0, na); return TSR_ENOMEM; }
    memcpy(acc, a0, sizeof(double) * (size_t)(ar * ac));
    fn_free_doubles(a0, na);
    int rc = TSR_OK;
    for (int64_t i = 1; i < k && rc == TSR_OK; i++) {
        const int b1d = items[i].arr.ndim == 1;
        const int64_t br = items[i].arr.shape[0];
        const int64_t bc = b1d ? 1 : items[i].arr.shape[1];
        if (ac != br) { fn_set_error("multi_dot: shapes do not align at argument %lld", (long long)i); rc = TSR_EARG; break; }
        int64_t nb; double *b = fn_arg_doubles(&items[i], &nb);
        if (!b) { rc = TSR_ENOMEM; break; }
        double *np_ = (double *)calloc((size_t)(ar * bc > 0 ? ar * bc : 1), sizeof(double));
        if (!np_) { fn_free_doubles(b, nb); rc = TSR_ENOMEM; break; }
        for (int64_t r = 0; r < ar; r++)
            for (int64_t t = 0; t < ac; t++) {
                const double av = acc[r * ac + t];
                if (av != 0.0) { const double *brow = b + t * bc; double *orow = np_ + r * bc; for (int64_t c = 0; c < bc; c++) orow[c] += av * brow[c]; }
            }
        fn_free_doubles(b, nb);
        free(acc);
        acc = np_; ac = bc;                                 /* ar unchanged */
    }
    if (rc != TSR_OK) { free(acc); return rc; }
    if (sq_first && sq_last) { const double v = acc[0]; free(acc); fn_result_num(&res[0], v); return TSR_OK; }   /* scalar */
    int32_t ond; int64_t osh[2];
    if (sq_first) { ond = 1; osh[0] = ac; }
    else if (sq_last) { ond = 1; osh[0] = ar; }
    else { ond = 2; osh[0] = ar; osh[1] = ac; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    if (!out) { free(acc); return TSR_ENOMEM; }
    memcpy(out, acc, sizeof(double) * (size_t)(ar * ac));
    free(acc);
    return TSR_OK;
}

/* C = A(m x k) . B(k x n), row-major */
static void mm_mul(const double *A, const double *B, double *C, int64_t m, int64_t k, int64_t n)
{
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            double s = 0.0;
            for (int64_t l = 0; l < k; l++) s += A[i * k + l] * B[l * n + j];
            C[i * n + j] = s;
        }
}

/* matrix_power(a, n): a**n for a square matrix, by exponentiation-by-squaring like numpy.linalg.matrix_power
   (n<0 inverts first; n==0 is the identity). Single M x M matrix. */
static int r_matrix_power(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("matrix_power: input must be a square 2-D array"); return TSR_EARG; }
    if (nargs < 2 || args[1].kind != 1) { fn_set_error("matrix_power: n must be an integer"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0];
    long long nexp = (long long) args[1].num;
    int64_t na; double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    int64_t sh[2] = {m, m};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    if (!out) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (nexp == 0) { for (int64_t i = 0; i < m * m; i++) out[i] = 0.0; for (int64_t i = 0; i < m; i++) out[i * m + i] = 1.0; fn_free_doubles(a, na); return TSR_OK; }
    double *base = (double *)malloc((size_t)(m * m) * sizeof(double));
    if (!base) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    memcpy(base, a, (size_t)(m * m) * sizeof(double));
    fn_free_doubles(a, na);
    int rc = TSR_OK;
    if (nexp < 0) {                                          /* invert the base */
        lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(m > 0 ? m : 1));
        if (!ipiv) { free(base); return TSR_ENOMEM; }
        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)m, base, (lapack_int)m, ipiv);
        if (info == 0) info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, (lapack_int)m, base, (lapack_int)m, ipiv);
        free(ipiv);
        if (info != 0) { free(base); fn_set_error("matrix_power: singular matrix cannot be raised to a negative power"); return TSR_EARG; }
        nexp = -nexp;
    }
    double *z = (double *)malloc((size_t)(m * m) * sizeof(double));     /* running square */
    double *result = (double *)malloc((size_t)(m * m) * sizeof(double));
    double *tmp = (double *)malloc((size_t)(m * m) * sizeof(double));
    if (!z || !result || !tmp) { free(base); free(z); free(result); free(tmp); return TSR_ENOMEM; }
    int have_z = 0, have_res = 0;
    while (nexp > 0 && rc == TSR_OK) {
        if (!have_z) { memcpy(z, base, (size_t)(m * m) * sizeof(double)); have_z = 1; }
        else { mm_mul(z, z, tmp, m, m, m); memcpy(z, tmp, (size_t)(m * m) * sizeof(double)); }
        const int bit = (int)(nexp & 1); nexp >>= 1;
        if (bit) {
            if (!have_res) { memcpy(result, z, (size_t)(m * m) * sizeof(double)); have_res = 1; }
            else { mm_mul(result, z, tmp, m, m, m); memcpy(result, tmp, (size_t)(m * m) * sizeof(double)); }
        }
    }
    memcpy(out, result, (size_t)(m * m) * sizeof(double));
    free(base); free(z); free(result); free(tmp);
    return rc;
}

/* matrix_transpose(x): swap the last two axes of a stack of matrices (numpy.linalg.matrix_transpose). */
static int r_matrix_transpose(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim < 2) { fn_set_error("matrix_transpose: input must be at least 2-D"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    const int64_t r = args[0].arr.shape[nd - 2], c = args[0].arr.shape[nd - 1];
    int64_t batch = 1; for (int d = 0; d < nd - 2; d++) batch *= args[0].arr.shape[d];
    int64_t na; double *a = mat_f64(&args[0], "x", &na); if (!a) return TSR_ENOMEM;
    int64_t osh[TSR_MAXDIM];
    for (int d = 0; d < nd - 2; d++) osh[d] = args[0].arr.shape[d];
    osh[nd - 2] = c; osh[nd - 1] = r;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, osh);
    if (!out) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t b = 0; b < batch; b++) {
        const double *ab = a + b * r * c; double *ob = out + b * r * c;
        for (int64_t i = 0; i < r; i++) for (int64_t j = 0; j < c; j++) ob[j * r + i] = ab[i * c + j];
    }
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* tensorsolve(a, b): solve a x = b for a tensor a, reshaping to a square (N x N) system where N = b.size;
   the result has a's trailing dims a.shape[b.ndim:] (numpy.linalg.tensorsolve, axes=None). */
static int r_tensorsolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("tensorsolve: a and b must be arrays"); return TSR_EARG; }
    const tsr_array *A = &args[0].arr, *B = &args[1].arr;
    int64_t N = 1; for (int d = 0; d < B->ndim; d++) N *= B->shape[d];
    int64_t asz = 1; for (int d = 0; d < A->ndim; d++) asz *= A->shape[d];
    if (asz != N * N) { fn_set_error("tensorsolve: a is not square with respect to b"); return TSR_EARG; }
    int64_t na, nb; double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "b", &nb); if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int64_t osh[TSR_MAXDIM]; int ond = 0;                    /* result dims = a.shape[b.ndim:] */
    for (int d = B->ndim; d < A->ndim; d++) osh[ond++] = A->shape[d];
    if (ond == 0) { osh[0] = 1; ond = 1; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(N > 0 ? N : 1));
    int rc = TSR_OK;
    if (!out || !ipiv) rc = TSR_ENOMEM;
    else {
        lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, (lapack_int)N, 1, a, (lapack_int)N, ipiv, b, 1);
        if (info != 0) { fn_set_error("tensorsolve: singular system"); rc = TSR_EARG; }
        else for (int64_t i = 0; i < N; i++) out[i] = b[i];
    }
    free(ipiv); fn_free_doubles(a, na); fn_free_doubles(b, nb);
    return rc;
}

/* tensorinv(a, ind=2): the tensor inverse, reshaping a to a square (N x N) matrix where N = prod(a.shape[:ind]),
   inverting, and reshaping to a.shape[ind:] + a.shape[:ind] (numpy.linalg.tensorinv). */
static int r_tensorinv(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("tensorinv: a must be an array"); return TSR_EARG; }
    const tsr_array *A = &args[0].arr;
    int ind = (nargs > 1 && args[1].kind == 1) ? (int) args[1].num : 2;
    if (ind < 1 || ind >= A->ndim) { fn_set_error("tensorinv: invalid ind"); return TSR_EARG; }
    int64_t N = 1; for (int d = 0; d < ind; d++) N *= A->shape[d];
    int64_t rest = 1; for (int d = ind; d < A->ndim; d++) rest *= A->shape[d];
    if (N != rest) { fn_set_error("tensorinv: a is not square (prod(shape[:ind]) != prod(shape[ind:]))"); return TSR_EARG; }
    int64_t na; double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    int64_t osh[TSR_MAXDIM]; int ond = 0;                    /* a.shape[ind:] + a.shape[:ind] */
    for (int d = ind; d < A->ndim; d++) osh[ond++] = A->shape[d];
    for (int d = 0; d < ind; d++) osh[ond++] = A->shape[d];
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(N > 0 ? N : 1));
    int rc = TSR_OK;
    if (!out || !ipiv) rc = TSR_ENOMEM;
    else {
        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)N, (lapack_int)N, a, (lapack_int)N, ipiv);
        if (info == 0) info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, (lapack_int)N, a, (lapack_int)N, ipiv);
        if (info != 0) { fn_set_error("tensorinv: singular matrix"); rc = TSR_EARG; }
        else memcpy(out, a, sizeof(double) * (size_t)(N * N));
    }
    free(ipiv); fn_free_doubles(a, na);
    return rc;
}

/* linalg.trace(x, offset=0): sum of the (offset) diagonal of a 2-D matrix (numpy.linalg.trace; Array-API). */
static int r_la_trace(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("trace: input must be a 2-D array"); return TSR_EARG; }
    const int off = (nargs > 1 && args[1].kind == 1) ? (int) args[1].num : 0;
    const int64_t M = args[0].arr.shape[0], N = args[0].arr.shape[1];
    int64_t na; double *a = mat_f64(&args[0], "x", &na); if (!a) return TSR_ENOMEM;
    double s = 0.0;
    for (int64_t r = (off < 0 ? -off : 0); r < M; r++) { const int64_t c = r + off; if (c >= 0 && c < N) s += a[r * N + c]; }
    fn_free_doubles(a, na);
    fn_result_num(res, s);
    return TSR_OK;
}

/* linalg.diagonal(x, offset=0): the (offset) diagonal of a 2-D matrix as a 1-D array (numpy.linalg.diagonal). */
static int r_la_diagonal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("diagonal: input must be a 2-D array"); return TSR_EARG; }
    const int off = (nargs > 1 && args[1].kind == 1) ? (int) args[1].num : 0;
    const int64_t M = args[0].arr.shape[0], N = args[0].arr.shape[1];
    int64_t na; double *a = mat_f64(&args[0], "x", &na); if (!a) return TSR_ENOMEM;
    int64_t len = 0;
    for (int64_t r = (off < 0 ? -off : 0); r < M; r++) { const int64_t c = r + off; if (c >= 0 && c < N) len++; }
    int64_t sh[1] = {len};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!out) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int64_t k = 0;
    for (int64_t r = (off < 0 ? -off : 0); r < M; r++) { const int64_t c = r + off; if (c >= 0 && c < N) out[k++] = a[r * N + c]; }
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* linalg.outer(x1, x2): outer product of two 1-D arrays (numpy.linalg.outer). */
static int r_la_outer(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("outer: inputs must be 1-D arrays"); return TSR_EARG; }
    int64_t m, n; double *a = mat_f64(&args[0], "x1", &m); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "x2", &n); if (!b) { fn_free_doubles(a, m); return TSR_ENOMEM; }
    int64_t sh[2] = {m, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    if (!out) { fn_free_doubles(a, m); fn_free_doubles(b, n); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) out[i * n + j] = a[i] * b[j];
    fn_free_doubles(a, m); fn_free_doubles(b, n);
    return TSR_OK;
}

/* linalg.cross(x1, x2, axis=-1): cross product of 3-element vectors (numpy.linalg.cross; 1-D). */
static int r_la_cross(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[0].arr.shape[0] != 3 || args[1].kind != 3 || args[1].arr.ndim != 1 || args[1].arr.shape[0] != 3) { fn_set_error("cross: inputs must be 1-D arrays of length 3"); return TSR_EARG; }
    int64_t m, n; double *a = mat_f64(&args[0], "x1", &m); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "x2", &n); if (!b) { fn_free_doubles(a, m); return TSR_ENOMEM; }
    int64_t sh[1] = {3};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!out) { fn_free_doubles(a, m); fn_free_doubles(b, n); return TSR_ENOMEM; }
    out[0] = a[1] * b[2] - a[2] * b[1];
    out[1] = a[2] * b[0] - a[0] * b[2];
    out[2] = a[0] * b[1] - a[1] * b[0];
    fn_free_doubles(a, m); fn_free_doubles(b, n);
    return TSR_OK;
}

/* linalg.vecdot(x1, x2, axis=-1): dot product of two 1-D arrays (numpy.linalg.vecdot). */
static int r_la_vecdot(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("vecdot: inputs must be 1-D arrays"); return TSR_EARG; }
    int64_t m, n; double *a = mat_f64(&args[0], "x1", &m); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "x2", &n); if (!b) { fn_free_doubles(a, m); return TSR_ENOMEM; }
    int rc = TSR_OK; double s = 0.0;
    if (m != n) { fn_set_error("vecdot: inputs must have the same length"); rc = TSR_EARG; }
    else for (int64_t i = 0; i < m; i++) s += a[i] * b[i];
    fn_free_doubles(a, m); fn_free_doubles(b, n);
    if (rc < 0) return rc;
    fn_result_num(res, s);
    return TSR_OK;
}

/* linalg.vector_norm(x, axis=None, keepdims=False, ord=2): the vector norm over the flattened array
   (numpy.linalg.vector_norm; axis=None). */
static int r_vector_norm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("vector_norm: input must be an array"); return TSR_ETYPE; }
    const tsr_arg *axarg = la_arg(args, nargs, 1);
    if (axarg->kind != 0) { fn_set_error("vector_norm: only axis=None is supported"); return TSR_EARG; }
    int okind, rc; double p;
    if ((rc = parse_ord(la_arg(args, nargs, 3), &okind, &p)) < 0) return rc;
    int64_t na; double *x = fn_arg_doubles(&args[0], &na);
    const int64_t total = tsr_shape_size(args[0].arr.ndim, args[0].arr.shape);
    if (!x && total > 0) { fn_set_error("vector_norm: out of memory"); return TSR_ENOMEM; }
    double out;
    if (okind == ORD_PINF) { out = 0.0; for (int64_t i = 0; i < total; i++) if (fabs(x[i]) > out) out = fabs(x[i]); }
    else if (okind == ORD_NINF) { out = FN_INF; for (int64_t i = 0; i < total; i++) if (fabs(x[i]) < out) out = fabs(x[i]); if (total == 0) out = 0.0; }
    else if (okind == ORD_NUM && p == 0.0) { int64_t c = 0; for (int64_t i = 0; i < total; i++) if (x[i] != 0.0) c++; out = (double)c; }
    else if (okind == ORD_NUM && p != 2.0) { double acc = 0.0; for (int64_t i = 0; i < total; i++) acc += pow(fabs(x[i]), p); out = pow(acc, 1.0 / p); }
    else { double acc = 0.0; for (int64_t i = 0; i < total; i++) acc += x[i] * x[i]; out = sqrt(acc); }
    fn_free_doubles(x, na);
    fn_result_num(&res[0], out);
    return TSR_OK;
}

/* linalg.matrix_norm(x, keepdims=False, ord='fro'): a matrix norm over the last two axes of a 2-D array
   (numpy.linalg.matrix_norm). fro/1/-1/inf/-inf directly; 2/-2/nuc from the singular values. */
static int r_matrix_norm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("matrix_norm: input must be a 2-D array"); return TSR_EARG; }
    int okind, rc; double p;
    if ((rc = parse_ord(la_arg(args, nargs, 2), &okind, &p)) < 0) return rc;
    if (okind == ORD_NONE) okind = ORD_FRO;
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], total = m * n;
    int64_t na; double *x = fn_arg_doubles(&args[0], &na);
    if (!x && total > 0) { fn_set_error("matrix_norm: out of memory"); return TSR_ENOMEM; }
    double out;
    if (okind == ORD_FRO) { double acc = 0.0; for (int64_t i = 0; i < total; i++) acc += x[i] * x[i]; out = sqrt(acc); }
    else if (okind == ORD_NUC || (okind == ORD_NUM && (p == 2.0 || p == -2.0))) {
        const int64_t k = m < n ? m : n;
        double *s = (double *)malloc(sizeof(double) * (size_t)(k > 0 ? k : 1));
        double *xc = (double *)malloc(sizeof(double) * (size_t)(total > 0 ? total : 1));
        if (!s || !xc) { free(s); free(xc); fn_free_doubles(x, na); return TSR_ENOMEM; }
        memcpy(xc, x, sizeof(double) * (size_t)total);
        lapack_int info = k > 0 ? svd_values(xc, m, n, s) : 0;
        if (info != 0) { free(s); free(xc); fn_free_doubles(x, na); fn_set_error("matrix_norm: SVD did not converge"); return TSR_EARG; }
        if (okind == ORD_NUC) { out = 0.0; for (int64_t i = 0; i < k; i++) out += s[i]; }
        else if (p == 2.0) { out = 0.0; for (int64_t i = 0; i < k; i++) if (s[i] > out) out = s[i]; }
        else { out = FN_INF; for (int64_t i = 0; i < k; i++) if (s[i] < out) out = s[i]; if (k == 0) out = 0.0; }
        free(s); free(xc);
    } else if (okind == ORD_NUM && (p == 1.0 || p == -1.0)) {
        out = p > 0 ? 0.0 : FN_INF;
        for (int64_t j = 0; j < n; j++) { double c = 0.0; for (int64_t i = 0; i < m; i++) c += fabs(x[i * n + j]); if (p > 0 ? c > out : c < out) out = c; }
    } else if (okind == ORD_PINF || okind == ORD_NINF) {
        out = okind == ORD_PINF ? 0.0 : FN_INF;
        for (int64_t i = 0; i < m; i++) { double r = 0.0; for (int64_t j = 0; j < n; j++) r += fabs(x[i * n + j]); if (okind == ORD_PINF ? r > out : r < out) out = r; }
    } else { fn_free_doubles(x, na); fn_set_error("matrix_norm: invalid ord"); return TSR_EARG; }
    fn_free_doubles(x, na);
    fn_result_num(&res[0], out);
    return TSR_OK;
}

/* linalg.matmul(x1, x2): matrix product (numpy.linalg.matmul / np.matmul). 1-D operands are promoted as in
   numpy (1-D @ 1-D -> scalar; 1-D @ 2-D / 2-D @ 1-D -> vector); stacks with equal batch dims are batched. */
static int r_la_matmul(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("matmul: inputs must be arrays"); return TSR_EARG; }
    const tsr_array *A = &args[0].arr, *B = &args[1].arr;
    const int nd1 = A->ndim, nd2 = B->ndim;
    int64_t sza, szb; double *a = mat_f64(&args[0], "x1", &sza); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "x2", &szb); if (!b) { fn_free_doubles(a, sza); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (nd1 == 1 && nd2 == 1) {
        if (A->shape[0] != B->shape[0]) { rc = TSR_EARG; }
        else { double s = 0.0; for (int64_t i = 0; i < A->shape[0]; i++) s += a[i] * b[i]; fn_free_doubles(a, sza); fn_free_doubles(b, szb); fn_result_num(&res[0], s); return TSR_OK; }
    } else if (nd1 == 2 && nd2 == 2) {
        const int64_t m = A->shape[0], k = A->shape[1], n = B->shape[1];
        if (B->shape[0] != k) rc = TSR_EARG;
        else { double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){m, n}); if (!out) rc = TSR_ENOMEM; else mm_mul(a, b, out, m, k, n); }
    } else if (nd1 == 1 && nd2 == 2) {
        const int64_t k = A->shape[0], n = B->shape[1];
        if (B->shape[0] != k) rc = TSR_EARG;
        else { double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){n}); if (!out) rc = TSR_ENOMEM; else for (int64_t j = 0; j < n; j++) { double s = 0.0; for (int64_t i = 0; i < k; i++) s += a[i] * b[i * n + j]; out[j] = s; } }
    } else if (nd1 == 2 && nd2 == 1) {
        const int64_t m = A->shape[0], k = A->shape[1];
        if (B->shape[0] != k) rc = TSR_EARG;
        else { double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){m}); if (!out) rc = TSR_ENOMEM; else for (int64_t i = 0; i < m; i++) { double s = 0.0; for (int64_t l = 0; l < k; l++) s += a[i * k + l] * b[l]; out[i] = s; } }
    } else if (nd1 >= 3 && nd1 == nd2) {                     /* batched, equal leading dims */
        int64_t batch = 1; int ok = 1;
        for (int d = 0; d < nd1 - 2; d++) { if (A->shape[d] != B->shape[d]) ok = 0; batch *= A->shape[d]; }
        const int64_t m = A->shape[nd1 - 2], k = A->shape[nd1 - 1], n = B->shape[nd2 - 1];
        if (!ok || B->shape[nd2 - 2] != k) rc = TSR_EARG;
        else {
            int64_t osh[TSR_MAXDIM]; for (int d = 0; d < nd1 - 2; d++) osh[d] = A->shape[d]; osh[nd1 - 2] = m; osh[nd1 - 1] = n;
            double *out = (double *)fn_result_array(&res[0], TSR_F64, nd1, osh);
            if (!out) rc = TSR_ENOMEM; else for (int64_t bi = 0; bi < batch; bi++) mm_mul(a + bi * m * k, b + bi * k * n, out + bi * m * n, m, k, n);
        }
    } else rc = TSR_EARG;
    if (rc == TSR_EARG) fn_set_error("matmul: incompatible shapes");
    fn_free_doubles(a, sza); fn_free_doubles(b, szb);
    return rc;
}

/* matvec(x1, x2): matrix-vector product over the last two axes of x1 and the last axis of x2 (numpy.matvec).
   x1 is (..., M, N), x2 is (..., N); the result is (..., M). Leading (batch) dims must match. */
static int r_matvec(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("matvec: inputs must be arrays"); return TSR_EARG; }
    const tsr_array *A = &args[0].arr, *X = &args[1].arr;
    if (A->ndim < 2 || X->ndim < 1) { fn_set_error("matvec: x1 must be >= 2-D and x2 >= 1-D"); return TSR_EARG; }
    const int64_t M = A->shape[A->ndim - 2], N = A->shape[A->ndim - 1];
    if (X->shape[X->ndim - 1] != N) { fn_set_error("matvec: core dimensions do not match"); return TSR_EARG; }
    int64_t batchA = 1; for (int d = 0; d < A->ndim - 2; d++) batchA *= A->shape[d];
    int64_t batchX = 1; for (int d = 0; d < X->ndim - 1; d++) batchX *= X->shape[d];
    if (batchA != batchX) { fn_set_error("matvec: batch dimensions do not match"); return TSR_EARG; }
    int64_t sza, szx; double *a = mat_f64(&args[0], "x1", &sza); if (!a) return TSR_ENOMEM;
    double *x = mat_f64(&args[1], "x2", &szx); if (!x) { fn_free_doubles(a, sza); return TSR_ENOMEM; }
    int64_t osh[TSR_MAXDIM]; const int ond = A->ndim - 1;
    for (int d = 0; d < A->ndim - 2; d++) osh[d] = A->shape[d];
    osh[ond - 1] = M;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    if (!out) { fn_free_doubles(a, sza); fn_free_doubles(x, szx); return TSR_ENOMEM; }
    for (int64_t bi = 0; bi < batchA; bi++) {
        const double *am = a + bi * M * N, *xv = x + bi * N; double *ov = out + bi * M;
        for (int64_t i = 0; i < M; i++) { double s = 0.0; for (int64_t j = 0; j < N; j++) s += am[i * N + j] * xv[j]; ov[i] = s; }
    }
    fn_free_doubles(a, sza); fn_free_doubles(x, szx);
    return TSR_OK;
}

/* vecmat(x1, x2): vector-matrix product; x1 is (..., N), x2 is (..., N, M), the result is (..., M).
   For real inputs this is sum_n x1[n] * x2[n, m] (numpy.vecmat conjugates x1 for complex). */
static int r_vecmat(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("vecmat: inputs must be arrays"); return TSR_EARG; }
    const tsr_array *V = &args[0].arr, *B = &args[1].arr;
    if (V->ndim < 1 || B->ndim < 2) { fn_set_error("vecmat: x1 must be >= 1-D and x2 >= 2-D"); return TSR_EARG; }
    const int64_t N = B->shape[B->ndim - 2], M = B->shape[B->ndim - 1];
    if (V->shape[V->ndim - 1] != N) { fn_set_error("vecmat: core dimensions do not match"); return TSR_EARG; }
    int64_t batchV = 1; for (int d = 0; d < V->ndim - 1; d++) batchV *= V->shape[d];
    int64_t batchB = 1; for (int d = 0; d < B->ndim - 2; d++) batchB *= B->shape[d];
    if (batchV != batchB) { fn_set_error("vecmat: batch dimensions do not match"); return TSR_EARG; }
    int64_t szv, szb; double *v = mat_f64(&args[0], "x1", &szv); if (!v) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "x2", &szb); if (!b) { fn_free_doubles(v, szv); return TSR_ENOMEM; }
    int64_t osh[TSR_MAXDIM]; const int ond = B->ndim - 1;
    for (int d = 0; d < B->ndim - 2; d++) osh[d] = B->shape[d];
    osh[ond - 1] = M;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, ond, osh);
    if (!out) { fn_free_doubles(v, szv); fn_free_doubles(b, szb); return TSR_ENOMEM; }
    for (int64_t bi = 0; bi < batchB; bi++) {
        const double *vv = v + bi * N, *bm = b + bi * N * M; double *ov = out + bi * M;
        for (int64_t m = 0; m < M; m++) { double s = 0.0; for (int64_t n = 0; n < N; n++) s += vv[n] * bm[n * M + m]; ov[m] = s; }
    }
    fn_free_doubles(v, szv); fn_free_doubles(b, szb);
    return TSR_OK;
}

/* ================================================================ table */

static const fn_def DEFS[] = {
    ROUTINE("linalg.matmul", 1, "x1, x2", "out", r_la_matmul, NULL, "Matrix product of two arrays (numpy.linalg.matmul)."),
    ROUTINE("linalg.vector_norm", 1, "x, axis=None, keepdims=False, ord=2", "out", r_vector_norm, NULL, "Vector norm over the flattened array (numpy.linalg.vector_norm)."),
    ROUTINE("linalg.matrix_norm", 1, "x, keepdims=False, ord='fro'", "out", r_matrix_norm, NULL, "Matrix norm over the last two axes (numpy.linalg.matrix_norm)."),
    ROUTINE("linalg.trace", 1, "x, offset=0", "out", r_la_trace, NULL, "Sum along the diagonal of a matrix (numpy.linalg.trace)."),
    ROUTINE("linalg.diagonal", 1, "x, offset=0", "out", r_la_diagonal, NULL, "The specified diagonal of a matrix (numpy.linalg.diagonal)."),
    ROUTINE("linalg.outer", 1, "x1, x2", "out", r_la_outer, NULL, "Outer product of two 1-D arrays (numpy.linalg.outer)."),
    ROUTINE("linalg.cross", 1, "x1, x2, axis=-1", "out", r_la_cross, NULL, "Cross product of 3-element vectors (numpy.linalg.cross)."),
    ROUTINE("linalg.vecdot", 1, "x1, x2, axis=-1", "out", r_la_vecdot, NULL, "Dot product of two vectors (numpy.linalg.vecdot)."),
    ROUTINE("np.vecdot", 1, "x1, x2, axis=-1", "out", r_la_vecdot, NULL, "Vector dot product over the last axis (numpy.vecdot)."),
    ROUTINE("np.matvec", 1, "x1, x2", "out", r_matvec, NULL, "Matrix-vector product over the last two axes of x1 and the last axis of x2 (numpy.matvec)."),
    ROUTINE("np.vecmat", 1, "x1, x2", "out", r_vecmat, NULL, "Vector-matrix product; x1 is (..., N), x2 is (..., N, M) (numpy.vecmat)."),
    ROUTINE("linalg.tensorsolve", 1, "a, b, axes=None", "out", r_tensorsolve, NULL, "Solve the tensor equation a x = b (numpy.linalg.tensorsolve)."),
    ROUTINE("linalg.tensorinv", 1, "a, ind=2", "out", r_tensorinv, NULL, "Inverse of an N-dimensional array as a tensor (numpy.linalg.tensorinv)."),
    ROUTINE("linalg.matrix_power", 1, "a, n", "out", r_matrix_power, NULL, "Raise a square matrix to the (integer) power n (numpy.linalg.matrix_power)."),
    ROUTINE("linalg.matrix_transpose", 1, "x", "out", r_matrix_transpose, NULL, "Transpose the last two axes of a stack of matrices (numpy.linalg.matrix_transpose)."),
    ROUTINE("linalg.multi_dot", 1, "*arrays", "out", r_multi_dot, NULL, "Chained matrix product with optimal ordering (numpy.linalg.multi_dot)."),
    ROUTINE("linalg.svdvals", 1, "a", "out", r_svdvals, NULL, "Singular values of a matrix (numpy.linalg.svdvals)."),
    ROUTINE("linalg.matrix_rank", 1, "A, tol=None, hermitian=False", "out", r_matrix_rank, NULL, "Rank by counting singular values above a tolerance (numpy.linalg.matrix_rank)."),
    ROUTINE("linalg.cond", 1, "x, p=None", "out", r_cond, NULL, "2-norm condition number (numpy.linalg.cond)."),
    ROUTINE("linalg.pinv", 1, "a, rcond=1e-15", "out", r_pinv, NULL, "Moore-Penrose pseudo-inverse via SVD (numpy.linalg.pinv)."),
    ROUTINE("linalg.eigvalsh", 1, "a, UPLO='L'", "out", r_eigvalsh, NULL, "Eigenvalues of a symmetric matrix, ascending, stacked (numpy.linalg.eigvalsh)."),
    ROUTINE("linalg.lstsq", 4, "a, b, rcond=None", "x, residuals, rank, singular_values", r_lstsq, NULL, "Least-squares solution (numpy.linalg.lstsq)."),
    ROUTINE("linalg.norm", 1, "x, ord=None, axis=None, keepdims=False", "out", r_norm, NULL, "Vector or matrix norm (numpy.linalg.norm)."),
    ROUTINE("linalg.eigh", 2, "a, UPLO='L'", "eigenvalues, eigenvectors", r_eigh, NULL, "Eigenvalues (ascending) and eigenvectors of a symmetric matrix, stacked (numpy.linalg.eigh)."),
    ROUTINE("linalg.svd", 3, "a, full_matrices=True", "U, S, Vh", r_svd, NULL, "Singular value decomposition A = U diag(S) Vh (numpy.linalg.svd)."),
    ROUTINE("linalg.qr", 2, "a, mode='reduced'", "Q, R", r_qr, NULL, "QR factorisation with a non-negative R diagonal (numpy.linalg.qr)."),
    ROUTINE("linalg.eigvals", 1, "a", "out", r_eigvals, NULL, "Eigenvalues of a general matrix, sorted (numpy.linalg.eigvals)."),
    ROUTINE("linalg.eig", 2, "a", "eigenvalues, eigenvectors", r_eig, NULL, "Eigenvalues and right eigenvectors of a general matrix (numpy.linalg.eig)."),
    ROUTINE("linalg.solve", 1, "a, b", "out", r_solve, NULL, "Solve a linear system a x = b, stacked (numpy.linalg.solve)."),
    ROUTINE("linalg.inv", 1, "a", "out", r_inv, NULL, "Multiplicative inverse of a square matrix, stacked (numpy.linalg.inv)."),
    ROUTINE("linalg.det", 1, "a", "out", r_det, NULL, "Determinant of a square matrix, stacked (numpy.linalg.det)."),
    ROUTINE("linalg.slogdet", 2, "a", "sign, logabsdet", r_det, &SLOGDET, "Sign and natural log of the absolute determinant, stacked (numpy.linalg.slogdet)."),
    ROUTINE("linalg.cholesky", 1, "a, upper=False", "out", r_cholesky, NULL, "Cholesky factor (lower by default), stacked (numpy.linalg.cholesky)."),
};

const fn_table TSR_NP_LINALG_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};

/* scipy.linalg.cholesky(a, lower=False): UPPER factor by default (numpy.linalg.cholesky is lower). dpotrf. 2-D. */
static int r_scipy_cholesky(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("cholesky: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2 || A->shape[0] != A->shape[1]) { fn_set_error("cholesky: input must be a square matrix"); return TSR_EARG; }
    int lower = 0;
    if (nargs > 1 && args[1].kind == 4) lower = args[1].num != 0;
    else if (nargs > 1 && args[1].kind == 1) lower = args[1].num != 0 || args[1].ival != 0;
    const int64_t n = A->shape[0];
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, A->shape);
    if (!out) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (n > 0) {
        lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, lower ? 'L' : 'U', (lapack_int)n, a, (lapack_int)n);
        if (info > 0) { fn_free_doubles(a, na); fn_set_error("Matrix is not positive definite"); return TSR_EARG; }
        if (info < 0) { fn_free_doubles(a, na); fn_set_error("cholesky: illegal argument %d", (int)info); return TSR_EARG; }
        for (int64_t r = 0; r < n; r++)
            for (int64_t c = 0; c < n; c++)
                out[r * n + c] = lower ? (c <= r ? a[r * n + c] : 0.0) : (c >= r ? a[r * n + c] : 0.0);
    }
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* scipy.linalg.cho_factor(a, lower=False): the Cholesky factor in the chosen triangle, the OTHER triangle left
   as the input (as SciPy/LAPACK do, not zeroed). Returns (c, lower). dpotrf, 2-D. */
static int r_cho_factor(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("cho_factor: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2 || A->shape[0] != A->shape[1]) { fn_set_error("cho_factor: input must be a square matrix"); return TSR_EARG; }
    int lower = 0;
    if (nargs > 1 && (args[1].kind == 4 || args[1].kind == 1)) lower = args[1].num != 0 || args[1].ival != 0;
    const int64_t n = A->shape[0];
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *c = (double *)fn_result_array(&res[0], TSR_F64, 2, A->shape);
    if (!c) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (n > 0) {
        memcpy(c, a, sizeof(double) * (size_t)(n * n));
        lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, lower ? 'L' : 'U', (lapack_int)n, c, (lapack_int)n);
        if (info > 0) { fn_free_doubles(a, na); fn_set_error("Matrix is not positive definite"); return TSR_EARG; }
        if (info < 0) { fn_free_doubles(a, na); fn_set_error("cho_factor: illegal argument %d", (int)info); return TSR_EARG; }
        for (int64_t r = 0; r < n; r++)                       /* restore the non-factor triangle to the input */
            for (int64_t cc = 0; cc < n; cc++)
                if (lower ? cc > r : cc < r) c[r * n + cc] = a[r * n + cc];
    }
    fn_free_doubles(a, na);
    res[1].kind = 4; res[1].num = lower ? 1.0 : 0.0;          /* the lower flag, echoed back */
    return TSR_OK;
}

/* scipy.linalg.cho_solve((c, lower), b): solve a x = b from a Cholesky factor. Tessero takes c, lower, b
   unpacked rather than the (c, lower) tuple. dpotrs, 2-D c. */
static int r_cho_solve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[2].kind != 3) { fn_set_error("cho_solve: c and b must be arrays"); return TSR_ETYPE; }
    const tsr_array *C = &args[0].arr, *B = &args[2].arr;
    if (C->ndim != 2 || C->shape[0] != C->shape[1]) { fn_set_error("cho_solve: c must be a square matrix"); return TSR_EARG; }
    const int64_t n = C->shape[0];
    int lower = 0;
    if (args[1].kind == 4 || args[1].kind == 1) lower = args[1].num != 0 || args[1].ival != 0;
    const int vector = B->ndim == 1;
    if (B->shape[0] != n) { fn_set_error("cho_solve: b has the wrong number of rows"); return TSR_EARG; }
    const int64_t nrhs = vector ? 1 : B->shape[1];
    int64_t nc, nb;
    double *c = mat_f64(&args[0], "c", &nc);
    if (!c) return TSR_ENOMEM;
    double *b = mat_f64(&args[2], "b", &nb);
    if (!b) { fn_free_doubles(c, nc); return TSR_ENOMEM; }
    int64_t rsh1[1] = {n}, rsh2[2] = {n, nrhs};
    double *x = (double *)(vector ? fn_result_array(&res[0], TSR_F64, 1, rsh1) : fn_result_array(&res[0], TSR_F64, 2, rsh2));
    if (!x) { fn_free_doubles(c, nc); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    memcpy(x, b, sizeof(double) * (size_t)(n * nrhs));
    lapack_int info = n > 0 ? LAPACKE_dpotrs(LAPACK_ROW_MAJOR, lower ? 'L' : 'U', (lapack_int)n, (lapack_int)nrhs, c, (lapack_int)n, x, (lapack_int)nrhs) : 0;
    fn_free_doubles(c, nc); fn_free_doubles(b, nb);
    if (info < 0) { fn_set_error("cho_solve: illegal argument %d", (int)info); return TSR_EARG; }
    return TSR_OK;
}

/* scipy.linalg.solve_triangular(a, b, trans=0, lower=False, unit_diagonal=False): dtrtrs. 2-D a. */
static int r_solve_triangular(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("solve_triangular: a and b must be arrays"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr, *B = &args[1].arr;
    if (A->ndim != 2 || A->shape[0] != A->shape[1]) { fn_set_error("solve_triangular: a must be a square matrix"); return TSR_EARG; }
    const int64_t n = A->shape[0];
    const int vector = B->ndim == 1;
    if (B->shape[0] != n) { fn_set_error("solve_triangular: b has the wrong number of rows"); return TSR_EARG; }
    const int64_t nrhs = vector ? 1 : B->shape[1];
    int64_t trans = 0, lower = 0, unit = 0;
    if (nargs > 2 && args[2].kind == 1) trans = args[2].ival ? args[2].ival : (int64_t)args[2].num;
    if (nargs > 3 && (args[3].kind == 4 || args[3].kind == 1)) lower = args[3].num != 0 || args[3].ival != 0;
    if (nargs > 4 && (args[4].kind == 4 || args[4].kind == 1)) unit = args[4].num != 0 || args[4].ival != 0;
    const char tc = trans == 1 ? 'T' : (trans == 2 ? 'C' : 'N');
    int64_t na, nb;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "b", &nb);
    if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    int64_t rsh1[1] = {n}, rsh2[2] = {n, nrhs};
    double *x = (double *)(vector ? fn_result_array(&res[0], TSR_F64, 1, rsh1) : fn_result_array(&res[0], TSR_F64, 2, rsh2));
    if (!x) { fn_free_doubles(a, na); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    memcpy(x, b, sizeof(double) * (size_t)(n * nrhs));
    lapack_int info = n > 0 ? LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, lower ? 'L' : 'U', tc, unit ? 'U' : 'N',
                                             (lapack_int)n, (lapack_int)nrhs, a, (lapack_int)n, x, (lapack_int)nrhs) : 0;
    fn_free_doubles(a, na);
    fn_free_doubles(b, nb);
    if (info > 0) { fn_set_error("Singular matrix"); return TSR_EARG; }
    if (info < 0) { fn_set_error("solve_triangular: illegal argument %d", (int)info); return TSR_EARG; }
    return TSR_OK;
}

/* scipy.linalg.lu(a): P, L, U with A = P @ L @ U (dgetrf; unique with partial pivoting). 2-D. */
static int r_lu(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("lu: input must be an array"); return TSR_ETYPE; }
    const tsr_array *A = &args[0].arr;
    if (A->ndim != 2) { fn_set_error("lu: input must be two-dimensional"); return TSR_EARG; }
    const int64_t m = A->shape[0], n = A->shape[1], k = m < n ? m : n;
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(k > 0 ? k : 1));
    int64_t *perm = (int64_t *)malloc(sizeof(int64_t) * (size_t)(m > 0 ? m : 1));
    if (!ipiv || !perm) { free(ipiv); free(perm); fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (k > 0) {
        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, a, (lapack_int)n, ipiv);
        if (info < 0) { free(ipiv); free(perm); fn_free_doubles(a, na); fn_set_error("lu: illegal argument %d", (int)info); return TSR_EARG; }
    }
    for (int64_t i = 0; i < m; i++) perm[i] = i;
    for (int64_t i = 0; i < k; i++) { const int64_t j = ipiv[i] - 1; const int64_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
    int64_t psh[2] = {m, m}, lsh[2] = {m, k}, ush[2] = {k, n};
    double *p = (double *)fn_result_array(&res[0], TSR_F64, 2, psh);
    double *l = (double *)fn_result_array(&res[1], TSR_F64, 2, lsh);
    double *u = (double *)fn_result_array(&res[2], TSR_F64, 2, ush);
    if (!p || !l || !u) { free(ipiv); free(perm); fn_free_doubles(a, na); return TSR_ENOMEM; }
    memset(p, 0, sizeof(double) * (size_t)(m * m));                         /* fn_result_array is not zeroed */
    for (int64_t row = 0; row < m; row++) p[perm[row] * m + row] = 1.0;     /* P: A = P L U */
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < k; j++) l[i * k + j] = i > j ? a[i * n + j] : (i == j ? 1.0 : 0.0);
    for (int64_t i = 0; i < k; i++)
        for (int64_t j = 0; j < n; j++) u[i * n + j] = j >= i ? a[i * n + j] : 0.0;
    free(ipiv); free(perm);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* toeplitz(c, r=None): the Toeplitz matrix with first column c and first row r (r defaults to c; r[0] is
   ignored, the diagonal is c[0]). T[i][j] = i>=j ? c[i-j] : r[j-i]. */
static int r_toeplitz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("toeplitz: c must be an array"); return TSR_ETYPE; }
    const int64_t m = args[0].arr.ndim >= 1 ? args[0].arr.shape[0] : 0;
    const int have_r = nargs > 1 && args[1].kind == 3 && args[1].arr.ndim >= 1;
    const int64_t n = have_r ? args[1].arr.shape[0] : m;
    int64_t nc, nr = 0;
    double *c = mat_f64(&args[0], "c", &nc);
    if (!c) return TSR_ENOMEM;
    double *r = have_r ? mat_f64(&args[1], "r", &nr) : c;
    if (have_r && !r) { fn_free_doubles(c, nc); return TSR_ENOMEM; }
    int64_t osh[2] = {m, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(c, nc); if (have_r) fn_free_doubles(r, nr); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++)
            out[i * n + j] = (i >= j) ? c[i - j] : r[j - i];
    fn_free_doubles(c, nc);
    if (have_r) fn_free_doubles(r, nr);
    return TSR_OK;
}

/* block_diag(*arrays): assemble 2-D blocks on the diagonal of a larger zero matrix. */
static int r_block_diag(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items = args; int64_t k = nargs;
    if (nargs == 1 && args[0].kind == 5) { items = args[0].items; k = args[0].count; }
    int64_t M = 0, N = 0;
    for (int64_t i = 0; i < k; i++) {
        if (items[i].kind != 3 || items[i].arr.ndim != 2) { fn_set_error("block_diag: every block must be a 2-D array"); return TSR_EARG; }
        M += items[i].arr.shape[0]; N += items[i].arr.shape[1];
    }
    int64_t osh[2] = {M, N};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) return TSR_ENOMEM;
    memset(out, 0, sizeof(double) * (size_t)(M * N));
    int64_t roff = 0, coff = 0;
    for (int64_t b = 0; b < k; b++) {
        const int64_t rb = items[b].arr.shape[0], cb = items[b].arr.shape[1];
        int64_t nb; double *blk = mat_f64(&items[b], "block", &nb);
        if (!blk) return TSR_ENOMEM;
        for (int64_t i = 0; i < rb; i++)
            for (int64_t j = 0; j < cb; j++)
                out[(roff + i) * N + (coff + j)] = blk[i * cb + j];
        fn_free_doubles(blk, nb);
        roff += rb; coff += cb;
    }
    return TSR_OK;
}

/* a small integer scalar argument (n, M, N) for the matrix builders below. */
static int sl_int(const tsr_arg *a, const char *name, int64_t *out)
{
    if (a->kind == 1 && (a->flags & 1)) { *out = a->ival; return TSR_OK; }
    fn_set_error("%s must be an integer", name);
    return TSR_EARG;
}

/* circulant(c): the circulant matrix whose first column is c; C[i][j] = c[(i-j) mod n] (scipy.linalg.circulant). */
static int r_circulant(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("circulant: c must be a 1-D array"); return TSR_ETYPE; }
    int64_t n; double *c = mat_f64(&args[0], "c", &n);
    if (!c) return TSR_ENOMEM;
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(c, n); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            int64_t k = (i - j) % n; if (k < 0) k += n;
            out[i * n + j] = c[k];
        }
    fn_free_doubles(c, n);
    return TSR_OK;
}

/* companion(a): companion matrix of the polynomial with coefficients a (a[0] != 0); the first row is
   -a[1:]/a[0] and the subdiagonal is 1 (scipy.linalg.companion). */
static int r_companion(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("companion: a must be a 1-D array"); return TSR_ETYPE; }
    int64_t n; double *a = mat_f64(&args[0], "a", &n);
    if (!a) return TSR_ENOMEM;
    if (n < 2) { fn_free_doubles(a, n); fn_set_error("The length of `a` must be at least 2."); return TSR_EARG; }
    if (a[0] == 0.0) { fn_free_doubles(a, n); fn_set_error("The first coefficient in `a` must not be zero."); return TSR_EARG; }
    int64_t m = n - 1, osh[2] = {m, m};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(a, n); return TSR_ENOMEM; }
    memset(out, 0, sizeof(double) * (size_t)(m * m));
    for (int64_t j = 0; j < m; j++) out[j] = -a[j + 1] / a[0];
    for (int64_t i = 1; i < m; i++) out[i * m + (i - 1)] = 1.0;
    fn_free_doubles(a, n);
    return TSR_OK;
}

/* hadamard(n): the n x n Sylvester-Hadamard matrix, n a power of two; H[i][j] = (-1)^popcount(i & j)
   (scipy.linalg.hadamard). */
static int r_hadamard(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    int64_t n; int rc = sl_int(&args[0], "n", &n);
    if (rc < 0) return rc;
    if (n < 1 || (n & (n - 1)) != 0) { fn_set_error("n must be a positive integer, and n must be a power of 2"); return TSR_EARG; }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            out[i * n + j] = (__builtin_popcountll((uint64_t)(i & j)) & 1) ? -1.0 : 1.0;
    return TSR_OK;
}

/* hilbert(n): the n x n Hilbert matrix, H[i][j] = 1/(i+j+1) (scipy.linalg.hilbert). */
static int r_hilbert(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    int64_t n; int rc = sl_int(&args[0], "n", &n);
    if (rc < 0) return rc;
    if (n < 0) { fn_set_error("n must be non-negative"); return TSR_EARG; }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            out[i * n + j] = 1.0 / (double)(i + j + 1);
    return TSR_OK;
}

/* hankel(c, r=None): Hankel matrix with first column c and last row r; H[i][j] = c[i+j] while i+j < len(c),
   else r[i+j-len(c)+1]. r defaults to zeros, so the lower-right triangle is zero (scipy.linalg.hankel). */
static int r_hankel(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim < 1) { fn_set_error("hankel: c must be an array"); return TSR_ETYPE; }
    const int have_r = nargs > 1 && args[1].kind == 3 && args[1].arr.ndim >= 1;
    int64_t m, nr = 0;
    double *c = mat_f64(&args[0], "c", &m);
    if (!c) return TSR_ENOMEM;
    double *r = have_r ? mat_f64(&args[1], "r", &nr) : NULL;
    if (have_r && !r) { fn_free_doubles(c, m); return TSR_ENOMEM; }
    const int64_t n = have_r ? nr : m;
    int64_t osh[2] = {m, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(c, m); if (r) fn_free_doubles(r, nr); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            const int64_t k = i + j;
            out[i * n + j] = (k < m) ? c[k] : (r ? r[k - m + 1] : 0.0);
        }
    fn_free_doubles(c, m);
    if (r) fn_free_doubles(r, nr);
    return TSR_OK;
}

/* fiedler(a): the symmetric Fiedler matrix F[i][j] = |a[i] - a[j]| (scipy.linalg.fiedler). */
static int r_fiedler(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("fiedler: a must be a 1-D array"); return TSR_ETYPE; }
    int64_t n; double *a = mat_f64(&args[0], "a", &n);
    if (!a) return TSR_ENOMEM;
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(a, n); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            out[i * n + j] = fabs(a[i] - a[j]);
    fn_free_doubles(a, n);
    return TSR_OK;
}

/* leslie(f, s): the Leslie population matrix with first row f (length n) and sub-diagonal s (length n-1)
   (scipy.linalg.leslie). */
static int r_leslie(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("leslie: f must be a 1-D array"); return TSR_ETYPE; }
    if (nargs < 2 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("leslie: s must be a 1-D array"); return TSR_ETYPE; }
    int64_t nf, ns;
    double *f = mat_f64(&args[0], "f", &nf);
    if (!f) return TSR_ENOMEM;
    double *s = mat_f64(&args[1], "s", &ns);
    if (!s) { fn_free_doubles(f, nf); return TSR_ENOMEM; }
    if (nf < 2) { fn_free_doubles(f, nf); fn_free_doubles(s, ns); fn_set_error("The length of f must be at least 2."); return TSR_EARG; }
    if (ns != nf - 1) { fn_free_doubles(f, nf); fn_free_doubles(s, ns); fn_set_error("The length of s must be one less than the length of f."); return TSR_EARG; }
    int64_t osh[2] = {nf, nf};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(f, nf); fn_free_doubles(s, ns); return TSR_ENOMEM; }
    memset(out, 0, sizeof(double) * (size_t)(nf * nf));
    for (int64_t j = 0; j < nf; j++) out[j] = f[j];
    for (int64_t i = 1; i < nf; i++) out[i * nf + (i - 1)] = s[i - 1];
    fn_free_doubles(f, nf);
    fn_free_doubles(s, ns);
    return TSR_OK;
}

/* C(n, k) as a double, rounded to the exact integer (these matrix builders use small orders). */
static double sl_binom(int64_t n, int64_t k)
{
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double r = 1.0;
    for (int64_t i = 0; i < k; i++) r = r * (double)(n - i) / (double)(i + 1);
    return floor(r + 0.5);
}

/* pascal(n, kind='symmetric'): the n x n Pascal matrix. symmetric S[i][j] = C(i+j, i); lower L[i][j] = C(i, j)
   (0 for j > i); upper U[i][j] = C(j, i) (scipy.linalg.pascal). */
static int r_pascal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t n; int rc = sl_int(&args[0], "n", &n);
    if (rc < 0) return rc;
    if (n < 0) { fn_set_error("pascal: n must be a non-negative integer"); return TSR_EARG; }
    int kind = 0;  /* 0 symmetric, 1 lower, 2 upper */
    if (nargs > 1 && args[1].kind == 2 && args[1].str) {
        if (!strcmp(args[1].str, "lower")) kind = 1;
        else if (!strcmp(args[1].str, "upper")) kind = 2;
        else if (strcmp(args[1].str, "symmetric") != 0) { fn_set_error("pascal: kind must be 'symmetric', 'lower' or 'upper'"); return TSR_EARG; }
    }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++)
            out[i * n + j] = kind == 0 ? sl_binom(i + j, i) : kind == 1 ? sl_binom(i, j) : sl_binom(j, i);
    return TSR_OK;
}

/* invpascal(n, kind='symmetric'): inverse of the Pascal matrix. lower: (-1)^(i+j) C(i, j); upper its
   transpose; symmetric: (-1)^(i+j) * sum_{k>=max(i,j)} C(k, i) C(k, j) (scipy.linalg.invpascal). */
static int r_invpascal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t n; int rc = sl_int(&args[0], "n", &n);
    if (rc < 0) return rc;
    if (n < 0) { fn_set_error("invpascal: n must be a non-negative integer"); return TSR_EARG; }
    int kind = 0;
    if (nargs > 1 && args[1].kind == 2 && args[1].str) {
        if (!strcmp(args[1].str, "lower")) kind = 1;
        else if (!strcmp(args[1].str, "upper")) kind = 2;
        else if (strcmp(args[1].str, "symmetric") != 0) { fn_set_error("invpascal: kind must be 'symmetric', 'lower' or 'upper'"); return TSR_EARG; }
    }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            const double sign = ((i + j) & 1) ? -1.0 : 1.0;
            double v;
            if (kind == 1) v = sign * sl_binom(i, j);
            else if (kind == 2) v = sign * sl_binom(j, i);
            else { double s = 0.0; for (int64_t k = (i > j ? i : j); k < n; k++) s += sl_binom(k, i) * sl_binom(k, j); v = sign * s; }
            out[i * n + j] = v;
        }
    return TSR_OK;
}

/* convolution_matrix(a, n, mode='full'): the Toeplitz matrix A with A @ v = convolve(a, v, mode). a is length
   m; the full matrix is (m+n-1, n) with A[i][j] = a[i-j] when 0 <= i-j < m; 'same'/'valid' take a centred /
   fully-overlapping row slice (scipy.linalg.convolution_matrix). */
static int r_convolution_matrix(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("convolution_matrix: a must be a 1-D array"); return TSR_ETYPE; }
    int64_t n; int rc = sl_int(&args[1], "n", &n);
    if (rc < 0) return rc;
    if (n < 1) { fn_set_error("convolution_matrix: n must be a positive integer"); return TSR_EARG; }
    int64_t m; double *a = mat_f64(&args[0], "a", &m);
    if (!a) return TSR_ENOMEM;
    if (m < 1) { fn_free_doubles(a, m); fn_set_error("convolution_matrix: a must have at least one element"); return TSR_EARG; }
    int mode = 0; /* 0 full, 1 same, 2 valid */
    if (nargs > 2 && args[2].kind == 2 && args[2].str) {
        if (!strcmp(args[2].str, "same")) mode = 1;
        else if (!strcmp(args[2].str, "valid")) mode = 2;
        else if (strcmp(args[2].str, "full") != 0) { fn_free_doubles(a, m); fn_set_error("convolution_matrix: mode must be 'full', 'same' or 'valid'"); return TSR_EARG; }
    }
    const int64_t mn = m < n ? m : n, mx = m > n ? m : n;
    int64_t M, offset;
    if (mode == 0) { M = m + n - 1; offset = 0; }
    else if (mode == 1) { M = mx; offset = (mn - 1) / 2; }
    else { M = mx - mn + 1; offset = mn - 1; }
    int64_t osh[2] = {M, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(a, m); return TSR_ENOMEM; }
    for (int64_t r = 0; r < M; r++) {
        const int64_t i = r + offset;
        for (int64_t j = 0; j < n; j++) { const int64_t k = i - j; out[r * n + j] = (k >= 0 && k < m) ? a[k] : 0.0; }
    }
    fn_free_doubles(a, m);
    return TSR_OK;
}

/* dft(n, scale=None): the n x n DFT matrix D[j][k] = exp(-2*pi*i*j*k/n); scale 'sqrtn' divides by sqrt(n),
   'n' by n (scipy.linalg.dft). Complex128 result, interleaved (re, im). */
static int r_dft(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t n; int rc = sl_int(&args[0], "n", &n);
    if (rc < 0) return rc;
    if (n < 0) { fn_set_error("dft: n must be a non-negative integer"); return TSR_EARG; }
    double sc = 1.0;
    if (nargs > 1 && args[1].kind == 2 && args[1].str) {
        if (!strcmp(args[1].str, "sqrtn")) sc = n > 0 ? 1.0 / sqrt((double)n) : 1.0;
        else if (!strcmp(args[1].str, "n")) sc = n > 0 ? 1.0 / (double)n : 1.0;
        else { fn_set_error("dft: scale must be None, 'sqrtn' or 'n'"); return TSR_EARG; }
    }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_C128, 2, osh);
    if (!out) return TSR_ENOMEM;
    const double twopi = 2.0 * 3.14159265358979323846;
    for (int64_t j = 0; j < n; j++)
        for (int64_t k = 0; k < n; k++) {
            const double theta = -twopi * (double)((j * k) % n) / (double)n;   /* reduce j*k mod n for accuracy */
            const int64_t idx = 2 * (j * n + k);
            out[idx] = cos(theta) * sc;
            out[idx + 1] = sin(theta) * sc;
        }
    return TSR_OK;
}

/* khatri_rao(a, b): the column-wise Kronecker product; a is (ra, k), b is (rb, k), result is (ra*rb, k)
   with result[i*rb + l][j] = a[i][j] * b[l][j] (scipy.linalg.khatri_rao). */
static int r_khatri_rao(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) {
        fn_set_error("khatri_rao: a and b must be 2-D arrays"); return TSR_ETYPE;
    }
    const int64_t ra = args[0].arr.shape[0], ka = args[0].arr.shape[1];
    const int64_t rb = args[1].arr.shape[0], kb = args[1].arr.shape[1];
    if (ka != kb) { fn_set_error("The number of columns of A and B must be equal."); return TSR_EARG; }
    int64_t na, nb;
    double *A = mat_f64(&args[0], "a", &na);
    if (!A) return TSR_ENOMEM;
    double *B = mat_f64(&args[1], "b", &nb);
    if (!B) { fn_free_doubles(A, na); return TSR_ENOMEM; }
    int64_t osh[2] = {ra * rb, ka};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(A, na); fn_free_doubles(B, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < ra; i++)
        for (int64_t l = 0; l < rb; l++)
            for (int64_t j = 0; j < ka; j++)
                out[(i * rb + l) * ka + j] = A[i * ka + j] * B[l * ka + j];
    fn_free_doubles(A, na); fn_free_doubles(B, nb);
    return TSR_OK;
}

/* diagsvd(s, M, N): an M x N matrix with the values s on its main diagonal (scipy.linalg.diagsvd);
   len(s) must be M or N. */
static int r_diagsvd(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("diagsvd: s must be a 1-D array"); return TSR_ETYPE; }
    int64_t ns; double *s = mat_f64(&args[0], "s", &ns);
    if (!s) return TSR_ENOMEM;
    int64_t M, N; int rc;
    if ((rc = sl_int(&args[1], "M", &M)) < 0 || (rc = sl_int(&args[2], "N", &N)) < 0) { fn_free_doubles(s, ns); return rc; }
    if (ns != M && ns != N) { fn_free_doubles(s, ns); fn_set_error("Length of s must be M or N."); return TSR_EARG; }
    int64_t osh[2] = {M, N};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(s, ns); return TSR_ENOMEM; }
    memset(out, 0, sizeof(double) * (size_t)(M * N));
    const int64_t d = M < N ? M : N;
    for (int64_t i = 0; i < d && i < ns; i++) out[i * N + i] = s[i];
    fn_free_doubles(s, ns);
    return TSR_OK;
}

/* orth(A, rcond=None): an orthonormal basis for the range of A -- the left singular vectors whose singular value
   exceeds amax(s) * rcond (rcond defaults to max(M,N) * eps) (scipy.linalg.orth). Each basis column is sign-fixed
   so its largest-magnitude entry is positive (the basis is otherwise unique only up to the sign of each column). */
static int r_orth(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("orth: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], k = m < n ? m : n;
    int64_t na; double *a = mat_f64(&args[0], "A", &na);
    if (!a) return TSR_ENOMEM;
    double *u = (double *)malloc(sizeof(double) * (size_t)(m * (k ? k : 1)));
    double *s = (double *)malloc(sizeof(double) * (size_t)(k ? k : 1));
    double *vt = (double *)malloc(sizeof(double) * (size_t)((k ? k : 1) * (n ? n : 1)));
    if (!u || !s || !vt) { free(u); free(s); free(vt); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int64_t rank = 0;
    if (k > 0) {
        lapack_int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', (lapack_int)m, (lapack_int)n, a, (lapack_int)n, s, u, (lapack_int)k, vt, (lapack_int)n);
        if (info != 0) { free(u); free(s); free(vt); fn_free_doubles(a, na); fn_set_error("orth: SVD did not converge"); return TSR_EARG; }
        double rtol = (double)(m > n ? m : n) * DBL_EPSILON;
        if (nargs > 1 && args[1].kind == 1) rtol = args[1].num;
        const double tol = s[0] * rtol;                  /* s is descending; s[0] is the largest singular value */
        for (int64_t i = 0; i < k; i++) if (s[i] > tol) rank++;
    }
    int64_t osh[2] = {m, rank};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { free(u); free(s); free(vt); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < rank; j++) out[i * rank + j] = u[i * k + j];
    for (int64_t j = 0; j < rank; j++) {
        int64_t p = 0; double mx = -1.0;
        for (int64_t i = 0; i < m; i++) { double v = fabs(out[i * rank + j]); if (v > mx) { mx = v; p = i; } }
        if (out[p * rank + j] < 0.0) for (int64_t i = 0; i < m; i++) out[i * rank + j] = -out[i * rank + j];
    }
    free(u); free(s); free(vt); fn_free_doubles(a, na);
    return TSR_OK;
}

/* null_space(A, rcond=None): an orthonormal basis for the null space of A -- the right singular vectors whose
   singular value does not exceed the tolerance (scipy.linalg.null_space). Columns are sign-fixed as in orth. */
static int r_null_space(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("null_space: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], k = m < n ? m : n;
    int64_t na; double *a = mat_f64(&args[0], "A", &na);
    if (!a) return TSR_ENOMEM;
    double *u = (double *)malloc(sizeof(double) * (size_t)((m ? m : 1) * (m ? m : 1)));
    double *s = (double *)malloc(sizeof(double) * (size_t)(k ? k : 1));
    double *vt = (double *)malloc(sizeof(double) * (size_t)((n ? n : 1) * (n ? n : 1)));
    if (!u || !s || !vt) { free(u); free(s); free(vt); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int64_t rank = 0;
    if (k > 0) {
        lapack_int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', (lapack_int)m, (lapack_int)n, a, (lapack_int)n, s, u, (lapack_int)m, vt, (lapack_int)n);
        if (info != 0) { free(u); free(s); free(vt); fn_free_doubles(a, na); fn_set_error("null_space: SVD did not converge"); return TSR_EARG; }
        double rtol = (double)(m > n ? m : n) * DBL_EPSILON;
        if (nargs > 1 && args[1].kind == 1) rtol = args[1].num;
        const double tol = s[0] * rtol;
        for (int64_t i = 0; i < k; i++) if (s[i] > tol) rank++;
    }
    const int64_t nc = n - rank;
    int64_t osh[2] = {n, nc};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { free(u); free(s); free(vt); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < nc; j++) out[i * nc + j] = vt[(rank + j) * n + i];   /* Vh[rank+j, i] */
    for (int64_t j = 0; j < nc; j++) {
        int64_t p = 0; double mx = -1.0;
        for (int64_t i = 0; i < n; i++) { double v = fabs(out[i * nc + j]); if (v > mx) { mx = v; p = i; } }
        if (out[p * nc + j] < 0.0) for (int64_t i = 0; i < n; i++) out[i * nc + j] = -out[i * nc + j];
    }
    free(u); free(s); free(vt); fn_free_doubles(a, na);
    return TSR_OK;
}

/* polar(a, side='right'): the polar decomposition a = u p ('right') or a = p u ('left') from the SVD a = U S Vh;
   u = U Vh (unitary) and p is the positive-semidefinite factor (scipy.linalg.polar). Returns (u, p). Sign-invariant
   (a flip of U column l and Vh row l cancels), so no canonicalisation is needed. */
static int r_polar(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("polar: input must be a 2-D array"); return TSR_ETYPE; }
    int right = 1;
    if (nargs > 1 && args[1].kind == 2 && args[1].str) {
        if (!strcmp(args[1].str, "left")) right = 0;
        else if (strcmp(args[1].str, "right") != 0) { fn_set_error("polar: side must be 'left' or 'right'"); return TSR_EARG; }
    }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], k = m < n ? m : n;
    int64_t na; double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *u = (double *)malloc(sizeof(double) * (size_t)(m * (k ? k : 1)));
    double *s = (double *)malloc(sizeof(double) * (size_t)(k ? k : 1));
    double *vt = (double *)malloc(sizeof(double) * (size_t)((k ? k : 1) * (n ? n : 1)));
    if (!u || !s || !vt) { free(u); free(s); free(vt); fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (k > 0) {
        lapack_int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', (lapack_int)m, (lapack_int)n, a, (lapack_int)n, s, u, (lapack_int)k, vt, (lapack_int)n);
        if (info != 0) { free(u); free(s); free(vt); fn_free_doubles(a, na); fn_set_error("polar: SVD did not converge"); return TSR_EARG; }
    }
    int64_t ush[2] = {m, n};
    double *uu = (double *)fn_result_array(&res[0], TSR_F64, 2, ush);
    const int64_t pn = right ? n : m;
    int64_t psh[2] = {pn, pn};
    double *pp = (double *)fn_result_array(&res[1], TSR_F64, 2, psh);
    if (!uu || !pp) { free(u); free(s); free(vt); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t l = 0; l < k; l++) acc += u[i * k + l] * vt[l * n + j];
            uu[i * n + j] = acc;
        }
    if (right)
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = 0; j < n; j++) {
                double acc = 0.0;
                for (int64_t l = 0; l < k; l++) acc += vt[l * n + i] * s[l] * vt[l * n + j];
                pp[i * n + j] = acc;
            }
    else
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < m; j++) {
                double acc = 0.0;
                for (int64_t l = 0; l < k; l++) acc += u[i * k + l] * s[l] * u[j * k + l];
                pp[i * m + j] = acc;
            }
    free(u); free(s); free(vt); fn_free_doubles(a, na);
    return TSR_OK;
}

/* lu_factor(a): the LU factorisation for lu_solve (dgetrf). Returns (lu, piv): lu is the combined L\U factor and
   piv the 0-based row pivots (scipy returns LAPACK's 1-based ipiv minus one) (scipy.linalg.lu_factor). */
static int r_lu_factor(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("lu_factor: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], k = m < n ? m : n;
    int64_t na; double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(k > 0 ? k : 1));
    if (!ipiv) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (k > 0) {
        lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, a, (lapack_int)n, ipiv);
        if (info < 0) { free(ipiv); fn_free_doubles(a, na); fn_set_error("lu_factor: illegal value in dgetrf"); return TSR_EARG; }
        /* info > 0 is exact singularity; scipy returns the factors anyway (lu_solve would then be unreliable) */
    }
    int64_t lsh[2] = {m, n}, psh[1] = {k};
    double *lu = (double *)fn_result_array(&res[0], TSR_F64, 2, lsh);
    double *piv = (double *)fn_result_array(&res[1], TSR_F64, 1, psh);   /* 0-based pivots; the slinalg fixtures compare as float64 */
    if (!lu || !piv) { free(ipiv); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m * n; i++) lu[i] = a[i];
    for (int64_t i = 0; i < k; i++) piv[i] = (double)(ipiv[i] - 1);
    free(ipiv); fn_free_doubles(a, na);
    return TSR_OK;
}

/* lu_solve(lu, piv, b, trans=0): solve a x = b from an lu_factor result (dgetrs); lu, piv and b are passed
   unpacked. piv is the 0-based pivots from lu_factor (converted back to LAPACK's 1-based). trans 0/1/2 selects
   a x = b, a^T x = b or a^H x = b (scipy.linalg.lu_solve). */
static int r_lu_solve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3 || args[2].kind != 3) { fn_set_error("lu_solve: lu, piv and b must be arrays"); return TSR_ETYPE; }
    const tsr_array *LU = &args[0].arr, *B = &args[2].arr;
    if (LU->ndim != 2 || LU->shape[0] != LU->shape[1]) { fn_set_error("lu_solve: lu must be a square 2-D array"); return TSR_EARG; }
    const int64_t n = LU->shape[0];
    const int vector = B->ndim == 1;
    if (B->ndim < 1 || B->shape[0] != n) { fn_set_error("lu_solve: b has the wrong number of rows"); return TSR_EARG; }
    const int64_t nrhs = vector ? 1 : B->shape[1];
    char tr = 'N';
    if (nargs > 3 && args[3].kind == 1) { int64_t t = (args[3].flags & 1) ? args[3].ival : (int64_t)args[3].num; tr = (t == 1) ? 'T' : ((t == 2) ? 'C' : 'N'); }
    int64_t nlu, npv, nb;
    double *lu = mat_f64(&args[0], "lu", &nlu);
    if (!lu) return TSR_ENOMEM;
    double *pivd = mat_f64(&args[1], "piv", &npv);
    if (!pivd) { fn_free_doubles(lu, nlu); return TSR_ENOMEM; }
    double *bsrc = mat_f64(&args[2], "b", &nb);
    if (!bsrc) { fn_free_doubles(lu, nlu); fn_free_doubles(pivd, npv); return TSR_ENOMEM; }
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(n > 0 ? n : 1));
    double *x = (double *)malloc(sizeof(double) * (size_t)(n * nrhs > 0 ? n * nrhs : 1));
    if (!ipiv || !x) { free(ipiv); free(x); fn_free_doubles(lu, nlu); fn_free_doubles(pivd, npv); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) ipiv[i] = (lapack_int)((int64_t)pivd[i] + 1);
    for (int64_t i = 0; i < n * nrhs; i++) x[i] = bsrc[i];
    if (n > 0 && nrhs > 0) {
        lapack_int info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, tr, (lapack_int)n, (lapack_int)nrhs, lu, (lapack_int)n, ipiv, x, (lapack_int)nrhs);
        if (info != 0) { free(ipiv); free(x); fn_free_doubles(lu, nlu); fn_free_doubles(pivd, npv); fn_free_doubles(bsrc, nb); fn_set_error("lu_solve: dgetrs failed"); return TSR_EARG; }
    }
    int64_t sh1[1] = {n}, sh2[2] = {n, nrhs};
    double *out = (double *)(vector ? fn_result_array(&res[0], TSR_F64, 1, sh1) : fn_result_array(&res[0], TSR_F64, 2, sh2));
    if (!out) { free(ipiv); free(x); fn_free_doubles(lu, nlu); fn_free_doubles(pivd, npv); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n * nrhs; i++) out[i] = x[i];
    free(ipiv); free(x); fn_free_doubles(lu, nlu); fn_free_doubles(pivd, npv); fn_free_doubles(bsrc, nb);
    return TSR_OK;
}

/* hessenberg(a): the upper Hessenberg form H of a square matrix (H = Q^H A Q), via dgehrd. calc_q (the Q factor)
   is not yet supported. The Householder reduction is deterministic, so H matches scipy element-wise. */
static int r_hessenberg(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("hessenberg: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("hessenberg: expected a square matrix"); return TSR_EARG; }
    if (nargs > 1 && ((args[1].kind == 4 && args[1].num != 0) || (args[1].kind == 1 && (args[1].num != 0 || args[1].ival != 0)))) {
        fn_set_error("hessenberg: calc_q=True is not yet supported");
        return TSR_EARG;
    }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *tau = (double *)malloc(sizeof(double) * (size_t)(n > 1 ? n - 1 : 1));
    if (!tau) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (n > 1) {
        lapack_int info = LAPACKE_dgehrd(LAPACK_ROW_MAJOR, (lapack_int)n, 1, (lapack_int)n, a, (lapack_int)n, tau);
        if (info != 0) { free(tau); fn_free_doubles(a, na); fn_set_error("hessenberg: dgehrd failed"); return TSR_EARG; }
    }
    int64_t osh[2] = {n, n};
    double *h = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!h) { free(tau); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) h[i * n + j] = (i <= j + 1) ? a[i * n + j] : 0.0;
    free(tau);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* schur(a, output='real'): the real Schur decomposition A = Z T Z^T via dgees (jobvs='V', no sorting). T is the
   quasi-upper-triangular Schur form and Z the orthogonal Schur vectors. Returns (T, Z). output='complex' is not
   yet supported. */
static int r_schur(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("schur: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("schur: expected a square matrix"); return TSR_EARG; }
    if (nargs > 1 && args[1].kind == 2 && args[1].str && strcmp(args[1].str, "real") != 0) {
        fn_set_error("schur: only output='real' is supported");
        return TSR_EARG;
    }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);             /* overwritten with T */
    if (!a) return TSR_ENOMEM;
    double *wr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *wi = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    if (!wr || !wi) { free(wr); free(wi); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int64_t tsh[2] = {n, n}, zsh[2] = {n, n};
    double *t = (double *)fn_result_array(&res[0], TSR_F64, 2, tsh);
    double *z = (double *)fn_result_array(&res[1], TSR_F64, 2, zsh);
    if (!t || !z) { free(wr); free(wi); fn_free_doubles(a, na); return TSR_ENOMEM; }
    if (n > 0) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, &sdim, wr, wi, z, (lapack_int)n);
        if (info != 0) { free(wr); free(wi); fn_free_doubles(a, na); fn_set_error("schur: dgees failed"); return TSR_EARG; }
    }
    for (int64_t i = 0; i < n * n; i++) t[i] = a[i];     /* a now holds the Schur form T */
    free(wr); free(wi);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* Real Schur of an n x n row-major matrix: overwrites `a` with the quasi-triangular T, fills z with the
   orthogonal Schur vectors (A = Z T Z^T). 0 on success. */
static int sl_schur(double *a, double *z, int64_t n)
{
    if (n == 0) return 0;
    double *wr = (double *)malloc(sizeof(double) * (size_t)n);
    double *wi = (double *)malloc(sizeof(double) * (size_t)n);
    if (!wr || !wi) { free(wr); free(wi); return -1; }
    lapack_int sdim = 0;
    lapack_int info = LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, &sdim, wr, wi, z, (lapack_int)n);
    free(wr); free(wi);
    return info == 0 ? 0 : -1;
}

/* Solve the Sylvester equation A X + X B = Q (A: n x n, B: m x m, Q,X: n x m) by Bartels-Stewart:
   Schur A = Ua Ta Ua^T, B = Ub Tb Ub^T; C = Ua^T Q Ub; Ta Y + Y Tb = C (dtrsyl); X = Ua Y Ub^T. 0 on success. */
static int sl_sylvester(const double *A, const double *B, const double *Q, double *X, int64_t n, int64_t m)
{
    int rc = -1;
    double *Ta = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *Ua = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *Tb = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));
    double *Ub = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));
    double *C  = (double *)malloc(sizeof(double) * (size_t)(n * m > 0 ? n * m : 1));
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(n * m > 0 ? n * m : 1));
    if (!Ta || !Ua || !Tb || !Ub || !C || !tmp) goto done;
    memcpy(Ta, A, sizeof(double) * (size_t)(n * n));
    memcpy(Tb, B, sizeof(double) * (size_t)(m * m));
    if (sl_schur(Ta, Ua, n) != 0 || sl_schur(Tb, Ub, m) != 0) goto done;
    /* tmp = Ua^T Q  (tmp[i][j] = sum_k Ua[k][i] Q[k][j]) */
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < m; j++) { double s = 0; for (int64_t k = 0; k < n; k++) s += Ua[k * n + i] * Q[k * m + j]; tmp[i * m + j] = s; }
    /* C = tmp Ub */
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < m; j++) { double s = 0; for (int64_t k = 0; k < m; k++) s += tmp[i * m + k] * Ub[k * m + j]; C[i * m + j] = s; }
    {
        double scale = 1.0;
        lapack_int info = LAPACKE_dtrsyl(LAPACK_ROW_MAJOR, 'N', 'N', 1, (lapack_int)n, (lapack_int)m, Ta, (lapack_int)n, Tb, (lapack_int)m, C, (lapack_int)m, &scale);
        if (info < 0) goto done;   /* info > 0: common eigenvalues, perturbed solution still returned (as scipy does) */
        if (scale != 1.0 && scale != 0.0) for (int64_t i = 0; i < n * m; i++) C[i] /= scale;
    }
    /* tmp = Ua Y;  X = tmp Ub^T */
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < m; j++) { double s = 0; for (int64_t k = 0; k < n; k++) s += Ua[i * n + k] * C[k * m + j]; tmp[i * m + j] = s; }
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < m; j++) { double s = 0; for (int64_t k = 0; k < m; k++) s += tmp[i * m + k] * Ub[j * m + k]; X[i * m + j] = s; }
    rc = 0;
done:
    free(Ta); free(Ua); free(Tb); free(Ub); free(C); free(tmp);
    return rc;
}

/* solve_sylvester(a, b, q): solve a x + x b = q (scipy.linalg.solve_sylvester). */
static int r_solve_sylvester(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2 || args[2].kind != 3 || args[2].arr.ndim != 2) { fn_set_error("solve_sylvester: a, b, q must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], m = args[1].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[1] != m || args[2].arr.shape[0] != n || args[2].arr.shape[1] != m) { fn_set_error("solve_sylvester: incompatible shapes"); return TSR_EARG; }
    int64_t na, nb, nq;
    double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "b", &nb); if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *q = mat_f64(&args[2], "q", &nq); if (!q) { fn_free_doubles(a, na); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    int64_t osh[2] = {n, m};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    int rc = TSR_OK;
    if (!x) rc = TSR_ENOMEM;
    else if (sl_sylvester(a, b, q, x, n, m) != 0) { fn_set_error("solve_sylvester: the Sylvester solver failed (dgees/dtrsyl)"); rc = TSR_EARG; }
    fn_free_doubles(a, na); fn_free_doubles(b, nb); fn_free_doubles(q, nq);
    return rc;
}

/* solve_continuous_lyapunov(a, q): solve a x + x a^H = q (scipy.linalg.solve_continuous_lyapunov). For a real
   matrix a^H = a^T, so this is the Sylvester equation a x + x a^T = q. */
static int r_solve_continuous_lyapunov(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("solve_continuous_lyapunov: a and q must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[1].arr.shape[1] != n) { fn_set_error("solve_continuous_lyapunov: a and q must be square and the same size"); return TSR_EARG; }
    int64_t na, nq;
    double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *q = mat_f64(&args[1], "q", &nq); if (!q) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *at = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    int rc = TSR_OK;
    if (!at) { fn_free_doubles(a, na); fn_free_doubles(q, nq); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) at[i * n + j] = a[j * n + i];
    int64_t osh[2] = {n, n};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!x) rc = TSR_ENOMEM;
    else if (sl_sylvester(a, at, q, x, n, n) != 0) { fn_set_error("solve_continuous_lyapunov: the solver failed (dgees/dtrsyl)"); rc = TSR_EARG; }
    free(at);
    fn_free_doubles(a, na); fn_free_doubles(q, nq);
    return rc;
}

/* eigvalsh_tridiagonal(d, e): all eigenvalues (ascending) of the symmetric tridiagonal matrix with diagonal d
   and off-diagonal e, via dsterf (scipy.linalg.eigvalsh_tridiagonal). */
static int r_eigvalsh_tridiagonal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("eigvalsh_tridiagonal: d and e must be 1-D arrays"); return TSR_EARG; }
    int64_t n, ne;
    double *d = mat_f64(&args[0], "d", &n); if (!d) return TSR_ENOMEM;
    double *e = mat_f64(&args[1], "e", &ne); if (!e) { fn_free_doubles(d, n); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (ne != n - 1) { fn_set_error("eigvalsh_tridiagonal: e must have length len(d)-1"); rc = TSR_EARG; }
    else {
        if (n > 0) { lapack_int info = LAPACKE_dsterf((lapack_int)n, d, e); if (info != 0) { fn_set_error("eigvalsh_tridiagonal: dsterf failed to converge"); rc = TSR_EARG; } }
        if (rc == TSR_OK) {
            int64_t sh[1] = {n};
            double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
            if (!out) rc = TSR_ENOMEM; else for (int64_t i = 0; i < n; i++) out[i] = d[i];
        }
    }
    fn_free_doubles(d, n); fn_free_doubles(e, ne);
    return rc;
}

static void la_set_bool(tsr_result *res, int ok) { memset(res, 0, sizeof *res); res->kind = 4; res->num = ok ? 1.0 : 0.0; }

/* issymmetric(a) / ishermitian(a) for a real square matrix: exact equality a[i][j] == a[j][i], which is the
   scipy default when atol and rtol are None (scipy.linalg.issymmetric / ishermitian). */
static int r_issymmetric(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("issymmetric: a must be a square 2-D array"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    int64_t na; double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    int ok = 1;
    for (int64_t i = 0; i < n && ok; i++)
        for (int64_t j = i + 1; j < n; j++) if (a[i * n + j] != a[j * n + i]) { ok = 0; break; }
    fn_free_doubles(a, na);
    la_set_bool(res, ok);
    return TSR_OK;
}

/* Solve the dense n x n system M x = b in place (b is n x nrhs, row-major) via dgesv. 0 on success. */
static int sl_dense_solve(double *M, double *b, int64_t n, int64_t nrhs)
{
    if (n == 0) return 0;
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)n);
    if (!ipiv) return -1;
    lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)nrhs, M, (lapack_int)n, ipiv, b, (lapack_int)nrhs);
    free(ipiv);
    return info == 0 ? 0 : -1;
}

/* solve_toeplitz(c_or_cr, b): solve the Toeplitz system T x = b. A single 1-D c_or_cr gives a symmetric (real
   Hermitian) Toeplitz T[i][j] = c[|i-j|]; a (c, r) pair uses first column c and first row r. Built dense and
   solved with dgesv — the same solution scipy's Levinson recursion returns (scipy.linalg.solve_toeplitz). */
static int r_solve_toeplitz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    const tsr_arg *carg, *rarg;
    if (args[0].kind == 5 && args[0].count >= 2) { carg = &args[0].items[0]; rarg = &args[0].items[1]; }
    else if (args[0].kind == 3) { carg = &args[0]; rarg = &args[0]; }
    else { fn_set_error("solve_toeplitz: c_or_cr must be an array or a (c, r) pair"); return TSR_EARG; }
    if (carg->kind != 3 || carg->arr.ndim != 1 || rarg->kind != 3 || rarg->arr.ndim != 1 || args[1].kind != 3) { fn_set_error("solve_toeplitz: c, r must be 1-D arrays and b an array"); return TSR_EARG; }
    int64_t nc, nr = 0; double *c = mat_f64(carg, "c", &nc); if (!c) return TSR_ENOMEM;
    double *r = (rarg == carg) ? c : mat_f64(rarg, "r", &nr); if (!r) { fn_free_doubles(c, nc); return TSR_ENOMEM; }
    const int64_t n = nc;
    const tsr_array *B = &args[1].arr;
    const int vector = B->ndim == 1;
    const int64_t k = vector ? 1 : B->shape[1];
    int rc = TSR_OK;
    if (B->shape[0] != n) { if (r != c) fn_free_doubles(r, nr); fn_free_doubles(c, nc); fn_set_error("solve_toeplitz: b has the wrong number of rows"); return TSR_EARG; }
    double *M = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    int64_t nb = 0; double *b = mat_f64(&args[1], "b", &nb);
    if (!M || !b) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) M[i * n + j] = (i >= j) ? c[i - j] : r[j - i];
        if (sl_dense_solve(M, b, n, k) != 0) { fn_set_error("solve_toeplitz: the system is singular"); rc = TSR_EARG; }
        else { int64_t osh[2] = {n, k}; double *x = (double *)fn_result_array(&res[0], TSR_F64, vector ? 1 : 2, osh); if (!x) rc = TSR_ENOMEM; else memcpy(x, b, sizeof(double) * (size_t)(n * k)); }
    }
    free(M); if (b) fn_free_doubles(b, nb);
    if (r != c) fn_free_doubles(r, nr);
    fn_free_doubles(c, nc);
    return rc;
}

/* solve_circulant(c, b): solve the circulant system C x = b, where C[i][j] = c[(i-j) mod n]. Built dense and
   solved with dgesv — the same solution scipy's FFT method returns (scipy.linalg.solve_circulant). */
static int r_solve_circulant(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3) { fn_set_error("solve_circulant: c must be 1-D and b an array"); return TSR_EARG; }
    int64_t n; double *c = mat_f64(&args[0], "c", &n); if (!c) return TSR_ENOMEM;
    const tsr_array *B = &args[1].arr;
    const int vector = B->ndim == 1;
    const int64_t k = vector ? 1 : B->shape[1];
    int rc = TSR_OK;
    if (B->shape[0] != n) { fn_free_doubles(c, n); fn_set_error("solve_circulant: b has the wrong number of rows"); return TSR_EARG; }
    double *M = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    int64_t nb = 0; double *b = mat_f64(&args[1], "b", &nb);
    if (!M || !b) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) { int64_t kk = (i - j) % n; if (kk < 0) kk += n; M[i * n + j] = c[kk]; }
        if (sl_dense_solve(M, b, n, k) != 0) { fn_set_error("solve_circulant: the system is singular"); rc = TSR_EARG; }
        else { int64_t osh[2] = {n, k}; double *x = (double *)fn_result_array(&res[0], TSR_F64, vector ? 1 : 2, osh); if (!x) rc = TSR_ENOMEM; else memcpy(x, b, sizeof(double) * (size_t)(n * k)); }
    }
    free(M); if (b) fn_free_doubles(b, nb);
    fn_free_doubles(c, n);
    return rc;
}

/* matmul_toeplitz(c_or_cr, x): the product T @ x where T is the Toeplitz matrix with first column c and first
   row r (a single 1-D c_or_cr gives r = c). x is (n,) or (n, k) (scipy.linalg.matmul_toeplitz). */
static int r_matmul_toeplitz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    const tsr_arg *carg, *rarg;
    if (args[0].kind == 5 && args[0].count >= 2) { carg = &args[0].items[0]; rarg = &args[0].items[1]; }
    else if (args[0].kind == 3) { carg = &args[0]; rarg = &args[0]; }
    else { fn_set_error("matmul_toeplitz: c_or_cr must be an array or a (c, r) pair"); return TSR_EARG; }
    if (carg->kind != 3 || carg->arr.ndim != 1 || rarg->kind != 3 || rarg->arr.ndim != 1 || args[1].kind != 3) { fn_set_error("matmul_toeplitz: c, r must be 1-D and x an array"); return TSR_EARG; }
    int64_t nc, nr = 0; double *c = mat_f64(carg, "c", &nc); if (!c) return TSR_ENOMEM;
    double *r = (rarg == carg) ? c : mat_f64(rarg, "r", &nr); if (!r) { fn_free_doubles(c, nc); return TSR_ENOMEM; }
    const int64_t m = nc, n = (rarg == carg) ? nc : nr;
    const tsr_array *X = &args[1].arr;
    const int vector = X->ndim == 1;
    const int64_t k = vector ? 1 : X->shape[1];
    int rc = TSR_OK;
    if (X->shape[0] != n) { if (r != c) fn_free_doubles(r, nr); fn_free_doubles(c, nc); fn_set_error("matmul_toeplitz: x has the wrong number of rows"); return TSR_EARG; }
    int64_t nx = 0; double *x = mat_f64(&args[1], "x", &nx);
    int64_t osh[2] = {m, k};
    double *out = x ? (double *)fn_result_array(&res[0], TSR_F64, vector ? 1 : 2, osh) : NULL;
    if (!x || !out) rc = TSR_ENOMEM;
    else
        for (int64_t i = 0; i < m; i++) for (int64_t col = 0; col < k; col++) {
            double s = 0.0;
            for (int64_t j = 0; j < n; j++) { const double t = (i >= j) ? c[i - j] : r[j - i]; s += t * x[j * k + col]; }
            out[i * k + col] = s;
        }
    if (x) fn_free_doubles(x, nx);
    if (r != c) fn_free_doubles(r, nr);
    fn_free_doubles(c, nc);
    return rc;
}

/* invhilbert(n): the inverse of the order-n Hilbert matrix, from the exact formula
   H^-1[i][j] = (-1)^(i+j) (i+j+1) C(n+i, n-j-1) C(n+j, n-i-1) C(i+j, i)^2 (scipy.linalg.invhilbert). */
static int r_invhilbert(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    int64_t n; int rc = sl_int(&args[0], "n", &n);
    if (rc < 0) return rc;
    if (n < 0) { fn_set_error("invhilbert: n must be a non-negative integer"); return TSR_EARG; }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            const double sign = ((i + j) & 1) ? -1.0 : 1.0;
            const double cij = sl_binom(i + j, i);
            out[i * n + j] = sign * (double)(i + j + 1) * sl_binom(n + i, n - j - 1) * sl_binom(n + j, n - i - 1) * cij * cij;
        }
    return TSR_OK;
}

/* eigh_tridiagonal(d, e): eigenvalues (ascending) and eigenvectors of a symmetric tridiagonal matrix via dstev;
   eigenvector columns are sign-canonicalised as in eigh (scipy.linalg.eigh_tridiagonal). Returns (w, v). */
static int r_eigh_tridiagonal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("eigh_tridiagonal: d and e must be 1-D arrays"); return TSR_EARG; }
    int64_t n, ne;
    double *d = mat_f64(&args[0], "d", &n); if (!d) return TSR_ENOMEM;
    double *e = mat_f64(&args[1], "e", &ne); if (!e) { fn_free_doubles(d, n); return TSR_ENOMEM; }
    if (ne != n - 1) { fn_free_doubles(d, n); fn_free_doubles(e, ne); fn_set_error("eigh_tridiagonal: e must have length len(d)-1"); return TSR_EARG; }
    int64_t wsh[1] = {n}, vsh[2] = {n, n};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, wsh);
    double *z = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *v = (double *)fn_result_array(&res[1], TSR_F64, 2, vsh);
    int rc = TSR_OK;
    if (!w || !z || !v) rc = TSR_ENOMEM;
    else {
        if (n > 0) { lapack_int info = LAPACKE_dstev(LAPACK_ROW_MAJOR, 'V', (lapack_int)n, d, e, z, (lapack_int)n); if (info != 0) { rc = TSR_EARG; fn_set_error("eigh_tridiagonal: dstev failed to converge"); } }
        if (rc == TSR_OK) {
            for (int64_t i = 0; i < n; i++) w[i] = d[i];
            sign_canon(z, n, n, n, NULL, 0);
            for (int64_t i = 0; i < n * n; i++) v[i] = z[i];
        }
    }
    free(z); fn_free_doubles(d, n); fn_free_doubles(e, ne);
    return rc;
}

/* eigvals_banded(a_band, lower=False): eigenvalues (ascending) of a symmetric banded matrix given in band
   storage (kd+1, n), via dsbevd with jobz='N' (scipy.linalg.eigvals_banded). */
static int r_eigvals_banded(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("eigvals_banded: a_band must be a 2-D array"); return TSR_ETYPE; }
    const int64_t kd = args[0].arr.shape[0] - 1, n = args[0].arr.shape[1];
    char uplo = 'U';
    if (nargs > 1 && ((args[1].kind == 4 && args[1].num != 0) || (args[1].kind == 1 && (args[1].num != 0 || args[1].ival != 0)))) uplo = 'L';
    int64_t nab; double *abd = mat_f64(&args[0], "a_band", &nab); if (!abd) return TSR_ENOMEM;
    const int64_t ldab = kd + 1;
    double *AB = (double *)malloc(sizeof(double) * (size_t)(ldab * n > 0 ? ldab * n : 1));
    double *zc = (double *)malloc(sizeof(double));
    int rc = TSR_OK;
    if (!AB || !zc) rc = TSR_ENOMEM;
    else {
        for (int64_t j = 0; j < n; j++) for (int64_t r = 0; r < ldab; r++) AB[r + j * ldab] = abd[r * n + j];
        int64_t wsh[1] = {n};
        double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, wsh);
        if (!w) rc = TSR_ENOMEM;
        else if (n > 0) { lapack_int info = LAPACKE_dsbevd(LAPACK_COL_MAJOR, 'N', uplo, (lapack_int)n, (lapack_int)kd, AB, (lapack_int)ldab, w, zc, 1); if (info != 0) { rc = TSR_EARG; fn_set_error("eigvals_banded: dsbevd failed to converge"); } }
    }
    free(AB); free(zc); fn_free_doubles(abd, nab);
    return rc;
}

/* cholesky_banded(ab, lower=False): the Cholesky factor of a symmetric positive-definite banded matrix given
   in band storage (kd+1, n), via dpbtrf; the factor is returned in the same band storage (scipy.linalg.cholesky_banded). */
static int r_cholesky_banded(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("cholesky_banded: ab must be a 2-D array"); return TSR_ETYPE; }
    const int64_t kd = args[0].arr.shape[0] - 1, n = args[0].arr.shape[1];
    char uplo = 'U';
    if (nargs > 1 && ((args[1].kind == 4 && args[1].num != 0) || (args[1].kind == 1 && (args[1].num != 0 || args[1].ival != 0)))) uplo = 'L';
    int64_t nab; double *abd = mat_f64(&args[0], "ab", &nab); if (!abd) return TSR_ENOMEM;
    const int64_t ldab = kd + 1;
    double *AB = (double *)malloc(sizeof(double) * (size_t)(ldab * n > 0 ? ldab * n : 1));
    int rc = TSR_OK;
    if (!AB) rc = TSR_ENOMEM;
    else {
        for (int64_t j = 0; j < n; j++) for (int64_t r = 0; r < ldab; r++) AB[r + j * ldab] = abd[r * n + j];
        int64_t osh[2] = {ldab, n};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
        if (!out) rc = TSR_ENOMEM;
        else {
            if (n > 0) { lapack_int info = LAPACKE_dpbtrf(LAPACK_COL_MAJOR, uplo, (lapack_int)n, (lapack_int)kd, AB, (lapack_int)ldab); if (info != 0) { rc = TSR_EARG; fn_set_error("cholesky_banded: the matrix is not positive definite"); } }
            if (rc == TSR_OK) for (int64_t j = 0; j < n; j++) for (int64_t r = 0; r < ldab; r++) out[r * n + j] = AB[r + j * ldab];
        }
    }
    free(AB); fn_free_doubles(abd, nab);
    return rc;
}

/* solve_discrete_lyapunov(a, q): solve the discrete Lyapunov (Stein) equation A X A^H - X + Q = 0, i.e.
   X - A X A^H = Q, by the direct method (I - A (x) conj(A)) vec(X) = vec(Q) with C-order vec; for real a this
   is A (x) A, solved with dgesv (scipy.linalg.solve_discrete_lyapunov, method='direct'). */
static int r_solve_discrete_lyapunov(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("solve_discrete_lyapunov: a and q must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[1].arr.shape[1] != n) { fn_set_error("solve_discrete_lyapunov: a and q must be square and the same size"); return TSR_EARG; }
    int64_t na, nq;
    double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *q = mat_f64(&args[1], "q", &nq); if (!q) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    const int64_t n2 = n * n;
    double *M = (double *)malloc(sizeof(double) * (size_t)(n2 * n2 > 0 ? n2 * n2 : 1));
    double *rhs = (double *)malloc(sizeof(double) * (size_t)(n2 > 0 ? n2 : 1));
    int rc = TSR_OK;
    if (!M || !rhs) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < n; i++) for (int64_t k = 0; k < n; k++) {
            const int64_t row = i * n + k;
            for (int64_t j = 0; j < n; j++) for (int64_t l = 0; l < n; l++) {
                const int64_t col = j * n + l;
                M[row * n2 + col] = (row == col ? 1.0 : 0.0) - a[i * n + j] * a[k * n + l];
            }
        }
        for (int64_t i = 0; i < n2; i++) rhs[i] = q[i];
        if (sl_dense_solve(M, rhs, n2, 1) != 0) { fn_set_error("solve_discrete_lyapunov: the Stein system is singular"); rc = TSR_EARG; }
        else { int64_t osh[2] = {n, n}; double *x = (double *)fn_result_array(&res[0], TSR_F64, 2, osh); if (!x) rc = TSR_ENOMEM; else for (int64_t i = 0; i < n2; i++) x[i] = rhs[i]; }
    }
    free(M); free(rhs); fn_free_doubles(a, na); fn_free_doubles(q, nq);
    return rc;
}

/* helmert(n, full=False): the Helmert matrix of order n. Contrast row k (0-based, k = 0..n-2) is
   1/sqrt((k+1)(k+2)) in columns 0..k and -(k+1)/sqrt((k+1)(k+2)) in column k+1; full prepends the mean row
   1/sqrt(n) (scipy.linalg.helmert). */
static int r_helmert(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t n; int rc = sl_int(&args[0], "n", &n);
    if (rc < 0) return rc;
    if (n < 1) { fn_set_error("helmert: n must be a positive integer"); return TSR_EARG; }
    int full = 0;
    if (nargs > 1 && ((args[1].kind == 4 && args[1].num != 0) || (args[1].kind == 1 && (args[1].num != 0 || args[1].ival != 0)))) full = 1;
    const int64_t m = full ? n : n - 1;
    int64_t osh[2] = {m, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) return TSR_ENOMEM;
    for (int64_t i = 0; i < m * n; i++) out[i] = 0.0;
    int64_t base = 0;
    if (full) { const double v = 1.0 / sqrt((double)n); for (int64_t j = 0; j < n; j++) out[j] = v; base = 1; }
    for (int64_t k = 0; k < n - 1; k++) {
        const double d = (double)(k + 1), s = sqrt(d * (d + 1.0));
        double *row = out + (base + k) * n;
        for (int64_t j = 0; j <= k; j++) row[j] = 1.0 / s;
        row[k + 1] = -d / s;
    }
    return TSR_OK;
}

/* cho_solve_banded(cb, lower, b): solve A x = b given the banded Cholesky factor cb (from cholesky_banded) and
   the lower flag, via dpbtrs. SciPy's ((cb, lower), b) tuple is taken unpacked here (scipy.linalg.cho_solve_banded). */
static int r_cho_solve_banded(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[2].kind != 3) { fn_set_error("cho_solve_banded: cb must be 2-D and b an array"); return TSR_EARG; }
    int lower = 0;
    if (args[1].kind == 4 || args[1].kind == 1) lower = (args[1].num != 0 || args[1].ival != 0);
    const char uplo = lower ? 'L' : 'U';
    const int64_t kd = args[0].arr.shape[0] - 1, n = args[0].arr.shape[1];
    const tsr_array *B = &args[2].arr;
    const int vector = B->ndim == 1;
    const int64_t nrhs = vector ? 1 : B->shape[1];
    if (B->shape[0] != n) { fn_set_error("cho_solve_banded: b has the wrong number of rows"); return TSR_EARG; }
    int64_t ncb, nb;
    double *cbd = mat_f64(&args[0], "cb", &ncb); if (!cbd) return TSR_ENOMEM;
    double *b = mat_f64(&args[2], "b", &nb); if (!b) { fn_free_doubles(cbd, ncb); return TSR_ENOMEM; }
    const int64_t ldab = kd + 1;
    double *AB = (double *)malloc(sizeof(double) * (size_t)(ldab * n > 0 ? ldab * n : 1));
    double *bcm = (double *)malloc(sizeof(double) * (size_t)(n * nrhs > 0 ? n * nrhs : 1));
    int rc = TSR_OK;
    if (!AB || !bcm) rc = TSR_ENOMEM;
    else {
        for (int64_t j = 0; j < n; j++) for (int64_t r = 0; r < ldab; r++) AB[r + j * ldab] = cbd[r * n + j];
        for (int64_t i = 0; i < n; i++) for (int64_t cc = 0; cc < nrhs; cc++) bcm[i + cc * n] = b[i * nrhs + cc];
        if (n > 0) { lapack_int info = LAPACKE_dpbtrs(LAPACK_COL_MAJOR, uplo, (lapack_int)n, (lapack_int)kd, (lapack_int)nrhs, AB, (lapack_int)ldab, bcm, (lapack_int)n); if (info != 0) { rc = TSR_EARG; fn_set_error("cho_solve_banded: dpbtrs failed"); } }
        if (rc == TSR_OK) {
            int64_t rsh1[1] = {n}, rsh2[2] = {n, nrhs};
            double *x = (double *)(vector ? fn_result_array(&res[0], TSR_F64, 1, rsh1) : fn_result_array(&res[0], TSR_F64, 2, rsh2));
            if (!x) rc = TSR_ENOMEM; else for (int64_t i = 0; i < n; i++) for (int64_t cc = 0; cc < nrhs; cc++) x[i * nrhs + cc] = bcm[i + cc * n];
        }
    }
    free(AB); free(bcm); fn_free_doubles(cbd, ncb); fn_free_doubles(b, nb);
    return rc;
}

/* rq(a, mode='full'): the RQ decomposition a = R Q of a square matrix via dgerqf + dorgrq (R upper triangular,
   Q orthogonal). Uses the same LAPACK routines as scipy, so the factors match exactly (scipy.linalg.rq).
   Returns (R, Q). */
static int r_rq(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("rq: a must be a square 2-D array"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    int64_t na; double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *tau = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    int64_t rsh[2] = {n, n}, qsh[2] = {n, n};
    double *R = (double *)fn_result_array(&res[0], TSR_F64, 2, rsh);
    double *Q = (double *)fn_result_array(&res[1], TSR_F64, 2, qsh);
    int rc = TSR_OK;
    if (!tau || !R || !Q) rc = TSR_ENOMEM;
    else if (n > 0) {
        lapack_int info = LAPACKE_dgerqf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n, a, (lapack_int)n, tau);
        if (info != 0) { rc = TSR_EARG; fn_set_error("rq: dgerqf failed"); }
        else {
            for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) R[i * n + j] = (j >= i) ? a[i * n + j] : 0.0;
            info = LAPACKE_dorgrq(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n, (lapack_int)n, a, (lapack_int)n, tau);
            if (info != 0) { rc = TSR_EARG; fn_set_error("rq: dorgrq failed"); }
            else for (int64_t i = 0; i < n * n; i++) Q[i] = a[i];
        }
    }
    free(tau); fn_free_doubles(a, na);
    return rc;
}

/* orthogonal_procrustes(A, B): the orthogonal matrix R minimising ||A R - B||_F, and scale = sum of singular
   values. R = U Vt where U S Vt = svd(A^T B); scale = sum(S) (scipy.linalg.orthogonal_procrustes). Returns
   (R, scale). */
static int r_orthogonal_procrustes(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("orthogonal_procrustes: A and B must be 2-D arrays"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1];
    if (args[1].arr.shape[0] != m || args[1].arr.shape[1] != n) { fn_set_error("orthogonal_procrustes: A and B must have the same shape"); return TSR_EARG; }
    int64_t nA, nB;
    double *A = mat_f64(&args[0], "A", &nA); if (!A) return TSR_ENOMEM;
    double *B = mat_f64(&args[1], "B", &nB); if (!B) { fn_free_doubles(A, nA); return TSR_ENOMEM; }
    double *M = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *U = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *Vt = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *s = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    int rc = TSR_OK;
    if (!M || !U || !Vt || !s) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) { double acc = 0; for (int64_t k = 0; k < m; k++) acc += A[k * n + i] * B[k * n + j]; M[i * n + j] = acc; }
        lapack_int info = n > 0 ? LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', (lapack_int)n, (lapack_int)n, M, (lapack_int)n, s, U, (lapack_int)n, Vt, (lapack_int)n) : 0;
        if (info != 0) { rc = TSR_EARG; fn_set_error("orthogonal_procrustes: SVD did not converge"); }
        else {
            int64_t rsh[2] = {n, n};
            double *R = (double *)fn_result_array(&res[0], TSR_F64, 2, rsh);
            if (!R) rc = TSR_ENOMEM;
            else {
                for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) { double acc = 0; for (int64_t k = 0; k < n; k++) acc += U[i * n + k] * Vt[k * n + j]; R[i * n + j] = acc; }
                double sc = 0; for (int64_t i = 0; i < n; i++) sc += s[i];
                fn_result_num(&res[1], sc);
            }
        }
    }
    free(M); free(U); free(Vt); free(s); fn_free_doubles(A, nA); fn_free_doubles(B, nB);
    return rc;
}

/* matrix_balance(A, ...): balance a square matrix via dgebal (job='B'); returns the balanced matrix B and the
   similarity transform T = diag(scale) with B = T^-1 A T. For a generic dense matrix dgebal performs no
   permutation, so T is the diagonal scaling (scipy.linalg.matrix_balance, separate=False). Returns (B, T). */
static int r_matrix_balance(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("matrix_balance: A must be a square 2-D array"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    int64_t na; double *a = mat_f64(&args[0], "A", &na); if (!a) return TSR_ENOMEM;
    double *scale = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    int rc = TSR_OK;
    lapack_int ilo = 1, ihi = (lapack_int)n;
    if (!scale) rc = TSR_ENOMEM;
    else if (n > 0) {
        lapack_int info = LAPACKE_dgebal(LAPACK_ROW_MAJOR, 'B', (lapack_int)n, a, (lapack_int)n, &ilo, &ihi, scale);
        if (info != 0) { rc = TSR_EARG; fn_set_error("matrix_balance: dgebal failed"); }
    }
    if (rc == TSR_OK) {
        int64_t bsh[2] = {n, n}, tsh[2] = {n, n};
        double *B = (double *)fn_result_array(&res[0], TSR_F64, 2, bsh);
        double *T = (double *)fn_result_array(&res[1], TSR_F64, 2, tsh);
        if (!B || !T) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < n * n; i++) { B[i] = a[i]; T[i] = 0.0; } for (int64_t i = 0; i < n; i++) T[i * n + i] = scale[i]; }
    }
    free(scale); fn_free_doubles(a, na);
    return rc;
}

/* solve_banded(l, u, ab, b): solve a banded system a x = b (dgbsv). l/u are the sub/super-diagonal counts and ab
   is the (l+u+1) x n band storage ab[u+i-j, j] = a[i,j]; args are passed unpacked. Returns x. */
static int r_solve_banded(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    int64_t l, u, rc;
    if ((rc = sl_int(&args[0], "l", &l)) < 0) return (int)rc;
    if ((rc = sl_int(&args[1], "u", &u)) < 0) return (int)rc;
    if (args[2].kind != 3 || args[2].arr.ndim != 2 || args[3].kind != 3) { fn_set_error("solve_banded: ab must be 2-D and b an array"); return TSR_ETYPE; }
    const int64_t nbands = args[2].arr.shape[0], n = args[2].arr.shape[1];
    if (nbands != l + u + 1) { fn_set_error("solve_banded: ab must have l+u+1 rows"); return TSR_EARG; }
    const tsr_array *B = &args[3].arr;
    if (B->shape[0] != n) { fn_set_error("solve_banded: b has the wrong number of rows"); return TSR_EARG; }
    const int vector = B->ndim == 1;
    const int64_t nrhs = vector ? 1 : B->shape[1];
    int64_t nab, nb;
    double *abd = mat_f64(&args[2], "ab", &nab);
    if (!abd) return TSR_ENOMEM;
    double *bsrc = mat_f64(&args[3], "b", &nb);
    if (!bsrc) { fn_free_doubles(abd, nab); return TSR_ENOMEM; }
    const int64_t ldab = 2 * l + u + 1;
    double *AB = (double *)calloc((size_t)(ldab * n > 0 ? ldab * n : 1), sizeof(double));
    double *Bc = (double *)malloc(sizeof(double) * (size_t)(n * nrhs > 0 ? n * nrhs : 1));
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(n > 0 ? n : 1));
    if (!AB || !Bc || !ipiv) { free(AB); free(Bc); free(ipiv); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t j = 0; j < n; j++)
        for (int64_t r = 0; r < nbands; r++) AB[(l + r) + j * ldab] = abd[r * n + j];
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < nrhs; j++) Bc[i + j * n] = bsrc[i * nrhs + j];
    lapack_int info = 0;
    if (n > 0) info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, (lapack_int)n, (lapack_int)l, (lapack_int)u, (lapack_int)nrhs, AB, (lapack_int)ldab, ipiv, Bc, (lapack_int)n);
    if (info != 0) { free(AB); free(Bc); free(ipiv); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb); fn_set_error("solve_banded: dgbsv failed (singular or illegal value)"); return TSR_EARG; }
    int64_t sh1[1] = {n}, sh2[2] = {n, nrhs};
    double *out = (double *)(vector ? fn_result_array(&res[0], TSR_F64, 1, sh1) : fn_result_array(&res[0], TSR_F64, 2, sh2));
    if (!out) { free(AB); free(Bc); free(ipiv); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < nrhs; j++) out[i * nrhs + j] = Bc[i + j * n];
    free(AB); free(Bc); free(ipiv); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb);
    return TSR_OK;
}

/* solveh_banded(ab, b, lower=False): solve a Hermitian positive-definite banded system (dpbsv). ab is the
   (kd+1) x n band storage (upper by default); its layout is exactly LAPACK's. Returns x. */
static int r_solveh_banded(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3) { fn_set_error("solveh_banded: ab must be 2-D and b an array"); return TSR_ETYPE; }
    const int64_t kd = args[0].arr.shape[0] - 1, n = args[0].arr.shape[1];
    const tsr_array *B = &args[1].arr;
    if (B->shape[0] != n) { fn_set_error("solveh_banded: b has the wrong number of rows"); return TSR_EARG; }
    const int vector = B->ndim == 1;
    const int64_t nrhs = vector ? 1 : B->shape[1];
    char uplo = 'U';
    if (nargs > 2 && ((args[2].kind == 4 && args[2].num != 0) || (args[2].kind == 1 && (args[2].num != 0 || args[2].ival != 0)))) uplo = 'L';
    int64_t nab, nb;
    double *abd = mat_f64(&args[0], "ab", &nab);
    if (!abd) return TSR_ENOMEM;
    double *bsrc = mat_f64(&args[1], "b", &nb);
    if (!bsrc) { fn_free_doubles(abd, nab); return TSR_ENOMEM; }
    const int64_t ldab = kd + 1;
    double *AB = (double *)malloc(sizeof(double) * (size_t)(ldab * n > 0 ? ldab * n : 1));
    double *Bc = (double *)malloc(sizeof(double) * (size_t)(n * nrhs > 0 ? n * nrhs : 1));
    if (!AB || !Bc) { free(AB); free(Bc); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t j = 0; j < n; j++)
        for (int64_t r = 0; r < ldab; r++) AB[r + j * ldab] = abd[r * n + j];
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < nrhs; j++) Bc[i + j * n] = bsrc[i * nrhs + j];
    lapack_int info = 0;
    if (n > 0) info = LAPACKE_dpbsv(LAPACK_COL_MAJOR, uplo, (lapack_int)n, (lapack_int)kd, (lapack_int)nrhs, AB, (lapack_int)ldab, Bc, (lapack_int)n);
    if (info != 0) { free(AB); free(Bc); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb); fn_set_error("solveh_banded: dpbsv failed (not positive definite or illegal value)"); return TSR_EARG; }
    int64_t sh1[1] = {n}, sh2[2] = {n, nrhs};
    double *out = (double *)(vector ? fn_result_array(&res[0], TSR_F64, 1, sh1) : fn_result_array(&res[0], TSR_F64, 2, sh2));
    if (!out) { free(AB); free(Bc); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < nrhs; j++) out[i * nrhs + j] = Bc[i + j * n];
    free(AB); free(Bc); fn_free_doubles(abd, nab); fn_free_doubles(bsrc, nb);
    return TSR_OK;
}

/* eig_banded(ab, lower=False): eigenvalues (ascending) and eigenvectors of a symmetric banded matrix via dsbevd.
   ab is the (kd+1) x n band storage (upper by default). Returns (w, v); eigenvector columns are sign-canonicalised
   like eigh. eigvals_only / select are not yet supported. */
static int r_eig_banded(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("eig_banded: a_band must be a 2-D array"); return TSR_ETYPE; }
    const int64_t kd = args[0].arr.shape[0] - 1, n = args[0].arr.shape[1];
    char uplo = 'U';
    if (nargs > 1 && ((args[1].kind == 4 && args[1].num != 0) || (args[1].kind == 1 && (args[1].num != 0 || args[1].ival != 0)))) uplo = 'L';
    if (nargs > 2 && ((args[2].kind == 4 && args[2].num != 0) || (args[2].kind == 1 && (args[2].num != 0 || args[2].ival != 0)))) {
        fn_set_error("eig_banded: eigvals_only=True is not yet supported");
        return TSR_EARG;
    }
    int64_t nab;
    double *abd = mat_f64(&args[0], "a_band", &nab);
    if (!abd) return TSR_ENOMEM;
    const int64_t ldab = kd + 1;
    double *AB = (double *)malloc(sizeof(double) * (size_t)(ldab * n > 0 ? ldab * n : 1));
    double *zc = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    if (!AB || !zc) { free(AB); free(zc); fn_free_doubles(abd, nab); return TSR_ENOMEM; }
    for (int64_t j = 0; j < n; j++)
        for (int64_t r = 0; r < ldab; r++) AB[r + j * ldab] = abd[r * n + j];
    int64_t wsh[1] = {n}, vsh[2] = {n, n};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, wsh);
    double *v = (double *)fn_result_array(&res[1], TSR_F64, 2, vsh);
    if (!w || !v) { free(AB); free(zc); fn_free_doubles(abd, nab); return TSR_ENOMEM; }
    if (n > 0) {
        lapack_int info = LAPACKE_dsbevd(LAPACK_COL_MAJOR, 'V', uplo, (lapack_int)n, (lapack_int)kd, AB, (lapack_int)ldab, w, zc, (lapack_int)n);
        if (info != 0) { free(AB); free(zc); fn_free_doubles(abd, nab); fn_set_error("eig_banded: dsbevd failed to converge"); return TSR_EARG; }
    }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) v[i * n + j] = zc[i + j * n];   /* col-major eigenvectors -> row-major */
    sign_canon(v, n, n, n, NULL, 0);
    free(AB); free(zc);
    fn_free_doubles(abd, nab);
    return TSR_OK;
}

/* qz(A, B, output='real'): the generalised (real) Schur decomposition A = Q AA Z^T, B = Q BB Z^T via dgges
   (jobvsl=jobvsr='V', no sorting). Returns (AA, BB, Q, Z). output='complex' is not yet supported. */
static int r_qz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("qz: A and B must be 2-D arrays"); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[1].arr.shape[1] != n) { fn_set_error("qz: A and B must be square and the same size"); return TSR_EARG; }
    if (nargs > 2 && args[2].kind == 2 && args[2].str && strcmp(args[2].str, "real") != 0) { fn_set_error("qz: only output='real' is supported"); return TSR_EARG; }
    int64_t na, nb;
    double *a = mat_f64(&args[0], "A", &na);
    if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "B", &nb);
    if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *ar = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *ai = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *be = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    int64_t sh[2] = {n, n};
    double *AA = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    double *BB = (double *)fn_result_array(&res[1], TSR_F64, 2, sh);
    double *Q = (double *)fn_result_array(&res[2], TSR_F64, 2, sh);
    double *Z = (double *)fn_result_array(&res[3], TSR_F64, 2, sh);
    if (!ar || !ai || !be || !AA || !BB || !Q || !Z) { free(ar); free(ai); free(be); fn_free_doubles(a, na); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    if (n > 0) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgges(LAPACK_ROW_MAJOR, 'V', 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, b, (lapack_int)n, &sdim, ar, ai, be, Q, (lapack_int)n, Z, (lapack_int)n);
        if (info != 0) { free(ar); free(ai); free(be); fn_free_doubles(a, na); fn_free_doubles(b, nb); fn_set_error("qz: dgges failed"); return TSR_EARG; }
    }
    for (int64_t i = 0; i < n * n; i++) { AA[i] = a[i]; BB[i] = b[i]; }
    free(ar); free(ai); free(be);
    fn_free_doubles(a, na); fn_free_doubles(b, nb);
    return TSR_OK;
}

/* sqrtm(a): the principal matrix square root via the real Schur method (dgees) and the upper-triangular
   square-root recurrence, then Z U Z^T. Supports matrices with non-negative real eigenvalues (the principal
   square root is then real and unique, matching scipy); complex eigenvalues (2x2 Schur blocks) or negative real
   eigenvalues are rejected. scipy.linalg.sqrtm default disp=True returns just the matrix. */
static int r_sqrtm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("sqrtm: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("sqrtm: expected a square matrix"); return TSR_EARG; }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);             /* -> Schur form T */
    if (!a) return TSR_ENOMEM;
    double *wr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *wi = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *z = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *u = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    if (!wr || !wi || !z || !u || !tmp) { free(wr); free(wi); free(z); free(u); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (n > 0) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, &sdim, wr, wi, z, (lapack_int)n);
        if (info != 0) { rc = TSR_EARG; fn_set_error("sqrtm: dgees failed"); }
    }
    for (int64_t i = 0; rc == TSR_OK && i < n; i++) {
        if (fabs(wi[i]) > 1e-12 * (1.0 + fabs(wr[i]))) { rc = TSR_EARG; fn_set_error("sqrtm: complex eigenvalues are not yet supported"); }
        else if (a[i * n + i] < 0.0) { rc = TSR_EARG; fn_set_error("sqrtm: the matrix has a negative real eigenvalue (no real square root)"); }
        else u[i * n + i] = sqrt(a[i * n + i]);
    }
    for (int64_t d = 1; rc == TSR_OK && d < n; d++)
        for (int64_t i = 0; i + d < n; i++) {
            const int64_t j = i + d;
            double s = a[i * n + j];
            for (int64_t k = i + 1; k < j; k++) s -= u[i * n + k] * u[k * n + j];
            const double den = u[i * n + i] + u[j * n + j];
            u[i * n + j] = (den != 0.0) ? s / den : 0.0;
        }
    if (rc != TSR_OK) { free(wr); free(wi); free(z); free(u); free(tmp); fn_free_doubles(a, na); return rc; }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { free(wr); free(wi); free(z); free(u); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)                       /* tmp = Z U */
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += z[i * n + k] * u[k * n + j];
            tmp[i * n + j] = acc;
        }
    for (int64_t i = 0; i < n; i++)                       /* out = tmp Z^T */
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += tmp[i * n + k] * z[j * n + k];
            out[i * n + j] = acc;
        }
    free(wr); free(wi); free(z); free(u); free(tmp);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* logm(a): the principal matrix logarithm via the real Schur method (dgees) and the Parlett recurrence
   (f = log) on the triangular factor, then Z F Z^T. Supports matrices with distinct positive real eigenvalues
   (the principal log is then real; the Parlett recurrence needs distinct diagonal entries). scipy.linalg.logm
   default disp=True returns just the matrix. */
static int r_logm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("logm: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("logm: expected a square matrix"); return TSR_EARG; }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);             /* -> Schur form T */
    if (!a) return TSR_ENOMEM;
    double *wr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *wi = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *zz = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *f = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    if (!wr || !wi || !zz || !f || !tmp) { free(wr); free(wi); free(zz); free(f); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (n > 0) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, &sdim, wr, wi, zz, (lapack_int)n);
        if (info != 0) { rc = TSR_EARG; fn_set_error("logm: dgees failed"); }
    }
    for (int64_t i = 0; rc == TSR_OK && i < n; i++) {
        if (fabs(wi[i]) > 1e-12 * (1.0 + fabs(wr[i]))) { rc = TSR_EARG; fn_set_error("logm: complex eigenvalues are not yet supported"); }
        else if (a[i * n + i] <= 0.0) { rc = TSR_EARG; fn_set_error("logm: the matrix has a non-positive real eigenvalue (no real logarithm)"); }
        else f[i * n + i] = log(a[i * n + i]);
    }
    for (int64_t d = 1; rc == TSR_OK && d < n; d++)
        for (int64_t i = 0; i + d < n; i++) {
            const int64_t j = i + d;
            const double denom = a[i * n + i] - a[j * n + j];
            double num = a[i * n + j] * (f[i * n + i] - f[j * n + j]);
            for (int64_t k = i + 1; k < j; k++) num += f[i * n + k] * a[k * n + j] - a[i * n + k] * f[k * n + j];
            if (denom == 0.0) {
                if (num != 0.0) { rc = TSR_EARG; fn_set_error("logm: repeated eigenvalues are not supported (Parlett recurrence)"); }
                f[i * n + j] = 0.0;
            } else {
                f[i * n + j] = num / denom;
            }
        }
    if (rc != TSR_OK) { free(wr); free(wi); free(zz); free(f); free(tmp); fn_free_doubles(a, na); return rc; }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { free(wr); free(wi); free(zz); free(f); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += zz[i * n + k] * f[k * n + j];
            tmp[i * n + j] = acc;
        }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += tmp[i * n + k] * zz[j * n + k];
            out[i * n + j] = acc;
        }
    free(wr); free(wi); free(zz); free(f); free(tmp);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* ldl(a): the LDL^T factorisation of a symmetric matrix via dsytrf, returned as scipy does -- (lu, d, perm) with
   lu[perm] triangular, d block-diagonal, and a = lu @ d @ lu^T. A faithful port of scipy's _ldl_sanitize_ipiv,
   _ldl_get_d_and_l and _ldl_construct_tri_factor (lower, real). perm is returned as float64 (slinalg fixtures
   compare arrays as float64). */
static int r_ldl(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("ldl: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("ldl: expected a square matrix"); return TSR_EARG; }
    if (nargs > 1 && args[1].kind == 4 && args[1].num == 0) { fn_set_error("ldl: lower=False is not yet supported"); return TSR_EARG; }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);             /* -> compact LDL factor after dsytrf */
    if (!a) return TSR_ENOMEM;
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(n > 0 ? n : 1));
    int64_t *swap = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    int *pivs = (int *)calloc((size_t)(n > 0 ? n : 1), sizeof(int));
    int64_t *perm = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    double *d = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    double *lu = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    if (!ipiv || !swap || !pivs || !perm || !d || !lu) { free(ipiv); free(swap); free(pivs); free(perm); free(d); free(lu); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (n > 0) {
        lapack_int info = LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)n, a, (lapack_int)n, ipiv);
        if (info < 0) { rc = TSR_EARG; fn_set_error("ldl: dsytrf reported an illegal value"); }
    }
    for (int64_t i = 0; i < n; i++) { swap[i] = i; perm[i] = i; }
    /* _ldl_sanitize_ipiv (lower: x=1, y=0) */
    for (int64_t ind = 0; rc == TSR_OK && ind < n; ind++) {
        const lapack_int cur = ipiv[ind];
        if (cur > 0) {
            if (cur != (lapack_int)(ind + 1)) swap[ind] = swap[cur - 1];
            pivs[ind] = 1;
        } else if (cur < 0 && ind + 1 < n && cur == ipiv[ind + 1]) {
            if (-cur != (lapack_int)(ind + 2)) swap[ind + 1] = swap[-cur - 1];
            pivs[ind] = 2;
            ind++;                                       /* skip the 2x2 partner */
        } else {
            rc = TSR_EARG; fn_set_error("ldl: invalid pivot array from dsytrf");
        }
    }
    if (rc != TSR_OK) { free(ipiv); free(swap); free(pivs); free(perm); free(d); free(lu); fn_free_doubles(a, na); return rc; }
    /* _ldl_get_d_and_l (lower: x=1, y=0): d = block-diagonal, lu = unit lower-triangular */
    for (int64_t i = 0; i < n; i++) {
        d[i * n + i] = a[i * n + i];
        for (int64_t j = 0; j < i; j++) lu[i * n + j] = a[i * n + j];
        lu[i * n + i] = 1.0;
    }
    int64_t blk_i = 0;
    for (int64_t p = 0; p < n; p++) {
        if (pivs[p] == 0) continue;
        if (pivs[p] == 2) {
            d[(blk_i + 1) * n + blk_i] = a[(blk_i + 1) * n + blk_i];
            d[blk_i * n + (blk_i + 1)] = a[(blk_i + 1) * n + blk_i];
            lu[(blk_i + 1) * n + blk_i] = 0.0;
        }
        blk_i += pivs[p];
    }
    /* _ldl_construct_tri_factor (lower: rs=n-1, re=-1, ri=-1) */
    for (int64_t ind = n - 1; ind >= 0; ind--) {
        const int64_t s_ind = swap[ind];
        if (s_ind != ind) {
            int64_t col_s = ind;
            if (pivs[ind] == 0) col_s -= 1;              /* second row of a 2x2 block */
            for (int64_t c = col_s; c < n; c++) {
                double t = lu[s_ind * n + c]; lu[s_ind * n + c] = lu[ind * n + c]; lu[ind * n + c] = t;
            }
            int64_t t = perm[s_ind]; perm[s_ind] = perm[ind]; perm[ind] = t;
        }
    }
    int64_t sh2[2] = {n, n}, sh1[1] = {n};
    double *lu_out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh2);
    double *d_out = (double *)fn_result_array(&res[1], TSR_F64, 2, sh2);
    double *perm_out = (double *)fn_result_array(&res[2], TSR_F64, 1, sh1);
    if (!lu_out || !d_out || !perm_out) { free(ipiv); free(swap); free(pivs); free(perm); free(d); free(lu); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n * n; i++) { lu_out[i] = lu[i]; d_out[i] = d[i]; }
    for (int64_t i = 0; i < n; i++) perm_out[perm[i]] = (double)i;   /* argsort of a permutation = its inverse */
    free(ipiv); free(swap); free(pivs); free(perm); free(d); free(lu);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* Apply a scalar function f to a matrix via the real Schur method (dgees) and the Parlett recurrence, then
   Z F Z^T. Shared by sinm/cosm/sinhm/coshm/tanhm (scipy.linalg). Requires real, distinct eigenvalues (f is
   defined on the whole real line for these functions, so no domain restriction beyond the Parlett one). */
static int matfun_impl(const tsr_arg *args, int nargs, tsr_result *res, double (*f)(double), const char *name)
{
    (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("%s: input must be a 2-D array", name); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("%s: expected a square matrix", name); return TSR_EARG; }
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *wr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *wi = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *zz = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *fm = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    if (!wr || !wi || !zz || !fm || !tmp) { free(wr); free(wi); free(zz); free(fm); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (n > 0) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, &sdim, wr, wi, zz, (lapack_int)n);
        if (info != 0) { rc = TSR_EARG; fn_set_error("%s: dgees failed", name); }
    }
    for (int64_t i = 0; rc == TSR_OK && i < n; i++) {
        if (fabs(wi[i]) > 1e-12 * (1.0 + fabs(wr[i]))) { rc = TSR_EARG; fn_set_error("%s: complex eigenvalues are not yet supported", name); }
        else fm[i * n + i] = f(a[i * n + i]);
    }
    for (int64_t d = 1; rc == TSR_OK && d < n; d++)
        for (int64_t i = 0; i + d < n; i++) {
            const int64_t j = i + d;
            const double denom = a[i * n + i] - a[j * n + j];
            double num = a[i * n + j] * (fm[i * n + i] - fm[j * n + j]);
            for (int64_t k = i + 1; k < j; k++) num += fm[i * n + k] * a[k * n + j] - a[i * n + k] * fm[k * n + j];
            if (denom == 0.0) {
                if (num != 0.0) { rc = TSR_EARG; fn_set_error("%s: repeated eigenvalues are not supported (Parlett recurrence)", name); }
                fm[i * n + j] = 0.0;
            } else {
                fm[i * n + j] = num / denom;
            }
        }
    if (rc != TSR_OK) { free(wr); free(wi); free(zz); free(fm); free(tmp); fn_free_doubles(a, na); return rc; }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { free(wr); free(wi); free(zz); free(fm); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += zz[i * n + k] * fm[k * n + j];
            tmp[i * n + j] = acc;
        }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += tmp[i * n + k] * zz[j * n + k];
            out[i * n + j] = acc;
        }
    free(wr); free(wi); free(zz); free(fm); free(tmp);
    fn_free_doubles(a, na);
    return TSR_OK;
}
static int r_sinm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, sin, "sinm"); }
static int r_cosm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, cos, "cosm"); }
static int r_sinhm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, sinh, "sinhm"); }
static int r_coshm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, cosh, "coshm"); }
static int r_tanhm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, tanh, "tanhm"); }
static int r_expm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, exp, "expm"); }
static int r_tanm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, tan, "tanm"); }
/* matrix sign function: sign(A) via the Schur-Parlett method with f(x) = sign(x) (scipy.linalg.signm). For a
   real matrix with a real, well-separated spectrum this is the value scipy's funm-based signm returns. */
static double sl_signfun(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }
static int r_signm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return matfun_impl(args, nargs, res, sl_signfun, "signm"); }

/* fractional_matrix_power(a, t): a^t via the real Schur method and the Parlett recurrence with f(x) = x^t
   (scipy.linalg.fractional_matrix_power). Real spectrum with positive eigenvalues (so x^t is real and the
   principal power is unique, matching scipy). */
static int r_fractional_matrix_power(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("fractional_matrix_power: input must be a 2-D array"); return TSR_ETYPE; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("fractional_matrix_power: expected a square matrix"); return TSR_EARG; }
    double t = 1.0;
    if (nargs > 1 && args[1].kind == 1) t = (args[1].flags & 1) ? (double)args[1].ival : args[1].num;
    int64_t na;
    double *a = mat_f64(&args[0], "a", &na);
    if (!a) return TSR_ENOMEM;
    double *wr = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *wi = (double *)malloc(sizeof(double) * (size_t)(n > 0 ? n : 1));
    double *zz = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *fm = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    if (!wr || !wi || !zz || !fm || !tmp) { free(wr); free(wi); free(zz); free(fm); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (n > 0) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, &sdim, wr, wi, zz, (lapack_int)n);
        if (info != 0) { rc = TSR_EARG; fn_set_error("fractional_matrix_power: dgees failed"); }
    }
    for (int64_t i = 0; rc == TSR_OK && i < n; i++) {
        if (fabs(wi[i]) > 1e-12 * (1.0 + fabs(wr[i]))) { rc = TSR_EARG; fn_set_error("fractional_matrix_power: complex eigenvalues are not yet supported"); }
        else if (a[i * n + i] < 0.0) { rc = TSR_EARG; fn_set_error("fractional_matrix_power: negative real eigenvalue (result would be complex)"); }
        else fm[i * n + i] = pow(a[i * n + i], t);
    }
    for (int64_t dd = 1; rc == TSR_OK && dd < n; dd++)
        for (int64_t i = 0; i + dd < n; i++) {
            const int64_t j = i + dd;
            const double denom = a[i * n + i] - a[j * n + j];
            double num = a[i * n + j] * (fm[i * n + i] - fm[j * n + j]);
            for (int64_t k = i + 1; k < j; k++) num += fm[i * n + k] * a[k * n + j] - a[i * n + k] * fm[k * n + j];
            if (denom == 0.0) {
                if (num != 0.0) { rc = TSR_EARG; fn_set_error("fractional_matrix_power: repeated eigenvalues are not supported (Parlett recurrence)"); }
                fm[i * n + j] = 0.0;
            } else {
                fm[i * n + j] = num / denom;
            }
        }
    if (rc != TSR_OK) { free(wr); free(wi); free(zz); free(fm); free(tmp); fn_free_doubles(a, na); return rc; }
    int64_t osh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { free(wr); free(wi); free(zz); free(fm); free(tmp); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += zz[i * n + k] * fm[k * n + j];
            tmp[i * n + j] = acc;
        }
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            double acc = 0.0;
            for (int64_t k = 0; k < n; k++) acc += tmp[i * n + k] * zz[j * n + k];
            out[i * n + j] = acc;
        }
    free(wr); free(wi); free(zz); free(fm); free(tmp);
    fn_free_doubles(a, na);
    return TSR_OK;
}

/* C = X @ Y for n x n row-major matrices; C must not alias X or Y. */
static void mm_nn(const double *X, const double *Y, double *C, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            double s = 0.0;
            for (int64_t k = 0; k < n; k++) s += X[i * n + k] * Y[k * n + j];
            C[i * n + j] = s;
        }
}

/* Matrix exponential of an n x n row-major real matrix via scaling and squaring with the degree-13 Padé
   approximant (Higham, 2005). Writes the result into out (caller-allocated n*n). This is the reusable core the
   Frechet-derivative routines below rely on through the block-enlarge identity. 0 on success, -1 on failure. */
static int sl_expm(const double *A, int64_t n, double *out)
{
    if (n == 0) return 0;
    const size_t nn = (size_t)(n * n);
    static const double b[14] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0, 1187353796428800.0,
        129060195264000.0, 10559470521600.0, 670442572800.0, 33522128640.0,
        1323241920.0, 40840800.0, 960960.0, 16380.0, 182.0, 1.0};
    double *a  = (double *)malloc(sizeof(double) * nn);
    double *A2 = (double *)malloc(sizeof(double) * nn);
    double *A4 = (double *)malloc(sizeof(double) * nn);
    double *A6 = (double *)malloc(sizeof(double) * nn);
    double *U  = (double *)malloc(sizeof(double) * nn);
    double *V  = (double *)malloc(sizeof(double) * nn);
    double *W  = (double *)malloc(sizeof(double) * nn);
    double *P  = (double *)malloc(sizeof(double) * nn);
    double *Q  = (double *)malloc(sizeof(double) * nn);
    if (!a || !A2 || !A4 || !A6 || !U || !V || !W || !P || !Q) {
        free(a); free(A2); free(A4); free(A6); free(U); free(V); free(W); free(P); free(Q); return -1;
    }
    memcpy(a, A, sizeof(double) * nn);
    double norm1 = 0.0;                                      /* 1-norm = max column abs-sum */
    for (int64_t j = 0; j < n; j++) { double c = 0.0; for (int64_t i = 0; i < n; i++) c += fabs(a[i * n + j]); if (c > norm1) norm1 = c; }
    const double theta13 = 5.371920351148152;
    int s = 0;
    if (norm1 > theta13) { s = (int)ceil(log2(norm1 / theta13)); if (s < 0) s = 0; }
    if (s > 0) { const double scale = ldexp(1.0, -s); for (size_t i = 0; i < nn; i++) a[i] *= scale; }
    mm_nn(a, a, A2, n);
    mm_nn(A2, A2, A4, n);
    mm_nn(A2, A4, A6, n);
    /* U = a @ (A6 @ (b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I) */
    for (size_t i = 0; i < nn; i++) W[i] = b[13] * A6[i] + b[11] * A4[i] + b[9] * A2[i];
    mm_nn(A6, W, P, n);
    for (size_t i = 0; i < nn; i++) P[i] += b[7] * A6[i] + b[5] * A4[i] + b[3] * A2[i];
    for (int64_t i = 0; i < n; i++) P[i * n + i] += b[1];
    mm_nn(a, P, U, n);
    /* V = A6 @ (b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I */
    for (size_t i = 0; i < nn; i++) W[i] = b[12] * A6[i] + b[10] * A4[i] + b[8] * A2[i];
    mm_nn(A6, W, V, n);
    for (size_t i = 0; i < nn; i++) V[i] += b[6] * A6[i] + b[4] * A4[i] + b[2] * A2[i];
    for (int64_t i = 0; i < n; i++) V[i * n + i] += b[0];
    for (size_t i = 0; i < nn; i++) { const double u = U[i], v = V[i]; P[i] = u + v; Q[i] = v - u; }
    int rc = sl_dense_solve(Q, P, n, n);                     /* P <- R = (V - U)^{-1} (V + U) */
    if (rc == 0) {
        memcpy(out, P, sizeof(double) * nn);
        for (int k = 0; k < s; k++) { mm_nn(out, out, W, n); memcpy(out, W, sizeof(double) * nn); }
    }
    free(a); free(A2); free(A4); free(A6); free(U); free(V); free(W); free(P); free(Q);
    return rc;
}

/* expm_frechet(A, E): the matrix exponential expm(A) together with its Frechet derivative L(A, E) in the
   direction E, via the block-enlarge identity expm([[A, E], [0, A]]) = [[expm(A), L(A, E)], [0, expm(A)]]
   (scipy.linalg.expm_frechet, default compute_expm=True). Returns (expm, frechet). */
static int r_expm_frechet(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("expm_frechet: A and E must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[1].arr.shape[1] != n) { fn_set_error("expm_frechet: A and E must be square and the same size"); return TSR_EARG; }
    int64_t na, ne;
    double *A = mat_f64(&args[0], "A", &na); if (!A) return TSR_ENOMEM;
    double *E = mat_f64(&args[1], "E", &ne); if (!E) { fn_free_doubles(A, na); return TSR_ENOMEM; }
    const int64_t m = 2 * n;
    double *B = (double *)calloc((size_t)(m * m > 0 ? m * m : 1), sizeof(double));
    double *M = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));
    int rc = TSR_OK;
    if (!B || !M) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) {
            B[i * m + j] = A[i * n + j];
            B[i * m + (n + j)] = E[i * n + j];
            B[(n + i) * m + (n + j)] = A[i * n + j];
        }
        if (sl_expm(B, m, M) != 0) { fn_set_error("expm_frechet: the matrix exponential failed"); rc = TSR_EARG; }
        else {
            int64_t osh[2] = {n, n};
            double *ex = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
            double *fr = (double *)fn_result_array(&res[1], TSR_F64, 2, osh);
            if (!ex || !fr) rc = TSR_ENOMEM;
            else for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) {
                ex[i * n + j] = M[i * m + j];
                fr[i * n + j] = M[i * m + (n + j)];
            }
        }
    }
    free(B); free(M);
    fn_free_doubles(A, na); fn_free_doubles(E, ne);
    return rc;
}

/* expm_cond(A): the relative condition number of the matrix exponential, kappa = ||K|| ||A||_F / ||expm(A)||_F,
   where K is the n^2 x n^2 Kronecker form of the Frechet derivative (its columns are the vectorised L(A, E_ij)
   for the standard-basis directions) and ||K|| is its spectral (induced 2-) norm (scipy.linalg.expm_cond). A
   row/column permutation of K leaves its spectral norm unchanged, so the exact vec ordering is immaterial. */
static int r_expm_cond(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("expm_cond: A must be a square 2-D array"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    int64_t na;
    double *A = mat_f64(&args[0], "A", &na); if (!A) return TSR_ENOMEM;
    const int64_t m = 2 * n, n2 = n * n;
    double *X  = (double *)malloc(sizeof(double) * (size_t)(n2 > 0 ? n2 : 1));
    double *B  = (double *)calloc((size_t)(m * m > 0 ? m * m : 1), sizeof(double));
    double *M  = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));
    double *K  = (double *)malloc(sizeof(double) * (size_t)(n2 * n2 > 0 ? n2 * n2 : 1));
    double *sv = (double *)malloc(sizeof(double) * (size_t)(n2 > 0 ? n2 : 1));
    int rc = TSR_OK;
    if (!X || !B || !M || !K || !sv) rc = TSR_ENOMEM;
    else if (sl_expm(A, n, X) != 0) { fn_set_error("expm_cond: the matrix exponential failed"); rc = TSR_EARG; }
    else {
        for (int64_t i = 0; i < n && rc == TSR_OK; i++)
            for (int64_t j = 0; j < n && rc == TSR_OK; j++) {
                const int64_t p = i * n + j;                 /* column p = direction E_ij */
                memset(B, 0, sizeof(double) * (size_t)(m * m));
                for (int64_t r = 0; r < n; r++) for (int64_t c = 0; c < n; c++) {
                    B[r * m + c] = A[r * n + c];
                    B[(n + r) * m + (n + c)] = A[r * n + c];
                }
                B[i * m + (n + j)] = 1.0;
                if (sl_expm(B, m, M) != 0) { fn_set_error("expm_cond: the matrix exponential failed"); rc = TSR_EARG; break; }
                for (int64_t r = 0; r < n; r++) for (int64_t c = 0; c < n; c++)
                    K[(r * n + c) * n2 + p] = M[r * m + (n + c)];
            }
        if (rc == TSR_OK) {
            double a_fro = 0.0, x_fro = 0.0, k_norm = 0.0;   /* ||A||_F, ||expm(A)||_F, ||K||_2 */
            for (int64_t i = 0; i < n2; i++) { a_fro += A[i] * A[i]; x_fro += X[i] * X[i]; }
            a_fro = sqrt(a_fro); x_fro = sqrt(x_fro);
            if (n2 > 0) { lapack_int info = svd_values(K, n2, n2, sv); if (info != 0) { fn_set_error("expm_cond: the SVD of the Kronecker form failed"); rc = TSR_EARG; } else k_norm = sv[0]; }
            if (rc == TSR_OK) fn_result_num(&res[0], x_fro > 0.0 ? (k_norm * a_fro) / x_fro : FN_INF);
        }
    }
    free(X); free(B); free(M); free(K); free(sv);
    fn_free_doubles(A, na);
    return rc;
}

/* Read an array argument as a flat row-major interleaved-complex buffer (2 doubles per element, re then im) of
   logical length *n_out. complex128 inputs are read directly (honouring strides and offset); any real dtype is
   read through mat_f64 with zero imaginary parts. Caller frees with free(). NULL on error. */
static double *cplx_read(const tsr_arg *a, const char *what, int64_t *n_out)
{
    if (a->kind != 3) { fn_set_error("%s must be an array", what); return NULL; }
    const tsr_array *ar = &a->arr;
    int64_t sz = 1;
    for (int d = 0; d < ar->ndim; d++) sz *= ar->shape[d];
    if (sz < 0) sz = 0;
    double *buf = (double *)malloc(sizeof(double) * (size_t)(2 * (sz > 0 ? sz : 1)));
    if (!buf) { fn_set_error("%s: out of memory", what); return NULL; }
    if (ar->dtype != TSR_C128) {                             /* real input: cast via mat_f64, zero imaginary */
        int64_t m; double *re = mat_f64(a, what, &m);
        if (!re) { free(buf); return NULL; }
        for (int64_t k = 0; k < sz; k++) { buf[2 * k] = re[k]; buf[2 * k + 1] = 0.0; }
        fn_free_doubles(re, m);
        *n_out = sz; return buf;
    }
    const char *base = (const char *)ar->data + ar->offset;
    int64_t idx[32] = {0};
    for (int64_t k = 0; k < sz; k++) {
        const char *p = base;
        for (int d = 0; d < ar->ndim; d++) p += idx[d] * ar->strides[d];
        buf[2 * k] = ((const double *)p)[0];
        buf[2 * k + 1] = ((const double *)p)[1];
        for (int d = (int)ar->ndim - 1; d >= 0; d--) { if (++idx[d] < ar->shape[d]) break; idx[d] = 0; }
    }
    *n_out = sz; return buf;
}

/* rsf2csf(T, Z): convert a real Schur form (T real quasi-triangular, Z real orthogonal, A = Z T Z^T) to the
   complex Schur form (T upper-triangular, Z unitary) by the Givens rotations that zero each 2x2 block's
   subdiagonal (scipy.linalg.rsf2csf). Deterministic given the inputs; outputs are complex128. */
static int r_rsf2csf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("rsf2csf: T and Z must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[1].arr.shape[1] != n) { fn_set_error("rsf2csf: T and Z must be square and the same size"); return TSR_EARG; }
    int64_t nt, nz;
    double *Tr = mat_f64(&args[0], "T", &nt); if (!Tr) return TSR_ENOMEM;
    double *Zr = mat_f64(&args[1], "Z", &nz); if (!Zr) { fn_free_doubles(Tr, nt); return TSR_ENOMEM; }
    const size_t cc = (size_t)(2 * (n * n > 0 ? n * n : 1));
    double *T = (double *)malloc(sizeof(double) * cc);       /* interleaved complex */
    double *Z = (double *)malloc(sizeof(double) * cc);
    int rc = TSR_OK;
    if (!T || !Z) rc = TSR_ENOMEM;
    else {
        for (int64_t k = 0; k < n * n; k++) { T[2 * k] = Tr[k]; T[2 * k + 1] = 0.0; Z[2 * k] = Zr[k]; Z[2 * k + 1] = 0.0; }
        const double eps = 2.220446049250313e-16;
        for (int64_t m = n - 1; m >= 1; m--) {
            const int64_t sub = m * n + (m - 1);
            const double atsub = hypot(T[2 * sub], T[2 * sub + 1]);
            const double amm1 = hypot(T[2 * ((m - 1) * n + (m - 1))], T[2 * ((m - 1) * n + (m - 1)) + 1]);
            const double amm = hypot(T[2 * (m * n + m)], T[2 * (m * n + m) + 1]);
            if (atsub > eps * (amm1 + amm)) {
                /* 2x2 block [[p, q], [r, s]]; its conjugate-pair eigenvalues are (tr +- i*sqrt(-disc))/2, and
                   LAPACK/scipy order the positive-imaginary one first, so mu = eigvals[0] - s. */
                const double p = T[2 * ((m - 1) * n + (m - 1))], q = T[2 * ((m - 1) * n + m)];
                const double r_ = T[2 * sub], s = T[2 * (m * n + m)];
                const double tr = p + s, det = p * s - q * r_, disc = tr * tr - 4.0 * det;
                double mu_re, mu_im;
                if (disc < 0.0) { mu_re = tr * 0.5 - s; mu_im = sqrt(-disc) * 0.5; }
                else { mu_re = (tr + sqrt(disc)) * 0.5 - s; mu_im = 0.0; }
                const double rn = hypot(hypot(mu_re, mu_im), r_);   /* norm([mu, T[m,m-1]]) */
                const double cre = mu_re / rn, cim = mu_im / rn;    /* c = mu / rn (complex) */
                const double sr = r_ / rn;                          /* s = T[m,m-1] / rn (real) */
                /* G = [[conj(c), s], [-s, c]]. Left: rows (m-1, m), columns (m-1 .. n-1). */
                for (int64_t col = m - 1; col < n; col++) {
                    const int64_t a0 = (m - 1) * n + col, a1 = m * n + col;
                    const double xr = T[2 * a0], xi = T[2 * a0 + 1], yr = T[2 * a1], yi = T[2 * a1 + 1];
                    /* row0 = conj(c)*x + s*y ; row1 = -s*x + c*y */
                    T[2 * a0]     = cre * xr + cim * xi + sr * yr;
                    T[2 * a0 + 1] = cre * xi - cim * xr + sr * yi;
                    T[2 * a1]     = -sr * xr + cre * yr - cim * yi;
                    T[2 * a1 + 1] = -sr * xi + cre * yi + cim * yr;
                }
                /* Right: T[:m+1, (m-1, m)] @ G^H, G^H = [[c, -s], [s, conj(c)]]. */
                for (int64_t row = 0; row <= m; row++) {
                    const int64_t a0 = row * n + (m - 1), a1 = row * n + m;
                    const double ur = T[2 * a0], ui = T[2 * a0 + 1], vr = T[2 * a1], vi = T[2 * a1 + 1];
                    /* col0 = u*c + v*s ; col1 = -u*s + v*conj(c) */
                    T[2 * a0]     = ur * cre - ui * cim + vr * sr;
                    T[2 * a0 + 1] = ur * cim + ui * cre + vi * sr;
                    T[2 * a1]     = -ur * sr + vr * cre + vi * cim;
                    T[2 * a1 + 1] = -ui * sr + vi * cre - vr * cim;
                }
                /* Z[:, (m-1, m)] @ G^H, all rows. */
                for (int64_t row = 0; row < n; row++) {
                    const int64_t a0 = row * n + (m - 1), a1 = row * n + m;
                    const double ur = Z[2 * a0], ui = Z[2 * a0 + 1], vr = Z[2 * a1], vi = Z[2 * a1 + 1];
                    Z[2 * a0]     = ur * cre - ui * cim + vr * sr;
                    Z[2 * a0 + 1] = ur * cim + ui * cre + vi * sr;
                    Z[2 * a1]     = -ur * sr + vr * cre + vi * cim;
                    Z[2 * a1 + 1] = -ui * sr + vi * cre - vr * cim;
                }
            }
            T[2 * sub] = 0.0; T[2 * sub + 1] = 0.0;
        }
        int64_t osh[2] = {n, n};
        double *To = (double *)fn_result_array(&res[0], TSR_C128, 2, osh);
        double *Zo = (double *)fn_result_array(&res[1], TSR_C128, 2, osh);
        if (!To || !Zo) rc = TSR_ENOMEM;
        else { memcpy(To, T, sizeof(double) * cc); memcpy(Zo, Z, sizeof(double) * cc); }
    }
    free(T); free(Z);
    fn_free_doubles(Tr, nt); fn_free_doubles(Zr, nz);
    return rc;
}

/* cdf2rdf(w, v): convert complex eigenvalues w and eigenvectors v (as returned by eig) into the real block
   diagonal form wr and the real eigenvectors vr with vr @ wr @ inv(vr) == original (scipy.linalg.cdf2rdf).
   Each conjugate pair (positions j, k=j+1) becomes a 2x2 block [[a, b], [-b, a]]; vr = Re(v @ u) with u the
   block-mixing matrix. Deterministic given the inputs. Returns (wr, vr), both real. */
static int r_cdf2rdf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("cdf2rdf: w must be 1-D and v 2-D"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    if (args[1].arr.shape[0] != n || args[1].arr.shape[1] != n) { fn_set_error("cdf2rdf: v must be n x n with n = len(w)"); return TSR_EARG; }
    int64_t nw, nv;
    double *w = cplx_read(&args[0], "w", &nw); if (!w) return TSR_ENOMEM;
    double *v = cplx_read(&args[1], "v", &nv); if (!v) { free(w); return TSR_ENOMEM; }
    int rc = TSR_OK;
    /* complex positions, in order; conjugate pairs are consecutive (eig returns +imag first). */
    int64_t *cidx = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    double *u = (double *)calloc((size_t)(2 * (n * n > 0 ? n * n : 1)), sizeof(double));  /* interleaved complex */
    if (!cidx || !u) rc = TSR_ENOMEM;
    else {
        int64_t nc = 0;
        for (int64_t i = 0; i < n; i++) if (w[2 * i + 1] != 0.0) cidx[nc++] = i;
        if (nc % 2 != 0) { fn_set_error("cdf2rdf: expected complex-conjugate pairs of eigenvalues"); rc = TSR_EARG; }
        else {
            int64_t osh2[2] = {n, n};
            double *wr = (double *)fn_result_array(&res[0], TSR_F64, 2, osh2);
            double *vr = (double *)fn_result_array(&res[1], TSR_F64, 2, osh2);
            if (!wr || !vr) rc = TSR_ENOMEM;
            else {
                for (int64_t i = 0; i < n * n; i++) wr[i] = 0.0;
                for (int64_t i = 0; i < n; i++) { wr[i * n + i] = w[2 * i]; u[2 * (i * n + i)] = 1.0; }  /* diag */
                for (int64_t pr = 0; pr < nc; pr += 2) {
                    const int64_t j = cidx[pr], k = cidx[pr + 1];
                    wr[j * n + k] = w[2 * j + 1];               /* +b */
                    wr[k * n + j] = w[2 * k + 1];               /* -b */
                    u[2 * (j * n + j)]     = 0.0; u[2 * (j * n + j) + 1] = 0.5;   /* u[j,j] = 0.5i (replaces 1) */
                    u[2 * (j * n + k)]     = 0.5; u[2 * (j * n + k) + 1] = 0.0;   /* u[j,k] = 0.5  */
                    u[2 * (k * n + j)]     = 0.0; u[2 * (k * n + j) + 1] = -0.5;  /* u[k,j] = -0.5i */
                    u[2 * (k * n + k)]     = 0.5; u[2 * (k * n + k) + 1] = 0.0;   /* u[k,k] = 0.5 (replaces 1) */
                }
                /* vr = Re(v @ u) */
                for (int64_t i = 0; i < n; i++)
                    for (int64_t c = 0; c < n; c++) {
                        double acc = 0.0;
                        for (int64_t l = 0; l < n; l++) {
                            const double vr_ = v[2 * (i * n + l)], vi_ = v[2 * (i * n + l) + 1];
                            const double ur_ = u[2 * (l * n + c)], ui_ = u[2 * (l * n + c) + 1];
                            acc += vr_ * ur_ - vi_ * ui_;       /* real part of v*u */
                        }
                        vr[i * n + c] = acc;
                    }
            }
        }
    }
    free(cidx); free(u); free(w); free(v);
    return rc;
}

/* Orthonormal basis for the column space of an m x n row-major matrix A (as scipy.linalg.orth): the leading
   left singular vectors whose singular value exceeds max(s) * eps * max(m, n). Writes Q (m x rank, row-major)
   into Qout (buffer >= m*min(m,n)) and the rank into *rank_out. 0 on success, -1 on failure. */
static int sl_orth(const double *A, int64_t m, int64_t n, double *Qout, int64_t *rank_out)
{
    const int64_t k = m < n ? m : n;
    *rank_out = 0;
    if (k == 0) return 0;
    double *a = (double *)malloc(sizeof(double) * (size_t)(m * n));
    double *u = (double *)malloc(sizeof(double) * (size_t)(m * k));
    double *s = (double *)malloc(sizeof(double) * (size_t)k);
    double *vt = (double *)malloc(sizeof(double) * (size_t)(k * n));
    int rc = -1;
    if (a && u && s && vt) {
        memcpy(a, A, sizeof(double) * (size_t)(m * n));
        lapack_int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S', (lapack_int)m, (lapack_int)n, a, (lapack_int)n, s, u, (lapack_int)k, vt, (lapack_int)n);
        if (info == 0) {
            const double tol = s[0] * (double)(m > n ? m : n) * DBL_EPSILON;
            int64_t rank = 0;
            for (int64_t i = 0; i < k; i++) if (s[i] > tol) rank++;
            for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < rank; j++) Qout[i * rank + j] = u[i * k + j];
            *rank_out = rank;
            rc = 0;
        }
    }
    free(a); free(u); free(s); free(vt);
    return rc;
}

/* subspace_angles(A, B): the principal angles between the column spaces of A and B (scipy.linalg.subspace_angles).
   QA = orth(A), QB = orth(B); sigma = svdvals(QA^T QB) are the cosines; the sines come from the SVD of the
   residual of the smaller basis projected onto the larger. The result is invariant to the (sign-free) choice of
   orthonormal bases, so it is deterministic. Returns the 1-D angle array. */
static int r_subspace_angles(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("subspace_angles: A and B must be 2-D arrays"); return TSR_EARG; }
    const int64_t nrow = args[0].arr.shape[0], nA = args[0].arr.shape[1], nB = args[1].arr.shape[1];
    if (args[1].arr.shape[0] != nrow) { fn_set_error("subspace_angles: A and B must have the same number of rows"); return TSR_EARG; }
    int64_t naA, naB;
    double *A = mat_f64(&args[0], "A", &naA); if (!A) return TSR_ENOMEM;
    double *B = mat_f64(&args[1], "B", &naB); if (!B) { fn_free_doubles(A, naA); return TSR_ENOMEM; }
    const int64_t kA = nrow < nA ? nrow : nA, kB = nrow < nB ? nrow : nB;
    double *QA = (double *)malloc(sizeof(double) * (size_t)(nrow * (kA ? kA : 1)));
    double *QB = (double *)malloc(sizeof(double) * (size_t)(nrow * (kB ? kB : 1)));
    int rc = TSR_OK;
    int64_t rA = 0, rB = 0;
    if (!QA || !QB) rc = TSR_ENOMEM;
    else if (sl_orth(A, nrow, nA, QA, &rA) != 0 || sl_orth(B, nrow, nB, QB, &rB) != 0) { fn_set_error("subspace_angles: SVD did not converge"); rc = TSR_EARG; }
    else {
        const int64_t p = rA < rB ? rA : rB;                 /* number of angles */
        double *M = (double *)malloc(sizeof(double) * (size_t)((rA * rB > 0 ? rA * rB : 1)));
        double *Mc = (double *)malloc(sizeof(double) * (size_t)((rA * rB > 0 ? rA * rB : 1)));
        double *sigma = (double *)malloc(sizeof(double) * (size_t)(p ? p : 1));
        const int64_t bcols = rA >= rB ? rB : rA;
        double *Bmat = (double *)malloc(sizeof(double) * (size_t)((nrow * bcols > 0 ? nrow * bcols : 1)));
        double *svb = (double *)malloc(sizeof(double) * (size_t)(p ? p : 1));
        if (!M || !Mc || !sigma || !Bmat || !svb) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < rA; i++) for (int64_t j = 0; j < rB; j++) {
                double s = 0.0; for (int64_t l = 0; l < nrow; l++) s += QA[l * rA + i] * QB[l * rB + j];
                M[i * rB + j] = s;
            }
            if (rA >= rB) {                                  /* Bmat = QB - QA @ M  (nrow x rB) */
                for (int64_t i = 0; i < nrow; i++) for (int64_t j = 0; j < rB; j++) {
                    double s = 0.0; for (int64_t l = 0; l < rA; l++) s += QA[i * rA + l] * M[l * rB + j];
                    Bmat[i * rB + j] = QB[i * rB + j] - s;
                }
            } else {                                         /* Bmat = QA - QB @ M^T  (nrow x rA) */
                for (int64_t i = 0; i < nrow; i++) for (int64_t j = 0; j < rA; j++) {
                    double s = 0.0; for (int64_t l = 0; l < rB; l++) s += QB[i * rB + l] * M[j * rB + l];
                    Bmat[i * rA + j] = QA[i * rA + j] - s;
                }
            }
            if (p > 0) {
                memcpy(Mc, M, sizeof(double) * (size_t)(rA * rB));
                lapack_int i1 = svd_values(Mc, rA, rB, sigma);
                lapack_int i2 = svd_values(Bmat, nrow, bcols, svb);
                if (i1 != 0 || i2 != 0) { fn_set_error("subspace_angles: SVD did not converge"); rc = TSR_EARG; }
            }
            if (rc == TSR_OK) {
                int64_t osh[1] = {p};
                double *th = (double *)fn_result_array(&res[0], TSR_F64, 1, osh);
                if (!th) rc = TSR_ENOMEM;
                else for (int64_t i = 0; i < p; i++) {
                    const double sg = sigma[i];
                    if (sg * sg >= 0.5) { double x = svb[i]; x = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); th[i] = asin(x); }
                    else { double x = sigma[p - 1 - i]; x = x < -1.0 ? -1.0 : (x > 1.0 ? 1.0 : x); th[i] = acos(x); }
                }
            }
        }
        free(M); free(Mc); free(sigma); free(Bmat); free(svb);
    }
    free(QA); free(QB);
    fn_free_doubles(A, naA); fn_free_doubles(B, naB);
    return rc;
}

/* In-place inverse of an n x n row-major matrix via dgetrf + dgetri. 0 on success, -1 on singular/failure. */
static int sl_inv(double *A, int64_t n)
{
    if (n == 0) return 0;
    lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)n);
    if (!ipiv) return -1;
    lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)n, A, (lapack_int)n, ipiv);
    if (info == 0) info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, (lapack_int)n, A, (lapack_int)n, ipiv);
    free(ipiv);
    return info == 0 ? 0 : -1;
}

/* Stable-eigenvalue selectors for the ordered real Schur form: continuous (open left half-plane) and discrete
   (open unit disc). dgees puts the selected eigenvalues' Schur vectors in the leading columns. */
static lapack_logical care_select(const double *wr, const double *wi) { (void)wi; return *wr < 0.0; }
static lapack_logical dare_select(const double *wr, const double *wi) { return (*wr * *wr + *wi * *wi) < 1.0; }

/* Given a 2m x 2m real matrix whose `select`-ed invariant subspace has dimension m, compute the Riccati
   solution X = U2 U1^{-1} (symmetrised), where [U1; U2] are the first m ordered Schur vectors. X depends only
   on the subspace, not the basis, so it is unique. Returns 0, or -1 on failure. Overwrites Hmat. */
static int sl_riccati_from(double *Hmat, int64_t m, LAPACK_D_SELECT2 select, double *out)
{
    const int64_t N = 2 * m;
    if (m == 0) return 0;
    double *wr = (double *)malloc(sizeof(double) * (size_t)N);
    double *wi = (double *)malloc(sizeof(double) * (size_t)N);
    double *vs = (double *)malloc(sizeof(double) * (size_t)(N * N));
    double *U1 = (double *)malloc(sizeof(double) * (size_t)(m * m));
    double *U2 = (double *)malloc(sizeof(double) * (size_t)(m * m));
    int rc = -1;
    if (wr && wi && vs && U1 && U2) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgees(LAPACK_ROW_MAJOR, 'V', 'S', select, (lapack_int)N, Hmat, (lapack_int)N, &sdim, wr, wi, vs, (lapack_int)N);
        if (info == 0 && sdim == (lapack_int)m) {
            for (int64_t i = 0; i < m; i++)
                for (int64_t j = 0; j < m; j++) { U1[i * m + j] = vs[i * N + j]; U2[i * m + j] = vs[(m + i) * N + j]; }
            if (sl_inv(U1, m) == 0) {                        /* X = U2 @ inv(U1), then symmetrise */
                for (int64_t i = 0; i < m; i++)
                    for (int64_t j = 0; j < m; j++) { double s = 0.0; for (int64_t k = 0; k < m; k++) s += U2[i * m + k] * U1[k * m + j]; out[i * m + j] = s; }
                for (int64_t i = 0; i < m; i++)
                    for (int64_t j = i + 1; j < m; j++) { const double a = 0.5 * (out[i * m + j] + out[j * m + i]); out[i * m + j] = a; out[j * m + i] = a; }
                rc = 0;
            }
        }
    }
    free(wr); free(wi); free(vs); free(U1); free(U2);
    return rc;
}

/* B @ inv(R) @ B^T for B (m x n) and R (n x n), into G (m x m). Returns 0, or -1 if R is singular. */
static int sl_bRinvBt(const double *B, const double *R, int64_t m, int64_t n, double *G)
{
    double *Rinv = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *BR = (double *)malloc(sizeof(double) * (size_t)(m * n > 0 ? m * n : 1));
    int rc = -1;
    if (Rinv && BR) {
        memcpy(Rinv, R, sizeof(double) * (size_t)(n * n));
        if (sl_inv(Rinv, n) == 0) {
            for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) { double s = 0.0; for (int64_t k = 0; k < n; k++) s += B[i * n + k] * Rinv[k * n + j]; BR[i * n + j] = s; }
            for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < m; j++) { double s = 0.0; for (int64_t k = 0; k < n; k++) s += BR[i * n + k] * B[j * n + k]; G[i * m + j] = s; }
            rc = 0;
        }
    }
    free(Rinv); free(BR);
    return rc;
}

/* solve_continuous_are(a, b, q, r): the unique symmetric stabilising solution X of the continuous-time
   algebraic Riccati equation a^T X + X a - X b r^{-1} b^T X + q = 0, via the stable invariant subspace of the
   Hamiltonian H = [[a, -b r^{-1} b^T], [-q, -a^T]] (scipy.linalg.solve_continuous_are, real standard case). */
static int r_solve_continuous_are(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    for (int i = 0; i < 4; i++) if (args[i].kind != 3 || args[i].arr.ndim != 2) { fn_set_error("solve_continuous_are: a, b, q, r must be 2-D arrays"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[1].arr.shape[1];
    if (args[0].arr.shape[1] != m || args[1].arr.shape[0] != m || args[2].arr.shape[0] != m || args[2].arr.shape[1] != m || args[3].arr.shape[0] != n || args[3].arr.shape[1] != n) { fn_set_error("solve_continuous_are: incompatible shapes"); return TSR_EARG; }
    int64_t na, nb, nq, nr;
    double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "b", &nb); if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *q = mat_f64(&args[2], "q", &nq); if (!q) { fn_free_doubles(a, na); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    double *r = mat_f64(&args[3], "r", &nr); if (!r) { fn_free_doubles(a, na); fn_free_doubles(b, nb); fn_free_doubles(q, nq); return TSR_ENOMEM; }
    const int64_t N = 2 * m;
    double *G = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));
    double *H = (double *)calloc((size_t)(N * N > 0 ? N * N : 1), sizeof(double));
    int rc = TSR_OK;
    if (!G || !H) rc = TSR_ENOMEM;
    else if (sl_bRinvBt(b, r, m, n, G) != 0) { fn_set_error("solve_continuous_are: r is singular"); rc = TSR_EARG; }
    else {
        for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < m; j++) {
            H[i * N + j] = a[i * m + j];
            H[i * N + (m + j)] = -G[i * m + j];
            H[(m + i) * N + j] = -q[i * m + j];
            H[(m + i) * N + (m + j)] = -a[j * m + i];        /* -a^T */
        }
        int64_t osh[2] = {m, m};
        double *x = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
        if (!x) rc = TSR_ENOMEM;
        else if (sl_riccati_from(H, m, care_select, x) != 0) { fn_set_error("solve_continuous_are: failed to find a finite stabilising solution"); rc = TSR_EARG; }
    }
    free(G); free(H);
    fn_free_doubles(a, na); fn_free_doubles(b, nb); fn_free_doubles(q, nq); fn_free_doubles(r, nr);
    return rc;
}

/* solve_discrete_are(a, b, q, r): the unique symmetric stabilising solution X of the discrete-time algebraic
   Riccati equation a^T X a - X - (a^T X b)(r + b^T X b)^{-1}(b^T X a) + q = 0, via the invariant subspace inside
   the unit disc of the symplectic matrix Z = [[a + G a^{-T} q, -G a^{-T}], [-a^{-T} q, a^{-T}]], G = b r^{-1} b^T
   (scipy.linalg.solve_discrete_are, real standard case with a invertible). */
static int r_solve_discrete_are(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    for (int i = 0; i < 4; i++) if (args[i].kind != 3 || args[i].arr.ndim != 2) { fn_set_error("solve_discrete_are: a, b, q, r must be 2-D arrays"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], n = args[1].arr.shape[1];
    if (args[0].arr.shape[1] != m || args[1].arr.shape[0] != m || args[2].arr.shape[0] != m || args[2].arr.shape[1] != m || args[3].arr.shape[0] != n || args[3].arr.shape[1] != n) { fn_set_error("solve_discrete_are: incompatible shapes"); return TSR_EARG; }
    int64_t na, nb, nq, nr;
    double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "b", &nb); if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *q = mat_f64(&args[2], "q", &nq); if (!q) { fn_free_doubles(a, na); fn_free_doubles(b, nb); return TSR_ENOMEM; }
    double *r = mat_f64(&args[3], "r", &nr); if (!r) { fn_free_doubles(a, na); fn_free_doubles(b, nb); fn_free_doubles(q, nq); return TSR_ENOMEM; }
    const int64_t N = 2 * m;
    double *G = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));
    double *AmT = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));   /* a^{-T} */
    double *GA = (double *)malloc(sizeof(double) * (size_t)(m * m > 0 ? m * m : 1));    /* G a^{-T} */
    double *Z = (double *)calloc((size_t)(N * N > 0 ? N * N : 1), sizeof(double));
    int rc = TSR_OK;
    if (!G || !AmT || !GA || !Z) rc = TSR_ENOMEM;
    else if (sl_bRinvBt(b, r, m, n, G) != 0) { fn_set_error("solve_discrete_are: r is singular"); rc = TSR_EARG; }
    else {
        memcpy(AmT, a, sizeof(double) * (size_t)(m * m));
        if (sl_inv(AmT, m) != 0) { fn_set_error("solve_discrete_are: a is singular"); rc = TSR_EARG; }
        else {
            /* transpose AmT in place: currently holds inv(a); we need inv(a)^T */
            for (int64_t i = 0; i < m; i++) for (int64_t j = i + 1; j < m; j++) { const double t = AmT[i * m + j]; AmT[i * m + j] = AmT[j * m + i]; AmT[j * m + i] = t; }
            for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < m; j++) { double s = 0.0; for (int64_t k = 0; k < m; k++) s += G[i * m + k] * AmT[k * m + j]; GA[i * m + j] = s; }  /* G a^{-T} */
            for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < m; j++) {
                double gaq = 0.0, amtq = 0.0;
                for (int64_t k = 0; k < m; k++) { gaq += GA[i * m + k] * q[k * m + j]; amtq += AmT[i * m + k] * q[k * m + j]; }
                Z[i * N + j] = a[i * m + j] + gaq;           /* a + G a^{-T} q */
                Z[i * N + (m + j)] = -GA[i * m + j];         /* -G a^{-T} */
                Z[(m + i) * N + j] = -amtq;                  /* -a^{-T} q */
                Z[(m + i) * N + (m + j)] = AmT[i * m + j];   /* a^{-T} */
            }
            int64_t osh[2] = {m, m};
            double *x = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
            if (!x) rc = TSR_ENOMEM;
            else if (sl_riccati_from(Z, m, dare_select, x) != 0) { fn_set_error("solve_discrete_are: failed to find a finite stabilising solution"); rc = TSR_EARG; }
        }
    }
    free(G); free(AmT); free(GA); free(Z);
    fn_free_doubles(a, na); fn_free_doubles(b, nb); fn_free_doubles(q, nq); fn_free_doubles(r, nr);
    return rc;
}

/* qr_multiply(a, c, mode='right'): the product of c with the economic orthogonal factor Q of a (formed via
   dgeqrf + dorgqr with LAPACK's standard, non-canonicalised signs, matching scipy.linalg.qr_multiply).
   mode='left' returns (Q @ c, R); mode='right' returns (c @ Q, R), with R the economic upper-triangular factor. */
static int r_qr_multiply(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("qr_multiply: a and c must be 2-D arrays"); return TSR_EARG; }
    int left = 0;
    if (nargs > 2 && args[2].kind == 2 && args[2].str) {
        if (!strcmp(args[2].str, "left")) left = 1;
        else if (strcmp(args[2].str, "right") != 0) { fn_set_error("qr_multiply: mode must be 'left' or 'right'"); return TSR_EARG; }
    }
    const int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], k = m < n ? m : n;
    const int64_t cr = args[1].arr.shape[0], cc = args[1].arr.shape[1];
    int64_t na, nc;
    double *a = mat_f64(&args[0], "a", &na); if (!a) return TSR_ENOMEM;
    double *c = mat_f64(&args[1], "c", &nc); if (!c) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *tau = (double *)malloc(sizeof(double) * (size_t)(k > 0 ? k : 1));
    double *q = (double *)malloc(sizeof(double) * (size_t)(m * (k > 0 ? k : 1)));
    int rc = TSR_OK;
    if (!tau || !q) rc = TSR_ENOMEM;
    else if (k > 0) {
        lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, a, (lapack_int)n, tau);
        if (info != 0) { rc = TSR_EARG; fn_set_error("qr_multiply: dgeqrf failed"); }
        else {
            for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < k; j++) q[i * k + j] = j < n ? a[i * n + j] : 0.0;
            lapack_int info2 = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)k, (lapack_int)k, q, (lapack_int)k, tau);
            if (info2 != 0) { rc = TSR_EARG; fn_set_error("qr_multiply: dorgqr failed"); }
        }
    }
    if (rc == TSR_OK) {
        if (left && cr != k) { fn_set_error("qr_multiply: shapes incompatible for Q @ c"); rc = TSR_EARG; }
        else if (!left && cc != m) { fn_set_error("qr_multiply: shapes incompatible for c @ Q"); rc = TSR_EARG; }
    }
    if (rc == TSR_OK) {
        const int64_t outr = left ? m : cr, outc = left ? cc : k;
        int64_t osh[2] = {outr, outc}, rsh[2] = {k, n};
        double *cq = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
        double *rout = (double *)fn_result_array(&res[1], TSR_F64, 2, rsh);
        if (!cq || !rout) rc = TSR_ENOMEM;
        else {
            if (left) for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < cc; j++) { double s = 0.0; for (int64_t l = 0; l < k; l++) s += q[i * k + l] * c[l * cc + j]; cq[i * cc + j] = s; }
            else for (int64_t i = 0; i < cr; i++) for (int64_t j = 0; j < k; j++) { double s = 0.0; for (int64_t l = 0; l < m; l++) s += c[i * m + l] * q[l * k + j]; cq[i * k + j] = s; }
            for (int64_t i = 0; i < k; i++) for (int64_t j = 0; j < n; j++) rout[i * n + j] = j >= i ? a[i * n + j] : 0.0;
        }
    }
    free(tau); free(q);
    fn_free_doubles(a, na); fn_free_doubles(c, nc);
    return rc;
}

/* Economic QR of an m x n row-major matrix, canonicalised exactly as the parity fixtures' qr_canon (R's
   diagonal made non-negative, the matching Q column flipped), written into res[0]=Q (m x k) and res[1]=R
   (k x n), k = min(m, n). Overwrites A. Returns TSR_OK or an error code. */
static int sl_qr_canon_result(double *A, int64_t m, int64_t n, tsr_result *res)
{
    const int64_t k = m < n ? m : n;
    int64_t qsh[2] = {m, k}, rsh[2] = {k, n};
    double *q = (double *)fn_result_array(&res[0], TSR_F64, 2, qsh);
    double *r = (double *)fn_result_array(&res[1], TSR_F64, 2, rsh);
    double *tau = (double *)malloc(sizeof(double) * (size_t)(k > 0 ? k : 1));
    if (!q || !r || !tau) { free(tau); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (k > 0) {
        lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)n, A, (lapack_int)n, tau);
        if (info != 0) { rc = TSR_EARG; fn_set_error("qr update: dgeqrf failed"); }
        else {
            for (int64_t i = 0; i < k; i++) for (int64_t j = 0; j < n; j++) r[i * n + j] = j >= i ? A[i * n + j] : 0.0;
            for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < k; j++) q[i * k + j] = j < n ? A[i * n + j] : 0.0;
            lapack_int info2 = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (lapack_int)m, (lapack_int)k, (lapack_int)k, q, (lapack_int)k, tau);
            if (info2 != 0) { rc = TSR_EARG; fn_set_error("qr update: dorgqr failed"); }
            else for (int64_t i = 0; i < k; i++)
                if (r[i * n + i] < 0.0) {
                    for (int64_t j = 0; j < n; j++) r[i * n + j] = -r[i * n + j];
                    for (int64_t row = 0; row < m; row++) q[row * k + i] = -q[row * k + i];
                }
        }
    }
    free(tau);
    return rc;
}

/* Reconstruct A = Q R (Q m x kk, R kk x n) into a freshly allocated m x n buffer. NULL on OOM. */
static double *sl_qr_reconstruct(const double *Q, const double *R, int64_t m, int64_t kk, int64_t n)
{
    double *A = (double *)malloc(sizeof(double) * (size_t)(m * n > 0 ? m * n : 1));
    if (!A) return NULL;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) { double s = 0.0; for (int64_t l = 0; l < kk; l++) s += Q[i * kk + l] * R[l * n + j]; A[i * n + j] = s; }
    return A;
}

static int sl_arg_int(const tsr_arg *a, int64_t *out)
{
    if (a->kind == 1) { *out = (a->flags & 1) ? a->ival : (int64_t)a->num; return 0; }
    return -1;
}

/* qr_update(Q, R, u, v): the economic QR of (Q R + u v^T), canonicalised (scipy.linalg.qr_update). */
static int r_qr_update(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2 || args[2].kind != 3 || args[3].kind != 3) { fn_set_error("qr_update: Q, R must be 2-D and u, v arrays"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], kk = args[0].arr.shape[1], n = args[1].arr.shape[1];
    int64_t nq, nr, nu, nv;
    double *Q = mat_f64(&args[0], "Q", &nq); if (!Q) return TSR_ENOMEM;
    double *R = mat_f64(&args[1], "R", &nr); if (!R) { fn_free_doubles(Q, nq); return TSR_ENOMEM; }
    double *u = mat_f64(&args[2], "u", &nu); if (!u) { fn_free_doubles(Q, nq); fn_free_doubles(R, nr); return TSR_ENOMEM; }
    double *v = mat_f64(&args[3], "v", &nv); if (!v) { fn_free_doubles(Q, nq); fn_free_doubles(R, nr); fn_free_doubles(u, nu); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (nu != m || nv != n) { fn_set_error("qr_update: u must have len m and v len n"); rc = TSR_EARG; }
    else {
        double *A = sl_qr_reconstruct(Q, R, m, kk, n);
        if (!A) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) A[i * n + j] += u[i] * v[j]; rc = sl_qr_canon_result(A, m, n, res); free(A); }
    }
    fn_free_doubles(Q, nq); fn_free_doubles(R, nr); fn_free_doubles(u, nu); fn_free_doubles(v, nv);
    return rc;
}

/* qr_insert(Q, R, u, k, which='row'): the economic QR of A = Q R with the row (len n) or column (len m) u
   inserted at index k, canonicalised (scipy.linalg.qr_insert; single row/column). */
static int r_qr_insert(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2 || args[2].kind != 3) { fn_set_error("qr_insert: Q, R must be 2-D and u an array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], kk = args[0].arr.shape[1], n = args[1].arr.shape[1];
    int64_t kidx = 0;
    if (nargs < 4 || sl_arg_int(&args[3], &kidx) != 0) { fn_set_error("qr_insert: k must be an integer"); return TSR_EARG; }
    int col = 0;
    if (nargs > 4 && args[4].kind == 2 && args[4].str) { if (!strcmp(args[4].str, "col") || !strcmp(args[4].str, "column")) col = 1; else if (strcmp(args[4].str, "row") != 0) { fn_set_error("qr_insert: which must be 'row' or 'col'"); return TSR_EARG; } }
    int64_t nq, nr, nu;
    double *Q = mat_f64(&args[0], "Q", &nq); if (!Q) return TSR_ENOMEM;
    double *R = mat_f64(&args[1], "R", &nr); if (!R) { fn_free_doubles(Q, nq); return TSR_ENOMEM; }
    double *u = mat_f64(&args[2], "u", &nu); if (!u) { fn_free_doubles(Q, nq); fn_free_doubles(R, nr); return TSR_ENOMEM; }
    int rc = TSR_OK;
    double *A = sl_qr_reconstruct(Q, R, m, kk, n);
    if (!A) rc = TSR_ENOMEM;
    else if (col) {                                          /* insert a column (len m) at col kidx -> m x (n+1) */
        if (nu != m || kidx < 0 || kidx > n) { fn_set_error("qr_insert: bad column insert"); rc = TSR_EARG; }
        else {
            const int64_t nn = n + 1;
            double *A2 = (double *)malloc(sizeof(double) * (size_t)(m * nn));
            if (!A2) rc = TSR_ENOMEM;
            else { for (int64_t i = 0; i < m; i++) { int64_t c = 0; for (int64_t j = 0; j < nn; j++) A2[i * nn + j] = (j == kidx) ? u[i] : A[i * n + (c++)]; } rc = sl_qr_canon_result(A2, m, nn, res); free(A2); }
        }
    } else {                                                 /* insert a row (len n) at row kidx -> (m+1) x n */
        if (nu != n || kidx < 0 || kidx > m) { fn_set_error("qr_insert: bad row insert"); rc = TSR_EARG; }
        else {
            const int64_t mm = m + 1;
            double *A2 = (double *)malloc(sizeof(double) * (size_t)(mm * n));
            if (!A2) rc = TSR_ENOMEM;
            else { int64_t ri = 0; for (int64_t i = 0; i < mm; i++) { if (i == kidx) for (int64_t j = 0; j < n; j++) A2[i * n + j] = u[j]; else { for (int64_t j = 0; j < n; j++) A2[i * n + j] = A[ri * n + j]; ri++; } } rc = sl_qr_canon_result(A2, mm, n, res); free(A2); }
        }
    }
    free(A);
    fn_free_doubles(Q, nq); fn_free_doubles(R, nr); fn_free_doubles(u, nu);
    return rc;
}

/* qr_delete(Q, R, k, p=1, which='row'): the economic QR of A = Q R with p rows or columns removed starting at
   index k, canonicalised (scipy.linalg.qr_delete). */
static int r_qr_delete(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("qr_delete: Q and R must be 2-D arrays"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0], kk = args[0].arr.shape[1], n = args[1].arr.shape[1];
    int64_t kidx = 0, p = 1;
    if (nargs < 3 || sl_arg_int(&args[2], &kidx) != 0) { fn_set_error("qr_delete: k must be an integer"); return TSR_EARG; }
    if (nargs > 3 && args[3].kind == 1) sl_arg_int(&args[3], &p);
    int col = 0;
    if (nargs > 4 && args[4].kind == 2 && args[4].str) { if (!strcmp(args[4].str, "col") || !strcmp(args[4].str, "column")) col = 1; else if (strcmp(args[4].str, "row") != 0) { fn_set_error("qr_delete: which must be 'row' or 'col'"); return TSR_EARG; } }
    int64_t nq, nr;
    double *Q = mat_f64(&args[0], "Q", &nq); if (!Q) return TSR_ENOMEM;
    double *R = mat_f64(&args[1], "R", &nr); if (!R) { fn_free_doubles(Q, nq); return TSR_ENOMEM; }
    int rc = TSR_OK;
    double *A = sl_qr_reconstruct(Q, R, m, kk, n);
    if (!A) rc = TSR_ENOMEM;
    else if (col) {                                          /* delete p columns at kidx -> m x (n-p) */
        if (kidx < 0 || p < 1 || kidx + p > n) { fn_set_error("qr_delete: bad column delete"); rc = TSR_EARG; }
        else {
            const int64_t nn = n - p;
            double *A2 = (double *)malloc(sizeof(double) * (size_t)(m * (nn > 0 ? nn : 1)));
            if (!A2) rc = TSR_ENOMEM;
            else { for (int64_t i = 0; i < m; i++) { int64_t c = 0; for (int64_t j = 0; j < n; j++) if (j < kidx || j >= kidx + p) A2[i * nn + (c++)] = A[i * n + j]; } rc = sl_qr_canon_result(A2, m, nn, res); free(A2); }
        }
    } else {                                                 /* delete p rows at kidx -> (m-p) x n */
        if (kidx < 0 || p < 1 || kidx + p > m) { fn_set_error("qr_delete: bad row delete"); rc = TSR_EARG; }
        else {
            const int64_t mm = m - p;
            double *A2 = (double *)malloc(sizeof(double) * (size_t)((mm > 0 ? mm : 1) * n));
            if (!A2) rc = TSR_ENOMEM;
            else { int64_t ri = 0; for (int64_t i = 0; i < m; i++) if (i < kidx || i >= kidx + p) { for (int64_t j = 0; j < n; j++) A2[ri * n + j] = A[i * n + j]; ri++; } rc = sl_qr_canon_result(A2, mm, n, res); free(A2); }
        }
    }
    free(A);
    fn_free_doubles(Q, nq); fn_free_doubles(R, nr);
    return rc;
}

/* cossin(X, p, q): the cosine-sine decomposition of a real orthogonal m x m matrix partitioned at (p, q):
   X = U @ CS @ VH with U = diag(U1, U2), VH = diag(V1^T, V2^T) block-orthogonal and CS the cosine-sine matrix,
   via LAPACK dorcsd (scipy.linalg.cossin, default swap_sign=False, separate=False). Returns (U, CS, VH). */
static int r_cossin(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("cossin: X must be a 2-D array"); return TSR_EARG; }
    const int64_t m = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != m) { fn_set_error("cossin: X must be square"); return TSR_EARG; }
    int64_t p = 0, q = 0;
    if (sl_arg_int(&args[1], &p) != 0 || sl_arg_int(&args[2], &q) != 0) { fn_set_error("cossin: p and q must be integers"); return TSR_EARG; }
    if (p <= 0 || p >= m || q <= 0 || q >= m) { fn_set_error("cossin: need 0<p<m and 0<q<m"); return TSR_EARG; }
    int64_t nx; double *X = mat_f64(&args[0], "X", &nx); if (!X) return TSR_ENOMEM;
    const int64_t mp = m - p, mq = m - q;
    const int64_t r = (p < q ? p : q) < (mp < mq ? mp : mq) ? (p < q ? p : q) : (mp < mq ? mp : mq);
    double *x11 = (double *)malloc(sizeof(double) * (size_t)(p * q));
    double *x12 = (double *)malloc(sizeof(double) * (size_t)(p * mq));
    double *x21 = (double *)malloc(sizeof(double) * (size_t)(mp * q));
    double *x22 = (double *)malloc(sizeof(double) * (size_t)(mp * mq));
    double *theta = (double *)malloc(sizeof(double) * (size_t)(r > 0 ? r : 1));
    double *u1 = (double *)malloc(sizeof(double) * (size_t)(p * p));
    double *u2 = (double *)malloc(sizeof(double) * (size_t)(mp * mp));
    double *v1t = (double *)malloc(sizeof(double) * (size_t)(q * q));
    double *v2t = (double *)malloc(sizeof(double) * (size_t)(mq * mq));
    int rc = TSR_OK;
    if (!x11 || !x12 || !x21 || !x22 || !theta || !u1 || !u2 || !v1t || !v2t) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < p; i++) for (int64_t j = 0; j < q; j++) x11[i * q + j] = X[i * m + j];
        for (int64_t i = 0; i < p; i++) for (int64_t j = 0; j < mq; j++) x12[i * mq + j] = X[i * m + (q + j)];
        for (int64_t i = 0; i < mp; i++) for (int64_t j = 0; j < q; j++) x21[i * q + j] = X[(p + i) * m + j];
        for (int64_t i = 0; i < mp; i++) for (int64_t j = 0; j < mq; j++) x22[i * mq + j] = X[(p + i) * m + (q + j)];
        lapack_int info = LAPACKE_dorcsd(LAPACK_ROW_MAJOR, 'Y', 'Y', 'Y', 'Y', 'N', 'D',
                                         (lapack_int)m, (lapack_int)p, (lapack_int)q,
                                         x11, (lapack_int)q, x12, (lapack_int)mq, x21, (lapack_int)q, x22, (lapack_int)mq,
                                         theta, u1, (lapack_int)p, u2, (lapack_int)mp, v1t, (lapack_int)q, v2t, (lapack_int)mq);
        if (info != 0) { fn_set_error("cossin: dorcsd failed"); rc = TSR_EARG; }
        else {
            int64_t osh[2] = {m, m};
            double *U = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
            double *CS = (double *)fn_result_array(&res[1], TSR_F64, 2, osh);
            double *VH = (double *)fn_result_array(&res[2], TSR_F64, 2, osh);
            if (!U || !CS || !VH) rc = TSR_ENOMEM;
            else {
                for (int64_t i = 0; i < m * m; i++) { U[i] = 0.0; VH[i] = 0.0; CS[i] = 0.0; }
                for (int64_t i = 0; i < p; i++) for (int64_t j = 0; j < p; j++) U[i * m + j] = u1[i * p + j];
                for (int64_t i = 0; i < mp; i++) for (int64_t j = 0; j < mp; j++) U[(p + i) * m + (p + j)] = u2[i * mp + j];
                for (int64_t i = 0; i < q; i++) for (int64_t j = 0; j < q; j++) VH[i * m + j] = v1t[i * q + j];
                for (int64_t i = 0; i < mq; i++) for (int64_t j = 0; j < mq; j++) VH[(q + i) * m + (q + j)] = v2t[i * mq + j];
                /* CS assembly (scipy _cossin, swap_sign=False) */
                const int64_t minpq = p < q ? p : q, minpmq = p < mq ? p : mq;
                const int64_t minmpq = mp < q ? mp : q, minmpmq = mp < mq ? mp : mq;
                const int64_t n11 = minpq - r, n12 = minpmq - r, n21 = minmpq - r, n22 = minmpmq - r;
                for (int64_t i = 0; i < n11; i++) CS[i * m + i] = 1.0;
                { int64_t xs = n11 + r, ys = n11 + n21 + n22 + 2 * r; for (int64_t t = 0; t < n12; t++) CS[(xs + t) * m + (ys + t)] = -1.0; }
                { int64_t xs = p + n22 + r, ys = n11 + r; for (int64_t t = 0; t < n21; t++) CS[(xs + t) * m + (ys + t)] = 1.0; }
                for (int64_t t = 0; t < n22; t++) CS[(p + t) * m + (q + t)] = 1.0;
                for (int64_t t = 0; t < r; t++) CS[(n11 + t) * m + (n11 + t)] = cos(theta[t]);
                { int64_t xs = p + n22, ys = n11 + r + n21 + n22; for (int64_t t = 0; t < r; t++) CS[(xs + t) * m + (ys + t)] = cos(theta[t]); }
                { int64_t xs = n11, ys = n11 + n21 + n22 + r; for (int64_t t = 0; t < r; t++) CS[(xs + t) * m + (ys + t)] = -sin(theta[t]); }
                { int64_t xs = p + n22, ys = n11; for (int64_t t = 0; t < r; t++) CS[(xs + t) * m + (ys + t)] = sin(theta[t]); }
            }
        }
    }
    free(x11); free(x12); free(x21); free(x22); free(theta); free(u1); free(u2); free(v1t); free(v2t);
    fn_free_doubles(X, nx);
    return rc;
}

/* ordqz(A, B, sort='lhp', output='real'): the generalised real Schur decomposition reordered so the eigenvalues
   selected by `sort` (lhp/rhp/iuc/ouc) appear first, via dgges then dtgsen (scipy.linalg.ordqz). Returns
   (AA, BB, alpha, beta, Q, Z); alpha is complex128, beta real. */
static int r_ordqz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("ordqz: A and B must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[1].arr.shape[1] != n) { fn_set_error("ordqz: A and B must be square and the same size"); return TSR_EARG; }
    int sort = 0;                                            /* 0 lhp, 1 rhp, 2 iuc, 3 ouc */
    if (nargs > 2 && args[2].kind == 2 && args[2].str) {
        const char *s = args[2].str;
        if (!strcmp(s, "lhp")) sort = 0; else if (!strcmp(s, "rhp")) sort = 1;
        else if (!strcmp(s, "iuc")) sort = 2; else if (!strcmp(s, "ouc")) sort = 3;
        else { fn_set_error("ordqz: sort must be 'lhp', 'rhp', 'iuc' or 'ouc'"); return TSR_EARG; }
    }
    if (nargs > 3 && args[3].kind == 2 && args[3].str && strcmp(args[3].str, "real") != 0) { fn_set_error("ordqz: only output='real' is supported"); return TSR_EARG; }
    int64_t na, nb;
    double *a = mat_f64(&args[0], "A", &na); if (!a) return TSR_ENOMEM;
    double *b = mat_f64(&args[1], "B", &nb); if (!b) { fn_free_doubles(a, na); return TSR_ENOMEM; }
    const size_t nz = (size_t)(n * n > 0 ? n * n : 1), nv = (size_t)(n > 0 ? n : 1);
    double *alphar = (double *)malloc(sizeof(double) * nv);
    double *alphai = (double *)malloc(sizeof(double) * nv);
    double *beta = (double *)malloc(sizeof(double) * nv);
    double *Q = (double *)malloc(sizeof(double) * nz);
    double *Z = (double *)malloc(sizeof(double) * nz);
    lapack_logical *sel = (lapack_logical *)malloc(sizeof(lapack_logical) * nv);
    int rc = TSR_OK;
    if (!alphar || !alphai || !beta || !Q || !Z || !sel) rc = TSR_ENOMEM;
    else if (n > 0) {
        lapack_int sdim = 0;
        lapack_int info = LAPACKE_dgges(LAPACK_ROW_MAJOR, 'V', 'V', 'N', NULL, (lapack_int)n, a, (lapack_int)n, b, (lapack_int)n, &sdim, alphar, alphai, beta, Q, (lapack_int)n, Z, (lapack_int)n);
        if (info != 0) { rc = TSR_EARG; fn_set_error("ordqz: dgges failed"); }
        else {
            for (int64_t i = 0; i < n; i++) {
                const double ar = alphar[i], ai = alphai[i], be = beta[i];
                int s;
                if (sort == 0) s = (be != 0.0) && (ar / be < 0.0);
                else if (sort == 1) s = (be != 0.0) && (ar / be > 0.0);
                else if (sort == 2) s = (be != 0.0) && (ar * ar + ai * ai < be * be);
                else { if (ar == 0.0 && ai == 0.0 && be == 0.0) s = 0; else if (be == 0.0) s = 1; else s = (ar * ar + ai * ai > be * be); }
                sel[i] = s ? 1 : 0;
            }
            lapack_int mout = 0; double pl = 0.0, pr = 0.0, dif[2] = {0.0, 0.0};
            lapack_int info2 = LAPACKE_dtgsen(LAPACK_ROW_MAJOR, 0, 1, 1, sel, (lapack_int)n, a, (lapack_int)n, b, (lapack_int)n,
                                              alphar, alphai, beta, Q, (lapack_int)n, Z, (lapack_int)n, &mout, &pl, &pr, dif);
            if (info2 == 1) { rc = TSR_EARG; fn_set_error("ordqz: reordering failed (ill-conditioned pencil)"); }
            else if (info2 != 0) { rc = TSR_EARG; fn_set_error("ordqz: dtgsen failed"); }
        }
    }
    if (rc == TSR_OK) {
        int64_t sh2[2] = {n, n}, sh1[1] = {n};
        double *AA = (double *)fn_result_array(&res[0], TSR_F64, 2, sh2);
        double *BB = (double *)fn_result_array(&res[1], TSR_F64, 2, sh2);
        double *alpha = (double *)fn_result_array(&res[2], TSR_C128, 1, sh1);
        double *bet = (double *)fn_result_array(&res[3], TSR_F64, 1, sh1);
        double *Qo = (double *)fn_result_array(&res[4], TSR_F64, 2, sh2);
        double *Zo = (double *)fn_result_array(&res[5], TSR_F64, 2, sh2);
        if (!AA || !BB || !alpha || !bet || !Qo || !Zo) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < n * n; i++) { AA[i] = a[i]; BB[i] = b[i]; Qo[i] = Q[i]; Zo[i] = Z[i]; }
            for (int64_t i = 0; i < n; i++) { alpha[2 * i] = alphar[i]; alpha[2 * i + 1] = alphai[i]; bet[i] = beta[i]; }
        }
    }
    free(alphar); free(alphai); free(beta); free(Q); free(Z); free(sel);
    fn_free_doubles(a, na); fn_free_doubles(b, nb);
    return rc;
}

/* scipy.linalg: the functions whose default behaviour matches numpy.linalg reuse the same routines (the extra
   scipy-only keyword arguments do not change the result for the covered cases). Always-complex eig/eigvals and
   expm come in later commits. */
static const int ALWAYS_COMPLEX = 1;                       /* scipy.linalg.eig / eigvals always return complex128 */
static const fn_def SCIPY_DEFS[] = {
    ROUTINE("slinalg.solve", 1, "a, b", "out", r_solve, NULL, "Solve a linear system a x = b (scipy.linalg.solve)."),
    ROUTINE("slinalg.inv", 1, "a", "out", r_inv, NULL, "Inverse of a square matrix (scipy.linalg.inv)."),
    ROUTINE("slinalg.det", 1, "a", "out", r_det, NULL, "Determinant of a square matrix (scipy.linalg.det)."),
    ROUTINE("slinalg.svdvals", 1, "a", "out", r_svdvals, NULL, "Singular values of a matrix (scipy.linalg.svdvals)."),
    ROUTINE("slinalg.lstsq", 4, "a, b, cond=None", "x, residues, rank, s", r_lstsq, NULL, "Least-squares solution (scipy.linalg.lstsq)."),
    ROUTINE("slinalg.pinv", 1, "a", "out", r_pinv, NULL, "Moore-Penrose pseudo-inverse via SVD (scipy.linalg.pinv)."),
    ROUTINE("slinalg.pinvh", 1, "a", "out", r_pinvh, NULL, "Moore-Penrose pseudo-inverse of a symmetric matrix via eigendecomposition (scipy.linalg.pinvh)."),
    ROUTINE("slinalg.norm", 1, "a, ord=None, axis=None, keepdims=False", "out", r_norm, NULL, "Vector or matrix norm (scipy.linalg.norm)."),
    ROUTINE("slinalg.eigvalsh", 1, "a", "out", r_eigvalsh, NULL, "Eigenvalues of a symmetric matrix, ascending (scipy.linalg.eigvalsh)."),
    ROUTINE("slinalg.svd", 3, "a, full_matrices=True", "U, s, Vh", r_svd, NULL, "Singular value decomposition (scipy.linalg.svd)."),
    ROUTINE("slinalg.eigh", 2, "a", "eigenvalues, eigenvectors", r_eigh, NULL, "Eigenvalues and eigenvectors of a symmetric matrix (scipy.linalg.eigh)."),
    ROUTINE("slinalg.qr", 2, "a, mode='full'", "Q, R", r_qr, NULL, "QR factorisation (scipy.linalg.qr; mode 'full' or 'economic')."),
    ROUTINE("slinalg.cholesky", 1, "a, lower=False", "out", r_scipy_cholesky, NULL, "Cholesky factor, upper by default (scipy.linalg.cholesky)."),
    ROUTINE("slinalg.cho_factor", 2, "a, lower=False", "c, lower", r_cho_factor, NULL, "Cholesky factor for cho_solve, other triangle left as input (scipy.linalg.cho_factor)."),
    ROUTINE("slinalg.cho_solve", 1, "c, lower, b", "out", r_cho_solve, NULL, "Solve a x = b from a Cholesky factor; takes c, lower, b unpacked (scipy.linalg.cho_solve)."),
    ROUTINE("slinalg.solve_triangular", 1, "a, b, trans=0, lower=False, unit_diagonal=False", "out", r_solve_triangular, NULL, "Solve a triangular system (scipy.linalg.solve_triangular)."),
    ROUTINE("slinalg.lu", 3, "a, permute_l=False", "p, l, u", r_lu, NULL, "LU factorisation with partial pivoting, A = P L U (scipy.linalg.lu)."),
    ROUTINE("slinalg.toeplitz", 1, "c, r=None", "out", r_toeplitz, NULL, "Toeplitz matrix with first column c and first row r (scipy.linalg.toeplitz)."),
    ROUTINE("slinalg.block_diag", 1, "*arrs", "out", r_block_diag, NULL, "Assemble 2-D blocks on the diagonal of a larger zero matrix (scipy.linalg.block_diag)."),
    ROUTINE("slinalg.circulant", 1, "c", "out", r_circulant, NULL, "Circulant matrix with first column c (scipy.linalg.circulant)."),
    ROUTINE("slinalg.companion", 1, "a", "out", r_companion, NULL, "Companion matrix of a polynomial (scipy.linalg.companion)."),
    ROUTINE("slinalg.hadamard", 1, "n", "out", r_hadamard, NULL, "Sylvester-Hadamard matrix of order n, a power of two (scipy.linalg.hadamard)."),
    ROUTINE("slinalg.hilbert", 1, "n", "out", r_hilbert, NULL, "Hilbert matrix of order n (scipy.linalg.hilbert)."),
    ROUTINE("slinalg.hankel", 1, "c, r=None", "out", r_hankel, NULL, "Hankel matrix with first column c and last row r (scipy.linalg.hankel)."),
    ROUTINE("slinalg.fiedler", 1, "a", "out", r_fiedler, NULL, "Symmetric Fiedler matrix of absolute differences |a_i - a_j| (scipy.linalg.fiedler)."),
    ROUTINE("slinalg.leslie", 1, "f, s", "out", r_leslie, NULL, "Leslie population matrix with first row f and sub-diagonal s (scipy.linalg.leslie)."),
    ROUTINE("slinalg.pascal", 1, "n, kind='symmetric'", "out", r_pascal, NULL, "Pascal matrix of order n (scipy.linalg.pascal)."),
    ROUTINE("slinalg.invpascal", 1, "n, kind='symmetric'", "out", r_invpascal, NULL, "Inverse of the Pascal matrix of order n (scipy.linalg.invpascal)."),
    ROUTINE("slinalg.convolution_matrix", 1, "a, n, mode='full'", "out", r_convolution_matrix, NULL, "Toeplitz matrix A with A @ v = convolve(a, v, mode) (scipy.linalg.convolution_matrix)."),
    ROUTINE("slinalg.dft", 1, "n, scale=None", "out", r_dft, NULL, "The n x n discrete Fourier transform matrix (scipy.linalg.dft)."),
    ROUTINE("slinalg.khatri_rao", 1, "a, b", "out", r_khatri_rao, NULL, "Column-wise Kronecker (Khatri-Rao) product (scipy.linalg.khatri_rao)."),
    ROUTINE("slinalg.diagsvd", 1, "s, M, N", "out", r_diagsvd, NULL, "M x N matrix with s on the diagonal (scipy.linalg.diagsvd)."),
    ROUTINE("slinalg.orth", 1, "A, rcond=None", "out", r_orth, NULL, "Orthonormal basis for the range of A via SVD (scipy.linalg.orth)."),
    ROUTINE("slinalg.null_space", 1, "A, rcond=None", "out", r_null_space, NULL, "Orthonormal basis for the null space of A via SVD (scipy.linalg.null_space)."),
    ROUTINE("slinalg.polar", 2, "a, side='right'", "u, p", r_polar, NULL, "Polar decomposition a = u p (right) or p u (left) via SVD (scipy.linalg.polar)."),
    ROUTINE("slinalg.lu_factor", 2, "a", "lu, piv", r_lu_factor, NULL, "LU factorisation for lu_solve: combined factors and 0-based pivots (scipy.linalg.lu_factor)."),
    ROUTINE("slinalg.lu_solve", 1, "lu, piv, b, trans=0", "out", r_lu_solve, NULL, "Solve a x = b from an lu_factor result; takes lu, piv, b unpacked (scipy.linalg.lu_solve)."),
    ROUTINE("slinalg.hessenberg", 1, "a, calc_q=False", "out", r_hessenberg, NULL, "Upper Hessenberg form via dgehrd (scipy.linalg.hessenberg)."),
    ROUTINE("slinalg.schur", 2, "a, output='real'", "T, Z", r_schur, NULL, "Real Schur decomposition A = Z T Z^T via dgees (scipy.linalg.schur)."),
    ROUTINE("slinalg.solve_sylvester", 1, "a, b, q", "out", r_solve_sylvester, NULL, "Solve the Sylvester equation a x + x b = q (scipy.linalg.solve_sylvester)."),
    ROUTINE("slinalg.solve_continuous_lyapunov", 1, "a, q", "out", r_solve_continuous_lyapunov, NULL, "Solve the continuous Lyapunov equation a x + x a^H = q (scipy.linalg.solve_continuous_lyapunov)."),
    ROUTINE("slinalg.eigvalsh_tridiagonal", 1, "d, e", "out", r_eigvalsh_tridiagonal, NULL, "Eigenvalues of a symmetric tridiagonal matrix via dsterf (scipy.linalg.eigvalsh_tridiagonal)."),
    ROUTINE("slinalg.issymmetric", 1, "a, atol=None, rtol=None", "out", r_issymmetric, NULL, "Whether a square matrix is symmetric (scipy.linalg.issymmetric)."),
    ROUTINE("slinalg.ishermitian", 1, "a, atol=None, rtol=None", "out", r_issymmetric, NULL, "Whether a square matrix is Hermitian; for real input, symmetric (scipy.linalg.ishermitian)."),
    ROUTINE("slinalg.solve_toeplitz", 1, "c_or_cr, b", "out", r_solve_toeplitz, NULL, "Solve a Toeplitz system T x = b (scipy.linalg.solve_toeplitz)."),
    ROUTINE("slinalg.solve_circulant", 1, "c, b", "out", r_solve_circulant, NULL, "Solve a circulant system C x = b (scipy.linalg.solve_circulant)."),
    ROUTINE("slinalg.matmul_toeplitz", 1, "c_or_cr, x", "out", r_matmul_toeplitz, NULL, "The product of a Toeplitz matrix with x (scipy.linalg.matmul_toeplitz)."),
    ROUTINE("slinalg.invhilbert", 1, "n", "out", r_invhilbert, NULL, "Inverse of the Hilbert matrix of order n (scipy.linalg.invhilbert)."),
    ROUTINE("slinalg.eigh_tridiagonal", 2, "d, e", "eigenvalues, eigenvectors", r_eigh_tridiagonal, NULL, "Eigenvalues and eigenvectors of a symmetric tridiagonal matrix via dstev (scipy.linalg.eigh_tridiagonal)."),
    ROUTINE("slinalg.eigvals_banded", 1, "a_band, lower=False", "out", r_eigvals_banded, NULL, "Eigenvalues of a symmetric banded matrix in band storage via dsbevd (scipy.linalg.eigvals_banded)."),
    ROUTINE("slinalg.cholesky_banded", 1, "ab, lower=False", "out", r_cholesky_banded, NULL, "Cholesky factor of a symmetric positive-definite banded matrix via dpbtrf (scipy.linalg.cholesky_banded)."),
    ROUTINE("slinalg.solve_discrete_lyapunov", 1, "a, q", "out", r_solve_discrete_lyapunov, NULL, "Solve the discrete Lyapunov equation a x a^H - x + q = 0 (scipy.linalg.solve_discrete_lyapunov)."),
    ROUTINE("slinalg.helmert", 1, "n, full=False", "out", r_helmert, NULL, "Helmert matrix of order n (scipy.linalg.helmert)."),
    ROUTINE("slinalg.cho_solve_banded", 1, "cb, lower, b", "out", r_cho_solve_banded, NULL, "Solve A x = b from a banded Cholesky factor via dpbtrs (scipy.linalg.cho_solve_banded)."),
    ROUTINE("slinalg.rq", 2, "a, mode='full'", "R, Q", r_rq, NULL, "RQ decomposition a = R Q of a square matrix via dgerqf (scipy.linalg.rq)."),
    ROUTINE("slinalg.orthogonal_procrustes", 2, "A, B", "R, scale", r_orthogonal_procrustes, NULL, "Orthogonal Procrustes solution R minimising ||A R - B|| and scale (scipy.linalg.orthogonal_procrustes)."),
    ROUTINE("slinalg.matrix_balance", 2, "A, permute=True, scale=True, separate=False", "B, T", r_matrix_balance, NULL, "Balance a matrix via dgebal; returns the balanced matrix and the scaling transform (scipy.linalg.matrix_balance)."),
    ROUTINE("slinalg.solve_banded", 1, "l, u, ab, b", "out", r_solve_banded, NULL, "Solve a banded linear system via dgbsv; takes l, u, ab, b unpacked (scipy.linalg.solve_banded)."),
    ROUTINE("slinalg.solveh_banded", 1, "ab, b, lower=False", "out", r_solveh_banded, NULL, "Solve a Hermitian positive-definite banded system via dpbsv (scipy.linalg.solveh_banded)."),
    ROUTINE("slinalg.eig_banded", 2, "ab, lower=False, eigvals_only=False", "w, v", r_eig_banded, NULL, "Eigenvalues and eigenvectors of a symmetric banded matrix via dsbevd (scipy.linalg.eig_banded)."),
    ROUTINE("slinalg.expm_frechet", 2, "A, E", "expm, frechet", r_expm_frechet, NULL, "Matrix exponential and its Frechet derivative in direction E via the block-enlarge identity (scipy.linalg.expm_frechet)."),
    ROUTINE("slinalg.expm_cond", 1, "A", "out", r_expm_cond, NULL, "Relative condition number of the matrix exponential from the Kronecker form of its Frechet derivative (scipy.linalg.expm_cond)."),
    ROUTINE("slinalg.rsf2csf", 2, "T, Z", "T, Z", r_rsf2csf, NULL, "Convert a real Schur form to the complex (upper-triangular) Schur form (scipy.linalg.rsf2csf)."),
    ROUTINE("slinalg.cdf2rdf", 2, "w, v", "wr, vr", r_cdf2rdf, NULL, "Convert complex eigenvalues/eigenvectors to real block-diagonal form (scipy.linalg.cdf2rdf)."),
    ROUTINE("slinalg.subspace_angles", 2, "A, B", "out", r_subspace_angles, NULL, "Principal angles between the column spaces of A and B (scipy.linalg.subspace_angles)."),
    ROUTINE("slinalg.solve_continuous_are", 4, "a, b, q, r", "out", r_solve_continuous_are, NULL, "Stabilising solution of the continuous-time algebraic Riccati equation (scipy.linalg.solve_continuous_are)."),
    ROUTINE("slinalg.solve_discrete_are", 4, "a, b, q, r", "out", r_solve_discrete_are, NULL, "Stabilising solution of the discrete-time algebraic Riccati equation (scipy.linalg.solve_discrete_are)."),
    ROUTINE("slinalg.signm", 1, "A", "out", r_signm, NULL, "Matrix sign function via the Schur-Parlett method (scipy.linalg.signm; real, well-separated spectrum)."),
    ROUTINE("slinalg.qr_multiply", 3, "a, c, mode='right'", "CQ, R", r_qr_multiply, NULL, "Product of c with the economic orthogonal factor Q of a, plus R (scipy.linalg.qr_multiply)."),
    ROUTINE("slinalg.qr_update", 4, "Q, R, u, v", "Q, R", r_qr_update, NULL, "Economic QR of (Q R + u v^T), canonicalised (scipy.linalg.qr_update)."),
    ROUTINE("slinalg.qr_insert", 5, "Q, R, u, k, which='row'", "Q, R", r_qr_insert, NULL, "Economic QR after inserting a row/column, canonicalised (scipy.linalg.qr_insert)."),
    ROUTINE("slinalg.qr_delete", 5, "Q, R, k, p=1, which='row'", "Q, R", r_qr_delete, NULL, "Economic QR after deleting rows/columns, canonicalised (scipy.linalg.qr_delete)."),
    ROUTINE("slinalg.cossin", 3, "X, p, q", "u, cs, vh", r_cossin, NULL, "Cosine-sine decomposition of a partitioned orthogonal matrix via dorcsd (scipy.linalg.cossin)."),
    ROUTINE("slinalg.ordqz", 6, "A, B, sort='lhp', output='real'", "AA, BB, alpha, beta, Q, Z", r_ordqz, NULL, "Reordered generalised Schur (QZ) decomposition via dgges and dtgsen (scipy.linalg.ordqz)."),
    ROUTINE("slinalg.qz", 4, "A, B, output='real'", "AA, BB, Q, Z", r_qz, NULL, "Generalised real Schur decomposition via dgges (scipy.linalg.qz)."),
    ROUTINE("slinalg.sqrtm", 1, "a, disp=True", "out", r_sqrtm, NULL, "Principal matrix square root via the Schur method (scipy.linalg.sqrtm; real spectrum)."),
    ROUTINE("slinalg.logm", 1, "a, disp=True", "out", r_logm, NULL, "Principal matrix logarithm via the Schur-Parlett method (scipy.linalg.logm; distinct positive real spectrum)."),
    ROUTINE("slinalg.ldl", 3, "a, lower=True", "lu, d, perm", r_ldl, NULL, "LDL^T factorisation of a symmetric matrix via dsytrf (scipy.linalg.ldl)."),
    ROUTINE("slinalg.sinm", 1, "a", "out", r_sinm, NULL, "Matrix sine via the Schur-Parlett method (scipy.linalg.sinm; real spectrum)."),
    ROUTINE("slinalg.cosm", 1, "a", "out", r_cosm, NULL, "Matrix cosine via the Schur-Parlett method (scipy.linalg.cosm; real spectrum)."),
    ROUTINE("slinalg.sinhm", 1, "a", "out", r_sinhm, NULL, "Matrix hyperbolic sine via the Schur-Parlett method (scipy.linalg.sinhm; real spectrum)."),
    ROUTINE("slinalg.coshm", 1, "a", "out", r_coshm, NULL, "Matrix hyperbolic cosine via the Schur-Parlett method (scipy.linalg.coshm; real spectrum)."),
    ROUTINE("slinalg.tanhm", 1, "a", "out", r_tanhm, NULL, "Matrix hyperbolic tangent via the Schur-Parlett method (scipy.linalg.tanhm; real spectrum)."),
    ROUTINE("slinalg.expm", 1, "a", "out", r_expm, NULL, "Matrix exponential via the Schur-Parlett method (scipy.linalg.expm; real spectrum)."),
    ROUTINE("slinalg.tanm", 1, "a", "out", r_tanm, NULL, "Matrix tangent via the Schur-Parlett method (scipy.linalg.tanm; real spectrum)."),
    ROUTINE("slinalg.fractional_matrix_power", 1, "a, t", "out", r_fractional_matrix_power, NULL, "Fractional matrix power a^t via the Schur-Parlett method (scipy.linalg.fractional_matrix_power; positive real spectrum)."),
    ROUTINE("slinalg.eigvals", 1, "a", "out", r_eigvals, &ALWAYS_COMPLEX, "Eigenvalues of a general matrix, complex, sorted (scipy.linalg.eigvals)."),
    ROUTINE("slinalg.eig", 2, "a", "eigenvalues, eigenvectors", r_eig, &ALWAYS_COMPLEX, "Eigenvalues and right eigenvectors of a general matrix, complex (scipy.linalg.eig)."),
};
const fn_table TSR_SCIPY_LINALG_TABLE = {SCIPY_DEFS, (int)(sizeof SCIPY_DEFS / sizeof SCIPY_DEFS[0])};
