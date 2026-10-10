/* scipy.sparse.linalg ("splinalg." prefix -> Tessero\SparseLinalg facade).
 *
 * Direct solves, matrix norms and structure queries for sparse matrices. As with the rest of the sparse
 * parity surface, matrix arguments arrive dense (the generators build them from a dense array) and the
 * routines compute the value SciPy returns; both backends read this table at runtime.
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

extern int sl_dense_solve(double *M, double *b, int64_t n, int64_t nrhs);   /* np_linalg.c (row-major dgesv) */
extern int sl_expm(const double *A, int64_t n, double *out);                /* np_linalg.c (Pade scaling-squaring) */

/* C = A * B, all n-by-n row-major; C must be distinct from A and B. */
static void spl_mm(double *C, const double *A, const double *B, int64_t n)
{
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) {
        double s = 0.0; for (int64_t k = 0; k < n; k++) s += A[i * n + k] * B[k * n + j];
        C[i * n + j] = s;
    }
}

static void spl_bool(tsr_result *r, int b) { memset(r, 0, sizeof *r); r->kind = 4; r->num = b ? 1.0 : 0.0; }

/* spsolve(A, b): solve A x = b for a square A (dense here). b is a vector (n,) or a matrix (n, k); x matches. */
static int r_spsolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3) { fn_set_error("spsolve: A must be 2-D and b an array"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("spsolve: A must be square"); return TSR_EARG; }
    const tsr_array *B = &args[1].arr;
    int vector = B->ndim == 1;
    int64_t k = vector ? 1 : B->shape[1];
    if (B->shape[0] != n) { fn_set_error("spsolve: b has the wrong number of rows"); return TSR_EARG; }
    int64_t La, Lb; double *A = fn_arg_doubles(&args[0], &La); if (!A) return TSR_ENOMEM;
    double *b = fn_arg_doubles(&args[1], &Lb); if (!b) { fn_free_doubles(A, La); return TSR_ENOMEM; }
    int64_t sh1[1] = {n}, sh2[2] = {n, k};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, vector ? 1 : 2, vector ? sh1 : sh2);
    int rc = TSR_OK;
    if (!x) rc = TSR_ENOMEM;
    else {
        memcpy(x, b, (size_t)(n * k) * sizeof(double));                      /* sl_dense_solve overwrites rhs in place */
        if (sl_dense_solve(A, x, n, k) != 0) { fn_set_error("spsolve: matrix is singular"); rc = TSR_EARG; }
    }
    fn_free_doubles(A, La); fn_free_doubles(b, Lb);
    return rc;
}

/* spsolve_triangular(A, b, lower=True, overwrite_A=False, overwrite_b=False, unit_diagonal=False). */
static int r_spsolve_triangular(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3) { fn_set_error("spsolve_triangular: A must be 2-D and b an array"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("spsolve_triangular: A must be square"); return TSR_EARG; }
    int lower = 1;
    if (nargs > 2 && args[2].kind == 4) lower = args[2].num != 0.0;
    int unit = 0;
    if (nargs > 5 && args[5].kind == 4) unit = args[5].num != 0.0;
    const tsr_array *B = &args[1].arr;
    int vector = B->ndim == 1;
    int64_t k = vector ? 1 : B->shape[1];
    if (B->shape[0] != n) { fn_set_error("spsolve_triangular: b has the wrong number of rows"); return TSR_EARG; }
    int64_t La, Lb; double *A = fn_arg_doubles(&args[0], &La); if (!A) return TSR_ENOMEM;
    double *b = fn_arg_doubles(&args[1], &Lb); if (!b) { fn_free_doubles(A, La); return TSR_ENOMEM; }
    int64_t sh1[1] = {n}, sh2[2] = {n, k};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, vector ? 1 : 2, vector ? sh1 : sh2);
    int rc = TSR_OK;
    if (!x) rc = TSR_ENOMEM;
    else {
        memcpy(x, b, (size_t)(n * k) * sizeof(double));
        for (int64_t c = 0; c < k && rc == TSR_OK; c++) {
            if (lower) {
                for (int64_t i = 0; i < n; i++) {
                    double s = x[i * k + c];
                    for (int64_t j = 0; j < i; j++) s -= A[i * n + j] * x[j * k + c];
                    double d = unit ? 1.0 : A[i * n + i];
                    if (d == 0.0) { fn_set_error("spsolve_triangular: zero on the diagonal"); rc = TSR_EARG; break; }
                    x[i * k + c] = s / d;
                }
            } else {
                for (int64_t i = n - 1; i >= 0; i--) {
                    double s = x[i * k + c];
                    for (int64_t j = i + 1; j < n; j++) s -= A[i * n + j] * x[j * k + c];
                    double d = unit ? 1.0 : A[i * n + i];
                    if (d == 0.0) { fn_set_error("spsolve_triangular: zero on the diagonal"); rc = TSR_EARG; break; }
                    x[i * k + c] = s / d;
                }
            }
        }
    }
    fn_free_doubles(A, La); fn_free_doubles(b, Lb);
    return rc;
}

/* norm(A, ord=None): matrix norms as scipy.sparse.linalg.norm -- fro (None/'fro'), 1, -1, inf, -inf. */
static int r_norm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("norm: A must be a 2-D matrix"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], L;
    double *A = fn_arg_doubles(&args[0], &L); if (!A) return TSR_ENOMEM;
    /* ord: None/'fro' -> Frobenius; string 'fro'/'nuc'; number 1,-1,inf,-inf */
    int kind = 0;                                                            /* 0 fro, 1 one, 2 negone, 3 inf, 4 neginf */
    if (nargs > 1 && args[1].kind == 2) {
        if (!strcmp(args[1].str, "fro")) kind = 0;
        else { fn_free_doubles(A, L); fn_set_error("norm: unsupported string ord"); return TSR_EARG; }
    } else if (nargs > 1 && args[1].kind == 1) {
        double o = args[1].num;
        if (isinf(o)) kind = o > 0 ? 3 : 4;
        else if (o == 1.0) kind = 1;
        else if (o == -1.0) kind = 2;
        else { fn_free_doubles(A, L); fn_set_error("norm: unsupported ord for a sparse matrix"); return TSR_EARG; }
    }
    double val = 0.0;
    if (kind == 0) {
        for (int64_t i = 0; i < m * n; i++) val += A[i] * A[i];
        val = sqrt(val);
    } else if (kind == 1 || kind == 2) {                                     /* max/min column abs-sum */
        for (int64_t j = 0; j < n; j++) {
            double s = 0.0; for (int64_t i = 0; i < m; i++) s += fabs(A[i * n + j]);
            if (j == 0) val = s; else if (kind == 1) { if (s > val) val = s; } else if (s < val) val = s;
        }
    } else {                                                                 /* max/min row abs-sum */
        for (int64_t i = 0; i < m; i++) {
            double s = 0.0; for (int64_t j = 0; j < n; j++) s += fabs(A[i * n + j]);
            if (i == 0) val = s; else if (kind == 3) { if (s > val) val = s; } else if (s < val) val = s;
        }
    }
    fn_free_doubles(A, L);
    fn_result_num(&res[0], val);
    return TSR_OK;
}

/* spbandwidth(A): (lower, upper) -- the largest i-j and j-i over the nonzeros. */
static int r_spbandwidth(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("spbandwidth: A must be a 2-D matrix"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], L;
    double *A = fn_arg_doubles(&args[0], &L); if (!A) return TSR_ENOMEM;
    int64_t lo = 0, up = 0;
    for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++)
        if (A[i * n + j] != 0.0) { if (i - j > lo) lo = i - j; if (j - i > up) up = j - i; }
    fn_free_doubles(A, L);
    fn_result_int(&res[0], lo);
    fn_result_int(&res[1], up);
    return TSR_OK;
}

/* is_sptriangular(A): (is_lower, is_upper). */
static int r_is_sptriangular(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("is_sptriangular: A must be a 2-D matrix"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], L;
    double *A = fn_arg_doubles(&args[0], &L); if (!A) return TSR_ENOMEM;
    int is_lower = 1, is_upper = 1;
    for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) if (A[i * n + j] != 0.0) {
        if (j > i) is_lower = 0;
        if (j < i) is_upper = 0;
    }
    fn_free_doubles(A, L);
    spl_bool(&res[0], is_lower);
    spl_bool(&res[1], is_upper);
    return TSR_OK;
}

/* inv(A): dense inverse of a square (sparse) matrix -- scipy.sparse.linalg.inv returns the (typically dense)
   inverse; here it is returned as a dense array. Solve A X = I. */
static int r_inv(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("inv: A must be square"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0], L; double *A = fn_arg_doubles(&args[0], &L);
    if (!A) return TSR_ENOMEM;
    double *M = (double *)malloc((size_t)(n * n) * sizeof(double));
    int64_t sh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    int rc = TSR_OK;
    if (!M || !out) rc = TSR_ENOMEM;
    else {
        memcpy(M, A, (size_t)(n * n) * sizeof(double));
        for (int64_t i = 0; i < n * n; i++) out[i] = 0.0;
        for (int64_t i = 0; i < n; i++) out[i * n + i] = 1.0;
        if (sl_dense_solve(M, out, n, n) != 0) { fn_set_error("inv: matrix is singular"); rc = TSR_EARG; }
    }
    free(M); fn_free_doubles(A, L);
    return rc;
}

/* matrix_power(A, p): A**p by binary exponentiation (dense). p<0 inverts first. */
static int r_matrix_power(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("matrix_power: A must be square"); return TSR_EARG; }
    if (nargs < 2) { fn_set_error("matrix_power: power is required"); return TSR_EARG; }
    int64_t p = (args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num;
    int64_t n = args[0].arr.shape[0], L; double *A = fn_arg_doubles(&args[0], &L);
    if (!A) return TSR_ENOMEM;
    int64_t sh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    double *base = (double *)malloc((size_t)(n * n) * sizeof(double));
    double *tmp = (double *)malloc((size_t)(n * n) * sizeof(double));
    int rc = TSR_OK;
    if (!out || !base || !tmp) rc = TSR_ENOMEM;
    else {
        memcpy(base, A, (size_t)(n * n) * sizeof(double));
        if (p < 0) {                                                        /* base <- inv(A) */
            double *M = (double *)malloc((size_t)(n * n) * sizeof(double));
            if (!M) rc = TSR_ENOMEM;
            else {
                memcpy(M, A, (size_t)(n * n) * sizeof(double));
                for (int64_t i = 0; i < n * n; i++) base[i] = 0.0;
                for (int64_t i = 0; i < n; i++) base[i * n + i] = 1.0;
                if (sl_dense_solve(M, base, n, n) != 0) { fn_set_error("matrix_power: matrix is singular"); rc = TSR_EARG; }
                free(M); p = -p;
            }
        }
        if (rc == TSR_OK) {
            for (int64_t i = 0; i < n * n; i++) out[i] = 0.0;             /* out <- identity */
            for (int64_t i = 0; i < n; i++) out[i * n + i] = 1.0;
            while (p > 0) {
                if (p & 1) { spl_mm(tmp, out, base, n); memcpy(out, tmp, (size_t)(n * n) * sizeof(double)); }
                p >>= 1;
                if (p > 0) { spl_mm(tmp, base, base, n); memcpy(base, tmp, (size_t)(n * n) * sizeof(double)); }
            }
        }
    }
    free(base); free(tmp); fn_free_doubles(A, L);
    return rc;
}

/* expm(A): matrix exponential (dense Pade scaling-squaring), the value scipy.sparse.linalg.expm returns. */
static int r_expm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("expm: A must be square"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0], L; double *A = fn_arg_doubles(&args[0], &L);
    if (!A) return TSR_ENOMEM;
    int64_t sh[2] = {n, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    int rc = TSR_OK;
    if (!out) rc = TSR_ENOMEM;
    else if (sl_expm(A, n, out) != 0) { fn_set_error("expm: the matrix exponential failed"); rc = TSR_EARG; }
    fn_free_doubles(A, L);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("splinalg.spsolve", 1, "A, b, permc_spec=None, use_umfpack=True", "x", r_spsolve, NULL, "Solve a sparse linear system A x = b (scipy.sparse.linalg.spsolve)."),
    ROUTINE("splinalg.spsolve_triangular", 1, "A, b, lower=True, overwrite_A=False, overwrite_b=False, unit_diagonal=False", "x", r_spsolve_triangular, NULL, "Solve a triangular sparse system (scipy.sparse.linalg.spsolve_triangular)."),
    ROUTINE("splinalg.norm", 1, "x, ord=None, axis=None", "n", r_norm, NULL, "Norm of a sparse matrix (scipy.sparse.linalg.norm)."),
    ROUTINE("splinalg.spbandwidth", 2, "A", "lower, upper", r_spbandwidth, NULL, "Lower and upper bandwidth of a sparse matrix (scipy.sparse.linalg.spbandwidth)."),
    ROUTINE("splinalg.is_sptriangular", 2, "A", "lower, upper", r_is_sptriangular, NULL, "Whether a sparse matrix is lower/upper triangular (scipy.sparse.linalg.is_sptriangular)."),
    ROUTINE("splinalg.inv", 1, "A", "Ainv", r_inv, NULL, "Inverse of a square sparse matrix, returned dense (scipy.sparse.linalg.inv)."),
    ROUTINE("splinalg.matrix_power", 1, "A, power", "Ap", r_matrix_power, NULL, "Integer matrix power of a square sparse matrix (scipy.sparse.linalg.matrix_power)."),
    ROUTINE("splinalg.expm", 1, "A", "eA", r_expm, NULL, "Matrix exponential of a sparse matrix (scipy.sparse.linalg.expm)."),
};

const fn_table TSR_SCIPY_SPARSE_LINALG_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
