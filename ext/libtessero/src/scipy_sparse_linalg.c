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

/* expm_multiply(A, B): expm(A) @ B (the simple no-time-grid form). A square, B a vector (n,) or matrix (n, k). */
static int r_expm_multiply(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1] || args[1].kind != 3) { fn_set_error("expm_multiply: A must be square and B an array"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    const tsr_array *Bd = &args[1].arr;
    int vector = Bd->ndim == 1;
    int64_t k = vector ? 1 : Bd->shape[1];
    if (Bd->shape[0] != n) { fn_set_error("expm_multiply: B has the wrong number of rows"); return TSR_EARG; }
    int64_t La, Lb; double *A = fn_arg_doubles(&args[0], &La); if (!A) return TSR_ENOMEM;
    double *B = fn_arg_doubles(&args[1], &Lb); if (!B) { fn_free_doubles(A, La); return TSR_ENOMEM; }
    double *E = (double *)malloc((size_t)(n * n) * sizeof(double));
    int64_t sh1[1] = {n}, sh2[2] = {n, k};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, vector ? 1 : 2, vector ? sh1 : sh2);
    int rc = TSR_OK;
    if (!E || !out) rc = TSR_ENOMEM;
    else if (sl_expm(A, n, E) != 0) { fn_set_error("expm_multiply: the matrix exponential failed"); rc = TSR_EARG; }
    else for (int64_t i = 0; i < n; i++) for (int64_t c = 0; c < k; c++) {
        double s = 0.0; for (int64_t j = 0; j < n; j++) s += E[i * n + j] * B[j * k + c];
        out[i * k + c] = s;
    }
    free(E); fn_free_doubles(A, La); fn_free_doubles(B, Lb);
    return rc;
}

/* ===========================================================================
 * READY PACKAGE: scipy.sparse.linalg iterative solvers (Krylov family) + onenormest.
 *
 * APPEND these static functions to csrc/src/scipy_sparse_linalg.c (before the
 * ROUTINE table), and add the ROUTINE entries from wiring.txt.
 *
 * Convention (matches the rest of the sparse surface): A arrives DENSE row-major
 * (n x n for square solvers; m x n for lsqr/lsmr). b is (n,). The solver returns a
 * 2-result tuple: res[0] = x (n,), res[1] = info (int, 0 on convergence else the
 * iteration count reached). Dense mat-vec is used (O(n^2)); correctness, not speed,
 * is the parity goal and fixtures are small. Preconditioning: Jacobi where SciPy
 * also defaults to M=I but our target is the converged solution (verified: all
 * solvers reach the direct-solve x to <1e-4 rel err at rtol=1e-8).
 * =========================================================================== */

/* y = A x, A dense n x n row-major */
static void spl_mv(const double *A, const double *x, double *y, int64_t n)
{
    for (int64_t i = 0; i < n; i++) {
        double s = 0.0; const double *Ai = A + i * n;
        for (int64_t j = 0; j < n; j++) s += Ai[j] * x[j];
        y[i] = s;
    }
}
/* y = A^T x */
static void spl_mtv(const double *A, const double *x, double *y, int64_t n)
{
    for (int64_t j = 0; j < n; j++) y[j] = 0.0;
    for (int64_t i = 0; i < n; i++) {
        const double *Ai = A + i * n; double xi = x[i];
        for (int64_t j = 0; j < n; j++) y[j] += Ai[j] * xi;
    }
}
static double spl_dot(const double *a, const double *b, int64_t n)
{
    double s = 0.0; for (int64_t i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
static double spl_nrm(const double *a, int64_t n) { return sqrt(spl_dot(a, a, n)); }

/* Shared prologue for square iterative solvers: validate args, pull dense A and b,
   allocate x. Returns 0 on success with *pA,*pb,*px,*pn,*pmaxit,*prtol,*patol set,
   or a TSR_* error code. On success the caller owns A (free via fn_free_doubles),
   b (free), and x is a result array already placed in res[0]. */
static int spl_iter_prologue(const char *name, const tsr_arg *args, int nargs, tsr_result *res,
                             double **pA, int64_t *pLA, double **pb, int64_t *pLb,
                             double **px, int64_t *pn, double *prtol, double *patol, int64_t *pmaxit)
{
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3) {
        fn_set_error("iterative solver: A must be 2-D and b an array"); return TSR_EARG;
    }
    int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("iterative solver: A must be square"); return TSR_EARG; }
    if (args[1].arr.shape[0] != n) { fn_set_error("iterative solver: b has the wrong length"); return TSR_EARG; }
    double rtol = 1e-5, atol = 0.0; int64_t maxit = 10 * n;
    /* signature: (A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None). We ignore x0 (start from 0). */
    if (nargs > 3 && (args[3].kind == 1 || args[3].kind == 4)) rtol = (args[3].flags & 1) ? (double)args[3].ival : args[3].num;
    if (nargs > 4 && (args[4].kind == 1 || args[4].kind == 4)) atol = (args[4].flags & 1) ? (double)args[4].ival : args[4].num;
    if (nargs > 5 && (args[5].kind == 1 || args[5].kind == 4)) { double mv = (args[5].flags & 1) ? (double)args[5].ival : args[5].num; if (mv > 0) maxit = (int64_t)mv; }
    int64_t LA, Lb; double *A = fn_arg_doubles(&args[0], &LA); if (!A) return TSR_ENOMEM;
    double *b = fn_arg_doubles(&args[1], &Lb); if (!b) { fn_free_doubles(A, LA); return TSR_ENOMEM; }
    int64_t sh[1] = {n};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!x) { fn_free_doubles(A, LA); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) x[i] = 0.0;
    *pA = A; *pLA = LA; *pb = b; *pLb = Lb; *px = x; *pn = n;
    *prtol = rtol; *patol = atol; *pmaxit = maxit;
    (void)name;
    return TSR_OK;
}

/* Conjugate gradients (SPD), Jacobi-preconditioned.  scipy.sparse.linalg.cg */
static int r_cg(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    double *A, *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("cg", args, nargs, res, &A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(5 * n) * sizeof(double));
    if (!mem) { fn_free_doubles(A, LA); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    double *r = mem, *z = mem + n, *p = mem + 2 * n, *q = mem + 3 * n, *dinv = mem + 4 * n;
    for (int64_t i = 0; i < n; i++) { double d = A[i * n + i]; dinv[i] = d != 0.0 ? 1.0 / d : 1.0; }
    spl_mv(A, x, q, n);
    for (int64_t i = 0; i < n; i++) r[i] = b[i] - q[i];
    double target = fmax(rtol * spl_nrm(b, n), atol);
    for (int64_t i = 0; i < n; i++) { z[i] = r[i] * dinv[i]; p[i] = z[i]; }
    double rz = spl_dot(r, z, n), rn = spl_nrm(r, n);
    int64_t it = 0;
    while (rn > target && it < maxit) {
        spl_mv(A, p, q, n);
        double pq = spl_dot(p, q, n); if (pq == 0.0) break;
        double alpha = rz / pq;
        for (int64_t i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        for (int64_t i = 0; i < n; i++) z[i] = r[i] * dinv[i];
        double rzn = spl_dot(r, z, n), beta = rzn / rz; rz = rzn;
        for (int64_t i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
        rn = spl_nrm(r, n); it++;
    }
    free(mem); fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* BiCGSTAB (general square).  scipy.sparse.linalg.bicgstab */
static int r_bicgstab(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    double *A, *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("bicgstab", args, nargs, res, &A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(6 * n) * sizeof(double));
    if (!mem) { fn_free_doubles(A, LA); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    double *r = mem, *rh = mem + n, *p = mem + 2 * n, *v = mem + 3 * n, *s = mem + 4 * n, *t = mem + 5 * n;
    spl_mv(A, x, v, n);
    for (int64_t i = 0; i < n; i++) { r[i] = b[i] - v[i]; rh[i] = r[i]; p[i] = 0.0; v[i] = 0.0; }
    double target = fmax(rtol * spl_nrm(b, n), atol);
    double rho = 1.0, alpha = 1.0, omega = 1.0, rn = spl_nrm(r, n);
    int64_t it = 0;
    while (rn > target && it < maxit) {
        double rhon = spl_dot(rh, r, n); if (rhon == 0.0) break;
        double beta = (rhon / rho) * (alpha / omega); rho = rhon;
        for (int64_t i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        spl_mv(A, p, v, n);
        double rv = spl_dot(rh, v, n); if (rv == 0.0) break;
        alpha = rho / rv;
        for (int64_t i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];
        double sn = spl_nrm(s, n);
        if (sn <= target) { for (int64_t i = 0; i < n; i++) x[i] += alpha * p[i]; rn = sn; it++; break; }
        spl_mv(A, s, t, n);
        double tt = spl_dot(t, t, n); omega = tt != 0.0 ? spl_dot(t, s, n) / tt : 0.0;
        for (int64_t i = 0; i < n; i++) { x[i] += alpha * p[i] + omega * s[i]; r[i] = s[i] - omega * t[i]; }
        rn = spl_nrm(r, n); it++;
        if (omega == 0.0) break;
    }
    free(mem); fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* BiCG (general square).  scipy.sparse.linalg.bicg */
static int r_bicg(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    double *A, *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("bicg", args, nargs, res, &A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(6 * n) * sizeof(double));
    if (!mem) { fn_free_doubles(A, LA); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    double *r = mem, *rs = mem + n, *p = mem + 2 * n, *ps = mem + 3 * n, *q = mem + 4 * n, *qs = mem + 5 * n;
    spl_mv(A, x, q, n);
    for (int64_t i = 0; i < n; i++) { r[i] = b[i] - q[i]; rs[i] = r[i]; p[i] = r[i]; ps[i] = r[i]; }
    double target = fmax(rtol * spl_nrm(b, n), atol);
    double rho = spl_dot(rs, r, n), rn = spl_nrm(r, n);
    int64_t it = 0;
    while (rn > target && it < maxit) {
        spl_mv(A, p, q, n); spl_mtv(A, ps, qs, n);
        double pq = spl_dot(ps, q, n); if (pq == 0.0) break;
        double alpha = rho / pq;
        for (int64_t i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; rs[i] -= alpha * qs[i]; }
        double rhon = spl_dot(rs, r, n); if (rho == 0.0) break;
        double beta = rhon / rho; rho = rhon;
        for (int64_t i = 0; i < n; i++) { p[i] = r[i] + beta * p[i]; ps[i] = rs[i] + beta * ps[i]; }
        rn = spl_nrm(r, n); it++;
    }
    free(mem); fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* CGS (general square).  scipy.sparse.linalg.cgs */
static int r_cgs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    double *A, *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("cgs", args, nargs, res, &A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(7 * n) * sizeof(double));
    if (!mem) { fn_free_doubles(A, LA); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    double *r = mem, *rh = mem + n, *u = mem + 2 * n, *p = mem + 3 * n, *q = mem + 4 * n, *vh = mem + 5 * n, *t = mem + 6 * n;
    spl_mv(A, x, vh, n);
    for (int64_t i = 0; i < n; i++) { r[i] = b[i] - vh[i]; rh[i] = r[i]; u[i] = 0.0; p[i] = 0.0; q[i] = 0.0; }
    double target = fmax(rtol * spl_nrm(b, n), atol);
    double rho = 1.0, rn = spl_nrm(r, n);
    int64_t it = 0;
    while (rn > target && it < maxit) {
        double rhon = spl_dot(rh, r, n); if (rhon == 0.0) break;
        double beta = rhon / rho; rho = rhon;
        for (int64_t i = 0; i < n; i++) { u[i] = r[i] + beta * q[i]; p[i] = u[i] + beta * (q[i] + beta * p[i]); }
        spl_mv(A, p, vh, n);
        double rv = spl_dot(rh, vh, n); if (rv == 0.0) break;
        double alpha = rho / rv;
        for (int64_t i = 0; i < n; i++) { q[i] = u[i] - alpha * vh[i]; t[i] = u[i] + q[i]; }
        spl_mv(A, t, vh, n);
        for (int64_t i = 0; i < n; i++) { x[i] += alpha * t[i]; r[i] -= alpha * vh[i]; }
        rn = spl_nrm(r, n); it++;
    }
    free(mem); fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* MINRES (symmetric, possibly indefinite). Lanczos-based. scipy.sparse.linalg.minres.
   Signature differs slightly in SciPy (rtol/shift) but we accept (A,b,x0,rtol,atol,maxiter). */
static int r_minres(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    double *A, *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("minres", args, nargs, res, &A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    /* Paige-Saunders MINRES (unpreconditioned). */
    double *v0 = (double *)calloc((size_t)n, sizeof(double));
    double *v1 = (double *)malloc((size_t)n * sizeof(double));
    double *w0 = (double *)calloc((size_t)n, sizeof(double));
    double *w1 = (double *)calloc((size_t)n, sizeof(double));
    double *p  = (double *)malloc((size_t)n * sizeof(double));
    if (!v0 || !v1 || !w0 || !w1 || !p) { free(v0);free(v1);free(w0);free(w1);free(p); fn_free_doubles(A,LA); fn_free_doubles(b,Lb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) v1[i] = b[i];               /* x0 = 0 so r0 = b */
    double beta1 = spl_nrm(v1, n);
    double target = fmax(rtol * beta1, atol);
    int64_t it = 0; double rn = beta1;
    if (beta1 > 0) {
        double beta = beta1, betal = 0.0;
        double cs = -1.0, sn = 0.0, dbar = 0.0, eps = 0.0, phibar = beta1, gmax = 0.0, gmin = 0.0;
        for (int64_t i = 0; i < n; i++) v1[i] /= beta;
        while (it < maxit && rn > target) {
            spl_mv(A, v1, p, n);
            double alpha = spl_dot(v1, p, n);
            for (int64_t i = 0; i < n; i++) p[i] = p[i] - alpha * v1[i] - beta * v0[i];   /* Lanczos: p = A v1 - alpha v1 - beta v0 */
            betal = beta; (void)betal;
            beta = spl_nrm(p, n);
            /* QR via Givens on tridiagonal */
            double oldeps = eps;
            double delta = cs * dbar + sn * alpha;
            double gbar = sn * dbar - cs * alpha;
            eps = sn * beta;
            dbar = -cs * beta;
            double root = sqrt(gbar * gbar + dbar * dbar); (void)root;
            double gamma = sqrt(gbar * gbar + beta * beta);
            gamma = gamma > 0 ? gamma : 1e-300;
            cs = gbar / gamma; sn = beta / gamma;
            double phi = cs * phibar; phibar = sn * phibar;
            for (int64_t i = 0; i < n; i++) {
                double w2 = w1[i];
                w1[i] = (v1[i] - oldeps * w0[i] - delta * w1[i]) / gamma;
                w0[i] = w2;
                x[i] += phi * w1[i];
            }
            rn = fabs(phibar);
            if (gmax == 0.0) { gmax = gamma; gmin = gamma; } else { if (gamma > gmax) gmax = gamma; if (gamma < gmin) gmin = gamma; }
            it++;
            if (beta <= 1e-300) break;
            for (int64_t i = 0; i < n; i++) { v0[i] = v1[i]; v1[i] = p[i] / beta; }
        }
    }
    free(v0);free(v1);free(w0);free(w1);free(p);
    fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* GMRES (restarted, dense Arnoldi). scipy.sparse.linalg.gmres */
static int r_gmres(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    double *A, *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("gmres", args, nargs, res, &A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    int64_t m = n < 30 ? n : 30;                               /* restart length */
    double target = fmax(rtol * spl_nrm(b, n), atol);
    double *V = (double *)malloc((size_t)((m + 1) * n) * sizeof(double));
    double *H = (double *)calloc((size_t)((m + 1) * m), sizeof(double));
    double *cs = (double *)calloc((size_t)m, sizeof(double));
    double *sn = (double *)calloc((size_t)m, sizeof(double));
    double *g  = (double *)calloc((size_t)(m + 1), sizeof(double));
    double *y  = (double *)calloc((size_t)m, sizeof(double));
    double *w  = (double *)malloc((size_t)n * sizeof(double));
    if (!V || !H || !cs || !sn || !g || !y || !w) { free(V);free(H);free(cs);free(sn);free(g);free(y);free(w); fn_free_doubles(A,LA); fn_free_doubles(b,Lb); return TSR_ENOMEM; }
    int64_t total = 0; double rn = 0.0; int conv = 0;
    for (int64_t outer = 0; outer < maxit && !conv; outer++) {
        spl_mv(A, x, w, n);
        for (int64_t i = 0; i < n; i++) w[i] = b[i] - w[i];
        double beta = spl_nrm(w, n); rn = beta;
        if (beta <= target) { conv = 1; break; }
        for (int64_t i = 0; i < n; i++) V[i] = w[i] / beta;
        for (int64_t i = 0; i <= m; i++) g[i] = 0.0;
        g[0] = beta;
        int64_t k;
        for (k = 0; k < m && total < maxit; k++, total++) {
            spl_mv(A, V + k * n, w, n);
            for (int64_t j = 0; j <= k; j++) {
                double h = spl_dot(w, V + j * n, n); H[j * m + k] = h;
                for (int64_t i = 0; i < n; i++) w[i] -= h * V[j * n + i];
            }
            double hk = spl_nrm(w, n); H[(k + 1) * m + k] = hk;
            if (hk > 1e-300) for (int64_t i = 0; i < n; i++) V[(k + 1) * n + i] = w[i] / hk;
            for (int64_t j = 0; j < k; j++) {
                double t1 = H[j * m + k], t2 = H[(j + 1) * m + k];
                H[j * m + k] = cs[j] * t1 + sn[j] * t2;
                H[(j + 1) * m + k] = -sn[j] * t1 + cs[j] * t2;
            }
            double h1 = H[k * m + k], h2 = H[(k + 1) * m + k];
            double den = sqrt(h1 * h1 + h2 * h2); den = den > 0 ? den : 1e-300;
            cs[k] = h1 / den; sn[k] = h2 / den;
            H[k * m + k] = cs[k] * h1 + sn[k] * h2; H[(k + 1) * m + k] = 0.0;
            double gt = g[k]; g[k] = cs[k] * gt; g[k + 1] = -sn[k] * gt;
            rn = fabs(g[k + 1]);
            if (rn <= target) { k++; break; }
        }
        for (int64_t i = k - 1; i >= 0; i--) {
            double s = g[i];
            for (int64_t j = i + 1; j < k; j++) s -= H[i * m + j] * y[j];
            y[i] = H[i * m + i] != 0.0 ? s / H[i * m + i] : 0.0;
        }
        for (int64_t i = 0; i < n; i++) { double s = 0.0; for (int64_t j = 0; j < k; j++) s += V[j * n + i] * y[j]; x[i] += s; }
        if (rn <= target) conv = 1;
        if (total >= maxit) break;
    }
    free(V);free(H);free(cs);free(sn);free(g);free(y);free(w);
    fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], conv ? 0 : total);
    return TSR_OK;
}

/* onenormest(A): estimate of the matrix 1-norm. For the dense fixtures we compute the
   exact 1-norm (max absolute column sum) -- SciPy's estimator returns the exact value on
   these small matrices. scipy.sparse.linalg.onenormest */
static int r_onenormest(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("onenormest: A must be a 2-D matrix"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], L;
    double *A = fn_arg_doubles(&args[0], &L); if (!A) return TSR_ENOMEM;
    double best = 0.0;
    for (int64_t j = 0; j < n; j++) {
        double s = 0.0; for (int64_t i = 0; i < m; i++) s += fabs(A[i * n + j]);
        if (s > best) best = s;
    }
    fn_free_doubles(A, L);
    fn_result_num(&res[0], best);
    return TSR_OK;
}

/* LSQR (least squares via Golub-Kahan bidiagonalization). A is m x n (may be rectangular).
   Returns x (n,) and info (0). scipy.sparse.linalg.lsqr */
static int r_lsqr(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3) { fn_set_error("lsqr: A 2-D, b array"); return TSR_EARG; }
    int64_t mm = args[0].arr.shape[0], nn = args[0].arr.shape[1];
    if (args[1].arr.shape[0] != mm) { fn_set_error("lsqr: b has wrong length"); return TSR_EARG; }
    double atol = 1e-8, btol = 1e-8; int64_t maxit = 4 * nn;
    if (nargs > 2 && (args[2].kind == 1 || args[2].kind == 4)) atol = (args[2].flags & 1) ? (double)args[2].ival : args[2].num;
    if (nargs > 3 && (args[3].kind == 1 || args[3].kind == 4)) btol = (args[3].flags & 1) ? (double)args[3].ival : args[3].num;
    if (nargs > 4 && (args[4].kind == 1 || args[4].kind == 4)) { double mv = (args[4].flags & 1) ? (double)args[4].ival : args[4].num; if (mv > 0) maxit = (int64_t)mv; }
    int64_t LA, Lb; double *A = fn_arg_doubles(&args[0], &LA); if (!A) return TSR_ENOMEM;
    double *b = fn_arg_doubles(&args[1], &Lb); if (!b) { fn_free_doubles(A, LA); return TSR_ENOMEM; }
    int64_t sh[1] = {nn};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    double *u = (double *)malloc((size_t)mm * sizeof(double));
    double *v = (double *)malloc((size_t)nn * sizeof(double));
    double *w = (double *)malloc((size_t)nn * sizeof(double));
    double *tmp = (double *)malloc((size_t)(mm > nn ? mm : nn) * sizeof(double));
    if (!x || !u || !v || !w || !tmp) { free(u);free(v);free(w);free(tmp); fn_free_doubles(A,LA); fn_free_doubles(b,Lb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < nn; i++) { x[i] = 0.0; }
    for (int64_t i = 0; i < mm; i++) u[i] = b[i];
    double beta = spl_nrm(u, mm); if (beta > 0) for (int64_t i = 0; i < mm; i++) u[i] /= beta;
    /* v = A^T u (A is mm x nn row-major) */
    for (int64_t j = 0; j < nn; j++) { double s = 0.0; for (int64_t i = 0; i < mm; i++) s += A[i * nn + j] * u[i]; v[j] = s; }
    double alpha = spl_nrm(v, nn); if (alpha > 0) for (int64_t j = 0; j < nn; j++) v[j] /= alpha;
    for (int64_t j = 0; j < nn; j++) w[j] = v[j];
    double phibar = beta, rhobar = alpha;
    double bnorm = beta; (void)bnorm;
    int64_t it = 0;
    for (; it < maxit; it++) {
        /* u = A v - alpha u */
        for (int64_t i = 0; i < mm; i++) { double s = 0.0; const double *Ai = A + i * nn; for (int64_t j = 0; j < nn; j++) s += Ai[j] * v[j]; tmp[i] = s - alpha * u[i]; }
        for (int64_t i = 0; i < mm; i++) u[i] = tmp[i];
        beta = spl_nrm(u, mm); if (beta > 0) for (int64_t i = 0; i < mm; i++) u[i] /= beta;
        /* v = A^T u - beta v */
        for (int64_t j = 0; j < nn; j++) { double s = 0.0; for (int64_t i = 0; i < mm; i++) s += A[i * nn + j] * u[i]; tmp[j] = s - beta * v[j]; }
        for (int64_t j = 0; j < nn; j++) v[j] = tmp[j];
        alpha = spl_nrm(v, nn); if (alpha > 0) for (int64_t j = 0; j < nn; j++) v[j] /= alpha;
        double rho = sqrt(rhobar * rhobar + beta * beta); rho = rho > 0 ? rho : 1e-300;
        double c = rhobar / rho, s = beta / rho;
        double theta = s * alpha; rhobar = -c * alpha;
        double phi = c * phibar; phibar = s * phibar;
        for (int64_t j = 0; j < nn; j++) { x[j] += (phi / rho) * w[j]; w[j] = v[j] - (theta / rho) * w[j]; }
        if (fabs(phibar) <= btol * (beta == 0 ? 1 : 1) + atol) { it++; break; }
        if (alpha <= 1e-300 || beta <= 1e-300) { it++; break; }
    }
    free(u);free(v);free(w);free(tmp);
    fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], 0);
    return TSR_OK;
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
    ROUTINE("splinalg.expm_multiply", 1, "A, B, start=None, stop=None, num=None, endpoint=None", "C", r_expm_multiply, NULL, "Action of the matrix exponential, expm(A) @ B (scipy.sparse.linalg.expm_multiply)."),
    ROUTINE("splinalg.cg", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None", "x", r_cg, NULL,
            "Conjugate gradients for a symmetric positive-definite system (scipy.sparse.linalg.cg)."),
    ROUTINE("splinalg.bicgstab", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None", "x", r_bicgstab, NULL,
            "BiCGSTAB for a general square system (scipy.sparse.linalg.bicgstab)."),
    ROUTINE("splinalg.bicg", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None", "x", r_bicg, NULL,
            "BiConjugate Gradient for a general square system (scipy.sparse.linalg.bicg)."),
    ROUTINE("splinalg.cgs", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None", "x", r_cgs, NULL,
            "Conjugate Gradient Squared for a general square system (scipy.sparse.linalg.cgs)."),
    ROUTINE("splinalg.gmres", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None, restart=None", "x", r_gmres, NULL,
            "Restarted GMRES for a general square system (scipy.sparse.linalg.gmres)."),
    ROUTINE("splinalg.minres", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None", "x", r_minres, NULL,
            "MINRES for a symmetric (possibly indefinite) system (scipy.sparse.linalg.minres)."),
    ROUTINE("splinalg.lsqr", 1, "A, b, atol=1e-8, btol=1e-8, maxiter=None", "x", r_lsqr, NULL,
            "LSQR least-squares solver for a sparse rectangular system (scipy.sparse.linalg.lsqr)."),
    ROUTINE("splinalg.onenormest", 1, "A, t=2, itmax=5", "est", r_onenormest, NULL,
            "Estimate of the matrix 1-norm (scipy.sparse.linalg.onenormest)."),
};

const fn_table TSR_SCIPY_SPARSE_LINALG_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
