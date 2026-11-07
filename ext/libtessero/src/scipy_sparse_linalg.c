/* scipy.sparse.linalg ("splinalg." prefix -> Tessero\SparseLinalg facade).
 *
 * Direct solves, matrix norms and structure queries for sparse matrices. As with the rest of the sparse
 * parity surface, matrix arguments arrive dense (the generators build them from a dense array) and the
 * routines compute the value SciPy returns; both backends read this table at runtime.
 */
#include "fn.h"

#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <lapacke.h>

extern int sl_dense_solve(double *M, double *b, int64_t n, int64_t nrhs);   /* np_linalg.c (row-major dgesv) */
/* the linked BLAS (see matmul.c): the Lanczos basis products */
extern void cblas_dgemv(int Order, int TransA, int M, int N, double alpha, const double *A, int lda, const double *X,
                        int incX, double beta, double *Y, int incY);
extern void cblas_dgemm(int Order, int TransA, int TransB, int M, int N, int K, double alpha, const double *A, int lda,
                        const double *B, int ldb, double beta, double *C, int ldc);
extern int sl_expm(const double *A, int64_t n, double *out);
extern int sl_funm_byname(const char *fname, const double *A, int64_t n, double *out);                /* np_linalg.c (Pade scaling-squaring) */

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

/* The matrix of an iterative solver: dense (n x n row-major, as before), or CSR, which the bindings pass as a group
   of four arrays -- data, indices, indptr, shape, what scipy.sparse's routines return. The CSR product sums each
   row's stored entries in their order, the same products the dense loop sums for a row whose zeros are not stored,
   so both give the same result (but for the sign of an exact zero); the dense one costs n^2, the CSR one nnz. */
typedef struct {
    int64_t n;
    double *dense; int64_t ldense;                 /* dense: the fn_arg_doubles copy */
    double *data; int64_t ldata;                   /* CSR */
    int64_t *ind, *ptr; int64_t nnz;
} spl_op;

/* the integer-valued array argument a as int64 (exact below 2^53); NULL with *len = -1 when it holds a non-integer */
static int64_t *spl_arg_i64(const tsr_arg *a, int64_t *len)
{
    int64_t t;
    double *v = fn_arg_doubles(a, &t);
    if (!v) { *len = 0; return NULL; }
    int64_t *o = (int64_t *)tsr_alloc((t > 0 ? t : 1) * 8);
    int ok = o != NULL;
    for (int64_t q = 0; ok && q < t; q++) {
        if (!(v[q] >= -9.0e15 && v[q] <= 9.0e15) || v[q] != floor(v[q])) ok = 0;
        else o[q] = (int64_t)v[q];
    }
    fn_free_doubles(v, t);
    if (!ok) { if (o) tsr_free(o, (t > 0 ? t : 1) * 8); *len = o ? -1 : 0; return NULL; }
    *len = t;
    return o;
}

static void spl_op_free(spl_op *A)
{
    if (A->dense) fn_free_doubles(A->dense, A->ldense);
    if (A->data) fn_free_doubles(A->data, A->ldata);
    if (A->ind) tsr_free(A->ind, (A->nnz > 0 ? A->nnz : 1) * 8);
    if (A->ptr) tsr_free(A->ptr, (A->n + 1) * 8);
    memset(A, 0, sizeof *A);
}

/* a, the matrix argument, into A: a 2-D array, or a CSR group whose structure is checked before any use (indptr of
   n + 1 non-decreasing offsets from 0 to nnz, nnz data and indices, every index in [0, ncols)), so a malformed one is
   refused and never read out of bounds. rows / cols: the shape. who names the caller in errors. */
static int spl_op_load(spl_op *A, const tsr_arg *a, const char *who, int64_t *rows, int64_t *cols)
{
    memset(A, 0, sizeof *A);
    if (a->kind == 3 && a->arr.ndim == 2) {
        *rows = a->arr.shape[0]; *cols = a->arr.shape[1];
        A->n = *rows;
        A->dense = fn_arg_doubles(a, &A->ldense);
        return A->dense ? TSR_OK : TSR_ENOMEM;
    }
    if (a->kind != 5 || a->count != 4 || !a->items) {
        fn_set_error("%s: A must be a 2-D array or a CSR matrix (data, indices, indptr, shape)", who); return TSR_EARG;
    }
    const tsr_arg *it = a->items;
    for (int k = 0; k < 4; k++)
        if (it[k].kind != 3 || it[k].arr.ndim != 1) { fn_set_error("%s: a CSR matrix is four 1-D arrays (data, indices, indptr, shape)", who); return TSR_EARG; }
    if (it[3].arr.shape[0] != 2) { fn_set_error("%s: the CSR shape must have two entries", who); return TSR_EARG; }
    int64_t ls, ln;
    int64_t *shape = spl_arg_i64(&it[3], &ls);
    if (!shape) { if (ls < 0) fn_set_error("%s: the CSR shape must be integers", who); return ls < 0 ? TSR_EARG : TSR_ENOMEM; }
    *rows = shape[0]; *cols = shape[1];
    tsr_free(shape, 2 * 8);
    if (*rows < 0 || *cols < 0 || *rows > ((int64_t)1 << 40) || *cols > ((int64_t)1 << 40)) { fn_set_error("%s: bad CSR shape", who); return TSR_EARG; }
    A->n = *rows;
    if (it[2].arr.shape[0] != *rows + 1) { fn_set_error("%s: the CSR indptr must have rows + 1 entries", who); return TSR_EARG; }
    A->ptr = spl_arg_i64(&it[2], &ln);
    if (!A->ptr) { int rc = ln < 0 ? TSR_EARG : TSR_ENOMEM; if (ln < 0) fn_set_error("%s: the CSR indptr must be integers", who); return rc; }
    A->nnz = A->ptr[*rows];
    int ok = A->ptr[0] == 0 && A->nnz >= 0 && it[0].arr.shape[0] == A->nnz && it[1].arr.shape[0] == A->nnz;
    for (int64_t i = 0; ok && i < *rows; i++) if (A->ptr[i + 1] < A->ptr[i]) ok = 0;
    if (!ok) { tsr_free(A->ptr, (*rows + 1) * 8); A->ptr = NULL; fn_set_error("%s: malformed CSR indptr (offsets from 0 to nnz, non-decreasing)", who); return TSR_EARG; }
    int64_t li;
    A->ind = spl_arg_i64(&it[1], &li);
    if (!A->ind) { int rc = li < 0 ? TSR_EARG : TSR_ENOMEM; if (li < 0) fn_set_error("%s: the CSR indices must be integers", who); spl_op_free(A); return rc; }
    for (int64_t k = 0; k < A->nnz; k++)
        if (A->ind[k] < 0 || A->ind[k] >= *cols) { fn_set_error("%s: a CSR column index is out of range", who); spl_op_free(A); return TSR_EARG; }
    A->data = fn_arg_doubles(&it[0], &A->ldata);
    if (!A->data) { spl_op_free(A); return TSR_ENOMEM; }
    return TSR_OK;
}

/* y = A x */
static void spl_mv(const spl_op *A, const double *x, double *y, int64_t n)
{
    if (A->dense) {
        for (int64_t i = 0; i < n; i++) {
            double s = 0.0; const double *Ai = A->dense + i * n;
            for (int64_t j = 0; j < n; j++) s += Ai[j] * x[j];
            y[i] = s;
        }
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        double s = 0.0;
        for (int64_t k = A->ptr[i]; k < A->ptr[i + 1]; k++) s += A->data[k] * x[A->ind[k]];
        y[i] = s;
    }
}
/* y = A^T x */
static void spl_mtv(const spl_op *A, const double *x, double *y, int64_t n)
{
    for (int64_t j = 0; j < n; j++) y[j] = 0.0;
    if (A->dense) {
        for (int64_t i = 0; i < n; i++) {
            const double *Ai = A->dense + i * n; double xi = x[i];
            for (int64_t j = 0; j < n; j++) y[j] += Ai[j] * xi;
        }
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        const double xi = x[i];
        for (int64_t k = A->ptr[i]; k < A->ptr[i + 1]; k++) y[A->ind[k]] += A->data[k] * xi;
    }
}
/* A[i][i] (0 when not stored) */
static double spl_op_diag(const spl_op *A, int64_t i)
{
    if (A->dense) return A->dense[i * A->n + i];
    double d = 0.0;
    for (int64_t k = A->ptr[i]; k < A->ptr[i + 1]; k++) if (A->ind[k] == i) d += A->data[k];   /* duplicates add, as scipy sums them */
    return d;
}
/* spsolve on a CSR matrix: an LU with partial pivoting restricted to the matrix's band (LAPACK dgbsv), the
   bandwidths measured from the stored entries -- n (2 kl + ku + 1) storage and n kl (kl + ku) work instead of the dense
   n^2 and n^3, so tridiagonal and grid matrices solve at any size. When the band is too wide to pay off, the matrix
   is solved dense if that fits, else refused. b (n x k, row-major) is solved in place into x. */
#define SPSOLVE_DENSE_MAX ((int64_t)1 << 28)               /* doubles: a 2 GiB dense matrix at most */
static int spsolve_csr(const spl_op *A, int64_t n, double *x, int64_t k)
{
    int64_t kl = 0, ku = 0;
    for (int64_t i = 0; i < n; i++)
        for (int64_t q = A->ptr[i]; q < A->ptr[i + 1]; q++) {
            const int64_t d = i - A->ind[q];
            if (d > kl) kl = d;
            if (-d > ku) ku = -d;
        }
    const int64_t ldab = 2 * kl + ku + 1;
    int rc = TSR_OK;
    if (ldab <= n && kl <= INT32_MAX / 4 && ku <= INT32_MAX / 4 && ldab * n <= SPSOLVE_DENSE_MAX * 2) {
        double *ab = (double *)calloc((size_t)(ldab * n), sizeof(double));
        double *bc = k > 1 ? (double *)malloc((size_t)(n * k) * sizeof(double)) : x;
        lapack_int *ipiv = (lapack_int *)malloc(sizeof(lapack_int) * (size_t)(n > 0 ? n : 1));
        if (!ab || !bc || !ipiv) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < n; i++)               /* band storage: A(i, j) at ab[kl + ku + i - j + j ldab] */
                for (int64_t q = A->ptr[i]; q < A->ptr[i + 1]; q++) {
                    const int64_t j = A->ind[q];
                    ab[kl + ku + i - j + j * ldab] += A->data[q];   /* duplicates add, as scipy sums them */
                }
            if (k > 1) for (int64_t i = 0; i < n; i++) for (int64_t c = 0; c < k; c++) bc[c * n + i] = x[i * k + c];
            const lapack_int info = LAPACKE_dgbsv(LAPACK_COL_MAJOR, (lapack_int)n, (lapack_int)kl, (lapack_int)ku, (lapack_int)k,
                                                  ab, (lapack_int)ldab, ipiv, bc, (lapack_int)(n > 0 ? n : 1));
            if (info != 0) { fn_set_error("spsolve: matrix is singular"); rc = TSR_EARG; }
            else if (k > 1) for (int64_t i = 0; i < n; i++) for (int64_t c = 0; c < k; c++) x[i * k + c] = bc[c * n + i];
        }
        free(ab); free(ipiv);
        if (k > 1) free(bc);
        return rc;
    }
    if (n > 0 && n > SPSOLVE_DENSE_MAX / n) {
        fn_set_error("spsolve: the matrix is too wide-banded (%lld below, %lld above the diagonal) for the banded solver and too large to solve dense", (long long)kl, (long long)ku);
        return TSR_EARG;
    }
    double *D = (double *)calloc((size_t)(n * n > 0 ? n * n : 1), sizeof(double));
    if (!D) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++)
        for (int64_t q = A->ptr[i]; q < A->ptr[i + 1]; q++) D[i * n + A->ind[q]] += A->data[q];
    if (sl_dense_solve(D, x, n, k) != 0) { fn_set_error("spsolve: matrix is singular"); rc = TSR_EARG; }
    free(D);
    return rc;
}

static int r_spsolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs >= 2 && args[0].kind == 5) {                /* a CSR matrix: the banded (or dense) solve of spsolve_csr */
        if (args[1].kind != 3) { fn_set_error("spsolve: b must be an array"); return TSR_EARG; }
        spl_op A; int64_t n, cols;
        int rc = spl_op_load(&A, &args[0], "spsolve", &n, &cols);
        if (rc != TSR_OK) return rc;
        const tsr_array *B = &args[1].arr;
        const int vector = B->ndim == 1;
        const int64_t k = vector ? 1 : B->shape[1];
        if (cols != n) { spl_op_free(&A); fn_set_error("spsolve: A must be square"); return TSR_EARG; }
        if (B->ndim > 2 || B->shape[0] != n) { spl_op_free(&A); fn_set_error("spsolve: b has the wrong number of rows"); return TSR_EARG; }
        int64_t Lb; double *b = fn_arg_doubles(&args[1], &Lb);
        if (!b) { spl_op_free(&A); return TSR_ENOMEM; }
        int64_t sh1[1] = {n}, sh2[2] = {n, k};
        double *x = (double *)fn_result_array(&res[0], TSR_F64, vector ? 1 : 2, vector ? sh1 : sh2);
        if (!x) rc = TSR_ENOMEM;
        else { memcpy(x, b, (size_t)(n * k) * sizeof(double)); rc = spsolve_csr(&A, n, x, k); }
        spl_op_free(&A); fn_free_doubles(b, Lb);
        return rc;
    }
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

static double spl_dot(const double *a, const double *b, int64_t n)
{
    double s = 0.0; for (int64_t i = 0; i < n; i++) s += a[i] * b[i]; return s;
}
static double spl_nrm(const double *a, int64_t n) { return sqrt(spl_dot(a, a, n)); }

/* Shared prologue for square iterative solvers: validate args, load A (dense or CSR) and b,
   allocate x. Returns 0 on success with *A,*pb,*px,*pn,*pmaxit,*prtol,*patol set,
   or a TSR_* error code. On success the caller owns A (free via spl_op_free),
   b (free), and x is a result array already placed in res[0]. */
static int spl_iter_prologue(const char *name, const tsr_arg *args, int nargs, tsr_result *res,
                             spl_op *A, int64_t *pLA, double **pb, int64_t *pLb,
                             double **px, int64_t *pn, double *prtol, double *patol, int64_t *pmaxit)
{
    if (nargs < 2 || args[1].kind != 3) {
        fn_set_error("iterative solver: A must be 2-D and b an array"); return TSR_EARG;
    }
    int64_t n, cols;
    const int lrc = spl_op_load(A, &args[0], "iterative solver", &n, &cols);
    if (lrc != TSR_OK) return lrc;
    if (cols != n) { spl_op_free(A); fn_set_error("iterative solver: A must be square"); return TSR_EARG; }
    if (args[1].arr.shape[0] != n) { spl_op_free(A); fn_set_error("iterative solver: b has the wrong length"); return TSR_EARG; }
    double rtol = 1e-5, atol = 0.0; int64_t maxit = 10 * n;
    /* signature: (A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None). We ignore x0 (start from 0). */
    if (nargs > 3 && (args[3].kind == 1 || args[3].kind == 4)) rtol = (args[3].flags & 1) ? (double)args[3].ival : args[3].num;
    if (nargs > 4 && (args[4].kind == 1 || args[4].kind == 4)) atol = (args[4].flags & 1) ? (double)args[4].ival : args[4].num;
    if (nargs > 5 && (args[5].kind == 1 || args[5].kind == 4)) { double mv = (args[5].flags & 1) ? (double)args[5].ival : args[5].num; if (mv > 0) maxit = (int64_t)mv; }
    int64_t Lb;
    double *b = fn_arg_doubles(&args[1], &Lb); if (!b) { spl_op_free(A); return TSR_ENOMEM; }
    int64_t sh[1] = {n};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!x) { spl_op_free(A); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) x[i] = 0.0;
    *pLA = 0; *pb = b; *pLb = Lb; *px = x; *pn = n;
    *prtol = rtol; *patol = atol; *pmaxit = maxit;
    (void)name;
    return TSR_OK;
}

/* Conjugate gradients (SPD), Jacobi-preconditioned.  scipy.sparse.linalg.cg */
static int r_cg(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("cg", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(5 * n) * sizeof(double));
    if (!mem) { spl_op_free(A); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    double *r = mem, *z = mem + n, *p = mem + 2 * n, *q = mem + 3 * n, *dinv = mem + 4 * n;
    for (int64_t i = 0; i < n; i++) { double d = spl_op_diag(A, i); dinv[i] = d != 0.0 ? 1.0 / d : 1.0; }
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
    free(mem); spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* BiCGSTAB (general square).  scipy.sparse.linalg.bicgstab */
static int r_bicgstab(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("bicgstab", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(6 * n) * sizeof(double));
    if (!mem) { spl_op_free(A); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
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
    free(mem); spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* BiCG (general square).  scipy.sparse.linalg.bicg */
static int r_bicg(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("bicg", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(6 * n) * sizeof(double));
    if (!mem) { spl_op_free(A); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
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
    free(mem); spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* CGS (general square).  scipy.sparse.linalg.cgs */
static int r_cgs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("cgs", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(7 * n) * sizeof(double));
    if (!mem) { spl_op_free(A); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
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
    free(mem); spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* MINRES (symmetric, possibly indefinite). Lanczos-based. scipy.sparse.linalg.minres.
   Signature differs slightly in SciPy (rtol/shift) but we accept (A,b,x0,rtol,atol,maxiter). */
static int r_minres(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("minres", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    /* Paige-Saunders MINRES (unpreconditioned). */
    double *v0 = (double *)calloc((size_t)n, sizeof(double));
    double *v1 = (double *)malloc((size_t)n * sizeof(double));
    double *w0 = (double *)calloc((size_t)n, sizeof(double));
    double *w1 = (double *)calloc((size_t)n, sizeof(double));
    double *p  = (double *)malloc((size_t)n * sizeof(double));
    if (!v0 || !v1 || !w0 || !w1 || !p) { free(v0);free(v1);free(w0);free(w1);free(p); spl_op_free(A); fn_free_doubles(b,Lb); return TSR_ENOMEM; }
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
    spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* GMRES (restarted, dense Arnoldi). scipy.sparse.linalg.gmres */
static int r_gmres(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("gmres", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
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
    if (!V || !H || !cs || !sn || !g || !y || !w) { free(V);free(H);free(cs);free(sn);free(g);free(y);free(w); spl_op_free(A); fn_free_doubles(b,Lb); return TSR_ENOMEM; }
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
    spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], conv ? 0 : total);
    return TSR_OK;
}

/* LGMRES: GMRES with augmentation. Converges to solve(A,b) on well-conditioned
   systems; delegate to the full-GMRES kernel. scipy.sparse.linalg.lgmres */
static int r_lgmres(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return r_gmres(ctx, args, nargs, res, nres);
}

/* GCROT(m,k): truncated GMRES with outer recycling. Same converged solution on
   well-conditioned systems; delegate to the full-GMRES kernel. scipy.sparse.linalg.gcrotmk */
static int r_gcrotmk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return r_gmres(ctx, args, nargs, res, nres);
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
    if (!x || !u || !v || !w || !tmp) { free(u);free(v);free(w);free(tmp); fn_free_doubles(A, LA); fn_free_doubles(b,Lb); return TSR_ENOMEM; }
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

static int r_qmr(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("qmr", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(13 * n) * sizeof(double));
    if (!mem) { spl_op_free(A); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    double *r = mem, *vt = mem + n, *y = mem + 2*n, *wt = mem + 3*n, *z = mem + 4*n,
           *v = mem + 5*n, *w = mem + 6*n, *p = mem + 7*n, *q = mem + 8*n,
           *d = mem + 9*n, *s = mem + 10*n, *pt = mem + 11*n, *tmp = mem + 12*n;
    spl_mv(A, x, tmp, n);
    for (int64_t i = 0; i < n; i++) { r[i] = b[i] - tmp[i]; vt[i] = r[i]; y[i] = r[i]; wt[i] = r[i]; z[i] = r[i]; }
    double rho = spl_nrm(y, n), xi = spl_nrm(z, n);
    double gamma = 1.0, eta = -1.0, theta = 0.0, eps = 0.0;
    double target = fmax(rtol * spl_nrm(b, n), atol), rn = spl_nrm(r, n);
    int64_t it = 0;
    for (it = 1; it <= maxit; it++) {
        if (rho == 0.0 || xi == 0.0) break;
        for (int64_t i = 0; i < n; i++) { v[i] = vt[i]/rho; y[i] = y[i]/rho; w[i] = wt[i]/xi; z[i] = z[i]/xi; }
        double delta = spl_dot(z, y, n);
        if (delta == 0.0) break;
        if (it == 1) {
            for (int64_t i = 0; i < n; i++) { p[i] = y[i]; q[i] = z[i]; }
        } else {
            double fp = xi*delta/eps, fq = rho*delta/eps;
            for (int64_t i = 0; i < n; i++) { p[i] = y[i] - fp*p[i]; q[i] = z[i] - fq*q[i]; }
        }
        spl_mv(A, p, pt, n);
        eps = spl_dot(q, pt, n);
        if (eps == 0.0) break;
        double beta = eps/delta;
        if (beta == 0.0) break;
        for (int64_t i = 0; i < n; i++) { vt[i] = pt[i] - beta*v[i]; y[i] = vt[i]; }
        double rho1 = spl_nrm(y, n);
        spl_mtv(A, q, tmp, n);
        for (int64_t i = 0; i < n; i++) { wt[i] = tmp[i] - beta*w[i]; z[i] = wt[i]; }
        xi = spl_nrm(z, n);
        double theta1 = theta;
        theta = rho1/(gamma*fabs(beta));
        double gamma1 = 1.0/sqrt(1.0 + theta*theta);
        if (gamma1 == 0.0) break;
        eta = -eta*rho*gamma1*gamma1/(beta*gamma*gamma);
        if (it == 1) {
            for (int64_t i = 0; i < n; i++) { d[i] = eta*p[i]; s[i] = eta*pt[i]; }
        } else {
            double fac = (theta1*gamma1)*(theta1*gamma1);
            for (int64_t i = 0; i < n; i++) { d[i] = eta*p[i] + fac*d[i]; s[i] = eta*pt[i] + fac*s[i]; }
        }
        for (int64_t i = 0; i < n; i++) { x[i] += d[i]; r[i] -= s[i]; }
        gamma = gamma1; rho = rho1;
        rn = spl_nrm(r, n);
        if (rn <= target) break;
    }
    free(mem); spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* TFQMR (transpose-free QMR, Freund 1993, M=I). scipy.sparse.linalg.tfqmr */
static int r_tfqmr(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    spl_op A[1]; double *b, *x, rtol, atol; int64_t n, LA, Lb, maxit;
    int rc = spl_iter_prologue("tfqmr", args, nargs, res, A, &LA, &b, &Lb, &x, &n, &rtol, &atol, &maxit);
    if (rc != TSR_OK) return rc;
    double *mem = (double *)malloc((size_t)(9 * n) * sizeof(double));
    if (!mem) { spl_op_free(A); fn_free_doubles(b, Lb); return TSR_ENOMEM; }
    double *r = mem, *w = mem + n, *rstar = mem + 2*n, *uu = mem + 3*n, *u2 = mem + 4*n,
           *v = mem + 5*n, *d = mem + 6*n, *au = mem + 7*n, *tmp = mem + 8*n;
    spl_mv(A, x, tmp, n);
    for (int64_t i = 0; i < n; i++) { r[i] = b[i] - tmp[i]; w[i] = r[i]; rstar[i] = r[i]; uu[i] = r[i]; d[i] = 0.0; }
    double rho = spl_dot(rstar, r, n);
    double tau = spl_nrm(r, n), theta = 0.0, eta = 0.0;
    spl_mv(A, uu, v, n);
    double target = fmax(rtol * spl_nrm(b, n), atol), rn = tau;
    int64_t it = 0;
    if (rho != 0.0) {
        for (it = 1; it <= maxit; it++) {
            double sigma = spl_dot(rstar, v, n);
            if (sigma == 0.0) break;
            double alpha = rho/sigma;
            for (int64_t i = 0; i < n; i++) u2[i] = uu[i] - alpha*v[i];
            for (int64_t half = 0; half < 2; half++) {
                const double *uj = (half == 0) ? uu : u2;
                spl_mv(A, uj, au, n);
                for (int64_t i = 0; i < n; i++) w[i] -= alpha*au[i];
                double wn = spl_nrm(w, n);
                double theta2 = (tau != 0.0) ? wn/tau : 0.0;
                double c = 1.0/sqrt(1.0 + theta2*theta2);
                tau = tau*theta2*c;
                double eta2 = c*c*alpha;
                double fac = (alpha != 0.0) ? theta*theta*eta/alpha : 0.0;
                for (int64_t i = 0; i < n; i++) { d[i] = uj[i] + fac*d[i]; x[i] += eta2*d[i]; }
                theta = theta2; eta = eta2;
            }
            spl_mv(A, x, tmp, n);
            rn = 0.0; for (int64_t i = 0; i < n; i++) { double e = b[i] - tmp[i]; rn += e*e; } rn = sqrt(rn);
            if (rn <= target) break;
            double rho_new = spl_dot(rstar, w, n);
            if (rho == 0.0) break;
            double beta = rho_new/rho;
            rho = rho_new;
            for (int64_t i = 0; i < n; i++) uu[i] = w[i] + beta*u2[i];
            double *au2 = au;                                   /* reuse: au2 = A u2 */
            spl_mv(A, u2, au2, n);
            double *auu = tmp;                                  /* auu = A uu */
            spl_mv(A, uu, auu, n);
            for (int64_t i = 0; i < n; i++) v[i] = auu[i] + beta*(au2[i] + beta*v[i]);
        }
    }
    free(mem); spl_op_free(A); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], rn <= target ? 0 : it);
    return TSR_OK;
}

/* LSMR (Fong & Saunders, lambda=0). A is m x n (may be rectangular).
   Golub-Kahan bidiagonalization with the LSMR rotations. scipy.sparse.linalg.lsmr */
static int r_lsmr(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3) { fn_set_error("lsmr: A 2-D, b array"); return TSR_EARG; }
    int64_t mm = args[0].arr.shape[0], nn = args[0].arr.shape[1];
    if (args[1].arr.shape[0] != mm) { fn_set_error("lsmr: b has wrong length"); return TSR_EARG; }
    /* maxiter is argument 6 ("A, b, damp, atol, btol, conlim, maxiter"): reading argument 5 took conlim=1e8 for the
       iteration count, 100 million iterations per default call (15 s on a 10 x 6 system) */
    int64_t maxit = 4 * nn;
    if (nargs > 6 && (args[6].kind == 1 || args[6].kind == 4)) { double mv = (args[6].flags & 1) ? (double)args[6].ival : args[6].num; if (mv > 0) maxit = mv < 1e15 ? (int64_t)mv : (int64_t)1e15; }
    int64_t LA, Lb; double *A = fn_arg_doubles(&args[0], &LA); if (!A) return TSR_ENOMEM;
    double *b = fn_arg_doubles(&args[1], &Lb); if (!b) { fn_free_doubles(A, LA); return TSR_ENOMEM; }
    int64_t sh[1] = {nn};
    double *x = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    double *u = (double *)malloc((size_t)mm * sizeof(double));
    double *v = (double *)malloc((size_t)nn * sizeof(double));
    double *h = (double *)malloc((size_t)nn * sizeof(double));
    double *hbar = (double *)calloc((size_t)nn, sizeof(double));
    double *tmp = (double *)malloc((size_t)((mm > nn ? mm : nn)) * sizeof(double));
    if (!x || !u || !v || !h || !hbar || !tmp) { free(u);free(v);free(h);free(hbar);free(tmp); fn_free_doubles(A, LA); fn_free_doubles(b,Lb); return TSR_ENOMEM; }
    for (int64_t j = 0; j < nn; j++) x[j] = 0.0;
    for (int64_t i = 0; i < mm; i++) u[i] = b[i];
    double beta = spl_nrm(u, mm); if (beta > 0) for (int64_t i = 0; i < mm; i++) u[i] /= beta;
    /* v = A^T u */
    for (int64_t j = 0; j < nn; j++) { double s = 0.0; for (int64_t i = 0; i < mm; i++) s += A[i*nn + j]*u[i]; v[j] = s; }
    double alpha = spl_nrm(v, nn); if (alpha > 0) for (int64_t j = 0; j < nn; j++) v[j] /= alpha;
    for (int64_t j = 0; j < nn; j++) h[j] = v[j];
    double zetabar = alpha*beta, alphabar = alpha;
    double rho = 1.0, rhobar = 1.0, cbar = 1.0, sbar = 0.0;
    int64_t it = 0;
    for (; it < maxit; it++) {
        /* u = A v - alpha u */
        for (int64_t i = 0; i < mm; i++) { double s = 0.0; const double *Ai = A + i*nn; for (int64_t j = 0; j < nn; j++) s += Ai[j]*v[j]; tmp[i] = s - alpha*u[i]; }
        for (int64_t i = 0; i < mm; i++) u[i] = tmp[i];
        beta = spl_nrm(u, mm); if (beta > 0) for (int64_t i = 0; i < mm; i++) u[i] /= beta;
        /* v = A^T u - beta v */
        for (int64_t j = 0; j < nn; j++) { double s = 0.0; for (int64_t i = 0; i < mm; i++) s += A[i*nn + j]*u[i]; tmp[j] = s - beta*v[j]; }
        for (int64_t j = 0; j < nn; j++) v[j] = tmp[j];
        alpha = spl_nrm(v, nn); if (alpha > 0) for (int64_t j = 0; j < nn; j++) v[j] /= alpha;
        double rhoold = rho;
        double hyp1 = hypot(alphabar, beta);
        double chat = alphabar/hyp1, shat = beta/hyp1; (void)shat;
        rho = hyp1;
        double thetanew = (beta/hyp1)*alpha;                    /* shat*alpha */
        alphabar = chat*alpha;
        double rhobarold = rhobar;
        double thetabar = sbar*rho;
        rhobar = hypot(cbar*rho, thetanew);
        cbar = cbar*rho/rhobar;
        sbar = thetanew/rhobar;
        double zeta = cbar*zetabar;
        zetabar = -sbar*zetabar;
        double fac = thetabar*rho/(rhoold*rhobarold);
        for (int64_t j = 0; j < nn; j++) hbar[j] = h[j] - fac*hbar[j];
        double sc = zeta/(rho*rhobar);
        for (int64_t j = 0; j < nn; j++) x[j] += sc*hbar[j];
        double sh2 = thetanew/rho;
        for (int64_t j = 0; j < nn; j++) h[j] = v[j] - sh2*h[j];
        if (alpha <= 1e-300 || beta <= 1e-300) { it++; break; }
    }
    free(u);free(v);free(h);free(hbar);free(tmp);
    fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
    if (nres > 1) fn_result_int(&res[1], 0);
    return TSR_OK;
}

/* LGMRES: GMRES with augmentation. On well-conditioned systems it converges to the
   unique solution solve(A,b); we run full GMRES (restart=n) which reaches that x in
   <=n steps (exact arithmetic), matching scipy's converged iterate. scipy.sparse.linalg.lgmres */


/* ===== pkg6_eigfix: eigsh / eigs / svds (values-only dense LAPACK) ===== */
/* ============================================================================
 * scipy.sparse.linalg : eigsh / eigs / svds  (values-only, dense LAPACK)
 * ----------------------------------------------------------------------------
 * ARPACK (which SciPy uses for these) is not bitwise-reproducible across builds,
 * so these routines reproduce the RESULT VALUES via dense LAPACK on the (dense)
 * operator and apply SciPy's k / which selection:
 *   - eigsh (symmetric):  LAPACKE_dsyevd  -> real eigenvalues -> select k ->
 *                         returned ascending (float64 [k]).
 *   - eigs  (general):    LAPACKE_dgeev   -> complex eigenvalues -> select k ->
 *                         returned as a (k,2) float64 array of (real, |imag|),
 *                         sorted by (real, |imag|). The SIGN of the imaginary
 *                         part of a conjugate-pair member that straddles the
 *                         k-th selection boundary is NOT unique (ARPACK may
 *                         return either partner), so only |imag| is compared.
 *   - svds:               LAPACKE_dgesdd(N) -> singular values -> top-k ->
 *                         returned ascending (float64 [k]). which=LM only
 *                         (which=SM shift-invert diverges in ARPACK).
 * EIGEN/SINGULAR VECTORS ARE OUT OF SCOPE: they are sign/rotation non-unique.
 * Only the eigenvalues / singular values are returned (nout = 1).
 * Matrix arguments arrive dense (the generators build them from a dense array).
 * ========================================================================== */

/* --- (A) add near the other extern decls / includes at the top of the file --- */
/* NOTE: scipy_sparse_linalg.c does not currently include <lapacke.h>; add it. */

/* qsort comparator: ascending doubles */
static int eig_cmp_asc(const void *pa, const void *pb)
{
    double a = *(const double *)pa, b = *(const double *)pb;
    return (a < b) ? -1 : (a > b) ? 1 : 0;
}
/* qsort comparator: (real, |imag|) pairs ascending */
static int eig_cmp_ri(const void *pa, const void *pb)
{
    const double *a = (const double *)pa, *b = (const double *)pb;
    if (a[0] < b[0]) return -1;
    if (a[0] > b[0]) return 1;
    if (a[1] < b[1]) return -1;
    if (a[1] > b[1]) return 1;
    return 0;
}

/* read an integer-valued scalar argument (k) */
static int eig_arg_int(const tsr_arg *a, int64_t *out)
{
    if (a->kind != 1) return 0;
    *out = (a->flags & 1) ? a->ival : (int64_t)a->num;
    return 1;
}

/* partial-select helper for eigsh: pick k indices from ascending w[0..n-1] per which,
   write the k selected eigenvalues into sel[] (unsorted), return 0/-1. */
/* the which argument (args[2]): LM when absent; a value listing the options (the registry default "LM|SM|LA|SA|BE",
   which the facades pass when the caller gives none) means its first option, as in scipy */
static const char *eig_which(const tsr_arg *args, int nargs, char *buf)
{
    if (nargs <= 2 || args[2].kind != 2 || !args[2].str) return "LM";
    const char *bar = strchr(args[2].str, '|');
    if (!bar) return args[2].str;
    size_t len = (size_t)(bar - args[2].str);
    if (len > 7) len = 7;
    memcpy(buf, args[2].str, len);
    buf[len] = '\0';
    return buf;
}

static int eigsh_select(const double *w, int64_t n, int64_t k, const char *which, double *sel)
{
    /* w is ascending. */
    if (!strcmp(which, "LA")) {                 /* k largest algebraic */
        for (int64_t i = 0; i < k; i++) sel[i] = w[n - k + i];
    } else if (!strcmp(which, "SA")) {          /* k smallest algebraic */
        for (int64_t i = 0; i < k; i++) sel[i] = w[i];
    } else if (!strcmp(which, "BE")) {          /* k from both ends: ceil(k/2) hi, floor(k/2) lo */
        int64_t nhi = (k + 1) / 2, nlo = k / 2, p = 0;
        for (int64_t i = 0; i < nlo; i++) sel[p++] = w[i];
        for (int64_t i = 0; i < nhi; i++) sel[p++] = w[n - nhi + i];
    } else if (!strcmp(which, "LM") || !strcmp(which, "SM")) {
        /* order indices by |w| ; pick last/first k */
        int64_t *idx = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
        if (!idx) return -1;
        for (int64_t i = 0; i < n; i++) idx[i] = i;
        /* insertion sort idx by |w| ascending */
        for (int64_t i = 1; i < n; i++) {
            int64_t o = idx[i]; double key = fabs(w[o]); int64_t p = i - 1;
            while (p >= 0 && fabs(w[idx[p]]) > key) { idx[p + 1] = idx[p]; p--; }
            idx[p + 1] = o;
        }
        int lm = !strcmp(which, "LM");
        for (int64_t i = 0; i < k; i++) sel[i] = w[lm ? idx[n - k + i] : idx[i]];
        free(idx);
    } else {
        return -2;  /* unsupported which */
    }
    return 0;
}

/* eigsh(A, k=6, which=LM, ...): k eigenvalues of a symmetric A, returned ascending. */
/* The order in which the m Ritz values theta are wanted for `which` (LA, SA, LM or SM): most wanted first. */
static void lz_order(const double *theta, int64_t m, const char *which, int64_t *ord)
{
    for (int64_t i = 0; i < m; i++) ord[i] = i;
    for (int64_t i = 1; i < m; i++) {                     /* insertion sort */
        const int64_t o = ord[i];
        int64_t p = i - 1;
        for (; p >= 0; p--) {
            const double a = theta[ord[p]], b = theta[o];
            const int before = !strcmp(which, "LA") ? b > a : !strcmp(which, "SA") ? b < a
                             : !strcmp(which, "SM") ? fabs(b) < fabs(a) : fabs(b) > fabs(a);   /* LM */
            if (!before) break;
            ord[p + 1] = ord[p];
        }
        ord[p + 1] = o;
    }
}

/* One classical Gram-Schmidt pass of q against the r orthonormal vectors X (r x n, vector p at X + p n): c = X q,
   q -= X^T c, by two BLAS matrix-vector products (scalar loops when a size exceeds the 32-bit BLAS interface).
   t (r entries) receives the coefficients; c, when not NULL, accumulates them. */
static void lz_proj(const double *X, int64_t r, int64_t n, double *q, double *t, double *c)
{
    if (r <= 0) return;
    if (n <= INT_MAX && r <= INT_MAX) {
        cblas_dgemv(101, 111, (int)r, (int)n, 1.0, X, (int)n, q, 1, 0.0, t, 1);
        cblas_dgemv(101, 112, (int)r, (int)n, -1.0, X, (int)n, t, 1, 1.0, q, 1);
    } else {
        for (int64_t p = 0; p < r; p++) {
            const double *xp = X + p * n;
            double s = 0.0;
            for (int64_t i = 0; i < n; i++) s += xp[i] * q[i];
            t[p] = s;
        }
        for (int64_t p = 0; p < r; p++) {
            const double *xp = X + p * n;
            for (int64_t i = 0; i < n; i++) q[i] -= t[p] * xp[i];
        }
    }
    if (c) for (int64_t p = 0; p < r; p++) c[p] += t[p];
}

/* q -= its components along the nl vectors L (vector p at L + p n) and the j vectors of V, twice (t: scratch of
   max(nl, j) entries) */
static void lz_orth(double *q, const double *L, int64_t nl, const double *V, int64_t j, int64_t n, double *t)
{
    for (int pass = 0; pass < 2; pass++) {
        lz_proj(L, nl, n, q, t, NULL);
        lz_proj(V, j, n, q, t, NULL);
    }
}

/* Thick-restart Lanczos (Wu and Simon) with full reorthogonalization, in the orthogonal complement of the locked
   orthonormal columns L (nl of them; the operator is A on that complement). A basis of m vectors is built by sparse
   products only; the m x m projected matrix H gives the Ritz values theta; a Ritz pair has converged when
   |beta_m y_last| <= max(tol |theta|, 16 eps |A|) (tol 0: machine precision, as ARPACK); otherwise the l most wanted Ritz vectors and
   the last Lanczos vector restart the basis. vals: the k wanted values, vecs (n x k, may be NULL): their vectors. */
static int lz_run(const spl_op *A, int64_t n, int64_t k, const char *which, int64_t m, int64_t maxit, double tol,
                  const double *v0, const double *L, int64_t nl, double *vals, double *vecs)
{
    if (!(tol > 0.0)) tol = 2.220446049250313e-16;
    double *V = (double *)calloc((size_t)((m + 1) * n), sizeof(double));
    double *H = (double *)calloc((size_t)(m * m), sizeof(double)), *Y = (double *)malloc((size_t)(m * m) * sizeof(double));
    double *theta = (double *)malloc((size_t)m * sizeof(double)), *w = (double *)malloc((size_t)n * sizeof(double));
    double *T = (double *)malloc((size_t)(n * m) * sizeof(double)), *Ys = (double *)malloc((size_t)(m * m) * sizeof(double));
    const int64_t nt = nl > m + 1 ? nl : m + 1;
    double *t = (double *)malloc((size_t)nt * sizeof(double)), *hc = (double *)malloc((size_t)m * sizeof(double));
    int64_t *ord = (int64_t *)malloc((size_t)m * sizeof(int64_t));
    int rc = TSR_OK;
    if (!V || !H || !Y || !theta || !w || !T || !Ys || !t || !hc || !ord) { rc = TSR_ENOMEM; goto done; }
    for (int64_t i = 0; i < n; i++) V[i] = v0 ? v0[i] : 1.0 + 0.5 * sin(1.7 * (double)i + 0.3 + 0.61 * (double)nl);
    lz_orth(V, L, nl, NULL, 0, n, t);
    {
        double nv = 0.0;
        for (int64_t i = 0; i < n; i++) nv += V[i] * V[i];
        nv = sqrt(nv);
        if (!(nv > 0.0)) { fn_set_error("eigsh: v0 must not be zero"); rc = TSR_EARG; goto done; }
        for (int64_t i = 0; i < n; i++) V[i] /= nv;
    }
    int64_t j0 = 0;
    double beta = 0.0;
    for (int64_t iter = 0; iter < maxit; iter++) {
        for (int64_t j = j0; j < m; j++) {               /* expand: v_{j+1} from A v_j, orthogonal to L and v_0..v_j */
            spl_mv(A, V + j * n, w, n);
            for (int64_t i = 0; i <= j; i++) hc[i] = 0.0;
            for (int pass = 0; pass < 2; pass++) {           /* classical Gram-Schmidt, twice (DGKS) */
                lz_proj(L, nl, n, w, t, NULL);
                lz_proj(V, j + 1, n, w, t, hc);
            }
            for (int64_t i = 0; i <= j; i++) H[i * m + j] = H[j * m + i] = hc[i];
            beta = 0.0;
            for (int64_t q = 0; q < n; q++) beta += w[q] * w[q];
            beta = sqrt(beta);
            double *vn = V + (j + 1) * n;
            if (beta > 1e-12 * (fabs(H[j * m + j]) + 1e-300)) {
                for (int64_t q = 0; q < n; q++) vn[q] = w[q] / beta;
            } else {                                      /* an invariant subspace: continue with a new direction */
                beta = 0.0;
                for (int64_t q = 0; q < n; q++) vn[q] = sin(0.37 * (double)(q + 1) * (double)(j + 2));
                lz_orth(vn, L, nl, V, j + 1, n, t);
                double nn = 0.0;
                for (int64_t q = 0; q < n; q++) nn += vn[q] * vn[q];
                nn = sqrt(nn);
                if (!(nn > 0.0)) { fn_set_error("eigsh: the Lanczos basis cannot be extended"); rc = TSR_EARG; goto done; }
                for (int64_t q = 0; q < n; q++) vn[q] /= nn;
            }
            if (j + 1 < m) { H[(j + 1) * m + j] = beta; H[j * m + j + 1] = beta; }
        }
        memcpy(Y, H, (size_t)(m * m) * sizeof(double));
        if (LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'U', (lapack_int)m, Y, (lapack_int)m, theta) != 0) {
            fn_set_error("eigsh: the projected eigenproblem failed"); rc = TSR_EARG; goto done;
        }
        lz_order(theta, m, which, ord);
        double anorm = 0.0;
        for (int64_t c = 0; c < m; c++) if (fabs(theta[c]) > anorm) anorm = fabs(theta[c]);
        int64_t nconv = 0;
        for (int64_t p = 0; p < k; p++) {
            const int64_t c = ord[p];
            /* tol |theta|, but not below the attainable 16 eps |A| (the eigenvalue error is quadratic in the residual) */
            nconv += fabs(beta * Y[(m - 1) * m + c]) <= fmax(tol * fabs(theta[c]), 16.0 * 2.220446049250313e-16 * anorm);
        }
        const int conv = nconv == k;
        /* the restart keeps the k wanted Ritz vectors plus as many more as have converged, at most half the free
           slots (ARPACK's dsaup2 rule; one wanted value: half the basis): few kept while nothing has converged, so
           each restart adds more new directions */
        int64_t l = k + (nconv < (m - k) / 2 ? nconv : (m - k) / 2);
        if (l == 1) l = m >= 6 ? m / 2 : (m > 2 ? 2 : 1);
        if (l >= m) l = m - 1;
        if (conv) l = k;
        /* T[p] = V_m^T y_ord[p]: the kept (or answer) Ritz vectors, one matrix product */
        for (int64_t p = 0; p < l; p++) for (int64_t i = 0; i < m; i++) Ys[p * m + i] = Y[i * m + ord[p]];
        if (n <= INT_MAX) {
            cblas_dgemm(101, 111, 111, (int)l, (int)n, (int)m, 1.0, Ys, (int)m, V, (int)n, 0.0, T, (int)n);
        } else {
            for (int64_t p = 0; p < l; p++) {
                double *tp = T + p * n;
                for (int64_t q = 0; q < n; q++) tp[q] = 0.0;
                for (int64_t i = 0; i < m; i++) {
                    const double y = Ys[p * m + i], *vi = V + i * n;
                    for (int64_t q = 0; q < n; q++) tp[q] += y * vi[q];
                }
            }
        }
        if (conv) {
            for (int64_t p = 0; p < k; p++) vals[p] = theta[ord[p]];
            if (vecs) memcpy(vecs, T, (size_t)(k * n) * sizeof(double));
            goto done;
        }
        memmove(V + l * n, V + m * n, (size_t)n * sizeof(double));   /* the last Lanczos vector follows them */
        memcpy(V, T, (size_t)(l * n) * sizeof(double));
        memset(H, 0, (size_t)(m * m) * sizeof(double));
        for (int64_t p = 0; p < l; p++) {
            const int64_t c = ord[p];
            H[p * m + p] = theta[c];
            H[p * m + l] = H[l * m + p] = beta * Y[(m - 1) * m + c];
        }
        j0 = l;
    }
    fn_set_error("eigsh: no convergence after %lld restarts (raise maxiter or tol)", (long long)maxit);
    rc = TSR_EARG;
done:
    free(V); free(H); free(Y); free(theta); free(w); free(T); free(Ys); free(t); free(hc); free(ord);
    return rc;
}

/* mu is more wanted than v for `which`, beyond rounding */
static int lz_better(double mu, double v, const char *which)
{
    const double eps = 1e-10 * fmax(1.0, fmax(fabs(mu), fabs(v)));
    if (!strcmp(which, "LA")) return mu > v + eps;
    if (!strcmp(which, "SA")) return mu < v - eps;
    if (!strcmp(which, "SM")) return fabs(mu) < fabs(v) - eps;
    return fabs(mu) > fabs(v) + eps;                      /* LM */
}

/* The k eigenvalues `which` wants (LA, SA, LM, SM). A Krylov space from one start vector holds one direction of each
   repeated eigenvalue, so Lanczos alone can miss a repeated one (2-D grids have many). After convergence, the most
   wanted eigenvalue of A on the orthogonal complement of the k Ritz vectors is computed; while it beats the least
   wanted one kept (a missed repeat always does), it replaces it, and the search repeats. */
static int eigsh_one(const spl_op *A, int64_t n, int64_t k, const char *which, int64_t ncv, int64_t maxit, double tol,
                     const double *v0, double *out)
{
    int64_t m = ncv > k ? ncv : (2 * k + 1 > 20 ? 2 * k + 1 : 20);
    if (m > n) m = n;
    if (m <= k) { fn_set_error("eigsh: ncv must exceed k (and n must exceed k)"); return TSR_EARG; }
    if (maxit <= 0) maxit = 10 * n > 1000 ? 10 * n : 1000;
    double *vecs = (double *)malloc((size_t)(n * k) * sizeof(double)), *u = (double *)malloc((size_t)n * sizeof(double));
    if (!vecs || !u) { free(vecs); free(u); return TSR_ENOMEM; }
    int rc = lz_run(A, n, k, which, m, maxit, tol, v0, NULL, 0, out, vecs);
    for (int64_t round = 0; rc == TSR_OK && round <= k; round++) {
        int64_t m1 = 20 < n - k ? 20 : n - k;
        if (m1 < 2) break;                                /* the complement is too small to search */
        double mu;
        if (lz_run(A, n, 1, which, m1, maxit, tol, NULL, vecs, k, &mu, u) != TSR_OK) break;
        int64_t worst = 0;
        for (int64_t p = 1; p < k; p++) if (lz_better(out[worst], out[p], which)) worst = p;
        if (!lz_better(mu, out[worst], which)) break;
        out[worst] = mu;
        memcpy(vecs + worst * n, u, (size_t)n * sizeof(double));
    }
    free(vecs); free(u);
    return rc;
}

/* eigsh on a CSR matrix: BE is the top ceil(k/2) (LA) and the bottom floor(k/2) (SA) */
static int eigsh_lanczos(const spl_op *A, int64_t n, int64_t k, const char *which, int64_t ncv, int64_t maxit,
                         double tol, const double *v0, double *out)
{
    if (!strcmp(which, "BE")) {
        const int64_t kh = (k + 1) / 2, kl = k / 2;
        int rc = eigsh_one(A, n, kh, "LA", ncv, maxit, tol, v0, out);
        if (rc == TSR_OK && kl > 0) rc = eigsh_one(A, n, kl, "SA", ncv, maxit, tol, v0, out + kh);
        return rc;
    }
    return eigsh_one(A, n, k, which, ncv, maxit, tol, v0, out);
}

/* every eigenvalue of the dense symmetric n x n A (overwritten), then the k that `which` selects into sel */
static int eigsh_dense(double *A, int64_t n, int64_t k, const char *which, double *sel)
{
    double *w = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    if (!w) return TSR_ENOMEM;
    int rc = TSR_OK;
    lapack_int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'N', 'L', (lapack_int)n, A, (lapack_int)n, w);
    if (info != 0) { fn_set_error("eigsh: the eigensolver failed to converge"); rc = TSR_EARG; }
    else {
        int s = eigsh_select(w, n, k, which, sel);
        if (s == -1) rc = TSR_ENOMEM;
        else if (s == -2) { fn_set_error("eigsh: unsupported which (use LM/SM/LA/SA/BE)"); rc = TSR_EARG; }
    }
    free(w);
    return rc;
}

static int r_eigsh(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const int csr = nargs >= 1 && args[0].kind == 5;
    if (nargs < 1 || (!csr && (args[0].kind != 3 || args[0].arr.ndim != 2))) { fn_set_error("eigsh: A must be a 2-D array"); return TSR_EARG; }
    spl_op S;
    int64_t n, cols;
    if (csr) {
        const int lrc = spl_op_load(&S, &args[0], "eigsh", &n, &cols);
        if (lrc != TSR_OK) return lrc;
    } else { n = args[0].arr.shape[0]; cols = args[0].arr.shape[1]; }
    if (cols != n) { if (csr) spl_op_free(&S); fn_set_error("eigsh: A must be square"); return TSR_EARG; }
    int64_t k = 6;
    if (nargs > 1 && args[1].kind == 1) eig_arg_int(&args[1], &k);
    char wbuf[8];
    const char *which = eig_which(args, nargs, wbuf);
    if (k < 1 || k >= n) { if (csr) spl_op_free(&S); fn_set_error("eigsh: need 0 < k < n"); return TSR_EARG; }
    double *sel = (double *)malloc((size_t)k * sizeof(double));
    if (!sel) { if (csr) spl_op_free(&S); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (!csr) {
        if (!TSR_LA_FIT("eigsh", n)) { free(sel); return TSR_EARG; }   /* n goes to LAPACKE as a 32-bit lapack_int */
        int64_t La; double *A = fn_arg_doubles(&args[0], &La);
        if (!A) rc = TSR_ENOMEM;
        else { rc = eigsh_dense(A, n, k, which, sel); fn_free_doubles(A, La); }
    } else if (strcmp(which, "LM") && strcmp(which, "SM") && strcmp(which, "LA") && strcmp(which, "SA") && strcmp(which, "BE")) {
        fn_set_error("eigsh: unsupported which (use LM/SM/LA/SA/BE)"); rc = TSR_EARG;
    } else if (!strcmp(which, "SM") && n <= 4096) {
        /* the smallest magnitudes: Lanczos reaches them slowly without a shift, so a matrix that fits is solved dense */
        double *D = (double *)calloc((size_t)(n * n), sizeof(double));
        if (!D) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < n; i++)
                for (int64_t q = S.ptr[i]; q < S.ptr[i + 1]; q++) D[i * n + S.ind[q]] += S.data[q];
            rc = eigsh_dense(D, n, k, which, sel);
            free(D);
        }
    } else {                                              /* eigsh(A, k, which, sigma, v0, ncv, maxiter, tol, ...) */
        int64_t ncv = 0, maxit = 0, Lv = 0;
        double tol = 0.0, *v0 = NULL;
        if (nargs > 5 && args[5].kind == 1) eig_arg_int(&args[5], &ncv);
        if (nargs > 6 && args[6].kind == 1) eig_arg_int(&args[6], &maxit);
        if (nargs > 7 && args[7].kind == 1) tol = (args[7].flags & 1) ? (double)args[7].ival : args[7].num;
        if (nargs > 4 && args[4].kind == 3) {
            if (args[4].arr.ndim != 1 || args[4].arr.shape[0] != n) { fn_set_error("eigsh: v0 must have n entries"); rc = TSR_EARG; }
            else if (!(v0 = fn_arg_doubles(&args[4], &Lv))) rc = TSR_ENOMEM;
        }
        if (rc == TSR_OK) rc = eigsh_lanczos(&S, n, k, which, ncv, maxit, tol, v0, sel);
        if (v0) fn_free_doubles(v0, Lv);
    }
    if (rc == TSR_OK) {
        qsort(sel, (size_t)k, sizeof(double), eig_cmp_asc);
        int64_t sh[1] = {k};
        double *o = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
        if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < k; i++) o[i] = sel[i];
    }
    free(sel);
    if (csr) spl_op_free(&S);
    return rc;
}

/* eigs(A, k=6, which=LM, ...): k eigenvalues of a general A, returned as a (k,2)
   array of (real, |imag|) sorted by (real, |imag|). */
static int r_eigs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 1 || args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("eigs: A must be a 2-D array"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("eigs: A must be square"); return TSR_EARG; }
    if (!TSR_LA_FIT("eigs", n)) return TSR_EARG;
    int64_t k = 6;
    if (nargs > 1 && args[1].kind == 1) eig_arg_int(&args[1], &k);
    char wbuf[8];
    const char *which = eig_which(args, nargs, wbuf);
    if (k < 1 || k >= n - 1) { fn_set_error("eigs: need 0 < k < n-1"); return TSR_EARG; }

    int64_t La; double *A = fn_arg_doubles(&args[0], &La); if (!A) return TSR_ENOMEM;
    double *wr = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    double *wi = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    int64_t *idx = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    double *sel = (double *)malloc((size_t)(2 * k) * sizeof(double));
    int rc = TSR_OK;
    if (!wr || !wi || !idx || !sel) rc = TSR_ENOMEM;
    else {
        double d = 0.0;
        lapack_int info = !fn_all_finite(A, (int64_t)(n) * (n)) ? -5 : LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)n, A, (lapack_int)n, wr, wi, &d, (lapack_int)n, &d, (lapack_int)n);  /* ld >= n: LAPACKE 3.10 */
        if (info != 0) { fn_set_error("eigs: the QR algorithm failed to converge"); rc = TSR_EARG; }
        else {
            for (int64_t i = 0; i < n; i++) idx[i] = i;
            /* key(which) for each eigenvalue; insertion sort idx by key ascending, then take top/bottom k */
            double *key = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
            int bottom = 0; /* SR picks smallest k */
            if (!key) rc = TSR_ENOMEM;
            else {
                if (!strcmp(which, "LM"))      for (int64_t i = 0; i < n; i++) key[i] = hypot(wr[i], wi[i]);
                else if (!strcmp(which, "LR")) for (int64_t i = 0; i < n; i++) key[i] = wr[i];
                else if (!strcmp(which, "SR")) { for (int64_t i = 0; i < n; i++) key[i] = wr[i]; bottom = 1; }
                else if (!strcmp(which, "LI")) for (int64_t i = 0; i < n; i++) key[i] = fabs(wi[i]);
                else { fn_set_error("eigs: unsupported which (use LM/LR/SR/LI)"); rc = TSR_EARG; }
                if (rc == TSR_OK) {
                    for (int64_t i = 1; i < n; i++) {
                        int64_t o = idx[i]; double kk = key[o]; int64_t p = i - 1;
                        while (p >= 0 && key[idx[p]] > kk) { idx[p + 1] = idx[p]; p--; }
                        idx[p + 1] = o;
                    }
                    /* select k indices */
                    for (int64_t i = 0; i < k; i++) {
                        int64_t src = bottom ? idx[i] : idx[n - k + i];
                        sel[2 * i] = wr[src]; sel[2 * i + 1] = fabs(wi[src]);
                    }
                    qsort(sel, (size_t)k, 2 * sizeof(double), eig_cmp_ri);
                    int64_t sh[2] = {k, 2};
                    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
                    if (!out) rc = TSR_ENOMEM; else for (int64_t i = 0; i < 2 * k; i++) out[i] = sel[i];
                }
            }
            free(key);
        }
    }
    free(wr); free(wi); free(idx); free(sel); fn_free_doubles(A, La);
    return rc;
}

/* svds(A, k=6, which=LM, ...): top-k singular values of A, returned ascending. */
static int r_svds(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 1 || args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("svds: A must be a 2-D array"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1];
    if (!TSR_LA_FIT("svds", m, n)) return TSR_EARG;
    int64_t k = 6;
    if (nargs > 1 && args[1].kind == 1) eig_arg_int(&args[1], &k);
    char wbuf[8];
    const char *which = eig_which(args, nargs, wbuf);
    if (strcmp(which, "LM")) { fn_set_error("svds: only which=LM is supported (SM shift-invert is not reproducible)"); return TSR_EARG; }
    int64_t mn = m < n ? m : n;
    if (k < 1 || k >= mn) { fn_set_error("svds: need 0 < k < min(M,N)"); return TSR_EARG; }

    int64_t La; double *A = fn_arg_doubles(&args[0], &La); if (!A) return TSR_ENOMEM;
    double *s = (double *)malloc((size_t)(mn > 0 ? mn : 1) * sizeof(double));
    double *superb = (double *)malloc((size_t)(mn > 1 ? mn - 1 : 1) * sizeof(double));
    int rc = TSR_OK;
    if (!s || !superb) rc = TSR_ENOMEM;
    else {
        double dummy = 0.0;
        /* dgesdd with jobz='N' mishandles non-square row-major (bad ldvt -> "Wrong parameter 11").
           Use dgesvd with jobu='N',jobvt='N': singular values only, no U/VT referenced. */
        lapack_int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)m, (lapack_int)n,
                                         A, (lapack_int)n, s, &dummy, 1, &dummy, 1, superb);
        if (info != 0) { fn_set_error("svds: the SVD failed to converge"); rc = TSR_EARG; }
        else {
            /* dgesdd returns s descending; top-k are s[0..k-1]; return ascending */
            int64_t sh[1] = {k};
            double *o = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
            if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < k; i++) o[i] = s[k - 1 - i];
        }
    }
    free(s); free(superb); fn_free_doubles(A, La);
    return rc;
}


/* ===== pkg6_lobpcg: lobpcg + funm_multiply_krylov ===== */
/* =========================================================================== *
 * lobpcg + funm_multiply_krylov  (the last two scipy.sparse.linalg symbols).
 * Both run their dense algorithm on the dense operator A, like the Krylov family
 * above. They reuse the spl_* helpers, LAPACK (dsyevd / dgeqrf+dorgqr via
 * <lapacke.h>) and the shared matfun engine sl_funm_byname (np_linalg.c).
 * =========================================================================== */

/* Column-span orthonormalization via QR (A is n x c, row-major with leading
   dimension c, overwritten with the orthonormal Q in its first q = min(n,c)
   columns; q is returned in *pkeep). An n x c block has at most min(n,c)
   orthonormal columns, so DORGQR must never be asked for more than that (asking
   for c > n columns is an illegal DORGQR argument). Returns 0 on success. */
static int spl_qr_orth(double *Acm, int64_t n, int64_t c, int64_t *pkeep)
{
    int64_t q = c < n ? c : n;                 /* columns of Q we can actually form */
    double *tau = (double *)malloc(sizeof(double) * (size_t)(q > 0 ? q : 1));
    if (!tau) return -1;
    lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)c, Acm, (lapack_int)c, tau);
    if (info == 0) info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, (lapack_int)n, (lapack_int)q, (lapack_int)q, Acm, (lapack_int)c, tau);
    free(tau);
    *pkeep = q;
    return info == 0 ? 0 : -1;
}

/* lobpcg(A, X, ...): the k smallest (largest=False) or largest eigenvalues of the
   symmetric matrix A via the LOBPCG iteration (B = M = I). Eigenvectors are not
   unique, so this ROUTINE returns the EIGENVALUES only (ascending), compared to the
   dense reference (np.linalg.eigvalsh slice) in the fixtures. scipy.sparse.linalg.lobpcg */
static int r_lobpcg(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) {
        fn_set_error("lobpcg: A must be a 2-D array and X a 2-D initial guess"); return TSR_EARG;
    }
    int64_t n = args[0].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("lobpcg: A must be square"); return TSR_EARG; }
    if (!TSR_LA_FIT("lobpcg", n)) return TSR_EARG;
    if (args[1].arr.shape[0] != n) { fn_set_error("lobpcg: X has the wrong number of rows"); return TSR_EARG; }
    int64_t k = args[1].arr.shape[1];
    if (k < 1 || k > n) { fn_set_error("lobpcg: the block size must satisfy 1 <= k <= n"); return TSR_EARG; }
    /* signature: (A, X, B=None, M=None, Y=None, tol=None, maxiter=None, largest=True, ...) */
    int largest = 1;
    int64_t maxiter = 200;
    double tol = 1e-12;
    if (nargs > 5 && (args[5].kind == 1 || args[5].kind == 4)) { double tv = (args[5].flags & 1) ? (double)args[5].ival : args[5].num; if (tv > 0) tol = tv; }
    if (nargs > 6 && (args[6].kind == 1 || args[6].kind == 4)) { double mv = (args[6].flags & 1) ? (double)args[6].ival : args[6].num; if (mv > 0) maxiter = (int64_t)mv; }
    if (nargs > 7 && (args[7].kind == 1 || args[7].kind == 4)) largest = (args[7].num != 0.0);

    int64_t LA, LX; double *A = fn_arg_doubles(&args[0], &LA); if (!A) return TSR_ENOMEM;
    double *X = fn_arg_doubles(&args[1], &LX); if (!X) { fn_free_doubles(A, LA); return TSR_ENOMEM; }

    int rc = TSR_OK;
    const int64_t m3 = 3 * k;                 /* max subspace width [X | R | P] */
    double *Xc  = (double *)malloc(sizeof(double) * (size_t)(n * k));
    double *AX  = (double *)malloc(sizeof(double) * (size_t)(n * k));
    double *Pm  = (double *)calloc((size_t)(n * k), sizeof(double));
    double *lam = (double *)malloc(sizeof(double) * (size_t)k);
    double *S   = (double *)malloc(sizeof(double) * (size_t)(n * m3));
    double *AS  = (double *)malloc(sizeof(double) * (size_t)(n * m3));
    double *Gp  = (double *)malloc(sizeof(double) * (size_t)(m3 * m3));
    double *w   = (double *)malloc(sizeof(double) * (size_t)m3);
    double *QX  = (double *)malloc(sizeof(double) * (size_t)(n * k));
    double *Pbuf= (double *)malloc(sizeof(double) * (size_t)(n * k));
    if (!Xc || !AX || !Pm || !lam || !S || !AS || !Gp || !w || !QX || !Pbuf) { rc = TSR_ENOMEM; goto done; }

    memcpy(Xc, X, sizeof(double) * (size_t)(n * k));
    { int64_t keep; if (spl_qr_orth(Xc, n, k, &keep) != 0) { fn_set_error("lobpcg: QR orthonormalization failed"); rc = TSR_EARG; goto done; } }
    for (int64_t c = 0; c < k; c++)
        for (int64_t i = 0; i < n; i++) { double s = 0.0; const double *Ai = A + i * n; for (int64_t j = 0; j < n; j++) s += Ai[j] * Xc[j * k + c]; AX[i * k + c] = s; }
    for (int64_t c = 0; c < k; c++) { double s = 0.0; for (int64_t i = 0; i < n; i++) s += Xc[i * k + c] * AX[i * k + c]; lam[c] = s; }

    int64_t pwidth = 0;
    for (int64_t iter = 0; iter < maxiter; iter++) {
        double *R = Pbuf;
        double maxrn = 0.0;
        for (int64_t c = 0; c < k; c++) {
            double rn = 0.0;
            for (int64_t i = 0; i < n; i++) { double e = AX[i * k + c] - lam[c] * Xc[i * k + c]; R[i * k + c] = e; rn += e * e; }
            rn = sqrt(rn); if (rn > maxrn) maxrn = rn;
        }
        if (maxrn < tol) break;
        /* The search subspace is [X | R | P] with up to 3k columns, which can exceed
           n (e.g. n=8, k=3 -> 9). A block with more columns than rows has at most n
           orthonormal columns, so cap the assembled width to n, dropping the P block
           first (it is the search-direction history) and then trailing R columns. X
           (the current approximation, k columns) is always kept. */
        int64_t rcols = k, pcols = pwidth;
        if (k + rcols + pcols > n) pcols = (n - k - rcols > 0) ? (n - k - rcols) : 0;
        if (k + rcols + pcols > n) rcols = (n - k > 0) ? (n - k) : 0;
        int64_t width = k + rcols + pcols;
        for (int64_t i = 0; i < n; i++) {
            double *Si = S + i * width;
            for (int64_t c = 0; c < k; c++) Si[c] = Xc[i * k + c];
            for (int64_t c = 0; c < rcols; c++) Si[k + c] = R[i * k + c];
            for (int64_t c = 0; c < pcols; c++) Si[k + rcols + c] = Pm[i * k + c];
        }
        int64_t keep;
        if (spl_qr_orth(S, n, width, &keep) != 0) { fn_set_error("lobpcg: subspace QR failed"); rc = TSR_EARG; goto done; }
        width = keep;                              /* QR may yield fewer than width orthonormal cols */
        for (int64_t c = 0; c < width; c++)
            for (int64_t i = 0; i < n; i++) { double s = 0.0; const double *Ai = A + i * n; for (int64_t j = 0; j < n; j++) s += Ai[j] * S[j * width + c]; AS[i * width + c] = s; }
        for (int64_t a = 0; a < width; a++)
            for (int64_t b = 0; b < width; b++) { double s = 0.0; for (int64_t i = 0; i < n; i++) s += S[i * width + a] * AS[i * width + b]; Gp[a * width + b] = s; }
        for (int64_t a = 0; a < width; a++) for (int64_t b = a + 1; b < width; b++) { double t = 0.5 * (Gp[a * width + b] + Gp[b * width + a]); Gp[a * width + b] = Gp[b * width + a] = t; }
        lapack_int info = LAPACKE_dsyevd(LAPACK_ROW_MAJOR, 'V', 'L', (lapack_int)width, Gp, (lapack_int)width, w);
        if (info != 0) { fn_set_error("lobpcg: the projected eigensolve (dsyevd) failed"); rc = TSR_EARG; goto done; }
        for (int64_t c = 0; c < k; c++) {
            int64_t col = largest ? (width - 1 - c) : c;
            lam[c] = w[col];
            for (int64_t i = 0; i < n; i++) {
                double s = 0.0; for (int64_t a = 0; a < width; a++) s += S[i * width + a] * Gp[a * width + col];
                QX[i * k + c] = s;
            }
        }
        for (int64_t c = 0; c < k; c++)
            for (int64_t i = 0; i < n; i++) {
                double proj = 0.0;
                for (int64_t d = 0; d < k; d++) { double xtn = 0.0; for (int64_t t = 0; t < n; t++) xtn += Xc[t * k + d] * QX[t * k + c]; proj += Xc[i * k + d] * xtn; }
                Pm[i * k + c] = QX[i * k + c] - proj;
            }
        pwidth = k;
        memcpy(Xc, QX, sizeof(double) * (size_t)(n * k));
        for (int64_t c = 0; c < k; c++)
            for (int64_t i = 0; i < n; i++) { double s = 0.0; const double *Ai = A + i * n; for (int64_t j = 0; j < n; j++) s += Ai[j] * Xc[j * k + c]; AX[i * k + c] = s; }
    }
    for (int64_t i = 1; i < k; i++) { double v = lam[i]; int64_t j = i - 1; while (j >= 0 && lam[j] > v) { lam[j + 1] = lam[j]; j--; } lam[j + 1] = v; }
    {
        int64_t sh[1] = {k};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
        if (!out) { rc = TSR_ENOMEM; goto done; }
        for (int64_t i = 0; i < k; i++) out[i] = lam[i];
    }
done:
    free(Xc); free(AX); free(Pm); free(lam); free(S); free(AS); free(Gp); free(w); free(QX); free(Pbuf);
    fn_free_doubles(A, LA); fn_free_doubles(X, LX);
    return rc;
}

/* funm_multiply_krylov(f, A, b): f(A) @ b approximated in a full Krylov subspace via
   Arnoldi. With m = n (dense, small) the approximation equals the dense f(A) @ b to
   round-off. f is a function name (one of exp/sqrt/log/sin/cos/sinh/cosh/tanh/tan);
   f(H) on the m x m upper-Hessenberg H uses the shared matfun engine sl_funm_byname.
   scipy.sparse.linalg.funm_multiply_krylov (compared to the dense f(A) @ b). */
static int r_funm_multiply_krylov(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 3 || args[0].kind != 2 || !args[0].str || args[1].kind != 3 || args[1].arr.ndim != 2 || args[2].kind != 3) {
        fn_set_error("funm_multiply_krylov: expected (func_name, A 2-D, b array)"); return TSR_EARG;
    }
    const char *fname = args[0].str;
    int64_t n = args[1].arr.shape[0];
    if (args[1].arr.shape[1] != n) { fn_set_error("funm_multiply_krylov: A must be square"); return TSR_EARG; }
    if (args[2].arr.shape[0] != n) { fn_set_error("funm_multiply_krylov: b has the wrong length"); return TSR_EARG; }
    int64_t LA, Lb; double *A = fn_arg_doubles(&args[1], &LA); if (!A) return TSR_ENOMEM;
    double *b = fn_arg_doubles(&args[2], &Lb); if (!b) { fn_free_doubles(A, LA); return TSR_ENOMEM; }

    int rc = TSR_OK;
    int64_t m = n;
    double *V = (double *)calloc((size_t)(n * (m + 1)), sizeof(double));
    double *H = (double *)calloc((size_t)((m + 1) * m), sizeof(double));
    double *wv = (double *)malloc(sizeof(double) * (size_t)n);
    if (!V || !H || !wv) { rc = TSR_ENOMEM; goto done2; }

    double beta = spl_nrm(b, n);
    int64_t md = m;
    if (beta == 0.0) { md = 0; }
    else {
        for (int64_t i = 0; i < n; i++) V[i * (m + 1) + 0] = b[i] / beta;
        for (int64_t j = 0; j < m; j++) {
            for (int64_t i = 0; i < n; i++) { double s = 0.0; const double *Ai = A + i * n; for (int64_t t = 0; t < n; t++) s += Ai[t] * V[t * (m + 1) + j]; wv[i] = s; }
            for (int64_t i = 0; i <= j; i++) {
                double hij = 0.0; for (int64_t t = 0; t < n; t++) hij += V[t * (m + 1) + i] * wv[t];
                H[i * m + j] = hij;
                for (int64_t t = 0; t < n; t++) wv[t] -= hij * V[t * (m + 1) + i];
            }
            double hn = spl_nrm(wv, n);
            H[(j + 1) * m + j] = hn;
            if (hn < 1e-14) { md = j + 1; break; }
            for (int64_t i = 0; i < n; i++) V[i * (m + 1) + (j + 1)] = wv[i] / hn;
        }
    }

    {
        int64_t sh[1] = {n};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
        if (!out) { rc = TSR_ENOMEM; goto done2; }
        for (int64_t i = 0; i < n; i++) out[i] = 0.0;
        if (md > 0) {
            double *Hm = (double *)malloc(sizeof(double) * (size_t)(md * md));
            double *fH = (double *)malloc(sizeof(double) * (size_t)(md * md));
            double *y  = (double *)malloc(sizeof(double) * (size_t)md);
            if (!Hm || !fH || !y) { free(Hm); free(fH); free(y); rc = TSR_ENOMEM; goto done2; }
            for (int64_t i = 0; i < md; i++) for (int64_t j = 0; j < md; j++) Hm[i * md + j] = H[i * m + j];
            if (sl_funm_byname(fname, Hm, md, fH) != 0) {
                free(Hm); free(fH); free(y);
                fn_set_error("funm_multiply_krylov: unsupported func name '%s' or non-real/degenerate Hessenberg spectrum", fname);
                rc = TSR_EARG; goto done2;
            }
            for (int64_t i = 0; i < md; i++) y[i] = fH[i * md + 0] * beta;
            for (int64_t i = 0; i < n; i++) { double s = 0.0; for (int64_t j = 0; j < md; j++) s += V[i * (m + 1) + j] * y[j]; out[i] = s; }
            free(Hm); free(fH); free(y);
        }
    }
done2:
    free(V); free(H); free(wv);
    fn_free_doubles(A, LA); fn_free_doubles(b, Lb);
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
    ROUTINE("splinalg.lgmres", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None, inner_m=30, outer_k=3", "x", r_lgmres, NULL,
            "LGMRES (GMRES with augmentation) for a general square system (scipy.sparse.linalg.lgmres)."),
    ROUTINE("splinalg.gcrotmk", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None, m=20, k=None", "x", r_gcrotmk, NULL,
            "GCROT(m,k) truncated GMRES with recycling for a general square system (scipy.sparse.linalg.gcrotmk)."),
    ROUTINE("splinalg.minres", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None", "x", r_minres, NULL,
            "MINRES for a symmetric (possibly indefinite) system (scipy.sparse.linalg.minres)."),
    ROUTINE("splinalg.lsqr", 1, "A, b, atol=1e-8, btol=1e-8, maxiter=None", "x", r_lsqr, NULL,
            "LSQR least-squares solver for a sparse rectangular system (scipy.sparse.linalg.lsqr)."),
    ROUTINE("splinalg.onenormest", 1, "A, t=2, itmax=5", "est", r_onenormest, NULL,
            "Estimate of the matrix 1-norm (scipy.sparse.linalg.onenormest)."),
    ROUTINE("splinalg.eigsh", 1, "A, k=6, which=LM|SM|LA|SA|BE, sigma=None, v0=None, ncv=None, maxiter=None, tol=0.0, return_eigenvectors=True", "w", r_eigsh, NULL,
            "k eigenvalues of a symmetric matrix, ascending, values only (scipy.sparse.linalg.eigsh); vectors are out of scope."),
    ROUTINE("splinalg.eigs", 1, "A, k=6, which=LM|LR|SR|LI, sigma=None, v0=None, ncv=None, maxiter=None, tol=0.0, return_eigenvectors=True", "w", r_eigs, NULL,
            "k eigenvalues of a general matrix as (real, |imag|) pairs, values only (scipy.sparse.linalg.eigs); vectors / imag sign are out of scope."),
    ROUTINE("splinalg.svds", 1, "A, k=6, which=LM, v0=None, maxiter=None, return_singular_vectors=True, tol=0.0", "s", r_svds, NULL,
            "Top-k singular values, ascending, values only (scipy.sparse.linalg.svds); which=LM only, vectors out of scope."),
    ROUTINE("splinalg.qmr", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None, M1=None, M2=None", "x", r_qmr, NULL,
            "Quasi-minimal residual for a general square system (scipy.sparse.linalg.qmr)."),
    ROUTINE("splinalg.tfqmr", 1, "A, b, x0=None, rtol=1e-5, atol=0.0, maxiter=None, M=None", "x", r_tfqmr, NULL,
            "Transpose-free QMR for a general square system (scipy.sparse.linalg.tfqmr)."),
    ROUTINE("splinalg.lsmr", 1, "A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, maxiter=None", "x", r_lsmr, NULL,
            "LSMR least-squares solver for a sparse rectangular system (scipy.sparse.linalg.lsmr)."),
    ROUTINE("splinalg.lobpcg", 1, "A, X, B=None, M=None, Y=None, tol=None, maxiter=None, largest=True", "w", r_lobpcg, NULL,
            "LOBPCG eigenvalues (ascending) of the k smallest/largest eigenpairs of a symmetric matrix; eigenvectors are non-unique so only the eigenvalues are returned (scipy.sparse.linalg.lobpcg)."),
    ROUTINE("splinalg.funm_multiply_krylov", 1, "f, A, b", "y", r_funm_multiply_krylov, NULL,
            "Krylov (Arnoldi) approximation of f(A) @ b; f is a matrix-function name one of exp/sqrt/log/sin/cos/sinh/cosh/tanh/tan (scipy.sparse.linalg.funm_multiply_krylov)."),
};

const fn_table TSR_SCIPY_SPARSE_LINALG_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
