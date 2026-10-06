/* scipy.sparse constructors from a dense 2-D array, returning the storage arrays as a dict (data + the index
 * arrays + shape). This matches scipy's attribute layout exactly (indices/indptr/row/col are int32, sorted as
 * scipy sorts them), so the sparse formats are verified by comparing the raw storage. The object-with-methods
 * API is replaced by these stateless routines (real float64 input).
 *
 *   csr_array / csr_matrix   -> data, indices (column), indptr, shape
 *   csc_array / csc_matrix   -> data, indices (row),    indptr, shape
 *   coo_array / coo_matrix   -> data, row, col, shape
 *   dia_array / dia_matrix   -> data (offsets x ncols), offsets, shape
 */
#include "fn.h"

#include <stdlib.h>
#include <string.h>

static int sp_input(const tsr_arg *a, const double **p, int64_t *m, int64_t *n, int64_t *tot, double **owned)
{
    if (a->kind != 3 || a->arr.ndim != 2) { fn_set_error("sparse: input must be a 2-D array"); return TSR_EARG; }
    *m = a->arr.shape[0];
    *n = a->arr.shape[1];
    *owned = fn_arg_doubles(a, tot);
    if (!*owned) { fn_set_error("sparse: input could not be read as float64"); return TSR_ENOMEM; }
    *p = *owned;
    return TSR_OK;
}

static void sp_shape(tsr_result *r, int64_t m, int64_t n)
{
    int64_t sh[1] = {2};
    int64_t *s = (int64_t *)fn_result_array(r, TSR_I64, 1, sh);
    if (s) { s[0] = m; s[1] = n; }
}

/* csr_array(A): row-major nonzeros -> data, indices (column of each), indptr (row pointers), shape */
static int r_csr_array(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    const double *A; int64_t m, n, tot; double *owned;
    int rc = sp_input(&args[0], &A, &m, &n, &tot, &owned);
    if (rc < 0) return rc;
    int64_t nnz = 0;
    for (int64_t i = 0; i < m * n; i++) if (A[i] != 0.0) nnz++;
    int64_t dsh[1] = {nnz}, psh[1] = {m + 1};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int32_t *ind = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh);
    int32_t *iptr = (int32_t *)fn_result_array(&res[2], TSR_I32, 1, psh);
    if (!data || !ind || !iptr) { fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    int64_t k = 0;
    iptr[0] = 0;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) { const double v = A[i * n + j]; if (v != 0.0) { data[k] = v; ind[k] = (int32_t)j; k++; } }
        iptr[i + 1] = (int32_t)k;
    }
    sp_shape(&res[3], m, n);
    fn_free_doubles(owned, tot);
    return TSR_OK;
}

/* csc_array(A): column-major nonzeros -> data, indices (row of each), indptr (column pointers), shape */
static int r_csc_array(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    const double *A; int64_t m, n, tot; double *owned;
    int rc = sp_input(&args[0], &A, &m, &n, &tot, &owned);
    if (rc < 0) return rc;
    int64_t nnz = 0;
    for (int64_t i = 0; i < m * n; i++) if (A[i] != 0.0) nnz++;
    int64_t dsh[1] = {nnz}, psh[1] = {n + 1};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int32_t *ind = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh);
    int32_t *iptr = (int32_t *)fn_result_array(&res[2], TSR_I32, 1, psh);
    if (!data || !ind || !iptr) { fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    int64_t k = 0;
    iptr[0] = 0;
    for (int64_t j = 0; j < n; j++) {
        for (int64_t i = 0; i < m; i++) { const double v = A[i * n + j]; if (v != 0.0) { data[k] = v; ind[k] = (int32_t)i; k++; } }
        iptr[j + 1] = (int32_t)k;
    }
    sp_shape(&res[3], m, n);
    fn_free_doubles(owned, tot);
    return TSR_OK;
}

/* coo_array(A): row-major nonzeros -> data, row, col, shape */
static int r_coo_array(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    const double *A; int64_t m, n, tot; double *owned;
    int rc = sp_input(&args[0], &A, &m, &n, &tot, &owned);
    if (rc < 0) return rc;
    int64_t nnz = 0;
    for (int64_t i = 0; i < m * n; i++) if (A[i] != 0.0) nnz++;
    int64_t dsh[1] = {nnz};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int32_t *row = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh);
    int32_t *col = (int32_t *)fn_result_array(&res[2], TSR_I32, 1, dsh);
    if (!data || !row || !col) { fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    int64_t k = 0;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) { const double v = A[i * n + j]; if (v != 0.0) { data[k] = v; row[k] = (int32_t)i; col[k] = (int32_t)j; k++; } }
    sp_shape(&res[3], m, n);
    fn_free_doubles(owned, tot);
    return TSR_OK;
}

/* dia_array(A): diagonal storage. offsets (ascending) are the diagonals that hold a nonzero; data is
   (noffsets x ncols) with data[d][j] = A[j-offset][j] where that row exists, else 0 (as scipy stores it). */
static int r_dia_array(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    const double *A; int64_t m, n, tot; double *owned;
    int rc = sp_input(&args[0], &A, &m, &n, &tot, &owned);
    if (rc < 0) return rc;
    const int64_t lo = -(m - 1), hi = n - 1;
    const int64_t span = (m + n - 1) > 0 ? (m + n - 1) : 1;
    int32_t *offs = (int32_t *)malloc(sizeof(int32_t) * (size_t)span);
    if (!offs) { fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    int64_t noff = 0;
    for (int64_t off = lo; off <= hi; off++) {
        int has = 0;
        for (int64_t j = 0; j < n && !has; j++) { const int64_t i = j - off; if (i >= 0 && i < m && A[i * n + j] != 0.0) has = 1; }
        if (has) offs[noff++] = (int32_t)off;
    }
    int64_t dsh[2] = {noff, n}, osh[1] = {noff};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 2, dsh);
    int32_t *offsets = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, osh);
    if (!data || !offsets) { free(offs); fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    for (int64_t d = 0; d < noff; d++) {
        offsets[d] = offs[d];
        const int64_t off = offs[d];
        for (int64_t j = 0; j < n; j++) { const int64_t i = j - off; data[d * n + j] = (i >= 0 && i < m) ? A[i * n + j] : 0.0; }
    }
    free(offs);
    sp_shape(&res[2], m, n);
    fn_free_doubles(owned, tot);
    return TSR_OK;
}

/* emit the CSR storage of a dense m x n matrix into res[0..3] (data, indices, indptr, shape) */
static int emit_csr(tsr_result *res, const double *A, int64_t m, int64_t n)
{
    int64_t nnz = 0;
    for (int64_t i = 0; i < m * n; i++) if (A[i] != 0.0) nnz++;
    int64_t dsh[1] = {nnz}, psh[1] = {m + 1};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int32_t *ind = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh);
    int32_t *iptr = (int32_t *)fn_result_array(&res[2], TSR_I32, 1, psh);
    if (!data || !ind || !iptr) return TSR_ENOMEM;
    int64_t k = 0;
    iptr[0] = 0;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) { const double v = A[i * n + j]; if (v != 0.0) { data[k] = v; ind[k] = (int32_t)j; k++; } }
        iptr[i + 1] = (int32_t)k;
    }
    sp_shape(&res[3], m, n);
    return TSR_OK;
}

/* hstack(*blocks): horizontally stack 2-D blocks (same row count) -> CSR storage (scipy.sparse.hstack.tocsr) */
static int r_hstack(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items = args; int64_t k = nargs;
    if (nargs == 1 && args[0].kind == 5) { items = args[0].items; k = args[0].count; }
    if (k < 1) { fn_set_error("hstack: at least one block is required"); return TSR_EARG; }
    int64_t m = -1, ncols = 0;
    for (int64_t i = 0; i < k; i++) {
        if (items[i].kind != 3 || items[i].arr.ndim != 2) { fn_set_error("hstack: every block must be a 2-D array"); return TSR_EARG; }
        if (m < 0) m = items[i].arr.shape[0];
        else if (items[i].arr.shape[0] != m) { fn_set_error("hstack: blocks must have the same number of rows"); return TSR_EARG; }
        ncols += items[i].arr.shape[1];
    }
    double *dense = (double *)calloc((size_t)(m * ncols > 0 ? m * ncols : 1), sizeof(double));
    if (!dense) return TSR_ENOMEM;
    int64_t coloff = 0;
    for (int64_t i = 0; i < k; i++) {
        int64_t nb; double *b = fn_arg_doubles(&items[i], &nb);
        if (!b) { free(dense); return TSR_ENOMEM; }
        const int64_t bc = items[i].arr.shape[1];
        for (int64_t r = 0; r < m; r++) for (int64_t c = 0; c < bc; c++) dense[r * ncols + coloff + c] = b[r * bc + c];
        fn_free_doubles(b, nb); coloff += bc;
    }
    int rc = emit_csr(res, dense, m, ncols);
    free(dense);
    return rc;
}

/* vstack(*blocks): vertically stack 2-D blocks (same column count) -> CSR storage */
static int r_vstack(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items = args; int64_t k = nargs;
    if (nargs == 1 && args[0].kind == 5) { items = args[0].items; k = args[0].count; }
    if (k < 1) { fn_set_error("vstack: at least one block is required"); return TSR_EARG; }
    int64_t n = -1, nrows = 0;
    for (int64_t i = 0; i < k; i++) {
        if (items[i].kind != 3 || items[i].arr.ndim != 2) { fn_set_error("vstack: every block must be a 2-D array"); return TSR_EARG; }
        if (n < 0) n = items[i].arr.shape[1];
        else if (items[i].arr.shape[1] != n) { fn_set_error("vstack: blocks must have the same number of columns"); return TSR_EARG; }
        nrows += items[i].arr.shape[0];
    }
    double *dense = (double *)calloc((size_t)(nrows * n > 0 ? nrows * n : 1), sizeof(double));
    if (!dense) return TSR_ENOMEM;
    int64_t rowoff = 0;
    for (int64_t i = 0; i < k; i++) {
        int64_t nb; double *b = fn_arg_doubles(&items[i], &nb);
        if (!b) { free(dense); return TSR_ENOMEM; }
        const int64_t mb = items[i].arr.shape[0];
        for (int64_t r = 0; r < mb; r++) for (int64_t c = 0; c < n; c++) dense[(rowoff + r) * n + c] = b[r * n + c];
        fn_free_doubles(b, nb); rowoff += mb;
    }
    int rc = emit_csr(res, dense, nrows, n);
    free(dense);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("sparse.hstack", 4, "*blocks", "data, indices, indptr, shape", r_hstack, NULL, "Stack sparse blocks horizontally, as CSR storage (scipy.sparse.hstack)."),
    ROUTINE("sparse.vstack", 4, "*blocks", "data, indices, indptr, shape", r_vstack, NULL, "Stack sparse blocks vertically, as CSR storage (scipy.sparse.vstack)."),
    ROUTINE("sparse.csr_array", 4, "A", "data, indices, indptr, shape", r_csr_array, NULL, "Compressed sparse row storage of a dense matrix (scipy.sparse.csr_array)."),
    ROUTINE("sparse.csr_matrix", 4, "A", "data, indices, indptr, shape", r_csr_array, NULL, "Compressed sparse row storage of a dense matrix (scipy.sparse.csr_matrix)."),
    ROUTINE("sparse.csc_array", 4, "A", "data, indices, indptr, shape", r_csc_array, NULL, "Compressed sparse column storage of a dense matrix (scipy.sparse.csc_array)."),
    ROUTINE("sparse.csc_matrix", 4, "A", "data, indices, indptr, shape", r_csc_array, NULL, "Compressed sparse column storage of a dense matrix (scipy.sparse.csc_matrix)."),
    ROUTINE("sparse.coo_array", 4, "A", "data, row, col, shape", r_coo_array, NULL, "Coordinate (triplet) storage of a dense matrix (scipy.sparse.coo_array)."),
    ROUTINE("sparse.coo_matrix", 4, "A", "data, row, col, shape", r_coo_array, NULL, "Coordinate (triplet) storage of a dense matrix (scipy.sparse.coo_matrix)."),
    ROUTINE("sparse.dia_array", 3, "A", "data, offsets, shape", r_dia_array, NULL, "Diagonal storage of a dense matrix (scipy.sparse.dia_array)."),
    ROUTINE("sparse.dia_matrix", 3, "A", "data, offsets, shape", r_dia_array, NULL, "Diagonal storage of a dense matrix (scipy.sparse.dia_matrix)."),
};

const fn_table TSR_SCIPY_SPARSE_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
