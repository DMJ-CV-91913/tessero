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

static double *sp_dense(int64_t m, int64_t n);   /* a checked dense scratch matrix (below) */

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
    double *dense = sp_dense(m, ncols);
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
    double *dense = sp_dense(nrows, n);
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

/* emit a dense m-by-n matrix (row-major) as canonical CSR: data, column indices, row pointers, shape. */
static int sp_emit_csr(tsr_result *res, const double *D, int64_t m, int64_t n)
{
    int64_t nnz = 0; for (int64_t i = 0; i < m * n; i++) if (D[i] != 0.0) nnz++;
    int64_t dsh[1] = {nnz}, psh[1] = {m + 1};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int32_t *ind = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh);
    int32_t *iptr = (int32_t *)fn_result_array(&res[2], TSR_I32, 1, psh);
    if (!data || !ind || !iptr) return TSR_ENOMEM;
    int64_t k = 0; iptr[0] = 0;
    for (int64_t i = 0; i < m; i++) { for (int64_t j = 0; j < n; j++) { const double v = D[i * n + j]; if (v != 0.0) { data[k] = v; ind[k] = (int32_t)j; k++; } } iptr[i + 1] = (int32_t)k; }
    sp_shape(&res[3], m, n);
    return TSR_OK;
}
/* an integer argument (axis, offset, block size, grid size): truncated as int(); NaN or infinity reads as the
   default and the value is clamped to +-2^40 (beyond any int32-indexed shape), so the callers' index arithmetic
   (i + k, nr * nc) cannot overflow */
static int64_t sp_int(const tsr_arg *a, int n, int idx, int64_t dflt)
{
    if (idx >= n || a[idx].kind != 1) return dflt;
    const int64_t v = fn_num_i64(&a[idx], dflt), lim = (int64_t)1 << 40;
    return v < -lim ? -lim : v > lim ? lim : v;
}

/* a dimension argument (eye's m and n, spdiags' m and n, an entry of diags' shape): int() truncation, NaN and
   infinity refused with Python's message, negative refused */
static int sp_dim(const tsr_arg *a, const char *who, int64_t *out)
{
    int rc = fn_arg_i64(a, 0, INT64_MIN, INT64_MAX, who, out);
    if (rc < 0) return rc;
    if (*out < 0) { fn_set_error("'shape' elements cannot be negative (%s)", who); return TSR_EARG; }
    return TSR_OK;
}

/* a diagonal offset read from a float64 array: integral (NumPy refuses a fractional or NaN offset) */
static int sp_offset(double v, int64_t *out)
{
    if (v != v) { fn_set_error("cannot convert float NaN to integer"); return TSR_EARG; }
    if (v != floor(v)) { fn_set_error("'numpy.float64' object cannot be interpreted as an integer"); return TSR_ETYPE; }
    /* every offset beyond +-2^40 lies outside any int32-indexed matrix: clamp so later arithmetic stays defined */
    *out = v < -1099511627776.0 ? -((int64_t)1 << 40) : v > 1099511627776.0 ? ((int64_t)1 << 40) : (int64_t)v;
    return TSR_OK;
}

/* Value of diagonal p at (i, j) for the direct diagonal builder; 1 when that diagonal holds an entry there. */
typedef int (*sp_diag_get)(const void *ctx, int64_t p, int64_t i, int64_t j, double *v);

/* CSR (data, indices, indptr, shape in res[0..3]) of an m x n matrix whose entries lie on nd diagonals (offset
   off[p] for diagonal p), built row by row without a dense m x n intermediate (eye(10**5) needed 80 GB, and the
   m * n product overflowed int64 for large shapes). Diagonals with equal offsets combine in p order: summed (sum)
   or the last one wins; zero results are dropped, exactly as the dense build did. */
static int sp_diag_csr(tsr_result *res, int64_t m, int64_t n, int64_t nd, const int64_t *off, int sum,
                       sp_diag_get get, const void *ctx)
{
    if (m > INT32_MAX || n > INT32_MAX) { fn_set_error("sparse: the shape is too large for int32 indices"); return TSR_ENOMEM; }
    /* the diagonals that intersect the matrix, in ascending offset order (stable): columns ascend in a row */
    int64_t *ord = (int64_t *)malloc((size_t)(nd > 0 ? nd : 1) * sizeof(int64_t));
    if (!ord) return TSR_ENOMEM;
    int64_t no = 0;
    for (int64_t p = 0; p < nd; p++) {
        if (off[p] <= -m || off[p] >= n) continue;
        int64_t q = no++;
        while (q > 0 && off[ord[q - 1]] > off[p]) { ord[q] = ord[q - 1]; q--; }
        ord[q] = p;
    }
    double *data = NULL; int32_t *ind = NULL, *iptr = NULL;
    int64_t nnz = 0;
    for (int pass = 0; pass < 2; pass++) {
        if (pass == 1) {
            if (nnz > INT32_MAX) { free(ord); fn_set_error("sparse: too many stored entries for int32 indices"); return TSR_ENOMEM; }
            int64_t dsh[1] = {nnz}, psh[1] = {m + 1};
            data = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
            ind = data ? (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh) : NULL;
            iptr = ind ? (int32_t *)fn_result_array(&res[2], TSR_I32, 1, psh) : NULL;
            if (!iptr) { free(ord); return TSR_ENOMEM; }
            iptr[0] = 0;
            nnz = 0;
        }
        for (int64_t i = 0; i < m; i++) {
            for (int64_t q = 0; q < no; ) {
                const int64_t k = off[ord[q]], j = i + k;
                int64_t r = q;
                while (r < no && off[ord[r]] == k) r++;
                if (j >= 0 && j < n) {
                    double acc = 0.0, v;
                    int have = 0;
                    for (int64_t t = q; t < r; t++)
                        if (get(ctx, ord[t], i, j, &v)) { acc = sum ? acc + v : v; have = 1; }
                    if (have && acc != 0.0) {
                        if (pass == 1) { data[nnz] = acc; ind[nnz] = (int32_t)j; }
                        nnz++;
                    }
                }
                q = r;
            }
            if (pass == 1) iptr[i + 1] = (int32_t)nnz;
        }
    }
    free(ord);
    sp_shape(&res[3], m, n);
    return TSR_OK;
}

static int sp_get_one(const void *ctx, int64_t p, int64_t i, int64_t j, double *v) { (void)ctx; (void)p; (void)i; (void)j; *v = 1.0; return 1; }

/* a dense m x n scratch matrix (zeroed); NULL with an error when the shape cannot be indexed with int32 or is
   implausibly large (the product is checked before it is formed) */
static double *sp_dense(int64_t m, int64_t n)
{
    if (m < 0 || n < 0 || m > INT32_MAX || n > INT32_MAX || (n > 0 && m > ((int64_t)1 << 36) / n)) {
        fn_set_error("sparse: the shape is too large");
        return NULL;
    }
    const size_t cnt = (size_t)(m * n);
    return (double *)calloc(cnt ? cnt : 1, sizeof(double));
}

/* an offset argument (eye's k, diags' scalar offsets): int() truncation, NaN / infinity refused, clamped to
   +-2^40 (every larger offset lies outside any int32-indexed matrix) */
static int sp_koff(const tsr_arg *a, int64_t *out)
{
    int rc = fn_arg_i64(a, 0, INT64_MIN, INT64_MAX, "k", out);
    if (rc < 0) return rc;
    const int64_t lim = (int64_t)1 << 40;
    *out = *out < -lim ? -lim : *out > lim ? lim : *out;
    return TSR_OK;
}

/* eye(m, n=None, k=0): sparse identity-like matrix with ones on diagonal k, as CSR. */
static int r_lil_array(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    const double *A; int64_t m, n, tot; double *owned;
    int rc = sp_input(&args[0], &A, &m, &n, &tot, &owned);
    if (rc < 0) return rc;
    int64_t nnz = 0;
    for (int64_t i = 0; i < m * n; i++) if (A[i] != 0.0) nnz++;
    int64_t dsh[1] = {nnz}, psh[1] = {m + 1};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int32_t *rows = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh);
    int32_t *iptr = (int32_t *)fn_result_array(&res[2], TSR_I32, 1, psh);
    if (!data || !rows || !iptr) { fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    int64_t k = 0;
    iptr[0] = 0;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) { const double v = A[i * n + j]; if (v != 0.0) { data[k] = v; rows[k] = (int32_t)j; k++; } }
        iptr[i + 1] = (int32_t)k;
    }
    sp_shape(&res[3], m, n);
    fn_free_doubles(owned, tot);
    return TSR_OK;
}

/* dok_array(A): dict-of-keys. The stored (i,j)->v entries, canonicalised by
   sorting on (row, col), are exactly the row-major coordinate triplets ->
   row, col, data, shape (same layout as coo_array). */
static int r_dok_array(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
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

/* bsr_array(A, R=1, C=1): block sparse row. A is tiled into R-by-C blocks
   (m%R==C-divisible checked); a block is stored iff it has any nonzero. The
   block grid is traversed row-major, so block-column indices are sorted
   within each block-row (scipy's canonical layout). Returns
     data    (nnz_blocks x R x C, 3-D),
     indices (block column of each stored block, int32),
     indptr  (block-row boundaries, int32),
     shape (m, n),
     blocksize (R, C).
   SciPy's default blocksize is (1,1); the facade passes it explicitly. */
static int r_bsr_array(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const double *A; int64_t m, n, tot; double *owned;
    int rc = sp_input(&args[0], &A, &m, &n, &tot, &owned);
    if (rc < 0) return rc;
    const int64_t R = sp_int(args, nargs, 1, 1), C = sp_int(args, nargs, 2, 1);
    if (R <= 0 || C <= 0 || (m % R) != 0 || (n % C) != 0) {
        fn_set_error("sparse.bsr_array: blocksize must divide the matrix shape");
        fn_free_doubles(owned, tot); return TSR_EARG;
    }
    const int64_t mb = m / R, nb = n / C;
    /* count nonzero blocks */
    int64_t nblk = 0;
    for (int64_t bi = 0; bi < mb; bi++)
        for (int64_t bj = 0; bj < nb; bj++) {
            int has = 0;
            for (int64_t r = 0; r < R && !has; r++)
                for (int64_t c = 0; c < C && !has; c++)
                    if (A[(bi * R + r) * n + (bj * C + c)] != 0.0) has = 1;
            if (has) nblk++;
        }
    int64_t dsh[3] = {nblk, R, C}, ish[1] = {nblk}, psh[1] = {mb + 1};
    double *data = (double *)fn_result_array(&res[0], TSR_F64, 3, dsh);
    int32_t *ind = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, ish);
    int32_t *iptr = (int32_t *)fn_result_array(&res[2], TSR_I32, 1, psh);
    if (!data || !ind || !iptr) { fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    int64_t k = 0;
    iptr[0] = 0;
    for (int64_t bi = 0; bi < mb; bi++) {
        for (int64_t bj = 0; bj < nb; bj++) {
            int has = 0;
            for (int64_t r = 0; r < R && !has; r++)
                for (int64_t c = 0; c < C && !has; c++)
                    if (A[(bi * R + r) * n + (bj * C + c)] != 0.0) has = 1;
            if (!has) continue;
            ind[k] = (int32_t)bj;
            for (int64_t r = 0; r < R; r++)
                for (int64_t c = 0; c < C; c++)
                    data[(k * R + r) * C + c] = A[(bi * R + r) * n + (bj * C + c)];
            k++;
        }
        iptr[bi + 1] = (int32_t)k;
    }
    sp_shape(&res[3], m, n);
    { int64_t bsh[1] = {2}; int64_t *bs = (int64_t *)fn_result_array(&res[4], TSR_I64, 1, bsh); if (bs) { bs[0] = R; bs[1] = C; } }
    fn_free_doubles(owned, tot);
    return TSR_OK;
}

static int r_sp_eye(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t m = 0, n, k = 0;
    int rc;
    if (nargs > 2 && a[2].kind == 1 && (rc = sp_koff(&a[2], &k)) < 0) return rc;
    if (nargs > 0 && a[0].kind == 1) {
        if ((rc = fn_arg_i64(&a[0], 0, INT64_MIN, INT64_MAX, "m", &m)) < 0) return rc;
        if (m < 0) { fn_set_error("sparse.eye: m must be >= 0"); return TSR_EARG; }
    }
    n = m;
    if (nargs > 1 && a[1].kind == 1) {
        if ((rc = fn_arg_i64(&a[1], 0, INT64_MIN, INT64_MAX, "n", &n)) < 0) return rc;
        if (n < 0) n = m;                                 /* negative n: the old "None" sentinel */
    }
    return sp_diag_csr(res, m, n, 1, &k, 1, sp_get_one, NULL);
}
static int r_sp_identity(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; int64_t n = 0;
    if (nargs > 0 && a[0].kind == 1) { int rc = fn_arg_i64(&a[0], 0, INT64_MIN, INT64_MAX, "n", &n); if (rc < 0) return rc; }
    if (n < 0) { fn_set_error("sparse.identity: n must be >= 0"); return TSR_EARG; }
    const int64_t k = 0;
    return sp_diag_csr(res, n, n, 1, &k, 1, sp_get_one, NULL);
}
/* kron(A, B): Kronecker product of two dense matrices, as CSR. */
static int r_sp_kron(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (a[0].kind != 3 || a[0].arr.ndim != 2 || a[1].kind != 3 || a[1].arr.ndim != 2) { fn_set_error("sparse.kron: A, B must be 2-D arrays"); return TSR_EARG; }
    const int64_t ma = a[0].arr.shape[0], na = a[0].arr.shape[1], mb = a[1].arr.shape[0], nb = a[1].arr.shape[1];
    int64_t la, lb; double *A = fn_arg_doubles(&a[0], &la), *B = A ? fn_arg_doubles(&a[1], &lb) : NULL;
    int rc = B ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        const int64_t M = ma * mb, N = na * nb; double *D = sp_dense(M, N);
        if (!D) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < ma; i++) for (int64_t j = 0; j < na; j++) { double av = A[i * na + j]; if (av != 0.0) for (int64_t p = 0; p < mb; p++) for (int64_t q = 0; q < nb; q++) D[(i * mb + p) * N + (j * nb + q)] = av * B[p * nb + q]; }
            rc = sp_emit_csr(res, D, M, N); free(D); }
    }
    fn_free_doubles(A, la); fn_free_doubles(B, lb);
    return rc;
}
/* kronsum(A, B): Kronecker sum kron(I_b, A) + kron(B, I_a) of square A (a x a), B (b x b), as CSR. */
static int r_sp_kronsum(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (a[0].kind != 3 || a[0].arr.ndim != 2 || a[1].kind != 3 || a[1].arr.ndim != 2) { fn_set_error("sparse.kronsum: A, B must be 2-D arrays"); return TSR_EARG; }
    const int64_t aa = a[0].arr.shape[0], bb = a[1].arr.shape[0];
    if (a[0].arr.shape[1] != aa || a[1].arr.shape[1] != bb) { fn_set_error("sparse.kronsum: A and B must be square"); return TSR_EARG; }
    int64_t la, lb; double *A = fn_arg_doubles(&a[0], &la), *B = A ? fn_arg_doubles(&a[1], &lb) : NULL;
    int rc = B ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        const int64_t N = aa * bb; double *D = sp_dense(N, N);
        if (!D) rc = TSR_ENOMEM;
        else { for (int64_t blk = 0; blk < bb; blk++) for (int64_t u = 0; u < aa; u++) for (int64_t v = 0; v < aa; v++) D[(blk * aa + u) * N + (blk * aa + v)] += A[u * aa + v];
            for (int64_t p = 0; p < bb; p++) for (int64_t qq = 0; qq < bb; qq++) { double bv = B[p * bb + qq]; if (bv != 0.0) for (int64_t u = 0; u < aa; u++) D[(p * aa + u) * N + (qq * aa + u)] += bv; }
            rc = sp_emit_csr(res, D, N, N); free(D); }
    }
    fn_free_doubles(A, la); fn_free_doubles(B, lb);
    return rc;
}
/* tril(A, k) / triu(A, k): lower / upper triangle of a dense matrix, as CSR (lower=1 selects tril). */
static int sp_tri(const tsr_arg *a, int nargs, tsr_result *res, int lower)
{
    if (a[0].kind != 3 || a[0].arr.ndim != 2) { fn_set_error("sparse.tri: A must be a 2-D array"); return TSR_EARG; }
    const int64_t m = a[0].arr.shape[0], n = a[0].arr.shape[1], k = sp_int(a, nargs, 1, 0);
    int64_t la; double *A = fn_arg_doubles(&a[0], &la);
    int rc = A ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { double *D = sp_dense(m, n);
        if (!D) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) { int keep = lower ? (j <= i + k) : (j >= i + k); if (keep) D[i * n + j] = A[i * n + j]; }
            rc = sp_emit_csr(res, D, m, n); free(D); } }
    fn_free_doubles(A, la);
    return rc;
}
static int r_sp_tril(const void *ctx, const tsr_arg *a, int n, tsr_result *res, int nres) { (void)ctx; (void)nres; return sp_tri(a, n, res, 1); }
static int r_sp_triu(const void *ctx, const tsr_arg *a, int n, tsr_result *res, int nres) { (void)ctx; (void)nres; return sp_tri(a, n, res, 0); }
/* find(A): row indices, column indices and values of the nonzeros, in row-major order. */
static int r_sp_find(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (a[0].kind != 3 || a[0].arr.ndim != 2) { fn_set_error("sparse.find: A must be a 2-D array"); return TSR_EARG; }
    const int64_t m = a[0].arr.shape[0], n = a[0].arr.shape[1];
    int64_t la; double *A = fn_arg_doubles(&a[0], &la);
    int rc = A ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        int64_t nnz = 0; for (int64_t i = 0; i < m * n; i++) if (A[i] != 0.0) nnz++;
        int64_t dsh[1] = {nnz};
        int32_t *row = (int32_t *)fn_result_array(&res[0], TSR_I32, 1, dsh);
        int32_t *col = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, dsh);
        double *data = (double *)fn_result_array(&res[2], TSR_F64, 1, dsh);
        if (!row || !col || !data) rc = TSR_ENOMEM;
        else { int64_t k = 0; for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) { const double v = A[i * n + j]; if (v != 0.0) { row[k] = (int32_t)i; col[k] = (int32_t)j; data[k] = v; k++; } } }
    }
    fn_free_doubles(A, la);
    return rc;
}

/* spdiags: diagonal p takes data[p, j] at column j (j < L) */
typedef struct { const double *data; int64_t L; } spd_ctx;
static int spd_get(const void *vc, int64_t p, int64_t i, int64_t j, double *v)
{
    const spd_ctx *c = (const spd_ctx *)vc; (void)i;
    if (j >= c->L) return 0;
    *v = c->data[p * c->L + j];
    return 1;
}
/* diags: entry t of diagonal p (offset k) sits at row t + max(0, -k) */
typedef struct { double *const *d; const int64_t *len, *off; } dg_ctx;
static int dg_get(const void *vc, int64_t p, int64_t i, int64_t j, double *v)
{
    const dg_ctx *c = (const dg_ctx *)vc; (void)j;
    const int64_t t = i - (c->off[p] < 0 ? -c->off[p] : 0);
    if (t < 0 || t >= c->len[p]) return 0;
    *v = c->d[p][t];
    return 1;
}

/* spdiags(data, diags, m, n): set diagonal diags[p] of an m-by-n matrix from data[p], as CSR. */
static int r_sp_spdiags(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (a[0].kind != 3 || a[0].arr.ndim != 2 || a[1].kind != 3 || a[1].arr.ndim != 1) { fn_set_error("sparse.spdiags: data 2-D, diags 1-D"); return TSR_EARG; }
    const int64_t nd = a[0].arr.shape[0], L = a[0].arr.shape[1];
    int64_t m = 0, n = 0;
    int rc;
    if (nargs > 2 && a[2].kind == 1 && (rc = sp_dim(&a[2], "m", &m)) < 0) return rc;
    if (nargs > 3 && a[3].kind == 1 && (rc = sp_dim(&a[3], "n", &n)) < 0) return rc;
    if (a[1].arr.shape[0] != nd) {                        /* offs[p] was read for every data row */
        fn_set_error("number of diagonals (%lld) does not match the number of offsets (%lld)", (long long)nd, (long long)a[1].arr.shape[0]);
        return TSR_EARG;
    }
    int64_t ld, lo; double *data = fn_arg_doubles(&a[0], &ld), *offs = data ? fn_arg_doubles(&a[1], &lo) : NULL;
    int64_t *ko = offs ? (int64_t *)malloc((size_t)(nd > 0 ? nd : 1) * sizeof(int64_t)) : NULL;
    rc = ko ? TSR_OK : TSR_ENOMEM;
    for (int64_t p = 0; p < nd && rc == TSR_OK; p++) rc = sp_offset(offs[p], &ko[p]);
    if (rc == TSR_OK) {
        const spd_ctx c = { data, L };
        rc = sp_diag_csr(res, m, n, nd, ko, 0, spd_get, &c);   /* spdiags: a repeated offset overwrites */
    }
    free(ko); fn_free_doubles(data, ld); fn_free_doubles(offs, lo);
    return rc;
}
/* diags(diagonals, offsets=0, shape=None): build an m-by-n matrix placing diagonals[p] on diagonal offsets[p]
   (entry t of diagonal k -> (t+max(0,-k), t+max(0,k))); duplicate offsets sum, as SciPy. Returned CSR. */
static int r_sp_diags(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items; int64_t nd;
    if (a[0].kind == 5) { items = a[0].items; nd = a[0].count; }
    else if (a[0].kind == 3 && a[0].arr.ndim == 1) { items = &a[0]; nd = 1; }
    else { fn_set_error("sparse.diags: diagonals must be a sequence of 1-D arrays"); return TSR_EARG; }
    for (int64_t p = 0; p < nd; p++)                    /* each diagonal's length is its shape[0] */
        if (items[p].kind != 3 || items[p].arr.ndim != 1) { fn_set_error("sparse.diags: each diagonal must be a 1-D array"); return TSR_EARG; }
    int64_t *offs = (int64_t *)malloc((size_t)(nd > 0 ? nd : 1) * 3 * sizeof(int64_t));
    double **dv = (double **)calloc((size_t)(nd > 0 ? nd : 1), sizeof(double *));
    if (!offs || !dv) { free(offs); free(dv); return TSR_ENOMEM; }
    int64_t *len = offs + (nd > 0 ? nd : 1), *got = len + (nd > 0 ? nd : 1);
    int rc = TSR_OK;
    if (nargs > 1 && a[1].kind == 3) {
        int64_t lo; double *od = fn_arg_doubles(&a[1], &lo);
        if (!od) rc = TSR_ENOMEM;
        else if (lo != nd) { fn_set_error("Different number of diagonals and offsets."); rc = TSR_EARG; }   /* od[p] was read past lo */
        for (int64_t p = 0; p < nd && rc == TSR_OK; p++) rc = sp_offset(od[p], &offs[p]);
        fn_free_doubles(od, lo);
    } else {
        int64_t o = 0;
        if (nargs > 1 && a[1].kind == 1) rc = sp_koff(&a[1], &o);
        for (int64_t p = 0; p < nd; p++) offs[p] = o;
    }
    /* shape: explicit (m, n) as a 2-int array/sequence, else inferred square. */
    int64_t m = -1, n = -1;
    if (rc == TSR_OK && nargs > 2 && a[2].kind == 3 && a[2].arr.ndim == 1) {
        int64_t ls; double *sd = fn_arg_doubles(&a[2], &ls);
        if (!sd) rc = TSR_ENOMEM;
        else if (ls >= 2) {
            tsr_arg e0, e1;
            memset(&e0, 0, sizeof e0); memset(&e1, 0, sizeof e1);
            e0.kind = e1.kind = 1; e0.num = sd[0]; e1.num = sd[1];
            if ((rc = sp_dim(&e0, "shape", &m)) == TSR_OK) rc = sp_dim(&e1, "shape", &n);
        }
        fn_free_doubles(sd, ls);
    } else if (rc == TSR_OK && nargs > 2 && a[2].kind == 5 && a[2].count >= 2) {
        if ((rc = sp_dim(&a[2].items[0], "shape", &m)) == TSR_OK) rc = sp_dim(&a[2].items[1], "shape", &n);
    }
    for (int64_t p = 0; p < nd; p++) { len[p] = items[p].arr.shape[0]; got[p] = 0; }
    if (rc == TSR_OK && m < 0) {                          /* offsets are clamped to +-2^40: no overflow */
        int64_t mx = 0;
        for (int64_t p = 0; p < nd; p++) { const int64_t k = offs[p] < 0 ? -offs[p] : offs[p]; if (len[p] + k > mx) mx = len[p] + k; }
        m = n = mx;
    }
    for (int64_t p = 0; p < nd && rc == TSR_OK; p++) { dv[p] = fn_arg_doubles(&items[p], &got[p]); if (!dv[p]) rc = TSR_ENOMEM; }
    if (rc == TSR_OK) {
        const dg_ctx c = { dv, len, offs };
        rc = sp_diag_csr(res, m, n, nd, offs, 1, dg_get, &c);   /* duplicate offsets sum */
    }
    for (int64_t p = 0; p < nd; p++) if (dv[p]) fn_free_doubles(dv[p], got[p]);
    free(dv); free(offs);
    return rc;
}

/* block_diag(mats): block-diagonal assembly of a sequence of dense blocks, as CSR. */
static int r_sp_block_diag(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items = args; int64_t k = nargs;
    if (nargs == 1 && args[0].kind == 5) { items = args[0].items; k = args[0].count; }
    if (k < 1) { fn_set_error("sparse.block_diag: at least one block is required"); return TSR_EARG; }
    int64_t R = 0, C = 0;
    for (int64_t i = 0; i < k; i++) { if (items[i].kind != 3 || items[i].arr.ndim != 2) { fn_set_error("sparse.block_diag: every block must be a 2-D array"); return TSR_EARG; } R += items[i].arr.shape[0]; C += items[i].arr.shape[1]; }
    double *D = sp_dense(R, C);
    if (!D) return TSR_ENOMEM;
    int64_t roff = 0, coff = 0; int rc = TSR_OK;
    for (int64_t i = 0; i < k && rc == TSR_OK; i++) {
        int64_t nb; double *b = fn_arg_doubles(&items[i], &nb);
        if (!b) { rc = TSR_ENOMEM; break; }
        const int64_t br = items[i].arr.shape[0], bc = items[i].arr.shape[1];
        for (int64_t r = 0; r < br; r++) for (int64_t c = 0; c < bc; c++) D[(roff + r) * C + coff + c] = b[r * bc + c];
        fn_free_doubles(b, nb); roff += br; coff += bc;
    }
    if (rc == TSR_OK) rc = emit_csr(res, D, R, C);
    free(D);
    return rc;
}

/* ===== APPEND to csrc/src/scipy_sparse.c, just before the DEFS[] table ===== */

/* bmat / block_array(blocks, nrows, ncols): 2-D grid of dense blocks (None allowed), row-major flat sequence
   of length nrows*ncols, assembled like numpy.block, as CSR. None entries are zero blocks whose row height
   comes from another present block in the same grid-row and column width from the same grid-column. */
static int r_sp_bmat(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const tsr_arg *items; int64_t k;
    if (a[0].kind == 5) { items = a[0].items; k = a[0].count; }
    else { fn_set_error("sparse.bmat: blocks must be a sequence"); return TSR_EARG; }
    const int64_t nr = sp_int(a, nargs, 1, 0), nc = sp_int(a, nargs, 2, 0);
    if (nr < 1 || nc < 1 || nr > k || nc > k || k != nr * nc) { fn_set_error("sparse.bmat: blocks length must equal nrows*ncols"); return TSR_EARG; }
    int64_t *rh = (int64_t *)malloc((size_t)nr * sizeof(int64_t));
    int64_t *cw = (int64_t *)malloc((size_t)nc * sizeof(int64_t));
    if (!rh || !cw) { free(rh); free(cw); return TSR_ENOMEM; }
    for (int64_t i = 0; i < nr; i++) rh[i] = -1;
    for (int64_t j = 0; j < nc; j++) cw[j] = -1;
    /* infer row heights / column widths from present (non-None) blocks */
    for (int64_t i = 0; i < nr; i++) for (int64_t j = 0; j < nc; j++) {
        const tsr_arg *b = &items[i * nc + j];
        if (b->kind == 0) continue;                         /* None */
        if (b->kind != 3 || b->arr.ndim != 2) { free(rh); free(cw); fn_set_error("sparse.bmat: each block must be a 2-D array or None"); return TSR_EARG; }
        const int64_t br = b->arr.shape[0], bc = b->arr.shape[1];
        if (rh[i] < 0) rh[i] = br; else if (rh[i] != br) { free(rh); free(cw); fn_set_error("sparse.bmat: inconsistent block row heights"); return TSR_EARG; }
        if (cw[j] < 0) cw[j] = bc; else if (cw[j] != bc) { free(rh); free(cw); fn_set_error("sparse.bmat: inconsistent block column widths"); return TSR_EARG; }
    }
    for (int64_t i = 0; i < nr; i++) if (rh[i] < 0) { free(rh); free(cw); fn_set_error("sparse.bmat: a grid row has no present block"); return TSR_EARG; }
    for (int64_t j = 0; j < nc; j++) if (cw[j] < 0) { free(rh); free(cw); fn_set_error("sparse.bmat: a grid column has no present block"); return TSR_EARG; }
    int64_t R = 0, C = 0, *roff = (int64_t *)malloc((size_t)nr * sizeof(int64_t)), *coff = (int64_t *)malloc((size_t)nc * sizeof(int64_t));
    if (!roff || !coff) { free(rh); free(cw); free(roff); free(coff); return TSR_ENOMEM; }
    for (int64_t i = 0; i < nr; i++) { roff[i] = R; R += rh[i]; }
    for (int64_t j = 0; j < nc; j++) { coff[j] = C; C += cw[j]; }
    double *Dn = sp_dense(R, C);
    if (!Dn) { free(rh); free(cw); free(roff); free(coff); return TSR_ENOMEM; }
    int rc = TSR_OK;
    for (int64_t i = 0; i < nr && rc == TSR_OK; i++) for (int64_t j = 0; j < nc && rc == TSR_OK; j++) {
        const tsr_arg *b = &items[i * nc + j];
        if (b->kind == 0) continue;
        int64_t nb; double *bb = fn_arg_doubles(b, &nb);
        if (!bb) { rc = TSR_ENOMEM; break; }
        const int64_t br = b->arr.shape[0], bc = b->arr.shape[1];
        for (int64_t r = 0; r < br; r++) for (int64_t c = 0; c < bc; c++) Dn[(roff[i] + r) * C + coff[j] + c] = bb[r * bc + c];
        fn_free_doubles(bb, nb);
    }
    if (rc == TSR_OK) rc = emit_csr(res, Dn, R, C);
    free(Dn); free(rh); free(cw); free(roff); free(coff);
    return rc;
}

/* permute_dims / swapaxes on a 2-D sparse matrix: transpose (the only non-trivial 2-D permutation), as CSR.
   permute_dims(A, axes): axes is a 2-int sequence or None (reverse = transpose); (0,1) is identity.
   swapaxes(A, axis1, axis2): transpose when the two axes differ, identity otherwise. */
static int sp_transpose_or_copy(const tsr_arg *a, tsr_result *res, int transpose)
{
    if (a[0].kind != 3 || a[0].arr.ndim != 2) { fn_set_error("sparse: input must be a 2-D array"); return TSR_EARG; }
    const int64_t m = a[0].arr.shape[0], n = a[0].arr.shape[1];
    int64_t la; double *A = fn_arg_doubles(&a[0], &la);
    if (!A) return TSR_ENOMEM;
    int rc;
    if (transpose) {
        double *D = sp_dense(m, n);
        if (!D) { fn_free_doubles(A, la); return TSR_ENOMEM; }
        for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) D[j * m + i] = A[i * n + j];
        rc = emit_csr(res, D, n, m);
        free(D);
    } else {
        rc = emit_csr(res, A, m, n);
    }
    fn_free_doubles(A, la);
    return rc;
}
static int r_sp_permute_dims(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int transpose = 1;                                        /* axes=None -> reverse axes -> transpose */
    if (nargs > 1 && a[1].kind == 3 && a[1].arr.ndim == 1) {
        int64_t lo; double *ax = fn_arg_doubles(&a[1], &lo);
        if (!ax) return TSR_ENOMEM;
        if (lo >= 2 && ax[0] == 0.0 && ax[1] == 1.0) transpose = 0;   /* (0,1) identity (compared as doubles: no cast) */
        fn_free_doubles(ax, lo);
    } else if (nargs > 1 && a[1].kind == 5 && a[1].count >= 2) {
        int64_t a0 = fn_num_i64(&a[1].items[0], -1);
        int64_t a1 = fn_num_i64(&a[1].items[1], -1);
        if (a0 == 0 && a1 == 1) transpose = 0;
    }
    return sp_transpose_or_copy(a, res, transpose);
}
static int r_sp_swapaxes(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t ax1 = sp_int(a, nargs, 1, 0), ax2 = sp_int(a, nargs, 2, 1);
    return sp_transpose_or_copy(a, res, ax1 != ax2);
}


static int r_sp_expand_dims(const void *ctx, const tsr_arg *a, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const double *A; int64_t m, n, tot; double *owned;
    int rc = sp_input(&a[0], &A, &m, &n, &tot, &owned);
    if (rc < 0) return rc;
    int64_t axis = sp_int(a, nargs, 1, 0);
    const int64_t ond = 3;
    if (axis < 0) axis += ond;
    if (axis < 0 || axis >= ond) { fn_free_doubles(owned, tot); fn_set_error("sparse.expand_dims: axis out of range"); return TSR_EARG; }
    int64_t osh[3];
    {
        int64_t src[2] = {m, n};
        int64_t s = 0;
        for (int64_t d = 0; d < ond; d++) osh[d] = (d == axis) ? 1 : src[s++];
    }
    int64_t nnz = 0;
    for (int64_t i = 0; i < m * n; i++) if (A[i] != 0.0) nnz++;
    int64_t dsh[1] = {nnz}, csh[2] = {nnz, ond}, ssh[1] = {ond};
    double  *data   = (double  *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int32_t *coords = (int32_t *)fn_result_array(&res[1], TSR_I32, 2, csh);
    int64_t *shape  = (int64_t *)fn_result_array(&res[2], TSR_I64, 1, ssh);
    if (!data || !coords || !shape) { fn_free_doubles(owned, tot); return TSR_ENOMEM; }
    for (int64_t d = 0; d < ond; d++) shape[d] = osh[d];
    int64_t k = 0;
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            const double v = A[i * n + j];
            if (v == 0.0) continue;
            data[k] = v;
            int64_t src2[2] = {i, j};
            int64_t s = 0;
            for (int64_t d = 0; d < ond; d++) coords[k * ond + d] = (int32_t)((d == axis) ? 0 : src2[s++]);
            k++;
        }
    }
    fn_free_doubles(owned, tot);
    return TSR_OK;
}

static const fn_def DEFS[] = {
    ROUTINE("sparse.lil_array", 4, "A", "data, rows, indptr, shape", r_lil_array, NULL, "List-of-lists row storage of a dense matrix (scipy.sparse.lil_array)."),
    ROUTINE("sparse.lil_matrix", 4, "A", "data, rows, indptr, shape", r_lil_array, NULL, "List-of-lists row storage of a dense matrix (scipy.sparse.lil_matrix)."),
    ROUTINE("sparse.dok_array", 4, "A", "data, row, col, shape", r_dok_array, NULL, "Dictionary-of-keys storage of a dense matrix (scipy.sparse.dok_array)."),
    ROUTINE("sparse.dok_matrix", 4, "A", "data, row, col, shape", r_dok_array, NULL, "Dictionary-of-keys storage of a dense matrix (scipy.sparse.dok_matrix)."),
    ROUTINE("sparse.bsr_array", 5, "A, R=1, C=1", "data, indices, indptr, shape, blocksize", r_bsr_array, NULL, "Block sparse row storage of a dense matrix (scipy.sparse.bsr_array)."),
    ROUTINE("sparse.bsr_matrix", 5, "A, R=1, C=1", "data, indices, indptr, shape, blocksize", r_bsr_array, NULL, "Block sparse row storage of a dense matrix (scipy.sparse.bsr_matrix)."),
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
    ROUTINE("sparse.eye", 4, "m, n=None, k=0", "data, indices, indptr, shape", r_sp_eye, NULL, "Sparse matrix with ones on a diagonal, as CSR (scipy.sparse.eye)."),
    ROUTINE("sparse.eye_array", 4, "m, n=None, k=0", "data, indices, indptr, shape", r_sp_eye, NULL, "Sparse matrix with ones on a diagonal, as CSR (scipy.sparse.eye_array)."),
    ROUTINE("sparse.identity", 4, "n", "data, indices, indptr, shape", r_sp_identity, NULL, "Sparse identity matrix, as CSR (scipy.sparse.identity)."),
    ROUTINE("sparse.kron", 4, "A, B", "data, indices, indptr, shape", r_sp_kron, NULL, "Kronecker product of two matrices, as CSR (scipy.sparse.kron)."),
    ROUTINE("sparse.kronsum", 4, "A, B", "data, indices, indptr, shape", r_sp_kronsum, NULL, "Kronecker sum of two square matrices, as CSR (scipy.sparse.kronsum)."),
    ROUTINE("sparse.tril", 4, "A, k=0", "data, indices, indptr, shape", r_sp_tril, NULL, "Lower-triangular part of a matrix, as CSR (scipy.sparse.tril)."),
    ROUTINE("sparse.triu", 4, "A, k=0", "data, indices, indptr, shape", r_sp_triu, NULL, "Upper-triangular part of a matrix, as CSR (scipy.sparse.triu)."),
    ROUTINE("sparse.find", 3, "A", "row, col, data", r_sp_find, NULL, "Row, column and value of each nonzero, row-major (scipy.sparse.find)."),
    ROUTINE("sparse.spdiags", 4, "data, diags, m, n", "data, indices, indptr, shape", r_sp_spdiags, NULL, "Sparse matrix from diagonals, as CSR (scipy.sparse.spdiags)."),
    ROUTINE("sparse.block_diag", 4, "*blocks", "data, indices, indptr, shape", r_sp_block_diag, NULL, "Block-diagonal assembly of blocks, as CSR (scipy.sparse.block_diag)."),
    ROUTINE("sparse.diags", 4, "diagonals[], offsets=0, shape=None", "data, indices, indptr, shape", r_sp_diags, NULL, "Sparse matrix from diagonals, as CSR (scipy.sparse.diags)."),
    ROUTINE("sparse.diags_array", 4, "diagonals[], offsets=0, shape=None", "data, indices, indptr, shape", r_sp_diags, NULL, "Sparse matrix from diagonals, as CSR (scipy.sparse.diags_array)."),
    ROUTINE("sparse.bmat", 4, "blocks[], nrows, ncols", "data, indices, indptr, shape", r_sp_bmat, NULL, "Assemble a 2-D grid of blocks, as CSR (scipy.sparse.bmat)."),
    ROUTINE("sparse.block_array", 4, "blocks[], nrows, ncols", "data, indices, indptr, shape", r_sp_bmat, NULL, "Assemble a 2-D grid of blocks, as CSR (scipy.sparse.block_array)."),
    ROUTINE("sparse.permute_dims", 4, "A, axes=None", "data, indices, indptr, shape", r_sp_permute_dims, NULL, "Permute the axes of a 2-D sparse matrix (transpose), as CSR (scipy.sparse.permute_dims)."),
    ROUTINE("sparse.swapaxes", 4, "A, axis1=0, axis2=1", "data, indices, indptr, shape", r_sp_swapaxes, NULL, "Swap two axes of a 2-D sparse matrix (transpose), as CSR (scipy.sparse.swapaxes)."),
    ROUTINE("sparse.expand_dims", 3, "A, axis=0", "data, coords, shape", r_sp_expand_dims, NULL, "Insert a length-1 axis into a 2-D array, returning N-D coordinate storage (scipy.sparse.expand_dims)."),
};

const fn_table TSR_SCIPY_SPARSE_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
