/* Compressed sparse row (float64) kernels and Krylov solvers. */
#include "internal.h"
#include <math.h>

TSR_CLONES int tsr_csr_matvec(int64_t rows, const int64_t *indptr, const int64_t *indices, const double *data,
                              const double *x, double *y)
{
    if (rows < 0 || !indptr) return TSR_EARG;
    for (int64_t i = 0; i < rows; i++) {
        double acc = 0.0;
        for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) acc += data[p] * x[indices[p]];
        y[i] = acc;
    }
    return TSR_OK;
}

TSR_CLONES int tsr_csr_matmat(int64_t rows, int64_t k, const int64_t *indptr, const int64_t *indices,
                              const double *data, const double *b, double *c)
{
    if (rows < 0 || k < 0 || !indptr) return TSR_EARG;
    for (int64_t i = 0; i < rows; i++) {
        double *ci = c + i * k;
        memset(ci, 0, (size_t)k * sizeof(double));
        for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) {
            const double v = data[p];
            const double *bj = b + indices[p] * k;
            for (int64_t j = 0; j < k; j++) ci[j] += v * bj[j];
        }
    }
    return TSR_OK;
}

int64_t tsr_dense_nnz(int64_t rows, int64_t cols, const double *a)
{
    int64_t nnz = 0;
    for (int64_t i = 0; i < rows * cols; i++) nnz += a[i] != 0.0;
    return nnz;
}

int tsr_dense_to_csr(int64_t rows, int64_t cols, const double *a, int64_t *indptr, int64_t *indices, double *data)
{
    int64_t nnz = 0;
    indptr[0] = 0;
    for (int64_t i = 0; i < rows; i++) {
        for (int64_t j = 0; j < cols; j++) {
            double v = a[i * cols + j];
            if (v != 0.0) {
                indices[nnz] = j;
                data[nnz] = v;
                nnz++;
            }
        }
        indptr[i + 1] = nnz;
    }
    return TSR_OK;
}

/* Counting-sort transpose; output column indices come out sorted. */
int tsr_csr_transpose(int64_t rows, int64_t cols, const int64_t *indptr, const int64_t *indices, const double *data,
                      int64_t *t_indptr, int64_t *t_indices, double *t_data)
{
    const int64_t nnz = indptr[rows];
    memset(t_indptr, 0, (size_t)(cols + 1) * sizeof(int64_t));
    for (int64_t p = 0; p < nnz; p++) {
        if (indices[p] < 0 || indices[p] >= cols) return TSR_EINDEX;
        t_indptr[indices[p] + 1]++;
    }
    for (int64_t j = 0; j < cols; j++) t_indptr[j + 1] += t_indptr[j];
    int64_t *next = (int64_t *)tsr_alloc((cols + 1) * (int64_t)sizeof(int64_t));
    if (!next) return TSR_ENOMEM;
    memcpy(next, t_indptr, (size_t)(cols + 1) * sizeof(int64_t));
    for (int64_t i = 0; i < rows; i++)
        for (int64_t p = indptr[i]; p < indptr[i + 1]; p++) {
            int64_t dst = next[indices[p]]++;
            t_indices[dst] = i;
            t_data[dst] = data[p];
        }
    tsr_free(next, (cols + 1) * (int64_t)sizeof(int64_t));
    return TSR_OK;
}

static double dot(int64_t n, const double *a, const double *b)
{
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

/*
 * Conjugate gradients with Jacobi preconditioning for SPD A (n x n).
 * x holds the initial guess on entry and the solution on exit. Returns the
 * iteration count, or -(iterations) when maxiter was reached without
 * ||r|| <= max(tol * ||b||, atol); *resid receives the final residual norm.
 */
int64_t tsr_csr_cg(int64_t n, const int64_t *indptr, const int64_t *indices, const double *data, const double *b,
                   double *x, double tol, double atol, int64_t maxiter, double *resid)
{
    int64_t bytes = 5 * n * (int64_t)sizeof(double);
    double *mem = (double *)tsr_alloc(bytes > 0 ? bytes : 8);
    if (!mem) return TSR_ENOMEM;
    double *r = mem, *z = mem + n, *p = mem + 2 * n, *q = mem + 3 * n, *dinv = mem + 4 * n;
    for (int64_t i = 0; i < n; i++) {
        double d = 0.0;
        for (int64_t k = indptr[i]; k < indptr[i + 1]; k++)
            if (indices[k] == i) d += data[k];
        dinv[i] = d != 0.0 ? 1.0 / d : 1.0;
    }

    tsr_csr_matvec(n, indptr, indices, data, x, q);
    for (int64_t i = 0; i < n; i++) r[i] = b[i] - q[i];
    const double bnorm = sqrt(dot(n, b, b));
    const double target = fmax(tol * bnorm, atol);

    for (int64_t i = 0; i < n; i++) { z[i] = r[i] * dinv[i]; p[i] = z[i]; }
    double rz = dot(n, r, z);
    double rn = sqrt(dot(n, r, r));
    int64_t it = 0;
    while (rn > target && it < maxiter) {
        tsr_csr_matvec(n, indptr, indices, data, p, q);
        double pq = dot(n, p, q);
        if (pq == 0.0) break;
        double alpha = rz / pq;
        for (int64_t i = 0; i < n; i++) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        for (int64_t i = 0; i < n; i++) z[i] = r[i] * dinv[i];
        double rz_new = dot(n, r, z);
        double beta = rz_new / rz;
        rz = rz_new;
        for (int64_t i = 0; i < n; i++) p[i] = z[i] + beta * p[i];
        rn = sqrt(dot(n, r, r));
        it++;
    }
    if (resid) *resid = rn;
    tsr_free(mem, bytes > 0 ? bytes : 8);
    return rn <= target ? it : -(it > 0 ? it : 1);
}

/* BiCGSTAB for general square A; same conventions as tsr_csr_cg. */
int64_t tsr_csr_bicgstab(int64_t n, const int64_t *indptr, const int64_t *indices, const double *data, const double *b,
                         double *x, double tol, double atol, int64_t maxiter, double *resid)
{
    int64_t bytes = 6 * n * (int64_t)sizeof(double);
    double *mem = (double *)tsr_alloc(bytes > 0 ? bytes : 8);
    if (!mem) return TSR_ENOMEM;
    double *r = mem, *rh = mem + n, *p = mem + 2 * n, *v = mem + 3 * n, *s = mem + 4 * n, *t = mem + 5 * n;
    tsr_csr_matvec(n, indptr, indices, data, x, v);
    for (int64_t i = 0; i < n; i++) { r[i] = b[i] - v[i]; rh[i] = r[i]; p[i] = 0.0; v[i] = 0.0; }
    const double target = fmax(tol * sqrt(dot(n, b, b)), atol);
    double rho = 1.0, alpha = 1.0, omega = 1.0;
    double rn = sqrt(dot(n, r, r));
    int64_t it = 0;
    while (rn > target && it < maxiter) {
        double rho_new = dot(n, rh, r);
        if (rho_new == 0.0) break;
        double beta = (rho_new / rho) * (alpha / omega);
        rho = rho_new;
        for (int64_t i = 0; i < n; i++) p[i] = r[i] + beta * (p[i] - omega * v[i]);
        tsr_csr_matvec(n, indptr, indices, data, p, v);
        double rv = dot(n, rh, v);
        if (rv == 0.0) break;
        alpha = rho / rv;
        for (int64_t i = 0; i < n; i++) s[i] = r[i] - alpha * v[i];
        double sn = sqrt(dot(n, s, s));
        if (sn <= target) {
            for (int64_t i = 0; i < n; i++) x[i] += alpha * p[i];
            rn = sn;
            it++;
            break;
        }
        tsr_csr_matvec(n, indptr, indices, data, s, t);
        double tt = dot(n, t, t);
        omega = tt != 0.0 ? dot(n, t, s) / tt : 0.0;
        for (int64_t i = 0; i < n; i++) { x[i] += alpha * p[i] + omega * s[i]; r[i] = s[i] - omega * t[i]; }
        rn = sqrt(dot(n, r, r));
        it++;
        if (omega == 0.0) break;
    }
    if (resid) *resid = rn;
    tsr_free(mem, bytes > 0 ? bytes : 8);
    return rn <= target ? it : -(it > 0 ? it : 1);
}
