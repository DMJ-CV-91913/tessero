/* numpy polynomial helpers: polyfit (least-squares fit), polyval (Horner), roots (companion-matrix eigenvalues).
 * Coefficients are highest-degree-first, as NumPy uses them. Real float64 in; roots returns complex128. */
#include "fn.h"

#include <lapacke.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* polyfit(x, y, deg): least-squares polynomial coefficients, highest degree first (numpy.polyfit). */
static int r_polyfit(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("polyfit: x and y must be arrays"); return TSR_EARG; }
    int64_t deg = 1;
    if (nargs > 2 && args[2].kind == 1) deg = (args[2].flags & 1) ? args[2].ival : (int64_t)args[2].num;
    if (deg < 0) { fn_set_error("polyfit: deg must be >= 0"); return TSR_EARG; }
    int64_t nx, ny;
    double *x = fn_arg_doubles(&args[0], &nx);
    if (!x) return TSR_ENOMEM;
    double *y = fn_arg_doubles(&args[1], &ny);
    if (!y) { fn_free_doubles(x, nx); return TSR_ENOMEM; }
    if (nx != ny) { fn_free_doubles(x, nx); fn_free_doubles(y, ny); fn_set_error("polyfit: x and y have different lengths"); return TSR_EARG; }
    const int64_t n = nx, m = deg + 1;
    double *V = (double *)malloc(sizeof(double) * (size_t)(n * m));
    const int64_t ldb = n > m ? n : m;
    double *b = (double *)calloc((size_t)ldb, sizeof(double));
    if (!V || !b) { free(V); free(b); fn_free_doubles(x, nx); fn_free_doubles(y, ny); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) {
        double p = 1.0;
        for (int64_t j = m - 1; j >= 0; j--) { V[i * m + j] = p; p *= x[i]; }   /* V[i][j] = x[i]^(deg-j) */
        b[i] = y[i];
    }
    fn_free_doubles(x, nx); fn_free_doubles(y, ny);
    lapack_int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', (lapack_int)n, (lapack_int)m, 1, V, (lapack_int)m, b, 1);
    free(V);
    if (info != 0) { free(b); fn_set_error("polyfit: the least-squares solve failed (%d)", (int)info); return TSR_EARG; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){m});
    if (!out) { free(b); return TSR_ENOMEM; }
    memcpy(out, b, sizeof(double) * (size_t)m);
    free(b);
    return TSR_OK;
}

/* comparator for sort_complex-style ordering: real then imaginary */
static int cplx_cmp(const void *a, const void *b)
{
    const double *x = (const double *)a, *y = (const double *)b;
    if (x[0] < y[0]) return -1;
    if (x[0] > y[0]) return 1;
    if (x[1] < y[1]) return -1;
    if (x[1] > y[1]) return 1;
    return 0;
}

/* roots(p): the roots of the polynomial with coefficients p (highest first), as complex128 sorted by (re, im)
   (numpy.roots, canonicalised to a stable order). Companion-matrix eigenvalues via dgeev. */
static int r_roots(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("roots: p must be an array"); return TSR_EARG; }
    int64_t m;
    double *p = fn_arg_doubles(&args[0], &m);
    if (!p) return TSR_ENOMEM;
    int64_t s = 0;
    while (s < m && p[s] == 0.0) s++;                      /* strip leading zeros */
    const int64_t n = (m - s) - 1;                         /* number of roots */
    if (n <= 0) { fn_free_doubles(p, m); return fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){0}) ? TSR_OK : TSR_ENOMEM; }
    double *comp = (double *)calloc((size_t)(n * n), sizeof(double));
    double *wr = (double *)malloc(sizeof(double) * (size_t)n), *wi = (double *)malloc(sizeof(double) * (size_t)n);
    if (!comp || !wr || !wi) { free(comp); free(wr); free(wi); fn_free_doubles(p, m); return TSR_ENOMEM; }
    const double p0 = p[s];
    for (int64_t j = 0; j < n; j++) comp[0 * n + j] = -p[s + 1 + j] / p0;   /* first row */
    for (int64_t i = 1; i < n; i++) comp[i * n + (i - 1)] = 1.0;            /* sub-diagonal */
    fn_free_doubles(p, m);
    lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)n, comp, (lapack_int)n, wr, wi, NULL, 1, NULL, 1);
    free(comp);
    if (info != 0) { free(wr); free(wi); fn_set_error("roots: the eigenvalue solve failed (%d)", (int)info); return TSR_EARG; }
    double *pairs = (double *)malloc(sizeof(double) * (size_t)(2 * n));
    for (int64_t i = 0; i < n; i++) { pairs[2 * i] = wr[i]; pairs[2 * i + 1] = wi[i]; }
    free(wr); free(wi);
    qsort(pairs, (size_t)n, 2 * sizeof(double), cplx_cmp);
    double *out = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){n});
    if (!out) { free(pairs); return TSR_ENOMEM; }
    memcpy(out, pairs, sizeof(double) * (size_t)(2 * n));
    free(pairs);
    return TSR_OK;
}

static const fn_def DEFS[] = {
    ROUTINE("np.polyfit", 1, "x, y, deg", "out", r_polyfit, NULL, "Least-squares polynomial fit, highest-degree coefficient first (numpy.polyfit)."),
    ROUTINE("np.roots", 1, "p", "out", r_roots, NULL, "Roots of a polynomial, complex128 sorted by (re, im) (numpy.roots)."),
};

const fn_table TSR_NP_POLY_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
