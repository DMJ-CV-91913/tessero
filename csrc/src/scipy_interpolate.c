/* scipy.interpolate ("interpolate." prefix -> Tessero\Interpolate facade).
 *
 *   interpolate.pchip_interpolate(xi, yi, x)     monotone piecewise-cubic (PCHIP) interpolation
 *   interpolate.PchipInterpolator(x, y, xnew)    PCHIP, evaluated form of the class
 *   interpolate.Akima1DInterpolator(x, y, xnew)  Akima piecewise-cubic
 *   interpolate.CubicSpline(x, y, xnew)          not-a-knot cubic spline
 *   interpolate.CubicHermiteSpline(x, y, dydx, xnew)   cubic Hermite with given derivatives
 *
 * Each builds the per-knot first derivatives d[k] its own way, then evaluates the piecewise cubic in the
 * Hermite basis (algebraically identical to SciPy's PPoly/BPoly forms up to basis-level rounding). The node
 * derivatives replicate SciPy's constructions exactly: PCHIP Fritsch-Carlson, Akima's extended-slope blend,
 * and the not-a-knot tridiagonal system (solved with the shared LAPACK dense solver).
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

extern int sl_dense_solve(double *M, double *b, int64_t n, int64_t nrhs);   /* np_linalg.c (row-major dgesv) */

static double ip_sign(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }

/* Evaluate a piecewise cubic Hermite spline (knots x, values y, node derivatives d) at the points xq. */
static void ip_hermite(const double *x, const double *y, const double *d, int64_t n,
                       const double *xq, int64_t q, double *out)
{
    for (int64_t j = 0; j < q; j++) {
        const double v = xq[j];
        int64_t lo = 0, hi = n;                            /* searchsorted(x, v, side='right') */
        while (lo < hi) { const int64_t mid = (lo + hi) / 2; if (x[mid] <= v) lo = mid + 1; else hi = mid; }
        int64_t idx = lo - 1;
        if (idx < 0) idx = 0; else if (idx > n - 2) idx = n - 2;
        const double h = x[idx + 1] - x[idx], s = (v - x[idx]) / h, s2 = s * s, s3 = s2 * s;
        const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0, h01 = -2.0 * s3 + 3.0 * s2;
        const double h10 = s3 - 2.0 * s2 + s, h11 = s3 - s2;
        out[j] = y[idx] * h00 + y[idx + 1] * h01 + d[idx] * h * h10 + d[idx + 1] * h * h11;
    }
}

/* SciPy PchipInterpolator._edge_case + _find_derivatives: shape-preserving monotone slopes d[0..n-1]. */
static double pchip_edge(double h0, double h1, double m0, double m1)
{
    double d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (ip_sign(d) != ip_sign(m0)) d = 0.0;
    else if (ip_sign(m0) != ip_sign(m1) && fabs(d) > 3.0 * fabs(m0)) d = 3.0 * m0;
    return d;
}

static int ip_pchip_deriv(const double *x, const double *y, int64_t n, double *d)
{
    double *hk = (double *)malloc((size_t)(n - 1) * sizeof(double));
    double *mk = (double *)malloc((size_t)(n - 1) * sizeof(double));
    if (!hk || !mk) { free(hk); free(mk); return TSR_ENOMEM; }
    for (int64_t k = 0; k < n - 1; k++) { hk[k] = x[k + 1] - x[k]; mk[k] = (y[k + 1] - y[k]) / hk[k]; }
    if (n == 2) { d[0] = mk[0]; d[1] = mk[0]; free(hk); free(mk); return TSR_OK; }
    for (int64_t i = 0; i < n - 2; i++) {
        const int cond = (ip_sign(mk[i + 1]) != ip_sign(mk[i])) || mk[i + 1] == 0.0 || mk[i] == 0.0;
        if (cond) d[i + 1] = 0.0;
        else {
            const double w1 = 2.0 * hk[i + 1] + hk[i], w2 = hk[i + 1] + 2.0 * hk[i];
            d[i + 1] = 1.0 / ((w1 / mk[i] + w2 / mk[i + 1]) / (w1 + w2));
        }
    }
    d[0] = pchip_edge(hk[0], hk[1], mk[0], mk[1]);
    d[n - 1] = pchip_edge(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3]);
    free(hk); free(mk);
    return TSR_OK;
}

/* SciPy Akima1DInterpolator (method='akima'): extended slope array with the shape-preserving blend. */
static int ip_akima_deriv(const double *x, const double *y, int64_t n, double *t)
{
    if (n == 2) { const double m = (y[1] - y[0]) / (x[1] - x[0]); t[0] = m; t[1] = m; return TSR_OK; }
    double *M = (double *)malloc((size_t)(n + 3) * sizeof(double));     /* indices 0..n+2 */
    double *dm = (double *)malloc((size_t)(n + 2) * sizeof(double));
    if (!M || !dm) { free(M); free(dm); return TSR_ENOMEM; }
    for (int64_t j = 0; j < n - 1; j++) M[2 + j] = (y[j + 1] - y[j]) / (x[j + 1] - x[j]);
    M[1] = 2.0 * M[2] - M[3]; M[0] = 2.0 * M[1] - M[2];
    M[n + 1] = 2.0 * M[n] - M[n - 1]; M[n + 2] = 2.0 * M[n + 1] - M[n];
    for (int64_t j = 0; j < n + 2; j++) dm[j] = fabs(M[j + 1] - M[j]);
    double mmax = -INFINITY;
    for (int64_t k = 0; k < n; k++) { const double f12 = dm[k] + dm[k + 2]; if (f12 > mmax) mmax = f12; }
    for (int64_t k = 0; k < n; k++) {
        const double f2 = dm[k], f1 = dm[k + 2], f12 = f1 + f2;
        if (f12 > 1e-9 * mmax) t[k] = M[k + 1] + (f2 / f12) * (M[k + 2] - M[k + 1]);
        else t[k] = 0.5 * (M[k] + M[k + 3]);
    }
    free(M); free(dm);
    return TSR_OK;
}

/* SciPy CubicSpline (bc_type='not-a-knot'): node derivatives s[0..n-1] from a tridiagonal system (dense solve). */
static int ip_cubicspline_deriv(const double *x, const double *y, int64_t n, double *s)
{
    double *dx = (double *)malloc((size_t)(n - 1) * sizeof(double));
    double *sl = (double *)malloc((size_t)(n - 1) * sizeof(double));
    if (!dx || !sl) { free(dx); free(sl); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n - 1; i++) { dx[i] = x[i + 1] - x[i]; sl[i] = (y[i + 1] - y[i]) / dx[i]; }
    if (n == 2) { s[0] = sl[0]; s[1] = sl[0]; free(dx); free(sl); return TSR_OK; }
    double *M = (double *)calloc((size_t)(n * n), sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    int rc = (!M || !b) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        if (n == 3) {                                      /* not-a-knot n==3: parabola through the points */
            M[0 * 3 + 0] = 1.0; M[0 * 3 + 1] = 1.0; b[0] = 2.0 * sl[0];
            M[1 * 3 + 0] = dx[1]; M[1 * 3 + 1] = 2.0 * (dx[0] + dx[1]); M[1 * 3 + 2] = dx[0];
            b[1] = 3.0 * (dx[0] * sl[1] + dx[1] * sl[0]);
            M[2 * 3 + 1] = 1.0; M[2 * 3 + 2] = 1.0; b[2] = 2.0 * sl[1];
        } else {
            const double d0 = x[2] - x[0];
            M[0] = dx[1]; M[1] = d0; b[0] = ((dx[0] + 2.0 * d0) * dx[1] * sl[0] + dx[0] * dx[0] * sl[1]) / d0;
            for (int64_t i = 1; i < n - 1; i++) {
                M[i * n + (i - 1)] = dx[i]; M[i * n + i] = 2.0 * (dx[i - 1] + dx[i]); M[i * n + (i + 1)] = dx[i - 1];
                b[i] = 3.0 * (dx[i] * sl[i - 1] + dx[i - 1] * sl[i]);
            }
            const double de = x[n - 1] - x[n - 3];
            M[(n - 1) * n + (n - 2)] = de; M[(n - 1) * n + (n - 1)] = dx[n - 3];
            b[n - 1] = (dx[n - 2] * dx[n - 2] * sl[n - 3] + (2.0 * de + dx[n - 2]) * dx[n - 3] * sl[n - 2]) / de;
        }
        rc = sl_dense_solve(M, b, n, 1);
        if (rc == TSR_OK) for (int64_t i = 0; i < n; i++) s[i] = b[i];
        else fn_set_error("CubicSpline: the spline system is singular");
    }
    free(M); free(b); free(dx); free(sl);
    return rc;
}

/* common entry: read x, y (and query), build derivatives via `deriv`, Hermite-evaluate at xnew. `der_given`
   points to an externally supplied derivative array (CubicHermiteSpline) when deriv is NULL. */
static int ip_spline_eval(const tsr_arg *args, tsr_result *res, int (*deriv)(const double *, const double *, int64_t, double *),
                          int has_dydx, const char *name)
{
    const int qi = has_dydx ? 3 : 2;                       /* index of the query array */
    for (int i = 0; i <= qi; i++) if (args[i].kind != 3 || args[i].arr.ndim != 1) { fn_set_error("%s: inputs must be 1-D arrays", name); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], qn = args[qi].arr.shape[0];
    if (args[1].arr.shape[0] != n || (has_dydx && args[2].arr.shape[0] != n)) { fn_set_error("%s: x, y (and dydx) must have the same length", name); return TSR_EARG; }
    if (n < 2) { fn_set_error("%s: need at least two sample points", name); return TSR_EARG; }
    int64_t lx, ly, lq, ld = 0;
    double *x = fn_arg_doubles(&args[0], &lx);
    double *y = x ? fn_arg_doubles(&args[1], &ly) : NULL;
    double *dydx = (y && has_dydx) ? fn_arg_doubles(&args[2], &ld) : NULL;
    double *xq = (y && (!has_dydx || dydx)) ? fn_arg_doubles(&args[qi], &lq) : NULL;
    double *d = xq ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *out = d ? (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){qn}) : NULL;
    int rc = (!x || !y || (has_dydx && !dydx) || !xq || !d || !out) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        if (has_dydx) memcpy(d, dydx, (size_t)n * sizeof(double));
        else rc = deriv(x, y, n, d);
    }
    if (rc == TSR_OK) ip_hermite(x, y, d, n, xq, qn, out);
    free(d);
    fn_free_doubles(x, lx); fn_free_doubles(y, ly); if (dydx) fn_free_doubles(dydx, ld); fn_free_doubles(xq, lq);
    return rc;
}

static int r_pchip_interpolate(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nargs; (void)nres; return ip_spline_eval(args, res, ip_pchip_deriv, 0, "pchip_interpolate"); }

static int r_pchip_class(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nargs; (void)nres; return ip_spline_eval(args, res, ip_pchip_deriv, 0, "PchipInterpolator"); }

static int r_akima(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nargs; (void)nres; return ip_spline_eval(args, res, ip_akima_deriv, 0, "Akima1DInterpolator"); }

static int r_cubicspline(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nargs; (void)nres; return ip_spline_eval(args, res, ip_cubicspline_deriv, 0, "CubicSpline"); }

static int r_cubic_hermite(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nargs; (void)nres; return ip_spline_eval(args, res, NULL, 1, "CubicHermiteSpline"); }

/* barycentric_interpolate(xi, yi, x): barycentric Lagrange interpolation (scipy.interpolate, der=0). The global
   scale of the weights cancels in the barycentric quotient, so the evaluated values match SciPy's scaled form. */
static int r_barycentric(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    for (int i = 0; i < 3; i++) if (args[i].kind != 3 || args[i].arr.ndim != 1) { fn_set_error("barycentric_interpolate: xi, yi and x must be 1-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], q = args[2].arr.shape[0];
    if (args[1].arr.shape[0] != n) { fn_set_error("barycentric_interpolate: xi and yi must have the same length"); return TSR_EARG; }
    if (n < 1) { fn_set_error("barycentric_interpolate: need at least one sample point"); return TSR_EARG; }
    int64_t lx, ly, lq;
    double *x = fn_arg_doubles(&args[0], &lx), *y = x ? fn_arg_doubles(&args[1], &ly) : NULL, *xq = y ? fn_arg_doubles(&args[2], &lq) : NULL;
    double *w = xq ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *out = w ? (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){q}) : NULL;
    int rc = (!x || !y || !xq || !w || !out) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        for (int64_t j = 0; j < n; j++) { double p = 1.0; for (int64_t k = 0; k < n; k++) if (k != j) p *= (x[j] - x[k]); w[j] = 1.0 / p; }
        for (int64_t t = 0; t < q; t++) {
            const double v = xq[t];
            int64_t hit = -1;
            for (int64_t j = 0; j < n; j++) if (v == x[j]) { hit = j; break; }
            if (hit >= 0) out[t] = y[hit];
            else { double num = 0.0, den = 0.0; for (int64_t j = 0; j < n; j++) { const double c = w[j] / (v - x[j]); num += c * y[j]; den += c; } out[t] = num / den; }
        }
    }
    free(w); fn_free_doubles(x, lx); fn_free_doubles(y, ly); fn_free_doubles(xq, lq);
    return rc;
}

/* krogh_interpolate(xi, yi, x): polynomial interpolation via Newton divided differences (scipy.interpolate,
   der=0). The interpolating polynomial is unique, so the values match SciPy's KroghInterpolator. */
static int r_krogh(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    for (int i = 0; i < 3; i++) if (args[i].kind != 3 || args[i].arr.ndim != 1) { fn_set_error("krogh_interpolate: xi, yi and x must be 1-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], q = args[2].arr.shape[0];
    if (args[1].arr.shape[0] != n) { fn_set_error("krogh_interpolate: xi and yi must have the same length"); return TSR_EARG; }
    if (n < 1) { fn_set_error("krogh_interpolate: need at least one sample point"); return TSR_EARG; }
    int64_t lx, ly, lq;
    double *x = fn_arg_doubles(&args[0], &lx), *y = x ? fn_arg_doubles(&args[1], &ly) : NULL, *xq = y ? fn_arg_doubles(&args[2], &lq) : NULL;
    double *c = xq ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *out = c ? (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){q}) : NULL;
    int rc = (!x || !y || !xq || !c || !out) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i < n; i++) c[i] = y[i];                          /* Newton divided differences */
        for (int64_t j = 1; j < n; j++) for (int64_t i = n - 1; i >= j; i--) c[i] = (c[i] - c[i - 1]) / (x[i] - x[i - j]);
        for (int64_t t = 0; t < q; t++) { const double v = xq[t]; double p = c[n - 1]; for (int64_t k = n - 2; k >= 0; k--) p = p * (v - x[k]) + c[k]; out[t] = p; }
    }
    free(c); fn_free_doubles(x, lx); fn_free_doubles(y, ly); fn_free_doubles(xq, lq);
    return rc;
}

/* Evaluate a spline in B-spline form (knots t, coefficients c, degree k) at x via the de Boor recurrence --
   shared by BSpline and splev. Args are (t, c, k, x). */
static int ip_bspline_eval(const tsr_arg *args, tsr_result *res, const char *name)
{
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1 ||
        args[3].kind != 3 || args[3].arr.ndim != 1) { fn_set_error("%s: t, c and x must be 1-D arrays", name); return TSR_EARG; }
    if (args[2].kind != 1) { fn_set_error("%s: k must be an integer degree", name); return TSR_EARG; }
    const int64_t nt = args[0].arr.shape[0], nc = args[1].arr.shape[0], q = args[3].arr.shape[0];
    const int k = (int)args[2].num;
    if (k < 0 || nc < k + 1 || nt < nc + k + 1) { fn_set_error("%s: inconsistent (t, c, k) sizes", name); return TSR_EARG; }
    int64_t lt, lc, lq;
    double *t = fn_arg_doubles(&args[0], &lt), *c = t ? fn_arg_doubles(&args[1], &lc) : NULL, *xq = c ? fn_arg_doubles(&args[3], &lq) : NULL;
    double *out = xq ? (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){q}) : NULL;
    double *d = out ? (double *)malloc((size_t)(k + 1) * sizeof(double)) : NULL;
    int rc = (!t || !c || !xq || !out || !d) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        const int64_t n = nc;                              /* number of coefficients; domain [t[k], t[n]] */
        for (int64_t e = 0; e < q; e++) {
            const double x = xq[e];
            int64_t mu = k;                                /* knot interval: t[mu] <= x < t[mu+1], clamped to [k, n-1] */
            while (mu < n - 1 && t[mu + 1] <= x) mu++;
            for (int j = 0; j <= k; j++) d[j] = c[mu - k + j];
            for (int r = 1; r <= k; r++)
                for (int j = k; j >= r; j--) {
                    const double denom = t[j + mu - r + 1] - t[j + mu - k];
                    const double a = denom != 0.0 ? (x - t[j + mu - k]) / denom : 0.0;
                    d[j] = (1.0 - a) * d[j - 1] + a * d[j];
                }
            out[e] = d[k];
        }
    }
    free(d); fn_free_doubles(t, lt); fn_free_doubles(c, lc); fn_free_doubles(xq, lq);
    return rc;
}

/* BSpline(t, c, k, x) / splev(t, c, k, x): evaluate a B-spline via de Boor (scipy.interpolate.BSpline / splev). */
static int r_bspline(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nargs; (void)nres; return ip_bspline_eval(args, res, "BSpline"); }

static int r_splev(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nargs; (void)nres; return ip_bspline_eval(args, res, "splev"); }

/* pade(an, m, n=None): Pade [m/n] rational approximant from Taylor coefficients an (scipy.interpolate.pade).
   Builds C = [eye(N+1, n+1) | Bkj], solves C @ pq = an, returns numerator p and denominator q coefficients in
   highest-degree-first order (as numpy.poly1d.coeffs). N = m+n. */
static int r_pade(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("pade: an must be a 1-D array"); return TSR_EARG; }
    if (args[1].kind != 1) { fn_set_error("pade: m must be an integer"); return TSR_EARG; }
    const int64_t L = args[0].arr.shape[0], m = (int64_t)args[1].num;
    const int64_t n = (nargs > 2 && args[2].kind == 1) ? (int64_t)args[2].num : (L - 1 - m);
    if (m < 0 || n < 0) { fn_set_error("pade: m and n must be nonnegative"); return TSR_EARG; }
    const int64_t N = m + n;
    if (N > L - 1) { fn_set_error("pade: m + n must be smaller than len(an)"); return TSR_EARG; }
    int64_t la; double *an = fn_arg_doubles(&args[0], &la);
    if (!an) return TSR_ENOMEM;
    const int64_t S = N + 1;
    double *C = (double *)calloc((size_t)(S * S), sizeof(double));
    double *b = (double *)malloc((size_t)S * sizeof(double));
    int rc = (!C || !b) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i <= n; i++) C[i * S + i] = 1.0;                  /* Akj = eye(S, n+1) */
        for (int64_t row = 1; row <= N; row++) {                             /* Bkj (columns n+1..n+m) */
            const int64_t cmax = (row <= m) ? row : m;
            for (int64_t cc = 0; cc < cmax; cc++) C[row * S + (n + 1 + cc)] = -an[row - 1 - cc];
        }
        for (int64_t i = 0; i < S; i++) b[i] = an[i];
        rc = sl_dense_solve(C, b, S, 1);
        if (rc != TSR_OK) fn_set_error("pade: the Pade linear system is singular");
    }
    if (rc == TSR_OK) {
        double *rp = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){n + 1});
        double *rq = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){m + 1});
        if (!rp || !rq) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i <= n; i++) rp[i] = b[n - i];                /* p[::-1] */
            for (int64_t i = 0; i <= m; i++) { const int64_t j = m - i; rq[i] = (j == 0) ? 1.0 : b[n + j]; }  /* q[::-1], q=[1, pq[n+1:]] */
        }
    }
    free(C); free(b); fn_free_doubles(an, la);
    return rc;
}

/* make_interp_spline(x, y, k, xnew): interpolating spline evaluated at xnew (scipy.interpolate.make_interp_spline,
   default bc). k=1 is piecewise linear; k=3 is the not-a-knot cubic (same spline as CubicSpline, evaluated via
   its node derivatives). Other degrees are not supported here. */
static int r_make_interp_spline(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1 ||
        args[3].kind != 3 || args[3].arr.ndim != 1) { fn_set_error("make_interp_spline: x, y and xnew must be 1-D arrays"); return TSR_EARG; }
    if (args[2].kind != 1) { fn_set_error("make_interp_spline: k must be an integer degree"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], q = args[3].arr.shape[0];
    const int k = (int)args[2].num;
    if (args[1].arr.shape[0] != n) { fn_set_error("make_interp_spline: x and y must have the same length"); return TSR_EARG; }
    if (n < 2) { fn_set_error("make_interp_spline: need at least two sample points"); return TSR_EARG; }
    if (k != 1 && k != 3) { fn_set_error("make_interp_spline: only k=1 and k=3 are supported"); return TSR_EARG; }
    int64_t lx, ly, lq;
    double *x = fn_arg_doubles(&args[0], &lx), *y = x ? fn_arg_doubles(&args[1], &ly) : NULL, *xq = y ? fn_arg_doubles(&args[3], &lq) : NULL;
    double *d = (xq && k == 3) ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *out = (xq && (k == 1 || d)) ? (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){q}) : NULL;
    int rc = (!x || !y || !xq || !out || (k == 3 && !d)) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        if (k == 3) { rc = ip_cubicspline_deriv(x, y, n, d); if (rc == TSR_OK) ip_hermite(x, y, d, n, xq, q, out); }
        else for (int64_t j = 0; j < q; j++) {              /* k == 1: piecewise linear */
            const double v = xq[j];
            int64_t lo = 0, hi = n; while (lo < hi) { const int64_t mid = (lo + hi) / 2; if (x[mid] <= v) lo = mid + 1; else hi = mid; }
            int64_t idx = lo - 1; if (idx < 0) idx = 0; else if (idx > n - 2) idx = n - 2;
            out[j] = y[idx] + (y[idx + 1] - y[idx]) * (v - x[idx]) / (x[idx + 1] - x[idx]);
        }
    }
    free(d); fn_free_doubles(x, lx); fn_free_doubles(y, ly); fn_free_doubles(xq, lq);
    return rc;
}

/* splder(t, c, k, n=1): derivative of a B-spline in FITPACK (t, c, k) form (scipy.interpolate.splder). c is the
   FITPACK-padded coefficient array (length len(t)). Each pass: c' = (c[1:]-c[:-1])*k/(t[k+1:-1]-t[1:-k-1]) padded
   with k zeros, knots t[1:-1], degree k-1. Returns (t, c, k). */
static int r_splder(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("splder: t and c must be 1-D arrays"); return TSR_EARG; }
    if (args[2].kind != 1) { fn_set_error("splder: k must be an integer degree"); return TSR_EARG; }
    const int64_t nt = args[0].arr.shape[0];
    int kk = (int)args[2].num;
    const int nn = (nargs > 3 && args[3].kind == 1) ? (int)args[3].num : 1;
    if (nn < 0 || nn > kk) { fn_set_error("splder: order of derivative must be 0 <= n <= k"); return TSR_EARG; }
    int64_t lt, lc;
    double *t0 = fn_arg_doubles(&args[0], &lt), *c0 = t0 ? fn_arg_doubles(&args[1], &lc) : NULL;
    double *tcur = t0 ? (double *)malloc((size_t)nt * sizeof(double)) : NULL;
    double *ccur = tcur ? (double *)malloc((size_t)(nt > args[1].arr.shape[0] ? nt : args[1].arr.shape[0]) * sizeof(double)) : NULL;
    int rc = (!t0 || !c0 || !tcur || !ccur) ? TSR_ENOMEM : TSR_OK;
    int64_t tn = nt;
    if (rc == TSR_OK) {
        memcpy(tcur, t0, (size_t)nt * sizeof(double));
        memcpy(ccur, c0, (size_t)args[1].arr.shape[0] * sizeof(double));
        for (int j = 0; j < nn && rc == TSR_OK; j++) {
            const int64_t ld = tn - kk - 2;                /* core length */
            double *cnew = (double *)malloc((size_t)(tn - 2) * sizeof(double));
            if (!cnew) { rc = TSR_ENOMEM; break; }
            for (int64_t i = 0; i < ld; i++) { const double dt = tcur[kk + 1 + i] - tcur[1 + i]; cnew[i] = (ccur[1 + i] - ccur[i]) * (double)kk / dt; }
            for (int64_t i = ld; i < tn - 2; i++) cnew[i] = 0.0;   /* pad with k zeros */
            for (int64_t i = 0; i < tn - 2; i++) tcur[i] = tcur[i + 1];   /* t[1:-1] */
            tn -= 2; free(ccur); ccur = cnew; kk -= 1;
        }
    }
    if (rc == TSR_OK) {
        double *rt = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){tn});
        double *rcc = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){tn});
        if (!rt || !rcc) rc = TSR_ENOMEM;
        else { memcpy(rt, tcur, (size_t)tn * sizeof(double)); memcpy(rcc, ccur, (size_t)tn * sizeof(double)); fn_result_int(&res[2], (int64_t)kk); }
    }
    free(tcur); free(ccur); fn_free_doubles(t0, lt); fn_free_doubles(c0, lc);
    return rc;
}

/* splantider(t, c, k, n=1): antiderivative of a B-spline (scipy.interpolate.splantider). Inverse of splder:
   c_new = [0, cumsum(c[:-k-1]*(t[k+1:]-t[:-k-1]))/(k+1), (last) x (k+2)], knots [t[0], t, t[-1]], degree k+1. */
static int r_splantider(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("splantider: t and c must be 1-D arrays"); return TSR_EARG; }
    if (args[2].kind != 1) { fn_set_error("splantider: k must be an integer degree"); return TSR_EARG; }
    const int64_t nt = args[0].arr.shape[0];
    int kk = (int)args[2].num;
    const int nn = (nargs > 3 && args[3].kind == 1) ? (int)args[3].num : 1;
    if (nn < 0) { fn_set_error("splantider: order must be nonnegative"); return TSR_EARG; }
    int64_t lt, lc;
    double *t0 = fn_arg_doubles(&args[0], &lt), *c0 = t0 ? fn_arg_doubles(&args[1], &lc) : NULL;
    const int64_t cap = nt + 2 * (int64_t)nn + 2;
    double *tcur = t0 ? (double *)malloc((size_t)cap * sizeof(double)) : NULL;
    double *ccur = tcur ? (double *)malloc((size_t)cap * sizeof(double)) : NULL;
    int rc = (!t0 || !c0 || !tcur || !ccur) ? TSR_ENOMEM : TSR_OK;
    int64_t tn = nt;
    if (rc == TSR_OK) {
        memcpy(tcur, t0, (size_t)nt * sizeof(double));
        memcpy(ccur, c0, (size_t)args[1].arr.shape[0] * sizeof(double));
        for (int j = 0; j < nn && rc == TSR_OK; j++) {
            const int64_t ld = tn - kk - 1;                /* core length */
            double *cnew = (double *)malloc((size_t)(tn + 2) * sizeof(double));
            double *tnew = (double *)malloc((size_t)(tn + 2) * sizeof(double));
            if (!cnew || !tnew) { free(cnew); free(tnew); rc = TSR_ENOMEM; break; }
            cnew[0] = 0.0; double acc = 0.0;
            for (int64_t i = 0; i < ld; i++) { const double dt = tcur[kk + 1 + i] - tcur[i]; acc += ccur[i] * dt; cnew[1 + i] = acc / (double)(kk + 1); }
            const double last = cnew[ld];
            for (int64_t i = 0; i < kk + 2; i++) cnew[1 + ld + i] = last;
            tnew[0] = tcur[0];
            for (int64_t i = 0; i < tn; i++) tnew[1 + i] = tcur[i];
            tnew[tn + 1] = tcur[tn - 1];
            free(tcur); free(ccur); tcur = tnew; ccur = cnew; tn += 2; kk += 1;
        }
    }
    if (rc == TSR_OK) {
        double *rt = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){tn});
        double *rcc = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){tn});
        if (!rt || !rcc) rc = TSR_ENOMEM;
        else { memcpy(rt, tcur, (size_t)tn * sizeof(double)); memcpy(rcc, ccur, (size_t)tn * sizeof(double)); fn_result_int(&res[2], (int64_t)kk); }
    }
    free(tcur); free(ccur); fn_free_doubles(t0, lt); fn_free_doubles(c0, lc);
    return rc;
}

/* splint(a, b, t, c, k): definite integral of a B-spline over [a, b] (scipy.interpolate.splint). Matches FITPACK,
   which silently takes the spline to be zero outside the base interval [t[k], t[n]] (n = len(t)-k-1) -- i.e. it is
   BSpline.integrate with extrapolate=False. Clamp the limits to the base interval; if the clamped interval is empty
   the integral is zero. Otherwise build the order-1 antiderivative (the splantider recurrence) and difference it at
   the two ends via de Boor. Scalar output. */
static int r_splint(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 1 || args[1].kind != 1) { fn_set_error("splint: a and b must be numbers"); return TSR_EARG; }
    if (args[2].kind != 3 || args[2].arr.ndim != 1 || args[3].kind != 3 || args[3].arr.ndim != 1) { fn_set_error("splint: t and c must be 1-D arrays"); return TSR_EARG; }
    if (args[4].kind != 1) { fn_set_error("splint: k must be an integer degree"); return TSR_EARG; }
    const int k = (int)args[4].num;
    const int64_t nt = args[2].arr.shape[0], n = nt - k - 1;   /* n coefficients; base interval [t[k], t[n]] */
    if (k < 0 || n < 1 || args[3].arr.shape[0] < n) { fn_set_error("splint: inconsistent (t, c, k) sizes"); return TSR_EARG; }
    double a = (args[0].flags & 1) ? (double)args[0].ival : args[0].num;
    double b = (args[1].flags & 1) ? (double)args[1].ival : args[1].num;
    double sign = 1.0;
    if (b < a) { const double tmp = a; a = b; b = tmp; sign = -1.0; }
    int64_t lt, lc;
    double *t = fn_arg_doubles(&args[2], &lt), *c = t ? fn_arg_doubles(&args[3], &lc) : NULL;
    double *ta = NULL, *ca = NULL, *d = NULL;
    int rc = (!t || !c) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        if (a < t[k]) a = t[k];
        if (b > t[n]) b = t[n];
        if (a >= b) { fn_result_num(&res[0], 0.0); }           /* clamped interval empty -> zero outside base interval */
        else {
            const int64_t nta = nt + 2; const int ka = k + 1;  /* antiderivative: knots ta, coeffs ca, degree k+1 */
            ta = (double *)malloc((size_t)nta * sizeof(double));
            ca = (double *)malloc((size_t)nta * sizeof(double));
            d = (double *)malloc((size_t)(ka + 1) * sizeof(double));
            if (!ta || !ca || !d) rc = TSR_ENOMEM;
            else {
                ta[0] = t[0];
                for (int64_t i = 0; i < nt; i++) ta[1 + i] = t[i];
                ta[nt + 1] = t[nt - 1];
                ca[0] = 0.0; double acc = 0.0;
                for (int64_t i = 0; i < n; i++) { const double dt = t[k + 1 + i] - t[i]; acc += c[i] * dt; ca[1 + i] = acc / (double)(k + 1); }
                for (int64_t i = 0; i < k + 2; i++) ca[1 + n + i] = ca[n];   /* pad with k+2 copies of the last */
                const int64_t na = nta - ka - 1;               /* antiderivative coefficient count = n + 1 */
                const double xs[2] = {a, b}; double F[2];
                for (int e = 0; e < 2; e++) {
                    const double x = xs[e];
                    int64_t mu = ka; while (mu < na - 1 && ta[mu + 1] <= x) mu++;
                    for (int j = 0; j <= ka; j++) d[j] = ca[mu - ka + j];
                    for (int r = 1; r <= ka; r++)
                        for (int j = ka; j >= r; j--) {
                            const double denom = ta[j + mu - r + 1] - ta[j + mu - ka];
                            const double aa = denom != 0.0 ? (x - ta[j + mu - ka]) / denom : 0.0;
                            d[j] = (1.0 - aa) * d[j - 1] + aa * d[j];
                        }
                    F[e] = d[ka];
                }
                fn_result_num(&res[0], sign * (F[1] - F[0]));
            }
        }
    }
    free(ta); free(ca); free(d); fn_free_doubles(t, lt); fn_free_doubles(c, lc);
    return rc;
}

/* de Boor evaluation of a B-spline (knots t, nc coefficients, degree k) at a scalar x; x is clamped to the active
   interval exactly as ip_bspline_eval. k <= 3 here (sproot is cubic). */
static double ip_deboor_scalar(const double *t, const double *c, int64_t nc, int k, double x)
{
    double d[8];
    int64_t mu = k; while (mu < nc - 1 && t[mu + 1] <= x) mu++;
    for (int j = 0; j <= k; j++) d[j] = c[mu - k + j];
    for (int r = 1; r <= k; r++)
        for (int j = k; j >= r; j--) {
            const double denom = t[j + mu - r + 1] - t[j + mu - k];
            const double a = denom != 0.0 ? (x - t[j + mu - k]) / denom : 0.0;
            d[j] = (1.0 - a) * d[j - 1] + a * d[j];
        }
    return d[k];
}

static double ip_cub_eval(const double *a, double u) { return ((a[3] * u + a[2]) * u + a[1]) * u + a[0]; }
static double ip_cub_der(const double *a, double u) { return (3.0 * a[3] * u + 2.0 * a[2]) * u + a[1]; }

/* find a root of the cubic a[] (power basis a0..a3) in [lo, hi] when it brackets a sign change: bisect to a tight
   bracket, then a few Newton steps. Returns 1 and sets *out, else 0. The right endpoint is left to the next
   interval (so a root at a shared knot is reported once). */
static int ip_bracket_root(const double *a, double lo, double hi, double *out)
{
    const double flo = ip_cub_eval(a, lo), fhi = ip_cub_eval(a, hi);
    if (flo == 0.0) { *out = lo; return 1; }
    if (fhi == 0.0) return 0;
    if (flo * fhi > 0.0) return 0;
    double aa = lo, bb = hi, fa = flo, m = lo;
    for (int it = 0; it < 80; it++) {
        m = 0.5 * (aa + bb); const double fm = ip_cub_eval(a, m);
        if (fa * fm <= 0.0) bb = m; else { aa = m; fa = fm; }
        if (bb - aa < 1e-15 * (fabs(m) + 1.0)) break;
    }
    double x = 0.5 * (aa + bb);
    for (int it = 0; it < 4; it++) {
        const double f = ip_cub_eval(a, x), df = ip_cub_der(a, x);
        if (df != 0.0) { const double nx = x - f / df; if (nx >= lo && nx <= hi) x = nx; }
    }
    *out = x; return 1;
}

static int ip_cmp_double(const void *A, const void *B)
{ const double x = *(const double *)A, y = *(const double *)B; return x < y ? -1 : (x > y ? 1 : 0); }

/* sproot(t, c, k, mest=10): roots of a cubic (k=3) B-spline (scipy.interpolate.sproot). On each knot interval
   [t[i], t[i+1]] the spline is one cubic; recover its power-basis coefficients by evaluating at 4 interior points
   (exact, since it is cubic there) and solving a 4x4 Vandermonde via the shared dense solver, then isolate roots
   by the cubic's critical points and bracket each sign change. Roots are returned sorted ascending, at most mest
   of them (FITPACK's cap). Matches scipy to ~1e-14 for simple transversal roots not coincident with a knot. */
static int r_sproot(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("sproot: t and c must be 1-D arrays"); return TSR_EARG; }
    if (args[2].kind != 1) { fn_set_error("sproot: k must be an integer degree"); return TSR_EARG; }
    const int k = (int)args[2].num;
    if (k != 3) { fn_set_error("sproot works only for cubic (k=3) splines"); return TSR_EARG; }
    const int64_t nt = args[0].arr.shape[0];
    if (nt < 8) { fn_set_error("sproot: the number of knots must be >= 8"); return TSR_EARG; }
    const int64_t nc = nt - k - 1;
    if (args[1].arr.shape[0] < nc) { fn_set_error("sproot: inconsistent (t, c, k) sizes"); return TSR_EARG; }
    const int64_t mest = (nargs > 3 && args[3].kind == 1) ? (args[3].flags & 1 ? args[3].ival : (int64_t)args[3].num) : 10;
    int64_t lt, lc;
    double *t = fn_arg_doubles(&args[0], &lt), *c = t ? fn_arg_doubles(&args[1], &lc) : NULL;
    const int64_t n4 = nt - 4, cap = 3 * (n4 - 3) + 8;      /* at most 3 roots per interval */
    double *zeros = (t && c) ? (double *)malloc((size_t)(cap > 0 ? cap : 1) * sizeof(double)) : NULL;
    int rc = (!t || !c || !zeros) ? TSR_ENOMEM : TSR_OK;
    int64_t m = 0;
    for (int64_t i = 3; i < n4 && rc == TSR_OK; i++) {
        const double lo_x = t[i], hi_x = t[i + 1];
        if (hi_x <= lo_x) continue;
        const double xm = 0.5 * (lo_x + hi_x), h = hi_x - lo_x;
        double M[16], b[4];
        for (int r = 0; r < 4; r++) {
            const double u = (lo_x + h * (r + 0.5) / 4.0) - xm, y = ip_deboor_scalar(t, c, nc, k, xm + u);
            double p = 1.0;
            for (int col = 0; col < 4; col++) { M[r * 4 + col] = p; p *= u; }
            b[r] = y;
        }
        rc = sl_dense_solve(M, b, 4, 1);
        if (rc != TSR_OK) { fn_set_error("sproot: failed to form the local cubic"); break; }
        const double a[4] = {b[0], b[1], b[2], b[3]}, lo = lo_x - xm, hi = hi_x - xm;
        double pts[4]; int npts = 0; pts[npts++] = lo;        /* break [lo, hi] at the cubic's critical points */
        const double A = 3.0 * a[3], B = 2.0 * a[2], C = a[1];
        if (fabs(A) < 1e-14 * (fabs(B) + fabs(C) + 1.0)) {
            if (fabs(B) > 0.0) { const double r0 = -C / B; if (r0 > lo && r0 < hi) pts[npts++] = r0; }
        } else {
            const double disc = B * B - 4.0 * A * C;
            if (disc > 0.0) {
                const double s = sqrt(disc), r1 = (-B + s) / (2.0 * A), r2 = (-B - s) / (2.0 * A);
                if (r1 > lo && r1 < hi) pts[npts++] = r1;
                if (r2 > lo && r2 < hi) pts[npts++] = r2;
            }
        }
        pts[npts++] = hi;
        qsort(pts, (size_t)npts, sizeof(double), ip_cmp_double);
        for (int s = 0; s + 1 < npts; s++) {
            double u;
            if (ip_bracket_root(a, pts[s], pts[s + 1], &u)) {
                const double x = xm + u;
                if (x >= lo_x - 1e-12 && x < hi_x - 1e-12 && m < cap) zeros[m++] = x;
            }
        }
    }
    if (rc == TSR_OK) {
        qsort(zeros, (size_t)m, sizeof(double), ip_cmp_double);
        int64_t w = 0;                                        /* dedup a root reported from both sides of a knot */
        for (int64_t j = 0; j < m; j++) if (w == 0 || zeros[j] - zeros[w - 1] > 1e-9) zeros[w++] = zeros[j];
        m = (w > mest) ? mest : w;
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){m});
        if (!out) rc = TSR_ENOMEM; else for (int64_t j = 0; j < m; j++) out[j] = zeros[j];
    }
    free(zeros); fn_free_doubles(t, lt); fn_free_doubles(c, lc);
    return rc;
}

/* lagrange(x, w): Lagrange interpolating polynomial through (x, w) returned as numpy.poly1d coefficients
   (highest-degree first), via scipy's Newton-basis summation p = Σ_j w[j]·Π_{k≠j}(X-x[k])/(x[j]-x[k]). Each basis
   product is accumulated by polynomial convolution. Numerically unstable for many points (as scipy warns); exact
   for generic degree-(M-1) data. */
static int r_lagrange(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("lagrange: x and w must be 1-D arrays"); return TSR_EARG; }
    const int64_t M = args[0].arr.shape[0];
    if (args[1].arr.shape[0] != M) { fn_set_error("lagrange: x and w must have the same length"); return TSR_EARG; }
    if (M < 1) { fn_set_error("lagrange: need at least one point"); return TSR_EARG; }
    int64_t lx, lw;
    double *x = fn_arg_doubles(&args[0], &lx), *w = x ? fn_arg_doubles(&args[1], &lw) : NULL;
    double *p = x ? (double *)calloc((size_t)M, sizeof(double)) : NULL;   /* lowest-first accumulator, degree <= M-1 */
    double *pt = p ? (double *)malloc((size_t)M * sizeof(double)) : NULL; /* lowest-first running basis product */
    int rc = (!x || !w || !p || !pt) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        for (int64_t j = 0; j < M; j++) {
            pt[0] = w[j]; int64_t deg = 0;                               /* pt = [w[j]] */
            for (int64_t k = 0; k < M; k++) {
                if (k == j) continue;
                const double fac = x[j] - x[k], c0 = -x[k] / fac, c1 = 1.0 / fac;   /* factor (X - x[k])/fac */
                for (int64_t i = deg + 1; i >= 0; i--) {                 /* pt = convolve(pt, [c0, c1]) in place */
                    double v = (i <= deg) ? pt[i] * c0 : 0.0;
                    if (i >= 1) v += pt[i - 1] * c1;
                    pt[i] = v;
                }
                deg++;
            }
            for (int64_t i = 0; i <= deg; i++) p[i] += pt[i];            /* p += pt */
        }
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){M});
        if (!out) rc = TSR_ENOMEM;
        else for (int64_t i = 0; i < M; i++) out[i] = p[M - 1 - i];      /* reverse to highest-first (poly1d order) */
    }
    free(p); free(pt); fn_free_doubles(x, lx); fn_free_doubles(w, lw);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("interpolate.pchip_interpolate", 1, "xi, yi, x", "y", r_pchip_interpolate, NULL, "Monotone piecewise-cubic (PCHIP) interpolation evaluated at x (scipy.interpolate.pchip_interpolate)."),
    ROUTINE("interpolate.PchipInterpolator", 1, "x, y, xnew", "y", r_pchip_class, NULL, "PCHIP monotone cubic interpolation evaluated at xnew (scipy.interpolate.PchipInterpolator)."),
    ROUTINE("interpolate.Akima1DInterpolator", 1, "x, y, xnew", "y", r_akima, NULL, "Akima piecewise-cubic interpolation evaluated at xnew (scipy.interpolate.Akima1DInterpolator)."),
    ROUTINE("interpolate.CubicSpline", 1, "x, y, xnew", "y", r_cubicspline, NULL, "Not-a-knot cubic spline evaluated at xnew (scipy.interpolate.CubicSpline)."),
    ROUTINE("interpolate.CubicHermiteSpline", 1, "x, y, dydx, xnew", "y", r_cubic_hermite, NULL, "Cubic Hermite spline with given derivatives evaluated at xnew (scipy.interpolate.CubicHermiteSpline)."),
    ROUTINE("interpolate.barycentric_interpolate", 1, "xi, yi, x", "y", r_barycentric, NULL, "Barycentric Lagrange polynomial interpolation evaluated at x (scipy.interpolate.barycentric_interpolate)."),
    ROUTINE("interpolate.krogh_interpolate", 1, "xi, yi, x", "y", r_krogh, NULL, "Polynomial interpolation (Newton divided differences) evaluated at x (scipy.interpolate.krogh_interpolate)."),
    ROUTINE("interpolate.BSpline", 1, "t, c, k, x", "y", r_bspline, NULL, "Evaluate a B-spline (knots t, coefficients c, degree k) at x via de Boor's algorithm (scipy.interpolate.BSpline)."),
    ROUTINE("interpolate.pade", 2, "an, m, n=None", "p, q", r_pade, NULL, "Pade rational approximant (numerator p, denominator q) from Taylor coefficients (scipy.interpolate.pade)."),
    ROUTINE("interpolate.make_interp_spline", 1, "x, y, k, xnew", "y", r_make_interp_spline, NULL, "Interpolating spline (k=1 linear or k=3 not-a-knot cubic) evaluated at xnew (scipy.interpolate.make_interp_spline)."),
    ROUTINE("interpolate.splev", 1, "t, c, k, x", "y", r_splev, NULL, "Evaluate a B-spline (knots t, coefficients c, degree k) at x via de Boor's algorithm (scipy.interpolate.splev)."),
    ROUTINE("interpolate.splder", 3, "t, c, k, n=1", "t, c, k", r_splder, NULL, "Derivative of a B-spline in (t, c, k) form (scipy.interpolate.splder)."),
    ROUTINE("interpolate.splantider", 3, "t, c, k, n=1", "t, c, k", r_splantider, NULL, "Antiderivative of a B-spline in (t, c, k) form (scipy.interpolate.splantider)."),
    ROUTINE("interpolate.splint", 1, "a, b, t, c, k", "integral", r_splint, NULL, "Definite integral of a B-spline over [a, b], taking the spline as zero outside its base interval (scipy.interpolate.splint)."),
    ROUTINE("interpolate.sproot", 1, "t, c, k, mest=10", "zeros", r_sproot, NULL, "Roots of a cubic (k=3) B-spline, sorted ascending, at most mest of them (scipy.interpolate.sproot)."),
    ROUTINE("interpolate.lagrange", 1, "x, w", "c", r_lagrange, NULL, "Lagrange interpolating polynomial through (x, w) as poly1d coefficients, highest-degree first (scipy.interpolate.lagrange)."),
};

const fn_table TSR_SCIPY_INTERPOLATE_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
