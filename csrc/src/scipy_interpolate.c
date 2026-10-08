/* scipy.interpolate ("interpolate." prefix -> Tessero\Interpolate facade).
 *
 *   interpolate.pchip_interpolate(xi, yi, x)   monotone piecewise-cubic (PCHIP) interpolation at x
 *
 * pchip_interpolate(xi, yi, x) == PchipInterpolator(xi, yi)(x). The shape-preserving derivatives follow SciPy's
 * Fritsch-Carlson construction (_find_derivatives / _edge_case) exactly; the piecewise cubic is then evaluated
 * in the Hermite basis (algebraically identical to SciPy's Bernstein form, up to basis-level rounding).
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static double ip_sign(double v) { return v > 0.0 ? 1.0 : (v < 0.0 ? -1.0 : 0.0); }

/* SciPy PchipInterpolator._edge_case: one-sided three-point derivative estimate, shape-limited. */
static double pchip_edge(double h0, double h1, double m0, double m1)
{
    double d = ((2.0 * h0 + h1) * m0 - h0 * m1) / (h0 + h1);
    if (ip_sign(d) != ip_sign(m0)) d = 0.0;
    else if (ip_sign(m0) != ip_sign(m1) && fabs(d) > 3.0 * fabs(m0)) d = 3.0 * m0;
    return d;
}

/* SciPy PchipInterpolator._find_derivatives: per-knot slopes d[0..n-1] (n>=2). */
static int pchip_derivatives(const double *x, const double *y, int64_t n, double *d)
{
    double *hk = (double *)malloc((size_t)(n - 1) * sizeof(double));
    double *mk = (double *)malloc((size_t)(n - 1) * sizeof(double));
    if (!hk || !mk) { free(hk); free(mk); return TSR_ENOMEM; }
    for (int64_t k = 0; k < n - 1; k++) { hk[k] = x[k + 1] - x[k]; mk[k] = (y[k + 1] - y[k]) / hk[k]; }
    if (n == 2) { d[0] = mk[0]; d[1] = mk[0]; free(hk); free(mk); return TSR_OK; }
    for (int64_t i = 0; i < n - 2; i++) {                 /* interior knots k = i+1 */
        const int cond = (ip_sign(mk[i + 1]) != ip_sign(mk[i])) || mk[i + 1] == 0.0 || mk[i] == 0.0;
        if (cond) d[i + 1] = 0.0;
        else {
            const double w1 = 2.0 * hk[i + 1] + hk[i], w2 = hk[i + 1] + 2.0 * hk[i];
            const double whmean = (w1 / mk[i] + w2 / mk[i + 1]) / (w1 + w2);
            d[i + 1] = 1.0 / whmean;
        }
    }
    d[0] = pchip_edge(hk[0], hk[1], mk[0], mk[1]);
    d[n - 1] = pchip_edge(hk[n - 2], hk[n - 3], mk[n - 2], mk[n - 3]);
    free(hk); free(mk);
    return TSR_OK;
}

/* pchip_interpolate(xi, yi, x): monotone cubic interpolation evaluated at x (scipy.interpolate.pchip_interpolate,
   der=0). xi strictly increasing. */
static int r_pchip_interpolate(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1 ||
        args[2].kind != 3 || args[2].arr.ndim != 1) { fn_set_error("pchip_interpolate: xi, yi and x must be 1-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], q = args[2].arr.shape[0];
    if (args[1].arr.shape[0] != n) { fn_set_error("pchip_interpolate: xi and yi must have the same length"); return TSR_EARG; }
    if (n < 2) { fn_set_error("pchip_interpolate: need at least two sample points"); return TSR_EARG; }
    int64_t lx, ly, lq;
    double *x = fn_arg_doubles(&args[0], &lx);
    double *y = x ? fn_arg_doubles(&args[1], &ly) : NULL;
    double *xq = y ? fn_arg_doubles(&args[2], &lq) : NULL;
    double *d = xq ? (double *)malloc((size_t)n * sizeof(double)) : NULL;
    double *out = d ? (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){q}) : NULL;
    int rc = (!x || !y || !xq || !d || !out) ? TSR_ENOMEM : pchip_derivatives(x, y, n, d);
    if (rc == TSR_OK) {
        for (int64_t j = 0; j < q; j++) {
            const double v = xq[j];
            int64_t lo = 0, hi = n;                        /* searchsorted(x, v, side='right') */
            while (lo < hi) { const int64_t mid = (lo + hi) / 2; if (x[mid] <= v) lo = mid + 1; else hi = mid; }
            int64_t idx = lo - 1;
            if (idx < 0) idx = 0; else if (idx > n - 2) idx = n - 2;
            const double h = x[idx + 1] - x[idx], s = (v - x[idx]) / h, s2 = s * s, s3 = s2 * s;
            const double h00 = 2.0 * s3 - 3.0 * s2 + 1.0, h01 = -2.0 * s3 + 3.0 * s2;
            const double h10 = s3 - 2.0 * s2 + s, h11 = s3 - s2;
            out[j] = y[idx] * h00 + y[idx + 1] * h01 + d[idx] * h * h10 + d[idx + 1] * h * h11;
        }
    }
    free(d);
    fn_free_doubles(x, lx); fn_free_doubles(y, ly); fn_free_doubles(xq, lq);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("interpolate.pchip_interpolate", 1, "xi, yi, x", "y", r_pchip_interpolate, NULL, "Monotone piecewise-cubic (PCHIP) interpolation evaluated at x (scipy.interpolate.pchip_interpolate)."),
};

const fn_table TSR_SCIPY_INTERPOLATE_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
