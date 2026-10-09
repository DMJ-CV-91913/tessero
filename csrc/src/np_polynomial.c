/* numpy.polynomial ("npoly." prefix -> Tessero\Polynomial facade). Evaluation and calculus in the six standard
 * polynomial bases (power, Chebyshev T, Legendre, Laguerre, physicists' Hermite, probabilists' HermiteE), sharing
 * a single Clenshaw evaluator that replicates numpy's per-basis *val recurrences exactly.
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <lapacke.h>

enum { NB_POLY = 0, NB_CHEB, NB_LEG, NB_LAG, NB_HERM, NB_HERME };

/* evaluate the coefficient series c[0..n-1] in the given basis at scalar x (numpy chebval/legval/... Clenshaw). */
static double npoly_eval(int basis, const double *c, int64_t n, double x)
{
    if (n <= 0) return 0.0;
    if (basis == NB_POLY) { double b = 0.0; for (int64_t k = n - 1; k >= 0; k--) b = c[k] + x * b; return b; }
    double c0, c1;
    if (n == 1) { c0 = c[0]; c1 = 0.0; }
    else if (n == 2) { c0 = c[0]; c1 = c[1]; }
    else {
        double nd = (double)n; c0 = c[n - 2]; c1 = c[n - 1];
        for (int64_t i = 3; i <= n; i++) {
            const double tmp = c0; nd -= 1.0;
            switch (basis) {
            case NB_CHEB:  c0 = c[n - i] - c1;                       c1 = tmp + c1 * (2.0 * x); break;
            case NB_LEG:   c0 = c[n - i] - c1 * ((nd - 1.0) / nd);    c1 = tmp + c1 * x * ((2.0 * nd - 1.0) / nd); break;
            case NB_LAG:   c0 = c[n - i] - (c1 * (nd - 1.0)) / nd;    c1 = tmp + (c1 * ((2.0 * nd - 1.0) - x)) / nd; break;
            case NB_HERM:  c0 = c[n - i] - c1 * (2.0 * (nd - 1.0));   c1 = tmp + c1 * (2.0 * x); break;
            case NB_HERME: c0 = c[n - i] - c1 * (nd - 1.0);          c1 = tmp + c1 * x; break;
            }
        }
    }
    switch (basis) {
    case NB_LAG:  return c0 + c1 * (1.0 - x);
    case NB_HERM: return c0 + c1 * (2.0 * x);
    default:      return c0 + c1 * x;   /* cheb, leg, herme */
    }
}

/* evaluate coefficients `c` (argument cidx) at the points `x` (argument xidx) in `basis`, writing a same-shaped
   result. Shared by the *val functions (x, c) and the basis classes (c, x). */
static int npoly_eval_routine(const tsr_arg *args, tsr_result *res, int basis, int xidx, int cidx)
{
    if (args[xidx].kind != 3 || args[cidx].kind != 3 || args[cidx].arr.ndim != 1) { fn_set_error("npoly: coefficients must be a 1-D array and x an array"); return TSR_EARG; }
    const int64_t nc = args[cidx].arr.shape[0];
    int64_t lx, lc; double *x = fn_arg_doubles(&args[xidx], &lx), *c = x ? fn_arg_doubles(&args[cidx], &lc) : NULL;
    const tsr_array *xa = &args[xidx].arr;
    double *out = c ? (double *)fn_result_array(&res[0], TSR_F64, xa->ndim, xa->shape) : NULL;
    int rc = (!x || !c || !out) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        int64_t nx = 1; for (int d = 0; d < xa->ndim; d++) nx *= xa->shape[d];
        for (int64_t i = 0; i < nx; i++) out[i] = npoly_eval(basis, c, nc, x[i]);
    }
    fn_free_doubles(x, lx); fn_free_doubles(c, lc);
    return rc;
}

/* *val(x, c): evaluate a series in the basis at x (numpy.polynomial.<basis>.<b>val). */
#define NPOLY_VAL(fn, basis) \
    static int fn(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr) \
    { (void)ctx; (void)n; (void)nr; return npoly_eval_routine(a, r, basis, 0, 1); }
NPOLY_VAL(r_polyval, NB_POLY)
NPOLY_VAL(r_chebval, NB_CHEB)
NPOLY_VAL(r_legval, NB_LEG)
NPOLY_VAL(r_lagval, NB_LAG)
NPOLY_VAL(r_hermval, NB_HERM)
NPOLY_VAL(r_hermeval, NB_HERME)

/* <Basis>(coef, x): the evaluated form of each numpy.polynomial basis class. */
#define NPOLY_CLASS(fn, basis) \
    static int fn(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr) \
    { (void)ctx; (void)n; (void)nr; return npoly_eval_routine(a, r, basis, 1, 0); }
NPOLY_CLASS(r_cls_polynomial, NB_POLY)
NPOLY_CLASS(r_cls_chebyshev, NB_CHEB)
NPOLY_CLASS(r_cls_legendre, NB_LEG)
NPOLY_CLASS(r_cls_laguerre, NB_LAG)
NPOLY_CLASS(r_cls_hermite, NB_HERM)
NPOLY_CLASS(r_cls_hermitee, NB_HERME)

/* ---- numpy.polynomial.polynomial: power-basis arithmetic and calculus ---------------------------------- */

/* optional scalar argument idx (NumPy integer/float), else default. */
static int64_t npoly_opt_int(const tsr_arg *a, int n, int idx, int64_t dflt)
{ if (idx >= n || a[idx].kind != 1) return dflt; return (a[idx].flags & 1) ? a[idx].ival : (int64_t)a[idx].num; }
static double npoly_opt_num(const tsr_arg *a, int n, int idx, double dflt)
{ if (idx >= n || a[idx].kind != 1) return dflt; return a[idx].num; }

/* allocate a 1-D float64 result of length m, return its buffer (NULL on failure). */
static double *npoly_out1(tsr_result *r, int64_t m)
{ int64_t sh[1] = {m}; return (double *)fn_result_array(r, TSR_F64, 1, sh); }

/* trailing coefficients with |c|<=tol dropped, keeping length >= 1 (numpy trimseq/trimcoef). */
static int64_t npoly_trimlen(const double *c, int64_t n, double tol)
{ while (n > 1 && fabs(c[n - 1]) <= tol) n--; return n; }

/* c1 (+/-) c2, padded to the longer length (numpy polyadd / polysub). */
static int npoly_addsub(const tsr_arg *a, int n, tsr_result *r, int sign)
{
    (void)n;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly: coefficients must be arrays"); return TSR_EARG; }
    int64_t n1, n2; double *c1 = fn_arg_doubles(&a[0], &n1), *c2 = c1 ? fn_arg_doubles(&a[1], &n2) : NULL;
    int64_t m = n1 > n2 ? n1 : n2; double *out = c2 ? npoly_out1(r, m) : NULL;
    int rc = (!c1 || !c2 || !out) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) for (int64_t i = 0; i < m; i++) out[i] = (i < n1 ? c1[i] : 0.0) + sign * (i < n2 ? c2[i] : 0.0);
    fn_free_doubles(c1, n1); fn_free_doubles(c2, n2);
    return rc;
}
static int r_polyadd(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)nr; return npoly_addsub(a, n, r, +1); }
static int r_polysub(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)nr; return npoly_addsub(a, n, r, -1); }

/* convolution of two coefficient series (numpy polymul = np.convolve). */
static int r_polymul(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly: coefficients must be arrays"); return TSR_EARG; }
    int64_t n1, n2; double *c1 = fn_arg_doubles(&a[0], &n1), *c2 = c1 ? fn_arg_doubles(&a[1], &n2) : NULL;
    int64_t m = (n1 && n2) ? n1 + n2 - 1 : 0; double *out = c2 ? npoly_out1(r, m > 0 ? m : 1) : NULL;
    int rc = (!c1 || !c2 || !out) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) { if (m <= 0) out[0] = 0.0; else { for (int64_t i = 0; i < m; i++) out[i] = 0.0;
        for (int64_t i = 0; i < n1; i++) for (int64_t j = 0; j < n2; j++) out[i + j] += c1[i] * c2[j]; } }
    fn_free_doubles(c1, n1); fn_free_doubles(c2, n2);
    return rc;
}

/* multiply a power series by x (numpy polymulx): [0, c0, c1, ...]. */
static int r_polymulx(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (nc == 1 && c[0] == 0.0) { double *out = npoly_out1(r, 1); if (!out) rc = TSR_ENOMEM; else out[0] = 0.0; }
        else { double *out = npoly_out1(r, nc + 1); if (!out) rc = TSR_ENOMEM; else { out[0] = 0.0; for (int64_t i = 0; i < nc; i++) out[i + 1] = c[i]; } }
    }
    fn_free_doubles(c, nc);
    return rc;
}

/* integer power of a power series by repeated convolution (numpy polypow, maxpower default 16). */
static int r_polypow(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t pw = npoly_opt_int(a, n, 1, 0), maxp = npoly_opt_int(a, n, 2, 16);
    if (pw < 0) { fn_set_error("npoly.polypow: Power must be non-negative"); return TSR_EARG; }
    if (pw > maxp) { fn_set_error("npoly.polypow: Power is too large"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (pw == 0) { double *o = npoly_out1(r, 1); if (!o) rc = TSR_ENOMEM; else o[0] = 1.0; }
        else {
            int64_t m = (nc - 1) * pw + 1; double *acc = calloc(m, sizeof(double)), *tmp = calloc(m, sizeof(double));
            if (!acc || !tmp) rc = TSR_ENOMEM;
            else {
                int64_t la = nc;
                for (int64_t i = 0; i < nc; i++) acc[i] = c[i];
                for (int64_t p = 1; p < pw; p++) {
                    int64_t lt = la + nc - 1;
                    for (int64_t i = 0; i < lt; i++) tmp[i] = 0.0;
                    for (int64_t i = 0; i < la; i++) for (int64_t j = 0; j < nc; j++) tmp[i + j] += acc[i] * c[j];
                    for (int64_t i = 0; i < lt; i++) acc[i] = tmp[i];
                    la = lt;
                }
                double *o = npoly_out1(r, m); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < m; i++) o[i] = acc[i];
            }
            free(acc); free(tmp);
        }
    }
    fn_free_doubles(c, nc);
    return rc;
}

/* quotient and remainder of power-series division (numpy polydiv -> (quo, rem)). */
static int r_polydiv(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly: coefficients must be arrays"); return TSR_EARG; }
    int64_t n1, n2; double *c1 = fn_arg_doubles(&a[0], &n1), *c2 = c1 ? fn_arg_doubles(&a[1], &n2) : NULL;
    int rc = c2 ? TSR_OK : TSR_ENOMEM;
    double *w = NULL;
    if (rc == TSR_OK && c2[n2 - 1] == 0.0) { fn_set_error("npoly.polydiv: division by zero"); rc = TSR_EARG; }
    if (rc == TSR_OK) {
        if (n1 < n2) {                                   /* quo = [0], rem = c1 */
            double *q = npoly_out1(&r[0], 1), *rem = q ? npoly_out1(&r[1], n1) : NULL;
            if (!q || !rem) rc = TSR_ENOMEM; else { q[0] = 0.0; for (int64_t i = 0; i < n1; i++) rem[i] = c1[i]; }
        } else if (n2 == 1) {                            /* exact: quo = c1/scl, rem = [0] */
            double *q = npoly_out1(&r[0], n1), *rem = q ? npoly_out1(&r[1], 1) : NULL;
            if (!q || !rem) rc = TSR_ENOMEM; else { for (int64_t i = 0; i < n1; i++) q[i] = c1[i] / c2[0]; rem[0] = 0.0; }
        } else {
            w = malloc(n1 * sizeof(double)); if (!w) rc = TSR_ENOMEM;
            if (rc == TSR_OK) {
                for (int64_t i = 0; i < n1; i++) w[i] = c1[i];
                double scl = c2[n2 - 1]; int64_t dlen = n1 - n2, i = dlen, j = n1 - 1;
                while (i >= 0) { for (int64_t k = 0; k < n2 - 1; k++) w[i + k] -= (c2[k] / scl) * w[j]; i--; j--; }
                int64_t ql = n1 - (n2 - 1), rl = npoly_trimlen(w, n2 - 1, 0.0);
                double *q = npoly_out1(&r[0], ql), *rem = q ? npoly_out1(&r[1], rl) : NULL;
                if (!q || !rem) rc = TSR_ENOMEM;
                else { for (int64_t k = 0; k < ql; k++) q[k] = w[(n2 - 1) + k] / scl; for (int64_t k = 0; k < rl; k++) rem[k] = w[k]; }
            }
        }
    }
    free(w); fn_free_doubles(c1, n1); fn_free_doubles(c2, n2);
    return rc;
}

/* m-th derivative of a power series (numpy polyder, scl default 1). */
static int r_polyder(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t m = npoly_opt_int(a, n, 1, 1); double scl = npoly_opt_num(a, n, 2, 1.0);
    if (m < 0) { fn_set_error("npoly.polyder: The order of derivation must be non-negative"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        int64_t len = nc;
        for (int64_t it = 0; it < m && len > 0; it++) {
            if (len <= 1) { len = 1; c[0] = 0.0; break; }
            for (int64_t j = 1; j < len; j++) c[j - 1] = c[j] * (double)j * scl;
            len -= 1;
        }
        if (m >= nc) { len = 1; c[0] = 0.0; }
        double *o = npoly_out1(r, len); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < len; i++) o[i] = c[i];
    }
    fn_free_doubles(c, nc);
    return rc;
}

/* m-th antiderivative of a power series (numpy polyint; k, lbnd, scl default 0, 0, 1). */
static int r_polyint(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t m = npoly_opt_int(a, n, 1, 1); double kc = npoly_opt_num(a, n, 2, 0.0), lbnd = npoly_opt_num(a, n, 3, 0.0), scl = npoly_opt_num(a, n, 4, 1.0);
    if (m < 0) { fn_set_error("npoly.polyint: The order of integration must be non-negative"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    double *buf = NULL;
    if (rc == TSR_OK) {
        int64_t len = nc; buf = malloc((nc + m + 1) * sizeof(double)); if (!buf) rc = TSR_ENOMEM;
        if (rc == TSR_OK) {
            for (int64_t i = 0; i < len; i++) buf[i] = c[i];
            for (int64_t it = 0; it < m; it++) {
                for (int64_t i = len; i >= 1; i--) buf[i] = buf[i - 1] / (double)i * scl;
                buf[0] = 0.0; len += 1;
                buf[0] += kc - npoly_eval(NB_POLY, buf, len, lbnd);
            }
            double *o = npoly_out1(r, len); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < len; i++) o[i] = buf[i];
        }
    }
    free(buf); fn_free_doubles(c, nc);
    return rc;
}

/* coefficients of the monic polynomial with the given roots (numpy polyfromroots). */
static int r_polyfromroots(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: roots must be an array"); return TSR_EARG; }
    int64_t nr_; double *rt = fn_arg_doubles(&a[0], &nr_);
    int rc = rt ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        double *o = npoly_out1(r, nr_ + 1); if (!o) rc = TSR_ENOMEM;
        else {
            o[0] = 1.0; int64_t len = 1;
            for (int64_t k = 0; k < nr_; k++) {
                double root = rt[k]; o[len] = o[len - 1];
                for (int64_t i = len - 1; i >= 1; i--) o[i] = o[i - 1] - root * o[i];
                o[0] = -root * o[0]; len += 1;
            }
        }
    }
    fn_free_doubles(rt, nr_);
    return rc;
}

/* a line off + scl*x (numpy polyline). */
static int r_polyline(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)nr;
    double off = npoly_opt_num(a, n, 0, 0.0), scl = npoly_opt_num(a, n, 1, 0.0);
    if (scl != 0.0) { double *o = npoly_out1(r, 2); if (!o) return TSR_ENOMEM; o[0] = off; o[1] = scl; }
    else { double *o = npoly_out1(r, 1); if (!o) return TSR_ENOMEM; o[0] = off; }
    return TSR_OK;
}

/* trailing-coefficient trim (numpy polytrim = trimcoef, tol default 0). */
static int r_polytrim(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    double tol = npoly_opt_num(a, n, 1, 0.0);
    if (tol < 0.0) { fn_set_error("npoly.polytrim: tol must be non-negative"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { int64_t len = npoly_trimlen(c, nc, tol); double *o = npoly_out1(r, len);
        if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < len; i++) o[i] = c[i]; }
    fn_free_doubles(c, nc);
    return rc;
}

/* pseudo-Vandermonde matrix of the power basis, shape (len(x), deg+1) (numpy polyvander). */
static int r_polyvander(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: x must be an array"); return TSR_EARG; }
    int64_t deg = npoly_opt_int(a, n, 1, 0);
    if (deg < 0) { fn_set_error("npoly.polyvander: deg must be non-negative"); return TSR_EARG; }
    int64_t lx; double *xs = fn_arg_doubles(&a[0], &lx);
    int rc = xs ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { int64_t sh[2] = {lx, deg + 1}; double *V = (double *)fn_result_array(r, TSR_F64, 2, sh);
        if (!V) rc = TSR_ENOMEM; else for (int64_t i = 0; i < lx; i++) { double *row = V + i * (deg + 1); row[0] = 1.0;
            for (int64_t j = 1; j <= deg; j++) row[j] = row[j - 1] * xs[i]; } }
    fn_free_doubles(xs, lx);
    return rc;
}

/* evaluate prod_k (x - r[k]) at each x (numpy polyvalfromroots, tensor=True for 1-D r). */
static int r_polyvalfromroots(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly: x and r must be arrays"); return TSR_EARG; }
    int64_t lx, lr; double *xs = fn_arg_doubles(&a[0], &lx), *rt = xs ? fn_arg_doubles(&a[1], &lr) : NULL;
    const tsr_array *xa = &a[0].arr; double *out = rt ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) { double p = 1.0; for (int64_t k = 0; k < lr; k++) p *= (xs[i] - rt[k]); out[i] = p; }
    fn_free_doubles(xs, lx); fn_free_doubles(rt, lr);
    return rc;
}

/* ---- multi-dimensional evaluation, the Vandermonde matrices, roots, companion and fit --------------------- */

/* evaluate c[a,:] (power basis) at y for each outer index a, combining by Horner in x (numpy polyval2d). */
static int r_polyval2d(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[2].arr.ndim != 2) { fn_set_error("npoly.polyval2d: x, y arrays and c a 2-D array"); return TSR_EARG; }
    int64_t lx, ly, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *c = ys ? fn_arg_doubles(&a[2], &lc) : NULL;
    const int64_t nx = a[2].arr.shape[0], ny = a[2].arr.shape[1];
    const tsr_array *xa = &a[0].arr; double *out = c ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) {
        double acc = 0.0; for (int64_t aa = nx - 1; aa >= 0; aa--) acc = acc * xs[i] + npoly_eval(NB_POLY, c + aa * ny, ny, ys[i]);
        out[i] = acc;
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(c, lc);
    return rc;
}

/* numpy polyval3d: Horner in x of (Horner in y of c[a,b,:] at z). */
static int r_polyval3d(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[3].kind != 3 || a[3].arr.ndim != 3) { fn_set_error("npoly.polyval3d: x, y, z arrays and c a 3-D array"); return TSR_EARG; }
    int64_t lx, ly, lz, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL,
        *zs = ys ? fn_arg_doubles(&a[2], &lz) : NULL, *c = zs ? fn_arg_doubles(&a[3], &lc) : NULL;
    const int64_t nx = a[3].arr.shape[0], ny = a[3].arr.shape[1], nz = a[3].arr.shape[2];
    const tsr_array *xa = &a[0].arr; double *out = c ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) {
        double acc = 0.0;
        for (int64_t aa = nx - 1; aa >= 0; aa--) {
            double bacc = 0.0; for (int64_t bb = ny - 1; bb >= 0; bb--) bacc = bacc * ys[i] + npoly_eval(NB_POLY, c + (aa * ny + bb) * nz, nz, zs[i]);
            acc = acc * xs[i] + bacc;
        }
        out[i] = acc;
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(zs, lz); fn_free_doubles(c, lc);
    return rc;
}

/* numpy polygrid2d: the 2-D grid x (outer) by y of the power series c. */
static int r_polygrid2d(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[2].arr.ndim != 2) { fn_set_error("npoly.polygrid2d: x, y arrays and c a 2-D array"); return TSR_EARG; }
    int64_t lx, ly, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *c = ys ? fn_arg_doubles(&a[2], &lc) : NULL;
    const int64_t nx = a[2].arr.shape[0], ny = a[2].arr.shape[1];
    int64_t sh[2] = {lx, ly}; double *out = c ? (double *)fn_result_array(r, TSR_F64, 2, sh) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t p = 0; p < lx; p++) for (int64_t q = 0; q < ly; q++) {
        double acc = 0.0; for (int64_t aa = nx - 1; aa >= 0; aa--) acc = acc * xs[p] + npoly_eval(NB_POLY, c + aa * ny, ny, ys[q]);
        out[p * ly + q] = acc;
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(c, lc);
    return rc;
}

/* numpy polygrid3d: the 3-D grid x by y by z of the power series c. */
static int r_polygrid3d(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[3].kind != 3 || a[3].arr.ndim != 3) { fn_set_error("npoly.polygrid3d: x, y, z arrays and c a 3-D array"); return TSR_EARG; }
    int64_t lx, ly, lz, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL,
        *zs = ys ? fn_arg_doubles(&a[2], &lz) : NULL, *c = zs ? fn_arg_doubles(&a[3], &lc) : NULL;
    const int64_t nx = a[3].arr.shape[0], ny = a[3].arr.shape[1], nz = a[3].arr.shape[2];
    int64_t sh[3] = {lx, ly, lz}; double *out = c ? (double *)fn_result_array(r, TSR_F64, 3, sh) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t p = 0; p < lx; p++) for (int64_t q = 0; q < ly; q++) for (int64_t w = 0; w < lz; w++) {
        double acc = 0.0;
        for (int64_t aa = nx - 1; aa >= 0; aa--) {
            double bacc = 0.0; for (int64_t bb = ny - 1; bb >= 0; bb--) bacc = bacc * ys[q] + npoly_eval(NB_POLY, c + (aa * ny + bb) * nz, nz, zs[w]);
            acc = acc * xs[p] + bacc;
        }
        out[(p * ly + q) * lz + w] = acc;
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(zs, lz); fn_free_doubles(c, lc);
    return rc;
}

/* numpy polyvander2d: columns x**i * y**j, index i*(dy+1)+j. */
static int r_polyvander2d(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3) { fn_set_error("npoly.polyvander2d: x, y and a 2-element deg"); return TSR_EARG; }
    int64_t lx, ly, ld; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *dg = ys ? fn_arg_doubles(&a[2], &ld) : NULL;
    int rc = dg ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        int64_t dx = (int64_t)dg[0], dy = (int64_t)dg[1], ncol = (dx + 1) * (dy + 1);
        int64_t sh[2] = {lx, ncol}; double *V = (double *)fn_result_array(r, TSR_F64, 2, sh);
        if (!V) rc = TSR_ENOMEM;
        else for (int64_t i = 0; i < lx; i++) { double xp = 1.0;
            for (int64_t ii = 0; ii <= dx; ii++) { double yp = 1.0;
                for (int64_t jj = 0; jj <= dy; jj++) { V[i * ncol + ii * (dy + 1) + jj] = xp * yp; yp *= ys[i]; }
                xp *= xs[i]; } }
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(dg, ld);
    return rc;
}

/* numpy polyvander3d: columns x**i * y**j * z**k, index (i*(dy+1)+j)*(dz+1)+k. */
static int r_polyvander3d(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[3].kind != 3) { fn_set_error("npoly.polyvander3d: x, y, z and a 3-element deg"); return TSR_EARG; }
    int64_t lx, ly, lz, ld; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL,
        *zs = ys ? fn_arg_doubles(&a[2], &lz) : NULL, *dg = zs ? fn_arg_doubles(&a[3], &ld) : NULL;
    int rc = dg ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        int64_t dx = (int64_t)dg[0], dy = (int64_t)dg[1], dz = (int64_t)dg[2], ncol = (dx + 1) * (dy + 1) * (dz + 1);
        int64_t sh[2] = {lx, ncol}; double *V = (double *)fn_result_array(r, TSR_F64, 2, sh);
        if (!V) rc = TSR_ENOMEM;
        else for (int64_t i = 0; i < lx; i++) { double xp = 1.0;
            for (int64_t ii = 0; ii <= dx; ii++) { double yp = 1.0;
                for (int64_t jj = 0; jj <= dy; jj++) { double zp = 1.0;
                    for (int64_t kk = 0; kk <= dz; kk++) { V[i * ncol + (ii * (dy + 1) + jj) * (dz + 1) + kk] = xp * yp * zp; zp *= zs[i]; }
                    yp *= ys[i]; }
                xp *= xs[i]; } }
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(zs, lz); fn_free_doubles(dg, ld);
    return rc;
}

/* numpy polycompanion: the companion matrix of a power series (len(c) >= 2). */
static int r_polycompanion(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK && nc < 2) { fn_set_error("npoly.polycompanion: Series must have maximum degree of at least 1"); rc = TSR_EARG; }
    if (rc == TSR_OK) {
        int64_t m = nc - 1; int64_t sh[2] = {m, m}; double *M = (double *)fn_result_array(r, TSR_F64, 2, sh);
        if (!M) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < m * m; i++) M[i] = 0.0;
            for (int64_t i = 1; i < m; i++) M[i * m + (i - 1)] = 1.0;
            for (int64_t i = 0; i < m; i++) M[i * m + (m - 1)] -= c[i] / c[nc - 1]; }
    }
    fn_free_doubles(c, nc);
    return rc;
}

static int npoly_cmp(const void *p, const void *q) { double a = *(const double *)p, b = *(const double *)q; return a < b ? -1 : a > b ? 1 : 0; }

/* numpy polyroots: eigenvalues of the companion matrix, sorted. (Fixtures use real-rooted series.) */
static int r_polyroots(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (nc <= 1) { int64_t sh[1] = {0}; if (!fn_result_array(r, TSR_F64, 1, sh)) rc = TSR_ENOMEM; }
        else if (nc == 2) { double *o = npoly_out1(r, 1); if (!o) rc = TSR_ENOMEM; else o[0] = -c[0] / c[1]; }
        else {
            int64_t m = nc - 1; double *M = calloc(m * m, sizeof(double)), *wr = malloc(m * sizeof(double)), *wi = malloc(m * sizeof(double));
            if (!M || !wr || !wi) rc = TSR_ENOMEM;
            else {
                for (int64_t i = 1; i < m; i++) M[i * m + (i - 1)] = 1.0;
                for (int64_t i = 0; i < m; i++) M[i * m + (m - 1)] -= c[i] / c[nc - 1];
                lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)m, M, (lapack_int)m, wr, wi, NULL, 1, NULL, 1);
                if (info != 0) { fn_set_error("npoly.polyroots: eigenvalue iteration did not converge"); rc = TSR_ECONVERGE; }
                else { qsort(wr, m, sizeof(double), npoly_cmp); double *o = npoly_out1(r, m);
                    if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < m; i++) o[i] = wr[i]; }
            }
            free(M); free(wr); free(wi);
        }
    }
    fn_free_doubles(c, nc);
    return rc;
}

/* numpy polyfit: least-squares power-series fit of degree deg (column-scaled polyvander, rcond = len*eps, dgelsd). */
static int r_polyfit(const void *vx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)vx; (void)nr;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly.polyfit: x and y must be arrays"); return TSR_EARG; }
    int64_t deg = npoly_opt_int(a, n, 2, 0);
    if (deg < 0) { fn_set_error("npoly.polyfit: expected deg >= 0"); return TSR_EARG; }
    int64_t lx, ly; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL;
    int rc = ys ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK && lx != ly) { fn_set_error("npoly.polyfit: expected x and y to have same length"); rc = TSR_EARG; }
    if (rc == TSR_OK) {
        int p = (int)deg + 1; int64_t mb = lx > p ? lx : p;
        double *V = malloc((size_t)lx * p * sizeof(double)), *b = malloc((size_t)mb * sizeof(double)),
               *s = malloc((size_t)p * sizeof(double)), *scale = malloc((size_t)p * sizeof(double));
        if (!V || !b || !s || !scale) rc = TSR_ENOMEM;
        else {
            for (int k = 0; k < p; k++) { double sq = 0.0; for (int64_t i = 0; i < lx; i++) { double e = pow(xs[i], (double)k); sq += e * e; } scale[k] = sq > 0.0 ? sqrt(sq) : 1.0; }
            for (int64_t i = 0; i < lx; i++) { for (int k = 0; k < p; k++) V[i * p + k] = pow(xs[i], (double)k) / scale[k]; b[i] = ys[i]; }
            lapack_int rank = 0;
            lapack_int info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, (lapack_int)lx, p, 1, V, p, b, 1, s, (double)lx * DBL_EPSILON, &rank);
            if (info != 0) { fn_set_error("npoly.polyfit: least-squares solve failed"); rc = TSR_EARG; }
            else { double *o = npoly_out1(r, p); if (!o) rc = TSR_ENOMEM; else for (int k = 0; k < p; k++) o[k] = b[k] / scale[k]; }
        }
        free(V); free(b); free(s); free(scale);
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly);
    return rc;
}

/* the four power-basis module constants (numpy.polynomial.polynomial.polyzero / polyone / polyx / polydomain).
   polyzero/polyone/polyx are integer arrays in NumPy; polydomain is float64. */
static int npoly_iconst(tsr_result *r, const int64_t *v, int64_t m) { int64_t sh[1] = {m}; int64_t *o = (int64_t *)fn_result_array(r, TSR_I64, 1, sh); if (!o) return TSR_ENOMEM; for (int64_t i = 0; i < m; i++) o[i] = v[i]; return TSR_OK; }
static int r_polyzero(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {0}; return npoly_iconst(r, v, 1); }
static int r_polyone(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)  { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {1}; return npoly_iconst(r, v, 1); }
static int r_polyx(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)    { (void)x; (void)a; (void)n; (void)nr; int64_t v[2] = {0, 1}; return npoly_iconst(r, v, 2); }
static int r_polydomain(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; double *o = npoly_out1(r, 2); if (!o) return TSR_ENOMEM; o[0] = -1.0; o[1] = 1.0; return TSR_OK; }

static const fn_def DEFS[] = {
    ROUTINE("npoly.polyval", 1, "x, c", "out", r_polyval, NULL, "Evaluate a power-series polynomial at x (numpy.polynomial.polynomial.polyval)."),
    ROUTINE("npoly.chebval", 1, "x, c", "out", r_chebval, NULL, "Evaluate a Chebyshev series at x (numpy.polynomial.chebyshev.chebval)."),
    ROUTINE("npoly.legval", 1, "x, c", "out", r_legval, NULL, "Evaluate a Legendre series at x (numpy.polynomial.legendre.legval)."),
    ROUTINE("npoly.lagval", 1, "x, c", "out", r_lagval, NULL, "Evaluate a Laguerre series at x (numpy.polynomial.laguerre.lagval)."),
    ROUTINE("npoly.hermval", 1, "x, c", "out", r_hermval, NULL, "Evaluate a Hermite series at x (numpy.polynomial.hermite.hermval)."),
    ROUTINE("npoly.hermeval", 1, "x, c", "out", r_hermeval, NULL, "Evaluate a HermiteE series at x (numpy.polynomial.hermite_e.hermeval)."),
    ROUTINE("npoly.Polynomial", 1, "coef, x", "out", r_cls_polynomial, NULL, "Power-series polynomial evaluated at x (numpy.polynomial.Polynomial)."),
    ROUTINE("npoly.Chebyshev", 1, "coef, x", "out", r_cls_chebyshev, NULL, "Chebyshev series evaluated at x (numpy.polynomial.Chebyshev)."),
    ROUTINE("npoly.Legendre", 1, "coef, x", "out", r_cls_legendre, NULL, "Legendre series evaluated at x (numpy.polynomial.Legendre)."),
    ROUTINE("npoly.Laguerre", 1, "coef, x", "out", r_cls_laguerre, NULL, "Laguerre series evaluated at x (numpy.polynomial.Laguerre)."),
    ROUTINE("npoly.Hermite", 1, "coef, x", "out", r_cls_hermite, NULL, "Hermite series evaluated at x (numpy.polynomial.Hermite)."),
    ROUTINE("npoly.HermiteE", 1, "coef, x", "out", r_cls_hermitee, NULL, "HermiteE series evaluated at x (numpy.polynomial.HermiteE)."),
    ROUTINE("npoly.polyadd", 1, "c1, c2", "out", r_polyadd, NULL, "Sum of two power series (numpy.polynomial.polynomial.polyadd)."),
    ROUTINE("npoly.polysub", 1, "c1, c2", "out", r_polysub, NULL, "Difference of two power series (numpy.polynomial.polynomial.polysub)."),
    ROUTINE("npoly.polymul", 1, "c1, c2", "out", r_polymul, NULL, "Product of two power series (numpy.polynomial.polynomial.polymul)."),
    ROUTINE("npoly.polymulx", 1, "c", "out", r_polymulx, NULL, "Multiply a power series by x (numpy.polynomial.polynomial.polymulx)."),
    ROUTINE("npoly.polypow", 1, "c, pow, maxpower=16", "out", r_polypow, NULL, "Power series raised to an integer power (numpy.polynomial.polynomial.polypow)."),
    ROUTINE("npoly.polydiv", 2, "c1, c2", "quo, rem", r_polydiv, NULL, "Quotient and remainder of power-series division (numpy.polynomial.polynomial.polydiv)."),
    ROUTINE("npoly.polyder", 1, "c, m=1, scl=1", "out", r_polyder, NULL, "Derivative of a power series (numpy.polynomial.polynomial.polyder)."),
    ROUTINE("npoly.polyint", 1, "c, m=1, k=0, lbnd=0, scl=1", "out", r_polyint, NULL, "Antiderivative of a power series (numpy.polynomial.polynomial.polyint)."),
    ROUTINE("npoly.polyfromroots", 1, "roots", "out", r_polyfromroots, NULL, "Power series with the given roots (numpy.polynomial.polynomial.polyfromroots)."),
    ROUTINE("npoly.polyline", 1, "off, scl", "out", r_polyline, NULL, "Power series for off + scl*x (numpy.polynomial.polynomial.polyline)."),
    ROUTINE("npoly.polytrim", 1, "c, tol=0", "out", r_polytrim, NULL, "Trim trailing small coefficients (numpy.polynomial.polynomial.polytrim)."),
    ROUTINE("npoly.polyvander", 1, "x, deg", "out", r_polyvander, NULL, "Pseudo-Vandermonde matrix of the power basis (numpy.polynomial.polynomial.polyvander)."),
    ROUTINE("npoly.polyvalfromroots", 1, "x, r", "out", r_polyvalfromroots, NULL, "Evaluate a polynomial from its roots (numpy.polynomial.polynomial.polyvalfromroots)."),
    ROUTINE("npoly.polyval2d", 1, "x, y, c", "out", r_polyval2d, NULL, "Evaluate a 2-D power series at points (x, y) (numpy.polynomial.polynomial.polyval2d)."),
    ROUTINE("npoly.polyval3d", 1, "x, y, z, c", "out", r_polyval3d, NULL, "Evaluate a 3-D power series at points (x, y, z) (numpy.polynomial.polynomial.polyval3d)."),
    ROUTINE("npoly.polygrid2d", 1, "x, y, c", "out", r_polygrid2d, NULL, "Evaluate a 2-D power series on the grid x by y (numpy.polynomial.polynomial.polygrid2d)."),
    ROUTINE("npoly.polygrid3d", 1, "x, y, z, c", "out", r_polygrid3d, NULL, "Evaluate a 3-D power series on the grid x by y by z (numpy.polynomial.polynomial.polygrid3d)."),
    ROUTINE("npoly.polyvander2d", 1, "x, y, deg", "out", r_polyvander2d, NULL, "Pseudo-Vandermonde matrix of a 2-D power basis (numpy.polynomial.polynomial.polyvander2d)."),
    ROUTINE("npoly.polyvander3d", 1, "x, y, z, deg", "out", r_polyvander3d, NULL, "Pseudo-Vandermonde matrix of a 3-D power basis (numpy.polynomial.polynomial.polyvander3d)."),
    ROUTINE("npoly.polycompanion", 1, "c", "out", r_polycompanion, NULL, "Companion matrix of a power series (numpy.polynomial.polynomial.polycompanion)."),
    ROUTINE("npoly.polyroots", 1, "c", "out", r_polyroots, NULL, "Roots of a power series (numpy.polynomial.polynomial.polyroots)."),
    ROUTINE("npoly.polyfit", 1, "x, y, deg", "out", r_polyfit, NULL, "Least-squares power-series fit (numpy.polynomial.polynomial.polyfit)."),
    ROUTINE("npoly.polyzero", 1, "", "out", r_polyzero, NULL, "The zero power series (numpy.polynomial.polynomial.polyzero)."),
    ROUTINE("npoly.polyone", 1, "", "out", r_polyone, NULL, "The one power series (numpy.polynomial.polynomial.polyone)."),
    ROUTINE("npoly.polyx", 1, "", "out", r_polyx, NULL, "The identity power series x (numpy.polynomial.polynomial.polyx)."),
    ROUTINE("npoly.polydomain", 1, "", "out", r_polydomain, NULL, "The default power-basis domain [-1, 1] (numpy.polynomial.polynomial.polydomain)."),
};

const fn_table TSR_NP_POLYNOMIAL_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
