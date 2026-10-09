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

/* ---- basis-parameterized layer (shared by chebyshev, legendre, laguerre, hermite, hermite_e) -------------
   Each basis provides its coefficient-space primitives; generic routines below take a `const pbasis *` as the
   ROUTINE ctx, so one implementation serves every basis. The power basis keeps its own direct routines above. */
typedef struct pbasis {
    int id;                                                   /* NB_* for npoly_eval */
    const char *sub;                                          /* submodule tag, e.g. "cheb" (for diagnostics) */
    int64_t (*mulx)(const double *c, int64_t n, double *out); /* x * series -> out (cap n+1); returns length */
    int64_t (*mul)(const double *a, int64_t na, const double *b, int64_t nb, double *out); /* out cap na+nb-1 */
    int64_t (*der)(const double *c, int64_t n, double scl, double *out);   /* one derivative step; out cap n */
    int64_t (*integ)(const double *c, int64_t n, double scl, double *out); /* one integral step (k=0,lbnd=0); out cap n+1 */
    int64_t (*line)(double off, double scl, double *out);     /* off + scl*x in this basis; out cap 2 */
    void (*vander)(const double *x, int64_t nx, int64_t deg, double *V);   /* row-major (nx, deg+1) */
    void (*companion)(const double *c, int64_t n, double *M);              /* (n-1)x(n-1), row-major */
} pbasis;

/* ---- chebyshev primitives --------------------------------------------------------------------------------- */
static int64_t cheb_mulx(const double *c, int64_t n, double *out)
{
    if (n == 1 && c[0] == 0.0) { out[0] = 0.0; return 1; }
    for (int64_t i = 0; i <= n; i++) out[i] = 0.0;
    out[1] = c[0];
    for (int64_t i = 1; i < n; i++) { out[i + 1] += c[i] / 2.0; out[i - 1] += c[i] / 2.0; }
    return n + 1;
}
/* c-series <-> z-series (numpy _cseries_to_zseries / _zseries_to_cseries) for Chebyshev multiplication. */
static int64_t cheb_mul(const double *a, int64_t na, const double *b, int64_t nb, double *out)
{
    int64_t za = 2 * na - 1, zb = 2 * nb - 1, zc = za + zb - 1;
    double *A = calloc(za, sizeof(double)), *B = calloc(zb, sizeof(double)), *Z = calloc(zc, sizeof(double));
    if (!A || !B || !Z) { free(A); free(B); free(Z); return -1; }
    for (int64_t i = 0; i < na; i++) { A[na - 1 + i] += a[i] / 2.0; A[na - 1 - i] += a[i] / 2.0; }
    for (int64_t i = 0; i < nb; i++) { B[nb - 1 + i] += b[i] / 2.0; B[nb - 1 - i] += b[i] / 2.0; }
    for (int64_t i = 0; i < za; i++) if (A[i] != 0.0) for (int64_t j = 0; j < zb; j++) Z[i + j] += A[i] * B[j];
    int64_t m = na + nb - 1;                                  /* zc = 2m-1 */
    out[0] = Z[m - 1]; for (int64_t i = 1; i < m; i++) out[i] = Z[m - 1 + i] * 2.0;
    free(A); free(B); free(Z);
    return m;
}
static int64_t cheb_der(const double *c, int64_t n, double scl, double *out)
{
    if (n == 1) { out[0] = 0.0; return 1; }
    double *cc = calloc(n, sizeof(double)); if (!cc) return -1;
    for (int64_t i = 0; i < n; i++) cc[i] = c[i];
    for (int64_t i = 0; i < n - 1; i++) out[i] = 0.0;
    for (int64_t j = n - 1; j >= 3; j--) { out[j - 1] = (2.0 * j) * cc[j]; cc[j - 2] += (double)j * cc[j] / (double)(j - 2); }
    if (n > 2) out[1] = 4.0 * cc[2];
    out[0] = cc[1];
    for (int64_t i = 0; i < n - 1; i++) out[i] *= scl;
    free(cc);
    return n - 1;
}
static int64_t cheb_integ(const double *c, int64_t n, double scl, double *out)
{
    double *cc = calloc(n, sizeof(double)); if (!cc) return -1;
    for (int64_t i = 0; i < n; i++) cc[i] = c[i] * scl;
    for (int64_t i = 0; i <= n; i++) out[i] = 0.0;
    out[1] = cc[0];
    if (n > 1) out[2] = cc[1] / 4.0;
    for (int64_t j = 2; j < n; j++) { out[j + 1] = cc[j] / (2.0 * (j + 1)); out[j - 1] -= cc[j] / (2.0 * (j - 1)); }
    out[0] += -npoly_eval(NB_CHEB, out, n + 1, 0.0);          /* k=0, lbnd=0 */
    free(cc);
    return n + 1;
}
static int64_t cheb_line(double off, double scl, double *out) { out[0] = off; if (scl != 0.0) { out[1] = scl; return 2; } return 1; }
static void cheb_vander(const double *x, int64_t nx, int64_t deg, double *V)
{
    int64_t w = deg + 1;
    for (int64_t i = 0; i < nx; i++) { double *r = V + i * w; r[0] = 1.0;
        if (deg > 0) { r[1] = x[i]; for (int64_t j = 2; j <= deg; j++) r[j] = 2.0 * x[i] * r[j - 1] - r[j - 2]; } }
}
static void cheb_companion(const double *c, int64_t n, double *M)
{
    int64_t m = n - 1;
    if (m == 1) { M[0] = -c[0] / c[1]; return; }
    for (int64_t i = 0; i < m * m; i++) M[i] = 0.0;
    for (int64_t i = 0; i < m - 1; i++) { double v = (i == 0) ? sqrt(0.5) : 0.5; M[i * m + (i + 1)] = v; M[(i + 1) * m + i] = v; }
    double scl_last = (m - 1 >= 1) ? sqrt(0.5) : 1.0;
    for (int64_t i = 0; i < m; i++) { double scl_i = (i == 0) ? 1.0 : sqrt(0.5); M[i * m + (m - 1)] -= (c[i] / c[n - 1]) * (scl_i / scl_last) * 0.5; }
}
static const pbasis CHEB = {NB_CHEB, "cheb", cheb_mulx, cheb_mul, cheb_der, cheb_integ, cheb_line, cheb_vander, cheb_companion};

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

/* ---- generic basis routines (ctx = const pbasis *) -------------------------------------------------------- */

static int b_addsub(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int sign)
{ (void)ctx; return npoly_addsub(a, n, r, sign); }
static int rb_add(const void *c, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)nr; return b_addsub(c, a, n, r, +1); }
static int rb_sub(const void *c, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)nr; return b_addsub(c, a, n, r, -1); }
static int rb_trim(const void *c, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)c; return r_polytrim(NULL, a, n, r, nr); }

static int rb_mulx(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { double *sc = malloc((nc + 1) * sizeof(double)); if (!sc) rc = TSR_ENOMEM;
        else { int64_t m = B->mulx(c, nc, sc); double *o = npoly_out1(r, m); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < m; i++) o[i] = sc[i]; free(sc); } }
    fn_free_doubles(c, nc);
    return rc;
}
static int rb_mul(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly: coefficients must be arrays"); return TSR_EARG; }
    int64_t n1, n2; double *c1 = fn_arg_doubles(&a[0], &n1), *c2 = c1 ? fn_arg_doubles(&a[1], &n2) : NULL;
    int rc = c2 ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { int64_t cap = n1 + n2 - 1; double *buf = malloc(cap * sizeof(double)); if (!buf) rc = TSR_ENOMEM;
        else { int64_t m = B->mul(c1, n1, c2, n2, buf); if (m < 0) rc = TSR_ENOMEM; else { double *o = npoly_out1(r, m); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < m; i++) o[i] = buf[i]; } free(buf); } }
    fn_free_doubles(c1, n1); fn_free_doubles(c2, n2);
    return rc;
}
static int rb_pow(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t pw = npoly_opt_int(a, n, 1, 0), maxp = npoly_opt_int(a, n, 2, 16);
    if (pw < 0) { fn_set_error("npoly.pow: Power must be non-negative"); return TSR_EARG; }
    if (pw > maxp) { fn_set_error("npoly.pow: Power is too large"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (pw == 0) { double *o = npoly_out1(r, 1); if (!o) rc = TSR_ENOMEM; else o[0] = 1.0; }
        else {
            int64_t cap = (nc - 1) * pw + 1; double *acc = malloc(cap * sizeof(double)), *tmp = malloc(cap * sizeof(double));
            if (!acc || !tmp) rc = TSR_ENOMEM;
            else { int64_t la = nc; for (int64_t i = 0; i < nc; i++) acc[i] = c[i];
                for (int64_t p = 1; p < pw && rc == TSR_OK; p++) { int64_t lt = B->mul(acc, la, c, nc, tmp); if (lt < 0) rc = TSR_ENOMEM; else { for (int64_t i = 0; i < lt; i++) acc[i] = tmp[i]; la = lt; } }
                if (rc == TSR_OK) { double *o = npoly_out1(r, la); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < la; i++) o[i] = acc[i]; } }
            free(acc); free(tmp);
        }
    }
    fn_free_doubles(c, nc);
    return rc;
}
static int rb_div(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly: coefficients must be arrays"); return TSR_EARG; }
    int64_t n1, n2; double *c1 = fn_arg_doubles(&a[0], &n1), *c2 = c1 ? fn_arg_doubles(&a[1], &n2) : NULL;
    int rc = c2 ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK && c2[n2 - 1] == 0.0) { fn_set_error("npoly.div: division by zero"); rc = TSR_EARG; }
    if (rc == TSR_OK && n2 == 1) { double *q = npoly_out1(&r[0], n1), *rem = q ? npoly_out1(&r[1], 1) : NULL;
        if (!q || !rem) rc = TSR_ENOMEM; else { for (int64_t i = 0; i < n1; i++) q[i] = c1[i] / c2[0]; rem[0] = 0.0; } }
    else if (rc == TSR_OK && n1 < n2) { double *q = npoly_out1(&r[0], 1), *rem = q ? npoly_out1(&r[1], n1) : NULL;
        if (!q || !rem) rc = TSR_ENOMEM; else { q[0] = 0.0; for (int64_t i = 0; i < n1; i++) rem[i] = c1[i]; } }
    else if (rc == TSR_OK) {
        int64_t ql = n1 - n2 + 1; double *quo = calloc(ql, sizeof(double)), *rem = malloc(n1 * sizeof(double)),
               *unit = malloc((n1 - n2 + 1) * sizeof(double)), *p = malloc(n1 * sizeof(double));
        if (!quo || !rem || !unit || !p) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < n1; i++) rem[i] = c1[i];
            for (int64_t i = n1 - n2; i >= 0 && rc == TSR_OK; i--) {
                for (int64_t k = 0; k <= i; k++) unit[k] = 0.0;
                unit[i] = 1.0;
                int64_t pl = B->mul(unit, i + 1, c2, n2, p);   /* pl = i + n2 */
                if (pl < 0) { rc = TSR_ENOMEM; break; }
                double q = rem[i + n2 - 1] / p[i + n2 - 1];
                for (int64_t k = 0; k < pl; k++) rem[k] -= q * p[k];
                quo[i] = q;
            }
            if (rc == TSR_OK) { int64_t rl = npoly_trimlen(rem, n2 - 1, 0.0);
                double *qo = npoly_out1(&r[0], ql), *ro = qo ? npoly_out1(&r[1], rl) : NULL;
                if (!qo || !ro) rc = TSR_ENOMEM; else { for (int64_t k = 0; k < ql; k++) qo[k] = quo[k]; for (int64_t k = 0; k < rl; k++) ro[k] = rem[k]; } }
        }
        free(quo); free(rem); free(unit); free(p);
    }
    fn_free_doubles(c1, n1); fn_free_doubles(c2, n2);
    return rc;
}
static int rb_der(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t m = npoly_opt_int(a, n, 1, 1); double scl = npoly_opt_num(a, n, 2, 1.0);
    if (m < 0) { fn_set_error("npoly.der: order must be non-negative"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { double *cur = malloc(nc * sizeof(double)), *nxt = malloc(nc * sizeof(double));
        if (!cur || !nxt) rc = TSR_ENOMEM;
        else { int64_t len = nc; for (int64_t i = 0; i < nc; i++) cur[i] = c[i];
            for (int64_t it = 0; it < m; it++) { int64_t nl = B->der(cur, len, scl, nxt); double *t = cur; cur = nxt; nxt = t; len = nl; }
            double *o = npoly_out1(r, len); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < len; i++) o[i] = cur[i]; }
        free(cur); free(nxt); }
    fn_free_doubles(c, nc);
    return rc;
}
static int rb_int(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t m = npoly_opt_int(a, n, 1, 1); double scl = npoly_opt_num(a, n, 4, 1.0);
    if (m < 0) { fn_set_error("npoly.int: order must be non-negative"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { double *cur = malloc((nc + m) * sizeof(double)), *nxt = malloc((nc + m) * sizeof(double));
        if (!cur || !nxt) rc = TSR_ENOMEM;
        else { int64_t len = nc; for (int64_t i = 0; i < nc; i++) cur[i] = c[i];
            for (int64_t it = 0; it < m; it++) { int64_t nl = B->integ(cur, len, scl, nxt); double *t = cur; cur = nxt; nxt = t; len = nl; }
            double *o = npoly_out1(r, len); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < len; i++) o[i] = cur[i]; }
        free(cur); free(nxt); }
    fn_free_doubles(c, nc);
    return rc;
}
static int rb_line(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)nr; const pbasis *B = ctx;
    double off = npoly_opt_num(a, n, 0, 0.0), scl = npoly_opt_num(a, n, 1, 0.0);
    double buf[2]; int64_t m = B->line(off, scl, buf); double *o = npoly_out1(r, m);
    if (!o) return TSR_ENOMEM;
    for (int64_t i = 0; i < m; i++) o[i] = buf[i];
    return TSR_OK;
}
static int rb_fromroots(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: roots must be an array"); return TSR_EARG; }
    int64_t nrt; double *rt = fn_arg_doubles(&a[0], &nrt);
    int rc = rt ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (nrt == 0) { double *o = npoly_out1(r, 1); if (!o) rc = TSR_ENOMEM; else o[0] = 1.0; }
        else { int64_t cap = nrt + 1; double *acc = malloc(cap * sizeof(double)), *tmp = malloc(cap * sizeof(double)); double ln[2];
            if (!acc || !tmp) rc = TSR_ENOMEM;
            else { acc[0] = 1.0; int64_t la = 1;
                for (int64_t k = 0; k < nrt && rc == TSR_OK; k++) { int64_t ll = B->line(-rt[k], 1.0, ln); int64_t m = B->mul(acc, la, ln, ll, tmp); if (m < 0) rc = TSR_ENOMEM; else { for (int64_t i = 0; i < m; i++) acc[i] = tmp[i]; la = m; } }
                if (rc == TSR_OK) { double *o = npoly_out1(r, la); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < la; i++) o[i] = acc[i]; } }
            free(acc); free(tmp); }
    }
    fn_free_doubles(rt, nrt);
    return rc;
}
static int rb_vander(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: x must be an array"); return TSR_EARG; }
    int64_t deg = npoly_opt_int(a, n, 1, 0);
    if (deg < 0) { fn_set_error("npoly.vander: deg must be non-negative"); return TSR_EARG; }
    int64_t lx; double *xs = fn_arg_doubles(&a[0], &lx);
    int rc = xs ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { int64_t sh[2] = {lx, deg + 1}; double *V = (double *)fn_result_array(r, TSR_F64, 2, sh); if (!V) rc = TSR_ENOMEM; else B->vander(xs, lx, deg, V); }
    fn_free_doubles(xs, lx);
    return rc;
}
/* NumPy's *val2d/val3d/grid2d/grid3d reduce the coefficient tensor one axis at a time, axis 0 (x) first; we
   gather each axis into a contiguous buffer and evaluate in that same order, so the arithmetic matches NumPy. */
static int rb_val2d(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[2].arr.ndim != 2) { fn_set_error("npoly.val2d: x, y arrays and c 2-D"); return TSR_EARG; }
    int64_t lx, ly, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *c = ys ? fn_arg_doubles(&a[2], &lc) : NULL;
    const int64_t nx = a[2].arr.shape[0], ny = a[2].arr.shape[1];
    const tsr_array *xa = &a[0].arr; double *out = c ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    double *ca = out ? malloc(nx * sizeof(double)) : NULL, *g = ca ? malloc(ny * sizeof(double)) : NULL;
    int rc = g ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) {
        for (int64_t bb = 0; bb < ny; bb++) { for (int64_t aa = 0; aa < nx; aa++) ca[aa] = c[aa * ny + bb]; g[bb] = npoly_eval(B->id, ca, nx, xs[i]); }
        out[i] = npoly_eval(B->id, g, ny, ys[i]);
    }
    free(ca); free(g); fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(c, lc);
    return rc;
}
static int rb_val3d(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[3].kind != 3 || a[3].arr.ndim != 3) { fn_set_error("npoly.val3d: x, y, z arrays and c 3-D"); return TSR_EARG; }
    int64_t lx, ly, lz, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *zs = ys ? fn_arg_doubles(&a[2], &lz) : NULL, *c = zs ? fn_arg_doubles(&a[3], &lc) : NULL;
    const int64_t nx = a[3].arr.shape[0], ny = a[3].arr.shape[1], nz = a[3].arr.shape[2];
    const tsr_array *xa = &a[0].arr; double *out = c ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    double *ca = out ? malloc(nx * sizeof(double)) : NULL, *g = ca ? malloc(ny * nz * sizeof(double)) : NULL, *cb = g ? malloc(ny * sizeof(double)) : NULL, *h = cb ? malloc(nz * sizeof(double)) : NULL;
    int rc = h ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) {
        for (int64_t bb = 0; bb < ny; bb++) for (int64_t dd = 0; dd < nz; dd++) { for (int64_t aa = 0; aa < nx; aa++) ca[aa] = c[(aa * ny + bb) * nz + dd]; g[bb * nz + dd] = npoly_eval(B->id, ca, nx, xs[i]); }
        for (int64_t dd = 0; dd < nz; dd++) { for (int64_t bb = 0; bb < ny; bb++) cb[bb] = g[bb * nz + dd]; h[dd] = npoly_eval(B->id, cb, ny, ys[i]); }
        out[i] = npoly_eval(B->id, h, nz, zs[i]);
    }
    free(ca); free(g); free(cb); free(h); fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(zs, lz); fn_free_doubles(c, lc);
    return rc;
}
static int rb_grid2d(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[2].arr.ndim != 2) { fn_set_error("npoly.grid2d: x, y arrays and c 2-D"); return TSR_EARG; }
    int64_t lx, ly, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *c = ys ? fn_arg_doubles(&a[2], &lc) : NULL;
    const int64_t nx = a[2].arr.shape[0], ny = a[2].arr.shape[1];
    int64_t sh[2] = {lx, ly}; double *out = c ? (double *)fn_result_array(r, TSR_F64, 2, sh) : NULL;
    double *ca = out ? malloc(nx * sizeof(double)) : NULL, *g = ca ? malloc(ny * sizeof(double)) : NULL;
    int rc = g ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t p = 0; p < lx; p++) {
        for (int64_t bb = 0; bb < ny; bb++) { for (int64_t aa = 0; aa < nx; aa++) ca[aa] = c[aa * ny + bb]; g[bb] = npoly_eval(B->id, ca, nx, xs[p]); }
        for (int64_t q = 0; q < ly; q++) out[p * ly + q] = npoly_eval(B->id, g, ny, ys[q]);
    }
    free(ca); free(g); fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(c, lc);
    return rc;
}
static int rb_grid3d(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[3].kind != 3 || a[3].arr.ndim != 3) { fn_set_error("npoly.grid3d: x, y, z arrays and c 3-D"); return TSR_EARG; }
    int64_t lx, ly, lz, lc; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *zs = ys ? fn_arg_doubles(&a[2], &lz) : NULL, *c = zs ? fn_arg_doubles(&a[3], &lc) : NULL;
    const int64_t nx = a[3].arr.shape[0], ny = a[3].arr.shape[1], nz = a[3].arr.shape[2];
    int64_t sh[3] = {lx, ly, lz}; double *out = c ? (double *)fn_result_array(r, TSR_F64, 3, sh) : NULL;
    double *ca = out ? malloc(nx * sizeof(double)) : NULL, *g = ca ? malloc(ny * nz * sizeof(double)) : NULL, *cb = g ? malloc(ny * sizeof(double)) : NULL, *h = cb ? malloc(nz * sizeof(double)) : NULL;
    int rc = h ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t p = 0; p < lx; p++) {
        for (int64_t bb = 0; bb < ny; bb++) for (int64_t dd = 0; dd < nz; dd++) { for (int64_t aa = 0; aa < nx; aa++) ca[aa] = c[(aa * ny + bb) * nz + dd]; g[bb * nz + dd] = npoly_eval(B->id, ca, nx, xs[p]); }
        for (int64_t q = 0; q < ly; q++) {
            for (int64_t dd = 0; dd < nz; dd++) { for (int64_t bb = 0; bb < ny; bb++) cb[bb] = g[bb * nz + dd]; h[dd] = npoly_eval(B->id, cb, ny, ys[q]); }
            for (int64_t w = 0; w < lz; w++) out[(p * ly + q) * lz + w] = npoly_eval(B->id, h, nz, zs[w]);
        }
    }
    free(ca); free(g); free(cb); free(h); fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(zs, lz); fn_free_doubles(c, lc);
    return rc;
}
static int rb_vander2d(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3) { fn_set_error("npoly.vander2d: x, y and 2-element deg"); return TSR_EARG; }
    int64_t lx, ly, ld; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *dg = ys ? fn_arg_doubles(&a[2], &ld) : NULL;
    int rc = dg ? TSR_OK : TSR_ENOMEM;
    double *VX = NULL, *VY = NULL;
    if (rc == TSR_OK) {
        int64_t dx = (int64_t)dg[0], dy = (int64_t)dg[1], ncol = (dx + 1) * (dy + 1);
        VX = malloc(lx * (dx + 1) * sizeof(double)); VY = malloc(lx * (dy + 1) * sizeof(double));
        if (!VX || !VY) rc = TSR_ENOMEM;
        else { B->vander(xs, lx, dx, VX); B->vander(ys, lx, dy, VY);
            int64_t sh[2] = {lx, ncol}; double *V = (double *)fn_result_array(r, TSR_F64, 2, sh);
            if (!V) rc = TSR_ENOMEM; else for (int64_t i = 0; i < lx; i++) for (int64_t ii = 0; ii <= dx; ii++) for (int64_t jj = 0; jj <= dy; jj++) V[i * ncol + ii * (dy + 1) + jj] = VX[i * (dx + 1) + ii] * VY[i * (dy + 1) + jj]; }
    }
    free(VX); free(VY); fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(dg, ld);
    return rc;
}
static int rb_vander3d(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3 || a[2].kind != 3 || a[3].kind != 3) { fn_set_error("npoly.vander3d: x, y, z and 3-element deg"); return TSR_EARG; }
    int64_t lx, ly, lz, ld; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL, *zs = ys ? fn_arg_doubles(&a[2], &lz) : NULL, *dg = zs ? fn_arg_doubles(&a[3], &ld) : NULL;
    int rc = dg ? TSR_OK : TSR_ENOMEM;
    double *VX = NULL, *VY = NULL, *VZ = NULL;
    if (rc == TSR_OK) {
        int64_t dx = (int64_t)dg[0], dy = (int64_t)dg[1], dz = (int64_t)dg[2], ncol = (dx + 1) * (dy + 1) * (dz + 1);
        VX = malloc(lx * (dx + 1) * sizeof(double)); VY = malloc(lx * (dy + 1) * sizeof(double)); VZ = malloc(lx * (dz + 1) * sizeof(double));
        if (!VX || !VY || !VZ) rc = TSR_ENOMEM;
        else { B->vander(xs, lx, dx, VX); B->vander(ys, lx, dy, VY); B->vander(zs, lx, dz, VZ);
            int64_t sh[2] = {lx, ncol}; double *V = (double *)fn_result_array(r, TSR_F64, 2, sh);
            if (!V) rc = TSR_ENOMEM; else for (int64_t i = 0; i < lx; i++) for (int64_t ii = 0; ii <= dx; ii++) for (int64_t jj = 0; jj <= dy; jj++) for (int64_t kk = 0; kk <= dz; kk++) V[i * ncol + (ii * (dy + 1) + jj) * (dz + 1) + kk] = VX[i * (dx + 1) + ii] * VY[i * (dy + 1) + jj] * VZ[i * (dz + 1) + kk]; }
    }
    free(VX); free(VY); free(VZ); fn_free_doubles(xs, lx); fn_free_doubles(ys, ly); fn_free_doubles(zs, lz); fn_free_doubles(dg, ld);
    return rc;
}
static int rb_companion(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK && nc < 2) { fn_set_error("npoly.companion: series must have degree >= 1"); rc = TSR_EARG; }
    if (rc == TSR_OK) { int64_t m = nc - 1; int64_t sh[2] = {m, m}; double *M = (double *)fn_result_array(r, TSR_F64, 2, sh); if (!M) rc = TSR_ENOMEM; else B->companion(c, nc, M); }
    fn_free_doubles(c, nc);
    return rc;
}
static int rb_roots(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)n; (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (nc <= 1) { int64_t sh[1] = {0}; if (!fn_result_array(r, TSR_F64, 1, sh)) rc = TSR_ENOMEM; }
        else if (nc == 2) { double M1; B->companion(c, 2, &M1); double *o = npoly_out1(r, 1); if (!o) rc = TSR_ENOMEM; else o[0] = M1; }
        else { int64_t m = nc - 1; double *M = malloc(m * m * sizeof(double)), *wr = malloc(m * sizeof(double)), *wi = malloc(m * sizeof(double));
            if (!M || !wr || !wi) rc = TSR_ENOMEM;
            else { B->companion(c, nc, M);
                lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)m, M, (lapack_int)m, wr, wi, NULL, 1, NULL, 1);
                if (info != 0) { fn_set_error("npoly.roots: eigenvalue iteration did not converge"); rc = TSR_ECONVERGE; }
                else { qsort(wr, m, sizeof(double), npoly_cmp); double *o = npoly_out1(r, m); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < m; i++) o[i] = wr[i]; } }
            free(M); free(wr); free(wi); }
    }
    fn_free_doubles(c, nc);
    return rc;
}
static int rb_fit(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)nr; const pbasis *B = ctx;
    if (a[0].kind != 3 || a[1].kind != 3) { fn_set_error("npoly.fit: x and y must be arrays"); return TSR_EARG; }
    int64_t deg = npoly_opt_int(a, n, 2, 0);
    if (deg < 0) { fn_set_error("npoly.fit: expected deg >= 0"); return TSR_EARG; }
    int64_t lx, ly; double *xs = fn_arg_doubles(&a[0], &lx), *ys = xs ? fn_arg_doubles(&a[1], &ly) : NULL;
    int rc = ys ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK && lx != ly) { fn_set_error("npoly.fit: x and y must have the same length"); rc = TSR_EARG; }
    if (rc == TSR_OK) {
        int p = (int)deg + 1; int64_t mb = lx > p ? lx : p;
        double *V = malloc((size_t)lx * p * sizeof(double)), *b = malloc((size_t)mb * sizeof(double)), *s = malloc((size_t)p * sizeof(double)), *scale = malloc((size_t)p * sizeof(double));
        if (!V || !b || !s || !scale) rc = TSR_ENOMEM;
        else { B->vander(xs, lx, deg, V);                      /* row-major (lx, p) */
            for (int k = 0; k < p; k++) { double sq = 0.0; for (int64_t i = 0; i < lx; i++) { double e = V[i * p + k]; sq += e * e; } scale[k] = sq > 0.0 ? sqrt(sq) : 1.0; }
            for (int64_t i = 0; i < lx; i++) { for (int k = 0; k < p; k++) V[i * p + k] /= scale[k]; b[i] = ys[i]; }
            lapack_int rank = 0;
            lapack_int info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, (lapack_int)lx, p, 1, V, p, b, 1, s, (double)lx * DBL_EPSILON, &rank);
            if (info != 0) { fn_set_error("npoly.fit: least-squares solve failed"); rc = TSR_EARG; }
            else { double *o = npoly_out1(r, p); if (!o) rc = TSR_ENOMEM; else for (int k = 0; k < p; k++) o[k] = b[k] / scale[k]; } }
        free(V); free(b); free(s); free(scale);
    }
    fn_free_doubles(xs, lx); fn_free_doubles(ys, ly);
    return rc;
}

/* ---- chebyshev-specific: basis conversions, points, Gauss quadrature, weight, constants ------------------- */
/* cheb2poly: NumPy's Horner on the power basis using polyadd/polymulx/polysub. */
static int r_cheb2poly(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (nc < 3) { double *o = npoly_out1(r, nc); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < nc; i++) o[i] = c[i]; }
        else { int64_t cap = nc + 1; double *c0 = calloc(cap, sizeof(double)), *c1 = calloc(cap, sizeof(double)), *tmp = calloc(cap, sizeof(double)), *mx = calloc(cap, sizeof(double));
            if (!c0 || !c1 || !tmp || !mx) rc = TSR_ENOMEM;
            else { int64_t l0 = 1, l1 = 1; c0[0] = c[nc - 2]; c1[0] = c[nc - 1];
                for (int64_t i = nc - 1; i >= 2; i--) {
                    int64_t lt = l0; for (int64_t k = 0; k < l0; k++) tmp[k] = c0[k];                 /* tmp = c0 */
                    int64_t ns = l1;                                                                   /* polysub([c[i-2]], c1) -> length l1 */
                    for (int64_t k = 0; k < ns; k++) c0[k] = (k == 0 ? c[i - 2] : 0.0) - c1[k];
                    l0 = ns;
                    int64_t lmx = (l1 == 1 && c1[0] == 0.0) ? 1 : l1 + 1;                               /* 2*polymulx(c1) */
                    if (lmx == 1) mx[0] = 0.0; else { mx[0] = 0.0; for (int64_t k = 0; k < l1; k++) mx[k + 1] = 2.0 * c1[k]; }
                    int64_t ln = lt > lmx ? lt : lmx;                                                   /* c1 = polyadd(tmp, 2*mulx(c1)) */
                    for (int64_t k = 0; k < ln; k++) c1[k] = (k < lt ? tmp[k] : 0.0) + (k < lmx ? mx[k] : 0.0);
                    l1 = ln;
                }
                int64_t lmx = (l1 == 1 && c1[0] == 0.0) ? 1 : l1 + 1;                                   /* polyadd(c0, polymulx(c1)) */
                if (lmx == 1) mx[0] = 0.0; else { mx[0] = 0.0; for (int64_t k = 0; k < l1; k++) mx[k + 1] = c1[k]; }
                int64_t lr = l0 > lmx ? l0 : lmx; double *o = npoly_out1(r, lr);
                if (!o) rc = TSR_ENOMEM; else for (int64_t k = 0; k < lr; k++) o[k] = (k < l0 ? c0[k] : 0.0) + (k < lmx ? mx[k] : 0.0);
            }
            free(c0); free(c1); free(tmp); free(mx); }
    }
    fn_free_doubles(c, nc);
    return rc;
}
/* poly2cheb: res = chebadd(chebmulx(res), [pol[i]]) folded from the top. */
static int r_poly2cheb(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *pol = fn_arg_doubles(&a[0], &nc);
    int rc = pol ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { double *res = calloc(nc + 1, sizeof(double)), *mx = calloc(nc + 1, sizeof(double));
        if (!res || !mx) rc = TSR_ENOMEM;
        else { int64_t lr = 1; res[0] = 0.0;
            for (int64_t i = nc - 1; i >= 0; i--) {
                int64_t lm = cheb_mulx(res, lr, mx);                                                    /* chebmulx(res) */
                for (int64_t k = 0; k < lm; k++) res[k] = mx[k];
                res[0] += pol[i]; lr = lm;                                                              /* + [pol[i]] */
            }
            double *o = npoly_out1(r, lr); if (!o) rc = TSR_ENOMEM; else for (int64_t k = 0; k < lr; k++) o[k] = res[k]; }
        free(res); free(mx); }
    fn_free_doubles(pol, nc);
    return rc;
}
static int r_chebpts1(const void *x, const tsr_arg *a, int nn, tsr_result *r, int nr)
{
    (void)x; (void)nr; int64_t m = npoly_opt_int(a, nn, 0, 0);
    if (m < 1) { fn_set_error("npoly.chebpts1: npts must be >= 1"); return TSR_EARG; }
    double *o = npoly_out1(r, m); if (!o) return TSR_ENOMEM;
    /* numpy: 0.5*pi/m * arange(-m+1, m+1, 2), then sin (so the midpoint is sin(0) = 0 exactly) */
    for (int64_t i = 0; i < m; i++) o[i] = sin(0.5 * M_PI / (double)m * (double)(-m + 1 + 2 * i));
    return TSR_OK;
}
static int r_chebpts2(const void *x, const tsr_arg *a, int nn, tsr_result *r, int nr)
{
    (void)x; (void)nr; int64_t m = npoly_opt_int(a, nn, 0, 0);
    if (m < 2) { fn_set_error("npoly.chebpts2: npts must be >= 2"); return TSR_EARG; }
    double *o = npoly_out1(r, m); if (!o) return TSR_ENOMEM;
    for (int64_t i = 0; i < m; i++) o[i] = cos(M_PI * (double)(m - 1 - i) / (double)(m - 1));
    return TSR_OK;
}
/* chebgauss: nodes cos(pi*(2k+1)/(2n)), equal weights pi/n. */
static int r_chebgauss(const void *x, const tsr_arg *a, int nn, tsr_result *r, int nr)
{
    (void)x; (void)nr; int64_t m = npoly_opt_int(a, nn, 0, 0);
    if (m < 1) { fn_set_error("npoly.chebgauss: deg must be >= 1"); return TSR_EARG; }
    double *xo = npoly_out1(&r[0], m), *wo = xo ? npoly_out1(&r[1], m) : NULL;
    if (!xo || !wo) return TSR_ENOMEM;
    for (int64_t k = 0; k < m; k++) { xo[k] = cos(M_PI * (double)(2 * k + 1) / (2.0 * (double)m)); wo[k] = M_PI / (double)m; }
    return TSR_OK;
}
/* chebweight: 1/sqrt(1-x^2). */
static int r_chebweight(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly.chebweight: x must be an array"); return TSR_EARG; }
    int64_t lx; double *xs = fn_arg_doubles(&a[0], &lx);
    const tsr_array *xa = &a[0].arr; double *out = xs ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) out[i] = 1.0 / (sqrt(1.0 + xs[i]) * sqrt(1.0 - xs[i]));
    fn_free_doubles(xs, lx);
    return rc;
}
static int r_chebzero(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {0}; return npoly_iconst(r, v, 1); }
static int r_chebone(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)  { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {1}; return npoly_iconst(r, v, 1); }
static int r_chebx(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)    { (void)x; (void)a; (void)n; (void)nr; int64_t v[2] = {0, 1}; return npoly_iconst(r, v, 2); }
static int r_chebdomain(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; double *o = npoly_out1(r, 2); if (!o) return TSR_ENOMEM; o[0] = -1.0; o[1] = 1.0; return TSR_OK; }

/* ---- legendre primitives --------------------------------------------------------------------------------- */
static int64_t leg_mulx(const double *c, int64_t n, double *out)
{
    if (n == 1 && c[0] == 0.0) { out[0] = 0.0; return 1; }
    for (int64_t i = 0; i <= n; i++) out[i] = 0.0;
    out[1] = c[0];
    for (int64_t i = 1; i < n; i++) { int64_t j = i + 1, k = i - 1; double s = i + j; out[j] = c[i] * (double)j / s; out[k] += c[i] * (double)i / s; }
    return n + 1;
}
/* NumPy legmul: Clenshaw product using legmulx and the Legendre recurrence (c1 has no bare-c1 term). */
static int64_t leg_mul(const double *A, int64_t nA, const double *Bv, int64_t nB, double *out)
{
    const double *c, *xs; int64_t nc, nxs;
    if (nA > nB) { c = Bv; nc = nB; xs = A; nxs = nA; } else { c = A; nc = nA; xs = Bv; nxs = nB; }
    int64_t cap = nA + nB + 2;
    double *c0 = calloc(cap, sizeof(double)), *c1 = calloc(cap, sizeof(double)), *tmp = calloc(cap, sizeof(double)), *mx = calloc(cap, sizeof(double));
    int64_t ret = -1;
    if (c0 && c1 && tmp && mx) {
        int64_t l0, l1;
        if (nc == 1) { for (int64_t k = 0; k < nxs; k++) c0[k] = c[0] * xs[k]; l0 = nxs; c1[0] = 0.0; l1 = 1; }
        else {
            for (int64_t k = 0; k < nxs; k++) { c0[k] = c[nc - 2] * xs[k]; c1[k] = c[nc - 1] * xs[k]; } l0 = nxs; l1 = nxs;
            double nd = (double)nc;
            for (int64_t i = 3; i <= nc; i++) {
                int64_t lt = l0; for (int64_t k = 0; k < l0; k++) tmp[k] = c0[k];
                nd -= 1.0;
                double f = (nd - 1.0) / nd, sc = c[nc - i]; int64_t nl0 = nxs > l1 ? nxs : l1;
                for (int64_t k = 0; k < nl0; k++) c0[k] = (k < nxs ? sc * xs[k] : 0.0) - (k < l1 ? f * c1[k] : 0.0);
                l0 = nl0;
                int64_t lm = leg_mulx(c1, l1, mx); double g = (2.0 * nd - 1.0) / nd; int64_t nl1 = lt > lm ? lt : lm;
                for (int64_t k = 0; k < nl1; k++) c1[k] = (k < lt ? tmp[k] : 0.0) + (k < lm ? g * mx[k] : 0.0);
                l1 = nl1;
            }
        }
        int64_t lm = leg_mulx(c1, l1, mx); int64_t nr = l0 > lm ? l0 : lm;
        for (int64_t k = 0; k < nr; k++) out[k] = (k < l0 ? c0[k] : 0.0) + (k < lm ? mx[k] : 0.0);
        ret = nr;
    }
    free(c0); free(c1); free(tmp); free(mx);
    return ret;
}
static int64_t leg_der(const double *c, int64_t n, double scl, double *out)
{
    if (n == 1) { out[0] = 0.0; return 1; }
    double *cc = calloc(n, sizeof(double)); if (!cc) return -1;
    for (int64_t i = 0; i < n; i++) cc[i] = c[i];
    for (int64_t i = 0; i < n - 1; i++) out[i] = 0.0;
    for (int64_t j = n - 1; j >= 3; j--) { out[j - 1] = (2.0 * j - 1.0) * cc[j]; cc[j - 2] += cc[j]; }
    if (n > 2) out[1] = 3.0 * cc[2];
    out[0] = cc[1];
    for (int64_t i = 0; i < n - 1; i++) out[i] *= scl;
    free(cc);
    return n - 1;
}
static int64_t leg_integ(const double *c, int64_t n, double scl, double *out)
{
    double *cc = calloc(n, sizeof(double)); if (!cc) return -1;
    for (int64_t i = 0; i < n; i++) cc[i] = c[i] * scl;
    for (int64_t i = 0; i <= n; i++) out[i] = 0.0;
    out[1] = cc[0];
    if (n > 1) out[2] = cc[1] / 3.0;
    for (int64_t j = 2; j < n; j++) { double t = cc[j] / (2.0 * j + 1.0); out[j + 1] = t; out[j - 1] -= t; }
    out[0] += -npoly_eval(NB_LEG, out, n + 1, 0.0);
    free(cc);
    return n + 1;
}
static int64_t leg_line(double off, double scl, double *out) { out[0] = off; if (scl != 0.0) { out[1] = scl; return 2; } return 1; }
static void leg_vander(const double *x, int64_t nx, int64_t deg, double *V)
{
    int64_t w = deg + 1;
    for (int64_t i = 0; i < nx; i++) { double *rr = V + i * w; rr[0] = 1.0;
        if (deg > 0) { rr[1] = x[i]; for (int64_t j = 2; j <= deg; j++) rr[j] = (rr[j - 1] * x[i] * (2.0 * j - 1.0) - rr[j - 2] * (j - 1.0)) / (double)j; } }
}
static void leg_companion(const double *c, int64_t n, double *M)
{
    int64_t m = n - 1;
    if (m == 1) { M[0] = -c[0] / c[1]; return; }
    for (int64_t i = 0; i < m * m; i++) M[i] = 0.0;
    double *scl = malloc(m * sizeof(double)); if (!scl) return;
    for (int64_t i = 0; i < m; i++) scl[i] = 1.0 / sqrt(2.0 * i + 1.0);
    for (int64_t i = 0; i < m - 1; i++) { double v = (i + 1.0) * scl[i] * scl[i + 1]; M[i * m + (i + 1)] = v; M[(i + 1) * m + i] = v; }
    for (int64_t i = 0; i < m; i++) M[i * m + (m - 1)] -= (c[i] / c[n - 1]) * (scl[i] / scl[m - 1]) * ((double)m / (2.0 * m - 1.0));
    free(scl);
}
static const pbasis LEG = {NB_LEG, "leg", leg_mulx, leg_mul, leg_der, leg_integ, leg_line, leg_vander, leg_companion};

/* leg2poly / poly2leg (power-basis Horner with the Legendre recurrence). */
static int r_leg2poly(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (nc < 3) { double *o = npoly_out1(r, nc); if (!o) rc = TSR_ENOMEM; else for (int64_t i = 0; i < nc; i++) o[i] = c[i]; }
        else { int64_t cap = nc + 1; double *c0 = calloc(cap, sizeof(double)), *c1 = calloc(cap, sizeof(double)), *tmp = calloc(cap, sizeof(double)), *mx = calloc(cap, sizeof(double));
            if (!c0 || !c1 || !tmp || !mx) rc = TSR_ENOMEM;
            else { int64_t l0 = 1, l1 = 1; c0[0] = c[nc - 2]; c1[0] = c[nc - 1];
                for (int64_t i = nc - 1; i >= 2; i--) {
                    int64_t lt = l0; for (int64_t k = 0; k < l0; k++) tmp[k] = c0[k];
                    double f = (i - 1.0) / (double)i; int64_t ns = l1;
                    for (int64_t k = 0; k < ns; k++) c0[k] = (k == 0 ? c[i - 2] : 0.0) - c1[k] * f;
                    l0 = ns > 1 ? ns : 1;
                    double g = (2.0 * i - 1.0) / (double)i;
                    int64_t lmx = (l1 == 1 && c1[0] == 0.0) ? 1 : l1 + 1;
                    if (lmx == 1) mx[0] = 0.0; else { mx[0] = 0.0; for (int64_t k = 0; k < l1; k++) mx[k + 1] = c1[k] * g; }
                    int64_t ln = lt > lmx ? lt : lmx;
                    for (int64_t k = 0; k < ln; k++) c1[k] = (k < lt ? tmp[k] : 0.0) + (k < lmx ? mx[k] : 0.0);
                    l1 = ln;
                }
                int64_t lmx = (l1 == 1 && c1[0] == 0.0) ? 1 : l1 + 1;
                if (lmx == 1) mx[0] = 0.0; else { mx[0] = 0.0; for (int64_t k = 0; k < l1; k++) mx[k + 1] = c1[k]; }
                int64_t lr = l0 > lmx ? l0 : lmx; double *o = npoly_out1(r, lr);
                if (!o) rc = TSR_ENOMEM; else for (int64_t k = 0; k < lr; k++) o[k] = (k < l0 ? c0[k] : 0.0) + (k < lmx ? mx[k] : 0.0);
            }
            free(c0); free(c1); free(tmp); free(mx); }
    }
    fn_free_doubles(c, nc);
    return rc;
}
static int r_poly2leg(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *pol = fn_arg_doubles(&a[0], &nc);
    int rc = pol ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { double *res = calloc(nc + 1, sizeof(double)), *mx = calloc(nc + 1, sizeof(double));
        if (!res || !mx) rc = TSR_ENOMEM;
        else { int64_t lr = 1; res[0] = 0.0;
            for (int64_t i = nc - 1; i >= 0; i--) { int64_t lm = leg_mulx(res, lr, mx); for (int64_t k = 0; k < lm; k++) res[k] = mx[k]; res[0] += pol[i]; lr = lm; }
            double *o = npoly_out1(r, lr); if (!o) rc = TSR_ENOMEM; else for (int64_t k = 0; k < lr; k++) o[k] = res[k]; }
        free(res); free(mx); }
    fn_free_doubles(pol, nc);
    return rc;
}
/* leggauss: nodes = eigenvalues of the symmetric companion of P_deg, one Newton step, symmetrized weights. */
static int r_leggauss(const void *x, const tsr_arg *a, int nn, tsr_result *r, int nr)
{
    (void)x; (void)nr; int64_t m = npoly_opt_int(a, nn, 0, 0);
    if (m < 1) { fn_set_error("npoly.leggauss: deg must be >= 1"); return TSR_EARG; }
    double *xo = npoly_out1(&r[0], m), *wo = xo ? npoly_out1(&r[1], m) : NULL;
    if (!xo || !wo) return TSR_ENOMEM;
    double *c = calloc(m + 1, sizeof(double)); if (c) c[m] = 1.0;
    double *cder = calloc(m > 0 ? m : 1, sizeof(double)); if (c && cder) leg_der(c, m + 1, 1.0, cder);
    double *comp = calloc(m * m, sizeof(double)), *ev = malloc(m * sizeof(double)), *df = malloc(m * sizeof(double)), *fm = malloc(m * sizeof(double)), *w = malloc(m * sizeof(double));
    int rc = (c && cder && comp && ev && df && fm && w) ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (m == 1) ev[0] = 0.0;
        else { leg_companion(c, m + 1, comp);
            if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'L', (lapack_int)m, comp, (lapack_int)m, ev) != 0) { fn_set_error("npoly.leggauss: eigenvalue solve failed"); rc = TSR_ECONVERGE; } }
    }
    if (rc == TSR_OK) {
        double fmax = 0.0, dmax = 0.0;
        for (int64_t k = 0; k < m; k++) { double dy = npoly_eval(NB_LEG, c, m + 1, ev[k]); df[k] = npoly_eval(NB_LEG, cder, m, ev[k]); ev[k] -= dy / df[k]; }
        for (int64_t k = 0; k < m; k++) { fm[k] = npoly_eval(NB_LEG, c + 1, m, ev[k]); if (fabs(fm[k]) > fmax) fmax = fabs(fm[k]); if (fabs(df[k]) > dmax) dmax = fabs(df[k]); }
        double wsum = 0.0;
        for (int64_t k = 0; k < m; k++) w[k] = 1.0 / ((fm[k] / fmax) * (df[k] / dmax));
        for (int64_t k = 0; k < m; k++) { double ws = (w[k] + w[m - 1 - k]) / 2.0; wo[k] = ws; wsum += ws; xo[k] = (ev[k] - ev[m - 1 - k]) / 2.0; }
        for (int64_t k = 0; k < m; k++) wo[k] *= 2.0 / wsum;
    }
    free(c); free(cder); free(comp); free(ev); free(df); free(fm); free(w);
    return rc;
}
static int r_legweight(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly.legweight: x must be an array"); return TSR_EARG; }
    int64_t lx; double *xs = fn_arg_doubles(&a[0], &lx);
    const tsr_array *xa = &a[0].arr; double *out = xs ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) out[i] = 1.0;
    fn_free_doubles(xs, lx);
    return rc;
}
static int r_legzero(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {0}; return npoly_iconst(r, v, 1); }
static int r_legone(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)  { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {1}; return npoly_iconst(r, v, 1); }
static int r_legx(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)    { (void)x; (void)a; (void)n; (void)nr; int64_t v[2] = {0, 1}; return npoly_iconst(r, v, 2); }
static int r_legdomain(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; double *o = npoly_out1(r, 2); if (!o) return TSR_ENOMEM; o[0] = -1.0; o[1] = 1.0; return TSR_OK; }

/* ---- laguerre primitives (domain [0, 1], weight exp(-x)) ------------------------------------------------- */
static int64_t lag_mulx(const double *c, int64_t n, double *out)
{
    if (n == 1 && c[0] == 0.0) { out[0] = 0.0; return 1; }
    for (int64_t i = 0; i <= n; i++) out[i] = 0.0;
    out[0] = c[0]; out[1] = -c[0];
    for (int64_t i = 1; i < n; i++) { out[i + 1] = -c[i] * (i + 1.0); out[i] += c[i] * (2.0 * i + 1.0); out[i - 1] -= c[i] * (double)i; }
    return n + 1;
}
/* NumPy lagmul: Clenshaw product; the c1 update and the final combine both carry a bare-c1 term. */
static int64_t lag_mul(const double *A, int64_t nA, const double *Bv, int64_t nB, double *out)
{
    const double *c, *xs; int64_t nc, nxs;
    if (nA > nB) { c = Bv; nc = nB; xs = A; nxs = nA; } else { c = A; nc = nA; xs = Bv; nxs = nB; }
    int64_t cap = nA + nB + 2;
    double *c0 = calloc(cap, sizeof(double)), *c1 = calloc(cap, sizeof(double)), *tmp = calloc(cap, sizeof(double)), *mx = calloc(cap, sizeof(double)), *nc1 = calloc(cap, sizeof(double));
    int64_t ret = -1;
    if (c0 && c1 && tmp && mx && nc1) {
        int64_t l0, l1;
        if (nc == 1) { for (int64_t k = 0; k < nxs; k++) c0[k] = c[0] * xs[k]; l0 = nxs; c1[0] = 0.0; l1 = 1; }
        else {
            for (int64_t k = 0; k < nxs; k++) { c0[k] = c[nc - 2] * xs[k]; c1[k] = c[nc - 1] * xs[k]; } l0 = nxs; l1 = nxs;
            double nd = (double)nc;
            for (int64_t i = 3; i <= nc; i++) {
                int64_t lt = l0; for (int64_t k = 0; k < l0; k++) tmp[k] = c0[k];
                nd -= 1.0;
                double f = (nd - 1.0) / nd, sc = c[nc - i]; int64_t nl0 = nxs > l1 ? nxs : l1;
                for (int64_t k = 0; k < nl0; k++) c0[k] = (k < nxs ? sc * xs[k] : 0.0) - (k < l1 ? f * c1[k] : 0.0);
                l0 = nl0;
                int64_t lm = lag_mulx(c1, l1, mx); double g = 2.0 * nd - 1.0;      /* c1 = tmp + ((2nd-1)*c1 - mulx(c1))/nd */
                int64_t nl1 = lt > l1 ? lt : l1; if (lm > nl1) nl1 = lm;
                for (int64_t k = 0; k < nl1; k++) nc1[k] = (k < lt ? tmp[k] : 0.0) + ((k < l1 ? g * c1[k] : 0.0) - (k < lm ? mx[k] : 0.0)) / nd;
                for (int64_t k = 0; k < nl1; k++) c1[k] = nc1[k];
                l1 = nl1;
            }
        }
        int64_t lm = lag_mulx(c1, l1, mx);                                          /* out = c0 + (c1 - mulx(c1)) */
        int64_t nr = l0 > l1 ? l0 : l1; if (lm > nr) nr = lm;
        for (int64_t k = 0; k < nr; k++) out[k] = (k < l0 ? c0[k] : 0.0) + (k < l1 ? c1[k] : 0.0) - (k < lm ? mx[k] : 0.0);
        ret = nr;
    }
    free(c0); free(c1); free(tmp); free(mx); free(nc1);
    return ret;
}
static int64_t lag_der(const double *c, int64_t n, double scl, double *out)
{
    if (n == 1) { out[0] = 0.0; return 1; }
    double *cc = calloc(n, sizeof(double)); if (!cc) return -1;
    for (int64_t i = 0; i < n; i++) cc[i] = c[i] * scl;
    int64_t nn = n - 1;
    for (int64_t i = 0; i < nn; i++) out[i] = 0.0;
    for (int64_t j = nn; j >= 2; j--) { out[j - 1] = -cc[j]; cc[j - 1] += cc[j]; }
    out[0] = -cc[1];
    free(cc);
    return nn;
}
static int64_t lag_integ(const double *c, int64_t n, double scl, double *out)
{
    double *cc = calloc(n, sizeof(double)); if (!cc) return -1;
    for (int64_t i = 0; i < n; i++) cc[i] = c[i] * scl;
    for (int64_t i = 0; i <= n; i++) out[i] = 0.0;
    out[0] = cc[0]; out[1] = -cc[0];
    for (int64_t j = 1; j < n; j++) { out[j] += cc[j]; out[j + 1] = -cc[j]; }
    out[0] += -npoly_eval(NB_LAG, out, n + 1, 0.0);
    free(cc);
    return n + 1;
}
static int64_t lag_line(double off, double scl, double *out) { if (scl != 0.0) { out[0] = off + scl; out[1] = -scl; return 2; } out[0] = off; return 1; }
static void lag_vander(const double *x, int64_t nx, int64_t deg, double *V)
{
    int64_t w = deg + 1;
    for (int64_t i = 0; i < nx; i++) { double *rr = V + i * w; rr[0] = 1.0;
        if (deg > 0) { rr[1] = 1.0 - x[i]; for (int64_t j = 2; j <= deg; j++) rr[j] = (rr[j - 1] * (2.0 * j - 1.0 - x[i]) - rr[j - 2] * (j - 1.0)) / (double)j; } }
}
static void lag_companion(const double *c, int64_t n, double *M)
{
    int64_t m = n - 1;
    if (m == 1) { M[0] = 1.0 + c[0] / c[1]; return; }
    for (int64_t i = 0; i < m * m; i++) M[i] = 0.0;
    for (int64_t i = 0; i < m; i++) M[i * m + i] = 2.0 * i + 1.0;
    for (int64_t i = 0; i < m - 1; i++) { double v = -(i + 1.0); M[i * m + (i + 1)] = v; M[(i + 1) * m + i] = v; }
    for (int64_t i = 0; i < m; i++) M[i * m + (m - 1)] += (c[i] / c[n - 1]) * (double)m;
}
static const pbasis LAG = {NB_LAG, "lag", lag_mulx, lag_mul, lag_der, lag_integ, lag_line, lag_vander, lag_companion};

static int r_lag2poly(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *c = fn_arg_doubles(&a[0], &nc);
    int rc = c ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        if (nc == 1) { double *o = npoly_out1(r, 1); if (!o) rc = TSR_ENOMEM; else o[0] = c[0]; }
        else { int64_t cap = nc + 1; double *c0 = calloc(cap, sizeof(double)), *c1 = calloc(cap, sizeof(double)), *tmp = calloc(cap, sizeof(double)), *mx = calloc(cap, sizeof(double));
            if (!c0 || !c1 || !tmp || !mx) rc = TSR_ENOMEM;
            else { int64_t l0 = 1, l1 = 1; c0[0] = c[nc - 2]; c1[0] = c[nc - 1];
                for (int64_t i = nc - 1; i >= 2; i--) {
                    int64_t lt = l0; for (int64_t k = 0; k < l0; k++) tmp[k] = c0[k];
                    double f = (i - 1.0) / (double)i; int64_t ns = l1;                                 /* c0 = polysub([c[i-2]], c1*f) */
                    for (int64_t k = 0; k < ns; k++) c0[k] = (k == 0 ? c[i - 2] : 0.0) - c1[k] * f;
                    l0 = ns > 1 ? ns : 1;
                    /* c1 = polyadd(tmp, polysub((2i-1)*c1, polymulx(c1))/i) */
                    int64_t lmx = (l1 == 1 && c1[0] == 0.0) ? 1 : l1 + 1;
                    if (lmx == 1) mx[0] = 0.0; else { mx[0] = 0.0; for (int64_t k = 0; k < l1; k++) mx[k + 1] = c1[k]; }
                    double g = 2.0 * i - 1.0; int64_t ld = l1 > lmx ? l1 : lmx;                         /* (2i-1)*c1 - mulx(c1) */
                    int64_t ln = lt > ld ? lt : ld;
                    for (int64_t k = 0; k < ln; k++) c1[k] = (k < lt ? tmp[k] : 0.0) + ((k < l1 ? g * c1[k] : 0.0) - (k < lmx ? mx[k] : 0.0)) / (double)i;
                    l1 = ln;
                }
                int64_t lmx = (l1 == 1 && c1[0] == 0.0) ? 1 : l1 + 1;                                   /* out = polyadd(c0, polysub(c1, polymulx(c1))) */
                if (lmx == 1) mx[0] = 0.0; else { mx[0] = 0.0; for (int64_t k = 0; k < l1; k++) mx[k + 1] = c1[k]; }
                int64_t ld = l1 > lmx ? l1 : lmx; int64_t lr = l0 > ld ? l0 : ld; double *o = npoly_out1(r, lr);
                if (!o) rc = TSR_ENOMEM; else for (int64_t k = 0; k < lr; k++) o[k] = (k < l0 ? c0[k] : 0.0) + (k < l1 ? c1[k] : 0.0) - (k < lmx ? mx[k] : 0.0);
            }
            free(c0); free(c1); free(tmp); free(mx); }
    }
    fn_free_doubles(c, nc);
    return rc;
}
static int r_poly2lag(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly: coefficients must be an array"); return TSR_EARG; }
    int64_t nc; double *pol = fn_arg_doubles(&a[0], &nc);
    int rc = pol ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { double *res = calloc(nc + 1, sizeof(double)), *mx = calloc(nc + 1, sizeof(double));
        if (!res || !mx) rc = TSR_ENOMEM;
        else { int64_t lr = 1; res[0] = 0.0;
            for (int64_t i = nc - 1; i >= 0; i--) { int64_t lm = lag_mulx(res, lr, mx); for (int64_t k = 0; k < lm; k++) res[k] = mx[k]; res[0] += pol[i]; lr = lm; }
            double *o = npoly_out1(r, lr); if (!o) rc = TSR_ENOMEM; else for (int64_t k = 0; k < lr; k++) o[k] = res[k]; }
        free(res); free(mx); }
    fn_free_doubles(pol, nc);
    return rc;
}
/* laggauss: eigenvalues of the symmetric companion, one Newton step, weights normalized to sum 1. */
static int r_laggauss(const void *x, const tsr_arg *a, int nn, tsr_result *r, int nr)
{
    (void)x; (void)nr; int64_t m = npoly_opt_int(a, nn, 0, 0);
    if (m < 1) { fn_set_error("npoly.laggauss: deg must be >= 1"); return TSR_EARG; }
    double *xo = npoly_out1(&r[0], m), *wo = xo ? npoly_out1(&r[1], m) : NULL;
    if (!xo || !wo) return TSR_ENOMEM;
    double *c = calloc(m + 1, sizeof(double)); if (c) c[m] = 1.0;
    double *cder = calloc(m > 0 ? m : 1, sizeof(double)); if (c && cder) lag_der(c, m + 1, 1.0, cder);
    double *comp = calloc(m * m, sizeof(double)), *ev = malloc(m * sizeof(double)), *df = malloc(m * sizeof(double)), *fm = malloc(m * sizeof(double));
    int rc = (c && cder && comp && ev && df && fm) ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { lag_companion(c, m + 1, comp);
        if (m == 1) ev[0] = comp[0];
        else if (LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'L', (lapack_int)m, comp, (lapack_int)m, ev) != 0) { fn_set_error("npoly.laggauss: eigenvalue solve failed"); rc = TSR_ECONVERGE; } }
    if (rc == TSR_OK) {
        double fmax = 0.0, dmax = 0.0, wsum = 0.0;
        for (int64_t k = 0; k < m; k++) { double dy = npoly_eval(NB_LAG, c, m + 1, ev[k]); df[k] = npoly_eval(NB_LAG, cder, m, ev[k]); ev[k] -= dy / df[k]; }
        for (int64_t k = 0; k < m; k++) { fm[k] = npoly_eval(NB_LAG, c + 1, m, ev[k]); if (fabs(fm[k]) > fmax) fmax = fabs(fm[k]); if (fabs(df[k]) > dmax) dmax = fabs(df[k]); }
        for (int64_t k = 0; k < m; k++) { wo[k] = 1.0 / ((fm[k] / fmax) * (df[k] / dmax)); wsum += wo[k]; xo[k] = ev[k]; }
        for (int64_t k = 0; k < m; k++) wo[k] /= wsum;
    }
    free(c); free(cder); free(comp); free(ev); free(df); free(fm);
    return rc;
}
static int r_lagweight(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)
{
    (void)x; (void)n; (void)nr;
    if (a[0].kind != 3) { fn_set_error("npoly.lagweight: x must be an array"); return TSR_EARG; }
    int64_t lx; double *xs = fn_arg_doubles(&a[0], &lx);
    const tsr_array *xa = &a[0].arr; double *out = xs ? (double *)fn_result_array(r, TSR_F64, xa->ndim, xa->shape) : NULL;
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t i = 0; i < lx; i++) out[i] = exp(-xs[i]);
    fn_free_doubles(xs, lx);
    return rc;
}
static int r_lagzero(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {0}; return npoly_iconst(r, v, 1); }
static int r_lagone(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)  { (void)x; (void)a; (void)n; (void)nr; int64_t v[1] = {1}; return npoly_iconst(r, v, 1); }
static int r_lagx(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr)    { (void)x; (void)a; (void)n; (void)nr; int64_t v[2] = {1, -1}; return npoly_iconst(r, v, 2); }
static int r_lagdomain(const void *x, const tsr_arg *a, int n, tsr_result *r, int nr) { (void)x; (void)a; (void)n; (void)nr; double *o = npoly_out1(r, 2); if (!o) return TSR_ENOMEM; o[0] = 0.0; o[1] = 1.0; return TSR_OK; }

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

    /* numpy.polynomial.chebyshev (Chebyshev T basis) */
    ROUTINE("npoly.chebadd", 1, "c1, c2", "out", rb_add, NULL, "Sum of two Chebyshev series (numpy.polynomial.chebyshev.chebadd)."),
    ROUTINE("npoly.chebsub", 1, "c1, c2", "out", rb_sub, NULL, "Difference of two Chebyshev series (numpy.polynomial.chebyshev.chebsub)."),
    ROUTINE("npoly.chebmul", 1, "c1, c2", "out", rb_mul, &CHEB, "Product of two Chebyshev series (numpy.polynomial.chebyshev.chebmul)."),
    ROUTINE("npoly.chebmulx", 1, "c", "out", rb_mulx, &CHEB, "Multiply a Chebyshev series by x (numpy.polynomial.chebyshev.chebmulx)."),
    ROUTINE("npoly.chebpow", 1, "c, pow, maxpower=16", "out", rb_pow, &CHEB, "Chebyshev series raised to a power (numpy.polynomial.chebyshev.chebpow)."),
    ROUTINE("npoly.chebdiv", 2, "c1, c2", "quo, rem", rb_div, &CHEB, "Quotient and remainder of Chebyshev-series division (numpy.polynomial.chebyshev.chebdiv)."),
    ROUTINE("npoly.chebder", 1, "c, m=1, scl=1", "out", rb_der, &CHEB, "Derivative of a Chebyshev series (numpy.polynomial.chebyshev.chebder)."),
    ROUTINE("npoly.chebint", 1, "c, m=1, k=0, lbnd=0, scl=1", "out", rb_int, &CHEB, "Antiderivative of a Chebyshev series (numpy.polynomial.chebyshev.chebint)."),
    ROUTINE("npoly.chebfromroots", 1, "roots", "out", rb_fromroots, &CHEB, "Chebyshev series with the given roots (numpy.polynomial.chebyshev.chebfromroots)."),
    ROUTINE("npoly.chebline", 1, "off, scl", "out", rb_line, &CHEB, "Chebyshev series for off + scl*x (numpy.polynomial.chebyshev.chebline)."),
    ROUTINE("npoly.chebtrim", 1, "c, tol=0", "out", rb_trim, NULL, "Trim trailing small coefficients (numpy.polynomial.chebyshev.chebtrim)."),
    ROUTINE("npoly.chebvander", 1, "x, deg", "out", rb_vander, &CHEB, "Pseudo-Vandermonde matrix of the Chebyshev basis (numpy.polynomial.chebyshev.chebvander)."),
    ROUTINE("npoly.chebval2d", 1, "x, y, c", "out", rb_val2d, &CHEB, "Evaluate a 2-D Chebyshev series (numpy.polynomial.chebyshev.chebval2d)."),
    ROUTINE("npoly.chebval3d", 1, "x, y, z, c", "out", rb_val3d, &CHEB, "Evaluate a 3-D Chebyshev series (numpy.polynomial.chebyshev.chebval3d)."),
    ROUTINE("npoly.chebgrid2d", 1, "x, y, c", "out", rb_grid2d, &CHEB, "Evaluate a Chebyshev series on a 2-D grid (numpy.polynomial.chebyshev.chebgrid2d)."),
    ROUTINE("npoly.chebgrid3d", 1, "x, y, z, c", "out", rb_grid3d, &CHEB, "Evaluate a Chebyshev series on a 3-D grid (numpy.polynomial.chebyshev.chebgrid3d)."),
    ROUTINE("npoly.chebvander2d", 1, "x, y, deg", "out", rb_vander2d, &CHEB, "Pseudo-Vandermonde matrix of a 2-D Chebyshev basis (numpy.polynomial.chebyshev.chebvander2d)."),
    ROUTINE("npoly.chebvander3d", 1, "x, y, z, deg", "out", rb_vander3d, &CHEB, "Pseudo-Vandermonde matrix of a 3-D Chebyshev basis (numpy.polynomial.chebyshev.chebvander3d)."),
    ROUTINE("npoly.chebcompanion", 1, "c", "out", rb_companion, &CHEB, "Companion matrix of a Chebyshev series (numpy.polynomial.chebyshev.chebcompanion)."),
    ROUTINE("npoly.chebroots", 1, "c", "out", rb_roots, &CHEB, "Roots of a Chebyshev series (numpy.polynomial.chebyshev.chebroots)."),
    ROUTINE("npoly.chebfit", 1, "x, y, deg", "out", rb_fit, &CHEB, "Least-squares Chebyshev-series fit (numpy.polynomial.chebyshev.chebfit)."),
    ROUTINE("npoly.cheb2poly", 1, "c", "out", r_cheb2poly, NULL, "Convert a Chebyshev series to a power series (numpy.polynomial.chebyshev.cheb2poly)."),
    ROUTINE("npoly.poly2cheb", 1, "pol", "out", r_poly2cheb, NULL, "Convert a power series to a Chebyshev series (numpy.polynomial.chebyshev.poly2cheb)."),
    ROUTINE("npoly.chebpts1", 1, "npts", "out", r_chebpts1, NULL, "Chebyshev points of the first kind (numpy.polynomial.chebyshev.chebpts1)."),
    ROUTINE("npoly.chebpts2", 1, "npts", "out", r_chebpts2, NULL, "Chebyshev points of the second kind (numpy.polynomial.chebyshev.chebpts2)."),
    ROUTINE("npoly.chebgauss", 2, "deg", "x, w", r_chebgauss, NULL, "Gauss-Chebyshev quadrature nodes and weights (numpy.polynomial.chebyshev.chebgauss)."),
    ROUTINE("npoly.chebweight", 1, "x", "out", r_chebweight, NULL, "Chebyshev weight 1/sqrt(1-x^2) (numpy.polynomial.chebyshev.chebweight)."),
    ROUTINE("npoly.chebzero", 1, "", "out", r_chebzero, NULL, "The zero Chebyshev series (numpy.polynomial.chebyshev.chebzero)."),
    ROUTINE("npoly.chebone", 1, "", "out", r_chebone, NULL, "The one Chebyshev series (numpy.polynomial.chebyshev.chebone)."),
    ROUTINE("npoly.chebx", 1, "", "out", r_chebx, NULL, "The identity Chebyshev series x (numpy.polynomial.chebyshev.chebx)."),
    ROUTINE("npoly.chebdomain", 1, "", "out", r_chebdomain, NULL, "The default Chebyshev domain [-1, 1] (numpy.polynomial.chebyshev.chebdomain)."),

    /* numpy.polynomial.legendre */
    ROUTINE("npoly.legadd", 1, "c1, c2", "out", rb_add, NULL, "Sum of two Legendre series (numpy.polynomial.legendre.legadd)."),
    ROUTINE("npoly.legsub", 1, "c1, c2", "out", rb_sub, NULL, "Difference of two Legendre series (numpy.polynomial.legendre.legsub)."),
    ROUTINE("npoly.legmul", 1, "c1, c2", "out", rb_mul, &LEG, "Product of two Legendre series (numpy.polynomial.legendre.legmul)."),
    ROUTINE("npoly.legmulx", 1, "c", "out", rb_mulx, &LEG, "Multiply a Legendre series by x (numpy.polynomial.legendre.legmulx)."),
    ROUTINE("npoly.legpow", 1, "c, pow, maxpower=16", "out", rb_pow, &LEG, "Legendre series raised to a power (numpy.polynomial.legendre.legpow)."),
    ROUTINE("npoly.legdiv", 2, "c1, c2", "quo, rem", rb_div, &LEG, "Quotient and remainder of Legendre-series division (numpy.polynomial.legendre.legdiv)."),
    ROUTINE("npoly.legder", 1, "c, m=1, scl=1", "out", rb_der, &LEG, "Derivative of a Legendre series (numpy.polynomial.legendre.legder)."),
    ROUTINE("npoly.legint", 1, "c, m=1, k=0, lbnd=0, scl=1", "out", rb_int, &LEG, "Antiderivative of a Legendre series (numpy.polynomial.legendre.legint)."),
    ROUTINE("npoly.legfromroots", 1, "roots", "out", rb_fromroots, &LEG, "Legendre series with the given roots (numpy.polynomial.legendre.legfromroots)."),
    ROUTINE("npoly.legline", 1, "off, scl", "out", rb_line, &LEG, "Legendre series for off + scl*x (numpy.polynomial.legendre.legline)."),
    ROUTINE("npoly.legtrim", 1, "c, tol=0", "out", rb_trim, NULL, "Trim trailing small coefficients (numpy.polynomial.legendre.legtrim)."),
    ROUTINE("npoly.legvander", 1, "x, deg", "out", rb_vander, &LEG, "Pseudo-Vandermonde matrix of the Legendre basis (numpy.polynomial.legendre.legvander)."),
    ROUTINE("npoly.legval2d", 1, "x, y, c", "out", rb_val2d, &LEG, "Evaluate a 2-D Legendre series (numpy.polynomial.legendre.legval2d)."),
    ROUTINE("npoly.legval3d", 1, "x, y, z, c", "out", rb_val3d, &LEG, "Evaluate a 3-D Legendre series (numpy.polynomial.legendre.legval3d)."),
    ROUTINE("npoly.leggrid2d", 1, "x, y, c", "out", rb_grid2d, &LEG, "Evaluate a Legendre series on a 2-D grid (numpy.polynomial.legendre.leggrid2d)."),
    ROUTINE("npoly.leggrid3d", 1, "x, y, z, c", "out", rb_grid3d, &LEG, "Evaluate a Legendre series on a 3-D grid (numpy.polynomial.legendre.leggrid3d)."),
    ROUTINE("npoly.legvander2d", 1, "x, y, deg", "out", rb_vander2d, &LEG, "Pseudo-Vandermonde matrix of a 2-D Legendre basis (numpy.polynomial.legendre.legvander2d)."),
    ROUTINE("npoly.legvander3d", 1, "x, y, z, deg", "out", rb_vander3d, &LEG, "Pseudo-Vandermonde matrix of a 3-D Legendre basis (numpy.polynomial.legendre.legvander3d)."),
    ROUTINE("npoly.legcompanion", 1, "c", "out", rb_companion, &LEG, "Companion matrix of a Legendre series (numpy.polynomial.legendre.legcompanion)."),
    ROUTINE("npoly.legroots", 1, "c", "out", rb_roots, &LEG, "Roots of a Legendre series (numpy.polynomial.legendre.legroots)."),
    ROUTINE("npoly.legfit", 1, "x, y, deg", "out", rb_fit, &LEG, "Least-squares Legendre-series fit (numpy.polynomial.legendre.legfit)."),
    ROUTINE("npoly.leg2poly", 1, "c", "out", r_leg2poly, NULL, "Convert a Legendre series to a power series (numpy.polynomial.legendre.leg2poly)."),
    ROUTINE("npoly.poly2leg", 1, "pol", "out", r_poly2leg, NULL, "Convert a power series to a Legendre series (numpy.polynomial.legendre.poly2leg)."),
    ROUTINE("npoly.leggauss", 2, "deg", "x, w", r_leggauss, NULL, "Gauss-Legendre quadrature nodes and weights (numpy.polynomial.legendre.leggauss)."),
    ROUTINE("npoly.legweight", 1, "x", "out", r_legweight, NULL, "Legendre weight, identically 1 (numpy.polynomial.legendre.legweight)."),
    ROUTINE("npoly.legzero", 1, "", "out", r_legzero, NULL, "The zero Legendre series (numpy.polynomial.legendre.legzero)."),
    ROUTINE("npoly.legone", 1, "", "out", r_legone, NULL, "The one Legendre series (numpy.polynomial.legendre.legone)."),
    ROUTINE("npoly.legx", 1, "", "out", r_legx, NULL, "The identity Legendre series x (numpy.polynomial.legendre.legx)."),
    ROUTINE("npoly.legdomain", 1, "", "out", r_legdomain, NULL, "The default Legendre domain [-1, 1] (numpy.polynomial.legendre.legdomain)."),

    /* numpy.polynomial.laguerre */
    ROUTINE("npoly.lagadd", 1, "c1, c2", "out", rb_add, NULL, "Sum of two Laguerre series (numpy.polynomial.laguerre.lagadd)."),
    ROUTINE("npoly.lagsub", 1, "c1, c2", "out", rb_sub, NULL, "Difference of two Laguerre series (numpy.polynomial.laguerre.lagsub)."),
    ROUTINE("npoly.lagmul", 1, "c1, c2", "out", rb_mul, &LAG, "Product of two Laguerre series (numpy.polynomial.laguerre.lagmul)."),
    ROUTINE("npoly.lagmulx", 1, "c", "out", rb_mulx, &LAG, "Multiply a Laguerre series by x (numpy.polynomial.laguerre.lagmulx)."),
    ROUTINE("npoly.lagpow", 1, "c, pow, maxpower=16", "out", rb_pow, &LAG, "Laguerre series raised to a power (numpy.polynomial.laguerre.lagpow)."),
    ROUTINE("npoly.lagdiv", 2, "c1, c2", "quo, rem", rb_div, &LAG, "Quotient and remainder of Laguerre-series division (numpy.polynomial.laguerre.lagdiv)."),
    ROUTINE("npoly.lagder", 1, "c, m=1, scl=1", "out", rb_der, &LAG, "Derivative of a Laguerre series (numpy.polynomial.laguerre.lagder)."),
    ROUTINE("npoly.lagint", 1, "c, m=1, k=0, lbnd=0, scl=1", "out", rb_int, &LAG, "Antiderivative of a Laguerre series (numpy.polynomial.laguerre.lagint)."),
    ROUTINE("npoly.lagfromroots", 1, "roots", "out", rb_fromroots, &LAG, "Laguerre series with the given roots (numpy.polynomial.laguerre.lagfromroots)."),
    ROUTINE("npoly.lagline", 1, "off, scl", "out", rb_line, &LAG, "Laguerre series for off + scl*x (numpy.polynomial.laguerre.lagline)."),
    ROUTINE("npoly.lagtrim", 1, "c, tol=0", "out", rb_trim, NULL, "Trim trailing small coefficients (numpy.polynomial.laguerre.lagtrim)."),
    ROUTINE("npoly.lagvander", 1, "x, deg", "out", rb_vander, &LAG, "Pseudo-Vandermonde matrix of the Laguerre basis (numpy.polynomial.laguerre.lagvander)."),
    ROUTINE("npoly.lagval2d", 1, "x, y, c", "out", rb_val2d, &LAG, "Evaluate a 2-D Laguerre series (numpy.polynomial.laguerre.lagval2d)."),
    ROUTINE("npoly.lagval3d", 1, "x, y, z, c", "out", rb_val3d, &LAG, "Evaluate a 3-D Laguerre series (numpy.polynomial.laguerre.lagval3d)."),
    ROUTINE("npoly.laggrid2d", 1, "x, y, c", "out", rb_grid2d, &LAG, "Evaluate a Laguerre series on a 2-D grid (numpy.polynomial.laguerre.laggrid2d)."),
    ROUTINE("npoly.laggrid3d", 1, "x, y, z, c", "out", rb_grid3d, &LAG, "Evaluate a Laguerre series on a 3-D grid (numpy.polynomial.laguerre.laggrid3d)."),
    ROUTINE("npoly.lagvander2d", 1, "x, y, deg", "out", rb_vander2d, &LAG, "Pseudo-Vandermonde matrix of a 2-D Laguerre basis (numpy.polynomial.laguerre.lagvander2d)."),
    ROUTINE("npoly.lagvander3d", 1, "x, y, z, deg", "out", rb_vander3d, &LAG, "Pseudo-Vandermonde matrix of a 3-D Laguerre basis (numpy.polynomial.laguerre.lagvander3d)."),
    ROUTINE("npoly.lagcompanion", 1, "c", "out", rb_companion, &LAG, "Companion matrix of a Laguerre series (numpy.polynomial.laguerre.lagcompanion)."),
    ROUTINE("npoly.lagroots", 1, "c", "out", rb_roots, &LAG, "Roots of a Laguerre series (numpy.polynomial.laguerre.lagroots)."),
    ROUTINE("npoly.lagfit", 1, "x, y, deg", "out", rb_fit, &LAG, "Least-squares Laguerre-series fit (numpy.polynomial.laguerre.lagfit)."),
    ROUTINE("npoly.lag2poly", 1, "c", "out", r_lag2poly, NULL, "Convert a Laguerre series to a power series (numpy.polynomial.laguerre.lag2poly)."),
    ROUTINE("npoly.poly2lag", 1, "pol", "out", r_poly2lag, NULL, "Convert a power series to a Laguerre series (numpy.polynomial.laguerre.poly2lag)."),
    ROUTINE("npoly.laggauss", 2, "deg", "x, w", r_laggauss, NULL, "Gauss-Laguerre quadrature nodes and weights (numpy.polynomial.laguerre.laggauss)."),
    ROUTINE("npoly.lagweight", 1, "x", "out", r_lagweight, NULL, "Laguerre weight exp(-x) (numpy.polynomial.laguerre.lagweight)."),
    ROUTINE("npoly.lagzero", 1, "", "out", r_lagzero, NULL, "The zero Laguerre series (numpy.polynomial.laguerre.lagzero)."),
    ROUTINE("npoly.lagone", 1, "", "out", r_lagone, NULL, "The one Laguerre series (numpy.polynomial.laguerre.lagone)."),
    ROUTINE("npoly.lagx", 1, "", "out", r_lagx, NULL, "The identity Laguerre series x (numpy.polynomial.laguerre.lagx)."),
    ROUTINE("npoly.lagdomain", 1, "", "out", r_lagdomain, NULL, "The default Laguerre domain [0, 1] (numpy.polynomial.laguerre.lagdomain)."),
};

const fn_table TSR_NP_POLYNOMIAL_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
