/* numpy.polynomial ("npoly." prefix -> Tessero\Polynomial facade). Evaluation and calculus in the six standard
 * polynomial bases (power, Chebyshev T, Legendre, Laguerre, physicists' Hermite, probabilists' HermiteE), sharing
 * a single Clenshaw evaluator that replicates numpy's per-basis *val recurrences exactly.
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

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
};

const fn_table TSR_NP_POLYNOMIAL_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
