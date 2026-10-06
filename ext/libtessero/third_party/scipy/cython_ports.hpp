/*
 * C++ ports of the Cython inline functions behind scipy.special ufuncs (SciPy 1.17.1, BSD-3-Clause, see
 * LICENSE.txt): _convex_analysis.pxd, _boxcox.pxd, _legacy.pxd, orthogonal_eval.pxd, _hyp0f1.pxd,
 * _hypergeometric.pxd, _agm.pxd, _factorial.pxd, _cdflib_wrappers.pxd, _ndtri_exp.pxd, and the wrappers of
 * xsf_wrappers.cpp that add logic of their own. The code follows the originals line by line; the only change
 * is that SciPy's Python warnings are dropped (sf_error is a no-op).
 */
#pragma once

#include <cfloat>
#include <cmath>
#include <limits>

#include "xsf/alg.h"
#include "xsf/bessel.h"
#include "xsf/beta.h"
#include "xsf/binom.h"
#include "xsf/cephes/ellpk.h"
#include "xsf/cephes/jv.h"
#include "xsf/cephes/ndtri.h"
#include "xsf/cephes/poch.h"
#include "xsf/cephes/polevl.h"
#include "xsf/gamma.h"
#include "xsf/hyp2f1.h"
#include "xsf/log_exp.h"
#include "xsf/stats.h"
#include "xsf/trig.h"
#include "xsf/specfun.h"

extern "C" {
#include "cdflib.h"
}

namespace scipy_port {

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double Inf = std::numeric_limits<double>::infinity();

/* ---------------------------------------------------------------- _convex_analysis.pxd */

inline double entr(double x)
{
    if (std::isnan(x)) return x;
    if (x > 0) return -x * std::log(x);
    if (x == 0) return 0;
    return -Inf;
}

inline double kl_div(double x, double y)
{
    if (std::isnan(x) || std::isnan(y)) return NaN;
    if (x > 0 && y > 0) return x * std::log(x / y) - x + y;
    if (x == 0 && y >= 0) return y;
    return Inf;
}

inline double rel_entr(double x, double y)
{
    if (std::isnan(x) || std::isnan(y)) return NaN;
    if (x <= 0 || y <= 0) {
        if (x == 0 && y >= 0) return 0;
        return Inf;
    }
    const double ratio = x / y;
    if (0.5 < ratio && ratio < 2) return x * std::log1p((x - y) / y);
    if (DBL_MIN < ratio && ratio < Inf) return x * std::log(ratio);
    return x * (std::log(x) - std::log(y));
}

inline double huber(double delta, double r)
{
    if (delta < 0) return Inf;
    if (std::fabs(r) <= delta) return 0.5 * r * r;
    return delta * (std::fabs(r) - 0.5 * delta);
}

inline double pseudo_huber(double delta, double r)
{
    if (delta < 0) return Inf;
    if (delta == 0 || r == 0) return 0;
    const double u = delta, v = r / delta;
    return u * u * std::expm1(0.5 * std::log1p(v * v));
}

/* ---------------------------------------------------------------- _boxcox.pxd */

inline double boxcox(double x, double lmbda)
{
    if (std::fabs(lmbda) < 1e-19) return std::log(x);
    if (lmbda * std::log(x) < 709.78) return std::expm1(lmbda * std::log(x)) / lmbda;
    return std::copysign(1., lmbda) * std::exp(lmbda * std::log(x) - std::log(std::fabs(lmbda))) - 1 / lmbda;
}

inline double boxcox1p(double x, double lmbda)
{
    const double lgx = std::log1p(x);
    if (std::fabs(lmbda) < 1e-19 || (std::fabs(lgx) < 1e-289 && std::fabs(lmbda) < 1e273)) return lgx;
    if (lmbda * lgx < 709.78) return std::expm1(lmbda * lgx) / lmbda;
    return std::copysign(1., lmbda) * std::exp(lmbda * lgx - std::log(std::fabs(lmbda))) - 1 / lmbda;
}

inline double inv_boxcox(double x, double lmbda)
{
    if (lmbda == 0) return std::exp(x);
    if (lmbda * x < 1.79e308) return std::exp(std::log1p(lmbda * x) / lmbda);
    return std::exp((std::log(std::copysign(1., lmbda) * (x + 1 / lmbda)) + std::log(std::fabs(lmbda))) / lmbda);
}

inline double inv_boxcox1p(double x, double lmbda)
{
    if (lmbda == 0) return std::expm1(x);
    if (std::fabs(lmbda * x) < 1e-154) return x;
    if (lmbda * x < 1.79e308) return std::expm1(std::log1p(lmbda * x) / lmbda);
    return std::expm1((std::log(std::copysign(1., lmbda) * (x + 1 / lmbda)) + std::log(std::fabs(lmbda))) / lmbda);
}

/* ---------------------------------------------------------------- _factorial.pxd, _agm.pxd */

inline double factorial(double n) { return n < 0 ? 0 : xsf::gamma(n + 1); }

inline double agm_iter(double a, double b)
{
    int count = 20;
    double amean = 0.5 * a + 0.5 * b;
    while (count > 0 && (amean != a && amean != b)) {
        const double gmean = std::sqrt(a) * std::sqrt(b);
        a = amean;
        b = gmean;
        amean = 0.5 * a + 0.5 * b;
        count -= 1;
    }
    return amean;
}

inline double agm(double a, double b)
{
    const double sqrthalfmax = 9.480751908109176e+153, invsqrthalfmax = 1.0547686614863e-154;
    if (std::isnan(a) || std::isnan(b)) return NaN;
    if ((a < 0 && b > 0) || (a > 0 && b < 0)) return NaN;
    if ((std::isinf(a) || std::isinf(b)) && (a == 0 || b == 0)) return NaN;
    if (a == 0 || b == 0) return 0.0;
    if (a == b) return a;
    int sgn = 1;
    if (a < 0) { sgn = -1; a = -a; b = -b; }
    if ((invsqrthalfmax < a && a < sqrthalfmax) && (invsqrthalfmax < b && b < sqrthalfmax)) {
        const double e = 4 * a * b / ((a + b) * (a + b));
        return sgn * (M_PI / 4) * (a + b) / xsf::cephes::ellpk(e);
    }
    return sgn * agm_iter(a, b);
}

/* ---------------------------------------------------------------- _hyp0f1.pxd, _hypergeometric.pxd */

inline double hyp0f1_asy(double v, double z)
{
    const double arg = std::sqrt(z);
    const double v1 = std::fabs(v - 1);
    const double x = 2.0 * arg / v1;
    const double p1 = std::sqrt(1.0 + x * x);
    const double eta = p1 + std::log(x) - std::log1p(p1);
    double arg_exp_i = -0.5 * std::log(p1);
    arg_exp_i -= 0.5 * std::log(2.0 * M_PI * v1);
    arg_exp_i += xsf::gammaln(v);
    const double gs = xsf::gammasgn(v);
    double arg_exp_k = arg_exp_i;
    arg_exp_i += v1 * eta;
    arg_exp_k -= v1 * eta;
    const double pp = 1.0 / p1, p2 = pp * pp, p4 = p2 * p2, p6 = p4 * p2;
    const double u1 = (3.0 - 5.0 * p2) * pp / 24.0;
    const double u2 = (81.0 - 462.0 * p2 + 385.0 * p4) * p2 / 1152.0;
    const double u3 = (30375.0 - 369603.0 * p2 + 765765.0 * p4 - 425425.0 * p6) * pp * p2 / 414720.0;
    const double u_corr_i = 1.0 + u1 / v1 + u2 / (v1 * v1) + u3 / (v1 * v1 * v1);
    double result = std::exp(arg_exp_i - xsf::xlogy(v1, arg)) * gs * u_corr_i;
    if (v - 1 < 0) {
        const double u_corr_k = 1.0 - u1 / v1 + u2 / (v1 * v1) - u3 / (v1 * v1 * v1);
        result += std::exp(arg_exp_k + xsf::xlogy(v1, arg)) * gs * 2.0 * xsf::sinpi(v1) * u_corr_k;
    }
    return result;
}

inline double hyp0f1(double v, double z)
{
    if (v <= 0.0 && v == std::floor(v)) return NaN;
    if (z == 0.0 && v != 0.0) return 1.0;
    if (std::fabs(z) < 1e-6 * (1.0 + std::fabs(v))) return 1.0 + z / v + z * z / (2.0 * v * (v + 1.0));
    if (z > 0) {
        const double arg = std::sqrt(z);
        const double arg_exp = xsf::xlogy(1.0 - v, arg) + xsf::gammaln(v);
        const double bess_val = xsf::cyl_bessel_i(v - 1, 2.0 * arg);
        if (arg_exp > std::log(DBL_MAX) || bess_val == 0 || arg_exp < std::log(DBL_MIN) || std::isinf(bess_val))
            return hyp0f1_asy(v, z);
        return std::exp(arg_exp) * xsf::gammasgn(v) * bess_val;
    }
    const double arg = std::sqrt(-z);
    return std::pow(arg, 1.0 - v) * xsf::gamma(v) * xsf::cephes::jv(v - 1, 2 * arg);
}

inline double hyperu(double a, double b, double x)
{
    if (std::isnan(a) || std::isnan(b) || std::isnan(x)) return NaN;
    if (x < 0.0) return NaN;
    if (x == 0.0) {
        if (b > 1.0) return Inf;
        return xsf::cephes::poch(1.0 - b + a, -a);
    }
    if (b == 1 && x < 1 && -0.25 < a && a < 0.3)
        return (x + 1 + 2 * a) * xsf::hypu(a + 1, 1, x) - (a + 1) * (a + 1) * xsf::hypu(a + 2, 1, x);
    return xsf::hypu(a, b, x);
}

/* ---------------------------------------------------------------- _legacy.pxd (double -> int casts) */

inline double bdtr_unsafe(double k, double n, double p) { return (std::isnan(n) || std::isinf(n)) ? NaN : xsf::bdtr(k, (int)n, p); }
inline double bdtrc_unsafe(double k, double n, double p) { return (std::isnan(n) || std::isinf(n)) ? NaN : xsf::bdtrc(k, (int)n, p); }
inline double bdtri_unsafe(double k, double n, double p) { return (std::isnan(n) || std::isinf(n)) ? NaN : xsf::bdtri(k, (int)n, p); }
inline double expn_unsafe(double n, double x) { return std::isnan(n) ? n : xsf::cephes::expn((int)n, x); }
inline double nbdtrc_unsafe(double k, double n, double p) { return (std::isnan(k) || std::isnan(n)) ? NaN : xsf::cephes::nbdtrc((int)k, (int)n, p); }
inline double nbdtr_unsafe(double k, double n, double p) { return (std::isnan(k) || std::isnan(n)) ? NaN : xsf::cephes::nbdtr((int)k, (int)n, p); }
inline double nbdtri_unsafe(double k, double n, double p) { return (std::isnan(k) || std::isnan(n)) ? NaN : xsf::cephes::nbdtri((int)k, (int)n, p); }
inline double pdtri_unsafe(double k, double y) { return std::isnan(k) ? k : xsf::cephes::pdtri((int)k, y); }
inline double kn_unsafe(double n, double x) { return std::isnan(n) ? n : xsf::cyl_bessel_k((double)(int)n, x); }
inline double yn_unsafe(double n, double x) { return std::isnan(n) ? n : xsf::cephes::yn((int)n, x); }
inline double smirnov_unsafe(double n, double e) { return std::isnan(n) ? n : xsf::cephes::smirnov((int)n, e); }
inline double smirnovc_unsafe(double n, double e) { return std::isnan(n) ? n : xsf::cephes::smirnovc((int)n, e); }
inline double smirnovp_unsafe(double n, double e) { return std::isnan(n) ? n : xsf::cephes::smirnovp((int)n, e); }
inline double smirnovi_unsafe(double n, double p) { return std::isnan(n) ? n : xsf::cephes::smirnovi((int)n, p); }
inline double smirnovci_unsafe(double n, double p) { return std::isnan(n) ? n : xsf::cephes::smirnovci((int)n, p); }

/* ---------------------------------------------------------------- orthogonal_eval.pxd (real) */

inline double eval_jacobi(double n, double alpha, double beta, double x)
{
    const double d = xsf::binom(n + alpha, n);
    return d * xsf::hyp2f1(-n, n + alpha + beta + 1, alpha + 1, 0.5 * (1 - x));
}

inline double eval_jacobi_l(long n, double alpha, double beta, double x)
{
    if (n < 0) return eval_jacobi((double)n, alpha, beta, x);
    if (n == 0) return 1.0;
    if (n == 1) return 0.5 * (2 * (alpha + 1) + (alpha + beta + 2) * (x - 1));
    double d = (alpha + beta + 2) * (x - 1) / (2 * (alpha + 1));
    double p = d + 1;
    for (long kk = 0; kk < n - 1; kk++) {
        const double k = kk + 1.0;
        const double t = 2 * k + alpha + beta;
        d = ((t * (t + 1) * (t + 2)) * (x - 1) * p + 2 * k * (k + beta) * (t + 2) * d) / (2 * (k + alpha + 1) * (k + alpha + beta + 1) * t);
        p = d + p;
    }
    return xsf::binom(n + alpha, n) * p;
}

inline double eval_sh_jacobi(double n, double p, double q, double x) { return eval_jacobi(n, p - q, q - 1, 2 * x - 1) / xsf::binom(2 * n + p - 1, n); }
inline double eval_sh_jacobi_l(long n, double p, double q, double x) { return eval_jacobi_l(n, p - q, q - 1, 2 * x - 1) / xsf::binom(2 * n + p - 1, n); }

inline double eval_gegenbauer(double n, double alpha, double x)
{
    const double d = xsf::gamma(n + 2 * alpha) / xsf::gamma(1 + n) / xsf::gamma(2 * alpha);
    return d * xsf::hyp2f1(-n, n + 2 * alpha, alpha + 0.5, (1 - x) / 2.0);
}

inline double eval_gegenbauer_l(long n, double alpha, double x)
{
    if (std::isnan(alpha) || std::isnan(x)) return NaN;
    if (n < 0) return 0.0;
    if (n == 0) return 1.0;
    if (n == 1) return 2 * alpha * x;
    if (alpha == 0.0) return eval_gegenbauer((double)n, alpha, x);
    if (std::fabs(x) < 1e-5) {
        const long a = n / 2;
        double d = (a % 2 == 0) ? 1 : -1;
        d /= xsf::beta(alpha, 1 + a);
        if (n == 2 * a) d /= (a + alpha);
        else d *= 2 * x;
        double p = 0;
        for (long kk = 0; kk < a + 1; kk++) {
            p += d;
            d *= -4 * (x * x) * (a - kk) * (-a + alpha + kk + n) / ((n + 1 - 2 * a + 2 * kk) * (n + 2 - 2 * a + 2 * kk));
            if (std::fabs(d) < 1e-20 * std::fabs(p)) break;
        }
        return p;
    }
    double d = x - 1, p = x;
    for (long kk = 0; kk < n - 1; kk++) {
        const double k = kk + 1.0;
        d = (2 * (k + alpha) / (k + 2 * alpha)) * (x - 1) * p + (k / (k + 2 * alpha)) * d;
        p = d + p;
    }
    if (std::fabs(alpha / n) < 1e-8) return 2 * alpha / n * p;
    return xsf::binom(n + 2 * alpha - 1, n) * p;
}

inline double eval_chebyt(double n, double x) { return xsf::hyp2f1(-n, n, 0.5, 0.5 * (1 - x)); }

inline double eval_chebyt_l(long k, double x)
{
    if (k < 0) k = -k;
    double b2 = 0, b1 = -1, b0 = 0;
    x = 2 * x;
    for (long m = 0; m < k + 1; m++) { b2 = b1; b1 = b0; b0 = x * b1 - b2; }
    return (b0 - b2) / 2.0;
}

inline double eval_chebyu(double n, double x) { return (n + 1) * xsf::hyp2f1(-n, n + 2, 1.5, 0.5 * (1 - x)); }

inline double eval_chebyu_l(long k, double x)
{
    int sign;
    if (k == -1) return 0;
    if (k < -1) { k = -k - 2; sign = -1; }
    else sign = 1;
    double b2 = 0, b1 = -1, b0 = 0;
    x = 2 * x;
    for (long m = 0; m < k + 1; m++) { b2 = b1; b1 = b0; b0 = x * b1 - b2; }
    return b0 * sign;
}

inline double eval_chebys(double n, double x) { return eval_chebyu(n, 0.5 * x); }
inline double eval_chebys_l(long n, double x) { return eval_chebyu_l(n, 0.5 * x); }
inline double eval_chebyc(double n, double x) { return 2 * eval_chebyt(n, 0.5 * x); }
inline double eval_chebyc_l(long n, double x) { return 2 * eval_chebyt_l(n, 0.5 * x); }
inline double eval_sh_chebyt(double n, double x) { return eval_chebyt(n, 2 * x - 1); }
inline double eval_sh_chebyt_l(long n, double x) { return eval_chebyt_l(n, 2 * x - 1); }
inline double eval_sh_chebyu(double n, double x) { return eval_chebyu(n, 2 * x - 1); }
inline double eval_sh_chebyu_l(long n, double x) { return eval_chebyu_l(n, 2 * x - 1); }

inline double eval_legendre(double n, double x) { return xsf::hyp2f1(-n, n + 1, 1, 0.5 * (1 - x)); }

inline double eval_legendre_l(long n, double x)
{
    if (n < 0) n = -n - 1;
    if (n == 0) return 1.0;
    if (n == 1) return x;
    if (std::fabs(x) < 1e-5) {
        const long a = n / 2;
        double d = (a % 2 == 0) ? 1 : -1;
        if (n == 2 * a) d *= -2 / xsf::beta(a + 1, -0.5);
        else d *= 2 * x / xsf::beta(a + 1, 0.5);
        double p = 0;
        for (long kk = 0; kk < a + 1; kk++) {
            p += d;
            d *= -2 * (x * x) * (a - kk) * (2 * n + 1 - 2 * a + 2 * kk) / ((n + 1 - 2 * a + 2 * kk) * (n + 2 - 2 * a + 2 * kk));
            if (std::fabs(d) < 1e-20 * std::fabs(p)) break;
        }
        return p;
    }
    double d = x - 1, p = x;
    for (long kk = 0; kk < n - 1; kk++) {
        const double k = kk + 1.0;
        d = ((2 * k + 1) / (k + 1)) * (x - 1) * p + (k / (k + 1)) * d;
        p = d + p;
    }
    return p;
}

inline double eval_sh_legendre(double n, double x) { return eval_legendre(n, 2 * x - 1); }
inline double eval_sh_legendre_l(long n, double x) { return eval_legendre_l(n, 2 * x - 1); }

inline double eval_genlaguerre(double n, double alpha, double x)
{
    if (alpha <= -1) return NaN;
    const double d = xsf::binom(n + alpha, n);
    return d * xsf::hyp1f1(-n, alpha + 1, x);
}

inline double eval_genlaguerre_l(long n, double alpha, double x)
{
    if (alpha <= -1) return NaN;
    if (std::isnan(alpha) || std::isnan(x)) return NaN;
    if (n < 0) return 0.0;
    if (n == 0) return 1.0;
    if (n == 1) return -x + alpha + 1;
    double d = -x / (alpha + 1), p = d + 1;
    for (long kk = 0; kk < n - 1; kk++) {
        const double k = kk + 1.0;
        d = -x / (k + alpha + 1) * p + (k / (k + alpha + 1)) * d;
        p = d + p;
    }
    return xsf::binom(n + alpha, n) * p;
}

inline double eval_laguerre(double n, double x) { return eval_genlaguerre(n, 0., x); }
inline double eval_laguerre_l(long n, double x) { return eval_genlaguerre_l(n, 0., x); }

inline double eval_hermitenorm(long n, double x)
{
    if (std::isnan(x)) return x;
    if (n < 0) return NaN;
    if (n == 0) return 1.0;
    if (n == 1) return x;
    double y3 = 0.0, y2 = 1.0, y1;
    for (long k = n; k > 1; k--) {
        y1 = x * y2 - k * y3;
        y3 = y2;
        y2 = y1;
    }
    return x * y2 - y3;
}

inline double eval_hermite(long n, double x)
{
    if (n < 0) return NaN;
    return eval_hermitenorm(n, std::sqrt(2) * x) * std::pow(2.0, n / 2.0);
}

/* ---------------------------------------------------------------- _cdflib_wrappers.pxd */

inline double cdf_result(double result, int status, double bound, int return_bound)
{
    if (status < 0) return NaN;
    if (status == 0) return result;
    if (status == 1 || status == 2) return return_bound ? bound : NaN;
    return NaN;
}

inline double bdtrik(double p, double xn, double pr)
{
    if (std::isnan(p) || !std::isfinite(xn) || std::isnan(pr)) return NaN;
    TupleDID r = cdfbin_which2(p, 1.0 - p, xn, pr, 1.0 - pr);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double bdtrin(double s, double p, double pr)
{
    if (std::isnan(s) || std::isnan(p) || std::isnan(pr)) return NaN;
    TupleDID r = cdfbin_which3(p, 1.0 - p, s, pr, 1.0 - pr);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double fdtridfd(double dfn, double p, double f)
{
    if (std::isnan(dfn) || std::isnan(p) || std::isnan(f)) return NaN;
    TupleDID r = cdff_which4(p, 1.0 - p, f, dfn);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double nbdtrik(double p, double xn, double pr)
{
    if (std::isnan(p) || !std::isfinite(xn) || std::isnan(pr)) return NaN;
    TupleDID r = cdfnbn_which2(p, 1.0 - p, xn, pr, 1.0 - pr);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double nbdtrin(double s, double p, double pr)
{
    if (std::isnan(s) || std::isnan(p) || std::isnan(pr)) return NaN;
    TupleDID r = cdfnbn_which3(p, 1.0 - p, s, pr, 1.0 - pr);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double ncfdtridfd(double dfn, double p, double nc, double f)
{
    if (std::isnan(dfn) || std::isnan(p) || std::isnan(nc) || std::isnan(f)) return NaN;
    TupleDID r = cdffnc_which4(p, 1.0 - p, f, dfn, nc);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double ncfdtridfn(double p, double dfd, double nc, double f)
{
    if (std::isnan(p) || std::isnan(dfd) || std::isnan(nc) || std::isnan(f)) return NaN;
    TupleDID r = cdffnc_which3(p, 1.0 - p, f, dfd, nc);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double ncfdtrinc(double dfn, double dfd, double p, double f)
{
    if (std::isnan(dfn) || std::isnan(dfd) || std::isnan(p) || std::isnan(f)) return NaN;
    TupleDID r = cdffnc_which5(p, 1.0 - p, f, dfn, dfd);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double nctdtridf(double p, double nc, double t)
{
    if (std::isnan(p) || std::isnan(nc) || std::isnan(t)) return NaN;
    tsr_cdflib_runaway = 0;
    TupleDID r = cdftnc_which3(p, 1.0 - p, t, nc);
    if (tsr_cdflib_runaway) return NaN;
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double nctdtrinc(double df, double p, double t)
{
    if (std::isnan(df) || std::isnan(p) || std::isnan(t)) return NaN;
    tsr_cdflib_runaway = 0;
    TupleDID r = cdftnc_which4(p, 1.0 - p, t, df);
    if (tsr_cdflib_runaway) return NaN;
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double nrdtrimn(double p, double std_, double x)
{
    if (std::isnan(p) || std::isnan(std_) || std::isnan(x)) return NaN;
    TupleDID r = cdfnor_which3(p, 1.0 - p, x, std_);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double nrdtrisd(double mn, double p, double x)
{
    if (std::isnan(mn) || std::isnan(p) || std::isnan(x)) return NaN;
    TupleDID r = cdfnor_which4(p, 1.0 - p, x, mn);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}
inline double stdtridf(double p, double t)
{
    const double q = 1.0 - p;
    if (std::isnan(p) || std::isnan(q) || std::isnan(t)) return NaN;
    TupleDID r = cdft_which3(p, q, t);
    return cdf_result(r.d1, r.i1, r.d2, 1);
}

/* ---------------------------------------------------------------- _ndtri_exp.pxd */

inline double ndtri_exp_small_y(double y)
{
    static const double P1[] = {4.05544892305962419923, 3.15251094599893866154e1, 5.71628192246421288162e1,
                                4.40805073893200834700e1, 1.46849561928858024014e1, 2.18663306850790267539,
                                -1.40256079171354495875e-1, -3.50424626827848203418e-2, -8.57456785154685413611e-4};
    static const double Q1[] = {1.57799883256466749731e1, 4.53907635128879210584e1, 4.13172038254672030440e1,
                                1.50425385692907503408e1, 2.50464946208309415979, -1.42182922854787788574e-1,
                                -3.80806407691578277194e-2, -9.33259480895457427372e-4};
    static const double P2[] = {3.23774891776946035970, 6.91522889068984211695, 3.93881025292474443415,
                                1.33303460815807542389, 2.01485389549179081538e-1, 1.23716634817820021358e-2,
                                3.01581553508235416007e-4, 2.65806974686737550832e-6, 6.23974539184983293730e-9};
    static const double Q2[] = {6.02427039364742014255, 3.67983563856160859403, 1.37702099489081330271,
                                2.16236993594496635890e-1, 1.34204006088543189037e-2, 3.28014464682127739104e-4,
                                2.89247864745380683936e-6, 6.79019408009981274425e-9};
    double x;
    if (y >= -DBL_MAX * 0.5) x = std::sqrt(-2 * y);
    else x = M_SQRT2 * std::sqrt(-y);
    const double x0 = x - std::log(x) / x;
    const double z = 1 / x;
    double x1;
    if (x < 8.0) x1 = z * xsf::cephes::polevl(z, P1, 8) / xsf::cephes::p1evl(z, Q1, 8);
    else x1 = z * xsf::cephes::polevl(z, P2, 8) / xsf::cephes::p1evl(z, Q2, 8);
    return x1 - x0;
}

inline double ndtri_exp(double y)
{
    if (y < -DBL_MAX) return -Inf;
    if (y < -2.0) return ndtri_exp_small_y(y);
    if (y > -0.14541345786885906) return -xsf::cephes::ndtri(-std::expm1(y));
    return xsf::cephes::ndtri(std::exp(y));
}

} // namespace scipy_port
