/*
 * scipy.stats: pareto, pearson3, powerlaw, powerlognorm, powernorm, rayleigh, rdist, recipinvgauss, reciprocal,
 * rel_breitwigner, rice, semicircular, skewcauchy, skewnorm (scipy/stats/_continuous_distns.py) and
 * studentized_range (its integrands from scipy/stats/_stats.pyx, integrated as scipy.integrate.nquad does),
 * SciPy 1.17.1, private method by private method. See stats_dist.hpp.
 */
#include <exception>
#include "stats_dist.hpp"

#include <algorithm>
#include <complex>

extern "C" {
#include "quadpack/quadpack.h"
}

namespace tsd {
namespace {

using cplx = std::complex<double>;

const double NORM_PDF_C = std::sqrt(2 * M_PI);
const double NORM_PDF_LOGC = std::log(NORM_PDF_C);
const double EULER = 0.577215664901532860606512090082402431042;
const double SQRT_PI = 1.772453850905516;
const double SQRT_2_OVER_PI = 0.7978845608028654;

inline double norm_pdf(double x) { return std::exp(-(x * x) / 2.0) / NORM_PDF_C; }
inline double norm_logpdf(double x) { return -(x * x) / 2.0 - NORM_PDF_LOGC; }

/* NumPy's complex multiply on this machine is the SIMD loop (npyv_muladdsub): each part is one fused step */
inline cplx cmul(cplx a, cplx b)
{
    return cplx(std::fma(a.real(), b.real(), -(a.imag() * b.imag())), std::fma(a.real(), b.imag(), a.imag() * b.real()));
}
/* NumPy's complex division (Smith's method, loops.c.src) */
inline cplx cdiv(cplx a, cplx b)
{
    const double in1r = a.real(), in1i = a.imag(), in2r = b.real(), in2i = b.imag();
    const double in2r_abs = std::fabs(in2r), in2i_abs = std::fabs(in2i);
    if (in2r_abs >= in2i_abs) {
        if (in2r_abs == 0 && in2i_abs == 0) return cplx(in1r / in2r_abs, in1i / in2i_abs);
        const double rat = in2i / in2r;
        const double scl = 1.0 / (in2r + in2i * rat);
        return cplx((in1r + in1i * rat) * scl, (in1i - in1r * rat) * scl);
    }
    const double rat = in2r / in2i;
    const double scl = 1.0 / (in2i + in2r * rat);
    return cplx((in1r * rat + in1i) * scl, (in1i * rat - in1r) * scl);
}

/* ---------------------------------------------------------------- pareto */

struct Pareto : Cont {
    Pareto() { name = "pareto"; shapes = "b"; nshape = 1; a = 1.0; doc = "A Pareto continuous random variable."; }
    double pdf(double x, const double *s) const override { return s[0] * std::pow(x, -s[0] - 1); }
    double cdf(double x, const double *s) const override { return 1 - std::pow(x, -s[0]); }
    double ppf(double q, const double *s) const override { return std::pow(1 - q, -1.0 / s[0]); }
    double sf(double x, const double *s) const override { return std::pow(x, -s[0]); }
    double isf(double q, const double *s) const override { return std::pow(q, -1.0 / s[0]); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double b = s[0];
        if (moments & MOM_M) st.set(0, b > 1 ? b / (b - 1.0) : INF);
        if (moments & MOM_V) st.set(1, b > 2 ? b / (b - 2.0) / sq(b - 1.0) : INF);
        if (moments & MOM_S) st.set(2, b > 3 ? 2 * (b + 1.0) * std::sqrt(b - 2.0) / ((b - 3.0) * std::sqrt(b)) : NaN);
        if (moments & MOM_K) {
            /* np.polyval([1, 1, -6, -2], b) / np.polyval([1, -7, 12, 0], b) */
            const double num = ((((0 * b + 1.0) * b + 1.0) * b - 6) * b - 2);
            const double den = ((((0 * b + 1.0) * b - 7.0) * b + 12.0) * b + 0.0);
            st.set(3, b > 4 ? 6.0 * num / den : NaN);
        }
    }
    double entropy(const double *s) const override { return 1 + 1.0 / s[0] - std::log(s[0]); }
};
Registrar r_pareto(new Pareto);

/* ---------------------------------------------------------------- pearson3 */

/* gamma.logpdf / gamma.cdf / gamma.sf (the public methods, loc 0, scale 1) as pearson3 calls them */
double gamma_pub_logpdf(double x, double a)
{
    if (!(a > 0) || std::isnan(x)) return NaN;
    if (!(0 <= x)) return -INF;
    return sc::xlogy(a - 1.0, x) - x - sc::gammaln(a);
}
double gamma_pub_cdf(double x, double a)
{
    if (!(a > 0) || std::isnan(x)) return NaN;
    if (x >= INF) return 1.0;
    if (0 < x) return sc::gammainc(a, x);
    return 0.0;
}
double gamma_pub_sf(double x, double a)
{
    if (!(a > 0) || std::isnan(x)) return NaN;
    if (x <= 0) return 1.0;
    if (x < INF) return sc::gammaincc(a, x);
    return 0.0;
}

struct Pearson3 : Cont {
    Pearson3() { name = "pearson3"; shapes = "skew"; nshape = 1; doc = "A pearson type III continuous random variable."; }
    static bool small(double skew) { return std::fabs(skew) < 1.6e-05; }
    /* _preprocess (loc 0, scale 1) for an element outside the normal mask */
    static void pre(double x, double skew, double &beta, double &alpha, double &zeta, double &transx)
    {
        beta = 2.0 / (skew * 1.0);
        alpha = sq(1.0 * beta);
        zeta = 0.0 - alpha / beta;
        transx = beta * (x - zeta);
    }
    bool argcheck(const double *s) const override { return std::isfinite(s[0]); }
    void stats(const double *s, int, Stats4 &st) const override { st.set(0.0, 1.0, s[0], 1.5 * sq(s[0])); }
    double pdf(double x, const double *s) const override
    {
        const double ans = std::exp(logpdf(x, s));
        return std::isnan(ans) ? 0.0 : ans;
    }
    double logpdf(double x, const double *s) const override
    {
        if (small(s[0])) return std::log(norm_pdf(x));
        double beta, alpha, zeta, transx;
        pre(x, s[0], beta, alpha, zeta, transx);
        return std::log(std::fabs(beta)) + gamma_pub_logpdf(transx, alpha);
    }
    double cdf(double x, const double *s) const override
    {
        const double skew = s[0];
        if (small(skew)) return sc::ndtr(x);
        double beta, alpha, zeta, transx;
        pre(x, skew, beta, alpha, zeta, transx);
        if (skew > 0) return gamma_pub_cdf(transx, alpha);
        if (skew < 0) return gamma_pub_sf(transx, alpha);
        return 1.0;
    }
    double sf(double x, const double *s) const override
    {
        const double skew = s[0];
        if (small(skew)) return sc::ndtr(-x);
        double beta, alpha, zeta, transx;
        pre(x, skew, beta, alpha, zeta, transx);
        if (skew > 0) return gamma_pub_sf(transx, alpha);
        if (skew < 0) return gamma_pub_cdf(transx, alpha);
        return 1.0;
    }
    double ppf(double q, const double *s) const override
    {
        if (small(s[0])) return sc::ndtri(q);
        double beta, alpha, zeta, transx;
        pre(0.0, s[0], beta, alpha, zeta, transx);
        if (beta < 0) q = 1 - q;
        return sc::gammaincinv(alpha, q) / beta + zeta;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* the normals for the elements in the normal mask first, then the gammas for the others */
        for (int64_t i = 0; i < n; i++)
            if (small(sa[0][i])) out[i] = random_standard_normal(bg);
        for (int64_t i = 0; i < n; i++) {
            if (small(sa[0][i])) continue;
            double beta, alpha, zeta, transx;
            pre(0.0, sa[0][i], beta, alpha, zeta, transx);
            out[i] = random_standard_gamma(bg, alpha) / beta + zeta;
        }
    }
};
Registrar r_pearson3(new Pearson3);

/* ---------------------------------------------------------------- powerlaw */

struct Powerlaw : Cont {
    Powerlaw() { name = "powerlaw"; shapes = "a"; nshape = 1; a = 0.0; b = 1.0; doc = "A power-function continuous random variable."; }
    /* _support_mask excludes x = 0 when a < 1: the public pdf is 0 there (the only way _pdf sees x = 0) */
    double pdf(double x, const double *s) const override
    {
        if (x == 0 && !(s[0] >= 1)) return 0.0;
        return s[0] * std::pow(x, s[0] - 1.0);
    }
    double logpdf(double x, const double *s) const override
    {
        if (x == 0 && !(s[0] >= 1)) return -INF;
        return std::log(s[0]) + sc::xlogy(s[0] - 1, x);
    }
    double cdf(double x, const double *s) const override { return std::pow(x, s[0] * 1.0); }
    double logcdf(double x, const double *s) const override { return s[0] * std::log(x); }
    double ppf(double q, const double *s) const override { return std::pow(q, 1.0 / s[0]); }
    double sf(double p, const double *s) const override { return -sc::powm1(p, s[0]); }
    double munp(int n, const double *s) const override { return s[0] / (s[0] + n); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double a = s[0];
        const double poly = (((0 * a + 1) * a - 1) * a - 6) * a + 2;   /* np.polyval([1, -1, -6, 2], a) */
        st.set(a / (a + 1.0), a / (a + 2.0) / sq(a + 1.0), -2.0 * ((a - 1.0) / (a + 3.0)) * std::sqrt((a + 2.0) / a),
               6 * poly / (a * (a + 3.0) * (a + 4)));
    }
    double entropy(const double *s) const override { return 1 - 1.0 / s[0] - std::log(s[0]); }
};
Registrar r_powerlaw(new Powerlaw);

/* ---------------------------------------------------------------- powerlognorm */

struct Powerlognorm : Cont {
    Powerlognorm() { name = "powerlognorm"; shapes = "c, s"; nshape = 2; a = 0.0; doc = "A power log-normal continuous random variable."; }
    /* _support_mask = _open_support_mask: the public pdf is 0 at the end points */
    double pdf(double x, const double *s) const override
    {
        if (!(0 < x && x < INF)) return 0.0;
        return std::exp(logpdf(x, s));
    }
    double logpdf(double x, const double *s) const override
    {
        if (!(0 < x && x < INF)) return -INF;
        const double c = s[0], sd = s[1];
        return std::log(c) - std::log(x) - std::log(sd) + norm_logpdf(std::log(x) / sd) +
               sc::log_ndtr(-std::log(x) / sd) * (c - 1.0);
    }
    double cdf(double x, const double *s) const override { return -sc::expm1(logsf(x, s)); }
    double ppf(double q, const double *s) const override { return isf(1 - q, s); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return sc::log_ndtr(-std::log(x) / s[1]) * s[0]; }
    double isf(double q, const double *s) const override { return std::exp(-sc::ndtri(std::pow(q, 1.0 / s[0])) * s[1]); }
};
Registrar r_powerlognorm(new Powerlognorm);

/* ---------------------------------------------------------------- powernorm */

struct Powernorm : Cont {
    Powernorm() { name = "powernorm"; shapes = "c"; nshape = 1; doc = "A power normal continuous random variable."; }
    double pdf(double x, const double *s) const override { return s[0] * norm_pdf(x) * std::pow(sc::ndtr(-x), s[0] - 1.0); }
    double logpdf(double x, const double *s) const override
    {
        return std::log(s[0]) + norm_logpdf(x) + (s[0] - 1) * sc::log_ndtr(-x);
    }
    double cdf(double x, const double *s) const override { return -sc::expm1(logsf(x, s)); }
    double ppf(double q, const double *s) const override { return -sc::ndtri(std::pow(1.0 - q, 1.0 / s[0])); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return s[0] * sc::log_ndtr(-x); }
    double isf(double q, const double *s) const override { return -sc::ndtri(std::exp(std::log(q) / s[0])); }
};
Registrar r_powernorm(new Powernorm);

/* ---------------------------------------------------------------- rayleigh */

struct Rayleigh : Cont {
    Rayleigh() { name = "rayleigh"; a = 0.0; doc = "A Rayleigh continuous random variable."; }
    /* _support_mask = _open_support_mask: the public pdf is 0 at the end points */
    double pdf(double r, const double *s) const override
    {
        if (!(0 < r && r < INF)) return 0.0;
        return std::exp(logpdf(r, s));
    }
    double logpdf(double r, const double *) const override
    {
        if (!(0 < r && r < INF)) return -INF;
        return std::log(r) - 0.5 * r * r;
    }
    double cdf(double r, const double *) const override { return -sc::expm1(-0.5 * (r * r)); }
    double ppf(double q, const double *) const override { return std::sqrt(-2 * sc::log1p(-q)); }
    double sf(double r, const double *s) const override { return std::exp(logsf(r, s)); }
    double logsf(double r, const double *) const override { return -0.5 * r * r; }
    double isf(double q, const double *) const override { return std::sqrt(-2 * std::log(q)); }
    void stats(const double *, int, Stats4 &st) const override
    {
        const double val = 4 - M_PI;
        st.set(std::sqrt(M_PI / 2), val / 2, 2 * (M_PI - 3) * std::sqrt(M_PI) / std::pow(val, 1.5),
               6 * M_PI / val - 16 / std::pow(val, 2.0));
    }
    double entropy(const double *) const override { return EULER / 2.0 + 1 - 0.5 * std::log(2.0); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        /* chi.rvs(2) = sqrt(chi2.rvs(2)) = sqrt(random_state.chisquare(2)) */
        for (int64_t i = 0; i < n; i++) out[i] = std::sqrt(random_chisquare(bg, 2.0));
    }
};
Registrar r_rayleigh(new Rayleigh);

/* ---------------------------------------------------------------- rdist */

inline double beta_logpdf_priv(double x, double a, double b)
{
    double lPx = sc::xlog1py(b - 1.0, -x) + sc::xlogy(a - 1.0, x);
    lPx -= sc::betaln(a, b);
    return lPx;
}
inline double rdist_ppf(double q, double c) { return 2 * sc::_beta_ppf(q, c / 2, c / 2) - 1; }

struct Rdist : Cont {
    Rdist() { name = "rdist"; shapes = "c"; nshape = 1; a = -1.0; b = 1.0; doc = "An R-distributed (symmetric beta) continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        return -std::log(2.0) + beta_logpdf_priv((x + 1) / 2, s[0] / 2, s[0] / 2);
    }
    double cdf(double x, const double *s) const override { return sc::betainc(s[0] / 2, s[0] / 2, (x + 1) / 2); }
    double sf(double x, const double *s) const override { return sc::betaincc(s[0] / 2, s[0] / 2, (x + 1) / 2); }
    double ppf(double q, const double *s) const override { return rdist_ppf(q, s[0]); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = 2 * random_beta(bg, sa[0][i] / 2, sa[0][i] / 2) - 1;
    }
    double munp(int n, const double *s) const override
    {
        const double numerator = (double)(1 - n % 2) * sc::beta((n + 1.0) / 2, s[0] / 2.0);
        return numerator / sc::beta(1.0 / 2, s[0] / 2.0);
    }
};
Registrar r_rdist(new Rdist);

/* ---------------------------------------------------------------- recipinvgauss */

struct Recipinvgauss : Cont {
    Recipinvgauss() { name = "recipinvgauss"; shapes = "mu"; nshape = 1; a = 0.0; doc = "A reciprocal inverse Gaussian continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double mu = s[0];
        if (!(x > 0)) return -INF;
        return -sq(1 - mu * x) / (2 * x * sq(mu)) - 0.5 * std::log(2 * M_PI * x);
    }
    double cdf(double x, const double *s) const override
    {
        const double mu = s[0];
        const double trm1 = 1.0 / mu - x, trm2 = 1.0 / mu + x;
        const double isqx = 1.0 / std::sqrt(x);
        return sc::ndtr(-isqx * trm1) - std::exp(2.0 / mu) * sc::ndtr(-isqx * trm2);
    }
    double sf(double x, const double *s) const override
    {
        const double mu = s[0];
        const double trm1 = 1.0 / mu - x, trm2 = 1.0 / mu + x;
        const double isqx = 1.0 / std::sqrt(x);
        return sc::ndtr(isqx * trm1) + std::exp(2.0 / mu) * sc::ndtr(-isqx * trm2);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = 1.0 / random_wald(bg, sa[0][i], 1.0);
    }
};
Registrar r_recipinvgauss(new Recipinvgauss);

/* ---------------------------------------------------------------- reciprocal */

/* numpy.remainder */
inline double py_mod(double a, double b)
{
    double mod = std::fmod(a, b);
    if (!b) return mod;
    if (mod) {
        if ((b < 0) != (mod < 0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

/* _log_diff(log_p, log_q) = sc.logsumexp([log_p, log_q + pi*1j], axis=0), SciPy's complex logsumexp for two
   elements (scipy/special/_logsumexp.py) */
cplx log_diff(double log_p, double log_q)
{
    const cplx e[2] = {cplx(log_p, 0.0), cplx(log_q, M_PI)};
    /* the direct form, used where the stable one is not finite */
    const cplx sum_direct = std::exp(e[0]) + std::exp(e[1]);
    const cplx out_inf = std::log(sum_direct);
    /* _elements_and_indices_with_max_real: the last element whose real part is the maximum */
    double mx = std::max(e[0].real(), e[1].real());
    int imax = -1;
    for (int i = 0; i < 2; i++)
        if (e[i].real() == mx) imax = i;
    cplx a_max(0.0, 0.0), m(0.0, 0.0);
    cplx s(0.0, 0.0);
    if (imax < 0) {                      /* NaN: no element matches, every element is selected */
        a_max = e[0] + e[1];
        m = cplx(0.0, 0.0);
        s = std::exp(e[0] - a_max) + std::exp(e[1] - a_max);
    } else {
        a_max = e[imax];
        m = cplx(1.0, 0.0);
        const cplx ninf(-INF, 0.0);
        const cplx t0 = std::exp((imax == 0 ? ninf : e[0]) - a_max);
        const cplx t1 = std::exp((imax == 1 ? ninf : e[1]) - a_max);
        s = t0 + t1;
    }
    if (!(s == cplx(0.0, 0.0))) s = cdiv(s, m);
    /* np.log1p for complex: (log(hypot(re + 1, im)), atan2(im, re + 1)) */
    const cplx l1p(std::log(std::hypot(s.real() + 1, s.imag())), std::atan2(s.imag(), s.real() + 1));
    cplx out = l1p + std::log(m) + a_max;
    if (!(std::isfinite(out.real()) && std::isfinite(out.imag()))) out = out_inf;
    /* _wrap_radians on the imaginary part */
    double im = out.imag();
    if (!(std::fabs(im) < M_PI)) im = -(py_mod(-im + M_PI, 2 * M_PI) - M_PI);
    return cplx(out.real(), im);
}

struct Reciprocal : Cont {
    Reciprocal() { name = "reciprocal"; shapes = "a, b"; nshape = 2; doc = "A loguniform or reciprocal continuous random variable."; }
    bool argcheck(const double *s) const override { return (s[0] > 0) && (s[1] > s[0]); }
    void get_support(const double *s, double &lo, double &hi) const override { lo = s[0]; hi = s[1]; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override { return -std::log(x) - std::log(std::log(s[1]) - std::log(s[0])); }
    double cdf(double x, const double *s) const override
    {
        return (std::log(x) - std::log(s[0])) / (std::log(s[1]) - std::log(s[0]));
    }
    double ppf(double q, const double *s) const override
    {
        return std::exp(std::log(s[0]) + q * (std::log(s[1]) - std::log(s[0])));
    }
    double munp(int n, const double *s) const override
    {
        if (n == 0) return 1.0;
        const double a = s[0], b = s[1];
        const double t1 = 1 / (std::log(b) - std::log(a)) / n;
        const double t2 = std::exp(log_diff(n * std::log(b), n * std::log(a))).real();
        return t1 * t2;
    }
    double entropy(const double *s) const override
    {
        const double a = s[0], b = s[1];
        return 0.5 * (std::log(a) + std::log(b)) + std::log(std::log(b) - std::log(a));
    }
};
Registrar r_reciprocal(new Reciprocal);

/* ---------------------------------------------------------------- rel_breitwigner */

struct RelBreitwigner : Cont {
    RelBreitwigner() { name = "rel_breitwigner"; shapes = "rho"; nshape = 1; a = 0.0; doc = "A relativistic Breit-Wigner random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0; }
    double pdf(double x, const double *s) const override
    {
        const double rho = s[0];
        const double C = std::sqrt(2 * (1 + 1 / sq(rho)) / (1 + std::sqrt(1 + 1 / sq(rho)))) * 2 / M_PI;
        return C / (sq((x - rho) * (x + rho) / rho) + 1);
    }
    double cdf(double x, const double *s) const override
    {
        const double rho = s[0];
        const double C = std::sqrt(2 / (1 + std::sqrt(1 + 1 / sq(rho)))) / M_PI;
        const cplx i_over_rho = cdiv(cplx(0.0, 1.0), cplx(rho, 0.0));
        const cplx A = std::sqrt(cplx(-1.0, 0.0) + i_over_rho);
        const cplx den = std::sqrt(cmul(cplx(-rho, 0.0), cplx(rho, 1.0)));
        const cplx B = std::atan(cdiv(cplx(x, 0.0), den));
        const double result = C * 2 * cmul(A, B).imag();
        return std::isnan(result) ? result : std::min(result, 1.0);
    }
    double munp(int n, const double *s) const override
    {
        const double rho = s[0];
        if (n == 0) return 1.0;
        if (n == 1) {
            const double C = std::sqrt(2 * (1 + 1 / sq(rho)) / (1 + std::sqrt(1 + 1 / sq(rho)))) / M_PI * rho;
            return C * (M_PI / 2 + std::atan(rho));
        }
        if (n == 2) {
            const double C = std::sqrt((1 + 1 / sq(rho)) / (2 * (1 + std::sqrt(1 + 1 / sq(rho))))) * rho;
            const cplx i_over_rho = cdiv(cplx(0.0, 1.0), cplx(rho, 0.0));
            const cplx rho_i = cmul(cplx(rho, 0.0), cplx(0.0, 1.0));
            const cplx result = cdiv(cplx(1.0, 0.0) - rho_i, std::sqrt(cplx(-1.0, 0.0) - i_over_rho));
            return 2 * C * result.real();
        }
        return INF;
    }
    void stats(const double *, int, Stats4 &st) const override
    {
        st.set(2, NaN);
        st.set(3, NaN);
    }
};
Registrar r_rel_breitwigner(new RelBreitwigner);

/* ---------------------------------------------------------------- rice */

struct Rice : Cont {
    Rice() { name = "rice"; shapes = "b"; nshape = 1; a = 0.0; doc = "A Rice continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* t = b / sqrt(2) + standard_normal(size=(2,) + size): the first row, then the second */
        std::vector<double> t0((size_t)n);
        for (int64_t i = 0; i < n; i++) t0[(size_t)i] = sa[0][i] / std::sqrt(2.0) + random_standard_normal(bg);
        for (int64_t i = 0; i < n; i++) {
            const double t1 = sa[0][i] / std::sqrt(2.0) + random_standard_normal(bg);
            out[i] = std::sqrt(t0[(size_t)i] * t0[(size_t)i] + t1 * t1);
        }
    }
    double cdf(double x, const double *s) const override { return sc::chndtr(x * x, 2, s[0] * s[0]); }
    double ppf(double q, const double *s) const override { return std::sqrt(sc::chndtrix(q, 2, s[0] * s[0])); }
    double pdf(double x, const double *s) const override
    {
        const double b = s[0];
        return x * std::exp(-(x - b) * (x - b) / 2.0) * sc::i0e(x * b);
    }
    double munp(int n, const double *s) const override
    {
        const double b = s[0];
        const double nd2 = n / 2.0;
        const double n1 = 1 + nd2;
        const double b2 = b * b / 2.0;
        return std::pow(2.0, nd2) * std::exp(-b2) * sc::gamma(n1) * sc::hyp1f1(n1, 1, b2);
    }
};
Registrar r_rice(new Rice);

/* ---------------------------------------------------------------- semicircular */

struct Semicircular : Cont {
    Semicircular() { name = "semicircular"; a = -1.0; b = 1.0; doc = "A semicircular continuous random variable."; }
    double pdf(double x, const double *) const override { return 2.0 / M_PI * std::sqrt(1 - x * x); }
    double logpdf(double x, const double *) const override { return std::log(2 / M_PI) + 0.5 * sc::log1p(-x * x); }
    double cdf(double x, const double *) const override { return 0.5 + 1.0 / M_PI * (x * std::sqrt(1 - x * x) + std::asin(x)); }
    double ppf(double q, const double *) const override { return rdist_ppf(q, 3); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = std::sqrt(random_standard_uniform(bg));
        for (int64_t i = 0; i < n; i++) out[i] = out[i] * std::cos(M_PI * random_standard_uniform(bg));
    }
    void stats(const double *, int, Stats4 &st) const override { st.set(0, 0.25, 0, -1.0); }
    double entropy(const double *) const override { return 0.6447298858494002; }
};
Registrar r_semicircular(new Semicircular);

/* ---------------------------------------------------------------- skewcauchy */

inline double np_sign(double x) { return std::isnan(x) ? x : (x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)); }

struct Skewcauchy : Cont {
    Skewcauchy() { name = "skewcauchy"; shapes = "a"; nshape = 1; doc = "A skewed Cauchy random variable."; }
    bool argcheck(const double *s) const override { return std::fabs(s[0]) < 1; }
    double pdf(double x, const double *s) const override
    {
        const double a = s[0];
        return 1 / (M_PI * ((x * x) / sq(a * np_sign(x) + 1) + 1));
    }
    double cdf(double x, const double *s) const override
    {
        const double a = s[0];
        return x <= 0 ? (1 - a) / 2 + (1 - a) / M_PI * std::atan(x / (1 - a)) : (1 - a) / 2 + (1 + a) / M_PI * std::atan(x / (1 + a));
    }
    double ppf(double x, const double *s) const override
    {
        const double a = s[0];
        const bool i = x < cdf(0.0, s);
        return i ? std::tan(M_PI / (1 - a) * (x - (1 - a) / 2)) * (1 - a) : std::tan(M_PI / (1 + a) * (x - (1 - a) / 2)) * (1 + a);
    }
    void stats(const double *, int, Stats4 &st) const override { st.set(NaN, NaN, NaN, NaN); }
};
Registrar r_skewcauchy(new Skewcauchy);

/* ---------------------------------------------------------------- skewnorm */

struct Skewnorm : Cont {
    Skewnorm() { name = "skewnorm"; shapes = "a"; nshape = 1; doc = "A skew-normal random variable."; }
    bool argcheck(const double *s) const override { return std::isfinite(s[0]); }
    double pdf(double x, const double *s) const override
    {
        const double a = s[0];
        return a == 0 ? norm_pdf(x) : 2.0 * norm_pdf(x) * sc::ndtr(a * x);
    }
    double logpdf(double x, const double *s) const override
    {
        const double a = s[0];
        return a == 0 ? norm_logpdf(x) : std::log(2.0) + norm_logpdf(x) + sc::log_ndtr(a * x);
    }
    double cdf(double x, const double *s) const override
    {
        const double a = s[0];
        double c = sc::_skewnorm_cdf(x, 0.0, 1.0, a);
        if (c < 1e-06 && a > 0) c = Cont::cdf(x, s);   /* super()._cdf: quad of _pdf from -inf */
        return std::isnan(c) ? c : std::min(std::max(c, 0.0), 1.0);
    }
    double ppf(double x, const double *s) const override { return sc::_skewnorm_ppf(x, 0.0, 1.0, s[0]); }
    double sf(double x, const double *s) const override
    {
        const double ms = -s[0];
        return cdf(-x, &ms);
    }
    double isf(double x, const double *s) const override { return sc::_skewnorm_isf(x, 0.0, 1.0, s[0]); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<double> u0((size_t)n);
        for (int64_t i = 0; i < n; i++) u0[(size_t)i] = random_normal(bg, 0.0, 1.0);
        for (int64_t i = 0; i < n; i++) {
            const double v = random_normal(bg, 0.0, 1.0);
            const double a = sa[0][i];
            const double d = a / std::sqrt(1 + a * a);
            const double u1 = d * u0[(size_t)i] + v * std::sqrt(1 - d * d);
            out[i] = u0[(size_t)i] >= 0 ? u1 : -u1;
        }
    }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double a = s[0];
        const double cst = std::sqrt(2 / M_PI) * a / std::sqrt(1 + a * a);
        if (moments & MOM_M) st.set(0, cst);
        if (moments & MOM_V) st.set(1, 1 - cst * cst);
        if (moments & MOM_S) st.set(2, (4 - M_PI) / 2 * std::pow(cst / std::sqrt(1 - cst * cst), 3.0));
        if (moments & MOM_K) st.set(3, 2 * (M_PI - 3) * (std::pow(cst, 4.0) / sq(1 - cst * cst)));
    }
    double munp(int order, const double *s) const override
    {
        static const double C1[] = {1};
        static const double C3[] = {3, -1};
        static const double C5[] = {15, -10, 3};
        static const double C7[] = {105, -105, 63, -15};
        static const double C9[] = {945, -1260, 1134, -540, 105};
        static const double C11[] = {10395, -17325, 20790, -14850, 5775, -945};
        static const double C13[] = {135135, -270270, 405405, -386100, 225225, -73710, 10395};
        static const double C15[] = {2027025, -4729725, 8513505, -10135125, 7882875, -3869775, 1091475, -135135};
        static const double C17[] = {34459425, -91891800, 192972780, -275675400, 268017750, -175429800, 74220300, -18378360, 2027025};
        static const double C19[] = {654729075, -1964187225, 4714049340, -7856748900, 9166207050, -7499623950, 4230557100, -1571349780, 346621275, -34459425};
        static const double *const P[] = {C1, C3, C5, C7, C9, C11, C13, C15, C17, C19};
        const double a = s[0];
        if (order % 2) {
            if (order > 19) { fail("skewnorm noncentral moments not implemented for odd orders greater than 19."); return NaN; }
            const double delta = a / std::sqrt(1 + a * a);
            const int k = order / 2;          /* the polynomial for `order` has k + 1 coefficients */
            const double *c = P[k];
            const double x = delta * delta;
            /* numpy.polynomial.polynomial.polyval: c0 = c[-1] + x*0; c0 = c[-i] + c0*x */
            double c0 = c[k] + x * 0;
            for (int i = k - 1; i >= 0; i--) c0 = c[i] + c0 * x;
            return delta * c0 * SQRT_2_OVER_PI;
        }
        return sc::gamma((order + 1) / 2.0) * std::pow(2.0, order / 2.0) / SQRT_PI;
    }
};
Registrar r_skewnorm(new Skewnorm);

/* ---------------------------------------------------------------- studentized_range */

/* scipy.integrate.quad with explicit tolerances (QUADPACK qagse / qagie, limit 50), nestable */
struct TolQuad {
    double (*fn)(void *, double);
    void *ctx;
    std::exception_ptr err;   /* an exception from the integrand, rethrown after QUADPACK (C) returns */
};
thread_local TolQuad *g_tq = nullptr;
double tq_thunk(double *x)
{
    TolQuad *c = g_tq;
    if (c->err) return std::numeric_limits<double>::quiet_NaN();
    try {
        return c->fn(c->ctx, *x);
    } catch (...) {
        c->err = std::current_exception();
        return std::numeric_limits<double>::quiet_NaN();
    }
}

template <class F> double quad_tol(F &&f, double a, double b, double epsabs, double epsrel)
{
    if (a == b) return 0.0;
    TolQuad call;
    call.ctx = (void *)&f;
    call.fn = [](void *p, double x) -> double { return (*(typename std::remove_reference<F>::type *)p)(x); };
    const bool flip = b < a;
    const double lo = std::min(a, b), hi = std::max(a, b);
    enum { LIMIT = 50 };
    double alist[LIMIT], blist[LIMIT], rlist[LIMIT], elist[LIMIT];
    int iord[LIMIT];
    double result = 0.0, abserr = 0.0;
    int neval = 0, ier = 6, last = 0;
    TolQuad *saved = g_tq;
    g_tq = &call;
    if (hi != INF && lo != -INF) {
        dqagse(tq_thunk, lo, hi, epsabs, epsrel, LIMIT, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);
    } else {
        int inf;
        double bound;
        if (hi == INF && lo != -INF) { inf = 1; bound = lo; }
        else if (hi == INF && lo == -INF) { inf = 2; bound = 0.0; }
        else { inf = -1; bound = hi; }
        dqagie(tq_thunk, bound, inf, epsabs, epsrel, LIMIT, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);
    }
    g_tq = saved;
    if (call.err) std::rethrow_exception(call.err);
    if (ier == 6) fail("quad: the input is invalid");
    return flip ? -result : result;
}

const double SR_EPSABS = 1e-11, SR_EPSREL = 1e-12;

/* _stats.pyx helpers */
inline double sr_phi(double z) { return 0.3989422804014327 * std::exp(-0.5 * z * z); }
inline double sr_logphi(double z) { return -0.9189385332046727 - 0.5 * z * z; }
inline double sr_Phi(double z) { return 0.5 * std::erfc(-z * 0.7071067811865475); }

double sr_cdf_logconst(double k, double df)
{
    const double log_2 = 0.6931471805599453;
    return std::log(k) + (df / 2) * std::log(df) - (std::lgamma(df / 2) + (df / 2 - 1) * log_2);
}
double sr_pdf_logconst(double k, double df)
{
    const double log_2 = 0.6931471805599453;
    return std::log(k) + std::log(k - 1) + (df / 2) * std::log(df) - (std::lgamma(df / 2) + (df / 2 - 1) * log_2);
}
double sr_cdf_integrand(double z, double s, double q, double k, double df, double c)
{
    const double log_terms = c + (df - 1) * std::log(s) - (df * s * s / 2) + sr_logphi(z);
    return std::exp(log_terms) * std::pow(sr_Phi(z + q * s) - sr_Phi(z), k - 1);
}
double sr_cdf_asymptotic(double z, double q, double k) { return k * sr_phi(z) * std::pow(sr_Phi(z + q) - sr_Phi(z), k - 1); }
double sr_pdf_integrand(double z, double s, double q, double k, double df, double c)
{
    const double log_terms = c + df * std::log(s) - df * s * s / 2 + sr_logphi(z) + sr_logphi(s * q + z);
    return std::exp(log_terms) * std::pow(sr_Phi(s * q + z) - sr_Phi(z), k - 2);
}
double sr_pdf_asymptotic(double z, double q, double k)
{
    return k * (k - 1) * sr_phi(z) * sr_phi(z + q) * std::pow(sr_Phi(z + q) - sr_Phi(z), k - 2);
}

struct StudentizedRange : Cont {
    StudentizedRange() { name = "studentized_range"; shapes = "k, df"; nshape = 2; a = 0.0; doc = "A studentized range continuous random variable."; }
    bool argcheck(const double *s) const override { return (s[0] > 1) && (s[1] > 0); }
    /* nquad(func, [(-inf, inf), (0, inf)]): the outer quad over s, the inner over z */
    double pdf(double x, const double *s) const override
    {
        const double q = x, k = s[0], df = s[1];
        if (df < 100000) {
            const double c = sr_pdf_logconst(k, df);
            return quad_tol([&](double sv) {
                return quad_tol([&](double z) { return sr_pdf_integrand(z, sv, q, k, df, c); }, -INF, INF, SR_EPSABS, SR_EPSREL);
            }, 0.0, INF, SR_EPSABS, SR_EPSREL);
        }
        return quad_tol([&](double z) { return sr_pdf_asymptotic(z, q, k); }, -INF, INF, SR_EPSABS, SR_EPSREL);
    }
    double cdf(double x, const double *s) const override
    {
        const double q = x, k = s[0], df = s[1];
        double r;
        if (df < 100000) {
            const double c = sr_cdf_logconst(k, df);
            r = quad_tol([&](double sv) {
                return quad_tol([&](double z) { return sr_cdf_integrand(z, sv, q, k, df, c); }, -INF, INF, SR_EPSABS, SR_EPSREL);
            }, 0.0, INF, SR_EPSABS, SR_EPSREL);
        } else {
            r = quad_tol([&](double z) { return sr_cdf_asymptotic(z, q, k); }, -INF, INF, SR_EPSABS, SR_EPSREL);
        }
        return std::isnan(r) ? r : std::min(std::max(r, 0.0), 1.0);
    }
    /* nquad over [(-inf, inf), (0, inf), (0, inf)]: q outermost, then s, then z */
    double munp(int n, const double *s) const override
    {
        const double K = n, k = s[0], df = s[1];
        const double c = sr_pdf_logconst(k, df);
        return quad_tol([&](double q) {
            return quad_tol([&](double sv) {
                return quad_tol([&](double z) { return std::pow(q, K) * sr_pdf_integrand(z, sv, q, k, df, c); }, -INF, INF,
                                SR_EPSABS, SR_EPSREL);
            }, 0.0, INF, SR_EPSABS, SR_EPSREL);
        }, 0.0, INF, SR_EPSABS, SR_EPSREL);
    }
};
Registrar r_studentized_range(new StudentizedRange);

}  // namespace
}  // namespace tsd
