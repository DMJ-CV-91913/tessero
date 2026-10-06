/*
 * scipy.stats: exponnorm, exponpow, exponweib, fatiguelife, fisk, foldcauchy, foldnorm, gausshyper, genexpon,
 * genextreme, gengamma, genhalflogistic, genhyperbolic, geninvgauss, genlogistic, gennorm, genpareto, gibrat,
 * gompertz (scipy/stats/_continuous_distns.py, with the helpers of scipy/stats/_stats.pyx and scipy.special's
 * lambertw), SciPy 1.17.1, private method by private method. See stats_dist.hpp.
 */
#include <exception>
#include "stats_dist.hpp"

#include <algorithm>
#include <complex>
#include <vector>

extern "C" {
#include "quadpack/quadpack.h"
}

namespace tsd {
namespace {

const double NORM_PDF_C = std::sqrt(2 * M_PI);
const double EULER = 0.577215664901532860606512090082402431042;   /* scipy.stats._constants */
const double ZETA3 = 1.202056903159594285399738161511449990765;
const double XMIN = std::numeric_limits<double>::min();          /* _XMIN = np.finfo(float).tiny */

inline double norm_pdf(double x) { return std::exp(-(x * x) / 2.0) / NORM_PDF_C; }
inline double sign(double x) { return std::isnan(x) ? x : (x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)); }

/* ---------------------------------------------------------------- scipy.special.lambertw(z, k=0, tol=1e-8) */

/* xsf::cevalpoly */
std::complex<double> cevalpoly(const double *coeffs, int degree, std::complex<double> z)
{
    double a = coeffs[0], b = coeffs[1];
    const double r = 2 * z.real(), s = std::norm(z);
    for (int j = 2; j < degree + 1; j++) {
        const double tmp = b;
        b = std::fma(-s, a, coeffs[j]);
        a = std::fma(r, a, tmp);
    }
    return z * a + b;
}

/* xsf::lambertw for the principal branch (xsf/lambertw.h) */
std::complex<double> lambertw0(std::complex<double> z, double tol)
{
    const double EXPN1 = 0.36787944117144232159553, OMEGA = 0.56714329040978387299997;
    if (std::isnan(z.real()) || std::isnan(z.imag())) return z;
    if (z.real() == INF) return z;
    if (z.real() == -INF) return -z + M_PI * std::complex<double>(0, 1);
    if (z == 0.0) return z;
    if (z == 1.0) return OMEGA;
    std::complex<double> w;
    if (std::abs(z + EXPN1) < 0.3) {
        const double coeffs[] = {-1.0 / 3.0, 1.0, -1.0};
        const std::complex<double> p = std::sqrt(2.0 * (M_E * z + 1.0));
        w = cevalpoly(coeffs, 2, p);
    } else if (-1.0 < z.real() && z.real() < 1.5 && std::abs(z.imag()) < 1.0 && -2.5 * std::abs(z.imag()) - 0.2 < z.real()) {
        const double num[] = {12.85106382978723404255, 12.34042553191489361902, 1.0};
        const double denom[] = {32.53191489361702127660, 14.34042553191489361702, 1.0};
        w = z * cevalpoly(num, 2, z) / cevalpoly(denom, 2, z);
    } else {
        w = std::log(z);
        w = w - std::log(w);
    }
    std::complex<double> ew, wew, wewz, wn;
    if (w.real() >= 0) {
        for (int i = 0; i < 100; i++) {
            ew = std::exp(-w);
            wewz = w - z * ew;
            wn = w - wewz / (w + 1.0 - (w + 2.0) * wewz / (2.0 * w + 2.0));
            if (std::abs(wn - w) <= tol * std::abs(wn)) return wn;
            w = wn;
        }
    } else {
        for (int i = 0; i < 100; i++) {
            ew = std::exp(w);
            wew = w * ew;
            wewz = wew - z;
            wn = w - wewz / (wew + ew - (w + 2.0) * wewz / (2.0 * w + 2.0));
            if (std::abs(wn - w) <= tol * std::abs(wn)) return wn;
            w = wn;
        }
    }
    return {NaN, NaN};
}

inline double lambertw_real(double z) { return lambertw0(std::complex<double>(z, 0.0), 1e-8).real(); }

/* ---------------------------------------------------------------- continuous distributions */

struct Exponnorm : Cont {
    Exponnorm() { name = "exponnorm"; shapes = "K"; nshape = 1; doc = "An exponentially modified Normal continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double K = s[0], invK = 1.0 / K;
        const double exparg = invK * (0.5 * invK - x);
        return exparg + sc::log_ndtr(x - invK) - std::log(K);
    }
    double cdf(double x, const double *s) const override
    {
        const double invK = 1.0 / s[0];
        const double expval = invK * (0.5 * invK - x);
        const double logprod = expval + sc::log_ndtr(x - invK);
        return sc::ndtr(x) - std::exp(logprod);
    }
    double sf(double x, const double *s) const override
    {
        const double invK = 1.0 / s[0];
        const double expval = invK * (0.5 * invK - x);
        const double logprod = expval + sc::log_ndtr(x - invK);
        return sc::ndtr(-x) + std::exp(logprod);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double K = s[0], K2 = K * K, opK2 = 1.0 + K2;
        const double skw = 2 * std::pow(K, 3.0) * std::pow(opK2, -1.5);
        const double krt = 6.0 * K2 * K2 * std::pow(opK2, -2.0);
        st.set(K, opK2, skw, krt);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_exponential(bg) * sa[0][i];
        for (int64_t i = 0; i < n; i++) out[i] = out[i] + random_standard_normal(bg);
    }
};
Registrar r_exponnorm(new Exponnorm);

struct Exponpow : Cont {
    Exponpow() { name = "exponpow"; shapes = "b"; nshape = 1; a = 0.0; doc = "An exponential power continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double b = s[0], xb = std::pow(x, b);
        return 1 + std::log(b) + sc::xlogy(b - 1.0, x) + xb - std::exp(xb);
    }
    double cdf(double x, const double *s) const override { return -sc::expm1(-sc::expm1(std::pow(x, s[0]))); }
    double sf(double x, const double *s) const override { return std::exp(-sc::expm1(std::pow(x, s[0]))); }
    double isf(double x, const double *s) const override { return std::pow(sc::log1p(-std::log(x)), 1. / s[0]); }
    double ppf(double q, const double *s) const override { return std::pow(sc::log1p(-sc::log1p(-q)), 1.0 / s[0]); }
};
Registrar r_exponpow(new Exponpow);

/* _pow1pm1(x, y) = (1 + x)**y - 1 */
inline double pow1pm1(double x, double y) { return std::expm1(sc::xlog1py(y, x)); }

struct Exponweib : Cont {
    Exponweib() { name = "exponweib"; shapes = "a, c"; nshape = 2; a = 0.0; doc = "An exponentiated Weibull continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double a = s[0], c = s[1];
        const double negxc = -std::pow(x, c);
        const double exm1c = -sc::expm1(negxc);
        return std::log(a) + std::log(c) + sc::xlogy(a - 1.0, exm1c) + negxc + sc::xlogy(c - 1.0, x);
    }
    double cdf(double x, const double *s) const override
    {
        const double exm1c = -sc::expm1(-std::pow(x, s[1]));
        return std::pow(exm1c, s[0]);
    }
    double ppf(double q, const double *s) const override
    {
        return std::pow(-sc::log1p(-std::pow(q, 1.0 / s[0])), 1.0 / s[1]);
    }
    double sf(double x, const double *s) const override { return -pow1pm1(-std::exp(-std::pow(x, s[1])), s[0]); }
    double isf(double p, const double *s) const override
    {
        return std::pow(-std::log(-pow1pm1(-p, 1 / s[0])), 1 / s[1]);
    }
};
Registrar r_exponweib(new Exponweib);

struct Fatiguelife : Cont {
    Fatiguelife() { name = "fatiguelife"; shapes = "c"; nshape = 1; a = 0.0; doc = "A fatigue-life (Birnbaum-Saunders) continuous random variable."; }
    /* _support_mask = _open_support_mask: pdf/logpdf are 0/-inf at the end points x = 0 and x = inf */
    double pdf(double x, const double *s) const override { return (a < x && x < b) ? std::exp(logpdf(x, s)) : 0.0; }
    double logpdf(double x, const double *s) const override
    {
        if (!(a < x && x < b)) return -INF;
        const double c = s[0];
        return std::log(x + 1) - (x - 1) * (x - 1) / (2.0 * x * (c * c)) - std::log(2 * c) -
               0.5 * (std::log(2 * M_PI) + 3 * std::log(x));
    }
    double cdf(double x, const double *s) const override { return sc::ndtr(1.0 / s[0] * (std::sqrt(x) - 1.0 / std::sqrt(x))); }
    double ppf(double q, const double *s) const override
    {
        const double tmp = s[0] * sc::ndtri(q);
        return 0.25 * sq(tmp + std::sqrt(tmp * tmp + 4));
    }
    double sf(double x, const double *s) const override { return sc::ndtr(-(1.0 / s[0] * (std::sqrt(x) - 1.0 / std::sqrt(x)))); }
    double isf(double q, const double *s) const override
    {
        const double tmp = -s[0] * sc::ndtri(q);
        return 0.25 * sq(tmp + std::sqrt(tmp * tmp + 4));
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0], c2 = c * c;
        const double mu = c2 / 2.0 + 1.0;
        const double den = 5.0 * c2 + 4.0;
        const double mu2 = c2 * den / 4.0;
        const double g1 = 4 * c * (11 * c2 + 6.0) / std::pow(den, 1.5);
        const double g2 = 6 * c2 * (93 * c2 + 40.0) / (den * den);
        st.set(mu, mu2, g1, g2);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_normal(bg);
        for (int64_t i = 0; i < n; i++) {
            const double x = 0.5 * sa[0][i] * out[i];
            const double x2 = x * x;
            out[i] = 1.0 + 2 * x2 + 2 * x * std::sqrt(1 + x2);
        }
    }
};
Registrar r_fatiguelife(new Fatiguelife);

/* burr_gen's methods at d = 1.0, which fisk_gen calls */
struct Fisk : Cont {
    Fisk() { name = "fisk"; shapes = "c"; nshape = 1; a = 0.0; doc = "A Fisk continuous random variable."; }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0], d = 1.0;
        if (x == 0) return c * d * std::pow(x, c * d - 1) / (1 + std::pow(x, c));
        return c * d * std::pow(x, -c - 1.0) / std::pow(1 + std::pow(x, -c), d + 1.0);
    }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0], d = 1.0;
        if (x == 0) return std::log(c) + std::log(d) + sc::xlogy(c * d - 1, x) - (d + 1) * sc::log1p(std::pow(x, c));
        return std::log(c) + std::log(d) + sc::xlogy(-c - 1, x) - sc::xlog1py(d + 1, std::pow(x, -c));
    }
    /* (...)**(-d) with the Python float d = 1.0: NumPy's reciprocal fast path */
    double cdf(double x, const double *s) const override { return 1.0 / (1 + std::pow(x, -s[0])); }
    double logcdf(double x, const double *s) const override { return sc::log1p(std::pow(x, -s[0])) * (-1.0); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return std::log1p(-(1.0 / (1 + std::pow(x, -s[0])))); }
    double ppf(double q, const double *s) const override { return std::pow(1.0 / q - 1, -1.0 / s[0]); }
    double isf(double q, const double *s) const override
    {
        const double q_ = sc::xlog1py(-1.0 / 1.0, -q);
        return std::pow(sc::expm1(q_), -1.0 / s[0]);
    }
    double munp(int n, const double *s) const override
    {
        const double c = s[0], d = 1.0;
        if (!(c > n)) return NaN;
        const double nc = 1. * n / c;
        return d * sc::beta(1.0 - nc, d + nc);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0], d = 1.0;
        double e[4];
        for (int k = 0; k < 4; k++) {
            const double nc = (double)(k + 1) / c;
            e[k] = sc::beta(d + nc, 1. - nc) * d;
        }
        const double e1 = e[0], e2 = e[1], e3 = e[2], e4 = e[3];
        const double mu = c > 1.0 ? e1 : NaN;
        const double mu2_if_c = e2 - mu * mu;
        const double mu2 = c > 2.0 ? mu2_if_c : NaN;
        const double g1 = c > 3.0 ? (e3 - 3 * e2 * e1 + 2 * std::pow(e1, 3.0)) / std::sqrt(std::pow(mu2_if_c, 3.0)) : NaN;
        const double g2 = c > 4.0 ? ((e4 - 4 * e3 * e1 + 6 * e2 * (e1 * e1) - 3 * std::pow(e1, 4.0)) / (mu2_if_c * mu2_if_c)) - 3
                                  : NaN;
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *s) const override { return 2 - std::log(s[0]); }
};
Registrar r_fisk(new Fisk);

struct Foldcauchy : Cont {
    Foldcauchy() { name = "foldcauchy"; shapes = "c"; nshape = 1; a = 0.0; doc = "A folded Cauchy continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0; }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0];
        return 1.0 / M_PI * (1.0 / (1 + (x - c) * (x - c)) + 1.0 / (1 + (x + c) * (x + c)));
    }
    double cdf(double x, const double *s) const override { return 1.0 / M_PI * (std::atan(x - s[0]) + std::atan(x + s[0])); }
    double sf(double x, const double *s) const override { return (std::atan2(1, x - s[0]) + std::atan2(1, x + s[0])) / M_PI; }
    void stats(const double *, int, Stats4 &st) const override { st.set(INF, INF, NaN, NaN); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* abs(cauchy.rvs(loc=c)): cauchy draws by inversion of a uniform */
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_uniform(bg);
        for (int64_t i = 0; i < n; i++) out[i] = std::fabs(sc::_cauchy_ppf(out[i], 0, 1) * 1.0 + sa[0][i]);
    }
};
Registrar r_foldcauchy(new Foldcauchy);

struct Foldnorm : Cont {
    Foldnorm() { name = "foldnorm"; shapes = "c"; nshape = 1; a = 0.0; doc = "A folded normal continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0; }
    double pdf(double x, const double *s) const override { return norm_pdf(x + s[0]) + norm_pdf(x - s[0]); }
    double cdf(double x, const double *s) const override
    {
        const double sqrt_two = std::sqrt(2.0), c = s[0];
        return 0.5 * (sc::erf((x - c) / sqrt_two) + sc::erf((x + c) / sqrt_two));
    }
    double sf(double x, const double *s) const override { return sc::ndtr(-(x - s[0])) + sc::ndtr(-(x + s[0])); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0], c2 = c * c;
        const double expfac = std::exp(-0.5 * c2) / std::sqrt(2. * M_PI);
        const double mu = 2. * expfac + c * sc::erf(c / std::sqrt(2.0));
        const double mu2 = c2 + 1 - mu * mu;
        double g1 = 2. * (mu * mu * mu - c2 * mu - expfac);
        g1 /= std::pow(mu2, 1.5);
        double g2 = c2 * (c2 + 6.) + 3 + 8. * expfac * mu;
        g2 += (2. * (c2 - 3.) - 3. * (mu * mu)) * (mu * mu);
        g2 = g2 / (mu2 * mu2) - 3.;
        st.set(mu, mu2, g1, g2);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = std::fabs(random_standard_normal(bg) + sa[0][i]);
    }
};
Registrar r_foldnorm(new Foldnorm);

struct Gausshyper : Cont {
    Gausshyper() { name = "gausshyper"; shapes = "a, b, c, z"; nshape = 4; a = 0.0; b = 1.0; doc = "A Gauss hypergeometric continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[1] > 0 && s[2] == s[2] && s[3] > -1; }
    double pdf(double x, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2], z = s[3];
        const double normalization_constant = sc::beta(a, b) * sc::hyp2f1(c, a, a + b, -z);
        return 1. / normalization_constant * std::pow(x, a - 1.) * std::pow(1. - x, b - 1.0) / std::pow(1.0 + z * x, c);
    }
    double munp(int n, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2], z = s[3];
        const double fac = sc::beta(n + a, b) / sc::beta(a, b);
        const double num = sc::hyp2f1(c, a + n, a + b + n, -z);
        const double den = sc::hyp2f1(c, a, a + b, -z);
        return fac * num / den;
    }
};
Registrar r_gausshyper(new Gausshyper);

struct Genexpon : Cont {
    Genexpon() { name = "genexpon"; shapes = "a, b, c"; nshape = 3; a = 0.0; doc = "A generalized exponential continuous random variable."; }
    double pdf(double x, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2];
        return (a + b * (-sc::expm1(-c * x))) * std::exp((-a - b) * x + b * (-sc::expm1(-c * x)) / c);
    }
    double logpdf(double x, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2];
        return std::log(a + b * (-sc::expm1(-c * x))) + (-a - b) * x + b * (-sc::expm1(-c * x)) / c;
    }
    double cdf(double x, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2];
        return -sc::expm1((-a - b) * x + b * (-sc::expm1(-c * x)) / c);
    }
    double ppf(double p, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2];
        const double sum = a + b;
        const double t = (b - c * std::log1p(-p)) / sum;
        return (t + lambertw_real(-b / sum * std::exp(-t))) / c;
    }
    double sf(double x, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2];
        return std::exp((-a - b) * x + b * (-sc::expm1(-c * x)) / c);
    }
    double isf(double p, const double *s) const override
    {
        const double a = s[0], b = s[1], c = s[2];
        const double sum = a + b;
        const double t = (b - c * std::log(p)) / sum;
        return (t + lambertw_real(-b / sum * std::exp(-t))) / c;
    }
};
Registrar r_genexpon(new Genexpon);

struct Genextreme : Cont {
    Genextreme() { name = "genextreme"; shapes = "c"; nshape = 1; doc = "A generalized extreme value continuous random variable."; }
    bool argcheck(const double *s) const override { return std::isfinite(s[0]); }
    void get_support(const double *s, double &lo, double &hi) const override
    {
        const double c = s[0];
        hi = c > 0 ? 1.0 / std::max(c, XMIN) : INF;
        lo = c < 0 ? 1.0 / std::min(c, -XMIN) : -INF;
    }
    static double loglogcdf(double x, double c) { return (x == x && c != 0) ? sc::log1p(-c * x) / c : -x; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0];
        const double cx = (x == x && c != 0) ? c * x : 0.0;
        const double logex2 = sc::log1p(-cx);
        double logpex2 = loglogcdf(x, c);
        const double pex2 = std::exp(logpex2);
        if (c == 0 && x == -INF) logpex2 = 0.0;
        double logpdf = !(cx == 1 || cx == -INF) ? -pex2 + logpex2 - logex2 : -INF;
        if (c == 1 && x == 1) logpdf = 0.0;
        return logpdf;
    }
    double logcdf(double x, const double *s) const override { return -std::exp(loglogcdf(x, s[0])); }
    double cdf(double x, const double *s) const override { return std::exp(logcdf(x, s)); }
    double sf(double x, const double *s) const override { return -sc::expm1(logcdf(x, s)); }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0], x = -std::log(-std::log(q));
        return (x == x && c != 0) ? -sc::expm1(-c * x) / c : x;
    }
    double isf(double q, const double *s) const override
    {
        const double c = s[0], x = -std::log(-sc::log1p(-q));
        return (x == x && c != 0) ? -sc::expm1(-c * x) / c : x;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0];
        const double g1 = sc::gamma(1 * c + 1), g2 = sc::gamma(2 * c + 1), g3 = sc::gamma(3 * c + 1), g4 = sc::gamma(4 * c + 1);
        const double g2mg12 = std::fabs(c) < 1e-7 ? sq(c * M_PI) / 6.0 : g2 - g1 * g1;
        const double gam2k = std::fabs(c) >= 1e-7 ? sc::expm1(sc::gammaln(2.0 * c + 1.0) - 2 * sc::gammaln(c + 1.0)) / (c * c)
                                                  : std::pow(M_PI, 2.0) / 6.0;
        const double eps = 1e-14;
        const double gamk = std::fabs(c) >= eps ? sc::expm1(sc::gammaln(c + 1)) / c : -EULER;
        const double m = c < -1.0 ? NaN : -gamk;
        const double v = c < -0.5 ? NaN : g1 * g1 * gam2k;
        const double sk_fill = 12 * std::sqrt(6.0) * ZETA3 / std::pow(M_PI, 3.0);
        double sk;
        if (std::fabs(c) > std::pow(eps, 0.29))
            sk = c >= -1. / 3 ? sign(c) * (-g3 + (g2 + 2 * g2mg12) * g1) / std::pow(g2mg12, 1.5) : NaN;
        else
            sk = sk_fill;
        double ku;
        if (std::fabs(c) > std::pow(eps, 0.23))
            ku = c >= -1. / 4 ? (g4 + (-4 * g3 + 3 * (g2 + g2mg12) * g1) * g1) / (g2mg12 * g2mg12) - 3 : NaN;
        else
            ku = 12.0 / 5.0;
        st.set(m, v, sk, ku);
    }
    double munp(int n, const double *s) const override
    {
        const double c = s[0];
        std::vector<double> terms((size_t)n + 1);
        for (int k = 0; k <= n; k++) terms[(size_t)k] = sc::binom(n, k) * (double)(k % 2 ? -1 : 1) * sc::gamma(c * k + 1);
        /* c**n for an array c: n = 1 is c itself, 2 squares, others pow */
        const double cn = n == 1 ? c : (n == 2 ? c * c : std::pow(c, (double)n));
        const double vals = 1.0 / cn * tsr_psum(terms.data(), n + 1);
        return c * n > -1 ? vals : INF;
    }
    double entropy(const double *s) const override { return EULER * (1 - s[0]) + 1; }
};
Registrar r_genextreme(new Genextreme);

const double NORM_ENTROPY = 0.5 * (std::log(2 * M_PI) + 1);

struct Gengamma : Cont {
    Gengamma() { name = "gengamma"; shapes = "a, c"; nshape = 2; a = 0.0; doc = "A generalized gamma continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[1] != 0; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double a = s[0], c = s[1];
        if (!(x != 0 || c > 0)) return -INF;
        return std::log(std::fabs(c)) + sc::xlogy(c * a - 1, x) - std::pow(x, c) - sc::gammaln(a);
    }
    double cdf(double x, const double *s) const override
    {
        const double a = s[0], c = s[1], xc = std::pow(x, c);
        return c > 0 ? sc::gammainc(a, xc) : sc::gammaincc(a, xc);
    }
    double sf(double x, const double *s) const override
    {
        const double a = s[0], c = s[1], xc = std::pow(x, c);
        return c > 0 ? sc::gammaincc(a, xc) : sc::gammainc(a, xc);
    }
    double ppf(double q, const double *s) const override
    {
        const double a = s[0], c = s[1];
        return std::pow(c > 0 ? sc::gammaincinv(a, q) : sc::gammainccinv(a, q), 1.0 / c);
    }
    double isf(double q, const double *s) const override
    {
        const double a = s[0], c = s[1];
        return std::pow(c > 0 ? sc::gammainccinv(a, q) : sc::gammaincinv(a, q), 1.0 / c);
    }
    double munp(int n, const double *s) const override { return sc::poch(s[0], n * 1.0 / s[1]); }
    double entropy(const double *s) const override
    {
        const double a = s[0], c = s[1];
        if (a >= 200)
            return NORM_ENTROPY - std::log(a) / 2 - std::log(std::fabs(c)) + (1.0 / a) / 6 - std::pow(a, -3.) / 90 +
                   (std::log(a) - (1.0 / a) / 2 - std::pow(a, -2.) / 12 + std::pow(a, -4.) / 120) / c;
        const double val = sc::psi(a);
        const double A = a * (1 - val) + val / c;
        const double B = sc::gammaln(a) - std::log(std::fabs(c));
        return A + B;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_gamma(bg, sa[0][i]);
        for (int64_t i = 0; i < n; i++) out[i] = std::pow(out[i], 1. / sa[1][i]);
    }
};
Registrar r_gengamma(new Gengamma);

struct Genhalflogistic : Cont {
    Genhalflogistic() { name = "genhalflogistic"; shapes = "c"; nshape = 1; a = 0.0; doc = "A generalized half-logistic continuous random variable."; }
    void get_support(const double *s, double &lo, double &hi) const override { lo = a; hi = 1.0 / s[0]; }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0], limit = 1.0 / c;
        const double tmp = 1 - c * x;
        const double tmp0 = std::pow(tmp, limit - 1);
        const double tmp2 = tmp0 * tmp;
        return 2 * tmp0 / sq(1 + tmp2);
    }
    double cdf(double x, const double *s) const override
    {
        const double c = s[0], limit = 1.0 / c;
        const double tmp2 = std::pow(1 - c * x, limit);
        return (1.0 - tmp2) / (1 + tmp2);
    }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0];
        return 1.0 / c * (1 - std::pow((1.0 - q) / (1.0 + q), c));
    }
    double entropy(const double *s) const override { return 2 - (2 * s[0] + 1) * std::log(2.0); }
};
Registrar r_genhalflogistic(new Genhalflogistic);

/* scipy.integrate.quad of a LowLevelCallable with explicit tolerances (limit 50), for _integrate_pdf */
thread_local double (*g_llc)(double, const double *) = nullptr;
thread_local const double *g_llc_data = nullptr;
thread_local std::exception_ptr g_llc_err;   /* an exception from the integrand, rethrown after QUADPACK (C) returns */
double llc_thunk(double *x)
{
    if (g_llc_err) return std::numeric_limits<double>::quiet_NaN();
    try {
        return g_llc(*x, g_llc_data);
    } catch (...) {
        g_llc_err = std::current_exception();
        return std::numeric_limits<double>::quiet_NaN();
    }
}

double quad_llc(double (*f)(double, const double *), const double *data, double a, double b, double epsabs, double epsrel)
{
    if (a == b) return 0.0;
    const bool flip = b < a;
    const double lo = std::min(a, b), hi = std::max(a, b);
    enum { LIMIT = 50 };
    double alist[LIMIT], blist[LIMIT], rlist[LIMIT], elist[LIMIT];
    int iord[LIMIT];
    double result = 0.0, abserr = 0.0;
    int neval = 0, ier = 6, last = 0;
    auto saved_f = g_llc;
    auto saved_d = g_llc_data;
    std::exception_ptr saved_e = g_llc_err;
    g_llc_err = nullptr;
    g_llc = f;
    g_llc_data = data;
    if (hi != INF && lo != -INF) {
        dqagse(llc_thunk, lo, hi, epsabs, epsrel, LIMIT, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);
    } else {
        int inf;
        double bound;
        if (hi == INF && lo != -INF) { inf = 1; bound = lo; }
        else if (hi == INF && lo == -INF) { inf = 2; bound = 0.0; }
        else { inf = -1; bound = hi; }
        dqagie(llc_thunk, bound, inf, epsabs, epsrel, LIMIT, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);
    }
    g_llc = saved_f;
    g_llc_data = saved_d;
    std::exception_ptr e = g_llc_err;
    g_llc_err = saved_e;
    if (e) std::rethrow_exception(e);
    if (ier == 6) fail("quad: the input is invalid");
    return flip ? -result : result;
}

/* _stats.pyx: _log_norming_constant, _genhyperbolic_logpdf_kernel */
double gh_log_norming_constant(double p, double a, double b)
{
    const double t1 = (a + b) * (a - b);
    const double t2 = p * 0.5 * std::log(t1);
    const double t3 = 0.5 * std::log(2 * M_PI);
    const double t4 = (p - 0.5) * std::log(a);
    const double t5 = std::sqrt(t1);
    const double t6 = std::log(sc::kve(p, t5)) - t5;
    return t2 - t3 - t4 - t6;
}

double gh_logpdf_kernel(double x, double p, double a, double b)
{
    const double t1 = gh_log_norming_constant(p, a, b);
    const double t2 = std::sqrt(1.0 + x * x);
    const double t3 = (p - 0.5) * std::log(t2);
    const double t4 = std::log(sc::kve(p - 0.5, a * t2)) - a * t2;
    const double t5 = b * x;
    return t1 + t3 + t4 + t5;
}

double gh_pdf_llc(double x, const double *d) { return std::exp(gh_logpdf_kernel(x, d[0], d[1], d[2])); }

/* _stats.pyx: _geninvgauss_logpdf_kernel, _geninvgauss_pdf */
double gig_logpdf_kernel(double x, double p, double b)
{
    if (x <= 0) return -INF;
    const double z = sc::kve(p, b);
    if (std::isinf(z)) return NaN;
    const double c = -std::log(2) - std::log(z) + b;
    return c + (p - 1) * std::log(x) - b * (x + 1 / x) / 2;
}

double gig_pdf_llc(double x, const double *d)
{
    if (x <= 0) return 0.;
    return std::exp(gig_logpdf_kernel(x, d[0], d[1]));
}

struct Geninvgauss : Cont {
    Geninvgauss() { name = "geninvgauss"; shapes = "p, b"; nshape = 2; a = 0.0; doc = "A Generalized Inverse Gaussian continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] == s[0] && s[1] > 0; }
    double logpdf(double x, const double *s) const override { return gig_logpdf_kernel(x, s[0], s[1]); }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double cdf(double x, const double *s) const override
    {
        double lo, hi;
        get_support(s, lo, hi);
        const double data[2] = {s[0], s[1]};
        return quad([&](double t) { return gig_pdf_llc(t, data); }, lo, x);
    }
    static double logquasipdf(double x, double p, double b) { return x > 0 ? (p - 1) * std::log(x) - b * (x + 1 / x) / 2 : -INF; }
    static double mode(double p, double b)
    {
        if (p < 1) return b / (std::sqrt(std::pow(p - 1, 2.0) + std::pow(b, 2.0)) + 1 - p);
        return (std::sqrt(std::pow(1 - p, 2.0) + std::pow(b, 2.0)) - (1 - p)) / b;
    }
    /* _rvs_scalar: N variates for one (p, b) */
    static bool rvs_scalar(double p, double b, int64_t N, bitgen_t *bg, double *x)
    {
        bool invert_res = false;
        if (p < 0) {
            p = -p;
            invert_res = true;
        }
        const double m = mode(p, b);
        bool ratio_unif = true, mode_shift = false;
        if (p >= 1 || b > 1) mode_shift = true;
        else if (b >= std::min(0.5, 2 * std::sqrt(1 - p) / 3)) mode_shift = false;
        else ratio_unif = false;
        int64_t simulated = 0;
        std::vector<double> u, v;
        if (ratio_unif) {
            double vmin, vmax, umax, c, lm = 0.0;
            if (mode_shift) {
                const double a2 = -2 * (p + 1) / b - m;
                const double a1 = 2 * m * (p - 1) / b - 1;
                const double p1 = a1 - std::pow(a2, 2.0) / 3;
                const double q1 = 2 * std::pow(a2, 3.0) / 27 - a2 * a1 / 3 + m;
                const double phi = std::acos(-q1 * std::sqrt(-27 / std::pow(p1, 3.0)) / 2);
                const double s1 = -std::sqrt(-4 * p1 / 3);
                const double root1 = s1 * std::cos(phi / 3 + M_PI / 3) - a2 / 3;
                const double root2 = -s1 * std::cos(phi / 3) - a2 / 3;
                lm = logquasipdf(m, p, b);
                const double d1 = logquasipdf(root1, p, b) - lm;
                const double d2 = logquasipdf(root2, p, b) - lm;
                vmin = (root1 - m) * std::exp(0.5 * d1);
                vmax = (root2 - m) * std::exp(0.5 * d2);
                umax = 1;
                c = m;
            } else {
                umax = std::exp(0.5 * logquasipdf(m, p, b));
                const double xplus = ((1 + p) + std::sqrt(std::pow(1 + p, 2.0) + std::pow(b, 2.0))) / b;
                vmin = 0;
                vmax = xplus * std::exp(0.5 * logquasipdf(xplus, p, b));
                c = 0;
            }
            if (vmin >= vmax) { fail("vmin must be smaller than vmax."); return false; }
            if (umax <= 0) { fail("umax must be positive."); return false; }
            int64_t i = 1;
            while (simulated < N) {
                const int64_t k = N - simulated;
                u.resize((size_t)k);
                v.resize((size_t)k);
                for (int64_t j = 0; j < k; j++) u[(size_t)j] = umax * random_standard_uniform(bg);
                for (int64_t j = 0; j < k; j++) v[(size_t)j] = random_standard_uniform(bg);
                for (int64_t j = 0; j < k; j++) {
                    const double vv = vmin + (vmax - vmin) * v[(size_t)j];
                    const double r = vv / u[(size_t)j] + c;
                    const double lq = mode_shift ? logquasipdf(r, p, b) - lm : logquasipdf(r, p, b);
                    if (2 * std::log(u[(size_t)j]) <= lq) x[simulated++] = r;
                }
                if (simulated == 0 && i * N >= 50000) {
                    fail("Not a single random variate could be generated. Sampling does not appear to work for the provided parameters.");
                    return false;
                }
                i += 1;
            }
        } else {
            const double x0 = b / (1 - p);
            const double xs = std::max(x0, 2 / b);
            const double k1 = std::exp(logquasipdf(m, p, b));
            const double A1 = k1 * x0;
            double k2, A2;
            if (x0 < 2 / b) {
                k2 = std::exp(-b);
                if (p > 0) A2 = k2 * (std::pow(2 / b, p) - std::pow(x0, p)) / p;
                else A2 = k2 * std::log(2 / (b * b));
            } else {
                k2 = 0;
                A2 = 0;
            }
            const double k3 = std::pow(xs, p - 1);
            const double A3 = 2 * k3 * std::exp(-xs * b / 2) / b;
            const double A = A1 + A2 + A3;
            while (simulated < N) {
                const int64_t k = N - simulated;
                u.resize((size_t)k);
                v.resize((size_t)k);
                for (int64_t j = 0; j < k; j++) u[(size_t)j] = random_standard_uniform(bg);
                for (int64_t j = 0; j < k; j++) v[(size_t)j] = A * random_standard_uniform(bg);
                for (int64_t j = 0; j < k; j++) {
                    const double vj = v[(size_t)j];
                    double r, h;
                    if (vj <= A1) {
                        r = x0 * vj / A1;
                        h = k1;
                    } else if (vj <= A1 + A2) {
                        if (p > 0) r = std::pow(std::pow(x0, p) + (vj - A1) * p / k2, 1 / p);
                        else r = b * std::exp((vj - A1) * std::exp(b));
                        h = k2 * std::pow(r, p - 1);
                    } else {
                        const double z = std::exp(-xs * b / 2) - b * (vj - A1 - A2) / (2 * k3);
                        r = -2 / b * std::log(z);
                        h = k3 * std::exp(-r * b / 2);
                    }
                    if (std::log(u[(size_t)j] * h) <= logquasipdf(r, p, b)) x[simulated++] = r;
                }
            }
        }
        if (invert_res)
            for (int64_t j = 0; j < N; j++) x[j] = 1 / x[j];
        return true;
    }
    /* _rvs: one block for scalar parameters; otherwise one variate per element (np.nditer over p, b) */
    static bool rvs_gig(bitgen_t *bg, int64_t n, const double *p, const double *b, double *out)
    {
        bool scalar = true;
        for (int64_t i = 1; i < n && scalar; i++)
            if (!(p[i] == p[0] && b[i] == b[0])) scalar = false;
        if (scalar) return rvs_scalar(p[0], b[0], n, bg, out);
        for (int64_t i = 0; i < n; i++)
            if (!rvs_scalar(p[i], b[i], 1, bg, out + i)) return false;
        return true;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        if (!rvs_gig(bg, n, sa[0], sa[1], out))
            for (int64_t i = 0; i < n; i++) out[i] = NaN;
    }
    double munp(int n, const double *s) const override
    {
        const double num = sc::kve(s[0] + n, s[1]);
        const double denom = sc::kve(s[0], s[1]);
        if (std::isinf(num) || std::isinf(denom)) return NaN;
        return num / denom;
    }
};
Registrar r_geninvgauss(new Geninvgauss);

struct Genhyperbolic : Cont {
    Genhyperbolic() { name = "genhyperbolic"; shapes = "p, a, b"; nshape = 3; doc = "A generalized hyperbolic continuous random variable."; }
    bool argcheck(const double *s) const override
    {
        const double p = s[0], a = s[1], b = s[2];
        return (std::fabs(b) < a && p >= 0) || (std::fabs(b) <= a && p < 0);
    }
    double logpdf(double x, const double *s) const override { return gh_logpdf_kernel(x, s[0], s[1], s[2]); }
    double pdf(double x, const double *s) const override { return std::exp(gh_logpdf_kernel(x, s[0], s[1], s[2])); }
    static double integrate_pdf(double x0, double x1, double p, double a, double b)
    {
        const double data[3] = {p, a, b};
        const double d = std::sqrt((a + b) * (a - b));
        const double mean = b / d * sc::kv(p + 1, d) / sc::kv(p, d);
        const double epsrel = 1e-10, epsabs = 0;
        double intgrl;
        if (x0 < mean && mean < x1)
            intgrl = quad_llc(gh_pdf_llc, data, x0, mean, epsabs, epsrel) + quad_llc(gh_pdf_llc, data, mean, x1, epsabs, epsrel);
        else
            intgrl = quad_llc(gh_pdf_llc, data, x0, x1, epsabs, epsrel);
        /* max(0.0, min(1.0, intgrl)) with Python's min/max (NaN -> 1.0) */
        const double m = intgrl < 1.0 ? intgrl : 1.0;
        return 0.0 < m ? m : 0.0;
    }
    double cdf(double x, const double *s) const override { return integrate_pdf(-INF, x, s[0], s[1], s[2]); }
    double sf(double x, const double *s) const override { return integrate_pdf(x, INF, s[0], s[1], s[2]); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* geninvgauss.rvs(p=p, b=t2, scale=t3), then norm.rvs */
        std::vector<double> t2(n), t3(n);
        for (int64_t i = 0; i < n; i++) {
            const double t1 = std::pow(sa[1][i], 2.0) - std::pow(sa[2][i], 2.0);
            t2[i] = std::pow(t1, 0.5);
            t3[i] = std::pow(t1, -0.5);
        }
        std::vector<double> gig(n);
        if (!Geninvgauss::rvs_gig(bg, n, sa[0], t2.data(), gig.data())) {
            for (int64_t i = 0; i < n; i++) out[i] = NaN;
            return;
        }
        for (int64_t i = 0; i < n; i++) gig[i] = gig[i] * t3[i] + 0.0;
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_normal(bg);
        for (int64_t i = 0; i < n; i++) out[i] = sa[2][i] * gig[i] + std::sqrt(gig[i]) * out[i];
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double p = s[0], a = s[1], b = s[2];
        double t1 = std::pow(a, 2.0) - std::pow(b, 2.0);
        t1 = std::pow(t1, 0.5);
        const double t2 = std::pow(1.0, 2.0) * std::pow(t1, -1.0);
        const double b0 = sc::kv(p + 0.0, t1), b1 = sc::kv(p + 1.0, t1), b2 = sc::kv(p + 2.0, t1), b3 = sc::kv(p + 3.0, t1),
                     b4 = sc::kv(p + 4.0, t1);
        const double r1 = b1 / b0, r2 = b2 / b0, r3 = b3 / b0, r4 = b4 / b0;
        const double m = b * t2 * r1;
        const double v = t2 * r1 + std::pow(b, 2.0) * std::pow(t2, 2.0) * (r2 - std::pow(r1, 2.0));
        const double m3e = std::pow(b, 3.0) * std::pow(t2, 3.0) * (r3 - 3 * b2 * b1 * std::pow(b0, -2.0) + 2 * std::pow(r1, 3.0)) +
                           3 * b * std::pow(t2, 2.0) * (r2 - std::pow(r1, 2.0));
        const double sk = m3e * std::pow(v, -3.0 / 2);
        const double m4e = std::pow(b, 4.0) * std::pow(t2, 4.0) *
                               (r4 - 4 * b3 * b1 * std::pow(b0, -2.0) + 6 * b2 * std::pow(b1, 2.0) * std::pow(b0, -3.0) -
                                3 * std::pow(r1, 4.0)) +
                           std::pow(b, 2.0) * std::pow(t2, 3.0) * (6 * r3 - 12 * b2 * b1 * std::pow(b0, -2.0) + 6 * std::pow(r1, 3.0)) +
                           3 * std::pow(t2, 2.0) * r2;
        const double k = m4e * std::pow(v, -2.0) - 3;
        st.set(m, v, sk, k);
    }
};
Registrar r_genhyperbolic(new Genhyperbolic);

struct Genlogistic : Cont {
    Genlogistic() { name = "genlogistic"; shapes = "c"; nshape = 1; doc = "A generalized logistic continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0];
        const double mult = -(c - 1) * (double)(x < 0) - 1;
        const double absx = std::fabs(x);
        return std::log(c) + mult * absx - (c + 1) * sc::log1p(std::exp(-absx));
    }
    double cdf(double x, const double *s) const override { return std::pow(1 + std::exp(-x), -s[0]); }
    double logcdf(double x, const double *s) const override { return -s[0] * std::log1p(std::exp(-x)); }
    double ppf(double q, const double *s) const override { return -std::log(sc::powm1(q, -1.0 / s[0])); }
    double sf(double x, const double *s) const override { return -sc::expm1(logcdf(x, s)); }
    double isf(double q, const double *s) const override { return ppf(1 - q, s); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0];
        const double mu = EULER + sc::psi(c);
        const double mu2 = M_PI * M_PI / 6.0 + sc::_zeta(2, c);
        double g1 = -2 * sc::_zeta(3, c) + 2 * ZETA3;
        g1 /= std::pow(mu2, 1.5);
        double g2 = std::pow(M_PI, 4.0) / 15.0 + 6 * sc::_zeta(4, c);
        g2 /= mu2 * mu2;
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *s) const override
    {
        const double c = s[0];
        if (c < 8e6) return -std::log(c) + sc::psi(c + 1) + EULER + 1;
        return 1 / (2 * c) + EULER + 1;
    }
};
Registrar r_genlogistic(new Genlogistic);

struct Gennorm : Cont {
    Gennorm() { name = "gennorm"; shapes = "beta"; nshape = 1; doc = "A generalized normal continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double beta = s[0];
        return std::log(0.5 * beta) - sc::gammaln(1.0 / beta) - std::pow(std::fabs(x), beta);
    }
    double cdf(double x, const double *s) const override
    {
        const double beta = s[0], c = 0.5 * sign(x);
        return (0.5 + c) - c * sc::gammaincc(1.0 / beta, std::pow(std::fabs(x), beta));
    }
    double ppf(double x, const double *s) const override
    {
        const double beta = s[0], c = sign(x - 0.5);
        return c * std::pow(sc::gammainccinv(1.0 / beta, (1.0 + c) - 2.0 * c * x), 1.0 / beta);
    }
    double sf(double x, const double *s) const override { return cdf(-x, s); }
    double isf(double x, const double *s) const override { return -ppf(x, s); }
    double munp(int n, const double *s) const override
    {
        const double beta = s[0];
        if (n == 0) return 1.;
        if (n % 2 == 0) {
            const double c1 = sc::gammaln(1.0 / beta), cn = sc::gammaln((n + 1.0) / beta);
            return std::exp(cn - c1);
        }
        return 0.;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double beta = s[0];
        const double c1 = sc::gammaln(1.0 / beta), c3 = sc::gammaln(3.0 / beta), c5 = sc::gammaln(5.0 / beta);
        st.set(0., std::exp(c3 - c1), 0., std::exp(c5 + c1 - 2.0 * c3) - 3.);
    }
    double entropy(const double *s) const override
    {
        const double beta = s[0];
        return 1. / beta - std::log(.5 * beta) + sc::gammaln(1. / beta);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_gamma(bg, 1 / sa[0][i], 1.0);
        for (int64_t i = 0; i < n; i++) out[i] = std::pow(out[i], 1 / sa[0][i]);
        for (int64_t i = 0; i < n; i++)
            if (random_standard_uniform(bg) < 0.5) out[i] = -out[i];
    }
};
Registrar r_gennorm(new Gennorm);

struct Genpareto : Cont {
    Genpareto() { name = "genpareto"; shapes = "c"; nshape = 1; a = 0.0; doc = "A generalized Pareto continuous random variable."; }
    bool argcheck(const double *s) const override { return std::isfinite(s[0]); }
    void get_support(const double *s, double &lo, double &hi) const override
    {
        const double c = s[0];
        lo = a;
        hi = c < 0 ? -1. / c : INF;
    }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0];
        return (x == x && c != 0) ? -sc::xlog1py(c + 1., c * x) / c : -x;
    }
    double cdf(double x, const double *s) const override { return -sc::inv_boxcox1p(-x, -s[0]); }
    double sf(double x, const double *s) const override { return sc::inv_boxcox(-x, -s[0]); }
    double logsf(double x, const double *s) const override
    {
        const double c = s[0];
        return (x == x && c != 0) ? -sc::log1p(c * x) / c : -x;
    }
    double ppf(double q, const double *s) const override { return -sc::boxcox1p(-q, -s[0]); }
    double isf(double q, const double *s) const override { return -sc::boxcox(q, -s[0]); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double xi = s[0];
        if (moments & MOM_M) st.set(0, xi < 1 ? 1 / (1 - xi) : INF);
        if (moments & MOM_V) st.set(1, xi < 1. / 2 ? 1 / sq(1 - xi) / (1 - 2 * xi) : NaN);
        if (moments & MOM_S) st.set(2, xi < 1. / 3 ? 2 * (1 + xi) * std::sqrt(1 - 2 * xi) / (1 - 3 * xi) : NaN);
        if (moments & MOM_K)
            st.set(3, xi < 1. / 4 ? 3 * (1 - 2 * xi) * (2 * (xi * xi) + xi + 3) / (1 - 3 * xi) / (1 - 4 * xi) - 3 : NaN);
    }
    double munp(int n, const double *s) const override
    {
        const double c = s[0];
        if (c == 0) return sc::gamma(n + 1);
        double val = 0.0;
        for (int k = 0; k <= n; k++) val = val + sc::binom(n, k) * (double)(k % 2 ? -1 : 1) / (1.0 - c * k);
        /* (-1.0 / c) ** n for an array: n = 1 is the base itself, 2 squares, others pow */
        const double base = -1.0 / c;
        const double pw = n == 1 ? base : (n == 2 ? base * base : std::pow(base, (double)n));
        return c * n < 1 ? val * pw : INF;
    }
    double entropy(const double *s) const override { return 1. + s[0]; }
};
Registrar r_genpareto(new Genpareto);

struct Gibrat : Cont {
    Gibrat() { name = "gibrat"; a = 0.0; doc = "A Gibrat continuous random variable."; }
    /* _support_mask = _open_support_mask: pdf/logpdf are 0/-inf at the end points x = 0 and x = inf */
    double pdf(double x, const double *s) const override { return (a < x && x < b) ? std::exp(logpdf(x, s)) : 0.0; }
    double logpdf(double x, const double *) const override
    {
        if (!(a < x && x < b)) return -INF;
        /* _lognorm_logpdf(x, 1.0) */
        const double s = 1.0;
        if (x == 0) return -INF;
        return -sq(std::log(x)) / (2 * (s * s)) - std::log(s * x * std::sqrt(2 * M_PI));
    }
    double cdf(double x, const double *) const override { return sc::ndtr(std::log(x)); }
    double ppf(double q, const double *) const override { return std::exp(sc::ndtri(q)); }
    double sf(double x, const double *) const override { return sc::ndtr(-std::log(x)); }
    double isf(double p, const double *) const override { return std::exp(-sc::ndtri(p)); }
    void stats(const double *, int, Stats4 &st) const override
    {
        const double p = M_E;
        const double mu = std::sqrt(p);
        const double mu2 = p * (p - 1);
        const double g1 = std::sqrt(p - 1) * (2 + p);
        double g2 = 0.0;                             /* np.polyval([1, 2, 3, 0, -6.0], p) */
        for (double c : {1.0, 2.0, 3.0, 0.0, -6.0}) g2 = g2 * p + c;
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *) const override { return 0.5 * std::log(2 * M_PI) + 0.5; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = std::exp(random_standard_normal(bg));
    }
};
Registrar r_gibrat(new Gibrat);

struct Gompertz : Cont {
    Gompertz() { name = "gompertz"; shapes = "c"; nshape = 1; a = 0.0; doc = "A Gompertz (or truncated Gumbel) continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override { return std::log(s[0]) + x - s[0] * sc::expm1(x); }
    double cdf(double x, const double *s) const override { return -sc::expm1(-s[0] * sc::expm1(x)); }
    double ppf(double q, const double *s) const override { return sc::log1p(-1.0 / s[0] * sc::log1p(-q)); }
    double sf(double x, const double *s) const override { return std::exp(-s[0] * sc::expm1(x)); }
    double isf(double p, const double *s) const override { return sc::log1p(-std::log(p) / s[0]); }
    double entropy(const double *s) const override { return 1.0 - std::log(s[0]) - sc::_scaled_exp1(s[0]) / s[0]; }
};
Registrar r_gompertz(new Gompertz);

}  // namespace
}  // namespace tsd
