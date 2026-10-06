/*
 * scipy.stats: laplace, laplace_asymmetric, levy, levy_l, loggamma, logistic, loglaplace, lognorm, loguniform, lomax,
 * maxwell, mielke, moyal, nakagami, ncf, nct, ncx2, norminvgauss (scipy/stats/_continuous_distns.py) and levy_stable
 * (scipy/stats/_levy_stable/__init__.py with c_src/levyst.c), SciPy 1.17.1, private method by private method.
 * See stats_dist.hpp.
 */
#include <exception>
#include <stdexcept>
#include "stats_dist.hpp"

#include <algorithm>
#include <complex>
#include <vector>

extern "C" {
#include "quadpack/quadpack.h"
}

namespace tsd {
namespace {

const double C4_EULER = 0.577215664901532860606512090082402431042;
const double C4_SQRT_2_OVER_PI = 0.7978845608028654;
const double C4_LOG_SQRT_2_OVER_PI = -0.22579135264472744;
const double C4_XMIN = std::numeric_limits<double>::min();
const double C4_LOGXMIN = std::log(C4_XMIN);
const double C4_NORM_ENTROPY = 0.5 * (std::log(2 * M_PI) + 1);

/* _norm_* helpers of _continuous_distns.py */
inline double norm_cdf(double x) { return sc::ndtr(x); }
inline double norm_sf(double x) { return sc::ndtr(-x); }
inline double norm_ppf(double q) { return sc::ndtri(q); }
inline double norm_isf(double q) { return -sc::ndtri(q); }

/* np.polyval(p, x): y = 0; y = y * x + pv */
inline double polyval(std::initializer_list<double> p, double x)
{
    double y = 0.0;
    for (double pv : p) y = y * x + pv;
    return y;
}

struct Laplace : Cont {
    Laplace() { name = "laplace"; doc = "A Laplace continuous random variable."; }
    double pdf(double x, const double *) const override { return 0.5 * std::exp(-std::fabs(x)); }
    double cdf(double x, const double *) const override { return x > 0 ? 1.0 - 0.5 * std::exp(-x) : 0.5 * std::exp(x); }
    double sf(double x, const double *s) const override { return cdf(-x, s); }
    double ppf(double q, const double *) const override { return q > 0.5 ? -std::log(2 * (1 - q)) : std::log(2 * q); }
    double isf(double q, const double *s) const override { return -ppf(q, s); }
    void stats(const double *, int, Stats4 &st) const override { st.set(0, 2, 0, 3); }
    double entropy(const double *) const override { return std::log(2) + 1; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_laplace(bg, 0.0, 1.0);
    }
};
Registrar r_laplace(new Laplace);

struct LaplaceAsymmetric : Cont {
    LaplaceAsymmetric()
    {
        name = "laplace_asymmetric"; shapes = "kappa"; nshape = 1;
        doc = "An asymmetric Laplace continuous random variable.";
    }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double kappa = s[0], kapinv = 1 / kappa;
        double lPx = x * (x >= 0 ? -kappa : kapinv);
        lPx -= std::log(kappa + kapinv);
        return lPx;
    }
    double cdf(double x, const double *s) const override
    {
        const double kappa = s[0], kapinv = 1 / kappa, kk = kappa + kapinv;
        return x >= 0 ? 1 - std::exp(-x * kappa) * (kapinv / kk) : std::exp(x * kapinv) * (kappa / kk);
    }
    double sf(double x, const double *s) const override
    {
        const double kappa = s[0], kapinv = 1 / kappa, kk = kappa + kapinv;
        return x >= 0 ? std::exp(-x * kappa) * (kapinv / kk) : 1 - std::exp(x * kapinv) * (kappa / kk);
    }
    double ppf(double q, const double *s) const override
    {
        const double kappa = s[0], kapinv = 1 / kappa, kk = kappa + kapinv;
        return q >= kappa / kk ? -std::log((1 - q) * kk * kappa) * kapinv : std::log(q * kk / kappa) * kappa;
    }
    double isf(double q, const double *s) const override
    {
        const double kappa = s[0], kapinv = 1 / kappa, kk = kappa + kapinv;
        return q <= kapinv / kk ? -std::log(q * kk * kappa) * kapinv : std::log((1 - q) * kk / kappa) * kappa;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double kappa = s[0], kapinv = 1 / kappa;
        const double mn = kapinv - kappa;
        const double var = kapinv * kapinv + kappa * kappa;
        const double g1 = 2.0 * (1 - std::pow(kappa, 6.0)) / std::pow(1 + std::pow(kappa, 4.0), 1.5);
        const double g2 = 6.0 * (1 + std::pow(kappa, 8.0)) / std::pow(1 + std::pow(kappa, 4.0), 2.0);
        st.set(mn, var, g1, g2);
    }
    double entropy(const double *s) const override { return 1 + std::log(s[0] + 1 / s[0]); }
};
Registrar r_laplace_asymmetric(new LaplaceAsymmetric);

/* levy, levy_l and norminvgauss have SciPy's _open_support_mask: the public pdf is 0 (logpdf -inf) at the support
   end points, which the framework's closed-interval test lets through, so _pdf returns 0 there */
struct Levy : Cont {
    Levy() { name = "levy"; a = 0.0; doc = "A Levy continuous random variable."; }
    double pdf(double x, const double *) const override
    {
        if (x == 0.0) return 0.0;              /* _open_support_mask */
        return 1 / std::sqrt(2 * M_PI * x) / x * std::exp(-1 / (2 * x));
    }
    double cdf(double x, const double *) const override { return sc::erfc(std::sqrt(0.5 / x)); }
    double sf(double x, const double *) const override { return sc::erf(std::sqrt(0.5 / x)); }
    double ppf(double q, const double *) const override
    {
        const double val = norm_isf(q / 2);
        return 1.0 / (val * val);
    }
    double isf(double p, const double *) const override
    {
        const double e = sc::erfinv(p);
        return 1 / (2 * (e * e));
    }
    void stats(const double *, int, Stats4 &st) const override { st.set(INF, INF, NaN, NaN); }
};
Registrar r_levy(new Levy);

struct LevyL : Cont {
    LevyL() { name = "levy_l"; b = 0.0; doc = "A left-skewed Levy continuous random variable."; }
    double pdf(double x, const double *) const override
    {
        if (x == 0.0) return 0.0;              /* _open_support_mask */
        const double ax = std::fabs(x);
        return 1 / std::sqrt(2 * M_PI * ax) / ax * std::exp(-1 / (2 * ax));
    }
    double cdf(double x, const double *) const override { return 2 * norm_cdf(1 / std::sqrt(std::fabs(x))) - 1; }
    double sf(double x, const double *) const override { return 2 * norm_sf(1 / std::sqrt(std::fabs(x))); }
    double ppf(double q, const double *) const override
    {
        const double val = norm_ppf((q + 1.0) / 2);
        return -1.0 / (val * val);
    }
    double isf(double p, const double *) const override
    {
        const double v = norm_isf(p / 2);
        return -1 / (v * v);
    }
    void stats(const double *, int, Stats4 &st) const override { st.set(INF, INF, NaN, NaN); }
};
Registrar r_levy_l(new LevyL);

struct LogGamma : Cont {
    LogGamma() { name = "loggamma"; shapes = "c"; nshape = 1; doc = "A log gamma continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(s[0] * x - std::exp(x) - sc::gammaln(s[0])); }
    double logpdf(double x, const double *s) const override { return s[0] * x - std::exp(x) - sc::gammaln(s[0]); }
    double cdf(double x, const double *s) const override
    {
        const double c = s[0];
        return x < C4_LOGXMIN ? std::exp(c * x - sc::gammaln(c + 1)) : sc::gammainc(c, std::exp(x));
    }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0], g = sc::gammaincinv(c, q);
        return g < C4_XMIN ? (std::log(q) + sc::gammaln(c + 1)) / c : std::log(g);
    }
    double sf(double x, const double *s) const override
    {
        const double c = s[0];
        return x < C4_LOGXMIN ? -std::expm1(c * x - sc::gammaln(c + 1)) : sc::gammaincc(c, std::exp(x));
    }
    double isf(double q, const double *s) const override
    {
        const double c = s[0], g = sc::gammainccinv(c, q);
        return g < C4_XMIN ? (std::log1p(-q) + sc::gammaln(c + 1)) / c : std::log(g);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0];
        /* sc.polygamma(n, c) = (-1)**(n+1) * gamma(n+1) * zeta(n+1, c) */
        auto polygamma = [](double n, double x) { return std::pow(-1.0, n + 1) * sc::gamma(n + 1.0) * sc::_zeta(n + 1, x); };
        const double mean = sc::digamma(c);
        const double var = polygamma(1, c);
        const double skewness = polygamma(2, c) / std::pow(var, 1.5);
        const double excess_kurtosis = polygamma(3, c) / (var * var);
        st.set(mean, var, skewness, excess_kurtosis);
    }
    double entropy(const double *s) const override
    {
        const double c = s[0];
        if (c >= 45) {
            const double term = -0.5 * std::log(c) + (1 / c) / 6 - std::pow(c, -3.) / 90 + std::pow(c, -5.) / 210;
            return C4_NORM_ENTROPY + term;
        }
        return sc::gammaln(c) - c * sc::digamma(c) + c;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_gamma(bg, sa[0][i] + 1, 1.0);
        for (int64_t i = 0; i < n; i++) out[i] = std::log(out[i]) + std::log(random_standard_uniform(bg)) / sa[0][i];
    }
};
Registrar r_loggamma(new LogGamma);

struct Logistic : Cont {
    Logistic() { name = "logistic"; doc = "A logistic (or Sech-squared) continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *) const override
    {
        const double y = -std::fabs(x);
        return y - 2. * sc::log1p(std::exp(y));
    }
    double cdf(double x, const double *) const override { return sc::expit(x); }
    double logcdf(double x, const double *) const override { return sc::log_expit(x); }
    double ppf(double q, const double *) const override { return sc::logit(q); }
    double sf(double x, const double *) const override { return sc::expit(-x); }
    double logsf(double x, const double *) const override { return sc::log_expit(-x); }
    double isf(double q, const double *) const override { return -sc::logit(q); }
    void stats(const double *, int, Stats4 &st) const override { st.set(0, M_PI * M_PI / 3.0, 0, 6.0 / 5.0); }
    double entropy(const double *) const override { return 2.0; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_logistic(bg, 0.0, 1.0);
    }
};
Registrar r_logistic(new Logistic);

struct LogLaplace : Cont {
    LogLaplace() { name = "loglaplace"; shapes = "c"; nshape = 1; a = 0.0; doc = "A log-Laplace continuous random variable."; }
    double pdf(double x, const double *s) const override
    {
        const double cd2 = s[0] / 2.0;
        const double c = x < 1 ? s[0] : -s[0];
        return cd2 * std::pow(x, c - 1);
    }
    double cdf(double x, const double *s) const override
    {
        const double c = s[0];
        return x < 1 ? 0.5 * std::pow(x, c) : 1 - 0.5 * std::pow(x, -c);
    }
    double sf(double x, const double *s) const override
    {
        const double c = s[0];
        return x < 1 ? 1 - 0.5 * std::pow(x, c) : 0.5 * std::pow(x, -c);
    }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0];
        return q < 0.5 ? std::pow(2.0 * q, 1.0 / c) : std::pow(2 * (1.0 - q), -1.0 / c);
    }
    double isf(double q, const double *s) const override
    {
        const double c = s[0];
        return q > 0.5 ? std::pow(2.0 * (1.0 - q), 1.0 / c) : std::pow(2 * q, -1.0 / c);
    }
    double munp(int n, const double *s) const override
    {
        const double c2 = s[0] * s[0], n2 = (double)(n * n);
        return n2 < c2 ? c2 / (c2 - n2) : INF;
    }
    double entropy(const double *s) const override { return std::log(2.0 / s[0]) + 1.0; }
};
Registrar r_loglaplace(new LogLaplace);

struct LogNorm : Cont {
    LogNorm() { name = "lognorm"; shapes = "s"; nshape = 1; a = 0.0; doc = "A lognormal continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        /* _lognorm_logpdf */
        if (x == 0) return -INF;
        const double sh = s[0], lx = std::log(x);
        return -(lx * lx) / (2 * (sh * sh)) - std::log(sh * x * std::sqrt(2 * M_PI));
    }
    double cdf(double x, const double *s) const override { return norm_cdf(std::log(x) / s[0]); }
    double logcdf(double x, const double *s) const override { return sc::log_ndtr(std::log(x) / s[0]); }
    double ppf(double q, const double *s) const override { return std::exp(s[0] * norm_ppf(q)); }
    double sf(double x, const double *s) const override { return norm_sf(std::log(x) / s[0]); }
    double logsf(double x, const double *s) const override { return sc::log_ndtr(-(std::log(x) / s[0])); }
    double isf(double q, const double *s) const override { return std::exp(s[0] * norm_isf(q)); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double sh = s[0];
        const double p = std::exp(sh * sh);
        const double mu = std::sqrt(p);
        const double mu2 = p * (p - 1);
        const double g1 = std::sqrt(p - 1) * (2 + p);
        const double g2 = polyval({1, 2, 3, 0, -6.0}, p);
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *s) const override { return 0.5 * (1 + std::log(2 * M_PI) + 2 * std::log(s[0])); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = std::exp(sa[0][i] * random_standard_normal(bg));
    }
};
Registrar r_lognorm(new LogNorm);

/* _log_diff(log_p, log_q) = logsumexp([log_p, log_q + pi*1j], axis=0) for real log_p, log_q (scipy.special
   logsumexp on the two complex elements, with numpy's complex exp/log1p/log) */
std::complex<double> log_diff(double log_p, double log_q)
{
    typedef std::complex<double> C;
    const C e[2] = {C(log_p, 0.0), C(log_q, M_PI)};
    /* the plain sum, used where the result is not finite */
    C sum_(-0.0, -0.0);
    for (int i = 0; i < 2; i++) sum_ += std::exp(e[i]);
    const C out_inf = std::log(sum_);
    /* _elements_and_indices_with_max_real: the last element with the maximal real part */
    const double mr = std::max(e[0].real(), e[1].real());
    int im = -1;
    for (int i = 0; i < 2; i++)
        if (e[i].real() == mr) im = i;
    if (std::isnan(mr)) im = -1;
    C a_max(-0.0, -0.0);
    for (int i = 0; i < 2; i++) a_max += (i == im) ? e[i] : C(0.0, 0.0);
    const double m = im >= 0 ? 1.0 : 0.0;
    C s(-0.0, -0.0);
    for (int i = 0; i < 2; i++) s += std::exp((i == im ? C(-INF, e[i].imag() * 0.0) : e[i]) - a_max);
    if (!(s == C(0.0, 0.0))) s = s / C(m, 0.0);
    /* numpy's complex log1p: (log(hypot(re + 1, im)), atan2(im, re + 1)) */
    const C l1p(std::log(std::hypot(s.real() + 1, s.imag())), std::atan2(s.imag(), s.real() + 1));
    C out = l1p + std::log(C(m, 0.0)) + a_max;
    if (!(std::isfinite(out.real()) && std::isfinite(out.imag()))) out = out_inf;
    /* _wrap_radians on the imaginary part */
    double imv = out.imag();
    if (!(std::fabs(imv) < M_PI)) imv = -(std::fmod(-imv + M_PI, 2 * M_PI) - M_PI);
    return C(out.real(), imv);
}

struct LogUniform : Cont {
    LogUniform() { name = "loguniform"; shapes = "a, b"; nshape = 2; doc = "A loguniform or reciprocal continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[1] > s[0]; }
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
Registrar r_loguniform(new LogUniform);

struct Lomax : Cont {
    Lomax() { name = "lomax"; shapes = "c"; nshape = 1; a = 0.0; doc = "A Lomax (Pareto of the second kind) continuous random variable."; }
    double pdf(double x, const double *s) const override { return s[0] * 1.0 / std::pow(1.0 + x, s[0] + 1.0); }
    double logpdf(double x, const double *s) const override { return std::log(s[0]) - (s[0] + 1) * sc::log1p(x); }
    double cdf(double x, const double *s) const override { return -sc::expm1(-s[0] * sc::log1p(x)); }
    double sf(double x, const double *s) const override { return std::exp(-s[0] * sc::log1p(x)); }
    double logsf(double x, const double *s) const override { return -s[0] * sc::log1p(x); }
    double ppf(double q, const double *s) const override { return sc::expm1(-sc::log1p(-q) / s[0]); }
    double isf(double q, const double *s) const override { return std::pow(q, -1.0 / s[0]) - 1; }
    void stats(const double *s, int, Stats4 &st) const override
    {
        /* pareto.stats(c, loc=-1.0, moments='mvsk') */
        const double bt = s[0];
        const double mu = bt > 1 ? bt / (bt - 1.0) : INF;
        const double mu2 = bt > 2 ? bt / (bt - 2.0) / ((bt - 1.0) * (bt - 1.0)) : INF;
        const double g1 = bt > 3 ? 2 * (bt + 1.0) * std::sqrt(bt - 2.0) / ((bt - 3.0) * std::sqrt(bt)) : NaN;
        const double g2 = bt > 4 ? 6.0 * polyval({1.0, 1.0, -6, -2}, bt) / polyval({1.0, -7.0, 12.0, 0.0}, bt) : NaN;
        st.set(mu * 1.0 + -1.0, mu2 * 1.0 * 1.0, g1, g2);
    }
    double entropy(const double *s) const override { return 1 + 1.0 / s[0] - std::log(s[0]); }
};
Registrar r_lomax(new Lomax);

struct Maxwell : Cont {
    Maxwell() { name = "maxwell"; a = 0.0; doc = "A Maxwell continuous random variable."; }
    double pdf(double x, const double *) const override { return C4_SQRT_2_OVER_PI * x * x * std::exp(-x * x / 2.0); }
    double logpdf(double x, const double *) const override { return C4_LOG_SQRT_2_OVER_PI + 2 * std::log(x) - 0.5 * x * x; }
    double cdf(double x, const double *) const override { return sc::gammainc(1.5, x * x / 2.0); }
    double ppf(double q, const double *) const override { return std::sqrt(2 * sc::gammaincinv(1.5, q)); }
    double sf(double x, const double *) const override { return sc::gammaincc(1.5, x * x / 2.0); }
    double isf(double q, const double *) const override { return std::sqrt(2 * sc::gammainccinv(1.5, q)); }
    void stats(const double *, int, Stats4 &st) const override
    {
        const double val = 3 * M_PI - 8;
        st.set(2 * std::sqrt(2.0 / M_PI), 3 - 8 / M_PI, std::sqrt(2) * (32 - 10 * M_PI) / std::pow(val, 1.5),
               (-12 * M_PI * M_PI + 160 * M_PI - 384) / std::pow(val, 2.0));
    }
    double entropy(const double *) const override { return C4_EULER + 0.5 * std::log(2 * M_PI) - 0.5; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        /* chi.rvs(3.0) = sqrt(chi2.rvs(3.0)) */
        for (int64_t i = 0; i < n; i++) out[i] = std::sqrt(random_chisquare(bg, 3.0));
    }
};
Registrar r_maxwell(new Maxwell);

struct Mielke : Cont {
    Mielke() { name = "mielke"; shapes = "k, s"; nshape = 2; a = 0.0; doc = "A Mielke Beta-Kappa / Dagum continuous random variable."; }
    double pdf(double x, const double *p) const override
    {
        const double k = p[0], s = p[1];
        return k * std::pow(x, k - 1.0) / std::pow(1.0 + std::pow(x, s), 1.0 + k * 1.0 / s);
    }
    double logpdf(double x, const double *p) const override
    {
        const double k = p[0], s = p[1];
        return std::log(k) + std::log(x) * (k - 1) - std::log1p(std::pow(x, s)) * (1 + k / s);
    }
    double cdf(double x, const double *p) const override
    {
        const double k = p[0], s = p[1];
        return std::pow(x, k) / std::pow(1.0 + std::pow(x, s), k * 1.0 / s);
    }
    double ppf(double q, const double *p) const override
    {
        const double k = p[0], s = p[1];
        const double qsk = std::pow(q, s * 1.0 / k);
        return std::pow(qsk / (1.0 - qsk), 1.0 / s);
    }
    double munp(int n, const double *p) const override
    {
        const double k = p[0], s = p[1];
        if (!(n < s)) return INF;
        return sc::gamma((k + n) / s) * sc::gamma(1 - n / s) / sc::gamma(k / s);
    }
};
Registrar r_mielke(new Mielke);

struct Moyal : Cont {
    Moyal() { name = "moyal"; doc = "A Moyal continuous random variable."; }
    double pdf(double x, const double *) const override { return std::exp(-0.5 * (x + std::exp(-x))) / std::sqrt(2 * M_PI); }
    double cdf(double x, const double *) const override { return sc::erfc(std::exp(-0.5 * x) / std::sqrt(2)); }
    double sf(double x, const double *) const override { return sc::erf(std::exp(-0.5 * x) / std::sqrt(2)); }
    double ppf(double x, const double *) const override
    {
        const double e = sc::erfcinv(x);
        return -std::log(2 * (e * e));
    }
    void stats(const double *, int, Stats4 &st) const override
    {
        const double mu = std::log(2) + C4_EULER;
        const double mu2 = std::pow(M_PI, 2.0) / 2;
        const double g1 = 28 * std::sqrt(2) * sc::_riemann_zeta(3) / std::pow(M_PI, 3.0);
        st.set(mu, mu2, g1, 4.);
    }
    double munp(int n, const double *s) const override
    {
        const double l = std::log(2) + C4_EULER;
        if (n == 1) return l;
        if (n == 2) return std::pow(M_PI, 2.0) / 2 + std::pow(l, 2.0);
        if (n == 3) return 1.5 * std::pow(M_PI, 2.0) * l + std::pow(l, 3.0) + 14 * sc::_riemann_zeta(3);
        if (n == 4)
            return 4 * 14 * sc::_riemann_zeta(3) * l + 3 * std::pow(M_PI, 2.0) * std::pow(l, 2.0) + std::pow(l, 4.0) +
                   7 * std::pow(M_PI, 4.0) / 4;
        return Cont::munp(n, s);   /* _mom1_sc: quad of ppf(q)**n over [0, 1] */
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        /* gamma.rvs(a=0.5, scale=2) */
        for (int64_t i = 0; i < n; i++) out[i] = -std::log(random_standard_gamma(bg, 0.5) * 2.0 + 0.0);
    }
};
Registrar r_moyal(new Moyal);

struct Nakagami : Cont {
    Nakagami() { name = "nakagami"; shapes = "nu"; nshape = 1; a = 0.0; doc = "A Nakagami continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double nu = s[0];
        return std::log(2) + sc::xlogy(nu, nu) - sc::gammaln(nu) + sc::xlogy(2 * nu - 1, x) - nu * (x * x);
    }
    double cdf(double x, const double *s) const override { return sc::gammainc(s[0], s[0] * x * x); }
    double ppf(double q, const double *s) const override { return std::sqrt(1.0 / s[0] * sc::gammaincinv(s[0], q)); }
    double sf(double x, const double *s) const override { return sc::gammaincc(s[0], s[0] * x * x); }
    double isf(double p, const double *s) const override { return std::sqrt(1 / s[0] * sc::gammainccinv(s[0], p)); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double nu = s[0];
        const double mu = sc::poch(nu, 0.5) / std::sqrt(nu);
        const double mu2 = 1.0 - mu * mu;
        const double g1 = mu * (1 - 4 * nu * mu2) / 2.0 / nu / std::pow(mu2, 1.5);
        double g2 = -6 * std::pow(mu, 4.0) * nu + (8 * nu - 2) * (mu * mu) - 2 * nu + 1;
        g2 /= nu * (mu2 * mu2);
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *s) const override
    {
        const double nu = s[0];
        const double A = sc::gammaln(nu);
        const double B = nu - (nu - 0.5) * sc::digamma(nu);
        const double C = -0.5 * std::log(nu) - std::log(2);
        if (nu > 5e4) return C + C4_NORM_ENTROPY - 1 / (12 * nu);
        return A + B + C;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = std::sqrt(random_standard_gamma(bg, sa[0][i]) / sa[0][i]);
    }
};
Registrar r_nakagami(new Nakagami);

struct Ncf : Cont {
    Ncf() { name = "ncf"; shapes = "dfn, dfd, nc"; nshape = 3; a = 0.0; doc = "A non-central F distribution continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[1] > 0 && s[2] >= 0; }
    double pdf(double x, const double *s) const override { return sc::_ncf_pdf(x, s[0], s[1], s[2]); }
    double cdf(double x, const double *s) const override { return sc::ncfdtr(s[0], s[1], s[2], x); }
    double ppf(double q, const double *s) const override { return sc::ncfdtri(s[0], s[1], s[2], q); }
    double sf(double x, const double *s) const override { return sc::_ncf_sf(x, s[0], s[1], s[2]); }
    double isf(double x, const double *s) const override { return sc::_ncf_isf(x, s[0], s[1], s[2]); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        st.set(0, sc::_ncf_mean(s[0], s[1], s[2]));
        st.set(1, sc::_ncf_variance(s[0], s[1], s[2]));
        if (moments & MOM_S) st.set(2, sc::_ncf_skewness(s[0], s[1], s[2]));
        if (moments & MOM_K) st.set(3, sc::_ncf_kurtosis_excess(s[0], s[1], s[2]) - 3);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_noncentral_f(bg, sa[0][i], sa[1][i], sa[2][i]);
    }
};
Registrar r_ncf(new Ncf);

struct Nct : Cont {
    Nct() { name = "nct"; shapes = "df, nc"; nshape = 2; doc = "A non-central Student's t continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[1] == s[1]; }
    double pdf(double x, const double *s) const override { return sc::_nct_pdf(x, s[0], s[1]); }
    double cdf(double x, const double *s) const override { return sc::nctdtr(s[0], s[1], x); }
    double ppf(double q, const double *s) const override { return sc::nctdtrit(s[0], s[1], q); }
    double sf(double x, const double *s) const override
    {
        const double v = sc::_nct_sf(x, s[0], s[1]);   /* np.clip(v, 0, 1) */
        return std::isnan(v) ? v : std::min(std::max(v, 0.0), 1.0);
    }
    double isf(double x, const double *s) const override { return sc::_nct_isf(x, s[0], s[1]); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        st.set(0, sc::_nct_mean(s[0], s[1]));
        st.set(1, sc::_nct_variance(s[0], s[1]));
        if (moments & MOM_S) st.set(2, sc::_nct_skewness(s[0], s[1]));
        if (moments & MOM_K) st.set(3, sc::_nct_kurtosis_excess(s[0], s[1]));
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* norm.rvs(loc=nc) for all, then chi2.rvs(df) for all */
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_normal(bg) * 1.0 + sa[1][i];
        for (int64_t i = 0; i < n; i++) {
            const double c2 = random_chisquare(bg, sa[0][i]);
            out[i] = out[i] * std::sqrt(sa[0][i]) / std::sqrt(c2);
        }
    }
};
Registrar r_nct(new Nct);

/* chi2's private methods, as ncx2 calls them for nc == 0 */
inline double chi2_logpdf(double x, double df) { return sc::xlogy(df / 2. - 1, x) - x / 2. - sc::gammaln(df / 2.) - (std::log(2) * df) / 2.; }

struct Ncx2 : Cont {
    Ncx2() { name = "ncx2"; shapes = "df, nc"; nshape = 2; a = 0.0; doc = "A non-central chi-squared continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && std::isfinite(s[0]) && s[1] >= 0; }
    double logpdf(double x, const double *s) const override
    {
        const double df = s[0], nc = s[1];
        if (nc == 0) return chi2_logpdf(x, df);
        /* _ncx2_log_pdf */
        const double df2 = df / 2.0 - 1.0;
        const double xs = std::sqrt(x), ns = std::sqrt(nc);
        const double res = sc::xlogy(df2 / 2.0, x / nc) - 0.5 * ((xs - ns) * (xs - ns));
        const double corr = sc::ive(df2, xs * ns) / 2.0;
        return corr > 0 ? res + std::log(corr) : -INF;
    }
    double pdf(double x, const double *s) const override
    {
        return s[1] != 0 ? sc::_ncx2_pdf(x, s[0], s[1]) : std::exp(chi2_logpdf(x, s[0]));
    }
    double cdf(double x, const double *s) const override { return s[1] != 0 ? sc::chndtr(x, s[0], s[1]) : sc::chdtr(s[0], x); }
    double ppf(double q, const double *s) const override
    {
        return s[1] != 0 ? sc::chndtrix(q, s[0], s[1]) : 2 * sc::gammaincinv(s[0] / 2, q);
    }
    double sf(double x, const double *s) const override { return s[1] != 0 ? sc::_ncx2_sf(x, s[0], s[1]) : sc::chdtrc(s[0], x); }
    double isf(double x, const double *s) const override { return s[1] != 0 ? sc::_ncx2_isf(x, s[0], s[1]) : sc::chdtri(s[0], x); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double df = s[0], nc = s[1];
        auto k_plus_cl = [](double k, double l, double c) { return k + c * l; };
        const double k2 = k_plus_cl(df, nc, 2.0);
        st.set(df + nc, 2.0 * k2, std::sqrt(8.0) * k_plus_cl(df, nc, 3) / std::sqrt(std::pow(k2, 3.0)),
               12.0 * k_plus_cl(df, nc, 4.0) / (k2 * k2));
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_noncentral_chisquare(bg, sa[0][i], sa[1][i]);
    }
};
Registrar r_ncx2(new Ncx2);

struct NormInvGauss : Cont {
    NormInvGauss() { name = "norminvgauss"; shapes = "a, b"; nshape = 2; doc = "A Normal Inverse Gaussian continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && std::fabs(s[1]) < s[0]; }
    double pdf(double x, const double *s) const override
    {
        if (std::isinf(x)) return 0.0;         /* _open_support_mask */
        const double a = s[0], b = s[1];
        const double gamma = std::sqrt(a * a - b * b);
        const double fac1 = a / M_PI;
        const double sq = std::hypot(1, x);
        return fac1 * sc::k1e(a * sq) * std::exp(b * x - a * sq + gamma) / sq;
    }
    double sf(double x, const double *s) const override
    {
        return quad([&](double t) { return pdf(t, s); }, x, INF);
    }
    double isf(double q, const double *s) const override
    {
        const double a = s[0], b = s[1];
        auto eq = [&](double x) { return sf(x, s) - q; };
        const double xm = b / std::sqrt(a * a - b * b) * 1.0 + 0.0;   /* self.mean(a, b) */
        const double em = eq(xm);
        if (em == 0) return xm;
        double left, right, delta = 1;
        if (em > 0) {
            left = xm;
            right = xm + delta;
            while (eq(right) > 0) {
                delta = 2 * delta;
                right = xm + delta;
            }
        } else {
            right = xm;
            left = xm - delta;
            while (eq(left) < 0) {
                delta = 2 * delta;
                left = xm - delta;
            }
        }
        return brentq(eq, left, right, xtol);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double a = s[0], b = s[1];
        const double gamma = std::sqrt(a * a - b * b);
        st.set(b / gamma, a * a / std::pow(gamma, 3.0), 3.0 * b / (a * std::sqrt(gamma)), 3.0 * (1 + 4 * (b * b) / (a * a)) / gamma);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* invgauss.rvs(mu=1/gamma) for all, then norm.rvs for all */
        for (int64_t i = 0; i < n; i++) {
            const double a = sa[0][i], b = sa[1][i];
            const double gamma = std::sqrt(a * a - b * b);
            out[i] = random_wald(bg, 1 / gamma, 1.0) * 1.0 + 0.0;
        }
        for (int64_t i = 0; i < n; i++) {
            const double ig = out[i];
            out[i] = sa[1][i] * ig + std::sqrt(ig) * random_standard_normal(bg);
        }
    }
};
Registrar r_norminvgauss(new NormInvGauss);


/* ---------------------------------------------------------------- levy_stable (scipy/stats/_levy_stable) */

/* scipy.integrate.quad(f, a, b, points=..., limit=, epsabs=, epsrel=): QUADPACK dqagpe on the unique break points
   strictly inside (a, b), followed by two zeros */
struct QpCall {
    double (*fn)(void *, double);
    void *ctx;
    std::exception_ptr err;   /* an exception from the integrand, rethrown after QUADPACK (C) returns */
};
thread_local QpCall *g_qp = nullptr;
double qp_thunk(double *x)
{
    QpCall *c = g_qp;
    if (c->err) return std::numeric_limits<double>::quiet_NaN();
    try {
        return c->fn(c->ctx, *x);
    } catch (...) {
        c->err = std::current_exception();
        return std::numeric_limits<double>::quiet_NaN();
    }
}

template <class F> double quad_points(F &&f, double a, double b, std::vector<double> points, int limit, double epsabs, double epsrel)
{
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    std::vector<double> pts_in;
    for (double p : points)
        if (a < p && p < b) pts_in.push_back(p);
    pts_in.push_back(0.0);
    pts_in.push_back(0.0);
    const int npts2 = (int)pts_in.size();
    std::vector<double> alist(limit), blist(limit), rlist(limit), elist(limit), pts(npts2);
    std::vector<int> iord(limit), level(limit), ndin(npts2);
    double result = 0.0, abserr = 0.0;
    int neval = 0, ier = 6, last = 0;
    QpCall c;
    c.ctx = (void *)&f;
    c.fn = [](void *q, double x) -> double { return (*(typename std::remove_reference<F>::type *)q)(x); };
    QpCall *saved = g_qp;
    g_qp = &c;
    dqagpe(qp_thunk, a, b, npts2, pts_in.data(), epsabs, epsrel, limit, &result, &abserr, &neval, &ier, alist.data(),
           blist.data(), rlist.data(), elist.data(), pts.data(), iord.data(), level.data(), ndin.data(), &last);
    g_qp = saved;
    if (c.err) std::rethrow_exception(c.err);
    if (ier == 6) fail("quad: the input is invalid");
    return result;
}

/* scipy.optimize.bisect (scipy/optimize/Zeros/bisect.c, rtol 4 eps, maxiter 100, NaN raises) */
template <class F> double bisect(F &&f, double xa, double xb, double xtol)
{
    const double rtol = 4 * std::numeric_limits<double>::epsilon();
    bool nan = false;
    auto fw = [&](double x) {
        const double v = f(x);
        if (std::isnan(v)) nan = true;
        return v;
    };
    const double fa = fw(xa);
    if (nan) { fail("bisect: the function value is NaN; solver cannot continue"); return NaN; }
    const double fb = fw(xb);
    if (nan) { fail("bisect: the function value is NaN; solver cannot continue"); return NaN; }
    if (fa == 0) return xa;
    if (fb == 0) return xb;
    if (std::signbit(fa) == std::signbit(fb)) { fail("bisect: f(a) and f(b) must have different signs"); return NaN; }
    double dm = xb - xa;
    for (int i = 0; i < 100; i++) {
        dm *= .5;
        const double xm = xa + dm;
        const double fm = fw(xm);
        if (nan) { fail("bisect: the function value is NaN; solver cannot continue"); return NaN; }
        if (std::signbit(fm) == std::signbit(fa)) xa = xm;
        if (fm == 0 || std::fabs(dm) < xtol + rtol * std::fabs(xm)) return xm;
    }
    fail("bisect: Failed to converge after 100 iterations.");
    return NaN;
}

/* levyst.c: Nolan's g(theta) with its precomputed terms */
struct Nolan {
    double alpha, zeta, xi, zeta_prefactor, alpha_exp, alpha_xi, zeta_offset, two_beta_div_pi, pi_div_two_beta,
        x0_div_term, c1, c2, c3;
    bool ne_one;
    Nolan(double alpha_, double beta, double x0)
    {
        alpha = alpha_;
        zeta = -beta * std::tan(M_PI_2 * alpha);
        if (alpha != 1.) {
            ne_one = true;
            xi = std::atan(-zeta) / alpha;
            zeta_prefactor = std::pow(std::pow(zeta, 2.) + 1., -1. / (2. * (alpha - 1.)));
            alpha_exp = alpha / (alpha - 1.);
            alpha_xi = std::atan(-zeta);
            zeta_offset = x0 - zeta;
            if (alpha < 1.) {
                c1 = 0.5 - xi * M_1_PI;
                c3 = M_1_PI;
            } else {
                c1 = 1.;
                c3 = -M_1_PI;
            }
            c2 = alpha * M_1_PI / std::fabs(alpha - 1.) / (x0 - zeta);
        } else {
            ne_one = false;
            xi = M_PI_2;
            two_beta_div_pi = beta * M_2_PI;
            pi_div_two_beta = M_PI_2 / beta;
            x0_div_term = x0 / two_beta_div_pi;
            c1 = 0.;
            c2 = .5 / std::fabs(beta);
            c3 = M_1_PI;
        }
    }
    double g(double theta) const
    {
        if (ne_one) {
            if (theta == -xi) return alpha < 1 ? 0 : INF;
            if (theta == M_PI_2) return alpha < 1 ? INF : 0;
            const double cos_theta = std::cos(theta);
            return zeta_prefactor * std::pow(cos_theta / std::sin(alpha_xi + alpha * theta) * zeta_offset, alpha_exp) *
                   std::cos(alpha_xi + (alpha - 1) * theta) / cos_theta;
        }
        if (theta == -xi) return 0;
        if (theta == M_PI_2) return INF;
        return (1 + theta * two_beta_div_pi) * std::exp((pi_div_two_beta + theta) * std::tan(theta) - x0_div_term) / std::cos(theta);
    }
};

const double LS_QUAD_EPS = 1.2e-14;
const double LS_X_TOL_NEAR_ZETA = 0.005;
const double LS_ALPHA_TOL_NEAR_ONE = 0.005;

inline double ls_round_x_near_zeta(double x0, double alpha, double zeta)
{
    if (std::fabs(x0 - zeta) < LS_X_TOL_NEAR_ZETA * std::pow(alpha, 1 / alpha)) x0 = zeta;
    return x0;
}

/* np.isclose(-xi, pi/2, rtol=1e-14, atol=1e-14) */
inline bool ls_isclose_half_pi(double v) { return std::fabs(v - M_PI / 2) <= 1e-14 + 1e-14 * std::fabs(M_PI / 2); }

double ls_pdf_post_rounding_Z0(double x0, double alpha, double beta, int depth = 0)
{
    /* SciPy recurses without a bound here (the reflection branch can call itself forever, e.g. alpha = 1 with
       |x| beyond zeta = -beta tan(pi/2)) and stops with RecursionError; Tessero raises the same way */
    if (depth > 64) throw std::runtime_error("maximum recursion depth exceeded in comparison");
    const Nolan nl(alpha, beta, x0);
    const double zeta = nl.zeta, xi = nl.xi, c2 = nl.c2;
    x0 = ls_round_x_near_zeta(x0, alpha, zeta);
    if (x0 == zeta) return sc::gamma(1 + 1 / alpha) * std::cos(xi) / M_PI / std::pow(1 + std::pow(zeta, 2.0), 1 / alpha / 2);
    if (x0 < zeta) return ls_pdf_post_rounding_Z0(-x0, alpha, -beta, depth + 1);
    if (ls_isclose_half_pi(-xi)) return 0.0;
    auto integrand = [&](double theta) {
        double g_1 = nl.g(theta);
        if (!std::isfinite(g_1) || g_1 < 0) g_1 = 0;
        return g_1 * std::exp(-g_1);
    };
    const double peak = bisect([&](double t) { return nl.g(t) - 1; }, -xi, M_PI / 2, LS_QUAD_EPS);
    std::vector<double> points = {0, peak};
    for (double h : {100.0, 10.0, 5.0}) points.push_back(bisect([&](double t) { return nl.g(t) - h; }, -xi, M_PI / 2, 2e-12));
    const double intg = quad_points(integrand, -xi, M_PI / 2, points, 100, 0.0, LS_QUAD_EPS);
    return c2 * intg;
}

double ls_pdf_Z1(double x, double alpha, double beta)
{
    /* _pdf_single_value_piecewise_Z1 -> _Z0 */
    double zeta = -beta * std::tan(M_PI * alpha / 2.0);
    double x0 = alpha != 1 ? x + zeta : x;
    zeta = -beta * std::tan(M_PI * alpha / 2.0);
    if (std::fabs(alpha - 1) < LS_ALPHA_TOL_NEAR_ONE) alpha = 1.0;
    x0 = ls_round_x_near_zeta(x0, alpha, zeta);
    if (alpha == 2.0) {
        const double y = x0 / std::sqrt(2);
        return std::exp(-(y * y) / 2.0) / std::sqrt(2 * M_PI) / std::sqrt(2);
    }
    if (alpha == 0.5 && beta == 1.0) {
        const double x_ = x0 + 1;
        if (x_ <= 0) return 0;
        return 1 / std::sqrt(2 * M_PI * x_) / x_ * std::exp(-1 / (2 * x_));
    }
    if (alpha == 0.5 && beta == 0.0 && x0 != 0) {
        double S, C;
        sc::fresnel(1 / std::sqrt(2 * M_PI * std::fabs(x0)), S, C);
        const double arg = 1 / (4 * std::fabs(x0));
        return (std::sin(arg) * (0.5 - S) + std::cos(arg) * (0.5 - C)) / std::sqrt(2 * M_PI * std::pow(std::fabs(x0), 3.0));
    }
    if (alpha == 1.0 && beta == 0.0) return 1 / (1 + x0 * x0) / M_PI;
    return ls_pdf_post_rounding_Z0(x0, alpha, beta);
}

double ls_cdf_post_rounding_Z0(double x0, double alpha, double beta, int depth = 0)
{
    /* SciPy recurses without a bound here (the reflection branch can call itself forever, e.g. alpha = 1 with
       |x| beyond zeta = -beta tan(pi/2)) and stops with RecursionError; Tessero raises the same way */
    if (depth > 64) throw std::runtime_error("maximum recursion depth exceeded in comparison");
    const Nolan nl(alpha, beta, x0);
    const double zeta = nl.zeta, xi = nl.xi, c1 = nl.c1, c3 = nl.c3;
    x0 = ls_round_x_near_zeta(x0, alpha, zeta);
    if ((alpha == 1 && beta < 0) || x0 < zeta) return 1 - ls_cdf_post_rounding_Z0(-x0, alpha, -beta, depth + 1);
    if (x0 == zeta) return 0.5 - xi / M_PI;
    if (ls_isclose_half_pi(-xi)) return c1;
    auto integrand = [&](double theta) { return std::exp(-nl.g(theta)); };
    /* SciPy shrinks the support with L-BFGS-B only when the integrand is nonzero at the end point where g is
       infinite; g returns inf exactly there (theta == -xi or pi/2), so that branch is never taken */
    const double left_support = -xi, right_support = M_PI / 2;
    if ((alpha > 1 && integrand(-xi) != 0.0) || (!(alpha > 1) && integrand(M_PI / 2) != 0.0)) {
        fail("levy_stable: L-BFGS-B support shrinking is not ported");
        return NaN;
    }
    const double intg = quad_points(integrand, left_support, right_support, {left_support, right_support}, 100, 0.0, LS_QUAD_EPS);
    return c1 + c3 * intg;
}

double ls_cdf_Z1(double x, double alpha, double beta)
{
    double zeta = -beta * std::tan(M_PI * alpha / 2.0);
    double x0 = alpha != 1 ? x + zeta : x;
    zeta = -beta * std::tan(M_PI * alpha / 2.0);
    if (std::fabs(alpha - 1) < LS_ALPHA_TOL_NEAR_ONE) alpha = 1.0;
    x0 = ls_round_x_near_zeta(x0, alpha, zeta);
    if (alpha == 2.0) return sc::ndtr(x0 / std::sqrt(2));
    if (alpha == 0.5 && beta == 1.0) {
        const double x_ = x0 + 1;
        if (x_ <= 0) return 0;
        return sc::erfc(std::sqrt(0.5 / x_));
    }
    if (alpha == 1.0 && beta == 0.0) return 0.5 + std::atan(x0) / M_PI;
    return ls_cdf_post_rounding_Z0(x0, alpha, beta);
}

/* The S1 parameterization (SciPy's default). SciPy's public pdf, cdf and rvs (overridden in levy_stable_gen)
   shift loc by 2 beta scale log(scale) / pi when alpha == 1 (public_loc_shift); its other methods do not. */
struct LevyStable : Cont {
    LevyStable() { name = "levy_stable"; shapes = "alpha, beta"; nshape = 2; doc = "A Levy-stable continuous random variable."; }
    double public_loc_shift(const double *s, double scale) const override
    {
        return s[0] == 1.0 ? 2 * s[1] * scale * std::log(scale) / M_PI : 0.0;
    }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[0] <= 2 && s[1] <= 1 && s[1] >= -1; }
    double pdf(double x, const double *s) const override { return ls_pdf_Z1(x, s[0], s[1]); }
    double cdf(double x, const double *s) const override { return ls_cdf_Z1(x, s[0], s[1]); }
    double entropy(const double *) const override
    {
        /* rv_continuous._entropy integrates entr(self._pdf(x)), but levy_stable's _pdf returns a 1-element array,
           which quad cannot convert to a float under NumPy 2: SciPy raises TypeError for every parameter */
        fail("only 0-dimensional arrays can be converted to Python scalars");
        return NaN;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double alpha = s[0];
        st.set(alpha > 1 ? 0 : NaN, alpha == 2 ? 2 : INF, alpha == 2.0 ? 0.0 : NaN, alpha == 2.0 ? 0.0 : NaN);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* _rvs_Z1: TH = uniform.rvs(loc=-pi/2, scale=pi) for all, then W = expon.rvs() for all */
        std::vector<double> W((size_t)n);
        for (int64_t i = 0; i < n; i++) out[i] = random_uniform(bg, 0.0, 1.0) * M_PI + -M_PI / 2.0;
        for (int64_t i = 0; i < n; i++) W[(size_t)i] = random_standard_exponential(bg) * 1.0 + 0.0;
        for (int64_t i = 0; i < n; i++) {
            const double alpha = sa[0][i], beta = sa[1][i], TH = out[i], w = W[(size_t)i];
            const double aTH = alpha * TH, bTH = beta * TH, cosTH = std::cos(TH), tanTH = std::tan(TH);
            double r;
            if (alpha == 1) {
                r = 2 / M_PI * ((M_PI / 2 + bTH) * tanTH - beta * std::log((M_PI / 2 * w * cosTH) / (M_PI / 2 + bTH)));
            } else if (beta == 0) {
                r = w / (cosTH / std::tan(aTH) + std::sin(TH)) * std::pow((std::cos(aTH) + std::sin(aTH) * tanTH) / w, 1.0 / alpha);
            } else {
                const double val0 = beta * std::tan(M_PI * alpha / 2);
                const double th0 = std::atan(val0) / alpha;
                const double val3 = w / (cosTH / std::tan(alpha * (th0 + TH)) + std::sin(TH));
                r = val3 * std::pow((std::cos(aTH) + std::sin(aTH) * tanTH - val0 * (std::sin(aTH) - std::cos(aTH) * tanTH)) / w,
                                    1.0 / alpha);
            }
            out[i] = r;
        }
    }
};
Registrar r_levy_stable(new LevyStable);

}  // namespace
}  // namespace tsd
