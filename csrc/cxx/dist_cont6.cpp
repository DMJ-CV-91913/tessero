/*
 * scipy.stats: trapezoid, triang, truncexpon, truncnorm, truncpareto, truncweibull_min, tukeylambda, vonmises,
 * vonmises_line, wald, weibull_max, weibull_min, wrapcauchy (scipy/stats/_continuous_distns.py, SciPy 1.17.1),
 * with the helpers they call: von_mises_cdf (scipy/stats/_stats.pyx), tukeylambda_variance / _kurtosis
 * (scipy/stats/_tukeylambda_stats.py), _log_sum / _log_diff / _log_gauss_mass (truncnorm) and invgauss's
 * methods at mu = 1 (wald). See stats_dist.hpp.
 */
#include "stats_dist.hpp"

#include <algorithm>

namespace tsd {
namespace {

const double EULER = 0.577215664901532860606512090082402431042;   /* scipy.stats._constants._EULER */
const double NORM_PDF_C = std::sqrt(2 * M_PI);
const double NORM_PDF_LOGC = std::log(NORM_PDF_C);

inline double norm_pdf(double x) { return std::exp(-(x * x) / 2.0) / NORM_PDF_C; }
inline double norm_logpdf(double x) { return -(x * x) / 2.0 - NORM_PDF_LOGC; }

/* np.mod (npy_remainder) */
double np_mod(double a, double b)
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

/* Some classes override the public rvs to wrap the variates after loc and scale are applied. The framework
   applies `out * scale + loc` after _rvs, so the wrapped value w is returned as the standard-form t with
   t * scale + loc == w exactly (the nearest double to (w - loc) / scale that maps onto w). */
double unscale(double w, double loc, double scale)
{
    const double t = (w - loc) / scale;
    if (!(scale != 0) || !std::isfinite(t) || t * scale + loc == w) return t;
    double lo = t, hi = t;
    for (int k = 0; k < 16; k++) {
        lo = std::nextafter(lo, -INF);
        hi = std::nextafter(hi, INF);
        if (lo * scale + loc == w) return lo;
        if (hi * scale + loc == w) return hi;
    }
    return t;
}

/* ---------------------------------------------------------------- trapezoid */

struct Trapezoid : Cont {
    Trapezoid() { name = "trapezoid"; shapes = "c, d"; nshape = 2; a = 0.0; b = 1.0; doc = "A trapezoidal continuous random variable."; }
    bool argcheck(const double *s) const override
    {
        const double c = s[0], d = s[1];
        return (c >= 0) & (c <= 1) & (d >= 0) & (d <= 1) & (d >= c);
    }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0], d = s[1];
        const double u = 2 / (d - c + 1);
        double r = 0.0;                               /* _lazyselect: the last matching condition wins */
        if (x < c) r = u * x / c;
        if ((c <= x) & (x <= d)) r = u;
        if (x > d) r = u * (1 - x) / (1 - d);
        return r;
    }
    double cdf(double x, const double *s) const override
    {
        const double c = s[0], d = s[1];
        double r = 0.0;
        if (x < c) r = x * x / c / (d - c + 1);
        if ((c <= x) & (x <= d)) r = (c + 2 * (x - c)) / (d - c + 1);
        if (x > d) r = 1 - ((1 - x) * (1 - x) / (d - c + 1) / (1 - d));
        return r;
    }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0], d = s[1];
        const double qc = cdf(c, s), qd = cdf(d, s);
        /* np.select: the first matching condition wins; default 0 */
        if (q < qc) return std::sqrt(q * c * (1 + d - c));
        if (q <= qd) return 0.5 * q * (1 + d - c) + 0.5 * c;
        if (q > qd) return 1 - std::sqrt((1 - q) * (d - c + 1) * (1 - d));
        return 0.0;
    }
    double munp(int n, const double *s) const override
    {
        const double c = s[0], d = s[1];
        const int e = n + 1;
        const double ab_term = e == 1 ? c : e == 2 ? c * c : std::pow(c, (double)e);
        double dc_term = 0.0;
        if (d == 0.0) dc_term = 1.0;
        if ((0.0 < d) & (d < 1.0)) dc_term = std::expm1((n + 2) * std::log(d)) / (d - 1.0);
        if (d == 1.0) dc_term = n + 2;
        return 2.0 / (1.0 + d - c) * (dc_term - ab_term) / ((n + 1) * (n + 2));
    }
    double entropy(const double *s) const override
    {
        const double c = s[0], d = s[1];
        return 0.5 * (1.0 - d + c) / (1.0 + d - c) + std::log(0.5 * (1.0 + d - c));
    }
};
Registrar r_trapezoid(new Trapezoid);

/* ---------------------------------------------------------------- triang */

struct Triang : Cont {
    Triang() { name = "triang"; shapes = "c"; nshape = 1; a = 0.0; b = 1.0; doc = "A triangular continuous random variable."; }
    bool argcheck(const double *s) const override { return (s[0] >= 0) & (s[0] <= 1); }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0];
        double r = 0.0;                               /* _lazyselect: the last matching condition wins */
        if (c == 0) r = 2 - 2 * x;
        if (x < c) r = 2 * x / c;
        if ((x >= c) & (c != 1)) r = 2 * (1 - x) / (1 - c);
        if (c == 1) r = 2 * x;
        return r;
    }
    double cdf(double x, const double *s) const override
    {
        const double c = s[0];
        double r = 0.0;
        if (c == 0) r = 2 * x - x * x;
        if (x < c) r = x * x / c;
        if ((x >= c) & (c != 1)) r = (x * x - 2 * x + c) / (c - 1);
        if (c == 1) r = x * x;
        return r;
    }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0];
        return q < c ? std::sqrt(c * q) : 1 - std::sqrt((1 - c) * (1 - q));
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0];
        st.set((c + 1.0) / 3.0, (1.0 - c + c * c) / 18,
               std::sqrt(2.0) * (2 * c - 1) * (c + 1) * (c - 2) / (5 * std::pow((1.0 - c + c * c), 1.5)), -3.0 / 5.0);
    }
    double entropy(const double *) const override { return 0.5 - std::log(2.0); }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_triangular(bg, 0, sa[0][i], 1);
    }
};
Registrar r_triang(new Triang);

/* ---------------------------------------------------------------- truncexpon */

struct Truncexpon : Cont {
    Truncexpon() { name = "truncexpon"; shapes = "b"; nshape = 1; a = 0.0; doc = "A truncated exponential continuous random variable."; }
    void get_support(const double *s, double &lo, double &hi) const override { lo = a; hi = s[0]; }
    double pdf(double x, const double *s) const override { return std::exp(-x) / (-sc::expm1(-s[0])); }
    double logpdf(double x, const double *s) const override { return -x - std::log(-sc::expm1(-s[0])); }
    double cdf(double x, const double *s) const override { return sc::expm1(-x) / sc::expm1(-s[0]); }
    double ppf(double q, const double *s) const override { return -sc::log1p(q * sc::expm1(-s[0])); }
    double sf(double x, const double *s) const override { return (std::exp(-s[0]) - std::exp(-x)) / sc::expm1(-s[0]); }
    double isf(double q, const double *s) const override { return -std::log(std::exp(-s[0]) - q * sc::expm1(-s[0])); }
    double munp(int n, const double *s) const override
    {
        const double b = s[0];
        if (n == 1) return (1 - (b + 1) * std::exp(-b)) / (-sc::expm1(-b));
        if (n == 2) return 2 * (1 - 0.5 * (b * b + 2 * b + 2) * std::exp(-b)) / (-sc::expm1(-b));
        return Cont::munp(n, s);
    }
    double entropy(const double *s) const override
    {
        const double b = s[0];
        const double eB = std::exp(b);
        return std::log(eB - 1) + (1 + eB * (b - 1.0)) / (1.0 - eB);
    }
};
Registrar r_truncexpon(new Truncexpon);

/* ---------------------------------------------------------------- truncnorm */

/* _log_sum: sc.logsumexp([log_p, log_q], axis=0) for real input */
double log_sum(double p, double q)
{
    if (std::isnan(p) || std::isnan(q)) return NaN;
    if (p == q) {
        if (std::isinf(p)) return p;                 /* -inf, -inf: the direct log(exp(p) + exp(q)) */
        return (std::log1p(0.0) + std::log(2.0)) + p;
    }
    const double mx = std::max(p, q), mn = std::min(p, q);
    const double s = std::exp(mn - mx);
    return (std::log1p(s) + 0.0) + mx;
}

/* _log_diff: sc.logsumexp([log_p, log_q + pi*1j], axis=0), real part. The element with the larger real part
   (the second on ties) is taken out; the other contributes s = exp(min - max) * (cos(pi), +-sin(pi)), and
   numpy's complex log1p is log(hypot(1 + Re s, Im s)). */
double log_diff(double p, double q)
{
    if (std::isnan(p) || std::isnan(q)) return NaN;
    if (p == -INF && q == -INF) return -INF;         /* non-finite result: log(exp(p) - exp(q)) = log(0) */
    const double mx = p > q ? p : q, mn = p > q ? q : p;
    const double e = std::exp(mn - mx);
    const double re = -e, im = e * std::sin(M_PI);
    return (std::log(std::hypot(re + 1, im)) + 0.0) + mx;
}

/* _log_gauss_mass(a, b) */
double log_gauss_mass(double a, double b)
{
    if (b <= 0) return log_diff(sc::log_ndtr(b), sc::log_ndtr(a));
    if (a > 0) return log_diff(sc::log_ndtr(-a), sc::log_ndtr(-b));
    return sc::log1p(-sc::ndtr(a) - sc::ndtr(-b));
}

struct Truncnorm : Cont {
    Truncnorm() { name = "truncnorm"; shapes = "a, b"; nshape = 2; momtype = 1; doc = "A truncated normal continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] < s[1]; }
    void get_support(const double *s, double &lo, double &hi) const override { lo = s[0]; hi = s[1]; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override { return norm_logpdf(x) - log_gauss_mass(s[0], s[1]); }
    double cdf(double x, const double *s) const override { return std::exp(logcdf(x, s)); }
    double logcdf(double x, const double *s) const override
    {
        const double a = s[0], b = s[1];
        double r = log_gauss_mass(a, x) - log_gauss_mass(a, b);
        if (r > -0.1) r = std::log1p(-std::exp(logsf(x, s)));   /* avoid catastrophic cancellation */
        return r;
    }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override
    {
        const double a = s[0], b = s[1];
        double r = log_gauss_mass(x, b) - log_gauss_mass(a, b);
        if (r > -0.1) r = std::log1p(-std::exp(logcdf(x, s)));
        return r;
    }
    double entropy(const double *s) const override
    {
        const double a = s[0], b = s[1];
        const double A = sc::ndtr(a), B = sc::ndtr(b);
        const double Z = B - A;
        const double C = std::log(std::sqrt(2 * M_PI * M_E) * Z);
        const double D = (a * norm_pdf(a) - b * norm_pdf(b)) / (2 * Z);
        return C + D;
    }
    double ppf(double q, const double *s) const override
    {
        const double a = s[0], b = s[1];
        if (a < 0) return sc::ndtri_exp(log_sum(sc::log_ndtr(a), std::log(q) + log_gauss_mass(a, b)));
        return -sc::ndtri_exp(log_sum(sc::log_ndtr(-b), std::log1p(-q) + log_gauss_mass(a, b)));
    }
    double isf(double q, const double *s) const override
    {
        const double a = s[0], b = s[1];
        if (b < 0) return sc::ndtri_exp(log_diff(sc::log_ndtr(b), std::log(q) + log_gauss_mass(a, b)));
        return -sc::ndtri_exp(log_diff(sc::log_ndtr(-a), std::log1p(-q) + log_gauss_mass(a, b)));
    }
    double munp(int n, const double *s) const override
    {
        const double a = s[0], b = s[1];
        if (!((n >= 0) & (a == a) & (b == b))) return NaN;
        const double ab[2] = {a, b};
        const double probs[2] = {pdf(a, s), -pdf(b, s)};
        double m_prev = 0, m_last = 1;                 /* moments = [0, 1] */
        for (int k = 1; k <= n; k++) {
            double v[2];
            for (int j = 0; j < 2; j++) {
                const double y = ab[j];
                const double p = k - 1 == 0 ? 1.0 : k - 1 == 1 ? y : k - 1 == 2 ? y * y : std::pow(y, (double)(k - 1));
                v[j] = probs[j] != 0 ? probs[j] * p : 0.0;
            }
            const double mk = (v[0] + v[1]) + (k - 1) * m_prev;
            m_prev = m_last;
            m_last = mk;
        }
        return m_last;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double a = s[0], b = s[1];
        const double pA = pub_pdf(a, s), pB = pub_pdf(b, s);
        const double ab[2] = {a, b};
        const double m1 = pA - pB;
        const double mu = m1;
        const double probs[2] = {pA, -pB};
        auto sum = [&](auto f) {
            double v[2];
            for (int j = 0; j < 2; j++) v[j] = probs[j] != 0 ? f(probs[j], ab[j]) : 0.0;
            return v[0] + v[1];
        };
        const double m2 = 1 + sum([](double x, double y) { return x * y; });
        const double mu2 = 1 + sum([&](double x, double y) { return x * (y - mu); });
        const double m3 = 2 * m1 + sum([](double x, double y) { return x * (y * y); });
        const double m4 = 3 * m2 + sum([](double x, double y) { return x * std::pow(y, 3.0); });
        const double mu3 = m3 + m1 * (-3 * m2 + 2 * (m1 * m1));
        const double g1 = mu3 / std::pow(mu2, 1.5);
        const double mu4 = m4 + m1 * (-4 * m3 + 3 * m1 * (2 * m2 - m1 * m1));
        const double g2 = mu4 / (mu2 * mu2) - 3;
        st.set(mu, mu2, g1, g2);
    }
};
Registrar r_truncnorm(new Truncnorm);

/* ---------------------------------------------------------------- truncpareto */

struct Truncpareto : Cont {
    Truncpareto() { name = "truncpareto"; shapes = "b, c"; nshape = 2; a = 1.0; doc = "An upper truncated Pareto continuous random variable."; }
    bool argcheck(const double *s) const override { return (s[0] != 0.) & (s[1] > 1.); }
    void get_support(const double *s, double &lo, double &hi) const override { lo = a; hi = s[1]; }
    double pdf(double x, const double *s) const override
    {
        const double b = s[0], c = s[1];
        return b * std::pow(x, -(b + 1)) / (1 - 1 / std::pow(c, b));
    }
    double logpdf(double x, const double *s) const override
    {
        const double b = s[0], c = s[1];
        if (b > 0) return std::log(b) - std::log(-std::expm1(-b * std::log(c))) - (b + 1) * std::log(x);
        return std::log(pdf(x, s));
    }
    double cdf(double x, const double *s) const override
    {
        const double b = s[0], c = s[1];
        return (1 - std::pow(x, -b)) / (1 - 1 / std::pow(c, b));
    }
    double logcdf(double x, const double *s) const override
    {
        const double b = s[0], c = s[1];
        if (b > 0) return std::log1p(-std::pow(x, -b)) - std::log1p(-1 / std::pow(c, b));
        return Cont::logcdf(x, s);
    }
    double ppf(double q, const double *s) const override
    {
        const double b = s[0], c = s[1];
        return std::pow(1 - (1 - 1 / std::pow(c, b)) * q, -1 / b);
    }
    double sf(double x, const double *s) const override
    {
        const double b = s[0], c = s[1];
        return (std::pow(x, -b) - 1 / std::pow(c, b)) / (1 - 1 / std::pow(c, b));
    }
    double logsf(double x, const double *s) const override
    {
        const double b = s[0], c = s[1];
        if (b > 0) return std::log(std::pow(x, -b) - 1 / std::pow(c, b)) - std::log1p(-1 / std::pow(c, b));
        return Cont::logsf(x, s);
    }
    double isf(double q, const double *s) const override
    {
        const double b = s[0], c = s[1];
        return std::pow(1 / std::pow(c, b) + (1 - 1 / std::pow(c, b)) * q, -1 / b);
    }
    double entropy(const double *s) const override
    {
        const double b = s[0], c = s[1];
        return -(std::log(b / (1 - 1 / std::pow(c, b))) + (b + 1) * (std::log(c) / (std::pow(c, b) - 1) - 1 / b));
    }
    double munp(int n, const double *s) const override
    {
        const double b = s[0], c = s[1], nn = n;
        if (nn == b) return b * std::log(c) / (1 - 1 / std::pow(c, b));
        return b / (b - nn) * (std::pow(c, b) - std::pow(c, nn)) / (std::pow(c, b) - 1);
    }
};
Registrar r_truncpareto(new Truncpareto);

/* ---------------------------------------------------------------- truncweibull_min */

struct TruncweibullMin : Cont {
    TruncweibullMin()
    {
        name = "truncweibull_min";
        shapes = "c, a, b";
        nshape = 3;
        doc = "A doubly truncated Weibull minimum continuous random variable.";
    }
    bool argcheck(const double *s) const override { return (s[1] >= 0.) & (s[2] > s[1]) & (s[0] > 0.); }
    void get_support(const double *s, double &lo, double &hi) const override { lo = s[1]; hi = s[2]; }
    static double denum(double c, double a, double b) { return std::exp(-std::pow(a, c)) - std::exp(-std::pow(b, c)); }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0];
        return (c * std::pow(x, c - 1) * std::exp(-std::pow(x, c))) / denum(c, s[1], s[2]);
    }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0];
        const double logdenum = std::log(denum(c, s[1], s[2]));
        return std::log(c) + sc::xlogy(c - 1, x) - std::pow(x, c) - logdenum;
    }
    double cdf(double x, const double *s) const override
    {
        const double c = s[0], a = s[1];
        return (std::exp(-std::pow(a, c)) - std::exp(-std::pow(x, c))) / denum(c, a, s[2]);
    }
    double logcdf(double x, const double *s) const override
    {
        const double c = s[0], a = s[1];
        const double lognum = std::log(std::exp(-std::pow(a, c)) - std::exp(-std::pow(x, c)));
        return lognum - std::log(denum(c, a, s[2]));
    }
    double sf(double x, const double *s) const override
    {
        const double c = s[0], b = s[2];
        return (std::exp(-std::pow(x, c)) - std::exp(-std::pow(b, c))) / denum(c, s[1], b);
    }
    double logsf(double x, const double *s) const override
    {
        const double c = s[0], b = s[2];
        const double lognum = std::log(std::exp(-std::pow(x, c)) - std::exp(-std::pow(b, c)));
        return lognum - std::log(denum(c, s[1], b));
    }
    double isf(double q, const double *s) const override
    {
        const double c = s[0], a = s[1], b = s[2];
        return std::pow(-std::log((1 - q) * std::exp(-std::pow(b, c)) + q * std::exp(-std::pow(a, c))), 1 / c);
    }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0], a = s[1], b = s[2];
        return std::pow(-std::log((1 - q) * std::exp(-std::pow(a, c)) + q * std::exp(-std::pow(b, c))), 1 / c);
    }
    double munp(int n, const double *s) const override
    {
        const double c = s[0], a = s[1], b = s[2];
        const double gamma_fun = sc::gamma(n / c + 1.) * (sc::gammainc(n / c + 1., std::pow(b, c)) - sc::gammainc(n / c + 1., std::pow(a, c)));
        return gamma_fun / denum(c, a, b);
    }
};
Registrar r_truncweibull_min(new TruncweibullMin);

/* ---------------------------------------------------------------- tukeylambda */

/* np.polyval(p, x) with p highest degree first (poly1d(coefs[::-1])) */
double polyval_rev(const double *coef, int m, double x)
{
    double y = 0.0;
    for (int k = m - 1; k >= 0; k--) y = y * x + coef[k];
    return y;
}

double tukeylambda_variance(double lam)
{
    static const double pc[5] = {3.289868133696453, 0.7306125098871127, -0.5370742306855439, 0.17292046290190008, -0.02371146284628187};
    static const double qc[5] = {1.0, 3.683605511659861, 4.184152498888124, 1.7660926747377275, 0.2643989311168465};
    if (lam < -0.5) return NaN;
    if (lam == -0.5) return INF;
    if (std::fabs(lam) < 0.075) return polyval_rev(pc, 5, lam) / polyval_rev(qc, 5, lam);
    return (2.0 / (lam * lam)) * (1.0 / (1.0 + 2 * lam) - sc::beta(lam + 1, lam + 1));
}

double tukeylambda_kurtosis(double lam)
{
    static const double pc[5] = {1.2, -5.853465139719495, -22.653447381131077, 0.20601184383406815, 4.59796302262789};
    static const double qc[5] = {1.0, 7.171149192233599, 12.96663094361842, 0.43075235247853005, -2.789746758009912};
    if (lam < -0.25) return NaN;
    if (lam == -0.25) return INF;
    if (std::fabs(lam) < 0.055) return polyval_rev(pc, 5, lam) / polyval_rev(qc, 5, lam);
    const double numer = 1.0 / (4 * lam + 1) - 4 * sc::beta(3 * lam + 1, lam + 1) + 3 * sc::beta(2 * lam + 1, 2 * lam + 1);
    const double t = 1.0 / (2 * lam + 1) - sc::beta(lam + 1, lam + 1);
    const double denom = 2 * (t * t);
    return numer / denom - 3;
}

struct Tukeylambda : Cont {
    Tukeylambda() { name = "tukeylambda"; shapes = "lam"; nshape = 1; doc = "A Tukey-Lamdba continuous random variable."; }
    bool argcheck(const double *s) const override { return std::isfinite(s[0]); }
    void get_support(const double *s, double &lo, double &hi) const override
    {
        const double lam = s[0];
        const double b = lam > 0 ? 1 / lam : INF;
        lo = -b;
        hi = b;
    }
    double pdf(double x, const double *s) const override
    {
        const double lam = s[0];
        const double Fx = sc::tklmbda(x, lam);
        double Px = std::pow(Fx, lam - 1.0) + std::pow(1 - Fx, lam - 1.0);
        Px = 1.0 / Px;
        return ((lam <= 0) | (std::fabs(x) < 1.0 / lam)) ? Px : 0.0;
    }
    double cdf(double x, const double *s) const override { return sc::tklmbda(x, s[0]); }
    double ppf(double q, const double *s) const override { return sc::boxcox(q, s[0]) - sc::boxcox1p(-q, s[0]); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        st.set(0, tukeylambda_variance(s[0]), 0, tukeylambda_kurtosis(s[0]));
    }
    double entropy(const double *s) const override
    {
        const double lam = s[0];
        return quad([&](double p) { return std::log(std::pow(p, lam - 1) + std::pow(1 - p, lam - 1)); }, 0.0, 1.0);
    }
};
Registrar r_tukeylambda(new Tukeylambda);

/* ---------------------------------------------------------------- vonmises, vonmises_line */

/* _stats.pyx: von_mises_cdf_series */
double von_mises_cdf_series(double k, double x, unsigned int p)
{
    const double s = std::sin(x), c = std::cos(x);
    double sn = std::sin(p * x), cn = std::cos(p * x);
    double R = 0, V = 0;
    for (unsigned int n = p - 1; n > 0; n--) {
        const double sn2 = sn * c - cn * s, cn2 = cn * c + sn * s;
        sn = sn2;
        cn = cn2;
        R = k / (2 * n + k * R);
        V = R * (sn / n + V);
    }
    return 0.5 + x / (2 * M_PI) + V / M_PI;
}

/* _stats.pyx: von_mises_cdf(k, x) for one element */
double von_mises_cdf(double k, double x)
{
    const double ix = std::nearbyint(x / (2 * M_PI));
    x = x - ix * (2 * M_PI);
    const double CK = 50;
    const double a1 = 28., a2 = 0.5, a3 = 100., a4 = 5.;
    double result;
    if (k < CK) {
        const unsigned int p = (unsigned int)(int)(1 + a1 + a2 * k - a3 / (k + a4));
        result = von_mises_cdf_series(k, x, p);
        result = result < 0 ? 0 : result > 1 ? 1 : result;
    } else {
        const double SQRT2_PI = 0.79788456080286535588;   /* sqrt(2/pi) */
        const double b = SQRT2_PI / sc::i0e(k);
        const double z = b * std::sin(x / 2.);
        result = sc::ndtr(z);                             /* scipy.stats.norm.cdf(z) */
    }
    return result + ix;
}

struct Vonmises : Cont {
    Vonmises(const char *nm, double lo, double hi)
    {
        name = nm;
        shapes = "kappa";
        nshape = 1;
        a = lo;
        b = hi;
        doc = "A Von Mises continuous random variable.";
    }
    bool argcheck(const double *s) const override { return s[0] >= 0; }
    double pdf(double x, const double *s) const override
    {
        const double kappa = s[0];
        return std::exp(kappa * sc::cosm1(x)) / (2 * M_PI * sc::i0e(kappa));
    }
    double logpdf(double x, const double *s) const override
    {
        const double kappa = s[0];
        return kappa * sc::cosm1(x) - std::log(2 * M_PI) - std::log(sc::i0e(kappa));
    }
    double cdf(double x, const double *s) const override { return von_mises_cdf(s[0], x); }
    double entropy(const double *s) const override
    {
        const double kappa = s[0];
        return (-kappa * sc::i1e(kappa) / sc::i0e(kappa) + std::log(2 * M_PI * sc::i0e(kappa)) + kappa);
    }
    /* _rvs, then the public rvs wraps loc + scale * X onto [-pi, pi): np.mod(rvs + pi, 2*pi) - pi */
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        const double *loc = sa[nshape], *scale = sa[nshape + 1];
        for (int64_t i = 0; i < n; i++) out[i] = random_vonmises(bg, 0.0, sa[0][i]);
        for (int64_t i = 0; i < n; i++) {
            const double y = out[i] * scale[i] + loc[i];
            out[i] = unscale(np_mod(y + M_PI, 2 * M_PI) - M_PI, loc[i], scale[i]);
        }
    }
};
Registrar r_vonmises(new Vonmises("vonmises", -INF, INF));
Registrar r_vonmises_line(new Vonmises("vonmises_line", -M_PI, M_PI));

/* ---------------------------------------------------------------- wald (invgauss methods at mu = 1.0) */

struct Wald : Cont {
    Wald() { name = "wald"; a = 0.0; doc = "A Wald continuous random variable."; }
    /* _support_mask = _open_support_mask: the public pdf is 0 (logpdf -inf) at the end points 0 and inf */
    double pdf(double x, const double *) const override
    {
        if (!(a < x && x < b)) return 0.0;
        const double mu = 1.0;
        return 1.0 / std::sqrt(2 * M_PI * std::pow(x, 3.0)) * std::exp(-1.0 / (2 * x) * sq(x / mu - 1));
    }
    double logpdf(double x, const double *) const override
    {
        if (!(a < x && x < b)) return -INF;
        const double mu = 1.0;
        return -0.5 * std::log(2 * M_PI) - 1.5 * std::log(x) - sq(x / mu - 1) / (2 * x);
    }
    double logcdf(double x, const double *) const override
    {
        const double mu = 1.0;
        const double fac = 1 / std::sqrt(x);
        const double a_ = sc::log_ndtr(fac * (x / mu - 1));
        const double b_ = 2 / mu + sc::log_ndtr(-fac * (x / mu + 1));
        return a_ + std::log1p(std::exp(b_ - a_));
    }
    double logsf(double x, const double *) const override
    {
        const double mu = 1.0;
        const double fac = 1 / std::sqrt(x);
        const double a_ = sc::log_ndtr(-(fac * (x / mu - 1)));
        const double b_ = 2 / mu + sc::log_ndtr(-fac * (x / mu + 1));
        return a_ + std::log1p(-std::exp(b_ - a_));
    }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double cdf(double x, const double *s) const override { return std::exp(logcdf(x, s)); }
    double ppf(double q, const double *s) const override
    {
        double r = q > 0.5 ? sc::_invgauss_isf(1 - q, 1.0, 1) : sc::_invgauss_ppf(q, 1.0, 1);
        if (std::isnan(r)) r = Cont::ppf(q, s);        /* rv_continuous._ppf (brentq on the cdf) */
        return r;
    }
    double isf(double q, const double *s) const override
    {
        double r = q > 0.5 ? sc::_invgauss_ppf(1 - q, 1.0, 1) : sc::_invgauss_isf(q, 1.0, 1);
        if (std::isnan(r)) r = ppf(1.0 - q, s);         /* rv_generic._isf */
        return r;
    }
    void stats(const double *, int, Stats4 &st) const override { st.set(1.0, 1.0, 3.0, 15.0); }
    double entropy(const double *) const override
    {
        const double mu = 1.0;
        const double a_ = 1. + std::log(2 * M_PI) + 3 * std::log(mu);
        const double r = 2 / mu;
        const double b_ = sc::_scaled_exp1(r) / r;
        return 0.5 * a_ - 1.5 * b_;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_wald(bg, 1.0, 1.0);
    }
};
Registrar r_wald(new Wald);

/* ---------------------------------------------------------------- weibull_max, weibull_min */

struct WeibullMax : Cont {
    WeibullMax() { name = "weibull_max"; shapes = "c"; nshape = 1; b = 0.0; doc = "Weibull maximum continuous random variable."; }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0];
        return c * std::pow(-x, c - 1) * std::exp(-std::pow(-x, c));
    }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0];
        return std::log(c) + sc::xlogy(c - 1, -x) - std::pow(-x, c);
    }
    double cdf(double x, const double *s) const override { return std::exp(-std::pow(-x, s[0])); }
    double logcdf(double x, const double *s) const override { return -std::pow(-x, s[0]); }
    double sf(double x, const double *s) const override { return -sc::expm1(-std::pow(-x, s[0])); }
    double ppf(double q, const double *s) const override { return -std::pow(-std::log(q), 1.0 / s[0]); }
    double munp(int n, const double *s) const override
    {
        const double val = sc::gamma(1.0 + n * 1.0 / s[0]);
        const int sgn = n % 2 ? -1 : 1;
        return sgn * val;
    }
    double entropy(const double *s) const override { return -EULER / s[0] - std::log(s[0]) + EULER + 1; }
};
Registrar r_weibull_max(new WeibullMax);

struct WeibullMin : Cont {
    WeibullMin() { name = "weibull_min"; shapes = "c"; nshape = 1; a = 0.0; doc = "Weibull minimum continuous random variable."; }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0];
        return c * std::pow(x, c - 1) * std::exp(-std::pow(x, c));
    }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0];
        return std::log(c) + sc::xlogy(c - 1, x) - std::pow(x, c);
    }
    double cdf(double x, const double *s) const override { return -sc::expm1(-std::pow(x, s[0])); }
    double ppf(double q, const double *s) const override { return std::pow(-sc::log1p(-q), 1.0 / s[0]); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return -std::pow(x, s[0]); }
    double isf(double q, const double *s) const override { return std::pow(-std::log(q), 1 / s[0]); }
    double munp(int n, const double *s) const override { return sc::gamma(1.0 + n * 1.0 / s[0]); }
    double entropy(const double *s) const override { return -EULER / s[0] - std::log(s[0]) + EULER + 1; }
};
Registrar r_weibull_min(new WeibullMin);

/* ---------------------------------------------------------------- wrapcauchy */

struct Wrapcauchy : Cont {
    Wrapcauchy() { name = "wrapcauchy"; shapes = "c"; nshape = 1; a = 0.0; b = 2 * M_PI; doc = "A wrapped Cauchy continuous random variable."; }
    bool argcheck(const double *s) const override { return (s[0] > 0) & (s[0] < 1); }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0];
        return (1.0 - c * c) / (2 * M_PI * (1 + c * c - 2 * c * std::cos(x)));
    }
    double cdf(double x, const double *s) const override
    {
        const double c = s[0];
        const double cr = (1 + c) / (1 - c);
        if (x < M_PI) return 1 / M_PI * std::atan(cr * std::tan(x / 2));
        return 1 - 1 / M_PI * std::atan(cr * std::tan((2 * M_PI - x) / 2));
    }
    double ppf(double q, const double *s) const override
    {
        const double c = s[0];
        const double val = (1.0 - c) / (1.0 + c);
        const double rcq = 2 * std::atan(val * std::tan(M_PI * q));
        const double rcmq = 2 * M_PI - 2 * std::atan(val * std::tan(M_PI * (1 - q)));
        return q < 1.0 / 2 ? rcq : rcmq;
    }
    double entropy(const double *s) const override { return std::log(2 * M_PI * (1 - s[0] * s[0])); }
    /* the generic _rvs, then the public rvs wraps loc + scale * X onto [0, 2*pi): np.mod(rvs, 2*pi) */
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        Cont::rvs(bg, n, sa, out);
        const double *loc = sa[nshape], *scale = sa[nshape + 1];
        for (int64_t i = 0; i < n; i++) {
            const double y = out[i] * scale[i] + loc[i];
            out[i] = unscale(np_mod(y, 2 * M_PI), loc[i], scale[i]);
        }
    }
};
Registrar r_wrapcauchy(new Wrapcauchy);

}  // namespace
}  // namespace tsd
