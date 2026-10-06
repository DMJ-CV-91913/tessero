/*
 * scipy.stats: alpha, anglit, arcsine, argus, betaprime, bradford, burr, burr12, cauchy, chi, cosine, crystalball,
 * dgamma, dpareto_lognorm, dweibull, erlang (scipy/stats/_continuous_distns.py), SciPy 1.17.1, private method by
 * private method. See stats_dist.hpp.
 */
#include "stats_dist.hpp"

#include <algorithm>

namespace tsd {
namespace {

/* scipy.stats._continuous_distns helpers and scipy.stats._constants */
const double NORM_PDF_C = std::sqrt(2 * M_PI);
const double NORM_PDF_LOGC = std::log(NORM_PDF_C);
const double EULER = 0.577215664901532860606512090082402431042;
const double LOG_PI = 1.1447298858494002;

inline double norm_pdf(double x) { return std::exp(-(x * x) / 2.0) / NORM_PDF_C; }
inline double norm_logpdf(double x) { return -(x * x) / 2.0 - NORM_PDF_LOGC; }
inline double norm_cdf(double x) { return sc::ndtr(x); }
inline double norm_sf(double x) { return norm_cdf(-x); }
inline double norm_ppf(double q) { return sc::ndtri(q); }

/* numpy's npy_logaddexp */
inline double logaddexp(double x, double y)
{
    if (x == y) return x + M_LN2;
    const double t = x - y;
    if (t > 0) return x + std::log1p(std::exp(-t));
    if (t <= 0) return y + std::log1p(std::exp(t));
    return t;
}

inline double np_sign(double x) { return std::isnan(x) ? x : (x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)); }

/* scipy.special.logsumexp([a0, a1], b=[b0, b1], return_sign=True) for real input */
double logsumexp2(double a0, double a1, double b0, double b1, double &sgn_out)
{
    const double sum_ = b0 * std::exp(a0) + b1 * std::exp(a1);
    const double sgn_inf = np_sign(sum_);
    const double out_inf = std::log(std::fabs(sum_));
    /* _logsumexp */
    if (b0 == 0) a0 = -INF;
    if (b1 == 0) a1 = -INF;
    const double a_max = (std::isnan(a0) || std::isnan(a1)) ? NaN : std::max(a0, a1);
    const bool i0 = a0 == a_max, i1 = a1 == a_max;
    if (i0) a0 = -INF;
    if (i1) a1 = -INF;
    double m = b0 * (double)i0 + b1 * (double)i1;
    double s = b0 * std::exp(a0 - a_max) + b1 * std::exp(a1 - a_max);
    s = s == 0 ? s : s / m;
    const double sgn = np_sign(s + 1) * np_sign(m);
    if (s < -1) s = -s - 2;
    m = std::fabs(m);
    const double out = std::log1p(s) + std::log(m) + a_max;
    if (std::isfinite(out)) {
        sgn_out = sgn;
        return out;
    }
    sgn_out = sgn_inf;
    return out_inf;
}

/* gamma_gen._entropy */
double gamma_entropy(double a)
{
    if (a < 250) return sc::psi(a) * (1 - a) + a + sc::gammaln(a);
    return 0.5 * (1. + std::log(2 * M_PI) + std::log(a)) - 1 / (3 * a) - std::pow(a, -2.) / 12 - std::pow(a, -3.) / 90 +
           std::pow(a, -4.) / 120;
}

/* weibull_min_gen._entropy */
inline double weibull_min_entropy(double c) { return -EULER / c - std::log(c) + EULER + 1; }

struct Alpha : Cont {
    Alpha() { name = "alpha"; shapes = "a"; nshape = 1; a = 0.0; doc = "An alpha continuous random variable."; }
    /* _support_mask = _open_support_mask: the public pdf is 0 at the end points */
    double pdf(double x, const double *s) const override
    {
        if (!(x > 0 && x < INF)) return 0.0;
        return 1.0 / (x * x) / norm_cdf(s[0]) * norm_pdf(s[0] - 1.0 / x);
    }
    double logpdf(double x, const double *s) const override
    {
        if (!(x > 0 && x < INF)) return -INF;
        return -2 * std::log(x) + norm_logpdf(s[0] - 1.0 / x) - std::log(norm_cdf(s[0]));
    }
    double cdf(double x, const double *s) const override { return norm_cdf(s[0] - 1.0 / x) / norm_cdf(s[0]); }
    double ppf(double q, const double *s) const override { return 1.0 / (s[0] - norm_ppf(q * norm_cdf(s[0]))); }
    void stats(const double *, int, Stats4 &st) const override { st.set(INF, INF, NaN, NaN); }
};
Registrar r_alpha(new Alpha);

struct Anglit : Cont {
    Anglit() { name = "anglit"; a = -M_PI / 4; b = M_PI / 4; doc = "An anglit continuous random variable."; }
    double pdf(double x, const double *) const override { return std::cos(2 * x); }
    double cdf(double x, const double *) const override { return sq(std::sin(x + M_PI / 4)); }
    double sf(double x, const double *) const override { return sq(std::cos(x + M_PI / 4)); }
    double ppf(double q, const double *) const override { return std::asin(std::sqrt(q)) - M_PI / 4; }
    void stats(const double *, int, Stats4 &st) const override
    {
        st.set(0.0, M_PI * M_PI / 16 - 0.5, 0.0, -2 * (std::pow(M_PI, 4.0) - 96) / std::pow(M_PI * M_PI - 8, 2.0));
    }
    double entropy(const double *) const override { return 1 - std::log(2); }
};
Registrar r_anglit(new Anglit);

struct Arcsine : Cont {
    Arcsine() { name = "arcsine"; a = 0.0; b = 1.0; doc = "An arcsine continuous random variable."; }
    double pdf(double x, const double *) const override { return 1.0 / M_PI / std::sqrt(x * (1 - x)); }
    double cdf(double x, const double *) const override { return 2.0 / M_PI * std::asin(std::sqrt(x)); }
    double ppf(double q, const double *) const override { return sq(std::sin(M_PI / 2.0 * q)); }
    void stats(const double *, int, Stats4 &st) const override { st.set(0.5, 1.0 / 8, 0, -3.0 / 2.0); }
    double entropy(const double *) const override { return -0.24156447527049044468; }
};
Registrar r_arcsine(new Arcsine);

inline double argus_phi(double chi) { return sc::gammainc(1.5, chi * chi / 2) / 2; }

struct Argus : Cont {
    Argus() { name = "argus"; shapes = "chi"; nshape = 1; a = 0.0; b = 1.0; doc = "Argus distribution"; }
    double logpdf(double x, const double *s) const override
    {
        const double chi = s[0];
        const double y = 1.0 - x * x;
        const double A = 3 * std::log(chi) - NORM_PDF_LOGC - std::log(argus_phi(chi));
        return A + std::log(x) + 0.5 * std::log1p(-x * x) - chi * chi * y / 2;
    }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double cdf(double x, const double *s) const override { return 1.0 - sf(x, s); }
    double sf(double x, const double *s) const override
    {
        const double chi = s[0];
        return argus_phi(chi * std::sqrt((1 - x) * (1 + x))) / argus_phi(chi);
    }
    /* _rvs_scalar: rejection sampling, each round draws whole arrays of the missing size */
    static void rvs_scalar(bitgen_t *bg, double chi, int64_t N, double *x)
    {
        int64_t simulated = 0;
        const double chi2 = chi * chi;
        std::vector<double> u((size_t)N), v((size_t)N);
        if (chi <= 0.5) {
            const double d = -chi2 / 2;
            while (simulated < N) {
                const int64_t k = N - simulated;
                for (int64_t i = 0; i < k; i++) u[(size_t)i] = random_standard_uniform(bg);
                for (int64_t i = 0; i < k; i++) v[(size_t)i] = random_standard_uniform(bg);
                for (int64_t i = 0; i < k; i++) {
                    const double z = std::pow(v[(size_t)i], 2.0 / 3);
                    if (std::log(u[(size_t)i]) <= d * z) x[simulated++] = std::sqrt(1 - z);
                }
            }
        } else if (chi <= 1.8) {
            const double echi = std::exp(-chi2 / 2);
            while (simulated < N) {
                const int64_t k = N - simulated;
                for (int64_t i = 0; i < k; i++) u[(size_t)i] = random_standard_uniform(bg);
                for (int64_t i = 0; i < k; i++) v[(size_t)i] = random_standard_uniform(bg);
                for (int64_t i = 0; i < k; i++) {
                    const double vi = v[(size_t)i];
                    const double z = 2 * std::log(echi * (1 - vi) + vi) / chi2;
                    if (u[(size_t)i] * u[(size_t)i] + z <= 0) x[simulated++] = std::sqrt(1 + z);
                }
            }
        } else {
            while (simulated < N) {
                const int64_t k = N - simulated;
                for (int64_t i = 0; i < k; i++) u[(size_t)i] = random_standard_gamma(bg, 1.5);
                for (int64_t i = 0; i < k; i++)
                    if (u[(size_t)i] <= chi2 / 2) x[simulated++] = u[(size_t)i];
            }
            for (int64_t i = 0; i < N; i++) x[i] = std::sqrt(1 - 2 * x[i] / chi2);
        }
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* a scalar chi: one call for all samples; an array of chi: one sample per element, in order */
        bool scalar = true;
        for (int64_t i = 1; i < n; i++)
            if (!(sa[0][i] == sa[0][0])) scalar = false;
        if (scalar) {
            if (n > 0) rvs_scalar(bg, sa[0][0], n, out);
            return;
        }
        for (int64_t i = 0; i < n; i++) rvs_scalar(bg, sa[0][i], 1, out + i);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double chi = s[0];
        const double phi = argus_phi(chi);
        const double m = std::sqrt(M_PI / 8) * chi * sc::ive(1, chi * chi / 4) / phi;
        double mu2;
        if (chi > 0.1) {
            const double c = chi;
            mu2 = 1 - 3 / (c * c) + c * norm_pdf(c) / phi;
        } else {
            static const double coef[9] = {-358 / 65690625., 0, -94 / 1010625., 0, 2 / 2625., 0, 6 / 175., 0, 0.4};
            mu2 = 0.0;
            for (double pv : coef) mu2 = mu2 * chi + pv;   /* np.polyval */
        }
        st.set(0, m);
        st.set(1, mu2 - m * m);
    }
};
Registrar r_argus(new Argus);

struct Betaprime : Cont {
    Betaprime() { name = "betaprime"; shapes = "a, b"; nshape = 2; a = 0.0; doc = "A beta prime continuous random variable."; }
    /* _support_mask = _open_support_mask: the public pdf is 0 at the end points */
    double pdf(double x, const double *s) const override
    {
        if (!(x > 0 && x < INF)) return 0.0;
        return std::exp(logpdf(x, s));
    }
    double logpdf(double x, const double *s) const override
    {
        if (!(x > 0 && x < INF)) return -INF;
        return sc::xlogy(s[0] - 1.0, x) - sc::xlog1py(s[0] + s[1], x) - sc::betaln(s[0], s[1]);
    }
    double cdf(double x, const double *s) const override
    {
        return x > 1 ? sc::betaincc(s[1], s[0], 1 / (1 + x)) : sc::betainc(s[0], s[1], x / (1 + x));
    }
    double sf(double x, const double *s) const override
    {
        return x > 1 ? sc::betainc(s[1], s[0], 1 / (1 + x)) : sc::betaincc(s[0], s[1], x / (1 + x));
    }
    double ppf(double p, const double *s) const override
    {
        const double r = sc::_beta_ppf(p, s[0], s[1]);
        if (r > 0.9999) return 1 / sc::betainccinv(s[1], s[0], p) - 1;
        return r / (1 - r);
    }
    double munp(int n, const double *s) const override
    {
        const double a = s[0], b = s[1];
        if (!(b > n)) return INF;
        double prod = 1.0;
        for (int i = 1; i <= n; i++) prod *= (a + i - 1) / (b - i);
        return prod;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<double> u1((size_t)n);
        for (int64_t i = 0; i < n; i++) u1[(size_t)i] = random_standard_gamma(bg, sa[0][i]);
        for (int64_t i = 0; i < n; i++) out[i] = u1[(size_t)i] / random_standard_gamma(bg, sa[1][i]);
    }
};
Registrar r_betaprime(new Betaprime);

struct Bradford : Cont {
    Bradford() { name = "bradford"; shapes = "c"; nshape = 1; a = 0.0; b = 1.0; doc = "A Bradford continuous random variable."; }
    double pdf(double x, const double *s) const override { return s[0] / (s[0] * x + 1.0) / sc::log1p(s[0]); }
    double cdf(double x, const double *s) const override { return sc::log1p(s[0] * x) / sc::log1p(s[0]); }
    double ppf(double q, const double *s) const override { return sc::expm1(q * sc::log1p(s[0])) / s[0]; }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double c = s[0];
        const double k = std::log(1.0 + c);
        st.set(0, (c - k) / (c * k));
        st.set(1, ((c + 2.0) * k - 2.0 * c) / (2 * c * k * k));
        if (moments & MOM_S) {
            double g1 = std::sqrt(2) * (12 * c * c - 9 * c * k * (c + 2) + 2 * k * k * (c * (c + 3) + 3));
            g1 /= std::sqrt(c * (c * (k - 2) + 2 * k)) * (3 * c * (k - 2) + 6 * k);
            st.set(2, g1);
        }
        if (moments & MOM_K) {
            double g2 = std::pow(c, 3.0) * (k - 3) * (k * (3 * k - 16) + 24) + 12 * k * c * c * (k - 4) * (k - 3) +
                        6 * c * k * k * (3 * k - 14) + 12 * std::pow(k, 3.0);
            g2 /= 3 * c * sq(c * (k - 2) + 2 * k);
            st.set(3, g2);
        }
    }
    double entropy(const double *s) const override
    {
        const double c = s[0];
        const double k = std::log(1 + c);
        return k / 2.0 - std::log(c / k);
    }
};
Registrar r_bradford(new Bradford);

struct Burr : Cont {
    Burr() { name = "burr"; shapes = "c, d"; nshape = 2; a = 0.0; doc = "A Burr (Type III) continuous random variable."; }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0], d = s[1];
        if (x == 0) return c * d * std::pow(x, c * d - 1) / (1 + std::pow(x, c));
        return c * d * std::pow(x, -c - 1.0) / std::pow(1 + std::pow(x, -c), d + 1.0);
    }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0], d = s[1];
        if (x == 0) return std::log(c) + std::log(d) + sc::xlogy(c * d - 1, x) - (d + 1) * sc::log1p(std::pow(x, c));
        return std::log(c) + std::log(d) + sc::xlogy(-c - 1, x) - sc::xlog1py(d + 1, std::pow(x, -c));
    }
    double cdf(double x, const double *s) const override { return std::pow(1 + std::pow(x, -s[0]), -s[1]); }
    double logcdf(double x, const double *s) const override { return sc::log1p(std::pow(x, -s[0])) * (-s[1]); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return std::log1p(-std::pow(1 + std::pow(x, -s[0]), -s[1])); }
    double ppf(double q, const double *s) const override { return std::pow(std::pow(q, -1.0 / s[1]) - 1, -1.0 / s[0]); }
    double isf(double q, const double *s) const override
    {
        const double q_ = sc::xlog1py(-1.0 / s[1], -q);
        return std::pow(sc::expm1(q_), -1.0 / s[0]);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double c = s[0], d = s[1];
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
    double munp(int n, const double *s) const override
    {
        const double c = s[0], d = s[1];
        if (!(c > n && d == d)) return NaN;
        const double nc = 1. * n / c;
        return d * sc::beta(1.0 - nc, d + nc);
    }
};
Registrar r_burr(new Burr);

struct Burr12 : Cont {
    Burr12() { name = "burr12"; shapes = "c, d"; nshape = 2; a = 0.0; doc = "A Burr (Type XII) continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0], d = s[1];
        return std::log(c) + std::log(d) + sc::xlogy(c - 1, x) + sc::xlog1py(-d - 1, std::pow(x, c));
    }
    double cdf(double x, const double *s) const override { return -sc::expm1(logsf(x, s)); }
    double logcdf(double x, const double *s) const override { return sc::log1p(-std::pow(1 + std::pow(x, s[0]), -s[1])); }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double logsf(double x, const double *s) const override { return sc::xlog1py(-s[1], std::pow(x, s[0])); }
    double ppf(double q, const double *s) const override { return std::pow(sc::expm1(-1 / s[1] * sc::log1p(-q)), 1 / s[0]); }
    double isf(double p, const double *s) const override { return std::pow(sc::expm1(-1 / s[1] * std::log(p)), 1 / s[0]); }
    double munp(int n, const double *s) const override
    {
        const double c = s[0], d = s[1];
        if (!(c * d > n)) return NaN;
        const double nc = 1. * n / c;
        return d * sc::beta(1.0 + nc, d - nc);
    }
};
Registrar r_burr12(new Burr12);

struct Cauchy : Cont {
    Cauchy() { name = "cauchy"; doc = "A Cauchy continuous random variable."; }
    double pdf(double x, const double *) const override { return 1.0 / M_PI / (1.0 + x * x); }
    double logpdf(double x, const double *) const override
    {
        const double absx = std::fabs(x);
        if (absx < 1) return -LOG_PI - std::log1p(absx * absx);
        const double r = 1 / absx;
        return -LOG_PI - (2 * std::log(absx) + std::log1p(r * r));
    }
    double cdf(double x, const double *) const override { return std::atan2(1, -x) / M_PI; }
    double ppf(double q, const double *) const override { return sc::_cauchy_ppf(q, 0, 1); }
    double sf(double x, const double *) const override { return std::atan2(1, x) / M_PI; }
    double isf(double q, const double *) const override { return sc::_cauchy_isf(q, 0, 1); }
    void stats(const double *, int, Stats4 &st) const override { st.set(NaN, NaN, NaN, NaN); }
    double entropy(const double *) const override { return std::log(4 * M_PI); }
};
Registrar r_cauchy(new Cauchy);

struct Chi : Cont {
    Chi() { name = "chi"; shapes = "df"; nshape = 1; a = 0.0; doc = "A chi continuous random variable."; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = std::sqrt(random_chisquare(bg, sa[0][i]));
    }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double df = s[0];
        const double l = std::log(2) - .5 * std::log(2) * df - sc::gammaln(.5 * df);
        return l + sc::xlogy(df - 1., x) - .5 * (x * x);
    }
    double cdf(double x, const double *s) const override { return sc::gammainc(.5 * s[0], .5 * (x * x)); }
    double sf(double x, const double *s) const override { return sc::gammaincc(.5 * s[0], .5 * (x * x)); }
    double ppf(double q, const double *s) const override { return std::sqrt(2 * sc::gammaincinv(.5 * s[0], q)); }
    double isf(double q, const double *s) const override { return std::sqrt(2 * sc::gammainccinv(.5 * s[0], q)); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double df = s[0];
        const double mu = std::sqrt(2) * sc::poch(0.5 * df, 0.5);
        const double mu2 = df - mu * mu;
        const double g1 = (2 * std::pow(mu, 3.0) + mu * (1 - 2 * df)) / std::pow(mu2, 1.5);
        double g2 = 2 * df * (1.0 - df) - 6 * std::pow(mu, 4.0) + 4 * (mu * mu) * (2 * df - 1);
        g2 /= mu2 * mu2;
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *s) const override
    {
        const double df = s[0];
        if (df < 300) return sc::gammaln(.5 * df) + 0.5 * (df - std::log(2) - (df - 1) * sc::psi(0.5 * df));
        return 0.5 + std::log(M_PI) / 2 - (1 / df) / 6 - std::pow(df, -2.0) / 6 - 4.0 / 45 * std::pow(df, -3.0) +
               std::pow(df, -4.0) / 15;
    }
};
Registrar r_chi(new Chi);

struct Cosine : Cont {
    Cosine() { name = "cosine"; a = -M_PI; b = M_PI; doc = "A cosine continuous random variable."; }
    double pdf(double x, const double *) const override { return 1.0 / 2 / M_PI * (1 + std::cos(x)); }
    double logpdf(double x, const double *) const override
    {
        const double c = std::cos(x);
        return c != -1 ? std::log1p(c) - std::log(2 * M_PI) : -INF;
    }
    double cdf(double x, const double *) const override { return sc::_cosine_cdf(x); }
    double sf(double x, const double *) const override { return sc::_cosine_cdf(-x); }
    double ppf(double p, const double *) const override { return sc::_cosine_invcdf(p); }
    double isf(double p, const double *) const override { return -sc::_cosine_invcdf(p); }
    void stats(const double *, int, Stats4 &st) const override
    {
        const double v = (M_PI * M_PI / 3.0) - 2.0;
        const double k = -6.0 * (std::pow(M_PI, 4.0) - 90) / (5.0 * std::pow(M_PI * M_PI - 6, 2.0));
        st.set(0.0, v, 0.0, k);
    }
    double entropy(const double *) const override { return std::log(4 * M_PI) - 1.0; }
};
Registrar r_cosine(new Cosine);

struct Crystalball : Cont {
    Crystalball() { name = "crystalball"; shapes = "beta, m"; nshape = 2; doc = "Crystalball distribution"; }
    bool argcheck(const double *s) const override { return s[1] > 1 && s[0] > 0; }
    static double normconst(double beta, double m)
    {
        return 1.0 / (m / beta / (m - 1) * std::exp(-(beta * beta) / 2.0) + NORM_PDF_C * norm_cdf(beta));
    }
    double pdf(double x, const double *s) const override
    {
        const double beta = s[0], m = s[1];
        const double N = normconst(beta, m);
        if (x > -beta) return N * std::exp(-(x * x) / 2);
        return N * (std::pow(m / beta, m) * std::exp(-(beta * beta) / 2.0) * std::pow(m / beta - beta - x, -m));
    }
    double logpdf(double x, const double *s) const override
    {
        const double beta = s[0], m = s[1];
        const double N = normconst(beta, m);
        if (x > -beta) return std::log(N) + -(x * x) / 2;
        return std::log(N) + (m * std::log(m / beta) - beta * beta / 2 - m * std::log(m / beta - beta - x));
    }
    double cdf(double x, const double *s) const override
    {
        const double beta = s[0], m = s[1];
        const double N = normconst(beta, m);
        if (x > -beta)
            return N * ((m / beta) * std::exp(-(beta * beta) / 2.0) / (m - 1) + NORM_PDF_C * (norm_cdf(x) - norm_cdf(-beta)));
        return N * (std::pow(m / beta, m) * std::exp(-(beta * beta) / 2.0) * std::pow(m / beta - beta - x, -m + 1) / (m - 1));
    }
    double sf(double x, const double *s) const override
    {
        const double beta = s[0], m = s[1];
        if (x > -beta) {
            const double M = m / beta / (m - 1) * std::exp(-(beta * beta) / 2) + NORM_PDF_C * norm_cdf(beta);
            return NORM_PDF_C * norm_sf(x) / M;
        }
        return 1 - cdf(x, s);
    }
    double ppf(double p, const double *s) const override
    {
        const double beta = s[0], m = s[1];
        const double N0 = normconst(beta, m);
        const double pbeta = N0 * (m / beta) * std::exp(-(beta * beta) / 2) / (m - 1);
        const double eb2 = std::exp(-(beta * beta) / 2);
        const double C = (m / beta) * eb2 / (m - 1);
        const double N = 1 / (C + NORM_PDF_C * norm_cdf(beta));
        if (p < pbeta) return m / beta - beta - std::pow((m - 1) * std::pow(m / beta, -m) / eb2 * p / N, 1 / (1 - m));
        return norm_ppf(norm_cdf(-beta) + (1 / NORM_PDF_C) * (p / N - C));
    }
    double munp(int n, const double *s) const override
    {
        const double beta = s[0], m = s[1];
        const double N = normconst(beta, m);
        if (!(n + 1 < m)) return N * INF;
        /* n_th_moment, called through np.vectorize: n a Python int, beta and m Python floats */
        const double A = std::pow(m / beta, m) * std::exp(-std::pow(beta, 2.0) / 2.0);
        const double B = m / beta - beta;
        const double sgn_n = (n % 2) ? -1.0 : 1.0;
        const double rhs = std::pow(2.0, (n - 1) / 2.0) * sc::gamma((n + 1) / 2.0) *
                           (1.0 + sgn_n * sc::gammainc((n + 1) / 2.0, std::pow(beta, 2.0) / 2));
        double lhs = 0.0;
        for (int k = 0; k <= n; k++) {
            const double sgn_k = (k % 2) ? -1.0 : 1.0;
            lhs += sc::binom(n, k) * std::pow(B, (double)(n - k)) * sgn_k / (m - k - 1) * std::pow(m / beta, -m + k + 1);
        }
        return N * (A * lhs + rhs);
    }
};
Registrar r_crystalball(new Crystalball);

struct Dgamma : Cont {
    Dgamma() { name = "dgamma"; shapes = "a"; nshape = 1; doc = "A double gamma continuous random variable."; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<double> u((size_t)n);
        for (int64_t i = 0; i < n; i++) u[(size_t)i] = random_standard_uniform(bg);
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_gamma(bg, sa[0][i]) * (u[(size_t)i] >= 0.5 ? 1 : -1);
    }
    double pdf(double x, const double *s) const override
    {
        const double a = s[0], ax = std::fabs(x);
        return 1.0 / (2 * sc::gamma(a)) * std::pow(ax, a - 1.0) * std::exp(-ax);
    }
    double logpdf(double x, const double *s) const override
    {
        const double a = s[0], ax = std::fabs(x);
        return sc::xlogy(a - 1.0, ax) - ax - std::log(2) - sc::gammaln(a);
    }
    double cdf(double x, const double *s) const override
    {
        return x > 0 ? 0.5 + 0.5 * sc::gammainc(s[0], x) : 0.5 * sc::gammaincc(s[0], -x);
    }
    double sf(double x, const double *s) const override
    {
        return x > 0 ? 0.5 * sc::gammaincc(s[0], x) : 0.5 + 0.5 * sc::gammainc(s[0], -x);
    }
    double entropy(const double *s) const override { return gamma_entropy(s[0]) - std::log(0.5); }
    double ppf(double q, const double *s) const override
    {
        return q > 0.5 ? sc::gammaincinv(s[0], 2 * q - 1) : -sc::gammainccinv(s[0], 2 * q);
    }
    double isf(double q, const double *s) const override
    {
        return q > 0.5 ? -sc::gammaincinv(s[0], 2 * q - 1) : sc::gammainccinv(s[0], 2 * q);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double a = s[0];
        const double mu2 = a * (a + 1.0);
        st.set(0.0, mu2, 0.0, (a + 2.0) * (a + 3.0) / mu2 - 3.0);
    }
};
Registrar r_dgamma(new Dgamma);

struct DparetoLognorm : Cont {
    DparetoLognorm()
    {
        name = "dpareto_lognorm"; shapes = "u, s, a, b"; nshape = 4; a = 0.0;
        doc = "A double Pareto lognormal continuous random variable.";
    }
    /* _logphi = norm._logpdf, _logPhi = norm._logcdf, _logPhic = norm._logsf */
    static double logphi(double z) { return norm_logpdf(z); }
    static double logR(double z) { return sc::log_ndtr(-z) - logphi(z); }
    bool argcheck(const double *p) const override { return p[1] > 0 && p[2] > 0 && p[3] > 0; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        std::vector<double> Z((size_t)n), E1((size_t)n);
        for (int64_t i = 0; i < n; i++) Z[(size_t)i] = random_normal(bg, sa[0][i], sa[1][i]);
        for (int64_t i = 0; i < n; i++) E1[(size_t)i] = random_standard_exponential(bg);
        for (int64_t i = 0; i < n; i++) {
            const double E2 = random_standard_exponential(bg);
            out[i] = std::exp(Z[(size_t)i] + E1[(size_t)i] / sa[2][i] - E2 / sa[3][i]);
        }
    }
    double logpdf(double x, const double *p) const override
    {
        const double u = p[0], s = p[1], a = p[2], b = p[3];
        if (x == 0 || std::isinf(x)) return -INF;
        const double log_y = std::log(x), m = u;
        const double z = (log_y - m) / s;
        const double x1 = a * s - z;
        const double x2 = b * s + z;
        double out = std::log(a) + std::log(b) - std::log(a + b) - log_y;
        out += logphi(z);
        out += logaddexp(logR(x1), logR(x2));
        return out;
    }
    double logcdf(double x, const double *p) const override
    {
        const double u = p[0], s = p[1], a = p[2], b = p[3];
        if (x == 0) return -INF;
        const double log_y = std::log(x), m = u;
        const double z = (log_y - m) / s;
        const double x1 = a * s - z;
        const double x2 = b * s + z;
        const double t1 = sc::log_ndtr(z);
        const double t2 = logphi(z);
        const double t3 = std::log(b) + logR(x1);
        const double t4 = std::log(a) + logR(x2);
        double sign, sign2;
        const double t5 = logsumexp2(t3, t4, 1.0, -1.0, sign);
        return logsumexp2(t1, t2 + t5 - std::log(a + b), 1.0, -1.0 * sign, sign2);
    }
    double logsf(double x, const double *p) const override { return sc::_log1mexp(logcdf(x, p)); }
    double pdf(double x, const double *p) const override { return std::exp(logpdf(x, p)); }
    double cdf(double x, const double *p) const override { return std::exp(logcdf(x, p)); }
    double sf(double x, const double *p) const override { return std::exp(logsf(x, p)); }
    double munp(int n, const double *p) const override
    {
        const double m = p[0], s = p[1], a = p[2], b = p[3], k = (double)n;
        if (a <= k) return NaN;
        return (a * b) / ((a - k) * (b + k)) * std::exp(k * m + (k * k) * (s * s) / 2);
    }
};
Registrar r_dpareto_lognorm(new DparetoLognorm);

struct Dweibull : Cont {
    Dweibull() { name = "dweibull"; shapes = "c"; nshape = 1; doc = "A double Weibull continuous random variable."; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        /* u = uniform(size); w = weibull_min.rvs(c): uniform(size), then weibull_min._ppf */
        std::vector<double> u((size_t)n);
        for (int64_t i = 0; i < n; i++) u[(size_t)i] = random_standard_uniform(bg);
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_uniform(bg);
        for (int64_t i = 0; i < n; i++) {
            const double w = std::pow(-sc::log1p(-out[i]), 1.0 / sa[0][i]);
            out[i] = w * (u[(size_t)i] >= 0.5 ? 1 : -1);
        }
    }
    double pdf(double x, const double *s) const override
    {
        const double c = s[0], ax = std::fabs(x);
        return c / 2.0 * std::pow(ax, c - 1.0) * std::exp(-std::pow(ax, c));
    }
    double logpdf(double x, const double *s) const override
    {
        const double c = s[0], ax = std::fabs(x);
        return std::log(c) - std::log(2.0) + sc::xlogy(c - 1.0, ax) - std::pow(ax, c);
    }
    double cdf(double x, const double *s) const override
    {
        const double Cx1 = 0.5 * std::exp(-std::pow(std::fabs(x), s[0]));
        return x > 0 ? 1 - Cx1 : Cx1;
    }
    double ppf(double q, const double *s) const override
    {
        double fac = 2. * (q <= 0.5 ? q : 1. - q);
        fac = std::pow(-std::log(fac), 1.0 / s[0]);
        return q > 0.5 ? fac : -fac;
    }
    double sf(double x, const double *s) const override
    {
        const double h = 0.5 * std::exp(-std::pow(std::fabs(x), s[0]));   /* weibull_min._sf = exp(-pow(x, c)) */
        return x > 0 ? h : 1 - h;
    }
    double isf(double q, const double *s) const override
    {
        const double double_q = 2. * (q <= 0.5 ? q : 1. - q);
        const double w = std::pow(-std::log(double_q), 1 / s[0]);   /* weibull_min._isf */
        return q > 0.5 ? -w : w;
    }
    double munp(int n, const double *s) const override { return (double)(1 - (n % 2)) * sc::gamma(1.0 + 1.0 * n / s[0]); }
    void stats(const double *, int, Stats4 &st) const override
    {
        st.set(0, 0.0);
        st.set(2, 0.0);
    }
    double entropy(const double *s) const override { return weibull_min_entropy(s[0]) - std::log(0.5); }
};
Registrar r_dweibull(new Dweibull);

/* erlang_gen(gamma_gen): gamma's methods */
struct Erlang : Cont {
    Erlang() { name = "erlang"; shapes = "a"; nshape = 1; a = 0.0; doc = "An Erlang continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_gamma(bg, sa[0][i]);
    }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override { return sc::xlogy(s[0] - 1.0, x) - x - sc::gammaln(s[0]); }
    double cdf(double x, const double *s) const override { return sc::gammainc(s[0], x); }
    double sf(double x, const double *s) const override { return sc::gammaincc(s[0], x); }
    double ppf(double q, const double *s) const override { return sc::gammaincinv(s[0], q); }
    double isf(double q, const double *s) const override { return sc::gammainccinv(s[0], q); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double a = s[0];
        st.set(a, a, 2.0 / std::sqrt(a), 6.0 / a);
    }
    double munp(int n, const double *s) const override { return sc::poch(s[0], n); }
    double entropy(const double *s) const override { return gamma_entropy(s[0]); }
};
Registrar r_erlang(new Erlang);

}  // namespace
}  // namespace tsd
