/*
 * scipy.stats: norm, expon, uniform, gamma, beta, t, chi2, f (scipy/stats/_continuous_distns.py) and binom, poisson
 * (scipy/stats/_discrete_distns.py), SciPy 1.17.1, private method by private method. See stats_dist.hpp.
 */
#include "stats_dist.hpp"

namespace tsd {
namespace {

const double NORM_PDF_C = std::sqrt(2 * M_PI);
const double NORM_PDF_LOGC = std::log(NORM_PDF_C);

struct Norm : Cont {
    Norm() { name = "norm"; doc = "A normal continuous random variable."; }
    double pdf(double x, const double *) const override { return std::exp(-(x * x) / 2.0) / NORM_PDF_C; }
    double logpdf(double x, const double *) const override { return -(x * x) / 2.0 - NORM_PDF_LOGC; }
    double cdf(double x, const double *) const override { return sc::ndtr(x); }
    double logcdf(double x, const double *) const override { return sc::log_ndtr(x); }
    double sf(double x, const double *) const override { return sc::ndtr(-x); }
    double logsf(double x, const double *) const override { return sc::log_ndtr(-x); }
    double ppf(double q, const double *) const override { return sc::ndtri(q); }
    double isf(double q, const double *) const override { return -sc::ndtri(q); }
    void stats(const double *, int, Stats4 &st) const override { st.set(0.0, 1.0, 0.0, 0.0); }
    double entropy(const double *) const override { return 0.5 * (std::log(2 * M_PI) + 1); }
    double munp(int n, const double *) const override
    {
        if (n == 0) return 1.0;
        if (n % 2 == 0) return factorial2(n - 1);   /* sc.factorial2(int(n) - 1) */
        return 0.0;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_normal(bg);
    }
};
Registrar r_norm(new Norm);

struct Expon : Cont {
    Expon() { name = "expon"; a = 0.0; doc = "An exponential continuous random variable."; }
    double pdf(double x, const double *) const override { return std::exp(-x); }
    double logpdf(double x, const double *) const override { return -x; }
    double cdf(double x, const double *) const override { return -sc::expm1(-x); }
    double ppf(double q, const double *) const override { return -sc::log1p(-q); }
    double sf(double x, const double *) const override { return std::exp(-x); }
    double logsf(double x, const double *) const override { return -x; }
    double isf(double q, const double *) const override { return -std::log(q); }
    void stats(const double *, int, Stats4 &st) const override { st.set(1.0, 1.0, 2.0, 6.0); }
    double entropy(const double *) const override { return 1.0; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_exponential(bg);
    }
};
Registrar r_expon(new Expon);

struct Uniform : Cont {
    Uniform() { name = "uniform"; a = 0.0; b = 1.0; doc = "A uniform continuous random variable."; }
    double pdf(double x, const double *) const override { return 1.0 * (x == x); }
    double cdf(double x, const double *) const override { return x; }
    double ppf(double q, const double *) const override { return q; }
    void stats(const double *, int, Stats4 &st) const override { st.set(0.5, 1.0 / 12, 0, -1.2); }
    double entropy(const double *) const override { return 0.0; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_uniform(bg, 0.0, 1.0);
    }
};
Registrar r_uniform(new Uniform);

struct Gamma : Cont {
    Gamma() { name = "gamma"; shapes = "a"; nshape = 1; a = 0.0; doc = "A gamma continuous random variable."; }
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
    double entropy(const double *s) const override
    {
        const double a = s[0];
        if (a < 250) return sc::psi(a) * (1 - a) + a + sc::gammaln(a);
        return 0.5 * (1. + std::log(2 * M_PI) + std::log(a)) - 1 / (3 * a) - std::pow(a, -2.) / 12 - std::pow(a, -3.) / 90 +
               std::pow(a, -4.) / 120;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_gamma(bg, sa[0][i]);
    }
};
Registrar r_gamma(new Gamma);

struct Beta : Cont {
    Beta() { name = "beta"; shapes = "a, b"; nshape = 2; a = 0.0; b = 1.0; doc = "A beta continuous random variable."; }
    double pdf(double x, const double *s) const override { return sc::_beta_pdf(x, s[0], s[1]); }
    double logpdf(double x, const double *s) const override
    {
        double lPx = sc::xlog1py(s[1] - 1.0, -x) + sc::xlogy(s[0] - 1.0, x);
        lPx -= sc::betaln(s[0], s[1]);
        return lPx;
    }
    double cdf(double x, const double *s) const override { return sc::betainc(s[0], s[1], x); }
    double sf(double x, const double *s) const override { return sc::betaincc(s[0], s[1], x); }
    double isf(double x, const double *s) const override { return sc::betainccinv(s[0], s[1], x); }
    double ppf(double q, const double *s) const override { return sc::_beta_ppf(q, s[0], s[1]); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double a = s[0], b = s[1];
        const double apb = a + b;
        const double mean = a / apb;
        const double var = a * b / (apb * apb * (apb + 1));
        const double skew = (2 * (b - a) * std::sqrt(apb + 1)) / ((apb + 2) * std::sqrt(a * b));
        const double kn = 6 * ((a - b) * (a - b) * (apb + 1) - a * b * (apb + 2));
        const double kd = a * b * (apb + 2) * (apb + 3);
        st.set(mean, var, skew, kn / kd);
    }
    double entropy(const double *s) const override
    {
        const double a = s[0], b = s[1];
        auto regular = [](double a, double b) {
            return sc::betaln(a, b) - (a - 1) * sc::psi(a) - (b - 1) * sc::psi(b) + (a + b - 2) * sc::psi(a + b);
        };
        auto asymptotic_ab_large = [](double a, double b) {
            const double sum_ab = a + b;
            const double log_term = 0.5 * (std::log(2 * M_PI) + std::log(a) + std::log(b) - 3 * std::log(sum_ab) + 1);
            const double t1 = 110 / sum_ab + 20 * std::pow(sum_ab, -2.0) + std::pow(sum_ab, -3.0) - 2 * std::pow(sum_ab, -4.0);
            const double t2 = -50 / a - 10 * std::pow(a, -2.0) - std::pow(a, -3.0) + std::pow(a, -4.0);
            const double t3 = -50 / b - 10 * std::pow(b, -2.0) - std::pow(b, -3.0) + std::pow(b, -4.0);
            return log_term + (t1 + t2 + t3) / 120;
        };
        auto asymptotic_b_large = [](double a, double b) {
            const double sum_ab = a + b;
            const double t1 = sc::gammaln(a) - (a - 1) * sc::psi(a);
            const double t2 = -1 / (2 * b) + 1 / (12 * b) - std::pow(b, -2.0) / 12 - std::pow(b, -3.0) / 120 + std::pow(b, -4.0) / 120 +
                              std::pow(b, -5.0) / 252 - std::pow(b, -6.0) / 252 + 1 / sum_ab - 1 / (12 * sum_ab) +
                              std::pow(sum_ab, -2.0) / 6 + std::pow(sum_ab, -3.0) / 120 - std::pow(sum_ab, -4.0) / 60 -
                              std::pow(sum_ab, -5.0) / 252 + std::pow(sum_ab, -6.0) / 126;
            const double log_term = sum_ab * std::log1p(a / b) + std::log(b) - 2 * std::log(sum_ab);
            return t1 + t2 + log_term;
        };
        auto threshold_large = [](double v) {
            const double j = std::floor(std::log10(v));
            const double d = std::floor(v / std::pow(10.0, j)) + 2;
            return v != 1.0 ? d * std::pow(10.0, 7 + j) : 1000.0;
        };
        const double ta = threshold_large(a), tb = threshold_large(b);
        /* _lazyselect: every matching choice is placed in turn, so the last matching one wins; default 0 */
        double r = 0.0;
        if (a >= 4.96e6 && b >= 4.96e6) r = asymptotic_ab_large(a, b);
        if (a <= 4.9e6 && b - a >= 1e6 && b >= ta) r = asymptotic_b_large(a, b);
        if (b <= 4.9e6 && a - b >= 1e6 && a >= tb) r = asymptotic_b_large(b, a);
        if (a < 4.9e6 && b < 4.9e6) r = regular(a, b);
        return r;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_beta(bg, sa[0][i], sa[1][i]);
    }
};
Registrar r_beta(new Beta);

const double NORM_ENTROPY = 0.5 * (std::log(2 * M_PI) + 1);

struct T : Cont {
    T() { name = "t"; shapes = "df"; nshape = 1; doc = "A Student's t continuous random variable."; }
    static double t_logpdf(double x, double df)
    {
        return std::log(sc::poch(0.5 * df, 0.5)) - 0.5 * (std::log(df) + std::log(M_PI)) - (df + 1) / 2 * std::log1p(x * x / df);
    }
    double pdf(double x, const double *s) const override
    {
        const double df = s[0];
        return df == INF ? std::exp(-(x * x) / 2.0) / NORM_PDF_C : std::exp(logpdf(x, s));
    }
    double logpdf(double x, const double *s) const override
    {
        const double df = s[0];
        return df == INF ? -(x * x) / 2.0 - NORM_PDF_LOGC : t_logpdf(x, df);
    }
    double cdf(double x, const double *s) const override { return sc::stdtr(s[0], x); }
    double sf(double x, const double *s) const override { return sc::stdtr(s[0], -x); }
    double ppf(double q, const double *s) const override { return sc::stdtrit(s[0], q); }
    double isf(double q, const double *s) const override { return -sc::stdtrit(s[0], q); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double df = s[0];
        const bool infinite_df = df == INF;
        const double mu = df > 1 ? 0.0 : INF;
        double mu2 = NaN;                          /* _lazyselect: the last matching condition wins */
        if (df > 1 && df <= 2) mu2 = INF;
        if (df > 2 && std::isfinite(df)) mu2 = df / (df - 2.0);
        if (infinite_df) mu2 = 1;
        const double g1 = df > 3 ? 0.0 : NaN;
        double g2 = NaN;
        if (df > 2 && df <= 4) g2 = INF;
        if (df > 4 && std::isfinite(df)) g2 = 6.0 / (df - 4.0);
        if (infinite_df) g2 = 0;
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *s) const override
    {
        const double df = s[0];
        if (df == INF) return NORM_ENTROPY;
        if (df >= 100)
            return NORM_ENTROPY + 1 / df + std::pow(df, -2.) / 4 - std::pow(df, -3.) / 6 - std::pow(df, -4.) / 8 +
                   3.0 / 10 * std::pow(df, -5.) + std::pow(df, -6.) / 4;
        const double half = df / 2, half1 = (df + 1) / 2;
        return half1 * (sc::psi(half1) - sc::psi(half)) + std::log(std::sqrt(df) * sc::beta(half, 0.5));
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_standard_t(bg, sa[0][i]);
    }
};
Registrar r_t(new T);

struct Chi2 : Cont {
    Chi2() { name = "chi2"; shapes = "df"; nshape = 1; a = 0.0; doc = "A chi-squared continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double df = s[0];
        return sc::xlogy(df / 2. - 1, x) - x / 2. - sc::gammaln(df / 2.) - (std::log(2) * df) / 2.;
    }
    double cdf(double x, const double *s) const override { return sc::chdtr(s[0], x); }
    double sf(double x, const double *s) const override { return sc::chdtrc(s[0], x); }
    double isf(double p, const double *s) const override { return sc::chdtri(s[0], p); }
    double ppf(double p, const double *s) const override { return 2 * sc::gammaincinv(s[0] / 2, p); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double df = s[0];
        st.set(df, 2 * df, 2 * std::sqrt(2.0 / df), 12.0 / df);
    }
    double entropy(const double *s) const override
    {
        const double half_df = 0.5 * s[0];
        if (half_df < 125) return half_df + std::log(2) + sc::gammaln(half_df) + (1 - half_df) * sc::psi(half_df);
        const double c = std::log(2) + 0.5 * (1 + std::log(2 * M_PI));
        const double h = 0.5 / half_df;
        return h * (-2.0 / 3 + h * (-1.0 / 3 + h * (-4.0 / 45 + h / 7.5))) + 0.5 * std::log(half_df) + c;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_chisquare(bg, sa[0][i]);
    }
};
Registrar r_chi2(new Chi2);

struct F : Cont {
    F() { name = "f"; shapes = "dfn, dfd"; nshape = 2; a = 0.0; doc = "An F continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double n = 1.0 * s[0], m = 1.0 * s[1];
        return m / 2 * std::log(m) + n / 2 * std::log(n) + sc::xlogy(n / 2 - 1, x) -
               ((n + m) / 2 * std::log(m + n * x) + sc::betaln(n / 2, m / 2));
    }
    double cdf(double x, const double *s) const override { return sc::fdtr(s[0], s[1], x); }
    double sf(double x, const double *s) const override { return sc::fdtrc(s[0], s[1], x); }
    double ppf(double q, const double *s) const override { return sc::fdtri(s[0], s[1], q); }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double v1 = 1. * s[0], v2 = 1. * s[1];
        const double v2_2 = v2 - 2., v2_4 = v2 - 4., v2_6 = v2 - 6., v2_8 = v2 - 8.;
        const double mu = v2 > 2 ? v2 / v2_2 : INF;
        const double mu2 = v2 > 4 ? 2 * v2 * v2 * (v1 + v2_2) / (v1 * (v2_2 * v2_2) * v2_4) : INF;
        double g1 = v2 > 6 ? (2 * v1 + v2_2) / v2_6 * std::sqrt(v2_4 / (v1 * (v1 + v2_2))) : NaN;
        g1 *= std::sqrt(8.);
        double g2 = v2 > 8 ? (8 + g1 * g1 * v2_6) / v2_8 : NaN;
        g2 *= 3. / 2.;
        st.set(mu, mu2, g1, g2);
    }
    double entropy(const double *s) const override
    {
        const double dfn = s[0], dfd = s[1];
        const double half_dfn = 0.5 * dfn, half_dfd = 0.5 * dfd, half_sum = 0.5 * (dfn + dfd);
        return std::log(dfd) - std::log(dfn) + sc::betaln(half_dfn, half_dfd) + (1 - half_dfn) * sc::psi(half_dfn) -
               (1 + half_dfd) * sc::psi(half_dfd) + half_sum * sc::psi(half_sum);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_f(bg, sa[0][i], sa[1][i]);
    }
};
Registrar r_f(new F);

/* ---------------------------------------------------------------- discrete */

inline bool isintegral(double x) { return x == std::nearbyint(x); }   /* x == np.round(x) (half to even) */

struct Binom : Disc {
    Binom() { name = "binom"; shapes = "n, p"; nshape = 2; doc = "A binomial discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0 && isintegral(s[0]) && s[1] >= 0 && s[1] <= 1; }
    void get_support(const double *s, double &lo, double &hi) const override { lo = a; hi = s[0]; }
    double logpmf(double x, const double *s) const override
    {
        const double n = s[0], p = s[1], k = std::floor(x);
        const double combiln = sc::gammaln(n + 1) - (sc::gammaln(k + 1) + sc::gammaln(n - k + 1));
        return combiln + sc::xlogy(k, p) + sc::xlog1py(n - k, -p);
    }
    double pmf(double x, const double *s) const override { return sc::_binom_pmf(x, s[0], s[1]); }
    double cdf(double x, const double *s) const override { return sc::_binom_cdf(std::floor(x), s[0], s[1]); }
    double sf(double x, const double *s) const override { return sc::_binom_sf(std::floor(x), s[0], s[1]); }
    double isf(double x, const double *s) const override { return sc::_binom_isf(x, s[0], s[1]); }
    double ppf(double q, const double *s) const override { return sc::_binom_ppf(q, s[0], s[1]); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double n = s[0], p = s[1];
        const double mu = n * p;
        st.set(0, mu);
        st.set(1, mu - n * (p * p));
        if (moments & MOM_S) {
            const double pq = p - p * p;
            const double npq_sqrt = std::sqrt(n * pq);
            st.set(2, 1.0 / npq_sqrt - (2.0 * p) / npq_sqrt);
        }
        if (moments & MOM_K) {
            const double pq = p - p * p;
            const double npq = n * pq;
            st.set(3, 1.0 / npq - 6.0 / n);
        }
    }
    double entropy(const double *s) const override
    {
        const int64_t m = alloc_count(std::floor(s[0]) + 1);   /* np.r_[0:n + 1] */
        std::vector<double> v((size_t)m);
        for (int64_t k = 0; k < m; k++) v[(size_t)k] = entr(pmf((double)k, s));
        return tsr_psum(v.data(), m);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) {
            binomial_t bin = {};
            out[i] = (double)random_binomial(bg, sa[1][i], np_int64(sa[0][i]), &bin);
        }
    }
};
Registrar r_binom(new Binom);

struct Poisson : Disc {
    Poisson() { name = "poisson"; shapes = "mu"; nshape = 1; doc = "A Poisson discrete random variable."; }
    bool argcheck(const double *s) const override { return s[0] >= 0; }
    double logpmf(double k, const double *s) const override { return sc::xlogy(k, s[0]) - sc::gammaln(k + 1) - s[0]; }
    double pmf(double k, const double *s) const override { return std::exp(logpmf(k, s)); }
    double cdf(double x, const double *s) const override { return sc::pdtr(std::floor(x), s[0]); }
    double sf(double x, const double *s) const override { return sc::pdtrc(std::floor(x), s[0]); }
    double ppf(double q, const double *s) const override
    {
        const double mu = s[0];
        const double vals = std::ceil(sc::pdtrik(q, mu));
        const double vals1 = std::max(vals - 1, 0.0);
        const double temp = sc::pdtr(vals1, mu);
        return temp >= q ? vals1 : vals;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double mu = s[0];
        st.set(mu, mu, mu > 0 ? std::sqrt(1.0 / mu) : INF, mu > 0 ? 1.0 / mu : INF);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = (double)random_poisson(bg, sa[0][i]);
    }
};
Registrar r_poisson(new Poisson);

}  // namespace
}  // namespace tsd
