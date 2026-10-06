/*
 * scipy.stats: gumbel_l, gumbel_r, halfcauchy, halfgennorm, halflogistic, halfnorm, hypsecant, invgamma, invgauss,
 * invweibull, irwinhall, jf_skew_t, johnsonsb, johnsonsu, kappa3, kappa4, ksone, kstwo, kstwobign, landau
 * (scipy/stats/_continuous_distns.py, SciPy 1.17.1), private method by private method. kstwo ports
 * scipy/stats/_ksstats.py; irwinhall ports the B-spline evaluation of scipy.interpolate.BSpline
 * (basis_element, antiderivative / splantider, _dierckx.evaluate_spline). See stats_dist.hpp.
 */
#include "stats_dist.hpp"
#include "u128.hpp"

#include <algorithm>
#include <vector>

namespace tsd {
namespace {

/* scipy.stats._constants */
const double EULER = 0.577215664901532860606512090082402431042;
const double ZETA3 = 1.202056903159594285399738161511449990765;

/* _norm_pdf_C and the _norm_* helpers of _continuous_distns.py */
const double NORM_PDF_C = std::sqrt(2 * M_PI);
inline double norm_pdf(double x) { return std::exp(-(x * x) / 2.0) / NORM_PDF_C; }
inline double norm_cdf(double x) { return sc::ndtr(x); }
inline double norm_sf(double x) { return sc::ndtr(-x); }
inline double norm_ppf(double q) { return sc::ndtri(q); }
inline double norm_isf(double q) { return -sc::ndtri(q); }
inline double norm_logcdf(double x) { return sc::log_ndtr(x); }
inline double norm_logsf(double x) { return sc::log_ndtr(-x); }

/* libm pow for Python float ** float and NumPy scalar ** scalar: called through a pointer so that GCC does not
   fold pow(x, 2.0) into x * x (glibc's pow is not always the correctly rounded square) */
double (*volatile libm_pow)(double, double) = ::pow;
inline double lpow(double x, double y) { return libm_pow(x, y); }

/* ---------------------------------------------------------------- gumbel_l, gumbel_r */

struct GumbelL : Cont {
    GumbelL() { name = "gumbel_l"; doc = "A left-skewed Gumbel continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *) const override { return x - std::exp(x); }
    double cdf(double x, const double *) const override { return -sc::expm1(-std::exp(x)); }
    double ppf(double q, const double *) const override { return std::log(-sc::log1p(-q)); }
    double logsf(double x, const double *) const override { return -std::exp(x); }
    double sf(double x, const double *) const override { return std::exp(-std::exp(x)); }
    double isf(double x, const double *) const override { return std::log(-std::log(x)); }
    void stats(const double *, int, Stats4 &st) const override
    {
        st.set(-EULER, M_PI * M_PI / 6.0, -12 * std::sqrt(6.0) / lpow(M_PI, 3.0) * ZETA3, 12.0 / 5);
    }
    double entropy(const double *) const override { return EULER + 1.; }
};
Registrar r_gumbel_l(new GumbelL);

struct GumbelR : Cont {
    GumbelR() { name = "gumbel_r"; doc = "A right-skewed Gumbel continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *) const override { return -x - std::exp(-x); }
    double cdf(double x, const double *) const override { return std::exp(-std::exp(-x)); }
    double logcdf(double x, const double *) const override { return -std::exp(-x); }
    double ppf(double q, const double *) const override { return -std::log(-std::log(q)); }
    double sf(double x, const double *) const override { return -sc::expm1(-std::exp(-x)); }
    double isf(double p, const double *) const override { return -std::log(-std::log1p(-p)); }
    void stats(const double *, int, Stats4 &st) const override
    {
        st.set(EULER, M_PI * M_PI / 6.0, 12 * std::sqrt(6.0) / lpow(M_PI, 3.0) * ZETA3, 12.0 / 5);
    }
    double entropy(const double *) const override { return EULER + 1.; }
};
Registrar r_gumbel_r(new GumbelR);

/* ---------------------------------------------------------------- half distributions, hypsecant */

struct HalfCauchy : Cont {
    HalfCauchy() { name = "halfcauchy"; a = 0.0; doc = "A Half-Cauchy continuous random variable."; }
    double pdf(double x, const double *) const override { return 2.0 / M_PI / (1.0 + x * x); }
    double logpdf(double x, const double *) const override { return std::log(2.0 / M_PI) - sc::log1p(x * x); }
    double cdf(double x, const double *) const override { return 2.0 / M_PI * std::atan(x); }
    double ppf(double q, const double *) const override { return std::tan(M_PI / 2 * q); }
    double sf(double x, const double *) const override { return 2.0 / M_PI * std::atan2(1.0, x); }
    double isf(double p, const double *) const override { return 1.0 / std::tan(M_PI * p / 2); }
    void stats(const double *, int, Stats4 &st) const override { st.set(INF, INF, NaN, NaN); }
    double entropy(const double *) const override { return std::log(2 * M_PI); }
};
Registrar r_halfcauchy(new HalfCauchy);

struct HalfGennorm : Cont {
    HalfGennorm()
    {
        name = "halfgennorm"; shapes = "beta"; nshape = 1; a = 0.0;
        doc = "The upper half of a generalized normal continuous random variable.";
    }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double beta = s[0];
        return std::log(beta) - sc::gammaln(1.0 / beta) - std::pow(x, beta);
    }
    double cdf(double x, const double *s) const override { return sc::gammainc(1.0 / s[0], std::pow(x, s[0])); }
    double ppf(double x, const double *s) const override { return std::pow(sc::gammaincinv(1.0 / s[0], x), 1.0 / s[0]); }
    double sf(double x, const double *s) const override { return sc::gammaincc(1.0 / s[0], std::pow(x, s[0])); }
    double isf(double x, const double *s) const override { return std::pow(sc::gammainccinv(1.0 / s[0], x), 1.0 / s[0]); }
    double entropy(const double *s) const override
    {
        const double beta = s[0];
        return 1.0 / beta - std::log(beta) + sc::gammaln(1.0 / beta);
    }
};
Registrar r_halfgennorm(new HalfGennorm);

struct HalfLogistic : Cont {
    HalfLogistic() { name = "halflogistic"; a = 0.0; doc = "A half-logistic continuous random variable."; }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *) const override { return std::log(2.0) - x - 2. * sc::log1p(std::exp(-x)); }
    double cdf(double x, const double *) const override { return std::tanh(x / 2.0); }
    double ppf(double q, const double *) const override { return 2 * std::atanh(q); }
    double sf(double x, const double *) const override { return 2 * sc::expit(-x); }
    double isf(double q, const double *) const override
    {
        return q < 0.5 ? -sc::logit(0.5 * q) : 2 * std::atanh(1 - q);
    }
    double munp(int n, const double *) const override
    {
        if (n == 0) return 1;
        if (n == 1) return 2 * std::log(2.0);
        if (n == 2) return M_PI * M_PI / 3.0;
        if (n == 3) return 9 * ZETA3;
        if (n == 4) return 7 * lpow(M_PI, 4.0) / 15.0;
        return 2 * (1 - std::pow(2.0, (double)(1 - n))) * sc::gamma((double)n + 1) * sc::_zeta((double)n, 1.0);
    }
    double entropy(const double *) const override { return 2 - std::log(2.0); }
};
Registrar r_halflogistic(new HalfLogistic);

struct HalfNorm : Cont {
    HalfNorm() { name = "halfnorm"; a = 0.0; doc = "A half-normal continuous random variable."; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = std::fabs(random_standard_normal(bg));
    }
    double pdf(double x, const double *) const override { return std::sqrt(2.0 / M_PI) * std::exp(-x * x / 2.0); }
    double logpdf(double x, const double *) const override { return 0.5 * std::log(2.0 / M_PI) - x * x / 2.0; }
    double cdf(double x, const double *) const override { return sc::erf(x / std::sqrt(2.0)); }
    double ppf(double q, const double *) const override { return norm_ppf((1 + q) / 2.0); }
    double sf(double x, const double *) const override { return 2 * norm_sf(x); }
    double isf(double p, const double *) const override { return norm_isf(p / 2); }
    void stats(const double *, int, Stats4 &st) const override
    {
        st.set(std::sqrt(2.0 / M_PI), 1 - 2.0 / M_PI, std::sqrt(2.0) * (4 - M_PI) / lpow(M_PI - 2, 1.5),
               8 * (M_PI - 3) / lpow(M_PI - 2, 2.0));
    }
    double entropy(const double *) const override { return 0.5 * std::log(M_PI / 2.0) + 0.5; }
};
Registrar r_halfnorm(new HalfNorm);

struct Hypsecant : Cont {
    Hypsecant() { name = "hypsecant"; doc = "A hyperbolic secant continuous random variable."; }
    double pdf(double x, const double *) const override { return 1.0 / (M_PI * std::cosh(x)); }
    double cdf(double x, const double *) const override { return 2.0 / M_PI * std::atan(std::exp(x)); }
    double ppf(double q, const double *) const override { return std::log(std::tan(M_PI * q / 2.0)); }
    double sf(double x, const double *) const override { return 2.0 / M_PI * std::atan(std::exp(-x)); }
    double isf(double q, const double *) const override { return -std::log(std::tan(M_PI * q / 2.0)); }
    void stats(const double *, int, Stats4 &st) const override { st.set(0, M_PI * M_PI / 4, 0, 2); }
    double entropy(const double *) const override { return std::log(2 * M_PI); }
};
Registrar r_hypsecant(new Hypsecant);

/* ---------------------------------------------------------------- invgamma, invgauss, invweibull */

/* the distributions with _support_mask = _open_support_mask: pdf and logpdf are masked at the end points */
inline bool open_in(double x, double lo, double hi) { return lo < x && x < hi; }

struct InvGamma : Cont {
    InvGamma() { name = "invgamma"; shapes = "a"; nshape = 1; a = 0.0; doc = "An inverted gamma continuous random variable."; }
    double pdf(double x, const double *s) const override { return open_in(x, a, b) ? std::exp(logpdf(x, s)) : 0.0; }
    double logpdf(double x, const double *s) const override
    {
        if (!open_in(x, a, b)) return -INF;
        const double a_ = s[0];
        return -(a_ + 1) * std::log(x) - sc::gammaln(a_) - 1.0 / x;
    }
    double cdf(double x, const double *s) const override { return sc::gammaincc(s[0], 1.0 / x); }
    double ppf(double q, const double *s) const override { return 1.0 / sc::gammainccinv(s[0], q); }
    double sf(double x, const double *s) const override { return sc::gammainc(s[0], 1.0 / x); }
    double isf(double q, const double *s) const override { return 1.0 / sc::gammaincinv(s[0], q); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double x = s[0];
        st.set(0, x > 1 ? 1. / (x - 1.) : INF);
        st.set(1, x > 2 ? 1. / sq(x - 1.) / (x - 2.) : INF);
        if (moments & MOM_S) st.set(2, x > 3 ? 4. * std::sqrt(x - 2.) / (x - 3.) : NaN);
        if (moments & MOM_K) st.set(3, x > 4 ? 6. * (5. * x - 11.) / (x - 3.) / (x - 4.) : NaN);
    }
    double entropy(const double *s) const override
    {
        const double a_ = s[0];
        if (a_ >= 200)
            return (1 - 3 * std::log(a_) + std::log(2.0) + std::log(M_PI)) / 2 + 2.0 / 3 * (1 / a_) + std::pow(a_, -2.) / 12 -
                   std::pow(a_, -3.) / 90 - std::pow(a_, -4.) / 120;
        return a_ - (a_ + 1.0) * sc::psi(a_) + sc::gammaln(a_);
    }
};
Registrar r_invgamma(new InvGamma);

struct InvGauss : Cont {
    InvGauss() { name = "invgauss"; shapes = "mu"; nshape = 1; a = 0.0; doc = "An inverse Gaussian continuous random variable."; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_wald(bg, sa[0][i], 1.0);
    }
    double pdf(double x, const double *s) const override
    {
        if (!open_in(x, a, b)) return 0.0;
        const double mu = s[0];
        return 1.0 / std::sqrt(2 * M_PI * std::pow(x, 3.0)) * std::exp(-1.0 / (2 * x) * sq(x / mu - 1));
    }
    double logpdf(double x, const double *s) const override
    {
        if (!open_in(x, a, b)) return -INF;
        const double mu = s[0];
        return -0.5 * std::log(2 * M_PI) - 1.5 * std::log(x) - sq(x / mu - 1) / (2 * x);
    }
    double logcdf(double x, const double *s) const override
    {
        const double mu = s[0];
        const double fac = 1 / std::sqrt(x);
        const double a_ = norm_logcdf(fac * (x / mu - 1));
        const double b_ = 2 / mu + norm_logcdf(-fac * (x / mu + 1));
        return a_ + std::log1p(std::exp(b_ - a_));
    }
    double logsf(double x, const double *s) const override
    {
        const double mu = s[0];
        const double fac = 1 / std::sqrt(x);
        const double a_ = norm_logsf(fac * (x / mu - 1));
        const double b_ = 2 / mu + norm_logcdf(-fac * (x / mu + 1));
        return a_ + std::log1p(-std::exp(b_ - a_));
    }
    double sf(double x, const double *s) const override { return std::exp(logsf(x, s)); }
    double cdf(double x, const double *s) const override { return std::exp(logcdf(x, s)); }
    double ppf(double x, const double *s) const override
    {
        const double mu = s[0];
        double r = sc::_invgauss_ppf(x, mu, 1);
        if (x > 0.5) r = sc::_invgauss_isf(1 - x, mu, 1);   /* "wrong tail" */
        if (std::isnan(r)) r = Cont::ppf(x, s);                /* super()._ppf */
        return r;
    }
    double isf(double x, const double *s) const override
    {
        const double mu = s[0];
        double r = sc::_invgauss_isf(x, mu, 1);
        if (x > 0.5) r = sc::_invgauss_ppf(1 - x, mu, 1);
        if (std::isnan(r)) r = ppf(1.0 - x, s);               /* super()._isf: self._ppf(1 - q) */
        return r;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double mu = s[0];
        st.set(mu, std::pow(mu, 3.0), 3 * std::sqrt(mu), 15 * mu);
    }
    double entropy(const double *s) const override
    {
        const double mu = s[0];
        const double a_ = 1. + std::log(2 * M_PI) + 3 * std::log(mu);
        const double r = 2 / mu;
        const double b_ = sc::_scaled_exp1(r) / r;
        return 0.5 * a_ - 1.5 * b_;
    }
};
Registrar r_invgauss(new InvGauss);

struct InvWeibull : Cont {
    InvWeibull() { name = "invweibull"; shapes = "c"; nshape = 1; a = 0.0; doc = "An inverted Weibull continuous random variable."; }
    double pdf(double x, const double *s) const override
    {
        if (!open_in(x, a, b)) return 0.0;
        const double c = s[0];
        const double xc1 = std::pow(x, -c - 1.0);
        const double xc2 = std::exp(-std::pow(x, -c));
        return c * xc1 * xc2;
    }
    double logpdf(double x, const double *s) const override { return open_in(x, a, b) ? std::log(pdf(x, s)) : -INF; }
    double cdf(double x, const double *s) const override { return std::exp(-std::pow(x, -s[0])); }
    double sf(double x, const double *s) const override { return -std::expm1(-std::pow(x, -s[0])); }
    double ppf(double q, const double *s) const override { return std::pow(-std::log(q), -1.0 / s[0]); }
    double isf(double p, const double *s) const override { return std::pow(-std::log1p(-p), -1 / s[0]); }
    double munp(int n, const double *s) const override { return sc::gamma(1 - n / s[0]); }
    double entropy(const double *s) const override { return 1 + EULER + EULER / s[0] - std::log(s[0]); }
};
Registrar r_invweibull(new InvWeibull);

/* ---------------------------------------------------------------- irwinhall */

/* _dierckx.evaluate_spline for one point (nu = 0, extrapolate = True): _find_interval, _deBoor_D and the
   linear combination with the coefficients */
double bspline_eval(const std::vector<double> &t, const std::vector<double> &c, int k, double x)
{
    const int len_t = (int)t.size();
    const int n = len_t - k - 1;
    if (x != x) return NaN;
    int l = k;                                   /* the previous interval starts at k */
    while (x < t[l] && l != k) l -= 1;
    l += 1;
    while (x >= t[l] && l != n) l += 1;
    const int ell = l - 1;
    std::vector<double> work(2 * k + 2);
    double *h = work.data(), *hh = work.data() + k + 1;
    h[0] = 1.0;
    for (int j = 1; j <= k; j++) {
        for (int i = 0; i < j; i++) hh[i] = h[i];
        h[0] = 0.0;
        for (int m = 1; m <= j; m++) {
            const int ind = ell + m;
            const double xb = t[ind], xa = t[ind - j];
            if (xb == xa) {
                h[m] = 0.0;
                continue;
            }
            const double w = hh[m - 1] / (xb - xa);
            h[m - 1] += w * (xb - x);
            h[m] = w * (x - xa);
        }
    }
    double out = 0.;
    for (int a = 0; a < k + 1; a++) out = out + c[ell + a - k] * work[a];
    return out;
}

/* BSpline.basis_element(arange(n + 1)): knots and coefficients of the cardinal B-spline of degree n - 1 */
void cardbspl(double nf, std::vector<double> &t, std::vector<double> &c, int &k)
{
    const int64_t n1_ = alloc_count(std::ceil(nf + 1));   /* np.arange(n + 1) */
    if (n1_ > 100000000) throw std::bad_alloc();       /* the spline's (n + 1)^2 work would not fit either */
    const int n1 = (int)n1_;
    std::vector<double> ti((size_t)n1);
    for (int i = 0; i < n1; i++) ti[(size_t)i] = (double)i;
    k = n1 - 2;
    t.clear();
    for (int i = 0; i < k; i++) t.push_back(ti[0] - 1);
    for (double v : ti) t.push_back(v);
    for (int i = 0; i < k; i++) t.push_back(ti[(size_t)n1 - 1] + 1);
    c.assign(t.size(), 0.0);
    c[(size_t)k] = 1.;
}

/* BSpline.antiderivative(): _fitpack_impl.splantider with n = 1 */
void antiderivative(std::vector<double> &t, std::vector<double> &c, int &k)
{
    const size_t m = t.size() - (size_t)k - 1;
    std::vector<double> cs(m);
    double acc = 0.0;
    for (size_t i = 0; i < m; i++) {
        const double dt = t[i + (size_t)k + 1] - t[i];
        acc = i == 0 ? c[i] * dt : acc + c[i] * dt;   /* cumulative_sum */
        cs[i] = acc;
    }
    std::vector<double> nc;
    nc.push_back(0.0);
    for (size_t i = 0; i < m; i++) nc.push_back(cs[i] / (k + 1));
    const double last = nc.back();
    for (int i = 0; i < k + 2; i++) nc.push_back(last);
    std::vector<double> nt;
    nt.push_back(t.front());
    for (double v : t) nt.push_back(v);
    nt.push_back(t.back());
    t.swap(nt);
    c.swap(nc);
    k += 1;
}

/* x / y for exact unsigned integers, correctly rounded (Python's int true division) */
double exact_ratio(tsr128::u128 num, tsr128::u128 den)
{
    if (num == 0) return 0.0;
    using tsr128::u128;
    u128 q = num / den, r = num % den;
    int e = 0;
    bool sticky = false;
    const u128 lo = u128(1) << 54, hi = u128(1) << 55;
    while (q >= hi) {
        sticky = sticky || (q & 1);
        q >>= 1;
        e++;
    }
    while (q < lo) {
        const bool top = static_cast<bool>(r >> 127);
        r <<= 1;
        q <<= 1;
        if (top || r >= den) {
            r -= den;
            q |= 1;
        }
        e--;
    }
    sticky = sticky || r != 0;
    uint64_t mant = (uint64_t)(q >> 2);
    const unsigned low2 = (unsigned)(q & 3);
    if (low2 == 3 || (low2 == 2 && (sticky || (mant & 1)))) mant++;
    return std::ldexp((double)mant, e + 2);
}

struct IrwinHall : Cont {
    IrwinHall() { name = "irwinhall"; shapes = "n"; nshape = 1; doc = "An Irwin-Hall (Uniform Sum) continuous random variable."; }
    bool argcheck(const double *s) const override { return s[0] > 0 && s[0] == std::nearbyint(s[0]); }
    void get_support(const double *s, double &lo, double &hi) const override { lo = 0; hi = s[0]; }
    double munp(int order, const double *s) const override
    {
        /* stirling2(n + order, n, exact=True) / comb(n + order, n, exact=True), exact integers */
        const int64_t n = (int64_t)s[0];
        if (order < 0) return NaN;
        using tsr128::u128;
        std::vector<u128> prev((size_t)order + 1, 0), cur((size_t)order + 1, 0);
        bool ovf = false;
        prev[0] = 1;                                  /* S(0 + d, 0): 1 for d = 0 */
        for (int64_t j = 1; j <= n; j++) {
            for (int d = 0; d <= order; d++) {
                if (d == 0) { cur[0] = 1; continue; }
                u128 t;
                if (tsr128::mul_overflow(u128((uint64_t)j), cur[(size_t)d - 1], &t)) ovf = true;
                if (tsr128::add_overflow(t, prev[(size_t)d], &t)) ovf = true;
                cur[(size_t)d] = t;
            }
            prev.swap(cur);
        }
        u128 comb = 1;
        for (int i = 1; i <= order; i++) {
            u128 t;
            if (tsr128::mul_overflow(comb, u128((uint64_t)(n + i)), &t)) ovf = true;
            comb = t / u128((uint64_t)i);
        }
        if (!ovf && !(prev[(size_t)order] >> 126) && !(comb >> 126)) return exact_ratio(prev[(size_t)order], comb);
        /* beyond 126 bits: long double recurrence (not exact) */
        std::vector<long double> lp((size_t)order + 1, 0), lc((size_t)order + 1, 0);
        lp[0] = 1;
        for (int64_t j = 1; j <= n; j++) {
            lc[0] = 1;
            for (int d = 1; d <= order; d++) lc[(size_t)d] = (long double)j * lc[(size_t)d - 1] + lp[(size_t)d];
            lp.swap(lc);
        }
        long double cb = 1;
        for (int i = 1; i <= order; i++) cb = cb * (long double)(n + i) / i;
        return (double)(lp[(size_t)order] / cb);
    }
    double pdf(double x, const double *s) const override
    {
        std::vector<double> t, c;
        int k;
        cardbspl(s[0], t, c, k);
        return bspline_eval(t, c, k, x);
    }
    double cdf(double x, const double *s) const override
    {
        std::vector<double> t, c;
        int k;
        cardbspl(s[0], t, c, k);
        antiderivative(t, c, k);
        return bspline_eval(t, c, k, x);
    }
    double sf(double x, const double *s) const override
    {
        std::vector<double> t, c;
        int k;
        cardbspl(s[0], t, c, k);
        antiderivative(t, c, k);
        return bspline_eval(t, c, k, s[0] - x);
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        bool scalar = true;
        for (int64_t i = 1; i < n; i++)
            if (!(sa[0][i] == sa[0][0])) scalar = false;
        if (scalar && n > 0) {
            /* uniform(size=(m, *size)).sum(axis=0): m rows of n draws, summed row by row */
            const int64_t m = alloc_count(std::floor(sa[0][0]));
            std::vector<double> u((size_t)alloc_count((double)m * (double)n));
            for (auto &v : u) v = random_standard_uniform(bg);
            for (int64_t j = 0; j < n; j++) {
                double acc = m > 0 ? u[(size_t)j] : 0.0;
                for (int64_t r = 1; r < m; r++) acc += u[(size_t)(r * n + j)];
                out[j] = acc;
            }
            return;
        }
        /* _vectorize_rvs_over_shapes: one call per element, uniform(size=(m,)).sum(axis=0) */
        for (int64_t i = 0; i < n; i++) {
            const int64_t m = alloc_count(std::floor(sa[0][i]));
            std::vector<double> u((size_t)m);
            for (auto &v : u) v = random_standard_uniform(bg);
            out[i] = tsr_psum(u.data(), m);
        }
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double n = s[0];
        st.set(n / 2, n / 12, 0, -6 / (5 * n));
    }
};
Registrar r_irwinhall(new IrwinHall);

/* ---------------------------------------------------------------- jf_skew_t */

struct JFSkewT : Cont {
    JFSkewT() { name = "jf_skew_t"; shapes = "a, b"; nshape = 2; doc = "Jones and Faddy skew-t distribution."; }
    double pdf(double x, const double *s) const override
    {
        const double a_ = s[0], b_ = s[1];
        const double c = std::pow(2.0, a_ + b_ - 1) * sc::beta(a_, b_) * std::sqrt(a_ + b_);
        const double d1 = std::pow(1 + x / std::sqrt(a_ + b_ + x * x), a_ + 0.5);
        const double d2 = std::pow(1 - x / std::sqrt(a_ + b_ + x * x), b_ + 0.5);
        return d1 * d2 / c;
    }
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override
    {
        for (int64_t i = 0; i < n; i++) out[i] = random_beta(bg, sa[0][i], sa[1][i]);
        for (int64_t i = 0; i < n; i++) {
            const double d1 = out[i], a_ = sa[0][i], b_ = sa[1][i];
            const double d2 = (2 * d1 - 1) * std::sqrt(a_ + b_);
            const double d3 = 2 * std::sqrt(d1 * (1 - d1));
            out[i] = d2 / d3;
        }
    }
    double cdf(double x, const double *s) const override
    {
        const double y = (1 + x / std::sqrt(s[0] + s[1] + x * x)) * 0.5;
        return sc::betainc(s[0], s[1], y);
    }
    double sf(double x, const double *s) const override
    {
        const double y = (1 + x / std::sqrt(s[0] + s[1] + x * x)) * 0.5;
        return sc::betaincc(s[0], s[1], y);
    }
    double ppf(double q, const double *s) const override
    {
        const double d1 = sc::_beta_ppf(q, s[0], s[1]);    /* beta.ppf(q, a, b), 0 < q < 1 */
        const double d2 = (2 * d1 - 1) * std::sqrt(s[0] + s[1]);
        const double d3 = 2 * std::sqrt(d1 * (1 - d1));
        return d2 / d3;
    }
    double munp(int n, const double *s) const override
    {
        const double a_ = s[0], b_ = s[1];
        if (!(a_ > 0.5 * n && b_ > 0.5 * n && n >= 0)) return NaN;
        const double num = std::pow(a_ + b_, 0.5 * n);
        const double denom = std::ldexp(1.0, n) * sc::beta(a_, b_);
        std::vector<double> terms((size_t)n + 1);
        for (int i = 0; i <= n; i++) {
            const double sgn = i % 2 > 0 ? -1 : 1;
            const double d = sc::beta(a_ + 0.5 * n - i, b_ - 0.5 * n + i);
            terms[(size_t)i] = sc::binom((double)n, (double)i) * sgn * d;
        }
        return num / denom * tsr_psum(terms.data(), n + 1);
    }
};
Registrar r_jf_skew_t(new JFSkewT);

/* ---------------------------------------------------------------- johnsonsb, johnsonsu */

struct JohnsonSB : Cont {
    JohnsonSB() { name = "johnsonsb"; shapes = "a, b"; nshape = 2; a = 0.0; b = 1.0; doc = "A Johnson SB continuous random variable."; }
    bool argcheck(const double *s) const override { return s[1] > 0 && s[0] == s[0]; }
    double pdf(double x, const double *s) const override
    {
        if (!open_in(x, a, b)) return 0.0;
        const double trm = norm_pdf(s[0] + s[1] * sc::logit(x));
        return s[1] * 1.0 / (x * (1 - x)) * trm;
    }
    double logpdf(double x, const double *s) const override { return open_in(x, a, b) ? std::log(pdf(x, s)) : -INF; }
    double cdf(double x, const double *s) const override { return norm_cdf(s[0] + s[1] * sc::logit(x)); }
    double ppf(double q, const double *s) const override { return sc::expit(1.0 / s[1] * (norm_ppf(q) - s[0])); }
    double sf(double x, const double *s) const override { return norm_sf(s[0] + s[1] * sc::logit(x)); }
    double isf(double q, const double *s) const override { return sc::expit(1.0 / s[1] * (norm_isf(q) - s[0])); }
};
Registrar r_johnsonsb(new JohnsonSB);

struct JohnsonSU : Cont {
    JohnsonSU() { name = "johnsonsu"; shapes = "a, b"; nshape = 2; doc = "A Johnson SU continuous random variable."; }
    bool argcheck(const double *s) const override { return s[1] > 0 && s[0] == s[0]; }
    double pdf(double x, const double *s) const override
    {
        const double x2 = x * x;
        const double trm = norm_pdf(s[0] + s[1] * std::asinh(x));
        return s[1] * 1.0 / std::sqrt(x2 + 1.0) * trm;
    }
    double cdf(double x, const double *s) const override { return norm_cdf(s[0] + s[1] * std::asinh(x)); }
    double ppf(double q, const double *s) const override { return std::sinh((norm_ppf(q) - s[0]) / s[1]); }
    double sf(double x, const double *s) const override { return norm_sf(s[0] + s[1] * std::asinh(x)); }
    double isf(double x, const double *s) const override { return std::sinh((norm_isf(x) - s[0]) / s[1]); }
    void stats(const double *s, int moments, Stats4 &st) const override
    {
        const double a_ = s[0], b_ = s[1];
        const double bn2 = std::pow(b_, -2.);
        const double expbn2 = std::exp(bn2);
        const double a_b = a_ / b_;
        if (moments & MOM_M) st.set(0, -std::sqrt(expbn2) * std::sinh(a_b));
        if (moments & MOM_V) st.set(1, 0.5 * sc::expm1(bn2) * (expbn2 * std::cosh(2 * a_b) + 1));
        if (moments & MOM_S) {
            const double t1 = std::sqrt(expbn2) * std::sqrt(sc::expm1(bn2));
            const double t2 = 3 * std::sinh(a_b);
            const double t3 = expbn2 * (expbn2 + 2) * std::sinh(3 * a_b);
            const double denom = std::sqrt(2.0) * std::pow(1 + expbn2 * std::cosh(2 * a_b), 3.0 / 2);
            st.set(2, -t1 * (t2 + t3) / denom);
        }
        if (moments & MOM_K) {
            const double t1 = 3 + 6 * expbn2;
            const double t2 = 4 * sq(expbn2) * (expbn2 + 2) * std::cosh(2 * a_b);
            const double t3 = sq(expbn2) * std::cosh(4 * a_b);
            const double t4 = -3 + 3 * sq(expbn2) + 2 * std::pow(expbn2, 3.0) + std::pow(expbn2, 4.0);
            const double denom = 2 * sq(1 + expbn2 * std::cosh(2 * a_b));
            st.set(3, (t1 + t2 + t3 * t4) / denom - 3);
        }
    }
};
Registrar r_johnsonsu(new JohnsonSU);

/* ---------------------------------------------------------------- kappa3, kappa4 */

struct Kappa3 : Cont {
    Kappa3() { name = "kappa3"; shapes = "a"; nshape = 1; a = 0.0; doc = "Kappa 3 parameter distribution."; }
    double pdf(double x, const double *s) const override
    {
        const double a_ = s[0];
        return a_ * std::pow(a_ + std::pow(x, a_), -1.0 / a_ - 1);
    }
    double cdf(double x, const double *s) const override
    {
        const double a_ = s[0];
        return x * std::pow(a_ + std::pow(x, a_), -1.0 / a_);
    }
    double sf(double x, const double *s) const override
    {
        const double a_ = s[0];
        const double sf1 = 1.0 - cdf(x, s);
        const double cutoff = 0.01;
        if (sf1 < cutoff) {
            const double sf2 = -sc::expm1(sc::xlog1py(-1.0 / a_, a_ * std::pow(x, -a_)));
            return sf2 > cutoff ? sf1 : sf2;
        }
        return sf1;
    }
    double ppf(double q, const double *s) const override
    {
        const double a_ = s[0];
        return std::pow(a_ / (std::pow(q, -a_) - 1.0), 1.0 / a_);
    }
    double isf(double q, const double *s) const override
    {
        const double a_ = s[0];
        const double lg = sc::xlog1py(-a_, -q);
        const double denom = sc::expm1(lg);
        return std::pow(a_ / denom, 1.0 / a_);
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        for (int i = 1; i <= 4; i++)
            if (!(i < s[0])) st.set(i - 1, NaN);
    }
    double munp(int n, const double *s) const override
    {
        if (n >= s[0]) return NaN;                     /* _mom1_sc */
        return Cont::munp(n, s);
    }
};
Registrar r_kappa3(new Kappa3);

struct Kappa4 : Cont {
    Kappa4() { name = "kappa4"; shapes = "h, k"; nshape = 2; doc = "Kappa 4 parameter distribution."; }
    bool argcheck(const double *) const override { return true; }
    void get_support(const double *s, double &lo, double &hi) const override
    {
        const double h = s[0], k = s[1];
        lo = NaN;
        hi = NaN;
        if (h > 0 && k > 0) { lo = (1.0 - std::pow(h, -k)) / k; hi = 1.0 / k; }
        if (h > 0 && k == 0) { lo = std::log(h); hi = INF; }
        if (h > 0 && k < 0) { lo = (1.0 - std::pow(h, -k)) / k; hi = INF; }
        if (h <= 0 && k > 0) { lo = -INF; hi = 1.0 / k; }
        if (h <= 0 && k == 0) { lo = -INF; hi = INF; }
        if (h <= 0 && k < 0) { lo = 1.0 / k; hi = INF; }
    }
    double pdf(double x, const double *s) const override { return std::exp(logpdf(x, s)); }
    double logpdf(double x, const double *s) const override
    {
        const double h = s[0], k = s[1];
        double r = NaN;
        if (h != 0 && k != 0) r = sc::xlog1py(1.0 / k - 1.0, -k * x) + sc::xlog1py(1.0 / h - 1.0, -h * std::pow(1.0 - k * x, 1.0 / k));
        if (h == 0 && k != 0) r = sc::xlog1py(1.0 / k - 1.0, -k * x) - std::pow(1.0 - k * x, 1.0 / k);
        if (h != 0 && k == 0) r = -x + sc::xlog1py(1.0 / h - 1.0, -h * std::exp(-x));
        if (h == 0 && k == 0) r = -x - std::exp(-x);
        return r;
    }
    double cdf(double x, const double *s) const override { return std::exp(logcdf(x, s)); }
    double logcdf(double x, const double *s) const override
    {
        const double h = s[0], k = s[1];
        double r = NaN;
        if (h != 0 && k != 0) r = (1.0 / h) * sc::log1p(-h * std::pow(1.0 - k * x, 1.0 / k));
        if (h == 0 && k != 0) r = -std::pow(1.0 - k * x, 1.0 / k);
        if (h != 0 && k == 0) r = (1.0 / h) * sc::log1p(-h * std::exp(-x));
        if (h == 0 && k == 0) r = -std::exp(-x);
        return r;
    }
    double ppf(double q, const double *s) const override
    {
        const double h = s[0], k = s[1];
        double r = NaN;
        if (h != 0 && k != 0) r = 1.0 / k * (1.0 - std::pow((1.0 - std::pow(q, h)) / h, k));
        if (h == 0 && k != 0) r = 1.0 / k * (1.0 - std::pow(-std::log(q), k));
        if (h != 0 && k == 0) r = -sc::log1p(-std::pow(q, h)) + std::log(h);
        if (h == 0 && k == 0) r = -std::log(-std::log(q));
        return r;
    }
    static double stats_info(double h, double k)
    {
        double r = 5;                                  /* _lazyselect default; the last match wins */
        if (h < 0 && k >= 0) r = (double)(int64_t)(-1.0 / h * k);
        if (k < 0) r = (double)(int64_t)(-1.0 / k);
        return r;
    }
    void stats(const double *s, int, Stats4 &st) const override
    {
        const double maxr = stats_info(s[0], s[1]);
        for (int r = 1; r <= 4; r++)
            if (!(r < maxr)) st.set(r - 1, NaN);
    }
    double munp(int n, const double *s) const override
    {
        if (n >= stats_info(s[0], s[1])) return NaN;   /* _mom1_sc */
        return Cont::munp(n, s);
    }
};
Registrar r_kappa4(new Kappa4);

/* ---------------------------------------------------------------- ksone, kstwobign */

inline bool isintegral(double x) { return x == std::nearbyint(x); }

struct KSOne : Cont {
    KSOne() { name = "ksone"; shapes = "n"; nshape = 1; a = 0.0; b = 1.0; doc = "Kolmogorov-Smirnov one-sided test statistic distribution."; }
    bool argcheck(const double *s) const override { return s[0] >= 1 && isintegral(s[0]); }
    double pdf(double x, const double *s) const override { return -sc::_smirnovp(s[0], x); }
    double cdf(double x, const double *s) const override { return sc::_smirnovc(s[0], x); }
    double sf(double x, const double *s) const override { return sc::smirnov(s[0], x); }
    double ppf(double q, const double *s) const override { return sc::_smirnovci(s[0], q); }
    double isf(double q, const double *s) const override { return sc::smirnovi(s[0], q); }
};
Registrar r_ksone(new KSOne);

struct KSTwoBign : Cont {
    KSTwoBign()
    {
        name = "kstwobign"; a = 0.0;
        doc = "Limiting distribution of scaled Kolmogorov-Smirnov two-sided test statistic.";
    }
    double pdf(double x, const double *) const override { return -sc::_kolmogp(x); }
    double cdf(double x, const double *) const override { return sc::_kolmogc(x); }
    double sf(double x, const double *) const override { return sc::kolmogorov(x); }
    double ppf(double q, const double *) const override { return sc::_kolmogci(q); }
    double isf(double q, const double *) const override { return sc::kolmogi(q); }
};
Registrar r_kstwobign(new KSTwoBign);

/* ---------------------------------------------------------------- kstwo (scipy/stats/_ksstats.py) */

namespace ks {

const int E128 = 128;
const long double EP128 = std::ldexp(1.0L, E128);
const long double EM128 = std::ldexp(1.0L, -E128);
const double SQRT2PI = std::sqrt(2 * M_PI);
const double LOG_2PI = std::log(2 * M_PI);
const double MIN_LOG = -708;
const double SQRT3 = std::sqrt(3.0);
const double PI_SQUARED = lpow(M_PI, 2.0);
const double PI_FOUR = lpow(M_PI, 4.0);
const double PI_SIX = lpow(M_PI, 6.0);
const double STIRLING_COEFFS[8] = {-2.955065359477124183e-2, 6.4102564102564102564e-3, -1.9175269175269175269e-3,
                                   8.4175084175084175084e-4, -5.952380952380952381e-4, 7.9365079365079365079e-4,
                                   -2.7777777777777777778e-3, 8.3333333333333333333e-2};

/* a probability that NumPy may carry as np.longdouble (after a multiplication by _EP128 / _EM128) */
struct P {
    long double v;
    bool ld;
};
inline P pd(double v) { return {v, false}; }
inline double clip(double p) { return std::isnan(p) ? p : std::min(std::max(p, 0.0), 1.0); }
inline long double clipl(long double p) { return std::isnan(p) ? p : std::min(std::max(p, 0.0L), 1.0L); }
/* _select_and_clip_prob(p, 1 - p, cdf) */
inline P select_clip(P p, bool cdf)
{
    if (p.ld) return {clipl(cdf ? p.v : 1.0L - p.v), true};
    return pd(clip(cdf ? (double)p.v : 1.0 - (double)p.v));
}
inline P select_clip2(double cdfprob, double sfprob, bool cdf) { return pd(clip(cdf ? cdfprob : sfprob)); }

double log_nfactorial_div_n_pow_n(int n)
{
    const double rn = 1.0 / n;
    double y = 0.0;
    for (double c : STIRLING_COEFFS) y = y * (rn / n) + c;   /* np.polyval */
    return std::log((double)n) / 2 - n + LOG_2PI / 2 + rn * y;
}

/* OpenBLAS (SkylakeX kernel) ddot, which numpy's float64 dot calls: an FMA chain below 16 elements, blocks of
   16 (and 32 with 512-bit accumulators) above */
double blas_ddot(const double *x, const double *y, int n)
{
    double dot = 0.0;
    const int n1 = n & -16;
    if (n1) {
        double A[4][8] = {};
        int i = 0;
        const int n32 = n1 & ~31;
        for (; i < n32; i += 32)
            for (int q = 0; q < 4; q++)
                for (int l = 0; l < 8; l++) A[q][l] = std::fma(x[i + 8 * q + l], y[i + 8 * q + l], A[q][l]);
        double acc[4][4];
        for (int q = 0; q < 4; q++)
            for (int l = 0; l < 4; l++) acc[q][l] = A[q][l] + A[q][l + 4];
        for (; i < n1; i += 16)
            for (int q = 0; q < 4; q++)
                for (int l = 0; l < 4; l++) acc[q][l] = std::fma(x[i + 4 * q + l], y[i + 4 * q + l], acc[q][l]);
        double a0[4];
        for (int l = 0; l < 4; l++) a0[l] = ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l];
        dot = (a0[0] + a0[2]) + (a0[1] + a0[3]);
    }
    for (int i = n1; i < n; i++) dot = std::fma(y[i], x[i], dot);
    return 0.0 + dot;
}

/* np.convolve(a, v) (full mode): correlate(a, v[::-1]) with numpy's dot for the partial overlaps and its
   small_correlate (plain multiply-add, kernels up to 11 long) for the full ones */
std::vector<double> convolve(std::vector<double> a, std::vector<double> v)
{
    if (v.size() > a.size()) std::swap(a, v);
    const int n1 = (int)a.size(), n2 = (int)v.size();
    std::vector<double> vr(v.rbegin(), v.rend());
    std::vector<double> out;
    out.reserve((size_t)(n1 + n2 - 1));
    for (int i = 0; i < n2 - 1; i++) out.push_back(blas_ddot(a.data(), vr.data() + (n2 - 1 - i), i + 1));
    for (int i = 0; i < n1 - n2 + 1; i++) {
        if (n2 <= 11) {
            double s = 0;
            for (int j = 0; j < n2; j++) s += a[(size_t)(i + j)] * vr[(size_t)j];
            out.push_back(s);
        } else {
            out.push_back(blas_ddot(a.data() + i, vr.data(), n2));
        }
    }
    for (int i = 0; i < n2 - 1; i++) {
        const int n = n2 - 1 - i;
        out.push_back(blas_ddot(a.data() + (n1 - n), vr.data(), n));
    }
    return out;
}

/* np.matmul of square float64 matrices: OpenBLAS dgemm (SkylakeX), measured: an FMA chain over the inner
   index, except that for m > 16 with m % 8 in 1..4 the last m % 8 columns are dot products in 8 FMA lanes
   (k % 8) reduced as ((l0 + l1) + (l2 + l3)) + ((l4 + l5) + (l6 + l7)) (this last model reproduces most, not
   all, of those entries) */
std::vector<double> matmul(const std::vector<double> &A, const std::vector<double> &B, int m)
{
    std::vector<double> C((size_t)m * m);
    const int r = m % 8;
    const int lane_from = (m > 16 && r >= 1 && r <= 4) ? m - r : m;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            double s = 0.0;
            if (j < lane_from) {
                for (int k = 0; k < m; k++) s = std::fma(A[(size_t)(i * m + k)], B[(size_t)(k * m + j)], s);
            } else {
                double L[8] = {};
                for (int k = 0; k < m; k++) L[k % 8] = std::fma(A[(size_t)(i * m + k)], B[(size_t)(k * m + j)], L[k % 8]);
                s = ((L[0] + L[1]) + (L[2] + L[3])) + ((L[4] + L[5]) + (L[6] + L[7]));
            }
            C[(size_t)(i * m + j)] = s;
        }
    return C;
}

P kolmogn_DMTW(int n, double d, bool cdf)
{
    if (d >= 1.0) return select_clip2(1.0, 0.0, cdf);
    const double nd = n * d;
    if (nd <= 0.5) return select_clip2(0.0, 1.0, cdf);
    const int k = (int)std::ceil(nd);
    const double h = k - nd;
    const int m = 2 * k - 1;
    std::vector<double> H((size_t)m * m, 0.0), v((size_t)m), w((size_t)m);
    for (int j = 1; j <= m; j++) v[(size_t)j - 1] = 1.0 - std::pow(h, (double)j);
    double fac = 1.0;
    for (int j = 1; j <= m; j++) {
        w[(size_t)j - 1] = fac;
        fac /= j;
        v[(size_t)j - 1] *= fac;
    }
    const double tt = std::pow(std::max(2 * h - 1.0, 0.0), (double)m) - 2 * std::pow(h, (double)m);
    v[(size_t)m - 1] = (1.0 + tt) * fac;
    for (int i = 1; i < m; i++)
        for (int r = i - 1; r < m; r++) H[(size_t)(r * m + i)] = w[(size_t)(r - (i - 1))];
    for (int r = 0; r < m; r++) H[(size_t)(r * m)] = v[(size_t)r];
    for (int c = 0; c < m; c++) H[(size_t)((m - 1) * m + c)] = v[(size_t)(m - 1 - c)];
    std::vector<double> Hpwr((size_t)m * m, 0.0);
    for (int i = 0; i < m; i++) Hpwr[(size_t)(i * m + i)] = 1.0;
    int nn = n, expnt = 0, Hexpnt = 0;
    while (nn > 0) {
        if (nn % 2) {
            Hpwr = m == 1 ? std::vector<double>{Hpwr[0] * H[0]} : matmul(Hpwr, H, m);
            expnt += Hexpnt;
        }
        H = m == 1 ? std::vector<double>{H[0] * H[0]} : matmul(H, H, m);
        Hexpnt *= 2;
        if (std::fabs(H[(size_t)((k - 1) * m + k - 1)]) > EP128) {
            for (auto &e : H) e = (double)((long double)e / EP128);
            Hexpnt += E128;
        }
        nn = nn / 2;
    }
    P p = pd(Hpwr[(size_t)((k - 1) * m + k - 1)]);
    for (int i = 1; i <= n; i++) {
        if (p.ld) p.v = (long double)i * p.v / (long double)n;
        else p.v = (double)(i * (double)p.v / n);
        if (std::fabs(p.v) < EM128) {
            p.v *= EP128;
            p.ld = true;
            expnt -= E128;
        }
    }
    if (expnt != 0) p.v = p.ld ? std::ldexp(p.v, expnt) : (long double)std::ldexp((double)p.v, expnt);
    return select_clip(p, cdf);
}

void pomeranz_j1j2(int i, int n, int ll, int ceilf, int roundf, int &j1, int &j2)
{
    if (i == 0) {
        j1 = -ll - ceilf - 1;
        j2 = ll + ceilf - 1;
    } else {
        const int ip1div2 = (i + 1) / 2, ip1mod2 = (i + 1) % 2;
        if (ip1mod2 == 0) {
            if (ip1div2 == n + 1) {
                j1 = n - ll - ceilf - 1;
                j2 = n + ll + ceilf - 1;
            } else {
                j1 = ip1div2 - 1 - ll - roundf - 1;
                j2 = ip1div2 + ll - 1 + ceilf - 1;
            }
        } else {
            j1 = ip1div2 - 1 - ll - 1;
            j2 = ip1div2 + ll + roundf - 1;
        }
    }
    j1 = std::max(j1 + 2, 0);
    j2 = std::min(j2, n);
}

P kolmogn_Pomeranz(int n, double x, bool cdf)
{
    const double t = n * x;
    const int ll = (int)std::floor(t);
    const double f = 1.0 * (t - ll);
    const double g = std::min(f, 1.0 - f);
    const int ceilf = f > 0 ? 1 : 0;
    const int roundf = f > 0.5 ? 1 : 0;
    const int npwrs = 2 * (ll + 1);
    std::vector<double> gpower((size_t)npwrs), twogpower((size_t)npwrs), onem2gpower((size_t)npwrs);
    gpower[0] = twogpower[0] = onem2gpower[0] = 1.0;
    int expnt = 0;
    const double g_over_n = g / n, two_g_over_n = 2 * g / n, one_minus_two_g_over_n = (1 - 2 * g) / n;
    for (int m = 1; m < npwrs; m++) {
        gpower[(size_t)m] = gpower[(size_t)m - 1] * g_over_n / m;
        twogpower[(size_t)m] = twogpower[(size_t)m - 1] * two_g_over_n / m;
        onem2gpower[(size_t)m] = onem2gpower[(size_t)m - 1] * one_minus_two_g_over_n / m;
    }
    std::vector<double> V0((size_t)npwrs, 0.0), V1((size_t)npwrs, 0.0);
    V1[0] = 1;
    int V0s = 0, V1s = 0, j1, j2;
    pomeranz_j1j2(0, n, ll, ceilf, roundf, j1, j2);
    for (int i = 1; i < 2 * n + 2; i++) {
        const int k1 = j1;
        std::swap(V0, V1);
        std::swap(V0s, V1s);
        std::fill(V1.begin(), V1.end(), 0.0);
        pomeranz_j1j2(i, n, ll, ceilf, roundf, j1, j2);
        const std::vector<double> &pwrs = (i == 1 || i == 2 * n + 1) ? gpower : (i % 2 ? twogpower : onem2gpower);
        const int ln2 = j2 - k1 + 1;
        if (ln2 > 0) {
            const int s0 = std::max(k1 - V0s, 0), s1 = std::min(k1 - V0s + ln2, npwrs);   /* the slice */
            std::vector<double> a(V0.begin() + s0, V0.begin() + std::max(s1, s0));
            std::vector<double> pv(pwrs.begin(), pwrs.begin() + std::min(ln2, npwrs));
            const std::vector<double> conv = convolve(a, pv);
            const int conv_start = j1 - k1, conv_len = j2 - j1 + 1;
            for (int q = 0; q < conv_len && conv_start + q < (int)conv.size(); q++) V1[(size_t)q] = conv[(size_t)(conv_start + q)];
            const double mx = *std::max_element(V1.begin(), V1.end());
            if (0 < mx && mx < EM128) {
                for (auto &e : V1) e = (double)((long double)e * EP128);
                expnt -= E128;
            }
            V1s = V0s + j1 - k1;
        }
    }
    P ans = pd(V1[(size_t)(n - V1s)]);
    for (int m = 1; m <= n; m++) {
        if (std::fabs(ans.v) > EP128) {
            ans.v *= EM128;
            ans.ld = true;
            expnt += E128;
        }
        if (ans.ld) ans.v *= (long double)m;
        else ans.v = (double)ans.v * m;
    }
    if (expnt != 0) ans.v = ans.ld ? std::ldexp(ans.v, expnt) : (long double)std::ldexp((double)ans.v, expnt);
    return select_clip(ans, cdf);
}

double kolmogn_PelzGood(int n, double x, bool cdf)
{
    if (x <= 0.0) return clip(cdf ? 0.0 : 1.0);
    if (x >= 1.0) return clip(cdf ? 1.0 : 0.0);
    const double z = std::sqrt((double)n) * x;
    const double zsquared = lpow(z, 2.0), zthree = lpow(z, 3.0), zfour = lpow(z, 4.0), zsix = lpow(z, 6.0);
    const double qlog = -PI_SQUARED / 8 / zsquared;
    if (qlog < MIN_LOG) return clip(cdf ? 0.0 : 1.0);
    double q = std::exp(qlog);
    const double k1a = -zsquared;
    const double k1b = PI_SQUARED / 4;
    const double k2a = 6 * zsix + 2 * zfour;
    const double k2b = (2 * zfour - 5 * zsquared) * PI_SQUARED / 4;
    const double k2c = PI_FOUR * (1 - 2 * zsquared) / 16;
    const double k3d = PI_SIX * (5 - 30 * zsquared) / 64;
    const double k3c = PI_FOUR * (-60 * zsquared + 212 * zfour) / 16;
    const double k3b = PI_SQUARED * (135 * zfour - 96 * zsix) / 4;
    const double k3a = -30 * zsix - 90 * lpow(z, 8.0);
    double K[4] = {0, 0, 0, 0};
    const int maxk = (int)std::ceil(16 * z / M_PI);
    for (int k = maxk; k > 0; k--) {
        const long m = 2 * k - 1;
        const double msquared = (double)(m * m), mfour = (double)(m * m * m * m), msix = (double)(m * m * m * m * m * m);
        const double qpower = std::pow(q, (double)(8 * k));
        const double coeffs[4] = {1.0, k1a + k1b * msquared, k2a + k2b * msquared + k2c * mfour,
                                  k3a + k3b * msquared + k3c * mfour + k3d * msix};
        for (int j = 0; j < 4; j++) {
            K[j] *= qpower;
            K[j] += coeffs[j];
        }
    }
    for (int j = 0; j < 4; j++) K[j] *= q;
    for (int j = 0; j < 4; j++) K[j] *= SQRT2PI;
    const double dv[4] = {z, 6 * zfour, 72 * lpow(z, 7.0), 6480 * lpow(z, 10.0)};
    for (int j = 0; j < 4; j++) K[j] /= dv[j];
    q = std::exp(-PI_SQUARED / 2 / zsquared);
    const double sqrt3z = SQRT3 * z;
    std::vector<double> t2((size_t)std::max(maxk, 0)), t3((size_t)std::max(maxk, 0));
    for (int i = 0; i < maxk; i++) {
        const long ks = maxk - i;
        const double ksquared = (double)(ks * ks);
        const double kspi = M_PI * (double)ks;
        const double qp = std::pow(q, ksquared);
        t2[(size_t)i] = ksquared * qp;
        t3[(size_t)i] = (sqrt3z + kspi) * (sqrt3z - kspi) * ksquared * qp;
    }
    double k2extra = tsr_psum(t2.data(), maxk);
    k2extra *= PI_SQUARED * SQRT2PI / (-36 * zthree);
    K[2] += k2extra;
    double k3extra = tsr_psum(t3.data(), maxk);
    k3extra *= PI_SQUARED * SQRT2PI / (216 * zsix);
    K[3] += k3extra;
    for (int j = 0; j < 4; j++) K[j] /= std::pow(n * 1.0, j / 2.0);
    if (!cdf) {
        for (int j = 0; j < 4; j++) K[j] *= -1;
        K[0] += 1;
    }
    return ((0 + K[0]) + K[1]) + K[2] + K[3];
}

P kolmogn(int n, double x, bool cdf)
{
    if (n <= 0) return pd(NaN);
    if (x >= 1.0) return select_clip2(1.0, 0.0, cdf);
    if (x <= 0.0) return select_clip2(0.0, 1.0, cdf);
    const double t = n * x;
    if (t <= 1.0) {
        if (t <= 0.5) return select_clip2(0.0, 1.0, cdf);
        double prob;
        if (n <= 140) {
            prob = 1.0;
            for (int i = 1; i <= n; i++) {
                const double e = i * (1.0 / n) * (2 * t - 1);
                prob = i == 1 ? e : prob * e;
            }
        } else {
            prob = std::exp(log_nfactorial_div_n_pow_n(n) + n * std::log(2 * t - 1));
        }
        return select_clip2(prob, 1.0 - prob, cdf);
    }
    if (t >= n - 1) {
        const double prob = 2 * std::pow(1.0 - x, (double)n);
        return select_clip2(1 - prob, prob, cdf);
    }
    if (x >= 0.5) {
        const double prob = 2 * sc::smirnov((double)n, x);
        return select_clip2(1.0 - prob, prob, cdf);
    }
    const double nxsquared = t * x;
    if (n <= 140) {
        if (nxsquared <= 0.754693) return select_clip(kolmogn_DMTW(n, x, true), cdf);
        if (nxsquared <= 4) return select_clip(kolmogn_Pomeranz(n, x, true), cdf);
        const double prob = 2 * sc::smirnov((double)n, x);
        return select_clip2(1.0 - prob, prob, cdf);
    }
    if (!cdf) {
        if (nxsquared >= 370.0) return pd(0.0);
        if (nxsquared >= 2.2) return pd(clip(2 * sc::smirnov((double)n, x)));
    }
    P cdfprob;
    if (nxsquared >= 18.0) cdfprob = pd(1.0);
    else if (n <= 100000 && n * std::pow(x, 1.5) <= 1.4) cdfprob = kolmogn_DMTW(n, x, true);
    else cdfprob = pd(kolmogn_PelzGood(n, x, true));
    return select_clip(cdfprob, cdf);
}

/* the float64 that numpy stores for the result */
inline double f64(P p) { return (double)p.v; }

double kolmogn_p(int n, double x)
{
    if (n <= 0) return NaN;
    if (x >= 1.0 || x <= 0) return 0;
    const double t = n * x;
    if (t <= 1.0) {
        if (t <= 0.5) return 0.0;
        double prd;
        if (n <= 140) {
            prd = 1.0;
            for (int i = 1; i < n; i++) {
                const double e = i * (1.0 / n) * (2 * t - 1);
                prd = i == 1 ? e : prd * e;
            }
        } else {
            prd = std::exp(log_nfactorial_div_n_pow_n(n) + (n - 1) * std::log(2 * t - 1));
        }
        return prd * 2 * (double)((long)n * n);
    }
    if (t >= n - 1) return 2 * std::pow(1.0 - x, (double)(n - 1)) * n;
    if (x >= 0.5) return 2 * -sc::_smirnovp((double)n, x);     /* 2 * ksone.pdf(x, n) */
    double delta = x / lpow(2.0, 16.0);
    delta = std::min(delta, x - 1.0 / n);
    delta = std::min(delta, 0.5 - x);
    /* _derivative(kolmogn(n, .), x, dx=delta, order=5) */
    static const double w[5] = {1 / 12.0, -8 / 12.0, 0 / 12.0, 8 / 12.0, -1 / 12.0};
    double val = 0.0;
    for (int k = 0; k < 5; k++) val += w[k] * f64(kolmogn(n, x + (k - 2) * delta, true));
    return val / delta;
}

double kolmogni(int n, double p, double q)
{
    if (n <= 0) return NaN;
    if (p <= 0) return 1.0 / n;
    if (q <= 0) return 1.0;
    const double delta = std::exp((std::log(p) - sc::loggamma((double)n + 1)) / n);
    if (delta <= 1.0 / n) return (delta + 1.0 / n) / 2;
    const double x = -std::expm1(std::log(q / 2.0) / n);
    if (x >= 1 - 1.0 / n) return x;
    double x1 = sc::_kolmogci(p) / std::sqrt((double)n);
    x1 = std::min(x1, 1.0 - 1.0 / n);
    auto f = [&](double xx) {
        const P r = kolmogn(n, xx, true);
        return r.ld ? (double)(r.v - (long double)p) : (double)r.v - p;
    };
    return brentq(f, 1.0 / n, x1, 1e-14);
}

}  // namespace ks

struct KSTwo : Cont {
    KSTwo()
    {
        name = "kstwo"; shapes = "n"; nshape = 1; momtype = 0; a = 0.0; b = 1.0;
        doc = "Kolmogorov-Smirnov two-sided test statistic distribution.";
    }
    bool argcheck(const double *s) const override { return s[0] >= 1 && isintegral(s[0]); }
    void get_support(const double *s, double &lo, double &hi) const override { lo = 0.5 / s[0]; hi = 1.0; }
    /* kolmogn / kolmognp / kolmogni: NaN n passes through; a non-integral n raises */
    static bool bad_n(double n, double &out)
    {
        if (std::isnan(n)) { out = n; return true; }
        if ((double)(int64_t)n != n) { fail("n is not integral"); out = NaN; return true; }
        return false;
    }
    double pdf(double x, const double *s) const override
    {
        double r;
        if (bad_n(s[0], r)) return r;
        return ks::kolmogn_p((int)s[0], x);
    }
    double cdf(double x, const double *s) const override
    {
        double r;
        if (bad_n(s[0], r)) return r;
        return ks::f64(ks::kolmogn((int)s[0], x, true));
    }
    double sf(double x, const double *s) const override
    {
        double r;
        if (bad_n(s[0], r)) return r;
        return ks::f64(ks::kolmogn((int)s[0], x, false));
    }
    double ppf(double q, const double *s) const override
    {
        double r;
        if (bad_n(s[0], r)) return r;
        return ks::kolmogni((int)s[0], q, 1 - q);
    }
    double isf(double q, const double *s) const override
    {
        double r;
        if (bad_n(s[0], r)) return r;
        return ks::kolmogni((int)s[0], 1 - q, q);
    }
};
Registrar r_kstwo(new KSTwo);

/* ---------------------------------------------------------------- landau */

struct Landau : Cont {
    Landau() { name = "landau"; doc = "A Landau continuous random variable."; }
    double entropy(const double *) const override { return 2.37263644000448182; }
    double pdf(double x, const double *) const override { return sc::_landau_pdf(x, 0, 1); }
    double cdf(double x, const double *) const override { return sc::_landau_cdf(x, 0, 1); }
    double sf(double x, const double *) const override { return sc::_landau_sf(x, 0, 1); }
    double ppf(double p, const double *) const override { return sc::_landau_ppf(p, 0, 1); }
    double isf(double p, const double *) const override { return sc::_landau_isf(p, 0, 1); }
    void stats(const double *, int, Stats4 &st) const override { st.set(NaN, NaN, NaN, NaN); }
    double munp(int n, const double *) const override { return n > 0 ? NaN : 1; }
    void rvs(bitgen_t *bg, int64_t n, const double *const *, double *out) const override
    {
        const double pi_2 = M_PI / 2;
        std::vector<double> W((size_t)n);
        for (int64_t i = 0; i < n; i++) out[i] = random_uniform(bg, -M_PI / 2, M_PI / 2 - (-M_PI / 2));
        for (int64_t i = 0; i < n; i++) W[(size_t)i] = random_standard_exponential(bg);
        for (int64_t i = 0; i < n; i++) {
            const double U = out[i];
            out[i] = 2 / M_PI * ((pi_2 + U) * std::tan(U) - std::log((pi_2 * W[(size_t)i] * std::cos(U)) / (pi_2 + U)));
        }
    }
};
Registrar r_landau(new Landau);

}  // namespace
}  // namespace tsd
