/*
 * scipy.stats distributions for the function registry (ADR 0011): the machinery of rv_continuous and
 * rv_discrete (scipy/stats/_distn_infrastructure.py, SciPy 1.17.1), ported method by method, so that a
 * distribution written here with SciPy's private methods (_pdf, _cdf, _ppf, _stats, ...) behaves exactly like
 * SciPy's: the same support and argument checks in the public methods, the same generic fallbacks (QUADPACK
 * integration, brentq root finding, the discrete bisection and expectation sums) where a private method is
 * not given, and random variates drawn with NumPy's own distribution code from the caller's PCG64 stream.
 *
 * A distribution is a subclass of Cont or Disc that overrides the private methods SciPy's class defines and
 * registers itself with a static Registrar (see dist_*.cpp). Everything is evaluated for one element; the
 * shape parameters come in s[0..nshape-1].
 */
#pragma once

#include "gen_sc.hpp"   /* scipy.special as sc::name(...): the registry's own element functions */

#include <cmath>
#include <cstdint>
#include <limits>
#include <initializer_list>
#include <vector>
#include <new>
#include <exception>
#include <stdexcept>

extern "C" {
#include "numpy/random/distributions.h"
double tsr_psum(const double *x, int64_t n);   /* numpy's pairwise summation (reduce.c) */
int64_t tsr_budget(void);                       /* the kernel's memory budget in bytes (0: unlimited) */
}

namespace tsd {

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

/* SciPy's _stats returns a tuple of 4 values where None means "compute from moments" */
struct Stats4 {
    double v[4] = {NaN, NaN, NaN, NaN};
    bool has[4] = {false, false, false, false};
    void set(int k, double x) { v[k] = x; has[k] = true; }
    void set(double m, double v2, double s, double k) { set(0, m); set(1, v2); set(2, s); set(3, k); }
};

/* moments requested from _stats ('m' 1, 'v' 2, 's' 4, 'k' 8), as SciPy's `moments` keyword */
enum { MOM_M = 1, MOM_V = 2, MOM_S = 4, MOM_K = 8 };

/* record a failure that SciPy raises for the whole call (the binding throws ValueError) */
void fail(const char *msg);

/* scipy.integrate.quad (QUADPACK qagse/qagie, epsabs = epsrel = 1.49e-8, limit = 50) */
template <class F> double quad(F &&f, double a, double b, int *ier = nullptr);
/* scipy.optimize.brentq with SciPy's defaults (rtol = 4 eps, maxiter = 100); raises (fail) as SciPy does */
template <class F> double brentq(F &&f, double a, double b, double xtol);

struct Dist {
    const char *name = "";
    const char *shapes = "";                 /* "a, b" */
    int nshape = 0;
    const char *doc = "";
    double a = -INF, b = INF;                /* support of the standard form */
    bool discrete = false;
    virtual ~Dist() = default;

    /* rv_generic._argcheck: every shape parameter > 0 */
    virtual bool argcheck(const double *s) const
    {
        for (int k = 0; k < nshape; k++)
            if (!(s[k] > 0)) return false;
        return true;
    }
    virtual void get_support(const double *s, double &lo, double &hi) const
    {
        (void)s;
        lo = a;
        hi = b;
    }
    /* _stats: set the entries SciPy's _stats returns (not None); `moments` as SciPy passes it */
    virtual void stats(const double *s, int moments, Stats4 &st) const { (void)s; (void)moments; (void)st; }
    /* _munp(n): non-central moment; default generic_moment (integration or summation) */
    virtual double munp(int n, const double *s) const = 0;
    virtual double entropy(const double *s) const = 0;
    /* _rvs: `n` variates, shape parameters broadcast to n (sa[k][i]); default inversion of a uniform */
    virtual void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const = 0;
};

struct Cont : Dist {
    double xtol = 1e-14;
    int momtype = 1;                          /* 0: moments by integrating x^n pdf, 1: by integrating ppf^n */

    virtual double pdf(double x, const double *s) const;           /* default: derivative of _cdf */
    virtual double logpdf(double x, const double *s) const { return std::log(pdf(x, s)); }
    virtual double cdf(double x, const double *s) const;           /* default: quad(_pdf, a, x) */
    virtual double sf(double x, const double *s) const { return 1.0 - cdf(x, s); }
    virtual double logcdf(double x, const double *s) const;        /* default: rv_continuous (median split) */
    virtual double logsf(double x, const double *s) const;
    virtual double ppf(double q, const double *s) const;           /* default: _ppf_single (brentq) */
    virtual double isf(double q, const double *s) const { return ppf(1.0 - q, s); }
    /* an amount a distribution's own public pdf, cdf and rvs add to loc (levy_stable's S1 parameterization
       for alpha == 1: 2 beta scale log(scale) / pi); SciPy's other public methods do not apply it */
    virtual double public_loc_shift(const double *s, double scale) const { (void)s; (void)scale; return 0.0; }
    double munp(int n, const double *s) const override;            /* generic_moment */
    double entropy(const double *s) const override;                /* integral of entr(pdf) */
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override;

    /* the public methods on the standard form (loc 0, scale 1), with SciPy's support handling */
    double pub_cdf(double x, const double *s) const;
    double pub_pdf(double x, const double *s) const;
    double pub_ppf(double q, const double *s) const;
};

struct Disc : Dist {
    double inc = 1.0;
    double moment_tol = 1e-8;
    Disc() { discrete = true; a = 0; }

    virtual double pmf(double k, const double *s) const { return cdf(k, s) - cdf(k - 1, s); }
    virtual double logpmf(double k, const double *s) const { return std::log(pmf(k, s)); }
    virtual double cdf(double x, const double *s) const;           /* _cdfvec(floor(x)): sum of pmf */
    virtual double sf(double x, const double *s) const { return 1.0 - cdf(x, s); }
    virtual double logcdf(double x, const double *s) const { return std::log(cdf(x, s)); }
    virtual double logsf(double x, const double *s) const { return std::log(sf(x, s)); }
    virtual double ppf(double q, const double *s) const;           /* _drv2_ppfsingle */
    virtual double isf(double q, const double *s) const { return ppf(1.0 - q, s); }
    double munp(int n, const double *s) const override;             /* _drv2_moment */
    double entropy(const double *s) const override;
    void rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const override;
};

/* a registered distribution by its SciPy name (nullptr when absent), and its public methods with SciPy's
   semantics: method is DM_PDF ... DM_ISF (fn.h), x the point or probability, params the shape parameters
   followed by loc and (continuous) scale -- for the scipy.stats functions that compute p-values */
const Dist *find(const char *name);
double call(const Dist *d, int method, double x, std::initializer_list<double> params);

/* registration: a static Registrar in each distribution's file */
void register_dist(const Dist *d);
struct Registrar {
    explicit Registrar(const Dist *d) { register_dist(d); }
};

/* ---------------------------------------------------------------- numerical tools */

struct QuadCall {
    double (*fn)(void *, double);
    void *ctx;
};
double quad_impl(const QuadCall &call, double a, double b, int *ier);

/* The integrand and root functions are called back from SciPy's C code (QUADPACK, brentq). A C++ exception must
   not unwind through those C frames: the trampolines park it, return NaN to the C routine, and rethrow it once
   the C routine has returned. */
template <class F> struct CallbackCtx {
    F *f;
    std::exception_ptr err;
};

template <class F> double quad(F &&f, double a, double b, int *ier)
{
    using Fn = typename std::remove_reference<F>::type;
    CallbackCtx<Fn> ctx{&f, nullptr};
    QuadCall c;
    c.ctx = (void *)&ctx;
    c.fn = [](void *p, double x) -> double {
        auto *k = (CallbackCtx<Fn> *)p;
        if (k->err) return std::numeric_limits<double>::quiet_NaN();
        try {
            return (*k->f)(x);
        } catch (...) {
            k->err = std::current_exception();
            return std::numeric_limits<double>::quiet_NaN();
        }
    };
    const double r = quad_impl(c, a, b, ier);
    if (ctx.err) std::rethrow_exception(ctx.err);
    return r;
}

double brentq_impl(double (*fn)(double, void *), void *ctx, double a, double b, double xtol);

template <class F> double brentq(F &&f, double a, double b, double xtol)
{
    using Fn = typename std::remove_reference<F>::type;
    CallbackCtx<Fn> ctx{&f, nullptr};
    const double r = brentq_impl(
        [](double x, void *p) -> double {
            auto *k = (CallbackCtx<Fn> *)p;
            if (k->err) return std::numeric_limits<double>::quiet_NaN();
            try {
                return (*k->f)(x);
            } catch (...) {
                k->err = std::current_exception();
                return std::numeric_limits<double>::quiet_NaN();
            }
        },
        (void *)&ctx, a, b, xtol);
    if (ctx.err) std::rethrow_exception(ctx.err);
    return r;
}

/* an element count computed from a parameter (np.arange(n + 1), a support length, ...) as an allocation size.
   NaN or negative -> 0. Beyond any possible array: std::length_error ("Maximum allowed size exceeded", NumPy's
   ValueError); over the kernel's memory budget: std::bad_alloc (NumPy's MemoryError). fn_guard.cpp and the
   distribution entry points turn both into the call's error, never a crash or a double -> int overflow. */
double physical_memory_bytes();   /* stats_dist.cpp: installed RAM (0 when unknown) */

inline int64_t alloc_count(double count, double elem_bytes = sizeof(double))
{
    if (!(count > 0)) return 0;
    if (count > 4.5e15) throw std::length_error("Maximum allowed size exceeded");
    /* the kernel's budget when one is set; otherwise more than the machine's RAM is refused up front, as
       glibc's heuristic overcommit refuses NumPy's allocation (under overcommit_memory=1 the request would
       otherwise succeed and the process be OOM-killed while filling it) */
    const int64_t budget = tsr_budget();
    const double limit = budget > 0 ? (double)budget : physical_memory_bytes();
    if (limit > 0 && count * elem_bytes > limit) throw std::bad_alloc();
    return (int64_t)count;
}

/* NumPy's samplers as scipy.stats reaches them: through numpy.random.Generator, whose Python layer checks the
   parameters before calling the C code (numpy/random/_generator.pyx). The distributions call these by their C
   names; inside namespace tsd these checked versions are found first. An invalid parameter raises NumPy's
   ValueError, as SciPy's rvs does, instead of reaching C code that loops forever or overflows. */
constexpr double NP_POISSON_LAM_MAX = 9223372036854775807.0 - 3037000499.97605 * 10;
[[noreturn]] inline void np_param_error(const char *msg) { throw std::invalid_argument(msg); }
/* a float parameter NumPy converts to int64 (binomial n, hypergeometric counts): NaN and values outside int64
   fail there, never a C++ undefined conversion */
inline int64_t np_int64(double v)
{
    if (v != v) np_param_error("cannot convert float NaN to integer");
    if (!(v > -9223372036854775808.0 && v < 9223372036854775808.0)) np_param_error("Python int too large to convert to C long");
    return (int64_t)v;
}
inline int64_t np_checked_poisson(bitgen_t *bg, double lam)
{
    if (!(lam >= 0)) np_param_error("lam < 0 or lam is NaN");
    if (lam > NP_POISSON_LAM_MAX) np_param_error("lam value too large");
    return (int64_t)::random_poisson(bg, lam);
}
inline int64_t np_checked_negative_binomial(bitgen_t *bg, double n, double p)
{
    if (n != n) np_param_error("n must not be NaN");
    if (!(n > 0)) np_param_error("n <= 0");
    if (!(p > 0) || !(p <= 1)) np_param_error("p <= 0, p > 1 or p contains NaNs");
    if (!((1 - p) / p * (n + 10 * std::sqrt(n)) <= NP_POISSON_LAM_MAX))
        np_param_error("n too large or p too small, see Generator.negative_binomial Notes");
    return (int64_t)::random_negative_binomial(bg, n, p);
}
inline int64_t np_checked_binomial(bitgen_t *bg, double p, int64_t n, binomial_t *b)
{
    if (!(p >= 0) || !(p <= 1)) np_param_error("p < 0, p > 1 or p is NaN");
    if (n < 0) np_param_error("n < 0");
    return ::random_binomial(bg, p, n, b);
}
inline int64_t np_checked_zipf(bitgen_t *bg, double a)
{
    if (!(a > 1)) np_param_error("a <= 1 or a is NaN");
    return (int64_t)::random_zipf(bg, a);
}
inline int64_t np_checked_logseries(bitgen_t *bg, double p)
{
    if (!(p >= 0) || !(p < 1)) np_param_error("p < 0, p >= 1 or p is NaN");
    return ::random_logseries(bg, p);
}
inline int64_t np_checked_geometric(bitgen_t *bg, double p)
{
    if (!(p > 0) || !(p <= 1)) np_param_error("p <= 0, p > 1 or p contains NaNs");
    return ::random_geometric(bg, p);
}
inline int64_t np_checked_hypergeometric(bitgen_t *bg, int64_t good, int64_t bad, int64_t sample)
{
    if (good >= 1000000000 || bad >= 1000000000) np_param_error("both ngood and nbad must be less than 1000000000");
    if (good < 0) np_param_error("ngood < 0");
    if (bad < 0) np_param_error("nbad < 0");
    if (sample < 0) np_param_error("nsample < 0");
    if (good + bad < sample) np_param_error("ngood + nbad < nsample");
    return ::random_hypergeometric(bg, good, bad, sample);
}
inline double np_checked_wald(bitgen_t *bg, double mean, double scale)
{
    if (!(mean > 0)) np_param_error("mean <= 0");
    if (!(scale > 0)) np_param_error("scale <= 0");
    return ::random_wald(bg, mean, scale);
}
inline double np_checked_vonmises(bitgen_t *bg, double mu, double kappa)
{
    if (!(kappa >= 0)) np_param_error("kappa < 0");
    return ::random_vonmises(bg, mu, kappa);
}
inline double np_checked_noncentral_chisquare(bitgen_t *bg, double df, double nonc)
{
    if (!(df > 0)) np_param_error("df <= 0");
    if (!(nonc >= 0)) np_param_error("nonc < 0");
    return ::random_noncentral_chisquare(bg, df, nonc);
}
inline double np_checked_noncentral_f(bitgen_t *bg, double dfnum, double dfden, double nonc)
{
    if (!(dfnum > 0)) np_param_error("dfnum <= 0");
    if (!(dfden > 0)) np_param_error("dfden <= 0");
    if (!(nonc >= 0)) np_param_error("nonc < 0");
    return ::random_noncentral_f(bg, dfnum, dfden, nonc);
}

/* every later use of these C names in the distributions reaches the checked version (a plain overload would be
   ambiguous: argument-dependent lookup on bitgen_t also finds the C function) */
#define random_poisson np_checked_poisson
#define random_negative_binomial np_checked_negative_binomial
#define random_binomial np_checked_binomial
#define random_zipf np_checked_zipf
#define random_logseries np_checked_logseries
#define random_geometric np_checked_geometric
#define random_hypergeometric np_checked_hypergeometric
#define random_wald np_checked_wald
#define random_vonmises np_checked_vonmises
#define random_noncentral_chisquare np_checked_noncentral_chisquare
#define random_noncentral_f np_checked_noncentral_f

/* numpy helpers the distributions use */
inline double xlogy(double x, double y) { return (x == 0 && !std::isnan(y)) ? 0.0 : x * std::log(y); }
inline double xlog1py(double x, double y) { return (x == 0 && !std::isnan(y)) ? 0.0 : x * std::log1p(y); }
inline double entr(double x) { return std::isnan(x) ? x : (x > 0 ? -x * std::log(x) : (x == 0 ? 0.0 : -INF)); }
inline double sq(double x) { return x * x; }
/* scipy.special.factorial2(n) for a Python int n (exact=False): _factorialx_approx_core with k = 2 */
inline double factorial2(long n)
{
    if (n < 0) return 0.0;
    if (n == 0 || n == 1) return 1.0;
    const long r = n % 2;
    return std::pow(2.0, (double)(n - r) / 2) * sc::gamma((double)n / 2 + 1) / sc::gamma((double)r / 2 + 1) * (double)(r > 1 ? r : 1);
}

}  // namespace tsd
