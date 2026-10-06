/*
 * The generic machinery of scipy.stats rv_continuous / rv_discrete (SciPy 1.17.1,
 * scipy/stats/_distn_infrastructure.py) and the registry's C interface to it (fn_core.c calls dist_method and
 * dist_rvs_array). See stats_dist.hpp.
 */
#include "stats_dist.hpp"

#include <algorithm>
#include <atomic>
#include <new>
#include <stdexcept>
#include <cstdio>
#include <cstring>
#include <vector>

#include "../src/fn.h"

extern "C" {
#include "quadpack/quadpack.h"
#include "zeros/zeros.h"

}

#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace tsd {

double physical_memory_bytes()
{
    static const double bytes = [] {
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
        const long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGESIZE);
        return pages > 0 && size > 0 ? (double)pages * (double)size : 0.0;
#elif defined(_WIN32)
        MEMORYSTATUSEX st;
        st.dwLength = sizeof st;
        return GlobalMemoryStatusEx(&st) ? (double)st.ullTotalPhys : 0.0;
#else
        return 0.0;
#endif
    }();
    return bytes;
}


/* ---------------------------------------------------------------- errors */

/* process-wide (the element loops may run on OpenMP worker threads); the registry reads and clears it after
   each call. Concurrent calls from several PHP threads share it: the first message wins. */
static char g_fail[256];
static std::atomic<int> g_failed{0};

void fail(const char *msg)
{
    int expected = 0;
    if (g_failed.compare_exchange_strong(expected, 2)) {
        std::snprintf(g_fail, sizeof g_fail, "%s", msg);
        g_failed.store(1);
    }
}

/* ---------------------------------------------------------------- quad (scipy.integrate.quad) */

static thread_local const QuadCall *g_quad = nullptr;

static double quad_thunk(double *x) { return g_quad->fn(g_quad->ctx, *x); }

double quad_impl(const QuadCall &call, double a, double b, int *ier_out)
{
    /* quad(): a == b -> 0; limits flipped for b < a; qagse for finite limits, qagie otherwise */
    if (a == b) {
        if (ier_out) *ier_out = 0;
        return 0.0;
    }
    const bool flip = b < a;
    const double lo = std::min(a, b), hi = std::max(a, b);
    enum { LIMIT = 50 };
    double alist[LIMIT], blist[LIMIT], rlist[LIMIT], elist[LIMIT];
    int iord[LIMIT];
    double result = 0.0, abserr = 0.0;
    int neval = 0, ier = 6, last = 0;
    const QuadCall *saved = g_quad;         /* integrands may integrate (nested quad): keep a stack */
    g_quad = &call;
    if (hi != INF && lo != -INF) {
        dqagse(quad_thunk, lo, hi, 1.49e-8, 1.49e-8, LIMIT, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);
    } else {
        int inf;
        double bound;
        if (hi == INF && lo != -INF) { inf = 1; bound = lo; }
        else if (hi == INF && lo == -INF) { inf = 2; bound = 0.0; }
        else { inf = -1; bound = hi; }
        dqagie(quad_thunk, bound, inf, 1.49e-8, 1.49e-8, LIMIT, &result, &abserr, &neval, &ier, alist, blist, rlist, elist, iord, &last);
    }
    g_quad = saved;
    if (ier_out) *ier_out = ier;
    if (ier == 6) fail("quad: the input is invalid");
    return flip ? -result : result;
}

/* ---------------------------------------------------------------- brentq (scipy.optimize.brentq) */

struct BrentCtx {
    double (*fn)(double, void *);
    void *ctx;
    bool nan;
};

static double brent_thunk(double x, void *p)
{
    BrentCtx *c = (BrentCtx *)p;
    const double v = c->fn(x, c->ctx);
    if (std::isnan(v)) c->nan = true;     /* _wrap_nan_raise: SciPy raises ValueError at the first NaN */
    return v;
}

double brentq_impl(double (*fn)(double, void *), void *ctx, double a, double b, double xtol)
{
    BrentCtx c{fn, ctx, false};
    scipy_zeros_info info{0, 0, 0};
    const double rtol = 4 * std::numeric_limits<double>::epsilon();
    const double r = ::brentq(brent_thunk, a, b, xtol, rtol, 100, &c, &info);
    if (c.nan) { fail("brentq: the function value is NaN; solver cannot continue"); return NaN; }
    if (info.error_num == SIGNERR) { fail("brentq: f(a) and f(b) must have different signs"); return NaN; }
    if (info.error_num == CONVERR) { fail("brentq: failed to converge after 100 iterations"); return NaN; }
    return r;
}

/* ---------------------------------------------------------------- rv_continuous generics */

double Cont::pdf(double x, const double *s) const
{
    /* _derivative(self._cdf, x, dx=1e-5, order=5) */
    static const double w[5] = {1 / 12.0, -8 / 12.0, 0.0, 8 / 12.0, -1 / 12.0};
    const double dx = 1e-5;
    double val = 0.0;
    for (int k = 0; k < 5; k++) val += w[k] * cdf(x + (k - 2) * dx, s);
    return val / dx;
}

double Cont::cdf(double x, const double *s) const
{
    double lo, hi;
    get_support(s, lo, hi);
    return quad([&](double t) { return pdf(t, s); }, lo, x);
}

double Cont::logcdf(double x, const double *s) const
{
    const double median = ppf(0.5, s);
    return x < median ? std::log(cdf(x, s)) : std::log1p(-sf(x, s));
}

double Cont::logsf(double x, const double *s) const
{
    const double median = ppf(0.5, s);
    return x > median ? std::log(sf(x, s)) : std::log1p(-cdf(x, s));
}

double Cont::pub_cdf(double x, const double *s) const
{
    double lo, hi;
    get_support(s, lo, hi);
    if (!argcheck(s) || std::isnan(x)) return NaN;
    if (x >= hi) return 1.0;
    if (lo < x && x < hi) return cdf(x, s);
    return 0.0;
}

double Cont::pub_pdf(double x, const double *s) const
{
    double lo, hi;
    get_support(s, lo, hi);
    if (!argcheck(s) || std::isnan(x)) return NaN;
    if (lo <= x && x <= hi) return pdf(x, s);
    return 0.0;
}

double Cont::pub_ppf(double q, const double *s) const
{
    double lo, hi;
    get_support(s, lo, hi);
    if (!argcheck(s)) return NaN;
    if (q == 0) return lo;
    if (q == 1) return hi;
    if (0 < q && q < 1) return ppf(q, s);
    return NaN;
}

double Cont::ppf(double q, const double *s) const
{
    /* _ppf_single: bracket the root of cdf(x) - q (the public cdf), then brentq */
    const double factor = 10.0;
    double left, right;
    get_support(s, left, right);
    auto to_solve = [&](double x) { return pub_cdf(x, s) - q; };
    if (std::isinf(left)) {
        left = std::min(-factor, right);
        while (to_solve(left) > 0.0) {
            right = left;
            left = left * factor;
            if (std::isinf(left)) break;
        }
    }
    if (std::isinf(right)) {
        right = std::max(factor, left);
        while (to_solve(right) < 0.0) {
            left = right;
            right = right * factor;
            if (std::isinf(right)) break;
        }
    }
    return brentq(to_solve, left, right, xtol);
}

double Cont::munp(int n, const double *s) const
{
    if (momtype == 0) {
        double lo, hi;
        get_support(s, lo, hi);
        return quad([&](double x) { return std::pow(x, (double)n) * pub_pdf(x, s); }, lo, hi);
    }
    return quad([&](double q) { return std::pow(pub_ppf(q, s), (double)n); }, 0.0, 1.0);
}

double Cont::entropy(const double *s) const
{
    double lo, hi;
    get_support(s, lo, hi);
    auto integ = [&](double x) { return entr(pdf(x, s)); };
    const double h = quad(integ, lo, hi);
    if (!std::isnan(h)) return h;
    const double low = pub_ppf(1e-10, s), upp = pub_ppf(1.0 - 1e-10, s);
    return quad(integ, std::isinf(lo) ? low : lo, std::isinf(hi) ? upp : hi);
}

void Cont::rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const
{
    /* U = random_state.uniform(size=size); Y = self._ppf(U, *args) */
    for (int64_t i = 0; i < n; i++) out[i] = random_standard_uniform(bg);
    double s[16];
    for (int64_t i = 0; i < n; i++) {
        for (int k = 0; k < nshape; k++) s[k] = sa[k][i];
        out[i] = ppf(out[i], s);
    }
}

/* ---------------------------------------------------------------- rv_discrete generics */

double Disc::cdf(double x, const double *s) const
{
    /* _cdf: _cdfvec(floor(x)); _cdf_single: sum(_pmf(arange(int(_a), k + 1))) */
    const double k = std::floor(x);
    double lo, hi;
    get_support(s, lo, hi);
    const double start = (double)(int64_t)lo;
    if (!(k + 1 > start)) return 0.0;
    const int64_t m = alloc_count(std::ceil(k + 1 - start));
    std::vector<double> v((size_t)m);
    for (int64_t i = 0; i < m; i++) v[(size_t)i] = pmf(start + (double)i, s);
    return tsr_psum(v.data(), m);
}

double Disc::ppf(double q, const double *s) const
{
    /* _drv2_ppfsingle: bracket, then bisection on integers */
    double lo, hi;
    get_support(s, lo, hi);
    double b = hi, a = lo, qa = 0, qb = 0;
    double step = 10;
    if (std::isinf(b)) {
        b = std::max(100 * q, 10.0);
        for (;;) {
            if (b >= hi) { qb = 1.0; break; }
            qb = cdf(b, s);
            if (qb < q) { b += step; step *= 2; } else break;
        }
    } else {
        qb = 1.0;
    }
    step = 10;
    if (std::isinf(a)) {
        a = std::min(-100 * q, -10.0);
        for (;;) {
            if (a <= lo) { qb = 0.0; break; }
            qa = cdf(a, s);
            if (qa > q) { a -= step; step *= 2; } else break;
        }
    } else {
        qa = cdf(a, s);
    }
    if (std::isinf(a) || std::isinf(b)) {
        fail("Arguments that bracket the requested quantile could not be found.");
        return NaN;
    }
    for (int i = 0; i < 2046; i++) {
        if (qa == q) return a;
        if (qb == q) return b;
        if (b <= a + 1) return qa > q ? a : b;
        const double c = (double)(int64_t)((a + b) / 2.0);
        const double qc = cdf(c, s);
        if (qc < q) {
            if (a != c) a = c;
            else { fail("updating stopped, endless loop"); return NaN; }
            qa = qc;
        } else if (qc > q) {
            if (b != c) b = c;
            else { fail("updating stopped, endless loop"); return NaN; }
            qb = qc;
        } else {
            return c;
        }
    }
    return NaN;
}

/* _expect(fun, lb, ub, x0, inc): chunked summation from x0 outwards (scipy/stats/_distn_infrastructure.py) */
template <class F> static double expect_sum(F &&fun, double lb, double ub, double x0, double inc)
{
    const int maxcount = 1000, chunksize = 32;
    const double tolerance = 1e-10;
    if ((ub - lb) <= chunksize) {
        /* np.sum(fun(np.arange(lb, ub + 1, inc))) */
        const int64_t m = alloc_count(std::ceil((ub + 1 - lb) / inc));
        std::vector<double> v((size_t)m);
        for (int64_t i = 0; i < m; i++) v[(size_t)i] = fun(lb + (double)i * inc);
        return tsr_psum(v.data(), m);
    }
    if (x0 < lb) x0 = lb;
    if (x0 > ub) x0 = ub;
    int count = 0;
    double tot = 0.0;
    double buf[64];
    /* _iter_chunked(from, to, chunksize, inc): returns true when the maxcount limit ended the sum */
    auto run = [&](double from, double to, double step_inc) -> bool {
        const double stepsize = std::fabs(chunksize * step_inc);
        const double sgn = step_inc > 0 ? 1.0 : -1.0;
        double x = from;
        while ((x - to) * step_inc < 0) {
            const double delta = std::min(stepsize, std::fabs(x - to));
            const double stepv = delta * sgn;
            const int64_t m = (int64_t)std::ceil(stepv / step_inc);
            for (int64_t i = 0; i < m; i++) buf[i] = fun(x + (double)i * step_inc);
            x += stepv;
            count += (int)m;
            const double d = tsr_psum(buf, m);
            tot += d;
            if (std::fabs(d) < tolerance * (double)m) return false;
            if (count > maxcount) return true;
        }
        return false;
    };
    if (run(x0, ub + 1, inc)) return tot;
    run(x0 - 1, lb - 1, -inc);
    return tot;
}

double Disc::munp(int n, const double *s) const
{
    double lo, hi;
    get_support(s, lo, hi);
    return expect_sum([&](double x) { return std::pow(x, (double)n) * pmf(x, s); }, lo, hi, ppf(0.5, s), inc);
}

double Disc::entropy(const double *s) const
{
    double lo, hi;
    get_support(s, lo, hi);
    return expect_sum([&](double x) { return entr(pmf(x, s)); }, lo, hi, ppf(0.5, s), inc);
}

void Disc::rvs(bitgen_t *bg, int64_t n, const double *const *sa, double *out) const
{
    for (int64_t i = 0; i < n; i++) out[i] = random_standard_uniform(bg);
    double s[16];
    for (int64_t i = 0; i < n; i++) {
        for (int k = 0; k < nshape; k++) s[k] = sa[k][i];
        out[i] = ppf(out[i], s);
    }
}

/* ---------------------------------------------------------------- registry */

static std::vector<const Dist *> &dists()
{
    /* never destroyed: the registered distributions live for the whole process (and stay reachable for
       LeakSanitizer after static destructors run) */
    static std::vector<const Dist *> &v = *new std::vector<const Dist *>;
    return v;
}
static std::vector<fn_dist> &table()
{
    static std::vector<fn_dist> &t = *new std::vector<fn_dist>;   /* never destroyed, as dists() */
    return t;
}

}  // namespace tsd

extern "C" fn_dist_table TSR_STATS_DISTS;
fn_dist_table TSR_STATS_DISTS = {nullptr, 0};

void tsd::register_dist(const Dist *d)
{
    auto &v = dists();
    v.insert(std::upper_bound(v.begin(), v.end(), d, [](const Dist *x, const Dist *y) { return std::strcmp(x->name, y->name) < 0; }), d);
    auto &t = table();
    t.clear();
    for (const Dist *e : v) {
        fn_dist f;
        std::memset(&f, 0, sizeof f);
        f.name = e->name;
        f.shapes = e->shapes;
        f.nshape = e->nshape;
        f.discrete = e->discrete;
        f.doc = e->doc;
        f.a = e->a;
        f.b = e->b;
        f.extra = e;
        t.push_back(f);
    }
    TSR_STATS_DISTS.dists = t.data();
    TSR_STATS_DISTS.n = (int)t.size();
}

/* ---------------------------------------------------------------- the public methods (one element) */

namespace {

using namespace tsd;

/* rv_generic.stats for the requested moments (mask of MOM_*); out: mean, var, skew, kurtosis */
void generic_stats(const Dist *d, const double *s, int mask, double loc, double scale, double *out)
{
    Stats4 st;
    d->stats(s, mask, st);
    double mu = st.v[0], mu2 = st.v[1], g1 = st.v[2], g2 = st.v[3];
    bool hmu = st.has[0], hmu2 = st.has[1], hg1 = st.has[2], hg2 = st.has[3];
    for (int k = 0; k < 4; k++) out[k] = NaN;
    if (mask & MOM_M) {
        if (!hmu) { mu = d->munp(1, s); hmu = true; }
        out[0] = mu * scale + loc;
    }
    if (mask & MOM_V) {
        if (!hmu2) {
            const double mu2p = d->munp(2, s);
            if (!hmu) { mu = d->munp(1, s); hmu = true; }
            mu2 = !std::isinf(mu) ? mu2p - mu * mu : INF;
            hmu2 = true;
        }
        out[1] = mu2 * scale * scale;
    }
    if (mask & MOM_S) {
        if (!hg1) {
            const double mu3p = d->munp(3, s);
            if (!hmu) { mu = d->munp(1, s); hmu = true; }
            if (!hmu2) { const double mu2p = d->munp(2, s); mu2 = mu2p - mu * mu; hmu2 = true; }
            const double mu3 = (-mu * mu - 3 * mu2) * mu + mu3p;
            g1 = mu3 / std::pow(mu2, 1.5);
            hg1 = true;
        }
        out[2] = g1;
    }
    if (mask & MOM_K) {
        if (!hg2) {
            const double mu4p = d->munp(4, s);
            if (!hmu) { mu = d->munp(1, s); hmu = true; }
            if (!hmu2) { const double mu2p = d->munp(2, s); mu2 = mu2p - mu * mu; hmu2 = true; }
            double mu3;
            if (hg1) mu3 = g1 * std::pow(mu2, 1.5);
            else { const double mu3p = d->munp(3, s); mu3 = (-mu * mu - 3 * mu2) * mu + mu3p; }
            const double mu4 = ((-(mu * mu) - 6 * mu2) * mu - 4 * mu3) * mu + mu4p;
            g2 = mu4 / std::pow(mu2, 2.0) - 3.0;
        }
        out[3] = g2;
    }
}

/* _moment_from_stats */
double moment_from_stats(const Dist *d, int n, const Stats4 &st, const double *s)
{
    const double mu = st.v[0], mu2 = st.v[1], g1 = st.v[2], g2 = st.v[3];
    if (n == 0) return 1.0;
    if (n == 1) return st.has[0] ? mu : d->munp(1, s);
    if (n == 2) return (st.has[1] && st.has[0]) ? mu2 + mu * mu : d->munp(2, s);
    if (n == 3) {
        if (!st.has[2] || !st.has[1] || !st.has[0]) return d->munp(3, s);
        const double mu3 = g1 * std::pow(mu2, 1.5);
        return mu3 + 3 * mu * mu2 + mu * mu * mu;
    }
    if (n == 4) {
        if (!st.has[2] || !st.has[3] || !st.has[1] || !st.has[0]) return d->munp(4, s);
        const double mu4 = (g2 + 3.0) * std::pow(mu2, 2.0);
        const double mu3 = g1 * std::pow(mu2, 1.5);
        return mu4 + 4 * mu * mu3 + 6 * mu * mu * mu2 + mu * mu * mu * mu;
    }
    return d->munp(n, s);
}

double comb_exact(int n, int k)
{
    double r = 1;
    for (int i = 1; i <= k; i++) r = r * (n - k + i) / i;
    return std::round(r);
}

double generic_moment(const Dist *d, double order, const double *s, double loc, double scale)
{
    if (std::floor(order) != order) { fail("Moment must be an integer."); return NaN; }
    if (order < 0) { fail("Moment must be positive."); return NaN; }
    /* the private _munp(n) takes an int here; SciPy's Python int is unbounded (documented deviation) */
    if (order > 2147483647.0) { fail("Moment order too large."); return NaN; }
    const int n = (int)order;
    if (!(d->argcheck(s) && scale > 0)) return NaN;
    Stats4 st;
    if (n > 0 && n < 5) {
        static const int MASKS[5] = {0, MOM_M, MOM_V, MOM_V | MOM_S, MOM_M | MOM_V | MOM_S | MOM_K};
        d->stats(s, MASKS[n], st);
    }
    const double val = moment_from_stats(d, n, st, s);
    if (loc == 0) return std::pow(scale, (double)n) * val;
    double res2 = 0.0;
    const double fac = scale / loc;
    for (int k = 0; k < n; k++) {
        const double valk = moment_from_stats(d, k, st, s);
        res2 += comb_exact(n, k) * std::pow(fac, (double)k) * valk;
    }
    res2 += std::pow(fac, (double)n) * val;
    res2 *= std::pow(loc, (double)n);
    return res2;
}

/* the input vector: [x or q or the moment order or the stats mask] (methods that take one), shapes, loc, scale */
inline bool takes_first(int method) { return method < DM_STATS || method == DM_STATS || method == DM_MOMENT; }

void cont_method(const Cont *d, int method, const double *in, double *out)
{
    const double *s = in + (takes_first(method) ? 1 : 0);
    const double loc = s[d->nshape], scale = s[d->nshape + 1];
    const bool ok0 = d->argcheck(s) && scale > 0;
    double lo, hi;
    d->get_support(s, lo, hi);
    switch (method) {
    case DM_PDF: case DM_LOGPDF: case DM_CDF: case DM_LOGCDF: case DM_SF: case DM_LOGSF: {
        /* levy_stable_gen overrides only pdf and cdf (and rvs) to move loc; its logpdf, sf, ... are not moved */
        const double shift = (method == DM_PDF || method == DM_CDF) && ok0 ? d->public_loc_shift(s, scale) : 0.0;
        const double x = (in[0] - (loc + shift)) / scale;
        if (!ok0 || std::isnan(x)) { out[0] = NaN; return; }
        if (method == DM_PDF || method == DM_LOGPDF) {
            if (lo <= x && x <= hi) out[0] = method == DM_PDF ? d->pdf(x, s) / scale : d->logpdf(x, s) - std::log(scale);
            else out[0] = method == DM_LOGPDF ? -INF : 0.0;
            return;
        }
        const bool inside = lo < x && x < hi;
        if (method == DM_CDF || method == DM_LOGCDF) {
            if (x >= hi) out[0] = method == DM_CDF ? 1.0 : 0.0;
            else if (inside) out[0] = method == DM_CDF ? d->cdf(x, s) : d->logcdf(x, s);
            else out[0] = method == DM_CDF ? 0.0 : -INF;
            return;
        }
        if (x <= lo) out[0] = method == DM_SF ? 1.0 : 0.0;
        else if (inside) out[0] = method == DM_SF ? d->sf(x, s) : d->logsf(x, s);
        else out[0] = method == DM_SF ? 0.0 : -INF;
        return;
    }
    case DM_PPF: case DM_ISF: {
        const double q = in[0];
        out[0] = NaN;
        if (!(ok0 && loc == loc)) return;
        const double lower = lo * scale + loc, upper = hi * scale + loc;
        if (method == DM_PPF) {
            if (q == 0) out[0] = lower;
            else if (q == 1) out[0] = upper;
            else if (0 < q && q < 1) out[0] = d->ppf(q, s) * scale + loc;
        } else {
            if (q == 1) out[0] = lower;
            else if (q == 0) out[0] = upper;
            else if (0 < q && q < 1) out[0] = d->isf(q, s) * scale + loc;
        }
        return;
    }
    case DM_STATS: {
        if (!(ok0 && loc == loc)) { for (int k = 0; k < 4; k++) out[k] = NaN; return; }
        generic_stats(d, s, (int)in[0], loc, scale, out);
        return;
    }
    case DM_ENTROPY:
        out[0] = (ok0 && loc == loc) ? d->entropy(s) + std::log(scale) : NaN;
        return;
    case DM_SUPPORT:
        if (!ok0) { out[0] = out[1] = NaN; return; }
        out[0] = lo * scale + loc;
        out[1] = hi * scale + loc;
        return;
    case DM_MOMENT:
        out[0] = generic_moment(d, in[0], s, loc, scale);
        return;
    default:
        out[0] = NaN;
    }
}

inline double clip01(double p) { return std::isnan(p) ? p : std::min(std::max(p, 0.0), 1.0); }

void disc_method(const Disc *d, int method, const double *in, double *out)
{
    const double *s = in + (takes_first(method) ? 1 : 0);
    const double loc = s[d->nshape];
    const bool ok0 = d->argcheck(s);
    double lo, hi;
    d->get_support(s, lo, hi);
    switch (method) {
    case DM_PDF: case DM_LOGPDF: {
        const double k = in[0] - loc;
        if (!ok0 || std::isnan(k)) { out[0] = NaN; return; }
        if (!(k >= lo && k <= hi && std::floor(k) == k)) { out[0] = method == DM_PDF ? 0.0 : -INF; return; }
        out[0] = method == DM_PDF ? clip01(d->pmf(k, s)) : d->logpmf(k, s);
        return;
    }
    case DM_CDF: {
        const double k = in[0] - loc;
        if (!ok0 || std::isnan(k)) { out[0] = NaN; return; }
        if (k >= lo && k < hi && std::isfinite(k)) { out[0] = clip01(d->cdf(k, s)); return; }
        out[0] = k >= hi ? 1.0 : 0.0;
        return;
    }
    case DM_LOGCDF: {
        const double k = in[0] - loc;
        /* rv_discrete.logcdf places 0.0 for k >= b after the bad-argument NaN, without cond0 */
        if (!ok0 || std::isnan(k)) { out[0] = k >= hi ? 0.0 : NaN; return; }
        if (k >= lo && k < hi) { out[0] = d->logcdf(k, s); return; }
        out[0] = k >= hi ? 0.0 : -INF;
        return;
    }
    case DM_SF: {
        const double k = in[0] - loc;
        if (!ok0 || std::isnan(k)) { out[0] = NaN; return; }
        if (k >= lo && k < hi && std::isfinite(k)) { out[0] = clip01(d->sf(k, s)); return; }
        out[0] = (k < lo || (std::isinf(k) && k < 0)) ? 1.0 : 0.0;
        return;
    }
    case DM_LOGSF: {
        const double k = in[0] - loc;
        if (!ok0 || std::isnan(k)) { out[0] = NaN; return; }
        if (k >= lo && k < hi) { out[0] = d->logsf(k, s); return; }
        out[0] = k < lo ? 0.0 : -INF;
        return;
    }
    case DM_PPF: case DM_ISF: {
        const double q = in[0];
        out[0] = NaN;
        if (!(ok0 && loc == loc)) return;
        const double lower = lo - 1 + loc, upper = hi + loc;
        if (method == DM_PPF) {
            if (q == 0) out[0] = lower;
            else if (q == 1) out[0] = upper;
            else if (0 < q && q < 1) out[0] = d->ppf(q, s) + loc;
        } else {
            if (q == 1) out[0] = lower;
            else if (q == 0) out[0] = upper;
            else if (0 < q && q < 1) out[0] = d->isf(q, s) + loc;
        }
        return;
    }
    case DM_STATS: {
        if (!(ok0 && loc == loc)) { for (int k = 0; k < 4; k++) out[k] = NaN; return; }
        generic_stats(d, s, (int)in[0], loc, 1.0, out);
        return;
    }
    case DM_ENTROPY:
        out[0] = (ok0 && loc == loc) ? d->entropy(s) + std::log(1.0) : NaN;
        return;
    case DM_SUPPORT:
        if (!ok0) { out[0] = out[1] = NaN; return; }
        out[0] = lo + loc;
        out[1] = hi + loc;
        return;
    case DM_MOMENT:
        out[0] = generic_moment(d, in[0], s, loc, 1.0);
        return;
    default:
        out[0] = NaN;
    }
}

}  // namespace

const tsd::Dist *tsd::find(const char *name)
{
    for (const Dist *d : dists())
        if (std::strcmp(d->name, name) == 0) return d;
    return nullptr;
}

double tsd::call(const Dist *d, int method, double x, std::initializer_list<double> params)
{
    double in[20], out[4];
    int k = 0;
    in[k++] = x;
    for (double p : params) if (k < 20) in[k++] = p;
    if (d->discrete) disc_method((const Disc *)d, method, in, out);
    else cont_method((const Cont *)d, method, in, out);
    return out[0];
}

namespace tsd {
/* scipy.stats.poisson_binom: the number of successes in n independent Bernoulli trials with (possibly) different
 * probabilities p_i. Its shape parameter is the whole vector p, not a scalar the framework can broadcast, so the
 * distribution is built per call from p and carries its own state; nshape is 0, so disc_method's input vector is
 * just [x, loc]. The pmf is the discrete convolution (DP) f <- f * [(1-p_i), p_i], giving an O(1) pmf lookup; the
 * generic Disc methods (cdf, ppf, munp, entropy) then follow from it, matching rv_discrete. */
struct PoissonBinom : Disc {
    std::vector<double> table;   /* pmf at k = 0 .. n */
    std::vector<double> prob;    /* the probabilities (for argcheck) */
    int n;
    double mean_ = 0.0, var_ = 0.0;

    PoissonBinom(const double *p, int64_t nn) : n((int)nn)
    {
        a = 0.0;
        b = (double)nn;
        nshape = 0;
        discrete = true;
        inc = 1.0;
        prob.assign(p, p + nn);
        table.assign((size_t)nn + 1, 0.0);
        table[0] = 1.0;
        int len = 1;
        for (int i = 0; i < n; i++) {
            const double pi = p[i], qi = 1.0 - pi;
            for (int j = len; j >= 1; j--) table[(size_t)j] = table[(size_t)j] * qi + table[(size_t)(j - 1)] * pi;
            table[0] *= qi;
            len++;
            mean_ += pi;
            var_ += pi * qi;
        }
    }
    bool argcheck(const double *) const override
    {
        for (double pi : prob)
            if (!(pi >= 0.0 && pi <= 1.0)) return false;
        return true;
    }
    void get_support(const double *, double &lo, double &hi) const override { lo = 0.0; hi = (double)n; }
    double pmf(double k, const double *) const override
    {
        if (!(k >= 0.0 && k <= (double)n) || std::floor(k) != k) return 0.0;
        return table[(size_t)k];
    }
    void stats(const double *, int moments, Stats4 &st) const override
    {
        if (moments & MOM_M) st.set(0, mean_);
        if (moments & MOM_V) st.set(1, var_);
        /* skew and kurtosis fall through to generic_stats via munp, as in SciPy */
    }
};
}  // namespace tsd

extern "C" void tsr_pcg64_bitgen(uint64_t *state, bitgen_t *bg);

extern "C" {

void dist_method(const fn_dist *fd, int method, const double *in, double *out)
{
    const Dist *d = (const Dist *)fd->extra;
    /* no exception may leave C++ (see fn_guard.cpp): it becomes the call's error, as SciPy's would */
    try {
        if (d->discrete) disc_method((const Disc *)d, method, in, out);
        else cont_method((const Cont *)d, method, in, out);
    } catch (const std::bad_alloc &) {
        out[0] = NaN;
        tsd::fail("Unable to allocate memory for the result");
    } catch (const std::length_error &) {
        out[0] = NaN;
        tsd::fail("Maximum allowed size exceeded");
    } catch (const std::exception &e) {
        out[0] = NaN;
        tsd::fail(e.what());
    }
}

/* the failure SciPy would have raised during the last evaluations on this thread (NULL: none); clears it */
const char *dist_take_error(void)
{
    static thread_local char copy[256];
    if (tsd::g_failed.load() != 1) return nullptr;
    std::memcpy(copy, tsd::g_fail, sizeof copy);
    tsd::g_failed.store(0);
    return copy;
}

/* rvs: n variates; params[k][i] holds the shape parameters, loc (and scale) broadcast to n */
static int rvs_array_impl(const fn_dist *fd, void *bitgen, int64_t n, const double *const *params, double *out)
{
    const Dist *d = (const Dist *)fd->extra;
    const int ns = d->nshape;
    const double *loc = params[ns];
    const double *scale = d->discrete ? nullptr : params[ns + 1];
    double s[16];
    bool all_zero = n > 0 && scale != nullptr;
    for (int64_t i = 0; i < n; i++) {
        for (int k = 0; k < ns; k++) s[k] = params[k][i];
        const double sc = scale ? scale[i] : 1.0;
        if (!(d->argcheck(s) && sc >= 0)) {
            tsd::fail("Domain error in arguments. The `scale` parameter must be positive for all distributions, and "
                      "many distributions have restrictions on shape parameters.");
            return -1;
        }
        if (sc != 0) all_zero = false;
    }
    if (all_zero) {                            /* rvs: np.all(scale == 0) -> loc * ones(size) */
        for (int64_t i = 0; i < n; i++) out[i] = loc[i];
    } else {
        d->rvs((bitgen_t *)bitgen, n, params, out);
        for (int64_t i = 0; i < n; i++) out[i] = out[i] * (scale ? scale[i] : 1.0) + loc[i];
    }
    if (!d->discrete) {                        /* a distribution's own public rvs shifting the result (levy_stable) */
        const Cont *c = (const Cont *)d;
        for (int64_t i = 0; i < n; i++) {
            for (int k = 0; k < ns; k++) s[k] = params[k][i];
            out[i] += c->public_loc_shift(s, scale[i]);
        }
    }
    if (d->discrete)
        for (int64_t i = 0; i < n; i++) out[i] = (double)(int64_t)out[i];   /* vals.astype(np.int64) */
    return 0;
}

int dist_rvs_array(const fn_dist *fd, void *bitgen, int64_t n, const double *const *params, double *out)
{
    try {
        return rvs_array_impl(fd, bitgen, n, params, out);
    } catch (const std::bad_alloc &) {
        tsd::fail("Unable to allocate memory for the result");
    } catch (const std::length_error &) {
        tsd::fail("Maximum allowed size exceeded");
    } catch (const std::exception &e) {
        tsd::fail(e.what());
    }
    return -1;
}

/* poisson_binom value methods: build the distribution from p, then drive the same disc_method the registered
 * discrete distributions use (nshape 0, so the input vector is [first, loc]). Element methods (DM_PDF..DM_ISF,
 * DM_MOMENT) write one output per x; DM_STATS writes 4, DM_SUPPORT 2, DM_ENTROPY 1. Returns -1 on failure. */
__attribute__((visibility("default")))
int tsr_poisson_binom(int method, const double *xv, int64_t nx, const double *p, int64_t n, double loc, double *out)
{
    if (n < 0 || !p || !out) return -1;
    dist_take_error();   /* clear any stale per-thread failure left by earlier work so it is not misattributed here */
    try {
        tsd::PoissonBinom pb(p, n);
        if (method == DM_STATS) {
            double in[2] = {xv && nx > 0 ? xv[0] : 0.0, loc};
            disc_method(&pb, DM_STATS, in, out);
        } else if (method == DM_SUPPORT) {
            double in[1] = {loc};
            disc_method(&pb, DM_SUPPORT, in, out);
        } else if (method == DM_ENTROPY) {
            double in[1] = {loc};
            disc_method(&pb, DM_ENTROPY, in, out);
        } else {
            for (int64_t i = 0; i < nx; i++) {
                double in[2] = {xv[i], loc};
                disc_method(&pb, method, in, &out[i]);
            }
        }
    } catch (const std::exception &e) {
        tsd::fail(e.what());
        return -1;
    }
    return dist_take_error() ? -1 : 0;
}

/* poisson_binom random variates: SciPy's poisson_binom._rvs draws, for each variate, one Bernoulli(p_j) per
 * probability (numpy's binomial(1, p_j)) along the trailing axis and sums them. The output array is filled in C
 * order, so for each output element the n Bernoulli draws come in order j = 0 .. n-1, matching numpy's fill of the
 * (size, n) Bernoulli array. Cast to int64 like the other discrete distributions. */
__attribute__((visibility("default")))
int tsr_poisson_binom_rvs(uint64_t *state, const double *p, int64_t n, double loc, int64_t nsamp, double *out)
{
    if (n < 0 || !p || !state || !out) return -1;
    dist_take_error();
    try {
        bitgen_t bg;
        tsr_pcg64_bitgen(state, &bg);
        binomial_t bs;
        std::memset(&bs, 0, sizeof bs);
        for (int64_t i = 0; i < nsamp; i++) {
            int64_t s = 0;
            for (int64_t j = 0; j < n; j++) s += tsd::np_checked_binomial(&bg, p[j], 1, &bs);
            out[i] = (double)(int64_t)((double)s + loc);
        }
    } catch (const std::exception &e) {
        tsd::fail(e.what());
        return -1;
    }
    return dist_take_error() ? -1 : 0;
}

}  // extern "C"
