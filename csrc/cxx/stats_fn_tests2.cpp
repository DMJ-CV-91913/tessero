/*
 * scipy.stats hypothesis tests (group tests2), SciPy 1.17.1: shapiro, anderson, anderson_ksamp, cramervonmises,
 * cramervonmises_2samp, ks_1samp, kstest, ks_2samp, epps_singleton_2samp, alexandergovern, page_trend_test,
 * poisson_means_test, combine_pvalues, bws_test, quantile_test.
 *
 * Each function follows SciPy's code step by step (scipy/stats/_morestats.py, _hypotests.py, _stats_py.py,
 * _ksstats.py, _stats_pythran.py, _page_trend_test.py, _bws_test.py, _ansari_swilk_statistics.pyx) with NumPy's
 * summation orders. The functions SciPy decorates with _axis_nan_policy_factory are routines here that run the
 * same machinery (run_test below): axis=None ravels, other axes broadcast and move to the end, nan_policy
 * propagate/omit/raise, the too-small rule, keepdims, and the dtype rules of the vectorized and the
 * apply_along_axis paths.
 */
#include "stats_dist.hpp"

extern "C" {
#include "../src/fn.h"
}

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace {

using std::vector;
using std::string;

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = 2.220446049250313e-16;
constexpr double PI = 3.141592653589793;

/* ---------------------------------------------------------------- errors and arguments */

int fail(const char *msg)
{
    fn_set_error("%s", msg);
    return TSR_EARG;
}

struct NDA {
    vector<double> v;
    vector<int64_t> shape;
    bool isint = false;
    int64_t size() const { int64_t s = 1; for (auto d : shape) s *= d; return s; }
};

bool is_int_dtype(int dt) { return dt == TSR_I64 || dt == TSR_I32 || dt == TSR_U8 || dt == TSR_BOOL; }

bool has(const tsr_arg *a, int n, int k) { return k < n && a[k].kind != 0; }

/* an array argument (a number is a 0-d array); false when absent or not numeric */
bool get_nda(const tsr_arg *a, int n, int k, NDA &out)
{
    out.v.clear();
    out.shape.clear();
    if (k >= n) return false;
    if (a[k].kind == 1 || a[k].kind == 4) {
        out.v.push_back(a[k].num);
        out.isint = a[k].kind == 4;
        return true;
    }
    if (a[k].kind != 3 || a[k].arr.dtype == TSR_C128) return false;
    int64_t m = 0;
    double *p = fn_arg_doubles(&a[k], &m);
    if (!p) return false;
    out.v.assign(p, p + m);
    fn_free_doubles(p, m);
    out.shape.assign(a[k].arr.shape, a[k].arr.shape + a[k].arr.ndim);
    out.isint = is_int_dtype(a[k].arr.dtype);
    return true;
}

string get_str(const tsr_arg *a, int n, int k, const char *def)
{
    if (k >= n || a[k].kind != 2 || !a[k].str) return def;
    return a[k].str;
}

double get_num(const tsr_arg *a, int n, int k, double def)
{
    if (k >= n || (a[k].kind != 1 && a[k].kind != 4)) return def;
    return a[k].num;
}

bool get_bool(const tsr_arg *a, int n, int k, bool def)
{
    if (k >= n || (a[k].kind != 1 && a[k].kind != 4)) return def;
    return a[k].num != 0;
}

/* ---------------------------------------------------------------- NumPy arithmetic helpers */

double psum(const double *x, int64_t n) { return n <= 0 ? 0.0 : tsr_psum(x, n); }
double psum(const vector<double> &x) { return psum(x.data(), (int64_t)x.size()); }
double seqsum(const double *x, int64_t n)
{
    if (n <= 0) return 0.0;
    double s = x[0];
    for (int64_t i = 1; i < n; i++) s += x[i];
    return s;
}
/* np.sum along the core: pairwise unless NumPy reduces the axis in the outer loop */
double asum(const vector<double> &x, unsigned flags)
{
    return (flags & FN_SEQUENTIAL) ? seqsum(x.data(), (int64_t)x.size()) : psum(x);
}

/* numpy.floor_divide / remainder for float64 (npy_divmod) */
double np_divmod(double a, double b, double *modp)
{
    double mod = std::fmod(a, b);
    if (b == 0) { if (modp) *modp = mod; return a / b; }
    double div = (a - mod) / b;
    if (mod) {
        if ((b < 0) != (mod < 0)) { mod += b; div -= 1.0; }
    } else {
        mod = std::copysign(0.0, b);
    }
    double floordiv;
    if (div) {
        floordiv = std::floor(div);
        if (div - floordiv > 0.5) floordiv += 1.0;
    } else {
        floordiv = std::copysign(0.0, a / b);
    }
    if (modp) *modp = mod;
    return floordiv;
}
double np_floordiv(double a, double b) { return np_divmod(a, b, nullptr); }
double np_mod(double a, double b) { double m; np_divmod(a, b, &m); return m; }

/* np.round / np.around with decimals >= 0: x * 10**d, rint, / 10**d */
double np_around(double x, int d)
{
    const double f = std::pow(10.0, d);
    return std::nearbyint(x * f) / f;
}

/* Python floor division of integers */
int64_t py_floordiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q--;
    return q;
}

double clip01(double p) { return p != p ? p : (p < 0 ? 0.0 : (p > 1 ? 1.0 : p)); }

void sort_nan_last(vector<double> &v)
{
    std::sort(v.begin(), v.end(), [](double a, double b) { return a < b || (b != b && a == a); });
}

/* np.searchsorted on a sorted (NaN-free) array */
int64_t ss_left(const vector<double> &a, double x) { return std::lower_bound(a.begin(), a.end(), x) - a.begin(); }
int64_t ss_right(const vector<double> &a, double x) { return std::upper_bound(a.begin(), a.end(), x) - a.begin(); }

/* np.argmax / argmin: the first extremum; NaN wins (first NaN) */
int64_t np_argmax(const vector<double> &v)
{
    int64_t b = 0;
    for (int64_t i = 0; i < (int64_t)v.size(); i++) {
        if (v[i] != v[i]) return i;
        if (v[i] > v[b]) b = i;
    }
    return b;
}
int64_t np_argmin(const vector<double> &v)
{
    int64_t b = 0;
    for (int64_t i = 0; i < (int64_t)v.size(); i++) {
        if (v[i] != v[i]) return i;
        if (v[i] < v[b]) b = i;
    }
    return b;
}

/* scipy.stats.rankdata(method='average') of a NaN-free vector */
vector<double> rank_average(const vector<double> &x)
{
    const int64_t n = (int64_t)x.size();
    vector<int64_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) { return x[a] < x[b]; });
    vector<double> r(n);
    int64_t i = 0;
    while (i < n) {
        int64_t j = i + 1;
        while (j < n && x[idx[j]] == x[idx[i]]) j++;
        const double rank = 0.5 * (double)(j + i + 1);
        for (int64_t k = i; k < j; k++) r[idx[k]] = rank;
        i = j;
    }
    return r;
}

/* scipy.optimize.brentq (Zeros/brentq.c); *conv = false when the iteration limit is reached */
template <class F> double brentq(F &&f, double xa, double xb, double xtol, double rtol, int iter, int *status)
{
    double xpre = xa, xcur = xb;
    double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis;
    double delta, stry, dpre, dblk;
    fpre = f(xpre);
    fcur = f(xcur);
    if (status) *status = 0;
    if (fpre == 0) return xpre;
    if (fcur == 0) return xcur;
    if (std::signbit(fpre) == std::signbit(fcur)) { if (status) *status = -1; return 0.; }
    for (int i = 0; i < iter; i++) {
        if (fpre != 0 && fcur != 0 && (std::signbit(fpre) != std::signbit(fcur))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (std::fabs(fblk) < std::fabs(fcur)) {
            xpre = xcur; xcur = xblk; xblk = xpre;
            fpre = fcur; fcur = fblk; fblk = fpre;
        }
        delta = (xtol + rtol * std::fabs(xcur)) / 2;
        sbis = (xblk - xcur) / 2;
        if (fcur == 0 || std::fabs(sbis) < delta) return xcur;
        if (std::fabs(spre) > delta && std::fabs(fcur) < std::fabs(fpre)) {
            if (xpre == xblk) {
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                dpre = (fpre - fcur) / (xpre - xcur);
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            if (2 * std::fabs(stry) < std::min(std::fabs(spre), 3 * std::fabs(sbis) - delta)) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }
        xpre = xcur;
        fpre = fcur;
        if (std::fabs(scur) > delta) xcur += scur;
        else xcur += (sbis > 0 ? delta : -delta);
        fcur = f(xcur);
    }
    if (status) *status = 1;
    return xcur;
}

/* ---------------------------------------------------------------- distributions by name */

/* dist.cdf(x, *args) for a registered scipy.stats distribution; args are shapes, then loc, scale */
struct NamedCdf {
    const tsd::Dist *d = nullptr;
    vector<double> params;
    int init(const string &name, const vector<double> &args)
    {
        static const char *const OK[] = {"norm", "expon", "uniform", "gamma", "beta", "t", "chi2", "f"};
        bool known = false;
        for (auto s : OK) if (name == s) known = true;
        d = known ? tsd::find(name.c_str()) : nullptr;
        if (!d) {
            fn_set_error("cdf: distribution '%s' is not available (norm, expon, uniform, gamma, beta, t, chi2, f)", name.c_str());
            return TSR_EARG;
        }
        const int ns = d->nshape;
        if ((int)args.size() < ns) return fail("cdf() missing required shape arguments");
        if ((int)args.size() > ns + 2) return fail("cdf() got too many positional arguments");
        params = args;
        if ((int)params.size() < ns + 1) params.push_back(0.0);
        if ((int)params.size() < ns + 2) params.push_back(1.0);
        return TSR_OK;
    }
    double cdf(double x) const
    {
        switch (params.size()) {
        case 2: return tsd::call(d, DM_CDF, x, {params[0], params[1]});
        case 3: return tsd::call(d, DM_CDF, x, {params[0], params[1], params[2]});
        case 4: return tsd::call(d, DM_CDF, x, {params[0], params[1], params[2], params[3]});
        default: return tsd::call(d, DM_CDF, x, {params[0], params[1], params[2], params[3], params[4]});
        }
    }
};

/* ---------------------------------------------------------------- the _axis_nan_policy machinery */

enum { NP_PROPAGATE, NP_OMIT, NP_RAISE };

int parse_nan_policy(const tsr_arg *a, int n, int k, int *out)
{
    const string s = get_str(a, n, k, "propagate");
    if (s == "propagate") *out = NP_PROPAGATE;
    else if (s == "omit") *out = NP_OMIT;
    else if (s == "raise") *out = NP_RAISE;
    else return fail("nan_policy must be one of {'propagate', 'raise', 'omit'}");
    return TSR_OK;
}

typedef std::function<int(vector<vector<double>> &x, double *out, unsigned flags)> CoreFn;
/* core flag: the function runs vectorized on an N-d array (xp.stack/xpx.cov take their N-d branches) */
constexpr unsigned FN_VECND = 2u;

struct Test {
    int nout = 2;
    bool vectorized = false;
    bool paired = false;
    int too_small = 0;                          /* any sample with n <= too_small gives NaN */
    std::function<bool(const vector<int64_t> &)> too_small_fn;   /* overrides too_small */
    unsigned outint = 0;                        /* outputs that are integers when the function returns them */
    CoreFn core;
    bool small(const vector<int64_t> &n) const
    {
        if (too_small_fn) return too_small_fn(n);
        for (auto m : n) if (m <= too_small) return true;
        return false;
    }
};

struct Sample {
    vector<double> v;                           /* C-order data of the (broadcast-free) original array */
    vector<int64_t> shape;                      /* padded to nd */
};

/* results: nout outputs, scalars when shape is empty (and not keepdims) */
int emit(tsr_result *res, int nout, const vector<int64_t> &shape, bool scalar, const vector<vector<double>> &vals,
         unsigned intmask)
{
    for (int o = 0; o < nout; o++) {
        const bool isint = (intmask >> o) & 1u;
        if (scalar) {
            if (isint) fn_result_int(&res[o], (int64_t)vals[o][0]);
            else fn_result_num(&res[o], vals[o][0]);
            continue;
        }
        void *p = fn_result_array(&res[o], isint ? TSR_I64 : TSR_F64, (int32_t)shape.size(), shape.data());
        if (!p) return TSR_ENOMEM;
        for (size_t i = 0; i < vals[o].size(); i++) {
            if (isint) ((int64_t *)p)[i] = (int64_t)vals[o][i];
            else ((double *)p)[i] = vals[o][i];
        }
    }
    return TSR_OK;
}

bool any_nan(const vector<double> &v)
{
    for (double x : v) if (x != x) return true;
    return false;
}

void remove_nans(vector<vector<double>> &s, bool paired)
{
    if (!paired) {
        for (auto &v : s) v.erase(std::remove_if(v.begin(), v.end(), [](double x) { return x != x; }), v.end());
        return;
    }
    const size_t n = s[0].size();
    vector<char> keep(n, 1);
    for (auto &v : s) for (size_t i = 0; i < n; i++) if (v[i] != v[i]) keep[i] = 0;
    for (auto &v : s) {
        size_t k = 0;
        for (size_t i = 0; i < n; i++) if (keep[i]) v[k++] = v[i];
        v.resize(k);
    }
}

/*
 * Run a decorated hypothesis test. samples: the sample arrays; axis: tsr_arg (null = None, number = int);
 * results go to res[0..nout).
 */
int run_test(const Test &T, vector<NDA> samples, const tsr_arg *axis_arg, int nanpol, bool keepdims, tsr_result *res)
{
    const int ns = (int)samples.size();
    const int nout = T.nout;
    bool axis_none = !axis_arg || axis_arg->kind == 0;
    int64_t axis = 0;
    if (!axis_none) {
        if (axis_arg->kind != 1 || axis_arg->num != std::floor(axis_arg->num))
            return fail("`axis` must be an integer, a tuple of integers, or `None`.");
        axis = (int64_t)std::fmax(-1e9, std::fmin(1e9, axis_arg->num));   /* no UB for huge values */
    }
    for (auto &s : samples) if (s.shape.empty()) s.shape.push_back(1);   /* atleast_1d */
    int nd = 0;
    for (auto &s : samples) nd = std::max(nd, (int)s.shape.size());
    auto nanres = [&](vector<vector<double>> &vals, size_t count) {
        vals.assign(nout, vector<double>(count, NaN));
    };

    /* ---- 1-D path (axis=None ravels) */
    vector<vector<double>> x1;
    bool one_d = axis_none;
    if (!axis_none && nd <= 1) {
        const int64_t ax = axis < 0 ? axis + nd : axis;
        if (ax < 0 || ax >= nd) {
            fn_set_error("AxisError: `axis` is out of bounds for array of dimension %d", nd);
            return TSR_EARG;
        }
        if (T.paired) {
            int64_t n0 = -1;
            for (auto &s : samples) {
                const int64_t m = s.shape[0];
                if (m != 1 && n0 != -1 && n0 != 1 && m != n0) return fail("Array shapes are incompatible for broadcasting.");
                if (n0 == -1 || n0 == 1) n0 = m;
            }
            for (auto &s : samples) if (s.shape[0] != n0) { s.v.assign(n0, s.v[0]); s.shape[0] = n0; }
        }
        one_d = true;
    }
    if (one_d) {
        if (axis_none && T.paired) {
            /* paired samples broadcast before ravelling (weights with pvalues) */
            int64_t n0 = -1;
            for (auto &s : samples) {
                const int64_t m = s.size();
                if (n0 == -1 || n0 == 1) n0 = m;
                else if (m != 1 && m != n0) return fail("Array shapes are incompatible for broadcasting.");
            }
            for (auto &s : samples) if (s.size() != n0) s.v.assign(n0, s.v[0]);
        }
        for (auto &s : samples) x1.push_back(s.v);
        bool anynan = false;
        for (auto &v : x1) anynan = anynan || any_nan(v);
        if (anynan && nanpol == NP_RAISE) return fail("The input contains nan values");
        vector<vector<double>> vals;
        vector<int64_t> kshape;
        bool scalar = !keepdims;
        if (keepdims) kshape.assign(axis_none ? nd : 1, 1);
        if (anynan && nanpol == NP_PROPAGATE) {
            nanres(vals, 1);
            return emit(res, nout, kshape, scalar, vals, 0);
        }
        if (anynan && nanpol == NP_OMIT) remove_nans(x1, T.paired);
        vector<int64_t> lens;
        for (auto &v : x1) lens.push_back((int64_t)v.size());
        if (T.small(lens)) {
            nanres(vals, 1);
            return emit(res, nout, kshape, scalar, vals, 0);
        }
        vector<double> out(nout, NaN);
        int rc = T.core(x1, out.data(), 0);
        if (rc != TSR_OK) return rc;
        vals.assign(nout, vector<double>(1));
        for (int o = 0; o < nout; o++) vals[o][0] = out[o];
        return emit(res, nout, kshape, scalar, vals, T.outint);
    }

    /* ---- N-D path: broadcast the non-axis dimensions, move the axis to the end */
    for (auto &s : samples) {
        vector<int64_t> sh(nd, 1);
        std::copy(s.shape.begin(), s.shape.end(), sh.begin() + (nd - (int)s.shape.size()));
        s.shape = sh;
    }
    const int64_t ax = axis < 0 ? axis + nd : axis;
    if (ax < 0 || ax >= nd) {
        fn_set_error("AxisError: `axis` is out of bounds for array of dimension %d", nd);
        return TSR_EARG;
    }
    vector<int64_t> outer;                      /* broadcast shape without the axis */
    vector<int64_t> tgt(nd, 1);
    for (int d = 0; d < nd; d++) {
        if (d == ax && !T.paired) continue;
        int64_t m = 1;
        bool zero = false;
        for (auto &s : samples) {
            if (s.shape[d] == 0) zero = true;
            m = std::max(m, s.shape[d]);
        }
        if (zero) m = 0;
        for (auto &s : samples)
            if (s.shape[d] != 1 && s.shape[d] != m) return fail("Array shapes are incompatible for broadcasting.");
        tgt[d] = m;
        if (d != ax) outer.push_back(m);
    }
    int64_t nouter = 1;
    for (auto m : outer) nouter *= m;
    /* strides (elements) of each sample, 0 on broadcast dims */
    const int ns_ = ns;
    vector<vector<int64_t>> strides(ns_, vector<int64_t>(nd, 0));
    vector<int64_t> lens(ns_);
    for (int k = 0; k < ns_; k++) {
        int64_t st = 1;
        for (int d = nd - 1; d >= 0; d--) {
            const int64_t m = samples[k].shape[d];
            const bool bc = m == 1 && (d != ax || T.paired) && tgt[d] != 1;
            strides[k][d] = bc ? 0 : st;
            st *= m;
        }
        lens[k] = (T.paired) ? tgt[ax] : samples[k].shape[ax];
    }
    vector<int64_t> kshape = outer;
    if (keepdims) kshape.insert(kshape.begin() + ax, 1);
    vector<vector<double>> vals;
    bool anyempty = false;
    for (auto &s : samples) if (s.size() == 0) anyempty = true;
    if (anyempty && (T.small(lens) || nouter == 0)) {
        nanres(vals, (size_t)nouter);
        return emit(res, nout, kshape, false, vals, 0);
    }
    /* gather one slice */
    auto gather = [&](int64_t o, vector<vector<double>> &x) {
        x.assign(ns_, vector<double>());
        vector<int64_t> idx(outer.size());
        int64_t r = o;
        for (int d = (int)outer.size() - 1; d >= 0; d--) { idx[d] = outer[d] ? r % outer[d] : 0; if (outer[d]) r /= outer[d]; }
        for (int k = 0; k < ns_; k++) {
            int64_t base = 0;
            int j = 0;
            for (int d = 0; d < nd; d++) {
                if (d == ax) continue;
                base += idx[j++] * strides[k][d];
            }
            x[k].resize(lens[k]);
            for (int64_t i = 0; i < lens[k]; i++) x[k][i] = samples[k].v[base + i * strides[k][ax]];
        }
    };
    bool contains_nan = false;
    for (auto &s : samples) contains_nan = contains_nan || any_nan(s.v);
    if (contains_nan && nanpol == NP_RAISE) return fail("The input contains nan values");
    vals.assign(nout, vector<double>((size_t)nouter, NaN));
    vector<vector<double>> x;
    vector<double> out(nout);
    if (T.vectorized && !contains_nan) {
        /* the function runs on the whole array: NumPy reduces the moved axis sequentially when a later
           dimension of size > 1 is innermost in memory */
        unsigned flags = FN_VECND;
        for (int d = (int)ax + 1; d < nd; d++) if (d != ax && outer[d - 1] > 1) flags |= FN_SEQUENTIAL;
        for (int64_t o = 0; o < nouter; o++) {
            gather(o, x);
            std::fill(out.begin(), out.end(), NaN);
            int rc = T.core(x, out.data(), flags);
            if (rc != TSR_OK) return rc;
            for (int q = 0; q < nout; q++) vals[q][o] = out[q];
        }
        return emit(res, nout, kshape, false, vals, T.outint);
    }
    for (int64_t o = 0; o < nouter; o++) {
        gather(o, x);
        if (contains_nan && nanpol == NP_OMIT) {
            remove_nans(x, T.paired);
        } else if (contains_nan && nanpol == NP_PROPAGATE) {
            bool sn = false;
            for (auto &v : x) sn = sn || any_nan(v);
            if (sn) continue;
        }
        vector<int64_t> l;
        for (auto &v : x) l.push_back((int64_t)v.size());
        if (T.small(l)) continue;
        std::fill(out.begin(), out.end(), NaN);
        int rc = T.core(x, out.data(), 0);
        if (rc != TSR_OK) return rc;
        for (int q = 0; q < nout; q++) vals[q][o] = out[q];
    }
    return emit(res, nout, kshape, false, vals, 0);
}

/* the arguments axis, nan_policy, keepdims at positions k, k+1, k+2 */
struct AxisArgs {
    const tsr_arg *axis = nullptr;
    int nanpol = NP_PROPAGATE;
    bool keepdims = false;
};
int axis_args(const tsr_arg *a, int n, int kaxis, int knan, int kkeep, const char *axis_default, AxisArgs &out)
{
    static tsr_arg zero_axis, none_axis;
    zero_axis.kind = 1;
    zero_axis.num = 0;
    none_axis.kind = 0;
    out.axis = (kaxis < n) ? &a[kaxis] : (std::strcmp(axis_default, "None") == 0 ? &none_axis : &zero_axis);
    if (knan >= 0) {
        int rc = parse_nan_policy(a, n, knan, &out.nanpol);
        if (rc != TSR_OK) return rc;
    }
    out.keepdims = kkeep >= 0 ? get_bool(a, n, kkeep, false) : false;
    return TSR_OK;
}

/* ================================================================ MINPACK hybrd, for fsolve
 * Copied from SciPy 1.17.1 scipy/optimize/__minpack.c (the C translation of MINPACK; MINPACK is
 * Copyright (c) 1999, University of Chicago, see SciPy's LICENSES_bundled; SciPy is BSD-3-Clause), comments removed.
 */

const double dpmpar[3] = {2.220446049250313e-16, 2.2250738585072014e-308, 1.7976931348623157e+308};

double enorm(const int n, const double* x)
{
    int i;
    double agiant, s1, s2, s3, xabs, x1max, x3max;

    const double rgiant = pow(2.0, 63.5);
    const double rdwarf = 0.5 / rgiant;

    s1 = 0.0;
    s2 = 0.0;
    s3 = 0.0;
    x1max = 0.0;
    x3max = 0.0;
    agiant = rgiant / n;

    for (i = 0; i < n; i++)
    {
        xabs = fabs(x[i]);
        if ((xabs <= rdwarf) || (xabs >= agiant))
        {
            if (xabs > rdwarf)
            {
                if (xabs > x1max)
                {
                    s1 = 1.0 + s1*pow(x1max / xabs, 2.0);
                    x1max = xabs;
                } else {
                    s1 += pow(xabs / x1max, 2.0);
                }
            } else {
                if (xabs > x3max)
                {
                    s3 = 1.0 + s3*pow(x3max / xabs, 2.0);
                    x3max = xabs;
                } else {
                    if (xabs != 0.0)
                    {
                        s3 += pow(xabs / x3max, 2.0);
                    }
                }
            }
        } else {
            s2 += xabs*xabs;
        }
    }
    if (s1 != 0.0)
    {
        return x1max*sqrt(s1+(s2/x1max)/x1max);
    }
    if (s2 != 0.0)
    {
        if (s2 >= x3max)
        {
            return sqrt(s2*(1.0 + (x3max/s2)*(x3max*s3)));
        } else {
            return sqrt(x3max*((s2/x3max)+(x3max*s3)));
        }
    }

    return x3max*sqrt(s3);
}

void qrfac(const int m, const int n, double* a, const int lda, const int pivot,
           int* ipvt, double* rdiag, double* acnorm, double* wa)
{
    int i, j, k, kmax, minmn;
    double ajnorm, epsmch, ssum, temp;

    epsmch = dpmpar[0];
    for (j = 0; j < n; j++)
    {
        acnorm[j] = enorm(m, &a[j*lda]);
        rdiag[j] = acnorm[j];
        wa[j] = rdiag[j];
        if (pivot) { ipvt[j] = j; }
    }
    minmn = (m > n ? n : m);
    for (j = 0; j < minmn; j++)
    {
        if (pivot)
        {
            kmax = j;
            for (k = j; k < n; k++)
            {
                if (rdiag[k] > rdiag[kmax]) { kmax = k; }
            }
            if (kmax != j)
            {
                for (i = 0; i < m; i++)
                {
                    temp = a[i + lda*j];
                    a[i + lda*j] = a[i + lda*kmax];
                    a[i + lda*kmax] = temp;
                }
                rdiag[kmax] = rdiag[j];
                wa[kmax] = wa[j];
                k = ipvt[j];
                ipvt[j] = ipvt[kmax];
                ipvt[kmax] = k;
            }
        }
        ajnorm = enorm(m - j, &a[lda*j + j]);
        if (ajnorm == 0.0)
        {
            rdiag[j] = -ajnorm;
            continue;
        }

        if (a[lda*j + j] < 0.0) { ajnorm = -ajnorm; }
        for (i = j; i < m; i++)
        {
            a[i + lda*j] /= ajnorm;
        }

        a[j + lda*j] += 1.0;
        if (j == n - 1)
        {
            rdiag[j] = -ajnorm;
            continue;
        }

        for (k = j+1; k < n; k++)
        {
            ssum = 0.0;
            for (i = j; i < m; i++)
            {
                ssum += a[i + lda*j]*a[i + lda*k];
            }
            temp = ssum / a[j + lda*j];
            for (i = j; i < m; i++)
            {
                a[i + lda*k] -= temp*a[i + lda*j];
            }
            if ((pivot) && (rdiag[k] != 0.0))
            {
                temp = a[j + lda*k] / rdiag[k];
                rdiag[k] *= sqrt(fmax(0.0, 1.0 - temp*temp));
                if (0.05*pow(rdiag[k]/wa[k], 2.0) <= epsmch)
                {
                    rdiag[k] = enorm(m-j, &a[(j+1) + lda*k]);
                    wa[k] = rdiag[k];
                }
            }
        }
        rdiag[j] = -ajnorm;
    }

    return;
}

void qform(const int m, const int n, double* q, const int ldq, double* wa)
{
    int i, j, k, l, minmn, np1;
    double ssum, temp;
    minmn = (m > n ? n : m);
    if (minmn > 1)
    {
        for (j = 1; j < minmn; j++)
        {
            for (i = 0; i < j; i++)
            {
                q[i + j*ldq] = 0.0;
            }
        }
    }
    np1 = n + 1;
    if (m >= np1)
    {
        for (j = n; j < m; j++)
        {
            for (i = 0; i < m; i++)
            {
                q[i + j*ldq] = 0.0;
            }
            q[j + j*ldq] = 1.0;
        }
    }
    for (l = 0; l < minmn; l++)
    {
        k = minmn - l - 1;
        for (i = k; i < m; i++)
        {
            wa[i] = q[i + ldq*k];
            q[i + ldq*k] = 0.0;
        }
        q[k + ldq*k] = 1.0;
        if (wa[k] != 0.0)
        {
            for (j = k; j < m; j++)
            {
                ssum = 0.0;
                for (i = k; i < m; i++)
                {
                    ssum += q[i + ldq*j]*wa[i];
                }
                temp = ssum / wa[k];
                for (i = k; i < m; i++)
                {
                    q[i + ldq*j] -= temp*wa[i];
                }
            }
        }
    }

    return;
}

void r1mpyq(const int m, const int n, double* a, const int lda, const double* v,
            const double* w)
{
    int i, j, k;
    double ccos, ssin, temp;

    if (n < 2) { return; }
    for (k = 0; k < (n-1); k++) {
	    j = n - k - 2;
        if (fabs(v[j]) > 1.0) {
            ccos = 1 / v[j];
            ssin = sqrt(1.0 - ccos*ccos);
        } else {
            ssin = v[j];
            ccos = sqrt(1.0 - ssin*ssin);
        }

        for (i = 0; i < m; i++) {
            temp = ccos * a[i + j*lda] - ssin*a[i + (n-1)*lda];
            a[i + (n-1)*lda] = ssin*a[i + j*lda] + ccos*a[i + (n-1)*lda];
            a[i + j*lda] = temp;
        }
    }
    for (j = 0; j < (n-1); j++) {
        if (fabs(w[j]) > 1.) {
            ccos = 1 / w[j];
            ssin = sqrt(1 - ccos*ccos);
        } else {
            ssin = w[j];
            ccos = sqrt(1 - ssin*ssin);
        }

        for (i = 0; i < m; i++) {
            temp = ccos*a[i + lda*j] + ssin*a[i + (n-1)*lda];
            a[i + (n-1)*lda] = -ssin*a[i + j*lda] + ccos*a[i + (n-1)*lda];
            a[i + lda*j] = temp;
        }
    }

    return;
}

void r1updt(const int m, const int n, double* s, const double* u, double* v,
            double* w, int* sing)
{
    int i, j, jj, l, nmj, nm1;
    double ccos, ccotan, giant, ssin, ttan, tau, temp;
    giant = dpmpar[2];
    jj = (n*(2*m - n + 1))/2 - (m - n);
    l = jj;
    for (i = n; i <= m; i++) {
        w[i - 1] = s[l - 1];
        l++;
    }
    nm1 = n - 1;
    if (nm1 >= 1)
    {
        for (nmj = 1; nmj <= nm1; nmj++) {
            j = n - nmj;
            jj -= m - j + 1;
            w[j-1] = 0.0;
            if (v[j-1] != 0.0)
            {
                if (fabs(v[n-1]) < fabs(v[j-1]))
                {
                    ccotan = v[n-1]/v[j-1];
                    ssin = 0.5 / sqrt(0.25 + 0.25*(ccotan*ccotan));
                    ccos = ssin*ccotan;
                    tau = 1.0;
                    if (fabs(ccos)*giant > 1.0) { tau = 1.0/ccos; }
                } else {
                    ttan = v[j-1] / v[n-1];
                    ccos = 0.5 / sqrt(0.25 + 0.25*(ttan*ttan));
                    ssin = ccos*ttan;
                    tau = ssin;
                }
                v[n-1] = ssin*v[j-1] + ccos*v[n-1];
                v[j-1] = tau;
                l = jj;
                for (i = j; i <= m; i++) {
                    temp = ccos * s[l-1] - ssin * w[i-1];
                    w[i-1] = ssin * s[l-1] + ccos * w[i-1];
                    s[l-1] = temp;
                    l++;
                }
            }
        }
    }
    for (i = 0; i < m; i++)
    {
	    w[i] += v[n-1] * u[i];
    }
    *sing = 0;
    if (nm1 >= 1) {
        for (j = 1; j <= nm1; j++) {
            if (w[j-1] != 0.) {

                if (fabs(s[jj-1]) < fabs(w[j-1])) {
                    ccotan = s[jj-1] / w[j-1];
                    ssin = 0.5 / sqrt(0.25 + 0.25*pow(ccotan, 2.0));
                    ccos = ssin * ccotan;
                    tau = 1.0;
                    if (fabs(ccos)*giant > 1.0) { tau = 1 / ccos; }
                } else {
                    ttan = w[j-1] / s[jj-1];
                    ccos = 0.5 / sqrt(0.25 + 0.25*pow(ttan, 2.0));
                    ssin = ccos * ttan;
                    tau = ssin;
                }
                l = jj;
                for (i = j; i <= m; i++) {
                    temp = ccos*s[l-1] + ssin*w[i-1];
                    w[i-1] = -ssin*s[l-1] + ccos*w[i-1];
                    s[l-1] = temp;
                    l++;
                }
                w[j-1] = tau;
            }
            if (s[jj-1] == 0.0)
            {
                *sing = 1;
            }
            jj += (m - j + 1);
        }
    }
    l = jj;
    for (i = n; i <= m; i++) {
	    s[l-1] = w[i-1];
	    l++;
    }

    if (s[jj-1] == 0.) {
	*sing = 1;
    }

    return;
}

void dogleg(const int n, const double* r, const double* diag, const double* qtb,
            const double* delta, double* x, double* wa1, double* wa2)
{
    int i, j, jj, k, l;
    double alpha, bnorm, epsmch, gnorm, qnorm, sgnorm, ssum, temp;

    epsmch = dpmpar[0];
    jj = (n * (n + 1)) / 2;

    for (k = 0; k < n; k++)
    {
        j = n - k - 1;
        jj -= k + 1;
        l = jj + 1;
        ssum = 0.0;
        if (n - 1 > j)
        {
            for (i = j + 1; i < n; i++)
            {
                ssum += r[l]*x[i];
                l++;
            }
        }
        temp = r[jj];
        if (temp == 0.0)
        {
            l = j;
            for (i = 0; i <= j; i++)
            {
                temp = fmax(temp, fabs(r[l]));
                l += n - i - 1;
            }
            temp = epsmch*temp;
            if (temp == 0.0) { temp = epsmch; }
        }
        x[j] = (qtb[j] - ssum)/temp;
    }
    for (j = 0; j < n; j++)
    {
        wa1[j] = 0.0;
        wa2[j] = diag[j] * x[j];
    }
    qnorm = enorm(n, wa2);
    if (qnorm <= *delta) { return; }
    l = 0;
    for (j = 0; j < n; j++) {
        temp = qtb[j];
        for (i = j; i < n; i++) {
            wa1[i] += r[l] * temp;
            l++;
        }
        wa1[j] /= diag[j];
    }

    gnorm = enorm(n, wa1);
    sgnorm = 0.0;
    alpha = *delta / qnorm;
    if (gnorm != 0.0)
    {
        for (j = 0; j < n; j++)
        {
            wa1[j] /= gnorm;
            wa1[j] /= diag[j];
        }

        l = 0;
        for (j = 0; j < n; j++) {
            ssum = 0.0;
            for (i = j; i < n; i++) {
                ssum += r[l] * wa1[i];
                l++;
            }
            wa2[j] = ssum;
        }
        temp = enorm(n, wa2);
        sgnorm = (gnorm / temp) / temp;
        alpha = 0.0;
        if (sgnorm < *delta)
        {
            bnorm = enorm(n, qtb);
            temp = (bnorm/gnorm)*(bnorm/qnorm)*(sgnorm / *delta);
            temp = temp - (*delta / qnorm)*pow(sgnorm / *delta, 2.0)
                        + sqrt(pow(temp - (*delta / qnorm), 2.0)
                               + (
                                   (1.0 - pow(*delta / qnorm, 2.0) )
                                  *(1.0 - pow((sgnorm / *delta), 2.0) )
                                 )
                              );
            alpha = ( (*delta/qnorm)*(1.0 - pow(sgnorm / *delta, 2.0)) )/temp;
        }
    }
    temp = (1.0 - alpha)*fmin(sgnorm, *delta);
    for (j = 0; j < n; j++) {
        x[j] = temp*wa1[j] + alpha*x[j];
    }
    return;
}

void fdjac1(int(*fcn)(int* n, double* x, double* fvec, int* iflag), const int n, double* x,
            const double* fvec, double* fjac, const int ldfjac, int* iflag, const int ml,
            const int mu, const double epsfcn, double* wa1, double* wa2)
{
    int i, j, k, msum, mut_n;
    double h, temp;
    const double epsmch = dpmpar[0];
    const double eps = sqrt(fmax(epsfcn, epsmch));
    mut_n = n;
    msum = ml + mu + 1;
    if (msum >= n)
    {
        for (j = 0; j < n; j++)
        {
            temp = x[j];
            h = eps*fabs(temp);
            if (h == 0.0) { h = eps; }
            x[j] = temp + h;
            (*fcn)(&mut_n, x, wa1, iflag);
            if (*iflag < 0) { return; }
            x[j] = temp;
            for (i = 0; i < n; i++)
            {
                fjac[i + ldfjac*j] = (wa1[i] - fvec[i])/h;
            }
        }
        return;
    }
    for (k = 1; k <= msum; k++)
    {
	    for (j = k; j <= n; j += msum)
        {
            wa2[j-1] = x[j-1];
            h = eps*fabs(wa2[j-1]);
            if (h == 0.0) { h = eps; }
            x[j-1] = wa2[j-1] + h;
        }
        (*fcn)(&mut_n, x, wa1, iflag);
        if (*iflag < 0) { return; }

        for (j = k; j <= n; j += msum)
        {
            x[j-1] = wa2[j-1];
            h = eps * fabs(wa2[j-1]);
            if (h == 0.) { h = eps; }
            for (i = 1; i <= n; i++)
            {
                fjac[(i-1) + ldfjac*(j-1)] = 0.0;
                if ((i >= j - mu) && (i <= j + ml))
                {
                    fjac[(i-1) + ldfjac*(j-1)] = (wa1[i-1] - fvec[i-1]) / h;
                }
            }
        }
	}
    return;
}

void HYBRD(int(*fcn)(int* n, double* x, double* fvec, int* iflag), const int n,
           double* x, double* fvec, const double xtol, const int maxfev,
           const int ml, const int mu, const double epsfcn, double* diag,
           const int mode, const double factor, const int nprint,
           int* info, int* nfev, double* fjac, const int ldfjac, double* r,
           const int lr, double* qtf, double* wa1, double* wa2, double* wa3,
           double* wa4)
{
    int i, iflag, iter, j, jeval, sing, l, msum, mut_n, ncfail, ncsuc, nslow1, nslow2;
    double actred, delta, fnorm, fnorm1, pnorm, prered, ratio, ssum, temp, xnorm;
    int iwa[1];
    double epsmch = dpmpar[0];

    mut_n = n;
    *info = 0;
    iflag = 0;
    *nfev = 0;
    if ((n <= 0) || (xtol < 0.0) || (maxfev <= 0) || (ml < 0) || (mu < 0) ||
	    (factor <= 0.0) || (ldfjac < n) || (lr < n * (n + 1) / 2)) { goto EXIT300; }
    if (mode == 2)
    {
        for (j = 0; j < n; j++)
        {
            if (diag[j] <= 0.0) { goto EXIT300; }
        }
    }
    iflag = 1;
    (*fcn)(&mut_n, x, fvec, &iflag);
    *nfev = 1;
    if (iflag < 0) { goto EXIT300; }
    fnorm = enorm(n, fvec);
    msum = (ml+mu+1 > n ? n : ml+mu+1);
    iter = 1;
    ncsuc = 0;
    ncfail = 0;
    nslow1 = 0;
    nslow2 = 0;

    while (1)
    {
        jeval = 1;
        iflag = 2;
        fdjac1(fcn, n, x, fvec, fjac, ldfjac, &iflag, ml, mu, epsfcn, wa1, wa2);
        *nfev += msum;
        if (iflag < 0) { goto EXIT300; }
        qrfac(n, n, fjac, ldfjac, 0, iwa, wa1, wa2, wa3);
        if (iter == 1)
        {
            if (mode != 2)
            {
                for (j = 0; j < n; j++)
                {
                    diag[j] = wa2[j];
                    if (wa2[j] == 0.0)
                    {
                        diag[j] = 1.0;
                    }
                }
            }
            for (j = 0; j < n; j++)
            {
                wa3[j] = diag[j]*x[j];
            }

            xnorm = enorm(n, wa3);
            delta = factor*xnorm;
            delta = (factor*xnorm == 0.0 ? factor : factor*xnorm);
        }
        for (i = 0; i < n; i++)
        {
            qtf[i] = fvec[i];
        }

        for (j = 0; j < n; j++)
        {
            if (fjac[j + ldfjac*j] != 0.0)
            {
                ssum = 0.0;
                for (i = j; i < n; i++)
                {
                    ssum += fjac[i + ldfjac*j]*qtf[i];
                }
                temp = -ssum / fjac[j + ldfjac*j];
                for (i = j; i < n; i++)
                {
                    qtf[i] += fjac[i + ldfjac*j]*temp;
                }
            }
        }
        sing = 0;
        for (j = 0; j < n; j++)
        {
            l = j;
            if (j > 0)
            {
                for (i = 0; i < j; i++)
                {
                    r[l] = fjac[i + ldfjac*j];
                    l += n - i - 1;
                }
            }
            r[l] = wa1[j];
            if (wa1[j] == 0.0) { sing = 1; }
        }
        qform(n, n, fjac, ldfjac, wa1);
        if (mode != 2)
        {
            for (j = 0; j < n; j++)
            {
                diag[j] = fmax(diag[j], wa2[j]);
            }
        }
        while (1)
        {
            if (nprint > 0) {
                iflag = 0;
                if ((iter - 1) % nprint == 0) {
                    (*fcn)(&mut_n, x, fvec, &iflag);
                }
                if (iflag < 0) { goto EXIT300; }
            }
            dogleg(n, r, diag, qtf, &delta, wa1, wa2, wa3);
            for (j = 0; j < n; j++)
            {
                wa1[j] = -wa1[j];
                wa2[j] = x[j] + wa1[j];
                wa3[j] = diag[j]*wa1[j];
            }

            pnorm = enorm(n, wa3);
            if (iter == 1) { delta = fmin(delta, pnorm); }
            iflag = 1;
            (*fcn)(&mut_n, wa2, wa4, &iflag);
            *nfev += 1;
            if (iflag < 0) { goto EXIT300; }
            fnorm1 = enorm(n, wa4);
            actred = (fnorm1 < fnorm ? 1.0 - pow(fnorm1/fnorm, 2.0) : -1.0);
            l = 0;
            for (i = 0; i < n; i++)
            {
                ssum = 0.0;
                for (j = i; j < n; j++)
                {
                    ssum += r[l] * wa1[j];
                    l++;
                }
                wa3[i] = qtf[i] + ssum;
            }

            temp = enorm(n, wa3);
            prered = (temp < fnorm ? 1.0 - pow(temp/fnorm, 2.0) : 0.0);
            ratio = (prered > 0.0 ? actred/prered : 0.0);
            if (ratio < 0.1)
            {
                ncsuc = 0;
                ncfail++;
                delta = 0.5*delta;
            } else {
                ncfail = 0;
                ncsuc++;
                if ((ratio >= 0.5) || (ncsuc > 1)) { delta = fmax(delta, pnorm/0.5); }
                if (fabs(ratio - 1.0) <= 0.1) { delta = pnorm / 0.5; }
            }
            if (ratio >= 0.0001)
            {
                for (j = 0; j < n; j++)
                {
                    x[j] = wa2[j];
                    wa2[j] = diag[j] * x[j];
                    fvec[j] = wa4[j];
                }
                xnorm = enorm(n, wa2);
                fnorm = fnorm1;
                iter++;
            }
            nslow1++;
            if (actred >= 0.001) { nslow1 = 0; }
            if (jeval) { nslow2++; }
            if (actred >= 0.1) { nslow2 = 0; }
            if ((delta <= xtol*xnorm) || (fnorm == 0.0)) { *info = 1; }
            if (*info != 0) { goto EXIT300; }
            if (*nfev >= maxfev) { *info = 2; }
            if (0.1*fmax(0.1*delta, pnorm) <= epsmch*xnorm) { *info = 3; }
            if (nslow2 == 5) { *info = 4; }
            if (nslow1 == 10) { *info = 5; }
            if (*info != 0) { goto EXIT300; }
            if (ncfail == 2) { break; }
            for (j = 0; j < n; j++)
            {
                ssum = 0.0;
                for (i = 0; i < n; i++)
                {
                    ssum += fjac[i + ldfjac*j]*wa4[i];
                }
                wa2[j] = (ssum - wa3[j]) / pnorm;
                wa1[j] = diag[j]*((diag[j] * wa1[j]) / pnorm);
                if (ratio >= 0.0001) { qtf[j] = ssum; }
            }
            r1updt(n, n, r, wa1, wa2, wa3, &sing);
            r1mpyq(n, n, fjac, ldfjac, wa2, wa3);
            r1mpyq(1, n, qtf, 1, wa2, wa3);
            jeval = 0;
        }
    }
EXIT300:
    if (iflag < 0) { *info = iflag; }
    iflag = 0;
    if (nprint > 0) { (*fcn)(&mut_n, x, fvec, &iflag);}

    return;
}

/* ================================================================ shapiro (swilk, _ansari_swilk_statistics.pyx) */

double sw_poly(const double *c, int nord, double x)
{
    double res = c[0];
    if (nord == 1) return res;
    double p = x * c[nord - 1];
    if (nord == 2) return res + p;
    for (int ind = nord - 2; ind > 0; ind--) p = (p + c[ind]) * x;
    res += p;
    return res;
}

double sw_alnorm(double x, bool upper)
{
    const double A1 = 0.398942280444, A2 = 0.399903438504, A3 = 5.75885480458, A4 = 29.8213557808,
                 A5 = 2.62433121679, A6 = 48.6959930692, A7 = 5.92885724438;
    const double B1 = 0.398942280385, B2 = 3.8052e-8, B3 = 1.00000615302, B4 = 3.98064794e-4, B5 = 1.98615381364,
                 B6 = 0.151679116635, B7 = 5.29330324926, B8 = 4.8385912808, B9 = 15.1508972451,
                 B10 = 0.742380924027, B11 = 30.789933034, B12 = 3.99019417011;
    const double ltone = 7., utzero = 38., con = 1.28;
    double z = x, y, temp;
    if (!(z > 0)) {
        upper = false;
        z = -z;
    }
    if (!((z <= ltone) || (upper && z <= utzero))) return upper ? 0. : 1.;
    y = 0.5 * z * z;
    if (z <= con) {
        temp = 0.5 - z * (A1 - A2 * y / (y + A3 - A4 / (y + A5 + A6 / (y + A7))));
    } else {
        temp = B1 * std::exp(-y) / (z - B2 + B3 / (z + B4 + B5 / (z - B6 + B7 / (z + B8 - B9 / (z + B10 + B11 / (z + B12))))));
    }
    return upper ? temp : (1 - temp);
}

double sw_ppnd(double p)
{
    const double A0 = 2.50662823884, A1 = -18.61500062529, A2 = 41.39119773534, A3 = -25.44106049637;
    const double B1 = -8.47351093090, B2 = 23.08336743743, B3 = -21.06224101826, B4 = 3.13082909833;
    const double C0 = -2.78718931138, C1 = -2.29796479134, C2 = 4.85014127135, C3 = 2.32121276858;
    const double D1 = 3.54388924762, D2 = 1.63706781897;
    const double split = 0.42;
    double q = p - 0.5, r, temp;
    if (std::fabs(q) <= split) {
        r = q * q;
        temp = q * (((A3 * r + A2) * r + A1) * r + A0);
        temp = temp / ((((B4 * r + B3) * r + B2) * r + B1) * r + 1.);
        return temp;
    }
    r = p;
    if (q > 0) r = 1 - p;
    if (r > 0) r = std::sqrt(-std::log(r));
    else return 0.;
    temp = (((C3 * r + C2) * r + C1) * r + C0);
    temp /= (D2 * r + D1) * r + 1.;
    return q < 0 ? -temp : temp;
}

/* swilk(x sorted, a[n/2]) -> w, pw (init = False, n1 = n) */
void swilk(const vector<double> &x, vector<double> &a, double *wout, double *pwout)
{
    const int n = (int)x.size();
    const int n2 = (int)a.size();
    const int n1 = n;
    const double c1[6] = {0., 0.221157, -0.147981, -0.207119e1, 0.4434685e1, -0.2706056e1};
    const double c2[6] = {0., 0.42981e-1, -0.293762, -0.1752461e1, 0.5682633e1, -0.3582633e1};
    const double c3[4] = {0.5440, -0.39978, 0.25054e-1, -0.6714e-3};
    const double c4[4] = {0.13822e1, -0.77857, 0.62767e-1, -0.20322e-2};
    const double c5[4] = {-0.15861e1, -0.31082, -0.83751e-1, 0.38915e-2};
    const double c6[3] = {-0.4803, -0.82676e-1, 0.30302e-2};
    const double g[2] = {-0.2273e1, 0.459};
    const double SQRTH = std::sqrt(2.0) / 2, PI6 = 6 / PI, SMALL = 1e-19;
    double w = 1., pw = 1.;
    *wout = w;
    *pwout = pw;
    const int nn2 = n / 2;
    if (nn2 < n2 || n < 3) return;
    const double an = n;
    if (n == 3) {
        a[0] = SQRTH;
    } else {
        const double an25 = an + 0.25;
        double summ2 = 0.;
        for (int ind1 = 0; ind1 < n2; ind1++) {
            const double temp = sw_ppnd((ind1 + 1 - 0.375) / an25);
            a[ind1] = temp;
            summ2 += temp * temp;
        }
        summ2 *= 2.;
        const double ssumm2 = std::sqrt(summ2);
        const double rsn = 1 / std::sqrt(an);
        const double A1 = sw_poly(c1, 6, rsn) - (a[0] / ssumm2);
        int i1;
        double fac;
        if (n > 5) {
            i1 = 2;
            const double A2 = -a[1] / ssumm2 + sw_poly(c2, 6, rsn);
            fac = std::sqrt((summ2 - (2 * (a[0] * a[0])) - 2 * (a[1] * a[1])) / (1 - (2 * (A1 * A1)) - 2 * (A2 * A2)));
            a[1] = A2;
        } else {
            i1 = 1;
            fac = std::sqrt((summ2 - 2 * (a[0] * a[0])) / (1 - 2 * (A1 * A1)));
        }
        a[0] = A1;
        for (int ind1 = i1; ind1 < nn2; ind1++) a[ind1] *= -1. / fac;
    }
    const double RANGE = x[n1 - 1] - x[0];
    if (RANGE < SMALL) return;
    double XX = x[0] / RANGE, SX = XX, SA = -a[0];
    int ind2 = n - 2;
    for (int ind1 = 1; ind1 < n1; ind1++) {
        const double XI = x[ind1] / RANGE;
        SX += XI;
        if (ind1 != ind2) SA += (ind1 < ind2 ? -1 : 1) * a[std::min(ind1, ind2)];
        XX = XI;
        ind2 -= 1;
    }
    SA /= n1;
    SX /= n1;
    double SSA = 0., SSX = 0., SAX = 0.;
    ind2 = n - 1;
    for (int ind1 = 0; ind1 < n1; ind1++) {
        double ASA;
        if (ind1 != ind2) ASA = (ind1 < ind2 ? -1 : 1) * a[std::min(ind1, ind2)] - SA;
        else ASA = -SA;
        const double XSX = x[ind1] / RANGE - SX;
        SSA += ASA * ASA;
        SSX += XSX * XSX;
        SAX += ASA * XSX;
        ind2 -= 1;
    }
    const double SSASSX = std::sqrt(SSA * SSX);
    const double w1 = (SSASSX - SAX) * (SSASSX + SAX) / (SSA * SSX);
    w = 1 - w1;
    *wout = w;
    if (n == 3) {
        if (w < 0.75) { *wout = 0.75; *pwout = 0.; }
        else *pwout = 1. - PI6 * std::acos(std::sqrt(w));
        return;
    }
    double y = std::log(w1);
    XX = std::log(an);
    double m, s;
    if (n <= 11) {
        const double gamma = sw_poly(g, 2, an);
        if (y >= gamma) { *pwout = SMALL; return; }
        y = -std::log(gamma - y);
        m = sw_poly(c3, 4, an);
        s = std::exp(sw_poly(c4, 4, an));
    } else {
        m = sw_poly(c5, 4, XX);
        s = std::exp(sw_poly(c6, 3, XX));
    }
    *pwout = sw_alnorm((y - m) / s, true);
}

int core_shapiro(vector<vector<double>> &xs, double *out, unsigned)
{
    const vector<double> &x = xs[0];
    const int64_t N = (int64_t)x.size();
    if (N < 3) return fail("Data must be at least length 3.");
    vector<double> y = x;
    sort_nan_last(y);
    const double med = x[N / 2];
    for (auto &v : y) v -= med;
    vector<double> a(N / 2, 0.0);
    swilk(y, a, &out[0], &out[1]);
    return TSR_OK;
}

/* shapiro(x, *, axis=None, nan_policy='propagate', keepdims=False) */
int r_shapiro(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x;
    if (!get_nda(a, n, 0, x)) return fail("x must be an array");
    AxisArgs ax;
    int rc = axis_args(a, n, 1, 2, 3, "None", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 2;
    T.too_small = 2;
    T.core = core_shapiro;
    return run_test(T, {x}, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* ================================================================ Cramer-von Mises (_hypotests.py) */

double cdf_cvm_inf(double x)
{
    double tot = 0.0;
    const double pi15 = std::pow(PI, 1.5);
    for (int k = 0;; k++) {
        const double u = std::exp(sc::gammaln(k + 0.5) - sc::gammaln(k + 1)) / (pi15 * std::sqrt(x));
        const int y = 4 * k + 1;
        const double q = (double)(y * y) / (16 * x);
        const double b = sc::kv(0.25, q);
        const double z = u * std::sqrt((double)y) * std::exp(-q) * b;
        tot += z;
        if (!(std::fabs(z) >= 1e-7)) break;
    }
    return tot;
}

double cvm_ed2(double y)
{
    const double z = y * y / 4;
    const double b = sc::kv(1.0 / 4, z) + sc::kv(3.0 / 4, z);
    return std::exp(-z) * std::pow(y / 2, 3.0 / 2) * b / std::sqrt(PI);
}

double cvm_ed3(double y)
{
    const double z = y * y / 4;
    const double c = std::exp(-z) / std::sqrt(PI);
    const double kv_terms = 2 * sc::kv(1.0 / 4, z) + 3 * sc::kv(3.0 / 4, z) - sc::kv(5.0 / 4, z);
    return c * std::pow(y / 2, 5.0 / 2) * kv_terms;
}

double cvm_Ak(int k, double x)
{
    const int m = 2 * k + 1;
    const double sx = 2 * std::sqrt(x);
    const double y1 = std::pow(x, 3.0 / 4), y2 = std::pow(x, 5.0 / 4);
    const double g12 = sc::gamma(k + 1.0 / 2), g32 = sc::gamma(k + 3.0 / 2);
    const double e1 = m * g12 * cvm_ed2((4 * k + 3) / sx) / (9 * y1);
    const double e2 = g12 * cvm_ed3((4 * k + 1) / sx) / (72 * y2);
    const double e3 = 2 * (m + 2) * g32 * cvm_ed3((4 * k + 5) / sx) / (12 * y2);
    const double e4 = 7 * m * g12 * cvm_ed2((4 * k + 1) / sx) / (144 * y1);
    const double e5 = 7 * m * g12 * cvm_ed2((4 * k + 5) / sx) / (144 * y1);
    return e1 + e2 + e3 + e4 + e5;
}

double psi1_mod(double x)
{
    double tot = 0.0;
    for (int k = 0;; k++) {
        const double gamma_kp1 = sc::gamma(k + 1.0);
        const double z = -cvm_Ak(k, x) / (PI * gamma_kp1);
        tot = tot + z;
        if (!(std::fabs(z) >= 1e-7)) break;
    }
    return tot;
}

double cdf_cvm(double x, double n)
{
    double y = 0.0;
    if ((1. / (12 * n) < x) && (x < n / 3.)) y = cdf_cvm_inf(x) * (1 + 1. / (12 * n)) + psi1_mod(x) / n;
    if (x >= n / 3) y = 1.;
    return y;
}

/* cramervonmises(rvs, cdf, args=(), *, axis=0, nan_policy='propagate', keepdims=False) */
int r_cramervonmises(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x;
    if (!get_nda(a, n, 0, x)) return fail("rvs must be an array");
    if (!has(a, n, 1) || a[1].kind != 2) return fail("`cdf` must be the name of a distribution (a callable is not supported)");
    NDA args;
    vector<double> argv;
    if (has(a, n, 2)) {
        if (!get_nda(a, n, 2, args)) return fail("args must be a tuple of numbers");
        argv = args.v;
    }
    NamedCdf F;
    int rc = F.init(a[1].str, argv);
    if (rc != TSR_OK) return rc;
    AxisArgs ax;
    rc = axis_args(a, n, 3, 4, 5, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 2;
    T.too_small = 1;
    T.vectorized = true;
    T.core = [&F](vector<vector<double>> &xs, double *out, unsigned flags) -> int {
        vector<double> v = xs[0];
        const int64_t nn = (int64_t)v.size();
        if (nn <= 1) return fail("The sample must contain at least two observations.");
        const double N = (double)nn;
        sort_nan_last(v);
        vector<double> t(nn);
        for (int64_t i = 0; i < nn; i++) {
            const double u = (2 * (double)(i + 1) - 1) / (2 * N);
            const double d = u - F.cdf(v[i]);
            t[i] = d * d;
        }
        const double w = 1 / (12 * N) + asum(t, flags);
        double p = 1. - cdf_cvm(w, N);
        if (p < 0.) p = 0.;
        out[0] = w;
        out[1] = p;
        return TSR_OK;
    };
    return run_test(T, {x}, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* _pval_cvm_2samp_exact(s, m, n): the frequency table of Xiao, Gordon and Yakovlev (2006) */
double pval_cvm_2samp_exact(double s, int64_t m, int64_t n)
{
    const int64_t lcm = std::lcm(m, n);
    const int64_t a = lcm / m, b = lcm / n;
    const int64_t mn = m * n;
    const double zeta = np_floordiv((double)(lcm * lcm * (m + n)) * (6 * s - (double)(mn * (4 * mn - 1))),
                                    (double)(6 * mn * mn));
    typedef std::map<int64_t, double> Tab;   /* value -> frequency (exact integers) */
    vector<Tab> gs(m + 1);
    gs[0][0] = 1;
    for (int64_t u = 0; u <= n; u++) {
        vector<Tab> next(m + 1);
        Tab tmp;
        for (int64_t v = 0; v <= m; v++) {
            for (auto &kv : gs[v]) tmp[kv.first] += kv.second;
            const int64_t r = (a * v - b * u) * (a * v - b * u);
            Tab shifted;
            for (auto &kv : tmp) shifted[kv.first + r] = kv.second;
            tmp.swap(shifted);
            next[v] = tmp;
        }
        gs.swap(next);
    }
    double combos = 1;   /* math.comb(m + n, m), exact below 2^53 */
    {
        long double c = 1;
        for (int64_t i = 1; i <= m; i++) c = c * (long double)(n + i) / (long double)i;
        combos = (double)std::nearbyint((double)c);
    }
    uint64_t cnt = 0;
    for (auto &kv : gs[m]) if ((double)kv.first >= zeta) cnt += (uint64_t)kv.second;
    return (double)cnt / combos;
}

/* cramervonmises_2samp(x, y, method='auto', *, axis=0, nan_policy='propagate', keepdims=False) */
int r_cramervonmises_2samp(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x, y;
    if (!get_nda(a, n, 0, x) || !get_nda(a, n, 1, y)) return fail("x and y must be arrays");
    const string method = get_str(a, n, 2, "auto");
    AxisArgs ax;
    int rc = axis_args(a, n, 3, 4, 5, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 2;
    T.too_small = 1;
    T.vectorized = true;
    T.core = [method](vector<vector<double>> &xs, double *out, unsigned) -> int {
        const int64_t nx = (int64_t)xs[0].size(), ny = (int64_t)xs[1].size();
        if (nx <= 1 || ny <= 1) return fail("x and y must contain at least two observations.");
        string m = method;
        if (m != "auto" && m != "exact" && m != "asymptotic") return fail("method must be either auto, exact or asymptotic.");
        if (m == "auto") m = std::max(nx, ny) > 20 ? "asymptotic" : "exact";
        vector<double> xa = xs[0], ya = xs[1];
        sort_nan_last(xa);
        sort_nan_last(ya);
        vector<double> z = xa;
        z.insert(z.end(), ya.begin(), ya.end());
        const vector<double> r = rank_average(z);
        vector<double> dx(nx), dy(ny);
        for (int64_t i = 0; i < nx; i++) { const double d = r[i] - (double)(i + 1); dx[i] = d * d; }
        for (int64_t j = 0; j < ny; j++) { const double d = r[nx + j] - (double)(j + 1); dy[j] = d * d; }
        const double u = (double)nx * psum(dx) + (double)ny * psum(dy);
        /* SciPy computes these in Python integers (exact, unbounded); doubles are exact up to 2^53 and never
           overflow, where int64 products overflow at about 1e5 observations per sample */
        const double k = (double)nx * (double)ny, N = (double)nx + (double)ny, dnx = (double)nx, dny = (double)ny;
        const double t = u / (k * N) - (4 * k - 1) / (6 * N);
        double p;
        if (m == "exact") {
            p = pval_cvm_2samp_exact(u, nx, ny);
        } else {
            const double et = (1 + 1.0 / N) / 6;
            double vt = (N + 1) * (4 * k * N - 3 * (dnx * dnx + dny * dny) - 2 * k);
            vt = vt / (45 * N * N * 4 * k);
            const double tn = 1.0 / 6 + (t - et) / std::sqrt(45 * vt);
            if (tn >= 0.003) {
                p = 1. - cdf_cvm_inf(tn);
                if (p < 0.) p = 0.;
            } else {
                p = 1.;
            }
        }
        out[0] = t;
        out[1] = p;
        return TSR_OK;
    };
    return run_test(T, {x, y}, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* ================================================================ Kolmogorov-Smirnov distributions (_ksstats.py) */

const double E128 = 128;
const double EP128 = std::ldexp(1.0, 128), EM128 = std::ldexp(1.0, -128);

double ks_select(double cdfprob, double sfprob, bool cdf) { return clip01(cdf ? cdfprob : sfprob); }
long double ks_select_ld(long double cdfprob, long double sfprob, bool cdf)
{
    const long double p = cdf ? cdfprob : sfprob;
    return p != p ? p : (p < 0 ? 0.0L : (p > 1 ? 1.0L : p));
}

/* numpy matmul of small square matrices (row-major) */
void matmul(const vector<double> &A, const vector<double> &B, vector<double> &C, int m)
{
    vector<double> R(m * m);
    for (int i = 0; i < m; i++)
        for (int j = 0; j < m; j++) {
            double s = 0;
            for (int k = 0; k < m; k++) s = std::fma(A[i * m + k], B[k * m + j], s);
            R[i * m + j] = s;
        }
    C.swap(R);
}

long double kolmogn_DMTW(int64_t n, double d, bool cdf)
{
    if (d >= 1.0) return ks_select_ld(1.0L, 0.0L, cdf);
    const double nd = n * d;
    if (nd <= 0.5) return ks_select_ld(0.0L, 1.0L, cdf);
    const int64_t k = (int64_t)std::ceil(nd);
    const double h = k - nd;
    const int m = (int)(2 * k - 1);
    vector<double> H(m * m, 0.0), v(m), w(m);
    for (int j = 1; j <= m; j++) v[j - 1] = 1.0 - std::pow(h, (double)j);
    double fac = 1.0;
    for (int j = 1; j <= m; j++) {
        w[j - 1] = fac;
        fac /= j;
        v[j - 1] *= fac;
    }
    const double tt = std::pow(std::max(2 * h - 1.0, 0.0), (double)m) - 2 * std::pow(h, (double)m);
    v[m - 1] = (1.0 + tt) * fac;
    for (int i = 1; i < m; i++)
        for (int r = i - 1; r < m; r++) H[r * m + i] = w[r - (i - 1)];
    for (int r = 0; r < m; r++) H[r * m + 0] = v[r];
    for (int c = 0; c < m; c++) H[(m - 1) * m + c] = v[m - 1 - c];
    vector<double> Hp(m * m, 0.0);
    for (int i = 0; i < m; i++) Hp[i * m + i] = 1.0;
    int64_t nn = n;
    int expnt = 0, Hexpnt = 0;
    while (nn > 0) {
        if (nn % 2) {
            matmul(Hp, H, Hp, m);
            expnt += Hexpnt;
        }
        matmul(H, H, H, m);
        Hexpnt *= 2;
        if (std::fabs(H[(k - 1) * m + (k - 1)]) > EP128) {
            for (auto &e : H) e /= EP128;
            Hexpnt += (int)E128;
        }
        nn = nn / 2;
    }
    /* SciPy scales by the long double 2^128: once that happens, p is a long double */
    double p = Hp[(k - 1) * m + (k - 1)];
    long double pl = 0;
    bool promoted = false;
    for (int64_t i = 1; i <= n; i++) {
        if (!promoted) {
            p = i * p / n;
            if (std::fabs(p) < EM128) {
                pl = (long double)p * (long double)EP128;
                promoted = true;
                expnt -= (int)E128;
            }
        } else {
            pl = (long double)i * pl / (long double)n;
            if (std::fabs(pl) < (long double)EM128) {
                pl *= (long double)EP128;
                expnt -= (int)E128;
            }
        }
    }
    long double r = promoted ? pl : (long double)p;
    if (expnt != 0) r = promoted ? std::ldexp(pl, expnt) : (long double)std::ldexp(p, expnt);
    return ks_select_ld(r, 1.0L - r, cdf);
}

void pomeranz_j1j2(int64_t i, int64_t n, int64_t ll, int64_t ceilf, int64_t roundf, int64_t &j1o, int64_t &j2o)
{
    int64_t j1, j2;
    if (i == 0) {
        j1 = -ll - ceilf - 1;
        j2 = ll + ceilf - 1;
    } else {
        const int64_t ip1div2 = (i + 1) / 2, ip1mod2 = (i + 1) % 2;
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
    j1o = std::max(j1 + 2, (int64_t)0);
    j2o = std::min(j2, n);
}

/* numpy's dot of short vectors (cblas_ddot: sequential fused multiply-adds for short lengths) */
double np_dot(const double *a, const double *b, int64_t n)
{
    double s = 0;
    for (int64_t i = 0; i < n; i++) s = std::fma(a[i], b[i], s);
    return s;
}

long double kolmogn_Pomeranz(int64_t n, double x, bool cdf)
{
    const double t = n * x;
    const int64_t ll = (int64_t)std::floor(t);
    const double f = 1.0 * (t - ll);
    const double g = std::min(f, 1.0 - f);
    const int64_t ceilf = f > 0 ? 1 : 0, roundf = f > 0.5 ? 1 : 0;
    const int64_t npwrs = 2 * (ll + 1);
    vector<double> gpower(npwrs), twogpower(npwrs), onem2gpower(npwrs);
    gpower[0] = twogpower[0] = onem2gpower[0] = 1.0;
    int64_t expnt = 0;
    const double g_over_n = g / n, two_g_over_n = 2 * g / n, one_minus_two_g_over_n = (1 - 2 * g) / n;
    for (int64_t m = 1; m < npwrs; m++) {
        gpower[m] = gpower[m - 1] * g_over_n / m;
        twogpower[m] = twogpower[m - 1] * two_g_over_n / m;
        onem2gpower[m] = onem2gpower[m - 1] * one_minus_two_g_over_n / m;
    }
    vector<double> V0(npwrs, 0.0), V1(npwrs, 0.0);
    V1[0] = 1;
    int64_t V0s = 0, V1s = 0;
    int64_t j1, j2;
    pomeranz_j1j2(0, n, ll, ceilf, roundf, j1, j2);
    for (int64_t i = 1; i < 2 * n + 2; i++) {
        const int64_t k1 = j1;
        std::swap(V0, V1);
        std::swap(V0s, V1s);
        std::fill(V1.begin(), V1.end(), 0.0);
        pomeranz_j1j2(i, n, ll, ceilf, roundf, j1, j2);
        const vector<double> *pwrs;
        if (i == 1 || i == 2 * n + 1) pwrs = &gpower;
        else pwrs = (i % 2) ? &twogpower : &onem2gpower;
        const int64_t ln2 = j2 - k1 + 1;
        if (ln2 > 0) {
            /* np.convolve(V0[k1 - V0s : k1 - V0s + ln2], pwrs[:ln2]) (slices clamp to the arrays) */
            auto pyidx = [&](int64_t i) { if (i < 0) i += npwrs; return std::min(std::max(i, (int64_t)0), npwrs); };
            const int64_t s0 = pyidx(k1 - V0s), e0 = pyidx(k1 - V0s + ln2);
            const int64_t la = std::max(e0 - s0, (int64_t)0), lb = std::min(ln2, npwrs);
            const double *A = V0.data() + s0, *B = pwrs->data();
            /* numpy: convolve(a, v) = correlate(a, v[::-1], 'full'), the longer array first */
            const double *P = A, *Q = B;
            int64_t lp = la, lq = lb;
            if (lq > lp) { std::swap(P, Q); std::swap(lp, lq); }
            const int64_t lc = lp + lq - 1;
            vector<double> conv(lc > 0 ? lc : 0);
            vector<double> qrev(Q, Q + lq);
            std::reverse(qrev.begin(), qrev.end());
            for (int64_t o = 0; o < lc; o++) {
                /* correlate full: output o pairs P[i] with qrev[i - o + lq - 1] */
                const int64_t lo = std::max((int64_t)0, o - (lq - 1)), hi = std::min(lp - 1, o);
                const int64_t cnt = hi - lo + 1;
                conv[o] = cnt > 0 ? np_dot(P + lo, qrev.data() + (lo - o + lq - 1), cnt) : 0.0;
            }
            const int64_t conv_start = j1 - k1, conv_len = j2 - j1 + 1;
            for (int64_t q = 0; q < conv_len && q < npwrs; q++) {
                const int64_t src = conv_start + q;
                V1[q] = (src >= 0 && src < lc) ? conv[src] : 0.0;
            }
            double mx = V1[0];
            for (auto e : V1) mx = std::max(mx, e);
            if (0 < mx && mx < EM128) {
                for (auto &e : V1) e *= EP128;
                expnt -= (int64_t)E128;
            }
            V1s = V0s + j1 - k1;
        }
    }
    double ans = V1[n - V1s];
    long double al = 0;
    bool promoted = false;
    for (int64_t m = 1; m <= n; m++) {
        if (!promoted) {
            if (std::fabs(ans) > EP128) {
                al = (long double)ans * (long double)EM128;
                promoted = true;
                expnt += (int64_t)E128;
                al *= m;
            } else {
                ans *= m;
            }
        } else {
            if (std::fabs(al) > (long double)EP128) {
                al *= (long double)EM128;
                expnt += (int64_t)E128;
            }
            al *= m;
        }
    }
    long double r = promoted ? al : (long double)ans;
    if (expnt != 0) r = promoted ? std::ldexp(al, (int)expnt) : (long double)std::ldexp(ans, (int)expnt);
    return ks_select_ld(r, 1.0L - r, cdf);
}

double kolmogn_PelzGood(int64_t n, double x, bool cdf)
{
    if (x <= 0.0) return ks_select(0.0, 1.0, cdf);
    if (x >= 1.0) return ks_select(1.0, 0.0, cdf);
    const double PI2 = PI * PI, PI4 = std::pow(PI, 4), PI6 = std::pow(PI, 6);
    const double SQRT2PI = std::sqrt(2 * PI), SQRT3 = std::sqrt(3.0);
    const double z = std::sqrt((double)n) * x;
    const double zsquared = z * z, zthree = std::pow(z, 3), zfour = std::pow(z, 4), zsix = std::pow(z, 6);
    const double qlog = -PI2 / 8 / zsquared;
    if (qlog < -708) return ks_select(0.0, 1.0, cdf);
    double q = std::exp(qlog);
    const double k1a = -zsquared, k1b = PI2 / 4;
    const double k2a = 6 * zsix + 2 * zfour, k2b = (2 * zfour - 5 * zsquared) * PI2 / 4, k2c = PI4 * (1 - 2 * zsquared) / 16;
    const double k3d = PI6 * (5 - 30 * zsquared) / 64, k3c = PI4 * (-60 * zsquared + 212 * zfour) / 16;
    const double k3b = PI2 * (135 * zfour - 96 * zsix) / 4, k3a = -30 * zsix - 90 * std::pow(z, 8);
    double K[4] = {0, 0, 0, 0};
    const int64_t maxk = (int64_t)std::ceil(16 * z / PI);
    for (int64_t k = maxk; k > 0; k--) {
        const int64_t m = 2 * k - 1;
        const double msq = (double)(m * m), mfour = (double)(m * m * m * m), msix = (double)(m * m * m * m * m * m);
        const double qpower = std::pow(q, (double)(8 * k));
        const double coeffs[4] = {1.0, k1a + k1b * msq, k2a + k2b * msq + k2c * mfour, k3a + k3b * msq + k3c * mfour + k3d * msix};
        for (int j = 0; j < 4; j++) { K[j] *= qpower; K[j] += coeffs[j]; }
    }
    for (int j = 0; j < 4; j++) { K[j] *= q; K[j] *= SQRT2PI; }
    const double den[4] = {z, 6 * zfour, 72 * std::pow(z, 7), 6480 * std::pow(z, 10)};
    for (int j = 0; j < 4; j++) K[j] /= den[j];
    q = std::exp(-PI2 / 2 / zsquared);
    vector<double> t2(maxk), t3(maxk);
    for (int64_t i = 0; i < maxk; i++) {
        const int64_t ks = maxk - i;
        const double ksq = (double)(ks * ks);
        const double qp = std::pow(q, ksq);
        const double kspi = PI * ks;
        t2[i] = ksq * qp;
        t3[i] = (SQRT3 * z + kspi) * (SQRT3 * z - kspi) * ksq * qp;
    }
    double k2extra = psum(t2);
    k2extra *= PI2 * SQRT2PI / (-36 * zthree);
    K[2] += k2extra;
    double k3extra = psum(t3);
    k3extra *= PI2 * SQRT2PI / (216 * zsix);
    K[3] += k3extra;
    for (int j = 0; j < 4; j++) K[j] /= std::pow(n * 1.0, j / 2.0);
    if (!cdf) {
        for (int j = 0; j < 4; j++) K[j] *= -1;
        K[0] += 1;
    }
    return ((0.0 + K[0]) + K[1] + K[2]) + K[3];
}

double log_nfact_div_npown(double n)
{
    static const double C[8] = {-2.955065359477124183e-2, 6.4102564102564102564e-3, -1.9175269175269175269e-3,
                                8.4175084175084175084e-4, -5.952380952380952381e-4, 7.9365079365079365079e-4,
                                -2.7777777777777777778e-3, 8.3333333333333333333e-2};
    const double rn = 1.0 / n;
    const double x = rn / n;
    double y = 0;
    for (int i = 0; i < 8; i++) y = y * x + C[i];
    return std::log(n) / 2 - n + std::log(2 * PI) / 2 + rn * y;
}

double kolmogn(int64_t n, double x, bool cdf)
{
    if (n <= 0) return NaN;
    if (x >= 1.0) return ks_select(1.0, 0.0, cdf);
    if (x <= 0.0) return ks_select(0.0, 1.0, cdf);
    const double t = n * x;
    if (t <= 1.0) {
        if (t <= 0.5) return ks_select(0.0, 1.0, cdf);
        double prob;
        if (n <= 140) {
            prob = 1.0;
            for (int64_t i = 1; i <= n; i++) prob *= i * (1.0 / n) * (2 * t - 1);
        } else {
            prob = std::exp(log_nfact_div_npown((double)n) + n * std::log(2 * t - 1));
        }
        return ks_select(prob, 1.0 - prob, cdf);
    }
    if (t >= n - 1) {
        const double prob = 2 * std::pow(1.0 - x, (double)n);
        return ks_select(1 - prob, prob, cdf);
    }
    if (x >= 0.5) {
        const double prob = 2 * sc::smirnov((double)n, x);
        return ks_select(1.0 - prob, prob, cdf);
    }
    const double nxsquared = t * x;
    if (n <= 140) {
        if (nxsquared <= 0.754693) {
            const long double prob = kolmogn_DMTW(n, x, true);
            return (double)ks_select_ld(prob, 1.0L - prob, cdf);
        }
        if (nxsquared <= 4) {
            const long double prob = kolmogn_Pomeranz(n, x, true);
            return (double)ks_select_ld(prob, 1.0L - prob, cdf);
        }
        const double prob = 2 * sc::smirnov((double)n, x);
        return ks_select(1.0 - prob, prob, cdf);
    }
    if (!cdf) {
        if (nxsquared >= 370.0) return 0.0;
        if (nxsquared >= 2.2) return clip01(2 * sc::smirnov((double)n, x));
    }
    long double cdfprob;
    if (nxsquared >= 18.0) cdfprob = 1.0L;
    else if (n <= 100000 && n * std::pow(x, 1.5) <= 1.4) cdfprob = kolmogn_DMTW(n, x, true);
    else cdfprob = kolmogn_PelzGood(n, x, true);
    return (double)ks_select_ld(cdfprob, 1.0L - cdfprob, cdf);
}

bool ks_argcheck(double n) { return n >= 1 && n == std::nearbyint(n); }

/* ksone.sf(x, n): support [0, 1], _sf = smirnov(n, x) */
double ksone_sf(double x, double n)
{
    if (x != x || !ks_argcheck(n)) return NaN;
    if (x <= 0) return 1.0;
    if (x >= 1) return 0.0;
    return sc::smirnov(n, x);
}
/* kstwo.sf(x, n): support [0.5/n, 1], _sf = kolmogn(n, x, cdf=False) */
double kstwo_sf(double x, double n)
{
    if (x != x || !ks_argcheck(n)) return NaN;
    if (x <= 0.5 / n) return 1.0;
    if (x >= 1) return 0.0;
    return kolmogn((int64_t)n, x, false);
}
/* kstwobign.sf(x): support [0, inf), _sf = kolmogorov(x) */
double kstwobign_sf(double x)
{
    if (x != x) return NaN;
    if (x <= 0) return 1.0;
    if (x == INF) return 0.0;
    return sc::kolmogorov(x);
}

int norm_alternative(string &alt)
{
    string low = alt;
    for (auto &c : low) c = (char)std::tolower((unsigned char)c);
    if (!low.empty()) {
        if (low[0] == 't') alt = "two-sided";
        else if (low[0] == 'g') alt = "greater";
        else if (low[0] == 'l') alt = "less";
    }
    return (alt == "two-sided" || alt == "greater" || alt == "less") ? TSR_OK : TSR_EARG;
}

/* ks_1samp on one slice: statistic, pvalue, statistic_location, statistic_sign */
int ks1_core(const vector<double> &xin, const NamedCdf &F, const string &alternative, const string &method, double *out)
{
    vector<double> x = xin;
    const int64_t N = (int64_t)x.size();
    sort_nan_last(x);
    vector<double> cdfv(N), Dp(N), Dm(N);
    for (int64_t i = 0; i < N; i++) cdfv[i] = F.cdf(x[i]);
    for (int64_t i = 0; i < N; i++) {
        Dp[i] = (double)(i + 1) / N - cdfv[i];
        Dm[i] = cdfv[i] - (double)i / N;
    }
    const int64_t ip = np_argmax(Dp), im = np_argmax(Dm);
    const double Dplus = Dp[ip], Dminus = Dm[im];
    if (alternative == "greater") {
        out[0] = Dplus; out[1] = ksone_sf(Dplus, (double)N); out[2] = x[ip]; out[3] = 1;
        return TSR_OK;
    }
    if (alternative == "less") {
        out[0] = Dminus; out[1] = ksone_sf(Dminus, (double)N); out[2] = x[im]; out[3] = -1;
        return TSR_OK;
    }
    const bool iplus = Dplus > Dminus;
    const double D = iplus ? Dplus : Dminus;
    double prob;
    string mode = method == "auto" ? "exact" : method;
    if (mode == "exact") prob = kstwo_sf(D, (double)N);
    else if (mode == "asymp") prob = kstwobign_sf(D * std::sqrt((double)N));
    else prob = 2 * ksone_sf(D, (double)N);
    out[0] = D;
    out[1] = clip01(prob);
    out[2] = iplus ? x[ip] : x[im];
    out[3] = iplus ? 1 : -1;
    return TSR_OK;
}

/* ks_2samp helpers (_stats_py.py, _stats_pythran.py) */
double prob_outside_square(int64_t n, int64_t h)
{
    double P = 0.0;
    int64_t k = (int64_t)std::floor((double)n / h);
    while (k >= 0) {
        double p1 = 1.0;
        for (int64_t j = 0; j < h; j++) p1 = (double)(n - k * h - j) * p1 / (double)(n + k * h + j + 1);
        P = p1 * (1.0 - P);
        k -= 1;
    }
    return 2 * P;
}

double outer_prob_inside(int64_t m, int64_t n, int64_t g, int64_t h)
{
    if (m < n) std::swap(m, n);
    const int64_t mg = m / g, ng = n / g;
    int64_t minj = 0, maxj = std::min((int64_t)std::ceil((double)h / mg), n + 1);
    int64_t curlen = maxj - minj;
    const int64_t lenA = std::min(2 * maxj + 2, n + 1);
    vector<double> A(lenA, 1.0);
    for (int64_t j = minj; j < maxj && j < lenA; j++) A[j] = 0.0;
    for (int64_t i = 1; i <= m; i++) {
        const int64_t lastminj = minj, lastlen = curlen;
        minj = std::max((int64_t)std::floor((double)(ng * i - h) / mg) + 1, (int64_t)0);
        minj = std::min(minj, n);
        maxj = std::min((int64_t)std::ceil((double)(ng * i + h) / mg), n + 1);
        if (maxj <= minj) return 1.0;
        double val = minj == 0 ? 0.0 : 1.0;
        for (int64_t jj = 0; jj < maxj - minj; jj++) {
            const int64_t j = jj + minj;
            val = (A[jj + minj - lastminj] * i + val * j) / (i + j);
            A[jj] = val;
        }
        curlen = maxj - minj;
        if (lastlen > curlen) {
            for (int64_t q = maxj - minj; q < maxj - minj + (lastlen - curlen) && q < lenA; q++) A[q] = 1;
        }
    }
    return A[maxj - minj - 1];
}

/* returns false on a floating-point overflow/invalid (SciPy's errstate raise) */
bool count_paths_outside(int64_t m, int64_t n, int64_t g, int64_t h, double *out)
{
    if (m < n) std::swap(m, n);
    const int64_t mg = m / g, ng = n / g;
    const int64_t lxj = n + py_floordiv(mg - h, mg);
    if (lxj == 0) { *out = sc::binom((double)(m + n), (double)n); return std::isfinite(*out); }
    vector<int64_t> xj(lxj);
    for (int64_t j = 0; j < lxj; j++) xj[j] = py_floordiv(h + mg * j + ng - 1, ng);
    vector<double> B(lxj, 0.0);
    B[0] = 1;
    for (int64_t j = 1; j < lxj; j++) {
        double Bj = sc::binom((double)(xj[j] + j), (double)j);
        if (!std::isfinite(Bj)) return false;
        for (int64_t i = 0; i < j; i++) {
            const double bin = sc::binom((double)(xj[j] - xj[i] + j - i), (double)(j - i));
            const double prod = bin * B[i];
            if (!std::isfinite(bin) || !std::isfinite(prod)) return false;
            Bj -= prod;
            if (!std::isfinite(Bj)) return false;
        }
        B[j] = Bj;
    }
    double num_paths = 0;
    for (int64_t j = 0; j < lxj; j++) {
        const double bin = sc::binom((double)((m - xj[j]) + (n - j)), (double)(n - j));
        const double term = B[j] * bin;
        if (!std::isfinite(bin) || !std::isfinite(term)) return false;
        num_paths += term;
        if (!std::isfinite(num_paths)) return false;
    }
    *out = num_paths;
    return true;
}

bool attempt_exact_2kssamp(int64_t n1, int64_t n2, int64_t g, double &d, const string &alternative, double *prob)
{
    const int64_t lcm = (n1 / g) * n2;
    const int64_t h = (int64_t)std::nearbyint(d * (double)lcm);
    d = (double)h * 1.0 / (double)lcm;
    if (h == 0) { *prob = 1.0; return true; }
    double p = NaN;
    if (alternative == "two-sided") {
        if (n1 == n2) p = prob_outside_square(n1, h);
        else p = outer_prob_inside(n1, n2, g, h);
    } else {
        if (n1 == n2) {
            p = 1.0;
            for (int64_t j = 0; j < h; j++) p *= (double)(n1 - j) / ((double)n1 + (double)j + 1.0);
        } else {
            double num_paths;
            if (!count_paths_outside(n1, n2, g, h, &num_paths)) { *prob = NaN; return false; }
            const double bin = sc::binom((double)(n1 + n2), (double)n1);
            if (num_paths > bin || std::isinf(bin)) { *prob = NaN; return false; }
            p = num_paths / bin;
        }
    }
    *prob = p;
    if (!(0 <= p && p <= 1)) return false;
    return true;
}

int ks2_core(vector<double> d1, vector<double> d2, string alternative, const string &mode_in, double *out)
{
    string mode = mode_in;
    if (mode != "auto" && mode != "exact" && mode != "asymp") return fail("Invalid value for mode");
    if (norm_alternative(alternative) != TSR_OK) return fail("Invalid value for alternative");
    sort_nan_last(d1);
    sort_nan_last(d2);
    const int64_t n1 = (int64_t)d1.size(), n2 = (int64_t)d2.size();
    if (std::min(n1, n2) == 0) return fail("Data passed to ks_2samp must not be empty");
    vector<double> all = d1;
    all.insert(all.end(), d2.begin(), d2.end());
    vector<double> cdd(all.size());
    for (size_t i = 0; i < all.size(); i++) {
        const double c1 = (double)ss_right(d1, all[i]) / (double)n1;
        const double c2 = (double)ss_right(d2, all[i]) / (double)n2;
        cdd[i] = c1 - c2;
    }
    const int64_t amin = np_argmin(cdd), amax = np_argmax(cdd);
    const double locmin = all[amin], locmax = all[amax];
    double minS = -cdd[amin];
    minS = minS != minS ? minS : std::min(std::max(minS, 0.0), 1.0);
    const double maxS = cdd[amax];
    double d, dloc;
    int dsign;
    if (alternative == "less" || (alternative == "two-sided" && minS > maxS)) { d = minS; dloc = locmin; dsign = -1; }
    else { d = maxS; dloc = locmax; dsign = 1; }
    const int64_t g = std::gcd(n1, n2);
    const int64_t n1g = n1 / g, n2g = n2 / g;
    double prob = -INF;
    if (mode == "auto") mode = std::max(n1, n2) <= 10000 ? "exact" : "asymp";
    else if (mode == "exact") {
        if ((double)n1g >= 2147483647.0 / (double)n2g) mode = "asymp";
    }
    if (mode == "exact") {
        if (!attempt_exact_2kssamp(n1, n2, g, d, alternative, &prob)) mode = "asymp";
    }
    if (mode == "asymp") {
        double m = (double)n1, n = (double)n2;
        if (m < n) std::swap(m, n);
        const double en = m * n / (m + n);
        if (alternative == "two-sided") {
            prob = kstwo_sf(d, std::nearbyint(en));
        } else {
            const double z = std::sqrt(en) * d;
            const double expt = -2 * (z * z) - 2 * z * (m + 2 * n) / std::sqrt(m * n * (m + n)) / 3.0;
            prob = std::exp(expt);
        }
    }
    out[0] = d;
    out[1] = clip01(prob);
    out[2] = dloc;
    out[3] = dsign;
    return TSR_OK;
}

/* ks_1samp(x, cdf, args=(), alternative='two-sided', method='auto', *, axis=0, nan_policy, keepdims) */
int r_ks_1samp(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x;
    if (!get_nda(a, n, 0, x)) return fail("x must be an array");
    if (!has(a, n, 1) || a[1].kind != 2) return fail("`cdf` must be the name of a distribution (a callable is not supported)");
    vector<double> argv;
    if (has(a, n, 2)) {
        NDA args;
        if (!get_nda(a, n, 2, args)) return fail("args must be a tuple of numbers");
        argv = args.v;
    }
    NamedCdf F;
    int rc = F.init(a[1].str, argv);
    if (rc != TSR_OK) return rc;
    string alternative = get_str(a, n, 3, "two-sided");
    const string method = get_str(a, n, 4, "auto");
    if (norm_alternative(alternative) != TSR_OK) return fail("Unexpected value for alternative");
    AxisArgs ax;
    rc = axis_args(a, n, 5, 6, 7, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 4;
    T.vectorized = true;
    T.outint = 8u;
    T.core = [&](vector<vector<double>> &xs, double *out, unsigned) -> int { return ks1_core(xs[0], F, alternative, method, out); };
    return run_test(T, {x}, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* ks_2samp(data1, data2, alternative='two-sided', method='auto', *, axis=0, nan_policy, keepdims) */
int r_ks_2samp(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x, y;
    if (!get_nda(a, n, 0, x) || !get_nda(a, n, 1, y)) return fail("data1 and data2 must be arrays");
    const string alternative = get_str(a, n, 2, "two-sided");
    const string method = get_str(a, n, 3, "auto");
    AxisArgs ax;
    int rc = axis_args(a, n, 4, 5, 6, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 4;
    T.outint = 8u;
    if (x.isint && y.isint) T.outint |= 4u;   /* the location is an element of the integer data */
    T.core = [&](vector<vector<double>> &xs, double *out, unsigned) -> int { return ks2_core(xs[0], xs[1], alternative, method, out); };
    return run_test(T, {x, y}, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* kstest(rvs, cdf, args=(), N=20, alternative='two-sided', method='auto', *, axis=0, nan_policy, keepdims) */
int r_kstest(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x;
    if (has(a, n, 0) && a[0].kind == 2) return fail("kstest: `rvs` as a distribution name draws random variates, which is not supported");
    if (!get_nda(a, n, 0, x)) return fail("rvs must be an array");
    string alternative = get_str(a, n, 4, "two-sided");
    if (alternative == "two_sided") alternative = "two-sided";
    if (alternative != "two-sided" && alternative != "greater" && alternative != "less") return fail("Unexpected alternative");
    const string method = get_str(a, n, 5, "auto");
    AxisArgs ax;
    int rc = axis_args(a, n, 6, 7, 8, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 4;
    T.outint = 8u;
    if (has(a, n, 1) && a[1].kind == 2) {
        vector<double> argv;
        if (has(a, n, 2)) {
            NDA args;
            if (!get_nda(a, n, 2, args)) return fail("args must be a tuple of numbers");
            argv = args.v;
        }
        auto F = std::make_shared<NamedCdf>();
        rc = F->init(a[1].str, argv);
        if (rc != TSR_OK) return rc;
        T.core = [F, alternative, method](vector<vector<double>> &xs, double *out, unsigned) -> int {
            return ks1_core(xs[0], *F, alternative, method, out);
        };
        return run_test(T, {x}, ax.axis, ax.nanpol, ax.keepdims, res);
    }
    NDA y;
    if (!get_nda(a, n, 1, y)) return fail("`cdf` must be a distribution name or an array (a callable is not supported)");
    if (x.isint && y.isint) T.outint |= 4u;
    T.core = [alternative, method](vector<vector<double>> &xs, double *out, unsigned) -> int {
        return ks2_core(xs[0], xs[1], alternative, method, out);
    };
    return run_test(T, {x, y}, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* ================================================================ anderson (_morestats.py) */

/* numpy.mean / numpy.std(ddof) of a 1-D array (pairwise sums) */
double np_mean(const vector<double> &x) { return psum(x) / (double)x.size(); }
double np_var(const vector<double> &x, double ddof)
{
    const double m = np_mean(x);
    vector<double> d(x.size());
    for (size_t i = 0; i < x.size(); i++) { d[i] = x[i] - m; d[i] = d[i] * d[i]; }
    double rc = (double)x.size() - ddof;
    if (rc < 0) rc = 0;
    return psum(d) / rc;
}

/* numpy.interp for one point */
double np_interp(double x, const vector<double> &xp, const vector<double> &fp)
{
    const int64_t len = (int64_t)xp.size();
    if (x != x) return x;
    if (x < xp[0]) return fp[0];
    if (x > xp[len - 1]) return fp[len - 1];
    int64_t j = std::upper_bound(xp.begin(), xp.end(), x) - xp.begin() - 1;
    if (j == len - 1) return fp[j];
    if (xp[j] == x) return fp[j];
    const double slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    double r = slope * (x - xp[j]) + fp[j];
    if (r != r) {
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (r != r && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}

/* scipy.special.logsumexp of a 1-D array */
double logsumexp(const vector<double> &a)
{
    const size_t n = a.size();
    vector<double> e(n);
    for (size_t i = 0; i < n; i++) e[i] = std::exp(a[i]);
    const double out_inf = std::log(psum(e));
    double amax = a[0];
    bool nan = false;
    for (double v : a) { if (v != v) nan = true; else if (v > amax || amax != amax) amax = v; }
    if (nan) amax = NaN;
    vector<double> mask(n), ex(n);
    for (size_t i = 0; i < n; i++) mask[i] = a[i] == amax ? 1.0 : 0.0;
    const double m = psum(mask);
    for (size_t i = 0; i < n; i++) ex[i] = std::exp((mask[i] != 0 ? -INF : a[i]) - amax);
    double s = psum(ex);
    s = s == 0 ? s : s / m;
    const double sgn = (s + 1 > 0 ? 1.0 : (s + 1 < 0 ? -1.0 : (s + 1 == 0 ? 0.0 : NaN))) * (m > 0 ? 1.0 : (m < 0 ? -1.0 : 0.0));
    if (s < -1) s = -s - 2;
    double out = std::log1p(s) + std::log(std::fabs(m)) + amax;
    if (sgn < 0) out = NaN;
    return std::isfinite(out) ? out : out_inf;
}

/* gumbel_r.fit(data) -> loc, scale (the maximum likelihood equations solved with brentq) */
int gumbel_r_fit(const vector<double> &data, double *loc, double *scale)
{
    for (double v : data) if (!std::isfinite(v)) return fail("The data contains non-finite values.");
    const double dmean = np_mean(data);
    auto func = [&](double sc_) {
        const size_t n = data.size();
        vector<double> sdata(n);
        for (size_t i = 0; i < n; i++) sdata[i] = -data[i] / sc_;
        double mx = sdata[0];
        for (double v : sdata) if (v > mx) mx = v;
        vector<double> w(n), xw(n);
        for (size_t i = 0; i < n; i++) w[i] = std::exp(sdata[i] - mx);
        const double scl = psum(w);
        for (size_t i = 0; i < n; i++) xw[i] = data[i] * w[i];
        const double wavg = psum(xw) / scl;
        return dmean - wavg - sc_;
    };
    auto sign = [](double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : (v == 0 ? 0.0 : NaN)); };
    double l = 1.0 / 2, r = 1.0 * 2;
    auto contains = [&](double a, double b) { const double sa = sign(func(a)), sb = sign(func(b)); return sa != sb; };
    while (!contains(l, r) && (l > 0 || r < INF)) {
        l /= 2;
        r *= 2;
    }
    int st;
    const double s_ = brentq(func, l, r, 1e-14, 1e-14, 100, &st);
    if (st < 0) return fail("f(a) and f(b) must have different signs");
    const size_t n = data.size();
    vector<double> nd(n);
    for (size_t i = 0; i < n; i++) nd[i] = -data[i] / s_;
    *scale = s_;
    *loc = -s_ * (logsumexp(nd) - std::log((double)n));
    return TSR_OK;
}

/* anderson's logistic fit: fsolve(rootfunc, [mean, std], xtol=1e-5) with MINPACK hybrd */
struct HybrdCtx { const vector<double> *x; double N; };
thread_local HybrdCtx *g_hybrd;

int logistic_rootfunc(int *n, double *ab, double *fvec, int *iflag)
{
    const vector<double> &xj = *g_hybrd->x;
    const size_t m = xj.size();
    const double a = ab[0], b = ab[1];
    vector<double> t1(m), t2(m);
    for (size_t i = 0; i < m; i++) {
        const double tmp = (xj[i] - a) / b;
        const double tmp2 = std::exp(tmp);
        t1[i] = 1.0 / (1 + tmp2);
        t2[i] = tmp * (1.0 - tmp2) / (1 + tmp2);
    }
    fvec[0] = psum(t1) - 0.5 * g_hybrd->N;
    fvec[1] = psum(t2) + g_hybrd->N;
    return 0;
}

void fsolve_logistic(const vector<double> &x, double N, double *sol)
{
    HybrdCtx ctx{&x, N};
    g_hybrd = &ctx;
    const int n = 2;
    double xs[2] = {sol[0], sol[1]}, fvec[2], diag[2], fjac[4], r[3], qtf[2], wa[8];
    int info = 0, nfev = 0;
    /* _root_hybr: epsfcn = eps, ml = mu = n - 1, maxfev = 200 * (n + 1), factor 100, mode 1 (diag None) */
    HYBRD(logistic_rootfunc, n, xs, fvec, 1e-5, 200 * (n + 1), n - 1, n - 1, EPS, diag, 1, 100.0, 0, &info, &nfev,
          fjac, n, r, 3, qtf, wa, wa + 2, wa + 4, wa + 6);
    g_hybrd = nullptr;
    sol[0] = xs[0];
    sol[1] = xs[1];
}

/* anderson(x, dist='norm', *, method=None) */
int r_anderson(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA xa;
    if (!get_nda(a, n, 0, xa)) return fail("x must be an array");
    if (xa.shape.size() > 1) return fail("anderson: only 1-D samples are supported");
    string dist = get_str(a, n, 1, "norm");
    for (auto &c : dist) c = (char)std::tolower((unsigned char)c);
    if (dist == "extreme1" || dist == "gumbel") dist = "gumbel_l";
    if (dist == "weibull_min")
        return fail("anderson: dist='weibull_min' is not supported (it needs weibull_min.fit and optimize.root)");
    if (dist != "norm" && dist != "expon" && dist != "gumbel_l" && dist != "gumbel_r" && dist != "logistic")
        return fail("Invalid distribution; dist must be in {'norm', 'expon', 'gumbel_l', 'gumbel_r', 'logistic', 'weibull_min'}.");
    const bool interp = has(a, n, 2);
    if (interp) {
        if (a[2].kind != 2 || string(a[2].str) != "interpolate")
            return fail("`method` must be either 'interpolate' or an instance of `MonteCarloMethod`.");
    }
    const vector<double> &x = xa.v;
    vector<double> y = x;
    sort_nan_last(y);
    const int64_t N = (int64_t)y.size();
    const double Nd = (double)N;
    double xbar = np_mean(x);
    vector<double> logcdf(N), logsf(N), sig, Avals, critical;
    double denom;
    if (dist == "norm") {
        const double s = std::sqrt(np_var(x, 1));
        const tsd::Dist *nd = tsd::find("norm");
        for (int64_t i = 0; i < N; i++) {
            const double w = (y[i] - xbar) / s;
            logcdf[i] = tsd::call(nd, DM_LOGCDF, w, {0.0, 1.0});
            logsf[i] = tsd::call(nd, DM_LOGSF, w, {0.0, 1.0});
        }
        sig = {15, 10, 5, 2.5, 1};
        Avals = {0.561, 0.631, 0.752, 0.873, 1.035};
        denom = 1.0 + 0.75 / Nd + 2.25 / Nd / Nd;
    } else if (dist == "expon") {
        const tsd::Dist *ed = tsd::find("expon");
        for (int64_t i = 0; i < N; i++) {
            const double w = y[i] / xbar;
            logcdf[i] = tsd::call(ed, DM_LOGCDF, w, {0.0, 1.0});
            logsf[i] = tsd::call(ed, DM_LOGSF, w, {0.0, 1.0});
        }
        sig = {15, 10, 5, 2.5, 1};
        Avals = {0.916, 1.062, 1.321, 1.591, 1.959};
        denom = 1.0 + 0.6 / Nd;
    } else if (dist == "logistic") {
        double sol[2] = {xbar, std::sqrt(np_var(x, 1))};
        fsolve_logistic(x, Nd, sol);
        for (int64_t i = 0; i < N; i++) {
            const double w = (y[i] - sol[0]) / sol[1];
            logcdf[i] = sc::log_expit(w);
            logsf[i] = sc::log_expit(-w);
        }
        sig = {25, 10, 5, 2.5, 1, 0.5};
        Avals = {0.426, 0.563, 0.66, 0.769, 0.906, 1.01};
        denom = 1.0 + 0.25 / Nd;
    } else {
        double loc, scale;
        int rc;
        if (dist == "gumbel_r") {
            rc = gumbel_r_fit(x, &loc, &scale);
        } else {
            vector<double> neg(x.size());
            for (size_t i = 0; i < x.size(); i++) neg[i] = -x[i];
            rc = gumbel_r_fit(neg, &loc, &scale);
            loc = -loc;
        }
        if (rc != TSR_OK) return rc;
        for (int64_t i = 0; i < N; i++) {
            const double w = (y[i] - loc) / scale;
            if (dist == "gumbel_r") {
                logcdf[i] = -std::exp(-w);
                const double median = -std::log(-std::log(0.5));
                logsf[i] = w > median ? std::log(-sc::expm1(-std::exp(-w))) : std::log1p(-std::exp(-std::exp(-w)));
            } else {
                const double median = std::log(-sc::log1p(-0.5));
                logcdf[i] = w < median ? std::log(-sc::expm1(-std::exp(w))) : std::log1p(-std::exp(-std::exp(w)));
                logsf[i] = -std::exp(w);
            }
        }
        sig = {25, 10, 5, 2.5, 1};
        Avals = {0.474, 0.637, 0.757, 0.877, 1.038};
        denom = 1.0 + 0.2 / std::sqrt(Nd);
    }
    for (double v : Avals) critical.push_back(np_around(v / denom, 3));
    vector<double> terms(N);
    for (int64_t i = 0; i < N; i++) terms[i] = (2 * (double)(i + 1) - 1.0) / Nd * (logcdf[i] + logsf[N - 1 - i]);
    const double A2 = -Nd - psum(terms);
    if (interp) {
        vector<double> s100;
        for (double v : sig) s100.push_back(v / 100);
        fn_result_num(&res[0], A2);
        fn_result_num(&res[3], np_interp(A2, critical, s100));
        return TSR_OK;
    }
    fn_result_num(&res[0], A2);
    int64_t k = (int64_t)critical.size();
    double *c = (double *)fn_result_array(&res[1], TSR_F64, 1, &k);
    double *sg = (double *)fn_result_array(&res[2], TSR_F64, 1, &k);
    if (!c || !sg) return TSR_ENOMEM;
    for (int64_t i = 0; i < k; i++) { c[i] = critical[i]; sg[i] = sig[i]; }
    return TSR_OK;
}

/* ================================================================ anderson_ksamp */

inline bool lt_nan(double a, double b) { return a < b || (b != b && a == a); }
int64_t ssl(const vector<double> &a, double x) { return std::lower_bound(a.begin(), a.end(), x, lt_nan) - a.begin(); }
int64_t ssr(const vector<double> &a, double x) { return std::upper_bound(a.begin(), a.end(), x, lt_nan) - a.begin(); }

/* least squares c = argmin |A c - y| (Householder QR in long double): numpy.polyfit(x, y, 2) */
vector<double> polyfit2(const vector<double> &x, const vector<double> &y)
{
    const int m = (int)x.size(), nc = 3;
    vector<double> lhs(m * nc), scale(nc, 0.0);
    for (int i = 0; i < m; i++) { lhs[i * nc + 0] = x[i] * x[i]; lhs[i * nc + 1] = x[i]; lhs[i * nc + 2] = 1.0; }
    for (int j = 0; j < nc; j++) {
        double s = 0;
        for (int i = 0; i < m; i++) s += lhs[i * nc + j] * lhs[i * nc + j];
        scale[j] = std::sqrt(s);
    }
    vector<long double> A(m * nc), b(m);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < nc; j++) A[i * nc + j] = lhs[i * nc + j] / scale[j];
        b[i] = y[i];
    }
    for (int j = 0; j < nc; j++) {
        long double nrm = 0;
        for (int i = j; i < m; i++) nrm += A[i * nc + j] * A[i * nc + j];
        nrm = std::sqrt(nrm);
        if (nrm == 0) continue;
        const long double alpha = A[j * nc + j] > 0 ? -nrm : nrm;
        vector<long double> v(m, 0);
        for (int i = j; i < m; i++) v[i] = A[i * nc + j];
        v[j] -= alpha;
        long double vv = 0;
        for (int i = j; i < m; i++) vv += v[i] * v[i];
        if (vv == 0) continue;
        for (int c = j; c < nc; c++) {
            long double d = 0;
            for (int i = j; i < m; i++) d += v[i] * A[i * nc + c];
            d = 2 * d / vv;
            for (int i = j; i < m; i++) A[i * nc + c] -= d * v[i];
        }
        long double d = 0;
        for (int i = j; i < m; i++) d += v[i] * b[i];
        d = 2 * d / vv;
        for (int i = j; i < m; i++) b[i] -= d * v[i];
    }
    vector<long double> cz(nc);
    for (int j = nc - 1; j >= 0; j--) {
        long double s = b[j];
        for (int c = j + 1; c < nc; c++) s -= A[j * nc + c] * cz[c];
        cz[j] = s / A[j * nc + j];
    }
    vector<double> c(nc);
    for (int j = 0; j < nc; j++) c[j] = (double)cz[j] / scale[j];
    return c;
}

/* anderson_ksamp(samples, midrank=<no value>, *, variant=<no value>, method=None); samples: 2-D, one per row */
int r_anderson_ksamp(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA sa;
    if (!get_nda(a, n, 0, sa)) return fail("samples must be an array (one sample per row)");
    if (has(a, n, 3)) return fail("anderson_ksamp: `method` (permutation test) is not supported");
    if (sa.shape.size() < 1) return fail("anderson_ksamp needs at least two samples");
    const int64_t k = sa.shape[0];
    if (k < 2) return fail("anderson_ksamp needs at least two samples");
    const int64_t per = k ? sa.size() / k : 0;
    vector<vector<double>> samples(k);
    for (int64_t i = 0; i < k; i++) samples[i].assign(sa.v.begin() + i * per, sa.v.begin() + (i + 1) * per);
    vector<double> Z = sa.v;
    sort_nan_last(Z);
    const int64_t N = (int64_t)Z.size();
    vector<double> Zstar;
    for (int64_t i = 0; i < N; i++) {
        if (i == 0) { Zstar.push_back(Z[i]); continue; }
        const double p = Zstar.back();
        if (Z[i] == p || (Z[i] != Z[i] && p != p)) continue;
        Zstar.push_back(Z[i]);
    }
    if (Zstar.size() < 2) return fail("anderson_ksamp needs more than one distinct observation");
    if (per == 0) return fail("anderson_ksamp encountered sample without observations");
    const bool midrank_given = has(a, n, 1);
    const bool variant_given = has(a, n, 2);
    string variant;
    bool return_crit = false;
    if (!variant_given) {
        return_crit = true;
        variant = (!midrank_given || get_bool(a, n, 1, true)) ? "midrank" : "right";
    } else {
        variant = get_str(a, n, 2, "");
    }
    const double Nd = (double)N;
    const int64_t L = (int64_t)Zstar.size();
    double A2kN = 0.;
    if (variant == "midrank") {
        vector<double> lj(L), Bj(L);
        const bool allunique = N == L;
        for (int64_t j = 0; j < L; j++) {
            const int64_t left = ssl(Z, Zstar[j]);
            lj[j] = allunique ? 1.0 : (double)(ssr(Z, Zstar[j]) - left);
            Bj[j] = (double)left + lj[j] / 2.;
        }
        for (int64_t i = 0; i < k; i++) {
            vector<double> s = samples[i];
            sort_nan_last(s);
            vector<double> inner(L);
            for (int64_t j = 0; j < L; j++) {
                const int64_t r = ssr(s, Zstar[j]);
                double Mij = (double)r;
                const int64_t fij = r - ssl(s, Zstar[j]);
                Mij -= (double)fij / 2.;
                const double t = Nd * Mij - Bj[j] * (double)per;
                inner[j] = lj[j] / Nd * (t * t) / (Bj[j] * (Nd - Bj[j]) - Nd * lj[j] / 4.);
            }
            A2kN += psum(inner) / (double)per;
        }
        A2kN *= (Nd - 1.) / Nd;
    } else if (variant == "right") {
        vector<int64_t> lj(L - 1), Bj(L - 1);
        int64_t cum = 0;
        for (int64_t j = 0; j < L - 1; j++) {
            lj[j] = ssr(Z, Zstar[j]) - ssl(Z, Zstar[j]);
            cum += lj[j];
            Bj[j] = cum;
        }
        for (int64_t i = 0; i < k; i++) {
            vector<double> s = samples[i];
            sort_nan_last(s);
            vector<double> inner(L - 1);
            for (int64_t j = 0; j < L - 1; j++) {
                const int64_t Mij = ssr(s, Zstar[j]);
                const int64_t t = N * Mij - Bj[j] * per;
                inner[j] = (double)lj[j] / Nd * (double)(t * t) / (double)(Bj[j] * (N - Bj[j]));
            }
            A2kN += psum(inner) / (double)per;
        }
    } else if (variant == "continuous") {
        for (int64_t i = 0; i < k; i++) {
            vector<double> s = samples[i];
            sort_nan_last(s);
            vector<double> inner(N - 1);
            for (int64_t j = 1; j < N; j++) {
                const int64_t Mij = ssr(s, Z[j - 1]);
                const int64_t t = N * Mij - j * per;
                inner[j - 1] = (double)(t * t) / (double)(j * (N - j));
            }
            A2kN += psum(inner) / (double)per;
        }
        A2kN = A2kN / Nd;
    } else {
        return fail("`variant` must be one of 'midrank', 'right', or 'continuous'.");
    }
    /* H = (1. / n).sum(); h, g from the harmonic sums */
    vector<double> invn(k, 1. / (double)per);
    const double H = psum(invn);
    if (N - 1 <= 1) return fail("index -1 is out of bounds for axis 0 with size 0");
    vector<double> hs_cs, hsg;
    double cum = 0;
    for (int64_t v = N - 1; v > 1; v--) {
        cum = hs_cs.empty() ? 1. / (double)v : cum + 1. / (double)v;
        hs_cs.push_back(cum);
    }
    const double h = hs_cs.back() + 1;
    for (size_t i = 0; i < hs_cs.size(); i++) hsg.push_back(hs_cs[i] / (double)(i + 2));
    const double g = psum(hsg);
    const double kd = (double)k, k2 = (double)(k * k);
    const double A = (4 * g - 6) * (double)(k - 1) + (10 - 6 * g) * H;
    const double Bq = (2 * g - 4) * k2 + 8 * h * kd + (2 * g - 14 * h - 4) * H - 8 * h + 4 * g - 6;
    const double C = (6 * h + 2 * g - 2) * k2 + (4 * h - 4 * g + 6) * kd + (2 * h - 6) * H + 4 * h;
    const double D = (2 * h + 6) * k2 - 4 * h * kd;
    const double sigmasq = (A * (double)(N * N * N) + Bq * (double)(N * N) + C * Nd + D) / ((Nd - 1.) * (Nd - 2.) * (Nd - 3.));
    const int64_t m = k - 1;
    if (sigmasq < 0) return fail("math domain error");
    const double A2 = (A2kN - (double)m) / std::sqrt(sigmasq);
    const double b0[7] = {0.675, 1.281, 1.645, 1.96, 2.326, 2.573, 3.085};
    const double b1[7] = {-0.245, 0.25, 0.678, 1.149, 1.822, 2.364, 3.615};
    const double b2[7] = {-0.105, -0.305, -0.362, -0.391, -0.396, -0.345, -0.154};
    const double sig[7] = {0.25, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001};
    vector<double> critical(7);
    double cmin = INF, cmax = -INF;
    for (int i = 0; i < 7; i++) {
        critical[i] = b0[i] + b1[i] / std::sqrt((double)m) + b2[i] / (double)m;
        cmin = std::min(cmin, critical[i]);
        cmax = std::max(cmax, critical[i]);
    }
    double p;
    if (A2 < cmin) p = 0.25;
    else if (A2 > cmax) p = 0.001;
    else {
        vector<double> ls(7);
        for (int i = 0; i < 7; i++) ls[i] = std::log(sig[i]);
        const vector<double> pf = polyfit2(critical, ls);
        double yv = 0;
        for (double c : pf) yv = yv * A2 + c;
        p = std::exp(yv);
    }
    fn_result_num(&res[0], A2);
    if (return_crit) {
        int64_t seven = 7;
        double *c = (double *)fn_result_array(&res[1], TSR_F64, 1, &seven);
        if (!c) return TSR_ENOMEM;
        for (int i = 0; i < 7; i++) c[i] = critical[i];
    }
    fn_result_num(&res[2], p);
    return TSR_OK;
}

/* ================================================================ epps_singleton_2samp */

/* eigen-decomposition of a symmetric matrix (cyclic Jacobi, long double): A = V diag(w) V^T */
void sym_eig(vector<long double> A, int n, vector<long double> &w, vector<long double> &V)
{
    V.assign(n * n, 0);
    for (int i = 0; i < n; i++) V[i * n + i] = 1;
    for (int sweep = 0; sweep < 100; sweep++) {
        long double off = 0;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) off += A[i * n + j] * A[i * n + j];
        if (off == 0) break;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++) {
                const long double apq = A[p * n + q];
                if (apq == 0) continue;
                const long double theta = (A[q * n + q] - A[p * n + p]) / (2 * apq);
                const long double t = (theta >= 0 ? 1 : -1) / (std::fabs(theta) + std::sqrt(theta * theta + 1));
                const long double c = 1 / std::sqrt(t * t + 1), s = t * c;
                for (int k = 0; k < n; k++) {
                    const long double akp = A[k * n + p], akq = A[k * n + q];
                    A[k * n + p] = c * akp - s * akq;
                    A[k * n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; k++) {
                    const long double apk = A[p * n + k], aqk = A[q * n + k];
                    A[p * n + k] = c * apk - s * aqk;
                    A[q * n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; k++) {
                    const long double vkp = V[k * n + p], vkq = V[k * n + q];
                    V[k * n + p] = c * vkp - s * vkq;
                    V[k * n + q] = s * vkp + c * vkq;
                }
            }
    }
    w.resize(n);
    for (int i = 0; i < n; i++) w[i] = A[i * n + i];
}

/* scipy.stats.quantile(method='linear') of a sorted sample at probability p */
double quantile_linear(const vector<double> &ys, double p)
{
    const double n = (double)ys.size();
    const double m = 1 - p;
    const double jg = p * n + m;
    const double jp1 = np_floordiv(jg, 1);
    double j = jp1 - 1;
    double g = np_mod(jg, 1);
    if (j < 0) g = 0;
    const double jc = std::min(std::max(j, 0.), n - 1), jp1c = std::min(std::max(jp1, 0.), n - 1);
    return (1 - g) * ys[(int64_t)jc] + g * ys[(int64_t)jp1c];
}

int core_epps(vector<vector<double>> &xs, const vector<double> &t, double *out, unsigned flags)
{
    vector<double> x = xs[0], y = xs[1];
    const int64_t nx = (int64_t)x.size(), ny = (int64_t)y.size();
    if (nx < 5 || ny < 5) return fail("x and y should have at least 5 elements");
    const int64_t n = nx + ny;
    for (double v : t) if (v <= 0) return fail("t must contain positive elements only.");
    bool invalid = false;
    for (auto &v : x) if (!std::isfinite(v)) { v = 1.; invalid = true; }
    for (auto &v : y) if (!std::isfinite(v)) { v = 1.; invalid = true; }
    vector<double> z = x;
    z.insert(z.end(), y.begin(), y.end());
    sort_nan_last(z);
    const double sigma = (quantile_linear(z, 0.75) - quantile_linear(z, 0.25)) / 2;
    const int T = (int)t.size();
    const int d = 2 * T;
    vector<double> ts(T);
    for (int j = 0; j < T; j++) ts[j] = t[j] / sigma;
    auto build = [&](const vector<double> &v, vector<double> &G) {
        const int64_t m = (int64_t)v.size();
        G.assign(d * m, 0.0);
        for (int j = 0; j < T; j++)
            for (int64_t i = 0; i < m; i++) {
                const double a = ts[j] * v[i];
                G[j * m + i] = std::cos(a);
                G[(T + j) * m + i] = std::sin(a);
            }
    };
    vector<double> gx, gy;
    build(x, gx);
    build(y, gy);
    auto rowmean = [&](const vector<double> &G, int64_t m, int r) {
        vector<double> row(G.begin() + r * m, G.begin() + (r + 1) * m);
        return asum(row, flags) / (double)m;
    };
    /* biased covariance: np.cov (1-D input) or xpx.cov's N-d branch (divides by fact) */
    auto cov = [&](const vector<double> &G, int64_t m, vector<long double> &C) {
        vector<double> X = G;
        for (int r = 0; r < d; r++) {
            const double mu = rowmean(G, m, r);
            for (int64_t i = 0; i < m; i++) X[r * m + i] -= mu;
        }
        C.assign(d * d, 0);
        const double fact = (double)(m - 1);
        for (int r = 0; r < d; r++)
            for (int c = 0; c < d; c++) {
                long double s = 0;
                for (int64_t i = 0; i < m; i++) s += (long double)X[r * m + i] * X[c * m + i];
                double v = (double)s;
                if (flags & FN_VECND) v /= fact;
                else v *= 1.0 / fact;
                v = v * (double)(m - 1) / (double)m;
                C[r * d + c] = v;
            }
    };
    vector<long double> cx, cy;
    cov(gx, nx, cx);
    cov(gy, ny, cy);
    vector<long double> E(d * d);
    const double fx = (double)n / (double)nx, fy = (double)n / (double)ny;
    for (int i = 0; i < d * d; i++) E[i] = (double)(fx * (double)cx[i]) + (double)(fy * (double)cy[i]);
    for (int i = 0; i < d; i++) for (int j = i + 1; j < d; j++) { const long double s = (E[i * d + j] + E[j * d + i]) / 2; E[i * d + j] = E[j * d + i] = s; }
    vector<long double> w, V;
    sym_eig(E, d, w, V);
    long double smax = 0;
    for (auto v : w) smax = std::max(smax, (long double)std::fabs(v));
    const long double cutoff = 1e-15L * smax;
    vector<long double> winv(d, 0);
    long double imax = 0;
    for (int i = 0; i < d; i++) {
        if (std::fabs(w[i]) > cutoff) winv[i] = 1 / w[i];
        imax = std::max(imax, (long double)std::fabs(winv[i]));
    }
    int rank = 0;
    for (int i = 0; i < d; i++) if (std::fabs(winv[i]) > imax * d * (long double)EPS) rank++;
    vector<long double> gd(d);
    for (int r = 0; r < d; r++) gd[r] = (long double)(rowmean(gx, nx, r) - rowmean(gy, ny, r));
    long double q = 0;
    for (int i = 0; i < d; i++) {
        long double proj = 0;
        for (int r = 0; r < d; r++) proj += V[r * d + i] * gd[r];
        q += winv[i] * proj * proj;
    }
    double W = (double)n * (double)q;
    if (std::max(nx, ny) < 25) {
        const double corr = 1.0 / (1.0 + std::pow((double)n, -0.45) + 10.1 * (std::pow((double)nx, -1.7) + std::pow((double)ny, -1.7)));
        W *= corr;
    }
    double p = sc::chdtrc((double)rank, W);
    if (invalid) { W = NaN; p = NaN; }
    out[0] = W;
    out[1] = p;
    return TSR_OK;
}

/* epps_singleton_2samp(x, y, t=(0.4, 0.8), *, axis=0, nan_policy='propagate', keepdims=False) */
int r_epps_singleton_2samp(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x, y;
    if (!get_nda(a, n, 0, x) || !get_nda(a, n, 1, y)) return fail("x and y must be arrays");
    vector<double> t = {0.4, 0.8};
    if (has(a, n, 2)) {
        NDA ta;
        if (!get_nda(a, n, 2, ta)) return fail("t must be an array");
        if (ta.shape.size() > 1) return fail("t must be 1d");
        t = ta.v;
    }
    AxisArgs ax;
    int rc = axis_args(a, n, 3, 4, 5, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 2;
    T.too_small = 4;
    T.vectorized = true;
    T.core = [&t](vector<vector<double>> &xs, double *out, unsigned flags) -> int { return core_epps(xs, t, out, flags); };
    return run_test(T, {x, y}, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* ================================================================ alexandergovern */

int core_alexandergovern(vector<vector<double>> &xs, double *out, unsigned flags)
{
    const int k = (int)xs.size();
    if (k < 2) return fail("2 or more inputs required");
    for (auto &s : xs) if (s.size() <= 1) return fail("Input sample size must be greater than one.");
    vector<double> means(k), se2(k), se(k), inv(k);
    for (int i = 0; i < k; i++) {
        const vector<double> &s = xs[i];
        const double len = (double)s.size();
        const double mu = asum(s, flags) / len;
        vector<double> dd(s.size());
        for (size_t j = 0; j < s.size(); j++) { const double d = s[j] - mu; dd[j] = d * d; }
        double var = asum(dd, flags) / len;
        const double nc = len - 1;
        var *= nc > 0 ? len / nc : NaN;
        means[i] = mu;
        se2[i] = var / len;
        se[i] = std::sqrt(se2[i]);
        if (se[i] <= std::fabs(EPS * mu)) se[i] = NaN;
        inv[i] = 1 / se2[i];
    }
    /* sums over the stacked samples: 1-D -> pairwise, N-d (axis 0 of a stack) -> sequential */
    auto ksum = [&](const vector<double> &v) { return (flags & FN_VECND) ? seqsum(v.data(), k) : psum(v); };
    const double sinv = ksum(inv);
    vector<double> wm(k);
    for (int i = 0; i < k; i++) wm[i] = (inv[i] / sinv) * means[i];
    const double var_w = ksum(wm);
    double A = 0;
    for (int i = 0; i < k; i++) {
        const double t = (means[i] - var_w) / se[i];
        const double v = (double)xs[i].size() - 1;
        const double aa = v - .5;
        const double b = 48 * (aa * aa);
        const double c = std::sqrt(aa * std::log(1 + (t * t) / v));
        const double c3 = std::pow(c, 3), c5 = std::pow(c, 5), c7 = std::pow(c, 7), c4 = std::pow(c, 4);
        const double z = c + ((c3 + 3 * c) / b) - ((4 * c7 + 33 * c5 + 240 * c3 + 855 * c) / ((b * b) * 10 + 8 * b * c4 + 1000 * b));
        A = std::fma(z, z, A);
    }
    out[0] = A;
    out[1] = sc::chdtrc((double)(k - 1), A);
    return TSR_OK;
}

/* alexandergovern(*samples, nan_policy='propagate', axis=0, keepdims=False): samples as sample1, sample2, ... */
const int AG_MAX = 10;
int r_alexandergovern(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    vector<NDA> samples;
    for (int k = 0; k < AG_MAX; k++) {
        if (!has(a, n, k)) continue;
        NDA s;
        if (!get_nda(a, n, k, s)) return fail("samples must be arrays");
        samples.push_back(s);
    }
    if (samples.size() < 2) return fail("2 or more inputs required");
    AxisArgs ax;
    int rc = axis_args(a, n, AG_MAX + 1, AG_MAX, AG_MAX + 2, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 2;
    T.too_small = 1;
    T.vectorized = true;
    T.core = core_alexandergovern;
    return run_test(T, samples, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* ================================================================ combine_pvalues */

int r_combine_pvalues(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA p, w;
    if (!get_nda(a, n, 0, p)) return fail("pvalues must be an array");
    const string method = get_str(a, n, 1, "fisher");
    const bool haw = has(a, n, 2);
    if (haw && !get_nda(a, n, 2, w)) return fail("weights must be an array");
    AxisArgs ax;
    int rc = axis_args(a, n, 3, 4, 5, "0", ax);
    if (rc != TSR_OK) return rc;
    Test T;
    T.nout = 2;
    T.vectorized = true;
    T.paired = true;
    T.core = [method](vector<vector<double>> &xs, double *out, unsigned flags) -> int {
        const vector<double> &pv = xs[0];
        const size_t m = pv.size();
        const double nn = (double)m;
        vector<double> t(m), u(m);
        if (method == "fisher") {
            for (size_t i = 0; i < m; i++) t[i] = std::log(pv[i]);
            const double stat = -2 * asum(t, flags);
            out[0] = stat;
            out[1] = sc::chdtrc(2 * nn, stat);
        } else if (method == "pearson") {
            for (size_t i = 0; i < m; i++) t[i] = std::log1p(-pv[i]);
            const double stat = 2 * asum(t, flags);
            out[0] = stat;
            out[1] = sc::chdtr(2 * nn, -stat);
        } else if (method == "mudholkar_george") {
            const double nf = std::sqrt(3 / nn) / PI;
            for (size_t i = 0; i < m; i++) { t[i] = std::log(pv[i]); u[i] = std::log1p(-pv[i]); }
            const double stat = -asum(t, flags) + asum(u, flags);
            const double nu = 5 * nn + 4;
            const double af = std::sqrt(nu / (nu - 2));
            out[0] = stat;
            out[1] = sc::stdtr(nu, -(stat * nf * af));
        } else if (method == "tippett") {
            double mn = m ? pv[0] : NaN;
            for (double v : pv) { if (v != v) { mn = v; break; } if (v < mn) mn = v; }
            out[0] = mn;
            out[1] = sc::betainc(1.0, nn, mn);
        } else if (method == "stouffer") {
            vector<double> wt = xs.size() > 1 ? xs[1] : vector<double>(m, 1.0);
            vector<double> ww(m);
            for (size_t i = 0; i < m; i++) { t[i] = wt[i] * -sc::ndtri(pv[i]); ww[i] = wt[i] * wt[i]; }
            const double stat = asum(t, flags) / std::sqrt(asum(ww, flags));
            out[0] = stat;
            out[1] = sc::ndtr(-stat);
        } else {
            return fail("Invalid method. Valid methods are 'fisher', 'pearson', 'mudholkar_george', 'tippett', and 'stouffer'");
        }
        return TSR_OK;
    };
    vector<NDA> s = {p};
    if (haw) s.push_back(w);
    return run_test(T, s, ax.axis, ax.nanpol, ax.keepdims, res);
}

/* ================================================================ page_trend_test (_page_trend_test.py) */

/* the exact null distribution of Page's L: p(l, k, n) by [5] Equation 1 (the recursion SciPy evaluates) */
struct PageL {
    int64_t k, a, b;
    std::map<int64_t, std::map<int64_t, double>> tab;   /* n -> l -> pmf */
    explicit PageL(int64_t k_) : k(k_), a((k_ * (k_ + 1) * (k_ + 2)) / 6), b((k_ * (k_ + 1) * (2 * k_ + 1)) / 6) {}
    int init1()
    {
        if (k > 11) return fail("page_trend_test: the exact null distribution needs all permutations of the columns (too many)");
        vector<int64_t> perm(k);
        std::iota(perm.begin(), perm.end(), 1);
        vector<int64_t> counts(b - a + 1, 0);
        do {
            int64_t L = 0;
            for (int64_t i = 0; i < k; i++) L += (i + 1) * perm[i];
            counts[L - a]++;
        } while (std::next_permutation(perm.begin(), perm.end()));
        double fact = 1;
        for (int64_t i = 2; i <= k; i++) fact *= (double)i;
        for (int64_t l = a; l <= b; l++) tab[1][l] = (double)counts[l - a] / fact;
        return TSR_OK;
    }
    double pmf(int64_t l, int64_t n)
    {
        auto &t = tab[n];
        auto it = t.find(l);
        if (it != t.end()) return it->second;
        if (n == 1) return 0.0;
        double p = 0;
        const int64_t low = std::max(l - (n - 1) * b, a), high = std::min(l - (n - 1) * a, b);
        for (int64_t s = low; s <= high; s++) p += pmf(l - s, n - 1) * pmf(s, 1);
        tab[n][l] = p;
        return p;
    }
};

/* page_trend_test(data, ranked=False, predicted_ranks=None, method='auto') */
int r_page_trend_test(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA d;
    if (!get_nda(a, n, 0, d)) return fail("data must be an array");
    const string method = get_str(a, n, 3, "auto");
    if (method != "asymptotic" && method != "exact" && method != "auto") return fail("`method` must be in {'asymptotic', 'exact', 'auto'}");
    if (d.shape.size() != 2) return fail("`data` must be a 2d array.");
    const int64_t m = d.shape[0], nc = d.shape[1];
    if (m < 2 || nc < 3) return fail("Page's L is only appropriate for data with two or more rows and three or more columns.");
    for (double v : d.v) if (v != v) return fail("`data` contains NaNs, which cannot be ranked meaningfully");
    const bool ranked = get_bool(a, n, 1, false);
    if (has(a, n, 1) && a[1].kind != 4) return fail("`ranked` must be boolean.");
    vector<double> ranks(m * nc);
    bool intranks = false;
    if (ranked) {
        double mn = INF, mx = -INF;
        for (double v : d.v) { mn = std::min(mn, v); mx = std::max(mx, v); }
        if (!(mn >= 1 && mx <= (double)nc)) return fail("`data` is not properly ranked. Rank the data or pass `ranked=False`.");
        ranks = d.v;
        intranks = d.isint;
    } else {
        for (int64_t i = 0; i < m; i++) {
            vector<double> row(d.v.begin() + i * nc, d.v.begin() + (i + 1) * nc);
            const vector<double> r = rank_average(row);
            std::copy(r.begin(), r.end(), ranks.begin() + i * nc);
        }
    }
    vector<double> pred(nc);
    if (has(a, n, 2)) {
        NDA pr;
        if (!get_nda(a, n, 2, pr)) return fail("`predicted_ranks` must be an array");
        bool ok = pr.shape.size() >= 1 && (int64_t)pr.v.size() == nc;
        if (ok) {
            vector<char> seen(nc + 1, 0);
            for (double v : pr.v) {
                if (v != std::floor(v) || v < 1 || v > nc || seen[(int64_t)v]) { ok = false; break; }
                seen[(int64_t)v] = 1;
            }
        }
        if (!ok) return fail("`predicted_ranks` must include each integer from 1 to n (the number of columns in `data`) exactly once.");
        pred = pr.v;
    } else {
        for (int64_t j = 0; j < nc; j++) pred[j] = (double)(j + 1);
    }
    /* L = sum over columns of predicted_rank * column sum (column sums accumulate row by row) */
    vector<double> colsums(nc, 0.0), prod(nc);
    for (int64_t j = 0; j < nc; j++) {
        double s = ranks[j];
        for (int64_t i = 1; i < m; i++) s += ranks[i * nc + j];
        colsums[j] = s;
        prod[j] = pred[j] * s;
    }
    const double L = psum(prod);
    string meth = method;
    if (meth == "auto") meth = (nc > 8 || (m > 12 && nc > 3) || m > 20) ? "asymptotic" : "exact";
    double p;
    if (meth == "asymptotic") {
        const double E0 = (double)(m * nc * (nc + 1) * (nc + 1)) / 4;
        const double V0 = (double)(m * nc * nc * (nc + 1) * (nc * nc - 1)) / 144;
        const double Lambda = (L - E0) / std::sqrt(V0);
        p = sc::ndtr(-Lambda);
    } else {
        const int64_t Li = (int64_t)L;
        PageL st(nc);
        int rc = st.init1();
        if (rc != TSR_OK) return rc;
        vector<double> ps;
        for (int64_t l = Li; l <= m * st.b; l++) ps.push_back(st.pmf(l, m));
        p = psum(ps);
    }
    if (intranks) fn_result_int(&res[0], (int64_t)L);
    else fn_result_num(&res[0], L);
    fn_result_num(&res[1], p);
    return TSR_OK;
}

/* ================================================================ poisson_means_test */

/* poisson_means_test(k1, n1, k2, n2, *, diff=0, alternative='two-sided') */
int r_poisson_means_test(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    for (int k = 0; k < 4; k++) if (!has(a, n, k) || (a[k].kind != 1 && a[k].kind != 4)) return fail("k1, n1, k2, n2 must be numbers");
    const double k1 = a[0].num, n1 = a[1].num, k2 = a[2].num, n2 = a[3].num;
    const double diff = get_num(a, n, 4, 0.0);
    const string alternative = get_str(a, n, 5, "two-sided");
    if (k1 != std::trunc(k1) || k2 != std::trunc(k2)) return fail("`k1` and `k2` must be integers.");
    if (k1 < 0 || k2 < 0) return fail("`k1` and `k2` must be greater than or equal to 0.");
    if (n1 <= 0 || n2 <= 0) return fail("`n1` and `n2` must be greater than 0.");
    if (diff < 0) return fail("diff must be greater than or equal to 0.");
    string low = alternative;
    for (auto &c : low) c = (char)std::tolower((unsigned char)c);
    if (low != "two-sided" && low != "less" && low != "greater") return fail("Alternative must be one of 'two-sided', 'less', 'greater'.");
    const double lmbd_hat2 = ((k1 + k2) / (n1 + n2) - diff * n1 / (n1 + n2));
    if (lmbd_hat2 <= 0) {
        fn_result_int(&res[0], 0);
        fn_result_int(&res[1], 1);
        return TSR_OK;
    }
    const double var = k1 / (n1 * n1) + k2 / (n2 * n2);
    const double t_k1k2 = (k1 / n1 - k2 / n2 - diff) / std::sqrt(var);
    const double nl1 = n1 * (lmbd_hat2 + diff), nl2 = n2 * lmbd_hat2;
    const tsd::Dist *po = tsd::find("poisson");
    const double x1_lb = tsd::call(po, DM_PPF, 1e-10, {nl1, 0.0}), x1_ub = tsd::call(po, DM_PPF, 1 - 1e-16, {nl1, 0.0});
    const double x2_lb = tsd::call(po, DM_PPF, 1e-10, {nl2, 0.0}), x2_ub = tsd::call(po, DM_PPF, 1 - 1e-16, {nl2, 0.0});
    vector<double> x1, x2;
    for (double v = x1_lb; v < x1_ub + 1; v += 1) x1.push_back(v);
    for (double v = x2_lb; v < x2_ub + 1; v += 1) x2.push_back(v);
    vector<double> p1(x1.size()), p2(x2.size());
    for (size_t i = 0; i < x1.size(); i++) p1[i] = tsd::call(po, DM_PDF, x1[i], {nl1, 0.0});
    for (size_t j = 0; j < x2.size(); j++) p2[j] = tsd::call(po, DM_PDF, x2[j], {nl2, 0.0});
    vector<double> sel;
    for (size_t j = 0; j < x2.size(); j++) {
        const double l2 = x2[j] / n2;
        for (size_t i = 0; i < x1.size(); i++) {
            const double l1 = x1[i] / n1;
            const double t = (l1 - l2 - diff) / std::sqrt(l1 / n1 + l2 / n2);
            bool ind;
            if (alternative == "two-sided") ind = std::fabs(t) >= std::fabs(t_k1k2);
            else if (alternative == "less") ind = t <= t_k1k2;
            else ind = t >= t_k1k2;
            if (ind) sel.push_back(p1[i] * p2[j]);
        }
    }
    fn_result_num(&res[0], t_k1k2);
    fn_result_num(&res[1], psum(sel));
    return TSR_OK;
}

/* ================================================================ quantile_test */

/* quantile_test(x, *, q=0, p=0.5, alternative='two-sided') */
int r_quantile_test(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x;
    if (!get_nda(a, n, 0, x) || x.shape.size() > 1) return fail("`x` must be a one-dimensional array of numbers.");
    if (has(a, n, 1) && a[1].kind != 1 && a[1].kind != 4) return fail("`q` must be a scalar.");
    const double q = get_num(a, n, 1, 0.0);
    if (has(a, n, 2) && a[2].kind != 1) return fail("`p` must be a float strictly between 0 and 1.");
    const double p = get_num(a, n, 2, 0.5);
    if (!(p < 1) || !(p > 0)) {
        if (p >= 1 || p <= 0) return fail("`p` must be a float strictly between 0 and 1.");
    }
    const string H1 = get_str(a, n, 3, "two-sided");
    if (H1 != "two-sided" && H1 != "less" && H1 != "greater") return fail("`alternative` must be one of {'two-sided', 'less', 'greater'}");
    int64_t T1 = 0, T2 = 0;
    for (double v : x.v) { T1 += v <= q; T2 += v < q; }
    const double nn = (double)x.v.size();
    const tsd::Dist *bi = tsd::find("binom");
    const double pless = tsd::call(bi, DM_SF, (double)(T2 - 1), {nn, p, 0.0});
    const double pgreater = tsd::call(bi, DM_CDF, (double)T1, {nn, p, 0.0});
    int64_t stat, type;
    double pv;
    if (H1 == "less") { pv = pless; stat = T2; type = 2; }
    else if (H1 == "greater") { pv = pgreater; stat = T1; type = 1; }
    else {
        /* np.argsort([cdf(T1), sf(T2 - 1)]): NaN sorts last, ties keep the order */
        const bool first = !(pless < pgreater || (pgreater != pgreater && pless == pless));
        pv = clip01(2 * (first ? pgreater : pless));
        if (first) { stat = T1; type = 1; }
        else { stat = T2; type = 2; }
    }
    fn_result_int(&res[0], stat);
    fn_result_int(&res[1], type);
    fn_result_num(&res[2], pv);
    return TSR_OK;
}

/* ================================================================ bws_test (exact permutation distribution) */

double bws_statistic(vector<double> Ri, vector<double> Hj, bool two_sided)
{
    std::sort(Ri.begin(), Ri.end());
    std::sort(Hj.begin(), Hj.end());
    const int64_t nn = (int64_t)Ri.size(), m = (int64_t)Hj.size();
    vector<double> bx(nn), by(m);
    for (int64_t k = 0; k < nn; k++) {
        const double i = (double)(k + 1);
        double num = Ri[k] - (double)(m + nn) / (double)nn * i;
        num *= two_sided ? num : std::fabs(num);
        const double den = i / (double)(nn + 1) * (1 - i / (double)(nn + 1)) * (double)m * (double)(m + nn) / (double)nn;
        bx[k] = num / den;
    }
    for (int64_t k = 0; k < m; k++) {
        const double j = (double)(k + 1);
        double num = Hj[k] - (double)(m + nn) / (double)m * j;
        num *= two_sided ? num : std::fabs(num);
        const double den = j / (double)(m + 1) * (1 - j / (double)(m + 1)) * (double)nn * (double)(m + nn) / (double)m;
        by[k] = num / den;
    }
    const double Bx = 1 / (double)nn * psum(bx);
    const double By = 1 / (double)m * psum(by);
    return two_sided ? (Bx + By) / 2 : (Bx - By) / 2;
}

/* bws_test(x, y, *, alternative='two-sided', method=None) */
int r_bws_test(const void *, const tsr_arg *a, int n, tsr_result *res, int)
{
    NDA x, y;
    if (!get_nda(a, n, 0, x) || !get_nda(a, n, 1, y)) return fail("x and y must be arrays");
    if (x.shape.size() > 1 || y.shape.size() > 1) return fail("`x` and `y` must be exactly one-dimensional.");
    if (any_nan(x.v) || any_nan(y.v)) return fail("`x` and `y` must not contain NaNs.");
    if (x.v.empty() || y.v.empty()) return fail("`x` and `y` must be of nonzero size.");
    string alternative = get_str(a, n, 2, "two-sided");
    for (auto &c : alternative) c = (char)std::tolower((unsigned char)c);
    if (alternative != "two-sided" && alternative != "less" && alternative != "greater")
        return fail("`alternative` must be one of {'two-sided', 'less', 'greater'}.");
    if (has(a, n, 3)) return fail("bws_test: `method` (a PermutationMethod) is not supported");
    const int64_t nx = (int64_t)x.v.size(), ny = (int64_t)y.v.size(), N = nx + ny;
    if (nx < 2 || ny < 2) return fail("each sample in `data` must contain two or more observations along `axis`.");
    vector<double> z = x.v;
    z.insert(z.end(), y.v.begin(), y.v.end());
    const vector<double> r = rank_average(z);
    /* the number of distinct partitions: an exact test needs n_resamples (9999) >= comb(N, nx) */
    long double nmax = 1;
    for (int64_t i = 1; i <= nx; i++) nmax = nmax * (long double)(ny + i) / (long double)i;
    const double n_max = (double)std::nearbyint((double)nmax);
    if (n_max > 9999)
        return fail("bws_test: more than 9999 distinct permutations; SciPy's randomized permutation test is not supported");
    const bool two = alternative == "two-sided";
    vector<double> rx(r.begin(), r.begin() + nx), ry(r.begin() + nx, r.end());
    const double observed = bws_statistic(rx, ry, two);
    /* itertools.combinations(range(N), nx) in lexicographic order */
    vector<double> nulld;
    vector<int64_t> c(nx);
    std::iota(c.begin(), c.end(), 0);
    while (true) {
        vector<double> px, py;
        vector<char> in(N, 0);
        for (auto i : c) { px.push_back(r[i]); in[i] = 1; }
        for (int64_t i = 0; i < N; i++) if (!in[i]) py.push_back(r[i]);
        nulld.push_back(bws_statistic(px, py, two));
        int64_t i = nx - 1;
        while (i >= 0 && c[i] == N - nx + i) i--;
        if (i < 0) break;
        c[i]++;
        for (int64_t j = i + 1; j < nx; j++) c[j] = c[j - 1] + 1;
    }
    const double gamma = std::fabs(EPS * 100 * observed);
    int64_t cnt = 0;
    const bool less = alternative == "less";
    for (double v : nulld) cnt += less ? (v <= observed + gamma) : (v >= observed - gamma);
    const double pv = clip01((double)cnt / (double)nulld.size());
    fn_result_num(&res[0], observed);
    fn_result_num(&res[1], pv);
    int64_t len = (int64_t)nulld.size();
    double *nd = (double *)fn_result_array(&res[2], TSR_F64, 1, &len);
    if (!nd) return TSR_ENOMEM;
    std::copy(nulld.begin(), nulld.end(), nd);
    return TSR_OK;
}


}  // namespace

const fn_def DEFS[] = {
    ROUTINE("stats.shapiro", 2, "x, axis=None, nan_policy='propagate', keepdims=False", "statistic, pvalue", r_shapiro, nullptr,
            "Shapiro-Wilk test for normality (scipy.stats.shapiro)."),
    ROUTINE("stats.cramervonmises", 2, "rvs, cdf, args=None, axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue",
            r_cramervonmises, nullptr,
            "One-sample Cramer-von Mises test (scipy.stats.cramervonmises); cdf is a distribution name (norm, expon, uniform, gamma, beta, t, chi2, f), args its parameters."),
    ROUTINE("stats.cramervonmises_2samp", 2, "x, y, method='auto', axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue",
            r_cramervonmises_2samp, nullptr, "Two-sample Cramer-von Mises test (scipy.stats.cramervonmises_2samp)."),
    ROUTINE("stats.ks_1samp", 4, "x, cdf, args=None, alternative='two-sided', method='auto', axis=0, nan_policy='propagate', keepdims=False",
            "statistic, pvalue, statistic_location, statistic_sign", r_ks_1samp, nullptr,
            "One-sample Kolmogorov-Smirnov test (scipy.stats.ks_1samp); cdf is a distribution name, args its parameters."),
    ROUTINE("stats.ks_2samp", 4, "data1, data2, alternative='two-sided', method='auto', axis=0, nan_policy='propagate', keepdims=False",
            "statistic, pvalue, statistic_location, statistic_sign", r_ks_2samp, nullptr,
            "Two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp)."),
    ROUTINE("stats.kstest", 4, "rvs, cdf, args=None, N=20, alternative='two-sided', method='auto', axis=0, nan_policy='propagate', keepdims=False",
            "statistic, pvalue, statistic_location, statistic_sign", r_kstest, nullptr,
            "Kolmogorov-Smirnov test (scipy.stats.kstest): cdf is a distribution name (one-sample) or a second sample."),
    ROUTINE("stats.anderson", 4, "x, dist='norm', method=None", "statistic, critical_values, significance_level, pvalue", r_anderson,
            nullptr, "Anderson-Darling test (scipy.stats.anderson) for norm, expon, logistic, gumbel_r, gumbel_l (gumbel, extreme1); method='interpolate' gives a p-value."),
    ROUTINE("stats.anderson_ksamp", 3, "samples, midrank=None, variant=None, method=None", "statistic, critical_values, pvalue",
            r_anderson_ksamp, nullptr, "k-sample Anderson-Darling test (scipy.stats.anderson_ksamp); samples: one sample per row of a 2-D array."),
    ROUTINE("stats.epps_singleton_2samp", 2, "x, y, t=None, axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue",
            r_epps_singleton_2samp, nullptr, "Epps-Singleton two-sample test (scipy.stats.epps_singleton_2samp); t defaults to (0.4, 0.8)."),
    ROUTINE("stats.alexandergovern", 2, "sample1, sample2, sample3=None, sample4=None, sample5=None, sample6=None, sample7=None, sample8=None, sample9=None, sample10=None, nan_policy='propagate', axis=0, keepdims=False", "statistic, pvalue",
            r_alexandergovern, nullptr, "Alexander-Govern test (scipy.stats.alexandergovern); the samples are sample1, sample2, ... (up to 10)."),
    ROUTINE("stats.combine_pvalues", 2, "pvalues, method='fisher', weights=None, axis=0, nan_policy='propagate', keepdims=False",
            "statistic, pvalue", r_combine_pvalues, nullptr, "Combine p-values of independent tests (scipy.stats.combine_pvalues)."),
    ROUTINE("stats.page_trend_test", 2, "data, ranked=False, predicted_ranks=None, method='auto'", "statistic, pvalue",
            r_page_trend_test, nullptr, "Page's test for ordered alternatives (scipy.stats.page_trend_test); the method used is not returned."),
    ROUTINE("stats.poisson_means_test", 2, "k1, n1, k2, n2, diff=0, alternative='two-sided'", "statistic, pvalue",
            r_poisson_means_test, nullptr, "Poisson means test, the E-test (scipy.stats.poisson_means_test)."),
    ROUTINE("stats.quantile_test", 3, "x, q=0, p=0.5, alternative='two-sided'", "statistic, statistic_type, pvalue",
            r_quantile_test, nullptr, "Test that the p-th population quantile is q (scipy.stats.quantile_test)."),
    ROUTINE("stats.bws_test", 3, "x, y, alternative='two-sided', method=None", "statistic, pvalue, null_distribution",
            r_bws_test, nullptr, "Baumgartner-Weiss-Schindler test (scipy.stats.bws_test) with the exact permutation distribution (at most 9999 partitions)."),
};

extern "C" const fn_table TSR_STATS_FN_TESTS2_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
