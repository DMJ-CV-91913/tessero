/*
 * scipy.stats hypothesis tests for the function registry (ADR 0011), SciPy 1.17.1.
 *
 *   ttest_1samp, ttest_ind, ttest_rel, ttest_ind_from_stats, chisquare, power_divergence, chi2_contingency,
 *   fisher_exact (2x2), binomtest, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, f_oneway, levene,
 *   bartlett, fligner, ansari, mood, brunnermunzel, ranksums, median_test, normaltest, skewtest,
 *   kurtosistest, jarque_bera
 *
 * Every test SciPy decorates with _axis_nan_policy_factory (scipy/stats/_axis_nan_policy.py) runs through
 * `run_axis_nan_policy` below, a port of that decorator: samples promoted to at least 1-D, axis=None ravels,
 * otherwise the non-axis dimensions broadcast (all dimensions for paired tests) and the axis moves last; 1-D
 * input takes the decorator's 1-D path (nan_policy per sample, too_small -> NaN), N-D input the vectorized
 * path when there is no NaN (one call over every axis-slice, so decisions SciPy takes over the whole array --
 * mannwhitneyu's method='auto', ansari's and mood's tie handling, wilcoxon's zero/tie switch -- are taken over
 * the whole array here too) and the apply_along_axis path otherwise. keepdims re-inserts the reduced axes.
 * These are routines (not gufuncs) because several tests take any number of samples (kruskal, levene, ...)
 * and because the decorator's semantics (paired broadcasting, too_small, per-path result dtypes such as the
 * integer `df` of ttest_1samp) are part of the result.
 *
 * Each statistic follows SciPy's arithmetic: NumPy's pairwise sums (tsr_psum) for 1-D and freshly built
 * arrays (concatenations, ranks), sequential sums where NumPy reduces an axis that is not the innermost one
 * of the caller's layout (the gufunc engine's rule), np.var/_var/_moment as SciPy composes them, and the
 * p-values from the scipy.special functions SciPy's _Simple* distributions call.
 */
#include "stats_dist.hpp"
#include "../src/fn.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <new>
#include <numeric>
#include <string>
#include <vector>

namespace {

using std::vector;
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();

/* ---------------------------------------------------------------- errors */

struct Err {
    int code;
};

[[noreturn]] void fail(const char *fmt, ...)
{
    char buf[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fn_set_error("%s", buf);
    throw Err{TSR_EARG};
}

template <class F> int guarded(F &&f)
{
    try {
        return f();
    } catch (const Err &e) {
        return e.code;
    } catch (const std::bad_alloc &) {
        fn_set_error("out of memory");
        return TSR_ENOMEM;
    }
}

/* ---------------------------------------------------------------- numeric helpers */

inline double psum(const double *x, int64_t n) { return n <= 0 ? 0.0 : tsr_psum(x, n); }
/* numpy.add.reduce over one axis-slice: pairwise when the axis is innermost in memory, else sequential */
inline double lsum(const double *x, int64_t n, bool seq)
{
    if (n <= 0) return 0.0;
    if (!seq) return tsr_psum(x, n);
    double a = x[0];
    for (int64_t i = 1; i < n; i++) a += x[i];
    return a;
}
inline double lmean(const double *x, int64_t n, bool seq) { return lsum(x, n, seq) / (double)n; }
/* BLAS ddot (np.vecdot / np.dot): the reference order */
inline double dot(const double *x, const double *y, int64_t n)
{
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += x[i] * y[i];
    return s;
}
/* scipy.stats._stats_py._moment(a, 2, mean=mean) and _var(a, ddof) */
double moment_c(const double *x, int64_t n, double mean, int order, bool seq)
{
    vector<double> s((size_t)std::max<int64_t>(n, 1));
    for (int64_t i = 0; i < n; i++) {
        const double d = x[i] - mean;
        switch (order) {
        case 2: s[i] = d * d; break;
        case 3: s[i] = (d * d) * d; break;
        case 4: { const double q = d * d; s[i] = q * q; break; }
        default: s[i] = d; break;
        }
    }
    return lmean(s.data(), n, seq);
}
double var_ddof(const double *x, int64_t n, double ddof, bool seq)
{
    /* _var: _moment(x, 2, mean=None) then var *= n / (n - ddof) */
    const double m = lmean(x, n, seq);
    double v = moment_c(x, n, m, 2, seq);
    if (ddof != 0) {
        const double nn = (double)n;
        v *= (nn / (nn - ddof));
    }
    return v;
}
/* numpy.var(x, ddof): mean = sum / n, sum of squares / max(n - ddof, 0) */
double np_var(const double *x, int64_t n, double ddof, bool seq)
{
    const double m = lsum(x, n, seq) / (double)n;
    vector<double> s((size_t)std::max<int64_t>(n, 1));
    for (int64_t i = 0; i < n; i++) { const double d = x[i] - m; s[i] = d * d; }
    double cnt = (double)n - ddof;
    if (cnt < 0) cnt = 0;
    return lsum(s.data(), n, seq) / cnt;
}

inline bool lt_nan(double a, double b) { return a < b || (b != b && a == a); }

/* scipy.stats._stats_py._rankdata(x, 'average', return_ties=True): ranks in input order, tie counts in sorted
   order (count at the first position of each run, zeros elsewhere). NaN sorts last, each NaN its own run. */
void rank_avg(const double *x, int64_t n, double *ranks, double *ties)
{
    vector<int64_t> j((size_t)n);
    std::iota(j.begin(), j.end(), 0);
    std::stable_sort(j.begin(), j.end(), [&](int64_t a, int64_t b) { return lt_nan(x[a], x[b]); });
    int64_t i = 0;
    while (i < n) {
        int64_t e = i + 1;
        while (e < n && !(x[j[e - 1]] != x[j[e]])) e++;
        const int64_t cnt = e - i;
        const double r = (double)(i + 1) + ((double)cnt - 1) / 2;
        for (int64_t k = i; k < e; k++) ranks[j[k]] = r;
        if (ties) {
            ties[i] = (double)cnt;
            for (int64_t k = i + 1; k < e; k++) ties[k] = 0.0;
        }
        i = e;
    }
}

/* _SimpleNormal / _SimpleStudentT / _SimpleChi2 / _SimpleF and _get_pvalue */
enum Alt { ALT_TWO, ALT_LESS, ALT_GREATER, ALT_BAD };

Alt parse_alt(const tsr_arg *a, const char *def = "two-sided")
{
    const char *s = def;
    if (a && a->kind == 2 && a->str) s = a->str;
    else if (a && a->kind != 0) return ALT_BAD;
    if (!strcmp(s, "two-sided")) return ALT_TWO;
    if (!strcmp(s, "less")) return ALT_LESS;
    if (!strcmp(s, "greater")) return ALT_GREATER;
    return ALT_BAD;
}
void check_alt(Alt a)
{
    if (a == ALT_BAD) fail("`alternative` must be 'less', 'greater', or 'two-sided'.");
}
double pval_norm(double z, Alt alt)
{
    check_alt(alt);
    if (alt == ALT_LESS) return sc::ndtr(z);
    if (alt == ALT_GREATER) return sc::ndtr(-z);
    return 2 * sc::ndtr(-std::fabs(z));
}
double pval_t(double t, double df, Alt alt)
{
    check_alt(alt);
    if (alt == ALT_LESS) return sc::stdtr(df, t);
    if (alt == ALT_GREATER) return sc::stdtr(df, -t);
    return 2 * sc::stdtr(df, -std::fabs(t));
}

/* ---------------------------------------------------------------- arguments */

struct Arr {
    vector<double> v;
    vector<int64_t> shape;
    vector<int64_t> strides;                 /* the caller's byte strides (memory layout) */
    bool isint = false;
    int64_t size() const { int64_t s = 1; for (auto d : shape) s *= d; return s; }
    int ndim() const { return (int)shape.size(); }
};

bool is_array_like(const tsr_arg &a) { return a.kind == 1 || a.kind == 3 || a.kind == 4; }

Arr to_arr(const tsr_arg &a, const char *name)
{
    Arr r;
    if (a.kind == 1 || a.kind == 4) {
        r.v.push_back(a.num);
        return r;
    }
    if (a.kind != 3) fail("%s must be an array or a number", name);
    if (a.arr.dtype == TSR_C128) fail("%s: complex input is not supported", name);
    int64_t n = 0;
    double *p = fn_arg_doubles(&a, &n);
    if (!p) throw std::bad_alloc();
    r.v.assign(p, p + n);
    fn_free_doubles(p, n);
    for (int d = 0; d < a.arr.ndim; d++) { r.shape.push_back(a.arr.shape[d]); r.strides.push_back(a.arr.strides[d]); }
    const int dt = a.arr.dtype;
    r.isint = dt == TSR_I64 || dt == TSR_I32 || dt == TSR_U8 || dt == TSR_BOOL;
    return r;
}

const char *arg_str(const tsr_arg *args, int nargs, int k, const char *def)
{
    if (k >= nargs || args[k].kind == 0) return def;
    if (args[k].kind == 2) return args[k].str;
    return nullptr;
}
bool arg_bool(const tsr_arg *args, int nargs, int k, bool def)
{
    if (k >= nargs || args[k].kind == 0) return def;
    if (args[k].kind == 1 || args[k].kind == 4) return args[k].num != 0;
    fail("expected a boolean");
}
bool arg_is_bool(const tsr_arg *args, int nargs, int k)
{
    return k >= nargs || args[k].kind == 4 || args[k].kind == 0 ||
           (args[k].kind == 1 && (args[k].num == 0 || args[k].num == 1));
}
double arg_num(const tsr_arg *args, int nargs, int k, double def)
{
    if (k >= nargs || args[k].kind == 0) return def;
    if (args[k].kind == 1 || args[k].kind == 4) return args[k].num;
    fail("expected a number");
}

enum { NP_PROPAGATE, NP_OMIT, NP_RAISE };

int parse_nan_policy(const tsr_arg *args, int nargs, int k)
{
    const char *s = arg_str(args, nargs, k, "propagate");
    if (s && !strcmp(s, "propagate")) return NP_PROPAGATE;
    if (s && !strcmp(s, "omit")) return NP_OMIT;
    if (s && !strcmp(s, "raise")) return NP_RAISE;
    fail("nan_policy must be one of {'propagate', 'omit', 'raise'}");
}

struct AxisOpt {
    bool none = false;
    int64_t axis = 0;
};

AxisOpt parse_axis(const tsr_arg *args, int nargs, int k, bool def_none = false)
{
    AxisOpt a;
    if (k >= nargs || args[k].kind == 0) {
        if (k >= nargs) a.none = def_none;
        else a.none = true;
        return a;
    }
    if (args[k].kind != 1 || args[k].num != std::floor(args[k].num)) fail("`axis` must be an integer, a tuple of integers, or `None`.");
    a.axis = (int64_t)std::fmax(-1e9, std::fmin(1e9, args[k].num));   /* out of range stays out of range, no UB */
    return a;
}

/* ---------------------------------------------------------------- results */

struct Results {
    int nout = 0;
    vector<int64_t> shape;                  /* result shape (empty: scalars) */
    vector<double> val;                     /* nslices * nout, slice-major */
    unsigned isint = 0;                     /* bit j: output j is int64 */
    int present = 0;                        /* outputs actually returned (the rest are absent) */
    bool from_fn = false;                   /* the test's own result (1-D or vectorized), not NaN-filled or applied */
    vector<int64_t> red_axes;               /* the reduced axes (keepdims) */
    vector<int64_t> loop;                   /* the result shape before keepdims */
    int path = 0;                           /* 1: 1-D, 2: vectorized N-D, 3: apply_along_axis */
};

int emit(const Results &R, tsr_result *res, int nres)
{
    const int64_t ns = (int64_t)(R.val.size() / (R.nout ? R.nout : 1));
    for (int j = 0; j < R.present && j < nres; j++) {
        const bool ii = (R.isint >> j) & 1;
        if (R.shape.empty()) {
            const double v = R.val[j];
            if (ii) fn_result_int(&res[j], (int64_t)v);
            else fn_result_num(&res[j], v);
            continue;
        }
        void *p = fn_result_array(&res[j], ii ? TSR_I64 : TSR_F64, (int32_t)R.shape.size(), R.shape.data());
        if (!p) return TSR_ENOMEM;
        for (int64_t s = 0; s < ns; s++) {
            const double v = R.val[(size_t)(s * R.nout + j)];
            if (ii) ((int64_t *)p)[s] = std::isfinite(v) ? (int64_t)v : INT64_MIN;
            else ((double *)p)[s] = v;
        }
    }
    return TSR_OK;
}

/* ---------------------------------------------------------------- the _axis_nan_policy decorator */

/* the axis-slices a test sees: x(s, k) is sample k of slice s */
struct Batch {
    int K = 0;
    int64_t ns = 0;
    vector<vector<double>> d;               /* ns * K */
    vector<bool> seq;                        /* per sample: NumPy reduces its original layout sequentially */
    vector<bool> isint;                      /* per sample: integer dtype */
    bool all_int() const { for (bool b : isint) if (!b) return false; return !isint.empty(); }
    bool nd = false;                         /* the vectorized N-D call (arrays, not NumPy scalars) */
    const double *x(int64_t s, int k) const { return d[(size_t)(s * K + k)].data(); }
    int64_t n(int64_t s, int k) const { return (int64_t)d[(size_t)(s * K + k)].size(); }
    bool sq(int k) const { return k < (int)seq.size() && seq[k]; }
};

/* a test: fills out[s * nout + j]; returns the number of outputs it produced (the result object's fields) */
using TestFn = std::function<int(const Batch &, double *out)>;
using TooSmall = std::function<bool(const vector<int64_t> &)>;

struct Spec {
    int nout = 2;                            /* output slots per slice (the most fields the test returns) */
    int nout_nan = -1;                       /* fields of the NaN-filled results (decorator n_outputs); -1: nout */
    bool paired = false;
    bool vectorized = true;                  /* the test takes `axis` (else apply_along_axis always) */
    unsigned int_outs = 0;                   /* outputs that are integers when the test itself returns them */
    TooSmall too_small;                      /* default: any sample of length <= 0 */
};

bool default_too_small(const vector<int64_t> &n)
{
    for (auto v : n) if (v <= 0) return true;
    return false;
}

/* broadcast a set of shapes (NumPy rules, right-aligned, same ndim already) */
vector<int64_t> bcast(const vector<vector<int64_t>> &shapes, int skip_dim)
{
    const size_t nd = shapes.empty() ? 0 : shapes[0].size();
    vector<int64_t> out(nd, 1);
    for (size_t d = 0; d < nd; d++) {
        if ((int)d == skip_dim) continue;
        int64_t len = 1;
        bool zero = false;
        for (auto &s : shapes) {
            if (s[d] == 0) zero = true;
            if (s[d] > len) len = s[d];
        }
        if (zero) len = 0;
        for (auto &s : shapes)
            if (s[d] != 1 && s[d] != len) fail("Array shapes are incompatible for broadcasting.");
        out[d] = len;
    }
    return out;
}

/* NumPy reduces sequentially unless the reduced axis is the smallest-stride axis of the caller's layout */
bool layout_seq(const Arr &a, int64_t axis_in_a)
{
    if (a.ndim() <= 1) return false;
    int best = -1;
    int64_t bs = 0;
    for (int d = 0; d < a.ndim(); d++) {
        if (a.shape[d] <= 1) continue;
        const int64_t s = a.strides[d] < 0 ? -a.strides[d] : a.strides[d];
        if (best < 0 || s < bs) { best = d; bs = s; }
    }
    return best >= 0 && best != axis_in_a;
}

Results run_axis_nan_policy(vector<Arr> samples, const Spec &spec, AxisOpt ax, int nanpol, bool keepdims, const TestFn &fn)
{
    const int K = (int)samples.size();
    const TooSmall too_small = spec.too_small ? spec.too_small : TooSmall(default_too_small);
    for (auto &s : samples)
        if (s.shape.empty()) { s.shape.push_back(1); s.strides.push_back(8); }
    int nd = 0;
    for (auto &s : samples) nd = std::max(nd, s.ndim());
    Results R;
    R.nout = spec.nout;

    auto nan_results = [&](const vector<int64_t> &shape) {
        R.shape = shape;
        int64_t ns = 1;
        for (auto v : shape) ns *= v;
        R.val.assign((size_t)(ns * R.nout), NaN);
        R.isint = 0;
        R.present = spec.nout_nan < 0 ? spec.nout : spec.nout_nan;
    };
    auto keep_1d = [&](vector<int64_t> &shape) { if (keepdims) shape.assign((size_t)nd, 1); };

    /* ------------------------------------------------ 1-D path (axis=None ravels) */
    const bool one_d = ax.none || nd <= 1;
    if (one_d) {
        vector<vector<double>> xs((size_t)K);
        for (int k = 0; k < K; k++) xs[k] = samples[k].v;
        vector<int64_t> oshape;
        keep_1d(oshape);
        if (!ax.none) {
            if (ax.axis < -1 || ax.axis > 0) fail("`axis` is out of bounds for array of dimension %d", nd);
            if (spec.paired && K > 1) {       /* _broadcast_arrays(samples, axis=None) */
                int64_t len = 1;
                for (auto &x : xs) if ((int64_t)x.size() != 1) len = (int64_t)x.size();
                for (auto &x : xs) {
                    if ((int64_t)x.size() == len) continue;
                    if (x.size() != 1) fail("Array shapes are incompatible for broadcasting.");
                    x.assign((size_t)len, x[0]);
                }
            }
        }
        bool has_nan = false;
        vector<bool> nan_k((size_t)K, false);
        for (int k = 0; k < K; k++)
            for (double v : xs[k]) if (v != v) { nan_k[k] = true; has_nan = true; break; }
        if (has_nan && nanpol == NP_RAISE) fail("The input contains nan values");
        if (has_nan && nanpol == NP_PROPAGATE) { nan_results(oshape); return R; }
        if (has_nan && nanpol == NP_OMIT) {
            if (!spec.paired) {
                for (auto &x : xs) x.erase(std::remove_if(x.begin(), x.end(), [](double v) { return v != v; }), x.end());
            } else {
                /* _remove_nans(paired): one mask over every sample (NumPy broadcasting of the masks, then boolean
                   indexing of each sample, which fails for a sample whose length differs from the mask's) */
                size_t len = xs[0].size();
                for (auto &x : xs) if (x.size() != len && x.size() != 1 && len != 1) fail("operands could not be broadcast together");
                for (auto &x : xs) len = std::max(len, x.size());
                vector<bool> bad(len, false);
                for (auto &x : xs) for (size_t i = 0; i < len; i++) if (x[x.size() == 1 ? 0 : i] != x[x.size() == 1 ? 0 : i]) bad[i] = true;
                for (auto &x : xs) {
                    if (x.size() != len) fail("boolean index did not match indexed array along axis 0");
                    vector<double> y;
                    for (size_t i = 0; i < len; i++) if (!bad[i]) y.push_back(x[i]);
                    x.swap(y);
                }
            }
        }
        vector<int64_t> lens;
        for (auto &x : xs) lens.push_back((int64_t)x.size());
        if (too_small(lens)) { nan_results(oshape); return R; }
        Batch b;
        b.K = K;
        b.ns = 1;
        b.d = xs;
        b.seq.assign((size_t)K, false);
        for (auto &a : samples) b.isint.push_back(a.isint);
        R.val.assign((size_t)R.nout, NaN);
        R.present = fn(b, R.val.data());
        R.isint = spec.int_outs;
        R.shape = oshape;
        R.from_fn = true;
        R.path = 1;
        if (ax.none) for (int d = 0; d < nd; d++) R.red_axes.push_back(d);
        else R.red_axes.push_back(ax.axis);
        return R;
    }

    /* ------------------------------------------------ N-D path */
    int64_t axis = ax.axis;
    if (axis < -nd || axis >= nd) fail("`axis` is out of bounds for array of dimension %d", nd);
    if (axis < 0) axis += nd;
    vector<vector<int64_t>> shp((size_t)K);
    for (int k = 0; k < K; k++) {
        shp[k].assign((size_t)(nd - samples[k].ndim()), 1);
        shp[k].insert(shp[k].end(), samples[k].shape.begin(), samples[k].shape.end());
    }
    vector<int64_t> full = bcast(shp, spec.paired ? -1 : (int)axis);
    vector<vector<int64_t>> bshp((size_t)K);
    for (int k = 0; k < K; k++) {
        bshp[k] = full;
        if (!spec.paired) bshp[k][axis] = shp[k][axis];
    }
    vector<int64_t> L;
    for (int d = 0; d < nd; d++) if (d != axis) L.push_back(full[d]);
    int64_t ns = 1;
    for (auto v : L) ns *= v;
    vector<int64_t> lens;
    for (int k = 0; k < K; k++) lens.push_back(bshp[k][axis]);
    vector<int64_t> oshape = L;
    if (keepdims) oshape.insert(oshape.begin() + axis, 1);

    bool any_empty = false;
    for (int k = 0; k < K; k++) { int64_t s = 1; for (auto v : bshp[k]) s *= v; if (s == 0) any_empty = true; }
    if (any_empty && (too_small(lens) || ns == 0)) { nan_results(oshape); return R; }

    /* the axis-slices, gathered (C order over the loop dimensions) */
    Batch b;
    b.K = K;
    b.ns = ns;
    b.d.resize((size_t)(ns * K));
    bool has_nan = false;
    for (int k = 0; k < K; k++) {
        const Arr &a = samples[k];
        vector<int64_t> cst((size_t)nd, 0);     /* element strides of the (prepended) own shape, 0 where broadcast */
        int64_t st = 1;
        for (int d = nd - 1; d >= 0; d--) { cst[d] = shp[k][d] == 1 ? 0 : st; st *= shp[k][d]; }
        vector<int64_t> idx((size_t)nd, 0);
        for (int64_t s = 0; s < ns; s++) {
            int64_t base = 0;
            for (int d = 0; d < nd; d++) base += idx[d] * cst[d];
            vector<double> &x = b.d[(size_t)(s * K + k)];
            x.resize((size_t)lens[k]);
            for (int64_t j = 0; j < lens[k]; j++) {
                x[j] = a.v[(size_t)(base + j * cst[axis])];
                if (x[j] != x[j]) has_nan = true;
            }
            for (int d = nd - 1; d >= 0; d--) {
                if (d == axis) continue;
                if (++idx[d] < full[d]) break;
                idx[d] = 0;
            }
        }
        const int64_t off = nd - a.ndim();
        b.seq.push_back(layout_seq(a, axis - off));
        b.isint.push_back(a.isint);
    }
    if (has_nan && nanpol == NP_RAISE) fail("The input contains nan values");
    R.shape = oshape;
    R.val.assign((size_t)(ns * R.nout), NaN);
    if (spec.vectorized && !has_nan) {
        b.nd = true;
        R.present = fn(b, R.val.data());
        R.isint = spec.int_outs;
        R.from_fn = true;
        R.path = 2;
        R.loop = L;
        R.red_axes.push_back(ax.axis);
        return R;
    }
    /* apply_along_axis: one slice at a time, results as a float array */
    R.path = 3;
    R.present = spec.nout_nan < 0 ? spec.nout : spec.nout_nan;
    for (int64_t s = 0; s < ns; s++) {
        Batch one;
        one.K = K;
        one.ns = 1;
        one.seq.assign((size_t)K, false);
        one.isint = b.isint;
        for (int k = 0; k < K; k++) one.d.push_back(b.d[(size_t)(s * K + k)]);
        bool snan = false;
        for (auto &x : one.d) for (double v : x) if (v != v) snan = true;
        double *out = &R.val[(size_t)(s * R.nout)];
        if (snan && nanpol == NP_PROPAGATE) continue;
        if (snan && nanpol == NP_OMIT) {
            if (!spec.paired) {
                for (auto &x : one.d) x.erase(std::remove_if(x.begin(), x.end(), [](double v) { return v != v; }), x.end());
            } else {
                const size_t len = one.d[0].size();
                vector<bool> bad(len, false);
                for (auto &x : one.d) for (size_t i = 0; i < len; i++) if (x[i] != x[i]) bad[i] = true;
                for (auto &x : one.d) {
                    vector<double> y;
                    for (size_t i = 0; i < len; i++) if (!bad[i]) y.push_back(x[i]);
                    x.swap(y);
                }
            }
        }
        vector<int64_t> ln;
        for (auto &x : one.d) ln.push_back((int64_t)x.size());
        if (too_small(ln)) continue;
        vector<double> tmp((size_t)R.nout, NaN);
        R.present = std::max(R.present, fn(one, tmp.data()));
        for (int j = 0; j < R.nout; j++) out[j] = tmp[j];
    }
    R.isint = 0;
    return R;
}

/* gather the leading positional samples (sample1, sample2, ... up to the first null) */
vector<Arr> var_samples(const tsr_arg *args, int nargs, int maxs)
{
    vector<Arr> s;
    for (int k = 0; k < maxs && k < nargs; k++) {
        if (args[k].kind == 0) break;
        char name[32];
        snprintf(name, sizeof name, "sample%d", k + 1);
        s.push_back(to_arr(args[k], name));
    }
    return s;
}

/* ================================================================ t-tests */

void ttest_1samp_core(const double *a, int64_t n, double popmean, Alt alt, bool seq, double *out)
{
    const double df = (double)(n - 1);
    if (n == 0) { out[0] = out[1] = out[2] = NaN; return; }
    const double mean = lmean(a, n, seq);
    const double d = mean - popmean;
    const double v = var_ddof(a, n, 1, seq);
    const double denom = std::sqrt(v / (double)n);
    const double t = d / denom;
    out[0] = t;
    out[1] = pval_t(t, df, alt);
    out[2] = df;
}

int f_ttest_1samp(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* a, popmean, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False */
        vector<Arr> s = {to_arr(args[0], "a"), to_arr(args[1], "popmean")};
        const AxisOpt ax = parse_axis(args, nargs, 2);
        const int np = parse_nan_policy(args, nargs, 3);
        const Alt alt = parse_alt(nargs > 4 ? &args[4] : nullptr);
        const bool keep = arg_bool(args, nargs, 5, false);
        Spec sp;
        sp.nout = 3;
        sp.int_outs = 4;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            for (int64_t i = 0; i < b.ns; i++) {
                if (b.n(i, 1) != 1) fail("`popmean.shape[axis]` must equal 1.");
                ttest_1samp_core(b.x(i, 0), b.n(i, 0), b.x(i, 1)[0], alt, b.sq(0), out + 3 * i);
            }
            return 3;
        });
        return emit(R, res, nres);
    });
}

int f_ttest_rel(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* a, b, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False */
        vector<Arr> s = {to_arr(args[0], "a"), to_arr(args[1], "b")};
        const AxisOpt ax = parse_axis(args, nargs, 2);
        const int np = parse_nan_policy(args, nargs, 3);
        const Alt alt = parse_alt(nargs > 4 ? &args[4] : nullptr);
        const bool keep = arg_bool(args, nargs, 5, false);
        Spec sp;
        sp.nout = 3;
        sp.int_outs = 4;
        sp.paired = true;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            for (int64_t i = 0; i < b.ns; i++) {
                const double *x = b.x(i, 0), *y = b.x(i, 1);
                const int64_t nx = b.n(i, 0), ny = b.n(i, 1);
                if (nx != ny && nx != 1 && ny != 1) fail("operands could not be broadcast together");
                const int64_t n = std::max(nx, ny);
                vector<double> d((size_t)n);
                for (int64_t j = 0; j < n; j++) d[j] = x[nx == 1 ? 0 : j] - y[ny == 1 ? 0 : j];
                ttest_1samp_core(d.data(), n, 0.0, alt, b.sq(0), out + 3 * i);
            }
            return 3;
        });
        return emit(R, res, nres);
    });
}

/* ttest_ind with trim: _ttest_trim_var_mean_len on one slice */
void trim_var_mean_len(const double *x, int64_t n, double trim, bool seq, double &v, double &m, double &nn)
{
    vector<double> a(x, x + n);
    std::sort(a.begin(), a.end(), lt_nan);
    const int64_t g = (int64_t)((double)n * trim);
    if (g == 0) v = var_ddof(a.data(), n, 1, seq);
    else {
        vector<double> w = a;
        for (int64_t i = 0; i < g; i++) w[i] = a[g];
        for (int64_t i = n - g; i < n; i++) w[i] = a[n - g - 1];
        v = var_ddof(w.data(), n, (double)(2 * g + 1), seq);
    }
    nn = (double)(n - 2 * g);
    /* trim_mean(a, trim): the middle of the (partitioned) sorted sample */
    const int64_t lo = (int64_t)(trim * (double)n), hi = n - lo;
    if (lo > hi) fail("Proportion too big.");
    if (lo < 0) fail("kth(=%lld) out of bounds (%lld)", (long long)(hi - 1), (long long)n);
    m = lmean(a.data() + lo, hi - lo, seq);
}

void ttest_ind_stats(double m1, double v1, double n1, double m2, double v2, double n2, bool equal_var, double &df, double &denom)
{
    if (equal_var) {
        if (n1 == 1) v1 = 0.;
        if (n2 == 1) v2 = 0.;
        df = n1 + n2 - 2.0;
        const double svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / df;
        denom = std::sqrt(svar * (1.0 / n1 + 1.0 / n2));
    } else {
        const double vn1 = v1 / n1, vn2 = v2 / n2;
        df = (vn1 + vn2) * (vn1 + vn2) / (vn1 * vn1 / (n1 - 1) + vn2 * vn2 / (n2 - 1));
        if (df != df) df = 1.;
        denom = std::sqrt(vn1 + vn2);
    }
}

int f_ttest_ind(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* a, b, axis=0, equal_var=True, nan_policy='propagate', alternative='two-sided', trim=0, method=None,
           keepdims=False */
        vector<Arr> s = {to_arr(args[0], "a"), to_arr(args[1], "b")};
        const AxisOpt ax = parse_axis(args, nargs, 2);
        const bool equal_var = arg_bool(args, nargs, 3, true);
        const int np = parse_nan_policy(args, nargs, 4);
        const Alt alt = parse_alt(nargs > 5 ? &args[5] : nullptr);
        const double trim = arg_num(args, nargs, 6, 0);
        if (nargs > 7 && args[7].kind != 0) fail("ttest_ind: `method` (permutation / Monte Carlo) is not supported");
        const bool keep = arg_bool(args, nargs, 8, false);
        Spec sp;
        sp.nout = 3;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            if (!(0 <= trim && trim < .5)) fail("Trimming percentage should be 0 <= `trim` < .5.");
            for (int64_t i = 0; i < b.ns; i++) {
                const int64_t na = b.n(i, 0), nb = b.n(i, 1);
                double *o = out + 3 * i;
                if (na == 0 || nb == 0) { o[0] = o[1] = o[2] = NaN; continue; }
                double v1, v2, m1, m2, n1 = (double)na, n2 = (double)nb;
                if (trim == 0) {
                    v1 = var_ddof(b.x(i, 0), na, 1, b.sq(0));
                    v2 = var_ddof(b.x(i, 1), nb, 1, b.sq(1));
                    m1 = lmean(b.x(i, 0), na, b.sq(0));
                    m2 = lmean(b.x(i, 1), nb, b.sq(1));
                } else {
                    trim_var_mean_len(b.x(i, 0), na, trim, b.sq(0), v1, m1, n1);
                    trim_var_mean_len(b.x(i, 1), nb, trim, b.sq(1), v2, m2, n2);
                }
                double df, denom;
                ttest_ind_stats(m1, v1, n1, m2, v2, n2, equal_var, df, denom);
                const double t = (m1 - m2) / denom;
                o[0] = t;
                o[1] = pval_t(t, df, alt);
                o[2] = df;
            }
            return 3;
        });
        return emit(R, res, nres);
    });
}

/* elementwise broadcasting of routine arguments (ttest_ind_from_stats) */
struct Bc {
    vector<Arr> a;
    vector<int64_t> shape;
    int64_t size = 1;
    explicit Bc(vector<Arr> in) : a(std::move(in))
    {
        int nd = 0;
        for (auto &x : a) nd = std::max(nd, x.ndim());
        shape.assign((size_t)nd, 1);
        for (auto &x : a)
            for (int d = 0; d < x.ndim(); d++) {
                const int D = nd - x.ndim() + d;
                const int64_t l = x.shape[d];
                if (l == 1) continue;
                if (shape[D] != 1 && shape[D] != l) fail("operands could not be broadcast together");
                shape[D] = l;
            }
        for (auto v : shape) size *= v;
    }
    double at(int k, int64_t flat) const
    {
        const Arr &x = a[k];
        const int nd = (int)shape.size();
        int64_t off = 0, st = 1;
        for (int d = nd - 1; d >= 0; d--) {
            const int64_t i = flat % shape[d];
            flat /= shape[d];
            const int dd = d - (nd - x.ndim());
            if (dd >= 0) {
                if (x.shape[dd] != 1) off += i * st;
                st *= x.shape[dd];
            }
        }
        return x.v[(size_t)off];
    }
};

int f_ttest_ind_from_stats(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True, alternative='two-sided' */
        static const char *const names[] = {"mean1", "std1", "nobs1", "mean2", "std2", "nobs2"};
        vector<Arr> in;
        for (int k = 0; k < 6; k++) in.push_back(to_arr(args[k], names[k]));
        const bool equal_var = arg_bool(args, nargs, 6, true);
        const Alt alt = parse_alt(nargs > 7 ? &args[7] : nullptr);
        Bc bc(in);
        Results R;
        R.nout = 2;
        R.present = 2;
        R.shape = bc.shape;
        R.val.resize((size_t)(2 * bc.size));
        for (int64_t e = 0; e < bc.size; e++) {
            const double mean1 = bc.at(0, e), std1 = bc.at(1, e), n1 = bc.at(2, e);
            const double mean2 = bc.at(3, e), std2 = bc.at(4, e), n2 = bc.at(5, e);
            double df, denom;
            ttest_ind_stats(mean1, std1 * std1, n1, mean2, std2 * std2, n2, equal_var, df, denom);
            const double t = (mean1 - mean2) / denom;
            R.val[(size_t)(2 * e)] = t;
            R.val[(size_t)(2 * e + 1)] = pval_t(t, df, alt);
        }
        return emit(R, res, nres);
    });
}

/* ================================================================ chi-squared family */

bool lambda_from_arg(const tsr_arg *a, double &lam)
{
    if (!a || a->kind == 0) { lam = 1; return true; }
    if (a->kind == 1) { lam = a->num; return true; }
    if (a->kind != 2) fail("lambda_ must be a string, a number or None");
    struct { const char *n; double v; } names[] = {{"pearson", 1}, {"log-likelihood", 0}, {"freeman-tukey", -0.5},
                                                   {"mod-log-likelihood", -1}, {"neyman", -2}, {"cressie-read", 2.0 / 3.0}};
    for (auto &e : names) if (!strcmp(a->str, e.n)) { lam = e.v; return true; }
    fail("invalid string for lambda_: '%s'. Valid strings are 'pearson', 'log-likelihood', 'freeman-tukey', "
         "'mod-log-likelihood', 'neyman', 'cressie-read'", a->str);
}

/* numpy's power with a scalar exponent: the fast paths of ndarray ** scalar */
inline double np_pow_scalar(double x, double e)
{
    if (e == 2) return x * x;
    if (e == 0.5) return std::sqrt(x);
    if (e == -1) return 1.0 / x;
    if (e == 1) return x;
    if (e == 0) return 1.0;
    return std::pow(x, e);
}

/* _power_divergence on one slice: f_exp may be null (then the mean of f_obs) */
void power_div_slice(const double *fo, int64_t n, const double *fe_in, bool seq_o, bool seq_e, double ddof, double lam,
                     bool sum_check, double *out)
{
    vector<double> fe((size_t)std::max<int64_t>(n, 1));
    if (fe_in) {
        if (sum_check) {
            const double so = lsum(fo, n, seq_o), se = lsum(fe_in, n, seq_e);
            const double rel = std::fabs(so - se) / std::min(so, se);
            if (rel > 1.4901161193847656e-08)
                fail("For each axis slice, the sum of the observed frequencies must agree with the sum of the "
                     "expected frequencies to a relative tolerance of 1.4901161193847656e-08");
        }
        for (int64_t i = 0; i < n; i++) fe[i] = fe_in[i];
    } else {
        const double m = lmean(fo, n, seq_o);
        for (int64_t i = 0; i < n; i++) fe[i] = m;
    }
    vector<double> terms((size_t)std::max<int64_t>(n, 1));
    for (int64_t i = 0; i < n; i++) {
        const double o = fo[i], e = fe[i];
        double t;
        if (lam == 1) { const double d = o - e; t = d * d / e; }
        else if (lam == 0) t = 2.0 * sc::xlogy(o, o / e);
        else if (lam == -1) t = 2.0 * sc::xlogy(e, e / o);
        else {
            t = o * (np_pow_scalar(o / e, lam) - 1);
            t /= 0.5 * lam * (lam + 1);
        }
        terms[i] = t;
    }
    const double stat = lsum(terms.data(), n, seq_o);
    const double df = (double)n - 1 - ddof;
    out[0] = stat;
    out[1] = sc::chdtrc(df, stat);
}

/* numpy.expand_dims(shape, axes) */
vector<int64_t> expand_dims(const vector<int64_t> &shape, vector<int64_t> axes)
{
    const int64_t nd = (int64_t)shape.size() + (int64_t)axes.size();
    vector<bool> at((size_t)nd, false);
    for (auto a : axes) {
        if (a < 0) a += nd;
        if (a < 0 || a >= nd || at[(size_t)a]) fail("axis out of range in expand_dims");
        at[(size_t)a] = true;
    }
    vector<int64_t> out;
    size_t k = 0;
    for (int64_t d = 0; d < nd; d++) out.push_back(at[(size_t)d] ? 1 : shape[k++]);
    return out;
}

int power_divergence_impl(const tsr_arg *args, int nargs, tsr_result *res, int nres, bool chisq)
{
    /* chisquare(f_obs, f_exp=None, ddof=0, axis=0, sum_check=True, nan_policy, keepdims)
       power_divergence(f_obs, f_exp=None, ddof=0, axis=0, lambda_=None, nan_policy, keepdims) */
    vector<Arr> s = {to_arr(args[0], "f_obs")};
    const bool has_exp = nargs > 1 && args[1].kind != 0;
    if (has_exp) s.push_back(to_arr(args[1], "f_exp"));
    /* ddof is not a sample: an array ddof broadcasts against the statistic in the p-value */
    Arr ddofA;
    bool ddof_arr = false;
    double ddof = 0;
    if (nargs > 2 && args[2].kind == 3) {
        ddofA = to_arr(args[2], "ddof");
        ddof_arr = ddofA.ndim() > 0;
        if (!ddof_arr) ddof = ddofA.v[0];
    } else ddof = arg_num(args, nargs, 2, 0);
    const AxisOpt ax = parse_axis(args, nargs, 3);
    double lam = 1;
    bool sum_check = true;
    if (chisq) sum_check = arg_bool(args, nargs, 4, true);
    else lambda_from_arg(nargs > 4 ? &args[4] : nullptr, lam);
    const int np = parse_nan_policy(args, nargs, 5);
    const bool keep = arg_bool(args, nargs, 6, false);
    Spec sp;
    sp.paired = true;
    sp.nout = 3;                                          /* statistic, pvalue, and the number of observations */
    sp.nout_nan = 2;
    sp.too_small = [](const vector<int64_t> &) { return false; };   /* too_small=-1 */
    Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
        for (int64_t i = 0; i < b.ns; i++) {
            const int64_t n = b.n(i, 0);
            const double *fe = has_exp ? b.x(i, 1) : nullptr;
            double *o = out + 3 * i;
            if (has_exp && b.n(i, 1) != n) {
                /* 1-D axis=None: f_obs and f_exp broadcast against each other */
                const int64_t ne = b.n(i, 1);
                if (ne != 1 && n != 1) fail("shape mismatch: objects cannot be broadcast to a single shape");
                const int64_t m = std::max(n, ne);
                vector<double> ov((size_t)m), ev((size_t)m);
                for (int64_t j = 0; j < m; j++) { ov[j] = b.x(i, 0)[n == 1 ? 0 : j]; ev[j] = b.x(i, 1)[ne == 1 ? 0 : j]; }
                power_div_slice(ov.data(), m, ev.data(), false, false, ddof, lam, sum_check, o);
                o[2] = (double)m;
                continue;
            }
            power_div_slice(b.x(i, 0), n, fe, b.sq(0), has_exp ? b.sq(1) : false, ddof, lam, sum_check, o);
            o[2] = (double)n;
        }
        return 2;
    });
    if (!ddof_arr || !R.from_fn) {
        if (ddof_arr && R.path == 3) fail("setting an array element with a sequence (array-valued ddof with NaN slices)");
        return emit(R, res, nres);
    }
    /* statistic: as computed; pvalue: broadcast(statistic shape, ddof shape) */
    const vector<int64_t> &L = R.loop;
    vector<int64_t> PB;
    {
        const size_t nd = std::max(L.size(), ddofA.shape.size());
        PB.assign(nd, 1);
        for (size_t d = 0; d < nd; d++) {
            const int64_t a = d + L.size() >= nd ? L[d + L.size() - nd] : 1;
            const int64_t c = d + ddofA.shape.size() >= nd ? ddofA.shape[d + ddofA.shape.size() - nd] : 1;
            if (a != 1 && c != 1 && a != c) fail("operands could not be broadcast together");
            PB[d] = a == 1 ? c : a;
        }
    }
    int64_t np_ = 1;
    for (auto v : PB) np_ *= v;
    vector<double> pv((size_t)np_);
    for (int64_t e = 0; e < np_; e++) {
        /* index into L (stat) and ddof, right-aligned */
        int64_t rem = e, si = 0, di = 0, sst = 1, dst = 1;
        for (int64_t d = (int64_t)PB.size() - 1; d >= 0; d--) {
            const int64_t i = rem % PB[d];
            rem /= PB[d];
            const int64_t dl = d - ((int64_t)PB.size() - (int64_t)L.size());
            if (dl >= 0) { if (L[dl] != 1) si += i * sst; sst *= L[dl]; }
            const int64_t dd = d - ((int64_t)PB.size() - (int64_t)ddofA.shape.size());
            if (dd >= 0) { if (ddofA.shape[dd] != 1) di += i * dst; dst *= ddofA.shape[dd]; }
        }
        const double stat = R.val[(size_t)(3 * si)], nobs = R.val[(size_t)(3 * si + 2)];
        pv[(size_t)e] = sc::chdtrc(nobs - 1 - ddofA.v[(size_t)di], stat);
    }
    vector<int64_t> pshape = PB;
    if (keep) pshape = expand_dims(PB, R.red_axes);
    /* statistic (first slot of each slice) */
    Results S = R;
    S.present = 1;
    int rc = emit(S, res, nres);
    if (rc != TSR_OK) return rc;
    if (pshape.empty()) { fn_result_num(&res[1], pv[0]); return TSR_OK; }
    double *pp = (double *)fn_result_array(&res[1], TSR_F64, (int32_t)pshape.size(), pshape.data());
    if (!pp) return TSR_ENOMEM;
    std::copy(pv.begin(), pv.end(), pp);
    return TSR_OK;
}

int f_chisquare(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] { return power_divergence_impl(args, nargs, res, nres, true); });
}
int f_power_divergence(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] { return power_divergence_impl(args, nargs, res, nres, false); });
}

/* chi2_contingency on an N-D table (C order); returns stat, p, dof and the expected table */
struct Contingency {
    double stat, p;
    int64_t dof;
    vector<double> expected;
};

/* numpy.sum over axis `ax` (keepdims) of a C-order array: pairwise along the innermost axis, else sequential */
vector<double> sum_axis(const vector<double> &a, const vector<int64_t> &shape, int ax, vector<int64_t> &oshape)
{
    const int nd = (int)shape.size();
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < ax; d++) outer *= shape[d];
    for (int d = ax + 1; d < nd; d++) inner *= shape[d];
    const int64_t len = shape[ax];
    oshape = shape;
    oshape[ax] = 1;
    vector<double> out((size_t)(outer * inner), 0.0);
    for (int64_t o = 0; o < outer; o++)
        for (int64_t i = 0; i < inner; i++) {
            if (inner == 1) {
                out[(size_t)o] = psum(&a[(size_t)(o * len)], len);
            } else {
                double s = len ? a[(size_t)(o * len * inner + i)] : 0.0;
                for (int64_t j = 1; j < len; j++) s += a[(size_t)((o * len + j) * inner + i)];
                out[(size_t)(o * inner + i)] = s;
            }
        }
    return out;
}

Contingency chi2_contingency_calc(const vector<double> &obs_in, const vector<int64_t> &shape, bool correction, double lam)
{
    for (double v : obs_in) if (v < 0) fail("All values in `observed` must be nonnegative.");
    if (obs_in.empty()) fail("No data; `observed` has size 0.");
    const int nd = (int)shape.size();
    /* margins: apply_over_axes(np.sum, a, other axes) */
    vector<vector<double>> marg((size_t)nd);
    vector<vector<int64_t>> mshape((size_t)nd);
    for (int k = 0; k < nd; k++) {
        vector<double> cur = obs_in;
        vector<int64_t> cs = shape;
        for (int j = 0; j < nd; j++) {
            if (j == k) continue;
            vector<int64_t> os;
            cur = sum_axis(cur, cs, j, os);
            cs = os;
        }
        marg[k] = cur;
        mshape[k] = cs;
    }
    const int64_t N = (int64_t)obs_in.size();
    const double total = psum(obs_in.data(), N);
    const double denom = nd - 1 == 0 ? 1.0 : std::pow(total, (double)(nd - 1));
    Contingency C;
    C.expected.assign((size_t)N, 0.0);
    vector<int64_t> idx((size_t)nd, 0);
    for (int64_t e = 0; e < N; e++) {
        double prod = 0;
        for (int k = 0; k < nd; k++) {
            const double m = marg[k][(size_t)idx[k]];
            prod = k == 0 ? m : prod * m;
        }
        C.expected[(size_t)e] = prod / denom;
        for (int d = nd - 1; d >= 0; d--) { if (++idx[d] < shape[d]) break; idx[d] = 0; }
    }
    for (double v : C.expected) if (v == 0) fail("The internally computed table of expected frequencies has a zero element.");
    int64_t ssum = 0;
    for (auto v : shape) ssum += v;
    C.dof = N - ssum + nd - 1;
    if (C.dof == 0) { C.stat = 0.0; C.p = 1.0; return C; }
    vector<double> obs = obs_in;
    if (C.dof == 1 && correction) {
        for (int64_t e = 0; e < N; e++) {
            const double diff = C.expected[(size_t)e] - obs[(size_t)e];
            const double dir = diff > 0 ? 1.0 : (diff < 0 ? -1.0 : (diff == 0 ? 0.0 : NaN));
            const double mag = std::min(0.5, std::fabs(diff));
            obs[(size_t)e] = obs[(size_t)e] + mag * dir;
        }
    }
    double out[2];
    power_div_slice(obs.data(), N, C.expected.data(), false, false, (double)(N - 1 - C.dof), lam, true, out);
    C.stat = out[0];
    C.p = out[1];
    return C;
}

int f_chi2_contingency(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* observed, correction=True, lambda_=None, method=None */
        Arr o = to_arr(args[0], "observed");
        const bool correction = arg_bool(args, nargs, 1, true);
        double lam = 1;
        lambda_from_arg(nargs > 2 ? &args[2] : nullptr, lam);
        if (nargs > 3 && args[3].kind != 0) fail("chi2_contingency: resampling `method` is not supported");
        vector<int64_t> shape = o.shape;
        Contingency C = chi2_contingency_calc(o.v, shape, correction, lam);
        fn_result_num(&res[0], C.stat);
        fn_result_num(&res[1], C.p);
        fn_result_int(&res[2], C.dof);
        double *p = (double *)fn_result_array(&res[3], TSR_F64, (int32_t)shape.size(), shape.data());
        if (!p) return TSR_ENOMEM;
        std::copy(C.expected.begin(), C.expected.end(), p);
        return TSR_OK;
    });
}

/* ================================================================ exact tests: fisher_exact, binomtest */

/* scipy.stats.hypergeom(M, n, N) public pmf / cdf / sf (rv_discrete wrappers around the Boost ufuncs) */
struct Hypergeom {
    double M, n, N, a, b;
    bool ok;
    Hypergeom(double M_, double n_, double N_) : M(M_), n(n_), N(N_)
    {
        ok = M > 0 && n >= 0 && N >= 0 && n <= M && N <= M && M == std::floor(M) && n == std::floor(n) && N == std::floor(N);
        a = std::max(N - (M - n), 0.0);
        b = std::min(n, N);
    }
    static double clip(double v) { return v < 0 ? 0.0 : (v > 1 ? 1.0 : v); }
    double pmf(double k) const
    {
        if (!ok || k != k) return NaN;
        if (k >= a && k <= b && std::floor(k) == k) return clip(sc::_hypergeom_pmf(k, n, N, M));
        return 0.0;
    }
    double cdf(double k) const
    {
        if (!ok || k != k) return NaN;
        if (k >= b) return 1.0;
        if (k >= a && std::isfinite(k)) return clip(sc::_hypergeom_cdf(k, n, N, M));
        return 0.0;
    }
    double sf(double k) const
    {
        if (!ok || k != k) return NaN;
        if (k < a) return 1.0;
        if (k >= a && k < b && std::isfinite(k)) return clip(sc::_hypergeom_sf(k, n, N, M));
        return 0.0;
    }
};

template <class F> double binary_search_tst(F &&a, double d, double lo, double hi)
{
    /* scipy.stats._binomtest._binary_search_for_binom_tst */
    while (lo < hi) {
        const double mid = lo + std::floor((hi - lo) / 2);
        const double midval = a(mid);
        if (midval < d) lo = mid + 1;
        else if (midval > d) hi = mid - 1;
        else return mid;
    }
    if (a(lo) <= d) return lo;
    return lo - 1;
}

int f_fisher_exact(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* table, alternative=None, method=None */
        Arr t = to_arr(args[0], "table");
        /* np.asarray(table, dtype=np.int64): truncation (NaN becomes INT64_MIN) */
        for (double &v : t.v) v = v != v ? -9.2233720368547758e18 : std::trunc(v);
        if (t.ndim() != 2) fail("The input `table` must have two dimensions.");
        for (double v : t.v) if (v < 0) fail("All values in `table` must be nonnegative.");
        if (t.shape[0] != 2 || t.shape[1] != 2 || (nargs > 2 && args[2].kind != 0)) {
            /* _fisher_exact_rxc: its deterministic cases; the rest is a Monte Carlo / permutation test */
            if (nargs > 1 && args[1].kind != 0)
                fail("`alternative` must be the default (None) unless `table` has shape `(2, 2)` and `method is None`.");
            if (t.size() == 0) fail("`table` must have at least one row and one column.");
            bool all0 = true;
            for (double v : t.v) if (v != 0) all0 = false;
            if (t.shape[0] == 1 || t.shape[1] == 1 || all0) {
                fn_result_num(&res[0], 1.0);
                fn_result_num(&res[1], 1.0);
                return TSR_OK;
            }
            fail("fisher_exact: tables other than 2x2 need SciPy's Monte Carlo / permutation `method`, which is not supported");
        }
        const char *as = arg_str(args, nargs, 1, "two-sided");
        if (!as) fail("`alternative` should be one of {'two-sided', 'less', 'greater'}");
        /* SciPy: np.asarray(table, dtype=np.int64) -- a value outside int64 raises (OverflowError) */
        for (int q = 0; q < 4; q++)
            if (!(t.v[q] > -9223372036854775808.0 && t.v[q] < 9223372036854775808.0)) fail("Python int too large to convert to C long");
        const int64_t c00 = (int64_t)t.v[0], c01 = (int64_t)t.v[1], c10 = (int64_t)t.v[2], c11 = (int64_t)t.v[3];
        if (c00 + c10 == 0 || c01 + c11 == 0 || c00 + c01 == 0 || c10 + c11 == 0) {
            fn_result_num(&res[0], NaN);
            fn_result_num(&res[1], 1.0);
            return TSR_OK;
        }
        /* numpy int64 scalar arithmetic wraps around (with a RuntimeWarning); unsigned arithmetic does the same
           without C++'s signed-overflow undefined behaviour */
        auto mulw = [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x * (uint64_t)y); };
        auto addw = [](int64_t x, int64_t y) { return (int64_t)((uint64_t)x + (uint64_t)y); };
        const double odds = (c10 > 0 && c01 > 0) ? (double)mulw(c00, c11) / (double)mulw(c10, c01) : INF;
        const int64_t n1 = addw(c00, c01), n2 = addw(c10, c11), n = addw(c00, c10);
        const Hypergeom H((double)(n1 + n2), (double)n1, (double)n);
        double pvalue;
        if (!strcmp(as, "less")) pvalue = H.cdf((double)c00);
        else if (!strcmp(as, "greater")) pvalue = Hypergeom((double)(n1 + n2), (double)n1, (double)(c01 + c11)).cdf((double)c01);
        else if (!strcmp(as, "two-sided")) {
            const double mode = std::trunc((double)((n + 1) * (n1 + 1)) / (double)(n1 + n2 + 2));
            const double pexact = H.pmf((double)c00), pmode = H.pmf(mode);
            const double epsilon = 1e-14, gamma = 1 + epsilon;
            if (std::fabs(pexact - pmode) / std::max(pexact, pmode) <= epsilon) {
                fn_result_num(&res[0], odds);
                fn_result_num(&res[1], 1.0);
                return TSR_OK;
            } else if ((double)c00 < mode) {
                const double plower = H.cdf((double)c00);
                if (H.pmf((double)n) > pexact * gamma) {
                    fn_result_num(&res[0], odds);
                    fn_result_num(&res[1], plower);
                    return TSR_OK;
                }
                const double guess = binary_search_tst([&](double x) { return -H.pmf(x); }, -pexact * gamma, mode, (double)n);
                pvalue = plower + H.sf(guess);
            } else {
                const double pupper = H.sf((double)(c00 - 1));
                if (H.pmf(0) > pexact * gamma) {
                    fn_result_num(&res[0], odds);
                    fn_result_num(&res[1], pupper);
                    return TSR_OK;
                }
                const double guess = binary_search_tst([&](double x) { return H.pmf(x); }, pexact * gamma, 0, mode);
                pvalue = pupper + H.cdf(guess);
            }
        } else fail("`alternative` should be one of {'two-sided', 'less', 'greater'}");
        pvalue = std::min(pvalue, 1.0);
        fn_result_num(&res[0], odds);
        fn_result_num(&res[1], pvalue);
        return TSR_OK;
    });
}

int f_binomtest(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* k, n, p=0.5, alternative='two-sided' */
        if (args[0].kind != 1 || args[0].num != std::floor(args[0].num)) fail("`k` must be an integer.");
        if (args[1].kind != 1 || args[1].num != std::floor(args[1].num)) fail("`n` must be an integer.");
        const double k = args[0].num, n = args[1].num;
        if (k < 0) fail("k must be an integer not less than 0, but got %g.", k);
        if (n < 1) fail("n must be an integer not less than 1, but got %g.", n);
        if (k > n) fail("k (%g) must not be greater than n (%g).", k, n);
        const double p = arg_num(args, nargs, 2, 0.5);
        if (!(0 <= p && p <= 1)) fail("p (%g) must be in range [0,1]", p);
        const char *as = arg_str(args, nargs, 3, "two-sided");
        const tsd::Dist *B = tsd::find("binom");
        if (!B) fail("binom distribution unavailable");
        auto pmf = [&](double x) { return tsd::call(B, DM_PDF, x, {n, p, 0.0}); };
        auto cdf = [&](double x) { return tsd::call(B, DM_CDF, x, {n, p, 0.0}); };
        auto sf = [&](double x) { return tsd::call(B, DM_SF, x, {n, p, 0.0}); };
        double pval;
        if (as && !strcmp(as, "less")) pval = cdf(k);
        else if (as && !strcmp(as, "greater")) pval = sf(k - 1);
        else if (as && !strcmp(as, "two-sided")) {
            const double d = pmf(k);
            const double rerr = 1 + 1e-7;
            if (k == p * n) pval = 1.;
            else if (k < p * n) {
                const double ix = binary_search_tst([&](double x) { return -pmf(x); }, -d * rerr, std::ceil(p * n), n);
                const double y = n - ix + (double)(d * rerr == pmf(ix));
                pval = cdf(k) + sf(n - y);
            } else {
                const double ix = binary_search_tst(pmf, d * rerr, 0, std::floor(p * n));
                const double y = ix + 1;
                pval = cdf(y - 1) + sf(k - 1);
            }
            pval = std::min(1.0, pval);
        } else fail("alternative ('%s') not recognized; must be 'two-sided', 'less' or 'greater'", as ? as : "?");
        fn_result_num(&res[0], k / n);
        fn_result_num(&res[1], pval);
        fn_result_int(&res[2], (int64_t)k);
        fn_result_int(&res[3], (int64_t)n);
        fn_result_num(&res[4], k / n);
        return TSR_OK;
    });
}

/* ================================================================ rank tests */

/* _MWU: exact null distribution of the Mann-Whitney U statistic (Loffler's recurrence) */
struct MWU {
    int64_t n1, n2;
    vector<double> conf;                      /* configurations (counts) for u = 0 .. maxu */
    double total;
    MWU(int64_t a, int64_t b) : n1(std::min(a, b)), n2(std::max(a, b)) { total = sc::binom((double)(n1 + n2), (double)n1); }
    void build(int64_t maxu)
    {
        if ((int64_t)conf.size() >= maxu + 1) return;
        vector<int64_t> s((size_t)(maxu + 1), 0);
        for (int64_t d = 1; d <= n1; d++) for (int64_t i = d; i <= maxu; i += d) s[i] += d;
        for (int64_t d = n2 + 1; d <= n2 + n1; d++) for (int64_t i = d; i <= maxu; i += d) s[i] -= d;
        conf.assign((size_t)(maxu + 1), 0.0);
        conf[0] = 1;
        bool is_uint = true;
        const double uint_max = 18446744073709551615.0;
        for (int64_t u = 1; u <= maxu; u++) {
            /* np.dot(configurations[:u], s_array[1:][u-1::-1]) / u */
            double acc = 0.0;
            for (int64_t i = 0; i < u; i++) acc += conf[i] * (double)s[(size_t)(u - i)];
            const double nv = acc / (double)u;
            if (nv > uint_max && is_uint) is_uint = false;
            conf[u] = is_uint ? (double)(uint64_t)nv : nv;
        }
    }
    double pmf(int64_t k) { build(k); return conf[k] / total; }
    double cdf(int64_t k)
    {
        build(k);
        double c = 0;
        for (int64_t i = 0; i <= k; i++) c += conf[i] / total;
        return c;
    }
    double sf(int64_t k)
    {
        const int64_t kc = n1 * n2 - k;
        if (k < kc) return 1. - cdf(k) + pmf(k);
        return cdf(kc);
    }
};

enum { MWU_AUTO, MWU_ASYM, MWU_EXACT };

int f_mannwhitneyu(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* x, y, use_continuity=True, alternative='two-sided', axis=0, method='auto', nan_policy, keepdims */
        vector<Arr> s = {to_arr(args[0], "x"), to_arr(args[1], "y")};
        const bool cont_ok = arg_is_bool(args, nargs, 2);
        const bool use_continuity = cont_ok ? arg_bool(args, nargs, 2, true) : true;
        std::string alts = arg_str(args, nargs, 3, "two-sided") ? arg_str(args, nargs, 3, "two-sided") : "?";
        const AxisOpt ax = parse_axis(args, nargs, 4);
        std::string ms = arg_str(args, nargs, 5, "auto") ? arg_str(args, nargs, 5, "auto") : "?";
        const int np = parse_nan_policy(args, nargs, 6);
        const bool keep = arg_bool(args, nargs, 7, false);
        for (auto &c : alts) c = (char)tolower(c);
        for (auto &c : ms) c = (char)tolower(c);
        Spec sp;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            if (!cont_ok) fail("`use_continuity` must be one of {True, False}.");
            Alt alt = alts == "two-sided" ? ALT_TWO : alts == "less" ? ALT_LESS : alts == "greater" ? ALT_GREATER : ALT_BAD;
            if (alt == ALT_BAD) fail("`alternative` must be one of {'two-sided', 'less', 'greater'}.");
            int method = ms == "auto" ? MWU_AUTO : ms == "asymptotic" ? MWU_ASYM : ms == "exact" ? MWU_EXACT : -1;
            if (method < 0) fail("`method` must be one of {'asymptotic', 'exact', 'auto'}.");
            const int64_t n1 = b.n(0, 0), n2 = b.n(0, 1);
            vector<vector<double>> t((size_t)b.ns);
            vector<double> U1v((size_t)b.ns);
            bool ties = false;
            for (int64_t i = 0; i < b.ns; i++) {
                vector<double> xy(b.x(i, 0), b.x(i, 0) + n1);
                xy.insert(xy.end(), b.x(i, 1), b.x(i, 1) + n2);
                vector<double> r(xy.size());
                t[i].resize(xy.size());
                rank_avg(xy.data(), (int64_t)xy.size(), r.data(), t[i].data());
                for (double v : t[i]) if (v > 1) ties = true;
                const double R1 = psum(r.data(), n1);
                U1v[i] = R1 - (double)(n1 * (n1 + 1)) / 2;
            }
            if (method == MWU_AUTO) method = (n1 > 8 && n2 > 8) || ties ? MWU_ASYM : MWU_EXACT;
            MWU dist(n1, n2);
            for (int64_t i = 0; i < b.ns; i++) {
                const double U1 = U1v[i], U2 = (double)(n1 * n2) - U1;
                double U, f;
                if (alt == ALT_GREATER) { U = U1; f = 1; }
                else if (alt == ALT_LESS) { U = U2; f = 1; }
                else { U = std::max(U1, U2); f = 2; }
                double p;
                if (method == MWU_EXACT) p = dist.sf((int64_t)U);
                else {
                    const double mu = (double)(n1 * n2) / 2;
                    const int64_t n = n1 + n2;
                    vector<double> tt(t[i].size());
                    for (size_t j = 0; j < tt.size(); j++) tt[j] = t[i][j] * t[i][j] * t[i][j] - t[i][j];
                    const double tie_term = psum(tt.data(), (int64_t)tt.size());
                    const double sd = std::sqrt((double)(n1 * n2) / 12 * ((double)(n + 1) - tie_term / (double)(n * (n - 1))));
                    double num = U - mu;
                    if (use_continuity) num -= 0.5;
                    const double z = num / sd;
                    p = sc::ndtr(-z);
                }
                p *= f;
                p = std::min(std::max(p, 0.), 1.);
                out[2 * i] = U1;
                out[2 * i + 1] = p;
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

/* _get_wilcoxon_distr(n) */
vector<double> wilcoxon_distr(int64_t n)
{
    vector<double> c(1, 1.0);
    for (int64_t k = 1; k <= n; k++) {
        vector<double> nc((size_t)(k * (k + 1) / 2 + 1), 0.0);
        const size_t m = c.size();
        for (size_t i = 0; i < m; i++) nc[i] = c[i] * 0.5;
        for (size_t i = 0; i < m; i++) nc[nc.size() - m + i] += c[i] * 0.5;
        c.swap(nc);
    }
    return c;
}

/* WilcoxonDistribution.cdf / sf for one (k, n) */
double wdist_cdf(const vector<double> &pm, int64_t k, double mn);
double wdist_sf(const vector<double> &pm, int64_t k, double mn)
{
    const int64_t L = (int64_t)pm.size();
    if ((double)k <= mn) {
        const int64_t s = std::max<int64_t>(std::min(k < 0 ? std::max<int64_t>(L + k, 0) : k, L), 0);
        return psum(pm.data() + s, L - s);
    }
    /* 1 - _cdf(k - 1): pmfs[:k].sum() */
    int64_t e = k;
    if (e < 0) e = std::max<int64_t>(L + e, 0);
    e = std::min(e, L);
    return 1 - psum(pm.data(), e);
}
double wdist_cdf(const vector<double> &pm, int64_t k, double mn)
{
    const int64_t L = (int64_t)pm.size();
    if ((double)k <= mn) {
        int64_t e = k + 1;
        if (e < 0) e = std::max<int64_t>(L + e, 0);
        e = std::min(e, L);
        return psum(pm.data(), e);
    }
    /* 1 - _sf(k + 1): pmfs[k+1:].sum() */
    int64_t s = k + 1;
    if (s < 0) s = std::max<int64_t>(L + s, 0);
    s = std::min(s, L);
    return 1 - psum(pm.data() + s, L - s);
}

enum { ZM_WILCOX, ZM_PRATT, ZM_ZSPLIT };
enum { WM_AUTO, WM_ASYM, WM_EXACT, WM_PERM };

struct WStat {
    double r_plus, r_minus, se, z, count;
};

/* _wilcoxon_statistic on one slice; `ties_any` accumulates has_ties */
WStat wilcoxon_stat(const double *d_in, int64_t n, int zm, bool want_z, bool &ties_any, vector<double> *ranks_out,
                    vector<double> *dz_out)
{
    vector<double> d(d_in, d_in + n);
    vector<bool> zeros((size_t)n);
    for (int64_t i = 0; i < n; i++) zeros[i] = d[i] == 0;
    if (zm == ZM_WILCOX) for (int64_t i = 0; i < n; i++) if (zeros[i]) d[i] = NaN;
    int64_t n_nan = 0;
    for (double v : d) n_nan += v != v;
    const double count = (double)(n - n_nan);
    vector<double> ad((size_t)n), r((size_t)n), t((size_t)n);
    for (int64_t i = 0; i < n; i++) ad[i] = std::fabs(d[i]);
    rank_avg(ad.data(), n, r.data(), t.data());
    vector<double> tmp((size_t)n);
    for (int64_t i = 0; i < n; i++) tmp[i] = (d[i] > 0 ? 1.0 : 0.0) * r[i];
    double r_plus = psum(tmp.data(), n);
    for (int64_t i = 0; i < n; i++) tmp[i] = (d[i] < 0 ? 1.0 : 0.0) * r[i];
    double r_minus = psum(tmp.data(), n);
    for (double v : t) if (v == 0) ties_any = true;
    if (zm == ZM_ZSPLIT) {
        for (int64_t i = 0; i < n; i++) tmp[i] = (zeros[i] ? 1.0 : 0.0) * r[i];
        const double rz = psum(tmp.data(), n) / 2;
        r_plus += rz;
        r_minus += rz;
    }
    double mn = count * (count + 1.) * 0.25;
    double se = count * (count + 1.) * (2. * count + 1.);
    if (zm == ZM_PRATT) {
        int64_t nz = 0;
        for (bool z : zeros) nz += z;
        const double n_zero = (double)nz;
        mn -= n_zero * (n_zero + 1.) * 0.25;
        se -= n_zero * (n_zero + 1.) * (2. * n_zero + 1.);
        if (nz > 0 && n > 0) t[0] = 0.;
    }
    for (int64_t i = 0; i < n; i++) tmp[i] = t[i] * t[i] * t[i] - t[i];
    const double tie_correct = psum(tmp.data(), n);
    se = std::sqrt((se - tie_correct / 2) / 24);
    WStat w{r_plus, r_minus, se, want_z ? (r_plus - mn) / se : NaN, count};
    if (ranks_out) *ranks_out = r;
    if (dz_out) *dz_out = d;
    return w;
}

int f_wilcoxon(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* x, y=None, zero_method='wilcox', correction=False, alternative='two-sided', method='auto', axis=0,
           nan_policy='propagate', keepdims=False */
        vector<Arr> s = {to_arr(args[0], "x")};
        const bool has_y = nargs > 1 && args[1].kind != 0;
        if (has_y) s.push_back(to_arr(args[1], "y"));
        std::string zms = arg_str(args, nargs, 2, "wilcox") ? arg_str(args, nargs, 2, "wilcox") : "?";
        const bool corr_ok = arg_is_bool(args, nargs, 3);
        const bool correction = corr_ok ? arg_bool(args, nargs, 3, false) : false;
        std::string alts = arg_str(args, nargs, 4, "two-sided") ? arg_str(args, nargs, 4, "two-sided") : "?";
        std::string ms = arg_str(args, nargs, 5, "auto") ? arg_str(args, nargs, 5, "auto") : "?";
        const AxisOpt ax = parse_axis(args, nargs, 6);
        const int np = parse_nan_policy(args, nargs, 7);
        const bool keep = arg_bool(args, nargs, 8, false);
        Spec sp;
        sp.paired = true;
        sp.nout = 3;
        sp.nout_nan = ms == "asymptotic" ? 3 : 2;
        const std::string method_in = ms == "approx" ? "asymptotic" : ms;
        const bool output_z = method_in == "asymptotic";
        for (auto &c : zms) c = (char)tolower(c);
        for (auto &c : alts) c = (char)tolower(c);
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            const int zm = zms == "wilcox" ? ZM_WILCOX : zms == "pratt" ? ZM_PRATT : zms == "zsplit" ? ZM_ZSPLIT : -1;
            if (zm < 0) fail("`zero_method` must be one of {'wilcox', 'pratt', 'zsplit'}.");
            if (!corr_ok) fail("`correction` must be one of {True, False}.");
            const Alt alt = alts == "two-sided" ? ALT_TWO : alts == "less" ? ALT_LESS : alts == "greater" ? ALT_GREATER : ALT_BAD;
            if (alt == ALT_BAD) fail("`alternative` must be one of {'two-sided', 'less', 'greater'}.");
            int method = method_in == "auto" ? WM_AUTO : method_in == "asymptotic" ? WM_ASYM : method_in == "exact" ? WM_EXACT : -1;
            if (method < 0) fail("`method` must be one of {'auto', 'asymptotic', 'exact'} or an instance of `stats.PermutationMethod`.");
            /* d = x - y along the last axis */
            vector<vector<double>> dv((size_t)b.ns);
            int64_t n_zero = 0;
            for (int64_t i = 0; i < b.ns; i++) {
                const int64_t nx = b.n(i, 0);
                if (b.K == 2) {
                    const int64_t ny = b.n(i, 1);
                    if (nx != ny && nx != 1 && ny != 1) fail("operands could not be broadcast together");
                    const int64_t m = std::max(nx, ny);
                    dv[i].resize((size_t)m);
                    for (int64_t j = 0; j < m; j++) dv[i][j] = b.x(i, 0)[nx == 1 ? 0 : j] - b.x(i, 1)[ny == 1 ? 0 : j];
                } else dv[i].assign(b.x(i, 0), b.x(i, 0) + nx);
                for (double v : dv[i]) n_zero += v == 0;
            }
            const int64_t n = (int64_t)dv[0].size();
            if (method == WM_AUTO && n > 50) method = WM_ASYM;
            const int nres_out = output_z ? 3 : 2;
            if (n == 0) {
                for (int64_t i = 0; i < b.ns; i++) for (int j = 0; j < nres_out; j++) out[i * sp.nout + j] = NaN;
                return nres_out;
            }
            bool has_ties = false;
            const bool want_z = method == WM_ASYM || method == WM_AUTO;
            vector<WStat> ws((size_t)b.ns);
            vector<vector<double>> rk((size_t)b.ns), dz((size_t)b.ns);
            for (int64_t i = 0; i < b.ns; i++) ws[i] = wilcoxon_stat(dv[i].data(), n, zm, want_z, has_ties, &rk[i], &dz[i]);
            if (method == WM_AUTO) {
                if (!(has_ties || n_zero > 0)) method = WM_EXACT;
                else if (n <= 13) method = WM_PERM;
                else method = WM_ASYM;
            }
            for (int64_t i = 0; i < b.ns; i++) {
                WStat &w = ws[i];
                double z = w.z, p;
                if (method == WM_ASYM) {
                    if (correction) {
                        const double sign = alt == ALT_GREATER ? 1 : alt == ALT_LESS ? -1 : (z > 0 ? 1. : z < 0 ? -1. : z == 0 ? 0. : NaN);
                        z = z - sign * 0.5 / w.se;
                    }
                    p = pval_norm(z, alt);
                } else if (method == WM_EXACT) {
                    const int64_t cnt = (int64_t)w.count;
                    const vector<double> pm = wilcoxon_distr(cnt);
                    const double mn = (double)cnt * (double)(cnt + 1) / 4;
                    if (alt == ALT_LESS) p = wdist_cdf(pm, (int64_t)std::ceil(w.r_plus), mn);
                    else if (alt == ALT_GREATER) p = wdist_sf(pm, (int64_t)std::floor(w.r_plus), mn);
                    else {
                        p = 2 * std::min(wdist_sf(pm, (int64_t)std::floor(w.r_plus), mn), wdist_cdf(pm, (int64_t)std::ceil(w.r_plus), mn));
                        p = std::min(std::max(p, 0.), 1.);
                    }
                } else {
                    /* permutation_test(..., permutation_type='samples'): every sign flip (2**n <= 9999: exact) */
                    if (n <= 1) fail("each sample in `data` must contain two or more observations along `axis`.");
                    const double *dd = dz[i].data();
                    const double *r = rk[i].data();
                    double rz = 0;
                    if (zm == ZM_ZSPLIT) {
                        vector<double> tmp((size_t)n);
                        for (int64_t j = 0; j < n; j++) tmp[j] = (dv[i][j] == 0 ? 1.0 : 0.0) * r[j];
                        rz = psum(tmp.data(), n) / 2;
                    }
                    const double obs = w.r_plus;
                    const double gamma = std::fabs(2.220446049250313e-16 * 100 * obs);
                    const int64_t nperm = (int64_t)1 << n;
                    int64_t cl = 0, cg = 0;
                    vector<double> tmp((size_t)n);
                    for (int64_t m = 0; m < nperm; m++) {
                        for (int64_t j = 0; j < n; j++) {
                            /* itertools.product of permutations((0, 1)): bit set -> the pair is swapped (d -> -d) */
                            const bool flip = (m >> (n - 1 - j)) & 1;
                            const double v = flip ? -dd[j] : dd[j];
                            tmp[j] = (v > 0 ? 1.0 : 0.0) * r[j];
                        }
                        double st = psum(tmp.data(), n);
                        if (zm == ZM_ZSPLIT) st += rz;
                        cl += st <= obs + gamma;
                        cg += st >= obs - gamma;
                    }
                    const double pl = (double)cl / (double)nperm, pg = (double)cg / (double)nperm;
                    if (alt == ALT_LESS) p = pl;
                    else if (alt == ALT_GREATER) p = pg;
                    else p = std::min(pl, pg) * 2;
                    p = std::min(std::max(p, 0.), 1.);
                }
                const double stat = alt == ALT_TWO ? std::min(w.r_plus, w.r_minus) : w.r_plus;
                if (alt == ALT_TWO && method == WM_ASYM) z = -std::fabs(z);
                out[i * sp.nout] = stat;
                out[i * sp.nout + 1] = p;
                if (output_z) out[i * sp.nout + 2] = z;
            }
            return nres_out;
        });
        return emit(R, res, nres);
    });
}

int f_kruskal(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* sample1, ..., sample12, nan_policy='propagate', axis=0, keepdims=False */
        const int NS = 12;
        vector<Arr> s = var_samples(args, nargs, NS);
        const int np = parse_nan_policy(args, nargs, NS);
        const AxisOpt ax = parse_axis(args, nargs, NS + 1);
        const bool keep = arg_bool(args, nargs, NS + 2, false);
        Spec sp;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            const int k = b.K;
            if (k < 2) fail("Need at least two groups in stats.kruskal()");
            for (int64_t i = 0; i < b.ns; i++) {
                vector<double> all;
                vector<int64_t> n;
                for (int g = 0; g < k; g++) { n.push_back(b.n(i, g)); all.insert(all.end(), b.x(i, g), b.x(i, g) + b.n(i, g)); }
                const int64_t N = (int64_t)all.size();
                vector<double> r((size_t)N), t((size_t)N);
                rank_avg(all.data(), N, r.data(), t.data());
                for (auto &v : t) v = v * v * v - v;
                const double ties = 1 - psum(t.data(), N) / (double)(N * N * N - N);
                double ssbn = 0;
                int64_t off = 0;
                for (int g = 0; g < k; g++) {
                    const double sg = psum(r.data() + off, n[g]);
                    ssbn = ssbn + sg * sg / (double)n[g];
                    off += n[g];
                }
                double h = 12.0 / (double)(N * (N + 1)) * ssbn - 3 * (double)(N + 1);
                h /= ties;
                out[2 * i] = h;
                out[2 * i + 1] = sc::chdtrc((double)(k - 1), h);
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

int f_friedmanchisquare(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* sample1, ..., sample12, axis=0, nan_policy='propagate', keepdims=False */
        const int NS = 12;
        vector<Arr> s = var_samples(args, nargs, NS);
        const AxisOpt ax = parse_axis(args, nargs, NS);
        const int np = parse_nan_policy(args, nargs, NS + 1);
        const bool keep = arg_bool(args, nargs, NS + 2, false);
        Spec sp;
        sp.paired = true;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            const int k = b.K;
            if (k < 3) fail("At least 3 samples must be given for Friedman test, got %d.", k);
            for (int64_t i = 0; i < b.ns; i++) {
                int64_t n = b.n(i, 0);
                for (int g = 1; g < k; g++)
                    if (b.n(i, g) != n) {
                        if (n != 1 && b.n(i, g) != 1) fail("all input arrays must have the same shape");
                        n = std::max(n, b.n(i, g));
                    }
                if (n == 0) fail("One or more sample arguments is too small.");
                vector<double> row((size_t)k), r((size_t)k), t((size_t)k), colsum((size_t)k, 0.0);
                double ties = 0;
                for (int64_t j = 0; j < n; j++) {
                    for (int g = 0; g < k; g++) row[g] = b.x(i, g)[b.n(i, g) == 1 ? 0 : j];
                    rank_avg(row.data(), k, r.data(), t.data());
                    for (int g = 0; g < k; g++) {
                        colsum[g] = j == 0 ? r[g] : colsum[g] + r[g];
                        ties += t[g] * (t[g] * t[g] - 1);
                    }
                }
                const double kk = (double)k, nn = (double)n;
                const double c = 1 - ties / (double)(k * (k * k - 1) * n);
                for (auto &v : colsum) v = v * v;
                const double ssbn = psum(colsum.data(), k);
                const double stat = (12.0 / (double)(k * n * (k + 1)) * ssbn - 3 * nn * (kk + 1)) / c;
                out[2 * i] = stat;
                out[2 * i + 1] = sc::chdtrc(kk - 1, stat);
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

/* ================================================================ variance / location tests */

int f_f_oneway(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* sample1, ..., sample12, axis=0, equal_var=True, nan_policy='propagate', keepdims=False */
        const int NS = 12;
        vector<Arr> s = var_samples(args, nargs, NS);
        const AxisOpt ax = parse_axis(args, nargs, NS);
        const bool ev_ok = nargs <= NS + 1 || args[NS + 1].kind == 4 || args[NS + 1].kind == 0;
        const bool equal_var = ev_ok ? arg_bool(args, nargs, NS + 1, true) : true;
        const int np = parse_nan_policy(args, nargs, NS + 2);
        const bool keep = arg_bool(args, nargs, NS + 3, false);
        if (s.size() < 2) fail("At least two samples are required; got %d.", (int)s.size());
        Spec sp;
        sp.too_small = [](const vector<int64_t> &n) {
            for (auto v : n) if (v == 0) return true;
            for (auto v : n) if (v != 1) return false;
            return true;
        };
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            const int k = b.K;
            for (int64_t i = 0; i < b.ns; i++) {
                vector<double> all;
                vector<int64_t> n;
                bool any0 = false, all1 = true;
                for (int g = 0; g < k; g++) {
                    n.push_back(b.n(i, g));
                    all.insert(all.end(), b.x(i, g), b.x(i, g) + b.n(i, g));
                    if (b.n(i, g) == 0) any0 = true;
                    if (b.n(i, g) != 1) all1 = false;
                }
                if (any0 || all1) { out[2 * i] = out[2 * i + 1] = NaN; continue; }
                if (!ev_ok) fail("Expected a boolean value for 'equal_var'");
                const int64_t bign = (int64_t)all.size();
                bool all_const = true;
                for (int g = 0; g < k; g++)
                    for (int64_t j = 0; j + 1 < n[g]; j++) if (!(b.x(i, g)[j + 1] - b.x(i, g)[j] == 0)) { all_const = false; break; }
                bool all_same = true;
                for (int64_t j = 0; j + 1 < bign; j++) if (!(all[j + 1] - all[j] == 0)) { all_same = false; break; }
                double f, dfn, dfd;
                if (equal_var) {
                    const double offset = psum(all.data(), bign) / (double)bign;
                    for (auto &v : all) v = v - offset;
                    const double sa = psum(all.data(), bign);
                    const double normalized_ss = sa * sa / (double)bign;
                    const double sstot = dot(all.data(), all.data(), bign) - normalized_ss;
                    double ssbn = 0;
                    vector<double> tmp;
                    for (int g = 0; g < k; g++) {
                        tmp.assign(b.x(i, g), b.x(i, g) + n[g]);
                        for (auto &v : tmp) v = v - offset;
                        const double sm = lsum(tmp.data(), n[g], b.sq(g));
                        ssbn = ssbn + sm * sm / (double)n[g];
                    }
                    ssbn = ssbn - normalized_ss;
                    const double sswn = sstot - ssbn;
                    dfn = (double)(k - 1);
                    dfd = (double)(bign - k);
                    const double msb = ssbn / dfn, msw = sswn / dfd;
                    f = msb / msw;
                } else {
                    vector<double> y((size_t)k), nt((size_t)k), s2((size_t)k), w((size_t)k);
                    for (int g = 0; g < k; g++) {
                        y[g] = lmean(b.x(i, g), n[g], b.sq(g));
                        nt[g] = (double)n[g];
                        s2[g] = np_var(b.x(i, g), n[g], 1, b.sq(g));
                        w[g] = nt[g] / s2[g];
                    }
                    const bool seqk = b.ns > 1;
                    const double swt = lsum(w.data(), k, seqk);
                    const double yhat = dot(w.data(), y.data(), k) / lsum(w.data(), k, seqk);
                    vector<double> t1((size_t)k), a1((size_t)k), a2((size_t)k);
                    for (int g = 0; g < k; g++) { const double dd = y[g] - yhat; t1[g] = dd * dd; }
                    const double numerator = dot(w.data(), t1.data(), k) / (double)(k - 1);
                    for (int g = 0; g < k; g++) { a1[g] = 1 / (nt[g] - 1); const double q = 1 - w[g] / swt; a2[g] = q * q; }
                    const double vd = dot(a1.data(), a2.data(), k);
                    const double denominator = 1 + 2 * (double)(k - 2) / (double)(k * k - 1) * vd;
                    f = numerator / denominator;
                    dfn = (double)(k - 1);
                    dfd = (double)(k * k - 1) / (3 * dot(a1.data(), a2.data(), k));
                }
                if (all_const) f = INF;
                if (all_same) f = NaN;
                out[2 * i] = f;
                out[2 * i + 1] = sc::fdtrc(dfn, dfd, f);
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

enum { C_MEDIAN, C_MEAN, C_TRIMMED };

int parse_center(const tsr_arg *args, int nargs, int k)
{
    const char *c = arg_str(args, nargs, k, "median");
    if (c && !strcmp(c, "median")) return C_MEDIAN;
    if (c && !strcmp(c, "mean")) return C_MEAN;
    if (c && !strcmp(c, "trimmed")) return C_TRIMMED;
    return -1;
}

/* the centre of one sample: np.median, np.mean or trim_mean */
double center_of(const double *x, int64_t n, int center, double prop, bool seq)
{
    if (center == C_MEAN) return lmean(x, n, seq);
    vector<double> a(x, x + n);
    std::sort(a.begin(), a.end(), lt_nan);
    if (center == C_MEDIAN) {
        if (n == 0) return NaN;
        const int64_t h = n / 2;
        if (n % 2) return a[h];
        return (a[h - 1] + a[h]) / 2.0;
    }
    if (n == 0) return NaN;
    if (!(std::fabs(prop * (double)n) < 9.2e18)) fail("cannot convert float NaN to integer");
    const int64_t lo = (int64_t)(prop * (double)n), hi = n - lo;
    if (lo > hi) fail("Proportion too big.");
    if (lo < 0) fail("kth(=%lld) out of bounds (%lld)", (long long)(hi - 1), (long long)n);   /* np.partition */
    return lmean(a.data() + lo, hi - lo, seq);
}

int f_levene(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* sample1..12, center='median', proportiontocut=0.05, axis=0, nan_policy, keepdims */
        const int NS = 12;
        vector<Arr> s = var_samples(args, nargs, NS);
        const int center = parse_center(args, nargs, NS);
        const double prop = arg_num(args, nargs, NS + 1, 0.05);
        const AxisOpt ax = parse_axis(args, nargs, NS + 2);
        const int np = parse_nan_policy(args, nargs, NS + 3);
        const bool keep = arg_bool(args, nargs, NS + 4, false);
        Spec sp;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            if (center < 0) fail("center must be 'mean', 'median' or 'trimmed'.");
            const int k = b.K;
            if (k < 2) fail("Must provide at least two samples.");
            for (int64_t i = 0; i < b.ns; i++) {
                vector<vector<double>> Z((size_t)k);
                vector<double> Zbar_i((size_t)k);
                int64_t Ntot = 0;
                for (int g = 0; g < k; g++) {
                    const int64_t n = b.n(i, g);
                    Ntot += n;
                    const double yc = center_of(b.x(i, g), n, center, prop, b.sq(g));
                    Z[g].resize((size_t)n);
                    for (int64_t j = 0; j < n; j++) Z[g][j] = std::fabs(b.x(i, g)[j] - yc);
                    Zbar_i[g] = lmean(Z[g].data(), n, b.sq(g));
                }
                double zs = 0;
                for (int g = 0; g < k; g++) zs = zs + (double)b.n(i, g) * Zbar_i[g];
                const double Zbar = zs / (double)Ntot;
                const double dfd = (double)(Ntot - k);
                double sn = 0;
                for (int g = 0; g < k; g++) { const double dd = Zbar_i[g] - Zbar; sn = sn + (double)b.n(i, g) * (dd * dd); }
                const double numer = dfd * sn;
                const double dfn = k - 1.0;
                double sd = 0;
                for (int g = 0; g < k; g++) {
                    vector<double> q(Z[g].size());
                    for (size_t j = 0; j < q.size(); j++) { const double dd = Z[g][j] - Zbar_i[g]; q[j] = dd * dd; }
                    sd = sd + lsum(q.data(), (int64_t)q.size(), b.sq(g));
                }
                const double denom = dfn * sd;
                const double W = numer / denom;
                out[2 * i] = W;
                out[2 * i + 1] = sc::fdtrc(dfn, dfd, W);
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

int f_bartlett(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* sample1..12, axis=0, nan_policy, keepdims */
        const int NS = 12;
        vector<Arr> s = var_samples(args, nargs, NS);
        const AxisOpt ax = parse_axis(args, nargs, NS);
        const int np = parse_nan_policy(args, nargs, NS + 1);
        const bool keep = arg_bool(args, nargs, NS + 2, false);
        Spec sp;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            const int k = b.K;
            if (k < 2) fail("Must enter at least two input sample vectors.");
            /* Ni has the samples' (promoted) dtype: setting NaN into an integer array raises */
            if (b.all_int()) fail("cannot convert float NaN to integer");
            const bool seqk = b.ns > 1;
            for (int64_t i = 0; i < b.ns; i++) {
                vector<double> Ni((size_t)k), ssq((size_t)k), t1((size_t)k), t2((size_t)k), t3((size_t)k);
                for (int g = 0; g < k; g++) {
                    const int64_t n = b.n(i, g);
                    Ni[g] = n == 0 ? NaN : (double)n;
                    ssq[g] = np_var(b.x(i, g), n, 1, b.sq(g));
                }
                const double Ntot = lsum(Ni.data(), k, seqk);
                for (int g = 0; g < k; g++) t1[g] = (Ni[g] - 1) * ssq[g];
                const double spsq = lsum(t1.data(), k, seqk) / (Ntot - k);
                for (int g = 0; g < k; g++) t2[g] = (Ni[g] - 1) * std::log(ssq[g]);
                const double numer = (Ntot - k) * std::log(spsq) - lsum(t2.data(), k, seqk);
                for (int g = 0; g < k; g++) t3[g] = 1 / (Ni[g] - 1);
                const double denom = 1 + 1 / (3.0 * (double)(k - 1)) * (lsum(t3.data(), k, seqk) - 1 / (Ntot - k));
                double T = numer / denom;
                const double p = sc::chdtrc((double)(k - 1), T);
                T = T < 0 ? 0. : T;
                out[2 * i] = T;
                out[2 * i + 1] = p;
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

int f_fligner(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* sample1..12, center='median', proportiontocut=0.05, axis=0, nan_policy, keepdims */
        const int NS = 12;
        vector<Arr> s = var_samples(args, nargs, NS);
        const int center = parse_center(args, nargs, NS);
        const double prop = arg_num(args, nargs, NS + 1, 0.05);
        const AxisOpt ax = parse_axis(args, nargs, NS + 2);
        const int np = parse_nan_policy(args, nargs, NS + 3);
        const bool keep = arg_bool(args, nargs, NS + 4, false);
        Spec sp;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            if (center < 0) fail("center must be 'mean', 'median' or 'trimmed'.");
            const int k = b.K;
            if (k < 2) fail("Must provide at least two samples.");
            for (int64_t i = 0; i < b.ns; i++) {
                bool empty = false;
                for (int g = 0; g < k; g++) if (b.n(i, g) == 0) empty = true;
                if (empty) { out[2 * i] = out[2 * i + 1] = NaN; continue; }
                vector<double> all;
                vector<int64_t> ni;
                for (int g = 0; g < k; g++) {
                    const int64_t n = b.n(i, g);
                    ni.push_back(n);
                    const double c = center_of(b.x(i, g), n, center, prop, b.sq(g));
                    for (int64_t j = 0; j < n; j++) all.push_back(std::fabs(b.x(i, g)[j] - c));
                }
                const int64_t N = (int64_t)all.size();
                vector<double> r((size_t)N);
                rank_avg(all.data(), N, r.data(), nullptr);
                vector<double> a((size_t)N);
                for (int64_t j = 0; j < N; j++) a[j] = sc::ndtri(r[j] / (2 * ((double)N + 1.0)) + 0.5);
                const double abar = psum(a.data(), N) / (double)N;
                const double V2 = np_var(a.data(), N, 1, false);
                double st = 0;
                int64_t off = 0;
                for (int g = 0; g < k; g++) {
                    const double Ai = psum(a.data() + off, ni[g]) / (double)ni[g];
                    const double dd = Ai - abar;
                    st = st + (double)ni[g] * (dd * dd);
                    off += ni[g];
                }
                const double stat = st / V2;
                out[2 * i] = stat;
                out[2 * i + 1] = sc::chdtrc((double)(k - 1), stat);
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

/* ---------------------------------------------------------------- Ansari-Bradley (AS 93, float32 as SciPy) */

void start1(float *a, int n)
{
    const int lout = 1 + (n / 2);
    for (int i = 0; i < lout; i++) a[i] = 2;
    if ((n % 2) == 0) a[lout - 1] = 1;
}
void start2(float *a, int n)
{
    const int odd = n % 2;
    float A = 1.f, B = 3.f;
    const float C = odd ? 2.f : 0.f;
    const int ndo = (n + 2 + odd) / 2 - odd;
    for (int i = 0; i < ndo; i++) { a[i] = A; A += B; B = 4 - B; }
    A = 1; B = 3;
    for (int i = n - odd; i > ndo - 1; i--) { a[i] = A + C; A += B; B = 4 - B; }
    if (odd == 1) a[(ndo * 2) - 1] = 2;
}
int frqadd(float *a, const float *b, int lenb, int offset)
{
    const float two = 2;
    const int lout = lenb + offset;
    for (int i = 0; i < lenb; i++) a[offset + i] += two * b[i];
    return lout;
}
int imply(float *a, int curlen, int reslen, float *b, int offset)
{
    int i2 = -offset;
    int j2 = reslen - offset;
    const int j2min = (j2 + 1) / 2 - 1;
    const int nextlenb = j2;
    int j1 = reslen - 1;
    float summ, diff;
    j2 -= 1;
    for (int i1 = 0; i1 < (reslen + 1) / 2; i1++) {
        if (i2 < 0) summ = a[i1];
        else { summ = a[i1] + b[i2]; a[i1] = summ; }
        i2 += 1;
        if (j2 >= j2min) {
            if (j1 > curlen - 1) diff = summ;
            else diff = summ - a[j1];
            b[i1] = diff;
            b[j2] = diff;
            j2 -= 1;
        }
        a[j1] = summ;
        j1 -= 1;
    }
    return nextlenb;
}
/* gscale(test, other): astart and the frequencies */
int gscale(int test, int other, vector<float> &a1)
{
    const int m = std::min(test, other), n = std::max(test, other);
    const int astart = ((test + 1) / 2) * (1 + (test / 2));
    const int LL = (test * other) / 2 + 1;
    const bool symm = (m + n) % 2 == 0;
    const int odd = n % 2;
    a1.assign((size_t)LL + 4, 0.f);
    vector<float> a2((size_t)LL + 4, 0.f), a3((size_t)LL + 4, 0.f);
    if (m == 0) { a1[0] = 1; a1.resize((size_t)LL); return astart; }
    if (m == 1) {
        start1(a1.data(), n);
        if (!(symm || (other > test))) { a1[0] = 1; a1[LL - 1] = 2; }
        a1.resize((size_t)LL);
        return astart;
    }
    if (m == 2) {
        start2(a1.data(), n);
        if (!(symm || (other > test)))
            for (int i = 0; i < LL / 2; i++) std::swap(a1[LL - 1 - i], a1[i]);
        a1.resize((size_t)LL);
        return astart;
    }
    int len1, len2 = 0, len3 = 0, n2b1, n2b2, part_no, loop_m = 3;
    if (odd) {
        start1(a1.data(), n);
        start2(a2.data(), n - 1);
        len1 = 1 + (n / 2); len2 = n; n2b1 = 1; n2b2 = 2;
        part_no = 0;
    } else {
        start2(a1.data(), n);
        start1(a2.data(), n - 1);
        start2(a3.data(), n - 2);
        len1 = n + 1; len2 = n / 2; len3 = n - 1; n2b1 = 2; n2b2 = 1;
        part_no = 1;
    }
    while (loop_m <= m) {
        if (part_no == 0) {
            const int l1out = frqadd(a1.data(), a2.data(), len2, n2b1);
            len1 += n;
            len3 = imply(a1.data(), l1out, len1, a3.data(), loop_m);
            n2b1 += 1;
            loop_m += 1;
            part_no = 1;
        } else {
            const int l2out = frqadd(a2.data(), a3.data(), len3, n2b2);
            len2 += n - 1;
            imply(a2.data(), l2out, len2, a3.data(), loop_m);
            n2b2 += 1;
            loop_m += 1;
            part_no = 0;
        }
    }
    if (!symm) {
        const int ks = (m + 3) / 2 - 1;
        for (int i = 0; i < len2; i++) a1[ks + i] += a2[i];
    }
    if (other > test)
        for (int i = 0; i < LL / 2; i++) std::swap(a1[LL - 1 - i], a1[i]);
    a1.resize((size_t)LL);
    return astart;
}

int f_ansari(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* x, y, alternative='two-sided', axis=0, nan_policy, keepdims */
        vector<Arr> s = {to_arr(args[0], "x"), to_arr(args[1], "y")};
        const Alt alt = parse_alt(nargs > 2 ? &args[2] : nullptr);
        const AxisOpt ax = parse_axis(args, nargs, 3);
        const int np = parse_nan_policy(args, nargs, 4);
        const bool keep = arg_bool(args, nargs, 5, false);
        Spec sp;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            if (alt == ALT_BAD) fail("'alternative' must be 'two-sided', 'greater', or 'less'.");
            const int64_t n = b.n(0, 0), m = b.n(0, 1);
            if (m < 1) fail("Not enough other observations.");
            if (n < 1) fail("Not enough test observations.");
            const int64_t N = m + n;
            vector<vector<double>> sym((size_t)b.ns);
            bool repeats = false;
            for (int64_t i = 0; i < b.ns; i++) {
                vector<double> xy(b.x(i, 0), b.x(i, 0) + n);
                xy.insert(xy.end(), b.x(i, 1), b.x(i, 1) + m);
                vector<double> r((size_t)N), t((size_t)N);
                rank_avg(xy.data(), N, r.data(), t.data());
                for (double v : t) if (v > 1) repeats = true;
                sym[i].resize((size_t)N);
                for (int64_t j = 0; j < N; j++) sym[i][j] = std::min(r[j], (double)N - r[j] + 1);
            }
            const bool exact = m < 55 && n < 55 && !repeats;
            vector<float> a1;
            vector<double> freqs;
            double total = 0, astart = 0;
            if (exact) {
                astart = gscale((int)n, (int)m, a1);
                freqs.assign(a1.begin(), a1.end());
                total = psum(freqs.data(), (int64_t)freqs.size());
            }
            const int64_t L = (int64_t)freqs.size();
            auto clampi = [&](int64_t e) { if (e < 0) e = std::max<int64_t>(L + e, 0); return std::min(e, L); };
            for (int64_t i = 0; i < b.ns; i++) {
                const double AB = psum(sym[i].data(), n);
                double p;
                if (exact) {
                    const int64_t ic = (int64_t)std::ceil(AB - astart), isf = (int64_t)std::floor(AB - astart);
                    const int64_t ce = clampi(ic + 1), ss = clampi(isf);
                    const double cdf = psum(freqs.data(), ce) / total;
                    const double sf = psum(freqs.data() + ss, L - ss) / total;
                    if (alt == ALT_TWO) p = 2.0 * std::min(cdf, sf);
                    else if (alt == ALT_GREATER) p = cdf;
                    else p = sf;
                    p = std::min(p, 1.0);
                } else {
                    const double Nd = (double)N;
                    const double mnAB = N % 2 ? (double)n * ((Nd + 1.0) * (Nd + 1.0)) / 4.0 / Nd : (double)n * (Nd + 2.0) / 4.0;
                    double varAB;
                    if (repeats) {
                        vector<double> q((size_t)N);
                        for (int64_t j = 0; j < N; j++) q[j] = sym[i][j] * sym[i][j];
                        const double fac = psum(q.data(), N);
                        if (N % 2) {
                            const double n1 = (double)((N + 1) * (N + 1) * (N + 1) * (N + 1));
                            varAB = (double)(m * n) * (16 * Nd * fac - n1) / (16.0 * (double)(N * N) * (double)(N - 1));
                        } else {
                            varAB = (double)(m * n) * (16 * fac - (double)(N * (N + 2) * (N + 2))) / (16.0 * Nd * (double)(N - 1));
                        }
                    } else {
                        if (N % 2) varAB = (double)(n * m) * (Nd + 1.0) * (double)(3 + N * N) / (48.0 * (double)(N * N));
                        else varAB = (double)(m * n) * (double)(N + 2) * (Nd - 2.0) / 48 / (Nd - 1.0);
                    }
                    const double z = (mnAB - AB) / std::sqrt(varAB);
                    p = pval_norm(z, alt);
                }
                out[2 * i] = AB;
                out[2 * i + 1] = p;
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

int f_mood(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* x, y, axis=0, alternative='two-sided', nan_policy, keepdims */
        vector<Arr> s = {to_arr(args[0], "x"), to_arr(args[1], "y")};
        const AxisOpt ax = parse_axis(args, nargs, 2);
        const Alt alt = parse_alt(nargs > 3 ? &args[3] : nullptr);
        const int np = parse_nan_policy(args, nargs, 4);
        const bool keep = arg_bool(args, nargs, 5, false);
        Spec sp;
        sp.too_small = [](const vector<int64_t> &n) { return n[0] + n[1] < 3; };
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            const int64_t m = b.n(0, 0), n = b.n(0, 1), N = m + n;
            if (m == 0 || n == 0 || N < 3) {
                for (int64_t i = 0; i < b.ns; i++) out[2 * i] = out[2 * i + 1] = NaN;
                return 2;
            }
            vector<vector<double>> rr((size_t)b.ns), tt((size_t)b.ns);
            bool ties = false;
            for (int64_t i = 0; i < b.ns; i++) {
                vector<double> xy(b.x(i, 0), b.x(i, 0) + m);
                xy.insert(xy.end(), b.x(i, 1), b.x(i, 1) + n);
                rr[i].resize((size_t)N);
                tt[i].resize((size_t)N);
                rank_avg(xy.data(), N, rr[i].data(), tt[i].data());
                for (double v : tt[i]) if (v > 1) ties = true;
            }
            const double Nd = (double)N;
            for (int64_t i = 0; i < b.ns; i++) {
                double z;
                if (!ties) {
                    vector<double> q((size_t)m);
                    for (int64_t j = 0; j < m; j++) { const double d = rr[i][j] - (Nd + 1.0) / 2; q[j] = d * d; }
                    const double M = psum(q.data(), m);
                    const double E0 = (double)m * (Nd * Nd - 1.0) / 12;
                    const double varM = (double)(m * n) * (Nd + 1.0) * (double)(N + 2) * (double)(N - 2) / 180;
                    z = (M - E0) / std::sqrt(varM);
                } else {
                    const vector<double> &t = tt[i];
                    const double E0 = (double)(m * (N * N - 1)) / 12;
                    vector<double> S((size_t)N + 1);
                    S[0] = 0;
                    for (int64_t j = 0; j < N; j++) S[j + 1] = S[j] + t[j];
                    vector<double> q((size_t)N), phi((size_t)N);
                    for (int64_t j = 0; j < N; j++) {
                        const double Si = S[j + 1], Sm = S[j];
                        const double w = Nd - Si - Sm;
                        q[j] = t[j] * (t[j] * t[j] - 1) * (t[j] * t[j] - 4 + (15 * (w * w)));
                    }
                    const double varM = (double)(m * n) * (Nd + 1.0) * (double)(N * N - 4) / 180 -
                                        (double)(m * n) / (double)(180 * N * (N - 1)) * psum(q.data(), N);
                    auto sI2 = [](double a) { return a * (a + 1) * (2 * a + 1) / 6; };
                    auto sI = [](double a) { return a * (a + 1) / 2; };
                    const double c = (Nd + 1) / 2;
                    for (int64_t j = 0; j < N; j++) {
                        const double a = S[j] + 1, bb = S[j + 1];
                        const double sum_I2 = sI2(bb) - sI2(a) + a * a;
                        const double sum_I = sI(bb) - sI(a) + a;
                        const double sum_1 = (bb - a) + 1;
                        phi[j] = t[j] == 0 ? 0. : (sum_I2 - 2 * c * sum_I + sum_1 * (c * c)) / t[j];
                    }
                    vector<double> xs(b.x(i, 0), b.x(i, 0) + m);
                    std::sort(xs.begin(), xs.end(), lt_nan);
                    vector<double> rx((size_t)m), ax2((size_t)m);
                    rank_avg(xs.data(), m, rx.data(), ax2.data());
                    vector<double> apad = ax2;
                    apad.resize((size_t)N, 0.0);
                    vector<double> xy = xs;
                    xy.insert(xy.end(), b.x(i, 1), b.x(i, 1) + n);
                    vector<int64_t> idx((size_t)N);
                    std::iota(idx.begin(), idx.end(), 0);
                    std::stable_sort(idx.begin(), idx.end(), [&](int64_t p1, int64_t p2) { return lt_nan(xy[p1], xy[p2]); });
                    vector<double> a((size_t)N);
                    for (int64_t j = 0; j < N; j++) a[j] = apad[(size_t)idx[j]];
                    const double T = dot(a.data(), phi.data(), N);
                    z = (T - E0) / std::sqrt(varM);
                }
                out[2 * i] = z;
                out[2 * i + 1] = pval_norm(z, alt);
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

int f_brunnermunzel(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* x, y, alternative='two-sided', distribution='t', nan_policy='propagate', axis=0, keepdims=False */
        vector<Arr> s = {to_arr(args[0], "x"), to_arr(args[1], "y")};
        const Alt alt = parse_alt(nargs > 2 ? &args[2] : nullptr);
        const char *dist = arg_str(args, nargs, 3, "t");
        const int np = parse_nan_policy(args, nargs, 4);
        const AxisOpt ax = parse_axis(args, nargs, 5);
        const bool keep = arg_bool(args, nargs, 6, false);
        Spec sp;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            const int64_t nx = b.n(0, 0), ny = b.n(0, 1), N = nx + ny;
            const bool tdist = dist && !strcmp(dist, "t");
            if (!tdist && !(dist && !strcmp(dist, "normal"))) fail("distribution should be 't' or 'normal'");
            for (int64_t i = 0; i < b.ns; i++) {
                vector<double> xy(b.x(i, 0), b.x(i, 0) + nx);
                xy.insert(xy.end(), b.x(i, 1), b.x(i, 1) + ny);
                vector<double> rc((size_t)N), rx((size_t)nx), ry((size_t)ny);
                rank_avg(xy.data(), N, rc.data(), nullptr);
                rank_avg(b.x(i, 0), nx, rx.data(), nullptr);
                rank_avg(b.x(i, 1), ny, ry.data(), nullptr);
                const double rcx_m = psum(rc.data(), nx) / (double)nx, rcy_m = psum(rc.data() + nx, ny) / (double)ny;
                const double rx_m = psum(rx.data(), nx) / (double)nx, ry_m = psum(ry.data(), ny) / (double)ny;
                vector<double> tx((size_t)nx), ty((size_t)ny);
                for (int64_t j = 0; j < nx; j++) tx[j] = rc[j] - rx[j] - rcx_m + rx_m;
                for (int64_t j = 0; j < ny; j++) ty[j] = rc[nx + j] - ry[j] - rcy_m + ry_m;
                double Sx = dot(tx.data(), tx.data(), nx);
                Sx /= (double)(nx - 1);
                double Sy = dot(ty.data(), ty.data(), ny);
                Sy /= (double)(ny - 1);
                double wbfn = (double)(nx * ny) * (rcy_m - rcx_m);
                wbfn /= (double)N * std::sqrt((double)nx * Sx + (double)ny * Sy);
                double p;
                if (tdist) {
                    const double a = (double)nx * Sx + (double)ny * Sy;
                    const double df_numer = a * a;
                    const double bx = (double)nx * Sx, by = (double)ny * Sy;
                    double df_denom = bx * bx / (double)(nx - 1);
                    df_denom += by * by / (double)(ny - 1);
                    const double df = df_numer / df_denom;
                    p = pval_t(-wbfn, df, alt);
                } else p = pval_norm(-wbfn, alt);
                out[2 * i] = wbfn;
                out[2 * i + 1] = p;
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

int f_ranksums(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* x, y, alternative='two-sided', axis=0, nan_policy, keepdims */
        vector<Arr> s = {to_arr(args[0], "x"), to_arr(args[1], "y")};
        const Alt alt = parse_alt(nargs > 2 ? &args[2] : nullptr);
        const AxisOpt ax = parse_axis(args, nargs, 3);
        const int np = parse_nan_policy(args, nargs, 4);
        const bool keep = arg_bool(args, nargs, 5, false);
        Spec sp;
        sp.vectorized = false;
        Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
            for (int64_t i = 0; i < b.ns; i++) {
                const int64_t n1 = b.n(i, 0), n2 = b.n(i, 1);
                vector<double> xy(b.x(i, 0), b.x(i, 0) + n1);
                xy.insert(xy.end(), b.x(i, 1), b.x(i, 1) + n2);
                vector<double> r(xy.size());
                rank_avg(xy.data(), (int64_t)xy.size(), r.data(), nullptr);
                const double sx = psum(r.data(), n1);
                const double expected = (double)(n1 * (n1 + n2 + 1)) / 2.0;
                const double z = (sx - expected) / std::sqrt((double)(n1 * n2 * (n1 + n2 + 1)) / 12.0);
                out[2 * i] = z;
                out[2 * i + 1] = pval_norm(z, alt);
            }
            return 2;
        });
        return emit(R, res, nres);
    });
}

int f_median_test(const void *, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return guarded([&] {
        /* sample1..12, ties='below', correction=True, lambda_=1, nan_policy='propagate' */
        const int NS = 12;
        vector<Arr> s;
        for (int k = 0; k < NS && k < nargs; k++) {
            if (args[k].kind == 0) break;
            s.push_back(to_arr(args[k], "sample"));
        }
        if (s.size() < 2) fail("median_test requires two or more samples.");
        const char *ties = arg_str(args, nargs, NS, "below");
        if (!ties || (strcmp(ties, "below") && strcmp(ties, "above") && strcmp(ties, "ignore")))
            fail("invalid 'ties' option '%s'; 'ties' must be one of: 'below', 'above', 'ignore'", ties ? ties : "?");
        const bool correction = arg_bool(args, nargs, NS + 1, true);
        double lam = 1;
        lambda_from_arg(nargs > NS + 2 ? &args[NS + 2] : nullptr, lam);
        const char *nps = arg_str(args, nargs, NS + 3, "propagate");
        for (size_t k = 0; k < s.size(); k++) {
            if (s[k].size() == 0) fail("Sample %d is empty. All samples must contain at least one value.", (int)k + 1);
            if (s[k].ndim() != 1) fail("Sample %d has %d dimensions. All samples must be one-dimensional sequences.", (int)k + 1, s[k].ndim());
        }
        vector<double> cdata;
        for (auto &a : s) cdata.insert(cdata.end(), a.v.begin(), a.v.end());
        bool has_nan = false;
        for (double v : cdata) if (v != v) has_nan = true;
        if (!nps || (strcmp(nps, "propagate") && strcmp(nps, "omit") && strcmp(nps, "raise")))
            fail("nan_policy must be one of {'propagate', 'raise', 'omit'}");
        if (has_nan && !strcmp(nps, "raise")) fail("The input contains nan values");
        if (has_nan && !strcmp(nps, "propagate")) {
            /* SciPy returns MedianTestResult(nan, nan, nan, None): the table is absent */
            fn_result_num(&res[0], NaN);
            fn_result_num(&res[1], NaN);
            fn_result_num(&res[2], NaN);
            return TSR_OK;
        }
        vector<double> c;
        for (double v : cdata) if (v == v) c.push_back(v);
        std::sort(c.begin(), c.end(), lt_nan);
        const int64_t nc = (int64_t)c.size();
        double gm = NaN;
        if (nc > 0) gm = nc % 2 ? c[nc / 2] : (c[nc / 2 - 1] + c[nc / 2]) / 2.0;
        const int k = (int)s.size();
        vector<double> table((size_t)(2 * k), 0.0);
        for (int g = 0; g < k; g++) {
            int64_t above = 0, below = 0, size = 0;
            for (double v : s[g].v) {
                if (v != v) continue;
                size++;
                above += v > gm;
                below += v < gm;
            }
            const int64_t eq = size - (above + below);
            table[g] += (double)above;
            table[k + g] += (double)below;
            if (!strcmp(ties, "below")) table[k + g] += (double)eq;
            else if (!strcmp(ties, "above")) table[g] += (double)eq;
        }
        double r0 = 0, r1 = 0;
        for (int g = 0; g < k; g++) { r0 += table[g]; r1 += table[k + g]; }
        if (r0 == 0) fail("All values are below the grand median (%g).", gm);
        if (r1 == 0) fail("All values are above the grand median (%g).", gm);
        if (!strcmp(ties, "ignore"))
            for (int g = 0; g < k; g++)
                if (table[g] == 0 && table[k + g] == 0) fail("All values in sample %d are equal to the grand median, so they are ignored, resulting in an empty sample.", g + 1);
        Contingency C = chi2_contingency_calc(table, {2, (int64_t)k}, correction, lam);
        fn_result_num(&res[0], C.stat);
        fn_result_num(&res[1], C.p);
        fn_result_num(&res[2], gm);
        const int64_t shp[2] = {2, (int64_t)k};
        int64_t *tp = (int64_t *)fn_result_array(&res[3], TSR_I64, 2, shp);
        if (!tp) return TSR_ENOMEM;
        for (int j = 0; j < 2 * k; j++) tp[j] = (int64_t)table[j];
        return TSR_OK;
    });
}

/* ================================================================ normality tests */

/* skew(a) (bias=True) and kurtosis(a, fisher) on one slice */
double skew1(const double *a, int64_t n, bool seq)
{
    const double mean = lmean(a, n, seq);
    const double m2 = moment_c(a, n, mean, 2, seq), m3 = moment_c(a, n, mean, 3, seq);
    const double eps = 2.220446049250313e-16;
    const double em = eps * mean;
    const bool zero = m2 <= em * em;
    return zero ? NaN : m3 / std::pow(m2, 1.5);
}
double kurt1(const double *a, int64_t n, bool fisher, bool seq)
{
    const double mean = lmean(a, n, seq);
    const double m2 = moment_c(a, n, mean, 2, seq), m4 = moment_c(a, n, mean, 4, seq);
    const double eps = 2.220446049250313e-16;
    const double em = eps * mean;
    const bool zero = m2 <= em * em;
    const double v = zero ? NaN : m4 / (m2 * m2);
    return fisher ? v - 3 : v;
}

double skewtest_z(const double *a, int64_t n_, bool seq)
{
    const double b2 = skew1(a, n_, seq);
    double n = (double)n_;
    if (n < 8) n = NaN;
    double y = b2 * std::sqrt(((n + 1) * (n + 3)) / (6.0 * (n - 2)));
    const double beta2 = (3.0 * (n * n + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2.0) * (n + 5) * (n + 7) * (n + 9)));
    const double W2 = -1 + std::sqrt(2 * (beta2 - 1));
    const double delta = 1 / std::sqrt(0.5 * std::log(W2));
    const double alpha = std::sqrt(2.0 / (W2 - 1));
    if (y == 0) y = 1.;
    const double ya = y / alpha;
    return delta * std::log(ya + std::sqrt(ya * ya + 1));
}

double kurtosistest_z(const double *a, int64_t n_, bool seq, bool nd)
{
    const double b2 = kurt1(a, n_, false, seq);
    double n = (double)n_;
    if (n < 5) n = NaN;
    /* NumPy scalars (1-D input) use pow(x, 0.5); arrays take ndarray's sqrt fast path */
    auto p05 = [nd](double x) { return nd ? std::sqrt(x) : std::pow(x, 0.5); };
    const double E = 3.0 * (n - 1) / (n + 1);
    const double varb2 = 24.0 * n * (n - 2) * (n - 3) / ((n + 1) * (n + 1.) * (n + 3) * (n + 5));
    const double x = (b2 - E) / p05(varb2);
    const double sqrtbeta1 = 6.0 * (n * n - 5 * n + 2) / ((n + 7) * (n + 9)) * p05((6.0 * (n + 3) * (n + 5)) / (n * (n - 2) * (n - 3)));
    const double A = 6.0 + 8.0 / sqrtbeta1 * (2.0 / sqrtbeta1 + p05(1 + 4.0 / (sqrtbeta1 * sqrtbeta1)));
    const double term1 = 1 - 2 / (9.0 * A);
    const double denom = 1 + x * p05(2 / (A - 4.0));
    const double sgn = denom > 0 ? 1. : denom < 0 ? -1. : denom == 0 ? 0. : NaN;
    const double term2 = sgn * (denom == 0.0 ? NaN : std::pow((1 - 2.0 / A) / std::fabs(denom), 1.0 / 3.0));
    return (term1 - term2) / p05(2 / (9.0 * A));
}

int one_sample_test(const tsr_arg *args, int nargs, tsr_result *res, int nres, int which)
{
    /* skewtest / kurtosistest: a, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False
       normaltest: a, axis=0, nan_policy='propagate', keepdims=False
       jarque_bera: x, axis=None, nan_policy='propagate', keepdims=False */
    vector<Arr> s = {to_arr(args[0], "a")};
    const AxisOpt ax = parse_axis(args, nargs, 1, which == 3);
    const int np = parse_nan_policy(args, nargs, 2);
    Alt alt = ALT_TWO;
    int kpos = 3;
    if (which <= 1) { alt = parse_alt(nargs > 3 ? &args[3] : nullptr); kpos = 4; }
    const bool keep = arg_bool(args, nargs, kpos, false);
    Spec sp;
    if (which == 0 || which == 2) sp.too_small = [](const vector<int64_t> &n) { return n[0] <= 7; };
    else if (which == 1) sp.too_small = [](const vector<int64_t> &n) { return n[0] <= 4; };
    Results R = run_axis_nan_policy(s, sp, ax, np, keep, [&](const Batch &b, double *out) {
        for (int64_t i = 0; i < b.ns; i++) {
            const double *a = b.x(i, 0);
            const int64_t n = b.n(i, 0);
            if (which == 0) {
                const double Z = skewtest_z(a, n, b.sq(0));
                out[2 * i] = Z;
                out[2 * i + 1] = pval_norm(Z, alt);
            } else if (which == 1) {
                const double Z = kurtosistest_z(a, n, b.sq(0), b.nd);
                out[2 * i] = Z;
                out[2 * i + 1] = pval_norm(Z, alt);
            } else if (which == 2) {
                const double s1 = skewtest_z(a, n, b.sq(0)), k1 = kurtosistest_z(a, n, b.sq(0), b.nd);
                const double st = s1 * s1 + k1 * k1;
                out[2 * i] = st;
                out[2 * i + 1] = sc::chdtrc(2., st);
            } else {
                const double mu = lmean(a, n, b.sq(0));
                vector<double> d((size_t)n);
                for (int64_t j = 0; j < n; j++) d[j] = a[j] - mu;
                const double sk = skew1(d.data(), n, b.sq(0)), ku = kurt1(d.data(), n, true, b.sq(0));
                const double st = (double)n / 6 * (sk * sk + ku * ku / 4);
                out[2 * i] = st;
                out[2 * i + 1] = sc::chdtrc(2., st);
            }
        }
        return 2;
    });
    return emit(R, res, nres);
}

int f_skewtest(const void *, const tsr_arg *a, int n, tsr_result *r, int nr) { return guarded([&] { return one_sample_test(a, n, r, nr, 0); }); }
int f_kurtosistest(const void *, const tsr_arg *a, int n, tsr_result *r, int nr) { return guarded([&] { return one_sample_test(a, n, r, nr, 1); }); }
int f_normaltest(const void *, const tsr_arg *a, int n, tsr_result *r, int nr) { return guarded([&] { return one_sample_test(a, n, r, nr, 2); }); }
int f_jarque_bera(const void *, const tsr_arg *a, int n, tsr_result *r, int nr) { return guarded([&] { return one_sample_test(a, n, r, nr, 3); }); }

#define SAMPLES12 "sample1, sample2=None, sample3=None, sample4=None, sample5=None, sample6=None, sample7=None, " \
                  "sample8=None, sample9=None, sample10=None, sample11=None, sample12=None"

const fn_def DEFS[] = {
    ROUTINE("stats.ttest_1samp", 3, "a, popmean, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False",
            "statistic, pvalue, df", f_ttest_1samp, NULL,
            "T-test for the mean of one group of scores (scipy.stats.ttest_1samp)."),
    ROUTINE("stats.ttest_ind", 3,
            "a, b, axis=0, equal_var=True, nan_policy='propagate', alternative='two-sided', trim=0, method=None, keepdims=False",
            "statistic, pvalue, df", f_ttest_ind, NULL,
            "T-test for the means of two independent samples; Welch's test with equal_var=False, Yuen's with trim (scipy.stats.ttest_ind)."),
    ROUTINE("stats.ttest_rel", 3, "a, b, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False",
            "statistic, pvalue, df", f_ttest_rel, NULL, "T-test on two related samples (scipy.stats.ttest_rel)."),
    ROUTINE("stats.ttest_ind_from_stats", 2, "mean1, std1, nobs1, mean2, std2, nobs2, equal_var=True, alternative='two-sided'",
            "statistic, pvalue", f_ttest_ind_from_stats, NULL,
            "T-test for the means of two independent samples from descriptive statistics (scipy.stats.ttest_ind_from_stats)."),
    ROUTINE("stats.chisquare", 2, "f_obs, f_exp=None, ddof=0, axis=0, sum_check=True, nan_policy='propagate', keepdims=False",
            "statistic, pvalue", f_chisquare, NULL, "Pearson's chi-squared test (scipy.stats.chisquare)."),
    ROUTINE("stats.power_divergence", 2, "f_obs, f_exp=None, ddof=0, axis=0, lambda_=None, nan_policy='propagate', keepdims=False",
            "statistic, pvalue", f_power_divergence, NULL,
            "Cressie-Read power divergence statistic and goodness of fit test (scipy.stats.power_divergence)."),
    ROUTINE("stats.chi2_contingency", 4, "observed, correction=True, lambda_=None, method=None",
            "statistic, pvalue, dof, expected_freq", f_chi2_contingency, NULL,
            "Chi-square test of independence of variables in a contingency table (scipy.stats.chi2_contingency)."),
    ROUTINE("stats.fisher_exact", 2, "table, alternative=None, method=None", "statistic, pvalue", f_fisher_exact, NULL,
            "Fisher exact test on a 2x2 contingency table (scipy.stats.fisher_exact)."),
    ROUTINE("stats.binomtest", 5, "k, n, p=0.5, alternative='two-sided'", "statistic, pvalue, k, n, proportion_estimate",
            f_binomtest, NULL, "Test that the probability of success is p (scipy.stats.binomtest)."),
    ROUTINE("stats.mannwhitneyu", 2,
            "x, y, use_continuity=True, alternative='two-sided', axis=0, method='auto', nan_policy='propagate', keepdims=False",
            "statistic, pvalue", f_mannwhitneyu, NULL, "Mann-Whitney U rank test on two independent samples (scipy.stats.mannwhitneyu)."),
    ROUTINE("stats.wilcoxon", 3,
            "x, y=None, zero_method='wilcox', correction=False, alternative='two-sided', method='auto', axis=0, nan_policy='propagate', keepdims=False",
            "statistic, pvalue, zstatistic", f_wilcoxon, NULL,
            "Wilcoxon signed-rank test; zstatistic with method='asymptotic' / 'approx' (scipy.stats.wilcoxon)."),
    ROUTINE("stats.kruskal", 2, SAMPLES12 ", nan_policy='propagate', axis=0, keepdims=False", "statistic, pvalue", f_kruskal, NULL,
            "Kruskal-Wallis H-test for independent samples (scipy.stats.kruskal)."),
    ROUTINE("stats.friedmanchisquare", 2, SAMPLES12 ", axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue",
            f_friedmanchisquare, NULL, "Friedman test for repeated samples (scipy.stats.friedmanchisquare)."),
    ROUTINE("stats.f_oneway", 2, SAMPLES12 ", axis=0, equal_var=True, nan_policy='propagate', keepdims=False", "statistic, pvalue",
            f_f_oneway, NULL, "One-way ANOVA; Welch's ANOVA with equal_var=False (scipy.stats.f_oneway)."),
    ROUTINE("stats.levene", 2, SAMPLES12 ", center='median', proportiontocut=0.05, axis=0, nan_policy='propagate', keepdims=False",
            "statistic, pvalue", f_levene, NULL, "Levene test for equal variances (scipy.stats.levene)."),
    ROUTINE("stats.bartlett", 2, SAMPLES12 ", axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue", f_bartlett, NULL,
            "Bartlett's test for equal variances (scipy.stats.bartlett)."),
    ROUTINE("stats.fligner", 2, SAMPLES12 ", center='median', proportiontocut=0.05, axis=0, nan_policy='propagate', keepdims=False",
            "statistic, pvalue", f_fligner, NULL, "Fligner-Killeen test for equality of variance (scipy.stats.fligner)."),
    ROUTINE("stats.ansari", 2, "x, y, alternative='two-sided', axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue",
            f_ansari, NULL, "Ansari-Bradley test for equal scale parameters (scipy.stats.ansari)."),
    ROUTINE("stats.mood", 2, "x, y, axis=0, alternative='two-sided', nan_policy='propagate', keepdims=False", "statistic, pvalue",
            f_mood, NULL, "Mood's test for equal scale parameters (scipy.stats.mood)."),
    ROUTINE("stats.brunnermunzel", 2,
            "x, y, alternative='two-sided', distribution='t', nan_policy='propagate', axis=0, keepdims=False", "statistic, pvalue",
            f_brunnermunzel, NULL, "Brunner-Munzel test on samples x and y (scipy.stats.brunnermunzel)."),
    ROUTINE("stats.ranksums", 2, "x, y, alternative='two-sided', axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue",
            f_ranksums, NULL, "Wilcoxon rank-sum statistic for two samples (scipy.stats.ranksums)."),
    ROUTINE("stats.median_test", 4, SAMPLES12 ", ties='below', correction=True, lambda_=1, nan_policy='propagate'",
            "statistic, pvalue, median, table", f_median_test, NULL, "Mood's median test (scipy.stats.median_test)."),
    ROUTINE("stats.normaltest", 2, "a, axis=0, nan_policy='propagate', keepdims=False", "statistic, pvalue", f_normaltest, NULL,
            "D'Agostino and Pearson's test for normality (scipy.stats.normaltest)."),
    ROUTINE("stats.skewtest", 2, "a, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False", "statistic, pvalue",
            f_skewtest, NULL, "Test whether the skew is different from the normal distribution (scipy.stats.skewtest)."),
    ROUTINE("stats.kurtosistest", 2, "a, axis=0, nan_policy='propagate', alternative='two-sided', keepdims=False",
            "statistic, pvalue", f_kurtosistest, NULL,
            "Test whether a dataset has normal kurtosis (scipy.stats.kurtosistest)."),
    ROUTINE("stats.jarque_bera", 2, "x, axis=None, nan_policy='propagate', keepdims=False", "statistic, pvalue", f_jarque_bera, NULL,
            "Jarque-Bera goodness of fit test on sample data (scipy.stats.jarque_bera)."),
};

}  // namespace

extern "C" const fn_table TSR_STATS_FN_TESTS_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
