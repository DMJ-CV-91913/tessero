/*
 * scipy.stats correlation and regression functions for the function registry (SciPy 1.17.1):
 * pearsonr, spearmanr, spearmanrho, kendalltau, weightedtau, pointbiserialr, chatterjeexi, somersd,
 * linregress, theilslopes, siegelslopes, binned_statistic(_2d, _dd), quantile, boxcox and yeojohnson with
 * a given lambda.
 *
 * Arithmetic follows SciPy's code operation by operation. Where SciPy reaches BLAS through NumPy the
 * summation order of the OpenBLAS kernels NumPy ships with (scipy-openblas 0.3.31, SkylakeX/Haswell ddot,
 * the small-matrix dgemm kernel) is reproduced:
 *   - np.vecdot (pearsonr's r, quantile's harrell-davis): cblas_ddot. Unit stride: blocks of 32 in four
 *     8-wide FMA accumulators, then blocks of 16 in four 4-wide accumulators (the 512-bit accumulators folded
 *     to 256 bits), a pairwise horizontal sum, and a sequential FMA tail for n % 16. Other strides: two
 *     accumulators over groups of four products (t1 += m1 + m3; t2 += m2 + m4), FMA tail.
 *   - np.cov / np.corrcoef (linregress, spearmanr): dgemm of the centred data, each entry a sequential FMA
 *     chain over the observations (exact for K <= 384 and up to 8 variables; larger problems are blocked by
 *     OpenBLAS in an order not reproduced here).
 * The decorator scipy.stats._axis_nan_policy (axis, nan_policy, keepdims, too-small samples) is the
 * registry's gufunc engine for the functions with at most four results, and `drive()` below for the others.
 */
#include "stats_dist.hpp"
#include "u128.hpp"
#include "../src/fn.h"

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <functional>
#include <numeric>
#include <string>
#include <vector>

namespace {

using V = std::vector<double>;
using VI = std::vector<int64_t>;
constexpr double NaN = tsd::NaN;
constexpr double INF = tsd::INF;

int fail(const char *fmt, ...)
{
    char buf[400];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    fn_set_error("%s", buf);
    return TSR_EARG;
}

inline double psum(const double *x, int64_t n) { return n <= 0 ? 0.0 : tsr_psum(x, n); }

/* NumPy's sort order: NaN last */
inline bool lt_nan(double a, double b) { return a < b || (b != b && a == a); }

void argsort_stable(const double *x, int64_t n, VI &idx)
{
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [x](int64_t i, int64_t j) { return lt_nan(x[i], x[j]); });
}

/* ---------------------------------------------------------------- BLAS summation orders */

/* cblas_ddot, unit strides (OpenBLAS kernel/x86_64/ddot.c with the SkylakeX micro-kernel) */
double ddot_unit(int64_t n, const double *x, const double *y)
{
    const int64_t n1 = n & -16;
    double dot = 0.0;
    if (n1) {
        double z[4][8] = {{0}};
        int64_t i = 0;
        const int64_t n32 = n1 & ~(int64_t)31;
        for (; i < n32; i += 32)
            for (int a = 0; a < 4; a++)
                for (int l = 0; l < 8; l++) z[a][l] = std::fma(x[i + 8 * a + l], y[i + 8 * a + l], z[a][l]);
        double acc[4][4];
        for (int a = 0; a < 4; a++)
            for (int l = 0; l < 4; l++) acc[a][l] = z[a][l] + z[a][l + 4];
        for (; i < n1; i += 16)
            for (int a = 0; a < 4; a++)
                for (int l = 0; l < 4; l++) acc[a][l] = std::fma(x[i + 4 * a + l], y[i + 4 * a + l], acc[a][l]);
        double s[4];
        for (int l = 0; l < 4; l++) s[l] = ((acc[0][l] + acc[1][l]) + acc[2][l]) + acc[3][l];
        dot = (s[0] + s[2]) + (s[1] + s[3]);
    }
    for (int64_t i = n1; i < n; i++) dot = std::fma(y[i], x[i], dot);
    return dot;
}

/* cblas_ddot with a non-unit stride (the values gathered contiguously) */
double ddot_strided(int64_t n, const double *x, const double *y)
{
    double t1 = 0.0, t2 = 0.0;
    int64_t i = 0;
    const int64_t n1 = n & -4;
    for (; i < n1; i += 4) {
        const double m1 = y[i] * x[i], m2 = y[i + 1] * x[i + 1], m3 = y[i + 2] * x[i + 2], m4 = y[i + 3] * x[i + 3];
        t1 += m1 + m3;
        t2 += m2 + m4;
    }
    for (; i < n; i++) t1 = std::fma(y[i], x[i], t1);
    return t1 + t2;
}

inline double ddot(int64_t n, const double *x, const double *y, bool strided)
{
    return strided ? ddot_strided(n, x, y) : ddot_unit(n, x, y);
}

/* one entry of dgemm (small-matrix kernel): a sequential FMA chain */
double gemm_dot(int64_t n, const double *a, const double *b)
{
    double s = 0.0;
    for (int64_t k = 0; k < n; k++) s = std::fma(a[k], b[k], s);
    return s;
}

/* ---------------------------------------------------------------- small NumPy pieces */

/* np.median of a copy (NaN if any NaN or empty) */
double np_median(V v)
{
    const int64_t n = (int64_t)v.size();
    if (n == 0) return NaN;
    for (double t : v)
        if (t != t) return NaN;
    std::sort(v.begin(), v.end());
    if (n % 2) return v[n / 2];
    const double two[2] = {v[n / 2 - 1], v[n / 2]};
    return psum(two, 2) / 2.0;
}

/* scipy.stats.rankdata(method='average') on NaN-free data */
void rank_average(const double *x, int64_t n, double *out)
{
    VI j;
    argsort_stable(x, n, j);
    int64_t s = 0;
    while (s < n) {
        int64_t e = s + 1;
        while (e < n && x[j[e]] == x[j[e - 1]]) e++;
        const double r = (double)(s + 1) + ((double)(e - s) - 1.0) / 2.0;
        for (int64_t k = s; k < e; k++) out[j[k]] = r;
        s = e;
    }
}

/* rankdata(method='max') */
void rank_max(const double *x, int64_t n, double *out)
{
    VI j;
    argsort_stable(x, n, j);
    int64_t s = 0;
    while (s < n) {
        int64_t e = s + 1;
        while (e < n && x[j[e]] == x[j[e - 1]]) e++;
        for (int64_t k = s; k < e; k++) out[j[k]] = (double)e;
        s = e;
    }
}

/* _get_pvalue with a symmetric distribution given by its cdf and sf */
template <class C, class S> double pvalue_of(double stat, int alt, C cdf, S sf)
{
    if (alt == 1) return cdf(stat);
    if (alt == 2) return sf(stat);
    return 2 * sf(std::fabs(stat));
}

double norm_p(double z, int alt)
{
    return pvalue_of(z, alt, [](double v) { return sc::ndtr(v); }, [](double v) { return sc::ndtr(-v); });
}

double t_p(double df, double t, int alt)
{
    return pvalue_of(t, alt, [df](double v) { return sc::stdtr(df, v); }, [df](double v) { return sc::stdtr(df, -v); });
}

/* x ** y on a Python float or NumPy float64 scalar: the C library's pow(). GCC folds pow(x, 2.0) into x * x,
   which glibc's pow does not always equal to the last bit, so the exponent is hidden from the optimiser. */
double py_pow(double x, double y)
{
    volatile double yy = y;
    return std::pow(x, (double)yy);
}

/* Python's max(a, b) / min(a, b) with a float first argument */
inline double py_max(double a, double b) { return b > a ? b : a; }
inline double py_min(double a, double b) { return b < a ? b : a; }

/* numpy.clip(x, lo, hi): NaN stays NaN */
inline double np_clip(double x, double lo, double hi)
{
    const double t = x < lo ? lo : x;              /* minimum(maximum(x, lo), hi) */
    return t > hi ? hi : t;
}

/* double-double helpers for Python's exact integer arithmetic (factorials) */
struct DD {
    double hi, lo;
};
DD dd_norm(double a, double b)
{
    const double s = a + b;
    return {s, b - (s - a)};
}
DD dd_mul(DD a, double k)
{
    const double p = a.hi * k, e = std::fma(a.hi, k, -p);
    return dd_norm(p, a.lo * k + e);
}
DD dd_fact(int64_t n)
{
    DD f{1.0, 0.0};
    for (int64_t k = 2; k <= n; k++) f = dd_mul(f, (double)k);
    return f;
}
DD dd_div(DD a, double k)
{
    const double h = a.hi / k;
    const double rem = std::fma(-h, k, a.hi) + a.lo;
    return dd_norm(h, rem / k);
}
/* 1 / n! without overflow */
DD dd_invfact(int64_t n)
{
    DD f{1.0, 0.0};
    for (int64_t k = 2; k <= n; k++) f = dd_div(f, (double)k);
    return f;
}
/* correctly rounded (up to double-double accuracy) value of 1 / (d * k) */
double dd_recip_div(DD d, double k)
{
    const double q = 1.0 / d.hi;
    const double r = std::fma(-q, d.hi, 1.0) - q * d.lo;
    const DD qq = dd_norm(q, q * r);
    const double h = qq.hi / k;
    const double rem = std::fma(-h, k, qq.hi) + qq.lo;
    return h + rem / k;
}

/* ================================================================ pearsonr */

/* pearsonr on gathered slices. strided: np.vecdot would call ddot with a non-unit stride (the axis is not the
   innermost one); seq: np.add.reduce sums sequentially. Returns TSR_OK or an error. */
int pearson_core(const double *x, const double *y, int64_t n, int alt, bool seq, bool strided, double &r_out,
                 double &p_out)
{
    const unsigned flags = seq ? FN_SEQUENTIAL : 0u;
    bool cx = true, cy = true;
    for (int64_t i = 1; i < n; i++) {
        if (!(x[i] == x[0])) cx = false;
        if (!(y[i] == y[0])) cy = false;
    }
    V xm(n), ym(n), t(n);
    const double xmean = fn_sum(x, n, flags) / (double)n;
    const double ymean = fn_sum(y, n, flags) / (double)n;
    double xmax = 0, ymax = 0;
    bool xn = false, yn = false;
    for (int64_t i = 0; i < n; i++) {
        xm[i] = x[i] - xmean;
        ym[i] = y[i] - ymean;
        const double ax = std::fabs(xm[i]), ay = std::fabs(ym[i]);
        if (ax != ax) xn = true;
        else if (ax > xmax) xmax = ax;
        if (ay != ay) yn = true;
        else if (ay > ymax) ymax = ay;
    }
    if (xn) xmax = NaN;
    if (yn) ymax = NaN;
    for (int64_t i = 0; i < n; i++) { const double v = xm[i] / xmax; t[i] = v * v; }
    const double normxm = xmax * std::sqrt(fn_sum(t.data(), n, flags));
    for (int64_t i = 0; i < n; i++) { const double v = ym[i] / ymax; t[i] = v * v; }
    const double normym = ymax * std::sqrt(fn_sum(t.data(), n, flags));
    for (int64_t i = 0; i < n; i++) {
        xm[i] = xm[i] / normxm;
        ym[i] = ym[i] / normym;
    }
    double r = ddot(n, xm.data(), ym.data(), strided);
    r = np_clip(r, -1.0, 1.0);
    if (cx || cy) r = NaN;
    const double ab = (double)n / 2 - 1;
    double p;
    if (alt == 1) p = sc::betainc(ab, ab, (r - (-1.0)) / 2);
    else if (alt == 2) p = sc::betaincc(ab, ab, (r - (-1.0)) / 2);
    else p = 2 * sc::betaincc(ab, ab, (std::fabs(r) - (-1.0)) / 2);
    if (n == 2) {
        r = std::nearbyint(r);
        p = r != r ? NaN : 1.0;
    }
    r_out = r;
    p_out = p;
    return TSR_OK;
}

int c_pearsonr(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
               const int64_t *, void *, unsigned flags)
{
    if (n[0] != n[1]) return fail("`x` and `y` must have the same length along `axis`.");
    if (n[0] < 2) return fail("`x` and `y` must have length at least 2.");
    const bool seq = (flags & FN_SEQUENTIAL) != 0;
    return pearson_core(x[0], x[1], n[0], (int)p[0], seq, seq, out[0][0], out[1][0]);
}

/* pointbiserialr: pearsonr on each (contiguous) slice; too_small = 1 */
int c_pointbiserialr(const void *, const double *const *x, const int64_t *n, const double *, double *const *out,
                     const int64_t *, void *, unsigned)
{
    if (n[0] != n[1]) return fail("Array shapes are incompatible for broadcasting.");
    if (n[0] <= 1) { out[0][0] = out[1][0] = NaN; return TSR_OK; }
    return pearson_core(x[0], x[1], n[0], 0, false, false, out[0][0], out[1][0]);
}

/* spearmanrho: pearsonr of the average ranks (rankdata returns contiguous arrays) */
int c_spearmanrho(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
                  const int64_t *, void *, unsigned)
{
    if (n[0] != n[1]) return fail("Array shapes are incompatible for broadcasting.");
    const int64_t m = n[0];
    if (m <= 1) { out[0][0] = out[1][0] = NaN; return TSR_OK; }
    V rx(m), ry(m);
    rank_average(x[0], m, rx.data());
    rank_average(x[1], m, ry.data());
    return pearson_core(rx.data(), ry.data(), m, (int)p[0], false, false, out[0][0], out[1][0]);
}

/* ================================================================ chatterjeexi */

int c_chatterjeexi(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
                   const int64_t *, void *, unsigned)
{
    if (n[0] != n[1]) return fail("Array shapes are incompatible for broadcasting.");
    const int64_t m = n[0];
    if (m <= 1) { out[0][0] = out[1][0] = NaN; return TSR_OK; }
    const bool ycont = p[0] != 0;
    VI j;
    argsort_stable(x[0], m, j);        /* NumPy's default (unstable) argsort: ties in x are ordered differently */
    V ys(m), neg(m), r(m), l(m), t(m);
    for (int64_t i = 0; i < m; i++) { ys[i] = x[1][j[i]]; neg[i] = -ys[i]; }
    rank_max(ys.data(), m, r.data());
    rank_max(neg.data(), m, l.data());
    for (int64_t i = 0; i + 1 < m; i++) t[i] = std::fabs(r[i + 1] - r[i]);
    const double num = psum(t.data(), m - 1);
    const double dn = (double)m;
    double xi;
    if (ycont) {
        xi = 1 - 3 * num / (double)(m * m - 1);
    } else {
        for (int64_t i = 0; i < m; i++) t[i] = (dn - l[i]) * l[i];
        const double den = 2 * psum(t.data(), m);
        xi = 1 - dn * num / den;
    }
    double sd;
    if (ycont) {
        sd = std::sqrt(2.0 / 5) / std::sqrt(dn);
    } else {
        V u(r), v(m);
        std::sort(u.begin(), u.end(), lt_nan);
        double acc = 0;
        for (int64_t i = 0; i < m; i++) { acc = i ? acc + u[i] : u[i]; v[i] = acc; }
        /* 1/n**k with Python integers: correctly rounded reciprocals */
        using tsr128::i128;
        const i128 n3 = i128((int64_t)m) * i128((int64_t)m) * i128((int64_t)m), n4 = n3 * i128((int64_t)m), n5 = n4 * i128((int64_t)m);
        auto recip = [](i128 N) {
            const double hi = tsr128::to_double(N);
            const double lo = tsr128::to_double(N - tsr128::from_double(hi));
            return dd_recip_div(dd_norm(hi, lo), 1.0);
        };
        for (int64_t i = 0; i < m; i++) {
            const double ii = (double)(i + 1);
            t[i] = ((double)(2 * m) - 2 * ii + 1) * (u[i] * u[i]);
        }
        const double an = recip(n4) * psum(t.data(), m);
        for (int64_t i = 0; i < m; i++) {
            const double ii = (double)(i + 1);
            const double w = v[i] + (dn - ii) * u[i];
            t[i] = w * w;
        }
        const double bn = recip(n5) * psum(t.data(), m);
        for (int64_t i = 0; i < m; i++) {
            const double ii = (double)(i + 1);
            t[i] = ((double)(2 * m) - 2 * ii + 1) * u[i];
        }
        const double cn = recip(n3) * psum(t.data(), m);
        for (int64_t i = 0; i < m; i++) t[i] = l[i] * (dn - l[i]);
        const double dnn = recip(n3) * psum(t.data(), m);
        /* cn**2, dn**2 on NumPy scalars (1-D input): pow(); a multidimensional call without NaNs squares arrays
           (np.square), which can differ by an ulp */
        const double tau2 = (an - 2 * bn + py_pow(cn, 2.0)) / py_pow(dnn, 2.0);
        sd = std::sqrt(tau2) / std::sqrt(dn);
    }
    out[0][0] = xi;
    out[1][0] = sc::ndtr(-(xi / sd));
    return TSR_OK;
}

/* ================================================================ kendalltau */

/* scipy.stats._mstats_basic._kendall_p_exact */
int kendall_p_exact(int64_t n, int64_t c, int alt, double &out)
{
    const int64_t half = (n * (n - 1)) / 2;
    const bool in_right_tail = c >= half - c;
    const bool alt_greater = alt == 2;
    c = std::min(c, half - c);
    double prob, pmass = 0;
    if (n <= 0) return fail("n (%lld) must be positive", (long long)n);
    if (c < 0 || 4 * c > n * (n - 1)) return fail("c (%lld) must satisfy 0 <= 4c <= n(n-1) = %lld.", (long long)c, (long long)(n * (n - 1)));
    if (n == 1) { prob = 1.0; pmass = 1; }
    else if (n == 2) { prob = 1.0; pmass = 0.5; }
    else if (c == 0) {
        prob = n < 171 ? 2.0 / dd_fact(n).hi : 0.0;
        pmass = prob / 2;
    } else if (c == 1) {
        prob = n < 172 ? 2.0 / dd_fact(n - 1).hi : 0.0;
        pmass = dd_div(dd_invfact(n - 2), (double)n).hi;     /* (n-1)/n! = 1/(n (n-2)!), Python's exact int division */
    } else if (4 * c == n * (n - 1) && alt == 0) {
        prob = 1.0;
    } else if (n < 171) {
        V nw(c + 1, 0.0), tmp;
        nw[0] = 1.0;
        nw[1] = 1.0;
        for (int64_t j = 3; j <= n; j++) {
            for (int64_t k = 1; k <= c; k++) nw[k] = nw[k - 1] + nw[k];
            if (j <= c) {
                tmp.assign(nw.begin(), nw.begin() + (c + 1 - j));
                for (int64_t k = j; k <= c; k++) nw[k] -= tmp[k - j];
            }
        }
        const double f = dd_fact(n).hi;
        prob = 2.0 * psum(nw.data(), c + 1) / f;
        pmass = nw[c] / f;
    } else {
        V nw(c + 1, 0.0), tmp;
        nw[0] = 1.0;
        nw[1] = 1.0;
        for (int64_t j = 3; j <= n; j++) {
            for (int64_t k = 1; k <= c; k++) nw[k] = nw[k - 1] + nw[k];
            for (int64_t k = 0; k <= c; k++) nw[k] = nw[k] / (double)j;
            if (j <= c) {
                tmp.assign(nw.begin(), nw.begin() + (c + 1 - j));
                for (int64_t k = j; k <= c; k++) nw[k] -= tmp[k - j];
            }
        }
        prob = psum(nw.data(), c + 1);
        pmass = nw[c] / 2;
    }
    if (alt != 0) {
        if (in_right_tail == alt_greater) prob /= 2;
        else prob = 1 - prob / 2 + pmass;
    }
    out = np_clip(prob, 0.0, 1.0);
    return TSR_OK;
}

/* dense ranks 1..k (NaN last, each NaN its own rank) */
void dense_ranks(const double *x, int64_t n, VI &r, int64_t &nuniq)
{
    VI j;
    argsort_stable(x, n, j);
    r.resize(n);
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) {
        if (i == 0 || x[j[i]] != x[j[i - 1]]) k++;
        r[j[i]] = k;
    }
    nuniq = k;
}

struct TieStats {
    int64_t tie, t0, t1;
};
TieStats count_rank_tie(const VI &ranks, int64_t kmax)
{
    VI cnt(kmax + 1, 0);
    for (int64_t v : ranks) cnt[v]++;
    TieStats s{0, 0, 0};
    for (int64_t c : cnt)
        if (c > 1) {
            s.tie += c * (c - 1) / 2;
            s.t0 += c * (c - 1) * (c - 2);
            s.t1 += c * (c - 1) * (2 * c + 5);
        }
    return s;
}

int c_kendalltau(const void *, const double *const *x, const int64_t *nn, const double *p, double *const *out,
                 const int64_t *, void *, unsigned)
{
    if (nn[0] != nn[1]) return fail("Array shapes are incompatible for broadcasting.");
    const int64_t n = nn[0];
    const int method = (int)p[0], variant = (int)p[1], alt = (int)p[2];
    if (n <= 1) { out[0][0] = out[1][0] = NaN; return TSR_OK; }
    VI rx, ry;
    int64_t kx, ky;
    dense_ranks(x[0], n, rx, kx);
    dense_ranks(x[1], n, ry, ky);
    /* order by (x, y) */
    VI perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
        return rx[a] < rx[b] || (rx[a] == rx[b] && ry[a] < ry[b]);
    });
    VI xs(n), ys(n);
    for (int64_t i = 0; i < n; i++) { xs[i] = rx[perm[i]]; ys[i] = ry[perm[i]]; }
    /* _kendall_dis (Fenwick tree) */
    int64_t sup = 1 + *std::max_element(ys.begin(), ys.end());
    VI arr(sup + ((sup - 1) >> 14), 0);
    int64_t dis = 0;
    {
        int64_t i = 0, k = 0;
        while (i < n) {
            while (k < n && xs[i] == xs[k]) {
                dis += i;
                int64_t idx = ys[k];
                while (idx != 0) { dis -= arr[idx + (idx >> 14)]; idx = idx & (idx - 1); }
                k++;
            }
            while (i < k) {
                int64_t idx = ys[i];
                while (idx < sup) { arr[idx + (idx >> 14)] += 1; idx += idx & -idx; }
                i++;
            }
        }
    }
    int64_t ntie = 0;
    {
        int64_t start = 0;
        for (int64_t i = 1; i <= n; i++)
            if (i == n || xs[i] != xs[i - 1] || ys[i] != ys[i - 1]) {
                const int64_t c = i - start;
                ntie += c * (c - 1) / 2;
                start = i;
            }
    }
    const TieStats tx = count_rank_tie(xs, kx), ty = count_rank_tie(ys, ky);
    const int64_t tot = (n * (n - 1)) / 2;
    if (tx.tie == tot || ty.tie == tot) { out[0][0] = out[1][0] = NaN; return TSR_OK; }
    const int64_t cmd = tot - tx.tie - ty.tie + ntie - 2 * dis;
    double tau;
    if (variant == 0) {
        tau = (double)cmd / std::sqrt((double)(tot - tx.tie)) / std::sqrt((double)(tot - ty.tie));
    } else {
        const int64_t mc = std::min(kx, ky);
        tau = (double)(2 * cmd) / ((double)(n * n * (mc - 1)) / (double)mc);
    }
    tau = std::min(1.0, py_max(-1.0, tau));
    if (method == 2 && (tx.tie != 0 || ty.tie != 0)) return fail("Ties found, exact method cannot be used.");
    int meth = method;
    if (meth == 0) meth = (tx.tie == 0 && ty.tie == 0 && (n <= 33 || std::min(dis, tot - dis) <= 1)) ? 2 : 1;
    double pv;
    if (tx.tie == 0 && ty.tie == 0 && meth == 2) {
        const int rc = kendall_p_exact(n, tot - dis, alt, pv);
        if (rc != TSR_OK) return rc;
    } else {
        const double m = (double)n * ((double)n - 1.0);
        if (n == 2) return fail("float division by zero");
        const double var = ((m * (double)(2 * n + 5) - (double)tx.t1 - (double)ty.t1) / 18 +
                            tsr128::to_double(tsr128::i128(2) * tsr128::i128((int64_t)tx.tie) * tsr128::i128((int64_t)ty.tie)) / m +
                            tsr128::to_double(tsr128::i128((int64_t)tx.t0) * tsr128::i128((int64_t)ty.t0)) / (9 * m * (double)(n - 2)));
        const double z = (double)cmd / std::sqrt(var);
        pv = norm_p(z, alt);
    }
    out[0][0] = tau;
    out[1][0] = pv;
    return TSR_OK;
}


/* ================================================================ routine arguments */

struct Arr {
    V v;
    VI shape;
    int dtype = TSR_F64;
    bool present = false;
    int64_t size() const { return (int64_t)v.size(); }
    int ndim() const { return (int)shape.size(); }
    bool has_nan() const
    {
        for (double t : v)
            if (t != t) return true;
        return false;
    }
};

inline bool is_int_dtype(int dt) { return dt == TSR_I64 || dt == TSR_I32 || dt == TSR_U8; }

const tsr_arg *arg_at(const tsr_arg *args, int nargs, int k) { return k < nargs ? &args[k] : nullptr; }

bool is_none(const tsr_arg *a) { return !a || a->kind == 0; }

/* an array argument (numbers and booleans are 0-d arrays); absent/None leaves out.present false */
int get_arr(const tsr_arg *a, Arr &out, const char *name)
{
    out = Arr();
    if (is_none(a)) return TSR_OK;
    out.present = true;
    if (a->kind == 1 || a->kind == 4) {
        out.v.assign(1, a->num);
        out.dtype = a->kind == 4 ? TSR_BOOL : TSR_F64;
        return TSR_OK;
    }
    if (a->kind != 3) return fail("`%s` must be array-like", name);
    if (a->arr.dtype == TSR_C128) return fail("`%s` must be real", name);
    int64_t n = 0;
    double *d = fn_arg_doubles(a, &n);
    if (!d) return TSR_ENOMEM;
    out.v.assign(d, d + n);
    fn_free_doubles(d, n);
    out.shape.assign(a->arr.shape, a->arr.shape + a->arr.ndim);
    out.dtype = a->arr.dtype;
    return TSR_OK;
}

int get_str(const tsr_arg *a, const char *dflt, std::string &out, const char *name)
{
    if (is_none(a)) { out = dflt ? dflt : ""; return dflt ? TSR_OK : fail("`%s` must be a string", name); }
    if (a->kind != 2 || !a->str) return fail("`%s` must be a string", name);
    out = a->str;
    return TSR_OK;
}

int get_bool(const tsr_arg *a, bool dflt, bool &out, const char *name)
{
    if (is_none(a)) { out = dflt; return TSR_OK; }
    if (a->kind == 4 || a->kind == 1) { out = a->num != 0; return TSR_OK; }
    return fail("`%s` must be a boolean", name);
}

int parse_alt(const tsr_arg *a, int &alt)
{
    std::string s;
    int rc = get_str(a, "two-sided", s, "alternative");
    if (rc) return rc;
    if (s == "two-sided") alt = 0;
    else if (s == "less") alt = 1;
    else if (s == "greater") alt = 2;
    else return fail("`alternative` must be 'less', 'greater', or 'two-sided'.");
    return TSR_OK;
}

int parse_nan_policy(const tsr_arg *a, int &np_)
{
    std::string s;
    int rc = get_str(a, "propagate", s, "nan_policy");
    if (rc) return rc;
    if (s == "propagate") np_ = 0;
    else if (s == "omit") np_ = 1;
    else if (s == "raise") np_ = 2;
    else return fail("nan_policy must be one of {'propagate', 'omit', 'raise'}");
    return TSR_OK;
}

/* axis: None (kind 0) or an integer */
int parse_axis(const tsr_arg *a, bool dflt_none, int64_t dflt, bool &none, int64_t &axis)
{
    if (!a) { none = dflt_none; axis = dflt; return TSR_OK; }
    if (a->kind == 0) { none = true; axis = 0; return TSR_OK; }
    if (a->kind == 1 && a->num == std::floor(a->num) && std::isfinite(a->num)) { none = false; axis = (int64_t)std::fmax(-1e9, std::fmin(1e9, a->num)); return TSR_OK; }
    return fail("`axis` must be an integer or None");
}

int64_t prod(const VI &s)
{
    int64_t p = 1;
    for (int64_t d : s) p *= d;
    return p;
}

/* broadcast shapes (right-aligned) */
int broadcast(const std::vector<VI> &shapes, VI &out)
{
    size_t nd = 0;
    for (auto &s : shapes) nd = std::max(nd, s.size());
    out.assign(nd, 1);
    for (size_t k = 0; k < nd; k++) {
        int64_t len = 1;
        bool zero = false;
        for (auto &s : shapes) {
            const int64_t off = (int64_t)nd - (int64_t)s.size();
            if ((int64_t)k < off) continue;
            const int64_t l = s[k - off];
            if (l == 1) continue;
            if (l == 0) zero = true;
            if (len != 1 && len != l) return fail("Array shapes are incompatible for broadcasting.");
            len = l;
        }
        out[k] = zero ? 0 : len;
    }
    return TSR_OK;
}

/* element strides (in elements) of a C-contiguous array of shape s broadcast to the full shape B */
VI bstrides(const VI &s, const VI &B)
{
    VI st(B.size(), 0);
    int64_t acc = 1;
    const int64_t off = (int64_t)B.size() - (int64_t)s.size();
    for (int64_t k = (int64_t)s.size() - 1; k >= 0; k--) {
        st[k + off] = s[k] == 1 ? 0 : acc;
        acc *= s[k];
    }
    return st;
}

int result_arr(tsr_result *r, const VI &shape, const double *data, int dtype = TSR_F64)
{
    VI sh = shape;
    void *p = fn_result_array(r, dtype, (int32_t)sh.size(), sh.empty() ? nullptr : sh.data());
    if (!p) return TSR_ENOMEM;
    const int64_t n = prod(sh);
    if (n == 0) return TSR_OK;                   /* an empty result: data may be null (memcpy's argument must not) */
    if (dtype == TSR_I64)
        for (int64_t i = 0; i < n; i++)
            ((int64_t *)p)[i] = (data[i] > -9223372036854775808.0 && data[i] < 9223372036854775808.0) ? (int64_t)data[i] : INT64_MIN;
    else if (dtype == TSR_BOOL) for (int64_t i = 0; i < n; i++) ((uint8_t *)p)[i] = data[i] != 0;
    else std::memcpy(p, data, (size_t)n * sizeof(double));
    return TSR_OK;
}

/* a result of shape `shape`: a Python float when 0-d, an array otherwise */
int result_out(tsr_result *r, const VI &shape, const double *data)
{
    if (shape.empty()) { fn_result_num(r, data[0]); return TSR_OK; }
    return result_arr(r, shape, data);
}

/* ================================================================ the _axis_nan_policy decorator for routines */

using SliceFn = std::function<int(std::vector<V> &, double *)>;

/* samples: paired samples (broadcast together); fn gets the slices along the axis (contiguous, NaN
   handling done) and writes nout results */
int drive(std::vector<Arr> &samples, bool axis_none, int64_t axis, int nanpol, bool keepdims, int too_small,
          bool nan_prop, int nout, const SliceFn &fn, tsr_result *res)
{
    const int ns = (int)samples.size();
    int maxnd = 1;
    for (auto &s : samples) {
        if (s.shape.empty()) s.shape = {1};
        maxnd = std::max(maxnd, s.ndim());
    }
    VI B, loop, outshape;
    int64_t L = 0, ax = 0;
    std::vector<VI> st(ns);
    if (axis_none) {
        for (auto &s : samples) s.shape = {s.size()};
        for (int i = 1; i < ns; i++)
            if (samples[i].size() != samples[0].size()) return fail("Array shapes are incompatible for broadcasting.");
        L = samples[0].size();
        B = {L};
        ax = 0;
        for (int i = 0; i < ns; i++) st[i] = {1};
        if (keepdims) outshape.assign(maxnd, 1);
    } else {
        std::vector<VI> shapes;
        for (auto &s : samples) shapes.push_back(s.shape);
        int rc = broadcast(shapes, B);
        if (rc) return rc;
        const int64_t nd = (int64_t)B.size();
        ax = axis < 0 ? axis + nd : axis;
        if (ax < 0 || ax >= nd) return fail("`axis` is out of bounds for array of dimension %lld", (long long)nd);
        L = B[ax];
        for (int64_t k = 0; k < nd; k++) if (k != ax) loop.push_back(B[k]);
        for (int i = 0; i < ns; i++) st[i] = bstrides(samples[i].shape, B);
        if (keepdims) { outshape = B; outshape[ax] = 1; }
        else outshape = loop;
    }
    if (nanpol == 2)
        for (auto &s : samples)
            if (s.has_nan()) return fail("The input contains nan values");
    const int64_t nloop = prod(loop);
    std::vector<V> outs(nout, V(std::max<int64_t>(nloop, 1), NaN));
    std::vector<V> sl(ns);
    VI idx(loop.size(), 0);
    std::vector<double> o(nout);
    for (int64_t it = 0; it < nloop; it++) {
        /* base offsets */
        for (int i = 0; i < ns; i++) {
            int64_t base = 0, c = 0;
            for (int64_t k = 0; k < (int64_t)B.size(); k++) {
                if (k == ax) continue;
                base += idx[c++] * st[i][k];
            }
            sl[i].resize(L);
            const int64_t step = st[i][ax];
            for (int64_t t = 0; t < L; t++) sl[i][t] = samples[i].v[base + t * step];
        }
        bool nan_any = false;
        for (int i = 0; i < ns && !nan_any; i++)
            for (double t : sl[i]) if (t != t) { nan_any = true; break; }
        std::fill(o.begin(), o.end(), NaN);
        bool done = false;
        if (nan_any && nanpol == 0 && nan_prop) done = true;
        if (!done && nan_any && nanpol == 1) {
            int64_t k = 0;
            for (int64_t t = 0; t < L; t++) {
                bool bad = false;
                for (int i = 0; i < ns; i++) if (sl[i][t] != sl[i][t]) bad = true;
                if (!bad) { for (int i = 0; i < ns; i++) sl[i][k] = sl[i][t]; k++; }
            }
            for (int i = 0; i < ns; i++) sl[i].resize(k);
        }
        if (!done && (int64_t)sl[0].size() <= too_small) done = true;
        if (!done) {
            const int rc = fn(sl, o.data());
            if (rc) return rc;
        }
        for (int j = 0; j < nout; j++) outs[j][it] = o[j];
        for (int64_t c = (int64_t)loop.size() - 1; c >= 0; c--) { if (++idx[c] < loop[c]) break; idx[c] = 0; }
    }
    for (int j = 0; j < nout; j++) {
        const int rc = result_out(&res[j], outshape, outs[j].data());
        if (rc) return rc;
    }
    return TSR_OK;
}

/* ================================================================ linregress */

int r_linregress(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    std::vector<Arr> s(2);
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), s[0], "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), s[1], "y"))) return rc;
    if (!s[0].present || !s[1].present) return fail("linregress() requires `x` and `y`");
    int alt, nanpol;
    bool none, keep;
    int64_t axis;
    if ((rc = parse_alt(arg_at(args, nargs, 2), alt))) return rc;
    if ((rc = parse_axis(arg_at(args, nargs, 3), false, 0, none, axis))) return rc;
    if ((rc = parse_nan_policy(arg_at(args, nargs, 4), nanpol))) return rc;
    if ((rc = get_bool(arg_at(args, nargs, 5), false, keep, "keepdims"))) return rc;
    auto fn = [alt](std::vector<V> &sl, double *o) -> int {
        const V &x = sl[0], &y = sl[1];
        const int64_t n = (int64_t)x.size();
        if (n == 0) return fail("Inputs must not be empty.");
        const double mx = *std::max_element(x.begin(), x.end()), mn = *std::min_element(x.begin(), x.end());
        if (mx == mn && n > 1) return fail("Cannot calculate a linear regression if all x values are identical");
        const double xmean = psum(x.data(), n) / (double)n, ymean = psum(y.data(), n) / (double)n;
        V xc(n), yc(n);
        for (int64_t i = 0; i < n; i++) { xc[i] = x[i] - xmean; yc[i] = y[i] - ymean; }
        const double f = 1.0 / (double)n;
        const double ssxm = gemm_dot(n, xc.data(), xc.data()) * f;
        const double ssxym = gemm_dot(n, xc.data(), yc.data()) * f;
        const double ssym = gemm_dot(n, yc.data(), yc.data()) * f;
        double r;
        if (ssxm == 0.0 || ssym == 0.0) {
            r = ssxym == 0 ? NaN : 0.0;
        } else {
            r = ssxym / std::sqrt(ssxm * ssym);
            if (r > 1.0) r = 1.0;
            else if (r < -1.0) r = -1.0;
        }
        const double slope = ssxym / ssxm;
        const double intercept = ymean - slope * xmean;
        double prob, se, ise;
        if (n == 2) {
            prob = y[0] == y[1] ? 1.0 : 0.0;
            se = 0.0;
            ise = 0.0;
        } else {
            const double TINY = 1.0e-20;
            const double df = (double)(n - 2);
            const double t = r * std::sqrt(df / ((1.0 - r + TINY) * (1.0 + r + TINY)));
            prob = t_p(df, t, alt);
            se = std::sqrt((1 - py_pow(r, 2.0)) * ssym / ssxm / df);
            ise = se * std::sqrt(ssxm + py_pow(xmean, 2.0));
        }
        o[0] = slope; o[1] = intercept; o[2] = r; o[3] = prob; o[4] = se; o[5] = ise;
        return TSR_OK;
    };
    return drive(s, none, axis, nanpol, keep, 1, true, 6, fn, res);
}

/* ================================================================ theilslopes / siegelslopes */

/* Python's int(np.round(v)): error for inf, *nan for NaN */
int py_int_round(double v, int64_t &out, bool &nan)
{
    const double r = std::nearbyint(v);
    if (r != r) { nan = true; return TSR_OK; }
    if (std::isinf(r)) return fail("cannot convert float infinity to integer");
    out = (int64_t)r;
    return TSR_OK;
}

int r_theilslopes(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    std::vector<Arr> s(1);
    Arr xa;
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), s[0], "y"))) return rc;
    if (!s[0].present) return fail("theilslopes() requires `y`");
    if ((rc = get_arr(arg_at(args, nargs, 1), xa, "x"))) return rc;
    const bool hasx = xa.present;
    if (hasx) s.push_back(xa);
    const tsr_arg *al = arg_at(args, nargs, 2);
    double alpha = 0.95;
    if (al && al->kind != 0) {
        if (al->kind != 1) return fail("`alpha` must be a number");
        alpha = al->num;
    }
    std::string method;
    if ((rc = get_str(arg_at(args, nargs, 3), "separate", method, "method"))) return rc;
    bool none, keep;
    int64_t axis;
    int nanpol;
    if ((rc = parse_axis(arg_at(args, nargs, 4), true, 0, none, axis))) return rc;
    if ((rc = parse_nan_policy(arg_at(args, nargs, 5), nanpol))) return rc;
    if ((rc = get_bool(arg_at(args, nargs, 6), false, keep, "keepdims"))) return rc;
    if (method != "joint" && method != "separate")
        return fail("method must be either 'joint' or 'separate'.'%s' is invalid.", method.c_str());
    const bool joint = method == "joint";
    auto fn = [hasx, alpha, joint](std::vector<V> &sl, double *o) -> int {
        const V &y = sl[0];
        const int64_t n = (int64_t)y.size();
        V x(n);
        if (hasx) x = sl[1];
        else for (int64_t i = 0; i < n; i++) x[i] = (double)i;
        if (n < 2) return fail("`x` and `y` must have length at least 2.");
        V slopes;
        for (int64_t j = 0; j < n; j++)
            for (int64_t k = 0; k < n; k++) {
                const double dx = x[j] - x[k];
                if (dx > 0) slopes.push_back((y[j] - y[k]) / dx);
            }
        std::sort(slopes.begin(), slopes.end(), lt_nan);
        const double medslope = np_median(slopes);
        double medinter;
        if (joint) {
            V t(n);
            for (int64_t i = 0; i < n; i++) t[i] = y[i] - medslope * x[i];
            medinter = np_median(t);
        } else {
            medinter = np_median(y) - medslope * np_median(x);
        }
        double a = alpha;
        if (a > 0.5) a = 1. - a;
        const double z = tsd::call(tsd::find("norm"), DM_PPF, a / 2., {0.0, 1.0});
        auto reps = [](const V &v) {
            V w(v);
            std::sort(w.begin(), w.end(), lt_nan);
            int64_t acc = 0, start = 0;
            const int64_t m = (int64_t)w.size();
            for (int64_t i = 1; i <= m; i++)
                if (i == m || w[i] != w[i - 1]) {
                    const int64_t k = i - start;
                    if (k > 1) acc += k * (k - 1) * (2 * k + 5);
                    start = i;
                }
            return acc;
        };
        const int64_t nt = (int64_t)slopes.size();
        const double sigsq = 1 / 18. * (double)(n * (n - 1) * (2 * n + 5) - reps(x) - reps(y));
        const double sigma = std::sqrt(sigsq);
        double lo = NaN, hi = NaN;
        int64_t ru = 0, rl = 0;
        bool nan = false;
        int rc2 = py_int_round(((double)nt - z * sigma) / 2., ru, nan);
        if (rc2) return rc2;
        if (!nan) {
            ru = std::min(ru, nt - 1);
            rc2 = py_int_round(((double)nt + z * sigma) / 2., rl, nan);
            if (rc2) return rc2;
        }
        if (!nan) {
            rl = std::max(rl - 1, (int64_t)0);
            auto ok = [nt](int64_t i) { return i >= -nt && i < nt; };
            if (ok(rl) && ok(ru)) {
                lo = slopes[rl < 0 ? rl + nt : rl];
                hi = slopes[ru < 0 ? ru + nt : ru];
            }
        }
        o[0] = medslope; o[1] = medinter; o[2] = lo; o[3] = hi;
        return TSR_OK;
    };
    return drive(s, none, axis, nanpol, keep, 1, true, 4, fn, res);
}

/* the median of scipy.stats._stats_pythran (pythran's numpy.median); an empty input is undefined there */
double pythran_median(V v) { return np_median(std::move(v)); }

int r_siegelslopes(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    std::vector<Arr> s(1);
    Arr xa;
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), s[0], "y"))) return rc;
    if (!s[0].present) return fail("siegelslopes() requires `y`");
    if ((rc = get_arr(arg_at(args, nargs, 1), xa, "x"))) return rc;
    const bool hasx = xa.present;
    if (hasx) s.push_back(xa);
    std::string method;
    if ((rc = get_str(arg_at(args, nargs, 2), "hierarchical", method, "method"))) return rc;
    bool none, keep;
    int64_t axis;
    int nanpol;
    if ((rc = parse_axis(arg_at(args, nargs, 3), true, 0, none, axis))) return rc;
    if ((rc = parse_nan_policy(arg_at(args, nargs, 4), nanpol))) return rc;
    if ((rc = get_bool(arg_at(args, nargs, 5), false, keep, "keepdims"))) return rc;
    if (method != "hierarchical" && method != "separate") return fail("method can only be 'hierarchical' or 'separate'");
    const bool sep = method == "separate";
    auto fn = [hasx, sep](std::vector<V> &sl, double *o) -> int {
        const V &y = sl[0];
        const int64_t n = (int64_t)y.size();
        V x(n);
        if (hasx) x = sl[1];
        else for (int64_t i = 0; i < n; i++) x[i] = (double)i;
        if (n < 2) return fail("`x` and `y` must have length at least 2.");
        V slopes, inters, sj, ij;
        for (int64_t j = 0; j < n; j++) {
            sj.clear();
            ij.clear();
            for (int64_t k = 0; k < n; k++) {
                const double dx = x[j] - x[k];
                if (dx != 0) {
                    sj.push_back((y[j] - y[k]) / dx);
                    if (sep) ij.push_back((y[k] * x[j] - y[j] * x[k]) / dx);
                }
            }
            slopes.push_back(pythran_median(sj));
            if (sep) inters.push_back(pythran_median(ij));
        }
        const double medslope = pythran_median(slopes);
        double medinter;
        if (sep) medinter = pythran_median(inters);
        else {
            V t(n);
            for (int64_t i = 0; i < n; i++) t[i] = y[i] - medslope * x[i];
            medinter = pythran_median(t);
        }
        o[0] = medslope;
        o[1] = medinter;
        return TSR_OK;
    };
    return drive(s, none, axis, nanpol, keep, 1, true, 2, fn, res);
}

/* ================================================================ weightedtau */

/* _toint64: dense ranks, NaN lowest (0) */
void toint64(const V &x, VI &r)
{
    const int64_t l = (int64_t)x.size();
    VI perm;
    argsort_stable(x.data(), l, perm);
    r.assign(l, 0);
    int64_t i = l - 1, j = 0, ll = l;
    for (; i >= 0; i--) {
        if (!(x[perm[i]] != x[perm[i]])) break;
        r[perm[i]] = 0;
    }
    if (i < l - 1) { j = 1; ll = i + 1; }
    if (ll <= 0) return;
    for (int64_t k = 0; k < ll - 1; k++) {
        r[perm[k]] = j;
        if (x[perm[k]] != x[perm[k + 1]]) j++;
    }
    r[perm[ll - 1]] = j;
}

struct WTau {
    const VI *y;
    const VI *rank;
    VI perm, temp;
    bool additive;
    double exch = 0;
    double w(int64_t r) const { return 1. / (double)(1 + r); }
    double weigh(int64_t offset, int64_t length)
    {
        if (length == 1) return w((*rank)[perm[offset]]);
        const int64_t length0 = length / 2, length1 = length - length0, middle = offset + length0;
        double residual = weigh(offset, length0);
        const double weight = weigh(middle, length1) + residual;
        if ((*y)[perm[middle - 1]] < (*y)[perm[middle]]) return weight;
        int64_t i = 0, j = 0, k = 0;
        while (j < length0 && k < length1) {
            if ((*y)[perm[offset + j]] <= (*y)[perm[middle + k]]) {
                temp[i] = perm[offset + j];
                residual -= w((*rank)[temp[i]]);
                j++;
            } else {
                temp[i] = perm[middle + k];
                exch += additive ? w((*rank)[temp[i]]) * (double)(length0 - j) + residual
                                 : w((*rank)[temp[i]]) * residual;
                k++;
            }
            i++;
        }
        std::memmove(&perm[offset + i], &perm[offset + j], (size_t)(length0 - j) * sizeof(int64_t));
        std::memcpy(&perm[offset], &temp[0], (size_t)i * sizeof(int64_t));
        return weight;
    }
};

/* _weightedrankedtau(x, y, rank, weigher=None, additive); rank empty: computed from the order */
double weighted_ranked_tau(const VI &x, const VI &y, const VI *rank_in, bool additive)
{
    const int64_t n = (int64_t)x.size();
    WTau W;
    W.y = &y;
    W.additive = additive;
    W.perm.resize(n);
    std::iota(W.perm.begin(), W.perm.end(), 0);
    std::stable_sort(W.perm.begin(), W.perm.end(), [&](int64_t a, int64_t b) {
        return x[a] < x[b] || (x[a] == x[b] && y[a] < y[b]);
    });
    W.temp.assign(n, 0);
    VI rank;
    if (rank_in) rank = *rank_in;
    else {
        rank.assign(n, 0);
        for (int64_t i = 0; i < n; i++) rank[W.perm[n - 1 - i]] = i;
    }
    W.rank = &rank;
    const VI &perm = W.perm;
    auto seg = [&](auto same) {
        int64_t first = 0;
        double acc = 0, w0 = W.w(rank[perm[0]]), s = w0, sq = w0 * w0;
        for (int64_t i = 1; i < n; i++) {
            if (!same(perm[first], perm[i])) {
                acc += additive ? s * (double)(i - first - 1) : (s * s - sq) / 2;
                first = i;
                s = sq = 0;
            }
            const double w = W.w(rank[perm[i]]);
            s += w;
            sq += w * w;
        }
        acc += additive ? s * (double)(n - first - 1) : (s * s - sq) / 2;
        return std::make_pair(acc, first);
    };
    const double t = seg([&](int64_t a, int64_t b) { return x[a] == x[b] && y[a] == y[b]; }).first;
    auto uu = seg([&](int64_t a, int64_t b) { return x[a] == x[b]; });
    const double u = uu.first;
    if (uu.second == 0) return NaN;
    W.weigh(0, n);
    auto vv = seg([&](int64_t a, int64_t b) { return y[a] == y[b]; });
    const double v = vv.first;
    if (vv.second == 0) return NaN;
    double s = 0, sq = 0;
    for (int64_t i = 0; i < n; i++) {
        const double w = W.w(rank[perm[i]]);
        s += w;
        sq += w * w;
    }
    const double tot = additive ? s * (double)(n - 1) : (s * s - sq) / 2;
    const double tau = ((tot - (v + u - t)) - 2. * W.exch) / std::sqrt(tot - u) / std::sqrt(tot - v);
    return py_min(1., py_max(-1., tau));
}

int r_weightedtau(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    std::vector<Arr> s(2);
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), s[0], "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), s[1], "y"))) return rc;
    if (!s[0].present || !s[1].present) return fail("weightedtau() requires `x` and `y`");
    const tsr_arg *ra = arg_at(args, nargs, 2);
    int rank_mode;                               /* 0 True, 1 False, 2 None, 3 array */
    if (!ra || (ra->kind == 4 && ra->num != 0)) rank_mode = 0;
    else if (ra->kind == 4) rank_mode = 1;
    else if (ra->kind == 0) rank_mode = 2;
    else {
        rank_mode = 3;
        Arr r;
        if ((rc = get_arr(ra, r, "rank"))) return rc;
        s.push_back(r);
    }
    if (!is_none(arg_at(args, nargs, 3))) return fail("weightedtau(): a callable `weigher` is not supported (only None)");
    bool additive, keep;
    if ((rc = get_bool(arg_at(args, nargs, 4), true, additive, "additive"))) return rc;
    bool none;
    int64_t axis;
    int nanpol;
    if ((rc = parse_axis(arg_at(args, nargs, 5), true, 0, none, axis))) return rc;
    if ((rc = parse_nan_policy(arg_at(args, nargs, 6), nanpol))) return rc;
    if ((rc = get_bool(arg_at(args, nargs, 7), false, keep, "keepdims"))) return rc;
    auto fn = [rank_mode, additive](std::vector<V> &sl, double *o) -> int {
        const int64_t n = (int64_t)sl[0].size();
        if ((int64_t)sl[1].size() != n) return fail("Array shapes are incompatible for broadcasting.");
        VI x, y;
        toint64(sl[0], x);
        toint64(sl[1], y);
        double tau;
        if (rank_mode == 0) {
            tau = (weighted_ranked_tau(x, y, nullptr, additive) + weighted_ranked_tau(y, x, nullptr, additive)) / 2;
        } else {
            VI rank(n);
            if (rank_mode == 1) std::iota(rank.begin(), rank.end(), 0);
            else if (rank_mode == 3) {
                if ((int64_t)sl[2].size() != n)
                    return fail("All inputs to `weightedtau` must be of the same size, found x-size %lld and rank-size %lld",
                                (long long)n, (long long)sl[2].size());
                toint64(sl[2], rank);
            }
            tau = weighted_ranked_tau(x, y, rank_mode == 2 ? nullptr : &rank, additive);
        }
        o[0] = tau;
        o[1] = NaN;
        return TSR_OK;
    };
    return drive(s, none, axis, nanpol, keep, 1, false, 2, fn, res);
}

/* ================================================================ somersd */

/* np.unique levels (NaNs merged, last) and the inverse indices */
void unique_inverse(const V &x, V &levels, VI &inv)
{
    const int64_t n = (int64_t)x.size();
    VI j;
    argsort_stable(x.data(), n, j);
    levels.clear();
    inv.assign(n, 0);
    for (int64_t i = 0; i < n; i++) {
        const double v = x[j[i]];
        const bool same = i > 0 && (v == levels.back() || (v != v && levels.back() != levels.back()));
        if (!same) levels.push_back(v);
        inv[j[i]] = (int64_t)levels.size() - 1;
    }
}

int r_somersd(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr x, y;
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), x, "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), y, "y"))) return rc;
    int alt;
    if ((rc = parse_alt(arg_at(args, nargs, 2), alt))) return rc;
    if (!x.present) return fail("x must be either a 1D or 2D array");
    int64_t m, k;
    V A;
    int tdtype = TSR_I64;
    if (x.ndim() == 1) {
        if (!y.present) {
            if (x.size() != 1) return fail("Rankings must be of equal length.");
            return fail("crosstab() needs array-like inputs");
        }
        if (x.size() != y.size()) return fail("Rankings must be of equal length.");
        if (y.ndim() != 1) return fail("`y` must be one-dimensional");
        V lx, ly;
        VI ix, iy;
        unique_inverse(x.v, lx, ix);
        unique_inverse(y.v, ly, iy);
        m = (int64_t)lx.size();
        k = (int64_t)ly.size();
        A.assign(m * k, 0.0);
        for (int64_t i = 0; i < x.size(); i++) A[ix[i] * k + iy[i]] += 1;
    } else if (x.ndim() == 2) {
        for (double v : x.v) if (v < 0) return fail("All elements of the contingency table must be non-negative.");
        for (double v : x.v) if (v != std::trunc(v)) return fail("All elements of the contingency table must be integer.");
        int64_t nz = 0;
        for (double v : x.v) if (v != 0) nz++;
        if (nz < 2) return fail("At least two elements of the contingency table must be nonzero.");
        m = x.shape[0];
        k = x.shape[1];
        A = x.v;
        tdtype = is_int_dtype(x.dtype) ? TSR_I64 : (x.dtype == TSR_BOOL ? TSR_BOOL : TSR_F64);
    } else {
        return fail("x must be either a 1D or 2D array");
    }
    double d = NaN, p = NaN;
    if (m > 1 && k > 1) {
        auto at = [&](int64_t i, int64_t j) { return A[i * k + j]; };
        auto block = [&](int64_t i0, int64_t i1, int64_t j0, int64_t j1) {
            double s = 0;
            for (int64_t i = i0; i < i1; i++)
                for (int64_t j = j0; j < j1; j++) s += at(i, j);
            return s;
        };
        double PA = 0, QA = 0, AD = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = 0; j < k; j++) {
                const double aij = block(0, i, 0, j) + block(i + 1, m, j + 1, k);
                const double dij = block(i + 1, m, 0, j) + block(0, i, j + 1, k);
                PA += at(i, j) * aij;
                QA += at(i, j) * dij;
                AD += at(i, j) * py_pow(aij - dij, 2.0);
            }
        const double NA = psum(A.data(), m * k);
        const double NA2 = py_pow(NA, 2.0);
        V rows(m);
        for (int64_t i = 0; i < m; i++) { const double r = psum(&A[i * k], k); rows[i] = r * r; }
        const double Sri2 = psum(rows.data(), m);
        d = (PA - QA) / (NA2 - Sri2);
        const double S = AD - py_pow(PA - QA, 2.0) / NA;
        const double Z = (PA - QA) / py_pow(4 * S, 0.5);
        p = norm_p(Z, alt);
    }
    fn_result_num(&res[0], d);
    fn_result_num(&res[1], p);
    return result_arr(&res[2], {m, k}, A.data(), tdtype);
}

/* ================================================================ spearmanr */

/* numpy.corrcoef of the rows of M (nv x no) as spearmanr calls it; rs is nv x nv */
void np_corrcoef(const std::vector<V> &M, int64_t no, V &rs)
{
    const int64_t nv = (int64_t)M.size();
    std::vector<V> X(nv, V(no));
    for (int64_t v = 0; v < nv; v++) {
        const double mean = psum(M[v].data(), no) / (double)no;
        for (int64_t o = 0; o < no; o++) X[v][o] = M[v][o] - mean;
    }
    const double f = 1.0 / (double)(no - 1);
    rs.assign(nv * nv, 0);
    for (int64_t i = 0; i < nv; i++)
        for (int64_t j = 0; j < nv; j++) rs[i * nv + j] = gemm_dot(no, X[i].data(), X[j].data()) * f;
    V sd(nv);
    for (int64_t i = 0; i < nv; i++) sd[i] = std::sqrt(rs[i * nv + i]);
    for (int64_t i = 0; i < nv; i++)
        for (int64_t j = 0; j < nv; j++) {
            double c = rs[i * nv + j] / sd[i];
            c = c / sd[j];
            rs[i * nv + j] = np_clip(c, -1.0, 1.0);
        }
}

double spear_t(double rs, double dof)
{
    double q = dof / ((rs + 1.0) * (1.0 - rs));
    if (q < 0) q = 0;                              /* .clip(0): NaN stays */
    return rs * std::sqrt(q);
}

/* mstats_basic.spearmanr's _spearmanr_2cols (nan_policy='omit') */
int spearman_omit_pair(const V &a, const V &b, int alt, double &r, double &p)
{
    V xa, xb;
    for (size_t i = 0; i < a.size(); i++)
        if (std::isfinite(a[i]) && std::isfinite(b[i])) { xa.push_back(a[i]); xb.push_back(b[i]); }
    const int64_t n = (int64_t)xa.size();
    bool any = false;
    for (int64_t i = 0; i < n; i++) if (xa[i] != 0 || xb[i] != 0) any = true;
    if (!any) { r = p = NaN; return TSR_OK; }
    const int64_t dof = n - 2;
    if (dof < 0) return fail("The input must have at least 3 entries!");
    V ra(n), rb(n);
    rank_average(xa.data(), n, ra.data());
    rank_average(xb.data(), n, rb.data());
    /* ma.corrcoef: column means (sequential sums), dot / fact, division by the outer product of the stds with
       masked-array domain handling (|c| * tiny >= |s_i s_j| leaves c unchanged) */
    double sa = 0, sb = 0;
    for (int64_t i = 0; i < n; i++) { sa = i ? sa + ra[i] : ra[i]; sb = i ? sb + rb[i] : rb[i]; }
    const double ma = sa / (double)n, mb = sb / (double)n;
    for (int64_t i = 0; i < n; i++) { ra[i] -= ma; rb[i] -= mb; }
    const double fact = (double)(n - 1);
    const double c00 = gemm_dot(n, ra.data(), ra.data()) / fact;
    const double c11 = gemm_dot(n, rb.data(), rb.data()) / fact;
    const double c10 = gemm_dot(n, rb.data(), ra.data()) / fact;
    const double s0 = std::sqrt(c00), s1 = std::sqrt(c11);
    const double den = s1 * s0;
    const double rs = std::fabs(c10) * std::numeric_limits<double>::min() >= std::fabs(den) ? c10 : c10 / den;
    const double t = spear_t(rs, (double)dof);
    r = rs;
    p = t_p((double)dof, t, alt);
    return TSR_OK;
}

int r_spearmanr(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr a, b;
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), a, "a"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), b, "b"))) return rc;
    if (!a.present) return fail("spearmanr() requires `a`");
    bool none;
    int64_t axis;
    int nanpol, alt;
    if ((rc = parse_axis(arg_at(args, nargs, 2), false, 0, none, axis))) return rc;
    if ((rc = parse_nan_policy(arg_at(args, nargs, 3), nanpol))) return rc;
    if ((rc = parse_alt(arg_at(args, nargs, 4), alt))) return rc;
    if (!none && axis > 1)
        return fail("spearmanr only handles 1-D or 2-D arrays, supplied axis argument %lld, please use only values 0, 1 or None for axis", (long long)axis);
    if (!none && axis < 0) return fail("spearmanr: negative axis is not supported by SciPy (tuple index out of range)");
    const int64_t ao = none ? 0 : axis;
    auto chk = [none](Arr &t) {
        if (none) t.shape = {t.size()};
        if (t.shape.empty()) t.shape = {1};
    };
    chk(a);
    if (a.ndim() > 2) return fail("spearmanr only handles 1-D or 2-D arrays");
    /* variables as rows: M[v][o] */
    std::vector<V> M;
    auto add_vars = [&](const Arr &t) -> int {
        if (t.ndim() == 1) { M.push_back(t.v); return TSR_OK; }
        const int64_t r = t.shape[0], c = t.shape[1];
        if (ao == 0) for (int64_t j = 0; j < c; j++) { V col(r); for (int64_t i = 0; i < r; i++) col[i] = t.v[i * c + j]; M.push_back(col); }
        else for (int64_t i = 0; i < r; i++) M.push_back(V(t.v.begin() + i * c, t.v.begin() + (i + 1) * c));
        return TSR_OK;
    };
    if (!b.present) {
        if (a.ndim() < 2) return fail("`spearmanr` needs at least 2 variables to compare");
    } else {
        chk(b);
        if (b.ndim() > 2) return fail("all the input arrays must have same number of dimensions");
    }
    add_vars(a);
    if (b.present) add_vars(b);
    const int64_t nv = (int64_t)M.size();
    const int64_t no = nv ? (int64_t)M[0].size() : 0;
    for (auto &v : M) if ((int64_t)v.size() != no) return fail("all the input array dimensions except for the concatenation axis must match exactly");
    auto scalar = [&](double r, double p) {
        fn_result_num(&res[0], r);
        fn_result_num(&res[1], p);
        return TSR_OK;
    };
    if (no <= 1) return scalar(NaN, NaN);
    if (nv < 2) return fail("index 1 is out of bounds for axis 1 with size 1");
    auto constant = [&](const V &v) { for (double t : v) if (!(t == v[0])) return false; return true; };
    if (constant(M[0]) || constant(M[1])) return scalar(NaN, NaN);
    bool has_nan = false;
    std::vector<bool> var_nan(nv, false);
    for (int64_t v = 0; v < nv; v++) for (double t : M[v]) if (t != t) { has_nan = true; var_nan[v] = true; }
    if (has_nan) {
        if (nanpol == 2) return fail("The input contains nan values");
        if (nanpol == 1) {
            if (nv == 2) {
                double r, p;
                if ((rc = spearman_omit_pair(M[0], M[1], alt, r, p))) return rc;
                return scalar(r, p);
            }
            V R(nv * nv, 1.0), P(nv * nv, 0.0);
            for (int64_t i = 0; i + 1 < nv; i++)
                for (int64_t j = i + 1; j < nv; j++) {
                    double r, p;
                    if ((rc = spearman_omit_pair(M[i], M[j], alt, r, p))) return rc;
                    R[i * nv + j] = R[j * nv + i] = r;
                    P[i * nv + j] = P[j * nv + i] = p;
                }
            if ((rc = result_arr(&res[0], {nv, nv}, R.data()))) return rc;
            return result_arr(&res[1], {nv, nv}, P.data());
        }
        if (nv <= 2) return scalar(NaN, NaN);
    }
    std::vector<V> Rk(nv, V(no));
    for (int64_t v = 0; v < nv; v++) {
        if (var_nan[v]) std::fill(Rk[v].begin(), Rk[v].end(), NaN);
        else rank_average(M[v].data(), no, Rk[v].data());
    }
    V rs;
    np_corrcoef(Rk, no, rs);
    const double dof = (double)(no - 2);
    V P(nv * nv);
    for (int64_t i = 0; i < nv * nv; i++) P[i] = t_p(dof, spear_t(rs[i], dof), alt);
    if (nv == 2) return scalar(rs[2], P[2]);
    for (int64_t i = 0; i < nv; i++)
        for (int64_t j = 0; j < nv; j++)
            if (var_nan[i] || var_nan[j]) rs[i * nv + j] = NaN;
    if ((rc = result_arr(&res[0], {nv, nv}, rs.data()))) return rc;
    return result_arr(&res[1], {nv, nv}, P.data());
}

/* ================================================================ binned_statistic_dd */

/* numpy.around(v, decimals) for float64 */
double np_around(double v, int64_t decimals)
{
    static const double p10[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8};
    const bool neg = decimals < 0;
    int64_t d = neg ? -decimals : decimals;
    double f;
    if (d < 9) f = p10[d];
    else { f = 1e9; while (d-- > 9) f *= 10.; }
    if (!neg) return std::nearbyint(v * f) / f;
    return std::nearbyint(v / f) * f;
}

enum { ST_MEAN, ST_MEDIAN, ST_COUNT, ST_SUM, ST_STD, ST_MIN, ST_MAX };

struct BinDim {
    bool count = true;       /* number of bins (else explicit edges) */
    double nb = 10;
    V edges;
};

struct BinnedOut {
    V stat;
    VI stat_shape;
    std::vector<V> edges;
    V binnumber;
    VI bn_shape;
};

/* sample: D columns of length N; values: shape (Vdim, Vlen) flattened; vshape the input shape of values
   (empty when None) */
int binned_dd(const std::vector<V> &sample, int64_t N, const Arr &values, int stat, std::vector<BinDim> bins,
              bool bins_python_int, const Arr &range, bool expand, BinnedOut &out)
{
    const int64_t D = (int64_t)sample.size();
    if (bins_python_int)
        for (auto &c : sample)
            for (double v : c)
                if (!std::isfinite(v)) return fail("sample contains non-finite values.");
    /* values */
    VI in_shape;
    int64_t Vdim = 1, Vlen = 1;
    V vals;
    if (values.present) {
        in_shape = values.shape;
        if (values.ndim() > 2) return fail("too many values to unpack (expected 2)");
        if (values.ndim() == 2) { Vdim = values.shape[0]; Vlen = values.shape[1]; }
        else if (values.ndim() == 1) { Vdim = 1; Vlen = values.shape[0]; }
        vals = values.v;
    } else {
        if (stat != ST_COUNT) return fail("The number of `values` elements must match the length of each `sample` dimension.");
        vals.assign(1, NaN);
    }
    if (stat != ST_COUNT && Vlen != N) return fail("The number of `values` elements must match the length of each `sample` dimension.");
    if ((int64_t)bins.size() != D) return fail("The dimension of bins must be equal to the dimension of the sample x.");
    /* _bin_edges */
    V smin(D), smax(D);
    if (!range.present) {
        for (int64_t i = 0; i < D; i++) {
            double lo = NaN, hi = NaN;
            bool nan = false;
            for (int64_t k = 0; k < N; k++) {
                const double v = sample[i][k];
                if (v != v) nan = true;
                if (k == 0 || v < lo) lo = v;
                if (k == 0 || v > hi) hi = v;
            }
            if (N == 0) return fail("zero-size array to reduction operation minimum which has no identity");
            smin[i] = nan ? NaN : lo;
            smax[i] = nan ? NaN : hi;
        }
    } else {
        if (range.ndim() != 2 || range.shape[1] != 2 || range.shape[0] != D)
            return fail("range given for %lld dimensions; %lld required", (long long)(range.ndim() ? range.shape[0] : 0), (long long)D);
        for (int64_t i = 0; i < D; i++) {
            if (range.v[2 * i + 1] < range.v[2 * i]) {
                if (D > 1) return fail("In dimension %lld of range, start must be <= stop", (long long)(i + 1));
                return fail("In range, start must be <= stop");
            }
            smin[i] = range.v[2 * i];
            smax[i] = range.v[2 * i + 1];
        }
    }
    for (int64_t i = 0; i < D; i++)
        if (smin[i] == smax[i]) { smin[i] = smin[i] - .5; smax[i] = smax[i] + .5; }
    VI nbin(D);
    out.edges.assign(D, V());
    for (int64_t i = 0; i < D; i++) {
        V &e = out.edges[i];
        if (bins[i].count) {
            /* SciPy: int(bins) bins, then arrays of that length; past what fits is its MemoryError (no UB) */
            if (!(bins[i].nb == bins[i].nb) || std::fabs(bins[i].nb) > 1e8) {
                fn_set_error("Unable to allocate memory for %g bins", bins[i].nb);
                return TSR_ENOMEM;
            }
            const int64_t nb = (int64_t)bins[i].nb;
            nbin[i] = nb + 2;
            const int64_t num = nbin[i] - 1;
            if (num < 0) return fail("Number of samples, %lld, must be non-negative.", (long long)num);
            e.resize(num);
            const int64_t div = num - 1;
            const double delta = smax[i] - smin[i];
            const double step = div > 0 ? delta / (double)div : NaN;
            for (int64_t k = 0; k < num; k++) {
                double y = (double)k;
                if (div > 0 && step == 0) y = y / (double)div * delta;
                else if (div > 0) y = y * step;
                else y = y * delta;
                e[k] = y + smin[i];
            }
            if (num > 1) e[num - 1] = smax[i];
        } else {
            e = bins[i].edges;
            nbin[i] = (int64_t)e.size() + 1;
        }
    }
    /* _bin_numbers */
    std::vector<VI> sampBin(D, VI(N));
    for (int64_t i = 0; i < D; i++) {
        const V &e = out.edges[i];
        const int64_t ne = (int64_t)e.size();
        bool inc = true, dec = true;
        for (int64_t k = 0; k + 1 < ne; k++) {
            if (!(e[k + 1] >= e[k])) inc = false;
            if (!(e[k + 1] <= e[k])) dec = false;
        }
        if (!inc && !dec) return fail("bins must be monotonically increasing or decreasing");
        for (int64_t k = 0; k < N; k++) {
            const double x = sample[i][k];
            if (inc) {
                sampBin[i][k] = std::upper_bound(e.begin(), e.end(), x, lt_nan) - e.begin();
            } else {
                V r(e.rbegin(), e.rend());
                sampBin[i][k] = ne - (std::upper_bound(r.begin(), r.end(), x, lt_nan) - r.begin());
            }
        }
        if (ne < 2) return fail("zero-size array to reduction operation minimum which has no identity");
        double dmin = 0;
        bool dnan = false;
        for (int64_t k = 0; k + 1 < ne; k++) {
            const double dd = e[k + 1] - e[k];
            if (dd != dd) dnan = true;
            if (k == 0 || dd < dmin) dmin = dd;
        }
        if (dnan) dmin = NaN;
        if (dmin == 0) return fail("The smallest edge difference is numerically 0.");
        const double lg = -std::log10(dmin);
        if (!std::isfinite(lg)) {
            if (lg != lg) return fail("cannot convert float NaN to integer");
            return fail("cannot convert float infinity to integer");
        }
        const int64_t decimal = (int64_t)std::trunc(lg) + 6;
        const double last = np_around(e[ne - 1], decimal);
        for (int64_t k = 0; k < N; k++) {
            const double x = sample[i][k];
            if (x >= e[ne - 1] && np_around(x, decimal) == last) sampBin[i][k] -= 1;
        }
    }
    /* the flattened bin grid is prod(nbins + 2) cells: SciPy allocates it with numpy (MemoryError when it cannot);
       check the size in floating point first, so a huge grid is an error rather than an int64 overflow */
    {
        double cells = 1.0;
        for (int64_t i = 0; i < D; i++) cells *= (double)nbin[i];
        const double limit = tsd::physical_memory_bytes();
        const int64_t budget = tsr_budget();
        const double cap = budget > 0 ? (double)budget : (limit > 0 ? limit : 4.5e15);
        if (cells * 8.0 > cap || cells > 4.5e15) return fail("Unable to allocate memory for a bin grid of %.3g cells", cells);
    }
    const int64_t total = prod(nbin);
    VI bn(N);
    for (int64_t k = 0; k < N; k++) {
        int64_t b = 0;
        for (int64_t i = 0; i < D; i++) b = b * nbin[i] + sampBin[i][k];
        bn[k] = b;
    }
    /* statistics */
    V result((size_t)(Vdim * total), stat == ST_COUNT || stat == ST_SUM ? 0.0 : NaN);
    VI flatcount(total, 0);
    for (int64_t k = 0; k < N; k++) flatcount[bn[k]]++;
    auto vat = [&](int64_t vv, int64_t k) { return vals[vv * Vlen + k]; };
    for (int64_t vv = 0; vv < Vdim; vv++) {
        double *r = &result[vv * total];
        if (stat == ST_COUNT) {
            for (int64_t b = 0; b < total; b++) r[b] = (double)flatcount[b];
        } else if (stat == ST_SUM || stat == ST_MEAN || stat == ST_STD) {
            V flatsum(total, 0.0);
            for (int64_t k = 0; k < N; k++) flatsum[bn[k]] += vat(vv, k);
            if (stat == ST_SUM) {
                for (int64_t b = 0; b < total; b++) r[b] = flatsum[b];
            } else if (stat == ST_MEAN) {
                for (int64_t b = 0; b < total; b++) if (flatcount[b]) r[b] = flatsum[b] / (double)flatcount[b];
            } else {
                V sq(total, 0.0);
                for (int64_t k = 0; k < N; k++) {
                    const double delta = vat(vv, k) - flatsum[bn[k]] / (double)flatcount[bn[k]];
                    sq[bn[k]] += delta * delta;
                }
                for (int64_t b = 0; b < total; b++) if (flatcount[b]) r[b] = std::sqrt(sq[b] / (double)flatcount[b]);
            }
        } else if (stat == ST_MEDIAN) {
            VI i(N);
            std::iota(i.begin(), i.end(), 0);
            std::stable_sort(i.begin(), i.end(), [&](int64_t a, int64_t b) {
                if (bn[a] != bn[b]) return bn[a] < bn[b];
                return lt_nan(vat(vv, a), vat(vv, b));
            });
            int64_t s = 0;
            while (s < N) {
                int64_t e = s + 1;
                while (e < N && bn[i[e]] == bn[i[s]]) e++;
                const double mid = (double)s + ((double)(e - s) - 1) / 2;
                const double a = vat(vv, i[(int64_t)std::floor(mid)]), b = vat(vv, i[(int64_t)std::ceil(mid)]);
                r[bn[i[s]]] = (a + b) / 2;
                s = e;
            }
        } else {
            /* min: the smallest non-NaN value (NaN only if every value is NaN); max: NaN if any NaN */
            if (stat == ST_MIN) {
                for (int64_t k = 0; k < N; k++) {
                    const double v = vat(vv, k);
                    double &c = r[bn[k]];
                    if (c != c || v < c) c = v;
                }
            } else {
                std::vector<char> seen(total, 0), nan(total, 0);
                for (int64_t k = 0; k < N; k++) {
                    const double v = vat(vv, k);
                    const int64_t b = bn[k];
                    if (v != v) nan[b] = 1;
                    else if (!seen[b] || v > r[b]) { r[b] = v; seen[b] = 1; }
                }
                for (int64_t b = 0; b < total; b++) if (nan[b]) r[b] = NaN;
            }
        }
    }
    /* core (drop the outlier bins) */
    VI core(D);
    int64_t ncore = 1;
    for (int64_t i = 0; i < D; i++) { core[i] = nbin[i] - 2; ncore *= core[i] > 0 ? core[i] : 0; }
    out.stat.assign((size_t)(Vdim * ncore), 0.0);
    VI cidx(D, 0);
    for (int64_t vv = 0; vv < Vdim; vv++) {
        std::fill(cidx.begin(), cidx.end(), 0);
        for (int64_t t = 0; t < ncore; t++) {
            int64_t b = 0;
            for (int64_t i = 0; i < D; i++) b = b * nbin[i] + (cidx[i] + 1);
            out.stat[vv * ncore + t] = result[vv * total + b];
            for (int64_t i = D - 1; i >= 0; i--) { if (++cidx[i] < core[i]) break; cidx[i] = 0; }
        }
    }
    out.stat_shape.clear();
    if (in_shape.size() > 1) out.stat_shape.assign(in_shape.begin(), in_shape.end() - 1);
    for (int64_t i = 0; i < D; i++) out.stat_shape.push_back(core[i]);
    if (expand && D > 1) {
        out.binnumber.resize(D * N);
        for (int64_t i = 0; i < D; i++)
            for (int64_t k = 0; k < N; k++) out.binnumber[i * N + k] = (double)sampBin[i][k];
        out.bn_shape = {D, N};
    } else {
        out.binnumber.resize(N);
        for (int64_t k = 0; k < N; k++) out.binnumber[k] = (double)bn[k];
        out.bn_shape = {N};
    }
    return TSR_OK;
}

int parse_statistic(const tsr_arg *a, int &st)
{
    if (a && a->kind != 0 && a->kind != 2) return fail("binned_statistic(): a callable `statistic` is not supported");
    std::string s;
    int rc = get_str(a, "mean", s, "statistic");
    if (rc) return rc;
    static const char *names[] = {"mean", "median", "count", "sum", "std", "min", "max"};
    for (int k = 0; k < 7; k++) if (s == names[k]) { st = k; return TSR_OK; }
    return fail("invalid statistic '%s'", s.c_str());
}

/* bins as SciPy's binned_statistic_dd sees them, for D dimensions: a number, a 1-D array of D numbers
   (or edges when edges_1d), or a 2-D array of D edge rows */
int parse_bins_dd(const tsr_arg *a, int64_t D, std::vector<BinDim> &bins, bool &pyint)
{
    pyint = false;
    bins.assign(D, BinDim());
    if (is_none(a)) { pyint = true; return TSR_OK; }
    if (a->kind == 1) {
        pyint = true;
        for (auto &b : bins) { b.count = true; b.nb = a->num; }
        return TSR_OK;
    }
    Arr b;
    int rc = get_arr(a, b, "bins");
    if (rc) return rc;
    if (b.ndim() == 1) {
        if (b.size() != D) return fail("The dimension of bins must be equal to the dimension of the sample x.");
        for (int64_t i = 0; i < D; i++) { bins[i].count = true; bins[i].nb = b.v[i]; }
        return TSR_OK;
    }
    if (b.ndim() == 2) {
        if (b.shape[0] != D) return fail("The dimension of bins must be equal to the dimension of the sample x.");
        const int64_t k = b.shape[1];
        for (int64_t i = 0; i < D; i++) {
            bins[i].count = false;
            bins[i].edges.assign(b.v.begin() + i * k, b.v.begin() + (i + 1) * k);
        }
        return TSR_OK;
    }
    return fail("`bins` must be an integer or an array");
}

int emit_binned(const BinnedOut &o, tsr_result *res, int nedge_results)
{
    int rc = result_arr(&res[0], o.stat_shape, o.stat.data());
    if (rc) return rc;
    for (int i = 0; i < nedge_results; i++)
        if ((rc = result_arr(&res[1 + i], {(int64_t)o.edges[i].size()}, o.edges[i].data()))) return rc;
    return result_arr(&res[1 + nedge_results], o.bn_shape, o.binnumber.data(), TSR_I64);
}

int r_binned_statistic(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr x, values, range;
    int rc, st;
    if ((rc = get_arr(arg_at(args, nargs, 0), x, "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), values, "values"))) return rc;
    if ((rc = parse_statistic(arg_at(args, nargs, 2), st))) return rc;
    if (!x.present || x.ndim() != 1) return fail("`x` must be a 1-D array");
    const tsr_arg *ba = arg_at(args, nargs, 3);
    std::vector<BinDim> bins(1);
    bool pyint = false;
    if (is_none(ba) || ba->kind == 1) {
        pyint = true;
        bins[0].nb = is_none(ba) ? 10 : ba->num;
    } else {
        Arr b;
        if ((rc = get_arr(ba, b, "bins"))) return rc;
        if (b.ndim() == 0 || b.size() == 1) bins[0].nb = b.v[0];
        else { bins[0].count = false; bins[0].edges = b.v; }
    }
    if ((rc = get_arr(arg_at(args, nargs, 4), range, "range"))) return rc;
    if (range.present && range.ndim() == 1 && range.size() == 2) range.shape = {1, 2};
    BinnedOut o;
    if ((rc = binned_dd({x.v}, x.size(), values, st, bins, pyint, range, false, o))) return rc;
    return emit_binned(o, res, 1);
}

int r_binned_statistic_2d(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr x, y, values, range;
    int rc, st;
    if ((rc = get_arr(arg_at(args, nargs, 0), x, "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), y, "y"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 2), values, "values"))) return rc;
    if ((rc = parse_statistic(arg_at(args, nargs, 3), st))) return rc;
    if (!x.present || !y.present || x.ndim() != 1 || y.ndim() != 1) return fail("`x` and `y` must be 1-D arrays");
    if (x.size() != y.size()) return fail("setting an array element with a sequence. The requested array has an inhomogeneous shape");
    const tsr_arg *ba = arg_at(args, nargs, 4);
    std::vector<BinDim> bins(2);
    bool pyint = false;
    if (is_none(ba) || ba->kind == 1) {
        pyint = true;
        for (auto &b : bins) b.nb = is_none(ba) ? 10 : ba->num;
    } else {
        Arr b;
        if ((rc = get_arr(ba, b, "bins"))) return rc;
        if (b.ndim() == 0) { for (auto &bb : bins) bb.nb = b.v[0]; }
        else if (b.shape[0] == 1) return fail("The dimension of bins must be equal to the dimension of the sample x.");
        else if (b.shape[0] == 2 && b.ndim() == 1) { bins[0].nb = b.v[0]; bins[1].nb = b.v[1]; }
        else if (b.shape[0] == 2 && b.ndim() == 2) {
            const int64_t k = b.shape[1];
            for (int i = 0; i < 2; i++) { bins[i].count = false; bins[i].edges.assign(b.v.begin() + i * k, b.v.begin() + (i + 1) * k); }
        } else if (b.ndim() == 1) {
            for (auto &bb : bins) { bb.count = false; bb.edges = b.v; }
        } else return fail("`bins` must be an integer, a pair or an array of edges");
    }
    if ((rc = get_arr(arg_at(args, nargs, 5), range, "range"))) return rc;
    bool expand;
    if ((rc = get_bool(arg_at(args, nargs, 6), false, expand, "expand_binnumbers"))) return rc;
    BinnedOut o;
    if ((rc = binned_dd({x.v, y.v}, x.size(), values, st, bins, pyint, range, expand, o))) return rc;
    return emit_binned(o, res, 2);
}

/* bin_edges is returned as one (D, k) array: the registry cannot return a list of arrays */
int r_binned_statistic_dd(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr sample, values, range;
    int rc, st;
    if ((rc = get_arr(arg_at(args, nargs, 0), sample, "sample"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), values, "values"))) return rc;
    if ((rc = parse_statistic(arg_at(args, nargs, 2), st))) return rc;
    if (!sample.present) return fail("`sample` is required");
    if (!is_none(arg_at(args, nargs, 6))) return fail("binned_statistic_dd(): `binned_statistic_result` is not supported");
    std::vector<V> cols;
    int64_t N;
    if (sample.ndim() == 2) {
        N = sample.shape[0];
        const int64_t D = sample.shape[1];
        for (int64_t i = 0; i < D; i++) {
            V c(N);
            for (int64_t k = 0; k < N; k++) c[k] = sample.v[k * D + i];
            cols.push_back(c);
        }
    } else if (sample.ndim() <= 1) {
        N = sample.size();
        cols.push_back(sample.v);
    } else return fail("too many values to unpack (expected 2)");
    const int64_t D = (int64_t)cols.size();
    std::vector<BinDim> bins;
    bool pyint;
    if ((rc = parse_bins_dd(arg_at(args, nargs, 3), D, bins, pyint))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 4), range, "range"))) return rc;
    bool expand;
    if ((rc = get_bool(arg_at(args, nargs, 5), false, expand, "expand_binnumbers"))) return rc;
    BinnedOut o;
    if ((rc = binned_dd(cols, N, values, st, bins, pyint, range, expand, o))) return rc;
    if ((rc = result_arr(&res[0], o.stat_shape, o.stat.data()))) return rc;
    const int64_t k = (int64_t)o.edges[0].size();
    for (auto &e : o.edges) if ((int64_t)e.size() != k) return fail("binned_statistic_dd(): bin edges of different lengths cannot be returned as one array");
    V all;
    for (auto &e : o.edges) all.insert(all.end(), e.begin(), e.end());
    if ((rc = result_arr(&res[1], {D, k}, all.data()))) return rc;
    return result_arr(&res[2], o.bn_shape, o.binnumber.data(), TSR_I64);
}

/* ================================================================ quantile */

enum { Q_INV, Q_AVG_INV, Q_CLOSEST, Q_HAZEN, Q_INTERP_INV, Q_LINEAR, Q_MED_UNB, Q_NORM_UNB, Q_WEIBULL, Q_HD,
       Q_LOWER, Q_MIDPOINT, Q_HIGHER, Q_NEAREST, Q_R_OUT, Q_R_IN, Q_R_NEAR };

/* numpy remainder(a, 1.0) */
inline double mod1(double a)
{
    double m = std::fmod(a, 1.0);
    if (m != 0) { if (m < 0) m += 1.0; }
    else m = 0.0;
    return m;
}

int r_quantile(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr x, p, w;
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), x, "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), p, "p"))) return rc;
    if (!x.present || !p.present) return fail("quantile() requires `x` and `p`");
    std::string ms;
    if ((rc = get_str(arg_at(args, nargs, 2), "linear", ms, "method"))) return rc;
    bool none;
    int64_t axis;
    if ((rc = parse_axis(arg_at(args, nargs, 3), false, 0, none, axis))) return rc;
    std::string nps;
    if ((rc = get_str(arg_at(args, nargs, 4), "propagate", nps, "nan_policy"))) return rc;
    const tsr_arg *ka = arg_at(args, nargs, 5);
    int keep = -1;                                   /* None */
    if (ka && ka->kind == 4) keep = ka->num != 0;
    else if (ka && ka->kind != 0) return fail("If specified, `keepdims` must be True or False.");
    if ((rc = get_arr(arg_at(args, nargs, 6), w, "weights"))) return rc;
    if (x.dtype == TSR_BOOL) return fail("`x` must have real dtype.");
    if (p.dtype == TSR_BOOL || is_int_dtype(p.dtype)) return fail("`p` must have real floating dtype.");
    if (w.present && w.dtype == TSR_BOOL) return fail("`weights` must have real dtype.");
    const int64_t ndim = std::max(x.ndim(), p.ndim());
    if (none) {
        x.shape = {x.size()};
        p.shape = {p.size()};
        if (w.present) w.shape = {w.size()};
        axis = 0;
    } else if (axis >= ndim || axis < -ndim) {
        return fail("`axis` is not compatible with the shapes of the inputs.");
    }
    static const char *names[] = {"inverted_cdf", "averaged_inverted_cdf", "closest_observation", "hazen",
                                  "interpolated_inverted_cdf", "linear", "median_unbiased", "normal_unbiased",
                                  "weibull", "harrell-davis", "_lower", "_midpoint", "_higher", "_nearest",
                                  "round_outward", "round_inward", "round_nearest"};
    int method = -1;
    for (int k = 0; k < 17; k++) if (ms == names[k]) method = k;
    if (method < 0) return fail("`method` must be one of the supported methods, got '%s'", ms.c_str());
    if (w.present && method >= Q_HD) return fail("`method='%s'` does not support `weights`.", ms.c_str());
    int nanpol;
    if (nps == "propagate") nanpol = 0;
    else if (nps == "omit") nanpol = 1;
    else if (nps == "raise") nanpol = 2;
    else return fail("nan_policy must be one of {'propagate', 'omit', 'raise'}");
    const bool contains_nans = x.has_nan();
    if (contains_nans && nanpol == 2) return fail("The input contains nan values");
    /* a zero-length axis becomes a length-1 axis of NaN */
    {
        const int64_t xa = axis < 0 ? axis + x.ndim() : axis;
        if (x.ndim() == 0 || xa < 0 || xa >= x.ndim()) return fail("tuple index out of range");
        if (x.shape[xa] == 0) {
            x.shape[xa] = 1;
            x.v.assign(prod(x.shape), NaN);
        }
    }
    const bool weighted = w.present;
    /* full broadcast of x and weights */
    if (weighted) {
        VI bx;
        if ((rc = broadcast({x.shape, w.shape}, bx))) return rc;
        auto expand = [&](Arr &a) {
            const VI st = bstrides(a.shape, bx);
            V out(prod(bx));
            VI idx(bx.size(), 0);
            for (int64_t t = 0; t < (int64_t)out.size(); t++) {
                int64_t off = 0;
                for (size_t d = 0; d < bx.size(); d++) off += idx[d] * st[d];
                out[t] = a.v[off];
                for (int64_t d = (int64_t)bx.size() - 1; d >= 0; d--) { if (++idx[d] < bx[d]) break; idx[d] = 0; }
            }
            a.v = out;
            a.shape = bx;
        };
        expand(x);
        expand(w);
    }
    /* pad to a common ndim and broadcast every dimension but the axis */
    const int64_t nd = std::max(x.ndim(), p.ndim());
    auto pad = [nd](Arr &a) { VI s(nd - a.ndim(), 1); s.insert(s.end(), a.shape.begin(), a.shape.end()); a.shape = s; };
    pad(x);
    pad(p);
    if (weighted) pad(w);
    const int64_t ax = axis < 0 ? axis + nd : axis;
    if (ax < 0 || ax >= nd) return fail("`axis` is out of bounds for array of dimension %lld", (long long)nd);
    VI B(nd);
    for (int64_t d = 0; d < nd; d++) {
        if (d == ax) { B[d] = 0; continue; }
        const int64_t a = x.shape[d], b = p.shape[d];
        if (a != 1 && b != 1 && a != b) return fail("Array shapes are incompatible for broadcasting.");
        B[d] = (a == 0 || b == 0) ? 0 : std::max(a, b);
    }
    const int64_t L = x.shape[ax], K = p.shape[ax];
    if (keep == 0 && K != 1) return fail("`keepdims` may be False only if the length of `p` along `axis` is 1.");
    const bool keepdims = keep < 0 ? K != 1 : keep != 0;
    /* strides */
    auto strides_of = [nd](const VI &s) { VI st(nd); int64_t acc = 1; for (int64_t d = nd - 1; d >= 0; d--) { st[d] = s[d] == 1 ? 0 : acc; acc *= s[d]; } return st; };
    const VI sx = strides_of(x.shape), sp = strides_of(p.shape);
    const int64_t sxa = x.shape[ax] == 1 ? 0 : [&] { int64_t acc = 1; for (int64_t d = nd - 1; d > ax; d--) acc *= x.shape[d]; return acc; }();
    const int64_t spa = p.shape[ax] == 1 ? 0 : [&] { int64_t acc = 1; for (int64_t d = nd - 1; d > ax; d--) acc *= p.shape[d]; return acc; }();
    VI loop;
    for (int64_t d = 0; d < nd; d++) if (d != ax) loop.push_back(B[d]);
    const int64_t nloop = prod(loop);
    /* np.vecdot over the sorted copy of x: ddot with a non-unit stride unless the axis is innermost */
    int64_t after = 1;
    for (int64_t d = ax + 1; d < nd; d++) after *= x.shape[d];
    const bool strided = after != 1 && L > 1;
    V out((size_t)(std::max<int64_t>(nloop, 0) * K));
    VI idx(loop.size(), 0);
    /* harrell-davis zeroes the NaNs of the omitted data only when no slice at all is NaN-ed out */
    bool any_nan_out = false;
    if (contains_nans && method == Q_HD) {
        for (int64_t it = 0; it < nloop && !any_nan_out; it++) {
            int64_t bx = 0, c = 0;
            for (int64_t d = 0; d < nd; d++) if (d != ax) bx += idx[c++] * sx[d];
            int64_t cnt = 0;
            for (int64_t t = 0; t < L; t++) { const double v = x.v[bx + t * sxa]; if (v != v) cnt++; }
            if (nanpol == 0 ? cnt > 0 : cnt == L) any_nan_out = true;
            for (int64_t cc = (int64_t)loop.size() - 1; cc >= 0; cc--) { if (++idx[cc] < loop[cc]) break; idx[cc] = 0; }
        }
        std::fill(idx.begin(), idx.end(), 0);
    }
    V y(L), wy(L), pk(K);
    VI order;
    for (int64_t it = 0; it < nloop; it++) {
        int64_t bx = 0, bp = 0, c = 0;
        for (int64_t d = 0; d < nd; d++) {
            if (d == ax) continue;
            bx += idx[c] * sx[d];
            bp += idx[c] * sp[d];
            c++;
        }
        for (int64_t t = 0; t < L; t++) y[t] = x.v[bx + t * sxa];
        for (int64_t t = 0; t < K; t++) pk[t] = p.v[bp + t * spa];
        double n = (double)L;
        if (weighted) {
            for (int64_t t = 0; t < L; t++) wy[t] = w.v[bx + t * sxa];
            int64_t nz = 0;
            for (int64_t t = 0; t < L; t++) if (wy[t] == 0) { nz++; y[t] = INF; }
            argsort_stable(y.data(), L, order);
            V ys(L), ws(L);
            for (int64_t t = 0; t < L; t++) { ys[t] = y[order[t]]; ws[t] = wy[order[t]]; }
            y = ys;
            wy = ws;
            n = (double)(L - nz);
        } else {
            std::sort(y.begin(), y.end(), lt_nan);
        }
        bool nan_out = false;
        if (contains_nans) {
            int64_t cnt = 0;
            for (int64_t t = 0; t < L; t++) if (y[t] != y[t]) cnt++;
            if (nanpol == 0) nan_out = cnt > 0;
            else {
                n = (double)((int64_t)n - cnt);
                nan_out = n == 0;
                if (nan_out) n = (double)L;
            }
            if (nan_out) std::fill(y.begin(), y.end(), NaN);
            else if (cnt > 0 && method == Q_HD && !any_nan_out) for (int64_t t = 0; t < L; t++) if (y[t] != y[t]) y[t] = 0;
        }
        V cw;
        if (weighted) {
            cw.resize(L);
            double acc = 0;
            for (int64_t t = 0; t < L; t++) { acc = t ? acc + wy[t] : wy[t]; cw[t] = acc; }
        }
        for (int64_t q = 0; q < K; q++) {
            double pp = pk[q];
            const bool pmask = pp > 1 || pp < 0 || pp != pp;
            if (pmask) pp = 0.5;
            double r;
            bool oob = false;                        /* take_along_axis raises IndexError */
            auto take = [&](double j) {
                int64_t k = (int64_t)j;
                if (k < -L || k >= L) { oob = true; return NaN; }
                if (k < 0) k += L;
                return y[k];
            };
            if (method <= Q_WEIBULL) {
                double m;
                switch (method) {
                case Q_CLOSEST: m = -0.5; break;
                case Q_HAZEN: m = 0.5; break;
                case Q_WEIBULL: m = pp; break;
                case Q_LINEAR: m = 1 - pp; break;
                case Q_MED_UNB: m = pp / 3 + 1. / 3; break;
                case Q_NORM_UNB: m = pp / 4 + 3. / 8; break;
                default: m = 0; break;
                }
                double jg, jp1, j;
                if (!weighted) {
                    jg = pp * n + m;
                    jp1 = std::floor(jg);
                    j = jp1 - 1;
                } else {
                    const int64_t ni = (int64_t)n;
                    const int64_t ti = ni - 1 < 0 ? ni - 1 + L : ni - 1;
                    const double total = cw[ti];
                    jg = pp * total + m;
                    jp1 = (double)(std::upper_bound(cw.begin(), cw.end(), jg, lt_nan) - cw.begin());
                    j = (double)(std::upper_bound(cw.begin(), cw.end(), jg - 1, lt_nan) - cw.begin());
                }
                double g = mod1(jg);
                if (method == Q_INV) g = g > 0 ? 1.0 : 0.0;
                else if (method == Q_AVG_INV) g = (1 + (g > 0 ? 1.0 : 0.0)) / 2;
                else if (method == Q_CLOSEST) {
                    const double jm = std::fmod(j, 2.0);
                    const double jmod = jm < 0 ? jm + 2.0 : jm;
                    g = 1 - ((g == 0 && jmod == 1) ? 1.0 : 0.0);
                }
                if (method == Q_INV || method == Q_AVG_INV || method == Q_CLOSEST)
                    if (jg < 0) g = 0;
                if (j < 0) g = 0;
                j = np_clip(j, 0., n - 1);
                jp1 = np_clip(jp1, 0., n - 1);
                r = (1 - g) * take(j) + g * take(jp1);
            } else if (method == Q_HD) {
                const double a = pp * (n + 1), b = (1 - pp) * (n + 1);
                V wt(L + 1), wd(L);
                for (int64_t i = 0; i <= L; i++) wt[i] = sc::betainc(a, b, (double)i / n);
                for (int64_t i = 0; i < L; i++) { wd[i] = wt[i + 1] - wt[i]; if (wd[i] != wd[i]) wd[i] = 0; }
                r = ddot(L, wd.data(), y.data(), strided);
            } else if (method >= Q_R_OUT) {
                double j;
                if (method == Q_R_OUT) j = pp < 0.5 ? std::floor(pp * n) : std::ceil(n * pp - 1);
                else if (method == Q_R_IN) j = pp < 0.5 ? std::ceil(pp * n) : std::floor(n * pp - 1);
                else j = pp < 0.5 ? std::nearbyint(pp * n) : std::nearbyint(n * pp - 1);
                r = take(j);
            } else {
                const double ij = pp * (n - 1);
                if (method == Q_MIDPOINT) r = (take(std::floor(ij)) + take(std::ceil(ij))) / 2;
                else if (method == Q_LOWER) r = take(std::floor(ij));
                else if (method == Q_HIGHER) r = take(std::ceil(ij));
                else r = take(std::nearbyint(ij));
            }
            if (oob) return fail("index %lld is out of bounds for axis 0 with size %lld", (long long)L, (long long)L);
            if (pmask) r = NaN;
            out[it * K + q] = r;
        }
        for (int64_t cc = (int64_t)loop.size() - 1; cc >= 0; cc--) { if (++idx[cc] < loop[cc]) break; idx[cc] = 0; }
    }
    /* the result: loop dims with the axis (length K) in place; squeezed unless keepdims */
    VI shape;
    if (none && keepdims) {
        shape.assign(ndim > 1 ? ndim - 1 : 0, 1);
        shape.push_back(K);
    } else {
        for (int64_t d = 0; d < nd; d++) {
            if (d == ax) { if (keepdims) shape.push_back(K); }
            else shape.push_back(B[d]);
        }
        if (!keepdims && K != 1) return fail("cannot select an axis to squeeze out which has size not equal to one");
    }
    /* out is laid out as (loop..., K); move K to the axis position */
    V fin(out.size());
    if (none || !keepdims || ax == nd - 1) fin = out;
    else {
        const int64_t inner = [&] { int64_t acc = 1; for (int64_t d = ax + 1; d < nd; d++) acc *= B[d]; return acc; }();
        const int64_t outer = inner > 0 ? nloop / inner : 0;
        for (int64_t o2 = 0; o2 < outer; o2++)
            for (int64_t q = 0; q < K; q++)
                for (int64_t i = 0; i < inner; i++) fin[(o2 * K + q) * inner + i] = out[(o2 * inner + i) * K + q];
    }
    return result_out(&res[0], shape, fin.data());
}

/* ================================================================ boxcox / yeojohnson with lmbda */

int r_boxcox(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr x, l;
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), x, "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), l, "lmbda"))) return rc;
    if (!x.present) return fail("boxcox() requires `x`");
    if (!l.present) return fail("boxcox(): lmbda=None (maximum-likelihood lambda) is not supported; pass lmbda");
    VI B;
    if ((rc = broadcast({x.shape, l.shape}, B))) return rc;
    const VI sx = bstrides(x.shape, B), sl = bstrides(l.shape, B);
    const int64_t n = prod(B);
    V out(std::max<int64_t>(n, 1));
    VI idx(B.size(), 0);
    for (int64_t t = 0; t < n; t++) {
        int64_t ox = 0, ol = 0;
        for (size_t d = 0; d < B.size(); d++) { ox += idx[d] * sx[d]; ol += idx[d] * sl[d]; }
        out[t] = sc::boxcox(x.v[ox], l.v[ol]);
        for (int64_t d = (int64_t)B.size() - 1; d >= 0; d--) { if (++idx[d] < B[d]) break; idx[d] = 0; }
    }
    return result_out(&res[0], B, out.data());
}

int r_yeojohnson(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    Arr x, l;
    int rc;
    if ((rc = get_arr(arg_at(args, nargs, 0), x, "x"))) return rc;
    if ((rc = get_arr(arg_at(args, nargs, 1), l, "lmbda"))) return rc;
    if (!x.present) return fail("yeojohnson() requires `x`");
    if (x.size() == 0) return result_arr(&res[0], x.shape, x.v.data(), is_int_dtype(x.dtype) ? TSR_I64 : TSR_F64);
    if (!l.present) return fail("yeojohnson(): lmbda=None (maximum-likelihood lambda) is not supported; pass lmbda");
    if (l.size() != 1) return fail("The truth value of an array with more than one element is ambiguous.");
    const double lm = l.v[0];
    const double eps = std::numeric_limits<double>::epsilon();
    V out(x.size());
    for (int64_t i = 0; i < x.size(); i++) {
        const double v = x.v[i];
        if (v >= 0) out[i] = std::fabs(lm) < eps ? std::log1p(v) : std::expm1(lm * std::log1p(v)) / lm;
        else out[i] = std::fabs(lm - 2) > eps ? -std::expm1((2 - lm) * std::log1p(-v)) / (2 - lm) : -std::log1p(-v);
    }
    return result_out(&res[0], x.shape, out.data());
}

}  // namespace

/* ================================================================ the table */

#define ALT3 "alternative=two-sided|less|greater"

static const fn_def DEFS[] = {
    GUFUNC_AX("stats.pearsonr", 2, 2, "x, y", "statistic, pvalue", ALT3, c_pearsonr, NULL, CORE_SCALAR, 0, 0, "0", 2, NULL,
              AX_SINGLE | AX_NOKEEP,
              "Pearson correlation coefficient and p-value (scipy.stats.pearsonr; method=None only, the "
              "confidence_interval method is not provided)."),
    GUFUNC("stats.spearmanrho", 2, 2, "x, y", "statistic, pvalue", ALT3, c_spearmanrho, NULL, CORE_SCALAR, CORE_SCALAR, 0, 0,
           0, 0, "0", NANP_PAIRED, 2, NULL, "Spearman's rank correlation (scipy.stats.spearmanrho; method=None only)."),
    GUFUNC("stats.pointbiserialr", 2, 2, "x, y", "statistic, pvalue", "", c_pointbiserialr, NULL, CORE_SCALAR, CORE_SCALAR, 0,
           0, 0, 0, "0", NANP_PAIRED, 2, NULL, "Point-biserial correlation coefficient (scipy.stats.pointbiserialr)."),
    GUFUNC("stats.kendalltau", 2, 2, "x, y", "statistic, pvalue", "method=auto|asymptotic|exact, variant=b|c, " ALT3,
           c_kendalltau, NULL, CORE_SCALAR, CORE_SCALAR, 0, 0, 0, 0, "None", NANP_PAIRED, 2, NULL,
           "Kendall's tau-b or tau-c with an exact or asymptotic p-value (scipy.stats.kendalltau)."),
    GUFUNC("stats.chatterjeexi", 2, 2, "x, y", "statistic, pvalue", "y_continuous=False", c_chatterjeexi, NULL, CORE_SCALAR,
           CORE_SCALAR, 0, 0, 0, 0, "0", NANP_PAIRED, 2, NULL,
           "Chatterjee's xi correlation with the asymptotic p-value (scipy.stats.chatterjeexi; method='asymptotic')."),
    ROUTINE("stats.spearmanr", 2, "a, b=None, axis=0, nan_policy='propagate', alternative='two-sided'", "statistic, pvalue",
            r_spearmanr, NULL, "Spearman rank-order correlation coefficient(s) and p-value(s) (scipy.stats.spearmanr)."),
    ROUTINE("stats.weightedtau", 2, "x, y, rank=True, weigher=None, additive=True, axis=None, nan_policy='propagate', keepdims=False",
            "statistic, pvalue", r_weightedtau, NULL,
            "Weighted Kendall's tau with the hyperbolic weigher (scipy.stats.weightedtau; weigher=None only)."),
    ROUTINE("stats.linregress", 6, "x, y, alternative='two-sided', axis=0, nan_policy='propagate', keepdims=False",
            "slope, intercept, rvalue, pvalue, stderr, intercept_stderr", r_linregress, NULL,
            "Least-squares regression line of two sets of measurements (scipy.stats.linregress)."),
    ROUTINE("stats.theilslopes", 4, "y, x=None, alpha=0.95, method='separate', axis=None, nan_policy='propagate', keepdims=False",
            "slope, intercept, low_slope, high_slope", r_theilslopes, NULL,
            "Theil-Sen estimator of a regression line with a confidence interval for the slope (scipy.stats.theilslopes)."),
    ROUTINE("stats.siegelslopes", 2, "y, x=None, method='hierarchical', axis=None, nan_policy='propagate', keepdims=False",
            "slope, intercept", r_siegelslopes, NULL, "Siegel's repeated-medians regression line (scipy.stats.siegelslopes)."),
    ROUTINE("stats.somersd", 3, "x, y=None, alternative='two-sided'", "statistic, pvalue, table", r_somersd, NULL,
            "Somers' D of two rankings or of a contingency table (scipy.stats.somersd)."),
    ROUTINE("stats.binned_statistic", 3, "x, values, statistic='mean', bins=10, range=None", "statistic, bin_edges, binnumber",
            r_binned_statistic, NULL, "A statistic of the values in each bin of x (scipy.stats.binned_statistic; named statistics only)."),
    ROUTINE("stats.binned_statistic_2d", 4, "x, y, values, statistic='mean', bins=10, range=None, expand_binnumbers=False",
            "statistic, x_edge, y_edge, binnumber", r_binned_statistic_2d, NULL,
            "A two-dimensional binned statistic (scipy.stats.binned_statistic_2d; named statistics only)."),
    ROUTINE("stats.binned_statistic_dd", 3,
            "sample, values, statistic='mean', bins=10, range=None, expand_binnumbers=False, binned_statistic_result=None",
            "statistic, bin_edges, binnumber", r_binned_statistic_dd, NULL,
            "A multidimensional binned statistic (scipy.stats.binned_statistic_dd; named statistics only; bin_edges as "
            "one (D, k) array)."),
    ROUTINE("stats.quantile", 1, "x, p, method='linear', axis=0, nan_policy='propagate', keepdims=None, weights=None",
            "quantile", r_quantile, NULL, "Quantiles of the data along an axis (scipy.stats.quantile)."),
    ROUTINE("stats.boxcox", 1, "x, lmbda=None, alpha=None, optimizer=None", "boxcox", r_boxcox, NULL,
            "The Box-Cox power transformation with a given lambda (scipy.stats.boxcox; lmbda required)."),
    ROUTINE("stats.yeojohnson", 1, "x, lmbda=None", "yeojohnson", r_yeojohnson, NULL,
            "The Yeo-Johnson power transformation with a given lambda (scipy.stats.yeojohnson; lmbda required)."),
};

extern "C" const fn_table TSR_STATS_FN_CORR_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
