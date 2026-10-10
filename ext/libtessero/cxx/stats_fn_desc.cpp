/*
 * scipy.stats descriptive statistics for the function registry (SciPy 1.17.1, scipy/stats/_stats_py.py,
 * _morestats.py, _entropy.py, _axis_nan_policy.py): describe, the generalised means, moments and their
 * standardised forms, z-scores, trimmed statistics, ranks, entropies, frequency tables, k-statistics,
 * L-moments, circular statistics, FDR control, Box-Cox / Yeo-Johnson log-likelihoods and the CDF distances.
 *
 * Each function follows SciPy's arithmetic step by step: np.mean = pairwise sum / n (fn_sum: sequential when
 * NumPy would reduce a non-innermost axis), np.var as _var/_xp_var compute it (mean of squared deviations,
 * then the correction factor), exponentiation by squares in _moment, the _axis_nan_policy decorator's
 * nan_policy semantics ('propagate' NaN-fills a slice with a NaN, 'omit' drops NaNs, 'raise' fails) and its
 * too-small rule (a slice with too few observations gives NaN).
 *
 * Kinds: generalised ufuncs for reductions whose parameters are numbers, booleans or enums; routines (with
 * a local axis/keepdims helper) where SciPy takes tuples, None-or-number, strings or several result arrays.
 */
#include "stats_dist.hpp"
#include "../src/fn.h"

extern "C" void tsr_special_iv(const void *, const double *, double *);   /* modified Bessel I_v (for vonmises_fisher) */
extern "C" int sl_dense_solve(double *M, double *b, int64_t n, int64_t nrhs);   /* np_linalg.c (row-major dgesv) */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

namespace {

using std::vector;

constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
constexpr double INF = std::numeric_limits<double>::infinity();
constexpr double EPS = std::numeric_limits<double>::epsilon();
constexpr double PI = 3.141592653589793;

enum { NP_PROPAGATE = 0, NP_OMIT = 1, NP_RAISE = 2 };
#define NANPOL "nan_policy=propagate|omit|raise"

inline bool isnan_(double v) { return v != v; }
inline bool lt_nan(double a, double b) { return a < b || (b != b && a == a); }
inline void sort_nan_last(double *x, int64_t n) { std::sort(x, x + n, lt_nan); }

bool any_nan(const double *x, int64_t n)
{
    for (int64_t i = 0; i < n; i++)
        if (x[i] != x[i]) return true;
    return false;
}

int64_t count_nan(const double *x, int64_t n)
{
    int64_t c = 0;
    for (int64_t i = 0; i < n; i++) c += x[i] != x[i];
    return c;
}

/* nan_policy on one slice: 0 go on (NaNs removed for 'omit'), 1 the slice is NaN ('propagate'), -1 error */
int nan_slice(double *x, int64_t &n, int pol)
{
    if (!any_nan(x, n)) return 0;
    if (pol == NP_RAISE) { fn_set_error("The input contains nan values"); return -1; }
    if (pol == NP_OMIT) {
        int64_t k = 0;
        for (int64_t i = 0; i < n; i++) if (x[i] == x[i]) x[k++] = x[i];
        n = k;
        return 0;
    }
    return 1;
}

/* paired samples (a, weights): omit drops a position when either is NaN */
int nan_paired(double *x, double *w, int64_t &n, int pol)
{
    const bool has = any_nan(x, n) || (w && any_nan(w, n));
    if (!has) return 0;
    if (pol == NP_RAISE) { fn_set_error("The input contains nan values"); return -1; }
    if (pol == NP_OMIT) {
        int64_t k = 0;
        for (int64_t i = 0; i < n; i++) {
            if (x[i] != x[i] || (w && w[i] != w[i])) continue;
            x[k] = x[i];
            if (w) w[k] = w[i];
            k++;
        }
        n = k;
        return 0;
    }
    return 1;
}

/* numpy.dot / vecdot of float64 vectors goes to OpenBLAS ddot: below 16 elements its tail loop is a sequential
   fused multiply-add (the blocked SIMD kernel above that is CPU-specific; the same sequence is used) */
double blas_dot(const double *a, const double *b, int64_t n)
{
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s = std::fma(b[i], a[i], s);
    return s;
}

inline double mean_(const double *x, int64_t n, unsigned fl) { return fn_sum(x, n, fl) / (double)n; }

/* numpy's floored remainder (npy_divmod) */
double py_mod(double a, double b)
{
    if (b == 0) return NaN;
    double mod = std::fmod(a, b);
    if (mod != 0) {
        if ((b < 0) != (mod < 0)) mod += b;
    } else {
        mod = std::copysign(0.0, b);
    }
    return mod;
}

double py_floordiv(double a, double b)
{
    if (b == 0) return a / b;
    double mod = std::fmod(a, b);
    double div = (a - mod) / b;
    if (mod != 0) {
        if ((b < 0) != (mod < 0)) div -= 1.0;
    }
    double fl;
    if (div != 0) {
        fl = std::floor(div);
        if (div - fl > 0.5) fl += 1.0;
    } else {
        fl = std::copysign(0.0, a / b);
    }
    return fl;
}

/* _moment(a, order, mean=center): exponentiation by squares on the deviations; `center_given` false with
   order 1 is the first central moment (0) */
double moment_raw(const double *x, int64_t n, double order, bool center_given, double center, unsigned fl,
                  double *tmp, double *azm)
{
    if (n == 0) return NaN;
    if (order == 0) return 1.0;
    if (order == 1 && !center_given) return 0.0;
    vector<double> nl;
    nl.push_back(order);
    double cur = order;
    while (cur > 2) {
        if (std::fmod(cur, 2.0) != 0) cur = (cur - 1) / 2;
        else cur /= 2;
        nl.push_back(cur);
    }
    const double mean = center_given ? center : mean_(x, n, fl);
    for (int64_t i = 0; i < n; i++) azm[i] = x[i] - mean;
    if (nl.back() == 1) for (int64_t i = 0; i < n; i++) tmp[i] = azm[i];
    else for (int64_t i = 0; i < n; i++) tmp[i] = azm[i] * azm[i];
    for (int k = (int)nl.size() - 2; k >= 0; k--) {
        const bool odd = std::fmod(nl[k], 2.0) != 0;
        for (int64_t i = 0; i < n; i++) {
            double s = tmp[i] * tmp[i];
            if (odd) s *= azm[i];
            tmp[i] = s;
        }
    }
    return mean_(tmp, n, fl);
}

int64_t scratch_3n(const int64_t *n, const double *) { return (n[0] + 8) * 8 * 3 + 64; }
int64_t scratch_4n(const int64_t *n, const double *) { return (n[0] + 8) * 8 * 4 + 64; }

/* ================================================================ generalised means */

enum { MEAN_G = 0, MEAN_H = 1, MEAN_P = 2 };

/* _xp_mean(x, weights) on a NaN-free slice */
double xp_wmean(double *x, const double *w, int64_t n, unsigned fl)
{
    if (!w) return mean_(x, n, fl);
    vector<double> ww(w, w + n);
    const double norm = fn_sum(ww.data(), n, fl);
    for (int64_t i = 0; i < n; i++) x[i] = x[i] * w[i];
    const double wsum = fn_sum(x, n, fl);
    return wsum / norm;
}

int c_pmean(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
            const int64_t *, void *, unsigned fl)
{
    const int kind = *(const int *)ctx;
    double *a = (double *)x[0];
    int64_t m = n[0];
    double pw = 0;
    int wi = 1;
    if (kind == MEAN_P) { pw = x[1][0]; wi = 2; }
    const int pol = (int)p[0];
    double *w = (double *)x[wi];
    if (n[wi] == 1 && m != 1) w = nullptr;               /* weights omitted (the binding passes 1.0) */
    else if (n[wi] != m) { fn_set_error("shape mismatch: objects cannot be broadcast to a single shape"); return TSR_ESHAPE; }
    if (kind == MEAN_P) {
        if (std::isinf(pw)) { fn_set_error("Power mean only implemented for finite `p`"); return TSR_EARG; }
    }
    const int r = nan_paired(a, w, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m == 0) { out[0][0] = NaN; return TSR_OK; }
    if (kind == MEAN_G || (kind == MEAN_P && pw == 0)) {
        for (int64_t i = 0; i < m; i++) a[i] = std::log(a[i]);
        out[0][0] = std::exp(xp_wmean(a, w, m, fl));
        return TSR_OK;
    }
    for (int64_t i = 0; i < m; i++) if (a[i] < 0) a[i] = NaN;
    if (kind == MEAN_H) {
        for (int64_t i = 0; i < m; i++) a[i] = 1.0 / a[i];
        out[0][0] = 1.0 / xp_wmean(a, w, m, fl);
        return TSR_OK;
    }
    for (int64_t i = 0; i < m; i++) {
        const double v = a[i];
        a[i] = pw == 2 ? v * v : (pw == 0.5 ? std::sqrt(v) : (pw == -1 ? 1.0 / v : (pw == 1 ? v : std::pow(v, pw))));
    }
    out[0][0] = std::pow(xp_wmean(a, w, m, fl), 1.0 / pw);
    return TSR_OK;
}

const int K_G = MEAN_G, K_H = MEAN_H, K_P = MEAN_P;

/* ================================================================ skew, kurtosis, moment, variation, sem */

int c_skew(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
           const int64_t *, void *scratch, unsigned fl)
{
    const bool kurt = ctx != nullptr;
    double *a = (double *)x[0];
    int64_t m = n[0];
    const bool fisher = kurt ? p[0] != 0 : false;
    const bool bias = kurt ? p[1] != 0 : p[0] != 0;
    const int pol = (int)p[kurt ? 2 : 1];
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m == 0) { out[0][0] = NaN; return TSR_OK; }
    double *tmp = (double *)scratch, *azm = tmp + m + 8;
    const double mean = mean_(a, m, fl);
    const double m2 = moment_raw(a, m, 2, true, mean, fl, tmp, azm);
    const double mk = moment_raw(a, m, kurt ? 4 : 3, true, mean, fl, tmp, azm);
    const double e = EPS * mean;
    const bool zero = m2 <= e * e;
    const double N = (double)m;
    double vals;
    if (!kurt) {
        vals = zero ? NaN : mk / std::pow(m2, 1.5);
        if (!bias && !zero && m > 2) vals = std::pow((N - 1.0) * N, 0.5) / (N - 2.0) * mk / std::pow(m2, 1.5);
    } else {
        /* m2**2.0 on a NumPy scalar is libm pow (not always m2*m2); the 1-D form is the one reproduced */
        const double m22 = std::pow(m2, 2.0);
        vals = zero ? NaN : mk / m22;
        if (!bias && !zero && m > 3) {
            const double nval = 1.0 / (N - 2) / (N - 3) * ((N * N - 1.0) * mk / m22 - 3 * std::pow(N - 1, 2.0));
            vals = nval + 3.0;
        }
        if (fisher) vals = vals - 3;
    }
    out[0][0] = vals;
    return TSR_OK;
}

const int ONE = 1;

int c_moment(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
             const int64_t *nout, void *scratch, unsigned fl)
{
    double *a = (double *)x[0];
    int64_t m = n[0];
    const int pol = (int)p[0];
    const double center = p[1];
    const bool cgiven = !isnan_(center);
    const double *ord = x[1];
    const int64_t no = n[1];
    if (no == 0) { fn_set_error("`order` must be a scalar or a non-empty 1D array."); return TSR_EARG; }
    for (int64_t k = 0; k < no; k++)
        if (ord[k] != std::nearbyint(ord[k])) { fn_set_error("All elements of `order` must be integral."); return TSR_EARG; }
        /* an infinite order passes SciPy's integrality test and then never returns (its exponentiation-by-squaring
           loop halves inf forever): Tessero raises instead (docs/project/reference-deviations.md) */
        else if (!std::isfinite(ord[k]) || std::fabs(ord[k]) > 9007199254740992.0) { fn_set_error("`order` must be finite."); return TSR_EARG; }
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m == 0) { for (int64_t k = 0; k < nout[0]; k++) out[0][k] = NaN; return TSR_OK; }
    double *tmp = (double *)scratch, *azm = tmp + m + 8;
    double mean = 0;
    bool need_mean = false;
    for (int64_t k = 0; k < no; k++) if (!cgiven && ord[k] > 1) need_mean = true;
    if (need_mean) mean = mean_(a, m, fl);
    for (int64_t k = 0; k < no; k++) {
        const double o = ord[k];
        if (!cgiven && o > 1) out[0][k] = moment_raw(a, m, o, true, mean, fl, tmp, azm);
        else out[0][k] = moment_raw(a, m, o, cgiven, center, fl, tmp, azm);
    }
    return TSR_OK;
}

/* numpy.std / var of a slice: sum((x - sum/n)^2) / max(n - ddof, 0) */
double np_var(const double *x, int64_t n, double ddof, unsigned fl, double *tmp)
{
    const double mean = fn_sum(x, n, fl) / (double)n;
    for (int64_t i = 0; i < n; i++) { const double d = x[i] - mean; tmp[i] = d * d; }
    double dof = (double)n - ddof;
    if (dof < 0) dof = 0;
    return fn_sum(tmp, n, fl) / dof;
}

int c_variation(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
                const int64_t *, void *scratch, unsigned fl)
{
    double *a = (double *)x[0];
    int64_t m = n[0];
    const int pol = (int)p[0];
    const double ddof = p[1];
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m == 0) { out[0][0] = NaN; return TSR_OK; }
    const double N = (double)m;
    const double mean_a = mean_(a, m, fl);
    const double std_a = std::sqrt(np_var(a, m, 0, fl, (double *)scratch));
    const double correction = std::pow(N / (N - ddof), 0.5);
    double res = std_a * correction / mean_a;
    if (ddof == N) res = std_a > 0 ? std::copysign(INF, mean_a) : NaN;
    out[0][0] = res;
    return TSR_OK;
}

int c_sem(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
          const int64_t *, void *scratch, unsigned fl)
{
    double *a = (double *)x[0];
    int64_t m = n[0];
    const double ddof = p[0];
    const int pol = (int)p[1];
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m <= 1) { out[0][0] = NaN; return TSR_OK; }     /* too_small = 1 */
    out[0][0] = std::sqrt(np_var(a, m, ddof, fl, (double *)scratch)) / std::pow((double)m, 0.5);
    return TSR_OK;
}

int64_t scratch_n(const int64_t *n, const double *) { return (n[0] + 8) * 8 + 64; }

/* ================================================================ z-scores, gstd (_xp_mean / _xp_var) */

/* _xp_mean with nan_policy: 'omit' gives NaNs zero weight (the sums still run over the whole slice) */
double xpm_mean(const double *x, int64_t n, bool omit, unsigned fl, double *tmp)
{
    if (!omit) return mean_(x, n, fl);
    for (int64_t i = 0; i < n; i++) tmp[i] = isnan_(x[i]) ? 0.0 : 1.0;
    const double norm = fn_sum(tmp, n, fl);
    for (int64_t i = 0; i < n; i++) tmp[i] = isnan_(x[i]) ? 0.0 : x[i];
    const double wsum = fn_sum(tmp, n, fl);
    return wsum / norm;
}

/* _xp_var(x, correction=ddof, nan_policy) */
double xpm_var(const double *x, int64_t n, double ddof, bool omit, unsigned fl, double *tmp, double *d)
{
    const double mean = xpm_mean(x, n, omit, fl, tmp);
    for (int64_t i = 0; i < n; i++) { const double t = x[i] - mean; d[i] = t * t; }
    double var = xpm_mean(d, n, omit, fl, tmp);
    if (ddof != 0) {
        double nn = (double)n;
        if (omit) nn = nn - (double)count_nan(x, n);
        const double nc = nn - ddof;
        const double factor = nc > 0 ? nn / nc : NaN;
        var *= factor;
    }
    return var;
}

enum { Z_SCORE = 0, Z_MAP = 1, Z_G = 2 };
const int K_ZS = Z_SCORE, K_ZM = Z_MAP, K_ZG = Z_G;

int c_zmap(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
           const int64_t *nout, void *scratch, unsigned fl)
{
    const int kind = *(const int *)ctx;
    const double ddof = p[0];
    const int pol = (int)p[1];
    double *s = (double *)x[0];
    const int64_t ns = n[0];
    double *c = kind == Z_MAP ? (double *)x[1] : s;
    const int64_t nc = kind == Z_MAP ? n[1] : ns;
    if (kind == Z_G) for (int64_t i = 0; i < ns; i++) s[i] = std::log(s[i]);
    if (pol == NP_RAISE && any_nan(c, nc)) { fn_set_error("The input contains nan values"); return TSR_EARG; }
    double *tmp = (double *)scratch, *d = tmp + nc + 8;
    double mn, sd;
    if (nc == 0) { mn = NaN; sd = NaN; }
    else {
        const bool omit = pol == NP_OMIT;
        mn = xpm_mean(c, nc, omit, fl, tmp);
        sd = std::sqrt(xpm_var(c, nc, ddof, omit, fl, tmp, d));
    }
    const bool zero = kind != Z_MAP && sd <= std::fabs(EPS * mn);
    for (int64_t i = 0; i < nout[0]; i++) out[0][i] = zero ? NaN : (s[i] - mn) / sd;
    return TSR_OK;
}

int64_t scratch_zmap(const int64_t *n, const double *) { return (std::max(n[0], n[1]) + 8) * 16 + 64; }
int64_t scratch_z1(const int64_t *n, const double *) { return (n[0] + 8) * 16 + 64; }

int c_gstd(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
           const int64_t *, void *scratch, unsigned fl)
{
    double *a = (double *)x[0];
    const int64_t m = n[0];
    const double ddof = p[0];
    const int pol = (int)p[1];
    if (pol == NP_RAISE && any_nan(a, m)) { fn_set_error("The input contains nan values"); return TSR_EARG; }
    if (m == 0) { out[0][0] = NaN; return TSR_OK; }
    for (int64_t i = 0; i < m; i++) a[i] = std::log(a[i]);
    double *tmp = (double *)scratch, *d = tmp + m + 8;
    out[0][0] = std::exp(std::sqrt(xpm_var(a, m, ddof, pol == NP_OMIT, fl, tmp, d)));
    return TSR_OK;
}

/* ================================================================ trim_mean */

int c_trim_mean(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
                const int64_t *, void *, unsigned fl)
{
    double *a = (double *)x[0];
    int64_t m = n[0];
    const double prop = x[1][0];
    const int pol = (int)p[0];
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m == 0) { out[0][0] = NaN; return TSR_OK; }
    if (!(std::fabs(prop * (double)m) < 9.2e18)) { fn_set_error("cannot convert float %s to integer", prop != prop ? "NaN" : "infinity"); return TSR_EARG; }
    const int64_t lowercut = (int64_t)(prop * (double)m);
    const int64_t uppercut = m - lowercut;
    if (lowercut > uppercut) { fn_set_error("Proportion too big."); return TSR_EARG; }
    /* np.partition(a, (lowercut, uppercut - 1)): a negative cut puts uppercut - 1 past the end */
    if (lowercut < 0) { fn_set_error("kth(=%lld) out of bounds (%lld)", (long long)(uppercut - 1), (long long)m); return TSR_EARG; }
    sort_nan_last(a, m);
    const int64_t k = uppercut - lowercut;
    out[0][0] = k > 0 ? fn_sum(a + lowercut, k, fl) / (double)k : NaN;
    return TSR_OK;
}

/* ================================================================ k-statistics */

double kstat_of(const double *x, int64_t n, int k, unsigned fl, double *tmp)
{
    const double N = (double)n;
    double S[5] = {0, 0, 0, 0, 0};
    for (int j = 1; j <= k; j++) {
        for (int64_t i = 0; i < n; i++) {
            const double v = x[i];
            tmp[i] = j == 1 ? v : (j == 2 ? v * v : std::pow(v, (double)j));
        }
        S[j] = fn_sum(tmp, n, fl);
    }
    switch (k) {
    case 1: return S[1] * 1.0 / N;
    case 2: return (N * S[2] - std::pow(S[1], 2.0)) / (N * (N - 1.0));
    case 3: return (2 * std::pow(S[1], 3.0) - 3 * N * S[1] * S[2] + N * N * S[3]) / (N * (N - 1.0) * (N - 2.0));
    default:
        return ((-6 * std::pow(S[1], 4.0) + 12 * N * std::pow(S[1], 2.0) * S[2] - 3 * N * (N - 1.0) * std::pow(S[2], 2.0) -
                 4 * N * (N + 1) * S[1] * S[3] + N * N * (N + 1) * S[4]) /
                (N * (N - 1.0) * (N - 2.0) * (N - 3.0)));
    }
}

int c_kstat(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
            const int64_t *, void *scratch, unsigned fl)
{
    const bool var = ctx != nullptr;
    double *a = (double *)x[0];
    int64_t m = n[0];
    const double kk = x[1][0];
    const int pol = (int)p[0];
    if (kk != kk) { fn_set_error("cannot convert float NaN to integer"); return TSR_EARG; }   /* int(n) */
    if (!var) {
        if (kk > 4 || kk < 1) { fn_set_error("k-statistics only supported for 1<=n<=4"); return TSR_EARG; }
    } else if (kk != 1 && kk != 2) {
        fn_set_error("Only n=1 or n=2 supported.");
        return TSR_EARG;
    }
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m == 0) { out[0][0] = NaN; return TSR_OK; }
    double *tmp = (double *)scratch;
    const double N = (double)m;
    if (!var) { out[0][0] = kstat_of(a, m, (int)kk, fl, tmp); return TSR_OK; }
    if (kk == 1) { out[0][0] = kstat_of(a, m, 2, fl, tmp) * 1.0 / N; return TSR_OK; }
    const double k2 = kstat_of(a, m, 2, fl, tmp), k4 = kstat_of(a, m, 4, fl, tmp);
    out[0][0] = (2 * N * std::pow(k2, 2.0) + (N - 1) * k4) / (N * (N + 1));
    return TSR_OK;
}

/* ================================================================ circular statistics */

enum { CIRC_MEAN = 0, CIRC_VAR = 1, CIRC_STD = 2 };
const int K_CM = CIRC_MEAN, K_CV = CIRC_VAR, K_CS = CIRC_STD;

int c_circ(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
           const int64_t *, void *scratch, unsigned fl)
{
    const int kind = *(const int *)ctx;
    double *a = (double *)x[0];
    int64_t m = n[0];
    const double high = x[1][0], low = x[2][0];
    const int pol = (int)p[0];
    const bool normalize = kind == CIRC_STD ? p[1] != 0 : false;
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1 || m == 0) { out[0][0] = NaN; return TSR_OK; }
    const double period = high - low;
    double *sn = (double *)scratch, *cs = sn + m + 8;
    const double f = (2.0 * PI) / period;
    for (int64_t i = 0; i < m; i++) {
        const double s = a[i] * f;
        sn[i] = std::sin(s);
        cs[i] = std::cos(s);
    }
    if (kind == CIRC_MEAN) {
        const double ss = fn_sum(sn, m, fl), cc = fn_sum(cs, m, fl);
        const double res = std::atan2(ss, cc);
        out[0][0] = py_mod(res * (period / (2.0 * PI)) - low, period) + low;
        return TSR_OK;
    }
    const double sm = mean_(sn, m, fl), cm = mean_(cs, m, fl);
    const double hyp = std::pow(std::pow(sm, 2.0) + std::pow(cm, 2.0), 0.5);
    const double R = hyp > 1.0 ? 1.0 : hyp;
    if (kind == CIRC_VAR) { out[0][0] = 1. - R; return TSR_OK; }
    double res = std::pow(-2 * std::log(R), 0.5) + 0.0;
    if (!normalize) res *= (high - low) / (2. * PI);
    out[0][0] = res;
    return TSR_OK;
}

int64_t scratch_2n(const int64_t *n, const double *) { return (n[0] + 8) * 16 + 64; }

/* ================================================================ entropy */

int c_entropy(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
              const int64_t *, void *, unsigned fl)
{
    double *pk = (double *)x[0];
    int64_t m = n[0];
    double *qk = (double *)x[1];
    if (n[1] == 1 && m != 1) qk = nullptr;
    else if (n[1] != m) { fn_set_error("shape mismatch: objects cannot be broadcast to a single shape"); return TSR_ESHAPE; }
    const double base = p[0];
    const int pol = (int)p[1];
    if (!isnan_(base) && base <= 0) { fn_set_error("`base` must be a positive number or `None`."); return TSR_EARG; }
    const int r = nan_paired(pk, qk, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1) { out[0][0] = NaN; return TSR_OK; }
    if (qk) {
        const double sq = fn_sum(qk, m, fl);
        for (int64_t i = 0; i < m; i++) qk[i] = qk[i] / sq;
    }
    const double sp = fn_sum(pk, m, fl);
    for (int64_t i = 0; i < m; i++) pk[i] = pk[i] / sp;
    for (int64_t i = 0; i < m; i++) pk[i] = qk ? sc::rel_entr(pk[i], qk[i]) : sc::entr(pk[i]);
    double S = fn_sum(pk, m, fl);
    if (!isnan_(base)) S /= std::log(base);
    out[0][0] = S;
    return TSR_OK;
}

/* ================================================================ differential entropy */

enum { DE_AUTO, DE_VASICEK, DE_VANES, DE_CORREA, DE_EBRAHIMI };

int c_diffent(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
              const int64_t *, void *, unsigned)
{
    double *a = (double *)x[0];
    int64_t m = n[0];
    const double wl_in = p[0], base = p[1];
    int method = (int)p[2];
    const int pol = (int)p[3];
    const int r = nan_slice(a, m, pol);
    if (r < 0) return TSR_EARG;
    if (r == 1) { out[0][0] = NaN; return TSR_OK; }
    const double N = (double)m;
    const double wl = isnan_(wl_in) ? std::floor(std::sqrt(N) + 0.5) : wl_in;
    if (!(2 <= 2 * wl && 2 * wl < N)) { out[0][0] = NaN; return TSR_OK; }   /* too small */
    if (!isnan_(base) && base <= 0) { fn_set_error("`base` must be a positive number or `None`."); return TSR_EARG; }
    const int64_t mw = (int64_t)wl;
    sort_nan_last(a, m);
    if (method == DE_AUTO) method = m <= 10 ? DE_VANES : (m <= 1000 ? DE_EBRAHIMI : DE_VASICEK);
    vector<double> X((size_t)(m + 2 * mw));
    for (int64_t i = 0; i < mw; i++) { X[i] = a[0]; X[m + mw + i] = a[m - 1]; }
    for (int64_t i = 0; i < m; i++) X[mw + i] = a[i];
    vector<double> t((size_t)m + 1);
    double res;
    const double M = (double)mw;
    if (method == DE_VASICEK) {
        for (int64_t i = 0; i < m; i++) t[i] = std::log(N / (2 * M) * (X[i + 2 * mw] - X[i]));
        res = tsr_psum(t.data(), m) / N;
    } else if (method == DE_VANES) {
        const int64_t k = m - mw;
        for (int64_t i = 0; i < k; i++) t[i] = std::log((N + 1) / M * (a[i + mw] - a[i]));
        const double term1 = 1 / (N - M) * tsr_psum(t.data(), k);
        const int64_t nk = m + 1 - mw;
        for (int64_t i = 0; i < nk; i++) t[i] = 1 / (M + (double)i);
        res = term1 + tsr_psum(t.data(), nk) + std::log(M) - std::log(N + 1);
    } else if (method == DE_EBRAHIMI) {
        for (int64_t i = 0; i < m; i++) {
            const double ii = (double)(i + 1);
            double ci = ii <= M ? 1 + (ii - 1) / M : 2.;
            if (ii >= N - M + 1) ci = 1 + (N - ii) / M;
            t[i] = std::log(N * (X[i + 2 * mw] - X[i]) / (ci * M));
        }
        res = tsr_psum(t.data(), m) / N;
    } else {
        /* correa: windows of 2m+1 padded values; reductions over the window axis are sequential */
        const int64_t w = 2 * mw + 1;
        for (int64_t i = 0; i < m; i++) {
            /* j0 = i + dj + m - 1 (1-based i) -> padded indices i .. i + 2m (0-based i) */
            double acc = X[i];
            for (int64_t d = 1; d < w; d++) acc += X[i + d];
            const double xbar = acc / (double)w;
            double num = 0, den = 0;
            for (int64_t d = 0; d < w; d++) {
                const double diff = X[i + d] - xbar;
                const double dj = (double)(d - mw);
                if (d == 0) { num = diff * dj; den = diff * diff; }
                else { num += diff * dj; den += diff * diff; }
            }
            den = N * den;
            t[i] = std::log(num / den);
        }
        res = -(tsr_psum(t.data(), m) / N);
    }
    if (!isnan_(base)) res /= std::log(base);
    out[0][0] = res;
    return TSR_OK;
}

/* ================================================================ false_discovery_control */

int c_fdc(const void *, const double *const *x, const int64_t *n, const double *p, double *const *out,
          const int64_t *, void *, unsigned)
{
    const double *ps = x[0];
    const int64_t m = n[0];
    const int by = (int)p[0] == 1;
    for (int64_t i = 0; i < m; i++)
        if (!(ps[i] >= 0 && ps[i] <= 1)) { fn_set_error("`ps` must include only numbers between 0 and 1."); return TSR_EARG; }
    if (m <= 1) { for (int64_t i = 0; i < m; i++) out[0][i] = ps[i]; return TSR_OK; }
    vector<int64_t> order((size_t)m);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return ps[a] < ps[b]; });
    vector<double> s((size_t)m);
    const double M = (double)m;
    for (int64_t k = 0; k < m; k++) s[k] = ps[order[k]] * (M / (double)(k + 1));
    if (by) {
        vector<double> inv((size_t)m);
        for (int64_t k = 0; k < m; k++) inv[k] = 1 / (double)(k + 1);
        const double h = tsr_psum(inv.data(), m);
        for (int64_t k = 0; k < m; k++) s[k] = s[k] * h;
    }
    for (int64_t k = m - 2; k >= 0; k--) if (s[k + 1] < s[k]) s[k] = s[k + 1];
    for (int64_t k = 0; k < m; k++) {
        double v = s[k];
        if (v < 0) v = 0;
        if (v > 1) v = 1;
        out[0][order[k]] = v;
    }
    return TSR_OK;
}

/* ================================================================ routines: argument helpers */

bool is_null(const tsr_arg *a, int nargs, int k) { return k >= nargs || a[k].kind == 0; }

double arg_num(const tsr_arg *a, int nargs, int k, double def)
{
    if (k >= nargs || a[k].kind == 0) return def;
    if (a[k].kind == 1 || a[k].kind == 4) return a[k].num;
    return def;
}

bool arg_bool(const tsr_arg *a, int nargs, int k, bool def) { return arg_num(a, nargs, k, def ? 1 : 0) != 0; }

const char *arg_str(const tsr_arg *a, int nargs, int k, const char *def)
{
    if (k >= nargs || a[k].kind != 2 || !a[k].str) return def;
    return a[k].str;
}

bool is_int_dtype(int dt) { return dt == TSR_I64 || dt == TSR_I32 || dt == TSR_U8 || dt == TSR_BOOL; }

int nan_policy_of(const tsr_arg *a, int nargs, int k, int *pol)
{
    const char *s = arg_str(a, nargs, k, "propagate");
    if (!strcmp(s, "propagate")) *pol = NP_PROPAGATE;
    else if (!strcmp(s, "omit")) *pol = NP_OMIT;
    else if (!strcmp(s, "raise")) *pol = NP_RAISE;
    else { fn_set_error("nan_policy must be one of {'propagate', 'raise', 'omit'}"); return TSR_EARG; }
    return TSR_OK;
}

/* an array argument as doubles (numbers become 0-d arrays) */
struct Arr {
    vector<double> v;
    int ndim = 0;
    int64_t shape[TSR_MAXDIM] = {0};
    int64_t strides[TSR_MAXDIM] = {0};
    int dtype = TSR_F64;
    bool ok = false;
    int64_t size() const { return (int64_t)v.size(); }
};

bool get_arr(const tsr_arg *a, int nargs, int k, Arr &A)
{
    if (k >= nargs) return false;
    if (a[k].kind == 1 || a[k].kind == 4) {
        A.v.assign(1, a[k].num);
        A.ndim = 0;
        A.dtype = a[k].kind == 4 ? TSR_BOOL : (a[k].num == std::floor(a[k].num) && a[k].flags == 0 ? TSR_F64 : TSR_F64);
        A.ok = true;
        return true;
    }
    if (a[k].kind != 3) return false;
    const tsr_array &x = a[k].arr;
    if (x.dtype == TSR_C128) return false;
    int64_t n = 0;
    double *d = fn_arg_doubles(&a[k], &n);
    if (!d) return false;
    A.v.assign(d, d + n);
    fn_free_doubles(d, n);
    A.ndim = x.ndim;
    for (int i = 0; i < x.ndim; i++) { A.shape[i] = x.shape[i]; A.strides[i] = x.strides[i]; }
    A.dtype = x.dtype;
    A.ok = true;
    return true;
}

/* lanes along one axis (or the flattened array for axis None), C order over the other dimensions */
struct Lanes {
    vector<double> buf;
    int64_t nl = 0, len = 0;
    int ndim_in = 0;
    int64_t shape_in[TSR_MAXDIM] = {0};
    int axis = -1;                            /* -1: None */
    int ond = 0;
    int64_t oshape[TSR_MAXDIM] = {0};
    unsigned flags = 0;
    bool is_int = false;
    double *lane(int64_t i) { return buf.data() + i * len; }
};

int make_lanes(const Arr &A, const tsr_arg *args, int nargs, int kaxis, Lanes &L, bool axis_default_none)
{
    L.ndim_in = A.ndim;
    L.is_int = is_int_dtype(A.dtype);
    for (int i = 0; i < A.ndim; i++) L.shape_in[i] = A.shape[i];
    bool none = axis_default_none;
    long ax = 0;
    if (kaxis >= 0 && kaxis < nargs) {
        if (args[kaxis].kind == 0) none = true;
        else if (args[kaxis].kind == 1) {
            none = false;
            const double v = args[kaxis].num;
            if (v != std::floor(v)) { fn_set_error("axis must be an integer"); return TSR_EARG; }
            ax = (long)std::fmax(-1e9, std::fmin(1e9, v));   /* out of range stays out of range, no UB */
        }
    }
    if (none) {
        L.axis = -1;
        L.nl = 1;
        L.len = A.size();
        L.buf = A.v;
        L.ond = 0;
        return TSR_OK;
    }
    int nd = A.ndim;
    int64_t shp[TSR_MAXDIM];
    int64_t st[TSR_MAXDIM];
    for (int i = 0; i < nd; i++) { shp[i] = A.shape[i]; st[i] = A.strides[i]; }
    if (nd == 0) { nd = 1; shp[0] = 1; st[0] = 8; L.ndim_in = 1; L.shape_in[0] = 1; }
    if (ax < -nd || ax >= nd) { fn_set_error("axis %ld is out of bounds for array of dimension %d", ax, nd); return TSR_EINDEX; }
    if (ax < 0) ax += nd;
    L.axis = (int)ax;
    L.len = shp[ax];
    L.ond = 0;
    int64_t outer = 1;
    for (int i = 0; i < nd; i++) if (i != ax) { L.oshape[L.ond++] = shp[i]; outer *= shp[i]; }
    L.nl = outer;
    L.buf.assign((size_t)(outer * L.len), 0.0);
    /* C-order strides in elements of the logical array */
    int64_t cst[TSR_MAXDIM];
    int64_t s = 1;
    for (int i = nd - 1; i >= 0; i--) { cst[i] = s; s *= shp[i]; }
    for (int64_t o = 0; o < outer; o++) {
        /* multi-index of lane o over the non-axis dims */
        int64_t rem = o, base = 0;
        for (int i = nd - 1; i >= 0; i--) {
            if (i == ax) continue;
            const int64_t idx = rem % shp[i];
            rem /= shp[i];
            base += idx * cst[i];
        }
        for (int64_t t = 0; t < L.len; t++) L.buf[o * L.len + t] = A.v[base + t * cst[ax]];
    }
    /* NumPy reduces pairwise only along the innermost (smallest stride) axis */
    if (nd > 1) {
        int best = -1;
        int64_t bs = 0;
        for (int i = 0; i < nd; i++) {
            if (shp[i] <= 1) continue;
            const int64_t sa = st[i] < 0 ? -st[i] : st[i];
            if (best < 0 || sa < bs) { best = i; bs = sa; }
        }
        if (best >= 0 && best != ax) L.flags |= FN_SEQUENTIAL;
    }
    return TSR_OK;
}

/* one result per lane: a scalar for a 0-d result, else an array of the lane shape (keepdims: axis kept as 1) */
int emit(tsr_result *r, const Lanes &L, const double *vals, bool keepdims, bool as_int)
{
    int nd = 0;
    int64_t shp[TSR_MAXDIM];
    if (keepdims) {
        if (L.axis < 0) { nd = L.ndim_in > 0 ? L.ndim_in : 1; for (int i = 0; i < nd; i++) shp[i] = 1; }
        else { nd = L.ndim_in; for (int i = 0; i < nd; i++) shp[i] = i == L.axis ? 1 : L.shape_in[i]; }
    } else {
        nd = L.ond;
        for (int i = 0; i < nd; i++) shp[i] = L.oshape[i];
    }
    if (nd == 0) {
        if (as_int) fn_result_int(r, (int64_t)vals[0]);
        else fn_result_num(r, vals[0]);
        return TSR_OK;
    }
    const int64_t cnt = L.nl;
    if (as_int) {
        int64_t *o = (int64_t *)fn_result_array(r, TSR_I64, nd, shp);
        if (!o) return TSR_ENOMEM;
        for (int64_t i = 0; i < cnt; i++) o[i] = (int64_t)vals[i];
    } else {
        double *o = (double *)fn_result_array(r, TSR_F64, nd, shp);
        if (!o) return TSR_ENOMEM;
        for (int64_t i = 0; i < cnt; i++) o[i] = vals[i];
    }
    return TSR_OK;
}

/* an array result of arbitrary shape */
int emit_arr(tsr_result *r, int nd, const int64_t *shp, const double *vals, bool as_int)
{
    int64_t cnt = 1;
    for (int i = 0; i < nd; i++) cnt *= shp[i];
    if (as_int) {
        int64_t *o = (int64_t *)fn_result_array(r, TSR_I64, nd, shp);
        if (!o) return TSR_ENOMEM;
        for (int64_t i = 0; i < cnt; i++) o[i] = (int64_t)vals[i];
    } else {
        double *o = (double *)fn_result_array(r, TSR_F64, nd, shp);
        if (!o) return TSR_ENOMEM;
        for (int64_t i = 0; i < cnt; i++) o[i] = vals[i];
    }
    return TSR_OK;
}

#define NEED_ARRAY(k, A, name)                                                           \
    Arr A;                                                                               \
    if (!get_arr(args, nargs, k, A)) { fn_set_error("%s must be an array", name); return TSR_EARG; }


/* lanes of new length k (C order over the other dims) back into an array with the axis in its place */
void unlanes(const Lanes &L, int64_t k, const double *lanes, int *nd, int64_t *shp, vector<double> &out)
{
    if (L.axis < 0) {
        *nd = 1;
        shp[0] = k;
        out.assign(lanes, lanes + k);
        return;
    }
    *nd = L.ndim_in;
    for (int i = 0; i < L.ndim_in; i++) shp[i] = i == L.axis ? k : L.shape_in[i];
    int64_t total = L.nl * k;
    out.assign((size_t)total, 0.0);
    int64_t cst[TSR_MAXDIM];
    int64_t s = 1;
    for (int i = L.ndim_in - 1; i >= 0; i--) { cst[i] = s; s *= shp[i]; }
    for (int64_t o = 0; o < L.nl; o++) {
        int64_t rem = o, base = 0;
        for (int i = L.ndim_in - 1; i >= 0; i--) {
            if (i == L.axis) continue;
            const int64_t idx = rem % shp[i];
            rem /= shp[i];
            base += idx * cst[i];
        }
        for (int64_t t = 0; t < k; t++) out[base + t * cst[L.axis]] = lanes[o * k + t];
    }
}

/* ================================================================ describe */

int r_describe(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "a");
    const double ddof = arg_num(args, nargs, 2, 1);
    const bool bias = arg_bool(args, nargs, 3, true);
    int pol;
    if (nan_policy_of(args, nargs, 4, &pol) != TSR_OK) return TSR_EARG;
    Lanes L;
    int rc = make_lanes(A, args, nargs, 1, L, false);
    if (rc != TSR_OK) return rc;
    const bool hasnan = any_nan(A.v.data(), A.size());
    if (hasnan && pol == NP_RAISE) { fn_set_error("The input contains nan values"); return TSR_EARG; }
    const bool omit = hasnan && pol == NP_OMIT;
    if (!omit && A.size() == 0) { fn_set_error("The input must not be empty."); return TSR_EARG; }
    const int64_t nl = L.nl, n = L.len;
    vector<double> nobs(nl), mn(nl), mx(nl), mean(nl), var(nl), sk(nl), ku(nl), tmp(n + 8), azm(n + 8), f(n + 8);
    const unsigned fl = L.flags;
    for (int64_t o = 0; o < nl; o++) {
        double *x = L.lane(o);
        if (!omit) {
            nobs[o] = (double)n;
            double lo = x[0], hi = x[0];
            for (int64_t i = 1; i < n; i++) {
                if (isnan_(lo)) break;
                if (isnan_(x[i])) { lo = hi = x[i]; break; }
                if (x[i] < lo) lo = x[i];
                if (x[i] > hi) hi = x[i];
            }
            mn[o] = lo; mx[o] = hi;
            const double m = mean_(x, n, fl);
            mean[o] = m;
            double v = moment_raw(x, n, 2, true, m, fl, tmp.data(), azm.data());
            if (ddof != 0) v *= ((double)n / ((double)n - ddof));
            var[o] = v;
            if (any_nan(x, n)) { sk[o] = ku[o] = NaN; continue; }
            const double m2 = v != v ? v : moment_raw(x, n, 2, true, m, fl, tmp.data(), azm.data());
            const double m3 = moment_raw(x, n, 3, true, m, fl, tmp.data(), azm.data());
            const double m4 = moment_raw(x, n, 4, true, m, fl, tmp.data(), azm.data());
            const double e = EPS * m;
            const bool zero = m2 <= e * e;
            const double N = (double)n;
            double s = zero ? NaN : m3 / std::pow(m2, 1.5);
            if (!bias && !zero && n > 2) s = std::pow((N - 1.0) * N, 0.5) / (N - 2.0) * m3 / std::pow(m2, 1.5);
            const double m22 = L.ond == 0 ? std::pow(m2, 2.0) : m2 * m2;    /* scalar ** 2.0 is pow */
            double k = zero ? NaN : m4 / m22;
            if (!bias && !zero && n > 3) k = 1.0 / (N - 2) / (N - 3) * ((N * N - 1.0) * m4 / m22 - 3 * std::pow(N - 1, 2.0)) + 3.0;
            sk[o] = s;
            ku[o] = k - 3;
        } else {
            /* numpy.ma: masked NaNs, sums over the filled (0) data */
            int64_t cnt = 0;
            double lo = NaN, hi = NaN;
            for (int64_t i = 0; i < n; i++) {
                if (isnan_(x[i])) { f[i] = 0; continue; }
                f[i] = x[i];
                if (cnt == 0 || x[i] < lo) lo = x[i];
                if (cnt == 0 || x[i] > hi) hi = x[i];
                cnt++;
            }
            nobs[o] = (double)cnt;
            mn[o] = lo; mx[o] = hi;
            const double C = (double)cnt;
            const double m = fn_sum(f.data(), n, fl) * 1. / C;
            mean[o] = m;
            for (int64_t i = 0; i < n; i++) { const double d = isnan_(x[i]) ? 0 : x[i] - m; azm[i] = d; tmp[i] = d * d; }
            const double ss = fn_sum(tmp.data(), n, fl);
            var[o] = ss / (C - ddof);
            const double m2 = ss / C;
            for (int64_t i = 0; i < n; i++) tmp[i] = azm[i] * azm[i] * azm[i];
            const double m3 = fn_sum(tmp.data(), n, fl) / C;
            for (int64_t i = 0; i < n; i++) { const double q = azm[i] * azm[i]; tmp[i] = q * q; }
            const double m4 = fn_sum(tmp.data(), n, fl) / C;
            const double e = 1e-15 * m;
            const bool zero = m2 <= e * e;
            double s = zero ? 0 : m3 / std::pow(m2, 1.5);
            if (!bias && !zero && cnt > 2) s = std::sqrt((C - 1.0) * C) / (C - 2.0) * m3 / std::pow(m2, 1.5);
            const double m22 = L.ond == 0 ? std::pow(m2, 2.0) : m2 * m2;
            double k = zero ? 0 : m4 / m22;
            if (!bias && !zero && cnt > 3) k = 1.0 / (C - 2) / (C - 3) * ((C * C - 1.0) * m4 / m22 - 3 * std::pow(C - 1, 2.0)) + 3.0;
            sk[o] = s;
            ku[o] = k - 3;
        }
    }
    if (!omit) {
        fn_result_int(&res[0], n);
    } else if ((rc = emit(&res[0], L, nobs.data(), false, true)) != TSR_OK) return rc;
    /* minmax: (min, max) stacked on a new first axis */
    {
        const bool as_int = L.is_int;
        int nd = L.ond + 1;
        int64_t shp[TSR_MAXDIM];
        shp[0] = 2;
        for (int i = 0; i < L.ond; i++) shp[i + 1] = L.oshape[i];
        vector<double> mm(2 * nl);
        for (int64_t o = 0; o < nl; o++) { mm[o] = mn[o]; mm[nl + o] = mx[o]; }
        if ((rc = emit_arr(&res[1], nd, shp, mm.data(), as_int && !omit)) != TSR_OK) return rc;
    }
    if ((rc = emit(&res[2], L, mean.data(), false, false)) != TSR_OK) return rc;
    if ((rc = emit(&res[3], L, var.data(), false, false)) != TSR_OK) return rc;
    if ((rc = emit(&res[4], L, sk.data(), false, false)) != TSR_OK) return rc;
    return emit(&res[5], L, ku.data(), false, false);
}

/* ================================================================ mode */

int r_mode(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "a");
    int pol;
    if (nan_policy_of(args, nargs, 2, &pol) != TSR_OK) return TSR_EARG;
    const bool keep = arg_bool(args, nargs, 3, false);
    Lanes L;
    int rc = make_lanes(A, args, nargs, 1, L, false);
    if (rc != TSR_OK) return rc;
    const bool hasnan = any_nan(A.v.data(), A.size());
    if (hasnan && pol == NP_RAISE) { fn_set_error("The input contains nan values"); return TSR_EARG; }
    const bool nd_path = L.axis >= 0 && L.ndim_in > 1;
    bool as_float = nd_path && (A.size() == 0 || (hasnan && pol == NP_OMIT));
    vector<double> md(L.nl + 1), ct(L.nl + 1);
    for (int64_t o = 0; o < L.nl; o++) {
        double *a = L.lane(o);
        int64_t m = L.len;
        if (pol == NP_OMIT) nan_slice(a, m, pol);
        if (m == 0) { md[o] = NaN; ct[o] = 0; if (!nd_path) as_float = true; continue; }
        sort_nan_last(a, m);
        double best = a[0];
        int64_t bestc = 0, i = 0;
        while (i < m) {
            int64_t j = i + 1;
            if (isnan_(a[i])) j = m;                      /* NaNs are grouped */
            else while (j < m && a[j] == a[i]) j++;
            if (j - i > bestc) { bestc = j - i; best = a[i]; }
            i = j;
        }
        md[o] = best;
        ct[o] = (double)bestc;
    }
    if ((rc = emit(&res[0], L, md.data(), keep, L.is_int && !as_float)) != TSR_OK) return rc;
    return emit(&res[1], L, ct.data(), keep, !as_float);
}

/* ================================================================ trimmed statistics */

enum { T_MEAN, T_VAR, T_STD, T_SEM, T_MIN, T_MAX };
const int K_TMEAN = T_MEAN, K_TVAR = T_VAR, K_TSTD = T_STD, K_TSEM = T_SEM, K_TMIN = T_MIN, K_TMAX = T_MAX;

int r_trimmed(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    const int kind = *(const int *)ctx;
    NEED_ARRAY(0, A, "a");
    double lo = NaN, hi = NaN;
    bool lo_inc = true, hi_inc = true;
    int kaxis, kddof = -1, knan, kkeep;
    if (kind == T_MIN || kind == T_MAX) {
        const double lim = arg_num(args, nargs, 1, NaN);
        const bool inc = arg_bool(args, nargs, 3, true);
        if (kind == T_MIN) { lo = lim; lo_inc = inc; } else { hi = lim; hi_inc = inc; }
        kaxis = 2; knan = 4; kkeep = 5;
    } else {
        if (!is_null(args, nargs, 1)) {
            Arr lim;
            if (!get_arr(args, nargs, 1, lim) || lim.size() != 2) { fn_set_error("limits must be a pair (lower, upper)"); return TSR_EARG; }
            lo = lim.v[0]; hi = lim.v[1];
        }
        if (!is_null(args, nargs, 2)) {
            Arr inc;
            if (!get_arr(args, nargs, 2, inc) || inc.size() != 2) { fn_set_error("inclusive must be a pair of booleans"); return TSR_EARG; }
            lo_inc = inc.v[0] != 0; hi_inc = inc.v[1] != 0;
        }
        kaxis = 3;
        if (kind == T_MEAN) { knan = 4; kkeep = 5; }
        else { kddof = 4; knan = 5; kkeep = 6; }
    }
    const double ddof = kddof >= 0 ? arg_num(args, nargs, kddof, 1) : 0;
    int pol;
    if (nan_policy_of(args, nargs, knan, &pol) != TSR_OK) return TSR_EARG;
    const bool keep = arg_bool(args, nargs, kkeep, false);
    Lanes L;
    int rc = make_lanes(A, args, nargs, kaxis, L, kind == T_MEAN);
    if (rc != TSR_OK) return rc;
    auto masked = [&](double v) {
        bool m = false;
        if (!isnan_(lo)) m = m || (lo_inc ? v < lo : v <= lo);
        if (!isnan_(hi)) m = m || (hi_inc ? v > hi : v >= hi);
        return m;
    };
    const bool hasnan = any_nan(A.v.data(), A.size());
    if (hasnan && pol == NP_RAISE) { fn_set_error("The input contains nan values"); return TSR_EARG; }
    if (!hasnan && A.size() > 0) {
        bool all = true;
        for (double v : A.v) if (!masked(v)) { all = false; break; }
        if (all) { fn_set_error("No array values within given limits"); return TSR_EARG; }
    }
    const int64_t nl = L.nl;
    vector<double> out(nl > 0 ? nl : 1);
    vector<double> t(L.len + 8), u(L.len + 8), d(L.len + 8);
    bool invalid_any = false;
    for (int64_t o = 0; o < nl; o++) {
        double *x = L.lane(o);
        int64_t n = L.len;
        if (hasnan) {
            const int r = nan_slice(x, n, pol);
            if (r == 1) { out[o] = NaN; continue; }
            if (n > 0) {
                bool all = true;
                for (int64_t i = 0; i < n; i++) if (!masked(x[i])) { all = false; break; }
                if (all) { fn_set_error("No array values within given limits"); return TSR_EARG; }
            }
        }
        if (n == 0) { out[o] = NaN; continue; }
        const unsigned fl = hasnan ? 0u : L.flags;
        switch (kind) {
        case T_MEAN: {
            double cnt = 0;
            for (int64_t i = 0; i < n; i++) {
                const bool m = masked(x[i]);
                t[i] = m ? 0.0 : x[i];
                u[i] = m ? 0.0 : 1.0;
            }
            const double s = fn_sum(t.data(), n, fl);
            cnt = fn_sum(u.data(), n, fl);
            out[o] = cnt != 0 ? s / cnt : NaN;
            break;
        }
        case T_VAR: case T_STD: case T_SEM: {
            for (int64_t i = 0; i < n; i++) if (masked(x[i])) x[i] = NaN;
            const double v = xpm_var(x, n, ddof, true, fl, t.data(), d.data());
            if (kind == T_VAR) out[o] = v;
            else if (kind == T_STD) out[o] = std::pow(v, 0.5);
            else {
                const double sd = std::pow(v, 0.5);
                const double nobs = (double)(n - count_nan(x, n));
                out[o] = sd / std::pow(nobs, 0.5);
            }
            break;
        }
        default: {
            bool allm = true;
            double best = 0;
            bool first = true;
            for (int64_t i = 0; i < n; i++) {
                double v = x[i];
                if (masked(v)) v = kind == T_MIN ? (L.is_int ? 9223372036854775807.0 : INF) : (L.is_int ? -9223372036854775808.0 : -INF);
                else allm = false;
                if (first || (kind == T_MIN ? v < best : v > best)) best = v;
                first = false;
            }
            if (allm) { out[o] = NaN; invalid_any = true; }
            else out[o] = best;
            break;
        }
        }
    }
    const bool as_int = (kind == T_MIN || kind == T_MAX) && L.is_int && !invalid_any && !hasnan;
    return emit(&res[0], L, out.data(), keep, as_int);
}

/* ================================================================ trimboth, trim1 */

int r_trim(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    const bool one = ctx != nullptr;
    NEED_ARRAY(0, A, "a");
    const double prop = arg_num(args, nargs, 1, NaN);
    const int kaxis = one ? 3 : 2;
    const bool as_int = is_int_dtype(A.dtype);
    if (!one && A.size() == 0) {
        int64_t shp[TSR_MAXDIM];
        for (int i = 0; i < A.ndim; i++) shp[i] = A.shape[i];
        return emit_arr(&res[0], A.ndim, shp, A.v.data(), as_int);
    }
    Lanes L;
    int rc = make_lanes(A, args, nargs, kaxis, L, false);
    if (rc != TSR_OK) return rc;
    const int64_t nobs = L.len;
    int64_t lowercut, uppercut;
    if (!(std::fabs(prop * (double)nobs) < 9.2e18)) { fn_set_error("cannot convert float %s to integer", prop != prop ? "NaN" : "infinity"); return TSR_EARG; }
    if (!one) {
        lowercut = (int64_t)(prop * (double)nobs);
        uppercut = nobs - lowercut;
        if (lowercut >= uppercut) { fn_set_error("Proportion too big."); return TSR_EARG; }
        if (lowercut < 0) { fn_set_error("kth(=%lld) out of bounds (%lld)", (long long)(uppercut - 1), (long long)nobs); return TSR_EARG; }
    } else {
        if (prop >= 1) { const int64_t z = 0; return fn_result_array(&res[0], TSR_F64, 1, &z) ? TSR_OK : TSR_ENOMEM; }
        const char *tail = arg_str(args, nargs, 2, "right");
        std::string t(tail);
        for (auto &c : t) c = (char)tolower(c);
        if (t == "right") { lowercut = 0; uppercut = nobs - (int64_t)(prop * (double)nobs); }
        else if (t == "left") { lowercut = (int64_t)(prop * (double)nobs); uppercut = nobs; }
        else { fn_set_error("tail must be 'right' or 'left'"); return TSR_EARG; }
        /* np.partition's kth may be negative down to -nobs; atmp[lowercut:uppercut] then slices from the end */
        const int64_t kth = t == "right" ? uppercut - 1 : lowercut;
        if (kth < -nobs || kth >= nobs) { fn_set_error("kth(=%lld) out of bounds (%lld)", (long long)kth, (long long)nobs); return TSR_EARG; }
        if (lowercut < 0) lowercut += nobs;
        if (uppercut < lowercut) uppercut = lowercut;
    }
    const int64_t k = uppercut - lowercut;
    vector<double> lanes((size_t)(L.nl * k));
    for (int64_t o = 0; o < L.nl; o++) {
        double *x = L.lane(o);
        sort_nan_last(x, nobs);
        for (int64_t t = 0; t < k; t++) lanes[o * k + t] = x[lowercut + t];
    }
    int nd;
    int64_t shp[TSR_MAXDIM];
    vector<double> out;
    unlanes(L, k, lanes.data(), &nd, shp, out);
    return emit_arr(&res[0], nd, shp, out.data(), as_int);
}

/* ================================================================ sigmaclip */

int r_sigmaclip(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "a");
    const double low = arg_num(args, nargs, 1, 4.0), high = arg_num(args, nargs, 2, 4.0);
    vector<double> c = A.v, tmp(c.size() + 8);
    double critlower = NaN, critupper = NaN;
    int64_t delta = 1;
    while (delta) {
        const int64_t size = (int64_t)c.size();
        const double c_std = std::sqrt(np_var(c.data(), size, 0, 0, tmp.data()));
        const double c_mean = tsr_psum(c.data(), size) / (double)size;
        critlower = c_mean - c_std * low;
        critupper = c_mean + c_std * high;
        vector<double> keep;
        for (double v : c) if (v >= critlower && v <= critupper) keep.push_back(v);
        delta = size - (int64_t)keep.size();
        c.swap(keep);
    }
    const int64_t n = (int64_t)c.size();
    int rc = emit_arr(&res[0], 1, &n, c.data(), is_int_dtype(A.dtype));
    if (rc != TSR_OK) return rc;
    fn_result_num(&res[1], critlower);
    fn_result_num(&res[2], critupper);
    return TSR_OK;
}

/* ================================================================ scoreatpercentile, percentileofscore */

int r_scoreatpercentile(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "a");
    Arr P;
    if (!get_arr(args, nargs, 1, P)) { fn_set_error("per must be a number or an array"); return TSR_EARG; }
    const bool per_scalar = args[1].kind == 1;
    const char *meth = arg_str(args, nargs, 3, "fraction");
    if (A.size() == 0) {
        if (per_scalar) { fn_result_num(&res[0], NaN); return TSR_OK; }
        vector<double> v(P.size() > 0 ? P.size() : 1, NaN);
        return emit_arr(&res[0], P.ndim, P.shape, v.data(), false);
    }
    Arr B = A;
    if (!is_null(args, nargs, 2)) {
        Arr lim;
        if (!get_arr(args, nargs, 2, lim)) { fn_set_error("limit must be a pair"); return TSR_EARG; }
        if (lim.size() > 0) {
            if (lim.size() < 2) { fn_set_error("limit must be a pair"); return TSR_EARG; }
            vector<double> kept;
            for (double v : A.v) if (lim.v[0] <= v && v <= lim.v[1]) kept.push_back(v);
            B.v = kept;
            B.ndim = 1;
            B.shape[0] = (int64_t)kept.size();
            B.strides[0] = 8;
        }
    }
    Lanes L;
    int rc = make_lanes(B, args, nargs, 4, L, true);
    if (rc != TSR_OK) return rc;
    const int64_t np_ = P.size(), nl = L.nl, n = L.len;
    for (int64_t o = 0; o < nl; o++) sort_nan_last(L.lane(o), n);
    vector<double> out((size_t)(np_ * nl + 1));
    for (int64_t q = 0; q < np_; q++) {
        const double per = P.v[q];
        if (!(0 <= per && per <= 100)) { fn_set_error("percentile must be in the range [0, 100]"); return TSR_EARG; }
        double idx = per / 100. * (double)(n - 1);
        if ((double)(int64_t)idx != idx) {
            if (!strcmp(meth, "lower")) idx = std::floor(idx);
            else if (!strcmp(meth, "higher")) idx = std::ceil(idx);
            else if (strcmp(meth, "fraction") != 0) {
                fn_set_error("interpolation_method can only be 'fraction', 'lower' or 'higher'");
                return TSR_EARG;
            }
        }
        const int64_t i = (int64_t)idx;
        for (int64_t o = 0; o < nl; o++) {
            const double *s = L.lane(o);
            double v;
            if ((double)i == idx) {
                v = n == 0 ? 0.0 : s[i] * 1 / 1.0;            /* an empty slice sums to 0 */
            } else if (n == 0) {
                fn_set_error("operands could not be broadcast together with shapes (0,) (2,)");
                return TSR_EARG;
            } else {
                const double j = (double)(i + 1);
                const double w0 = j - idx, w1 = idx - (double)i;
                v = (s[i] * w0 + s[i + 1] * w1) / (w0 + w1);
            }
            out[q * nl + o] = v;
        }
    }
    if (per_scalar) return emit(&res[0], L, out.data(), false, false);
    int nd = 1 + L.ond;
    int64_t shp[TSR_MAXDIM];
    shp[0] = np_;
    for (int i = 0; i < L.ond; i++) shp[i + 1] = L.oshape[i];
    return emit_arr(&res[0], nd, shp, out.data(), false);
}

int r_percentileofscore(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "a");
    Arr S;
    if (!get_arr(args, nargs, 1, S)) { fn_set_error("score must be a number or an array"); return TSR_EARG; }
    const char *kind = arg_str(args, nargs, 2, "rank");
    int pol;
    if (nan_policy_of(args, nargs, 3, &pol) != TSR_OK) return TSR_EARG;
    if (A.ndim != 1) { fn_set_error("`a` must be 1-dimensional."); return TSR_EARG; }
    int64_t n = A.size();
    const bool cna = any_nan(A.v.data(), n), cns = any_nan(S.v.data(), S.size());
    if ((cna || cns) && pol == NP_RAISE) { fn_set_error("The input contains nan values"); return TSR_EARG; }
    vector<double> a;
    for (double v : A.v) if (!isnan_(v)) a.push_back(v);
    if (cna) {
        if (pol == NP_OMIT) n = (int64_t)a.size();
        if (pol == NP_PROPAGATE) n = 0;
    }
    const int64_t ns = S.size();
    vector<double> out(ns + 1, NaN);
    if (n != 0) {
        int k;
        if (!strcmp(kind, "rank")) k = 0;
        else if (!strcmp(kind, "strict")) k = 1;
        else if (!strcmp(kind, "weak")) k = 2;
        else if (!strcmp(kind, "mean")) k = 3;
        else { fn_set_error("kind can only be 'rank', 'strict', 'weak' or 'mean'"); return TSR_EARG; }
        const double N = (double)n;
        for (int64_t q = 0; q < ns; q++) {
            const double s = S.v[q];
            if (isnan_(s)) continue;
            int64_t left = 0, right = 0;
            for (double v : a) { left += v < s; right += v <= s; }
            double p;
            switch (k) {
            case 0: p = (double)(left + right + (left < right ? 1 : 0)) * (50.0 / N); break;
            case 1: p = (double)left * (100.0 / N); break;
            case 2: p = (double)right * (100.0 / N); break;
            default: p = (double)(left + right) * (50.0 / N); break;
            }
            out[q] = p;
        }
    }
    if (S.ndim == 0) { fn_result_num(&res[0], out[0]); return TSR_OK; }
    return emit_arr(&res[0], S.ndim, S.shape, out.data(), false);
}

/* ================================================================ rankdata, tiecorrect */

int r_rankdata(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "a");
    const char *method = arg_str(args, nargs, 1, "average");
    int mth;
    if (!strcmp(method, "average")) mth = 0;
    else if (!strcmp(method, "min")) mth = 1;
    else if (!strcmp(method, "max")) mth = 2;
    else if (!strcmp(method, "dense")) mth = 3;
    else if (!strcmp(method, "ordinal")) mth = 4;
    else { fn_set_error("unknown method \"%s\"", method); return TSR_EARG; }
    int pol;
    if (nan_policy_of(args, nargs, 3, &pol) != TSR_OK) return TSR_EARG;
    Lanes L;
    int rc = make_lanes(A, args, nargs, 2, L, true);
    if (rc != TSR_OK) return rc;
    const bool as_int_m = mth != 0;
    if (A.size() == 0) {
        int nd;
        int64_t shp[TSR_MAXDIM];
        vector<double> out;
        unlanes(L, L.len, L.buf.data(), &nd, shp, out);
        if (L.axis < 0) { nd = 1; shp[0] = 0; }
        return emit_arr(&res[0], nd, shp, out.data(), as_int_m);
    }
    const bool cn = any_nan(A.v.data(), A.size());
    if (cn && pol == NP_RAISE) { fn_set_error("The input contains nan values"); return TSR_EARG; }
    const int64_t n = L.len;
    vector<double> ranks((size_t)(L.nl * n));
    vector<int64_t> j((size_t)n);
    for (int64_t o = 0; o < L.nl; o++) {
        const double *x = L.lane(o);
        std::iota(j.begin(), j.end(), 0);
        std::stable_sort(j.begin(), j.end(), [&](int64_t a, int64_t b) { return lt_nan(x[a], x[b]); });
        double *r = ranks.data() + o * n;
        int64_t k = 0, dense = 0;
        while (k < n) {
            int64_t e = k + 1;
            while (e < n && !(x[j[e]] != x[j[e - 1]])) e++;
            dense++;
            const int64_t cnt = e - k;
            for (int64_t t = k; t < e; t++) {
                double v;
                switch (mth) {
                case 0: v = (double)(k + 1) + ((double)cnt - 1) / 2; break;
                case 1: v = (double)(k + 1); break;
                case 2: v = (double)(k + 1 + cnt - 1); break;
                case 3: v = (double)dense; break;
                default: v = (double)(t + 1); break;
                }
                r[j[t]] = v;
            }
            k = e;
        }
        if (cn) {
            const bool lane_nan = any_nan(x, n);
            for (int64_t t = 0; t < n; t++)
                if (pol == NP_OMIT ? isnan_(x[t]) : lane_nan) r[t] = NaN;
        }
    }
    int nd;
    int64_t shp[TSR_MAXDIM];
    vector<double> out;
    unlanes(L, n, ranks.data(), &nd, shp, out);
    return emit_arr(&res[0], nd, shp, out.data(), as_int_m && !cn);
}

int r_tiecorrect(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "rankvals");
    if (A.ndim != 1) { fn_set_error("rankvals must be 1-dimensional"); return TSR_EARG; }
    vector<double> arr = A.v;
    const int64_t n = (int64_t)arr.size();
    if (n < 2) { fn_result_num(&res[0], 1.0); return TSR_OK; }
    sort_nan_last(arr.data(), n);
    vector<double> t;
    int64_t k = 0;
    while (k < n) {
        int64_t e = k + 1;
        while (e < n && !(arr[e] != arr[e - 1])) e++;
        const double c = (double)(e - k);
        t.push_back(std::pow(c, 3.0) - c);
        k = e;
    }
    const double size = (double)n;
    fn_result_num(&res[0], 1.0 - tsr_psum(t.data(), (int64_t)t.size()) / (std::pow(size, 3.0) - size));
    return TSR_OK;
}

/* ================================================================ cumfreq, relfreq (scipy's _histogram over np.histogram) */

int r_freq(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    const bool rel = ctx != nullptr;
    NEED_ARRAY(0, A, "a");
    const double nbd = arg_num(args, nargs, 1, 10);
    if (nbd != std::floor(nbd) || nbd < 1) { fn_set_error("`bins` must be positive, when an integer"); return TSR_EARG; }
    if (nbd > 1e8) { fn_set_error("Unable to allocate memory for %.0f bins", nbd); return TSR_ENOMEM; }   /* np.histogram's MemoryError, no UB */
    const int64_t nb = (int64_t)nbd;
    const vector<double> &a = A.v;
    const int64_t n = (int64_t)a.size();
    double first, last;
    bool lim_int = false;
    if (is_null(args, nargs, 2)) {
        if (n == 0) { first = 0; last = 1; lim_int = true; }
        else {
            double mn = a[0], mx = a[0];
            for (int64_t i = 1; i < n; i++) {
                if (isnan_(mn)) break;
                if (isnan_(a[i])) { mn = mx = a[i]; break; }
                if (a[i] < mn) mn = a[i];
                if (a[i] > mx) mx = a[i];
            }
            const double s = (mx - mn) / (2. * ((double)nb - 1.));
            first = mn - s;
            last = mx + s;
        }
    } else {
        Arr lim;
        if (!get_arr(args, nargs, 2, lim) || lim.size() != 2) { fn_set_error("defaultreallimits must be a pair"); return TSR_EARG; }
        first = lim.v[0];
        last = lim.v[1];
        lim_int = is_int_dtype(lim.dtype);
    }
    const double lower = first, upper = last;
    vector<double> w;
    const bool weighted = !is_null(args, nargs, 3);
    if (weighted) {
        Arr W;
        if (!get_arr(args, nargs, 3, W)) { fn_set_error("weights must be an array"); return TSR_EARG; }
        if (W.size() != n || W.ndim != 1) { fn_set_error("weights should have the same shape as a."); return TSR_EARG; }
        w = W.v;
    }
    /* np.histogram: range checks, equal edges widened by 0.5 */
    double fe = first, le = last;
    if (fe > le) { fn_set_error("max must be larger than min in range parameter."); return TSR_EARG; }
    if (!std::isfinite(fe) || !std::isfinite(le)) { fn_set_error("supplied range of [%g, %g] is not finite", fe, le); return TSR_EARG; }
    if (fe == le) { fe = fe - 0.5; le = le + 0.5; }
    vector<double> edges((size_t)nb + 1);
    {
        const double delta = le - fe, step = delta / (double)nb;
        for (int64_t i = 0; i <= nb; i++) edges[i] = (step == 0 ? ((double)i / (double)nb) * delta : (double)i * step) + fe;
        edges[nb] = le;
    }
    vector<double> hist((size_t)nb, 0.0);
    {
        const int64_t BLOCK = 65536;
        const double norm_denom = le - fe, norm_num = (double)nb;
        vector<double> blk((size_t)nb);
        for (int64_t b0 = 0; b0 < n; b0 += BLOCK) {
            const int64_t b1 = std::min(n, b0 + BLOCK);
            std::fill(blk.begin(), blk.end(), 0.0);
            for (int64_t i = b0; i < b1; i++) {
                const double v = a[i];
                if (!(v >= fe) || !(v <= le)) continue;
                const double f = ((v - fe) / norm_denom) * norm_num;
                int64_t idx = (int64_t)f;
                if (idx == nb) idx -= 1;
                if (v < edges[idx]) idx -= 1;
                if (v >= edges[idx + 1] && idx != nb - 1) idx += 1;
                if (weighted) blk[idx] += w[i];
                else blk[idx] += 1;
            }
            for (int64_t i = 0; i < nb; i++) hist[i] += blk[i];
        }
    }
    const double binsize = edges[1] - edges[0];
    int64_t extra = 0;
    for (double v : a) if (lower > v || v > upper) extra++;
    if (rel) {
        if (A.ndim == 0) { fn_set_error("tuple index out of range"); return TSR_EINDEX; }
        const double d0 = (double)A.shape[0];
        for (auto &h : hist) h = h / d0;
    } else {
        for (int64_t i = 1; i < nb; i++) hist[i] = hist[i - 1] + hist[i];
    }
    int rc = emit_arr(&res[0], 1, &nb, hist.data(), false);
    if (rc != TSR_OK) return rc;
    if (lim_int) fn_result_int(&res[1], (int64_t)lower);
    else fn_result_num(&res[1], lower);
    fn_result_num(&res[2], binsize);
    fn_result_int(&res[3], extra);
    return TSR_OK;
}

/* ================================================================ obrientransform */

int r_obrien(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    vector<vector<double>> arrays;
    const double TINY = std::sqrt(EPS);
    for (int k = 0; k < nargs; k++) {
        if (args[k].kind == 0) continue;
        Arr A;
        if (!get_arr(args, nargs, k, A) || A.ndim != 1) { fn_set_error("each sample must be a 1-D array"); return TSR_EARG; }
        const int64_t n = A.size();
        const double N = (double)n;
        const double mu = tsr_psum(A.v.data(), n) / N;
        vector<double> sq(n + 1);
        for (int64_t i = 0; i < n; i++) { const double d = A.v[i] - mu; sq[i] = d * d; }
        const double sumsq = tsr_psum(sq.data(), n);
        vector<double> t(n + 1);
        for (int64_t i = 0; i < n; i++) t[i] = ((N - 1.5) * N * sq[i] - 0.5 * sumsq) / ((N - 1) * (N - 2));
        const double var = sumsq / (N - 1);
        const double mt = tsr_psum(t.data(), n) / N;
        if (std::fabs(var - mt) > TINY) { fn_set_error("Lack of convergence in obrientransform."); return TSR_EARG; }
        t.resize(n);
        arrays.push_back(t);
    }
    if (arrays.empty()) { fn_set_error("at least one sample is required"); return TSR_EARG; }
    const int64_t len = (int64_t)arrays[0].size();
    for (auto &v : arrays)
        if ((int64_t)v.size() != len) { fn_set_error("samples of different lengths (an object array) are not supported"); return TSR_EARG; }
    int64_t shp[2] = {(int64_t)arrays.size(), len};
    vector<double> flat;
    for (auto &v : arrays) flat.insert(flat.end(), v.begin(), v.end());
    return emit_arr(&res[0], 2, shp, flat.data(), false);
}

/* ================================================================ lmoment */

int r_lmoment(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "sample");
    vector<double> order;
    bool order_scalar = false;
    if (is_null(args, nargs, 1)) order = {1, 2, 3, 4};
    else {
        Arr O;
        if (!get_arr(args, nargs, 1, O)) { fn_set_error("`order` must be a scalar or a non-empty array of positive integers."); return TSR_EARG; }
        order_scalar = O.ndim == 0;
        bool bad = O.size() == 0 || O.ndim > 1 || (O.ndim > 0 && !is_int_dtype(O.dtype));
        for (double v : O.v) if (!(v > 0) || v != std::floor(v)) bad = true;
        if (bad) { fn_set_error("`order` must be a scalar or a non-empty array of positive integers."); return TSR_EARG; }
        order = O.v;
    }
    const bool sorted = arg_bool(args, nargs, 3, false);
    const bool standardize = arg_bool(args, nargs, 4, true);
    int pol;
    if (nan_policy_of(args, nargs, 5, &pol) != TSR_OK) return TSR_EARG;
    const bool keep = arg_bool(args, nargs, 6, false);
    Lanes L;
    int rc = make_lanes(A, args, nargs, 2, L, false);
    if (rc != TSR_OK) return rc;
    const int64_t no = (int64_t)order.size(), nl = L.nl;
    /* SciPy allocates max(order) moments (numpy arrays of that length): an order past what fits is its
       MemoryError, never a double -> int overflow here */
    double nmax = 0;
    for (double v : order) nmax = std::max(nmax, v);
    if (nmax > 100000000.0) { fn_set_error("Unable to allocate memory for %.0f L-moments", nmax); return TSR_ENOMEM; }
    int nm = (int)nmax;
    vector<double> out((size_t)(no * nl + 1));
    for (int64_t o = 0; o < nl; o++) {
        double *x = L.lane(o);
        int64_t n = L.len;
        const int r = nan_slice(x, n, pol);
        if (r < 0) return TSR_EARG;
        if (r == 1 || n == 0) { for (int64_t q = 0; q < no; q++) out[q * nl + o] = NaN; continue; }
        if (!sorted) sort_nan_last(x, n);
        const double N = (double)n;
        vector<double> bk(nm), lm(nm);
        for (int rr = 0; rr < nm; rr++) {
            if (rr >= n) { bk[rr] = 0; continue; }
            vector<double> bj(n);
            for (int64_t j = 0; j < n; j++) bj[j] = sc::binom((double)j, (double)rr);
            bk[rr] = blas_dot(bj.data(), x, n) / sc::binom(N - 1, (double)rr) / N;
        }
        for (int rr = 0; rr < nm; rr++) {
            vector<double> prk(nm);
            for (int k = 0; k < nm; k++)
                prk[k] = std::pow(-1.0, (double)(rr - k)) * sc::binom((double)rr, (double)k) * sc::binom((double)(rr + k), (double)k);
            lm[rr] = blas_dot(prk.data(), bk.data(), nm);
        }
        if (standardize && nm > 2) for (int rr = 2; rr < nm; rr++) lm[rr] = lm[rr] / lm[1];
        for (int rr = (int)std::min<int64_t>(n, nm); rr < nm; rr++) lm[rr] = NaN;
        for (int64_t q = 0; q < no; q++) out[q * nl + o] = lm[(int)order[q] - 1];
    }
    if (order_scalar) return emit(&res[0], L, out.data(), keep, false);
    /* stacked: (len(order), lane shape) */
    int nd = 0;
    int64_t shp[TSR_MAXDIM];
    shp[nd++] = no;
    if (keep) {
        if (L.axis < 0) for (int i = 0; i < std::max(1, L.ndim_in); i++) shp[nd++] = 1;
        else for (int i = 0; i < L.ndim_in; i++) shp[nd++] = i == L.axis ? 1 : L.shape_in[i];
    } else for (int i = 0; i < L.ond; i++) shp[nd++] = L.oshape[i];
    return emit_arr(&res[0], nd, shp, out.data(), false);
}

/* ================================================================ expectile */

double np_average(const vector<double> &v, const vector<double> *w)
{
    const int64_t n = (int64_t)v.size();
    if (!w) return tsr_psum(v.data(), n) / (double)n;
    vector<double> ww = *w, p(n);
    const double scl = tsr_psum(ww.data(), n);
    for (int64_t i = 0; i < n; i++) p[i] = v[i] * (*w)[i];
    return tsr_psum(p.data(), n) / scl;
}

int r_expectile(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "a");
    const double alpha = arg_num(args, nargs, 1, 0.5);
    if (alpha < 0 || alpha > 1) { fn_set_error("The expectile level alpha must be in the range [0, 1]."); return TSR_EARG; }
    const vector<double> &a = A.v;
    const int64_t n = (int64_t)a.size();
    vector<double> w;
    const bool weighted = !is_null(args, nargs, 2);
    if (weighted) {
        Arr W;
        if (!get_arr(args, nargs, 2, W)) { fn_set_error("weights must be an array"); return TSR_EARG; }
        if (W.size() == 1) w.assign(n, W.v[0]);
        else if (W.size() == n) w = W.v;
        else { fn_set_error("weights cannot be broadcast to the shape of a"); return TSR_EARG; }
        double s = tsr_psum(w.data(), n);
        if (s == 0.0) { fn_set_error("Weights sum to zero, can't be normalized"); return TSR_EARG; }
    }
    if (n == 0) { fn_set_error("zero-size array to reduction operation maximum which has no identity"); return TSR_EARG; }
    const vector<double> *wp = weighted ? &w : nullptr;
    vector<double> tmp(n);
    auto f = [&](double t) {
        for (int64_t i = 0; i < n; i++) tmp[i] = std::fabs((double)(a[i] <= t) - alpha) * (t - a[i]);
        return np_average(tmp, wp);
    };
    double amax = a[0], amin = a[0];
    for (double v : a) {
        if (isnan_(v) || isnan_(amax)) { amax = amin = NaN; continue; }
        amax = std::max(amax, v);
        amin = std::min(amin, v);
    }
    double x0, x1;
    if (alpha >= 0.5) { x0 = np_average(a, wp); x1 = amax; }
    else { x1 = np_average(a, wp); x0 = amin; }
    if (x0 == x1) { fn_result_num(&res[0], x0); return TSR_OK; }
    /* scipy.optimize.newton, secant branch (tol 1.48e-8, maxiter 50, rtol 0) */
    const double tol = 1.48e-8;
    double p0 = x0 * 1.0, p1 = x1;
    double q0 = f(p0), q1 = f(p1);
    if (std::fabs(q1) < std::fabs(q0)) { std::swap(p0, p1); std::swap(q0, q1); }
    double p = NaN;
    for (int itr = 0; itr < 50; itr++) {
        if (q1 == q0) { p = (p1 + p0) / 2.0; fn_result_num(&res[0], p); return TSR_OK; }
        if (std::fabs(q1) > std::fabs(q0)) p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1);
        else p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0);
        if (std::fabs(p - p1) <= tol || p == p1) { fn_result_num(&res[0], p); return TSR_OK; }
        p0 = p1; q0 = q1;
        p1 = p;
        q1 = f(p1);
    }
    fn_result_num(&res[0], p);
    return TSR_OK;
}

/* ================================================================ iqr, median_abs_deviation */

/* scipy.stats.quantile of a sorted NaN-free slice (no weights) */
int quantile_sorted(const double *y, int64_t n, double p, const std::string &m, double *out)
{
    const double N = (double)n;
    if (m == "_lower" || m == "_higher" || m == "_nearest" || m == "_midpoint") {
        const double ij = p * (N - 1);
        if (m == "_midpoint") { *out = (y[(int64_t)std::floor(ij)] + y[(int64_t)std::ceil(ij)]) / 2; return TSR_OK; }
        const double k = m == "_lower" ? std::floor(ij) : (m == "_higher" ? std::ceil(ij) : std::nearbyint(ij));
        *out = y[(int64_t)k];
        return TSR_OK;
    }
    double mm;
    if (m == "inverted_cdf" || m == "averaged_inverted_cdf" || m == "interpolated_inverted_cdf") mm = 0;
    else if (m == "closest_observation") mm = -0.5;
    else if (m == "hazen") mm = 0.5;
    else if (m == "weibull") mm = p;
    else if (m == "linear") mm = 1 - p;
    else if (m == "median_unbiased") mm = p / 3 + 1.0 / 3;
    else if (m == "normal_unbiased") mm = p / 4 + 3.0 / 8;
    else { fn_set_error("`method` must be one of the supported quantile methods"); return TSR_EARG; }
    const double jg = p * N + mm;
    double jp1 = py_floordiv(jg, 1);
    double j = jp1 - 1;
    double g = py_mod(jg, 1);
    if (m == "inverted_cdf") g = g > 0 ? 1.0 : 0.0;
    else if (m == "averaged_inverted_cdf") g = (1 + (g > 0 ? 1.0 : 0.0)) / 2;
    else if (m == "closest_observation") g = 1 - ((g == 0 && py_mod(j, 2) == 1) ? 1.0 : 0.0);
    if ((m == "inverted_cdf" || m == "averaged_inverted_cdf" || m == "closest_observation") && jg < 0) g = 0;
    if (j < 0) g = 0;
    j = std::min(std::max(j, 0.), N - 1);
    jp1 = std::min(std::max(jp1, 0.), N - 1);
    *out = (1 - g) * y[(int64_t)j] + g * y[(int64_t)jp1];
    return TSR_OK;
}

int r_iqr(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "x");
    double scale = 1.0;
    if (nargs > 3 && args[3].kind == 2) {
        std::string s(args[3].str);
        for (auto &c : s) c = (char)tolower(c);
        if (s == "normal") scale = 1.3489795003921636;
        else { fn_set_error("%s not a valid scale for `iqr`", args[3].str); return TSR_EARG; }
    } else scale = arg_num(args, nargs, 3, 1.0);
    double r0 = 25, r1 = 75;
    if (!is_null(args, nargs, 2)) {
        Arr R;
        if (!get_arr(args, nargs, 2, R) || R.size() != 2) { fn_set_error("`rng` must be a two element sequence."); return TSR_EARG; }
        r0 = R.v[0]; r1 = R.v[1];
    }
    if (isnan_(r0) || isnan_(r1)) { fn_set_error("`rng` must not contain NaNs."); return TSR_EARG; }
    double p0, p1;
    if (r0 < r1) { p0 = r0 / 100; p1 = r1 / 100; } else { p0 = r1 / 100; p1 = r0 / 100; }
    if (p0 < 0 || p1 > 1) { fn_set_error("Elements of `rng` must be in the range [0, 100]."); return TSR_EARG; }
    int pol;
    if (nan_policy_of(args, nargs, 4, &pol) != TSR_OK) return TSR_EARG;
    std::string interp(arg_str(args, nargs, 5, "linear"));
    if (interp == "lower" || interp == "midpoint" || interp == "higher" || interp == "nearest") interp = "_" + interp;
    const bool keep = arg_bool(args, nargs, 6, false);
    Lanes L;
    int rc = make_lanes(A, args, nargs, 1, L, true);
    if (rc != TSR_OK) return rc;
    vector<double> out(L.nl + 1);
    for (int64_t o = 0; o < L.nl; o++) {
        double *x = L.lane(o);
        int64_t n = L.len;
        const int r = nan_slice(x, n, pol);
        if (r < 0) return TSR_EARG;
        if (r == 1 || n == 0) {
            double dummy;
            if (quantile_sorted(&dummy, 1, 0.5, interp, &dummy) != TSR_OK) return TSR_EARG;
            out[o] = NaN;
            continue;
        }
        sort_nan_last(x, n);
        double q0, q1;
        if (quantile_sorted(x, n, p0, interp, &q0) != TSR_OK) return TSR_EARG;
        if (quantile_sorted(x, n, p1, interp, &q1) != TSR_OK) return TSR_EARG;
        double v = q1 - q0;
        if (scale != 1.0) v /= scale;
        out[o] = v;
    }
    return emit(&res[0], L, out.data(), keep, false);
}

double median_of(double *x, int64_t n)
{
    if (n == 0 || any_nan(x, n)) return NaN;
    sort_nan_last(x, n);
    const int64_t h = n / 2;
    return n % 2 ? x[h] : (x[h - 1] + x[h]) / 2.0;
}

int r_mad(const void *, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    NEED_ARRAY(0, A, "x");
    if (!is_null(args, nargs, 2)) { fn_set_error("The argument 'center' must be callable (only the default median is supported)."); return TSR_EARG; }
    double scale = 1.0;
    if (nargs > 3 && args[3].kind == 2) {
        std::string s(args[3].str);
        for (auto &c : s) c = (char)tolower(c);
        if (s == "normal") scale = 0.6744897501960817;
        else { fn_set_error("%s is not a valid scale value.", args[3].str); return TSR_EARG; }
    } else scale = arg_num(args, nargs, 3, 1.0);
    int pol;
    if (nan_policy_of(args, nargs, 4, &pol) != TSR_OK) return TSR_EARG;
    const bool keep = arg_bool(args, nargs, 5, false);
    Lanes L;
    int rc = make_lanes(A, args, nargs, 1, L, false);
    if (rc != TSR_OK) return rc;
    vector<double> out(L.nl + 1), t(L.len + 1);
    for (int64_t o = 0; o < L.nl; o++) {
        double *x = L.lane(o);
        int64_t n = L.len;
        const int r = nan_slice(x, n, pol);
        if (r < 0) return TSR_EARG;
        if (r == 1 || n == 0) { out[o] = NaN; continue; }
        for (int64_t i = 0; i < n; i++) t[i] = x[i];
        const double med = median_of(t.data(), n);
        for (int64_t i = 0; i < n; i++) t[i] = std::fabs(x[i] - med);
        out[o] = median_of(t.data(), n) / scale;
    }
    return emit(&res[0], L, out.data(), keep, false);
}

/* ================================================================ boxcox_llf, yeojohnson_llf */

/* scipy.special.logsumexp over a 1-D slice (b optional; the sign is returned when sgn != nullptr) */
double logsumexp(const double *a_in, const double *b, int64_t n, unsigned fl, double *sgn)
{
    vector<double> a(a_in, a_in + n), t(n + 1);
    /* the direct computation, used where the careful one is not finite */
    for (int64_t i = 0; i < n; i++) t[i] = b ? b[i] * std::exp(a[i]) : std::exp(a[i]);
    double sum_ = fn_sum(t.data(), n, fl);
    double sgn_inf = 0;
    if (sgn) { sgn_inf = sum_ > 0 ? 1 : (sum_ < 0 ? -1 : (sum_ == 0 ? 0 : NaN)); sum_ = std::fabs(sum_); }
    const double out_inf = std::log(sum_);
    if (b) for (int64_t i = 0; i < n; i++) if (b[i] == 0) a[i] = -INF;
    double amax = a[0];
    for (int64_t i = 1; i < n; i++) {
        if (isnan_(amax)) break;
        if (isnan_(a[i])) { amax = a[i]; break; }
        if (a[i] > amax) amax = a[i];
    }
    vector<double> im(n + 1);
    for (int64_t i = 0; i < n; i++) {
        const bool is_max = a[i] == amax;
        im[i] = b ? b[i] * (is_max ? 1.0 : 0.0) : (is_max ? 1.0 : 0.0);
        if (is_max) a[i] = -INF;
    }
    double m = fn_sum(im.data(), n, fl);
    for (int64_t i = 0; i < n; i++) t[i] = b ? b[i] * std::exp(a[i] - amax) : std::exp(a[i] - amax);
    double s = fn_sum(t.data(), n, fl);
    s = s == 0 ? s : s / m;
    auto sign = [](double v) { return v > 0 ? 1.0 : (v < 0 ? -1.0 : (v == 0 ? 0.0 : NaN)); };
    const double sg = sign(s + 1) * sign(m);
    if (s < -1) s = -s - 2;
    m = std::fabs(m);
    double out = std::log1p(s) + std::log(m) + amax;
    if (!sgn && sg < 0) out = NaN;
    double sgv = sg;
    if (!std::isfinite(out)) { out = out_inf; sgv = sgn_inf; }
    if (sgn) *sgn = sgv;
    return out;
}

int r_llf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    const bool yj = ctx != nullptr;
    if (nargs < 1 || args[0].kind != 1) { fn_set_error("lmb must be a number"); return TSR_EARG; }
    const double lmb = args[0].num;
    NEED_ARRAY(1, A, "data");
    const int knan = yj ? 3 : 4, kkeep = yj ? 4 : 3;
    int pol;
    if (nan_policy_of(args, nargs, knan, &pol) != TSR_OK) return TSR_EARG;
    const bool keep = arg_bool(args, nargs, kkeep, false);
    Lanes L;
    int rc = make_lanes(A, args, nargs, 2, L, false);
    if (rc != TSR_OK) return rc;
    vector<double> out(L.nl + 1), t(L.len + 1), u(L.len + 1);
    for (int64_t o = 0; o < L.nl; o++) {
        double *x = L.lane(o);
        int64_t n = L.len;
        const int r = nan_slice(x, n, pol);
        if (r < 0) return TSR_EARG;
        if (r == 1 || n == 0) { out[o] = NaN; continue; }
        const unsigned fl = (r == 0 && n == L.len) ? L.flags : 0u;
        const double N = (double)n;
        if (!yj) {
            for (int64_t i = 0; i < n; i++) t[i] = std::log(x[i]);
            double logvar;
            if (lmb == 0) logvar = std::log(np_var(t.data(), n, 0, fl, u.data()));
            else {
                vector<double> logx(n);
                for (int64_t i = 0; i < n; i++) logx[i] = lmb * t[i];
                const double logmean = logsumexp(logx.data(), nullptr, n, fl, nullptr) - std::log(N);
                vector<double> lxm(n);
                const double b2[2] = {1.0, -1.0};
                for (int64_t i = 0; i < n; i++) {
                    const double pair[2] = {logx[i], logmean};
                    double sg;
                    lxm[i] = 2 * logsumexp(pair, b2, 2, 0, &sg);
                }
                logvar = logsumexp(lxm.data(), nullptr, n, fl, nullptr) - std::log(N) - 2 * std::log(std::fabs(lmb));
            }
            out[o] = (lmb - 1) * fn_sum(t.data(), n, fl) - N / 2 * logvar;
        } else {
            for (int64_t i = 0; i < n; i++) {
                const double v = x[i];
                double y;
                if (v >= 0) y = std::fabs(lmb) < EPS ? std::log1p(v) : std::expm1(lmb * std::log1p(v)) / lmb;
                else y = std::fabs(lmb - 2) > EPS ? -std::expm1((2 - lmb) * std::log1p(-v)) / (2 - lmb) : -std::log1p(-v);
                t[i] = y;
            }
            const double sigma = np_var(t.data(), n, 0, fl, u.data());
            const double log_sigma = sigma >= 2.2250738585072014e-308 ? std::log(sigma) : -INF;
            for (int64_t i = 0; i < n; i++) {
                const double v = x[i];
                const double sg = v > 0 ? 1.0 : (v < 0 ? -1.0 : 0.0);
                t[i] = sg * std::log1p(std::fabs(v));
            }
            out[o] = -N / 2 * log_sigma + (lmb - 1) * fn_sum(t.data(), n, fl);
        }
    }
    return emit(&res[0], L, out.data(), keep, false);
}

/* ================================================================ energy_distance, wasserstein_distance */

int validate_dist(const tsr_arg *args, int nargs, int kv, int kw, vector<double> &v, vector<double> &w, bool &has_w)
{
    Arr V;
    if (!get_arr(args, nargs, kv, V) || V.ndim != 1) { fn_set_error("values must be a 1-D array"); return TSR_EARG; }
    if (V.size() == 0) { fn_set_error("Distribution can't be empty."); return TSR_EARG; }
    v = V.v;
    has_w = !is_null(args, nargs, kw);
    if (has_w) {
        Arr W;
        if (!get_arr(args, nargs, kw, W) || W.size() != V.size()) {
            fn_set_error("Value and weight array-likes for the same empirical distribution must be of the same size.");
            return TSR_EARG;
        }
        for (double x : W.v) if (x < 0) { fn_set_error("All weights must be non-negative."); return TSR_EARG; }
        vector<double> c = W.v;
        const double s = tsr_psum(c.data(), (int64_t)c.size());
        if (!(0 < s && s < INF)) {
            fn_set_error("Weight array-like sum must be positive and finite. Set as None for an equal distribution of weight.");
            return TSR_EARG;
        }
        w = W.v;
    }
    return TSR_OK;
}

int r_cdf_distance(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int)
{
    const int p = *(const int *)ctx;
    vector<double> u, v, uw, vw;
    bool hu, hv;
    if (validate_dist(args, nargs, 0, 2, u, uw, hu) != TSR_OK) return TSR_EARG;
    if (validate_dist(args, nargs, 1, 3, v, vw, hv) != TSR_OK) return TSR_EARG;
    const int64_t nu = (int64_t)u.size(), nv = (int64_t)v.size();
    auto sorter = [](const vector<double> &x) {
        vector<int64_t> s(x.size());
        std::iota(s.begin(), s.end(), 0);
        std::stable_sort(s.begin(), s.end(), [&](int64_t a, int64_t b) { return lt_nan(x[a], x[b]); });
        return s;
    };
    const vector<int64_t> us = sorter(u), vs = sorter(v);
    vector<double> su(nu), sv(nv);
    for (int64_t i = 0; i < nu; i++) su[i] = u[us[i]];
    for (int64_t i = 0; i < nv; i++) sv[i] = v[vs[i]];
    vector<double> all(su);
    all.insert(all.end(), sv.begin(), sv.end());
    std::stable_sort(all.begin(), all.end(), lt_nan);
    const int64_t na = (int64_t)all.size();
    auto cdf = [&](const vector<double> &sorted, const vector<double> &wt, const vector<int64_t> &ord, bool hw, vector<double> &out) {
        const int64_t n = (int64_t)sorted.size();
        vector<double> cw;
        if (hw) {
            cw.assign(n + 1, 0.0);
            for (int64_t i = 0; i < n; i++) cw[i + 1] = i == 0 ? wt[ord[0]] : cw[i] + wt[ord[i]];
        }
        out.assign(na - 1, 0.0);
        for (int64_t k = 0; k < na - 1; k++) {
            const int64_t idx = std::upper_bound(sorted.begin(), sorted.end(), all[k], lt_nan) - sorted.begin();
            out[k] = hw ? cw[idx] / cw[n] : (double)idx / (double)n;
        }
    };
    vector<double> ucdf, vcdf;
    cdf(su, uw, us, hu, ucdf);
    cdf(sv, vw, vs, hv, vcdf);
    vector<double> t(na), deltas(na);
    for (int64_t k = 0; k < na - 1; k++) {
        const double d = ucdf[k] - vcdf[k];
        t[k] = p == 1 ? std::fabs(d) : d * d;
        deltas[k] = all[k + 1] - all[k];
    }
    const double acc = blas_dot(t.data(), deltas.data(), na - 1);
    double r = p == 1 ? acc : std::sqrt(acc);
    if (p == 2) r = std::sqrt(2.0) * r;
    fn_result_num(&res[0], r);
    return TSR_OK;
}

const int P1 = 1, P2 = 2;

/* gaussian_kde(dataset, points): Gaussian kernel density estimate (Scott's factor), evaluated at points.
   dataset is (d, n) or (n,) for d=1; points is (d, m) or (m,); returns the density at the m points. */
static int r_gaussian_kde(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("gaussian_kde: dataset and points must be arrays"); return TSR_EARG; }
    const tsr_array &da = args[0].arr, &pa = args[1].arr;
    int64_t d, n, m;
    if (da.ndim == 1) { d = 1; n = da.shape[0]; } else if (da.ndim == 2) { d = da.shape[0]; n = da.shape[1]; } else { fn_set_error("gaussian_kde: dataset must be 1-D or 2-D"); return TSR_EARG; }
    if (pa.ndim == 1) { if (d != 1) { fn_set_error("gaussian_kde: points dimensionality must match dataset"); return TSR_EARG; } m = pa.shape[0]; }
    else if (pa.ndim == 2) { if (pa.shape[0] != d) { fn_set_error("gaussian_kde: points dimensionality must match dataset"); return TSR_EARG; } m = pa.shape[1]; }
    else { fn_set_error("gaussian_kde: points must be 1-D or 2-D"); return TSR_EARG; }
    if (n < 2) { fn_set_error("gaussian_kde: need at least two data points"); return TSR_EARG; }
    int64_t ld, lp; double *data = fn_arg_doubles(&args[0], &ld), *pts = data ? fn_arg_doubles(&args[1], &lp) : nullptr;
    if (!data || !pts) { fn_free_doubles(data, ld); return TSR_ENOMEM; }
    std::vector<double> mean(d, 0.0), C(d * d, 0.0), L(d * d, 0.0), diff(d), z(d);
    for (int64_t a = 0; a < d; a++) { double s = 0.0; for (int64_t i = 0; i < n; i++) s += data[a * n + i]; mean[a] = s / (double)n; }
    const double factor2 = std::pow((double)n, -2.0 / (double)(d + 4));   /* Scott's factor, squared */
    for (int64_t a = 0; a < d; a++) for (int64_t b = 0; b < d; b++) { double s = 0.0; for (int64_t i = 0; i < n; i++) s += (data[a * n + i] - mean[a]) * (data[b * n + i] - mean[b]); C[a * d + b] = (s / (double)(n - 1)) * factor2; }
    for (int64_t a = 0; a < d; a++) for (int64_t b = 0; b <= a; b++) {   /* Cholesky C = L L^T */
        double s = C[a * d + b]; for (int64_t c = 0; c < b; c++) s -= L[a * d + c] * L[b * d + c];
        if (a == b) { if (s <= 0.0) { fn_free_doubles(data, ld); fn_free_doubles(pts, lp); fn_set_error("gaussian_kde: singular covariance"); return TSR_EARG; } L[a * d + a] = std::sqrt(s); }
        else L[a * d + b] = s / L[b * d + b];
    }
    double diagprod = 1.0; for (int64_t a = 0; a < d; a++) diagprod *= L[a * d + a];
    const double norm = (double)n * std::pow(2.0 * M_PI, (double)d / 2.0) * diagprod;
    int64_t osh[1] = {m};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, osh);
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t j = 0; j < m; j++) {
        double acc = 0.0;
        for (int64_t i = 0; i < n; i++) {
            for (int64_t a = 0; a < d; a++) diff[a] = pts[a * m + j] - data[a * n + i];
            for (int64_t a = 0; a < d; a++) { double s = diff[a]; for (int64_t c = 0; c < a; c++) s -= L[a * d + c] * z[c]; z[a] = s / L[a * d + a]; }
            double maha = 0.0; for (int64_t a = 0; a < d; a++) maha += z[a] * z[a];
            acc += std::exp(-0.5 * maha);
        }
        out[j] = acc / norm;
    }
    fn_free_doubles(data, ld); fn_free_doubles(pts, lp);
    return rc;
}
/* ecdf(sample): empirical CDF -> the sorted unique sample values and the cumulative proportion <= each. */
static int r_ecdf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("ecdf: sample must be an array"); return TSR_EARG; }
    int64_t ls; double *s = fn_arg_doubles(&args[0], &ls);
    if (!s) return TSR_ENOMEM;
    std::vector<double> v(s, s + ls); std::sort(v.begin(), v.end());
    std::vector<double> q, p;
    for (int64_t i = 0; i < ls;) { double val = v[i]; int64_t j = i; while (j < ls && v[j] == val) j++; q.push_back(val); p.push_back((double)j / (double)ls); i = j; }
    const int64_t nq = (int64_t)q.size();
    int64_t qsh[1] = {nq};
    double *qo = (double *)fn_result_array(&res[0], TSR_F64, 1, qsh);
    double *po = qo ? (double *)fn_result_array(&res[1], TSR_F64, 1, qsh) : nullptr;
    int rc = po ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t k = 0; k < nq; k++) { qo[k] = q[k]; po[k] = p[k]; }
    fn_free_doubles(s, ls);
    return rc;
}

/* Filliben's order-statistic medians mapped through the normal quantile (for probplot / normplot). */
static void stat_osm(double *a, int64_t n)
{
    if (n == 1) { a[0] = sc::ndtri(0.5); return; }
    double last = std::pow(0.5, 1.0 / (double)n);
    a[n - 1] = sc::ndtri(last);
    a[0] = sc::ndtri(1.0 - last);
    for (int64_t p = 1; p < n - 1; p++) a[p] = sc::ndtri(((double)(p + 1) - 0.3175) / ((double)n + 0.365));
}
/* Pearson correlation of a[] and b[] (length n); also returns slope/intercept of b on a via out params. */
static double stat_linr(const double *a, const double *b, int64_t n, double *slope, double *intercept)
{
    double abar = 0.0, bbar = 0.0;
    for (int64_t i = 0; i < n; i++) { abar += a[i]; bbar += b[i]; }
    abar /= (double)n; bbar /= (double)n;
    double num = 0.0, da = 0.0, db = 0.0;
    for (int64_t i = 0; i < n; i++) { double u = a[i] - abar, v = b[i] - bbar; num += u * v; da += u * u; db += v * v; }
    if (slope) *slope = num / da;
    if (intercept) *intercept = bbar - (num / da) * abar;
    return num / std::sqrt(da * db);
}
/* directional_stats(samples): normalize each row to unit length; mean direction and mean resultant length. */
static int r_directional_stats(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("directional_stats: samples must be a 2-D array"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], d = args[0].arr.shape[1];
    int64_t ls; double *S = fn_arg_doubles(&args[0], &ls);
    if (!S) return TSR_ENOMEM;
    std::vector<double> mv(d, 0.0);
    for (int64_t i = 0; i < n; i++) { double nr = 0.0; for (int64_t a = 0; a < d; a++) nr += S[i * d + a] * S[i * d + a]; nr = std::sqrt(nr);
        for (int64_t a = 0; a < d; a++) mv[a] += S[i * d + a] / nr; }
    for (int64_t a = 0; a < d; a++) mv[a] /= (double)n;
    double R = 0.0; for (int64_t a = 0; a < d; a++) R += mv[a] * mv[a]; R = std::sqrt(R);
    int64_t dsh[1] = {d};
    double *md = (double *)fn_result_array(&res[0], TSR_F64, 1, dsh);
    int rc = md ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) { for (int64_t a = 0; a < d; a++) md[a] = mv[a] / R; fn_result_num(&res[1], R); }
    fn_free_doubles(S, ls);
    return rc;
}
/* probplot(x): order-statistic medians osm, ordered data osr, and the least-squares slope/intercept/r. */
static int r_probplot(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("probplot: x must be an array"); return TSR_EARG; }
    int64_t lx; double *x = fn_arg_doubles(&args[0], &lx);
    if (!x) return TSR_ENOMEM;
    const int64_t n = lx;
    std::vector<double> osr(x, x + n), osm(n);
    std::sort(osr.begin(), osr.end());
    stat_osm(osm.data(), n);
    int64_t sh[1] = {n};
    double *om = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    double *orr = om ? (double *)fn_result_array(&res[1], TSR_F64, 1, sh) : nullptr;
    int rc = orr ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        double slope, intercept, r = stat_linr(osm.data(), osr.data(), n, &slope, &intercept);
        for (int64_t i = 0; i < n; i++) { om[i] = osm[i]; orr[i] = osr[i]; }
        fn_result_num(&res[2], slope); fn_result_num(&res[3], intercept); fn_result_num(&res[4], r);
    }
    fn_free_doubles(x, lx);
    return rc;
}
/* probability-plot correlation of transformed data over a range of lambdas (boxcox / yeo-johnson normplot). */
static int stat_normplot(const tsr_arg *args, int nargs, tsr_result *res, int yeo)
{
    if (args[0].kind != 3) { fn_set_error("normplot: x must be an array"); return TSR_EARG; }
    const double la = args[1].num, lb = args[2].num;
    const int64_t N = (nargs > 3 && args[3].kind == 1) ? ((args[3].flags & 1) ? args[3].ival : (int64_t)args[3].num) : 80;
    int64_t lx; double *x = fn_arg_doubles(&args[0], &lx);
    if (!x) return TSR_ENOMEM;
    const int64_t n = lx;
    std::vector<double> osm(n), y(n);
    stat_osm(osm.data(), n);
    int64_t sh[1] = {N};
    double *lm = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    double *pp = lm ? (double *)fn_result_array(&res[1], TSR_F64, 1, sh) : nullptr;
    int rc = pp ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t k = 0; k < N; k++) {
        double lmb = (N == 1) ? la : la + (lb - la) * (double)k / (double)(N - 1);
        for (int64_t i = 0; i < n; i++) {
            double xi = x[i], t;
            if (!yeo) t = (lmb == 0.0) ? std::log(xi) : (std::pow(xi, lmb) - 1.0) / lmb;
            else if (xi >= 0.0) t = (lmb == 0.0) ? std::log1p(xi) : (std::pow(xi + 1.0, lmb) - 1.0) / lmb;
            else t = (lmb == 2.0) ? -std::log1p(-xi) : -(std::pow(-xi + 1.0, 2.0 - lmb) - 1.0) / (2.0 - lmb);
            y[i] = t;
        }
        std::sort(y.begin(), y.end());
        lm[k] = lmb; pp[k] = stat_linr(osm.data(), y.data(), n, nullptr, nullptr);
    }
    fn_free_doubles(x, lx);
    return rc;
}
static int r_boxcox_normplot(const void *ctx, const tsr_arg *a, int n, tsr_result *res, int nres) { (void)ctx; (void)nres; return stat_normplot(a, n, res, 0); }
static int r_yeojohnson_normplot(const void *ctx, const tsr_arg *a, int n, tsr_result *res, int nres) { (void)ctx; (void)nres; return stat_normplot(a, n, res, 1); }

/* multivariate_normal(mean, cov, x): the pdf and log-pdf at each row of x (shape (m, d)). */
static int r_multivariate_normal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3 || args[1].arr.ndim != 2 || args[2].kind != 3 || args[2].arr.ndim != 2) { fn_set_error("multivariate_normal: mean 1-D, cov 2-D, x (m,d)"); return TSR_EARG; }
    const int64_t d = args[0].arr.shape[0], m = args[2].arr.shape[0];
    int64_t lm, lc, lx; double *mean = fn_arg_doubles(&args[0], &lm), *C = mean ? fn_arg_doubles(&args[1], &lc) : nullptr, *X = C ? fn_arg_doubles(&args[2], &lx) : nullptr;
    if (!X) { fn_free_doubles(mean, lm); fn_free_doubles(C, lc); return TSR_ENOMEM; }
    std::vector<double> L(d * d, 0.0), z(d);
    int rc = TSR_OK;
    for (int64_t a = 0; a < d && rc == TSR_OK; a++) for (int64_t b = 0; b <= a; b++) {
        double s = C[a * d + b]; for (int64_t c = 0; c < b; c++) s -= L[a * d + c] * L[b * d + c];
        if (a == b) { if (s <= 0.0) { rc = TSR_EARG; fn_set_error("multivariate_normal: covariance not positive definite"); break; } L[a * d + a] = std::sqrt(s); }
        else L[a * d + b] = s / L[b * d + b];
    }
    double logdet = 0.0; for (int64_t a = 0; a < d; a++) logdet += 2.0 * std::log(L[a * d + a]);
    const double c0 = (double)d * std::log(2.0 * M_PI);
    int64_t sh[1] = {m};
    double *pdf = (rc == TSR_OK) ? (double *)fn_result_array(&res[0], TSR_F64, 1, sh) : nullptr;
    double *lpdf = pdf ? (double *)fn_result_array(&res[1], TSR_F64, 1, sh) : nullptr;
    if (rc == TSR_OK && !lpdf) rc = TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t j = 0; j < m; j++) {
        for (int64_t a = 0; a < d; a++) { double s = X[j * d + a] - mean[a]; for (int64_t c = 0; c < a; c++) s -= L[a * d + c] * z[c]; z[a] = s / L[a * d + a]; }
        double maha = 0.0; for (int64_t a = 0; a < d; a++) maha += z[a] * z[a];
        double lp = -0.5 * (c0 + logdet + maha); lpdf[j] = lp; pdf[j] = std::exp(lp);
    }
    fn_free_doubles(mean, lm); fn_free_doubles(C, lc); fn_free_doubles(X, lx);
    return rc;
}
/* dirichlet(alpha, x): the pdf at each row of x (shape (m, k)); rows must lie on the simplex. */
static int r_dirichlet(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("dirichlet: alpha 1-D, x (m,k)"); return TSR_EARG; }
    const int64_t k = args[0].arr.shape[0], m = args[1].arr.shape[0];
    int64_t la, lx; double *al = fn_arg_doubles(&args[0], &la), *X = al ? fn_arg_doubles(&args[1], &lx) : nullptr;
    if (!X) { fn_free_doubles(al, la); return TSR_ENOMEM; }
    double asum = 0.0, lg = 0.0; for (int64_t i = 0; i < k; i++) { asum += al[i]; lg += std::lgamma(al[i]); }
    const double c = std::lgamma(asum) - lg;
    int64_t sh[1] = {m}; double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t j = 0; j < m; j++) { double s = c; for (int64_t i = 0; i < k; i++) s += (al[i] - 1.0) * std::log(X[j * k + i]); out[j] = std::exp(s); }
    fn_free_doubles(al, la); fn_free_doubles(X, lx);
    return rc;
}
/* multinomial(n, p, x): the pmf at each row of x (shape (m, k)). */
static int r_multinomial(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 1 || args[1].kind != 3 || args[2].kind != 3 || args[2].arr.ndim != 2) { fn_set_error("multinomial: n int, p 1-D, x (m,k)"); return TSR_EARG; }
    const double nn = args[0].num; const int64_t k = args[1].arr.shape[0], m = args[2].arr.shape[0];
    int64_t lp, lx; double *p = fn_arg_doubles(&args[1], &lp), *X = p ? fn_arg_doubles(&args[2], &lx) : nullptr;
    if (!X) { fn_free_doubles(p, lp); return TSR_ENOMEM; }
    int64_t sh[1] = {m}; double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t j = 0; j < m; j++) { double s = std::lgamma(nn + 1.0);
        for (int64_t i = 0; i < k; i++) { s -= std::lgamma(X[j * k + i] + 1.0); if (p[i] > 0.0) s += X[j * k + i] * std::log(p[i]); }
        out[j] = std::exp(s); }
    fn_free_doubles(p, lp); fn_free_doubles(X, lx);
    return rc;
}
/* multivariate_hypergeom(m, n, x): the pmf at each row of x (shape (mm, k)); m are the per-color counts. */
static int r_multivariate_hypergeom(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 1 || args[2].kind != 3 || args[2].arr.ndim != 2) { fn_set_error("multivariate_hypergeom: m 1-D, n int, x (mm,k)"); return TSR_EARG; }
    const int64_t k = args[0].arr.shape[0], mm = args[2].arr.shape[0]; const double n = args[1].num;
    int64_t lmc, lx; double *mc = fn_arg_doubles(&args[0], &lmc), *X = mc ? fn_arg_doubles(&args[2], &lx) : nullptr;
    if (!X) { fn_free_doubles(mc, lmc); return TSR_ENOMEM; }
    double M = 0.0; for (int64_t i = 0; i < k; i++) M += mc[i];
    auto logC = [](double a, double b) { return std::lgamma(a + 1.0) - std::lgamma(b + 1.0) - std::lgamma(a - b + 1.0); };
    const double denom = logC(M, n);
    int64_t sh[1] = {mm}; double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t j = 0; j < mm; j++) { double s = -denom; for (int64_t i = 0; i < k; i++) s += logC(mc[i], X[j * k + i]); out[j] = std::exp(s); }
    fn_free_doubles(mc, lmc); fn_free_doubles(X, lx);
    return rc;
}
/* lower-Cholesky of a d-by-d SPD matrix C (row-major) into L; returns false if not positive definite. */
static bool stat_chol(const double *C, int64_t d, double *L)
{
    for (int64_t a = 0; a < d; a++) for (int64_t b = 0; b <= a; b++) {
        double s = C[a * d + b]; for (int64_t c = 0; c < b; c++) s -= L[a * d + c] * L[b * d + c];
        if (a == b) { if (s <= 0.0) return false; L[a * d + a] = std::sqrt(s); } else L[a * d + b] = s / L[b * d + b];
    }
    return true;
}
/* multivariate_t(loc, shape, df, x): pdf and log-pdf of the multivariate Student-t at each row of x. */
static int r_multivariate_t(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3 || args[1].arr.ndim != 2 || args[2].kind != 1 || args[3].kind != 3 || args[3].arr.ndim != 2) { fn_set_error("multivariate_t: loc 1-D, shape 2-D, df scalar, x (m,d)"); return TSR_EARG; }
    const int64_t d = args[0].arr.shape[0], m = args[3].arr.shape[0]; const double df = args[2].num;
    int64_t ll, ls, lx; double *loc = fn_arg_doubles(&args[0], &ll), *S = loc ? fn_arg_doubles(&args[1], &ls) : nullptr, *X = S ? fn_arg_doubles(&args[3], &lx) : nullptr;
    if (!X) { fn_free_doubles(loc, ll); fn_free_doubles(S, ls); return TSR_ENOMEM; }
    std::vector<double> L(d * d, 0.0), z(d);
    int rc = stat_chol(S, d, L.data()) ? TSR_OK : TSR_EARG;
    if (rc != TSR_OK) fn_set_error("multivariate_t: shape not positive definite");
    double logdet = 0.0; for (int64_t a = 0; a < d; a++) logdet += 2.0 * std::log(L[a * d + a]);
    const double c = std::lgamma((df + d) / 2.0) - std::lgamma(df / 2.0) - 0.5 * d * std::log(df * M_PI) - 0.5 * logdet;
    int64_t sh[1] = {m};
    double *pdf = (rc == TSR_OK) ? (double *)fn_result_array(&res[0], TSR_F64, 1, sh) : nullptr;
    double *lpdf = pdf ? (double *)fn_result_array(&res[1], TSR_F64, 1, sh) : nullptr;
    if (rc == TSR_OK && !lpdf) rc = TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t j = 0; j < m; j++) {
        for (int64_t a = 0; a < d; a++) { double s = X[j * d + a] - loc[a]; for (int64_t cc = 0; cc < a; cc++) s -= L[a * d + cc] * z[cc]; z[a] = s / L[a * d + a]; }
        double maha = 0.0; for (int64_t a = 0; a < d; a++) maha += z[a] * z[a];
        double lp = c - 0.5 * (df + d) * std::log1p(maha / df); lpdf[j] = lp; pdf[j] = std::exp(lp);
    }
    fn_free_doubles(loc, ll); fn_free_doubles(S, ls); fn_free_doubles(X, lx);
    return rc;
}
/* vonmises_fisher(mu, kappa, x): pdf of the von Mises-Fisher distribution at each row of x (unit vectors). */
static int r_vonmises_fisher(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 1 || args[2].kind != 3 || args[2].arr.ndim != 2) { fn_set_error("vonmises_fisher: mu 1-D, kappa scalar, x (m,d)"); return TSR_EARG; }
    const int64_t d = args[0].arr.shape[0], m = args[2].arr.shape[0]; const double kappa = args[1].num;
    int64_t lm, lx; double *mu = fn_arg_doubles(&args[0], &lm), *X = mu ? fn_arg_doubles(&args[2], &lx) : nullptr;
    if (!X) { fn_free_doubles(mu, lm); return TSR_ENOMEM; }
    const double halfd = (double)d / 2.0;
    double in2[2] = {halfd - 1.0, kappa}, iv = 0.0;
    tsr_special_iv(nullptr, in2, &iv);
    const double logC = (halfd - 1.0) * std::log(kappa) - halfd * std::log(2.0 * M_PI) - std::log(iv);
    int64_t sh[1] = {m}; double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    int rc = out ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) for (int64_t j = 0; j < m; j++) { double dot = 0.0; for (int64_t a = 0; a < d; a++) dot += mu[a] * X[j * d + a]; out[j] = std::exp(logC + kappa * dot); }
    fn_free_doubles(mu, lm); fn_free_doubles(X, lx);
    return rc;
}
/* log-determinant of a d-by-d SPD matrix via Cholesky; sets *ok=0 if not positive definite. */
static double stat_logdet(const double *C, int64_t d, int *ok)
{
    std::vector<double> L(d * d, 0.0);
    if (!stat_chol(C, d, L.data())) { *ok = 0; return 0.0; }
    double s = 0.0; for (int64_t a = 0; a < d; a++) s += 2.0 * std::log(L[a * d + a]);
    *ok = 1; return s;
}
/* multivariate log-gamma Gamma_d(a). */
static double stat_mgl(double a, int64_t d)
{
    double s = (double)d * (d - 1) / 4.0 * std::log(M_PI);
    for (int64_t j = 1; j <= d; j++) s += std::lgamma(a + (1.0 - (double)j) / 2.0);
    return s;
}
/* matrix_normal(M, U, V, X): matrix-normal pdf and log-pdf at a single matrix X (n-by-p). */
static int r_matrix_normal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    for (int i = 0; i < 4; i++) if (args[i].kind != 3 || args[i].arr.ndim != 2) { fn_set_error("matrix_normal: M, U, V, X must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], p = args[0].arr.shape[1];
    int64_t lM, lU, lV, lX; double *M = fn_arg_doubles(&args[0], &lM), *U = M ? fn_arg_doubles(&args[1], &lU) : nullptr,
        *V = U ? fn_arg_doubles(&args[2], &lV) : nullptr, *X = V ? fn_arg_doubles(&args[3], &lX) : nullptr;
    int rc = X ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        std::vector<double> A(n * p), Ucopy(U, U + n * n), B, Vcopy(V, V + p * p), M2(p * p, 0.0);
        for (int64_t k = 0; k < n * p; k++) A[k] = X[k] - M[k];
        B = A;
        int okU, okV; double ldU = stat_logdet(U, n, &okU), ldV = stat_logdet(V, p, &okV);
        if (!okU || !okV) { fn_set_error("matrix_normal: U or V not positive definite"); rc = TSR_EARG; }
        else if (sl_dense_solve(Ucopy.data(), B.data(), n, p) != 0) { fn_set_error("matrix_normal: U is singular"); rc = TSR_EARG; }
        else {
            for (int64_t i = 0; i < p; i++) for (int64_t j = 0; j < p; j++) { double s = 0.0; for (int64_t k = 0; k < n; k++) s += A[k * p + i] * B[k * p + j]; M2[i * p + j] = s; }
            if (sl_dense_solve(Vcopy.data(), M2.data(), p, p) != 0) { fn_set_error("matrix_normal: V is singular"); rc = TSR_EARG; }
            else { double tr = 0.0; for (int64_t i = 0; i < p; i++) tr += M2[i * p + i];
                double lp = -0.5 * ((double)n * p * std::log(2.0 * M_PI) + (double)p * ldU + (double)n * ldV + tr);
                fn_result_num(&res[0], std::exp(lp)); fn_result_num(&res[1], lp); }
        }
    }
    fn_free_doubles(M, lM); fn_free_doubles(U, lU); fn_free_doubles(V, lV); fn_free_doubles(X, lX);
    return rc;
}
/* wishart / inverse-wishart pdf and log-pdf at a single SPD matrix X (d-by-d). inv=1 selects inverse-Wishart. */
static int stat_wishart(const tsr_arg *args, tsr_result *res, int inv)
{
    if (args[0].kind != 1 || args[1].kind != 3 || args[1].arr.ndim != 2 || args[2].kind != 3 || args[2].arr.ndim != 2) { fn_set_error("wishart: df scalar, scale and X 2-D"); return TSR_EARG; }
    const double df = args[0].num; const int64_t d = args[1].arr.shape[0];
    int64_t lS, lX; double *S = fn_arg_doubles(&args[1], &lS), *X = S ? fn_arg_doubles(&args[2], &lX) : nullptr;
    int rc = X ? TSR_OK : TSR_ENOMEM;
    if (rc == TSR_OK) {
        int okS, okX; double ldS = stat_logdet(S, d, &okS), ldX = stat_logdet(X, d, &okX);
        if (!okS || !okX) { fn_set_error("wishart: scale or X not positive definite"); rc = TSR_EARG; }
        else {
            std::vector<double> Acopy, Bcopy;                 /* solve A Y = B -> trace(Y) */
            if (!inv) { Acopy.assign(S, S + d * d); Bcopy.assign(X, X + d * d); }   /* tr(scale^-1 X) */
            else { Acopy.assign(X, X + d * d); Bcopy.assign(S, S + d * d); }        /* tr(X^-1 scale) */
            if (sl_dense_solve(Acopy.data(), Bcopy.data(), d, d) != 0) { fn_set_error("wishart: singular matrix"); rc = TSR_EARG; }
            else {
                double tr = 0.0; for (int64_t i = 0; i < d; i++) tr += Bcopy[i * d + i];
                double mgl = stat_mgl(0.5 * df, d), lp;
                if (!inv) lp = 0.5 * (df - d - 1) * ldX - 0.5 * tr - 0.5 * df * d * std::log(2.0) - 0.5 * df * ldS - mgl;
                else lp = 0.5 * df * ldS - 0.5 * (df + d + 1) * ldX - 0.5 * tr - 0.5 * df * d * std::log(2.0) - mgl;
                fn_result_num(&res[0], std::exp(lp)); fn_result_num(&res[1], lp);
            }
        }
    }
    fn_free_doubles(S, lS); fn_free_doubles(X, lX);
    return rc;
}
static int r_wishart(const void *ctx, const tsr_arg *a, int n, tsr_result *res, int nres) { (void)ctx; (void)n; (void)nres; return stat_wishart(a, res, 0); }
static int r_invwishart(const void *ctx, const tsr_arg *a, int n, tsr_result *res, int nres) { (void)ctx; (void)n; (void)nres; return stat_wishart(a, res, 1); }

}  // namespace

/* ================================================================ registry */

#define GU(NAME, NIN, NOUT, ARGS, OUTS, PARAMS, CORE, SCR, C0, C1, OUTINT, AXIS, CTX, AXR, DOC)                 \
    {NAME, FN_GUFUNC, NIN, NOUT, ARGS, OUTS, PARAMS, DOC, NULL, NULL, 0, CORE, SCR, {C0, C1, 0, 0}, OUTINT, 0, NULL, \
     AXIS, 0, NIN, CTX, AXR, FN_NO_RANDOM}

static const fn_def DEFS[] = {
    GU("stats.gmean", 2, 1, "a, weights?", "gmean", NANPOL, c_pmean, NULL, CORE_SCALAR, 0, 0, "0", &K_G, 0,
       "Weighted geometric mean along an axis (scipy.stats.gmean; dtype is not supported)."),
    GU("stats.hmean", 2, 1, "a, weights?", "hmean", NANPOL, c_pmean, NULL, CORE_SCALAR, 0, 0, "0", &K_H, 0,
       "Weighted harmonic mean along an axis (scipy.stats.hmean; dtype is not supported)."),
    GU("stats.pmean", 3, 1, "a, p, weights?", "pmean", NANPOL, c_pmean, NULL, CORE_SCALAR, 0, 0, "0", &K_P, 0,
       "Weighted power mean along an axis (scipy.stats.pmean; dtype is not supported)."),
    GU("stats.kurtosis", 1, 1, "a", "kurtosis", "fisher=True, bias=True, " NANPOL, c_skew, scratch_3n, CORE_SCALAR, 0, 0, "0", &ONE, 0,
       "Kurtosis (Fisher or Pearson) of a data set (scipy.stats.kurtosis)."),
    GU("stats.skew", 1, 1, "a", "skew", "bias=True, " NANPOL, c_skew, scratch_3n, CORE_SCALAR, 0, 0, "0", NULL, 0,
       "Sample skewness of a data set (scipy.stats.skew)."),
    GU("stats.moment", 2, 1, "a, order=1", "moment", NANPOL ", center=None", c_moment, scratch_3n, CORE_LIKE1 | CORE_FIRST, 0, 0, "0", NULL, 0,
       "n-th central moment (or about center) of a sample; an array of orders gives one result each (scipy.stats.moment)."),
    GU("stats.variation", 1, 1, "a", "variation", NANPOL ", ddof=0", c_variation, scratch_n, CORE_SCALAR, 0, 0, "0", NULL, 0,
       "Coefficient of variation, std / mean (scipy.stats.variation)."),
    GU("stats.sem", 1, 1, "a", "sem", "ddof=1, " NANPOL, c_sem, scratch_n, CORE_SCALAR, 0, 0, "0", NULL, 0,
       "Standard error of the mean (scipy.stats.sem)."),
    GU("stats.zscore", 1, 1, "a", "zscore", "ddof=0, " NANPOL, c_zmap, scratch_z1, CORE_LIKE0 | CORE_INPLACE, 0, 0, "0", &K_ZS,
       AX_SINGLE | AX_NOKEEP, "z-scores of each value relative to the sample mean and standard deviation (scipy.stats.zscore)."),
    GU("stats.zmap", 2, 1, "scores, compare", "zmap", "ddof=0, " NANPOL, c_zmap, scratch_zmap, CORE_LIKE0 | CORE_INPLACE, 0, 0, "0", &K_ZM,
       AX_SINGLE | AX_NOKEEP, "z-scores of scores relative to the mean and standard deviation of compare (scipy.stats.zmap)."),
    GU("stats.gzscore", 1, 1, "a", "gzscore", "ddof=0, " NANPOL, c_zmap, scratch_z1, CORE_LIKE0 | CORE_INPLACE, 0, 0, "0", &K_ZG,
       AX_SINGLE | AX_NOKEEP, "Geometric z-scores: z-scores of log(a) (scipy.stats.gzscore)."),
    GU("stats.gstd", 1, 1, "a", "gstd", "ddof=1, " NANPOL, c_gstd, scratch_z1, CORE_SCALAR, 0, 0, "0", NULL, 0,
       "Geometric standard deviation (scipy.stats.gstd)."),
    GU("stats.trim_mean", 2, 1, "a, proportiontocut", "trim_mean", NANPOL, c_trim_mean, NULL, CORE_SCALAR, 0, 0, "0", NULL, 0,
       "Mean after trimming a proportion of the smallest and largest values (scipy.stats.trim_mean)."),
    GU("stats.kstat", 2, 1, "data, n=2", "kstat", NANPOL, c_kstat, scratch_n, CORE_SCALAR, 0, 0, "None", NULL, 0,
       "n-th k-statistic, the unbiased estimator of the n-th cumulant, 1 <= n <= 4 (scipy.stats.kstat)."),
    GU("stats.kstatvar", 2, 1, "data, n=2", "kstatvar", NANPOL, c_kstat, scratch_n, CORE_SCALAR, 0, 0, "None", &ONE, 0,
       "Unbiased estimator of the variance of the k-statistic, n = 1 or 2 (scipy.stats.kstatvar)."),
    GU("stats.circmean", 3, 1, "samples, high=6.283185307179586, low=0", "circmean", NANPOL, c_circ, scratch_2n, CORE_SCALAR, 0, 0,
       "None", &K_CM, 0, "Circular mean of samples in [low, high] (scipy.stats.circmean)."),
    GU("stats.circvar", 3, 1, "samples, high=6.283185307179586, low=0", "circvar", NANPOL, c_circ, scratch_2n, CORE_SCALAR, 0, 0,
       "None", &K_CV, 0, "Circular variance, 1 - R (scipy.stats.circvar)."),
    GU("stats.circstd", 3, 1, "samples, high=6.283185307179586, low=0", "circstd", NANPOL ", normalize=False", c_circ, scratch_2n,
       CORE_SCALAR, 0, 0, "None", &K_CS, 0, "Circular standard deviation, sqrt(-2 log R) (scipy.stats.circstd)."),
    GU("stats.entropy", 2, 1, "pk, qk?", "entropy", "base=None, " NANPOL, c_entropy, NULL, CORE_SCALAR, 0, 0, "0", NULL, 0,
       "Shannon entropy of pk, or the relative entropy D(pk||qk) (scipy.stats.entropy)."),
    GU("stats.differential_entropy", 1, 1, "values", "differential_entropy",
       "window_length=None, base=None, method=auto|vasicek|van es|correa|ebrahimi, " NANPOL, c_diffent, NULL, CORE_SCALAR, 0, 0, "0",
       NULL, 0, "Differential entropy estimated from a sample (scipy.stats.differential_entropy)."),
    GU("stats.false_discovery_control", 1, 1, "ps", "adjusted", "method=bh|by", c_fdc, NULL,
       CORE_LIKE0 | CORE_FLAT | CORE_INPLACE, 0, 0, "0", NULL, AX_SINGLE | AX_NOKEEP,
       "Benjamini-Hochberg / Benjamini-Yekutieli adjusted p-values (scipy.stats.false_discovery_control)."),

    ROUTINE("stats.describe", 6, "a, axis=0, ddof=1, bias=True, nan_policy='propagate'",
            "nobs, minmax, mean, variance, skewness, kurtosis", r_describe, NULL,
            "Descriptive statistics; minmax is returned as one array stacking (min, max) (scipy.stats.describe)."),
    ROUTINE("stats.mode", 2, "a, axis=0, nan_policy='propagate', keepdims=False", "mode, count", r_mode, NULL,
            "Modal (most common) value and its count; ties give the smallest value (scipy.stats.mode)."),
    ROUTINE("stats.tmean", 1, "a, limits=None, inclusive=None, axis=None, nan_policy='propagate', keepdims=False", "tmean",
            r_trimmed, &K_TMEAN, "Trimmed mean; inclusive=None means (True, True) (scipy.stats.tmean)."),
    ROUTINE("stats.tvar", 1, "a, limits=None, inclusive=None, axis=0, ddof=1, nan_policy='propagate', keepdims=False", "tvar",
            r_trimmed, &K_TVAR, "Trimmed variance; inclusive=None means (True, True) (scipy.stats.tvar)."),
    ROUTINE("stats.tstd", 1, "a, limits=None, inclusive=None, axis=0, ddof=1, nan_policy='propagate', keepdims=False", "tstd",
            r_trimmed, &K_TSTD, "Trimmed standard deviation; inclusive=None means (True, True) (scipy.stats.tstd)."),
    ROUTINE("stats.tsem", 1, "a, limits=None, inclusive=None, axis=0, ddof=1, nan_policy='propagate', keepdims=False", "tsem",
            r_trimmed, &K_TSEM, "Trimmed standard error of the mean; inclusive=None means (True, True) (scipy.stats.tsem)."),
    ROUTINE("stats.tmin", 1, "a, lowerlimit=None, axis=0, inclusive=True, nan_policy='propagate', keepdims=False", "tmin",
            r_trimmed, &K_TMIN, "Trimmed minimum (scipy.stats.tmin)."),
    ROUTINE("stats.tmax", 1, "a, upperlimit=None, axis=0, inclusive=True, nan_policy='propagate', keepdims=False", "tmax",
            r_trimmed, &K_TMAX, "Trimmed maximum (scipy.stats.tmax)."),
    ROUTINE("stats.trimboth", 1, "a, proportiontocut, axis=0", "trimmed", r_trim, NULL,
            "Slice off a proportion of items from both ends; the kept values come sorted (scipy.stats.trimboth)."),
    ROUTINE("stats.trim1", 1, "a, proportiontocut, tail='right', axis=0", "trimmed", r_trim, &ONE,
            "Slice off a proportion from one end; the kept values come sorted (scipy.stats.trim1)."),
    ROUTINE("stats.sigmaclip", 3, "a, low=4.0, high=4.0", "clipped, lower, upper", r_sigmaclip, NULL,
            "Iterative sigma-clipping of the data (scipy.stats.sigmaclip)."),
    ROUTINE("stats.scoreatpercentile", 1, "a, per, limit=None, interpolation_method='fraction', axis=None", "score",
            r_scoreatpercentile, NULL, "Score at the given percentile(s) of the data (scipy.stats.scoreatpercentile)."),
    ROUTINE("stats.percentileofscore", 1, "a, score, kind='rank', nan_policy='propagate'", "percentile", r_percentileofscore,
            NULL, "Percentile rank of score(s) relative to a (scipy.stats.percentileofscore)."),
    ROUTINE("stats.rankdata", 1, "a, method='average', axis=None, nan_policy='propagate'", "ranks", r_rankdata, NULL,
            "Ranks of the data, ties handled by method (scipy.stats.rankdata)."),
    ROUTINE("stats.tiecorrect", 1, "rankvals", "factor", r_tiecorrect, NULL,
            "Tie correction factor for Mann-Whitney U and Kruskal-Wallis H (scipy.stats.tiecorrect)."),
    ROUTINE("stats.cumfreq", 4, "a, numbins=10, defaultreallimits=None, weights=None", "cumcount, lowerlimit, binsize, extrapoints",
            r_freq, NULL, "Cumulative frequency histogram (scipy.stats.cumfreq)."),
    ROUTINE("stats.relfreq", 4, "a, numbins=10, defaultreallimits=None, weights=None", "frequency, lowerlimit, binsize, extrapoints",
            r_freq, &ONE, "Relative frequency histogram (scipy.stats.relfreq)."),
    ROUTINE("stats.obrientransform", 1,
            "sample1, sample2=None, sample3=None, sample4=None, sample5=None, sample6=None, sample7=None, sample8=None",
            "transformed", r_obrien, NULL, "O'Brien transform of one or more equal-length samples (scipy.stats.obrientransform)."),
    ROUTINE("stats.lmoment", 1, "sample, order=None, axis=0, sorted=False, standardize=True, nan_policy='propagate', keepdims=False",
            "lmoment", r_lmoment, NULL, "Sample L-moments, standardized as L-moment ratios by default (scipy.stats.lmoment)."),
    ROUTINE("stats.expectile", 1, "a, alpha=0.5, weights=None", "expectile", r_expectile, NULL,
            "Expectile of the data at level alpha (scipy.stats.expectile)."),
    ROUTINE("stats.iqr", 1, "x, axis=None, rng=None, scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False",
            "iqr", r_iqr, NULL, "Interquartile range; rng=None means (25, 75) (scipy.stats.iqr)."),
    ROUTINE("stats.median_abs_deviation", 1, "x, axis=0, center=None, scale=1.0, nan_policy='propagate', keepdims=False", "mad",
            r_mad, NULL, "Median absolute deviation; center must be None (the median) (scipy.stats.median_abs_deviation)."),
    ROUTINE("stats.gaussian_kde", 1, "dataset, points", "density", r_gaussian_kde, NULL,
            "Gaussian kernel density estimate (Scott's factor) evaluated at points (scipy.stats.gaussian_kde)."),
    ROUTINE("stats.ecdf", 2, "sample", "quantiles, probabilities", r_ecdf, NULL,
            "Empirical CDF: sorted unique values and cumulative proportions (scipy.stats.ecdf)."),
    ROUTINE("stats.directional_stats", 2, "samples", "mean_direction, mean_resultant_length", r_directional_stats, NULL,
            "Mean direction and resultant length of unit vectors (scipy.stats.directional_stats)."),
    ROUTINE("stats.probplot", 5, "x", "osm, osr, slope, intercept, r", r_probplot, NULL,
            "Probability plot data: order-statistic medians, ordered values, and the fit (scipy.stats.probplot)."),
    ROUTINE("stats.boxcox_normplot", 2, "x, la, lb, N=80", "lmbdas, ppcc", r_boxcox_normplot, NULL,
            "Box-Cox normality plot: lambdas and probability-plot correlations (scipy.stats.boxcox_normplot)."),
    ROUTINE("stats.yeojohnson_normplot", 2, "x, la, lb, N=80", "lmbdas, ppcc", r_yeojohnson_normplot, NULL,
            "Yeo-Johnson normality plot: lambdas and probability-plot correlations (scipy.stats.yeojohnson_normplot)."),
    ROUTINE("stats.multivariate_normal", 2, "mean, cov, x", "pdf, logpdf", r_multivariate_normal, NULL,
            "Multivariate normal pdf and log-pdf at each row of x (scipy.stats.multivariate_normal)."),
    ROUTINE("stats.dirichlet", 1, "alpha, x", "pdf", r_dirichlet, NULL,
            "Dirichlet pdf at each row of x (scipy.stats.dirichlet)."),
    ROUTINE("stats.multinomial", 1, "n, p, x", "pmf", r_multinomial, NULL,
            "Multinomial pmf at each row of x (scipy.stats.multinomial)."),
    ROUTINE("stats.multivariate_hypergeom", 1, "m, n, x", "pmf", r_multivariate_hypergeom, NULL,
            "Multivariate hypergeometric pmf at each row of x (scipy.stats.multivariate_hypergeom)."),
    ROUTINE("stats.multivariate_t", 2, "loc, shape, df, x", "pdf, logpdf", r_multivariate_t, NULL,
            "Multivariate Student-t pdf and log-pdf at each row of x (scipy.stats.multivariate_t)."),
    ROUTINE("stats.vonmises_fisher", 1, "mu, kappa, x", "pdf", r_vonmises_fisher, NULL,
            "Von Mises-Fisher pdf at each row of x (scipy.stats.vonmises_fisher)."),
    ROUTINE("stats.matrix_normal", 2, "mean, rowcov, colcov, x", "pdf, logpdf", r_matrix_normal, NULL,
            "Matrix-normal pdf and log-pdf at a matrix x (scipy.stats.matrix_normal)."),
    ROUTINE("stats.wishart", 2, "df, scale, x", "pdf, logpdf", r_wishart, NULL,
            "Wishart pdf and log-pdf at an SPD matrix x (scipy.stats.wishart)."),
    ROUTINE("stats.invwishart", 2, "df, scale, x", "pdf, logpdf", r_invwishart, NULL,
            "Inverse-Wishart pdf and log-pdf at an SPD matrix x (scipy.stats.invwishart)."),
    ROUTINE("stats.boxcox_llf", 1, "lmb, data, axis=0, keepdims=False, nan_policy='propagate'", "llf", r_llf, NULL,
            "Box-Cox log-likelihood (scipy.stats.boxcox_llf)."),
    ROUTINE("stats.yeojohnson_llf", 1, "lmb, data, axis=0, nan_policy='propagate', keepdims=False", "llf", r_llf, &ONE,
            "Yeo-Johnson log-likelihood (scipy.stats.yeojohnson_llf)."),
    ROUTINE("stats.energy_distance", 1, "u_values, v_values, u_weights=None, v_weights=None", "distance", r_cdf_distance, &P2,
            "Energy distance between two 1-D distributions (scipy.stats.energy_distance)."),
    ROUTINE("stats.wasserstein_distance", 1, "u_values, v_values, u_weights=None, v_weights=None", "distance", r_cdf_distance, &P1,
            "First Wasserstein distance between two 1-D distributions (scipy.stats.wasserstein_distance)."),
};

extern "C" const fn_table TSR_STATS_FN_DESC_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
