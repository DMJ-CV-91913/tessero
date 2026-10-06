/*
 * NumPy statistics, NaN-aware reductions, cumulative functions, searching and counting (Tessero\Np).
 *
 * Each function follows NumPy 2.4's algorithm step by step so results are NumPy's to the bit where NumPy
 * defines them: quantile methods (virtual index, gamma correction, the two-sided lerp), median as the mean of
 * the middle pair, nan-functions via NaN replacement and pairwise sums, digitize via searchsorted with NaN
 * ordered last. Registry kinds: generalised ufuncs over an axis (gufunc) and routines (np_sets.c).
 */
#include "fn.h"

#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------- ordering helpers (NaN sorts last, as NumPy) */

static inline int lt_nan(double a, double b) { return a < b || (b != b && a == a); }

static int cmp_d(const void *pa, const void *pb)
{
    const double a = *(const double *)pa, b = *(const double *)pb;
    if (lt_nan(a, b)) return -1;
    if (lt_nan(b, a)) return 1;
    return 0;
}

void np_sort_doubles(double *x, int64_t n) { if (n > 1) qsort(x, (size_t)n, sizeof(double), cmp_d); }

static int has_nan(const double *x, int64_t n)
{
    for (int64_t i = 0; i < n; i++) if (x[i] != x[i]) return 1;
    return 0;
}

static int64_t drop_nan(double *x, int64_t n)
{
    int64_t k = 0;
    for (int64_t i = 0; i < n; i++) if (x[i] == x[i]) x[k++] = x[i];
    return k;
}

/* ---------------------------------------------------------------- median */

static double median_sorted(const double *s, int64_t n)
{
    if (n == 0) return FN_NAN;
    const int64_t h = n / 2;
    if (n % 2 == 1) return s[h];
    return (s[h - 1] + s[h]) / 2.0;                      /* numpy: mean of the two middle values */
}

static int c_median(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                    const int64_t *nout, void *scratch, unsigned flags)
{
    double *v = (double *)x[0];
    int64_t m = n[0];
    if (ctx) m = drop_nan(v, m);                          /* nanmedian */
    else if (has_nan(v, m)) { out[0][0] = FN_NAN; return TSR_OK; }
    np_sort_doubles(v, m);
    out[0][0] = median_sorted(v, m);
    return TSR_OK;
}

/* ---------------------------------------------------------------- quantile / percentile */

enum { QM_LINEAR, QM_INVERTED_CDF, QM_AVERAGED_INVERTED_CDF, QM_CLOSEST_OBSERVATION, QM_INTERPOLATED_INVERTED_CDF,
       QM_HAZEN, QM_WEIBULL, QM_MEDIAN_UNBIASED, QM_NORMAL_UNBIASED, QM_LOWER, QM_HIGHER, QM_MIDPOINT, QM_NEAREST };

#define QMETHODS "linear|inverted_cdf|averaged_inverted_cdf|closest_observation|interpolated_inverted_cdf|hazen|weibull|median_unbiased|normal_unbiased|lower|higher|midpoint|nearest"

static inline double virtual_index(double n, double q, double alpha, double beta)
{
    return n * q + (alpha + q * (1 - alpha - beta)) - 1;
}

/* numpy's _discrete_interpolation_to_boundaries */
static int64_t discrete_index(double index, int closest)
{
    const double previous = floor(index);
    const double next = previous + 1;
    const double gamma = index - previous;
    int use_prev = closest ? (gamma == 0 && fmod(floor(index), 2.0) == 1.0) : (gamma == 0);
    if (closest && gamma == 0) {
        const double f = floor(index);
        use_prev = (f - 2.0 * floor(f / 2.0)) == 1.0;          /* numpy: floor(index) % 2 == 1 (floored modulo) */
    }
    double r = use_prev ? previous : next;
    if (r < 0) r = 0;
    return (int64_t)r;
}

static inline double take(const double *s, int64_t n, int64_t i) { return s[i < 0 ? n + i : i]; }

/* the discrete methods pick one sample: its index (negative: from the end), or INT64_MIN for the others */
static int64_t discrete_pick(int64_t n, double q, int method)
{
    const double N = (double)n;
    switch (method) {
    case QM_LOWER: return (int64_t)floor((N - 1) * q);
    case QM_HIGHER: return (int64_t)ceil((N - 1) * q);
    case QM_NEAREST: return (int64_t)nearbyint((N - 1) * q);
    case QM_INVERTED_CDF: return discrete_index(N * q - 1, 0);
    case QM_CLOSEST_OBSERVATION: return discrete_index(N * q - 1 - 0.5, 1);
    default: return INT64_MIN;
    }
}

/* one quantile of the sorted, NaN-free sample s[0..n) */
static double quantile_sorted(const double *s, int64_t n, double q, int method)
{
    const double N = (double)n;
    const int64_t pick = discrete_pick(n, q, method);
    if (pick != INT64_MIN) return take(s, n, pick);
    double v;
    switch (method) {
    case QM_LINEAR: v = (N - 1) * q; break;
    case QM_AVERAGED_INVERTED_CDF: v = N * q - 1; break;
    case QM_INTERPOLATED_INVERTED_CDF: v = virtual_index(N, q, 0, 1); break;
    case QM_HAZEN: v = virtual_index(N, q, 0.5, 0.5); break;
    case QM_WEIBULL: v = virtual_index(N, q, 0, 0); break;
    case QM_MEDIAN_UNBIASED: v = virtual_index(N, q, 1 / 3.0, 1 / 3.0); break;
    case QM_NORMAL_UNBIASED: v = virtual_index(N, q, 3 / 8.0, 3 / 8.0); break;
    case QM_MIDPOINT: v = 0.5 * (floor((N - 1) * q) + ceil((N - 1) * q)); break;
    default: return FN_NAN;
    }
    /* numpy's _get_indexes */
    double prev = floor(v), next = prev + 1;
    if (v >= N - 1) { prev = -1; next = -1; }
    if (v < 0) { prev = 0; next = 0; }
    if (v != v) { prev = -1; next = -1; }
    const int64_t ip = (int64_t)prev, in = (int64_t)next;
    double gamma = v - (double)ip;
    if (method == QM_AVERAGED_INVERTED_CDF) gamma = (gamma == 0) ? 0.5 : 1.0;
    else if (method == QM_MIDPOINT) gamma = (fmod(v, 1.0) == 0) ? 0.0 : 0.5;
    const double a = take(s, n, ip), b = take(s, n, in);
    /* numpy's _lerp: a + (b-a) t, or b - (b-a)(1-t) when t >= 0.5 */
    const double diff = b - a;
    double r = a + diff * gamma;
    if (gamma >= 0.5) r = b - diff * (1 - gamma);
    return r;
}

static int cmp_i64(const void *a, const void *b)
{
    const int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

typedef struct { double scale; int nan; } qctx;
/* numpy.quantile keeps an integer or bool dtype for inverted_cdf, closest_observation, lower, higher, nearest */
static const fn_keep Q_KEEP = {0, (1u << QM_INVERTED_CDF) | (1u << QM_CLOSEST_OBSERVATION) | (1u << QM_LOWER) | (1u << QM_HIGHER) | (1u << QM_NEAREST)};
static const qctx Q_PLAIN = {1.0, 0}, Q_PCT = {0.01, 0}, Q_NAN = {1.0, 1}, Q_NANPCT = {0.01, 1};

static int c_quantile(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                      const int64_t *nout, void *scratch, unsigned flags)
{
    const qctx *c = (const qctx *)ctx;
    double *v = (double *)x[0];
    int64_t m = n[0];
    const int method = (int)p[0];
    const int64_t nq = n[1];
    for (int64_t k = 0; k < nq; k++) {
        const double q = c->scale == 1.0 ? x[1][k] : x[1][k] / 100.0;
        if (!(q >= 0 && q <= 1)) {
            fn_set_error(c->scale == 1.0 ? "Quantiles must be in the range [0, 1]" : "Percentiles must be in the range [0, 100]");
            return TSR_EARG;
        }
    }
    if (flags & FN_INT64) {
        /* integer input with a discrete method: NumPy returns the sample itself, exactly (fn_keep) */
        int64_t *iv = (int64_t *)x[0];
        if (m == 0) {
            fn_set_error("cannot compute quantiles of an empty array");
            return TSR_EINDEX;
        }
        qsort(iv, (size_t)m, sizeof *iv, cmp_i64);
        for (int64_t k = 0; k < nq; k++) {
            const double q = c->scale == 1.0 ? x[1][k] : x[1][k] / 100.0;
            const int64_t pick = discrete_pick(m, q, method);
            ((int64_t *)out[0])[k] = iv[pick < 0 ? m + pick : pick];
        }
        return TSR_OK;
    }
    if (c->nan) m = drop_nan(v, m);
    else if (has_nan(v, m)) {
        for (int64_t k = 0; k < nq; k++) out[0][k] = FN_NAN;
        return TSR_OK;
    }
    if (m == 0) {
        if (c->nan) { for (int64_t k = 0; k < nq; k++) out[0][k] = FN_NAN; return TSR_OK; }
        fn_set_error("cannot compute quantiles of an empty array");
        return TSR_EINDEX;
    }
    np_sort_doubles(v, m);
    for (int64_t k = 0; k < nq; k++) {
        const double q = c->scale == 1.0 ? x[1][k] : x[1][k] / 100.0;
        out[0][k] = quantile_sorted(v, m, q, method);
    }
    return TSR_OK;
}

/* ---------------------------------------------------------------- simple reductions */

static int c_ptp(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                 const int64_t *nout, void *scratch, unsigned flags)
{
    if (n[0] == 0) { fn_set_error("zero-size array to reduction operation maximum which has no identity"); return TSR_EARG; }
    if (flags & FN_INT64) {                      /* integer input: max - min in int64, wrapping as NumPy */
        const int64_t *xi = (const int64_t *)x[0];
        int64_t lo = xi[0], hi = xi[0];
        for (int64_t i = 1; i < n[0]; i++) { if (xi[i] < lo) lo = xi[i]; if (xi[i] > hi) hi = xi[i]; }
        const int64_t d = (int64_t)((uint64_t)hi - (uint64_t)lo);
        memcpy(&out[0][0], &d, 8);
        return TSR_OK;
    }
    double lo = x[0][0], hi = x[0][0];
    for (int64_t i = 1; i < n[0] && lo == lo; i++) {
        const double v = x[0][i];
        if (v != v) { lo = hi = v; break; }
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    out[0][0] = hi - lo;
    return TSR_OK;
}

static int c_count_nonzero(const void *ctx, const double *const *x, const int64_t *n, const double *p,
                           double *const *out, const int64_t *nout, void *scratch, unsigned flags)
{
    int64_t c = 0;
    for (int64_t i = 0; i < n[0]; i++) c += x[0][i] != 0;
    out[0][0] = (double)c;
    return TSR_OK;
}

enum { NR_SUM, NR_PROD, NR_MIN, NR_MAX, NR_ARGMIN, NR_ARGMAX, NR_MEAN, NR_VAR, NR_STD };

static int c_nanreduce(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                       const int64_t *nout, void *scratch, unsigned flags)
{
    const int op = *(const int *)ctx;
    double *v = (double *)x[0];
    const int64_t m = n[0];
    if (flags & FN_INT64) {                      /* integer input (no NaN): NumPy's int64 arithmetic, wrapping */
        const int64_t *xi = (const int64_t *)x[0];
        uint64_t acc = op == NR_PROD ? 1u : 0u;
        int64_t r;
        switch (op) {
        case NR_SUM: for (int64_t i = 0; i < m; i++) acc += (uint64_t)xi[i]; r = (int64_t)acc; break;
        case NR_PROD: for (int64_t i = 0; i < m; i++) acc *= (uint64_t)xi[i]; r = (int64_t)acc; break;
        case NR_MIN: case NR_MAX:
            if (m == 0) { fn_set_error("zero-size array to reduction operation %s which has no identity", op == NR_MIN ? "fmin" : "fmax"); return TSR_EARG; }
            r = xi[0];
            for (int64_t i = 1; i < m; i++) if (op == NR_MIN ? xi[i] < r : xi[i] > r) r = xi[i];
            break;
        default: fn_set_error("internal: no int64 mode"); return TSR_EARG;
        }
        memcpy(&out[0][0], &r, 8);
        return TSR_OK;
    }
    switch (op) {
    case NR_SUM:
        for (int64_t i = 0; i < m; i++) if (v[i] != v[i]) v[i] = 0.0;
        out[0][0] = fn_sum(v, m, flags);
        return TSR_OK;
    case NR_PROD: {
        double r = 1.0;
        for (int64_t i = 0; i < m; i++) if (v[i] == v[i]) r *= v[i];
        out[0][0] = r;
        return TSR_OK;
    }
    case NR_MIN: case NR_MAX: {
        if (m == 0) { fn_set_error("zero-size array to reduction operation %s which has no identity", op == NR_MIN ? "fmin" : "fmax"); return TSR_EARG; }
        double best = FN_NAN;
        for (int64_t i = 0; i < m; i++) {
            const double t = v[i];
            if (t != t) continue;
            if (best != best || (op == NR_MIN ? t < best : t > best)) best = t;
        }
        out[0][0] = best;
        return TSR_OK;
    }
    case NR_ARGMIN: case NR_ARGMAX: {
        int64_t bi = -1;
        double best = 0;
        for (int64_t i = 0; i < m; i++) {
            const double t = v[i];
            if (t != t) continue;
            if (bi < 0 || (op == NR_ARGMIN ? t < best : t > best)) { best = t; bi = i; }
        }
        if (bi < 0) { fn_set_error("All-NaN slice encountered"); return TSR_EARG; }
        out[0][0] = (double)bi;
        return TSR_OK;
    }
    default: break;
    }
    /* nanmean: numpy replaces NaN with 0, counts the rest, sums (pairwise) and divides */
    int64_t cnt = 0;
    for (int64_t i = 0; i < m; i++) {
        if (v[i] != v[i]) v[i] = 0.0;
        else cnt++;
    }
    out[0][0] = fn_sum(v, m, flags) / (double)cnt;
    return TSR_OK;
}

/* nanvar needs the NaN mask: a dedicated core */
static int c_nanvar(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                    const int64_t *nout, void *scratch, unsigned flags)
{
    const int is_std = ctx != NULL;
    double *v = (double *)x[0];
    const int64_t m = n[0];
    const double ddof = p[0];
    unsigned char *mask = (unsigned char *)scratch;
    int64_t cnt = 0;
    for (int64_t i = 0; i < m; i++) {
        mask[i] = v[i] != v[i];
        if (mask[i]) v[i] = 0.0;
        else cnt++;
    }
    const double avg = fn_sum(v, m, flags) / (double)cnt;
    for (int64_t i = 0; i < m; i++) {
        if (!mask[i]) {
            const double d = v[i] - avg;
            v[i] = d * d;
        }
    }
    const double s = fn_sum(v, m, flags);
    const double dof = (double)cnt - ddof;
    double var = s / dof;
    if (dof <= 0) var = FN_NAN;
    out[0][0] = is_std ? sqrt(var) : var;
    return TSR_OK;
}

static int64_t nanvar_scratch(const int64_t *n, const double *p) { (void)p; return n[0] + 64; }

/* ---------------------------------------------------------------- cumulative */

typedef struct { int prod; int nan; } cumctx;
static const cumctx CUM_NANSUM = {0, 1}, CUM_NANPROD = {1, 1}, CUM_SUM = {0, 0}, CUM_PROD = {1, 0};

static int c_cumulative(const void *ctx, const double *const *x, const int64_t *n, const double *p,
                        double *const *out, const int64_t *nout, void *scratch, unsigned flags)
{
    const cumctx *c = (const cumctx *)ctx;
    const int initial = c->nan ? 0 : (p && p[0] != 0);   /* truthiness, as fn_core's CORE_ADD */
    if (flags & FN_INT64) {                      /* integer input: running sum/product in int64, wrapping */
        const int64_t *xi = (const int64_t *)x[0];
        int64_t *oi = (int64_t *)out[0];
        uint64_t acc = c->prod ? 1u : 0u;
        int64_t k = 0;
        if (initial) oi[k++] = (int64_t)acc;
        for (int64_t i = 0; i < n[0]; i++) {
            acc = i == 0 ? (uint64_t)xi[i] : (c->prod ? acc * (uint64_t)xi[i] : acc + (uint64_t)xi[i]);
            oi[k++] = (int64_t)acc;
        }
        return TSR_OK;
    }
    double acc = c->prod ? 1.0 : 0.0;
    int64_t k = 0;
    if (initial) out[0][k++] = acc;
    for (int64_t i = 0; i < n[0]; i++) {
        double t = x[0][i];
        if (c->nan && t != t) t = c->prod ? 1.0 : 0.0;
        if (i == 0) acc = t;                           /* numpy's accumulate starts from the first element */
        else acc = c->prod ? acc * t : acc + t;
        out[0][k++] = acc;
    }
    return TSR_OK;
}

/* ---------------------------------------------------------------- average */

static int c_average(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                     const int64_t *nout, void *scratch, unsigned flags)
{
    double *a = (double *)x[0];
    const double *w = x[1];
    const int64_t m = n[0];
    if (n[1] == 1 && m != 1) {                            /* no weights (the binding passes a scalar one) */
        out[0][0] = fn_sum(a, m, flags) / (double)m;
        return TSR_OK;
    }
    if (n[1] != m) { fn_set_error("Length of weights not compatible with specified axis."); return TSR_ESHAPE; }
    double *ws = (double *)scratch;
    memcpy(ws, w, (size_t)m * sizeof(double));
    const double scl = fn_sum(ws, m, flags);
    if (scl == 0.0) { fn_set_error("Weights sum to zero, can't be normalized"); return TSR_EARG; }
    for (int64_t i = 0; i < m; i++) a[i] = a[i] * w[i];
    out[0][0] = fn_sum(a, m, flags) / scl;
    return TSR_OK;
}

static int64_t average_scratch(const int64_t *n, const double *p) { (void)p; return n[1] * 8 + 64; }

/* ---------------------------------------------------------------- searching */

/* numpy.searchsorted on a sorted run: side 0 left, 1 right; NaN ordered last */
static int64_t bsearch_side(const double *a, int64_t n, double v, int right)
{
    int64_t lo = 0, hi = n;
    while (lo < hi) {
        const int64_t mid = lo + (hi - lo) / 2;
        const int go_right = right ? !lt_nan(v, a[mid]) : lt_nan(a[mid], v);
        if (go_right) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static int c_searchsorted(const void *ctx, const double *const *x, const int64_t *n, const double *p,
                          double *const *out, const int64_t *nout, void *scratch, unsigned flags)
{
    const int right = (int)p[0];
    for (int64_t k = 0; k < n[1]; k++) out[0][k] = (double)bsearch_side(x[0], n[0], x[1][k], right);
    return TSR_OK;
}

static int c_digitize(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                      const int64_t *nout, void *scratch, unsigned flags)
{
    const double *bins = x[1];
    const int64_t nb = n[1];
    const int right = p[0] != 0 && p[0] == p[0];
    /* numpy's _monotonicity: 1 non-decreasing, -1 non-increasing, 0 neither (NaN compares false) */
    int mono = 1;
    if (nb >= 2) {
        int inc = 1, dec = 1;
        for (int64_t i = 0; i + 1 < nb; i++) {
            if (!(bins[i] <= bins[i + 1])) inc = 0;
            if (!(bins[i] >= bins[i + 1])) dec = 0;
        }
        mono = inc ? 1 : (dec ? -1 : 0);
    }
    if (mono == 0) { fn_set_error("bins must be monotonically increasing or decreasing"); return TSR_EARG; }
    const int side_right = !right;                        /* numpy: side = 'left' if right else 'right' */
    if (mono == 1) {
        for (int64_t k = 0; k < n[0]; k++) out[0][k] = (double)bsearch_side(bins, nb, x[0][k], side_right);
    } else {
        double *rev = (double *)scratch;
        for (int64_t i = 0; i < nb; i++) rev[i] = bins[nb - 1 - i];
        for (int64_t k = 0; k < n[0]; k++) out[0][k] = (double)(nb - bsearch_side(rev, nb, x[0][k], side_right));
    }
    return TSR_OK;
}

static int64_t digitize_scratch(const int64_t *n, const double *p) { (void)p; return n[1] * 8 + 64; }

static int c_isin(const void *ctx, const double *const *x, const int64_t *n, const double *p, double *const *out,
                  const int64_t *nout, void *scratch, unsigned flags)
{
    const int invert = p[0] != 0 && p[0] == p[0];
    double *t = (double *)scratch;
    memcpy(t, x[1], (size_t)n[1] * sizeof(double));
    const int64_t m = n[1];
    np_sort_doubles(t, m);
    for (int64_t k = 0; k < n[0]; k++) {
        const double v = x[0][k];
        int found = 0;
        if (v == v) {
            const int64_t i = bsearch_side(t, m, v, 0);
            found = i < m && t[i] == v;
        }
        out[0][k] = (found ^ invert) ? 1.0 : 0.0;
    }
    return TSR_OK;
}

static int64_t isin_scratch(const int64_t *n, const double *p) { (void)p; return n[1] * 8 + 64; }

/* ---------------------------------------------------------------- partition */

static int c_partition(const void *ctx, const double *const *x, const int64_t *n, const double *p,
                       double *const *out, const int64_t *nout, void *scratch, unsigned flags)
{
    /* a full sort is a valid partition for every kth; NumPy leaves the order within partitions unspecified */
    const int arg = ctx != NULL;
    const int64_t m = n[0];
    for (int64_t k = 0; k < n[1]; k++) {
        double kth = x[1][k];
        if (kth < 0) kth += (double)m;
        if (!(kth >= 0 && kth < (double)m)) { fn_set_error("kth(=%g) out of bounds (%lld)", x[1][k], (long long)m); return TSR_EARG; }
    }
    if (!arg && (flags & FN_INT64)) {            /* integer input: sort the int64 values themselves */
        memcpy(out[0], x[0], (size_t)m * sizeof(int64_t));
        qsort(out[0], (size_t)m, sizeof(int64_t), cmp_i64);
        return TSR_OK;
    }
    if (!arg) {
        memcpy(out[0], x[0], (size_t)m * sizeof(double));
        np_sort_doubles(out[0], m);
        return TSR_OK;
    }
    /* argpartition: stable argsort (value, index) */
    double *pairs = (double *)scratch;
    for (int64_t i = 0; i < m; i++) { pairs[2 * i] = x[0][i]; pairs[2 * i + 1] = (double)i; }
    extern void np_argsort_pairs(double *pairs, int64_t n);
    np_argsort_pairs(pairs, m);
    for (int64_t i = 0; i < m; i++) out[0][i] = pairs[2 * i + 1];
    return TSR_OK;
}

static int64_t partition_scratch(const int64_t *n, const double *p) { (void)p; return n[0] * 16 + 64; }

/* stable sort of (value, index) pairs, NaN last: bottom-up merge sort */
void np_argsort_pairs(double *pairs, int64_t n)
{
    if (n < 2) return;
    double *tmp = (double *)tsr_alloc(n * 16);
    if (!tmp) {                                           /* fall back to insertion sort (stable) */
        for (int64_t i = 1; i < n; i++) {
            const double v = pairs[2 * i], ix = pairs[2 * i + 1];
            int64_t j = i - 1;
            while (j >= 0 && lt_nan(v, pairs[2 * j])) { pairs[2 * (j + 1)] = pairs[2 * j]; pairs[2 * (j + 1) + 1] = pairs[2 * j + 1]; j--; }
            pairs[2 * (j + 1)] = v;
            pairs[2 * (j + 1) + 1] = ix;
        }
        return;
    }
    double *src = pairs, *dst = tmp;
    for (int64_t w = 1; w < n; w *= 2) {
        for (int64_t lo = 0; lo < n; lo += 2 * w) {
            const int64_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n;
            int64_t i = lo, j = mid, k = lo;
            while (i < mid && j < hi) {
                if (lt_nan(src[2 * j], src[2 * i])) { dst[2 * k] = src[2 * j]; dst[2 * k + 1] = src[2 * j + 1]; j++; }
                else { dst[2 * k] = src[2 * i]; dst[2 * k + 1] = src[2 * i + 1]; i++; }
                k++;
            }
            while (i < mid) { dst[2 * k] = src[2 * i]; dst[2 * k + 1] = src[2 * i + 1]; i++; k++; }
            while (j < hi) { dst[2 * k] = src[2 * j]; dst[2 * k + 1] = src[2 * j + 1]; j++; k++; }
        }
        double *t = src; src = dst; dst = t;
    }
    if (src != pairs) memcpy(pairs, src, (size_t)n * 16);
    tsr_free(tmp, n * 16);
}

/* ---------------------------------------------------------------- registry */

static const int OP_SUM = NR_SUM, OP_PROD = NR_PROD, OP_MIN = NR_MIN, OP_MAX = NR_MAX, OP_ARGMIN = NR_ARGMIN,
                 OP_ARGMAX = NR_ARGMAX, OP_MEAN = NR_MEAN;
static const int IS_STD = 1;

/* running operations and partitions: 1-D output along the axis, in its place; axis=None ravels */
#define CUM_CORE (CORE_LIKE0 | CORE_FLAT | CORE_INPLACE | CORE_INTLIKE | CORE_INT64)

const fn_def NP_STATS[] = {
    GUFUNC("np.median", 1, 1, "a", "median", "", c_median, NULL, CORE_SCALAR, 0, 0, 0, 0, 0, "None", 0, 1, NULL,
           "Median along the given axes (numpy.median)."),
    GUFUNC("np.nanmedian", 1, 1, "a", "median", "", c_median, NULL, CORE_SCALAR, 0, 0, 0, 0, 0, "None", 0, 1, &IS_STD,
           "Median ignoring NaNs (numpy.nanmedian)."),
    GUFUNC_K("np.quantile", 2, 1, "a, q", "quantile", "method=" QMETHODS, c_quantile, NULL, CORE_LIKE1 | CORE_FIRST | CORE_INT64, 0, 0, 0, 0, 0,
           "None", 0, 1, &Q_PLAIN, &Q_KEEP, "q-th quantiles along the given axes, q in [0, 1] (numpy.quantile)."),
    GUFUNC_K("np.percentile", 2, 1, "a, q", "percentile", "method=" QMETHODS, c_quantile, NULL, CORE_LIKE1 | CORE_FIRST | CORE_INT64, 0, 0, 0, 0, 0,
           "None", 0, 1, &Q_PCT, &Q_KEEP, "q-th percentiles along the given axes, q in [0, 100] (numpy.percentile)."),
    GUFUNC_K("np.nanquantile", 2, 1, "a, q", "quantile", "method=" QMETHODS, c_quantile, NULL, CORE_LIKE1 | CORE_FIRST | CORE_INT64, 0, 0, 0, 0, 0,
           "None", 0, 1, &Q_NAN, &Q_KEEP, "Quantiles ignoring NaNs (numpy.nanquantile)."),
    GUFUNC_K("np.nanpercentile", 2, 1, "a, q", "percentile", "method=" QMETHODS, c_quantile, NULL, CORE_LIKE1 | CORE_FIRST | CORE_INT64, 0, 0, 0, 0, 0,
           "None", 0, 1, &Q_NANPCT, &Q_KEEP, "Percentiles ignoring NaNs (numpy.nanpercentile)."),
    GUFUNC("np.ptp", 1, 1, "a", "ptp", "", c_ptp, NULL, CORE_SCALAR | CORE_INTLIKE | CORE_INT64 | CORE_NOBOOL, 0, 0, 0, 0, 0, "None", 0, 1, NULL,
           "Range of values, maximum - minimum (numpy.ptp)."),
    GUFUNC("np.count_nonzero", 1, 1, "a", "count", "", c_count_nonzero, NULL, CORE_SCALAR, 0, 0, 0, 1, 0, "None", 0, 1, NULL,
           "Number of non-zero values (numpy.count_nonzero)."),
    GUFUNC("np.nansum", 1, 1, "a", "sum", "", c_nanreduce, NULL, CORE_SCALAR | CORE_INTLIKE | CORE_INT64, 0, 0, 0, 0, 0, "None", 0, 1, &OP_SUM,
           "Sum treating NaN as zero (numpy.nansum)."),
    GUFUNC("np.nanprod", 1, 1, "a", "prod", "", c_nanreduce, NULL, CORE_SCALAR | CORE_INTLIKE | CORE_INT64, 0, 0, 0, 0, 0, "None", 0, 1, &OP_PROD,
           "Product treating NaN as one (numpy.nanprod)."),
    GUFUNC("np.nanmin", 1, 1, "a", "min", "", c_nanreduce, NULL, CORE_SCALAR | CORE_INTLIKE | CORE_INT64, 0, 0, 0, 0, 0, "None", 0, 1, &OP_MIN,
           "Minimum ignoring NaN; NaN for an all-NaN slice (numpy.nanmin)."),
    GUFUNC("np.nanmax", 1, 1, "a", "max", "", c_nanreduce, NULL, CORE_SCALAR | CORE_INTLIKE | CORE_INT64, 0, 0, 0, 0, 0, "None", 0, 1, &OP_MAX,
           "Maximum ignoring NaN; NaN for an all-NaN slice (numpy.nanmax)."),
    GUFUNC_AX("np.nanargmin", 1, 1, "a", "index", "", c_nanreduce, NULL, CORE_SCALAR, 1, 0, "None", 1, &OP_ARGMIN, AX_SINGLE,
           "Index of the minimum ignoring NaN (numpy.nanargmin)."),
    GUFUNC_AX("np.nanargmax", 1, 1, "a", "index", "", c_nanreduce, NULL, CORE_SCALAR, 1, 0, "None", 1, &OP_ARGMAX, AX_SINGLE,
           "Index of the maximum ignoring NaN (numpy.nanargmax)."),
    GUFUNC("np.nanmean", 1, 1, "a", "mean", "", c_nanreduce, NULL, CORE_SCALAR, 0, 0, 0, 0, 0, "None", 0, 1, &OP_MEAN,
           "Mean ignoring NaN (numpy.nanmean)."),
    GUFUNC("np.nanvar", 1, 1, "a", "var", "ddof=0", c_nanvar, nanvar_scratch, CORE_SCALAR, 0, 0, 0, 0, 0, "None", 0, 1, NULL,
           "Variance ignoring NaN (numpy.nanvar)."),
    GUFUNC("np.nanstd", 1, 1, "a", "std", "ddof=0", c_nanvar, nanvar_scratch, CORE_SCALAR, 0, 0, 0, 0, 0, "None", 0, 1, &IS_STD,
           "Standard deviation ignoring NaN (numpy.nanstd)."),
    GUFUNC_AX("np.nancumsum", 1, 1, "a", "cumsum", "", c_cumulative, NULL, CUM_CORE, 0, 0, "None", 1, &CUM_NANSUM, AX_SINGLE | AX_NOKEEP,
           "Cumulative sum treating NaN as zero (numpy.nancumsum)."),
    GUFUNC_AX("np.nancumprod", 1, 1, "a", "cumprod", "", c_cumulative, NULL, CUM_CORE, 0, 0, "None", 1, &CUM_NANPROD, AX_SINGLE | AX_NOKEEP,
           "Cumulative product treating NaN as one (numpy.nancumprod)."),
    GUFUNC_AX("np.cumulative_sum", 1, 1, "x", "cumsum", "include_initial=False", c_cumulative, NULL, CUM_CORE | CORE_ADD(0), 0, 0,
           "None", 1, &CUM_SUM, AX_SINGLE | AX_NOKEEP | AX_NONE_1D, "Cumulative sum, optionally starting with 0 (numpy.cumulative_sum)."),
    GUFUNC_AX("np.cumulative_prod", 1, 1, "x", "cumprod", "include_initial=False", c_cumulative, NULL, CUM_CORE | CORE_ADD(0), 0, 0,
           "None", 1, &CUM_PROD, AX_SINGLE | AX_NOKEEP | AX_NONE_1D, "Cumulative product, optionally starting with 1 (numpy.cumulative_prod)."),
    GUFUNC_AX("np.average", 2, 1, "a, weights?", "average", "", c_average, average_scratch, CORE_SCALAR, 0, 0,
              "None", 2, NULL, AX_WEIGHTS, "Weighted average along the given axes (numpy.average)."),
    GUFUNC_R("np.searchsorted", 2, 1, "a, v", "indices", "side=left|right", c_searchsorted, NULL, CORE_LIKE1, 0, 0, 0, 1, 0, "hidden", 0, 0, NULL,
           AX_IN0_1D,
           "Indices where v would be inserted into the sorted array a (numpy.searchsorted)."),
    GUFUNC("np.digitize", 2, 1, "x, bins", "indices", "right=False", c_digitize, digitize_scratch, CORE_LIKE0, 0, 0, 0, 1, 0, "hidden", 0, 0, NULL,
           "Indices of the bins to which each value belongs (numpy.digitize)."),
    GUFUNC("np.isin", 2, 1, "element, test_elements", "mask", "invert=False", c_isin, isin_scratch, CORE_LIKE0, 0, 0, 0, 0, 1, "hidden", 0, 0, NULL,
           "Whether each element is in test_elements (numpy.isin)."),
    GUFUNC_AX("np.partition", 2, 1, "a, kth", "partitioned", "", c_partition, NULL, CUM_CORE, 0, 0, "-1", 1, NULL, AX_SINGLE | AX_NOKEEP,
           "Partially sorted copy: the kth element in its sorted place (numpy.partition)."),
    GUFUNC_AX("np.argpartition", 2, 1, "a, kth", "indices", "", c_partition, partition_scratch, CORE_LIKE0 | CORE_FLAT | CORE_INPLACE, 1, 0, "-1", 1, &IS_STD,
           AX_SINGLE | AX_NOKEEP,
           "Indices that partition the array (numpy.argpartition)."),
};
const fn_table TSR_NP_STATS_TABLE = {NP_STATS, (int)(sizeof NP_STATS / sizeof NP_STATS[0])};
