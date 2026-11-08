/* scipy.spatial.distance: pdist, cdist, squareform (real float64).
 *
 * pdist(X, metric, p)      condensed pairwise distances between the rows of X (m x n) -> vector of m(m-1)/2
 * cdist(XA, XB, metric, p) distances between every pair of rows of XA (mA x n) and XB (mB x n) -> mA x mB
 * squareform(X)            a condensed vector <-> a square symmetric matrix with a zero diagonal
 *
 * The metric is a string (default euclidean); p is the Minkowski order (default 2). The computation is the
 * plain O(pairs * n) one, which is enough for the parity fixtures. */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <lapacke.h>

enum { D_EUCLID, D_SQEUCLID, D_CITYBLOCK, D_CHEBYSHEV, D_COSINE, D_CORRELATION, D_HAMMING, D_BRAYCURTIS, D_CANBERRA, D_MINKOWSKI };

static int parse_metric(const tsr_arg *a, int *metric)
{
    if (!a || a->kind == 0) { *metric = D_EUCLID; return TSR_OK; }
    if (a->kind != 2 || !a->str) { fn_set_error("distance: metric must be a string"); return TSR_EARG; }
    const struct { const char *n; int m; } T[] = {
        {"euclidean", D_EUCLID}, {"sqeuclidean", D_SQEUCLID}, {"cityblock", D_CITYBLOCK}, {"manhattan", D_CITYBLOCK},
        {"chebyshev", D_CHEBYSHEV}, {"chebychev", D_CHEBYSHEV}, {"cosine", D_COSINE}, {"correlation", D_CORRELATION},
        {"hamming", D_HAMMING}, {"braycurtis", D_BRAYCURTIS}, {"canberra", D_CANBERRA}, {"minkowski", D_MINKOWSKI},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        if (strcmp(a->str, T[i].n) == 0) { *metric = T[i].m; return TSR_OK; }
    fn_set_error("distance: metric '%s' is not supported", a->str);
    return TSR_EARG;
}

static double parse_p(const tsr_arg *a)
{
    if (!a || a->kind == 0) return 2.0;
    if (a->kind == 1) return (a->flags & 1) ? (double)a->ival : a->num;
    if (a->kind == 3 && a->arr.ndim == 0) { /* a 0-d array */ }
    return 2.0;
}

static double vdist(const double *u, const double *v, int64_t n, int metric, double p)
{
    double s = 0.0, m = 0.0;
    switch (metric) {
    case D_EUCLID:    for (int64_t i = 0; i < n; i++) { double d = u[i] - v[i]; s += d * d; } return sqrt(s);
    case D_SQEUCLID:  for (int64_t i = 0; i < n; i++) { double d = u[i] - v[i]; s += d * d; } return s;
    case D_CITYBLOCK: for (int64_t i = 0; i < n; i++) s += fabs(u[i] - v[i]); return s;
    case D_CHEBYSHEV: for (int64_t i = 0; i < n; i++) { double d = fabs(u[i] - v[i]); if (d > m) m = d; } return m;
    case D_MINKOWSKI: for (int64_t i = 0; i < n; i++) s += pow(fabs(u[i] - v[i]), p); return pow(s, 1.0 / p);
    case D_HAMMING:   for (int64_t i = 0; i < n; i++) if (u[i] != v[i]) s += 1.0; return n ? s / (double)n : 0.0;
    case D_BRAYCURTIS: { double den = 0.0; for (int64_t i = 0; i < n; i++) { s += fabs(u[i] - v[i]); den += fabs(u[i] + v[i]); } return s / den; }
    case D_CANBERRA:  for (int64_t i = 0; i < n; i++) { double den = fabs(u[i]) + fabs(v[i]); if (den > 0.0) s += fabs(u[i] - v[i]) / den; } return s;
    case D_COSINE: {
        double uv = 0.0, uu = 0.0, vv = 0.0;
        for (int64_t i = 0; i < n; i++) { uv += u[i] * v[i]; uu += u[i] * u[i]; vv += v[i] * v[i]; }
        return 1.0 - uv / (sqrt(uu) * sqrt(vv));
    }
    case D_CORRELATION: {
        double mu = 0.0, mv = 0.0;
        for (int64_t i = 0; i < n; i++) { mu += u[i]; mv += v[i]; }
        mu /= (double)n; mv /= (double)n;
        double uv = 0.0, uu = 0.0, vv = 0.0;
        for (int64_t i = 0; i < n; i++) { double a = u[i] - mu, b = v[i] - mv; uv += a * b; uu += a * a; vv += b * b; }
        return 1.0 - uv / (sqrt(uu) * sqrt(vv));
    }
    default: return NAN;
    }
}

/* Individual pairwise distance functions distance.<metric>(u, v[, p]) -> scalar (scipy.spatial.distance.*). */
static int r_dmetric(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int metric = ctx ? *(const int *)ctx : D_EUCLID;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("distance: u and v must be arrays"); return TSR_EARG; }
    int64_t nu, nv;
    double *u = fn_arg_doubles(&args[0], &nu); if (!u) return TSR_ENOMEM;
    double *v = fn_arg_doubles(&args[1], &nv); if (!v) { fn_free_doubles(u, nu); return TSR_ENOMEM; }
    if (nu != nv) { fn_free_doubles(u, nu); fn_free_doubles(v, nv); fn_set_error("distance: u and v must have the same length"); return TSR_EARG; }
    const double p = (metric == D_MINKOWSKI) ? parse_p(nargs > 2 ? &args[2] : NULL) : 2.0;
    const double d = vdist(u, v, nu, metric, p);
    fn_free_doubles(u, nu); fn_free_doubles(v, nv);
    fn_result_num(res, d);
    return TSR_OK;
}

static const int MET_EUCLID = D_EUCLID, MET_SQEUCLID = D_SQEUCLID, MET_CITYBLOCK = D_CITYBLOCK,
    MET_CHEBYSHEV = D_CHEBYSHEV, MET_COSINE = D_COSINE, MET_CORRELATION = D_CORRELATION,
    MET_HAMMING = D_HAMMING, MET_BRAYCURTIS = D_BRAYCURTIS, MET_CANBERRA = D_CANBERRA, MET_MINKOWSKI = D_MINKOWSKI;

/* Boolean (set) distance functions: counts ntt/ntf/nft/nff of agreeing/disagreeing TRUE/FALSE components. */
enum { B_DICE, B_JACCARD, B_ROGERS, B_RUSSELL, B_SOKALM, B_SOKALS, B_YULE, B_KULCZ };
static const int BM_DICE = B_DICE, BM_JACCARD = B_JACCARD, BM_ROGERS = B_ROGERS, BM_RUSSELL = B_RUSSELL,
    BM_SOKALS = B_SOKALS, BM_YULE = B_YULE;

static int r_bmetric(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nargs; (void)nres;
    const int which = ctx ? *(const int *)ctx : B_JACCARD;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("distance: u and v must be arrays"); return TSR_EARG; }
    int64_t nu, nv;
    double *u = fn_arg_doubles(&args[0], &nu); if (!u) return TSR_ENOMEM;
    double *v = fn_arg_doubles(&args[1], &nv); if (!v) { fn_free_doubles(u, nu); return TSR_ENOMEM; }
    if (nu != nv) { fn_free_doubles(u, nu); fn_free_doubles(v, nv); fn_set_error("distance: u and v must have the same length"); return TSR_EARG; }
    int64_t ntt = 0, ntf = 0, nft = 0, nff = 0;
    for (int64_t i = 0; i < nu; i++) {
        const int bu = u[i] != 0.0, bv = v[i] != 0.0;
        if (bu && bv) ntt++; else if (bu) ntf++; else if (bv) nft++; else nff++;
    }
    fn_free_doubles(u, nu); fn_free_doubles(v, nv);
    const double R = 2.0 * (double)(ntf + nft);
    double d = 0.0, den;
    switch (which) {
    case B_DICE:    den = 2.0 * (double)ntt + (double)(ntf + nft); d = den ? (double)(ntf + nft) / den : 0.0; break;
    case B_JACCARD: den = (double)(ntt + ntf + nft);               d = den ? (double)(ntf + nft) / den : 0.0; break;
    case B_ROGERS:  d = R / ((double)(ntt + nff) + R); break;
    case B_RUSSELL: d = nu ? (double)(nu - ntt) / (double)nu : 0.0; break;
    case B_SOKALM:  d = R / ((double)(ntt + nff) + R); break;
    case B_SOKALS:  d = R / ((double)ntt + R); break;
    case B_YULE:    den = (double)(ntt * nff + ntf * nft); d = den ? (2.0 * (double)ntf * (double)nft) / den : 0.0; break;
    case B_KULCZ:   d = (double)ntt / (double)(ntf + nft); break;   /* may be inf (scipy's behaviour) */
    default: d = NAN;
    }
    fn_result_num(res, d);
    return TSR_OK;
}

/* mahalanobis(u, v, VI): sqrt((u-v) . VI . (u-v)), VI the inverse covariance (scipy.spatial.distance.mahalanobis). */
static int r_mahalanobis(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3 || args[2].kind != 3) { fn_set_error("mahalanobis: u, v, VI must be arrays"); return TSR_EARG; }
    int64_t nu, nv, nvi;
    double *u = fn_arg_doubles(&args[0], &nu); if (!u) return TSR_ENOMEM;
    double *v = fn_arg_doubles(&args[1], &nv); if (!v) { fn_free_doubles(u, nu); return TSR_ENOMEM; }
    double *vi = fn_arg_doubles(&args[2], &nvi); if (!vi) { fn_free_doubles(u, nu); fn_free_doubles(v, nv); return TSR_ENOMEM; }
    int rc = TSR_OK; double d = 0.0;
    if (nu != nv || nvi != nu * nu) { fn_set_error("mahalanobis: shapes of u, v and VI are inconsistent"); rc = TSR_EARG; }
    else {
        double *delta = (double *)malloc((size_t)(nu > 0 ? nu : 1) * sizeof(double));
        if (!delta) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < nu; i++) delta[i] = u[i] - v[i];
            double acc = 0.0;
            for (int64_t i = 0; i < nu; i++) { double row = 0.0; for (int64_t j = 0; j < nu; j++) row += vi[i * nu + j] * delta[j]; acc += delta[i] * row; }
            d = sqrt(acc);
            free(delta);
        }
    }
    fn_free_doubles(u, nu); fn_free_doubles(v, nv); fn_free_doubles(vi, nvi);
    if (rc < 0) return rc;
    fn_result_num(res, d);
    return TSR_OK;
}

/* seuclidean(u, v, V): standardized Euclidean, sqrt(sum((u-v)^2 / V)) (scipy.spatial.distance.seuclidean). */
static int r_seuclidean(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3 || args[2].kind != 3) { fn_set_error("seuclidean: u, v, V must be arrays"); return TSR_EARG; }
    int64_t nu, nv, nV;
    double *u = fn_arg_doubles(&args[0], &nu); if (!u) return TSR_ENOMEM;
    double *v = fn_arg_doubles(&args[1], &nv); if (!v) { fn_free_doubles(u, nu); return TSR_ENOMEM; }
    double *V = fn_arg_doubles(&args[2], &nV); if (!V) { fn_free_doubles(u, nu); fn_free_doubles(v, nv); return TSR_ENOMEM; }
    int rc = TSR_OK; double d = 0.0;
    if (nu != nv || nV != nu) { fn_set_error("seuclidean: u, v and V must have the same length"); rc = TSR_EARG; }
    else { double acc = 0.0; for (int64_t i = 0; i < nu; i++) { double delta = u[i] - v[i]; acc += delta * delta / V[i]; } d = sqrt(acc); }
    fn_free_doubles(u, nu); fn_free_doubles(v, nv); fn_free_doubles(V, nV);
    if (rc < 0) return rc;
    fn_result_num(res, d);
    return TSR_OK;
}

/* jensenshannon(p, q, base=None): the Jensen-Shannon distance = sqrt(JS divergence) (scipy.spatial.distance.jensenshannon). */
static int r_jensenshannon(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("jensenshannon: p and q must be arrays"); return TSR_EARG; }
    int64_t np_, nq;
    double *p = fn_arg_doubles(&args[0], &np_); if (!p) return TSR_ENOMEM;
    double *q = fn_arg_doubles(&args[1], &nq); if (!q) { fn_free_doubles(p, np_); return TSR_ENOMEM; }
    int rc = TSR_OK; double d = 0.0;
    if (np_ != nq) { fn_set_error("jensenshannon: p and q must have the same length"); rc = TSR_EARG; }
    else {
        double sp = 0.0, sq = 0.0;
        for (int64_t i = 0; i < np_; i++) { sp += p[i]; sq += q[i]; }
        double js = 0.0;
        for (int64_t i = 0; i < np_; i++) {
            const double pn = p[i] / sp, qn = q[i] / sq, mn = (pn + qn) / 2.0;
            if (pn > 0.0) js += pn * log(pn / mn);
            if (qn > 0.0) js += qn * log(qn / mn);
        }
        js *= 0.5;
        if (nargs > 2 && args[2].kind == 1) js /= log(args[2].num);
        d = sqrt(js);
    }
    fn_free_doubles(p, np_); fn_free_doubles(q, nq);
    if (rc < 0) return rc;
    fn_result_num(res, d);
    return TSR_OK;
}

/* num_obs_dm(d): the number of observations in a square redundant distance matrix (scipy.spatial.distance). */
static int r_num_obs_dm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("num_obs_dm: input must be a square 2-D array"); return TSR_EARG; }
    fn_result_int(res, args[0].arr.shape[0]);
    return TSR_OK;
}

/* num_obs_y(Y): the number of observations for a condensed distance vector of length n(n-1)/2 (scipy...). */
static int r_num_obs_y(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("num_obs_y: input must be a 1-D condensed vector"); return TSR_EARG; }
    const int64_t L = args[0].arr.shape[0];
    const int64_t n = (int64_t)llround((1.0 + sqrt(1.0 + 8.0 * (double)L)) / 2.0);
    if (n * (n - 1) / 2 != L || n < 2) { fn_set_error("num_obs_y: the length of the condensed vector is not a binomial coefficient"); return TSR_EARG; }
    fn_result_int(res, n);
    return TSR_OK;
}

static void set_bool(tsr_result *res, int ok) { memset(res, 0, sizeof *res); res->kind = 4; res->num = ok ? 1 : 0; }

/* is_valid_dm(D, tol=0.0): whether D is a valid redundant distance matrix -- square, zero diagonal, symmetric. */
static int r_is_valid_dm(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { set_bool(res, 0); return TSR_OK; }
    const double tol = (nargs > 1 && args[1].kind == 1) ? args[1].num : 0.0;
    const int64_t n = args[0].arr.shape[0];
    int64_t tot; double *D = fn_arg_doubles(&args[0], &tot); if (!D) return TSR_ENOMEM;
    int ok = 1;
    for (int64_t i = 0; i < n && ok; i++) {
        if (fabs(D[i * n + i]) > tol) ok = 0;
        for (int64_t j = i + 1; j < n && ok; j++) if (fabs(D[i * n + j] - D[j * n + i]) > tol) ok = 0;
    }
    fn_free_doubles(D, tot);
    set_bool(res, ok);
    return TSR_OK;
}

/* is_valid_y(Y): whether Y is a valid condensed distance vector -- 1-D with a binomial-coefficient length. */
static int r_is_valid_y(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { set_bool(res, 0); return TSR_OK; }
    const int64_t L = args[0].arr.shape[0];
    const int64_t n = (int64_t)llround((1.0 + sqrt(1.0 + 8.0 * (double)L)) / 2.0);
    set_bool(res, (n >= 2 && n * (n - 1) / 2 == L));
    return TSR_OK;
}

/* pdist(X, metric=None, p=None) */
static int r_pdist(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("pdist: X must be a 2-D array"); return TSR_EARG; }
    int metric, rc; if ((rc = parse_metric(nargs > 1 ? &args[1] : NULL, &metric)) < 0) return rc;
    const double p = parse_p(nargs > 2 ? &args[2] : NULL);
    const int64_t mrows = args[0].arr.shape[0], n = args[0].arr.shape[1];
    int64_t tot; double *X = fn_arg_doubles(&args[0], &tot);
    if (!X) { fn_set_error("pdist: X could not be read as float64"); return TSR_ENOMEM; }
    const int64_t L = mrows * (mrows - 1) / 2;
    const int64_t osh[1] = {L};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, osh);
    if (!out) { fn_free_doubles(X, tot); return TSR_ENOMEM; }
    int64_t k = 0;
    for (int64_t i = 0; i < mrows; i++)
        for (int64_t j = i + 1; j < mrows; j++)
            out[k++] = vdist(X + i * n, X + j * n, n, metric, p);
    fn_free_doubles(X, tot);
    return TSR_OK;
}

/* cdist(XA, XB, metric=None, p=None) */
static int r_cdist(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("cdist: XA and XB must be 2-D arrays"); return TSR_EARG; }
    const int64_t mA = args[0].arr.shape[0], nA = args[0].arr.shape[1];
    const int64_t mB = args[1].arr.shape[0], nB = args[1].arr.shape[1];
    if (nA != nB) { fn_set_error("cdist: XA and XB must have the same number of columns"); return TSR_EARG; }
    int metric, rc; if ((rc = parse_metric(nargs > 2 ? &args[2] : NULL, &metric)) < 0) return rc;
    const double p = parse_p(nargs > 3 ? &args[3] : NULL);
    int64_t ta, tb; double *A = fn_arg_doubles(&args[0], &ta); if (!A) { fn_set_error("cdist: XA unreadable"); return TSR_ENOMEM; }
    double *B = fn_arg_doubles(&args[1], &tb); if (!B) { fn_free_doubles(A, ta); fn_set_error("cdist: XB unreadable"); return TSR_ENOMEM; }
    const int64_t osh[2] = {mA, mB};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
    if (!out) { fn_free_doubles(A, ta); fn_free_doubles(B, tb); return TSR_ENOMEM; }
    for (int64_t a = 0; a < mA; a++)
        for (int64_t b = 0; b < mB; b++)
            out[a * mB + b] = vdist(A + a * nA, B + b * nA, nA, metric, p);
    fn_free_doubles(A, ta); fn_free_doubles(B, tb);
    return TSR_OK;
}

/* squareform(X): a condensed vector -> a square symmetric matrix, or a square matrix -> its condensed vector */
static int r_squareform(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3) { fn_set_error("squareform: X must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    int64_t tot; double *X = fn_arg_doubles(&args[0], &tot);
    if (!X) { fn_set_error("squareform: X could not be read as float64"); return TSR_ENOMEM; }
    if (nd == 1) {
        const int64_t L = args[0].arr.shape[0];
        int64_t m = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)L)) / 2.0 + 0.5);
        if (m * (m - 1) / 2 != L) { fn_free_doubles(X, tot); fn_set_error("squareform: vector length is not a valid condensed matrix"); return TSR_EARG; }
        const int64_t osh[2] = {m, m};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
        if (!out) { fn_free_doubles(X, tot); return TSR_ENOMEM; }
        memset(out, 0, (size_t)(m * m) * sizeof(double));
        int64_t k = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = i + 1; j < m; j++) { const double v = X[k++]; out[i * m + j] = v; out[j * m + i] = v; }
        fn_free_doubles(X, tot);
        return TSR_OK;
    }
    if (nd == 2 && args[0].arr.shape[0] == args[0].arr.shape[1]) {
        const int64_t m = args[0].arr.shape[0];
        const int64_t L = m * (m - 1) / 2;
        const int64_t osh[1] = {L};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, osh);
        if (!out) { fn_free_doubles(X, tot); return TSR_ENOMEM; }
        int64_t k = 0;
        for (int64_t i = 0; i < m; i++)
            for (int64_t j = i + 1; j < m; j++) out[k++] = X[i * m + j];
        fn_free_doubles(X, tot);
        return TSR_OK;
    }
    fn_free_doubles(X, tot);
    fn_set_error("squareform: X must be a 1-D condensed vector or a 2-D square matrix");
    return TSR_EARG;
}

/* ---- scipy.spatial top-level Minkowski helpers (credited to scipy.spatial.* via scope.py) ---- */
/* p-th-power Minkowski contribution between two k-vectors; p==inf -> max |a-b| (root is then the identity). */
static inline double spt_mink_p(const double *a, const double *b, int64_t k, double p, int inf)
{
    if (inf) { double m = 0.0; for (int64_t d = 0; d < k; d++) { double t = fabs(a[d] - b[d]); if (t > m) m = t; } return m; }
    double s = 0.0; for (int64_t d = 0; d < k; d++) s += pow(fabs(a[d] - b[d]), p);
    return s;
}
static double spt_read_p(const tsr_arg *args, int nargs, int idx)
{
    if (nargs > idx && args[idx].kind == 1) return (args[idx].flags & 1) ? (double)args[idx].ival : args[idx].num;
    return 2.0;
}

/* distance_matrix(x, y, p=2): x (m, k), y (n, k) -> (m, n) Minkowski distances. */
static int r_distance_matrix(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("distance_matrix: x and y must be 2-D"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], k = args[0].arr.shape[1], n = args[1].arr.shape[0];
    if (args[1].arr.shape[1] != k) { fn_set_error("distance_matrix: x and y must have the same number of columns"); return TSR_EARG; }
    double p = spt_read_p(args, nargs, 2); int inf = isinf(p);
    int64_t lx, ly; double *X = fn_arg_doubles(&args[0], &lx); if (!X) return TSR_ENOMEM;
    double *Y = fn_arg_doubles(&args[1], &ly); if (!Y) { fn_free_doubles(X, lx); return TSR_ENOMEM; }
    int64_t sh[2] = {m, n};
    double *D = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    if (!D) { fn_free_doubles(X, lx); fn_free_doubles(Y, ly); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) for (int64_t j = 0; j < n; j++) {
        double s = spt_mink_p(&X[i * k], &Y[j * k], k, p, inf);
        D[i * n + j] = inf ? s : pow(s, 1.0 / p);
    }
    fn_free_doubles(X, lx); fn_free_doubles(Y, ly);
    return TSR_OK;
}

/* x as rows of k values: the array itself when its last axis already has k entries, otherwise (a 0-d array or a
   last axis of 1, broadcast as NumPy does) a malloc'ed copy with the single value repeated; NULL when out of memory */
static double *spt_rows(double *X, int64_t rows, int64_t kx, int64_t k)
{
    if (kx == k) return X;
    double *R = (double *)malloc((size_t)(rows * k > 0 ? rows * k : 1) * sizeof(double));
    if (R) for (int64_t i = 0; i < rows; i++) for (int64_t d = 0; d < k; d++) R[i * k + d] = X[i];
    return R;
}

/* per-point Minkowski between x and y, each 0-d, (k,) or (m, k), broadcast as NumPy does (scipy sums |y - x|^p
   over the last axis); both at most 1-D -> a scalar. root applies the 1/p power. */
static int spt_pointwise(const tsr_arg *args, int nargs, tsr_result *res, int root)
{
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("minkowski_distance: x and y must be arrays"); return TSR_EARG; }
    double p = spt_read_p(args, nargs, 2); int inf = isinf(p);
    const tsr_array *xa = &args[0].arr, *ya = &args[1].arr;
    if (xa->ndim > 2 || ya->ndim > 2) { fn_set_error("minkowski_distance: inputs with more than 2 dimensions are not supported"); return TSR_EDIM; }
    /* a 0-d input has no last axis (it indexed shape[-1]): it broadcasts like a length-1 axis */
    const int64_t kx = xa->ndim ? xa->shape[xa->ndim - 1] : 1, ky = ya->ndim ? ya->shape[ya->ndim - 1] : 1;
    const int64_t mx = xa->ndim == 2 ? xa->shape[0] : 1, my = ya->ndim == 2 ? ya->shape[0] : 1;
    if ((kx != ky && kx != 1 && ky != 1) || (mx != my && mx != 1 && my != 1)) {
        fn_set_error("operands could not be broadcast together"); return TSR_EARG;
    }
    const int64_t k = kx == 1 ? ky : kx, m = mx == 1 ? my : mx;
    int scalar = (xa->ndim <= 1 && ya->ndim <= 1);
    int64_t lx, ly; double *X0 = fn_arg_doubles(&args[0], &lx); if (!X0) return TSR_ENOMEM;
    double *Y0 = fn_arg_doubles(&args[1], &ly); if (!Y0) { fn_free_doubles(X0, lx); return TSR_ENOMEM; }
    double *X = spt_rows(X0, mx, kx, k), *Y = X ? spt_rows(Y0, my, ky, k) : NULL;
    int rc = TSR_OK;
    if (!X || !Y) rc = TSR_ENOMEM;
    else if (scalar) {
        double s = spt_mink_p(X, Y, k, p, inf);
        fn_result_num(&res[0], (root && !inf) ? pow(s, 1.0 / p) : s);
    } else {
        int64_t sh[1] = {m};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
        if (!out) rc = TSR_ENOMEM;
        else for (int64_t i = 0; i < m; i++) {
            const double *xi = mx > 1 ? &X[i * k] : X;
            const double *yi = my > 1 ? &Y[i * k] : Y;
            double s = spt_mink_p(xi, yi, k, p, inf);
            out[i] = (root && !inf) ? pow(s, 1.0 / p) : s;
        }
    }
    if (X && X != X0) free(X);
    if (Y && Y != Y0) free(Y);
    fn_free_doubles(X0, lx); fn_free_doubles(Y0, ly);
    return rc;
}
static int r_minkowski_distance(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return spt_pointwise(args, nargs, res, 1); }
static int r_minkowski_distance_p(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return spt_pointwise(args, nargs, res, 0); }

/* geometric_slerp(start, end, t): spherical interpolation of two unit vectors; result (len(t), dim). */
static int r_geometric_slerp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1 || args[2].kind != 3 || args[2].arr.ndim != 1) { fn_set_error("geometric_slerp: start, end (1-D) and t (1-D) are required"); return TSR_EARG; }
    int64_t dim = args[0].arr.shape[0], nt = args[2].arr.shape[0];
    if (args[1].arr.shape[0] != dim) { fn_set_error("geometric_slerp: start and end must have the same length"); return TSR_EARG; }
    int64_t ls, le, lt; double *S = fn_arg_doubles(&args[0], &ls); if (!S) return TSR_ENOMEM;
    double *E = fn_arg_doubles(&args[1], &le); if (!E) { fn_free_doubles(S, ls); return TSR_ENOMEM; }
    double *T = fn_arg_doubles(&args[2], &lt); if (!T) { fn_free_doubles(S, ls); fn_free_doubles(E, le); return TSR_ENOMEM; }
    double dot = 0.0; for (int64_t d = 0; d < dim; d++) dot += S[d] * E[d];
    if (dot > 1.0) dot = 1.0; else if (dot < -1.0) dot = -1.0;
    double omega = acos(dot), so = sin(omega);
    int64_t sh[2] = {nt, dim};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    if (!out) { fn_free_doubles(S, ls); fn_free_doubles(E, le); fn_free_doubles(T, lt); return TSR_ENOMEM; }
    for (int64_t i = 0; i < nt; i++) {
        double a, b;
        if (so < 1e-12) { a = 1.0 - T[i]; b = T[i]; }                      /* (near-)parallel: linear */
        else { a = sin((1.0 - T[i]) * omega) / so; b = sin(T[i] * omega) / so; }
        for (int64_t d = 0; d < dim; d++) out[i * dim + d] = a * S[d] + b * E[d];
    }
    fn_free_doubles(S, ls); fn_free_doubles(E, le); fn_free_doubles(T, lt);
    return TSR_OK;
}

/* procrustes(data1, data2): standardize both (center columns, scale to unit Frobenius norm), rotate data2
   onto data1 by the orthogonal Procrustes solution, return (mtx1, mtx2, disparity). SVD via LAPACK dgesdd. */
static int r_procrustes(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("procrustes: data1 and data2 must be 2-D"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0], k = args[0].arr.shape[1];
    if (args[1].arr.shape[0] != n || args[1].arr.shape[1] != k) { fn_set_error("procrustes: inputs must have the same shape"); return TSR_EARG; }
    int64_t l1, l2; double *D1 = fn_arg_doubles(&args[0], &l1); if (!D1) return TSR_ENOMEM;
    double *D2 = fn_arg_doubles(&args[1], &l2); if (!D2) { fn_free_doubles(D1, l1); return TSR_ENOMEM; }
    int64_t sh[2] = {n, k};
    double *A = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);          /* mtx1 (standardized data1) */
    double *B2 = (double *)fn_result_array(&res[1], TSR_F64, 2, sh);         /* mtx2 (rotated data2)     */
    double *M = (double *)calloc((size_t)(k * k), sizeof(double));
    double *B = (double *)malloc((size_t)(n * k) * sizeof(double));
    double *U = (double *)malloc((size_t)(k * k) * sizeof(double));
    double *Vt = (double *)malloc((size_t)(k * k) * sizeof(double));
    double *w = (double *)malloc((size_t)k * sizeof(double));
    double *R = (double *)malloc((size_t)(k * k) * sizeof(double));
    int rc = TSR_OK;
    if (!A || !B2 || !M || !B || !U || !Vt || !w || !R) rc = TSR_ENOMEM;
    else {
        /* center columns and scale to unit Frobenius norm */
        for (int64_t c = 0; c < k; c++) {
            double mu1 = 0.0, mu2 = 0.0;
            for (int64_t i = 0; i < n; i++) { mu1 += D1[i * k + c]; mu2 += D2[i * k + c]; }
            mu1 /= (double)n; mu2 /= (double)n;
            for (int64_t i = 0; i < n; i++) { A[i * k + c] = D1[i * k + c] - mu1; B[i * k + c] = D2[i * k + c] - mu2; }
        }
        double f1 = 0.0, f2 = 0.0;
        for (int64_t i = 0; i < n * k; i++) { f1 += A[i] * A[i]; f2 += B[i] * B[i]; }
        f1 = sqrt(f1); f2 = sqrt(f2);
        if (f1 == 0.0 || f2 == 0.0) { fn_set_error("procrustes: input matrices must contain >1 unique point"); rc = TSR_EARG; }
        else {
            for (int64_t i = 0; i < n * k; i++) { A[i] /= f1; B[i] /= f2; }
            /* M = A^T @ B  (k x k) */
            for (int64_t p = 0; p < k; p++) for (int64_t q = 0; q < k; q++) {
                double s = 0.0; for (int64_t i = 0; i < n; i++) s += A[i * k + p] * B[i * k + q];
                M[p * k + q] = s;
            }
            /* dgesdd does not converge (it can loop) on a NaN or an infinity: refuse them first */
            int fin = 1; for (int64_t q = 0; q < k * k; q++) if (!isfinite(M[q])) fin = 0;
            lapack_int info = fin ? LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'A', (lapack_int)k, (lapack_int)k, M, (lapack_int)k, w, U, (lapack_int)k, Vt, (lapack_int)k) : 1;
            if (info != 0) { fn_set_error("procrustes: SVD failed"); rc = TSR_EARG; }
            else {
                double scale = 0.0; for (int64_t i = 0; i < k; i++) scale += w[i];
                for (int64_t p = 0; p < k; p++) for (int64_t q = 0; q < k; q++) {   /* R = U @ Vt */
                    double s = 0.0; for (int64_t r = 0; r < k; r++) s += U[p * k + r] * Vt[r * k + q];
                    R[p * k + q] = s;
                }
                for (int64_t i = 0; i < n; i++) for (int64_t d = 0; d < k; d++) {    /* B2 = (B @ R^T) * scale */
                    double s = 0.0; for (int64_t q = 0; q < k; q++) s += B[i * k + q] * R[d * k + q];
                    B2[i * k + d] = s * scale;
                }
                double disp = 0.0; for (int64_t i = 0; i < n * k; i++) { double t = A[i] - B2[i]; disp += t * t; }
                fn_result_num(&res[2], disp);
            }
        }
    }
    free(M); free(B); free(U); free(Vt); free(w); free(R);
    fn_free_doubles(D1, l1); fn_free_doubles(D2, l2);
    return rc;
}

/* ---- kd-tree k-nearest neighbours (scipy.spatial.cKDTree / KDTree .query, Euclidean) ----
   Built per call: median splits on the widest axis down to leaves of KD_LEAF points, each node with its bounding
   box. Each query keeps its k best (distance, index) pairs in a max-heap and enters a subtree only while the box's
   lower-bound distance could still beat or tie the worst kept pair (with a relative margin, so rounding never
   prunes a tie). Distances are sqrt(sum_j (p_j - q_j)^2), the axis sum in order, ties broken by the smaller index:
   the same neighbours, distances and order as an exhaustive search (Tessero\Spatial\KDTree before 2026-11-06),
   in about O(log n) per query instead of O(n log n). */
#define KD_LEAF 16
typedef struct { int64_t lo, hi, left, right; } kd_node;
/* check: a tree handed back by the caller, its nodes and indices checked as the search reaches them (kd_search);
   n: the number of points; bad: set when a check failed */
typedef struct { const double *p; int64_t m; int64_t *idx; kd_node *nodes; double *box; int64_t nn; int check; int64_t n; int bad; } kd_tree;

/* move the median (by coordinate dim) of idx[lo..hi) to position mid, smaller ones before it */
static void kd_select(kd_tree *t, int64_t lo, int64_t hi, int64_t mid, int dim)
{
    int64_t *ix = t->idx;
    const double *p = t->p;
    const int64_t m = t->m;
    hi--;
    while (hi > lo) {
        const double pv = p[ix[lo + (hi - lo) / 2] * m + dim];
        int64_t i = lo, j = hi;
        while (i <= j) {
            while (p[ix[i] * m + dim] < pv) i++;
            while (p[ix[j] * m + dim] > pv) j--;
            if (i <= j) { const int64_t s = ix[i]; ix[i] = ix[j]; ix[j] = s; i++; j--; }
        }
        if (mid <= j) hi = j;
        else if (mid >= i) lo = i;
        else return;
    }
}

static int64_t kd_build(kd_tree *t, int64_t lo, int64_t hi)
{
    const int64_t id = t->nn++, m = t->m;
    kd_node *nd = &t->nodes[id];
    nd->lo = lo; nd->hi = hi; nd->left = nd->right = -1;
    double *bmin = t->box + id * 2 * m, *bmax = bmin + m;
    for (int64_t j = 0; j < m; j++) { bmin[j] = INFINITY; bmax[j] = -INFINITY; }
    for (int64_t r = lo; r < hi; r++) {
        const double *pt = t->p + t->idx[r] * m;
        for (int64_t j = 0; j < m; j++) { if (pt[j] < bmin[j]) bmin[j] = pt[j]; if (pt[j] > bmax[j]) bmax[j] = pt[j]; }
    }
    if (hi - lo <= KD_LEAF) return id;
    int dim = 0;
    double spread = -1.0;
    for (int64_t j = 0; j < m; j++) if (bmax[j] - bmin[j] > spread) { spread = bmax[j] - bmin[j]; dim = (int)j; }
    if (!(spread > 0.0)) return id;                       /* all points equal: one leaf */
    const int64_t mid = lo + (hi - lo) / 2;
    kd_select(t, lo, hi, mid, dim);
    const int64_t l = kd_build(t, lo, mid);
    const int64_t r = kd_build(t, mid, hi);
    t->nodes[id].left = l; t->nodes[id].right = r;
    return id;
}

typedef struct { double *d; int64_t *i; int64_t k, n; } kd_heap;   /* max-heap of the k best: (d, i) lexicographic */
static inline int kd_worse(double da, int64_t ia, double db, int64_t ib) { return da > db || (da == db && ia > ib); }

static void kd_sift_down(kd_heap *h, double d, int64_t i)
{
    int64_t c = 0;
    for (;;) {
        int64_t ch = 2 * c + 1;
        if (ch >= h->n) break;
        if (ch + 1 < h->n && kd_worse(h->d[ch + 1], h->i[ch + 1], h->d[ch], h->i[ch])) ch++;
        if (!kd_worse(h->d[ch], h->i[ch], d, i)) break;
        h->d[c] = h->d[ch]; h->i[c] = h->i[ch]; c = ch;
    }
    h->d[c] = d; h->i[c] = i;
}

static void kd_push(kd_heap *h, double d, int64_t i)
{
    if (h->n < h->k) {                                    /* sift up */
        int64_t c = h->n++;
        while (c > 0) { const int64_t pa = (c - 1) / 2; if (!kd_worse(d, i, h->d[pa], h->i[pa])) break; h->d[c] = h->d[pa]; h->i[c] = h->i[pa]; c = pa; }
        h->d[c] = d; h->i[c] = i;
        return;
    }
    if (!kd_worse(h->d[0], h->i[0], d, i)) return;        /* not better than the worst kept */
    kd_sift_down(h, d, i);                                 /* replace the root */
}

/* the squared distance from q to the box of node id (0 inside) */
static double kd_box_lb(const kd_tree *t, int64_t id, const double *q)
{
    const int64_t m = t->m;
    const double *bmin = t->box + id * 2 * m, *bmax = bmin + m;
    double lb = 0.0;
    for (int64_t j = 0; j < m; j++) {
        const double g = q[j] < bmin[j] ? bmin[j] - q[j] : q[j] > bmax[j] ? q[j] - bmax[j] : 0.0;
        lb += g * g;
    }
    return lb;
}

/* A tree handed back by the caller (the idx, nodes, box outputs of an earlier call on the same data) is never read out
   of bounds: each node the search reaches is checked before its fields are used (its range within the points; an inner
   node of at least two points split exactly as kd_build splits, its children after it), each index before its point
   is read, and that point must be finite. The checks are the same per node as a pass over the whole tree, so a child
   holds at most half of its parent's points (rounded up) and the depth stays below log2(n) + 2, but cost only the
   nodes visited. Box values are not checked: a wrong box only prunes wrongly. */
static int kd_node_ok(const kd_tree *t, int64_t id)
{
    const kd_node *nd = &t->nodes[id];
    if (nd->lo < 0 || nd->hi > t->n || nd->lo > nd->hi) return 0;
    if (nd->left == -1 && nd->right == -1) return 1;                   /* a leaf */
    if (nd->hi - nd->lo < 2 || nd->left <= id || nd->right <= id || nd->left >= t->nn || nd->right >= t->nn) return 0;
    const int64_t mid = nd->lo + (nd->hi - nd->lo) / 2;
    const kd_node *l = &t->nodes[nd->left], *r = &t->nodes[nd->right];
    return l->lo == nd->lo && l->hi == mid && r->lo == mid && r->hi == nd->hi;
}

static void kd_search(kd_tree *t, int64_t id, const double *q, kd_heap *h)
{
    const int64_t m = t->m;
    if (t->bad) return;
    if (t->check && !kd_node_ok(t, id)) { t->bad = 1; return; }
    const kd_node *nd = &t->nodes[id];
    if (h->n == h->k) {
        const double w = h->d[0];
        if (kd_box_lb(t, id, q) > w * w * (1.0 + 1e-12)) return;   /* nothing in here can beat or tie the worst kept */
    }
    if (nd->left < 0) {
        for (int64_t r = nd->lo; r < nd->hi; r++) {
            const int64_t pi = t->idx[r];
            if (t->check && (pi < 0 || pi >= t->n)) { t->bad = 1; return; }
            const double *pt = t->p + pi * m;
            if (t->check)
                for (int64_t j = 0; j < m; j++) if (!isfinite(pt[j])) { t->bad = 2; return; }
            double s = 0.0;
            for (int64_t j = 0; j < m; j++) { const double d = pt[j] - q[j]; s += d * d; }
            kd_push(h, sqrt(s), pi);
        }
        return;
    }
    if (kd_box_lb(t, nd->left, q) <= kd_box_lb(t, nd->right, q)) { kd_search(t, nd->left, q, h); kd_search(t, nd->right, q, h); }
    else { kd_search(t, nd->right, q, h); kd_search(t, nd->left, q, h); }
}

/* the integer-valued array argument a as int64 (exact below 2^53); NULL with *len = -1 when it holds a non-integer */
static int64_t *kd_arg_i64(const tsr_arg *a, int64_t *len)
{
    int64_t t;
    double *v = fn_arg_doubles(a, &t);
    if (!v) { *len = 0; return NULL; }
    int64_t *o = (int64_t *)tsr_alloc((t > 0 ? t : 1) * 8);
    int ok = o != NULL;
    for (int64_t q = 0; ok && q < t; q++) {
        if (!(v[q] >= -9.0e15 && v[q] <= 9.0e15) || v[q] != floor(v[q])) ok = 0;
        else o[q] = (int64_t)v[q];
    }
    fn_free_doubles(v, t);
    if (!ok) { if (o) tsr_free(o, (t > 0 ? t : 1) * 8); *len = o ? -1 : 0; return NULL; }
    *len = t;
    return o;
}

/* The array arguments of the k-d tree are read in place when they already have the layout the code needs (C-contiguous
   float64 or int64: what kdtree_build returns and what the PHP classes pass), else from a converted copy; *owned says
   which. A query then costs no pass over the data beyond the finiteness check. */
static const double *kd_f64(const tsr_arg *a, int64_t *len, int *owned)
{
    if (a->kind == 3 && a->arr.dtype == TSR_F64 && tsr_array_is_contiguous(&a->arr)) {
        *owned = 0;
        *len = tsr_array_size(&a->arr);
        return (const double *)((const char *)a->arr.data + a->arr.offset);
    }
    *owned = 1;
    return fn_arg_doubles(a, len);
}

static const int64_t *kd_i64(const tsr_arg *a, int64_t *len, int *owned)
{
    if (a->kind == 3 && a->arr.dtype == TSR_I64 && tsr_array_is_contiguous(&a->arr)) {
        *owned = 0;
        *len = tsr_array_size(&a->arr);
        return (const int64_t *)((const char *)a->arr.data + a->arr.offset);
    }
    *owned = 1;
    return kd_arg_i64(a, len);
}

static void kd_release_f64(const double *p, int64_t len, int owned) { if (owned && p) fn_free_doubles((double *)p, len); }
static void kd_release_i64(const int64_t *p, int64_t len, int owned) { if (owned && p) tsr_free((void *)p, (len > 0 ? len : 1) * 8); }

/* data checked: 2-D, at least one point with at least one coordinate, all finite (who names the caller) */
/* the data argument (n x m, read in place when it can be); finite: check every value (a search of a caller's tree checks
   the points it reaches instead) */
static const double *kd_data(const tsr_arg *a, const char *who, int64_t *n, int64_t *m, int64_t *total, int *owned, int finite,
                             int *rc)
{
    *rc = TSR_EARG;
    if (a->kind != 3 || a->arr.ndim != 2) { fn_set_error("%s: data must be a 2-D array", who); return NULL; }
    *n = a->arr.shape[0]; *m = a->arr.shape[1];
    if (*n == 0) { fn_set_error("%s: the tree has no points", who); return NULL; }
    if (*m == 0) { fn_set_error("%s: the points have no coordinates", who); return NULL; }
    const double *P = kd_f64(a, total, owned);
    if (!P) { *rc = TSR_ENOMEM; return NULL; }
    for (int64_t q = 0; finite && q < *n * *m; q++)
        if (!isfinite(P[q])) { kd_release_f64(P, *total, *owned); fn_set_error("%s: data must be finite", who); return NULL; }
    *rc = TSR_OK;
    return P;
}

/* build the tree of P (n x m) into t: idx, nodes and box from tsr_alloc, sized by *cap; TSR_ENOMEM leaves nothing */
static int kd_new_tree(kd_tree *t, const double *P, int64_t n, int64_t m, int64_t *cap)
{
    *cap = n / 4 + 16;                                    /* nodes: every leaf holds at least KD_LEAF / 2 points */
    *t = (kd_tree){ P, m, (int64_t *)tsr_alloc(n * 8), (kd_node *)tsr_alloc(*cap * (int64_t)sizeof(kd_node)),
                    (double *)tsr_alloc(*cap * 2 * m * 8), 0 };
    if (!t->idx || !t->nodes || !t->box) {
        if (t->idx) tsr_free(t->idx, n * 8);
        if (t->nodes) tsr_free(t->nodes, *cap * (int64_t)sizeof(kd_node));
        if (t->box) tsr_free(t->box, *cap * 2 * m * 8);
        t->idx = NULL; t->nodes = NULL; t->box = NULL;
        return TSR_ENOMEM;
    }
    for (int64_t r = 0; r < n; r++) t->idx[r] = r;
    kd_build(t, 0, n);
    return TSR_OK;
}

static void kd_free_tree(kd_tree *t, int64_t n, int64_t m, int64_t cap)
{
    tsr_free(t->idx, n * 8);
    tsr_free(t->nodes, cap * (int64_t)sizeof(kd_node));
    tsr_free(t->box, cap * 2 * m * 8);
}

/* kdtree_build(data) -> (idx, nodes, box): the k-d tree of data, idx (n,) the point order, nodes (nn, 4) the rows
   (lo, hi, left, right) and box (nn, 2m) each node's bounding box (minima, then maxima). Passed to kdtree_query with
   the same data, it spares the query the build (scipy.spatial.cKDTree builds once, in its constructor). */
static int r_kdtree_build(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 1) { fn_set_error("kdtree_build: data must be a 2-D array"); return TSR_EARG; }
    int64_t n, m, td, cap;
    int rc, own;
    const double *P = kd_data(&args[0], "kdtree_build", &n, &m, &td, &own, 1, &rc);
    if (!P) return rc;
    kd_tree t;
    rc = kd_new_tree(&t, P, n, m, &cap);
    if (rc == TSR_OK) {
        const int64_t nn = t.nn, ish[1] = {n}, nsh[2] = {nn, 4}, bsh[2] = {nn, 2 * m};
        int64_t *oidx = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, ish);
        int64_t *onodes = oidx ? (int64_t *)fn_result_array(&res[1], TSR_I64, 2, nsh) : NULL;
        double *obox = onodes ? (double *)fn_result_array(&res[2], TSR_F64, 2, bsh) : NULL;
        if (!obox) rc = TSR_ENOMEM;
        else {
            memcpy(oidx, t.idx, (size_t)n * 8);
            for (int64_t id = 0; id < nn; id++) {
                onodes[4 * id] = t.nodes[id].lo; onodes[4 * id + 1] = t.nodes[id].hi;
                onodes[4 * id + 2] = t.nodes[id].left; onodes[4 * id + 3] = t.nodes[id].right;
            }
            memcpy(obox, t.box, (size_t)(nn * 2 * m) * 8);
        }
        kd_free_tree(&t, n, m, cap);
    }
    kd_release_f64(P, td, own);
    return rc;
}

/* kdtree_query(data, x, k=1, idx=None, nodes=None, box=None) -> (d, i), each (len(x), min(k, len(data))): the k
   nearest data rows of every row of x, nearest first, ties by index. With idx, nodes and box (kdtree_build of the
   same data) the query searches that tree; without, it builds its own. */
static int r_kdtree_query(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    _Static_assert(sizeof(kd_node) == 4 * sizeof(int64_t), "kd_node is one row of nodes");
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) {
        fn_set_error("kdtree_query: data and x must be 2-D arrays"); return TSR_EARG;
    }
    const int64_t nq = args[1].arr.shape[0];
    if (args[1].arr.shape[1] != args[0].arr.shape[1]) {
        fn_set_error("kdtree_query: x must have %lld columns", (long long)args[0].arr.shape[1]); return TSR_EARG;
    }
    int64_t k = 1;
    if (nargs > 2 && args[2].kind == 1) {
        const double kv = (args[2].flags & 1) ? (double)args[2].ival : args[2].num;
        if (!(kv >= 1 && kv <= 9.0e15) || kv != floor(kv)) { fn_set_error("kdtree_query: k must be a positive integer"); return TSR_EARG; }
        k = (int64_t)kv;
    }
    int given = 0, other = 0;
    for (int a = 3; a < 6 && a < nargs; a++) {
        if (args[a].kind == 3) given++;
        else if (args[a].kind != 0) other = 1;
    }
    if (other || (given != 0 && given != 3)) {
        fn_set_error("kdtree_query: pass all of idx, nodes and box (kdtree_build of the same data), or none"); return TSR_EARG;
    }
    int64_t n, m, td, tx, cap = 0;
    int rc, ownp, ownx;
    const double *P = kd_data(&args[0], "kdtree_query", &n, &m, &td, &ownp, given == 0, &rc);
    if (!P) return rc;
    const double *X = kd_f64(&args[1], &tx, &ownx);
    if (!X) { kd_release_f64(P, td, ownp); return TSR_ENOMEM; }
    for (int64_t q = 0; q < nq * m; q++)
        if (!isfinite(X[q])) { kd_release_f64(P, td, ownp); kd_release_f64(X, tx, ownx); fn_set_error("kdtree_query: x must be finite"); return TSR_EARG; }
    const int64_t kk = k < n ? k : n;

    int64_t nidx = 0, nnodes = 0, tbox = 0;
    int owni = 0, ownn = 0, ownb = 0;
    const int64_t *gidx = NULL, *gnodes = NULL;
    const double *gbox = NULL;
    kd_tree t = { P, m, NULL, NULL, NULL, 0 };
    if (given) {                                          /* the caller's tree: checked, then searched in place */
        gidx = kd_i64(&args[3], &nidx, &owni);
        gnodes = gidx ? kd_i64(&args[4], &nnodes, &ownn) : NULL;
        gbox = gnodes ? kd_f64(&args[5], &tbox, &ownb) : NULL;
        if (!gidx || !gnodes || !gbox) rc = (nidx < 0 || nnodes < 0) ? TSR_EARG : TSR_ENOMEM;
        else if (args[3].arr.ndim != 1 || args[4].arr.ndim != 2 || args[4].arr.shape[1] != 4 || args[5].arr.ndim != 2
                 || args[5].arr.shape[1] != 2 * m || args[5].arr.shape[0] != args[4].arr.shape[0]
                 || nidx != n || args[4].arr.shape[0] < 1 || gnodes[0] != 0 || gnodes[1] != n) rc = TSR_EARG;
        if (rc == TSR_EARG) fn_set_error("kdtree_query: idx, nodes and box are not a tree of this data (pass kdtree_build's outputs)");
        if (rc == TSR_OK) {                               /* only read below: the casts drop const for the struct */
            t.nn = args[4].arr.shape[0];
            t.idx = (int64_t *)gidx;
            t.nodes = (kd_node *)gnodes;
            t.box = (double *)gbox;
            t.check = 1;                                  /* the nodes and indices checked as the search reaches them */
            t.n = n;
        }
    } else {
        rc = kd_new_tree(&t, P, n, m, &cap);
    }
    double *hd = rc == TSR_OK ? (double *)tsr_alloc(kk * 8) : NULL;
    int64_t *hi = hd ? (int64_t *)tsr_alloc(kk * 8) : NULL;
    if (rc == TSR_OK && !hi) rc = TSR_ENOMEM;
    if (rc == TSR_OK) {
        const int64_t osh[2] = {nq, kk};
        double *od = (double *)fn_result_array(&res[0], TSR_F64, 2, osh);
        int64_t *oi = od ? (int64_t *)fn_result_array(&res[1], TSR_I64, 2, osh) : NULL;
        if (!oi) rc = TSR_ENOMEM;
        else
            for (int64_t q = 0; q < nq; q++) {
                kd_heap h = { hd, hi, kk, 0 };
                kd_search(&t, 0, X + q * m, &h);
                if (t.bad) break;                         /* the heap may be short: nothing is popped */
                for (int64_t s = kk - 1; s >= 0; s--) {     /* pop the worst into the last free slot: ascending order */
                    od[q * kk + s] = h.d[0]; oi[q * kk + s] = h.i[0];
                    const double ld = h.d[h.n - 1];
                    const int64_t li = h.i[h.n - 1];
                    h.n--;
                    if (h.n > 0) kd_sift_down(&h, ld, li);
                }
            }
        if (rc == TSR_OK && t.bad) {
            fn_set_error(t.bad == 2 ? "kdtree_query: data must be finite"
                                    : "kdtree_query: idx, nodes and box are not a tree of this data (pass kdtree_build's outputs)");
            rc = TSR_EARG;
        }
    }
    if (given) {
        kd_release_i64(gidx, nidx, owni);
        kd_release_i64(gnodes, nnodes, ownn);
        kd_release_f64(gbox, tbox, ownb);
    } else if (t.idx) {
        kd_free_tree(&t, n, m, cap);
    }
    if (hd) tsr_free(hd, kk * 8);
    if (hi) tsr_free(hi, kk * 8);
    kd_release_f64(P, td, ownp);
    kd_release_f64(X, tx, ownx);
    return rc;
}

/* ------------------------------------------------------------------ 2-D convex hull and Delaunay triangulation */

/* the 2-D points argument: an (n, 2) array as doubles (rows of x, y) */
static double *g2_points(const tsr_arg *a, const char *who, int64_t *n, int64_t *total, int *rc)
{
    *rc = TSR_EARG;
    if (a->kind != 3 || a->arr.ndim != 2 || a->arr.shape[1] != 2) { fn_set_error("%s: points must be an (n, 2) array", who); return NULL; }
    *n = a->arr.shape[0];
    double *P = fn_arg_doubles(a, total);
    if (!P) { *rc = TSR_ENOMEM; return NULL; }
    for (int64_t i = 0; i < *n * 2; i++)
        if (!isfinite(P[i])) { fn_free_doubles(P, *total); fn_set_error("%s: points must be finite", who); return NULL; }
    *rc = TSR_OK;
    return P;
}

typedef struct { double x, y; int64_t i; } g2_pt;

/* convex_hull_2d(points) -> vertices: Andrew's monotone chain, the hull vertices counter-clockwise starting from
   the lowest x (lowest y among equals), collinear points dropped; fewer than 3 when the points are collinear.
   What Spatial\ConvexHull computed in PHP: points sorted by x then y (equal points in input order, as PHP's stable
   sort keeps them), a point popped while the turn (cross product) is <= 0. */
static int r_convex_hull_2d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 1) { fn_set_error("convex_hull_2d: points must be an (n, 2) array"); return TSR_EARG; }
    int64_t n, tp;
    int prc;
    double *P = g2_points(&args[0], "convex_hull_2d", &n, &tp, &prc);
    if (!P) return prc;
    g2_pt *s = (g2_pt *)malloc((size_t)(n > 0 ? n : 1) * sizeof(g2_pt));
    int64_t *h = (int64_t *)malloc((size_t)(2 * n + 2) * sizeof(int64_t));
    int64_t *o1 = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t)), *o2 = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    double *c = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    int rc = TSR_OK;
    if (!s || !h || !o1 || !o2 || !c) rc = TSR_ENOMEM;
    else {
        /* sorted by x, then y, then input order: a stable radix argsort by x (equal keys keep their order,
           -0.0 == +0.0), then each run of equal x by y, stably (insertion, or the argsort for a long run) */
        for (int64_t i = 0; i < n; i++) c[i] = P[2 * i];
        rc = tsr_argsort(TSR_F64, n, c, o1);
        for (int64_t a = 0; rc == TSR_OK && a < n;) {
            int64_t b = a + 1;
            while (b < n && P[2 * o1[b]] == P[2 * o1[a]]) b++;
            if (b - a > 32) {
                for (int64_t j = a; j < b; j++) c[j - a] = P[2 * o1[j] + 1];
                rc = tsr_argsort(TSR_F64, b - a, c, o2);
                for (int64_t j = a; rc == TSR_OK && j < b; j++) c[j - a] = (double)o1[j];
                for (int64_t j = a; rc == TSR_OK && j < b; j++) o1[j] = (int64_t)c[o2[j - a]];
            } else {
                for (int64_t j = a + 1; j < b; j++) {
                    const int64_t v = o1[j];
                    int64_t q = j;
                    while (q > a && P[2 * o1[q - 1] + 1] > P[2 * v + 1]) { o1[q] = o1[q - 1]; q--; }
                    o1[q] = v;
                }
            }
            a = b;
        }
    }
    if (rc == TSR_OK) {
        for (int64_t j = 0; j < n; j++) { const int64_t i = o1[j]; s[j] = (g2_pt){ P[2 * i], P[2 * i + 1], i }; }
#define G2_CROSS(o, a, b) ((s[a].x - s[o].x) * (s[b].y - s[o].y) - (s[a].y - s[o].y) * (s[b].x - s[o].x))
        int64_t k = 0;                                    /* h holds sorted positions: the lower chain, then the upper */
        for (int64_t i = 0; i < n; i++) {
            while (k >= 2 && G2_CROSS(h[k - 2], h[k - 1], i) <= 0.0) k--;
            h[k++] = i;
        }
        const int64_t lower = k;                          /* the lower chain without its last point ... */
        int64_t u = lower;
        for (int64_t i = n - 1; i >= 0; i--) {            /* ... then the upper chain, built on its own */
            while (u - lower >= 2 && G2_CROSS(h[u - 2], h[u - 1], i) <= 0.0) u--;
            h[u++] = i;
        }
#undef G2_CROSS
        const int64_t nl = lower > 0 ? lower - 1 : 0, nu = u - lower > 0 ? u - lower - 1 : 0, m = nl + nu;
        int64_t sh[1] = {m};
        int64_t *o = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, sh);
        if (!o) rc = TSR_ENOMEM;
        else {
            for (int64_t j = 0; j < nl; j++) o[j] = s[h[j]].i;
            for (int64_t j = 0; j < nu; j++) o[nl + j] = s[h[lower + j]].i;
        }
    }
    free(s); free(h); free(o1); free(o2); free(c);
    fn_free_doubles(P, tp);
    return rc;
}

/* Delaunay triangulation in 2-D: the sweep-hull algorithm of Delaunator (V. Agafonkin, ISC licence), ported in its
   own conventions. Seed: the point nearest the bounding-box centre, its nearest neighbour, and the third point
   giving the smallest circumcircle; the other points are added in order of distance from that circle's centre,
   each joined to the visible edges of the current convex hull (found through a hash on the angle about the centre),
   the Delaunay condition restored by edge flips. O(n log n); the hull is exact by construction (no super
   triangle). For points in general position the triangulation is unique, so it is Qhull's. */
#define DL_EPS 2.220446049250313e-16
static double dl_orient(double ax, double ay, double bx, double by, double cx, double cy)
{
    return (ay - cy) * (bx - cx) - (ax - cx) * (by - cy);    /* > 0: clockwise (delaunator's orient2d sign) */
}
static int dl_incircle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py)
{
    const double dx = ax - px, dy = ay - py, ex = bx - px, ey = by - py, fx = cx - px, fy = cy - py;
    const double ap = dx * dx + dy * dy, bp = ex * ex + ey * ey, cp = fx * fx + fy * fy;
    return dx * (ey * cp - bp * fy) - dy * (ex * cp - bp * fx) + ap * (ex * fy - ey * fx) < 0.0;
}
static double dl_circumradius(double ax, double ay, double bx, double by, double cx, double cy)
{
    const double dx = bx - ax, dy = by - ay, ex = cx - ax, ey = cy - ay;
    const double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, d = 0.5 / (dx * ey - dy * ex);
    const double x = (ey * bl - dy * cl) * d, y = (dx * cl - ex * bl) * d;
    return x * x + y * y;
}
typedef struct {
    const double *c;
    int64_t *tri, *half, len, cap;
    int over;
    int64_t *hprev, *hnext, *htri, *hhash, hsize, hstart;
    double cx, cy;
} dl_state;
static int64_t dl_hashkey(const dl_state *s, double x, double y)
{
    const double dx = x - s->cx, dy = y - s->cy, p = dx / (fabs(dx) + fabs(dy));
    double a = (dy > 0.0 ? 3.0 - p : 1.0 + p) / 4.0;           /* the pseudo-angle in [0, 1] */
    if (!(a >= 0.0 && a <= 1.0)) a = 0.0;                  /* a point at the centre itself */
    int64_t k = (int64_t)floor(a * (double)s->hsize);
    k %= s->hsize;
    return k < 0 ? k + s->hsize : k;
}
static void dl_link(dl_state *s, int64_t a, int64_t b) { s->half[a] = b; if (b != -1) s->half[b] = a; }
static int64_t dl_add(dl_state *s, int64_t i0, int64_t i1, int64_t i2, int64_t a, int64_t b, int64_t c)
{
    const int64_t t = s->len;
    if (t + 3 > s->cap) { s->over = 1; return t - 3 >= 0 ? t - 3 : 0; }   /* never on a valid triangulation */
    s->tri[t] = i0; s->tri[t + 1] = i1; s->tri[t + 2] = i2;
    dl_link(s, t, a); dl_link(s, t + 1, b); dl_link(s, t + 2, c);
    s->len += 3;
    return t;
}
static int64_t dl_legalize(dl_state *s, int64_t a)
{
    int64_t stack[512];
    int i = 0;
    int64_t ar = 0;
    const double *c = s->c;
    for (;;) {
        const int64_t b = s->half[a], a0 = a - a % 3;
        ar = a0 + (a + 2) % 3;
        if (b == -1) {                                    /* a convex hull edge */
            if (i == 0) break;
            a = stack[--i];
            continue;
        }
        const int64_t b0 = b - b % 3, al = a0 + (a + 1) % 3, bl = b0 + (b + 2) % 3;
        const int64_t p0 = s->tri[ar], pr = s->tri[a], pl = s->tri[al], p1 = s->tri[bl];
        if (dl_incircle(c[2 * p0], c[2 * p0 + 1], c[2 * pr], c[2 * pr + 1], c[2 * pl], c[2 * pl + 1], c[2 * p1], c[2 * p1 + 1])) {
            s->tri[a] = p1;
            s->tri[b] = p0;
            const int64_t hbl = s->half[bl];
            if (hbl == -1) {                              /* the edge swapped on the other side of the hull */
                int64_t e = s->hstart;
                do {
                    if (s->htri[e] == bl) { s->htri[e] = a; break; }
                    e = s->hprev[e];
                } while (e != s->hstart);
            }
            dl_link(s, a, hbl);
            dl_link(s, b, s->half[ar]);
            dl_link(s, ar, bl);
            const int64_t br = b0 + (b + 1) % 3;
            if (i < 512) stack[i++] = br;
        } else {
            if (i == 0) break;
            a = stack[--i];
        }
    }
    return ar;
}
typedef struct { double d; int64_t i; } dl_key;
static int dl_cmp_dist(const void *pa, const void *pb)        /* distance, then index */
{
    const dl_key *a = pa, *b = pb;
    if (a->d != b->d) return a->d < b->d ? -1 : 1;
    return a->i < b->i ? -1 : a->i > b->i;
}

/* delaunay_2d(points) -> simplices: the Delaunay triangles as (ntri, 3) point indices (no rows when the points are
   collinear or fewer than 3) */
static int r_delaunay_2d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 1) { fn_set_error("delaunay_2d: points must be an (n, 2) array"); return TSR_EARG; }
    int64_t n, tp;
    int prc;
    double *P = g2_points(&args[0], "delaunay_2d", &n, &tp, &prc);
    if (!P) return prc;
    const int64_t maxt = n > 2 ? 2 * n - 5 : 0, hs = (int64_t)ceil(sqrt((double)(n > 0 ? n : 1)));
    dl_state s = { P, NULL, NULL, 0, maxt * 3, 0, NULL, NULL, NULL, NULL, hs > 0 ? hs : 1, 0, 0.0, 0.0 };
    int64_t *ids = NULL;
    dl_key *keys = NULL;
    int rc = TSR_OK;
    if (n >= 3) {
        s.tri = (int64_t *)malloc((size_t)(maxt * 3) * sizeof(int64_t));
        s.half = (int64_t *)malloc((size_t)(maxt * 3) * sizeof(int64_t));
        s.hprev = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        s.hnext = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        s.htri = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        s.hhash = (int64_t *)malloc((size_t)s.hsize * sizeof(int64_t));
        ids = (int64_t *)malloc((size_t)n * sizeof(int64_t));
        keys = (dl_key *)malloc((size_t)n * sizeof(dl_key));
        if (!s.tri || !s.half || !s.hprev || !s.hnext || !s.htri || !s.hhash || !ids || !keys) rc = TSR_ENOMEM;
    }
    if (rc == TSR_OK && n >= 3) {
        const double *c = P;
        double minx = INFINITY, miny = INFINITY, maxx = -INFINITY, maxy = -INFINITY;
        for (int64_t i = 0; i < n; i++) {
            const double x = c[2 * i], y = c[2 * i + 1];
            if (x < minx) minx = x;
            if (y < miny) miny = y;
            if (x > maxx) maxx = x;
            if (y > maxy) maxy = y;
            ids[i] = i;
        }
        const double bcx = (minx + maxx) / 2.0, bcy = (miny + maxy) / 2.0;
        int64_t i0 = 0, i1 = 0, i2 = 0;
        double md = INFINITY;
        for (int64_t i = 0; i < n; i++) {                 /* a seed close to the centre */
            const double dx = bcx - c[2 * i], dy = bcy - c[2 * i + 1], d = dx * dx + dy * dy;
            if (d < md) { i0 = i; md = d; }
        }
        const double i0x = c[2 * i0], i0y = c[2 * i0 + 1];
        md = INFINITY;
        for (int64_t i = 0; i < n; i++) {                 /* the point closest to the seed */
            if (i == i0) continue;
            const double dx = i0x - c[2 * i], dy = i0y - c[2 * i + 1], d = dx * dx + dy * dy;
            if (d < md && d > 0.0) { i1 = i; md = d; }
        }
        double i1x = c[2 * i1], i1y = c[2 * i1 + 1], minr = INFINITY;
        for (int64_t i = 0; i < n; i++) {                 /* the third point with the smallest circumcircle */
            if (i == i0 || i == i1) continue;
            const double r = dl_circumradius(i0x, i0y, i1x, i1y, c[2 * i], c[2 * i + 1]);
            if (r < minr) { i2 = i; minr = r; }
        }
        double i2x = c[2 * i2], i2y = c[2 * i2 + 1];
        if (minr < INFINITY) {                            /* else collinear: no triangles */
            if (dl_orient(i0x, i0y, i1x, i1y, i2x, i2y) < 0.0) {
                const int64_t t = i1; i1 = i2; i2 = t;
                const double tx = i1x, ty = i1y; i1x = i2x; i1y = i2y; i2x = tx; i2y = ty;
            }
            {                                             /* the seed triangle's circumcentre */
                const double dx = i1x - i0x, dy = i1y - i0y, ex = i2x - i0x, ey = i2y - i0y;
                const double bl = dx * dx + dy * dy, cl = ex * ex + ey * ey, d = 0.5 / (dx * ey - dy * ex);
                s.cx = i0x + (ey * bl - dy * cl) * d;
                s.cy = i0y + (dx * cl - ex * bl) * d;
            }
            for (int64_t i = 0; i < n; i++) { const double dx = c[2 * i] - s.cx, dy = c[2 * i + 1] - s.cy; keys[i] = (dl_key){ dx * dx + dy * dy, i }; }
            qsort(keys, (size_t)n, sizeof(dl_key), dl_cmp_dist);
            for (int64_t i = 0; i < n; i++) ids[i] = keys[i].i;
            s.hstart = i0;
            s.hnext[i0] = s.hprev[i2] = i1;
            s.hnext[i1] = s.hprev[i0] = i2;
            s.hnext[i2] = s.hprev[i1] = i0;
            s.htri[i0] = 0; s.htri[i1] = 1; s.htri[i2] = 2;
            for (int64_t q = 0; q < s.hsize; q++) s.hhash[q] = -1;
            s.hhash[dl_hashkey(&s, i0x, i0y)] = i0;
            s.hhash[dl_hashkey(&s, i1x, i1y)] = i1;
            s.hhash[dl_hashkey(&s, i2x, i2y)] = i2;
            dl_add(&s, i0, i1, i2, -1, -1, -1);
            double xp = 0.0, yp = 0.0;
            for (int64_t k = 0; k < n; k++) {
                const int64_t i = ids[k];
                const double x = c[2 * i], y = c[2 * i + 1];
                if (k > 0 && fabs(x - xp) <= DL_EPS && fabs(y - yp) <= DL_EPS) continue;   /* a near-duplicate */
                xp = x; yp = y;
                if (i == i0 || i == i1 || i == i2) continue;
                int64_t start = 0;                        /* a visible hull edge, through the angle hash */
                const int64_t key = dl_hashkey(&s, x, y);
                for (int64_t j = 0; j < s.hsize; j++) {
                    start = s.hhash[(key + j) % s.hsize];
                    if (start != -1 && start != s.hnext[start]) break;
                }
                if (start == -1) start = s.hstart;
                start = s.hprev[start];
                int64_t e = start, q;
                for (;;) {
                    q = s.hnext[e];
                    if (!(dl_orient(x, y, c[2 * e], c[2 * e + 1], c[2 * q], c[2 * q + 1]) >= 0.0)) break;
                    e = q;
                    if (e == start) { e = -1; break; }
                }
                if (e == -1) continue;                    /* inside the hull by rounding: a near-duplicate */
                int64_t t = dl_add(&s, e, i, s.hnext[e], -1, -1, s.htri[e]);
                s.htri[i] = dl_legalize(&s, t + 2);
                s.htri[e] = t;
                int64_t nx = s.hnext[e];                  /* forward along the hull */
                for (;;) {
                    q = s.hnext[nx];
                    if (!(dl_orient(x, y, c[2 * nx], c[2 * nx + 1], c[2 * q], c[2 * q + 1]) < 0.0)) break;
                    t = dl_add(&s, nx, i, q, s.htri[i], -1, s.htri[nx]);
                    s.htri[i] = dl_legalize(&s, t + 2);
                    s.hnext[nx] = nx;                     /* removed from the hull */
                    nx = q;
                }
                if (e == start) {                         /* backward from the other side */
                    for (;;) {
                        q = s.hprev[e];
                        if (!(dl_orient(x, y, c[2 * q], c[2 * q + 1], c[2 * e], c[2 * e + 1]) < 0.0)) break;
                        t = dl_add(&s, q, i, e, -1, s.htri[e], s.htri[q]);
                        dl_legalize(&s, t + 2);
                        s.htri[q] = t;
                        s.hnext[e] = e;
                        e = q;
                    }
                }
                s.hstart = s.hprev[i] = e;
                s.hnext[e] = s.hprev[nx] = i;
                s.hnext[i] = nx;
                s.hhash[dl_hashkey(&s, x, y)] = i;
                s.hhash[dl_hashkey(&s, c[2 * e], c[2 * e + 1])] = e;
            }
        }
    }
    if (rc == TSR_OK && s.over) { fn_set_error("delaunay_2d: the triangulation failed (degenerate input)"); rc = TSR_EARG; }
    if (rc == TSR_OK) {
        const int64_t nt = s.len / 3, sh[2] = {nt, 3};
        int64_t *o = (int64_t *)fn_result_array(&res[0], TSR_I64, 2, sh);
        if (!o) rc = TSR_ENOMEM;
        else if (nt > 0) memcpy(o, s.tri, (size_t)(nt * 3) * sizeof(int64_t));
    }
    free(s.tri); free(s.half); free(s.hprev); free(s.hnext); free(s.htri); free(s.hhash); free(ids); free(keys);
    fn_free_doubles(P, tp);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("distance.procrustes", 3, "data1, data2", "mtx1, mtx2, disparity", r_procrustes, NULL, "Procrustes analysis: standardize and align two point sets (scipy.spatial.procrustes)."),
    ROUTINE("distance.distance_matrix", 1, "x, y, p=2, threshold=1000000", "D", r_distance_matrix, NULL, "Matrix of all pairwise Minkowski distances between two point sets (scipy.spatial.distance_matrix)."),
    ROUTINE("distance.minkowski_distance", 1, "x, y, p=2", "d", r_minkowski_distance, NULL, "Per-point Minkowski distance between two arrays of points (scipy.spatial.minkowski_distance)."),
    ROUTINE("distance.minkowski_distance_p", 1, "x, y, p=2", "d", r_minkowski_distance_p, NULL, "p-th power of the Minkowski distance, no final root (scipy.spatial.minkowski_distance_p)."),
    ROUTINE("distance.geometric_slerp", 1, "start, end, t", "out", r_geometric_slerp, NULL, "Geometric spherical linear interpolation between two unit vectors (scipy.spatial.geometric_slerp)."),
    ROUTINE("distance.pdist", 1, "X, metric=None, p=None", "out", r_pdist, NULL, "Pairwise distances between the rows of X, condensed (scipy.spatial.distance.pdist); real float64."),
    ROUTINE("distance.kdtree_query", 2, "data, x, k=1, idx=None, nodes=None, box=None", "d, i", r_kdtree_query, NULL, "The k nearest rows of data for each row of x by a k-d tree, Euclidean, nearest first, ties by index (scipy.spatial.cKDTree.query); with idx, nodes and box from distance.kdtree_build of the same data it searches that tree instead of building one."),
    ROUTINE("distance.kdtree_build", 3, "data", "idx, nodes, box", r_kdtree_build, NULL, "The k-d tree of data as arrays (point order, nodes, bounding boxes) for distance.kdtree_query, built once and searched by every query (scipy.spatial.cKDTree's constructor)."),
    ROUTINE("distance.convex_hull_2d", 1, "points", "vertices", r_convex_hull_2d, NULL, "The 2-D convex hull vertices of points, counter-clockwise from the lowest x (Andrew's monotone chain), for Spatial\\ConvexHull."),
    ROUTINE("distance.delaunay_2d", 1, "points", "simplices", r_delaunay_2d, NULL, "The 2-D Delaunay triangles of points as index triples (sweep-hull, Delaunator), for Spatial\\Delaunay."),
    ROUTINE("distance.cdist", 1, "XA, XB, metric=None, p=None", "out", r_cdist, NULL, "Distances between each pair of rows of XA and XB (scipy.spatial.distance.cdist); real float64."),
    ROUTINE("distance.squareform", 1, "X", "out", r_squareform, NULL, "Convert between a condensed distance vector and a square form (scipy.spatial.distance.squareform)."),
    ROUTINE("distance.euclidean", 1, "u, v, w=None", "out", r_dmetric, &MET_EUCLID, "Euclidean distance between two 1-D arrays (scipy.spatial.distance.euclidean)."),
    ROUTINE("distance.sqeuclidean", 1, "u, v, w=None", "out", r_dmetric, &MET_SQEUCLID, "Squared Euclidean distance (scipy.spatial.distance.sqeuclidean)."),
    ROUTINE("distance.cityblock", 1, "u, v, w=None", "out", r_dmetric, &MET_CITYBLOCK, "Manhattan (city block) distance (scipy.spatial.distance.cityblock)."),
    ROUTINE("distance.chebyshev", 1, "u, v, w=None", "out", r_dmetric, &MET_CHEBYSHEV, "Chebyshev (L-infinity) distance (scipy.spatial.distance.chebyshev)."),
    ROUTINE("distance.cosine", 1, "u, v, w=None", "out", r_dmetric, &MET_COSINE, "Cosine distance between two 1-D arrays (scipy.spatial.distance.cosine)."),
    ROUTINE("distance.correlation", 1, "u, v, w=None, centered=True", "out", r_dmetric, &MET_CORRELATION, "Correlation distance between two 1-D arrays (scipy.spatial.distance.correlation)."),
    ROUTINE("distance.hamming", 1, "u, v, w=None", "out", r_dmetric, &MET_HAMMING, "Hamming distance, the proportion of disagreeing components (scipy.spatial.distance.hamming)."),
    ROUTINE("distance.braycurtis", 1, "u, v, w=None", "out", r_dmetric, &MET_BRAYCURTIS, "Bray-Curtis distance between two 1-D arrays (scipy.spatial.distance.braycurtis)."),
    ROUTINE("distance.canberra", 1, "u, v, w=None", "out", r_dmetric, &MET_CANBERRA, "Canberra distance between two 1-D arrays (scipy.spatial.distance.canberra)."),
    ROUTINE("distance.minkowski", 1, "u, v, p=2, w=None", "out", r_dmetric, &MET_MINKOWSKI, "Minkowski distance of order p (scipy.spatial.distance.minkowski)."),
    ROUTINE("distance.dice", 1, "u, v, w=None", "out", r_bmetric, &BM_DICE, "Dice dissimilarity between two boolean 1-D arrays (scipy.spatial.distance.dice)."),
    ROUTINE("distance.jaccard", 1, "u, v, w=None", "out", r_bmetric, &BM_JACCARD, "Jaccard-Needham dissimilarity between two boolean 1-D arrays (scipy.spatial.distance.jaccard)."),
    ROUTINE("distance.rogerstanimoto", 1, "u, v, w=None", "out", r_bmetric, &BM_ROGERS, "Rogers-Tanimoto dissimilarity between two boolean 1-D arrays (scipy.spatial.distance.rogerstanimoto)."),
    ROUTINE("distance.russellrao", 1, "u, v, w=None", "out", r_bmetric, &BM_RUSSELL, "Russell-Rao dissimilarity between two boolean 1-D arrays (scipy.spatial.distance.russellrao)."),
    ROUTINE("distance.sokalsneath", 1, "u, v, w=None", "out", r_bmetric, &BM_SOKALS, "Sokal-Sneath dissimilarity between two boolean 1-D arrays (scipy.spatial.distance.sokalsneath)."),
    ROUTINE("distance.yule", 1, "u, v, w=None", "out", r_bmetric, &BM_YULE, "Yule dissimilarity between two boolean 1-D arrays (scipy.spatial.distance.yule)."),
    ROUTINE("distance.mahalanobis", 1, "u, v, VI", "out", r_mahalanobis, NULL, "Mahalanobis distance given the inverse covariance VI (scipy.spatial.distance.mahalanobis)."),
    ROUTINE("distance.seuclidean", 1, "u, v, V", "out", r_seuclidean, NULL, "Standardized Euclidean distance given the variance vector V (scipy.spatial.distance.seuclidean)."),
    ROUTINE("distance.jensenshannon", 1, "p, q, base=None", "out", r_jensenshannon, NULL, "Jensen-Shannon distance between two probability arrays (scipy.spatial.distance.jensenshannon)."),
    ROUTINE("distance.num_obs_dm", 1, "d", "out", r_num_obs_dm, NULL, "Number of observations in a square distance matrix (scipy.spatial.distance.num_obs_dm)."),
    ROUTINE("distance.num_obs_y", 1, "Y", "out", r_num_obs_y, NULL, "Number of observations in a condensed distance vector (scipy.spatial.distance.num_obs_y)."),
    ROUTINE("distance.is_valid_dm", 1, "D, tol=0.0, throw=False, name='D', warning=False", "out", r_is_valid_dm, NULL, "Whether D is a valid redundant distance matrix (scipy.spatial.distance.is_valid_dm)."),
    ROUTINE("distance.is_valid_y", 1, "y, warning=False, throw=False, name=None", "out", r_is_valid_y, NULL, "Whether y is a valid condensed distance vector (scipy.spatial.distance.is_valid_y)."),
};

const fn_table TSR_SPATIAL_DISTANCE_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
