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

static const fn_def DEFS[] = {
    ROUTINE("distance.pdist", 1, "X, metric=None, p=None", "out", r_pdist, NULL, "Pairwise distances between the rows of X, condensed (scipy.spatial.distance.pdist); real float64."),
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
