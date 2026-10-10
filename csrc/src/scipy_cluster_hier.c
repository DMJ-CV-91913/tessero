/* scipy.cluster.hierarchy ("hierarchy." prefix -> Tessero\Cluster\Hierarchy facade).
 *
 * Agglomerative hierarchical clustering (linkage and its per-method aliases) plus the pure-numeric routines
 * that operate on a linkage matrix Z (cophenet, maxdists, inconsistent, fcluster, leaves_list, validity, ...).
 * Deterministic arithmetic; both backends read this table at runtime. Z rows are [c1, c2, dist, count] with
 * c1 < c2, new clusters numbered n, n+1, ...; this matches scipy's linkage layout exactly.
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* Lance-Williams update of the merged cluster's distance to k. For ward/centroid/median the stored values are
   SQUARED distances; for single/complete/average/weighted they are the distances themselves. */
static double hier_lw(int method, double dik, double djk, double dij, double ni, double nj, double nk)
{
    switch (method) {
    case 0: return dik < djk ? dik : djk;                                   /* single   */
    case 1: return dik > djk ? dik : djk;                                   /* complete */
    case 2: return (ni * dik + nj * djk) / (ni + nj);                       /* average  */
    case 3: return (dik + djk) / 2.0;                                       /* weighted */
    case 4: return ((ni + nk) * dik + (nj + nk) * djk - nk * dij) / (ni + nj + nk);        /* ward     */
    case 5: return (ni * dik + nj * djk) / (ni + nj) - ni * nj * dij / ((ni + nj) * (ni + nj)); /* centroid */
    default: return dik / 2.0 + djk / 2.0 - dij / 4.0;                      /* median   */
    }
}

/* run agglomeration over the n-by-n working matrix W (modified in place), writing Z (n-1 rows x 4). */
static void hier_core(double *W, int64_t n, int method, double *Z)
{
    const int sq = method >= 4;
    int64_t *act = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *sz = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *cid = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!act || !sz || !cid) { free(act); free(sz); free(cid); return; }
    for (int64_t i = 0; i < n; i++) { act[i] = i; sz[i] = 1; cid[i] = i; }
    int64_t na = n, nxt = n;
    for (int64_t step = 0; step < n - 1; step++) {
        double best = INFINITY; int64_t bi = 0, bj = 1;
        for (int64_t a = 0; a < na; a++)
            for (int64_t b = a + 1; b < na; b++) { double d = W[act[a] * n + act[b]]; if (d < best) { best = d; bi = a; bj = b; } }
        int64_t A = act[bi], B = act[bj]; double dij = best;
        int64_t c1 = cid[A] < cid[B] ? cid[A] : cid[B], c2 = cid[A] < cid[B] ? cid[B] : cid[A];
        Z[step * 4 + 0] = (double)c1; Z[step * 4 + 1] = (double)c2;
        Z[step * 4 + 2] = sq ? sqrt(dij) : dij; Z[step * 4 + 3] = (double)(sz[A] + sz[B]);
        for (int64_t a = 0; a < na; a++) { int64_t k = act[a]; if (k == A || k == B) continue;
            double nv = hier_lw(method, W[A * n + k], W[B * n + k], dij, (double)sz[A], (double)sz[B], (double)sz[k]);
            W[A * n + k] = W[k * n + A] = nv; }
        sz[A] += sz[B]; cid[A] = nxt++;
        for (int64_t a = bj; a < na - 1; a++) act[a] = act[a + 1];
        na--;
    }
    free(act); free(sz); free(cid);
}

/* linkage(y, method): y is a condensed distance vector (1-D) or an observation matrix (2-D, Euclidean). */
static int linkage_impl(const tsr_arg *a, tsr_result *res, int method)
{
    if (a->kind != 3) { fn_set_error("linkage: y must be a 1-D condensed distance or 2-D observation array"); return TSR_EARG; }
    const int sq = method >= 4;
    int64_t n; double *W = NULL; int rc = TSR_OK;
    if (a->arr.ndim == 1) {
        int64_t L; double *y = fn_arg_doubles(a, &L);
        if (!y) return TSR_ENOMEM;
        n = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)L)) / 2.0 + 0.5);
        W = (double *)calloc((size_t)(n * n), sizeof(double));
        if (!W) { fn_free_doubles(y, L); return TSR_ENOMEM; }
        int64_t idx = 0;
        for (int64_t i = 0; i < n; i++) for (int64_t j = i + 1; j < n; j++) { double d = y[idx++]; if (sq) d *= d; W[i * n + j] = W[j * n + i] = d; }
        fn_free_doubles(y, L);
    } else if (a->arr.ndim == 2) {
        int64_t m = a->arr.shape[0], dim = a->arr.shape[1], lo; double *obs = fn_arg_doubles(a, &lo);
        if (!obs) return TSR_ENOMEM;
        n = m; W = (double *)calloc((size_t)(n * n), sizeof(double));
        if (!W) { fn_free_doubles(obs, lo); return TSR_ENOMEM; }
        for (int64_t i = 0; i < n; i++) for (int64_t j = i + 1; j < n; j++) {
            double s = 0.0; for (int64_t c = 0; c < dim; c++) { double df = obs[i * dim + c] - obs[j * dim + c]; s += df * df; }
            double d = sq ? s : sqrt(s); W[i * n + j] = W[j * n + i] = d;
        }
        fn_free_doubles(obs, lo);
    } else { fn_set_error("linkage: y must be 1-D or 2-D"); return TSR_EARG; }
    for (int64_t i = 0; i < n; i++) W[i * n + i] = INFINITY;
    double *Z = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n - 1, 4});
    if (!Z) rc = TSR_ENOMEM; else hier_core(W, n, method, Z);
    free(W);
    return rc;
}
static int hier_method_code(const char *m)
{
    if (!strcmp(m, "single")) return 0;
    if (!strcmp(m, "complete")) return 1;
    if (!strcmp(m, "average")) return 2;
    if (!strcmp(m, "weighted")) return 3;
    if (!strcmp(m, "ward")) return 4;
    if (!strcmp(m, "centroid")) return 5;
    if (!strcmp(m, "median")) return 6;
    return -1;
}
static int r_linkage(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int code = 0;
    if (nargs > 1 && args[1].kind == 2) { code = hier_method_code(args[1].str); if (code < 0) { fn_set_error("linkage: unknown method"); return TSR_EARG; } }
    return linkage_impl(&args[0], res, code);
}
#define HIER_ALIAS(fn, code) \
    static int fn(const void *ctx, const tsr_arg *a, int n, tsr_result *r, int nr) \
    { (void)ctx; (void)n; (void)nr; return linkage_impl(&a[0], r, code); }
HIER_ALIAS(r_single, 0)
HIER_ALIAS(r_complete, 1)
HIER_ALIAS(r_average, 2)
HIER_ALIAS(r_weighted, 3)
HIER_ALIAS(r_ward, 4)
HIER_ALIAS(r_centroid, 5)
HIER_ALIAS(r_median, 6)

/* ------------------------------------------------------------------ routines over an existing linkage Z */
static void hier_bool(tsr_result *r, int b) { memset(r, 0, sizeof *r); r->kind = 4; r->num = b ? 1.0 : 0.0; }

/* Z is a valid (m x 4) linkage matrix argument? */
static int is_Z(const tsr_arg *a) { return a->kind == 3 && a->arr.ndim == 2 && a->arr.shape[1] == 4; }

/* out[i] = max over the subtree rooted at merge i (cluster n_obs+i) of val[j] for every merge j in it.
   Children always have a lower merge index, so out[] is filled bottom-up in one forward pass. */
static void hier_subtree_max(const double *Z, int64_t m, int64_t n_obs, const double *val, double *out)
{
    for (int64_t i = 0; i < m; i++) {
        double mx = val[i];
        int64_t c0 = (int64_t)Z[i * 4 + 0], c1 = (int64_t)Z[i * 4 + 1];
        if (c0 >= n_obs && out[c0 - n_obs] > mx) mx = out[c0 - n_obs];
        if (c1 >= n_obs && out[c1 - n_obs] > mx) mx = out[c1 - n_obs];
        out[i] = mx;
    }
}

static int r_num_obs_linkage(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("num_obs_linkage: Z must be (m, 4)"); return TSR_EARG; }
    fn_result_int(&res[0], args[0].arr.shape[0] + 1);
    return TSR_OK;
}

static int r_is_valid_linkage(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { hier_bool(&res[0], 0); return TSR_OK; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    int64_t n_obs = m + 1; int ok = 1;
    for (int64_t i = 0; i < m && ok; i++) {
        double c0 = Z[i * 4 + 0], c1 = Z[i * 4 + 1], d = Z[i * 4 + 2];
        if (c0 < 0 || c1 < 0 || d < 0 || c0 >= (double)(n_obs + i) || c1 >= (double)(n_obs + i)) ok = 0;
    }
    fn_free_doubles(Z, L);
    hier_bool(&res[0], ok);
    return TSR_OK;
}

static int r_is_monotonic(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("is_monotonic: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    int mono = 1;
    for (int64_t i = 1; i < m; i++) if (Z[i * 4 + 2] < Z[(i - 1) * 4 + 2]) { mono = 0; break; }
    fn_free_doubles(Z, L);
    hier_bool(&res[0], mono);
    return TSR_OK;
}

static int r_is_valid_im(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { hier_bool(&res[0], 0); return TSR_OK; }
    int64_t m = args[0].arr.shape[0], L; double *R = fn_arg_doubles(&args[0], &L);
    if (!R) return TSR_ENOMEM;
    int ok = 1;
    for (int64_t i = 0; i < m && ok; i++) if (R[i * 4 + 0] < 0 || R[i * 4 + 1] < 0 || R[i * 4 + 2] < 0) ok = 0;
    fn_free_doubles(R, L);
    hier_bool(&res[0], ok);
    return TSR_OK;
}

static int r_correspond(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2) { fn_set_error("correspond: Z and Y are required"); return TSR_EARG; }
    const tsr_arg *Za = &args[0], *Ya = &args[1];
    if (!is_Z(Za) || Ya->kind != 3 || Ya->arr.ndim != 1) { hier_bool(&res[0], 0); return TSR_OK; }
    int64_t n_obs_z = Za->arr.shape[0] + 1, Ly = Ya->arr.shape[0];
    int64_t d = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)Ly)) / 2.0 + 0.5);
    hier_bool(&res[0], d == n_obs_z && d * (d - 1) / 2 == Ly);
    return TSR_OK;
}

static int r_maxdists(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("maxdists: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    double *val = (double *)malloc((size_t)m * sizeof(double));
    int64_t sh[1] = {m};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!val || !out) { free(val); fn_free_doubles(Z, L); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) val[i] = Z[i * 4 + 2];
    hier_subtree_max(Z, m, m + 1, val, out);
    free(val); fn_free_doubles(Z, L);
    return TSR_OK;
}

/* collect link heights within `depth` levels at and below merge `node` (the node itself is level 1). */
static void incon_collect(const double *Z, int64_t n_obs, int64_t node, int depth, double *buf, int64_t *cnt)
{
    int64_t i = node - n_obs;
    buf[(*cnt)++] = Z[i * 4 + 2];
    if (depth <= 1) return;
    int64_t c0 = (int64_t)Z[i * 4 + 0], c1 = (int64_t)Z[i * 4 + 1];
    if (c0 >= n_obs) incon_collect(Z, n_obs, c0, depth - 1, buf, cnt);
    if (c1 >= n_obs) incon_collect(Z, n_obs, c1, depth - 1, buf, cnt);
}

static int r_inconsistent(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("inconsistent: Z must be (m, 4)"); return TSR_EARG; }
    int d = 2;
    if (nargs > 1) d = (int)((args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num);
    if (d < 1) { fn_set_error("inconsistent: d must be >= 1"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    int64_t n_obs = m + 1, sh[2] = {m, 4};
    double *R = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    double *buf = (double *)malloc((size_t)m * sizeof(double));
    if (!R || !buf) { free(buf); fn_free_doubles(Z, L); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) {
        int64_t cnt = 0;
        incon_collect(Z, n_obs, n_obs + i, d, buf, &cnt);
        double s = 0.0; for (int64_t k = 0; k < cnt; k++) s += buf[k];
        double mean = s / (double)cnt, var = 0.0;
        for (int64_t k = 0; k < cnt; k++) { double df = buf[k] - mean; var += df * df; }
        double sd = cnt < 2 ? 0.0 : sqrt(var / (double)(cnt - 1));        /* SciPy uses the sample std (ddof=1) */
        R[i * 4 + 0] = mean; R[i * 4 + 1] = sd; R[i * 4 + 2] = (double)cnt;
        R[i * 4 + 3] = (cnt < 2 || sd == 0.0) ? 0.0 : (Z[i * 4 + 2] - mean) / sd;
    }
    free(buf); fn_free_doubles(Z, L);
    return TSR_OK;
}

static int r_maxinconsts(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || !is_Z(&args[0]) || !is_Z(&args[1])) { fn_set_error("maxinconsts: Z and R must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], Lz, Lr; double *Z = fn_arg_doubles(&args[0], &Lz);
    if (!Z) return TSR_ENOMEM;
    double *R = fn_arg_doubles(&args[1], &Lr);
    double *val = R ? (double *)malloc((size_t)m * sizeof(double)) : NULL;
    int64_t sh[1] = {m};
    double *out = val ? (double *)fn_result_array(&res[0], TSR_F64, 1, sh) : NULL;
    if (!R || !val || !out) { free(val); if (R) fn_free_doubles(R, Lr); fn_free_doubles(Z, Lz); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) val[i] = R[i * 4 + 3];
    hier_subtree_max(Z, m, m + 1, val, out);
    free(val); fn_free_doubles(R, Lr); fn_free_doubles(Z, Lz);
    return TSR_OK;
}

static int r_maxRstat(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 3 || !is_Z(&args[0]) || !is_Z(&args[1])) { fn_set_error("maxRstat: Z and R must be (m, 4)"); return TSR_EARG; }
    int64_t col = (args[2].flags & 1) ? args[2].ival : (int64_t)args[2].num;
    if (col < 0 || col > 3) { fn_set_error("maxRstat: i must be in 0..3"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], Lz, Lr; double *Z = fn_arg_doubles(&args[0], &Lz);
    if (!Z) return TSR_ENOMEM;
    double *R = fn_arg_doubles(&args[1], &Lr);
    double *val = R ? (double *)malloc((size_t)m * sizeof(double)) : NULL;
    int64_t sh[1] = {m};
    double *out = val ? (double *)fn_result_array(&res[0], TSR_F64, 1, sh) : NULL;
    if (!R || !val || !out) { free(val); if (R) fn_free_doubles(R, Lr); fn_free_doubles(Z, Lz); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) val[i] = R[i * 4 + col];
    hier_subtree_max(Z, m, m + 1, val, out);
    free(val); fn_free_doubles(R, Lr); fn_free_doubles(Z, Lz);
    return TSR_OK;
}

/* leaves_list(Z): the leaf ids in the order a left-first pre-order dendrogram traversal visits them. */
static int r_leaves_list(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("leaves_list: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    int64_t n_obs = m + 1, sh[1] = {n_obs};
    int64_t *out = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, sh);
    int64_t *stack = (int64_t *)malloc((size_t)(2 * n_obs) * sizeof(int64_t));
    if (!out || !stack) { free(stack); fn_free_doubles(Z, L); return TSR_ENOMEM; }
    int64_t sp = 0, oi = 0;
    stack[sp++] = n_obs + m - 1;                                   /* root cluster */
    while (sp > 0) {
        int64_t node = stack[--sp];
        if (node < n_obs) { out[oi++] = node; }
        else { int64_t i = node - n_obs; stack[sp++] = (int64_t)Z[i * 4 + 1]; stack[sp++] = (int64_t)Z[i * 4 + 0]; }
    }
    free(stack); fn_free_doubles(Z, L);
    return TSR_OK;
}

/* cophenet(Z): condensed cophenetic distances; d(i, j) is the height of the merge that first unites i and j. */
static int r_cophenet(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("cophenet: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    int64_t n_obs = m + 1, nc = n_obs * (n_obs - 1) / 2, total = n_obs + m, sh[1] = {nc};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    int64_t **mem = (int64_t **)calloc((size_t)total, sizeof(int64_t *));
    int64_t *msz = (int64_t *)calloc((size_t)total, sizeof(int64_t));
    if (!out || !mem || !msz) { free(mem); free(msz); fn_free_doubles(Z, L); return TSR_ENOMEM; }
    int rc = TSR_OK;
    for (int64_t i = 0; i < n_obs; i++) { mem[i] = (int64_t *)malloc(sizeof(int64_t)); if (mem[i]) { mem[i][0] = i; msz[i] = 1; } else rc = TSR_ENOMEM; }
    for (int64_t i = 0; i < m && rc == TSR_OK; i++) {
        int64_t l = (int64_t)Z[i * 4 + 0], r = (int64_t)Z[i * 4 + 1], nl = msz[l], nr = msz[r];
        double h = Z[i * 4 + 2];
        for (int64_t x = 0; x < nl; x++) for (int64_t y = 0; y < nr; y++) {
            int64_t p = mem[l][x], q = mem[r][y], aa = p < q ? p : q, bb = p < q ? q : p;
            out[aa * n_obs - aa * (aa + 1) / 2 + (bb - aa - 1)] = h;
        }
        int64_t id = n_obs + i;
        mem[id] = (int64_t *)malloc((size_t)(nl + nr) * sizeof(int64_t));
        if (!mem[id]) { rc = TSR_ENOMEM; break; }
        memcpy(mem[id], mem[l], (size_t)nl * sizeof(int64_t));
        memcpy(mem[id] + nl, mem[r], (size_t)nr * sizeof(int64_t));
        msz[id] = nl + nr;
    }
    for (int64_t i = 0; i < total; i++) free(mem[i]);
    free(mem); free(msz); fn_free_doubles(Z, L);
    return rc;
}

/* to_mlab_linkage(Z): the MATLAB form -- the first three columns, with the cluster indices made 1-based. */
static int r_to_mlab_linkage(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("to_mlab_linkage: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    int64_t sh[2] = {m, 3};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    if (!out) { fn_free_doubles(Z, L); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) { out[i * 3 + 0] = Z[i * 4 + 0] + 1.0; out[i * 3 + 1] = Z[i * 4 + 1] + 1.0; out[i * 3 + 2] = Z[i * 4 + 2]; }
    fn_free_doubles(Z, L);
    return TSR_OK;
}

/* from_mlab_linkage(Z): invert to_mlab_linkage -- 0-based indices and a recomputed cluster-size column. */
static int r_from_mlab_linkage(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    const tsr_arg *a = &args[0];
    if (a->kind != 3 || a->arr.ndim != 2 || a->arr.shape[1] != 3) { fn_set_error("from_mlab_linkage: Z must be (m, 3)"); return TSR_EARG; }
    int64_t m = a->arr.shape[0], L; double *Z = fn_arg_doubles(a, &L);
    if (!Z) return TSR_ENOMEM;
    int64_t n_obs = m + 1, total = n_obs + m, sh[2] = {m, 4};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    int64_t *sz = (int64_t *)malloc((size_t)total * sizeof(int64_t));
    if (!out || !sz) { free(sz); fn_free_doubles(Z, L); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n_obs; i++) sz[i] = 1;
    for (int64_t i = 0; i < m; i++) {
        int64_t l = (int64_t)(Z[i * 3 + 0] - 1.0), r = (int64_t)(Z[i * 3 + 1] - 1.0), cnt = sz[l] + sz[r];
        out[i * 4 + 0] = (double)l; out[i * 4 + 1] = (double)r; out[i * 4 + 2] = Z[i * 3 + 2]; out[i * 4 + 3] = (double)cnt;
        sz[n_obs + i] = cnt;
    }
    free(sz); fn_free_doubles(Z, L);
    return TSR_OK;
}

static const fn_def DEFS[] = {
    ROUTINE("hierarchy.linkage", 1, "y, method='single'", "Z", r_linkage, NULL, "Agglomerative hierarchical clustering linkage matrix (scipy.cluster.hierarchy.linkage)."),
    ROUTINE("hierarchy.single", 1, "y", "Z", r_single, NULL, "Single/nearest-point linkage (scipy.cluster.hierarchy.single)."),
    ROUTINE("hierarchy.complete", 1, "y", "Z", r_complete, NULL, "Complete/farthest-point linkage (scipy.cluster.hierarchy.complete)."),
    ROUTINE("hierarchy.average", 1, "y", "Z", r_average, NULL, "Average (UPGMA) linkage (scipy.cluster.hierarchy.average)."),
    ROUTINE("hierarchy.weighted", 1, "y", "Z", r_weighted, NULL, "Weighted (WPGMA) linkage (scipy.cluster.hierarchy.weighted)."),
    ROUTINE("hierarchy.ward", 1, "y", "Z", r_ward, NULL, "Ward variance-minimizing linkage (scipy.cluster.hierarchy.ward)."),
    ROUTINE("hierarchy.centroid", 1, "y", "Z", r_centroid, NULL, "Centroid (UPGMC) linkage (scipy.cluster.hierarchy.centroid)."),
    ROUTINE("hierarchy.median", 1, "y", "Z", r_median, NULL, "Median (WPGMC) linkage (scipy.cluster.hierarchy.median)."),
    ROUTINE("hierarchy.num_obs_linkage", 1, "Z", "n", r_num_obs_linkage, NULL, "Number of original observations in a linkage matrix (scipy.cluster.hierarchy.num_obs_linkage)."),
    ROUTINE("hierarchy.is_valid_linkage", 1, "Z", "valid", r_is_valid_linkage, NULL, "Whether Z is a valid linkage matrix (scipy.cluster.hierarchy.is_valid_linkage)."),
    ROUTINE("hierarchy.is_monotonic", 1, "Z", "monotonic", r_is_monotonic, NULL, "Whether the linkage is monotonic -- non-decreasing merge heights (scipy.cluster.hierarchy.is_monotonic)."),
    ROUTINE("hierarchy.is_valid_im", 1, "R", "valid", r_is_valid_im, NULL, "Whether R is a valid inconsistency matrix (scipy.cluster.hierarchy.is_valid_im)."),
    ROUTINE("hierarchy.correspond", 2, "Z, Y", "corr", r_correspond, NULL, "Whether a linkage Z and condensed distances Y have matching numbers of observations (scipy.cluster.hierarchy.correspond)."),
    ROUTINE("hierarchy.maxdists", 1, "Z", "MD", r_maxdists, NULL, "Maximum merge distance below each non-singleton cluster (scipy.cluster.hierarchy.maxdists)."),
    ROUTINE("hierarchy.inconsistent", 1, "Z, d=2", "R", r_inconsistent, NULL, "Inconsistency matrix of a linkage (scipy.cluster.hierarchy.inconsistent)."),
    ROUTINE("hierarchy.maxinconsts", 2, "Z, R", "MI", r_maxinconsts, NULL, "Maximum inconsistency coefficient below each non-singleton cluster (scipy.cluster.hierarchy.maxinconsts)."),
    ROUTINE("hierarchy.maxRstat", 3, "Z, R, i", "MR", r_maxRstat, NULL, "Maximum of inconsistency-matrix statistic i below each non-singleton cluster (scipy.cluster.hierarchy.maxRstat)."),
    ROUTINE("hierarchy.leaves_list", 1, "Z", "L", r_leaves_list, NULL, "Leaf ids in left-first pre-order dendrogram order (scipy.cluster.hierarchy.leaves_list)."),
    ROUTINE("hierarchy.cophenet", 1, "Z", "d", r_cophenet, NULL, "Condensed cophenetic distances of a linkage (scipy.cluster.hierarchy.cophenet)."),
    ROUTINE("hierarchy.to_mlab_linkage", 1, "Z", "mZ", r_to_mlab_linkage, NULL, "Convert a linkage matrix to MATLAB form (scipy.cluster.hierarchy.to_mlab_linkage)."),
    ROUTINE("hierarchy.from_mlab_linkage", 1, "Z", "Z", r_from_mlab_linkage, NULL, "Convert a MATLAB-form linkage matrix to SciPy form (scipy.cluster.hierarchy.from_mlab_linkage)."),
};

const fn_table TSR_SCIPY_HIERARCHY_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
