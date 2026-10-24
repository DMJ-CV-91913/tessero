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
static int hier_core(double *W, int64_t n, int method, double *Z)
{
    const int sq = method >= 4;
    int64_t *act = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *sz = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *cid = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!act || !sz || !cid) { free(act); free(sz); free(cid); return TSR_ENOMEM; }
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
    return TSR_OK;
}

/* SciPy's linkage refusals before any work: an empty condensed matrix (num_obs_y) and non-finite distances */
static int hier_check_y(const double *y, int64_t L)
{
    if (L == 0) { fn_set_error("The number of observations cannot be determined on an empty distance matrix."); return TSR_EARG; }
    for (int64_t i = 0; i < L; i++)
        if (!isfinite(y[i])) { fn_set_error("The condensed distance matrix must contain only finite values."); return TSR_EARG; }
    return TSR_OK;
}

/* linkage(y, method): y is a condensed distance vector (1-D) or an observation matrix (2-D, Euclidean). */
static int linkage_impl(const tsr_arg *a, tsr_result *res, int method)
{
    if (a->kind != 3) { fn_set_error("linkage: y must be a 1-D condensed distance or 2-D observation array"); return TSR_EARG; }
    const int sq = method >= 4;
    int64_t n, wbytes = 0; double *W = NULL; int rc = TSR_OK;   /* W: the n*n workspace, counted (tsr_alloc); every entry is written below */
    if (a->arr.ndim == 1) {
        int64_t L; double *y = fn_arg_doubles(a, &L);
        if (!y) return TSR_ENOMEM;
        n = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)L)) / 2.0 + 0.5);
        /* the workspace and the fill loop below read n*(n-1)/2 entries of y: a length that is not a binomial
           coefficient (e.g. 5) would read past it (SciPy raises the same ValueError) */
        if (n < 1 || tsr_count_mul(n, n - 1) / 2 != L) {
            fn_free_doubles(y, L);
            fn_set_error("linkage: Length n of condensed distance matrix 'y' must be a binomial coefficient, i.e. "
                         "there must be a k such that (k \\choose 2)=n)!");
            return TSR_EARG;
        }
        if ((rc = hier_check_y(y, L)) != TSR_OK) { fn_free_doubles(y, L); return rc; }
        wbytes = tsr_bytes_n(tsr_count_mul(n, n), sizeof(double), 0);
        W = (double *)tsr_alloc(wbytes);
        if (!W) { fn_free_doubles(y, L); return TSR_ENOMEM; }
        int64_t idx = 0;
        for (int64_t i = 0; i < n; i++) for (int64_t j = i + 1; j < n; j++) { double d = y[idx++]; if (sq) d *= d; W[i * n + j] = W[j * n + i] = d; }
        fn_free_doubles(y, L);
    } else if (a->arr.ndim == 2) {
        int64_t m = a->arr.shape[0], dim = a->arr.shape[1], lo;
        if (m < 2) { fn_set_error("The number of observations cannot be determined on an empty distance matrix."); return TSR_EARG; }
        double *obs = fn_arg_doubles(a, &lo);
        if (!obs) return TSR_ENOMEM;
        n = m; wbytes = tsr_bytes_n(tsr_count_mul(n, n), sizeof(double), 0); W = (double *)tsr_alloc(wbytes);
        if (!W) { fn_free_doubles(obs, lo); return TSR_ENOMEM; }
        for (int64_t i = 0; i < n && rc == TSR_OK; i++) for (int64_t j = i + 1; j < n; j++) {
            double s = 0.0; for (int64_t c = 0; c < dim; c++) { double df = obs[i * dim + c] - obs[j * dim + c]; s += df * df; }
            if (!isfinite(s)) { fn_set_error("The condensed distance matrix must contain only finite values."); rc = TSR_EARG; break; }
            double d = sq ? s : sqrt(s); W[i * n + j] = W[j * n + i] = d;
        }
        fn_free_doubles(obs, lo);
        if (rc != TSR_OK) { tsr_free(W, wbytes); return rc; }
    } else { fn_set_error("linkage: y must be 1-D or 2-D"); return TSR_EARG; }
    for (int64_t i = 0; i < n; i++) W[i * n + i] = INFINITY;
    double *Z = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n - 1, 4});
    if (!Z) rc = TSR_ENOMEM; else rc = hier_core(W, n, method, Z);
    tsr_free(W, wbytes);
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

/* SciPy's _is_valid_linkage on the (m x 4) contents, in its order: 0 when valid, else the index of the failed check
   (its message below). Every routine here indexes Z by the child ids, so two refusals go beyond SciPy: ids must be
   finite, and a one-row Z (which SciPy accepts unchecked) must still name the two leaves. Ids truncate toward zero
   as SciPy's C code reads them, and two ids naming the same cluster after truncation are a repeat. */
enum { HZ_OK, HZ_EMPTY, HZ_NONFINITE, HZ_NEGIDX, HZ_NEGDIST, HZ_NEGCNT, HZ_EXCESS, HZ_BEFORE, HZ_REPEAT, HZ_NOMEM };
static int hier_Z_status(const double *Z, int64_t m)
{
    if (m < 1) return HZ_EMPTY;
    int negidx = 0, negdist = 0, negcnt = 0, excess = 0, before = 0;
    for (int64_t i = 0; i < m; i++) {
        const double c0 = Z[i * 4 + 0], c1 = Z[i * 4 + 1];
        if (!isfinite(c0) || !isfinite(c1)) return HZ_NONFINITE;
        if (c0 < 0 || c1 < 0) negidx = 1;
        if (Z[i * 4 + 2] < 0) negdist = 1;
        if (Z[i * 4 + 3] < 0) negcnt = 1;
        if (Z[i * 4 + 3] > (double)(m + 1)) excess = 1;
        if ((c0 > c1 ? c0 : c1) >= (double)(m + 1 + i)) before = 1;
    }
    if (negidx) return HZ_NEGIDX;
    if (m == 1) {                                     /* SciPy checks nothing more for one row, but the two ids must */
        if (before) return HZ_BEFORE;                 /* be the two leaves: cophenet's pair index for (0, 0) is -1 */
        return (int64_t)Z[0] == (int64_t)Z[1] ? HZ_REPEAT : HZ_OK;
    }
    if (negdist) return HZ_NEGDIST;
    if (negcnt) return HZ_NEGCNT;
    if (excess) return HZ_EXCESS;
    if (before) return HZ_BEFORE;
    unsigned char *seen = (unsigned char *)calloc((size_t)(2 * m + 1), 1);   /* ids are now in [0, 2m) */
    if (!seen) return HZ_NOMEM;
    int rep = 0;
    for (int64_t i = 0; i < 2 * m && !rep; i++) {
        const int64_t id = (int64_t)Z[(i >> 1) * 4 + (i & 1)];
        if (seen[id]) rep = 1; else seen[id] = 1;
    }
    free(seen);
    return rep ? HZ_REPEAT : HZ_OK;
}

/* raise SciPy's message for a failed linkage check; name is "'Z' " or "" (correspond passes no name) */
static int hier_Z_raise(int st, const char *name)
{
    switch (st) {
    case HZ_OK: return TSR_OK;
    case HZ_NOMEM: return TSR_ENOMEM;
    case HZ_EMPTY: fn_set_error("Linkage must be computed on at least two observations."); break;
    case HZ_NONFINITE: fn_set_error("Linkage %scontains non-finite indices.", name); break;
    case HZ_NEGIDX: fn_set_error("Linkage %scontains negative indices.", name); break;
    case HZ_NEGDIST: fn_set_error("Linkage %scontains negative distances.", name); break;
    case HZ_NEGCNT: fn_set_error("Linkage %scontains negative counts.", name); break;
    case HZ_EXCESS: fn_set_error("Linkage %scontains excessive observations in a cluster", name); break;
    case HZ_BEFORE: fn_set_error("Linkage %suses non-singleton cluster before it is formed.", name); break;
    default: fn_set_error("Linkage %suses the same cluster more than once.", name); break;
    }
    return TSR_EARG;
}

/* the linkage argument converted and validated (SciPy's _is_valid_linkage(Z, throw=True)): NULL with *rc set, or
   the doubles (released with fn_free_doubles(Z, *L)). The caller has checked the (m, 4) shape. */
static double *hier_Z(const tsr_arg *a, int64_t *L, const char *name, int *rc)
{
    double *Z = fn_arg_doubles(a, L);
    if (!Z) { *rc = TSR_ENOMEM; return NULL; }
    *rc = hier_Z_raise(hier_Z_status(Z, a->arr.shape[0]), name);
    if (*rc != TSR_OK) { fn_free_doubles(Z, *L); return NULL; }
    return Z;
}

/* SciPy's _is_valid_im(R, throw=True, name='R') on an (r x 4) R (its message has the doubled space) */
static int hier_check_R(const double *R, int64_t r)
{
    if (r < 1) { fn_set_error("Inconsistency matrix 'R' must have at least one row."); return TSR_EARG; }
    int mean = 0, sd = 0, cnt = 0;
    for (int64_t i = 0; i < r; i++) { if (R[i * 4] < 0) mean = 1; if (R[i * 4 + 1] < 0) sd = 1; if (R[i * 4 + 2] < 0) cnt = 1; }
    if (mean) { fn_set_error("Inconsistency matrix 'R'  contains negative link height means."); return TSR_EARG; }
    if (sd) { fn_set_error("Inconsistency matrix 'R'  contains negative link height standard deviations."); return TSR_EARG; }
    if (cnt) { fn_set_error("Inconsistency matrix 'R'  contains negative link counts."); return TSR_EARG; }
    return TSR_OK;
}

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
    int64_t L; int rc; double *Z = hier_Z(&args[0], &L, "'Z' ", &rc);
    if (!Z) return rc;
    fn_free_doubles(Z, L);
    fn_result_int(&res[0], args[0].arr.shape[0] + 1);
    return TSR_OK;
}

/* is_valid_linkage(Z): SciPy's checks as a bool (a one-row Z is valid unchecked in SciPy; non-finite ids are not) */
static int r_is_valid_linkage(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { hier_bool(&res[0], 0); return TSR_OK; }
    int64_t m = args[0].arr.shape[0], L; double *Z = fn_arg_doubles(&args[0], &L);
    if (!Z) return TSR_ENOMEM;
    int st = HZ_OK;
    if (m == 1) { if (!isfinite(Z[0]) || !isfinite(Z[1])) st = HZ_NONFINITE; }
    else st = hier_Z_status(Z, m);
    fn_free_doubles(Z, L);
    if (st == HZ_NOMEM) return TSR_ENOMEM;
    hier_bool(&res[0], st == HZ_OK);
    return TSR_OK;
}

static int r_is_monotonic(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("is_monotonic: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; int zrc; double *Z = hier_Z(&args[0], &L, "'Z' ", &zrc);
    if (!Z) return zrc;
    int mono = 1;
    for (int64_t i = 1; i < m; i++) if (Z[i * 4 + 2] < Z[(i - 1) * 4 + 2]) { mono = 0; break; }
    fn_free_doubles(Z, L);
    hier_bool(&res[0], mono);
    return TSR_OK;
}

static int r_is_valid_im(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0]) || args[0].arr.shape[0] < 1) { hier_bool(&res[0], 0); return TSR_OK; }
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
    { int64_t L; int zrc; double *Z = hier_Z(Za, &L, "", &zrc); if (!Z) return zrc; fn_free_doubles(Z, L); }
    int64_t n_obs_z = Za->arr.shape[0] + 1, Ly = Ya->arr.shape[0];
    int64_t d = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)Ly)) / 2.0 + 0.5);
    hier_bool(&res[0], d == n_obs_z && d * (d - 1) / 2 == Ly);
    return TSR_OK;
}

static int r_maxdists(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("maxdists: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; int zrc; double *Z = hier_Z(&args[0], &L, "'Z' ", &zrc);
    if (!Z) return zrc;
    double *val = (double *)malloc((size_t)m * sizeof(double));
    int64_t sh[1] = {m};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!val || !out) { free(val); fn_free_doubles(Z, L); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) val[i] = Z[i * 4 + 2];
    hier_subtree_max(Z, m, m + 1, val, out);
    free(val); fn_free_doubles(Z, L);
    return TSR_OK;
}

/* collect link heights within `depth` levels at and below merge `node` (the node itself is level 1), in pre-order
   (node, left subtree, right subtree). Iterative, so the depth of the tree never reaches the C stack; on a valid
   linkage (hier_Z) each merge is reached once, so buf and the stack (2 x m entries) never overflow. */
static void incon_collect(const double *Z, int64_t n_obs, int64_t node, int64_t depth, double *buf, int64_t *cnt,
                          int64_t *stack)
{
    int64_t sp = 0;
    stack[sp++] = node; stack[sp++] = depth;
    while (sp > 0) {
        const int64_t dep = stack[--sp], nd = stack[--sp], i = nd - n_obs;
        buf[(*cnt)++] = Z[i * 4 + 2];
        if (dep <= 1) continue;
        const int64_t c0 = (int64_t)Z[i * 4 + 0], c1 = (int64_t)Z[i * 4 + 1];
        if (c1 >= n_obs) { stack[sp++] = c1; stack[sp++] = dep - 1; }     /* right after left: pushed first */
        if (c0 >= n_obs) { stack[sp++] = c0; stack[sp++] = dep - 1; }
    }
}

/* fill the (m x 4) inconsistency matrix R from a valid linkage Z at depth d (d = 0 is the node alone, as d = 1,
   in SciPy's C code). Returns TSR_ENOMEM when the scratch cannot be allocated. */
static int fill_inconsistent(const double *Z, int64_t m, int64_t n_obs, int64_t d, double *R)
{
    double *buf = (double *)malloc((size_t)m * sizeof(double));
    int64_t *stack = (int64_t *)malloc((size_t)(2 * m) * sizeof(int64_t));
    if (!buf || !stack) { free(buf); free(stack); return TSR_ENOMEM; }
    for (int64_t i = 0; i < m; i++) {
        int64_t cnt = 0;
        incon_collect(Z, n_obs, n_obs + i, d, buf, &cnt, stack);
        double s = 0.0; for (int64_t k = 0; k < cnt; k++) s += buf[k];
        double mean = s / (double)cnt, var = 0.0;
        for (int64_t k = 0; k < cnt; k++) { double df = buf[k] - mean; var += df * df; }
        double sd = cnt < 2 ? 0.0 : sqrt(var / (double)(cnt - 1));        /* SciPy uses the sample std (ddof=1) */
        R[i * 4 + 0] = mean; R[i * 4 + 1] = sd; R[i * 4 + 2] = (double)cnt;
        R[i * 4 + 3] = (cnt < 2 || sd == 0.0) ? 0.0 : (Z[i * 4 + 2] - mean) / sd;
    }
    free(buf); free(stack);
    return TSR_OK;
}

/* inconsistent's depth d (also fcluster's and fclusterdata's `depth`): SciPy's "d != floor(d) or d < 0" check, then
   the C int its kernel takes */
static int hier_depth(const tsr_arg *a, int64_t *d)
{
    if (a->kind == 0) { *d = 2; return TSR_OK; }
    if (a->kind != 1 && a->kind != 4) { fn_set_error("The second argument d must be a nonnegative integer value."); return TSR_ETYPE; }
    const double v = (a->kind == 1 && (a->flags & 1)) ? (double)a->ival : a->num;
    if (!(v == floor(v)) || v < 0) { fn_set_error("The second argument d must be a nonnegative integer value."); return TSR_EARG; }
    if (v > 2147483647.0) { fn_set_error("Python int too large to convert to C int"); return TSR_EARG; }
    *d = (int64_t)v;
    return TSR_OK;
}

static int r_inconsistent(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("inconsistent: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; int rc; double *Z = hier_Z(&args[0], &L, "'Z' ", &rc);
    if (!Z) return rc;
    int64_t d = 2;
    if (nargs > 1 && (rc = hier_depth(&args[1], &d)) != TSR_OK) { fn_free_doubles(Z, L); return rc; }
    int64_t n_obs = m + 1, sh[2] = {m, 4};
    double *R = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    rc = R ? fill_inconsistent(Z, m, n_obs, d, R) : TSR_ENOMEM;
    fn_free_doubles(Z, L);
    return rc;
}

/* maxinconsts / maxRstat: a valid Z and R with the same number of rows (SciPy's checks), then the subtree maximum of
   R[:, col] */
static int hier_max_rstat(const tsr_arg *args, int64_t col, tsr_result *res)
{
    int64_t m = args[0].arr.shape[0], Lz, Lr; int rc;
    double *Z = hier_Z(&args[0], &Lz, "'Z' ", &rc);
    if (!Z) return rc;
    double *R = fn_arg_doubles(&args[1], &Lr);
    if (!R) { fn_free_doubles(Z, Lz); return TSR_ENOMEM; }
    rc = hier_check_R(R, args[1].arr.shape[0]);
    if (rc == TSR_OK && args[1].arr.shape[0] != m) {
        fn_set_error("The inconsistency matrix and linkage matrix each have a different number of rows.");
        rc = TSR_EARG;
    }
    double *val = rc == TSR_OK ? (double *)malloc((size_t)m * sizeof(double)) : NULL;
    double *out = val ? (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){m}) : NULL;
    if (rc == TSR_OK && !out) rc = TSR_ENOMEM;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i < m; i++) val[i] = R[i * 4 + col];
        hier_subtree_max(Z, m, m + 1, val, out);
    }
    free(val); fn_free_doubles(R, Lr); fn_free_doubles(Z, Lz);
    return rc;
}

static int r_maxinconsts(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || !is_Z(&args[0]) || !is_Z(&args[1])) { fn_set_error("maxinconsts: Z and R must be (m, 4)"); return TSR_EARG; }
    return hier_max_rstat(args, 3, res);
}

static int r_maxRstat(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 3 || !is_Z(&args[0]) || !is_Z(&args[1])) { fn_set_error("maxRstat: Z and R must be (m, 4)"); return TSR_EARG; }
    if (args[2].kind != 1 || !(args[2].flags & 1)) { fn_set_error("The third argument must be an integer."); return TSR_ETYPE; }
    const int64_t col = args[2].ival;
    if (col < 0 || col > 3) { fn_set_error("i must be an integer between 0 and 3 inclusive."); return TSR_EARG; }
    return hier_max_rstat(args, col, res);
}

/* leaves_list(Z): the leaf ids in the order a left-first pre-order dendrogram traversal visits them. */
static int r_leaves_list(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("leaves_list: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; int zrc; double *Z = hier_Z(&args[0], &L, "'Z' ", &zrc);
    if (!Z) return zrc;
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
    int64_t m = args[0].arr.shape[0], L; int zrc; double *Z = hier_Z(&args[0], &L, "'Z' ", &zrc);
    if (!Z) return zrc;
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
    int64_t m = args[0].arr.shape[0], L; int zrc; double *Z = hier_Z(&args[0], &L, "'Z' ", &zrc);
    if (!Z) return zrc;
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
    /* SciPy's format check (min of the ids != 1 and max != 2m), then each 1-based id must name an observation or an
       earlier merge: the cluster sizes below are read through them (SciPy's C code trusts them) */
    if (m > 0) {
        double mn = Z[0], mx = Z[0];
        for (int64_t i = 0; i < m; i++) for (int c = 0; c < 2; c++) { const double v = Z[i * 3 + c]; if (v < mn) mn = v; if (v > mx) mx = v; }
        int bad = mn != 1.0 && mx != (double)(2 * m);
        for (int64_t i = 0; i < m && !bad; i++) for (int c = 0; c < 2; c++) {
            const double v = Z[i * 3 + c];
            if (!(v >= 1.0 && v < (double)(n_obs + i) + 1.0)) bad = 1;
        }
        if (bad) { fn_free_doubles(Z, L); fn_set_error("The format of the indices is not 1..N"); return TSR_EARG; }
    }
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

/* ------------------------------------------------------------------ flat clusters (fcluster family) */
/* SciPy's cluster_monocrit: fill T[0..n-1] with 1-based flat-cluster labels. A flat cluster forms at the
   highest node whose criterion MC[node] <= cutoff; labels are numbered in the left-first post-order the
   traversal finalises leaves, matching scipy.cluster.hierarchy exactly. Z is a valid linkage (hier_Z), so the
   traversal stack (one entry per tree level) stays within 2n. */
static int cluster_monocrit(const double *Z, const double *MC, int64_t *T, double cutoff, int64_t n)
{
    int64_t *curr = (int64_t *)malloc((size_t)(2 * n) * sizeof(int64_t));
    unsigned char *vis = (unsigned char *)calloc((size_t)(2 * n), 1);
    if (!curr || !vis) { free(curr); free(vis); return TSR_ENOMEM; }
    int64_t k = 0, n_cluster = 0, cluster_leader = -1;
    curr[0] = 2 * n - 2;
    while (k >= 0) {
        int64_t root = curr[k];
        int64_t ilc = (int64_t)Z[(root - n) * 4 + 0], irc = (int64_t)Z[(root - n) * 4 + 1];
        if (cluster_leader == -1 && MC[root - n] <= cutoff) { cluster_leader = root; n_cluster++; }
        if (ilc >= n && !vis[ilc]) { vis[ilc] = 1; curr[++k] = ilc; continue; }
        if (irc >= n && !vis[irc]) { vis[irc] = 1; curr[++k] = irc; continue; }
        if (ilc < n) { if (cluster_leader == -1) n_cluster++; T[ilc] = n_cluster; }
        if (irc < n) { if (cluster_leader == -1) n_cluster++; T[irc] = n_cluster; }
        if (cluster_leader == root) cluster_leader = -1;
        k--;
    }
    free(curr); free(vis);
    return TSR_OK;
}

/* number of flat clusters a cutoff on the monotonic criterion MC would produce. */
static int64_t count_monocrit(const double *Z, const double *MC, double thresh, int64_t n,
                              int64_t *curr, unsigned char *vis)
{
    memset(vis, 0, (size_t)(2 * n));
    int64_t k = 0, nc = 0;
    curr[0] = 2 * n - 2;
    while (k >= 0) {
        int64_t root = curr[k];
        int64_t ilc = (int64_t)Z[(root - n) * 4 + 0], irc = (int64_t)Z[(root - n) * 4 + 1];
        if (MC[root - n] <= thresh) { nc++; k--; continue; }                 /* whole subtree is one cluster */
        if (ilc >= n && !vis[ilc]) { vis[ilc] = 1; curr[++k] = ilc; continue; }
        if (irc >= n && !vis[irc]) { vis[irc] = 1; curr[++k] = irc; continue; }
        if (ilc < n) nc++;
        if (irc < n) nc++;
        k--;
    }
    return nc;
}

/* maxclust / maxclust_monocrit: SciPy's index-bracketing binary search over the criterion MC (indices
   0..n-2), choosing cutoff = MC[upper] where upper is the bracket's high side. Reproduces scipy exactly
   even when MC is not monotonic (as maxinconsts can be). */
static int cluster_maxclust_monocrit(const double *Z, const double *MC, int64_t *T, int64_t n, int64_t max_nc)
{
    int64_t *curr = (int64_t *)malloc((size_t)(2 * n) * sizeof(int64_t));
    unsigned char *vis = (unsigned char *)malloc((size_t)(2 * n));
    if (!curr || !vis) { free(curr); free(vis); return TSR_ENOMEM; }
    int64_t lo = 0, hi = n - 1;                                              /* SciPy brackets with hi = n-1 */
    while (hi - lo > 1) {
        int64_t mid = (lo + hi) / 2;                                         /* mid < hi <= n-1, so MC[mid] is in range */
        if (count_monocrit(Z, MC, MC[mid], n, curr, vis) > max_nc) lo = mid; else hi = mid;
    }
    double cutoff = MC[hi < n - 1 ? hi : n - 2];                             /* clamp the degenerate hi == n-1 case */
    free(curr); free(vis);
    return cluster_monocrit(Z, MC, T, cutoff, n);
}

/* run fcluster into T (n_obs = m+1 entries, zeroed first as SciPy's np.zeros) given a valid linkage Z (m x 4).
   crit 0 uses R when given (validated, m rows) or inconsistent(Z, depth); crits 2 and 5 read monocrit (m values). */
static int fcluster_core(const double *Z, int64_t m, int64_t *T, double t, int64_t tmax, int crit, int64_t depth,
                         const double *R, const double *monocrit)
{
    int64_t n_obs = m + 1;
    memset(T, 0, (size_t)n_obs * sizeof(int64_t));
    double *MC = (double *)malloc((size_t)m * sizeof(double));
    double *val = (double *)malloc((size_t)m * sizeof(double));
    if (!MC || !val) { free(MC); free(val); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (crit == 0) {                              /* inconsistent: MC = max over subtree of R[:,3] */
        double *Rw = R ? NULL : (double *)malloc((size_t)(m * 4) * sizeof(double));
        if (!R && !Rw) rc = TSR_ENOMEM;
        else if (!R) rc = fill_inconsistent(Z, m, n_obs, depth, Rw);
        if (rc == TSR_OK) {
            const double *Ru = R ? R : Rw;
            for (int64_t i = 0; i < m; i++) val[i] = Ru[i * 4 + 3];
            hier_subtree_max(Z, m, n_obs, val, MC);
        }
        free(Rw);
    } else if (crit == 1 || crit == 4) {          /* distance | maxclust: MC = maxdists */
        for (int64_t i = 0; i < m; i++) val[i] = Z[i * 4 + 2];
        hier_subtree_max(Z, m, n_obs, val, MC);
    } else {                                      /* monocrit | maxclust_monocrit: user-supplied vector */
        for (int64_t i = 0; i < m; i++) MC[i] = monocrit[i];
    }
    if (rc == TSR_OK) {
        if (crit == 4 || crit == 5) rc = cluster_maxclust_monocrit(Z, MC, T, n_obs, tmax);
        else rc = cluster_monocrit(Z, MC, T, t, n_obs);
    }
    free(MC); free(val);
    return rc;
}

/* crit codes: 0 inconsistent, 1 distance, 2 monocrit, 4 maxclust, 5 maxclust_monocrit. */
static int fcluster_crit(const char *s)
{
    if (!strcmp(s, "inconsistent")) return 0;
    if (!strcmp(s, "distance")) return 1;
    if (!strcmp(s, "monocrit")) return 2;
    if (!strcmp(s, "maxclust")) return 4;
    if (!strcmp(s, "maxclust_monocrit")) return 5;
    return -1;
}

/* fcluster's threshold t: a float for the cutoff criteria; maxclust passes it to a C int (an int is required) and
   maxclust_monocrit takes int(t) (truncating, refusing NaN and the infinities) */
static int fcluster_t(const tsr_arg *a, int crit, double *t, int64_t *tmax)
{
    if (a->kind != 1 && a->kind != 4) { fn_set_error("fcluster: t must be a number"); return TSR_ETYPE; }
    *t = (a->kind == 1 && (a->flags & 1)) ? (double)a->ival : a->num;
    *tmax = 0;
    if (crit == 4) return fn_arg_i64(a, 1, INT32_MIN, INT32_MAX, "t", tmax);
    if (crit == 5) return fn_arg_i64(a, 0, INT32_MIN, INT32_MAX, "t", tmax);
    return TSR_OK;
}

static int r_fcluster(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || !is_Z(&args[0])) { fn_set_error("fcluster: Z must be (m, 4) and t is required"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; int rc;
    double *Z = hier_Z(&args[0], &L, "'Z' ", &rc);       /* SciPy validates Z first */
    if (!Z) return rc;
    int crit = 0;
    if (nargs > 2 && args[2].kind == 2) {
        crit = fcluster_crit(args[2].str);
        if (crit < 0) { fn_set_error("Invalid cluster formation criterion: %s", args[2].str); fn_free_doubles(Z, L); return TSR_EARG; }
    }
    double t; int64_t tmax, depth = 2;
    double *Rb = NULL, *mcbuf = NULL; int64_t Lr = 0, Lmc = 0;
    rc = fcluster_t(&args[1], crit, &t, &tmax);
    const int haveR = crit == 0 && nargs > 4 && args[4].kind == 3;
    if (rc == TSR_OK && crit == 0 && !haveR && nargs > 3) rc = hier_depth(&args[3], &depth);
    if (rc == TSR_OK && haveR) {                  /* a given inconsistency matrix (SciPy's _is_valid_im), one row per merge */
        if (!is_Z(&args[4])) { fn_set_error("fcluster: R must be (m, 4)"); rc = TSR_EARG; }
        else if (!(Rb = fn_arg_doubles(&args[4], &Lr))) rc = TSR_ENOMEM;
        else if ((rc = hier_check_R(Rb, args[4].arr.shape[0])) == TSR_OK && args[4].arr.shape[0] != m) {
            fn_set_error("The inconsistency matrix and linkage matrix each have a different number of rows.");
            rc = TSR_EARG;
        }
    }
    if (rc == TSR_OK && (crit == 2 || crit == 5)) {
        if (nargs < 6 || args[5].kind != 3) { fn_set_error("fcluster: monocrit vector required for this criterion"); rc = TSR_EARG; }
        else if (!(mcbuf = fn_arg_doubles(&args[5], &Lmc))) rc = TSR_ENOMEM;
        else if (Lmc != m) { fn_set_error("fcluster: monocrit must have Z.shape[0] (%lld) elements", (long long)m); rc = TSR_EARG; }
    }
    int64_t *T = rc == TSR_OK ? (int64_t *)fn_result_array(&res[0], TSR_I64, 1, (int64_t[]){m + 1}) : NULL;
    if (rc == TSR_OK) rc = T ? fcluster_core(Z, m, T, t, tmax, crit, depth, Rb, mcbuf) : TSR_ENOMEM;
    fn_free_doubles(Z, L);
    if (Rb) fn_free_doubles(Rb, Lr);
    if (mcbuf) fn_free_doubles(mcbuf, Lmc);
    return rc;
}

/* fclusterdata(X, t, criterion='inconsistent', metric='euclidean', depth=2, method='single'):
   pdist(X, metric) -> linkage(method) -> inconsistent(Z, depth) -> fcluster. Euclidean metric only (SciPy's default).
   SciPy always forms R, so depth is checked for every criterion; the monocrit criteria have no monocrit here. */
static int r_fclusterdata(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("The observation matrix X must be an n by m array."); return TSR_ETYPE; }
    int crit = 0;
    if (nargs > 2 && args[2].kind == 2) { crit = fcluster_crit(args[2].str); if (crit < 0) { fn_set_error("Invalid cluster formation criterion: %s", args[2].str); return TSR_EARG; } }
    if (crit == 2 || crit == 5) { fn_set_error("fclusterdata: criterion '%s' needs a monocrit vector, which fclusterdata does not take", args[2].str); return TSR_EARG; }
    if (nargs > 3 && args[3].kind == 2 && strcmp(args[3].str, "euclidean")) { fn_set_error("fclusterdata: only metric='euclidean' is supported"); return TSR_EARG; }
    double t; int64_t tmax, depth = 2;
    int rc = fcluster_t(&args[1], crit, &t, &tmax);
    if (rc == TSR_OK && nargs > 4) rc = hier_depth(&args[4], &depth);
    if (rc != TSR_OK) return rc;
    int method = 0;
    if (nargs > 5 && args[5].kind == 2) { method = hier_method_code(args[5].str); if (method < 0) { fn_set_error("Invalid method: %s", args[5].str); return TSR_EARG; } }
    int64_t n = args[0].arr.shape[0], dim = args[0].arr.shape[1], lo;
    if (n < 2) { fn_set_error("The number of observations cannot be determined on an empty distance matrix."); return TSR_EARG; }
    double *X = fn_arg_doubles(&args[0], &lo);
    if (!X) return TSR_ENOMEM;
    const int sq = method >= 4;
    const int64_t wbytes = tsr_bytes_n(tsr_count_mul(n, n), sizeof(double), 0);   /* counted n*n workspace */
    double *W = (double *)tsr_alloc(wbytes);
    double *Z = (double *)tsr_scratch_n(tsr_count_mul(n - 1, 4), sizeof(double));
    if (!W || !Z) { if (W) tsr_free(W, wbytes); free(Z); fn_free_doubles(X, lo); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n && rc == TSR_OK; i++) for (int64_t j = i + 1; j < n; j++) {
        double s = 0.0; for (int64_t c = 0; c < dim; c++) { double df = X[i * dim + c] - X[j * dim + c]; s += df * df; }
        if (!isfinite(s)) { fn_set_error("The condensed distance matrix must contain only finite values."); rc = TSR_EARG; break; }
        double d = sq ? s : sqrt(s); W[i * n + j] = W[j * n + i] = d;
    }
    for (int64_t i = 0; i < n; i++) W[i * n + i] = INFINITY;
    if (rc == TSR_OK) rc = hier_core(W, n, method, Z);
    fn_free_doubles(X, lo); tsr_free(W, wbytes);
    int64_t *T = rc == TSR_OK ? (int64_t *)fn_result_array(&res[0], TSR_I64, 1, (int64_t[]){n}) : NULL;
    if (rc == TSR_OK) rc = T ? fcluster_core(Z, n - 1, T, t, tmax, crit, depth, NULL, NULL) : TSR_ENOMEM;
    free(Z);
    return rc;
}

static int cmp_i64(const void *a, const void *b)
{
    const int64_t x = *(const int64_t *)a, y = *(const int64_t *)b;
    return (x > y) - (x < y);
}

/* leaders(Z, T): for each flat cluster the leader node -- the root of its maximal single-label subtree.
   Returns parallel arrays L (leader node ids) and M (their cluster labels), in SciPy's discovery order. T holds
   int32 labels (any value, -1 included); a node whose leaves carry more than one label is marked MIXED. */
#define HIER_MIXED INT64_MIN
static int r_leaders(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || !is_Z(&args[0]) || args[1].kind != 3) { fn_set_error("leaders: Z (m, 4) and T are required"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], Lz, Lt; int rc;
    double *Z = hier_Z(&args[0], &Lz, "'Z' ", &rc);
    if (!Z) return rc;
    if (args[1].arr.ndim != 1) { fn_free_doubles(Z, Lz); fn_set_error("T must be a 1-D array of dtype int32."); return TSR_ETYPE; }
    double *Td = fn_arg_doubles(&args[1], &Lt); if (!Td) { fn_free_doubles(Z, Lz); return TSR_ENOMEM; }
    int64_t n = m + 1;
    if (Lt != n) { fn_free_doubles(Z, Lz); fn_free_doubles(Td, Lt); fn_set_error("Mismatch: len(T)!=Z.shape[0] + 1."); return TSR_EARG; }
    for (int64_t i = 0; i < n; i++)                           /* SciPy requires dtype int32 */
        if (!(Td[i] == floor(Td[i])) || Td[i] < -2147483648.0 || Td[i] > 2147483647.0) {
            fn_free_doubles(Z, Lz); fn_free_doubles(Td, Lt); fn_set_error("T must be a 1-D array of dtype int32."); return TSR_ETYPE;
        }
    int64_t *cid = (int64_t *)malloc((size_t)(2 * n - 1) * sizeof(int64_t));
    int64_t *srt = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    if (!cid || !srt) { free(cid); free(srt); fn_free_doubles(Z, Lz); fn_free_doubles(Td, Lt); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) srt[i] = cid[i] = (int64_t)Td[i];
    for (int64_t i = 0; i < m; i++) cid[n + i] = HIER_MIXED;
    qsort(srt, (size_t)n, sizeof(int64_t), cmp_i64);          /* number of distinct labels (SciPy's nunique) */
    int64_t nc = n > 0 ? 1 : 0;
    for (int64_t i = 1; i < n; i++) if (srt[i] != srt[i - 1]) nc++;
    free(srt);
    int64_t *Lb = (int64_t *)calloc((size_t)(nc > 0 ? nc : 1), sizeof(int64_t));   /* zeros, as SciPy's np.zeros */
    int64_t *Mb = (int64_t *)calloc((size_t)(nc > 0 ? nc : 1), sizeof(int64_t));
    if (!Lb || !Mb) { free(Lb); free(Mb); free(cid); fn_free_doubles(Z, Lz); fn_free_doubles(Td, Lt); return TSR_ENOMEM; }
    int64_t k = 0, bad = -1;
    for (int64_t i = 0; i < m && bad < 0; i++) {
        int64_t lc = (int64_t)Z[i * 4 + 0], rc2 = (int64_t)Z[i * 4 + 1], lf = cid[lc], rf = cid[rc2];
        if (lf == rf) { cid[n + i] = lf; continue; }
        if (lf != HIER_MIXED) { if (k >= nc) { bad = lc; break; } Lb[k] = lc; Mb[k] = lf; k++; }
        if (rf != HIER_MIXED) { if (k >= nc) { bad = rc2; break; } Lb[k] = rc2; Mb[k] = rf; k++; }
        cid[n + i] = HIER_MIXED;
    }
    if (bad < 0 && cid[2 * n - 2] != HIER_MIXED) { if (k >= nc) bad = 2 * n - 2; else { Lb[k] = 2 * n - 2; Mb[k] = cid[2 * n - 2]; k++; } }
    free(cid); fn_free_doubles(Z, Lz); fn_free_doubles(Td, Lt);
    if (bad >= 0) {
        free(Lb); free(Mb);
        fn_set_error("T is not a valid assignment vector. Error found when examining linkage node %lld (< 2n-1).", (long long)bad);
        return TSR_EARG;
    }
    int64_t sh[1] = {nc};
    int64_t *Lo = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, sh);
    int64_t *Mo = (int64_t *)fn_result_array(&res[1], TSR_I64, 1, sh);
    if (!Lo || !Mo) { free(Lb); free(Mb); return TSR_ENOMEM; }
    memcpy(Lo, Lb, (size_t)nc * sizeof(int64_t));
    memcpy(Mo, Mb, (size_t)nc * sizeof(int64_t));
    free(Lb); free(Mb);
    return TSR_OK;
}

/* np.searchsorted(np.arange(nobs), v) (side 'left'): how many of 0..nobs-1 are below v; NaN sorts last */
static int64_t hier_searchsorted_arange(double v, int64_t nobs)
{
    if (v != v || v > (double)(nobs - 1)) return nobs;
    if (v <= 0.0) return 0;
    return (int64_t)ceil(v);
}

/* cut_tree(Z, n_clusters=None, height=None): group membership at each requested cut, SciPy's relabelling
   (merge the two members into the lower group id, compact ids above the higher one). Output (nobs, ncols). */
static int r_cut_tree(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (!is_Z(&args[0])) { fn_set_error("cut_tree: Z must be (m, 4)"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], L; int zrc; double *Z = hier_Z(&args[0], &L, "'Z' ", &zrc);
    if (!Z) return zrc;
    int64_t n = m + 1;                                                       /* nobs */
    /* requested columns: step index at which to read the grouping (0..m). */
    int rc = TSR_OK;
    const tsr_arg *ncl = nargs > 1 ? &args[1] : NULL, *hgt = nargs > 2 ? &args[2] : NULL;
    int64_t ncols; int64_t *cols = NULL;                                     /* step index per output column */
    if (ncl && ncl->kind != 0 && hgt && hgt->kind != 0) { fn_free_doubles(Z, L); fn_set_error("At least one of either height or n_clusters must be None"); return TSR_EARG; }
    if (ncl && ncl->kind == 3) {                                            /* n_clusters array */
        int64_t Ln; double *v = fn_arg_doubles(ncl, &Ln); if (!v) { fn_free_doubles(Z, L); return TSR_ENOMEM; }
        ncols = Ln; cols = (int64_t *)malloc((size_t)(ncols > 0 ? ncols : 1) * sizeof(int64_t));
        if (cols) for (int64_t j = 0; j < ncols; j++) cols[j] = n - hier_searchsorted_arange(v[j], n);
        fn_free_doubles(v, Ln);
    } else if (ncl && ncl->kind == 1) {                                     /* n_clusters scalar */
        ncols = 1; cols = (int64_t *)malloc(sizeof(int64_t));
        if (cols) cols[0] = n - hier_searchsorted_arange((ncl->flags & 1) ? (double)ncl->ival : ncl->num, n);
    } else if (hgt && hgt->kind == 3) {                                     /* height array: searchsorted on merge dists */
        int64_t Lh; double *h = fn_arg_doubles(hgt, &Lh); if (!h) { fn_free_doubles(Z, L); return TSR_ENOMEM; }
        ncols = Lh; cols = (int64_t *)malloc((size_t)(ncols > 0 ? ncols : 1) * sizeof(int64_t));
        if (cols) for (int64_t j = 0; j < ncols; j++) {
            int64_t s = 0;
            if (h[j] != h[j]) s = m;                                        /* NaN sorts after every height */
            else while (s < m && Z[s * 4 + 2] < h[j]) s++;
            cols[j] = s;
        }
        fn_free_doubles(h, Lh);
    } else {                                                                /* full cut tree: columns 0..n-1 */
        ncols = n; cols = (int64_t *)malloc((size_t)ncols * sizeof(int64_t));
        if (cols) for (int64_t j = 0; j < ncols; j++) cols[j] = j;
    }
    int64_t *last = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *cur = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t sh[2] = {n, ncols};
    int64_t *G = (int64_t *)fn_result_array(&res[0], TSR_I64, 2, sh);        /* (nobs, ncols) */
    /* member lists per cluster id, as in cophenet */
    int64_t total = n + m;
    int64_t **mem = (int64_t **)calloc((size_t)total, sizeof(int64_t *));
    int64_t *msz = (int64_t *)calloc((size_t)total, sizeof(int64_t));
    if (!cols || !last || !cur || !G || !mem || !msz) { rc = TSR_ENOMEM; goto done; }
    memset(G, 0, (size_t)(n * ncols) * sizeof(int64_t));                   /* SciPy's np.zeros: columns no step reaches */
    for (int64_t i = 0; i < n; i++) { last[i] = i; mem[i] = (int64_t *)malloc(sizeof(int64_t)); if (mem[i]) { mem[i][0] = i; msz[i] = 1; } else rc = TSR_ENOMEM; }
    /* step 0 grouping = identity */
    for (int64_t j = 0; j < ncols; j++) if (cols[j] == 0) for (int64_t r = 0; r < n; r++) G[r * ncols + j] = last[r];
    for (int64_t i = 0; i < m && rc == TSR_OK; i++) {
        int64_t lc = (int64_t)Z[i * 4 + 0], rcid = (int64_t)Z[i * 4 + 1];
        /* idx = members of node i (left ++ right); build its member list too */
        int64_t nl = msz[lc], nr = msz[rcid], id = n + i;
        mem[id] = (int64_t *)malloc((size_t)(nl + nr) * sizeof(int64_t));
        if (!mem[id]) { rc = TSR_ENOMEM; break; }
        memcpy(mem[id], mem[lc], (size_t)nl * sizeof(int64_t));
        memcpy(mem[id] + nl, mem[rcid], (size_t)nr * sizeof(int64_t));
        msz[id] = nl + nr;
        /* mn = min last[idx], mxg = max last[idx] */
        int64_t mn = last[mem[id][0]], mxg = mn;
        for (int64_t x = 0; x < nl + nr; x++) { int64_t g = last[mem[id][x]]; if (g < mn) mn = g; if (g > mxg) mxg = g; }
        for (int64_t r = 0; r < n; r++) cur[r] = last[r];
        for (int64_t x = 0; x < nl + nr; x++) cur[mem[id][x]] = mn;
        for (int64_t r = 0; r < n; r++) if (cur[r] > mxg) cur[r] -= 1;
        for (int64_t j = 0; j < ncols; j++) if (cols[j] == i + 1) for (int64_t r = 0; r < n; r++) G[r * ncols + j] = cur[r];
        for (int64_t r = 0; r < n; r++) last[r] = cur[r];
    }
done:
    for (int64_t i = 0; i < total; i++) if (mem && mem[i]) free(mem[i]);
    free(mem); free(msz); free(cols); free(last); free(cur); fn_free_doubles(Z, L);
    return rc;
}

/* total order on doubles for label comparison (NaN equal to NaN, after every number) */
static int cmp_lbl(double x, double y)
{
    if (x != x) return (y != y) ? 0 : 1;
    if (y != y) return -1;
    return (x > y) - (x < y);
}
typedef struct { double a, b; } iso_pair;
static int cmp_iso_pair(const void *p, const void *q)
{
    const iso_pair *x = (const iso_pair *)p, *y = (const iso_pair *)q;
    const int c = cmp_lbl(x->a, y->a);
    return c ? c : cmp_lbl(x->b, y->b);
}

/* is_isomorphic(T1, T2): do two flat-cluster labellings describe the same partition (a bijection of labels)? SciPy's
   dictionary walk is equivalent to: #distinct (T1[i], T2[i]) pairs == #distinct T1 labels == #distinct T2 labels.
   Labels are compared as values, never used as indices, so any label (negative, fractional, huge) is safe. */
static int r_is_isomorphic(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || args[0].kind != 3 || args[1].kind != 3) { fn_set_error("is_isomorphic: two label arrays are required"); return TSR_EARG; }
    if (args[0].arr.ndim != 1) { fn_set_error("T1 must be one-dimensional."); return TSR_EARG; }
    if (args[1].arr.ndim != 1) { fn_set_error("T2 must be one-dimensional."); return TSR_EARG; }
    if (args[0].arr.shape[0] != args[1].arr.shape[0]) { fn_set_error("T1 and T2 must have the same number of elements."); return TSR_EARG; }
    int64_t L1, L2; double *A = fn_arg_doubles(&args[0], &L1); if (!A) return TSR_ENOMEM;
    double *B = fn_arg_doubles(&args[1], &L2); if (!B) { fn_free_doubles(A, L1); return TSR_ENOMEM; }
    const int64_t n = L1;
    iso_pair *p = (iso_pair *)malloc((size_t)(n > 0 ? n : 1) * sizeof(iso_pair));
    if (!p) { fn_free_doubles(A, L1); fn_free_doubles(B, L2); return TSR_ENOMEM; }
    int64_t pairs = 0, da = 0, db = 0;
    if (n > 0) {
        for (int64_t i = 0; i < n; i++) { p[i].a = A[i]; p[i].b = B[i]; }
        qsort(p, (size_t)n, sizeof(iso_pair), cmp_iso_pair);        /* by (T1, T2): distinct pairs and T1 labels */
        pairs = da = 1;
        for (int64_t i = 1; i < n; i++) {
            if (cmp_lbl(p[i].a, p[i - 1].a)) { da++; pairs++; }
            else if (cmp_lbl(p[i].b, p[i - 1].b)) pairs++;
        }
        for (int64_t i = 0; i < n; i++) { p[i].a = B[i]; p[i].b = 0.0; }
        qsort(p, (size_t)n, sizeof(iso_pair), cmp_iso_pair);        /* by T2: distinct T2 labels */
        db = 1;
        for (int64_t i = 1; i < n; i++) if (cmp_lbl(p[i].a, p[i - 1].a)) db++;
    }
    free(p); fn_free_doubles(A, L1); fn_free_doubles(B, L2);
    hier_bool(&res[0], pairs == da && pairs == db);
    return TSR_OK;
}

/* ------------------------------------------------------------------ optimal_leaf_ordering */

/* dual-swap and the exact scikit-learn simultaneous quicksort scipy uses, so the (u, w) / (m, k)
   tie-breaking matches scipy's _optimal_leaf_ordering bit for bit. Operates on d[0..size), ix[0..size). */
static void olo_dual_swap(float *d, int *ix, int i, int j)
{
    float dt = d[i]; d[i] = d[j]; d[j] = dt;
    int it = ix[i]; ix[i] = ix[j]; ix[j] = it;
}
static void olo_simsort(float *d, int *ix, int size)
{
    if (size <= 1) return;
    if (size == 2) { if (d[0] > d[1]) olo_dual_swap(d, ix, 0, 1); return; }
    if (size == 3) {
        if (d[0] > d[1]) olo_dual_swap(d, ix, 0, 1);
        if (d[1] > d[2]) { olo_dual_swap(d, ix, 1, 2); if (d[0] > d[1]) olo_dual_swap(d, ix, 0, 1); }
        return;
    }
    int pivot_idx = size / 2, i, store_idx;
    if (d[0] > d[size - 1]) olo_dual_swap(d, ix, 0, size - 1);
    if (d[size - 1] > d[pivot_idx]) {
        olo_dual_swap(d, ix, size - 1, pivot_idx);
        if (d[0] > d[size - 1]) olo_dual_swap(d, ix, 0, size - 1);
    }
    float pivot_val = d[size - 1];
    store_idx = 0;
    for (i = 0; i < size - 1; i++) if (d[i] < pivot_val) { olo_dual_swap(d, ix, i, store_idx); store_idx++; }
    olo_dual_swap(d, ix, store_idx, size - 1);
    pivot_idx = store_idx;
    if (pivot_idx > 1) olo_simsort(d, ix, pivot_idx);
    if (pivot_idx + 2 < size) olo_simsort(d + pivot_idx + 1, ix + pivot_idx + 1, size - pivot_idx - 1);
}

#define OLO_BIG 1073741824.0f

/* optimal_leaf_ordering(Z, y, metric='euclidean'): reorder the linkage Z to minimise the sum of distances
   between adjacent leaves (Bar-Joseph). y is condensed distances (1-D), or a 2-D observation matrix
   (euclidean pdist is formed). Returns a copy of Z with children of some rows swapped; heights/sizes kept. */
static int r_optimal_leaf_ordering(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 2 || !is_Z(&args[0])) { fn_set_error("optimal_leaf_ordering: Z must be (m, 4)"); return TSR_EARG; }
    if (nargs > 2 && args[2].kind == 2 && strcmp(args[2].str, "euclidean"))
        { fn_set_error("optimal_leaf_ordering: only metric='euclidean' is supported"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], n = m + 1, zl; int zrc; double *Z = hier_Z(&args[0], &zl, "'Z' ", &zrc);
    if (!Z) return zrc;
    /* the Bar-Joseph kernel below indexes n x n tables with C int arithmetic (u * n + w), as SciPy's does */
    if (n > 46340) { fn_free_doubles(Z, zl); fn_set_error("optimal_leaf_ordering: more than 46340 observations"); return TSR_EARG; }

    /* full n-by-n distance matrix D (double) from y */
    const int64_t dbytes = tsr_bytes_n(tsr_count_mul(n, n), sizeof(double), 0);   /* counted n*n distances */
    double *D = (double *)tsr_alloc(dbytes);
    if (!D) { fn_free_doubles(Z, zl); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (args[1].kind == 3 && args[1].arr.ndim == 2) {
        int64_t nn = args[1].arr.shape[0], dim = args[1].arr.shape[1], yl; double *Y = fn_arg_doubles(&args[1], &yl);
        if (!Y) { tsr_free(D, dbytes); fn_free_doubles(Z, zl); return TSR_ENOMEM; }
        if (nn != n) { fn_set_error("optimal_leaf_ordering: observation count does not match Z"); tsr_free(D, dbytes); fn_free_doubles(Y, yl); fn_free_doubles(Z, zl); return TSR_EARG; }
        for (int64_t i = 0; i < n; i++) { D[i * n + i] = 0.0;
            for (int64_t j = i + 1; j < n; j++) { double s = 0.0;
                for (int64_t c = 0; c < dim; c++) { double df = Y[i * dim + c] - Y[j * dim + c]; s += df * df; }
                double d = sqrt(s); D[i * n + j] = D[j * n + i] = d; } }
        fn_free_doubles(Y, yl);
        for (int64_t i = 0; i < n * n && rc == TSR_OK; i++)
            if (!isfinite(D[i])) { fn_set_error("The condensed distance matrix must contain only finite values."); rc = TSR_EARG; }
        if (rc != TSR_OK) { tsr_free(D, dbytes); fn_free_doubles(Z, zl); return rc; }
    } else if (args[1].kind == 3 && args[1].arr.ndim == 1) {
        int64_t yl; double *Y = fn_arg_doubles(&args[1], &yl);
        if (!Y) { tsr_free(D, dbytes); fn_free_doubles(Z, zl); return TSR_ENOMEM; }
        int64_t nc = n * (n - 1) / 2;
        if (args[1].arr.shape[0] != nc) { fn_set_error("optimal_leaf_ordering: condensed distance length does not match Z"); tsr_free(D, dbytes); fn_free_doubles(Y, yl); fn_free_doubles(Z, zl); return TSR_EARG; }
        int64_t p = 0;
        for (int64_t i = 0; i < n; i++) { D[i * n + i] = 0.0;
            for (int64_t j = i + 1; j < n; j++) { double d = Y[p++]; D[i * n + j] = D[j * n + i] = d; } }
        fn_free_doubles(Y, yl);
        for (int64_t i = 0; i < n * n && rc == TSR_OK; i++)
            if (!isfinite(D[i])) { fn_set_error("The condensed distance matrix must contain only finite values."); rc = TSR_EARG; }
        if (rc != TSR_OK) { tsr_free(D, dbytes); fn_free_doubles(Z, zl); return rc; }
    } else { fn_set_error("optimal_leaf_ordering: y must be a condensed distance vector or an observation matrix"); tsr_free(D, dbytes); fn_free_doubles(Z, zl); return TSR_EARG; }

    /* sorted_leaves = leaves_list(Z) (left-first pre-order); map original leaf -> sorted position */
    int64_t *sl = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    int64_t *stack = (int64_t *)malloc((size_t)(2 * n) * sizeof(int64_t));
    int *o2s = (int *)malloc((size_t)n * sizeof(int));
    int *szl = NULL, *cr = NULL, *ss = NULL, *must_swap = NULL, *midx = NULL, *kidx = NULL;
    float *sD = NULL, *M = NULL, *mvals = NULL, *kvals = NULL;
    if (!sl || !stack || !o2s) { rc = TSR_ENOMEM; goto done; }
    { int64_t sp = 0, oi = 0; stack[sp++] = n + m - 1;
      while (sp > 0) { int64_t nd = stack[--sp];
        if (nd < n) sl[oi++] = nd;
        else { int64_t i = nd - n; stack[sp++] = (int64_t)Z[i * 4 + 1]; stack[sp++] = (int64_t)Z[i * 4 + 0]; } }
      for (int64_t s = 0; s < n; s++) o2s[sl[s]] = (int)s; }

    /* sorted_Z: children relabelled to sorted leaf positions; sorted_D permuted by sorted_leaves (float32) */
    szl = (int *)tsr_scratch_n(tsr_count_mul(m, 2), sizeof(int));                 /* [v_l, v_r] per row */
    sD = (float *)tsr_scratch_n(tsr_count_mul(n, n), sizeof(float));
    cr = (int *)tsr_scratch_n(tsr_count_mul(2 * n - 1, 2), sizeof(int));        /* cluster_ranges */
    if (!szl || !sD || !cr) { rc = TSR_ENOMEM; goto done; }
    for (int64_t i = 0; i < m; i++) {
        int64_t vl = (int64_t)Z[i * 4 + 0], vr = (int64_t)Z[i * 4 + 1];
        szl[i * 2 + 0] = (int)(vl < n ? o2s[vl] : vl);
        szl[i * 2 + 1] = (int)(vr < n ? o2s[vr] : vr);
    }
    for (int64_t a = 0; a < n; a++) for (int64_t b = 0; b < n; b++) sD[a * n + b] = (float)D[sl[a] * n + sl[b]];
    for (int64_t i = 0; i < n; i++) { cr[i * 2 + 0] = (int)i; cr[i * 2 + 1] = (int)i + 1; }
    for (int64_t li = 0; li < m; li++) { int64_t v = li + n; cr[v * 2 + 0] = cr[szl[li * 2 + 0] * 2 + 0]; cr[v * 2 + 1] = cr[szl[li * 2 + 1] * 2 + 1]; }

    /* identify_swaps: Bar-Joseph DP over the sorted tree, with M in float32 */
    M = (float *)tsr_scratch0_n(tsr_count_mul(n, n), sizeof(float));
    ss = (int *)tsr_scratch0_n(tsr_count_mul(tsr_count_mul(n, n), 2), sizeof(int));               /* swap_status */
    must_swap = (int *)calloc((size_t)m, sizeof(int));
    mvals = (float *)malloc((size_t)n * sizeof(float));
    midx = (int *)malloc((size_t)n * sizeof(int));
    kvals = (float *)malloc((size_t)n * sizeof(float));
    kidx = (int *)malloc((size_t)n * sizeof(int));
    if (!M || !ss || !must_swap || !mvals || !midx || !kvals || !kidx) { rc = TSR_ENOMEM; goto done; }

    for (int64_t i = 0; i < m; i++) {
        int vl = szl[i * 2 + 0], vr = szl[i * 2 + 1];
        int vlmin = cr[vl * 2 + 0], vlmax = cr[vl * 2 + 1], vrmin = cr[vr * 2 + 0], vrmax = cr[vr * 2 + 1];
        int u_cl[2], m_cl[2], w_cl[2], k_cl[2], tot_u, tot_w;
        if (vl < n) { tot_u = 1; u_cl[0] = vl; m_cl[0] = vl; }
        else { tot_u = 2; u_cl[0] = szl[(vl - n) * 2 + 0]; m_cl[0] = szl[(vl - n) * 2 + 1];
               u_cl[1] = szl[(vl - n) * 2 + 1]; m_cl[1] = szl[(vl - n) * 2 + 0]; }
        if (vr < n) { tot_w = 1; w_cl[0] = vr; k_cl[0] = vr; }
        else { tot_w = 2; w_cl[0] = szl[(vr - n) * 2 + 1]; w_cl[1] = szl[(vr - n) * 2 + 0];
               k_cl[0] = szl[(vr - n) * 2 + 0]; k_cl[1] = szl[(vr - n) * 2 + 1]; }
        for (int sL = 0; sL < tot_u; sL++) for (int sR = 0; sR < tot_w; sR++) {
            int umin = cr[u_cl[sL] * 2], umax = cr[u_cl[sL] * 2 + 1];
            int mmin = cr[m_cl[sL] * 2], mmax = cr[m_cl[sL] * 2 + 1];
            int wmin = cr[w_cl[sR] * 2], wmax = cr[w_cl[sR] * 2 + 1];
            int kmin = cr[k_cl[sR] * 2], kmax = cr[k_cl[sR] * 2 + 1];
            float min_km = OLO_BIG;
            for (int mm = mmin; mm < mmax; mm++) for (int kk = kmin; kk < kmax; kk++)
                if (sD[mm * n + kk] < min_km) min_km = sD[mm * n + kk];
            for (int u = umin; u < umax; u++) {
                int mc = mmax - mmin;
                for (int t = 0; t < mc; t++) { mvals[t] = M[(mmin + t) * n + u]; midx[t] = mmin + t; }
                olo_simsort(mvals, midx, mc);
                for (int w = wmin; w < wmax; w++) {
                    int kc = kmax - kmin;
                    for (int t = 0; t < kc; t++) { kvals[t] = M[(kmin + t) * n + w]; kidx[t] = kmin + t; }
                    olo_simsort(kvals, kidx, kc);
                    float cur = OLO_BIG;
                    for (int mi = 0; mi < mc; mi++) {
                        int mm = midx[mi];
                        if (M[u * n + mm] + M[w * n + kidx[0]] + min_km >= cur) break;
                        for (int ki = 0; ki < kc; ki++) {
                            int kk = kidx[ki];
                            if (M[u * n + mm] + M[w * n + kk] + min_km >= cur) break;
                            float cm = M[u * n + mm] + M[w * n + kk] + sD[mm * n + kk];
                            if (cm < cur) cur = cm;
                        }
                    }
                    M[u * n + w] = cur; M[w * n + u] = cur;
                    ss[(u * n + w) * 2 + 0] = sL; ss[(w * n + u) * 2 + 0] = sL;
                    ss[(u * n + w) * 2 + 1] = sR; ss[(w * n + u) * 2 + 1] = sR;
                }
            }
        }
        float cur = OLO_BIG; int bu = 0, bw = 0;
        for (int u = vlmin; u < vlmax; u++) for (int w = vrmin; w < vrmax; w++)
            if (M[u * n + w] < cur) { cur = M[u * n + w]; bu = u; bw = w; }
        if (vl >= n) must_swap[vl - n] = ss[(bu * n + bw) * 2 + 0];
        if (vr >= n) must_swap[vr - n] = ss[(bu * n + bw) * 2 + 1];
    }

    /* propagate swaps to descendants (XOR with every swapped ancestor), then emit the reordered Z.
       A row's final swap is must_swap[i] XOR the XOR of must_swap over all its ancestor rows; processing
       rows from the root downward lets each parent fold its own final swap into its internal children. */
    {
        char *fs = (char *)calloc((size_t)m, 1);                            /* final_swap per row */
        if (!fs) { rc = TSR_ENOMEM; goto done; }
        for (int64_t i = 0; i < m; i++) fs[i] = (char)must_swap[i];
        for (int64_t i = m - 1; i >= 0; i--) {
            if (!fs[i]) continue;
            int vl = szl[i * 2 + 0], vr = szl[i * 2 + 1];
            if (vl >= n) fs[vl - n] ^= 1;
            if (vr >= n) fs[vr - n] ^= 1;
        }
        int64_t sh[2] = {m, 4};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
        if (!out) { free(fs); rc = TSR_ENOMEM; goto done; }
        for (int64_t i = 0; i < m; i++) {
            double il = Z[i * 4 + 0], ir = Z[i * 4 + 1];
            out[i * 4 + 0] = fs[i] ? ir : il;
            out[i * 4 + 1] = fs[i] ? il : ir;
            out[i * 4 + 2] = Z[i * 4 + 2];
            out[i * 4 + 3] = Z[i * 4 + 3];
        }
        free(fs);
    }

done:
    free(M); free(ss); free(must_swap); free(mvals); free(midx); free(kvals); free(kidx);
    free(szl); free(sD); free(cr);
    free(sl); free(stack); free(o2s); tsr_free(D, dbytes); fn_free_doubles(Z, zl);
    return rc;
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
    ROUTINE("hierarchy.fcluster", 1, "Z, t, criterion='inconsistent', depth=2, R=None, monocrit=None", "T", r_fcluster, NULL, "Flat clusters from a linkage matrix (scipy.cluster.hierarchy.fcluster)."),
    ROUTINE("hierarchy.fclusterdata", 1, "X, t, criterion='inconsistent', metric='euclidean', depth=2, method='single'", "T", r_fclusterdata, NULL, "Flat clusters directly from observations (scipy.cluster.hierarchy.fclusterdata)."),
    ROUTINE("hierarchy.leaders", 2, "Z, T", "L, M", r_leaders, NULL, "Root (leader) nodes of the flat clusters in T (scipy.cluster.hierarchy.leaders)."),
    ROUTINE("hierarchy.cut_tree", 1, "Z, n_clusters=None, height=None", "groups", r_cut_tree, NULL, "Group membership at each requested cut of the dendrogram (scipy.cluster.hierarchy.cut_tree)."),
    ROUTINE("hierarchy.is_isomorphic", 2, "T1, T2", "iso", r_is_isomorphic, NULL, "Whether two flat-cluster labellings are isomorphic (scipy.cluster.hierarchy.is_isomorphic)."),
    ROUTINE("hierarchy.optimal_leaf_ordering", 2, "Z, y, metric='euclidean'", "Z", r_optimal_leaf_ordering, NULL, "Reorder a linkage to minimise distances between adjacent leaves (scipy.cluster.hierarchy.optimal_leaf_ordering)."),
};

const fn_table TSR_SCIPY_HIERARCHY_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
