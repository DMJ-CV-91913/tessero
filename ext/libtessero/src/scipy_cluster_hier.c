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

static const fn_def DEFS[] = {
    ROUTINE("hierarchy.linkage", 1, "y, method='single'", "Z", r_linkage, NULL, "Agglomerative hierarchical clustering linkage matrix (scipy.cluster.hierarchy.linkage)."),
    ROUTINE("hierarchy.single", 1, "y", "Z", r_single, NULL, "Single/nearest-point linkage (scipy.cluster.hierarchy.single)."),
    ROUTINE("hierarchy.complete", 1, "y", "Z", r_complete, NULL, "Complete/farthest-point linkage (scipy.cluster.hierarchy.complete)."),
    ROUTINE("hierarchy.average", 1, "y", "Z", r_average, NULL, "Average (UPGMA) linkage (scipy.cluster.hierarchy.average)."),
    ROUTINE("hierarchy.weighted", 1, "y", "Z", r_weighted, NULL, "Weighted (WPGMA) linkage (scipy.cluster.hierarchy.weighted)."),
    ROUTINE("hierarchy.ward", 1, "y", "Z", r_ward, NULL, "Ward variance-minimizing linkage (scipy.cluster.hierarchy.ward)."),
    ROUTINE("hierarchy.centroid", 1, "y", "Z", r_centroid, NULL, "Centroid (UPGMC) linkage (scipy.cluster.hierarchy.centroid)."),
    ROUTINE("hierarchy.median", 1, "y", "Z", r_median, NULL, "Median (WPGMC) linkage (scipy.cluster.hierarchy.median)."),
};

const fn_table TSR_SCIPY_HIERARCHY_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
