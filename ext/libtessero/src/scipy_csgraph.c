/* scipy.sparse.csgraph: connected_components on a dense adjacency matrix (real float64).
 *
 *   connected_components(csgraph, directed=True, connection='weak')  -> (n_components, labels)
 *
 * Weak connectivity (the SciPy default) treats the graph as undirected and labels components 0-based in node
 * order, exactly as SciPy does. connection='strong' (Tarjan SCC) is not implemented here. */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static int r_connected_components(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) {
        fn_set_error("connected_components: the graph must be a square 2-D adjacency matrix");
        return TSR_EARG;
    }
    if (nargs > 2 && args[2].kind == 2 && args[2].str && strcmp(args[2].str, "weak") != 0) {
        fn_set_error("connected_components: only connection='weak' is supported");
        return TSR_EARG;
    }
    const int64_t n = args[0].arr.shape[0];
    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    int32_t *lab = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, (int64_t[]){n});
    if (!lab) { fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) lab[i] = -1;
    int64_t *stack = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    if (!stack) { fn_free_doubles(A, tot); return TSR_ENOMEM; }
    int32_t comp = 0;
    for (int64_t s = 0; s < n; s++) {
        if (lab[s] >= 0) continue;
        int64_t sp = 0; stack[sp++] = s; lab[s] = comp;
        while (sp > 0) {
            const int64_t u = stack[--sp];
            for (int64_t v = 0; v < n; v++)                 /* undirected: edge if A[u][v] or A[v][u] is nonzero */
                if (lab[v] < 0 && (A[u * n + v] != 0.0 || A[v * n + u] != 0.0)) { lab[v] = comp; stack[sp++] = v; }
        }
        comp++;
    }
    free(stack);
    fn_free_doubles(A, tot);
    fn_result_int(&res[0], comp);
    return TSR_OK;
}

/* ---------------------------------------------------------------- shortest paths (dense, real float64) -----
 * Input is a dense square adjacency matrix: a zero entry means "no edge", any non-zero entry is an edge weight
 * (as SciPy's dense csgraph convention). The all-pairs shortest-path distance matrix is independent of the
 * method (Dijkstra / Bellman-Ford / Floyd-Warshall / Johnson) for any valid input, so every method routine
 * computes it the same way (Floyd-Warshall), matching SciPy's output: 0 on the diagonal, inf where no path
 * exists, negative-cycle detection as an error. indices=None (all sources) and return_predecessors=False. */

/* weight matrix from the adjacency (0 -> inf); undirected symmetrises by the smaller of the two directions. */
static double *csg_weights(const double *A, int64_t n, int directed, int unweighted)
{
    double *W = (double *)malloc(sizeof(double) * (size_t)(n * n));
    if (!W) return NULL;
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < n; j++) {
            const double a = A[i * n + j];
            W[i * n + j] = (a != 0.0) ? (unweighted ? 1.0 : a) : INFINITY;
        }
    if (!directed)
        for (int64_t i = 0; i < n; i++)
            for (int64_t j = i + 1; j < n; j++) {
                const double m = W[i * n + j] < W[j * n + i] ? W[i * n + j] : W[j * n + i];
                W[i * n + j] = m; W[j * n + i] = m;
            }
    for (int64_t i = 0; i < n; i++) W[i * n + i] = 0.0;
    return W;
}

/* parse a boolean-ish argument (default when omitted/None) */
static int csg_flag(const tsr_arg *args, int nargs, int idx, int dflt)
{
    if (idx >= nargs) return dflt;
    const tsr_arg *a = &args[idx];
    if (a->kind == 0) return dflt;
    if (a->kind == 1 || a->kind == 4) return a->num != 0;
    return dflt;
}

/* all-pairs shortest paths via Floyd-Warshall; emits an n-by-n float64 distance matrix. */
static int csg_shortest(const tsr_arg *args, int nargs, int directed_idx, int unweighted_idx, tsr_result *res)
{
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) {
        fn_set_error("shortest_path: the graph must be a square 2-D adjacency matrix"); return TSR_EARG;
    }
    const int directed = csg_flag(args, nargs, directed_idx, 1);
    const int unweighted = unweighted_idx >= 0 ? csg_flag(args, nargs, unweighted_idx, 0) : 0;
    const int64_t n = args[0].arr.shape[0];
    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    double *W = csg_weights(A, n, directed, unweighted);
    fn_free_doubles(A, tot);
    if (!W) return TSR_ENOMEM;
    for (int64_t k = 0; k < n; k++)
        for (int64_t i = 0; i < n; i++) {
            const double wik = W[i * n + k];
            if (isinf(wik)) continue;
            for (int64_t j = 0; j < n; j++) {
                const double wkj = W[k * n + j];
                if (isinf(wkj)) continue;
                const double t = wik + wkj;
                if (t < W[i * n + j]) W[i * n + j] = t;
            }
        }
    for (int64_t i = 0; i < n; i++)
        if (W[i * n + i] < 0.0) { free(W); fn_set_error("Negative cycle detected on node %lld", (long long)i); return TSR_EARG; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
    if (!out) { free(W); return TSR_ENOMEM; }
    memcpy(out, W, sizeof(double) * (size_t)(n * n));
    free(W);
    return TSR_OK;
}

/* method routines differ only in the positions of directed / unweighted in their SciPy signatures */
static int r_shortest_path(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return csg_shortest(args, nargs, 2, 4, &res[0]); }   /* (csgraph, method, directed, return_pred, unweighted, ...) */
static int r_dijkstra(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return csg_shortest(args, nargs, 1, 4, &res[0]); }   /* (csgraph, directed, indices, return_pred, unweighted, ...) */
static int r_bellman_ford(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return csg_shortest(args, nargs, 1, 4, &res[0]); }
static int r_johnson(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return csg_shortest(args, nargs, 1, 4, &res[0]); }
static int r_floyd_warshall(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return csg_shortest(args, nargs, 1, 3, &res[0]); }   /* (csgraph, directed, return_pred, unweighted, overwrite) */

/* ---------------------------------------------------------------- traversal + structure (dense adjacency) */
/* SciPy orders a node's neighbours as out-edges (row, ascending) first, then in-edges (column, ascending)
   not already seen as out-edges; directed graphs use the out pass only. The neighbour at scan position t<n
   is the out-candidate v=t; t>=n is the in-only candidate v=t-n. */

/* laplacian(csgraph, normed=False): L = diag(d) - A with d the column sums; normed gives the symmetric
   normalized Laplacian I - D^{-1/2} A D^{-1/2}. Returned dense (SciPy returns dense for a dense input). */
static int r_laplacian(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("laplacian: graph must be a square adjacency matrix"); return TSR_EARG; }
    int normed = nargs > 1 && args[1].kind == 4 && args[1].num != 0.0;
    int64_t n = args[0].arr.shape[0], tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    double *d = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
    if (!d || !out) { free(d); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t j = 0; j < n; j++) { double s = 0.0; for (int64_t i = 0; i < n; i++) s += A[i * n + j]; d[j] = s; }
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) {
        if (!normed) out[i * n + j] = (i == j ? d[i] : 0.0) - A[i * n + j];
        else {
            double wi = d[i] > 0.0 ? sqrt(d[i]) : 1.0;     /* isolated vertices normalize by 1 (SciPy) */
            double wj = d[j] > 0.0 ? sqrt(d[j]) : 1.0;
            double diag = (i == j && d[i] > 0.0) ? 1.0 : 0.0;
            out[i * n + j] = diag - A[i * n + j] / (wi * wj);
        }
    }
    free(d); fn_free_doubles(A, tot);
    return TSR_OK;
}

/* breadth_first_order(csgraph, i_start, directed=True, return_predecessors=True) -> (node_array, predecessors) */
static int r_breadth_first_order(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1] || nargs < 2) { fn_set_error("breadth_first_order: graph (square) and i_start required"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    int64_t start = (args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num;
    int directed = !(nargs > 2 && args[2].kind == 4 && args[2].num == 0.0);
    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot); if (!A) return TSR_ENOMEM;
    int64_t *order = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int32_t *pred = (int32_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    unsigned char *vis = (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1);
    if (!order || !pred || !vis) { free(order); free(pred); free(vis); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) pred[i] = -9999;
    int64_t cnt = 0, head = 0;
    vis[start] = 1; order[cnt++] = start;
    int64_t lim = directed ? n : 2 * n;
    while (head < cnt) {
        int64_t u = order[head++];
        for (int64_t t = 0; t < lim; t++) {
            int64_t v = t < n ? t : t - n;
            int out = A[u * n + v] != 0.0, in_ = A[v * n + u] != 0.0;
            if ((t < n ? out : (in_ && !out)) && !vis[v]) { vis[v] = 1; pred[v] = (int32_t)u; order[cnt++] = v; }
        }
    }
    int32_t *no = (int32_t *)fn_result_array(&res[0], TSR_I32, 1, (int64_t[]){cnt});
    int32_t *pr = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, (int64_t[]){n});
    if (!no || !pr) { free(order); free(pred); free(vis); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < cnt; i++) no[i] = (int32_t)order[i];
    memcpy(pr, pred, (size_t)n * sizeof(int32_t));
    free(order); free(pred); free(vis); fn_free_doubles(A, tot);
    return TSR_OK;
}

/* depth_first_order(csgraph, i_start, directed=True, return_predecessors=True) -> (node_array, predecessors) */
static int r_depth_first_order(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1] || nargs < 2) { fn_set_error("depth_first_order: graph (square) and i_start required"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    int64_t start = (args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num;
    int directed = !(nargs > 2 && args[2].kind == 4 && args[2].num == 0.0);
    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot); if (!A) return TSR_ENOMEM;
    int64_t *order = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *stack = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *iter = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int32_t *pred = (int32_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    unsigned char *vis = (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1);
    if (!order || !stack || !iter || !pred || !vis) { free(order); free(stack); free(iter); free(pred); free(vis); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) pred[i] = -9999;
    int64_t cnt = 0, sp = 0;
    int64_t lim = directed ? n : 2 * n;
    vis[start] = 1; order[cnt++] = start; stack[sp] = start; iter[start] = 0;
    while (sp >= 0) {
        int64_t u = stack[sp], t = iter[u], found = -1;
        while (t < lim) {
            int64_t v = t < n ? t : t - n;
            int out = A[u * n + v] != 0.0, in_ = A[v * n + u] != 0.0;
            if ((t < n ? out : (in_ && !out)) && !vis[v]) { found = v; break; }
            t++;
        }
        iter[u] = t + 1;
        if (found >= 0) { vis[found] = 1; pred[found] = (int32_t)u; order[cnt++] = found; sp++; stack[sp] = found; iter[found] = 0; }
        else sp--;
    }
    int32_t *no = (int32_t *)fn_result_array(&res[0], TSR_I32, 1, (int64_t[]){cnt});
    int32_t *pr = (int32_t *)fn_result_array(&res[1], TSR_I32, 1, (int64_t[]){n});
    if (!no || !pr) { free(order); free(stack); free(iter); free(pred); free(vis); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < cnt; i++) no[i] = (int32_t)order[i];
    memcpy(pr, pred, (size_t)n * sizeof(int32_t));
    free(order); free(stack); free(iter); free(pred); free(vis); fn_free_doubles(A, tot);
    return TSR_OK;
}

/* augmenting-path search for bipartite matching on the nonzero pattern (rows -> columns). */
static int csg_kuhn(const double *A, int64_t n, int64_t u, int64_t *matchR, unsigned char *seen)
{
    for (int64_t v = 0; v < n; v++) if (A[u * n + v] != 0.0 && !seen[v]) {
        seen[v] = 1;
        if (matchR[v] < 0 || csg_kuhn(A, n, matchR[v], matchR, seen)) { matchR[v] = u; return 1; }
    }
    return 0;
}

/* structural_rank(graph): size of a maximum matching of the sparsity pattern (bipartite rows/columns). */
static int r_structural_rank(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("structural_rank: graph must be square"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0], tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    int64_t *matchR = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    unsigned char *seen = (unsigned char *)malloc((size_t)(n > 0 ? n : 1));
    if (!matchR || !seen) { free(matchR); free(seen); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) matchR[i] = -1;
    int64_t rank = 0;
    for (int64_t u = 0; u < n; u++) { memset(seen, 0, (size_t)n); if (csg_kuhn(A, n, u, matchR, seen)) rank++; }
    free(matchR); free(seen); fn_free_doubles(A, tot);
    fn_result_int(&res[0], rank);
    return TSR_OK;
}

/* BFS/DFS predecessor arrays (pred[j] = parent, -1 if unreached/start), SciPy's out-then-in neighbour order. */
static void csg_bfs_pred(const double *A, int64_t n, int64_t start, int directed, int64_t *pred, int64_t *q)
{
    unsigned char *vis = (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1);
    if (!vis) return;
    for (int64_t i = 0; i < n; i++) pred[i] = -1;
    int64_t cnt = 0, head = 0, lim = directed ? n : 2 * n;
    vis[start] = 1; q[cnt++] = start;
    while (head < cnt) {
        int64_t u = q[head++];
        for (int64_t t = 0; t < lim; t++) {
            int64_t v = t < n ? t : t - n;
            int out = A[u * n + v] != 0.0, in_ = A[v * n + u] != 0.0;
            if ((t < n ? out : (in_ && !out)) && !vis[v]) { vis[v] = 1; pred[v] = u; q[cnt++] = v; }
        }
    }
    free(vis);
}
static void csg_dfs_pred(const double *A, int64_t n, int64_t start, int directed, int64_t *pred, int64_t *stack, int64_t *iter)
{
    unsigned char *vis = (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1);
    if (!vis) return;
    for (int64_t i = 0; i < n; i++) pred[i] = -1;
    int64_t sp = 0, lim = directed ? n : 2 * n;
    vis[start] = 1; stack[sp] = start; iter[start] = 0;
    while (sp >= 0) {
        int64_t u = stack[sp], t = iter[u], found = -1;
        while (t < lim) {
            int64_t v = t < n ? t : t - n;
            int out = A[u * n + v] != 0.0, in_ = A[v * n + u] != 0.0;
            if ((t < n ? out : (in_ && !out)) && !vis[v]) { found = v; break; }
            t++;
        }
        iter[u] = t + 1;
        if (found >= 0) { vis[found] = 1; pred[found] = u; sp++; stack[sp] = found; iter[found] = 0; }
        else sp--;
    }
    free(vis);
}

/* emit a dense tree from a predecessor vector: edge (pred[j], j) at [pred[j]][j] with its graph weight. */
static void csg_emit_tree(const double *A, int64_t n, int directed, const int64_t *pred, double *out)
{
    for (int64_t i = 0; i < n * n; i++) out[i] = 0.0;
    for (int64_t j = 0; j < n; j++) {
        int64_t p = pred[j];
        if (p < 0 || p >= n) continue;
        double w = A[p * n + j] != 0.0 ? A[p * n + j] : (!directed ? A[j * n + p] : 0.0);
        out[p * n + j] = w;
    }
}

/* construct_dist_matrix(graph, predecessors, directed=True): distance from i to j along the predecessor tree
   rooted at i; 0 on the diagonal, inf where no path. */
static int r_construct_dist_matrix(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1] || nargs < 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("construct_dist_matrix: graph (square) and 2-D predecessors required"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    int directed = !(nargs > 2 && args[2].kind == 4 && args[2].num == 0.0);
    int64_t ta, tp; double *A = fn_arg_doubles(&args[0], &ta); if (!A) return TSR_ENOMEM;
    double *P = fn_arg_doubles(&args[1], &tp); if (!P) { fn_free_doubles(A, ta); return TSR_ENOMEM; }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
    if (!out) { fn_free_doubles(A, ta); fn_free_doubles(P, tp); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) {
        if (i == j) { out[i * n + j] = 0.0; continue; }
        double d = 0.0; int64_t cur = j, steps = 0, ok = 0;
        while (steps++ <= n) {
            int64_t p = (int64_t)P[i * n + cur];
            if (p < 0 || p >= n) break;                                   /* -9999 sentinel: no path */
            double w = A[p * n + cur] != 0.0 ? A[p * n + cur] : (!directed ? A[cur * n + p] : 0.0);
            d += w; cur = p;
            if (cur == i) { ok = 1; break; }
        }
        out[i * n + j] = ok ? d : INFINITY;
    }
    fn_free_doubles(A, ta); fn_free_doubles(P, tp);
    return TSR_OK;
}

static int r_reconstruct_path(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1] || nargs < 2 || args[1].kind != 3) { fn_set_error("reconstruct_path: graph (square) and predecessors required"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    int directed = !(nargs > 2 && args[2].kind == 4 && args[2].num == 0.0);
    int64_t ta, tp; double *A = fn_arg_doubles(&args[0], &ta); if (!A) return TSR_ENOMEM;
    double *P = fn_arg_doubles(&args[1], &tp); if (!P) { fn_free_doubles(A, ta); return TSR_ENOMEM; }
    int64_t *pred = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
    if (!pred || !out) { free(pred); fn_free_doubles(A, ta); fn_free_doubles(P, tp); return TSR_ENOMEM; }
    for (int64_t j = 0; j < n; j++) { int64_t p = (int64_t)P[j]; pred[j] = (p >= 0 && p < n) ? p : -1; }
    csg_emit_tree(A, n, directed, pred, out);
    free(pred); fn_free_doubles(A, ta); fn_free_doubles(P, tp);
    return TSR_OK;
}

static int r_first_tree(const tsr_arg *args, int nargs, tsr_result *res, int bfs)
{
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1] || nargs < 2) { fn_set_error("*_tree: graph (square) and i_start required"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0];
    int64_t start = (args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num;
    int directed = !(nargs > 2 && args[2].kind == 4 && args[2].num == 0.0);
    int64_t ta; double *A = fn_arg_doubles(&args[0], &ta); if (!A) return TSR_ENOMEM;
    int64_t *pred = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *s1 = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *s2 = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
    if (!pred || !s1 || !s2 || !out) { free(pred); free(s1); free(s2); fn_free_doubles(A, ta); return TSR_ENOMEM; }
    if (bfs) csg_bfs_pred(A, n, start, directed, pred, s1);
    else csg_dfs_pred(A, n, start, directed, pred, s1, s2);
    csg_emit_tree(A, n, directed, pred, out);
    free(pred); free(s1); free(s2); fn_free_doubles(A, ta);
    return TSR_OK;
}
static int r_breadth_first_tree(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return r_first_tree(args, nargs, res, 1); }
static int r_depth_first_tree(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return r_first_tree(args, nargs, res, 0); }

/* minimum_spanning_tree(csgraph): Kruskal MST of the undirected graph, returned dense with each tree edge
   at [min(i,j)][max(i,j)] = weight (SciPy's orientation). Unique (so order-independent) for distinct weights. */
typedef struct { int64_t i, j; double w; } csg_ed;
static int csg_edcmp(const void *a, const void *b)
{
    double wa = ((const csg_ed *)a)->w, wb = ((const csg_ed *)b)->w;
    return wa < wb ? -1 : (wa > wb ? 1 : 0);
}
static int64_t csg_find(int64_t *p, int64_t x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }

static int r_minimum_spanning_tree(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("minimum_spanning_tree: graph must be square"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0], tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
    csg_ed *ed = (csg_ed *)malloc((size_t)(n * n + 1) * sizeof(csg_ed));
    int64_t *p = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    if (!out || !ed || !p) { free(ed); free(p); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n * n; i++) out[i] = 0.0;
    int64_t ne = 0;
    for (int64_t i = 0; i < n; i++) for (int64_t j = i + 1; j < n; j++) {
        double w = A[i * n + j] != 0.0 ? A[i * n + j] : A[j * n + i];
        if (w != 0.0) { ed[ne].i = i; ed[ne].j = j; ed[ne].w = w; ne++; }
    }
    qsort(ed, (size_t)ne, sizeof(csg_ed), csg_edcmp);
    for (int64_t i = 0; i < n; i++) p[i] = i;
    for (int64_t e = 0; e < ne; e++) {
        int64_t ri = csg_find(p, ed[e].i), rj = csg_find(p, ed[e].j);
        if (ri != rj) { p[ri] = rj; out[ed[e].i * n + ed[e].j] = ed[e].w; }
    }
    free(ed); free(p); fn_free_doubles(A, tot);
    return TSR_OK;
}

/* min_weight_full_bipartite_matching(biadjacency): minimum-weight perfect matching of a square biadjacency
   (stored entry = edge weight, 0 = no edge). Hungarian algorithm (O(n^3), potentials). Returns (row_ind,
   col_ind); unique -- hence SciPy-identical -- when the finite edge weights are distinct. */
static int r_min_weight_full_bipartite_matching(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) { fn_set_error("min_weight_full_bipartite_matching: a square biadjacency is required here"); return TSR_EARG; }
    int64_t n = args[0].arr.shape[0], tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    const double BIG = 1e15;
    double *u = (double *)calloc((size_t)(n + 1), sizeof(double));
    double *v = (double *)calloc((size_t)(n + 1), sizeof(double));
    double *minv = (double *)malloc((size_t)(n + 1) * sizeof(double));
    int64_t *p = (int64_t *)calloc((size_t)(n + 1), sizeof(int64_t));        /* p[j] = row (1-based) matched to col j */
    int64_t *way = (int64_t *)calloc((size_t)(n + 1), sizeof(int64_t));
    char *used = (char *)malloc((size_t)(n + 1));
    if (!u || !v || !minv || !p || !way || !used) { free(u); free(v); free(minv); free(p); free(way); free(used); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 1; i <= n; i++) {
        p[0] = i; int64_t j0 = 0;
        for (int64_t j = 0; j <= n; j++) { minv[j] = BIG * 1e3; used[j] = 0; }
        do {
            used[j0] = 1;
            int64_t i0 = p[j0], j1 = -1; double delta = BIG * 1e3;
            for (int64_t j = 1; j <= n; j++) if (!used[j]) {
                double cost = (A[(i0 - 1) * n + (j - 1)] != 0.0 ? A[(i0 - 1) * n + (j - 1)] : BIG);
                double cur = cost - u[i0] - v[j];
                if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
                if (minv[j] < delta) { delta = minv[j]; j1 = j; }
            }
            for (int64_t j = 0; j <= n; j++) {
                if (used[j]) { u[p[j]] += delta; v[j] -= delta; }
                else minv[j] -= delta;
            }
            j0 = j1;
        } while (p[j0] != 0);
        do { int64_t j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
    }
    int64_t sh[1] = {n};
    int64_t *ri = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, sh);
    int64_t *ci = (int64_t *)fn_result_array(&res[1], TSR_I64, 1, sh);
    if (!ri || !ci) { free(u); free(v); free(minv); free(p); free(way); free(used); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) ri[i] = i;
    for (int64_t j = 1; j <= n; j++) ci[p[j] - 1] = j - 1;                   /* column matched to each row */
    free(u); free(v); free(minv); free(p); free(way); free(used); fn_free_doubles(A, tot);
    return TSR_OK;
}

static int csg_dinic_bfs(const int64_t *cap, int64_t n, int64_t s, int64_t t, int64_t *level, int64_t *q)
{
    for (int64_t i = 0; i < n; i++) level[i] = -1;
    int64_t head = 0, tail = 0;
    level[s] = 0; q[tail++] = s;
    while (head < tail) {
        const int64_t u = q[head++];
        const int64_t *row = cap + u * n;
        for (int64_t v = 0; v < n; v++)
            if (row[v] > 0 && level[v] < 0) { level[v] = level[u] + 1; q[tail++] = v; }
    }
    return level[t] >= 0;
}

/* Dinic blocking-flow DFS, iterative (an explicit stack avoids any recursion depth concern on
 * large n). `it[u]` is the current-arc pointer for u (reset each phase). Returns the flow pushed
 * along one augmenting path in the level graph, 0 when none remains. `pathu`/`pathcap` are scratch
 * of length n+1 holding the current partial path (nodes and the residual of the arc entering them).*/
static int64_t csg_dinic_push(int64_t *cap, int64_t n, int64_t s, int64_t t,
                              const int64_t *level, int64_t *it, int64_t *pathu, int64_t *pathcap)
{
    int64_t depth = 0;
    pathu[0] = s; pathcap[0] = INT64_MAX;                 /* source has no entering arc */
    while (depth >= 0) {
        const int64_t u = pathu[depth];
        if (u == t) {                                     /* found an augmenting path: push bottleneck */
            int64_t f = INT64_MAX;
            for (int64_t k = 1; k <= depth; k++) if (pathcap[k] < f) f = pathcap[k];
            for (int64_t k = 1; k <= depth; k++) {
                const int64_t a = pathu[k - 1], b = pathu[k];
                cap[a * n + b] -= f; cap[b * n + a] += f;
            }
            return f;
        }
        int64_t advanced = 0;
        const int64_t *row = cap + u * n;
        for (int64_t v = it[u]; v < n; v++) {
            if (row[v] > 0 && level[v] == level[u] + 1) {
                it[u] = v;                                /* remember where we are for this u */
                pathu[depth + 1] = v;
                pathcap[depth + 1] = row[v];
                depth++;
                advanced = 1;
                break;
            }
        }
        if (!advanced) {                                  /* dead end: retreat, disable u this phase */
            it[u] = n;
            depth--;
            if (depth >= 0) it[pathu[depth]]++;           /* don't revisit the arc into u */
        }
    }
    return 0;
}

/* maximum_flow(csgraph, source, sink, method='dinic'): Dinic max-flow on a dense INTEGER capacity
 * matrix. Returns {flow_value (int), flow (n-by-n int32, antisymmetric net flow)}. */
static int r_maximum_flow(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) {
        fn_set_error("maximum_flow: the graph must be a square 2-D capacity matrix");
        return TSR_EARG;
    }
    if (nargs < 3) { fn_set_error("maximum_flow: source and sink are required"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    const int64_t s = (args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num;
    const int64_t t = (args[2].flags & 1) ? args[2].ival : (int64_t)args[2].num;
    if (s < 0 || s >= n || t < 0 || t >= n) { fn_set_error("maximum_flow: source/sink out of range"); return TSR_EARG; }

    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;

    int64_t *cap = (int64_t *)malloc((size_t)(n * n) * sizeof(int64_t));
    int64_t *orig = (int64_t *)malloc((size_t)(n * n) * sizeof(int64_t));
    int64_t *level = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *q = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *it = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *pu = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    int64_t *pc = (int64_t *)malloc((size_t)(n + 1) * sizeof(int64_t));
    int32_t *flow = (int32_t *)fn_result_array(&res[1], TSR_I32, 2, (int64_t[]){n, n});
    if (!cap || !orig || !level || !q || !it || !pu || !pc || !flow) {
        free(cap); free(orig); free(level); free(q); free(it); free(pu); free(pc);
        fn_free_doubles(A, tot); return TSR_ENOMEM;
    }

    /* integer capacities; residual (cap) starts as a copy of the original (orig). Capacities on the
     * reverse arcs begin at 0 and grow as flow is pushed -- this is what makes `used` antisymmetric. */
    for (int64_t i = 0; i < n * n; i++) { int64_t c = (int64_t)A[i]; orig[i] = c; cap[i] = c; }
    fn_free_doubles(A, tot);

    int64_t flow_value = 0;
    if (s != t) {
        while (csg_dinic_bfs(cap, n, s, t, level, q)) {
            for (int64_t i = 0; i < n; i++) it[i] = 0;
            int64_t f;
            while ((f = csg_dinic_push(cap, n, s, t, level, it, pu, pc)) > 0) flow_value += f;
        }
    }

    /* net flow F[i][j] = orig[i][j] - cap[i][j] (consumed residual). Already antisymmetric:
     * forward arc loses capacity (positive), its reverse gains the same (negative). */
    for (int64_t i = 0; i < n * n; i++) flow[i] = (int32_t)(orig[i] - cap[i]);

    fn_result_int(&res[0], flow_value);
    free(cap); free(orig); free(level); free(q); free(it); free(pu); free(pc);
    return TSR_OK;
}

/* ---------------------------------------------------------------- reverse_cuthill_mckee ----------
 * reverse_cuthill_mckee(graph, symmetric_mode=False) -> node_array (int32, length n)
 *
 * RCM reordering of a sparse symmetric matrix, matching scipy.sparse.csgraph.reverse_cuthill_mckee.
 * The graph arrives as a dense square matrix (0 = no edge). The sparsity pattern is symmetrised
 * (edge if graph[i][j] or graph[j][i] is nonzero), as SciPy does for symmetric_mode either way here.
 * Algorithm: per connected sweep seed the not-yet-ordered node of minimum structural degree (ties ->
 * smallest index); BFS from it, expanding each node's not-yet-ordered neighbours in ascending-degree
 * (ties: ascending index) order; finally reverse the whole order. The permutation is non-unique under
 * degree ties, so the parity harness compares it semantically by permuted bandwidth. */
static int r_reverse_cuthill_mckee(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) {
        fn_set_error("reverse_cuthill_mckee: the graph must be a square 2-D matrix");
        return TSR_EARG;
    }
    const int64_t n = args[0].arr.shape[0];
    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    const size_t nn = (size_t)(n > 0 ? n : 1);
    int64_t *deg = (int64_t *)malloc(nn * sizeof(int64_t));
    int64_t *order = (int64_t *)malloc(nn * sizeof(int64_t));
    unsigned char *ordered = (unsigned char *)calloc(nn, 1);
    if (!deg || !order || !ordered) { free(deg); free(order); free(ordered); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    /* structural degree over the symmetrised pattern */
    for (int64_t i = 0; i < n; i++) {
        int64_t d = 0;
        for (int64_t j = 0; j < n; j++)
            if (i != j && (A[i * n + j] != 0.0 || A[j * n + i] != 0.0)) d++;
        deg[i] = d;
    }
    int64_t cnt = 0;
    while (cnt < n) {
        /* seed = not-yet-ordered node of minimum degree (ties: smallest index) */
        int64_t seed = -1;
        for (int64_t i = 0; i < n; i++)
            if (!ordered[i] && (seed < 0 || deg[i] < deg[seed])) seed = i;
        ordered[seed] = 1; order[cnt++] = seed;
        int64_t head = cnt - 1;
        while (head < cnt) {
            const int64_t u = order[head++];
            /* append u's not-yet-ordered neighbours, sorted by ascending degree then index */
            const int64_t base = cnt;
            for (int64_t v = 0; v < n; v++)
                if (v != u && !ordered[v] && (A[u * n + v] != 0.0 || A[v * n + u] != 0.0)) {
                    ordered[v] = 1; order[cnt++] = v;
                }
            /* insertion sort the freshly-added block [base, cnt) by (deg, index) */
            for (int64_t p = base + 1; p < cnt; p++) {
                const int64_t key = order[p];
                int64_t q = p - 1;
                while (q >= base && (deg[order[q]] > deg[key] || (deg[order[q]] == deg[key] && order[q] > key))) {
                    order[q + 1] = order[q]; q--;
                }
                order[q + 1] = key;
            }
        }
    }
    /* reverse */
    int32_t *out = (int32_t *)fn_result_array(&res[0], TSR_I32, 1, (int64_t[]){n});
    if (!out) { free(deg); free(order); free(ordered); fn_free_doubles(A, tot); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) out[i] = (int32_t)order[n - 1 - i];
    free(deg); free(order); free(ordered); fn_free_doubles(A, tot);
    return TSR_OK;
}

/* ---------------------------------------------------------------- maximum_bipartite_matching ---------------
 * maximum_bipartite_matching(graph, perm_type='row') -> node_array
 *
 * scipy.sparse.csgraph.maximum_bipartite_matching finds a maximum-cardinality matching of a bipartite graph
 * given by an m x n biadjacency matrix (rows on one side, columns on the other); any structural nonzero is an
 * edge, weights are ignored. The result convention (confirmed in VERIFY.py against scipy 1.17.1):
 *   perm_type='row'    -> an array of length n (one per COLUMN); entry j is the ROW matched to column j, or -1.
 *   perm_type='column' -> an array of length m (one per ROW);    entry i is the COLUMN matched to row i, or -1.
 * The matching itself is non-unique (only its cardinality is), so the parity harness validates it semantically.
 *
 * We compute a maximum matching of the nonzero pattern with Kuhn's augmenting-path algorithm (iterative DFS per
 * row), identical in cardinality to scipy's Hopcroft-Karp result. Input here is a dense m x n matrix (the CSR
 * biadjacency densified by the binding), matching how the other csgraph routines read their graph. */
static int r_maximum_bipartite_matching(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) {
        fn_set_error("maximum_bipartite_matching: the graph must be a 2-D biadjacency matrix");
        return TSR_EARG;
    }
    const int64_t m = args[0].arr.shape[0];    /* rows side */
    const int64_t n = args[0].arr.shape[1];    /* columns side */
    /* perm_type: 'row' (default) returns one entry per column; 'column' returns one entry per row. */
    int col_perm = 0;                          /* 0 -> 'row' output (len n), 1 -> 'column' output (len m) */
    if (nargs > 1 && args[1].kind == 2 && args[1].str) {
        if (strcmp(args[1].str, "column") == 0) col_perm = 1;
        else if (strcmp(args[1].str, "row") != 0) {
            fn_set_error("maximum_bipartite_matching: perm_type must be 'row' or 'column'");
            return TSR_EARG;
        }
    }

    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;

    /* matchR[v] = row matched to column v (-1 none); matchC[u] = column matched to row u (-1 none). */
    int64_t *matchR = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    int64_t *matchC = (int64_t *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    unsigned char *seen = (unsigned char *)malloc((size_t)(n > 0 ? n : 1));
    /* iterative-DFS scratch so deep augmenting paths cannot overflow the C stack. */
    int64_t *stk_u = (int64_t *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    int64_t *stk_v = (int64_t *)malloc((size_t)(m > 0 ? m : 1) * sizeof(int64_t));
    if (!matchR || !matchC || !seen || !stk_u || !stk_v) {
        free(matchR); free(matchC); free(seen); free(stk_u); free(stk_v);
        fn_free_doubles(A, tot); return TSR_ENOMEM;
    }
    for (int64_t v = 0; v < n; v++) matchR[v] = -1;
    for (int64_t u = 0; u < m; u++) matchC[u] = -1;

    /* Kuhn's algorithm, one augmenting search per row, with an explicit stack standing in for the recursion.
     * Each frame is a row u scanning columns from stk_v[sp]; when it finds a free (or re-augmentable) column
     * it either matches it or descends into matchR[v]'s own search; on exhaustion it backtracks. */
    for (int64_t s = 0; s < m; s++) {
        for (int64_t v = 0; v < n; v++) seen[v] = 0;
        int64_t sp = 0;
        stk_u[0] = s; stk_v[0] = 0;
        while (sp >= 0) {
            int64_t u = stk_u[sp];
            int64_t v = stk_v[sp];
            int64_t chosen = -1;
            while (v < n) {
                if (A[u * n + v] != 0.0 && !seen[v]) { chosen = v; break; }
                v++;
            }
            if (chosen < 0) { sp--; continue; }     /* row u exhausted: backtrack */
            seen[chosen] = 1;
            stk_v[sp] = chosen + 1;                 /* resume after `chosen` if we return to this frame */
            if (matchR[chosen] < 0) {
                /* free column -> augment the whole path back up the stack */
                int64_t cv = chosen;
                for (int64_t f = sp; f >= 0; f--) {
                    int64_t ru = stk_u[f];
                    int64_t prev = matchC[ru];      /* column ru previously held, to re-link one level up */
                    matchR[cv] = ru; matchC[ru] = cv;
                    cv = prev;
                }
                break;                              /* this row s is matched; move to the next s */
            } else {
                /* column taken: descend into its current owner to look for an alternative */
                sp++;
                stk_u[sp] = matchR[chosen];
                stk_v[sp] = 0;
            }
        }
    }

    if (col_perm) {
        int32_t *out = (int32_t *)fn_result_array(&res[0], TSR_I32, 1, (int64_t[]){m});
        if (!out) { free(matchR); free(matchC); free(seen); free(stk_u); free(stk_v); fn_free_doubles(A, tot); return TSR_ENOMEM; }
        for (int64_t u = 0; u < m; u++) out[u] = (int32_t)matchC[u];
    } else {
        int32_t *out = (int32_t *)fn_result_array(&res[0], TSR_I32, 1, (int64_t[]){n});
        if (!out) { free(matchR); free(matchC); free(seen); free(stk_u); free(stk_v); fn_free_doubles(A, tot); return TSR_ENOMEM; }
        for (int64_t v = 0; v < n; v++) out[v] = (int32_t)matchR[v];
    }

    free(matchR); free(matchC); free(seen); free(stk_u); free(stk_v);
    fn_free_doubles(A, tot);
    return TSR_OK;
}

/* ---------------------------------------------------------------- Yen's K loopless shortest paths -------
 * scipy.sparse.csgraph.yen(csgraph, source, sink, K, directed=True) -> array of the shortest loopless path
 * DISTANCES between source and sink, sorted ascending. SciPy 1.17.1 returns ONLY the paths actually found
 * (length min(#paths, K)), so when fewer than K paths exist the array is SHORTER than K (an empty array
 * when the sink is unreachable) -- it is NOT inf-padded. Dense adjacency convention (0 = no edge).
 * Dijkstra for the base shortest path; Yen deviations remove root-path nodes and the already-used spur
 * edges to enumerate the next candidate. */

/* Dijkstra from src to sink over weight matrix W (n-by-n, INF = no edge), honouring removed nodes
 * (node_blocked[v]!=0) and removed directed edges (edge_blocked[u*n+v]!=0). Writes the path (sink..src
 * reversed into path[], length *plen) and returns its cost, or INFINITY when unreachable. */
static double csg_yen_dijkstra(const double *W, int64_t n, int64_t src, int64_t sink,
                               const char *node_blocked, const char *edge_blocked,
                               double *dist, int64_t *prev, char *done,
                               int64_t *path, int64_t *plen)
{
    for (int64_t i = 0; i < n; i++) { dist[i] = INFINITY; prev[i] = -1; done[i] = 0; }
    dist[src] = 0.0;
    for (;;) {
        /* O(n^2) selection: pick the unfinished node of least tentative distance */
        int64_t u = -1; double best = INFINITY;
        for (int64_t v = 0; v < n; v++)
            if (!done[v] && dist[v] < best) { best = dist[v]; u = v; }
        if (u < 0) break;
        done[u] = 1;
        if (u == sink) break;
        for (int64_t v = 0; v < n; v++) {
            if (v == u || done[v]) continue;
            if (node_blocked[v]) continue;
            if (edge_blocked && edge_blocked[u * n + v]) continue;
            const double w = W[u * n + v];
            if (isinf(w)) continue;
            const double nd = dist[u] + w;
            if (nd < dist[v]) { dist[v] = nd; prev[v] = u; }
        }
    }
    if (isinf(dist[sink])) { *plen = 0; return INFINITY; }
    /* reconstruct sink..src then reverse */
    int64_t len = 0, cur = sink;
    while (cur != -1) { path[len++] = cur; if (cur == src) break; cur = prev[cur]; }
    if (path[len - 1] != src) { *plen = 0; return INFINITY; }
    for (int64_t a = 0, b = len - 1; a < b; a++, b--) { int64_t tmp = path[a]; path[a] = path[b]; path[b] = tmp; }
    *plen = len;
    return dist[sink];
}

static int r_yen(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) {
        fn_set_error("yen: the graph must be a square 2-D adjacency matrix"); return TSR_EARG;
    }
    if (nargs < 4) { fn_set_error("yen: source, sink and K are required"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    const int64_t src  = (args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num;
    const int64_t sink = (args[2].flags & 1) ? args[2].ival : (int64_t)args[2].num;
    const int64_t K    = (args[3].flags & 1) ? args[3].ival : (int64_t)args[3].num;
    const int directed = csg_flag(args, nargs, 4, 1);
    if (src < 0 || src >= n || sink < 0 || sink >= n) { fn_set_error("yen: source/sink out of range"); return TSR_EARG; }
    if (K <= 0) { fn_set_error("yen: K must be positive"); return TSR_EARG; }

    int64_t tot; double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    double *W = csg_weights(A, n, directed, 0);       /* 0->INF, undirected symmetrises by smaller dir */
    fn_free_doubles(A, tot);
    if (!W) return TSR_ENOMEM;

    /* scratch for Dijkstra */
    double  *dist = (double *)malloc((size_t)(n > 0 ? n : 1) * sizeof(double));
    int64_t *prev = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    char    *done = (char *)malloc((size_t)(n > 0 ? n : 1) * sizeof(char));
    char    *node_blocked = (char *)malloc((size_t)(n > 0 ? n : 1) * sizeof(char));
    char    *edge_blocked = (char *)malloc((size_t)(n * n > 0 ? n * n : 1) * sizeof(char));
    int64_t *tmp_path = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));

    /* accepted paths A: each stored as (cost, length, node list of <=n nodes) */
    double  *A_cost = (double *)malloc((size_t)K * sizeof(double));
    int64_t *A_len  = (int64_t *)malloc((size_t)K * sizeof(int64_t));
    int64_t *A_node = (int64_t *)malloc((size_t)K * (size_t)(n > 0 ? n : 1) * sizeof(int64_t));
    /* candidate pool B: up to K*n candidates, each a full path */
    const int64_t Bcap = (K * (n > 0 ? n : 1)) + 1;
    double  *B_cost = (double *)malloc((size_t)Bcap * sizeof(double));
    int64_t *B_len  = (int64_t *)malloc((size_t)Bcap * sizeof(int64_t));
    int64_t *B_node = (int64_t *)malloc((size_t)Bcap * (size_t)(n > 0 ? n : 1) * sizeof(int64_t));

    if (!dist || !prev || !done || !node_blocked || !edge_blocked || !tmp_path ||
        !A_cost || !A_len || !A_node || !B_cost || !B_len || !B_node) {
        free(W); free(dist); free(prev); free(done); free(node_blocked); free(edge_blocked); free(tmp_path);
        free(A_cost); free(A_len); free(A_node); free(B_cost); free(B_len); free(B_node);
        return TSR_ENOMEM;
    }

    int64_t nA = 0, nB = 0;

    /* base shortest path */
    memset(node_blocked, 0, (size_t)(n > 0 ? n : 1));
    int64_t plen;
    double d0 = csg_yen_dijkstra(W, n, src, sink, node_blocked, NULL, dist, prev, done, tmp_path, &plen);
    if (!isinf(d0)) {
        A_cost[0] = d0; A_len[0] = plen;
        for (int64_t i = 0; i < plen; i++) A_node[i] = tmp_path[i];
        nA = 1;
    }

    while (nA > 0 && nA < K) {
        const int64_t *prev_path = &A_node[(nA - 1) * n];
        const int64_t prev_plen = A_len[nA - 1];
        for (int64_t i = 0; i < prev_plen - 1; i++) {
            const int64_t spur_node = prev_path[i];
            /* root_path = prev_path[0..i] */
            /* block edges whose root prefix equals this root path */
            memset(edge_blocked, 0, (size_t)(n * n));
            for (int64_t a = 0; a < nA; a++) {
                const int64_t *p = &A_node[a * n];
                if (A_len[a] <= i + 1) continue;
                int same = 1;
                for (int64_t b = 0; b <= i; b++) if (p[b] != prev_path[b]) { same = 0; break; }
                if (same) edge_blocked[p[i] * n + p[i + 1]] = 1;
            }
            /* block root-path nodes except the spur node itself */
            memset(node_blocked, 0, (size_t)n);
            for (int64_t b = 0; b < i; b++) node_blocked[prev_path[b]] = 1;

            int64_t slen;
            double dspur = csg_yen_dijkstra(W, n, spur_node, sink, node_blocked, edge_blocked,
                                            dist, prev, done, tmp_path, &slen);
            if (isinf(dspur)) continue;

            /* total path = root_path[0..i-1] + spur_path (spur_path starts at spur_node) */
            int64_t total_len = i + slen;             /* i prefix nodes + slen spur nodes */
            if (total_len > n) continue;              /* loopless must fit in n nodes */
            int64_t tp[1]; (void)tp;
            /* build into B slot nB */
            int64_t *dest = &B_node[nB * n];
            for (int64_t b = 0; b < i; b++) dest[b] = prev_path[b];
            for (int64_t b = 0; b < slen; b++) dest[i + b] = tmp_path[b];
            /* root cost */
            double root_cost = 0.0;
            for (int64_t b = 0; b < i; b++) root_cost += W[prev_path[b] * n + prev_path[b + 1]];
            double total_cost = root_cost + dspur;

            /* loopless check: no repeated node */
            int loop = 0;
            for (int64_t a = 0; a < total_len && !loop; a++)
                for (int64_t b = a + 1; b < total_len; b++)
                    if (dest[a] == dest[b]) { loop = 1; break; }
            if (loop) continue;

            /* dedup against existing B and A */
            int dup = 0;
            for (int64_t a = 0; a < nB && !dup; a++) {
                if (B_len[a] != total_len) continue;
                int eq = 1;
                const int64_t *q = &B_node[a * n];
                for (int64_t b = 0; b < total_len; b++) if (q[b] != dest[b]) { eq = 0; break; }
                if (eq) dup = 1;
            }
            for (int64_t a = 0; a < nA && !dup; a++) {
                if (A_len[a] != total_len) continue;
                int eq = 1;
                const int64_t *q = &A_node[a * n];
                for (int64_t b = 0; b < total_len; b++) if (q[b] != dest[b]) { eq = 0; break; }
                if (eq) dup = 1;
            }
            if (dup) continue;

            B_cost[nB] = total_cost; B_len[nB] = total_len; nB++;
        }

        if (nB == 0) break;
        /* pop the minimum-cost candidate from B into A */
        int64_t bmin = 0;
        for (int64_t a = 1; a < nB; a++) if (B_cost[a] < B_cost[bmin]) bmin = a;
        A_cost[nA] = B_cost[bmin]; A_len[nA] = B_len[bmin];
        for (int64_t b = 0; b < B_len[bmin]; b++) A_node[nA * n + b] = B_node[bmin * n + b];
        nA++;
        /* remove bmin from B (compact) */
        for (int64_t a = bmin; a < nB - 1; a++) {
            B_cost[a] = B_cost[a + 1]; B_len[a] = B_len[a + 1];
            for (int64_t b = 0; b < B_len[a]; b++) B_node[a * n + b] = B_node[(a + 1) * n + b];
        }
        nB--;
    }

    /* emit exactly the found distances (min(nA, K)), sorted ascending -- SciPy does not inf-pad */
    for (int64_t a = 0; a < nA; a++)
        for (int64_t b = a + 1; b < nA; b++)
            if (A_cost[b] < A_cost[a]) { double t = A_cost[a]; A_cost[a] = A_cost[b]; A_cost[b] = t; }
    const int64_t nout = nA < K ? nA : K;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nout});
    if (!out) {
        free(W); free(dist); free(prev); free(done); free(node_blocked); free(edge_blocked); free(tmp_path);
        free(A_cost); free(A_len); free(A_node); free(B_cost); free(B_len); free(B_node);
        return TSR_ENOMEM;
    }
    for (int64_t a = 0; a < nout; a++) out[a] = A_cost[a];

    free(W); free(dist); free(prev); free(done); free(node_blocked); free(edge_blocked); free(tmp_path);
    free(A_cost); free(A_len); free(A_node); free(B_cost); free(B_len); free(B_node);
    return TSR_OK;
}

/* dense <-> CSR converters: scipy.sparse.csgraph.csgraph_to_dense / csgraph_from_dense.
 * A scipy csr_array is not portable across Tessero's two backends, so the CSR side is carried as
 * the raw (indptr, indices, data) arrays plus the node count, as maximum_flow carries a raw
 * capacity matrix.
 *
 *   csgraph_to_dense(data, indices, indptr, n_nodes, null_value=0) -> dense (n x n) float64
 *       dense is filled with null_value, then every stored (i,j) entry is written; among duplicate
 *       (i,j) entries the MINIMUM value wins (SciPy's documented behaviour). No diagonal casing.
 *
 *   csgraph_from_dense(graph, null_value=0, nan_null=True, infinity_null=True) -> (indptr, indices, data)
 *       a CSR triple with indices ascending within each row. A[i][j] is a stored edge unless it is
 *       null: NaN (when nan_null), +/-inf (when infinity_null), or == null_value (when not None). */

static int r_csgraph_to_dense(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 4 || args[0].kind != 3 || args[1].kind != 3 || args[2].kind != 3) {
        fn_set_error("csgraph_to_dense: data, indices, indptr and n_nodes are required");
        return TSR_EARG;
    }
    const int64_t n = (args[3].flags & 1) ? args[3].ival : (int64_t)args[3].num;
    if (n < 0) { fn_set_error("csgraph_to_dense: n_nodes must be non-negative"); return TSR_EARG; }
    const double null_value = (nargs > 4 && args[4].kind != 0) ? args[4].num : 0.0;
    int64_t ndat, nind, nptr;
    double *data = fn_arg_doubles(&args[0], &ndat);
    double *indices = fn_arg_doubles(&args[1], &nind);
    double *indptr = fn_arg_doubles(&args[2], &nptr);
    if (!data || !indices || !indptr) {
        fn_free_doubles(data, ndat); fn_free_doubles(indices, nind); fn_free_doubles(indptr, nptr);
        return TSR_ENOMEM;
    }
    if (nptr != n + 1) {
        fn_free_doubles(data, ndat); fn_free_doubles(indices, nind); fn_free_doubles(indptr, nptr);
        fn_set_error("csgraph_to_dense: indptr must have n_nodes+1 entries"); return TSR_EARG;
    }
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
    char *seen = (char *)calloc((size_t)(n * n > 0 ? n * n : 1), 1);
    if (!out || !seen) {
        free(seen);
        fn_free_doubles(data, ndat); fn_free_doubles(indices, nind); fn_free_doubles(indptr, nptr);
        return TSR_ENOMEM;
    }
    for (int64_t k = 0; k < n * n; k++) out[k] = null_value;
    for (int64_t i = 0; i < n; i++) {
        const int64_t lo = (int64_t)indptr[i], hi = (int64_t)indptr[i + 1];
        for (int64_t k = lo; k < hi; k++) {
            const int64_t j = (int64_t)indices[k];
            if (j < 0 || j >= n) continue;
            const double v = data[k];
            const int64_t p = i * n + j;
            if (!seen[p]) { out[p] = v; seen[p] = 1; }
            else if (v < out[p]) out[p] = v;
        }
    }
    free(seen);
    fn_free_doubles(data, ndat); fn_free_doubles(indices, nind); fn_free_doubles(indptr, nptr);
    return TSR_OK;
}

static int r_csgraph_from_dense(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (nargs < 1 || args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[0] != args[0].arr.shape[1]) {
        fn_set_error("csgraph_from_dense: graph must be a square 2-D matrix");
        return TSR_EARG;
    }
    const int64_t n = args[0].arr.shape[0];
    const int have_null = !(nargs > 1 && args[1].kind == 0);
    const double null_value = (nargs > 1 && args[1].kind != 0) ? args[1].num : 0.0;
    const int nan_null = csg_flag(args, nargs, 2, 1);
    const int infinity_null = csg_flag(args, nargs, 3, 1);
    int64_t tot;
    double *A = fn_arg_doubles(&args[0], &tot);
    if (!A) return TSR_ENOMEM;
    int64_t nnz = 0;
    for (int64_t k = 0; k < n * n; k++) {
        const double v = A[k];
        int is_null = 0;
        if (nan_null && isnan(v)) is_null = 1;
        else if (infinity_null && isinf(v)) is_null = 1;
        else if (have_null && v == null_value) is_null = 1;
        if (!is_null) nnz++;
    }
    int64_t *indptr = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, (int64_t[]){n + 1});
    int64_t *indices = (int64_t *)fn_result_array(&res[1], TSR_I64, 1, (int64_t[]){nnz});
    double *data = (double *)fn_result_array(&res[2], TSR_F64, 1, (int64_t[]){nnz});
    if (!indptr || !indices || (nnz > 0 && !data)) { fn_free_doubles(A, tot); return TSR_ENOMEM; }
    int64_t pos = 0;
    indptr[0] = 0;
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < n; j++) {
            const double v = A[i * n + j];
            int is_null = 0;
            if (nan_null && isnan(v)) is_null = 1;
            else if (infinity_null && isinf(v)) is_null = 1;
            else if (have_null && v == null_value) is_null = 1;
            if (!is_null) { indices[pos] = j; data[pos] = v; pos++; }
        }
        indptr[i + 1] = pos;
    }
    fn_free_doubles(A, tot);
    return TSR_OK;
}

static const fn_def DEFS[] = {
    ROUTINE("csgraph.maximum_flow", 2, "csgraph, source, sink, method='dinic'", "flow_value, flow", r_maximum_flow, NULL, "Maximum single-source single-sink flow on a dense integer capacity matrix via Dinic's algorithm (scipy.sparse.csgraph.maximum_flow)."),
    ROUTINE("csgraph.reverse_cuthill_mckee", 1, "graph, symmetric_mode=False", "node_array", r_reverse_cuthill_mckee, NULL, "Reverse Cuthill-McKee ordering of a sparse symmetric matrix (scipy.sparse.csgraph.reverse_cuthill_mckee)."),
    ROUTINE("csgraph.maximum_bipartite_matching", 1, "graph, perm_type='row'", "node_array", r_maximum_bipartite_matching, NULL, "Maximum bipartite matching of a sparse bipartite graph (scipy.sparse.csgraph.maximum_bipartite_matching)."),
    ROUTINE("csgraph.yen", 1, "csgraph, source, sink, K, directed=True", "dist_array", r_yen, NULL, "K shortest loopless paths' distances via Yen's algorithm (scipy.sparse.csgraph.yen)."),
    ROUTINE("csgraph.csgraph_to_dense", 1, "data, indices, indptr, n_nodes, null_value=0", "graph", r_csgraph_to_dense, NULL, "Dense adjacency matrix from a CSR triple (scipy.sparse.csgraph.csgraph_to_dense; CSR passed as data/indices/indptr/n_nodes)."),
    ROUTINE("csgraph.csgraph_from_dense", 3, "graph, null_value=0, nan_null=True, infinity_null=True", "indptr, indices, data", r_csgraph_from_dense, NULL, "CSR triple (indptr, indices, data) from a dense adjacency matrix (scipy.sparse.csgraph.csgraph_from_dense)."),
    ROUTINE("csgraph.connected_components", 2, "csgraph, directed=True, connection='weak'", "n_components, labels", r_connected_components, NULL, "Connected components of a graph, weak connectivity (scipy.sparse.csgraph.connected_components)."),
    ROUTINE("csgraph.shortest_path", 1, "csgraph, method='auto', directed=True, return_predecessors=False, unweighted=False, overwrite=False, indices=None", "dist_matrix", r_shortest_path, NULL, "All-pairs shortest path distance matrix (scipy.sparse.csgraph.shortest_path; dense, indices=None)."),
    ROUTINE("csgraph.dijkstra", 1, "csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False", "dist_matrix", r_dijkstra, NULL, "Dijkstra all-pairs shortest paths, non-negative weights (scipy.sparse.csgraph.dijkstra; indices=None)."),
    ROUTINE("csgraph.bellman_ford", 1, "csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False", "dist_matrix", r_bellman_ford, NULL, "Bellman-Ford all-pairs shortest paths, allows negative weights (scipy.sparse.csgraph.bellman_ford; indices=None)."),
    ROUTINE("csgraph.johnson", 1, "csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False", "dist_matrix", r_johnson, NULL, "Johnson all-pairs shortest paths (scipy.sparse.csgraph.johnson; indices=None)."),
    ROUTINE("csgraph.floyd_warshall", 1, "csgraph, directed=True, return_predecessors=False, unweighted=False, overwrite=False", "dist_matrix", r_floyd_warshall, NULL, "Floyd-Warshall all-pairs shortest paths (scipy.sparse.csgraph.floyd_warshall)."),
    ROUTINE("csgraph.laplacian", 1, "csgraph, normed=False", "L", r_laplacian, NULL, "Graph Laplacian, plain or symmetric-normalized (scipy.sparse.csgraph.laplacian)."),
    ROUTINE("csgraph.breadth_first_order", 2, "csgraph, i_start, directed=True, return_predecessors=True", "node_array, predecessors", r_breadth_first_order, NULL, "Breadth-first traversal order and predecessors (scipy.sparse.csgraph.breadth_first_order)."),
    ROUTINE("csgraph.depth_first_order", 2, "csgraph, i_start, directed=True, return_predecessors=True", "node_array, predecessors", r_depth_first_order, NULL, "Depth-first traversal order and predecessors (scipy.sparse.csgraph.depth_first_order)."),
    ROUTINE("csgraph.structural_rank", 1, "graph", "rank", r_structural_rank, NULL, "Structural rank of a graph's sparsity pattern (scipy.sparse.csgraph.structural_rank)."),
    ROUTINE("csgraph.minimum_spanning_tree", 1, "csgraph, overwrite=False", "mst", r_minimum_spanning_tree, NULL, "Minimum spanning tree of an undirected graph, returned dense (scipy.sparse.csgraph.minimum_spanning_tree)."),
    ROUTINE("csgraph.reconstruct_path", 1, "csgraph, predecessors, directed=True", "cstree", r_reconstruct_path, NULL, "Reconstruct the tree of a shortest-path predecessor list, returned dense (scipy.sparse.csgraph.reconstruct_path)."),
    ROUTINE("csgraph.construct_dist_matrix", 1, "graph, predecessors, directed=True", "dist_matrix", r_construct_dist_matrix, NULL, "Distance matrix from a predecessor tree (scipy.sparse.csgraph.construct_dist_matrix)."),
    ROUTINE("csgraph.min_weight_full_bipartite_matching", 2, "biadjacency_matrix, maximize=False", "row_ind, col_ind", r_min_weight_full_bipartite_matching, NULL, "Minimum-weight full bipartite matching of a square biadjacency (scipy.sparse.csgraph.min_weight_full_bipartite_matching)."),
    ROUTINE("csgraph.breadth_first_tree", 1, "csgraph, i_start, directed=True", "cstree", r_breadth_first_tree, NULL, "Breadth-first spanning tree, returned dense (scipy.sparse.csgraph.breadth_first_tree)."),
    ROUTINE("csgraph.depth_first_tree", 1, "csgraph, i_start, directed=True", "cstree", r_depth_first_tree, NULL, "Depth-first spanning tree, returned dense (scipy.sparse.csgraph.depth_first_tree)."),
};

const fn_table TSR_SCIPY_CSGRAPH_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
