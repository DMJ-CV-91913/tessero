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

static const fn_def DEFS[] = {
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
    ROUTINE("csgraph.breadth_first_tree", 1, "csgraph, i_start, directed=True", "cstree", r_breadth_first_tree, NULL, "Breadth-first spanning tree, returned dense (scipy.sparse.csgraph.breadth_first_tree)."),
    ROUTINE("csgraph.depth_first_tree", 1, "csgraph, i_start, directed=True", "cstree", r_depth_first_tree, NULL, "Depth-first spanning tree, returned dense (scipy.sparse.csgraph.depth_first_tree)."),
};

const fn_table TSR_SCIPY_CSGRAPH_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
