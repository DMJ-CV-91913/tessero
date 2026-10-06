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

static const fn_def DEFS[] = {
    ROUTINE("csgraph.connected_components", 2, "csgraph, directed=True, connection='weak'", "n_components, labels", r_connected_components, NULL, "Connected components of a graph, weak connectivity (scipy.sparse.csgraph.connected_components)."),
    ROUTINE("csgraph.shortest_path", 1, "csgraph, method='auto', directed=True, return_predecessors=False, unweighted=False, overwrite=False, indices=None", "dist_matrix", r_shortest_path, NULL, "All-pairs shortest path distance matrix (scipy.sparse.csgraph.shortest_path; dense, indices=None)."),
    ROUTINE("csgraph.dijkstra", 1, "csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False", "dist_matrix", r_dijkstra, NULL, "Dijkstra all-pairs shortest paths, non-negative weights (scipy.sparse.csgraph.dijkstra; indices=None)."),
    ROUTINE("csgraph.bellman_ford", 1, "csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False", "dist_matrix", r_bellman_ford, NULL, "Bellman-Ford all-pairs shortest paths, allows negative weights (scipy.sparse.csgraph.bellman_ford; indices=None)."),
    ROUTINE("csgraph.johnson", 1, "csgraph, directed=True, indices=None, return_predecessors=False, unweighted=False", "dist_matrix", r_johnson, NULL, "Johnson all-pairs shortest paths (scipy.sparse.csgraph.johnson; indices=None)."),
    ROUTINE("csgraph.floyd_warshall", 1, "csgraph, directed=True, return_predecessors=False, unweighted=False, overwrite=False", "dist_matrix", r_floyd_warshall, NULL, "Floyd-Warshall all-pairs shortest paths (scipy.sparse.csgraph.floyd_warshall)."),
};

const fn_table TSR_SCIPY_CSGRAPH_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
