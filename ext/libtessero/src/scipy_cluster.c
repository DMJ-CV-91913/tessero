/* scipy.cluster.vq ("cluster." prefix -> Tessero\Cluster facade).
 *
 *   cluster.whiten(obs)            per-feature normalization by population std (scipy.cluster.vq.whiten)
 *   cluster.vq(obs, code_book)     nearest code-book vector per observation (scipy.cluster.vq.vq)
 *
 * Deterministic arithmetic over the observation matrix; both backends read this table at runtime.
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>

/* whiten(obs): divide each column of the (nobs, nfeat) matrix by its population standard deviation (ddof=0);
   columns whose std is zero are left unchanged (std treated as 1), matching scipy.cluster.vq.whiten. */
static int r_whiten(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("whiten: obs must be a 2-D array"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], f = args[0].arr.shape[1];
    int64_t lo; double *obs = fn_arg_doubles(&args[0], &lo);
    double *out = obs ? (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, f}) : NULL;
    double *sd = out ? (double *)calloc((size_t)(f ? f : 1), sizeof(double)) : NULL;
    int rc = (!obs || !out || !sd) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        for (int64_t c = 0; c < f; c++) {
            double mean = 0.0;
            for (int64_t i = 0; i < n; i++) mean += obs[i * f + c];
            mean /= (double)n;
            double v = 0.0;
            for (int64_t i = 0; i < n; i++) { const double d = obs[i * f + c] - mean; v += d * d; }
            sd[c] = sqrt(v / (double)n);
            if (sd[c] == 0.0) sd[c] = 1.0;
        }
        for (int64_t i = 0; i < n; i++)
            for (int64_t c = 0; c < f; c++) out[i * f + c] = obs[i * f + c] / sd[c];
    }
    free(sd); fn_free_doubles(obs, lo);
    return rc;
}

/* vq(obs, code_book): assign each (nobs, nfeat) observation to the nearest code-book row by Euclidean distance;
   returns the index array (int64) and the distances (scipy.cluster.vq.vq). Ties keep the first (lowest) index. */
static int r_vq(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("vq: obs and code_book must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], f = args[0].arr.shape[1], m = args[1].arr.shape[0];
    if (args[1].arr.shape[1] != f) { fn_set_error("vq: obs and code_book must have the same number of features"); return TSR_EARG; }
    int64_t lo, lc;
    double *obs = fn_arg_doubles(&args[0], &lo), *cb = obs ? fn_arg_doubles(&args[1], &lc) : NULL;
    int64_t *code = cb ? (int64_t *)fn_result_array(&res[0], TSR_I64, 1, (int64_t[]){n}) : NULL;
    double *dist = code ? (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){n}) : NULL;
    int rc = (!obs || !cb || !code || !dist) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i < n; i++) {
            int64_t best = 0; double bestd = -1.0;
            for (int64_t j = 0; j < m; j++) {
                double s = 0.0;
                for (int64_t c = 0; c < f; c++) { const double d = obs[i * f + c] - cb[j * f + c]; s += d * d; }
                const double dd = sqrt(s);
                if (bestd < 0.0 || dd < bestd) { bestd = dd; best = j; }
            }
            code[i] = best; dist[i] = bestd;
        }
    }
    fn_free_doubles(obs, lo); fn_free_doubles(cb, lc);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("cluster.whiten", 1, "obs", "out", r_whiten, NULL, "Normalize observations per feature by their population standard deviation (scipy.cluster.vq.whiten)."),
    ROUTINE("cluster.vq", 2, "obs, code_book", "code, dist", r_vq, NULL, "Assign each observation to the nearest code-book vector by Euclidean distance (scipy.cluster.vq.vq)."),
};

const fn_table TSR_SCIPY_CLUSTER_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
