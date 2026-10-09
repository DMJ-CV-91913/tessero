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
#include <string.h>

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

/* kmeans2(data, k, iter=10, thresh=1e-5, minit='matrix', missing='warn'): k-means with explicit initial centroids
   (minit='matrix' only; k is the (nc, nfeat) initial code book) (scipy.cluster.vq.kmeans2). Runs `iter` Lloyd
   iterations -- assign each observation to the nearest centroid, then recompute each centroid as the mean of its
   members (empty clusters keep their previous centroid, missing='warn') -- returning the final centroids and the
   labels from the last assignment (one step behind the centroids, exactly as scipy). */
static int r_kmeans2(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("kmeans2: data and k (the initial centroids, minit='matrix') must be 2-D arrays"); return TSR_EARG; }
    if (nargs > 4 && args[4].kind == 2 && args[4].str && strcmp(args[4].str, "matrix") != 0) { fn_set_error("kmeans2: only minit='matrix' (explicit initial centroids) is supported here"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], f = args[0].arr.shape[1], nc = args[1].arr.shape[0];
    if (args[1].arr.shape[1] != f) { fn_set_error("kmeans2: data and k must have the same number of features"); return TSR_EARG; }
    const int iter = (nargs > 2 && args[2].kind == 1) ? (int)args[2].num : 10;
    int64_t ld, lk;
    double *data = fn_arg_doubles(&args[0], &ld), *init = data ? fn_arg_doubles(&args[1], &lk) : NULL;
    double *cb = init ? (double *)malloc((size_t)(nc * f) * sizeof(double)) : NULL;
    double *sums = cb ? (double *)malloc((size_t)(nc * f) * sizeof(double)) : NULL;
    int64_t *counts = sums ? (int64_t *)malloc((size_t)nc * sizeof(int64_t)) : NULL;
    int64_t *label = counts ? (int64_t *)fn_result_array(&res[1], TSR_I64, 1, (int64_t[]){n}) : NULL;
    int rc = (!data || !init || !cb || !sums || !counts || !label) ? TSR_ENOMEM : TSR_OK;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i < nc * f; i++) cb[i] = init[i];
        for (int t = 0; t < iter; t++) {
            for (int64_t i = 0; i < n; i++) {                       /* assign to nearest centroid */
                int64_t best = 0; double bestd = -1.0;
                for (int64_t c = 0; c < nc; c++) {
                    double s = 0.0;
                    for (int64_t j = 0; j < f; j++) { const double d = data[i * f + j] - cb[c * f + j]; s += d * d; }
                    if (bestd < 0.0 || s < bestd) { bestd = s; best = c; }
                }
                label[i] = best;
            }
            for (int64_t c = 0; c < nc; c++) { counts[c] = 0; for (int64_t j = 0; j < f; j++) sums[c * f + j] = 0.0; }
            for (int64_t i = 0; i < n; i++) { const int64_t c = label[i]; counts[c]++; for (int64_t j = 0; j < f; j++) sums[c * f + j] += data[i * f + j]; }
            for (int64_t c = 0; c < nc; c++) if (counts[c] > 0) for (int64_t j = 0; j < f; j++) cb[c * f + j] = sums[c * f + j] / (double)counts[c];
        }
        double *cout = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){nc, f});
        if (!cout) rc = TSR_ENOMEM;
        else for (int64_t i = 0; i < nc * f; i++) cout[i] = cb[i];
    }
    free(cb); free(sums); free(counts);
    fn_free_doubles(data, ld); fn_free_doubles(init, lk);
    return rc;
}

/* kmeans(obs, k_or_guess, iter=20, thresh=1e-5): k-means from an explicit initial code book (guess must be a
   2-D array; the scalar-k RNG form is not supported) (scipy.cluster.vq.kmeans). Iterates assign -> recompute
   centroids until the mean distortion changes by less than thresh, dropping any empty clusters (the code book
   may shrink). Returns the final code book and the final mean distortion. */
static int r_kmeans(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("kmeans: obs and k_or_guess (an explicit code book) must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], f = args[0].arr.shape[1];
    int64_t nc = args[1].arr.shape[0];
    if (args[1].arr.shape[1] != f) { fn_set_error("kmeans: obs and k_or_guess must have the same number of features"); return TSR_EARG; }
    const double thresh = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1e-5;
    int64_t lo, lg;
    double *obs = fn_arg_doubles(&args[0], &lo), *g = obs ? fn_arg_doubles(&args[1], &lg) : NULL;
    double *cb = g ? (double *)malloc((size_t)(nc * f) * sizeof(double)) : NULL;
    double *sums = cb ? (double *)malloc((size_t)(nc * f) * sizeof(double)) : NULL;
    int64_t *counts = sums ? (int64_t *)malloc((size_t)nc * sizeof(int64_t)) : NULL;
    int64_t *label = counts ? (int64_t *)malloc((size_t)n * sizeof(int64_t)) : NULL;
    int rc = (!obs || !g || !cb || !sums || !counts || !label) ? TSR_ENOMEM : TSR_OK;
    double cur_avg = 0.0;
    if (rc == TSR_OK) {
        for (int64_t i = 0; i < nc * f; i++) cb[i] = g[i];
        double prev_avg = INFINITY, diff = INFINITY;
        while (diff > thresh && nc > 0) {
            double sumdist = 0.0;
            for (int64_t i = 0; i < n; i++) {
                int64_t best = 0; double bestd = -1.0;
                for (int64_t c = 0; c < nc; c++) {
                    double s = 0.0;
                    for (int64_t j = 0; j < f; j++) { const double d = obs[i * f + j] - cb[c * f + j]; s += d * d; }
                    if (bestd < 0.0 || s < bestd) { bestd = s; best = c; }
                }
                label[i] = best; sumdist += sqrt(bestd);
            }
            cur_avg = sumdist / (double)n;
            diff = fabs(prev_avg - cur_avg);
            prev_avg = cur_avg;
            for (int64_t c = 0; c < nc; c++) { counts[c] = 0; for (int64_t j = 0; j < f; j++) sums[c * f + j] = 0.0; }
            for (int64_t i = 0; i < n; i++) { const int64_t c = label[i]; counts[c]++; for (int64_t j = 0; j < f; j++) sums[c * f + j] += obs[i * f + j]; }
            int64_t w = 0;                                   /* compact non-empty centroids (drop empties) */
            for (int64_t c = 0; c < nc; c++) if (counts[c] > 0) { for (int64_t j = 0; j < f; j++) cb[w * f + j] = sums[c * f + j] / (double)counts[c]; w++; }
            nc = w;
        }
        double *cout = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){nc, f});
        if (!cout) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < nc * f; i++) cout[i] = cb[i]; fn_result_num(&res[1], cur_avg); }
    }
    free(cb); free(sums); free(counts); free(label);
    fn_free_doubles(obs, lo); fn_free_doubles(g, lg);
    return rc;
}

static const fn_def DEFS[] = {
    ROUTINE("cluster.whiten", 1, "obs", "out", r_whiten, NULL, "Normalize observations per feature by their population standard deviation (scipy.cluster.vq.whiten)."),
    ROUTINE("cluster.vq", 2, "obs, code_book", "code, dist", r_vq, NULL, "Assign each observation to the nearest code-book vector by Euclidean distance (scipy.cluster.vq.vq)."),
    ROUTINE("cluster.kmeans2", 2, "data, k, iter=10, thresh=1e-5, minit='matrix', missing='warn'", "centroid, label", r_kmeans2, NULL, "k-means clustering from explicit initial centroids (minit='matrix'); returns centroids and labels (scipy.cluster.vq.kmeans2)."),
    ROUTINE("cluster.kmeans", 2, "obs, k_or_guess, iter=20, thresh=1e-5", "codebook, distortion", r_kmeans, NULL, "k-means from an explicit initial code book; returns the code book and mean distortion (scipy.cluster.vq.kmeans)."),
};

const fn_table TSR_SCIPY_CLUSTER_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
