/* The registry's module tables (ADR 0011). Bindings look functions up by name, so the order only fixes ids. */
#include "fn.h"

#ifndef TSR_NO_REGISTRY

extern const fn_table TSR_GEN_SPECIAL_TABLE;
extern const fn_table TSR_NP_STATS_TABLE;
extern const fn_table TSR_NP_SETS_TABLE;
extern const fn_table TSR_NP_RANDOM_TABLE;
extern const fn_table TSR_NP_SHAPE_TABLE;
extern const fn_table TSR_NP_NUMERIC_TABLE;
extern const fn_table TSR_NP_LINALG_TABLE;
extern const fn_table TSR_SCIPY_LINALG_TABLE;
extern const fn_table TSR_SPECIAL_COMPLEX_TABLE;   /* C++ (cxx/special_complex.cpp) */
extern const fn_table TSR_SPATIAL_DISTANCE_TABLE;  /* scipy.spatial.distance (src/spatial_distance.c) */
extern const fn_table TSR_SCIPY_SPARSE_TABLE;      /* scipy.sparse (src/scipy_sparse.c) */
extern const fn_table TSR_SCIPY_SIGNAL_TABLE;      /* scipy.signal (src/scipy_signal.c) */
extern const fn_table TSR_SCIPY_NDIMAGE_TABLE;     /* scipy.ndimage (src/scipy_ndimage.c) */
extern const fn_table TSR_SCIPY_CSGRAPH_TABLE;     /* scipy.sparse.csgraph (src/scipy_csgraph.c) */
extern const fn_table TSR_NP_IO_TABLE;             /* numpy.loadtxt (src/np_io.c) */
extern const fn_table TSR_NP_POLY_TABLE;           /* numpy.polyfit/polyval/roots (src/np_poly.c) */
/* scipy.stats functions (cxx/stats_fn_*.cpp). Weak while the modules are being written: a missing one is
   simply absent from the registry. */
#if defined(__GNUC__)
#define TSR_WEAK __attribute__((weak))
#else
#define TSR_WEAK
#endif
extern const fn_table TSR_STATS_FN_DESC_TABLE TSR_WEAK;
extern const fn_table TSR_STATS_FN_CORR_TABLE TSR_WEAK;
extern const fn_table TSR_STATS_FN_TESTS_TABLE TSR_WEAK;
extern const fn_table TSR_STATS_FN_TESTS2_TABLE TSR_WEAK;

const fn_table *const TSR_FN_TABLES[] = {&TSR_GEN_SPECIAL_TABLE, &TSR_NP_STATS_TABLE, &TSR_NP_SETS_TABLE,
                                             &TSR_NP_RANDOM_TABLE, &TSR_NP_SHAPE_TABLE, &TSR_NP_NUMERIC_TABLE, &TSR_NP_LINALG_TABLE, &TSR_SCIPY_LINALG_TABLE, &TSR_SPECIAL_COMPLEX_TABLE, &TSR_SPATIAL_DISTANCE_TABLE, &TSR_SCIPY_SPARSE_TABLE, &TSR_SCIPY_SIGNAL_TABLE, &TSR_SCIPY_NDIMAGE_TABLE, &TSR_SCIPY_CSGRAPH_TABLE, &TSR_NP_IO_TABLE, &TSR_NP_POLY_TABLE, &TSR_STATS_FN_DESC_TABLE, &TSR_STATS_FN_CORR_TABLE,
                                             &TSR_STATS_FN_TESTS_TABLE, &TSR_STATS_FN_TESTS2_TABLE};
const int TSR_FN_NTABLES = (int)(sizeof TSR_FN_TABLES / sizeof TSR_FN_TABLES[0]);

extern fn_dist_table TSR_STATS_DISTS;     /* filled by the distributions' registrars (cxx/stats_dist.cpp) */
const fn_dist_table *const TSR_DIST_TABLES[] = {&TSR_STATS_DISTS};
const int TSR_DIST_NTABLES = 1;

#else   /* fuzz builds: the parsers only, no C++ */

static const fn_table NO_FNS = {NULL, 0};
const fn_table *const TSR_FN_TABLES[] = {&NO_FNS};
const int TSR_FN_NTABLES = 1;
static const fn_dist_table NO_DISTS = {NULL, 0};
const fn_dist_table *const TSR_DIST_TABLES[] = {&NO_DISTS};
const int TSR_DIST_NTABLES = 1;
void dist_method(const fn_dist *d, int m, const double *in, double *out) { (void)d; (void)m; (void)in; (void)out; }
const char *dist_take_error(void) { return NULL; }
/* the C-only (fuzz) build has no C++ code, so nothing can throw: direct calls */
int fn_call_core(fn_core f, const void *ctx, const double *const *x, const int64_t *n, const double *p,
                 double *const *out, const int64_t *out_n, void *scratch, unsigned flags)
{
    return f(ctx, x, n, p, out, out_n, scratch, flags);
}
int fn_call_routine(fn_routine f, const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    return f(ctx, args, nargs, res, nres);
}
int dist_rvs_array(const fn_dist *d, void *b, int64_t n, const double *const *p, double *o) { (void)d; (void)b; (void)n; (void)p; (void)o; return -1; }
int tsr_random_run(const fn_def *d, uint64_t *state, const double *params, int32_t ndim, const int64_t *shape,
                   int nop, void **data, const int64_t *strides, int out_dtype)
{
    (void)d; (void)state; (void)params; (void)ndim; (void)shape; (void)nop; (void)data; (void)strides; (void)out_dtype;
    return -1;
}

#endif
