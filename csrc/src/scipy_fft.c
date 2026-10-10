/* scipy.fft real trigonometric transforms ("scipyfft." prefix -> Tessero\ScipyFft facade).
 *
 * dct / idct / dst / idst for a 1-D real array, types 2 and 3, norm='backward' (the SciPy default): the
 * plain O(N^2) direct summations, which reproduce scipy.fft exactly. Both backends read this table.
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* unnormalized DCT of type 2 or 3 (scipy's norm='backward'). */
static void fft_dctcore(const double *x, int64_t N, int type, double *y)
{
    if (type == 2) {
        for (int64_t k = 0; k < N; k++) { double s = 0.0; for (int64_t n = 0; n < N; n++) s += x[n] * cos(M_PI * (double)(2 * n + 1) * (double)k / (double)(2 * N)); y[k] = 2.0 * s; }
    } else { /* type 3 */
        for (int64_t k = 0; k < N; k++) { double s = x[0]; for (int64_t n = 1; n < N; n++) s += 2.0 * x[n] * cos(M_PI * (double)n * (double)(2 * k + 1) / (double)(2 * N)); y[k] = s; }
    }
}
/* unnormalized DST of type 2 or 3 (scipy's norm='backward'). */
static void fft_dstcore(const double *x, int64_t N, int type, double *y)
{
    if (type == 2) {
        for (int64_t k = 0; k < N; k++) { double s = 0.0; for (int64_t n = 0; n < N; n++) s += x[n] * sin(M_PI * (double)(2 * n + 1) * (double)(k + 1) / (double)(2 * N)); y[k] = 2.0 * s; }
    } else { /* type 3 */
        for (int64_t k = 0; k < N; k++) { double s = ((k & 1) ? -1.0 : 1.0) * x[N - 1]; for (int64_t n = 0; n < N - 1; n++) s += 2.0 * x[n] * sin(M_PI * (double)(n + 1) * (double)(2 * k + 1) / (double)(2 * N)); y[k] = s; }
    }
}

static int fft_type(const tsr_arg *args, int nargs)
{
    int t = 2;
    if (nargs > 1 && args[1].kind == 1) t = (int)((args[1].flags & 1) ? args[1].ival : (int64_t)args[1].num);
    return t;
}

/* dct/dst dispatch: forward = core(type); inverse = core(swapped type) / (2N). which: 0 cosine, 1 sine. */
static int fft_run(const tsr_arg *args, int nargs, tsr_result *res, int which, int inverse)
{
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("scipy.fft: input must be a 1-D array"); return TSR_EARG; }
    int type = fft_type(args, nargs);
    if (type != 2 && type != 3) { fn_set_error("scipy.fft: only type 2 and 3 are supported here"); return TSR_EARG; }
    int eff = inverse ? (type == 2 ? 3 : 2) : type;
    int64_t N, L; double *x = fn_arg_doubles(&args[0], &L); if (!x) return TSR_ENOMEM;
    N = args[0].arr.shape[0];
    int64_t sh[1] = {N};
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!y) { fn_free_doubles(x, L); return TSR_ENOMEM; }
    if (which == 0) fft_dctcore(x, N, eff, y); else fft_dstcore(x, N, eff, y);
    if (inverse) { double sc = 1.0 / (double)(2 * N); for (int64_t k = 0; k < N; k++) y[k] *= sc; }
    fn_free_doubles(x, L);
    return TSR_OK;
}

static int r_dct(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run(args, nargs, res, 0, 0); }
static int r_idct(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run(args, nargs, res, 0, 1); }
static int r_dst(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run(args, nargs, res, 1, 0); }
static int r_idst(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run(args, nargs, res, 1, 1); }

/* one in-place 1-D transform of a length-L vector buf (which: 0 cos, 1 sin; forward/inverse as above). */
static void fft_1d(double *buf, int64_t L, int which, int type, int inverse)
{
    int eff = inverse ? (type == 2 ? 3 : 2) : type;
    double *y = (double *)malloc((size_t)L * sizeof(double));
    if (!y) return;
    if (which == 0) fft_dctcore(buf, L, eff, y); else fft_dstcore(buf, L, eff, y);
    double sc = inverse ? 1.0 / (double)(2 * L) : 1.0;
    for (int64_t i = 0; i < L; i++) buf[i] = y[i] * sc;
    free(y);
}

/* separable n-D transform over a 2-D array: transform every row, then every column. */
static int fft_run_nd(const tsr_arg *args, int nargs, tsr_result *res, int which, int inverse)
{
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("scipy.fft: dctn/dstn input must be 2-D here"); return TSR_EARG; }
    int type = fft_type(args, nargs);
    if (type != 2 && type != 3) { fn_set_error("scipy.fft: only type 2 and 3 are supported here"); return TSR_EARG; }
    int64_t m = args[0].arr.shape[0], n = args[0].arr.shape[1], L;
    double *x = fn_arg_doubles(&args[0], &L); if (!x) return TSR_ENOMEM;
    int64_t sh[2] = {m, n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, sh);
    double *col = (double *)malloc((size_t)m * sizeof(double));
    if (!out || !col) { free(col); fn_free_doubles(x, L); return TSR_ENOMEM; }
    memcpy(out, x, (size_t)(m * n) * sizeof(double));
    for (int64_t i = 0; i < m; i++) fft_1d(&out[i * n], n, which, type, inverse);     /* rows */
    for (int64_t j = 0; j < n; j++) {                                                   /* columns */
        for (int64_t i = 0; i < m; i++) col[i] = out[i * n + j];
        fft_1d(col, m, which, type, inverse);
        for (int64_t i = 0; i < m; i++) out[i * n + j] = col[i];
    }
    free(col); fn_free_doubles(x, L);
    return TSR_OK;
}
static int r_dctn(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run_nd(args, nargs, res, 0, 0); }
static int r_idctn(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run_nd(args, nargs, res, 0, 1); }
static int r_dstn(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run_nd(args, nargs, res, 1, 0); }
static int r_idstn(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fft_run_nd(args, nargs, res, 1, 1); }

static const fn_def DEFS[] = {
    ROUTINE("scipyfft.dct", 1, "x, type=2, n=None, axis=-1, norm=None", "y", r_dct, NULL, "Discrete cosine transform, types 2/3, norm='backward' (scipy.fft.dct)."),
    ROUTINE("scipyfft.idct", 1, "x, type=2, n=None, axis=-1, norm=None", "y", r_idct, NULL, "Inverse discrete cosine transform, types 2/3, norm='backward' (scipy.fft.idct)."),
    ROUTINE("scipyfft.dst", 1, "x, type=2, n=None, axis=-1, norm=None", "y", r_dst, NULL, "Discrete sine transform, types 2/3, norm='backward' (scipy.fft.dst)."),
    ROUTINE("scipyfft.idst", 1, "x, type=2, n=None, axis=-1, norm=None", "y", r_idst, NULL, "Inverse discrete sine transform, types 2/3, norm='backward' (scipy.fft.idst)."),
    ROUTINE("scipyfft.dctn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_dctn, NULL, "n-D discrete cosine transform over a 2-D array, types 2/3 (scipy.fft.dctn)."),
    ROUTINE("scipyfft.idctn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_idctn, NULL, "n-D inverse discrete cosine transform over a 2-D array (scipy.fft.idctn)."),
    ROUTINE("scipyfft.dstn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_dstn, NULL, "n-D discrete sine transform over a 2-D array, types 2/3 (scipy.fft.dstn)."),
    ROUTINE("scipyfft.idstn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_idstn, NULL, "n-D inverse discrete sine transform over a 2-D array (scipy.fft.idstn)."),
};

const fn_table TSR_SCIPY_FFT_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
