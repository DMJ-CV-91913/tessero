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
    if (nargs > 1 && args[1].kind == 1) { const int64_t v = fn_num_i64(&args[1], 0); t = v >= 1 && v <= 4 ? (int)v : 0; }   /* others: refused */
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

/* ---- FFTLog fast Hankel transform (scipy.fft.fht / ifht / fhtoffset) ---- */
int tsr_rfft(int64_t n, int64_t rows, const double *in, double *out);
int tsr_irfft(int64_t n, int64_t rows, const double *in, double *out);
void tsr_special_poch(const void *, const double *, double *);
void tsr_loggamma_cpx(const double *in, double *out);

/* u_m = (kr)^{-i 2m pi/(n dlnr)} 2^q Gamma((mu+1+q+..)/2)/Gamma((mu+1-q-..)/2); see Hamilton (2000). */
static void fht_coeff(int64_t n, double dln, double mu, double lnkr, double q, int inverse, double *u)
{
    const double LN2 = 0.69314718055994530942;
    const int64_t h = n / 2 + 1;
    const double xp = (mu + 1.0 + q) / 2.0, xm = (mu + 1.0 - q) / 2.0;
    for (int64_t m = 0; m < h; m++) {
        const double y = M_PI * (double)m / ((double)n * dln);
        double in2[2], vo[2], uo[2];
        in2[0] = xm; in2[1] = y; tsr_loggamma_cpx(in2, vo);
        in2[0] = xp; in2[1] = y; tsr_loggamma_cpx(in2, uo);
        const double re = uo[0] - vo[0] + LN2 * q;
        const double im = uo[1] + vo[1] + 2.0 * (LN2 - lnkr) * y;
        const double er = exp(re);
        u[2 * m] = er * cos(im);
        u[2 * m + 1] = er * sin(im);
    }
    if (n % 2 == 0) u[2 * (h - 1) + 1] = 0.0;
    if (!isfinite(u[0]) || !isfinite(u[1])) {
        double pin[2] = { xm, xp - xm }, po[1];
        tsr_special_poch(NULL, pin, po);
        u[0] = pow(2.0, q) * po[0];
        u[1] = 0.0;
    }
    if (isinf(u[0]) && !inverse) { u[0] = 0.0; u[1] = 0.0; }
    else if (u[0] == 0.0 && u[1] == 0.0 && inverse) { u[0] = INFINITY; u[1] = 0.0; }
}

static int fht_q_rows(int64_t n, int64_t rows, const double *a, const double *u, int inverse, double *out)
{
    const int64_t h = n / 2 + 1;
    double *A = (double *)tsr_alloc(2 * h * (int64_t)sizeof(double));
    if (!A) return TSR_ENOMEM;
    for (int64_t r = 0; r < rows; r++) {
        if (tsr_rfft(n, 1, a + r * n, A) != TSR_OK) { tsr_free(A, 2 * h * (int64_t)sizeof(double)); return TSR_EARG; }
        for (int64_t m = 0; m < h; m++) {
            const double Ar = A[2 * m], Ai = A[2 * m + 1], ur = u[2 * m], ui = u[2 * m + 1];
            if (!inverse) { A[2 * m] = Ar * ur - Ai * ui; A[2 * m + 1] = Ar * ui + Ai * ur; }
            else { const double d = ur * ur + ui * ui; A[2 * m] = (Ar * ur - Ai * ui) / d; A[2 * m + 1] = (Ar * ui + Ai * ur) / d; }
        }
        double *orow = out + r * n;
        if (tsr_irfft(n, 1, A, orow) != TSR_OK) { tsr_free(A, 2 * h * (int64_t)sizeof(double)); return TSR_EARG; }
        for (int64_t i = 0; i < n / 2; i++) { const double t = orow[i]; orow[i] = orow[n - 1 - i]; orow[n - 1 - i] = t; }
    }
    tsr_free(A, 2 * h * (int64_t)sizeof(double));
    return TSR_OK;
}

static double fht_scalar(const tsr_arg *args, int nargs, int i, double def)
{
    if (nargs > i && args[i].kind == 1) return args[i].flags & 1 ? (double)args[i].ival : args[i].num;
    return def;
}

static int fht_run(const tsr_arg *args, int nargs, tsr_result *res, int inverse)
{
    if (nargs < 3 || args[0].kind != 3) { fn_set_error("fht: a must be an array, with dln and mu"); return TSR_EARG; }
    const tsr_array *A = &args[0].arr;
    /* scipy indexes a.shape[-1] (IndexError for a 0-d array); a 0-d array indexed shape[-1] here */
    if (A->ndim < 1) { fn_set_error("fht: a must have at least one dimension"); return TSR_EDIM; }
    const int64_t n = A->shape[A->ndim - 1];
    if (n < 1) { fn_set_error("fht: last axis must be non-empty"); return TSR_EARG; }
    int64_t na;
    double *a = fn_arg_doubles(&args[0], &na);
    if (!a) return TSR_ENOMEM;
    const int64_t rows = na / n;
    const double dln = fht_scalar(args, nargs, 1, 0.0), mu = fht_scalar(args, nargs, 2, 0.0);
    const double offset = fht_scalar(args, nargs, 3, 0.0), bias = fht_scalar(args, nargs, 4, 0.0);
    double *u = (double *)tsr_alloc(2 * (n / 2 + 1) * (int64_t)sizeof(double));
    double *out = (double *)fn_result_array(&res[0], TSR_F64, A->ndim, A->shape);
    int rc = TSR_OK;
    if (!u || !out) { rc = TSR_ENOMEM; goto done; }
    if (bias != 0.0) {                                   /* input bias */
        const double jc = (double)(n - 1) / 2.0;
        for (int64_t r = 0; r < rows; r++)
            for (int64_t j = 0; j < n; j++) {
                const double e = inverse ? exp(bias * (((double)j - jc) * dln + offset)) : exp(-bias * ((double)j - jc) * dln);
                a[r * n + j] *= e;
            }
    }
    fht_coeff(n, dln, mu, offset, bias, inverse, u);
    rc = fht_q_rows(n, rows, a, u, inverse, out);
    if (rc == TSR_OK && bias != 0.0) {                   /* output bias */
        const double jc = (double)(n - 1) / 2.0;
        for (int64_t r = 0; r < rows; r++)
            for (int64_t j = 0; j < n; j++) {
                const double e = inverse ? exp(bias * ((double)j - jc) * dln) : exp(-bias * (((double)j - jc) * dln + offset));
                out[r * n + j] *= e;
            }
    }
done:
    if (u) tsr_free(u, 2 * (n / 2 + 1) * (int64_t)sizeof(double));
    fn_free_doubles(a, na);
    return rc;
}

static int r_fht(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fht_run(args, nargs, res, 0); }
static int r_ifht(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return fht_run(args, nargs, res, 1); }

static int r_fhtoffset(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const double dln = fht_scalar(args, nargs, 0, 0.0), mu = fht_scalar(args, nargs, 1, 0.0);
    const double initial = fht_scalar(args, nargs, 2, 0.0), bias = fht_scalar(args, nargs, 3, 0.0);
    const double LN2 = 0.69314718055994530942, q = bias, lnkr = initial;
    const double xp = (mu + 1.0 + q) / 2.0, xm = (mu + 1.0 - q) / 2.0, y = M_PI / (2.0 * dln);
    double in2[2], zp[2], zm[2];
    in2[0] = xp; in2[1] = y; tsr_loggamma_cpx(in2, zp);
    in2[0] = xm; in2[1] = y; tsr_loggamma_cpx(in2, zm);
    const double arg = (LN2 - lnkr) / dln + (zp[1] + zm[1]) / M_PI;
    fn_result_num(&res[0], lnkr + (arg - round(arg)) * dln);
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
    ROUTINE("scipyfft.fht", 3, "a, dln, mu, offset=0, bias=0", "A", r_fht, NULL, "Fast Hankel transform via the FFTLog algorithm (scipy.fft.fht)."),
    ROUTINE("scipyfft.ifht", 3, "A, dln, mu, offset=0, bias=0", "a", r_ifht, NULL, "Inverse fast Hankel transform (scipy.fft.ifht)."),
    ROUTINE("scipyfft.fhtoffset", 2, "dln, mu, initial=0, bias=0", "offset", r_fhtoffset, NULL, "Optimal low-ringing offset for fht (scipy.fft.fhtoffset)."),
    ROUTINE("scipyfft.idct", 1, "x, type=2, n=None, axis=-1, norm=None", "y", r_idct, NULL, "Inverse discrete cosine transform, types 2/3, norm='backward' (scipy.fft.idct)."),
    ROUTINE("scipyfft.dst", 1, "x, type=2, n=None, axis=-1, norm=None", "y", r_dst, NULL, "Discrete sine transform, types 2/3, norm='backward' (scipy.fft.dst)."),
    ROUTINE("scipyfft.idst", 1, "x, type=2, n=None, axis=-1, norm=None", "y", r_idst, NULL, "Inverse discrete sine transform, types 2/3, norm='backward' (scipy.fft.idst)."),
    ROUTINE("scipyfft.dctn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_dctn, NULL, "n-D discrete cosine transform over a 2-D array, types 2/3 (scipy.fft.dctn)."),
    ROUTINE("scipyfft.idctn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_idctn, NULL, "n-D inverse discrete cosine transform over a 2-D array (scipy.fft.idctn)."),
    ROUTINE("scipyfft.dstn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_dstn, NULL, "n-D discrete sine transform over a 2-D array, types 2/3 (scipy.fft.dstn)."),
    ROUTINE("scipyfft.idstn", 1, "x, type=2, s=None, axes=None, norm=None", "y", r_idstn, NULL, "n-D inverse discrete sine transform over a 2-D array (scipy.fft.idstn)."),
};

const fn_table TSR_SCIPY_FFT_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
