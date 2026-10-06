/* scipy.signal: 1-D convolve and lfilter (real float64, direct computation), plus the closed-form
 * scipy.signal.windows (the "windows." prefix -> Tessero\SignalWindows facade).
 *
 *   convolve(a, v, mode='full')  full/same/valid convolution of two 1-D sequences
 *   lfilter(b, a, x)             IIR/FIR filter via the difference equation, zero initial state
 *   windows.<name>(M, ..., sym=True)   a length-M window (hann, hamming, blackman, boxcar, triang, ...)
 */
#include "fn.h"

#include <complex.h>
#include <float.h>
#include <lapacke.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

enum { CONV_FULL, CONV_SAME, CONV_VALID };

static int gesolve(int n, double *A, double *b);   /* small dense solver, defined below */

static int parse_mode(const tsr_arg *a, int *mode)
{
    if (!a || a->kind == 0) { *mode = CONV_FULL; return TSR_OK; }
    if (a->kind != 2 || !a->str) { fn_set_error("convolve: mode must be a string"); return TSR_EARG; }
    if (strcmp(a->str, "full") == 0) { *mode = CONV_FULL; return TSR_OK; }
    if (strcmp(a->str, "same") == 0) { *mode = CONV_SAME; return TSR_OK; }
    if (strcmp(a->str, "valid") == 0) { *mode = CONV_VALID; return TSR_OK; }
    fn_set_error("convolve: mode '%s' is not supported", a->str);
    return TSR_EARG;
}

/* convolve(a, v, mode='full'): 1-D convolution */
static int r_convolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("convolve: a and v must be 1-D arrays"); return TSR_EARG; }
    int mode, rc; if ((rc = parse_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    const int64_t M = args[0].arr.shape[0], N = args[1].arr.shape[0];
    int64_t ta, tv; double *a = fn_arg_doubles(&args[0], &ta); if (!a) return TSR_ENOMEM;
    double *v = fn_arg_doubles(&args[1], &tv); if (!v) { fn_free_doubles(a, ta); return TSR_ENOMEM; }
    const int64_t L = (M > 0 && N > 0) ? M + N - 1 : 0;
    double *full = (double *)calloc((size_t)(L > 0 ? L : 1), sizeof(double));
    if (!full) { fn_free_doubles(a, ta); fn_free_doubles(v, tv); return TSR_ENOMEM; }
    for (int64_t j = 0; j < M; j++)
        for (int64_t k = 0; k < N; k++) full[j + k] += a[j] * v[k];
    int64_t start, len;
    if (mode == CONV_FULL) { start = 0; len = L; }
    else if (mode == CONV_SAME) { start = (N - 1) / 2; len = M; }
    else { const int64_t mn = M < N ? M : N, mx = M > N ? M : N; start = mn - 1; len = mx - mn + 1; }   /* valid */
    if (len < 0) len = 0;
    int64_t osh[1] = {len};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, osh);
    if (!out) { free(full); fn_free_doubles(a, ta); fn_free_doubles(v, tv); return TSR_ENOMEM; }
    for (int64_t i = 0; i < len; i++) out[i] = full[start + i];
    free(full);
    fn_free_doubles(a, ta); fn_free_doubles(v, tv);
    return TSR_OK;
}

/* lfilter(b, a, x): y[n] = (sum_k b[k] x[n-k] - sum_{k>=1} a[k] y[n-k]) / a[0], zero initial conditions */
static int r_lfilter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1 ||
        args[2].kind != 3 || args[2].arr.ndim != 1) { fn_set_error("lfilter: b, a and x must be 1-D arrays"); return TSR_EARG; }
    const int64_t nb = args[0].arr.shape[0], na = args[1].arr.shape[0], L = args[2].arr.shape[0];
    int64_t tb, taa, tx;
    double *b = fn_arg_doubles(&args[0], &tb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &taa); if (!a) { fn_free_doubles(b, tb); return TSR_ENOMEM; }
    double *x = fn_arg_doubles(&args[2], &tx); if (!x) { fn_free_doubles(b, tb); fn_free_doubles(a, taa); return TSR_ENOMEM; }
    if (na < 1 || a[0] == 0.0) { fn_free_doubles(b, tb); fn_free_doubles(a, taa); fn_free_doubles(x, tx); fn_set_error("lfilter: a[0] must be nonzero"); return TSR_EARG; }
    int64_t osh[1] = {L};
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, osh);
    if (!y) { fn_free_doubles(b, tb); fn_free_doubles(a, taa); fn_free_doubles(x, tx); return TSR_ENOMEM; }
    const double a0 = a[0];
    for (int64_t n = 0; n < L; n++) {
        double acc = 0.0;
        for (int64_t k = 0; k < nb; k++) if (n - k >= 0) acc += b[k] * x[n - k];
        for (int64_t k = 1; k < na; k++) if (n - k >= 0) acc -= a[k] * y[n - k];
        y[n] = acc / a0;
    }
    fn_free_doubles(b, tb); fn_free_doubles(a, taa); fn_free_doubles(x, tx);
    return TSR_OK;
}

/* freqz(b, a, worN=512, whole=False): the frequency response H(e^jw) of a digital filter at worN points
   w = [0, lastpoint) (lastpoint = 2*pi if whole else pi). Returns (w, h); h is complex (scipy.signal.freqz). */
static int r_freqz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("freqz: b and a must be 1-D arrays"); return TSR_EARG; }
    const int64_t N = (nargs > 2 && args[2].kind == 1) ? (int64_t)args[2].num : 512;
    const int whole = (nargs > 3 && (args[3].kind == 1 || args[3].kind == 4)) ? args[3].num != 0 : 0;
    if (N <= 0) { fn_set_error("freqz: worN must be positive"); return TSR_EARG; }
    const int64_t nb = args[0].arr.shape[0], na = args[1].arr.shape[0];
    int64_t tb, ta; double *b = fn_arg_doubles(&args[0], &tb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &ta); if (!a) { fn_free_doubles(b, tb); return TSR_ENOMEM; }
    int64_t sh[1] = {N};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    double *h = w ? (double *)fn_result_array(&res[1], TSR_C128, 1, sh) : NULL;   /* interleaved re, im */
    if (!w || !h) { fn_free_doubles(b, tb); fn_free_doubles(a, ta); return TSR_ENOMEM; }
    const double last = whole ? 2.0 * M_PI : M_PI;
    const double step = last / (double)N;                    /* match numpy.linspace(0, last, N, endpoint=False): w[j] = j*step */
    for (int64_t j = 0; j < N; j++) {
        const double wj = (double)j * step;
        w[j] = wj;
        const double complex zm1 = cos(wj) - I * sin(wj);    /* = exp(-i*wj), matching numpy's cos/sin complex exp */
        double complex num = b[nb - 1], den = a[na - 1];     /* Horner, as numpy.polynomial.polyval (matches scipy.freqz) */
        for (int64_t k = nb - 2; k >= 0; k--) num = b[k] + num * zm1;
        for (int64_t k = na - 2; k >= 0; k--) den = a[k] + den * zm1;
        /* complex division via Smith's method with reciprocal-multiply, matching numpy's npy_cdivide */
        const double nr = creal(num), ni = cimag(num), dr = creal(den), di = cimag(den);
        double hr, hi;
        if (fabs(dr) >= fabs(di)) { const double rat = di / dr, scl = 1.0 / (dr + di * rat); hr = (nr + ni * rat) * scl; hi = (ni - nr * rat) * scl; }
        else { const double rat = dr / di, scl = 1.0 / (dr * rat + di); hr = (nr * rat + ni) * scl; hi = (ni * rat - nr) * scl; }
        h[2 * j] = hr; h[2 * j + 1] = hi;
    }
    fn_free_doubles(b, tb); fn_free_doubles(a, ta);
    return TSR_OK;
}

/* ------------------------------------------------------------------ IIR filter design (zpk/ba) ----- */

/* polynomial coefficients (highest degree first) from complex roots, by sequential multiplication (numpy.poly). */
static void cpoly(const double complex *roots, int nr, double complex *out)
{
    for (int i = 0; i <= nr; i++) out[i] = 0;
    out[0] = 1.0;
    for (int i = 0; i < nr; i++)
        for (int j = i + 1; j >= 1; j--) out[j] = out[j] - roots[i] * out[j - 1];
}

/* zpk -> (b, a): b = k*poly(z), a = poly(p); the imaginary parts cancel for conjugate-symmetric roots
   (numpy.poly takes the real part), so we emit real coefficients. Caller allocates b[nz+1], a[np_+1]. */
static void zpk2tf_real(const double complex *z, int nz, const double complex *p, int np_, double k, double *b, double *a)
{
    double complex *cb = (double complex *)malloc((size_t)(nz + 1) * sizeof(double complex));
    double complex *ca = (double complex *)malloc((size_t)(np_ + 1) * sizeof(double complex));
    cpoly(z, nz, cb);
    cpoly(p, np_, ca);
    for (int i = 0; i <= nz; i++) b[i] = k * creal(cb[i]);
    for (int i = 0; i <= np_; i++) a[i] = creal(ca[i]);
    free(cb); free(ca);
}

/* read a 1-D array argument (real or complex128) into a freshly-allocated double complex buffer */
static double complex *read_carr(const tsr_arg *arg, int64_t *n)
{
    if (arg->kind != 3) { *n = 0; return NULL; }
    const tsr_array *x = &arg->arr;
    int64_t sz = 1; for (int d = 0; d < x->ndim; d++) sz *= x->shape[d];
    *n = sz;
    double complex *out = (double complex *)malloc((size_t)(sz > 0 ? sz : 1) * sizeof(double complex));
    if (!out) return NULL;
    const int64_t stride = x->ndim ? x->strides[0] : 0;
    for (int64_t i = 0; i < sz; i++) {
        const char *pp = (const char *)x->data + x->offset + i * stride;
        if (x->dtype == TSR_C128) out[i] = *(const double *)pp + I * (*(const double *)(pp + 8));
        else if (x->dtype == TSR_F64) out[i] = *(const double *)pp;
        else if (x->dtype == TSR_F32) out[i] = *(const float *)pp;
        else if (x->dtype == TSR_I64) out[i] = (double)*(const int64_t *)pp;
        else if (x->dtype == TSR_I32) out[i] = (double)*(const int32_t *)pp;
        else out[i] = 0;
    }
    return out;
}

/* zpk2tf(z, p, k): transfer-function coefficients from zeros, poles and gain (scipy.signal.zpk2tf). */
static int r_zpk2tf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nz, np_;
    double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    double *b = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nz + 1});
    double *a = b ? (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){np_ + 1}) : NULL;
    if (!b || !a) { free(z); free(p); return TSR_ENOMEM; }
    zpk2tf_real(z, (int)nz, p, (int)np_, k, b, a);
    free(z); free(p);
    return TSR_OK;
}

/* butter(N, Wn, btype='low', output='ba'): digital Butterworth filter (scipy.signal.butter); lowpass/highpass,
   output 'ba'. Analog prototype -> frequency transform -> bilinear -> transfer function, matching scipy. */
static int r_butter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const int64_t N = (args[0].kind == 1) ? (int64_t)args[0].num : 0;
    if (N < 1 || N > 64) { fn_set_error("butter: order N must be between 1 and 64"); return TSR_EARG; }
    if (!(args[1].kind == 1)) { fn_set_error("butter: only a scalar Wn (lowpass/highpass) is supported"); return TSR_EARG; }
    const double Wn = args[1].num;
    int hp = 0;
    if (nargs > 2 && args[2].kind == 2 && args[2].str) {
        if (strcmp(args[2].str, "high") == 0 || strcmp(args[2].str, "highpass") == 0) hp = 1;
        else if (!(strcmp(args[2].str, "low") == 0 || strcmp(args[2].str, "lowpass") == 0)) { fn_set_error("butter: only lowpass/highpass are supported"); return TSR_EARG; }
    }
    /* buttap: analog prototype poles on the unit circle, no zeros, gain 1 */
    double complex *p = (double complex *)malloc((size_t)N * sizeof(double complex));
    for (int64_t i = 0; i < N; i++) { const double m = (double)(-N + 1 + 2 * i); p[i] = -cexp(I * M_PI * m / (2.0 * (double)N)); }
    double k = 1.0;
    const double fs = 2.0, warped = 2.0 * fs * tan(M_PI * Wn / fs);
    double complex *z = (double complex *)malloc((size_t)(N + 1) * sizeof(double complex));   /* room for appended zeros */
    int nz = 0;
    if (!hp) {                                               /* lp2lp: scale poles, no zeros */
        for (int64_t i = 0; i < N; i++) p[i] *= warped;
        for (int64_t i = 0; i < N; i++) k *= warped;         /* k *= warped**degree, degree=N */
    } else {                                                 /* lp2hp: poles -> warped/p, N zeros at origin */
        double complex pprod = 1.0;
        for (int64_t i = 0; i < N; i++) pprod *= -p[i];
        k *= creal(1.0 / pprod);                             /* k * real(prod(-z)/prod(-p)), prod(-z)=1 */
        for (int64_t i = 0; i < N; i++) p[i] = warped / p[i];
        for (int64_t i = 0; i < N; i++) z[i] = 0.0;
        nz = (int)N;
    }
    /* bilinear_zpk(fs=2): fs2=4; Mobius transform; append (degree) zeros at -1; scale gain */
    const double fs2 = 2.0 * fs;
    double complex num = 1.0, den = 1.0;                     /* prod(fs2 - z)/prod(fs2 - p) using pre-bilinear z,p */
    for (int i = 0; i < nz; i++) num *= (fs2 - z[i]);
    for (int64_t i = 0; i < N; i++) den *= (fs2 - p[i]);
    k *= creal(num / den);
    for (int i = 0; i < nz; i++) z[i] = (fs2 + z[i]) / (fs2 - z[i]);
    for (int64_t i = 0; i < N; i++) p[i] = (fs2 + p[i]) / (fs2 - p[i]);
    const int degree = (int)N - nz;
    for (int i = 0; i < degree; i++) z[nz + i] = -1.0;       /* zeros at -1 (Nyquist) */
    nz += degree;
    /* zpk2tf, emitted as a (b, a) 2-sequence */
    double *bb = (double *)malloc((size_t)(nz + 1) * sizeof(double));
    double *aa = (double *)malloc((size_t)(N + 1) * sizeof(double));
    int rc = TSR_OK;
    if (!bb || !aa) { rc = TSR_ENOMEM; goto done; }
    zpk2tf_real(z, nz, p, (int)N, k, bb, aa);
    {
        double *ob = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nz + 1});
        double *oa = ob ? (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){N + 1}) : NULL;
        if (!ob || !oa) { rc = TSR_ENOMEM; goto done; }
        for (int i = 0; i <= nz; i++) ob[i] = bb[i];
        for (int64_t i = 0; i <= N; i++) oa[i] = aa[i];
    }
done:
    free(p); free(z); free(bb); free(aa);
    return rc;
}

/* Common digital IIR pipeline from an analog lowpass prototype (poles p[0..N-1], prototype zeros z0[0..nz0-1],
   gain k): frequency transform (lp2lp / lp2hp) with pre-warping, bilinear (fs=2), then zpk2tf -> (b, a).
   Takes ownership of p (frees it); z0 is read-only (may be NULL when nz0==0). Emits (b, a) to res[0], res[1]. */
static int iir_lp_proto_to_ba_z(double complex *z0, int nz0, double complex *p, int N, double k, int hp, double Wn, tsr_result *res)
{
    const double fs = 2.0, warped = 2.0 * fs * tan(M_PI * Wn / fs);
    const int degree = N - nz0;                              /* len(p) - len(z) of the prototype */
    double complex *z = (double complex *)malloc((size_t)(N + 1) * sizeof(double complex));
    int nz = 0, rc = TSR_OK;
    double *bb = NULL, *aa = NULL;
    if (!z) { rc = TSR_ENOMEM; goto done; }
    if (!hp) {                                               /* lp2lp: scale zeros and poles by warped */
        for (int i = 0; i < nz0; i++) z[i] = warped * z0[i];
        for (int i = 0; i < N; i++) p[i] *= warped;
        for (int i = 0; i < degree; i++) k *= warped;        /* k *= warped**degree */
        nz = nz0;
    } else {                                                 /* lp2hp: zeros/poles -> warped/root, append `degree` zeros at origin */
        double complex zp = 1.0, pp = 1.0;
        for (int i = 0; i < nz0; i++) zp *= -z0[i];
        for (int i = 0; i < N; i++) pp *= -p[i];
        k *= creal(zp / pp);
        for (int i = 0; i < nz0; i++) z[i] = warped / z0[i];
        for (int i = 0; i < N; i++) p[i] = warped / p[i];
        for (int i = 0; i < degree; i++) z[nz0 + i] = 0.0;
        nz = nz0 + degree;
    }
    const double fs2 = 2.0 * fs;
    double complex num = 1.0, den = 1.0;
    for (int i = 0; i < nz; i++) num *= (fs2 - z[i]);
    for (int i = 0; i < N; i++) den *= (fs2 - p[i]);
    k *= creal(num / den);
    for (int i = 0; i < nz; i++) z[i] = (fs2 + z[i]) / (fs2 - z[i]);
    for (int i = 0; i < N; i++) p[i] = (fs2 + p[i]) / (fs2 - p[i]);
    const int bdeg = N - nz;                                 /* bilinear appends this many zeros at -1 */
    for (int i = 0; i < bdeg; i++) z[nz + i] = -1.0;
    nz += bdeg;
    bb = (double *)malloc((size_t)(nz + 1) * sizeof(double));
    aa = (double *)malloc((size_t)(N + 1) * sizeof(double));
    if (!bb || !aa) { rc = TSR_ENOMEM; goto done; }
    zpk2tf_real(z, nz, p, N, k, bb, aa);
    {
        double *ob = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nz + 1});
        double *oa = ob ? (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){N + 1}) : NULL;
        if (!ob || !oa) { rc = TSR_ENOMEM; goto done; }
        for (int i = 0; i <= nz; i++) ob[i] = bb[i];
        for (int i = 0; i <= N; i++) oa[i] = aa[i];
    }
done:
    free(p); free(z); free(bb); free(aa);
    return rc;
}

/* cheby1(N, rp, Wn, btype='low'): Chebyshev type I IIR filter, digital lowpass/highpass, output 'ba'
   (scipy.signal.cheby1). Analog prototype cheb1ap (equiripple passband), then the shared digital pipeline. */
static int r_cheby1(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const int N = (args[0].kind == 1) ? (int)args[0].num : 0;
    if (N < 1 || N > 64) { fn_set_error("cheby1: order N must be between 1 and 64"); return TSR_EARG; }
    if (!(args[1].kind == 1) || !(args[2].kind == 1)) { fn_set_error("cheby1: rp and a scalar Wn are required"); return TSR_EARG; }
    const double rp = args[1].num, Wn = args[2].num;
    int hp = 0;
    if (nargs > 3 && args[3].kind == 2 && args[3].str) {
        if (strcmp(args[3].str, "high") == 0 || strcmp(args[3].str, "highpass") == 0) hp = 1;
        else if (!(strcmp(args[3].str, "low") == 0 || strcmp(args[3].str, "lowpass") == 0)) { fn_set_error("cheby1: only lowpass/highpass are supported"); return TSR_EARG; }
    }
    const double eps = sqrt(pow(10.0, 0.1 * rp) - 1.0);
    const double mu = asinh(1.0 / eps) / (double)N;
    double complex *p = (double complex *)malloc((size_t)N * sizeof(double complex));
    if (!p) return TSR_ENOMEM;
    for (int i = 0; i < N; i++) { const double theta = M_PI * (double)(-N + 1 + 2 * i) / (2.0 * (double)N); p[i] = -csinh(mu + I * theta); }
    double complex kp = 1.0; for (int i = 0; i < N; i++) kp *= -p[i];
    double k = creal(kp);
    if (N % 2 == 0) k /= sqrt(1.0 + eps * eps);
    return iir_lp_proto_to_ba_z(NULL, 0, p, N, k, hp, Wn, &res[0]);
}

/* cheby2(N, rs, Wn, btype='low'): Chebyshev type II IIR filter, digital lowpass/highpass, output 'ba'
   (scipy.signal.cheby2). Analog prototype cheb2ap (equiripple stopband; has zeros), then the shared pipeline. */
static int r_cheby2(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const int N = (args[0].kind == 1) ? (int)args[0].num : 0;
    if (N < 1 || N > 64) { fn_set_error("cheby2: order N must be between 1 and 64"); return TSR_EARG; }
    if (!(args[1].kind == 1) || !(args[2].kind == 1)) { fn_set_error("cheby2: rs and a scalar Wn are required"); return TSR_EARG; }
    const double rs = args[1].num, Wn = args[2].num;
    int hp = 0;
    if (nargs > 3 && args[3].kind == 2 && args[3].str) {
        if (strcmp(args[3].str, "high") == 0 || strcmp(args[3].str, "highpass") == 0) hp = 1;
        else if (!(strcmp(args[3].str, "low") == 0 || strcmp(args[3].str, "lowpass") == 0)) { fn_set_error("cheby2: only lowpass/highpass are supported"); return TSR_EARG; }
    }
    const double de = 1.0 / sqrt(pow(10.0, 0.1 * rs) - 1.0);
    const double mu = asinh(1.0 / de) / (double)N;
    /* zeros: 1j / sin(m*pi/(2N)) for m excluding 0 when N is odd */
    const int nz0 = (N % 2) ? N - 1 : N;
    double complex *z0 = (double complex *)malloc((size_t)(nz0 > 0 ? nz0 : 1) * sizeof(double complex));
    double complex *p = (double complex *)malloc((size_t)N * sizeof(double complex));
    if (!z0 || !p) { free(z0); free(p); return TSR_ENOMEM; }
    int zi = 0;
    for (int i = 0; i < N; i++) {
        const int m = -N + 1 + 2 * i;
        if (m == 0) continue;                                /* skip the zero at infinity for odd N */
        z0[zi++] = I / sin((double)m * M_PI / (2.0 * (double)N));   /* scipy: -conj(1j/sin) = +1j/sin (sin real) */
    }
    for (int i = 0; i < N; i++) {
        const double theta = M_PI * (double)(-N + 1 + 2 * i) / (2.0 * (double)N);
        const double complex pp = -cexp(I * theta);
        const double complex ps = sinh(mu) * creal(pp) + I * cosh(mu) * cimag(pp);
        p[i] = 1.0 / ps;
    }
    double complex zp = 1.0, pp2 = 1.0;
    for (int i = 0; i < nz0; i++) zp *= -z0[i];
    for (int i = 0; i < N; i++) pp2 *= -p[i];
    double k = creal(pp2 / zp);
    int rc = iir_lp_proto_to_ba_z(z0, nz0, p, N, k, hp, Wn, &res[0]);   /* frees p */
    free(z0);
    return rc;
}

/* sosfilt(sos, x): cascade of second-order sections (biquads) in transposed direct form II, zero initial
   state (scipy.signal.sosfilt; 1-D real). sos is (n_sections, 6): [b0,b1,b2,a0,a1,a2] per row. */
static int r_sosfilt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[1] != 6) { fn_set_error("sosfilt: sos must be an (n_sections, 6) array"); return TSR_EARG; }
    if (args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("sosfilt: x must be a 1-D array"); return TSR_EARG; }
    const int64_t nsec = args[0].arr.shape[0], n = args[1].arr.shape[0];
    int64_t tsos, tx;
    double *sos = fn_arg_doubles(&args[0], &tsos); if (!sos) return TSR_ENOMEM;
    double *x = fn_arg_doubles(&args[1], &tx); if (!x) { fn_free_doubles(sos, tsos); return TSR_ENOMEM; }
    int64_t sh[1] = {n};
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!y) { fn_free_doubles(sos, tsos); fn_free_doubles(x, tx); return TSR_ENOMEM; }
    for (int64_t i = 0; i < n; i++) y[i] = x[i];
    for (int64_t s = 0; s < nsec; s++) {
        const double a0 = sos[s * 6 + 3];
        const double b0 = sos[s * 6 + 0] / a0, b1 = sos[s * 6 + 1] / a0, b2 = sos[s * 6 + 2] / a0;
        const double a1 = sos[s * 6 + 4] / a0, a2 = sos[s * 6 + 5] / a0;
        double w1 = 0.0, w2 = 0.0;                            /* transposed direct form II state */
        for (int64_t i = 0; i < n; i++) {
            const double xi = y[i];
            const double yi = b0 * xi + w1;
            w1 = b1 * xi - a1 * yi + w2;
            w2 = b2 * xi - a2 * yi;
            y[i] = yi;
        }
    }
    fn_free_doubles(sos, tsos); fn_free_doubles(x, tx);
    return TSR_OK;
}

/* Transposed-direct-form-II IIR filter: y[i] from b[0..K-1], a[0..K-1] (a[0]==1), state z[0..K-2] updated. */
static void df2t_filter(const double *b, const double *a, int K, const double *x, int64_t n, double *z, double *out)
{
    for (int64_t i = 0; i < n; i++) {
        const double xi = x[i];
        const double yi = b[0] * xi + (K > 1 ? z[0] : 0.0);
        for (int j = 0; j < K - 2; j++) z[j] = b[j + 1] * xi + z[j + 1] - a[j + 1] * yi;
        if (K > 1) z[K - 2] = b[K - 1] * xi - a[K - 1] * yi;
        out[i] = yi;
    }
}

/* Steady-state initial conditions zi[0..K-2] for lfilter (scipy.signal.lfilter_zi): zi solves
   (I - companion(a)^T) zi = b[1:] - a[1:]*b[0], with a, b normalised by a[0] and zero-padded to length K. */
static int lfilter_zi_into(const double *bin, int64_t nb, const double *ain, int64_t na, int K, double *zi)
{
    const double a0 = ain[0];
    double *a = (double *)calloc((size_t)K, sizeof(double));
    double *b = (double *)calloc((size_t)K, sizeof(double));
    if (!a || !b) { free(a); free(b); return TSR_ENOMEM; }
    for (int64_t i = 0; i < na; i++) a[i] = ain[i] / a0;
    for (int64_t i = 0; i < nb; i++) b[i] = bin[i] / a0;
    const int m = K - 1;
    if (m <= 0) { free(a); free(b); return TSR_OK; }
    double *M = (double *)calloc((size_t)m * m, sizeof(double));
    double *B = (double *)calloc((size_t)m, sizeof(double));
    if (!M || !B) { free(a); free(b); free(M); free(B); return TSR_ENOMEM; }
    for (int r = 0; r < m; r++) M[r * m + r] = 1.0;           /* I */
    for (int j = 0; j < m; j++) M[j * m + 0] += a[j + 1];     /* - companion^T first column (-a[1:]) */
    for (int i = 1; i < m; i++) M[(i - 1) * m + i] -= 1.0;    /* - companion^T superdiagonal */
    for (int j = 0; j < m; j++) B[j] = b[j + 1] - a[j + 1] * b[0];
    int rc = gesolve(m, M, B);
    if (rc == 0) for (int j = 0; j < m; j++) zi[j] = B[j];
    free(a); free(b); free(M); free(B);
    return rc < 0 ? TSR_EARG : TSR_OK;
}

/* lfilter_zi(b, a): the steady-state step-response initial conditions (scipy.signal.lfilter_zi). */
static int r_lfilter_zi(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("lfilter_zi: b and a must be 1-D arrays"); return TSR_EARG; }
    const int64_t nb = args[0].arr.shape[0], na = args[1].arr.shape[0];
    const int K = (int)(na > nb ? na : nb);
    int64_t tb, ta; double *b = fn_arg_doubles(&args[0], &tb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &ta); if (!a) { fn_free_doubles(b, tb); return TSR_ENOMEM; }
    int64_t sh[1] = {K - 1 > 0 ? K - 1 : 0};
    double *zi = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    int rc = (zi || sh[0] == 0) ? lfilter_zi_into(b, nb, a, na, K, zi) : TSR_ENOMEM;
    fn_free_doubles(b, tb); fn_free_doubles(a, ta);
    return rc;
}

/* filtfilt(b, a, x): zero-phase forward-backward filtering with odd padding (scipy.signal.filtfilt;
   padtype='odd', padlen=3*max(len(a),len(b)), method='pad'). 1-D real. */
static int r_filtfilt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1 || args[2].kind != 3 || args[2].arr.ndim != 1) { fn_set_error("filtfilt: b, a and x must be 1-D arrays"); return TSR_EARG; }
    const int64_t nb = args[0].arr.shape[0], na = args[1].arr.shape[0], n = args[2].arr.shape[0];
    const int K = (int)(na > nb ? na : nb);
    const int64_t edge = 3 * K;
    if (n <= edge) { fn_set_error("filtfilt: the length of the input vector x must be greater than padlen, which is %lld", (long long)edge); return TSR_EARG; }
    int64_t tb, ta, tx;
    double *bi = fn_arg_doubles(&args[0], &tb); if (!bi) return TSR_ENOMEM;
    double *ai = fn_arg_doubles(&args[1], &ta); if (!ai) { fn_free_doubles(bi, tb); return TSR_ENOMEM; }
    double *x = fn_arg_doubles(&args[2], &tx); if (!x) { fn_free_doubles(bi, tb); fn_free_doubles(ai, ta); return TSR_ENOMEM; }
    const double a0 = ai[0];
    double *b = (double *)calloc((size_t)K, sizeof(double));
    double *a = (double *)calloc((size_t)K, sizeof(double));
    double *zi = (double *)calloc((size_t)(K > 1 ? K - 1 : 1), sizeof(double));
    const int64_t next = n + 2 * edge;
    double *ext = (double *)malloc((size_t)next * sizeof(double));
    double *yf = (double *)malloc((size_t)next * sizeof(double));
    double *z = (double *)malloc((size_t)(K > 1 ? K - 1 : 1) * sizeof(double));
    int rc = TSR_OK;
    if (!b || !a || !zi || !ext || !yf || !z) { rc = TSR_ENOMEM; goto done; }
    for (int64_t i = 0; i < nb; i++) b[i] = bi[i] / a0;
    for (int64_t i = 0; i < na; i++) a[i] = ai[i] / a0;
    if ((rc = lfilter_zi_into(bi, nb, ai, na, K, zi)) < 0) goto done;
    for (int64_t k = 0; k < edge; k++) ext[k] = 2.0 * x[0] - x[edge - k];           /* odd left */
    for (int64_t i = 0; i < n; i++) ext[edge + i] = x[i];
    for (int64_t k = 0; k < edge; k++) ext[edge + n + k] = 2.0 * x[n - 1] - x[n - 2 - k];  /* odd right */
    for (int j = 0; j < K - 1; j++) z[j] = zi[j] * ext[0];                           /* forward */
    df2t_filter(b, a, K, ext, next, z, yf);
    for (int64_t i = 0; i < next / 2; i++) { double t = yf[i]; yf[i] = yf[next - 1 - i]; yf[next - 1 - i] = t; }  /* reverse in place */
    for (int j = 0; j < K - 1; j++) z[j] = zi[j] * yf[0];                            /* backward (on reversed) */
    df2t_filter(b, a, K, yf, next, z, ext);                                          /* reuse ext as output buffer */
    int64_t sh[1] = {n};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!out) { rc = TSR_ENOMEM; goto done; }
    for (int64_t i = 0; i < n; i++) out[i] = ext[next - 1 - (edge + i)];             /* reverse back + trim padding */
done:
    fn_free_doubles(bi, tb); fn_free_doubles(ai, ta); fn_free_doubles(x, tx);
    free(b); free(a); free(zi); free(ext); free(yf); free(z);
    return rc;
}

/* Apply a cascade of biquads (transposed DF2-II) to x -> out, with optional per-section state zi[2*nsec]
   (initial state in, final state out, when non-NULL). Section coefficients are normalised by a0. */
static void sosfilt_apply(const double *sos, int64_t nsec, const double *x, int64_t n, double *zi, double *out)
{
    for (int64_t i = 0; i < n; i++) out[i] = x[i];
    for (int64_t s = 0; s < nsec; s++) {
        const double a0 = sos[s * 6 + 3];
        const double b0 = sos[s * 6 + 0] / a0, b1 = sos[s * 6 + 1] / a0, b2 = sos[s * 6 + 2] / a0;
        const double a1 = sos[s * 6 + 4] / a0, a2 = sos[s * 6 + 5] / a0;
        double w1 = zi ? zi[s * 2 + 0] : 0.0, w2 = zi ? zi[s * 2 + 1] : 0.0;
        for (int64_t i = 0; i < n; i++) {
            const double xi = out[i];
            const double yi = b0 * xi + w1;
            w1 = b1 * xi - a1 * yi + w2;
            w2 = b2 * xi - a2 * yi;
            out[i] = yi;
        }
        if (zi) { zi[s * 2 + 0] = w1; zi[s * 2 + 1] = w2; }
    }
}

/* sosfilt_zi(sos): per-section steady-state initial conditions (scipy.signal.sosfilt_zi), shape (n_sections, 2);
   each section's lfilter_zi scaled by the cumulative DC gain of the preceding sections. */
static int r_sosfilt_zi(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[1] != 6) { fn_set_error("sosfilt_zi: sos must be an (n_sections, 6) array"); return TSR_EARG; }
    const int64_t nsec = args[0].arr.shape[0];
    int64_t tsos; double *sos = fn_arg_doubles(&args[0], &tsos); if (!sos) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){nsec, 2});
    if (!out) { fn_free_doubles(sos, tsos); return TSR_ENOMEM; }
    double scale = 1.0; int rc = TSR_OK;
    for (int64_t s = 0; s < nsec && rc == TSR_OK; s++) {
        const double *b = &sos[s * 6 + 0], *a = &sos[s * 6 + 3];
        double zi2[2] = {0, 0};
        rc = lfilter_zi_into(b, 3, a, 3, 3, zi2);
        out[s * 2 + 0] = scale * zi2[0];
        out[s * 2 + 1] = scale * zi2[1];
        scale *= (b[0] + b[1] + b[2]) / (a[0] + a[1] + a[2]);
    }
    fn_free_doubles(sos, tsos);
    return rc;
}

/* sosfiltfilt(sos, x): zero-phase forward-backward SOS filtering with odd padding (scipy.signal.sosfiltfilt). */
static int r_sosfiltfilt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nargs; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[1] != 6) { fn_set_error("sosfiltfilt: sos must be an (n_sections, 6) array"); return TSR_EARG; }
    if (args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("sosfiltfilt: x must be a 1-D array"); return TSR_EARG; }
    const int64_t nsec = args[0].arr.shape[0], n = args[1].arr.shape[0];
    int64_t tsos, tx;
    double *sos = fn_arg_doubles(&args[0], &tsos); if (!sos) return TSR_ENOMEM;
    double *x = fn_arg_doubles(&args[1], &tx); if (!x) { fn_free_doubles(sos, tsos); return TSR_ENOMEM; }
    int z2 = 0, z5 = 0;
    for (int64_t s = 0; s < nsec; s++) { if (sos[s * 6 + 2] == 0.0) z2++; if (sos[s * 6 + 5] == 0.0) z5++; }
    const int64_t ntaps = 2 * nsec + 1 - (z2 < z5 ? z2 : z5);
    const int64_t edge = 3 * ntaps, next = n + 2 * edge;
    double *zi = (double *)calloc((size_t)(nsec * 2 > 0 ? nsec * 2 : 1), sizeof(double));
    double *st = (double *)malloc((size_t)(nsec * 2 > 0 ? nsec * 2 : 1) * sizeof(double));
    double *ext = (double *)malloc((size_t)next * sizeof(double));
    double *yf = (double *)malloc((size_t)next * sizeof(double));
    int rc = TSR_OK;
    if (n <= edge) { fn_set_error("sosfiltfilt: the length of x must be greater than padlen (%lld)", (long long)edge); rc = TSR_EARG; goto done; }
    if (!zi || !st || !ext || !yf) { rc = TSR_ENOMEM; goto done; }
    { double scale = 1.0;                                     /* sosfilt_zi inline */
      for (int64_t s = 0; s < nsec && rc == TSR_OK; s++) {
          const double *b = &sos[s * 6 + 0], *a = &sos[s * 6 + 3];
          double zi2[2] = {0, 0};
          rc = lfilter_zi_into(b, 3, a, 3, 3, zi2);
          zi[s * 2 + 0] = scale * zi2[0]; zi[s * 2 + 1] = scale * zi2[1];
          scale *= (b[0] + b[1] + b[2]) / (a[0] + a[1] + a[2]);
      } }
    if (rc < 0) goto done;
    for (int64_t k = 0; k < edge; k++) ext[k] = 2.0 * x[0] - x[edge - k];
    for (int64_t i = 0; i < n; i++) ext[edge + i] = x[i];
    for (int64_t k = 0; k < edge; k++) ext[edge + n + k] = 2.0 * x[n - 1] - x[n - 2 - k];
    for (int64_t j = 0; j < nsec * 2; j++) st[j] = zi[j] * ext[0];       /* forward */
    sosfilt_apply(sos, nsec, ext, next, st, yf);
    for (int64_t i = 0; i < next / 2; i++) { double t = yf[i]; yf[i] = yf[next - 1 - i]; yf[next - 1 - i] = t; }
    for (int64_t j = 0; j < nsec * 2; j++) st[j] = zi[j] * yf[0];        /* backward on reversed */
    sosfilt_apply(sos, nsec, yf, next, st, ext);                         /* reuse ext as output */
    {
        int64_t sh[1] = {n};
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
        if (!out) { rc = TSR_ENOMEM; goto done; }
        for (int64_t i = 0; i < n; i++) out[i] = ext[next - 1 - (edge + i)];
    }
done:
    fn_free_doubles(sos, tsos); fn_free_doubles(x, tx);
    free(zi); free(st); free(ext); free(yf);
    return rc;
}

/* N-D convolution / correlation by direct computation, sliced per mode. scipy computes fftconvolve/oaconvolve
   via the FFT, but the result is the same operation; a direct sum matches it to within tolerance. correlate is
   convolve with the second input reversed in every axis (real inputs; the conjugate is a no-op). Real float64. */
static int nd_convolve(const tsr_arg *A, const tsr_arg *V, int mode, int reverse_v, const char *who, tsr_result *res)
{
    if (A->kind != 3 || V->kind != 3) { fn_set_error("%s: inputs must be arrays", who); return TSR_EARG; }
    const int32_t nd = A->arr.ndim;
    if (V->arr.ndim != nd) { fn_set_error("%s: inputs must have the same number of dimensions", who); return TSR_EARG; }
    if (nd < 1 || nd > TSR_MAXDIM) { fn_set_error("%s: unsupported dimensionality", who); return TSR_EARG; }
    int64_t s1[TSR_MAXDIM], s2[TSR_MAXDIM], f[TSR_MAXDIM], fst[TSR_MAXDIM], vst[TSR_MAXDIM];
    int64_t fullsize = 1;
    for (int32_t d = 0; d < nd; d++) {
        s1[d] = A->arr.shape[d]; s2[d] = V->arr.shape[d];
        f[d] = (s1[d] > 0 && s2[d] > 0) ? s1[d] + s2[d] - 1 : 0;
        fullsize *= f[d];
    }
    { int64_t acc = 1; for (int32_t d = nd - 1; d >= 0; d--) { fst[d] = acc; acc *= f[d]; } }
    { int64_t acc = 1; for (int32_t d = nd - 1; d >= 0; d--) { vst[d] = acc; acc *= s2[d]; } }

    int64_t n1, n2;
    double *a = fn_arg_doubles(A, &n1); if (!a) { fn_set_error("%s: in1 must be a real numeric array", who); return TSR_ENOMEM; }
    double *v = fn_arg_doubles(V, &n2); if (!v) { fn_free_doubles(a, n1); fn_set_error("%s: in2 must be a real numeric array", who); return TSR_ENOMEM; }
    double *full = (double *)calloc((size_t)(fullsize > 0 ? fullsize : 1), sizeof(double));
    if (!full) { fn_free_doubles(a, n1); fn_free_doubles(v, n2); return TSR_ENOMEM; }

    int64_t J[TSR_MAXDIM]; for (int32_t d = 0; d < nd; d++) J[d] = 0;
    for (int64_t ja = 0; ja < n1; ja++) {
        const double av = a[ja];
        if (av != 0.0 && fullsize > 0) {
            int64_t K[TSR_MAXDIM]; for (int32_t d = 0; d < nd; d++) K[d] = 0;
            for (int64_t kv = 0; kv < n2; kv++) {
                int64_t off = 0, vi = 0;
                for (int32_t d = 0; d < nd; d++) {
                    off += (J[d] + K[d]) * fst[d];
                    vi += (reverse_v ? (s2[d] - 1 - K[d]) : K[d]) * vst[d];
                }
                full[off] += av * v[vi];
                for (int32_t d = nd - 1; d >= 0; d--) { if (++K[d] < s2[d]) break; K[d] = 0; }
            }
        }
        for (int32_t d = nd - 1; d >= 0; d--) { if (++J[d] < s1[d]) break; J[d] = 0; }
    }

    int64_t osh[TSR_MAXDIM], ost[TSR_MAXDIM];
    for (int32_t d = 0; d < nd; d++) {
        if (mode == CONV_FULL) { osh[d] = f[d]; ost[d] = 0; }
        else if (mode == CONV_SAME) { osh[d] = s1[d]; ost[d] = (f[d] - s1[d]) / 2; }
        else {                                                /* valid: in1 must be >= in2 in every axis */
            if (s1[d] < s2[d]) { free(full); fn_free_doubles(a, n1); fn_free_doubles(v, n2);
                fn_set_error("%s: for 'valid' mode, in1 must be at least as large as in2 in every dimension", who); return TSR_EARG; }
            osh[d] = s1[d] - s2[d] + 1; ost[d] = s2[d] - 1;
        }
    }
    double *out = (double *)fn_result_array(res, TSR_F64, nd, osh);
    if (!out) { free(full); fn_free_doubles(a, n1); fn_free_doubles(v, n2); return TSR_ENOMEM; }
    int64_t osize = 1; for (int32_t d = 0; d < nd; d++) osize *= osh[d];
    int64_t O[TSR_MAXDIM]; for (int32_t d = 0; d < nd; d++) O[d] = 0;
    for (int64_t i = 0; i < osize; i++) {
        int64_t off = 0; for (int32_t d = 0; d < nd; d++) off += (O[d] + ost[d]) * fst[d];
        out[i] = full[off];
        for (int32_t d = nd - 1; d >= 0; d--) { if (++O[d] < osh[d]) break; O[d] = 0; }
    }
    free(full); fn_free_doubles(a, n1); fn_free_doubles(v, n2);
    return TSR_OK;
}

/* reject a non-None axes argument (we convolve over all axes) */
static int no_axes(const tsr_arg *args, int nargs, int idx, const char *who)
{
    if (idx < nargs && args[idx].kind != 0) { fn_set_error("%s: the axes argument is not supported", who); return TSR_EARG; }
    return TSR_OK;
}

/* fftconvolve(in1, in2, mode='full', axes=None): N-D convolution (scipy.signal.fftconvolve) */
static int r_fftconvolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int mode, rc; if ((rc = parse_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    if ((rc = no_axes(args, nargs, 3, "fftconvolve")) < 0) return rc;
    return nd_convolve(&args[0], &args[1], mode, 0, "fftconvolve", &res[0]);
}

/* oaconvolve(in1, in2, mode='full', axes=None): overlap-add convolution; same result as fftconvolve */
static int r_oaconvolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int mode, rc; if ((rc = parse_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    if ((rc = no_axes(args, nargs, 3, "oaconvolve")) < 0) return rc;
    return nd_convolve(&args[0], &args[1], mode, 0, "oaconvolve", &res[0]);
}

/* correlate(in1, in2, mode='full', method='auto'): N-D cross-correlation (scipy.signal.correlate) */
static int r_correlate(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int mode, rc; if ((rc = parse_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    return nd_convolve(&args[0], &args[1], mode, 1, "correlate", &res[0]);
}

/* -------------------------------------------------- detrend / savgol (least-squares helpers) ----- */

/* Solve the n-by-n dense system A x = b in place (A row-major, destroyed; b holds x on return) by
   Gaussian elimination with partial pivoting. Returns 0, or -1 if singular. Small n (<= polyorder+1). */
static int gesolve(int n, double *A, double *b)
{
    for (int c = 0; c < n; c++) {
        int piv = c; double best = fabs(A[c * n + c]);
        for (int r = c + 1; r < n; r++) { double v = fabs(A[r * n + c]); if (v > best) { best = v; piv = r; } }
        if (best == 0.0) return -1;
        if (piv != c) { for (int k = 0; k < n; k++) { double t = A[c * n + k]; A[c * n + k] = A[piv * n + k]; A[piv * n + k] = t; } double t = b[c]; b[c] = b[piv]; b[piv] = t; }
        const double d = A[c * n + c];
        for (int r = 0; r < n; r++) {
            if (r == c) continue;
            const double f = A[r * n + c] / d;
            if (f == 0.0) continue;
            for (int k = c; k < n; k++) A[r * n + k] -= f * A[c * n + k];
            b[r] -= f * b[c];
        }
    }
    for (int i = 0; i < n; i++) b[i] /= A[i * n + i];
    return 0;
}

/* savgol_coeffs(window_length, polyorder, deriv, delta, pos, use='conv') into out[window_length].
   Minimum-norm least-squares solution of the underdetermined A c = y, A[r][j] = x[j]**r, via LAPACK dgelsd
   (SVD) — the same routine scipy's lstsq uses, so the result matches scipy bit-for-bit. */
static int savgol_coeffs_into(int64_t wl, int64_t po, int64_t deriv, double delta, double pos, int use_conv, double *out)
{
    if (wl <= 0 || po < 0 || po >= wl) { fn_set_error("savgol_coeffs: require 0 <= polyorder < window_length"); return TSR_EARG; }
    const int p = (int)po + 1;                                /* rows of A (polynomial terms); cols = wl */
    double fact = 1.0; for (int64_t k = 2; k <= deriv; k++) fact *= (double)k;
    double dp = 1.0; for (int64_t k = 0; k < deriv; k++) dp *= delta;
    double *A = (double *)malloc((size_t)p * (size_t)wl * sizeof(double));
    double *b = (double *)calloc((size_t)wl, sizeof(double)); /* RHS sized max(m,n) = wl */
    double *s = (double *)malloc((size_t)p * sizeof(double));
    if (!A || !b || !s) { free(A); free(b); free(s); return TSR_ENOMEM; }
    for (int r = 0; r < p; r++)
        for (int64_t j = 0; j < wl; j++) {
            double xj = use_conv ? (-pos + (double)(wl - 1 - j)) : (-pos + (double)j);
            A[r * wl + j] = pow(xj, (double)r);
        }
    if (deriv < p) b[deriv] = fact / dp;
    lapack_int rank = 0;
    lapack_int info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, p, (lapack_int)wl, 1, A, (lapack_int)wl, b, 1, s, -1.0, &rank);
    if (info != 0) { free(A); free(b); free(s); fn_set_error("savgol_coeffs: least-squares solve failed"); return TSR_EARG; }
    for (int64_t j = 0; j < wl; j++) out[j] = b[j];
    free(A); free(b); free(s);
    return TSR_OK;
}

static int64_t win_len(const tsr_arg *args, int nargs, int *err);   /* defined in the windows section below */

/* r_savgol_coeffs(window_length, polyorder, deriv=0, delta=1.0, pos=None, use='conv') */
static int r_savgol_coeffs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t wl = win_len(args, nargs, &err); if (err) return TSR_EARG;   /* reuse the length parser */
    int64_t po = (nargs > 1 && args[1].kind == 1) ? (int64_t)args[1].num : 0;
    int64_t deriv = (nargs > 2 && args[2].kind == 1) ? (int64_t)args[2].num : 0;
    double delta = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1.0;
    int64_t halflen = wl / 2, rem = wl % 2;
    double pos = (nargs > 4 && args[4].kind == 1) ? args[4].num : (rem == 0 ? (double)halflen - 0.5 : (double)halflen);
    int use_conv = 1;
    if (nargs > 5 && args[5].kind == 2 && args[5].str && strcmp(args[5].str, "dot") == 0) use_conv = 0;
    int64_t sh[1] = {wl};
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!out) return TSR_ENOMEM;
    return savgol_coeffs_into(wl, po, deriv, delta, pos, use_conv, out);
}

/* boundary index fold for savgol_filter convolution modes (mirror/nearest/wrap); constant -> -1 sentinel */
enum { SG_MIRROR, SG_CONSTANT, SG_NEAREST, SG_WRAP };
static int64_t sg_fold(int64_t p, int64_t n, int mode)
{
    if (n <= 0) return -1;
    if (p >= 0 && p < n) return p;
    switch (mode) {
    case SG_NEAREST: return p < 0 ? 0 : n - 1;
    case SG_WRAP:    { int64_t m = p % n; return m < 0 ? m + n : m; }
    case SG_CONSTANT: return -1;
    default: {                                                /* mirror: (d c b | a b c d), period 2n-2 */
        if (n == 1) return 0;
        int64_t period = 2 * (n - 1);
        int64_t m = p % period; if (m < 0) m += period;
        return m >= n ? period - m : m;
    }
    }
}

/* least-squares polynomial fit (degree po) to wl points (t=0..wl-1, values v), replicating numpy.polyfit
   exactly: descending Vandermonde, per-column L2 scaling, rcond = len*eps, LAPACK dgelsd. Writes ascending
   coefficients a[0..po] (a[m] = coefficient of t**m). For savgol_filter mode='interp' edge handling. */
static int poly_lsfit(int64_t wl, int64_t po, const double *v, double *a)
{
    const int p = (int)po + 1;                                /* columns; rows = wl (overdetermined) */
    double *W = (double *)malloc((size_t)wl * (size_t)p * sizeof(double));
    double *b = (double *)malloc((size_t)wl * sizeof(double)); /* RHS sized max(m,n) = wl */
    double *s = (double *)malloc((size_t)p * sizeof(double));
    double *scale = (double *)malloc((size_t)p * sizeof(double));
    if (!W || !b || !s || !scale) { free(W); free(b); free(s); free(scale); return TSR_ENOMEM; }
    for (int k = 0; k < p; k++) {                             /* descending column k -> power (po-k); scale by its L2 norm */
        double sq = 0.0; for (int64_t t = 0; t < wl; t++) { double e = pow((double)t, (double)(po - k)); sq += e * e; }
        scale[k] = sq > 0.0 ? sqrt(sq) : 1.0;
    }
    for (int64_t t = 0; t < wl; t++) { for (int k = 0; k < p; k++) W[t * p + k] = pow((double)t, (double)(po - k)) / scale[k]; b[t] = v[t]; }
    lapack_int rank = 0;
    lapack_int info = LAPACKE_dgelsd(LAPACK_ROW_MAJOR, (lapack_int)wl, p, 1, W, p, b, 1, s, (double)wl * DBL_EPSILON, &rank);
    if (info == 0) for (int k = 0; k < p; k++) a[po - k] = b[k] / scale[k];   /* descending -> ascending */
    free(W); free(b); free(s); free(scale);
    return info == 0 ? TSR_OK : TSR_EARG;
}

/* evaluate the deriv-th derivative of an ascending-coeff polynomial a[0..po] at t, divided by delta**deriv */
static double poly_deriv_val(const double *a, int64_t po, int64_t deriv, double t, double delta)
{
    double acc = 0.0;
    for (int64_t m = 0; m + deriv <= po; m++) {
        double coef = a[m + deriv];
        for (int64_t q = 0; q < deriv; q++) coef *= (double)(m + deriv - q);   /* (m+deriv)!/m! */
        acc += coef * pow(t, (double)m);
    }
    double dp = 1.0; for (int64_t k = 0; k < deriv; k++) dp *= delta;
    return acc / dp;
}

/* savgol_filter(x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0): 1-D real.
   Interior by convolving with savgol_coeffs; edges by a local polynomial fit when mode='interp'. */
static int r_savgol_filter(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("savgol_filter: x must be a 1-D array"); return TSR_EARG; }
    int64_t wl = (nargs > 1 && args[1].kind == 1) ? (int64_t)args[1].num : 0;
    int64_t po = (nargs > 2 && args[2].kind == 1) ? (int64_t)args[2].num : 0;
    int64_t deriv = (nargs > 3 && args[3].kind == 1) ? (int64_t)args[3].num : 0;
    double delta = (nargs > 4 && args[4].kind == 1) ? args[4].num : 1.0;
    const char *modes = (nargs > 6 && args[6].kind == 2 && args[6].str) ? args[6].str : "interp";
    double cval = (nargs > 7 && args[7].kind == 1) ? args[7].num : 0.0;
    if (wl <= 0 || po < 0 || po >= wl) { fn_set_error("savgol_filter: require 0 <= polyorder < window_length"); return TSR_EARG; }
    int interp = 0, fmode = SG_MIRROR;
    if (strcmp(modes, "interp") == 0) interp = 1;
    else if (strcmp(modes, "mirror") == 0) fmode = SG_MIRROR;
    else if (strcmp(modes, "nearest") == 0) fmode = SG_NEAREST;
    else if (strcmp(modes, "wrap") == 0) fmode = SG_WRAP;
    else if (strcmp(modes, "constant") == 0) fmode = SG_CONSTANT;
    else { fn_set_error("savgol_filter: mode '%s' is not supported", modes); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    if (interp && wl > n) { fn_set_error("savgol_filter: window_length must be <= size of x for mode='interp'"); return TSR_EARG; }
    int64_t tx; double *x = fn_arg_doubles(&args[0], &tx); if (!x) return TSR_ENOMEM;
    double *coef = (double *)malloc((size_t)wl * sizeof(double));
    if (!coef) { fn_free_doubles(x, tx); return TSR_ENOMEM; }
    int rc = savgol_coeffs_into(wl, po, deriv, delta, (wl % 2 == 0 ? (double)(wl / 2) - 0.5 : (double)(wl / 2)), 1, coef);
    if (rc < 0) { fn_free_doubles(x, tx); free(coef); return rc; }
    int64_t sh[1] = {n};
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!y) { fn_free_doubles(x, tx); free(coef); return TSR_ENOMEM; }
    const int64_t half = wl / 2;
    for (int64_t i = 0; i < n; i++) {
        double acc = 0.0;
        for (int64_t k = 0; k < wl; k++) {
            int64_t src = i + half - k;
            if (interp) { if (src < 0 || src >= n) { src = -1; } }      /* interp: interior uses 'constant' (edges overwritten) */
            else src = sg_fold(src, n, fmode);
            acc += coef[k] * (src < 0 ? (interp ? 0.0 : cval) : x[src]);
        }
        y[i] = acc;
    }
    if (interp && wl > 1) {
        double *a = (double *)malloc((size_t)(po + 1) * sizeof(double));
        double *v = (double *)malloc((size_t)wl * sizeof(double));
        if (!a || !v) { fn_free_doubles(x, tx); free(coef); free(a); free(v); return TSR_ENOMEM; }
        /* first edge: fit x[0..wl-1], set y[0..half-1] */
        for (int64_t t = 0; t < wl; t++) v[t] = x[t];
        if ((rc = poly_lsfit(wl, po, v, a)) == TSR_OK)
            for (int64_t i = 0; i < half; i++) y[i] = poly_deriv_val(a, po, deriv, (double)i, delta);
        /* last edge: fit x[n-wl..n-1], set y[n-half..n-1] */
        if (rc == TSR_OK) {
            for (int64_t t = 0; t < wl; t++) v[t] = x[n - wl + t];
            if ((rc = poly_lsfit(wl, po, v, a)) == TSR_OK)
                for (int64_t i = n - half; i < n; i++) y[i] = poly_deriv_val(a, po, deriv, (double)(i - (n - wl)), delta);
        }
        free(a); free(v);
        if (rc < 0) { fn_free_doubles(x, tx); free(coef); return rc; }
    }
    fn_free_doubles(x, tx); free(coef);
    return TSR_OK;
}

/* detrend(data, axis=-1, type='linear', bp=0): remove a constant or piecewise-linear least-squares trend
   along one axis (scipy.signal.detrend). Real float64; bp as a scalar or 1-D array of break indices. */
static int r_detrend(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("detrend: data must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1) { fn_set_error("detrend: data must have at least one dimension"); return TSR_EARG; }
    int64_t ax = (nargs > 1 && args[1].kind == 1) ? (int64_t)args[1].num : -1;
    if (ax < 0) ax += nd;
    if (ax < 0 || ax >= nd) { fn_set_error("detrend: axis out of range"); return TSR_EARG; }
    int linear = 1;
    if (nargs > 2 && args[2].kind == 2 && args[2].str) {
        if (strcmp(args[2].str, "constant") == 0 || strcmp(args[2].str, "c") == 0) linear = 0;
        else if (!(strcmp(args[2].str, "linear") == 0 || strcmp(args[2].str, "l") == 0)) { fn_set_error("detrend: type must be 'linear' or 'constant'"); return TSR_EARG; }
    }
    const int64_t N = args[0].arr.shape[ax];
    /* break points: 0 and N always; plus a scalar or 1-D array argument */
    int64_t bpbuf[64]; int nbp = 0; bpbuf[nbp++] = 0;
    if (nargs > 3 && args[3].kind == 3 && args[3].arr.ndim >= 1) {
        int64_t tb; double *bv = fn_arg_doubles(&args[3], &tb);
        if (bv) { for (int64_t i = 0; i < tb && nbp < 62; i++) { int64_t b = (int64_t)bv[i]; if (b > 0 && b < N) bpbuf[nbp++] = b; } fn_free_doubles(bv, tb); }
    } else if (nargs > 3 && args[3].kind == 1 && args[3].num != 0) {
        int64_t b = (int64_t)args[3].num; if (b > 0 && b < N) bpbuf[nbp++] = b;
    }
    bpbuf[nbp++] = N;
    /* sort/unique the break points */
    for (int i = 1; i < nbp; i++) for (int j = i; j > 0 && bpbuf[j] < bpbuf[j - 1]; j--) { int64_t t = bpbuf[j]; bpbuf[j] = bpbuf[j - 1]; bpbuf[j - 1] = t; }
    int uq = 0; for (int i = 0; i < nbp; i++) if (i == 0 || bpbuf[i] != bpbuf[uq - 1]) bpbuf[uq++] = bpbuf[i]; nbp = uq;

    int64_t tin; double *in = fn_arg_doubles(&args[0], &tin); if (!in) return TSR_ENOMEM;
    double *out = (double *)fn_result_array(&res[0], TSR_F64, nd, args[0].arr.shape);
    if (!out) { fn_free_doubles(in, tin); return TSR_ENOMEM; }
    const int64_t total = tin;
    memcpy(out, in, (size_t)total * sizeof(double));
    /* row-major strides; iterate every lane along `ax` */
    int64_t stride[TSR_MAXDIM], acc = 1;
    for (int32_t d = nd - 1; d >= 0; d--) { stride[d] = acc; acc *= args[0].arr.shape[d]; }
    const int64_t axstr = stride[ax];
    const int64_t nlanes = total / (N > 0 ? N : 1);
    int64_t idx[TSR_MAXDIM]; for (int32_t d = 0; d < nd; d++) idx[d] = 0;
    for (int64_t lane = 0; lane < nlanes; lane++) {
        int64_t base = 0; for (int32_t d = 0; d < nd; d++) if (d != ax) base += idx[d] * stride[d];
        for (int seg = 0; seg + 1 < nbp; seg++) {
            const int64_t s0 = bpbuf[seg], s1 = bpbuf[seg + 1], Np = s1 - s0;
            if (Np <= 0) continue;
            if (!linear) {
                double mean = 0.0; for (int64_t i = 0; i < Np; i++) mean += out[base + (s0 + i) * axstr];
                mean /= (double)Np;
                for (int64_t i = 0; i < Np; i++) out[base + (s0 + i) * axstr] -= mean;
            } else {
                /* least-squares fit of columns [t, 1], t = (i+1)/Np, then subtract (2x2 normal equations) */
                double s_tt = 0, s_t = 0, s_1 = (double)Np, s_ty = 0, s_y = 0;
                for (int64_t i = 0; i < Np; i++) { double t = (double)(i + 1) / (double)Np, yv = out[base + (s0 + i) * axstr]; s_tt += t * t; s_t += t; s_ty += t * yv; s_y += yv; }
                double A[4] = {s_tt, s_t, s_t, s_1}, b[2] = {s_ty, s_y};
                if (gesolve(2, A, b) == 0) for (int64_t i = 0; i < Np; i++) { double t = (double)(i + 1) / (double)Np; out[base + (s0 + i) * axstr] -= b[0] * t + b[1]; }
            }
        }
        for (int32_t d = nd - 1; d >= 0; d--) { if (d == ax) continue; if (++idx[d] < args[0].arr.shape[d]) break; idx[d] = 0; }
    }
    fn_free_doubles(in, tin);
    return TSR_OK;
}

/* ----------------------------------------------------------------- scipy.signal.windows ----- */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* The window length M (argument 0). Negative M is an error; stored through *err. */
static int64_t win_len(const tsr_arg *args, int nargs, int *err)
{
    *err = 0;
    if (nargs < 1) { *err = 1; return 0; }
    const tsr_arg *a = &args[0];
    int64_t M = (a->kind == 1 && (a->flags & 1)) ? a->ival : (a->kind == 1 ? (int64_t)a->num : -1);
    if (M < 0) { fn_set_error("window length M must be a non-negative integer"); *err = 1; }
    return M;
}

/* sym flag (default True) at argument index `idx`. */
static int win_sym(const tsr_arg *args, int nargs, int idx)
{
    if (idx >= nargs) return 1;
    const tsr_arg *a = &args[idx];
    if (a->kind == 0) return 1;                      /* None/omitted -> default True */
    if (a->kind == 4 || a->kind == 1) return a->num != 0;
    return 1;
}

/* Trivial length guard (scipy _len_guards): handles M<=1, returning 1 when it filled the result. */
static int win_guard(int64_t M, tsr_result *res, int *rc)
{
    *rc = TSR_OK;
    if (M > 1) return 0;
    int64_t sh[1] = {M < 0 ? 0 : M};
    double *o = (double *)fn_result_array(res, TSR_F64, 1, sh);
    if (!o) { *rc = TSR_ENOMEM; return 1; }
    if (M == 1) o[0] = 1.0;
    return 1;
}

/* w[i] = sum_k a[k] cos(k * fac_i), fac from -pi..pi over Mext points (scipy windows.general_cosine). */
static int win_cosine_sum(int64_t M, const double *a, int na, int sym, tsr_result *res)
{
    int rc; if (win_guard(M, res, &rc)) return rc;
    int64_t Mext = sym ? M : M + 1;
    double den = (double)(Mext - 1);
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(res, TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double fac = -M_PI + 2.0 * M_PI * (double)i / den;
        double s = 0.0;
        for (int k = 0; k < na; k++) s += a[k] * cos((double)k * fac);
        w[i] = s;
    }
    return TSR_OK;
}

#define WIN_COSINE(fn, sym_idx, ...)                                                            \
    static int fn(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres) { \
        (void)ctx; (void)nres;                                                                  \
        int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;              \
        static const double a[] = {__VA_ARGS__};                                                \
        return win_cosine_sum(M, a, (int)(sizeof a / sizeof a[0]), win_sym(args, nargs, sym_idx), &res[0]); \
    }

WIN_COSINE(r_win_hann, 1, 0.5, 0.5)
WIN_COSINE(r_win_hamming, 1, 0.54, 0.46)
WIN_COSINE(r_win_blackman, 1, 0.42, 0.5, 0.08)
WIN_COSINE(r_win_blackmanharris, 1, 0.35875, 0.48829, 0.14128, 0.01168)
WIN_COSINE(r_win_nuttall, 1, 0.3635819, 0.4891775, 0.1365995, 0.0106411)
WIN_COSINE(r_win_flattop, 1, 0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368)

/* general_cosine(M, a, sym=True): coefficients supplied as an array argument. */
static int r_win_general_cosine(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    if (nargs < 2 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("general_cosine: a must be a 1-D array of coefficients"); return TSR_EARG; }
    int64_t na; double *a = fn_arg_doubles(&args[1], &na);
    if (!a) return TSR_ENOMEM;
    int rc = win_cosine_sum(M, a, (int)na, win_sym(args, nargs, 2), &res[0]);
    fn_free_doubles(a, na);
    return rc;
}

/* general_hamming(M, alpha, sym=True) = general_cosine(M, [alpha, 1-alpha]). */
static int r_win_general_hamming(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    double alpha = (nargs > 1 && args[1].kind == 1) ? args[1].num : 0.54;
    double a[2] = {alpha, 1.0 - alpha};
    return win_cosine_sum(M, a, 2, win_sym(args, nargs, 2), &res[0]);
}

/* boxcar(M, sym=True): all ones. */
static int r_win_boxcar(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) w[i] = 1.0;
    return TSR_OK;
}

/* triang(M, sym=True). */
static int r_win_triang(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int64_t Mext = win_sym(args, nargs, 1) ? M : M + 1;
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    int64_t half = (Mext + 1) / 2;
    for (int64_t i = 0; i < M; i++) {
        int64_t k = i < half ? i : Mext - 1 - i;           /* mirror index into the rising half */
        w[i] = (Mext % 2 == 0) ? (2.0 * (double)(k + 1) - 1.0) / (double)Mext
                               : 2.0 * (double)(k + 1) / (double)(Mext + 1);
    }
    return TSR_OK;
}

/* bartlett(M, sym=True): triangular, zero at both ends. */
static int r_win_bartlett(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int64_t Mext = win_sym(args, nargs, 1) ? M : M + 1;
    double den = (double)(Mext - 1);
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double x = (double)i;
        w[i] = (x <= den / 2.0) ? 2.0 * x / den : 2.0 - 2.0 * x / den;
    }
    return TSR_OK;
}

/* cosine(M, sym=True): sin(pi/Mext * (i + 0.5)). */
static int r_win_cosine(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int64_t Mext = win_sym(args, nargs, 1) ? M : M + 1;
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) w[i] = sin(M_PI / (double)Mext * ((double)i + 0.5));
    return TSR_OK;
}

/* lanczos(M, sym=True): sinc(2 i/(Mext-1) - 1). */
static int r_win_lanczos(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int64_t Mext = win_sym(args, nargs, 1) ? M : M + 1;
    double den = (double)(Mext - 1);
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double x = 2.0 * (double)i / den - 1.0;
        w[i] = x == 0.0 ? 1.0 : sin(M_PI * x) / (M_PI * x);
    }
    return TSR_OK;
}

/* bohman(M, sym=True). */
static int r_win_bohman(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int sym = win_sym(args, nargs, 1);
    int64_t Mext = sym ? M : M + 1;
    double den = (double)(Mext - 1);
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double fac = fabs(-1.0 + 2.0 * (double)i / den);
        w[i] = (1.0 - fac) * cos(M_PI * fac) + 1.0 / M_PI * sin(M_PI * fac);
    }
    w[0] = 0.0;
    if (sym) w[M - 1] = 0.0;                               /* the periodic form drops the trailing zero */
    return TSR_OK;
}

/* barthann(M, sym=True). */
static int r_win_barthann(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int64_t Mext = win_sym(args, nargs, 1) ? M : M + 1;
    double den = (double)(Mext - 1);
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double fac = fabs((double)i / den - 0.5);
        w[i] = 0.62 - 0.48 * fac + 0.38 * cos(2.0 * M_PI * fac);
    }
    return TSR_OK;
}

/* parzen(M, sym=True). */
static int r_win_parzen(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int64_t Mext = win_sym(args, nargs, 1) ? M : M + 1;
    double h = (double)Mext / 2.0;
    double quarter = (double)(Mext - 1) / 4.0;
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double n = -((double)(Mext - 1)) / 2.0 + (double)i;
        double an = fabs(n), r = an / h;
        w[i] = (an <= quarter) ? 1.0 - 6.0 * r * r + 6.0 * r * r * r : 2.0 * (1.0 - r) * (1.0 - r) * (1.0 - r);
    }
    return TSR_OK;
}

/* scipy.special.i0 (xsf::cyl_bessel_i0), the same routine scipy.signal.windows.kaiser uses. */
extern void tsr_special_i0(const void *ctx, const double *in, double *out);

/* A float argument at index idx (default d when omitted or None). */
static double win_dbl(const tsr_arg *args, int nargs, int idx, double d)
{
    if (idx >= nargs) return d;
    const tsr_arg *a = &args[idx];
    if (a->kind == 0) return d;
    if (a->kind == 1 || a->kind == 4) return a->num;
    return d;
}

/* kaiser(M, beta, sym=True). */
static int r_win_kaiser(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    double beta = win_dbl(args, nargs, 1, 0.0);
    int64_t Mext = win_sym(args, nargs, 2) ? M : M + 1;
    double alpha = (double)(Mext - 1) / 2.0;
    double den; tsr_special_i0(NULL, &beta, &den);
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double t = ((double)i - alpha) / alpha;
        double arg = beta * sqrt(1.0 - t * t), num;
        tsr_special_i0(NULL, &arg, &num);
        w[i] = num / den;
    }
    return TSR_OK;
}

/* gaussian(M, std, sym=True). */
static int r_win_gaussian(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    double std = win_dbl(args, nargs, 1, 1.0);
    int64_t Mext = win_sym(args, nargs, 2) ? M : M + 1;
    double c = (double)(Mext - 1) / 2.0;
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) {
        double n = (double)i - c;
        w[i] = exp(-(n * n) / (2.0 * std * std));
    }
    return TSR_OK;
}

/* general_gaussian(M, p, sig, sym=True). */
static int r_win_general_gaussian(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    double p = win_dbl(args, nargs, 1, 1.0), sig = win_dbl(args, nargs, 2, 1.0);
    int64_t Mext = win_sym(args, nargs, 3) ? M : M + 1;
    double c = (double)(Mext - 1) / 2.0;
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++)
        w[i] = exp(-0.5 * pow(fabs(((double)i - c) / sig), 2.0 * p));
    return TSR_OK;
}

/* exponential(M, center=None, tau=1.0, sym=True). */
static int r_win_exponential(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    int sym = win_sym(args, nargs, 3);
    int64_t Mext = sym ? M : M + 1;
    int center_given = nargs > 1 && args[1].kind != 0;
    double center = center_given ? win_dbl(args, nargs, 1, 0.0) : (double)(Mext - 1) / 2.0;
    double tau = win_dbl(args, nargs, 2, 1.0);
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    for (int64_t i = 0; i < M; i++) w[i] = exp(-fabs((double)i - center) / tau);
    return TSR_OK;
}

/* tukey(M, alpha=0.5, sym=True): tapered cosine. */
static int r_win_tukey(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int err; int64_t M = win_len(args, nargs, &err); if (err) return TSR_EARG;
    int rc; if (win_guard(M, &res[0], &rc)) return rc;
    double alpha = win_dbl(args, nargs, 1, 0.5);
    int sym = win_sym(args, nargs, 2);
    int64_t Mext = sym ? M : M + 1;
    int64_t sh[1] = {M};
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, sh);
    if (!w) return TSR_ENOMEM;
    if (alpha <= 0.0) { for (int64_t i = 0; i < M; i++) w[i] = 1.0; return TSR_OK; }
    if (alpha >= 1.0) { static const double a[] = {0.5, 0.5}; return win_cosine_sum(M, a, 2, sym, &res[0]); }
    double den = (double)(Mext - 1);
    int64_t width = (int64_t)(alpha * den / 2.0);          /* floor */
    for (int64_t i = 0; i < M; i++) {
        if (i <= width)
            w[i] = 0.5 * (1.0 + cos(M_PI * (-1.0 + 2.0 * (double)i / (alpha * den))));
        else if (i >= Mext - width - 1)
            w[i] = 0.5 * (1.0 + cos(M_PI * (-2.0 / alpha + 1.0 + 2.0 * (double)i / (alpha * den))));
        else
            w[i] = 1.0;
    }
    return TSR_OK;
}

static const fn_def DEFS[] = {
    ROUTINE("signal.convolve", 1, "a, v, mode='full'", "out", r_convolve, NULL, "1-D convolution of two sequences, modes full/same/valid (scipy.signal.convolve)."),
    ROUTINE("signal.lfilter", 1, "b, a, x", "out", r_lfilter, NULL, "Filter a 1-D signal with an IIR or FIR filter (scipy.signal.lfilter; zero initial state)."),
    ROUTINE("signal.sosfilt", 1, "sos, x, axis=-1, zi=None", "out", r_sosfilt, NULL, "Filter a 1-D signal through a cascade of second-order sections (scipy.signal.sosfilt; zero initial state)."),
    ROUTINE("signal.lfilter_zi", 1, "b, a", "out", r_lfilter_zi, NULL, "Steady-state step-response initial conditions for lfilter (scipy.signal.lfilter_zi)."),
    ROUTINE("signal.filtfilt", 1, "b, a, x, axis=-1, padtype='odd', padlen=None, method='pad', irlen=None", "out", r_filtfilt, NULL, "Zero-phase forward-backward IIR filtering with odd padding (scipy.signal.filtfilt; 1-D)."),
    ROUTINE("signal.sosfilt_zi", 1, "sos", "out", r_sosfilt_zi, NULL, "Per-section steady-state initial conditions for sosfilt (scipy.signal.sosfilt_zi)."),
    ROUTINE("signal.sosfiltfilt", 1, "sos, x, axis=-1, padtype='odd', padlen=None", "out", r_sosfiltfilt, NULL, "Zero-phase forward-backward SOS filtering with odd padding (scipy.signal.sosfiltfilt; 1-D)."),
    ROUTINE("signal.freqz", 2, "b, a=1, worN=512, whole=False", "w, h", r_freqz, NULL, "Frequency response of a digital filter (scipy.signal.freqz)."),
    ROUTINE("signal.zpk2tf", 2, "z, p, k", "b, a", r_zpk2tf, NULL, "Transfer-function (b, a) from zeros, poles and gain (scipy.signal.zpk2tf)."),
    ROUTINE("signal.butter", 2, "N, Wn, btype='low', analog=False, output='ba', fs=None", "b, a", r_butter, NULL, "Butterworth IIR filter design, digital lowpass/highpass, output 'ba' (scipy.signal.butter)."),
    ROUTINE("signal.cheby1", 2, "N, rp, Wn, btype='low', analog=False, output='ba', fs=None", "b, a", r_cheby1, NULL, "Chebyshev type I IIR filter design, digital lowpass/highpass, output 'ba' (scipy.signal.cheby1)."),
    ROUTINE("signal.cheby2", 2, "N, rs, Wn, btype='low', analog=False, output='ba', fs=None", "b, a", r_cheby2, NULL, "Chebyshev type II IIR filter design, digital lowpass/highpass, output 'ba' (scipy.signal.cheby2)."),
    ROUTINE("signal.fftconvolve", 1, "in1, in2, mode='full', axes=None", "out", r_fftconvolve, NULL, "N-D convolution of two arrays, modes full/same/valid (scipy.signal.fftconvolve; real)."),
    ROUTINE("signal.oaconvolve", 1, "in1, in2, mode='full', axes=None", "out", r_oaconvolve, NULL, "N-D convolution by the overlap-add method, modes full/same/valid (scipy.signal.oaconvolve; real)."),
    ROUTINE("signal.correlate", 1, "in1, in2, mode='full', method='auto'", "out", r_correlate, NULL, "N-D cross-correlation of two arrays, modes full/same/valid (scipy.signal.correlate; real)."),
    ROUTINE("signal.detrend", 1, "data, axis=-1, type='linear', bp=0, overwrite_data=False", "out", r_detrend, NULL, "Remove a constant or piecewise-linear least-squares trend along an axis (scipy.signal.detrend)."),
    ROUTINE("signal.savgol_coeffs", 1, "window_length, polyorder, deriv=0, delta=1.0, pos=None, use='conv'", "out", r_savgol_coeffs, NULL, "Savitzky-Golay filter coefficients (scipy.signal.savgol_coeffs)."),
    ROUTINE("signal.savgol_filter", 1, "x, window_length, polyorder, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0", "out", r_savgol_filter, NULL, "Apply a Savitzky-Golay filter to a 1-D signal (scipy.signal.savgol_filter)."),
    ROUTINE("windows.general_cosine", 1, "M, a, sym=True", "out", r_win_general_cosine, NULL, "Generic weighted sum of cosines window (scipy.signal.windows.general_cosine)."),
    ROUTINE("windows.general_hamming", 1, "M, alpha, sym=True", "out", r_win_general_hamming, NULL, "Generalized Hamming window (scipy.signal.windows.general_hamming)."),
    ROUTINE("windows.hann", 1, "M, sym=True", "out", r_win_hann, NULL, "Hann window (scipy.signal.windows.hann)."),
    ROUTINE("windows.hamming", 1, "M, sym=True", "out", r_win_hamming, NULL, "Hamming window (scipy.signal.windows.hamming)."),
    ROUTINE("windows.blackman", 1, "M, sym=True", "out", r_win_blackman, NULL, "Blackman window (scipy.signal.windows.blackman)."),
    ROUTINE("windows.blackmanharris", 1, "M, sym=True", "out", r_win_blackmanharris, NULL, "Minimum 4-term Blackman-Harris window (scipy.signal.windows.blackmanharris)."),
    ROUTINE("windows.nuttall", 1, "M, sym=True", "out", r_win_nuttall, NULL, "Nuttall's minimum 4-term Blackman-Harris window (scipy.signal.windows.nuttall)."),
    ROUTINE("windows.flattop", 1, "M, sym=True", "out", r_win_flattop, NULL, "Flat top window (scipy.signal.windows.flattop)."),
    ROUTINE("windows.boxcar", 1, "M, sym=True", "out", r_win_boxcar, NULL, "Rectangular window (scipy.signal.windows.boxcar)."),
    ROUTINE("windows.triang", 1, "M, sym=True", "out", r_win_triang, NULL, "Triangular window (scipy.signal.windows.triang)."),
    ROUTINE("windows.bartlett", 1, "M, sym=True", "out", r_win_bartlett, NULL, "Bartlett (triangular, zero-ended) window (scipy.signal.windows.bartlett)."),
    ROUTINE("windows.cosine", 1, "M, sym=True", "out", r_win_cosine, NULL, "Simple cosine (half-period sine) window (scipy.signal.windows.cosine)."),
    ROUTINE("windows.lanczos", 1, "M, sym=True", "out", r_win_lanczos, NULL, "Lanczos (sinc) window (scipy.signal.windows.lanczos)."),
    ROUTINE("windows.bohman", 1, "M, sym=True", "out", r_win_bohman, NULL, "Bohman window (scipy.signal.windows.bohman)."),
    ROUTINE("windows.barthann", 1, "M, sym=True", "out", r_win_barthann, NULL, "Bartlett-Hann window (scipy.signal.windows.barthann)."),
    ROUTINE("windows.parzen", 1, "M, sym=True", "out", r_win_parzen, NULL, "Parzen window (scipy.signal.windows.parzen)."),
    ROUTINE("windows.kaiser", 1, "M, beta, sym=True", "out", r_win_kaiser, NULL, "Kaiser window with shape beta (scipy.signal.windows.kaiser)."),
    ROUTINE("windows.gaussian", 1, "M, std, sym=True", "out", r_win_gaussian, NULL, "Gaussian window with standard deviation std (scipy.signal.windows.gaussian)."),
    ROUTINE("windows.general_gaussian", 1, "M, p, sig, sym=True", "out", r_win_general_gaussian, NULL, "Generalized Gaussian window (scipy.signal.windows.general_gaussian)."),
    ROUTINE("windows.exponential", 1, "M, center=None, tau=1.0, sym=True", "out", r_win_exponential, NULL, "Exponential (Poisson) window (scipy.signal.windows.exponential)."),
    ROUTINE("windows.tukey", 1, "M, alpha=0.5, sym=True", "out", r_win_tukey, NULL, "Tukey (tapered cosine) window (scipy.signal.windows.tukey)."),
};

const fn_table TSR_SCIPY_SIGNAL_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
