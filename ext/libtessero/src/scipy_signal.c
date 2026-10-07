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
extern int tsr_fft(int64_t n, int64_t rows, int inverse, const double *in, double *out);   /* core FFT (fft.c) */
extern int tsr_rfft(int64_t n, int64_t rows, const double *in, double *out);                /* real FFT -> n/2+1 bins */
extern int tsr_irfft(int64_t n, int64_t rows, const double *in, double *out);               /* inverse real FFT (1/n) */

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
/* corr2d_same: use scipy's correlate2d 'same' centering (offset s2/2) instead of the convolve/ correlate
   centering ((f-s1)/2); the two differ by one for even kernel dimensions. */
static int nd_convolve(const tsr_arg *A, const tsr_arg *V, int mode, int reverse_v, int corr2d_same, const char *who, tsr_result *res)
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
        else if (mode == CONV_SAME) { osh[d] = s1[d]; ost[d] = corr2d_same ? s2[d] / 2 : (f[d] - s1[d]) / 2; }
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
    return nd_convolve(&args[0], &args[1], mode, 0, 0, "fftconvolve", &res[0]);
}

/* oaconvolve(in1, in2, mode='full', axes=None): overlap-add convolution; same result as fftconvolve */
static int r_oaconvolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int mode, rc; if ((rc = parse_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    if ((rc = no_axes(args, nargs, 3, "oaconvolve")) < 0) return rc;
    return nd_convolve(&args[0], &args[1], mode, 0, 0, "oaconvolve", &res[0]);
}

/* correlate(in1, in2, mode='full', method='auto'): N-D cross-correlation (scipy.signal.correlate) */
static int r_correlate(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int mode, rc; if ((rc = parse_mode(nargs > 2 ? &args[2] : NULL, &mode)) < 0) return rc;
    return nd_convolve(&args[0], &args[1], mode, 1, 0, "correlate", &res[0]);
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

/* ---- waveform generators (scipy.signal) ---- */

/* square(t, duty=0.5): +1 where (t mod 2pi) < duty*2pi, else -1; NaN if duty is outside [0, 1]. */
static int r_square(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("square: t must be an array"); return TSR_EARG; }
    int64_t n; double *t = fn_arg_doubles(&args[0], &n); if (!t) return TSR_ENOMEM;
    const double duty = (nargs > 1 && args[1].kind == 1) ? args[1].num : 0.5;
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){n});
    int rc = TSR_OK;
    if (!y) rc = TSR_ENOMEM;
    else {
        const double tp = 2.0 * M_PI;
        const int bad = (duty > 1.0 || duty < 0.0);
        for (int64_t i = 0; i < n; i++) {
            if (bad) { y[i] = FN_NAN; continue; }
            double tm = fmod(t[i], tp); if (tm < 0.0) tm += tp;
            y[i] = (tm < duty * tp) ? 1.0 : -1.0;
        }
    }
    fn_free_doubles(t, n);
    return rc;
}

/* sawtooth(t, width=1): rising ramp over [0, width*2pi) then falling; NaN if width outside [0, 1]. */
static int r_sawtooth(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("sawtooth: t must be an array"); return TSR_EARG; }
    int64_t n; double *t = fn_arg_doubles(&args[0], &n); if (!t) return TSR_ENOMEM;
    const double w = (nargs > 1 && args[1].kind == 1) ? args[1].num : 1.0;
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){n});
    int rc = TSR_OK;
    if (!y) rc = TSR_ENOMEM;
    else {
        const double tp = 2.0 * M_PI;
        const int bad = (w > 1.0 || w < 0.0);
        for (int64_t i = 0; i < n; i++) {
            if (bad) { y[i] = FN_NAN; continue; }
            double tm = fmod(t[i], tp); if (tm < 0.0) tm += tp;
            y[i] = (tm < w * tp) ? (tm / (M_PI * w) - 1.0) : ((M_PI * (w + 1.0) - tm) / (M_PI * (1.0 - w)));
        }
    }
    fn_free_doubles(t, n);
    return rc;
}

/* chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True): cos(phase(t) + pi*phi/180). */
static int r_chirp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("chirp: t must be an array"); return TSR_EARG; }
    int64_t n; double *t = fn_arg_doubles(&args[0], &n); if (!t) return TSR_ENOMEM;
    const double f0 = args[1].num, t1 = args[2].num, f1 = args[3].num;
    const char *method = (nargs > 4 && args[4].kind == 2 && args[4].str) ? args[4].str : "linear";
    const double phi = (nargs > 5 && args[5].kind == 1) ? args[5].num : 0.0;
    const int vertex_zero = !(nargs > 6 && (args[6].kind == 4 || args[6].kind == 1) && args[6].num == 0.0);
    int meth = 0;                                            /* 0 linear, 1 quadratic, 2 log, 3 hyperbolic */
    if (!strcmp(method, "quadratic") || !strcmp(method, "quad") || !strcmp(method, "q")) meth = 1;
    else if (!strcmp(method, "logarithmic") || !strcmp(method, "log") || !strcmp(method, "lo")) meth = 2;
    else if (!strcmp(method, "hyperbolic") || !strcmp(method, "hyp")) meth = 3;
    else if (strcmp(method, "linear") && strcmp(method, "lin") && strcmp(method, "li")) { fn_free_doubles(t, n); fn_set_error("chirp: unknown method"); return TSR_EARG; }
    if ((meth == 2 && f0 * f1 <= 0.0) || (meth == 3 && (f0 == 0.0 || f1 == 0.0))) { fn_free_doubles(t, n); fn_set_error("chirp: f0, f1 invalid for this method"); return TSR_EARG; }
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){n});
    int rc = TSR_OK;
    if (!y) rc = TSR_ENOMEM;
    else {
        const double phioff = M_PI * phi / 180.0;
        const double beta_l = (f1 - f0) / t1, beta_q = (f1 - f0) / (t1 * t1);
        const double sing = (meth == 3 && f0 != f1) ? -f1 * t1 / (f0 - f1) : 0.0;
        const double beta_log = (meth == 2 && f0 != f1) ? t1 / log(f1 / f0) : 0.0;
        for (int64_t i = 0; i < n; i++) {
            const double x = t[i]; double ph;
            if (meth == 0) ph = 2.0 * M_PI * (f0 * x + 0.5 * beta_l * x * x);
            else if (meth == 1) ph = vertex_zero ? 2.0 * M_PI * (f0 * x + beta_q * x * x * x / 3.0)
                                                  : 2.0 * M_PI * (f1 * x + beta_q * ((t1 - x) * (t1 - x) * (t1 - x) - t1 * t1 * t1) / 3.0);
            else if (meth == 2) ph = (f0 == f1) ? 2.0 * M_PI * f0 * x : 2.0 * M_PI * beta_log * f0 * (pow(f1 / f0, x / t1) - 1.0);
            else ph = (f0 == f1) ? 2.0 * M_PI * f0 * x : 2.0 * M_PI * (-sing * f0) * log(fabs(1.0 - x / sing));
            y[i] = cos(ph + phioff);
        }
    }
    fn_free_doubles(t, n);
    return rc;
}

/* gausspulse(t, fc=1000, bw=0.5, bwr=-6): the real (in-phase) Gaussian-modulated sinusoid (default outputs). */
static int r_gausspulse(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("gausspulse: t must be an array (string 'cutoff' is not supported)"); return TSR_EARG; }
    int64_t n; double *t = fn_arg_doubles(&args[0], &n); if (!t) return TSR_ENOMEM;
    const double fc = (nargs > 1 && args[1].kind == 1) ? args[1].num : 1000.0;
    const double bw = (nargs > 2 && args[2].kind == 1) ? args[2].num : 0.5;
    const double bwr = (nargs > 3 && args[3].kind == 1) ? args[3].num : -6.0;
    int rc = TSR_OK;
    if (fc < 0.0 || bw <= 0.0 || bwr >= 0.0) { fn_free_doubles(t, n); fn_set_error("gausspulse: need fc>=0, bw>0, bwr<0"); return TSR_EARG; }
    const double ref = pow(10.0, bwr / 20.0);
    const double a = -(M_PI * fc * bw) * (M_PI * fc * bw) / (4.0 * log(ref));
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){n});
    if (!y) rc = TSR_ENOMEM;
    else for (int64_t i = 0; i < n; i++) y[i] = exp(-a * t[i] * t[i]) * cos(2.0 * M_PI * fc * t[i]);
    fn_free_doubles(t, n);
    return rc;
}

/* unit_impulse(shape, idx=None): a length-`shape` array of zeros with a 1 at idx (int, or 'mid'; default 0). */
static int r_unit_impulse(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 1) { fn_set_error("unit_impulse: shape must be an integer length"); return TSR_EARG; }
    const int64_t n = (int64_t)args[0].num;
    if (n < 0) { fn_set_error("unit_impulse: shape must be non-negative"); return TSR_EARG; }
    int64_t idx = 0;
    if (nargs > 1) {
        if (args[1].kind == 2 && args[1].str && !strcmp(args[1].str, "mid")) idx = n / 2;
        else if (args[1].kind == 1) { idx = (int64_t)args[1].num; if (idx < 0) idx += n; }   /* numpy negative indexing */
    }
    double *y = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){n});
    if (!y) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++) y[i] = 0.0;
    if (idx >= 0 && idx < n) y[idx] = 1.0;
    return TSR_OK;
}

/* ---- transfer-function / zpk / sos conversions (scipy.signal) ---- */

static int cmp_double_sig(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static int sig_cplx_cmp(const void *a, const void *b)
{
    const double *x = (const double *)a, *y = (const double *)b;
    if (x[0] < y[0]) return -1; if (x[0] > y[0]) return 1;
    if (x[1] < y[1]) return -1; if (x[1] > y[1]) return 1;
    return 0;
}

/* Roots of a real polynomial p (highest-degree first, length m) as a malloc'd interleaved-complex buffer sorted
   by (re, im); *nz gets the count (0 for a constant, -1 on failure). Companion-matrix eigenvalues, as numpy.roots. */
static double *sig_polyroots(const double *p, int64_t m, int64_t *nz)
{
    int64_t s = 0; while (s < m && p[s] == 0.0) s++;         /* strip leading zeros */
    const int64_t n = (m - s) - 1;
    if (n <= 0) { *nz = 0; return NULL; }
    double *comp = (double *)calloc((size_t)(n * n), sizeof(double));
    double *wr = (double *)malloc(sizeof(double) * (size_t)n), *wi = (double *)malloc(sizeof(double) * (size_t)n);
    double *pairs = (double *)malloc(sizeof(double) * (size_t)(2 * n));
    if (!comp || !wr || !wi || !pairs) { free(comp); free(wr); free(wi); free(pairs); *nz = -1; return NULL; }
    const double p0 = p[s];
    for (int64_t j = 0; j < n; j++) comp[j] = -p[s + 1 + j] / p0;
    for (int64_t i = 1; i < n; i++) comp[i * n + (i - 1)] = 1.0;
    lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)n, comp, (lapack_int)n, wr, wi, NULL, 1, NULL, 1);
    free(comp);
    if (info != 0) { free(wr); free(wi); free(pairs); *nz = -1; return NULL; }
    for (int64_t i = 0; i < n; i++) { pairs[2 * i] = wr[i]; pairs[2 * i + 1] = wi[i]; }
    free(wr); free(wi);
    qsort(pairs, (size_t)n, 2 * sizeof(double), sig_cplx_cmp);
    *nz = n;
    return pairs;
}

/* normalize(b, a) for 1-D b, a: trim a's leading exact zeros (keep >=1), scale both by a[0], then trim b's
   leading entries with |.| <= 1e-14 (keep >=1). Writes b2/a2 into freshly malloc'd buffers; returns 0/-1. */
static int sig_normalize(const double *b, int64_t nb, const double *a, int64_t na,
                         double **b2, int64_t *nb2, double **a2, int64_t *na2)
{
    int64_t as = 0; while (as < na - 1 && a[as] == 0.0) as++;
    const double a0 = a[as];
    if (a0 == 0.0) return -1;
    const int64_t alen = na - as;
    double *bn = (double *)malloc(sizeof(double) * (size_t)(nb > 0 ? nb : 1));
    double *an = (double *)malloc(sizeof(double) * (size_t)alen);
    if (!bn || !an) { free(bn); free(an); return -1; }
    for (int64_t i = 0; i < nb; i++) bn[i] = b[i] / a0;
    for (int64_t i = 0; i < alen; i++) an[i] = a[as + i] / a0;
    int64_t bs = 0; while (bs < nb - 1 && fabs(bn[bs]) <= 1e-14) bs++;
    const int64_t blen = nb - bs;
    double *bt = (double *)malloc(sizeof(double) * (size_t)(blen > 0 ? blen : 1));
    if (!bt) { free(bn); free(an); return -1; }
    for (int64_t i = 0; i < blen; i++) bt[i] = bn[bs + i];
    free(bn);
    *b2 = bt; *nb2 = blen; *a2 = an; *na2 = alen;
    return 0;
}

/* normalize(b, a): return the normalized (b, a) of a transfer function (scipy.signal.normalize; 1-D). */
static int r_signal_normalize(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("normalize: b and a must be arrays (a 1-D)"); return TSR_EARG; }
    int64_t nb, na; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    double *b2 = NULL, *a2 = NULL; int64_t nb2 = 0, na2 = 0;
    int rc = TSR_OK;
    if (sig_normalize(b, nb, a, na, &b2, &nb2, &a2, &na2) != 0) { fn_set_error("normalize: denominator has no nonzero element"); rc = TSR_EARG; }
    else {
        double *ob = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nb2});
        double *oa = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){na2});
        if (!ob || !oa) rc = TSR_ENOMEM; else { memcpy(ob, b2, sizeof(double) * (size_t)nb2); memcpy(oa, a2, sizeof(double) * (size_t)na2); }
    }
    free(b2); free(a2);
    fn_free_doubles(b, nb); fn_free_doubles(a, na);
    return rc;
}

/* tf2zpk(b, a): zeros, poles and gain of a transfer function (scipy.signal.tf2zpk). Roots sorted by (re, im). */
static int r_tf2zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("tf2zpk: b and a must be arrays (a 1-D)"); return TSR_EARG; }
    int64_t nb, na; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    double *b2 = NULL, *a2 = NULL; int64_t nb2 = 0, na2 = 0;
    int rc = TSR_OK;
    if (sig_normalize(b, nb, a, na, &b2, &nb2, &a2, &na2) != 0) { fn_set_error("tf2zpk: normalize failed"); rc = TSR_EARG; }
    else {
        const double k = b2[0];
        for (int64_t i = 0; i < nb2; i++) b2[i] /= k;        /* monic numerator for roots */
        int64_t nz = 0, npz = 0;
        double *z = sig_polyroots(b2, nb2, &nz);
        double *p = sig_polyroots(a2, na2, &npz);
        if (nz < 0 || npz < 0) { fn_set_error("tf2zpk: root solve failed"); rc = TSR_EARG; }
        else {
            double *oz = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){nz});
            double *op = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){npz});
            if (!oz || !op) rc = TSR_ENOMEM;
            else { if (nz) memcpy(oz, z, sizeof(double) * (size_t)(2 * nz)); if (npz) memcpy(op, p, sizeof(double) * (size_t)(2 * npz)); fn_result_num(&res[2], k); }
        }
        free(z); free(p);
    }
    free(b2); free(a2);
    fn_free_doubles(b, nb); fn_free_doubles(a, na);
    return rc;
}

/* sos2zpk(sos): zeros, poles and gain of a second-order-sections cascade (scipy.signal.sos2zpk). Each section's
   roots are found via tf2zpk; z and p (length 2*n_sections, with zero padding where a section is lower order)
   are returned sorted by (re, im) -- a zpk set is order-free, and the fixtures sort the same way. */
static int r_sos2zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[1] != 6) { fn_set_error("sos2zpk: sos must be an (n, 6) array"); return TSR_EARG; }
    const int64_t nsec = args[0].arr.shape[0];
    int64_t ns; double *sos = fn_arg_doubles(&args[0], &ns); if (!sos) return TSR_ENOMEM;
    const int64_t nn = 2 * nsec;
    double *z = (double *)calloc((size_t)(2 * (nn ? nn : 1)), sizeof(double));
    double *p = (double *)calloc((size_t)(2 * (nn ? nn : 1)), sizeof(double));
    int rc = TSR_OK;
    double k = 1.0;
    for (int64_t s = 0; s < nsec && rc == TSR_OK; s++) {
        const double *sb = sos + s * 6, *sa = sos + s * 6 + 3;
        double *bn = NULL, *an = NULL; int64_t nbn = 0, nan = 0;
        if (sig_normalize(sb, 3, sa, 3, &bn, &nbn, &an, &nan) != 0) { rc = TSR_EARG; fn_set_error("sos2zpk: a section's a is zero"); break; }
        const double ks = bn[0];
        for (int64_t i = 0; i < nbn; i++) bn[i] /= (ks != 0.0 ? ks : 1.0);
        int64_t nz = 0, np_ = 0;
        double *zr = sig_polyroots(bn, nbn, &nz), *pr = sig_polyroots(an, nan, &np_);
        if (nz < 0 || np_ < 0) { rc = TSR_EARG; fn_set_error("sos2zpk: root solve failed"); }
        else {
            for (int64_t i = 0; i < nz && i < 2; i++) { z[2 * (2 * s + i)] = zr[2 * i]; z[2 * (2 * s + i) + 1] = zr[2 * i + 1]; }
            for (int64_t i = 0; i < np_ && i < 2; i++) { p[2 * (2 * s + i)] = pr[2 * i]; p[2 * (2 * s + i) + 1] = pr[2 * i + 1]; }
            k *= ks;
        }
        free(zr); free(pr); free(bn); free(an);
    }
    if (rc == TSR_OK) {
        qsort(z, (size_t)(nn ? nn : 1), 2 * sizeof(double), sig_cplx_cmp);
        qsort(p, (size_t)(nn ? nn : 1), 2 * sizeof(double), sig_cplx_cmp);
        double *oz = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){nn});
        double *op = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){nn});
        if (!oz || !op) rc = TSR_ENOMEM;
        else { memcpy(oz, z, sizeof(double) * (size_t)(2 * nn)); memcpy(op, p, sizeof(double) * (size_t)(2 * nn)); fn_result_num(&res[2], k); }
    }
    free(z); free(p); fn_free_doubles(sos, ns);
    return rc;
}

/* sos2tf(sos): transfer function (b, a) of a second-order-sections cascade (scipy.signal.sos2tf). */
static int r_sos2tf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[1] != 6) { fn_set_error("sos2tf: sos must be an (n, 6) array"); return TSR_EARG; }
    const int64_t nsec = args[0].arr.shape[0];
    int64_t ns; double *sos = fn_arg_doubles(&args[0], &ns); if (!sos) return TSR_ENOMEM;
    const int64_t len = 2 * nsec + 1;
    double *b = (double *)calloc((size_t)len, sizeof(double));
    double *a = (double *)calloc((size_t)len, sizeof(double));
    double *tb = (double *)calloc((size_t)len, sizeof(double));
    double *ta = (double *)calloc((size_t)len, sizeof(double));
    int rc = TSR_OK;
    if (!b || !a || !tb || !ta) rc = TSR_ENOMEM;
    else {
        b[0] = 1.0; a[0] = 1.0; int64_t blen = 1, alen = 1;  /* current polynomial lengths */
        for (int64_t sct = 0; sct < nsec; sct++) {
            const double *sb = sos + sct * 6, *sa = sos + sct * 6 + 3;
            for (int64_t i = 0; i < blen + 2; i++) tb[i] = 0.0;
            for (int64_t i = 0; i < alen + 2; i++) ta[i] = 0.0;
            for (int64_t i = 0; i < blen; i++) for (int64_t j = 0; j < 3; j++) tb[i + j] += b[i] * sb[j];
            for (int64_t i = 0; i < alen; i++) for (int64_t j = 0; j < 3; j++) ta[i + j] += a[i] * sa[j];
            blen += 2; alen += 2;
            memcpy(b, tb, sizeof(double) * (size_t)blen);
            memcpy(a, ta, sizeof(double) * (size_t)alen);
        }
        double *ob = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){len});
        double *oa = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){len});
        if (!ob || !oa) rc = TSR_ENOMEM; else { memcpy(ob, b, sizeof(double) * (size_t)len); memcpy(oa, a, sizeof(double) * (size_t)len); }
    }
    free(b); free(a); free(tb); free(ta);
    fn_free_doubles(sos, ns);
    return rc;
}

/* ---- state-space <-> transfer function (scipy.signal) ---- */

/* Characteristic polynomial of an n*n row-major matrix, matching numpy.poly(A): eigenvalues via dgeev, then
   the monic polynomial prod(x - lambda_i) formed by complex convolution, real part kept. out has length n+1.
   Returns 0 on success, -1 on allocation/LAPACK failure. */
static int ss_charpoly(const double *M, int64_t n, double *out)
{
    out[0] = 1.0;
    if (n == 0) return 0;
    double *a = (double *)malloc(sizeof(double) * (size_t)(n * n));
    double *wr = (double *)malloc(sizeof(double) * (size_t)n);
    double *wi = (double *)malloc(sizeof(double) * (size_t)n);
    double *cr = (double *)calloc((size_t)(n + 1), sizeof(double));
    double *ci = (double *)calloc((size_t)(n + 1), sizeof(double));
    if (!a || !wr || !wi || !cr || !ci) { free(a); free(wr); free(wi); free(cr); free(ci); return -1; }
    memcpy(a, M, sizeof(double) * (size_t)(n * n));
    lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', (lapack_int)n, a, (lapack_int)n, wr, wi, NULL, 1, NULL, 1);
    free(a);
    if (info != 0) { free(wr); free(wi); free(cr); free(ci); return -1; }
    cr[0] = 1.0; ci[0] = 0.0;
    int64_t L = 1;                                            /* current coefficient count */
    for (int64_t i = 0; i < n; i++) {
        const double zr = wr[i], zi = wi[i];
        cr[L] = 0.0; ci[L] = 0.0;
        for (int64_t k = L; k >= 1; k--) {                    /* c[k] = c[k] - z*c[k-1], high->low */
            const double pr = cr[k - 1], pi = ci[k - 1];
            cr[k] -= zr * pr - zi * pi;
            ci[k] -= zr * pi + zi * pr;
        }
        L++;
    }
    for (int64_t k = 0; k <= n; k++) out[k] = cr[k];          /* conjugate-symmetric roots -> real */
    free(wr); free(wi); free(cr); free(ci);
    return 0;
}

/* tf2ss(num, den): controller-canonical state-space (A, B, C, D) of a SISO transfer function
   (scipy.signal.tf2ss; 1-D num/den). */
static int r_tf2ss(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 3 || args[0].arr.ndim != 1 || args[1].arr.ndim != 1) {
        fn_set_error("tf2ss: num and den must be 1-D arrays"); return TSR_EARG; }
    int64_t nb, na; double *num = fn_arg_doubles(&args[0], &nb); if (!num) return TSR_ENOMEM;
    double *den = fn_arg_doubles(&args[1], &na); if (!den) { fn_free_doubles(num, nb); return TSR_ENOMEM; }
    double *b2 = NULL, *a2 = NULL; int64_t M = 0, K = 0;
    int rc = TSR_OK;
    if (sig_normalize(num, nb, den, na, &b2, &M, &a2, &K) != 0) { fn_set_error("tf2ss: normalize failed"); rc = TSR_EARG; }
    else if (M > K) { fn_set_error("tf2ss: improper transfer function (num longer than den)"); rc = TSR_EARG; }
    else {
        /* pad numerator on the left to length K */
        double *np_ = (double *)calloc((size_t)(K > 0 ? K : 1), sizeof(double));
        if (!np_) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < M; i++) np_[K - M + i] = b2[i];
            const double D0 = np_[0];
            if (K <= 1) {                                     /* null / gain-only system */
                double *A = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){1, 1});
                double *B = (double *)fn_result_array(&res[1], TSR_F64, 2, (int64_t[]){1, 1});
                double *C = (double *)fn_result_array(&res[2], TSR_F64, 2, (int64_t[]){1, 1});
                double *D = (double *)fn_result_array(&res[3], TSR_F64, 2, (int64_t[]){1, 1});
                if (!A || !B || !C || !D) rc = TSR_ENOMEM;
                else { A[0] = 0.0; B[0] = 0.0; C[0] = 0.0; D[0] = D0; }
            } else {
                const int64_t m = K - 1;                      /* number of states */
                double *A = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){m, m});
                double *B = (double *)fn_result_array(&res[1], TSR_F64, 2, (int64_t[]){m, 1});
                double *C = (double *)fn_result_array(&res[2], TSR_F64, 2, (int64_t[]){1, m});
                double *D = (double *)fn_result_array(&res[3], TSR_F64, 2, (int64_t[]){1, 1});
                if (!A || !B || !C || !D) rc = TSR_ENOMEM;
                else {
                    for (int64_t j = 0; j < m; j++) A[j] = -a2[1 + j];          /* row 0 = -den[1:] */
                    for (int64_t r = 1; r < m; r++) for (int64_t c = 0; c < m; c++) A[r * m + c] = (c == r - 1) ? 1.0 : 0.0;
                    B[0] = 1.0; for (int64_t r = 1; r < m; r++) B[r] = 0.0;
                    for (int64_t j = 0; j < m; j++) C[j] = np_[1 + j] - np_[0] * a2[1 + j];
                    D[0] = D0;
                }
            }
            free(np_);
        }
    }
    free(b2); free(a2);
    fn_free_doubles(num, nb); fn_free_doubles(den, na);
    return rc;
}

/* abcd_normalize(A, B, C, D): validate state-space matrix shapes and return them as 2-D arrays
   (scipy.signal.abcd_normalize; all four matrices supplied, 2-D). */
static int r_abcd_normalize(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    for (int i = 0; i < 4; i++) if (args[i].kind != 3 || args[i].arr.ndim != 2) {
        fn_set_error("abcd_normalize: A, B, C, D must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    const int64_t p = args[1].arr.shape[1];
    const int64_t q = args[2].arr.shape[0];
    if (args[0].arr.shape[1] != n) { fn_set_error("abcd_normalize: A must be square"); return TSR_EARG; }
    if (args[1].arr.shape[0] != n) { fn_set_error("abcd_normalize: A and B must agree on the number of states"); return TSR_EARG; }
    if (args[2].arr.shape[1] != n) { fn_set_error("abcd_normalize: A and C must agree on the number of states"); return TSR_EARG; }
    if (args[3].arr.shape[0] != q || args[3].arr.shape[1] != p) { fn_set_error("abcd_normalize: C and D (or B and D) are inconsistent"); return TSR_EARG; }
    const int64_t shp[4][2] = {{n, n}, {n, p}, {q, n}, {q, p}};
    int rc = TSR_OK;
    for (int i = 0; i < 4 && rc == TSR_OK; i++) {
        int64_t cnt; double *src = fn_arg_doubles(&args[i], &cnt);
        if (!src) { rc = TSR_ENOMEM; break; }
        double *out = (double *)fn_result_array(&res[i], TSR_F64, 2, (int64_t[]){shp[i][0], shp[i][1]});
        if (!out) rc = TSR_ENOMEM; else memcpy(out, src, sizeof(double) * (size_t)cnt);
        fn_free_doubles(src, cnt);
    }
    return rc;
}

/* ss2tf(A, B, C, D, input=0): transfer function (num, den) of a state-space system, selecting one input
   (scipy.signal.ss2tf). num has one row per output; den = poly(A). */
static int r_ss2tf(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    for (int i = 0; i < 4; i++) if (args[i].kind != 3 || args[i].arr.ndim != 2) {
        fn_set_error("ss2tf: A, B, C, D must be 2-D arrays"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0];
    const int64_t p = args[3].arr.shape[1];
    const int64_t q = args[3].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[2].arr.shape[1] != n ||
        args[1].arr.shape[1] != p || args[2].arr.shape[0] != q) { fn_set_error("ss2tf: inconsistent state-space shapes"); return TSR_EARG; }
    int64_t inp = 0;
    if (nargs > 4 && args[4].kind == 1) inp = (int64_t)args[4].num;
    if (inp < 0 || inp >= p) { fn_set_error("ss2tf: system does not have the input specified"); return TSR_EARG; }
    int64_t na_, nb_, nc_, nd_;
    double *A = fn_arg_doubles(&args[0], &na_); double *B = fn_arg_doubles(&args[1], &nb_);
    double *C = fn_arg_doubles(&args[2], &nc_); double *D = fn_arg_doubles(&args[3], &nd_);
    int rc = TSR_OK;
    double *den = (double *)malloc(sizeof(double) * (size_t)(n + 1));
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(n * n > 0 ? n * n : 1));
    double *numk = (double *)malloc(sizeof(double) * (size_t)(n + 1));
    if (!A || !B || !C || !D || !den || !tmp || !numk) rc = TSR_ENOMEM;
    else if (ss_charpoly(A, n, den) != 0) { fn_set_error("ss2tf: eigenvalue computation failed"); rc = TSR_EARG; }
    else {
        double *num = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){q, n + 1});
        double *od  = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){n + 1});
        if (!num || !od) rc = TSR_ENOMEM;
        else {
            memcpy(od, den, sizeof(double) * (size_t)(n + 1));
            for (int64_t k = 0; k < q && rc == TSR_OK; k++) {
                /* tmp = A - B[:,inp] (outer) C[k,:] */
                for (int64_t i = 0; i < n; i++)
                    for (int64_t j = 0; j < n; j++)
                        tmp[i * n + j] = A[i * n + j] - B[i * p + inp] * C[k * n + j];
                if (ss_charpoly(tmp, n, numk) != 0) { fn_set_error("ss2tf: eigenvalue computation failed"); rc = TSR_EARG; break; }
                const double dk = D[k * p + inp];
                for (int64_t j = 0; j <= n; j++) num[k * (n + 1) + j] = numk[j] + (dk - 1.0) * den[j];
            }
        }
    }
    free(den); free(tmp); free(numk);
    fn_free_doubles(A, na_); fn_free_doubles(B, nb_); fn_free_doubles(C, nc_); fn_free_doubles(D, nd_);
    return rc;
}

extern int sl_expm(const double *A, int64_t n, double *out);                 /* np_linalg.c */
extern int sl_dense_solve(double *M, double *b, int64_t n, int64_t nrhs);    /* np_linalg.c (row-major dgesv) */

/* out(xr x yc) = X(xr x xc) @ Y(xc x yc), row-major */
static void sig_mm(const double *X, int64_t xr, int64_t xc, const double *Y, int64_t yc, double *out)
{
    for (int64_t i = 0; i < xr; i++) for (int64_t j = 0; j < yc; j++) { double s = 0.0; for (int64_t k = 0; k < xc; k++) s += X[i * xc + k] * Y[k * yc + j]; out[i * yc + j] = s; }
}

/* cont2discrete((A,B,C,D), dt, method='zoh', alpha=None): convert a continuous state-space system to discrete
   (scipy.signal.cont2discrete), state-space form. Methods: zoh, gbt (needs alpha), bilinear/tustin (alpha=0.5),
   euler/forward_diff (alpha=0), backward_diff (alpha=1). Returns (Ad, Bd, Cd, Dd, dt). */
static int r_cont2discrete(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    for (int i = 0; i < 4; i++) if (args[i].kind != 3 || args[i].arr.ndim != 2) { fn_set_error("cont2discrete: A, B, C, D must be 2-D arrays"); return TSR_EARG; }
    if (args[4].kind != 1) { fn_set_error("cont2discrete: dt must be a number"); return TSR_EARG; }
    const int64_t n = args[0].arr.shape[0], m = args[1].arr.shape[1], q = args[2].arr.shape[0];
    if (args[0].arr.shape[1] != n || args[1].arr.shape[0] != n || args[2].arr.shape[1] != n ||
        args[3].arr.shape[0] != q || args[3].arr.shape[1] != m) { fn_set_error("cont2discrete: inconsistent state-space shapes"); return TSR_EARG; }
    const double dt = args[4].num;
    const char *method = (nargs > 5 && args[5].kind == 2 && args[5].str) ? args[5].str : "zoh";
    int64_t na, nb, nc, nd;
    double *A = fn_arg_doubles(&args[0], &na), *B = fn_arg_doubles(&args[1], &nb);
    double *C = fn_arg_doubles(&args[2], &nc), *D = fn_arg_doubles(&args[3], &nd);
    int rc = TSR_OK;
    if (!A || !B || !C || !D) rc = TSR_ENOMEM;
    else if (!strcmp(method, "zoh")) {
        const int64_t N = n + m;
        double *M = (double *)calloc((size_t)(N * N), sizeof(double));
        double *EM = (double *)malloc(sizeof(double) * (size_t)(N * N));
        if (!M || !EM) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < n; i++) { for (int64_t j = 0; j < n; j++) M[i * N + j] = dt * A[i * n + j]; for (int64_t j = 0; j < m; j++) M[i * N + n + j] = dt * B[i * m + j]; }
            if (sl_expm(M, N, EM) != 0) { fn_set_error("cont2discrete: the matrix exponential failed"); rc = TSR_EARG; }
            else {
                double *Ad = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
                double *Bd = (double *)fn_result_array(&res[1], TSR_F64, 2, (int64_t[]){n, m});
                double *Cd = (double *)fn_result_array(&res[2], TSR_F64, 2, (int64_t[]){q, n});
                double *Dd = (double *)fn_result_array(&res[3], TSR_F64, 2, (int64_t[]){q, m});
                if (!Ad || !Bd || !Cd || !Dd) rc = TSR_ENOMEM;
                else {
                    for (int64_t i = 0; i < n; i++) { for (int64_t j = 0; j < n; j++) Ad[i * n + j] = EM[i * N + j]; for (int64_t j = 0; j < m; j++) Bd[i * m + j] = EM[i * N + n + j]; }
                    memcpy(Cd, C, sizeof(double) * (size_t)(q * n)); memcpy(Dd, D, sizeof(double) * (size_t)(q * m));
                    fn_result_num(&res[4], dt);
                }
            }
        }
        free(M); free(EM);
    } else {
        double alpha;
        if (!strcmp(method, "gbt")) { if (nargs <= 6 || args[6].kind != 1) { fn_set_error("cont2discrete: alpha must be specified for the gbt method"); rc = TSR_EARG; alpha = 0.0; } else alpha = args[6].num; }
        else if (!strcmp(method, "bilinear") || !strcmp(method, "tustin")) alpha = 0.5;
        else if (!strcmp(method, "euler") || !strcmp(method, "forward_diff")) alpha = 0.0;
        else if (!strcmp(method, "backward_diff")) alpha = 1.0;
        else { fn_set_error("cont2discrete: unknown or unsupported method '%s'", method); rc = TSR_EARG; alpha = 0.0; }
        if (rc == TSR_OK) {
            double *ima = (double *)malloc(sizeof(double) * (size_t)(n * n));
            double *imac = (double *)malloc(sizeof(double) * (size_t)(n * n));
            double *imaT = (double *)malloc(sizeof(double) * (size_t)(n * n));
            double *CT = (double *)malloc(sizeof(double) * (size_t)(n * q));
            double *Cbd = (double *)malloc(sizeof(double) * (size_t)(q * m));
            double *Ad = (double *)fn_result_array(&res[0], TSR_F64, 2, (int64_t[]){n, n});
            double *Bd = (double *)fn_result_array(&res[1], TSR_F64, 2, (int64_t[]){n, m});
            double *Cd = (double *)fn_result_array(&res[2], TSR_F64, 2, (int64_t[]){q, n});
            double *Dd = (double *)fn_result_array(&res[3], TSR_F64, 2, (int64_t[]){q, m});
            if (!ima || !imac || !imaT || !CT || !Cbd || !Ad || !Bd || !Cd || !Dd) rc = TSR_ENOMEM;
            else {
                for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) ima[i * n + j] = (i == j ? 1.0 : 0.0) - alpha * dt * A[i * n + j];
                /* Ad = solve(ima, I + (1-alpha)*dt*A) */
                for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) Ad[i * n + j] = (i == j ? 1.0 : 0.0) + (1.0 - alpha) * dt * A[i * n + j];
                memcpy(imac, ima, sizeof(double) * (size_t)(n * n));
                if (sl_dense_solve(imac, Ad, n, n) != 0) { fn_set_error("cont2discrete: (I - alpha*dt*A) is singular"); rc = TSR_EARG; }
                /* Bd = solve(ima, dt*B) */
                if (rc == TSR_OK) { for (int64_t i = 0; i < n * m; i++) Bd[i] = dt * B[i]; memcpy(imac, ima, sizeof(double) * (size_t)(n * n)); if (sl_dense_solve(imac, Bd, n, m) != 0) { fn_set_error("cont2discrete: singular system"); rc = TSR_EARG; } }
                /* Cd = solve(ima^T, C^T)^T */
                if (rc == TSR_OK) {
                    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < n; j++) imaT[i * n + j] = ima[j * n + i];
                    for (int64_t i = 0; i < n; i++) for (int64_t j = 0; j < q; j++) CT[i * q + j] = C[j * n + i];
                    if (sl_dense_solve(imaT, CT, n, q) != 0) { fn_set_error("cont2discrete: singular system"); rc = TSR_EARG; }
                    else { for (int64_t i = 0; i < q; i++) for (int64_t j = 0; j < n; j++) Cd[i * n + j] = CT[j * q + i]; }
                }
                /* Dd = D + alpha * C @ Bd */
                if (rc == TSR_OK) { sig_mm(C, q, n, Bd, m, Cbd); for (int64_t i = 0; i < q * m; i++) Dd[i] = D[i] + alpha * Cbd[i]; fn_result_num(&res[4], dt); }
            }
            free(ima); free(imac); free(imaT); free(CT); free(Cbd);
        }
    }
    fn_free_doubles(A, na); fn_free_doubles(B, nb); fn_free_doubles(C, nc); fn_free_doubles(D, nd);
    return rc;
}

/* ---- partial-fraction expansion (scipy.signal) ---- */

enum { RT_MIN, RT_MAX, RT_AVG };
static int sig_rtype(const tsr_arg *a, int dflt)
{
    if (!a || a->kind != 2 || !a->str) return dflt;
    if (!strcmp(a->str, "max") || !strcmp(a->str, "maximum")) return RT_MAX;
    if (!strcmp(a->str, "min") || !strcmp(a->str, "minimum")) return RT_MIN;
    if (!strcmp(a->str, "avg") || !strcmp(a->str, "mean")) return RT_AVG;
    return dflt;
}

/* reduce a group of complex values per rtype (min/max compare real then imag; avg = mean) */
static double complex sig_creduce(const double complex *v, int64_t n, int rt)
{
    if (rt == RT_AVG) { double complex s = 0; for (int64_t i = 0; i < n; i++) s += v[i]; return s / (double)n; }
    double complex best = v[0];
    for (int64_t i = 1; i < n; i++) {
        const int gt = (creal(v[i]) > creal(best)) || (creal(v[i]) == creal(best) && cimag(v[i]) > cimag(best));
        if ((rt == RT_MAX) == (gt != 0) && v[i] != best) best = v[i];
    }
    return best;
}

/* unique_roots(p, tol=1e-3, rtype='min'): group roots within Euclidean tol, in input order
   (scipy.signal.unique_roots). Returns (unique complex, multiplicity int64). */
static int r_unique_roots(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t n; double complex *p = read_carr(&args[0], &n); if (!p) return TSR_ENOMEM;
    const double tol = (nargs > 1 && args[1].kind == 1) ? args[1].num : 1e-3;
    const int rt = sig_rtype(nargs > 2 ? &args[2] : NULL, RT_MIN);
    unsigned char *used = (unsigned char *)calloc((size_t)(n > 0 ? n : 1), 1);
    double complex *uq = (double complex *)malloc(sizeof(double complex) * (size_t)(n > 0 ? n : 1));
    int64_t *mult = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    double complex *grp = (double complex *)malloc(sizeof(double complex) * (size_t)(n > 0 ? n : 1));
    int rc = TSR_OK;
    if (!used || !uq || !mult || !grp) rc = TSR_ENOMEM;
    else {
        int64_t nu = 0;
        for (int64_t i = 0; i < n; i++) {
            if (used[i]) continue;
            int64_t g = 0;
            for (int64_t j = 0; j < n; j++) {
                if (used[j]) continue;
                const double dr = creal(p[j]) - creal(p[i]), di = cimag(p[j]) - cimag(p[i]);
                if (sqrt(dr * dr + di * di) <= tol) { grp[g++] = p[j]; used[j] = 1; }
            }
            uq[nu] = sig_creduce(grp, g, rt); mult[nu] = g; nu++;
        }
        double *ou = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){nu});
        int64_t *om = (int64_t *)fn_result_array(&res[1], TSR_I64, 1, (int64_t[]){nu});
        if (!ou || !om) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i < nu; i++) { ou[2 * i] = creal(uq[i]); ou[2 * i + 1] = cimag(uq[i]); om[i] = mult[i]; } }
    }
    free(used); free(uq); free(mult); free(grp); free(p);
    return rc;
}

/* real polynomial multiply (highest-first), r = a*b, length na+nb-1 (malloc'd; *nr set, -1 on OOM) */
static double *rp_mul(const double *a, int64_t na, const double *b, int64_t nb, int64_t *nr)
{
    const int64_t n = na + nb - 1;
    double *r = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!r) { *nr = -1; return NULL; }
    for (int64_t i = 0; i < na; i++) for (int64_t j = 0; j < nb; j++) r[i + j] += a[i] * b[j];
    *nr = n; return r;
}

/* real polynomial add (highest-first, tail-aligned), r = a + b; length max(na,nb) (malloc'd) */
static double *rp_add(const double *a, int64_t na, const double *b, int64_t nb, int64_t *nr)
{
    const int64_t n = (na > nb) ? na : nb;
    double *r = (double *)calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    if (!r) { *nr = -1; return NULL; }
    for (int64_t t = 0; t < n; t++) {
        const double av = (t < na) ? a[na - 1 - t] : 0.0, bv = (t < nb) ? b[nb - 1 - t] : 0.0;
        r[n - 1 - t] = av + bv;
    }
    *nr = n; return r;
}

/* _group_poles: adjacent grouping (scipy.signal._group_poles); representative is each run's first pole. */
static void sig_group_poles(const double *p, int64_t n, double tol, double *uq, int64_t *mult, int64_t *nu)
{
    int64_t k = 0;
    if (n == 0) { *nu = 0; return; }
    double pole = p[0]; int64_t cnt = 1;
    for (int64_t i = 1; i < n; i++) {
        if (fabs(p[i] - pole) <= tol) cnt++;
        else { uq[k] = pole; mult[k] = cnt; k++; pole = p[i]; cnt = 1; }
    }
    uq[k] = pole; mult[k] = cnt; k++;
    *nu = k;
}

/* _compute_factors(include_powers=True) for real poles: factors[t] = product of all (x-p_j)^m_j divided by
   (x-p_i)^(t+1) for each repeated-power term; also returns the full denominator. factors stored as rows in
   a flat buffer of width (total+1); returns total number of factors (= sum of multiplicities). */
static int sig_compute_factors(const double *up, const int64_t *mult, int64_t nu,
                               double **facs, int64_t **faclen, int64_t *nfac,
                               double **den, int64_t *nden)
{
    int64_t total = 0; for (int64_t i = 0; i < nu; i++) total += mult[i];
    const int64_t W = total + 1;                         /* max factor length */
    /* suffixes[i] = product_{j>i} (x-p_j)^m_j, highest-first; store flat rows of width W with lengths */
    double *suf = (double *)calloc((size_t)(nu * W), sizeof(double));
    int64_t *sufn = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nu > 0 ? nu : 1));
    double *facbuf = (double *)calloc((size_t)(total * W), sizeof(double));
    int64_t *flen = (int64_t *)malloc(sizeof(int64_t) * (size_t)(total > 0 ? total : 1));
    if (!suf || !sufn || !facbuf || !flen) { free(suf); free(sufn); free(facbuf); free(flen); return TSR_ENOMEM; }
    int rc = TSR_OK;
    /* build suffixes from the last pole down to index 1 */
    double *cur = (double *)malloc(sizeof(double) * (size_t)W); int64_t ncur = 1; cur[0] = 1.0;
    /* suffixes list in forward index: suffix[nu-1] = [1]; fill descending */
    if (nu > 0) { suf[(nu - 1) * W] = 1.0; sufn[nu - 1] = 1; }
    for (int64_t i = nu - 1; i >= 1 && rc == TSR_OK; i--) {
        const double mono[2] = {1.0, -up[i]};
        for (int64_t t = 0; t < mult[i]; t++) {
            int64_t nn; double *nx = rp_mul(cur, ncur, mono, 2, &nn);
            if (!nx) { rc = TSR_ENOMEM; break; }
            free(cur); cur = nx; ncur = nn;
        }
        if (rc == TSR_OK) { memcpy(suf + (i - 1) * W, cur, sizeof(double) * (size_t)ncur); sufn[i - 1] = ncur; }
    }
    free(cur);
    /* forward pass accumulating `current` = product_{j<i}; emit factor rows */
    double *current = (double *)malloc(sizeof(double) * (size_t)W); int64_t nce = 1; current[0] = 1.0;
    int64_t fi = 0;
    for (int64_t i = 0; i < nu && rc == TSR_OK; i++) {
        const double mono[2] = {1.0, -up[i]};
        for (int64_t t = 0; t < mult[i] && rc == TSR_OK; t++) {
            int64_t nb; double *blk = rp_mul(current, nce, suf + i * W, sufn[i], &nb);
            if (!blk) { rc = TSR_ENOMEM; break; }
            /* store into reversed position within this pole's block (reversed(block)) */
            memcpy(facbuf + (fi + (mult[i] - 1 - t)) * W, blk, sizeof(double) * (size_t)nb);
            flen[fi + (mult[i] - 1 - t)] = nb;
            free(blk);
            int64_t nn; double *nx = rp_mul(current, nce, mono, 2, &nn);
            if (!nx) { rc = TSR_ENOMEM; break; }
            free(current); current = nx; nce = nn;
        }
        fi += mult[i];
    }
    if (rc == TSR_OK) { *den = current; *nden = nce; *facs = facbuf; *faclen = flen; *nfac = total; }
    else { free(current); free(facbuf); free(flen); }
    free(suf); free(sufn);
    return rc;
}

/* shared back-end for invres/invresz; z_domain reverses polynomials (negative powers of z). */
static int sig_invres_common(const tsr_arg *args, int nargs, int z_domain, const char *who, tsr_result *res)
{
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("%s: r and p must be arrays", who); return TSR_EARG; }
    int64_t nr, np_, nk = 0;
    double *r = fn_arg_doubles(&args[0], &nr); if (!r) return TSR_ENOMEM;
    double *p = fn_arg_doubles(&args[1], &np_); if (!p) { fn_free_doubles(r, nr); return TSR_ENOMEM; }
    double *kk = NULL;
    if (nargs > 2 && args[2].kind == 3 && args[2].arr.ndim >= 1 && args[2].arr.shape[0] > 0) kk = fn_arg_doubles(&args[2], &nk);
    const double tol = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1e-3;
    /* trim k: 'f' (front) for invres, 'b' (back) for invresz */
    int64_t ks = 0, ke = nk;
    if (kk) { if (z_domain) { while (ke > 0 && kk[ke - 1] == 0.0) ke--; } else { while (ks < ke && kk[ks] == 0.0) ks++; } }
    const int64_t nkt = ke - ks;
    double *up = (double *)malloc(sizeof(double) * (size_t)(np_ > 0 ? np_ : 1));
    int64_t *mult = (int64_t *)malloc(sizeof(int64_t) * (size_t)(np_ > 0 ? np_ : 1));
    int rc = TSR_OK;
    double *facs = NULL, *den = NULL; int64_t *flen = NULL, nfac = 0, nden = 0, nu = 0;
    if (!up || !mult) rc = TSR_ENOMEM;
    else {
        sig_group_poles(p, np_, tol, up, mult, &nu);
        rc = sig_compute_factors(up, mult, nu, &facs, &flen, &nfac, &den, &nden);
    }
    if (rc == TSR_OK && nfac != nr) { fn_set_error("%s: number of residues must match the number of poles", who); rc = TSR_EARG; }
    if (rc == TSR_OK) {
        const int64_t W = nfac + 1;
        /* numerator accumulator (highest-first in the working domain) */
        double *num = NULL; int64_t nnum = 0;
        if (nkt == 0) { num = (double *)calloc(1, sizeof(double)); nnum = 1; }
        else {
            double *kt = (double *)malloc(sizeof(double) * (size_t)nkt);
            for (int64_t i = 0; i < nkt; i++) kt[i] = z_domain ? kk[ke - 1 - i] : kk[ks + i];   /* k[::-1] for z */
            double *dd = den; int64_t ndd = nden; double *drev = NULL;
            if (z_domain) { drev = (double *)malloc(sizeof(double) * (size_t)nden); for (int64_t i = 0; i < nden; i++) drev[i] = den[nden - 1 - i]; dd = drev; }
            num = rp_mul(kt, nkt, dd, ndd, &nnum);
            free(kt); free(drev);
            if (!num) rc = TSR_ENOMEM;
        }
        for (int64_t i = 0; i < nfac && rc == TSR_OK; i++) {
            double *fac = facs + i * W; int64_t nf = flen[i];
            double *scaled = (double *)malloc(sizeof(double) * (size_t)nf);
            for (int64_t t = 0; t < nf; t++) scaled[t] = r[i] * (z_domain ? fac[nf - 1 - t] : fac[t]);   /* factor[::-1] for z */
            int64_t nsum; double *s = rp_add(num, nnum, scaled, nf, &nsum);
            free(scaled); free(num);
            if (!s) { rc = TSR_ENOMEM; num = NULL; break; }
            num = s; nnum = nsum;
        }
        if (rc == TSR_OK) {
            /* invresz returns numerator[::-1]; invres returns it as-is */
            double *ob = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nnum});
            double *oa = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){nden});
            if (!ob || !oa) rc = TSR_ENOMEM;
            else { for (int64_t i = 0; i < nnum; i++) ob[i] = z_domain ? num[nnum - 1 - i] : num[i]; memcpy(oa, den, sizeof(double) * (size_t)nden); }
        }
        free(num);
    }
    free(facs); free(flen); free(den); free(up); free(mult);
    fn_free_doubles(r, nr); fn_free_doubles(p, np_); if (kk) fn_free_doubles(kk, nk);
    return rc;
}
static int r_invres(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return sig_invres_common(args, nargs, 0, "invres", res); }
static int r_invresz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return sig_invres_common(args, nargs, 1, "invresz", res); }

/* complex polynomial helpers (highest-first) for residue/residuez */
static double complex cp_val(const double complex *p, int64_t n, double complex x)
{ double complex r = 0; for (int64_t i = 0; i < n; i++) r = r * x + p[i]; return r; }

static double complex *cp_mul(const double complex *a, int64_t na, const double complex *b, int64_t nb, int64_t *nr)
{
    const int64_t n = na + nb - 1;
    double complex *r = (double complex *)calloc((size_t)(n > 0 ? n : 1), sizeof(double complex));
    if (!r) { *nr = -1; return NULL; }
    for (int64_t i = 0; i < na; i++) for (int64_t j = 0; j < nb; j++) r[i + j] += a[i] * b[j];
    *nr = n; return r;
}
static double complex *cp_addsub(const double complex *a, int64_t na, const double complex *b, int64_t nb, int sub, int64_t *nr)
{
    const int64_t n = (na > nb) ? na : nb;
    double complex *r = (double complex *)calloc((size_t)(n > 0 ? n : 1), sizeof(double complex));
    if (!r) { *nr = -1; return NULL; }
    for (int64_t t = 0; t < n; t++) {
        const double complex av = (t < na) ? a[na - 1 - t] : 0.0, bv = (t < nb) ? b[nb - 1 - t] : 0.0;
        r[n - 1 - t] = sub ? av - bv : av + bv;
    }
    *nr = n; return r;
}
/* numpy.polydiv (complex): quotient length max(nu-nv+1,1), remainder with leading |.|<=1e-8 trimmed (len>=1). */
static int cp_div(const double complex *u, int64_t nu, const double complex *v, int64_t nv,
                  double complex **q, int64_t *nq, double complex **r, int64_t *nr)
{
    const int64_t m = nu - 1, n = nv - 1;
    const double complex scale = 1.0 / v[0];
    const int64_t lq = (m - n + 1 > 1) ? (m - n + 1) : 1;
    double complex *Q = (double complex *)calloc((size_t)lq, sizeof(double complex));
    double complex *R = (double complex *)malloc(sizeof(double complex) * (size_t)nu);
    if (!Q || !R) { free(Q); free(R); return TSR_ENOMEM; }
    memcpy(R, u, sizeof(double complex) * (size_t)nu);
    for (int64_t k = 0; k <= m - n; k++) { const double complex d = scale * R[k]; Q[k] = d; for (int64_t j = 0; j <= n; j++) R[k + j] -= d * v[j]; }
    int64_t s = 0; while (s < nu - 1 && cabs(R[s]) <= 1e-8) s++;
    const int64_t rn = nu - s;
    double complex *RR = (double complex *)malloc(sizeof(double complex) * (size_t)rn);
    if (!RR) { free(Q); free(R); return TSR_ENOMEM; }
    memcpy(RR, R + s, sizeof(double complex) * (size_t)rn); free(R);
    *q = Q; *nq = lq; *r = RR; *nr = rn;
    return TSR_OK;
}

/* complex _compute_factors(include_powers=False): one factor per unique pole = full denominator / (x-p_i)^m_i.
   factors stored as flat rows of width (total+1); den returned separately. */
static int cp_compute_factors(const double complex *up, const int64_t *mult, int64_t nu,
                              double complex **facs, int64_t **faclen, double complex **den, int64_t *nden)
{
    int64_t total = 0; for (int64_t i = 0; i < nu; i++) total += mult[i];
    const int64_t W = total + 1;
    double complex *suf = (double complex *)calloc((size_t)(nu * W), sizeof(double complex));
    int64_t *sufn = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nu > 0 ? nu : 1));
    double complex *facbuf = (double complex *)calloc((size_t)((nu ? nu : 1) * W), sizeof(double complex));
    int64_t *flen = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nu > 0 ? nu : 1));
    if (!suf || !sufn || !facbuf || !flen) { free(suf); free(sufn); free(facbuf); free(flen); return TSR_ENOMEM; }
    int rc = TSR_OK;
    double complex *cur = (double complex *)malloc(sizeof(double complex) * (size_t)W); int64_t ncur = 1; cur[0] = 1.0;
    if (nu > 0) { suf[(nu - 1) * W] = 1.0; sufn[nu - 1] = 1; }
    for (int64_t i = nu - 1; i >= 1 && rc == TSR_OK; i--) {
        const double complex mono[2] = {1.0, -up[i]};
        for (int64_t t = 0; t < mult[i]; t++) { int64_t nn; double complex *nx = cp_mul(cur, ncur, mono, 2, &nn); if (!nx) { rc = TSR_ENOMEM; break; } free(cur); cur = nx; ncur = nn; }
        if (rc == TSR_OK) { memcpy(suf + (i - 1) * W, cur, sizeof(double complex) * (size_t)ncur); sufn[i - 1] = ncur; }
    }
    free(cur);
    double complex *current = (double complex *)malloc(sizeof(double complex) * (size_t)W); int64_t nce = 1; current[0] = 1.0;
    for (int64_t i = 0; i < nu && rc == TSR_OK; i++) {
        int64_t nb; double complex *blk = cp_mul(current, nce, suf + i * W, sufn[i], &nb);
        if (!blk) { rc = TSR_ENOMEM; break; }
        memcpy(facbuf + i * W, blk, sizeof(double complex) * (size_t)nb); flen[i] = nb; free(blk);
        const double complex mono[2] = {1.0, -up[i]};
        for (int64_t t = 0; t < mult[i] && rc == TSR_OK; t++) { int64_t nn; double complex *nx = cp_mul(current, nce, mono, 2, &nn); if (!nx) { rc = TSR_ENOMEM; break; } free(current); current = nx; nce = nn; }
    }
    if (rc == TSR_OK) { *den = current; *nden = nce; *facs = facbuf; *faclen = flen; }
    else { free(current); free(facbuf); free(flen); }
    free(suf); free(sufn);
    return rc;
}

/* complex _compute_residues (scipy.signal._compute_residues). residues length = sum(mult). */
static int cp_compute_residues(const double complex *up, const int64_t *mult, int64_t nu,
                               const double complex *numer, int64_t nnumer, double complex **out, int64_t *nout)
{
    double complex *facs = NULL, *den = NULL; int64_t *flen = NULL, nden = 0;
    int rc = cp_compute_factors(up, mult, nu, &facs, &flen, &den, &nden);
    if (rc != TSR_OK) return rc;
    const int64_t W = nden;                              /* den length = total+1 */
    int64_t total = 0; for (int64_t i = 0; i < nu; i++) total += mult[i];
    double complex *R = (double complex *)malloc(sizeof(double complex) * (size_t)(total > 0 ? total : 1));
    if (!R) { free(facs); free(flen); free(den); return TSR_ENOMEM; }
    int64_t ri = 0;
    for (int64_t i = 0; i < nu && rc == TSR_OK; i++) {
        const double complex pole = up[i]; const int64_t m = mult[i];
        double complex *factor = facs + i * W; int64_t nf = flen[i];
        if (m == 1) { R[ri++] = cp_val(numer, nnumer, pole) / cp_val(factor, nf, pole); }
        else {
            const double complex mono[2] = {1.0, -pole};
            double complex *fq = NULL, *fd = NULL; int64_t nfq = 0, nfd = 0;
            rc = cp_div(factor, nf, mono, 2, &fq, &nfq, &fd, &nfd);
            if (rc != TSR_OK) break;
            double complex *numc = (double complex *)malloc(sizeof(double complex) * (size_t)nnumer);
            memcpy(numc, numer, sizeof(double complex) * (size_t)nnumer); int64_t nnc = nnumer;
            double complex *block = (double complex *)malloc(sizeof(double complex) * (size_t)m);
            for (int64_t t = 0; t < m && rc == TSR_OK; t++) {
                double complex *nq = NULL, *nn = NULL; int64_t nnq = 0, nnn = 0;
                rc = cp_div(numc, nnc, mono, 2, &nq, &nnq, &nn, &nnn);
                if (rc != TSR_OK) { free(nq); free(nn); break; }
                const double complex rr = nn[0] / fd[0];
                block[t] = rr;
                /* numc = polysub(nq_was_numer_quotient?, rr*fq) -- scipy: numer=nq; numer=polysub(numer, rr*factor) */
                double complex *scaled = (double complex *)malloc(sizeof(double complex) * (size_t)nfq);
                for (int64_t z = 0; z < nfq; z++) scaled[z] = rr * fq[z];
                int64_t ns; double complex *sub = cp_addsub(nq, nnq, scaled, nfq, 1, &ns);
                free(scaled); free(nn); free(nq); free(numc);
                if (!sub) { rc = TSR_ENOMEM; numc = NULL; break; }
                numc = sub; nnc = ns;
            }
            for (int64_t t = 0; t < m; t++) R[ri + t] = block[m - 1 - t];   /* reversed(block) */
            ri += m;
            free(block); free(numc); free(fq); free(fd);
        }
    }
    free(facs); free(flen); free(den);
    if (rc == TSR_OK) { *out = R; *nout = ri; } else free(R);
    return rc;
}

/* sort unique poles by magnitude (ascending, stable); carry multiplicities. */
static void cp_cmplx_sort(double complex *up, int64_t *mult, int64_t nu)
{
    for (int64_t i = 1; i < nu; i++) {
        double complex kp = up[i]; int64_t km = mult[i]; const double ka = cabs(kp);
        int64_t j = i - 1;
        while (j >= 0 && cabs(up[j]) > ka) { up[j + 1] = up[j]; mult[j + 1] = mult[j]; j--; }
        up[j + 1] = kp; mult[j + 1] = km;
    }
}

/* shared residue/residuez back-end (real b, a). z_domain selects negative powers of z. */
static int sig_residue_common(const tsr_arg *args, int nargs, int z_domain, const char *who, tsr_result *res)
{
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("%s: b and a must be arrays", who); return TSR_EARG; }
    int64_t nb0, na0;
    double *bb = fn_arg_doubles(&args[0], &nb0); if (!bb) return TSR_ENOMEM;
    double *aa = fn_arg_doubles(&args[1], &na0); if (!aa) { fn_free_doubles(bb, nb0); return TSR_ENOMEM; }
    const double tol = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1e-3;
    const int rt = sig_rtype(nargs > 3 ? &args[3] : NULL, RT_AVG);
    /* trim_zeros: 'f' (front) for residue, 'b' (back) for residuez */
    int64_t bs = 0, be = nb0, as = 0, ae = na0;
    if (z_domain) { while (be > bs && bb[be - 1] == 0.0) be--; while (ae > as && aa[ae - 1] == 0.0) ae--; }
    else { while (bs < be && bb[bs] == 0.0) bs++; while (as < ae && aa[as] == 0.0) as++; }
    const int64_t nbt = be - bs, nat = ae - as;
    int rc = TSR_OK;
    double *aTrim = NULL; double *rootbuf = NULL;
    double complex *poles = NULL; int64_t npoles = 0;
    double complex *up = NULL; int64_t *mult = NULL; int64_t nu = 0;
    double complex *resid = NULL; int64_t nresid = 0;
    double *kout = NULL; int64_t nk = 0;
    if (nat < 1) { fn_set_error("%s: denominator `a` is zero", who); rc = TSR_EARG; }
    else {
        /* roots of the trimmed real denominator */
        aTrim = (double *)malloc(sizeof(double) * (size_t)nat);
        for (int64_t i = 0; i < nat; i++) aTrim[i] = aa[as + i];
        int64_t nz = 0; rootbuf = sig_polyroots(aTrim, nat, &nz);
        npoles = (nz > 0) ? nz : 0;
        if (nz < 0) { fn_set_error("%s: root solve failed", who); rc = TSR_EARG; }
    }
    if (rc == TSR_OK) {
        /* numerator polynomial (complex), reversed if z_domain */
        const double a_scale = z_domain ? aa[ae - 1] : aa[as];   /* a_rev[0]=a[-1] ; a[0] for residue */
        double *bwork = (double *)malloc(sizeof(double) * (size_t)(nbt > 0 ? nbt : 1));
        for (int64_t i = 0; i < nbt; i++) bwork[i] = z_domain ? bb[be - 1 - i] : bb[bs + i];
        double *awork = (double *)malloc(sizeof(double) * (size_t)(nat > 0 ? nat : 1));
        for (int64_t i = 0; i < nat; i++) awork[i] = z_domain ? aa[ae - 1 - i] : aa[as + i];
        /* k, b = polydiv(b, a) when numerator degree >= denominator degree */
        double complex *bcx = (double complex *)malloc(sizeof(double complex) * (size_t)(nbt > 0 ? nbt : 1));
        for (int64_t i = 0; i < nbt; i++) bcx[i] = bwork[i];
        int64_t nbcx = nbt;
        if (nbt == 0) { /* numerator is zero: residues all zero, k empty */
            nresid = npoles; resid = (double complex *)calloc((size_t)(npoles > 0 ? npoles : 1), sizeof(double complex));
            /* sorted poles for output */
            double complex *ps = (double complex *)malloc(sizeof(double complex) * (size_t)(npoles > 0 ? npoles : 1));
            int64_t *m1 = (int64_t *)malloc(sizeof(int64_t) * (size_t)(npoles > 0 ? npoles : 1));
            for (int64_t i = 0; i < npoles; i++) { ps[i] = rootbuf[2 * i] + I * rootbuf[2 * i + 1]; m1[i] = 1; }
            cp_cmplx_sort(ps, m1, npoles);
            free(rootbuf); rootbuf = NULL; poles = (double complex *)malloc(sizeof(double complex) * (size_t)(npoles > 0 ? npoles : 1));
            for (int64_t i = 0; i < npoles; i++) poles[i] = ps[i];
            free(ps); free(m1);
            up = NULL; nk = 0;
            free(bcx); free(bwork); free(awork);
            goto emit;
        }
        if (nbt >= nat) {
            double complex *acx = (double complex *)malloc(sizeof(double complex) * (size_t)nat);
            for (int64_t i = 0; i < nat; i++) acx[i] = awork[i];
            double complex *q = NULL, *r = NULL; int64_t nq = 0, nr = 0;
            rc = cp_div(bcx, nbcx, acx, nat, &q, &nq, &r, &nr);
            free(acx);
            if (rc == TSR_OK) { kout = (double *)malloc(sizeof(double) * (size_t)nq); for (int64_t i = 0; i < nq; i++) kout[i] = creal(q[i]); nk = nq; free(bcx); bcx = r; nbcx = nr; free(q); }
            else { free(q); free(r); }
        }
        if (rc == TSR_OK) {
            /* unique_roots on the (unsorted) complex poles, then cmplx_sort */
            up = (double complex *)malloc(sizeof(double complex) * (size_t)(npoles > 0 ? npoles : 1));
            mult = (int64_t *)malloc(sizeof(int64_t) * (size_t)(npoles > 0 ? npoles : 1));
            double complex *grp = (double complex *)malloc(sizeof(double complex) * (size_t)(npoles > 0 ? npoles : 1));
            unsigned char *used = (unsigned char *)calloc((size_t)(npoles > 0 ? npoles : 1), 1);
            double complex *pc = (double complex *)malloc(sizeof(double complex) * (size_t)(npoles > 0 ? npoles : 1));
            for (int64_t i = 0; i < npoles; i++) pc[i] = rootbuf[2 * i] + I * rootbuf[2 * i + 1];
            nu = 0;
            for (int64_t i = 0; i < npoles; i++) {
                if (used[i]) continue; int64_t g = 0;
                for (int64_t j = 0; j < npoles; j++) { if (used[j]) continue; const double dr = creal(pc[j]) - creal(pc[i]), di = cimag(pc[j]) - cimag(pc[i]); if (sqrt(dr * dr + di * di) <= tol) { grp[g++] = pc[j]; used[j] = 1; } }
                up[nu] = sig_creduce(grp, g, rt); mult[nu] = g; nu++;
            }
            free(grp); free(used); free(pc);
            cp_cmplx_sort(up, mult, nu);
            /* numerator for compute_residues (residuez uses 1/unique_poles) */
            double complex *cpoles = up;
            double complex *recip = NULL;
            if (z_domain) { recip = (double complex *)malloc(sizeof(double complex) * (size_t)(nu > 0 ? nu : 1)); for (int64_t i = 0; i < nu; i++) recip[i] = 1.0 / up[i]; cpoles = recip; }
            rc = cp_compute_residues(cpoles, mult, nu, bcx, nbcx, &resid, &nresid);
            free(recip);
            if (rc == TSR_OK) {
                /* expand poles and scale residues */
                double complex *pexp = (double complex *)malloc(sizeof(double complex) * (size_t)(nresid > 0 ? nresid : 1));
                int64_t idx = 0;
                for (int64_t i = 0; i < nu; i++) for (int64_t t = 0; t < mult[i]; t++) pexp[idx++] = up[i];
                if (z_domain) {
                    int64_t pidx = 0;
                    for (int64_t i = 0; i < nu; i++) for (int64_t t = 0; t < mult[i]; t++) { const double complex pw = cpow(-up[i], (double)(t + 1)); resid[pidx] = resid[pidx] * pw / a_scale; pidx++; }
                } else { for (int64_t i = 0; i < nresid; i++) resid[i] = resid[i] / a_scale; }
                poles = pexp; npoles = nresid;
            }
        }
        free(bcx); free(bwork); free(awork);
    }
emit:
    if (rc == TSR_OK) {
        double *orr = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){nresid});
        double *opp = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){npoles});
        double *okk = (double *)fn_result_array(&res[2], TSR_F64, 1, (int64_t[]){nk});
        if (!orr || !opp || !okk) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < nresid; i++) { orr[2 * i] = creal(resid[i]); orr[2 * i + 1] = cimag(resid[i]); }
            for (int64_t i = 0; i < npoles; i++) { opp[2 * i] = creal(poles[i]); opp[2 * i + 1] = cimag(poles[i]); }
            for (int64_t i = 0; i < nk; i++) okk[i] = z_domain ? kout[nk - 1 - i] : kout[i];   /* k_rev[::-1] for z */
        }
    }
    free(resid); free(poles); free(rootbuf); free(up); free(mult); free(kout); free(aTrim);
    fn_free_doubles(bb, nb0); fn_free_doubles(aa, na0);
    return rc;
}
static int r_residue(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return sig_residue_common(args, nargs, 0, "residue", res); }
static int r_residuez(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return sig_residue_common(args, nargs, 1, "residuez", res); }

/* ---- analog lowpass-prototype transforms (scipy.signal) ---- */

static double sig_comb(int64_t n, int64_t k)
{
    if (k < 0 || k > n) return 0.0;
    double r = 1.0;
    for (int64_t i = 0; i < k; i++) r = r * (double)(n - i) / (double)(i + 1);
    return r;
}

/* Polynomial multiply (lowest-first coefficients): r = a * b, length na+nb-1; r must not alias a or b. */
static void sig_polymul_lo(const double *a, int64_t na, const double *b, int64_t nb, double *r)
{
    const int64_t nr = na + nb - 1;
    for (int64_t i = 0; i < nr; i++) r[i] = 0.0;
    for (int64_t i = 0; i < na; i++) for (int64_t j = 0; j < nb; j++) r[i + j] += a[i] * b[j];
}

/* Emit normalize(bp, ap) into res; consumes bp/ap (caller frees). */
static int sig_emit_normalized(const double *bp, int64_t nbp, const double *ap, int64_t nap, tsr_result *res)
{
    double *b2 = NULL, *a2 = NULL; int64_t nb2 = 0, na2 = 0;
    if (sig_normalize(bp, nbp, ap, nap, &b2, &nb2, &a2, &na2) != 0) { fn_set_error("transform: normalize failed"); return TSR_EARG; }
    int rc = TSR_OK;
    double *ob = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nb2});
    double *oa = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){na2});
    if (!ob || !oa) rc = TSR_ENOMEM; else { memcpy(ob, b2, sizeof(double) * (size_t)nb2); memcpy(oa, a2, sizeof(double) * (size_t)na2); }
    free(b2); free(a2);
    return rc;
}

/* lp2lp(b, a, wo=1.0): transform a lowpass analog prototype to a different cutoff (scipy.signal.lp2lp). */
static int r_lp2lp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nb, na; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    const double wo = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    const int64_t M = na > nb ? na : nb;
    const int64_t start1 = nb > na ? nb - na : 0, start2 = na > nb ? na - nb : 0;
    const double ps1 = pow(wo, (double)(M - 1 - start1));
    double *bb = (double *)malloc(sizeof(double) * (size_t)nb), *aa = (double *)malloc(sizeof(double) * (size_t)na);
    int rc = TSR_OK;
    if (!bb || !aa) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < nb; i++) bb[i] = b[i] * ps1 / pow(wo, (double)(M - 1 - (start2 + i)));
        for (int64_t i = 0; i < na; i++) aa[i] = a[i] * ps1 / pow(wo, (double)(M - 1 - (start1 + i)));
        rc = sig_emit_normalized(bb, nb, aa, na, res);
    }
    free(bb); free(aa); fn_free_doubles(b, nb); fn_free_doubles(a, na);
    return rc;
}

/* lp2hp(b, a, wo=1.0): lowpass analog prototype to highpass (scipy.signal.lp2hp). */
static int r_lp2hp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nb, na; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    const double wo = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    const int64_t M = na > nb ? na : nb;
    double *pwo = (double *)malloc(sizeof(double) * (size_t)M);
    int rc = TSR_OK;
    if (!pwo) { fn_free_doubles(b, nb); fn_free_doubles(a, na); return TSR_ENOMEM; }
    for (int64_t i = 0; i < M; i++) pwo[i] = (wo != 1.0) ? pow(wo, (double)i) : 1.0;
    const int64_t L = na >= nb ? na : nb;
    double *outb = (double *)calloc((size_t)L, sizeof(double)), *outa = (double *)calloc((size_t)L, sizeof(double));
    if (!outb || !outa) rc = TSR_ENOMEM;
    else {
        if (na >= nb) {
            for (int64_t i = 0; i < na; i++) outa[i] = a[na - 1 - i] * pwo[i];
            for (int64_t i = 0; i < nb; i++) outb[i] = b[nb - 1 - i] * pwo[i];
        } else {
            for (int64_t i = 0; i < nb; i++) outb[i] = b[nb - 1 - i] * pwo[i];
            for (int64_t i = 0; i < na; i++) outa[i] = a[na - 1 - i] * pwo[i];
        }
        rc = sig_emit_normalized(outb, L, outa, L, res);
    }
    free(pwo); free(outb); free(outa); fn_free_doubles(b, nb); fn_free_doubles(a, na);
    return rc;
}

/* lp2bp(b, a, wo=1.0, bw=1.0): lowpass analog prototype to bandpass (scipy.signal.lp2bp). */
static int r_lp2bp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nb, na; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    const double wo = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    const double bw = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1.0;
    const int64_t N = nb - 1, D = na - 1, ma = N > D ? N : D, Np = N + ma, Dp = D + ma;
    const double wosq = wo * wo;
    double *bp = (double *)calloc((size_t)(Np + 1), sizeof(double)), *ap = (double *)calloc((size_t)(Dp + 1), sizeof(double));
    int rc = TSR_OK;
    if (!bp || !ap) rc = TSR_ENOMEM;
    else {
        for (int64_t j = 0; j <= Np; j++) { double v = 0.0;
            for (int64_t i = 0; i <= N; i++) for (int64_t k = 0; k <= i; k++) if (ma - i + 2 * k == j) v += sig_comb(i, k) * b[N - i] * pow(wosq, (double)(i - k)) / pow(bw, (double)i);
            bp[Np - j] = v; }
        for (int64_t j = 0; j <= Dp; j++) { double v = 0.0;
            for (int64_t i = 0; i <= D; i++) for (int64_t k = 0; k <= i; k++) if (ma - i + 2 * k == j) v += sig_comb(i, k) * a[D - i] * pow(wosq, (double)(i - k)) / pow(bw, (double)i);
            ap[Dp - j] = v; }
        rc = sig_emit_normalized(bp, Np + 1, ap, Dp + 1, res);
    }
    free(bp); free(ap); fn_free_doubles(b, nb); fn_free_doubles(a, na);
    return rc;
}

/* lp2bs(b, a, wo=1.0, bw=1.0): lowpass analog prototype to bandstop (scipy.signal.lp2bs). */
static int r_lp2bs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nb, na; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    const double wo = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    const double bw = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1.0;
    const int64_t N = nb - 1, D = na - 1, M = N > D ? N : D, Np = 2 * M, Dp = 2 * M;
    const double wosq = wo * wo;
    double *bp = (double *)calloc((size_t)(Np + 1), sizeof(double)), *ap = (double *)calloc((size_t)(Dp + 1), sizeof(double));
    int rc = TSR_OK;
    if (!bp || !ap) rc = TSR_ENOMEM;
    else {
        for (int64_t j = 0; j <= Np; j++) { double v = 0.0;
            for (int64_t i = 0; i <= N; i++) for (int64_t k = 0; k <= M - i; k++) if (i + 2 * k == j) v += sig_comb(M - i, k) * b[N - i] * pow(wosq, (double)(M - i - k)) * pow(bw, (double)i);
            bp[Np - j] = v; }
        for (int64_t j = 0; j <= Dp; j++) { double v = 0.0;
            for (int64_t i = 0; i <= D; i++) for (int64_t k = 0; k <= M - i; k++) if (i + 2 * k == j) v += sig_comb(M - i, k) * a[D - i] * pow(wosq, (double)(M - i - k)) * pow(bw, (double)i);
            ap[Dp - j] = v; }
        rc = sig_emit_normalized(bp, Np + 1, ap, Dp + 1, res);
    }
    free(bp); free(ap); fn_free_doubles(b, nb); fn_free_doubles(a, na);
    return rc;
}

/* bilinear(b, a, fs=1.0): bilinear transform of an analog filter to a digital one (scipy.signal.bilinear). */
static int r_bilinear(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nb0, na0; double *b0 = fn_arg_doubles(&args[0], &nb0); if (!b0) return TSR_ENOMEM;
    double *a0 = fn_arg_doubles(&args[1], &na0); if (!a0) { fn_free_doubles(b0, nb0); return TSR_ENOMEM; }
    const double fs = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    int64_t bs = 0; while (bs < nb0 - 1 && b0[bs] == 0.0) bs++;   /* trim leading zeros */
    int64_t as = 0; while (as < na0 - 1 && a0[as] == 0.0) as++;
    const double *b = b0 + bs, *a = a0 + as;
    const int64_t nb = nb0 - bs, na = na0 - as;
    const int64_t Ndeg = (nb > na ? nb : na) - 1;             /* output degree */
    const double fac = sqrt(2.0 * fs);
    const double P[2] = {1.0 / fac, 1.0 / fac};               /* (z+1)/fac, lowest-first */
    const double Q[2] = {-fac, fac};                          /* (z-1)*fac, lowest-first */
    double *num = (double *)calloc((size_t)(Ndeg + 1), sizeof(double));
    double *den = (double *)calloc((size_t)(Ndeg + 1), sizeof(double));
    double *tp = (double *)malloc(sizeof(double) * (size_t)(Ndeg + 2));   /* P^m accumulator */
    double *tq = (double *)malloc(sizeof(double) * (size_t)(Ndeg + 2));   /* Q^m accumulator */
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(Ndeg + 2));
    double *term = (double *)malloc(sizeof(double) * (size_t)(Ndeg + 2));
    int rc = TSR_OK;
    if (!num || !den || !tp || !tq || !tmp || !term) rc = TSR_ENOMEM;
    else {
        for (int pass = 0; pass < 2; pass++) {
            const double *c = pass ? a : b; const int64_t nc = pass ? na : nb; double *acc = pass ? den : num;
            for (int64_t q = 0; q < nc; q++) {
                const int64_t ep = Ndeg - q, eq = q;          /* P^ep * Q^eq, degree ep+eq = Ndeg */
                int64_t lp = 1; tp[0] = 1.0;                  /* P^ep (lowest-first) */
                for (int64_t m = 0; m < ep; m++) { sig_polymul_lo(tp, lp, P, 2, tmp); lp += 1; memcpy(tp, tmp, sizeof(double) * (size_t)lp); }
                int64_t lq = 1; tq[0] = 1.0;                  /* Q^eq */
                for (int64_t m = 0; m < eq; m++) { sig_polymul_lo(tq, lq, Q, 2, tmp); lq += 1; memcpy(tq, tmp, sizeof(double) * (size_t)lq); }
                sig_polymul_lo(tp, lp, tq, lq, term);         /* degree Ndeg -> length Ndeg+1 */
                const double cf = c[nc - 1 - q];              /* c[::-1][q] */
                for (int64_t t = 0; t <= Ndeg; t++) acc[t] += cf * term[t];
            }
        }
        /* reverse to highest-first, then normalize */
        double *bh = (double *)malloc(sizeof(double) * (size_t)(Ndeg + 1)), *ah = (double *)malloc(sizeof(double) * (size_t)(Ndeg + 1));
        if (!bh || !ah) rc = TSR_ENOMEM;
        else { for (int64_t i = 0; i <= Ndeg; i++) { bh[i] = num[Ndeg - i]; ah[i] = den[Ndeg - i]; } rc = sig_emit_normalized(bh, Ndeg + 1, ah, Ndeg + 1, res); }
        free(bh); free(ah);
    }
    free(num); free(den); free(tp); free(tq); free(tmp); free(term);
    fn_free_doubles(b0, nb0); fn_free_doubles(a0, na0);
    return rc;
}

/* ---- frequency response (analog freqs, zpk forms) ---- */

/* complex divide via Smith's method, matching numpy's npy_cdivide (as r_freqz does). */
static double complex sig_cdiv(double complex num, double complex den)
{
    const double nr = creal(num), ni = cimag(num), dr = creal(den), di = cimag(den);
    double hr, hi;
    if (fabs(dr) >= fabs(di)) { const double rat = di / dr, scl = 1.0 / (dr + di * rat); hr = (nr + ni * rat) * scl; hi = (ni - nr * rat) * scl; }
    else { const double rat = dr / di, scl = 1.0 / (dr * rat + di); hr = (nr * rat + ni) * scl; hi = (ni * rat - nr) * scl; }
    return hr + I * hi;
}

/* freqs(b, a, worN): analog frequency response H(jw) = polyval(b, jw)/polyval(a, jw) at the given frequencies
   (scipy.signal.freqs; worN must be an explicit 1-D frequency array). Returns (w, h). */
static int r_freqs(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("freqs: b and a must be 1-D arrays"); return TSR_EARG; }
    if (args[2].kind != 3 || args[2].arr.ndim != 1) { fn_set_error("freqs: worN must be a 1-D array of frequencies"); return TSR_EARG; }
    int64_t nb, na, nw; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    double *wv = fn_arg_doubles(&args[2], &nw); if (!wv) { fn_free_doubles(b, nb); fn_free_doubles(a, na); return TSR_ENOMEM; }
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nw});
    double *h = w ? (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){nw}) : NULL;
    int rc = TSR_OK;
    if (!w || !h) rc = TSR_ENOMEM;
    else for (int64_t j = 0; j < nw; j++) {
        const double complex s = I * wv[j];
        double complex num = b[0], den = a[0];
        for (int64_t k = 1; k < nb; k++) num = num * s + b[k];
        for (int64_t k = 1; k < na; k++) den = den * s + a[k];
        const double complex hv = sig_cdiv(num, den);
        w[j] = wv[j]; h[2 * j] = creal(hv); h[2 * j + 1] = cimag(hv);
    }
    fn_free_doubles(b, nb); fn_free_doubles(a, na); fn_free_doubles(wv, nw);
    return rc;
}

/* freqs_zpk(z, p, k, worN): analog response H(jw) = k*prod(jw - z)/prod(jw - p) (scipy.signal.freqs_zpk;
   worN an explicit 1-D frequency array). Returns (w, h). */
static int r_freqs_zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[3].kind != 3 || args[3].arr.ndim != 1) { fn_set_error("freqs_zpk: worN must be a 1-D array of frequencies"); return TSR_EARG; }
    int64_t nz, np_, nw; double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (args[2].kind == 1) ? args[2].num : 1.0;
    double *wv = fn_arg_doubles(&args[3], &nw); if (!wv) { free(z); free(p); return TSR_ENOMEM; }
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nw});
    double *h = w ? (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){nw}) : NULL;
    int rc = TSR_OK;
    if (!w || !h) rc = TSR_ENOMEM;
    else for (int64_t j = 0; j < nw; j++) {
        const double complex s = I * wv[j];
        double complex num = 1.0, den = 1.0;
        for (int64_t i = 0; i < nz; i++) num *= (s - z[i]);
        for (int64_t i = 0; i < np_; i++) den *= (s - p[i]);
        const double complex hv = k * sig_cdiv(num, den);
        w[j] = wv[j]; h[2 * j] = creal(hv); h[2 * j + 1] = cimag(hv);
    }
    free(z); free(p); fn_free_doubles(wv, nw);
    return rc;
}

/* freqz_zpk(z, p, k, worN=512, whole=False): digital response H(e^jw) = k*prod(e^jw - z)/prod(e^jw - p) on a
   linear grid [0, pi) (or [0, 2pi) if whole) (scipy.signal.freqz_zpk; fs defaults to 2*pi so w is returned raw). */
static int r_freqz_zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    const int64_t N = (nargs > 3 && args[3].kind == 1) ? (int64_t)args[3].num : 512;
    const int whole = (nargs > 4 && (args[4].kind == 1 || args[4].kind == 4)) ? args[4].num != 0 : 0;
    if (N <= 0) { fn_set_error("freqz_zpk: worN must be positive"); return TSR_EARG; }
    int64_t nz, np_; double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (args[2].kind == 1) ? args[2].num : 1.0;
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){N});
    double *h = w ? (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){N}) : NULL;
    int rc = TSR_OK;
    if (!w || !h) rc = TSR_ENOMEM;
    else {
        const double step = (whole ? 2.0 * M_PI : M_PI) / (double)N;
        for (int64_t j = 0; j < N; j++) {
            const double wj = (double)j * step;
            const double complex zm1 = cos(wj) + I * sin(wj);
            double complex num = 1.0, den = 1.0;
            for (int64_t i = 0; i < nz; i++) num *= (zm1 - z[i]);
            for (int64_t i = 0; i < np_; i++) den *= (zm1 - p[i]);
            const double complex hv = k * sig_cdiv(num, den);
            w[j] = wj; h[2 * j] = creal(hv); h[2 * j + 1] = cimag(hv);
        }
    }
    free(z); free(p);
    return rc;
}

/* sosfreqz(sos, worN=512, whole=False): frequency response of an SOS cascade = product of the per-section
   responses (scipy.signal.sosfreqz / freqz_sos). Returns (w, h). */
static int r_sosfreqz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[0].arr.shape[1] != 6) { fn_set_error("sosfreqz: sos must be an (n, 6) array"); return TSR_EARG; }
    const int64_t nsec = args[0].arr.shape[0];
    const int64_t N = (nargs > 1 && args[1].kind == 1) ? (int64_t)args[1].num : 512;
    const int whole = (nargs > 2 && (args[2].kind == 1 || args[2].kind == 4)) ? args[2].num != 0 : 0;
    if (N <= 0) { fn_set_error("sosfreqz: worN must be positive"); return TSR_EARG; }
    int64_t ns; double *sos = fn_arg_doubles(&args[0], &ns); if (!sos) return TSR_ENOMEM;
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){N});
    double *h = w ? (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){N}) : NULL;
    int rc = TSR_OK;
    if (!w || !h) rc = TSR_ENOMEM;
    else {
        const double step = (whole ? 2.0 * M_PI : M_PI) / (double)N;
        for (int64_t j = 0; j < N; j++) {
            const double wj = (double)j * step;
            const double complex zm1 = cos(wj) - I * sin(wj);
            double complex hv = 1.0;
            for (int64_t s = 0; s < nsec; s++) {
                const double *sb = sos + s * 6, *sa = sos + s * 6 + 3;
                const double complex num = sb[0] + zm1 * (sb[1] + zm1 * sb[2]);
                const double complex den = sa[0] + zm1 * (sa[1] + zm1 * sa[2]);
                hv *= sig_cdiv(num, den);
            }
            w[j] = wj; h[2 * j] = creal(hv); h[2 * j + 1] = cimag(hv);
        }
    }
    fn_free_doubles(sos, ns);
    return rc;
}

/* group_delay(b, a, w=512, whole=False): group delay -d(arg H)/dw of a digital filter (scipy.signal.group_delay).
   With c = convolve(b, a[::-1]): gd(w) = Re( sum_k k*c[k] z^k / sum_k c[k] z^k ) - (len(a)-1), z = exp(-i w);
   non-finite values (singularities) are set to 0. Returns (w, gd). */
static int r_group_delay(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("group_delay: b and a must be 1-D arrays"); return TSR_EARG; }
    const int64_t N = (nargs > 2 && args[2].kind == 1) ? (int64_t)args[2].num : 512;
    const int whole = (nargs > 3 && (args[3].kind == 1 || args[3].kind == 4)) ? args[3].num != 0 : 0;
    if (N <= 0) { fn_set_error("group_delay: w must be positive"); return TSR_EARG; }
    int64_t nb, na; double *b = fn_arg_doubles(&args[0], &nb); if (!b) return TSR_ENOMEM;
    double *a = fn_arg_doubles(&args[1], &na); if (!a) { fn_free_doubles(b, nb); return TSR_ENOMEM; }
    const int64_t nc = nb + na - 1;                          /* c = convolve(b, reverse(a)) */
    double *c = (double *)calloc((size_t)nc, sizeof(double));
    double *w = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){N});
    double *gd = w ? (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){N}) : NULL;
    int rc = TSR_OK;
    if (!c || !w || !gd) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < nb; i++) for (int64_t k = 0; k < na; k++) c[i + k] += b[i] * a[na - 1 - k];
        const double step = (whole ? 2.0 * M_PI : M_PI) / (double)N;
        for (int64_t j = 0; j < N; j++) {
            const double wj = (double)j * step;
            const double complex z = cos(wj) - I * sin(wj);
            double complex num = 0.0, den = 0.0;          /* polyval(c[::-1], z) and polyval(cr[::-1], z), Horner as numpy */
            for (int64_t k = nc - 1; k >= 0; k--) { den = den * z + c[k]; num = num * z + (double)k * c[k]; }
            const double complex ratio = sig_cdiv(num, den);
            double g = creal(ratio) - (double)(na - 1);
            if (!isfinite(g)) g = 0.0;
            w[j] = wj; gd[j] = g;
        }
    }
    free(c); fn_free_doubles(b, nb); fn_free_doubles(a, na);
    return rc;
}

/* hilbert(x, N=None): the analytic signal of a real sequence, x + i*H{x}, via the FFT (scipy.signal.hilbert).
   Xf = fft(x, N); double the positive-frequency bins and zero the negative ones; return ifft(Xf) (complex). */
static int r_hilbert(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("hilbert: x must be a 1-D real array"); return TSR_EARG; }
    int64_t n0; double *x = fn_arg_doubles(&args[0], &n0); if (!x) return TSR_ENOMEM;
    const int64_t N = (nargs > 1 && args[1].kind == 1) ? (int64_t)args[1].num : n0;
    int rc = TSR_OK;
    if (N <= 0) { fn_free_doubles(x, n0); fn_set_error("hilbert: N must be positive"); return TSR_EARG; }
    double *in = (double *)calloc((size_t)(2 * N), sizeof(double));
    double *Xf = (double *)malloc(sizeof(double) * (size_t)(2 * N));
    double *out = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){N});
    if (!in || !Xf || !out) rc = TSR_ENOMEM;
    else {
        const int64_t m = n0 < N ? n0 : N;
        for (int64_t i = 0; i < m; i++) in[2 * i] = x[i];    /* real input, zero-padded/truncated to N */
        tsr_fft(N, 1, 0, in, Xf);                            /* forward FFT */
        const int64_t half = N / 2;
        if (N % 2 == 0) {
            for (int64_t k = 1; k < half; k++) { Xf[2 * k] *= 2.0; Xf[2 * k + 1] *= 2.0; }
            for (int64_t k = half + 1; k < N; k++) { Xf[2 * k] = 0.0; Xf[2 * k + 1] = 0.0; }
        } else {
            for (int64_t k = 1; k <= half; k++) { Xf[2 * k] *= 2.0; Xf[2 * k + 1] *= 2.0; }   /* half=(N-1)/2 */
            for (int64_t k = half + 1; k < N; k++) { Xf[2 * k] = 0.0; Xf[2 * k + 1] = 0.0; }
        }
        tsr_fft(N, 1, 1, Xf, out);                           /* inverse FFT (scales by 1/N) */
    }
    free(in); free(Xf); fn_free_doubles(x, n0);
    return rc;
}

/* periodogram(x, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True,
   scaling='density'): the power spectral density estimate from a single segment (scipy.signal.periodogram,
   real x). window is 'boxcar' (default) or an explicit array. Returns (f, Pxx). */
static int r_periodogram(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("periodogram: x must be a 1-D real array"); return TSR_EARG; }
    int64_t N0; double *x = fn_arg_doubles(&args[0], &N0); if (!x) return TSR_ENOMEM;
    const double fs = (nargs > 1 && args[1].kind == 1) ? args[1].num : 1.0;
    int64_t nperseg = N0, fftlen = N0;
    if (nargs > 3 && args[3].kind == 1) { const int64_t nf = (int64_t)args[3].num; if (nf < N0) { nperseg = nf; fftlen = nf; } else { fftlen = nf; } }
    int det = 1;                                             /* 1 constant, 2 linear, 0 none */
    if (nargs > 4) { if (args[4].kind == 2 && args[4].str) { det = !strcmp(args[4].str, "linear") ? 2 : (!strcmp(args[4].str, "constant") ? 1 : 1); } else if (args[4].kind == 4 && args[4].num == 0.0) det = 0; }
    const int onesided = (nargs > 5 && args[5].kind == 4) ? (args[5].num != 0.0) : 1;
    const int density = !(nargs > 6 && args[6].kind == 2 && args[6].str && !strcmp(args[6].str, "spectrum"));
    int rc = TSR_OK;
    double *win = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *seg = (double *)malloc(sizeof(double) * (size_t)nperseg);
    int64_t nwin = 0; double *warr = (nargs > 2 && args[2].kind == 3) ? fn_arg_doubles(&args[2], &nwin) : NULL;
    if (!win || !seg || (nargs > 2 && args[2].kind == 3 && !warr)) rc = TSR_ENOMEM;
    else if (warr && nwin != nperseg) { fn_set_error("periodogram: window length must equal the segment length"); rc = TSR_EARG; }
    else {
        for (int64_t i = 0; i < nperseg; i++) win[i] = warr ? warr[i] : 1.0;   /* boxcar default */
        for (int64_t i = 0; i < nperseg; i++) seg[i] = x[i];
        if (det == 1) { double m = 0; for (int64_t i = 0; i < nperseg; i++) m += seg[i]; m /= (double)nperseg; for (int64_t i = 0; i < nperseg; i++) seg[i] -= m; }
        else if (det == 2 && nperseg > 1) {                  /* subtract the least-squares line */
            double st = 0, sx = 0, stt = 0, stx = 0; const double n = (double)nperseg;
            for (int64_t i = 0; i < nperseg; i++) { st += i; sx += seg[i]; stt += (double)i * i; stx += (double)i * seg[i]; }
            const double b = (n * stx - st * sx) / (n * stt - st * st), a = (sx - b * st) / n;
            for (int64_t i = 0; i < nperseg; i++) seg[i] -= a + b * (double)i;
        }
        double sw2 = 0, sw = 0;
        for (int64_t i = 0; i < nperseg; i++) { sw2 += win[i] * win[i]; sw += win[i]; seg[i] *= win[i]; }
        const double scale = density ? 1.0 / (fs * sw2) : 1.0 / (sw * sw);
        if (onesided) {
            const int64_t nb = fftlen / 2 + 1;
            double *in = (double *)calloc((size_t)fftlen, sizeof(double));
            double *Xf = (double *)malloc(sizeof(double) * (size_t)(2 * nb));
            double *f = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nb});
            double *P = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){nb});
            if (!in || !Xf || !f || !P) rc = TSR_ENOMEM;
            else {
                for (int64_t i = 0; i < nperseg; i++) in[i] = seg[i];
                tsr_rfft(fftlen, 1, in, Xf);
                for (int64_t k = 0; k < nb; k++) { P[k] = (Xf[2 * k] * Xf[2 * k] + Xf[2 * k + 1] * Xf[2 * k + 1]) * scale; f[k] = (double)k * fs / (double)fftlen; }
                const int64_t last = (fftlen % 2) ? nb : nb - 1;   /* odd: double 1.. ; even: double 1..nb-2 */
                for (int64_t k = 1; k < last; k++) P[k] *= 2.0;
            }
            free(in); free(Xf);
        } else {
            double *in = (double *)calloc((size_t)(2 * fftlen), sizeof(double));
            double *Xf = (double *)malloc(sizeof(double) * (size_t)(2 * fftlen));
            double *f = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){fftlen});
            double *P = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){fftlen});
            if (!in || !Xf || !f || !P) rc = TSR_ENOMEM;
            else {
                for (int64_t i = 0; i < nperseg; i++) in[2 * i] = seg[i];
                tsr_fft(fftlen, 1, 0, in, Xf);
                for (int64_t k = 0; k < fftlen; k++) {
                    P[k] = (Xf[2 * k] * Xf[2 * k] + Xf[2 * k + 1] * Xf[2 * k + 1]) * scale;
                    const int64_t kk = (k <= (fftlen - 1) / 2) ? k : k - fftlen;    /* fftfreq */
                    f[k] = (double)kk * fs / (double)fftlen;
                }
            }
            free(in); free(Xf);
        }
    }
    free(win); free(seg); if (warr) fn_free_doubles(warr, nwin);
    fn_free_doubles(x, N0);
    return rc;
}

/* welch(x, fs=1.0, window='hann_periodic', nperseg=None, noverlap=None, nfft=None, detrend='constant',
   return_onesided=True, scaling='density', average='mean'): Welch PSD estimate -- the per-segment periodogram
   averaged over overlapping segments (scipy.signal.welch, real x). Default window is a periodic Hann; nperseg
   defaults to min(256, len(x)); noverlap to nperseg//2. Returns (f, Pxx). */
static int r_welch(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("welch: x must be a 1-D real array"); return TSR_EARG; }
    int64_t nx; double *x = fn_arg_doubles(&args[0], &nx); if (!x) return TSR_ENOMEM;
    const double fs = (nargs > 1 && args[1].kind == 1) ? args[1].num : 1.0;
    /* window: array, or a name (boxcar / hann[_periodic], the default) */
    int64_t nwin = 0; double *warr = (nargs > 2 && args[2].kind == 3) ? fn_arg_doubles(&args[2], &nwin) : NULL;
    int win_boxcar = (nargs > 2 && args[2].kind == 2 && args[2].str && !strcmp(args[2].str, "boxcar"));
    int64_t nperseg = (nargs > 3 && args[3].kind == 1) ? (int64_t)args[3].num : (warr ? nwin : (nx < 256 ? nx : 256));
    if (nperseg > nx) nperseg = nx;
    const int64_t noverlap = (nargs > 4 && args[4].kind == 1) ? (int64_t)args[4].num : nperseg / 2;
    const int64_t nfft = (nargs > 5 && args[5].kind == 1) ? (int64_t)args[5].num : nperseg;
    int det = 1;
    if (nargs > 6) { if (args[6].kind == 2 && args[6].str) det = !strcmp(args[6].str, "linear") ? 2 : 1; else if (args[6].kind == 4 && args[6].num == 0.0) det = 0; }
    const int onesided = (nargs > 7 && args[7].kind == 4) ? (args[7].num != 0.0) : 1;
    const int density = !(nargs > 8 && args[8].kind == 2 && args[8].str && !strcmp(args[8].str, "spectrum"));
    /* args[9] is axis (1-D only, ignored); average is args[10] to match scipy's parameter order */
    const int median = (nargs > 10 && args[10].kind == 2 && args[10].str && !strcmp(args[10].str, "median"));
    const int64_t nstep = nperseg - noverlap;
    int rc = TSR_OK;
    if (nstep <= 0) { if (warr) fn_free_doubles(warr, nwin); fn_free_doubles(x, nx); fn_set_error("welch: noverlap must be less than nperseg"); return TSR_EARG; }
    const int64_t nseg = nperseg <= nx ? 1 + (nx - nperseg) / nstep : 0;
    const int64_t nb = onesided ? nfft / 2 + 1 : nfft;
    double *win = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *acc = (double *)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double));         /* mean accumulator */
    double *allP = median ? (double *)malloc(sizeof(double) * (size_t)(nb * (nseg > 0 ? nseg : 1))) : NULL;   /* per-seg for median */
    double *seg = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *in = (double *)calloc((size_t)(onesided ? nfft : 2 * nfft), sizeof(double));
    double *Xf = (double *)malloc(sizeof(double) * (size_t)(2 * nb));
    if (!win || !acc || !seg || !in || !Xf || (median && !allP) || (nargs > 2 && args[2].kind == 3 && !warr)) rc = TSR_ENOMEM;
    else if (warr && nwin != nperseg) { fn_set_error("welch: window length must equal nperseg"); rc = TSR_EARG; }
    else if (nseg < 1) { fn_set_error("welch: nperseg exceeds the signal length"); rc = TSR_EARG; }
    else {
        for (int64_t i = 0; i < nperseg; i++) {                 /* periodic Hann default, or boxcar, or array */
            if (warr) win[i] = warr[i];
            else if (win_boxcar) win[i] = 1.0;
            else win[i] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)nperseg);
        }
        double sw2 = 0, sw = 0; for (int64_t i = 0; i < nperseg; i++) { sw2 += win[i] * win[i]; sw += win[i]; }
        const double scale = density ? 1.0 / (fs * sw2) : 1.0 / (sw * sw);
        for (int64_t s = 0; s < nseg; s++) {
            const double *src = x + s * nstep;
            for (int64_t i = 0; i < nperseg; i++) seg[i] = src[i];
            if (det == 1) { double m = 0; for (int64_t i = 0; i < nperseg; i++) m += seg[i]; m /= (double)nperseg; for (int64_t i = 0; i < nperseg; i++) seg[i] -= m; }
            else if (det == 2 && nperseg > 1) { double st = 0, sx = 0, stt = 0, stx = 0; const double n = (double)nperseg; for (int64_t i = 0; i < nperseg; i++) { st += i; sx += seg[i]; stt += (double)i * i; stx += (double)i * seg[i]; } const double b = (n * stx - st * sx) / (n * stt - st * st), a = (sx - b * st) / n; for (int64_t i = 0; i < nperseg; i++) seg[i] -= a + b * (double)i; }
            for (int64_t i = 0; i < nperseg; i++) seg[i] *= win[i];
            if (onesided) { for (int64_t i = 0; i < nfft; i++) in[i] = i < nperseg ? seg[i] : 0.0; tsr_rfft(nfft, 1, in, Xf); }
            else { for (int64_t i = 0; i < nfft; i++) { in[2 * i] = i < nperseg ? seg[i] : 0.0; in[2 * i + 1] = 0.0; } tsr_fft(nfft, 1, 0, in, Xf); }
            for (int64_t k = 0; k < nb; k++) {
                const double p = (Xf[2 * k] * Xf[2 * k] + Xf[2 * k + 1] * Xf[2 * k + 1]) * scale;
                if (median) allP[s * nb + k] = p; else acc[k] += p;
            }
        }
        double *f = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nb});
        double *P = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){nb});
        if (!f || !P) rc = TSR_ENOMEM;
        else {
            double bias = 1.0;
            if (median) { for (int64_t k = 1; k <= (nseg - 1) / 2; k++) bias += 1.0 / (double)(2 * k + 1) - 1.0 / (double)(2 * k); }
            double *col = median ? (double *)malloc(sizeof(double) * (size_t)nseg) : NULL;
            for (int64_t k = 0; k < nb; k++) {
                if (median) { for (int64_t s = 0; s < nseg; s++) col[s] = allP[s * nb + k]; qsort(col, (size_t)nseg, sizeof(double), cmp_double_sig); P[k] = (nseg % 2 ? col[nseg / 2] : 0.5 * (col[nseg / 2 - 1] + col[nseg / 2])) / bias; }
                else P[k] = acc[k] / (double)nseg;
                if (onesided) { const int64_t kk = (nfft % 2) ? (k >= 1 ? k : -1) : (k >= 1 && k < nb - 1 ? k : -1); if (kk >= 0) P[k] *= 2.0; }
                if (onesided) f[k] = (double)k * fs / (double)nfft;
                else { const int64_t kk = (k <= (nfft - 1) / 2) ? k : k - nfft; f[k] = (double)kk * fs / (double)nfft; }
            }
            free(col);
        }
    }
    free(win); free(acc); free(allP); free(seg); free(in); free(Xf); if (warr) fn_free_doubles(warr, nwin);
    fn_free_doubles(x, nx);
    return rc;
}

/* csd(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant',
   return_onesided=True, scaling='density', axis=-1, average='mean'): the cross power spectral density, the
   Welch estimate with per-segment cross spectrum Xf_x * conj(Xf_y) (scipy.signal.csd, real inputs, mean
   average). Returns (f, Pxy) with Pxy complex. */
static int r_csd(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("csd: x and y must be 1-D real arrays"); return TSR_EARG; }
    int64_t nx, ny; double *x = fn_arg_doubles(&args[0], &nx); if (!x) return TSR_ENOMEM;
    double *y = fn_arg_doubles(&args[1], &ny); if (!y) { fn_free_doubles(x, nx); return TSR_ENOMEM; }
    const int64_t nmin = nx < ny ? nx : ny;                  /* scipy zero-extends the shorter; here keep common length */
    const double fs = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    int64_t nwin = 0; double *warr = (nargs > 3 && args[3].kind == 3) ? fn_arg_doubles(&args[3], &nwin) : NULL;
    const int win_boxcar = (nargs > 3 && args[3].kind == 2 && args[3].str && !strcmp(args[3].str, "boxcar"));
    int64_t nperseg = (nargs > 4 && args[4].kind == 1) ? (int64_t)args[4].num : (warr ? nwin : (nmin < 256 ? nmin : 256));
    if (nperseg > nmin) nperseg = nmin;
    const int64_t noverlap = (nargs > 5 && args[5].kind == 1) ? (int64_t)args[5].num : nperseg / 2;
    const int64_t nfft = (nargs > 6 && args[6].kind == 1) ? (int64_t)args[6].num : nperseg;
    int det = 1;
    if (nargs > 7) { if (args[7].kind == 2 && args[7].str) det = !strcmp(args[7].str, "linear") ? 2 : 1; else if (args[7].kind == 4 && args[7].num == 0.0) det = 0; }
    const int onesided = (nargs > 8 && args[8].kind == 4) ? (args[8].num != 0.0) : 1;
    const int density = !(nargs > 9 && args[9].kind == 2 && args[9].str && !strcmp(args[9].str, "spectrum"));
    const int64_t nstep = nperseg - noverlap;
    int rc = TSR_OK;
    if (nstep <= 0) { if (warr) fn_free_doubles(warr, nwin); fn_free_doubles(x, nx); fn_free_doubles(y, ny); fn_set_error("csd: noverlap must be less than nperseg"); return TSR_EARG; }
    const int64_t nseg = nperseg <= nmin ? 1 + (nmin - nperseg) / nstep : 0;
    const int64_t nb = onesided ? nfft / 2 + 1 : nfft;
    double *win = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *accr = (double *)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double));
    double *acci = (double *)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double));
    double *sx = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *sy = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *inx = (double *)calloc((size_t)(onesided ? nfft : 2 * nfft), sizeof(double));
    double *iny = (double *)calloc((size_t)(onesided ? nfft : 2 * nfft), sizeof(double));
    double *Fx = (double *)malloc(sizeof(double) * (size_t)(2 * nb));
    double *Fy = (double *)malloc(sizeof(double) * (size_t)(2 * nb));
    if (!win || !accr || !acci || !sx || !sy || !inx || !iny || !Fx || !Fy || (nargs > 3 && args[3].kind == 3 && !warr)) rc = TSR_ENOMEM;
    else if (warr && nwin != nperseg) { fn_set_error("csd: window length must equal nperseg"); rc = TSR_EARG; }
    else if (nseg < 1) { fn_set_error("csd: nperseg exceeds the signal length"); rc = TSR_EARG; }
    else {
        for (int64_t i = 0; i < nperseg; i++) win[i] = warr ? warr[i] : (win_boxcar ? 1.0 : 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)nperseg));
        double sw2 = 0, sw = 0; for (int64_t i = 0; i < nperseg; i++) { sw2 += win[i] * win[i]; sw += win[i]; }
        const double scale = density ? 1.0 / (fs * sw2) : 1.0 / (sw * sw);
        for (int64_t s = 0; s < nseg; s++) {
            const double *px = x + s * nstep, *py = y + s * nstep;
            for (int64_t i = 0; i < nperseg; i++) { sx[i] = px[i]; sy[i] = py[i]; }
            for (int pass = 0; pass < 2; pass++) {
                double *seg = pass ? sy : sx;
                if (det == 1) { double m = 0; for (int64_t i = 0; i < nperseg; i++) m += seg[i]; m /= (double)nperseg; for (int64_t i = 0; i < nperseg; i++) seg[i] -= m; }
                else if (det == 2 && nperseg > 1) { double st = 0, ss = 0, stt = 0, stx = 0; const double n = (double)nperseg; for (int64_t i = 0; i < nperseg; i++) { st += i; ss += seg[i]; stt += (double)i * i; stx += (double)i * seg[i]; } const double b = (n * stx - st * ss) / (n * stt - st * st), a = (ss - b * st) / n; for (int64_t i = 0; i < nperseg; i++) seg[i] -= a + b * (double)i; }
                for (int64_t i = 0; i < nperseg; i++) seg[i] *= win[i];
            }
            if (onesided) {
                for (int64_t i = 0; i < nfft; i++) { inx[i] = i < nperseg ? sx[i] : 0.0; iny[i] = i < nperseg ? sy[i] : 0.0; }
                tsr_rfft(nfft, 1, inx, Fx); tsr_rfft(nfft, 1, iny, Fy);
            } else {
                for (int64_t i = 0; i < nfft; i++) { inx[2 * i] = i < nperseg ? sx[i] : 0.0; inx[2 * i + 1] = 0.0; iny[2 * i] = i < nperseg ? sy[i] : 0.0; iny[2 * i + 1] = 0.0; }
                tsr_fft(nfft, 1, 0, inx, Fx); tsr_fft(nfft, 1, 0, iny, Fy);
            }
            for (int64_t k = 0; k < nb; k++) {               /* conj(Xf_x) * Xf_y (scipy's csd convention) */
                const double xr = Fx[2 * k], xi = Fx[2 * k + 1], yr = Fy[2 * k], yi = Fy[2 * k + 1];
                accr[k] += (xr * yr + xi * yi) * scale; acci[k] += (xr * yi - xi * yr) * scale;
            }
        }
        double *f = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nb});
        double *P = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){nb});
        if (!f || !P) rc = TSR_ENOMEM;
        else for (int64_t k = 0; k < nb; k++) {
            double pr = accr[k] / (double)nseg, pi = acci[k] / (double)nseg;
            if (onesided) { const int fold = (nfft % 2) ? (k >= 1) : (k >= 1 && k < nb - 1); if (fold) { pr *= 2.0; pi *= 2.0; } f[k] = (double)k * fs / (double)nfft; }
            else { const int64_t kk = (k <= (nfft - 1) / 2) ? k : k - nfft; f[k] = (double)kk * fs / (double)nfft; }
            P[2 * k] = pr; P[2 * k + 1] = pi;
        }
    }
    free(win); free(accr); free(acci); free(sx); free(sy); free(inx); free(iny); free(Fx); free(Fy);
    if (warr) fn_free_doubles(warr, nwin); fn_free_doubles(x, nx); fn_free_doubles(y, ny);
    return rc;
}

/* coherence(x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', axis=-1):
   the magnitude-squared coherence Cxy = |Pxy|^2 / (Pxx*Pyy) by Welch's method (scipy.signal.coherence). The
   per-segment scale and one-sided doubling cancel in the ratio, so only the segment means are needed. (f, Cxy). */
static int r_coherence(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1 || args[1].kind != 3 || args[1].arr.ndim != 1) { fn_set_error("coherence: x and y must be 1-D real arrays"); return TSR_EARG; }
    int64_t nx, ny; double *x = fn_arg_doubles(&args[0], &nx); if (!x) return TSR_ENOMEM;
    double *y = fn_arg_doubles(&args[1], &ny); if (!y) { fn_free_doubles(x, nx); return TSR_ENOMEM; }
    const int64_t nmin = nx < ny ? nx : ny;
    const double fs = (nargs > 2 && args[2].kind == 1) ? args[2].num : 1.0;
    int64_t nwin = 0; double *warr = (nargs > 3 && args[3].kind == 3) ? fn_arg_doubles(&args[3], &nwin) : NULL;
    const int win_boxcar = (nargs > 3 && args[3].kind == 2 && args[3].str && !strcmp(args[3].str, "boxcar"));
    int64_t nperseg = (nargs > 4 && args[4].kind == 1) ? (int64_t)args[4].num : (warr ? nwin : (nmin < 256 ? nmin : 256));
    if (nperseg > nmin) nperseg = nmin;
    const int64_t noverlap = (nargs > 5 && args[5].kind == 1) ? (int64_t)args[5].num : nperseg / 2;
    const int64_t nfft = (nargs > 6 && args[6].kind == 1) ? (int64_t)args[6].num : nperseg;
    int det = 1;
    if (nargs > 7) { if (args[7].kind == 2 && args[7].str) det = !strcmp(args[7].str, "linear") ? 2 : 1; else if (args[7].kind == 4 && args[7].num == 0.0) det = 0; }
    const int64_t nstep = nperseg - noverlap;
    int rc = TSR_OK;
    if (nstep <= 0) { if (warr) fn_free_doubles(warr, nwin); fn_free_doubles(x, nx); fn_free_doubles(y, ny); fn_set_error("coherence: noverlap must be less than nperseg"); return TSR_EARG; }
    const int64_t nseg = nperseg <= nmin ? 1 + (nmin - nperseg) / nstep : 0;
    const int64_t nb = nfft / 2 + 1;                         /* coherence is one-sided */
    double *win = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *sxx = (double *)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double));
    double *syy = (double *)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double));
    double *pr = (double *)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double));
    double *pi = (double *)calloc((size_t)(nb > 0 ? nb : 1), sizeof(double));
    double *s1 = (double *)malloc(sizeof(double) * (size_t)nperseg), *s2 = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *in1 = (double *)calloc((size_t)nfft, sizeof(double)), *in2 = (double *)calloc((size_t)nfft, sizeof(double));
    double *F1 = (double *)malloc(sizeof(double) * (size_t)(2 * nb)), *F2 = (double *)malloc(sizeof(double) * (size_t)(2 * nb));
    if (!win || !sxx || !syy || !pr || !pi || !s1 || !s2 || !in1 || !in2 || !F1 || !F2 || (nargs > 3 && args[3].kind == 3 && !warr)) rc = TSR_ENOMEM;
    else if (warr && nwin != nperseg) { fn_set_error("coherence: window length must equal nperseg"); rc = TSR_EARG; }
    else if (nseg < 1) { fn_set_error("coherence: nperseg exceeds the signal length"); rc = TSR_EARG; }
    else {
        for (int64_t i = 0; i < nperseg; i++) win[i] = warr ? warr[i] : (win_boxcar ? 1.0 : 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)nperseg));
        for (int64_t s = 0; s < nseg; s++) {
            const double *px = x + s * nstep, *py = y + s * nstep;
            for (int64_t i = 0; i < nperseg; i++) { s1[i] = px[i]; s2[i] = py[i]; }
            for (int pass = 0; pass < 2; pass++) {
                double *seg = pass ? s2 : s1;
                if (det == 1) { double m = 0; for (int64_t i = 0; i < nperseg; i++) m += seg[i]; m /= (double)nperseg; for (int64_t i = 0; i < nperseg; i++) seg[i] -= m; }
                else if (det == 2 && nperseg > 1) { double st = 0, ss = 0, stt = 0, stx = 0; const double n = (double)nperseg; for (int64_t i = 0; i < nperseg; i++) { st += i; ss += seg[i]; stt += (double)i * i; stx += (double)i * seg[i]; } const double b = (n * stx - st * ss) / (n * stt - st * st), a = (ss - b * st) / n; for (int64_t i = 0; i < nperseg; i++) seg[i] -= a + b * (double)i; }
                for (int64_t i = 0; i < nperseg; i++) seg[i] *= win[i];
            }
            for (int64_t i = 0; i < nfft; i++) { in1[i] = i < nperseg ? s1[i] : 0.0; in2[i] = i < nperseg ? s2[i] : 0.0; }
            tsr_rfft(nfft, 1, in1, F1); tsr_rfft(nfft, 1, in2, F2);
            for (int64_t k = 0; k < nb; k++) {
                const double ar = F1[2 * k], ai = F1[2 * k + 1], br = F2[2 * k], bi = F2[2 * k + 1];
                sxx[k] += ar * ar + ai * ai; syy[k] += br * br + bi * bi;
                pr[k] += ar * br + ai * bi; pi[k] += ar * bi - ai * br;   /* conj(X)*Y */
            }
        }
        double *f = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nb});
        double *C = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){nb});
        if (!f || !C) rc = TSR_ENOMEM;
        else for (int64_t k = 0; k < nb; k++) { const double den = sxx[k] * syy[k]; C[k] = den > 0.0 ? (pr[k] * pr[k] + pi[k] * pi[k]) / den : 0.0; f[k] = (double)k * fs / (double)nfft; }
    }
    free(win); free(sxx); free(syy); free(pr); free(pi); free(s1); free(s2); free(in1); free(in2); free(F1); free(F2);
    if (warr) fn_free_doubles(warr, nwin); fn_free_doubles(x, nx); fn_free_doubles(y, ny);
    return rc;
}

/* resample(x, num): resample a real signal to num samples via the FFT -- rfft, resize the spectrum to num
   bins (with the standard unpaired-Nyquist-bin adjustment and 1/s_fac scaling), then irfft (scipy.signal.
   resample, real x, window=None, domain='time'). Returns the length-num resampled signal. */
static int r_resample(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("resample: x must be a 1-D real array"); return TSR_EARG; }
    if (nargs < 2 || args[1].kind != 1) { fn_set_error("resample: num must be an integer"); return TSR_EARG; }
    const int64_t num = (int64_t)args[1].num;
    if (num <= 0) { fn_set_error("resample: num must be positive"); return TSR_EARG; }
    if (nargs > 4 && args[4].kind != 0 && !(args[4].kind == 4 && args[4].num == 0.0)) { fn_set_error("resample: only window=None is supported"); return TSR_EARG; }
    int64_t Nx; double *x = fn_arg_doubles(&args[0], &Nx); if (!x) return TSR_ENOMEM;
    const int64_t nxb = Nx / 2 + 1, nyb = num / 2 + 1;
    const int64_t m = num < Nx ? num : Nx, m2 = m / 2 + 1;
    double *Xf = (double *)malloc(sizeof(double) * (size_t)(2 * nxb));
    double *Y = (double *)calloc((size_t)(2 * nyb), sizeof(double));
    double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){num});
    int rc = TSR_OK;
    if (!Xf || !Y || !out) rc = TSR_ENOMEM;
    else {
        tsr_rfft(Nx, 1, x, Xf);
        const double s_fac = (double)Nx / (double)num;
        const int64_t ncopy = m2 < nyb ? m2 : nyb;
        for (int64_t k = 0; k < ncopy; k++) { Y[2 * k] = Xf[2 * k]; Y[2 * k + 1] = Xf[2 * k + 1]; }
        if (m % 2 == 0 && num != Nx) {                       /* unpaired Nyquist bin at m/2 */
            const int64_t idx = m / 2; const double fac = num < Nx ? 2.0 : 0.5;
            if (idx < nyb) { Y[2 * idx] *= fac; Y[2 * idx + 1] *= fac; }
        }
        for (int64_t k = 0; k < nyb; k++) { Y[2 * k] /= s_fac; Y[2 * k + 1] /= s_fac; }
        tsr_irfft(num, 1, Y, out);
    }
    free(Xf); free(Y); fn_free_doubles(x, Nx);
    return rc;
}

static double sig_sinc(double x) { if (x == 0.0) return 1.0; const double px = M_PI * x; return sin(px) / px; }

static double sig_kaiser_beta(double a)
{
    if (a > 50.0) return 0.1102 * (a - 8.7);
    if (a > 21.0) return 0.5842 * pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
    return 0.0;
}

/* kaiser_atten(numtaps, width): Kaiser-window attenuation (dB) for a filter length and transition width. */
static int r_kaiser_atten(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 1 || args[1].kind != 1) { fn_set_error("kaiser_atten: numtaps and width must be numbers"); return TSR_EARG; }
    const double numtaps = args[0].num, width = args[1].num;
    fn_result_num(&res[0], 2.285 * (numtaps - 1.0) * M_PI * width + 7.95);
    return TSR_OK;
}

/* kaiser_beta(a): the Kaiser-window shape beta for a given attenuation a (dB). */
static int r_kaiser_beta(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 1) { fn_set_error("kaiser_beta: a must be a number"); return TSR_EARG; }
    fn_result_num(&res[0], sig_kaiser_beta(args[0].num));
    return TSR_OK;
}

/* kaiserord(ripple, width): the Kaiser filter length and beta for a ripple attenuation and transition width
   (scipy.signal.kaiserord). Returns (numtaps, beta). */
static int r_kaiserord(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 1 || args[1].kind != 1) { fn_set_error("kaiserord: ripple and width must be numbers"); return TSR_EARG; }
    const double A = fabs(args[0].num), width = args[1].num;
    if (A < 8.0) { fn_set_error("kaiserord: ripple attenuation is too small for the Kaiser formula"); return TSR_EARG; }
    const double beta = sig_kaiser_beta(A);
    const double nt = (A - 7.95) / 2.285 / (M_PI * width) + 1.0;
    fn_result_int(&res[0], (int64_t)ceil(nt));
    fn_result_num(&res[1], beta);
    return TSR_OK;
}

extern void tsr_special_ellipk(const void *ctx, const double *in, double *out);
extern void tsr_special_ellipkm1(const void *ctx, const double *in, double *out);

/* shared front-end for the IIR order-selection routines (scipy.signal.*ord), lowpass/highpass only:
   validate scalar wp/ws, pre-warp (digital) and compute the lowpass-prototype natural frequency `nat`.
   filter_type 1=low, 2=high. Returns the passband edge in *passb, analog flag and fs for post-processing. */
static int sig_ord_prep(const tsr_arg *args, int nargs, int *ftype, double *passb, double *nat, int *analog, double *fsv)
{
    if (args[0].kind != 1 || args[1].kind != 1) { fn_set_error("filter order: only a scalar wp/ws (lowpass/highpass) is supported"); return TSR_EARG; }
    double wp = args[0].num, ws = args[1].num;
    const int an = (nargs > 4 && (args[4].kind == 1 || args[4].kind == 4) && args[4].num != 0.0);
    double fs = -1.0;
    if (nargs > 5 && args[5].kind == 1) {
        if (an) { fn_set_error("filter order: fs cannot be specified for an analog filter"); return TSR_EARG; }
        fs = args[5].num; wp = 2.0 * wp / fs; ws = 2.0 * ws / fs;
    }
    const int ft = (wp >= ws) ? 2 : 1;
    const double pb = an ? wp : tan(M_PI * wp / 2.0);
    const double sb = an ? ws : tan(M_PI * ws / 2.0);
    *ftype = ft; *passb = pb; *nat = fabs(ft == 1 ? sb / pb : pb / sb); *analog = an; *fsv = fs;
    return TSR_OK;
}
static double sig_postprocess_wn(double WN, int analog, double fsv)
{
    double wn = analog ? WN : atan(WN) * 2.0 / M_PI;
    if (fsv > 0.0) wn *= fsv / 2.0;
    return wn;
}

/* buttord(wp, ws, gpass, gstop, analog=False, fs=None): Butterworth order and natural frequency
   (scipy.signal.buttord), lowpass/highpass. */
static int r_buttord(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int ft, analog; double pb, nat, fsv;
    int rc = sig_ord_prep(args, nargs, &ft, &pb, &nat, &analog, &fsv); if (rc != TSR_OK) return rc;
    const double gpass = args[2].num, gstop = args[3].num;
    const double GSTOP = pow(10.0, 0.1 * fabs(gstop)), GPASS = pow(10.0, 0.1 * fabs(gpass));
    const int ord = (int)ceil(log10((GSTOP - 1.0) / (GPASS - 1.0)) / (2.0 * log10(nat)));
    const double W0 = pow(GPASS - 1.0, -1.0 / (2.0 * (double)ord));
    const double WN = (ft == 1) ? W0 * pb : pb / W0;
    fn_result_int(&res[0], ord);
    fn_result_num(&res[1], sig_postprocess_wn(WN, analog, fsv));
    return TSR_OK;
}

/* cheb1ord(wp, ws, gpass, gstop, analog=False, fs=None): Chebyshev-I order and natural frequency
   (scipy.signal.cheb1ord), lowpass/highpass. */
static int r_cheb1ord(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int ft, analog; double pb, nat, fsv;
    int rc = sig_ord_prep(args, nargs, &ft, &pb, &nat, &analog, &fsv); if (rc != TSR_OK) return rc;
    const double gpass = args[2].num, gstop = args[3].num;
    const double GSTOP = pow(10.0, 0.1 * fabs(gstop)), GPASS = pow(10.0, 0.1 * fabs(gpass));
    const double v = acosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0)));
    const int ord = (int)ceil(v / acosh(nat));
    fn_result_int(&res[0], ord);
    fn_result_num(&res[1], sig_postprocess_wn(pb, analog, fsv));   /* natural freq = passband edge */
    return TSR_OK;
}

/* cheb2ord(wp, ws, gpass, gstop, analog=False, fs=None): Chebyshev-II order and natural frequency
   (scipy.signal.cheb2ord), lowpass/highpass. */
static int r_cheb2ord(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int ft, analog; double pb, nat, fsv;
    int rc = sig_ord_prep(args, nargs, &ft, &pb, &nat, &analog, &fsv); if (rc != TSR_OK) return rc;
    const double gpass = args[2].num, gstop = args[3].num;
    const double GSTOP = pow(10.0, 0.1 * fabs(gstop)), GPASS = pow(10.0, 0.1 * fabs(gpass));
    const double v = acosh(sqrt((GSTOP - 1.0) / (GPASS - 1.0)));
    const int ord = (int)ceil(v / acosh(nat));
    const double new_freq = 1.0 / cosh((1.0 / (double)ord) * v);
    const double WN = (ft == 1) ? pb / new_freq : pb * new_freq;
    fn_result_int(&res[0], ord);
    fn_result_num(&res[1], sig_postprocess_wn(WN, analog, fsv));
    return TSR_OK;
}

/* ellipord(wp, ws, gpass, gstop, analog=False, fs=None): elliptic order and natural frequency
   (scipy.signal.ellipord), lowpass/highpass. */
static int r_ellipord(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int ft, analog; double pb, nat, fsv; (void)ft;
    int rc = sig_ord_prep(args, nargs, &ft, &pb, &nat, &analog, &fsv); if (rc != TSR_OK) return rc;
    const double gpass = args[2].num, gstop = args[3].num;
    const double arg1_sq = expm1(0.1 * gpass * M_LN10) / expm1(0.1 * gstop * M_LN10);   /* _pow10m1 ratio */
    const double m0 = 1.0 / (nat * nat);
    double k0, km1_0, k1, km1_1;
    tsr_special_ellipk(NULL, &m0, &k0); tsr_special_ellipkm1(NULL, &m0, &km1_0);
    tsr_special_ellipk(NULL, &arg1_sq, &k1); tsr_special_ellipkm1(NULL, &arg1_sq, &km1_1);
    const int ord = (int)ceil(k0 * km1_1 / (km1_0 * k1));
    fn_result_int(&res[0], ord);
    fn_result_num(&res[1], sig_postprocess_wn(pb, analog, fsv));
    return TSR_OK;
}

/* firwin(numtaps, cutoff, window='hamming', pass_zero=True, scale=True, fs=None): FIR filter design by the
   window method (scipy.signal.firwin). cutoff is a 1-D array of band edges (normalised by fs/2); the passbands
   are built from pass_zero, summed as windowed sinc differences, and (if scale) normalised to unit gain at the
   first passband's reference frequency. Default window is a symmetric Hamming. */
static int r_firwin(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 1) { fn_set_error("firwin: numtaps must be an integer"); return TSR_EARG; }
    const int64_t M = (int64_t)args[0].num;
    if (M < 1) { fn_set_error("firwin: numtaps must be >= 1"); return TSR_EARG; }
    if (args[1].kind != 3) { fn_set_error("firwin: cutoff must be an array"); return TSR_EARG; }
    int64_t nc; double *cut0 = fn_arg_doubles(&args[1], &nc); if (!cut0) return TSR_ENOMEM;
    const int pass_zero = (nargs > 3 && args[3].kind == 4) ? (args[3].num != 0.0) : 1;
    const int do_scale = (nargs > 4 && args[4].kind == 4) ? (args[4].num != 0.0) : 1;
    const double fs = (nargs > 5 && args[5].kind == 1) ? args[5].num : 2.0;
    const double nyq = 0.5 * fs;
    int64_t nwin = 0; double *warr = (nargs > 2 && args[2].kind == 3) ? fn_arg_doubles(&args[2], &nwin) : NULL;
    const int win_boxcar = (nargs > 2 && args[2].kind == 2 && args[2].str && !strcmp(args[2].str, "boxcar"));
    const int win_hann = (nargs > 2 && args[2].kind == 2 && args[2].str && !strcmp(args[2].str, "hann"));
    int rc = TSR_OK;
    const int pass_nyq = ((nc % 2 == 0) == pass_zero);
    double *cut = (double *)malloc(sizeof(double) * (size_t)(nc + 2));
    double *h = (double *)calloc((size_t)M, sizeof(double));
    double *win = (double *)malloc(sizeof(double) * (size_t)M);
    if (!cut || !h || !win || (nargs > 2 && args[2].kind == 3 && !warr)) rc = TSR_ENOMEM;
    else if (pass_nyq && M % 2 == 0) { fn_set_error("firwin: an even-length filter must have zero response at Nyquist"); rc = TSR_EARG; }
    else if (warr && nwin != M) { fn_set_error("firwin: window length must equal numtaps"); rc = TSR_EARG; }
    else {
        int64_t ne = 0;                                      /* assemble band edges: [0?] cutoff [1?] */
        if (pass_zero) cut[ne++] = 0.0;
        for (int64_t i = 0; i < nc; i++) cut[ne++] = cut0[i] / nyq;
        if (pass_nyq) cut[ne++] = 1.0;
        const double alpha = 0.5 * (double)(M - 1);
        for (int64_t b = 0; b + 1 < ne; b += 2) {
            const double left = cut[b], right = cut[b + 1];
            for (int64_t n = 0; n < M; n++) { const double m = (double)n - alpha; h[n] += right * sig_sinc(right * m) - left * sig_sinc(left * m); }
        }
        for (int64_t n = 0; n < M; n++) {                    /* symmetric window (fftbins=False) */
            if (warr) win[n] = warr[n];
            else if (win_boxcar) win[n] = 1.0;
            else if (win_hann) win[n] = (M == 1) ? 1.0 : 0.5 - 0.5 * cos(2.0 * M_PI * (double)n / (double)(M - 1));
            else win[n] = (M == 1) ? 1.0 : 0.54 - 0.46 * cos(2.0 * M_PI * (double)n / (double)(M - 1));   /* hamming */
            h[n] *= win[n];
        }
        if (do_scale) {
            const double left = cut[0], right = cut[1];
            const double sf = (left == 0.0) ? 0.0 : (right == 1.0 ? 1.0 : 0.5 * (left + right));
            double s = 0.0; for (int64_t n = 0; n < M; n++) s += h[n] * cos(M_PI * ((double)n - alpha) * sf);
            if (s != 0.0) for (int64_t n = 0; n < M; n++) h[n] /= s;
        }
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){M});
        if (!out) rc = TSR_ENOMEM; else memcpy(out, h, sizeof(double) * (size_t)M);
    }
    free(cut); free(h); free(win); if (warr) fn_free_doubles(warr, nwin);
    fn_free_doubles(cut0, nc);
    return rc;
}

/* firwin2(numtaps, freq, gain, nfreqs=None, window='hamming', antisymmetric=False, fs=None): FIR design by
   frequency sampling (scipy.signal.firwin2, antisymmetric=False). Linearly interpolates the desired (freq,
   gain) response onto a uniform mesh, applies the linear-phase shift, inverse-FFTs, and windows. */
static int r_firwin2(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 1 || args[1].kind != 3 || args[2].kind != 3) { fn_set_error("firwin2: numtaps int, freq and gain arrays"); return TSR_EARG; }
    const int64_t M = (int64_t)args[0].num;
    const double fs = (nargs > 6 && args[6].kind == 1) ? args[6].num : 2.0;
    const double nyq = 0.5 * fs;
    if (nargs > 5 && args[5].kind == 4 && args[5].num != 0.0) { fn_set_error("firwin2: antisymmetric=True is not yet supported"); return TSR_EARG; }
    int64_t nf, ng; double *freq = fn_arg_doubles(&args[1], &nf); if (!freq) return TSR_ENOMEM;
    double *gain = fn_arg_doubles(&args[2], &ng); if (!gain) { fn_free_doubles(freq, nf); return TSR_ENOMEM; }
    int rc = TSR_OK;
    int64_t nwin = 0; double *warr = (nargs > 4 && args[4].kind == 3) ? fn_arg_doubles(&args[4], &nwin) : NULL;
    const int win_boxcar = (nargs > 4 && args[4].kind == 2 && args[4].str && !strcmp(args[4].str, "boxcar"));
    const int win_hann = (nargs > 4 && args[4].kind == 2 && args[4].str && !strcmp(args[4].str, "hann"));
    if (M < 1 || nf != ng || nf < 2) { rc = TSR_EARG; fn_set_error("firwin2: bad numtaps or freq/gain lengths"); }
    else if (freq[0] != 0.0 || fabs(freq[nf - 1] - nyq) > 1e-12 * nyq) { rc = TSR_EARG; fn_set_error("firwin2: freq must start at 0 and end at fs/2"); }
    else if (warr && nwin != M) { rc = TSR_EARG; fn_set_error("firwin2: window length must equal numtaps"); }
    else {
        int64_t nfreqs = (nargs > 3 && args[3].kind == 1) ? (int64_t)args[3].num : 1 + (1LL << (int)ceil(log2((double)M)));
        const int64_t nfull = 2 * (nfreqs - 1);
        double *fx2 = (double *)malloc(sizeof(double) * (size_t)(2 * nfreqs));
        double *outf = (double *)malloc(sizeof(double) * (size_t)nfull);
        double *out = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){M});
        if (!fx2 || !outf || !out) rc = TSR_ENOMEM;
        else {
            int64_t seg = 0;
            for (int64_t i = 0; i < nfreqs; i++) {
                const double xi = (double)i * nyq / (double)(nfreqs - 1);
                while (seg < nf - 2 && freq[seg + 1] < xi) seg++;
                double fxi;                                  /* linear interpolation of gain at xi */
                if (xi <= freq[0]) fxi = gain[0];
                else if (xi >= freq[nf - 1]) fxi = gain[nf - 1];
                else { const double d = freq[seg + 1] - freq[seg]; fxi = d > 0 ? gain[seg] + (gain[seg + 1] - gain[seg]) * (xi - freq[seg]) / d : gain[seg]; }
                const double ph = -0.5 * (double)(M - 1) * M_PI * xi / nyq;   /* shift = exp(-i*(M-1)/2*pi*x/nyq) */
                fx2[2 * i] = fxi * cos(ph); fx2[2 * i + 1] = fxi * sin(ph);
            }
            tsr_irfft(nfull, 1, fx2, outf);
            for (int64_t n = 0; n < M; n++) {
                double w;
                if (warr) w = warr[n];
                else if (win_boxcar) w = 1.0;
                else if (win_hann) w = (M == 1) ? 1.0 : 0.5 - 0.5 * cos(2.0 * M_PI * (double)n / (double)(M - 1));
                else w = (M == 1) ? 1.0 : 0.54 - 0.46 * cos(2.0 * M_PI * (double)n / (double)(M - 1));   /* hamming */
                out[n] = outf[n] * w;
            }
        }
        free(fx2); free(outf);
    }
    if (warr) fn_free_doubles(warr, nwin);
    fn_free_doubles(freq, nf); fn_free_doubles(gain, ng);
    return rc;
}

/* emit a sorted complex array (interleaved) + real gain as (z, p, k) results. zo/po are double complex buffers
   (layout-compatible with interleaved doubles); consumed by the caller. */
static int sig_emit_zpk(double complex *zo, int64_t nz, double complex *po, int64_t np_, double k, tsr_result *res)
{
    qsort(zo, (size_t)(nz ? nz : 1), sizeof(double complex), sig_cplx_cmp);
    qsort(po, (size_t)(np_ ? np_ : 1), sizeof(double complex), sig_cplx_cmp);
    double *oz = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){nz});
    double *op = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){np_});
    if ((nz && !oz) || (np_ && !op) || !oz || !op) return TSR_ENOMEM;
    if (nz) memcpy(oz, zo, sizeof(double complex) * (size_t)nz);
    if (np_) memcpy(op, po, sizeof(double complex) * (size_t)np_);
    fn_result_num(&res[2], k);
    return TSR_OK;
}

/* lp2lp_zpk(z, p, k, wo=1.0): scale an analog lowpass zpk prototype to cutoff wo (scipy.signal.lp2lp_zpk). */
static int r_lp2lp_zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nz, np_; double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (args[2].kind == 1) ? args[2].num : 1.0;
    const double wo = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1.0;
    const int64_t degree = np_ - nz;
    for (int64_t i = 0; i < nz; i++) z[i] *= wo;
    for (int64_t i = 0; i < np_; i++) p[i] *= wo;
    int rc = sig_emit_zpk(z, nz, p, np_, k * pow(wo, (double)degree), res);
    free(z); free(p);
    return rc;
}

/* lp2hp_zpk(z, p, k, wo=1.0): analog lowpass zpk -> highpass (scipy.signal.lp2hp_zpk). */
static int r_lp2hp_zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nz, np_; double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (args[2].kind == 1) ? args[2].num : 1.0;
    const double wo = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1.0;
    const int64_t degree = np_ - nz;
    double complex pz = 1.0, pp = 1.0;
    for (int64_t i = 0; i < nz; i++) pz *= -z[i];
    for (int64_t i = 0; i < np_; i++) pp *= -p[i];
    double complex *zo = (double complex *)malloc(sizeof(double complex) * (size_t)((nz + degree) ? (nz + degree) : 1));
    double complex *po = (double complex *)malloc(sizeof(double complex) * (size_t)(np_ ? np_ : 1));
    int rc;
    if (!zo || !po) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < nz; i++) zo[i] = wo / z[i];
        for (int64_t i = 0; i < degree; i++) zo[nz + i] = 0.0;
        for (int64_t i = 0; i < np_; i++) po[i] = wo / p[i];
        rc = sig_emit_zpk(zo, nz + degree, po, np_, k * creal(pz / pp), res);
    }
    free(zo); free(po); free(z); free(p);
    return rc;
}

/* bilinear_zpk(z, p, k, fs): bilinear transform of an analog zpk to digital (scipy.signal.bilinear_zpk). */
static int r_bilinear_zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    int64_t nz, np_; double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (args[2].kind == 1) ? args[2].num : 1.0;
    const double fs = (args[3].kind == 1) ? args[3].num : 1.0;
    const double fs2 = 2.0 * fs;
    const int64_t degree = np_ - nz;
    double complex pz = 1.0, pp = 1.0;
    for (int64_t i = 0; i < nz; i++) pz *= (fs2 - z[i]);
    for (int64_t i = 0; i < np_; i++) pp *= (fs2 - p[i]);
    double complex *zo = (double complex *)malloc(sizeof(double complex) * (size_t)((nz + degree) ? (nz + degree) : 1));
    double complex *po = (double complex *)malloc(sizeof(double complex) * (size_t)(np_ ? np_ : 1));
    int rc;
    if (!zo || !po) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < nz; i++) zo[i] = (fs2 + z[i]) / (fs2 - z[i]);
        for (int64_t i = 0; i < degree; i++) zo[nz + i] = -1.0;
        for (int64_t i = 0; i < np_; i++) po[i] = (fs2 + p[i]) / (fs2 - p[i]);
        rc = sig_emit_zpk(zo, nz + degree, po, np_, k * creal(pz / pp), res);
    }
    free(zo); free(po); free(z); free(p);
    return rc;
}

/* convolve2d(in1, in2, mode='full', boundary='fill', fillvalue=0) / correlate2d(...): 2-D convolution and
   cross-correlation, the zero-fill-boundary case handled by the shared N-D direct-sum core (scipy.signal). */
static int conv2d_common(const tsr_arg *args, int nargs, int reverse_v, const char *who, tsr_result *res)
{
    if (args[0].kind != 3 || args[0].arr.ndim != 2 || args[1].kind != 3 || args[1].arr.ndim != 2) { fn_set_error("%s: in1 and in2 must be 2-D arrays", who); return TSR_EARG; }
    if (nargs > 3 && args[3].kind == 2 && args[3].str && strcmp(args[3].str, "fill") != 0) { fn_set_error("%s: only boundary='fill' is supported", who); return TSR_EARG; }
    if (nargs > 4 && args[4].kind == 1 && args[4].num != 0.0) { fn_set_error("%s: only fillvalue=0 is supported", who); return TSR_EARG; }
    int mode = CONV_FULL;
    if (nargs > 2 && parse_mode(&args[2], &mode) != TSR_OK) return TSR_EARG;
    return nd_convolve(&args[0], &args[1], mode, reverse_v, reverse_v, who, res);
}
static int r_convolve2d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return conv2d_common(args, nargs, 0, "convolve2d", res); }
static int r_correlate2d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{ (void)ctx; (void)nres; return conv2d_common(args, nargs, 1, "correlate2d", res); }

/* N-D order-statistic filter over an arbitrary footprint, zero-padded at the borders (scipy.ndimage.rank_filter
   with mode='constant', cval=0). The footprint is centred over each pixel (fshape[d] must be odd). For each
   output pixel the selected neighbours (footprint nonzero; out-of-bounds taken as 0) are sorted ascending and
   the rank-th (0-based) element is returned. Backs scipy.signal.order_filter/medfilt/medfilt2d. */
static int sig_order_filter_nd(const double *a, const int64_t *shape, int32_t nd,
                               const int64_t *fshape, const unsigned char *fp, int64_t rank,
                               const char *who, tsr_result *res)
{
    int64_t total = 1, ftot = 1, fcount = 0;
    for (int32_t d = 0; d < nd; d++) { total *= shape[d]; ftot *= fshape[d]; }
    for (int64_t i = 0; i < ftot; i++) if (fp[i]) fcount++;
    if (fcount == 0) { fn_set_error("%s: footprint has no nonzero element", who); return TSR_EARG; }
    if (rank < 0) rank += fcount;
    if (rank < 0 || rank >= fcount) { fn_set_error("%s: rank is out of range", who); return TSR_EARG; }
    double *out = (double *)fn_result_array(res, TSR_F64, nd, (int64_t *)shape);
    if (!out) return TSR_ENOMEM;
    double *win = (double *)malloc(sizeof(double) * (size_t)fcount);
    if (!win) return TSR_ENOMEM;
    int64_t ist[TSR_MAXDIM], c[TSR_MAXDIM];
    { int64_t acc = 1; for (int32_t d = nd - 1; d >= 0; d--) { ist[d] = acc; acc *= shape[d]; } }
    for (int32_t d = 0; d < nd; d++) c[d] = fshape[d] / 2;
    int64_t O[TSR_MAXDIM]; for (int32_t d = 0; d < nd; d++) O[d] = 0;
    for (int64_t idx = 0; idx < total; idx++) {
        int64_t w = 0, K[TSR_MAXDIM]; for (int32_t d = 0; d < nd; d++) K[d] = 0;
        for (int64_t fi = 0; fi < ftot; fi++) {
            if (fp[fi]) {
                int64_t off = 0; int inb = 1;
                for (int32_t d = 0; d < nd; d++) {
                    const int64_t ii = O[d] + K[d] - c[d];
                    if (ii < 0 || ii >= shape[d]) inb = 0;
                    off += ii * ist[d];
                }
                win[w++] = inb ? a[off] : 0.0;
            }
            for (int32_t d = nd - 1; d >= 0; d--) { if (++K[d] < fshape[d]) break; K[d] = 0; }
        }
        qsort(win, (size_t)fcount, sizeof(double), cmp_double_sig);
        out[idx] = win[rank];
        for (int32_t d = nd - 1; d >= 0; d--) { if (++O[d] < shape[d]) break; O[d] = 0; }
    }
    free(win);
    return TSR_OK;
}

/* a scalar or an nd-length sequence/array -> one odd kernel size per axis (default dflt). */
static int sig_perax_odd(const tsr_arg *a, int32_t nd, int64_t dflt, int64_t *out, const char *who)
{
    if (!a || a->kind == 0) { for (int32_t d = 0; d < nd; d++) out[d] = dflt; }
    else if (a->kind == 1) { const int64_t v = (int64_t)a->num; for (int32_t d = 0; d < nd; d++) out[d] = v; }
    else if (a->kind == 3 && a->arr.ndim == 1 && a->arr.shape[0] == nd) {
        int64_t t; double *p = fn_arg_doubles(a, &t); if (!p) return TSR_ENOMEM;
        for (int32_t d = 0; d < nd; d++) out[d] = (int64_t)p[d]; fn_free_doubles(p, t);
    } else { fn_set_error("%s: kernel_size must be a scalar or a length-%d sequence", who, (int)nd); return TSR_EARG; }
    for (int32_t d = 0; d < nd; d++) if ((out[d] % 2) != 1) { fn_set_error("%s: each element of kernel_size should be odd", who); return TSR_EARG; }
    return TSR_OK;
}

/* medfilt(volume, kernel_size=None): zero-padded N-D median filter (scipy.signal.medfilt). */
static int r_medfilt(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3) { fn_set_error("medfilt: volume must be an array"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (nd < 1 || nd > TSR_MAXDIM) { fn_set_error("medfilt: unsupported dimensionality"); return TSR_EARG; }
    int64_t ks[TSR_MAXDIM];
    int rc = sig_perax_odd(nargs > 1 ? &args[1] : NULL, nd, 3, ks, "medfilt");
    if (rc != TSR_OK) return rc;
    int64_t V = 1; for (int32_t d = 0; d < nd; d++) V *= ks[d];
    unsigned char *fp = (unsigned char *)malloc((size_t)V);
    if (!fp) return TSR_ENOMEM;
    memset(fp, 1, (size_t)V);
    int64_t na; double *a = fn_arg_doubles(&args[0], &na);
    if (!a) { free(fp); return TSR_ENOMEM; }
    rc = sig_order_filter_nd(a, args[0].arr.shape, nd, ks, fp, V / 2, "medfilt", &res[0]);
    free(fp); fn_free_doubles(a, na);
    return rc;
}

/* medfilt2d(input, kernel_size=3): zero-padded 2-D median filter (scipy.signal.medfilt2d). */
static int r_medfilt2d(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("medfilt2d: input must be a 2-D array"); return TSR_EARG; }
    int64_t ks[2];
    int rc = sig_perax_odd(nargs > 1 ? &args[1] : NULL, 2, 3, ks, "medfilt2d");
    if (rc != TSR_OK) return rc;
    const int64_t V = ks[0] * ks[1];
    unsigned char *fp = (unsigned char *)malloc((size_t)V);
    if (!fp) return TSR_ENOMEM;
    memset(fp, 1, (size_t)V);
    int64_t na; double *a = fn_arg_doubles(&args[0], &na);
    if (!a) { free(fp); return TSR_ENOMEM; }
    rc = sig_order_filter_nd(a, args[0].arr.shape, 2, ks, fp, V / 2, "medfilt2d", &res[0]);
    free(fp); fn_free_doubles(a, na);
    return rc;
}

/* order_filter(a, domain, rank): rank-th order statistic over the footprint `domain` (scipy.signal.order_filter). */
static int r_order_filter_sig(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("order_filter: a and domain must be arrays"); return TSR_EARG; }
    const int32_t nd = args[0].arr.ndim;
    if (args[1].arr.ndim != nd) { fn_set_error("order_filter: a and domain must have the same number of dimensions"); return TSR_EARG; }
    if (nd < 1 || nd > TSR_MAXDIM) { fn_set_error("order_filter: unsupported dimensionality"); return TSR_EARG; }
    int64_t fshape[TSR_MAXDIM], ftot = 1;
    for (int32_t d = 0; d < nd; d++) {
        fshape[d] = args[1].arr.shape[d]; ftot *= fshape[d];
        if ((fshape[d] % 2) != 1) { fn_set_error("order_filter: each dimension of domain should have an odd number of elements"); return TSR_EARG; }
    }
    const int64_t rank = (args[2].kind == 1) ? (int64_t)args[2].num : 0;
    int64_t nd2; double *dom = fn_arg_doubles(&args[1], &nd2);
    if (!dom) return TSR_ENOMEM;
    unsigned char *fp = (unsigned char *)malloc((size_t)(ftot > 0 ? ftot : 1));
    if (!fp) { fn_free_doubles(dom, nd2); return TSR_ENOMEM; }
    for (int64_t i = 0; i < ftot; i++) fp[i] = (dom[i] != 0.0) ? 1 : 0;
    fn_free_doubles(dom, nd2);
    int64_t na; double *a = fn_arg_doubles(&args[0], &na);
    if (!a) { free(fp); return TSR_ENOMEM; }
    int rc = sig_order_filter_nd(a, args[0].arr.shape, nd, fshape, fp, rank, "order_filter", &res[0]);
    free(fp); fn_free_doubles(a, na);
    return rc;
}

static const int SIG_ARGREL_MAX = 1, SIG_ARGREL_MIN = -1;
/* argrelmax/argrelmin (1-D, mode='clip'): indices of relative extrema -- points strictly greater (ctx=+1) or
   less (ctx=-1) than their `order` neighbours on each side, clamped at the edges (scipy.signal.argrel*). */
static int r_argrel(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)nres;
    const int want_max = ctx && *(const int *)ctx > 0;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("argrel: data must be a 1-D array"); return TSR_EARG; }
    const int64_t order = (nargs > 2 && args[2].kind == 1) ? (int64_t)args[2].num : 1;
    if (order < 1) { fn_set_error("argrel: order must be an int >= 1"); return TSR_EARG; }
    int64_t n; double *a = fn_arg_doubles(&args[0], &n); if (!a) return TSR_ENOMEM;
    int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n > 0 ? n : 1));
    int rc = TSR_OK;
    if (!idx) rc = TSR_ENOMEM;
    else {
        int64_t cnt = 0;
        for (int64_t i = 0; i < n; i++) {
            int ext = 1;
            for (int64_t s = 1; s <= order && ext; s++) {
                int64_t ip = i + s, im = i - s;                 /* clip boundary */
                if (ip > n - 1) ip = n - 1; if (im < 0) im = 0;
                const double hi = a[i];
                if (want_max) { if (!(hi > a[ip]) || !(hi > a[im])) ext = 0; }
                else { if (!(hi < a[ip]) || !(hi < a[im])) ext = 0; }
            }
            if (ext) idx[cnt++] = i;
        }
        int64_t *out = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, (int64_t[]){cnt});
        if (!out) rc = TSR_ENOMEM; else memcpy(out, idx, sizeof(int64_t) * (size_t)cnt);
    }
    free(idx); fn_free_doubles(a, n);
    return rc;
}

/* gauss_spline(x, n): Gaussian approximation to the B-spline basis of order n (scipy.signal.gauss_spline).
   1/sqrt(2*pi*s) * exp(-x^2/(2s)) with s = (n+1)/12. */
static int r_gauss_spline(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 1) { fn_set_error("gauss_spline: x must be an array and n an integer"); return TSR_EARG; }
    int64_t n; double *x = fn_arg_doubles(&args[0], &n); if (!x) return TSR_ENOMEM;
    const double s = ((double)args[1].num + 1.0) / 12.0;
    const double c = 1.0 / sqrt(2.0 * M_PI * s);
    double *out = (double *)fn_result_array(&res[0], TSR_F64, args[0].arr.ndim, args[0].arr.shape);
    int rc = TSR_OK;
    if (!out) rc = TSR_ENOMEM; else for (int64_t i = 0; i < n; i++) out[i] = c * exp(-x[i] * x[i] / 2.0 / s);
    fn_free_doubles(x, n);
    return rc;
}

/* vectorstrength(events, period): the vector strength (phase synchrony) and mean phase of events over a single
   period (scipy.signal.vectorstrength, scalar period). Returns (strength, phase). */
static int r_vectorstrength(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 1) { fn_set_error("vectorstrength: events must be an array and period a scalar"); return TSR_EARG; }
    const double period = args[1].num;
    if (period <= 0.0) { fn_set_error("vectorstrength: period must be positive"); return TSR_EARG; }
    int64_t ne; double *ev = fn_arg_doubles(&args[0], &ne); if (!ev) return TSR_ENOMEM;
    double sr = 0.0, si = 0.0;
    for (int64_t i = 0; i < ne; i++) { const double ph = 2.0 * M_PI * ev[i] / period; sr += cos(ph); si += sin(ph); }
    if (ne > 0) { sr /= (double)ne; si /= (double)ne; }
    fn_free_doubles(ev, ne);
    fn_result_num(&res[0], hypot(sr, si));
    fn_result_num(&res[1], atan2(si, sr));
    return TSR_OK;
}

/* cheb1ap(N, rp): the analog Chebyshev type I lowpass prototype (scipy.signal.cheb1ap): no zeros, poles on an
   ellipse in the left half-plane, gain set for the ripple rp (dB). (z, p, k), poles sorted by (re, im). */
static int r_cheb1ap(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 1 || args[1].kind != 1) { fn_set_error("cheb1ap: N and rp must be numbers"); return TSR_EARG; }
    const int64_t N = (int64_t)args[0].num; const double rp = args[1].num;
    if (N < 0) { fn_set_error("cheb1ap: filter order must be nonnegative"); return TSR_EARG; }
    if (N == 0) { double *z = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){0}); double *p = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){0}); if (!z || !p) return TSR_ENOMEM; fn_result_num(&res[2], pow(10.0, -rp / 20.0)); return TSR_OK; }
    const double eps = sqrt(pow(10.0, 0.1 * rp) - 1.0);
    const double mu = asinh(1.0 / eps) / (double)N;
    double complex *po = (double complex *)malloc(sizeof(double complex) * (size_t)N);
    double complex *zo = (double complex *)malloc(sizeof(double complex));   /* empty */
    int rc;
    if (!po || !zo) rc = TSR_ENOMEM;
    else {
        double complex prod = 1.0;
        for (int64_t i = 0; i < N; i++) { const double theta = M_PI * (double)(-N + 1 + 2 * i) / (2.0 * (double)N); po[i] = -csinh(mu + I * theta); prod *= -po[i]; }
        double k = creal(prod);
        if (N % 2 == 0) k /= sqrt(1.0 + eps * eps);
        rc = sig_emit_zpk(zo, 0, po, N, k, res);
    }
    free(po); free(zo);
    return rc;
}

/* cheb2ap(N, rs): the analog Chebyshev type II (inverse Chebyshev) lowpass prototype (scipy.signal.cheb2ap):
   zeros on the imaginary axis, poles, gain set for the stopband attenuation rs (dB). (z, p, k), sorted. */
static int r_cheb2ap(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 1 || args[1].kind != 1) { fn_set_error("cheb2ap: N and rs must be numbers"); return TSR_EARG; }
    const int64_t N = (int64_t)args[0].num; const double rs = args[1].num;
    if (N < 0) { fn_set_error("cheb2ap: filter order must be nonnegative"); return TSR_EARG; }
    if (N == 0) { double *z = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){0}); double *p = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){0}); if (!z || !p) return TSR_ENOMEM; fn_result_num(&res[2], 1.0); return TSR_OK; }
    const double de = 1.0 / sqrt(pow(10.0, 0.1 * rs) - 1.0);
    const double mu = asinh(1.0 / de) / (double)N;
    const int64_t nz = (N % 2) ? N - 1 : N;
    double complex *zo = (double complex *)malloc(sizeof(double complex) * (size_t)(nz ? nz : 1));
    double complex *po = (double complex *)malloc(sizeof(double complex) * (size_t)N);
    int rc;
    if (!zo || !po) rc = TSR_ENOMEM;
    else {
        int64_t zi = 0;                                      /* zeros: m = (-N+1..-2, 2..N-1) for odd N, else -N+1..N-1 */
        for (int64_t m = -N + 1; m <= N - 1; m += 2) { if ((N % 2) && m == 0) continue; zo[zi++] = I / sin((double)m * M_PI / (2.0 * (double)N)); }
        double complex pz = 1.0, pp = 1.0;
        for (int64_t i = 0; i < N; i++) { const double theta = M_PI * (double)(-N + 1 + 2 * i) / (2.0 * (double)N); po[i] = -1.0 / csinh(mu + I * theta); pp *= -po[i]; }
        for (int64_t i = 0; i < nz; i++) pz *= -zo[i];
        rc = sig_emit_zpk(zo, nz, po, N, creal(pp / pz), res);
    }
    free(zo); free(po);
    return rc;
}

/* lp2bp_zpk(z, p, k, wo=1.0, bw=1.0): analog lowpass zpk -> bandpass (scipy.signal.lp2bp_zpk). */
static int r_lp2bp_zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nz, np_; double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (args[2].kind == 1) ? args[2].num : 1.0;
    const double wo = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1.0;
    const double bw = (nargs > 4 && args[4].kind == 1) ? args[4].num : 1.0;
    const int64_t degree = np_ - nz;
    double complex *zo = (double complex *)malloc(sizeof(double complex) * (size_t)((2 * nz + degree) ? (2 * nz + degree) : 1));
    double complex *po = (double complex *)malloc(sizeof(double complex) * (size_t)((2 * np_) ? (2 * np_) : 1));
    int rc;
    if (!zo || !po) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < nz; i++) { const double complex zl = z[i] * bw / 2.0, s = csqrt(zl * zl - wo * wo); zo[i] = zl + s; zo[nz + i] = zl - s; }
        for (int64_t i = 0; i < degree; i++) zo[2 * nz + i] = 0.0;
        for (int64_t i = 0; i < np_; i++) { const double complex pl = p[i] * bw / 2.0, s = csqrt(pl * pl - wo * wo); po[i] = pl + s; po[np_ + i] = pl - s; }
        rc = sig_emit_zpk(zo, 2 * nz + degree, po, 2 * np_, k * pow(bw, (double)degree), res);
    }
    free(zo); free(po); free(z); free(p);
    return rc;
}

/* lp2bs_zpk(z, p, k, wo=1.0, bw=1.0): analog lowpass zpk -> bandstop (scipy.signal.lp2bs_zpk). */
static int r_lp2bs_zpk(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    int64_t nz, np_; double complex *z = read_carr(&args[0], &nz); if (!z) return TSR_ENOMEM;
    double complex *p = read_carr(&args[1], &np_); if (!p) { free(z); return TSR_ENOMEM; }
    const double k = (args[2].kind == 1) ? args[2].num : 1.0;
    const double wo = (nargs > 3 && args[3].kind == 1) ? args[3].num : 1.0;
    const double bw = (nargs > 4 && args[4].kind == 1) ? args[4].num : 1.0;
    const int64_t degree = np_ - nz;
    double complex pz = 1.0, pp = 1.0;
    for (int64_t i = 0; i < nz; i++) pz *= -z[i];
    for (int64_t i = 0; i < np_; i++) pp *= -p[i];
    double complex *zo = (double complex *)malloc(sizeof(double complex) * (size_t)((2 * nz + 2 * degree) ? (2 * nz + 2 * degree) : 1));
    double complex *po = (double complex *)malloc(sizeof(double complex) * (size_t)((2 * np_) ? (2 * np_) : 1));
    int rc;
    if (!zo || !po) rc = TSR_ENOMEM;
    else {
        for (int64_t i = 0; i < nz; i++) { const double complex zh = (bw / 2.0) / z[i], s = csqrt(zh * zh - wo * wo); zo[i] = zh + s; zo[nz + i] = zh - s; }
        for (int64_t i = 0; i < degree; i++) { zo[2 * nz + i] = I * wo; zo[2 * nz + degree + i] = -I * wo; }
        for (int64_t i = 0; i < np_; i++) { const double complex ph = (bw / 2.0) / p[i], s = csqrt(ph * ph - wo * wo); po[i] = ph + s; po[np_ + i] = ph - s; }
        rc = sig_emit_zpk(zo, 2 * nz + 2 * degree, po, 2 * np_, k * creal(pz / pp), res);
    }
    free(zo); free(po); free(z); free(p);
    return rc;
}

/* buttap(N): the analog Butterworth lowpass prototype of order N (scipy.signal.buttap): no zeros, poles
   p = -exp(1j*pi*m/(2N)) for m = -N+1, -N+3, ..., N-1, gain 1. Poles returned sorted by (re, im). (z, p, k). */
static int r_buttap(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 1) { fn_set_error("buttap: N must be an integer"); return TSR_EARG; }
    const int64_t N = (int64_t)args[0].num;
    if (N < 0) { fn_set_error("buttap: filter order must be nonnegative"); return TSR_EARG; }
    double *z = (double *)fn_result_array(&res[0], TSR_C128, 1, (int64_t[]){0});
    double *p = (double *)fn_result_array(&res[1], TSR_C128, 1, (int64_t[]){N});
    if ((N > 0 && !p) || !z) return TSR_ENOMEM;
    double *tmp = (double *)malloc(sizeof(double) * (size_t)(2 * (N ? N : 1)));
    if (!tmp) return TSR_ENOMEM;
    for (int64_t i = 0; i < N; i++) { const double m = (double)(-N + 1 + 2 * i); const double ang = M_PI * m / (2.0 * (double)N); tmp[2 * i] = -cos(ang); tmp[2 * i + 1] = -sin(ang); }
    qsort(tmp, (size_t)N, 2 * sizeof(double), sig_cplx_cmp);
    for (int64_t i = 0; i < N; i++) { p[2 * i] = tmp[2 * i]; p[2 * i + 1] = tmp[2 * i + 1]; }
    free(tmp);
    fn_result_num(&res[2], 1.0);
    return TSR_OK;
}

/* deconvolve(signal, divisor): polynomial division signal = convolve(divisor, quotient) + remainder
   (scipy.signal.deconvolve). Returns (quotient, remainder); remainder has the length of signal. */
static int r_deconvolve(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres; (void)nargs;
    if (args[0].kind != 3 || args[1].kind != 3) { fn_set_error("deconvolve: signal and divisor must be arrays"); return TSR_EARG; }
    int64_t N, D; double *sig = fn_arg_doubles(&args[0], &N); if (!sig) return TSR_ENOMEM;
    double *den = fn_arg_doubles(&args[1], &D); if (!den) { fn_free_doubles(sig, N); return TSR_ENOMEM; }
    int rc = TSR_OK;
    if (D < 1 || den[0] == 0.0) { fn_set_error("deconvolve: divisor must be nonzero (den[0] != 0)"); rc = TSR_EARG; }
    else if (D > N) {                                        /* quotient empty, remainder = signal */
        double *q = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){0});
        double *r = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){N});
        if (!r) rc = TSR_ENOMEM; else { (void)q; memcpy(r, sig, sizeof(double) * (size_t)N); }
    } else {
        const int64_t L = N - D + 1;                         /* quotient length */
        double *work = (double *)malloc(sizeof(double) * (size_t)N);   /* long-division remainder */
        double *q = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){L});
        double *r = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){N});
        if (!work || !q || !r) rc = TSR_ENOMEM;
        else {
            for (int64_t i = 0; i < N; i++) work[i] = sig[i];
            for (int64_t i = 0; i < L; i++) { const double c = work[i] / den[0]; q[i] = c; for (int64_t j = 0; j < D; j++) work[i + j] -= c * den[j]; }
            for (int64_t i = 0; i < N; i++) r[i] = 0.0;       /* remainder = signal - convolve(quotient, divisor) */
            for (int64_t i = 0; i < L; i++) for (int64_t j = 0; j < D; j++) r[i + j] += q[i] * den[j];
            for (int64_t i = 0; i < N; i++) r[i] = sig[i] - r[i];
        }
        free(work);
    }
    fn_free_doubles(sig, N); fn_free_doubles(den, D);
    return rc;
}

/* correlation_lags(in1_len, in2_len, mode='full'): the lag indices for signal.correlate's output
   (scipy.signal.correlation_lags). Integer array. */
static int r_correlation_lags(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 1 || args[1].kind != 1) { fn_set_error("correlation_lags: in1_len and in2_len must be integers"); return TSR_EARG; }
    const int64_t n1 = (int64_t)args[0].num, n2 = (int64_t)args[1].num;
    int mode = 0;                                            /* 0 full, 1 same, 2 valid */
    if (nargs > 2 && args[2].kind == 2 && args[2].str) {
        if (!strcmp(args[2].str, "same")) mode = 1; else if (!strcmp(args[2].str, "valid")) mode = 2;
        else if (strcmp(args[2].str, "full")) { fn_set_error("correlation_lags: mode must be 'full', 'same' or 'valid'"); return TSR_EARG; }
    }
    int64_t lo, hi;                                          /* lags are the half-open integer range [lo, hi) */
    if (mode == 0) { lo = -n2 + 1; hi = n1; }
    else if (mode == 1) { const int64_t L = n1 + n2 - 1, mid = L / 2, lb = n1 / 2; const int64_t base = -n2 + 1; lo = base + (mid - lb); hi = base + (mid + lb) + (n1 % 2 ? 1 : 0); }
    else { const int64_t lb = n1 - n2; if (lb >= 0) { lo = 0; hi = lb + 1; } else { lo = lb; hi = 1; } }
    const int64_t len = hi > lo ? hi - lo : 0;
    int64_t *out = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, (int64_t[]){len});
    if (!out) return TSR_ENOMEM;
    for (int64_t i = 0; i < len; i++) out[i] = lo + i;
    return TSR_OK;
}

/* periodic Tukey window of length M with taper fraction alpha, into w (matches scipy windows.tukey sym=False). */
static void sig_tukey_periodic(double *w, int64_t M, double alpha)
{
    if (alpha <= 0.0) { for (int64_t i = 0; i < M; i++) w[i] = 1.0; return; }
    if (alpha >= 1.0) { for (int64_t i = 0; i < M; i++) w[i] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)M); return; }   /* hann periodic */
    const int64_t Me = M + 1;                                /* periodic: compute on M+1, drop the last */
    const int64_t width = (int64_t)(alpha * (double)(Me - 1) / 2.0);
    for (int64_t n = 0; n < M; n++) {
        if (n <= width) w[n] = 0.5 * (1.0 + cos(M_PI * (-1.0 + 2.0 * (double)n / alpha / (double)(Me - 1))));
        else if (n >= Me - width - 1) w[n] = 0.5 * (1.0 + cos(M_PI * (-2.0 / alpha + 1.0 + 2.0 * (double)n / alpha / (double)(Me - 1))));
        else w[n] = 1.0;
    }
}

/* spectrogram(x, fs=1.0, window=('tukey',0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant',
   return_onesided=True, scaling='density', axis=-1, mode='psd'): per-segment PSD over time (scipy.signal.
   spectrogram, mode='psd'). Default window is a periodic Tukey(0.25); noverlap defaults to nperseg//8.
   Returns (f, t, Sxx) with Sxx shaped (nfreq, nseg). */
static int r_spectrogram(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 1) { fn_set_error("spectrogram: x must be a 1-D real array"); return TSR_EARG; }
    if (nargs > 10 && args[10].kind == 2 && args[10].str && strcmp(args[10].str, "psd") != 0) { fn_set_error("spectrogram: only mode='psd' is supported"); return TSR_EARG; }
    int64_t nx; double *x = fn_arg_doubles(&args[0], &nx); if (!x) return TSR_ENOMEM;
    const double fs = (nargs > 1 && args[1].kind == 1) ? args[1].num : 1.0;
    int64_t nwin = 0; double *warr = (nargs > 2 && args[2].kind == 3) ? fn_arg_doubles(&args[2], &nwin) : NULL;
    const int win_boxcar = (nargs > 2 && args[2].kind == 2 && args[2].str && !strcmp(args[2].str, "boxcar"));
    const int win_hann = (nargs > 2 && args[2].kind == 2 && args[2].str && !strcmp(args[2].str, "hann"));
    int64_t nperseg = (nargs > 3 && args[3].kind == 1) ? (int64_t)args[3].num : (warr ? nwin : (nx < 256 ? nx : 256));
    if (nperseg > nx) nperseg = nx;
    const int64_t noverlap = (nargs > 4 && args[4].kind == 1) ? (int64_t)args[4].num : nperseg / 8;
    const int64_t nfft = (nargs > 5 && args[5].kind == 1) ? (int64_t)args[5].num : nperseg;
    int det = 1;
    if (nargs > 6) { if (args[6].kind == 2 && args[6].str) det = !strcmp(args[6].str, "linear") ? 2 : 1; else if (args[6].kind == 4 && args[6].num == 0.0) det = 0; }
    const int onesided = (nargs > 7 && args[7].kind == 4) ? (args[7].num != 0.0) : 1;
    const int density = !(nargs > 8 && args[8].kind == 2 && args[8].str && !strcmp(args[8].str, "spectrum"));
    const int64_t nstep = nperseg - noverlap;
    int rc = TSR_OK;
    if (nstep <= 0) { if (warr) fn_free_doubles(warr, nwin); fn_free_doubles(x, nx); fn_set_error("spectrogram: noverlap must be less than nperseg"); return TSR_EARG; }
    const int64_t nseg = nperseg <= nx ? 1 + (nx - nperseg) / nstep : 0;
    const int64_t nb = onesided ? nfft / 2 + 1 : nfft;
    double *win = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *seg = (double *)malloc(sizeof(double) * (size_t)nperseg);
    double *in = (double *)calloc((size_t)(onesided ? nfft : 2 * nfft), sizeof(double));
    double *Xf = (double *)malloc(sizeof(double) * (size_t)(2 * nb));
    if (!win || !seg || !in || !Xf || (nargs > 2 && args[2].kind == 3 && !warr)) rc = TSR_ENOMEM;
    else if (warr && nwin != nperseg) { fn_set_error("spectrogram: window length must equal nperseg"); rc = TSR_EARG; }
    else if (nseg < 1) { fn_set_error("spectrogram: nperseg exceeds the signal length"); rc = TSR_EARG; }
    else {
        if (warr) for (int64_t i = 0; i < nperseg; i++) win[i] = warr[i];
        else if (win_boxcar) for (int64_t i = 0; i < nperseg; i++) win[i] = 1.0;
        else if (win_hann) for (int64_t i = 0; i < nperseg; i++) win[i] = 0.5 - 0.5 * cos(2.0 * M_PI * (double)i / (double)nperseg);
        else sig_tukey_periodic(win, nperseg, 0.25);          /* default */
        double sw2 = 0, sw = 0; for (int64_t i = 0; i < nperseg; i++) { sw2 += win[i] * win[i]; sw += win[i]; }
        const double scale = density ? 1.0 / (fs * sw2) : 1.0 / (sw * sw);
        double *f = (double *)fn_result_array(&res[0], TSR_F64, 1, (int64_t[]){nb});
        double *t = (double *)fn_result_array(&res[1], TSR_F64, 1, (int64_t[]){nseg});
        double *S = (double *)fn_result_array(&res[2], TSR_F64, 2, (int64_t[]){nb, nseg});
        if (!f || !t || !S) rc = TSR_ENOMEM;
        else {
            for (int64_t k = 0; k < nb; k++) f[k] = onesided ? (double)k * fs / (double)nfft : ((k <= (nfft - 1) / 2 ? k : k - nfft) * fs / (double)nfft);
            for (int64_t s = 0; s < nseg; s++) {
                t[s] = ((double)nperseg / 2.0 + (double)(s * nstep)) / fs;
                const double *src = x + s * nstep;
                for (int64_t i = 0; i < nperseg; i++) seg[i] = src[i];
                if (det == 1) { double m = 0; for (int64_t i = 0; i < nperseg; i++) m += seg[i]; m /= (double)nperseg; for (int64_t i = 0; i < nperseg; i++) seg[i] -= m; }
                else if (det == 2 && nperseg > 1) { double st = 0, ss = 0, stt = 0, stx = 0; const double n = (double)nperseg; for (int64_t i = 0; i < nperseg; i++) { st += i; ss += seg[i]; stt += (double)i * i; stx += (double)i * seg[i]; } const double b = (n * stx - st * ss) / (n * stt - st * st), a = (ss - b * st) / n; for (int64_t i = 0; i < nperseg; i++) seg[i] -= a + b * (double)i; }
                for (int64_t i = 0; i < nperseg; i++) seg[i] *= win[i];
                if (onesided) { for (int64_t i = 0; i < nfft; i++) in[i] = i < nperseg ? seg[i] : 0.0; tsr_rfft(nfft, 1, in, Xf); }
                else { for (int64_t i = 0; i < nfft; i++) { in[2 * i] = i < nperseg ? seg[i] : 0.0; in[2 * i + 1] = 0.0; } tsr_fft(nfft, 1, 0, in, Xf); }
                for (int64_t k = 0; k < nb; k++) {
                    double p = (Xf[2 * k] * Xf[2 * k] + Xf[2 * k + 1] * Xf[2 * k + 1]) * scale;
                    if (onesided) { const int fold = (nfft % 2) ? (k >= 1) : (k >= 1 && k < nb - 1); if (fold) p *= 2.0; }
                    S[k * nseg + s] = p;
                }
            }
        }
    }
    free(win); free(seg); free(in); free(Xf); if (warr) fn_free_doubles(warr, nwin);
    fn_free_doubles(x, nx);
    return rc;
}

/* transpose an r x c interleaved-complex matrix into dst (c x r). */
static void sig_ctranspose(const double *src, int64_t r, int64_t c, double *dst)
{
    for (int64_t i = 0; i < r; i++)
        for (int64_t j = 0; j < c; j++) { dst[2 * (j * r + i)] = src[2 * (i * c + j)]; dst[2 * (j * r + i) + 1] = src[2 * (i * c + j) + 1]; }
}

/* hilbert2(x, N=None): the 2-D analytic signal of a real matrix via the 2-D FFT (scipy.signal.hilbert2).
   fft2 -> double the positive-frequency half-planes along each axis (outer product of the 1-D hilbert
   multipliers), zero the negative halves -> ifft2. N is the full x shape by default, or a single int for both. */
static int r_hilbert2(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    if (args[0].kind != 3 || args[0].arr.ndim != 2) { fn_set_error("hilbert2: x must be a 2-D real array"); return TSR_EARG; }
    const int64_t r0 = args[0].arr.shape[0], c0 = args[0].arr.shape[1];
    int64_t N0 = r0, N1 = c0;
    if (nargs > 1 && args[1].kind == 1) { N0 = N1 = (int64_t)args[1].num; if (N0 <= 0) { fn_set_error("hilbert2: N must be positive"); return TSR_EARG; } }
    int64_t nx; double *x = fn_arg_doubles(&args[0], &nx); if (!x) return TSR_ENOMEM;
    const size_t sz = (size_t)(2 * N0 * N1);
    double *a = (double *)calloc(sz, sizeof(double));
    double *b = (double *)malloc(sz * sizeof(double));
    double *t = (double *)malloc(sz * sizeof(double));
    double *b2 = (double *)malloc(sz * sizeof(double));
    double *Xf = (double *)malloc(sz * sizeof(double));
    double *out = (double *)fn_result_array(&res[0], TSR_C128, 2, (int64_t[]){N0, N1});
    int rc = TSR_OK;
    if (!a || !b || !t || !b2 || !Xf || !out) rc = TSR_ENOMEM;
    else {
        const int64_t mr = r0 < N0 ? r0 : N0, mc = c0 < N1 ? c0 : N1;
        for (int64_t i = 0; i < mr; i++) for (int64_t j = 0; j < mc; j++) a[2 * (i * N1 + j)] = x[i * c0 + j];
        tsr_fft(N1, N0, 0, a, b);                            /* FFT each row (axis 1) */
        sig_ctranspose(b, N0, N1, t);
        tsr_fft(N0, N1, 0, t, b2);                           /* FFT each column (axis 0), in transposed layout */
        sig_ctranspose(b2, N1, N0, Xf);                      /* Xf[i][j] = fft2(x) */
        const int64_t k0 = (N0 + 1) / 2, k1 = (N1 + 1) / 2;
        for (int64_t i = 0; i < N0; i++) {
            const double ri = (i == 0) ? 1.0 : (i < k0 ? 2.0 : 0.0);
            for (int64_t j = 0; j < N1; j++) {
                const double f = ri * ((j == 0) ? 1.0 : (j < k1 ? 2.0 : 0.0));
                Xf[2 * (i * N1 + j)] *= f; Xf[2 * (i * N1 + j) + 1] *= f;
            }
        }
        tsr_fft(N1, N0, 1, Xf, b);                           /* ifft rows (1/N1) */
        sig_ctranspose(b, N0, N1, t);
        tsr_fft(N0, N1, 1, t, b2);                           /* ifft cols (1/N0) */
        sig_ctranspose(b2, N1, N0, out);
    }
    free(a); free(b); free(t); free(b2); free(Xf); fn_free_doubles(x, nx);
    return rc;
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
    ROUTINE("signal.square", 1, "t, duty=0.5", "out", r_square, NULL, "Square-wave of period 2pi with the given duty cycle (scipy.signal.square)."),
    ROUTINE("signal.sawtooth", 1, "t, width=1", "out", r_sawtooth, NULL, "Sawtooth/triangle wave of period 2pi with the given rising width (scipy.signal.sawtooth)."),
    ROUTINE("signal.chirp", 4, "t, f0, t1, f1, method='linear', phi=0, vertex_zero=True", "out", r_chirp, NULL, "Frequency-swept cosine (linear/quadratic/logarithmic/hyperbolic) (scipy.signal.chirp)."),
    ROUTINE("signal.gausspulse", 1, "t, fc=1000, bw=0.5, bwr=-6", "out", r_gausspulse, NULL, "Gaussian-modulated sinusoid, in-phase component (scipy.signal.gausspulse)."),
    ROUTINE("signal.unit_impulse", 1, "shape, idx=None", "out", r_unit_impulse, NULL, "Unit impulse: zeros with a single 1 at idx (scipy.signal.unit_impulse)."),
    ROUTINE("signal.normalize", 2, "b, a", "b, a", r_signal_normalize, NULL, "Normalize a transfer-function representation (scipy.signal.normalize)."),
    ROUTINE("signal.tf2zpk", 3, "b, a", "z, p, k", r_tf2zpk, NULL, "Zeros, poles and gain from transfer-function coefficients (scipy.signal.tf2zpk)."),
    ROUTINE("signal.sos2tf", 2, "sos", "b, a", r_sos2tf, NULL, "Transfer function (b, a) from a second-order-sections cascade (scipy.signal.sos2tf)."),
    ROUTINE("signal.tf2ss", 4, "num, den", "A, B, C, D", r_tf2ss, NULL, "Controller-canonical state-space from transfer-function coefficients (scipy.signal.tf2ss)."),
    ROUTINE("signal.abcd_normalize", 4, "A, B, C, D", "A, B, C, D", r_abcd_normalize, NULL, "Validate and 2-D-normalize state-space matrices (scipy.signal.abcd_normalize)."),
    ROUTINE("signal.ss2tf", 2, "A, B, C, D, input=0", "num, den", r_ss2tf, NULL, "Transfer function (num, den) from a state-space system (scipy.signal.ss2tf)."),
    ROUTINE("signal.cont2discrete", 5, "A, B, C, D, dt, method='zoh', alpha=None", "Ad, Bd, Cd, Dd, dt", r_cont2discrete, NULL, "Continuous to discrete state-space conversion (scipy.signal.cont2discrete)."),
    ROUTINE("signal.unique_roots", 2, "p, tol=1e-3, rtype='min'", "unique, multiplicity", r_unique_roots, NULL, "Unique roots and multiplicities from a list of roots (scipy.signal.unique_roots)."),
    ROUTINE("signal.invres", 2, "r, p, k, tol=1e-3, rtype='avg'", "b, a", r_invres, NULL, "Transfer function (b, a) from a partial-fraction expansion, positive powers (scipy.signal.invres)."),
    ROUTINE("signal.invresz", 2, "r, p, k, tol=1e-3, rtype='avg'", "b, a", r_invresz, NULL, "Transfer function (b, a) from a partial-fraction expansion, negative powers of z (scipy.signal.invresz)."),
    ROUTINE("signal.residue", 3, "b, a, tol=1e-3, rtype='avg'", "r, p, k", r_residue, NULL, "Partial-fraction expansion of b(s)/a(s), positive powers (scipy.signal.residue)."),
    ROUTINE("signal.residuez", 3, "b, a, tol=1e-3, rtype='avg'", "r, p, k", r_residuez, NULL, "Partial-fraction expansion of b(z)/a(z), negative powers of z (scipy.signal.residuez)."),
    ROUTINE("signal.lp2lp", 2, "b, a, wo=1.0", "b, a", r_lp2lp, NULL, "Transform a lowpass analog prototype to a different cutoff (scipy.signal.lp2lp)."),
    ROUTINE("signal.lp2hp", 2, "b, a, wo=1.0", "b, a", r_lp2hp, NULL, "Transform a lowpass analog prototype to highpass (scipy.signal.lp2hp)."),
    ROUTINE("signal.lp2bp", 2, "b, a, wo=1.0, bw=1.0", "b, a", r_lp2bp, NULL, "Transform a lowpass analog prototype to bandpass (scipy.signal.lp2bp)."),
    ROUTINE("signal.lp2bs", 2, "b, a, wo=1.0, bw=1.0", "b, a", r_lp2bs, NULL, "Transform a lowpass analog prototype to bandstop (scipy.signal.lp2bs)."),
    ROUTINE("signal.bilinear", 2, "b, a, fs=1.0", "b, a", r_bilinear, NULL, "Bilinear transform of an analog filter to a digital filter (scipy.signal.bilinear)."),
    ROUTINE("signal.freqs", 2, "b, a, worN", "w, h", r_freqs, NULL, "Analog filter frequency response at the given frequencies (scipy.signal.freqs)."),
    ROUTINE("signal.freqs_zpk", 2, "z, p, k, worN", "w, h", r_freqs_zpk, NULL, "Analog zpk frequency response at the given frequencies (scipy.signal.freqs_zpk)."),
    ROUTINE("signal.freqz_zpk", 2, "z, p, k, worN=512, whole=False", "w, h", r_freqz_zpk, NULL, "Digital zpk frequency response on a linear grid (scipy.signal.freqz_zpk)."),
    ROUTINE("signal.sosfreqz", 2, "sos, worN=512, whole=False", "w, h", r_sosfreqz, NULL, "Frequency response of a second-order-sections cascade (scipy.signal.sosfreqz)."),
    ROUTINE("signal.group_delay", 2, "b, a, w=512, whole=False", "w, gd", r_group_delay, NULL, "Group delay of a digital filter (scipy.signal.group_delay)."),
    ROUTINE("signal.hilbert", 1, "x, N=None", "out", r_hilbert, NULL, "Analytic signal of a real sequence via the FFT (scipy.signal.hilbert)."),
    ROUTINE("signal.hilbert2", 1, "x, N=None", "out", r_hilbert2, NULL, "2-D analytic signal of a real matrix via the 2-D FFT (scipy.signal.hilbert2)."),
    ROUTINE("signal.periodogram", 2, "x, fs=1.0, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='density'", "f, Pxx", r_periodogram, NULL, "Power spectral density estimate from a single segment (scipy.signal.periodogram)."),
    ROUTINE("signal.welch", 2, "x, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean'", "f, Pxx", r_welch, NULL, "Welch's averaged-periodogram power spectral density estimate (scipy.signal.welch)."),
    ROUTINE("signal.csd", 2, "x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, average='mean'", "f, Pxy", r_csd, NULL, "Cross power spectral density by Welch's method (scipy.signal.csd)."),
    ROUTINE("signal.coherence", 2, "x, y, fs=1.0, window='hann', nperseg=None, noverlap=None, nfft=None, detrend='constant', axis=-1", "f, Cxy", r_coherence, NULL, "Magnitude-squared coherence by Welch's method (scipy.signal.coherence)."),
    ROUTINE("signal.spectrogram", 3, "x, fs=1.0, window='tukey', nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd'", "f, t, Sxx", r_spectrogram, NULL, "Spectrogram (per-segment PSD over time) by Welch's segmenting; default window periodic Tukey(0.25) (scipy.signal.spectrogram, mode='psd')."),
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
    ROUTINE("signal.correlation_lags", 1, "in1_len, in2_len, mode='full'", "out", r_correlation_lags, NULL, "Lag indices for the output of signal.correlate (scipy.signal.correlation_lags)."),
    ROUTINE("signal.resample", 1, "x, num, t=None, axis=0, window=None, domain='time'", "out", r_resample, NULL, "Resample a real signal to num samples via the FFT (scipy.signal.resample; window=None)."),
    ROUTINE("signal.firwin", 1, "numtaps, cutoff, window='hamming', pass_zero=True, scale=True, fs=None", "out", r_firwin, NULL, "FIR filter design by the window method (scipy.signal.firwin)."),
    ROUTINE("signal.kaiser_atten", 1, "numtaps, width", "out", r_kaiser_atten, NULL, "Kaiser-window attenuation (dB) for a filter length and transition width (scipy.signal.kaiser_atten)."),
    ROUTINE("signal.kaiser_beta", 1, "a", "out", r_kaiser_beta, NULL, "Kaiser-window shape beta for a given attenuation (scipy.signal.kaiser_beta)."),
    ROUTINE("signal.kaiserord", 2, "ripple, width", "numtaps, beta", r_kaiserord, NULL, "Kaiser filter length and beta for a ripple and transition width (scipy.signal.kaiserord)."),
    ROUTINE("signal.buttord", 2, "wp, ws, gpass, gstop, analog=False, fs=None", "ord, wn", r_buttord, NULL, "Butterworth filter order selection, lowpass/highpass (scipy.signal.buttord)."),
    ROUTINE("signal.cheb1ord", 2, "wp, ws, gpass, gstop, analog=False, fs=None", "ord, wn", r_cheb1ord, NULL, "Chebyshev type I filter order selection, lowpass/highpass (scipy.signal.cheb1ord)."),
    ROUTINE("signal.cheb2ord", 2, "wp, ws, gpass, gstop, analog=False, fs=None", "ord, wn", r_cheb2ord, NULL, "Chebyshev type II filter order selection, lowpass/highpass (scipy.signal.cheb2ord)."),
    ROUTINE("signal.ellipord", 2, "wp, ws, gpass, gstop, analog=False, fs=None", "ord, wn", r_ellipord, NULL, "Elliptic filter order selection, lowpass/highpass (scipy.signal.ellipord)."),
    ROUTINE("signal.firwin2", 1, "numtaps, freq, gain, nfreqs=None, window='hamming', antisymmetric=False, fs=None", "out", r_firwin2, NULL, "FIR filter design by frequency sampling (scipy.signal.firwin2)."),
    ROUTINE("signal.sos2zpk", 3, "sos", "z, p, k", r_sos2zpk, NULL, "Zeros, poles and gain from a second-order-sections cascade (scipy.signal.sos2zpk)."),
    ROUTINE("signal.deconvolve", 2, "signal, divisor", "quotient, remainder", r_deconvolve, NULL, "Deconvolve a divisor out of a signal by polynomial division (scipy.signal.deconvolve)."),
    ROUTINE("signal.buttap", 3, "N", "z, p, k", r_buttap, NULL, "Analog Butterworth lowpass prototype of order N (scipy.signal.buttap)."),
    ROUTINE("signal.lp2lp_zpk", 3, "z, p, k, wo=1.0", "z, p, k", r_lp2lp_zpk, NULL, "Scale an analog lowpass zpk prototype to a new cutoff (scipy.signal.lp2lp_zpk)."),
    ROUTINE("signal.lp2hp_zpk", 3, "z, p, k, wo=1.0", "z, p, k", r_lp2hp_zpk, NULL, "Transform an analog lowpass zpk prototype to highpass (scipy.signal.lp2hp_zpk)."),
    ROUTINE("signal.bilinear_zpk", 3, "z, p, k, fs", "z, p, k", r_bilinear_zpk, NULL, "Bilinear transform of an analog zpk to a digital zpk (scipy.signal.bilinear_zpk)."),
    ROUTINE("signal.lp2bp_zpk", 3, "z, p, k, wo=1.0, bw=1.0", "z, p, k", r_lp2bp_zpk, NULL, "Transform an analog lowpass zpk prototype to bandpass (scipy.signal.lp2bp_zpk)."),
    ROUTINE("signal.lp2bs_zpk", 3, "z, p, k, wo=1.0, bw=1.0", "z, p, k", r_lp2bs_zpk, NULL, "Transform an analog lowpass zpk prototype to bandstop (scipy.signal.lp2bs_zpk)."),
    ROUTINE("signal.cheb1ap", 3, "N, rp", "z, p, k", r_cheb1ap, NULL, "Analog Chebyshev type I lowpass prototype (scipy.signal.cheb1ap)."),
    ROUTINE("signal.cheb2ap", 3, "N, rs", "z, p, k", r_cheb2ap, NULL, "Analog Chebyshev type II lowpass prototype (scipy.signal.cheb2ap)."),
    ROUTINE("signal.gauss_spline", 1, "x, n", "out", r_gauss_spline, NULL, "Gaussian approximation to the B-spline basis of order n (scipy.signal.gauss_spline)."),
    ROUTINE("signal.vectorstrength", 2, "events, period", "strength, phase", r_vectorstrength, NULL, "Vector strength and mean phase of events over a period (scipy.signal.vectorstrength)."),
    ROUTINE("signal.medfilt", 1, "volume, kernel_size=None", "out", r_medfilt, NULL, "Zero-padded N-D median filter (scipy.signal.medfilt)."),
    ROUTINE("signal.medfilt2d", 1, "input, kernel_size=3", "out", r_medfilt2d, NULL, "Zero-padded 2-D median filter (scipy.signal.medfilt2d)."),
    ROUTINE("signal.order_filter", 1, "a, domain, rank", "out", r_order_filter_sig, NULL, "Rank-order filter over a footprint (scipy.signal.order_filter)."),
    ROUTINE("signal.argrelmax", 1, "data, axis=0, order=1, mode='clip'", "out", r_argrel, &SIG_ARGREL_MAX, "Indices of relative maxima of a 1-D array (scipy.signal.argrelmax)."),
    ROUTINE("signal.argrelmin", 1, "data, axis=0, order=1, mode='clip'", "out", r_argrel, &SIG_ARGREL_MIN, "Indices of relative minima of a 1-D array (scipy.signal.argrelmin)."),
    ROUTINE("signal.convolve2d", 1, "in1, in2, mode='full', boundary='fill', fillvalue=0", "out", r_convolve2d, NULL, "2-D convolution, zero-fill boundary (scipy.signal.convolve2d)."),
    ROUTINE("signal.correlate2d", 1, "in1, in2, mode='full', boundary='fill', fillvalue=0", "out", r_correlate2d, NULL, "2-D cross-correlation, zero-fill boundary (scipy.signal.correlate2d)."),
};

const fn_table TSR_SCIPY_SIGNAL_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
