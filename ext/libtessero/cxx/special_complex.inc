/*
 * Complex-valued scipy.special functions (PAR-1.3): wofz, hankel1, hankel2, lambertw. They take and/or return
 * complex128, which the element-wise ufunc engine does not carry, so they are registry ROUTINES (the routine
 * path marshals complex128 arrays on both backends). Values come from SciPy's own xsf library (ADR 0011).
 *
 * This is #included into gen_special.cpp (which already includes special_common.hpp): the xsf/scipy code is not
 * multi-TU safe, so all of it lives in that one translation unit alongside gen_special_internal.inc.
 */
#include "../src/fn.h"
#include "xsf/erf.h"
#include "xsf/bessel.h"
#include "xsf/lambertw.h"
#include "xsf/kelvin.h"
#include "xsf/sph_harm.h"

#include <complex>
#include <cstdint>
#include <cstring>

namespace tsr_scplx {

using cplx = std::complex<double>;
constexpr int MAXD = 32;

/* one element of an array as a complex value (real dtypes -> imaginary part 0) */
inline void rd_cplx(const tsr_array *a, int64_t off, double *re, double *im)
{
    const char *p = (const char *)a->data + off;
    switch (a->dtype) {
    case TSR_C128: memcpy(re, p, 8); memcpy(im, p + 8, 8); break;
    case TSR_F64:  memcpy(re, p, 8); *im = 0.0; break;
    case TSR_F32:  { float f; memcpy(&f, p, 4); *re = f; *im = 0.0; break; }
    case TSR_I64:  { int64_t v; memcpy(&v, p, 8); *re = (double)v; *im = 0.0; break; }
    case TSR_I32:  { int32_t v; memcpy(&v, p, 4); *re = v; *im = 0.0; break; }
    case TSR_U8:
    case TSR_BOOL: *re = *(const uint8_t *)p; *im = 0.0; break;
    default:       *re = 0.0; *im = 0.0; break;
    }
}
inline double rd_real(const tsr_array *a, int64_t off) { double r, i; rd_cplx(a, off, &r, &i); return r; }

/* an argument as an array view: an NDArray as given, a real/bool scalar as a 0-d array (its value in buf) */
struct argarr { tsr_array a; double buf[2]; };
inline int as_arr(const tsr_arg *x, argarr *h)
{
    memset(h, 0, sizeof *h);
    if (x->kind == 3) { h->a = x->arr; return TSR_OK; }
    if (x->kind == 1) { h->buf[0] = (x->flags & 1) ? (double)x->ival : x->num; h->a.data = h->buf; h->a.dtype = TSR_F64; h->a.ndim = 0; return TSR_OK; }
    if (x->kind == 4) { h->buf[0] = x->num != 0; h->a.data = h->buf; h->a.dtype = TSR_BOOL; h->a.ndim = 0; return TSR_OK; }
    fn_set_error("argument must be an array or a number");
    return TSR_EARG;
}

/* broadcast k array views to a common shape; fills sh[*nd] and st[j] = broadcast strides of argument j */
inline int bcast_n(const argarr *in, int k, int32_t *nd, int64_t *sh, int64_t st[][MAXD])
{
    int32_t cnd = in[0].a.ndim;
    int64_t cs[MAXD];
    for (int d = 0; d < cnd; d++) cs[d] = in[0].a.shape[d];
    for (int j = 1; j < k; j++) {
        int32_t rnd;
        int64_t rs[MAXD];
        if (tsr_broadcast_shape(cnd, cs, in[j].a.ndim, in[j].a.shape, &rnd, rs) != 0) return TSR_ESHAPE;
        cnd = rnd;
        for (int d = 0; d < rnd; d++) cs[d] = rs[d];
    }
    *nd = cnd;
    for (int d = 0; d < cnd; d++) sh[d] = cs[d];
    for (int j = 0; j < k; j++)
        if (tsr_broadcast_strides(&in[j].a, cnd, sh, st[j]) != 0) return TSR_ESHAPE;
    return TSR_OK;
}

}  // namespace tsr_scplx

/* wofz(z): Faddeeva function, element-wise, complex128 -> complex128 */
extern "C" int tsr_scplx_wofz(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    using namespace tsr_scplx;
    (void)ctx; (void)nargs; (void)nres;
    argarr z;
    int rc = as_arr(&args[0], &z);
    if (rc < 0) return rc;
    double *out = (double *)fn_result_array(&res[0], TSR_C128, z.a.ndim, z.a.shape);
    if (!out) return TSR_ENOMEM;
    const int64_t n = tsr_shape_size(z.a.ndim, z.a.shape);
    int64_t ix[MAXD] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t off = z.a.offset;
        for (int d = 0; d < z.a.ndim; d++) off += ix[d] * z.a.strides[d];
        double re, im;
        rd_cplx(&z.a, off, &re, &im);
        const cplx w = xsf::wofz(cplx(re, im));
        out[2 * q] = w.real();
        out[2 * q + 1] = w.imag();
        for (int d = z.a.ndim - 1; d >= 0; d--) { if (++ix[d] < z.a.shape[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}

/* hankel1 / hankel2 (v, z): order v (real) broadcast with the complex argument z. ctx: 2 = second kind. */
extern "C" int tsr_scplx_hankel(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    using namespace tsr_scplx;
    (void)nargs; (void)nres;
    const int kind = ctx ? *(const int *)ctx : 1;        /* 1 hankel1, 2 hankel2, 3 hankel1e, 4 hankel2e */
    argarr v, z;
    int rc = as_arr(&args[0], &v);
    if (rc < 0) return rc;
    if ((rc = as_arr(&args[1], &z)) < 0) return rc;
    int32_t nd;
    int64_t sh[MAXD];
    if (tsr_broadcast_shape(v.a.ndim, v.a.shape, z.a.ndim, z.a.shape, &nd, sh) != 0) {
        fn_set_error("hankel: operands could not be broadcast together");
        return TSR_ESHAPE;
    }
    int64_t vst[MAXD], zst[MAXD];
    if (tsr_broadcast_strides(&v.a, nd, sh, vst) != 0 || tsr_broadcast_strides(&z.a, nd, sh, zst) != 0) {
        fn_set_error("hankel: operands could not be broadcast together");
        return TSR_ESHAPE;
    }
    double *out = (double *)fn_result_array(&res[0], TSR_C128, nd, sh);
    if (!out) return TSR_ENOMEM;
    const int64_t n = tsr_shape_size(nd, sh);
    int64_t ix[MAXD] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t vo = v.a.offset, zo = z.a.offset;
        for (int d = 0; d < nd; d++) { vo += ix[d] * vst[d]; zo += ix[d] * zst[d]; }
        double zre, zim;
        rd_cplx(&z.a, zo, &zre, &zim);
        const double vv = rd_real(&v.a, vo);
        const cplx zc(zre, zim);
        cplx h;
        switch (kind) {
        case 2: h = xsf::cyl_hankel_2(vv, zc); break;
        case 3: h = xsf::cyl_hankel_1e(vv, zc); break;
        case 4: h = xsf::cyl_hankel_2e(vv, zc); break;
        default: h = xsf::cyl_hankel_1(vv, zc); break;
        }
        out[2 * q] = h.real();
        out[2 * q + 1] = h.imag();
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}
static const int TSR_SCPLX_HANKEL2 = 2, TSR_SCPLX_HANKEL1E = 3, TSR_SCPLX_HANKEL2E = 4;

/* h1vp / h2vp (v, z, n=1): first derivative of the Hankel function, H_v'(z) = 0.5*(H_{v-1}(z) - H_{v+1}(z)).
   ctx: 2 = second kind. */
extern "C" int tsr_scplx_hvp(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    using namespace tsr_scplx;
    (void)nargs; (void)nres;
    const int kind = ctx ? *(const int *)ctx : 1;
    argarr v, z;
    int rc = as_arr(&args[0], &v);
    if (rc < 0) return rc;
    if ((rc = as_arr(&args[1], &z)) < 0) return rc;
    int32_t nd;
    int64_t sh[MAXD];
    if (tsr_broadcast_shape(v.a.ndim, v.a.shape, z.a.ndim, z.a.shape, &nd, sh) != 0) {
        fn_set_error("h1vp: operands could not be broadcast together");
        return TSR_ESHAPE;
    }
    int64_t vst[MAXD], zst[MAXD];
    if (tsr_broadcast_strides(&v.a, nd, sh, vst) != 0 || tsr_broadcast_strides(&z.a, nd, sh, zst) != 0) {
        fn_set_error("h1vp: operands could not be broadcast together");
        return TSR_ESHAPE;
    }
    double *out = (double *)fn_result_array(&res[0], TSR_C128, nd, sh);
    if (!out) return TSR_ENOMEM;
    const int64_t n = tsr_shape_size(nd, sh);
    int64_t ix[MAXD] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t vo = v.a.offset, zo = z.a.offset;
        for (int d = 0; d < nd; d++) { vo += ix[d] * vst[d]; zo += ix[d] * zst[d]; }
        double zre, zim;
        rd_cplx(&z.a, zo, &zre, &zim);
        const double vv = rd_real(&v.a, vo);
        const cplx zc(zre, zim);
        cplx h = (kind == 2)
            ? 0.5 * (xsf::cyl_hankel_2(vv - 1.0, zc) - xsf::cyl_hankel_2(vv + 1.0, zc))
            : 0.5 * (xsf::cyl_hankel_1(vv - 1.0, zc) - xsf::cyl_hankel_1(vv + 1.0, zc));
        out[2 * q] = h.real();
        out[2 * q + 1] = h.imag();
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}
static const int TSR_SCPLX_HVP2 = 2;

/* sph_harm_y(n, m, theta, phi): spherical harmonic, complex, 4-way broadcast (n, m integer) */
extern "C" int tsr_scplx_sph_harm_y(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    using namespace tsr_scplx;
    (void)ctx; (void)nargs; (void)nres;
    argarr a[4];
    for (int j = 0; j < 4; j++) { int rc = as_arr(&args[j], &a[j]); if (rc < 0) return rc; }
    int32_t nd;
    int64_t sh[MAXD], st[4][MAXD];
    if (bcast_n(a, 4, &nd, sh, st) != 0) { fn_set_error("sph_harm_y: operands could not be broadcast together"); return TSR_ESHAPE; }
    double *out = (double *)fn_result_array(&res[0], TSR_C128, nd, sh);
    if (!out) return TSR_ENOMEM;
    const int64_t n = tsr_shape_size(nd, sh);
    int64_t ix[MAXD] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t o0 = a[0].a.offset, o1 = a[1].a.offset, o2 = a[2].a.offset, o3 = a[3].a.offset;
        for (int d = 0; d < nd; d++) { o0 += ix[d] * st[0][d]; o1 += ix[d] * st[1][d]; o2 += ix[d] * st[2][d]; o3 += ix[d] * st[3][d]; }
        /* xsf takes int orders and its recurrence coefficients are int products: (2n+1)*(4(n-1)^2-1) in
           sph_legendre_p_recurrence_n overflows int once |n| > 645 (SciPy's own result there is undefined
           behaviour). An order that is NaN, an infinity or beyond +-645 gives NaN, never a float->int cast of an
           unchecked value or an int overflow. */
        const int64_t n64 = fn_dbl_i64(rd_real(&a[0].a, o0), INT64_MIN), m64 = fn_dbl_i64(rd_real(&a[1].a, o1), INT64_MIN);
        const int64_t lim = 645;
        if (n64 < -lim || n64 > lim || m64 < -lim || m64 > lim) {
            out[2 * q] = out[2 * q + 1] = std::numeric_limits<double>::quiet_NaN();
        } else {
            const cplx y = xsf::sph_harm_y((int)n64, (int)m64, rd_real(&a[2].a, o2), rd_real(&a[3].a, o3));
            out[2 * q] = y.real();
            out[2 * q + 1] = y.imag();
        }
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}

/* kelvin(x): (Be, Ke, Bep, Kep), each complex, from the real argument x (xsf::kelvin) */
extern "C" int tsr_scplx_kelvin(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    using namespace tsr_scplx;
    (void)ctx; (void)nargs; (void)nres;
    argarr x;
    int rc = as_arr(&args[0], &x);
    if (rc < 0) return rc;
    double *o[4];
    for (int j = 0; j < 4; j++) {
        o[j] = (double *)fn_result_array(&res[j], TSR_C128, x.a.ndim, x.a.shape);
        if (!o[j]) return TSR_ENOMEM;
    }
    const int64_t n = tsr_shape_size(x.a.ndim, x.a.shape);
    int64_t ix[MAXD] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t off = x.a.offset;
        for (int d = 0; d < x.a.ndim; d++) off += ix[d] * x.a.strides[d];
        cplx be, ke, bep, kep;
        xsf::kelvin(rd_real(&x.a, off), be, ke, bep, kep);
        const cplx vals[4] = {be, ke, bep, kep};
        for (int j = 0; j < 4; j++) { o[j][2 * q] = vals[j].real(); o[j][2 * q + 1] = vals[j].imag(); }
        for (int d = x.a.ndim - 1; d >= 0; d--) { if (++ix[d] < x.a.shape[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}

/* lambertw(z, k=0, tol=1e-8): branch k (int scalar), element-wise over z */
extern "C" int tsr_scplx_lambertw(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    using namespace tsr_scplx;
    (void)ctx; (void)nres;
    argarr z;
    int rc = as_arr(&args[0], &z);
    if (rc < 0) return rc;
    long k = 0;
    double tol = 1e-8;
    if (nargs > 1 && args[1].kind == 1) {          /* branch index: checked, never a cast of NaN/inf/huge */
        int64_t kk;
        const int krc = fn_arg_i64(&args[1], 0, -((int64_t)1 << 53), (int64_t)1 << 53, "k", &kk);
        if (krc < 0) return krc;
        k = (long)kk;
    }
    if (nargs > 2 && args[2].kind == 1) tol = (args[2].flags & 1) ? (double)args[2].ival : args[2].num;
    double *out = (double *)fn_result_array(&res[0], TSR_C128, z.a.ndim, z.a.shape);
    if (!out) return TSR_ENOMEM;
    const int64_t n = tsr_shape_size(z.a.ndim, z.a.shape);
    int64_t ix[MAXD] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t off = z.a.offset;
        for (int d = 0; d < z.a.ndim; d++) off += ix[d] * z.a.strides[d];
        double re, im;
        rd_cplx(&z.a, off, &re, &im);
        const cplx w = xsf::lambertw(cplx(re, im), k, tol);
        out[2 * q] = w.real();
        out[2 * q + 1] = w.imag();
        for (int d = z.a.ndim - 1; d >= 0; d--) { if (++ix[d] < z.a.shape[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}

static const fn_def TSR_SCPLX_DEFS[] = {
    ROUTINE("special.wofz", 1, "z", "out", tsr_scplx_wofz, NULL, "Faddeeva function w(z) = exp(-z^2) erfc(-iz) (scipy.special.wofz)."),
    ROUTINE("special.hankel1", 1, "v, z", "out", tsr_scplx_hankel, NULL, "Hankel function of the first kind (scipy.special.hankel1)."),
    ROUTINE("special.hankel2", 1, "v, z", "out", tsr_scplx_hankel, &TSR_SCPLX_HANKEL2, "Hankel function of the second kind (scipy.special.hankel2)."),
    ROUTINE("special.hankel1e", 1, "v, z", "out", tsr_scplx_hankel, &TSR_SCPLX_HANKEL1E, "Exponentially scaled Hankel function of the first kind (scipy.special.hankel1e)."),
    ROUTINE("special.hankel2e", 1, "v, z", "out", tsr_scplx_hankel, &TSR_SCPLX_HANKEL2E, "Exponentially scaled Hankel function of the second kind (scipy.special.hankel2e)."),
    ROUTINE("special.h1vp", 1, "v, z, n=1", "out", tsr_scplx_hvp, NULL, "First derivative of the Hankel function of the first kind (scipy.special.h1vp)."),
    ROUTINE("special.h2vp", 1, "v, z, n=1", "out", tsr_scplx_hvp, &TSR_SCPLX_HVP2, "First derivative of the Hankel function of the second kind (scipy.special.h2vp)."),
    ROUTINE("special.lambertw", 1, "z, k=0, tol=1e-8", "out", tsr_scplx_lambertw, NULL, "Lambert W function, branch k (scipy.special.lambertw)."),
    ROUTINE("special.sph_harm_y", 1, "n, m, theta, phi", "out", tsr_scplx_sph_harm_y, NULL, "Spherical harmonic Y_n^m (scipy.special.sph_harm_y)."),
    ROUTINE("special.kelvin", 4, "x", "Be, Ke, Bep, Kep", tsr_scplx_kelvin, NULL, "Kelvin functions as complex pairs (scipy.special.kelvin)."),
};

extern "C" const fn_table TSR_SPECIAL_COMPLEX_TABLE = {TSR_SCPLX_DEFS, (int)(sizeof TSR_SCPLX_DEFS / sizeof TSR_SCPLX_DEFS[0])};

/* The first nt zeros of a Kelvin function (kd 1..8: ber, bei, ker, kei, ber', bei', ker', kei') by SciPy's own
   specfun KLVNZO (xsf::klvnzo, SciPy 1.17.1, which ber_zeros .. kelvin_zeros call): each zero starts from the
   previous one plus 4.44 (the first from a table) and is refined by Newton steps on klvna's values until a step moves
   it by at most 5e-10. The same steps on the same numbers as xsf, with one guard xsf lacks: a zero that has not
   converged after 100 steps, or has turned NaN (past klvna's range), ends the search, so a large nt cannot loop
   forever. Returns how many zeros were found (nt when all converged). */
extern "C" int64_t tsr_kelvin_zeros_klvnzo(int64_t nt, int kd, double *zo)
{
    try {
        double ber, bei, ger, gei, der, dei, her, hei;
        double rt0[9] = {0.0, 2.84891, 5.02622, 1.71854, 3.91467, 6.03871, 3.77268, 2.66584, 4.93181};
        double rt = rt0[kd];
        for (int64_t m = 1; m <= nt; m++) {
            for (int it = 0;; it++) {
                xsf::detail::klvna(rt, &ber, &bei, &ger, &gei, &der, &dei, &her, &hei);
                if (kd == 1) rt -= ber / der;
                else if (kd == 2) rt -= bei / dei;
                else if (kd == 3) rt -= ger / her;
                else if (kd == 4) rt -= gei / hei;
                else if (kd == 5) rt -= der / (-bei - der / rt);
                else if (kd == 6) rt -= dei / (ber - dei / rt);
                else if (kd == 7) rt -= her / (-gei - her / rt);
                else rt -= hei / (ger - hei / rt);
                if (std::fabs(rt - rt0[kd]) <= 5e-10) break;
                rt0[kd] = rt;
                if (it >= 100 || !std::isfinite(rt)) return m - 1;
            }
            zo[m - 1] = rt;
            rt += 4.44;
        }
        return nt;
    } catch (...) {
        return 0;
    }
}

/* The first nt zeros of J_n, J_n', Y_n and Y_n' (n >= 0) by SciPy's own specfun JYZO (xsf::specfun::jyzo, SciPy
   1.17.1, which jn_zeros, jnp_zeros, yn_zeros, ynp_zeros and jnyn_zeros call): Newton steps on jyndd's values from
   asymptotic starts (with SciPy's GH-18859 corrections), each next zero started from the spacing of the last two.
   The same statements on the same numbers as xsf, with one guard xsf lacks: a zero that has not converged after 200
   steps ends the search (-1) instead of iterating forever. 0 when all 4 x nt zeros were found. */
extern "C" int tsr_bessel_zeros_jyzo(int n, int nt, double *rj0, double *rj1, double *ry0, double *ry1)
{
    try {
        using xsf::specfun::jyndd;
        const int maxit = 200;
        int L, it = 0;
        double b, h, x, x0, bjn, djn, fjn, byn, dyn, fyn;
        const double pi = 3.141592653589793;
        // -- Newton method for j_{N,L}
        if (n == 0) {
            x = 2.4;
        } else {
            x = n + 1.85576 * pow(n, 0.33333) + 1.03315 / pow(n, 0.33333);
        }
        L = 0;
    L10:
        x0 = x;
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        x -= bjn / djn;
        if (fabs(x - x0) > 1e-11) {
            if (++it > maxit) return -1;
            goto L10;
        }
        it = 0;
        L += 1;
        rj0[L - 1] = x;
        if (L == 1) {
            if (n == 0) {
                x = 5.52;
            } else {
                x = n + 3.24460 * pow(n, 0.33333) + 3.15824 / pow(n, 0.33333);
            }
        } else {
            x = rj0[L - 1] + (rj0[L - 1] - rj0[L - 2]);
        }
        if (L <= (n + 10)) {
            jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
            h = atan(fabs(djn) / sqrt(fabs(fjn * bjn)));
            b = -djn / (bjn * atan(h));
            x -= (h - pi / 2) / b;
        }
        if (L < nt) {
            goto L10;
        }
        // -- Newton method for j_{N,L+1}'
        if (n == 0) {
            x = 3.8317;
        } else {
            x = n + 0.80861 * pow(n, 0.33333) + 0.07249 / pow(n, 0.33333);
        }
        L = 0;
    L15:
        x0 = x;
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        x -= djn / fjn;
        if (fabs(x - x0) > 1e-11) {
            if (++it > maxit) return -1;
            goto L15;
        }
        it = 0;
        L += 1;
        rj1[L - 1] = x;
        if (L < nt) {
            x = rj1[L - 1] + (rj0[L] - rj0[L - 1]);
            goto L15;
        }
        // -- Newton method for y_{N,L}
        if (n == 0) {
            x = 0.89357697;
        } else {
            x = n + 0.93158 * pow(n, 0.33333) + 0.26035 / pow(n, 0.33333);
        }
        L = 0;
    L20:
        x0 = x;
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        x -= byn / dyn;
        if (fabs(x - x0) > 1.0e-11) {
            if (++it > maxit) return -1;
            goto L20;
        }
        it = 0;
        L += 1;
        ry0[L - 1] = x;
        if (L == 1) {
            if (n == 0) {
                x = 3.957678419314858;
            } else {
                x = n + 2.59626 * pow(n, 0.33333) + 2.022183 / pow(n, 0.33333);
            }
        } else {
            x = ry0[L - 1] + (ry0[L - 1] - ry0[L - 2]);
        }
        if (L <= n + 10) {
            jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
            h = atan(fabs(dyn) / sqrt(fabs(fyn * byn)));
            b = -dyn / (byn * tan(h));
            x -= (h - pi / 2) / b;
        }
        if (L < nt)
            goto L20;
        // -- Newton method for y_{N,L+1}'
        if (n == 0) {
            x = 2.67257;
        } else {
            x = n + 1.8211 * pow(n, 0.33333) + 0.94001 / pow(n, 0.33333);
        }
        L = 0;
    L25:
        x0 = x;
        jyndd(n, x, &bjn, &djn, &fjn, &byn, &dyn, &fyn);
        x -= dyn / fyn;
        if (fabs(x - x0) > 1.0e-11) {
            if (++it > maxit) return -1;
            goto L25;
        }
        it = 0;
        L += 1;
        ry1[L - 1] = x;
        if (L < nt) {
            x = ry1[L - 1] + (ry0[L] - ry0[L - 1]);
            goto L25;
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

/* The spheroidal characteristic values for modes m..n (n - m <= 199) by one call of SciPy's specfun SEGV, as
   scipy.special.pro_cv_seq (kd 1) and obl_cv_seq (kd -1) return segv(m, n, c, kd)[1][:n - m + 1]. 0, or -1. */
extern "C" int tsr_spheroidal_cv_seq(int m, int n, double c, int kd, double *out)
{
    try {
        double cv = 0.0, eg[200] = {0.0};
        if (xsf::specfun::segv(m, n, c, kd, &cv, eg) != xsf::specfun::Status::OK) return -1;
        for (int i = 0; i <= n - m; i++) out[i] = eg[i];
        return 0;
    } catch (...) {
        return -1;
    }
}

/* The first nt zeros of Y0 (kf 0), Y1 (kf 1) or Y1' (kf 2) on the real axis and the value Y1, Y1' (= Y0) or Y1
   there, by SciPy's own specfun CYZO (xsf::specfun::cyzo with kc = 1, as y0_zeros, y1_zeros and y1p_zeros call it):
   Newton steps in the complex plane, deflated by the zeros already found. Its iterations are bounded (50 per zero).
   zo and zv are nt complex values each, interleaved. 0, or -1 out of memory. */
extern "C" int tsr_y_zeros_cyzo(int nt, int kf, double *zo, double *zv)
{
    try {
        std::vector<std::complex<double>> z((size_t)nt), v((size_t)nt);
        xsf::specfun::cyzo(nt, kf, 1, z.data(), v.data());
        for (int i = 0; i < nt; i++) {
            zo[2 * i] = z[(size_t)i].real(); zo[2 * i + 1] = z[(size_t)i].imag();
            zv[2 * i] = v[(size_t)i].real(); zv[2 * i + 1] = v[(size_t)i].imag();
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

/* The first nt zeros of Ai and Ai' (kf 1) or Bi and Bi' (kf 2) with the function's value at the derivative's zeros
   and the derivative's at the function's, by SciPy's own specfun AIRYZO (xsf::airyzo, which ai_zeros and bi_zeros
   call): Newton steps from asymptotic starts. The same statements on the same numbers as xsf, with one guard: a zero
   that has not converged after 200 steps (a NaN error never does) ends the search (-1) instead of iterating forever.
   0 when all were found. */
extern "C" int tsr_airy_zeros_airyzo(int nt, int kf, double *xa, double *xb, double *xc, double *xd)
{
    try {
        const double pi = 3.141592653589793;
        const int maxit = 200;
        int i;
        double rt = 0.0, rt0, u = 0.0, u1 = 0.0, x, ai, bi, ad, bd, err;
        for (i = 1; i <= nt; ++i) {
            rt0 = 0.0;
            if (kf == 1) {
                u = 3.0 * pi * (4.0 * i - 1) / 8.0;
                u1 = 1 / (u * u);
            } else if (kf == 2) {
                if (i == 1) {
                    rt0 = -1.17371;
                } else {
                    u = 3.0 * pi * (4.0 * i - 3.0) / 8.0;
                    u1 = 1 / (u * u);
                }
            }
            if (rt0 == 0) {
                rt0 = -pow(u * u, 1.0 / 3.0) *
                      (1.0 +
                       u1 * (5.0 / 48.0 + u1 * (-5.0 / 36.0 + u1 * (77125.0 / 82944.0 + u1 * (-108056875.0 / 6967296.0)))));
            }
            for (int it = 0;; it++) {
                x = rt0;
                xsf::airyb(x, &ai, &bi, &ad, &bd);
                if (kf == 1) {
                    rt = rt0 - ai / ad;
                } else if (kf == 2) {
                    rt = rt0 - bi / bd;
                }
                err = fabs((rt - rt0) / rt);
                if (err <= 1.0e-12) {
                    break;
                } else {
                    rt0 = rt;
                }
                if (it >= maxit) return -1;
            }
            xa[i - 1] = rt;
            if (err > 1.0e-14) {
                xsf::airyb(rt, &ai, &bi, &ad, &bd);
            }
            if (kf == 1) {
                xd[i - 1] = ad;
            } else if (kf == 2) {
                xd[i - 1] = bd;
            }
        }
        for (i = 1; i <= nt; ++i) {
            rt0 = 0.0;
            if (kf == 1) {
                if (i == 1) {
                    rt0 = -1.01879;
                } else {
                    u = 3.0 * pi * (4.0 * i - 3.0) / 8.0;
                    u1 = 1 / (u * u);
                }
            } else if (kf == 2) {
                if (i == 1) {
                    rt0 = -2.29444;
                } else {
                    u = 3.0 * pi * (4.0 * i - 1.0) / 8.0;
                    u1 = 1 / (u * u);
                }
            }
            if (rt0 == 0) {
                rt0 = -pow(u * u, 1.0 / 3.0) *
                      (1.0 + u1 * (-7.0 / 48.0 +
                                   u1 * (35.0 / 288.0 + u1 * (-181223.0 / 207360.0 + u1 * (18683371.0 / 1244160.0)))));
            }
            for (int it = 0;; it++) {
                x = rt0;
                xsf::airyb(x, &ai, &bi, &ad, &bd);
                if (kf == 1) {
                    rt = rt0 - ad / (ai * x);
                } else if (kf == 2) {
                    rt = rt0 - bd / (bi * x);
                }
                err = fabs((rt - rt0) / rt);
                if (err <= 1.0e-12) {
                    break;
                } else {
                    rt0 = rt;
                }
                if (it >= maxit) return -1;
            }
            xb[i - 1] = rt;
            if (err > 1.0e-14) {
                xsf::airyb(rt, &ai, &bi, &ad, &bd);
            }
            if (kf == 1) {
                xc[i - 1] = ai;
            } else if (kf == 2) {
                xc[i - 1] = bi;
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}
