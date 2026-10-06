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
        const int nn = (int)rd_real(&a[0].a, o0), mm = (int)rd_real(&a[1].a, o1);
        const cplx y = xsf::sph_harm_y(nn, mm, rd_real(&a[2].a, o2), rd_real(&a[3].a, o3));
        out[2 * q] = y.real();
        out[2 * q + 1] = y.imag();
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
    if (nargs > 1 && args[1].kind == 1) k = (long)((args[1].flags & 1) ? (double)args[1].ival : args[1].num);
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
    ROUTINE("special.lambertw", 1, "z, k=0, tol=1e-8", "out", tsr_scplx_lambertw, NULL, "Lambert W function, branch k (scipy.special.lambertw)."),
    ROUTINE("special.sph_harm_y", 1, "n, m, theta, phi", "out", tsr_scplx_sph_harm_y, NULL, "Spherical harmonic Y_n^m (scipy.special.sph_harm_y)."),
    ROUTINE("special.kelvin", 4, "x", "Be, Ke, Bep, Kep", tsr_scplx_kelvin, NULL, "Kelvin functions as complex pairs (scipy.special.kelvin)."),
};

extern "C" const fn_table TSR_SPECIAL_COMPLEX_TABLE = {TSR_SCPLX_DEFS, (int)(sizeof TSR_SCPLX_DEFS / sizeof TSR_SCPLX_DEFS[0])};
