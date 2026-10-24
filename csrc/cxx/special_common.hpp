/*
 * Common includes for the C++ part of libtessero: the special-function libraries SciPy 1.17.1 itself uses,
 * so Tessero's scipy.special and scipy.stats results come from the same code (ADR 0011):
 *
 *   third_party/xsf       SciPy's special-function library (BSD-3-Clause), pinned to SciPy 1.17.1's commit
 *   third_party/boost     the Boost.Math headers SciPy uses (Boost Software License 1.0), standalone mode
 *   third_party/scipy     SciPy's own glue (BSD-3-Clause): Boost wrappers, cdflib, Cython ports, xsf wrappers
 *
 * Nothing here throws across the C boundary: every exported function catches, and returns what SciPy returns
 * (without SciPy's Python warnings).
 */
#pragma once

#ifndef BOOST_MATH_STANDALONE
#  define BOOST_MATH_STANDALONE 1
#endif

#include "boost_error_format.hpp"   /* first: Boost's error messages formatted without iostreams */

#include <cmath>
#include <complex>
#include <limits>

#include "scipy/xsf_wrappers_real.hpp"
#include "xsf/iv_ratio.h"
#include "scipy/gen_harmonic.h"
extern "C" {
#include "scipy/_cosine.h"
}
#include "scipy/cython_ports.hpp"
#include "scipy/boost_special_functions.h"
#include "scipy/wright.hh"
#include "scipy/stirling2.h"

#define ELLINT_NO_VALIDATE_RELATIVE_ERROR_BOUND
#include "scipy/ellint_carlson_cpp_lite/ellint_carlson.hh"

/* the real-valued Carlson integrals of scipy/special/ellint_carlson_wrap.cxx */
namespace tsr_ellint {
static constexpr double ellip_rerr = 5e-16;
inline double fellint_RC(double x, double y) { double r; ellint_carlson::rc(x, y, ellip_rerr, r); return r; }
inline double fellint_RD(double x, double y, double z) { double r; ellint_carlson::rd(x, y, z, ellip_rerr, r); return r; }
inline double fellint_RF(double x, double y, double z) { double r; ellint_carlson::rf(x, y, z, ellip_rerr, r); return r; }
inline double fellint_RG(double x, double y, double z) { double r; ellint_carlson::rg(x, y, z, ellip_rerr, r); return r; }
inline double fellint_RJ(double x, double y, double z, double p) { double r; ellint_carlson::rj(x, y, z, p, ellip_rerr, r); return r; }
}

/* run a statement; any C++ exception becomes NaN in every output */
#define TSR_GUARD(NOUT, STMT)                                                             \
    try {                                                                                 \
        STMT                                                                              \
    } catch (...) {                                                                       \
        for (int k_ = 0; k_ < (NOUT); k_++) o[k_] = std::numeric_limits<double>::quiet_NaN(); \
    }


/* Complex loggamma for the FFTLog fast Hankel transforms (scipy_fft.c r_fht/r_ifht/r_fhtoffset). */
extern "C" void tsr_loggamma_cpx(const double *in, double *out)
{
    std::complex<double> z = xsf::loggamma(std::complex<double>(in[0], in[1]));
    out[0] = z.real();
    out[1] = z.imag();
}

/* Modified Fresnel integrals (scipy.special.modfresnelp/modfresnelm): real x -> two complex outputs
   F and K, written interleaved as [F.re, F.im, K.re, K.im] (np_numeric.c r_modfresnelp/m). */
#include "xsf/fresnel.h"
extern "C" void tsr_modfresnel_plus(double x, double *out)
{
    std::complex<double> F, K;
    xsf::modified_fresnel_plus(x, F, K);
    out[0] = F.real(); out[1] = F.imag(); out[2] = K.real(); out[3] = K.imag();
}
extern "C" void tsr_modfresnel_minus(double x, double *out)
{
    std::complex<double> F, K;
    xsf::modified_fresnel_minus(x, F, K);
    out[0] = F.real(); out[1] = F.imag(); out[2] = K.real(); out[3] = K.imag();
}

/* Mathieu Fourier coefficients (scipy.special.mathieu_even_coef/mathieu_odd_coef): the same xsf/specfun
   routines SciPy uses -- cva2 for the characteristic value, then fcoef for the expansion coefficients.
   kd selects ce/se and even/odd m (1,2 = cem even/odd; 4,3 = sem even/odd). fc must hold >= 251 doubles. */
#include "xsf/specfun/specfun.h"
extern "C" void tsr_mathieu_coef(int kd, int m, double q, double *fc)
{
    double a = xsf::specfun::cva2(kd, m, q);
    xsf::specfun::fcoef(kd, m, q, a, fc);
}

/* Complex zeros of erf (cerzo) and of the Fresnel integrals (fcszo: kf=1 C, kf=2 S); written interleaved
   [re, im, ...] into out (np_numeric.c r_erf_zeros / r_fresnel*_zeros). The zeros are found in nt-sized work
   storage on the heap (a fixed [260] stack array overflowed for nt > 260). Returns 0, or -1 when out of memory. */
#include <vector>
extern "C" int tsr_cerzo(int nt, double *out)
{
    try {
        std::vector<std::complex<double>> zo((size_t)nt);
        xsf::specfun::cerzo(nt, zo.data());
        for (int i = 0; i < nt; i++) { out[2 * i] = zo[i].real(); out[2 * i + 1] = zo[i].imag(); }
    } catch (...) { return -1; }
    return 0;
}
extern "C" int tsr_fcszo(int kf, int nt, double *out)
{
    try {
        std::vector<std::complex<double>> zo((size_t)nt);
        xsf::fcszo(kf, nt, zo.data());
        for (int i = 0; i < nt; i++) { out[2 * i] = zo[i].real(); out[2 * i + 1] = zo[i].imag(); }
    } catch (...) { return -1; }
    return 0;
}

/* Bernoulli / Euler numbers 0..n (scipy.special.bernoulli / euler): xsf's specfun bernob / eulerb on n1 = max(n, 2)
   entries, zero-initialised (the odd entries they leave unset are 0, as in SciPy), the first n + 1 copied out.
   Returns 0, or -1 when out of memory. */
extern "C" int tsr_bernob_eulerb(int euler, int n, double *out)
{
    const int n1 = n < 2 ? 2 : n;
    try {
        std::vector<double> w((size_t)n1 + 1, 0.0);
        if (euler) xsf::specfun::eulerb(n1, w.data()); else xsf::specfun::bernob(n1, w.data());
        for (int i = 0; i <= n; i++) out[i] = w[(size_t)i];
    } catch (...) { return -1; }
    return 0;
}

/* Parabolic cylinder sequences (scipy.special.pbdv_seq / pbvv_seq): xsf's specfun recurrence for order v1 in one
   call, as SciPy's _specfun.pbdv / pbvv wrapper returns it, and the first len entries of DV / VV and their
   derivatives (np_numeric.c pb_seq; v1 = n1 + v0 with |v0| < 1, n1 >= 1). The work arrays get xsf's own size,
   |int(v1)| + 2. A NaN x gives NaN, as xsf::pbdv / pbvv do. Returns 0, or -1 when out of memory. */
#include "xsf/par_cyl.h"
extern "C" int tsr_pb_seq(int vv, double v1, double x, int64_t len, double *dv, double *dp)
{
    if (std::isnan(x)) {
        for (int64_t k = 0; k < len; k++) dv[k] = dp[k] = std::numeric_limits<double>::quiet_NaN();
        return 0;
    }
    const int64_t num = (int64_t)std::abs((int)v1) + 2;
    if (len > num) return -1;
    try {
        std::vector<double> w((size_t)(2 * num), 0.0);
        double pf, pd;
        if (vv) xsf::detail::pbvv(x, v1, w.data(), w.data() + num, &pf, &pd);
        else xsf::detail::pbdv(x, v1, w.data(), w.data() + num, &pf, &pd);
        for (int64_t k = 0; k < len; k++) { dv[k] = w[(size_t)k]; dp[k] = w[(size_t)(num + k)]; }
    } catch (...) { return -1; }
    return 0;
}

/* Zeros of Jn and Jn' arranged by magnitude (scipy.special.jnjnp_zeros), via xsf's jdzo -- the same
   routine SciPy uses. jdzo fills n/m/p at [0..nt-1] and zo at [1..nt] (matching scipy's zo[1:nt+1]). */
extern "C" void tsr_jdzo(int nt, double *zoOut, int64_t *nOut, int64_t *mOut, int64_t *tOut)
{
    double zo[1202];
    int nn[1202], mm[1202], pp[1202];
    xsf::specfun::jdzo(nt, zo, nn, mm, pp);
    for (int i = 0; i < nt; i++) {
        zoOut[i] = zo[i + 1];
        nOut[i] = nn[i]; mOut[i] = mm[i]; tOut[i] = pp[i];
    }
}

/* Complex parabolic cylinder functions D_n(z) and D_n'(z) for orders 0..n (scipy.special.pbdn_seq),
   written interleaved into bOut / dOut. */
extern "C" int tsr_cpbdn(int n, double zre, double zim, double *bOut, double *dOut)
{
    /* n + 1 entries on the heap (a fixed [300] stack array overflowed for n >= 300); 0, or -1 out of memory */
    try {
        std::vector<std::complex<double>> cpb((size_t)n + 2), cpd((size_t)n + 2);
        xsf::specfun::cpbdn(n, std::complex<double>(zre, zim), cpb.data(), cpd.data());
        for (int i = 0; i <= n; i++) {
            bOut[2 * i] = cpb[i].real(); bOut[2 * i + 1] = cpb[i].imag();
            dOut[2 * i] = cpd[i].real(); dOut[2 * i + 1] = cpd[i].imag();
        }
    } catch (...) { return -1; }
    return 0;
}
