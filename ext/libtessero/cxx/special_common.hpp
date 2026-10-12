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
   [re, im, ...] into out (np_numeric.c r_erf_zeros / r_fresnel*_zeros). nt is small (fixtures <= ~16). */
extern "C" void tsr_cerzo(int nt, double *out)
{
    std::complex<double> zo[260];
    xsf::specfun::cerzo(nt, zo);
    for (int i = 0; i < nt; i++) { out[2 * i] = zo[i].real(); out[2 * i + 1] = zo[i].imag(); }
}
extern "C" void tsr_fcszo(int kf, int nt, double *out)
{
    std::complex<double> zo[260];
    xsf::fcszo(kf, nt, zo);
    for (int i = 0; i < nt; i++) { out[2 * i] = zo[i].real(); out[2 * i + 1] = zo[i].imag(); }
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
extern "C" void tsr_cpbdn(int n, double zre, double zim, double *bOut, double *dOut)
{
    std::complex<double> cpb[300], cpd[300];
    xsf::specfun::cpbdn(n, std::complex<double>(zre, zim), cpb, cpd);
    for (int i = 0; i <= n; i++) {
        bOut[2 * i] = cpb[i].real(); bOut[2 * i + 1] = cpb[i].imag();
        dOut[2 * i] = cpd[i].real(); dOut[2 * i + 1] = cpd[i].imag();
    }
}
