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
