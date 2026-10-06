/*
 * Vectorisable exp and log.
 *
 * glibc's exp/log are accurate but scalar: a loop calling them runs one
 * element at a time. These versions are branch-free (special cases are
 * blended in at the end), use only arithmetic, comparisons and 64-bit integer
 * shifts/adds, and are static inline, so GCC and Clang vectorise a loop over
 * them (SSE2, AVX2 or AVX-512, per TSR_CLONES) without -ffast-math.
 *
 * Algorithms and polynomial coefficients are fdlibm's (e_exp.c, e_log.c;
 * Copyright (C) 1993 by Sun Microsystems, Inc.; "Permission to use, copy,
 * modify, and distribute this software is freely granted, provided that this
 * notice is preserved."). Accuracy: < 1 ulp for exp and log (float64), checked
 * against long double references in tests (csrc/tests/test_main.c) and by
 * tools/ulp_check.py over 10^7 points. float32 versions evaluate the float64
 * kernels and round once, so they are faithfully rounded.
 *
 * Special values follow C99 Annex F: exp(NaN) = NaN, exp(+inf) = +inf,
 * exp(-inf) = 0, overflow to +inf above 709.78..., gradual underflow to
 * subnormals and 0 below -708.39...; log(x < 0) = NaN, log(+-0) = -inf,
 * log(+inf) = +inf, log(NaN) = NaN, subnormal inputs handled exactly.
 * errno is never set (Tessero builds with -fno-math-errno).
 */
#ifndef TSR_VMATH_H
#define TSR_VMATH_H

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline uint64_t tsr_bits(double x) { uint64_t u; memcpy(&u, &x, 8); return u; }
static inline double tsr_from_bits(uint64_t u) { double x; memcpy(&x, &u, 8); return x; }

/* Branch-free select through bit masks: some GCC versions turn `c ? a : b` on doubles back into a branch. */
static inline uint64_t tsr_mask(int c) { return (uint64_t)0 - (uint64_t)(c != 0); }
static inline double tsr_blend(uint64_t mask, double a, double b)
{
    return tsr_from_bits((tsr_bits(a) & mask) | (tsr_bits(b) & ~mask));
}

#define TSR_SHIFTER 6755399441055744.0            /* 1.5 * 2^52: x + SHIFTER rounds x to an integer in the low bits */

static inline double tsr_exp(double x)
{
    const double LOG2E = 1.44269504088896338700e+00;
    const double LN2HI = 6.93147180369123816490e-01, LN2LO = 1.90821492927058770002e-10;
    const double P1 = 1.66666666666666019037e-01, P2 = -2.77777777770155933842e-03, P3 = 6.61375632143793436117e-05,
                 P4 = -1.65339022054652515390e-06, P5 = 4.13813679705723846039e-08;
    const double OVER = 7.09782712893383973096e+02, UNDER = -7.45133219101941108420e+02;

    /* clamp so the arithmetic below stays finite; the true special results are blended in at the end.
       Every value is computed unconditionally and only selected (no conditionally executed arithmetic),
       which is what lets the compiler if-convert and vectorise without -fno-trapping-math. */
    double xc = x > 709.9 ? 709.9 : x;
    xc = xc < -745.2 ? -745.2 : xc;
    xc = x == x ? xc : 0.0;
    const double nanres = x + x;

    const double kd = (xc * LOG2E + TSR_SHIFTER) - TSR_SHIFTER;     /* round to nearest integer */
    const double hi = xc - kd * LN2HI, lo = kd * LN2LO;
    const double r = hi - lo;
    const double t = r * r;
    const double c = r - t * (P1 + t * (P2 + t * (P3 + t * (P4 + t * P5))));
    const double y = 1.0 - ((lo - (r * c) / (2.0 - c)) - hi);

    /* y * 2^k in two steps so that k in [-1075, 1024] never leaves the exponent range */
    const double adj = kd > 1000.0 ? -600.0 : (kd < -1000.0 ? 600.0 : 0.0);
    const double s2 = kd > 1000.0 ? 0x1p600 : (kd < -1000.0 ? 0x1p-600 : 1.0);
    const uint64_t kb = tsr_bits(kd + adj + TSR_SHIFTER);          /* low bits hold k + adj (two's complement) */
    const double s1 = tsr_from_bits((kb + 1023u) << 52);
    double res = (y * s1) * s2;

    res = x > OVER ? INFINITY : res;
    res = x < UNDER ? 0.0 : res;
    return x == x ? res : nanres;
}

static inline double tsr_log(double x)
{
    const double LN2HI = 6.93147180369123816490e-01, LN2LO = 1.90821492927058770002e-10;
    const double Lg1 = 6.666666666666735130e-01, Lg2 = 3.999999999940941908e-01, Lg3 = 2.857142874366239149e-01,
                 Lg4 = 2.222219843214978396e-01, Lg5 = 1.818357216161805012e-01, Lg6 = 1.531383769920937332e-01,
                 Lg7 = 1.479819860511658591e-01;

    /* subnormals: scale by 2^54 first */
    const uint64_t sub = tsr_mask(x < 0x1p-1022);
    const double x54 = x * 0x1p54;
    const double xs = tsr_blend(sub, x54, x);
    const double nanres = x + x;
    const uint64_t u = tsr_bits(xs);
    const uint64_t e = (u >> 52) & 0x7ff;
    const uint64_t mant = u & 0x000fffffffffffffull;
    const uint64_t hx = mant >> 32;                                   /* top 20 mantissa bits, as fdlibm's hx */
    const uint64_t i = (mant + 0x00095f6400000000ull) & 0x0010000000000000ull;   /* set when the mantissa >= sqrt(2) */
    const double m = tsr_from_bits(mant | (i ^ 0x3ff0000000000000ull));         /* in [sqrt(2)/2, sqrt(2)) */
    /* k = e - 1023 + [i] - (sub ? 54 : 0), as an exact double without int64 -> double conversion */
    const double kd = (tsr_from_bits(0x4330000000000000ull | (e + (i >> 52))) - 4503599627370496.0) - tsr_blend(sub, 1077.0, 1023.0);

    const double f = m - 1.0;
    const double hfsq = 0.5 * f * f;
    const double s = f / (2.0 + f);
    const double z = s * s, w = z * z;
    const double t1 = w * (Lg2 + w * (Lg4 + w * Lg6));
    const double t2 = z * (Lg1 + w * (Lg3 + w * (Lg5 + w * Lg7)));
    const double R = t2 + t1;
    /* fdlibm picks one of two equivalent forms by where f lies; both are computed and blended */
    const uint64_t sel = tsr_mask((hx >= 0x6147au) & (hx <= 0x6b851u));   /* fdlibm: (hx - 0x6147a) | (0x6b851 - hx) > 0 */
    const double a = kd * LN2HI - ((hfsq - (s * (hfsq + R) + kd * LN2LO)) - f);
    const double b = kd * LN2HI - ((s * (f - R) - kd * LN2LO) - f);
    double res = tsr_blend(sel, a, b);

    res = tsr_blend(tsr_mask(x == 0.0), -INFINITY, res);
    res = tsr_blend(tsr_mask(x < 0.0), NAN, res);
    res = tsr_blend(tsr_mask(x == INFINITY), x, res);
    return tsr_blend(tsr_mask(x == x), res, nanres);
}

static inline float tsr_expf(float x) { return (float)tsr_exp((double)x); }
static inline float tsr_logf(float x) { return (float)tsr_log((double)x); }

#endif
