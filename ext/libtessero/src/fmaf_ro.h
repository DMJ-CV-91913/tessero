/* fmaf_ro.h -- a correctly rounded fmaf for C runtimes whose fmaf rounds twice.
 *
 * MinGW-w64's fmaf (used when the target has no FMA instruction, i.e. the default x86-64 build) evaluates the
 * result in a wider type and then rounds that to float: two roundings. It passes the selfcheck's exact-product
 * probe, but on inputs whose exact result lies near a float halfway point it can differ from the single-rounded
 * value, and the canonical KAT then differs (Windows CI: "KAT reference == frozen 433b888214b2d50d").
 *
 * tsr_fmaf_ro is round-to-odd (Boldo and Melquiond): the product of two floats is exact in double (24 + 24 bits),
 * TwoSum gives the exact error of the double sum, and the sum is made odd in its last bit whenever it was
 * inexact. Rounding that round-to-odd double (53 bits >= 24 + 2) to float is then exactly the correctly rounded
 * a*b + c, with IEEE signed zeros, infinities and NaNs. It is tested against the C library's fmaf on every
 * platform whose fmaf is correct (csrc/tests/test_main.c), and used in place of fmaf only on MinGW without FMA.
 *
 * Library-internal: the public csrc/include/tessero_repro_f32.h stays byte-identical (Talario compiles a copy);
 * the library's own translation units, including repro_f32.c, see fmaf mapped here (platform-debt PD-16).
 */
#ifndef TSR_FMAF_RO_H
#define TSR_FMAF_RO_H

#include <math.h>
#include <stdint.h>
#include <string.h>

static inline float tsr_fmaf_ro(float a, float b, float c)
{
    const double p = (double)a * (double)b;                 /* exact */
    double s = p + (double)c;
    if (!isfinite(s)) return (float)s;                       /* an infinite or NaN operand: IEEE as is */
    const double bb = s - p, err = (p - (s - bb)) + ((double)c - bb);
    if (err != 0) {                                          /* inexact (then s != 0): round to odd */
        uint64_t u;
        memcpy(&u, &s, sizeof u);
        if ((u & 1u) == 0) {
            u = ((err > 0) == (s > 0)) ? u + 1 : u - 1;      /* one step toward the exact value */
            memcpy(&s, &u, sizeof s);
        }
    }
    return (float)s;
}

#if defined(__MINGW32__) && !defined(__FMA__)
#  define fmaf(a, b, c) tsr_fmaf_ro((a), (b), (c))
#  define TSR_FMAF_IS_RO 1
#  define TSR_FMAF_FN tsr_fmaf_ro      /* a function designator: _Generic does not expand fmaf(...) */
#else
#  define TSR_FMAF_IS_RO 0
#  define TSR_FMAF_FN fmaf
#endif

#endif
