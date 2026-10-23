/*
 * csrc/include/tessero_repro_f32.h
 *
 * Reproducible f32 primitives: the SINGLE definition. Header-only (static
 * inline), valid C11 and C++20, so every consumer compiles the identical code:
 *
 *   - csrc/src/repro_f32.c        wraps these as the exported tsr_repro_*_f32 ABI
 *   - Talario (CPU twins)         compiles a byte-identical copy of THIS file
 *                                 (talario/src/tessero_repro_f32.h); Talario's
 *                                 build refuses to proceed if the two differ
 *   - Talario GPU shaders         shaders/repro_f32.glsl mirrors these functions
 *                                 in the same order; its constant block must be
 *                                 byte-identical to the one below (also checked)
 *
 * Goal: one DEFINED f32 bit-result per input, identical on every SIMD width,
 * thread count, compiler, CPU architecture and conformant GPU, BY CONSTRUCTION.
 * Only the portably correctly-rounded op set is used: + - * and fmaf (one
 * rounding), plus exact integer bit construction. No libm transcendental, no
 * divide, no sqrt, no hardware estimate, no f64 or x87 intermediate.
 *
 * Device-dependent float behaviour is removed by ONE explicit helper,
 * tsr__canon_f32: a subnormal becomes a signed zero and any NaN becomes the
 * canonical quiet NaN 0x7FC00000, using integer bit operations only. A device
 * that flushes subnormals and one that preserves them therefore produce the
 * same bits wherever canon is applied (every primitive input, and every output
 * that can land in the subnormal range). Domains are guarded so that no
 * intermediate overflows or goes subnormal: recip and rsqrt rescale large
 * arguments by an exact power of two (bit-identical to the unscaled path where
 * both are normal), tanh saturates where the correctly rounded f32 tanh is +-1.
 *
 * Every in-domain result is bit-identical to the pre-header implementation, so
 * the published goldens (csrc/repro/GOLDENS.md) are unchanged.
 *
 * COMPOSITES (end of file): sigmoid, tanh-GELU, and the row-wise softmax /
 * rmsnorm / layernorm, built only from the primitives above in ONE fixed
 * operation order. Talario's CPU backend calls these directly and its GPU
 * shaders mirror them statement for statement, so the composition itself, not
 * just the primitives, has a single definition with its own goldens.
 *
 * Build: consumers MUST compile with -ffp-contract=off -fno-fast-math, so the
 * compiler cannot fuse a separate mul+add into an fma or reassociate.
 */
#ifndef TESSERO_REPRO_F32_H
#define TESSERO_REPRO_F32_H

#include <math.h>
#include <stdint.h>
#include <string.h>

/* BEGIN TSR_REPRO_CONSTANTS (byte-identical in talario shaders/repro_f32.glsl) */
#define TSR_CANON_NAN     0x7FC00000u
#define TSR_POS_INF       0x7F800000u
#define TSR_POW2_M32      0x2F800000u
#define TSR_POW2_M64      0x1F800000u
#define TSR_BIG_BITS      0x5F800000u
#define TSR_RECIP_SEED    0x7EF127EAu
#define TSR_RSQRT_SEED    0x5f3759dfu
#define TSR_EXP2_OVF      128.0f
#define TSR_EXP2_UNF      -150.0f
#define TSR_EXP2_C0       1.535336188319500E-4f
#define TSR_EXP2_C1       1.339887440266574E-3f
#define TSR_EXP2_C2       9.618437357674640E-3f
#define TSR_EXP2_C3       5.550332471162809E-2f
#define TSR_EXP2_C4       2.402264791363012E-1f
#define TSR_EXP2_C5       6.931472028550421E-1f
#define TSR_EXP_OVF       88.72283905f
#define TSR_EXP_UNF       -103.972084f
#define TSR_LOG2E         1.44269504088896341f
#define TSR_LN2_HI        6.93359375E-1f
#define TSR_LN2_LO        -2.12194440E-4f
#define TSR_EXP_Q0        1.9875691500E-4f
#define TSR_EXP_Q1        1.3981999507E-3f
#define TSR_EXP_Q2        8.3334519073E-3f
#define TSR_EXP_Q3        4.1665795894E-2f
#define TSR_EXP_Q4        1.6666665459E-1f
#define TSR_EXP_Q5        5.0000001201E-1f
#define TSR_ERF_SMALL     0.5f
#define TSR_ERF_SAT       4.0f
#define TSR_ERF_B5        -8.5483270689E-4f
#define TSR_ERF_B4        5.2239776255E-3f
#define TSR_ERF_B3        -2.6866170645E-2f
#define TSR_ERF_B2        1.1283791671E-1f
#define TSR_ERF_B1        -3.7612638903E-1f
#define TSR_ERF_B0        1.1283791671f
#define TSR_ERF_P         3.275911E-1f
#define TSR_ERF_A1        2.54829592E-1f
#define TSR_ERF_A2        -2.84496736E-1f
#define TSR_ERF_A3        1.421413741f
#define TSR_ERF_A4        -1.453152027f
#define TSR_ERF_A5        1.061405429f
#define TSR_TANH_TINY     2.44140625E-4f
#define TSR_TANH_POLY     0.25f
#define TSR_TANH_SAT      9.1f
#define TSR_TANH_C3       -3.3333333333E-1f
#define TSR_TANH_C5       1.3333333333E-1f
#define TSR_TANH_C7       -5.3968253968E-2f
#define TSR_TANH_C9       2.1869488536E-2f
#define TSR_TANH_C11      -8.8632355299E-3f
#define TSR_GELU_C        0.7978845608028654f
#define TSR_GELU_A        0.044715f
/* END TSR_REPRO_CONSTANTS */

/* ---- exact bit reinterpretation (no aliasing hazard) ---- */
static inline float tsr__f32_from_bits(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

static inline uint32_t tsr__bits_from_f32(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    return b;
}

/* NaN test on the bit pattern (cannot be folded away by any float mode). */
static inline int tsr__isnan_bits(uint32_t b) {
    return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}

/*
 * The canonicalization helper. Subnormal -> signed zero; NaN -> canonical
 * quiet NaN; every other value (zeros, normals, infinities) unchanged. Pure
 * integer arithmetic, so it reads the same on a flush-to-zero device and a
 * subnormal-preserving one. Identity on every normal value, so it never moves
 * an in-domain result.
 */
static inline float tsr__canon_f32(float v) {
    uint32_t b = tsr__bits_from_f32(v);
    if (tsr__isnan_bits(b)) return tsr__f32_from_bits(TSR_CANON_NAN);
    if ((b & 0x7F800000u) == 0u) b &= 0x80000000u;      /* subnormal -> +-0 */
    return tsr__f32_from_bits(b);
}

/*
 * y * 2^n, for n in [-150, 128]. n is split into two near-equal halves so each
 * factor is a NORMAL power of two; the first multiply is exact for every caller
 * (y in [0.5, 2)), so the second is one correctly-rounded step. Left-associated.
 */
static inline float tsr__scale2_f32(float y, int n) {
    int n1 = n / 2;
    int n2 = n - n1;
    float f1 = tsr__f32_from_bits((uint32_t)(n1 + 127) << 23);
    float f2 = tsr__f32_from_bits((uint32_t)(n2 + 127) << 23);
    return (y * f1) * f2;
}

/*
 * Raw reciprocal core, 1/b for b in [2^-126, ~2^125.9] (the range where the
 * integer seed is a normal float). Integer bit-hack seed + five Newton steps in
 * fmaf; reaches the same ~1 ULP value on every CPU and GPU. Callers inside this
 * header only pass in-range arguments; external callers use tsr__recip_f32.
 */
static inline float tsr__rrecip_f32(float b) {
    uint32_t seed = TSR_RECIP_SEED - tsr__bits_from_f32(b);   /* ~1/b */
    float t = tsr__f32_from_bits(seed);
    t = t * fmaf(-b, t, 2.0f);           /* t <- t*(2 - b*t), quadratic decay */
    t = t * fmaf(-b, t, 2.0f);
    t = t * fmaf(-b, t, 2.0f);
    t = t * fmaf(-b, t, 2.0f);
    t = t * fmaf(-b, t, 2.0f);
    return t;
}

/*
 * Guarded reciprocal, any f32 input. NaN -> canonical NaN; +-0 (and subnormal,
 * after canon) -> +-Inf; +-Inf -> +-0; |b| >= 2^64 computed as
 * rrecip(|b| * 2^-64) * 2^-64, which is bit-identical to the raw core where
 * both are normal and keeps every intermediate normal; a result in the
 * subnormal range is canonicalized to a signed zero.
 */
static inline float tsr__recip_f32(float b) {
    b = tsr__canon_f32(b);
    uint32_t bb = tsr__bits_from_f32(b);
    if (tsr__isnan_bits(bb)) return b;
    uint32_t sign = bb & 0x80000000u;
    uint32_t ab = bb & 0x7FFFFFFFu;
    if (ab == 0u)          return tsr__f32_from_bits(sign | TSR_POS_INF);
    if (ab == TSR_POS_INF) return tsr__f32_from_bits(sign);
    float a = tsr__f32_from_bits(ab);
    float r;
    if (ab >= TSR_BIG_BITS) {
        float s = tsr__f32_from_bits(TSR_POW2_M64);
        r = tsr__rrecip_f32(a * s) * s;
    } else {
        r = tsr__rrecip_f32(a);
    }
    r = tsr__canon_f32(r);
    return tsr__f32_from_bits(tsr__bits_from_f32(r) | sign);
}

/*
 * rsqrt core, 1/sqrt(x) for x in [2^-126, 2^64): integer seed + five Newton
 * steps r <- r*(1.5 - 0.5*x*r^2) in fmaf.
 */
static inline float tsr__rsqrt_core_f32(float x) {
    uint32_t seed = TSR_RSQRT_SEED - (tsr__bits_from_f32(x) >> 1);  /* ~1/sqrt(x) */
    float r = tsr__f32_from_bits(seed);
    float h = 0.5f * x;                   /* exact (power-of-two multiply) */
    r = r * fmaf(-h, r * r, 1.5f);
    r = r * fmaf(-h, r * r, 1.5f);
    r = r * fmaf(-h, r * r, 1.5f);
    r = r * fmaf(-h, r * r, 1.5f);
    r = r * fmaf(-h, r * r, 1.5f);
    return r;
}

/*
 * Guarded rsqrt, any f32 input. NaN -> canonical NaN; x < 0 -> canonical NaN;
 * +-0 (and subnormal) -> +-Inf; +Inf -> +0; x >= 2^64 computed as
 * core(x * 2^-64) * 2^-32 (bit-identical to the core where both are normal; it
 * keeps r*r out of the subnormal range, which would otherwise make the result
 * depend on the device's flush mode). Unchanged on the golden vector.
 */
static inline float tsr__rrsqrt_f32(float x) {
    x = tsr__canon_f32(x);
    uint32_t xb = tsr__bits_from_f32(x);
    if (tsr__isnan_bits(xb)) return x;
    if ((xb & 0x7FFFFFFFu) == 0u) return tsr__f32_from_bits((xb & 0x80000000u) | TSR_POS_INF);
    if (xb & 0x80000000u)         return tsr__f32_from_bits(TSR_CANON_NAN);
    if (xb == TSR_POS_INF)        return 0.0f;
    if (xb >= TSR_BIG_BITS)
        return tsr__rsqrt_core_f32(x * tsr__f32_from_bits(TSR_POW2_M64))
               * tsr__f32_from_bits(TSR_POW2_M32);
    return tsr__rsqrt_core_f32(x);
}

/*
 * exp2(x) = 2^x. k = floorf(x + 0.5) (ties toward +Inf), r = x - k in
 * [-0.5, 0.5], 2^r = 1 + r*P(r) (Cephes exp2f degree-5, Horner in fmaf), then
 * scale2. Input and output canonicalized.
 */
static inline float tsr__exp2_f32(float x) {
    x = tsr__canon_f32(x);
    if (tsr__isnan_bits(tsr__bits_from_f32(x))) return x;
    if (x >= TSR_EXP2_OVF) return tsr__f32_from_bits(TSR_POS_INF);
    if (x <= TSR_EXP2_UNF) return 0.0f;

    float k = floorf(x + 0.5f);
    float r = x - k;
    int   n = (int)k;

    float p = fmaf(TSR_EXP2_C0, r, TSR_EXP2_C1);
    p = fmaf(p, r, TSR_EXP2_C2);
    p = fmaf(p, r, TSR_EXP2_C3);
    p = fmaf(p, r, TSR_EXP2_C4);
    p = fmaf(p, r, TSR_EXP2_C5);
    float y = fmaf(p, r, 1.0f);             /* 2^r */

    return tsr__canon_f32(tsr__scale2_f32(y, n));
}

/*
 * exp(x) = e^x. k = round(x*log2e); r = x - k*ln2 with ln2 split in two f32
 * parts (reduction in fmaf); e^r = 1 + r + r^2*P(r) (Cephes expf degree-5);
 * scale2. Input and output canonicalized.
 */
static inline float tsr__exp_f32(float x) {
    x = tsr__canon_f32(x);
    if (tsr__isnan_bits(tsr__bits_from_f32(x))) return x;
    if (x >= TSR_EXP_OVF) return tsr__f32_from_bits(TSR_POS_INF);
    if (x <= TSR_EXP_UNF) return 0.0f;

    float k = floorf(fmaf(x, TSR_LOG2E, 0.5f));  /* round(x*log2e) */
    float r = fmaf(-k, TSR_LN2_HI, x);           /* x - k*ln2_hi */
    r = fmaf(-k, TSR_LN2_LO, r);                 /* - k*ln2_lo */
    int   n = (int)k;

    float z = r * r;
    float p = fmaf(TSR_EXP_Q0, r, TSR_EXP_Q1);
    p = fmaf(p, r, TSR_EXP_Q2);
    p = fmaf(p, r, TSR_EXP_Q3);
    p = fmaf(p, r, TSR_EXP_Q4);
    p = fmaf(p, r, TSR_EXP_Q5);
    float er = fmaf(p, z, r);                    /* r^2*P(r) + r */
    er = er + 1.0f;                              /* e^r */

    return tsr__canon_f32(tsr__scale2_f32(er, n));
}

/*
 * erf(x). |x| <= 0.5: Maclaurin series x*(B0 + B1 w + ... + B5 w^5), w = x^2
 * (avoids the 1 - P*exp cancellation near 0). |x| > 0.5: Abramowitz & Stegun
 * 7.1.26, 1 - P(t)*e^(-x^2), t = 1/(1 + p|x|) via the raw reciprocal (argument
 * always in [1.16, 2.32]). Saturates to +-1 for |x| >= 4. Input canonicalized;
 * every output is a normal, a signed zero or +-1.
 */
static inline float tsr__erf_f32(float x) {
    x = tsr__canon_f32(x);
    if (tsr__isnan_bits(tsr__bits_from_f32(x))) return x;
    float ax = fabsf(x);

    if (ax <= TSR_ERF_SMALL) {
        float w = x * x;
        float p = fmaf(TSR_ERF_B5, w, TSR_ERF_B4);
        p = fmaf(p, w, TSR_ERF_B3);
        p = fmaf(p, w, TSR_ERF_B2);
        p = fmaf(p, w, TSR_ERF_B1);
        p = fmaf(p, w, TSR_ERF_B0);
        return tsr__canon_f32(x * p);             /* sign preserved, erf(-0) = -0 */
    }

    float s = copysignf(1.0f, x);
    if (ax >= TSR_ERF_SAT) return s;              /* +-Inf -> +-1 */

    float t = tsr__rrecip_f32(fmaf(TSR_ERF_P, ax, 1.0f));
    float poly = fmaf(TSR_ERF_A5, t, TSR_ERF_A4);
    poly = fmaf(poly, t, TSR_ERF_A3);
    poly = fmaf(poly, t, TSR_ERF_A2);
    poly = fmaf(poly, t, TSR_ERF_A1);
    poly = poly * t;

    float e = tsr__exp_f32(-(ax * ax));
    float y = fmaf(-poly, e, 1.0f);
    return s * y;
}

/*
 * tanh(u). Three frozen forms by magnitude, each a deterministic function of u:
 *   |u| <  2^-12 : u (the correctly rounded f32 tanh; no subnormal intermediate)
 *   |u| <= 0.25  : odd Taylor series u + u^3*(C3 + C5 w + ... + C11 w^4), w = u^2,
 *                  Horner in fmaf (truncation < 3e-10 relative)
 *   |u| <  9.1   : (e^2a - 1) * recip(e^2a + 1), a = |u|, sign restored by bit
 *                  (exactly odd); reproducible exp + raw reciprocal (argument
 *                  in [2.6, 8.0e7], always in range)
 *   |u| >= 9.1   : +-1 (the correctly rounded f32 tanh is exactly +-1 there;
 *                  without this e^2u overflows at u >= 44.36)
 * Input canonicalized; NaN -> canonical NaN.
 */
static inline float tsr__tanh_f32(float u) {
    u = tsr__canon_f32(u);
    if (tsr__isnan_bits(tsr__bits_from_f32(u))) return u;
    float au = fabsf(u);
    if (au < TSR_TANH_TINY) return u;
    if (au >= TSR_TANH_SAT) return copysignf(1.0f, u);
    if (au <= TSR_TANH_POLY) {
        float w = u * u;
        float p = fmaf(TSR_TANH_C11, w, TSR_TANH_C9);
        p = fmaf(p, w, TSR_TANH_C7);
        p = fmaf(p, w, TSR_TANH_C5);
        p = fmaf(p, w, TSR_TANH_C3);
        return fmaf(u * w, p, u);                 /* u + u^3*P(w) */
    }
    float e2 = tsr__exp_f32(2.0f * au);
    float t  = (e2 - 1.0f) * tsr__rrecip_f32(e2 + 1.0f);   /* tanh(|u|) > 0 */
    return tsr__f32_from_bits(tsr__bits_from_f32(t) | (tsr__bits_from_f32(u) & 0x80000000u));
}

/* ==========================================================================
 * Composites. Loads and stores canonicalized; sums are plain sequential f32
 * adds in index order (consumers must compile with -ffp-contract=off).
 * ======================================================================== */

/* sigmoid(x) = recip(1 + exp(-x)) */
static inline float tsr__sigmoid_f32(float x) {
    return tsr__canon_f32(tsr__recip_f32(1.0f + tsr__exp_f32(-tsr__canon_f32(x))));
}

/* tanh-approximation GELU: 0.5*x*(1 + tanh(c*(x + a*x^3))), c = sqrt(2/pi), a = 0.044715 */
static inline float tsr__gelu_tanh_f32(float x) {
    const float xv    = tsr__canon_f32(x);
    const float inner = TSR_GELU_C * (xv + TSR_GELU_A * xv * xv * xv);
    const float tnh   = tsr__tanh_f32(inner);
    const float h     = tsr__canon_f32(0.5f * xv);
    return tsr__canon_f32(h * (1.0f + tnh));
}

/* softmax over one row of n: ordered max, reproducible exp, guarded reciprocal */
static inline void tsr__softmax_row_f32(const float *x, float *y, int64_t n) {
    float mx = tsr__canon_f32(x[0]);
    for (int64_t j = 1; j < n; ++j) { const float v = tsr__canon_f32(x[j]); mx = (v > mx) ? v : mx; }
    float sum = 0.0f;
    for (int64_t j = 0; j < n; ++j) sum += tsr__exp_f32(tsr__canon_f32(x[j]) - mx);
    const float inv = tsr__recip_f32(sum);
    for (int64_t j = 0; j < n; ++j) y[j] = tsr__canon_f32(tsr__exp_f32(tsr__canon_f32(x[j]) - mx) * inv);
}

/* rmsnorm over one row of n: y = gamma * x * rsqrt(mean(x^2) + eps) */
static inline void tsr__rmsnorm_row_f32(const float *x, const float *gamma, float eps, float *y, int64_t n) {
    const float invn = tsr__recip_f32((float)n);
    float ms = 0.0f;
    for (int64_t j = 0; j < n; ++j) { const float v = tsr__canon_f32(x[j]); ms += v * v; }
    ms = ms * invn;
    const float rrms = tsr__rrsqrt_f32(ms + eps);
    for (int64_t j = 0; j < n; ++j) y[j] = tsr__canon_f32(tsr__canon_f32(gamma[j]) * tsr__canon_f32(x[j]) * rrms);
}

/* layernorm over one row of n: y = gamma * ((x - mu) * rsqrt(var + eps)) + beta */
static inline void tsr__layernorm_row_f32(const float *x, const float *gamma, const float *beta, float eps,
                                          float *y, int64_t n) {
    const float invn = tsr__recip_f32((float)n);
    float mu = 0.0f;
    for (int64_t j = 0; j < n; ++j) mu += tsr__canon_f32(x[j]);
    mu = mu * invn;
    float var = 0.0f;
    for (int64_t j = 0; j < n; ++j) { const float d = tsr__canon_f32(x[j]) - mu; var += d * d; }
    var = var * invn;
    const float rstd = tsr__rrsqrt_f32(var + eps);
    for (int64_t j = 0; j < n; ++j)
        y[j] = tsr__canon_f32(tsr__canon_f32(gamma[j]) * ((tsr__canon_f32(x[j]) - mu) * rstd) + tsr__canon_f32(beta[j]));
}

#endif /* TESSERO_REPRO_F32_H */
