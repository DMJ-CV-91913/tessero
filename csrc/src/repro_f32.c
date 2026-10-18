/*
 * csrc/src/repro_f32.c
 *
 * Reproducible f32 transcendental primitives (see REPRO_F32_KICKOFF.md and
 * talario/docs/papers/fma-only-transcendentals-design.md).
 *
 * These are a NEW surface, distinct from the accuracy-first f64 ufuncs in
 * np_*.c / cxx/ (erf/gamma/... backed by boost/xsf). The goal here is not
 * maximum accuracy, it is a single DEFINED f32 bit-result that is identical on
 * every SIMD width, every thread count, every compiler, and later on the GPU,
 * BY CONSTRUCTION. That is achieved by computing each value from a fixed
 * polynomial evaluated only in correctly-rounded f32 ops: +, -, *, / , sqrtf
 * (all IEEE-correctly-rounded), and fmaf (one rounding). No libm expf/exp2f/
 * erff, no SVML, no reciprocal/rsqrt estimate, no powf, no f64 or x87
 * intermediates, no reassociation.
 *
 * The CPU result computed here is the canonical the Talario GPU shader mirrors
 * bit-for-bit. The exact constants (as hex floats) and the evaluation order are
 * the spec; they are recorded in csrc/repro/GOLDENS.md and printed by
 * csrc/repro/determinism_test.c (the `spec` mode).
 *
 * Build: this TU MUST be compiled with -ffp-contract=off -fno-fast-math so the
 * compiler cannot contract or reassociate the non-fma arithmetic. The csrc
 * Makefile has a dedicated rule for this object that adds those flags on top of
 * the inherited CFLAGS; do not remove it.
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

/* --------------------------------------------------------------------------
 * Shared helpers
 * ------------------------------------------------------------------------ */

/* Reinterpret a uint32 bit pattern as an f32 (no aliasing hazard). */
static inline float tsr__f32_from_bits(uint32_t b) {
    float f;
    memcpy(&f, &b, sizeof f);
    return f;
}

/* Reinterpret an f32 as its uint32 bit pattern (companion to the above). */
static inline uint32_t tsr__bits_from_f32(float f) {
    uint32_t b;
    memcpy(&b, &f, sizeof b);
    return b;
}

/*
 * Deterministic reciprocal 1/b for b > 0, and reciprocal-sqrt 1/sqrt(x) for
 * x > 0, built from ONLY correctly-rounded f32 ops (* and fmaf) plus an exact
 * integer bit-hack seed. No sqrtf and no `/`: IEEE sqrt and divide are
 * correctly rounded on a CPU, but Vulkan guarantees correct rounding ONLY for
 * + - * fma (OpSqrt/OpFDiv are 2.5-3 ULP and vendor-approximate), so a
 * sqrtf/divide definition cannot be reproduced bit-for-bit on the GPU. These
 * reach the same f32 bits on every CPU and GPU because the seed is pure integer
 * arithmetic (identical everywhere) and the Newton refinement uses only
 * correctly-rounded * and fma. Five iterations reach the f32 fixed point for the
 * primitive input domains; the result is ~1 ULP accurate and, crucially, the
 * SAME ~1 ULP value everywhere (reproducibility, not correct rounding, is the
 * guarantee). A GPU shader mirrors these with floatBitsToUint / uintBitsToFloat
 * and precise fma() in the identical order.
 */
static inline float tsr__rrecip_f32(float b) {
    uint32_t seed = 0x7EF127EAu - tsr__bits_from_f32(b);   /* ~1/b */
    float t = tsr__f32_from_bits(seed);
    t = t * fmaf(-b, t, 2.0f);           /* t <- t*(2 - b*t), quadratic decay */
    t = t * fmaf(-b, t, 2.0f);
    t = t * fmaf(-b, t, 2.0f);
    t = t * fmaf(-b, t, 2.0f);
    t = t * fmaf(-b, t, 2.0f);
    return t;
}

static inline float tsr__rrsqrt_f32(float x) {
    uint32_t seed = 0x5f3759dfu - (tsr__bits_from_f32(x) >> 1);  /* ~1/sqrt(x) */
    float r = tsr__f32_from_bits(seed);
    float h = 0.5f * x;                   /* exact (power-of-two multiply) */
    r = r * fmaf(-h, r * r, 1.5f);        /* r <- r*(1.5 - 0.5*x*r^2) */
    r = r * fmaf(-h, r * r, 1.5f);
    r = r * fmaf(-h, r * r, 1.5f);
    r = r * fmaf(-h, r * r, 1.5f);
    r = r * fmaf(-h, r * r, 1.5f);
    return r;
}

/*
 * y * 2^n, for n in [-150, 128]. Built only from an exact power-of-two bit
 * construction and two f32 multiplies, so a GPU shader can mirror it exactly.
 *
 * n is split into two near-equal halves so each factor is a NORMAL power of two
 * (exponent field in range). The first multiply y*f1 stays in the normal range
 * for every (y, n) the callers produce (y in [0.5, 2), n1 in [-75, 64]), so it
 * is exact; the second multiply is therefore a single rounding, correctly
 * rounded even when the result lands in the subnormal range. Left-associated,
 * fixed order.
 */
static inline float tsr__scale2_f32(float y, int n) {
    int n1 = n / 2;
    int n2 = n - n1;
    float f1 = tsr__f32_from_bits((uint32_t)(n1 + 127) << 23);
    float f2 = tsr__f32_from_bits((uint32_t)(n2 + 127) << 23);
    return (y * f1) * f2;
}

/* --------------------------------------------------------------------------
 * exp2(x) = 2^x
 *
 * Range reduce x = k + r, k = round-to-nearest (ties toward +Inf via
 * floorf(x+0.5)), r in [-0.5, 0.5]. 2^r = 1 + r*P(r) with P a fixed degree-5
 * minimax (Cephes exp2f coefficients), evaluated by Horner in fmaf in a fixed
 * order. 2^k applied by tsr__scale2_f32 (exact exponent construction).
 * ------------------------------------------------------------------------ */

/* 2^r minimax coefficients, highest degree first. Decimal source values from
 * Cephes exp2f; the exact f32 bit patterns (hex floats) are the spec, recorded
 * in GOLDENS.md and printed by determinism_test `spec`. */
#define TSR_EXP2_C0 1.535336188319500E-4f
#define TSR_EXP2_C1 1.339887440266574E-3f
#define TSR_EXP2_C2 9.618437357674640E-3f
#define TSR_EXP2_C3 5.550332471162809E-2f
#define TSR_EXP2_C4 2.402264791363012E-1f
#define TSR_EXP2_C5 6.931472028550421E-1f

static inline float tsr__exp2_f32(float x) {
    if (isnan(x)) return x;                 /* NaN propagates */
    if (x >= 128.0f)  return INFINITY;      /* overflow; +Inf lands here */
    if (x <= -150.0f) return 0.0f;          /* underflow to 0; -Inf lands here */

    float k = floorf(x + 0.5f);             /* round to nearest, ties -> +Inf */
    float r = x - k;                        /* r in [-0.5, 0.5] */
    int   n = (int)k;                       /* n in [-150, 128] */

    float p = fmaf(TSR_EXP2_C0, r, TSR_EXP2_C1);
    p = fmaf(p, r, TSR_EXP2_C2);
    p = fmaf(p, r, TSR_EXP2_C3);
    p = fmaf(p, r, TSR_EXP2_C4);
    p = fmaf(p, r, TSR_EXP2_C5);
    float y = fmaf(p, r, 1.0f);             /* 2^r */

    return tsr__scale2_f32(y, n);           /* 2^x */
}

/* --------------------------------------------------------------------------
 * exp(x) = e^x
 *
 * k = round(x * log2e); r = x - k*ln2 with ln2 split into two f32 parts so the
 * subtraction keeps the low bits (reduction in fmaf). e^r = 1 + r + r^2*P(r),
 * P a fixed degree-5 minimax (Cephes expf). Reconstruct with tsr__scale2_f32.
 * ------------------------------------------------------------------------ */

#define TSR_LOG2E  1.44269504088896341f     /* log2(e) */
/* ln2 = TSR_LN2_HI + TSR_LN2_LO (TSR_LN2_LO < 0); Cephes expf split. */
#define TSR_LN2_HI 6.93359375E-1f
#define TSR_LN2_LO -2.12194440E-4f
/* e^r minimax coefficients, highest degree first (Cephes expf). */
#define TSR_EXP_Q0 1.9875691500E-4f
#define TSR_EXP_Q1 1.3981999507E-3f
#define TSR_EXP_Q2 8.3334519073E-3f
#define TSR_EXP_Q3 4.1665795894E-2f
#define TSR_EXP_Q4 1.6666665459E-1f
#define TSR_EXP_Q5 5.0000001201E-1f

static inline float tsr__exp_f32(float x) {
    if (isnan(x)) return x;                      /* NaN propagates */
    if (x >= 88.72283905f)  return INFINITY;     /* overflow; +Inf lands here */
    if (x <= -103.972084f)  return 0.0f;         /* underflow to 0; -Inf here */

    float k = floorf(fmaf(x, TSR_LOG2E, 0.5f));  /* round(x*log2e) */
    float r = fmaf(-k, TSR_LN2_HI, x);           /* x - k*ln2_hi */
    r = fmaf(-k, TSR_LN2_LO, r);                 /* - k*ln2_lo -> r = x - k*ln2 */
    int   n = (int)k;

    float z = r * r;
    float p = fmaf(TSR_EXP_Q0, r, TSR_EXP_Q1);
    p = fmaf(p, r, TSR_EXP_Q2);
    p = fmaf(p, r, TSR_EXP_Q3);
    p = fmaf(p, r, TSR_EXP_Q4);
    p = fmaf(p, r, TSR_EXP_Q5);
    float er = fmaf(p, z, r);                    /* r^2*P(r) + r */
    er = er + 1.0f;                              /* e^r */

    return tsr__scale2_f32(er, n);               /* e^x */
}

/* --------------------------------------------------------------------------
 * erf(x)
 *
 * Odd function, two frozen forms by magnitude:
 *
 *   |x| <= 0.5 : the Maclaurin series erf(x) = x * (B0 + B1 w + ... + B5 w^5),
 *                w = x^2, Bk = (2/sqrt(pi)) * (-1)^k / (k! (2k+1)), Horner in
 *                fmaf. This region is dominated by the 1 - P*exp cancellation in
 *                the A&S form below (catastrophic near 0, and this is exactly the
 *                active region for gelu), so the direct series is used instead.
 *   |x| >  0.5 : Abramowitz & Stegun 7.1.26, erf(a) = 1 - P(t)*e^(-a^2),
 *                t = 1/(1 + p*a), P a fixed degree-5 polynomial in t by Horner in
 *                fmaf. e^(-a^2) uses the reproducible tsr__exp_f32 above so the
 *                whole thing stays inside the reproducible op set.
 *
 * Saturates to +-1 for |x| >= 4 (where erf rounds to +-1 in f32). The branch is
 * a deterministic function of x (same on every lane and thread), so it stays
 * bit-reproducible. Max abs error ~1.5e-7.
 * ------------------------------------------------------------------------ */

/* small-|x| series coefficients, highest degree first: Bk = (2/sqrt(pi))(-1)^k/(k!(2k+1)) */
#define TSR_ERF_B5 -8.5483270689E-4f
#define TSR_ERF_B4 5.2239776255E-3f
#define TSR_ERF_B3 -2.6866170645E-2f
#define TSR_ERF_B2 1.1283791671E-1f
#define TSR_ERF_B1 -3.7612638903E-1f
#define TSR_ERF_B0 1.1283791671f
/* |x| > 0.5 Abramowitz & Stegun 7.1.26 coefficients */
#define TSR_ERF_P  3.275911E-1f
#define TSR_ERF_A1 2.54829592E-1f
#define TSR_ERF_A2 -2.84496736E-1f
#define TSR_ERF_A3 1.421413741f
#define TSR_ERF_A4 -1.453152027f
#define TSR_ERF_A5 1.061405429f

static inline float tsr__erf_f32(float x) {
    if (isnan(x)) return x;                       /* NaN propagates */
    float ax = fabsf(x);

    if (ax <= 0.5f) {                             /* Maclaurin series, odd in x */
        float w = x * x;
        float p = fmaf(TSR_ERF_B5, w, TSR_ERF_B4);
        p = fmaf(p, w, TSR_ERF_B3);
        p = fmaf(p, w, TSR_ERF_B2);
        p = fmaf(p, w, TSR_ERF_B1);
        p = fmaf(p, w, TSR_ERF_B0);
        return x * p;                             /* preserves sign, erf(-0) = -0 */
    }

    float s  = copysignf(1.0f, x);                /* sign of x (+-1) */
    if (ax >= 4.0f) return s;                     /* saturate; +-Inf -> +-1 */

    float t = tsr__rrecip_f32(fmaf(TSR_ERF_P, ax, 1.0f));   /* t = 1/(1+p*ax), fma-only */
    float poly = fmaf(TSR_ERF_A5, t, TSR_ERF_A4);
    poly = fmaf(poly, t, TSR_ERF_A3);
    poly = fmaf(poly, t, TSR_ERF_A2);
    poly = fmaf(poly, t, TSR_ERF_A1);
    poly = poly * t;                              /* P(t)*t */

    float e = tsr__exp_f32(-(ax * ax));           /* e^(-ax^2), reproducible */
    float y = fmaf(-poly, e, 1.0f);               /* 1 - poly*e */
    return s * y;
}

/* --------------------------------------------------------------------------
 * Public ABI. Contiguous f32 arrays, length n. Returns 0, or -1 on a bad
 * argument (negative n, or null pointer with n > 0). n == 0 is a no-op.
 *
 * Each element is independent and runs the identical fmaf sequence, so the
 * OpenMP parallel-for is bit-identical across thread counts by construction
 * (no reduction, no cross-element state).
 * ------------------------------------------------------------------------ */

int tsr_repro_rsqrt_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tsr__rrsqrt_f32(x[i]);       /* deterministic seed + fma Newton */
    }
    return 0;
}

int tsr_repro_exp2_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tsr__exp2_f32(x[i]);
    }
    return 0;
}

int tsr_repro_exp_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tsr__exp_f32(x[i]);
    }
    return 0;
}

int tsr_repro_erf_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tsr__erf_f32(x[i]);
    }
    return 0;
}
