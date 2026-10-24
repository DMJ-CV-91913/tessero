/* selfcheck.c -- canonical-form self-check: the runtime approval + substitution guard.
 *
 * Design intent (owner-conceived): do NOT trust that the build produced the canonical
 * kernel. VERIFY it against a frozen known-answer at init / build time, and on any
 * divergence refuse the build or substitute the canonical reference -- forcing the
 * numerics to converge to the canonical form. This is what catches a silent mock /
 * non-FMA / wrong-clone build (the failure that returned a non-canonical gemm hash),
 * instead of shipping wrong bits quietly. Standalone proof: csrc/repro/approval_gate_poc.c.
 *
 * No em dashes. Pure C11 + libm, no external deps. Built automatically (src/*.c glob). */
#include "internal.h"
#include <stdio.h>

/* Frozen canonical digest of the KAT gemm_f32 output (FNV-1a over the little-endian bit patterns, see
   tsr__kat_digest). A build that produces any other digest is non-canonical (mock / non-FMA / wrong order /
   wrong clone) and tsr_selfcheck returns negative. Override at build with -DTSR_CANON_GEMM_F32_KAT=0x...
   only if the canonical form is deliberately re-baselined (and re-verify cross-supplier); 0 = bootstrap.

   History. The first KAT (baked 2026-10-20, 0x06adeb12fccd30c0) multiplied 8x8 matrices of k/64 with
   |k| <= 200: every product and every partial sum was exact in f32, so FMA vs mul+add, any summation order
   and cblas_sgemm all gave the SAME digest and it could not detect what it claimed to. The KAT below uses
   full-mantissa inputs (all 23 fraction bits random), exponents 2^-8..2^8 and both signs, and an odd
   K = 67 reduction, so a two-rounding mul+add, a reordered sum or a double accumulator each change the bits
   (proved by test_numerics_review against the deliberately wrong variants in tsr_selfcheck_digest).

   APPROVED golden (2026-10-24): the dispatched tsr_gemm (1 and 4 OpenMP threads), the no-clone fmaf reference
   and a standalone scalar -O0 -ffp-contract=off reference agree on it; it was reproduced on independent x86-64
   hosts (two microarchitectures) and by a second compiler (gcc and clang), while BLAS sgemm, a wider
   accumulator, mul+add and a reordered reduction each give a different digest. */
#ifndef TSR_CANON_GEMM_F32_KAT
#define TSR_CANON_GEMM_F32_KAT 0x86bbab2bc87f8e3bULL
#endif

#define TSR_KAT_M 8
#define TSR_KAT_N 8
#define TSR_KAT_K 67                                /* odd and >= 67: no power-of-two pairing of the sum */

/* PCG32 (O'Neill, pcg32_random_r): a fixed, portable integer stream, so the inputs are the same bits on
   every platform. */
static uint32_t tsr__pcg32(uint64_t *s)
{
    const uint64_t old = *s;
    *s = old * 6364136223846793005ULL + 1442695040888963407ULL;
    const uint32_t xs = (uint32_t)(((old >> 18) ^ old) >> 27), rot = (uint32_t)(old >> 59);
    return (xs >> rot) | (xs << ((-rot) & 31u));
}

/* one full-mantissa float: random sign, exponent in [-8, 8], all 23 fraction bits random */
static float tsr__kat_value(uint64_t *s)
{
    const uint32_t r = tsr__pcg32(s), q = tsr__pcg32(s);
    const uint32_t bits = (q & 0x80000000u) | ((uint32_t)(127 + (int)(q % 17u) - 8) << 23) | (r & 0x007FFFFFu);
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

static void tsr__kat_fill(float *a, float *b)
{
    uint64_t s = 0x853c49e6748fea9bULL;               /* fixed seed -> fixed, portable input */
    for (int i = 0; i < TSR_KAT_M * TSR_KAT_K; i++) a[i] = tsr__kat_value(&s);
    for (int i = 0; i < TSR_KAT_K * TSR_KAT_N; i++) b[i] = tsr__kat_value(&s);
}

/* Reference: explicit round-once fmaf, fixed i/p/j order, no SIMD clone. The canonical truth. */
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
__attribute__((target("default")))
#endif
static void tsr__kat_ref(const float *a, const float *b, float *c)
{
    for (int i = 0; i < TSR_KAT_M * TSR_KAT_N; i++) c[i] = 0.0f;
    for (int i = 0; i < TSR_KAT_M; i++)
        for (int p = 0; p < TSR_KAT_K; p++) {
            float s = a[i * TSR_KAT_K + p]; const float *bp = b + p * TSR_KAT_N; float *ci = c + i * TSR_KAT_N;
            for (int j = 0; j < TSR_KAT_N; j++) ci[j] = fmaf(s, bp[j], ci[j]);
        }
}

/* Deliberately NON-canonical variants, kept so the test can prove the digest is sensitive to them.
   reverse == 0: the same order with a separate multiply and add (two roundings: what a build without
   round-once FMA computes; the volatile product stops the compiler from contracting it back into an FMA).
   reverse == 1: round-once fmaf but the k sum in reverse order (a reordered / blocked / BLAS reduction). */
static void tsr__kat_wrong(const float *a, const float *b, float *c, int reverse)
{
    for (int i = 0; i < TSR_KAT_M * TSR_KAT_N; i++) c[i] = 0.0f;
    for (int i = 0; i < TSR_KAT_M; i++)
        for (int q = 0; q < TSR_KAT_K; q++) {
            const int p = reverse ? TSR_KAT_K - 1 - q : q;
            float s = a[i * TSR_KAT_K + p]; const float *bp = b + p * TSR_KAT_N; float *ci = c + i * TSR_KAT_N;
            for (int j = 0; j < TSR_KAT_N; j++) {
                if (reverse) ci[j] = fmaf(s, bp[j], ci[j]);
                else { volatile float prod = s * bp[j]; ci[j] = ci[j] + prod; }
            }
        }
}

/* FNV-1a over the IEEE bit patterns, little-endian byte order fixed explicitly, so the digest does not
   depend on host endianness */
static uint64_t tsr__kat_digest(const float *c, int n)
{
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) {
        uint32_t u;
        memcpy(&u, &c[i], sizeof u);
        for (int k = 0; k < 4; k++) { h ^= (u >> (8 * k)) & 0xFFu; h *= 1099511628211ULL; }
    }
    return h;
}

uint64_t tsr_selfcheck_digest(int variant)
{
    float a[TSR_KAT_M * TSR_KAT_K], b[TSR_KAT_K * TSR_KAT_N], c[TSR_KAT_M * TSR_KAT_N];
    tsr__kat_fill(a, b);
    switch (variant) {
    case 0: tsr__kat_ref(a, b, c); break;
    case 1: tsr_gemm(TSR_F32, 0, 0, TSR_KAT_M, TSR_KAT_N, TSR_KAT_K, 1.0, a, TSR_KAT_K, b, TSR_KAT_N, 0.0, c, TSR_KAT_N); break;
    case 2: tsr__kat_wrong(a, b, c, 0); break;
    case 3: tsr__kat_wrong(a, b, c, 1); break;
    default: return 0;
    }
    return tsr__kat_digest(c, TSR_KAT_M * TSR_KAT_N);
}

uint64_t tsr_selfcheck_canonical(void) { return (uint64_t)TSR_CANON_GEMM_F32_KAT; }

/* FP-environment probes (added 2026-10-20). They pin behaviour the canonical digest silently rides on:
 * a correctly rounded fmaf, denormal preservation (no flush-to-zero) and signed zero. The CONTROL REGISTER
 * itself (MXCSR / FPCR rounding mode, FTZ/DAZ, trap masks, on every OpenMP worker) is read directly by
 * tsr_fpenv_check (fpenv.c), which tsr_selfcheck runs first. This is the CPU twin of the SPIR-V
 * float_controls pinning on the GPU side (see talario docs/determinism/float-controls-hardening.md).
 * Inputs are volatile so the probe measures the RUNTIME environment, not a compile-time fold. */
static int tsr__fp_env_probe(void)
{
    /* (1) libm fmaf is correctly rounded. a = b = 1 + 2^-23 (the float just above 1.0). The exact product
       is 1 + 2^-22 + 2^-46; c = -(the rounded product) = -(1 + 2^-22). A correct fmaf keeps the low 2^-46
       term (result 2^-46); an fmaf that rounds the product first (a broken libm or a naive software
       fallback) yields exactly 0. This is an explicit library call, so it does NOT test whether the
       compiler contracts a*b+c in kernels: that is what the KAT's mul+add sensitivity covers. */
    volatile float a = 0x1.000002p0f, b = 0x1.000002p0f, c = -0x1.000004p0f;
    if (fmaf((float)a, (float)b, (float)c) != 0x1p-46f) return -3;

    /* (2) denormal preservation (no flush-to-zero). 2^-140 is subnormal (min normal is 2^-126);
       under FTZ it reads or propagates as 0. */
    volatile float sub = 0x1p-140f;
    if ((float)sub == 0.0f)              return -4;
    if ((float)sub * 2.0f != 0x1p-139f)  return -4;

    /* (3) signed zero per IEEE-754: -0 is distinct from +0 and divides to -inf. */
    volatile float nz = -0.0f;
    if (copysignf(1.0f, (float)nz) != -1.0f) return -5;
    if (1.0f / (float)nz != -INFINITY)       return -5;

    return 0;
}

int tsr_selfcheck(void)
{
    /* the control registers first: an unmasked trap would fault inside the probes below */
    if (tsr_fpenv_check(NULL, 0) != 0) return -6;             /* -6 FP control register non-canonical */
    int env = tsr__fp_env_probe();
    if (env != 0) return env;                                 /* -3 fmaf not round-once, -4 FTZ, -5 signed-zero */
    const uint64_t hf = tsr_selfcheck_digest(1);              /* the library's real dispatched path */
    const uint64_t hr = tsr_selfcheck_digest(0);              /* no-clone explicit-fma reference */
    const uint64_t want = (uint64_t)TSR_CANON_GEMM_F32_KAT;

    if (want == 0ULL) {                                       /* bootstrap: emit the value to bake */
        fprintf(stderr, "[tessero selfcheck] bootstrap -- canonical KAT gemm_f32 digest = %016llx "
                        "(bake into TSR_CANON_GEMM_F32_KAT)\n", (unsigned long long)hr);
        return (hf == hr) ? 0 : -1;                           /* still require fast == reference */
    }
    if (hr != want) return -2;                                /* reference not canonical: bad toolchain */
    if (hf != hr)   return -1;                                /* fast clone diverged: substitute reference */
    return 0;                                                 /* canonical: fast == reference == frozen */
}
