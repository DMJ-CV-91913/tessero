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

/* Frozen canonical FNV-1a of the KAT gemm_f32 output bytes. Baked 2026-10-20 from matching runs on
   x86-64 under BOTH gcc-13.3 (Ubuntu 24.04 default) and gcc-12.4 -- the fast dispatched path and the
   no-clone explicit-fma reference agreed on this value under both compilers, which is the determinism
   this guard enforces. A build that produces any other digest is non-canonical (mock / non-FMA /
   wrong-clone) and tsr_selfcheck returns negative. Override at build with -DTSR_CANON_GEMM_F32_KAT=0x...
   only if the canonical form is deliberately re-baselined (and re-verify cross-supplier). */
#ifndef TSR_CANON_GEMM_F32_KAT
#define TSR_CANON_GEMM_F32_KAT 0x06adeb12fccd30c0ULL
#endif

#define TSR_KAT_N 8

static void tsr__kat_fill(float *a, float *b)
{
    uint32_t s = 0x01234567u;                       /* fixed seed -> fixed, portable input */
    for (int i = 0; i < TSR_KAT_N * TSR_KAT_N; i++) { s = s * 1664525u + 1013904223u; a[i] = (float)((int)((s >> 9) % 400) - 200) / 64.0f; }
    for (int i = 0; i < TSR_KAT_N * TSR_KAT_N; i++) { s = s * 1664525u + 1013904223u; b[i] = (float)((int)((s >> 9) % 400) - 200) / 64.0f; }
}

/* Reference: explicit round-once fmaf, fixed i/p/j order, no SIMD clone. The canonical truth. */
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
__attribute__((target("default")))
#endif
static void tsr__kat_ref(const float *a, const float *b, float *c)
{
    for (int i = 0; i < TSR_KAT_N; i++) for (int j = 0; j < TSR_KAT_N; j++) c[i * TSR_KAT_N + j] = 0.0f;
    for (int i = 0; i < TSR_KAT_N; i++)
        for (int p = 0; p < TSR_KAT_N; p++) {
            float s = a[i * TSR_KAT_N + p]; const float *bp = b + p * TSR_KAT_N; float *ci = c + i * TSR_KAT_N;
            for (int j = 0; j < TSR_KAT_N; j++) ci[j] = fmaf(s, bp[j], ci[j]);
        }
}

static uint64_t tsr__fnv(const void *p, size_t n)
{
    const uint8_t *b = p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

/* FP-environment probes (added 2026-10-20). The gemm KAT pins the reduction ORDER and the
 * round-once contraction of the accumulation via digest equality. These pin the rest of the
 * floating-point environment that the canonical digest silently rides on -- round-once FMA,
 * denormal preservation (no flush-to-zero), and signed zero -- so a build whose arithmetic
 * environment is non-canonical (fast-math, FTZ, a contracted mul-add) returns negative here
 * instead of shipping a plausible-but-different digest. This is the CPU twin of the SPIR-V
 * float_controls pinning on the GPU side (see talario docs/determinism/float-controls-hardening.md).
 * Inputs are volatile so the probe measures the RUNTIME environment, not a compile-time fold. */
static int tsr__fp_env_probe(void)
{
    /* (1) round-once FMA. a = b = 1 + 2^-23 (the float just above 1.0). The exact product is
       1 + 2^-22 + 2^-46; c = -(the rounded product) = -(1 + 2^-22). A round-once fmaf keeps the
       low 2^-46 term (result 2^-46); a contracted or mul-then-round path rounds the product first
       and yields exactly 0. */
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
    float a[TSR_KAT_N * TSR_KAT_N], b[TSR_KAT_N * TSR_KAT_N];
    float cf[TSR_KAT_N * TSR_KAT_N], cr[TSR_KAT_N * TSR_KAT_N];
    int env = tsr__fp_env_probe();                            /* FP environment before the KAT */
    if (env != 0) return env;                                 /* -3 contracted, -4 FTZ, -5 signed-zero */
    tsr__kat_fill(a, b);
    tsr__kat_ref(a, b, cr);                                   /* no-clone explicit-fma reference */
    tsr_gemm(TSR_F32, 0, 0, TSR_KAT_N, TSR_KAT_N, TSR_KAT_N,  /* the library's real dispatched path */
             1.0, a, TSR_KAT_N, b, TSR_KAT_N, 0.0, cf, TSR_KAT_N);
    const uint64_t hf = tsr__fnv(cf, sizeof cf);
    const uint64_t hr = tsr__fnv(cr, sizeof cr);
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
