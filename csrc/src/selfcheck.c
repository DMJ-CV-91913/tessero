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

/* Frozen canonical FNV-1a of the KAT gemm_f32 output bytes. 0 = UNBAKED: the first run on a
   known-good build prints the value to stderr; bake it here (or via -DTSR_CANON_GEMM_F32_KAT=0x...)
   to turn the hard canonical gate on. Until baked, the self-check still enforces fast == reference. */
#ifndef TSR_CANON_GEMM_F32_KAT
#define TSR_CANON_GEMM_F32_KAT 0ULL
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

int tsr_selfcheck(void)
{
    float a[TSR_KAT_N * TSR_KAT_N], b[TSR_KAT_N * TSR_KAT_N];
    float cf[TSR_KAT_N * TSR_KAT_N], cr[TSR_KAT_N * TSR_KAT_N];
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
