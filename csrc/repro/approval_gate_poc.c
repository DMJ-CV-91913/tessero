/* approval_gate_poc.c -- the runtime approval + substitution layer, in miniature.
 *
 * The determinism contract: a fast vectorized clone may run ONLY if it reproduces
 * the canonical reference bit-for-bit. At init the engine runs a known-answer GEMM
 * on both the REFERENCE kernel (explicit fmaf, no clone, fixed order) and the FAST
 * kernel (explicit fmaf WITH target_clones), hashes the f32 bytes, and:
 *   - hashes equal  -> APPROVE the fast path (use it; it is provably equal here).
 *   - hashes differ -> REJECT it and SUBSTITUTE the reference kernel at runtime,
 *                      so the result converges to canonical no matter what the
 *                      compiler/clone did. The substitution is reported, never silent.
 *
 * HONEST LIMIT (stated on purpose): a known-answer test proves equality on THIS
 * input, not for every input. So the sound posture is that the REFERENCE kernel is
 * the attested path; this gate's real job is to catch a build/compiler regression
 * (today's f4b5bd0c) and refuse to run it silently, not to bless arbitrary clones.
 *
 * Pure C11 + libm. No em dashes. Deterministic fixed input. Exit code: 0 APPROVED,
 * 1 SUBSTITUTED (gate did its job), 2 reference self-inconsistent (hard failure). */
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define N 96

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) && !defined(NO_CLONES)
#  define CLONES __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#  define REFONLY __attribute__((target("default")))      /* baseline, no auto-dispatch */
#else
#  define CLONES
#  define REFONLY
#endif

static float A[N * N], B[N * N];

static void fill(void)
{
    uint32_t s = 0x9E3779B9u;
    for (int i = 0; i < N * N; i++) { s = s * 1664525u + 1013904223u; A[i] = (float)((int)((s >> 9) % 2000) - 1000) / 256.0f; }
    for (int i = 0; i < N * N; i++) { s = s * 1664525u + 1013904223u; B[i] = (float)((int)((s >> 9) % 2000) - 1000) / 256.0f; }
}

/* REFERENCE: explicit round-once fmaf, fixed i/p/j order, no clone. The canonical truth. */
REFONLY static void gemm_ref(const float *a, const float *b, float *c)
{
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) c[i * N + j] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int p = 0; p < N; p++) {
            float s = a[i * N + p]; const float *bp = b + p * N; float *ci = c + i * N;
            for (int j = 0; j < N; j++) ci[j] = fmaf(s, bp[j], ci[j]);
        }
}

/* FAST: same explicit fmaf and order, but WITH target_clones (auto-vectorized). */
CLONES static void gemm_fast(const float *a, const float *b, float *c)
{
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) c[i * N + j] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int p = 0; p < N; p++) {
            float s = a[i * N + p]; const float *bp = b + p * N; float *ci = c + i * N;
            for (int j = 0; j < N; j++) ci[j] = fmaf(s, bp[j], ci[j]);
        }
}

static uint64_t fnv(const void *p, size_t n)
{
    const uint8_t *b = p; uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

int main(void)
{
    fill();
    static float Cr[N * N], Cr2[N * N], Cf[N * N];
    gemm_ref(A, B, Cr);
    gemm_ref(A, B, Cr2);                      /* reference must be self-consistent */
    gemm_fast(A, B, Cf);
    uint64_t hr = fnv(Cr, sizeof Cr), hr2 = fnv(Cr2, sizeof Cr2), hf = fnv(Cf, sizeof Cf);

    printf("reference   = %016llx (repeat %s)\n", (unsigned long long)hr, hr == hr2 ? "OK" : "FAIL");
    printf("fast clone  = %016llx\n", (unsigned long long)hf);

    if (hr != hr2) { printf("GATE: reference self-inconsistent -- hard failure\n"); return 2; }
    if (hf == hr)  { printf("GATE: APPROVED -- fast clone equals reference, use the fast path\n"); return 0; }
    printf("GATE: REJECTED -- fast clone diverges; SUBSTITUTE reference kernel (result forced to %016llx)\n",
           (unsigned long long)hr);
    return 1;
}
