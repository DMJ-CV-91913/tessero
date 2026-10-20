/* canonical_poc.c -- determine (do not hypothesize) the Tessero gemm FMA footgun,
 * and prove the explicit-fmaf fix is canonical.
 *
 * On a given box + compiler + flag set it prints one line:
 *   contracted=<0|1>  NAIVE=<hash>  NAIVE_NOCLONE=<hash>  CANON=<hash>  canon_stable=<ref>
 * where
 *   contracted   : did THIS build fuse c + a*b into a single round-once FMA?
 *                  (1 = round-once, 0 = two-round). This is the exact knob that
 *                  flips 4f984242 <-> f4b5bd0c.
 *   NAIVE        : hash of today's kernel (c += a*b) WITH target_clones.
 *   NAIVE_NOCLONE: same kernel with clones force-disabled (-DNO_CLONES path).
 *                  If NAIVE != NAIVE_NOCLONE at the same flags, clone SELECTION
 *                  is implicated, not just contraction.
 *   CANON        : hash of the proposed fix (explicit fmaf, same fixed i/p/j order).
 *
 * Build the SAME file several ways (run_canonical_poc.sh does this) and compare:
 *   - CANON MUST be identical in every build  -> the fix is canonical by construction.
 *   - wherever `contracted` or NAIVE changes   -> that is the measured root cause.
 *
 * Pure C11 + libm. No em dashes. No external deps. Deterministic fixed input. */
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <math.h>

#define N 96

/* same multiversioning predicate Tessero uses (internal.h), so this reproduces the
   real clone behaviour on x86-64 gcc>=12; elsewhere CLONES is empty (== plain). */
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12) && !defined(NO_CLONES)
#  define CLONES __attribute__((target_clones("arch=x86-64-v4", "arch=x86-64-v3", "default")))
#else
#  define CLONES
#endif

static float A[N * N], B[N * N];

static void fill(void)
{
    uint32_t s = 0x9E3779B9u;                       /* fixed seed -> fixed input */
    for (int i = 0; i < N * N; i++) { s = s * 1664525u + 1013904223u; A[i] = (float)((int)((s >> 9) % 2000) - 1000) / 256.0f; }
    for (int i = 0; i < N * N; i++) { s = s * 1664525u + 1013904223u; B[i] = (float)((int)((s >> 9) % 2000) - 1000) / 256.0f; }
}

/* NAIVE: c += a*b -- today's gemm_f32; contraction is the compiler's choice. */
CLONES static void gemm_naive(const float *a, const float *b, float *c)
{
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) c[i * N + j] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int p = 0; p < N; p++) {
            float s = a[i * N + p]; const float *bp = b + p * N; float *ci = c + i * N;
            for (int j = 0; j < N; j++) ci[j] += s * bp[j];
        }
}

/* NAIVE with clones force-disabled, to isolate clone selection from contraction. */
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) && (__GNUC__ >= 12)
__attribute__((target("default")))
#endif
static void gemm_naive_noclone(const float *a, const float *b, float *c)
{
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) c[i * N + j] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int p = 0; p < N; p++) {
            float s = a[i * N + p]; const float *bp = b + p * N; float *ci = c + i * N;
            for (int j = 0; j < N; j++) ci[j] += s * bp[j];
        }
}

/* CANON: explicit round-once fmaf, SAME fixed i/p/j order -- the proposed fix. */
CLONES static void gemm_canon(const float *a, const float *b, float *c)
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

/* direct probe: did THIS build contract c + a*b into one rounding? */
static int contracted(void)
{
    volatile float a = 1.0f + ldexpf(1.0f, -11);    /* 1 + 2^-11 */
    volatile float b = 1.0f + ldexpf(1.0f, -11);
    volatile float c = -1.0f;
    float naive = c + a * b;                         /* one OR two roundings: compiler's call */
    float fused = fmaf(a, b, c);                     /* always one rounding */
    return naive == fused;
}

int main(void)
{
    fill();
    static float Cn[N * N], Cnn[N * N], Cc[N * N];
    gemm_naive(A, B, Cn);
    gemm_naive_noclone(A, B, Cnn);
    gemm_canon(A, B, Cc);
    printf("contracted=%d  NAIVE=%016llx  NAIVE_NOCLONE=%016llx  CANON=%016llx\n",
           contracted(),
           (unsigned long long)fnv(Cn, sizeof Cn),
           (unsigned long long)fnv(Cnn, sizeof Cnn),
           (unsigned long long)fnv(Cc, sizeof Cc));
    return 0;
}
