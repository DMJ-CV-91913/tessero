/*
 * csrc/repro/demo_rsqrt.c
 *
 * Proof-of-concept for the reproducible f32 transcendental work (see
 * REPRO_F32_KICKOFF.md). First primitive: rsqrt(x) = 1/sqrt(x).
 *
 * Thesis: IEEE-754 pins sqrt and divide as correctly-rounded (unlike exp/log),
 * so 1.0f/sqrtf(x) built from them is bit-identical across SIMD level, compiler,
 * vendor, and the GPU, with NO polynomial and NO SFU estimate - as long as we do
 * NOT use -ffast-math (which would swap in the non-reproducible vrsqrtps + Newton).
 *
 * This program fills a fixed deterministic positive-f32 input, computes
 * repro_rsqrt_f32, and writes the raw f32 output bytes to stdout. Pipe to
 * sha256sum. Compile it two ways (scalar vs AVX2) and compare the hashes:
 * equal == bit-identical across SIMD == reproducible. That hash is the golden.
 *
 * Not wired into Tessero's ABI/Makefile yet; that is the follow-up once this
 * proves out. Build/run: see run_demo.sh in this directory.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* The reproducible primitive. Only sqrtf (IEEE correctly-rounded) and divide. */
static void repro_rsqrt_f32(int64_t n, const float *x, float *y) {
    for (int64_t i = 0; i < n; ++i) {
        y[i] = 1.0f / sqrtf(x[i]);
    }
}

int main(void) {
    const int64_t N = 1 << 20;                 /* 1,048,576 elements */
    float *x = (float *)malloc((size_t)N * sizeof(float));
    float *y = (float *)malloc((size_t)N * sizeof(float));
    if (!x || !y) { fprintf(stderr, "alloc failed\n"); return 1; }

    /* Fixed, deterministic, strictly-positive input with a magnitude spread.
       No RNG, no time: the input is a pure function of the index, so the golden
       is stable for everyone. Magnitudes span ~1e-4 .. 1e4 to exercise rounding. */
    for (int64_t i = 0; i < N; ++i) {
        uint32_t s   = (uint32_t)i * 2654435761u;              /* Knuth multiplicative hash */
        float    base = (float)((s >> 8) & 0xffffu) / 65535.0f; /* [0, 1] */
        float    mag  = (i % 7 == 0) ? 1.0e4f : (i % 11 == 0) ? 1.0e-4f : 1.0f;
        x[i] = (base + (1.0f / 1024.0f)) * mag;                /* strictly > 0 */
    }

    repro_rsqrt_f32(N, x, y);

    if (fwrite(y, sizeof(float), (size_t)N, stdout) != (size_t)N) {
        fprintf(stderr, "write failed\n"); return 1;
    }
    free(x); free(y);
    return 0;
}
