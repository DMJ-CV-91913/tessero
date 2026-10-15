/* Controlled back-to-back comparison: the kernel's cache-blocked fallback gemm vs OpenBLAS cblas_dgemm,
   same process/matrices/instant, so the ratio is machine-load-independent. Build:
     gcc bench_gemm.c -O3 -std=gnu11 -mavx2 -fopenmp -o bench_gemm -lblas -lm
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void cblas_dgemm(int, int, int, int, int, int, double,
                        const double *, int, const double *, int, double, double *, int);

#define BLOCK 64
/* mirrors csrc/src/matmul.c MATMUL(mm_f64,...) — the pre-fix ext path */
static void blocked_gemm(int n, const double *a, const double *b, double *c)
{
    for (int i = 0; i < n; i++) memset(c + (long)i * n, 0, (size_t)n * sizeof(double));
    for (int kk = 0; kk < n; kk += BLOCK)
        for (int jj = 0; jj < n; jj += BLOCK)
            for (int i = 0; i < n; i++) {
                double *ci = c + (long)i * n;
                int km = kk + BLOCK < n ? kk + BLOCK : n;
                int jm = jj + BLOCK < n ? jj + BLOCK : n;
                for (int p = kk; p < km; p++) {
                    double aip = a[(long)i * n + p];
                    const double *bp = b + (long)p * n;
                    for (int j = jj; j < jm; j++) ci[j] += aip * bp[j];
                }
            }
}

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec / 1e9; }

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 1024;
    double *a = malloc((size_t)n * n * 8), *b = malloc((size_t)n * n * 8), *c = malloc((size_t)n * n * 8);
    for (long i = 0; i < (long)n * n; i++) { a[i] = 1.0; b[i] = 1.0; }
    double flops = 2.0 * n * n * n;

    blocked_gemm(n, a, b, c);                          /* warm */
    double t0 = now();
    for (int r = 0; r < 3; r++) blocked_gemm(n, a, b, c);
    double tb = (now() - t0) / 3;

    cblas_dgemm(101, 111, 111, n, n, n, 1.0, a, n, b, n, 0.0, c, n);   /* warm */
    t0 = now();
    for (int r = 0; r < 3; r++) cblas_dgemm(101, 111, 111, n, n, n, 1.0, a, n, b, n, 0.0, c, n);
    double tc = (now() - t0) / 3;

    printf("matmul %dx%d double-precision (same process, median of 3):\n", n, n);
    printf("  fallback (hand-written blocked gemm): %8.2f ms  %6.1f GFLOPS\n", tb * 1e3, flops / tb / 1e9);
    printf("  OpenBLAS (cblas_dgemm):               %8.2f ms  %6.1f GFLOPS\n", tc * 1e3, flops / tc / 1e9);
    printf("  speedup: %.1fx\n", tb / tc);
    return 0;
}
