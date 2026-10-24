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
 * polynomial evaluated only in the PORTABLY correctly-rounded f32 ops: +, -, *
 * and fmaf (one rounding), plus exact integer bit construction. No divide and
 * no sqrtf (Vulkan does not guarantee their rounding), no libm expf/exp2f/erff,
 * no SVML, no reciprocal/rsqrt estimate, no powf, no f64 or x87 intermediates,
 * no reassociation.
 *
 * The primitives themselves live in csrc/include/tessero_repro_f32.h (the single
 * header-only definition, shared byte-identically with Talario); this TU wraps
 * them as the exported array ABI. The CPU result is the canonical the Talario
 * GPU shader mirrors bit-for-bit. The exact constants (as hex floats) and the evaluation order are
 * the spec; they are recorded in csrc/repro/GOLDENS.md and printed by
 * csrc/repro/determinism_test.c (the `spec` mode).
 *
 * Build: this TU MUST be compiled with -ffp-contract=off -fno-fast-math so the
 * compiler cannot contract or reassociate the non-fma arithmetic. The csrc
 * Makefile has a dedicated rule for this object that adds those flags on top of
 * the inherited CFLAGS; do not remove it.
 */
#include <stdint.h>

#include "../include/tessero_repro_f32.h"

/* Thread count of the parallel loops: the kernel's tsr_set_threads setting (1 = serial, the default), so these
 * regions follow tessero.threads like every other kernel loop instead of OMP_NUM_THREADS (all cores in a php-fpm
 * worker). The count cannot change a result (see below). csrc/repro/run_determinism.sh compiles this file alone,
 * without alloc.c: there the weak reference is null and the loops fall back to OpenMP's default, so its
 * OMP_NUM_THREADS=1 vs 4 comparison still exercises the threaded path. */
#ifdef _OPENMP
#  include <omp.h>
#  if defined(__GNUC__) && !defined(_WIN32)
int tsr_get_threads(void) __attribute__((weak));
static inline int tsr_repro_threads(void) { return tsr_get_threads ? tsr_get_threads() : omp_get_max_threads(); }
#  else
int tsr_get_threads(void);
static inline int tsr_repro_threads(void) { return tsr_get_threads(); }
#  endif
#endif

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
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
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
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
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
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
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
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tsr__erf_f32(x[i]);
    }
    return 0;
}

int tsr_repro_recip_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tsr__recip_f32(x[i]);        /* guarded: any f32 input */
    }
    return 0;
}

int tsr_repro_sigmoid_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) y[i] = tsr__sigmoid_f32(x[i]);
    return 0;
}

int tsr_repro_gelu_tanh_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) y[i] = tsr__gelu_tanh_f32(x[i]);
    return 0;
}

/* Row-wise composites over an m x n row-major matrix. Rows are independent and each
 * runs the identical sequential sequence, so the parallel-for is thread-count invariant. */
int tsr_repro_softmax_f32(int64_t m, int64_t n, const float *x, float *y) {
    if (m < 0 || n < 0) return -1;
    if (m > 0 && n > 0 && (!x || !y)) return -1;
    if (n == 0) return 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t r = 0; r < m; ++r) tsr__softmax_row_f32(x + r * n, y + r * n, n);
    return 0;
}

int tsr_repro_rmsnorm_f32(int64_t m, int64_t n, const float *x, const float *gamma, float eps, float *y) {
    if (m < 0 || n < 0) return -1;
    if (m > 0 && n > 0 && (!x || !gamma || !y)) return -1;
    if (n == 0) return 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t r = 0; r < m; ++r) tsr__rmsnorm_row_f32(x + r * n, gamma, eps, y + r * n, n);
    return 0;
}

int tsr_repro_layernorm_f32(int64_t m, int64_t n, const float *x, const float *gamma, const float *beta,
                            float eps, float *y) {
    if (m < 0 || n < 0) return -1;
    if (m > 0 && n > 0 && (!x || !gamma || !beta || !y)) return -1;
    if (n == 0) return 0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t r = 0; r < m; ++r) tsr__layernorm_row_f32(x + r * n, gamma, beta, eps, y + r * n, n);
    return 0;
}

int tsr_repro_tanh_f32(int64_t n, const float *x, float *y) {
    if (n < 0) return -1;
    if (n > 0 && (!x || !y)) return -1;
#ifdef _OPENMP
#pragma omp parallel for num_threads(tsr_repro_threads()) schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        y[i] = tsr__tanh_f32(x[i]);
    }
    return 0;
}
