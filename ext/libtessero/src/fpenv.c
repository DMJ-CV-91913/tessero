/* fpenv.c -- runtime floating-point environment gate.
 *
 * Every bit-exact guarantee (tier A in docs/reference/numerical-guarantees.md) assumes the IEEE default
 * environment: round-to-nearest-even, subnormals preserved (no flush-to-zero / denormals-are-zero) and
 * exceptions masked. A host process can break that silently: a library built with -ffast-math links
 * crtfastmath.o, which sets FTZ/DAZ at load for the WHOLE process, and any code can call fesetround or
 * write MXCSR/FPCR. tsr_fpenv_check reads the control register on the calling thread and on every OpenMP
 * worker and reports what is wrong; tsr_fpenv_canonicalize is the explicit, opt-in repair (never applied
 * silently: changing another library's FP mode under it would be a surprise of its own).
 *
 * Cost: one register read + fegetround per thread. tsr_fpenv_check_thread (calling thread only) is a few
 * nanoseconds; tsr_fpenv_check adds one OpenMP parallel region. Neither runs per element or per call: the
 * check runs at load (extension MINIT, the FFI binding) and in tsr_selfcheck. No em dashes. */
#include "internal.h"
#include <fenv.h>
#include <stdio.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#endif

/* bitmask returned by tsr_fpenv_check (0 = canonical); see tessero.h */
#define FPE_FTZ    1      /* x86 MXCSR.FTZ (bit 15) / aarch64 FPCR.FZ (bit 24): results flushed to zero */
#define FPE_DAZ    2      /* x86 MXCSR.DAZ (bit 6): subnormal inputs read as zero */
#define FPE_RMODE  4      /* MXCSR.RC (bits 14:13) / FPCR.RMode (bits 23:22) not round-to-nearest */
#define FPE_TRAPS  8      /* MXCSR exception masks (bits 12:7) not all set / FPCR trap enables (8-12, 15) set */
#define FPE_ROUND  16     /* fegetround() != FE_TONEAREST (x87 control word, or the only probe elsewhere) */
#define FPE_FZ16   32     /* aarch64 FPCR.FZ16 (bit 19): half-precision flush-to-zero */
#define FPE_WORKER 64     /* set together with the bits above when an OpenMP worker (not the caller) differs */

static int fpenv_probe(void)
{
    int bad = 0;
#if defined(__x86_64__) || (defined(__i386__) && defined(__SSE__))
    uint32_t csr;
    __asm__ volatile("stmxcsr %0" : "=m"(csr));
    if (csr & (1u << 15)) bad |= FPE_FTZ;
    if (csr & (1u << 6)) bad |= FPE_DAZ;
    if (csr & (3u << 13)) bad |= FPE_RMODE;
    if ((csr & (0x3Fu << 7)) != (0x3Fu << 7)) bad |= FPE_TRAPS;
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    if (fpcr & (1u << 24)) bad |= FPE_FTZ;
    if (fpcr & (1u << 19)) bad |= FPE_FZ16;
    if (fpcr & (3u << 22)) bad |= FPE_RMODE;
    if (fpcr & ((0x1Fu << 8) | (1u << 15))) bad |= FPE_TRAPS;
#endif
#if defined(FE_TONEAREST)
    if (fegetround() != FE_TONEAREST) bad |= FPE_ROUND;
#endif
    return bad;
}

static void fpenv_reason(int bad, char *buf, int64_t cap)
{
    if (!buf || cap <= 0) return;
    buf[0] = '\0';
    if (!bad) { snprintf(buf, (size_t)cap, "canonical"); return; }
    static const struct { int bit; const char *what; } names[] = {
#if defined(__aarch64__)
        {FPE_FTZ, "FPCR.FZ (bit 24) set: flush-to-zero"},
        {FPE_FZ16, "FPCR.FZ16 (bit 19) set: half-precision flush-to-zero"},
        {FPE_RMODE, "FPCR.RMode (bits 23:22) not round-to-nearest"},
        {FPE_TRAPS, "FPCR trap enables (bits 8-12, 15) set"},
#else
        {FPE_FTZ, "MXCSR.FTZ (bit 15) set: flush-to-zero"},
        {FPE_DAZ, "MXCSR.DAZ (bit 6) set: denormals-are-zero"},
        {FPE_RMODE, "MXCSR.RC (bits 14:13) not round-to-nearest"},
        {FPE_TRAPS, "MXCSR exception masks (bits 12:7) not all set: FP traps enabled"},
#endif
        {FPE_ROUND, "fegetround() != FE_TONEAREST"},
        {FPE_WORKER, "(on an OpenMP worker thread)"},
    };
    size_t used = 0;
    for (size_t k = 0; k < sizeof names / sizeof names[0]; k++) {
        if (!(bad & names[k].bit)) continue;
        int w = snprintf(buf + used, (size_t)cap - used, "%s%s", used ? "; " : "", names[k].what);
        if (w < 0 || (size_t)w >= (size_t)cap - used) break;
        used += (size_t)w;
    }
}

int tsr_fpenv_check_thread(char *buf, int64_t cap)
{
    const int bad = fpenv_probe();
    fpenv_reason(bad, buf, cap);
    return bad;
}

int tsr_fpenv_check(char *buf, int64_t cap)
{
    int bad = fpenv_probe();
#ifdef _OPENMP
    /* every thread the library's parallel loops can use: the larger of its own setting and OpenMP's */
    int nt = tsr_get_threads();
    if (omp_get_max_threads() > nt) nt = omp_get_max_threads();
    if (nt > 1) {
        int wbad = 0;
#pragma omp parallel num_threads(nt) reduction(| : wbad)
        wbad |= fpenv_probe();
        if (wbad & ~bad) bad |= wbad | FPE_WORKER;
    }
#endif
    fpenv_reason(bad, buf, cap);
    return bad;
}

static void fpenv_repair(void)
{
#if defined(__x86_64__) || (defined(__i386__) && defined(__SSE__))
    uint32_t csr;
    __asm__ volatile("stmxcsr %0" : "=m"(csr));
    csr &= ~((1u << 15) | (1u << 6) | (3u << 13));            /* FTZ, DAZ off; round to nearest */
    csr |= 0x3Fu << 7;                                          /* all exceptions masked */
    __asm__ volatile("ldmxcsr %0" : : "m"(csr));
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    fpcr &= ~(uint64_t)((1u << 24) | (1u << 19) | (3u << 22) | (0x1Fu << 8) | (1u << 15));
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
#endif
#if defined(FE_TONEAREST)
    fesetround(FE_TONEAREST);
#endif
}

int tsr_fpenv_canonicalize(void)
{
    fpenv_repair();
#ifdef _OPENMP
    int nt = tsr_get_threads();
    if (omp_get_max_threads() > nt) nt = omp_get_max_threads();
    if (nt > 1) {
#pragma omp parallel num_threads(nt)
        fpenv_repair();
    }
#endif
    return tsr_fpenv_check(NULL, 0);
}


/* ---- opt-in GPU-matching profile: tiny numbers and NaN bits (2026-10) ----------------------------
 * NVIDIA GPUs flush subnormal inputs and results to signed zero and expose no preserve mode, while the
 * canonical environment above keeps subnormals (IEEE-754), and NaN results can carry different bit patterns
 * on different processors. These functions let a caller reproduce the GPU's behaviour on the CPU, or
 * normalise outputs, WITHOUT changing any existing function: nothing here runs unless it is called.
 *
 * tsr_fpenv_flush_enter / tsr_fpenv_flush_leave: switch the CALLING THREAD to flush-to-zero (x86-64 MXCSR
 * FTZ + DAZ; aarch64 FPCR.FZ) for a scope and restore its previous control word exactly. A multi-threaded
 * caller enters and leaves on each of its own threads. Inside a scope tsr_fpenv_check reports the thread as
 * non-canonical and tsr_selfcheck returns -6, by design.
 * tsr_flush_subnormal_f32 / tsr_canon_nan_f32 rewrite values in place with integer bit operations only (no
 * FP arithmetic, so the result does not depend on the FP environment); tsr_nan_first_f32 locates the first
 * NaN. Ownership: the caller owns x; nothing is allocated. */
int tsr_fpenv_flush_enter(uint64_t *saved)
{
    if (!saved) return -1;
#if defined(__x86_64__) || (defined(__i386__) && defined(__SSE__))
    uint32_t csr;
    __asm__ volatile("stmxcsr %0" : "=m"(csr));
    *saved = csr;
    csr |= (1u << 15) | (1u << 6);                              /* FTZ: results flushed; DAZ: inputs read as zero */
    __asm__ volatile("ldmxcsr %0" : : "m"(csr));
    return 0;
#elif defined(__aarch64__)
    uint64_t fpcr;
    __asm__ volatile("mrs %0, fpcr" : "=r"(fpcr));
    *saved = fpcr;
    fpcr |= (uint64_t)1u << 24;                                 /* FZ: subnormal inputs and results flushed */
    __asm__ volatile("msr fpcr, %0" : : "r"(fpcr));
    return 0;
#else
    *saved = 0;
    return -2;                                                  /* no flush control on this target */
#endif
}

int tsr_fpenv_flush_leave(uint64_t saved)
{
#if defined(__x86_64__) || (defined(__i386__) && defined(__SSE__))
    uint32_t csr = (uint32_t)saved;
    __asm__ volatile("ldmxcsr %0" : : "m"(csr));
    return 0;
#elif defined(__aarch64__)
    __asm__ volatile("msr fpcr, %0" : : "r"(saved));
    return 0;
#else
    (void)saved;
    return -2;
#endif
}

int64_t tsr_flush_subnormal_f32(float *x, int64_t n)
{
    if (!x || n <= 0) return 0;
    int64_t changed = 0;
    for (int64_t i = 0; i < n; i++) {
        uint32_t b;
        memcpy(&b, &x[i], sizeof b);
        if ((b & 0x7F800000u) == 0u && (b & 0x007FFFFFu) != 0u) {   /* subnormal -> signed zero */
            b &= 0x80000000u;
            memcpy(&x[i], &b, sizeof b);
            changed++;
        }
    }
    return changed;
}

int64_t tsr_canon_nan_f32(float *x, int64_t n)
{
    if (!x || n <= 0) return 0;
    int64_t changed = 0;
    for (int64_t i = 0; i < n; i++) {
        uint32_t b;
        memcpy(&b, &x[i], sizeof b);
        if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u && b != 0x7FC00000u) {
            b = 0x7FC00000u;                                    /* the canonical quiet NaN */
            memcpy(&x[i], &b, sizeof b);
            changed++;
        }
    }
    return changed;
}

int64_t tsr_nan_first_f32(const float *x, int64_t n)
{
    if (!x || n <= 0) return -1;
    for (int64_t i = 0; i < n; i++) {
        uint32_t b;
        memcpy(&b, &x[i], sizeof b);
        if ((b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u) return i;
    }
    return -1;
}

/* f64 twins. NVIDIA GPUs keep f64 subnormals (their flush applies to f32 only), so f64 work should normally NOT
 * be flushed; canonical NaN and the first-NaN locator are useful in f64 regardless. */
int64_t tsr_flush_subnormal_f64(double *x, int64_t n)
{
    if (!x || n <= 0) return 0;
    int64_t changed = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t b;
        memcpy(&b, &x[i], sizeof b);
        if ((b & 0x7FF0000000000000ull) == 0u && (b & 0x000FFFFFFFFFFFFFull) != 0u) {
            b &= 0x8000000000000000ull;
            memcpy(&x[i], &b, sizeof b);
            changed++;
        }
    }
    return changed;
}

int64_t tsr_canon_nan_f64(double *x, int64_t n)
{
    if (!x || n <= 0) return 0;
    int64_t changed = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t b;
        memcpy(&b, &x[i], sizeof b);
        if ((b & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && (b & 0x000FFFFFFFFFFFFFull) != 0u &&
            b != 0x7FF8000000000000ull) {
            b = 0x7FF8000000000000ull;                          /* the canonical quiet NaN */
            memcpy(&x[i], &b, sizeof b);
            changed++;
        }
    }
    return changed;
}

int64_t tsr_nan_first_f64(const double *x, int64_t n)
{
    if (!x || n <= 0) return -1;
    for (int64_t i = 0; i < n; i++) {
        uint64_t b;
        memcpy(&b, &x[i], sizeof b);
        if ((b & 0x7FF0000000000000ull) == 0x7FF0000000000000ull && (b & 0x000FFFFFFFFFFFFFull) != 0u) return i;
    }
    return -1;
}
