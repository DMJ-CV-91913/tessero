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
    if (fpcr & (1u << 24)) bad |= FPE_FZ;
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
