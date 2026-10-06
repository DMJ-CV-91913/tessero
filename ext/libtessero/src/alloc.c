#include <stdlib.h>
#include <stdatomic.h>
#include "internal.h"

#ifdef _WIN32
#  include <malloc.h>
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#endif
#if defined(__linux__)
#  include <sys/mman.h>
#endif

/* Blocks this large are 2 MiB aligned and marked for transparent huge pages
   (as NumPy does), which removes most first-touch page-fault cost. Below 32 MiB the
   madvise + 2 MiB zeroing costs more than it saves for short-lived temporaries.
   Build with -DTSR_NO_HUGE to disable. */
#define TSR_HUGE_THRESHOLD ((int64_t)32 << 20)
#define TSR_HUGE_ALIGN ((size_t)2 << 20)

static _Atomic int64_t g_allocated = 0;
static _Atomic int64_t g_peak = 0;
static _Atomic int64_t g_budget = 0;

const char *tsr_version(void) { return "0.2.0"; }

#ifdef _OPENMP
#  include <omp.h>
#endif
static _Atomic int g_threads = 1;

void tsr_set_threads(int n)
{
    if (n < 1) n = 1;
#ifdef _OPENMP
    int max = omp_get_num_procs();
    if (n > max * 4) n = max * 4;
#else
    n = 1;
#endif
    atomic_store(&g_threads, n);
}

int tsr_get_threads(void) { return atomic_load(&g_threads); }

int tsr_openmp(void)
{
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}

int tsr_simd_level(void)
{
#if defined(__x86_64__) && defined(__GNUC__)
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")) return 2;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) return 1;
    return 0;
#elif defined(__aarch64__)
    return 3;
#else
    return 0;
#endif
}

static void *aligned(int64_t bytes)
{
    size_t size = (size_t)((bytes + 63) & ~(int64_t)63);
    if (size == 0) size = 64;
#ifdef _WIN32
    return _aligned_malloc(size, 64);
#else
#  if defined(__linux__) && defined(MADV_HUGEPAGE) && !defined(TSR_NO_HUGE)
    if (bytes >= TSR_HUGE_THRESHOLD) {
        size_t big = (size + TSR_HUGE_ALIGN - 1) & ~(TSR_HUGE_ALIGN - 1);
        void *p = aligned_alloc(TSR_HUGE_ALIGN, big);
        if (p != NULL) madvise(p, big, MADV_HUGEPAGE);
        return p;
    }
#  endif
    return aligned_alloc(64, size);
#endif
}

/* the machine's physical memory in bytes (0 when unknown), read once (internal.h) */
int64_t tsr_physical_bytes(void)
{
    static _Atomic int64_t cached = -1;
    int64_t v = atomic_load(&cached);
    if (v >= 0) return v;
#if defined(_WIN32)
    MEMORYSTATUSEX st;
    st.dwLength = sizeof st;
    v = GlobalMemoryStatusEx(&st) ? (int64_t)st.ullTotalPhys : 0;
#elif defined(_SC_PHYS_PAGES) && defined(_SC_PAGESIZE)
    const long pages = sysconf(_SC_PHYS_PAGES), size = sysconf(_SC_PAGESIZE);
    v = pages > 0 && size > 0 && (double)pages * (double)size < 9.2e18 ? (int64_t)pages * (int64_t)size : 0;
#else
    v = 0;
#endif
    atomic_store(&cached, v);
    return v;
}

void *tsr_alloc(int64_t bytes)
{
    if (bytes < 0) return NULL;
    /* Without a budget, one block larger than the machine's RAM is refused (a MemoryError) instead of being
       handed out by an overcommitting allocator and killing the process when it is filled (ADR 0004). */
    if (atomic_load(&g_budget) == 0) {
        const int64_t phys = tsr_physical_bytes();
        if (phys > 0 && bytes > phys) return NULL;
    }
    int64_t now = atomic_fetch_add(&g_allocated, bytes) + bytes;
    int64_t budget = atomic_load(&g_budget);
    if (budget > 0 && now > budget) {
        atomic_fetch_sub(&g_allocated, bytes);
        return NULL;
    }
    void *p = aligned(bytes);
    if (p == NULL) {
        atomic_fetch_sub(&g_allocated, bytes);
        return NULL;
    }
    int64_t peak = atomic_load(&g_peak);
    while (now > peak && !atomic_compare_exchange_weak(&g_peak, &peak, now)) {
    }
    return p;
}

void *tsr_calloc(int64_t bytes)
{
    void *p = tsr_alloc(bytes);
    if (p != NULL) memset(p, 0, (size_t)bytes);
    return p;
}

void tsr_free(void *p, int64_t bytes)
{
    if (p == NULL) return;
    atomic_fetch_sub(&g_allocated, bytes);
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

int64_t tsr_allocated(void) { return atomic_load(&g_allocated); }
int64_t tsr_peak(void) { return atomic_load(&g_peak); }
void tsr_set_budget(int64_t bytes) { atomic_store(&g_budget, bytes < 0 ? 0 : bytes); }
int64_t tsr_budget(void) { return atomic_load(&g_budget); }
