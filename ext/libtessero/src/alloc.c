/* sched_getaffinity / CPU_COUNT and dladdr / RTLD_DEFAULT are GNU extensions (glibc, musl) */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#  define _GNU_SOURCE
#endif
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>
#include "internal.h"
#include "../include/tessero_version.h"

#ifdef _WIN32
#  include <malloc.h>
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <unistd.h>
#  include <dlfcn.h>
#  include <pthread.h>
#endif
#if defined(__linux__)
#  include <sys/mman.h>
#  include <sched.h>
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

/* ---------------------------------------------------------------- version and capabilities (ADR 0014) */

const char *tsr_version(void) { return TSR_VERSION_STRING; }
int tsr_abi_version(void) { return TSR_ABI_VERSION; }

#ifdef _OPENMP
#  include <omp.h>
#endif
static _Atomic int g_threads = 1;

/* The CPU quota of this process's cgroup (docker --cpus, Kubernetes limits), rounded up; 0 when there is none. */
static int tsr_cgroup_cpus(void)
{
#if defined(__linux__)
    long long quota = -1, period = 0;
    char path[512], line[256];
    FILE *f = NULL;
    /* cgroup v2: <mount>/<own cgroup>/cpu.max holds "max 100000" or "200000 100000"; a namespaced container
       sees its own cgroup at the mount root */
    FILE *cg = fopen("/proc/self/cgroup", "r");
    if (cg) {
        while (fgets(line, sizeof line, cg)) {
            if (strncmp(line, "0::", 3) == 0) {
                line[strcspn(line, "\n")] = '\0';
                snprintf(path, sizeof path, "/sys/fs/cgroup%s/cpu.max", line + 3);
                f = fopen(path, "r");
                break;
            }
        }
        fclose(cg);
    }
    if (!f) f = fopen("/sys/fs/cgroup/cpu.max", "r");
    if (f) {
        char q[32] = {0};
        if (fscanf(f, "%31s %lld", q, &period) == 2 && strcmp(q, "max") != 0) quota = atoll(q);
        fclose(f);
    } else if ((f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) != NULL) {   /* cgroup v1 */
        if (fscanf(f, "%lld", &quota) != 1) quota = -1;
        fclose(f);
        if ((f = fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) != NULL) {
            if (fscanf(f, "%lld", &period) != 1) period = 0;
            fclose(f);
        }
    }
    if (quota > 0 && period > 0) {
        const long long n = (quota + period - 1) / period;
        return n < 1 ? 1 : (n > (1 << 20) ? (1 << 20) : (int)n);
    }
#endif
    return 0;
}

/* The CPUs this process may actually run on, computed once: the scheduler affinity mask (taskset, cpusets,
   the Kubernetes static CPU manager) and the cgroup CPU quota, whichever is smaller, at least 1.
   omp_get_num_procs() sees the affinity mask libgomp started with but not a quota. */
int tsr_effective_cpus(void)
{
    static _Atomic int cached = 0;
    int v = atomic_load(&cached);
    if (v > 0) return v;
    v = 0;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) v = CPU_COUNT(&set);
#endif
#ifdef _OPENMP
    if (v <= 0) v = omp_get_num_procs();
#elif !defined(_WIN32) && defined(_SC_NPROCESSORS_ONLN)
    if (v <= 0) v = (int)sysconf(_SC_NPROCESSORS_ONLN);
#endif
    const int q = tsr_cgroup_cpus();
    if (q > 0 && (v <= 0 || q < v)) v = q;
    if (v < 1) v = 1;
    atomic_store(&cached, v);
    return v;
}

void tsr_set_threads(int n)
{
    if (n < 1) n = 1;
#ifdef _OPENMP
    const int max = tsr_effective_cpus();
    if (n > max * 4) n = max * 4;
#else
    n = 1;
#endif
    atomic_store(&g_threads, n);
}

/* set once a kernel has asked for (and so may have started) a multi-threaded OpenMP region in this process */
static _Atomic int g_threaded = 0;

int tsr_get_threads(void)
{
    const int n = atomic_load(&g_threads);
    if (n > 1 && !atomic_load_explicit(&g_threaded, memory_order_relaxed))
        atomic_store_explicit(&g_threaded, 1, memory_order_relaxed);
    return n;
}

/* Fork safety: the child of a process that ran multi-threaded OpenMP regions must not start another one (libgomp's
   worker pool belongs to the parent and is not recreated, so the child can deadlock). Such a child therefore runs
   serial until the thread count is set again explicitly: the extension's RINIT does that per request (NTS), an
   FFI caller calls tsr_set_threads. A parent that only configured threads and never computed with them (a
   php-fpm or Apache master after MINIT) is safe to fork and its children keep the setting. Running Tessero
   compute in a process that forks afterwards (opcache preload, a master process) is not supported. */
#if !defined(_WIN32) && defined(__GNUC__)
static void tsr_atfork_child(void)
{
    if (atomic_load(&g_threaded)) {
        atomic_store(&g_threads, 1);
        atomic_store(&g_threaded, 0);
    }
}
__attribute__((constructor)) static void tsr_install_atfork(void) { pthread_atfork(NULL, NULL, tsr_atfork_child); }
#endif

/* ---------------------------------------------------------------- BLAS thread pool (ADR 0014) */

/* The BLAS in the process (OpenBLAS through liblapacke/libblas on Linux, or MKL) is found by name at first use,
   so the kernel has no link-time dependency on a particular BLAS. This library's own dependency tree is searched
   first (an FFI dlopen without RTLD_GLOBAL keeps libblas out of the global scope), then the global scope. */
typedef void (*tsr_blas_set_fn)(int);
typedef int (*tsr_blas_get_fn)(void);
static tsr_blas_set_fn g_blas_set;
static tsr_blas_get_fn g_blas_get;
static _Atomic int g_blas_state = 0;     /* 0 unresolved, 1 resolving, 2 resolved */

#if !defined(_WIN32)
static void *tsr_blas_sym(void *self, const char *name)
{
    void *p = self ? dlsym(self, name) : NULL;
    if (!p) p = dlsym(RTLD_DEFAULT, name);
    return p;
}
#endif

static void tsr_blas_resolve(void)
{
    if (atomic_load_explicit(&g_blas_state, memory_order_acquire) == 2) return;
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_blas_state, &expected, 1)) {
        while (atomic_load_explicit(&g_blas_state, memory_order_acquire) != 2) { }
        return;
    }
#if !defined(_WIN32)
    void *self = NULL;
    Dl_info info;
    if (dladdr((void *)&tsr_blas_resolve, &info) && info.dli_fname) self = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
    static const char *const setters[] = {"openblas_set_num_threads", "goto_set_num_threads", "MKL_Set_Num_Threads"};
    static const char *const getters[] = {"openblas_get_num_threads", NULL, "MKL_Get_Max_Threads"};
    for (size_t i = 0; i < sizeof setters / sizeof setters[0] && !g_blas_set; i++) {
        void *s = tsr_blas_sym(self, setters[i]);
        if (!s) continue;
        g_blas_set = (tsr_blas_set_fn)s;
        if (getters[i]) g_blas_get = (tsr_blas_get_fn)tsr_blas_sym(self, getters[i]);
    }
    if (self) dlclose(self);   /* drops the reference RTLD_NOLOAD took; the object itself stays loaded */
#endif
    atomic_store_explicit(&g_blas_state, 2, memory_order_release);
}

int tsr_set_blas_threads(int n)
{
    if (n < 1) return TSR_EARG;
    tsr_blas_resolve();
    if (!g_blas_set) return TSR_ETYPE;
    g_blas_set(n);
    return TSR_OK;
}

int tsr_get_blas_threads(void)
{
    tsr_blas_resolve();
    return g_blas_get ? g_blas_get() : 0;
}

#if defined(__linux__) && defined(__ELF__)
#  include <elf.h>
#  include <link.h>
/* 1 when this shared object was linked with -z nodelete (DF_1_NODELETE in its own DT_FLAGS_1), else 0 */
static int tsr_linked_nodelete(void)
{
    extern ElfW(Dyn) _DYNAMIC[] __attribute__((weak));
    if (!_DYNAMIC) return 0;
    for (const ElfW(Dyn) *d = _DYNAMIC; d->d_tag != DT_NULL; d++)
        if (d->d_tag == DT_FLAGS_1) return (d->d_un.d_val & DF_1_NODELETE) != 0;
    return 0;
}
#else
static int tsr_linked_nodelete(void) { return 0; }
#endif

int tsr_feature(const char *name)
{
    if (!name) return 0;
    if (strcmp(name, "repro_f32") == 0) return 1;        /* tsr_repro_* (csrc/repro/GOLDENS.md) */
    if (strcmp(name, "gemm_canonical") == 0) return 1;   /* tsr_gemm / tsr_axpy: explicit round-once FMA */
    if (strcmp(name, "lapack") == 0) {
#ifdef TSR_NO_LAPACK
        return 0;
#else
        return 1;                                         /* linalg.* through LAPACKE (np_linalg.c) */
#endif
    }
    if (strcmp(name, "openblas_threads") == 0) { tsr_blas_resolve(); return g_blas_set != NULL; }
    if (strcmp(name, "openmp") == 0) return tsr_openmp();
    if (strcmp(name, "fpenv_check") == 0) return 1;     /* fpenv.c: tsr_fpenv_check / tsr_fpenv_check_thread */
    if (strcmp(name, "fp_flush_profile") == 0) return 1; /* fpenv.c: flush scope + subnormal/NaN helpers */
    if (strcmp(name, "ownership_check") == 0) return tsr_ownership_check();   /* -DTSR_OWNERSHIP_CHECK build (ADR 0015) */
    if (strcmp(name, "nodelete") == 0) return tsr_linked_nodelete();   /* linked -z nodelete (ADR 0014) */
    return 0;
}

/* n*elsize + pad bytes (tsr_bytes_n, internal.h), counted like tsr_alloc; NULL on a negative size or overflow */
void *tsr_alloc_n(int64_t n, int64_t elsize, int64_t pad)
{
    const int64_t bytes = tsr_bytes_n(n, elsize, pad);
    return bytes < 0 ? NULL : tsr_alloc(bytes);
}

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

/* ---------------------------------------------------------------- ownership check (ADR 0015)
   Built with -DTSR_OWNERSHIP_CHECK (make sanitize, the fuzz builds, the ASan CI jobs; never the release build),
   every tsr_alloc block carries a 64-byte header in front of the pointer handed out, and every live block is
   recorded with its size. That turns the golden rule (the layer that allocates memory owns its deallocation)
   into hard failures:
     - free()/efree()/_aligned_free() of a tsr_alloc pointer is an invalid free (the pointer is 64 bytes into the
       allocator's block): ASan and the C library abort;
     - tsr_free() of a pointer tsr_alloc did not return, or a second tsr_free() of the same pointer: abort;
     - tsr_free() with a byte count other than the one allocated (the budget would drift): abort.
   The release build compiles none of this. */
#ifdef TSR_OWNERSHIP_CHECK
#define TSR_OWN_HDR 64
#define TSR_OWN_MAGIC 0x6b636f6c6c615354ull   /* "TSalloc k" */
#define TSR_OWN_DEAD 0x6565726664616544ull    /* "Deadfree" */
typedef struct { uint64_t magic; int64_t bytes; } tsr_own_hdr;

static struct { void **key; int64_t *bytes; size_t cap, used; } g_own;
#ifdef _WIN32
static SRWLOCK g_own_lock = SRWLOCK_INIT;
#  define OWN_LOCK() AcquireSRWLockExclusive(&g_own_lock)
#  define OWN_UNLOCK() ReleaseSRWLockExclusive(&g_own_lock)
#else
static pthread_mutex_t g_own_lock = PTHREAD_MUTEX_INITIALIZER;
#  define OWN_LOCK() pthread_mutex_lock(&g_own_lock)
#  define OWN_UNLOCK() pthread_mutex_unlock(&g_own_lock)
#endif
#define OWN_TOMB ((void *)1)

static size_t own_slot(void *p, size_t cap) { return (size_t)(((uintptr_t)p >> 6) * 0x9E3779B97F4A7C15ull) & (cap - 1); }

static void own_fail(const char *what, const void *p, int64_t bytes, int64_t recorded)
{
    fprintf(stderr, "libtessero ownership violation: %s (pointer %p, tsr_free bytes %lld, allocated bytes %lld). "
                    "Memory from tsr_alloc must be released with tsr_free and the allocated size, by the layer that "
                    "allocated it (ADR 0015).\n", what, p, (long long)bytes, (long long)recorded);
    fflush(stderr);
    abort();
}

static void own_grow(void)   /* caller holds the lock; raw libc memory: the registry is not tsr_alloc's own client */
{
    const size_t cap = g_own.cap ? g_own.cap * 2 : 4096;
    void **key = calloc(cap, sizeof *key);
    int64_t *bytes = calloc(cap, sizeof *bytes);
    if (key == NULL || bytes == NULL) own_fail("ownership registry out of memory", NULL, 0, 0);
    for (size_t i = 0; i < g_own.cap; i++)
        if (g_own.key[i] != NULL && g_own.key[i] != OWN_TOMB) {
            size_t j = own_slot(g_own.key[i], cap);
            while (key[j] != NULL) j = (j + 1) & (cap - 1);
            key[j] = g_own.key[i]; bytes[j] = g_own.bytes[i];
        }
    free(g_own.key); free(g_own.bytes);
    g_own.key = key; g_own.bytes = bytes; g_own.cap = cap;
    size_t used = 0;
    for (size_t i = 0; i < cap; i++) used += key[i] != NULL;
    g_own.used = used;
}

static void own_record(void *p, int64_t bytes)
{
    OWN_LOCK();
    if ((g_own.used + 1) * 2 > g_own.cap) own_grow();
    size_t i = own_slot(p, g_own.cap);
    while (g_own.key[i] != NULL && g_own.key[i] != OWN_TOMB) i = (i + 1) & (g_own.cap - 1);
    if (g_own.key[i] == NULL) g_own.used++;
    g_own.key[i] = p; g_own.bytes[i] = bytes;
    OWN_UNLOCK();
}

static void own_release(void *p, int64_t bytes)
{
    OWN_LOCK();
    size_t i = g_own.cap ? own_slot(p, g_own.cap) : 0;
    if (g_own.cap)
        while (g_own.key[i] != NULL && g_own.key[i] != p) i = (i + 1) & (g_own.cap - 1);
    if (g_own.cap == 0 || g_own.key[i] != p) {
        OWN_UNLOCK();
        own_fail("tsr_free of a pointer that tsr_alloc did not return, or that was already freed", p, bytes, -1);
    }
    const int64_t recorded = g_own.bytes[i];
    if (recorded != bytes) {
        OWN_UNLOCK();
        own_fail("tsr_free with a byte count other than the allocated size", p, bytes, recorded);
    }
    g_own.key[i] = OWN_TOMB;
    OWN_UNLOCK();
}

int tsr_ownership_check(void) { return 1; }
#else
int tsr_ownership_check(void) { return 0; }
#endif

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
#ifdef TSR_OWNERSHIP_CHECK
    char *base = aligned(bytes + TSR_OWN_HDR);
    void *p = base == NULL ? NULL : base + TSR_OWN_HDR;
    if (p != NULL) {
        tsr_own_hdr h = {TSR_OWN_MAGIC, bytes};
        memcpy(base, &h, sizeof h);
        own_record(p, bytes);
    }
#else
    void *p = aligned(bytes);
#endif
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
#ifdef TSR_OWNERSHIP_CHECK
    own_release(p, bytes);
    char *base = (char *)p - TSR_OWN_HDR;
    const tsr_own_hdr dead = {TSR_OWN_DEAD, bytes};
    memcpy(base, &dead, sizeof dead);
    p = base;
#endif
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
