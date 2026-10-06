/*
 * Memory-mapped files and the .npy header format, shared by both bindings.
 *
 *   tsr_mmap_plan      pure arithmetic: validates a request (dtype, shape,
 *                      offset, mode) against a file size and computes the
 *                      page-aligned window to map. No I/O, so it is a fuzz
 *                      target (tests/fuzz/fuzz_mmap_plan.c).
 *   tsr_mmap_open      open + fstat + plan + extend + mmap/MapViewOfFile.
 *   tsr_mmap_flush     msync (MS_ASYNC, or MS_SYNC + fsync) / FlushViewOfFile.
 *   tsr_mmap_close     unmap and close; the handle is freed.
 *   tsr_npy_header     strict parser for the .npy preamble (v1, v2, v3).
 *   tsr_npy_write_header
 *
 * Modes follow numpy.memmap: 0 'r' read-only (PROT_READ), 1 'r+' read/write
 * an existing file (extended when shorter), 2 'w+' create or truncate,
 * 3 'c' copy-on-write (MAP_PRIVATE: the file never changes).
 *
 * Mapped bytes are not counted against the allocation budget (they are file
 * backed; the OS can drop clean pages at any time). tsr_mmap_bytes() reports
 * the process total. On failure the functions return a negative code and
 * tsr_mmap_error() describes it (per thread).
 *
 * A file truncated by another process while it is mapped raises SIGBUS on
 * access (POSIX) or an access violation (Windows); that is inherent to
 * memory mapping and documented in the guide.
 */
#include "internal.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#  define TSR_TLS __declspec(thread)
#else
#  include <sys/mman.h>
#  include <unistd.h>
#  define TSR_TLS _Thread_local
#endif

#ifndef O_CLOEXEC
#  define O_CLOEXEC 0
#endif
#ifndef O_BINARY
#  define O_BINARY 0
#endif

#if defined(_MSC_VER)
static volatile LONG64 g_mapped = 0;
#  define MAPPED_ADD(n) InterlockedExchangeAdd64(&g_mapped, (LONG64)(n))
#  define MAPPED_GET() InterlockedCompareExchange64(&g_mapped, 0, 0)
#else
static int64_t g_mapped = 0;
#  define MAPPED_ADD(n) __atomic_fetch_add(&g_mapped, (int64_t)(n), __ATOMIC_RELAXED)
#  define MAPPED_GET() __atomic_load_n(&g_mapped, __ATOMIC_RELAXED)
#endif

static TSR_TLS char g_err[512];

static int fail(int code, const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_err, sizeof(g_err), fmt, ap);
    va_end(ap);
    return code;
}

const char *tsr_mmap_error(void) { return g_err; }
int64_t tsr_mmap_bytes(void) { return (int64_t)MAPPED_GET(); }

int64_t tsr_mmap_granularity(void)
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return (int64_t)si.dwAllocationGranularity;   /* MapViewOfFile offsets must be multiples of this */
#else
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (int64_t)p : 4096;
#endif
}

/* ================================================================ planning */

int tsr_mmap_plan(int dtype, int mode, int32_t *ndim, int64_t *shape, int64_t offset, int64_t file_size,
                  int64_t granularity, int64_t *bytes, int64_t *map_start, int64_t *map_length, int64_t *new_size)
{
    g_err[0] = '\0';
    if (!ndim || !shape || !bytes || !map_start || !map_length || !new_size) return fail(TSR_EARG, "null argument");
    if (dtype < 0 || dtype >= TSR_NDTYPES) return fail(TSR_EARG, "unknown dtype %d", dtype);
    if (mode < 0 || mode > 3) return fail(TSR_EARG, "mode must be 0 (r), 1 (r+), 2 (w+) or 3 (c)");
    if (granularity <= 0 || (granularity & (granularity - 1)) != 0) return fail(TSR_EARG, "granularity must be a power of two");
    if (file_size < 0) return fail(TSR_EARG, "negative file size");
    const int64_t item = tsr_itemsize(dtype);
    if (offset < 0 || offset % item != 0)
        return fail(TSR_EARG, "offset must be a non-negative multiple of the item size (%lld bytes)", (long long)item);
    if (mode == 2) file_size = 0;                          /* w+ truncates */

    if (*ndim < 0) {                                       /* 1-D from the file size */
        if (mode == 2) return fail(TSR_EARG, "a shape is required with mode 'w+'");
        const int64_t avail = file_size - offset;
        if (avail <= 0 || avail % item != 0)
            return fail(TSR_ESHAPE, "%lld bytes after offset %lld are not a whole number of %lld-byte items; pass a shape",
                        (long long)(avail < 0 ? 0 : avail), (long long)offset, (long long)item);
        *ndim = 1;
        shape[0] = avail / item;
    }
    if (*ndim > TSR_MAXDIM) return fail(TSR_EDIM, "more than %d dimensions", TSR_MAXDIM);

    int64_t count = 1;
    for (int d = 0; d < *ndim; d++) {
        if (shape[d] < 0) return fail(TSR_EARG, "negative dimension");
        if (shape[d] != 0 && count > INT64_MAX / shape[d]) return fail(TSR_ENOMEM, "the requested shape is too large to map");
        count *= shape[d];
    }
    if (count == 0) return fail(TSR_ESHAPE, "cannot map an empty array (a dimension is 0)");
    if (count > (INT64_MAX - offset) / item) return fail(TSR_ENOMEM, "the requested shape is too large to map");
    const int64_t need = offset + count * item;
    if ((uint64_t)need > (uint64_t)SIZE_MAX) return fail(TSR_ENOMEM, "the requested shape is too large to map");

    *new_size = -1;
    if (file_size < need) {
        if (mode == 0 || mode == 3)
            return fail(TSR_ESHAPE, "the file has %lld bytes but offset %lld + shape needs %lld",
                        (long long)file_size, (long long)offset, (long long)need);
        *new_size = need;
    }
    *bytes = count * item;
    *map_start = offset - offset % granularity;
    *map_length = need - *map_start;
    return TSR_OK;
}

/* ================================================================ mapping */

typedef struct tsr_mmap {
    void *addr;
    int64_t length;
    int fd;
    int mode;
#ifdef _WIN32
    HANDLE mapping;
#endif
} tsr_mmap;

static int os_fail(const char *what, const char *path)
{
#ifdef _WIN32
    return fail(TSR_EIO, "%s '%s' failed (Windows error %lu)", what, path, (unsigned long)GetLastError());
#else
    return fail(TSR_EIO, "%s '%s' failed: %s", what, path, strerror(errno));
#endif
}

int tsr_mmap_open(const char *path, int mode, int dtype, int32_t *ndim, int64_t *shape, int64_t offset,
                  void **handle, void **data)
{
    g_err[0] = '\0';
    if (!path || !*path || !handle || !data) return fail(TSR_EARG, "empty path");
    if (strstr(path, "://")) return fail(TSR_EARG, "stream wrappers cannot be memory-mapped: '%s'", path);
    if (mode < 0 || mode > 3) return fail(TSR_EARG, "mode must be 0 (r), 1 (r+), 2 (w+) or 3 (c)");
    *handle = NULL;
    *data = NULL;

    int flags = O_CLOEXEC | O_BINARY;
    if (mode == 0 || mode == 3) flags |= O_RDONLY;
    else if (mode == 1) flags |= O_RDWR;
    else flags |= O_RDWR | O_CREAT | O_TRUNC;
#ifdef _WIN32
    const int fd = _open(path, flags, _S_IREAD | _S_IWRITE);
#else
    const int fd = open(path, flags, 0666);
#endif
    if (fd < 0) return os_fail("open", path);

#ifdef _WIN32
    struct _stat64 st;
    if (_fstat64(fd, &st) != 0) { const int r = os_fail("stat", path); _close(fd); return r; }
#else
    struct stat st;
    if (fstat(fd, &st) != 0) { const int r = os_fail("stat", path); close(fd); return r; }
    if (!S_ISREG(st.st_mode)) { close(fd); return fail(TSR_EARG, "'%s' is not a regular file", path); }
#endif

    int64_t bytes, start, length, new_size;
    int rc = tsr_mmap_plan(dtype, mode, ndim, shape, offset, (int64_t)st.st_size, tsr_mmap_granularity(),
                           &bytes, &start, &length, &new_size);
    if (rc != TSR_OK) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return rc;
    }
    if (new_size >= 0) {
#ifdef _WIN32
        if (_chsize_s(fd, new_size) != 0) { rc = os_fail("resize", path); _close(fd); return rc; }
#else
        if (ftruncate(fd, (off_t)new_size) != 0) { rc = os_fail("resize", path); close(fd); return rc; }
#endif
    }

    tsr_mmap *m = calloc(1, sizeof(tsr_mmap));
    if (!m) {
#ifdef _WIN32
        _close(fd);
#else
        close(fd);
#endif
        return fail(TSR_ENOMEM, "out of memory");
    }
#ifdef _WIN32
    HANDLE fh = (HANDLE)_get_osfhandle(fd);
    const DWORD protect = mode == 0 ? PAGE_READONLY : mode == 3 ? PAGE_WRITECOPY : PAGE_READWRITE;
    HANDLE mh = CreateFileMappingW(fh, NULL, protect, 0, 0, NULL);
    if (mh == NULL) { rc = os_fail("CreateFileMapping", path); free(m); _close(fd); return rc; }
    const DWORD access = mode == 0 ? FILE_MAP_READ : mode == 3 ? FILE_MAP_COPY : FILE_MAP_WRITE;
    void *addr = MapViewOfFile(mh, access, (DWORD)((uint64_t)start >> 32), (DWORD)((uint64_t)start & 0xffffffffu), (SIZE_T)length);
    if (addr == NULL) { rc = os_fail("MapViewOfFile", path); CloseHandle(mh); free(m); _close(fd); return rc; }
    m->mapping = mh;
#else
    const int prot = mode == 0 ? PROT_READ : (PROT_READ | PROT_WRITE);
    const int mflags = mode == 3 ? MAP_PRIVATE : MAP_SHARED;
    void *addr = mmap(NULL, (size_t)length, prot, mflags, fd, (off_t)start);
    if (addr == MAP_FAILED) { rc = os_fail("mmap", path); free(m); close(fd); return rc; }
#endif
    m->addr = addr;
    m->length = length;
    m->fd = fd;
    m->mode = mode;
    MAPPED_ADD(length);
    *handle = m;
    *data = (char *)addr + (offset - start);
    return TSR_OK;
}

int tsr_mmap_flush(void *handle, int sync)
{
    tsr_mmap *m = handle;
    if (!m) return fail(TSR_EARG, "null handle");
    if (m->mode == 0 || m->mode == 3) return TSR_OK;      /* nothing to write back (as numpy) */
#ifdef _WIN32
    if (!FlushViewOfFile(m->addr, (SIZE_T)m->length) || (sync && !FlushFileBuffers((HANDLE)_get_osfhandle(m->fd))))
        return os_fail("flush", "mapping");
#else
    if (msync(m->addr, (size_t)m->length, sync ? MS_SYNC : MS_ASYNC) != 0 || (sync && fsync(m->fd) != 0))
        return os_fail("flush", "mapping");
#endif
    return TSR_OK;
}

int64_t tsr_mmap_length(void *handle) { return handle ? ((tsr_mmap *)handle)->length : 0; }

void tsr_mmap_close(void *handle)
{
    tsr_mmap *m = handle;
    if (!m) return;
#ifdef _WIN32
    UnmapViewOfFile(m->addr);
    CloseHandle(m->mapping);
    _close(m->fd);
#else
    munmap(m->addr, (size_t)m->length);
    close(m->fd);
#endif
    MAPPED_ADD(-m->length);
    free(m);
}

/* ================================================================ .npy header */

static const char NPY_MAGIC[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};

static const char *skip_ws(const char *p, const char *end)
{
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* a quoted Python string literal ('x' or "x") without escapes; returns the position after it */
static const char *parse_str(const char *p, const char *end, const char **s, int64_t *n)
{
    if (p >= end || (*p != '\'' && *p != '"')) return NULL;
    const char q = *p++;
    const char *b = p;
    while (p < end && *p != q) {
        if (*p == '\\' || *p == '\n') return NULL;
        p++;
    }
    if (p >= end) return NULL;
    *s = b;
    *n = p - b;
    return p + 1;
}

static int descr_dtype(const char *s, int64_t n)
{
    /* byte order: '<' little, '|' not applicable, '=' native; '>' (big-endian) is not supported */
    static const struct { const char *code; int dt; int bytes; } T[] = {
        {"f8", TSR_F64, 8}, {"f4", TSR_F32, 4}, {"i8", TSR_I64, 8}, {"i4", TSR_I32, 4},
        {"u1", TSR_U8, 1}, {"b1", TSR_BOOL, 1}, {"c16", TSR_C128, 16}, {"?", TSR_BOOL, 1},
    };
    if (n < 2) return TSR_EARG;
    const char order = s[0];
    if (order != '<' && order != '|' && order != '=' && order != '>') return TSR_EARG;
    for (size_t i = 0; i < sizeof(T) / sizeof(T[0]); i++) {
        const int64_t len = (int64_t)strlen(T[i].code);
        if (n - 1 == len && memcmp(s + 1, T[i].code, (size_t)len) == 0) {
            if (order == '>' && T[i].bytes > 1) return TSR_ETYPE;
#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            if (order != '|' && T[i].bytes > 1) return TSR_ETYPE;   /* libtessero assumes little-endian data */
#endif
            return T[i].dt;
        }
    }
    return TSR_ETYPE;
}

int tsr_npy_header(const char *buf, int64_t len, int *dtype, int32_t *ndim, int64_t *shape, int *fortran, int64_t *data_offset)
{
    if (!buf || !dtype || !ndim || !shape || !fortran || !data_offset || len < 10) return TSR_EARG;
    if (memcmp(buf, NPY_MAGIC, 6) != 0) return TSR_EARG;
    const unsigned char major = (unsigned char)buf[6];
    int64_t hlen, start;
    if (major == 1) {
        hlen = (int64_t)(unsigned char)buf[8] | ((int64_t)(unsigned char)buf[9] << 8);
        start = 10;
    } else if (major == 2 || major == 3) {
        if (len < 12) return TSR_EARG;
        hlen = (int64_t)(unsigned char)buf[8] | ((int64_t)(unsigned char)buf[9] << 8)
             | ((int64_t)(unsigned char)buf[10] << 16) | ((int64_t)(unsigned char)buf[11] << 24);
        start = 12;
    } else {
        return TSR_ETYPE;
    }
    if (hlen <= 0 || hlen > len - start) return TSR_EARG;

    const char *p = buf + start, *end = buf + start + hlen;
    int seen_descr = 0, seen_fortran = 0, seen_shape = 0;
    p = skip_ws(p, end);
    if (p >= end || *p++ != '{') return TSR_EARG;
    for (;;) {
        p = skip_ws(p, end);
        if (p < end && *p == '}') { p++; break; }
        const char *key;
        int64_t klen;
        p = parse_str(p, end, &key, &klen);
        if (!p) return TSR_EARG;
        p = skip_ws(p, end);
        if (p >= end || *p++ != ':') return TSR_EARG;
        p = skip_ws(p, end);
        if (klen == 5 && memcmp(key, "descr", 5) == 0) {
            if (seen_descr++) return TSR_EARG;
            const char *s;
            int64_t n;
            if (p < end && (*p == '[' || *p == '{')) return TSR_ETYPE;   /* structured dtypes are not supported */
            p = parse_str(p, end, &s, &n);
            if (!p) return TSR_EARG;
            const int dt = descr_dtype(s, n);
            if (dt < 0) return dt == TSR_EARG ? TSR_ETYPE : dt;
            *dtype = dt;
        } else if (klen == 13 && memcmp(key, "fortran_order", 13) == 0) {
            if (seen_fortran++) return TSR_EARG;
            if (end - p >= 4 && memcmp(p, "True", 4) == 0) { *fortran = 1; p += 4; }
            else if (end - p >= 5 && memcmp(p, "False", 5) == 0) { *fortran = 0; p += 5; }
            else return TSR_EARG;
        } else if (klen == 5 && memcmp(key, "shape", 5) == 0) {
            if (seen_shape++) return TSR_EARG;
            if (p >= end || *p++ != '(') return TSR_EARG;
            int32_t nd = 0;
            int64_t count = 1;
            for (;;) {
                p = skip_ws(p, end);
                if (p < end && *p == ')') { p++; break; }
                if (p >= end || *p < '0' || *p > '9') return TSR_EARG;
                int64_t v = 0;
                while (p < end && *p >= '0' && *p <= '9') {
                    if (v > (INT64_MAX - 9) / 10) return TSR_ESHAPE;
                    v = v * 10 + (*p++ - '0');
                }
                if (p < end && *p == 'L') p++;       /* Python 2 longs */
                if (nd >= TSR_MAXDIM) return TSR_EDIM;
                if (v != 0 && count > INT64_MAX / v) return TSR_ESHAPE;
                count *= v;
                shape[nd++] = v;
                p = skip_ws(p, end);
                if (p < end && *p == ',') { p++; continue; }
                if (p < end && *p == ')') { p++; break; }
                return TSR_EARG;
            }
            *ndim = nd;
        } else {
            return TSR_EARG;                         /* unknown key */
        }
        p = skip_ws(p, end);
        if (p < end && *p == ',') { p++; continue; }
        p = skip_ws(p, end);
        if (p < end && *p == '}') { p++; break; }
        return TSR_EARG;
    }
    p = skip_ws(p, end);
    if (p != end) return TSR_EARG;                   /* only padding may follow the dict */
    if (!seen_descr || !seen_fortran || !seen_shape) return TSR_EARG;
    /* the data must fit an int64 byte count */
    int64_t count = 1;
    for (int d = 0; d < *ndim; d++) {
        if (shape[d] != 0 && count > INT64_MAX / shape[d]) return TSR_ESHAPE;
        count *= shape[d];
    }
    if (count > INT64_MAX / tsr_itemsize(*dtype)) return TSR_ESHAPE;
    *data_offset = start + hlen;
    return TSR_OK;
}

int64_t tsr_npy_write_header(int dtype, int32_t ndim, const int64_t *shape, int fortran, char *buf, int64_t cap)
{
    static const char *DESCR[TSR_NDTYPES] = {"<f8", "<f4", "<i8", "<i4", "|u1", "|b1", "<c16"};
    if (dtype < 0 || dtype >= TSR_NDTYPES || ndim < 0 || ndim > TSR_MAXDIM || (ndim > 0 && !shape)) return TSR_EARG;
    char dict[1024];
    int n = snprintf(dict, sizeof(dict), "{'descr': '%s', 'fortran_order': %s, 'shape': (", DESCR[dtype], fortran ? "True" : "False");
    for (int d = 0; d < ndim; d++) {
        if (shape[d] < 0) return TSR_EARG;
        n += snprintf(dict + n, sizeof(dict) - (size_t)n, d ? ", %lld" : "%lld", (long long)shape[d]);
    }
    n += snprintf(dict + n, sizeof(dict) - (size_t)n, "%s), }", ndim == 1 ? "," : "");
    int major = 1;
    int64_t prefix = 10;
    if (n + prefix + 1 > 65535) { major = 2; prefix = 12; }
    const int64_t total = prefix + n + 1;
    const int64_t pad = (64 - total % 64) % 64;
    const int64_t need = total + pad;
    if (!buf || cap < need) return need;             /* size query */
    memcpy(buf, NPY_MAGIC, 6);
    buf[6] = (char)major;
    buf[7] = 0;
    const int64_t hlen = n + pad + 1;
    buf[8] = (char)(hlen & 0xff);
    buf[9] = (char)((hlen >> 8) & 0xff);
    if (major == 2) { buf[10] = (char)((hlen >> 16) & 0xff); buf[11] = (char)((hlen >> 24) & 0xff); }
    memcpy(buf + prefix, dict, (size_t)n);
    memset(buf + prefix + n, ' ', (size_t)pad);
    buf[need - 1] = '\n';
    return need;
}
