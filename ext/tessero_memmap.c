/*
 * NDArray::memmap(): arrays backed by a memory-mapped file.
 *
 *   $a = NDArray::memmap('/data/load.f64', 'r+', [8760, 1200], 'float64');
 *
 * The array's data pointer is the address returned by mmap() (POSIX) or
 * MapViewOfFile() (Windows), so nothing is read up front: the kernel pages
 * data in on access and evicts it under memory pressure, which lets a PHP
 * process work on files larger than RAM. Every operation that works on an
 * NDArray works on a memmap; results of computations are ordinary (heap)
 * arrays, while slices and views keep pointing into the file.
 *
 * Modes follow numpy.memmap:
 *   'r'   read-only; any write throws (the pages are mapped PROT_READ, so a
 *         write that slipped past the checks would fault instead of corrupting)
 *   'r+'  read/write an existing file; extended when shorter than requested
 *   'w+'  create or truncate the file to the requested size (zero filled)
 *   'c'   copy-on-write: writes stay in this process, the file never changes
 *
 * The mapping itself (open, size checks, page alignment, mmap/MapViewOfFile,
 * flush, unmap) is libtessero's tsr_mmap_* (csrc/src/mmap.c), shared with the
 * FFI package; this file adds argument checks, open_basedir, the .npy
 * functions (load, save, openMemmap) and the object lifecycle.
 *
 * Lifecycle: the root memmap object owns the mapping (tsr_obj.map). Views hold
 * a reference to the root, so the mapping lives until the last view is gone;
 * free_obj then calls tsr_memmap_release() -> tsr_mmap_close().
 * flush() writes dirty pages back: MS_ASYNC by default (schedules the
 * write-back and returns immediately), MS_SYNC + fsync with sync: true.
 *
 * Mapped bytes are not counted against tessero.memory_budget: they are file
 * backed, and the kernel can drop clean pages at any time. The process total
 * is reported as Engine::info()['memory_mapped'].
 *
 * Paths go through open_basedir; stream wrappers (php://, http://, phar://)
 * are refused because they have no file descriptor to map.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "main/php_open_temporary_file.h"
#include "main/fopen_wrappers.h"
#include "Zend/zend_exceptions.h"
#include "php_tessero.h"

#include "main/php_streams.h"
#include "ext/standard/file.h"

#include <string.h>

int64_t tsr_memmap_bytes(void) { return tsr_mmap_bytes(); }

/* kernel mode codes */
static int kmode(char mode) { return mode == 'r' ? 0 : mode == '+' ? 1 : mode == 'w' ? 2 : 3; }

static void throw_kernel(int rc)
{
    zend_class_entry *ce = (rc == -6 || rc == -5 || rc == -3) ? tsr_ce_shape_exception : tsr_ce_exception;
    zend_throw_exception_ex(ce, 0, "memmap: %s", tsr_mmap_error());
}

/* Resolve and check a path against open_basedir; refuses stream wrappers. SUCCESS or FAILURE (exception set). */
static int resolve_path(zend_string *filename, uint32_t argnum, char *resolved)
{
    if (ZSTR_LEN(filename) == 0) { zend_argument_value_error(argnum, "cannot be empty"); return FAILURE; }
    if (strstr(ZSTR_VAL(filename), "://") != NULL) {
        zend_argument_value_error(argnum, "must be a local file path (stream wrappers cannot be memory-mapped)");
        return FAILURE;
    }
    if (!expand_filepath(ZSTR_VAL(filename), resolved)) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "memmap: cannot resolve path '%s'", ZSTR_VAL(filename));
        return FAILURE;
    }
    if (php_check_open_basedir_ex(resolved, 0)) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "memmap: '%s' is outside the allowed path(s) (open_basedir)", resolved);
        return FAILURE;
    }
    return SUCCESS;
}

/* Map `resolved` and return a root memmap array in rv. strides NULL = C order, else F order (npy fortran_order). */
static int map_array(zval *rv, const char *resolved, char mode, int dt, int32_t nd_in, const int64_t *shape_in, int64_t offset, int fortran)
{
    int32_t nd = nd_in;
    int64_t shape[32];
    if (nd > 0) memcpy(shape, shape_in, sizeof(int64_t) * (size_t)nd);
    void *handle = NULL, *data = NULL;
    const int rc = tsr_mmap_open(resolved, kmode(mode), dt, &nd, shape, offset, &handle, &data);
    if (rc != 0) { throw_kernel(rc); return FAILURE; }

    object_init_ex(rv, tsr_ce_ndarray);
    tsr_obj *o = Z_TSR_P(rv);
    tsr_array_init(&o->a, dt, nd, shape, data);
    if (fortran) {
        static const int64_t ITEMSZ[] = {8, 4, 8, 4, 1, 1, 16};
        int64_t acc = ITEMSZ[dt];
        for (int d = 0; d < nd; d++) { o->a.strides[d] = acc; acc *= shape[d] > 0 ? shape[d] : 1; }
    }
    o->flags = TSR_F_MMAP | (mode == 'r' ? TSR_F_READONLY : 0);
    tsr_map *m = emalloc(sizeof(tsr_map));
    m->handle = handle;
    m->path = zend_string_init(resolved, strlen(resolved), 0);
    m->mode = mode;
    o->map = m;
    return SUCCESS;
}

/* Parse mode: returns 'r', '+', 'w' or 'c', 0 when invalid. */
static char parse_mode(const zend_string *m)
{
    if (zend_string_equals_literal(m, "r") || zend_string_equals_literal(m, "readonly")) return 'r';
    if (zend_string_equals_literal(m, "r+") || zend_string_equals_literal(m, "readwrite")) return '+';
    if (zend_string_equals_literal(m, "w+") || zend_string_equals_literal(m, "write")) return 'w';
    if (zend_string_equals_literal(m, "c") || zend_string_equals_literal(m, "copyonwrite")) return 'c';
    return 0;
}

PHP_METHOD(NDArray, memmap)
{
    zend_string *filename, *mode_str = NULL, *dtype_str = NULL;
    HashTable *shape_ht = NULL;
    zend_long offset = 0;
    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_PATH_STR(filename)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(mode_str)
        Z_PARAM_ARRAY_HT_OR_NULL(shape_ht)
        Z_PARAM_STR(dtype_str)
        Z_PARAM_LONG(offset)
    ZEND_PARSE_PARAMETERS_END();

    char mode = mode_str ? parse_mode(mode_str) : '+';
    if (!mode) { zend_argument_value_error(2, "must be one of 'r', 'r+', 'w+' or 'c'"); RETURN_THROWS(); }
    int dt = dtype_str ? tsr_dtype_from_name(ZSTR_VAL(dtype_str), ZSTR_LEN(dtype_str)) : 0;
    if (dt < 0) { zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "memmap: unknown dtype '%s'", ZSTR_VAL(dtype_str)); RETURN_THROWS(); }
    static const int64_t ITEM[] = {8, 4, 8, 4, 1, 1, 16};
    const int64_t item = ITEM[dt];
    if (offset < 0 || offset % item != 0) {
        zend_argument_value_error(5, "must be a non-negative multiple of the item size (%lld bytes)", (long long)item);
        RETURN_THROWS();
    }
    /* shape: explicit, or 1-D from the file size (not for 'w+', which creates the file) */
    int64_t shape[32];
    int32_t nd = 0;
    if (shape_ht) {
        zval *zv;
        ZEND_HASH_FOREACH_VAL(shape_ht, zv) {
            if (nd >= 32) { zend_throw_exception(tsr_ce_shape_exception, "memmap: more than 32 dimensions", 0); RETURN_THROWS(); }
            ZVAL_DEREF(zv);
            if (Z_TYPE_P(zv) != IS_LONG || Z_LVAL_P(zv) < 0) {
                zend_argument_value_error(3, "must be a list of non-negative integers");
                RETURN_THROWS();
            }
            shape[nd++] = Z_LVAL_P(zv);
        } ZEND_HASH_FOREACH_END();
    } else if (mode == 'w') {
        zend_argument_value_error(3, "is required with mode 'w+'");
        RETURN_THROWS();
    }

    if (!shape_ht) nd = -1;                         /* 1-D from the file size */

    char resolved[MAXPATHLEN];
    if (resolve_path(filename, 1, resolved) == FAILURE) RETURN_THROWS();
    if (map_array(return_value, resolved, mode, dt, nd, shape, offset, 0) == FAILURE) RETURN_THROWS();
}

void tsr_memmap_release(tsr_obj *o)
{
    tsr_map *m = o->map;
    if (!m) return;
    tsr_mmap_close(m->handle);
    zend_string_release(m->path);
    efree(m);
    o->map = NULL;
    o->a.data = NULL;
}

PHP_METHOD(NDArray, flush)
{
    bool sync = 0;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_BOOL(sync)
    ZEND_PARSE_PARAMETERS_END();
    tsr_obj *root = tsr_root(Z_TSR_P(ZEND_THIS));
    tsr_map *m = root->map;
    /* nothing to write back: heap arrays, read-only maps, copy-on-write maps (as numpy) */
    if (!m || m->mode == 'r' || m->mode == 'c') RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
    if (tsr_mmap_flush(m->handle, sync) != 0) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "memmap: %s (%s)", tsr_mmap_error(), ZSTR_VAL(m->path));
        RETURN_THROWS();
    }
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(NDArray, isMemmap)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_BOOL(tsr_root(Z_TSR_P(ZEND_THIS))->map != NULL);
}

PHP_METHOD(NDArray, isReadonly)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_BOOL((tsr_root(Z_TSR_P(ZEND_THIS))->flags & TSR_F_READONLY) != 0);
}

PHP_METHOD(NDArray, filename)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_map *m = tsr_root(Z_TSR_P(ZEND_THIS))->map;
    if (!m) RETURN_NULL();
    RETURN_STR_COPY(m->path);
}

/* ================================================================ .npy files */

static const int64_t NPY_ITEM[] = {8, 4, 8, 4, 1, 1, 16};

static void throw_npy(int rc, const char *path)
{
    if (rc == -4)
        zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "npy: '%s' has a dtype or byte order Tessero does not support "
                                "(supported: little-endian float64, float32, int64, int32, uint8, bool, complex128)", path);
    else if (rc == -5 || rc == -6)
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "npy: '%s' has a shape Tessero cannot hold", path);
    else
        zend_throw_exception_ex(tsr_ce_exception, 0, "npy: '%s' is not a valid .npy file", path);
}

/* Read the first bytes of a file (enough for any .npy preamble). */
static zend_string *read_head(const char *path, size_t max)
{
    php_stream *st = php_stream_open_wrapper((char *)path, "rb", 0, NULL);
    if (!st) return NULL;
    zend_string *buf = zend_string_alloc(max, 0);
    size_t got = 0;
    while (got < max) {
        ssize_t n = php_stream_read(st, ZSTR_VAL(buf) + got, max - got);
        if (n <= 0) break;
        got += (size_t)n;
    }
    php_stream_close(st);
    ZSTR_LEN(buf) = got;
    ZSTR_VAL(buf)[got] = '\0';
    return buf;
}

/* Map an existing .npy file: header from the file, data mapped in place (mode r, r+ or c). */
static int map_npy(zval *rv, zend_string *filename, char mode)
{
    char resolved[MAXPATHLEN];
    if (resolve_path(filename, 1, resolved) == FAILURE) return FAILURE;
    zend_string *head = read_head(resolved, 65536 + 12);
    if (!head) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "npy: cannot open '%s'", resolved);
        return FAILURE;
    }
    int dt = 0, fortran = 0;
    int32_t nd = 0;
    int64_t shape[32], off = 0;
    const int rc = tsr_npy_header(ZSTR_VAL(head), (int64_t)ZSTR_LEN(head), &dt, &nd, shape, &fortran, &off);
    zend_string_release(head);
    if (rc != 0) { throw_npy(rc, resolved); return FAILURE; }
    if (off % NPY_ITEM[dt] != 0) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "npy: the data in '%s' is not aligned to its item size; load() it instead", resolved);
        return FAILURE;
    }
    int64_t n = 1;
    for (int d = 0; d < nd; d++) n *= shape[d];
    if (n == 0) {                                              /* empty: nothing to map, return an empty heap array */
        return tsr_ndarray_new(rv, dt, nd, shape, 1);
    }
    return map_array(rv, resolved, mode, dt, nd, shape, off, fortran);
}

/* NDArray::load(string $path, ?string $mmapMode = null): arrays from .npy files (numpy.load). */
PHP_METHOD(NDArray, load)
{
    zend_string *path, *mm = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_PATH_STR(path)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(mm)
    ZEND_PARSE_PARAMETERS_END();
    if (mm) {
        const char mode = parse_mode(mm);
        if (!mode || mode == 'w') { zend_argument_value_error(2, "must be null, 'r', 'r+' or 'c' (openMemmap() creates files)"); RETURN_THROWS(); }
        if (map_npy(return_value, path, mode) == FAILURE) RETURN_THROWS();
        return;
    }
    php_stream *st = php_stream_open_wrapper(ZSTR_VAL(path), "rb", 0, NULL);
    if (!st) { zend_throw_exception_ex(tsr_ce_exception, 0, "npy: cannot open '%s'", ZSTR_VAL(path)); RETURN_THROWS(); }
    zend_string *buf = php_stream_copy_to_mem(st, PHP_STREAM_COPY_ALL, 0);
    php_stream_close(st);
    if (!buf) buf = ZSTR_EMPTY_ALLOC();

    int dt = 0, fortran = 0;
    int32_t nd = 0;
    int64_t shape[32], off = 0;
    int rc = tsr_npy_header(ZSTR_VAL(buf), (int64_t)ZSTR_LEN(buf), &dt, &nd, shape, &fortran, &off);
    if (rc != 0) { throw_npy(rc, ZSTR_VAL(path)); zend_string_release(buf); RETURN_THROWS(); }
    int64_t n = 1;
    for (int d = 0; d < nd; d++) n *= shape[d];
    const int64_t bytes = n * NPY_ITEM[dt];
    if ((int64_t)ZSTR_LEN(buf) - off < bytes) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "npy: '%s' is truncated (%lld data bytes, %lld expected)", ZSTR_VAL(path),
                                (long long)((int64_t)ZSTR_LEN(buf) - off), (long long)bytes);
        zend_string_release(buf);
        RETURN_THROWS();
    }
    if (!fortran || nd < 2) {
        if (tsr_ndarray_new(return_value, dt, nd, shape, 0) == FAILURE) { zend_string_release(buf); RETURN_THROWS(); }
        if (bytes) memcpy(Z_TSR_P(return_value)->a.data, ZSTR_VAL(buf) + off, (size_t)bytes);
        zend_string_release(buf);
        return;
    }
    /* Fortran order: the bytes are the C-order array of the reversed shape; copy its transpose */
    int64_t rshape[32];
    for (int d = 0; d < nd; d++) rshape[d] = shape[nd - 1 - d];
    zval R;
    if (tsr_ndarray_new(&R, dt, nd, rshape, 0) == FAILURE) { zend_string_release(buf); RETURN_THROWS(); }
    tsr_array *r = &Z_TSR(R)->a;
    if (bytes) memcpy(r->data, ZSTR_VAL(buf) + off, (size_t)bytes);
    zend_string_release(buf);
    if (tsr_ndarray_new(return_value, dt, nd, shape, 0) == FAILURE) { zval_ptr_dtor(&R); RETURN_THROWS(); }
    int64_t tst[32];
    for (int d = 0; d < nd; d++) tst[d] = r->strides[nd - 1 - d];
    tsr_array *o = &Z_TSR_P(return_value)->a;
    rc = n ? tsr_copy(dt, dt, nd, shape, r->data, tst, o->data, o->strides) : 0;
    zval_ptr_dtor(&R);
    if (rc != 0) { tsr_throw_rc(rc, "load"); RETURN_THROWS(); }
}

/* $a->save(string $path): write a .npy file (numpy.save), C order. */
PHP_METHOD(NDArray, save)
{
    zend_string *path;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_PATH_STR(path)
    ZEND_PARSE_PARAMETERS_END();
    tsr_array *a = &Z_TSR_P(ZEND_THIS)->a;
    char head[4096];
    const int64_t hlen = tsr_npy_write_header(a->dtype, a->ndim, a->shape, 0, head, sizeof(head));
    if (hlen <= 0 || hlen > (int64_t)sizeof(head)) { tsr_throw_rc((int)(hlen <= 0 ? hlen : -1), "save"); RETURN_THROWS(); }

    zval C;
    ZVAL_UNDEF(&C);
    const tsr_array *src = a;
    const int64_t n = tsr_array_size(a);
    if (n > 0 && !tsr_array_is_contiguous(a)) {
        if (tsr_ndarray_new(&C, a->dtype, a->ndim, a->shape, 0) == FAILURE) RETURN_THROWS();
        tsr_array *c = &Z_TSR(C)->a;
        tsr_copy(a->dtype, a->dtype, a->ndim, a->shape, (char *)a->data + a->offset, a->strides, c->data, c->strides);
        src = c;
    }
    php_stream *st = php_stream_open_wrapper(ZSTR_VAL(path), "wb", 0, NULL);
    if (!st) {
        zval_ptr_dtor(&C);
        zend_throw_exception_ex(tsr_ce_exception, 0, "npy: cannot write '%s'", ZSTR_VAL(path));
        RETURN_THROWS();
    }
    const size_t bytes = (size_t)(n * NPY_ITEM[a->dtype]);
    ssize_t w1 = php_stream_write(st, head, (size_t)hlen);
    ssize_t w2 = bytes ? php_stream_write(st, (const char *)src->data + src->offset, bytes) : 0;
    php_stream_close(st);
    zval_ptr_dtor(&C);
    if (w1 != (ssize_t)hlen || w2 != (ssize_t)bytes) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "npy: short write to '%s'", ZSTR_VAL(path));
        RETURN_THROWS();
    }
}

/*
 * NDArray::openMemmap(string $path, string $mode = 'r+', ?array $shape = null, string $dtype = 'float64', bool $fortranOrder = false)
 * A memory-mapped .npy file (numpy.lib.format.open_memmap): 'w+' creates the file with a header for shape
 * and dtype; the other modes read shape and dtype from the file's header.
 */
PHP_METHOD(NDArray, openMemmap)
{
    zend_string *path, *mode_str = NULL, *dtype_str = NULL;
    HashTable *shape_ht = NULL;
    bool fortran = 0;
    ZEND_PARSE_PARAMETERS_START(1, 5)
        Z_PARAM_PATH_STR(path)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR(mode_str)
        Z_PARAM_ARRAY_HT_OR_NULL(shape_ht)
        Z_PARAM_STR(dtype_str)
        Z_PARAM_BOOL(fortran)
    ZEND_PARSE_PARAMETERS_END();
    const char mode = mode_str ? parse_mode(mode_str) : '+';
    if (!mode) { zend_argument_value_error(2, "must be one of 'r', 'r+', 'w+' or 'c'"); RETURN_THROWS(); }
    if (mode != 'w') {
        if (map_npy(return_value, path, mode) == FAILURE) RETURN_THROWS();
        return;
    }
    if (!shape_ht) { zend_argument_value_error(3, "is required with mode 'w+'"); RETURN_THROWS(); }
    const int dt = dtype_str ? tsr_dtype_from_name(ZSTR_VAL(dtype_str), ZSTR_LEN(dtype_str)) : 0;
    if (dt < 0) { zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "openMemmap: unknown dtype '%s'", ZSTR_VAL(dtype_str)); RETURN_THROWS(); }
    int64_t shape[32];
    int32_t nd = 0;
    zval *zv;
    ZEND_HASH_FOREACH_VAL(shape_ht, zv) {
        ZVAL_DEREF(zv);
        if (nd >= 32 || Z_TYPE_P(zv) != IS_LONG || Z_LVAL_P(zv) < 0) {
            zend_argument_value_error(3, "must be a list of at most 32 non-negative integers");
            RETURN_THROWS();
        }
        shape[nd++] = Z_LVAL_P(zv);
    } ZEND_HASH_FOREACH_END();

    char resolved[MAXPATHLEN];
    if (resolve_path(path, 1, resolved) == FAILURE) RETURN_THROWS();
    char head[4096];
    const int64_t hlen = tsr_npy_write_header(dt, nd, shape, fortran, head, sizeof(head));
    if (hlen <= 0 || hlen > (int64_t)sizeof(head)) { tsr_throw_rc(-1, "openMemmap"); RETURN_THROWS(); }
    /* write the header (creating or truncating the file), then map the data after it; r+ extends the file */
    php_stream *st = php_stream_open_wrapper(resolved, "wb", 0, NULL);
    if (!st) { zend_throw_exception_ex(tsr_ce_exception, 0, "npy: cannot write '%s'", resolved); RETURN_THROWS(); }
    const ssize_t w = php_stream_write(st, head, (size_t)hlen);
    php_stream_close(st);
    if (w != (ssize_t)hlen) { zend_throw_exception_ex(tsr_ce_exception, 0, "npy: short write to '%s'", resolved); RETURN_THROWS(); }
    if (map_array(return_value, resolved, '+', dt, nd, shape, hlen, fortran) == FAILURE) RETURN_THROWS();
}
