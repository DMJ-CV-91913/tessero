/*
 * tessero - native Zend extension for the Tessero numeric platform.
 *
 * Classes (namespace Tessero\Ext):
 *   NDArray   n-d typed array in native memory; operators, slicing, foreach,
 *             JSON, serialize, reductions, matmul, sorting
 *   Engine    process settings (threads, memory budget) and the native
 *             solvers: linprog, milp, MDP, FFT, random numbers
 *   Operand   base class that gives userland classes (the FFI Tessero\NDArray)
 *             + - * / ** % by dispatching to add()/sub()/... methods
 *   Math      universal functions (ufuncs): element-wise functions over any
 *             shape with broadcasting, type resolution and an optional
 *             output array (tessero_ufunc.c)
 *
 * NDArray::memmap() maps a file into an array (tessero_memmap.c): the data
 * pointer is the address returned by mmap()/MapViewOfFile(), so arrays larger
 * than RAM are paged in on demand and writes go back to the file.
 *   Exception, ShapeException, IndexException, DTypeException, MemoryException
 *
 * All numeric work is done by libtessero, compiled into this module (see
 * libtessero/, kept in sync with csrc/ by tools/sync-ext.sh).
 */
#ifndef PHP_TESSERO_H
#define PHP_TESSERO_H

#include "php.h"
/* tessero.h is FFI-parseable, so it cannot carry an include guard: guard it here */
#ifndef TSR_TESSERO_H_INCLUDED
#  define TSR_TESSERO_H_INCLUDED
#  include "libtessero/include/tessero.h"
#endif

extern zend_module_entry tessero_module_entry;
#define phpext_tessero_ptr &tessero_module_entry

#define PHP_TESSERO_VERSION "0.2.0"

#if defined(ZTS) && defined(COMPILE_DL_TESSERO)
ZEND_TSRMLS_CACHE_EXTERN()
#endif

/* Per-thread (ZTS) / per-process (NTS) settings; libtessero itself is process-wide. */
ZEND_BEGIN_MODULE_GLOBALS(tessero)
    zend_long threads;          /* applied to libtessero at request start (tessero.threads) */
    zend_long memory_budget;    /* bytes, 0 = unlimited (tessero.memory_budget) */
    double epsilon;             /* default tolerance for iterative solvers (tessero.epsilon) */
ZEND_END_MODULE_GLOBALS(tessero)

ZEND_EXTERN_MODULE_GLOBALS(tessero)
#define TESSERO_G(v) ZEND_MODULE_GLOBALS_ACCESSOR(tessero, v)

/* ---------------------------------------------------------------- NDArray object */

/* Array flags (tsr_obj.flags); views inherit them from their root owner. */
#define TSR_F_MMAP      0x1u   /* data is a file mapping, released by tsr_memmap_release() */
#define TSR_F_READONLY  0x2u   /* writes throw (memmap mode 'r') */

/* File mapping owned by a memmap root array (the mapping itself is libtessero's tsr_mmap_*). */
typedef struct {
    void *handle;             /* tsr_mmap_open() handle: unmapped + closed by tsr_mmap_close() */
    zend_string *path;        /* resolved path, for messages and filename() */
    char mode;                /* 'r', '+' (r+), 'w' (w+), 'c' (copy-on-write) */
} tsr_map;

typedef struct {
    tsr_array a;              /* shape, byte strides, offset, dtype; a.data = start of the block */
    void *block;              /* owned tsr_alloc() allocation (NULL for views and memmaps) */
    int64_t block_bytes;
    zend_object *base;        /* root owner of the block when this is a view (reference held) */
    uint32_t flags;           /* TSR_F_* (meaningful on the root owner) */
    tsr_map *map;             /* memmap root only: the mapping to release in free_obj */
    zend_object std;          /* must be last */
} tsr_obj;

static inline tsr_obj *tsr_obj_from(zend_object *o)
{
    return (tsr_obj *)((char *)o - XtOffsetOf(tsr_obj, std));
}
#define Z_TSR_P(zv) tsr_obj_from(Z_OBJ_P(zv))
#define Z_TSR(zv) tsr_obj_from(Z_OBJ(zv))

extern zend_class_entry *tsr_ce_ndarray, *tsr_ce_engine, *tsr_ce_operand, *tsr_ce_math;
extern zend_class_entry *tsr_ce_exception, *tsr_ce_shape_exception, *tsr_ce_index_exception,
    *tsr_ce_dtype_exception, *tsr_ce_memory_exception;

/* tessero_ndarray.c */
void tsr_register_ndarray(void);
int tsr_ndarray_new(zval *rv, int dtype, int32_t ndim, const int64_t *shape, int zero);   /* SUCCESS/FAILURE (exception set) */
/* wrap a tsr_alloc() block of `bytes` (a kernel routine result): the new array owns and frees it */
int tsr_ndarray_adopt(zval *rv, int dtype, int32_t ndim, const int64_t *shape, void *block, int64_t bytes);
int tsr_ndarray_from_zval(zval *value, int dtype_hint, zval *out);                         /* array/scalar/NDArray -> NDArray (new ref) */
int tsr_ndarray_contiguous_as(zval *arr, int dtype, zval *out);                             /* C-contiguous copy/cast when needed */
int tsr_dtype_from_name(const char *name, size_t len);
const char *tsr_dtype_name(int dtype);
void tsr_throw_rc(int rc, const char *what);

/* The root owner of an array's memory (itself when it is not a view). */
static inline tsr_obj *tsr_root(tsr_obj *o) { return o->base ? tsr_obj_from(o->base) : o; }
/* SUCCESS, or FAILURE with an exception when the array is read-only (memmap mode 'r'). */
int tsr_require_writable(tsr_obj *o, const char *what);

/* Element-wise cores shared by NDArray methods, operators and Math (ufuncs).
 * `dst` (nullable) is an output array: when its shape and dtype match the
 * result the kernel writes straight into it, otherwise the result is cast
 * into it (same-kind or safer casts only). rv receives the result (dst when given). */
int tsr_binary_into(int op, zval *left, zval *right, tsr_obj *dst, zval *rv);
int tsr_unary_into(int op, zval *src, tsr_obj *dst, zval *rv);
int tsr_operand(zval *z, int scalar_dtype, zval *out);          /* NDArray / PHP array / scalar -> NDArray */
int tsr_result_dtype(zval *left, zval *right);                   /* NumPy 2 promotion incl. weak PHP scalars */
int tsr_cast_into(const tsr_array *src, tsr_obj *dst, const char *what);   /* broadcast + same-kind cast copy */
int tsr_can_cast_same_kind(int from, int to);
void tsr_shape_str(char *buf, size_t cap, const tsr_array *a);

/* tessero_ufunc.c */
void tsr_register_math(void);

/* tessero_fn.c: Special, Stats, Np from the kernel's function registry (ADR 0011) */
extern zend_class_entry *tsr_ce_special, *tsr_ce_stats, *tsr_ce_np, *tsr_ce_distribution, *tsr_ce_generator;
void tsr_register_generator(const zend_function_entry *registry_methods, int n);
uint64_t *tsr_generator_state(zval *obj);
void tsr_register_fn(void);
void tsr_fn_dist_call(int id, int method, zval *args, int nargs, zval *rv);
void tsr_fn_call_routine(const char *name, zval *args, int nargs, zval *rv);
/* tessero_dist.c: Tessero\Ext\Distribution (frozen scipy.stats distributions) */
void tsr_register_dist(void);
void tsr_distribution_new(zval *rv, int id, const char *label, zval *args, int nargs);
void tsr_stats_register_pb(zend_function_entry *fes, int *nfe);

/* tessero_memmap.c */
PHP_METHOD(NDArray, memmap);
PHP_METHOD(NDArray, flush);
PHP_METHOD(NDArray, isMemmap);
PHP_METHOD(NDArray, isReadonly);
PHP_METHOD(NDArray, filename);
PHP_METHOD(NDArray, load);
PHP_METHOD(NDArray, save);
PHP_METHOD(NDArray, openMemmap);
void tsr_memmap_release(tsr_obj *o);             /* munmap + close; called from free_obj */
int64_t tsr_memmap_bytes(void);                  /* bytes currently mapped by this process */

#if PHP_VERSION_ID >= 80400
#  define TSR_STATIC_FE(name, handler, ai) ZEND_RAW_FENTRY(name, handler, ai, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC, NULL, NULL)
#else
#  define TSR_STATIC_FE(name, handler, ai) ZEND_RAW_FENTRY(name, handler, ai, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
#endif

#endif
