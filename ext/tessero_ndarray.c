/*
 * Tessero\Ext\NDArray - the native array object.
 *
 * Memory model
 *   A root array owns one 64-byte aligned block from tsr_alloc (counted
 *   against tessero.memory_budget). Views (slices, transposes, reshapes,
 *   row iteration) own nothing: they hold a reference to the root's
 *   zend_object, so the block lives until the last view is destroyed, and
 *   free_obj releases either the block (root) or the reference (view).
 *
 * Layout
 *   tsr_array: data (block start), byte offset, shape, byte strides (may be
 *   0 for broadcast or negative for reversed views), dtype. All loops run in
 *   libtessero; this file only resolves shapes, dtypes and ownership.
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "php.h"
#include "ext/standard/info.h"
#include "ext/json/php_json.h"
#include "zend_exceptions.h"
#include "zend_interfaces.h"
#include "zend_smart_str.h"
#include "main/snprintf.h"
#include "php_tessero.h"
#include "libtessero/src/ops.h"

zend_class_entry *tsr_ce_ndarray;
static zend_object_handlers tsr_handlers;

enum { D_F64 = 0, D_F32 = 1, D_I64 = 2, D_I32 = 3, D_U8 = 4, D_BOOL = 5, D_C128 = 6 };
static const int64_t ITEMSIZE[7] = {8, 4, 8, 4, 1, 1, 16};

/* ================================================================ dtypes */

int tsr_dtype_from_name(const char *n, size_t len)
{
    static const struct { const char *name; int dt; } names[] = {
        {"float64", D_F64}, {"f8", D_F64}, {"double", D_F64}, {"float", D_F64},
        {"float32", D_F32}, {"f4", D_F32}, {"single", D_F32},
        {"int64", D_I64}, {"i8", D_I64}, {"int", D_I64},
        {"int32", D_I32}, {"i4", D_I32},
        {"uint8", D_U8}, {"u1", D_U8},
        {"bool", D_BOOL},
        {"complex128", D_C128}, {"c16", D_C128}, {"complex", D_C128},
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); i++)
        if (strlen(names[i].name) == len && strncasecmp(names[i].name, n, len) == 0) return names[i].dt;
    return -1;
}

const char *tsr_dtype_name(int dt)
{
    static const char *names[7] = {"float64", "float32", "int64", "int32", "uint8", "bool", "complex128"};
    return dt >= 0 && dt < 7 ? names[dt] : "unknown";
}

static int kind(int dt) { return dt == D_BOOL ? 0 : (dt == D_F32 || dt == D_F64) ? 2 : dt == D_C128 ? 3 : 1; }
static int is_int(int dt) { return dt == D_I64 || dt == D_I32 || dt == D_U8; }
static int to_float(int dt) { return (dt == D_F32 || dt == D_F64 || dt == D_C128) ? dt : D_F64; }

/* NumPy 2 promotion (same table as Tessero\DType::promote). */
static int promote(int a, int b)
{
    if (a == b) return a;
    int ka = kind(a), kb = kind(b);
    if (ka == 3 || kb == 3) return D_C128;
    if (ka == 2 || kb == 2) {
        if (ka == 2 && kb == 2) return D_F64;
        int f = ka == 2 ? a : b, o = ka == 2 ? b : a;
        return (f == D_F32 && (o == D_BOOL || o == D_U8)) ? D_F32 : D_F64;
    }
    if (ka == 0) return b;
    if (kb == 0) return a;
    return ITEMSIZE[a] >= ITEMSIZE[b] ? a : b;
}

/* dtype of (array of dtype dt) op (PHP scalar), scalars being "weak" (NEP 50) */
static int with_scalar(int dt, zval *s)
{
    if (Z_TYPE_P(s) == IS_TRUE || Z_TYPE_P(s) == IS_FALSE) return dt;
    if (Z_TYPE_P(s) == IS_DOUBLE) return kind(dt) >= 2 ? dt : D_F64;
    return dt == D_BOOL ? D_I64 : dt;
}

void tsr_throw_rc(int rc, const char *what)
{
    switch (rc) {
    case -2: zend_throw_exception_ex(tsr_ce_index_exception, 0, "%s: index out of bounds", what); break;
    case -3: zend_throw_exception_ex(tsr_ce_memory_exception, 0, "%s: out of memory or over tessero.memory_budget (%lld bytes in use)", what, (long long)tsr_allocated()); break;
    case -4: zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "%s: not supported for this dtype", what); break;
    case -5: zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s: more than 32 dimensions", what); break;
    case -6: zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s: shapes are not compatible", what); break;
    case -8: zend_throw_exception_ex(tsr_ce_exception, 0, "%s: did not converge", what); break;
    default: zend_throw_exception_ex(tsr_ce_exception, 0, "%s: invalid argument (%d)", what, rc); break;
    }
}

static void shape_str(char *buf, size_t cap, const tsr_array *a)
{
    size_t n = (size_t)snprintf(buf, cap, "(");
    for (int d = 0; d < a->ndim && n < cap; d++)
        n += (size_t)snprintf(buf + n, cap - n, d ? ", %lld" : "%lld", (long long)a->shape[d]);
    if (n < cap) snprintf(buf + n, cap - n, a->ndim == 1 ? ",)" : ")");
}

/* ================================================================ object lifecycle */

static zend_object *tsr_create(zend_class_entry *ce)
{
    tsr_obj *o = zend_object_alloc(sizeof(tsr_obj), ce);
    memset(&o->a, 0, sizeof(o->a));
    o->block = NULL;
    o->block_bytes = 0;
    o->base = NULL;
    o->flags = 0;
    o->map = NULL;
    zend_object_std_init(&o->std, ce);
    object_properties_init(&o->std, ce);
    o->std.handlers = &tsr_handlers;
    return &o->std;
}

static void tsr_free_obj(zend_object *obj)
{
    tsr_obj *o = tsr_obj_from(obj);
    if (o->block) {
        tsr_free(o->block, o->block_bytes);
        o->block = NULL;
    }
    if (o->map) {
        /* memmap root: its views hold a reference to this object, so no view can outlive the mapping */
        tsr_memmap_release(o);
    }
    if (o->base) {
        OBJ_RELEASE(o->base);
        o->base = NULL;
    }
    zend_object_std_dtor(obj);
}

static HashTable *tsr_get_gc(zend_object *obj, zval **table, int *n)
{
    tsr_obj *o = tsr_obj_from(obj);
    if (o->base) {
        zend_get_gc_buffer *buf = zend_get_gc_buffer_create();
        zend_get_gc_buffer_add_obj(buf, o->base);
        zend_get_gc_buffer_use(buf, table, n);
    } else {
        *table = NULL;
        *n = 0;
    }
    return zend_std_get_properties(obj);
}

static inline char *elem0(const tsr_array *a) { return (char *)a->data + a->offset; }

int tsr_ndarray_new(zval *rv, int dtype, int32_t ndim, const int64_t *shape, int zero)
{
    if (ndim > 32) { tsr_throw_rc(-5, "array"); return FAILURE; }
    int64_t n = 1;
    for (int d = 0; d < ndim; d++) {
        if (shape[d] < 0) { zend_throw_exception(tsr_ce_shape_exception, "Dimensions must be non-negative", 0); return FAILURE; }
        /* an overflowing element count would allocate a small block for a huge shape: refuse it */
        if (shape[d] != 0 && n > (INT64_MAX / 16) / shape[d]) {
            zend_throw_exception(tsr_ce_memory_exception, "array: the requested shape is too large (element count overflows)", 0);
            return FAILURE;
        }
        n *= shape[d];
    }
    int64_t bytes = n * ITEMSIZE[dtype];
    if (bytes < 64) bytes = 64;
    void *block = zero ? tsr_calloc(bytes) : tsr_alloc(bytes);
    if (!block) { tsr_throw_rc(-3, "allocate"); return FAILURE; }
    object_init_ex(rv, tsr_ce_ndarray);
    tsr_obj *o = Z_TSR_P(rv);
    tsr_array_init(&o->a, dtype, ndim, shape, block);
    o->block = block;
    o->block_bytes = bytes;
    return SUCCESS;
}

int tsr_ndarray_adopt(zval *rv, int dtype, int32_t ndim, const int64_t *shape, void *block, int64_t bytes)
{
    if (ndim > 32 || !block) { tsr_free(block, bytes); tsr_throw_rc(-1, "adopt"); return FAILURE; }
    object_init_ex(rv, tsr_ce_ndarray);
    tsr_obj *o = Z_TSR_P(rv);
    tsr_array_init(&o->a, dtype, ndim, shape, block);
    o->block = block;
    o->block_bytes = bytes;
    return SUCCESS;
}

/* A new object viewing `src`'s memory with metadata `meta`. */
static void make_view(zval *rv, tsr_obj *src, const tsr_array *meta)
{
    object_init_ex(rv, tsr_ce_ndarray);
    tsr_obj *o = Z_TSR_P(rv);
    o->a = *meta;
    zend_object *root = src->base ? src->base : &src->std;
    GC_ADDREF(root);
    o->base = root;
}

static int is_ndarray(zval *z) { return Z_TYPE_P(z) == IS_OBJECT && Z_OBJCE_P(z) == tsr_ce_ndarray; }

/* ================================================================ element <-> zval */

static void elem_to_zval(int dt, const char *p, zval *z)
{
    switch (dt) {
    case D_F64: ZVAL_DOUBLE(z, *(const double *)p); break;
    case D_F32: ZVAL_DOUBLE(z, (double)*(const float *)p); break;
    case D_I64: ZVAL_LONG(z, (zend_long)*(const int64_t *)p); break;
    case D_I32: ZVAL_LONG(z, *(const int32_t *)p); break;
    case D_U8: ZVAL_LONG(z, *(const uint8_t *)p); break;
    case D_BOOL: ZVAL_BOOL(z, *(const uint8_t *)p != 0); break;
    case D_C128:
        array_init_size(z, 2);
        add_next_index_double(z, ((const double *)p)[0]);
        add_next_index_double(z, ((const double *)p)[1]);
        break;
    }
}

/* Store a PHP scalar into an element of dtype dt. Returns FAILURE for non-numeric input. */
static int zval_to_elem(zval *z, int dt, char *p)
{
    double d;
    int64_t l;
    int isd;
    switch (Z_TYPE_P(z)) {
    case IS_LONG: l = Z_LVAL_P(z); d = (double)l; isd = 0; break;
    case IS_DOUBLE: d = Z_DVAL_P(z); l = (int64_t)d; isd = 1; break;
    case IS_TRUE: l = 1; d = 1.0; isd = 0; break;
    case IS_FALSE: l = 0; d = 0.0; isd = 0; break;
    default: return FAILURE;
    }
    switch (dt) {
    case D_F64: *(double *)p = d; break;
    case D_F32: *(float *)p = (float)d; break;
    case D_I64: *(int64_t *)p = isd ? (int64_t)d : l; break;
    case D_I32: *(int32_t *)p = (int32_t)(isd ? (int64_t)d : l); break;
    case D_U8: *(uint8_t *)p = (uint8_t)(isd ? (int64_t)d : l); break;
    case D_BOOL: *(uint8_t *)p = isd ? d != 0.0 : l != 0; break;
    case D_C128: ((double *)p)[0] = d; ((double *)p)[1] = 0.0; break;
    }
    return SUCCESS;
}

/* ================================================================ PHP arrays -> NDArray */

typedef struct { int ndim; int64_t shape[32]; int any_float, all_bool, any_value; } scan_t;

static int scan_shape(zval *v, scan_t *s)
{
    s->ndim = 0;
    zval *cur = v;
    while (Z_TYPE_P(cur) == IS_ARRAY) {
        if (s->ndim >= 32) { tsr_throw_rc(-5, "array"); return FAILURE; }
        HashTable *ht = Z_ARRVAL_P(cur);
        s->shape[s->ndim++] = zend_hash_num_elements(ht);
        if (zend_hash_num_elements(ht) == 0) break;
        zval *first = NULL;
        ZEND_HASH_FOREACH_VAL(ht, first) { break; } ZEND_HASH_FOREACH_END();
        cur = first;
        ZVAL_DEREF(cur);
    }
    return SUCCESS;
}

static int scan_values(zval *v, int depth, scan_t *s)
{
    ZVAL_DEREF(v);
    if (depth == s->ndim) {
        switch (Z_TYPE_P(v)) {
        case IS_DOUBLE: s->any_float = 1; s->all_bool = 0; break;
        case IS_LONG: s->all_bool = 0; break;
        case IS_TRUE: case IS_FALSE: break;
        default:
            zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "Arrays hold numbers or booleans, got %s", zend_zval_type_name(v));
            return FAILURE;
        }
        s->any_value = 1;
        return SUCCESS;
    }
    if (Z_TYPE_P(v) != IS_ARRAY || (int64_t)zend_hash_num_elements(Z_ARRVAL_P(v)) != s->shape[depth]) {
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "Ragged nested array: every row at depth %d must have %lld elements", depth, (long long)s->shape[depth]);
        return FAILURE;
    }
    zval *e;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) {
        if (scan_values(e, depth + 1, s) == FAILURE) return FAILURE;
    } ZEND_HASH_FOREACH_END();
    return SUCCESS;
}

static void fill_values(zval *v, int depth, const scan_t *s, int dt, char **p)
{
    ZVAL_DEREF(v);
    if (depth == s->ndim) {
        zval_to_elem(v, dt, *p);
        *p += ITEMSIZE[dt];
        return;
    }
    zval *e;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) { fill_values(e, depth + 1, s, dt, p); } ZEND_HASH_FOREACH_END();
}

static int ndarray_from_php(zval *value, int dtype, zval *out)
{
    scan_t s;
    memset(&s, 0, sizeof(s));
    s.all_bool = 1;
    if (Z_TYPE_P(value) != IS_ARRAY) {
        s.ndim = 0;
    } else if (scan_shape(value, &s) == FAILURE) {
        return FAILURE;
    }
    if (scan_values(value, 0, &s) == FAILURE) return FAILURE;
    int dt = dtype >= 0 ? dtype : (!s.any_value ? D_F64 : s.any_float ? D_F64 : s.all_bool ? D_BOOL : D_I64);
    if (tsr_ndarray_new(out, dt, s.ndim, s.shape, 0) == FAILURE) return FAILURE;
    char *p = elem0(&Z_TSR_P(out)->a);
    fill_values(value, 0, &s, dt, &p);
    return SUCCESS;
}

/* Copy (with cast) src into a new C-contiguous array of dtype dt. */
static int cast_copy(const tsr_array *src, int dt, zval *out)
{
    if (tsr_ndarray_new(out, dt, src->ndim, src->shape, 0) == FAILURE) return FAILURE;
    tsr_obj *o = Z_TSR_P(out);
    if (tsr_array_size(src) == 0) return SUCCESS;
    int rc = tsr_copy(src->dtype, dt, src->ndim, src->shape, elem0(src), src->strides, elem0(&o->a), o->a.strides);
    if (rc != 0) { zval_ptr_dtor(out); ZVAL_UNDEF(out); tsr_throw_rc(rc, "astype"); return FAILURE; }
    return SUCCESS;
}

static void to_php(const tsr_array *a, int d, const char *p, zval *rv);

/* does a nested PHP array hold NDArray objects (a list of arrays, as numpy.array([a, b]) accepts)? */
static int holds_ndarray(zval *v)
{
    ZVAL_DEREF(v);
    if (Z_TYPE_P(v) != IS_ARRAY) return 0;
    zval *e;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) {
        ZVAL_DEREF(e);
        if (is_ndarray(e) || holds_ndarray(e)) return 1;
    } ZEND_HASH_FOREACH_END();
    return 0;
}

/* a copy of v with every NDArray replaced by its nested PHP array */
static void unwrap_ndarrays(zval *v, zval *rv)
{
    ZVAL_DEREF(v);
    if (is_ndarray(v)) {
        const tsr_array *a = &Z_TSR_P(v)->a;
        to_php(a, 0, (const char *)a->data + a->offset, rv);
        return;
    }
    if (Z_TYPE_P(v) != IS_ARRAY) { ZVAL_COPY(rv, v); return; }
    array_init_size(rv, zend_hash_num_elements(Z_ARRVAL_P(v)));
    zval *e;
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(v), e) {
        zval c;
        unwrap_ndarrays(e, &c);
        add_next_index_zval(rv, &c);
    } ZEND_HASH_FOREACH_END();
}

int tsr_ndarray_from_zval(zval *value, int dtype, zval *out)
{
    ZVAL_DEREF(value);
    if (is_ndarray(value)) {
        tsr_obj *o = Z_TSR_P(value);
        if (dtype < 0 || dtype == o->a.dtype) { ZVAL_COPY(out, value); return SUCCESS; }
        return cast_copy(&o->a, dtype, out);
    }
    if (Z_TYPE_P(value) == IS_ARRAY && holds_ndarray(value)) {
        zval plain;
        unwrap_ndarrays(value, &plain);
        const int r = ndarray_from_php(&plain, dtype, out);
        zval_ptr_dtor(&plain);
        return r;
    }
    return ndarray_from_php(value, dtype, out);
}

int tsr_ndarray_contiguous_as(zval *arr, int dtype, zval *out)
{
    tsr_obj *o = Z_TSR_P(arr);
    if (o->a.dtype == dtype && tsr_array_is_contiguous(&o->a)) { ZVAL_COPY(out, arr); return SUCCESS; }
    return cast_copy(&o->a, dtype, out);
}

/* ================================================================ NDArray -> PHP */

static void to_php(const tsr_array *a, int d, const char *p, zval *rv)
{
    if (d == a->ndim) { elem_to_zval(a->dtype, p, rv); return; }
    const int64_t n = a->shape[d], st = a->strides[d];
    array_init_size(rv, (uint32_t)n);
    if (n == 0) return;
    zend_hash_real_init_packed(Z_ARRVAL_P(rv));
    if (d == a->ndim - 1 && a->dtype != D_C128) {
        ZEND_HASH_FILL_PACKED(Z_ARRVAL_P(rv)) {
            for (int64_t i = 0; i < n; i++) {
                const char *q = p + i * st;
                switch (a->dtype) {
                case D_F64: ZEND_HASH_FILL_SET_DOUBLE(*(const double *)q); break;
                case D_F32: ZEND_HASH_FILL_SET_DOUBLE((double)*(const float *)q); break;
                case D_I64: ZEND_HASH_FILL_SET_LONG((zend_long)*(const int64_t *)q); break;
                case D_I32: ZEND_HASH_FILL_SET_LONG(*(const int32_t *)q); break;
                case D_U8: ZEND_HASH_FILL_SET_LONG(*(const uint8_t *)q); break;
                default: { zval b; ZVAL_BOOL(&b, *(const uint8_t *)q != 0); ZEND_HASH_FILL_SET(&b); } break;
                }
                ZEND_HASH_FILL_NEXT();
            }
        } ZEND_HASH_FILL_END();
        return;
    }
    for (int64_t i = 0; i < n; i++) {
        zval child;
        to_php(a, d + 1, p + i * st, &child);
        add_next_index_zval(rv, &child);
    }
}

/* ================================================================ core operations on metadata */

static int binary_core_into(int op, zval *left, zval *right, tsr_obj *dst, zval *rv);
static int binary_core(int op, zval *left, zval *right, zval *rv) { return binary_core_into(op, left, right, NULL, rv); }
static int cast_into(const tsr_array *src, tsr_obj *dst, const char *what);

/* dst can take the kernel's output directly: same shape, and no input overlaps it except element for element. */
static int direct_ok(const tsr_obj *dst, int32_t nd, const int64_t *shape, const tsr_array *in, const int64_t *in_st)
{
    if (dst->a.ndim != nd || memcmp(dst->a.shape, shape, sizeof(int64_t) * (size_t)nd) != 0) return 0;
    if (in->data != dst->a.data) return 1;          /* different memory blocks */
    if (in->offset != dst->a.offset) return 0;
    for (int d = 0; d < nd; d++) if (in_st[d] != dst->a.strides[d] && shape[d] > 1) return 0;
    return 1;                                       /* exact aliasing (in-place): element-wise kernels allow it */
}

/* A zval (NDArray, PHP array or scalar) as an NDArray; scalars take dtype `scalar_dt`. */
static int operand(zval *z, int scalar_dt, zval *out)
{
    ZVAL_DEREF(z);
    if (is_ndarray(z)) { ZVAL_COPY(out, z); return SUCCESS; }
    if (Z_TYPE_P(z) == IS_LONG || Z_TYPE_P(z) == IS_DOUBLE || Z_TYPE_P(z) == IS_TRUE || Z_TYPE_P(z) == IS_FALSE) {
        if (tsr_ndarray_new(out, scalar_dt, 0, NULL, 0) == FAILURE) return FAILURE;
        zval_to_elem(z, scalar_dt, elem0(&Z_TSR_P(out)->a));
        return SUCCESS;
    }
    if (Z_TYPE_P(z) == IS_ARRAY) return ndarray_from_php(z, -1, out);
    zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "Unsupported operand of type %s", zend_zval_type_name(z));
    return FAILURE;
}

static int is_scalar_zv(zval *z)
{
    return Z_TYPE_P(z) == IS_LONG || Z_TYPE_P(z) == IS_DOUBLE || Z_TYPE_P(z) == IS_TRUE || Z_TYPE_P(z) == IS_FALSE;
}

static int dtype_of_zv(zval *z)
{
    if (is_ndarray(z)) return Z_TSR_P(z)->a.dtype;
    if (Z_TYPE_P(z) == IS_DOUBLE) return D_F64;
    if (Z_TYPE_P(z) == IS_LONG) return D_I64;
    if (Z_TYPE_P(z) == IS_TRUE || Z_TYPE_P(z) == IS_FALSE) return D_BOOL;
    return -1;
}

static int is_cmp(int op) { return op >= OP_EQ && op <= OP_GE; }
static int is_logic(int op) { return op == OP_AND || op == OP_OR || op == OP_XOR; }

/*
 * left (op) right with NumPy broadcasting and promotion. Either side may be a
 * PHP scalar (weakly typed) or PHP array. With dst, the result goes into that
 * array: straight from the kernel when shape and dtype match, else by a cast copy.
 */
static int binary_core_into(int op, zval *left, zval *right, tsr_obj *dst, zval *rv)
{
    ZVAL_DEREF(left);
    ZVAL_DEREF(right);
    zval A, B, Ac, Bc, T;
    ZVAL_UNDEF(&A); ZVAL_UNDEF(&B); ZVAL_UNDEF(&Ac); ZVAL_UNDEF(&Bc); ZVAL_UNDEF(&T);
    int ret = FAILURE;
    int dl = dtype_of_zv(left), dr = dtype_of_zv(right);
    int dt;
    if (is_scalar_zv(left) && !is_scalar_zv(right)) {
        if (operand(right, 0, &B) == FAILURE) goto out;
        dt = with_scalar(Z_TSR(B)->a.dtype, left);
        if (operand(left, dt, &A) == FAILURE) goto out;
    } else if (is_scalar_zv(right)) {
        if (operand(left, dl >= 0 ? dl : D_F64, &A) == FAILURE) goto out;
        dt = is_scalar_zv(left) ? promote(dl, dr) : with_scalar(Z_TSR(A)->a.dtype, right);
        if (operand(right, dt, &B) == FAILURE) goto out;
    } else {
        if (operand(left, 0, &A) == FAILURE || operand(right, 0, &B) == FAILURE) goto out;
        dt = promote(Z_TSR(A)->a.dtype, Z_TSR(B)->a.dtype);
    }

    int opdt = dt, resdt = dt;
    if (is_logic(op)) { opdt = resdt = D_BOOL; }
    else if (is_cmp(op)) { resdt = D_BOOL; if (dt == D_BOOL) opdt = D_U8; }
    else if (op == OP_DIV || op == OP_ATAN2 || op == OP_HYPOT) { opdt = resdt = to_float(dt); }
    else if (op == OP_POW && (is_int(dt) || dt == D_BOOL)) { opdt = D_F64; resdt = dt == D_BOOL ? D_I64 : dt; }
    else if (dt == D_BOOL) {
        if (op == OP_MAX || op == OP_MIN) { opdt = D_U8; resdt = D_BOOL; }
        else opdt = resdt = D_I64;
    }

    tsr_array a = Z_TSR(A)->a, b = Z_TSR(B)->a;
    /* bool and uint8 share a layout: reinterpret instead of copying */
    if (a.dtype != opdt) {
        if ((a.dtype == D_BOOL && opdt == D_U8) || (a.dtype == D_U8 && opdt == D_BOOL)) a.dtype = opdt;
        else { if (cast_copy(&a, opdt, &Ac) == FAILURE) goto out; a = Z_TSR(Ac)->a; }
    }
    if (b.dtype != opdt) {
        if ((b.dtype == D_BOOL && opdt == D_U8) || (b.dtype == D_U8 && opdt == D_BOOL)) b.dtype = opdt;
        else { if (cast_copy(&b, opdt, &Bc) == FAILURE) goto out; b = Z_TSR(Bc)->a; }
    }
    int32_t nd;
    int64_t shape[32], sa[32], sb[32];
    int rc = tsr_broadcast_shape(a.ndim, a.shape, b.ndim, b.shape, &nd, shape);
    if (rc != 0) {
        char s1[160], s2[160];
        shape_str(s1, sizeof(s1), &a);
        shape_str(s2, sizeof(s2), &b);
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "Shapes %s and %s cannot be broadcast together", s1, s2);
        goto out;
    }
    tsr_broadcast_strides(&a, nd, shape, sa);
    tsr_broadcast_strides(&b, nd, shape, sb);
    int computedt = is_cmp(op) ? D_BOOL : (opdt == D_U8 && resdt == D_BOOL ? D_BOOL : opdt);
    if (dst && computedt == resdt && dst->a.dtype == resdt && direct_ok(dst, nd, shape, &a, sa) && direct_ok(dst, nd, shape, &b, sb)) {
        if (tsr_array_size(&dst->a) > 0) {
            rc = tsr_binary(op, opdt, nd, shape, elem0(&a), sa, elem0(&b), sb, elem0(&dst->a), dst->a.strides);
            if (rc != 0) { tsr_throw_rc(rc, "binary operation"); goto out; }
        }
        ZVAL_OBJ_COPY(rv, &dst->std);
        ret = SUCCESS;
        goto out;
    }
    if (tsr_ndarray_new(&T, computedt, nd, shape, 0) == FAILURE) goto out;
    tsr_obj *t = Z_TSR(T);
    if (tsr_array_size(&t->a) > 0) {
        rc = tsr_binary(op, opdt, nd, shape, elem0(&a), sa, elem0(&b), sb, elem0(&t->a), t->a.strides);
        if (rc != 0) { tsr_throw_rc(rc, "binary operation"); goto out; }
    }
    if (dst) {
        tsr_array r = t->a;
        r.dtype = computedt;
        if (computedt != resdt && !(computedt == D_BOOL && resdt == D_BOOL)) {
            zval R;
            if (cast_copy(&t->a, resdt, &R) == FAILURE) goto out;
            int cr = cast_into(&Z_TSR(R)->a, dst, "out");
            zval_ptr_dtor(&R);
            if (cr == FAILURE) goto out;
        } else if (cast_into(&r, dst, "out") == FAILURE) goto out;
        ZVAL_OBJ_COPY(rv, &dst->std);
    } else if (computedt != resdt) {
        if (cast_copy(&t->a, resdt, rv) == FAILURE) goto out;
    } else {
        ZVAL_COPY(rv, &T);
    }
    ret = SUCCESS;
out:
    zval_ptr_dtor(&A); zval_ptr_dtor(&B); zval_ptr_dtor(&Ac); zval_ptr_dtor(&Bc); zval_ptr_dtor(&T);
    return ret;
}

static const int FLOAT_UNARY[] = {U_SQRT, U_EXP, U_LOG, U_LOG10, U_LOG2, U_SIN, U_COS, U_TAN, U_ARCSIN, U_ARCCOS,
                                  U_ARCTAN, U_SINH, U_COSH, U_TANH, U_EXPM1, U_LOG1P, U_RECIPROCAL};

static int unary_core_into(int op, tsr_obj *src, tsr_obj *dst, zval *rv);
static int unary_core(int op, tsr_obj *src, zval *rv) { return unary_core_into(op, src, NULL, rv); }

static int unary_core_into(int op, tsr_obj *src, tsr_obj *dst, zval *rv)
{
    tsr_array a = src->a;
    zval C;
    ZVAL_UNDEF(&C);
    int outdt = a.dtype;
    int isf = 0;
    for (size_t i = 0; i < sizeof(FLOAT_UNARY) / sizeof(FLOAT_UNARY[0]); i++) if (FLOAT_UNARY[i] == op) isf = 1;
    if (isf && to_float(a.dtype) != a.dtype) {
        if (cast_copy(&a, to_float(a.dtype), &C) == FAILURE) return FAILURE;
        a = Z_TSR(C)->a;
        outdt = a.dtype;
    } else if (op == U_ISNAN || op == U_ISINF || op == U_ISFINITE) {
        outdt = D_BOOL;
        if (a.dtype == D_BOOL) a.dtype = D_U8;
    } else if (op == U_REAL || op == U_IMAG || op == U_CABS || op == U_ANGLE) {
        outdt = D_F64;
    } else if (a.dtype == D_BOOL && op != U_NOT) {
        if (cast_copy(&a, D_I64, &C) == FAILURE) return FAILURE;
        a = Z_TSR(C)->a;
        outdt = D_I64;
    } else if ((op == U_FLOOR || op == U_CEIL || op == U_RINT) && kind(a.dtype) < 2) {
        if (dst) {
            if (cast_into(&a, dst, "out") == FAILURE) return FAILURE;
            ZVAL_OBJ_COPY(rv, &dst->std);
            return SUCCESS;
        }
        return cast_copy(&a, a.dtype, rv);
    }
    if (dst && dst->a.dtype == outdt && direct_ok(dst, a.ndim, a.shape, &a, a.strides)) {
        int rc = tsr_array_size(&a) ? tsr_unary(op, a.dtype, a.ndim, a.shape, elem0(&a), a.strides, elem0(&dst->a), dst->a.strides) : 0;
        zval_ptr_dtor(&C);
        if (rc != 0) { tsr_throw_rc(rc, "unary operation"); return FAILURE; }
        ZVAL_OBJ_COPY(rv, &dst->std);
        return SUCCESS;
    }
    if (dst) {
        zval T;
        int r = unary_core_into(op, src, NULL, &T);
        zval_ptr_dtor(&C);
        if (r == FAILURE) return FAILURE;
        r = cast_into(&Z_TSR(T)->a, dst, "out");
        zval_ptr_dtor(&T);
        if (r == FAILURE) return FAILURE;
        ZVAL_OBJ_COPY(rv, &dst->std);
        return SUCCESS;
    }
    if (tsr_ndarray_new(rv, outdt, a.ndim, a.shape, 0) == FAILURE) { zval_ptr_dtor(&C); return FAILURE; }
    tsr_obj *o = Z_TSR_P(rv);
    int rc = tsr_array_size(&a) ? tsr_unary(op, a.dtype, a.ndim, a.shape, elem0(&a), a.strides, elem0(&o->a), o->a.strides) : 0;
    zval_ptr_dtor(&C);
    if (rc != 0) { zval_ptr_dtor(rv); ZVAL_UNDEF(rv); tsr_throw_rc(rc, "unary operation"); return FAILURE; }
    return SUCCESS;
}

/* Reduce along `axis` (-1 = all elements) -> NDArray (or scalar when axis = -1). */
static int reduce_core(int op, tsr_obj *src, zend_long axis, zend_bool has_axis, zval *rv)
{
    tsr_array a = src->a;
    zval C;
    ZVAL_UNDEF(&C);
    if (a.dtype == D_BOOL) a.dtype = D_U8;
    if (a.dtype == D_C128) {
        if (op != R_SUM) { zend_throw_exception(tsr_ce_dtype_exception, "Only sum() is supported for complex arrays", 0); return FAILURE; }
        /* view complex as float64 with a trailing axis of 2 and sum the leading axes */
        if (a.ndim >= 31) { tsr_throw_rc(-5, "sum"); return FAILURE; }
        a.dtype = D_F64;
        a.shape[a.ndim] = 2;
        a.strides[a.ndim] = 8;
        a.ndim++;
        if (!has_axis) {
            if (cast_copy(&a, D_F64, &C) == FAILURE) return FAILURE;
            tsr_array f = Z_TSR(C)->a;
            int64_t n = tsr_array_size(&f) / 2;
            f.ndim = 2; f.shape[0] = n; f.shape[1] = 2; f.strides[0] = 16; f.strides[1] = 8;
            double out[2];
            int64_t so[1] = {8};
            int rc = tsr_reduce(R_SUM, D_F64, 2, f.shape, elem0(&f), f.strides, 0, D_F64, out, so);
            zval_ptr_dtor(&C);
            if (rc != 0) { tsr_throw_rc(rc, "sum"); return FAILURE; }
            array_init_size(rv, 2);
            add_next_index_double(rv, out[0]);
            add_next_index_double(rv, out[1]);
            return SUCCESS;
        }
        zend_long ax = axis < 0 ? axis + a.ndim - 1 : axis;
        if (ax < 0 || ax >= a.ndim - 1) { zend_throw_exception(tsr_ce_shape_exception, "axis out of range", 0); return FAILURE; }
        int64_t shape[32];
        int nd = 0;
        for (int d = 0; d < a.ndim; d++) if (d != ax) shape[nd++] = a.shape[d];
        zval F;
        if (tsr_ndarray_new(&F, D_F64, nd, shape, 0) == FAILURE) return FAILURE;
        tsr_obj *f = Z_TSR(F);
        int rc = tsr_reduce(R_SUM, D_F64, a.ndim, a.shape, elem0(&a), a.strides, (int32_t)ax, D_F64, elem0(&f->a), f->a.strides);
        if (rc != 0) { zval_ptr_dtor(&F); tsr_throw_rc(rc, "sum"); return FAILURE; }
        /* reinterpret (..., 2) float64 as complex128 */
        tsr_array_init(&f->a, D_C128, nd - 1, shape, f->a.data);
        ZVAL_COPY_VALUE(rv, &F);
        return SUCCESS;
    }
    if (!has_axis) {
        if (!tsr_array_is_contiguous(&a)) {
            if (cast_copy(&a, a.dtype, &C) == FAILURE) return FAILURE;
            a = Z_TSR(C)->a;
        }
        a.shape[0] = tsr_array_size(&a);
        a.strides[0] = ITEMSIZE[a.dtype];
        a.ndim = 1;
        axis = 0;
    } else {
        if (a.ndim == 0) { zend_throw_exception(tsr_ce_shape_exception, "axis given for a 0-d array", 0); return FAILURE; }
        if (axis < 0) axis += a.ndim;
        if (axis < 0 || axis >= a.ndim) { zend_throw_exception_ex(tsr_ce_shape_exception, 0, "axis %ld is out of bounds for a %d-d array", (long)axis, a.ndim); return FAILURE; }
    }
    int outdt;
    switch (op) {
    case R_SUM: case R_PROD: outdt = kind(a.dtype) == 2 ? a.dtype : D_I64; break;
    case R_MIN: case R_MAX: outdt = a.dtype; break;
    case R_ARGMIN: case R_ARGMAX: outdt = D_I64; break;
    default: outdt = D_BOOL;
    }
    int64_t len = a.shape[axis];
    if (len == 0 && (op == R_MIN || op == R_MAX || op == R_ARGMIN || op == R_ARGMAX)) {
        zval_ptr_dtor(&C);
        zend_throw_exception(tsr_ce_shape_exception, "zero-size array to a reduction operation that has no identity", 0);
        return FAILURE;
    }
    int64_t shape[32];
    int nd = 0;
    for (int d = 0; d < a.ndim; d++) if (d != axis) shape[nd++] = a.shape[d];
    zval R;
    if (tsr_ndarray_new(&R, outdt, nd, shape, 0) == FAILURE) { zval_ptr_dtor(&C); return FAILURE; }
    tsr_obj *r = Z_TSR(R);
    int rc = 0;
    if (tsr_array_size(&r->a) > 0) {
        if (axis < a.ndim - 1 && (op == R_SUM || op == R_PROD || op == R_MIN || op == R_MAX) && tsr_array_is_contiguous(&a)) {
            int64_t outer = 1, inner = 1;
            for (int d = 0; d < axis; d++) outer *= a.shape[d];
            for (int d = (int)axis + 1; d < a.ndim; d++) inner *= a.shape[d];
            rc = tsr_reduce_mid(op, a.dtype, outer, len, inner, elem0(&a), elem0(&r->a));
        } else {
            rc = tsr_reduce(op, a.dtype, a.ndim, a.shape, elem0(&a), a.strides, (int32_t)axis, outdt, elem0(&r->a), r->a.strides);
        }
    }
    zval_ptr_dtor(&C);
    if (rc != 0) { zval_ptr_dtor(&R); tsr_throw_rc(rc, "reduce"); return FAILURE; }
    if (src->a.dtype == D_BOOL && (op == R_MIN || op == R_MAX)) r->a.dtype = D_BOOL;
    if (!has_axis) {
        elem_to_zval(r->a.dtype, elem0(&r->a), rv);
        zval_ptr_dtor(&R);
    } else {
        ZVAL_COPY_VALUE(rv, &R);
    }
    return SUCCESS;
}

/* mean (var = 0) or variance with ddof (var = 1) */
static int moments_core(tsr_obj *src, zend_long axis, zend_bool has_axis, int var, zend_long ddof, int sqrt_it, zval *rv)
{
    tsr_array a = src->a;
    zval C;
    ZVAL_UNDEF(&C);
    if (a.dtype == D_BOOL) a.dtype = D_U8;
    if (a.dtype == D_C128) { zend_throw_exception(tsr_ce_dtype_exception, "mean/var of complex arrays is not supported", 0); return FAILURE; }
    if (!has_axis) {
        if (!tsr_array_is_contiguous(&a)) { if (cast_copy(&a, a.dtype, &C) == FAILURE) return FAILURE; a = Z_TSR(C)->a; }
        a.shape[0] = tsr_array_size(&a);
        a.strides[0] = ITEMSIZE[a.dtype];
        a.ndim = 1;
        axis = 0;
    } else {
        if (axis < 0) axis += a.ndim;
        if (axis < 0 || axis >= a.ndim) { zend_throw_exception(tsr_ce_shape_exception, "axis out of range", 0); return FAILURE; }
    }
    int64_t len = a.shape[axis];
    int64_t shape[32];
    int nd = 0;
    for (int d = 0; d < a.ndim; d++) if (d != axis) shape[nd++] = a.shape[d];
    zval M, V;
    if (tsr_ndarray_new(&M, D_F64, nd, shape, 0) == FAILURE) { zval_ptr_dtor(&C); return FAILURE; }
    if (tsr_ndarray_new(&V, D_F64, nd, shape, 0) == FAILURE) { zval_ptr_dtor(&C); zval_ptr_dtor(&M); return FAILURE; }
    tsr_obj *m = Z_TSR(M), *v = Z_TSR(V);
    int64_t n = tsr_array_size(&m->a);
    int rc = n > 0 && len > 0 ? tsr_moments(a.dtype, a.ndim, a.shape, elem0(&a), a.strides, (int32_t)axis,
                                           (double *)elem0(&m->a), m->a.strides, (double *)elem0(&v->a), v->a.strides) : 0;
    zval_ptr_dtor(&C);
    if (rc != 0) { zval_ptr_dtor(&M); zval_ptr_dtor(&V); tsr_throw_rc(rc, "moments"); return FAILURE; }
    double *out = (double *)elem0(var ? &v->a : &m->a);
    double div = (double)(len - ddof);
    for (int64_t i = 0; i < n; i++) {
        double x = len == 0 ? NAN : var ? (div > 0 ? out[i] / div : NAN) : out[i];
        out[i] = sqrt_it ? sqrt(x) : x;
    }
    zval *keep = var ? &V : &M, *drop = var ? &M : &V;
    zval_ptr_dtor(drop);
    if (!has_axis) {
        ZVAL_DOUBLE(rv, out[0]);
        zval_ptr_dtor(keep);
    } else {
        ZVAL_COPY_VALUE(rv, keep);
    }
    return SUCCESS;
}

/* Copy `value` (NDArray / PHP array / scalar, broadcast and cast) into the view `dst`. */
static int assign_core(const tsr_array *dst, zval *value)
{
    zval V, C;
    ZVAL_UNDEF(&V);
    ZVAL_UNDEF(&C);
    if (operand(value, dst->dtype, &V) == FAILURE) return FAILURE;
    tsr_array s = Z_TSR(V)->a;
    /* the source may overlap the destination (a['1:'] = a[':-1']): go through a copy */
    if (s.data == dst->data) {
        if (cast_copy(&s, s.dtype, &C) == FAILURE) { zval_ptr_dtor(&V); return FAILURE; }
        s = Z_TSR(C)->a;
    }
    int64_t st[32];
    int rc = tsr_broadcast_strides(&s, dst->ndim, dst->shape, st);
    if (rc == 0 && tsr_array_size(dst) > 0)
        rc = tsr_copy(s.dtype, dst->dtype, dst->ndim, dst->shape, elem0(&s), st, elem0(dst), dst->strides);
    zval_ptr_dtor(&V);
    zval_ptr_dtor(&C);
    if (rc != 0) {
        if (rc == -6) {
            char s1[160], s2[160];
            shape_str(s1, sizeof(s1), &s);
            shape_str(s2, sizeof(s2), dst);
            zend_throw_exception_ex(tsr_ce_shape_exception, 0, "Cannot assign shape %s into %s", s1, s2);
        } else {
            tsr_throw_rc(rc, "assign");
        }
        return FAILURE;
    }
    return SUCCESS;
}

/* ================================================================ indexing */

static int slice_view(tsr_obj *o, const char *spec, zval *rv, int scalar_ok)
{
    tsr_array v;
    int rc = tsr_array_slice(&o->a, spec, &v);
    if (rc == -1) { zend_throw_exception_ex(tsr_ce_index_exception, 0, "Invalid index '%s'", spec); return FAILURE; }
    if (rc == -2) {
        char shp[256];
        shape_str(shp, sizeof(shp), &o->a);
        zend_throw_exception_ex(tsr_ce_index_exception, 0, "Index '%s' is out of bounds for an array of shape %s", spec, shp);
        return FAILURE;
    }
    if (rc != 0) { tsr_throw_rc(rc, "index"); return FAILURE; }
    if (scalar_ok && v.ndim == 0) { elem_to_zval(v.dtype, elem0(&v), rv); return SUCCESS; }
    make_view(rv, o, &v);
    return SUCCESS;
}

static int int_view(tsr_obj *o, zend_long i, zval *rv, int scalar_ok)
{
    if (o->a.ndim == 0) { zend_throw_exception(tsr_ce_index_exception, "Cannot index a 0-d array", 0); return FAILURE; }
    zend_long len = o->a.shape[0];
    zend_long j = i < 0 ? i + len : i;
    if (j < 0 || j >= len) {
        zend_throw_exception_ex(tsr_ce_index_exception, 0, "Index %ld is out of bounds for axis 0 with size %ld", (long)i, (long)len);
        return FAILURE;
    }
    tsr_array v = o->a;
    v.offset += j * o->a.strides[0];
    for (int d = 1; d < v.ndim; d++) { v.shape[d - 1] = v.shape[d]; v.strides[d - 1] = v.strides[d]; }
    v.ndim--;
    if (scalar_ok && v.ndim == 0) { elem_to_zval(v.dtype, elem0(&v), rv); return SUCCESS; }
    make_view(rv, o, &v);
    return SUCCESS;
}

/* a[mask] -> 1-D copy of the selected elements */
static int mask_select(tsr_obj *o, tsr_obj *mask, zval *rv)
{
    if (mask->a.ndim != o->a.ndim || memcmp(mask->a.shape, o->a.shape, sizeof(int64_t) * (size_t)o->a.ndim) != 0) {
        zend_throw_exception(tsr_ce_shape_exception, "Boolean mask must have the same shape as the array", 0);
        return FAILURE;
    }
    zval S, M;
    ZVAL_UNDEF(&S);
    ZVAL_UNDEF(&M);
    zval src, mz;
    ZVAL_OBJ(&src, &o->std);
    ZVAL_OBJ(&mz, &mask->std);
    if (tsr_ndarray_contiguous_as(&src, o->a.dtype, &S) == FAILURE) return FAILURE;
    if (tsr_ndarray_contiguous_as(&mz, D_BOOL, &M) == FAILURE) { zval_ptr_dtor(&S); return FAILURE; }
    int64_t n = tsr_array_size(&Z_TSR(M)->a);
    int64_t k = tsr_count_true(n, (const uint8_t *)elem0(&Z_TSR(M)->a));
    int64_t shape[1] = {k};
    int ret = tsr_ndarray_new(rv, o->a.dtype, 1, shape, 0);
    if (ret == SUCCESS && k > 0)
        tsr_compress(ITEMSIZE[o->a.dtype], n, elem0(&Z_TSR(S)->a), (const uint8_t *)elem0(&Z_TSR(M)->a), elem0(&Z_TSR_P(rv)->a));
    zval_ptr_dtor(&S);
    zval_ptr_dtor(&M);
    return ret;
}

/* a[[i, j, ...]] -> gather along axis 0 */
static int take_rows(tsr_obj *o, zval *idx, zval *rv)
{
    zval I, S, src;
    ZVAL_UNDEF(&I);
    ZVAL_UNDEF(&S);
    if (o->a.ndim == 0) { zend_throw_exception(tsr_ce_index_exception, "Cannot index a 0-d array", 0); return FAILURE; }
    if (tsr_ndarray_from_zval(idx, D_I64, &I) == FAILURE) return FAILURE;
    zval IC;
    if (tsr_ndarray_contiguous_as(&I, D_I64, &IC) == FAILURE) { zval_ptr_dtor(&I); return FAILURE; }
    zval_ptr_dtor(&I);
    ZVAL_OBJ(&src, &o->std);
    if (tsr_ndarray_contiguous_as(&src, o->a.dtype, &S) == FAILURE) { zval_ptr_dtor(&IC); return FAILURE; }
    tsr_array *ia = &Z_TSR(IC)->a, *sa = &Z_TSR(S)->a;
    int64_t inner = 1;
    for (int d = 1; d < sa->ndim; d++) inner *= sa->shape[d];
    int64_t shape[32];
    int nd = 0;
    for (int d = 0; d < ia->ndim; d++) shape[nd++] = ia->shape[d];
    for (int d = 1; d < sa->ndim; d++) shape[nd++] = sa->shape[d];
    int ret = FAILURE;
    if (nd > 32) { tsr_throw_rc(-5, "take"); goto out; }
    if (tsr_ndarray_new(rv, sa->dtype, nd, shape, 0) == FAILURE) goto out;
    int64_t ni = tsr_array_size(ia);
    int rc = ni > 0 ? tsr_take(ITEMSIZE[sa->dtype], 1, sa->shape[0], inner, elem0(sa), (const int64_t *)elem0(ia), ni, elem0(&Z_TSR_P(rv)->a)) : 0;
    if (rc != 0) { zval_ptr_dtor(rv); ZVAL_UNDEF(rv); tsr_throw_rc(rc, "take"); goto out; }
    ret = SUCCESS;
out:
    zval_ptr_dtor(&IC);
    zval_ptr_dtor(&S);
    return ret;
}

static zval *tsr_read_dimension(zend_object *obj, zval *offset, int type, zval *rv)
{
    tsr_obj *o = tsr_obj_from(obj);
    if (offset == NULL) { zend_throw_exception(tsr_ce_index_exception, "Arrays have a fixed size; [] is not supported", 0); return &EG(uninitialized_zval); }
    ZVAL_DEREF(offset);
    int r;
    switch (Z_TYPE_P(offset)) {
    case IS_LONG: r = int_view(o, Z_LVAL_P(offset), rv, 1); break;
    case IS_STRING: r = slice_view(o, Z_STRVAL_P(offset), rv, 1); break;
    case IS_ARRAY: r = take_rows(o, offset, rv); break;
    case IS_OBJECT:
        if (is_ndarray(offset)) {
            tsr_obj *m = Z_TSR_P(offset);
            r = m->a.dtype == D_BOOL ? mask_select(o, m, rv) : take_rows(o, offset, rv);
            break;
        }
        ZEND_FALLTHROUGH;
    default:
        zend_throw_exception_ex(tsr_ce_index_exception, 0, "Unsupported index type %s", zend_zval_type_name(offset));
        r = FAILURE;
    }
    if (r == FAILURE) return &EG(uninitialized_zval);
    return rv;
}

int tsr_require_writable(tsr_obj *o, const char *what)
{
    if (tsr_root(o)->flags & TSR_F_READONLY) {
        zend_throw_exception_ex(tsr_ce_exception, 0, "%s: the array is read-only (memory-mapped with mode 'r')", what);
        return FAILURE;
    }
    return SUCCESS;
}

static void tsr_write_dimension(zend_object *obj, zval *offset, zval *value)
{
    tsr_obj *o = tsr_obj_from(obj);
    if (tsr_require_writable(o, "assignment") == FAILURE) return;
    if (offset == NULL) { zend_throw_exception(tsr_ce_index_exception, "Arrays have a fixed size; [] is not supported", 0); return; }
    ZVAL_DEREF(offset);
    zval view;
    ZVAL_UNDEF(&view);
    if (Z_TYPE_P(offset) == IS_OBJECT && is_ndarray(offset) && Z_TSR_P(offset)->a.dtype == D_BOOL) {
        /* a[mask] = scalar | values */
        tsr_obj *m = Z_TSR_P(offset);
        if (m->a.ndim != o->a.ndim || memcmp(m->a.shape, o->a.shape, sizeof(int64_t) * (size_t)o->a.ndim) != 0) {
            zend_throw_exception(tsr_ce_shape_exception, "Boolean mask must have the same shape as the array", 0);
            return;
        }
        zval M, V, W;
        if (tsr_ndarray_contiguous_as(offset, D_BOOL, &M) == FAILURE) return;
        if (operand(value, o->a.dtype, &V) == FAILURE) { zval_ptr_dtor(&M); return; }
        if (tsr_ndarray_contiguous_as(&V, o->a.dtype, &W) == FAILURE) { zval_ptr_dtor(&M); zval_ptr_dtor(&V); return; }
        int contiguous = tsr_array_is_contiguous(&o->a);
        zval T;
        ZVAL_UNDEF(&T);
        tsr_array target = o->a;
        if (!contiguous) {
            if (cast_copy(&o->a, o->a.dtype, &T) == FAILURE) goto mask_out;
            target = Z_TSR(T)->a;
        }
        int rc = tsr_mask_assign(ITEMSIZE[o->a.dtype], tsr_array_size(&target), elem0(&target), (const uint8_t *)elem0(&Z_TSR(M)->a),
                                 elem0(&Z_TSR(W)->a), tsr_array_size(&Z_TSR(W)->a));
        if (rc != 0) tsr_throw_rc(rc, "mask assignment");
        else if (!contiguous) {
            int64_t st[32];
            memcpy(st, target.strides, sizeof(st));
            tsr_copy(o->a.dtype, o->a.dtype, o->a.ndim, o->a.shape, elem0(&target), st, elem0(&o->a), o->a.strides);
        }
    mask_out:
        zval_ptr_dtor(&T);
        zval_ptr_dtor(&M);
        zval_ptr_dtor(&V);
        zval_ptr_dtor(&W);
        return;
    }
    int r;
    if (Z_TYPE_P(offset) == IS_LONG) r = int_view(o, Z_LVAL_P(offset), &view, 0);
    else if (Z_TYPE_P(offset) == IS_STRING) r = slice_view(o, Z_STRVAL_P(offset), &view, 0);
    else { zend_throw_exception_ex(tsr_ce_index_exception, 0, "Unsupported index type %s for assignment", zend_zval_type_name(offset)); return; }
    if (r == FAILURE) return;
    assign_core(&Z_TSR(view)->a, value);
    zval_ptr_dtor(&view);
}

static int tsr_has_dimension(zend_object *obj, zval *offset, int check_empty)
{
    tsr_obj *o = tsr_obj_from(obj);
    ZVAL_DEREF(offset);
    if (Z_TYPE_P(offset) == IS_LONG) {
        if (o->a.ndim == 0) return 0;
        zend_long i = Z_LVAL_P(offset), len = o->a.shape[0];
        return (i < 0 ? i + len : i) >= 0 && (i < 0 ? i + len : i) < len;
    }
    if (Z_TYPE_P(offset) == IS_STRING) {
        tsr_array v;
        return tsr_array_slice(&o->a, Z_STRVAL_P(offset), &v) == 0;
    }
    return 0;
}

static void tsr_unset_dimension(zend_object *obj, zval *offset)
{
    zend_throw_exception(tsr_ce_index_exception, "Array elements cannot be unset", 0);
}

static zend_result tsr_count_elements(zend_object *obj, zend_long *count)
{
    tsr_obj *o = tsr_obj_from(obj);
    if (o->a.ndim == 0) { zend_throw_exception(tsr_ce_exception, "count() of a 0-d array", 0); return FAILURE; }
    *count = o->a.shape[0];
    return SUCCESS;
}

static zend_object *tsr_clone(zend_object *old)
{
    tsr_obj *o = tsr_obj_from(old);
    zval copy;
    if (cast_copy(&o->a, o->a.dtype, &copy) == FAILURE) {
        /* exception pending: hand back an empty 0-d array so the engine can unwind */
        zval empty;
        tsr_ndarray_new(&empty, D_F64, 0, NULL, 1);
        return Z_OBJ(empty);
    }
    return Z_OBJ(copy);
}

static int tsr_compare(zval *z1, zval *z2)
{
    ZEND_COMPARE_OBJECTS_FALLBACK(z1, z2);
    tsr_obj *a = Z_TSR_P(z1), *b = Z_TSR_P(z2);
    if (a == b) return 0;
    if (a->a.dtype != b->a.dtype || a->a.ndim != b->a.ndim || memcmp(a->a.shape, b->a.shape, sizeof(int64_t) * (size_t)a->a.ndim) != 0)
        return ZEND_UNCOMPARABLE;
    zval ca, cb, za, zb;
    ZVAL_OBJ(&za, &a->std);
    ZVAL_OBJ(&zb, &b->std);
    if (tsr_ndarray_contiguous_as(&za, a->a.dtype, &ca) == FAILURE) return ZEND_UNCOMPARABLE;
    if (tsr_ndarray_contiguous_as(&zb, b->a.dtype, &cb) == FAILURE) { zval_ptr_dtor(&ca); return ZEND_UNCOMPARABLE; }
    int eq = memcmp(elem0(&Z_TSR(ca)->a), elem0(&Z_TSR(cb)->a), (size_t)(tsr_array_size(&a->a) * ITEMSIZE[a->a.dtype])) == 0;
    zval_ptr_dtor(&ca);
    zval_ptr_dtor(&cb);
    return eq ? 0 : ZEND_UNCOMPARABLE;
}

static zend_result tsr_do_operation(uint8_t opcode, zval *result, zval *op1, zval *op2)
{
    int op;
    switch (opcode) {
    case ZEND_ADD: op = OP_ADD; break;
    case ZEND_SUB: op = OP_SUB; break;
    case ZEND_MUL: op = OP_MUL; break;
    case ZEND_DIV: op = OP_DIV; break;
    case ZEND_POW: op = OP_POW; break;
    case ZEND_MOD: op = OP_MOD; break;
    default: return FAILURE;
    }
    zval tmp;
    ZVAL_UNDEF(&tmp);
    int r = binary_core(op, op1, op2, &tmp);
    if (result == op1) zval_ptr_dtor(op1);
    if (r == FAILURE) { ZVAL_NULL(result); return SUCCESS; } /* exception pending */
    ZVAL_COPY_VALUE(result, &tmp);
    return SUCCESS;
}

static HashTable *tsr_debug_info(zend_object *obj, int *is_temp)
{
    tsr_obj *o = tsr_obj_from(obj);
    HashTable *ht;
    ALLOC_HASHTABLE(ht);
    zend_hash_init(ht, 6, NULL, ZVAL_PTR_DTOR, 0);
    zval v, shp, str;
    array_init(&shp);
    array_init(&str);
    for (int d = 0; d < o->a.ndim; d++) {
        add_next_index_long(&shp, o->a.shape[d]);
        add_next_index_long(&str, o->a.strides[d]);
    }
    ZVAL_STRING(&v, tsr_dtype_name(o->a.dtype));
    zend_hash_str_update(ht, "dtype", 5, &v);
    zend_hash_str_update(ht, "shape", 5, &shp);
    zend_hash_str_update(ht, "strides", 7, &str);
    ZVAL_BOOL(&v, o->base != NULL);
    zend_hash_str_update(ht, "view", 4, &v);
    if (tsr_array_size(&o->a) <= 100) {
        zval data;
        to_php(&o->a, 0, elem0(&o->a), &data);
        zend_hash_str_update(ht, "data", 4, &data);
    }
    *is_temp = 1;
    return ht;
}

/* ================================================================ iterator (foreach over axis 0) */

typedef struct {
    zend_object_iterator it;
    zend_long index;
    zval current;
} tsr_iter_t;

static void it_dtor(zend_object_iterator *it)
{
    tsr_iter_t *i = (tsr_iter_t *)it;
    zval_ptr_dtor(&i->current);
    zval_ptr_dtor(&it->data);
}

static zend_result it_valid(zend_object_iterator *it)
{
    tsr_iter_t *i = (tsr_iter_t *)it;
    tsr_obj *o = Z_TSR(it->data);
    return i->index < o->a.shape[0] ? SUCCESS : FAILURE;
}

static zval *it_current(zend_object_iterator *it)
{
    tsr_iter_t *i = (tsr_iter_t *)it;
    zval_ptr_dtor(&i->current);
    ZVAL_UNDEF(&i->current);
    if (int_view(Z_TSR(it->data), i->index, &i->current, 1) == FAILURE) return &EG(uninitialized_zval);
    return &i->current;
}

static void it_key(zend_object_iterator *it, zval *key) { ZVAL_LONG(key, ((tsr_iter_t *)it)->index); }
static void it_next(zend_object_iterator *it) { ((tsr_iter_t *)it)->index++; }
static void it_rewind(zend_object_iterator *it) { ((tsr_iter_t *)it)->index = 0; }

static const zend_object_iterator_funcs tsr_iter_funcs = {
    it_dtor, it_valid, it_current, it_key, it_next, it_rewind, NULL, NULL,
};

static zend_object_iterator *tsr_get_iterator(zend_class_entry *ce, zval *object, int by_ref)
{
    if (by_ref) { zend_throw_error(NULL, "An NDArray cannot be iterated by reference"); return NULL; }
    if (Z_TSR_P(object)->a.ndim == 0) { zend_throw_exception(tsr_ce_exception, "Iteration over a 0-d array", 0); return NULL; }
    tsr_iter_t *i = emalloc(sizeof(tsr_iter_t));
    zend_iterator_init(&i->it);
    ZVAL_OBJ_COPY(&i->it.data, Z_OBJ_P(object));
    i->it.funcs = &tsr_iter_funcs;
    i->index = 0;
    ZVAL_UNDEF(&i->current);
    return &i->it;
}

/* ================================================================ methods */

#define THIS_OBJ Z_TSR_P(ZEND_THIS)

static int shape_from_args(zval *args, uint32_t argc, int64_t *shape, int32_t *nd)
{
    zval *src = args;
    uint32_t n = argc;
    HashTable *ht = NULL;
    if (argc == 1 && Z_TYPE(args[0]) == IS_ARRAY) { ht = Z_ARRVAL(args[0]); n = zend_hash_num_elements(ht); }
    if (n > 32) { tsr_throw_rc(-5, "shape"); return FAILURE; }
    *nd = (int32_t)n;
    if (ht) {
        uint32_t k = 0;
        zval *e;
        ZEND_HASH_FOREACH_VAL(ht, e) {
            if (Z_TYPE_P(e) != IS_LONG) { zend_throw_exception(tsr_ce_shape_exception, "Dimensions must be integers", 0); return FAILURE; }
            shape[k++] = Z_LVAL_P(e);
        } ZEND_HASH_FOREACH_END();
        return SUCCESS;
    }
    for (uint32_t k = 0; k < n; k++) {
        if (Z_TYPE(src[k]) != IS_LONG) { zend_throw_exception(tsr_ce_shape_exception, "Dimensions must be integers", 0); return FAILURE; }
        shape[k] = Z_LVAL(src[k]);
    }
    return SUCCESS;
}

static int dtype_arg(zend_string *s, int dflt)
{
    if (!s) return dflt;
    int dt = tsr_dtype_from_name(ZSTR_VAL(s), ZSTR_LEN(s));
    if (dt < 0) zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "Unknown dtype '%s'", ZSTR_VAL(s));
    return dt;
}

PHP_METHOD(NDArray, __construct)
{
    zend_throw_error(NULL, "Use NDArray::array(), zeros(), ones(), full() or arange()");
}

PHP_METHOD(NDArray, array)
{
    zval *data;
    zend_string *dts = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(data)
        Z_PARAM_OPTIONAL
        Z_PARAM_STR_OR_NULL(dts)
    ZEND_PARSE_PARAMETERS_END();
    int dt = dtype_arg(dts, -1);
    if (dts && dt < 0) RETURN_THROWS();
    ZVAL_DEREF(data);
    if (is_ndarray(data)) { cast_copy(&Z_TSR_P(data)->a, dt < 0 ? Z_TSR_P(data)->a.dtype : dt, return_value); return; }
    ndarray_from_php(data, dt, return_value);
}

static void fill_new(INTERNAL_FUNCTION_PARAMETERS, int which)
{
    zval *shape_zv, *value = NULL;
    zend_string *dts = NULL;
    if (which == 2) {
        ZEND_PARSE_PARAMETERS_START(2, 3)
            Z_PARAM_ZVAL(shape_zv)
            Z_PARAM_ZVAL(value)
            Z_PARAM_OPTIONAL
            Z_PARAM_STR_OR_NULL(dts)
        ZEND_PARSE_PARAMETERS_END();
    } else {
        ZEND_PARSE_PARAMETERS_START(1, 2)
            Z_PARAM_ZVAL(shape_zv)
            Z_PARAM_OPTIONAL
            Z_PARAM_STR_OR_NULL(dts)
        ZEND_PARSE_PARAMETERS_END();
    }
    int64_t shape[32];
    int32_t nd;
    if (Z_TYPE_P(shape_zv) == IS_LONG) { nd = 1; shape[0] = Z_LVAL_P(shape_zv); }
    else if (shape_from_args(shape_zv, 1, shape, &nd) == FAILURE) RETURN_THROWS();
    int dflt = which == 2 ? (Z_TYPE_P(value) == IS_DOUBLE ? D_F64 : (Z_TYPE_P(value) == IS_LONG ? D_I64 : D_BOOL)) : D_F64;
    int dt = dtype_arg(dts, dflt);
    if (dt < 0) RETURN_THROWS();
    if (tsr_ndarray_new(return_value, dt, nd, shape, which == 0) == FAILURE) RETURN_THROWS();
    if (which == 0) return;
    char v[16] = {0};
    zval one;
    ZVAL_LONG(&one, 1);
    if (zval_to_elem(which == 1 ? &one : value, dt, v) == FAILURE) {
        zval_ptr_dtor(return_value);
        ZVAL_NULL(return_value);
        zend_throw_exception(tsr_ce_dtype_exception, "full() needs a number or bool", 0);
        RETURN_THROWS();
    }
    tsr_obj *o = Z_TSR_P(return_value);
    tsr_fill(dt, tsr_array_size(&o->a), elem0(&o->a), v);
}

PHP_METHOD(NDArray, zeros) { fill_new(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0); }
PHP_METHOD(NDArray, ones) { fill_new(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1); }
PHP_METHOD(NDArray, full) { fill_new(INTERNAL_FUNCTION_PARAM_PASSTHRU, 2); }
/* numpy.empty: an array of the given shape and dtype; Tessero zero-fills it (deterministic, unlike NumPy). */
PHP_METHOD(NDArray, empty) { fill_new(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0); }

PHP_METHOD(NDArray, arange)
{
    zval *start, *stop = NULL, *step = NULL;
    zend_string *dts = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_NUMBER(start)
        Z_PARAM_OPTIONAL
        Z_PARAM_NUMBER_OR_NULL(stop)
        Z_PARAM_NUMBER_OR_NULL(step)
        Z_PARAM_STR_OR_NULL(dts)
    ZEND_PARSE_PARAMETERS_END();
    double a = zval_get_double(start), b, s = step ? zval_get_double(step) : 1.0;
    int all_int = Z_TYPE_P(start) == IS_LONG && (!stop || Z_TYPE_P(stop) == IS_LONG) && (!step || Z_TYPE_P(step) == IS_LONG);
    if (stop) b = zval_get_double(stop); else { b = a; a = 0.0; }
    if (s == 0.0) { zend_throw_exception(tsr_ce_exception, "arange step must be non-zero", 0); RETURN_THROWS(); }
    int64_t n = (int64_t)ceil((b - a) / s);
    if (n < 0) n = 0;
    int dt = dtype_arg(dts, all_int ? D_I64 : D_F64);
    if (dt < 0) RETURN_THROWS();
    int64_t shape[1] = {n};
    if (tsr_ndarray_new(return_value, dt, 1, shape, 0) == FAILURE) RETURN_THROWS();
    if (n > 0) tsr_arange(dt, n, a, s, elem0(&Z_TSR_P(return_value)->a));
}

PHP_METHOD(NDArray, linspace)
{
    double a, b;
    zend_long num = 50;
    zend_bool endpoint = 1;
    ZEND_PARSE_PARAMETERS_START(2, 4)
        Z_PARAM_DOUBLE(a)
        Z_PARAM_DOUBLE(b)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(num)
        Z_PARAM_BOOL(endpoint)
    ZEND_PARSE_PARAMETERS_END();
    if (num < 0) { zend_throw_exception(tsr_ce_exception, "linspace num must be >= 0", 0); RETURN_THROWS(); }
    int64_t shape[1] = {num};
    if (tsr_ndarray_new(return_value, D_F64, 1, shape, 0) == FAILURE) RETURN_THROWS();
    double *p = (double *)elem0(&Z_TSR_P(return_value)->a);
    zend_long div = endpoint ? num - 1 : num;
    double step = div > 0 ? (b - a) / div : 0.0;
    if (num > 0) tsr_arange(D_F64, num, a, step, p);
    if (endpoint && num > 1) p[num - 1] = b;
}

PHP_METHOD(NDArray, eye)
{
    zend_long n, m = -1, k = 0;
    zend_string *dts = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 4)
        Z_PARAM_LONG(n)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(m)
        Z_PARAM_LONG(k)
        Z_PARAM_STR_OR_NULL(dts)
    ZEND_PARSE_PARAMETERS_END();
    if (m < 0) m = n;
    int dt = dtype_arg(dts, D_F64);
    if (dt < 0) RETURN_THROWS();
    int64_t shape[2] = {n, m};
    if (tsr_ndarray_new(return_value, dt, 2, shape, 1) == FAILURE) RETURN_THROWS();
    char *p = elem0(&Z_TSR_P(return_value)->a);
    zval one;
    ZVAL_LONG(&one, 1);
    for (zend_long i = k < 0 ? -k : 0; i < n && i + k < m; i++) zval_to_elem(&one, dt, p + (i * m + i + k) * ITEMSIZE[dt]);
}

/* Element count of an untrusted shape (fromBytes, unserialize): -1 for a negative dimension or an overflow. */
static int64_t checked_count(int32_t nd, const int64_t *shape)
{
    int64_t n = 1;
    for (int32_t d = 0; d < nd; d++) {
        if (shape[d] < 0) return -1;
        if (shape[d] != 0 && n > (INT64_MAX / 16) / shape[d]) return -1;
        n *= shape[d];
    }
    return n;
}

PHP_METHOD(NDArray, fromBytes)
{
    zend_string *bytes, *dts;
    zval *shape_zv = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_STR(bytes)
        Z_PARAM_STR(dts)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_OR_NULL(shape_zv)
    ZEND_PARSE_PARAMETERS_END();
    int dt = dtype_arg(dts, -1);
    if (dt < 0) RETURN_THROWS();
    int64_t n = (int64_t)ZSTR_LEN(bytes) / ITEMSIZE[dt];
    int64_t shape[32] = {n};
    int32_t nd = 1;
    if (shape_zv && shape_from_args(shape_zv, 1, shape, &nd) == FAILURE) RETURN_THROWS();
    const int64_t total = checked_count(nd, shape);
    if (total < 0 || total * ITEMSIZE[dt] != (int64_t)ZSTR_LEN(bytes)) {
        zend_throw_exception(tsr_ce_shape_exception, "Byte length does not match the shape and dtype", 0);
        RETURN_THROWS();
    }
    if (tsr_ndarray_new(return_value, dt, nd, shape, 0) == FAILURE) RETURN_THROWS();
    memcpy(elem0(&Z_TSR_P(return_value)->a), ZSTR_VAL(bytes), ZSTR_LEN(bytes));
}

PHP_METHOD(NDArray, where)
{
    zval *cond, *x, *y;
    ZEND_PARSE_PARAMETERS_START(3, 3)
        Z_PARAM_ZVAL(cond)
        Z_PARAM_ZVAL(x)
        Z_PARAM_ZVAL(y)
    ZEND_PARSE_PARAMETERS_END();
    zval C, X, Y, Xc, Yc, Cc;
    ZVAL_UNDEF(&C); ZVAL_UNDEF(&X); ZVAL_UNDEF(&Y); ZVAL_UNDEF(&Xc); ZVAL_UNDEF(&Yc); ZVAL_UNDEF(&Cc);
    if (tsr_ndarray_from_zval(cond, -1, &C) == FAILURE) RETURN_THROWS();
    int dx = dtype_of_zv(x), dy = dtype_of_zv(y);
    if (dx < 0 && Z_TYPE_P(x) == IS_ARRAY) dx = D_F64;
    if (dy < 0 && Z_TYPE_P(y) == IS_ARRAY) dy = D_F64;
    int dt = is_scalar_zv(x) && !is_scalar_zv(y) ? with_scalar(dy, x) : (is_scalar_zv(y) && !is_scalar_zv(x) ? with_scalar(dx, y) : promote(dx, dy));
    if (operand(x, dt, &X) == FAILURE || operand(y, dt, &Y) == FAILURE) goto out;
    if (tsr_ndarray_contiguous_as(&X, dt, &Xc) == FAILURE || tsr_ndarray_contiguous_as(&Y, dt, &Yc) == FAILURE) goto out;
    if (Z_TSR(C)->a.dtype != D_BOOL) { if (cast_copy(&Z_TSR(C)->a, D_BOOL, &Cc) == FAILURE) goto out; }
    else ZVAL_COPY(&Cc, &C);
    {
        tsr_array *c = &Z_TSR(Cc)->a, *xa = &Z_TSR(Xc)->a, *ya = &Z_TSR(Yc)->a;
        int32_t nd, nd2;
        int64_t sh1[32], sh[32], sc[32], sx[32], sy[32];
        if (tsr_broadcast_shape(c->ndim, c->shape, xa->ndim, xa->shape, &nd, sh1) != 0 ||
            tsr_broadcast_shape(nd, sh1, ya->ndim, ya->shape, &nd2, sh) != 0) {
            zend_throw_exception(tsr_ce_shape_exception, "where: shapes cannot be broadcast together", 0);
            goto out;
        }
        tsr_broadcast_strides(c, nd2, sh, sc);
        tsr_broadcast_strides(xa, nd2, sh, sx);
        tsr_broadcast_strides(ya, nd2, sh, sy);
        if (tsr_ndarray_new(return_value, dt, nd2, sh, 0) == FAILURE) goto out;
        tsr_obj *o = Z_TSR_P(return_value);
        if (tsr_array_size(&o->a) > 0)
            tsr_where(dt, nd2, sh, elem0(c), sc, elem0(xa), sx, elem0(ya), sy, elem0(&o->a), o->a.strides);
    }
out:
    zval_ptr_dtor(&C); zval_ptr_dtor(&X); zval_ptr_dtor(&Y); zval_ptr_dtor(&Xc); zval_ptr_dtor(&Yc); zval_ptr_dtor(&Cc);
}

/* complex(real, imag=null): a complex128 array from float64 real and imaginary parts (broadcast), as
   Tessero\NDArray::complex does on FFI. */
PHP_METHOD(NDArray, complex)
{
    zval *zr, *zi = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(zr)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(zi)
    ZEND_PARSE_PARAMETERS_END();
    zval R, I;
    ZVAL_UNDEF(&R); ZVAL_UNDEF(&I);
    if (tsr_ndarray_from_zval(zr, D_F64, &R) == FAILURE) RETURN_THROWS();
    if (zi && Z_TYPE_P(zi) != IS_NULL && tsr_ndarray_from_zval(zi, D_F64, &I) == FAILURE) { zval_ptr_dtor(&R); RETURN_THROWS(); }
    tsr_array *ra = &Z_TSR(R)->a;
    tsr_array *ia = (Z_TYPE(I) != IS_UNDEF) ? &Z_TSR(I)->a : NULL;
    int32_t nd;
    int64_t sh[32], rst[32], ist[32];
    if (ia) {
        if (tsr_broadcast_shape(ra->ndim, ra->shape, ia->ndim, ia->shape, &nd, sh) != 0) {
            zend_throw_exception(tsr_ce_shape_exception, "complex: real and imag shapes cannot be broadcast together", 0);
            zval_ptr_dtor(&R); zval_ptr_dtor(&I); RETURN_THROWS();
        }
    } else {
        nd = ra->ndim;
        for (int d = 0; d < nd; d++) sh[d] = ra->shape[d];
    }
    tsr_broadcast_strides(ra, nd, sh, rst);
    if (ia) tsr_broadcast_strides(ia, nd, sh, ist);
    if (tsr_ndarray_new(return_value, D_C128, nd, sh, 0) == FAILURE) { zval_ptr_dtor(&R); if (ia) zval_ptr_dtor(&I); RETURN_THROWS(); }
    tsr_obj *o = Z_TSR_P(return_value);
    double *out = (double *)elem0(&o->a);
    const int64_t n = tsr_array_size(&o->a);
    int64_t ix[32] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t ro = 0, io = 0;
        for (int d = 0; d < nd; d++) { ro += ix[d] * rst[d]; if (ia) io += ix[d] * ist[d]; }
        out[2 * q] = *(const double *)((const char *)elem0(ra) + ro);
        out[2 * q + 1] = ia ? *(const double *)((const char *)elem0(ia) + io) : 0.0;
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    zval_ptr_dtor(&R);
    if (ia) zval_ptr_dtor(&I);
}

PHP_METHOD(NDArray, shape)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    array_init_size(return_value, (uint32_t)o->a.ndim);
    for (int d = 0; d < o->a.ndim; d++) add_next_index_long(return_value, o->a.shape[d]);
}

PHP_METHOD(NDArray, strides)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    array_init_size(return_value, (uint32_t)o->a.ndim);
    for (int d = 0; d < o->a.ndim; d++) add_next_index_long(return_value, o->a.strides[d]);
}

PHP_METHOD(NDArray, ndim) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_LONG(THIS_OBJ->a.ndim); }
PHP_METHOD(NDArray, size) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_LONG(tsr_array_size(&THIS_OBJ->a)); }
PHP_METHOD(NDArray, dtype) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_STRING(tsr_dtype_name(THIS_OBJ->a.dtype)); }
PHP_METHOD(NDArray, isContiguous) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_BOOL(tsr_array_is_contiguous(&THIS_OBJ->a)); }
PHP_METHOD(NDArray, isView) { ZEND_PARSE_PARAMETERS_NONE(); RETURN_BOOL(THIS_OBJ->base != NULL); }

PHP_METHOD(NDArray, count)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zend_long n;
    if (tsr_count_elements(Z_OBJ_P(ZEND_THIS), &n) == FAILURE) RETURN_THROWS();
    RETURN_LONG(n);
}

PHP_METHOD(NDArray, getIterator)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zend_create_internal_iterator_zval(return_value, ZEND_THIS);
}

PHP_METHOD(NDArray, toArray)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    to_php(&o->a, 0, elem0(&o->a), return_value);
}

PHP_METHOD(NDArray, jsonSerialize)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    to_php(&o->a, 0, elem0(&o->a), return_value);
}

PHP_METHOD(NDArray, toList)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    tsr_array flat = o->a;
    flat.ndim = 1;
    zval C;
    ZVAL_UNDEF(&C);
    if (!tsr_array_is_contiguous(&o->a)) {
        if (cast_copy(&o->a, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
        flat = Z_TSR(C)->a;
        flat.ndim = 1;
    }
    flat.shape[0] = tsr_array_size(&o->a);
    flat.strides[0] = ITEMSIZE[o->a.dtype];
    to_php(&flat, 0, elem0(&flat), return_value);
    zval_ptr_dtor(&C);
}

PHP_METHOD(NDArray, toBytes)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    zval self, C;
    ZVAL_OBJ(&self, &o->std);
    if (tsr_ndarray_contiguous_as(&self, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
    RETVAL_STRINGL(elem0(&Z_TSR(C)->a), (size_t)(tsr_array_size(&o->a) * ITEMSIZE[o->a.dtype]));
    zval_ptr_dtor(&C);
}

/* JSON written straight from strided memory with PHP's own double formatting
   (serialize_precision, shortest round-trip by default): output is byte-identical
   to json_encode($a->toArray(), JSON_PRESERVE_ZERO_FRACTION). */
/* exactly php_json_encode_double() with JSON_PRESERVE_ZERO_FRACTION */
static void json_double(smart_str *b, double v, int prec)
{
    if (!zend_finite(v)) { smart_str_appendl(b, "null", 4); return; }
    char num[ZEND_DOUBLE_MAX_LENGTH];
    php_gcvt(v, prec, '.', 'e', num);
    size_t len = strlen(num);
    smart_str_appendl(b, num, len);
    if (strchr(num, '.') == NULL) smart_str_appendl(b, ".0", 2);
}

static void json_rec(smart_str *b, const tsr_array *a, int d, const char *p, int prec)
{
    if (d == a->ndim) {
        switch (a->dtype) {
        case D_F64: case D_F32: {
            json_double(b, a->dtype == D_F64 ? *(const double *)p : (double)*(const float *)p, prec);
            break;
        }
        case D_I64: smart_str_append_long(b, (zend_long)*(const int64_t *)p); break;
        case D_I32: smart_str_append_long(b, *(const int32_t *)p); break;
        case D_U8: smart_str_append_long(b, *(const uint8_t *)p); break;
        case D_BOOL: if (*(const uint8_t *)p) smart_str_appendl(b, "true", 4); else smart_str_appendl(b, "false", 5); break;
        case D_C128: {
            const double *c = (const double *)p;
            smart_str_appendc(b, '[');
            json_double(b, c[0], prec);
            smart_str_appendc(b, ',');
            json_double(b, c[1], prec);
            smart_str_appendc(b, ']');
            break;
        }
        }
        return;
    }
    smart_str_appendc(b, '[');
    for (int64_t i = 0; i < a->shape[d]; i++) {
        if (i) smart_str_appendc(b, ',');
        json_rec(b, a, d + 1, p + i * a->strides[d], prec);
    }
    smart_str_appendc(b, ']');
}

PHP_METHOD(NDArray, toJson)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    smart_str b = {0};
    int64_t n = tsr_array_size(&o->a);
    smart_str_alloc(&b, (size_t)(n > 0 ? n * 8 : 16), 0);
    json_rec(&b, &o->a, 0, elem0(&o->a), (int)PG(serialize_precision));
    smart_str_0(&b);
    RETURN_STR(b.s);
}

PHP_METHOD(NDArray, __toString)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    char shp[256];
    shape_str(shp, sizeof(shp), &o->a);
    smart_str buf = {0};
    smart_str_appends(&buf, "array(");
    if (tsr_array_size(&o->a) <= 1000) {
        int64_t need = tsr_array_json(&o->a, NULL, 0);
        char *tmp = emalloc((size_t)need + 1);
        tsr_array_json(&o->a, tmp, need);
        smart_str_appendl(&buf, tmp, (size_t)need);
        efree(tmp);
    } else {
        smart_str_appends(&buf, "...");
    }
    smart_str_appends(&buf, ", shape=");
    smart_str_appends(&buf, shp);
    smart_str_appends(&buf, ", dtype=");
    smart_str_appends(&buf, tsr_dtype_name(o->a.dtype));
    smart_str_appendc(&buf, ')');
    smart_str_0(&buf);
    RETURN_STR(buf.s);
}

PHP_METHOD(NDArray, item)
{
    zval *args = NULL;
    uint32_t argc = 0;
    ZEND_PARSE_PARAMETERS_START(0, -1)
        Z_PARAM_VARIADIC('*', args, argc)
    ZEND_PARSE_PARAMETERS_END();
    tsr_obj *o = THIS_OBJ;
    if (argc == 0) {
        if (tsr_array_size(&o->a) != 1) { zend_throw_exception(tsr_ce_exception, "item() without indices needs a one-element array", 0); RETURN_THROWS(); }
        elem_to_zval(o->a.dtype, elem0(&o->a), return_value);
        return;
    }
    if ((int)argc != o->a.ndim) { zend_throw_exception_ex(tsr_ce_index_exception, 0, "item() needs %d indices", o->a.ndim); RETURN_THROWS(); }
    int64_t idx[32];
    for (uint32_t i = 0; i < argc; i++) idx[i] = zval_get_long(&args[i]);
    void *p = tsr_array_at(&o->a, idx);
    if (!p) { zend_throw_exception(tsr_ce_index_exception, "item() index out of bounds", 0); RETURN_THROWS(); }
    elem_to_zval(o->a.dtype, p, return_value);
}

PHP_METHOD(NDArray, copy)
{
    ZEND_PARSE_PARAMETERS_NONE();
    cast_copy(&THIS_OBJ->a, THIS_OBJ->a.dtype, return_value);
}

PHP_METHOD(NDArray, astype)
{
    zend_string *dts;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(dts)
    ZEND_PARSE_PARAMETERS_END();
    int dt = dtype_arg(dts, -1);
    if (dt < 0) RETURN_THROWS();
    cast_copy(&THIS_OBJ->a, dt, return_value);
}

PHP_METHOD(NDArray, reshape)
{
    zval *args = NULL;
    uint32_t argc = 0;
    ZEND_PARSE_PARAMETERS_START(1, -1)
        Z_PARAM_VARIADIC('+', args, argc)
    ZEND_PARSE_PARAMETERS_END();
    int64_t shape[32];
    int32_t nd;
    if (shape_from_args(args, argc, shape, &nd) == FAILURE) RETURN_THROWS();
    tsr_obj *o = THIS_OBJ;
    tsr_array v;
    int rc = tsr_array_reshape(&o->a, nd, shape, &v);
    if (rc == -7) {
        zval C;
        if (cast_copy(&o->a, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
        rc = tsr_array_reshape(&Z_TSR(C)->a, nd, shape, &v);
        if (rc == 0) make_view(return_value, Z_TSR(C), &v);
        zval_ptr_dtor(&C);
    } else if (rc == 0) {
        make_view(return_value, o, &v);
    }
    if (rc != 0) {
        char s[256];
        shape_str(s, sizeof(s), &o->a);
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "Cannot reshape an array of shape %s into the requested shape", s);
        RETURN_THROWS();
    }
}

PHP_METHOD(NDArray, ravel)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    int64_t shape[1] = {-1};
    tsr_array v;
    if (tsr_array_reshape(&o->a, 1, shape, &v) == 0) { make_view(return_value, o, &v); return; }
    zval C;
    if (cast_copy(&o->a, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
    tsr_array_reshape(&Z_TSR(C)->a, 1, shape, &v);
    make_view(return_value, Z_TSR(C), &v);
    zval_ptr_dtor(&C);
}

PHP_METHOD(NDArray, transpose)
{
    HashTable *axes = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ARRAY_HT_OR_NULL(axes)
    ZEND_PARSE_PARAMETERS_END();
    tsr_obj *o = THIS_OBJ;
    int32_t ax[32];
    if (axes) {
        if ((int)zend_hash_num_elements(axes) != o->a.ndim) { zend_throw_exception(tsr_ce_shape_exception, "transpose() needs one axis per dimension", 0); RETURN_THROWS(); }
        int k = 0;
        zval *e;
        ZEND_HASH_FOREACH_VAL(axes, e) { ax[k++] = (int32_t)zval_get_long(e); } ZEND_HASH_FOREACH_END();
    }
    tsr_array v;
    if (tsr_array_transpose(&o->a, axes ? ax : NULL, &v) != 0) { zend_throw_exception(tsr_ce_shape_exception, "transpose() axes must be a permutation", 0); RETURN_THROWS(); }
    make_view(return_value, o, &v);
}

PHP_METHOD(NDArray, t)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_array v;
    tsr_array_transpose(&THIS_OBJ->a, NULL, &v);
    make_view(return_value, THIS_OBJ, &v);
}

static int dtype_bytes(int dt)
{
    switch (dt) {
    case D_F64: case D_I64: return 8;
    case D_F32: case D_I32: return 4;
    case D_U8:  case D_BOOL: return 1;
    case D_C128: return 16;
    default: return 0;
    }
}

/* A contiguous 1-D copy (numpy.ndarray.flatten always copies, unlike ravel). */
PHP_METHOD(NDArray, flatten)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    zval C;
    if (cast_copy(&o->a, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
    int64_t shape[1] = {-1};
    tsr_array v;
    tsr_array_reshape(&Z_TSR(C)->a, 1, shape, &v);
    make_view(return_value, Z_TSR(C), &v);
    zval_ptr_dtor(&C);
}

/* A view with the two axes exchanged (numpy.ndarray.swapaxes). */
PHP_METHOD(NDArray, swapAxes)
{
    zend_long a, b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_LONG(a)
        Z_PARAM_LONG(b)
    ZEND_PARSE_PARAMETERS_END();
    tsr_obj *o = THIS_OBJ;
    const int nd = o->a.ndim;
    if (a < 0) a += nd;
    if (b < 0) b += nd;
    if (a < 0 || a >= nd || b < 0 || b >= nd) { zend_throw_exception(tsr_ce_shape_exception, "swapAxes(): axis out of range", 0); RETURN_THROWS(); }
    int32_t ax[32];
    for (int i = 0; i < nd; i++) ax[i] = i;
    int32_t t = ax[a]; ax[a] = ax[b]; ax[b] = t;
    tsr_array v;
    tsr_array_transpose(&o->a, ax, &v);
    make_view(return_value, o, &v);
}

/* A view (or copy, when non-contiguous) with length-1 axes removed (numpy.ndarray.squeeze). */
PHP_METHOD(NDArray, squeeze)
{
    zval *axisz = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(axisz)
    ZEND_PARSE_PARAMETERS_END();
    tsr_obj *o = THIS_OBJ;
    const int nd = o->a.ndim;
    int64_t shape[32];
    int k = 0;
    if (axisz && Z_TYPE_P(axisz) != IS_NULL) {
        zend_long ax = zval_get_long(axisz);
        if (ax < 0) ax += nd;
        if (ax < 0 || ax >= nd) { zend_throw_exception(tsr_ce_shape_exception, "squeeze(): axis out of range", 0); RETURN_THROWS(); }
        if (o->a.shape[ax] != 1) { zend_throw_exception(tsr_ce_shape_exception, "cannot select an axis to squeeze out which has size not equal to one", 0); RETURN_THROWS(); }
        for (int d = 0; d < nd; d++) if (d != (int)ax) shape[k++] = o->a.shape[d];
    } else {
        for (int d = 0; d < nd; d++) if (o->a.shape[d] != 1) shape[k++] = o->a.shape[d];
    }
    tsr_array v;
    if (tsr_array_reshape(&o->a, k, shape, &v) == 0) { make_view(return_value, o, &v); return; }
    zval C;
    if (cast_copy(&o->a, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
    tsr_array_reshape(&Z_TSR(C)->a, k, shape, &v);
    make_view(return_value, Z_TSR(C), &v);
    zval_ptr_dtor(&C);
}

/* Total bytes consumed by the array's elements (numpy.ndarray.nbytes). */
PHP_METHOD(NDArray, nbytes)
{
    ZEND_PARSE_PARAMETERS_NONE();
    RETURN_LONG(tsr_array_size(&THIS_OBJ->a) * dtype_bytes(THIS_OBJ->a.dtype));
}

/* Reinterpret the same bytes as another dtype of equal itemsize (numpy.ndarray.view). */
PHP_METHOD(NDArray, view)
{
    zend_string *dts;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(dts)
    ZEND_PARSE_PARAMETERS_END();
    int dt = dtype_arg(dts, -1);
    if (dt < 0) RETURN_THROWS();
    tsr_obj *o = THIS_OBJ;
    if (dtype_bytes(dt) != dtype_bytes(o->a.dtype)) { zend_throw_exception(tsr_ce_dtype_exception, "view() needs a dtype with the same itemsize.", 0); RETURN_THROWS(); }
    tsr_array v = o->a;
    v.dtype = dt;
    make_view(return_value, o, &v);
}

/* Round to the given number of decimals, half to even (numpy.ndarray.round -> np.round). */
PHP_METHOD(NDArray, round)
{
    zend_long dec = 0;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(dec)
    ZEND_PARSE_PARAMETERS_END();
    zval args[2];
    ZVAL_COPY_VALUE(&args[0], ZEND_THIS);
    ZVAL_LONG(&args[1], dec);
    tsr_fn_call_routine("np.round", args, 2, return_value);
    if (EG(exception)) RETURN_THROWS();
}

/* Gather elements at the given indices along an axis, or from the flattened array (numpy.ndarray.take -> np.take). */
PHP_METHOD(NDArray, take)
{
    zval *indices, *axis = NULL;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(indices)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(axis)
    ZEND_PARSE_PARAMETERS_END();
    zval args[3], axnull;
    ZVAL_COPY_VALUE(&args[0], ZEND_THIS);
    ZVAL_COPY_VALUE(&args[1], indices);
    if (axis) ZVAL_COPY_VALUE(&args[2], axis);
    else { ZVAL_NULL(&axnull); ZVAL_COPY_VALUE(&args[2], &axnull); }
    tsr_fn_call_routine("np.take", args, 3, return_value);
    if (EG(exception)) RETURN_THROWS();
}

/* Write values at the given flat indices, in place (numpy.ndarray.put -> np.put). */
PHP_METHOD(NDArray, put)
{
    zval *indices, *values, *axis = NULL;
    ZEND_PARSE_PARAMETERS_START(2, 3)
        Z_PARAM_ZVAL(indices)
        Z_PARAM_ZVAL(values)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(axis)
    ZEND_PARSE_PARAMETERS_END();
    (void)axis;   /* np.put is flat, matching the FFI backend's flat put */
    zval args[3], rv;
    ZVAL_COPY_VALUE(&args[0], ZEND_THIS);
    ZVAL_COPY_VALUE(&args[1], indices);
    ZVAL_COPY_VALUE(&args[2], values);
    ZVAL_UNDEF(&rv);
    tsr_fn_call_routine("np.put", args, 3, &rv);
    zval_ptr_dtor(&rv);
    if (EG(exception)) RETURN_THROWS();
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

PHP_METHOD(NDArray, slice)
{
    zend_string *spec;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_STR(spec)
    ZEND_PARSE_PARAMETERS_END();
    slice_view(THIS_OBJ, ZSTR_VAL(spec), return_value, 0);
}

PHP_METHOD(NDArray, assign)
{
    zval *value;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(value)
    ZEND_PARSE_PARAMETERS_END();
    if (tsr_require_writable(THIS_OBJ, "assign") == FAILURE) RETURN_THROWS();
    if (assign_core(&THIS_OBJ->a, value) == FAILURE) RETURN_THROWS();
    RETURN_OBJ_COPY(Z_OBJ_P(ZEND_THIS));
}

/* ---- binary operations: one handler, the method name picks the op ---- */

static const struct { const char *name; int op; } BIN_NAMES[] = {
    {"add", OP_ADD}, {"sub", OP_SUB}, {"mul", OP_MUL}, {"div", OP_DIV}, {"pow", OP_POW}, {"mod", OP_MOD},
    {"floordiv", OP_FLOORDIV}, {"maximum", OP_MAX}, {"minimum", OP_MIN}, {"atan2", OP_ATAN2}, {"hypot", OP_HYPOT},
    {"eq", OP_EQ}, {"ne", OP_NE}, {"lt", OP_LT}, {"le", OP_LE}, {"gt", OP_GT}, {"ge", OP_GE},
    {"logicaland", OP_AND}, {"logicalor", OP_OR}, {"logicalxor", OP_XOR},
};

static int lookup_op(zend_execute_data *execute_data, const void *table, size_t count, size_t stride)
{
    zend_string *fn = EX(func)->common.function_name;
    for (size_t i = 0; i < count; i++) {
        const char *name = *(const char *const *)((const char *)table + i * stride);
        if (strlen(name) == ZSTR_LEN(fn) && strncasecmp(name, ZSTR_VAL(fn), ZSTR_LEN(fn)) == 0)
            return *(const int *)((const char *)table + i * stride + sizeof(const char *));
    }
    return -1;
}

PHP_METHOD(NDArray, binaryOp)
{
    zval *other;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other)
    ZEND_PARSE_PARAMETERS_END();
    int op = lookup_op(execute_data, BIN_NAMES, sizeof(BIN_NAMES) / sizeof(BIN_NAMES[0]), sizeof(BIN_NAMES[0]));
    binary_core(op, ZEND_THIS, other, return_value);
}

PHP_METHOD(NDArray, reflectedOp)
{
    zval *other;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other)
    ZEND_PARSE_PARAMETERS_END();
    zend_string *fn = EX(func)->common.function_name;
    int op = zend_string_equals_literal_ci(fn, "rsub") ? OP_SUB : zend_string_equals_literal_ci(fn, "rdiv") ? OP_DIV : OP_POW;
    binary_core(op, other, ZEND_THIS, return_value);
}

static const struct { const char *name; int op; } UN_NAMES[] = {
    {"neg", U_NEG}, {"abs", U_ABS}, {"square", U_SQUARE}, {"sign", U_SIGN}, {"sqrt", U_SQRT}, {"exp", U_EXP},
    {"log", U_LOG}, {"log10", U_LOG10}, {"log2", U_LOG2}, {"sin", U_SIN}, {"cos", U_COS}, {"tan", U_TAN},
    {"arcsin", U_ARCSIN}, {"arccos", U_ARCCOS}, {"arctan", U_ARCTAN}, {"sinh", U_SINH}, {"cosh", U_COSH},
    {"tanh", U_TANH}, {"floor", U_FLOOR}, {"ceil", U_CEIL}, {"rint", U_RINT}, {"expm1", U_EXPM1},
    {"log1p", U_LOG1P}, {"reciprocal", U_RECIPROCAL}, {"isnan", U_ISNAN}, {"isfinite", U_ISFINITE},
    {"isinf", U_ISINF}, {"invert", U_NOT}, {"real", U_REAL}, {"imag", U_IMAG}, {"conj", U_CONJ}, {"angle", U_ANGLE},
};

/* op applied to the array `self` (an NDArray zval), into dst when given. */
static int unary_dispatch(int op, zval *self, tsr_obj *dst, zval *rv)
{
    tsr_obj *o = Z_TSR_P(self);
    if (op == U_ABS && o->a.dtype == D_C128) op = U_CABS;
    if (o->a.dtype != D_C128 && (op == U_REAL || op == U_IMAG || op == U_CONJ || op == U_ANGLE)) {
        zval T;
        ZVAL_UNDEF(&T);
        int r;
        if (op == U_IMAG) {
            r = tsr_ndarray_new(&T, to_float(o->a.dtype), o->a.ndim, o->a.shape, 1);
        } else if (op == U_ANGLE) {
            zval z, lt, pi;
            ZVAL_LONG(&z, 0);
            if (binary_core(OP_LT, self, &z, &lt) == FAILURE) return FAILURE;
            ZVAL_DOUBLE(&pi, M_PI);
            r = binary_core(OP_MUL, &lt, &pi, &T);
            zval_ptr_dtor(&lt);
        } else {
            r = cast_copy(&o->a, op == U_REAL ? to_float(o->a.dtype) : o->a.dtype, &T);
        }
        if (r == FAILURE) return FAILURE;
        if (!dst) { ZVAL_COPY_VALUE(rv, &T); return SUCCESS; }
        r = cast_into(&Z_TSR(T)->a, dst, "out");
        zval_ptr_dtor(&T);
        if (r == FAILURE) return FAILURE;
        ZVAL_OBJ_COPY(rv, &dst->std);
        return SUCCESS;
    }
    return unary_core_into(op, o, dst, rv);
}

PHP_METHOD(NDArray, unaryOp)
{
    ZEND_PARSE_PARAMETERS_NONE();
    int op = lookup_op(execute_data, UN_NAMES, sizeof(UN_NAMES) / sizeof(UN_NAMES[0]), sizeof(UN_NAMES[0]));
    unary_dispatch(op, ZEND_THIS, NULL, return_value);
}

PHP_METHOD(NDArray, logicalNot)
{
    ZEND_PARSE_PARAMETERS_NONE();
    zval B;
    if (cast_copy(&THIS_OBJ->a, D_BOOL, &B) == FAILURE) RETURN_THROWS();
    unary_core(U_NOT, Z_TSR(B), return_value);
    zval_ptr_dtor(&B);
}

PHP_METHOD(NDArray, clip)
{
    zval *lo = NULL, *hi = NULL;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_NUMBER_OR_NULL(lo)
        Z_PARAM_NUMBER_OR_NULL(hi)
    ZEND_PARSE_PARAMETERS_END();
    zval cur, next;
    ZVAL_COPY(&cur, ZEND_THIS);
    if (lo) { if (binary_core(OP_MAX, &cur, lo, &next) == FAILURE) { zval_ptr_dtor(&cur); RETURN_THROWS(); } zval_ptr_dtor(&cur); cur = next; }
    if (hi) { if (binary_core(OP_MIN, &cur, hi, &next) == FAILURE) { zval_ptr_dtor(&cur); RETURN_THROWS(); } zval_ptr_dtor(&cur); cur = next; }
    if (!lo && !hi) { zval_ptr_dtor(&cur); cast_copy(&THIS_OBJ->a, THIS_OBJ->a.dtype, return_value); return; }
    RETURN_COPY_VALUE(&cur);
}

/* ---- reductions ---- */

static const struct { const char *name; int op; } RED_NAMES[] = {
    {"sum", R_SUM}, {"prod", R_PROD}, {"min", R_MIN}, {"max", R_MAX}, {"argmin", R_ARGMIN}, {"argmax", R_ARGMAX},
    {"any", R_ANY}, {"all", R_ALL},
};

/*
 * Axis arguments: null (all axes, flattened), an int, or a list of ints. The
 * list is normalised (negative axes count from the end), checked for range
 * and duplicates, and sorted in descending order. Returns the count, or -1
 * for null; FAILURE (-2) with an exception set.
 */
static int parse_axes(zval *z, int ndim, int32_t *axes, int multi_ok, const char *what)
{
    if (z == NULL || Z_TYPE_P(z) == IS_NULL) return -1;
    int n = 0;
    zval one, *zv;
    if (Z_TYPE_P(z) == IS_LONG) {
        ZVAL_COPY_VALUE(&one, z);
        z = &one;
    } else if (Z_TYPE_P(z) != IS_ARRAY) {
        zend_argument_type_error(1, "must be of type array|int|null, %s given", zend_zval_type_name(z));
        return -2;
    }
    if (Z_TYPE_P(z) == IS_LONG) {
        axes[n++] = (int32_t)Z_LVAL_P(z);
        zend_long ax = Z_LVAL_P(z);
        if (ax < 0) ax += ndim;
        if (ndim == 0 || ax < 0 || ax >= ndim) {
            zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s: axis " ZEND_LONG_FMT " is out of bounds for a %d-d array", what, Z_LVAL_P(z), ndim);
            return -2;
        }
        axes[0] = (int32_t)ax;
        return 1;
    }
    ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(z), zv) {
        ZVAL_DEREF(zv);
        if (Z_TYPE_P(zv) != IS_LONG) { zend_argument_type_error(1, "must contain only integers"); return -2; }
        zend_long ax = Z_LVAL_P(zv);
        if (ax < 0) ax += ndim;
        if (ax < 0 || ax >= ndim || n >= 32) {
            zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s: axis " ZEND_LONG_FMT " is out of bounds for a %d-d array", what, Z_LVAL_P(zv), ndim);
            return -2;
        }
        for (int i = 0; i < n; i++)
            if (axes[i] == ax) { zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s: duplicate axis in reduction", what); return -2; }
        axes[n++] = (int32_t)ax;
    } ZEND_HASH_FOREACH_END();
    if (n > 1 && !multi_ok) { zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s takes a single axis", what); return -2; }
    for (int i = 1; i < n; i++)                        /* insertion sort, descending */
        for (int j = i; j > 0 && axes[j] > axes[j - 1]; j--) { int32_t t = axes[j]; axes[j] = axes[j - 1]; axes[j - 1] = t; }
    return n;
}

/* Give a fresh C-contiguous result the input's rank: reduced axes become length 1 (keepdims). */
static void keep_dims(zval *rv, int ndim, const int64_t *shape, const int32_t *axes, int count)
{
    if (Z_TYPE_P(rv) != IS_OBJECT) return;
    tsr_array *r = &Z_TSR_P(rv)->a;
    int64_t ks[32];
    for (int d = 0; d < ndim; d++) ks[d] = shape[d];
    if (count < 0) for (int d = 0; d < ndim; d++) ks[d] = 1;
    else for (int i = 0; i < count; i++) ks[axes[i]] = 1;
    int64_t st = ITEMSIZE[r->dtype];
    for (int d = ndim - 1; d >= 0; d--) { r->strides[d] = st; st *= ks[d] > 0 ? ks[d] : 1; }
    memcpy(r->shape, ks, sizeof(int64_t) * (size_t)ndim);
    r->ndim = ndim;
}

/* The array flattened to 1-D (a contiguous copy when needed), as a new object in out. */
static int flat_view(tsr_obj *src, zval *out)
{
    if (cast_copy(&src->a, src->a.dtype, out) == FAILURE) return FAILURE;
    tsr_array *f = &Z_TSR_P(out)->a;
    f->shape[0] = tsr_array_size(f);
    f->strides[0] = ITEMSIZE[f->dtype];
    f->ndim = 1;
    return SUCCESS;
}

/* sum/prod/min/max/argmin/argmax/any/all(int|array|null $axis = null, bool $keepdims = false):
   several axes are reduced one at a time from the last, as the FFI package does, so both backends agree to the bit. */
PHP_METHOD(NDArray, reduceOp)
{
    zval *axis_zv = NULL;
    bool keepdims = 0;
    ZEND_PARSE_PARAMETERS_START(0, 2)
        Z_PARAM_OPTIONAL
        Z_PARAM_ZVAL_OR_NULL(axis_zv)
        Z_PARAM_BOOL(keepdims)
    ZEND_PARSE_PARAMETERS_END();
    int op = lookup_op(execute_data, RED_NAMES, sizeof(RED_NAMES) / sizeof(RED_NAMES[0]), sizeof(RED_NAMES[0]));
    tsr_obj *o = THIS_OBJ;
    int32_t axes[32];
    const int n = parse_axes(axis_zv, o->a.ndim, axes, op != R_ARGMIN && op != R_ARGMAX, "reduce");
    if (n == -2) RETURN_THROWS();
    const int ndim = o->a.ndim;
    int64_t shape[32];
    memcpy(shape, o->a.shape, sizeof(int64_t) * (size_t)ndim);
    if (n < 0) {
        if (!keepdims) { reduce_core(op, o, 0, 0, return_value); return; }
        zval F;
        if (flat_view(o, &F) == FAILURE) RETURN_THROWS();
        const int r = reduce_core(op, Z_TSR(F), 0, 1, return_value);
        zval_ptr_dtor(&F);
        if (r == FAILURE) RETURN_THROWS();
        keep_dims(return_value, ndim, shape, axes, -1);
        return;
    }
    if (n == 0) { cast_copy(&o->a, o->a.dtype, return_value); return; }
    zval cur, next;
    ZVAL_OBJ_COPY(&cur, &o->std);
    for (int i = 0; i < n; i++) {
        if (reduce_core(op, Z_TSR(cur), axes[i], 1, &next) == FAILURE) { zval_ptr_dtor(&cur); RETURN_THROWS(); }
        zval_ptr_dtor(&cur);
        cur = next;
    }
    RETVAL_COPY_VALUE(&cur);
    if (keepdims) keep_dims(return_value, ndim, shape, axes, n);
}

/* mean / var / std over null, one or several axes (several: moved to the end and merged, as the FFI package does). */
static void moments_method(INTERNAL_FUNCTION_PARAMETERS, int var, int sq)
{
    zval *axis_zv = NULL;
    zend_long ddof = 0;
    bool keepdims = 0;
    if (var) {
        ZEND_PARSE_PARAMETERS_START(0, 3)
            Z_PARAM_OPTIONAL
            Z_PARAM_ZVAL_OR_NULL(axis_zv)
            Z_PARAM_LONG(ddof)
            Z_PARAM_BOOL(keepdims)
        ZEND_PARSE_PARAMETERS_END();
    } else {
        ZEND_PARSE_PARAMETERS_START(0, 2)
            Z_PARAM_OPTIONAL
            Z_PARAM_ZVAL_OR_NULL(axis_zv)
            Z_PARAM_BOOL(keepdims)
        ZEND_PARSE_PARAMETERS_END();
    }
    tsr_obj *o = THIS_OBJ;
    int32_t axes[32];
    const int n = parse_axes(axis_zv, o->a.ndim, axes, 1, var ? (sq ? "std" : "var") : "mean");
    if (n == -2) RETURN_THROWS();
    const int ndim = o->a.ndim;
    int64_t shape[32];
    memcpy(shape, o->a.shape, sizeof(int64_t) * (size_t)ndim);
    if (n < 0 && !keepdims) { moments_core(o, 0, 0, var, ddof, sq, return_value); return; }
    if (n == 1) {
        if (moments_core(o, axes[0], 1, var, ddof, sq, return_value) == FAILURE) RETURN_THROWS();
        if (keepdims) keep_dims(return_value, ndim, shape, axes, 1);
        return;
    }
    zval C;
    if (n < 0) {
        if (flat_view(o, &C) == FAILURE) RETURN_THROWS();
        const int r = moments_core(Z_TSR(C), 0, 1, var, ddof, sq, return_value);
        zval_ptr_dtor(&C);
        if (r == FAILURE) RETURN_THROWS();
        keep_dims(return_value, ndim, shape, axes, -1);
        return;
    }
    /* several axes: transpose to (kept..., reduced...), copy contiguous, merge the reduced axes */
    int32_t perm[32];
    int nk = 0;
    for (int d = 0; d < ndim; d++) {
        int red = 0;
        for (int i = 0; i < n; i++) if (axes[i] == d) red = 1;
        if (!red) perm[nk++] = d;
    }
    for (int i = n - 1; i >= 0; i--) perm[nk + (n - 1 - i)] = axes[i];   /* ascending order, as array_diff + axes */
    tsr_array t;
    if (tsr_array_transpose(&o->a, perm, &t) != 0) { tsr_throw_rc(-1, "moments"); RETURN_THROWS(); }
    if (cast_copy(&t, t.dtype, &C) == FAILURE) RETURN_THROWS();
    tsr_array *c = &Z_TSR(C)->a;
    int64_t merged = 1;
    for (int d = nk; d < ndim; d++) merged *= c->shape[d];
    c->shape[nk] = merged;
    c->ndim = nk + 1;
    int64_t st = ITEMSIZE[c->dtype];
    for (int d = c->ndim - 1; d >= 0; d--) { c->strides[d] = st; st *= c->shape[d] > 0 ? c->shape[d] : 1; }
    const int r = moments_core(Z_TSR(C), nk, 1, var, ddof, sq, return_value);
    zval_ptr_dtor(&C);
    if (r == FAILURE) RETURN_THROWS();
    if (keepdims) keep_dims(return_value, ndim, shape, axes, n);
}

PHP_METHOD(NDArray, mean) { moments_method(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0, 0); }
PHP_METHOD(NDArray, var) { moments_method(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1, 0); }
PHP_METHOD(NDArray, std) { moments_method(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1, 1); }

static void cumulative(INTERNAL_FUNCTION_PARAMETERS, int op)
{
    zend_long axis = 0;
    zend_bool axis_null = 1;
    ZEND_PARSE_PARAMETERS_START(0, 1)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG_OR_NULL(axis, axis_null)
    ZEND_PARSE_PARAMETERS_END();
    tsr_obj *o = THIS_OBJ;
    if (o->a.dtype == D_C128) { zend_throw_exception(tsr_ce_dtype_exception, "cumsum/cumprod of complex arrays is not supported", 0); RETURN_THROWS(); }
    zval C;
    if (cast_copy(&o->a, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
    tsr_array a = Z_TSR(C)->a;
    if (axis_null) { a.shape[0] = tsr_array_size(&a); a.ndim = 1; axis = 0; }
    if (axis < 0) axis += a.ndim;
    if (a.ndim == 0 || axis < 0 || axis >= a.ndim) { zval_ptr_dtor(&C); zend_throw_exception(tsr_ce_shape_exception, "axis out of range", 0); RETURN_THROWS(); }
    int64_t outer = 1, inner = 1;
    for (int d = 0; d < axis; d++) outer *= a.shape[d];
    for (int d = (int)axis + 1; d < a.ndim; d++) inner *= a.shape[d];
    int outdt = kind(a.dtype) == 2 ? a.dtype : D_I64;
    if (tsr_ndarray_new(return_value, outdt, a.ndim, a.shape, 0) == FAILURE) { zval_ptr_dtor(&C); RETURN_THROWS(); }
    if (tsr_array_size(&a) > 0) tsr_cumulative(op, a.dtype, outer, a.shape[axis], inner, elem0(&a), elem0(&Z_TSR_P(return_value)->a));
    zval_ptr_dtor(&C);
}
PHP_METHOD(NDArray, cumsum) { cumulative(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0); }
PHP_METHOD(NDArray, cumprod) { cumulative(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1); }

/* ---- sorting (last axis) ---- */

static void sort_impl(INTERNAL_FUNCTION_PARAMETERS, int arg)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    if (o->a.dtype == D_C128) { zend_throw_exception(tsr_ce_dtype_exception, "Sorting complex arrays is not supported", 0); RETURN_THROWS(); }
    if (o->a.ndim == 0) { cast_copy(&o->a, o->a.dtype, return_value); return; }
    zval C;
    if (cast_copy(&o->a, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
    tsr_array *a = &Z_TSR(C)->a;
    int64_t len = a->shape[a->ndim - 1];
    int64_t rows = len ? tsr_array_size(a) / len : 0;
    int dt = a->dtype == D_BOOL ? D_U8 : a->dtype;
    if (arg) {
        if (tsr_ndarray_new(return_value, D_I64, a->ndim, a->shape, 0) == FAILURE) { zval_ptr_dtor(&C); RETURN_THROWS(); }
        int64_t *out = (int64_t *)elem0(&Z_TSR_P(return_value)->a);
        int rc = 0;
        for (int64_t r = 0; r < rows && rc == 0; r++) rc = tsr_argsort(dt, len, elem0(a) + r * len * ITEMSIZE[dt], out + r * len);
        zval_ptr_dtor(&C);
        /* a failed pass (over the memory budget) must not return a half-sorted result */
        if (rc != 0) { zval_ptr_dtor(return_value); ZVAL_UNDEF(return_value); tsr_throw_rc(rc, "argsort"); RETURN_THROWS(); }
        return;
    }
    int rc = 0;
    for (int64_t r = 0; r < rows && rc == 0; r++) rc = tsr_sort(dt, len, elem0(a) + r * len * ITEMSIZE[dt]);
    if (rc != 0) { zval_ptr_dtor(&C); tsr_throw_rc(rc, "sort"); RETURN_THROWS(); }
    RETURN_COPY_VALUE(&C);
}
PHP_METHOD(NDArray, sort) { sort_impl(INTERNAL_FUNCTION_PARAM_PASSTHRU, 0); }
PHP_METHOD(NDArray, argsort) { sort_impl(INTERNAL_FUNCTION_PARAM_PASSTHRU, 1); }

/* ---- matrix product ---- */

PHP_METHOD(NDArray, matmul)
{
    zval *other;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(other)
    ZEND_PARSE_PARAMETERS_END();
    zval B, Ac, Bc;
    ZVAL_UNDEF(&B); ZVAL_UNDEF(&Ac); ZVAL_UNDEF(&Bc);
    if (tsr_ndarray_from_zval(other, -1, &B) == FAILURE) RETURN_THROWS();
    tsr_obj *a = THIS_OBJ, *b = Z_TSR(B);
    if (a->a.ndim < 1 || a->a.ndim > 2 || b->a.ndim < 1 || b->a.ndim > 2) {
        zend_throw_exception(tsr_ce_shape_exception, "matmul supports 1-D and 2-D operands", 0);
        goto out;
    }
    int dt = promote(a->a.dtype, b->a.dtype);
    if (dt == D_C128) { zend_throw_exception(tsr_ce_dtype_exception, "matmul of complex arrays is not supported", 0); goto out; }
    int opdt = (dt == D_F64 || dt == D_F32 || dt == D_I64) ? dt : D_I64;
    int64_t m = a->a.ndim == 2 ? a->a.shape[0] : 1, k = a->a.shape[a->a.ndim - 1];
    int64_t k2 = b->a.ndim == 2 ? b->a.shape[0] : b->a.shape[0], n = b->a.ndim == 2 ? b->a.shape[1] : 1;
    if (k != k2) {
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "matmul: inner dimensions differ (%lld vs %lld)", (long long)k, (long long)k2);
        goto out;
    }
    if (tsr_ndarray_contiguous_as(ZEND_THIS, opdt, &Ac) == FAILURE || tsr_ndarray_contiguous_as(&B, opdt, &Bc) == FAILURE) goto out;
    {
        int64_t shape[2];
        int nd = 0;
        if (a->a.ndim == 2) shape[nd++] = m;
        if (b->a.ndim == 2) shape[nd++] = n;
        zval R;
        if (tsr_ndarray_new(&R, opdt, nd, shape, 1) == FAILURE) goto out;
        if (m > 0 && n > 0 && k > 0)
            tsr_matmul(opdt, m, n, k, elem0(&Z_TSR(Ac)->a), k, elem0(&Z_TSR(Bc)->a), n, elem0(&Z_TSR(R)->a), n);
        if (nd == 0) { elem_to_zval(opdt, elem0(&Z_TSR(R)->a), return_value); zval_ptr_dtor(&R); }
        else if (opdt != dt && dt != D_BOOL) { cast_copy(&Z_TSR(R)->a, dt, return_value); zval_ptr_dtor(&R); }
        else ZVAL_COPY_VALUE(return_value, &R);
    }
out:
    zval_ptr_dtor(&B); zval_ptr_dtor(&Ac); zval_ptr_dtor(&Bc);
}

/* ---- serialization (queues, cache, sessions) ---- */

PHP_METHOD(NDArray, __serialize)
{
    ZEND_PARSE_PARAMETERS_NONE();
    tsr_obj *o = THIS_OBJ;
    zval self, C, shp;
    ZVAL_OBJ(&self, &o->std);
    if (tsr_ndarray_contiguous_as(&self, o->a.dtype, &C) == FAILURE) RETURN_THROWS();
    array_init_size(return_value, 4);
    add_assoc_long(return_value, "v", 1);
    add_assoc_string(return_value, "dtype", tsr_dtype_name(o->a.dtype));
    array_init(&shp);
    for (int d = 0; d < o->a.ndim; d++) add_next_index_long(&shp, o->a.shape[d]);
    add_assoc_zval(return_value, "shape", &shp);
    add_assoc_stringl(return_value, "data", elem0(&Z_TSR(C)->a), (size_t)(tsr_array_size(&o->a) * ITEMSIZE[o->a.dtype]));
    zval_ptr_dtor(&C);
}

PHP_METHOD(NDArray, __unserialize)
{
    HashTable *data;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ARRAY_HT(data)
    ZEND_PARSE_PARAMETERS_END();
    zval *dtz = zend_hash_str_find(data, "dtype", 5), *shz = zend_hash_str_find(data, "shape", 5), *bz = zend_hash_str_find(data, "data", 4);
    if (!dtz || !shz || !bz || Z_TYPE_P(dtz) != IS_STRING || Z_TYPE_P(shz) != IS_ARRAY || Z_TYPE_P(bz) != IS_STRING) {
        zend_throw_exception(tsr_ce_exception, "Invalid serialized NDArray", 0);
        RETURN_THROWS();
    }
    int dt = tsr_dtype_from_name(Z_STRVAL_P(dtz), Z_STRLEN_P(dtz));
    int64_t shape[32];
    int32_t nd;
    if (dt < 0 || shape_from_args(shz, 1, shape, &nd) == FAILURE) {
        if (!EG(exception)) zend_throw_exception(tsr_ce_exception, "Invalid serialized NDArray", 0);
        RETURN_THROWS();
    }
    /* serialized data may come from a cache or a queue an attacker can write: validate before allocating */
    const int64_t total = checked_count(nd, shape);
    if (total < 0 || total * ITEMSIZE[dt] != (int64_t)Z_STRLEN_P(bz)) { zend_throw_exception(tsr_ce_exception, "Serialized NDArray data length mismatch", 0); RETURN_THROWS(); }
    tsr_obj *o = THIS_OBJ;
    if (o->block || o->base || o->map) { zend_throw_exception(tsr_ce_exception, "Cannot unserialize into an initialised array", 0); RETURN_THROWS(); }
    int64_t bytes = total * ITEMSIZE[dt];
    if (bytes < 64) bytes = 64;
    void *block = tsr_alloc(bytes);
    if (!block) { tsr_throw_rc(-3, "unserialize"); RETURN_THROWS(); }
    if (tsr_array_init(&o->a, dt, nd, shape, block) != 0) {
        tsr_free(block, bytes);
        zend_throw_exception(tsr_ce_exception, "Invalid serialized NDArray", 0);
        RETURN_THROWS();
    }
    memcpy(block, Z_STRVAL_P(bz), Z_STRLEN_P(bz));
    o->block = block;
    o->block_bytes = bytes;
}

/* ================================================================ shared with Math (ufuncs) and memmap */

void tsr_shape_str(char *buf, size_t cap, const tsr_array *a) { shape_str(buf, cap, a); }

int tsr_can_cast_same_kind(int from, int to)
{
    if (from == to) return 1;
    int kf = kind(from), kt = kind(to);
    if (kf != kt) return kf < kt;
    return 1;   /* within a kind (float64 -> float32, int64 -> uint8 ...) as NumPy's same_kind */
}

/* Broadcast src into dst (shape must broadcast to dst's), casting same-kind or safer. */
static int cast_into(const tsr_array *src, tsr_obj *dst, const char *what)
{
    if (tsr_require_writable(dst, what) == FAILURE) return FAILURE;
    if (!tsr_can_cast_same_kind(src->dtype, dst->a.dtype)) {
        zend_throw_exception_ex(tsr_ce_dtype_exception, 0, "%s: cannot cast the result from %s to %s", what,
                                tsr_dtype_name(src->dtype), tsr_dtype_name(dst->a.dtype));
        return FAILURE;
    }
    int64_t st[32];
    int rc = tsr_broadcast_strides(src, dst->a.ndim, dst->a.shape, st);
    if (rc != 0) {
        char s1[160], s2[160];
        shape_str(s1, sizeof(s1), src);
        shape_str(s2, sizeof(s2), &dst->a);
        zend_throw_exception_ex(tsr_ce_shape_exception, 0, "%s: result of shape %s does not fit an output of shape %s", what, s1, s2);
        return FAILURE;
    }
    if (tsr_array_size(&dst->a) == 0) return SUCCESS;
    rc = tsr_copy(src->dtype, dst->a.dtype, dst->a.ndim, dst->a.shape, elem0(src), st, elem0(&dst->a), dst->a.strides);
    if (rc != 0) { tsr_throw_rc(rc, what); return FAILURE; }
    return SUCCESS;
}

int tsr_cast_into(const tsr_array *src, tsr_obj *dst, const char *what) { return cast_into(src, dst, what); }

int tsr_operand(zval *z, int scalar_dtype, zval *out) { return operand(z, scalar_dtype, out); }

int tsr_result_dtype(zval *left, zval *right)
{
    ZVAL_DEREF(left);
    if (right == NULL) return dtype_of_zv(left);
    ZVAL_DEREF(right);
    int dl = dtype_of_zv(left), dr = dtype_of_zv(right);
    if (dl < 0 || dr < 0) {
        /* PHP arrays: scan them the way operand() would */
        zval A, B;
        if (dl < 0) { if (operand(left, 0, &A) == FAILURE) return -1; dl = Z_TSR(A)->a.dtype; zval_ptr_dtor(&A); }
        if (dr < 0) { if (operand(right, 0, &B) == FAILURE) return -1; dr = Z_TSR(B)->a.dtype; zval_ptr_dtor(&B); }
        return promote(dl, dr);
    }
    if (is_scalar_zv(left) && !is_scalar_zv(right)) return with_scalar(dr, left);
    if (is_scalar_zv(right) && !is_scalar_zv(left)) return with_scalar(dl, right);
    return promote(dl, dr);
}

int tsr_binary_into(int op, zval *left, zval *right, tsr_obj *dst, zval *rv)
{
    if (dst && tsr_require_writable(dst, "out") == FAILURE) return FAILURE;
    return binary_core_into(op, left, right, dst, rv);
}

int tsr_unary_into(int op, zval *src, tsr_obj *dst, zval *rv)
{
    if (dst && tsr_require_writable(dst, "out") == FAILURE) return FAILURE;
    zval A;
    ZVAL_DEREF(src);
    int dt = dtype_of_zv(src);
    if (operand(src, dt >= 0 ? dt : D_F64, &A) == FAILURE) return FAILURE;
    int r = unary_dispatch(op, &A, dst, rv);
    zval_ptr_dtor(&A);
    return r;
}

/* ================================================================ registration */

ZEND_BEGIN_ARG_INFO_EX(ai_none, 0, 0, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_memmap, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, filename, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, mode, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 1)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, offset, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_load, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, mmapMode, IS_STRING, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_save, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_open_memmap, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, path, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, mode, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, shape, IS_ARRAY, 1)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, fortranOrder, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_flush, 0, 0, 0)
    ZEND_ARG_TYPE_INFO(0, sync, _IS_BOOL, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_one, 0, 0, 1)
    ZEND_ARG_INFO(0, value)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_array, 0, 0, 1)
    ZEND_ARG_INFO(0, data)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_shape_dtype, 0, 0, 1)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_full, 0, 0, 2)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, value)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_arange, 0, 0, 1)
    ZEND_ARG_INFO(0, start)
    ZEND_ARG_INFO(0, stop)
    ZEND_ARG_INFO(0, step)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_linspace, 0, 0, 2)
    ZEND_ARG_INFO(0, start)
    ZEND_ARG_INFO(0, stop)
    ZEND_ARG_INFO(0, num)
    ZEND_ARG_INFO(0, endpoint)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_eye, 0, 0, 1)
    ZEND_ARG_INFO(0, n)
    ZEND_ARG_INFO(0, m)
    ZEND_ARG_INFO(0, k)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_frombytes, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, bytes, IS_STRING, 0)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
    ZEND_ARG_INFO(0, shape)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_where, 0, 0, 3)
    ZEND_ARG_INFO(0, cond)
    ZEND_ARG_INFO(0, x)
    ZEND_ARG_INFO(0, y)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_complex, 0, 0, 1)
    ZEND_ARG_INFO(0, real)
    ZEND_ARG_INFO(0, imag)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_variadic, 0, 0, 0)
    ZEND_ARG_VARIADIC_INFO(0, args)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_axis, 0, 0, 0)
    ZEND_ARG_TYPE_INFO(0, axis, IS_LONG, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_axis_ddof, 0, 0, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_ARRAY | MAY_BE_NULL, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, ddof, IS_LONG, 0, "0")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, keepdims, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_axes_keep, 0, 0, 0)
    ZEND_ARG_TYPE_MASK(0, axis, MAY_BE_LONG | MAY_BE_ARRAY | MAY_BE_NULL, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, keepdims, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_axis_keep, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, axis, IS_LONG, 1, "null")
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, keepdims, _IS_BOOL, 0, "false")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_clip, 0, 0, 0)
    ZEND_ARG_INFO(0, min)
    ZEND_ARG_INFO(0, max)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_spec, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, spec, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_swapaxes, 0, 0, 2)
    ZEND_ARG_TYPE_INFO(0, axis1, IS_LONG, 0)
    ZEND_ARG_TYPE_INFO(0, axis2, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_round, 0, 0, 0)
    ZEND_ARG_TYPE_INFO_WITH_DEFAULT_VALUE(0, decimals, IS_LONG, 0, "0")
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_take, 0, 0, 1)
    ZEND_ARG_INFO(0, indices)
    ZEND_ARG_TYPE_INFO(0, axis, IS_LONG, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_put, 0, 0, 2)
    ZEND_ARG_INFO(0, indices)
    ZEND_ARG_INFO(0, values)
    ZEND_ARG_TYPE_INFO(0, axis, IS_LONG, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_dtype, 0, 0, 1)
    ZEND_ARG_TYPE_INFO(0, dtype, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_INFO_EX(ai_axes, 0, 0, 0)
    ZEND_ARG_TYPE_INFO(0, axes, IS_ARRAY, 1)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(ai_json, 0, 0, IS_MIXED, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_OBJ_INFO_EX(ai_getiterator, 0, 0, Iterator, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(ai_count, 0, 0, IS_LONG, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(ai_tostring, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(ai_serialize, 0, 0, IS_ARRAY, 0)
ZEND_END_ARG_INFO()
ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(ai_unserialize, 0, 1, IS_VOID, 0)
    ZEND_ARG_TYPE_INFO(0, data, IS_ARRAY, 0)
ZEND_END_ARG_INFO()

#if PHP_VERSION_ID >= 80400
#  define TSR_FE(name, handler, ai) ZEND_RAW_FENTRY(name, handler, ai, ZEND_ACC_PUBLIC, NULL, NULL)
#else
#  define TSR_FE(name, handler, ai) ZEND_RAW_FENTRY(name, handler, ai, ZEND_ACC_PUBLIC)
#endif
#define BIN_ME(name) TSR_FE(#name, zim_NDArray_binaryOp, ai_one)
#define REF_ME(name) TSR_FE(#name, zim_NDArray_reflectedOp, ai_one)
#define UN_ME(name) TSR_FE(#name, zim_NDArray_unaryOp, ai_none)
#define RED_ME(name) TSR_FE(#name, zim_NDArray_reduceOp, ai_axes_keep)
#define ARG_ME(name) TSR_FE(#name, zim_NDArray_reduceOp, ai_axis_keep)
#define STATIC_ME(name, ai) PHP_ME(NDArray, name, ai, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

static const zend_function_entry ndarray_methods[] = {
    PHP_ME(NDArray, __construct, ai_none, ZEND_ACC_PRIVATE)
    STATIC_ME(array, ai_array)
    STATIC_ME(zeros, ai_shape_dtype)
    STATIC_ME(ones, ai_shape_dtype)
    STATIC_ME(empty, ai_shape_dtype)
    STATIC_ME(full, ai_full)
    STATIC_ME(arange, ai_arange)
    STATIC_ME(linspace, ai_linspace)
    STATIC_ME(eye, ai_eye)
    STATIC_ME(fromBytes, ai_frombytes)
    STATIC_ME(where, ai_where)
    STATIC_ME(complex, ai_complex)
    PHP_ME(NDArray, shape, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, strides, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, ndim, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, size, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, dtype, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, isContiguous, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, isView, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, count, ai_count, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, getIterator, ai_getiterator, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, toArray, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, toList, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, toBytes, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, toJson, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, jsonSerialize, ai_json, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, __toString, ai_tostring, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, item, ai_variadic, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, copy, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, astype, ai_dtype, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, reshape, ai_variadic, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, ravel, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, transpose, ai_axes, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, t, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, flatten, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, swapAxes, ai_swapaxes, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, squeeze, ai_axis, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, nbytes, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, view, ai_dtype, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, round, ai_round, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, take, ai_take, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, put, ai_put, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, slice, ai_spec, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, assign, ai_one, ZEND_ACC_PUBLIC)
    BIN_ME(add) BIN_ME(sub) BIN_ME(mul) BIN_ME(div) BIN_ME(pow) BIN_ME(mod) BIN_ME(floorDiv)
    BIN_ME(maximum) BIN_ME(minimum) BIN_ME(atan2) BIN_ME(hypot)
    BIN_ME(eq) BIN_ME(ne) BIN_ME(lt) BIN_ME(le) BIN_ME(gt) BIN_ME(ge)
    BIN_ME(logicalAnd) BIN_ME(logicalOr) BIN_ME(logicalXor)
    REF_ME(rsub) REF_ME(rdiv) REF_ME(rpow)
    UN_ME(neg) UN_ME(abs) UN_ME(square) UN_ME(sign) UN_ME(sqrt) UN_ME(exp) UN_ME(log) UN_ME(log10) UN_ME(log2)
    UN_ME(sin) UN_ME(cos) UN_ME(tan) UN_ME(arcsin) UN_ME(arccos) UN_ME(arctan) UN_ME(sinh) UN_ME(cosh) UN_ME(tanh)
    UN_ME(floor) UN_ME(ceil) UN_ME(rint) UN_ME(expm1) UN_ME(log1p) UN_ME(reciprocal) UN_ME(isnan) UN_ME(isfinite)
    UN_ME(isinf) UN_ME(invert) UN_ME(real) UN_ME(imag) UN_ME(conj) UN_ME(angle)
    PHP_ME(NDArray, logicalNot, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, clip, ai_clip, ZEND_ACC_PUBLIC)
    RED_ME(sum) RED_ME(prod) RED_ME(min) RED_ME(max) ARG_ME(argmin) ARG_ME(argmax) RED_ME(any) RED_ME(all)
    PHP_ME(NDArray, mean, ai_axes_keep, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, var, ai_axis_ddof, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, std, ai_axis_ddof, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, cumsum, ai_axis, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, cumprod, ai_axis, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, sort, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, argsort, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, matmul, ai_one, ZEND_ACC_PUBLIC)
    TSR_FE("dot", zim_NDArray_matmul, ai_one)
    STATIC_ME(memmap, ai_memmap)
    STATIC_ME(load, ai_load)
    STATIC_ME(openMemmap, ai_open_memmap)
    PHP_ME(NDArray, save, ai_save, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, flush, ai_flush, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, isMemmap, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, isReadonly, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, filename, ai_none, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, __serialize, ai_serialize, ZEND_ACC_PUBLIC)
    PHP_ME(NDArray, __unserialize, ai_unserialize, ZEND_ACC_PUBLIC)
    PHP_FE_END
};

void tsr_register_ndarray(void)
{
    zend_class_entry ce;
    INIT_NS_CLASS_ENTRY(ce, "Tessero\\Ext", "NDArray", ndarray_methods);
    tsr_ce_ndarray = zend_register_internal_class(&ce);
    tsr_ce_ndarray->ce_flags |= ZEND_ACC_FINAL | ZEND_ACC_NO_DYNAMIC_PROPERTIES;
    tsr_ce_ndarray->create_object = tsr_create;
    tsr_ce_ndarray->get_iterator = tsr_get_iterator;
    zend_class_implements(tsr_ce_ndarray, 4, zend_ce_aggregate, zend_ce_countable, php_json_serializable_ce, zend_ce_stringable);

    memcpy(&tsr_handlers, zend_get_std_object_handlers(), sizeof(zend_object_handlers));
    tsr_handlers.offset = XtOffsetOf(tsr_obj, std);
    tsr_handlers.free_obj = tsr_free_obj;
    tsr_handlers.clone_obj = tsr_clone;
    tsr_handlers.read_dimension = tsr_read_dimension;
    tsr_handlers.write_dimension = tsr_write_dimension;
    tsr_handlers.has_dimension = tsr_has_dimension;
    tsr_handlers.unset_dimension = tsr_unset_dimension;
    tsr_handlers.count_elements = tsr_count_elements;
    tsr_handlers.compare = tsr_compare;
    tsr_handlers.do_operation = tsr_do_operation;
    tsr_handlers.get_gc = tsr_get_gc;
    tsr_handlers.get_debug_info = tsr_debug_info;
}
