/*
 * NumPy array construction and manipulation (Tessero\Np): the flip/roll/rot90/tile/repeat family, joining and
 * splitting, axis moves, diagonals and triangles, the *_like constructors, index conversion and gathering.
 *
 * Every function follows NumPy 2.4's definition (numpy/_core/shape_base.py, numeric.py, fromnumeric.py,
 * numpy/lib/_shape_base_impl.py, _function_base_impl.py, _twodim_base_impl.py, _index_tricks_impl.py): the
 * same result shapes, dtypes (NumPy 2 promotion, NEP 50 for Python scalars where NumPy keeps them weak), values,
 * and errors. Results are fresh arrays: where NumPy returns a view (flip, transpose-like moves, diagonal,
 * broadcast_to), Tessero returns a copy with the same values (ADR 0013).
 *
 * Registry kind: routine. Sequence arguments (numpy's `arrays`, `*xi`) arrive as kind 5; tuple and list
 * results leave as kind 6 sequences.
 */
#include "fn.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================ values and dtypes */

typedef struct {
    tsr_array a;
    unsigned char s[16];                         /* storage of a Python scalar argument (0-d) */
} harr;

static int is_intdt(int d) { return d == TSR_I64 || d == TSR_I32 || d == TSR_U8; }
static int is_intlike(int d) { return is_intdt(d) || d == TSR_BOOL; }
static int is_floatdt(int d) { return d == TSR_F64 || d == TSR_F32; }

/* numpy.result_type of two array dtypes */
static int promote2(int a, int b)
{
    if (a == b) return a;
    if (a == TSR_C128 || b == TSR_C128) return TSR_C128;
    if (a == TSR_BOOL) return b;
    if (b == TSR_BOOL) return a;
    if (a == TSR_F64 || b == TSR_F64) return TSR_F64;
    if (a == TSR_F32 || b == TSR_F32) {
        const int o = a == TSR_F32 ? b : a;
        return o == TSR_U8 ? TSR_F32 : TSR_F64;       /* int32 or int64 with float32 -> float64 */
    }
    if (a == TSR_I64 || b == TSR_I64) return TSR_I64;
    return TSR_I32;                                   /* uint8 with int32 */
}

/* a Python scalar kept weak (NEP 50): 'i' int, 'f' float, 'b' bool joining an array dtype */
static int promote_weak(int dt, char kind)
{
    if (kind == 'b') return dt;
    if (kind == 'i') return dt == TSR_BOOL ? TSR_I64 : dt;
    /* float */
    if (dt == TSR_BOOL || is_intdt(dt)) return TSR_F64;
    return dt;
}

/* numpy.can_cast(from, to, casting) for casting 0 no, 1 equiv, 2 safe, 3 same_kind, 4 unsafe */
static int can_cast(int from, int to, int casting)
{
    if (from == to || casting == 4) return 1;
    if (casting <= 1) return 0;
    static const char safe[7][8] = {
        /* F64 */ "1000001",   /* to: F64 F32 I64 I32 U8 BOOL C128 */
        /* F32 */ "1100001",
        /* I64 */ "1010001",
        /* I32 */ "1011001",
        /* U8  */ "1111101",
        /* BOOL*/ "1111111",
        /* C128*/ "0000001",
    };
    static const char same_kind[7][8] = {
        "1100001", "1100001", "1111001", "1111001", "1111101", "1111111", "0000001",
    };
    return (casting == 2 ? safe : same_kind)[from][to] == '1';
}

static const char *dtname(int d)
{
    static const char *const N[] = {"float64", "float32", "int64", "int32", "uint8", "bool", "complex128"};
    return d >= 0 && d < TSR_NDTYPES ? N[d] : "?";
}

/* a dtype argument: None -> -1 (keep), or a NumPy dtype name */
static int arg_dtype(const tsr_arg *a, int *dt)
{
    *dt = -1;
    if (!a || a->kind == 0) return TSR_OK;
    if (a->kind != 2 || !a->str) { fn_set_error("dtype must be a dtype name such as 'float64' or 'int64'"); return TSR_EARG; }
    static const struct { const char *n; int d; } T[] = {
        {"float64", TSR_F64}, {"float", TSR_F64}, {"f8", TSR_F64}, {"d", TSR_F64}, {"double", TSR_F64}, {"float_", TSR_F64},
        {"float32", TSR_F32}, {"f4", TSR_F32}, {"f", TSR_F32}, {"single", TSR_F32},
        {"int64", TSR_I64}, {"int", TSR_I64}, {"i8", TSR_I64}, {"intp", TSR_I64}, {"long", TSR_I64}, {"int_", TSR_I64},
        {"int32", TSR_I32}, {"i4", TSR_I32}, {"intc", TSR_I32},
        {"uint8", TSR_U8}, {"u1", TSR_U8}, {"ubyte", TSR_U8}, {"B", TSR_U8},
        {"bool", TSR_BOOL}, {"?", TSR_BOOL}, {"bool_", TSR_BOOL},
        {"complex128", TSR_C128}, {"complex", TSR_C128}, {"c16", TSR_C128}, {"D", TSR_C128}, {"cdouble", TSR_C128},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        if (strcmp(a->str, T[i].n) == 0) { *dt = T[i].d; return TSR_OK; }
    fn_set_error("data type '%s' not understood", a->str);
    return TSR_EARG;
}

static int arg_casting(const tsr_arg *a, int *casting)
{
    *casting = 3;
    if (!a || a->kind == 0) return TSR_OK;
    static const char *const C[] = {"no", "equiv", "safe", "same_kind", "unsafe"};
    if (a->kind == 2 && a->str)
        for (int i = 0; i < 5; i++) if (strcmp(a->str, C[i]) == 0) { *casting = i; return TSR_OK; }
    fn_set_error("casting must be one of 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'");
    return TSR_EARG;
}

static const tsr_arg *argk(const tsr_arg *args, int nargs, int k)
{
    static const tsr_arg none = {0};
    return k < nargs ? &args[k] : &none;
}

static int is_none(const tsr_arg *a) { return a->kind == 0; }

static void c_strides(int32_t nd, const int64_t *sh, int64_t isz, int64_t *st)
{
    int64_t s = isz;
    for (int32_t d = nd - 1; d >= 0; d--) { st[d] = s; s *= sh[d] > 0 ? sh[d] : 1; }
}

/* an argument as an array: arrays as given, Python scalars as 0-d int64 / float64 / bool (numpy.asarray) */
static int get_arr(const tsr_arg *x, harr *h, const char *what)
{
    memset(h, 0, sizeof *h);
    if (x->kind == 3) { h->a = x->arr; return TSR_OK; }
    int dt;
    if (x->kind == 1 && (x->flags & 1)) { memcpy(h->s, &x->ival, 8); dt = TSR_I64; }
    else if (x->kind == 1) { memcpy(h->s, &x->num, 8); dt = TSR_F64; }
    else if (x->kind == 4) { h->s[0] = x->num != 0; dt = TSR_BOOL; }
    else { fn_set_error("%s must be an array or a number", what); return TSR_EARG; }
    h->a.data = h->s;
    h->a.offset = 0;
    h->a.ndim = 0;
    h->a.dtype = dt;
    return TSR_OK;
}

/* Python scalar kind of an argument: 'i', 'f', 'b', or 0 for arrays (strong) */
static char weak_kind(const tsr_arg *x)
{
    if (x->kind == 1) return (x->flags & 1) ? 'i' : 'f';
    if (x->kind == 4) return 'b';
    return 0;
}

static inline char *elem0(const tsr_array *a) { return (char *)a->data + a->offset; }

static int64_t rd_i64(const char *p, int dt)
{
    switch (dt) {
    case TSR_I64: { int64_t v; memcpy(&v, p, 8); return v; }
    case TSR_I32: { int32_t v; memcpy(&v, p, 4); return v; }
    case TSR_U8: case TSR_BOOL: return *(const uint8_t *)p;
    case TSR_F64: { double v; memcpy(&v, p, 8); return (v > -9223372036854775808.0 && v < 9223372036854775808.0) ? (int64_t)v : INT64_MIN; }
    case TSR_F32: { float v; memcpy(&v, p, 4); return (v > -9223372036854775808.0f && v < 9223372036854775808.0f) ? (int64_t)v : INT64_MIN; }
    default: return 0;
    }
}

static double rd_f64(const char *p, int dt)
{
    switch (dt) {
    case TSR_F64: { double v; memcpy(&v, p, 8); return v; }
    case TSR_F32: { float v; memcpy(&v, p, 4); return v; }
    case TSR_I64: { int64_t v; memcpy(&v, p, 8); return (double)v; }
    case TSR_I32: { int32_t v; memcpy(&v, p, 4); return v; }
    case TSR_U8: case TSR_BOOL: return *(const uint8_t *)p;
    case TSR_C128: { double v; memcpy(&v, p, 8); return v; }
    default: return 0;
    }
}

static int truthy(const char *p, int dt)
{
    switch (dt) {
    case TSR_F64: { double v; memcpy(&v, p, 8); return v != 0; }
    case TSR_F32: { float v; memcpy(&v, p, 4); return v != 0; }
    case TSR_C128: { double v[2]; memcpy(v, p, 16); return v[0] != 0 || v[1] != 0; }
    default: return rd_i64(p, dt) != 0;
    }
}

/* copy (and cast) a strided view into a strided destination; 0-d views too */
static int copy_view(const tsr_array *v, int dto, void *dst, const int64_t *dst_st)
{
    if (v->ndim == 0) {
        const int64_t one = 1, z = 0;
        return tsr_copy(v->dtype, dto, 1, &one, elem0(v), &z, dst, &z);
    }
    return tsr_copy(v->dtype, dto, v->ndim, v->shape, elem0(v), v->strides, dst, dst_st);
}

/* a 0-d result as NumPy's scalar (the bindings return PHP numbers) */
static int result_scalar(tsr_result *r, const char *p, int dt)
{
    switch (dt) {
    case TSR_F64: case TSR_F32: fn_result_num(r, rd_f64(p, dt)); return TSR_OK;
    case TSR_BOOL: memset(r, 0, sizeof *r); r->kind = 4; r->num = *(const uint8_t *)p != 0; return TSR_OK;
    case TSR_I64: case TSR_I32: case TSR_U8: fn_result_int(r, rd_i64(p, dt)); return TSR_OK;
    default: return TSR_EARG;
    }
}

/* materialise a view as a fresh C-contiguous array of dtype dto (0-d -> a scalar) */
static int emit(tsr_result *r, const tsr_array *v, int dto)
{
    if (dto < 0) dto = v->dtype;
    if (v->ndim == 0 && dto != TSR_C128) {
        unsigned char tmp[16];
        const int64_t one = 1, z = 0;
        int rc = tsr_copy(v->dtype, dto, 1, &one, elem0(v), &z, tmp, &z);
        if (rc < 0) return rc;
        return result_scalar(r, (const char *)tmp, dto);
    }
    void *p = fn_result_array(r, dto, v->ndim, v->shape);
    if (!p) return TSR_ENOMEM;
    if (tsr_shape_size(v->ndim, v->shape) == 0) return TSR_OK;
    int64_t st[TSR_MAXDIM];
    c_strides(v->ndim, v->shape, tsr_itemsize(dto), st);
    return copy_view(v, dto, p, st);
}

/* a fresh zero-filled result of the given shape and dtype */
static void *new_result(tsr_result *r, int dt, int32_t nd, const int64_t *sh)
{
    void *p = fn_result_array(r, dt, nd, sh);
    if (p) {
        const int64_t n = tsr_shape_size(nd, sh);
        if (n > 0) memset(p, 0, (size_t)(n * tsr_itemsize(dt)));
    }
    return p;
}

/* ================================================================ integer arguments */

/* operator.index(x): ints, bools, 0-d integer arrays */
static int int_param(const tsr_arg *a, const char *name, int64_t *out)
{
    if (a->kind == 1 && (a->flags & 1)) { *out = a->ival; return TSR_OK; }
    if (a->kind == 4) { *out = a->num != 0; return TSR_OK; }
    if (a->kind == 3 && a->arr.ndim == 0 && is_intlike(a->arr.dtype)) { *out = rd_i64(elem0(&a->arr), a->arr.dtype); return TSR_OK; }
    if (a->kind == 1 || (a->kind == 3 && a->arr.ndim == 0)) {
        fn_set_error("'float' object cannot be interpreted as an integer (%s)", name);
        return TSR_ETYPE;                                 /* numpy: TypeError */
    }
    fn_set_error("%s must be an integer", name);
    return TSR_EARG;
}

/* an int or a sequence of ints (an axis tuple, reps, a shape); *scalar says which */
static int int_list(const tsr_arg *a, const char *name, int64_t *out, int max, int *n, int *scalar)
{
    *n = 0;
    if (scalar) *scalar = 0;
    if (a->kind == 5) {
        if (a->count > max) { fn_set_error("%s: too many entries", name); return TSR_EARG; }
        for (int64_t i = 0; i < a->count; i++) {
            int rc = int_param(&a->items[i], name, &out[i]);
            if (rc < 0) return rc;
        }
        *n = (int)a->count;
        return TSR_OK;
    }
    if (a->kind == 3 && a->arr.ndim == 1) {
        const int64_t len = a->arr.shape[0];
        if (len == 0) return TSR_OK;                        /* [] (float64 when built from an empty PHP list) */
        if (!is_intlike(a->arr.dtype)) { fn_set_error("'float' object cannot be interpreted as an integer (%s)", name); return TSR_EARG; }
        if (len > max) { fn_set_error("%s: too many entries", name); return TSR_EARG; }
        for (int64_t i = 0; i < len; i++) out[i] = rd_i64(elem0(&a->arr) + i * a->arr.strides[0], a->arr.dtype);
        *n = (int)len;
        return TSR_OK;
    }
    if (a->kind == 3 && a->arr.ndim > 1) { fn_set_error("%s must be an integer or a sequence of integers", name); return TSR_EARG; }
    int rc = int_param(a, name, &out[0]);
    if (rc < 0) return rc;
    *n = 1;
    if (scalar) *scalar = 1;
    return TSR_OK;
}

/* integer index data (numpy's conversion to intp): ints and bools as they are; a Python float or a PHP list of
   floats by int() (NaN raises); a float ndarray is a TypeError ('safe' casting). *v gets a 0-d or strided int
   view is not needed: the values come back flat in C order in *out (tsr_alloc'ed, *bytes). */
static int ints_of(const tsr_arg *a, const char *what, int64_t **out, int64_t *bytes, int32_t *nd, int64_t *sh)
{
    harr h;
    int rc = get_arr(a, &h, what);
    if (rc < 0) return rc;
    const int64_t n = tsr_shape_size(h.a.ndim, h.a.shape);
    if (n < 0) return TSR_ENOMEM;
    *nd = h.a.ndim;
    memcpy(sh, h.a.shape, sizeof(int64_t) * (size_t)h.a.ndim);
    const int fl = is_floatdt(h.a.dtype);
    if (fl && a->kind == 3 && !(a->flags & 2) && n > 0) {
        fn_set_error("Cannot cast array data from dtype('%s') to dtype('int64') according to the rule 'safe'", dtname(h.a.dtype));
        return TSR_EARG;
    }
    if (h.a.dtype == TSR_C128 && n > 0) { fn_set_error("Cannot cast array data from dtype('complex128') to dtype('int64')"); return TSR_EARG; }
    *bytes = (n + 1) * 8 + 64;
    *out = (int64_t *)tsr_alloc(*bytes);
    if (!*out) return TSR_ENOMEM;
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t off = 0;
        for (int d = 0; d < h.a.ndim; d++) off += ix[d] * h.a.strides[d];
        const char *p = elem0(&h.a) + off;
        if (fl) {
            const double x = rd_f64(p, h.a.dtype);
            if (!(x > -9.2e18 && x < 9.2e18)) {
                tsr_free(*out, *bytes);
                *out = NULL;
                fn_set_error(x != x ? "cannot convert float NaN to integer" : "cannot convert float infinity to integer");
                return TSR_EARG;
            }
            (*out)[q] = (int64_t)x;
        } else (*out)[q] = rd_i64(p, h.a.dtype);
        for (int d = h.a.ndim - 1; d >= 0; d--) { if (++ix[d] < h.a.shape[d]) break; ix[d] = 0; }
    }
    if (a->kind == 1 && (a->flags & 1)) (*out)[0] = a->ival;
    return TSR_OK;
}

static int norm_axis(int64_t ax, int nd, int *out)
{
    if (ax < -(int64_t)nd || ax >= nd) {
        fn_set_error("axis %lld is out of bounds for array of dimension %d", (long long)ax, nd);
        return TSR_EARG;
    }
    *out = (int)(ax < 0 ? ax + nd : ax);
    return TSR_OK;
}

/* numpy.lib.array_utils.normalize_axis_tuple */
static int axis_tuple(const tsr_arg *a, int nd, int *axes, int *n, int allow_dup, const char *argname)
{
    int64_t v[TSR_MAXDIM + 1];
    int rc = int_list(a, argname ? argname : "axis", v, TSR_MAXDIM + 1, n, NULL);
    if (rc < 0) return rc;
    for (int i = 0; i < *n; i++) {
        if ((rc = norm_axis(v[i], nd, &axes[i])) < 0) return rc;
        for (int j = 0; j < i && !allow_dup; j++)
            if (axes[j] == axes[i]) {
                if (argname) fn_set_error("repeated axis in `%s` argument", argname);
                else fn_set_error("repeated axis");
                return TSR_EARG;
            }
    }
    return TSR_OK;
}

/* ================================================================ views */

static void v_flip(tsr_array *a, int ax)
{
    if (a->shape[ax] > 0) a->offset += (a->shape[ax] - 1) * a->strides[ax];
    a->strides[ax] = -a->strides[ax];
}

static void v_transpose(tsr_array *a, const int *perm)
{
    int64_t sh[TSR_MAXDIM], st[TSR_MAXDIM];
    for (int d = 0; d < a->ndim; d++) { sh[d] = a->shape[perm[d]]; st[d] = a->strides[perm[d]]; }
    memcpy(a->shape, sh, sizeof(int64_t) * (size_t)a->ndim);
    memcpy(a->strides, st, sizeof(int64_t) * (size_t)a->ndim);
}

static void v_swap(tsr_array *a, int i, int j)
{
    int64_t t = a->shape[i]; a->shape[i] = a->shape[j]; a->shape[j] = t;
    t = a->strides[i]; a->strides[i] = a->strides[j]; a->strides[j] = t;
}

static int v_insert_axis(tsr_array *a, int pos)
{
    if (a->ndim >= TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    for (int d = a->ndim; d > pos; d--) { a->shape[d] = a->shape[d - 1]; a->strides[d] = a->strides[d - 1]; }
    a->shape[pos] = 1;
    a->strides[pos] = 0;
    a->ndim++;
    return TSR_OK;
}

static void v_slice(tsr_array *a, int ax, int64_t start, int64_t stop)
{
    a->offset += start * a->strides[ax];
    a->shape[ax] = stop > start ? stop - start : 0;
}

/* ravel (C order) into a fresh buffer when the view is not C-contiguous: *owned gets the block to free */
typedef struct { void *p; int64_t bytes; } owned;

static void own_free(owned *o) { if (o->p) tsr_free(o->p, o->bytes); o->p = NULL; }

static int v_ravel(const tsr_array *a, tsr_array *out, owned *o)
{
    o->p = NULL;
    const int64_t n = tsr_shape_size(a->ndim, a->shape);
    if (n < 0) return TSR_ENOMEM;
    const int64_t isz = tsr_itemsize(a->dtype);
    *out = *a;
    out->ndim = 1;
    out->shape[0] = n;
    out->strides[0] = isz;
    if (a->ndim == 1) { out->strides[0] = a->strides[0]; return TSR_OK; }
    if (a->ndim == 0) { out->strides[0] = 0; return TSR_OK; }
    int contig = 1;
    int64_t s = isz;
    for (int d = a->ndim - 1; d >= 0; d--) {
        if (a->shape[d] != 1 && a->strides[d] != s) { contig = 0; break; }
        s *= a->shape[d];
    }
    if (contig || n == 0) return TSR_OK;
    o->bytes = n * isz + 64;
    o->p = tsr_alloc(o->bytes);
    if (!o->p) return TSR_ENOMEM;
    int64_t st[TSR_MAXDIM];
    c_strides(a->ndim, a->shape, isz, st);
    int rc = copy_view(a, a->dtype, o->p, st);
    out->data = o->p;
    out->offset = 0;
    return rc;
}

/* broadcast shapes (numpy.broadcast_shapes) */
static int bshape(int32_t na, const int64_t *sa, int32_t nb, const int64_t *sb, int32_t *nd, int64_t *out)
{
    int64_t a[TSR_MAXDIM], b[TSR_MAXDIM];                /* out may alias sa or sb */
    memcpy(a, sa, sizeof(int64_t) * (size_t)na);
    memcpy(b, sb, sizeof(int64_t) * (size_t)nb);
    if (tsr_broadcast_shape(na, a, nb, b, nd, out) != 0) {
        fn_set_error("shape mismatch: objects cannot be broadcast to a single shape");
        return TSR_ESHAPE;
    }
    return TSR_OK;
}

static int v_broadcast(tsr_array *a, int32_t nd, const int64_t *sh)
{
    int64_t st[TSR_MAXDIM];
    if (tsr_broadcast_strides(a, nd, sh, st) != 0) {
        fn_set_error("operands could not be broadcast together with remapped shapes");
        return TSR_ESHAPE;
    }
    a->ndim = nd;
    memcpy(a->shape, sh, sizeof(int64_t) * (size_t)nd);
    memcpy(a->strides, st, sizeof(int64_t) * (size_t)nd);
    return TSR_OK;
}

/* ================================================================ sequences */

/* the items of a sequence argument as arrays: a list (kind 5), or an array iterated along its first axis */
typedef struct { int n; harr *h; } arrlist;

static void al_free(arrlist *l) { if (l->h) tsr_free(l->h, (int64_t)l->n * (int64_t)sizeof(harr) + 64); l->h = NULL; l->n = 0; }

static int al_get(const tsr_arg *a, arrlist *l, const char *what)
{
    l->n = 0;
    l->h = NULL;
    if (a->kind == 5) {
        if (a->count > (1 << 24)) { fn_set_error("%s: too many arrays", what); return TSR_EARG; }
        l->n = (int)a->count;
        l->h = (harr *)tsr_alloc((int64_t)l->n * (int64_t)sizeof(harr) + 64);
        if (!l->h) return TSR_ENOMEM;
        for (int i = 0; i < l->n; i++) {
            int rc = get_arr(&a->items[i], &l->h[i], what);
            if (rc < 0) { al_free(l); return rc; }
        }
        return TSR_OK;
    }
    if (a->kind == 3) {
        if (a->arr.ndim == 0) { fn_set_error("%s must be a sequence of arrays", what); return TSR_EARG; }
        if (a->arr.shape[0] > (1 << 24)) { fn_set_error("%s: too many arrays", what); return TSR_EARG; }
        l->n = (int)a->arr.shape[0];
        l->h = (harr *)tsr_alloc((int64_t)l->n * (int64_t)sizeof(harr) + 64);
        if (!l->h) return TSR_ENOMEM;
        for (int i = 0; i < l->n; i++) {
            memset(&l->h[i], 0, sizeof l->h[i]);
            tsr_array *v = &l->h[i].a;
            *v = a->arr;
            v->offset += i * a->arr.strides[0];
            v->ndim--;
            memmove(v->shape, v->shape + 1, sizeof(int64_t) * (size_t)v->ndim);
            memmove(v->strides, v->strides + 1, sizeof(int64_t) * (size_t)v->ndim);
        }
        return TSR_OK;
    }
    fn_set_error("%s must be a sequence of arrays", what);
    return TSR_EARG;
}

/* ================================================================ flip, rot90, roll */

static int r_flip(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int mode = ctx ? *(const int *)ctx : 0;       /* 0 flip, 1 fliplr, 2 flipud */
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "m");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    if (mode == 1) {
        if (v.ndim < 2) { fn_set_error("Input must be >= 2-d."); return TSR_EARG; }
        v_flip(&v, 1);
    } else if (mode == 2) {
        if (v.ndim < 1) { fn_set_error("Input must be >= 1-d."); return TSR_EARG; }
        v_flip(&v, 0);
    } else {
        const tsr_arg *ax = argk(args, nargs, 1);
        if (is_none(ax)) {
            for (int d = 0; d < v.ndim; d++) v_flip(&v, d);
        } else {
            int axes[TSR_MAXDIM + 1], n;
            if ((rc = axis_tuple(ax, v.ndim, axes, &n, 0, NULL)) < 0) return rc;
            for (int i = 0; i < n; i++) v_flip(&v, axes[i]);
        }
    }
    return emit(&res[0], &v, -1);
}
static const int FLIP = 0, FLIPLR = 1, FLIPUD = 2;

static int r_rot90(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "m");
    if (rc < 0) return rc;
    int64_t k = 1;
    const tsr_arg *ka = argk(args, nargs, 1);
    if (!is_none(ka) && ka->kind == 1 && !(ka->flags & 1)) {
        /* a float k: numpy's k %= 4, then the branches k == 0, 2, 1 and otherwise the k == 3 one (NaN too) */
        const double r = ka->num - 4.0 * floor(ka->num / 4.0);
        k = r == 0 ? 0 : r == 1 ? 1 : r == 2 ? 2 : 3;
    } else if (!is_none(ka) && (rc = int_param(ka, "k", &k)) < 0) return rc;
    int64_t ax[3] = {0, 1, 0};
    int n = 2;
    if (!is_none(argk(args, nargs, 2)) && (rc = int_list(argk(args, nargs, 2), "axes", ax, 3, &n, NULL)) < 0) return rc;
    if (n != 2) { fn_set_error("len(axes) must be 2."); return TSR_EARG; }
    tsr_array v = h.a;
    const int nd = v.ndim;
    if (ax[0] == ax[1] || llabs(ax[0] - ax[1]) == nd) { fn_set_error("Axes must be different."); return TSR_EARG; }
    if (ax[0] >= nd || ax[0] < -nd || ax[1] >= nd || ax[1] < -nd) {
        fn_set_error("Axes=(%lld, %lld) out of range for array of ndim=%d.", (long long)ax[0], (long long)ax[1], nd);
        return TSR_EARG;
    }
    const int a0 = (int)(ax[0] < 0 ? ax[0] + nd : ax[0]), a1 = (int)(ax[1] < 0 ? ax[1] + nd : ax[1]);
    k = ((k % 4) + 4) % 4;
    if (k == 2) { v_flip(&v, a0); v_flip(&v, a1); }
    else if (k == 1) { v_flip(&v, a1); v_swap(&v, a0, a1); }
    else if (k == 3) { v_swap(&v, a0, a1); v_flip(&v, a1); }
    return emit(&res[0], &v, -1);
}

/* python's % for a positive modulus */
static int64_t pymod(int64_t a, int64_t m) { int64_t r = a % m; return r < 0 ? r + m : r; }

/* per-axis gather: out[i0, i1, ...] = src at sum_d tab[d][i_d] bytes (itemsize copies, no casting) */
static int gather_tables(const tsr_array *src, int32_t nd, const int64_t *osh, int64_t *const *tab, void *dst)
{
    const int64_t n = tsr_shape_size(nd, osh);
    if (n <= 0) return n < 0 ? TSR_ENOMEM : TSR_OK;
    const int64_t isz = tsr_itemsize(src->dtype);
    const char *base = elem0(src);
    char *o = (char *)dst;
    if (nd == 0) { memcpy(o, base, (size_t)isz); return TSR_OK; }
    int64_t idx[TSR_MAXDIM] = {0};
    int64_t off = 0;
    for (int d = 0; d < nd; d++) off += tab[d][0];
    const int last = nd - 1;
    const int64_t *tl = tab[last];
    for (;;) {
        const int64_t outer = off - tl[0];
        for (int64_t i = 0; i < osh[last]; i++, o += isz) memcpy(o, base + outer + tl[i], (size_t)isz);
        int d = last - 1;
        for (; d >= 0; d--) {
            off -= tab[d][idx[d]];
            if (++idx[d] < osh[d]) { off += tab[d][idx[d]]; break; }
            idx[d] = 0;
            off += tab[d][0];
        }
        if (d < 0) break;
    }
    return TSR_OK;
}

/* allocate the per-axis tables for an output shape */
typedef struct { int64_t *t[TSR_MAXDIM]; int64_t bytes; void *block; } tables;

static int tables_new(tables *T, int32_t nd, const int64_t *osh)
{
    int64_t total = 0;
    for (int d = 0; d < nd; d++) total += osh[d] > 0 ? osh[d] : 1;
    T->bytes = total * 8 + 64;
    T->block = tsr_alloc(T->bytes);
    if (!T->block) return TSR_ENOMEM;
    int64_t *p = (int64_t *)T->block;
    for (int d = 0; d < nd; d++) { T->t[d] = p; p[0] = 0; p += osh[d] > 0 ? osh[d] : 1; }
    return TSR_OK;
}

static void tables_free(tables *T) { if (T->block) tsr_free(T->block, T->bytes); T->block = NULL; }

/* gather into a new result of shape osh (dtype of src) from per-axis tables */
static int emit_tables(tsr_result *r, const tsr_array *src, int32_t nd, const int64_t *osh, tables *T)
{
    void *p = fn_result_array(r, src->dtype, nd, osh);
    if (!p) return TSR_ENOMEM;
    int rc = gather_tables(src, nd, osh, T->t, p);
    if (rc == TSR_OK && nd == 0) {                       /* a scalar */
        tsr_result s;
        rc = result_scalar(&s, (const char *)p, src->dtype);
        if (rc == TSR_OK) { fn_results_free(r, 1); *r = s; }
        else rc = TSR_OK;                                 /* complex: keep the 0-d array */
    }
    return rc;
}

static int r_roll(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    const tsr_arg *sa = argk(args, nargs, 1), *aa = argk(args, nargs, 2);
    int32_t ond = v.ndim;
    int64_t osh[TSR_MAXDIM];
    memcpy(osh, v.shape, sizeof(int64_t) * (size_t)v.ndim);
    if (is_none(aa)) {                                    /* roll(a.ravel(), shift, 0).reshape(a.shape) */
        if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc;
    }
    /* shifts: ints, or floats truncated by int() (a sequence broadcast against the axes) */
    double sh[TSR_MAXDIM + 64];
    int ns = 0;
    if (sa->kind == 5 || (sa->kind == 3 && sa->arr.ndim >= 1)) {
        harr hs;
        const int64_t cnt = sa->kind == 5 ? sa->count : sa->arr.shape[0];
        if (sa->kind == 3 && sa->arr.ndim > 1) { own_free(&o); fn_set_error("'shift' and 'axis' should be scalars or 1D sequences"); return TSR_EARG; }
        if (cnt > TSR_MAXDIM + 64) { own_free(&o); fn_set_error("shift: too many entries"); return TSR_EARG; }
        for (int64_t i = 0; i < cnt; i++) {
            if (sa->kind == 5) {
                if ((rc = get_arr(&sa->items[i], &hs, "shift")) < 0 || hs.a.ndim != 0) {
                    own_free(&o);
                    if (rc == TSR_OK) fn_set_error("'shift' and 'axis' should be scalars or 1D sequences");
                    return TSR_EARG;
                }
                sh[ns++] = rd_f64(elem0(&hs.a), hs.a.dtype);
                if (is_intdt(hs.a.dtype)) sh[ns - 1] = (double)rd_i64(elem0(&hs.a), hs.a.dtype);
            } else {
                sh[ns++] = rd_f64(elem0(&sa->arr) + i * sa->arr.strides[0], sa->arr.dtype);
            }
        }
    } else {
        harr hs;
        if ((rc = get_arr(sa, &hs, "shift")) < 0) { own_free(&o); return rc; }
        sh[ns++] = rd_f64(elem0(&hs.a), hs.a.dtype);
    }
    int64_t shi[TSR_MAXDIM + 64];
    char exact[TSR_MAXDIM + 64] = {0};
    /* exact int64 shifts when given as integers; floats go through int() */
    if (sa->kind == 1 && (sa->flags & 1)) { shi[0] = sa->ival; exact[0] = 1; }
    else if (sa->kind == 3 && sa->arr.ndim <= 1 && is_intlike(sa->arr.dtype))
        for (int i = 0; i < ns; i++) { shi[i] = rd_i64(elem0(&sa->arr) + (sa->arr.ndim ? i * sa->arr.strides[0] : 0), sa->arr.dtype); exact[i] = 1; }
    else if (sa->kind == 5)
        for (int i = 0; i < ns; i++) if (sa->items[i].kind == 1 && (sa->items[i].flags & 1)) { shi[i] = sa->items[i].ival; exact[i] = 1; }
    for (int i = 0; i < ns; i++) {
        if (exact[i]) continue;
        if (!(sh[i] > -9.2e18 && sh[i] < 9.2e18)) { own_free(&o); fn_set_error("cannot convert float NaN or infinity to integer"); return TSR_EARG; }
        shi[i] = (int64_t)sh[i];
    }
    int axes[TSR_MAXDIM + 64];
    int na = 1;
    if (is_none(aa)) axes[0] = 0;
    else if ((rc = axis_tuple(aa, v.ndim, axes, &na, 1, NULL)) < 0) { own_free(&o); return rc; }
    if (ns != na && ns != 1 && na != 1) { own_free(&o); fn_set_error("shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between arg 0 with shape (%d,) and arg 1 with shape (%d,).", ns, na); return TSR_EARG; }
    int64_t tot[TSR_MAXDIM] = {0};
    const int nb = ns > na ? ns : na;
    for (int i = 0; i < nb; i++) {
        const int ax = axes[na == 1 ? 0 : i];
        tot[ax] = (int64_t)((uint64_t)tot[ax] + (uint64_t)shi[ns == 1 ? 0 : i]);
    }
    tables T;
    if ((rc = tables_new(&T, v.ndim, v.shape)) < 0) { own_free(&o); return rc; }
    for (int d = 0; d < v.ndim; d++) {
        const int64_t len = v.shape[d];
        const int64_t off = pymod(tot[d], len ? len : 1);
        for (int64_t i = 0; i < len; i++) T.t[d][i] = pymod(i - off, len) * v.strides[d];
    }
    tsr_result tmp;
    rc = emit_tables(&tmp, &v, v.ndim, v.shape, &T);
    tables_free(&T);
    own_free(&o);
    if (rc < 0) return rc;
    if (is_none(aa) && tmp.kind == 3) {                  /* reshape back */
        tmp.arr.ndim = ond;
        memcpy(tmp.arr.shape, osh, sizeof(int64_t) * (size_t)ond);
        c_strides(ond, osh, tsr_itemsize(tmp.arr.dtype), tmp.arr.strides);
        if (ond == 0) {
            tsr_result s;
            if (result_scalar(&s, (const char *)tmp.arr.data, tmp.arr.dtype) == TSR_OK) { fn_results_free(&tmp, 1); tmp = s; }
        }
    }
    res[0] = tmp;
    return TSR_OK;
}

/* ================================================================ tile, repeat, resize */

static int r_tile(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "A");
    if (rc < 0) return rc;
    int64_t reps[TSR_MAXDIM + 1];
    int d;
    if ((rc = int_list(argk(args, nargs, 1), "reps", reps, TSR_MAXDIM + 1, &d, NULL)) < 0) return rc;
    if (d > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    tsr_array c = h.a;
    while (c.ndim < d) if ((rc = v_insert_axis(&c, 0)) < 0) return rc;   /* ndmin=d */
    int64_t tup[TSR_MAXDIM];
    for (int i = 0; i < c.ndim; i++) tup[i] = 1;
    for (int i = 0; i < d; i++) tup[c.ndim - d + i] = reps[i];
    int64_t osh[TSR_MAXDIM];
    for (int i = 0; i < c.ndim; i++) {
        if (tup[i] < 0) { fn_set_error("negative dimensions are not allowed"); return TSR_EARG; }
        if (c.shape[i] != 0 && tup[i] > INT64_MAX / c.shape[i]) { fn_set_error("array is too big"); return TSR_EARG; }
        osh[i] = c.shape[i] * tup[i];
    }
    if (tsr_shape_size(c.ndim, osh) < 0) { fn_set_error("array is too big; `arr.size * arr.dtype.itemsize` is larger than the maximum possible size."); return TSR_EARG; }
    tables T;
    if ((rc = tables_new(&T, c.ndim, osh)) < 0) return rc;
    for (int i = 0; i < c.ndim; i++)
        for (int64_t j = 0; j < osh[i]; j++) T.t[i][j] = (j % c.shape[i]) * c.strides[i];
    rc = emit_tables(&res[0], &c, c.ndim, osh, &T);
    tables_free(&T);
    return rc;
}

static int r_repeat(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    const tsr_arg *aa = argk(args, nargs, 2);
    int ax = 0;
    if (is_none(aa)) { if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc; }
    else {
        int64_t a64;
        if ((rc = int_param(aa, "axis", &a64)) < 0 || (rc = norm_axis(a64, v.ndim, &ax)) < 0) return rc;
    }
    int64_t *rep, rb;
    int32_t rnd;
    int64_t rsh[TSR_MAXDIM];
    if ((rc = ints_of(argk(args, nargs, 1), "repeats", &rep, &rb, &rnd, rsh)) < 0) { own_free(&o); return rc; }
    if (rnd > 1) { tsr_free(rep, rb); own_free(&o); fn_set_error("object too deep for desired array"); return TSR_EARG; }
    const int64_t len = v.ndim ? v.shape[ax] : 1;
    const int64_t nr = rnd ? rsh[0] : 1;
    if (nr != 1 && nr != len) {
        tsr_free(rep, rb);
        own_free(&o);
        fn_set_error("operands could not be broadcast together with shape (%lld,) (%lld,)", (long long)len, (long long)nr);
        return TSR_EARG;
    }
    int64_t total = 0;
    for (int64_t i = 0; i < len; i++) {
        const int64_t r = rep[nr > 1 ? i : 0];
        if (r < 0) { tsr_free(rep, rb); own_free(&o); fn_set_error("repeats may not contain negative values."); return TSR_EARG; }
        if (total > INT64_MAX / 16 - r) { tsr_free(rep, rb); own_free(&o); fn_set_error("array is too big"); return TSR_EARG; }
        total += r;
    }
    int64_t osh[TSR_MAXDIM];
    memcpy(osh, v.shape, sizeof(int64_t) * (size_t)v.ndim);
    osh[ax] = total;
    if (tsr_shape_size(v.ndim, osh) < 0) { tsr_free(rep, rb); own_free(&o); fn_set_error("array is too big"); return TSR_EARG; }
    tables T;
    if ((rc = tables_new(&T, v.ndim, osh)) < 0) { tsr_free(rep, rb); own_free(&o); return rc; }
    for (int d = 0; d < v.ndim; d++) {
        if (d != ax) { for (int64_t j = 0; j < osh[d]; j++) T.t[d][j] = j * v.strides[d]; continue; }
        int64_t k = 0;
        for (int64_t i = 0; i < len; i++) {
            const int64_t r = rep[nr > 1 ? i : 0];
            for (int64_t j = 0; j < r; j++) T.t[d][k++] = i * v.strides[d];
        }
    }
    tsr_free(rep, rb);
    rc = emit_tables(&res[0], &v, v.ndim, osh, &T);
    tables_free(&T);
    own_free(&o);
    return rc;
}

static int r_resize(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int64_t sh[TSR_MAXDIM + 1];
    int nd;
    if ((rc = int_list(argk(args, nargs, 1), "new_shape", sh, TSR_MAXDIM + 1, &nd, NULL)) < 0) return rc;
    if (nd > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    for (int d = 0; d < nd; d++) if (sh[d] < 0) { fn_set_error("all elements of `new_shape` must be non-negative"); return TSR_EARG; }
    const int64_t nn = tsr_shape_size(nd, sh);
    if (nn < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    tsr_array v;
    owned o = {0};
    if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc;
    const int64_t n = v.shape[0];
    if (n == 0 || nn == 0) {                              /* zeros_like(a, shape=new_shape) */
        own_free(&o);
        if (nd == 0) { unsigned char z[16] = {0}; return result_scalar(&res[0], (const char *)z, h.a.dtype); }
        return new_result(&res[0], h.a.dtype, nd, sh) ? TSR_OK : TSR_ENOMEM;
    }
    tables T;
    const int64_t one = nn;
    if ((rc = tables_new(&T, 1, &one)) < 0) { own_free(&o); return rc; }
    for (int64_t i = 0; i < nn; i++) T.t[0][i] = (i % n) * v.strides[0];
    rc = emit_tables(&res[0], &v, 1, &one, &T);
    tables_free(&T);
    own_free(&o);
    if (rc == TSR_OK && res[0].kind == 3) {
        res[0].arr.ndim = nd;
        memcpy(res[0].arr.shape, sh, sizeof(int64_t) * (size_t)nd);
        c_strides(nd, sh, tsr_itemsize(res[0].arr.dtype), res[0].arr.strides);
        if (nd == 0) {
            tsr_result s;
            if (result_scalar(&s, (const char *)res[0].arr.data, res[0].arr.dtype) == TSR_OK) { fn_results_free(&res[0], 1); res[0] = s; }
        }
    }
    return rc;
}

/* ================================================================ axes */

static int r_expand_dims(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int64_t raw[TSR_MAXDIM + 1];
    int n;
    if ((rc = int_list(argk(args, nargs, 1), "axis", raw, TSR_MAXDIM + 1, &n, NULL)) < 0) return rc;
    const int ond = n + h.a.ndim;
    if (ond > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    int axes[TSR_MAXDIM + 1];
    char mark[TSR_MAXDIM] = {0};
    for (int i = 0; i < n; i++) {
        if ((rc = norm_axis(raw[i], ond, &axes[i])) < 0) return rc;
        if (mark[axes[i]]) { fn_set_error("repeated axis"); return TSR_EARG; }
        mark[axes[i]] = 1;
    }
    tsr_array v = h.a;
    int k = 0;
    int64_t sh[TSR_MAXDIM], st[TSR_MAXDIM];
    for (int d = 0; d < ond; d++) {
        if (mark[d]) { sh[d] = 1; st[d] = 0; }
        else { sh[d] = h.a.shape[k]; st[d] = h.a.strides[k]; k++; }
    }
    v.ndim = ond;
    memcpy(v.shape, sh, sizeof sh);
    memcpy(v.strides, st, sizeof st);
    return emit(&res[0], &v, -1);
}

static int r_squeeze(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    char drop[TSR_MAXDIM] = {0};
    const tsr_arg *aa = argk(args, nargs, 1);
    if (is_none(aa)) {
        for (int d = 0; d < v.ndim; d++) drop[d] = v.shape[d] == 1;
    } else {
        int axes[TSR_MAXDIM + 1], n;
        int64_t one;
        /* a 0-d array accepts a single axis 0 or -1 (numpy's PyArray_ConvertMultiAxis) */
        if (v.ndim == 0 && (aa->kind == 1 || aa->kind == 4 || (aa->kind == 3 && aa->arr.ndim == 0)) && int_param(aa, "axis", &one) == TSR_OK && (one == 0 || one == -1))
            return emit(&res[0], &v, -1);
        if ((rc = axis_tuple(aa, v.ndim, axes, &n, 0, NULL)) < 0) return rc;
        for (int i = 0; i < n; i++) {
            if (v.shape[axes[i]] != 1) { fn_set_error("cannot select an axis to squeeze out which has size not equal to one"); return TSR_EARG; }
            drop[axes[i]] = 1;
        }
    }
    int k = 0;
    for (int d = 0; d < v.ndim; d++) if (!drop[d]) { v.shape[k] = v.shape[d]; v.strides[k] = v.strides[d]; k++; }
    v.ndim = k;
    return emit(&res[0], &v, -1);
}

static int r_swapaxes(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int64_t a1, a2;
    int i, j;
    if ((rc = int_param(argk(args, nargs, 1), "axis1", &a1)) < 0 || (rc = int_param(argk(args, nargs, 2), "axis2", &a2)) < 0) return rc;
    if ((rc = norm_axis(a1, h.a.ndim, &i)) < 0 || (rc = norm_axis(a2, h.a.ndim, &j)) < 0) return rc;
    tsr_array v = h.a;
    v_swap(&v, i, j);
    return emit(&res[0], &v, -1);
}

static int r_matrix_transpose(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    if (h.a.ndim < 2) { fn_set_error("Input array must be at least 2-dimensional, but it is %d", h.a.ndim); return TSR_EARG; }
    tsr_array v = h.a;
    v_swap(&v, v.ndim - 2, v.ndim - 1);
    return emit(&res[0], &v, -1);
}

static int r_moveaxis(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    const int nd = h.a.ndim;
    int src[TSR_MAXDIM + 1], dst[TSR_MAXDIM + 1], ns, ndst;
    if ((rc = axis_tuple(argk(args, nargs, 1), nd, src, &ns, 0, "source")) < 0) return rc;
    if ((rc = axis_tuple(argk(args, nargs, 2), nd, dst, &ndst, 0, "destination")) < 0) return rc;
    if (ns != ndst) { fn_set_error("`source` and `destination` arguments must have the same number of elements"); return TSR_EARG; }
    int order[TSR_MAXDIM], no = 0;
    for (int d = 0; d < nd; d++) {
        int in = 0;
        for (int i = 0; i < ns; i++) if (src[i] == d) in = 1;
        if (!in) order[no++] = d;
    }
    /* for dest, src in sorted(zip(destination, source)): order.insert(dest, src) */
    int idx[TSR_MAXDIM + 1];
    for (int i = 0; i < ns; i++) idx[i] = i;
    for (int i = 1; i < ns; i++)
        for (int j = i; j > 0 && (dst[idx[j]] < dst[idx[j - 1]] || (dst[idx[j]] == dst[idx[j - 1]] && src[idx[j]] < src[idx[j - 1]])); j--) {
            int t = idx[j]; idx[j] = idx[j - 1]; idx[j - 1] = t;
        }
    for (int i = 0; i < ns; i++) {
        int pos = dst[idx[i]];
        if (pos > no) pos = no;
        memmove(order + pos + 1, order + pos, sizeof(int) * (size_t)(no - pos));
        order[pos] = src[idx[i]];
        no++;
    }
    tsr_array v = h.a;
    v_transpose(&v, order);
    return emit(&res[0], &v, -1);
}

static int r_rollaxis(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    const int n = h.a.ndim;
    int64_t a64, start = 0;
    int axis;
    if ((rc = int_param(argk(args, nargs, 1), "axis", &a64)) < 0 || (rc = norm_axis(a64, n, &axis)) < 0) return rc;
    if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "start", &start)) < 0) return rc;
    const int64_t given = start;
    if (start < 0) start += n;
    if (!(start >= 0 && start < n + 1)) {
        fn_set_error("'start' arg requires %d <= start < %d, but %lld was passed in", -n, n + 1, (long long)(given < 0 ? given + n : given));
        return TSR_EARG;
    }
    if (axis < start) start--;
    tsr_array v = h.a;
    if (axis != start) {
        int axes[TSR_MAXDIM], k = 0;
        for (int d = 0; d < n; d++) if (d != axis) axes[k++] = d;
        memmove(axes + start + 1, axes + start, sizeof(int) * (size_t)(n - 1 - start));
        axes[start] = axis;
        v_transpose(&v, axes);
    }
    return emit(&res[0], &v, -1);
}

/* atleast_1d / 2d / 3d views of one array */
static int atleast(tsr_array *v, int k)
{
    int rc = TSR_OK;
    if (k == 1) { if (v->ndim == 0) rc = v_insert_axis(v, 0); }
    else if (k == 2) {
        if (v->ndim == 0) { rc = v_insert_axis(v, 0); if (rc == 0) rc = v_insert_axis(v, 0); }
        else if (v->ndim == 1) rc = v_insert_axis(v, 0);
    } else {
        if (v->ndim == 0) { for (int i = 0; i < 3 && rc == 0; i++) rc = v_insert_axis(v, 0); }
        else if (v->ndim == 1) { rc = v_insert_axis(v, 0); if (rc == 0) rc = v_insert_axis(v, 2); }
        else if (v->ndim == 2) rc = v_insert_axis(v, 2);
    }
    return rc;
}

/* several result arrays of `elems` elements in total: MemoryError when they cannot all be held */
static int check_total(double elems, int64_t itemsize)
{
    if (tsr_fits(elems * (double)itemsize)) return TSR_OK;
    fn_set_error("Unable to allocate %.3g GiB for the results", elems * (double)itemsize / 1073741824.0);
    return TSR_ENOMEM;
}

/* emit a list of views: one -> the array, several -> a tuple (numpy's atleast_*, broadcast_arrays keeps tuple) */
static int emit_list(tsr_result *r, tsr_array *vs, int n, int single_unwrap, int dto)
{
    double total = 0;
    for (int i = 0; i < n; i++) total += (double)tsr_shape_size(vs[i].ndim, vs[i].shape) * (double)tsr_itemsize(dto < 0 ? vs[i].dtype : dto);
    if (check_total(total, 1) < 0) return TSR_ENOMEM;
    if (n == 1 && single_unwrap) return emit(r, &vs[0], dto);
    tsr_result *items = fn_result_seq(r, n);
    if (!items) return TSR_ENOMEM;
    for (int i = 0; i < n; i++) {
        int rc = emit(&items[i], &vs[i], dto);
        if (rc < 0) return rc;                           /* the caller frees the partial sequence */
    }
    return TSR_OK;
}

static int r_atleast(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int k = *(const int *)ctx;
    arrlist L;
    int rc = al_get(argk(args, nargs, 0), &L, "arys");
    if (rc < 0) return rc;
    tsr_array *vs = (tsr_array *)tsr_alloc((int64_t)(L.n + 1) * (int64_t)sizeof(tsr_array));
    if (!vs) { al_free(&L); return TSR_ENOMEM; }
    for (int i = 0; i < L.n && rc == 0; i++) { vs[i] = L.h[i].a; rc = atleast(&vs[i], k); }
    if (rc == 0) rc = emit_list(&res[0], vs, L.n, 1, -1);
    tsr_free(vs, (int64_t)(L.n + 1) * (int64_t)sizeof(tsr_array));
    al_free(&L);
    return rc;
}
static const int ONE = 1, TWO = 2, THREE = 3;

/* a shape argument: an int or a sequence of non-negative ints; count_only: the shape only has to count its
   elements in int64 (numpy's intp), as for ravel_multi_index and unravel_index, which allocate nothing of it */
static int shape_arg_ex(const tsr_arg *a, const char *name, int32_t *nd, int64_t *sh, const char *count_only)
{
    int n;
    int64_t v[TSR_MAXDIM + 1];
    int rc = int_list(a, name, v, TSR_MAXDIM + 1, &n, NULL);
    if (rc < 0) return rc;
    if (n > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    for (int i = 0; i < n; i++) if (v[i] < 0) { fn_set_error("negative dimensions are not allowed"); return TSR_EARG; }
    *nd = n;
    memcpy(sh, v, sizeof(int64_t) * (size_t)n);
    if (count_only) {
        int64_t prod = 1;
        for (int i = 0; i < n; i++) if (sh[i] == 0) prod = 0;
        for (int i = 0; i < n && prod != 0; i++) {
            if (prod > INT64_MAX / sh[i]) { fn_set_error("%s", count_only); return TSR_EARG; }
            prod *= sh[i];
        }
        return TSR_OK;
    }
    if (tsr_shape_size(n, sh) < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    return TSR_OK;
}
static int shape_arg(const tsr_arg *a, const char *name, int32_t *nd, int64_t *sh) { return shape_arg_ex(a, name, nd, sh, NULL); }

static int r_broadcast_to(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "array");
    if (rc < 0) return rc;
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    const tsr_arg *s = argk(args, nargs, 1);
    {   /* numpy: negative entries -> "all elements of broadcast shape must be non-negative" */
        int n;
        int64_t v[TSR_MAXDIM + 1];
        if ((rc = int_list(s, "shape", v, TSR_MAXDIM + 1, &n, NULL)) < 0) return rc;
        for (int i = 0; i < n; i++) if (v[i] < 0) { fn_set_error("all elements of broadcast shape must be non-negative"); return TSR_EARG; }
    }
    if ((rc = shape_arg(s, "shape", &nd, sh)) < 0) return rc;
    if (h.a.ndim > nd) { fn_set_error("input operand has more dimensions than allowed by the axis remapping"); return TSR_EARG; }
    tsr_array v = h.a;
    if (v_broadcast(&v, nd, sh) < 0) {
        fn_set_error("operands could not be broadcast together with remapped shapes [original->remapped]");
        return TSR_EARG;
    }
    return emit(&res[0], &v, -1);
}

static int r_broadcast_arrays(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    arrlist L;
    int rc = al_get(argk(args, nargs, 0), &L, "args");
    if (rc < 0) return rc;
    int32_t nd = 0;
    int64_t sh[TSR_MAXDIM];
    for (int i = 0; i < L.n && rc == 0; i++) rc = bshape(nd, sh, L.h[i].a.ndim, L.h[i].a.shape, &nd, sh);
    tsr_array *vs = NULL;
    if (rc == 0) {
        vs = (tsr_array *)tsr_alloc((int64_t)(L.n + 1) * (int64_t)sizeof(tsr_array));
        if (!vs) rc = TSR_ENOMEM;
    }
    for (int i = 0; i < L.n && rc == 0; i++) { vs[i] = L.h[i].a; rc = v_broadcast(&vs[i], nd, sh); }
    if (rc == 0) rc = emit_list(&res[0], vs, L.n, 0, -1);
    if (vs) tsr_free(vs, (int64_t)(L.n + 1) * (int64_t)sizeof(tsr_array));
    al_free(&L);
    return rc;
}

static int r_broadcast_shapes(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_arg *a = argk(args, nargs, 0);
    int32_t nd = 0;
    int64_t sh[TSR_MAXDIM];
    const int64_t n = a->kind == 5 ? a->count : 0;
    for (int64_t i = 0; i < n; i++) {
        int32_t k;
        int64_t s[TSR_MAXDIM];
        int rc;
        {
            int m;
            int64_t v[TSR_MAXDIM + 1];
            if ((rc = int_list(&a->items[i], "shape", v, TSR_MAXDIM + 1, &m, NULL)) < 0) return rc;
            for (int j = 0; j < m; j++) if (v[j] < 0) { fn_set_error("negative dimensions are not allowed"); return TSR_EARG; }
        }
        if ((rc = shape_arg(&a->items[i], "shape", &k, s)) < 0) return rc;
        if ((rc = bshape(nd, sh, k, s, &nd, sh)) < 0) return rc;
    }
    tsr_result *items = fn_result_seq(&res[0], nd);
    if (!items) return TSR_ENOMEM;
    for (int d = 0; d < nd; d++) fn_result_int(&items[d], sh[d]);
    return TSR_OK;
}

/* ================================================================ joining */

/* numpy.concatenate of views: same ndim, same shape except axis; result dtype dt (-1: promoted); casting */
static int concat_views(tsr_result *r, tsr_array *vs, int n, int axis_none, int64_t axis_raw, int dt, int casting)
{
    if (n == 0) { fn_set_error("need at least one array to concatenate"); return TSR_EARG; }
    owned *own = NULL;
    int rc = TSR_OK;
    if (axis_none) {
        own = (owned *)tsr_calloc((int64_t)n * (int64_t)sizeof(owned) + 64);
        if (!own) return TSR_ENOMEM;
        for (int i = 0; i < n && rc == 0; i++) { tsr_array t = vs[i]; rc = v_ravel(&t, &vs[i], &own[i]); }
        axis_raw = 0;
    }
    if (rc == 0 && vs[0].ndim == 0) { fn_set_error("zero-dimensional arrays cannot be concatenated"); rc = TSR_EARG; }
    int axis = 0;
    if (rc == 0) rc = norm_axis(axis_raw, vs[0].ndim, &axis);
    int rdt = vs[0].dtype;
    for (int i = 0; i < n && rc == 0; i++) {
        if (vs[i].ndim != vs[0].ndim) {
            fn_set_error("all the input arrays must have same number of dimensions, but the array at index 0 has %d dimension(s) and the array at index %d has %d dimension(s)",
                         vs[0].ndim, i, vs[i].ndim);
            rc = TSR_EARG;
            break;
        }
        for (int d = 0; d < vs[0].ndim; d++)
            if (d != axis && vs[i].shape[d] != vs[0].shape[d]) {
                fn_set_error("all the input array dimensions except for the concatenation axis must match exactly, but along dimension %d, the array at index 0 has size %lld and the array at index %d has size %lld",
                             d, (long long)vs[0].shape[d], i, (long long)vs[i].shape[d]);
                rc = TSR_EARG;
                break;
            }
        rdt = promote2(rdt, vs[i].dtype);
    }
    if (rc == 0 && dt >= 0) {
        for (int i = 0; i < n; i++)
            if (!can_cast(vs[i].dtype, dt, casting)) {
                static const char *const C[] = {"no", "equiv", "safe", "same_kind", "unsafe"};
                fn_set_error("Cannot cast array data from dtype('%s') to dtype('%s') according to the rule '%s'", dtname(vs[i].dtype), dtname(dt), C[casting]);
                rc = TSR_EARG;
                break;
            }
        rdt = dt;
    } else if (rc == 0 && casting < 3) {
        for (int i = 0; i < n; i++)
            if (!can_cast(vs[i].dtype, rdt, casting)) {
                static const char *const C[] = {"no", "equiv", "safe", "same_kind", "unsafe"};
                fn_set_error("Cannot cast array data from dtype('%s') to dtype('%s') according to the rule '%s'", dtname(vs[i].dtype), dtname(rdt), C[casting]);
                rc = TSR_EARG;
                break;
            }
    }
    if (rc == 0) {
        int64_t sh[TSR_MAXDIM];
        memcpy(sh, vs[0].shape, sizeof(int64_t) * (size_t)vs[0].ndim);
        int64_t tot = 0;
        for (int i = 0; i < n; i++) {
            if (tot > INT64_MAX / 16 - vs[i].shape[axis]) { fn_set_error("array is too big"); rc = TSR_EARG; break; }
            tot += vs[i].shape[axis];
        }
        sh[axis] = tot;
        if (rc == 0 && tsr_shape_size(vs[0].ndim, sh) < 0) { fn_set_error("array is too big"); rc = TSR_EARG; }
        if (rc == 0) {
            char *p = (char *)fn_result_array(r, rdt, vs[0].ndim, sh);
            if (!p) rc = TSR_ENOMEM;
            else {
                int64_t st[TSR_MAXDIM];
                c_strides(vs[0].ndim, sh, tsr_itemsize(rdt), st);
                int64_t at = 0;
                for (int i = 0; i < n && rc == 0; i++) {
                    if (tsr_shape_size(vs[i].ndim, vs[i].shape) > 0) rc = copy_view(&vs[i], rdt, p + at * st[axis], st);
                    at += vs[i].shape[axis];
                }
            }
        }
    }
    if (own) {
        for (int i = 0; i < n; i++) own_free(&own[i]);
        tsr_free(own, (int64_t)n * (int64_t)sizeof(owned) + 64);
    }
    return rc;
}

/* the views of a list argument, with atleast_k applied (k 0: as given) */
static int list_views(const tsr_arg *a, const char *what, int k, tsr_array **vs, int *n, arrlist *L)
{
    int rc = al_get(a, L, what);
    if (rc < 0) return rc;
    *n = L->n;
    *vs = (tsr_array *)tsr_alloc((int64_t)(L->n + 1) * (int64_t)sizeof(tsr_array));
    if (!*vs) { al_free(L); return TSR_ENOMEM; }
    for (int i = 0; i < L->n && rc == 0; i++) { (*vs)[i] = L->h[i].a; if (k) rc = atleast(&(*vs)[i], k); }
    if (rc < 0) { tsr_free(*vs, (int64_t)(L->n + 1) * (int64_t)sizeof(tsr_array)); al_free(L); }
    return rc;
}

static void list_views_free(tsr_array *vs, int n, arrlist *L)
{
    tsr_free(vs, (int64_t)(n + 1) * (int64_t)sizeof(tsr_array));
    al_free(L);
}

/* concatenate(arrays, axis=0, dtype=None, casting='same_kind') */
static int r_concatenate(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    int dt, casting;
    int rc = arg_dtype(argk(args, nargs, 2), &dt);
    if (rc == 0) rc = arg_casting(argk(args, nargs, 3), &casting);
    if (rc < 0) return rc;
    const tsr_arg *aa = argk(args, nargs, 1);
    int64_t ax = 0;
    if (!is_none(aa) && (rc = int_param(aa, "axis", &ax)) < 0) return rc;
    tsr_array *vs;
    int n;
    arrlist L;
    if ((rc = list_views(argk(args, nargs, 0), "arrays", 0, &vs, &n, &L)) < 0) return rc;
    rc = concat_views(&res[0], vs, n, is_none(aa), ax, dt, casting);
    list_views_free(vs, n, &L);
    return rc;
}

/* stack(arrays, axis=0, dtype=None, casting='same_kind') */
static int r_stack(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    int dt, casting;
    int rc = arg_dtype(argk(args, nargs, 2), &dt);
    if (rc == 0) rc = arg_casting(argk(args, nargs, 3), &casting);
    if (rc < 0) return rc;
    int64_t ax = 0;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "axis", &ax)) < 0) return rc;
    tsr_array *vs;
    int n;
    arrlist L;
    if ((rc = list_views(argk(args, nargs, 0), "arrays", 0, &vs, &n, &L)) < 0) return rc;
    if (n == 0) { list_views_free(vs, n, &L); fn_set_error("need at least one array to stack"); return TSR_EARG; }
    for (int i = 1; i < n; i++)
        if (vs[i].ndim != vs[0].ndim || memcmp(vs[i].shape, vs[0].shape, sizeof(int64_t) * (size_t)vs[0].ndim) != 0) {
            list_views_free(vs, n, &L);
            fn_set_error("all input arrays must have the same shape");
            return TSR_EARG;
        }
    int axis;
    if ((rc = norm_axis(ax, vs[0].ndim + 1, &axis)) < 0) { list_views_free(vs, n, &L); return rc; }
    for (int i = 0; i < n && rc == 0; i++) rc = v_insert_axis(&vs[i], axis);
    if (rc == 0) rc = concat_views(&res[0], vs, n, 0, axis, dt, casting);
    list_views_free(vs, n, &L);
    return rc;
}

/* vstack / hstack (tup, dtype=None, casting='same_kind'); dstack / column_stack (tup) */
static int r_xstack(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;                 /* 0 vstack, 1 hstack, 2 dstack, 3 column_stack */
    int dt = -1, casting = 3, rc = TSR_OK;
    if (kind <= 1) {
        rc = arg_dtype(argk(args, nargs, 1), &dt);
        if (rc == 0) rc = arg_casting(argk(args, nargs, 2), &casting);
        if (rc < 0) return rc;
    }
    tsr_array *vs;
    int n;
    arrlist L;
    const int k = kind == 0 ? 2 : kind == 1 ? 1 : kind == 2 ? 3 : 0;
    if ((rc = list_views(argk(args, nargs, 0), "tup", k, &vs, &n, &L)) < 0) return rc;
    int64_t axis = kind == 0 ? 0 : kind == 2 ? 2 : 1;
    if (kind == 1 && n > 0 && vs[0].ndim == 1) axis = 0;
    if (kind == 3)
        for (int i = 0; i < n && rc == 0; i++) {
            if (vs[i].ndim == 0) { rc = v_insert_axis(&vs[i], 0); if (rc == 0) rc = v_insert_axis(&vs[i], 0); }
            else if (vs[i].ndim == 1) rc = v_insert_axis(&vs[i], 1);     /* array(arr, ndmin=2).T */
        }
    if (rc == 0) rc = concat_views(&res[0], vs, n, 0, axis, dt, casting);
    list_views_free(vs, n, &L);
    return rc;
}
static const int VSTACK = 0, HSTACK = 1, DSTACK = 2, COLUMN_STACK = 3;

static int r_unstack(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "x");
    if (rc < 0) return rc;
    if (h.a.ndim == 0) { fn_set_error("Input array must be at least 1-d."); return TSR_EARG; }
    int64_t a64 = 0;
    int ax;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "axis", &a64)) < 0) return rc;
    if ((rc = norm_axis(a64, h.a.ndim, &ax)) < 0) return rc;
    const int64_t n = h.a.shape[ax];
    tsr_result *items = fn_result_seq(&res[0], n);
    if (!items) return TSR_ENOMEM;
    for (int64_t i = 0; i < n; i++) {
        tsr_array v = h.a;
        v.offset += i * v.strides[ax];
        for (int d = ax; d < v.ndim - 1; d++) { v.shape[d] = v.shape[d + 1]; v.strides[d] = v.strides[d + 1]; }
        v.ndim--;
        if ((rc = emit(&items[i], &v, -1)) < 0) return rc;
    }
    return TSR_OK;
}

/* ================================================================ splitting */

/* split / array_split / hsplit / vsplit / dsplit (ary, indices_or_sections[, axis]) */
static int r_split(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;                 /* 0 split, 1 array_split, 2 hsplit, 3 vsplit, 4 dsplit */
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "ary");
    if (rc < 0) return rc;
    int64_t a64 = 0;
    if (kind == 2) {
        if (h.a.ndim == 0) { fn_set_error("hsplit only works on arrays of 1 or more dimensions"); return TSR_EARG; }
        a64 = h.a.ndim > 1 ? 1 : 0;
    } else if (kind == 3) {
        if (h.a.ndim < 2) { fn_set_error("vsplit only works on arrays of 2 or more dimensions"); return TSR_EARG; }
    } else if (kind == 4) {
        if (h.a.ndim < 3) { fn_set_error("dsplit only works on arrays of 3 or more dimensions"); return TSR_EARG; }
        a64 = 2;
    } else if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "axis", &a64)) < 0) return rc;
    int ax;
    if (h.a.ndim == 0) { fn_set_error("tuple index out of range"); return TSR_EARG; }
    if ((rc = norm_axis(a64, h.a.ndim, &ax)) < 0) { fn_set_error("tuple index out of range"); return rc; }
    const int64_t N = h.a.shape[ax];
    const tsr_arg *is = argk(args, nargs, 1);
    const int is_list = is->kind == 5 || (is->kind == 3 && is->arr.ndim >= 1);
    int64_t nsec, each = 0, extras = 0;
    int64_t *pts = NULL;                                 /* div_points for an index list, nsec + 1 of them */
    int64_t pts_bytes = 0;
    if (is_list) {
        const int64_t m = is->kind == 5 ? is->count : is->arr.shape[0];
        if (is->kind == 3 && is->arr.ndim > 1) { fn_set_error("slice indices must be integers or None or have an __index__ method"); return TSR_EARG; }
        nsec = m + 1;
        pts_bytes = (nsec + 1) * 8 + 64;
        pts = (int64_t *)tsr_alloc(pts_bytes);
        if (!pts) return TSR_ENOMEM;
        pts[0] = 0;
        for (int64_t i = 0; i < m; i++) {
            int64_t v = 0;
            if (is->kind == 5) rc = int_param(&is->items[i], "indices_or_sections", &v);
            else if (!is_intlike(is->arr.dtype)) { fn_set_error("slice indices must be integers or None or have an __index__ method"); rc = TSR_EARG; }
            else v = rd_i64(elem0(&is->arr) + i * is->arr.strides[0], is->arr.dtype);
            if (rc < 0) { tsr_free(pts, pts_bytes); if (is->kind == 5) fn_set_error("slice indices must be integers or None or have an __index__ method"); return rc; }
            pts[i + 1] = v;
        }
        pts[nsec] = N;
    } else {
        harr hs;
        if ((rc = get_arr(is, &hs, "indices_or_sections")) < 0) return rc;
        const double sv = rd_f64(elem0(&hs.a), hs.a.dtype);
        int64_t s;
        if (is_intlike(hs.a.dtype)) s = is->kind == 1 && (is->flags & 1) ? is->ival : rd_i64(elem0(&hs.a), hs.a.dtype);
        else {
            if (!(sv > -9.2e18 && sv < 9.2e18)) { fn_set_error("cannot convert float NaN to integer"); return TSR_EARG; }
            s = (int64_t)sv;
        }
        if (kind != 1) {                                  /* split: N % sections must be 0 (float modulo for floats) */
            if (is_intlike(hs.a.dtype)) {
                if (s == 0) { fn_set_error("integer modulo by zero"); return TSR_EARG; }
                if (N % s != 0) { fn_set_error("array split does not result in an equal division"); return TSR_EARG; }   /* N >= 0: no overflow */
            } else {
                if (sv == 0) { fn_set_error("float modulo by zero"); return TSR_EARG; }
                if (fmod((double)N, sv) != 0) { fn_set_error("array split does not result in an equal division"); return TSR_EARG; }
            }
        }
        if (s <= 0) { fn_set_error("number sections must be larger than 0."); return TSR_EARG; }
        nsec = s;
        each = N / s;
        extras = N % s;
    }
    /* the result list first: a section count the memory cannot hold fails here, before any work */
    tsr_result *items = fn_result_seq(&res[0], nsec);
    if (!items) { if (pts) tsr_free(pts, pts_bytes); return TSR_ENOMEM; }
    int64_t at = 0;
    for (int64_t i = 0; i < nsec && rc == 0; i++) {
        /* python slicing sary[st:end] of the axis */
        int64_t st, en;
        if (pts) { st = pts[i]; en = pts[i + 1]; }
        else { st = at; en = at + (i < extras ? each + 1 : each); at = en; }
        if (st < 0) { st += N; if (st < 0) st = 0; } else if (st > N) st = N;
        if (en < 0) { en += N; if (en < 0) en = 0; } else if (en > N) en = N;
        tsr_array v = h.a;
        v_slice(&v, ax, st, en);
        rc = emit(&items[i], &v, -1);
    }
    if (pts) tsr_free(pts, pts_bytes);
    return rc;
}
static const int SPLIT = 0, ARRAY_SPLIT = 1, HSPLIT = 2, VSPLIT = 3, DSPLIT = 4;

/* ================================================================ append, insert, delete, trim_zeros */

static int r_append(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha, hv;
    int rc = get_arr(argk(args, nargs, 0), &ha, "arr");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hv, "values");
    if (rc < 0) return rc;
    tsr_array vs[2] = {ha.a, hv.a};
    const tsr_arg *aa = argk(args, nargs, 2);
    if (is_none(aa)) return concat_views(&res[0], vs, 2, 1, 0, -1, 3);
    int64_t ax;
    if ((rc = int_param(aa, "axis", &ax)) < 0) return rc;
    return concat_views(&res[0], vs, 2, 0, ax, -1, 3);
}

/* the ints of an index argument (a scalar or a 1-D integer array); *scalar for a Python/0-d int */
static int index_list(const tsr_arg *a, int64_t **out, int64_t *n, int64_t *bytes, int *scalar, int *is_bool, const char *fn)
{
    *out = NULL;
    *n = 0;
    *scalar = 0;
    *is_bool = 0;
    harr h;
    int rc;
    if (a->kind == 5) {                                   /* a PHP list of ints */
        *n = a->count;
        *bytes = (*n + 1) * 8 + 64;
        *out = (int64_t *)tsr_alloc(*bytes);
        if (!*out) return TSR_ENOMEM;
        for (int64_t i = 0; i < *n; i++)
            if ((rc = int_param(&a->items[i], "obj", &(*out)[i])) < 0) { tsr_free(*out, *bytes); *out = NULL; return rc; }
        return TSR_OK;
    }
    if ((rc = get_arr(a, &h, "obj")) < 0) return rc;
    if (h.a.ndim > 1) {
        fn_set_error("index array argument obj to %s must be one dimensional or scalar", fn);
        return TSR_EARG;
    }
    const int64_t m = h.a.ndim ? h.a.shape[0] : 1;
    if (h.a.dtype == TSR_BOOL) *is_bool = 1;
    else if (!is_intdt(h.a.dtype) && m > 0) {
        fn_set_error("arrays used as indices must be of integer (or boolean) type");
        return TSR_EARG;
    }
    *n = m;
    *scalar = h.a.ndim == 0;
    *bytes = (m + 1) * 8 + 64;
    *out = (int64_t *)tsr_alloc(*bytes);
    if (!*out) return TSR_ENOMEM;
    for (int64_t i = 0; i < m; i++) (*out)[i] = rd_i64(elem0(&h.a) + (h.a.ndim ? i * h.a.strides[0] : 0), h.a.dtype);
    if (*scalar && a->kind == 1 && (a->flags & 1)) (*out)[0] = a->ival;
    return TSR_OK;
}

static int r_delete(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "arr");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    const tsr_arg *aa = argk(args, nargs, 2);
    int ax;
    if (is_none(aa)) {
        if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc;
        ax = 0;
    } else {
        int64_t a64;
        if ((rc = int_param(aa, "axis", &a64)) < 0 || (rc = norm_axis(a64, v.ndim, &ax)) < 0) return rc;
    }
    const int64_t N = v.shape[ax];
    int64_t *idx, n, bytes;
    int scalar, isb;
    const tsr_arg *ob = argk(args, nargs, 1);
    if ((rc = index_list(ob, &idx, &n, &bytes, &scalar, &isb, "delete")) < 0) { own_free(&o); return rc; }
    if (ob->kind == 4) isb = 0, scalar = 1;               /* a Python bool is an int here? numpy: np.asarray(True) -> bool array */
    if (ob->kind == 4) { isb = 1; scalar = 1; }
    uint8_t *keep = (uint8_t *)tsr_alloc(N + 64);
    if (!keep) { tsr_free(idx, bytes); own_free(&o); return TSR_ENOMEM; }
    memset(keep, 1, (size_t)N);
    if (isb) {
        if (scalar || n != N) {
            fn_set_error("boolean array argument obj to delete must be one dimensional and match the axis length of %lld", (long long)N);
            rc = TSR_EARG;
        } else for (int64_t i = 0; i < n; i++) keep[i] = !idx[i];
    } else {
        const int single = scalar || n == 1;
        for (int64_t i = 0; i < n && rc == 0; i++) {
            int64_t j = idx[i];
            if (j < -N || j >= N) {
                if (single) fn_set_error("index %lld is out of bounds for axis %d with size %lld", (long long)j, ax, (long long)N);
                else fn_set_error("index %lld is out of bounds for axis 0 with size %lld", (long long)j, (long long)N);
                rc = TSR_EARG;
                break;
            }
            keep[j < 0 ? j + N : j] = 0;
        }
    }
    if (rc == 0) {
        int64_t cnt = 0;
        for (int64_t i = 0; i < N; i++) cnt += keep[i];
        int64_t osh[TSR_MAXDIM];
        memcpy(osh, v.shape, sizeof(int64_t) * (size_t)v.ndim);
        osh[ax] = cnt;
        tables T;
        if ((rc = tables_new(&T, v.ndim, osh)) == 0) {
            for (int d = 0; d < v.ndim; d++) {
                if (d != ax) { for (int64_t j = 0; j < osh[d]; j++) T.t[d][j] = j * v.strides[d]; continue; }
                int64_t k = 0;
                for (int64_t i = 0; i < N; i++) if (keep[i]) T.t[d][k++] = i * v.strides[d];
            }
            rc = emit_tables(&res[0], &v, v.ndim, osh, &T);
            tables_free(&T);
        }
    }
    tsr_free(keep, N + 64);
    tsr_free(idx, bytes);
    own_free(&o);
    return rc;
}

/* stable argsort of int64 (insertion for short, merge otherwise) */
static void argsort_i64(const int64_t *v, int64_t n, int64_t *ord, int64_t *tmp)
{
    for (int64_t i = 0; i < n; i++) ord[i] = i;
    for (int64_t w = 1; w < n; w *= 2) {
        for (int64_t lo = 0; lo < n; lo += 2 * w) {
            int64_t mid = lo + w < n ? lo + w : n, hi = lo + 2 * w < n ? lo + 2 * w : n, a = lo, b = mid, k = lo;
            while (a < mid && b < hi) tmp[k++] = v[ord[b]] < v[ord[a]] ? ord[b++] : ord[a++];
            while (a < mid) tmp[k++] = ord[a++];
            while (b < hi) tmp[k++] = ord[b++];
        }
        memcpy(ord, tmp, sizeof(int64_t) * (size_t)n);
    }
}

/* numpy's assignment broadcasting: a value with more dimensions than the target loses leading length-1 axes */
static void strip_leading_ones(tsr_array *v, int nd)
{
    while (v->ndim > nd && v->shape[0] == 1) {
        memmove(v->shape, v->shape + 1, sizeof(int64_t) * (size_t)(v->ndim - 1));
        memmove(v->strides, v->strides + 1, sizeof(int64_t) * (size_t)(v->ndim - 1));
        v->ndim--;
    }
}

/* insert(arr, obj, values, axis=None) */
static int r_insert(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h, hv;
    int rc = get_arr(argk(args, nargs, 0), &h, "arr");
    if (rc == 0) rc = get_arr(argk(args, nargs, 2), &hv, "values");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    const tsr_arg *aa = argk(args, nargs, 3);
    int ax;
    if (is_none(aa)) {
        if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc;
        ax = 0;
    } else {
        int64_t a64;
        if ((rc = int_param(aa, "axis", &a64)) < 0 || (rc = norm_axis(a64, v.ndim, &ax)) < 0) return rc;
    }
    const int nd = v.ndim;
    const int64_t N = v.shape[ax];
    const int dt = v.dtype;
    /* Python floats/NaN into an integer array: numpy's array(values, dtype=int) raises */
    const tsr_arg *va = argk(args, nargs, 2);
    if (va->kind == 1 && !(va->flags & 1) && is_intlike(dt) && !isfinite(va->num)) {
        own_free(&o);
        fn_set_error("cannot convert float %s to integer", isnan(va->num) ? "NaN" : "infinity");
        return TSR_EARG;
    }
    int64_t *idx, n, bytes;
    int scalar, isb;
    const tsr_arg *ob = argk(args, nargs, 1);
    if ((rc = index_list(ob, &idx, &n, &bytes, &scalar, &isb, "insert")) < 0) { own_free(&o); return rc; }
    if (isb) {                                           /* flatnonzero(obj) */
        if (scalar) { tsr_free(idx, bytes); own_free(&o); fn_set_error("boolean array argument obj to insert must be one dimensional"); return TSR_EARG; }
        int64_t k = 0;
        for (int64_t i = 0; i < n; i++) if (idx[i]) idx[k++] = i;
        n = k;
    }
    tsr_array val = hv.a;
    int64_t osh[TSR_MAXDIM];
    memcpy(osh, v.shape, sizeof(int64_t) * (size_t)nd);
    char *out = NULL;
    int64_t ost[TSR_MAXDIM];
    if (n == 1) {                                        /* one index: a block of values */
        int64_t index = idx[0];
        if (index < -N || index > N) {
            fn_set_error("index %lld is out of bounds for axis %d with size %lld", (long long)index, ax, (long long)N);
            rc = TSR_EARG;
            goto done;
        }
        if (index < 0) index += N;
        while (val.ndim < nd) if ((rc = v_insert_axis(&val, 0)) < 0) goto done;   /* ndmin=arr.ndim */
        if (scalar && !isb) {                             /* values = moveaxis(values, 0, axis) */
            int perm[TSR_MAXDIM], k = 0;
            for (int d = 1; d < val.ndim; d++) { if (k == ax) perm[k++] = 0; perm[k++] = d; }
            if (k == ax) perm[k++] = 0;
            v_transpose(&val, perm);
        }
        const int64_t numnew = val.shape[ax];             /* values.shape[axis], before the assignment broadcasts */
        strip_leading_ones(&val, nd);
        if (val.ndim > nd) { fn_set_error("could not broadcast input array from shape into the insertion slot"); rc = TSR_EARG; goto done; }
        osh[ax] = N + numnew;
        /* broadcast values to the slot shape (the slot has numnew along axis) */
        int64_t slot[TSR_MAXDIM];
        memcpy(slot, v.shape, sizeof(int64_t) * (size_t)nd);
        slot[ax] = numnew;
        if (v_broadcast(&val, nd, slot) < 0) {
            fn_set_error("could not broadcast input array into the insertion slot");
            rc = TSR_EARG;
            goto done;
        }
        if (tsr_shape_size(nd, osh) < 0) { fn_set_error("array is too big"); rc = TSR_EARG; goto done; }
        out = (char *)fn_result_array(&res[0], dt, nd, osh);
        if (!out) { rc = TSR_ENOMEM; goto done; }
        if (tsr_shape_size(nd, osh) == 0) goto done;
        c_strides(nd, osh, tsr_itemsize(dt), ost);
        tsr_array part = v;
        v_slice(&part, ax, 0, index);
        if (tsr_shape_size(nd, part.shape) > 0 && (rc = copy_view(&part, dt, out, ost)) < 0) goto done;
        if (tsr_shape_size(nd, val.shape) > 0 && (rc = copy_view(&val, dt, out + index * ost[ax], ost)) < 0) goto done;
        part = v;
        v_slice(&part, ax, index, N);
        if (tsr_shape_size(nd, part.shape) > 0) rc = copy_view(&part, dt, out + (index + numnew) * ost[ax], ost);
        goto done;
    }
    {   /* several indices: sorted placement, values broadcast to (..., numnew, ...) */
        const int64_t numnew = n;
        for (int64_t i = 0; i < n; i++) {
            if (idx[i] < -N || idx[i] > N) {
                fn_set_error("index %lld is out of bounds for axis %d with size %lld", (long long)idx[i], ax, (long long)(N + numnew));
                rc = TSR_EARG;
                goto done;
            }
            if (idx[i] < 0) idx[i] += N;
        }
        const int64_t tb = (n + 1) * 16 + 64;
        int64_t *ord = (int64_t *)tsr_alloc(tb);
        if (!ord) { rc = TSR_ENOMEM; goto done; }
        argsort_i64(idx, n, ord, ord + n + 1);
        for (int64_t i = 0; i < n; i++) idx[ord[i]] += i;
        tsr_free(ord, tb);
        osh[ax] = N + numnew;
        if (tsr_shape_size(nd, osh) < 0) { fn_set_error("array is too big"); rc = TSR_EARG; goto done; }
        uint8_t *old = (uint8_t *)tsr_alloc(osh[ax] + 64);
        if (!old) { rc = TSR_ENOMEM; goto done; }
        memset(old, 1, (size_t)osh[ax]);
        for (int64_t i = 0; i < n; i++) old[idx[i]] = 0;
        /* new[..., indices, ...] = values: values broadcast to the index-selected shape */
        int64_t slot[TSR_MAXDIM];
        memcpy(slot, v.shape, sizeof(int64_t) * (size_t)nd);
        slot[ax] = numnew;
        strip_leading_ones(&val, nd);
        if (val.ndim > nd || v_broadcast(&val, nd, slot) < 0) {
            tsr_free(old, osh[ax] + 64);
            fn_set_error("shape mismatch: value array could not be broadcast to indexing result");
            rc = TSR_EARG;
            goto done;
        }
        out = (char *)fn_result_array(&res[0], dt, nd, osh);
        if (!out) { tsr_free(old, osh[ax] + 64); rc = TSR_ENOMEM; goto done; }
        if (tsr_shape_size(nd, osh) == 0) { tsr_free(old, osh[ax] + 64); goto done; }
        c_strides(nd, osh, tsr_itemsize(dt), ost);
        /* assignment order: values at each index (later duplicates win: numpy's fancy assignment) */
        for (int64_t i = 0; i < n && rc == 0; i++) {
            tsr_array part = val;
            v_slice(&part, ax, i, i + 1);
            int64_t pst[TSR_MAXDIM];
            memcpy(pst, ost, sizeof pst);
            rc = copy_view(&part, dt, out + idx[i] * ost[ax], pst);
        }
        int64_t k = 0;
        for (int64_t j = 0; j < osh[ax] && rc == 0; j++) {
            if (!old[j]) continue;
            tsr_array part = v;
            v_slice(&part, ax, k, k + 1);
            rc = copy_view(&part, dt, out + j * ost[ax], ost);
            k++;
        }
        tsr_free(old, osh[ax] + 64);
    }
done:
    tsr_free(idx, bytes);
    own_free(&o);
    if (rc == TSR_OK && res[0].kind == 3 && res[0].arr.ndim == 0) {
        tsr_result s;
        if (result_scalar(&s, (const char *)res[0].arr.data, res[0].arr.dtype) == TSR_OK) { fn_results_free(&res[0], 1); res[0] = s; }
    }
    return rc;
}

static int r_trim_zeros(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "filt");
    if (rc < 0) return rc;
    const tsr_arg *ta = argk(args, nargs, 1);
    const char *trim = ta->kind == 2 && ta->str ? ta->str : "fb";
    if (ta->kind != 0 && ta->kind != 2) { fn_set_error("trim must be a string"); return TSR_EARG; }
    char low[8] = {0};
    const size_t tl = strlen(trim);
    for (size_t i = 0; i < tl && i < 7; i++) low[i] = (char)(trim[i] >= 'A' && trim[i] <= 'Z' ? trim[i] + 32 : trim[i]);
    if (tl > 2 || !(strcmp(low, "fb") == 0 || strcmp(low, "bf") == 0 || strcmp(low, "f") == 0 || strcmp(low, "b") == 0)) {
        fn_set_error("unexpected character(s) in `trim`: '%s'", trim);
        return TSR_EARG;
    }
    const int front = strchr(low, 'f') != NULL, back = strchr(low, 'b') != NULL;
    tsr_array v = h.a;
    const int nd = v.ndim;
    char sel[TSR_MAXDIM] = {0};
    const tsr_arg *aa = argk(args, nargs, 2);
    if (is_none(aa)) { for (int d = 0; d < nd; d++) sel[d] = 1; }
    else {
        int axes[TSR_MAXDIM + 1], n;
        if ((rc = axis_tuple(aa, nd, axes, &n, 0, "axis")) < 0) return rc;
        for (int i = 0; i < n; i++) sel[axes[i]] = 1;
    }
    int any = 0;
    for (int d = 0; d < nd; d++) any |= sel[d];
    if (!any) return emit(&res[0], &v, -1);
    /* the bounding box of the non-zero entries (argwhere min/max) */
    int64_t lo[TSR_MAXDIM], hi[TSR_MAXDIM];
    for (int d = 0; d < nd; d++) { lo[d] = INT64_MAX; hi[d] = -1; }
    const int64_t total = tsr_shape_size(nd, v.shape);
    int64_t ix[TSR_MAXDIM] = {0};
    int found = 0;
    for (int64_t k = 0; k < total; k++) {
        int64_t off = 0;
        for (int d = 0; d < nd; d++) off += ix[d] * v.strides[d];
        if (truthy(elem0(&v) + off, v.dtype)) {
            found = 1;
            for (int d = 0; d < nd; d++) { if (ix[d] < lo[d]) lo[d] = ix[d]; if (ix[d] > hi[d]) hi[d] = ix[d]; }
        }
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < v.shape[d]) break; ix[d] = 0; }
    }
    for (int d = 0; d < nd; d++) {
        if (!sel[d]) continue;
        if (!found) { v_slice(&v, d, 0, 0); continue; }
        const int64_t s = front ? lo[d] : 0, e = back ? hi[d] + 1 : v.shape[d];
        v_slice(&v, d, s, e);
    }
    return emit(&res[0], &v, -1);
}

/* ================================================================ diagonals and triangles */

static int r_diag(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int flat = ctx ? *(const int *)ctx : 0;       /* diagflat */
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "v");
    if (rc < 0) return rc;
    int64_t k = 0;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "k", &k)) < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    if (flat) { if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc; }
    if (v.ndim == 1) {
        const int64_t s = v.shape[0];
        if (k > INT64_MAX / 4 || k < -INT64_MAX / 4 || s + (k < 0 ? -k : k) > (int64_t)3037000499) { own_free(&o); fn_set_error("array is too big"); return TSR_EARG; }
        const int64_t n = s + (k < 0 ? -k : k);
        const int64_t sh[2] = {n, n};
        char *p = (char *)new_result(&res[0], v.dtype, 2, sh);
        if (!p) { own_free(&o); return TSR_ENOMEM; }
        const int64_t isz = tsr_itemsize(v.dtype);
        for (int64_t i = 0; i < s; i++) {
            const int64_t r = k >= 0 ? i : i - k, c = k >= 0 ? i + k : i;
            memcpy(p + (r * n + c) * isz, elem0(&v) + i * v.strides[0], (size_t)isz);
        }
        own_free(&o);
        return TSR_OK;
    }
    if (v.ndim == 2) {                                   /* diagonal(v, k) */
        int64_t r0 = k >= 0 ? 0 : -k, c0 = k >= 0 ? k : 0;
        int64_t len = 0;
        if (r0 < v.shape[0] && c0 < v.shape[1]) len = (v.shape[0] - r0 < v.shape[1] - c0) ? v.shape[0] - r0 : v.shape[1] - c0;
        tsr_array d = v;
        d.ndim = 1;
        d.offset += (len ? r0 * v.strides[0] + c0 * v.strides[1] : 0);
        d.shape[0] = len;
        d.strides[0] = v.strides[0] + v.strides[1];
        return emit(&res[0], &d, -1);
    }
    own_free(&o);
    fn_set_error("Input must be 1- or 2-d.");
    return TSR_EARG;
}
static const int DIAGFLAT = 1;

/* the diagonal view of a (offset, axis1, axis2): the remaining axes, then the diagonal */
static int diag_view(const tsr_array *a, int64_t off, const tsr_arg *a1, const tsr_arg *a2, tsr_array *out)
{
    if (a->ndim < 2) { fn_set_error("diag requires an array of at least two dimensions"); return TSR_EARG; }
    int64_t x1 = 0, x2 = 1;
    int i1, i2, rc;
    if (!is_none(a1) && (rc = int_param(a1, "axis1", &x1)) < 0) return rc;
    if (!is_none(a2) && (rc = int_param(a2, "axis2", &x2)) < 0) return rc;
    if ((rc = norm_axis(x1, a->ndim, &i1)) < 0 || (rc = norm_axis(x2, a->ndim, &i2)) < 0) return rc;
    if (i1 == i2) { fn_set_error("axis1 and axis2 cannot be the same"); return TSR_EARG; }
    int64_t n1 = a->shape[i1], n2 = a->shape[i2], start = 0, len = 0;
    if (off >= 0) { if (off < n2) { len = n1 < n2 - off ? n1 : n2 - off; start = off * a->strides[i2]; } }
    else { if (off > -n1) { len = n1 + off < n2 ? n1 + off : n2; start = -off * a->strides[i1]; } }   /* no -INT64_MIN */
    *out = *a;
    int k = 0;
    for (int d = 0; d < a->ndim; d++) if (d != i1 && d != i2) { out->shape[k] = a->shape[d]; out->strides[k] = a->strides[d]; k++; }
    out->shape[k] = len;
    out->strides[k] = a->strides[i1] + a->strides[i2];
    out->ndim = k + 1;
    if (len > 0) out->offset += start;
    return TSR_OK;
}

static int r_diagonal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int64_t off = 0;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "offset", &off)) < 0) return rc;
    if (off < INT32_MIN || off > INT32_MAX) { fn_set_error("Python int too large to convert to C int"); return TSR_EARG; }
    tsr_array d;
    if ((rc = diag_view(&h.a, off, argk(args, nargs, 2), argk(args, nargs, 3), &d)) < 0) return rc;
    return emit(&res[0], &d, -1);
}

/* trace(a, offset=0, axis1=0, axis2=1, dtype=None): diagonal(...).sum(-1), NumPy's pairwise sum for floats */
static int r_trace(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int64_t off = 0;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "offset", &off)) < 0) return rc;
    if (off < INT32_MIN || off > INT32_MAX) { fn_set_error("Python int too large to convert to C int"); return TSR_EARG; }
    int dt;
    if ((rc = arg_dtype(argk(args, nargs, 4), &dt)) < 0) return rc;
    tsr_array d;
    if ((rc = diag_view(&h.a, off, argk(args, nargs, 2), argk(args, nargs, 3), &d)) < 0) return rc;
    const int src = d.dtype;
    if (dt < 0) dt = is_intlike(src) ? TSR_I64 : src;   /* add.reduce: small ints and bool sum in int64 */
    if (dt == TSR_C128 || src == TSR_C128) { fn_set_error("complex trace is not supported"); return TSR_ETYPE; }
    const int onD = d.ndim - 1;
    const int64_t len = d.shape[onD];
    int64_t osh[TSR_MAXDIM];
    memcpy(osh, d.shape, sizeof(int64_t) * (size_t)onD);
    const int64_t nout = tsr_shape_size(onD, osh);
    if (nout < 0) return TSR_ENOMEM;
    char *p;
    if (onD > 0) {
        p = (char *)fn_result_array(&res[0], dt, onD, osh);
        if (!p) return TSR_ENOMEM;
    } else p = NULL;
    const int64_t lb = len * 8 + 64;
    double *buf = (double *)tsr_alloc(lb);
    if (!buf) { if (p) fn_results_free(&res[0], 1); return TSR_ENOMEM; }
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t o = 0; o < nout; o++) {
        int64_t base = 0;
        for (int dd = 0; dd < onD; dd++) base += ix[dd] * d.strides[dd];
        const char *q = elem0(&d) + base;
        if (dt == TSR_F64 || dt == TSR_F32) {
            for (int64_t i = 0; i < len; i++) buf[i] = rd_f64(q + i * d.strides[onD], src);
            /* numpy.add.reduce: pairwise over the diagonal (float32 input is summed in double here and rounded) */
            const double s = len > 0 ? tsr_psum(buf, len) : 0.0;
            if (p) { if (dt == TSR_F64) ((double *)p)[o] = s; else ((float *)p)[o] = (float)s; }
            else fn_result_num(&res[0], s);
        } else {
            uint64_t acc = 0;
            for (int64_t i = 0; i < len; i++) acc += (uint64_t)rd_i64(q + i * d.strides[onD], src);
            if (dt == TSR_BOOL) acc = acc != 0;
            if (p) {
                switch (dt) {
                case TSR_I64: ((int64_t *)p)[o] = (int64_t)acc; break;
                case TSR_I32: ((int32_t *)p)[o] = (int32_t)acc; break;
                default: ((uint8_t *)p)[o] = (uint8_t)acc; break;
                }
            } else if (dt == TSR_BOOL) { memset(&res[0], 0, sizeof res[0]); res[0].kind = 4; res[0].num = acc != 0; }
            else fn_result_int(&res[0], dt == TSR_I32 ? (int64_t)(int32_t)acc : dt == TSR_U8 ? (int64_t)(uint8_t)acc : (int64_t)acc);
        }
        for (int dd = onD - 1; dd >= 0; dd--) { if (++ix[dd] < osh[dd]) break; ix[dd] = 0; }
    }
    tsr_free(buf, lb);
    return TSR_OK;
}

/* numpy.tri(N, M, k) is greater_equal.outer(arange(N), arange(-k, M - k)): tri[i, j] = i >= v(j). tril uses
   tri(k), triu and triu_indices ~tri(k - 1). For an exact int k, v(j) = -k + j in int64, wrapping around past
   INT64_MAX as NumPy's arange does, and there are M columns. For a float k, arange gives ceil((M - k) + k)
   columns (rounded in double: none for |k| = 1e20) and v(j) = int(-k) + j * (int(-k + 1) - int(-k)). */
typedef struct { int64_t v0, d, cols; } trik;

static int tri_k(const tsr_arg *a, int64_t M, int minus1, trik *t)
{
    memset(t, 0, sizeof *t);
    int64_t k = 0;
    double kf = 0;
    int isint = 1, rc;
    if (!is_none(a)) {
        if (a->kind == 1 && !(a->flags & 1)) { isint = 0; kf = a->num; }
        else if (a->kind == 3 && a->arr.ndim == 0 && (a->arr.dtype == TSR_F64 || a->arr.dtype == TSR_F32)) { isint = 0; kf = rd_f64(elem0(&a->arr), a->arr.dtype); }
        else if ((rc = int_param(a, "k", &k)) < 0) return rc;
    }
    if (isint) {
        if (M <= 0) return TSR_OK;                        /* arange(-k, M - k) is empty: nothing is converted */
        if (minus1) { if (k == INT64_MIN) goto overflow; k -= 1; }
        if (k == INT64_MIN) goto overflow;               /* -k is not an int64 */
        if (k == INT64_MIN + 1 && M >= 2) goto overflow; /* arange's second value -k + 1 is not either */
        t->v0 = -k;
        t->d = 1;
        t->cols = M;
        return TSR_OK;
    }
    if (minus1) kf -= 1.0;
    const double start = -kf, len = ceil(((double)M - kf) - start);
    if (!isfinite(len)) { fn_set_error("arange: cannot compute length"); return TSR_EARG; }
    if (len > 9.2e18) { fn_set_error("Maximum allowed size exceeded"); return TSR_EARG; }
    t->cols = len > 0 ? (int64_t)len : 0;
    if (t->cols == 0) return TSR_OK;
    /* NumPy casts start and start + 1 to int64; beyond its range the cast is undefined, Tessero raises */
    if (!(fabs(start) < 9223372036854775808.0) || !(fabs(start + 1.0) < 9223372036854775808.0)) goto overflow;
    t->v0 = (int64_t)start;
    t->d = (int64_t)(start + 1.0) - t->v0;               /* 1, or 0 for -1 < start < 0 and |k| >= 2^53 */
    return TSR_OK;
overflow:
    fn_set_error("Python int too large to convert to C long");
    return TSR_EARG;
}

static inline int tri_on(const trik *t, int64_t i, int64_t j)
{
    return i >= (int64_t)((uint64_t)t->v0 + (uint64_t)j * (uint64_t)t->d);
}

/* row i of tri: columns [0, *e) and [*w, cols) are set, [*e, *w) are not */
static void tri_row(const trik *t, int64_t i, int64_t *e, int64_t *w)
{
    if (t->d == 0) { *w = t->cols; *e = i >= t->v0 ? t->cols : 0; return; }
    int64_t jw = t->cols;                                 /* v(j) = v0 + j wraps for j > INT64_MAX - v0 */
    if (t->v0 > 0 && t->cols > 0 && (uint64_t)(INT64_MAX - t->v0) < (uint64_t)t->cols - 1) jw = INT64_MAX - t->v0 + 1;
    *w = jw;
    if (t->v0 > i) { *e = 0; return; }
    const uint64_t diff = (uint64_t)i - (uint64_t)t->v0;
    *e = diff >= (uint64_t)jw ? jw : (int64_t)diff + 1;
}

/* tri(N, M=None, k=0, dtype=float) */
static int r_tri(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    int64_t N, M;
    int rc = int_param(argk(args, nargs, 0), "N", &N);
    if (rc < 0) return rc;
    M = N;
    if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "M", &M)) < 0) return rc;
    int dt;
    if ((rc = arg_dtype(argk(args, nargs, 3), &dt)) < 0) return rc;
    if (dt < 0) dt = TSR_F64;
    if (N < 0) N = 0;                                     /* arange of a negative length is empty */
    if (M < 0) M = 0;
    trik t;
    if ((rc = tri_k(argk(args, nargs, 2), M, 0, &t)) < 0) return rc;
    const int64_t sh[2] = {N, t.cols};
    if (tsr_shape_size(2, sh) < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    char *p = (char *)new_result(&res[0], dt, 2, sh);
    if (!p) return TSR_ENOMEM;
    const int64_t isz = tsr_itemsize(dt);
    for (int64_t i = 0; i < N; i++) {
        int64_t e, w;
        tri_row(&t, i, &e, &w);
        for (int64_t j = 0; j < t.cols; j++) {
            if (j >= e && j < w) continue;
            char *q = p + (i * t.cols + j) * isz;
            switch (dt) {
            case TSR_F64: *(double *)q = 1.0; break;
            case TSR_F32: *(float *)q = 1.0f; break;
            case TSR_I64: *(int64_t *)q = 1; break;
            case TSR_I32: *(int32_t *)q = 1; break;
            case TSR_C128: *(double *)q = 1.0; break;
            default: *(uint8_t *)q = 1; break;
            }
        }
    }
    return TSR_OK;
}

/* tril / triu (m, k=0): on the last two axes; a 1-D m broadcasts against its (n, n) mask */
static int r_trilu(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int upper = *(const int *)ctx;
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "m");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    if (v.ndim == 0) { fn_set_error("tri() missing 1 required positional argument: 'N'"); return TSR_EARG; }
    int64_t R, C;
    if (v.ndim == 1) { R = v.shape[0]; C = v.shape[0]; }
    else { R = v.shape[v.ndim - 2]; C = v.shape[v.ndim - 1]; }
    trik t;
    if ((rc = tri_k(argk(args, nargs, 1), C, upper, &t)) < 0) return rc;
    /* result = broadcast of the mask (R, cols) and m; cols differs from C only for a float k far from 0 */
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    const int64_t msh[2] = {R, t.cols};
    if ((rc = bshape(2, msh, v.ndim, v.shape, &nd, sh)) < 0) return rc;
    tsr_array b = v;
    if ((rc = v_broadcast(&b, nd, sh)) < 0) return rc;
    if ((rc = emit(&res[0], &b, -1)) < 0) return rc;
    if (res[0].kind != 3) return TSR_OK;
    char *p = (char *)res[0].arr.data;
    const int64_t R2 = sh[nd - 2], C2 = sh[nd - 1];
    const int64_t isz = tsr_itemsize(v.dtype), total = tsr_shape_size(nd, sh), plane = R2 * C2;
    if (plane == 0) return TSR_OK;
    for (int64_t base = 0; base < total; base += plane)
        for (int64_t i = 0; i < R2; i++)
            for (int64_t j = 0; j < C2; j++) {
                const int on = tri_on(&t, i, t.cols == 1 ? 0 : j);
                /* tril keeps tri(k); triu zeroes tri(k - 1) */
                if (upper ? on : !on) memset(p + (base + i * C2 + j) * isz, 0, (size_t)isz);
            }
    return TSR_OK;
}
static const int TRIL = 0, TRIU = 1;

/* tril_indices / triu_indices (n, k=0, m=None); *_from (arr, k=0): nonzero(tri(n, m, k)), nonzero(~tri(n, m, k - 1)) */
static int r_trilu_indices(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;                  /* 0 tril, 1 triu, 2 tril_from, 3 triu_from */
    int64_t n, m;
    int rc;
    const tsr_arg *ka = argk(args, nargs, 1);
    if (kind >= 2) {
        harr h;
        if ((rc = get_arr(argk(args, nargs, 0), &h, "arr")) < 0) return rc;
        if (h.a.ndim != 2) { fn_set_error("input array must be 2-d"); return TSR_EARG; }
        n = h.a.shape[0];
        m = h.a.shape[1];
    } else {
        if ((rc = int_param(argk(args, nargs, 0), "n", &n)) < 0) return rc;
        m = n;
        if (!is_none(argk(args, nargs, 2)) && (rc = int_param(argk(args, nargs, 2), "m", &m)) < 0) return rc;
        if (n < 0) n = 0;                                 /* tri(n, m) of a negative size is empty */
        if (m < 0) m = 0;
    }
    const int upper = kind == 1 || kind == 3;
    trik t;
    if ((rc = tri_k(ka, m, upper, &t)) < 0) return rc;
    const int64_t sh2[2] = {n, t.cols};
    if (tsr_shape_size(2, sh2) < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    int64_t cnt = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t e, w;
        tri_row(&t, i, &e, &w);
        cnt += upper ? w - e : e + (t.cols - w);
    }
    if (check_total(2.0 * (double)cnt, 8) < 0) return TSR_ENOMEM;
    tsr_result *items = fn_result_seq(&res[0], 2);
    if (!items) return TSR_ENOMEM;
    int64_t *r = (int64_t *)fn_result_array(&items[0], TSR_I64, 1, &cnt);
    int64_t *c = r ? (int64_t *)fn_result_array(&items[1], TSR_I64, 1, &cnt) : NULL;
    if (!r || !c) return TSR_ENOMEM;
    int64_t q = 0;
    for (int64_t i = 0; i < n; i++) {                    /* only the selected columns: O(result) */
        int64_t e, w;
        tri_row(&t, i, &e, &w);
        if (upper) { for (int64_t j = e; j < w; j++) { r[q] = i; c[q] = j; q++; } }
        else {
            for (int64_t j = 0; j < e; j++) { r[q] = i; c[q] = j; q++; }
            for (int64_t j = w; j < t.cols; j++) { r[q] = i; c[q] = j; q++; }
        }
    }
    return TSR_OK;
}
static const int TRIL_I = 0, TRIU_I = 1, TRIL_IF = 2, TRIU_IF = 3;

/* diag_indices(n, ndim=2); diag_indices_from(arr) */
static int r_diag_indices(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int from = ctx ? *(const int *)ctx : 0;
    int64_t n, nd = 2;
    int rc;
    if (from) {
        harr h;
        if ((rc = get_arr(argk(args, nargs, 0), &h, "arr")) < 0) return rc;
        if (h.a.ndim < 2) { fn_set_error("input array must be at least 2-d"); return TSR_EARG; }
        for (int d = 1; d < h.a.ndim; d++) if (h.a.shape[d] != h.a.shape[0]) { fn_set_error("All dimensions of input must be of equal length"); return TSR_EARG; }
        n = h.a.shape[0];
        nd = h.a.ndim;
    } else {
        if ((rc = int_param(argk(args, nargs, 0), "n", &n)) < 0) return rc;
        if (!is_none(argk(args, nargs, 1)) && (rc = int_param(argk(args, nargs, 1), "ndim", &nd)) < 0) return rc;
        if (n < 0) { fn_set_error("negative dimensions are not allowed"); return TSR_EARG; }
        if (nd < 0) nd = 0;                                /* (idx,) * negative -> () */
        if (nd > 1 << 20) { fn_set_error("ndim is too large"); return TSR_EARG; }
    }
    if (check_total((double)n * (double)nd, 8) < 0) return TSR_ENOMEM;
    tsr_result *items = fn_result_seq(&res[0], nd);
    if (!items) return TSR_ENOMEM;
    for (int64_t i = 0; i < nd; i++) {
        int64_t *p = (int64_t *)fn_result_array(&items[i], TSR_I64, 1, &n);
        if (!p) return TSR_ENOMEM;
        for (int64_t j = 0; j < n; j++) p[j] = j;
    }
    return TSR_OK;
}
static const int FROM = 1;

/* identity(n, dtype=None) */
static int r_identity(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    int64_t n;
    int rc = int_param(argk(args, nargs, 0), "n", &n);
    if (rc < 0) return rc;
    int dt;
    if ((rc = arg_dtype(argk(args, nargs, 1), &dt)) < 0) return rc;
    if (dt < 0) dt = TSR_F64;
    if (n < 0) { fn_set_error("negative dimensions are not allowed"); return TSR_EARG; }
    const int64_t sh[2] = {n, n};
    if (tsr_shape_size(2, sh) < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    char *p = (char *)new_result(&res[0], dt, 2, sh);
    if (!p) return TSR_ENOMEM;
    const int64_t isz = tsr_itemsize(dt);
    for (int64_t i = 0; i < n; i++) {
        char *q = p + (i * n + i) * isz;
        switch (dt) {
        case TSR_F64: case TSR_C128: *(double *)q = 1.0; break;
        case TSR_F32: *(float *)q = 1.0f; break;
        case TSR_I64: *(int64_t *)q = 1; break;
        case TSR_I32: *(int32_t *)q = 1; break;
        default: *(uint8_t *)q = 1; break;
        }
    }
    return TSR_OK;
}

/* ================================================================ *_like */

/* cast one Python scalar / 0-d value into an element of dtype dt ("unsafe", as copyto) */
static int scalar_into(const tsr_array *s, int dt, void *dst)
{
    const int64_t one = 1, z = 0;
    return tsr_copy(s->dtype, dt, 1, &one, elem0(s), &z, dst, &z);
}

/* zeros_like / ones_like / empty_like (a, dtype=None, shape=None); full_like (a, fill_value, dtype=None, shape=None) */
static int r_like(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;                  /* 0 zeros, 1 ones, 2 empty, 3 full */
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    const int off = kind == 3 ? 1 : 0;
    int dt;
    if ((rc = arg_dtype(argk(args, nargs, 1 + off), &dt)) < 0) return rc;
    if (dt < 0) dt = h.a.dtype;
    int32_t nd = h.a.ndim;
    int64_t sh[TSR_MAXDIM];
    memcpy(sh, h.a.shape, sizeof(int64_t) * (size_t)nd);
    if (!is_none(argk(args, nargs, 2 + off)) && (rc = shape_arg(argk(args, nargs, 2 + off), "shape", &nd, sh)) < 0) return rc;
    unsigned char val[16] = {0};
    if (kind == 1) {
        const double one = 1.0;
        tsr_array s = {0};
        s.data = (void *)&one; s.dtype = TSR_F64;
        scalar_into(&s, dt, val);
    } else if (kind == 3) {
        const tsr_arg *fv = argk(args, nargs, 1);
        harr hf;
        if ((rc = get_arr(fv, &hf, "fill_value")) < 0) return rc;
        tsr_array b = hf.a;
        if (b.ndim > 0) {                                /* an array fill value broadcasts into the result */
            if (b.ndim > nd) { fn_set_error("could not broadcast input array into shape"); return TSR_EARG; }
            if (v_broadcast(&b, nd, sh) < 0) { fn_set_error("could not broadcast input array from its shape into the result shape"); return TSR_EARG; }
            return emit(&res[0], &b, dt);
        }
        if (fv->kind == 1 && (fv->flags & 1) && is_intdt(dt)) {
            /* a Python int that does not fit: numpy raises OverflowError */
            const int64_t x = fv->ival;
            if ((dt == TSR_I32 && (x < INT32_MIN || x > INT32_MAX)) || (dt == TSR_U8 && (x < 0 || x > 255))) {
                fn_set_error("Python integer %lld out of bounds for %s", (long long)x, dtname(dt));
                return TSR_EARG;
            }
        }
        scalar_into(&b, dt, val);
    }
    const int64_t n = tsr_shape_size(nd, sh);
    if (n < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    if (nd == 0 && dt != TSR_C128) return result_scalar(&res[0], (const char *)val, dt);
    void *p = fn_result_array(&res[0], dt, nd, sh);
    if (!p) return TSR_ENOMEM;
    if (n > 0) tsr_fill(dt, n, p, val);
    return TSR_OK;
}
static const int LIKE_ZEROS = 0, LIKE_ONES = 1, LIKE_EMPTY = 2, LIKE_FULL = 3;

/* ================================================================ gathering */

/* take(a, indices, axis=None, mode='raise') */
static int r_take(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    const tsr_arg *ma = argk(args, nargs, 3);
    int mode = 0;
    if (ma->kind == 2 && ma->str) {
        if (strcmp(ma->str, "raise") == 0) mode = 0;
        else if (strcmp(ma->str, "wrap") == 0) mode = 1;
        else if (strcmp(ma->str, "clip") == 0) mode = 2;
        else { fn_set_error("clipmode must be one of 'clip', 'raise', or 'wrap' (got '%s')", ma->str); return TSR_EARG; }
    }
    int64_t *iv, ib;
    int32_t ind;
    int64_t ish[TSR_MAXDIM];
    if ((rc = ints_of(argk(args, nargs, 1), "indices", &iv, &ib, &ind, ish)) < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    const tsr_arg *aa = argk(args, nargs, 2);
    int ax = 0;
    if (is_none(aa)) { if ((rc = v_ravel(&h.a, &v, &o)) < 0) { tsr_free(iv, ib); return rc; } }
    else {
        int64_t a64;
        if ((rc = int_param(aa, "axis", &a64)) < 0 || (rc = norm_axis(a64, v.ndim, &ax)) < 0) { tsr_free(iv, ib); return rc; }
    }
    const int64_t N = v.shape[ax];
    const int ond = v.ndim - 1 + ind;
    if (ond > TSR_MAXDIM) { tsr_free(iv, ib); own_free(&o); fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    int64_t osh[TSR_MAXDIM];
    int k = 0;
    for (int d = 0; d < ax; d++) osh[k++] = v.shape[d];
    for (int d = 0; d < ind; d++) osh[k++] = ish[d];
    for (int d = ax + 1; d < v.ndim; d++) osh[k++] = v.shape[d];
    const int64_t ni = tsr_shape_size(ind, ish);
    if (N == 0 && ni > 0 && mode != 0) { tsr_free(iv, ib); own_free(&o); fn_set_error("cannot do a non-empty take from an empty axes."); return TSR_EARG; }
    for (int64_t q = 0; q < ni; q++) {
        int64_t j = iv[q];
        if (mode == 0) {
            if (j < -N || j >= N) {
                fn_set_error("index %lld is out of bounds for axis %d with size %lld", (long long)j, ax, (long long)N);
                tsr_free(iv, ib); own_free(&o);
                return TSR_EARG;
            }
            if (j < 0) j += N;
        } else if (mode == 1) j = pymod(j, N);
        else j = j < 0 ? 0 : (j >= N ? N - 1 : j);
        iv[q] = j;
    }
    if (tsr_shape_size(ond, osh) < 0) { tsr_free(iv, ib); own_free(&o); fn_set_error("array is too big"); return TSR_EARG; }
    /* out[pre..., idx..., post...] = a[pre..., iv[flat(idx)], post...] */
    const int64_t pre = tsr_shape_size(ax, v.shape), post = tsr_shape_size(v.ndim - ax - 1, v.shape + ax + 1);
    const int64_t isz = tsr_itemsize(v.dtype);
    char *p = (char *)fn_result_array(&res[0], v.dtype, ond, osh);
    if (!p) { tsr_free(iv, ib); own_free(&o); return TSR_ENOMEM; }
    int64_t ipre[TSR_MAXDIM] = {0};
    for (int64_t a = 0; a < pre; a++) {
        int64_t offp = 0;
        for (int d = 0; d < ax; d++) offp += ipre[d] * v.strides[d];
        for (int64_t q = 0; q < ni; q++) {
            int64_t ipost[TSR_MAXDIM] = {0};
            for (int64_t b = 0; b < post; b++) {
                int64_t offq = 0;
                for (int d = ax + 1; d < v.ndim; d++) offq += ipost[d - ax - 1] * v.strides[d];
                memcpy(p, elem0(&v) + offp + iv[q] * v.strides[ax] + offq, (size_t)isz);
                p += isz;
                for (int d = v.ndim - ax - 2; d >= 0; d--) { if (++ipost[d] < v.shape[ax + 1 + d]) break; ipost[d] = 0; }
            }
        }
        for (int d = ax - 1; d >= 0; d--) { if (++ipre[d] < v.shape[d]) break; ipre[d] = 0; }
    }
    tsr_free(iv, ib);
    own_free(&o);
    if (ond == 0) {
        tsr_result sres;
        if (result_scalar(&sres, (const char *)res[0].arr.data, res[0].arr.dtype) == TSR_OK) { fn_results_free(&res[0], 1); res[0] = sres; }
    }
    return TSR_OK;
}

/* take_along_axis(arr, indices, axis=-1) */
static int r_take_along_axis(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr h, hi;
    int rc = get_arr(argk(args, nargs, 0), &h, "arr");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &hi, "indices");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    const tsr_arg *aa = argk(args, nargs, 2);
    int ax;
    if (is_none(aa)) {
        if (hi.a.ndim != 1) { fn_set_error("when axis=None, `indices` must have a single dimension."); return TSR_EARG; }
        if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc;
        ax = 0;
    } else {
        int64_t a64 = -1;
        if ((rc = int_param(aa, "axis", &a64)) < 0 || (rc = norm_axis(a64, v.ndim, &ax)) < 0) return rc;
    }
    if (!is_intdt(hi.a.dtype)) { own_free(&o); fn_set_error("`indices` must be an integer array"); return TSR_EARG; }
    if (hi.a.ndim != v.ndim) {
        own_free(&o);
        fn_set_error("`indices` and `arr` must have the same number of dimensions");
        return TSR_EARG;
    }
    /* broadcast every axis but `ax` */
    const int nd = v.ndim;
    int64_t osh[TSR_MAXDIM];
    for (int d = 0; d < nd; d++) {
        if (d == ax) { osh[d] = hi.a.shape[d]; continue; }
        const int64_t x = v.shape[d], y = hi.a.shape[d];
        if (x != y && x != 1 && y != 1) {
            own_free(&o);
            fn_set_error("shape mismatch: indexing arrays could not be broadcast together");
            return TSR_EARG;
        }
        osh[d] = x == 1 ? y : x;
    }
    const int64_t N = v.shape[ax], n = tsr_shape_size(nd, osh), isz = tsr_itemsize(v.dtype);
    if (n < 0) { own_free(&o); return TSR_ENOMEM; }
    char *p = (char *)fn_result_array(&res[0], v.dtype, nd, osh);
    if (!p) { own_free(&o); return TSR_ENOMEM; }
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t oi = 0, os = 0;
        for (int d = 0; d < nd; d++) {
            oi += (hi.a.shape[d] == 1 ? 0 : ix[d]) * hi.a.strides[d];
            if (d != ax) os += (v.shape[d] == 1 ? 0 : ix[d]) * v.strides[d];
        }
        int64_t j = rd_i64(elem0(&hi.a) + oi, hi.a.dtype);
        if (j < -N || j >= N) {
            fn_results_free(&res[0], 1);
            own_free(&o);
            fn_set_error("index %lld is out of bounds for axis %d with size %lld", (long long)j, ax, (long long)N);
            return TSR_EARG;
        }
        if (j < 0) j += N;
        memcpy(p + q * isz, elem0(&v) + os + j * v.strides[ax], (size_t)isz);
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < osh[d]) break; ix[d] = 0; }
    }
    own_free(&o);
    return TSR_OK;
}

/* nonzero(a) -> tuple; argwhere(a); flatnonzero(a) */
static int r_nonzero(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;                  /* 0 nonzero, 1 argwhere, 2 flatnonzero */
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    tsr_array v = h.a;
    owned o = {0};
    int zero_d = 0;
    if (kind == 2) { if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc; }
    else if (v.ndim == 0) {
        if (kind == 0) { fn_set_error("Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead."); return TSR_EARG; }
        zero_d = 1;                                      /* argwhere(atleast_1d(a))[:, :0] */
        v_insert_axis(&v, 0);
    }
    const int nd = v.ndim;
    const int64_t total = tsr_shape_size(nd, v.shape);
    int64_t cnt = 0;
    {
        int64_t ix[TSR_MAXDIM] = {0};
        for (int64_t q = 0; q < total; q++) {
            int64_t off = 0;
            for (int d = 0; d < nd; d++) off += ix[d] * v.strides[d];
            cnt += truthy(elem0(&v) + off, v.dtype);
            for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < v.shape[d]) break; ix[d] = 0; }
        }
    }
    int64_t *cols[TSR_MAXDIM];
    int64_t *mat = NULL;
    if (check_total((double)cnt * (double)(nd > 0 ? nd : 1), 8) < 0) { own_free(&o); return TSR_ENOMEM; }
    if (kind == 0) {
        tsr_result *items = fn_result_seq(&res[0], nd);
        if (!items) { own_free(&o); return TSR_ENOMEM; }
        for (int d = 0; d < nd; d++) {
            cols[d] = (int64_t *)fn_result_array(&items[d], TSR_I64, 1, &cnt);
            if (!cols[d]) { own_free(&o); return TSR_ENOMEM; }
        }
    } else if (kind == 1) {
        const int64_t sh[2] = {cnt, zero_d ? 0 : nd};
        mat = (int64_t *)fn_result_array(&res[0], TSR_I64, 2, sh);
        if (!mat) { own_free(&o); return TSR_ENOMEM; }
    } else {
        mat = (int64_t *)fn_result_array(&res[0], TSR_I64, 1, &cnt);
        if (!mat) { own_free(&o); return TSR_ENOMEM; }
    }
    int64_t ix[TSR_MAXDIM] = {0}, k = 0;
    for (int64_t q = 0; q < total; q++) {
        int64_t off = 0;
        for (int d = 0; d < nd; d++) off += ix[d] * v.strides[d];
        if (truthy(elem0(&v) + off, v.dtype)) {
            if (kind == 0) for (int d = 0; d < nd; d++) cols[d][k] = ix[d];
            else if (kind == 1) { if (!zero_d) for (int d = 0; d < nd; d++) mat[k * nd + d] = ix[d]; }
            else mat[k] = q;
            k++;
        }
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < v.shape[d]) break; ix[d] = 0; }
    }
    own_free(&o);
    return TSR_OK;
}
static const int NZ = 0, ARGWHERE = 1, FLATNZ = 2;

/* compress(condition, a, axis=None); extract(condition, arr) */
static int r_compress(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int extract = ctx ? *(const int *)ctx : 0;
    harr hc, h;
    int rc = get_arr(argk(args, nargs, 0), &hc, "condition");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &h, "a");
    if (rc < 0) return rc;
    tsr_array c = hc.a, v = h.a;
    owned oc = {0}, o = {0};
    int ax = 0;
    if (extract || is_none(argk(args, nargs, 2))) {
        if ((rc = v_ravel(&h.a, &v, &o)) < 0) return rc;
    } else {
        int64_t a64;
        if ((rc = int_param(argk(args, nargs, 2), "axis", &a64)) < 0 || (rc = norm_axis(a64, v.ndim, &ax)) < 0) return rc;
    }
    if (extract) { if ((rc = v_ravel(&hc.a, &c, &oc)) < 0) { own_free(&o); return rc; } }
    else if (c.ndim != 1) { own_free(&o); fn_set_error("condition must be a 1-d array"); return TSR_EARG; }
    const int64_t N = v.shape[ax], nc = c.shape[0];
    int64_t cnt = 0;
    for (int64_t i = 0; i < nc; i++) if (truthy(elem0(&c) + i * c.strides[0], c.dtype)) {
        if (i >= N) {
            own_free(&o); own_free(&oc);
            if (extract) fn_set_error("index %lld is out of bounds for axis 0 with size %lld", (long long)i, (long long)N);
            else fn_set_error("index %lld is out of bounds for axis %d with size %lld", (long long)i, ax, (long long)N);
            return TSR_EARG;
        }
        cnt++;
    }
    int64_t osh[TSR_MAXDIM];
    memcpy(osh, v.shape, sizeof(int64_t) * (size_t)v.ndim);
    osh[ax] = cnt;
    tables T;
    if ((rc = tables_new(&T, v.ndim, osh)) < 0) { own_free(&o); own_free(&oc); return rc; }
    for (int d = 0; d < v.ndim; d++) {
        if (d != ax) { for (int64_t j = 0; j < osh[d]; j++) T.t[d][j] = j * v.strides[d]; continue; }
        int64_t k = 0;
        for (int64_t i = 0; i < nc && i < N; i++) if (truthy(elem0(&c) + i * c.strides[0], c.dtype)) T.t[d][k++] = i * v.strides[d];
    }
    rc = emit_tables(&res[0], &v, v.ndim, osh, &T);
    tables_free(&T);
    own_free(&o);
    own_free(&oc);
    return rc;
}
static const int EXTRACT = 1;

/* ================================================================ index conversion */

/* unravel_index(indices, shape, order='C') */
static int r_unravel_index(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr hi;
    int rc = get_arr(argk(args, nargs, 0), &hi, "indices");
    if (rc < 0) return rc;
    int32_t nd;
    int64_t sh[TSR_MAXDIM];
    {
        int m;
        int64_t v[TSR_MAXDIM + 1];
        if ((rc = int_list(argk(args, nargs, 1), "shape", v, TSR_MAXDIM + 1, &m, NULL)) < 0) return rc;
        for (int j = 0; j < m; j++) if (v[j] < 0) { fn_set_error("dimensions must be non-negative"); return TSR_EARG; }
    }
    if ((rc = shape_arg_ex(argk(args, nargs, 1), "shape", &nd, sh,
                           "dimensions are too large; arrays and shapes with a total size greater than 'intp' are not supported.")) < 0) return rc;
    const tsr_arg *oa = argk(args, nargs, 2);
    const int fortran = oa->kind == 2 && oa->str && strcmp(oa->str, "F") == 0;
    if (oa->kind == 2 && oa->str && strcmp(oa->str, "C") != 0 && !fortran) { fn_set_error("only 'C' or 'F' order is permitted"); return TSR_EARG; }
    if (!is_intdt(hi.a.dtype) && !(hi.a.dtype == TSR_BOOL && 0)) {
        if (tsr_shape_size(hi.a.ndim, hi.a.shape) > 0 || hi.a.dtype != TSR_F64) {
            fn_set_error("only int indices permitted");
            return TSR_EARG;
        }
    }
    int64_t size = 1;                                     /* fits int64: checked by shape_arg_ex */
    for (int d = 0; d < nd; d++) size *= sh[d];
    const int64_t n = tsr_shape_size(hi.a.ndim, hi.a.shape);
    if (check_total((double)n * (double)nd, 8) < 0) return TSR_ENOMEM;
    tsr_result *items = fn_result_seq(&res[0], nd);
    if (!items) return TSR_ENOMEM;
    int64_t *outs[TSR_MAXDIM];
    const int scalar = hi.a.ndim == 0;
    for (int d = 0; d < nd; d++) {
        if (scalar) { outs[d] = NULL; continue; }
        outs[d] = (int64_t *)fn_result_array(&items[d], TSR_I64, hi.a.ndim, hi.a.shape);
        if (!outs[d]) return TSR_ENOMEM;
    }
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < n; q++) {
        int64_t off = 0;
        for (int d = 0; d < hi.a.ndim; d++) off += ix[d] * hi.a.strides[d];
        int64_t j = rd_i64(elem0(&hi.a) + off, hi.a.dtype);
        if (scalar && argk(args, nargs, 0)->kind == 1) j = argk(args, nargs, 0)->ival;
        if (j < 0 || j >= size) {
            fn_set_error("index %lld is out of bounds for array with size %lld", (long long)j, (long long)size);
            return TSR_EARG;
        }
        int64_t r = j;
        if (!fortran) for (int d = nd - 1; d >= 0; d--) { const int64_t c = r % sh[d]; r /= sh[d]; if (scalar) fn_result_int(&items[d], c); else outs[d][q] = c; }
        else for (int d = 0; d < nd; d++) { const int64_t c = r % sh[d]; r /= sh[d]; if (scalar) fn_result_int(&items[d], c); else outs[d][q] = c; }
        for (int d = hi.a.ndim - 1; d >= 0; d--) { if (++ix[d] < hi.a.shape[d]) break; ix[d] = 0; }
    }
    return TSR_OK;
}

/* ravel_multi_index(multi_index, dims, mode='raise', order='C') */
static int r_ravel_multi_index(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    tsr_array *vs;
    int n;
    arrlist L;
    int rc = list_views(argk(args, nargs, 0), "multi_index", 0, &vs, &n, &L);
    if (rc < 0) return rc;
    int32_t nd;
    int64_t dims[TSR_MAXDIM];
    if ((rc = shape_arg_ex(argk(args, nargs, 1), "dims", &nd, dims,
                           "invalid dims: array size defined by dims is larger than the maximum possible size.")) < 0) { list_views_free(vs, n, &L); return rc; }
    if (n != nd) {
        list_views_free(vs, n, &L);
        fn_set_error("parameter multi_index must be a sequence of length %d", nd);
        return TSR_EARG;
    }
    int modes[TSR_MAXDIM];
    const tsr_arg *ma = argk(args, nargs, 2);
    for (int d = 0; d < nd; d++) modes[d] = 0;
    if (ma->kind == 2 || ma->kind == 5) {
        const int cnt = ma->kind == 2 ? 1 : (int)ma->count;
        if (ma->kind == 5 && cnt != nd) { list_views_free(vs, n, &L); fn_set_error("mode must have length %d", nd); return TSR_EARG; }
        for (int d = 0; d < nd; d++) {
            const tsr_arg *m = ma->kind == 2 ? ma : &ma->items[d];
            if (m->kind != 2 || !m->str) { list_views_free(vs, n, &L); fn_set_error("clipmode not understood"); return TSR_EARG; }
            if (strcmp(m->str, "raise") == 0) modes[d] = 0;
            else if (strcmp(m->str, "wrap") == 0) modes[d] = 1;
            else if (strcmp(m->str, "clip") == 0) modes[d] = 2;
            else { list_views_free(vs, n, &L); fn_set_error("clipmode must be one of 'clip', 'raise', or 'wrap' (got '%s')", m->str); return TSR_EARG; }
        }
    }
    const tsr_arg *oa = argk(args, nargs, 3);
    const int fortran = oa->kind == 2 && oa->str && strcmp(oa->str, "F") == 0;
    int32_t bnd = 0;
    int64_t bsh[TSR_MAXDIM];
    for (int i = 0; i < n && rc == 0; i++) {
        if (!is_intdt(vs[i].dtype) && tsr_shape_size(vs[i].ndim, vs[i].shape) > 0) {
            fn_set_error("only int indices permitted");
            rc = TSR_EARG;
            break;
        }
        rc = bshape(bnd, bsh, vs[i].ndim, vs[i].shape, &bnd, bsh);
    }
    for (int i = 0; i < n && rc == 0; i++) rc = v_broadcast(&vs[i], bnd, bsh);
    if (rc < 0) { list_views_free(vs, n, &L); return rc; }
    /* the size must fit intp */
    {
        int64_t s = 1;
        for (int d = 0; d < nd; d++) {
            if (dims[d] != 0 && s > INT64_MAX / dims[d]) { list_views_free(vs, n, &L); fn_set_error("invalid dims: array size defined by dims is larger than the maximum possible size."); return TSR_EARG; }
            s *= dims[d];
        }
    }
    const int64_t total = tsr_shape_size(bnd, bsh);
    int64_t *p = NULL;
    if (bnd > 0) {
        p = (int64_t *)fn_result_array(&res[0], TSR_I64, bnd, bsh);
        if (!p) { list_views_free(vs, n, &L); return TSR_ENOMEM; }
    }
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < total; q++) {
        int64_t flat = 0;
        for (int k = 0; k < nd; k++) {
            const int d = fortran ? nd - 1 - k : k;
            int64_t off = 0;
            for (int e = 0; e < bnd; e++) off += ix[e] * vs[d].strides[e];
            int64_t j = rd_i64(elem0(&vs[d]) + off, vs[d].dtype);
            if (bnd == 0 && L.h && argk(args, nargs, 0)->kind == 5 && argk(args, nargs, 0)->items[d].kind == 1) j = argk(args, nargs, 0)->items[d].ival;
            const int64_t m = dims[d];
            if (modes[d] == 0) {
                if (j < 0 || j >= m) {
                    if (p) fn_results_free(&res[0], 1);
                    list_views_free(vs, n, &L);
                    fn_set_error("invalid entry in coordinates array");
                    return TSR_EARG;
                }
            } else if (modes[d] == 1) { if (m == 0) j = 0; else j = pymod(j, m); }
            else j = j < 0 ? 0 : (j >= m ? m - 1 : j);
            flat = flat * m + j;
        }
        if (p) p[q] = flat;
        else fn_result_int(&res[0], flat);
        for (int e = bnd - 1; e >= 0; e--) { if (++ix[e] < bsh[e]) break; ix[e] = 0; }
    }
    list_views_free(vs, n, &L);
    return TSR_OK;
}

/* indices(dimensions, dtype=int, sparse=False) */
static int r_indices(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    int32_t N;
    int64_t dims[TSR_MAXDIM];
    int rc = shape_arg(argk(args, nargs, 0), "dimensions", &N, dims);
    if (rc < 0) return rc;
    int dt;
    if ((rc = arg_dtype(argk(args, nargs, 1), &dt)) < 0) return rc;
    if (dt < 0) dt = TSR_I64;
    const tsr_arg *sp = argk(args, nargs, 2);
    const int sparse = (sp->kind == 4 || sp->kind == 1) && sp->num != 0;
    if (sparse) {
        tsr_result *items = fn_result_seq(&res[0], N);
        if (!items) return TSR_ENOMEM;
        for (int i = 0; i < N; i++) {
            int64_t sh[TSR_MAXDIM];
            for (int d = 0; d < N; d++) sh[d] = d == i ? dims[i] : 1;
            char *p = (char *)fn_result_array(&items[i], dt, N, sh);
            if (!p) return TSR_ENOMEM;
            for (int64_t j = 0; j < dims[i]; j++) {
                const double x = (double)j;
                tsr_array s = {0};
                s.data = (void *)&x; s.dtype = TSR_F64;
                if (dt == TSR_I64) ((int64_t *)p)[j] = j; else scalar_into(&s, dt, p + j * tsr_itemsize(dt));
            }
        }
        return TSR_OK;
    }
    if (N + 1 > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    int64_t sh[TSR_MAXDIM];
    sh[0] = N;
    memcpy(sh + 1, dims, sizeof(int64_t) * (size_t)N);
    if (tsr_shape_size(N + 1, sh) < 0) { fn_set_error("array is too big"); return TSR_EARG; }
    char *p = (char *)fn_result_array(&res[0], dt, N + 1, sh);
    if (!p) return TSR_ENOMEM;
    const int64_t plane = tsr_shape_size(N, dims), isz = tsr_itemsize(dt);
    for (int i = 0; i < N; i++) {
        int64_t ix[TSR_MAXDIM] = {0};
        for (int64_t q = 0; q < plane; q++) {
            const double x = (double)ix[i];
            tsr_array s = {0};
            s.data = (void *)&x; s.dtype = TSR_F64;
            char *e = p + (i * plane + q) * isz;
            if (dt == TSR_I64) *(int64_t *)e = ix[i]; else scalar_into(&s, dt, e);
            for (int d = N - 1; d >= 0; d--) { if (++ix[d] < dims[d]) break; ix[d] = 0; }
        }
    }
    return TSR_OK;
}

/* ix_(*args) */
static int r_ix(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_arg *a = argk(args, nargs, 0);
    const int nd = a->kind == 5 ? (int)a->count : 0;
    if (nd > TSR_MAXDIM) { fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    tsr_result *items = fn_result_seq(&res[0], nd);
    if (!items) return TSR_ENOMEM;
    for (int k = 0; k < nd; k++) {
        harr h;
        int rc = get_arr(&a->items[k], &h, "args");
        if (rc < 0) return rc;
        tsr_array v = h.a;
        if (v.ndim != 1) { fn_set_error("Cross index must be 1 dimensional"); return TSR_EARG; }
        if (v.shape[0] == 0 && v.dtype == TSR_F64) v.dtype = TSR_I64;     /* empty lists become intp */
        int64_t sh[TSR_MAXDIM];
        if (v.dtype == TSR_BOOL) {
            int64_t cnt = 0;
            for (int64_t i = 0; i < v.shape[0]; i++) cnt += *(elem0(&v) + i * v.strides[0]) != 0;
            for (int d = 0; d < nd; d++) sh[d] = d == k ? cnt : 1;
            int64_t *p = (int64_t *)fn_result_array(&items[k], TSR_I64, nd, sh);
            if (!p) return TSR_ENOMEM;
            int64_t q = 0;
            for (int64_t i = 0; i < v.shape[0]; i++) if (*(elem0(&v) + i * v.strides[0])) p[q++] = i;
            continue;
        }
        for (int d = 0; d < nd; d++) sh[d] = d == k ? v.shape[0] : 1;
        tsr_array w = v;
        w.ndim = nd;
        for (int d = 0; d < nd; d++) { w.shape[d] = sh[d]; w.strides[d] = d == k ? v.strides[0] : 0; }
        if (v.shape[0] == 0) {
            if (!fn_result_array(&items[k], v.dtype, nd, sh)) return TSR_ENOMEM;
            continue;
        }
        if ((rc = emit(&items[k], &w, -1)) < 0) return rc;
    }
    return TSR_OK;
}

/* meshgrid(*xi, copy=True, sparse=False, indexing='xy') */
static int r_meshgrid(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_arg *ia = argk(args, nargs, 3);
    const char *indexing = ia->kind == 2 && ia->str ? ia->str : "xy";
    if (strcmp(indexing, "xy") != 0 && strcmp(indexing, "ij") != 0) { fn_set_error("Valid values for `indexing` are 'xy' and 'ij'."); return TSR_EARG; }
    const tsr_arg *sp = argk(args, nargs, 2);
    const int sparse = (sp->kind == 4 || sp->kind == 1) && sp->num != 0;
    tsr_array *vs;
    int n;
    arrlist L;
    int rc = list_views(argk(args, nargs, 0), "xi", 0, &vs, &n, &L);
    if (rc < 0) return rc;
    if (n > TSR_MAXDIM) { list_views_free(vs, n, &L); fn_set_error("maximum supported dimension for an ndarray is 32"); return TSR_EDIM; }
    owned own[TSR_MAXDIM];
    memset(own, 0, sizeof own);
    for (int i = 0; i < n && rc == 0; i++) {
        tsr_array flat;
        if ((rc = v_ravel(&vs[i], &flat, &own[i])) < 0) break;
        int pos = i;
        if (strcmp(indexing, "xy") == 0 && n > 1 && i < 2) pos = 1 - i;
        tsr_array w = flat;
        w.ndim = n;
        for (int d = 0; d < n; d++) { w.shape[d] = d == pos ? flat.shape[0] : 1; w.strides[d] = d == pos ? flat.strides[0] : 0; }
        vs[i] = w;
    }
    if (rc == 0 && !sparse) {
        int32_t bnd = 0;
        int64_t bsh[TSR_MAXDIM];
        for (int i = 0; i < n && rc == 0; i++) rc = bshape(bnd, bsh, vs[i].ndim, vs[i].shape, &bnd, bsh);
        for (int i = 0; i < n && rc == 0; i++) rc = v_broadcast(&vs[i], bnd, bsh);
    }
    if (rc == 0) rc = emit_list(&res[0], vs, n, 0, -1);
    for (int i = 0; i < n; i++) own_free(&own[i]);
    list_views_free(vs, n, &L);
    return rc;
}

/* ================================================================ comparisons and metadata */

static int eq_elem(const char *a, int da, const char *b, int db, int equal_nan)
{
    if (da == TSR_C128 || db == TSR_C128) {
        double x[2] = {0, 0}, y[2] = {0, 0};
        if (da == TSR_C128) memcpy(x, a, 16); else x[0] = rd_f64(a, da);
        if (db == TSR_C128) memcpy(y, b, 16); else y[0] = rd_f64(b, db);
        if (equal_nan && (x[0] != x[0] || x[1] != x[1]) && (y[0] != y[0] || y[1] != y[1])) return 1;
        return x[0] == y[0] && x[1] == y[1];
    }
    if (is_intlike(da) && is_intlike(db)) {
        /* exact integer comparison (uint8/bool are non-negative) */
        return rd_i64(a, da) == rd_i64(b, db);
    }
    const int p = promote2(da, db);
    double x = rd_f64(a, da), y = rd_f64(b, db);
    if (p == TSR_F32) { x = (float)x; y = (float)y; }
    if (equal_nan && x != x && y != y) return 1;
    if (is_intlike(da) && da == TSR_I64 && p == TSR_F64) x = (double)rd_i64(a, da);
    return x == y;
}

/* array_equal(a1, a2, equal_nan=False); array_equiv(a1, a2) */
static int r_array_equal(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int equiv = ctx ? *(const int *)ctx : 0;
    harr h1, h2;
    int rc = get_arr(argk(args, nargs, 0), &h1, "a1");
    if (rc == 0) rc = get_arr(argk(args, nargs, 1), &h2, "a2");
    if (rc < 0) return rc;
    const tsr_arg *en = argk(args, nargs, 2);
    const int equal_nan = !equiv && (en->kind == 4 || en->kind == 1) && en->num != 0;
    tsr_array a = h1.a, b = h2.a;
    int ok = 1;
    memset(&res[0], 0, sizeof res[0]);
    res[0].kind = 4;
    if (equiv) {
        int32_t nd;
        int64_t sh[TSR_MAXDIM];
        if (tsr_broadcast_shape(a.ndim, a.shape, b.ndim, b.shape, &nd, sh) != 0) { res[0].num = 0; return TSR_OK; }
        v_broadcast(&a, nd, sh);
        v_broadcast(&b, nd, sh);
    } else if (a.ndim != b.ndim || memcmp(a.shape, b.shape, sizeof(int64_t) * (size_t)a.ndim) != 0) {
        res[0].num = 0;
        return TSR_OK;
    }
    const int64_t total = tsr_shape_size(a.ndim, a.shape);
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < total && ok; q++) {
        int64_t oa = 0, ob = 0;
        for (int d = 0; d < a.ndim; d++) { oa += ix[d] * a.strides[d]; ob += ix[d] * b.strides[d]; }
        ok = eq_elem(elem0(&a) + oa, a.dtype, elem0(&b) + ob, b.dtype, equal_nan);
        for (int d = a.ndim - 1; d >= 0; d--) { if (++ix[d] < a.shape[d]) break; ix[d] = 0; }
    }
    res[0].num = ok;
    return TSR_OK;
}
static const int EQUIV = 1;

/* ndim(a), shape(a), size(a, axis=None) */
static int r_meta(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const int kind = *(const int *)ctx;                  /* 0 ndim, 1 shape, 2 size */
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    if (kind == 0) { fn_result_int(&res[0], h.a.ndim); return TSR_OK; }
    if (kind == 1) {
        tsr_result *items = fn_result_seq(&res[0], h.a.ndim);
        if (!items) return TSR_ENOMEM;
        for (int d = 0; d < h.a.ndim; d++) fn_result_int(&items[d], h.a.shape[d]);
        return TSR_OK;
    }
    const tsr_arg *aa = argk(args, nargs, 1);
    if (is_none(aa)) { fn_result_int(&res[0], tsr_shape_size(h.a.ndim, h.a.shape)); return TSR_OK; }
    int axes[TSR_MAXDIM + 1], n;
    if ((rc = axis_tuple(aa, h.a.ndim, axes, &n, 0, NULL)) < 0) return rc;
    int64_t s = 1;
    for (int i = 0; i < n; i++) s *= h.a.shape[axes[i]];
    fn_result_int(&res[0], s);
    return TSR_OK;
}
static const int META_NDIM = 0, META_SHAPE = 1, META_SIZE = 2;

/* ================================================================ select, choose */

/* select(condlist, choicelist, default=0): Python scalars stay weak (NEP 50) in the result dtype */
static int r_select(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    const tsr_arg *ca = argk(args, nargs, 0), *ch = argk(args, nargs, 1), *df = argk(args, nargs, 2);
    arrlist C, V;
    int rc = al_get(ca, &C, "condlist");
    if (rc < 0) return rc;
    if ((rc = al_get(ch, &V, "choicelist")) < 0) { al_free(&C); return rc; }
    if (C.n != V.n) { rc = TSR_EARG; fn_set_error("list of cases must be same length as list of conditions"); goto out; }
    if (C.n == 0) { rc = TSR_EARG; fn_set_error("select with an empty condition list is not possible"); goto out; }
    harr hd;
    tsr_arg zero = {0};
    zero.kind = 1; zero.flags = 1; zero.ival = 0; zero.num = 0;
    const tsr_arg *dfa = is_none(df) && nargs < 3 ? &zero : df;
    if (is_none(dfa)) dfa = &zero;
    if ((rc = get_arr(dfa, &hd, "default")) < 0) goto out;
    /* result dtype: strong arrays first, then the weak Python scalars */
    {
        int dt = -1;
        char weak[3] = {0};
        int nweak = 0;
        for (int i = 0; i <= V.n; i++) {
            const tsr_arg *it = i < V.n ? (ch->kind == 5 ? &ch->items[i] : NULL) : dfa;
            const char wk = it ? weak_kind(it) : 0;
            const tsr_array *a = i < V.n ? &V.h[i].a : &hd.a;
            if (wk) { if (nweak < 3) weak[nweak++] = wk; continue; }
            dt = dt < 0 ? a->dtype : promote2(dt, a->dtype);
        }
        if (dt < 0) {                                    /* all Python scalars: the default dtypes */
            dt = TSR_BOOL;
            for (int i = 0; i < nweak; i++) dt = promote2(dt, weak[i] == 'f' ? TSR_F64 : weak[i] == 'i' ? TSR_I64 : TSR_BOOL);
        } else for (int i = 0; i < nweak; i++) dt = promote_weak(dt, weak[i]);
        /* broadcast conditions and choices together */
        int32_t nd = 0;
        int64_t sh[TSR_MAXDIM];
        for (int i = 0; i < C.n && rc == 0; i++) {
            if (C.h[i].a.dtype != TSR_BOOL) { fn_set_error("invalid entry %d in condlist: should be boolean ndarray", i); rc = TSR_EARG; break; }
            rc = bshape(nd, sh, C.h[i].a.ndim, C.h[i].a.shape, &nd, sh);
        }
        for (int i = 0; i < V.n && rc == 0; i++) rc = bshape(nd, sh, V.h[i].a.ndim, V.h[i].a.shape, &nd, sh);
        if (rc == 0) rc = bshape(nd, sh, hd.a.ndim, hd.a.shape, &nd, sh);
        if (rc < 0) goto out;
        tsr_array dv = hd.a;
        if ((rc = v_broadcast(&dv, nd, sh)) < 0) goto out;
        if ((rc = emit(&res[0], &dv, dt)) < 0) goto out;
        const int64_t total = tsr_shape_size(nd, sh), isz = tsr_itemsize(dt);
        char *p = res[0].kind == 3 ? (char *)res[0].arr.data : NULL;
        unsigned char scal[16];
        if (!p) {                                        /* a 0-d result: work in a local element */
            tsr_array s = dv;
            scalar_into(&s, dt, scal);
            p = (char *)scal;
        }
        for (int i = C.n - 1; i >= 0 && rc == 0; i--) {  /* reverse: the first condition wins */
            tsr_array c = C.h[i].a, v = V.h[i].a;
            if ((rc = v_broadcast(&c, nd, sh)) < 0 || (rc = v_broadcast(&v, nd, sh)) < 0) break;
            int64_t ix[TSR_MAXDIM] = {0};
            for (int64_t q = 0; q < total; q++) {
                int64_t oc = 0, ov = 0;
                for (int d = 0; d < nd; d++) { oc += ix[d] * c.strides[d]; ov += ix[d] * v.strides[d]; }
                if (*(elem0(&c) + oc)) {
                    tsr_array s = v;
                    s.ndim = 0;
                    s.offset += ov;
                    scalar_into(&s, dt, p + q * isz);
                }
                for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
            }
        }
        if (rc == 0 && res[0].kind != 3) { fn_results_free(&res[0], 1); rc = result_scalar(&res[0], (const char *)scal, dt); }
    }
out:
    al_free(&C);
    al_free(&V);
    return rc;
}

/* choose(a, choices, mode='raise') */
static int r_choose(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    harr ha;
    int rc = get_arr(argk(args, nargs, 0), &ha, "a");
    if (rc < 0) return rc;
    const tsr_arg *ma = argk(args, nargs, 2);
    int mode = 0;
    if (ma->kind == 2 && ma->str) {
        if (strcmp(ma->str, "raise") == 0) mode = 0;
        else if (strcmp(ma->str, "wrap") == 0) mode = 1;
        else if (strcmp(ma->str, "clip") == 0) mode = 2;
        else { fn_set_error("clipmode must be one of 'clip', 'raise', or 'wrap' (got '%s')", ma->str); return TSR_EARG; }
    }
    if (!is_intlike(ha.a.dtype)) {
        fn_set_error("Cannot cast array data from dtype('%s') to dtype('int64') according to the rule 'safe'", dtname(ha.a.dtype));
        return TSR_EARG;
    }
    arrlist L;
    if ((rc = al_get(argk(args, nargs, 1), &L, "choices")) < 0) return rc;
    const int n = L.n;
    if (n == 0) { al_free(&L); fn_set_error("0-length sequence."); return TSR_EARG; }
    int dt = -1;
    for (int i = 0; i < n; i++) dt = dt < 0 ? L.h[i].a.dtype : promote2(dt, L.h[i].a.dtype);
    int32_t nd = ha.a.ndim;
    int64_t sh[TSR_MAXDIM];
    memcpy(sh, ha.a.shape, sizeof(int64_t) * (size_t)nd);
    for (int i = 0; i < n && rc == 0; i++) rc = bshape(nd, sh, L.h[i].a.ndim, L.h[i].a.shape, &nd, sh);
    tsr_array av = ha.a;
    if (rc == 0) rc = v_broadcast(&av, nd, sh);
    for (int i = 0; i < n && rc == 0; i++) rc = v_broadcast(&L.h[i].a, nd, sh);
    if (rc < 0) { al_free(&L); return rc; }
    const int64_t total = tsr_shape_size(nd, sh), isz = tsr_itemsize(dt);
    char *p = (char *)fn_result_array(&res[0], dt, nd, sh);
    if (!p) { al_free(&L); return TSR_ENOMEM; }
    int64_t ix[TSR_MAXDIM] = {0};
    for (int64_t q = 0; q < total; q++) {
        int64_t oa = 0;
        for (int d = 0; d < nd; d++) oa += ix[d] * av.strides[d];
        int64_t j = rd_i64(elem0(&av) + oa, av.dtype);
        if (mode == 0) {
            if (j < 0 || j >= n) { fn_results_free(&res[0], 1); al_free(&L); fn_set_error("invalid entry in choice array"); return TSR_EARG; }
        } else if (mode == 1) j = pymod(j, n);
        else j = j < 0 ? 0 : (j >= n ? n - 1 : j);
        const tsr_array *c = &L.h[j].a;
        int64_t oc = 0;
        for (int d = 0; d < nd; d++) oc += ix[d] * c->strides[d];
        tsr_array s = *c;
        s.ndim = 0;
        s.offset += oc;
        scalar_into(&s, dt, p + q * isz);
        for (int d = nd - 1; d >= 0; d--) { if (++ix[d] < sh[d]) break; ix[d] = 0; }
    }
    al_free(&L);
    if (nd == 0) {
        tsr_result s;
        if (result_scalar(&s, (const char *)res[0].arr.data, dt) == TSR_OK) { fn_results_free(&res[0], 1); res[0] = s; }
    }
    return TSR_OK;
}

/* ascontiguousarray(a, dtype=None) / asfortranarray(a, dtype=None): a contiguous copy (numpy.asarray with
   ndim >= 1), cast to dtype if given. Tessero arrays are always C-contiguous and it has no Fortran memory
   order, so asfortranarray returns the same C-contiguous copy; the logical values, shape and dtype match
   NumPy exactly (only the .flags memory order, which Tessero does not expose, would differ). */
static int r_ascontiguous(const void *ctx, const tsr_arg *args, int nargs, tsr_result *res, int nres)
{
    (void)ctx; (void)nres;
    harr h;
    int rc = get_arr(argk(args, nargs, 0), &h, "a");
    if (rc < 0) return rc;
    int dt;
    if ((rc = arg_dtype(argk(args, nargs, 1), &dt)) < 0) return rc;
    if (dt < 0) dt = h.a.dtype;
    tsr_array v = h.a;
    if (v.ndim == 0) { v.ndim = 1; v.shape[0] = 1; v.strides[0] = 0; }   /* ndim >= 1 */
    return emit(&res[0], &v, dt);
}

/* ================================================================ table */

static const fn_def DEFS[] = {
    ROUTINE("np.ascontiguousarray", 1, "a, dtype=None", "out", r_ascontiguous, NULL, "A C-contiguous array (ndim >= 1) of the given dtype (numpy.ascontiguousarray)."),
    ROUTINE("np.asfortranarray", 1, "a, dtype=None", "out", r_ascontiguous, NULL, "A contiguous array (ndim >= 1); Tessero has no Fortran memory order, so the values, shape and dtype match numpy.asfortranarray but the memory layout is C order."),
    ROUTINE("np.flip", 1, "m, axis=None", "out", r_flip, &FLIP, "Reverse the order of elements along the given axes (numpy.flip)."),
    ROUTINE("np.fliplr", 1, "m", "out", r_flip, &FLIPLR, "Reverse the order along axis 1 (numpy.fliplr)."),
    ROUTINE("np.flipud", 1, "m", "out", r_flip, &FLIPUD, "Reverse the order along axis 0 (numpy.flipud)."),
    ROUTINE("np.rot90", 1, "m, k=1, axes=None", "out", r_rot90, NULL, "Rotate by 90 degrees in the plane of two axes, default (0, 1) (numpy.rot90)."),
    ROUTINE("np.roll", 1, "a, shift, axis=None", "out", r_roll, NULL, "Roll elements along the given axes; without axis on the flattened array (numpy.roll)."),
    ROUTINE("np.tile", 1, "A, reps", "out", r_tile, NULL, "Repeat an array the given number of times per axis (numpy.tile)."),
    ROUTINE("np.repeat", 1, "a, repeats, axis=None", "out", r_repeat, NULL, "Repeat each element (numpy.repeat)."),
    ROUTINE("np.resize", 1, "a, new_shape", "out", r_resize, NULL, "A new array of the given shape, repeating the data cyclically (numpy.resize)."),
    ROUTINE("np.expand_dims", 1, "a, axis", "out", r_expand_dims, NULL, "Insert length-1 axes (numpy.expand_dims)."),
    ROUTINE("np.squeeze", 1, "a, axis=None", "out", r_squeeze, NULL, "Remove length-1 axes (numpy.squeeze)."),
    ROUTINE("np.swapaxes", 1, "a, axis1, axis2", "out", r_swapaxes, NULL, "Interchange two axes (numpy.swapaxes)."),
    ROUTINE("np.moveaxis", 1, "a, source, destination", "out", r_moveaxis, NULL, "Move axes to new positions (numpy.moveaxis)."),
    ROUTINE("np.rollaxis", 1, "a, axis, start=0", "out", r_rollaxis, NULL, "Roll an axis backwards to a position (numpy.rollaxis)."),
    ROUTINE("np.matrix_transpose", 1, "x", "out", r_matrix_transpose, NULL, "Transpose the last two axes (numpy.matrix_transpose)."),
    ROUTINE("np.atleast_1d", 1, "*arys", "out", r_atleast, &ONE, "Arrays with at least one dimension; several arguments give a list (numpy.atleast_1d)."),
    ROUTINE("np.atleast_2d", 1, "*arys", "out", r_atleast, &TWO, "Arrays with at least two dimensions; several arguments give a list (numpy.atleast_2d)."),
    ROUTINE("np.atleast_3d", 1, "*arys", "out", r_atleast, &THREE, "Arrays with at least three dimensions; several arguments give a list (numpy.atleast_3d)."),
    ROUTINE("np.broadcast_to", 1, "array, shape", "out", r_broadcast_to, NULL, "Broadcast an array to a shape (numpy.broadcast_to)."),
    ROUTINE("np.broadcast_arrays", 1, "*args", "*arrays", r_broadcast_arrays, NULL, "Broadcast arrays against each other (numpy.broadcast_arrays)."),
    ROUTINE("np.broadcast_shapes", 1, "*args", "*shape", r_broadcast_shapes, NULL, "The broadcast shape of the given shapes (numpy.broadcast_shapes)."),
    ROUTINE("np.concatenate", 1, "arrays[], axis=0, dtype=None, casting='same_kind'", "out", r_concatenate, NULL, "Join arrays along an existing axis; axis=None flattens them first (numpy.concatenate)."),
    ROUTINE("np.concat", 1, "arrays[], axis=0, dtype=None, casting='same_kind'", "out", r_concatenate, NULL, "Join arrays along an existing axis (numpy.concat, the Array API name of concatenate)."),
    ROUTINE("np.stack", 1, "arrays[], axis=0, dtype=None, casting='same_kind'", "out", r_stack, NULL, "Join arrays along a new axis (numpy.stack)."),
    ROUTINE("np.vstack", 1, "tup[], dtype=None, casting='same_kind'", "out", r_xstack, &VSTACK, "Stack arrays vertically, row-wise (numpy.vstack)."),
    ROUTINE("np.hstack", 1, "tup[], dtype=None, casting='same_kind'", "out", r_xstack, &HSTACK, "Stack arrays horizontally, column-wise (numpy.hstack)."),
    ROUTINE("np.dstack", 1, "tup[]", "out", r_xstack, &DSTACK, "Stack arrays depth-wise, along the third axis (numpy.dstack)."),
    ROUTINE("np.column_stack", 1, "tup[]", "out", r_xstack, &COLUMN_STACK, "Stack 1-D arrays as columns of a 2-D array (numpy.column_stack)."),
    ROUTINE("np.unstack", 1, "x, axis=0", "*arrays", r_unstack, NULL, "Split an array into the arrays along an axis (numpy.unstack)."),
    ROUTINE("np.split", 1, "ary, indices_or_sections, axis=0", "*arrays", r_split, &SPLIT, "Split into equal sections or at the given indices (numpy.split)."),
    ROUTINE("np.array_split", 1, "ary, indices_or_sections, axis=0", "*arrays", r_split, &ARRAY_SPLIT, "Split into sections of nearly equal size (numpy.array_split)."),
    ROUTINE("np.hsplit", 1, "ary, indices_or_sections", "*arrays", r_split, &HSPLIT, "Split horizontally, column-wise (numpy.hsplit)."),
    ROUTINE("np.vsplit", 1, "ary, indices_or_sections", "*arrays", r_split, &VSPLIT, "Split vertically, row-wise (numpy.vsplit)."),
    ROUTINE("np.dsplit", 1, "ary, indices_or_sections", "*arrays", r_split, &DSPLIT, "Split along the third axis (numpy.dsplit)."),
    ROUTINE("np.append", 1, "arr, values, axis=None", "out", r_append, NULL, "Append values to the end of an array (numpy.append)."),
    ROUTINE("np.insert", 1, "arr, obj, values, axis=None", "out", r_insert, NULL, "Insert values before the given indices (numpy.insert)."),
    ROUTINE("np.delete", 1, "arr, obj, axis=None", "out", r_delete, NULL, "Remove the given indices along an axis (numpy.delete)."),
    ROUTINE("np.trim_zeros", 1, "filt, trim='fb', axis=None", "out", r_trim_zeros, NULL, "Trim leading and/or trailing zeros (numpy.trim_zeros)."),
    ROUTINE("np.diag", 1, "v, k=0", "out", r_diag, NULL, "Extract a diagonal, or build a 2-D array with the given diagonal (numpy.diag)."),
    ROUTINE("np.diagflat", 1, "v, k=0", "out", r_diag, &DIAGFLAT, "A 2-D array with the flattened input as a diagonal (numpy.diagflat)."),
    ROUTINE("np.diagonal", 1, "a, offset=0, axis1=0, axis2=1", "out", r_diagonal, NULL, "The given diagonal of an array (numpy.diagonal)."),
    ROUTINE("np.trace", 1, "a, offset=0, axis1=0, axis2=1, dtype=None", "out", r_trace, NULL, "Sum along a diagonal (numpy.trace)."),
    ROUTINE("np.tri", 1, "N, M=None, k=0, dtype='float64'", "out", r_tri, NULL, "Ones at and below the k-th diagonal, zeros elsewhere (numpy.tri)."),
    ROUTINE("np.tril", 1, "m, k=0", "out", r_trilu, &TRIL, "Lower triangle of an array (numpy.tril)."),
    ROUTINE("np.triu", 1, "m, k=0", "out", r_trilu, &TRIU, "Upper triangle of an array (numpy.triu)."),
    ROUTINE("np.tril_indices", 1, "n, k=0, m=None", "*indices", r_trilu_indices, &TRIL_I, "Indices of the lower triangle of an (n, m) array (numpy.tril_indices)."),
    ROUTINE("np.triu_indices", 1, "n, k=0, m=None", "*indices", r_trilu_indices, &TRIU_I, "Indices of the upper triangle of an (n, m) array (numpy.triu_indices)."),
    ROUTINE("np.tril_indices_from", 1, "arr, k=0", "*indices", r_trilu_indices, &TRIL_IF, "Indices of the lower triangle of a 2-D array (numpy.tril_indices_from)."),
    ROUTINE("np.triu_indices_from", 1, "arr, k=0", "*indices", r_trilu_indices, &TRIU_IF, "Indices of the upper triangle of a 2-D array (numpy.triu_indices_from)."),
    ROUTINE("np.diag_indices", 1, "n, ndim=2", "*indices", r_diag_indices, NULL, "Indices of the main diagonal of an n x ... x n array (numpy.diag_indices)."),
    ROUTINE("np.diag_indices_from", 1, "arr", "*indices", r_diag_indices, &FROM, "Indices of the main diagonal of an array (numpy.diag_indices_from)."),
    ROUTINE("np.identity", 1, "n, dtype=None", "out", r_identity, NULL, "The identity matrix (numpy.identity)."),
    ROUTINE("np.zeros_like", 1, "a, dtype=None, shape=None", "out", r_like, &LIKE_ZEROS, "Zeros with the shape and dtype of an array (numpy.zeros_like)."),
    ROUTINE("np.ones_like", 1, "a, dtype=None, shape=None", "out", r_like, &LIKE_ONES, "Ones with the shape and dtype of an array (numpy.ones_like)."),
    ROUTINE("np.empty_like", 1, "a, dtype=None, shape=None", "out", r_like, &LIKE_EMPTY, "An array with the shape and dtype of another; Tessero fills it with zeros (numpy.empty_like)."),
    ROUTINE("np.full_like", 1, "a, fill_value, dtype=None, shape=None", "out", r_like, &LIKE_FULL, "An array of fill_value with the shape and dtype of another (numpy.full_like)."),
    ROUTINE("np.take", 1, "a, indices, axis=None, mode='raise'", "out", r_take, NULL, "Elements at the given indices along an axis (numpy.take)."),
    ROUTINE("np.take_along_axis", 1, "arr, indices, axis=-1", "out", r_take_along_axis, NULL, "Pick values by matching 1-D index slices (numpy.take_along_axis)."),
    ROUTINE("np.nonzero", 1, "a", "*indices", r_nonzero, &NZ, "Indices of the non-zero elements, one array per axis (numpy.nonzero)."),
    ROUTINE("np.argwhere", 1, "a", "out", r_nonzero, &ARGWHERE, "Indices of the non-zero elements, one row per element (numpy.argwhere)."),
    ROUTINE("np.flatnonzero", 1, "a", "out", r_nonzero, &FLATNZ, "Indices of the non-zero elements of the flattened array (numpy.flatnonzero)."),
    ROUTINE("np.compress", 1, "condition, a, axis=None", "out", r_compress, NULL, "Selected slices along an axis (numpy.compress)."),
    ROUTINE("np.extract", 1, "condition, arr", "out", r_compress, &EXTRACT, "Elements where the condition holds, flattened (numpy.extract)."),
    ROUTINE("np.unravel_index", 1, "indices, shape, order='C'", "*coords", r_unravel_index, NULL, "Convert flat indices into a tuple of coordinate arrays (numpy.unravel_index)."),
    ROUTINE("np.ravel_multi_index", 1, "multi_index[], dims, mode[]='raise', order='C'", "out", r_ravel_multi_index, NULL, "Convert coordinate arrays into flat indices (numpy.ravel_multi_index)."),
    ROUTINE("np.indices", 1, "dimensions, dtype='int64', sparse=False", "out", r_indices, NULL, "An array of grid indices (numpy.indices); sparse=True gives a list of arrays."),
    ROUTINE("np.ix_", 1, "*args", "*arrays", r_ix, NULL, "An open mesh from several sequences (numpy.ix_)."),
    ROUTINE("np.meshgrid", 1, "*xi, copy=True, sparse=False, indexing='xy'", "*arrays", r_meshgrid, NULL, "Coordinate matrices from coordinate vectors (numpy.meshgrid)."),
    ROUTINE("np.array_equal", 1, "a1, a2, equal_nan=False", "out", r_array_equal, NULL, "True when two arrays have the same shape and elements (numpy.array_equal)."),
    ROUTINE("np.array_equiv", 1, "a1, a2", "out", r_array_equal, &EQUIV, "True when two arrays are shape-consistent and equal (numpy.array_equiv)."),
    ROUTINE("np.ndim", 1, "a", "out", r_meta, &META_NDIM, "Number of dimensions (numpy.ndim)."),
    ROUTINE("np.shape", 1, "a", "*shape", r_meta, &META_SHAPE, "The shape of an array (numpy.shape)."),
    ROUTINE("np.size", 1, "a, axis=None", "out", r_meta, &META_SIZE, "Number of elements, in total or along axes (numpy.size)."),
    ROUTINE("np.select", 1, "condlist[], choicelist[], default=0", "out", r_select, NULL, "Elements from the choices where the conditions hold (numpy.select)."),
    ROUTINE("np.choose", 1, "a, choices[], mode='raise'", "out", r_choose, NULL, "Build an array from an index array and a list of choices (numpy.choose)."),
};

const fn_table TSR_NP_SHAPE_TABLE = {DEFS, (int)(sizeof DEFS / sizeof DEFS[0])};
